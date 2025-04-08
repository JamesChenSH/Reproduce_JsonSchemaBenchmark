import guidance, time, torch, json

import llama_cpp
from guidance import models, gen, user, system, assistant
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.BaseModel import BaseModel


class GuidanceModel(BaseModel):
    
    def __init__(self, llm_name, is_cpp=False):
        super().__init__()  
        self.is_cpp = False
        # LlamaCpp method
        if is_cpp:
            self.is_cpp = True
            self.llm_name, self.file_name = llm_name.split('//')
            self.llm = llama_cpp.Llama.from_pretrained(
                repo_id=self.llm_name,
                filename=self.file_name,
                n_gpu_layers=-1,
                logits_all=True,
                n_ctx=2048+512,
                verbose=False,
                seed=19181111
            )    
            self.guidance_model = models.LlamaCpp(self.llm)
        else:
            # transformer method
            print(f"Loading model {llm_name}")
            self.llm_name = llm_name
            model = AutoModelForCausalLM.from_pretrained(llm_name,torch_dtype=torch.bfloat16, device_map='auto')
            print(f"Loading tokenizer {llm_name}")
            tokenizer = AutoTokenizer.from_pretrained(llm_name)    
            print(f"Model {llm_name} loaded")
            self.guidance_model = models.Transformers(model, tokenizer) 
        
    
    def compile_grammar(self, json_schema):
        return guidance.json(name='json_response', schema=json.loads(json_schema), temperature=0.6, max_tokens=512)
    
    
    def _call_engine(self, prompt, compiled_grammar, temperature):
        len_prompt = 0
        generator = self.guidance_model.stream()
        first_state_arr_time = None

        if isinstance(prompt, str):
            with user():
                generator = generator + prompt
            len_prompt = len(prompt)
        else:
            len_prompt = 0
            for i, p in enumerate(prompt):
                if 'DeepSeek-R1' in self.llm_name and p['role']=='system':
                    p['role'] = 'user'

                if p['role'] == 'user':
                    with user():
                        generator = generator + p['content']
                    len_prompt += len(p['content'])
                elif p['role'] == 'assistant':
                    with assistant():
                        generator = generator + p['content']
                    len_prompt += len(p['content'])
                elif p['role'] == 'system':
                    with system():
                        generator = generator + p['content']
                    len_prompt += len(p['content'])
            
            if "DeepSeek-R1" in self.llm_name:
                # If DeepSeek-R1, we need to generate the thinking process first
                # DeepSeek-R1 Support -- Only constrain after thinking
                start_of_think = "<think>"
                end_of_think = "</think>"
                with assistant():
                    think_gen = generator + start_of_think + gen(temperature=temperature, stop=end_of_think)            
                    for i, state in enumerate(think_gen):
                        if i == 0:
                            first_state_arr_time = time.time()
                    thoughts = str(state)[len_prompt:] + end_of_think
                    
                    generator = generator + thoughts + compiled_grammar 
                    for j, state in enumerate(generator):
                        if j == 0 and not first_state_arr_time:
                            first_state_arr_time = time.time()
                output = str(state)[len_prompt:]
                raw_input = str(state)[:len_prompt]
                return raw_input, output, first_state_arr_time, len(output)

        # Add grammar
        compiled_grammar.temperature = temperature
        with assistant():
            generator = generator + compiled_grammar 
            for j, state in enumerate(generator):
                if j == 0 and not first_state_arr_time:
                    first_state_arr_time = time.time()
            output = str(state)[len_prompt:]
        raw_input = str(state)[:len_prompt]
        return raw_input, output, first_state_arr_time, len(output)
    
    def close_model(self):
        if self.is_cpp:
            self.llm.close()
