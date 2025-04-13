import guidance, time, torch, json

import llama_cpp
from guidance import models, gen
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
                seed=19181111,
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
        
        # begin and end of overall context
        bot = '<｜begin▁of▁sentence｜>' if "DeepSeek" in self.llm_name else '<|begin_of_text|>'
        eot = '<｜end▁of▁sentence｜>' if "DeepSeek" in self.llm_name else '<|end_of_text|>'

        # begin and end of each role
        hdr = '<｜{role}｜>' if "DeepSeek" in self.llm_name else '<|start_header_id|>{role}<|end_header_id|>\n\n'
        eos = '' if "DeepSeek" in self.llm_name else '<|eot_id|>'

        if isinstance(prompt, str):
            generator = generator + prompt
            len_prompt = len(prompt)
        else:
            all_prompts = ''
            for i, p in enumerate(prompt):
                if i == 0:
                    all_prompts = all_prompts + bot
                if 'DeepSeek-R1' in self.llm_name and p['role']=='system':
                    p['role'] = 'user'
                if p['role'] == 'user':
                    all_prompts += (hdr.format(role='user') + p['content'] + eos)
                elif p['role'] == 'assistant':
                    if i == len(prompt) - 1:
                        len_prompt = len(all_prompts + hdr.format(role='assistant'))
                    all_prompts += (hdr.format(role='assistant') + p['content'])
                    if i != len(prompt) - 1:
                        all_prompts += eos
                elif p['role'] == 'system':
                    all_prompts += (hdr.format(role='system') + p['content'] + eos)

            if prompt[-1]['role'] != 'assistant':
                all_prompts = all_prompts + (hdr.format(role='assistant'))
                len_prompt = len(all_prompts)

            raw_input = all_prompts
            
            # DeepSeek-R1 Support -- Only constrain after thinking
            start_of_think = "<think>"
            end_of_think = "</think>"
            if "DeepSeek-R1" in self.llm_name and end_of_think not in all_prompts:
                if start_of_think not in all_prompts:
                    all_prompts = all_prompts + start_of_think
                think_gen = generator + all_prompts + gen(temperature=temperature, stop=end_of_think, max_tokens=2048)
                # Generate until </think> token
                for i, state in enumerate(think_gen):
                    if i == 0:
                        first_state_arr_time = time.time()
                    pass
                del think_gen
                generator = generator + (str(state) + end_of_think)
            else:
                # If not DeepSeek-R1, we just generate the whole prompt
                generator = generator + all_prompts

        # Add grammar
        compiled_grammar.temperature = temperature
        generator = generator + compiled_grammar 
        for j, state in enumerate(generator):
            if j == 0 and not first_state_arr_time:
                first_state_arr_time = time.time()
        output = str(state)[len_prompt:]
        # print(output)
        return raw_input, output, first_state_arr_time, len(output)
    
    def close_model(self):
        if self.is_cpp:
            self.llm.close()
