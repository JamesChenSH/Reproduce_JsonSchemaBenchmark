import guidance, time, torch, json
import llama_cpp
from guidance import models, system, assistant, user
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
                n_ctx=2048,
                verbose=False,
                seed=19181111
            )     
            self.guidance_model = models.llama_cpp.LlamaCpp(self.llm)
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
    
    
    def _call_engine(self, prompt, compiled_grammar, stream=False):
        len_prompt = 0
        generator = self.guidance_model.stream()
        if isinstance(prompt, str):
            generator = generator + prompt
            len_prompt = len(prompt)
        else:
            all_prompts = ''
            for i, p in enumerate(prompt):
                if i == 0:
                    all_prompts = all_prompts + '<|begin_of_text|>'
                if 'DeepSeek-R1' in self.llm_name and p['role']=='system':
                    p['role'] = 'user'
                if p['role'] == 'user':
                    all_prompts += '<|start_header_id|>user<|end_header_id|>' + p['content'] + '<|eot_id|>'
                elif p['role'] == 'assistant':
                    all_prompts += '<|start_header_id|>assistant<|end_header_id|>' + p['content'] + '<|eot_id|>'
                elif p['role'] == 'system':
                    all_prompts += '<|start_header_id|>system<|end_header_id|>' + p['content'] + '<|eot_id|>'
            all_prompts = all_prompts + '<|start_header_id|>assistant<|end_header_id|> '
            len_prompt = len(all_prompts)
            generator = generator + all_prompts
        generator = generator + compiled_grammar
        for i, state in enumerate(generator):
            if i == 0:
                first_state_arr_time = time.time()
        output = str(state)[len_prompt:]
        # print(output)
        return output, first_state_arr_time, i
    
    def close_model(self):
        if self.is_cpp:
            self.llm.close()
