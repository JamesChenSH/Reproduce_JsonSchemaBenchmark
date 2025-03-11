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
            self.llm = llama_cpp.Llama(
                model_path="cache/hub/models--QuantFactory--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/b6d5cca03f341fd97b7657420bd60e070835b7e5/Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
                n_gpu_layers=-1,
                logits_all=True,
                n_ctx=2048,
            )     
            self.guidance_model = models.llama_cpp.LlamaCpp(self.llm)
        else:
            # transformer method
            print(f"Loading model {llm_name}")
            model = AutoModelForCausalLM.from_pretrained(llm_name,torch_dtype=torch.bfloat16, device_map='auto')
            print(f"Loading tokenizer {llm_name}")
            tokenizer = AutoTokenizer.from_pretrained(llm_name)    
            print(f"Model {llm_name} loaded")
            self.guidance_model = models.Transformers(model, tokenizer) 
        
    
    def compile_grammar(self, json_schema):
        return guidance.json(name='json_response', schema=json.loads(json_schema), temperature=0.2, max_tokens=512)
    
    
    def _call_engine(self, prompt, compiled_grammar):
        len_prompt = 0
        if isinstance(prompt, str):
            generator = self.guidance_model.stream() + prompt
            len_prompt = len(prompt)
        else:
            generator = self.guidance_model.stream()
            for p in prompt:
                if p['role'] == 'user':
                    with user():
                        generator = generator + p['content']
                elif p['role'] == 'assistant':
                    with assistant():
                        generator = generator + p['content']
                elif p['role'] == 'system':
                    with system():
                        generator = generator + p['content']
                len_prompt += len(p['content'])

        generator = generator + compiled_grammar
        for i, state in enumerate(generator):
            if i == 0:
                first_state_arr_time = time.time()
        output = str(state)[len_prompt:]
        return output, first_state_arr_time, i
    
    def close_model(self):
        if self.is_cpp:
            self.llm.close()