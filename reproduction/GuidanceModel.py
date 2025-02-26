import guidance, time

from BaseModel import BaseModel
from guidance import models
from tqdm import tqdm

class GuidanceModel(BaseModel):
    
    def __init__(self):
        super().__init__()
        # self.guidance_model = models.Transformers(model, tokenizer)        
        self.guidance_model = models.LlamaCpp(
            'cache/hub/models--QuantFactory--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/b6d5cca03f341fd97b7657420bd60e070835b7e5/Meta-Llama-3.1-8B-Instruct.Q2_K.gguf', 
            n_gpu_layers=-1)
    
    def compile_grammar(self, json_schema):
        return guidance.json(name='json_response', schema=json_schema, temperature=0.2)
    
    def _call_engine(self, prompt, compiled_grammar):
        generator = self.guidance_model.stream() + prompt + compiled_grammar
        for i, state in enumerate(tqdm(generator)):
            if i == 0:
                first_state_arr_time = time.time()
        output = state
        return output, first_state_arr_time