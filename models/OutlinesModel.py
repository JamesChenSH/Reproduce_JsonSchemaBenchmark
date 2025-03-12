import outlines, time, llama_cpp, json

import outlines.generate
import outlines.generate.text
import outlines.models
import outlines.models.transformers

from models.BaseModel import BaseModel

class OutlinesModel(BaseModel):
    
    def __init__(self):
        super().__init__()
        
        # Llama cpp support is not available 
        # self.llm = llama_cpp.Llama(
        #     model_path='cache/hub/models--QuantFactory--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/b6d5cca03f341fd97b7657420bd60e070835b7e5/Meta-Llama-3.1-8B-Instruct.Q2_K.gguf',
        #     n_gpu_layers=-1
        # )
        
        # self.model = outlines.models.LlamaCpp(
        #     self.llm
        # )
        
        self.model = outlines.models.transformers(
            'unsloth/Meta-Llama-3.1-8B-Instruct', 
            device='cuda',
            model_kwargs={
                'device_map': 'auto',
                'torch_dtype': 'bfloat16'
            }
        )
        print("Outlines model loaded")
    
    def compile_grammar(self, json_schema):
        return outlines.generate.json(self.model, schema_object=json_schema)
    
    def _call_engine(self, prompt, compiled_grammar, stream=False):
        generator = compiled_grammar.stream(prompts = prompt)
        output = ""
        for i, token in enumerate(generator):
            if i == 0:
                first_tok_arr_time = time.time()
            output += token
            if i > 300:
                break
        return output, first_tok_arr_time, i
    
    def close_model(self):
        pass