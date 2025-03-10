import llama_cpp, time

import llama_cpp.llama_grammar
from models.BaseModel import BaseModel

class LlamaCppModel(BaseModel):
    
    def __init__(self):
        super().__init__()
        self.llama_cpp_model = llama_cpp.Llama(
            model_path='cache/hub/models--QuantFactory--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/b6d5cca03f341fd97b7657420bd60e070835b7e5/Meta-Llama-3.1-8B-Instruct.Q2_K.gguf',
            n_gpu_layers=-1
        )
    
    def compile_grammar(self, json_schema):
        return llama_cpp.llama_grammar.LlamaGrammar.from_json_schema(json_schema)
    
    def _call_engine(self, prompts, compiled_grammar):
        if isinstance(prompts, str):
            prompts = [
                {'role': 'user', 'content': prompts}
            ]
        generator = self.llama_cpp_model.create_chat_completion(prompts, grammar=compiled_grammar, temperature=0.7, stream=True)
        output = ""
        for i, content in enumerate(generator):
            if i == 0:
                first_tok_arr_time = time.time()
            try:
                token = content['choices'][0]['delta']['content']
            except KeyError as e:
                token = ''
            output += token
        return output, first_tok_arr_time, i