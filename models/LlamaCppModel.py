import llama_cpp, time

import llama_cpp.llama_grammar
from models.BaseModel import BaseModel

from llama_cpp.llama_grammar import LlamaGrammar
from typing import Dict, Any

class LlamaCppModel(BaseModel):
    
    def __init__(self, llm_name):
        super().__init__()
        self.llm_name, self.file_name = llm_name.split('//')
        self.llm = llama_cpp.Llama.from_pretrained(
            repo_id=self.llm_name,
            filename=self.file_name,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
            seed=19181111
        )
    
    def compile_grammar(self, json_schema):
        return llama_cpp.llama_grammar.LlamaGrammar.from_json_schema(json_schema)
    
    def _call_engine(self, prompts, compiled_grammar):
        segfault_check = self._check_grammar_safety(compiled_grammar)
        if isinstance(prompts, str):
            prompts = [
                {'role': 'user', 'content': prompts}
            ]
        generator = self.llama_cpp_model.create_chat_completion(
            prompts, 
            grammar=compiled_grammar, 
            temperature=0.2, 
            stream=True,
            logprobs=True,
            max_tokens=2048
        )
        output = ""
        for i, content in enumerate(generator):
            if i == self.llama_cpp_model.n_ctx:
                break
            if i == 0:
                first_tok_arr_time = time.time()
            try:
                token = content['choices'][0]['delta']['content']
            except KeyError as e:
                token = ''
            output += token
        return prompts, output, first_tok_arr_time, i
    
    def close_model(self):
        self.llama_cpp_model._sampler.close()
        self.llama_cpp_model.close()
        
    def _check_grammar_safety(self, grammar: "LlamaGrammar") -> Dict[str, Any]:
        import signal, os
        def child_process():
            from llama_cpp._internals import LlamaSampler

            signal.signal(signal.SIGALRM, lambda _, __: os._exit(2))
            signal.alarm(15)
            try:
                LlamaSampler().add_grammar(self.model._model, grammar)
                os._exit(0)
            except Exception:
                os._exit(1)

        id = os.fork()
        if id == 0:
            child_process()
        else:
            _, status = os.waitpid(id, 0)
            if os.WIFEXITED(status):
                exit_code = os.WEXITSTATUS(status)
                return {"success": exit_code == 0, "exit_code": exit_code}
            return {"success": False, "error": "Unknown status"}