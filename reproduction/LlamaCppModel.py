import llama_cpp, time
from BaseModel import BaseModel

class LlamaCppModel(BaseModel):
    def compile_grammar(self, json_schema):
        return llama_cpp.llama_grammar.LlamaGrammar.from_json_schema(json_schema)→
    def _call_engine(self, prompt, compiled_grammar):
        generator = self.llama_cpp_model.create_chat_completion(prompt,
        grammar=compiled_grammar, stream=True)
        output = ""
        for i, token in enumerate(generator):
            if i == 0:
                first_tok_arr_time = time.time()
            output += token
        return output, first_tok_arr_time