import time
# import stopit
class BaseModel:
    
    is_deepseek: bool
    
    # @stopit.threading_timeoutable(timeout=40)
    def compile_grammar(self, json_schema):
        status = "unknown"
        try:
            compiled_grammar = self._compile_grammar(json_schema)
            status = "success"
        except Exception as e:
        # Any exception in this block will be caught and considered as schema not supported
            compiled_grammar = None
            status = "schema_not_supported"
        return compiled_grammar, status
    
    def generate(self, prompt, json_schema=None, temperature=0.2):
        compile_start_time = time.time()
        compiled_grammar = self.compile_grammar(json_schema)
        compile_end_time = time.time()
        # GCT (Grammar Compilation Time)
        gct = compile_end_time - compile_start_time
        
        # print("Generating output")
        gen_start_time = time.time()
        output, first_tok_arr_time, gen_length = self._call_engine(prompt,compiled_grammar, temperature)
        # TTFT (Time to First Token)
        ttft = first_tok_arr_time - compile_start_time
        # print("Output generated")
        gen_end_time = time.time()
        # TGT (Total Generation Time)
        tgt = gen_end_time - gen_start_time
        avg_token_gen_time = tgt/gen_length
        
        return output, gct, ttft, tgt, avg_token_gen_time
    
    def generate_all(self, prompts, json_schema=None, temperature=0.2):
        compiled_grammar = self.compile_grammar(json_schema)
        raw_input, output, first_tok_arr_time, gen_length = self._call_engine(prompts, compiled_grammar, temperature)
        return raw_input, output, gen_length
    
    def _call_engine(self, prompt, compiled_grammar, temperature):
        raise NotImplementedError
    
    def close_model(self):
        raise NotImplementedError