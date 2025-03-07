import time
# import stopit
class BaseModel:
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
    
    def generate_steam(self, prompt, json_schema=None):
        compile_start_time = time.time()
        compiled_grammar = self.compile_grammar(json_schema)
        compile_end_time = time.time()
        # GCT (Grammar Compilation Time)
        gct = compile_end_time - compile_start_time
        
        # print("Generating output")
        gen_start_time = time.time()
        output, first_tok_arr_time, gen_length = self._call_engine(prompt,compiled_grammar)
        # TTFT (Time to First Token)
        ttft = first_tok_arr_time - gen_start_time
        # print("Output generated")
        gen_end_time = time.time()
        # TGT (Total Generation Time)
        tgt = gen_end_time - gen_start_time
        avg_token_gen_time = tgt/gen_length
        
        return output, gct, ttft, tgt, avg_token_gen_time
    
    def generate_all(self, prompt, json_schema=None):
        compiled_grammar = self.compile_grammar(json_schema)
        output, first_tok_arr_time, gen_length = self._call_engine(prompt,compiled_grammar)
        return output, gen_length
    
    def _call_engine(self, prompt, compiled_grammar):
        raise NotImplementedError