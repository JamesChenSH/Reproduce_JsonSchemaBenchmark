import xgrammar
from BaseModel import BaseModel

class TimingLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()
        self.timestamps = []
    def __call__(self, input_ids, scores):
        current_time = time.time()
        self.timestamps.append(current_time)
        return scores

class XGrammarModel(BaseModel):
    def compile_grammar(self, json_schema):
        return xgrammar.GrammarCompiler().compile_json_schema(json_schema)→
    
    def _call_engine(self, prompt, compiled_grammar):
        output = self.hf_model.generate(prompt,
        logits_processor=[compiled_grammar, timeit_logit_processor])→
        first_tok_arr_time = timeit_logit_processor.timestamps[0]
        return output, first_tok_arr_time