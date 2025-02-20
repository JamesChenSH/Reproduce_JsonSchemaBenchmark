import outlines, time

from BaseModel import BaseModel

class OutlinesModel(BaseModel):
    
    def __init__(self, model_name):
        super().__init__()
        self.generator = outlines.generate.SequenceGenerator(model=model_name)
    
    def compile_grammar(self, json_schema):
        return outlines.generate.json(schema_object=json_schema)
    
    def _call_engine(self, prompt, compiled_grammar):
        generator = self.generator.stream(prompt)
        output = ""
        for i, token in enumerate(generator):
            if i == 0:
                first_tok_arr_time = time.time()
            output += token
        return output, first_tok_arr_time