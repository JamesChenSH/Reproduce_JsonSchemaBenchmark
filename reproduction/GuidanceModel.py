import guidance, time
from BaseModel import BaseModel
from guidance import models

class GuidanceModel(BaseModel):
    
    def __init__(self, model, tokenizer):
        super().__init__()
        self.guidance_model = models.Transformers(model, tokenizer)
    
    def compile_grammar(self, json_schema):
        return guidance.json(schema=json_schema,)
    
    def _call_engine(self, prompt, compiled_grammar):
        generator = self.guidance_model.stream() + prompt + compiled_grammar
        for i, state in enumerate(generator):
            if i == 0:
                first_state_arr_time = time.time()
        output = state
        return output, first_state_arr_time