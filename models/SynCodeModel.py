from syncode import Syncode
import time

class SyncodeModel:
    
    def __init__(self, json_schema):
        self.syncode = Syncode(model="microsoft/phi-2", grammar='json', max_new_tokens=100)
        self.prompt = """
<s>[INST]<<SYS>> 
You are a helpful a ss is ta n t that answers in JSON . Here ’ s the json schema you must adhere to :
<schema>
{ ’ title ’: ’ Person ’ , ’ type ’: ’ object ’ , ’ properties ’: { ’ firstName ’: { ’ type ’: ’ string ’ , ’
description ’: " The person ’ s first name ."} , ’ lastName ’: { ’ type ’: ’ string ’ , ’ description ’:
" The person ’ s last name ."} , ’ age ’: { ’ description ’: ’ Age in years which must be equal to
or greater than zero . ’ , ’ type ’: ’ integer ’ , ’ minimum ’: 0}} , ’ required ’: [ ’ firstName ’ , ’
lastNName', 'age']}
</schema>
<</SYS>>
Please generate a JSON output for a person ’ s profile that includes their first name , last
name , and age . The first name should be ’ Alice ’ , the last name ’ Johnson ’ , and the age
35. [/INST]
        """
        
        self.sys_prompt = f'''You are a helpful assistant that answers in JSON. You need to generate a JSON object that matches the schema below.
        <schema>
        {json_schema}
        </schema>
        '''
            
    def compile_grammar(self, json_schema):
        # lark_grammar = lark
        return None
    
    def generate(self, prompt, json_schema=None):
        # compile_start_time = time.time()
        # compiled_grammar = self.compile_grammar(json_schema)
        # compile_end_time = time.time()
        # GCT (Grammar Compilation Time) -- Not useful in Syncode since it does not parse schemas directly
        # gct = compile_end_time - compile_start_time
        gen_start_time = time.time()
        output, first_tok_arr_time = self._call_engine(prompt,None)
        # TTFT (Time to First Token)
        ttft = first_tok_arr_time - gen_start_time
        gen_end_time = time.time()
        # TGT (Total Generation Time)
        tgt = gen_end_time - gen_start_time
        return output, ttft, tgt
    
    def _call_engine(self, prompt, compiled_grammar):
        self.syncode.infer(self.sys_prompt)


def test_efficiency(model, json_schemas):
    '''
    Pipeline for coverage test using SyncodeModel
    '''
    pass




if __name__ == '__main__':
    # test()
    
    
    # TODO: Create Pipeline to perform Last Letters, Shuffle Objects and GSM8K
    pass