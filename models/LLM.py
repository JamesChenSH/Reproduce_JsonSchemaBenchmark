from models.BaseModel import BaseModel
import llama_cpp, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class VanillaModel(BaseModel):
    def __init__(self, llm_name, is_cpp):
        super().__init__()
        self.is_cpp = False
        self.tokenizer = None
        if is_cpp:
            self.is_cpp = True
            self.llm_name, self.file_name = llm_name.split('//')
            self.llm = llama_cpp.Llama.from_pretrained(
                repo_id=self.llm_name,
                filename=self.file_name,
                n_gpu_layers=-1,
                logits_all=True,
                n_ctx=2048,
                verbose=False,
                seed=19181111
            )
        else:
            # transformer method
            # Pipeline for text generation
            print(f"Loading model {llm_name}")
            model = AutoModelForCausalLM.from_pretrained(llm_name,torch_dtype=torch.bfloat16, device_map='auto')
            print(f"Loading tokenizer {llm_name}")
            tokenizer = AutoTokenizer.from_pretrained(llm_name)    
            print(f"Model {llm_name} loaded")
            self.llm = pipeline(
                'text-generation', 
                model=model, 
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id
            )

    def compile_grammar(self, json_schema=None):
        return None
    
    def _call_engine(self, prompts, compiled_grammar, temperature):
        if self.is_cpp:
            # LlamaCpp Model
            # raw_input = self.llm.apply_chat_template(prompts)
            raw_input = str(prompts)

            generator = self.llm.create_chat_completion(
                prompts,  
                temperature=temperature, 
                stream=True,
                max_tokens=2048
            )
            output = ""
            for i, content in enumerate(generator):
                if i == 0:
                    first_tok_arr_time = time.time()
                try:
                    token = content['choices'][0]['delta']['content']
                except KeyError as e:
                    token = ''
                output += token
            return raw_input, output, first_tok_arr_time, i
        else:
            output = self.llm(prompts, max_length=512)[0]['generated_text'][-1]['content']
            return raw_input, output, None, len(output)
        
    def close_model(self):
        if self.is_cpp:
            self.llm._sampler.close()
            self.llm.close()