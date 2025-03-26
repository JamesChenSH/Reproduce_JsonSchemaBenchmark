from models.BaseModel import BaseModel
import llama_cpp, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class VanillaModel(BaseModel):
    def __init__(self, llm_name, is_cpp):
        super().__init__()
        self.is_cpp = False
        self.tokenizer = None
        if is_cpp:
            self.is_cpp = True
            self.llm = llama_cpp.Llama.from_pretrained(
                repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
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
    
    def _call_engine(self, prompts, compiled_grammar, stream=False):
        if self.is_cpp:
            # LlamaCpp Model
            generator = self.llm.create_chat_completion(
                prompts,  
                temperature=0.2, 
                stream=stream,
                max_tokens=512
            )
            output =  generator["choices"][0]["message"]["content"]
            return output, None, len(output)
        else:
            output = self.llm(prompts, max_length=512)[0]['generated_text'][-1]['content']
            return output, None, len(output)
        
    def close_model(self):
        if self.is_cpp:
            self.llm._sampler.close()
            self.llm.close()