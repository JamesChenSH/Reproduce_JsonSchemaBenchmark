import os
import torch
# import GuidanceModel

os.environ['HF_HOME'] = './cache/'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM
from guidance import models
from llama_cpp import Llama


model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
llama_model = Llama.from_pretrained( repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF", filename="Meta-Llama-3.1-8B-Instruct.Q2_K.gguf")

# gpts = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.bfloat16)
# gpt2_tkzr = AutoTokenizer.from_pretrained("gpt2", use_fast=False)  # fall back to gpt2 mapping
# llm = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = GuidanceModel.GuidanceModel(llm, tokenizer)