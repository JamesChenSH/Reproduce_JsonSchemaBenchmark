import os, sys, time, json

os.environ['HF_HOME'] = './cache/'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from BaseModel import BaseModel
from validation import validate_enhanaced

from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(llm_name, wrapper_name) -> BaseModel:
    model: BaseModel | None = None

    # if args.xgr or args.xgr_compliant or args.xgr_cpp:
    #     from .xgr_model import Xgrmodel

    #     assert not model, "Multiple models specified"
    #     model = Xgrmodel()
    #     model.compliant = args.xgr_compliant
    #     model.llama_cpp = args.xgr_cpp
    
    if wrapper_name == 'guidance':
        from GuidanceModel import GuidanceModel
        print(f"Loading model {llm_name}")
        transformer_model = AutoModelForCausalLM.from_pretrained(llm_name,torch_dtype=torch.bfloat16, device_map='auto')
        print(f"Loading tokenizer {llm_name}")
        transformer_tokenizer = AutoTokenizer.from_pretrained(llm_name)    
        print(f"Model {llm_name} loaded")
        assert not model, "Multiple models specified"
        model = GuidanceModel(transformer_model, transformer_tokenizer)

    if wrapper_name == 'outlines':
        from OutlinesModel import OutlinesModel

        assert not model, "Multiple models specified"
        model = OutlinesModel()

    if wrapper_name == 'llamacpp':
        from LlamaCppModel import LlamaCppModel

        assert not model, "Multiple models specified"
        model = LlamaCppModel()

    if not model:
        raise Exception("No grammar model specified")

    return model


if __name__ == "__main__":
    
    # model = GuidanceModel.GuidanceModel()
    # model = OutlinesModel.OutlinesModel()
    model = get_model('unsloth/Meta-Llama-3.1-8B-Instruct', 'llamacpp')
    
    data_path = "../data/Github_easy"
    prompt = ''
    
    files = os.listdir(data_path)
    n = len(files)
    all_gct = []
    all_ttft = []
    all_tgt = []
    
    # Load Json schemas
    json_schemas = []
    for file in files:
        with open(os.path.join(data_path, file), 'r') as f:
            schema = f.read()
        json_schemas.append(schema)
    
    error_schemas = []
    invalid_count = 0
    for schema in json_schemas:
        
        # print(schema)
        try:
            output, gct, ttft, tgt = model.generate(prompt, schema)
            
            all_gct.append(gct)
            all_ttft.append(ttft)
            all_tgt.append(tgt)

            validate_enhanaced(json.loads(output), json.reads(schema))
            # print(output)
            
        except Exception as e:
            # print(e)
            error_schemas.append(schema)
            invalid_count += 1
            continue
        
    print(f"Median GCT: {np.median(all_gct)}")
    print(f"Median TTFT: {np.median(all_ttft)}")
    print(f"Median TGT: {np.median(all_tgt)}")
    
    print(f"Mean GCT: {np.mean(all_gct)}")
    print(f"Mean TTFT: {np.mean(all_ttft)}")
    print(f"Mean TGT: {np.mean(all_tgt)}")
    
    print(f"Invalid count: {invalid_count}")
    print(f"Error schemas: {error_schemas}")
    