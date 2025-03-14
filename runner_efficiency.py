import os, sys, time, json
from argparse import ArgumentParser

os.environ['HF_HOME'] = './cache/'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
from models.BaseModel import BaseModel
from utils.validation import validate_enhanaced



def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--wrapper", type=str, default='guidance')
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--is_cpp", action='store_true', default=False)
    return parser


def get_model(args) -> BaseModel:
    model: BaseModel | None = None

    # if args.xgr or args.xgr_compliant or args.xgr_cpp:
    #     from .xgr_model import Xgrmodel

    #     assert not model, "Multiple models specified"
    #     model = Xgrmodel()
    #     model.compliant = args.xgr_compliant
    #     model.llama_cpp = args.xgr_cpp
    wrapper_name = args.wrapper
    llm_name = args.llm
    
    if wrapper_name == 'guidance':
        from models.GuidanceModel import GuidanceModel
        # llama cpp method
        assert not model, "Multiple models specified"
        model = GuidanceModel(llm_name, args.is_cpp)

    if wrapper_name == 'outlines':
        from models.OutlinesModel import OutlinesModel
        assert not model, "Multiple models specified"
        model = OutlinesModel()

    if wrapper_name == 'llamacpp':
        from models.LlamaCppModel import LlamaCppModel
        assert not model, "Multiple models specified"
        model = LlamaCppModel()

    if not model:
        raise Exception("No grammar model specified")

    return model


if __name__ == "__main__":
    
    # model = GuidanceModel.GuidanceModel()
    # model = OutlinesModel.OutlinesModel()
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    model = get_model(args)
    
    data_path = "../jsonschemabench/data/Github_easy"
    prompt = 'Generate a json object'
    
    files = os.listdir(data_path)
    files.sort()
    n = len(files)
    all_gct = []
    all_ttft = []
    all_tgt = []
    all_avg_tkn_t = []
    
    # Load Json schemas
    json_schemas = []
    problematic_files = [
        'o6176.json', 'o9765.json',
        'o9768.json', 'o9772.json',
        'o9773.json', 'o9774.json', 
        'o9783.json', 'o9812.json',
        'o9861.json', 'o9870.json',
        'o9871.json', 'o9892.json', 
        'o9896.json', 'o9893.json',
        'o9931.json', 'o9945.json',
        'o9955.json', 'o9966.json',
        'o9967.json'
    ]  # These files cause llama-cpp-python to crash with seg fault, to be investigated
    
    error_schemas = []
    invalid_count = 0
    
    for i, file in enumerate(files[:min(args.length, n)]):
        
        print(i, file, flush=True)
        
        if file in problematic_files:
            invalid_count += 1
            continue
        with open(os.path.join(data_path, file), 'r') as f:
            schema = f.read()
        json_schemas.append(schema)
    
        # print(json.loads(schema))
        try:
            output, gct, ttft, tgt, avg_tkn_t = model.generate_steam(prompt, schema)
            print(output, flush=True)
            
            validate_enhanaced(json.loads(output), json.loads(schema))
            
            all_gct.append(gct)
            all_ttft.append(ttft)
            all_tgt.append(tgt)
            all_avg_tkn_t.append(avg_tkn_t)

        except Exception as e:
            print(f"Error: {e}", flush=True)
            # raise e
            error_schemas.append(schema)
            invalid_count += 1
            continue
    
    print(f"Median GCT: {np.median(all_gct)}", flush=True)
    print(f"Median TTFT: {np.median(all_ttft)}", flush=True)
    print(f"Median TGT: {np.median(all_tgt)}", flush=True)
    print(f"Median Avg Token Time: {np.median(all_avg_tkn_t)}", flush=True)
    
    print(f"Mean GCT: {np.mean(all_gct)}", flush=True)
    print(f"Mean TTFT: {np.mean(all_ttft)}", flush=True)
    print(f"Mean TGT: {np.mean(all_tgt)}", flush=True)
    print(f"Mean Avg Token Time: {np.mean(all_avg_tkn_t)}", flush=True)
    
    print(f"Invalid count: {invalid_count}", flush=True)
    # print(f"Error schemas: {error_schemas}", flush=True)
    model.close_model()
    