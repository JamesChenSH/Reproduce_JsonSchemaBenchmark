import os, sys, time, json
import numpy as np
from BaseModel import BaseModel
os.environ['HF_HOME'] = './cache/'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(llm_name, wrapper_name) -> BaseModel:
    model: BaseModel | None = None

    # if args.xgr or args.xgr_compliant or args.xgr_cpp:
    #     from .xgr_model import Xgrmodel

    #     assert not model, "Multiple models specified"
    #     model = Xgrmodel()
    #     model.compliant = args.xgr_compliant
    #     model.llama_cpp = args.xgr_cpp

    transformer_model = AutoModelForCausalLM.from_pretrained(llm_name)
    transformer_tokenizer = AutoTokenizer.from_pretrained(llm_name)    
    
    if wrapper_name == 'guidance':
        from .GuidanceModel import GuidanceModel

        assert not model, "Multiple models specified"
        model = GuidanceModel()

    if wrapper_name == 'outlines':
        from .OutlinesModel import OutlinesModel

        assert not model, "Multiple models specified"
        model = OutlinesModel()

    if wrapper_name == 'llamacpp':
        from .LlamaCppModel import LlamaCppModel

        assert not model, "Multiple models specified"
        model = LlamaCppModel()

    if not model:
        raise Exception("No grammar model specified")

    return model


if __name__ == "__main__":
    
    # model = GuidanceModel.GuidanceModel()
    # model = OutlinesModel.OutlinesModel()
    model = get_model('unsloth/Meta-Llama-3.1-8B-Instruct', 'guidance')
    
    data_path = "../data/Github_easy"
    prompt = ''
    
    files = os.listdir(data_path)
    n = len(files)
    all_gct = []
    all_ttft = []
    all_tgt = []
    for file in files:
        with open(os.path.join(data_path, file), 'r') as f:
            schema = json.loads(f.read())
        output, gct, ttft, tgt = model.generate(prompt, schema)
        
        all_gct.append(gct)
        all_ttft.append(ttft)
        all_tgt.append(tgt)
        
        print(output)
        
    print(f"Median GCT: {np.median(all_gct)}")
    print(f"Median TTFT: {np.median(all_ttft)}")
    print(f"Median TGT: {np.median(all_tgt)}")
    
    print(f"Mean GCT: {np.mean(all_gct)}")
    print(f"Mean TTFT: {np.mean(all_ttft)}")
    print(f"Mean TGT: {np.mean(all_tgt)}")
    