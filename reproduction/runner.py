import os, sys, time
from BaseModel import BaseModel
os.environ['HF_HOME'] = './cache/'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(args) -> BaseModel:
    model: BaseModel | None = None

    # if args.xgr or args.xgr_compliant or args.xgr_cpp:
    #     from .xgr_model import Xgrmodel

    #     assert not model, "Multiple models specified"
    #     model = Xgrmodel()
    #     model.compliant = args.xgr_compliant
    #     model.llama_cpp = args.xgr_cpp

    transformer_model = AutoModelForCausalLM.from_pretrained(args.model)
    transformer_tokenizer = AutoTokenizer.from_pretrained(args.model)    
    
    if args.guidance:
        from .GuidanceModel import GuidanceModel

        assert not model, "Multiple models specified"
        model = GuidanceModel()

    if args.outlines:
        from .OutlinesModel import OutlinesModel

        assert not model, "Multiple models specified"
        model = OutlinesModel()

    if args.llamacpp:
        from .LlamaCppModel import LlamaCppModel

        assert not model, "Multiple models specified"
        model = LlamaCppModel()

    if not model:
        raise Exception("No grammar model specified")

    model.tokenizer_model_id = args.tokenizer

    return model


if __name__ == "__main__":
    
    # model = GuidanceModel.GuidanceModel()
    # model = OutlinesModel.OutlinesModel()
    model = ''
    
    data_path = "../data/Github_easy"
    