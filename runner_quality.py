import time
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
from datasets import load_dataset
from textwrap import dedent


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


def test_Quality(model: BaseModel, dataset):

    output_schema = '{"reasoning":<reasoning about the answer>, "answer": <final answer>}'
    messages = [{
        "role": "system",
        "content": dedent(f"""
        You are an expert in solving grade school math tasks. You will be presented with a grade-school math word problem and be asked to solve it.
        Before answering you should reason about the problem (using the "reasoning" field in the JSON response described below).
            
        You will always repond with JSON in the format described below:
        
        {output_schema}

        The "reasoning" field will contain your reasoning about the sequence of events, and the "answer" will contain the single letter representing the correct choice you are presented with.
        """)
    }]

    example_question = [
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
    ]

    example_response = [
        """{"reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.", "answer": 6}""",
        """{"reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.", "answer": 5}""",
        """{"reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.","answer": 39"""
    ]

    for i in range(len(example_question)):
        messages.append(
        {
            "role": "user",
            "content": """Question: {question}""".format(question=example_question[i])
        }
        )
        messages.append(
        {
            "role": "assistant",
            "content": example_response[i]        
        })

    for question in dataset:

        messages.append(
            {
                "role": "user",
                "content": """Question: {question}", """.format(question=question)
            })
        # Run LLM here
        output, gen_len = model.generate_all(messages, output_schema)
        print(output)

        # Remove the added question and results
        messages.pop()


        

if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    model = get_model(args)
    gsm8k = load_dataset('gsm8k', 'main')['test']
    
    test_Quality(model, gsm8k)