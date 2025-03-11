import time
import os, sys, time, json, random
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


def test_Quality(model: BaseModel, dataset, args):

    example_q = dataset['train']['question']
    example_a = dataset['train']['answer']
            
    output_schema = {
        "properties": {
            "reasoning": {
                "title": "Reasoning", 
                "type": "string"
            }, 
            "answer": {
                "title": "Answer", 
                "type": "integer"
            }
        }, 
        "required": [
            "reasoning", 
            "answer"
        ], 
        "title": "ReturnedModel", 
        "type": "object"
    }

    output_schema = json.dumps(output_schema)

    messages = [{
        "role": "system",
        "content": dedent("""
        You are an expert in solving grade school math tasks. You will be presented with a grade-school math word problem and be asked to solve it.
        Before answering you should reason about the problem (using the "reasoning" field in the JSON response described below).
            
        You will always repond with JSON in the format described below:
        
        {"reasoning":<reasoning about the answer>, "answer": <final answer>}

        The "reasoning" field will contain your reasoning about the sequence of events, and the "answer" will contain the single letter representing the correct choice you are presented with.
        """)
    }]


    for i in random.sample(range(len(example_q)), 8):
        messages.append(
        {
            "role": "user",
            "content": """Question: {question}""".format(question=example_q[i])
        }
        )
        messages.append(
        {
            "role": "assistant",
            "content": example_a[i]        
        })

    correct = 0
    incorrects = []
    
    questions = dataset['test']['question']
    answers = dataset['test']['answer']
    
    for question, answer in zip(questions, answers):
        answer = int(answer.split('####')[1].replace(",", "").strip())
        # We can use regex to find free generation results
        # answer_regex = r'"answer":[ ]?([1-9][0-9]{0,9})'
        messages.append(
            {
                "role": "user",
                "content": """Question: {question}, """.format(question=question)
            })
        # Run LLM here
        try: 
            output, gen_len = model.generate_all(messages, output_schema)
        except Exception as e:
            print(f"Generation Error: {e}", flush=True)
            incorrects.append({
                "question": question, 
                "reasoning": "Error During Generation",
                "generated_answer": output, 
                "correct_answer": answer
            })
            messages.pop()
            continue
        # Parse to json
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            print("Failed to parse output", flush=True)
            incorrects.append({
                "question": question, 
                "reasoning": "Failed to parse output",
                "generated_answer": output, 
                "correct_answer": answer
            })
            messages.pop()
            continue
        # Validate output
        gen_answer = output['answer']
        if gen_answer == answer:
            print("Correct", flush=True)
            correct += 1
        else:
            print("Incorrect", flush=True)
            incorrects.append({
                "question": question, 
                "reasoning": output['reasoning'],
                "generated_answer": gen_answer, 
                "correct_answer": answer
            })

        # Remove the added question and results
        messages.pop()

    print("Correct: {correct}/{total}, accuracy: {acc}%".format(correct=correct, total=len(dataset), acc=correct/len(dataset)), flush=True)
    with open(f"incorrects_{args.wrapper}.json", "w") as f:
        json.dump(incorrects, f, indent=4)
    model.close_model()
        

if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    model = get_model(args)
    gsm8k = load_dataset('gsm8k', 'main')
    
    test_Quality(model, gsm8k, args)