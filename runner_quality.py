import os, json, random, torch
from argparse import ArgumentParser

os.environ['HF_HOME'] = './cache/'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from models.BaseModel import BaseModel
from utils.logger import Logger

from datasets import load_dataset
from textwrap import dedent

def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--wrapper", type=str, default='guidance')
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--is_cpp", action='store_true', default=False)
    parser.add_argument("--json_shots", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)
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
    llm_name = args.model
    
    if wrapper_name == 'llm':
        from models.VanillaModel import VanillaModel
        assert not model, "Multiple models specified"
        model = VanillaModel(llm_name, args.is_cpp)
    
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


def validate_answer(question, output, answer, is_json=False) -> tuple:
    '''
    Validate the output against the answer. 
    If is_json is True, the output is expected to be a JSON string.
    
    Returns a tuple of (is_correct, error_message)
    if is_correct is True, error_message is None.
    '''
    if is_json:
        # The case we expect output to generate JSON outputs
        # Check validity
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            error_msg = 'Failed to parse output'
            error_json = {
                "question": question, 
                "reasoning": "Failed to parse output",
                "generated_answer": output, 
                "correct_answer": answer
            }
            return False, error_json, error_msg
        # Check json structure
        try:
            assert 'reasoning' in output
            assert 'answer' in output
        except KeyError:
            error_msg = 'Generated JSON does not fit schema'
            error_json = {
                "question": question, 
                "reasoning": "Generated JSON does not fit schema",
                "generated_answer": json.dumps(output), 
                "correct_answer": answer
            }
            return False, error_json, error_msg
        # Get reasoning and answer    
        reasoning = output['reasoning']
        gen_answer = output['answer']
    else:
        # The case of using Natural Language Shots, and generating NL outputs
        # This case should only be used for Vanilla LLMs. All wrappers should
        # generate JSON outputs
        try:
            reasoning, gen_answer = output.split('####')
            gen_answer = int(gen_answer.replace(",", "").strip())
        except ValueError:
            error_msg = 'Failed to parse output from NL'
            error_json = {
                "question": question, 
                "reasoning": "Failed to parse output",
                "generated_answer": output, 
                "correct_answer": answer
            }
            return False, error_json, error_msg
        
    # If we can get the answer, we can compare it
    if gen_answer == answer:
        # When it is correct
        return True, None, ""
    else:
        # When it is incorrect
        error_msg = f"Incorrect: Generated {gen_answer} instead of {answer}"
        error_json = {
            "question": question, 
            "reasoning": reasoning,
            "generated_answer": gen_answer, 
            "correct_answer": answer
        }
        return False, error_json, error_msg


def test_Quality(model: BaseModel, dataset, args, logger: Logger, output_file_name):

    # Log Metadata of Current Test here
    gpu_model = torch.cuda.get_device_name()
    cur_wrapper = f"{args.wrapper} wrapper" if args.wrapper != 'llm' else "Vanilla LLM"
    
    logger.log(f"Testing Quality of {cur_wrapper} with {args.num_shots} shots", force=True) 
    logger.log(f"GPU: {gpu_model}", force=True)
    
    if args.is_cpp or cur_wrapper == 'llamacpp':
        logger.log("Backend: LlamaCpp", force=True)
    else:
        logger.log("Backend: Transformers", force=True)
    if args.json_shots:
        logger.log("Using JSON Shots", force=True)
    
    # Load example questions and answers
    example_q = dataset['train']['question']
    all_answers = dataset['train']['answer']
    
    if args.json_shots:
        # Test Few-Shots with JSON format to ensure sample consistency
        example_a = []
        for answer in all_answers:
            example_ans = {
                "reasoning": answer.split('####')[0].strip(),
                "answer": int(answer.split('####')[1].replace(",", "").strip())
            }
            example_a.append(json.dumps(example_ans))
    else:
        # Use Natural Language Few Shots
        example_a = all_answers
    
    # Defines the schema for output   
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

    # Compose Few-Shot examples
    for i in random.sample(range(len(example_q)), args.num_shots):
        messages.append(
        {
            "role": "user",
            "content": """Question: {question}""".format(question=example_q[i])
        })
        messages.append(
        {
            "role": "assistant",
            "content": example_a[i]        
        })

    correct = 0
    incorrects = []
    questions = dataset['test']['question']
    answers = dataset['test']['answer']
    
    for i, (question, answer) in enumerate(zip(questions, answers)):
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
            # If run Vanilla LLM model, although we are passing the output_schema, it is not used
            output, _ = model.generate_all(messages, output_schema)
        except Exception as e:
            logger.log(f"Question {i}: Generation Error: {e}")
            incorrects.append({
                "question": question, 
                "reasoning": "Error During Generation",
                "generated_answer": "N/A",
                "error_message": str(e), 
                "correct_answer": answer
            })
            messages.pop()
            continue
        
        success, error_json, msg = validate_answer(question, output, answer, args.json_shots)
        if success:
            correct += 1
        else:
            logger.log(f"Question {i}: {msg}")
            incorrects.append(error_json)
        # Restore the messages
        messages.pop()
        
    acc = correct/len(questions)*100
    
    logger.log("Test Complete", force=True)
    logger.log("Correct: {correct}/{total}, accuracy: {acc}%".format(correct=correct, total=len(questions), acc=acc), force=True)
    
    # Save incorrect answers
    with open(output_file_name, "w") as f:
        json.dump(incorrects, f, indent=4)
        
    return acc
        

if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    model = get_model(args)
    gsm8k = load_dataset('gsm8k', 'main')
    
    logger = Logger(args.verbose)
    
    # Run Test 3 times and get average. Expect time is 3 hours    
    accs = []
    for run_count in range(3):    
        logger.log(f"Run {run_count + 1}", force=True, header="[SYSTEM]")
        output_file_name = f"quality_{args.wrapper}_{args.num_shots}_{'JSON' if args.json_shots else 'NL'}_shots_run_{run_count + 1}.json"
        acc = test_Quality(model, gsm8k, args, logger, output_file_name)
        accs.append(acc)
    
    model.close_model()
    logger.log(f"Average Accuracy: {sum(accs)/len(accs)}", force=True, header="[SYSTEM]")