import os, json, random, torch, time
from argparse import ArgumentParser

os.environ['HF_HOME'] = './cache/'
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from models.BaseModel import BaseModel
from utils.logger import Logger

import utils.prompts as prompts

from datasets import load_dataset

def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--wrapper", type=str, default='guidance')
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--is_cpp", action='store_true', default=False)
    parser.add_argument("--json_shots", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--n_range", type=int, default=None, help="Run a systematic test on the given wrapper from n_shot=1 to n_shot=n_range")
    parser.add_argument("--n_exp", type=int, default=3, help="Number of times to run each test")
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
        except AssertionError:
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


def test_Quality(
    model: BaseModel, 
    test_dataset,
    wrapper: str, 
    num_shots: int, 
    example_questions: list,
    example_answers: list,
    logger: Logger, 
    output_file_name: str, 
    use_json_shots=False, 
    is_cpp=True):
    '''
    Run a quality test using the given parameters. It gets the accuracy of the model performing
    on given dataset. 
    '''

    # Log Metadata of Current Test here
    gpu_model = torch.cuda.get_device_name()
    cur_wrapper = f"{wrapper} wrapper" if wrapper != 'llm' else "Vanilla LLM"
    
    logger.log(f"Testing Quality of {cur_wrapper} with {num_shots} shots", force=True, to_file=True) 
    logger.log(f"GPU: {gpu_model}", force=True, to_file=True)
    
    backend = "LlamaCpp" if is_cpp or wrapper == 'llamacpp' else "Transformers"
    logger.log(f"Backend: {backend}", force=True, to_file=True)
    if use_json_shots:
        logger.log("Using JSON Shots", force=True, to_file=True)
    
    # Load test questions and answers
    questions = test_dataset['question']
    answers = test_dataset['answer']
    
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
    
    messages = prompts.create_prompt_template(
        example_questions=example_questions, 
        example_answers=example_answers, 
        n_shots=num_shots, 
        is_json=use_json_shots
    )
    
    correct = 0
    incorrects = []
    
    for i, (question, answer) in enumerate(zip(questions, answers)):
        answer = int(answer.split('####')[1].replace(",", "").strip())
        # We can use regex to find free generation results
        # answer_regex = r'"answer":[ ]?([1-9][0-9]{0,9})'
        messages.append(
            {
                "role": "user",
                "content": """Question: {question}, """.format(question=question)
            })
        messages.append(
            {
                "role": "assistant",
                "content": ""
            })
        # Run LLM here
        try: 
            # If run Vanilla LLM model, although we are passing the output_schema, it is not used
            output, _ = model.generate(messages, output_schema)
            break
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
        success, error_json, msg = validate_answer(question, output, answer,use_json_shots)
        if success:
            correct += 1
        else:
            logger.log(f"Question {i}: {msg}")
            incorrects.append(error_json)
        # Restore the messages
        messages.pop()
        if i % 100 == 0:
            logger.log(f"Question {i} completed", force=True)
        
    acc = correct/len(questions)*100
    
    logger.log("Test Complete", force=True)
    logger.log("Correct: {correct}/{total}, accuracy: {acc}%".format(correct=correct, total=len(questions), acc=acc), force=True, to_file=True)
    
    # Save incorrect answers
    with open(output_file_name, "w") as f:
        json.dump(incorrects, f, indent=4)
        
    return acc




if __name__ == "__main__":
    
    random.seed(42)
    torch.manual_seed(42)
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    model = get_model(args)
    gsm8k = load_dataset('gsm8k', 'main')
    
    # Get unified n-shot prompt for tests
    example_questions = gsm8k['train']['question']
    raw_answers = gsm8k['train']['answer']

    
    if args.json_shots:
        example_answers = []
        for answer in raw_answers:
            example_answers.append(prompts.parse_answer(answer))
    else:
        example_answers = raw_answers
    
    # Get the number of shots we want to experiment with
    # By default, we use the first n samples in train dataset
    # along with the example questions and answers used in 
    # Say what you mean rebuttal.
    num_shots = range(1, args.n_range + 1) if args.n_range is not None else [args.num_shots]
    
    # Manage Output Directory
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    out_dir = os.path.join('outputs', f"quality_{args.wrapper}")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    run_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"quality_{args.wrapper}_{'JSON' if args.json_shots else 'NL'}_shots"
    if args.n_range is not None:
        run_name += f"_range_{args.n_range}"
    run_name += f"_{run_time}"
    out_dir = os.path.join(out_dir, run_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += '/'
    
    logger = Logger(args.verbose, out_dir + "log.txt")
    logger.log("Quality Test", force=True, header="[SYSTEM]", to_file=True)
    logger.log(f"Model: {args.model}", force=True, header="[SYSTEM]", to_file=True)
    
    all_accs = []
    for n in num_shots:
        # Run Test 3 times and get average. Expect time is 3 hours    
        accs = []
        for run_count in range(args.n_exp):    
            logger.log(f"Run {run_count + 1}", force=True, header="[SYSTEM]", to_file=True)
            output_file_path = out_dir + f"quality_{args.wrapper}_{n}_{'JSON' if args.json_shots else 'NL'}_shots_run_{run_count + 1}"
            
            acc = test_Quality(
                model=model, 
                test_dataset=gsm8k['test'],
                wrapper=args.wrapper, 
                num_shots=n,
                example_questions=example_questions,
                example_answers=example_answers,
                logger=logger, 
                output_file_name=output_file_path + ".json",
                use_json_shots=args.json_shots,
                is_cpp=args.is_cpp
            )
            accs.append(acc)
        logger.log(f"Average Accuracy: {sum(accs)/len(accs)}", force=True, header="[SYSTEM]", to_file=True)
        all_accs.append((accs, sum(accs)/len(accs)))
    
    logger.log("All Accuracies", force=True, header="[SYSTEM]", to_file=True)
    for n, accs in zip(num_shots, all_accs):
        logger.log(f"{n} shots: {accs[0]}, avg: {accs[1]}", force=True, header="[SYSTEM]", to_file=True)
    
    model.close_model()