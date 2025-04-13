import os, json, random, torch, time

import config
from argparse import ArgumentParser
from tqdm import tqdm

os.environ['HF_HOME'] = config.HF_HOME

# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from models.BaseModel import BaseModel
from utils.logger import Logger
from utils.math_extractor import get_math_from_reasoning
import utils.prompts as prompts

from datasets import load_dataset

OUTPUT_SCHEMA = {
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

OUTPUT_SCHEMA = json.dumps(OUTPUT_SCHEMA)

def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--wrapper", type=str, default='guidance')
    parser.add_argument("--length", type=int, default=100000)
    parser.add_argument("--n_shots", type=int, default=8)
    parser.add_argument("--n_question", type=int, default=10000)
    parser.add_argument("--is_cpp", action='store_true', default=False)
    parser.add_argument("--json_shots", action='store_true', default=False)
    parser.add_argument("--n_range", type=int, default=None, help="Run a systematic test on the given wrapper from n_shot=1 to n_shot=n_range")
    parser.add_argument("--n_exp", type=int, default=3, help="Number of times to run each test")
    parser.add_argument("--temperature", type=float, default=0.6)

    # Display options
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--tqdm", action='store_true', default=False, help="Use tqdm to show progress bar")
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
        from models.LLM import VanillaModel
        assert not model, "Multiple models specified"
        model = VanillaModel(llm_name, args.is_cpp)
    
    if wrapper_name == 'llm+debug':
        from models.LLMDebug import VanillaModel
        assert not model, "Multiple models specified"
        model = VanillaModel(llm_name, args.is_cpp)

    if wrapper_name == 'guidance':
        from models.Guidance_quality import GuidanceModel
        # llama cpp method
        assert not model, "Multiple models specified"
        model = GuidanceModel(llm_name, args.is_cpp)

    if wrapper_name == 'outlines':
        from models.Outlines import OutlinesModel
        assert not model, "Multiple models specified"
        model = OutlinesModel()

    if wrapper_name == 'llamacpp':
        from models.LlamaCpp import LlamaCppModel
        assert not model, "Multiple models specified"
        model = LlamaCppModel(llm_name)

    if not model:
        raise Exception("No grammar model specified")

    return model


def validate_answer(question, output: str, answer:int, is_json=False) -> tuple[bool, dict, str]:
    '''
    Validate the output against the answer. 
    If is_json is True, the output is expected to be a JSON string.
    
    Returns a tuple of (is_correct, error_message)
    if is_correct is True, error_message is None.
    '''
    na_reasoning = "N/A"
    if is_json:
        # The case we expect output to generate JSON outputs
        try:
            # Skip thinking process that may occur in DeepSeek model
            if "</think>" in output:
                output = output[output.find("</think>") + len("</think>"):]

            # Handle characters json cannot decode
            output = output.replace("\\", "")
            json_start_idx = output.find("{")
            if json_start_idx == -1:
                raise json.JSONDecodeError("Invalid JSON", output, 0)
            output = output[json_start_idx:]
            output = json.loads(output)

        except json.JSONDecodeError:
            error_msg = 'Failed to parse output'
            parsed_json = {
                "reasoning": na_reasoning,
                "generated_answer": output,
            }
            return False, parsed_json, error_msg
        # Check json structure
        try:
            assert 'reasoning' in output
            assert 'answer' in output
        except AssertionError:
            error_msg = 'Generated JSON does not fit schema'
            parsed_json = {
                "reasoning": na_reasoning,
                "generated_answer": json.dumps(output)
            }
            return False, parsed_json, error_msg
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
            parsed_json = {
                "reasoning": na_reasoning,
                "generated_answer": output
            }
            return False, parsed_json, error_msg
        
    # If we can get the answer, we can compare it
    parsed_json = {
        "reasoning": reasoning,
        "generated_answer": gen_answer
    }
    if gen_answer == answer:
        # When it is correct
        return True, parsed_json, ""
    else:
        # When it is incorrect
        error_msg = f"Incorrect: Generated {gen_answer} instead of {answer}"
        return False, parsed_json, error_msg


def test_Quality(
    model: BaseModel, 
    test_dataset,
    wrapper: str, 
    num_shots: int, 
    num_questions: int,
    example_questions: list,
    example_answers: list,
    logger: Logger, 
    output_file_name: str, 
    use_json_shots=False, 
    is_cpp=True,
    temperature: float = 0.6,
    is_deepseek=False,     
):
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
    questions = test_dataset['question'][:num_questions]
    answers = test_dataset['answer'][:num_questions]
    
    messages = prompts.create_prompt_template(
        example_questions=example_questions, 
        example_answers=example_answers, 
        n_shots=num_shots, 
        is_json=use_json_shots,
        is_deepseek=is_deepseek,
    )
    
    # Define Initial Statistics Values
    correct = 0
    total_potential_equations = 0
    total_parsed_equations = 0
    total_correct_verified_equations = 0
    
    with open(output_file_name, "w") as f:
        f.write("[")

    if args.tqdm:
        iterator = tqdm(enumerate(zip(questions, answers)), total=len(questions))
    else:
        iterator = enumerate(zip(questions, answers))
    for i, (question, answer) in iterator:
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
            raw_input, output, _ = model.generate_all(messages, OUTPUT_SCHEMA, temperature=temperature)
        except Exception as e:
            logger.log(f"Question {i}: Generation Error: {e}")
            question_log = {
                "question": question, 
                "full_prompt": raw_input,
                "correct_answer": answer,
                "generated_raw": "N/A",
                "parsed_json": "N/A",
                "error_message": "Error During Generation, error message: " + str(e),
                "correct": False
            }
            # Save logs
            with open(output_file_name, "a") as f:
                json.dump(question_log, f, indent=4)
                if i != len(questions) - 1:
                    f.write(",\n")

            messages.pop()
            messages.pop()
            continue

        is_correct, parsed_json, msg = validate_answer(question, output, answer, use_json_shots)

        question_log = {
            "question": question, 
            "full_prompt": raw_input,
            "correct_answer": answer,
            "generated_raw": output,
            "parsed_json": parsed_json,
            "error_message": msg,
            "correct": is_correct
        }

        if msg == "":
            # Parse math in reasoning
            reasoning = parsed_json['reasoning']
            n_equations, n_parsed_equations, n_correct, parsed_equations, incorrects = get_math_from_reasoning(reasoning)
            # Log the math parsers
            extracted_math = {
                "n_equations": n_equations,
                "n_parsed_equations": n_parsed_equations,
                "n_correct": n_correct,
                "parsed_equations": parsed_equations,
                "incorrects": incorrects
            }
            question_log['extracted_math'] = extracted_math
            total_potential_equations += n_equations
            total_parsed_equations += n_parsed_equations
            total_correct_verified_equations += n_correct

        # Save logs
        with open(output_file_name, "a") as f:
            json.dump(question_log, f, indent=4)
            if i != len(questions) - 1:
                f.write(",\n")

        if is_correct:
            correct += 1
        else:
            logger.log(f"Question {i}: {msg}")
        # Restore the messages
        messages.pop()
        messages.pop()


        if isinstance(iterator, tqdm):
            iterator.set_description(f"Correct: {correct}/{i+1}")
        else:
            if i % 100 == 0:
                logger.log(f"Question {i} completed", force=True)
        
    acc = correct/len(questions)*100
    
    logger.log("Test Complete", force=True)
    logger.log("Correct: {correct}/{total}, accuracy: {acc}%".format(correct=correct, total=len(questions), acc=acc), force=True, to_file=True)

    logger.log("Total Potential Equations: {total_potential_equations}".format(total_potential_equations=total_potential_equations), force=True, to_file=True)
    logger.log("Total Parsed Equations: {total_parsed_equations}".format(total_parsed_equations=total_parsed_equations), force=True, to_file=True)
    logger.log("Total Correct Verified Equations: {total_correct_verified_equations}".format(total_correct_verified_equations=total_correct_verified_equations), force=True, to_file=True)
    if total_potential_equations > 0:
        logger.log("Parsed Equation Accuracy: {parsed_equation_acc}%".format(parsed_equation_acc=total_correct_verified_equations/total_parsed_equations*100), force=True, to_file=True)
    
    # Save incorrect answers
    with open(output_file_name, "a") as f:
        f.write("\n]")
    logger.log("All logs saved to " + output_file_name, force=True, to_file=True)
        
    return acc




if __name__ == "__main__":
    
    random.seed(42)
    torch.manual_seed(42)
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("Getting model...")
    model = get_model(args)
    print("Model loaded")

    # Load the dataset
    print("Loading dataset...")
    gsm8k = load_dataset('gsm8k', 'main')
    print("Dataset loaded")
    
    
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
    num_shots = range(1, args.n_range + 1) if args.n_range is not None else [args.n_shots]
    
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
                num_questions=args.n_question,
                example_questions=example_questions,
                example_answers=example_answers,
                logger=logger, 
                output_file_name=output_file_path + ".json",
                use_json_shots=args.json_shots,
                is_cpp=args.is_cpp,
                is_deepseek='DeepSeek-R1' in args.model,
                temperature=args.temperature
            )
            accs.append(acc)
        logger.log(f"Average Accuracy: {sum(accs)/len(accs)}", force=True, header="[SYSTEM]", to_file=True)
        all_accs.append((accs, sum(accs)/len(accs)))
    
    logger.log("All Accuracies", force=True, header="[SYSTEM]", to_file=True)
    for n, accs in zip(num_shots, all_accs):
        logger.log(f"{n} shots: {accs[0]}, avg: {accs[1]}", force=True, header="[SYSTEM]", to_file=True)
    
    model.close_model()