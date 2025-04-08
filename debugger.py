'''
The goal of this debugger is to read initial prompts from a json and generate a first version of 
outputs, with thinking (For DeepSeek), and final output in the form of json schema.

TODO:
Functions:
1. Read json file for few-shot examples / use gem8k train set as default
2. generate(): Pipe in all examples, along with initial Assistant prompt, 
   and generate the output. By default, assistant prompt is empty.

Main Loop:
1. Write the main loop: Pipe in the few shot examples first, then pipe in
   the question. Wait for the assistant output and print it.
2. Ask users to decide whether the output is acceptable or not by typing [YES/NO]

3. If YES, write the output to a json file
4. If NO, ask the user to provide a better partial output. Then the loop takes
   in the new partial output, and pipe in to generate() function, and ask LLM
   to fill in the rest of the output.
5. Iterate until the user is satisfied with the output.
'''

from argparse import ArgumentParser
import json, os, config

os.environ["HF_HOME"] = config.HF_HOME

from datasets import load_dataset
from utils import prompts

from runner_quality import get_model, OUTPUT_SCHEMA


def pretty_print_messages(messages):
    """
    Pretty print the messages
    """
    output = ""
    for message in messages:
        output += "-" * 50 + "\n"
        if message['role'] == 'user':
            output += f"User:\n"
        elif message['role'] == 'assistant':
            output += "Assistant:\n"
        else:
            output += f"System:\n"
        output += message['content'] + "\n"
    output += "=" * 50
    print(output)

def add_args():
    parser = ArgumentParser()
    parser.add_argument("--input_json", type=str, default=None, help="Path to the input json containing few-shot examples")
    parser.add_argument("--json_shots", action='store_true', help="Use json format for few-shot examples")
    parser.add_argument("--n_shots", type=int, default=0, help="how many examples to use for few-shot learning")
    parser.add_argument("--model", type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--wrapper", type=str, default='guidance')
    parser.add_argument("--is_cpp", action='store_true', help="Use llama_cpp backend")
    parser.add_argument("--temperature", type=float, default=0.6)
    return parser


def debugger(args, example_questions, example_answers):
    model = get_model(args)
    # create prompt
    messages = prompts.create_prompt_template(
        example_questions=example_questions,
        example_answers=example_answers,
        n_shots=args.n_shots,
        is_json=args.json_shots,
        is_deepseek='DeepSeek-R1' in args.model,
        use_static_first_three=args.input_json is None  # If json is given, we use all examples from json.
    )
    
    starting_message = "=" * 50 + "\n"
    starting_message += "Starting Debugger\n"
    starting_message += "=" * 50 + "\n"

    pretty_print_messages(messages)

    # Main Loops Starts here
    while True:
        input_question = input("Enter your question [Type <EXIT> to quit program]: \n")

        if input_question.lower() == 'exit':
            print("Exiting Debugger...")
            break

        print("=" * 50)
        # Add question to the prompt
        messages.append({
            "role": "user",
            "content": input_question
        })

        messages.append({
            "role": "assistant",
            "content": ""
        })
        
        # Start debugging loop
        while True:
            # Let the model generate
            raw_input, output, _ = model.generate_all(messages, OUTPUT_SCHEMA, args.temperature)
            # save the output to a json file
            output_str = ""
            output_str += "Assistant:\n"
            output_str += output + "\n"
            output_str += "=" * 50 + "\n"

            
            # print the output
            print("Assistant: ")
            print(output)
            # check if the output is acceptable
            print("=" * 50)
            user_input = input("Do you want to refine the assistant message? [YES/NO]: ")

            while True:
                if user_input.lower() == 'no':
                    # TODO: Log the output
                    messages.pop()  # Pop the assistant field
                    messages.pop()  # Pop the previous question 
                    break
                elif user_input.lower() == 'yes':
                    # Log the current input 
                    # ask the user for a better partial output
                    print("=" * 50)
                    user_input = input("Please provide a better partial output: \n")
                    
                    messages.pop()
                    messages.append({
                        "role": "assistant",
                        "content": user_input
                    })
                    print("=" * 50)
                    break
                else:
                    user_input = input("Invalid input. Please enter [YES/NO].")
                    continue
            
            if user_input.lower() == 'no':
                # Exit the loop
                print("=" * 50)
                break


if __name__ == "__main__":
    # Read Json file
    example_questions = []
    example_answers = []

    arg_parser = add_args()
    args = arg_parser.parse_args()

    if args.input_json is None:
        # Use GSM8K training set as default
        print("Loading dataset...")
        gsm8k = load_dataset('gsm8k', 'main')
        print("Dataset loaded")
        
        # Get unified n-shot prompt for tests
        example_questions = gsm8k['train']['question'][0:args.n_shots]
        raw_answers = gsm8k['train']['answer'][0:args.n_shots]

        if args.json_shots:
            example_answers = []
            for answer in raw_answers:
                example_answers.append(prompts.parse_answer(answer))
        else:
            example_answers = raw_answers
    else:
        # Read the json file and parse it into a list of prompts
        print(f"Reading input json from {args.input_json}")
        with open(args.input_json, 'r') as f:
            data = json.load(f)
            for example in data[0:args.n_shots]:
                example_questions.append(example['question'])
                example_answers.append(example['answer'])
        
    # Run the debugger
    debug_log = debugger(args, example_questions, example_answers)

    # Save the debug log to a json file
    with open("debug_log.json", 'w') as f:
        json.dump(debug_log, f, indent=4)

