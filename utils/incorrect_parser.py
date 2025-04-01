import os, sys, json

def aggregate_gsm8k_incorrect(jsons: list[str])->dict:
    all_incorrect = {}
    for question in jsons:
        # print(question)
        if question['question'] not in all_incorrect:
            all_incorrect[question['question']] = {
                "question": question['question'],
                "correct_answer": question['correct_answer'],
                "reasonings": [],
                "incorrect_answers": [],
                # "mathematical_errors": []
            }
        all_incorrect[question['question']]['incorrect_answers'].append(question['generated_answer'])
        all_incorrect[question['question']]['reasonings'].append(question['reasoning'])
        # all_math_errs = find_mathematical_errors(question['reasoning'])
        # all_incorrect[question['question']]['mathematical_errors'].append(all_math_errs)
    return all_incorrect

def find_mathematical_errors(reasoning: str)->list[tuple[str, int, int]]:
    '''
    Find all equations in the list, then find all mathematical errors in the equations.
    Return a list of tuples of the form (equation, correct_ans, incorrect_ans)
    '''
    all_math_errs = []
    
    pass 


def parse_one_dir(dir: str):
    os.makedirs(output_dir, exist_ok=True)
    all_jsons = []
    for file in os.listdir(dir):
        if '.json' in file and 'all_incorrects' not in file:
            with open(os.path.join(dir, file), 'r') as f:
                jsons = json.load(f)
                all_jsons.extend(jsons)
                
    all_incorrect = aggregate_gsm8k_incorrect(all_jsons)
    with open(os.path.join(dir, "all_incorrects.json"), 'w') as f:
        json.dump(all_incorrect, f, indent=4)
    print("Aggregated all incorrect answers and reasoning into a single JSON file. There is a total of {} incorrect questions.".format(len(all_incorrect)))


if __name__ == "__main__":
    # output_dir = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_guidance/quality_guidance_JSON_shots_range_8_2025-03-17-22-34-42"
    # output_dir = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_llamacpp/quality_llamacpp_JSON_shots_range_8_2025-03-16-15-48-25"
    # output_dir = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_llm/quality_llm_JSON_shots_range_8_2025-03-16-15-48-16"
    # output_dir = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_llm/quality_llm_NL_shots_range_8_2025-03-17-13-18-10"
    # output_dir = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_guidance/quality_guidance_JSON_shots_2025-03-26-14-03-26_with_llama"
    output_dir = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_guidance/quality_guidance_JSON_shots_2025-03-26-15-33-27"
    os.makedirs(output_dir, exist_ok=True)
    
    all_jsons = []
    for file in os.listdir(output_dir):
        if '.json' in file and 'all_incorrects' not in file:
            with open(os.path.join(output_dir, file), 'r') as f:
                jsons = json.load(f)
                all_jsons.extend(jsons)
                
    all_incorrect = aggregate_gsm8k_incorrect(all_jsons)
    with open(os.path.join(output_dir, "all_incorrects.json"), 'w') as f:
        json.dump(all_incorrect, f, indent=4)
    print("Aggregated all incorrect answers and reasoning into a single JSON file. There is a total of {} incorrect questions.".format(len(all_incorrect)))