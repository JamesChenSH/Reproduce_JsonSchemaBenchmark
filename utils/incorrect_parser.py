import os, sys, json

def aggregate_gsm8k_incorrect(jsons)->dict:

    all_incorrect = {}
    for question in jsons:
        # print(question)
        if question['correct'] is True:
            # Ignore correct answers
            continue
        if question['question'] not in all_incorrect:
            all_incorrect[question['question']] = {
                "question": question['question'],
                "correct_answer": question['correct_answer'],
                "reasonings": [],
                "incorrect_answers": [],
                # "mathematical_errors": []
            }
        parsed_json = question['parsed_json']
        if parsed_json == 'N/A':
            reasoning = 'No Valid LLM Generation'
            generated_answer = 'N/A'
        else:
            reasoning = parsed_json['reasoning']
            generated_answer = parsed_json['generated_answer']
        all_incorrect[question['question']]['incorrect_answers'].append(generated_answer)
        all_incorrect[question['question']]['reasonings'].append(reasoning)
        # all_math_errs = find_mathematical_errors(question['reasoning'])
        # all_incorrect[question['question']]['mathematical_errors'].append(all_math_errs)
    return all_incorrect


def try_fix_incorrect_parse(fp)-> None:
    new_json = []
    cant_fix = 0
    with open(fp, 'r') as f:
        jsons = json.load(f)
    for question in jsons:
        if question['correct'] is True:
            # Ignore correct answers
            new_json.append(question)
            continue
        if question['parsed_json'] == 'N/A':
            # Skip questions without valid LLM generation
            cant_fix += 1
            new_json.append(question)
            continue
        if question['parsed_json']['reasoning'] != 'N/A':
            # Ignore questions that is confirmed incorrect answer
            new_json.append(question)
            continue

        try:
            generated_ans = question['parsed_json']['generated_answer']
            generated_ans = generated_ans.replace("\\", "").strip()
            first_bracket = generated_ans.find('{')
            if first_bracket == -1:
                # No json in output at all, which is just bad
                cant_fix += 1
                new_json.append(question)
                continue
            parsed_json = json.loads(generated_ans[first_bracket:])
            question['parsed_json'] = {
                "reasoning": parsed_json['reasoning'],
                "generated_answer": parsed_json['answer']
            }
            question['correct'] = parsed_json['answer'] == question['correct_answer']
            if not question['correct']:
                question['error_message'] = f"Incorrect: Generated {parsed_json['answer']} instead of {question['correct_answer']}"
            else:
                question['error_message'] = ''
        except json.JSONDecodeError as e:
            # Still incorrect
            cant_fix += 1
        new_json.append(question)
    # Save the modified jsons back to the file
    with open(fp, 'w') as f:
        json.dump(jsons, f, indent=4)
    print("There are {} questions that cannot be fixed.".format(cant_fix))
    


def parse_one_dir(dir: str):
    os.makedirs(dir, exist_ok=True)
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
    # output_dir = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_guidance/quality_guidance_JSON_shots_2025-03-26-15-33-27"

    # output_dir = "/mnt/d/3_research/Constraint Decoding/Reproduce_JsonSchemaBenchmark/outputs/quality_llm+guidance/quality_llm+guidance_JSON_shots_2025-04-02-17-46-32"
    output_dir = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_guidance/quality_guidance_JSON_shots_2025-04-03-02-20-23"
    parse_one_dir(output_dir)

    # try_fix_incorrect_parse('./quality_llm+guidance_8_JSON_shots_run_1.json')