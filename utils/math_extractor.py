import re, json, math

def eval_equation(equation):
    # Evaluate the equation
    equation = equation[2:-2]
    # Replace all whitespace and commas
    equation = equation.replace(" ", "")
    equation = equation.replace(",", "")
    try:
        equation = equation.split("=")
        lhs = equation[0]
        # Remove any whitespace from lhs
        rhs = equation[1]
        # Find decimal of rhs
        if len(rhs.split(".")) > 1:
            dec = len(rhs.split(".")[1])
            eval_res = round(eval(lhs), dec)
            rhs = ".".join(rhs.split(".")[0:2])
        else:
            eval_res = eval(lhs)
        if math.isclose(eval_res, float(eval(rhs))):
            return True
    except Exception as e:
        return False
    return False



def get_math_from_reasoning(string):
    '''
    Step 1, split reasoning by [,.;:!?]
    Step 2, use regex to estimate number of equations in each subsentence: 
        - if there is a match to [+-*/^], this subsentence is considered having a math equation
    Step 3, use regex to find all equations with <<>> format in the subsentence
        - if there is a match to <<>>, this subsentence is considered having a structured math equation
        - Evaluate the equation and check if it is correct
        - If incorrect, record the equation
    Step 4, Record # of equations, and # of structured equations in the reasoning

    Returns:
        - total_equations: total number of equations in the reasoning
        - total_structured_equations: total number of structured equations in the reasoning
        - equations: list of all parsed structured equations in the reasoning

    '''

    total_equations = 0
    total_structured_equations = 0
    correct_count = 0
    incorrect_equations = []
    all_matches = []

    # Step 1
    subsentences = re.split(r'[,.;:!?]', string)
    for subsentence in subsentences:
        # Step 2
        if re.search(r'[+-/*^]', subsentence):
            # Step 3
            matches = get_math(subsentence)
            if matches:
                total_equations += len(matches)
                all_matches.extend(matches)
                for match in matches:
                    # Step 4
                    total_structured_equations += 1
                    if not eval_equation(match):
                        incorrect_equations.append(match)
                    else:
                        correct_count += 1
            else:
                # Assume there is only one equation in the subsentence
                total_equations += 1
            
                
    return total_equations, total_structured_equations, correct_count, all_matches, incorrect_equations


def get_math(string):
    pattern = re.compile(r'<<.*?>>')
    matches = re.findall(pattern=pattern, string=string)
    return matches


def get_math_from_json(out_path):

    # out_path = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_guidance/quality_guidance_JSON_shots_2025-03-26-15-33-27_deepseek_r1"
    out_path = "/w/284/jameschen/Reproduce_JsonSchemaBenchmark/outputs/quality_guidance/quality_guidance_JSON_shots_2025-03-26-14-03-26_with_llama"
    json_file = f"{out_path}/all_incorrects.json"
    with open(json_file, 'r') as f:
        jsons = json.load(f)
        
    maths = {}
    no_valid_equation = {}
    for question in jsons.keys():
        maths[question] = []
        for reasoning in jsons[question]['reasonings']:
            parsed_math = get_math(reasoning)
            maths[question].extend(parsed_math)
            if not parsed_math:
                if question not in no_valid_equation:
                    no_valid_equation[question] = [reasoning] 
                else:
                    no_valid_equation[question].append(reasoning)
    
    c_total_incorrect = 0
    total_incorrect = []
    for question in list(maths.keys()):
        for equation in maths[question]:
            if not equation:
                continue
            # Extract equations from <<equation>> format
            equation = equation[2:-2]
            # Replace all whitespace and commas
            equation = equation.replace(" ", "")
            equation = equation.replace(",", "")
            
            equation = equation.split("=")
            lhs = equation[0]
            # Remove any whitespace from lhs
            rhs = equation[1]
            # Find decimal of rhs
            if len(rhs.split(".")) > 1:
                dec = len(rhs.split(".")[1])
                eval_res = round(eval(lhs), dec)
                rhs = ".".join(rhs.split(".")[0:2])
            else:
                eval_res = eval(lhs)
            if eval_res != float(rhs):
                c_total_incorrect += 1
                total_incorrect.append((question, f"{equation[0]}={eval(lhs)}", equation[1]))
            
    table = []
    from tabulate import tabulate
    for inc in total_incorrect:
        table.append(inc)
        
    with open(f"{out_path}/incorrect_maths.txt", 'w') as f:
        f.write("Question<split>Equation<split>Generated Result\n")
        for inc in total_incorrect:
            f.write(f"{inc[0]}<split>{inc[1]}<split>{inc[2]}\n")
    print(tabulate(table, headers=["Question", "Equation", "Generated Result"], maxcolwidths=[25, None, None], tablefmt="grid"))
    
    dump_file_name = f"{out_path}/invalid_maths.json"
    with open(dump_file_name, 'w') as f:
        json.dump(no_valid_equation, f, indent=4)
    
    print()
    print(f"From overall of {len(jsons)} equations:")
    print(f"Found incorrect math: {c_total_incorrect}")
    print(f"Invalid math reasoning: {len(no_valid_equation)}")


if __name__ == "__main__":
    pass
    # get_math_from_json(json_file)