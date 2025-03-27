import re, json

def get_math(string):
    pattern = re.compile(r'<<.*?>>')
    matches = re.findall(pattern=pattern, string=string)
    return matches

if __name__ == "__main__":
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
        
    with open(f"{out_path}/invalid_math.txt", 'w') as f:
        f.write("Question<split>Equation<split>Generated Result\n")
        for inc in total_incorrect:
            f.write(f"{inc[0]}<split>{inc[1]}<split>{inc[2]}\n")
    print(tabulate(table, headers=["Question", "Equation", "Generated Result"], maxcolwidths=[25, None, None], tablefmt="grid"))
    
    dump_file_name = f"{out_path}/invalid_math.json"
    with open(dump_file_name, 'w') as f:
        json.dump(no_valid_equation, f, indent=4)
    
    print()
    print(f"From overall of {len(jsons)} equations:")
    print(f"Found incorrect math: {c_total_incorrect}")
    print(f"Invalid math reasoning: {len(no_valid_equation)}")
