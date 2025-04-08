from textwrap import dedent
import json
'''
- question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? 
target: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8

- question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
target: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9

- question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
target: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29

- question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
target: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer
is 33

- question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
target: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15
dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8
'''
EXAMPLE_QUESTION = [
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
    "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
    "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
    "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
]
EXAMPLE_RESPONSE = [
    """{"reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = <<21-15=6>>6.", "answer": 6}""",
    """{"reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = <<3+2=5>>5.", "answer": 5}""",
    """{"reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = <<32+42=74>>74. After eating 35, they had 74 - 35 = <<74-35=39>>39.","answer": 39}""",
    """{"reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = <<20-12=8>>8.", "answer": 8}""",
    """{"reasoning": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = <<5+4=9>>9.", "answer": 9}""",
    """{"reasoning": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.", "answer": 29}""",
    """{"reasoning": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = <<58-23=35>>35. After losing 2 more, he had 35 - 2 = <<35-2=33>>33 golf balls.", "answer": 33}""",
    """{"reasoning": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = <<5*3=15>>15 dollars. So she has 23 - 15 dollars left. <<23-15=8>>8.", "answer": 8}"""
]

EXAMPLE_NL_RESPONSE = [
    "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = <<21-15=6>>6.####6",
    "There are originally 3 cars. 2 more cars arrive. 3 + 2 = <<3+2=5>>5.####5",
    "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = <<32+42=74>>74. After eating 35, they had 74 - 35 = <<74-35=39>>39.####39",
    "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = <<20-12=8>>8.####8",
    "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = <<5+4=9>>9.####9",
    "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.####29",
    "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = <<58-23=35>>35. After losing 2 more, he had 35 - 2 = <<35-2=33>>33 golf balls.####33",
    "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = <<5*3=15>>15 dollars. So she has 23 - 15 dollars left. <<23-15=8>>8.####8"
]

EXAMPLE_JSON_STRUCTURE = '{"reasoning":<reasoning about the answer>, "answer": <final answer>}'
EXAMPLE_NL_STRUCTURE = '<reasoning about the answer>####<final answer>'


def parse_answer(answer:str):
    '''
    Parse GSM8K question in natural language to a fixed JSON format.
    '''
    ans = {
        "reasoning": answer.split('####')[0].strip(),
        "answer": int(answer.split('####')[1].replace(",", "").strip())
    }
    return json.dumps(ans)
    

def create_prompt_template(example_questions=None, example_answers=None, n_shots=3, is_json=False, is_deepseek=False, use_static_first_three=True):
    '''
    Create the chat template prompt with 3 given example questions and responses.
    Then add the user's question to the prompt.
    '''
    n_needed = n_shots
    example_question = []
    example_answer = []
    if use_static_first_three:
        if n_shots <= len(EXAMPLE_QUESTION):
            example_question = EXAMPLE_QUESTION[:n_shots]
            example_answer = EXAMPLE_RESPONSE[:n_shots] if is_json else EXAMPLE_NL_RESPONSE[:n_shots]
            n_needed = 0
        else:
            example_question = EXAMPLE_QUESTION
            example_answer = EXAMPLE_RESPONSE if is_json else EXAMPLE_NL_RESPONSE
            n_needed = n_shots - len(EXAMPLE_QUESTION)
        
    example_question.extend(example_questions[:n_needed])
    example_answer.extend(example_answers[:n_needed])
    
    # assert len(example_question) == n_shots
        
    messages = [{
        "role": "user" if is_deepseek else "system",
        "content": dedent(f"""
        You are an expert in solving grade school math tasks. You will be presented with a grade-school math word problem and be asked to solve it.
        Before answering you should reason about the problem (using the "explanation" field in the JSON response described below).
        
        All mathematical calculations in the "explanation" field should be in the format LHS = <<LHS=RHS>>RHS, and without units.
        All thinkings should be less than 300 words
            
        You will always repond{" with JSON" if is_json else ""} in the format described below:
        
        {EXAMPLE_JSON_STRUCTURE if is_json else EXAMPLE_NL_STRUCTURE}
        
        The "explanation" field will contain your explanation about the sequence of events, and the "answer" field will contain the single number representing the correct answer you are presented with.
        """)
    },]
    
    for i in range(n_shots):
        messages.append({
            "role": "user",
            "content": """Question: {question}""".format(question=example_question[i].split['Question: '] if 'Question: ' in example_question[i] else example_question[i])
        })
        messages.append({
            "role": "assistant",
            "content": example_answer[i]        
        })
    
    return messages


if __name__ == "__main__":
    '''
    Print 8 shot examples
    '''
    import os
    os.environ['HF_HOME'] = '../cache/'
    from datasets import load_dataset
    
    gsm8k = load_dataset('gsm8k', 'main')
    
    # Get unified n-shot prompt for tests
    example_questions = gsm8k['train']['question']
    raw_answers = gsm8k['train']['answer']
    
    example_answers = []
    for answer in raw_answers:
        example_answers.append(parse_answer(answer))
        
    for i in range(8):
        print(example_questions[i])
        print(example_answers[i])
        print()