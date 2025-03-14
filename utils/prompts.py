from textwrap import dedent
import json

EXAMPLE_QUESTION = [
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
]
EXAMPLE_RESPONSE = [
    """{"reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.", "answer": 6}""",
    """{"reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.", "answer": 5}""",
    """{"reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.","answer": 39"""
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
    

def create_prompt_template(example_questions=None, example_answers=None, n_shots=3, is_json=False):
    '''
    Create the chat template prompt with 3 given example questions and responses.
    Then add the user's question to the prompt.
    '''
    n_needed = 0
    if n_shots <= len(EXAMPLE_QUESTION):
        example_question = EXAMPLE_QUESTION[:n_shots]
        example_answer = EXAMPLE_RESPONSE[:n_shots]
    else:
        example_question = EXAMPLE_QUESTION
        example_answer = EXAMPLE_RESPONSE
        n_needed = n_shots - len(EXAMPLE_QUESTION)
        
    example_question.extend(example_questions[:n_needed])
    example_answer.extend(example_answers[:n_needed])
    
    assert len(example_question) == n_shots
        
    messages = [{
        "role": "system",
        "content": dedent(f"""
        You are an expert in solving grade school math tasks. You will be presented with a grade-school math word problem and be asked to solve it.
        Before answering you should reason about the problem (using the "reasoning" field in the JSON response described below).
            
        You will always repond{" with JSON" if is_json else ""} in the format described below:
        
        {EXAMPLE_JSON_STRUCTURE if is_json else EXAMPLE_NL_STRUCTURE}

        The "reasoning" field will contain your reasoning about the sequence of events, and the "answer" will contain the single letter representing the correct choice you are presented with.
        """)
    },]
    
    for i in range(n_shots):
        messages.append({
            "role": "user",
            "content": """Question: {question}""".format(question=example_question[i])
        })
        messages.append({
            "role": "assistant",
            "content": example_answer[i]        
        })
    
    return messages