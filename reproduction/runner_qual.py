from syncode import Syncode
import time

def test_Quality(model, dataset):
    output_schema = '{"reasoning":<reasoning about the answer>, "answer": <final answer>}'
    
    sys_prompt = f"""You are an expert in solving grade school math tasks. You will be presented\
with a grade-school math word problem and be asked to solve it. Before\
answering, you should reason about the problem (using the "reasoning" field\
in the JSON response format described below). Always respond with JSON\
in the format: {output_schema}. The "reasoning" field contains your logical explanation, and\
the "answer" field contains the final numeric result."""

    syn_llm = Syncode(model=model, grammar='json', max_new_tokens=400)
    for i, question in dataset:
        question_prompt = question
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': question_prompt}
        ]
        
        out = syn_llm.infer(messages)[0]
        # Parse output to json
        
