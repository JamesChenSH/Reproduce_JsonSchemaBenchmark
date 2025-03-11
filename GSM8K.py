# %% [markdown]
# ## GSM8K
# 
# This notebook reproduces the results for the GSM8K evaluations

# %%
import json, os
os.environ['HF_HOME'] = './cache/'
import outlines
import torch
from transformers import AutoTokenizer
from textwrap import dedent
from datasets import load_dataset
import re
from outlines.samplers import greedy

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
# Load the dataset from HuggingFace
dataset = load_dataset("gsm8k", "main")
# You can inspect the dataset structure
print(dataset)

# %%
dataset['train']

# %%
all_evals = list(dataset['test'])

# %%
model = outlines.models.transformers(
    MODEL_NAME,
    device='cuda',
    model_kwargs={
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True
    })

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME)

# %%
example_question = [
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
]

# %%
example_response = [
    """{"reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.", "answer": 6}""",
    """{"reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.", "answer": 5}""",
    """{"reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.","answer": 39"""
]


# %%
def create_prompt(question, tokenizer):
    messages = [
        {
            "role": "system",
            "content": dedent("""
            You are an expert in solving grade school math tasks. You will be presented with a grade-school math word problem and be asked to solve it.
            Before answering you should reason about the problem (using the "reasoning" field in the JSON response described below).
              
            You will always repond with JSON in the format described below:
            
            {"reasoning": <reasoning about the answer>, "answer": <final answer>}

            The "reasoning" field will contain your reasoning about the sequence of events, and the "answer" will contain the single letter representing the correct choice you are presented with.
            """)
        },]
    for i in range(len(example_question)):
        messages.append(
        {
            "role": "user",
            "content": """Question: {question}""".format(question=example_question[i])
        }
        )
        messages.append(
        {
            "role": "assistant",
            "content": example_response[i]        
        })
    messages.append(
        {
            "role": "user",
            "content": """Question: {question}", """.format(question=question)
        })
    messages.append(
        {
            "role": "assistant",
            "content": ""
        }
    )
    return tokenizer.apply_chat_template(messages, tokenize=False)

print(create_prompt(all_evals[5]['question'], tokenizer))

# %%
from pydantic import BaseModel, Field, constr
from outlines.fsm.json_schema import build_regex_from_schema

class Response(BaseModel):
    reasoning: constr(max_length=1000)
    answer: int = Field(pattern=r'[1-9][0-9]{0,9}')


schema_regex = build_regex_from_schema(Response.schema_json())

# %%
all_evals[5]['question']

# %%
open_generator = outlines.generate.text(model, sampler=greedy())

# %%
open_generator(create_prompt(all_evals[5]['question'], tokenizer),max_tokens=256)

# %%
re.search(schema_regex, create_prompt(all_evals[5]['question'], tokenizer))

# %%
## Unstructured Generation

# %%
LAST = len(all_evals)
answer_regex = r'"answer":[ ]?([1-9][0-9]{0,9})'
answers = []
for ex_eval in all_evals[0:LAST]:
    raw_int = ex_eval['answer'].split('#### ')[1]
    raw_int = re.sub(",","",raw_int)
    answers.append(int(raw_int))

# %%
free_resp = [open_generator(create_prompt(all_evals[i]['question'], tokenizer), max_tokens=256) for i in range(LAST)]

# %%
free_resp[3]

# %%
free_resp_answers = [int(result[1].upper()) if result else "" for result in [re.search(answer_regex,resp) for resp in free_resp]]

# %%
import numpy as np
np.mean([result[0] == result[1] for result in zip(free_resp_answers, answers)])

# %% [markdown]
# ## Structured Generation

# %%
structured_generator = outlines.generate.regex(model, schema_regex, sampler=greedy())

# %%
structured_resp = [structured_generator(create_prompt(all_evals[i]['question'], tokenizer)) for i in range(LAST)]

# %%
structured_resp[3]

# %%
structured_resp_answers = [int(result[1].upper()) if result else "" for result in [re.search(answer_regex,resp) for resp in structured_resp]]

# %%
import numpy as np
np.mean([result[0] == result[1] for result in zip(structured_resp_answers, answers)])

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,figsize=(10,8),facecolor='white')
ax.bar('unstructured',np.mean([result[0] == result[1] for result in zip(free_resp_answers, answers)]),label='unstructured')
ax.bar('structured',np.mean([result[0] == result[1] for result in zip(structured_resp_answers, answers)]),label='structured')
ax.legend(loc="lower right")
ax.set_title(f"GSM8K - Unstructured vs. JSON Structured\n{MODEL_NAME}")

# %%



