# from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# messages = [
#     {
#         "role": "user",
#         "content": "What is the capital of France?"
#     },
#     {
#         "role": "assistant",
#         "content": "The capital of France is Paris."
#     }
# ]

# print(tokenizer.apply_chat_template(messages, tokenize=False))

# import json
# a = json.loads(
# "{\"reasoning\": \"The house was bought for 80,000. Then 50,000 was added for repairs. So the total cost is 80,000 + 50,000 = 130,000. The value of the house increased by 150%. 150% of 80,000 is 1.5 x 80,000 = 120,000. So the new value of the house is 80,000 + 120,000 = 200,000. The profit is 200,000 - 130,000 = 70,000.\", \"answer\": \"70,000\"}")
# print(a)

import llama_cpp, os


os.environ['HF_HOME'] = 'cache'

llm = llama_cpp.Llama.from_pretrained(
    repo_id='QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF',
    filename='Meta-Llama-3.1-8B-Instruct.Q6_K.gguf',
    n_gpu_layers=-1,
    n_ctx=4096,
    logits_all=True,
    verbose=False,
    seed=19181111
)

output = llm.create_chat_completion(
    [{
        'role': 'user',
        "content": "What is 2.40 + 1.60?"
        }], 
    temperature=0.2,
    logprobs=1,
    max_tokens=2048,
    top_logprobs=3
)      

# import json
# with open("temp.json", "w") as f:
#     json.dumps(str(output), f, indent=4)

print(output)