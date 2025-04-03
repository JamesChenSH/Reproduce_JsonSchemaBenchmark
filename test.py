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

import json
a = json.loads("{\"reasoning\": \"Ryan earns $6 per week for 3 weeks: 6 * 3 = 18. He spent $1.25 on ice cream cones for himself and 3 friends: 1.25 * 4 = 5. Therefore, he has 18 - 5 = 13 left. Each movie ticket costs $6.50, so he can buy 13 / 6.50 = 2 tickets.\", \"answer\": 2}")
print(a)