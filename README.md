Notes: 

For output json files recording the incorrect responses, *_NL/JSON.out suggests whether the few-shot examples are provided with Natural Language or JSON. All wrappers are generating outputs from json schema thus outputting jsons.

However, vanilla LLM has one experiment that generates Natural Language outputs

### To Run Guidance Wrapper with customized model for quality experiment:
```Bash
python ./runner_quality.py --wrapper guidance --model QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF//Meta-Llama-3.1-8B-Instruct.Q6_K.gguf --is_cpp --json_shots --n_exp 3
```

Alternatively, DeepSeek Model path is here:
```Bash
unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF//DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf
```