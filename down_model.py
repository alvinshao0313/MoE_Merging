from transformers import AutoModelForCausalLM, AutoTokenizer
model_list = [
    "deepseek-ai/deepseek-moe-16b-base",
    "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-ai/DeepSeek-V2",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen2-57B-A14B",
    "mistralai/Mixtral-8x7B-v0.1"
]

for model_name in model_list:
    print(f"Processing model: {model_name}")
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu", trust_remote_code=True
    )
    del model, tokenizer  # Free up memory
    print(f"Model {model_name} loaded successfully.")
