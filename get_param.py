import transformers

model_name = "deepseek-ai/deepseek-moe-16b-base"

model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 打印模型的参数到.txt文件
with open("dpsk_v1_parameters.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(f"Parameter: {name}, Size: {param.size()}\n")
