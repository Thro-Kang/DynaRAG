from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model

def initialize_model(model_path, lora_path):
    print("Initializing LoRA configuration...")
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # 调整 LoRA 秩
        lora_alpha=32,  # 调整 LoRA alpha
        lora_dropout=0.1  # Dropout 比例
    )

    print("Loading tokenizer...")
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

    print("Loading LoRA weights...")
    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

    print("Model and tokenizer initialized.")
    return tokenizer, model

def generate_response(model, tokenizer, question):
    print("Constructing prompt...")
    prompt = f"请判断你是否可以独立回答这个问题，只回答“True”或“False”：{question}"

    print("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    print("Generating response...")
    generated_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.5,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    print("Decoding response...")
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return response

def evaluate_response(response):
    if "True" in response:
        return "True"
    elif "False" in response:
        return "False"
    else:
        return "无法确定"

# 初始化模型和tokenizer
model_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct'
lora_path = './llama3_lora'
print("Initializing model and tokenizer...")
tokenizer, model = initialize_model(model_path, lora_path)

# 生成并打印响应
question = "what is the predominant religion in the ukraine"
print("Generating response for the question...")
response = generate_response(model, tokenizer, question)
print(f"Raw Response: {response}")

# 判断返回结果中是否有True或False
final_result = evaluate_response(response)
print(f"Final Result: {final_result}")
