import torch
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

# 读取数据
print("Loading dataset...")
df = pd.read_json('/root/autodl-tmp/nq_train.json')
ds = Dataset.from_pandas(df)

# 加载Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 作为 pad_token

# 处理函数
def process_func(example):
    MAX_LENGTH = 384  # 设置最大长度，保证数据完整性
    input_ids, attention_mask, labels = [], [], []

    # 对输入和输出进行 tokenization
    instruction = tokenizer(f"user\n\n{example['instruction'] + example['input']}assistant\n\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    # 拼接输入和输出，并添加 pad_token_id
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 处理数据集
print("Tokenizing dataset...")
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# 加载模型
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct',
                                             device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，需要执行此方法

# LoRA 配置
print("Configuring LoRA...")
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=16,  # 调整 LoRA 秩
    lora_alpha=32,  # 调整 LoRA alpha
    lora_dropout=0.1  # Dropout 比例
)

# 获取 PEFT 模型
print("Integrating LoRA with model...")
model = get_peft_model(model, config)

# 训练参数设置
print("Setting up training arguments...")
args = TrainingArguments(
    output_dir="./output/llama3",
    per_device_train_batch_size=8,  # 调整批量大小
    gradient_accumulation_steps=2,  # 调整梯度累积步数
    logging_steps=50,  # 记录日志的步数，适当增加
    num_train_epochs=3,  # 训练轮数，可以根据需要调整
    save_steps=500,  # 调整保存步数
    learning_rate=3e-5,  # 调整学习率
    save_on_each_node=True,
    gradient_checkpointing=True  # 可以选择开启或关闭
)

# Trainer 设置
print("Setting up trainer...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
print("Starting training...")
trainer.train()

# 保存模型和 tokenizer
print("Saving model and tokenizer...")
peft_model_id = "./llama3_lora"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

print("Done.")
