import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments

import wandb




# 初始化 wandb
wandb.init(project="MetaMath_Finetuning", name="SFT_Finetuning")






###################################################################
##加载模型
###################################################################

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")









###################################################################
##处理训练数据
###################################################################

from datasets import load_dataset
# 加载数据集
dataset = load_dataset("meta-math/MetaMathQA")


# # 数据格式化
# def format_example(example):
#     input_text = f"Question: {example['query']}\nAnswer:"
#     target_text = example["response"]
#     return {"input_text": input_text, "target_text": target_text}

# formatted_dataset = dataset.map(format_example)

# # 分词
# def tokenize_function(examples):
#     inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
#     outputs = tokenizer(examples["target_text"], max_length=512, truncation=True, padding="max_length")
#     inputs["labels"] = outputs["input_ids"]
#     # 设置 input_ids 中非 label 部分为 -100，以避免计算 loss
#     labels = inputs["labels"]
#     labels = [[-100 if idx < len(inputs["input_ids"][i]) else token for idx, token in enumerate(label)] for i, label in enumerate(labels)]
#     inputs["labels"] = labels
#     return inputs

# tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, batch_size=1000)



# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

# EOS_TOKEN = tokenizer.eos_token # 必须添加EOS_TOKEN


# 数据格式化
def format_example(example):
    input_text = f"Question: {example['query']}\nAnswer:"
    target_text = example["response"]
    return {"input_text": input_text, "target_text": target_text}

formatted_dataset = dataset.map(format_example)

# 分词
def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
    outputs = tokenizer(examples["target_text"], max_length=512, truncation=True, padding="max_length")
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)


tokenized_dataset = tokenized_dataset["train"]


print("11111111111111111")









###################################################################
##训练一个base generator
###################################################################



from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import accelerate

from transformers import TrainerCallback, Trainer, AdamW

from trl import SFTTrainer


# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=4,  # 减小批量大小以节省显存
#     per_device_eval_batch_size=4,
#     gradient_accumulation_steps=4,  # 梯度累积
#     num_train_epochs=3,
#     remove_unused_columns=False,
#     learning_rate=2e-5,  # 设置学习率
#     weight_decay=0.01,  # 权重衰减
#     warmup_ratio=0.1,  # 学习率预热
#     save_strategy="steps",
#     save_steps=1000,
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=10,
#     report_to="wandb",  # 将日志报告到 wandb
#     fp16=True,  # 混合精度训练
#     gradient_checkpointing=True,  # 启用梯度检查点以节省显存
# )


# # 定义优化器
# optimizer = AdamW(model.parameters(), lr=2e-5)

# # 定义 Trainer
# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     dataset_text_field = "text",    
#     tokenizer=tokenizer,
#     optimizers=(optimizer, None),  # 自定义优化器
# )

from trl import SFTTrainer
from transformers import TrainingArguments



# 确保数据集包含正确的分片

train_data = tokenized_dataset.select(range(int(0.9 * len(tokenized_dataset))))
eval_data = tokenized_dataset.select(range(int(0.9 * len(tokenized_dataset)), len(tokenized_dataset)))


args = TrainingArguments(
    output_dir='checkpoints_hf_sft',
    overwrite_output_dir=True, 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=True,
    torch_compile=True,
    evaluation_strategy='steps',
    prediction_loss_only=True,
    eval_accumulation_steps=1,
    learning_rate=0.00006,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.95,
    warmup_steps=1000,
    eval_steps=4000,
    save_steps=4000,
    save_total_limit=4,
    dataloader_num_workers=4,
    max_steps=12000,
    optim='adamw_torch_fused')



trainer = SFTTrainer(
    model,
    args = args,
    train_dataset=train_data,
    eval_dataset= eval_data,
    tokenizer=tokenizer,
)



# 开始训练
trainer.train(resume_from_checkpoint=False)



# 保存模型
model.save_pretrained("./Math_model")

# 结束 wandb 监控
wandb.finish()


