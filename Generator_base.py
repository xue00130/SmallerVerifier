import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM






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









###################################################################
##训练一个base generator
###################################################################



from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import accelerate
# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    save_steps=1000,
    logging_dir="./logs",
    fp16=True,
)

# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()



# 保存模型
model.save_pretrained("./Math_model")




