from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

model_name = "gpt2"
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset["train"] = dataset["train"].select(range(200))  # or even 50
dataset["validation"] = dataset["validation"].select(range(50))


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], return_special_tokens_mask=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
block_size = 128

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=1,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
)

trainer.train()
trainer.save_model("./checkpoints")
