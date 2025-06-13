from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import datasets

ds = datasets.load_dataset("json", data_files="quotes.jsonl", split="train")

model_name = "./checkpoints"          
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=128)
tokenized = ds.map(tokenize_fn, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="checkpoints-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=5e-5,
    fp16=False,
    save_total_limit=2,
    logging_steps=50,
    save_steps=200,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("checkpoints-finetuned")
tokenizer.save_pretrained("checkpoints-finetuned")

