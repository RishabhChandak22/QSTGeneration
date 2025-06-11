from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("./checkpoints")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model.eval()

prompt = "Who is the greatest baseball player of all time?"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
