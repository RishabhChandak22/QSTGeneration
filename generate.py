from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./checkpoints"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Load GPT2 tokenizer

prompt = input("Enter a prompt: ")
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=60,
    do_sample=True,
    temperature=0.5,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2
)

print("\nGenerated Text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
