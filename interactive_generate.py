from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./checkpoints")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

print("Welcome to interactive generation!")
print("Leave any setting blank to use the default.\n")

while True:
    prompt = input("\nEnter a prompt (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        break

    try:
        temperature = float(input("Temperature (default 0.7): ") or 0.7)
        top_k = int(input("Top-k (default 50): ") or 50)
        top_p = float(input("Top-p (default 0.95): ") or 0.95)
        max_length = int(input("Max length (default 60): ") or 60)
    except ValueError:
        print("Invalid input. Using default settings.")
        temperature, top_k, top_p, max_length = 0.7, 50, 0.95, 60

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True
    )

    print("\nGenerated Text:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
