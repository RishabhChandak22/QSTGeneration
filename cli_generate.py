import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Prompt to generate from")
    parser.add_argument("--max_length", type=int, default=100)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained("./checkpoints")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=args.max_length, do_sample=True)
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Generated Text:\n", text)

if __name__ == "__main__":
    main()

with open("output_log.txt", "a") as f:
    f.write(f"Prompt: {prompt}\nGenerated: {generated_text}\n\n")
