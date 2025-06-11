# QSTGeneration

This project implements basic text generation using a pretrained GPT-2 language model. It demonstrates how to tokenize input prompts, generate continuations, and decode model output using the Hugging Face Transformers library.

## 🔧 Features

- Loads GPT-2 with `AutoModelForCausalLM` and `AutoTokenizer`
- Generates text from any input prompt
- Lays the foundation for fine-tuning on custom datasets
- Easy to expand into Quantized Side Tuning (QST) for generative tasks
