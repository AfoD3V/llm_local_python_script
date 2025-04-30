from transformers import pipeline
import torch

# Use the pipeline for text generation with TinyLlama model
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Create a list of messages in the chat format
messages = [
    {
        "role": "system",
        "content": "You are a technical expert in ethical hacking.",
    },
    {
        "role": "user",
        "content": "Write a roadmap for learning ethical hacking for complete beginner.",
    },
]

# Tokenize the input using the chat template
prompt = pipe.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Generate the response
outputs = pipe(
    prompt, max_new_tokens=5000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
)

# Print the generated response
print(outputs[0]["generated_text"])
