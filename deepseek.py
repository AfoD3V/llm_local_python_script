from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Specify the model name from Hugging Face repository
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"


# Load the tokenizer - a tool for processing text into tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use lower precision to save memory
    device_map="auto",  # Automatically assign to available GPU or CPU
)

# Create a prompt
prompt = "Write a roadmap for learning ethical hacking, for each big step create 3 substeps, for next 3 substeps create another 3 substep until particular substep is atomic so it cannot be divided anymore. Use markdown syntax for formatting. Use only one code block for each substep. Do not use any other formatting"

# Tokenize the prompt - convert text into the "language of the model"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate the response
with (
    torch.no_grad()
):  # This disables gradient calculation, which we don't need for generation
    outputs = model.generate(
        inputs.input_ids,
        max_length=2000,  # Maximum response length (in tokens)
        temperature=0.1,  # Temperature - higher gives more creative responses
        do_sample=False,  # Use sampling instead of greedy selection of the best token
        top_p=0.95,  # Nucleus sampling - choose from the most probable tokens
    )

# Decode the tokens back to text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display the response
print(response)
