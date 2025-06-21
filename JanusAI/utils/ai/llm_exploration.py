# JanusAI/utils/ai/llm_exploration.py

from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(prompt: str) -> str:
    """
    Generates a response from a pre-trained LLM given a prompt.

    Args:
        prompt: The input text to the LLM.

    Returns:
        The generated response from the LLM.
    """
    # 1. Load the pre-trained model and tokenizer
    # We're using "distilgpt2", a smaller and faster version of GPT-2,
    # which is great for getting started.
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. Prepare the input
    # The tokenizer converts the prompt into a format that the model can understand.
    inputs = tokenizer(prompt, return_tensors="pt")

    # 3. Generate the response
    # The model generates a response based on the input.
    # We're setting a max_length to prevent overly long responses.
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)

    # 4. Decode the output
    # The tokenizer decodes the model's output back into human-readable text.
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

if __name__ == '__main__':
    # Example usage of the generate_response function
    test_prompt = "In the field of physics, the most important discovery of the 20th century was"
    generated_text = generate_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Generated Text: {generated_text}")
