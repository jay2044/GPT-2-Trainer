import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os


def load_model(model_path, device):
    """
    Load the GPT-2 model and tokenizer from the specified path and move the model to the designated device.
    """
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()  # setting the model to evaluation mode
    return tokenizer, model


def generate_response(tokenizer, model, prompt, device, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generate a response from the model based on the input prompt.
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode  generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract the response
    response = generated_text[len(prompt):].strip()
    return response


def main():
    """
    Main function to run the chat interface.
    """
    # Path to your fine-tuned model
    model_path = 'your model path'

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU.")

    print("Loading model and tokenizer...")
    tokenizer, model = load_model(model_path, device)
    print("Model and tokenizer loaded successfully!\n")

    print("Start chatting with the GPT-2 Slang model! (Type 'exit', 'quit', or 'bye' to stop)\n")

    # Conversation loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chat ended. Goodbye!")
            break
        # prepare the prompt and generate response
        prompt = user_input
        response = generate_response(tokenizer, model, prompt, device)
        print(f"GPT-2 Slang: {response}\n")


if __name__ == "__main__":
    main()
