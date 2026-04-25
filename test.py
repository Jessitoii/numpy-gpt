"""
Interactive test script for the trained WhatsApp GPT model.

This module provides a command-line interface to interact with a trained
model, allowing users to enter prompts and see generated responses.
"""

import cupy as cp
import pickle
import time
from main import DeepNanoGPT, EMBED_SIZE, NUM_HEADS, NUM_BLOCKS, clean_whatsapp
from utils import generate, MAX_SEQ_LEN
from saving import load_model
from dotenv import load_dotenv
import os

# Load environment configuration
load_dotenv()

# Preprocess vocabulary from the same dataset used in training
text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character mapping dictionaries
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }


def generate_response(model, start_str, length=150, temperature=0.8):
    """
    Predicts and generates a response sequence from a prompt.

    Args:
        model (DeepNanoGPT): The loaded Transformer model.
        start_str (str): The starting prompt text.
        length (int): Total number of characters to generate. Defaults to 150.
        temperature (float): Sampling temperature. Higher means more random.
                             Defaults to 0.8.

    Returns:
        str: The full generated sequence.
    """
    # Convert input characters to integer IDs, skipping unknown chars
    idx = cp.array([char_to_int[c] for c in start_str if c in char_to_int])
    idx = idx[cp.newaxis, :]  # (1, T) - Add batch dimension
    
    generated = start_str
    
    for _ in range(length):
        # Limit the context window to the model's supported MAX_SEQ_LEN
        idx_cond = idx[:, -MAX_SEQ_LEN:]
        logits = model.forward(idx_cond)
        
        # Focus on the last token's predictions and apply temperature scaling
        logits = logits[0, -1, :] / temperature
        
        # Softmax normalization with numerical stability
        probs = cp.exp(logits - cp.max(logits))
        probs /= cp.sum(probs)
        
        # Probabilistic sampling of the next character
        next_idx_raw = cp.random.choice(len(chars), size=1, p=probs)
        next_idx = int(next_idx_raw.item())

        # Decode the integer back to a character
        if next_idx in int_to_char:
            generated += int_to_char[next_idx]
        else:
            generated += " "
        
    return generated


if __name__ == "__main__":
    # Initialize the model with the same hyperparameters as training
    model = DeepNanoGPT(
        vocab_size=len(chars),
        embed_size=256,
        num_heads=8,
        num_blocks=4,
        seq_len=128
    )
    
    # Load the serialized weights
    load_model(model, "./data/final_model/whatsapp_gpt.pkl")

    print("\n" + "="*40)
    print("WhatsApp GPT — Interactive Mode")
    print("Type 'exit' to quit.")
    print("="*40 + "\n")

    # Main interactive loop
    while True:
        user_input = input("Prompt: ")
        if user_input.lower() == 'exit':
            break
        
        # Default prompt if empty
        if not user_input:
            user_input = "Hello "

        print("\nGenerating...\n")
        
        # Generate the text using the utility function
        response = generate(model, start_str=user_input, length=100, temperature=1.0)
        
        print("-" * 40)
        print(response)
        print("-" * 40 + "\n")