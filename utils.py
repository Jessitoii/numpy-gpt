"""
Utility functions for Transformer model interaction.

This module contains helper functions for high-level tasks like text generation
from a trained Transformer model.
"""

import numpy as np
from tokenizer import char_to_int, int_to_char, chars

# Fall back to NumPy if CuPy is unavailable for GPU acceleration
try:
    import cupy as cp
except (ImportError, Exception):
    cp = np

# Maximum sequence length supported by the model's positional encodings
MAX_SEQ_LEN = 128

def generate(model, start_str, length=100, temperature=1.0):
    """
    Generates a sequence of characters from the model starting with a given string.

    This function uses autoregressive decoding: it predicts the next character,
    appends it to the input, and repeats the process.

    Args:
        model (Transformer): The trained Transformer model.
        start_str (str): The initial string to start generation from.
        length (int): Number of characters to generate. Defaults to 100.
        temperature (float): Controls randomness (higher = more random). 
                             Defaults to 1.0.

    Returns:
        str: The complete generated string including the start_str.
    """
    # Convert start string to integer indices
    idx = cp.array([char_to_int[c] for c in start_str])
    idx = idx[cp.newaxis, :]  # Add batch dimension: (1, T)

    generated_str = start_str
    
    for _ in range(length):
        # Crop context to the maximum supported sequence length
        idx_cond = idx[:, -MAX_SEQ_LEN:]
        
        # Get predictions for the next token
        logits = model.forward(idx_cond)  # (1, T, V)
        
        # Focus on the logits of the last predicted token and apply temperature scaling
        # (1, T, V) -> (V,)
        logits = logits[0, -1, :] / temperature  
        
        # Apply softmax with numerical stability (max subtraction)
        probs = cp.exp(logits - cp.max(logits))
        probs /= cp.sum(probs)
        
        # Sample the next character based on the probability distribution
        next_idx = int(cp.random.choice(len(chars), size=1, p=probs))
        
        # Update the sequence of indices for the next iteration
        next_idx_arr = cp.array([[next_idx]])
        idx = cp.concatenate((idx, next_idx_arr), axis=1)
        
        # Append the decoded character to the result string
        generated_str += int_to_char[int(next_idx)]
        
    return generated_str
