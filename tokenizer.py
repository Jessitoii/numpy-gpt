"""
Text preprocessing and tokenization utilities.

This module handles reading and cleaning WhatsApp chat data, converting
text to integer sequences (and vice versa), and implementing neural
embedding layers with positional encoding.
"""

import numpy as np
import re
import os
from dotenv import load_dotenv

# Load environment variables (e.g., path to the dataset)
load_dotenv()

def clean_whatsapp(file_path):
    """
    Cleans a WhatsApp export text file by removing timestamps and system messages.

    Args:
        file_path (str): Path to the raw WhatsApp .txt export.

    Returns:
        str: A cleaned string containing only 'User: Message' lines.
    """
    if not file_path or not os.path.exists(file_path):
        return ""
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_data = []
    # Matches common WhatsApp timestamp formats across regions
    # e.g. "14.06.2025 00:28 - ", "[14/6 00:25] ", "14/06/25, 00:28 - "
    pattern = r'^(\[?\d{1,4}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s-\s|\[?\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s)'

    for line in lines:
        # Strip the timestamp
        line = re.sub(pattern, '', line).strip()
        
        # Specific cleanup for a common user placeholder if present
        if line.startswith("-:"):
            line = line.replace("-:", "Alperitto:", 1)
            
        # Filter out system messages and media placeholders
        if line and "mesajları ve aramalar uçtan uca" not in line.lower() and "<Medya dahil edilmedi>" not in line:
            # Only keep lines that follow the 'Sender: Message' format
            if ":" in line:
                cleaned_data.append(line)
            
    return "\n".join(cleaned_data)


# Fall back to NumPy if CuPy is unavailable for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
    print("✓ CuPy available (GPU acceleration enabled)")
except (ImportError, Exception) as e:
    cp = np
    HAS_GPU = False
    print(f"✗ CuPy not available ({str(e)[:50]}...), falling back to NumPy")

# Load and process the text data for vocabulary building
text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))

# Unique characters in the dataset form our vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping between characters and integers
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    """
    Converts a string into a list of integer indices.

    Args:
        s (str): The input string.

    Returns:
        list: A list of integers representing the string.
    """
    return [char_to_int[c] for c in s]

def decode(l):
    """
    Converts a list of integer indices back into a string.

    Args:
        l (list): The list of integer indices.

    Returns:
        str: The decoded string.
    """
    return ''.join([int_to_char[i] for i in l])


class Embedding:
    """
    Learnable Embedding layer.

    Maps discrete token indices to continuous dense vectors.

    Attributes:
        weights (ndarray): The learnable embedding matrix.
        input_indices (ndarray): Cached input indices for backward pass.
    """

    def __init__(self, vocab_size, embed_size):
        """
        Initializes the Embedding layer.

        Args:
            vocab_size (int): Total number of unique tokens in the vocabulary.
            embed_size (int): Dimensionality of token embeddings.
        """
        # Random initialization with small values
        self.weights = cp.random.randn(vocab_size, embed_size) * 0.01
        self.input_indices = None

    def forward(self, x):
        """
        Performs the forward lookup.

        Args:
            x (ndarray): Array of token indices.

        Returns:
            ndarray: The corresponding embedding vectors.
        """
        self.input_indices = x
        return self.weights[x]

    def backward(self, d_out):
        """
        Calculates gradients for the embedding weights.

        Args:
            d_out (ndarray): Upstream gradient.

        Returns:
            ndarray: Gradient with respect to embedding weights.
        """
        d_weights = cp.zeros_like(self.weights)
        # Use atomic add to accumulate gradients for repeated indices in the input
        cp.add.at(d_weights, self.input_indices, d_out)
        return d_weights


def get_positional_encoding(seq_len, embed_size):
    """
    Sinusoidal positional encoding.

    Provides the model with information about the relative position of tokens 
    in the sequence, as described in 'Attention Is All You Need'.

    Args:
        seq_len (int): Maximum sequence length to encode.
        embed_size (int): Embedding dimensionality.

    Returns:
        ndarray: A matrix of shape (seq_len, embed_size) containing encodings.
    """
    pe = cp.zeros((seq_len, embed_size))
    
    position = cp.arange(seq_len)[:, cp.newaxis]
    
    # Compute the division term using frequencies for different dimensions
    # Log space calculation is used for better numerical precision
    div_term = cp.exp(cp.arange(0, embed_size, 2) * -(cp.log(10000.0) / embed_size))
    
    # Apply sine to even indices and cosine to odd indices
    pe[:, 0::2] = cp.sin(position * div_term)
    pe[:, 1::2] = cp.cos(position * div_term)
    
    return pe

if __name__ == "__main__":
    # Test script for tokenization and embedding
    SEQ_LEN = 10
    EMBED_SIZE = 16

    # Initialize layer
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)

    # Encode a test word
    input_data = np.array(encode("kodlama"))
    embedded_output = embedding_layer.forward(input_data)

    print(f"Input shape:    {input_data.shape}")
    print(f"Embedded shape: {embedded_output.shape}")

    # Generate positional encodings
    pe_matrix = get_positional_encoding(SEQ_LEN, EMBED_SIZE)
    print(f"PE matrix shape: {pe_matrix.shape}")

    # Combine token embeddings with positional encodings
    current_seq_len = embedded_output.shape[0]
    pe_part = pe_matrix[:current_seq_len, :]

    final_input = embedded_output + pe_part
    print(f"Final vector shape: {final_input.shape}")
    print("Sample (first 2 tokens):")
    print(final_input[:2])