import numpy as np
import re
import os
from dotenv import load_dotenv
load_dotenv()

def clean_whatsapp(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_data = []
    # Matches common WhatsApp timestamp formats across regions
    # e.g. "14.06.2025 00:28 - ", "[14/6 00:25] ", "14/06/25, 00:28 - "
    pattern = r'^(\[?\d{1,4}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s-\s|\[?\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s)'

    for line in lines:
        line = re.sub(pattern, '', line).strip()
        if line.startswith("-:"):
            line = line.replace("-:", "Alperitto:", 1)
        if line and "mesajları ve aramalar uçtan uca" not in line.lower() and "<Medya dahil edilmedi>" not in line:
            if ":" in line:
                cleaned_data.append(line)
            
    return "\n".join(cleaned_data)

# Fall back to NumPy if CuPy is unavailable
try:
    import cupy as cp
    HAS_GPU = True
    print("✓ CuPy available (GPU acceleration enabled)")
except (ImportError, Exception) as e:
    cp = np
    HAS_GPU = False
    print(f"✗ CuPy not available ({str(e)[:50]}...), falling back to NumPy")

text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [char_to_int[c] for c in s]

def decode(l):
    return ''.join([int_to_char[i] for i in l])


class Embedding:
    def __init__(self, vocab_size, embed_size):
        """
        vocab_size: Total number of unique tokens in the vocabulary
        embed_size: Dimensionality of token embeddings
        """
        self.weights = cp.random.randn(vocab_size, embed_size) * 0.01
        self.input_indices = None

    def forward(self, x):
        self.input_indices = x
        return self.weights[x]

    def backward(self, d_out):
        d_weights = cp.zeros_like(self.weights)
        # Accumulate gradients for repeated indices
        cp.add.at(d_weights, self.input_indices, d_out)
        return d_weights


def get_positional_encoding(seq_len, embed_size):
    """
    Sinusoidal positional encoding as described in 'Attention Is All You Need'.

    seq_len:    Maximum sequence length
    embed_size: Embedding dimensionality
    """
    pe = cp.zeros((seq_len, embed_size))
    
    position = cp.arange(seq_len)[:, cp.newaxis]
    
    # Compute the division term in log space for numerical stability
    div_term = cp.exp(cp.arange(0, embed_size, 2) * -(cp.log(10000.0) / embed_size))
    
    pe[:, 0::2] = cp.sin(position * div_term)
    pe[:, 1::2] = cp.cos(position * div_term)
    
    return pe

if __name__ == "__main__":
    SEQ_LEN = 10
    EMBED_SIZE = 16

    embedding_layer = Embedding(vocab_size, EMBED_SIZE)

    input_data = np.array(encode("kodlama"))
    embedded_output = embedding_layer.forward(input_data)

    print(f"Input shape:    {input_data.shape}")
    print(f"Embedded shape: {embedded_output.shape}")

    pe_matrix = get_positional_encoding(SEQ_LEN, EMBED_SIZE)
    print(f"PE matrix shape: {pe_matrix.shape}")

    current_seq_len = embedded_output.shape[0]
    pe_part = pe_matrix[:current_seq_len, :]

    final_input = embedded_output + pe_part
    print(f"Final vector shape: {final_input.shape}")
    print("Sample (first 2 tokens):")
    print(final_input[:2])