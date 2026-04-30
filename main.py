"""
Main entry point for training the WhatsApp GPT model.

This module defines the high-level Transformer model architectures (NanoGPT and 
DeepNanoGPT) and coordinates the training loop, including data loading, 
batch sampling, loss monitoring, and checkpoint saving.
"""

import numpy as np
import tokenizer
from transformer import TransformerBlock, LayerNorm
from tokenizer import Embedding, encode, vocab_size, get_positional_encoding, int_to_char, decode, char_to_int
from train import Trainer, get_batch
from saving import save_model
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

# Load environment variables for configuration
load_dotenv()
from utils import generate, MAX_SEQ_LEN


# Fall back to NumPy if CuPy is unavailable for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False


class NanoGPT:
    """
    A lightweight, single-block Transformer architecture.

    Useful for testing purposes or for training on very small datasets.

    Attributes:
        vocab_size (int): Size of the character vocabulary.
        seq_len (int): Maximum supported sequence length.
        token_embedding (Embedding): The token embedding layer.
        positional_encoding (ndarray): Precomputed positional encodings.
        transformer_block (TransformerBlock): The core transformer layer.
        ln_f (LayerNorm): Final layer normalization.
        head (ndarray): Final linear output weights.
        ln_f_cache (tuple): Cache for final layer norm backward pass.
        last_x (ndarray): Cached input to the output head.
    """

    def __init__(self, vocab_size, embed_size, num_heads, seq_len):
        """
        Initializes the NanoGPT model.

        Args:
            vocab_size (int): Total unique tokens.
            embed_size (int): Dimensionality of embeddings.
            num_heads (int): Number of attention heads.
            seq_len (int): Maximum context window size.
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        self.token_embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = get_positional_encoding(seq_len, embed_size)
        self.transformer_block = TransformerBlock(embed_size, num_heads)
        
        self.ln_f = LayerNorm(embed_size)
        self.head = cp.random.randn(embed_size, vocab_size) * 0.01

    def forward(self, idx, return_attention=False):
        """
        Performs the forward pass.

        Args:
            idx (ndarray): Input indices of shape (T,) or (1, T).
            return_attention (bool): Whether to return attention weights.

        Returns:
            ndarray or tuple: Predicted logits, and optionally attention weights.
        """
        if isinstance(idx, np.ndarray):
            idx = cp.asarray(idx)
        
        # Ensure input is 1D for this simple version if batch size 1 is used
        if idx.ndim == 2:
            if idx.shape[0] != 1:
                raise ValueError(f"Batch size > 1 not supported in NanoGPT. Got shape: {idx.shape}")
            idx = idx[0]
        T = len(idx)
        
        # Process through layers
        x = self.token_embedding.forward(idx)
        x = x + self.positional_encoding[:T, :]
        
        if return_attention:
            x, attn_weights_out = self.transformer_block.forward(x, return_attention=True)
        else:
            x = self.transformer_block.forward(x)
        
        x, self.ln_f_cache = self.ln_f.forward(x)
        self.last_x = x
        
        # Final linear projection
        logits = x @ self.head
        if return_attention:
            return logits, attn_weights_out
        return logits

    
class DeepNanoGPT:
    """
    A multi-block Transformer architecture.

    This version supports multiple stacked Transformer blocks and batch processing.

    Attributes:
        token_embedding (Embedding): Token embedding layer.
        positional_encoding (ndarray): Precomputed positional encodings.
        blocks (list): List of stacked TransformerBlock instances.
        ln_f (LayerNorm): Final layer normalization.
        head (ndarray): Final linear output weights.
        ln_f_cache (tuple): Cache for final layer norm backward pass.
        last_x (ndarray): Cached input to the output head.
    """

    def __init__(self, vocab_size, embed_size, num_heads, num_blocks, seq_len):
        """
        Initializes the DeepNanoGPT model.

        Args:
            vocab_size (int): Total unique tokens.
            embed_size (int): Dimensionality of embeddings.
            num_heads (int): Number of attention heads.
            num_blocks (int): Number of stacked Transformer blocks.
            seq_len (int): Maximum context window size.
        """
        self.token_embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = get_positional_encoding(seq_len, embed_size)
        self.blocks = [TransformerBlock(embed_size, num_heads) for _ in range(num_blocks)]
        self.ln_f = LayerNorm(embed_size)
        self.head = cp.random.randn(embed_size, vocab_size) * 0.01

    def forward(self, idx, return_attention=False, layer_idx=0):
        """
        Performs the forward pass for a batch of sequences.

        Args:
            idx (ndarray): Input indices of shape (B, T).
            return_attention (bool): Whether to return attention weights.
            layer_idx (int): Which layer's attention weights to return.

        Returns:
            ndarray or tuple: Predicted logits, and optionally attention weights.
        """
        # Move data to GPU if necessary
        if isinstance(idx, np.ndarray):
            idx = cp.asarray(idx)
        
        B, T = idx.shape
        
        # Token embeddings + positional encoding
        # Broadcasting applies PE (T, C) across the batch dimension
        x = self.token_embedding.forward(idx)
        x = x + self.positional_encoding[:T, :]
        
        # Sequentially pass through all Transformer blocks
        attn_weights_out = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == layer_idx:
                x, attn_weights_out = block.forward(x, return_attention=True)
            else:
                x = block.forward(x)
            
        # Final normalization
        x, self.ln_f_cache = self.ln_f.forward(x)
        self.last_x = x
        
        # Output projection: (B, T, embed_size) @ (embed_size, vocab_size) -> (B, T, vocab_size)
        logits = x @ self.head
        
        if return_attention:
            return logits, attn_weights_out
        return logits


import re

def clean_whatsapp(file_path):
    """
    Cleans WhatsApp export files for training.

    Args:
        file_path (str): Path to raw export.

    Returns:
        str: Cleaned text.
    """
    if not file_path or not os.path.exists(file_path):
        return ""
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_data = []
    pattern = r'^(\[?\d{1,4}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s-\s|\[?\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s)'

    for line in lines:
        line = re.sub(pattern, '', line).strip()
        if line.startswith("-:"):
            line = line.replace("-:", "Alperitto:", 1)
        if line and "mesajları ve aramalar uçtan uca" not in line.lower() and "<Medya dahil edilmedi>" not in line:
            if ":" in line:
                cleaned_data.append(line)
            
    return "\n".join(cleaned_data)


# Model Hyperparameters
EMBED_SIZE = 256
NUM_HEADS = 8
NUM_BLOCKS = 4

# Load training data
text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

if __name__ == "__main__":
    print(f"Vocabulary built. Size: {vocab_size}")
    print(f"Sample characters: {chars[:20]}")

    # Initialize mappings
    char_to_int = { ch:i for i,ch in enumerate(chars) }
    int_to_char = { i:ch for i,ch in enumerate(chars) }

    # Set tokenizer globals
    tokenizer.char_to_int = char_to_int
    tokenizer.int_to_char = int_to_char
    tokenizer.vocab_size = vocab_size

    # Convert entire text to index array and move to GPU
    data_indices = [char_to_int[c] for c in text]
    data_indices_gpu = cp.array(data_indices)

    # Instantiate model and trainer
    model = DeepNanoGPT(vocab_size, EMBED_SIZE, NUM_HEADS, NUM_BLOCKS, MAX_SEQ_LEN)
    trainer = Trainer(model, learning_rate=0.0001)

    # Training progress bar
    pbar = tqdm(range(60000), desc="Training", unit="step")

    for step in pbar:
        # Sample a batch of data
        xb, yb = get_batch(data_indices_gpu, seq_len=128, batch_size=96)
        
        # Perform single training step
        loss = trainer.train_step(xb, yb)
        pbar.set_postfix({"loss": f"{loss:.4f}"})
        
        # Periodic logging and generation samples
        if step % 100 == 0:
            pbar.write(f"\nStep {step} | Loss: {loss:.4f}")
            try:
                # Generate a small sample from the model to monitor progress
                sample = generate(model, "Alperitto:", 50)
                pbar.write(f"Sample: {sample}\n")
            except Exception as e:
                pbar.write(f"Generation error: {e}")

        # Save checkpoints every 1000 steps
        if step % 1000 == 0 and step > 0:
            save_model(model, f"./data/checkpoints/model_step_{step}.pkl")
            pbar.write(f"Checkpoint saved: model_step_{step}.pkl")

    save_model(model)
