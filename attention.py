"""
Attention mechanisms for Transformer models.

This module provides implementations of standard Self-Attention and
Masked Self-Attention layers using NumPy (or CuPy for GPU acceleration).
"""

import numpy as np
from tokenizer import Embedding, encode, get_positional_encoding, vocab_size
import math

# Fall back to NumPy if CuPy is unavailable for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False

class SelfAttention:
    """
    Standard Self-Attention mechanism.

    This layer computes scaled dot-product attention as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Attributes:
        embed_size (int): The dimensionality of the embedding space.
        Wq (ndarray): Query projection weight matrix.
        Wk (ndarray): Key projection weight matrix.
        Wv (ndarray): Value projection weight matrix.
    """

    def __init__(self, embed_size):
        """
        Initializes the SelfAttention layer with random weights.

        Args:
            embed_size (int): The dimensionality of the input embeddings.
        """
        self.embed_size = embed_size
        
        # Weight matrices for Q, K, V projections — shape: (embed_size, embed_size)
        # Using a small standard deviation (0.01) for weight initialization
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        """
        Computes the softmax of the input array for numerical stability.

        Args:
            x (ndarray): The input scores to normalize.

        Returns:
            ndarray: The probability distribution over the last axis.
        """
        # Row-wise softmax with max subtraction for numerical stability
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        """
        Performs the forward pass of the self-attention layer.

        Args:
            x (ndarray): Input tensor of shape (seq_len, embed_size).

        Returns:
            tuple: A tuple containing:
                - output (ndarray): The context-aware embeddings (seq_len, embed_size).
                - attention_weights (ndarray): The attention scores (seq_len, seq_len).
        """
        # Project input into Q, K, V spaces
        # x: (N, D), W: (D, D) -> output: (N, D)
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # Scaled dot-product attention scores
        # Formula: Softmax( (Q @ K.T) / sqrt(d_k) ) @ V
        d_k = self.embed_size
        scores = (Q @ K.T) / cp.sqrt(d_k)

        # Attention weights via softmax normalization
        attention_weights = self.softmax(scores)

        # Compute the final context-aware output (weighted sum of values)
        output = attention_weights @ V

        return output, attention_weights
    

class MaskedSelfAttention:
    """
    Masked Self-Attention mechanism for decoder blocks.

    Prevents positions from attending to subsequent positions, ensuring
    predictions for position i can depend only on known outputs at positions < i.

    Attributes:
        embed_size (int): The dimensionality of the embedding space.
        Wq (ndarray): Query projection weight matrix.
        Wk (ndarray): Key projection weight matrix.
        Wv (ndarray): Value projection weight matrix.
    """

    def __init__(self, embed_size):
        """
        Initializes the MaskedSelfAttention layer with random weights.

        Args:
            embed_size (int): The dimensionality of the input embeddings.
        """
        self.embed_size = embed_size
        
        # Initialize projection matrices
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        """
        Computes the softmax of the input array for numerical stability.

        Args:
            x (ndarray): The input scores to normalize.

        Returns:
            ndarray: The probability distribution over the last axis.
        """
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        """
        Performs the forward pass with a causal mask.

        Args:
            x (ndarray): Input tensor of shape (seq_len, embed_size).

        Returns:
            tuple: A tuple containing:
                - output (ndarray): The context-aware embeddings (seq_len, embed_size).
                - attention_weights (ndarray): The attention scores (seq_len, seq_len).
        """
        seq_len, embed_size = x.shape
        
        # Project to Q, K, V
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # Compute raw attention scores
        scores = (Q @ K.T) / cp.sqrt(embed_size)

        # Causal mask: prevent attention to future positions
        # Create a lower triangular matrix of ones
        mask = cp.tril(cp.ones((seq_len, seq_len)))
        # Replace 0s with -inf so they become 0 after softmax
        scores = cp.where(mask == 0, -cp.inf, scores)

        # Masked positions become 0 after softmax
        attention_weights = self.softmax(scores)

        # Final weighted aggregation
        output = attention_weights @ V

        return output, attention_weights

if __name__ == "__main__":
    # Example usage for testing purposes
    EMBED_SIZE = 16
    sa = SelfAttention(EMBED_SIZE)

    # Encode a sample word
    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    # Apply positional encoding
    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)
    final_input = embedded_output + pe_matrix

    # Forward pass through Self-Attention
    output, weights = sa.forward(final_input)

    print(f"Attention output shape: {output.shape}")  # (7, 16)
    print(f"Attention distribution for first token:\n{weights[0]}")

    # Forward pass through Masked Self-Attention
    msa = MaskedSelfAttention(EMBED_SIZE)
    output, weights = msa.forward(final_input)
    print("Masked attention weights (first 3x3 block):")
    print(weights[:3, :3])