"""
Multi-Head Attention (MHA) implementation.

This module provides the MultiHeadAttention class, which allows the model
to jointly attend to information from different representation subspaces
at different positions.
"""

import numpy as np
from tokenizer import Embedding, encode, get_positional_encoding, vocab_size

# Fall back to NumPy if CuPy is unavailable for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.

    Splits the embedding dimension into multiple heads, performs attention
    independently on each head, and concatenates the results.

    Attributes:
        embed_size (int): The total dimensionality of the embedding space.
        num_heads (int): Number of parallel attention heads.
        head_dim (int): Dimensionality of each individual head.
        Wq (ndarray): Combined query projection weights.
        Wk (ndarray): Combined key projection weights.
        Wv (ndarray): Combined value projection weights.
        Wo (ndarray): Output projection weights to merge heads.
        cache (tuple): Cached tensors for potential backward passes.
    """

    def __init__(self, embed_size, num_heads):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            embed_size (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.

        Raises:
            AssertionError: If embed_size is not divisible by num_heads.
        """
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Single weight matrices for all heads combined — split during forward pass
        # Initialization with small random values
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01
        
        # Output projection to merge all heads back to the original embed_size
        self.Wo = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        """
        Computes the softmax of the input array for numerical stability.

        Args:
            x (ndarray): The input scores.

        Returns:
            ndarray: The normalized probability distribution.
        """
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        """
        Performs the multi-head forward pass with causal masking.

        Args:
            x (ndarray): Input tensor of shape (batch_size, seq_len, embed_size) 
                         or (seq_len, embed_size).

        Returns:
            tuple: A tuple containing:
                - output (ndarray): The context-aware embeddings (B, T, C).
                - attention_weights (ndarray): The attention scores (B, h, T, T).
        """
        # Ensure input is 3D (Batch, Time, Channel)
        if len(x.shape) == 2:
            x = x[cp.newaxis, :, :]
            
        B, T, C = x.shape
        h = self.num_heads
        d_k = self.head_dim

        # Project to Q, K, V: (B, T, C) @ (C, C) -> (B, T, C)
        Q_total = x @ self.Wq
        K_total = x @ self.Wk
        V_total = x @ self.Wv

        # Reshape and transpose to separate heads: (B, h, T, d_k)
        # This allows batch-wise matrix multiplication across all heads
        Q = Q_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        K = K_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        V = V_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention: (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / cp.sqrt(d_k)

        # Causal mask: Ensure positions cannot see future tokens
        mask = cp.tril(cp.ones((T, T)))
        scores = cp.where(mask == 0, -cp.inf, scores)

        # Apply softmax to get attention weights
        attn_weights = self.softmax(scores)  # (B, h, T, T)
        
        # Weighted sum of values for each head: (B, h, T, d_k)
        out = attn_weights @ V               

        # Concatenate heads back together: (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Final linear projection
        output = out @ self.Wo
        
        # Cache internal states for potential backpropagation implementation
        self.cache = (x, Q, K, V, attn_weights, out)
        
        return output, attn_weights


if __name__ == "__main__":
    # Example usage for Multi-Head Attention
    NUM_HEADS = 4
    EMBED_SIZE = 16
    mha = MultiHeadAttention(EMBED_SIZE, NUM_HEADS)

    # Encode test data
    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    # Add positional encoding
    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)
    final_input = embedded_output + pe_matrix

    # Forward pass
    # Reshape input to (1, seq_len, embed_size) to simulate batch size 1
    mha_output, mha_weights = mha.forward(final_input)

    print(f"MHA output shape: {mha_output.shape}")
    print(f"Heads: {NUM_HEADS}, head dim: {EMBED_SIZE // NUM_HEADS}")
