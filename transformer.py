"""
Transformer building blocks and layers.

This module implements the core components of a Transformer architecture,
including Dropout, Layer Normalization, Feed-Forward networks, and
the Transformer Block itself.
"""

import numpy as np
from mhe import MultiHeadAttention
from tokenizer import get_positional_encoding, encode, vocab_size, Embedding

# Fall back to NumPy if CuPy is unavailable for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False

EMBED_SIZE = 16
NUM_HEADS = 4

class Dropout:
    """
    Dropout layer for regularization.

    Randomly zeroes some of the elements of the input tensor with probability p.
    This helps prevent overfitting by forcing the network to learn redundant
    representations.

    Attributes:
        p (float): Probability of an element being zeroed.
        mask (ndarray): The mask applied during the last forward pass.
    """

    def __init__(self, p=0.1):
        """
        Initializes the Dropout layer.

        Args:
            p (float): Dropout probability. Defaults to 0.1.
        """
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        """
        Performs the forward pass of the dropout layer.

        Args:
            x (ndarray): Input tensor.
            training (bool): Whether the model is in training mode. Defaults to True.

        Returns:
            ndarray: The regularized output tensor.
        """
        if not training or self.p == 0:
            return x
        # Create a binary mask and scale the output to maintain expected values
        self.mask = (cp.random.rand(*x.shape) > self.p)
        return (x * self.mask) / (1.0 - self.p)

    def backward(self, d_out):
        """
        Performs the backward pass (gradient calculation).

        Args:
            d_out (ndarray): Upstream gradient.

        Returns:
            ndarray: The gradient with respect to the input.
        """
        return (d_out * self.mask) / (1.0 - self.p)


class LayerNorm:
    """
    Layer Normalization.

    Normalizes the input across the last dimension, helping to stabilize
    and accelerate training.

    Attributes:
        eps (float): Small constant for numerical stability.
        gamma (ndarray): Scale parameter.
        beta (ndarray): Shift parameter.
    """

    def __init__(self, embed_size, eps=1e-5):
        """
        Initializes the LayerNorm layer.

        Args:
            embed_size (int): The dimensionality of the input.
            eps (float): Stability constant. Defaults to 1e-5.
        """
        self.eps = eps
        self.gamma = cp.ones(embed_size)
        self.beta = cp.zeros(embed_size)

    def forward(self, x):
        """
        Performs the forward pass of layer normalization.

        Args:
            x (ndarray): Input tensor.

        Returns:
            tuple: A tuple containing:
                - out (ndarray): The normalized tensor.
                - cache (tuple): Internal states for backward pass.
        """
        mean = cp.mean(x, axis=-1, keepdims=True)
        var = cp.var(x, axis=-1, keepdims=True)
        # Standard normalization formula
        x_hat = (x - mean) / cp.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        cache = (x, x_hat, mean, var, self.gamma, self.eps)
        return out, cache


class FeedForward:
    """
    Position-wise Feed-Forward Network (FFN).

    Consists of two linear transformations with a ReLU activation in between.

    Attributes:
        W1 (ndarray): First linear layer weights.
        b1 (ndarray): First linear layer bias.
        W2 (ndarray): Second linear layer weights.
        b2 (ndarray): Second linear layer bias.
        dropout_p (float): Dropout probability for the hidden layer.
        cache (tuple): Cached values for backward pass.
    """

    def __init__(self, embed_size, dropout_p=0.1):
        """
        Initializes the FeedForward network.

        Args:
            embed_size (int): Input/Output dimensionality.
            dropout_p (float): Dropout probability. Defaults to 0.1.
        """
        # He-style initialization for better gradient flow in deep networks
        self.W1 = cp.random.randn(embed_size, 4 * embed_size) * cp.sqrt(2.0 / embed_size)
        self.b1 = cp.zeros(4 * embed_size)
        self.W2 = cp.random.randn(4 * embed_size, embed_size) * cp.sqrt(2.0 / (4 * embed_size))
        self.b2 = cp.zeros(embed_size)
        self.dropout_p = dropout_p

    def forward(self, x, training=True):
        """
        Performs the forward pass of the FFN.

        Args:
            x (ndarray): Input tensor.
            training (bool): Training mode flag. Defaults to True.

        Returns:
            ndarray: The transformed output tensor.
        """
        # First linear layer followed by ReLU activation
        h = cp.maximum(0, x @ self.W1 + self.b1)
        
        mask = None
        # Internal dropout for regularization
        if training and self.dropout_p > 0:
            mask = (cp.random.rand(*h.shape) > self.dropout_p)
            h = (h * mask) / (1.0 - self.dropout_p)
        
        # Second linear projection
        out = h @ self.W2 + self.b2
        self.cache = (x, h, mask, self.W1, self.b1, self.W2, self.b2)
        return out


class TransformerBlock:
    """
    A single Transformer Block.

    Combines Multi-Head Attention and Feed-Forward sub-layers with
    residual connections and Layer Normalization.

    Attributes:
        mha (MultiHeadAttention): The attention sub-layer.
        ffn (FeedForward): The position-wise FFN sub-layer.
        ln1 (LayerNorm): First layer norm (applied before MHA).
        ln2 (LayerNorm): Second layer norm (applied before FFN).
        ln1_cache (tuple): Cache for ln1 forward pass.
        ln2_cache (tuple): Cache for ln2 forward pass.
        mha_cache (tuple): Cache for mha forward pass.
        ffn_cache (tuple): Cache for ffn forward pass.
    """

    def __init__(self, embed_size, num_heads):
        """
        Initializes the Transformer Block.

        Args:
            embed_size (int): Embedding dimensionality.
            num_heads (int): Number of attention heads.
        """
        self.mha = MultiHeadAttention(embed_size, num_heads)
        self.ffn = FeedForward(embed_size)
        self.ln1 = LayerNorm(embed_size)
        self.ln2 = LayerNorm(embed_size)

    def forward(self, x):
        """
        Performs the forward pass using the Pre-norm architecture.

        Args:
            x (ndarray): Input tensor (Batch, SeqLen, EmbedSize).

        Returns:
            ndarray: The processed block output.
        """
        # Pre-norm architecture (GPT-style): Apply norm BEFORE the sub-layers
        
        # 1. Self-attention sub-layer with residual connection
        x_norm, self.ln1_cache = self.ln1.forward(x)
        attn_out, _ = self.mha.forward(x_norm)
        self.mha_cache = self.mha.cache
        x = x + attn_out  # Residual connection adds the original input back

        # 2. Feed-forward sub-layer with residual connection
        x_norm, self.ln2_cache = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        self.ffn_cache = self.ffn.cache
        x = x + ffn_out  # Residual connection adds the intermediate output back
        
        return x


if __name__ == "__main__":
    # Test script for the TransformerBlock
    block = TransformerBlock(EMBED_SIZE, NUM_HEADS)

    # Prepare sample input
    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    # Add positional encoding
    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)
    final_input = embedded_output + pe_matrix

    # Process through the block
    block_output = block.forward(final_input)
    print(f"Block output shape: {block_output.shape}")  # (7, 16)