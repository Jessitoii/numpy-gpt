import numpy as np
from mhe import MultiHeadAttention
from tokenizer import get_positional_encoding, encode, vocab_size, Embedding

# Fall back to NumPy if CuPy is unavailable
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False

EMBED_SIZE = 16
NUM_HEADS = 4

class Dropout:
    def __init__(self, p=0.1):
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.p == 0:
            return x
        self.mask = (cp.random.rand(*x.shape) > self.p)
        return (x * self.mask) / (1.0 - self.p)

    def backward(self, d_out):
        return (d_out * self.mask) / (1.0 - self.p)


class LayerNorm:
    def __init__(self, embed_size, eps=1e-5):
        self.eps = eps
        self.gamma = cp.ones(embed_size)
        self.beta = cp.zeros(embed_size)

    def forward(self, x):
        mean = cp.mean(x, axis=-1, keepdims=True)
        var = cp.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / cp.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        cache = (x, x_hat, mean, var, self.gamma, self.eps)
        return out, cache


class FeedForward:
    def __init__(self, embed_size, dropout_p=0.1):
        # He-style initialization
        self.W1 = cp.random.randn(embed_size, 4 * embed_size) * cp.sqrt(2.0 / embed_size)
        self.b1 = cp.zeros(4 * embed_size)
        self.W2 = cp.random.randn(4 * embed_size, embed_size) * cp.sqrt(2.0 / (4 * embed_size))
        self.b2 = cp.zeros(embed_size)
        self.dropout_p = dropout_p

    def forward(self, x, training=True):
        h = cp.maximum(0, x @ self.W1 + self.b1)  # Linear + ReLU
        
        mask = None
        if training and self.dropout_p > 0:
            mask = (cp.random.rand(*h.shape) > self.dropout_p)
            h = (h * mask) / (1.0 - self.dropout_p)
        
        out = h @ self.W2 + self.b2
        self.cache = (x, h, mask, self.W1, self.b1, self.W2, self.b2)
        return out


class TransformerBlock:
    def __init__(self, embed_size, num_heads):
        self.mha = MultiHeadAttention(embed_size, num_heads)
        self.ffn = FeedForward(embed_size)
        self.ln1 = LayerNorm(embed_size)
        self.ln2 = LayerNorm(embed_size)

    def forward(self, x):
        # Pre-norm architecture (GPT-style)
        
        # Self-attention sub-layer
        x_norm, self.ln1_cache = self.ln1.forward(x)
        attn_out, _ = self.mha.forward(x_norm)
        self.mha_cache = self.mha.cache
        x = x + attn_out  # Residual connection

        # Feed-forward sub-layer
        x_norm, self.ln2_cache = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        self.ffn_cache = self.ffn.cache
        x = x + ffn_out  # Residual connection
        
        return x


if __name__ == "__main__":
    block = TransformerBlock(EMBED_SIZE, NUM_HEADS)

    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)
    final_input = embedded_output + pe_matrix

    block_output = block.forward(final_input)
    print(f"Block output shape: {block_output.shape}")  # (7, 16)