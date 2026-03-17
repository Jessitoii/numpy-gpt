import numpy as np
from tokenizer import Embedding, encode, get_positional_encoding, vocab_size
from attention import SelfAttention, MaskedSelfAttention

# Fall back to NumPy if CuPy is unavailable
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False

class MultiHeadAttention:
    def __init__(self, embed_size, num_heads):
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Single weight matrices for all heads combined — split during forward pass
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01
        
        # Output projection to merge all heads back to embed_size
        self.Wo = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        h = self.num_heads
        d_k = self.head_dim

        # Project to Q, K, V: (B, T, C) @ (C, C) -> (B, T, C)
        Q_total = x @ self.Wq
        K_total = x @ self.Wk
        V_total = x @ self.Wv

        # Reshape and transpose to (B, h, T, d_k)
        Q = Q_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        K = K_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        V = V_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention: (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / cp.sqrt(d_k)

        # Causal mask
        mask = cp.tril(cp.ones((T, T)))
        scores = cp.where(mask == 0, -cp.inf, scores)

        attn_weights = self.softmax(scores)  # (B, h, T, T)
        out = attn_weights @ V               # (B, h, T, d_k)

        # Concatenate heads: (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        output = out @ self.Wo
        
        # Cache for backpropagation
        self.cache = (x, Q, K, V, attn_weights, out)
        
        return output, attn_weights


if __name__ == "__main__":
    NUM_HEADS = 4
    EMBED_SIZE = 16
    mha = MultiHeadAttention(EMBED_SIZE, NUM_HEADS)

    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)
    final_input = embedded_output + pe_matrix

    mha_output, mha_weights = mha.forward(final_input)

    print(f"MHA output shape: {mha_output.shape}")
    print(f"Heads: {NUM_HEADS}, head dim: {EMBED_SIZE // NUM_HEADS}")