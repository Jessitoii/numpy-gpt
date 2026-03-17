import numpy as np
from tokenizer import Embedding, encode, get_positional_encoding, vocab_size
import math

# Fall back to NumPy if CuPy is unavailable
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False

class SelfAttention:
    def __init__(self, embed_size):
        self.embed_size = embed_size
        
        # Weight matrices for Q, K, V projections — shape: (embed_size, embed_size)
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        # Row-wise softmax with max subtraction for numerical stability
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        """
        x: Input tensor of shape (seq_len, embed_size)
        """
        # Project input into Q, K, V spaces
        # x: (N, D), W: (D, D) -> output: (N, D)
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # Scaled dot-product attention scores
        d_k = self.embed_size
        scores = (Q @ K.T) / cp.sqrt(d_k)

        # Attention weights via softmax
        attention_weights = self.softmax(scores)

        # Weighted sum of values
        output = attention_weights @ V

        return output, attention_weights
    

class MaskedSelfAttention:
    def __init__(self, embed_size):
        self.embed_size = embed_size
        
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        seq_len, embed_size = x.shape
        
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        scores = (Q @ K.T) / cp.sqrt(embed_size)

        # Causal mask: prevent attention to future positions
        mask = cp.tril(cp.ones((seq_len, seq_len)))
        scores = cp.where(mask == 0, -cp.inf, scores)

        # Masked positions become 0 after softmax
        attention_weights = self.softmax(scores)

        output = attention_weights @ V

        return output, attention_weights

if __name__ == "__main__":
    EMBED_SIZE = 16
    sa = SelfAttention(EMBED_SIZE)

    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)
    final_input = embedded_output + pe_matrix

    output, weights = sa.forward(final_input)

    print(f"Attention output shape: {output.shape}")  # (7, 16)
    print(f"Attention distribution for first token:\n{weights[0]}")

    msa = MaskedSelfAttention(EMBED_SIZE)
    output, weights = msa.forward(final_input)
    print("Masked attention weights (first 3x3 block):")
    print(weights[:3, :3])