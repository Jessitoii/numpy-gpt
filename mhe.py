import numpy as np
from tokenizer import Embedding, encode, get_positional_encoding, vocab_size
from attention import SelfAttention, MaskedSelfAttention

# CuPy'yi dene, yoksa NumPy'yi kullan
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False
class MultiHeadAttention:
    def __init__(self, embed_size, num_heads):
        assert embed_size % num_heads == 0, "Embed size, kafa sayısına tam bölünmeli!"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads # Her kafanın derinliği
        
        # Tüm kafalar için ağırlıkları tek bir büyük matriste tutuyoruz - GPU'da
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01
        
        # Çıktıyı tekrar embed_size boyutuna getirecek olan ağırlık
        self.Wo = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
    # x şekli: (B, T, C) -> (Batch, Time, Embedding)
        B, T, C = x.shape
        h = self.num_heads
        d_k = self.head_dim

        # 1. Q, K, V hesapla: (B, T, C) @ (C, C) -> (B, T, C)
        Q_total = x @ self.Wq
        K_total = x @ self.Wk
        V_total = x @ self.Wv

        # 2. Kafalara ayır ve Transpose et
        # Hedef Şekil: (Batch, Head, Time, Head_Dim) -> (B, h, T, d_k)
        Q = Q_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        K = K_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        V = V_total.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

        # 3. Attention Skorları: (B, h, T, d_k) @ (B, h, d_k, T) -> (B, h, T, T)
        # K.transpose(0, 1, 3, 2) son iki boyutu (T ve d_k) yer değiştirir
        scores = (Q @ K.transpose(0, 1, 3, 2)) / cp.sqrt(d_k)

        # 4. Maskeleme (B, h, T, T) boyutuna uygun maske
        mask = cp.tril(cp.ones((T, T)))
        scores = cp.where(mask == 0, -cp.inf, scores)

        # 5. Softmax ve Değerlerle Çarpım
        attn_weights = self.softmax(scores) # (B, h, T, T)
        out = attn_weights @ V # (B, h, T, d_k)

        # 6. Kafaları Geri Birleştir (Concatenate)
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # 7. Final Projeksiyon
        output = out @ self.Wo
        
        # Geri yayılım için her şeyi sakla (Cache)
        self.cache = (x, Q, K, V, attn_weights, out)
        
        return output, attn_weights
# Uygulama
if __name__ == "__main__":
    NUM_HEADS = 4
    EMBED_SIZE = 16
    mha = MultiHeadAttention(EMBED_SIZE, NUM_HEADS)

    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    # Pozisyonel kodlamayı al
    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)

    # Embedding ve pozisyonel kodlamayı topla
    final_input = embedded_output + pe_matrix

    mha_output, mha_weights = mha.forward(final_input)

    print(f"MHA Çıktı Şekli: {mha_output.shape}") # (7, 16)
    print(f"Kafa sayısı: {NUM_HEADS}, Her kafanın boyutu: {EMBED_SIZE//NUM_HEADS}")