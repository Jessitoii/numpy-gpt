import numpy as np
from mhe import MultiHeadAttention
from tokenizer import get_positional_encoding, encode, vocab_size, Embedding

# CuPy'yi dene, yoksa NumPy'yi kullan
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
        self.p = p # Kapatılma olasılığı
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.p == 0:
            return x
        
        # Rastgele maske oluştur (0 ve 1'lerden oluşur)
        # 1-p olasılığıyla hayatta kalır, p olasılığıyla sıfırlanır
        self.mask = (cp.random.rand(*x.shape) > self.p)
        
        # Ölçeklendirme yaparak (1 / (1-p)) veriyi normalize et
        return (x * self.mask) / (1.0 - self.p)

    def backward(self, d_out):
        # Sadece hayatta kalan nöronların gradyanını geçir
        return (d_out * self.mask) / (1.0 - self.p)
class LayerNorm:
    def __init__(self, embed_size, eps=1e-5):
        self.eps = eps
        # Ağırlıkları GPU'da başlat
        self.gamma = cp.ones(embed_size)
        self.beta = cp.zeros(embed_size)

    def forward(self, x):
        mean = cp.mean(x, axis=-1, keepdims=True)
        var = cp.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / cp.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        
        # Geri yayılım için gerekli her şeyi bir sözlükte tutalım
        cache = (x, x_hat, mean, var, self.gamma, self.eps)
        return out, cache

class FeedForward:
    def __init__(self, embed_size, dropout_p=0.1):
        # Ağırlık başlatma (He initialization benzeri)
        self.W1 = cp.random.randn(embed_size, 4 * embed_size) * cp.sqrt(2.0 / embed_size)
        self.b1 = cp.zeros(4 * embed_size)
        self.W2 = cp.random.randn(4 * embed_size, embed_size) * cp.sqrt(2.0 / (4 * embed_size))
        self.b2 = cp.zeros(embed_size)
        
        self.dropout_p = dropout_p

    def forward(self, x, training=True):
        # 1. İlk Katman (Linear + ReLU)
        h = cp.maximum(0, x @ self.W1 + self.b1)
        
        # 2. Dropout Uygulama
        mask = None
        if training and self.dropout_p > 0:
            # Maskeyi oluştur: 1'ler hayatta kalanlar, 0'lar kapananlar
            mask = (cp.random.rand(*h.shape) > self.dropout_p)
            # Ölçeklendirme: h = (h * mask) / (1 - p)
            h = (h * mask) / (1.0 - self.dropout_p)
        
        # 3. İkinci Katman (Linear)
        out = h @ self.W2 + self.b2
        
        # Backward için gereken her şeyi cache'e koy
        # x: Giriş, h: ReLU/Dropout sonrası hali, mask: Dropout maskesi
        self.cache = (x, h, mask, self.W1, self.b1, self.W2, self.b2)        
        return out

class TransformerBlock:
    def __init__(self, embed_size, num_heads):
        self.mha = MultiHeadAttention(embed_size, num_heads)
        self.ffn = FeedForward(embed_size)
        self.ln1 = LayerNorm(embed_size)
        self.ln2 = LayerNorm(embed_size)

    def forward(self, x):
        # 1. Attention + Residual + LayerNorm
        # Modern Transformer'larda (GPT) "Pre-Norm" kullanılır:
        x_norm, self.ln1_cache = self.ln1.forward(x)
        attn_out, _ = self.mha.forward(x_norm)
        self.mha_cache = self.mha.cache
        x = x + attn_out # Residual connection
        
        # 2. Feed Forward + Residual + LayerNorm
        x_norm, self.ln2_cache = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        self.ffn_cache = self.ffn.cache
        x = x + ffn_out # Residual connection
        
        return x

if __name__ == "__main__":
    # Uygulama
    block = TransformerBlock(EMBED_SIZE, NUM_HEADS)

    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    # Pozisyonel kodlamayı al
    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)

    # Embedding ve pozisyonel kodlamayı topla
    final_input = embedded_output + pe_matrix
    block_output = block.forward(final_input)

    print(f"Blok Çıktı Şekli: {block_output.shape}") # (7, 16)