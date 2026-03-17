import numpy as np
from tokenizer import Embedding, encode, get_positional_encoding, vocab_size
import math

# CuPy'yi dene, yoksa NumPy'yi kullan
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False

class SelfAttention:
    def __init__(self, embed_size):
        self.embed_size = embed_size
        
        # Q, K ve V için ağırlık matrisleri (Wq, Wk, Wv) - GPU'da
        # Şekil: (embed_size, embed_size)
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        # Satır bazında softmax (sayısal kararlılık için max çıkarıyoruz)
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        """
        x: Girdi (seq_len, embed_size)
        """
        # 1. Q, K, V vektörlerini oluştur
        # x: (N, D), W: (D, D) -> Çıktı: (N, D)
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # 2. Attention Skorlarını hesapla (Q * K^T)
        # d_k'ya bölerek ölçeklendiriyoruz
        d_k = self.embed_size
        scores = (Q @ K.T) / cp.sqrt(d_k)

        # 3. Softmax ile olasılığa çevir (Hangi karaktere ne kadar bakmalı?)
        attention_weights = self.softmax(scores)

        # 4. Değerler (V) ile ağırlıkları çarp
        output = attention_weights @ V

        return output, attention_weights
    

class MaskedSelfAttention:
    def __init__(self, embed_size):
        self.embed_size = embed_size
        
        # Ağırlıklar (Wq, Wk, Wv) - GPU'da
        self.Wq = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wk = cp.random.randn(embed_size, embed_size) * 0.01
        self.Wv = cp.random.randn(embed_size, embed_size) * 0.01

    def softmax(self, x):
        # Sayısal kararlılık için her satırdan max değerini çıkarıyoruz
        exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x):
        seq_len, embed_size = x.shape
        
        # 1. Q, K, V hesaplama
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # 2. Skorları hesapla ve ölçeklendir
        scores = (Q @ K.T) / cp.sqrt(embed_size)

        # 3. MASKELEME: Geleceği görmeyi engelle
        # tril (triangular lower) fonksiyonu ile alt üçgen matrisi oluşturuyoruz
        mask = cp.tril(cp.ones((seq_len, seq_len)))
        
        # Maskedeki 0'ları -inf (eksi sonsuz) yapıyoruz
        scores = cp.where(mask == 0, -cp.inf, scores)

        # 4. Softmax (Artık -inf olan yerler 0 olacak)
        attention_weights = self.softmax(scores)

        # 5. Çıktı
        output = attention_weights @ V

        return output, attention_weights

if __name__ == "__main__":
    # Uygulama
    EMBED_SIZE = 16
    sa = SelfAttention(EMBED_SIZE)

    # Örnek girdi: "kodlama" (7 karakter)
    input_data = np.array(encode("kodlama"))
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)
    embedded_output = embedding_layer.forward(input_data)

    # Pozisyonel kodlamayı al
    pe_matrix = get_positional_encoding(embedded_output.shape[0], EMBED_SIZE)

    # Embedding ve pozisyonel kodlamayı topla
    final_input = embedded_output + pe_matrix

    output, weights = sa.forward(final_input)

    print(f"Attention Çıktı Şekli: {output.shape}") # (7, 16)
    print(f"İlk karakterin diğerlerine 'dikkat' dağılımı:\n{weights[0]}")
    msa = MaskedSelfAttention(EMBED_SIZE)
    output, weights = msa.forward(final_input)
    print("Maskeli Attention Ağırlıkları (İlk 3x3 bölüm):")
    print(weights[:3, :3])