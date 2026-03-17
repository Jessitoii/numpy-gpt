import numpy as np

# CuPy'yi dene, yoksa NumPy'yi kullan
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False


def cross_entropy_loss(logits, targets):
    # logits: (B, T, V) -> (Batch, Time, Vocab)
    # targets: (B, T)   -> (Batch, Time)
    B, T, V = logits.shape
    
    # 1. Her şeyi 2 boyutlu bir listeye çeviriyoruz (Düzleştirme)
    # (B, T, V) -> (B*T, V)
    # (B, T)    -> (B*T)
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    
    # 2. Softmax hesapla (Satır bazında)
    probs = cp.exp(logits_flat - cp.max(logits_flat, axis=-1, keepdims=True))
    probs /= cp.sum(probs, axis=-1, keepdims=True)
    
    # 3. Kayıp (Loss) hesapla
    N = B * T # Toplam token sayısı
    correct_logprobs = -cp.log(probs[cp.arange(N), targets_flat] + 1e-10)
    loss = cp.sum(correct_logprobs) / N
    
    # 4. Geri yayılım (Backward) gradyanı: (Tahmin - Gerçek)
    d_logits_flat = probs.copy()
    d_logits_flat[cp.arange(N), targets_flat] -= 1
    d_logits_flat /= N
    
    # 5. Gradyanı tekrar orijinal 3D şekline geri sok
    d_logits = d_logits_flat.reshape(B, T, V)
    
    return loss, d_logits
# Modelin en sonundaki 'head' ağırlıkları için
# Modelin en sonundaki 'head' ağırlıkları için
def backward_head(d_logits, last_layer_input, head_weights):
    # d_logits: (T, vocab_size)
    # last_layer_input: (T, embed_size)
    
    # Ağırlıklara giden gradyan: dL/dW = input^T * d_logits
    d_head = last_layer_input.T @ d_logits
    
    # Bir önceki katmana (Transformer bloğuna) giden gradyan:
    d_x = d_logits @ head_weights.T
    
    return d_head, d_x

def softmax_backward(d_out, softmax_output):
    """
    d_out: Softmax'tan sonraki katmandan gelen gradyan (Attention V ile çarpımından gelen)
    softmax_output: İleri besleme sırasında hesaplanan softmax matrisi (Attention weights)
    """
    # d_out ve softmax_output çarpımının satır bazında toplamı
    sum_dot = cp.sum(d_out * softmax_output, axis=-1, keepdims=True)
    # Gradyan formülü
    d_z = softmax_output * (d_out - sum_dot)
    return d_z

def mha_backward(self, d_out, cache):
    # d_out: (B, T, C)
    x, Q, K, V, attn_weights, out_before_proj = cache
    B, T, C = x.shape
    h = self.num_heads
    d_k = self.head_dim

    # 1. Output Projeksiyon Gradyanı
    # d_out_flat: (B*T, C), out_before_proj_flat: (B*T, C)
    d_Wo = out_before_proj.reshape(B*T, C).T @ d_out.reshape(B*T, C)
    d_out_before_proj = d_out @ self.Wo.T # (B, T, C)

    # 2. Kafalara Geri Ayır: (B, T, C) -> (B, h, T, d_k)
    d_heads_out = d_out_before_proj.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

    # 3. V ve Attention Weights Gradyanları
    # attn_weights: (B, h, T, T), d_heads_out: (B, h, T, d_k)
    d_V_heads = attn_weights.transpose(0, 1, 3, 2) @ d_heads_out # (B, h, T, d_k)
    d_attn_weights = d_heads_out @ V.transpose(0, 1, 3, 2) # (B, h, T, T)

    # 4. Softmax Backward (B, h, T, T)
    d_scores = attn_weights * (d_attn_weights - cp.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True))
    d_scores /= cp.sqrt(d_k)

    # 5. Q ve K Gradyanları
    d_Q_heads = d_scores @ K # (B, h, T, d_k)
    d_K_heads = d_scores.transpose(0, 1, 3, 2) @ Q # (B, h, T, d_k)

    # 6. 3D'ye Geri Dönüş
    d_Q = d_Q_heads.transpose(0, 2, 1, 3).reshape(B, T, C)
    d_K = d_K_heads.transpose(0, 2, 1, 3).reshape(B, T, C)
    d_V = d_V_heads.transpose(0, 2, 1, 3).reshape(B, T, C)

    # 7. Ağırlık Gradyanları (Wq, Wk, Wv) - Yine Düzleştirerek
    x_flat = x.reshape(B * T, C)
    d_Wq = x_flat.T @ d_Q.reshape(B * T, C)
    d_Wk = x_flat.T @ d_K.reshape(B * T, C)
    d_Wv = x_flat.T @ d_V.reshape(B * T, C)

    # 8. Giriş Verisine Giden Gradyan (Residual için)
    d_x = (d_Q @ self.Wq.T) + (d_K @ self.Wk.T) + (d_V @ self.Wv.T)

    return d_x, d_Wq, d_Wk, d_Wv, d_Wo

def ffn_backward(d_out, cache, dropout_p=0.1):
    x, h, mask, W1, b1, W2, b2 = cache
    B, T, C = d_out.shape

    # Matris çarpımı için düzleştirme
    d_out_flat = d_out.reshape(-1, C)
    h_flat = h.reshape(-1, h.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])

    # 1. W2 ve b2 Gradyanları
    d_W2 = h_flat.T @ d_out_flat
    d_b2 = cp.sum(d_out_flat, axis=0)

    # 2. Geriye giden gradyan (d_h)
    d_h_flat = d_out_flat @ W2.T

    # 3. Dropout Backward
    if mask is not None:
        mask_flat = mask.reshape(-1, mask.shape[-1])
        # İleri yoldaki ölçeklendirmenin aynısını gradyana da uygula
        d_h_flat = (d_h_flat * mask_flat) / (1.0 - dropout_p)

    # 4. ReLU Backward
    d_h_flat[h_flat <= 0] = 0

    # 5. W1 ve b1 Gradyanları
    d_W1 = x_flat.T @ d_h_flat
    d_b1 = cp.sum(d_h_flat, axis=0)

    # 6. Girişe giden gradyan (d_x)
    d_x = (d_h_flat @ W1.T).reshape(B, T, C)

    return d_x, d_W1, d_b1, d_W2, d_b2

def layernorm_backward(d_out, cache):
    # d_out: (B, T, C)
    x, x_hat, mean, var, gamma, eps = cache
    B, T, C = d_out.shape
    
    # 1. d_gamma ve d_beta (Parametre gradyanları)
    # Gamma ve Beta (C,) boyutundadır. Bu yüzden B ve T üzerinden toplamalıyız.
    d_gamma = cp.sum(d_out * x_hat, axis=(0, 1))
    d_beta = cp.sum(d_out, axis=(0, 1))
    
    # 2. Girişe giden gradyan (d_x)
    # x_hat'a giden gradyan
    d_x_hat = d_out * gamma
    
    # Varyans ve ortalama üzerinden geri yayılım
    std_inv = 1.0 / cp.sqrt(var + eps)
    
    # Katman normalizasyonu türevi formülü (Matris hali)
    # d_x = (1/C) * std_inv * (C * d_x_hat - sum(d_x_hat) - x_hat * sum(d_x_hat * x_hat))
    term1 = C * d_x_hat
    term2 = cp.sum(d_x_hat, axis=-1, keepdims=True)
    term3 = x_hat * cp.sum(d_x_hat * x_hat, axis=-1, keepdims=True)
    
    d_x = (1.0 / C) * std_inv * (term1 - term2 - term3)
    
    return d_x, d_gamma, d_beta

def get_batch(data_indices_gpu, seq_len, batch_size):
    # n: Toplam karakter sayısı (Zaten GPU'da)
    n = len(data_indices_gpu)
    
    # 1. Başlangıç indekslerini doğrudan GPU'da üret
    ix = cp.random.randint(0, n - seq_len, batch_size)
    
    # 2. np.array yerine cp.stack kullan (GPU'da kalsın)
    # List comprehension içinde dilimleme (slicing) hala GPU'da gerçekleşir
    x = cp.stack([data_indices_gpu[i:i+seq_len] for i in ix])
    y = cp.stack([data_indices_gpu[i+1:i+seq_len+1] for i in ix])
    
    return x, y

def full_backward(model, d_logits):
    # d_logits: (B, T, V)
    # model.last_x: (B, T, C)
    B, T, V = d_logits.shape
    _, _, C = model.last_x.shape
    
    grads = {}
    
    # 1. Output Head Gradyanı (Düzleştirerek Çarp)
    # (B*T, C).T @ (B*T, V) -> (C, V)
    x_flat = model.last_x.reshape(B * T, C)
    d_logits_flat = d_logits.reshape(B * T, V)
    grads['d_head'] = x_flat.T @ d_logits_flat
    
    # 2. Bir önceki katmana (LayerNorm) giden gradyan: 3D kalsın
    # (B, T, V) @ (V, C) -> (B, T, C)
    d_x = d_logits @ model.head.T
    
    # 3. Final LayerNorm Backward (3D destekli)
    d_x, grads['d_gamma_f'], grads['d_beta_f'] = layernorm_backward(d_x, model.ln_f_cache)
        
    # 3. Blokları Tersten Gez (Backwards through blocks)
    grads['blocks'] = []
    for i in reversed(range(len(model.blocks))):
        block = model.blocks[i]
        block_grads = {}
        
        # FFN Backward
        d_x_ffn, block_grads['dW1'], block_grads['db1'], \
        block_grads['dW2'], block_grads['db2'] = ffn_backward(d_x, block.ffn_cache)
        
        # Residual Connection (Add) -> Gradyan aynen aktarılır
        d_x = d_x + d_x_ffn
        
        # LN2 Backward
        d_x, block_grads['d_gamma2'], block_grads['d_beta2'] = layernorm_backward(d_x, block.ln2_cache)
        
        # MHA Backward
        d_x_attn, block_grads['dWq'], block_grads['dWk'], \
        block_grads['dWv'], block_grads['dWo'] = mha_backward(block.mha, d_x, block.mha_cache)
        
        # Residual Connection (Add)
        d_x = d_x + d_x_attn
        
        # LN1 Backward
        d_x, block_grads['d_gamma1'], block_grads['d_beta1'] = layernorm_backward(d_x, block.ln1_cache)
        
        grads['blocks'].append(block_grads)
        
    # 4. Embedding Backward
    # (Embedding katmanına gelen d_x ile ağırlıklar güncellenir)
    grads['d_emb'] = d_x # Bu gradyan embedding tablosuna np.add.at ile uygulanır
    
    return grads



import cupy as cp

class Trainer:
    def __init__(self, model, learning_rate=3e-4, beta1=0.9, beta2=0.99, eps=1e-8):
        self.model = model
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Adım sayacı (bias düzeltmesi için)
        
        # Tüm parametreler için m (1. moment) ve v (2. moment) sözlükleri
        self.m = {}
        self.v = {}

    def _adam_update(self, param, grad, name):
        """Tek bir parametre için Adam güncelleme kuralı"""
        if name not in self.m:
            self.m[name] = cp.zeros_like(param)
            self.v[name] = cp.zeros_like(param)
        
        # Momentumları güncelle
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad**2)
        
        # Bias düzeltmesi (Eğitimin başında m ve v sıfıra çok yakın olduğu için)
        m_hat = self.m[name] / (1 - self.beta1**self.t)
        v_hat = self.v[name] / (1 - self.beta2**self.t)
        
        # Ağırlığı güncelle (Geriye doğru yazar: in-place update)
        param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def apply_gradients(self, grads, x_batch):
        self.t += 1
        
        # --- 1. Embedding Güncelleme (scatter_add ile biriktirerek) ---
        x_flat = x_batch.reshape(-1)
        d_emb_flat = grads['d_emb'].reshape(-1, self.model.token_embedding.weights.shape[1])
        d_weights = cp.zeros_like(self.model.token_embedding.weights)
        d_weights.scatter_add(x_flat, d_emb_flat) # Gradyanları biriktir
        
        self._adam_update(self.model.token_embedding.weights, d_weights, 'emb')
        
        # --- 2. Output Head Güncelleme ---
        self._adam_update(self.model.head, grads['d_head'], 'head')
        
        # --- 3. Final LayerNorm Güncelleme ---
        self._adam_update(self.model.ln_f.gamma, grads['d_gamma_f'], 'ln_f_g')
        self._adam_update(self.model.ln_f.beta, grads['d_beta_f'], 'ln_f_b')
        
        # --- 4. Blokları Güncelleme ---
        for i, b_grad in enumerate(reversed(grads['blocks'])):
            block = self.model.blocks[i]
            prefix = f'b{i}_'
            
            # MHA Ağırlıkları
            self._adam_update(block.mha.Wq, b_grad['dWq'], prefix + 'Wq')
            self._adam_update(block.mha.Wk, b_grad['dWk'], prefix + 'Wk')
            self._adam_update(block.mha.Wv, b_grad['dWv'], prefix + 'Wv')
            self._adam_update(block.mha.Wo, b_grad['dWo'], prefix + 'Wo')
            
            # FFN Ağırlıkları
            self._adam_update(block.ffn.W1, b_grad['dW1'], prefix + 'W1')
            self._adam_update(block.ffn.b1, b_grad['db1'], prefix + 'b1')
            self._adam_update(block.ffn.W2, b_grad['dW2'], prefix + 'W2')
            self._adam_update(block.ffn.b2, b_grad['db2'], prefix + 'b2')
            
            # LayerNorm Ağırlıkları
            self._adam_update(block.ln1.gamma, b_grad['d_gamma1'], prefix + 'ln1g')
            self._adam_update(block.ln1.beta, b_grad['d_beta1'], prefix + 'ln1b')
            self._adam_update(block.ln2.gamma, b_grad['d_gamma2'], prefix + 'ln2g')
            self._adam_update(block.ln2.beta, b_grad['d_beta2'], prefix + 'ln2b')

    def train_step(self, x_batch, y_batch):
        # 1. Forward
        logits = self.model.forward(x_batch)
        # 2. Loss
        loss, d_logits = cross_entropy_loss(logits, y_batch)
        # 3. Backward
        grads = full_backward(self.model, d_logits)
        # 4. Adam Update
        self.apply_gradients(grads, x_batch)
        
        return float(loss)