import numpy as np
import tokenizer
from transformer import TransformerBlock, LayerNorm
from tokenizer import Embedding, encode, vocab_size, get_positional_encoding, int_to_char, decode, char_to_int
from train import Trainer, get_batch
from saving import save_model
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv
load_dotenv()

# CuPy'yi dene, yoksa NumPy'yi kullan
try:
    import cupy as cp
    HAS_GPU = True
except (ImportError, Exception):
    cp = np
    HAS_GPU = False
class NanoGPT:
    def __init__(self, vocab_size, embed_size, num_heads, seq_len):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # 1. Katmanlar - GPU'da
        self.token_embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = get_positional_encoding(seq_len, embed_size)
        
        # Tek bir Transformer Bloğu (NanoGPT'de bu bloktan çokça olur)
        self.transformer_block = TransformerBlock(embed_size, num_heads)
        
        # 2. Final Normalizasyon ve Çıkış Kafası - GPU'da
        self.ln_f = LayerNorm(embed_size)
        self.head = cp.random.randn(embed_size, vocab_size) * 0.01

    def forward(self, idx):
        # Support optional batch dimension of size 1
        if isinstance(idx, np.ndarray):
            idx = cp.asarray(idx)
        
        if idx.ndim == 2:
            if idx.shape[0] != 1:
                raise ValueError(f"Batch size >1 henüz desteklenmiyor. Gelen şekil: {idx.shape}")
            idx = idx[0]
        T = len(idx)
        
        # 1. Embedding + PE
        x = self.token_embedding.forward(idx)
        x = x + self.positional_encoding[:T, :]
        
        # 2. Transformer Bloğu
        x = self.transformer_block.forward(x)
        
        # 3. Final LayerNorm
        x, self.ln_f_cache = self.ln_f.forward(x) 
        
        # 4. Output Head için girişi sakla
        self.last_x = x 
        
        # 5. Logits
        logits = x @ self.head
        
        return logits
    
class DeepNanoGPT:
    def __init__(self, vocab_size, embed_size, num_heads, num_blocks, seq_len):
        self.token_embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = get_positional_encoding(seq_len, embed_size)
        
        # Birden fazla bloğu bir listede tutuyoruz
        self.blocks = [TransformerBlock(embed_size, num_heads) for _ in range(num_blocks)]
        
        self.ln_f = LayerNorm(embed_size)
        self.head = cp.random.randn(embed_size, vocab_size) * 0.01

    def forward(self, idx):
        # idx şekli: (batch_size, seq_len) -> Örn: (32, 128)
        if isinstance(idx, np.ndarray):
            idx = cp.asarray(idx)
        
        B, T = idx.shape # B: Batch Size, T: Sequence Length
        
        # 1. Token Gömme (Embedding)
        # weights[idx] yapıldığında CuPy (B, T, embed_size) döner
        x = self.token_embedding.forward(idx) 
        
        # 2. Positional Encoding Ekleme
        # self.positional_encoding: (MAX_SEQ_LEN, embed_size)
        # x: (B, T, embed_size)
        # Aşağıdaki toplama işleminde CuPy (T, C) olan PE'yi tüm batch'e (B) kopyalar (Broadcasting)
        x = x + self.positional_encoding[:T, :]
        
        # 3. Transformer Blokları
        # Bloklarımızın içindeki matris çarpımları (x @ W) 3D desteğine sahip olmalı
        for block in self.blocks:
            x = block.forward(x)
            
        # 4. Final Normalizasyon
        x, self.ln_f_cache = self.ln_f.forward(x)
        
        # 5. Backward için sakla
        self.last_x = x 
        
        # 6. Çıkış Logits: (B, T, embed_size) @ (embed_size, vocab_size) -> (B, T, vocab_size)
        logits = x @ self.head
        
        return logits
# --- MODELİ OLUŞTURMA VE TEST ETME ---

MAX_SEQ_LEN = 128
def generate(model, start_str, length=100, temperature=1.0):
    # 1. Başlangıç metnini sayılara çevir
    idx = cp.array([char_to_int[c] for c in start_str]) # (T,)
    
    # Model 3D (B, T) beklediği için batch boyutu ekle (1, T)
    idx = idx[cp.newaxis, :] 

    generated_str = start_str
    
    for _ in range(length):
        # Sadece modelin bakabileceği max pencereyi al (seq_len)
        idx_cond = idx[:, -MAX_SEQ_LEN:]
        
        # FORWARD: Burası hatanın olduğu yerdi!
        # model.forward artık tek bir değer (logits) döndürüyor
        logits = model.forward(idx_cond) # (1, T, V)
        
        # Sadece en son karakterin tahminini (logits) al ve sıcaklık ekle
        logits = logits[0, -1, :] / temperature # (V,)
        
        # Softmax ile olasılığa çevir
        probs = cp.exp(logits - cp.max(logits))
        probs /= cp.sum(probs)
        
        # Bir sonraki karakteri seç (Rastgele örnekleme)
        next_idx = int(cp.random.choice(len(chars), size=1, p=probs))        
        # Seçilen karakteri listeye ekle ve string'e çevir
        next_idx_arr = cp.array([[next_idx]]) # (1, 1)
        idx = cp.concatenate((idx, next_idx_arr), axis=1) # (1, T+1)
        
        generated_str += int_to_char[int(next_idx)]
        
    return generated_str



import re

def clean_whatsapp(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_data = []
    # Bu regex, neredeyse tüm WhatsApp tarih/saat formatlarını yakalar
    # Örn: "14.06.2025 00:28 - ", "[14/6 00:25] ", "14/06/25, 00:28 - "
    pattern = r'^(\[?\d{1,4}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s-\s|\[?\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s)'

    for line in lines:
        # Tarih/Saat kısmını temizle
        line = re.sub(pattern, '', line).strip()
        if line.startswith("-:"):
            line = line.replace("-:", "Alperitto:", 1)
        # Eğer satır boş değilse ve sistem mesajı (uçtan uca şifreleme vb.) değilse ekle
        if line and "mesajları ve aramalar uçtan uca" not in line.lower()and "<Medya dahil edilmedi>" not in line:
            # Senin istediğin "İsim: Mesaj" formatına zorlayalım
            # Eğer satırda zaten ':' varsa muhtemelen "İsim: Mesaj" şeklindedir
            if ":" in line:
                cleaned_data.append(line)
            
    return "\n".join(cleaned_data)

EMBED_SIZE = 128
NUM_HEADS = 8
NUM_BLOCKS = 4
text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))
# Veriyi yükle
# raw_data = clean_whatsapp("whatsapp_konusma.txt")
# 2. Tokenizer Hazırla
chars = sorted(list(set(text))) 
vocab_size = len(chars)

if __name__ == "__main__":


    print(f"Sözlük başarıyla oluşturuldu! Boyut: {vocab_size}")
    print(f"İçindeki bazı karakterler: {chars[:20]}")

    # 3. Eşleşmeleri (Mapping) yeniden tanımla
    char_to_int = { ch:i for i,ch in enumerate(chars) }
    int_to_char = { i:ch for i,ch in enumerate(chars) }

    # Update tokenizer module globals so encode/decode use new vocab
    tokenizer.char_to_int = char_to_int
    tokenizer.int_to_char = int_to_char
    tokenizer.vocab_size = vocab_size

    # 4. Şimdi çevirme işlemini yap (Artık KeyError vermeyecek)
    data_indices = [char_to_int[c] for c in text]
    data_indices_gpu = cp.array(data_indices)
    # 3. Modeli Oluştur (Daha derin bir yapı: 4 kafa, 2 blok)
    model = DeepNanoGPT(vocab_size, embed_size=256, num_heads=8, num_blocks=4, seq_len=128)
    trainer = Trainer(model, learning_rate=0.0001)

    # 4. Eğitim Döngüsü
    pbar = tqdm(range(60000), desc="🤖 Alperitto-GPT Eğitiliyor", unit="adım")

    for step in pbar:
        # 1. Veri Hazırlama
        xb, yb = get_batch(data_indices_gpu, seq_len=128, batch_size=96)
        
        # 2. Eğitim Adımı
        loss = trainer.train_step(xb, yb)
        
        # 3. İlerleme Çubuğunu Güncelle (Sağ tarafa Kayıp değerini ekler)
        # Bu sayede her adımda print beklemeden loss'un düştüğünü canlı görürsün
        pbar.set_postfix({"Loss": f"{loss:.4f}"})
        
        # 4. Belirli aralıklarla çıktı al (tqdm ile çakışmaması için pbar.write kullan)
        if step % 100 == 0:
            pbar.write(f"\n📢 Adım {step} | Güncel Kayıp: {loss:.4f}")
            try:
                # Örnek üretim
                sample = generate(model, "Alperitto:", 50)
                pbar.write(f"✨ Üretilen: {sample}\n")
            except Exception as e:
                pbar.write(f"⚠️ Üretim hatası: {e}")

        # 5. Model Kaydetme
        if step % 1000 == 0 and step > 0:
            save_model(model, f"./data/checkpoints/model_step_{step}.pkl")
            pbar.write(f"💾 Model kaydedildi: model_step_{step}.pkl")

    save_model(model)