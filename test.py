import cupy as cp
import pickle
import time
from main import DeepNanoGPT, EMBED_SIZE, NUM_HEADS, NUM_BLOCKS, MAX_SEQ_LEN, clean_whatsapp, generate
from saving import load_model
from dotenv import load_dotenv
import os

load_dotenv()

# --- 1. MODELİ YÜKLEME FONKSİYONU ---

text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))

# 2. Tokenizer Hazırla
chars = sorted(list(set(text))) 
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }
def load_trained_model(model, filename="whatsapp_gpt.pkl"):
    print(f"🔄 Model yükleniyor: {filename}...")
    try:
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        
        # Ağırlıkları GPU'ya (CuPy) göndererek modele yükle
        model.token_embedding.weights = cp.asarray(weights['emb'])
        model.head = cp.asarray(weights['head'])
        model.ln_f.gamma = cp.asarray(weights['ln_f_g'])
        model.ln_f.beta = cp.asarray(weights['ln_f_b'])
        
        for i, b in enumerate(model.blocks):
            bw = weights['blocks'][i]
            b.mha.Wq, b.mha.Wk = cp.asarray(bw['Wq']), cp.asarray(bw['Wk'])
            b.mha.Wv, b.mha.Wo = cp.asarray(bw['Wv']), cp.asarray(bw['Wo'])
            b.ffn.W1, b.ffn.b1 = cp.asarray(bw['W1']), cp.asarray(bw['b1'])
            b.ffn.W2, b.ffn.b2 = cp.asarray(bw['W2']), cp.asarray(bw['b2'])
            b.ln1.gamma, b.ln1.beta = cp.asarray(bw['ln1g']), cp.asarray(bw['ln1b'])
            b.ln2.gamma, b.ln2.beta = cp.asarray(bw['ln2g']), cp.asarray(bw['ln2b'])
            
        print("✅ Model başarıyla yüklendi! Konuşmaya hazır.")
    except FileNotFoundError:
        print("❌ Hata: Model dosyası bulunamadı!")
        exit()

# --- 2. GELİŞMİŞ GENERATE FONKSİYONU ---
def generate_response(model, start_str, length=150, temperature=0.8):
    # Başlangıç metnini indexlere çevir
    idx = cp.array([char_to_int[c] for c in start_str if c in char_to_int])
    idx = idx[cp.newaxis, :] # (1, T)
    
    generated = start_str
    
    for _ in range(length):
        # Sadece son seq_len kadar karakteri al
        idx_cond = idx[:, -MAX_SEQ_LEN:]
        
        # Tahmin (Forward)
        logits = model.forward(idx_cond)
        
        # Son karakterin logitslerini al ve sıcaklık uygula
        # $P_i = \frac{\exp(z_i / T)}{\sum \exp(z_j / T)}$
        logits = logits[0, -1, :] / temperature
        
        # Softmax ve Örnekleme
        probs = cp.exp(logits - cp.max(logits))
        probs /= cp.sum(probs)
        
        # CuPy random.choice fix (size=1 ekledik)
        next_idx = int(cp.random.choice(len(chars), size=1, p=probs))
        
        # Listeye ekle
        next_idx_raw = cp.random.choice(len(chars), size=1, p=probs)
        next_idx = int(next_idx_raw.item())

        # KeyError'u engellemek için kontrol ekleyelim:
        if next_idx in int_to_char:
            generated += int_to_char[next_idx]
        else:
            # Eğer sözlükte yoksa en azından program çökmesin, yerine boşluk koysun
            generated += " " 
            print(f"⚠️ Uyarı: Sözlükte {next_idx} indeksi bulunamadı!")
        
        # Eğer mesajın bittiğini simgeleyen bir karakter varsa (opsiyonel)
        # if int_to_char[next_idx] == '\n': break 
        
    return generated

# --- 3. ANA DÖNGÜ (CHAT LOOP) ---
if __name__ == "__main__":
    # Modelini aynı hiperparametrelerle oluştur
    # model = DeepNanoGPT(vocab_size, EMBED_SIZE, NUM_HEADS, NUM_BLOCKS, MAX_SEQ_LEN)
    
    model = DeepNanoGPT(vocab_size= len(chars), 
                         embed_size=256,
                         num_heads=8,
                         num_blocks=4,
                         seq_len=128)
    load_model(model, "./data/final_model/whatsapp_gpt.pkl") # En son kaydettiğin dosya

    print("\n" + "="*30)
    print("🤖 WHATSAPP GPT BAŞLATILDI")
    print("Çıkmak için 'exit' yazın.")
    print("="*30 + "\n")

    while True:
        user_input = input("Sen (Başlangıç ver): ")
        if user_input.lower() == 'exit':
            break
        
        # Eğer boş bırakırsan varsayılan bir başlangıç verelim
        if not user_input:
            user_input = "Hello "

        print("\n✨ Model üretiyor...\n")
        
        # Farklı sıcaklıklarla (temperature) oynamak yaratıcılığı artırır
        response = generate(model, start_str=user_input, length=100, temperature=1)
        
        print("-" * 20)
        print(response)
        print("-" * 20 + "\n")