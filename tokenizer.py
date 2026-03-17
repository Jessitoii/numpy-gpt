import numpy as np
# CuPy'yi dene, yoksa NumPy'yi kullan
import re
import os
from dotenv import load_dotenv
load_dotenv()

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
try:
    import cupy as cp
    HAS_GPU = True
    print("✓ CuPy kullanılacak (GPU desteği etkin)")
except (ImportError, Exception) as e:
    cp = np
    HAS_GPU = False
    print(f"✗ CuPy yüklenemedi ({str(e)[:50]}...), NumPy kullanılacak")

# Örnek bir veri seti (Eğitim verimiz)
text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))
# Veriyi yükle
# raw_data = clean_whatsapp("whatsapp_konusma.txt")
# 2. Tokenizer Hazırla
chars = sorted(list(set(text))) 
vocab_size = len(chars)

# Karakter -> Sayı ve Sayı -> Karakter eşleşmeleri
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

# Encode: Metni sayılara çevirir
def encode(s):
    return [char_to_int[c] for c in s]

# Decode: Sayıları metne çevirir
def decode(l):
    return ''.join([int_to_char[i] for i in l])


class Embedding:
    def __init__(self, vocab_size, embed_size):
        """
        vocab_size: Sözlükteki toplam benzersiz karakter sayısı (V)
        embed_size: Her karakterin temsil edileceği vektör boyutu (D)
        """
        # Ağırlıkları rastgele başlatıyoruz (Normal dağılım) - GPU'da
        self.weights = cp.random.randn(vocab_size, embed_size) * 0.01
        self.input_indices = None

    def forward(self, x):
        """
        x: Karakter indekslerinden oluşan bir dizi (örneğin: [5, 12, 3])
        """
        self.input_indices = x
        # GPU üzerinde embedding arama
        return self.weights[x]

    def backward(self, d_out):
        """
        d_out: Bir sonraki katmandan gelen gradyanlar
        """
        d_weights = cp.zeros_like(self.weights)
        
        # Gradyanları ilgili indekslere biriktiriyoruz
        # cp.add.at, aynı indeksten birden fazla varsa gradyanları toplar
        cp.add.at(d_weights, self.input_indices, d_out)
        
        return d_weights


def get_positional_encoding(seq_len, embed_size):
    """
    seq_len: Maksimum cümle/dizi uzunluğu
    embed_size: Embedding vektör boyutu (D)
    """
    # Boş bir PE matrisi oluştur (seq_len x embed_size) - GPU'da
    pe = cp.zeros((seq_len, embed_size))
    
    # Pozisyon vektörü (0, 1, 2, ..., seq_len-1)
    position = cp.arange(seq_len)[:, cp.newaxis]
    
    # Paydadaki 10000^(2i/d_model) kısmını hesaplayalım
    # Logaritmik uzayda hesaplamak sayısal kararlılık için daha iyidir
    div_term = cp.exp(cp.arange(0, embed_size, 2) * -(cp.log(10000.0) / embed_size))
    
    # Çift indekslere (0, 2, 4...) Sinüs uygula
    pe[:, 0::2] = cp.sin(position * div_term)
    
    # Tek indekslere (1, 3, 5...) Kosinüs uygula
    pe[:, 1::2] = cp.cos(position * div_term)
    
    return pe

if __name__ == "__main__":
    # Uygulama ve Test
    SEQ_LEN = 10 # Maksimum 10 karakterlik bir dizi varsayalım
    EMBED_SIZE = 16
    # Uygulama ve Test
    embedding_layer = Embedding(vocab_size, EMBED_SIZE)

    # 'kodlama' kelimesinin embedding halini alalım
    input_data = np.array(encode("kodlama"))
    embedded_output = embedding_layer.forward(input_data)

    print(f"Girdi şekli (Token sayısı): {input_data.shape}")
    print(f"Çıktı şekli (Token sayısı, Embedding boyutu): {embedded_output.shape}")

    pe_matrix = get_positional_encoding(SEQ_LEN, EMBED_SIZE)

    print(f"PE Matris Şekli: {pe_matrix.shape}") # (10, 16)

    # Bir önceki adımda aldığımız 'embedded_output'u hatırlayalım
    # embedded_output şekli: (seq_len, embed_size)

    # Örnek girdi: "kodlama" (7 karakter)
    input_data = np.array(encode("kodlama"))
    embedded_output = embedding_layer.forward(input_data)

    # Bu girdinin uzunluğuna göre PE matrisinden ilgili kısmı alıyoruz
    current_seq_len = embedded_output.shape[0]
    pe_part = pe_matrix[:current_seq_len, :]

    # TOPLAMA: Model artık hem karakterin anlamını hem de yerini biliyor!
    final_input = embedded_output + pe_part

    print(f"Final Vektör Şekli: {final_input.shape}")

    print("final vektrör örneği (ilk 2 token):")
    print(final_input[:2])