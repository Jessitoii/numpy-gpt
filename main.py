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
from utils import generate, MAX_SEQ_LEN


# Fall back to NumPy if CuPy is unavailable
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
        
        self.token_embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = get_positional_encoding(seq_len, embed_size)
        self.transformer_block = TransformerBlock(embed_size, num_heads)
        
        self.ln_f = LayerNorm(embed_size)
        self.head = cp.random.randn(embed_size, vocab_size) * 0.01

    def forward(self, idx):
        if isinstance(idx, np.ndarray):
            idx = cp.asarray(idx)
        
        if idx.ndim == 2:
            if idx.shape[0] != 1:
                raise ValueError(f"Batch size > 1 not supported. Got shape: {idx.shape}")
            idx = idx[0]
        T = len(idx)
        
        x = self.token_embedding.forward(idx)
        x = x + self.positional_encoding[:T, :]
        x = self.transformer_block.forward(x)
        
        x, self.ln_f_cache = self.ln_f.forward(x)
        self.last_x = x
        
        logits = x @ self.head
        return logits

    
class DeepNanoGPT:
    def __init__(self, vocab_size, embed_size, num_heads, num_blocks, seq_len):
        self.token_embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = get_positional_encoding(seq_len, embed_size)
        self.blocks = [TransformerBlock(embed_size, num_heads) for _ in range(num_blocks)]
        self.ln_f = LayerNorm(embed_size)
        self.head = cp.random.randn(embed_size, vocab_size) * 0.01

    def forward(self, idx):
        # idx: (B, T)
        if isinstance(idx, np.ndarray):
            idx = cp.asarray(idx)
        
        B, T = idx.shape
        
        # Token embeddings + positional encoding
        # Broadcasting applies PE (T, C) across the batch dimension
        x = self.token_embedding.forward(idx)
        x = x + self.positional_encoding[:T, :]
        
        for block in self.blocks:
            x = block.forward(x)
            
        x, self.ln_f_cache = self.ln_f.forward(x)
        self.last_x = x
        
        # (B, T, embed_size) @ (embed_size, vocab_size) -> (B, T, vocab_size)
        logits = x @ self.head
        return logits


import re

def clean_whatsapp(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_data = []
    pattern = r'^(\[?\d{1,4}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s-\s|\[?\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s\d{1,2}:\d{2}\]?\s)'

    for line in lines:
        line = re.sub(pattern, '', line).strip()
        if line.startswith("-:"):
            line = line.replace("-:", "Alperitto:", 1)
        if line and "mesajları ve aramalar uçtan uca" not in line.lower() and "<Medya dahil edilmedi>" not in line:
            if ":" in line:
                cleaned_data.append(line)
            
    return "\n".join(cleaned_data)


EMBED_SIZE = 256
NUM_HEADS = 8
NUM_BLOCKS = 4
text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))

chars = sorted(list(set(text)))
vocab_size = len(chars)

if __name__ == "__main__":
    print(f"Vocabulary built. Size: {vocab_size}")
    print(f"Sample characters: {chars[:20]}")

    char_to_int = { ch:i for i,ch in enumerate(chars) }
    int_to_char = { i:ch for i,ch in enumerate(chars) }

    tokenizer.char_to_int = char_to_int
    tokenizer.int_to_char = int_to_char
    tokenizer.vocab_size = vocab_size

    data_indices = [char_to_int[c] for c in text]
    data_indices_gpu = cp.array(data_indices)

    model = DeepNanoGPT(vocab_size, EMBED_SIZE, NUM_HEADS, NUM_BLOCKS, MAX_SEQ_LEN)
    trainer = Trainer(model, learning_rate=0.0001)

    pbar = tqdm(range(60000), desc="Training", unit="step")

    for step in pbar:
        xb, yb = get_batch(data_indices_gpu, seq_len=128, batch_size=96)
        loss = trainer.train_step(xb, yb)
        pbar.set_postfix({"loss": f"{loss:.4f}"})
        
        if step % 100 == 0:
            pbar.write(f"\nStep {step} | Loss: {loss:.4f}")
            try:
                sample = generate(model, "Alperitto:", 50)
                pbar.write(f"Sample: {sample}\n")
            except Exception as e:
                pbar.write(f"Generation error: {e}")

        if step % 1000 == 0 and step > 0:
            save_model(model, f"./data/checkpoints/model_step_{step}.pkl")
            pbar.write(f"Checkpoint saved: model_step_{step}.pkl")

    save_model(model)