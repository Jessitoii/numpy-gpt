import cupy as cp
import pickle
import time
from main import DeepNanoGPT, EMBED_SIZE, NUM_HEADS, NUM_BLOCKS, clean_whatsapp
from utils import generate, MAX_SEQ_LEN
from saving import load_model
from dotenv import load_dotenv
import os

load_dotenv()

text = clean_whatsapp(os.getenv("WHATSAPP_PATH"))

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }




def generate_response(model, start_str, length=150, temperature=0.8):
    idx = cp.array([char_to_int[c] for c in start_str if c in char_to_int])
    idx = idx[cp.newaxis, :]  # (1, T)
    
    generated = start_str
    
    for _ in range(length):
        idx_cond = idx[:, -MAX_SEQ_LEN:]
        logits = model.forward(idx_cond)
        
        # Temperature scaling: higher T = more random, lower T = more deterministic
        logits = logits[0, -1, :] / temperature
        
        probs = cp.exp(logits - cp.max(logits))
        probs /= cp.sum(probs)
        
        next_idx_raw = cp.random.choice(len(chars), size=1, p=probs)
        next_idx = int(next_idx_raw.item())

        if next_idx in int_to_char:
            generated += int_to_char[next_idx]
        else:
            generated += " "
        
    return generated


if __name__ == "__main__":
    model = DeepNanoGPT(
        vocab_size=len(chars),
        embed_size=256,
        num_heads=8,
        num_blocks=4,
        seq_len=128
    )
    load_model(model, "./data/final_model/whatsapp_gpt.pkl")

    print("\n" + "="*40)
    print("WhatsApp GPT — Interactive Mode")
    print("Type 'exit' to quit.")
    print("="*40 + "\n")

    while True:
        user_input = input("Prompt: ")
        if user_input.lower() == 'exit':
            break
        
        if not user_input:
            user_input = "Hello "

        print("\nGenerating...\n")
        response = generate(model, start_str=user_input, length=100, temperature=1)
        
        print("-" * 40)
        print(response)
        print("-" * 40 + "\n")