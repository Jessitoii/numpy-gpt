import numpy as np
import tokenizer
from tokenizer import char_to_int, int_to_char, chars

# Fall back to NumPy if CuPy is unavailable
try:
    import cupy as cp
except (ImportError, Exception):
    cp = np

MAX_SEQ_LEN = 128

def generate(model, start_str, length=100, temperature=1.0):
    idx = cp.array([char_to_int[c] for c in start_str])
    idx = idx[cp.newaxis, :]  # Add batch dimension: (1, T)

    generated_str = start_str
    
    for _ in range(length):
        idx_cond = idx[:, -MAX_SEQ_LEN:]
        logits = model.forward(idx_cond)  # (1, T, V)
        
        # Apply temperature scaling to the last token's logits
        logits = logits[0, -1, :] / temperature  # (V,)
        
        probs = cp.exp(logits - cp.max(logits))
        probs /= cp.sum(probs)
        
        next_idx = int(cp.random.choice(len(chars), size=1, p=probs))
        next_idx_arr = cp.array([[next_idx]])
        idx = cp.concatenate((idx, next_idx_arr), axis=1)
        
        generated_str += int_to_char[int(next_idx)]
        
    return generated_str
