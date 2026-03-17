import pickle
import cupy as cp
import numpy as np

def save_model(model, filename="whatsapp_gpt.pkl"):
    # 1. Tüm ağırlıkları bir sözlüğe topla ve GPU'dan CPU'ya çek
    weights = {
        'emb': cp.asnumpy(model.token_embedding.weights),
        'head': cp.asnumpy(model.head),
        'ln_f_g': cp.asnumpy(model.ln_f.gamma),
        'ln_f_b': cp.asnumpy(model.ln_f.beta),
        'blocks': []
    }
    
    for b in model.blocks:
        b_weights = {
            'Wq': cp.asnumpy(b.mha.Wq), 'Wk': cp.asnumpy(b.mha.Wk),
            'Wv': cp.asnumpy(b.mha.Wv), 'Wo': cp.asnumpy(b.mha.Wo),
            'W1': cp.asnumpy(b.ffn.W1), 'b1': cp.asnumpy(b.ffn.b1),
            'W2': cp.asnumpy(b.ffn.W2), 'b2': cp.asnumpy(b.ffn.b2),
            'ln1g': cp.asnumpy(b.ln1.gamma), 'ln1b': cp.asnumpy(b.ln1.beta),
            'ln2g': cp.asnumpy(b.ln2.gamma), 'ln2b': cp.asnumpy(b.ln2.beta)
        }
        weights['blocks'].append(b_weights)
        
    # 2. Dosyaya yaz
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)

def load_model(model, filename="whatsapp_gpt.pkl"):
    print(f"Model yükleniyor: {filename}...")
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
        
    # Ağırlıkları tekrar GPU'ya (CuPy) göndererek modele yükle
    model.token_embedding.weights = cp.asarray(weights['emb'])
    model.head = cp.asarray(weights['head'])
    model.ln_f.gamma = cp.asarray(weights['ln_f_g'])
    model.ln_f.beta = cp.asarray(weights['ln_f_b'])
    
    for i, b in enumerate(model.blocks):
        bw = weights['blocks'][i]
        b.mha.Wq = cp.asarray(bw['Wq'])
        b.mha.Wk = cp.asarray(bw['Wk'])
        b.mha.Wv = cp.asarray(bw['Wv'])
        b.mha.Wo = cp.asarray(bw['Wo'])
        b.ffn.W1 = cp.asarray(bw['W1'])
        b.ffn.b1 = cp.asarray(bw['b1'])
        b.ffn.W2 = cp.asarray(bw['W2'])
        b.ffn.b2 = cp.asarray(bw['b2'])
        b.ln1.gamma = cp.asarray(bw['ln1g'])
        b.ln1.beta = cp.asarray(bw['ln1b'])
        b.ln2.gamma = cp.asarray(bw['ln2g'])
        b.ln2.beta = cp.asarray(bw['ln2b'])
    print("Model başarıyla yüklendi!")