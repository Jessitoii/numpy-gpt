"""
Model serialization and deserialization utilities.

This module provides functions to save and load Transformer model weights
using pickle, with support for transferring weights between CPU and GPU.
"""

import pickle
import cupy as cp
import numpy as np


def save_model(model, filename="whatsapp_gpt.pkl"):
    """
    Serializes and saves the model weights to a file.

    All weights are moved from GPU (CuPy) to CPU (NumPy) before being
    saved to ensure compatibility.

    Args:
        model (Transformer): The Transformer model to save.
        filename (str): Target filename for the saved weights. 
                        Defaults to "whatsapp_gpt.pkl".
    """
    # Collect all weights into a dictionary
    # Moving them from GPU (CuPy) to CPU (NumPy) before serialization is critical
    weights = {
        'emb':    cp.asnumpy(model.token_embedding.weights),
        'head':   cp.asnumpy(model.head),
        'ln_f_g': cp.asnumpy(model.ln_f.gamma),
        'ln_f_b': cp.asnumpy(model.ln_f.beta),
        'blocks': []
    }
    
    # Iterate through each Transformer block to extract sub-layer weights
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
        
    # Write the weight dictionary to disk
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)


def load_model(model, filename="whatsapp_gpt.pkl"):
    """
    Loads model weights from a file and restores them into the model.

    Weights are moved from CPU (NumPy) back to GPU (CuPy) during the 
    restoration process.

    Args:
        model (Transformer): The Transformer model to restore weights into.
        filename (str): The filename to load from. 
                        Defaults to "whatsapp_gpt.pkl".
    """
    print(f"Loading model from: {filename}...")
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    
    # Restore global weights, moving them back to GPU
    model.token_embedding.weights = cp.asarray(weights['emb'])
    model.head                    = cp.asarray(weights['head'])
    model.ln_f.gamma              = cp.asarray(weights['ln_f_g'])
    model.ln_f.beta               = cp.asarray(weights['ln_f_b'])
    
    # Restore block-specific weights
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
        b.ln1.beta  = cp.asarray(bw['ln1b'])
        b.ln2.gamma = cp.asarray(bw['ln2g'])
        b.ln2.beta  = cp.asarray(bw['ln2b'])

    print("Model loaded successfully.")