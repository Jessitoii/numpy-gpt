import numpy as np
import matplotlib.pyplot as plt
from tokenizer import encode

# Set up cupy fallback if needed
try:
    import cupy as cp
except ImportError:
    cp = np

def visualize_attention(model, text, layer, head, average_heads=False, save_path=None):
    """
    Extracts and visualizes attention weights for a given input text.
    
    Args:
        model: The trained GPT model.
        text (str): Input text to visualize.
        layer (int): Index of the Transformer layer.
        head (int): Index of the attention head (ignored if average_heads=True).
        average_heads (bool): If True, averages attention weights across all heads.
        save_path (str): Optional path to save the generated heatmap as a PNG.
    """
    # 1. Encode text to token indices
    try:
        idx = encode(text)
    except KeyError as e:
        print(f"Error: Character {e} not in vocabulary.")
        return

    # Add batch dimension
    idx_tensor = np.array([idx])
    
    # 2. Run forward pass with return_attention=True
    # DeepNanoGPT supports this after our modifications
    _, attn_weights = model.forward(idx_tensor, return_attention=True, layer_idx=layer)
    
    # attn_weights shape is (B, h, T, T)
    # Convert back to numpy if it's a cupy array
    if hasattr(attn_weights, 'get'):
        attn_weights = attn_weights.get()
        
    attn_weights = np.array(attn_weights)
    
    # Extract the first batch item
    attn = attn_weights[0] # Shape: (h, T, T)
    
    if average_heads:
        # Average across all heads
        attn_matrix = np.mean(attn, axis=0)
        title = f"Attention Weights - Layer {layer} (Averaged Across Heads)"
    else:
        # Select the specific head
        attn_matrix = attn[head]
        title = f"Attention Weights - Layer {layer}, Head {head}"
        
    # Shape of attn_matrix should be (T, T)
    seq_len = len(text)
    
    # 3. Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    cax = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
    
    # Add colorbar
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    # Set ticks and labels
    # Use the actual characters as tick labels (representing tokens)
    tokens = list(text)
    
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    
    # Set axes labels
    ax.set_xlabel("Key Positions (Attended to)")
    ax.set_ylabel("Query Positions (Current Token)")
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    from main import DeepNanoGPT, EMBED_SIZE, NUM_HEADS, NUM_BLOCKS
    from tokenizer import vocab_size
    from utils import MAX_SEQ_LEN
    
    print("Loading model for demonstration...")
    # Instantiate model
    model = DeepNanoGPT(vocab_size, EMBED_SIZE, NUM_HEADS, NUM_BLOCKS, MAX_SEQ_LEN)
    
    # The text requested by the user
    sample_text = "the quick brown fox jumps over the lazy dog"
    
    # Fallback to a valid string if the above has missing chars in vocab
    try:
        encode(sample_text)
    except KeyError:
        print("Sample text contains characters not in vocab. Using a subset or alternate text.")
        # Fallback using only spaces and a few common letters if possible, or just the vocab chars
        from tokenizer import chars
        sample_text = "".join(chars[:min(20, len(chars))]) # Ensure valid string
    
    # 1. Visualize Layer 0, Head 0
    visualize_attention(model, sample_text, layer=0, head=0, save_path="attention_layer0_head0.png")
    
    # 2. Visualize with averaging across heads
    visualize_attention(model, sample_text, layer=0, head=0, average_heads=True, save_path="attention_layer0_avg.png")
