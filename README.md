# WhatsApp GPT

A character-level GPT model built entirely from scratch using only **NumPy** and **CuPy**. This project demonstrates the implementation of a Transformer architecture without the use of deep learning frameworks like PyTorch or TensorFlow. The model is trained on personal WhatsApp conversation exports to mimic a specific messaging style.

## Why I built this
Most people learn transformers through PyTorch abstractions. I wanted to understand what actually happens at the matrix level — so I implemented everything by hand: forward pass, backpropagation, and Adam optimizer, with no autograd.

## Architecture Overview
The model follows the standard GPT-style (decoder-only) Transformer architecture:
- **Tokenizer:** Custom character-level tokenizer.
- **Positional Encoding:** Sinusoidal positional embeddings.
- **Attention:** Multi-head causal self-attention with causal masking.
- **Blocks:** Pre-norm Transformer blocks.
- **Loss Function:** Cross-entropy loss implemented from scratch.
- **Optimizer:** Custom Adam optimizer implementation.

### Default Configuration
- `embed_size`: 256
- `num_heads`: 8
- `num_blocks`: 4
- `seq_len`: 128

## Project Structure
- `main.py`: Core model definition and architecture components.
- `train.py`: Script for training the model on WhatsApp text data.
- `test.py`: CLI script for running inference and generating text.
- `mainwindow.py`: PyQt5 GUI for a more interactive chat experience.
- `tokenizer.py`: Character-level tokenizer logic.
- `attention.py`: Implementation of multi-head self-attention modules.
- `transformer.py`: Definition of individual Transformer blocks.
- `mhe.py`: Specialized multi-head embedding and attention components.
- `saving.py`: Logic for saving and loading model weights as `.pkl` files.
- `utils.py`: General helper functions and text generation logic.

## Setup & Usage

### 1. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Set up Environment
Create a `.env` file (see `.env.example`) and specify the path to your WhatsApp export:
```
WHATSAPP_PATH=data/my_chat.txt
```

### 3. Export WhatsApp Chat
- Open a chat on WhatsApp.
- Tap the menu (three dots) or the contact/group name.
- Select **Export Chat** -> **Without Media**.
- Save the resulting `.txt` file to your data directory.

### 4. Training
Run the training script to start learning from your data:
```bash
python train.py
```

### 5. Inference
To test the model via the command line:
```bash
python test.py
```

### 6. UI Interaction
For a graphical chat interface:
```bash
python mainwindow.py
```

## Requirements
- **Python 3.9+**
- **NVIDIA GPU:** Recommended for **CuPy** acceleration. The code will automatically fall back to **NumPy** if a compatible GPU/CUDA installation is not found.
