# Data Assets

This directory serves as the centralized repository for all datasets utilized by the Jessitoii/Numpy-GPT Transformer implementation.

## Overview

High-quality data is the cornerstone of effective transformer training. This folder is structured to manage the lifecycle of data from raw acquisition to optimized tensor formats.

## Directory Structure

- **`raw/`**: Unprocessed source text files or external datasets.
- **`processed/`**: Tokenized and encoded data, typically stored as NumPy arrays (`.npy`) or optimized tensors for high-performance I/O during training.
- **`cache/`**: Temporary storage for intermediate processing steps and vocabulary mappings.

## Data Pipeline

1. **Ingestion**: Source text is placed in the `raw` subdirectory.
2. **Preprocessing**: The `tokenizer.py` utility processes raw text into numerical indices.
3. **Storage**: Final training and validation sets are serialized for efficient loading.

## Security & Privacy

Ensure that no sensitive or PII (Personally Identifiable Information) data is committed to this directory if the repository is shared. Use the `.gitignore` configuration to exclude large dataset files while maintaining the structure.
