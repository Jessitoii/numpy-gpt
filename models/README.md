# Model Checkpoints & Artifacts

This directory contains the serialized weights, hyperparameters, and structural metadata for the Jessitoii/Numpy-GPT Transformer models.

## Overview

The `models/` directory is designed for versioned storage of model states, enabling reproducible research and seamless inference deployment.

## Contents

- **Checkpoints**: Periodic snapshots of model weights (state dictionaries) saved during the training loop.
- **Final Weights**: Production-ready weights representing the optimal performance state.
- **Config Files**: JSON or YAML files defining the model architecture (layer depth, embedding dimensions, attention heads).

## Usage

### Loading a Model
To resume training or initiate inference, the model state can be loaded using the `saving.py` utility or standard serialization methods implemented in the core architecture.

```python
# Example conceptual usage
model.load_state("models/checkpoint_epoch_50.pt")
```

### Best Practices
- **Versioning**: Append timestamps or epoch numbers to filenames (e.g., `gpt_v1_20260425.npy`).
- **Metadata**: Always keep the corresponding configuration file alongside the weights to ensure architectural compatibility.

## Maintenance
It is recommended to prune intermediate checkpoints periodically to optimize storage, while retaining the "best" and "latest" iterations.
