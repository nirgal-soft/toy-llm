# toy-llm

A minimal transformer-based language model implementation in CUDA/C++ for educational purposes.

## Overview

This project implements a small-scale transformer model from scratch using CUDA for GPU acceleration. It includes training, inference, and text preprocessing tools.

## Features

- Custom transformer architecture with configurable hyperparameters
- CUDA-accelerated forward and backward passes
- Text preprocessing and tokenization
- Training pipeline with Adam optimizer
- Text generation capabilities

## Requirements

- CUDA Toolkit (tested with compute capability 8.6 / RTX 3070)
- C++17 compatible compiler (clang++)
- NVIDIA GPU with CUDA support

## Building

The project uses a Makefile-based build system that automatically detects tools in `src/tools/`.

```bash
# Build all components
make all

# Build only C++ tools
make cpp

# Build only CUDA tools
make cuda

# Build with debug symbols
make DEBUG=1 all

# List available tools
make list-tools

# Clean build artifacts
make clean
```

## Project Structure

```
.
├── src/
│   ├── host/         # CPU-side training and data prep code
│   ├── device/       # CUDA kernels
│   └── tools/        # Executable tools (train, generate, preprocess)
├── include/          # Header files
├── data/             # Training data and preprocessed files
└── bin/              # Build output directory
```

## Usage

### Preprocessing Text Data

```bash
# Preprocess text file into token IDs
./bin/preprocess_txt <input.txt> <output_dir>
```

### Training

```bash
# Train model with default parameters
./bin/train

# Train with custom data paths
./bin/train <token_ids_path> <vocab_path>
```

### Text Generation

```bash
# Generate text from trained model
./bin/generate
```

## Model Architecture

- Configurable number of transformer layers
- Multi-head self-attention
- Position embeddings
- Layer normalization
- MLP feedforward networks

Default configuration:
- Vocab size: 10,000
- Embedding dim: 256
- Layers: 6
- Attention heads: 8
- Sequence length: 256
- Batch size: 32

## License

See LICENSE file for details.
