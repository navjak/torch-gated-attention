# torch-gated-attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.06708-b31b1b.svg)](https://arxiv.org/abs/2505.06708)


An architecture-agnostic PyTorch implementation of the **Gated Attention mechanism** from **["Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"](https://arxiv.org/abs/2505.06708) (Qiu et al., 2025)** [![arXiv](https://img.shields.io/badge/arXiv-2505.06708-b31b1b.svg)](https://arxiv.org/abs/2505.06708).

Unlike the original implementation which is coupled to the Qwen architecture, this package provides a **modular, drop-in layer** compatible with any Transformer. It includes native support for modern inference techniques like **RoPE (Rotary Positional Embeddings)** and **KV Caching**.

## Features

- **Architecture Agnostic:** Decoupled from specific model codebases (Qwen, Llama, etc.).
- **Drop-in Replacement:** Can replace standard `nn.MultiheadAttention` in custom transformers.
- **Inference Ready:** Built-in support for **KV Caching** (via `past_kv`) for efficient text generation.
- **Customizable Position Embeddings:** Supports **RoPE** and other positional injection via a flexible `rotary_callback`.
- **PyTorch Native:** Built on top of `F.scaled_dot_product_attention` for Flash Attention efficiency.

## Installation

### From Source

```bash
git clone https://github.com/navjak/torch-gated-attention.git
cd torch-gated-attention
pip install -e .
```

## Usage

### 1. Basic Usage

Use it just like a standard attention layer.

```python
import torch
from gated_attention import GatedAttention

# Initialize the layer
# example: 512 dim, 8 heads
attn = GatedAttention(d_model=512, num_heads=8)

# Create dummy input [batch_size, seq_len, dim]
x = torch.randn(1, 10, 512)

# Forward pass
output, current_kv = attn(x)

print(output.shape)  # torch.Size([1, 10, 512])
```

### 2. With Rotary Embeddings (RoPE)

You can inject any rotary embedding logic using the `rotary_callback`. This allows the layer to remain agnostic to how RoPE is implemented in your specific model.

```python
def my_rope_function(q, k):
    # your custom RoPE logic
    # This is just a dummy example
    return q, k

# Pass the callback during forward
output, _ = attn(x, rotary_callback=my_rope_function)
```

### 3. Text Generation (KV Caching)

Efficiently generate tokens by passing the cached Key/Values from the previous step.

```python
# Step 1: Process the prompt
output_1, cache_1 = attn(token_1)

# Step 2: Generate next token using cache
# The layer automatically concatenates new KV with past KV
# Note: Argument name is 'past_kv'
output_2, cache_2 = attn(token_2, past_kv=cache_1)
```

## Citation

```bibtex
@article{qiu2025gated,
  title={Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free},
  author={Qiu, Zihan and Wang, Zekun and others},
  journal={arXiv preprint arXiv:2505.06708},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

