# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**grok-moe-lens** - An interpretability tool for analyzing Grok-1's MoE router weights. Built on top of the official X.AI Grok-1 repository.

Two main use cases:
1. **Router Analysis** (CPU only) - Extract and analyze routing patterns across 64 layers
2. **Model Inference** (GPU required) - Run the full 314B parameter model

## Commands

### Router Weight Analysis (No GPU Required)

```bash
# Install analysis dependencies only
pip install -r requirements-analysis.txt

# Extract router weights from checkpoint
python -m analysis.extract_routers --checkpoint checkpoints/ckpt-0 --output routers.npz

# Run analysis and generate figures
python -m analysis.visualize --input routers.npz --output-dir figures
```

### Full Model Inference (GPU Required)

```bash
# Install all dependencies (including JAX with CUDA)
pip install -r requirements.txt

# Run inference
python run.py
```

**Linting** (ruff configured in pyproject.toml):
```bash
ruff check .
ruff format .
```

## Architecture

### Analysis Module (`analysis/`)

- **`extract_routers.py`** - Extract router weights from checkpoint
  - `extract_router_weights()` - Main extraction function
  - `load_router_weights()` - Load from saved `.npz` file

- **`analyze_routers.py`** - Compute analysis metrics
  - `compute_layer_similarity()` - Cosine similarity between layers
  - `compute_hierarchical_clustering()` - Ward linkage clustering
  - `compute_expert_similarity_across_layers()` - Expert consistency
  - `reduce_dimensions()` - UMAP, t-SNE, PCA projections
  - `run_full_analysis()` - Run all analyses

- **`visualize.py`** - Generate publication-quality figures
  - `plot_layer_similarity_heatmap()` - 64×64 similarity matrix
  - `plot_layer_dendrogram()` - Hierarchical clustering tree
  - `plot_layer_umap()` / `plot_layer_embedding()` - 2D projections
  - `plot_expert_similarity()` - Per-expert similarity across layers
  - `generate_all_figures()` - Generate all visualizations

### Original Grok-1 Model Code

- **`run.py`** - Entry point that configures model specs and runs text generation
- **`model.py`** - Model architecture: `Transformer`, `LanguageModel`, `DecoderLayer`, `MoELayer`, `MultiHeadAttention`, `Router`, `RMSNorm`, `RotaryEmbedding`, and `QuantizedWeight8bit` for 8-bit quantization
- **`runners.py`** - Inference execution: `InferenceRunner` orchestrates distributed inference, `ModelRunner` handles sharding and forward passes, sampling utilities (`sample_token`, `top_p_filter`)
- **`checkpoint.py`** - Checkpoint loading with shared memory optimization and distributed tensor loading

### Router Weight Structure

Each of the 64 layers has a router with shape `[6144, 8]`:
- Router path: `decoder_layer_{i}/router/w`
- Embedding dimension: 6,144
- Number of experts: 8
- Total router parameters: 64 × 6,144 × 8 = 3.1M (~12MB in float32)

### Model Configuration

- 314B parameters, 64 layers
- MoE with 8 experts (2 selected per token)
- 48 query heads, 8 key/value heads
- Embedding size: 6,144
- Max sequence length: 8,192 tokens
- 8-bit weight quantization with scales

### Framework Patterns

- **JAX + Haiku**: Functional programming with `hk.Module` subclasses
- **Distributed execution**: Uses JAX mesh utilities and `PartitionSpec` for multi-host/multi-device sharding
- **Sharding constraints**: `with_sharding_constraint()` controls tensor layout across devices
- **KV cache**: `KVMemory` dataclass for efficient autoregressive generation

## Requirements

- **Analysis only**: Python 3.8+, numpy, scipy, matplotlib, seaborn, scikit-learn, umap-learn
- **Full inference**: Large GPU memory required (314B parameter model)
- Checkpoint must be downloaded separately: `huggingface-cli download xai-org/grok-1`

## Output Files

- `routers.npz` - Extracted router weights, shape `(64, 6144, 8)`
- `figures/*.png` - Generated visualizations
