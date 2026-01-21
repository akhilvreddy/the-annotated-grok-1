# grok-moe-lens

An interpretability tool for analyzing Grok-1's Mixture of Experts (MoE) router weights. This project extracts and visualizes the routing patterns across all 64 transformer layers to understand how the model decides which experts to activate for different inputs.

**No GPU required** - this is pure weight analysis that runs on CPU.

## Key Research Questions

- Do adjacent layers have similar routing patterns?
- Do early vs late layers cluster separately?
- Does the same expert (e.g., expert #3) specialize in similar things across layers?
- Are some experts "broader" (activate for more diverse inputs)?

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Checkpoint

Download the Grok-1 weights (only the router weights will be analyzed):

```bash
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

### 3. Extract Router Weights

```bash
python -m analysis.extract_routers --checkpoint checkpoints/ckpt-0 --output routers.npz
```

This extracts only the router weights (~12MB) from the full 314B parameter checkpoint.

### 4. Run Analysis & Generate Figures

```bash
python -m analysis.visualize --input routers.npz --output-dir figures
```

## Output Artifacts

- `routers.npz` - Extracted router weights, shape `(64, 6144, 8)`
- `figures/layer_similarity_heatmap.png` - 64x64 cosine similarity matrix between layers
- `figures/layer_dendrogram.png` - Hierarchical clustering of layers
- `figures/layer_umap.png` - UMAP projection showing layer relationships
- `figures/layer_tsne.png` - t-SNE projection of layer router weights
- `figures/expert_similarity_across_layers.png` - How each expert's pattern evolves
- `figures/expert_correlation_matrix.png` - Which experts have similar activation patterns
- `figures/adjacent_layer_similarity.png` - Layer-to-layer continuity analysis

## Analysis Details

### Router Weight Structure

Each of the 64 transformer layers has a router with shape `[6144, 8]`:
- **6144**: Embedding dimension (input feature size)
- **8**: Number of experts in the MoE layer

The router computes `softmax(input @ router_weights)` to produce expert selection probabilities. Two experts are selected per token.

### Analyses Performed

1. **Layer Similarity**: Cosine similarity between flattened router weight matrices
2. **Hierarchical Clustering**: Ward linkage clustering to find layer groups
3. **Expert Consistency**: Does expert #N do similar things across layers?
4. **Dimensionality Reduction**: UMAP, t-SNE, PCA projections
5. **Expert Breadth**: Which experts are selective vs. broad?

## Project Structure

```
grok-moe-lens/
├── analysis/
│   ├── __init__.py
│   ├── extract_routers.py   # Extract router weights from checkpoint
│   ├── analyze_routers.py   # Compute similarity, clustering, PCA, etc.
│   └── visualize.py         # Generate publication-quality figures
├── figures/                  # Output visualizations
├── checkpoints/             # Downloaded model weights
│   └── ckpt-0/
├── model.py                 # Original Grok-1 model architecture
├── checkpoint.py            # Checkpoint loading utilities
├── run.py                   # Original inference script
└── requirements.txt
```

## Original Grok-1 Code

This repository is based on the official X.AI Grok-1 release. The original code for loading and running the full 314B parameter model is preserved in `model.py`, `checkpoint.py`, `runners.py`, and `run.py`.

### Model Specifications

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Maximum Sequence Length:** 8,192 tokens

## License

The code and associated Grok-1 weights are licensed under the Apache 2.0 license.
