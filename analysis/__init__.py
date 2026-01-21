# grok-moe-lens analysis package
# Tools for extracting and analyzing Grok-1 MoE router weights

from .extract_routers import extract_router_weights, load_router_weights
from .analyze_routers import (
    compute_layer_similarity,
    compute_expert_similarity_across_layers,
    reduce_dimensions,
)
from .visualize import (
    plot_layer_similarity_heatmap,
    plot_layer_dendrogram,
    plot_layer_umap,
    plot_expert_similarity,
)

__all__ = [
    "extract_router_weights",
    "load_router_weights",
    "compute_layer_similarity",
    "compute_expert_similarity_across_layers",
    "reduce_dimensions",
    "plot_layer_similarity_heatmap",
    "plot_layer_dendrogram",
    "plot_layer_umap",
    "plot_expert_similarity",
]
