"""
Visualization functions for Grok-1 MoE router weight analysis.

Generates publication-quality figures:
- Layer similarity heatmaps
- Hierarchical clustering dendrograms
- 2D projections (UMAP, t-SNE, PCA)
- Expert analysis plots
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

# Set publication-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def plot_layer_similarity_heatmap(
    similarity: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Layer-to-Layer Router Similarity",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot heatmap of layer-to-layer similarity.

    Args:
        similarity: Similarity matrix of shape (64, 64)
        output_path: Path to save figure (optional)
        title: Plot title
        cmap: Colormap name
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(similarity, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", rotation=270, labelpad=15)

    # Labels
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)

    # Add tick marks for every 8 layers
    tick_positions = np.arange(0, 64, 8)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_layer_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Hierarchical Clustering of Layers by Router Weights",
    figsize: Tuple[int, int] = (14, 6),
    color_threshold: Optional[float] = None,
) -> plt.Figure:
    """
    Plot hierarchical clustering dendrogram.

    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy.linkage
        output_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size
        color_threshold: Distance threshold for coloring clusters

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create dendrogram
    dend = dendrogram(
        linkage_matrix,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold,
        labels=[f"L{i}" for i in range(64)],
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Distance")
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_layer_embedding(
    embedding: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "2D Projection of Layer Router Weights",
    method_name: str = "UMAP",
    figsize: Tuple[int, int] = (10, 8),
    colormap: str = "viridis",
) -> plt.Figure:
    """
    Plot 2D embedding of layers colored by layer index.

    Args:
        embedding: Array of shape (64, 2) containing 2D coordinates
        output_path: Path to save figure (optional)
        title: Plot title
        method_name: Name of dimensionality reduction method for axis labels
        figsize: Figure size
        colormap: Colormap for layer indices

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color by layer index
    layer_indices = np.arange(64)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=layer_indices,
        cmap=colormap,
        s=100,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add layer numbers as annotations
    for i, (x, y) in enumerate(embedding):
        ax.annotate(
            str(i),
            (x, y),
            fontsize=6,
            ha="center",
            va="center",
            color="white" if i > 32 else "black",
            weight="bold",
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Layer Index", rotation=270, labelpad=15)

    # Draw lines connecting consecutive layers
    for i in range(len(embedding) - 1):
        ax.plot(
            [embedding[i, 0], embedding[i + 1, 0]],
            [embedding[i, 1], embedding[i + 1, 1]],
            "k-",
            alpha=0.2,
            linewidth=0.5,
        )

    ax.set_xlabel(f"{method_name} Dimension 1")
    ax.set_ylabel(f"{method_name} Dimension 2")
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_layer_umap(
    embedding: np.ndarray,
    output_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Convenience wrapper for UMAP visualization."""
    return plot_layer_embedding(
        embedding,
        output_path=output_path,
        title="UMAP Projection of Layer Router Weights",
        method_name="UMAP",
        **kwargs,
    )


def plot_expert_similarity(
    expert_similarity: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Plot similarity matrices showing how each expert's pattern evolves across layers.

    Args:
        expert_similarity: Array of shape (8, 64, 64) - similarity matrix per expert
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    num_experts = expert_similarity.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()

    for expert_idx in range(num_experts):
        ax = axes[expert_idx]
        im = ax.imshow(
            expert_similarity[expert_idx],
            cmap="RdBu_r",
            aspect="auto",
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Expert {expert_idx}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")

        # Add ticks every 16 layers
        tick_positions = np.arange(0, 64, 16)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)

    # Add shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")

    fig.suptitle(
        "Expert Activation Pattern Similarity Across Layers\n"
        "(How similar is each expert's 'trigger pattern' between layers?)",
        y=1.02,
    )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_adjacent_layer_similarity(
    adjacent_similarities: np.ndarray,
    transition_indices: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot similarity between adjacent layers to identify transition points.

    Args:
        adjacent_similarities: Array of shape (63,) with similarity between layer i and i+1
        transition_indices: Indices where large transitions occur
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = np.arange(len(adjacent_similarities))
    ax.plot(layers, adjacent_similarities, "b-", linewidth=2, label="Similarity")
    ax.fill_between(layers, adjacent_similarities, alpha=0.3)

    # Mark transition points
    if len(transition_indices) > 0:
        for idx in transition_indices:
            if idx < len(adjacent_similarities):
                ax.axvline(idx, color="red", linestyle="--", alpha=0.7, label="Transition" if idx == transition_indices[0] else "")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Similarity with Next Layer")
    ax.set_title("Layer-to-Layer Routing Continuity")
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_expert_correlation_matrix(
    correlation_matrix: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot correlation between experts (averaged across layers).

    Args:
        correlation_matrix: Array of shape (8, 8) with expert correlations
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        xticklabels=[f"E{i}" for i in range(8)],
        yticklabels=[f"E{i}" for i in range(8)],
    )

    ax.set_title("Average Correlation Between Experts\n(across all layers)")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_expert_variance_analysis(
    expert_variance: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot expert variance metrics.

    Args:
        expert_variance: Dictionary from compute_expert_variance()
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    experts = np.arange(8)

    # Plot 1: Weight variance
    ax = axes[0]
    ax.bar(experts, expert_variance["weight_variance_mean"], color="steelblue")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Mean Weight Variance")
    ax.set_title("Expert Selectivity\n(higher = more selective)")
    ax.set_xticks(experts)

    # Plot 2: L2 norm
    ax = axes[1]
    ax.bar(experts, expert_variance["l2_norm_mean"], color="seagreen")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Expert Weight Magnitude")
    ax.set_xticks(experts)

    # Plot 3: Cross-layer variance
    ax = axes[2]
    ax.bar(experts, expert_variance["cross_layer_variance"], color="coral")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Cross-Layer Variance")
    ax.set_title("Expert Stability Across Layers\n(lower = more consistent)")
    ax.set_xticks(experts)

    fig.suptitle("Expert Characterization Metrics", y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def generate_all_figures(
    analysis_results: Dict,
    output_dir: str = "figures",
    routers: Optional[np.ndarray] = None,
) -> None:
    """
    Generate all visualization figures from analysis results.

    Args:
        analysis_results: Dictionary from run_full_analysis()
        output_dir: Directory to save figures
        routers: Original router weights (optional, for additional plots)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating figures in {output_dir}/...")

    # Layer similarity heatmap
    plot_layer_similarity_heatmap(
        analysis_results["layer_similarity"],
        output_path=output_dir / "layer_similarity_heatmap.png",
    )

    # Dendrogram
    plot_layer_dendrogram(
        analysis_results["linkage"],
        output_path=output_dir / "layer_dendrogram.png",
    )

    # 2D embeddings
    if analysis_results.get("umap_embedding") is not None:
        plot_layer_umap(
            analysis_results["umap_embedding"],
            output_path=output_dir / "layer_umap.png",
        )

    plot_layer_embedding(
        analysis_results["tsne_embedding"],
        output_path=output_dir / "layer_tsne.png",
        title="t-SNE Projection of Layer Router Weights",
        method_name="t-SNE",
    )

    plot_layer_embedding(
        analysis_results["pca_embedding"],
        output_path=output_dir / "layer_pca.png",
        title="PCA Projection of Layer Router Weights",
        method_name="PCA",
    )

    # Expert similarity
    plot_expert_similarity(
        analysis_results["expert_similarity"],
        output_path=output_dir / "expert_similarity_across_layers.png",
    )

    # Adjacent layer similarity
    plot_adjacent_layer_similarity(
        analysis_results["transitions"]["adjacent_similarities"],
        analysis_results["transitions"]["transition_indices"],
        output_path=output_dir / "adjacent_layer_similarity.png",
    )

    # Expert correlation
    plot_expert_correlation_matrix(
        analysis_results["expert_correlation"],
        output_path=output_dir / "expert_correlation_matrix.png",
    )

    # Expert variance
    plot_expert_variance_analysis(
        analysis_results["expert_variance"],
        output_path=output_dir / "expert_variance_analysis.png",
    )

    print(f"\nGenerated {len(list(output_dir.glob('*.png')))} figures in {output_dir}/")


if __name__ == "__main__":
    import argparse
    from analyze_routers import run_full_analysis
    from extract_routers import load_router_weights

    parser = argparse.ArgumentParser(
        description="Generate visualizations for Grok-1 router analysis"
    )
    parser.add_argument(
        "--input",
        default="routers.npz",
        help="Path to extracted router weights",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory to save figures",
    )

    args = parser.parse_args()

    print(f"Loading router weights from: {args.input}")
    routers = load_router_weights(args.input)

    print("Running analysis...")
    results = run_full_analysis(routers)

    print("\nGenerating figures...")
    generate_all_figures(results, args.output_dir, routers)
