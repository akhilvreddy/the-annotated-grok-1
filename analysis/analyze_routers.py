"""
Analyze Grok-1 MoE router weights.

This module provides functions to analyze patterns in the router weights:
- Layer-to-layer similarity analysis
- Expert specialization across layers
- Dimensionality reduction for visualization
- Hierarchical clustering

Key research questions:
- Do adjacent layers have similar routing patterns?
- Do early vs late layers cluster separately?
- Does the same expert specialize in similar things across layers?
- Are some experts "broader" (activate for more diverse inputs)?
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def compute_layer_similarity(
    routers: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute pairwise similarity between router weight matrices across layers.

    Each layer's router is flattened to a vector of 6144*8 = 49,152 values,
    then we compute the similarity between all pairs of layers.

    Args:
        routers: Array of shape (64, 6144, 8) containing router weights
        metric: Distance metric - 'cosine', 'correlation', or 'euclidean'

    Returns:
        Similarity matrix of shape (64, 64)
    """
    num_layers = routers.shape[0]

    # Flatten each layer's router to a vector
    flat_routers = routers.reshape(num_layers, -1)

    if metric == "cosine":
        # Cosine similarity
        # Normalize each vector
        norms = np.linalg.norm(flat_routers, axis=1, keepdims=True)
        normalized = flat_routers / (norms + 1e-8)
        similarity = normalized @ normalized.T
    elif metric == "correlation":
        # Pearson correlation
        similarity = np.corrcoef(flat_routers)
    elif metric == "euclidean":
        # Convert euclidean distance to similarity
        distances = squareform(pdist(flat_routers, metric="euclidean"))
        # Convert to similarity (inverse, normalized)
        similarity = 1 / (1 + distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarity


def compute_hierarchical_clustering(
    routers: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Perform hierarchical clustering on router weights.

    Args:
        routers: Array of shape (64, 6144, 8) containing router weights
        method: Linkage method ('ward', 'complete', 'average', 'single')
        metric: Distance metric for pdist

    Returns:
        Linkage matrix for dendrogram plotting
    """
    num_layers = routers.shape[0]
    flat_routers = routers.reshape(num_layers, -1)

    # Compute linkage
    if method == "ward":
        # Ward's method requires euclidean distance
        linkage_matrix = linkage(flat_routers, method="ward")
    else:
        linkage_matrix = linkage(flat_routers, method=method, metric=metric)

    return linkage_matrix


def compute_expert_similarity_across_layers(
    routers: np.ndarray,
) -> np.ndarray:
    """
    Compute how similar each expert's "trigger pattern" is across layers.

    For each expert (0-7), extract its column (what input patterns activate it)
    across all 64 layers, then compute pairwise similarity between layers
    for that expert.

    Args:
        routers: Array of shape (64, 6144, 8) containing router weights

    Returns:
        Array of shape (8, 64, 64) - similarity matrix for each expert
    """
    num_layers, emb_dim, num_experts = routers.shape

    expert_similarities = np.zeros((num_experts, num_layers, num_layers))

    for expert_idx in range(num_experts):
        # Extract this expert's column from all layers: shape (64, 6144)
        expert_columns = routers[:, :, expert_idx]

        # Compute cosine similarity between layers for this expert
        norms = np.linalg.norm(expert_columns, axis=1, keepdims=True)
        normalized = expert_columns / (norms + 1e-8)
        similarity = normalized @ normalized.T

        expert_similarities[expert_idx] = similarity

    return expert_similarities


def compute_expert_variance(routers: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze expert "breadth" - which experts have broader vs narrower activation patterns.

    Computes several metrics:
    - Weight magnitude variance: High variance = selective, low = broad
    - Entropy of normalized weights: High entropy = responds to many inputs
    - L1/L2 ratio (sparsity indicator)

    Args:
        routers: Array of shape (64, 6144, 8) containing router weights

    Returns:
        Dictionary with various expert variance metrics
    """
    num_layers, emb_dim, num_experts = routers.shape

    metrics = {}

    # Per-expert statistics (averaged across layers)
    # Weight magnitude variance per expert column
    weight_vars = np.var(routers, axis=1)  # (64, 8)
    metrics["weight_variance_per_layer"] = weight_vars
    metrics["weight_variance_mean"] = np.mean(weight_vars, axis=0)  # (8,)

    # L2 norm of each expert column
    l2_norms = np.linalg.norm(routers, axis=1)  # (64, 8)
    metrics["l2_norm_per_layer"] = l2_norms
    metrics["l2_norm_mean"] = np.mean(l2_norms, axis=0)

    # L1/L2 ratio (sparsity measure)
    l1_norms = np.sum(np.abs(routers), axis=1)  # (64, 8)
    l1_l2_ratio = l1_norms / (l2_norms + 1e-8)
    metrics["l1_l2_ratio_per_layer"] = l1_l2_ratio
    metrics["l1_l2_ratio_mean"] = np.mean(l1_l2_ratio, axis=0)

    # How much each expert's pattern changes across layers
    # (variance of the expert column across the 64 layers)
    expert_cross_layer_var = np.var(routers, axis=0)  # (6144, 8)
    metrics["cross_layer_variance"] = np.mean(expert_cross_layer_var, axis=0)  # (8,)

    return metrics


def reduce_dimensions(
    routers: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Reduce dimensionality of router weights for visualization.

    Each layer's router (6144*8 = 49,152 dimensions) is projected to
    n_components dimensions.

    Args:
        routers: Array of shape (64, 6144, 8) containing router weights
        method: 'umap', 'tsne', or 'pca'
        n_components: Number of output dimensions (2 or 3)
        **kwargs: Additional arguments for the reduction method

    Returns:
        Array of shape (64, n_components)
    """
    num_layers = routers.shape[0]
    flat_routers = routers.reshape(num_layers, -1)

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flat_routers)

    if method == "pca":
        reducer = PCA(n_components=n_components, **kwargs)
        embedding = reducer.fit_transform(scaled)

    elif method == "tsne":
        # t-SNE defaults
        perplexity = kwargs.pop("perplexity", min(30, num_layers - 1))
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=kwargs.pop("random_state", 42),
            **kwargs,
        )
        embedding = reducer.fit_transform(scaled)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP not installed. Install with: pip install umap-learn"
            )

        n_neighbors = kwargs.pop("n_neighbors", min(15, num_layers - 1))
        min_dist = kwargs.pop("min_dist", 0.1)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=kwargs.pop("random_state", 42),
            **kwargs,
        )
        embedding = reducer.fit_transform(scaled)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap', 'tsne', or 'pca'")

    return embedding


def analyze_layer_transitions(routers: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analyze how routing patterns change from one layer to the next.

    Computes:
    - Cosine similarity between adjacent layers
    - Rate of change (gradient) across layers
    - Identifies "transition points" where routing changes abruptly

    Args:
        routers: Array of shape (64, 6144, 8) containing router weights

    Returns:
        Dictionary with transition analysis results
    """
    num_layers = routers.shape[0]
    flat_routers = routers.reshape(num_layers, -1)

    # Normalize for cosine similarity
    norms = np.linalg.norm(flat_routers, axis=1, keepdims=True)
    normalized = flat_routers / (norms + 1e-8)

    # Adjacent layer similarities
    adjacent_similarities = np.array([
        np.dot(normalized[i], normalized[i + 1])
        for i in range(num_layers - 1)
    ])

    # Find transition points (where similarity drops)
    similarity_changes = np.diff(adjacent_similarities)
    transition_indices = np.where(np.abs(similarity_changes) > np.std(similarity_changes) * 2)[0]

    # Rate of change (L2 distance between adjacent layers)
    changes = np.linalg.norm(flat_routers[1:] - flat_routers[:-1], axis=1)

    return {
        "adjacent_similarities": adjacent_similarities,
        "transition_indices": transition_indices + 1,  # +1 because diff reduces length
        "layer_changes": changes,
        "mean_similarity": np.mean(adjacent_similarities),
        "min_similarity": np.min(adjacent_similarities),
        "max_change_layer": np.argmax(changes),
    }


def compute_expert_correlation_matrix(routers: np.ndarray) -> np.ndarray:
    """
    Compute correlation between experts within each layer, then average.

    This shows which experts tend to have similar activation patterns
    (suggesting redundancy) vs which are complementary.

    Args:
        routers: Array of shape (64, 6144, 8) containing router weights

    Returns:
        Array of shape (8, 8) - average correlation between experts
    """
    num_layers, emb_dim, num_experts = routers.shape

    # Compute correlation matrix for each layer
    correlations = np.zeros((num_layers, num_experts, num_experts))

    for layer_idx in range(num_layers):
        layer_weights = routers[layer_idx]  # (6144, 8)
        correlations[layer_idx] = np.corrcoef(layer_weights.T)

    # Average across layers
    mean_correlation = np.mean(correlations, axis=0)

    return mean_correlation


def run_full_analysis(routers: np.ndarray, verbose: bool = True) -> Dict:
    """
    Run all analyses and return comprehensive results.

    Args:
        routers: Array of shape (64, 6144, 8) containing router weights
        verbose: Whether to print progress

    Returns:
        Dictionary containing all analysis results
    """
    results = {}

    if verbose:
        print("Computing layer similarity matrix...")
    results["layer_similarity"] = compute_layer_similarity(routers, metric="cosine")

    if verbose:
        print("Computing hierarchical clustering...")
    results["linkage"] = compute_hierarchical_clustering(routers)

    if verbose:
        print("Computing expert similarity across layers...")
    results["expert_similarity"] = compute_expert_similarity_across_layers(routers)

    if verbose:
        print("Computing expert variance metrics...")
    results["expert_variance"] = compute_expert_variance(routers)

    if verbose:
        print("Analyzing layer transitions...")
    results["transitions"] = analyze_layer_transitions(routers)

    if verbose:
        print("Computing expert correlation matrix...")
    results["expert_correlation"] = compute_expert_correlation_matrix(routers)

    if verbose:
        print("Reducing dimensions (PCA)...")
    results["pca_embedding"] = reduce_dimensions(routers, method="pca")

    if verbose:
        print("Reducing dimensions (t-SNE)...")
    results["tsne_embedding"] = reduce_dimensions(routers, method="tsne")

    try:
        if verbose:
            print("Reducing dimensions (UMAP)...")
        results["umap_embedding"] = reduce_dimensions(routers, method="umap")
    except ImportError:
        if verbose:
            print("UMAP not available, skipping...")
        results["umap_embedding"] = None

    if verbose:
        print("\nAnalysis complete!")
        print(f"Layer similarity range: [{results['layer_similarity'].min():.3f}, {results['layer_similarity'].max():.3f}]")
        print(f"Mean adjacent layer similarity: {results['transitions']['mean_similarity']:.3f}")
        print(f"Transition points: {results['transitions']['transition_indices']}")

    return results


if __name__ == "__main__":
    import argparse
    from extract_routers import load_router_weights

    parser = argparse.ArgumentParser(
        description="Analyze Grok-1 MoE router weights"
    )
    parser.add_argument(
        "--input",
        default="routers.npz",
        help="Path to extracted router weights",
    )
    parser.add_argument(
        "--output",
        default="analysis_results.npz",
        help="Path to save analysis results",
    )

    args = parser.parse_args()

    print(f"Loading router weights from: {args.input}")
    routers = load_router_weights(args.input)
    print(f"Router weights shape: {routers.shape}")

    results = run_full_analysis(routers)

    # Save results (convert nested dicts to flat structure)
    save_dict = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                save_dict[f"{key}_{subkey}"] = subvalue
        elif value is not None:
            save_dict[key] = value

    np.savez_compressed(args.output, **save_dict)
    print(f"\nSaved analysis results to: {args.output}")
