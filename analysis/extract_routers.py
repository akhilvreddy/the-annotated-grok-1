"""
Extract router weights from Grok-1 checkpoint.

This script loads the Grok-1 checkpoint and extracts only the MoE router weights
for all 64 layers, saving them to a compact numpy file for analysis.

Router weights have shape [6144, 8] (embedding_dim × num_experts) per layer.
Total extracted data: 64 layers × 6144 × 8 = ~3.1M parameters (~12MB in float32).
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import QuantizedWeight8bit


def load_checkpoint_tensors(
    checkpoint_dir: str,
    tensor_filter: Optional[callable] = None,
) -> Dict[str, np.ndarray]:
    """
    Load tensors from a Grok-1 checkpoint directory.

    The checkpoint stores tensors as individual pickle files with names like
    'tensor00000_000'. This function loads them and reconstructs the parameter
    dictionary structure.

    Args:
        checkpoint_dir: Path to the ckpt-0 directory
        tensor_filter: Optional function to filter which parameters to load
                      (takes parameter path string, returns bool)

    Returns:
        Dictionary mapping parameter paths to numpy arrays
    """
    checkpoint_dir = Path(checkpoint_dir)

    # The checkpoint format uses pickled tensor files
    # First, try to find tensor files
    tensor_files = sorted(checkpoint_dir.glob("tensor*_000"))

    if not tensor_files:
        raise FileNotFoundError(
            f"No tensor files found in {checkpoint_dir}. "
            "Make sure you've downloaded the checkpoint to checkpoints/ckpt-0/"
        )

    print(f"Found {len(tensor_files)} tensor shards in checkpoint")

    # Load tensors - they're stored as individual pickle files
    tensors = {}
    for tensor_file in tensor_files:
        with open(tensor_file, "rb") as f:
            data = pickle.load(f)
            # Data might be a dict or a raw array
            if isinstance(data, dict):
                tensors.update(data)
            else:
                # Store with filename as key
                tensors[tensor_file.stem] = data

    return tensors


def extract_router_weights_from_params(params: Dict[str, Any]) -> np.ndarray:
    """
    Extract router weights from a loaded parameter dictionary.

    Navigates the nested parameter structure to find router weights at:
    params['decoder_layer_{i}']['router']['w'] for i in 0..63

    Args:
        params: Nested parameter dictionary from checkpoint

    Returns:
        Array of shape (64, 6144, 8) containing all router weights
    """
    num_layers = 64
    router_weights = []

    for i in range(num_layers):
        layer_key = f"decoder_layer_{i}"
        if layer_key not in params:
            raise KeyError(f"Layer {layer_key} not found in params. Available keys: {list(params.keys())[:5]}...")

        layer_params = params[layer_key]

        # Navigate to router weights - might be under 'moe/router' or just 'router'
        if "router" in layer_params:
            router_w = layer_params["router"]["w"]
        elif "moe" in layer_params and "router" in layer_params["moe"]:
            router_w = layer_params["moe"]["router"]["w"]
        else:
            raise KeyError(f"Router not found in layer {i}. Keys: {list(layer_params.keys())}")

        # Handle QuantizedWeight8bit if present (though router weights are typically float32)
        if isinstance(router_w, QuantizedWeight8bit):
            router_w = router_w.weight * router_w.scales

        router_weights.append(np.array(router_w))

    # Stack into (64, 6144, 8) array
    stacked = np.stack(router_weights, axis=0)
    print(f"Extracted router weights with shape: {stacked.shape}")
    return stacked


def extract_router_weights(
    checkpoint_path: str = "checkpoints/ckpt-0",
    output_path: str = "routers.npz",
) -> np.ndarray:
    """
    Main function to extract router weights from Grok-1 checkpoint.

    Args:
        checkpoint_path: Path to the ckpt-0 directory
        output_path: Path to save the extracted weights

    Returns:
        Array of shape (64, 6144, 8) containing router weights
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Register QuantizedWeight8bit for pickle loading
    sys.modules['__main__'].QuantizedWeight8bit = QuantizedWeight8bit

    # Try to load the checkpoint structure
    checkpoint_dir = Path(checkpoint_path)

    # Look for the main tensor index/structure file
    # The checkpoint might have a structure file or we need to reconstruct from tensors

    # Method 1: Try loading individual tensor files and finding router weights
    tensor_files = sorted(checkpoint_dir.glob("tensor*_000"))

    if not tensor_files:
        raise FileNotFoundError(
            f"No tensor files found in {checkpoint_path}. "
            "Download checkpoint with: huggingface-cli download xai-org/grok-1 --repo-type model "
            "--include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False"
        )

    print(f"Found {len(tensor_files)} tensor files")

    # Load all tensors and reconstruct the structure
    # The tensors are numbered in flattened tree order
    all_tensors = []
    for i, tensor_file in enumerate(tensor_files):
        if i % 100 == 0:
            print(f"Loading tensor {i}/{len(tensor_files)}...")
        with open(tensor_file, "rb") as f:
            tensor = pickle.load(f)
            all_tensors.append((tensor_file.name, tensor))

    # Now we need to identify which tensors are router weights
    # Router weights should have shape (6144, 8)
    router_weights = []
    router_indices = []

    for idx, (name, tensor) in enumerate(all_tensors):
        # Get the actual array shape
        if isinstance(tensor, QuantizedWeight8bit):
            shape = tensor.weight.shape
        elif hasattr(tensor, 'shape'):
            shape = tensor.shape
        else:
            continue

        # Router weights have shape (6144, 8)
        if shape == (6144, 8):
            if isinstance(tensor, QuantizedWeight8bit):
                # Dequantize
                arr = np.array(tensor.weight) * np.array(tensor.scales)
            else:
                arr = np.array(tensor)
            router_weights.append(arr)
            router_indices.append(idx)
            print(f"Found router weight at index {idx}: shape {shape}")

    if len(router_weights) != 64:
        print(f"Warning: Expected 64 router weights, found {len(router_weights)}")
        print(f"Found at indices: {router_indices}")

    if not router_weights:
        raise ValueError(
            "No router weights found. The checkpoint structure may be different than expected. "
            "Router weights should have shape (6144, 8)."
        )

    # Stack into final array
    routers = np.stack(router_weights, axis=0).astype(np.float32)
    print(f"Final router weights shape: {routers.shape}")

    # Save to file
    np.savez_compressed(output_path, routers=routers)
    print(f"Saved router weights to: {output_path}")

    return routers


def load_router_weights(path: str = "routers.npz") -> np.ndarray:
    """
    Load previously extracted router weights from file.

    Args:
        path: Path to the .npz file

    Returns:
        Array of shape (64, 6144, 8) containing router weights
    """
    data = np.load(path)
    return data["routers"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract MoE router weights from Grok-1 checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ckpt-0",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output",
        default="routers.npz",
        help="Output path for extracted weights",
    )

    args = parser.parse_args()

    extract_router_weights(args.checkpoint, args.output)
