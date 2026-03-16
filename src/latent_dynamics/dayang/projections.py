from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from latent_dynamics.dayang.activations import pool_activations


def analyze_layerwise_pca(
    activations_all: list[dict[str, Any]],
    pool_method: Literal["all", "first", "mid", "last", "mean"] = "all",
    include_bos: bool = False,
) -> list[PCA]:
    """Compute PCA for each layer and plot explained variance ratio."""
    num_layers = activations_all[0]["activations"].shape[1]
    pcas = []
    explained_ratios = []

    for layer_idx in tqdm(range(num_layers), desc="Computing layer-wise PCA"):
        # Pool activations for the current layer and aggregate across all samples
        activations_per_layer = []
        for sample in activations_all:
            activations = sample["activations"][:, layer_idx, :]
            activations_pooled = pool_activations(activations, pool_method, include_bos)
            activations_per_layer.append(activations_pooled)
        activations_per_layer = np.concatenate(activations_per_layer, axis=0)

        # Fit PCA for the current layer
        pca = PCA(n_components=20)
        pca.fit(activations_per_layer)
        pcas.append(pca)
        explained_ratios.append(pca.explained_variance_ratio_)

    plt.figure(figsize=(6, 4))
    plt.stackplot(range(num_layers), np.array(explained_ratios).T)
    plt.ylim(0, 1)
    plt.title(f"Explained Variance Ratio per PC (pool='{pool_method}', include_bos={include_bos})")
    plt.xlabel("Layer")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True, alpha=0.3)
    plt.show()


def compute_layerwise_pca(
    activations_all: list[dict[str, Any]],
    pool_method: Literal["all", "first", "mid", "last", "mean"] = "all",
    include_bos: bool = False,
) -> list[PCA]:
    """Compute PCA for each layer and plot explained variance ratio."""
    num_layers = activations_all[0]["activations"].shape[1]
    pcas = []
    explained_ratios = []

    for layer_idx in tqdm(range(num_layers), desc="Computing layer-wise PCA"):
        # Pool activations for the current layer and aggregate across all samples
        activations_per_layer = []
        for sample in activations_all:
            activations = sample["activations"][:, layer_idx, :]
            activations_pooled = pool_activations(activations, pool_method, include_bos)
            activations_per_layer.append(activations_pooled)
        activations_per_layer = np.concatenate(activations_per_layer, axis=0)

        # Fit PCA for the current layer
        pca = PCA(n_components=2)
        pca.fit(activations_per_layer)
        pcas.append(pca)
        explained_ratios.append(pca.explained_variance_ratio_.sum())

    plt.figure(figsize=(6, 4))
    plt.plot(range(num_layers), explained_ratios, marker="o")
    plt.ylim(0, 1)
    plt.title(f"Explained Variance Ratio of first 2 PCs (pool='{pool_method}', include_bos={include_bos})")
    plt.xlabel("Layer")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True, alpha=0.3)
    plt.show()

    return pcas
