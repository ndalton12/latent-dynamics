from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from latent_dynamics.dayang.activations import pool_activations


def compute_layerwise_pca(
    activations_all: list[dict[str, Any]],
    pool_method: Literal["all", "first", "mid", "last", "mean"] = "all",
    include_bos: bool = False,
    num_components: int = 2,
) -> list[PCA]:
    """Compute PCA for each layer and plot explained variance ratio."""
    num_layers = activations_all[0]["activations"].shape[1]
    pcas = []
    explained_ratios = []

    for layer_idx in tqdm(range(num_layers), desc="Computing layer-wise PCA"):
        # Aggregate activations across all samples for the current layer
        activations_per_layer = []
        for sample in activations_all:
            activations = sample["activations"][:, layer_idx, :]
            activations_pooled = pool_activations(activations, pool_method, include_bos)
            activations_per_layer.append(activations_pooled)
        activations_per_layer = np.concatenate(activations_per_layer, axis=0)

        # Fit PCA for the current layer
        pca = PCA(n_components=num_components)
        pca.fit(activations_per_layer)
        pcas.append(pca)
        explained_ratios.append(pca.explained_variance_ratio_.sum())

    return pcas


def plot_layerwise_pca_ratio(pcas: list[PCA]):
    """Compute PCA for each layer and plot explained variance ratio."""
    explained_ratios = []

    for pca in pcas:
        explained_ratios.append(pca.explained_variance_ratio_)

    plt.figure(figsize=(6, 4))
    plt.stackplot(range(len(pcas)), np.array(explained_ratios).T)
    plt.ylim(0, 1)
    plt.title("Explained Variance Ratio per PC")
    plt.xlabel("Layer")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True, alpha=0.3)
    plt.show()
