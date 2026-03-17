from __future__ import annotations

import math
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from latent_dynamics.dayang.activations import pool_activations, pool_tokens


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


def plot_layerwise_pca(
    activations_all: list[dict[str, Any]],
    pcas: list[PCA],
    pool_method: Literal["all", "last", "mean"] = "all",
    include_bos: bool = False,
    ncols: int = 5,
    backend: Literal["matplotlib", "plotly"] = "plotly",
) -> None:
    """Visualize PCA projections per layer."""
    num_layers = len(pcas)
    nrows = math.ceil(num_layers / ncols)

    if backend == "matplotlib":
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
        axes = axes.flatten()
        for layer_idx, pca in enumerate(tqdm(pcas, desc="Plotting layer-wise PCA")):
            ax = axes[layer_idx]

            for sample in activations_all:
                # Pool activations for the current layer
                activations = sample["activations"][:, layer_idx, :]
                activations_pooled = pool_activations(activations, pool_method, include_bos)
                # Project pooled activations onto the first 2 PCs
                activations_proj = pca.transform(activations_pooled)

                # Filter for extreme outliers
                mask_outlier = np.any(np.abs(activations_proj) >= 10000, axis=1)
                activations_proj = activations_proj[~mask_outlier]
                if sum(mask_outlier) > 0:
                    print(
                        f"Warning: Sample {sample['id']} has {sum(mask_outlier)} outlier tokens in layer {layer_idx} that are excluded from the plot"
                    )

                # Plot the activations in the PCA space
                color = "tab:green" if sample["is_safe"] else "tab:red"
                ax.plot(
                    activations_proj[:, 0],
                    activations_proj[:, 1],
                    "o-" if not sample["is_adversarial"] else "x-",
                    color=color,
                    alpha=0.25 if not sample["is_adversarial"] else 0.5,
                    markersize=3,
                    linewidth=1.0,
                )

                if pool_method == "all":
                    # Highlight the first and last token
                    ax.plot(
                        activations_proj[0, 0],
                        activations_proj[0, 1],
                        "o",
                        color=color,
                        alpha=0.75,
                        markersize=6,
                    )
                    ax.plot(
                        activations_proj[-1, 0],
                        activations_proj[-1, 1],
                        "^",
                        color=color,
                        alpha=0.75,
                        markersize=6,
                    )
                    # # Annotate with tokens
                    # if annotate_tokens:
                    #     tokens = sample["tokens"][1:] if not include_bos else sample["tokens"]
                    #     if layer_idx == 0:
                    #         print(tokens)
                    #     for token, (x, y) in zip(tokens, activations_proj):
                    #         ax.text(x, y, token, fontsize=6)

            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        # Hide unused subplots
        for ax in axes[num_layers:]:
            ax.axis("off")

        fig.suptitle(f"PCA Projections (pool='{pool_method}', include_bos={include_bos})")
        fig.tight_layout()
        plt.show()
    elif backend == "plotly":
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f"Layer {i}" for i in range(num_layers)])

        for layer_idx, pca in enumerate(tqdm(pcas, desc="Plotting layer-wise PCA")):
            row = (layer_idx // ncols) + 1
            col = (layer_idx % ncols) + 1

            for sample in activations_all:
                # Pool activations for the current layer
                activations = sample["activations"][:, layer_idx, :]
                activations_pooled = pool_activations(activations, pool_method, include_bos)
                # Project pooled activations onto the first 2 PCs
                activations_proj = pca.transform(activations_pooled)

                # Pool tokens for annotation
                tokens = sample["tokens"]
                tokens_pooled = pool_tokens(tokens, pool_method, include_bos)

                def process_text(text):
                    import html
                    import textwrap

                    return "<br>".join(textwrap.wrap(html.escape(text), width=80))

                text = [
                    f"ID: {sample['id']}<br>Token: '{token}'<br><extra>{process_text(sample['text'])}</extra>"
                    for token in tokens_pooled
                ]

                color = "green" if sample["is_safe"] else "red"
                symbol = "x" if sample["is_adversarial"] else "circle"
                alpha = 0.5 if sample["is_adversarial"] else 0.25
                fig.add_trace(
                    go.Scatter(
                        x=activations_proj[:, 0],
                        y=activations_proj[:, 1],
                        mode="lines+markers",
                        marker=dict(color=color, symbol=symbol, size=4),
                        line=dict(color=color, width=1.0),
                        opacity=alpha,
                        hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>%{text}",
                        text=text,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(title_text="PC1", row=row, col=col)
            fig.update_yaxes(title_text="PC2", row=row, col=col)

        fig.update_layout(
            height=300 * nrows,
            width=300 * ncols,
            title_text=f"PCA Projections (pool='{pool_method}', include_bos={include_bos})",
            showlegend=False,
            hovermode="closest",
        )
        fig.show()
    else:
        raise ValueError(f"Unknown backend: '{backend}'")
