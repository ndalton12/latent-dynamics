from __future__ import annotations

import html
import math
import textwrap
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from latent_dynamics.dayang.activations import Activations, PoolMethod


def tokens_to_text(tokens: list[str], highlight: int | None = None) -> str:
    tokens = ["<b>" + token + "</b>" if i == highlight else token for i, token in enumerate(tokens)]
    text = " ".join(tokens)
    return "<br>".join(textwrap.wrap(text, width=80))


def get_tooltip(sample: dict) -> list[str]:
    """Create tooltip text for a given sample."""
    tokens = [html.escape(token.replace("\n", "\\n")) for token in sample["tokens"]]
    tokens_all = [html.escape(token.replace("\n", "\\n")) for token in sample["tokens_all"]]
    text = [
        f"ID: {sample['id']}"
        f"<br>Position: {i + 1}/{len(sample['tokens'])}"
        f"<br>Token: '{token}'"
        f"<br><extra>{tokens_to_text(tokens_all, highlight=token_pos)}</extra>"
        for i, (token, token_pos) in enumerate(zip(tokens, sample["token_positions"]))
    ]
    return text


def compute_layerwise_pca(
    activations: Activations,
    pool_method: PoolMethod,
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    num_components: int = 2,
) -> list[PCA]:
    """Compute PCA for each layer and plot explained variance ratio."""
    pcas = []
    explained_ratios = []

    for layer_idx in tqdm(activations.layers, desc="Computing layer-wise PCA"):
        # Aggregate activations across all samples for the current layer
        samples = activations.get(
            layer_idx=layer_idx,
            pool_method=pool_method,
            exclude_bos=exclude_bos,
            exclude_special_tokens=exclude_special_tokens,
        )
        activations_per_layer = np.concatenate([sample["activations"] for sample in samples], axis=0)

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
    activations: Activations,
    pcas: list[PCA],
    pool_method: PoolMethod,
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    ncols: int = 5,
    backend: Literal["matplotlib", "plotly"] = "plotly",
) -> None:
    """Visualize PCA projections per layer."""
    num_layers = activations.num_layers
    nrows = math.ceil(num_layers / ncols)

    if backend == "matplotlib":
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
        axes = axes.flatten()
        for i, (layer_idx, pca) in enumerate(zip(tqdm(activations.layers, desc="Plotting layer-wise PCA"), pcas)):
            ax = axes[i]

            samples = activations.get(
                layer_idx=layer_idx,
                pool_method=pool_method,
                exclude_bos=exclude_bos,
                exclude_special_tokens=exclude_special_tokens,
            )
            for sample in samples:
                # Project pooled activations onto the first 2 PCs
                activations_proj = pca.transform(sample["activations"])

                # Plot the activations in the PCA space
                color = "tab:green" if sample["is_safe"] else "tab:red"
                symbol = "x-" if sample["is_adversarial"] else "o-"
                alpha = 0.5 if sample["is_adversarial"] else 0.25
                ax.plot(
                    activations_proj[:, 0],
                    activations_proj[:, 1],
                    symbol,
                    color=color,
                    alpha=alpha,
                    markersize=3,
                    linewidth=1.0,
                )

                # Plot start and endpoint if there are multiple activations
                if len(activations_proj) > 1:
                    ax.plot(
                        activations_proj[0, 0],
                        activations_proj[0, 1],
                        "o",
                        color=color,
                        markersize=6,
                    )
                    ax.plot(
                        activations_proj[-1, 0],
                        activations_proj[-1, 1],
                        "^",
                        color=color,
                        markersize=6,
                    )
                    # # Annotate with tokens
                    # if annotate_tokens:
                    #     tokens = sample["tokens"]
                    #     if layer_idx == 0:
                    #         print(tokens)
                    #     for token, (x, y) in zip(tokens, activations_proj):
                    #         ax.text(x, y, token, fontsize=6)

            ax.grid(True, alpha=0.25)
            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

        # Hide unused subplots
        for ax in axes[num_layers:]:
            ax.axis("off")

        fig.suptitle(f"PCA Projections (pool='{pool_method}')")
        fig.tight_layout()
        plt.show()
    elif backend == "plotly":
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f"Layer {layer}" for layer in activations.layers])

        for i, (layer_idx, pca) in enumerate(zip(tqdm(activations.layers, desc="Plotting layer-wise PCA"), pcas)):
            row = (i // ncols) + 1
            col = (i % ncols) + 1

            samples = activations.get(
                layer_idx=layer_idx,
                pool_method=pool_method,
                exclude_bos=exclude_bos,
                exclude_special_tokens=exclude_special_tokens,
            )
            for sample in samples:
                # Project pooled activations onto the first 2 PCs
                activations_proj = pca.transform(sample["activations"])

                # Plot the activations in the PCA space
                color = "green" if sample["is_safe"] else "red"
                symbol = "x" if sample["is_adversarial"] else "circle"
                alpha = 0.5 if sample["is_adversarial"] else 0.25
                fig.add_trace(
                    go.Scatter(
                        x=activations_proj[:, 0],
                        y=activations_proj[:, 1],
                        mode="lines+markers",
                        marker=dict(color=color, size=4, symbol=symbol),
                        line=dict(color=color, width=1.0),
                        opacity=alpha,
                        hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{text}",
                        text=get_tooltip(sample),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                # Plot start and endpoint if there are multiple activations
                if len(activations_proj) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=[activations_proj[0, 0], activations_proj[-1, 0]],
                            y=[activations_proj[0, 1], activations_proj[-1, 1]],
                            mode="markers",
                            marker=dict(color=color, size=6, symbol=["circle", "triangle-up"]),
                            hoverinfo="skip",
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
            title_text=f"PCA Projections (pool='{pool_method}')",
            showlegend=False,
            hovermode="closest",
        )
        fig.show()
    else:
        raise ValueError(f"Unknown backend: '{backend}'")
