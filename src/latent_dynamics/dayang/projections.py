from __future__ import annotations

import html as _html
import math
import textwrap
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from latent_dynamics.dayang.activations import Activations, PoolMethod


def escape_token(token: str, html: bool = False) -> str:
    if html:
        token = _html.escape(token)
    token = token if token.isprintable() else ascii(token)[1:-1]
    return token


def tokens_to_text(tokens: list[str], highlight: int | None = None, html: bool = False) -> str:
    tokens = [escape_token(t, html=html) for t in tokens]
    tokens = ["<b>" + t + "</b>" if i == highlight else t for i, t in enumerate(tokens)]
    text = " ".join(tokens)
    return "<br>".join(textwrap.wrap(text, width=80))


def topk_to_text(topk_tokens: list[str], topk_probs: list[float], html: bool = False) -> str:
    return ", ".join(f"'{escape_token(t, html=html)}' ({p:.2f})" for t, p in zip(topk_tokens, topk_probs))


def get_tooltip(sample: dict, topk: int = 3, html: bool = False) -> list[str]:
    """Create tooltip text for a given sample."""
    text = []
    for i, (token, token_pos, topk_tokens, topk_probs) in enumerate(
        zip(sample["tokens"], sample["token_positions"], sample["topk_tokens"], sample["topk_probs"])
    ):
        text.append(
            f"ID: {sample['id']}"
            f"<br>Position: {i + 1}/{len(sample['tokens'])}"
            f"<br>Token: '{escape_token(token, html=html)}'"
            f"<br>Top-k: {topk_to_text(topk_tokens[:topk], topk_probs[:topk], html=html) if topk_tokens is not None else 'N/A'}"
            f"<br><extra>{tokens_to_text(sample['tokens_all'], highlight=token_pos, html=html)}</extra>"
        )
    return text


def compute_layerwise_pca(
    activations: Activations,
    pool_method: PoolMethod = "last",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    num_components: int = 2,
) -> list[PCA]:
    """Compute PCA for each layer and plot explained variance ratio."""
    pcas = []
    explained_ratios = []

    for layer_idx in tqdm(activations.layers, desc="Computing layer-wise PCA"):
        # Aggregate activations across all samples for the current layer
        samples = activations.get_per_layer(
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
    pool_method: PoolMethod = "last",
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

            samples = activations.get_per_layer(
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

            samples = activations.get_per_layer(
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
                        text=get_tooltip(sample, html=True),
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


def plot_token_embeddings(
    model,
    tokenizer,
    token_groups: dict[str, list[int]],
    num_components: int = 5,
    num_samles_use: int = 500,
    num_samples_show: int = 1000,
    backend: str = "plotly",
):
    rng = np.random.default_rng(seed=0)

    def sample(token_ids, n):
        return rng.choice(token_ids, size=n, replace=len(token_ids) < n)

    # Compute PCA on a subset of token embeddings from each group
    indices = np.concatenate([sample(token_ids, n=num_samles_use) for token_ids in token_groups.values()])
    embeddings = model.get_input_embeddings().weight.float().cpu().numpy()
    embeddings = embeddings[indices]
    pca = PCA(n_components=num_components)
    pca.fit(embeddings)

    # Plot cumulative explained variance ratio
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.title("Cumulative explained variance ratio")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.show()

    # Create dataframe for plotting
    groups = []
    tokens = []
    pcs = {f"PC{i + 1}": [] for i in range(num_components)}
    for token_group, token_ids in token_groups.items():
        # Transform a subset of token embeddings from each group
        if len(token_ids) > num_samples_show:
            token_ids = sample(token_ids, n=num_samples_show)
        embeddings_proj = pca.transform(embeddings[token_ids])
        # Aggregate data for plotting
        groups.extend([token_group] * len(token_ids))
        tokens.extend(map(escape_token, tokenizer.convert_ids_to_tokens(token_ids)))
        for i in range(num_components):
            pcs[f"PC{i + 1}"].extend(embeddings_proj[:, i])
    df = pd.DataFrame({"token_group": groups, "token": tokens, **pcs})

    # Plot pairplot of principal components
    if backend == "seaborn":
        import seaborn as sns

        sns.pairplot(df, hue="token_group", plot_kws=dict(s=10, alpha=0.75))
    elif backend == "plotly":
        fig = px.scatter_matrix(
            df,
            dimensions=[f"PC{i + 1}" for i in range(num_components)],
            color="token_group",
            hover_data="token",
            opacity=0.5,
        )
        fig.update_traces(diagonal_visible=False, marker_size=3)
        fig.update_layout(title="PCA of token embeddings", width=200 * num_components, height=200 * num_components)
        fig.show()
    else:
        raise ValueError(f"Unknown backend: '{backend}'")
