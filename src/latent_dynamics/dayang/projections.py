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
from sklearn.decomposition import PCA as _PCA
from tqdm.auto import tqdm

from latent_dynamics.dayang.activations import Activations, PoolMethod


class PCA(_PCA):
    """PCA subclass which preserves the origin.

    Standard PCA centers data by subtracting the mean `\\mu`, which shifts the
    global origin. This class restores the origin by subtracting the projected
    offset, effectively calculating :math:`X_{projected} = X V^T`, where `V` are
    the principal components derived from the centered data.
    This preserves angular relationships (cosine similarity) from the
    original space to the visualization.

    Attributes:
        origin_offset_ (ndarray): The projected coordinates of the
            high-dimensional origin in the reduced space.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        super().fit(X, y)
        self.origin_proj_ = super().transform(np.zeros((1, X.shape[1])))
        return self

    def transform(self, X, keep_origin: bool = True):
        X_proj = super().transform(X)
        if keep_origin:
            X_proj -= self.origin_proj_
        return X_proj


def choice(
    token_ids: list[int] | np.array,
    at_most: int | None = None,
    exactly: int | None = None,
    seed: int = 42,
) -> np.array:
    if at_most is not None and exactly is not None:
        raise ValueError("Cannot specify both 'at_most' and 'exactly'.")

    token_ids = np.asarray(token_ids)
    rng = np.random.default_rng(seed)
    if exactly is not None:
        if len(token_ids) <= exactly:
            return np.concatenate([token_ids, rng.choice(token_ids, size=exactly - len(token_ids), replace=True)])
        else:
            return rng.choice(token_ids, size=exactly, replace=False)
    elif at_most is not None:
        if len(token_ids) <= at_most:
            return token_ids
        else:
            return rng.choice(token_ids, size=at_most, replace=False)
    else:
        return token_ids


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


def get_tooltip_per_token(sample: dict, topk: int = 3, html: bool = False) -> list[str]:
    """Create tooltip texts for each token for a given sample."""
    text = []
    for i, (token, token_pos, topk_tokens, topk_probs) in enumerate(
        zip(sample["tokens"], sample["token_positions"], sample["topk"].tokens, sample["topk"].probs)
    ):
        text.append(
            f"ID: {sample['id']}"
            f"<br>Position: {i + 1}/{len(sample['tokens'])}"
            f"<br>Token: '{escape_token(token, html=html)}'"
            f"<br>Top-k: {topk_to_text(topk_tokens[:topk], topk_probs[:topk], html=html) if topk_tokens is not None else 'N/A'}"
            f"<br><extra>{tokens_to_text(sample['tokens_all'], highlight=token_pos, html=html)}</extra>"
        )
    return text


def get_tooltip_per_layer(sample: dict, i: int, k: int = 3, html: bool = False) -> list[str]:
    """Create tooltip texts for each layer for a given sample."""
    token = sample["tokens"][i]
    token_pos = sample["token_positions"][i]
    text = []
    for layer_idx, topk_tokens, topk_probs in zip(sample["layers"], sample["topk"].tokens[i], sample["topk"].probs[i]):
        text.append(
            f"ID: {sample['id']}"
            f"<br>Layer: {layer_idx}/{sample['activations'].shape[1] - 1}"
            f"<br>Position: {i + 1}/{len(sample['tokens'])}"
            f"<br>Token: '{escape_token(token, html=html)}'"
            f"<br>Top-k: {topk_to_text(topk_tokens[:k], topk_probs[:k], html=html) if topk_tokens is not None else 'N/A'}"
            f"<br><extra>{tokens_to_text(sample['tokens_all'], highlight=token_pos, html=html)}</extra>"
        )
    return text


def compute_pca_per_layer(
    activations: Activations,
    pool_method: PoolMethod = "last",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    num_components: int = 2,
) -> list[PCA]:
    """Compute PCA for each layer and plot explained variance ratio."""
    pcas = []
    explained_ratios = []

    for layer_idx in tqdm(activations.layers, desc="Computing PCA per layer"):
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


def plot_pca_ratio_per_layer(pcas: list[PCA]):
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


def plot_pca_per_layer(
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
        for i, (layer_idx, pca) in enumerate(zip(tqdm(activations.layers, desc="Plotting PCA per layer"), pcas)):
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

        for i, (layer_idx, pca) in enumerate(zip(tqdm(activations.layers, desc="Plotting PCA per layer"), pcas)):
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
                        hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{hovertext}",
                        hovertext=get_tooltip_per_token(sample, html=True),
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


def plot_pca_per_token(
    activations: Activations,
    sample_id: str,
    pool_method: PoolMethod = "all",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    ncols: int = 5,
    backend: Literal["matplotlib", "plotly"] = "plotly",
    tokens_embeddings: tuple[list[str], np.array] | None = None,
):
    # Compute PCA
    sample = activations.get_per_token(
        sample_ids=sample_id,
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )
    pca = PCA(n_components=2)
    pca.fit(np.concatenate(sample["activations"]))
    # Project token embeddings
    if tokens_embeddings is not None:
        tokens, embeddings = tokens_embeddings
        embeddings_proj = pca.transform(embeddings)

    nrows = math.ceil(len(sample["tokens"]) / ncols)
    if backend == "matplotlib":
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.6 * nrows), sharex=True, sharey=True, squeeze=False
        )
        axes = axes.flatten()
        for i, (acts, token, token_pos) in enumerate(
            zip(sample["activations"], sample["tokens"], sample["token_positions"])
        ):
            ax = axes[i]
            # Plot token embeddings
            if tokens_embeddings is not None:
                ax.scatter(embeddings_proj[:, 0], embeddings_proj[:, 1], alpha=0.5, color="gray", s=10)
            # Plot activations
            activations_proj = pca.transform(acts)
            ax.plot(
                activations_proj[:, 0],
                activations_proj[:, 1],
                "-o",
                markersize=3,
            )
            # Plot start and endpoint
            ax.plot(
                activations_proj[0, 0],
                activations_proj[0, 1],
                "o",
                markersize=6,
            )
            ax.plot(
                activations_proj[-1, 0],
                activations_proj[-1, 1],
                "^",
                markersize=6,
            )
            # Annotate with layer indices
            for layer_idx in range(sample["activations"].shape[1]):
                ax.annotate(
                    str(activations.layers[layer_idx]),
                    (activations_proj[layer_idx, 0], activations_proj[layer_idx, 1]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=6,
                )
            ax.set_title(f"{token_pos}: {escape_token(token)}")
    elif backend == "plotly":
        fig = go.Figure()
        # Plot token embeddings
        if tokens_embeddings is not None:
            fig.add_trace(
                go.Scatter(
                    x=embeddings_proj[:, 0],
                    y=embeddings_proj[:, 1],
                    mode="markers",
                    marker=dict(size=4, color="gray", opacity=0.5),
                    hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{hovertext}",
                    hovertext=[escape_token(token) for token in tokens],
                    name="Embeddings",
                )
            )
        # Plot activations
        for i, (acts, token, token_pos) in enumerate(
            zip(sample["activations"], sample["tokens"], sample["token_positions"])
        ):
            activations_proj = pca.transform(acts)
            fig.add_trace(
                go.Scatter(
                    x=activations_proj[:, 0],
                    y=activations_proj[:, 1],
                    mode="lines+markers+text",
                    marker=dict(size=6),
                    text=[str(layer) for layer in activations.layers],
                    textposition="top center",
                    textfont=dict(size=6),
                    hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{hovertext}",
                    hovertext=get_tooltip_per_layer(sample, i, html=True),
                    name=f"{token_pos}: {escape_token(token)}",
                )
            )
        fig.update_layout(
            title=f"PCA per token: {sample_id}",
            xaxis_title="PC1",
            yaxis_title="PC2",
            legend_title="Tokens",
            width=1000,
            height=800,
        )
        fig.show()


def plot_token_embeddings(
    model,
    tokenizer,
    token_groups: dict[str, list[int]],
    num_components: int = 5,
    num_samles_use: int | None = 1000,
    num_samples_show: int | None = None,
    backend: str = "plotly",
):
    # Compute PCA on a subset of token embeddings from each group
    indices = np.concatenate([choice(token_ids, exactly=num_samles_use) for token_ids in token_groups.values()])
    embeddings = model.get_input_embeddings().weight.cpu().float().numpy()
    pca = PCA(n_components=num_components)
    pca.fit(embeddings[indices])

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
        # Project token embeddings
        token_ids = choice(token_ids, at_most=num_samples_show)
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
