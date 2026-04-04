from __future__ import annotations

import math
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
from latent_dynamics.dayang.utils import escape_token, get_tooltips_per_token, get_tooltips_per_layer, select


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


def compute_pca_per_layer(
    activations: Activations,
    pool_method: PoolMethod = "last",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    num_components: int = 2,
) -> list[PCA]:
    """Compute PCA for each layer and plot explained variance ratio."""
    samples = activations.get(
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )

    pcas = []
    explained_ratios = []
    for i, layer_idx in enumerate(tqdm(activations.layers, desc="Computing PCA per layer")):
        # Aggregate activations across all samples for the current layer
        activations_per_layer = np.concatenate([sample["activations"][:, i] for sample in samples], axis=0)

        # Fit PCA for the current layer
        pca = PCA(n_components=num_components)
        pca.fit(activations_per_layer)
        pcas.append(pca)
        explained_ratios.append(pca.explained_variance_ratio_.sum())

    return pcas


def plot_pca_ratio_per_layer(pcas: list[PCA]):
    """Compute PCA for each layer and plot percentage of explained variance."""
    explained_ratios = [pca.explained_variance_ratio_ for pca in pcas]

    plt.figure(figsize=(6, 4))
    plt.stackplot(range(len(pcas)), np.array(explained_ratios).T)
    plt.ylim(0, 1)
    plt.title("Percentage of explained variance per PC")
    plt.xlabel("Layer")
    plt.ylabel("Percentage of explained variance")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_pca_per_layer(
    activations: Activations,
    pcas: list[PCA],
    pool_method: PoolMethod = "last",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    tokens_embeddings: tuple[list[str], np.array] | None = None,
    ncols: int = 5,
) -> None:
    """Visualize PCA projections per layer."""
    samples = activations.get(
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )

    num_layers = activations.num_layers
    nrows = math.ceil(num_layers / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"Layer {layer}" for layer in activations.layers],
        horizontal_spacing=0.16 / ncols,
        vertical_spacing=0.24 / nrows,
    )
    for i, (layer_idx, pca) in enumerate(zip(tqdm(activations.layers, desc="Plotting PCA per layer"), pcas)):
        row = (i // ncols) + 1
        col = (i % ncols) + 1

        if tokens_embeddings is not None and layer_idx in [0, max(activations.layers)]:
            # Project token embeddings
            tokens, embeddings = tokens_embeddings
            embeddings_proj = pca.transform(embeddings)
            # Plot token embeddings
            fig.add_trace(
                go.Scatter(
                    x=embeddings_proj[:, 0],
                    y=embeddings_proj[:, 1],
                    mode="markers",
                    marker=dict(size=4, color="gray", opacity=0.5),
                    hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{hovertext}",
                    hovertext=[escape_token(token) for token in tokens],
                    name="Token embeddings",
                ),
                row=row,
                col=col,
            )

        for sample in samples:
            # Project pooled activations onto the first 2 PCs
            activations_proj = pca.transform(sample["activations"][:, i])

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
                    line=dict(color=color, width=1),
                    opacity=alpha,
                    hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{hovertext}",
                    hovertext=get_tooltips_per_layer(sample, i, html=True),
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
                        opacity=0.5,
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    fig.update_xaxes(title_text="PC1", row=nrows)
    fig.update_yaxes(title_text="PC2", col=1)
    fig.update_layout(
        width=300 * ncols,
        height=300 * nrows,
        title_text="PCA per layer",
        showlegend=False,
    )
    fig.show()


def compute_pca_per_token(
    activations: Activations,
    pool_method: PoolMethod = "all",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    separate: bool = False,
    num_components: int = 2,
) -> list[PCA]:
    """Compute PCA for each token across all layers and samples."""
    samples = activations.get(
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )

    if separate:
        if pool_method == "all" and len(samples) > 1:
            raise ValueError(
                "Cannot use 'separate=True' with 'pool_method=all' and multiple samples"
                " since it requires token sequences of the same length."
            )

        num_tokens = samples[0]["activations"].shape[0]
        pcas = []
        for token_idx in tqdm(range(num_tokens), desc="Computing PCA per token"):
            acts = np.concatenate(
                [sample["activations"][token_idx] for sample in samples]
            )  # (num_samples, num_layers, hidden_size)
            acts = acts.reshape(-1, acts.shape[-1])  # (num_samples * num_layers, hidden_size)
            pca = PCA(n_components=num_components)
            pca.fit(acts)
            pcas.append(pca)
        return pcas
    else:
        acts = np.concatenate(
            [sample["activations"] for sample in samples]
        )  # (num_samples * num_tokens, num_layers, hidden_size)
        acts = acts.reshape(-1, acts.shape[-1])  # (num_samples * num_tokens * num_layers, hidden_size)
        pca = PCA(n_components=num_components)
        pca.fit(acts)
        return [pca]


def plot_pca_per_token(
    activations: Activations,
    pca: PCA,
    pool_method: PoolMethod = "all",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    tokens_embeddings: tuple[list[str], np.array] | None = None,
    ncols: int = 3,
    separate: bool = False,
    colorby: Literal["auto", "token", "sample", "is_safe"] | None = "auto",
    showlegend: Literal["auto"] | bool = "auto",
):
    pca = pca[0]
    samples = activations.get(
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )

    if colorby == "auto":
        if len(samples) == 1 and not separate:
            colorby = "token"
        elif len(samples) <= 5:
            colorby = "sample"
        else:
            colorby = "is_safe"
    if showlegend == "auto":
        showlegend = len(samples) <= 5

    # Project token embeddings
    if tokens_embeddings is not None:
        tokens, embeddings = tokens_embeddings
        embeddings_proj = pca.transform(embeddings)

    # Create figure
    if separate:
        if pool_method == "all" and len(samples) > 1:
            raise ValueError(
                "Cannot use 'separate=True' with 'pool_method=all' and multiple samples"
                " since it requires token sequences of the same length."
            )

        num_tokens = samples[0]["activations"].shape[0]
        nrows = math.ceil(num_tokens / ncols)
        subplot_titles = []
        for token_idx in range(num_tokens):
            tokens = list(set(escape_token(sample["tokens"][token_idx]) for sample in samples))
            subplot_titles.append(
                f"Tokens: {', '.join(tokens) if len(tokens) <= 3 else ', '.join(tokens[:2]) + ', ...'}"
            )
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.1 / ncols,
            vertical_spacing=0.15 / nrows,
            shared_xaxes="all",
            shared_yaxes="all",
        )
        fig.update_xaxes(showticklabels=True)
        fig.update_yaxes(showticklabels=True)
    else:
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
                name="Token embeddings",
            ),
            row="all" if separate else None,
            col="all" if separate else None,
        )
    # Plot activations
    colors = fig.layout.template.layout.colorway
    for sample_idx, sample in enumerate(samples):
        if colorby == "token":
            color = None
        elif colorby == "sample":
            color = colors[sample_idx % len(colors)]
        elif colorby == "is_safe":
            color = "green" if sample["is_safe"] else "red"
        else:
            raise ValueError(f"Invalid colorby value: {colorby}")

        # Plot activations per token
        for token_idx, (acts, token, token_pos) in enumerate(
            zip(sample["activations"], sample["tokens"], sample["token_positions"])
        ):
            # Project activations
            activations_proj = pca.transform(acts)
            # Plot activations in PCA space
            if separate:
                text = [str(layer) for layer in activations.layers]
                legend_group = sample["id"] if len(samples) > 1 else None
                legend_group_title = None
                trace_name = sample["id"] if token_idx == 0 else None
                show_legend = token_idx == 0
            else:
                text = [f"{token_pos + 1}.{layer}" for layer in activations.layers]
                legend_group = sample["id"] if len(samples) > 1 else None
                legend_group_title = sample["id"] if len(samples) > 1 else None
                trace_name = f"{token_pos + 1}: {escape_token(token)}"
                show_legend = True
            fig.add_trace(
                go.Scatter(
                    x=activations_proj[:, 0],
                    y=activations_proj[:, 1],
                    mode="lines+markers+text",
                    marker=dict(color=color, size=4),
                    line=dict(color=color, width=1),
                    text=text,
                    textposition="top center",
                    textfont=dict(size=6),
                    hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{hovertext}",
                    hovertext=get_tooltips_per_token(sample, token_idx, html=True),
                    legendgroup=legend_group,
                    legendgrouptitle_text=legend_group_title,
                    name=trace_name,
                    showlegend=show_legend,
                ),
                row=(token_idx // ncols) + 1 if separate else None,
                col=(token_idx % ncols) + 1 if separate else None,
            )

    fig.update_xaxes(title_text="PC1", row=nrows if separate else None)
    fig.update_yaxes(title_text="PC2", col=1 if separate else None)
    fig.update_layout(
        width=500 * ncols if separate else 1000,
        height=400 * nrows if separate else 800,
        title="PCA per token",
        legend=dict(groupclick="togglegroup" if separate else "toggleitem"),
        showlegend=showlegend,
    )
    fig.show()


def plot_token_embeddings(
    model,
    tokenizer,
    token_groups: dict[str, list[int]],
    num_components: int = 5,
    num_samles_use: int | None = 1000,
    num_samples_show: int | None = None,
):
    # Compute PCA on a subset of token embeddings from each group
    indices = np.concatenate([select(token_ids, exactly=num_samles_use) for token_ids in token_groups.values()])
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
        token_ids = select(token_ids, at_most=num_samples_show)
        embeddings_proj = pca.transform(embeddings[token_ids])
        # Aggregate data for plotting
        groups.extend([token_group] * len(token_ids))
        tokens.extend(map(escape_token, tokenizer.convert_ids_to_tokens(token_ids)))
        for i in range(num_components):
            pcs[f"PC{i + 1}"].extend(embeddings_proj[:, i])
    df = pd.DataFrame({"token_group": groups, "token": tokens, **pcs})

    # Plot pairplot of principal components
    fig = px.scatter_matrix(
        df,
        dimensions=[f"PC{i + 1}" for i in range(num_components)],
        color="token_group",
        hover_data=["token_group", "token"],
        opacity=0.5,
    )
    fig.update_traces(diagonal_visible=False, marker_size=3)
    fig.update_layout(title="PCA of token embeddings", width=200 * num_components, height=200 * num_components)
    fig.show()
