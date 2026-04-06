from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from matplotlib.colors import to_rgba

from latent_dynamics.dayang.activations import Activations, PoolMethod
from latent_dynamics.dayang.utils import (
    escape_token,
    get_tooltips_per_layer,
    get_tooltips_per_token,
    select,
    select_from_grid,
)
from latent_dynamics.dayang.analysis import ActivationReaders


def to_rgba_str(color, alpha=1.0):
    rgba = to_rgba(color, alpha)
    return f"rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})"


def plot_reader_statistics(readers: dict[int, ActivationReaders], xlabel: str = "Position"):
    """Plot explained variance and accuracy of readers."""
    if isinstance(readers, defaultdict):
        positions = ["all"]
    else:
        positions = list(readers.keys())
    explained_variance = np.stack([readers[i].explained_variance_ratio_ for i in positions], axis=1)
    accuracy = np.stack([readers[i].accuracy_ for i in positions], axis=1)
    labels = readers[positions[0]].labels

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), layout="constrained")

    # Plot explained variance
    ax = axes[0]
    if len(positions) > 1:
        ax.stackplot(positions, explained_variance, labels=labels)
    else:
        bottom = np.zeros_like(explained_variance[0])
        for y, label in zip(explained_variance, labels):
            ax.bar(positions, y, bottom=bottom, label=label)
            bottom += y
        ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 1)
    ax.set_title("Explained variance by reader")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage of explained variance")
    ax.legend()

    # Plot accuracy
    ax = axes[1]
    if len(positions) > 1:
        for y, label in zip(accuracy, labels):
            ax.plot(positions, y, label=label)
    else:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % len(cmap.colors)) for i in range(accuracy.shape[0])]
        ax.bar(labels, accuracy.squeeze(), color=colors)
        if len(labels) <= 5:
            center = (len(labels) - 1) / 2
            ax.set_xlim(center - 5 / 2, center + 5 / 2)

    ax.axhline(0.5, color="gray", linestyle="--", label="Random guessing")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy of reader")
    if len(positions) > 1:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.legend()

    plt.show()


def plot_per_layer(
    activations: Activations,
    readers: dict[int, ActivationReaders],
    pool_method: PoolMethod = "last",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    separate: bool = True,
    ncols: int = 5,
    share_axes: bool = True,
    color_by: Literal["auto", "layer", "sample", "is_safe"] = "auto",
    show_legend: Literal["auto"] | bool = "auto",
    token_embeddings: tuple[list[str], np.array] | None = None,
    token_embeddings_resolution: int = 50,
) -> None:
    """Visualize projections per layer."""
    samples = activations.get(
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )

    if color_by == "auto":
        if len(samples) == 1 and not separate:
            color_by = "layer"
        elif len(samples) <= 5:
            color_by = "sample"
        else:
            color_by = "is_safe"
    if show_legend == "auto":
        show_legend = len(samples) <= 5

    # Create figure
    num_layers = activations.num_layers
    nrows = math.ceil(num_layers / ncols)
    if separate:
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[f"Layer {layer}" for layer in activations.layers],
            horizontal_spacing=0.2 / ncols,
            vertical_spacing=0.2 / nrows,
            shared_xaxes="all" if share_axes else False,
            shared_yaxes="all" if share_axes else False,
        )
    else:
        fig = go.Figure()

    # Plot per layer
    colors = fig.layout.template.layout.colorway
    for layer_idx, layer in zip(tqdm(range(num_layers), desc="Plotting per layer"), activations.layers):
        row = (layer_idx // ncols) + 1
        col = (layer_idx % ncols) + 1

        if token_embeddings is not None and layer in [0, max(activations.layers)]:
            # Project token embeddings
            tokens, embeddings = token_embeddings
            embeddings_proj = readers[layer].transform(embeddings)
            embeddings_proj = select_from_grid(embeddings_proj, resolution=token_embeddings_resolution)

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
                row=row if separate else None,
                col=col if separate else None,
            )

        for sample_idx, sample in enumerate(samples):
            if color_by == "layer":
                color = colors[layer_idx % len(colors)]
                alpha = 1.0
            elif color_by == "sample":
                color = colors[sample_idx % len(colors)]
                alpha = 1.0
            elif color_by == "is_safe":
                color = "green" if sample["is_safe"] else "red"
                alpha = 0.25
            else:
                raise ValueError(f"Invalid colorby value: {color_by}")

            # Project activations
            acts = sample["activations"][:, layer_idx]
            acts_proj = readers[layer].transform(acts)

            # Plot activations
            symbol = "x" if sample["is_adversarial"] else "circle"
            if separate:
                trace_legendgroup = sample["id"] if len(samples) > 1 else None
                trace_legendgrouptitle = None
                trace_name = sample["id"]
                trace_showlegend = layer_idx == 0
            else:
                trace_legendgroup = sample["id"] if len(samples) > 1 else None
                trace_legendgrouptitle = sample["id"] if len(samples) > 1 else None
                trace_name = layer_idx
                trace_showlegend = True
            fig.add_trace(
                go.Scatter(
                    x=acts_proj[:, 0],
                    y=acts_proj[:, 1],
                    mode="lines+markers",
                    marker=dict(color=to_rgba_str(color, alpha), size=4, symbol=symbol),
                    line=dict(color=to_rgba_str(color, alpha), width=1),
                    hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{hovertext}",
                    hovertext=get_tooltips_per_layer(sample, layer_idx, html=True),
                    legendgroup=trace_legendgroup,
                    legendgrouptitle_text=trace_legendgrouptitle,
                    name=trace_name,
                    showlegend=trace_showlegend,
                ),
                row=row if separate else None,
                col=col if separate else None,
            )

            # Plot start and endpoint if there are multiple activations
            if len(acts_proj) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=[acts_proj[0, 0], acts_proj[-1, 0]],
                        y=[acts_proj[0, 1], acts_proj[-1, 1]],
                        mode="markers",
                        marker=dict(color=color, size=6, symbol=["circle", "triangle-up"]),
                        hoverinfo="skip",
                        legendgroup=trace_legendgroup,
                        legendgrouptitle_text=trace_legendgrouptitle,
                        name=trace_name,
                        showlegend=False,
                    ),
                    row=row if separate else None,
                    col=col if separate else None,
                )

    labels = readers[0].labels
    fig.update_xaxes(title_text=labels[0], row=nrows if separate else None)
    fig.update_yaxes(title_text=labels[1], col=1 if separate else None)
    fig.update_layout(
        width=300 * ncols + 150 if separate else 1000,
        height=300 * nrows + 100 if separate else 800,
        title_text="Analysis per layer",
        legend=dict(groupclick="togglegroup" if separate else "toggleitem"),
        showlegend=show_legend,
    )
    fig.show()


def plot_per_token(
    activations: Activations,
    readers: dict[int, ActivationReaders],
    pool_method: PoolMethod = "all",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    separate: bool = False,
    ncols: int = 5,
    share_axes: bool = True,
    color_by: Literal["auto", "token", "sample", "is_safe"] = "auto",
    show_legend: Literal["auto"] | bool = "auto",
    token_embeddings: tuple[list[str], np.array] | None = None,
    token_embeddings_resolution: int = 50,
):
    """Visualize projections per token."""
    samples = activations.get(
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )

    if color_by == "auto":
        if len(samples) == 1 and not separate:
            color_by = "token"
        elif len(samples) <= 5:
            color_by = "sample"
        else:
            color_by = "is_safe"
    if show_legend == "auto":
        show_legend = len(samples) <= 5

    # Create figure
    num_tokens = samples[0]["activations"].shape[0]
    nrows = math.ceil(num_tokens / ncols)
    if separate:
        if pool_method == "all" and len(samples) > 1:
            raise ValueError(
                "Cannot use 'separate=True' with 'pool_method=all' and multiple samples"
                " since it requires token sequences of the same length."
            )

        subplot_titles = []
        for token_idx in range(num_tokens):
            tokens = list(set(escape_token(sample["tokens"][token_idx]) for sample in samples))
            subplot_titles.append(
                f"{token_idx + 1}: {', '.join(tokens) if len(tokens) <= 2 else ', '.join(tokens[:2]) + ', ...'}"
            )
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.2 / ncols,
            vertical_spacing=0.2 / nrows,
            shared_xaxes="all" if share_axes else False,
            shared_yaxes="all" if share_axes else False,
        )
        fig.update_xaxes(showticklabels=True)
        fig.update_yaxes(showticklabels=True)
    else:
        fig = go.Figure()

    # Plot per token
    colors = fig.layout.template.layout.colorway
    for token_idx in tqdm(range(num_tokens), desc="Plotting per token"):
        row = (token_idx // ncols) + 1
        col = (token_idx % ncols) + 1

        for sample_idx, sample in enumerate(samples):
            if color_by == "token":
                color = colors[token_idx % len(colors)]
                alpha = 1.0
            elif color_by == "sample":
                color = colors[sample_idx % len(colors)]
                alpha = 1.0
            elif color_by == "is_safe":
                color = "green" if sample["is_safe"] else "red"
                alpha = 0.25
            else:
                raise ValueError(f"Invalid colorby value: {color_by}")

            if token_embeddings is not None and sample_idx == 0 and (token_idx == 0 or separate):
                # Project token embeddings
                tokens, embeddings = token_embeddings
                embeddings_proj = readers[token_idx].transform(embeddings)
                embeddings_proj = select_from_grid(embeddings_proj, resolution=token_embeddings_resolution)
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
                        showlegend=token_idx == 0,
                    ),
                    row=row if separate else None,
                    col=col if separate else None,
                )

            # Project activations
            acts = sample["activations"][token_idx]
            acts_proj = readers[token_idx].transform(acts)
            # Plot activations
            token = sample["tokens"][token_idx]
            token_pos = sample["token_positions"][token_idx]
            symbol = "x" if sample["is_adversarial"] else "circle"
            if separate:
                trace_text = [str(layer) for layer in activations.layers]
                trace_legendgroup = sample["id"] if len(samples) > 1 else None
                trace_legendgrouptitle = None
                trace_name = sample["id"]
                trace_showlegend = token_idx == 0
            else:
                trace_text = [f"{token_pos + 1}.{layer}" for layer in activations.layers]
                trace_legendgroup = sample["id"] if len(samples) > 1 else None
                trace_legendgrouptitle = sample["id"] if len(samples) > 1 else None
                trace_name = f"{token_pos + 1}: {escape_token(token)}"
                trace_showlegend = True
            fig.add_trace(
                go.Scatter(
                    x=acts_proj[:, 0],
                    y=acts_proj[:, 1],
                    mode="lines+markers+text",
                    marker=dict(color=to_rgba_str(color, alpha), size=4, symbol=symbol),
                    line=dict(color=to_rgba_str(color, alpha), width=1),
                    text=trace_text,
                    textposition="top center",
                    textfont=dict(size=6),
                    hovertemplate="(%{x:.2f}, %{y:.2f})<br>%{hovertext}",
                    hovertext=get_tooltips_per_token(sample, token_idx, html=True),
                    legendgroup=trace_legendgroup,
                    legendgrouptitle_text=trace_legendgrouptitle,
                    name=trace_name,
                    showlegend=trace_showlegend,
                ),
                row=row if separate else None,
                col=col if separate else None,
            )

    labels = readers[0].labels
    fig.update_xaxes(title_text=labels[0], row=nrows if separate else None)
    fig.update_yaxes(title_text=labels[1], col=1 if separate else None)
    fig.update_layout(
        width=300 * ncols + 150 if separate else 1000,
        height=300 * nrows + 100 if separate else 800,
        title="Analysis per token",
        legend=dict(groupclick="togglegroup" if separate else "toggleitem"),
        showlegend=show_legend,
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
