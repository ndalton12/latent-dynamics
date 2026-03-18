from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from latent_dynamics.dayang.activations import Activations, PoolMethod


class Reader(ABC):
    """Base class for concept-vector readers."""

    @abstractmethod
    def train(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Extract concept representation from activations."""

    @abstractmethod
    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Predict concept score from activations."""


class DifferenceInMeanReader(Reader):
    """Concept direction is the difference in means between safe and unsafe examples."""

    def __init__(self) -> None:
        self.direction: np.ndarray | None = None

    def train(self, activations: np.ndarray, labels: np.ndarray) -> None:
        safe_mean = activations[labels == 1].mean(axis=0)
        unsafe_mean = activations[labels == 0].mean(axis=0)
        direction = safe_mean - unsafe_mean
        self.direction = direction / np.linalg.norm(direction)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        if self.direction is None:
            raise RuntimeError("Reader must be trained before calling predict.")
        return activations @ self.direction


class LinearProbe(Reader):
    """Concept direction is the normal vector of a trained logistic regression model."""

    def __init__(self, max_iter: int = 1000) -> None:
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=42),
                ),
            ]
        )

    def train(self, activations: np.ndarray, labels: np.ndarray) -> None:
        self.model.fit(activations, labels)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        return self.model.decision_function(activations).astype(np.float32)


def get_reader(method: str, **kwargs) -> Reader:
    if method == "difference_in_mean":
        return DifferenceInMeanReader(**kwargs)
    elif method == "linear_probe":
        return LinearProbe(**kwargs)
    else:
        raise ValueError(f"Unknown reader method: {method}")


def compute_layerwise_score(
    activations: Activations,
    method: str = "difference_in_mean",
    pool_method: PoolMethod = "all",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
) -> list[Reader]:
    """Train a given reader per layer on activations, differentiating safe vs. unsafe inputs."""
    readers = []

    for layer_idx in tqdm(activations.layers, desc="Training layer-wise reader"):
        # Aggregate activations and labels across all samples for the current layer
        samples = activations.get(
            layer_idx=layer_idx,
            pool_method=pool_method,
            exclude_bos=exclude_bos,
            exclude_special_tokens=exclude_special_tokens,
        )

        activations_per_layer = []
        labels_per_layer = []
        for sample in samples:
            acts = sample["activations"]
            activations_per_layer.append(acts)
            labels_per_layer.extend([int(sample["is_safe"])] * len(acts))

        activations_per_layer = np.concatenate(activations_per_layer, axis=0)
        labels_per_layer = np.array(labels_per_layer)

        # Fit reader for the current layer
        layer_reader = get_reader(method)
        layer_reader.train(activations_per_layer, labels_per_layer)
        readers.append(layer_reader)

    return readers


def plot_layerwise_score(
    activations: Activations,
    readers: list[Reader],
    pool_method: PoolMethod,
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    ncols: int = 5,
    backend: Literal["plotly"] = "plotly",
) -> None:
    """Visualize concept scores per layer."""
    if backend != "plotly":
        raise NotImplementedError("Only plotly backend is supported for now.")

    num_layers = len(readers)
    nrows = math.ceil(num_layers / ncols)

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f"Layer {layer}" for layer in activations.layers])

    import html
    import textwrap

    def escape_tokens(tokens: list[str]) -> list[str]:
        return [html.escape(token.replace("\n", "\\n")) for token in tokens]

    def process_tokens(tokens: list[str], highlight: int | None = None) -> str:
        tokens = ["<b>" + token + "</b>" if i == highlight else token for i, token in enumerate(tokens)]
        text = " ".join(tokens)
        return "<br>".join(textwrap.wrap(text, width=80))

    for i, (layer_idx, reader) in enumerate(zip(tqdm(activations.layers, desc="Plotting layer-wise scores"), readers)):
        row = (i // ncols) + 1
        col = (i % ncols) + 1

        samples = activations.get(
            layer_idx=layer_idx,
            pool_method=pool_method,
            exclude_bos=exclude_bos,
            exclude_special_tokens=exclude_special_tokens,
        )
        for sample in samples:
            # Predict scores using the reader
            scores = reader.predict(sample["activations"])

            # Create text for tooltip
            tokens = escape_tokens(sample["tokens"])
            tokens_all = escape_tokens(sample["tokens_all"])
            text = [
                f"ID: {sample['id']}"
                f"<br>Position: {i + 1}/{len(sample['tokens'])}"
                f"<br>Token: '{token}'"
                f"<br><extra>{process_tokens(tokens_all, highlight=token_pos)}</extra>"
                for i, (token, token_pos) in enumerate(zip(tokens, sample["token_positions"]))
            ]

            color = "green" if sample["is_safe"] else "red"
            symbol = "x" if sample["is_adversarial"] else "circle"
            alpha = 0.5 if sample["is_adversarial"] else 0.25

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(scores))),
                    y=scores,
                    mode="lines+markers",
                    marker=dict(color=color, symbol=symbol, size=4),
                    line=dict(color=color, width=1.0),
                    opacity=alpha,
                    hovertemplate="Token Pos: %{x}<br>Score: %{y:.2f}<br>%{text}",
                    text=text,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Token Position", row=row, col=col)
        fig.update_yaxes(title_text="Score", row=row, col=col)

    fig.update_layout(
        height=300 * nrows,
        width=300 * ncols,
        title_text=f"Layer-wise Concept Scores (pool='{pool_method}')",
        showlegend=False,
        hovermode="closest",
    )
    fig.show()
