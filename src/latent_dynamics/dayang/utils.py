from __future__ import annotations

import html as _html
import textwrap

import numpy as np
import pandas as pd


def select(
    array: list[int] | np.array,
    at_most: int | float | None = None,
    exactly: int | float | None = None,
    seed: int = 42,
) -> np.array:
    if at_most is not None and exactly is not None:
        raise ValueError("Cannot specify both 'at_most' and 'exactly'.")

    array = np.asarray(array)
    rng = np.random.default_rng(seed)
    if exactly is not None:
        if isinstance(exactly, float):
            exactly = int(len(array) * exactly)

        if len(array) <= exactly:
            return np.concatenate([array, rng.choice(array, size=exactly - len(array), replace=True)])
        else:
            return rng.choice(array, size=exactly, replace=False)
    elif at_most is not None:
        if isinstance(at_most, float):
            at_most = int(len(array) * at_most)

        if len(array) <= at_most:
            return array
        else:
            return rng.choice(array, size=at_most, replace=False)
    else:
        return array


def select_from_grid(points: np.ndarray, resolution: int | None = 100, samples_per_bin: int = 1):
    if resolution is None:
        return points

    x_bins = np.linspace(points[:, 0].min(), points[:, 0].max(), resolution)
    y_bins = np.linspace(points[:, 1].min(), points[:, 1].max(), resolution)

    x_indices = np.digitize(points[:, 0], x_bins)
    y_indices = np.digitize(points[:, 1], y_bins)

    df_points = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "x_bin": x_indices, "y_bin": y_indices})
    df_points_sampled = df_points.groupby(["x_bin", "y_bin"]).head(samples_per_bin)
    points_sampled = df_points_sampled[["x", "y"]].values

    return points_sampled


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


def get_tooltips_per_layer(sample: dict, layer_idx: int, topk: int = 3, html: bool = False) -> list[str]:
    """Create tooltip texts across tokens for a given sample and layer index."""
    tokens_all = sample["tokens_all"]
    tokens = sample["tokens"]
    text = []
    for token_idx, (token, token_pos, topk_tokens, topk_probs) in enumerate(
        zip(
            sample["tokens"],
            sample["token_positions"],
            sample["topk"].tokens[:, layer_idx],
            sample["topk"].probs[:, layer_idx],
        )
    ):
        text.append(
            f"ID: {sample['id']}"
            f"<br>Position: {token_pos + 1}/{len(tokens_all)} ({token_idx + 1}/{len(tokens)})"
            f"<br>Token: '{escape_token(token, html=html)}'"
            f"<br>Top-k: {topk_to_text(topk_tokens[:topk], topk_probs[:topk], html=html) if topk_tokens is not None else 'N/A'}"
            f"<br><extra>{tokens_to_text(tokens_all, highlight=token_pos, html=html)}</extra>"
        )
    return text


def get_tooltips_per_token(sample: dict, token_idx: int, k: int = 3, html: bool = False) -> list[str]:
    """Create tooltip texts across layers for a given sample and token index."""
    tokens_all = sample["tokens_all"]
    tokens = sample["tokens"]
    token = sample["tokens"][token_idx]
    token_pos = sample["token_positions"][token_idx]
    text = []
    for layer_idx, topk_tokens, topk_probs in zip(
        sample["layers"], sample["topk"].tokens[token_idx], sample["topk"].probs[token_idx]
    ):
        text.append(
            f"ID: {sample['id']}"
            f"<br>Layer: {layer_idx}/{sample['activations'].shape[1] - 1}"
            f"<br>Position: {token_pos + 1}/{len(tokens_all)} ({token_idx + 1}/{len(tokens)})"
            f"<br>Token: '{escape_token(token, html=html)}'"
            f"<br>Top-k: {topk_to_text(topk_tokens[:k], topk_probs[:k], html=html) if topk_tokens is not None else 'N/A'}"
            f"<br><extra>{tokens_to_text(tokens_all, highlight=token_pos, html=html)}</extra>"
        )
    return text
