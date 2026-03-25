from __future__ import annotations

import html as _html
import textwrap

import numpy as np


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
    tokens_all = sample["tokens_all"]
    tokens = sample["tokens"]
    text = []
    for i, (token, token_pos, topk_tokens, topk_probs) in enumerate(
        zip(sample["tokens"], sample["token_positions"], sample["topk"].tokens, sample["topk"].probs)
    ):
        text.append(
            f"ID: {sample['id']}"
            f"<br>Position: {token_pos + 1}/{len(tokens_all)} ({i + 1}/{len(tokens)})"
            f"<br>Token: '{escape_token(token, html=html)}'"
            f"<br>Top-k: {topk_to_text(topk_tokens[:topk], topk_probs[:topk], html=html) if topk_tokens is not None else 'N/A'}"
            f"<br><extra>{tokens_to_text(tokens_all, highlight=token_pos, html=html)}</extra>"
        )
    return text


def get_tooltip_per_layer(sample: dict, i: int, k: int = 3, html: bool = False) -> list[str]:
    """Create tooltip texts for each layer for a given sample."""
    tokens_all = sample["tokens_all"]
    tokens = sample["tokens"]
    token = sample["tokens"][i]
    token_pos = sample["token_positions"][i]
    text = []
    for layer_idx, topk_tokens, topk_probs in zip(sample["layers"], sample["topk"].tokens[i], sample["topk"].probs[i]):
        text.append(
            f"ID: {sample['id']}"
            f"<br>Layer: {layer_idx}/{sample['activations'].shape[1] - 1}"
            f"<br>Position: {token_pos + 1}/{len(tokens_all)} ({i + 1}/{len(tokens)})"
            f"<br>Token: '{escape_token(token, html=html)}'"
            f"<br>Top-k: {topk_to_text(topk_tokens[:k], topk_probs[:k], html=html) if topk_tokens is not None else 'N/A'}"
            f"<br><extra>{tokens_to_text(tokens_all, highlight=token_pos, html=html)}</extra>"
        )
    return text
