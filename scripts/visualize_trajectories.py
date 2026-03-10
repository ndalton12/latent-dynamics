#!/usr/bin/env python3
"""Publication-ready trajectory visualizations.

Run from repo root with project env (e.g. uv):

  uv run python scripts/visualize_trajectories.py --model qwen3_8b --activations activations/xstest/train/qwen3_8b/layer_24

Produces:
- UMAP of mean-pooled trajectories (safe/unsafe colored).
- 3 safe + 3 unsafe single-trajectory paths (PCA → 2D).
- Histograms of speed ||z_{t+1}-z_t|| and curvature.
- Captions JSON at figures/captions.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA

# Ensure package is on path (src layout)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_dynamics.hub import load_activations  # type: ignore[import]
from latent_dynamics.viz import _maybe_write_image  # type: ignore[import]

try:
    import umap  # type: ignore[import]

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

import plotly.graph_objects as go


def _mean_pool_trajectories(trajectories: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([traj.mean(axis=0) for traj in trajectories], axis=0)


def _compute_speed_and_curvature(
    trajectories: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    speeds: list[float] = []
    curvatures: list[float] = []

    for traj in trajectories:
        if traj.shape[0] < 3:
            continue
        diffs = np.diff(traj, axis=0)
        step_speeds = np.linalg.norm(diffs, axis=1)
        speeds.extend(step_speeds.tolist())

        v1 = diffs[:-1]
        v2 = diffs[1:]
        num = (v1 * v2).sum(axis=1)
        denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
        cos_theta = np.clip(num / denom, -1.0, 1.0)
        angles = np.arccos(cos_theta)
        curvatures.extend(angles.tolist())

    return np.array(speeds, dtype=np.float32), np.array(curvatures, dtype=np.float32)


def _pick_indices(labels: np.ndarray, value: int, k: int) -> list[int]:
    idx = np.where(labels == value)[0].tolist()
    return idx[:k]


def plot_umap(
    features: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
        )
        emb = reducer.fit_transform(features)
    else:
        # Fallback: PCA-only if UMAP is unavailable.
        pca = PCA(n_components=2)
        emb = pca.fit_transform(features)

    fig = go.Figure()
    for value, name, color in [(0, "safe", "seagreen"), (1, "unsafe", "crimson")]:
        mask = labels == value
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=emb[mask, 0],
                y=emb[mask, 1],
                mode="markers",
                marker={"color": color, "size": 6, "opacity": 0.8},
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="dim 1",
        yaxis_title="dim 2",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def plot_trajectory_paths(
    trajectories: Sequence[np.ndarray],
    labels: np.ndarray,
    title: str,
    out_path: Path,
    max_per_class: int = 3,
) -> None:
    # Fit PCA on all token vectors.
    all_tokens = np.concatenate(trajectories, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_tokens)

    safe_idx = _pick_indices(labels, 0, max_per_class)
    unsafe_idx = _pick_indices(labels, 1, max_per_class)

    fig = go.Figure()
    for i in safe_idx:
        traj_2d = pca.transform(trajectories[i])
        fig.add_trace(
            go.Scatter(
                x=traj_2d[:, 0],
                y=traj_2d[:, 1],
                mode="lines+markers",
                line={"color": "seagreen"},
                name=f"safe_{i}",
                opacity=0.8,
            )
        )
    for i in unsafe_idx:
        traj_2d = pca.transform(trajectories[i])
        fig.add_trace(
            go.Scatter(
                x=traj_2d[:, 0],
                y=traj_2d[:, 1],
                mode="lines+markers",
                line={"color": "crimson"},
                name=f"unsafe_{i}",
                opacity=0.8,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="PCA dim 1",
        yaxis_title="PCA dim 2",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def plot_histogram(
    values: np.ndarray,
    title: str,
    x_label: str,
    out_path: Path,
    bins: int = 50,
) -> None:
    if values.size == 0:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values.tolist(),
            nbinsx=bins,
            marker={"color": "steelblue"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Count",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize latent trajectories (UMAP, paths, dynamics histograms)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key (used only for naming).",
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to activations leaf directory (with metadata.json & trajectories.safetensors).",
    )
    parser.add_argument(
        "--fig_dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures and captions.json.",
    )
    args = parser.parse_args()

    trajectories, texts, labels, token_texts, cfg = load_activations(args.activations)
    labels_arr = labels if labels is not None else np.zeros(len(trajectories), dtype=np.int64)

    model_key = cfg.model_key
    layer = cfg.layer_idx
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    captions: dict[str, str] = {}

    # 1. UMAP of mean-pooled trajectories.
    pooled = _mean_pool_trajectories(trajectories)
    umap_path = fig_dir / f"umap_{model_key}_L{layer}"
    plot_umap(
        pooled,
        labels_arr,
        title=f"UMAP of mean-pooled trajectories — {model_key}, layer {layer}",
        out_path=umap_path,
    )
    captions[umap_path.with_suffix(".png").name] = (
        f"UMAP of mean-pooled hidden states for {model_key} layer {layer}, colored by safe (0) vs unsafe (1)."
    )

    # 2. 3 safe + 3 unsafe paths in PCA space.
    paths_path = fig_dir / f"paths_{model_key}_L{layer}"
    plot_trajectory_paths(
        trajectories,
        labels_arr,
        title=f"Example trajectories in PCA space — {model_key}, layer {layer}",
        out_path=paths_path,
        max_per_class=3,
    )
    captions[paths_path.with_suffix(".png").name] = (
        f"Three safe and three unsafe token-level trajectories projected into the first two PCA components "
        f"for {model_key} layer {layer}."
    )

    # 3. Speed and curvature histograms.
    speeds, curvatures = _compute_speed_and_curvature(trajectories)
    speed_path = fig_dir / f"speed_hist_{model_key}_L{layer}"
    plot_histogram(
        speeds,
        title=f"Speed histogram — {model_key}, layer {layer}",
        x_label="||z_{t+1} - z_t||",
        out_path=speed_path,
    )
    captions[speed_path.with_suffix(".png").name] = (
        f"Histogram of per-token speed ||z_(t+1) - z_t|| for {model_key} layer {layer}."
    )

    curvature_path = fig_dir / f"curvature_hist_{model_key}_L{layer}"
    plot_histogram(
        curvatures,
        title=f"Curvature histogram — {model_key}, layer {layer}",
        x_label="Angle between successive steps (radians)",
        out_path=curvature_path,
    )
    captions[curvature_path.with_suffix(".png").name] = (
        f"Histogram of per-step curvature (angle between successive step vectors) for {model_key} layer {layer}."
    )

    # Note: PaCE support-set churn plots will be added once sparse codes are integrated.

    captions_path = fig_dir / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2))
    print(f"Wrote captions to {captions_path}")


if __name__ == "__main__":
    main()

