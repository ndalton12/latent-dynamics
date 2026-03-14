#!/usr/bin/env python3
"""Publication-ready trajectory visualizations.

Run from repo root with project env (e.g. uv):

  uv run python scripts/visualize_trajectories.py --model qwen3_8b --activations activations/xstest/train/qwen3_8b/layer_24

Produces:
- UMAP of mean-pooled trajectories (safe/unsafe colored).
- Token-level single-trajectory paths (PCA → 2D/3D).
- Optional speed arrows overlaid on path plots.
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


def _sanitize_for_projection(values: np.ndarray, clip_value: float = 1e3) -> np.ndarray:
    """Keep projection numerics stable for noisy hidden-state tensors."""
    arr = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=clip_value, neginf=-clip_value)
    return np.clip(arr, -clip_value, clip_value)


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
    features = _sanitize_for_projection(features)
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
    path_dims: int = 2,
    show_arrows: bool = False,
    arrow_step: int = 1,
) -> None:
    if path_dims not in (2, 3):
        raise ValueError(f"path_dims must be 2 or 3, got {path_dims}")
    if arrow_step < 1:
        raise ValueError(f"arrow_step must be >= 1, got {arrow_step}")

    # Fit PCA on all token vectors.
    all_tokens = _sanitize_for_projection(np.concatenate(trajectories, axis=0))
    pca = PCA(n_components=path_dims)
    pca.fit(all_tokens)

    safe_idx = _pick_indices(labels, 0, max_per_class)
    unsafe_idx = _pick_indices(labels, 1, max_per_class)

    fig = go.Figure()

    def _add_arrows_2d(traj_proj: np.ndarray, color: str) -> None:
        if traj_proj.shape[0] < 2:
            return
        starts = traj_proj[:-1:arrow_step]
        deltas = np.diff(traj_proj, axis=0)[::arrow_step]
        if starts.size == 0:
            return
        scale = 0.35
        ends = starts + (scale * deltas)
        x_segments: list[float | None] = []
        y_segments: list[float | None] = []
        for s, e in zip(starts, ends):
            x_segments.extend([float(s[0]), float(e[0]), None])
            y_segments.extend([float(s[1]), float(e[1]), None])
        fig.add_trace(
            go.Scatter(
                x=x_segments,
                y=y_segments,
                mode="lines",
                line={"color": color, "width": 1},
                opacity=0.45,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    def _add_arrows_3d(traj_proj: np.ndarray, color: str) -> None:
        if traj_proj.shape[0] < 2:
            return
        starts = traj_proj[:-1:arrow_step]
        deltas = np.diff(traj_proj, axis=0)[::arrow_step]
        if starts.size == 0:
            return
        norms = np.linalg.norm(deltas, axis=1)
        max_norm = float(np.max(norms)) if norms.size else 0.0
        sizeref = max(max_norm * 0.25, 0.05)
        fig.add_trace(
            go.Cone(
                x=starts[:, 0],
                y=starts[:, 1],
                z=starts[:, 2],
                u=deltas[:, 0],
                v=deltas[:, 1],
                w=deltas[:, 2],
                anchor="tail",
                showscale=False,
                colorscale=[[0.0, color], [1.0, color]],
                cmin=0.0,
                cmax=1.0,
                sizemode="absolute",
                sizeref=sizeref,
                opacity=0.45,
                hoverinfo="skip",
            )
        )

    for i in safe_idx:
        traj_proj = pca.transform(_sanitize_for_projection(trajectories[i]))
        if path_dims == 2:
            fig.add_trace(
                go.Scatter(
                    x=traj_proj[:, 0],
                    y=traj_proj[:, 1],
                    mode="lines+markers",
                    line={"color": "seagreen"},
                    name=f"safe_{i}",
                    opacity=0.8,
                )
            )
            if show_arrows:
                _add_arrows_2d(traj_proj, "seagreen")
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=traj_proj[:, 0],
                    y=traj_proj[:, 1],
                    z=traj_proj[:, 2],
                    mode="lines+markers",
                    line={"color": "seagreen", "width": 4},
                    marker={"size": 3},
                    name=f"safe_{i}",
                    opacity=0.8,
                )
            )
            if show_arrows:
                _add_arrows_3d(traj_proj, "seagreen")

    for i in unsafe_idx:
        traj_proj = pca.transform(_sanitize_for_projection(trajectories[i]))
        if path_dims == 2:
            fig.add_trace(
                go.Scatter(
                    x=traj_proj[:, 0],
                    y=traj_proj[:, 1],
                    mode="lines+markers",
                    line={"color": "crimson"},
                    name=f"unsafe_{i}",
                    opacity=0.8,
                )
            )
            if show_arrows:
                _add_arrows_2d(traj_proj, "crimson")
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=traj_proj[:, 0],
                    y=traj_proj[:, 1],
                    z=traj_proj[:, 2],
                    mode="lines+markers",
                    line={"color": "crimson", "width": 4},
                    marker={"size": 3},
                    name=f"unsafe_{i}",
                    opacity=0.8,
                )
            )
            if show_arrows:
                _add_arrows_3d(traj_proj, "crimson")

    if path_dims == 2:
        fig.update_layout(
            title=title,
            xaxis_title="PCA dim 1",
            yaxis_title="PCA dim 2",
            template="plotly_white",
        )
    else:
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": "PCA dim 1",
                "yaxis_title": "PCA dim 2",
                "zaxis_title": "PCA dim 3",
            },
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
    parser.add_argument(
        "--path-dims",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of PCA dimensions for token-path plot (2 or 3).",
    )
    parser.add_argument(
        "--show-arrows",
        action="store_true",
        help="Overlay per-step velocity arrows on trajectory path plots.",
    )
    parser.add_argument(
        "--arrow-step",
        type=int,
        default=1,
        help="Subsample factor for arrows (1 = every step, 2 = every other step).",
    )
    parser.add_argument(
        "--max-paths-per-class",
        type=int,
        default=3,
        help="Max number of safe and unsafe trajectories to draw in path plots.",
    )
    args = parser.parse_args()
    if args.arrow_step < 1:
        parser.error("--arrow-step must be >= 1")
    if args.max_paths_per_class < 1:
        parser.error("--max-paths-per-class must be >= 1")

    trajectories, texts, labels, token_texts, _generated, cfg = load_activations(args.activations)
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

    # 2. Safe/unsafe token paths in PCA space (2D or 3D).
    path_name = "paths3d" if args.path_dims == 3 else "paths"
    paths_path = fig_dir / f"{path_name}_{model_key}_L{layer}"
    plot_trajectory_paths(
        trajectories,
        labels_arr,
        title=f"Example trajectories in PCA-{args.path_dims} space — {model_key}, layer {layer}",
        out_path=paths_path,
        max_per_class=args.max_paths_per_class,
        path_dims=args.path_dims,
        show_arrows=args.show_arrows,
        arrow_step=args.arrow_step,
    )
    arrow_text = " with per-step velocity arrows" if args.show_arrows else ""
    captions[paths_path.with_suffix(".png").name] = (
        f"Up to {args.max_paths_per_class} safe and {args.max_paths_per_class} unsafe token-level trajectories "
        f"projected into the first {args.path_dims} PCA components for {model_key} layer {layer}{arrow_text}."
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

