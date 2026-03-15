from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from latent_dynamics.hub import activation_subpath, load_activations, pull_from_hub
from latent_dynamics.rep_eng import (
    DifferenceInMeanReader,
    LinearProbeReader,
    PCAReader,
    Reader,
)
from latent_dynamics.utils import is_activation_leaf

ArrayLike = np.ndarray | Sequence[np.ndarray]


def _maybe_write_image(fig: go.Figure, path: Path, dpi: int = 300) -> None:
    """Best-effort static image export (requires kaleido); always write HTML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    html_path = path.with_suffix(".html")
    fig.write_html(str(html_path))
    try:
        # Approximate DPI by scaling width/height.
        fig.write_image(str(path.with_suffix(".png")), scale=dpi / 96.0)
    except Exception:
        # Static export is optional; HTML always available.
        return


def plot_concept_direction_over_time(
    scans: list[np.ndarray],
    labels: np.ndarray | None = None,
    max_traces: int = 16,
    title: str = "LAT scan (projection over token position)",
    save_path: Path | None = None,
) -> go.Figure:
    fig = go.Figure()
    n = min(max_traces, len(scans))

    for i in range(n):
        color = "crimson" if labels is not None and labels[i] == 1 else "seagreen"
        name = f"ex_{i}" if labels is None else f"ex_{i}_y{labels[i]}"
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scans[i]))),
                y=scans[i].tolist(),
                mode="lines",
                line={"width": 1.7, "color": color},
                name=name,
                opacity=0.7,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Token position",
        yaxis_title="Projection on concept direction",
        template="plotly_white",
    )

    if save_path is not None:
        _maybe_write_image(fig, save_path)

    return fig


def _pool_trajectory(traj: np.ndarray, pooling: str) -> np.ndarray:
    if traj.ndim != 2:
        raise ValueError(
            f"Expected each activation trajectory to have shape (T, D), got {traj.shape}."
        )

    if pooling == "last":
        return traj[-1]
    if pooling == "mean":
        return traj.mean(axis=0)
    if pooling == "max_norm":
        norms = np.linalg.norm(traj, axis=1)
        return traj[int(np.argmax(norms))]
    raise ValueError(f"Unknown pooling mode: {pooling}")


def _to_feature_matrix(activations: ArrayLike, pooling: str) -> np.ndarray:
    if isinstance(activations, np.ndarray):
        if activations.ndim != 2:
            raise ValueError(
                f"Expected matrix of shape (N, D) when given ndarray, got {activations.shape}."
            )
        return activations.astype(np.float32)

    acts = list(activations)
    if not acts:
        raise ValueError("Received empty activations.")

    first = acts[0]
    if first.ndim == 1:
        return np.stack([a.astype(np.float32) for a in acts], axis=0)

    if first.ndim == 2:
        pooled = [_pool_trajectory(a.astype(np.float32), pooling) for a in acts]
        return np.stack(pooled, axis=0)

    raise ValueError(
        f"Unsupported activation rank {first.ndim}; expected 1D or 2D arrays."
    )


def _as_binary_labels(labels: np.ndarray | Sequence[int]) -> np.ndarray:
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D. Got shape {y.shape}.")

    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(
            f"This visualization flow expects binary labels. Found classes: {classes}."
        )
    if np.array_equal(classes, np.array([0, 1])):
        return y.astype(np.int64)

    return (y == classes[-1]).astype(np.int64)


def _pad_ragged(curves: list[np.ndarray], fill_value: float = np.nan) -> np.ndarray:
    t_max = max(len(c) for c in curves)
    out = np.full((len(curves), t_max), fill_value, dtype=np.float32)
    for i, curve in enumerate(curves):
        out[i, : len(curve)] = curve.astype(np.float32)
    return out


def plot_pca_subspace(
    activations: ArrayLike,
    labels: np.ndarray | Sequence[int],
    pooling: str = "last",
    title: str = "PCA subspace",
    save_path: Path | None = None,
) -> go.Figure:
    X = _to_feature_matrix(activations, pooling=pooling)
    y = _as_binary_labels(labels)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched rows: X={X.shape[0]} vs y={y.shape[0]}.")

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    fig = go.Figure()
    for cls, color, name in ((0, "seagreen", "label_0"), (1, "crimson", "label_1")):
        mask = y == cls
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=Z[mask, 0],
                y=Z[mask, 1],
                mode="markers",
                name=name,
                marker={"color": color, "size": 7, "opacity": 0.8},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"PC1 ({100.0 * evr[0]:.1f}% var)",
        yaxis_title=f"PC2 ({100.0 * evr[1]:.1f}% var)",
        template="plotly_white",
    )

    if save_path is not None:
        _maybe_write_image(fig, save_path)

    return fig


def _find_activation_leaves(root_or_leaf: Path) -> list[Path]:
    root = root_or_leaf.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Activation path does not exist: {root}")
    if is_activation_leaf(root):
        return [root]

    leaves: list[Path] = []
    for metadata in sorted(root.rglob("metadata.json")):
        leaf = metadata.parent
        if is_activation_leaf(leaf):
            leaves.append(leaf)

    if not leaves:
        raise FileNotFoundError(
            f"No activation leaves found under {root} "
            "(expected metadata.json + trajectories.safetensors or shard files)."
        )
    return leaves


def _sorted_layer_items(
    trajectories_by_layer: dict[int, list[np.ndarray]],
) -> list[tuple[int, list[np.ndarray]]]:
    return sorted(trajectories_by_layer.items(), key=lambda x: x[0])


def _parse_layer_from_name(path: Path) -> int | None:
    match = re.fullmatch(r"layer_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_root_from_hub(
    hf_repo_id: str,
    dataset_key: str,
    model_key: str,
    hub_layer: int | None,
    cache_dir: Path,
) -> Path:
    if hub_layer is not None:
        subpath = activation_subpath(
            dataset_key=dataset_key,
            model_key=model_key,
            layer_idx=hub_layer,
        )
    else:
        subpath = Path(dataset_key) / model_key

    root = (cache_dir / hf_repo_id.replace("/", "__") / subpath).resolve()
    if not root.exists() or not _find_activation_leaves_safe(root):
        pull_from_hub(
            repo_id=hf_repo_id,
            local_dir=root,
            path_in_repo=str(subpath),
        )
    return root


def _find_activation_leaves_safe(root: Path) -> list[Path]:
    try:
        return _find_activation_leaves(root)
    except FileNotFoundError:
        return []


def _load_multi_layer_bundle(
    local_path: Path | str | None = None,
    hf_repo_id: str | None = None,
    dataset_key: str | None = None,
    model_key: str | None = None,
    hub_layer: int | None = None,
    cache_dir: Path = Path(".cache/hub"),
    layer_filter: set[int] | None = None,
) -> tuple[
    dict[int, list[np.ndarray]],
    np.ndarray,
    str,
    str,
    Path,
]:
    if hf_repo_id:
        for name, value in (("dataset_key", dataset_key), ("model_key", model_key)):
            if value is None:
                raise ValueError(f"{name} is required when loading from Hugging Face.")
        assert dataset_key is not None and model_key is not None
        root = _resolve_root_from_hub(
            hf_repo_id=hf_repo_id,
            dataset_key=dataset_key,
            model_key=model_key,
            hub_layer=hub_layer,
            cache_dir=cache_dir,
        )
    else:
        root = Path(local_path) if local_path is not None else Path("activations")

    leaves = _find_activation_leaves(root)
    trajectories_by_layer: dict[int, list[np.ndarray]] = {}
    labels_ref: np.ndarray | None = None
    dataset_ref: str | None = None
    model_ref: str | None = None

    for leaf in leaves:
        trajectories, _texts, labels, _token_texts, _generated, cfg = load_activations(
            leaf
        )
        if labels is None:
            raise ValueError(f"Labels missing in {leaf}; cannot run LAT scan.")
        y = _as_binary_labels(labels)

        layer_idx = int(cfg.layer_idx)
        parsed = _parse_layer_from_name(leaf)
        if parsed is not None and parsed != layer_idx:
            layer_idx = parsed
        if layer_filter is not None and layer_idx not in layer_filter:
            continue

        if labels_ref is None:
            labels_ref = y
            dataset_ref = cfg.dataset_key
            model_ref = cfg.model_key
        elif not np.array_equal(labels_ref, y):
            raise ValueError(
                f"Label mismatch across layers. First labels differ from layer {layer_idx} at {leaf}."
            )

        if layer_idx in trajectories_by_layer:
            raise ValueError(
                f"Found duplicate activations for layer {layer_idx}; "
                f"already loaded another leaf before {leaf}."
            )
        trajectories_by_layer[layer_idx] = trajectories

    if not trajectories_by_layer:
        raise ValueError("No layers available after applying filters.")
    if labels_ref is None or dataset_ref is None or model_ref is None:
        raise RuntimeError("Failed to load labels/model metadata from activations.")

    return trajectories_by_layer, labels_ref, dataset_ref, model_ref, root.resolve()


def _make_reader(method: str, pooling: str, seed: int) -> Reader:
    if method == "mean_diff":
        return DifferenceInMeanReader(pooling=pooling)
    if method == "pca":
        return PCAReader(pooling=pooling, random_state=seed)
    if method == "probe":
        return LinearProbeReader(pooling=pooling, random_state=seed)
    raise ValueError(f"Unknown method: {method}")


@dataclass(frozen=True)
class LayerScanResult:
    layer_idx: int
    scans: list[np.ndarray]
    accuracy: float
    auroc: float | None


def compute_layer_lat_scans(
    trajectories_by_layer: dict[int, list[np.ndarray]],
    labels: np.ndarray | Sequence[int],
    method: str = "probe",
    pooling: str = "last",
    test_size: float = 0.3,
    seed: int = 42,
) -> dict[int, LayerScanResult]:
    y = _as_binary_labels(labels)
    n = len(y)
    idx = np.arange(n)
    stratify = y if np.unique(y).size > 1 else None
    try:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    y_train = y[train_idx]
    y_test = y[test_idx]
    out: dict[int, LayerScanResult] = {}

    for layer_idx, trajectories in _sorted_layer_items(trajectories_by_layer):
        if len(trajectories) != n:
            raise ValueError(
                f"Layer {layer_idx} has {len(trajectories)} trajectories, expected {n}."
            )

        reader = _make_reader(method, pooling=pooling, seed=seed)
        train_acts = [trajectories[int(i)] for i in train_idx]
        test_acts = [trajectories[int(i)] for i in test_idx]
        reader.train(train_acts, y_train)

        scores = reader.predict(test_acts)
        preds = (scores >= 0).astype(np.int64)
        acc = float(accuracy_score(y_test, preds))
        if np.unique(y_test).size > 1:
            auc: float | None = float(roc_auc_score(y_test, scores))
        else:
            auc = None

        scans = [reader.predict(traj).astype(np.float32) for traj in trajectories]
        out[layer_idx] = LayerScanResult(
            layer_idx=layer_idx,
            scans=scans,
            accuracy=acc,
            auroc=auc,
        )

    return out


def plot_layer_lat_heatmap(
    scans_by_layer: dict[int, list[np.ndarray]],
    labels: np.ndarray | Sequence[int],
    title: str = "Layerwise LAT scan (class mean difference)",
    save_path: Path | None = None,
) -> go.Figure:
    y = _as_binary_labels(labels)
    layers = sorted(scans_by_layer.keys())
    rows: list[np.ndarray] = []
    t_max_global = 0

    for layer_idx in layers:
        padded = _pad_ragged(scans_by_layer[layer_idx])
        if padded.shape[0] != y.shape[0]:
            raise ValueError(
                f"Layer {layer_idx} has {padded.shape[0]} scans; expected {y.shape[0]}."
            )
        pos_mean = np.nanmean(padded[y == 1], axis=0)
        neg_mean = np.nanmean(padded[y == 0], axis=0)
        row = pos_mean - neg_mean
        rows.append(row.astype(np.float32))
        t_max_global = max(t_max_global, row.shape[0])

    z = np.full((len(rows), t_max_global), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        z[i, : row.shape[0]] = row

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=list(range(t_max_global)),
                y=[f"L{layer}" for layer in layers],
                colorscale="RdBu",
                zmid=0.0,
                colorbar={"title": "unsafe - safe"},
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Token position",
        yaxis_title="Layer",
        template="plotly_white",
    )

    if save_path is not None:
        _maybe_write_image(fig, save_path)

    return fig


def _select_best_layer(results: dict[int, LayerScanResult]) -> int:
    by_auc = [
        (layer_idx, result.auroc)
        for layer_idx, result in results.items()
        if result.auroc is not None
    ]
    if by_auc:
        return max(by_auc, key=lambda x: x[1])[0]
    return max(results.items(), key=lambda x: x[1].accuracy)[0]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layerwise LAT scan visualization using rep-eng concept vectors."
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("activations"),
        help="Local activation leaf or root directory (default: activations/).",
    )
    parser.add_argument(
        "--from-hub",
        type=str,
        default=None,
        help="Hugging Face dataset repo id (e.g. user/repo).",
    )
    parser.add_argument(
        "--hub-dataset",
        type=str,
        default=None,
        help="Dataset key for Hub path resolution.",
    )
    parser.add_argument(
        "--hub-model",
        type=str,
        default=None,
        help="Model key for Hub path resolution.",
    )
    parser.add_argument(
        "--hub-layer",
        type=int,
        default=None,
        help="Optional single layer index to load from Hub.",
    )
    parser.add_argument(
        "--layer",
        action="append",
        type=int,
        default=None,
        help="Optional local/hub layer filter (repeat for multiple layers).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["probe", "mean_diff", "pca"],
        default="probe",
        help="Concept-vector method from rep_eng readers.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["last", "mean", "max_norm"],
        default="last",
        help="Pooling mode for reader training and PCA subspace plot.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of samples reserved for per-layer evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and reader initialization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/lat_scan"),
        help="Directory for generated LAT/PCA plots.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/hub"),
        help="Cache directory used when pulling from Hugging Face Hub.",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=16,
        help="Maximum trajectories to draw in the single-layer LAT line plot.",
    )
    return parser


def _run_layerwise_lat_scan(args: argparse.Namespace) -> None:
    if args.from_hub:
        for name, value in (
            ("--hub-dataset", args.hub_dataset),
            ("--hub-model", args.hub_model),
        ):
            if value is None:
                raise ValueError(f"{name} is required with --from-hub.")

    layer_filter = set(args.layer) if args.layer else None
    local_path: Path | None = None if args.from_hub else args.activations
    trajectories_by_layer, labels, dataset_key, model_key, source_root = (
        _load_multi_layer_bundle(
            local_path=local_path,
            hf_repo_id=args.from_hub,
            dataset_key=args.hub_dataset,
            model_key=args.hub_model,
            hub_layer=args.hub_layer,
            cache_dir=args.cache_dir,
            layer_filter=layer_filter,
        )
    )

    results = compute_layer_lat_scans(
        trajectories_by_layer=trajectories_by_layer,
        labels=labels,
        method=args.method,
        pooling=args.pooling,
        test_size=args.test_size,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scans_by_layer = {layer: result.scans for layer, result in results.items()}
    heatmap_path = (
        args.output_dir / f"lat_heatmap_{model_key}_{dataset_key}_{args.method}"
    )
    plot_layer_lat_heatmap(
        scans_by_layer=scans_by_layer,
        labels=labels,
        title=f"Layerwise LAT scan — {model_key} / {dataset_key} / {args.method}",
        save_path=heatmap_path,
    )

    best_layer = _select_best_layer(results)
    best_scan_path = (
        args.output_dir
        / f"lat_scans_{model_key}_{dataset_key}_L{best_layer}_{args.method}"
    )
    plot_concept_direction_over_time(
        scans=results[best_layer].scans,
        labels=labels,
        max_traces=args.max_traces,
        title=f"LAT scan — {model_key} / {dataset_key} / L{best_layer} / {args.method}",
        save_path=best_scan_path,
    )

    pca_path = args.output_dir / f"pca_subspace_{model_key}_{dataset_key}_L{best_layer}"
    plot_pca_subspace(
        activations=trajectories_by_layer[best_layer],
        labels=labels,
        pooling=args.pooling,
        title=f"PCA subspace — {model_key} / {dataset_key} / L{best_layer}",
        save_path=pca_path,
    )

    print(
        f"Loaded {len(labels)} examples across {len(results)} layers from {source_root}\n"
        f"Saved:\n"
        f"- {heatmap_path}.html\n"
        f"- {best_scan_path}.html\n"
        f"- {pca_path}.html\n"
        "Per-layer metrics:"
    )
    for layer_idx in sorted(results.keys()):
        metric = results[layer_idx]
        auc_text = "n/a" if metric.auroc is None else f"{metric.auroc:.3f}"
        print(f"- L{layer_idx}: accuracy={metric.accuracy:.3f} auroc={auc_text}")


def plot_drift_curves(
    curves: list[np.ndarray],
    labels: np.ndarray,
    tau: float,
    max_traces: int = 16,
    title: str = "Trust-region drift over token position",
    save_path: Path | None = None,
) -> go.Figure:
    fig = go.Figure()

    for i in range(min(max_traces, len(curves))):
        color = "crimson" if labels[i] == 1 else "seagreen"
        fig.add_trace(
            go.Scatter(
                x=list(range(len(curves[i]))),
                y=curves[i].tolist(),
                mode="lines",
                line={"width": 1.7, "color": color},
                name=f"ex_{i}_y{labels[i]}",
                opacity=0.7,
            )
        )

    fig.add_hline(y=tau, line_dash="dash", line_color="black", annotation_text="tau")
    fig.update_layout(
        title=title,
        xaxis_title="Token position",
        yaxis_title="Distance from safe center",
        template="plotly_white",
    )

    if save_path is not None:
        _maybe_write_image(fig, save_path)

    return fig


if __name__ == "__main__":
    _run_layerwise_lat_scan(_build_arg_parser().parse_args())
