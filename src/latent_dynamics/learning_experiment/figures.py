from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.1,
            "lines.linewidth": 2.3,
            "lines.markersize": 5.5,
        }
    )


def _is_multi_seed(payload: dict[str, Any]) -> bool:
    return "per_seed" in payload and "aggregate_by_queried_labels" in payload


def _save_figure(fig: plt.Figure, out_base: Path) -> dict[str, str]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _curve_from_single_seed(
    payload: dict[str, Any],
    method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rounds = payload["runs"][method]["rounds"]
    x: list[float] = []
    y: list[float] = []
    for row in rounds:
        val = row.get(metric)
        if val is None:
            continue
        x.append(float(row["queried_labels"]))
        y.append(float(val))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), None


def _curve_from_multi_seed(
    payload: dict[str, Any],
    method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows = payload["aggregate_by_queried_labels"][method]
    x: list[float] = []
    y: list[float] = []
    e: list[float] = []
    for row in rows:
        stat = row.get(metric)
        if not isinstance(stat, dict):
            continue
        mean = stat.get("mean")
        if mean is None:
            continue
        x.append(float(row["queried_labels"]))
        y.append(float(mean))
        e.append(float(stat.get("stderr") or 0.0))
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(e, dtype=np.float64),
    )


def _extract_curve(
    payload: dict[str, Any],
    method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if _is_multi_seed(payload):
        return _curve_from_multi_seed(payload=payload, method=method, metric=metric)
    return _curve_from_single_seed(payload=payload, method=method, metric=metric)


def _compute_mean_stderr(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    return mean, float(std / np.sqrt(arr.size))


def _aggregate_stability_from_multiseed(
    payload: dict[str, Any],
    method: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    by_q: dict[int, list[float]] = {}
    for seed_payload in payload["per_seed"].values():
        rounds = seed_payload["runs"][method]["rounds"]
        for row in rounds:
            rs = row.get("ranking_stability")
            if not isinstance(rs, dict):
                continue
            val = rs.get(metric_key)
            if val is None:
                continue
            q = int(row["queried_labels"])
            by_q.setdefault(q, []).append(float(val))

    xs = sorted(by_q)
    ys: list[float] = []
    es: list[float] = []
    for q in xs:
        mean, stderr = _compute_mean_stderr(by_q[q])
        if mean is None:
            continue
        ys.append(mean)
        es.append(0.0 if stderr is None else stderr)
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(es, dtype=np.float64),
    )


def _stability_from_single_seed(
    payload: dict[str, Any],
    method: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rounds = payload["runs"][method]["rounds"]
    x: list[float] = []
    y: list[float] = []
    for row in rounds:
        rs = row.get("ranking_stability")
        if not isinstance(rs, dict):
            continue
        val = rs.get(metric_key)
        if val is None:
            continue
        x.append(float(row["queried_labels"]))
        y.append(float(val))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), None


def _extract_stability_curve(
    payload: dict[str, Any],
    method: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if _is_multi_seed(payload):
        return _aggregate_stability_from_multiseed(
            payload=payload,
            method=method,
            metric_key=metric_key,
        )
    return _stability_from_single_seed(payload=payload, method=method, metric_key=metric_key)


def _get_layer_auroc_series(
    payload: dict[str, Any],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if _is_multi_seed(payload):
        layer_metrics_root = payload.get("layer_metrics")
        if not isinstance(layer_metrics_root, dict):
            raise ValueError(
                "Missing 'layer_metrics' in multi-seed payload. "
                "Regenerate results with current comparison code to include layer metrics."
            )
        aggregate = layer_metrics_root.get("aggregate", {})
        if not isinstance(aggregate, dict) or method not in aggregate:
            raise ValueError(
                f"Missing aggregated layer metrics for method '{method}'."
            )
        metrics = aggregate[method]["test_auroc_by_layer"]
        layers = sorted(int(k) for k in metrics.keys())
        y = np.asarray([float(metrics[str(li)]["mean"]) for li in layers], dtype=np.float64)
        e = np.asarray([float(metrics[str(li)]["stderr"] or 0.0) for li in layers], dtype=np.float64)
        return np.asarray(layers, dtype=np.float64), y, e

    layer_metrics_root = payload.get("layer_metrics")
    if isinstance(layer_metrics_root, dict) and method in layer_metrics_root:
        metrics = layer_metrics_root[method]["test_auroc_by_layer"]
    else:
        run = payload.get("runs", {}).get(method, {})
        run_layer_metrics = run.get("layer_metrics")
        if not isinstance(run_layer_metrics, dict):
            raise ValueError(
                f"Missing layer metrics for method '{method}' in single-seed payload."
            )
        metrics = run_layer_metrics["test_auroc_by_layer"]

    layers = sorted(int(k) for k in metrics.keys())
    y = np.asarray([float(metrics[str(li)]) for li in layers], dtype=np.float64)
    return np.asarray(layers, dtype=np.float64), y, None


def _get_pairwise_disagreement_matrix(
    payload: dict[str, Any],
    method: str,
) -> tuple[np.ndarray, list[int]]:
    if _is_multi_seed(payload):
        layer_metrics_root = payload.get("layer_metrics")
        if not isinstance(layer_metrics_root, dict):
            raise ValueError(
                "Missing 'layer_metrics' in multi-seed payload. "
                "Regenerate results with current comparison code to include layer metrics."
            )
        aggregate = layer_metrics_root.get("aggregate", {})
        if not isinstance(aggregate, dict) or method not in aggregate:
            raise ValueError(
                f"Missing aggregated layer metrics for method '{method}'."
            )
        pair_rows = aggregate[method]["disagreement_by_layer_pair"]
    else:
        layer_metrics_root = payload.get("layer_metrics")
        if isinstance(layer_metrics_root, dict) and method in layer_metrics_root:
            pair_rows = layer_metrics_root[method]["disagreement_by_layer_pair"]
        else:
            run = payload.get("runs", {}).get(method, {})
            run_layer_metrics = run.get("layer_metrics")
            if not isinstance(run_layer_metrics, dict):
                raise ValueError(
                    f"Missing layer metrics for method '{method}' in single-seed payload."
                )
            pair_rows = run_layer_metrics["disagreement_by_layer_pair"]

    layer_set: set[int] = set()
    for row in pair_rows:
        layer_set.add(int(row["layer_a"]))
        layer_set.add(int(row["layer_b"]))
    layers = sorted(layer_set)
    if not layers:
        return np.zeros((0, 0), dtype=np.float64), []

    idx = {layer: i for i, layer in enumerate(layers)}
    mat = np.full((len(layers), len(layers)), np.nan, dtype=np.float64)
    np.fill_diagonal(mat, 0.0)

    for row in pair_rows:
        la = int(row["layer_a"])
        lb = int(row["layer_b"])
        if _is_multi_seed(payload):
            val = row["prediction_disagreement_rate"]["mean"]
        else:
            val = row["prediction_disagreement_rate"]
        if val is None:
            continue
        i = idx[la]
        j = idx[lb]
        mat[i, j] = float(val)
        mat[j, i] = float(val)

    return mat, layers


def _select_methods(payload: dict[str, Any]) -> list[str]:
    preferred = ["random", "static_disagreement", "dynamic_disagreement"]
    available = payload.get("acquisitions", [])
    selected = [m for m in preferred if m in available]
    if selected:
        return selected
    return available[:3]


def plot_figure1_al_curves(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    methods = _select_methods(payload)
    if not methods:
        raise ValueError("No acquisition methods available to plot Figure 1.")

    colors = {
        "random": "#4C78A8",
        "static_disagreement": "#F58518",
        "dynamic_disagreement": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for method in methods:
        x, y, err = _extract_curve(payload=payload, method=method, metric="test_auroc")
        if x.size == 0:
            continue
        color = colors.get(method, None)
        ax.plot(x, y, marker="o", label=method.replace("_", " "), color=color)
        if err is not None and err.size == y.size:
            ax.fill_between(x, y - err, y + err, alpha=0.22, color=color)

    ax.set_xlabel("Queried Labels")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Figure 1: Active Learning Curves")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    return _save_figure(fig, output_dir / "figure1_al_curves")


def plot_figure2_ranking_stability(
    payload: dict[str, Any], output_dir: Path
) -> dict[str, str]:
    dynamic_methods = [m for m in payload.get("acquisitions", []) if m.startswith("dynamic_")]
    if not dynamic_methods:
        raise ValueError("No dynamic methods found for Figure 2.")

    metric_specs = [
        ("top_k_overlap_at_k", "Top-k Overlap@k", (0.0, 1.0)),
        ("spearman_rho_shared_pool", "Spearman Rho (Shared Pool)", (-1.0, 1.0)),
        ("score_drift_mae_shared_pool", "Score Drift MAE (Shared Pool)", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharex=True)
    axes_flat = np.asarray(axes).flatten()

    colors = {
        "dynamic_uncertainty": "#4C78A8",
        "dynamic_disagreement": "#54A24B",
        "dynamic_uncertainty_diversity": "#B279A2",
    }

    for ax, (metric_key, title, y_lim) in zip(axes_flat, metric_specs, strict=False):
        plotted_any = False
        for method in dynamic_methods:
            x, y, err = _extract_stability_curve(
                payload=payload,
                method=method,
                metric_key=metric_key,
            )
            if x.size == 0:
                continue
            plotted_any = True
            color = colors.get(method, None)
            ax.plot(x, y, marker="o", label=method.replace("_", " "), color=color)
            if err is not None and err.size == y.size:
                ax.fill_between(x, y - err, y + err, alpha=0.22, color=color)

        ax.set_title(title)
        ax.set_xlabel("Queried Labels")
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if plotted_any:
            ax.legend(frameon=True, loc="best")

    fig.suptitle("Figure 2: Ranking Stability Across Rounds", y=1.01, fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "figure2_ranking_stability")


def plot_figure3_layerwise(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    methods = _select_methods(payload)
    if not methods:
        raise ValueError("No methods available for Figure 3.")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    colors = {
        "random": "#4C78A8",
        "static_disagreement": "#F58518",
        "dynamic_disagreement": "#54A24B",
    }

    layer_metrics_error: str | None = None
    try:
        for method in methods:
            layers, y, err = _get_layer_auroc_series(payload=payload, method=method)
            if layers.size == 0:
                continue
            color = colors.get(method, None)
            ax1.plot(layers, y, marker="o", label=method.replace("_", " "), color=color)
            if err is not None and err.size == y.size:
                ax1.fill_between(layers, y - err, y + err, alpha=0.22, color=color)

        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Test AUROC")
        ax1.set_title("AUROC by Layer")
        ax1.set_ylim(0.0, 1.0)
        ax1.legend(frameon=True, loc="best")

        heatmap_method = (
            "dynamic_disagreement"
            if "dynamic_disagreement" in payload.get("acquisitions", [])
            else methods[-1]
        )
        mat, layers = _get_pairwise_disagreement_matrix(payload=payload, method=heatmap_method)
        if mat.size == 0:
            ax2.text(0.5, 0.5, "No layer-pair metrics available", ha="center", va="center")
            ax2.set_axis_off()
        else:
            im = ax2.imshow(mat, cmap="magma", vmin=0.0, vmax=1.0)
            ax2.set_title(f"Disagreement by Layer Pair ({heatmap_method.replace('_', ' ')})")
            tick_positions = np.arange(len(layers))
            ax2.set_xticks(tick_positions)
            ax2.set_yticks(tick_positions)
            ax2.set_xticklabels(layers, rotation=45, ha="right")
            ax2.set_yticklabels(layers)
            ax2.set_xlabel("Layer")
            ax2.set_ylabel("Layer")
            cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label("Prediction Disagreement Rate")
    except ValueError as exc:
        layer_metrics_error = str(exc)
        ax1.set_axis_off()
        ax2.set_axis_off()
        fig.text(
            0.5,
            0.52,
            "Layerwise metrics unavailable in this result file.",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.45,
            layer_metrics_error,
            ha="center",
            va="center",
            fontsize=10,
        )

    fig.suptitle("Figure 3: Layerwise Diagnostics", y=1.02, fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "figure3_layerwise")


def generate_comparison_figures(
    comparison_json: Path,
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    _configure_style()
    payload = json.loads(comparison_json.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)

    out1 = plot_figure1_al_curves(payload=payload, output_dir=output_dir)
    out2 = plot_figure2_ranking_stability(payload=payload, output_dir=output_dir)
    out3 = plot_figure3_layerwise(payload=payload, output_dir=output_dir)
    plt.close("all")

    return {
        "figure1_al_curves": out1,
        "figure2_ranking_stability": out2,
        "figure3_layerwise": out3,
    }
