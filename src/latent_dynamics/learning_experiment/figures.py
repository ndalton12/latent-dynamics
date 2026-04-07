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
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
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
    return _stability_from_single_seed(
        payload=payload, method=method, metric_key=metric_key
    )


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
            raise ValueError(f"Missing aggregated layer metrics for method '{method}'.")
        metrics = aggregate[method]["test_auroc_by_layer"]
        layers = sorted(int(k) for k in metrics.keys())
        y = np.asarray(
            [float(metrics[str(li)]["mean"]) for li in layers], dtype=np.float64
        )
        e = np.asarray(
            [float(metrics[str(li)]["stderr"] or 0.0) for li in layers],
            dtype=np.float64,
        )
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
            raise ValueError(f"Missing aggregated layer metrics for method '{method}'.")
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


def _method_colors() -> dict[str, str]:
    return {
        "random": "#4C78A8",
        "static_disagreement": "#F58518",
        "dynamic_disagreement": "#54A24B",
        "dynamic_uncertainty": "#4C78A8",
        "dynamic_uncertainty_diversity": "#B279A2",
    }


def _select_heatmap_method(payload: dict[str, Any], methods: list[str]) -> str:
    acquisitions = payload.get("acquisitions", [])
    if "dynamic_disagreement" in acquisitions:
        return "dynamic_disagreement"
    if methods:
        return methods[-1]
    raise ValueError("No methods available for the layer-pair heatmap.")


def _curve_style(
    method_idx: int,
    method_count: int,
    n_points: int,
) -> dict[str, Any]:
    linestyles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    linestyle = linestyles[method_idx % len(linestyles)]
    marker = markers[method_idx % len(markers)]

    if n_points <= 0:
        marker_positions: list[int] | None = None
    else:
        step = max(1, int(method_count))
        start = method_idx % step
        marker_idx = np.arange(start, n_points, step, dtype=np.int64)
        if marker_idx.size == 0:
            marker_idx = np.asarray([min(method_idx, n_points - 1)], dtype=np.int64)
        marker_positions = marker_idx.tolist()

    return {
        "linestyle": linestyle,
        "marker": marker,
        "markevery": marker_positions,
    }


def _plot_metric_curve(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray | None,
    label: str,
    color: str | None,
    method_idx: int,
    method_count: int,
) -> None:
    style = _curve_style(
        method_idx=method_idx,
        method_count=method_count,
        n_points=int(x.size),
    )
    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markevery=style["markevery"],
        markerfacecolor="white",
        markeredgewidth=1.15,
    )
    if err is not None and err.size == y.size:
        ax.fill_between(x, y - err, y + err, alpha=0.2, color=color)


def _set_compact_y_limits(
    ax: Any,
    y_series: list[np.ndarray],
    err_series: list[np.ndarray | None],
    bounds: tuple[float, float] | None = None,
    pad_ratio: float = 0.08,
) -> None:
    lows: list[float] = []
    highs: list[float] = []
    for y, err in zip(y_series, err_series, strict=False):
        if y.size == 0:
            continue
        if err is not None and err.size == y.size:
            lo = y - err
            hi = y + err
        else:
            lo = y
            hi = y
        lo = lo[np.isfinite(lo)]
        hi = hi[np.isfinite(hi)]
        if lo.size == 0 or hi.size == 0:
            continue
        lows.append(float(np.min(lo)))
        highs.append(float(np.max(hi)))

    if not lows or not highs:
        return

    low = float(min(lows))
    high = float(max(highs))
    span = max(high - low, 1e-6)

    if bounds is not None:
        bounded_span = max(bounds[1] - bounds[0], 1e-6)
        min_span = 0.18 * bounded_span
    else:
        min_span = 0.12 * max(abs(low), abs(high), 1.0)

    if span < min_span:
        center = 0.5 * (low + high)
        half = 0.5 * min_span
        low = center - half
        high = center + half
        span = min_span

    pad = pad_ratio * span
    low -= pad
    high += pad

    if bounds is not None:
        low = max(bounds[0], low)
        high = min(bounds[1], high)
        if high - low < 1e-6:
            center = np.clip(0.5 * (low + high), bounds[0], bounds[1])
            half = min_span * 0.5
            low = max(bounds[0], center - half)
            high = min(bounds[1], center + half)

    ax.set_ylim(low, high)


def plot_figure1_al_curves(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    methods = _select_methods(payload)
    if not methods:
        raise ValueError("No acquisition methods available to plot Figure 1.")

    colors = _method_colors()

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    y_series: list[np.ndarray] = []
    err_series: list[np.ndarray | None] = []
    for method_idx, method in enumerate(methods):
        x, y, err = _extract_curve(payload=payload, method=method, metric="test_auroc")
        if x.size == 0:
            continue
        y_series.append(y)
        err_series.append(err)
        color = colors.get(method, None)
        _plot_metric_curve(
            ax=ax,
            x=x,
            y=y,
            err=err,
            label=method.replace("_", " "),
            color=color,
            method_idx=method_idx,
            method_count=len(methods),
        )

    ax.set_xlabel("Queried Labels")
    ax.set_ylabel("Test AUROC")
    _set_compact_y_limits(
        ax=ax,
        y_series=y_series,
        err_series=err_series,
        bounds=(0.0, 1.0),
    )
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    return _save_figure(fig, output_dir / "figure1_al_curves")


def plot_figure2_ranking_stability(
    payload: dict[str, Any], output_dir: Path
) -> dict[str, str]:
    dynamic_methods = [
        m for m in payload.get("acquisitions", []) if m.startswith("dynamic_")
    ]
    if not dynamic_methods:
        raise ValueError("No dynamic methods found for Figure 2.")

    metric_specs = [
        ("top_k_overlap_at_k", "Top-k Overlap@k", (0.0, 1.0)),
        ("spearman_rho_shared_pool", "Spearman Rho (Shared Pool)", (-1.0, 1.0)),
        ("score_drift_mae_shared_pool", "Score Drift MAE (Shared Pool)", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharex=True)
    axes_flat = np.asarray(axes).flatten()

    colors = _method_colors()

    for ax, (metric_key, _title, y_lim) in zip(axes_flat, metric_specs, strict=False):
        plotted_any = False
        y_series: list[np.ndarray] = []
        err_series: list[np.ndarray | None] = []
        for method_idx, method in enumerate(dynamic_methods):
            x, y, err = _extract_stability_curve(
                payload=payload,
                method=method,
                metric_key=metric_key,
            )
            if x.size == 0:
                continue
            plotted_any = True
            y_series.append(y)
            err_series.append(err)
            color = colors.get(method, None)
            _plot_metric_curve(
                ax=ax,
                x=x,
                y=y,
                err=err,
                label=method.replace("_", " "),
                color=color,
                method_idx=method_idx,
                method_count=len(dynamic_methods),
            )

        ax.set_xlabel("Queried Labels")
        _set_compact_y_limits(
            ax=ax,
            y_series=y_series,
            err_series=err_series,
            bounds=y_lim,
        )
        if plotted_any:
            ax.legend(frameon=True, loc="best")

    fig.tight_layout()
    return _save_figure(fig, output_dir / "figure2_ranking_stability")


def plot_figure3_layerwise(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    methods = _select_methods(payload)
    if not methods:
        raise ValueError("No methods available for Figure 3.")

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    colors = _method_colors()

    layer_metrics_error: str | None = None
    try:
        layerwise_y_series: list[np.ndarray] = []
        layerwise_err_series: list[np.ndarray | None] = []
        for method_idx, method in enumerate(methods):
            layers, y, err = _get_layer_auroc_series(payload=payload, method=method)
            if layers.size == 0:
                continue
            layerwise_y_series.append(y)
            layerwise_err_series.append(err)
            color = colors.get(method, None)
            _plot_metric_curve(
                ax=ax,
                x=layers,
                y=y,
                err=err,
                label=method.replace("_", " "),
                color=color,
                method_idx=method_idx,
                method_count=len(methods),
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Test AUROC")
        _set_compact_y_limits(
            ax=ax,
            y_series=layerwise_y_series,
            err_series=layerwise_err_series,
            bounds=(0.0, 1.0),
        )
        ax.legend(frameon=True, loc="best")
    except ValueError as exc:
        layer_metrics_error = str(exc)
        ax.set_axis_off()
        fig.text(
            0.5,
            0.52,
            "Layerwise metrics unavailable in this result file.",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.45,
            layer_metrics_error,
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.tight_layout()
    return _save_figure(fig, output_dir / "figure3_layerwise")


def plot_figure4_layerwise_heatmap(
    payload: dict[str, Any], output_dir: Path
) -> dict[str, str]:
    methods = _select_methods(payload)
    if not methods:
        raise ValueError("No methods available for Figure 4.")

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    layer_metrics_error: str | None = None
    try:
        heatmap_method = _select_heatmap_method(payload=payload, methods=methods)
        mat, layers = _get_pairwise_disagreement_matrix(
            payload=payload,
            method=heatmap_method,
        )
        if mat.size == 0:
            ax.text(
                0.5,
                0.5,
                "No layer-pair metrics available",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        else:
            im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=1.0)
            tick_positions = np.arange(len(layers))
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(layers, rotation=45, ha="right")
            ax.set_yticklabels(layers)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Layer")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Prediction Disagreement Rate")
    except ValueError as exc:
        layer_metrics_error = str(exc)
        ax.set_axis_off()
        fig.text(
            0.5,
            0.54,
            "Layer-pair metrics unavailable in this result file.",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.46,
            layer_metrics_error,
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.tight_layout()
    return _save_figure(fig, output_dir / "figure4_layerwise_heatmap")


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
    out4 = plot_figure4_layerwise_heatmap(payload=payload, output_dir=output_dir)
    plt.close("all")

    return {
        "figure1_al_curves": out1,
        "figure2_ranking_stability": out2,
        "figure3_layerwise": out3,
        "figure4_layerwise_heatmap": out4,
    }
