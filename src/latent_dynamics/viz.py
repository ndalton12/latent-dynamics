from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def plot_lat_scans(
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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path.with_suffix(".html")))

    return fig


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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path.with_suffix(".html")))

    return fig
