from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from latent_dynamics.drift import (
    compute_drift_metrics,
    evaluate_generator_shift,
    summarize_unsafe_drift,
)
from latent_dynamics.dynamics import (
    baseline_brt_align_scores,
    baseline_nglare_scores,
    baseline_sap_proxy_scores,
    fit_trust_region_model,
    per_token_mahalanobis_curve,
    save_trust_region_model,
    split_indices_70_15_15,
    to_table_rows,
    trust_region_scores,
)
from latent_dynamics.hub import load_activations
from latent_dynamics.viz import _maybe_write_image


@dataclass
class PipelineConfig:
    activations: Path
    shifted_activations: Path | None = None
    output_dir: Path = Path("experiments/outputs")
    model_output: Path | None = None
    seed: int = 42
    sap_repo_path: Path | None = None
    real_sap: bool = False
    plot_drift: bool = False
    sap_script_relative_path: str = "src/safety_polytope/polytope/run_beaver_pipeline.py"


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    header = "| method | auroc | fpr@95 | tpr@95 | prefix_to_failure_lead_time_mean |\n|---|---:|---:|---:|---:|"
    lines = [header]
    for r in rows:
        lines.append(
            "| {m} | {a} | {f} | {t} | {l} |".format(
                m=r["method"],
                a=f"{r['auroc']:.4f}" if r["auroc"] is not None else "NA",
                f=f"{r['fpr_at_95_safe_coverage']:.4f}" if r["fpr_at_95_safe_coverage"] is not None else "NA",
                t=f"{r['tpr_at_95_safe_coverage']:.4f}" if r["tpr_at_95_safe_coverage"] is not None else "NA",
                l=f"{r['prefix_to_failure_lead_time_mean']:.4f}" if r.get("prefix_to_failure_lead_time_mean") is not None else "NA",
            )
        )
    return "\n".join(lines) + "\n"


def _to_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Method & AUROC & FPR@95 & TPR@95 & LeadTime \\\\",
        "\\hline",
    ]
    for r in rows:
        a = f"{r['auroc']:.4f}" if r["auroc"] is not None else "NA"
        fpr = f"{r['fpr_at_95_safe_coverage']:.4f}" if r["fpr_at_95_safe_coverage"] is not None else "NA"
        tpr = f"{r['tpr_at_95_safe_coverage']:.4f}" if r["tpr_at_95_safe_coverage"] is not None else "NA"
        lead = f"{r['prefix_to_failure_lead_time_mean']:.4f}" if r.get("prefix_to_failure_lead_time_mean") is not None else "NA"
        lines.append(f"{r['method']} & {a} & {fpr} & {tpr} & {lead} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _to_drift_markdown(drift_summary: dict[str, Any]) -> str:
    header = "| metric | value |\n|---|---:|"
    keys = [
        "n_unsafe",
        "exit_time_median",
        "exit_time_mean",
        "prefix_to_failure_lead_time_median",
        "prefix_to_failure_lead_time_mean",
        "boundary_crossings_mean",
        "smoothness_total_variation_mean",
        "sparse_support_churn_mean",
    ]
    lines = [header]
    for k in keys:
        v = drift_summary.get(k)
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def _to_drift_latex(drift_summary: dict[str, Any]) -> str:
    keys = [
        "n_unsafe",
        "exit_time_median",
        "exit_time_mean",
        "prefix_to_failure_lead_time_median",
        "prefix_to_failure_lead_time_mean",
        "boundary_crossings_mean",
        "smoothness_total_variation_mean",
        "sparse_support_churn_mean",
    ]
    lines = [
        "\\begin{tabular}{lc}",
        "\\hline",
        "Metric & Value \\\\",
        "\\hline",
    ]
    for k in keys:
        v = drift_summary.get(k)
        if isinstance(v, float):
            lines.append(f"{k} & {v:.4f} \\\\")
        else:
            lines.append(f"{k} & {v} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _maybe_run_real_sap(cfg: PipelineConfig) -> dict[str, Any] | None:
    """Optional real SaP integration (external repo invocation)."""
    if not cfg.real_sap:
        return None
    repo_path = cfg.sap_repo_path or Path(".cache/baselines/SafetyPolytope")
    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = [
            "git",
            "clone",
            "https://github.com/lasgroup/SafetyPolytope.git",
            str(repo_path),
        ]
        clone_proc = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if clone_proc.returncode != 0:
            return {
                "status": "clone_failed",
                "repo_path": str(repo_path),
                "stdout_tail": clone_proc.stdout[-4000:],
                "stderr_tail": clone_proc.stderr[-4000:],
                "command": clone_cmd,
            }
    script_path = repo_path / cfg.sap_script_relative_path
    if not script_path.exists():
        return {
            "status": "missing_script",
            "script_path": str(script_path),
        }
    # Minimal invocation scaffold. Users can extend args for their setup.
    cmd = ["python", str(script_path), "--reduced_data"]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "status": "ok" if result.returncode == 0 else "failed",
            "returncode": int(result.returncode),
            "script_path": str(script_path),
            "repo_path": str(repo_path),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "command": cmd,
        }
    except Exception as exc:
        return {
            "status": "error",
            "script_path": str(script_path),
            "repo_path": str(repo_path),
            "error": str(exc),
            "command": cmd,
        }


def _plot_exit_time_histogram(exit_times: list[int], out_path: Path) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=exit_times,
            nbinsx=max(10, min(50, len(exit_times))),
            marker={"color": "crimson"},
        )
    )
    fig.update_layout(
        title="Exit-time distribution (unsafe trajectories)",
        xaxis_title="exit_time token index",
        yaxis_title="Count",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def _plot_boundary_overlay(curve: np.ndarray, tau: float, out_path: Path) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(curve))),
            y=curve.tolist(),
            mode="lines",
            line={"color": "steelblue", "width": 2},
            name="score_curve",
        )
    )
    fig.add_hline(y=float(tau), line_dash="dash", line_color="black", annotation_text="tau")
    fig.update_layout(
        title="Example trajectory with trust-region boundary overlay",
        xaxis_title="Token position",
        yaxis_title="Per-token score",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def fit_trust_region_only(
    activations: Path,
    model_output: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    trajectories, _texts, labels, _tokens, _gen, run_cfg = load_activations(activations)
    if labels is None:
        raise ValueError("Activations must contain labels.")
    train_idx, calib_idx, test_idx = split_indices_70_15_15(labels, seed=seed)
    model = fit_trust_region_model(
        trajectories=trajectories,
        labels=labels,
        train_idx=train_idx,
        calib_idx=calib_idx,
        safe_coverage_target=0.95,
    )
    model_out = model_output or (Path("models") / f"trust_region_{run_cfg.model_key}.pkl")
    save_trust_region_model(model_out, model)
    return {
        "model_path": str(model_out),
        "split_sizes": {
            "train": int(len(train_idx)),
            "calib": int(len(calib_idx)),
            "test": int(len(test_idx)),
        },
        "threshold_tau": float(model.threshold_tau),
    }


def compute_drift_only(
    activations: Path,
    model_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    from latent_dynamics.dynamics import load_trust_region_model

    trajectories, _texts, labels, _tokens, _gen, _cfg = load_activations(activations)
    if labels is None:
        raise ValueError("Activations must contain labels.")
    _train_idx, calib_idx, test_idx = split_indices_70_15_15(labels, seed=seed)
    model = load_trust_region_model(model_path)
    calib_safe_idx = [int(i) for i in calib_idx if labels[int(i)] == 0]
    calib_safe_trajs = [trajectories[i] for i in calib_safe_idx]
    token_tau = float(
        np.quantile(
            np.concatenate([per_token_mahalanobis_curve(t, model) for t in calib_safe_trajs]),
            0.95,
        )
    )
    unsafe_metrics = []
    unsafe_records = []
    for idx in test_idx:
        if int(labels[int(idx)]) != 1:
            continue
        curve = per_token_mahalanobis_curve(trajectories[int(idx)], model)
        metric = compute_drift_metrics(
            score_curve=curve,
            tau=token_tau,
            unsafe_output_token_position=(len(curve) - 1 if len(curve) else None),
            sparse_supports=None,
        )
        unsafe_metrics.append(metric)
        unsafe_records.append({
            "idx": int(idx),
            "exit_time": metric.exit_time,
            "prefix_to_failure_lead_time": metric.prefix_to_failure_lead_time,
            "boundary_crossings": metric.boundary_crossings,
            "smoothness_total_variation": metric.smoothness_total_variation,
        })
    return {
        "drift_summary": summarize_unsafe_drift(unsafe_metrics),
        "unsafe_records": unsafe_records,
    }


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    trajectories, _texts, labels, _tokens, _gen, run_cfg = load_activations(cfg.activations)
    if labels is None:
        raise ValueError("Activations must contain labels for Milestone 2/3 pipeline.")

    train_idx, calib_idx, test_idx = split_indices_70_15_15(labels, seed=cfg.seed)
    model = fit_trust_region_model(
        trajectories=trajectories,
        labels=labels,
        train_idx=train_idx,
        calib_idx=calib_idx,
        safe_coverage_target=0.95,
    )

    model_out = cfg.model_output or (Path("models") / f"trust_region_{run_cfg.model_key}.pkl")
    save_trust_region_model(model_out, model)

    test_trajs = [trajectories[int(i)] for i in test_idx]
    test_labels = labels[test_idx]
    calib_safe_idx = [int(i) for i in calib_idx if labels[int(i)] == 0]
    calib_safe_trajs = [trajectories[i] for i in calib_safe_idx]

    trust_test = trust_region_scores(test_trajs, model)
    trust_calib_safe = trust_region_scores(calib_safe_trajs, model)

    brt_test = baseline_brt_align_scores(test_trajs, model)
    brt_calib_safe = baseline_brt_align_scores(calib_safe_trajs, model)

    ng_test = baseline_nglare_scores(test_trajs)
    ng_calib_safe = baseline_nglare_scores(calib_safe_trajs)

    sap_test = baseline_sap_proxy_scores(test_trajs, model)
    sap_calib_safe = baseline_sap_proxy_scores(calib_safe_trajs, model)

    rows = to_table_rows(
        test_labels=test_labels,
        trust_scores=trust_test,
        brt_scores=brt_test,
        nglare_scores=ng_test,
        sap_scores=sap_test,
        calib_safe_scores={
            "trust_region": trust_calib_safe,
            "brt_align_simplified": brt_calib_safe,
            "nglare_simplified": ng_calib_safe,
            "sap_proxy": sap_calib_safe,
        },
    )

    # Milestone 3 drift summary on unsafe test trajectories.
    token_tau = float(np.quantile(np.concatenate([per_token_mahalanobis_curve(t, model) for t in calib_safe_trajs]), 0.95))
    unsafe_metrics = []
    unsafe_records: list[dict[str, Any]] = []
    for idx, y in zip(test_idx, test_labels):
        if int(y) != 1:
            continue
        curve = per_token_mahalanobis_curve(trajectories[int(idx)], model)
        metric = compute_drift_metrics(
            score_curve=curve,
            tau=token_tau,
            unsafe_output_token_position=(len(curve) - 1 if len(curve) else None),
            sparse_supports=None,
        )
        unsafe_metrics.append(metric)
        unsafe_records.append(
            {
                "idx": int(idx),
                "exit_time": metric.exit_time,
                "prefix_to_failure_lead_time": metric.prefix_to_failure_lead_time,
                "boundary_crossings": metric.boundary_crossings,
                "smoothness_total_variation": metric.smoothness_total_variation,
            }
        )
    drift_summary = summarize_unsafe_drift(unsafe_metrics)
    for row in rows:
        row["prefix_to_failure_lead_time_mean"] = (
            drift_summary.get("prefix_to_failure_lead_time_mean")
            if row["method"] == "trust_region"
            else None
        )

    shift_eval = None
    if cfg.shifted_activations is not None:
        s_traj, _stext, s_labels, _stok, _sgen, _scfg = load_activations(cfg.shifted_activations)
        if s_labels is not None:
            s_scores = trust_region_scores(s_traj, model)
            in_scores = trust_region_scores([trajectories[int(i)] for i in test_idx], model)
            shift_eval = evaluate_generator_shift(
                in_domain_labels=test_labels,
                in_domain_scores=in_scores,
                shifted_labels=s_labels,
                shifted_scores=s_scores,
            )

    sap_external = _maybe_run_real_sap(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = cfg.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    md_table = _to_markdown(rows)
    tex_table = _to_latex(rows)
    drift_md = _to_drift_markdown(drift_summary)
    drift_tex = _to_drift_latex(drift_summary)
    (cfg.output_dir / "milestone2_comparison.md").write_text(md_table)
    (cfg.output_dir / "milestone2_comparison.tex").write_text(tex_table)
    (cfg.output_dir / "milestone3_drift.md").write_text(drift_md)
    (cfg.output_dir / "milestone3_drift.tex").write_text(drift_tex)

    drift_plots = {}
    if cfg.plot_drift:
        exit_times = [int(v["exit_time"]) for v in unsafe_records if v["exit_time"] is not None]
        if exit_times:
            exit_hist_path = figures_dir / "milestone3_exit_time_histogram"
            _plot_exit_time_histogram(exit_times, exit_hist_path)
            drift_plots["exit_time_histogram"] = str(exit_hist_path.with_suffix(".png"))
        if unsafe_records:
            first_idx = int(unsafe_records[0]["idx"])
            curve = per_token_mahalanobis_curve(trajectories[first_idx], model)
            overlay_path = figures_dir / "milestone3_boundary_overlay_example"
            _plot_boundary_overlay(curve, token_tau, overlay_path)
            drift_plots["boundary_overlay_example"] = str(overlay_path.with_suffix(".png"))

    report = {
        "pipeline_config": asdict(cfg),
        "source_config": asdict(run_cfg),
        "split_sizes": {
            "train": int(len(train_idx)),
            "calib": int(len(calib_idx)),
            "test": int(len(test_idx)),
        },
        "milestone2_rows": rows,
        "milestone3_drift_summary": drift_summary,
        "milestone3_unsafe_records": unsafe_records,
        "milestone3_key_novelty_metrics": {
            "prefix_to_failure_lead_time_median": drift_summary.get("prefix_to_failure_lead_time_median"),
            "prefix_to_failure_lead_time_mean": drift_summary.get("prefix_to_failure_lead_time_mean"),
            "exit_time_median": drift_summary.get("exit_time_median"),
        },
        "milestone3_generator_shift": shift_eval,
        "sap_external_run": sap_external,
        "milestone3_plots": drift_plots if drift_plots else None,
        "model_path": str(model_out),
        "table_markdown_path": str(cfg.output_dir / "milestone2_comparison.md"),
        "table_latex_path": str(cfg.output_dir / "milestone2_comparison.tex"),
        "drift_markdown_path": str(cfg.output_dir / "milestone3_drift.md"),
        "drift_latex_path": str(cfg.output_dir / "milestone3_drift.tex"),
    }
    (cfg.output_dir / "milestone23_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
