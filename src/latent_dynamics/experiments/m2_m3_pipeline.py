from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


@dataclass
class PipelineConfig:
    activations: Path
    shifted_activations: Path | None = None
    output_dir: Path = Path("experiments/outputs")
    model_output: Path | None = None
    seed: int = 42


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    header = "| method | auroc | fpr@95 | tpr@95 |\n|---|---:|---:|---:|"
    lines = [header]
    for r in rows:
        lines.append(
            "| {m} | {a} | {f} | {t} |".format(
                m=r["method"],
                a=f"{r['auroc']:.4f}" if r["auroc"] is not None else "NA",
                f=f"{r['fpr_at_95_safe_coverage']:.4f}" if r["fpr_at_95_safe_coverage"] is not None else "NA",
                t=f"{r['tpr_at_95_safe_coverage']:.4f}" if r["tpr_at_95_safe_coverage"] is not None else "NA",
            )
        )
    return "\n".join(lines) + "\n"


def _to_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Method & AUROC & FPR@95 & TPR@95 \\\\",
        "\\hline",
    ]
    for r in rows:
        a = f"{r['auroc']:.4f}" if r["auroc"] is not None else "NA"
        fpr = f"{r['fpr_at_95_safe_coverage']:.4f}" if r["fpr_at_95_safe_coverage"] is not None else "NA"
        tpr = f"{r['tpr_at_95_safe_coverage']:.4f}" if r["tpr_at_95_safe_coverage"] is not None else "NA"
        lines.append(f"{r['method']} & {a} & {fpr} & {tpr} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    trajectories, _texts, labels, _tokens, run_cfg = load_activations(cfg.activations)
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
    for idx, y in zip(test_idx, test_labels):
        if int(y) != 1:
            continue
        curve = per_token_mahalanobis_curve(trajectories[int(idx)], model)
        unsafe_metrics.append(
            compute_drift_metrics(
                score_curve=curve,
                tau=token_tau,
                unsafe_output_token_position=(len(curve) - 1 if len(curve) else None),
                sparse_supports=None,
            )
        )
    drift_summary = summarize_unsafe_drift(unsafe_metrics)

    shift_eval = None
    if cfg.shifted_activations is not None:
        s_traj, _stext, s_labels, _stok, _scfg = load_activations(cfg.shifted_activations)
        if s_labels is not None:
            s_scores = trust_region_scores(s_traj, model)
            in_scores = trust_region_scores([trajectories[int(i)] for i in test_idx], model)
            shift_eval = evaluate_generator_shift(
                in_domain_labels=test_labels,
                in_domain_scores=in_scores,
                shifted_labels=s_labels,
                shifted_scores=s_scores,
            )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    md_table = _to_markdown(rows)
    tex_table = _to_latex(rows)
    (cfg.output_dir / "milestone2_comparison.md").write_text(md_table)
    (cfg.output_dir / "milestone2_comparison.tex").write_text(tex_table)

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
        "milestone3_generator_shift": shift_eval,
        "model_path": str(model_out),
        "table_markdown_path": str(cfg.output_dir / "milestone2_comparison.md"),
        "table_latex_path": str(cfg.output_dir / "milestone2_comparison.tex"),
    }
    (cfg.output_dir / "milestone23_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
