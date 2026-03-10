from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class DriftMetrics:
    exit_time: int | None
    prefix_to_failure_lead_time: int | None
    boundary_crossings: int
    smoothness_total_variation: float
    sparse_support_churn: float | None


def first_exit_time(score_curve: np.ndarray, tau: float) -> int | None:
    idx = np.where(score_curve > tau)[0]
    return int(idx[0]) if idx.size else None


def count_boundary_crossings(score_curve: np.ndarray, tau: float) -> int:
    if score_curve.size < 2:
        return 0
    above = score_curve > tau
    return int(np.sum(above[1:] != above[:-1]))


def total_variation(score_curve: np.ndarray) -> float:
    if score_curve.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(score_curve))))


def support_set_churn(supports: list[np.ndarray]) -> float:
    if len(supports) < 2:
        return 0.0
    vals: list[float] = []
    for i in range(1, len(supports)):
        a = np.unique(supports[i - 1])
        b = np.unique(supports[i])
        inter = np.intersect1d(a, b).size
        union = np.union1d(a, b).size
        vals.append(0.0 if union == 0 else (1.0 - (inter / union)))
    return float(np.mean(vals)) if vals else 0.0


def compute_drift_metrics(
    score_curve: np.ndarray,
    tau: float,
    unsafe_output_token_position: int | None = None,
    sparse_supports: list[np.ndarray] | None = None,
) -> DriftMetrics:
    exit_t = first_exit_time(score_curve, tau)
    failure_pos = unsafe_output_token_position
    if failure_pos is None:
        failure_pos = int(len(score_curve) - 1) if len(score_curve) else None
    lead = None
    if exit_t is not None and failure_pos is not None:
        lead = int(failure_pos - exit_t)
    churn = None
    if sparse_supports is not None:
        churn = support_set_churn(sparse_supports)
    return DriftMetrics(
        exit_time=exit_t,
        prefix_to_failure_lead_time=lead,
        boundary_crossings=count_boundary_crossings(score_curve, tau),
        smoothness_total_variation=total_variation(score_curve),
        sparse_support_churn=churn,
    )


def summarize_unsafe_drift(metrics: list[DriftMetrics]) -> dict[str, Any]:
    exits = [m.exit_time for m in metrics if m.exit_time is not None]
    leads = [m.prefix_to_failure_lead_time for m in metrics if m.prefix_to_failure_lead_time is not None]
    crossings = [m.boundary_crossings for m in metrics]
    smooth = [m.smoothness_total_variation for m in metrics]
    churn = [m.sparse_support_churn for m in metrics if m.sparse_support_churn is not None]

    return {
        "n_unsafe": len(metrics),
        "exit_time_median": float(np.median(exits)) if exits else None,
        "exit_time_mean": float(np.mean(exits)) if exits else None,
        "prefix_to_failure_lead_time_median": float(np.median(leads)) if leads else None,
        "prefix_to_failure_lead_time_mean": float(np.mean(leads)) if leads else None,
        "boundary_crossings_mean": float(np.mean(crossings)) if crossings else 0.0,
        "smoothness_total_variation_mean": float(np.mean(smooth)) if smooth else 0.0,
        "sparse_support_churn_mean": float(np.mean(churn)) if churn else None,
    }


def evaluate_generator_shift(
    in_domain_labels: np.ndarray,
    in_domain_scores: np.ndarray,
    shifted_labels: np.ndarray,
    shifted_scores: np.ndarray,
) -> dict[str, float | None]:
    in_auc = float(roc_auc_score(in_domain_labels, in_domain_scores)) if len(np.unique(in_domain_labels)) > 1 else None
    shift_auc = float(roc_auc_score(shifted_labels, shifted_scores)) if len(np.unique(shifted_labels)) > 1 else None
    drop = None
    if in_auc is not None and shift_auc is not None:
        drop = in_auc - shift_auc
    return {
        "in_domain_auroc": in_auc,
        "shifted_auroc": shift_auc,
        "auroc_drop": drop,
    }
