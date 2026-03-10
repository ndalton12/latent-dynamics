from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TrustRegionModel:
    feature_names: list[str]
    feature_weights: dict[str, float]
    safe_mean: np.ndarray
    safe_cov_inv: np.ndarray
    threshold_tau: float
    safe_coverage_target: float


def split_indices_70_15_15(labels: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/calib/test indices with 70/15/15 split."""
    idx = np.arange(len(labels))
    train_idx, tmp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    calib_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=labels[tmp_idx] if len(np.unique(labels[tmp_idx])) > 1 else None,
    )
    return np.sort(train_idx), np.sort(calib_idx), np.sort(test_idx)


def concept_churn(supports: list[np.ndarray]) -> float:
    """Fractional support-set change between consecutive steps (Jaccard distance mean)."""
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


def trajectory_curvature(z: np.ndarray) -> float:
    """Sum of angle changes between consecutive step vectors."""
    if z.shape[0] < 3:
        return 0.0
    v1 = z[1:-1] - z[:-2]
    v2 = z[2:] - z[1:-1]
    denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + 1e-8
    cos_theta = np.clip(np.sum(v1 * v2, axis=1) / denom, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    return float(np.sum(angles))


def mahalanobis_to_benign(z: np.ndarray, mu_safe: np.ndarray, sigma_inv_safe: np.ndarray) -> np.ndarray:
    """Per-token Mahalanobis distance to safe reference."""
    diffs = z - mu_safe[None, :]
    return np.sqrt(np.einsum("nd,dd,nd->n", diffs, sigma_inv_safe, diffs).clip(min=0.0))


def compute_features_for_trajectory(
    z: np.ndarray,
    mu_safe: np.ndarray,
    sigma_inv_safe: np.ndarray,
    sparse_supports: list[np.ndarray] | None = None,
) -> dict[str, float]:
    if z.shape[0] < 2:
        speed = 0.0
    else:
        speed = float(np.mean(np.linalg.norm(z[1:] - z[:-1], axis=1)))
    curv = trajectory_curvature(z)
    maha = float(np.mean(mahalanobis_to_benign(z, mu_safe, sigma_inv_safe)))
    churn = concept_churn(sparse_supports) if sparse_supports is not None else 0.0
    return {
        "speed": speed,
        "churn": churn,
        "curvature": curv,
        "mahalanobis": maha,
    }


def compute_conformity(features: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * float(v) for k, v in features.items()))


def _fit_safe_reference(safe_trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_safe = np.concatenate(safe_trajectories, axis=0).astype(np.float64)
    mu = all_safe.mean(axis=0)
    cov = np.cov(all_safe, rowvar=False)
    cov = cov + (1e-5 * np.eye(cov.shape[0], dtype=np.float64))
    cov_inv = np.linalg.pinv(cov)
    return mu.astype(np.float32), cov_inv.astype(np.float32)


def _calibrate_tau_with_mapie(safe_scores: np.ndarray, safe_coverage_target: float) -> float:
    """
    Calibrate threshold tau with MAPIE split-conformal if available.
    Falls back to empirical conformal quantile if MAPIE is unavailable.
    """
    alpha = 1.0 - safe_coverage_target
    try:
        from mapie.regression import MapieRegressor
        from sklearn.dummy import DummyRegressor

        x = np.arange(len(safe_scores), dtype=np.float32).reshape(-1, 1)
        y = safe_scores.astype(np.float32)
        model = MapieRegressor(estimator=DummyRegressor(strategy="mean"), method="plus", cv="split")
        model.fit(x, y)
        _, intervals = model.predict(x, alpha=alpha)
        if intervals.ndim == 3:
            upper = intervals[:, 1, 0]
        else:
            upper = intervals[:, 1]
        return float(np.max(upper))
    except Exception:
        # Split-conformal quantile fallback.
        vals = np.sort(safe_scores.astype(np.float64))
        n = len(vals)
        rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
        rank = max(0, min(n - 1, rank))
        return float(vals[rank])


def fit_trust_region_model(
    trajectories: list[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    feature_weights: dict[str, float] | None = None,
    safe_coverage_target: float = 0.95,
) -> TrustRegionModel:
    if feature_weights is None:
        feature_weights = {
            "speed": 0.15,
            "churn": 0.10,
            "curvature": 0.20,
            "mahalanobis": 0.55,
        }

    safe_train_idx = [int(i) for i in train_idx if labels[int(i)] == 0]
    if not safe_train_idx:
        raise ValueError("No safe examples in training split for trust-region fit.")
    safe_train_trajectories = [trajectories[i] for i in safe_train_idx]
    mu_safe, sigma_inv_safe = _fit_safe_reference(safe_train_trajectories)

    safe_calib_idx = [int(i) for i in calib_idx if labels[int(i)] == 0]
    if not safe_calib_idx:
        raise ValueError("No safe examples in calibration split for threshold fit.")
    safe_scores = []
    for i in safe_calib_idx:
        feats = compute_features_for_trajectory(trajectories[i], mu_safe, sigma_inv_safe)
        safe_scores.append(compute_conformity(feats, feature_weights))
    tau = _calibrate_tau_with_mapie(np.array(safe_scores, dtype=np.float32), safe_coverage_target)

    return TrustRegionModel(
        feature_names=["speed", "churn", "curvature", "mahalanobis"],
        feature_weights=feature_weights,
        safe_mean=mu_safe,
        safe_cov_inv=sigma_inv_safe,
        threshold_tau=tau,
        safe_coverage_target=safe_coverage_target,
    )


def trust_region_scores(
    trajectories: list[np.ndarray],
    model: TrustRegionModel,
) -> np.ndarray:
    out = []
    for traj in trajectories:
        feats = compute_features_for_trajectory(traj, model.safe_mean, model.safe_cov_inv)
        out.append(compute_conformity(feats, model.feature_weights))
    return np.array(out, dtype=np.float32)


def per_token_mahalanobis_curve(traj: np.ndarray, model: TrustRegionModel) -> np.ndarray:
    return mahalanobis_to_benign(traj, model.safe_mean, model.safe_cov_inv).astype(np.float32)


def baseline_brt_align_scores(
    trajectories: list[np.ndarray],
    model: TrustRegionModel,
) -> np.ndarray:
    """Simplified backward reachability: max suffix Mahalanobis score."""
    scores = []
    for traj in trajectories:
        curve = per_token_mahalanobis_curve(traj, model)
        suffix_max = np.maximum.accumulate(curve[::-1])[::-1]
        scores.append(float(np.mean(suffix_max)))
    return np.array(scores, dtype=np.float32)


def baseline_nglare_scores(trajectories: list[np.ndarray]) -> np.ndarray:
    """Simplified angular deviation: mean(1 - cosine(step_t, step_t+1))."""
    out = []
    for traj in trajectories:
        if traj.shape[0] < 3:
            out.append(0.0)
            continue
        s1 = traj[1:-1] - traj[:-2]
        s2 = traj[2:] - traj[1:-1]
        denom = (np.linalg.norm(s1, axis=1) * np.linalg.norm(s2, axis=1)) + 1e-8
        cos = np.clip(np.sum(s1 * s2, axis=1) / denom, -1.0, 1.0)
        out.append(float(np.mean(1.0 - cos)))
    return np.array(out, dtype=np.float32)


def baseline_sap_proxy_scores(
    trajectories: list[np.ndarray],
    model: TrustRegionModel,
) -> np.ndarray:
    """Simple SaP proxy: Mahalanobis score on pooled representation."""
    pooled = np.stack([t.mean(axis=0) for t in trajectories], axis=0)
    diffs = pooled - model.safe_mean[None, :]
    vals = np.sqrt(np.einsum("nd,dd,nd->n", diffs, model.safe_cov_inv, diffs).clip(min=0.0))
    return vals.astype(np.float32)


def compute_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    calib_safe_scores: np.ndarray,
    safe_coverage: float = 0.95,
) -> dict[str, float | None]:
    if len(labels) != len(scores):
        raise ValueError("labels and scores length mismatch.")
    auroc = None
    if len(np.unique(labels)) > 1:
        auroc = float(roc_auc_score(labels, scores))

    tau = float(np.quantile(calib_safe_scores, safe_coverage))
    pred_unsafe = scores > tau
    safe_mask = labels == 0
    unsafe_mask = labels == 1
    fpr = float(np.mean(pred_unsafe[safe_mask])) if np.any(safe_mask) else None
    tpr = float(np.mean(pred_unsafe[unsafe_mask])) if np.any(unsafe_mask) else None
    return {
        "auroc": auroc,
        "threshold_tau": tau,
        "fpr_at_95_safe_coverage": fpr,
        "tpr_at_95_safe_coverage": tpr,
    }


def save_trust_region_model(path: Path, model: TrustRegionModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_trust_region_model(path: Path) -> TrustRegionModel:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TrustRegionModel):
        raise TypeError("Loaded object is not TrustRegionModel.")
    return obj


def to_table_rows(
    test_labels: np.ndarray,
    trust_scores: np.ndarray,
    brt_scores: np.ndarray,
    nglare_scores: np.ndarray,
    sap_scores: np.ndarray,
    calib_safe_scores: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    methods = {
        "trust_region": (trust_scores, calib_safe_scores["trust_region"]),
        "brt_align_simplified": (brt_scores, calib_safe_scores["brt_align_simplified"]),
        "nglare_simplified": (nglare_scores, calib_safe_scores["nglare_simplified"]),
        "sap_proxy": (sap_scores, calib_safe_scores["sap_proxy"]),
    }
    for name, (scores, safe_calib) in methods.items():
        m = compute_binary_metrics(test_labels, scores, safe_calib, safe_coverage=0.95)
        rows.append({
            "method": name,
            "auroc": m["auroc"],
            "fpr_at_95_safe_coverage": m["fpr_at_95_safe_coverage"],
            "tpr_at_95_safe_coverage": m["tpr_at_95_safe_coverage"],
        })
    return rows
