from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    calib_size: float = 0.25,
    random_state: int = 7,
) -> tuple[Pipeline, tuple, tuple, tuple, dict]:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full,
        test_size=calib_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    probe.fit(X_train, y_train)

    test_probs = probe.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(np.int64)

    metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "report": classification_report(y_test, test_preds),
    }

    return (
        probe,
        (X_train, y_train),
        (X_calib, y_calib),
        (X_test, y_test),
        metrics,
    )


def _base_direction_method(method: str) -> str:
    return method.removeprefix("sparse_")


def make_direction(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    probe: Pipeline | None = None,
) -> np.ndarray:
    base = _base_direction_method(method)

    if base == "probe_weight":
        if probe is None:
            raise ValueError("Probe is required for probe_weight method.")
        scaler: StandardScaler = probe.named_steps["scaler"]
        clf: LogisticRegression = probe.named_steps["clf"]
        direction = clf.coef_[0] / scaler.scale_
    elif base == "mean_diff":
        direction = (
            X_train[y_train == 1].mean(axis=0)
            - X_train[y_train == 0].mean(axis=0)
        )
    elif base == "pca":
        pca = PCA(n_components=1)
        pca.fit(X_train)
        direction = pca.components_[0]
    else:
        raise ValueError(f"Unknown direction method: {method}")

    norm = np.linalg.norm(direction) + 1e-12
    return direction / norm


def lat_scan(
    trajectories: list[np.ndarray],
    direction: np.ndarray,
) -> list[np.ndarray]:
    return [traj @ direction for traj in trajectories]


def fit_trust_region(
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    q: float = 0.95,
) -> tuple[np.ndarray, float]:
    X_safe = X_calib[y_calib == 0]
    if len(X_safe) < 2:
        raise ValueError("Need at least 2 safe calibration points.")
    mu = X_safe.mean(axis=0)
    dists = np.linalg.norm(X_safe - mu, axis=1)
    tau = float(np.quantile(dists, q))
    return mu, tau


def drift_curve(
    traj: np.ndarray,
    safe_center: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(traj - safe_center[None, :], axis=1)
