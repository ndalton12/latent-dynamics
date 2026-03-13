"""Trajectory feature extractors for safe/unsafe prefix detection.

Two complementary approaches:

1. Geometric turning-angle features (N-GLARE-inspired, arXiv:2511.14195):
   Fit a benign manifold from safe training states. At each token step,
   measure the angle between the trajectory tangent and the outward normal
   to the manifold. Deviation from pi/2 signals movement *away* from the
   safe region.

2. Path signature features (rough path theory):
   Compute the truncated iterated-integral signature of each prefix.
   The signature is a universal, reparameterization-invariant feature map
   for sequential data -- a linear classifier on signatures is as expressive
   as any continuous functional on paths.

Both produce dict[int, np.ndarray] score maps (prompt index -> per-prefix
scores) that are directly compatible with the benchmark suite's
_calibrate_prefix_thresholds and _evaluate_detection functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Shared PCA reduction
# ---------------------------------------------------------------------------


def fit_trajectory_pca(
    trajectories: list[np.ndarray],
    train_idx: np.ndarray,
    n_components: int,
    seed: int = 42,
) -> PCA:
    """Fit PCA on all token states from training trajectories.

    Concatenates every hidden state across all training prompts, so PCA
    sees the full distribution of token representations, not just pooled
    representations.
    """
    states = np.concatenate([trajectories[int(i)] for i in train_idx], axis=0).astype(
        np.float32
    )
    max_components = int(min(states.shape[0], states.shape[1]))
    effective_n = int(min(n_components, max_components))
    if effective_n < 1:
        raise ValueError("n_components resolved to < 1.")
    pca = PCA(n_components=effective_n, random_state=seed)
    pca.fit(states)
    return pca


def reduce_trajectories(
    trajectories: list[np.ndarray],
    pca: PCA,
) -> list[np.ndarray]:
    """Project every trajectory through the fitted PCA."""
    return [pca.transform(t.astype(np.float32)) for t in trajectories]


# ---------------------------------------------------------------------------
# Geometric turning-angle features
# ---------------------------------------------------------------------------


@dataclass
class BenignManifold:
    """Low-rank linear approximation of the safe activation distribution.

    Attributes:
        mean:   (d,)    centroid of safe states
        basis:  (d, r)  top-r eigenvectors (columns) of safe state covariance
    """

    mean: np.ndarray
    basis: np.ndarray

    def project(self, h: np.ndarray) -> np.ndarray:
        """Project h onto the manifold subspace. h shape: (..., d)."""
        centered = h - self.mean
        coords = centered @ self.basis  # (..., r)
        return self.mean + coords @ self.basis.T  # (..., d)

    def residual(self, h: np.ndarray) -> np.ndarray:
        """Component of h orthogonal to the manifold (outward normal direction)."""
        return h - self.project(h)


def fit_benign_manifold(
    trajectories_pca: list[np.ndarray],
    train_idx: np.ndarray,
    labels: np.ndarray,
    rank: int = 8,
) -> BenignManifold:
    """Fit a low-rank benign manifold from safe (label=0) training states.

    Uses the top-r eigenvectors of the safe-state covariance, following
    the N-GLARE construction (Appendix C of arXiv:2511.14195).
    """
    safe_idx = [int(i) for i in train_idx if labels[int(i)] == 0]
    if len(safe_idx) < 2:
        raise ValueError(
            "Need at least 2 safe training trajectories to fit benign manifold."
        )

    safe_states = np.concatenate(
        [trajectories_pca[i] for i in safe_idx], axis=0
    ).astype(np.float32)

    mean = safe_states.mean(axis=0)
    centered = safe_states - mean
    n_states, d = centered.shape
    effective_rank = int(min(rank, n_states - 1, d))
    if effective_rank < 1:
        raise ValueError("Benign manifold rank resolved to < 1.")

    # Economy SVD: columns of V are eigenvectors of the covariance
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:effective_rank].T.astype(np.float32)  # (d, rank)

    return BenignManifold(mean=mean, basis=basis)


def compute_turning_angles(
    trajectory: np.ndarray,
    manifold: BenignManifold,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute the turning angle at each step of the trajectory.

    The turning angle theta_t is the angle between:
      - the trajectory tangent tau_t = h_{t+1} - h_t
      - the outward normal n_t = residual of h_t from the benign manifold

    A small angle means the trajectory is moving *away* from the safe
    region. We return (pi/2 - theta_t) so that positive values indicate
    outward drift, making anomaly scores intuitive.

    Returns an array of shape (T-1,). Empty array if T < 2.
    """
    if trajectory.shape[0] < 2:
        return np.array([], dtype=np.float32)

    tangents = np.diff(trajectory, axis=0).astype(np.float32)  # (T-1, d)
    normals = manifold.residual(trajectory[:-1]).astype(np.float32)  # (T-1, d)

    tau_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    n_norms = np.linalg.norm(normals, axis=1, keepdims=True)

    # Compute cosine of the angle; clamp for numerical safety
    cos_theta = np.sum(tangents * normals, axis=1) / (
        (tau_norms * n_norms).squeeze(-1) + eps
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta).astype(np.float32)

    # Deviation from orthogonal baseline (pi/2): positive = outward drift
    return (np.pi / 2.0 - theta).astype(np.float32)


def turning_angle_score_map(
    trajectories: list[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    pca_n_components: int = 16,
    manifold_rank: int = 8,
    max_prefix: int = 32,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Build a per-prefix anomaly score map using turning-angle geometry.

    Score at prefix p = cumulative mean of (pi/2 - theta_t) for t=1..p-1.
    Higher scores indicate greater outward drift from the benign manifold.

    Fits manifold on safe training states only, so no label leakage at
    inference time.
    """
    pca = fit_trajectory_pca(
        trajectories, train_idx, n_components=pca_n_components, seed=seed
    )
    trajs_pca = reduce_trajectories(trajectories, pca)
    manifold = fit_benign_manifold(trajs_pca, train_idx, labels, rank=manifold_rank)

    score_map: dict[int, np.ndarray] = {}
    for i, traj in enumerate(trajs_pca):
        angles = compute_turning_angles(traj, manifold)
        if len(angles) == 0:
            score_map[i] = np.array([], dtype=np.float32)
            continue
        max_p = min(max_prefix, int(len(angles)))
        angles_clipped = angles[:max_p]
        # Cumulative mean up to each prefix position
        cumscores = np.cumsum(angles_clipped) / np.arange(
            1, len(angles_clipped) + 1, dtype=np.float32
        )
        score_map[i] = cumscores.astype(np.float32)

    return score_map


# ---------------------------------------------------------------------------
# Path signature features
# ---------------------------------------------------------------------------


def _check_iisignature() -> Any:
    try:
        import iisignature

        return iisignature
    except ImportError as e:
        raise ImportError(
            "iisignature is required for path signature features. "
            "Install it with: pip install iisignature"
        ) from e


def compute_prefix_signatures(
    trajectory: np.ndarray,
    max_prefix: int,
    depth: int = 2,
    add_time: bool = True,
) -> dict[int, np.ndarray]:
    """Compute the truncated signature of each prefix of the trajectory.

    Args:
        trajectory:  (T, d) PCA-reduced trajectory
        max_prefix:  compute signatures for prefixes of length 2..max_prefix
        depth:       signature truncation depth (2 recommended for d up to ~20)
        add_time:    prepend a normalized time coordinate t/(T-1) to each step
                     so that the signature captures when events happen, not
                     just the geometric shape

    Returns:
        dict mapping prefix length -> signature vector (depth-2 length is
        d + d^2, or (d+1) + (d+1)^2 when add_time=True)
    """
    iisignature = _check_iisignature()

    T, d = trajectory.shape
    if add_time:
        time_col = (np.arange(T) / max(T - 1, 1)).reshape(-1, 1).astype(np.float32)
        path = np.concatenate([time_col, trajectory.astype(np.float32)], axis=1)
    else:
        path = trajectory.astype(np.float32)

    sigs: dict[int, np.ndarray] = {}
    for p in range(2, min(max_prefix, T) + 1):
        sig = iisignature.sig(path[:p], depth)
        sigs[p] = sig.astype(np.float32)

    return sigs


def signature_prefix_score_map(
    trajectories: list[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    pca_n_components: int = 16,
    depth: int = 2,
    add_time: bool = True,
    max_prefix: int = 32,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Build a per-prefix score map using path signature features + logistic regression.

    At each prefix length p, trains a logistic regression on the signature
    vectors of training trajectories (truncated to p tokens). Then scores
    all trajectories with P(unsafe | prefix_p).

    This is the signature analogue of the baseline_prefix_mean_lr method --
    it learns a linear classifier but over a richer, model-free feature space
    that captures the full shape of the trajectory up to token p.
    """
    iisignature = _check_iisignature()  # noqa: F841 -- validate import early

    pca = fit_trajectory_pca(
        trajectories, train_idx, n_components=pca_n_components, seed=seed
    )
    trajs_pca = reduce_trajectories(trajectories, pca)

    max_len = min(max_prefix, max(int(t.shape[0]) for t in trajs_pca))
    per_prefix_models: dict[int, Pipeline] = {}

    for p in range(2, max_len + 1):
        fit_ids = [int(i) for i in train_idx if trajs_pca[int(i)].shape[0] >= p]
        if len(fit_ids) < 4:
            break
        y = np.array([labels[i] for i in fit_ids], dtype=np.int64)
        if len(np.unique(y)) < 2:
            break

        sigs = []
        for i in fit_ids:
            prefix_sigs = compute_prefix_signatures(
                trajs_pca[i], max_prefix=p, depth=depth, add_time=add_time
            )
            if p not in prefix_sigs:
                break
            sigs.append(prefix_sigs[p])
        else:
            X = np.stack(sigs, axis=0)
            clf = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000, class_weight="balanced", random_state=seed
                        ),
                    ),
                ]
            )
            clf.fit(X, y)
            per_prefix_models[p] = clf

    score_map: dict[int, np.ndarray] = {}
    for i, traj in enumerate(trajs_pca):
        vals: list[float] = []
        for p in range(2, max_prefix + 1):
            clf = per_prefix_models.get(p)
            if clf is None or traj.shape[0] < p:
                break
            prefix_sigs = compute_prefix_signatures(
                traj, max_prefix=p, depth=depth, add_time=add_time
            )
            if p not in prefix_sigs:
                break
            feat = prefix_sigs[p].reshape(1, -1)
            vals.append(float(clf.predict_proba(feat)[0, 1]))
        score_map[i] = np.array(vals, dtype=np.float32)

    return score_map
