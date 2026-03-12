from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from latent_dynamics.config import RunConfig
from latent_dynamics.hub import activation_subpath, load_activations, pull_from_hub

ArrayLike = np.ndarray | Sequence[np.ndarray]


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
                "When passing a NumPy array to Reader.train/predict, expected shape (N, D)."
            )
        return activations.astype(np.float32)

    acts = list(activations)
    if not acts:
        raise ValueError("Received empty activations.")

    first = acts[0]
    if not isinstance(first, np.ndarray):
        raise TypeError(
            "Activations must be a 2D feature matrix or a sequence of NumPy arrays."
        )

    if first.ndim == 1:
        return np.stack([a.astype(np.float32) for a in acts], axis=0)

    if first.ndim == 2:
        pooled = [_pool_trajectory(a.astype(np.float32), pooling) for a in acts]
        return np.stack(pooled, axis=0)

    raise ValueError(
        f"Unsupported activation array rank: {first.ndim}. Expected 1D or 2D arrays."
    )


def _as_binary_labels(labels: np.ndarray | Sequence[int]) -> np.ndarray:
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError(f"Labels must be 1D. Got shape {y.shape}.")

    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(
            f"This baseline expects binary labels. Found {classes.size} class(es): {classes}."
        )

    if np.array_equal(classes, np.array([0, 1])):
        return y.astype(np.int64)

    return (y == classes[-1]).astype(np.int64)


class Reader(ABC):
    """Base class for concept-vector readers."""

    def __init__(self, pooling: str = "last") -> None:
        self.pooling = pooling

    @abstractmethod
    def train(
        self,
        activations: ArrayLike,
        labels: np.ndarray | Sequence[int],
    ) -> None:
        """Extract concept representation from activations."""

    @abstractmethod
    def predict(self, activations: ArrayLike) -> np.ndarray:
        """Predict concept score from activations."""

    def _features(self, activations: ArrayLike) -> np.ndarray:
        return _to_feature_matrix(activations, pooling=self.pooling)


class DifferenceInMeanReader(Reader):
    """Concept direction = mean(positive) - mean(negative)."""

    def __init__(self, pooling: str = "last") -> None:
        super().__init__(pooling=pooling)
        self.direction: np.ndarray | None = None
        self.midpoint: np.ndarray | None = None

    def train(self, activations: ArrayLike, labels: np.ndarray | Sequence[int]) -> None:
        X = self._features(activations)
        y = _as_binary_labels(labels)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatched shapes: X has {X.shape[0]} rows, y has {y.shape[0]}."
            )

        mu0 = X[y == 0].mean(axis=0)
        mu1 = X[y == 1].mean(axis=0)
        direction = mu1 - mu0
        norm = np.linalg.norm(direction)
        if norm <= 0:
            raise ValueError(
                "Zero mean-difference direction. Check activations/labels."
            )

        self.direction = direction / norm
        self.midpoint = 0.5 * (mu0 + mu1)

    def predict(self, activations: ArrayLike) -> np.ndarray:
        if self.direction is None or self.midpoint is None:
            raise RuntimeError(
                "DifferenceInMeanReader must be trained before predict()."
            )
        X = self._features(activations)
        return (X - self.midpoint[None, :]) @ self.direction


class PCAReader(Reader):
    """Concept direction = first principal component (sign aligned to labels)."""

    def __init__(self, pooling: str = "last", random_state: int = 42) -> None:
        super().__init__(pooling=pooling)
        self.random_state = random_state
        self.mean: np.ndarray | None = None
        self.direction: np.ndarray | None = None

    def train(self, activations: ArrayLike, labels: np.ndarray | Sequence[int]) -> None:
        X = self._features(activations)
        y = _as_binary_labels(labels)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatched shapes: X has {X.shape[0]} rows, y has {y.shape[0]}."
            )

        pca = PCA(n_components=1, random_state=self.random_state)
        pca.fit(X)

        direction = pca.components_[0].astype(np.float32)
        center = pca.mean_.astype(np.float32)

        scores = (X - center[None, :]) @ direction
        if float(scores[y == 1].mean()) < float(scores[y == 0].mean()):
            direction = -direction

        self.mean = center
        self.direction = direction

    def predict(self, activations: ArrayLike) -> np.ndarray:
        if self.mean is None or self.direction is None:
            raise RuntimeError("PCAReader must be trained before predict().")
        X = self._features(activations)
        return (X - self.mean[None, :]) @ self.direction


class LinearProbeReader(Reader):
    """Concept direction from a logistic linear probe."""

    def __init__(
        self,
        pooling: str = "last",
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        super().__init__(pooling=pooling)
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: Pipeline | None = None
        self.direction: np.ndarray | None = None

    def train(self, activations: ArrayLike, labels: np.ndarray | Sequence[int]) -> None:
        X = self._features(activations)
        y = _as_binary_labels(labels)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatched shapes: X has {X.shape[0]} rows, y has {y.shape[0]}."
            )

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        model.fit(X, y)

        scaler: StandardScaler = model.named_steps["scaler"]
        clf: LogisticRegression = model.named_steps["clf"]
        direction = clf.coef_[0] / scaler.scale_
        norm = np.linalg.norm(direction)
        self.direction = direction / (norm + 1e-12)
        self.model = model

    def predict(self, activations: ArrayLike) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LinearProbeReader must be trained before predict().")
        X = self._features(activations)
        return self.model.decision_function(X).astype(np.float32)


def _is_activation_leaf(path: Path) -> bool:
    return (path / "metadata.json").exists() and (
        path / "trajectories.safetensors"
    ).exists()


def _resolve_activation_leaf(root_or_leaf: Path) -> Path:
    path = root_or_leaf.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Activation path does not exist: {path}")

    if _is_activation_leaf(path):
        return path

    candidates: list[Path] = []
    for metadata in sorted(path.rglob("metadata.json")):
        leaf = metadata.parent
        if _is_activation_leaf(leaf):
            candidates.append(leaf)

    if not candidates:
        raise FileNotFoundError(
            f"No activation leaf found under {path} (expected metadata.json + trajectories.safetensors)."
        )
    return candidates[0]


def load_activation_bundle(
    local_path: Path | str | None = None,
    hf_repo_id: str | None = None,
    dataset_key: str | None = None,
    model_key: str | None = None,
    layer_idx: int | None = None,
    cache_dir: Path = Path(".cache/hub"),
) -> tuple[
    list[np.ndarray],
    list[str],
    np.ndarray | None,
    list[list[str]],
    list[str | None] | None,
    RunConfig,
    Path,
]:
    """Load activations either from local disk or from a Hugging Face dataset repo."""
    if hf_repo_id:
        for name, value in (
            ("dataset_key", dataset_key),
            ("model_key", model_key),
            ("layer_idx", layer_idx),
        ):
            if value is None:
                raise ValueError(f"{name} is required when loading from Hugging Face.")

        subpath = activation_subpath(
            dataset_key=dataset_key,
            model_key=model_key,
            layer_idx=int(layer_idx),
        )
        leaf = (cache_dir / hf_repo_id.replace("/", "__") / subpath).resolve()
        if not _is_activation_leaf(leaf):
            pull_from_hub(
                repo_id=hf_repo_id,
                local_dir=leaf,
                path_in_repo=str(subpath),
            )
    else:
        root = Path(local_path) if local_path is not None else Path("activations")
        leaf = _resolve_activation_leaf(root)

    trajectories, texts, labels, token_texts, generated_texts, cfg = load_activations(
        leaf
    )
    return trajectories, texts, labels, token_texts, generated_texts, cfg, leaf


def _run_smoke_test(args: argparse.Namespace) -> None:
    local_path: Path | None = None if args.from_hub else args.activations
    trajectories, _texts, labels, _token_texts, _generated, cfg, leaf = (
        load_activation_bundle(
            local_path=local_path,
            hf_repo_id=args.from_hub,
            dataset_key=args.hub_dataset,
            model_key=args.hub_model,
            layer_idx=args.hub_layer,
        )
    )
    print(f"Loaded {len(trajectories)} trajectories from {leaf}")
    print(
        f"Config: model={cfg.model_key} dataset={cfg.dataset_key} layer={cfg.layer_idx} pooling={args.pooling}"
    )

    if labels is None:
        raise ValueError(
            "Loaded activations do not contain labels, cannot run baselines."
        )

    y = _as_binary_labels(labels)
    idx = np.arange(len(trajectories))
    stratify = y if np.unique(y).size > 1 else None
    try:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=None,
        )

    train_acts = [trajectories[int(i)] for i in train_idx]
    test_acts = [trajectories[int(i)] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    readers: list[Reader] = [
        DifferenceInMeanReader(pooling=args.pooling),
        PCAReader(pooling=args.pooling, random_state=args.seed),
        LinearProbeReader(pooling=args.pooling, random_state=args.seed),
    ]

    print(f"Train size: {len(train_idx)} | Test size: {len(test_idx)}")
    print("Running baselines:")
    for reader in readers:
        reader_name = reader.__class__.__name__
        reader.train(train_acts, y_train)
        scores = reader.predict(test_acts)
        preds = (scores >= 0).astype(np.int64)
        acc = accuracy_score(y_test, preds)
        if np.unique(y_test).size > 1:
            auc = roc_auc_score(y_test, scores)
            auc_text = f"{auc:.3f}"
        else:
            auc_text = "n/a"
        print(f"- {reader_name}: accuracy={acc:.3f} auroc={auc_text}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Representation-engineering baselines for concept vectors."
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
        help="Layer index for Hub path resolution.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["last", "mean", "max_norm"],
        default="last",
        help="How to pool token trajectories into per-sample vectors.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of samples reserved for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and model initialization.",
    )
    return parser


if __name__ == "__main__":
    parsed_args = _build_arg_parser().parse_args()
    _run_smoke_test(parsed_args)
