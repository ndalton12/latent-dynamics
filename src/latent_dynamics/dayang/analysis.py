from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from latent_dynamics.dayang.activations import Activations, PoolMethod


def _normalize(w: np.ndarray, axis: int = 0, squared: bool = False) -> np.ndarray:
    """Normalize vector along a specified axis.

    Args:
        w: Input array to normalize.
        axis: Axis along which to normalize.
        squared: If True, normalize by squared norm instead of norm.

    Returns:
        Normalized array with the same shape as `w`.
    """
    if squared:
        norm = np.sum(w**2, axis=axis, keepdims=True)
    else:
        norm = np.linalg.norm(w, axis=axis, keepdims=True)
    norm[norm == 0] = 1.0  # avoid division by zero
    return w / norm


def _unscale(
    scaler: StandardScaler,
    w_scaled: np.ndarray,
    b_scaled: np.ndarray | int | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert linear directions and biases from a standardized space back to the original space.

    This maps the decision boundary satisfying `X_scaled @ w_scaled + b_scaled = 0`
    to the original space formulation `(X_orig - b_orig) @ w_orig = 0`, where `b_orig`
    is an intercept point strictly aligned along the normal vector `w_orig`.

    Args:
        scaler: Fitted StandardScaler containing `mean_` and `scale_`.
        w_scaled: Direction vectors of shape (.., num_features).
        b_scaled: Scalar intercepts per component of shape (..,), or a int or float.

    Returns:
        w_orig: Unscaled direction vectors of shape (.., num_features).
        b_orig: Unscaled origin-shift bias vectors of shape (.., num_features).
    """
    if isinstance(b_scaled, (int, float)):
        b_scaled = np.full(w_scaled.shape[:-1], b_scaled)

    # Unscale direction
    w_orig = w_scaled / scaler.scale_

    # Unscale bias
    b_scaled_shifted = np.dot(w_orig, scaler.mean_) - b_scaled
    b_orig = np.expand_dims(b_scaled_shifted, axis=-1) * _normalize(w_orig, axis=-1, squared=True)

    return w_orig, b_orig


class Reader(ABC):
    """Base class for concept-vector extraction methods."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def num_components(self) -> int:
        pass

    @abstractmethod
    def __call__(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Extract concept direction and bias.

        Args:
            X: Data array of shape (num_samples, num_features).
            y: Labels array of shape (num_samples,), optional.

        Returns:
            w: Direction vector(s) of shape (num_components, num_features).
            b: Bias vector(s) of shape (num_components, num_features).
        """
        pass


class PCAReader(Reader):
    """Extracts principal component directions, preserving the origin."""

    def __init__(self, num_components: int = 1, standardized: bool = False):
        self._num_components = num_components
        self.standardized = standardized

    @property
    def name(self) -> str:
        return f"PCA{' (standardized)' if self.standardized else ''}"

    @property
    def num_components(self) -> int:
        return self._num_components

    def __call__(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        # Standardize features
        if self.standardized:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Fit PCA
        pca = PCA(n_components=self.num_components)
        pca.fit(X)
        w = pca.components_
        b = np.zeros_like(w)  # preserve origin by setting bias to zero

        if self.standardized:
            w, b = _unscale(scaler, w, 0.0)  # preserve origin by setting scaled bisa to zero
            w = _normalize(w, axis=1)  # re-normalize after unscaling

        return w, b


class LinearProbeReader(Reader):
    """Extracts concept direction using a logistic regression probe."""

    def __init__(self, normalize: bool = True, max_iter: int = 1000):
        self.normalize = normalize
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "LinearProbe"

    @property
    def num_components(self) -> int:
        return 1

    def __call__(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if y is None:
            raise ValueError("Labels `y` must be provided for LinearProbeReader.")

        # Fit a logistic regression model
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = LogisticRegression(class_weight="balanced", max_iter=self.max_iter, random_state=42)
        model.fit(X, y)
        w = model.coef_[0]
        b = model.intercept_[0]

        # Reconstruct the unscaled direction and bias from the trained model
        w, b = _unscale(scaler, w, b)

        # Normalize direction
        if self.normalize:
            w = _normalize(w)

        return w[None, :], b[None, :]


class DiffInMeanReader(Reader):
    """Extracts concept direction using difference in means of safe and unsafe examples."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "DiffInMean"

    @property
    def num_components(self) -> int:
        return 1

    def __call__(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if y is None:
            raise ValueError("Labels `y` must be provided for DiffInMeanReader.")

        # Compute difference in means between safe and unsafe examples
        safe_mean = X[y == 1].mean(axis=0)
        unsafe_mean = X[y == 0].mean(axis=0)
        w = (safe_mean - unsafe_mean) / 2.0
        b = (safe_mean + unsafe_mean) / 2.0

        # Normalize direction
        if self.normalize:
            w = _normalize(w)
        else:
            w = _normalize(w, squared=True)

        return w[None, :], b[None, :]


def _get_reader(reader: str | Reader) -> Reader:
    if isinstance(reader, Reader):
        return reader
    if reader == "pca":
        return PCAReader()
    elif reader == "pca_standardized":
        return PCAReader(standardized=True)
    elif reader == "linear_probe":
        return LinearProbeReader()
    elif reader == "difference_in_mean":
        return DiffInMeanReader()
    else:
        raise ValueError(f"Unknown reader: {reader}")


def _parse_readers(readers: list[str | Reader]) -> list[Reader]:
    parsed_readers = []
    i = 0
    while i < len(readers):
        reader = readers[i]
        if isinstance(reader, str) and reader in ("pca", "pca_standardized"):
            pca_count = 1
            while i + pca_count < len(readers) and readers[i + pca_count] == reader:
                pca_count += 1
            parsed_readers.append(PCAReader(num_components=pca_count, standardized=(reader == "pca_standardized")))
            i += pca_count
        else:
            parsed_readers.append(_get_reader(reader))
            i += 1
    return parsed_readers


class ActivationReaders(BaseEstimator, TransformerMixin):
    """Sequentially applies linear readers in orthogonal residual subspaces."""

    def __init__(self, readers: list[str | Reader]) -> None:
        self.readers = _parse_readers(readers)
        self.components_: np.ndarray | None = None
        self.intercepts_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    @property
    def labels(self) -> list[str]:
        labels = []
        idx = 0
        for r in self.readers:
            labels.extend([f"{idx + i + 1}: {r.name}" for i in range(r.num_components)])
            idx += r.num_components
        return labels

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, pbar: bool = False) -> ActivationReaders:
        X_residual = X.copy()
        total_variance = np.var(X, axis=0).sum()

        components = []
        intercepts = []
        explained_variance = []

        for reader in tqdm(self.readers, disable=not pbar, leave=False, desc="Fitting readers"):
            w_batch, b_batch = reader(X_residual, y)  # shape: (num_components, num_features)

            for i in range(w_batch.shape[0]):
                w = w_batch[i]
                b = b_batch[i]
                components.append(w)
                intercepts.append(b)
                # Compute projection
                proj = (X - b) @ w
                # Compute explained variance
                var = np.var(proj)
                explained_variance.append(var)
                # Remove orthogonal projection from the residual
                w_norm = _normalize(w)
                X_residual = X_residual - np.outer(X_residual @ w_norm, w_norm)

        self.components_ = np.array(components)
        self.intercepts_ = np.array(intercepts)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.explained_variance_ratio_ = np.array(explained_variance) / total_variance
            self.explained_variance_ = np.array(explained_variance)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects activations onto the extracted components."""
        if self.components_ is None or self.intercepts_ is None:
            raise RuntimeError("ActivationReaders must be fitted before transforming.")

        num_samples = X.shape[0]
        num_components = self.components_.shape[0]
        X_proj = np.zeros((num_samples, num_components))

        for i in range(num_components):
            w = self.components_[i]
            b = self.intercepts_[i]
            X_proj[:, i] = (X - b) @ w

        return X_proj


def analyze_per_layer(
    activations: Activations,
    readers: list[str | Reader] = ["pca", "pca"],
    pool_method: PoolMethod = "all",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    separate: bool = True,
) -> dict[int, ActivationReaders]:
    """Train sequential readers per layer."""
    samples = activations.get(
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )

    if separate:
        # Train a reader separately for each layer
        num_layers = activations.num_layers
        models: dict[int, ActivationReaders] = {}
        for layer_idx, layer in zip(tqdm(range(num_layers), desc="Analyzing per layer"), activations.layers):
            X = np.concatenate(
                [sample["activations"][:, layer_idx] for sample in samples],
                axis=0,
            )
            y = np.concatenate(
                [np.full(len(sample["activations"][:, layer_idx]), int(sample["is_safe"])) for sample in samples],
                axis=0,
            )
            models[layer] = ActivationReaders(readers=readers).fit(X, y)
        return models
    else:
        # Train a single reader on all layers
        for _ in tqdm(range(1), desc="Analyzing all layers"):
            X = np.concatenate(
                [sample["activations"].reshape(-1, sample["activations"].shape[-1]) for sample in samples],
                axis=0,
            )
            y = np.concatenate(
                [np.full(np.prod(sample["activations"].shape[:-1]), int(sample["is_safe"])) for sample in samples],
                axis=0,
            )
            model = ActivationReaders(readers=readers).fit(X, y, pbar=True)
        return defaultdict(lambda: model)


def analyze_per_token(
    activations: Activations,
    readers: list[str | Reader] = ["pca", "pca"],
    pool_method: PoolMethod = "all",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
    separate: bool = True,
) -> dict[int, ActivationReaders]:
    """Train sequential readers per token position."""
    samples = activations.get(
        pool_method=pool_method,
        exclude_bos=exclude_bos,
        exclude_special_tokens=exclude_special_tokens,
    )

    # Check uniform sequence length
    if pool_method == "all" and len(samples) > 1:
        num_tokens = samples[0]["activations"].shape[0]
        if not all(sample["activations"].shape[0] == num_tokens for sample in samples):
            raise ValueError(
                "When analyzing per token with pool_method='all', a uniform sequence length is required."
                " Consider using a different pooling method or setting separate=False."
            )

    if separate:
        # Train a reader separately for each token position
        num_tokens = samples[0]["activations"].shape[0]
        models: dict[int, ActivationReaders] = {}
        for token_idx in tqdm(range(num_tokens), desc="Analyzing per token"):
            X = np.concatenate(
                [sample["activations"][token_idx] for sample in samples],
                axis=0,
            )
            y = np.concatenate(
                [np.full(len(sample["activations"][token_idx]), int(sample["is_safe"])) for sample in samples],
                axis=0,
            )
            models[token_idx] = ActivationReaders(readers=readers).fit(X, y)
        return models
    else:
        # Train a single reader on all token positions
        for _ in tqdm(range(1), desc="Analyzing all tokens"):
            X = np.concatenate(
                [sample["activations"].reshape(-1, sample["activations"].shape[-1]) for sample in samples],
                axis=0,
            )
            y = np.concatenate(
                [np.full(np.prod(sample["activations"].shape[:-1]), int(sample["is_safe"])) for sample in samples],
                axis=0,
            )
            model = ActivationReaders(readers=readers).fit(X, y, pbar=True)
        return defaultdict(lambda: model)
