from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from latent_dynamics.learning_experiment.data import ActivationFeatureBundle

_PROMPT_SUBSETS = {"all", "vanilla", "adversarial"}
_EXPECTED_FAMILIES = {
    "vanilla_benign",
    "vanilla_harmful",
    "adversarial_benign",
    "adversarial_harmful",
}


@dataclass
class ContrastiveProbeConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    hidden_dim: int = 256
    embedding_dim: int = 64
    dropout: float = 0.1
    temperature: float = 0.07
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 10
    train_prompt_subset: str = "all"
    test_prompt_subset: str = "all"
    max_probe_iter: int = 2000
    device: str = "cpu"


@dataclass
class ContrastiveProbeSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class ContrastiveProbeResult:
    config: dict[str, Any]
    label_field: str
    prompt_subset: str
    prompt_group_counts: dict[str, int] | None
    family_vocabulary: list[str]
    layers: list[int]
    n_examples: int
    split: dict[str, Any]
    per_layer: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ConstantProbe:
    def __init__(self, label: int) -> None:
        self.label = int(label)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        score = 1.0 if self.label == 1 else -1.0
        return np.full(X.shape[0], score, dtype=np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = 1.0 if self.label == 1 else 0.0
        out = np.empty((X.shape[0], 2), dtype=np.float32)
        out[:, 1] = p1
        out[:, 0] = 1.0 - p1
        return out


class _ContrastiveMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, p=2, dim=1)


def _safe_train_test_split(
    indices: np.ndarray,
    labels: np.ndarray,
    test_size: float | int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    stratify = labels if np.unique(labels).size > 1 else None
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def _validate_config(config: ContrastiveProbeConfig) -> None:
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if not 0.0 < config.val_size < 1.0:
        raise ValueError("val_size must be in (0, 1).")
    if config.batch_size <= 1:
        raise ValueError("batch_size must be > 1.")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be > 0.")
    if config.embedding_dim <= 0:
        raise ValueError("embedding_dim must be > 0.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if config.temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if config.max_epochs <= 0:
        raise ValueError("max_epochs must be > 0.")
    if config.patience < 0:
        raise ValueError("patience must be >= 0.")
    if config.max_probe_iter <= 0:
        raise ValueError("max_probe_iter must be > 0.")
    if config.train_prompt_subset not in _PROMPT_SUBSETS:
        raise ValueError(
            f"Unknown train_prompt_subset '{config.train_prompt_subset}'. "
            f"Expected one of {sorted(_PROMPT_SUBSETS)}."
        )
    if config.test_prompt_subset not in _PROMPT_SUBSETS:
        raise ValueError(
            f"Unknown test_prompt_subset '{config.test_prompt_subset}'. "
            f"Expected one of {sorted(_PROMPT_SUBSETS)}."
        )


def _count_values(
    values: np.ndarray | None, indices: np.ndarray
) -> dict[str, int] | None:
    if values is None:
        return None
    subset = values[np.asarray(indices, dtype=np.int64)]
    if subset.size == 0:
        return {}
    unique, counts = np.unique(subset.astype(object), return_counts=True)
    return {
        str(value): int(count)
        for value, count in zip(unique.tolist(), counts.tolist(), strict=False)
    }


def _indices_for_subset(
    prompt_groups: np.ndarray | None,
    subset: str,
    n_total: int,
) -> np.ndarray:
    if subset not in _PROMPT_SUBSETS:
        raise ValueError(
            f"Unknown prompt_subset '{subset}'. Expected one of {sorted(_PROMPT_SUBSETS)}."
        )
    if subset == "all":
        return np.arange(n_total, dtype=np.int64)
    if prompt_groups is None:
        raise ValueError(f"prompt_groups missing, cannot filter subset '{subset}'.")
    if prompt_groups.shape[0] != n_total:
        raise ValueError(
            "prompt_groups length does not match labels length. "
            f"Got prompt_groups={prompt_groups.shape[0]} labels={n_total}."
        )
    return np.where(prompt_groups.astype(object) == subset)[0].astype(np.int64)


def _split_train_val(
    trainval_indices: np.ndarray,
    binary_labels: np.ndarray,
    family_targets: np.ndarray,
    config: ContrastiveProbeConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    val_count = int(round(float(config.val_size) * float(trainval_indices.shape[0])))
    val_count = max(1, val_count)
    if val_count >= int(trainval_indices.shape[0]):
        raise ValueError(
            "val_size leaves no training examples. "
            f"Got trainval_size={trainval_indices.shape[0]} val_count={val_count}."
        )

    family_subset = family_targets[trainval_indices]
    if np.unique(family_subset).size > 1:
        try:
            train_idx, val_idx = train_test_split(
                trainval_indices,
                test_size=val_count,
                random_state=config.random_state + 2,
                stratify=family_subset,
            )
            return (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(val_idx, dtype=np.int64),
                "family",
            )
        except ValueError:
            pass

    binary_subset = binary_labels[trainval_indices]
    train_idx, val_idx = _safe_train_test_split(
        indices=trainval_indices,
        labels=binary_subset,
        test_size=val_count,
        random_state=config.random_state + 2,
    )
    if np.unique(binary_subset).size > 1:
        return train_idx, val_idx, "binary"
    return train_idx, val_idx, "none"


def make_contrastive_probe_split(
    bundle: ActivationFeatureBundle,
    config: ContrastiveProbeConfig,
) -> ContrastiveProbeSplit:
    labels = bundle.labels.astype(np.int64)
    n = int(labels.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 examples to run train/val/test splitting.")

    train_candidates = _indices_for_subset(
        prompt_groups=bundle.prompt_groups,
        subset=config.train_prompt_subset,
        n_total=n,
    )
    if train_candidates.size == 0:
        raise ValueError(
            f"No examples available for train_prompt_subset='{config.train_prompt_subset}'."
        )

    test_candidates = _indices_for_subset(
        prompt_groups=bundle.prompt_groups,
        subset=config.test_prompt_subset,
        n_total=n,
    )
    if test_candidates.size == 0:
        raise ValueError(
            f"No examples available for test_prompt_subset='{config.test_prompt_subset}'."
        )

    desired_test_count = int(
        round(float(config.test_size) * float(test_candidates.shape[0]))
    )
    desired_test_count = max(1, desired_test_count)
    desired_test_count = min(desired_test_count, int(test_candidates.shape[0]))

    if desired_test_count >= int(test_candidates.shape[0]):
        test_indices = np.asarray(test_candidates, dtype=np.int64)
    else:
        _, test_indices = _safe_train_test_split(
            indices=test_candidates,
            labels=labels[test_candidates],
            test_size=desired_test_count,
            random_state=config.random_state,
        )

    trainval_indices = np.setdiff1d(
        train_candidates,
        test_indices,
        assume_unique=False,
    ).astype(np.int64)
    if trainval_indices.size < 2:
        raise ValueError(
            "No train/val examples remain after removing held-out test indices. "
            f"train_prompt_subset='{config.train_prompt_subset}' "
            f"test_prompt_subset='{config.test_prompt_subset}'."
        )

    if bundle.example_families is None:
        raise ValueError(
            "example_families missing from activation metadata. "
            "Contrastive probe v1 requires example_metadata.data_type."
        )
    family_targets = bundle.example_families.astype(object)
    if np.any(np.isin(family_targets[trainval_indices], ["", "other", None])):
        raise ValueError(
            "Contrastive probe v1 requires non-empty data_type for all train/val examples."
        )

    train_indices, val_indices, _ = _split_train_val(
        trainval_indices=trainval_indices,
        binary_labels=labels,
        family_targets=family_targets,
        config=config,
    )
    return ContrastiveProbeSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=np.asarray(test_indices, dtype=np.int64),
    )


def _train_logistic_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    max_iter: int,
) -> Pipeline | _ConstantProbe:
    classes = np.unique(y_train)
    if classes.size < 2:
        return _ConstantProbe(label=int(classes[0]))

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    try:
        model.fit(X_train, y_train)
    except Exception:
        majority = int(np.round(float(y_train.mean())))
        return _ConstantProbe(label=majority)
    return model


def _evaluate_binary_probe(
    probe: Pipeline | _ConstantProbe,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prompt_groups: np.ndarray | None,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    scores = probe.decision_function(X_test).astype(np.float32)
    y_pred = (scores >= 0.0).astype(np.int64)
    accuracy = float(accuracy_score(y_test, y_pred))
    if np.unique(y_test).size > 1:
        auroc: float | None = float(roc_auc_score(y_test, scores))
    else:
        auroc = None
    positive_rate = float(np.mean(y_test))

    subgroup_metrics: dict[str, dict[str, float | int | None]] | None = None
    if prompt_groups is not None:
        subgroup_metrics = {}
        group_values = prompt_groups[test_indices]
        for group in sorted(set(str(x) for x in group_values)):
            mask = group_values == group
            if int(mask.sum()) == 0:
                continue
            y_group = y_test[mask]
            score_group = scores[mask]
            pred_group = y_pred[mask]
            group_acc = float(accuracy_score(y_group, pred_group))
            if np.unique(y_group).size > 1:
                group_auc: float | None = float(roc_auc_score(y_group, score_group))
            else:
                group_auc = None
            subgroup_metrics[group] = {
                "n": int(mask.sum()),
                "accuracy": group_acc,
                "auroc": group_auc,
                "positive_rate": float(np.mean(y_group)),
            }

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "positive_rate": positive_rate,
        "subgroup_metrics": subgroup_metrics,
    }


def _supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings to be 2D, got shape {tuple(embeddings.shape)}."
        )
    if labels.ndim != 1:
        raise ValueError(f"Expected labels to be 1D, got shape {tuple(labels.shape)}.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have the same batch size.")

    n = embeddings.shape[0]
    if n <= 1:
        return embeddings.new_tensor(0.0), 0

    logits = embeddings @ embeddings.T
    logits = logits / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    identity = torch.eye(n, device=embeddings.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positive_mask = positive_mask & ~identity

    exp_logits = torch.exp(logits) * (~identity).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_counts = positive_mask.sum(dim=1)
    valid_mask = positive_counts > 0
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return embeddings.new_tensor(0.0), 0

    mean_log_prob_pos = (positive_mask.to(log_prob.dtype) * log_prob).sum(
        dim=1
    ) / positive_counts.clamp_min(1).to(log_prob.dtype)
    loss = -mean_log_prob_pos[valid_mask].mean()
    return loss, valid_count


def _make_stratified_batches(
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape[0] == 0:
        return []
    unique_labels = np.unique(labels)
    label_to_positions = {
        int(label): rng.permutation(np.where(labels == label)[0])
        for label in unique_labels
    }
    label_cursors = {int(label): 0 for label in unique_labels}
    per_label = max(1, batch_size // max(1, len(unique_labels)))

    batches: list[np.ndarray] = []
    while True:
        batch_parts: list[np.ndarray] = []
        any_added = False
        for label in unique_labels:
            key = int(label)
            positions = label_to_positions[key]
            cursor = label_cursors[key]
            if cursor >= positions.shape[0]:
                continue
            next_cursor = min(cursor + per_label, positions.shape[0])
            chosen = positions[cursor:next_cursor]
            if chosen.size > 0:
                batch_parts.append(chosen)
                any_added = True
            label_cursors[key] = next_cursor

        if not any_added:
            break

        batch = np.concatenate(batch_parts).astype(np.int64)
        if batch.shape[0] > batch_size:
            batch = batch[:batch_size]
        batches.append(batch)

    if not batches:
        order = rng.permutation(labels.shape[0]).astype(np.int64)
        return [order]

    return batches


def _compute_epoch_loss(
    model: _ContrastiveMLP,
    X: np.ndarray,
    y: np.ndarray,
    temperature: float,
    device: torch.device,
) -> tuple[float, int]:
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(y.astype(np.int64)).to(device)
    with torch.no_grad():
        embeddings = model(X_tensor)
        loss, valid_count = _supervised_contrastive_loss(
            embeddings=embeddings,
            labels=y_tensor,
            temperature=temperature,
        )
    return float(loss.item()), valid_count


def _encode_numpy(
    model: _ContrastiveMLP,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        embeddings = model(X_tensor).cpu().numpy().astype(np.float32)
    return embeddings


def _train_contrastive_encoder(
    X_train: np.ndarray,
    y_train_family: np.ndarray,
    X_val: np.ndarray,
    y_val_family: np.ndarray,
    config: ContrastiveProbeConfig,
) -> tuple[_ContrastiveMLP, dict[str, Any]]:
    device = torch.device(config.device)
    model = _ContrastiveMLP(
        input_dim=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = math.inf
    epochs_without_improvement = 0
    epochs_trained = 0
    selection_metric = "val_loss"

    rng = np.random.default_rng(config.random_state)
    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        batches = _make_stratified_batches(
            labels=y_train_family,
            batch_size=config.batch_size,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        for batch_positions in batches:
            batch_X = torch.from_numpy(X_train[batch_positions].astype(np.float32)).to(
                device
            )
            batch_y = torch.from_numpy(
                y_train_family[batch_positions].astype(np.int64)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(batch_X)
            loss, valid_count = _supervised_contrastive_loss(
                embeddings=embeddings,
                labels=batch_y,
                temperature=config.temperature,
            )
            if valid_count == 0:
                continue
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        epochs_trained = epoch + 1
        model.eval()
        val_loss, val_valid = _compute_epoch_loss(
            model=model,
            X=X_val,
            y=y_val_family,
            temperature=config.temperature,
            device=device,
        )
        if val_valid == 0:
            train_loss, _ = _compute_epoch_loss(
                model=model,
                X=X_train,
                y=y_train_family,
                temperature=config.temperature,
                device=device,
            )
            metric_value = train_loss
            selection_metric = "train_loss_fallback"
        else:
            metric_value = val_loss

        if metric_value < best_val_loss:
            best_val_loss = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.patience > 0 and epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_trained": int(epochs_trained),
        "selection_metric": selection_metric,
    }


def _encode_families(
    families: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    observed = {str(value) for value in families.astype(object).tolist()}
    unexpected = sorted(observed.difference(_EXPECTED_FAMILIES))
    if unexpected:
        raise ValueError(
            "Unexpected family labels in example_metadata.data_type: "
            f"{unexpected}. Expected subset of {sorted(_EXPECTED_FAMILIES)}."
        )
    vocabulary = sorted(observed)
    lookup = {label: idx for idx, label in enumerate(vocabulary)}
    encoded = np.array(
        [lookup[str(value)] for value in families.astype(object)], dtype=np.int64
    )
    return encoded, vocabulary


def _group_counts(values: np.ndarray | None) -> dict[str, int] | None:
    if values is None:
        return None
    if values.shape[0] == 0:
        return {}
    unique, counts = np.unique(values.astype(object), return_counts=True)
    return {
        str(value): int(count)
        for value, count in zip(unique.tolist(), counts.tolist(), strict=False)
    }


def _run_single_layer(
    X_layer: np.ndarray,
    labels: np.ndarray,
    prompt_groups: np.ndarray | None,
    family_targets: np.ndarray,
    split: ContrastiveProbeSplit,
    config: ContrastiveProbeConfig,
) -> dict[str, Any]:
    train_idx = split.train_indices
    val_idx = split.val_indices
    test_idx = split.test_indices

    X_train = X_layer[train_idx]
    X_val = X_layer[val_idx]
    X_test = X_layer[test_idx]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    y_train_family = family_targets[train_idx]
    y_val_family = family_targets[val_idx]

    raw_probe = _train_logistic_probe(
        X_train=np.concatenate([X_train, X_val], axis=0),
        y_train=np.concatenate([y_train, y_val], axis=0),
        random_state=config.random_state,
        max_iter=config.max_probe_iter,
    )
    raw_metrics = _evaluate_binary_probe(
        probe=raw_probe,
        X_test=X_test,
        y_test=y_test,
        prompt_groups=prompt_groups,
        test_indices=test_idx,
    )

    encoder, training_summary = _train_contrastive_encoder(
        X_train=X_train_scaled,
        y_train_family=y_train_family,
        X_val=X_val_scaled,
        y_val_family=y_val_family,
        config=config,
    )
    device = torch.device(config.device)
    train_embeddings = _encode_numpy(encoder, X_train_scaled, device=device)
    val_embeddings = _encode_numpy(encoder, X_val_scaled, device=device)
    test_embeddings = _encode_numpy(encoder, X_test_scaled, device=device)
    contrastive_probe = _train_logistic_probe(
        X_train=np.concatenate([train_embeddings, val_embeddings], axis=0),
        y_train=np.concatenate([y_train, y_val], axis=0),
        random_state=config.random_state,
        max_iter=config.max_probe_iter,
    )
    contrastive_metrics = _evaluate_binary_probe(
        probe=contrastive_probe,
        X_test=test_embeddings,
        y_test=y_test,
        prompt_groups=prompt_groups,
        test_indices=test_idx,
    )

    return {
        "raw_probe": raw_metrics,
        "contrastive_probe": contrastive_metrics,
        "encoder_training": training_summary,
        "class_balance": {
            "train": _count_values(labels, train_idx),
            "val": _count_values(labels, val_idx),
            "test": _count_values(labels, test_idx),
        },
        "family_balance": {
            "train": _count_values(family_targets, train_idx),
            "val": _count_values(family_targets, val_idx),
            "test": _count_values(family_targets, test_idx),
        },
    }


def run_contrastive_probe_experiment(
    bundle: ActivationFeatureBundle,
    config: ContrastiveProbeConfig,
) -> ContrastiveProbeResult:
    _validate_config(config)
    if bundle.example_families is None:
        raise ValueError(
            "Activation bundle is missing exact example families. "
            "Expected example_metadata.data_type in the activation metadata."
        )

    split = make_contrastive_probe_split(bundle=bundle, config=config)
    family_targets, family_vocabulary = _encode_families(bundle.example_families)
    labels = bundle.labels.astype(np.int64)

    per_layer: dict[str, dict[str, Any]] = {}
    for layer in sorted(bundle.layers):
        X_layer = bundle.features_by_layer[layer].astype(np.float32)
        per_layer[str(layer)] = _run_single_layer(
            X_layer=X_layer,
            labels=labels,
            prompt_groups=bundle.prompt_groups,
            family_targets=family_targets,
            split=split,
            config=config,
        )

    split_summary = {
        "train_indices": split.train_indices.tolist(),
        "val_indices": split.val_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
        "train_size": int(split.train_indices.shape[0]),
        "val_size": int(split.val_indices.shape[0]),
        "test_size": int(split.test_indices.shape[0]),
        "train_prompt_subset": config.train_prompt_subset,
        "test_prompt_subset": config.test_prompt_subset,
        "train_prompt_group_counts": _count_values(
            bundle.prompt_groups, split.train_indices
        ),
        "val_prompt_group_counts": _count_values(
            bundle.prompt_groups, split.val_indices
        ),
        "test_prompt_group_counts": _count_values(
            bundle.prompt_groups, split.test_indices
        ),
    }

    return ContrastiveProbeResult(
        config=asdict(config),
        label_field=bundle.label_field,
        prompt_subset=bundle.prompt_subset,
        prompt_group_counts=_group_counts(bundle.prompt_groups),
        family_vocabulary=family_vocabulary,
        layers=sorted(bundle.layers),
        n_examples=int(bundle.labels.shape[0]),
        split=split_summary,
        per_layer=per_layer,
    )
