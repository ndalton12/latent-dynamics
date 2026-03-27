from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from latent_dynamics.learning_experiment.data import ActivationFeatureBundle

_RULES = {"uncertainty", "disagreement", "uncertainty_diversity"}
_ENSEMBLE_MODES = {"single_layer", "mean_score", "majority_vote"}
_PROMPT_SUBSETS = {"all", "vanilla", "adversarial"}


@dataclass
class ActiveLearningConfig:
    acquisition: str = "dynamic_uncertainty"
    initial_labeled_size: int = 120
    query_budget: int = 480
    batch_size: int = 60
    test_size: float = 0.2
    random_state: int = 42
    ensemble_mode: str = "single_layer"
    scoring_layer: int | None = None
    uncertainty_weight: float = 0.5
    diversity_prefilter_factor: int = 10
    max_probe_iter: int = 2000
    train_prompt_subset: str = "all"
    test_prompt_subset: str = "all"


@dataclass
class ActiveLearningSplit:
    initial_labeled_indices: np.ndarray
    unlabeled_pool_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class RoundResult:
    round_idx: int
    queried_labels: int
    labeled_size: int
    unlabeled_pool_size: int
    test_size: int
    test_accuracy: float
    test_auroc: float | None
    test_positive_rate: float
    subgroup_metrics: dict[str, dict[str, float | int | None]] | None
    ranking_stability: dict[str, float | int | None] | None = None


@dataclass
class ActiveLearningResult:
    config: dict[str, Any]
    label_field: str
    prompt_subset: str
    prompt_group_counts: dict[str, int] | None
    layers: list[int]
    acquisition_mode: str
    acquisition_rule: str | None
    n_examples: int
    initial_class_balance: dict[str, int]
    final_class_balance: dict[str, int]
    ranking_stability_summary: dict[str, float | int | None] | None
    layer_metrics: dict[str, Any] | None
    rounds: list[RoundResult]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rounds"] = [asdict(item) for item in self.rounds]
        return payload


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


def _validate_config(config: ActiveLearningConfig) -> None:
    if config.initial_labeled_size <= 0:
        raise ValueError("initial_labeled_size must be > 0.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if config.query_budget < 0:
        raise ValueError("query_budget must be >= 0.")
    if not 0.0 < config.test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if not 0.0 <= config.uncertainty_weight <= 1.0:
        raise ValueError("uncertainty_weight must be in [0, 1].")
    if config.diversity_prefilter_factor < 1:
        raise ValueError("diversity_prefilter_factor must be >= 1.")
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


def _parse_acquisition(acquisition: str) -> tuple[str, str | None]:
    if acquisition == "random":
        return "random", None

    if acquisition.startswith("static_"):
        rule = acquisition[len("static_") :]
        if rule not in _RULES:
            raise ValueError(f"Unknown static acquisition rule '{rule}'.")
        return "static", rule

    if acquisition.startswith("dynamic_"):
        rule = acquisition[len("dynamic_") :]
        if rule not in _RULES:
            raise ValueError(f"Unknown dynamic acquisition rule '{rule}'.")
        return "dynamic", rule

    raise ValueError(
        f"Unknown acquisition '{acquisition}'. Use one of: random, static_*, dynamic_*."
    )


def _make_diversity_features(features_by_layer: dict[int, np.ndarray]) -> np.ndarray:
    layers = sorted(features_by_layer.keys())
    first = features_by_layer[layers[0]].astype(np.float32)
    if len(layers) == 1:
        return first
    acc = np.zeros_like(first, dtype=np.float32)
    for layer in layers:
        acc += features_by_layer[layer].astype(np.float32)
    acc /= float(len(layers))
    return acc


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def _train_probe(
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


@dataclass
class _ProbeEnsemble:
    models: dict[int, Pipeline | _ConstantProbe]
    layers: list[int]
    mode: str
    scoring_layer: int

    def layer_scores(
        self,
        features_by_layer: dict[int, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        if indices.shape[0] == 0:
            return np.empty((0, len(self.layers)), dtype=np.float32)
        scores = np.empty((indices.shape[0], len(self.layers)), dtype=np.float32)
        for j, layer in enumerate(self.layers):
            X = features_by_layer[layer][indices]
            model = self.models[layer]
            scores[:, j] = model.decision_function(X).astype(np.float32)
        return scores

    def layer_probs(
        self,
        features_by_layer: dict[int, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        if indices.shape[0] == 0:
            return np.empty((0, len(self.layers)), dtype=np.float32)
        probs = np.empty((indices.shape[0], len(self.layers)), dtype=np.float32)
        for j, layer in enumerate(self.layers):
            X = features_by_layer[layer][indices]
            model = self.models[layer]
            probs[:, j] = model.predict_proba(X)[:, 1].astype(np.float32)
        return probs

    def aggregate_scores(self, layer_scores: np.ndarray) -> np.ndarray:
        if self.mode == "single_layer":
            layer_pos = self.layers.index(self.scoring_layer)
            return layer_scores[:, layer_pos]
        return layer_scores.mean(axis=1).astype(np.float32)

    def predict(self, layer_scores: np.ndarray) -> np.ndarray:
        if self.mode == "majority_vote":
            votes = (layer_scores >= 0.0).astype(np.int64)
            threshold = len(self.layers) / 2.0
            vote_sum = votes.sum(axis=1)
            preds = (vote_sum > threshold).astype(np.int64)
            tie_mask = vote_sum == threshold
            if np.any(tie_mask):
                tie_scores = self.aggregate_scores(layer_scores[tie_mask])
                preds[tie_mask] = (tie_scores >= 0.0).astype(np.int64)
            return preds
        scores = self.aggregate_scores(layer_scores)
        return (scores >= 0.0).astype(np.int64)


def _train_ensemble(
    features_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    layers: list[int],
    mode: str,
    scoring_layer: int | None,
    random_state: int,
    max_iter: int,
) -> _ProbeEnsemble:
    if mode not in _ENSEMBLE_MODES:
        raise ValueError(
            f"Unknown ensemble mode '{mode}'. "
            f"Expected one of {sorted(_ENSEMBLE_MODES)}."
        )

    selected_scoring_layer = layers[0] if scoring_layer is None else int(scoring_layer)
    if selected_scoring_layer not in layers:
        raise ValueError(
            f"scoring_layer={selected_scoring_layer} is not in active layers {layers}."
        )

    y = labels[train_indices]
    models: dict[int, Pipeline | _ConstantProbe] = {}
    for layer in layers:
        X = features_by_layer[layer][train_indices]
        models[layer] = _train_probe(
            X_train=X,
            y_train=y,
            random_state=random_state + int(layer),
            max_iter=max_iter,
        )

    return _ProbeEnsemble(
        models=models,
        layers=layers,
        mode=mode,
        scoring_layer=selected_scoring_layer,
    )


def _top_k(indices: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or indices.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if k >= indices.shape[0]:
        order = np.argsort(-scores, kind="mergesort")
        return indices[order]
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="mergesort")]
    return indices[order]


def _uncertainty_scores(aggregate_scores: np.ndarray) -> np.ndarray:
    return -np.abs(aggregate_scores)


def _uncertainty_diversity_order(
    candidate_indices: np.ndarray,
    uncertainty_scores: np.ndarray,
    diversity_features: np.ndarray,
    uncertainty_weight: float,
    k: int | None = None,
) -> np.ndarray:
    if candidate_indices.shape[0] == 0:
        return np.array([], dtype=np.int64)

    if k is None:
        k = candidate_indices.shape[0]
    k = min(k, candidate_indices.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64)

    feats = _normalize_rows(diversity_features[candidate_indices].astype(np.float32))

    unc = uncertainty_scores.astype(np.float32)
    unc_min = float(np.min(unc))
    unc_range = float(np.max(unc) - unc_min)
    unc_norm = (unc - unc_min) / (unc_range + 1e-12)

    selected_local: list[int] = []
    remaining = np.arange(candidate_indices.shape[0], dtype=np.int64)

    first_local = int(np.argmax(unc_norm))
    selected_local.append(first_local)
    remaining = remaining[remaining != first_local]

    if remaining.shape[0] == 0 or k == 1:
        return candidate_indices[np.array(selected_local, dtype=np.int64)]

    first_sim = feats[remaining] @ feats[first_local]
    min_dist = 1.0 - first_sim

    while len(selected_local) < k and remaining.shape[0] > 0:
        combined = (
            uncertainty_weight * unc_norm[remaining]
            + (1.0 - uncertainty_weight) * min_dist
        )
        best_pos = int(np.argmax(combined))
        chosen_local = int(remaining[best_pos])
        selected_local.append(chosen_local)

        keep_mask = np.ones(remaining.shape[0], dtype=bool)
        keep_mask[best_pos] = False
        remaining = remaining[keep_mask]
        if remaining.shape[0] == 0:
            break

        new_dist = 1.0 - (feats[remaining] @ feats[chosen_local])
        min_dist = np.minimum(min_dist[keep_mask], new_dist)

    return candidate_indices[np.array(selected_local, dtype=np.int64)]


def _rank_pool(
    rule: str,
    ensemble: _ProbeEnsemble,
    features_by_layer: dict[int, np.ndarray],
    pool_indices: np.ndarray,
    diversity_features: np.ndarray,
    uncertainty_weight: float,
    prefilter_factor: int,
    k: int | None,
) -> np.ndarray:
    if pool_indices.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if k is not None and k <= 0:
        return np.array([], dtype=np.int64)

    layer_scores = ensemble.layer_scores(features_by_layer, pool_indices)
    agg_scores = ensemble.aggregate_scores(layer_scores)

    if rule == "uncertainty":
        scores = _uncertainty_scores(agg_scores)
        if k is None:
            return _top_k(pool_indices, scores, pool_indices.shape[0])
        return _top_k(pool_indices, scores, k)

    if rule == "disagreement":
        if len(ensemble.layers) < 2:
            scores = _uncertainty_scores(agg_scores)
        else:
            layer_probs = ensemble.layer_probs(features_by_layer, pool_indices)
            scores = np.var(layer_probs, axis=1).astype(np.float32)
        if k is None:
            return _top_k(pool_indices, scores, pool_indices.shape[0])
        return _top_k(pool_indices, scores, k)

    if rule == "uncertainty_diversity":
        uncertainty = _uncertainty_scores(agg_scores)
        if k is None:
            candidates = pool_indices
            candidate_uncertainty = uncertainty
            return _uncertainty_diversity_order(
                candidate_indices=candidates,
                uncertainty_scores=candidate_uncertainty,
                diversity_features=diversity_features,
                uncertainty_weight=uncertainty_weight,
                k=candidates.shape[0],
            )

        pre_n = min(
            pool_indices.shape[0],
            max(k, int(prefilter_factor) * k),
        )
        preselected = _top_k(pool_indices, uncertainty, pre_n)
        uncertainty_lookup = dict(
            zip(pool_indices.tolist(), uncertainty.tolist(), strict=False)
        )
        candidate_uncertainty = np.array(
            [float(uncertainty_lookup[int(i)]) for i in preselected],
            dtype=np.float32,
        )
        return _uncertainty_diversity_order(
            candidate_indices=preselected,
            uncertainty_scores=candidate_uncertainty,
            diversity_features=diversity_features,
            uncertainty_weight=uncertainty_weight,
            k=k,
        )

    raise ValueError(f"Unknown ranking rule '{rule}'.")


def _compute_dynamic_pool_scores(
    rule: str,
    ensemble: _ProbeEnsemble,
    features_by_layer: dict[int, np.ndarray],
    pool_indices: np.ndarray,
) -> np.ndarray:
    if pool_indices.shape[0] == 0:
        return np.array([], dtype=np.float32)

    layer_scores = ensemble.layer_scores(features_by_layer, pool_indices)
    agg_scores = ensemble.aggregate_scores(layer_scores)

    if rule == "uncertainty":
        return _uncertainty_scores(agg_scores).astype(np.float32)

    if rule == "disagreement":
        if len(ensemble.layers) < 2:
            return _uncertainty_scores(agg_scores).astype(np.float32)
        layer_probs = ensemble.layer_probs(features_by_layer, pool_indices)
        return np.var(layer_probs, axis=1).astype(np.float32)

    if rule == "uncertainty_diversity":
        # Stability drift is measured on the base uncertainty signal.
        return _uncertainty_scores(agg_scores).astype(np.float32)

    raise ValueError(f"Unknown dynamic rule '{rule}'.")


def _compute_dynamic_rank_order(
    rule: str,
    ensemble: _ProbeEnsemble,
    features_by_layer: dict[int, np.ndarray],
    pool_indices: np.ndarray,
    diversity_features: np.ndarray,
    uncertainty_weight: float,
    prefilter_factor: int,
) -> np.ndarray:
    return _rank_pool(
        rule=rule,
        ensemble=ensemble,
        features_by_layer=features_by_layer,
        pool_indices=pool_indices,
        diversity_features=diversity_features,
        uncertainty_weight=uncertainty_weight,
        prefilter_factor=prefilter_factor,
        k=None,
    ).astype(np.int64)


def _spearman_rank_corr(
    prev_order: list[int],
    curr_order: list[int],
) -> tuple[float | None, int]:
    shared = set(curr_order).intersection(prev_order)
    if len(shared) < 2:
        return None, len(shared)

    prev_filtered = [idx for idx in prev_order if idx in shared]
    curr_filtered = [idx for idx in curr_order if idx in shared]
    n = len(curr_filtered)
    if n < 2:
        return None, n

    prev_rank = {idx: i for i, idx in enumerate(prev_filtered)}
    curr_rank = {idx: i for i, idx in enumerate(curr_filtered)}
    d2 = 0.0
    for idx in curr_filtered:
        d = prev_rank[idx] - curr_rank[idx]
        d2 += float(d * d)
    rho = 1.0 - (6.0 * d2) / float(n * (n * n - 1))
    return float(rho), n


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _count_prompt_groups(prompt_groups: np.ndarray | None) -> dict[str, int] | None:
    if prompt_groups is None:
        return None
    if prompt_groups.shape[0] == 0:
        return {}
    unique, counts = np.unique(prompt_groups.astype(object), return_counts=True)
    return {
        str(group): int(count)
        for group, count in zip(unique.tolist(), counts.tolist(), strict=False)
    }


def _summarize_ranking_stability(
    rounds: list[RoundResult],
) -> dict[str, float | int | None] | None:
    overlap = [
        float(row.ranking_stability["top_k_overlap_at_k"])
        for row in rounds
        if row.ranking_stability is not None
        and row.ranking_stability.get("top_k_overlap_at_k") is not None
    ]
    rho = [
        float(row.ranking_stability["spearman_rho_shared_pool"])
        for row in rounds
        if row.ranking_stability is not None
        and row.ranking_stability.get("spearman_rho_shared_pool") is not None
    ]
    mae = [
        float(row.ranking_stability["score_drift_mae_shared_pool"])
        for row in rounds
        if row.ranking_stability is not None
        and row.ranking_stability.get("score_drift_mae_shared_pool") is not None
    ]
    turnover = [
        float(row.ranking_stability["selected_set_turnover"])
        for row in rounds
        if row.ranking_stability is not None
        and row.ranking_stability.get("selected_set_turnover") is not None
    ]

    if not overlap and not rho and not mae and not turnover:
        return None

    return {
        "mean_top_k_overlap_at_k": _mean_or_none(overlap),
        "mean_spearman_rho_shared_pool": _mean_or_none(rho),
        "mean_score_drift_mae_shared_pool": _mean_or_none(mae),
        "mean_selected_set_turnover": _mean_or_none(turnover),
        "support_top_k_overlap_at_k": int(len(overlap)),
        "support_spearman_rho_shared_pool": int(len(rho)),
        "support_score_drift_mae_shared_pool": int(len(mae)),
        "support_selected_set_turnover": int(len(turnover)),
    }


def _evaluate_on_test(
    ensemble: _ProbeEnsemble,
    features_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    test_indices: np.ndarray,
    prompt_groups: np.ndarray | None,
) -> tuple[float, float | None, float, dict[str, dict[str, float | int | None]] | None]:
    layer_scores = ensemble.layer_scores(features_by_layer, test_indices)
    y_true = labels[test_indices]
    y_pred = ensemble.predict(layer_scores)
    scores = ensemble.aggregate_scores(layer_scores)

    accuracy = float(accuracy_score(y_true, y_pred))
    if np.unique(y_true).size > 1:
        auroc: float | None = float(roc_auc_score(y_true, scores))
    else:
        auroc = None
    positive_rate = float(np.mean(y_true))

    subgroup_metrics: dict[str, dict[str, float | int | None]] | None = None
    if prompt_groups is not None:
        subgroup_metrics = {}
        group_values = prompt_groups[test_indices]
        for group in sorted(set(str(x) for x in group_values)):
            mask = group_values == group
            if int(mask.sum()) == 0:
                continue
            y_group = y_true[mask]
            pred_group = y_pred[mask]
            score_group = scores[mask]
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

    return accuracy, auroc, positive_rate, subgroup_metrics


def _compute_layer_metrics_on_test(
    ensemble: _ProbeEnsemble,
    features_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    y_true = labels[test_indices]
    auroc_by_layer: dict[str, float | None] = {}
    probs_by_layer: dict[int, np.ndarray] = {}
    preds_by_layer: dict[int, np.ndarray] = {}

    for layer in ensemble.layers:
        X = features_by_layer[layer][test_indices]
        model = ensemble.models[layer]
        probs = model.predict_proba(X)[:, 1].astype(np.float32)
        probs_by_layer[layer] = probs
        preds_by_layer[layer] = (probs >= 0.5).astype(np.int64)
        if np.unique(y_true).size > 1:
            auroc_by_layer[str(layer)] = float(roc_auc_score(y_true, probs))
        else:
            auroc_by_layer[str(layer)] = None

    pair_metrics: list[dict[str, float | int]] = []
    for i, layer_a in enumerate(ensemble.layers):
        for layer_b in ensemble.layers[i + 1 :]:
            pred_disagreement_rate = float(
                np.mean(preds_by_layer[layer_a] != preds_by_layer[layer_b])
            )
            mean_abs_prob_diff = float(
                np.mean(np.abs(probs_by_layer[layer_a] - probs_by_layer[layer_b]))
            )
            pair_metrics.append(
                {
                    "layer_a": int(layer_a),
                    "layer_b": int(layer_b),
                    "prediction_disagreement_rate": pred_disagreement_rate,
                    "mean_abs_probability_diff": mean_abs_prob_diff,
                    "support_test_examples": int(test_indices.shape[0]),
                }
            )

    return {
        "test_auroc_by_layer": auroc_by_layer,
        "disagreement_by_layer_pair": pair_metrics,
    }


def _ensure_two_class_labeled_set(
    labels: np.ndarray,
    labeled_indices: np.ndarray,
    pool_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_labeled = labels[labeled_indices]
    classes = set(np.unique(y_labeled).tolist())
    if classes == {0, 1}:
        return labeled_indices, pool_indices

    needed = [cls for cls in (0, 1) if cls not in classes]
    if not needed:
        return labeled_indices, pool_indices

    additions: list[int] = []
    keep_pool_mask = np.ones(pool_indices.shape[0], dtype=bool)
    for needed_cls in needed:
        pool_pos = np.where(labels[pool_indices] == needed_cls)[0]
        if pool_pos.size == 0:
            continue
        chosen_pos = int(pool_pos[0])
        additions.append(int(pool_indices[chosen_pos]))
        keep_pool_mask[chosen_pos] = False

    if additions:
        labeled_indices = np.concatenate(
            [labeled_indices, np.array(additions, dtype=np.int64)]
        )
        pool_indices = pool_indices[keep_pool_mask]
    return labeled_indices, pool_indices


def _indices_for_subset(
    prompt_groups: np.ndarray | None,
    subset: str,
    n_total: int,
) -> np.ndarray:
    if subset not in _PROMPT_SUBSETS:
        raise ValueError(
            f"Unknown prompt subset '{subset}'. Expected one of {sorted(_PROMPT_SUBSETS)}."
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


def _validate_split(split: ActiveLearningSplit, n: int) -> None:
    initial = np.asarray(split.initial_labeled_indices, dtype=np.int64)
    pool = np.asarray(split.unlabeled_pool_indices, dtype=np.int64)
    test = np.asarray(split.test_indices, dtype=np.int64)

    for name, values in (
        ("initial_labeled_indices", initial),
        ("unlabeled_pool_indices", pool),
        ("test_indices", test),
    ):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1D array.")
        if values.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if np.any(values < 0) or np.any(values >= n):
            raise ValueError(f"{name} has out-of-range indices for n={n}.")

    if np.intersect1d(initial, pool).size > 0:
        raise ValueError(
            "Split invalid: initial_labeled_indices intersects unlabeled_pool_indices."
        )
    if np.intersect1d(initial, test).size > 0:
        raise ValueError(
            "Split invalid: initial_labeled_indices intersects test_indices."
        )
    if np.intersect1d(pool, test).size > 0:
        raise ValueError(
            "Split invalid: unlabeled_pool_indices intersects test_indices."
        )


def make_active_learning_split(
    bundle: ActivationFeatureBundle,
    initial_labeled_size: int,
    test_size: float,
    random_state: int,
    train_prompt_subset: str = "all",
    test_prompt_subset: str = "all",
) -> ActiveLearningSplit:
    labels = bundle.labels.astype(np.int64)
    n = int(labels.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 examples to run train/pool/test splitting.")

    train_candidates = _indices_for_subset(
        prompt_groups=bundle.prompt_groups,
        subset=train_prompt_subset,
        n_total=n,
    )
    if train_candidates.size == 0:
        raise ValueError(
            f"No examples available for train_prompt_subset='{train_prompt_subset}'."
        )

    test_candidates = _indices_for_subset(
        prompt_groups=bundle.prompt_groups,
        subset=test_prompt_subset,
        n_total=n,
    )
    if test_candidates.size == 0:
        raise ValueError(
            f"No examples available for test_prompt_subset='{test_prompt_subset}'."
        )

    desired_test_count = int(round(float(test_size) * float(test_candidates.shape[0])))
    desired_test_count = max(1, desired_test_count)
    desired_test_count = min(desired_test_count, int(test_candidates.shape[0]))

    if desired_test_count >= int(test_candidates.shape[0]):
        test_indices = np.asarray(test_candidates, dtype=np.int64)
    else:
        _, test_indices = _safe_train_test_split(
            indices=test_candidates,
            labels=labels[test_candidates],
            test_size=desired_test_count,
            random_state=random_state,
        )

    train_pool_indices = np.setdiff1d(
        train_candidates,
        test_indices,
        assume_unique=False,
    ).astype(np.int64)
    if train_pool_indices.size == 0:
        raise ValueError(
            "No train/pool examples remain after removing held-out test indices. "
            f"train_prompt_subset='{train_prompt_subset}' test_prompt_subset='{test_prompt_subset}'."
        )

    train_pool_indices, test_indices = (
        np.asarray(train_pool_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )

    if initial_labeled_size >= train_pool_indices.shape[0]:
        raise ValueError(
            "initial_labeled_size must be smaller than train+pool split size. "
            f"Got initial_labeled_size={initial_labeled_size}, "
            f"train_pool_size={train_pool_indices.shape[0]}."
        )

    train_pool_labels = labels[train_pool_indices]
    initial_labeled_idx, unlabeled_pool_idx = _safe_train_test_split(
        indices=train_pool_indices,
        labels=train_pool_labels,
        test_size=1.0
        - float(initial_labeled_size) / float(train_pool_indices.shape[0]),
        random_state=random_state + 1,
    )
    initial_labeled_idx, unlabeled_pool_idx = _ensure_two_class_labeled_set(
        labels=labels,
        labeled_indices=initial_labeled_idx,
        pool_indices=unlabeled_pool_idx,
    )
    split = ActiveLearningSplit(
        initial_labeled_indices=np.asarray(initial_labeled_idx, dtype=np.int64),
        unlabeled_pool_indices=np.asarray(unlabeled_pool_idx, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
    )
    _validate_split(split=split, n=n)
    return split


def run_active_learning_experiment(
    bundle: ActivationFeatureBundle,
    config: ActiveLearningConfig,
) -> ActiveLearningResult:
    _validate_config(config)
    split = make_active_learning_split(
        bundle=bundle,
        initial_labeled_size=config.initial_labeled_size,
        test_size=config.test_size,
        random_state=config.random_state,
        train_prompt_subset=config.train_prompt_subset,
        test_prompt_subset=config.test_prompt_subset,
    )
    return run_active_learning_experiment_with_split(
        bundle=bundle,
        config=config,
        split=split,
    )


def run_active_learning_experiment_with_split(
    bundle: ActivationFeatureBundle,
    config: ActiveLearningConfig,
    split: ActiveLearningSplit,
) -> ActiveLearningResult:
    _validate_config(config)

    features_by_layer = bundle.features_by_layer
    labels = bundle.labels.astype(np.int64)
    layers = sorted(bundle.layers)
    n = int(labels.shape[0])
    if n < 3:
        raise ValueError("Need at least 3 examples to run train/pool/test splitting.")
    _validate_split(split=split, n=n)

    acquisition_mode, acquisition_rule = _parse_acquisition(config.acquisition)
    initial_labeled_idx = np.asarray(split.initial_labeled_indices, dtype=np.int64)
    unlabeled_pool_idx = np.asarray(split.unlabeled_pool_indices, dtype=np.int64)
    test_indices = np.asarray(split.test_indices, dtype=np.int64)

    query_budget = min(int(config.query_budget), int(unlabeled_pool_idx.shape[0]))
    diversity_features = _make_diversity_features(features_by_layer)

    initial_class_balance = {
        "label_0": int(np.sum(labels[initial_labeled_idx] == 0)),
        "label_1": int(np.sum(labels[initial_labeled_idx] == 1)),
    }

    rounds: list[RoundResult] = []
    queried = 0
    round_idx = 0
    labeled_idx = np.asarray(initial_labeled_idx, dtype=np.int64)
    pool_idx = np.asarray(unlabeled_pool_idx, dtype=np.int64)

    static_ranking: np.ndarray | None = None
    static_rank_ptr = 0
    in_pool = np.zeros(n, dtype=bool)
    in_pool[pool_idx] = True
    prev_dynamic_rank_order: np.ndarray | None = None
    prev_dynamic_score_map: dict[int, float] | None = None
    prev_dynamic_selected: np.ndarray | None = None

    while True:
        ensemble = _train_ensemble(
            features_by_layer=features_by_layer,
            labels=labels,
            train_indices=labeled_idx,
            layers=layers,
            mode=config.ensemble_mode,
            scoring_layer=config.scoring_layer,
            random_state=config.random_state,
            max_iter=config.max_probe_iter,
        )

        (
            test_accuracy,
            test_auroc,
            test_positive_rate,
            subgroup_metrics,
        ) = _evaluate_on_test(
            ensemble=ensemble,
            features_by_layer=features_by_layer,
            labels=labels,
            test_indices=test_indices,
            prompt_groups=bundle.prompt_groups,
        )

        ranking_stability: dict[str, float | int | None] | None = None
        dynamic_rank_order: np.ndarray | None = None
        dynamic_score_map: dict[int, float] | None = None

        if acquisition_mode == "dynamic":
            if acquisition_rule is None:
                raise RuntimeError("Dynamic acquisition requires a ranking rule.")
            dynamic_rank_order = _compute_dynamic_rank_order(
                rule=acquisition_rule,
                ensemble=ensemble,
                features_by_layer=features_by_layer,
                pool_indices=pool_idx,
                diversity_features=diversity_features,
                uncertainty_weight=config.uncertainty_weight,
                prefilter_factor=config.diversity_prefilter_factor,
            )
            dynamic_scores = _compute_dynamic_pool_scores(
                rule=acquisition_rule,
                ensemble=ensemble,
                features_by_layer=features_by_layer,
                pool_indices=pool_idx,
            )
            dynamic_score_map = {
                int(idx): float(score)
                for idx, score in zip(
                    pool_idx.tolist(), dynamic_scores.tolist(), strict=False
                )
            }

            top_k_overlap: float | None = None
            spearman_rho: float | None = None
            score_drift_mae: float | None = None
            k_top: int | None = None
            shared_pool_size: int | None = None

            if prev_dynamic_rank_order is not None:
                shared_set = set(pool_idx.tolist())
                prev_filtered = [
                    int(idx)
                    for idx in prev_dynamic_rank_order.tolist()
                    if int(idx) in shared_set
                ]
                curr_filtered = dynamic_rank_order.tolist()
                shared_pool_size = int(len(curr_filtered))

                if curr_filtered and prev_filtered:
                    k_top = int(
                        min(config.batch_size, len(curr_filtered), len(prev_filtered))
                    )
                    if k_top > 0:
                        prev_top = set(prev_filtered[:k_top])
                        curr_top = set(curr_filtered[:k_top])
                        top_k_overlap = float(
                            len(prev_top.intersection(curr_top)) / k_top
                        )

                spearman_rho, _shared_n = _spearman_rank_corr(
                    prev_order=prev_filtered,
                    curr_order=curr_filtered,
                )

                if prev_dynamic_score_map is not None and dynamic_score_map is not None:
                    shared_scored = [
                        idx
                        for idx in curr_filtered
                        if idx in prev_dynamic_score_map and idx in dynamic_score_map
                    ]
                    if shared_scored:
                        diffs = [
                            abs(dynamic_score_map[idx] - prev_dynamic_score_map[idx])
                            for idx in shared_scored
                        ]
                        score_drift_mae = float(
                            np.mean(np.asarray(diffs, dtype=np.float64))
                        )

            ranking_stability = {
                "top_k_overlap_at_k": top_k_overlap,
                "spearman_rho_shared_pool": spearman_rho,
                "score_drift_mae_shared_pool": score_drift_mae,
                "selected_set_turnover": None,
                "k_for_top_k_overlap": (None if k_top is None else int(k_top)),
                "shared_pool_size": shared_pool_size,
            }

        if queried >= query_budget or pool_idx.shape[0] == 0:
            rounds.append(
                RoundResult(
                    round_idx=round_idx,
                    queried_labels=queried,
                    labeled_size=int(labeled_idx.shape[0]),
                    unlabeled_pool_size=int(pool_idx.shape[0]),
                    test_size=int(test_indices.shape[0]),
                    test_accuracy=test_accuracy,
                    test_auroc=test_auroc,
                    test_positive_rate=test_positive_rate,
                    subgroup_metrics=subgroup_metrics,
                    ranking_stability=ranking_stability,
                )
            )
            break

        k = min(
            int(config.batch_size),
            int(query_budget - queried),
            int(pool_idx.shape[0]),
        )
        if k <= 0:
            rounds.append(
                RoundResult(
                    round_idx=round_idx,
                    queried_labels=queried,
                    labeled_size=int(labeled_idx.shape[0]),
                    unlabeled_pool_size=int(pool_idx.shape[0]),
                    test_size=int(test_indices.shape[0]),
                    test_accuracy=test_accuracy,
                    test_auroc=test_auroc,
                    test_positive_rate=test_positive_rate,
                    subgroup_metrics=subgroup_metrics,
                    ranking_stability=ranking_stability,
                )
            )
            break

        if acquisition_mode == "random":
            rng = np.random.default_rng(config.random_state + round_idx + 17)
            selected = np.asarray(
                rng.choice(pool_idx, size=k, replace=False),
                dtype=np.int64,
            )
        elif acquisition_mode == "static":
            if acquisition_rule is None:
                raise RuntimeError("Static acquisition requires a ranking rule.")
            if static_ranking is None:
                static_ranking = _rank_pool(
                    rule=acquisition_rule,
                    ensemble=ensemble,
                    features_by_layer=features_by_layer,
                    pool_indices=pool_idx,
                    diversity_features=diversity_features,
                    uncertainty_weight=config.uncertainty_weight,
                    prefilter_factor=config.diversity_prefilter_factor,
                    k=None,
                )

            chosen: list[int] = []
            while static_rank_ptr < static_ranking.shape[0] and len(chosen) < k:
                idx = int(static_ranking[static_rank_ptr])
                static_rank_ptr += 1
                if in_pool[idx]:
                    chosen.append(idx)
            if len(chosen) < k:
                remaining = pool_idx[
                    np.isin(pool_idx, np.array(chosen, dtype=np.int64), invert=True)
                ]
                fallback = remaining[: k - len(chosen)]
                chosen.extend(int(x) for x in fallback)
            selected = np.array(chosen, dtype=np.int64)
        else:
            if acquisition_rule is None:
                raise RuntimeError("Dynamic acquisition requires a ranking rule.")
            selected = _rank_pool(
                rule=acquisition_rule,
                ensemble=ensemble,
                features_by_layer=features_by_layer,
                pool_indices=pool_idx,
                diversity_features=diversity_features,
                uncertainty_weight=config.uncertainty_weight,
                prefilter_factor=config.diversity_prefilter_factor,
                k=k,
            ).astype(np.int64)

        if selected.shape[0] == 0:
            rounds.append(
                RoundResult(
                    round_idx=round_idx,
                    queried_labels=queried,
                    labeled_size=int(labeled_idx.shape[0]),
                    unlabeled_pool_size=int(pool_idx.shape[0]),
                    test_size=int(test_indices.shape[0]),
                    test_accuracy=test_accuracy,
                    test_auroc=test_auroc,
                    test_positive_rate=test_positive_rate,
                    subgroup_metrics=subgroup_metrics,
                    ranking_stability=ranking_stability,
                )
            )
            break

        if ranking_stability is not None:
            selected_set = set(int(x) for x in selected.tolist())
            if selected_set:
                if prev_dynamic_selected is None:
                    ranking_stability["selected_set_turnover"] = None
                else:
                    prev_selected_set = set(
                        int(x) for x in prev_dynamic_selected.tolist()
                    )
                    retained = len(selected_set.intersection(prev_selected_set))
                    ranking_stability["selected_set_turnover"] = float(
                        1.0 - (retained / len(selected_set))
                    )

        rounds.append(
            RoundResult(
                round_idx=round_idx,
                queried_labels=queried,
                labeled_size=int(labeled_idx.shape[0]),
                unlabeled_pool_size=int(pool_idx.shape[0]),
                test_size=int(test_indices.shape[0]),
                test_accuracy=test_accuracy,
                test_auroc=test_auroc,
                test_positive_rate=test_positive_rate,
                subgroup_metrics=subgroup_metrics,
                ranking_stability=ranking_stability,
            )
        )

        labeled_idx = np.concatenate([labeled_idx, selected])
        in_pool[selected] = False
        pool_idx = pool_idx[np.isin(pool_idx, selected, invert=True)]
        queried += int(selected.shape[0])
        if acquisition_mode == "dynamic":
            prev_dynamic_rank_order = dynamic_rank_order
            prev_dynamic_score_map = dynamic_score_map
            prev_dynamic_selected = selected
        round_idx += 1

    final_class_balance = {
        "label_0": int(np.sum(labels[labeled_idx] == 0)),
        "label_1": int(np.sum(labels[labeled_idx] == 1)),
    }
    ranking_stability_summary = _summarize_ranking_stability(rounds)
    final_ensemble = _train_ensemble(
        features_by_layer=features_by_layer,
        labels=labels,
        train_indices=labeled_idx,
        layers=layers,
        mode=config.ensemble_mode,
        scoring_layer=config.scoring_layer,
        random_state=config.random_state,
        max_iter=config.max_probe_iter,
    )
    layer_metrics = _compute_layer_metrics_on_test(
        ensemble=final_ensemble,
        features_by_layer=features_by_layer,
        labels=labels,
        test_indices=test_indices,
    )

    return ActiveLearningResult(
        config=asdict(config),
        label_field=bundle.label_field,
        prompt_subset=bundle.prompt_subset,
        prompt_group_counts=_count_prompt_groups(bundle.prompt_groups),
        layers=layers,
        acquisition_mode=acquisition_mode,
        acquisition_rule=acquisition_rule,
        n_examples=n,
        initial_class_balance=initial_class_balance,
        final_class_balance=final_class_balance,
        ranking_stability_summary=ranking_stability_summary,
        layer_metrics=layer_metrics,
        rounds=rounds,
    )
