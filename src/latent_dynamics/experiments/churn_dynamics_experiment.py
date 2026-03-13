from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from latent_dynamics.activations import extract_hidden_trajectories
from latent_dynamics.config import MODEL_REGISTRY, RunConfig
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.hub import load_activations
from latent_dynamics.models import load_model_and_tokenizer, resolve_device


@dataclass
class ExperimentConfig:
    activations: Path | None = None
    model_key: str = "gemma3_4b"
    dataset_key: str = "toy_contrastive"
    split: str = "train"
    layer_idx: int = 5
    max_samples: int = 120
    max_input_tokens: int = 256
    use_generate: bool = False
    max_new_tokens: int = 24
    include_prompt_in_trajectory: bool = True
    device: str | None = None
    train_fraction: float = 0.6
    calib_fraction: float = 0.2
    dictionary_dim: int = 1024
    top_k: int = 32
    ar_ridge: float = 1e-4
    train_dynamics_on_safe_only: bool = True
    alpha: float = 0.1
    max_prefix: int = 32
    random_state: int = 7
    output_json: Path | None = None


@dataclass
class ChurnDynamicsModel:
    dictionary: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    top_k: int
    ar_bias: float
    ar_weight: float


def _parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Active-set churn experiment: random dictionary sparse support churn "
            "+ AR(1) dynamics + conformal prefix unsafe detection."
        ),
    )
    parser.add_argument("--activations", type=Path, default=None)
    parser.add_argument(
        "--model-key", choices=sorted(MODEL_REGISTRY.keys()), default="gemma3_4b"
    )
    parser.add_argument("--dataset-key", default="toy_contrastive")
    parser.add_argument("--split", default="train")
    parser.add_argument("--layer-idx", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--use-generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--no-include-prompt", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--dictionary-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--ar-ridge", type=float, default=1e-4)
    parser.add_argument("--train-dynamics-on-all-labels", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-prefix", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    return ExperimentConfig(
        activations=args.activations,
        model_key=args.model_key,
        dataset_key=args.dataset_key,
        split=args.split,
        layer_idx=args.layer_idx,
        max_samples=args.max_samples,
        max_input_tokens=args.max_input_tokens,
        use_generate=args.use_generate,
        max_new_tokens=args.max_new_tokens,
        include_prompt_in_trajectory=(not args.no_include_prompt),
        device=args.device,
        train_fraction=args.train_fraction,
        calib_fraction=args.calib_fraction,
        dictionary_dim=args.dictionary_dim,
        top_k=args.top_k,
        ar_ridge=args.ar_ridge,
        train_dynamics_on_safe_only=(not args.train_dynamics_on_all_labels),
        alpha=args.alpha,
        max_prefix=args.max_prefix,
        random_state=args.random_state,
        output_json=args.output_json,
    )


def _prepare_trajectories(
    cfg: ExperimentConfig,
) -> tuple[list[np.ndarray], np.ndarray, RunConfig]:
    if cfg.activations is not None:
        trajectories, _texts, labels, _tokens, _gen, run_cfg = load_activations(
            cfg.activations
        )
        if labels is None:
            raise ValueError(
                "Activations must include labels for safe/unsafe prediction."
            )
        return trajectories, labels, run_cfg

    run_cfg = RunConfig(
        model_key=cfg.model_key,
        dataset_key=cfg.dataset_key,
        max_samples=cfg.max_samples,
        max_input_tokens=cfg.max_input_tokens,
        layer_idx=cfg.layer_idx,
        device=resolve_device(cfg.device),
        use_generate=cfg.use_generate,
        max_new_tokens=cfg.max_new_tokens,
        include_prompt_in_trajectory=cfg.include_prompt_in_trajectory,
    )
    ds, spec = load_examples(run_cfg.dataset_key, run_cfg.max_samples)
    texts, labels = prepare_text_and_labels(
        ds,
        text_field=spec.text_field,
        label_field=spec.label_field,
        label_fn=spec.label_fn,
    )
    if labels is None:
        raise ValueError("Dataset must provide labels for safe/unsafe prediction.")

    model, tokenizer = load_model_and_tokenizer(
        run_cfg.model_key, run_cfg.device or "cpu"
    )
    result = extract_hidden_trajectories(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=run_cfg.layer_idx,
        cfg=run_cfg,
    )
    trajectories = result.per_layer[run_cfg.layer_idx]
    return trajectories, labels, run_cfg


def _split_indices(
    labels: np.ndarray,
    train_fraction: float,
    calib_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1).")
    if not (0.0 < calib_fraction < 1.0):
        raise ValueError("calib_fraction must be in (0, 1).")
    if train_fraction + calib_fraction >= 1.0:
        raise ValueError("train_fraction + calib_fraction must be < 1.0.")

    n = len(labels)
    idx = np.arange(n)
    temp_fraction = 1.0 - train_fraction
    calib_relative = calib_fraction / temp_fraction

    try:
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=temp_fraction,
            random_state=seed,
            stratify=labels,
        )
        calib_idx, heldout_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - calib_relative),
            random_state=seed,
            stratify=labels[temp_idx],
        )
    except ValueError:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        n_train = int(round(train_fraction * n))
        n_calib = int(round(calib_fraction * n))
        n_train = min(max(1, n_train), n - 2)
        n_calib = min(max(1, n_calib), n - n_train - 1)
        train_idx = perm[:n_train]
        calib_idx = perm[n_train : n_train + n_calib]
        heldout_idx = perm[n_train + n_calib :]

    return np.sort(train_idx), np.sort(calib_idx), np.sort(heldout_idx)


def _collect_states(trajectories: list[np.ndarray], idxs: np.ndarray) -> np.ndarray:
    return np.concatenate([trajectories[int(i)] for i in idxs], axis=0).astype(
        np.float32
    )


def _build_random_dictionary(
    input_dim: int,
    dictionary_dim: int,
    seed: int,
) -> np.ndarray:
    if dictionary_dim < 1:
        raise ValueError("dictionary_dim must be >= 1.")
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, 1.0, size=(input_dim, dictionary_dim)).astype(np.float32)
    d /= np.linalg.norm(d, axis=0, keepdims=True) + 1e-8
    return d


def _topk_support(values: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("top_k must be >= 1.")
    k_eff = min(k, values.shape[0])
    idx = np.argpartition(np.abs(values), -k_eff)[-k_eff:]
    return np.sort(idx.astype(np.int32))


def _jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).shape[0]
    union = int(a.shape[0] + b.shape[0] - inter)
    if union == 0:
        return 0.0
    return 1.0 - (float(inter) / float(union))


def _compute_churn_sequence(
    trajectory: np.ndarray,
    dictionary: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    top_k: int,
) -> np.ndarray:
    if trajectory.shape[0] < 2:
        return np.array([], dtype=np.float32)

    normed = (trajectory.astype(np.float32) - mean) / std
    proj = normed @ dictionary
    supports = [_topk_support(proj[t], top_k) for t in range(proj.shape[0])]

    churn: list[float] = []
    for t in range(1, len(supports)):
        churn.append(_jaccard_distance(supports[t], supports[t - 1]))
    return np.array(churn, dtype=np.float32)


def _fit_ar1(churn_seqs: list[np.ndarray], ridge: float) -> tuple[float, float]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for seq in churn_seqs:
        if len(seq) < 2:
            continue
        xs.append(seq[:-1])
        ys.append(seq[1:])
    if not xs:
        raise ValueError("Need churn sequences of length >= 2 to fit AR(1).")

    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    x_aug = np.stack([np.ones_like(x), x], axis=1)
    xtx = x_aug.T @ x_aug
    xty = x_aug.T @ y
    reg = ridge * np.eye(2, dtype=np.float32)
    w = np.linalg.solve(xtx + reg, xty)
    return float(w[0]), float(w[1])


def _residual_prefix_score(
    churn_seq: np.ndarray,
    bias: float,
    weight: float,
) -> np.ndarray:
    if len(churn_seq) < 2:
        return np.array([], dtype=np.float32)

    pred = bias + (weight * churn_seq[:-1])
    residual = np.abs(churn_seq[1:] - pred).astype(np.float32)
    return np.cumsum(residual) / np.arange(1, len(residual) + 1, dtype=np.float32)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if len(scores) == 0:
        raise ValueError("No calibration scores for conformal quantile.")
    vals = np.sort(scores.astype(np.float64))
    n = len(vals)
    rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(n - 1, rank))
    return float(vals[rank])


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return None
    return float((2 * tp) / denom)


def _build_prefix_score_matrix(
    trajectories: list[np.ndarray],
    idxs: np.ndarray,
    model: ChurnDynamicsModel,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    churn_scores: dict[int, np.ndarray] = {}
    residual_scores: dict[int, np.ndarray] = {}
    for i in idxs:
        idx = int(i)
        churn = _compute_churn_sequence(
            trajectories[idx],
            dictionary=model.dictionary,
            mean=model.mean,
            std=model.std,
            top_k=model.top_k,
        )
        residual_prefix = _residual_prefix_score(
            churn,
            bias=model.ar_bias,
            weight=model.ar_weight,
        )
        churn_scores[idx] = churn
        residual_scores[idx] = residual_prefix
    return churn_scores, residual_scores


def _calibrate_thresholds(
    prefix_scores: dict[int, np.ndarray],
    labels: np.ndarray,
    calib_idxs: np.ndarray,
    alpha: float,
    max_prefix: int,
) -> tuple[dict[int, float], int]:
    safe_idxs = [int(i) for i in calib_idxs if labels[int(i)] == 0]
    if not safe_idxs:
        raise ValueError("Calibration split has no safe samples.")

    max_available = max((len(prefix_scores[i]) for i in safe_idxs), default=0)
    max_p = min(max_prefix, int(max_available))
    if max_p < 1:
        raise ValueError("No usable safe calibration prefix scores.")

    thresholds: dict[int, float] = {}
    for p in range(1, max_p + 1):
        vals = [
            prefix_scores[i][p - 1] for i in safe_idxs if len(prefix_scores[i]) >= p
        ]
        if vals:
            thresholds[p] = _conformal_quantile(np.array(vals, dtype=np.float32), alpha)
    if not thresholds:
        raise ValueError("No conformal thresholds computed.")
    return thresholds, max(thresholds.keys())


def _evaluate_prefix_detection(
    prefix_scores: dict[int, np.ndarray],
    labels: np.ndarray,
    idxs: np.ndarray,
    thresholds: dict[int, float],
    max_prefix: int,
) -> dict[str, Any]:
    idx_list = [int(i) for i in idxs]
    usable = [i for i in idx_list if len(prefix_scores[i]) >= 1]
    if not usable:
        raise ValueError("No usable trajectories in evaluation split.")

    first_hit: dict[int, int | None] = {}
    final_scores: list[float] = []
    final_labels: list[int] = []

    for i in usable:
        scores = prefix_scores[i]
        hit = None
        max_p = min(max_prefix, len(scores))
        for p in range(1, max_p + 1):
            tau = thresholds.get(p)
            if tau is None:
                continue
            if scores[p - 1] > tau:
                hit = p
                break
        first_hit[i] = hit

        last_p = min(max_prefix, len(scores))
        final_scores.append(float(scores[last_p - 1]))
        final_labels.append(int(labels[i]))

    per_prefix: list[dict[str, float]] = []
    best_f1: float | None = None
    best_f1_prefix: int | None = None
    f1_at_max_prefix: float | None = None

    for p in range(1, max_prefix + 1):
        tau = thresholds.get(p)
        if tau is None:
            continue
        valid = [i for i in usable if len(prefix_scores[i]) >= p]
        if not valid:
            continue

        y = np.array([labels[i] for i in valid], dtype=np.int64)
        pred = np.array(
            [
                1 if (first_hit[i] is not None and first_hit[i] <= p) else 0
                for i in valid
            ],
            dtype=np.int64,
        )
        safe_mask = y == 0
        unsafe_mask = y == 1
        fpr = float(pred[safe_mask].mean()) if np.any(safe_mask) else float("nan")
        tpr = float(pred[unsafe_mask].mean()) if np.any(unsafe_mask) else float("nan")
        f1 = _binary_f1(y, pred)
        if f1 is not None and (best_f1 is None or f1 > best_f1):
            best_f1 = f1
            best_f1_prefix = p
        if p == max_prefix:
            f1_at_max_prefix = f1

        per_prefix.append(
            {
                "prefix": float(p),
                "n_eval": float(len(valid)),
                "threshold": float(tau),
                "unsafe_tpr": tpr,
                "safe_fpr": fpr,
                "f1": f1 if f1 is not None else float("nan"),
            }
        )

    y_all = np.array(final_labels, dtype=np.int64)
    score_all = np.array(final_scores, dtype=np.float32)
    auroc_final = _safe_roc_auc(y_all, score_all)

    unsafe_idxs = [i for i in usable if labels[i] == 1]
    safe_idxs = [i for i in usable if labels[i] == 0]
    unsafe_hits = [first_hit[i] for i in unsafe_idxs if first_hit[i] is not None]
    safe_hits = [first_hit[i] for i in safe_idxs if first_hit[i] is not None]

    return {
        "n_eval": len(usable),
        "n_unsafe": len(unsafe_idxs),
        "n_safe": len(safe_idxs),
        "detected_unsafe_fraction": (
            float(len(unsafe_hits) / len(unsafe_idxs)) if unsafe_idxs else None
        ),
        "false_positive_safe_fraction": (
            float(len(safe_hits) / len(safe_idxs)) if safe_idxs else None
        ),
        "median_first_detect_prefix_unsafe": (
            float(np.median(np.array(unsafe_hits, dtype=np.float32)))
            if unsafe_hits
            else None
        ),
        "mean_first_detect_prefix_unsafe": (
            float(np.mean(np.array(unsafe_hits, dtype=np.float32)))
            if unsafe_hits
            else None
        ),
        "final_prefix_auroc": auroc_final,
        "f1_at_max_prefix": f1_at_max_prefix,
        "best_f1": best_f1,
        "best_f1_prefix": (
            float(best_f1_prefix) if best_f1_prefix is not None else None
        ),
        "per_prefix": per_prefix,
    }


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    trajectories, labels, source_cfg = _prepare_trajectories(cfg)
    if len(trajectories) != len(labels):
        raise ValueError("Mismatch between number of trajectories and labels.")
    if len(np.unique(labels)) < 2:
        msg = (
            "Loaded labels contain only one class. This experiment needs both safe "
            "(0) and unsafe (1) labels."
        )
        if cfg.dataset_key == "wildjailbreak" and cfg.split == "train":
            msg += " Try `--split eval` for wildjailbreak."
        raise ValueError(msg)

    train_idx, calib_idx, heldout_idx = _split_indices(
        labels=labels,
        train_fraction=cfg.train_fraction,
        calib_fraction=cfg.calib_fraction,
        seed=cfg.random_state,
    )
    if len(heldout_idx) == 0:
        raise ValueError("Heldout split is empty. Adjust split fractions.")

    if cfg.top_k > cfg.dictionary_dim:
        raise ValueError(
            f"top_k must be <= dictionary_dim. Got top_k={cfg.top_k}, dictionary_dim={cfg.dictionary_dim}."
        )

    train_states = _collect_states(trajectories, train_idx)
    input_dim = int(train_states.shape[1])
    mean = train_states.mean(axis=0).astype(np.float32)
    std = (train_states.std(axis=0) + 1e-6).astype(np.float32)
    dictionary = _build_random_dictionary(
        input_dim=input_dim,
        dictionary_dim=cfg.dictionary_dim,
        seed=cfg.random_state,
    )

    if cfg.train_dynamics_on_safe_only:
        fit_idx = np.array(
            [i for i in train_idx if labels[int(i)] == 0], dtype=np.int64
        )
        if len(fit_idx) < 2:
            raise ValueError(
                "Need at least 2 safe train trajectories for safe-only dynamics."
            )
    else:
        fit_idx = train_idx.astype(np.int64)

    fit_churn = [
        _compute_churn_sequence(
            trajectories[int(i)],
            dictionary=dictionary,
            mean=mean,
            std=std,
            top_k=cfg.top_k,
        )
        for i in fit_idx
    ]
    ar_bias, ar_weight = _fit_ar1(fit_churn, ridge=cfg.ar_ridge)
    model = ChurnDynamicsModel(
        dictionary=dictionary,
        mean=mean,
        std=std,
        top_k=cfg.top_k,
        ar_bias=ar_bias,
        ar_weight=ar_weight,
    )

    calib_churn, calib_prefix = _build_prefix_score_matrix(
        trajectories, calib_idx, model
    )
    heldout_churn, heldout_prefix = _build_prefix_score_matrix(
        trajectories, heldout_idx, model
    )

    thresholds, max_calibrated_prefix = _calibrate_thresholds(
        prefix_scores=calib_prefix,
        labels=labels,
        calib_idxs=calib_idx,
        alpha=cfg.alpha,
        max_prefix=cfg.max_prefix,
    )

    calib_eval = _evaluate_prefix_detection(
        prefix_scores=calib_prefix,
        labels=labels,
        idxs=calib_idx,
        thresholds=thresholds,
        max_prefix=max_calibrated_prefix,
    )
    heldout_eval = _evaluate_prefix_detection(
        prefix_scores=heldout_prefix,
        labels=labels,
        idxs=heldout_idx,
        thresholds=thresholds,
        max_prefix=max_calibrated_prefix,
    )

    def _mean_last_prefix(
        prefix_map: dict[int, np.ndarray], ids: np.ndarray, y: int
    ) -> float | None:
        vals = [
            float(prefix_map[int(i)][-1])
            for i in ids
            if labels[int(i)] == y and len(prefix_map[int(i)]) > 0
        ]
        if not vals:
            return None
        return float(np.mean(np.array(vals, dtype=np.float32)))

    results: dict[str, Any] = {
        "experiment": asdict(cfg),
        "source_config": asdict(source_cfg),
        "data": {
            "n_trajectories": int(len(trajectories)),
            "n_train": int(len(train_idx)),
            "n_calib": int(len(calib_idx)),
            "n_heldout": int(len(heldout_idx)),
            "train_label_counts": {
                "safe_0": int(np.sum(labels[train_idx] == 0)),
                "unsafe_1": int(np.sum(labels[train_idx] == 1)),
            },
            "calib_label_counts": {
                "safe_0": int(np.sum(labels[calib_idx] == 0)),
                "unsafe_1": int(np.sum(labels[calib_idx] == 1)),
            },
            "heldout_label_counts": {
                "safe_0": int(np.sum(labels[heldout_idx] == 0)),
                "unsafe_1": int(np.sum(labels[heldout_idx] == 1)),
            },
            "input_dim": input_dim,
        },
        "dictionary": {
            "dictionary_dim": int(cfg.dictionary_dim),
            "top_k": int(cfg.top_k),
        },
        "dynamics": {
            "fit_trajectories": int(len(fit_idx)),
            "train_dynamics_on_safe_only": bool(cfg.train_dynamics_on_safe_only),
            "ar_bias": float(ar_bias),
            "ar_weight": float(ar_weight),
        },
        "conformal": {
            "alpha": float(cfg.alpha),
            "max_calibrated_prefix": int(max_calibrated_prefix),
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
        },
        "churn_summary": {
            "calib_mean_last_prefix_safe": _mean_last_prefix(
                calib_prefix, calib_idx, y=0
            ),
            "calib_mean_last_prefix_unsafe": _mean_last_prefix(
                calib_prefix, calib_idx, y=1
            ),
            "heldout_mean_last_prefix_safe": _mean_last_prefix(
                heldout_prefix, heldout_idx, y=0
            ),
            "heldout_mean_last_prefix_unsafe": _mean_last_prefix(
                heldout_prefix, heldout_idx, y=1
            ),
            "calib_mean_raw_churn_safe": _mean_last_prefix(calib_churn, calib_idx, y=0),
            "calib_mean_raw_churn_unsafe": _mean_last_prefix(
                calib_churn, calib_idx, y=1
            ),
            "heldout_mean_raw_churn_safe": _mean_last_prefix(
                heldout_churn, heldout_idx, y=0
            ),
            "heldout_mean_raw_churn_unsafe": _mean_last_prefix(
                heldout_churn, heldout_idx, y=1
            ),
        },
        "prefix_stats_calib": calib_eval,
        "prefix_stats_heldout": heldout_eval,
    }
    return results


def main() -> None:
    cfg = _parse_args()
    results = run_experiment(cfg)
    report = json.dumps(results, indent=2, default=str)
    print(report)

    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_json.write_text(report)
        print(f"Wrote results: {cfg.output_json}")


if __name__ == "__main__":
    main()
