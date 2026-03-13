from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from latent_dynamics.activations import extract_multi_layer_trajectories
from latent_dynamics.config import MODEL_REGISTRY, RunConfig
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.experiments.churn_dynamics_experiment import (
    ExperimentConfig as ChurnConfig,
)
from latent_dynamics.experiments.churn_dynamics_experiment import (
    run_experiment as run_churn,
)
from latent_dynamics.experiments.kalman_conformal_experiment import (
    ExperimentConfig as KalmanConfig,
)
from latent_dynamics.experiments.kalman_conformal_experiment import (
    run_experiment as run_kalman,
)
from latent_dynamics.experiments.trajectory_features import (
    signature_prefix_score_map,
    turning_angle_score_map,
)
from latent_dynamics.hub import activation_subpath, load_activations, save_activations
from latent_dynamics.judge import JudgeCache, SafetyJudge, judge_texts
from latent_dynamics.models import load_model_and_tokenizer, resolve_device


@dataclass
class BenchmarkConfig:
    activations_root: Path | None = None
    model_key: str = "gemma3_4b"
    dataset_key: str = "toy_contrastive"
    split: str = "train"
    layers: list[int] | None = None
    max_samples: int = 160
    max_input_tokens: int = 256
    use_generate: bool = False
    max_new_tokens: int = 24
    include_prompt_in_trajectory: bool = True
    device: str | None = None
    seeds: list[int] | None = None
    train_fraction: float = 0.6
    calib_fraction: float = 0.2
    alpha: float = 0.1
    max_prefix: int = 32
    fpr_target: float = 0.05
    kalman_state_dim: int = 64
    churn_dictionary_dim: int = 1024
    churn_top_k: int = 32
    linear_koopman_dim: int = 64
    signature_pca_dim: int = 16
    signature_depth: int = 2
    benign_manifold_rank: int = 8
    # LLM-as-judge options. When judge_model is set, completions are generated
    # for every prompt and labeled by the judge. These judge labels replace
    # dataset labels and become the ground truth for all classifiers.
    judge_model: str | None = None
    judge_max_new_tokens: int = 128
    judge_requests_per_minute: int = 120
    judge_max_concurrency: int = 12
    judge_batch_size: int = 32
    judge_cache_path: Path | None = None
    qd_report_json: Path | None = None
    output_json: Path | None = None


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Layer/seed benchmark harness with shared splits across baselines and "
            "dynamics methods for early unsafe prefix detection."
        ),
    )
    parser.add_argument("--activations-root", type=Path, default=None)
    parser.add_argument(
        "--model-key", choices=sorted(MODEL_REGISTRY.keys()), default="gemma3_4b"
    )
    parser.add_argument("--dataset-key", default="toy_contrastive")
    parser.add_argument("--split", default="train")
    parser.add_argument("--layers", type=int, nargs="+", default=[5])
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--use-generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--no-include-prompt", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-prefix", type=int, default=32)
    parser.add_argument("--fpr-target", type=float, default=0.05)
    parser.add_argument("--kalman-state-dim", type=int, default=64)
    parser.add_argument("--churn-dictionary-dim", type=int, default=1024)
    parser.add_argument("--churn-top-k", type=int, default=32)
    parser.add_argument("--linear-koopman-dim", type=int, default=64)
    parser.add_argument("--signature-pca-dim", type=int, default=16)
    parser.add_argument("--signature-depth", type=int, default=2)
    parser.add_argument("--benign-manifold-rank", type=int, default=8)
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "OpenAI-compatible model for LLM-as-judge labeling (e.g. gpt-4o-mini). "
            "When set, completions are generated for every prompt and judged; "
            "judge labels replace dataset labels as ground truth."
        ),
    )
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--judge-rpm", type=int, default=120)
    parser.add_argument("--judge-max-concurrency", type=int, default=12)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument("--judge-cache-path", type=Path, default=None)
    parser.add_argument("--qd-report-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    return BenchmarkConfig(
        activations_root=args.activations_root,
        model_key=args.model_key,
        dataset_key=args.dataset_key,
        split=args.split,
        layers=list(args.layers),
        max_samples=args.max_samples,
        max_input_tokens=args.max_input_tokens,
        use_generate=args.use_generate,
        max_new_tokens=args.max_new_tokens,
        include_prompt_in_trajectory=(not args.no_include_prompt),
        device=args.device,
        seeds=list(args.seeds),
        train_fraction=args.train_fraction,
        calib_fraction=args.calib_fraction,
        alpha=args.alpha,
        max_prefix=args.max_prefix,
        fpr_target=args.fpr_target,
        kalman_state_dim=args.kalman_state_dim,
        churn_dictionary_dim=args.churn_dictionary_dim,
        churn_top_k=args.churn_top_k,
        linear_koopman_dim=args.linear_koopman_dim,
        signature_pca_dim=args.signature_pca_dim,
        signature_depth=args.signature_depth,
        benign_manifold_rank=args.benign_manifold_rank,
        judge_model=args.judge_model,
        judge_max_new_tokens=args.judge_max_new_tokens,
        judge_requests_per_minute=args.judge_rpm,
        judge_max_concurrency=args.judge_max_concurrency,
        judge_batch_size=args.judge_batch_size,
        judge_cache_path=args.judge_cache_path,
        qd_report_json=args.qd_report_json,
        output_json=args.output_json,
    )


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


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    vals = np.sort(scores.astype(np.float64))
    n = len(vals)
    if n < 1:
        raise ValueError("No scores for conformal quantile.")
    rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(n - 1, rank))
    return float(vals[rank])


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (2 * tp) + fp + fn
    if denom == 0:
        return None
    return float((2 * tp) / denom)


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def _calibrate_prefix_thresholds(
    score_map: dict[int, np.ndarray],
    labels: np.ndarray,
    calib_idx: np.ndarray,
    alpha: float,
    max_prefix: int,
) -> tuple[dict[int, float], int]:
    safe_idxs = [int(i) for i in calib_idx if labels[int(i)] == 0]
    if not safe_idxs:
        raise ValueError("Calibration split has no safe samples.")

    max_available = max((len(score_map[i]) for i in safe_idxs), default=0)
    max_p = min(max_prefix, int(max_available))
    if max_p < 1:
        raise ValueError("No usable safe calibration prefixes.")

    thresholds: dict[int, float] = {}
    for p in range(1, max_p + 1):
        vals = [score_map[i][p - 1] for i in safe_idxs if len(score_map[i]) >= p]
        if vals:
            thresholds[p] = _conformal_quantile(np.array(vals, dtype=np.float32), alpha)
    if not thresholds:
        raise ValueError("No conformal thresholds computed.")
    return thresholds, max(thresholds.keys())


def _evaluate_detection(
    score_map: dict[int, np.ndarray],
    labels: np.ndarray,
    idxs: np.ndarray,
    thresholds: dict[int, float],
    max_prefix: int,
) -> dict[str, Any]:
    ids = [int(i) for i in idxs]
    usable = [i for i in ids if len(score_map[i]) >= 1]
    if not usable:
        raise ValueError("No usable trajectories in eval split.")

    first_hit: dict[int, int | None] = {}
    final_scores: list[float] = []
    final_labels: list[int] = []
    for i in usable:
        s = score_map[i]
        hit = None
        for p in range(1, min(max_prefix, len(s)) + 1):
            tau = thresholds.get(p)
            if tau is None:
                continue
            if s[p - 1] > tau:
                hit = p
                break
        first_hit[i] = hit
        final_scores.append(float(s[min(max_prefix, len(s)) - 1]))
        final_labels.append(int(labels[i]))

    per_prefix: list[dict[str, float]] = []
    best_f1: float | None = None
    best_f1_prefix: int | None = None
    f1_at_max_prefix: float | None = None
    for p in range(1, max_prefix + 1):
        tau = thresholds.get(p)
        if tau is None:
            continue
        valid = [i for i in usable if len(score_map[i]) >= p]
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
                "threshold": float(tau),
                "n_eval": float(len(valid)),
                "unsafe_tpr": tpr,
                "safe_fpr": fpr,
                "f1": f1 if f1 is not None else float("nan"),
            }
        )

    y_all = np.array(final_labels, dtype=np.int64)
    s_all = np.array(final_scores, dtype=np.float32)
    auroc = _safe_auroc(y_all, s_all)

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
        "final_prefix_auroc": auroc,
        "f1_at_max_prefix": f1_at_max_prefix,
        "best_f1": best_f1,
        "best_f1_prefix": float(best_f1_prefix) if best_f1_prefix is not None else None,
        "per_prefix": per_prefix,
    }


def _summary_from_prefix_stats(
    prefix_stats: dict[str, Any], fpr_target: float
) -> dict[str, Any]:
    per_prefix = prefix_stats.get("per_prefix", [])
    tpr_at_target: float | None = None
    prefix_at_target: float | None = None
    for row in per_prefix:
        fpr = row.get("safe_fpr")
        tpr = row.get("unsafe_tpr")
        if fpr is None or tpr is None:
            continue
        if np.isnan(fpr) or np.isnan(tpr):
            continue
        if fpr <= fpr_target:
            if tpr_at_target is None or tpr > tpr_at_target:
                tpr_at_target = float(tpr)
                prefix_at_target = float(row.get("prefix", np.nan))
    return {
        "best_f1": prefix_stats.get("best_f1"),
        "best_f1_prefix": prefix_stats.get("best_f1_prefix"),
        "f1_at_max_prefix": prefix_stats.get("f1_at_max_prefix"),
        "detected_unsafe_fraction": prefix_stats.get("detected_unsafe_fraction"),
        "false_positive_safe_fraction": prefix_stats.get(
            "false_positive_safe_fraction"
        ),
        "median_first_detect_prefix_unsafe": prefix_stats.get(
            "median_first_detect_prefix_unsafe"
        ),
        "final_prefix_auroc": prefix_stats.get("final_prefix_auroc"),
        "tpr_at_fpr_target": tpr_at_target,
        "prefix_at_fpr_target": prefix_at_target,
        "fpr_target": fpr_target,
    }


def _load_external_qd_summary(
    path: Path | None, fpr_target: float
) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"--qd-report-json not found: {path}")

    payload = json.loads(path.read_text())
    strategies = payload.get("evaluation", {}).get("strategies", {})
    qd = strategies.get("qd_uncertainty")
    if not isinstance(qd, dict):
        return None

    return {
        "best_f1": None,
        "best_f1_prefix": None,
        "f1_at_max_prefix": None,
        "detected_unsafe_fraction": None,
        "false_positive_safe_fraction": None,
        "median_first_detect_prefix_unsafe": None,
        "final_prefix_auroc": qd.get("heldout_auroc"),
        "tpr_at_fpr_target": None,
        "prefix_at_fpr_target": None,
        "fpr_target": fpr_target,
        "external_source": str(path),
        "heldout_brier": qd.get("heldout_brier"),
        "heldout_ece": qd.get("heldout_ece"),
        "uncertainty_mass_near_boundary": qd.get("uncertainty_mass_near_boundary"),
    }


def _prefix_feature(traj: np.ndarray, p: int, mode: str) -> np.ndarray:
    if mode == "mean":
        return traj[:p].mean(axis=0)
    if mode == "last":
        return traj[p - 1]
    raise ValueError(f"Unknown mode: {mode}")


def _lr_prefix_score_map(
    trajectories: list[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    max_prefix: int,
    mode: str,
    seed: int,
) -> dict[int, np.ndarray]:
    score_map: dict[int, np.ndarray] = {}
    max_len = min(max_prefix, max(int(t.shape[0]) for t in trajectories))
    models: dict[int, Pipeline] = {}
    for p in range(1, max_len + 1):
        fit_ids = [int(i) for i in train_idx if trajectories[int(i)].shape[0] >= p]
        if len(fit_ids) < 4:
            break
        y = np.array([labels[i] for i in fit_ids], dtype=np.int64)
        if len(np.unique(y)) < 2:
            break
        X = np.stack(
            [_prefix_feature(trajectories[i], p, mode) for i in fit_ids], axis=0
        )
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
        models[p] = clf

    for i, traj in enumerate(trajectories):
        vals: list[float] = []
        for p in range(1, max_prefix + 1):
            clf = models.get(p)
            if clf is None or traj.shape[0] < p:
                break
            feat = _prefix_feature(traj, p, mode).reshape(1, -1)
            vals.append(float(clf.predict_proba(feat)[0, 1]))
        score_map[i] = np.array(vals, dtype=np.float32)
    return score_map


def _delta_norm_score_map(
    trajectories: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for i, traj in enumerate(trajectories):
        if traj.shape[0] < 2:
            out[i] = np.array([], dtype=np.float32)
            continue
        x = (traj.astype(np.float32) - mean) / std
        d = np.diff(x, axis=0)
        step = np.linalg.norm(d, axis=1) / np.sqrt(float(d.shape[1]))
        out[i] = (
            np.cumsum(step) / np.arange(1, len(step) + 1, dtype=np.float32)
        ).astype(np.float32)
    return out


def _linear_koopman_score_map(
    trajectories: list[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    dim: int,
    ridge: float,
    safe_only: bool,
    seed: int,
) -> dict[int, np.ndarray]:
    fit_idx = [int(i) for i in train_idx if (labels[int(i)] == 0 or not safe_only)]
    if len(fit_idx) < 2:
        raise ValueError(
            "Need at least 2 fit trajectories for linear Koopman baseline."
        )

    train_states = np.concatenate([trajectories[i] for i in fit_idx], axis=0).astype(
        np.float32
    )
    max_rank = int(min(train_states.shape[0], train_states.shape[1]))
    eff_dim = int(min(dim, max_rank))
    pca = PCA(n_components=eff_dim, random_state=seed)
    pca.fit(train_states)

    z_traj = [pca.transform(t.astype(np.float32)) for t in trajectories]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in fit_idx:
        z = z_traj[i]
        if z.shape[0] < 2:
            continue
        xs.append(z[:-1])
        ys.append(z[1:])
    if not xs:
        raise ValueError("No transitions available for linear Koopman fit.")
    x_prev = np.concatenate(xs, axis=0)
    x_next = np.concatenate(ys, axis=0)
    eye = np.eye(eff_dim, dtype=np.float32)
    w = np.linalg.solve((x_prev.T @ x_prev) + (ridge * eye), x_prev.T @ x_next)

    score_map: dict[int, np.ndarray] = {}
    for i, z in enumerate(z_traj):
        if z.shape[0] < 2:
            score_map[i] = np.array([], dtype=np.float32)
            continue
        pred = z[:-1] @ w
        resid = np.linalg.norm(z[1:] - pred, axis=1) / np.sqrt(float(eff_dim))
        score_map[i] = (
            np.cumsum(resid) / np.arange(1, len(resid) + 1, dtype=np.float32)
        ).astype(np.float32)
    return score_map


def _judge_completions(
    prompts: list[str],
    completions: list[str],
    cfg: BenchmarkConfig,
) -> np.ndarray:
    """Judge already-generated (prompt, completion) pairs and return int64 labels.

    Separated from generation so the caller can reuse completions that were
    already produced during trajectory extraction.
    """
    cache: JudgeCache | None = None
    if cfg.judge_cache_path is not None:
        cache = JudgeCache(cfg.judge_cache_path)

    assert cfg.judge_model is not None
    judge = SafetyJudge(
        model=cfg.judge_model,
        requests_per_minute=cfg.judge_requests_per_minute,
        max_concurrency=cfg.judge_max_concurrency,
        batch_size=cfg.judge_batch_size,
        show_progress=True,
    )

    pairs = list(zip(prompts, completions, strict=False))
    results = judge_texts(pairs, judge, cache)
    return np.array([1 if r.unsafe else 0 for r in results], dtype=np.int64)


def _ensure_layer_paths(cfg: BenchmarkConfig) -> dict[int, Path]:
    layers = cfg.layers if cfg.layers is not None else [5]
    if cfg.activations_root is not None:
        paths = {
            li: cfg.activations_root
            / activation_subpath(cfg.dataset_key, cfg.model_key, li)
            for li in layers
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing activation directories under --activations-root:\n"
                + "\n".join(missing)
            )
        return paths

    out_root = Path(".cache/benchmark_activations")
    run_cfg = RunConfig(
        model_key=cfg.model_key,
        dataset_key=cfg.dataset_key,
        max_samples=cfg.max_samples,
        max_input_tokens=cfg.max_input_tokens,
        layer_idx=layers[0],
        device=resolve_device(cfg.device),
        use_generate=cfg.use_generate,
        max_new_tokens=cfg.max_new_tokens,
        include_prompt_in_trajectory=cfg.include_prompt_in_trajectory,
    )
    ds, spec = load_examples(
        run_cfg.dataset_key,
        max_samples=run_cfg.max_samples,
        stratify_labels=True,
    )
    texts, labels = prepare_text_and_labels(
        ds,
        text_field=spec.text_field,
        label_field=spec.label_field,
        label_fn=spec.label_fn,
    )
    if labels is None:
        raise ValueError("Dataset must provide labels for this benchmark.")

    # Generation-trajectory mode (--no-include-prompt): judge labeling is
    # required, and use_generate must be True so trajectories come from the
    # generated tokens. We force it here so the user doesn't need both flags.
    use_judge = not cfg.include_prompt_in_trajectory
    if use_judge:
        if cfg.judge_model is None:
            raise ValueError(
                "--no-include-prompt (generation-trajectory mode) requires "
                "--judge-model to be set so completions can be labeled. "
                "Example: --judge-model gpt-4o-mini"
            )
        run_cfg = RunConfig(
            **{
                **run_cfg.__dict__,
                "use_generate": True,
                "max_new_tokens": cfg.judge_max_new_tokens,
            }
        )
    elif cfg.judge_model is not None:
        print(
            "Warning: --judge-model is set but --no-include-prompt is not. "
            "Judge labeling is only used in generation-trajectory mode; "
            "dataset labels will be used instead."
        )

    model, tokenizer = load_model_and_tokenizer(
        run_cfg.model_key, run_cfg.device or "cpu"
    )

    result = extract_multi_layer_trajectories(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_indices=layers,
        cfg=run_cfg,
    )
    per_layer = result.per_layer
    token_texts = result.token_texts

    if use_judge:
        # result.generated_texts holds decoded completions (prompt excluded when
        # include_prompt_in_trajectory=False). Judge against the original prompt.
        completions = [g or "" for g in result.generated_texts]
        labels = _judge_completions(texts, completions, cfg)
        print(
            f"Judge labels: {int(labels.sum())} unsafe / "
            f"{int((labels == 0).sum())} safe"
        )

    paths: dict[int, Path] = {}
    for li in layers:
        p = out_root / activation_subpath(cfg.dataset_key, cfg.model_key, li)
        cfg_li = RunConfig(**{**run_cfg.__dict__, "layer_idx": li})
        save_activations(
            p,
            per_layer[li],
            result.input_prompts,
            labels,
            token_texts,
            cfg_li,
            generated_texts=result.generated_texts,
        )
        paths[li] = p
    return paths


def _run_baseline_family(
    trajectories: list[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    heldout_idx: np.ndarray,
    alpha: float,
    max_prefix: int,
    seed: int,
    linear_koopman_dim: int,
    signature_pca_dim: int = 16,
    signature_depth: int = 2,
    benign_manifold_rank: int = 8,
) -> dict[str, dict[str, Any]]:
    train_states = np.concatenate(
        [trajectories[int(i)] for i in train_idx], axis=0
    ).astype(np.float32)
    mean = train_states.mean(axis=0).astype(np.float32)
    std = (train_states.std(axis=0) + 1e-6).astype(np.float32)

    methods: dict[str, dict[int, np.ndarray]] = {
        "baseline_prefix_mean_lr": _lr_prefix_score_map(
            trajectories,
            labels,
            train_idx,
            max_prefix,
            mode="mean",
            seed=seed,
        ),
        "baseline_prefix_last_lr": _lr_prefix_score_map(
            trajectories,
            labels,
            train_idx,
            max_prefix,
            mode="last",
            seed=seed,
        ),
        "baseline_delta_norm": _delta_norm_score_map(trajectories, mean=mean, std=std),
        "linear_koopman_conformal": _linear_koopman_score_map(
            trajectories=trajectories,
            labels=labels,
            train_idx=train_idx,
            dim=linear_koopman_dim,
            ridge=1e-4,
            safe_only=True,
            seed=seed,
        ),
        "turning_angle_conformal": turning_angle_score_map(
            trajectories=trajectories,
            labels=labels,
            train_idx=train_idx,
            pca_n_components=signature_pca_dim,
            manifold_rank=benign_manifold_rank,
            max_prefix=max_prefix,
            seed=seed,
        ),
        "signature_prefix_lr": signature_prefix_score_map(
            trajectories=trajectories,
            labels=labels,
            train_idx=train_idx,
            pca_n_components=signature_pca_dim,
            depth=signature_depth,
            add_time=True,
            max_prefix=max_prefix,
            seed=seed,
        ),
    }

    out: dict[str, dict[str, Any]] = {}
    for name, score_map in methods.items():
        thresholds, max_p = _calibrate_prefix_thresholds(
            score_map=score_map,
            labels=labels,
            calib_idx=calib_idx,
            alpha=alpha,
            max_prefix=max_prefix,
        )
        held = _evaluate_detection(
            score_map=score_map,
            labels=labels,
            idxs=heldout_idx,
            thresholds=thresholds,
            max_prefix=max_p,
        )
        calib = _evaluate_detection(
            score_map=score_map,
            labels=labels,
            idxs=calib_idx,
            thresholds=thresholds,
            max_prefix=max_p,
        )
        out[name] = {
            "conformal": {"max_calibrated_prefix": max_p},
            "prefix_stats_calib": calib,
            "prefix_stats_heldout": held,
        }
    return out


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    layers = cfg.layers if cfg.layers is not None else [5]
    seeds = cfg.seeds if cfg.seeds is not None else [7, 11, 19]
    layer_paths = _ensure_layer_paths(cfg)
    external_qd_summary = _load_external_qd_summary(cfg.qd_report_json, cfg.fpr_target)

    runs: list[dict[str, Any]] = []
    for li in layers:
        trajectories, _texts, labels, _tokens, _generated, source_cfg = (
            load_activations(layer_paths[li])
        )
        if labels is None or len(np.unique(labels)) < 2:
            if labels is None:
                raise ValueError(
                    f"Layer {li} has missing labels; cannot benchmark prefix detectors."
                )
            uniq, cnt = np.unique(labels, return_counts=True)
            counts_str = ", ".join(
                f"{int(u)}:{int(c)}" for u, c in zip(uniq.tolist(), cnt.tolist())
            )
            hint = ""
            if source_cfg.dataset_key == "wildjailbreak":
                hint = (
                    " wildjailbreak train is typically one-class; try `--split eval`."
                )
            raise ValueError(
                f"Layer {li} has single-class labels ({counts_str}); cannot benchmark "
                f"prefix detectors.{hint}"
            )

        for seed in seeds:
            train_idx, calib_idx, heldout_idx = _split_indices(
                labels=labels,
                train_fraction=cfg.train_fraction,
                calib_fraction=cfg.calib_fraction,
                seed=seed,
            )

            baseline = _run_baseline_family(
                trajectories=trajectories,
                labels=labels,
                train_idx=train_idx,
                calib_idx=calib_idx,
                heldout_idx=heldout_idx,
                alpha=cfg.alpha,
                max_prefix=cfg.max_prefix,
                seed=seed,
                linear_koopman_dim=cfg.linear_koopman_dim,
                signature_pca_dim=cfg.signature_pca_dim,
                signature_depth=cfg.signature_depth,
                benign_manifold_rank=cfg.benign_manifold_rank,
            )

            k_res = run_kalman(
                KalmanConfig(
                    activations=layer_paths[li],
                    state_dim=cfg.kalman_state_dim,
                    train_fraction=cfg.train_fraction,
                    calib_fraction=cfg.calib_fraction,
                    train_on_safe_only=True,
                    alpha=cfg.alpha,
                    max_prefix=cfg.max_prefix,
                    random_state=seed,
                )
            )
            c_res = run_churn(
                ChurnConfig(
                    activations=layer_paths[li],
                    train_fraction=cfg.train_fraction,
                    calib_fraction=cfg.calib_fraction,
                    dictionary_dim=cfg.churn_dictionary_dim,
                    top_k=cfg.churn_top_k,
                    train_dynamics_on_safe_only=True,
                    alpha=cfg.alpha,
                    max_prefix=cfg.max_prefix,
                    random_state=seed,
                )
            )

            methods: dict[str, Any] = {}
            methods.update(baseline)
            methods["kalman_conformal"] = {
                "conformal": k_res["conformal"],
                "prefix_stats_calib": k_res["prefix_stats_calib"],
                "prefix_stats_heldout": k_res["prefix_stats_heldout"],
            }
            methods["churn_conformal"] = {
                "conformal": c_res["conformal"],
                "prefix_stats_calib": c_res["prefix_stats_calib"],
                "prefix_stats_heldout": c_res["prefix_stats_heldout"],
            }
            if external_qd_summary is not None:
                methods["qd_active_boundary_external"] = {
                    "summary": external_qd_summary,
                }

            summary: dict[str, Any] = {}
            for name, m in methods.items():
                if "prefix_stats_heldout" in m:
                    summary[name] = _summary_from_prefix_stats(
                        m["prefix_stats_heldout"], cfg.fpr_target
                    )
                elif "summary" in m:
                    summary[name] = m["summary"]
            runs.append(
                {
                    "layer": li,
                    "seed": seed,
                    "source_config": asdict(source_cfg),
                    "split_sizes": {
                        "n_train": int(len(train_idx)),
                        "n_calib": int(len(calib_idx)),
                        "n_heldout": int(len(heldout_idx)),
                    },
                    "methods": methods,
                    "summary": summary,
                }
            )

    method_names = sorted({m for r in runs for m in r["summary"].keys()})
    aggregate: dict[str, dict[str, float | None]] = {}
    for name in method_names:
        vals_f1 = [
            r["summary"][name]["best_f1"]
            for r in runs
            if r["summary"][name]["best_f1"] is not None
        ]
        vals_tpr = [
            r["summary"][name]["tpr_at_fpr_target"]
            for r in runs
            if r["summary"][name]["tpr_at_fpr_target"] is not None
        ]
        vals_auroc = [
            r["summary"][name]["final_prefix_auroc"]
            for r in runs
            if r["summary"][name]["final_prefix_auroc"] is not None
        ]
        aggregate[name] = {
            "mean_best_f1": float(np.mean(vals_f1)) if vals_f1 else None,
            "mean_tpr_at_fpr_target": float(np.mean(vals_tpr)) if vals_tpr else None,
            "mean_final_prefix_auroc": float(np.mean(vals_auroc))
            if vals_auroc
            else None,
        }

    return {
        "benchmark_config": asdict(cfg),
        "n_runs": len(runs),
        "runs": runs,
        "aggregate_summary": aggregate,
    }


def main() -> None:
    cfg = _parse_args()
    report = run_benchmark(cfg)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_json.write_text(text)
        print(f"Wrote results: {cfg.output_json}")


if __name__ == "__main__":
    main()
