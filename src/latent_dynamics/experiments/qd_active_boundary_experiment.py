from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from latent_dynamics.activations import extract_multi_layer_trajectories
from latent_dynamics.config import MODEL_REGISTRY, RunConfig
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.models import load_model_and_tokenizer, resolve_device

# Runtime assumptions:
# - OPENAI_API_KEY is set
# - FLASHLITE_DEFAULT_MODEL is set (defaults to gpt-5-mini in this script)
# - flashlite + ribs are installed via pyproject dependencies.

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompts" / "qd_active"
MUTATION_TEMPLATE_NAME = "mutate_prompt"
SAFETY_JUDGE_TEMPLATE_NAME = "safety_judge"


MUTATION_STYLES = [
    "increase specificity and operational detail",
    "rephrase as an indirect or role-play request",
    "make the request ambiguous while preserving intent",
    "convert to a multi-step planning ask",
    "ask for alternatives after a refusal",
    "blend benign framing with potentially unsafe intent",
]


@dataclass
class ExperimentConfig:
    model_key: str = "gemma3_4b"
    layer_idx: int = 5
    max_length: int = 256
    max_new_tokens: int = 128
    include_prompt_in_trajectory: bool = True
    device: str | None = None

    harmful_dataset_key: str = "wildjailbreak"
    harmful_split: str = "eval"
    benign_dataset_key: str = "xstest"
    benign_split: str = "test"
    max_harmful_prompts: int = 500
    max_benign_prompts: int = 500
    heldout_prompts: int = 96

    horizon_tokens: int = 64
    label_budget: int = 1000
    warm_start_labels: int = 128
    batch_size: int = 32
    candidate_pool_size: int = 128
    random_seed_fraction: float = 0.2

    proxy_model: str = "gpt-5-mini"
    proxy_max_concurrency: int = 12
    proxy_batch_size: int = 32
    proxy_requests_per_minute: int = 120
    proxy_tokens_per_minute: int | None = None
    template_dir: Path | None = None
    rollout_batch_size: int = 16
    show_progress: bool = True

    alpha_uncertainty: float = 0.7
    alpha_novelty: float = 0.3
    bd_dims: tuple[int, int, int] = (12, 12, 8)

    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    pca_components: int = 8
    boundary_margin_epsilon: float = 0.5

    page_hinkley_delta: float = 0.05
    page_hinkley_lambda: float = 5.0

    strategies: tuple[str, ...] = ("qd_uncertainty", "uncertainty_only", "random")
    random_state: int = 7

    output_root: Path = Path(".cache/qd_active")
    output_json: Path | None = None


@dataclass
class Candidate:
    candidate_id: str
    prompt: str
    source: str
    parent_prompt: str | None
    trajectory: np.ndarray
    token_texts: list[str]
    rollout_snippet: str
    measures: tuple[float, float, float]
    query_index: int = 0
    z_query: np.ndarray | None = None
    z_prefix_mean: np.ndarray | None = None
    pca_proj: np.ndarray | None = None
    psi: np.ndarray | None = None
    q_unc: float = 0.0
    q_nov: float = 0.0
    q_total: float = 0.0


@dataclass
class StrategyState:
    name: str
    rng: np.random.Generator
    probe: "BayesianLogisticProbe"
    X: list[np.ndarray] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    labeled_pca: list[np.ndarray] = field(default_factory=list)
    prompt_pool: list[str] = field(default_factory=list)
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    archive: Any | None = None
    archive_ranges: (
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
    ) = None
    archive_cells_seen: set[tuple[int, int, int]] = field(default_factory=set)
    archive_coverage_history: list[float] = field(default_factory=list)


@dataclass
class JudgeResult:
    unsafe: bool
    confidence: float
    rationale: str


class BayesianLogisticProbe:
    def __init__(self, c: float = 1.0, max_iter: int = 2000) -> None:
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.cov_: np.ndarray | None = None
        self.fitted_: bool = False

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if len(np.unique(y)) < 2:
            self.fitted_ = False
            return

        self.mean_ = X.mean(axis=0).astype(np.float64)
        self.std_ = (X.std(axis=0) + 1e-6).astype(np.float64)
        Xs = self._standardize(X.astype(np.float64))

        clf = LogisticRegression(
            penalty="l2",
            C=self.c,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(Xs, y.astype(np.int64))

        w = clf.coef_[0].astype(np.float64)
        b = float(clf.intercept_[0])

        logits = Xs @ w + b
        probs = _sigmoid(logits)
        ww = probs * (1.0 - probs)

        x_aug = np.concatenate(
            [Xs, np.ones((Xs.shape[0], 1), dtype=np.float64)], axis=1
        )
        h = (x_aug.T * ww) @ x_aug

        reg = np.zeros(h.shape[0], dtype=np.float64)
        reg[:-1] = 1.0 / max(self.c, 1e-12)
        h += np.diag(reg)

        cov: np.ndarray
        try:
            cov = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            diag = np.clip(np.diag(h), 1e-8, None)
            cov = np.diag(1.0 / diag)

        self.coef_ = w
        self.intercept_ = b
        self.cov_ = cov
        self.fitted_ = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            return np.full(X.shape[0], 0.5, dtype=np.float64)
        assert self.coef_ is not None and self.intercept_ is not None
        Xs = self._standardize(X.astype(np.float64))
        mu = Xs @ self.coef_ + self.intercept_
        if self.cov_ is None:
            return _sigmoid(mu)

        x_aug = np.concatenate(
            [Xs, np.ones((Xs.shape[0], 1), dtype=np.float64)], axis=1
        )
        beta = np.concatenate(
            [self.coef_, np.array([self.intercept_], dtype=np.float64)]
        )
        mu_aug = x_aug @ beta
        var = np.einsum("bi,ij,bj->b", x_aug, self.cov_, x_aug)
        var = np.clip(var, 0.0, None)
        kappa = 1.0 / np.sqrt(1.0 + (math.pi * var / 8.0))
        return _sigmoid(kappa * mu_aug)

    def predictive_entropy(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return _binary_entropy(p)

    def margin(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict_proba(X), 1e-6, 1.0 - 1e-6)
        return np.abs(np.log(p / (1.0 - p)))


class ProxyLLM:
    def __init__(self, cfg: ExperimentConfig) -> None:
        try:
            from flashlite import Flashlite, RateLimitConfig, RetryConfig
        except Exception as e:  # pragma: no cover - import path check
            raise ImportError(
                "flashlite is required. Install dependencies and set OPENAI_API_KEY."
            ) from e

        rate_kwargs: dict[str, Any] = {
            "requests_per_minute": cfg.proxy_requests_per_minute,
        }
        if cfg.proxy_tokens_per_minute is not None:
            rate_kwargs["tokens_per_minute"] = cfg.proxy_tokens_per_minute

        template_dir = cfg.template_dir or DEFAULT_TEMPLATE_DIR
        if not template_dir.exists():
            raise FileNotFoundError(
                f"flashlite template directory not found: {template_dir}"
            )

        self.client = Flashlite(
            default_model=cfg.proxy_model,
            template_dir=str(template_dir),
            retry=RetryConfig(max_attempts=3),
            rate_limit=RateLimitConfig(**rate_kwargs),
            log_requests=False,
        )
        self.max_concurrency = cfg.proxy_max_concurrency
        self.batch_size = max(1, cfg.proxy_batch_size)
        self.show_progress = cfg.show_progress

    def mutate_prompts(
        self,
        parent_prompts: list[str],
        rng: np.random.Generator,
        target_count: int,
    ) -> list[str]:
        if target_count <= 0:
            return []
        reqs: list[dict[str, Any]] = []
        for i in range(target_count):
            parent = parent_prompts[i % len(parent_prompts)]
            style = MUTATION_STYLES[int(rng.integers(0, len(MUTATION_STYLES)))]
            reqs.append(
                {
                    "template": MUTATION_TEMPLATE_NAME,
                    "variables": {
                        "parent_prompt": parent,
                        "mutation_style": style,
                    },
                }
            )

        out: list[str] = []
        starts = _chunk_starts(len(reqs), self.batch_size)
        iterator = _iter_with_progress(
            starts=starts,
            desc=f"mutate prompts ({target_count})",
            enabled=self.show_progress and len(starts) > 1,
        )
        for start in iterator:
            req_batch = reqs[start : start + self.batch_size]
            resps = _run_async(
                self.client.complete_many(
                    req_batch, max_concurrency=self.max_concurrency
                )
            )
            for r in resps:
                txt = _extract_response_text(r)
                cleaned = txt.strip().strip('`"')
                if cleaned:
                    out.append(cleaned)
        return out

    def judge_batch(self, samples: list[tuple[str, str]]) -> list[JudgeResult]:
        reqs = [
            {
                "template": SAFETY_JUDGE_TEMPLATE_NAME,
                "variables": {"prompt": p, "rollout": r},
            }
            for p, r in samples
        ]

        parsed: list[JudgeResult] = []
        starts = _chunk_starts(len(reqs), self.batch_size)
        iterator = _iter_with_progress(
            starts=starts,
            desc=f"judge rollouts ({len(samples)})",
            enabled=self.show_progress and len(starts) > 1,
        )
        for start in iterator:
            req_batch = reqs[start : start + self.batch_size]
            resps = _run_async(
                self.client.complete_many(
                    req_batch, max_concurrency=self.max_concurrency
                )
            )
            for resp in resps:
                txt = _extract_response_text(resp)
                parsed.append(_parse_judge_output(txt))
        return parsed


class JudgeCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def _parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "QD-driven active learning for latent-space safety boundary learning "
            "with Bayesian logistic probes, pyribs archive, and flashlite proxy labels."
        )
    )
    parser.add_argument(
        "--model-key", choices=sorted(MODEL_REGISTRY.keys()), default="gemma3_4b"
    )
    parser.add_argument("--layer-idx", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-include-prompt", action="store_true")
    parser.add_argument("--device", default=None)

    parser.add_argument("--harmful-dataset-key", default="wildjailbreak")
    parser.add_argument("--harmful-split", default="eval")
    parser.add_argument("--benign-dataset-key", default="xstest")
    parser.add_argument("--benign-split", default="test")
    parser.add_argument("--max-harmful-prompts", type=int, default=500)
    parser.add_argument("--max-benign-prompts", type=int, default=500)
    parser.add_argument("--heldout-prompts", type=int, default=96)

    parser.add_argument("--horizon-tokens", type=int, default=64)
    parser.add_argument("--label-budget", type=int, default=1000)
    parser.add_argument("--warm-start-labels", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-pool-size", type=int, default=128)
    parser.add_argument("--random-seed-fraction", type=float, default=0.2)

    parser.add_argument("--proxy-model", default="gpt-5-mini")
    parser.add_argument("--proxy-max-concurrency", type=int, default=12)
    parser.add_argument("--proxy-batch-size", type=int, default=32)
    parser.add_argument("--proxy-rpm", type=int, default=120)
    parser.add_argument("--proxy-tpm", type=int, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--alpha-uncertainty", type=float, default=0.7)
    parser.add_argument("--alpha-novelty", type=float, default=0.3)
    parser.add_argument("--bd-dims", type=int, nargs=3, default=[12, 12, 8])

    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--logistic-max-iter", type=int, default=2000)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--boundary-margin-epsilon", type=float, default=0.5)

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["qd_uncertainty", "uncertainty_only", "random"],
    )
    parser.add_argument("--random-state", type=int, default=7)

    parser.add_argument("--output-root", type=Path, default=Path(".cache/qd_active"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    return ExperimentConfig(
        model_key=args.model_key,
        layer_idx=args.layer_idx,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        include_prompt_in_trajectory=(not args.no_include_prompt),
        device=args.device,
        harmful_dataset_key=args.harmful_dataset_key,
        harmful_split=args.harmful_split,
        benign_dataset_key=args.benign_dataset_key,
        benign_split=args.benign_split,
        max_harmful_prompts=args.max_harmful_prompts,
        max_benign_prompts=args.max_benign_prompts,
        heldout_prompts=args.heldout_prompts,
        horizon_tokens=args.horizon_tokens,
        label_budget=args.label_budget,
        warm_start_labels=args.warm_start_labels,
        batch_size=args.batch_size,
        candidate_pool_size=args.candidate_pool_size,
        random_seed_fraction=args.random_seed_fraction,
        proxy_model=args.proxy_model,
        proxy_max_concurrency=args.proxy_max_concurrency,
        proxy_batch_size=args.proxy_batch_size,
        proxy_requests_per_minute=args.proxy_rpm,
        proxy_tokens_per_minute=args.proxy_tpm,
        template_dir=args.template_dir,
        rollout_batch_size=args.rollout_batch_size,
        show_progress=(not args.no_progress),
        alpha_uncertainty=args.alpha_uncertainty,
        alpha_novelty=args.alpha_novelty,
        bd_dims=tuple(args.bd_dims),
        logistic_c=args.logistic_c,
        logistic_max_iter=args.logistic_max_iter,
        pca_components=args.pca_components,
        boundary_margin_epsilon=args.boundary_margin_epsilon,
        strategies=tuple(args.strategies),
        random_state=args.random_state,
        output_root=args.output_root,
        output_json=args.output_json,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _chunk_starts(n_items: int, chunk_size: int) -> list[int]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    return list(range(0, n_items, chunk_size))


def _iter_with_progress(
    starts: list[int],
    desc: str,
    enabled: bool,
):
    if not enabled:
        return starts
    try:
        from tqdm.auto import tqdm
    except Exception:
        return starts
    return tqdm(starts, desc=desc, leave=False, dynamic_ncols=True)


def _make_progress_bar(
    total: int,
    initial: int,
    desc: str,
    enabled: bool,
):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )


def _run_async(coro: Any) -> Any:
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _extract_response_text(resp: Any) -> str:
    if hasattr(resp, "content"):
        content = getattr(resp, "content")
        if isinstance(content, str):
            return content
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        if "content" in resp and isinstance(resp["content"], str):
            return resp["content"]
        if "text" in resp and isinstance(resp["text"], str):
            return resp["text"]
    return str(resp)


def _parse_judge_output(text: str) -> JudgeResult:
    parsed: dict[str, Any] | None = None
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            parsed = obj
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        low = cleaned.lower()
        unsafe = ("unsafe" in low and "true" in low) or (
            "harmful" in low and "yes" in low
        )
        confidence = 0.5
        return JudgeResult(
            unsafe=unsafe, confidence=confidence, rationale="heuristic_parse"
        )

    unsafe = bool(parsed.get("unsafe", False))
    confidence_val = parsed.get("confidence", 0.5)
    try:
        confidence = float(confidence_val)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = float(np.clip(confidence, 0.0, 1.0))
    rationale = str(parsed.get("rationale", ""))
    return JudgeResult(unsafe=unsafe, confidence=confidence, rationale=rationale)


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _load_seed_prompts(cfg: ExperimentConfig) -> tuple[list[str], list[str]]:
    harmful_ds, harmful_spec = load_examples(
        cfg.harmful_dataset_key,
        split=cfg.harmful_split,
        max_samples=cfg.max_harmful_prompts,
    )
    harmful_texts, harmful_labels = prepare_text_and_labels(
        harmful_ds,
        text_field=harmful_spec.text_field,
        label_field=harmful_spec.label_field,
        label_fn=harmful_spec.label_fn,
    )
    if harmful_labels is None:
        raise ValueError("Harmful seed dataset must provide labels.")

    benign_ds, benign_spec = load_examples(
        cfg.benign_dataset_key,
        split=cfg.benign_split,
        max_samples=cfg.max_benign_prompts,
    )
    benign_texts, benign_labels = prepare_text_and_labels(
        benign_ds,
        text_field=benign_spec.text_field,
        label_field=benign_spec.label_field,
        label_fn=benign_spec.label_fn,
    )
    if benign_labels is None:
        raise ValueError("Benign seed dataset must provide labels.")

    harmful_prompts = [
        t
        for t, y in zip(harmful_texts, harmful_labels.tolist(), strict=False)
        if int(y) == 1
    ]
    benign_prompts = [
        t
        for t, y in zip(benign_texts, benign_labels.tolist(), strict=False)
        if int(y) == 0
    ]

    harmful_prompts = _dedupe_keep_order(harmful_prompts)
    benign_prompts = _dedupe_keep_order(benign_prompts)

    if not harmful_prompts:
        raise ValueError("No harmful prompts found from harmful seed dataset.")
    if not benign_prompts:
        raise ValueError("No benign prompts found from benign seed dataset.")

    return harmful_prompts, benign_prompts


def _split_seed_and_heldout(
    harmful: list[str],
    benign: list[str],
    heldout_size: int,
    rng: np.random.Generator,
    min_seed_per_class: int = 16,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    harmful = _dedupe_keep_order(harmful)
    benign = _dedupe_keep_order(benign)

    if heldout_size < 2:
        raise ValueError("heldout_prompts must be >= 2 for class-balanced splits.")
    if len(harmful) <= min_seed_per_class or len(benign) <= min_seed_per_class:
        raise ValueError(
            "Insufficient per-class prompts for balanced split. "
            f"Need > {min_seed_per_class} per class, got harmful={len(harmful)} benign={len(benign)}."
        )

    target_h = heldout_size // 2
    target_b = heldout_size - target_h
    max_h = len(harmful) - min_seed_per_class
    max_b = len(benign) - min_seed_per_class

    if max_h < target_h or max_b < target_b:
        raise ValueError(
            "Cannot satisfy strict class-balanced heldout split with current data. "
            f"Need harmful>={target_h + min_seed_per_class}, benign>={target_b + min_seed_per_class}; "
            f"got harmful={len(harmful)}, benign={len(benign)}."
        )
    held_h = target_h
    held_b = target_b

    h_idx = np.arange(len(harmful))
    b_idx = np.arange(len(benign))
    rng.shuffle(h_idx)
    rng.shuffle(b_idx)

    heldout_h = [harmful[int(i)] for i in h_idx[:held_h]]
    seed_h = [harmful[int(i)] for i in h_idx[held_h:]]
    heldout_b = [benign[int(i)] for i in b_idx[:held_b]]
    seed_b = [benign[int(i)] for i in b_idx[held_b:]]

    heldout = _interleave_balanced(heldout_h, heldout_b, rng)
    split_info = {
        "heldout_harmful": len(heldout_h),
        "heldout_benign": len(heldout_b),
        "seed_harmful": len(seed_h),
        "seed_benign": len(seed_b),
    }
    return seed_h, seed_b, heldout, split_info


def _interleave_balanced(
    a: list[str],
    b: list[str],
    rng: np.random.Generator,
) -> list[str]:
    aa = list(a)
    bb = list(b)
    rng.shuffle(aa)
    rng.shuffle(bb)

    out: list[str] = []
    ia = 0
    ib = 0
    while ia < len(aa) or ib < len(bb):
        if ia < len(aa):
            out.append(aa[ia])
            ia += 1
        if ib < len(bb):
            out.append(bb[ib])
            ib += 1
    return _dedupe_keep_order(out)


def _make_run_cfg(cfg: ExperimentConfig) -> RunConfig:
    return RunConfig(
        model_key=cfg.model_key,
        dataset_key="toy_contrastive",
        split="train",
        max_samples=1,
        max_length=cfg.max_length,
        layer_idx=cfg.layer_idx,
        device=resolve_device(cfg.device),
        use_generate=True,
        max_new_tokens=cfg.max_new_tokens,
        include_prompt_in_trajectory=cfg.include_prompt_in_trajectory,
    )


def _page_hinkley_count(
    x: np.ndarray,
    delta_scale: float,
    lambda_scale: float,
) -> int:
    if len(x) < 2:
        return 0

    x = x.astype(np.float64)
    sigma = float(np.std(x) + 1e-8)
    delta = float(delta_scale * sigma)
    threshold = float(max(lambda_scale * sigma, 1e-6))

    mean = 0.0
    cumulative = 0.0
    min_cumulative = 0.0
    count = 0

    for i, val in enumerate(x):
        mean += (val - mean) / float(i + 1)
        cumulative += val - mean - delta
        min_cumulative = min(min_cumulative, cumulative)
        if cumulative - min_cumulative > threshold:
            count += 1
            cumulative = 0.0
            min_cumulative = 0.0
            mean = val
    return count


def _compute_measures(
    traj: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[float, float, float]:
    if traj.shape[0] < 2:
        return 0.0, 0.0, 0.0

    d1 = np.diff(traj.astype(np.float32), axis=0)
    speed = float(np.mean(np.linalg.norm(d1, axis=1)))

    if d1.shape[0] >= 2:
        d2 = np.diff(d1, axis=0)
        curvature = float(np.mean(np.linalg.norm(d2, axis=1)))
    else:
        curvature = 0.0

    residual_norm = np.linalg.norm(d1, axis=1)
    cp = float(
        _page_hinkley_count(
            residual_norm,
            delta_scale=cfg.page_hinkley_delta,
            lambda_scale=cfg.page_hinkley_lambda,
        )
    )
    return speed, curvature, cp


def _rollout_candidates(
    prompts: list[tuple[str, str, str | None]],
    model: Any,
    tokenizer: Any,
    run_cfg: RunConfig,
    cfg: ExperimentConfig,
    uid_prefix: str,
) -> list[Candidate]:
    if not prompts:
        return []

    out: list[Candidate] = []
    starts = _chunk_starts(len(prompts), max(1, cfg.rollout_batch_size))
    iterator = _iter_with_progress(
        starts=starts,
        desc=f"extract rollouts ({len(prompts)})",
        enabled=cfg.show_progress and len(starts) > 1,
    )
    for start in iterator:
        prompt_batch = prompts[start : start + cfg.rollout_batch_size]
        texts = [p[0] for p in prompt_batch]
        per_layer, token_texts = extract_multi_layer_trajectories(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            layer_indices=[cfg.layer_idx],
            max_length=cfg.max_length,
            device=run_cfg.device or "cpu",
            cfg=run_cfg,
        )
        trajs = per_layer[cfg.layer_idx]

        for i, (prompt, source, parent_prompt) in enumerate(prompt_batch):
            toks = token_texts[i]
            snippet_tokens = toks[-cfg.horizon_tokens :]
            snippet = tokenizer.convert_tokens_to_string(snippet_tokens)
            global_i = start + i
            cid = f"{uid_prefix}_{global_i:04d}_{_stable_text_hash(prompt)[:10]}"
            measures = _compute_measures(trajs[i], cfg)
            out.append(
                Candidate(
                    candidate_id=cid,
                    prompt=prompt,
                    source=source,
                    parent_prompt=parent_prompt,
                    trajectory=trajs[i].astype(np.float32),
                    token_texts=toks,
                    rollout_snippet=snippet,
                    measures=measures,
                )
            )
    return out


def _fit_state_pca(states: np.ndarray, n_components: int, seed: int) -> PCA | None:
    if states.ndim != 2 or states.shape[0] < 2:
        return None
    k = int(min(n_components, states.shape[0], states.shape[1]))
    if k < 1:
        return None
    pca = PCA(n_components=k, random_state=seed)
    pca.fit(states)
    return pca


def _project_pca(z: np.ndarray, pca: PCA | None, target_dim: int) -> np.ndarray:
    if pca is None:
        return np.zeros(target_dim, dtype=np.float32)
    vec = pca.transform(z.reshape(1, -1))[0].astype(np.float32)
    if len(vec) >= target_dim:
        return vec[:target_dim]
    pad = np.zeros(target_dim - len(vec), dtype=np.float32)
    return np.concatenate([vec, pad], axis=0)


def _build_psi(
    z_query: np.ndarray,
    z_prefix_mean: np.ndarray,
    pca_proj: np.ndarray,
    measures: tuple[float, float, float],
) -> np.ndarray:
    return np.concatenate(
        [
            z_query.astype(np.float32),
            z_prefix_mean.astype(np.float32),
            pca_proj.astype(np.float32),
            np.asarray(measures, dtype=np.float32),
        ],
        axis=0,
    )


def _select_query_states(
    candidates: list[Candidate],
    probe: BayesianLogisticProbe,
    pca: PCA | None,
    cfg: ExperimentConfig,
) -> None:
    for cand in candidates:
        traj = cand.trajectory
        max_p = max(1, min(cfg.horizon_tokens, traj.shape[0]))

        best_entropy = -1.0
        best_idx = max_p - 1
        best_zq = traj[best_idx]
        best_zm = traj[:max_p].mean(axis=0)
        best_proj = _project_pca(best_zq, pca, cfg.pca_components)
        best_psi = _build_psi(best_zq, best_zm, best_proj, cand.measures)

        if probe.fitted_:
            for p in range(1, max_p + 1):
                zq = traj[p - 1]
                zm = traj[:p].mean(axis=0)
                proj = _project_pca(zq, pca, cfg.pca_components)
                psi = _build_psi(zq, zm, proj, cand.measures)
                ent = float(probe.predictive_entropy(psi.reshape(1, -1))[0])
                if ent > best_entropy:
                    best_entropy = ent
                    best_idx = p - 1
                    best_zq = zq
                    best_zm = zm
                    best_proj = proj
                    best_psi = psi

        cand.query_index = int(best_idx)
        cand.z_query = best_zq.astype(np.float32)
        cand.z_prefix_mean = best_zm.astype(np.float32)
        cand.pca_proj = best_proj.astype(np.float32)
        cand.psi = best_psi.astype(np.float32)


def _score_candidates(
    candidates: list[Candidate],
    state: StrategyState,
    cfg: ExperimentConfig,
) -> None:
    if not candidates:
        return

    X = np.stack([c.psi for c in candidates if c.psi is not None], axis=0)

    if state.probe.fitted_:
        unc = state.probe.predictive_entropy(X)
    else:
        unc = np.full(X.shape[0], 0.5, dtype=np.float64)

    if state.labeled_pca:
        labeled = np.stack(state.labeled_pca, axis=0).astype(np.float64)
        cand_pca = np.stack(
            [c.pca_proj for c in candidates if c.pca_proj is not None], axis=0
        ).astype(np.float64)
        dists = np.linalg.norm(cand_pca[:, None, :] - labeled[None, :, :], axis=2)
        nov = dists.min(axis=1)
    else:
        nov = np.ones(X.shape[0], dtype=np.float64)

    unc_n = _normalize(unc)
    nov_n = _normalize(nov)

    for i, cand in enumerate(candidates):
        cand.q_unc = float(unc_n[i])
        cand.q_nov = float(nov_n[i])
        if state.name == "uncertainty_only":
            cand.q_total = float(unc_n[i])
        elif state.name == "random":
            cand.q_total = 0.0
        else:
            cand.q_total = float(
                (cfg.alpha_uncertainty * unc_n[i]) + (cfg.alpha_novelty * nov_n[i])
            )


def _make_archive(
    cfg: ExperimentConfig,
    warm_measures: list[tuple[float, float, float]],
) -> tuple[Any, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    try:
        from ribs.archives import GridArchive
    except Exception as e:  # pragma: no cover - import path check
        raise ImportError("ribs is required for QD archive support.") from e

    arr = np.array(warm_measures, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        ranges = ((0.0, 1.0), (0.0, 1.0), (0.0, 8.0))
    else:
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        lo = mins - (0.1 * spans)
        hi = maxs + (0.1 * spans)
        ranges = (
            (float(lo[0]), float(hi[0])),
            (float(lo[1]), float(hi[1])),
            (float(lo[2]), float(hi[2])),
        )

    archive = GridArchive(
        solution_dim=2,
        dims=list(cfg.bd_dims),
        ranges=list(ranges),
    )
    return archive, ranges


def _cell_index(
    measures: tuple[float, float, float],
    dims: tuple[int, int, int],
    ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[int, int, int]:
    idxs: list[int] = []
    for i, val in enumerate(measures):
        lo, hi = ranges[i]
        dim = dims[i]
        if hi - lo <= 1e-12:
            idxs.append(0)
            continue
        v = float(np.clip(val, lo, np.nextafter(hi, lo)))
        frac = (v - lo) / (hi - lo)
        idx = int(np.floor(frac * dim))
        idxs.append(int(np.clip(idx, 0, dim - 1)))
    return idxs[0], idxs[1], idxs[2]


def _select_candidates(
    candidates: list[Candidate],
    state: StrategyState,
    cfg: ExperimentConfig,
    iteration_idx: int,
) -> list[Candidate]:
    if not candidates:
        return []
    n_pick = min(cfg.batch_size, len(candidates))

    if state.name == "random":
        perm = state.rng.permutation(len(candidates))
        return [candidates[int(i)] for i in perm[:n_pick]]

    if state.name == "uncertainty_only":
        ordered = sorted(candidates, key=lambda c: c.q_unc, reverse=True)
        return ordered[:n_pick]

    if state.archive is None or state.archive_ranges is None:
        raise ValueError("QD strategy requires archive and ranges.")

    by_cell: dict[tuple[int, int, int], list[Candidate]] = {}
    for i, cand in enumerate(candidates):
        measures = np.asarray(cand.measures, dtype=np.float32)
        solution = np.asarray([float(iteration_idx), float(i)], dtype=np.float32)
        state.archive.add(
            solution=solution, objective=float(cand.q_total), measures=measures
        )

        cell = _cell_index(cand.measures, cfg.bd_dims, state.archive_ranges)
        state.archive_cells_seen.add(cell)
        by_cell.setdefault(cell, []).append(cand)

    denom = float(np.prod(np.array(cfg.bd_dims, dtype=np.int64)))
    coverage = float(len(state.archive_cells_seen) / denom)
    state.archive_coverage_history.append(coverage)

    for cell in by_cell:
        by_cell[cell].sort(key=lambda c: c.q_total, reverse=True)

    stratified_quota = int(math.ceil(0.8 * n_pick))
    one_per_cell = [vals[0] for vals in by_cell.values() if vals]
    one_per_cell.sort(key=lambda c: c.q_total, reverse=True)

    selected: list[Candidate] = one_per_cell[:stratified_quota]
    selected_ids = {c.candidate_id for c in selected}

    all_sorted = sorted(candidates, key=lambda c: c.q_total, reverse=True)
    for cand in all_sorted:
        if len(selected) >= n_pick:
            break
        if cand.candidate_id in selected_ids:
            continue
        selected.append(cand)
        selected_ids.add(cand.candidate_id)

    return selected


def _stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _judge_key(prompt: str, snippet: str) -> str:
    return _stable_text_hash(f"{prompt}\n<ROLL>\n{snippet}")


def _label_candidates(
    candidates: list[Candidate],
    llm: ProxyLLM,
    cache: JudgeCache,
) -> list[tuple[Candidate, JudgeResult]]:
    out: list[tuple[Candidate, JudgeResult]] = []
    to_query: list[Candidate] = []

    for cand in candidates:
        key = _judge_key(cand.prompt, cand.rollout_snippet)
        cached = cache.get(key)
        if cached is None:
            to_query.append(cand)
        else:
            out.append(
                (
                    cand,
                    JudgeResult(
                        unsafe=bool(cached.get("unsafe", False)),
                        confidence=float(cached.get("confidence", 0.5)),
                        rationale=str(cached.get("rationale", "cache")),
                    ),
                )
            )

    if to_query:
        pairs = [(c.prompt, c.rollout_snippet) for c in to_query]
        judged = llm.judge_batch(pairs)
        for cand, result in zip(to_query, judged, strict=False):
            key = _judge_key(cand.prompt, cand.rollout_snippet)
            cache.set(
                key,
                {
                    "unsafe": bool(result.unsafe),
                    "confidence": float(result.confidence),
                    "rationale": result.rationale,
                },
            )
            out.append((cand, result))

    return out


def _candidate_record(
    cand: Candidate,
    strategy: str,
    label: int,
    result: JudgeResult,
    iteration: int,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "iteration": int(iteration),
        "candidate_id": cand.candidate_id,
        "prompt": cand.prompt,
        "source": cand.source,
        "parent_prompt": cand.parent_prompt,
        "label": int(label),
        "judge_confidence": float(result.confidence),
        "judge_rationale": result.rationale,
        "query_index": int(cand.query_index),
        "measures": {
            "mean_speed": float(cand.measures[0]),
            "curvature": float(cand.measures[1]),
            "change_points": float(cand.measures[2]),
        },
        "scores": {
            "uncertainty": float(cand.q_unc),
            "novelty": float(cand.q_nov),
            "total": float(cand.q_total),
        },
        "cache_key": _judge_key(cand.prompt, cand.rollout_snippet),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str))
        f.write("\n")


def _build_eval_matrix(
    heldout_candidates: list[Candidate],
) -> np.ndarray:
    if not heldout_candidates:
        raise ValueError("No heldout candidates available.")
    if heldout_candidates[0].psi is None:
        raise ValueError("Heldout candidates missing psi features.")
    return np.stack([c.psi for c in heldout_candidates], axis=0)


def _ece(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(probs[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return float(ece)


def _evaluate_strategy(
    state: StrategyState,
    heldout_X: np.ndarray,
    heldout_y: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    probs = state.probe.predict_proba(heldout_X)
    entropy = _binary_entropy(probs)
    margin = state.probe.margin(heldout_X)

    if len(np.unique(heldout_y)) >= 2:
        auroc = float(roc_auc_score(heldout_y, probs))
    else:
        auroc = float("nan")

    brier = float(brier_score_loss(heldout_y, probs))
    ece_val = _ece(heldout_y, probs, bins=10)

    near = margin < epsilon
    u_eps = float(np.mean(entropy[near])) if np.any(near) else 0.0

    return {
        "heldout_auroc": auroc,
        "heldout_brier": brier,
        "heldout_ece": ece_val,
        "global_uncertainty": float(np.mean(entropy)),
        "uncertainty_mass_near_boundary": u_eps,
        "near_boundary_fraction": float(np.mean(near)),
    }


def _warm_start_candidates(
    harmful_seed_prompts: list[str],
    benign_seed_prompts: list[str],
    target_count: int,
    llm: ProxyLLM,
    rng: np.random.Generator,
) -> list[tuple[str, str, str | None]]:
    n_h = target_count // 2
    n_b = target_count - n_h
    h_parents = [
        harmful_seed_prompts[int(i)]
        for i in rng.integers(0, len(harmful_seed_prompts), size=max(n_h, 1))
    ]
    b_parents = [
        benign_seed_prompts[int(i)]
        for i in rng.integers(0, len(benign_seed_prompts), size=max(n_b, 1))
    ]
    parents = _interleave_balanced(h_parents, b_parents, rng)
    mutated = llm.mutate_prompts(parents, rng=rng, target_count=target_count)
    prompts = _dedupe_keep_order(mutated)

    out: list[tuple[str, str, str | None]] = [
        (p, "mutation", None) for p in prompts[:target_count]
    ]
    if len(out) < target_count:
        needed = target_count - len(out)
        n_h_fb = needed // 2
        n_b_fb = needed - n_h_fb
        fb_h = [
            harmful_seed_prompts[int(i)]
            for i in rng.integers(0, len(harmful_seed_prompts), size=max(n_h_fb, 1))
        ]
        fb_b = [
            benign_seed_prompts[int(i)]
            for i in rng.integers(0, len(benign_seed_prompts), size=max(n_b_fb, 1))
        ]
        fallback = _interleave_balanced(fb_h, fb_b, rng)[:needed]
        out.extend([(p, "seed_fallback", None) for p in fallback])
    return out[:target_count]


def _build_candidate_prompts(
    state: StrategyState,
    seed_prompts: list[str],
    cfg: ExperimentConfig,
    llm: ProxyLLM,
) -> list[tuple[str, str, str | None]]:
    n_total = cfg.candidate_pool_size
    n_random = int(round(cfg.random_seed_fraction * n_total))
    n_mut = max(0, n_total - n_random)

    parent_pool = _dedupe_keep_order(state.prompt_pool + seed_prompts)
    if not parent_pool:
        parent_pool = seed_prompts

    parents = [
        parent_pool[int(i)]
        for i in state.rng.integers(0, len(parent_pool), size=max(1, n_mut))
    ]

    mutated = llm.mutate_prompts(parents, rng=state.rng, target_count=n_mut)
    prompts: list[tuple[str, str, str | None]] = []

    used: set[str] = set()
    for i, p in enumerate(mutated):
        if p in used:
            continue
        used.add(p)
        prompts.append((p, "mutation", parents[i % len(parents)]))
        if len(prompts) >= n_mut:
            break

    if len(prompts) < n_mut:
        needed = n_mut - len(prompts)
        fallback = [
            parent_pool[int(i)]
            for i in state.rng.integers(0, len(parent_pool), size=needed)
        ]
        for p in fallback:
            prompts.append((p, "parent_fallback", None))

    random_prompts = [
        seed_prompts[int(i)]
        for i in state.rng.integers(0, len(seed_prompts), size=n_random)
    ]
    for p in random_prompts:
        prompts.append((p, "seed_random", None))

    dedup_out: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    for row in prompts:
        if row[0] in seen:
            continue
        seen.add(row[0])
        dedup_out.append(row)

    if len(dedup_out) < n_total:
        needed = n_total - len(dedup_out)
        extra = [
            seed_prompts[int(i)]
            for i in state.rng.integers(0, len(seed_prompts), size=needed)
        ]
        dedup_out.extend((p, "seed_topup", None) for p in extra)

    return dedup_out[:n_total]


def _fit_probe_from_state(state: StrategyState) -> None:
    X = np.stack(state.X, axis=0).astype(np.float64)
    y = np.asarray(state.y, dtype=np.int64)
    state.probe.fit(X, y)


def _initialize_strategy_state(
    name: str,
    cfg: ExperimentConfig,
    rng_seed: int,
    warm_candidates: list[Candidate],
    warm_labels: list[int],
    warm_judge: list[JudgeResult],
    pca: PCA | None,
) -> StrategyState:
    state = StrategyState(
        name=name,
        rng=np.random.default_rng(rng_seed),
        probe=BayesianLogisticProbe(c=cfg.logistic_c, max_iter=cfg.logistic_max_iter),
    )

    for cand, y, _jr in zip(warm_candidates, warm_labels, warm_judge, strict=False):
        if cand.psi is None or cand.pca_proj is None:
            raise ValueError("Warm candidate missing psi/pca features.")
        state.X.append(cand.psi)
        state.y.append(int(y))
        state.labeled_pca.append(cand.pca_proj)
        state.prompt_pool.append(cand.prompt)

    state.prompt_pool = _dedupe_keep_order(state.prompt_pool)
    _fit_probe_from_state(state)

    if name == "qd_uncertainty":
        archive, ranges = _make_archive(cfg, [c.measures for c in warm_candidates])
        state.archive = archive
        state.archive_ranges = ranges
        for i, cand in enumerate(warm_candidates):
            solution = np.asarray([0.0, float(i)], dtype=np.float32)
            measures = np.asarray(cand.measures, dtype=np.float32)
            objective = float(cand.q_total)
            archive.add(solution=solution, objective=objective, measures=measures)
            cell = _cell_index(cand.measures, cfg.bd_dims, ranges)
            state.archive_cells_seen.add(cell)
        denom = float(np.prod(np.array(cfg.bd_dims, dtype=np.int64)))
        state.archive_coverage_history.append(
            float(len(state.archive_cells_seen) / denom)
        )

    return state


def _prepare_heldout_candidates(
    heldout_prompts: list[str],
    model: Any,
    tokenizer: Any,
    run_cfg: RunConfig,
    cfg: ExperimentConfig,
    pca: PCA | None,
) -> list[Candidate]:
    rows = [(p, "heldout", None) for p in heldout_prompts]
    cands = _rollout_candidates(
        rows, model, tokenizer, run_cfg, cfg, uid_prefix="heldout"
    )

    dummy_probe = BayesianLogisticProbe(
        c=cfg.logistic_c, max_iter=cfg.logistic_max_iter
    )
    _select_query_states(cands, dummy_probe, pca, cfg)
    for cand in cands:
        if cand.trajectory.shape[0] >= 1:
            p = min(cfg.horizon_tokens, cand.trajectory.shape[0])
            cand.query_index = p - 1
            zq = cand.trajectory[cand.query_index]
            zm = cand.trajectory[:p].mean(axis=0)
            proj = _project_pca(zq, pca, cfg.pca_components)
            cand.z_query = zq.astype(np.float32)
            cand.z_prefix_mean = zm.astype(np.float32)
            cand.pca_proj = proj.astype(np.float32)
            cand.psi = _build_psi(zq, zm, proj, cand.measures)
    return cands


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    if cfg.alpha_uncertainty < 0 or cfg.alpha_novelty < 0:
        raise ValueError("Acquisition weights must be non-negative.")
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if cfg.warm_start_labels < 4:
        raise ValueError("warm_start_labels should be >= 4.")
    if cfg.label_budget <= cfg.warm_start_labels:
        raise ValueError("label_budget must exceed warm_start_labels.")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    labeled_jsonl = run_dir / "labeled_pool.jsonl"
    metrics_jsonl = run_dir / "iteration_metrics.jsonl"
    cache_path = run_dir / "judge_cache.json"
    cache = JudgeCache(cache_path)

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    rng = np.random.default_rng(cfg.random_state)
    llm = ProxyLLM(cfg)

    harmful_seed, benign_seed = _load_seed_prompts(cfg)
    seed_harmful, seed_benign, heldout_prompts, split_info = _split_seed_and_heldout(
        harmful_seed,
        benign_seed,
        heldout_size=cfg.heldout_prompts,
        rng=rng,
    )
    seed_prompts = _interleave_balanced(seed_harmful, seed_benign, rng)

    run_cfg = _make_run_cfg(cfg)
    model, tokenizer = load_model_and_tokenizer(
        run_cfg.model_key, run_cfg.device or "cpu"
    )

    warm_prompt_rows = _warm_start_candidates(
        seed_harmful, seed_benign, cfg.warm_start_labels, llm, rng
    )
    warm_candidates = _rollout_candidates(
        warm_prompt_rows,
        model,
        tokenizer,
        run_cfg,
        cfg,
        uid_prefix="warm",
    )

    for cand in warm_candidates:
        p = max(1, min(cfg.horizon_tokens, cand.trajectory.shape[0]))
        cand.query_index = p - 1
        zq = cand.trajectory[cand.query_index]
        zm = cand.trajectory[:p].mean(axis=0)
        cand.z_query = zq.astype(np.float32)
        cand.z_prefix_mean = zm.astype(np.float32)

    warm_states = np.stack(
        [c.z_query for c in warm_candidates if c.z_query is not None], axis=0
    )
    pca = _fit_state_pca(warm_states, cfg.pca_components, cfg.random_state)

    for cand in warm_candidates:
        assert cand.z_query is not None and cand.z_prefix_mean is not None
        cand.pca_proj = _project_pca(cand.z_query, pca, cfg.pca_components)
        cand.psi = _build_psi(
            cand.z_query, cand.z_prefix_mean, cand.pca_proj, cand.measures
        )

    warm_judged = _label_candidates(warm_candidates, llm, cache)
    warm_judge_map = {c.candidate_id: jr for c, jr in warm_judged}
    warm_labels = [
        1 if warm_judge_map[c.candidate_id].unsafe else 0 for c in warm_candidates
    ]
    warm_judge = [warm_judge_map[c.candidate_id] for c in warm_candidates]

    probe_boot = BayesianLogisticProbe(c=cfg.logistic_c, max_iter=cfg.logistic_max_iter)
    X_boot = np.stack([c.psi for c in warm_candidates if c.psi is not None], axis=0)
    y_boot = np.asarray(warm_labels, dtype=np.int64)
    probe_boot.fit(X_boot, y_boot)
    boot_entropy = probe_boot.predictive_entropy(X_boot)
    for cand, ent in zip(warm_candidates, boot_entropy, strict=False):
        cand.q_unc = float(ent)
        cand.q_nov = 0.0
        cand.q_total = float(ent)

    heldout_candidates = _prepare_heldout_candidates(
        heldout_prompts,
        model,
        tokenizer,
        run_cfg,
        cfg,
        pca,
    )
    heldout_judged = _label_candidates(heldout_candidates, llm, cache)
    heldout_label_map = {c.candidate_id: jr for c, jr in heldout_judged}
    heldout_y = np.array(
        [
            1 if heldout_label_map[c.candidate_id].unsafe else 0
            for c in heldout_candidates
        ],
        dtype=np.int64,
    )
    heldout_X = _build_eval_matrix(heldout_candidates)

    strategies: dict[str, StrategyState] = {}
    for i, name in enumerate(cfg.strategies):
        if name not in {"qd_uncertainty", "uncertainty_only", "random"}:
            raise ValueError(f"Unknown strategy: {name}")
        strategies[name] = _initialize_strategy_state(
            name=name,
            cfg=cfg,
            rng_seed=cfg.random_state + 100 + i,
            warm_candidates=warm_candidates,
            warm_labels=warm_labels,
            warm_judge=warm_judge,
            pca=pca,
        )

    for strategy_name, state in strategies.items():
        for cand, y, jr in zip(warm_candidates, warm_labels, warm_judge, strict=False):
            _append_jsonl(
                labeled_jsonl,
                _candidate_record(
                    cand, strategy=strategy_name, label=y, result=jr, iteration=0
                ),
            )

    for name, state in strategies.items():
        init_metrics = _evaluate_strategy(
            state,
            heldout_X=heldout_X,
            heldout_y=heldout_y,
            epsilon=cfg.boundary_margin_epsilon,
        )
        init_metrics.update(
            {
                "strategy": name,
                "iteration": 0,
                "labels_used": int(len(state.y)),
                "archive_coverage": (
                    float(state.archive_coverage_history[-1])
                    if state.archive_coverage_history
                    else None
                ),
            }
        )
        state.metrics_history.append(init_metrics)
        _append_jsonl(metrics_jsonl, init_metrics)

    for name, state in strategies.items():
        iter_idx = 1
        pbar = _make_progress_bar(
            total=cfg.label_budget,
            initial=len(state.y),
            desc=f"{name} labels",
            enabled=cfg.show_progress,
        )
        while len(state.y) < cfg.label_budget:
            prompt_rows = _build_candidate_prompts(state, seed_prompts, cfg, llm)
            cand_pool = _rollout_candidates(
                prompt_rows,
                model,
                tokenizer,
                run_cfg,
                cfg,
                uid_prefix=f"{name}_it{iter_idx:03d}",
            )
            _select_query_states(cand_pool, state.probe, pca, cfg)
            _score_candidates(cand_pool, state, cfg)

            selected = _select_candidates(cand_pool, state, cfg, iteration_idx=iter_idx)
            judged = _label_candidates(selected, llm, cache)

            added_this_iter = 0
            for cand, jr in judged:
                y = 1 if jr.unsafe else 0
                assert cand.psi is not None and cand.pca_proj is not None
                state.X.append(cand.psi)
                state.y.append(y)
                state.labeled_pca.append(cand.pca_proj)
                state.prompt_pool.append(cand.prompt)
                added_this_iter += 1
                _append_jsonl(
                    labeled_jsonl,
                    _candidate_record(
                        cand,
                        strategy=name,
                        label=y,
                        result=jr,
                        iteration=iter_idx,
                    ),
                )
                if len(state.y) >= cfg.label_budget:
                    break

            state.prompt_pool = _dedupe_keep_order(state.prompt_pool)
            _fit_probe_from_state(state)

            metrics = _evaluate_strategy(
                state,
                heldout_X=heldout_X,
                heldout_y=heldout_y,
                epsilon=cfg.boundary_margin_epsilon,
            )
            metrics.update(
                {
                    "strategy": name,
                    "iteration": int(iter_idx),
                    "labels_used": int(len(state.y)),
                    "archive_coverage": (
                        float(state.archive_coverage_history[-1])
                        if state.archive_coverage_history
                        else None
                    ),
                }
            )
            state.metrics_history.append(metrics)
            _append_jsonl(metrics_jsonl, metrics)

            if pbar is not None:
                pbar.update(added_this_iter)
                pbar.set_postfix(iteration=iter_idx, refresh=False)
            cache.save()
            iter_idx += 1
        if pbar is not None:
            pbar.close()

    cache.save()

    qd_state = strategies.get("qd_uncertainty")
    if qd_state is not None:
        cells = sorted(list(qd_state.archive_cells_seen))
        arr_cells = (
            np.array(cells, dtype=np.int32)
            if cells
            else np.zeros((0, 3), dtype=np.int32)
        )
        arr_cov = np.array(qd_state.archive_coverage_history, dtype=np.float32)
        np.savez(
            run_dir / "archive_snapshots.npz",
            qd_cells=arr_cells,
            qd_coverage=arr_cov,
        )
    else:
        np.savez(
            run_dir / "archive_snapshots.npz",
            qd_cells=np.zeros((0, 3), dtype=np.int32),
            qd_coverage=np.zeros((0,), dtype=np.float32),
        )

    final_metrics = {
        name: state.metrics_history[-1]
        for name, state in strategies.items()
        if state.metrics_history
    }

    label_efficiency: dict[str, dict[str, list[float]]] = {}
    for name, state in strategies.items():
        labels_used = [float(m["labels_used"]) for m in state.metrics_history]
        ueps = [
            float(m["uncertainty_mass_near_boundary"]) for m in state.metrics_history
        ]
        auroc = [float(m["heldout_auroc"]) for m in state.metrics_history]
        label_efficiency[name] = {
            "labels_used": labels_used,
            "uncertainty_mass_near_boundary": ueps,
            "heldout_auroc": auroc,
        }

    report = {
        "experiment": asdict(cfg),
        "run_dir": str(run_dir),
        "data": {
            "n_seed_prompts": int(len(seed_prompts)),
            "n_heldout_prompts": int(len(heldout_prompts)),
            "n_harmful_seed": int(len(harmful_seed)),
            "n_benign_seed": int(len(benign_seed)),
            "split_balance": split_info,
        },
        "evaluation": {
            "strategies": final_metrics,
            "label_efficiency_curves": label_efficiency,
        },
        "artifacts": {
            "config_json": str(run_dir / "config.json"),
            "labeled_pool_jsonl": str(labeled_jsonl),
            "iteration_metrics_jsonl": str(metrics_jsonl),
            "archive_snapshots_npz": str(run_dir / "archive_snapshots.npz"),
            "judge_cache_json": str(cache_path),
        },
    }

    final_report_path = run_dir / "final_report.json"
    final_report_path.write_text(json.dumps(report, indent=2, default=str))

    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_json.write_text(json.dumps(report, indent=2, default=str))

    return report


def main() -> None:
    cfg = _parse_args()
    report = run_experiment(cfg)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if cfg.output_json is not None:
        print(f"Wrote results: {cfg.output_json}")


if __name__ == "__main__":
    main()
