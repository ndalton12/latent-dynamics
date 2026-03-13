"""LLM-as-judge utilities for safety classification.

Shared between the benchmark suite (generation-trajectory labeling) and the
QD active boundary experiment (candidate labeling). Provides:

  - JudgeResult / JudgeCache  -- dataclass and disk-backed cache
  - SafetyJudge               -- thin wrapper around flashlite for batched judging
  - judge_texts               -- high-level convenience: judge (prompt, completion) pairs
  - judge_activation_metadata -- convenience for activations metadata payloads

The judge uses a Jinja template at prompts/judge/safety_judge.jinja and
communicates with an OpenAI-compatible API via flashlite. OPENAI_API_KEY must
be set in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from latent_dynamics.utils import load_activation_bundle

DEFAULT_JUDGE_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompts" / "judge"
SAFETY_JUDGE_TEMPLATE_NAME = "safety_judge"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    unsafe: bool
    confidence: float
    rationale: str
    compliance: bool = False


class _StructuredJudgeOutput(BaseModel):
    unsafe: bool
    compliance: bool
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class JudgeCache:
    """Persistent JSON cache keyed by sha256(prompt + completion)."""

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

    def __len__(self) -> int:
        return len(self.data)


def stable_text_hash(text: str) -> str:
    """Stable sha256 hex-digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def judge_cache_key(prompt: str, completion: str) -> str:
    return stable_text_hash(f"{prompt}\n<ROLL>\n{completion}")


# ---------------------------------------------------------------------------
# SafetyJudge
# ---------------------------------------------------------------------------


class SafetyJudge:
    """Batched LLM safety judge backed by flashlite.

    Args:
        model:               OpenAI-compatible model name (e.g. "gpt-4o-mini").
        template_dir:        Directory containing safety_judge.jinja. Defaults
                             to prompts/judge/ inside the package.
        max_concurrency:     Max concurrent API requests.
        batch_size:          Max requests sent per batch.
        requests_per_minute: Rate limit.
        tokens_per_minute:   Optional token-level rate limit.
        show_progress:       Show tqdm bars for large batches.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        template_dir: Path | None = None,
        max_concurrency: int = 12,
        batch_size: int = 32,
        requests_per_minute: int = 120,
        tokens_per_minute: int | None = None,
        show_progress: bool = True,
    ) -> None:
        try:
            from flashlite import Flashlite, RateLimitConfig, RetryConfig
        except Exception as e:
            raise ImportError(
                "flashlite is required for LLM judging. "
                "Install it and set OPENAI_API_KEY."
            ) from e

        tdir = template_dir or DEFAULT_JUDGE_TEMPLATE_DIR
        if not tdir.exists():
            raise FileNotFoundError(f"Judge template directory not found: {tdir}")

        rate_kwargs: dict[str, Any] = {"requests_per_minute": requests_per_minute}
        if tokens_per_minute is not None:
            rate_kwargs["tokens_per_minute"] = tokens_per_minute

        self.client = Flashlite(
            default_model=model,
            template_dir=str(tdir),
            retry=RetryConfig(max_attempts=3),
            rate_limit=RateLimitConfig(**rate_kwargs),
            log_requests=False,
        )
        self.max_concurrency = max_concurrency
        self.batch_size = max(1, batch_size)
        self.show_progress = show_progress

    def judge_batch(self, pairs: list[tuple[str, str]]) -> list[JudgeResult]:
        """Judge a list of (prompt, completion) pairs. Returns JudgeResult per pair."""
        if not pairs:
            return []

        reqs = [
            {
                "template": SAFETY_JUDGE_TEMPLATE_NAME,
                "variables": {"prompt": prompt, "rollout": completion},
                "response_model": _StructuredJudgeOutput,
                "structured_retries": 3,
            }
            for prompt, completion in pairs
        ]

        results: list[JudgeResult] = []
        starts = _chunk_starts(len(reqs), self.batch_size)
        iterator = _iter_with_progress(
            starts,
            desc=f"judge ({len(pairs)})",
            enabled=self.show_progress and len(starts) > 1,
        )
        for start in iterator:
            batch = reqs[start : start + self.batch_size]
            resps = _run_async(
                self.client.complete_many(batch, max_concurrency=self.max_concurrency)
            )
            for resp in resps:
                if not isinstance(resp, _StructuredJudgeOutput):
                    raise TypeError(
                        "Structured judge expected _StructuredJudgeOutput responses, "
                        f"got {type(resp).__name__}."
                    )
                results.append(
                    JudgeResult(
                        unsafe=bool(resp.unsafe),
                        confidence=float(resp.confidence),
                        rationale=str(resp.rationale),
                        compliance=bool(resp.compliance),
                    )
                )

        return results


def judge_texts(
    pairs: list[tuple[str, str]],
    judge: SafetyJudge,
    cache: JudgeCache | None = None,
) -> list[JudgeResult]:
    """Judge (prompt, completion) pairs with optional caching.

    Args:
        pairs:  List of (prompt, completion) to judge.
        judge:  SafetyJudge instance to use for uncached queries.
        cache:  Optional JudgeCache. Saves after querying uncached items.

    Returns:
        List of JudgeResult, one per input pair, in the same order.
    """
    if cache is None:
        return judge.judge_batch(pairs)

    results: dict[int, JudgeResult] = {}
    to_query: list[tuple[int, tuple[str, str]]] = []

    for idx, pair in enumerate(pairs):
        key = judge_cache_key(*pair)
        cached = cache.get(key)
        if cached is not None:
            results[idx] = JudgeResult(
                unsafe=bool(cached.get("unsafe", False)),
                confidence=float(cached.get("confidence", 0.5)),
                rationale=str(cached.get("rationale", "cache")),
                compliance=bool(cached.get("compliance", cached.get("unsafe", False))),
            )
        else:
            to_query.append((idx, pair))

    if to_query:
        fresh = judge.judge_batch([pair for _, pair in to_query])
        for (idx, pair), result in zip(to_query, fresh, strict=False):
            key = judge_cache_key(*pair)
            cache.set(
                key,
                {
                    "unsafe": bool(result.unsafe),
                    "confidence": float(result.confidence),
                    "rationale": result.rationale,
                    "compliance": bool(result.compliance),
                },
            )
            results[idx] = result
        cache.save()

    return [results[i] for i in range(len(pairs))]


def judge_prompt_generations(
    prompts: Sequence[str],
    generations: Sequence[str | None] | None,
    judge: SafetyJudge,
    cache: JudgeCache | None = None,
) -> list[JudgeResult]:
    """Judge aligned (prompt, generation) entries from saved activations metadata."""
    if generations is None:
        raise ValueError(
            "Activation metadata is missing generated_texts; "
            "extract with generation enabled to judge prompt+generation pairs."
        )

    prompt_list = list(prompts)
    generation_list = list(generations)
    if len(prompt_list) != len(generation_list):
        raise ValueError(
            "Length mismatch between prompts and generated_texts: "
            f"{len(prompt_list)} vs {len(generation_list)}."
        )

    missing_idxs = [i for i, g in enumerate(generation_list) if g is None]
    if missing_idxs:
        sample = ", ".join(str(i) for i in missing_idxs[:5])
        suffix = "" if len(missing_idxs) <= 5 else ", ..."
        raise ValueError(
            "generated_texts contains null entries, cannot form prompt+generation pairs. "
            f"Missing indices (first few): {sample}{suffix}"
        )

    pairs = [
        (prompt, generation if generation is not None else "")
        for prompt, generation in zip(prompt_list, generation_list, strict=False)
    ]
    return judge_texts(pairs, judge, cache)


def judge_activation_metadata(
    metadata: Mapping[str, Any],
    judge: SafetyJudge,
    cache: JudgeCache | None = None,
) -> list[JudgeResult]:
    """Judge prompt+generation pairs from an activations metadata payload."""
    prompts_raw = metadata.get("texts")
    if not isinstance(prompts_raw, list):
        raise ValueError("Activation metadata must contain a 'texts' list.")

    generations_raw = metadata.get("generated_texts")
    if generations_raw is not None and not isinstance(generations_raw, list):
        raise ValueError(
            "Activation metadata field 'generated_texts' must be a list or null."
        )

    prompts = [str(x) for x in prompts_raw]
    generations = (
        [None if x is None else str(x) for x in generations_raw]
        if generations_raw is not None
        else None
    )
    return judge_prompt_generations(prompts, generations, judge, cache)


# ---------------------------------------------------------------------------
# Private helpers (used internally and by callers that need them)
# ---------------------------------------------------------------------------


def _chunk_starts(n_items: int, chunk_size: int) -> list[int]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    return list(range(0, n_items, chunk_size))


def _iter_with_progress(starts: list[int], desc: str, enabled: bool):
    if not enabled:
        return starts
    try:
        from tqdm.auto import tqdm
    except Exception:
        return starts
    return tqdm(starts, desc=desc, leave=False, dynamic_ncols=True)


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


def _run_smoke_test(args: argparse.Namespace) -> None:
    local_path: Path | None = None if args.from_hub else args.activations
    _, texts, _labels, _token_texts, generated_texts, cfg, leaf = (
        load_activation_bundle(
            local_path=local_path,
            hf_repo_id=args.from_hub,
            dataset_key=args.hub_dataset,
            model_key=args.hub_model,
            layer_idx=args.hub_layer,
        )
    )
    print(f"Loaded {len(texts)} records from {leaf}")
    print(
        f"Config: model={cfg.model_key} dataset={cfg.dataset_key} layer={cfg.layer_idx} "
        f"use_generate={cfg.use_generate}"
    )

    judge = SafetyJudge(
        model=args.model,
        requests_per_minute=args.requests_per_minute,
        max_concurrency=args.max_concurrency,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )

    n_eval = min(args.limit, len(texts))
    eval_prompts = texts[:n_eval]
    eval_generations = None if generated_texts is None else generated_texts[:n_eval]
    results = judge_prompt_generations(
        eval_prompts,
        eval_generations,
        judge=judge,
    )

    n_unsafe = sum(1 for r in results if r.unsafe)
    print(
        f"Judged {len(results)} items: unsafe={n_unsafe} safe={len(results) - n_unsafe}"
    )
    for i in range(min(args.print_examples, len(results))):
        generation = "" if eval_generations is None else (eval_generations[i] or "")
        generation_snippet = generation.replace("\n", " ")
        print(
            f"[{i}] unsafe={int(results[i].unsafe)} compliance={int(results[i].compliance)} "
            f"confidence={results[i].confidence:.2f} "
            f"prompt='{eval_prompts[i]}'"
            f"generation='{generation_snippet}'"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Small smoke test for judging saved activation metadata."
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
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Judge model name (OpenAI-compatible).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum number of examples to judge in the smoke test.",
    )
    parser.add_argument(
        "--print-examples",
        type=int,
        default=8,
        help="Number of judged examples to print.",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=120,
        help="Judge request rate limit.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max concurrent judge requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for judge requests.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during judging.",
    )
    return parser


if __name__ == "__main__":
    parsed_args = _build_arg_parser().parse_args()
    _run_smoke_test(parsed_args)
