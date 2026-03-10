"""LLM-as-judge utilities for safety classification.

Shared between the benchmark suite (generation-trajectory labeling) and the
QD active boundary experiment (candidate labeling). Provides:

  - JudgeResult / JudgeCache  -- dataclass and disk-backed cache
  - SafetyJudge               -- thin wrapper around flashlite for batched judging
  - judge_texts               -- high-level convenience: judge (prompt, completion) pairs

The judge uses a Jinja template at prompts/judge/safety_judge.jinja and
communicates with an OpenAI-compatible API via flashlite. OPENAI_API_KEY must
be set in the environment.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
            }
            for prompt, completion in pairs
        ]

        results: list[JudgeResult] = []
        starts = _chunk_starts(len(reqs), self.batch_size)
        iterator = _iter_with_progress(
            starts, desc=f"judge ({len(pairs)})", enabled=self.show_progress and len(starts) > 1
        )
        for start in iterator:
            batch = reqs[start : start + self.batch_size]
            resps = _run_async(self.client.complete_many(batch, max_concurrency=self.max_concurrency))
            for resp in resps:
                results.append(_parse_judge_output(_extract_response_text(resp)))

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
            )
        else:
            to_query.append((idx, pair))

    if to_query:
        fresh = judge.judge_batch([pair for _, pair in to_query])
        for (idx, pair), result in zip(to_query, fresh, strict=False):
            key = judge_cache_key(*pair)
            cache.set(key, {
                "unsafe": bool(result.unsafe),
                "confidence": float(result.confidence),
                "rationale": result.rationale,
            })
            results[idx] = result
        cache.save()

    return [results[i] for i in range(len(pairs))]


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
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()

    parsed: dict[str, Any] | None = None
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            parsed = obj
    except json.JSONDecodeError:
        pass

    if parsed is None:
        low = cleaned.lower()
        unsafe = ("unsafe" in low and "true" in low) or ("harmful" in low and "yes" in low)
        return JudgeResult(unsafe=unsafe, confidence=0.5, rationale="heuristic_parse")

    unsafe = bool(parsed.get("unsafe", False))
    raw_conf = parsed.get("confidence", 0.5)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(parsed.get("rationale", ""))
    return JudgeResult(unsafe=unsafe, confidence=confidence, rationale=rationale)
