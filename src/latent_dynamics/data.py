from __future__ import annotations

from typing import Any, Callable

import numpy as np
from datasets import (
    Dataset,
    load_dataset,
)

from latent_dynamics.config import DATASET_REGISTRY, TOY_CONTRASTIVE, DatasetSpec

STRATIFY_SEED = 42


def _label_from_row(row: dict[str, Any], spec: DatasetSpec) -> int | None:
    if spec.label_field is not None:
        return int(row[spec.label_field])
    if spec.label_fn is not None:
        return int(spec.label_fn(row))
    return None


def _balanced_sample_by_label(
    ds: Dataset,
    spec: DatasetSpec,
    max_samples: int,
    seed: int,
) -> Dataset:
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds

    buckets: dict[int, list[int]] = {}
    for idx, row in enumerate(ds):
        label = _label_from_row(row, spec)
        if label is None:
            continue
        buckets.setdefault(label, []).append(idx)

    if len(buckets) < 2:
        print(
            "Stratified sampling requested but this split has <2 labels; "
            "falling back to first max_samples rows."
        )
        return ds.select(range(max_samples))

    rng = np.random.default_rng(seed)
    labels = sorted(buckets.keys())
    for lab in labels:
        rng.shuffle(buckets[lab])

    per_label = max_samples // len(labels)
    remainder = max_samples % len(labels)
    selected: list[int] = []
    leftovers: list[int] = []
    for i, lab in enumerate(labels):
        target = per_label + (1 if i < remainder else 0)
        take = min(target, len(buckets[lab]))
        selected.extend(buckets[lab][:take])
        leftovers.extend(buckets[lab][take:])

    needed = max_samples - len(selected)
    if needed > 0 and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:needed])

    if len(selected) < max_samples:
        print(
            f"Only {len(selected)} labelled rows available for stratified sampling; "
            f"requested {max_samples}."
        )

    rng.shuffle(selected)
    return ds.select(selected)


def load_examples(
    dataset_key: str,
    max_samples: int | None = None,
    stratify_labels: bool = False,
) -> tuple[Dataset, DatasetSpec]:
    spec = DATASET_REGISTRY[dataset_key]

    if spec.path == "toy_contrastive":
        ds = Dataset.from_list(TOY_CONTRASTIVE)
    elif spec.path == "allenai/wildjailbreak":
        ds = load_dataset(
            spec.path,
            name=spec.split,
            split="train",
            delimiter="\t",
            keep_default_na=False,
        )
    else:
        ds = load_dataset(spec.path, split=spec.split)

    if max_samples and len(ds) > max_samples:
        if stratify_labels:
            ds = _balanced_sample_by_label(
                ds,
                spec,
                max_samples=max_samples,
                seed=STRATIFY_SEED,
            )
        else:
            rng = np.random.default_rng(STRATIFY_SEED)
            indices = rng.choice(len(ds), size=max_samples, replace=False)
            ds = ds.select(indices.tolist())

    return ds, spec


def prepare_text_and_labels(
    ds: Dataset,
    text_field: str,
    label_field: str | None = None,
    label_fn: Callable[[dict[str, Any]], int] | None = None,
) -> tuple[list[str], np.ndarray | None]:
    texts: list[str] = []
    labels: list[int] = []

    for row in ds:
        text = row[text_field]
        if text is None:
            continue
        texts.append(str(text))

        if label_field is not None:
            labels.append(int(row[label_field]))
        elif label_fn is not None:
            labels.append(int(label_fn(row)))

    if len(labels) == 0:
        return texts, None
    return texts, np.array(labels, dtype=np.int64)


if __name__ == "__main__":
    ds, spec = load_examples("wildjailbreak")
    texts, labels = prepare_text_and_labels(
        ds, spec.text_field, spec.label_field, spec.label_fn
    )
    breakpoint()
