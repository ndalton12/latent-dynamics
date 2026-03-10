from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from datasets.exceptions import DatasetGenerationError

from latent_dynamics.config import DATASET_REGISTRY, TOY_CONTRASTIVE, DatasetSpec


def _streaming_to_dataset(
    path: str,
    config_name: str | None,
    split_name: str,
    max_samples: int | None,
) -> Dataset:
    stream = load_dataset(path, config_name, split=split_name, streaming=True)

    rows: list[dict[str, Any]] = []
    for row in stream:
        rows.append(dict(row))
        if max_samples is not None and len(rows) >= max_samples:
            break

    if not rows:
        raise ValueError(
            "Streaming fallback produced zero rows. Check dataset config/split."
        )
    return Dataset.from_list(rows)


def _resolve_config_and_split(path: str, requested_split: str) -> tuple[str, str]:
    config_names = get_dataset_config_names(path)
    if not config_names:
        raise ValueError(f"No dataset configs found for '{path}'.")

    config_name = (
        requested_split if requested_split in config_names else config_names[0]
    )
    split_names = get_dataset_split_names(path, config_name)
    if requested_split in split_names:
        split_name = requested_split
    elif "train" in split_names:
        split_name = "train"
    else:
        split_name = split_names[0]

    if config_name != requested_split:
        print(
            f"Requested split='{requested_split}' is not a config for '{path}'. "
            f"Using config='{config_name}' split='{split_name}'."
        )
    return config_name, split_name


def _non_null_text_count(ds: Dataset, field: str, limit: int = 256) -> int:
    n = min(len(ds), limit)
    count = 0
    for i in range(n):
        val = ds[i].get(field)
        if val is not None and str(val).strip() != "":
            count += 1
    return count


def _resolve_text_field(spec: DatasetSpec, ds: Dataset) -> str:
    if spec.text_field in ds.column_names:
        return spec.text_field

    candidates = [
        "prompt",
        "adversarial",
        "vanilla",
        "text",
        "question",
        "input",
    ]
    best_field: str | None = None
    best_count = -1
    for field in candidates:
        if field not in ds.column_names:
            continue
        nn_count = _non_null_text_count(ds, field)
        if nn_count > best_count:
            best_count = nn_count
            best_field = field

    if best_field is None or best_count <= 0:
        raise ValueError(
            f"Could not resolve text field. Config has '{spec.text_field}', "
            f"dataset columns are {ds.column_names}."
        )
    return best_field


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
    split: str = "train",
    max_samples: int | None = None,
    stratify_labels: bool = False,
    stratify_seed: int = 42,
) -> tuple[Dataset, DatasetSpec]:
    spec = DATASET_REGISTRY[dataset_key]

    if spec.loader == "toy":
        ds = Dataset.from_list(TOY_CONTRASTIVE)
    else:
        try:
            if spec.name is not None:
                ds = load_dataset(spec.path, spec.name, split=split)
            else:
                ds = load_dataset(spec.path, split=split)
        except ValueError as e:
            # Some HF datasets (e.g. wildjailbreak) require a config name and
            # use names like "train"/"eval". Resolve config and split explicitly.
            if spec.name is None and "Config name is missing" in str(e):
                config_name, split_name = _resolve_config_and_split(spec.path, split)
                try:
                    ds = load_dataset(spec.path, config_name, split=split_name)
                except DatasetGenerationError:
                    print(
                        "Falling back to streaming load due dataset generation error "
                        f"(config='{config_name}', split='{split_name}')."
                    )
                    stream_cap = max_samples
                    if stratify_labels and max_samples is not None:
                        stream_cap = max(max_samples * 20, max_samples + 100)
                    ds = _streaming_to_dataset(
                        spec.path,
                        config_name,
                        split_name,
                        stream_cap,
                    )
            else:
                raise
        except DatasetGenerationError:
            print(
                "Falling back to streaming load due dataset generation error "
                f"(split='{split}')."
            )
            stream_cap = max_samples
            if stratify_labels and max_samples is not None:
                stream_cap = max(max_samples * 20, max_samples + 100)
            ds = _streaming_to_dataset(spec.path, spec.name, split, stream_cap)

    effective_text_field = _resolve_text_field(spec, ds)
    if effective_text_field != spec.text_field:
        print(
            f"Text field '{spec.text_field}' not found; using '{effective_text_field}'."
        )
    effective_spec = replace(spec, text_field=effective_text_field)

    if max_samples and len(ds) > max_samples:
        if stratify_labels:
            ds = _balanced_sample_by_label(
                ds,
                effective_spec,
                max_samples=max_samples,
                seed=stratify_seed,
            )
        else:
            ds = ds.select(range(max_samples))

    return ds, effective_spec


def make_70_15_15_split(
    ds: Dataset,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return train, calib, test splits with 70/15/15 proportion."""
    if len(ds) < 3:
        raise ValueError("Dataset too small for 70/15/15 split.")

    # First: 70% train, 30% temp.
    train_test = ds.train_test_split(test_size=0.30, seed=seed)
    train_ds = train_test["train"]
    temp_ds = train_test["test"]

    # Second: split temp into 15% calib, 15% test (equal halves of remaining 30%).
    calib_test = temp_ds.train_test_split(test_size=0.5, seed=seed)
    calib_ds = calib_test["train"]
    test_ds = calib_test["test"]

    return train_ds, calib_ds, test_ds


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


def inspect_schema(ds: Dataset, n: int = 3) -> None:
    print("Rows:", len(ds))
    print("Columns:", ds.column_names)
    for i in range(min(n, len(ds))):
        print(f"[{i}]", {k: ds[i][k] for k in ds.column_names})
