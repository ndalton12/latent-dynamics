from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from latent_dynamics.utils import (
    METADATA_FILE,
    TRAJECTORIES_FILE,
    is_activation_leaf,
    list_trajectory_shards,
    parse_trajectory_tensor_key,
)

_LAYER_NAME_PATTERN = re.compile(r"layer_(\d+)$")


@dataclass
class ActivationFeatureBundle:
    features_by_layer: dict[int, np.ndarray]
    labels: np.ndarray
    label_field: str
    layers: list[int]
    root: Path
    layer_dirs: dict[int, Path]
    prompt_groups: np.ndarray | None
    example_families: np.ndarray | None
    prompt_subset: str
    example_indices: np.ndarray
    n_total_examples: int


def _parse_layer_idx(name: str) -> int | None:
    match = _LAYER_NAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_layer_dirs(root_or_leaf: Path) -> dict[int, Path]:
    root = root_or_leaf.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Activation path does not exist: {root}")

    if is_activation_leaf(root):
        parsed = _parse_layer_idx(root.name)
        if parsed is None:
            return {0: root}
        return {parsed: root}

    layer_dirs: dict[int, Path] = {}
    for child in sorted(root.glob("layer_*")):
        if not is_activation_leaf(child):
            continue
        idx = _parse_layer_idx(child.name)
        if idx is None:
            continue
        layer_dirs[idx] = child

    if layer_dirs:
        return layer_dirs

    for metadata in sorted(root.rglob(METADATA_FILE)):
        leaf = metadata.parent
        if not is_activation_leaf(leaf):
            continue
        idx = _parse_layer_idx(leaf.name)
        if idx is None:
            continue
        layer_dirs[idx] = leaf

    if not layer_dirs:
        raise FileNotFoundError(
            f"No activation layers found under {root}. "
            "Expected directories like layer_*/metadata.json with trajectory files."
        )
    return layer_dirs


def _pool_trajectory(traj: np.ndarray, pooling: str) -> np.ndarray:
    if traj.ndim != 2:
        raise ValueError(
            f"Expected trajectory shape (T, D), got {traj.shape}."
        )
    if pooling == "last":
        return traj[-1].astype(np.float32)
    if pooling == "mean":
        return traj.mean(axis=0).astype(np.float32)
    if pooling == "max_norm":
        norms = np.linalg.norm(traj, axis=1)
        return traj[int(np.argmax(norms))].astype(np.float32)
    raise ValueError(f"Unknown pooling mode '{pooling}'.")


def _resolve_label_field(
    metadata: dict[str, Any],
    requested_label_field: str,
) -> str:
    if requested_label_field in metadata:
        return requested_label_field

    if requested_label_field != "auto":
        fallbacks = [
            "judge_unsafe_labels",
            "labels",
            "judge_compliance_labels",
        ]
        for field in fallbacks:
            if field in metadata:
                return field
        raise KeyError(
            f"Could not find label field '{requested_label_field}' in metadata "
            f"and no fallback fields were found."
        )

    for field in ("judge_unsafe_labels", "labels", "judge_compliance_labels"):
        if field in metadata:
            return field
    raise KeyError(
        "Could not find any known label field in metadata. "
        "Expected one of: judge_unsafe_labels, labels, judge_compliance_labels."
    )


def _bucket_prompt_group(example_row: Any) -> str:
    if not isinstance(example_row, dict):
        return "other"
    data_type = str(example_row.get("data_type", "")).strip().lower()
    if data_type.startswith("vanilla"):
        return "vanilla"
    if data_type.startswith("adversarial"):
        return "adversarial"
    return "other"


def _extract_prompt_groups(
    metadata: dict[str, Any],
    selected_indices: np.ndarray,
) -> np.ndarray | None:
    example_metadata = metadata.get("example_metadata")
    if not isinstance(example_metadata, list):
        return None
    if not example_metadata:
        return None
    return np.array(
        [_bucket_prompt_group(example_metadata[int(i)]) for i in selected_indices],
        dtype=object,
    )


def _extract_example_families(
    metadata: dict[str, Any],
    selected_indices: np.ndarray,
) -> np.ndarray | None:
    example_metadata = metadata.get("example_metadata")
    if not isinstance(example_metadata, list):
        return None
    if not example_metadata:
        return None

    families: list[str] = []
    for idx in selected_indices:
        row = example_metadata[int(idx)]
        if not isinstance(row, dict):
            return None
        data_type = str(row.get("data_type", "")).strip().lower()
        if not data_type:
            return None
        families.append(data_type)
    return np.array(families, dtype=object)


def _resolve_selected_indices(
    metadata: dict[str, Any],
    n_total: int,
    prompt_subset: str,
    max_examples: int | None,
) -> np.ndarray:
    if prompt_subset not in {"all", "vanilla", "adversarial"}:
        raise ValueError(
            f"Unknown prompt_subset '{prompt_subset}'. "
            "Expected one of: all, vanilla, adversarial."
        )

    if max_examples is not None and max_examples <= 0:
        raise ValueError(f"max_examples must be > 0. Got {max_examples}.")

    selected_indices = np.arange(n_total, dtype=np.int64)
    if prompt_subset != "all":
        example_metadata = metadata.get("example_metadata")
        if not isinstance(example_metadata, list) or not example_metadata:
            raise ValueError(
                "prompt_subset filtering requires non-empty metadata['example_metadata']."
            )
        if len(example_metadata) != n_total:
            raise ValueError(
                "example_metadata length does not match n_trajectories. "
                f"Got len(example_metadata)={len(example_metadata)} n_trajectories={n_total}."
            )
        groups = np.array(
            [_bucket_prompt_group(example_metadata[i]) for i in range(n_total)],
            dtype=object,
        )
        selected_indices = np.where(groups == prompt_subset)[0].astype(np.int64)
        if selected_indices.size == 0:
            raise ValueError(
                f"No examples found for prompt_subset='{prompt_subset}'."
            )

    if max_examples is not None:
        selected_indices = selected_indices[: int(max_examples)]
        if selected_indices.size == 0:
            raise ValueError(
                "No examples selected after applying max_examples."
            )

    return selected_indices


def _load_layer_features(
    layer_dir: Path,
    n_total: int,
    pooling: str,
    selected_indices: np.ndarray,
) -> np.ndarray:
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    n_selected = int(selected_indices.shape[0])

    global_to_local = np.full(n_total, -1, dtype=np.int64)
    global_to_local[selected_indices] = np.arange(n_selected, dtype=np.int64)

    seen = np.zeros(n_selected, dtype=bool)
    features: np.ndarray | None = None

    shard_paths = list_trajectory_shards(layer_dir)
    if not shard_paths:
        single_file = layer_dir / TRAJECTORIES_FILE
        if not single_file.exists():
            raise FileNotFoundError(
                f"No trajectory shards or {TRAJECTORIES_FILE} found in {layer_dir}"
            )
        shard_paths = [single_file]

    for shard_path in shard_paths:
        tensors = load_file(str(shard_path))
        for key, value in tensors.items():
            if not key.startswith("traj_"):
                continue
            global_idx = parse_trajectory_tensor_key(key)
            if global_idx < 0 or global_idx >= n_total:
                continue
            local_idx = int(global_to_local[global_idx])
            if local_idx < 0:
                continue
            pooled = _pool_trajectory(value, pooling=pooling)
            if features is None:
                features = np.zeros((n_selected, pooled.shape[0]), dtype=np.float32)
            features[local_idx] = pooled
            seen[local_idx] = True

    if features is None:
        raise ValueError(
            f"No trajectories were loaded for requested subset in {layer_dir}."
        )

    missing = np.where(~seen)[0]
    if missing.size > 0:
        preview = ", ".join(str(int(x)) for x in missing[:10])
        suffix = "" if missing.size <= 10 else ", ..."
        raise ValueError(
            f"Missing pooled features for {missing.size} selected examples in {layer_dir}. "
            f"Missing local indices: {preview}{suffix}"
        )

    return features


def load_activation_feature_bundle(
    root_or_leaf: Path,
    layers: list[int] | None = None,
    pooling: str = "last",
    label_field: str = "judge_unsafe_labels",
    prompt_subset: str = "all",
    max_examples: int | None = None,
) -> ActivationFeatureBundle:
    layer_dirs_all = _resolve_layer_dirs(root_or_leaf)
    available_layers = sorted(layer_dirs_all.keys())
    if not available_layers:
        raise ValueError(f"No usable layers found under {root_or_leaf}.")

    if layers is None or len(layers) == 0:
        use_layers = available_layers
    else:
        missing = sorted(set(layers).difference(layer_dirs_all))
        if missing:
            raise ValueError(
                f"Requested layer(s) not found: {missing}. "
                f"Available layers: {available_layers}"
            )
        use_layers = sorted(layers)
    layer_dirs = {li: layer_dirs_all[li] for li in use_layers}

    canonical_layer = use_layers[0]
    canonical_metadata_path = layer_dirs[canonical_layer] / METADATA_FILE
    metadata = json.loads(canonical_metadata_path.read_text())

    n_total = int(metadata["n_trajectories"])
    selected_indices = _resolve_selected_indices(
        metadata=metadata,
        n_total=n_total,
        prompt_subset=prompt_subset,
        max_examples=max_examples,
    )

    resolved_label_field = _resolve_label_field(metadata, requested_label_field=label_field)
    labels_full = np.asarray(metadata[resolved_label_field], dtype=np.int64)
    if labels_full.shape[0] != n_total:
        raise ValueError(
            f"Label field '{resolved_label_field}' has length {labels_full.shape[0]}, "
            f"expected {n_total}."
        )
    labels = labels_full[selected_indices]

    prompt_groups = _extract_prompt_groups(metadata, selected_indices=selected_indices)
    example_families = _extract_example_families(
        metadata=metadata,
        selected_indices=selected_indices,
    )

    features_by_layer: dict[int, np.ndarray] = {}
    for layer_idx, layer_dir in layer_dirs.items():
        features_by_layer[layer_idx] = _load_layer_features(
            layer_dir=layer_dir,
            n_total=n_total,
            pooling=pooling,
            selected_indices=selected_indices,
        )

    return ActivationFeatureBundle(
        features_by_layer=features_by_layer,
        labels=labels,
        label_field=resolved_label_field,
        layers=use_layers,
        root=root_or_leaf.expanduser().resolve(),
        layer_dirs=layer_dirs,
        prompt_groups=prompt_groups,
        example_families=example_families,
        prompt_subset=prompt_subset,
        example_indices=selected_indices,
        n_total_examples=n_total,
    )
