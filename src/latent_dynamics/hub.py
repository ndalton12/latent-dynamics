from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from latent_dynamics.config import RunConfig
from latent_dynamics.utils import (
    METADATA_FILE,
    TRAJECTORIES_FILE,
    build_trajectory_shard_manifest_entries,
    list_trajectory_shards,
    parse_trajectory_tensor_key,
    read_trajectory_shard_manifest,
    trajectory_shard_path,
    trajectory_tensor_key,
)

# Optional: for building HF Dataset with trajectory column
try:
    from datasets import Dataset as HFDataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

def activation_subpath(
    dataset_key: str,
    model_key: str,
    layer_idx: int,
) -> Path:
    """Canonical relative path: ``{dataset}/{model}/layer_{N}``."""
    return Path(dataset_key) / model_key / f"layer_{layer_idx}"


def _build_activation_metadata(
    trajectories_count: int,
    texts: list[str],
    labels: np.ndarray | None,
    token_texts: list[list[str]],
    cfg: RunConfig,
    generated_texts: list[str | None] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "config": asdict(cfg),
        "texts": texts,
        "labels": labels.tolist() if labels is not None else None,
        "token_texts": token_texts,
        "generated_texts": generated_texts,
        "n_trajectories": trajectories_count,
    }
    if extra_metadata is not None:
        overlap = set(metadata).intersection(extra_metadata)
        if overlap:
            overlap_s = ", ".join(sorted(overlap))
            raise ValueError(
                f"extra_metadata contains reserved key(s): {overlap_s}."
            )
        metadata.update(extra_metadata)
    return metadata


def write_activation_metadata(
    output_dir: Path,
    trajectories_count: int,
    texts: list[str],
    labels: np.ndarray | None,
    token_texts: list[list[str]],
    cfg: RunConfig,
    generated_texts: list[str | None] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _build_activation_metadata(
        trajectories_count=trajectories_count,
        texts=texts,
        labels=labels,
        token_texts=token_texts,
        cfg=cfg,
        generated_texts=generated_texts,
        extra_metadata=extra_metadata,
    )
    (output_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    return output_dir


def save_activations_shard(
    output_dir: Path,
    trajectories: list[np.ndarray],
    start_idx: int,
    shard_idx: int,
    manifest_entries: list[dict[str, Any]] | None = None,
) -> tuple[int, Path]:
    """Save one trajectories shard and return (next_start_idx, shard_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    next_idx = int(start_idx)
    for traj in trajectories:
        tensors[trajectory_tensor_key(next_idx)] = traj.astype(np.float32)
        next_idx += 1

    shard_path = trajectory_shard_path(output_dir, shard_idx)
    save_file(tensors, str(shard_path))
    if manifest_entries is not None and trajectories:
        manifest_entries.extend(
            build_trajectory_shard_manifest_entries(
                start_idx=int(start_idx),
                count=len(trajectories),
                shard_file=shard_path.name,
            )
        )
    return next_idx, shard_path


def save_activations(
    output_dir: Path,
    trajectories: list[np.ndarray],
    texts: list[str],
    labels: np.ndarray | None,
    token_texts: list[list[str]],
    cfg: RunConfig,
    generated_texts: list[str | None] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = {
        trajectory_tensor_key(i): traj.astype(np.float32)
        for i, traj in enumerate(trajectories)
    }
    save_file(tensors, str(output_dir / TRAJECTORIES_FILE))

    write_activation_metadata(
        output_dir=output_dir,
        trajectories_count=len(trajectories),
        texts=texts,
        labels=labels,
        token_texts=token_texts,
        cfg=cfg,
        generated_texts=generated_texts,
        extra_metadata=extra_metadata,
    )

    return output_dir


def _resolve_example_indices(
    n_trajectories: int,
    example_indices: list[int] | None,
) -> list[int]:
    if example_indices is None:
        return list(range(n_trajectories))
    resolved: list[int] = []
    for idx in example_indices:
        if idx < 0 or idx >= n_trajectories:
            raise ValueError(
                f"example index out of range ({idx}); expected [0, {n_trajectories - 1}]"
            )
        resolved.append(idx)
    return resolved


def _slice_list(values: list[Any], indices: list[int], field_name: str, n: int) -> list[Any]:
    if len(values) != n:
        raise ValueError(
            f"Metadata field '{field_name}' length mismatch: {len(values)} != {n}."
        )
    return [values[i] for i in indices]


def _load_sharded_trajectories_all(input_dir: Path, n: int) -> list[np.ndarray]:
    shards = list_trajectory_shards(input_dir)
    if not shards:
        raise FileNotFoundError(
            f"Missing {TRAJECTORIES_FILE} and no trajectory shards found in {input_dir}."
        )

    trajectories: list[np.ndarray | None] = [None] * n
    for shard in shards:
        tensors = load_file(str(shard))
        for key, value in tensors.items():
            if not key.startswith("traj_"):
                continue
            idx = parse_trajectory_tensor_key(key)
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"Trajectory index out of range ({idx}) in {shard}; expected [0, {n - 1}]."
                )
            trajectories[idx] = value

    missing = [i for i, traj in enumerate(trajectories) if traj is None]
    if missing:
        preview = ", ".join(str(i) for i in missing[:10])
        suffix = "" if len(missing) <= 10 else ", ..."
        raise ValueError(f"Missing trajectory tensors for indices: {preview}{suffix}")
    return [traj for traj in trajectories if traj is not None]


def _load_sharded_trajectories_with_manifest(
    input_dir: Path,
    n: int,
    requested_indices: list[int],
) -> list[np.ndarray]:
    manifest = read_trajectory_shard_manifest(input_dir)
    if manifest is None:
        all_trajectories = _load_sharded_trajectories_all(input_dir, n=n)
        return [all_trajectories[i] for i in requested_indices]

    invalid_manifest_indices = [idx for idx in manifest if idx < 0 or idx >= n]
    if invalid_manifest_indices:
        preview = ", ".join(str(i) for i in invalid_manifest_indices[:10])
        suffix = "" if len(invalid_manifest_indices) <= 10 else ", ..."
        raise ValueError(
            f"Manifest contains out-of-range indices: {preview}{suffix} (n={n})."
        )

    tasks_by_shard: dict[str, list[tuple[int, str]]] = {}
    for out_pos, idx in enumerate(requested_indices):
        location = manifest.get(idx)
        if location is None:
            raise ValueError(f"Manifest missing example_idx {idx}.")
        shard_file, tensor_key = location
        tasks_by_shard.setdefault(shard_file, []).append((out_pos, tensor_key))

    out: list[np.ndarray | None] = [None] * len(requested_indices)
    for shard_file, shard_tasks in tasks_by_shard.items():
        shard_path = input_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(
                f"Shard referenced by manifest does not exist: {shard_path}"
            )
        tensors = load_file(str(shard_path))
        for out_pos, tensor_key in shard_tasks:
            if tensor_key not in tensors:
                raise ValueError(
                    f"Tensor key '{tensor_key}' missing in shard {shard_path}."
                )
            out[out_pos] = tensors[tensor_key]

    missing = [i for i, traj in enumerate(out) if traj is None]
    if missing:
        preview = ", ".join(str(i) for i in missing[:10])
        suffix = "" if len(missing) <= 10 else ", ..."
        raise ValueError(f"Missing loaded trajectories at output slots: {preview}{suffix}")
    return [traj for traj in out if traj is not None]


def load_activations(
    input_dir: Path,
    example_indices: list[int] | None = None,
) -> tuple[
    list[np.ndarray],
    list[str],
    np.ndarray | None,
    list[list[str]],
    list[str | None] | None,
    RunConfig,
]:
    with open(input_dir / METADATA_FILE) as f:
        metadata = json.load(f)

    cfg_dict = metadata["config"]
    cfg_dict.pop("split", None)

    n = int(metadata["n_trajectories"])
    requested_indices = _resolve_example_indices(n, example_indices)

    single_file = input_dir / TRAJECTORIES_FILE
    if n == 0:
        trajectories = []
    elif single_file.exists():
        tensors = load_file(str(single_file))
        trajectories = [tensors[trajectory_tensor_key(i)] for i in requested_indices]
    else:
        trajectories = _load_sharded_trajectories_with_manifest(
            input_dir=input_dir,
            n=n,
            requested_indices=requested_indices,
        )

    texts = _slice_list(
        values=metadata["texts"],
        indices=requested_indices,
        field_name="texts",
        n=n,
    )
    labels_raw = metadata["labels"]
    if labels_raw is not None:
        labels_full = np.array(labels_raw, dtype=np.int64)
        if labels_full.shape[0] != n:
            raise ValueError(
                f"Metadata field 'labels' length mismatch: {labels_full.shape[0]} != {n}."
            )
        labels = labels_full[requested_indices]
    else:
        labels = None
    token_texts = _slice_list(
        values=metadata["token_texts"],
        indices=requested_indices,
        field_name="token_texts",
        n=n,
    )
    generated_texts_raw = metadata.get("generated_texts")
    generated_texts = (
        None
        if generated_texts_raw is None
        else _slice_list(
            values=generated_texts_raw,
            indices=requested_indices,
            field_name="generated_texts",
            n=n,
        )
    )
    cfg = RunConfig(**cfg_dict)

    return trajectories, texts, labels, token_texts, generated_texts, cfg


def push_to_hub(
    local_dir: Path,
    repo_id: str,
    path_in_repo: str | None = None,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo or "",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def build_trajectory_dataset(
    prompts: list[str],
    trajectories: list[np.ndarray],
    labels: list[int],
    splits: list[str],
    completions: list[str] | None = None,
) -> "HFDataset":
    """Build a HuggingFace Dataset with columns: prompt, completion, trajectory, label, split.

    trajectory is stored as list of lists (each row = one trajectory, list of token vectors).
    """
    if not _HAS_DATASETS:
        raise ImportError("datasets library is required to build trajectory dataset.")
    if completions is None:
        completions = [""] * len(prompts)
    n = len(prompts)
    if n != len(trajectories) or n != len(labels) or n != len(splits) or n != len(completions):
        raise ValueError("Length mismatch among prompts, trajectories, labels, splits, completions.")
    # Store each trajectory as list of lists for Arrow compatibility.
    traj_column = [traj.astype(np.float32).tolist() for traj in trajectories]
    return HFDataset.from_dict({
        "prompt": prompts,
        "completion": completions,
        "trajectory": traj_column,
        "label": labels,
        "split": splits,
    })


def push_trajectory_dataset_to_hub(
    prompts: list[str],
    trajectories: list[np.ndarray],
    labels: list[int],
    splits: list[str],
    repo_id: str,
    completions: list[str] | None = None,
    config_name: str | None = None,
) -> str:
    """Build Dataset and push to HuggingFace Hub. Returns dataset URL."""
    ds = build_trajectory_dataset(prompts, trajectories, labels, splits, completions)
    ds.push_to_hub(repo_id, config_name=config_name or "default")
    return f"https://huggingface.co/datasets/{repo_id}"


def pull_from_hub(
    repo_id: str,
    local_dir: Path,
    path_in_repo: str | None = None,
) -> Path:
    from huggingface_hub import HfApi

    api = HfApi()
    if path_in_repo:
        local_dir.mkdir(parents=True, exist_ok=True)
        for info in api.list_repo_tree(
            repo_id, repo_type="dataset", path_in_repo=path_in_repo
        ):
            if hasattr(info, "rfilename"):
                api.hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=info.rfilename,
                    local_dir=str(local_dir),
                )
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
        )
    return local_dir
