from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from latent_dynamics.config import RunConfig

METADATA_FILE = "metadata.json"
TRAJECTORIES_FILE = "trajectories.safetensors"
TRAJECTORY_SHARD_GLOB = "trajectories_shard_*.safetensors"
TRAJECTORY_SHARD_MANIFEST_FILE = "trajectory_shard_manifest.json"


def trajectory_tensor_key(example_idx: int) -> str:
    if example_idx < 0:
        raise ValueError(f"example_idx must be >= 0. Got {example_idx}.")
    return f"traj_{example_idx:04d}"


def parse_trajectory_tensor_key(key: str) -> int:
    if not key.startswith("traj_"):
        raise ValueError(f"Invalid trajectory key prefix for '{key}'.")
    suffix = key.split("_", maxsplit=1)[1]
    try:
        idx = int(suffix)
    except ValueError as e:
        raise ValueError(f"Invalid trajectory key '{key}'.") from e
    if idx < 0:
        raise ValueError(f"Trajectory index must be >= 0 in key '{key}'.")
    return idx


def trajectory_shard_filename(shard_idx: int) -> str:
    if shard_idx < 0:
        raise ValueError(f"shard_idx must be >= 0. Got {shard_idx}.")
    return f"trajectories_shard_{shard_idx:06d}.safetensors"


def trajectory_shard_path(output_dir: Path, shard_idx: int) -> Path:
    return output_dir / trajectory_shard_filename(shard_idx)


def list_trajectory_shards(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob(TRAJECTORY_SHARD_GLOB))


def build_trajectory_shard_manifest_entries(
    start_idx: int,
    count: int,
    shard_file: str | Path,
) -> list[dict[str, Any]]:
    shard_name = shard_file.name if isinstance(shard_file, Path) else shard_file
    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0. Got {start_idx}.")
    if count < 0:
        raise ValueError(f"count must be >= 0. Got {count}.")
    return [
        {
            "example_idx": i,
            "shard_file": shard_name,
            "tensor_key": trajectory_tensor_key(i),
        }
        for i in range(start_idx, start_idx + count)
    ]


def write_trajectory_shard_manifest(
    output_dir: Path,
    entries: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "entries": sorted(entries, key=lambda x: int(x["example_idx"])),
    }
    manifest_path = output_dir / TRAJECTORY_SHARD_MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def read_trajectory_shard_manifest(
    input_dir: Path,
) -> dict[int, tuple[str, str]] | None:
    manifest_path = input_dir / TRAJECTORY_SHARD_MANIFEST_FILE
    if not manifest_path.exists():
        return None

    payload = json.loads(manifest_path.read_text())
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Malformed manifest in {manifest_path}: missing entries list.")

    mapping: dict[int, tuple[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed manifest entry in {manifest_path}: {entry!r}")

        example_idx = entry.get("example_idx")
        shard_file = entry.get("shard_file")
        tensor_key = entry.get("tensor_key")
        if not isinstance(example_idx, int):
            raise ValueError(f"Invalid example_idx in manifest entry: {entry!r}")
        if not isinstance(shard_file, str):
            raise ValueError(f"Invalid shard_file in manifest entry: {entry!r}")
        if not isinstance(tensor_key, str):
            raise ValueError(f"Invalid tensor_key in manifest entry: {entry!r}")
        if example_idx in mapping:
            raise ValueError(
                f"Duplicate example_idx {example_idx} in manifest {manifest_path}."
            )
        mapping[example_idx] = (shard_file, tensor_key)
    return mapping


def is_activation_leaf(path: Path) -> bool:
    has_metadata = (path / METADATA_FILE).exists()
    has_single_file = (path / TRAJECTORIES_FILE).exists()
    has_shards = len(list_trajectory_shards(path)) > 0
    return has_metadata and (has_single_file or has_shards)


def resolve_activation_leaf(root_or_leaf: Path) -> Path:
    path = root_or_leaf.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Activation path does not exist: {path}")

    if is_activation_leaf(path):
        return path

    candidates: list[Path] = []
    for metadata in sorted(path.rglob(METADATA_FILE)):
        leaf = metadata.parent
        if is_activation_leaf(leaf):
            candidates.append(leaf)

    if not candidates:
        raise FileNotFoundError(
            f"No activation leaf found under {path} "
            "(expected metadata.json + trajectories.safetensors or shard files)."
        )
    return candidates[0]


def load_activation_bundle(
    local_path: Path | str | None = None,
    hf_repo_id: str | None = None,
    dataset_key: str | None = None,
    model_key: str | None = None,
    layer_idx: int | None = None,
    cache_dir: Path = Path(".cache/hub"),
) -> tuple[
    list[np.ndarray],
    list[str],
    np.ndarray | None,
    list[list[str]],
    list[str | None] | None,
    RunConfig,
    Path,
]:
    """Load activations either from local disk or a Hugging Face dataset repo."""
    from latent_dynamics.hub import activation_subpath, load_activations, pull_from_hub

    if hf_repo_id:
        for name, value in (
            ("dataset_key", dataset_key),
            ("model_key", model_key),
            ("layer_idx", layer_idx),
        ):
            if value is None:
                raise ValueError(f"{name} is required when loading from Hugging Face.")

        subpath = activation_subpath(
            dataset_key=dataset_key,
            model_key=model_key,
            layer_idx=int(layer_idx),
        )
        leaf = (cache_dir / hf_repo_id.replace("/", "__") / subpath).resolve()
        if not is_activation_leaf(leaf):
            pull_from_hub(
                repo_id=hf_repo_id,
                local_dir=leaf,
                path_in_repo=str(subpath),
            )
    else:
        root = Path(local_path) if local_path is not None else Path("activations")
        leaf = resolve_activation_leaf(root)

    trajectories, texts, labels, token_texts, generated_texts, cfg = load_activations(
        leaf
    )
    return trajectories, texts, labels, token_texts, generated_texts, cfg, leaf
