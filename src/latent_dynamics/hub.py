from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from latent_dynamics.config import RunConfig

# Optional: for building HF Dataset with trajectory column
try:
    from datasets import Dataset as HFDataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

TRAJECTORIES_FILE = "trajectories.safetensors"
METADATA_FILE = "metadata.json"


def activation_subpath(
    dataset_key: str,
    model_key: str,
    layer_idx: int,
) -> Path:
    """Canonical relative path: ``{dataset}/{model}/layer_{N}``."""
    return Path(dataset_key) / model_key / f"layer_{layer_idx}"


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
        f"traj_{i:04d}": traj.astype(np.float32)
        for i, traj in enumerate(trajectories)
    }
    save_file(tensors, str(output_dir / TRAJECTORIES_FILE))

    metadata = {
        "config": asdict(cfg),
        "texts": texts,
        "labels": labels.tolist() if labels is not None else None,
        "token_texts": token_texts,
        "generated_texts": generated_texts,
        "n_trajectories": len(trajectories),
    }
    if extra_metadata is not None:
        overlap = set(metadata).intersection(extra_metadata)
        if overlap:
            overlap_s = ", ".join(sorted(overlap))
            raise ValueError(
                f"extra_metadata contains reserved key(s): {overlap_s}."
            )
        metadata.update(extra_metadata)
    (output_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

    return output_dir


def load_activations(
    input_dir: Path,
) -> tuple[list[np.ndarray], list[str], np.ndarray | None, list[list[str]], list[str | None] | None, RunConfig]:
    tensors = load_file(str(input_dir / TRAJECTORIES_FILE))

    with open(input_dir / METADATA_FILE) as f:
        metadata = json.load(f)

    cfg_dict = metadata["config"]
    cfg_dict.pop("split", None)

    n = metadata["n_trajectories"]
    trajectories = [tensors[f"traj_{i:04d}"] for i in range(n)]
    texts = metadata["texts"]
    labels = (
        np.array(metadata["labels"], dtype=np.int64)
        if metadata["labels"] is not None
        else None
    )
    token_texts = metadata["token_texts"]
    generated_texts = metadata.get("generated_texts")
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
