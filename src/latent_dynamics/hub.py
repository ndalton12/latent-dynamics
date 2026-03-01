from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from latent_dynamics.config import RunConfig

TRAJECTORIES_FILE = "trajectories.safetensors"
METADATA_FILE = "metadata.json"


def save_activations(
    output_dir: Path,
    trajectories: list[np.ndarray],
    texts: list[str],
    labels: np.ndarray | None,
    token_texts: list[list[str]],
    cfg: RunConfig,
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
        "n_trajectories": len(trajectories),
    }
    (output_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))

    return output_dir


def load_activations(
    input_dir: Path,
) -> tuple[list[np.ndarray], list[str], np.ndarray | None, list[list[str]], RunConfig]:
    tensors = load_file(str(input_dir / TRAJECTORIES_FILE))

    with open(input_dir / METADATA_FILE) as f:
        metadata = json.load(f)

    n = metadata["n_trajectories"]
    trajectories = [tensors[f"traj_{i:04d}"] for i in range(n)]
    texts = metadata["texts"]
    labels = (
        np.array(metadata["labels"], dtype=np.int64)
        if metadata["labels"] is not None
        else None
    )
    token_texts = metadata["token_texts"]
    cfg = RunConfig(**metadata["config"])

    return trajectories, texts, labels, token_texts, cfg


def push_to_hub(local_dir: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def pull_from_hub(repo_id: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return local_dir
