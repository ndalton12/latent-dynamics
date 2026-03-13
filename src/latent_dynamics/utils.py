from __future__ import annotations

from pathlib import Path

import numpy as np

from latent_dynamics.config import RunConfig
from latent_dynamics.hub import activation_subpath, load_activations, pull_from_hub


def is_activation_leaf(path: Path) -> bool:
    return (path / "metadata.json").exists() and (
        path / "trajectories.safetensors"
    ).exists()


def resolve_activation_leaf(root_or_leaf: Path) -> Path:
    path = root_or_leaf.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Activation path does not exist: {path}")

    if is_activation_leaf(path):
        return path

    candidates: list[Path] = []
    for metadata in sorted(path.rglob("metadata.json")):
        leaf = metadata.parent
        if is_activation_leaf(leaf):
            candidates.append(leaf)

    if not candidates:
        raise FileNotFoundError(
            f"No activation leaf found under {path} (expected metadata.json + trajectories.safetensors)."
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
