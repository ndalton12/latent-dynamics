#!/usr/bin/env python3
"""Collect token-level hidden-state trajectories and optionally push to HuggingFace.

Run from repo root with project env (e.g. uv):
  uv run python scripts/collect_trajectories.py --model gemma3_4b --dataset toy_contrastive --num_samples 20 --output ./activations

Exact CLI from plan:
  uv run python scripts/collect_trajectories.py --model qwen3_8b --dataset xstest --num_samples 200 --layers 20 --4bit --push_hub alexlyu/llm-traj-safety-qwen3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package is on path when run as script (src layout)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from datasets import Dataset

from latent_dynamics.activations import extract_multi_layer_trajectories
from latent_dynamics.config import (
    DATASET_REGISTRY,
    DEFAULT_LAYERS,
    MODEL_REGISTRY,
    RunConfig,
)
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.hub import (
    activation_subpath,
    push_trajectory_dataset_to_hub,
    save_activations,
)
from latent_dynamics.models import load_model_and_tokenizer, resolve_device


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect hidden-state trajectories and save (safetensors + optional HF dataset).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="gemma3_4b",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model key from registry (qwen3_8b, llama_3_1_8b, gemma3_4b).",
    )
    # Dataset: support xstest, wildjailbreak, harmbench, toy_contrastive
    p.add_argument(
        "--dataset",
        type=str,
        default="toy_contrastive",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset key (xstest, wildjailbreak, harmbench, toy_contrastive).",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Max number of examples to process (for quick test use 20).",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices (e.g. 5 10 20). If not set, use default for model.",
    )
    p.add_argument(
        "--4bit",
        dest="load_4bit",
        action="store_true",
        help="Load model in 4-bit (CUDA only, bitsandbytes).",
    )
    p.add_argument(
        "--push_hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Push trajectory dataset to HuggingFace (e.g. alexlyu/llm-traj-safety-qwen3).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("./activations"),
        help="Root output directory for safetensors + metadata.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token sequence length.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda, mps, cpu).",
    )
    p.add_argument(
        "--use_70_15_15",
        action="store_true",
        help="Use 70/15/15 split and tag each example with train/calib/test.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    model_key = args.model
    dataset_key = args.dataset
    num_samples = args.num_samples
    layer_list = args.layers
    if layer_list is None:
        default_layer = DEFAULT_LAYERS.get(model_key, 5)
        layer_list = [default_layer]

    device = resolve_device(args.device)
    cfg = RunConfig(
        model_key=model_key,
        dataset_key=dataset_key,
        split="train",
        max_samples=num_samples,
        max_length=args.max_length,
        layer_idx=layer_list[0],
        device=device,
    )

    # Load data: single split or 70/15/15
    ds_full, spec = load_examples(dataset_key, split="train", max_samples=None)
    if args.use_70_15_15 and len(ds_full) >= 3:
        from datasets import concatenate_datasets

        train_ds, calib_ds, test_ds = make_70_15_15_split(ds_full, seed=42)
        n_train = min(int(0.7 * num_samples), len(train_ds))
        n_calib = min(max(0, int(0.15 * num_samples)), len(calib_ds))
        n_test = min(max(0, num_samples - n_train - n_calib), len(test_ds))
        n_total = n_train + n_calib + n_test
        if n_total == 0:
            n_train = min(1, len(train_ds))
            n_total = n_train
        train_ds = train_ds.select(range(n_train))
        calib_ds = (
            calib_ds.select(range(n_calib)) if n_calib else calib_ds.select(range(0))
        )
        test_ds = test_ds.select(range(n_test)) if n_test else test_ds.select(range(0))
        splits_used = ["train"] * n_train + ["calib"] * n_calib + ["test"] * n_test
        ds = concatenate_datasets([train_ds, calib_ds, test_ds])
    else:
        ds = ds_full.select(range(min(num_samples, len(ds_full))))
        splits_used = ["train"] * len(ds)

    texts, labels = prepare_text_and_labels(
        ds,
        text_field=spec.text_field,
        label_field=spec.label_field,
        label_fn=spec.label_fn,
    )
    if not texts:
        print("No texts loaded. Check dataset and fields.", file=sys.stderr)
        sys.exit(1)

    n = len(texts)
    if labels is None:
        labels = np.zeros(n, dtype=np.int64)

    print(f"Loading model {model_key} (4bit={args.load_4bit}) on {device}...")
    model, tokenizer = load_model_and_tokenizer(
        model_key, device, load_in_4bit=args.load_4bit
    )
    print(f"Extracting trajectories for {n} examples, layers {layer_list}...")
    per_layer, token_texts = extract_multi_layer_trajectories(
        model, tokenizer, texts, layer_list, cfg.max_length, device, cfg
    )

    # Success message with trajectory shape (first example, first layer)
    first_traj = per_layer[layer_list[0]][0]
    T, hidden_dim = first_traj.shape
    print(f"SUCCESS: trajectory shape ({T}, {hidden_dim})")

    # Save safetensors + metadata per layer
    for li in layer_list:
        sub = activation_subpath(dataset_key, cfg.split, model_key, li)
        out_dir = args.output / sub
        layer_cfg = RunConfig(**{**cfg.__dict__, "layer_idx": li})
        save_activations(
            out_dir,
            per_layer[li],
            texts,
            labels,
            token_texts,
            layer_cfg,
        )
        print(f"  Saved layer {li} -> {out_dir}")

    # Optionally push as HF Dataset (one layer: first in list)
    if args.push_hub:
        primary_layer = layer_list[0]
        trajectories = per_layer[primary_layer]
        completions = [""] * n  # no generation in this script by default
        url = push_trajectory_dataset_to_hub(
            prompts=texts,
            trajectories=trajectories,
            labels=labels.tolist(),
            splits=splits_used[:n],
            repo_id=args.push_hub,
            completions=completions,
            config_name=f"{dataset_key}_{model_key}_layer{primary_layer}",
        )
        print(f"Pushed dataset to {url}")


if __name__ == "__main__":
    main()
