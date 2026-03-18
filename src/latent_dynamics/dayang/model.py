from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelSpec:
    path: str
    dtype: str | None = None
    load_model_kwargs: dict[str, Any] | None = None
    load_tokenizer_kwargs: dict[str, Any] | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # Small models for dev purposes
    "gemma3_270m": ModelSpec(
        path="google/gemma-3-270m-it",
        dtype="bfloat16",
    ),
    "qwen3_0.6b": ModelSpec(
        path="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
    ),
    # Large models for final eval
    "qwen3_8b": ModelSpec(
        path="Qwen/Qwen3-8B",
        dtype="bfloat16",
    ),
    "llama3.1_8b": ModelSpec(
        path="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
    ),
    "gemma3_4b": ModelSpec(
        path="google/gemma-3-4b-it",
        dtype="bfloat16",
    ),
}


def get_torch_dtype(dtype: torch.dtype | str | None) -> torch.dtype | str | None:
    """Convert a string or torch.dtype to a torch.dtype."""
    if dtype is None or dtype == "auto":
        return dtype

    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid torch dtype: torch.{dtype}")

    return dtype


def load_model_and_tokenizer(
    model_spec: ModelSpec | str,
    device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from a ModelSpec."""
    if isinstance(model_spec, str):
        model_spec = MODEL_REGISTRY[model_spec]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_spec.path, **(model_spec.load_tokenizer_kwargs or {}))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.path,
        dtype=get_torch_dtype(model_spec.dtype),
        low_cpu_mem_usage=True,
        **(model_spec.load_model_kwargs or {}),
    )
    model.to(device)
    model.eval()

    print(
        f"Loaded model: {model_spec.path}"
        f"\n  Number of hidden layers:         {model.config.num_hidden_layers}"
        f"\n  Size of hidden layers:           {model.config.hidden_size}"
        f"\n  Size of activations (per token): {(model.config.num_hidden_layers + 1) * model.config.hidden_size * 4 // 1024} KB"
        f"\n  Model dtype:                     {model.dtype}"
        f"\n  Device:                          {model.device}"
    )
    return model, tokenizer
