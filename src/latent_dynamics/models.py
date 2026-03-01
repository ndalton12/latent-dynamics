from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_dynamics.config import MODEL_REGISTRY

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_key: str,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    spec = MODEL_REGISTRY[model_key]
    hf_id = spec["hf_id"]
    dtype = _TORCH_DTYPES.get(spec.get("dtype", "bfloat16"), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return model, tokenizer
