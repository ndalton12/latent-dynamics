from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_dynamics.config import RunConfig


def generate_full_sequence(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: RunConfig,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    gen_kwargs: dict = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cfg.do_sample:
        gen_kwargs["temperature"] = cfg.temperature
        gen_kwargs["top_p"] = cfg.top_p

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    if cfg.include_prompt_in_trajectory:
        return generated

    prompt_len = input_ids.shape[1]
    only_new = generated[:, prompt_len:]
    if only_new.shape[1] == 0:
        return generated[:, -1:]
    return only_new


def extract_hidden_trajectories(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer_idx: int,
    max_length: int,
    device: str,
    cfg: RunConfig,
) -> tuple[list[np.ndarray], list[list[str]]]:
    trajectories: list[np.ndarray] = []
    token_texts: list[list[str]] = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            if cfg.use_generate:
                seq_ids = generate_full_sequence(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cfg=cfg,
                    tokenizer=tokenizer,
                )
                seq_attn = torch.ones_like(seq_ids, device=device)
                out = model(
                    input_ids=seq_ids,
                    attention_mask=seq_attn,
                    output_hidden_states=True,
                )
                hs = out.hidden_states[layer_idx][0]
                ids = seq_ids[0].cpu().tolist()
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hs = out.hidden_states[layer_idx][0]
                attn = attention_mask[0].bool()
                hs = hs[attn]
                ids = input_ids[0][attn].cpu().tolist()

        hs = hs.float().cpu().numpy()
        trajectories.append(hs)
        token_texts.append(tokenizer.convert_ids_to_tokens(ids))

    return trajectories, token_texts


def pool_trajectory(traj: np.ndarray, mode: str = "last") -> np.ndarray:
    if mode == "last":
        return traj[-1]
    if mode == "mean":
        return traj.mean(axis=0)
    if mode == "max_norm":
        return traj[np.linalg.norm(traj, axis=1).argmax()]
    raise ValueError(f"Unknown pooling mode: {mode}")


def build_feature_matrix(
    trajectories: list[np.ndarray],
    pooling: str,
) -> np.ndarray:
    return np.stack([pool_trajectory(t, pooling) for t in trajectories], axis=0)
