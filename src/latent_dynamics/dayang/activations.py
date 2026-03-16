from __future__ import annotations

from typing import Any

import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    # include_response: bool = False,
    apply_chat_template: bool = True,
) -> Dataset:
    """Prepare prompts and completions in the `input` column."""

    if apply_chat_template:
        # Apply chat template
        def apply_chat_template(sample):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": sample["prompt"]}]}],
                add_generation_prompt=True,
                continue_final_message=False,
                tokenize=False,
            )
            # if "response" in sample:
            #     messages = [
            #         {"role": "user", "content": [{"type": "text", "text": sample["prompt"]}]},
            #         {"role": "assistant", "content": [{"type": "text", "text": sample["response"]}]},
            #     ]
            #     response = tokenizer.apply_chat_template(
            #         messages,
            #         add_generation_prompt=not include_response,
            #         continue_final_message=include_response,
            #         tokenize=False,
            #     )[len(prompt) :]
            return {
                "input": prompt,
                # "response": response if include_response else None,
            }

        dataset = dataset.map(
            apply_chat_template,
            desc="Applying chat template",
            remove_columns=["prompt"],
        )
    else:
        dataset = dataset.rename_column("prompt", "input")

    return dataset


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    layers: list[int] | None = None,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Extract activations from the model for the given dataset."""

    # Create dataloader
    def collate_fn(samples):
        input = tokenizer(
            [sample["input"] for sample in samples],
            padding=True,
            return_tensors="pt",
            # add_special_tokens=False,
        )
        return {
            "ids": [sample["id"] for sample in samples],
            "is_safe": [sample["is_safe"] for sample in samples],
            "is_adversarial": [sample["is_adversarial"] for sample in samples],
            **input,
        }

    dataset = prepare_dataset(dataset, tokenizer, apply_chat_template=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Extract activations
    activations_all = []
    for batch in tqdm(dataloader, desc="Extracting activations"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Extract hidden states for specified layers
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if layers is None:
            hidden_states = outputs.hidden_states
        else:
            hidden_states = [outputs.hidden_states[layer] for layer in layers]
        hidden_states = torch.stack(hidden_states, dim=2)  # shape: (batch_size, num_tokens, num_layers, hidden_size)

        # Save activations for each sample without padding
        for i, id in enumerate(batch["ids"]):
            mask = attention_mask[i].bool()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i, mask])
            text = tokenizer.convert_tokens_to_string(tokens)
            activations = hidden_states[i, mask]  # shape: (num_valid_tokens, num_layers, hidden_size)
            activations_all.append(
                {
                    "id": id,
                    "text": text,
                    "tokens": tokens,
                    "activations": activations.float().cpu().numpy(),
                    "is_safe": batch["is_safe"][i],
                    "is_adversarial": batch["is_adversarial"][i],
                }
            )

    print(
        f"Extracted activations for {len(activations_all)} samples"
        f"\n  Number of tokens:     {sum(result['activations'].shape[0] for result in activations_all)}"
        f"\n  Shape of activations: {activations_all[0]['activations'].shape[1:]}"
        f"\n  Total size:           {sum(result['activations'].nbytes for result in activations_all) / 1024 / 1024:.1f} MB"
    )

    return activations_all


def pool_activations(activations: np.ndarray, pool_method: str | slice, include_bos: bool) -> np.ndarray:
    """Helper to pool activations per sample."""
    if not include_bos:
        activations = activations[1:]

    if isinstance(pool_method, slice):
        return activations[pool_method]
    elif pool_method == "first":
        return activations[:1]
    elif pool_method == "mid":
        mid_idx = len(activations) // 2
        return activations[mid_idx : mid_idx + 1]
    elif pool_method == "last":
        return activations[-1:]
    elif pool_method == "mean":
        return activations.mean(axis=0, keepdims=True)
    elif pool_method == "all":
        return activations
    else:
        raise ValueError(f"Unknown pool_method: '{pool_method}'")


def pool_tokens(tokens: list[str], pool_method: str | slice, include_bos: bool) -> np.ndarray:
    """Helper to pool tokens per sample."""
    if not include_bos:
        tokens = tokens[1:]

    if isinstance(pool_method, slice):
        return tokens[pool_method]
    elif pool_method == "first":
        return tokens[:1]
    elif pool_method == "mid":
        mid_idx = len(tokens) // 2
        return tokens[mid_idx : mid_idx + 1]
    elif pool_method == "last":
        return tokens[-1:]
    elif pool_method == "mean":
        return ["mean"]
    elif pool_method == "all":
        return tokens
    else:
        raise ValueError(f"Unknown pool_method: '{pool_method}'")
