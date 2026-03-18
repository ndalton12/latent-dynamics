from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from natsort import natsort_keygen
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PoolMethod = Literal["all", "first", "mid", "last", "mean"] | slice

SPECIAL_TOKENS = {
    # Gemma3
    "<bos>",
    "<eos>",
    "<pad>",
    "<start_of_turn>",
    "<end_of_turn>",
    # Qwen3
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    # Llama3.1
    "<|begin_of_text|>",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
}


def _pool(
    activations: np.ndarray,
    tokens: list[str],
    pool_method: PoolMethod,
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
) -> tuple[np.ndarray, list[str]]:
    token_positions = list(range(len(tokens)))

    # Exclude BOS and/or special tokens
    if exclude_bos:
        activations = activations[1:]
        tokens = tokens[1:]
        token_positions = token_positions[1:]
    if exclude_special_tokens:
        exclude_special_tokens = SPECIAL_TOKENS if exclude_special_tokens is True else exclude_special_tokens
        mask = [t not in exclude_special_tokens for t in tokens]
        activations = activations[mask]
        tokens = [t for t, m in zip(tokens, mask) if m]
        token_positions = [p for p, m in zip(token_positions, mask) if m]

    # Convert pooling method to slices if possible
    if pool_method == "first":
        pool_method = slice(0, 1)
    elif pool_method == "mid":
        mid_idx = len(activations) // 2
        pool_method = slice(mid_idx, mid_idx + 1)
    elif pool_method == "last":
        pool_method = slice(-1, None)

    # Pool activations
    if isinstance(pool_method, slice):
        activations = activations[pool_method]
        tokens = tokens[pool_method]
        token_positions = token_positions[pool_method]
    elif pool_method == "mean":
        activations = activations.mean(axis=0, keepdims=True)
        tokens = ["mean"]
        token_positions = [None]
    elif pool_method == "all":
        pass
    else:
        raise ValueError(f"Unknown pool_method: '{pool_method}'")

    return activations, tokens, token_positions


@dataclass
class Activations:
    metadata: pd.DataFrame
    activations: dict[int, dict[Any, np.ndarray]]

    @property
    def num_samples(self) -> int:
        return len(self.metadata)

    @property
    def num_layers(self) -> int:
        return len(self.activations)

    @property
    def layers(self) -> list[int]:
        return sorted(list(self.activations.keys()))

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save metadata
        self.metadata.to_json(path / "metadata.json", orient="index", indent=2)
        # Save activations
        for layer_idx, acts in tqdm(self.activations.items(), desc="Saving activations"):
            np.savez_compressed(path / f"layer_{layer_idx}.npz", **{str(k): v for k, v in acts.items()})

    @classmethod
    def load(cls, path: str | Path) -> "Activations":
        # Load metadata
        path = Path(path)
        metadata = pd.read_json(path / "metadata.json", orient="index")
        metadata.index.name = "id"
        # Load activations
        activations = {}
        for file in path.glob("layer_*.npz"):
            layer_idx = int(file.stem.split("_")[1])
            activations[layer_idx] = np.load(file)
        return cls(metadata=metadata, activations=activations)

    def get(
        self,
        sample_id: str = None,
        layer_idx: int | None = None,
        pool_method: str | slice = "all",
        exclude_bos: bool = True,
        exclude_special_tokens: bool | list[str] = True,
    ) -> Any:
        """
        If `sample_id` and `layer_idx` are provided, returns a pooled dictionary for that sample.
        If only `layer_idx` is provided, returns a list of pooled dictionaries for all samples in that layer.
        """
        if sample_id is not None and layer_idx is not None:
            metadata = self.metadata.loc[sample_id].to_dict()
            activations = self.activations[layer_idx][sample_id]
            activations_pooled, tokens_pooled, token_positions = _pool(
                activations,
                metadata["tokens"],
                pool_method,
                exclude_bos,
                exclude_special_tokens,
            )
            return {
                "id": sample_id,
                **metadata,
                "activations": activations_pooled,
                "tokens": tokens_pooled,
                "tokens_all": metadata["tokens"],
                "token_positions": token_positions,
            }
        elif layer_idx is not None:
            results = []
            for sample_id in self.metadata.index:
                results.append(
                    self.get(
                        sample_id=sample_id,
                        layer_idx=layer_idx,
                        pool_method=pool_method,
                        exclude_bos=exclude_bos,
                        exclude_special_tokens=exclude_special_tokens,
                    )
                )
            return results
        else:
            raise ValueError("Must provide at least layer_idx")

    def select(self, sample_ids: list[str]) -> "Activations":
        """Return a new Activations object containing only the specified samples.

        This method only creates a subset of the metadata, while the activations dictionary is kept as a reference
        to save memory."""
        return Activations(metadata=self.metadata.loc[sample_ids], activations=self.activations)


def _prepare_sample(
    sample,
    tokenizer: AutoTokenizer,
    include_response: bool | str = False,
    apply_chat_template: bool = False,
) -> dict[str, str]:
    # Create conversation messages
    messages = [{"role": "user", "content": [{"type": "text", "text": sample["prompt"]}]}]
    if include_response:
        response = sample["response"] if isinstance(include_response, bool) else include_response
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    # Format messages into input string
    if apply_chat_template:
        input = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=not include_response,
            continue_final_message=include_response,
            tokenize=False,
        )
    else:
        input = "\n\n".join(message["content"][0]["text"] for message in messages)
    return {"input": input}


def _prepare_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    include_response: bool | str = False,
    apply_chat_template: bool = False,
) -> Dataset:
    """Prepare dataset by creating `input` column based on prompt and responses."""
    dataset = dataset.map(
        _prepare_sample,
        fn_kwargs={
            "tokenizer": tokenizer,
            "include_response": include_response,
            "apply_chat_template": apply_chat_template,
        },
        remove_columns=["prompt"] + (["response"] if include_response is True else []),
        desc="Preparing dataset",
    )
    return dataset


def _create_dataloader(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create dataloader from the prepared dataset."""

    def collate_fn(samples):
        inputs = tokenizer(
            [sample["input"] for sample in samples],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        return {
            "ids": [sample["id"] for sample in samples],
            "is_safe": [sample["is_safe"] for sample in samples],
            "is_adversarial": [sample["is_adversarial"] for sample in samples],
            **inputs,
        }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )
    return dataloader


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    include_response: bool | str = False,
    apply_chat_template: bool = False,
    layers: list[int] | None = None,
    batch_size: int = 8,
) -> Activations:
    """Extract activations from the model for the given dataset."""
    if layers is None:
        layers = list(range(model.config.num_hidden_layers + 1))

    # Prepare dataset
    dataset = _prepare_dataset(
        dataset,
        tokenizer,
        include_response=include_response,
        apply_chat_template=apply_chat_template,
    )

    # Create dataloader
    dataloader = _create_dataloader(
        dataset,
        tokenizer,
        batch_size=batch_size,
        add_special_tokens=not apply_chat_template,  # avoid adding second BOS token when applying chat template
    )

    # Extract activations
    metadata = []
    activations = {layer_idx: {} for layer_idx in layers}
    for batch in tqdm(dataloader, desc="Extracting activations"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract activations for each sample
        for i, sample_id in enumerate(batch["ids"]):
            # Get mask to remove padding
            mask = attention_mask[i].bool()
            # Store metadata
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i, mask])
            text = tokenizer.convert_tokens_to_string(tokens)
            metadata.append(
                {
                    "id": sample_id,
                    "text": text,
                    "tokens": tokens,
                    "is_safe": batch["is_safe"][i],
                    "is_adversarial": batch["is_adversarial"][i],
                }
            )
            # Store activations
            for layer_idx in layers:
                acts = outputs.hidden_states[layer_idx][i, mask]  # shape: (num_valid_tokens, hidden_size)
                activations[layer_idx][sample_id] = acts.float().cpu().numpy()

    num_samples = len(metadata)
    num_tokens = sum(len(meta["tokens"]) for meta in metadata)
    num_layers = len(activations)
    hidden_size = acts.shape[1:]
    num_bytes = sum(acts.nbytes for acts_per_layer in activations.values() for acts in acts_per_layer.values())
    print(
        f"Extracted activations for {num_samples} samples"
        f"\n  Number of tokens:     {num_tokens}"
        f"\n  Number of layers:     {num_layers}"
        f"\n  Shape of activations: {hidden_size}"
        f"\n  Total size:           {num_bytes / 1024 / 1024:.1f} MB"
    )

    metadata = pd.DataFrame(metadata).set_index("id")
    metadata.sort_index(inplace=True, key=natsort_keygen())
    return Activations(metadata=metadata, activations=activations)
