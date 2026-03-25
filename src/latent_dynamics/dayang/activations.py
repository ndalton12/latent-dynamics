from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from natsort import natsort_keygen
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

PoolMethod = Literal["all", "first", "mid", "last", "mean"] | int | slice | list[int]

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
    activations: list[Any] | np.ndarray,
    tokens: list[str] | np.ndarray,
    *arrays: list[Any] | np.ndarray,
    pool_method: PoolMethod = "all",
    exclude_bos: bool = True,
    exclude_special_tokens: bool | list[str] = True,
) -> tuple[np.ndarray, list[str], list[int | None]]:
    activations = np.asarray(activations)
    tokens = np.asarray(tokens)
    token_positions = np.arange(len(tokens))
    arrays = [np.asarray(array) for array in arrays]

    # Exclude BOS and/or special tokens
    if exclude_bos:
        activations = activations[1:]
        tokens = tokens[1:]
        token_positions = token_positions[1:]
        arrays = [array[1:] for array in arrays]
    if exclude_special_tokens:
        exclude_special_tokens = SPECIAL_TOKENS if exclude_special_tokens is True else exclude_special_tokens
        mask = [t not in exclude_special_tokens for t in tokens]

        activations = activations[mask]
        tokens = tokens[mask]
        token_positions = token_positions[mask]
        arrays = [array[mask] for array in arrays]

    # Convert pooling method to slices if possible
    if pool_method == "first":
        pool_method = slice(0, 1)
    elif pool_method == "mid":
        mid_idx = len(activations) // 2
        pool_method = slice(mid_idx, mid_idx + 1)
    elif pool_method == "last":
        pool_method = slice(-1, None)
    elif isinstance(pool_method, int):
        pool_method = [pool_method]

    # Pool activations
    if isinstance(pool_method, (slice | list)):
        activations = activations[pool_method]
        tokens = tokens[pool_method]
        token_positions = token_positions[pool_method]
        arrays = [array[pool_method] for array in arrays]
    elif pool_method == "mean":
        activations = activations.mean(axis=0, keepdims=True)
        tokens = ["mean"]
        token_positions = [None]
        arrays = [[None] for _ in arrays]
    elif pool_method == "all":
        pass
    else:
        raise ValueError(f"Unknown pool_method: '{pool_method}'")

    return activations, tokens.tolist(), token_positions.tolist(), *arrays


def _topk(logits: torch.Tensor, tokenizer: PreTrainedTokenizerBase, k: int = 10) -> TopK:
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_token_ids = probs.topk(k, dim=-1)
    topk_probs = topk_probs.float().cpu().numpy()  # shape: (num_tokens, topk)
    topk_token_ids = topk_token_ids.cpu().numpy()  # shape: (num_tokens, topk)
    topk_tokens = np.array([tokenizer.convert_ids_to_tokens(ids) for ids in topk_token_ids])
    return TopK(tokens=topk_tokens, probs=topk_probs)


@dataclass
class TopK:
    tokens: np.ndarray  # shape: (num_tokens, topk)
    probs: np.ndarray  # shape: (num_tokens, topk)

    def __post_init__(self):
        self.tokens = np.asarray(self.tokens)
        self.probs = np.asarray(self.probs)

        if self.tokens.shape != self.probs.shape:
            raise ValueError(
                f"Tokens and probabilities must have the same shape, got {self.tokens.shape} and {self.probs.shape}."
            )


@dataclass
class Activations:
    samples: pd.DataFrame
    activations: dict[int, dict[str, np.ndarray]]  # layer_idx -> sample_id -> activations
    topk: dict[int, dict[str, TopK]] = field(default_factory=dict)  # layer_idx -> sample_id -> topk

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def num_layers(self) -> int:
        return len(self.activations)

    @property
    def layers(self) -> list[int]:
        return sorted(list(self.activations.keys()))

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save samples
        self.samples.to_json(path / "samples.json", orient="index", indent=2)
        # Save activations
        for layer_idx, acts_per_layer in tqdm(self.activations.items(), desc="Saving activations"):
            np.savez_compressed(path / f"layer_{layer_idx}.npz", **acts_per_layer)
        # Save topk
        pd.DataFrame(self.topk).to_json(path / "topk.json", indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Activations":
        # Load samples
        path = Path(path)
        samples = pd.read_json(path / "samples.json", orient="index")
        samples.index.name = "id"
        # Load activations
        activations = {}
        for file in path.glob("layer_*.npz"):
            layer_idx = int(file.stem.split("_")[1])
            activations[layer_idx] = np.load(file)
        # Load topk
        topk = pd.read_json(path / "topk.json", orient="columns")
        topk = topk.map(lambda x: TopK(**x))
        topk = topk.to_dict()
        return cls(samples=samples, activations=activations, topk=topk)

    def extract_topk(
        self,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        layer_idx: int | None = None,
        k: int = 10,
    ):
        if layer_idx is None:
            layer_idx = self.num_layers - 1
        if layer_idx not in self.activations:
            raise ValueError(f"Layer {layer_idx} not found in activations")
        if model.lm_head is None:
            raise ValueError("Model does not have an lm_head to compute logits")

        self.topk[layer_idx] = {}
        for sample_id in tqdm(self.samples.index, desc=f"Extracting top-{k} tokens"):
            # Compute logits from the activations of the specified layer
            acts_per_sample = self.activations[layer_idx][sample_id]
            acts_per_sample = torch.tensor(acts_per_sample, dtype=model.dtype).to(model.device)
            logits_per_sample = model.lm_head(acts_per_sample)
            # Compute top-k next tokens and their probabilities
            self.topk[layer_idx][sample_id] = _topk(logits_per_sample, tokenizer, k=k)

        num_tokens = sum(len(topk_per_sample.tokens) for topk_per_sample in self.topk[layer_idx].values())
        num_bytes = sum(
            next_tokens.tokens.nbytes + next_tokens.probs.nbytes for next_tokens in self.topk[layer_idx].values()
        )
        print(
            f"Extracted top-{k} next tokens for layer {layer_idx}"
            f"\n  Number of tokens:    {num_tokens}"
            f"\n  Size of topk tokens: {num_bytes / 1024 / 1024:.1f} MB"
        )

    def get_per_layer(
        self,
        layer_idx: int,
        sample_ids: str | None = None,
        pool_method: PoolMethod = "all",
        exclude_bos: bool = True,
        exclude_special_tokens: bool | list[str] = True,
    ) -> Any:
        """Get activations for the specified layer over the pooled tokens."""
        if sample_ids is None:
            sample_ids = self.samples.index.tolist()

        if isinstance(sample_ids, str):
            sample_id = sample_ids
            sample = self.samples.loc[sample_id]
            activations = self.activations[layer_idx][sample_id]  # shape: (num_tokens, hidden_size)
            tokens = sample["tokens"]
            if layer_idx in self.topk:
                activations, tokens, token_positions, topk_tokens, topk_probs = _pool(
                    activations,
                    tokens,
                    self.topk[layer_idx][sample_id].tokens,
                    self.topk[layer_idx][sample_id].probs,
                    pool_method=pool_method,
                    exclude_bos=exclude_bos,
                    exclude_special_tokens=exclude_special_tokens,
                )
            else:
                activations, tokens, token_positions = _pool(
                    activations,
                    tokens,
                    pool_method=pool_method,
                    exclude_bos=exclude_bos,
                    exclude_special_tokens=exclude_special_tokens,
                )
                topk_tokens = [None] * len(tokens)
                topk_probs = [None] * len(tokens)
            sample["tokens_all"] = sample.pop("tokens")  # rename since key used for pooled tokens
            return {
                "id": sample_id,
                **sample.to_dict(),
                "activations": activations,  # shape: (num_pooled_tokens, hidden_size)
                "tokens": tokens,  # shape: (num_pooled_tokens,)
                "token_positions": token_positions,  # shape: (num_pooled_tokens,)
                "topk_tokens": topk_tokens,  # shape: (num_pooled_tokens, topk | None)
                "topk_probs": topk_probs,  # shape: (num_pooled_tokens, topk | None)
            }
        else:
            return [
                self.get_per_layer(
                    sample_ids=sample_id,
                    layer_idx=layer_idx,
                    pool_method=pool_method,
                    exclude_bos=exclude_bos,
                    exclude_special_tokens=exclude_special_tokens,
                )
                for sample_id in sample_ids
            ]

    def get_per_token(
        self,
        sample_ids: str | list[str] | None = None,
        pool_method: PoolMethod = "all",
        exclude_bos: bool = True,
        exclude_special_tokens: bool | list[str] = True,
    ) -> Any:
        """Get activations for the pooled tokens over all layers."""
        if sample_ids is None:
            sample_ids = self.samples.index.tolist()

        if isinstance(sample_ids, str):
            sample_id = sample_ids
            sample = self.samples.loc[sample_id]
            activations = np.stack(
                [self.activations[layer_idx][sample_id] for layer_idx in self.layers],
                axis=1,
            )  # shape: (num_tokens, num_layers, hidden_size)
            tokens = sample["tokens"]
            topk_tokens = np.empty((len(tokens), self.num_layers), dtype=object)
            topk_probs = np.empty((len(tokens), self.num_layers), dtype=object)
            for i, layer_idx in enumerate(self.layers):
                if layer_idx in self.topk:
                    topk_tokens[:, i] = list(self.topk[layer_idx][sample_id].tokens)
                    topk_probs[:, i] = list(self.topk[layer_idx][sample_id].probs)
                else:
                    topk_tokens[:, i] = None
                    topk_probs[:, i] = None
            activations, tokens, token_positions, topk_tokens, topk_probs = _pool(
                activations,
                tokens,
                topk_tokens,
                topk_probs,
                pool_method=pool_method,
                exclude_bos=exclude_bos,
                exclude_special_tokens=exclude_special_tokens,
            )
            sample["tokens_all"] = sample.pop("tokens")  # rename since key used for pooled tokens
            return {
                "id": sample_id,
                **sample.to_dict(),
                "activations": activations,  # shape: (num_pooled_tokens, num_layers, hidden_size)
                "tokens": tokens,  # shape: (num_pooled_tokens,)
                "token_positions": token_positions,  # shape: (num_pooled_tokens,)
                "topk_tokens": topk_tokens,  # shape: (num_pooled_tokens, num_layers, topk | None)
                "topk_probs": topk_probs,  # shape: (num_pooled_tokens, num_layers, topk | None)
            }
        else:
            return [
                self.get_per_token(
                    sample_ids=sample_id,
                    pool_method=pool_method,
                    exclude_bos=exclude_bos,
                    exclude_special_tokens=exclude_special_tokens,
                )
                for sample_id in sample_ids
            ]

    def select(self, sample_ids: list[str]) -> "Activations":
        """Return a new Activations object containing only the specified samples.

        This method only creates a subset of the samples, while the activations dictionary is kept as a reference
        to save memory.
        """
        return Activations(samples=self.samples.loc[sample_ids], activations=self.activations, topk=self.topk)


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

    avg_input_length_chars = np.mean([len(sample["input"]) for sample in dataset])
    avg_input_length_words = np.mean([len(sample["input"].split()) for sample in dataset])
    samples_safe = dataset.filter(lambda sample: sample["is_safe"]).select(range(5))
    samples_unsafe = dataset.filter(lambda sample: not sample["is_safe"]).select(range(5))
    print(
        f"Prepared dataset:"
        f"\n  Input length (avg):   {avg_input_length_chars:.1f} chars, {avg_input_length_words:.1f} words"
        f"\n  Samples:"
        f"\n    Safe:\n{'\n'.join(f'      - {sample}' for sample in samples_safe)}"
        f"\n    Unsafe:\n{'\n'.join(f'      - {sample}' for sample in samples_unsafe)}"
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


@contextlib.contextmanager
def _capture_pre_norm(model):
    """Context manager to capture last hidden state before layer norm in the model's forward pass."""
    last_hidden_state_pre_norm = None

    # Attach hook to capture the last hidden state before the final norm layer
    def hook_fn(module, input):
        nonlocal last_hidden_state_pre_norm
        last_hidden_state_pre_norm = input[0]

    base_model = getattr(model, model.base_model_prefix, model)
    handle = base_model.norm.register_forward_pre_hook(hook_fn)

    # Wrap the forward method to inject the captured pre-norm last hidden state
    original_forward = model.forward

    @wraps(original_forward)
    def wrapped_forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            outputs.hidden_states = (
                outputs.hidden_states[:-1] + (last_hidden_state_pre_norm,) + outputs.hidden_states[-1:]
            )
        return outputs

    model.forward = wrapped_forward.__get__(model, type(model))

    try:
        yield last_hidden_state_pre_norm
    finally:
        # Restore the original forward method and remove the hook
        model.forward = original_forward
        handle.remove()


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    include_response: bool | str = False,
    apply_chat_template: bool = False,
    layers: list[int] | None = None,
    batch_size: int = 8,
    k: int = 10,
) -> Activations:
    """Extract activations from the model for the given dataset."""
    if layers is None:
        layers = list(range(model.config.num_hidden_layers + 2))
    last_layer_idx = model.config.num_hidden_layers + 1  # index of the last layer (after the final norm)

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
    samples = []
    activations = {layer_idx: {} for layer_idx in layers}
    topk = {last_layer_idx: {}}
    for batch in tqdm(dataloader, desc="Extracting activations"):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Forward pass
        with _capture_pre_norm(model):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract activations for each sample
        for i, sample_id in enumerate(batch["ids"]):
            # Get mask to remove padding
            mask = attention_mask[i].bool()
            # Store samples
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i, mask])
            text = tokenizer.convert_tokens_to_string(tokens)
            samples.append(
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
                acts_per_sample = outputs.hidden_states[layer_idx][i, mask]  # shape: (num_valid_tokens, hidden_size)
                activations[layer_idx][sample_id] = acts_per_sample.float().cpu().numpy()
            # Store top-k next tokens and their probabilities for the last layer
            topk[last_layer_idx][sample_id] = _topk(outputs.logits[i, mask], tokenizer, k=k)

    num_samples = len(samples)
    num_tokens = sum(len(sample["tokens"]) for sample in samples)
    num_layers = len(activations)
    hidden_size = acts_per_sample.shape[1:]
    num_bytes_acts = sum(
        acts_per_layer.nbytes for acts_per_layer in activations.values() for acts_per_layer in acts_per_layer.values()
    )
    num_bytes_topk = sum(
        next_tokens.tokens.nbytes + next_tokens.probs.nbytes for next_tokens in topk[last_layer_idx].values()
    )
    print(
        f"Extracted activations for {num_samples} samples"
        f"\n  Number of tokens:     {num_tokens}"
        f"\n  Number of layers:     {num_layers - 2} + 2"
        f"\n  Shape of activations: {hidden_size}"
        f"\n  Size of activations:  {num_bytes_acts / 1024 / 1024:.1f} MB"
        f"\nExtracted top-{k} next tokens for layer {last_layer_idx}"
        f"\n  Number of tokens:    {num_tokens}"
        f"\n  Size of topk tokens: {num_bytes_topk / 1024 / 1024:.1f} MB"
    )

    samples = pd.DataFrame(samples).set_index("id")
    samples.sort_index(inplace=True, key=natsort_keygen())
    return Activations(samples=samples, activations=activations, topk=topk)
