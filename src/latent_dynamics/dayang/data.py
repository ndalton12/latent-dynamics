from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from datasets import Dataset, load_dataset


@dataclass
class DatasetSpec:
    path: str
    split: str | None = None
    name: str | None = None
    load_kwargs: dict[str, Any] | None = None
    filter_fn: Callable[[dict[str, Any]], bool] | None = None
    map_fn: Callable[[dict[str, Any]], int] | None = None


SORRY_BENCH_CATEGORY_DESCRIPTIONS = [
    "Personal Insulting Words",
    "Social-group Insulting Words",
    "Threatening Words",
    "Lewd and Obscene Words",
    "Libelous Words",
    "Self-Harm",
    "Violent Crimes",
    "Harassment",
    "Sexual Crimes",
    "Property Crimes",
    "Public Order Crimes",
    "General Advice for System Intrusion, Hacking & Cyber Attacks, Data Privacy and Confidentiality Violations",
    "Malware Code Generation",
    "Fraud, Scams, and Deception",
    "Financial Crimes",
    "Intellectual Property Infringement",
    "Personal Identifiable Information Violations",
    "Illegal Crafting, Manufacturing, and Distribution",
    "Terrorism",
    "Child-related Crimes",
    "Animal-related Crimes",
    "Environmental Crimes",
    "Evasion of Law Enforcement",
    "Human Elimination & World Domination",
    "Advice on Adult Content",
    "Sexual Explicit Content Generation",
    "Non-sexual Explicit Content Generation",
    "Fake News Generation",
    "False Advertising",
    "Advice on Discrimination Behaviors",
    "Military Use",
    "Political Belief Promotion",
    "Ethical Belief Promotion",
    "Religion Promotion",
    "Social Stereotype Promotion",
    "Extremist Content Promotion",
    "Conspiracy Theories Promotion",
    "False Common Knowledge Promotion",
    "Harmful & Unhealthy Behaviors Promotion",
    "Medical Advice",
    "Financial Advice",
    "Legal Consulting Advice",
    "Governance Decision Advice",
    "Dangerous Machinery Operation Advice",
]


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "wildjailbreak/train/vanilla": DatasetSpec(
        path="allenai/wildjailbreak",
        name="train",
        split="train",
        load_kwargs={"delimiter": "\t", "keep_default_na": False},
        filter_fn=lambda sample: sample["data_type"].startswith("vanilla"),
        map_fn=lambda sample, i: {
            "id": f"wildjailbreak/train/vanilla/{i}",
            "prompt": sample["vanilla"],
            "is_safe": sample["data_type"] == "vanilla_benign",
            "is_adversarial": False,
            "response": sample["completion"],
        },
    ),
    "wildjailbreak/train/adversarial": DatasetSpec(
        path="allenai/wildjailbreak",
        name="train",
        split="train",
        load_kwargs=dict(delimiter="\t", keep_default_na=False),
        filter_fn=lambda sample: sample["data_type"].startswith("adversarial"),
        map_fn=lambda sample, i: {
            "id": f"wildjailbreak/train/adversarial/{i}",
            "prompt": sample["adversarial"],
            "is_safe": sample["data_type"] == "adversarial_benign",
            "is_adversarial": True,
            "response": sample["completion"],
        },
    ),
    "wildjailbreak/eval/adversarial": DatasetSpec(
        path="allenai/wildjailbreak",
        name="eval",
        split="train",
        load_kwargs=dict(delimiter="\t", keep_default_na=False),
        map_fn=lambda sample, i: {
            "id": f"wildjailbreak/eval/adversarial/{i}",
            "prompt": sample["adversarial"],
            "is_safe": sample["data_type"] == "adversarial_benign",
            "is_adversarial": True,
        },
    ),
    "beavertails330k/train": DatasetSpec(
        path="PKU-Alignment/BeaverTails",
        split="330k_train",
        map_fn=lambda sample, i: {
            "id": f"beavertails330k/train/{i}",
            **sample,
            "is_adversarial": False,
        },
    ),
    "beavertails330k/test": DatasetSpec(
        path="PKU-Alignment/BeaverTails",
        split="330k_test",
        map_fn=lambda sample, i: {
            "id": f"beavertails330k/test/{i}",
            **sample,
            "is_adversarial": False,
        },
    ),
    "beavertails30k/train": DatasetSpec(
        path="PKU-Alignment/BeaverTails",
        split="30k_train",
        map_fn=lambda sample, i: {
            "id": f"beavertails30k/train/{i}",
            **sample,
            "is_adversarial": False,
        },
    ),
    "beavertails30k/test": DatasetSpec(
        path="PKU-Alignment/BeaverTails",
        split="30k_test",
        map_fn=lambda sample, i: {
            "id": f"beavertails30k/test/{i}",
            **sample,
            "is_adversarial": False,
        },
    ),
    "sorry_bench": DatasetSpec(
        path="sorry-bench/sorry-bench-202503",
        split="train",
        filter_fn=lambda sample: sample["turns"][0] is not None,  # filter out samples with null prompts
        map_fn=lambda sample, i: {
            "id": f"sorry_bench/{sample['question_id']}",
            "prompt": sample["turns"][0],
            "is_safe": False,
            "is_adversarial": False,
            "prompt_category": SORRY_BENCH_CATEGORY_DESCRIPTIONS[int(sample["category"]) - 1],
            "prompt_style": sample["prompt_style"],
        },
    ),
    "xstest": DatasetSpec(
        path="Paul/XSTest",
        split="train",
        map_fn=lambda sample, i: {
            "id": f"xstest/{sample['id']}",
            "prompt": sample["prompt"],
            "is_safe": sample["label"] == "safe",
            "is_adversarial": False,
            "prompt_type": sample["type"],
            "prompt_focus": sample["focus"],
        },
    ),
}


def load_dataset_from_spec(
    dataset_spec: DatasetSpec | str,
    max_samples: int | None = None,
) -> tuple[Dataset]:
    """Load a dataset from a DatasetSpec."""
    if isinstance(dataset_spec, str):
        dataset_spec = DATASET_REGISTRY[dataset_spec]
    # Load the dataset
    dataset: Dataset = load_dataset(
        dataset_spec.path,
        name=dataset_spec.name,
        split=dataset_spec.split,
        **(dataset_spec.load_kwargs or {}),
    )
    # Apply the filtering function if provided
    if dataset_spec.filter_fn is not None:
        dataset = dataset.filter(dataset_spec.filter_fn)
    # Apply the mapping function if provided
    if dataset_spec.map_fn is not None:
        dataset = dataset.map(dataset_spec.map_fn, remove_columns=dataset.column_names, with_indices=True)
    # Downsample the dataset
    if max_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    # Ensure certain columns exist
    for column in ["id", "prompt", "is_safe", "is_adversarial"]:
        if column not in dataset.column_names:
            raise ValueError(f"Dataset is missing required column '{column}'")

    avg_prompt_length_chars = np.mean([len(sample["prompt"]) for sample in dataset])
    avg_prompt_length_words = np.mean([len(sample["prompt"].split()) for sample in dataset])
    num_total = len(dataset)
    num_safe = sum(dataset["is_safe"])
    num_unsafe = num_total - num_safe
    samples_safe = dataset.filter(lambda sample: sample["is_safe"]).select(range(5))
    samples_unsafe = dataset.filter(lambda sample: not sample["is_safe"]).select(range(5))
    print(
        f"Loaded dataset: {dataset_spec.path}"
        f"\n  Number of samples:   {len(dataset)}"
        f"\n    Safe:              {num_safe} ({num_safe / num_total * 100:.1f}%)"
        f"\n    Unsafe:            {num_unsafe} ({num_unsafe / num_total * 100:.1f}%)"
        f"\n  Columns:             {dataset.column_names}"
        f"\n  Prompt length (avg): {avg_prompt_length_chars:.1f} chars, {avg_prompt_length_words:.1f} words"
        f"\n  Samples:"
        f"\n    Safe:\n{'\n'.join(f'      - {sample}' for sample in samples_safe)}"
        f"\n    Unsafe:\n{'\n'.join(f'      - {sample}' for sample in samples_unsafe)}"
    )

    return dataset
