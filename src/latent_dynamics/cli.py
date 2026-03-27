from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

from latent_dynamics.config import DATASET_REGISTRY, MODEL_REGISTRY

app = typer.Typer(
    name="latent-dynamics",
    help="Latent trajectory analysis for language models.",
    add_completion=False,
)

DEFAULT_GENERATION_JUDGE_MODEL = "gpt-5-mini"


class ModelKey(str, Enum):
    qwen3_8b = "qwen3_8b"
    llama_3_1_8b = "llama_3_1_8b"
    gemma3_4b = "gemma3_4b"


class DatasetKey(str, Enum):
    toy_contrastive = "toy_contrastive"
    wildjailbreak = "wildjailbreak"
    xstest = "xstest"


class DirectionMethod(str, Enum):
    probe_weight = "probe_weight"
    mean_diff = "mean_diff"
    pca = "pca"


class PoolingMode(str, Enum):
    last = "last"
    mean = "mean"
    max_norm = "max_norm"


class PromptSubsetMode(str, Enum):
    all = "all"
    vanilla = "vanilla"
    adversarial = "adversarial"


class AcquisitionMode(str, Enum):
    random = "random"
    static_uncertainty = "static_uncertainty"
    static_disagreement = "static_disagreement"
    static_uncertainty_diversity = "static_uncertainty_diversity"
    dynamic_uncertainty = "dynamic_uncertainty"
    dynamic_disagreement = "dynamic_disagreement"
    dynamic_uncertainty_diversity = "dynamic_uncertainty_diversity"


class ComparisonAcquisitionMode(str, Enum):
    all = "all"
    random = "random"
    static_uncertainty = "static_uncertainty"
    static_disagreement = "static_disagreement"
    static_uncertainty_diversity = "static_uncertainty_diversity"
    dynamic_uncertainty = "dynamic_uncertainty"
    dynamic_disagreement = "dynamic_disagreement"
    dynamic_uncertainty_diversity = "dynamic_uncertainty_diversity"


class EnsembleMode(str, Enum):
    single_layer = "single_layer"
    mean_score = "mean_score"
    majority_vote = "majority_vote"


@app.command()
def extract(
    model: Annotated[
        ModelKey, typer.Option(help="Model key from registry.")
    ] = ModelKey.gemma3_4b,
    dataset: Annotated[
        DatasetKey, typer.Option(help="Dataset key from registry.")
    ] = DatasetKey.toy_contrastive,
    layer: Annotated[
        List[int],
        typer.Option(help="Layer index (repeat for multiple: --layer 5 --layer 10)."),
    ] = [],
    all_layers: Annotated[
        bool,
        typer.Option("--all-layers/--no-all-layers", help="Extract all layers."),
    ] = False,
    max_samples: Annotated[int, typer.Option(help="Maximum number of examples.")] = 120,
    max_input_tokens: Annotated[
        int, typer.Option(help="Maximum token sequence length.")
    ] = 256,
    output: Annotated[
        Path,
        typer.Option(
            help="Root output directory. Activations are saved under {dataset}/{model}/layer_{N}/."
        ),
    ] = Path("./activations"),
    push_to_hub: Annotated[
        Optional[str], typer.Option(help="HuggingFace Hub repo id (e.g. user/repo).")
    ] = None,
    device: Annotated[
        Optional[str], typer.Option(help="Device override (cuda/mps/cpu).")
    ] = None,
    use_generate: Annotated[
        bool, typer.Option(help="Generate continuation before extracting.")
    ] = False,
    judge_generations: Annotated[
        bool,
        typer.Option(
            "--judge-generations",
            help=(
                "Judge generated responses and store judge labels in metadata. "
                f"Uses {DEFAULT_GENERATION_JUDGE_MODEL}."
            ),
            is_flag=True,
        ),
    ] = False,
    max_new_tokens: Annotated[
        int, typer.Option(help="Tokens to generate (requires --use-generate).")
    ] = 256,
    include_prompt: Annotated[
        bool,
        typer.Option(
            "--include-prompt/--no-include-prompt",
            help="Include prompt tokens in trajectory when generating.",
        ),
    ] = True,
    load_4bit: Annotated[
        bool,
        typer.Option(
            "--4bit/--no-4bit", help="Load model in 4-bit (CUDA + bitsandbytes)."
        ),
    ] = False,
    use_true_batch_inference: Annotated[
        bool,
        typer.Option(
            "--use-true-batch-inference",
            help="Use true batch inference.",
            is_flag=True,
        ),
    ] = False,
    inference_batch_size: Annotated[
        int,
        typer.Option(
            "--inference-batch-size",
            help=(
                "Batch size for true batch inference "
                "(used with --use-true-batch-inference)."
            ),
        ),
    ] = 16,
) -> None:
    """Extract hidden-state trajectories from a model and dataset, save to disk.

    Activations are stored under a structured path:
      {output}/{dataset}/{model}/layer_{N}/
    Each layer directory contains a metadata.json that records the layer index,
    input prompts, and generated text (when using --use-generate) so that
    activations can be matched back to their source data. With
    --judge-generations, metadata also includes judge labels for unsafe and
    compliance/refusal.
    """
    from tqdm.auto import tqdm

    from latent_dynamics.activations import extract_multi_layer_trajectories
    from latent_dynamics.config import RunConfig
    from latent_dynamics.data import load_examples, prepare_text_and_labels
    from latent_dynamics.hub import (
        METADATA_FILE,
        TRAJECTORIES_FILE,
        activation_subpath,
        save_activations_shard,
        write_activation_metadata,
    )
    from latent_dynamics.hub import (
        push_to_hub as _push,
    )
    from latent_dynamics.judge import SafetyJudge, judge_prompt_generations
    from latent_dynamics.models import load_model_and_tokenizer, resolve_device
    from latent_dynamics.utils import (
        TRAJECTORY_SHARD_MANIFEST_FILE,
        list_trajectory_shards,
        write_trajectory_shard_manifest,
    )

    if judge_generations and not use_generate:
        raise typer.BadParameter(
            "--judge-generations requires --use-generate so completions exist."
        )

    if inference_batch_size < 1:
        raise typer.BadParameter("--inference-batch-size must be >= 1.")

    layer_indices: list[int] | None = None
    if all_layers:
        layer_indices = None
    elif layer:
        layer_indices = layer
    else:
        layer_indices = [5]

    cfg = RunConfig(
        model_key=model.value,
        dataset_key=dataset.value,
        max_samples=max_samples,
        max_input_tokens=max_input_tokens,
        layer_idx=layer_indices[0] if layer_indices else 0,
        device=resolve_device(device),
        use_generate=use_generate,
        max_new_tokens=max_new_tokens,
        include_prompt_in_trajectory=include_prompt,
        use_true_batch_inference=use_true_batch_inference,
        inference_batch_size=inference_batch_size,
    )

    typer.echo(f"Device: {cfg.device}")
    typer.echo(f"Model:  {cfg.model_key}")
    typer.echo(f"Dataset: {cfg.dataset_key} (max_samples={cfg.max_samples})")
    typer.echo(f"Layers:  {'all' if layer_indices is None else layer_indices}")
    typer.echo(
        f"True batch inference: {cfg.use_true_batch_inference} "
        f"(batch_size={cfg.inference_batch_size})"
    )

    ds, spec = load_examples(cfg.dataset_key, cfg.max_samples, stratify_labels=True)
    texts, labels, metadata = prepare_text_and_labels(ds, spec, return_metadata=True)

    if labels is None:
        typer.echo(
            "Warning: no labels produced. Downstream analysis will be limited.",
            err=True,
        )

    typer.echo(f"Loaded {len(texts)} examples, loading model...")
    mdl, tokenizer = load_model_and_tokenizer(
        cfg.model_key, cfg.device, load_in_4bit=load_4bit
    )

    typer.echo(
        f"Extracting trajectories for layers "
        f"{'all' if layer_indices is None else layer_indices} ..."
    )
    empty_result = extract_multi_layer_trajectories(
        mdl,
        tokenizer,
        [],
        layer_indices=layer_indices,
        cfg=cfg,
        show_progress=False,
    )
    extracted_layers = sorted(empty_result.per_layer.keys())

    layer_dirs = {
        li: output / activation_subpath(cfg.dataset_key, cfg.model_key, li)
        for li in extracted_layers
    }
    for layer_dir in layer_dirs.values():
        layer_dir.mkdir(parents=True, exist_ok=True)
        single_path = layer_dir / TRAJECTORIES_FILE
        if single_path.exists():
            single_path.unlink()
        metadata_path = layer_dir / METADATA_FILE
        if metadata_path.exists():
            metadata_path.unlink()
        manifest_path = layer_dir / TRAJECTORY_SHARD_MANIFEST_FILE
        if manifest_path.exists():
            manifest_path.unlink()
        for shard_path in list_trajectory_shards(layer_dir):
            shard_path.unlink()

    all_token_texts: list[list[str]] = []
    all_generated_texts: list[str | None] = []
    next_idx_by_layer: dict[int, int] = {li: 0 for li in extracted_layers}
    shard_idx_by_layer: dict[int, int] = {li: 0 for li in extracted_layers}
    shard_manifest_entries_by_layer: dict[int, list[dict[str, object]]] = {
        li: [] for li in extracted_layers
    }

    judge: SafetyJudge | None = None
    judge_unsafe_labels: list[int] = []
    judge_compliance_labels: list[int] = []
    judge_confidences: list[float] = []
    if judge_generations:
        typer.echo(f"Initializing judge ({DEFAULT_GENERATION_JUDGE_MODEL})...")
        judge = SafetyJudge(
            model=DEFAULT_GENERATION_JUDGE_MODEL,
            max_concurrency=12,
            batch_size=32,
            requests_per_minute=120,
            show_progress=True,
        )

    chunk_size = max(1, int(cfg.inference_batch_size))
    chunk_starts = range(0, len(texts), chunk_size)
    chunk_iter = tqdm(chunk_starts, desc="extract chunks", disable=len(texts) == 0)
    for start in chunk_iter:
        chunk_texts = texts[start : start + chunk_size]
        chunk_result = extract_multi_layer_trajectories(
            mdl,
            tokenizer,
            chunk_texts,
            layer_indices=layer_indices,
            cfg=cfg,
            show_progress=False,
        )
        all_token_texts.extend(chunk_result.token_texts)
        all_generated_texts.extend(chunk_result.generated_texts)

        if judge is not None:
            chunk_judge_results = judge_prompt_generations(
                prompts=chunk_texts,
                generations=chunk_result.generated_texts,
                judge=judge,
                cache=None,
            )
            judge_unsafe_labels.extend(int(r.unsafe) for r in chunk_judge_results)
            judge_compliance_labels.extend(
                int(r.compliance) for r in chunk_judge_results
            )
            judge_confidences.extend(float(r.confidence) for r in chunk_judge_results)

        for li in extracted_layers:
            start_idx = next_idx_by_layer[li]
            next_idx, _ = save_activations_shard(
                output_dir=layer_dirs[li],
                trajectories=chunk_result.per_layer[li],
                start_idx=start_idx,
                shard_idx=shard_idx_by_layer[li],
                manifest_entries=shard_manifest_entries_by_layer[li],
            )
            next_idx_by_layer[li] = next_idx
            shard_idx_by_layer[li] += 1

    typer.echo(f"Extracted {len(texts)} trajectories x {len(extracted_layers)} layers.")

    extra_metadata: dict[str, object] = {}
    if metadata is not None:
        extra_metadata["example_metadata"] = metadata

    if judge is not None:
        n_unsafe = sum(judge_unsafe_labels)
        n_compliance = sum(judge_compliance_labels)
        typer.echo(
            f"Judge summary: unsafe={n_unsafe} safe={len(judge_unsafe_labels) - n_unsafe} "
            f"compliance={n_compliance} refusal={len(judge_compliance_labels) - n_compliance}"
        )
        extra_metadata["judge_model"] = DEFAULT_GENERATION_JUDGE_MODEL
        extra_metadata["judge_unsafe_labels"] = judge_unsafe_labels
        extra_metadata["judge_compliance_labels"] = judge_compliance_labels
        extra_metadata["judge_confidences"] = judge_confidences

    for li in extracted_layers:
        layer_cfg = RunConfig(**{**cfg.__dict__, "layer_idx": li})
        write_trajectory_shard_manifest(
            output_dir=layer_dirs[li],
            entries=shard_manifest_entries_by_layer[li],
        )
        write_activation_metadata(
            output_dir=layer_dirs[li],
            trajectories_count=next_idx_by_layer[li],
            texts=texts,
            labels=labels,
            token_texts=all_token_texts,
            cfg=layer_cfg,
            generated_texts=all_generated_texts,
            extra_metadata=(extra_metadata if extra_metadata else None),
        )
        typer.echo(f"  Saved layer {li} -> {layer_dirs[li]}")

    if push_to_hub:
        url = _push(output, push_to_hub)
        typer.echo(f"Pushed to {url}")


@app.command("run-active-learning")
def run_active_learning_cmd(
    activations_root: Annotated[
        Path,
        typer.Option(
            help=(
                "Activation root or layer leaf. For multi-layer runs, point to a "
                "directory containing layer_* subdirectories."
            )
        ),
    ] = Path("activations/wildjailbreak/gemma3_4b"),
    layer: Annotated[
        List[int],
        typer.Option(
            help="Layer index (repeat for multiple layers: --layer 7 --layer 34)."
        ),
    ] = [],
    pooling: Annotated[
        PoolingMode,
        typer.Option(help="Token pooling method for trajectory -> feature vector."),
    ] = PoolingMode.last,
    label_field: Annotated[
        str,
        typer.Option(
            help=(
                "Metadata label field to train against "
                "(default: judge_unsafe_labels). "
                "Use 'auto' to try judge_unsafe_labels, labels, then judge_compliance_labels."
            )
        ),
    ] = "judge_unsafe_labels",
    prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Limit examples by prompt group from metadata example_metadata.data_type. "
                "Use all, vanilla, or adversarial."
            )
        ),
    ] = PromptSubsetMode.all,
    train_prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Subset used for train+pool split construction (L_0 and U_0). "
                "Applied after --prompt-subset filtering."
            )
        ),
    ] = PromptSubsetMode.all,
    test_prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Subset used for final held-out test split. "
                "Applied after --prompt-subset filtering."
            )
        ),
    ] = PromptSubsetMode.all,
    max_examples: Annotated[
        Optional[int],
        typer.Option(
            help="Optional cap on number of examples (uses first N examples)."
        ),
    ] = None,
    acquisition: Annotated[
        AcquisitionMode,
        typer.Option(help="Acquisition strategy."),
    ] = AcquisitionMode.dynamic_uncertainty,
    ensemble_mode: Annotated[
        EnsembleMode,
        typer.Option(help="How to aggregate predictions across layers."),
    ] = EnsembleMode.single_layer,
    scoring_layer: Annotated[
        Optional[int],
        typer.Option(
            help="Layer used for single_layer mode (defaults to first loaded layer)."
        ),
    ] = None,
    initial_labeled_size: Annotated[
        int,
        typer.Option(help="Initial labeled set size (|L_0|)."),
    ] = 120,
    query_budget: Annotated[
        int,
        typer.Option(help="Total number of queried labels."),
    ] = 480,
    batch_size: Annotated[
        int,
        typer.Option(help="Batch size queried per round."),
    ] = 60,
    test_size: Annotated[
        float,
        typer.Option(help="Held-out test split fraction."),
    ] = 0.2,
    uncertainty_weight: Annotated[
        float,
        typer.Option(
            help="Weight of uncertainty in uncertainty+diversity acquisition."
        ),
    ] = 0.5,
    diversity_prefilter_factor: Annotated[
        int,
        typer.Option(
            help="Top-(factor*batch) uncertain candidates used before diversity selection."
        ),
    ] = 10,
    seed: Annotated[
        int,
        typer.Option(help="Random seed for splits and probe training."),
    ] = 42,
    output_json: Annotated[
        Path,
        typer.Option(help="Where to write the active-learning report JSON."),
    ] = Path("results/active_learning_report.json"),
) -> None:
    """Run active learning over activation features and report test performance vs queried labels."""
    from latent_dynamics.learning_experiment import (
        ActiveLearningConfig,
        load_activation_feature_bundle,
        run_active_learning_experiment,
    )

    selected_layers = layer if layer else None
    bundle = load_activation_feature_bundle(
        root_or_leaf=activations_root,
        layers=selected_layers,
        pooling=pooling.value,
        label_field=label_field,
        prompt_subset=prompt_subset.value,
        max_examples=max_examples,
    )
    group_counts: str | None = None
    if bundle.prompt_groups is not None:
        unique_groups, group_sizes = np.unique(bundle.prompt_groups, return_counts=True)
        group_counts = ", ".join(
            f"{str(g)}={int(c)}"
            for g, c in zip(unique_groups.tolist(), group_sizes.tolist(), strict=False)
        )
    typer.echo(
        f"Loaded activation features: n={bundle.labels.shape[0]} "
        f"layers={bundle.layers} label_field={bundle.label_field} "
        f"prompt_subset={bundle.prompt_subset} "
        f"train_prompt_subset={train_prompt_subset.value} "
        f"test_prompt_subset={test_prompt_subset.value}"
    )
    if group_counts is not None:
        typer.echo(f"Prompt groups in selection: {group_counts}")

    cfg = ActiveLearningConfig(
        acquisition=acquisition.value,
        initial_labeled_size=initial_labeled_size,
        query_budget=query_budget,
        batch_size=batch_size,
        test_size=test_size,
        random_state=seed,
        ensemble_mode=ensemble_mode.value,
        scoring_layer=scoring_layer,
        uncertainty_weight=uncertainty_weight,
        diversity_prefilter_factor=diversity_prefilter_factor,
        train_prompt_subset=train_prompt_subset.value,
        test_prompt_subset=test_prompt_subset.value,
    )
    result = run_active_learning_experiment(bundle=bundle, config=cfg)
    payload = result.to_dict()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    rounds = payload["rounds"]
    first = rounds[0]
    last = rounds[-1]
    typer.echo(
        "Performance trajectory:"
        f" queried={first['queried_labels']} acc={first['test_accuracy']:.3f} "
        f"auroc={first['test_auroc'] if first['test_auroc'] is not None else 'n/a'}"
    )
    typer.echo(
        "Final:"
        f" queried={last['queried_labels']} acc={last['test_accuracy']:.3f} "
        f"auroc={last['test_auroc'] if last['test_auroc'] is not None else 'n/a'}"
    )
    typer.echo(f"Wrote report: {output_json}")


@app.command("compare-active-learning-acquisitions")
def compare_active_learning_acquisitions_cmd(
    activations_root: Annotated[
        Path,
        typer.Option(
            help=(
                "Activation root or layer leaf. For multi-layer runs, point to a "
                "directory containing layer_* subdirectories."
            )
        ),
    ] = Path("activations/wildjailbreak/gemma3_4b"),
    layer: Annotated[
        List[int],
        typer.Option(
            help="Layer index (repeat for multiple layers: --layer 7 --layer 34)."
        ),
    ] = [],
    pooling: Annotated[
        PoolingMode,
        typer.Option(help="Token pooling method for trajectory -> feature vector."),
    ] = PoolingMode.last,
    label_field: Annotated[
        str,
        typer.Option(
            help=(
                "Metadata label field to train against "
                "(default: judge_unsafe_labels). "
                "Use 'auto' to try judge_unsafe_labels, labels, then judge_compliance_labels."
            )
        ),
    ] = "judge_unsafe_labels",
    prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Limit examples by prompt group from metadata example_metadata.data_type. "
                "Use all, vanilla, or adversarial."
            )
        ),
    ] = PromptSubsetMode.all,
    train_prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Subset used for train+pool split construction (L_0 and U_0). "
                "Applied after --prompt-subset filtering."
            )
        ),
    ] = PromptSubsetMode.all,
    test_prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Subset used for final held-out test split. "
                "Applied after --prompt-subset filtering."
            )
        ),
    ] = PromptSubsetMode.all,
    max_examples: Annotated[
        Optional[int],
        typer.Option(
            help="Optional cap on number of examples (uses first N examples)."
        ),
    ] = None,
    acquisition: Annotated[
        List[ComparisonAcquisitionMode],
        typer.Option(
            help=(
                "Acquisition functions to compare. Repeat this option to add more "
                "(e.g., --acquisition random --acquisition dynamic_uncertainty). "
                "Use --acquisition all to include all available acquisition functions."
            )
        ),
    ] = [ComparisonAcquisitionMode.all],
    ensemble_mode: Annotated[
        EnsembleMode,
        typer.Option(help="How to aggregate predictions across layers."),
    ] = EnsembleMode.single_layer,
    scoring_layer: Annotated[
        Optional[int],
        typer.Option(
            help="Layer used for single_layer mode (defaults to first loaded layer)."
        ),
    ] = None,
    initial_labeled_size: Annotated[
        int,
        typer.Option(help="Initial labeled set size (|L_0|)."),
    ] = 120,
    query_budget: Annotated[
        int,
        typer.Option(help="Total number of queried labels."),
    ] = 480,
    batch_size: Annotated[
        int,
        typer.Option(help="Batch size queried per round."),
    ] = 60,
    test_size: Annotated[
        float,
        typer.Option(help="Held-out test split fraction."),
    ] = 0.2,
    uncertainty_weight: Annotated[
        float,
        typer.Option(
            help="Weight of uncertainty in uncertainty+diversity acquisition."
        ),
    ] = 0.5,
    diversity_prefilter_factor: Annotated[
        int,
        typer.Option(
            help="Top-(factor*batch) uncertain candidates used before diversity selection."
        ),
    ] = 10,
    seed: Annotated[
        int,
        typer.Option(help="Random seed for shared split and probe training."),
    ] = 42,
    num_seeds: Annotated[
        int,
        typer.Option(
            help=(
                "Number of seeds to run for aggregation. "
                "When >1, compares each acquisition across multiple random_state values."
            )
        ),
    ] = 1,
    seed_step: Annotated[
        int,
        typer.Option(
            help=(
                "Step between seeds when --num-seeds > 1. "
                "Seed list is: seed + i*seed_step."
            )
        ),
    ] = 1,
    output_json: Annotated[
        Path,
        typer.Option(help="Where to write the acquisition-comparison report JSON."),
    ] = Path("results/active_learning_acquisition_comparison.json"),
) -> None:
    """Compare acquisition functions with a shared initial labeled set/pool/test split."""
    from latent_dynamics.learning_experiment import (
        ActiveLearningConfig,
        compare_acquisition_functions,
        compare_acquisition_functions_multi_seed,
        load_activation_feature_bundle,
    )

    selected_layers = layer if layer else None
    bundle = load_activation_feature_bundle(
        root_or_leaf=activations_root,
        layers=selected_layers,
        pooling=pooling.value,
        label_field=label_field,
        prompt_subset=prompt_subset.value,
        max_examples=max_examples,
    )
    group_counts: str | None = None
    if bundle.prompt_groups is not None:
        unique_groups, group_sizes = np.unique(bundle.prompt_groups, return_counts=True)
        group_counts = ", ".join(
            f"{str(g)}={int(c)}"
            for g, c in zip(unique_groups.tolist(), group_sizes.tolist(), strict=False)
        )
    all_acquisitions = [
        AcquisitionMode.random.value,
        AcquisitionMode.static_uncertainty.value,
        AcquisitionMode.static_disagreement.value,
        AcquisitionMode.static_uncertainty_diversity.value,
        AcquisitionMode.dynamic_uncertainty.value,
        AcquisitionMode.dynamic_disagreement.value,
        AcquisitionMode.dynamic_uncertainty_diversity.value,
    ]
    requested = [item.value for item in acquisition]
    if ComparisonAcquisitionMode.all.value in requested:
        acquisitions = all_acquisitions
    else:
        acquisitions = []
        for item in requested:
            if item not in acquisitions:
                acquisitions.append(item)
    typer.echo(
        f"Loaded activation features: n={bundle.labels.shape[0]} "
        f"layers={bundle.layers} label_field={bundle.label_field} "
        f"prompt_subset={bundle.prompt_subset} "
        f"train_prompt_subset={train_prompt_subset.value} "
        f"test_prompt_subset={test_prompt_subset.value}"
    )
    if group_counts is not None:
        typer.echo(f"Prompt groups in selection: {group_counts}")
    typer.echo(f"Comparing acquisitions on one shared split: {acquisitions}")
    if num_seeds < 1:
        raise typer.BadParameter("--num-seeds must be >= 1.")
    if seed_step <= 0:
        raise typer.BadParameter("--seed-step must be >= 1.")

    base_cfg = ActiveLearningConfig(
        acquisition=acquisitions[0],
        initial_labeled_size=initial_labeled_size,
        query_budget=query_budget,
        batch_size=batch_size,
        test_size=test_size,
        random_state=seed,
        ensemble_mode=ensemble_mode.value,
        scoring_layer=scoring_layer,
        uncertainty_weight=uncertainty_weight,
        diversity_prefilter_factor=diversity_prefilter_factor,
        train_prompt_subset=train_prompt_subset.value,
        test_prompt_subset=test_prompt_subset.value,
    )
    if num_seeds == 1:
        comparison = compare_acquisition_functions(
            bundle=bundle,
            base_config=base_cfg,
            acquisitions=acquisitions,
        )
        payload = comparison.to_dict()
    else:
        seeds = [seed + (i * seed_step) for i in range(num_seeds)]
        typer.echo(f"Running multi-seed comparison across seeds: {seeds}")
        comparison = compare_acquisition_functions_multi_seed(
            bundle=bundle,
            base_config=base_cfg,
            acquisitions=acquisitions,
            seeds=seeds,
        )
        payload = comparison.to_dict()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    if num_seeds == 1:
        for acq in payload["acquisitions"]:
            metrics = payload["summary"][acq]
            aulc_acc = metrics.get("aulc_test_accuracy_normalized")
            aulc_auc = metrics.get("aulc_test_auroc_normalized")
            typer.echo(
                f"{acq}: queried={metrics['final_queried_labels']} "
                f"acc={metrics['final_test_accuracy']:.3f} "
                f"auroc={metrics['final_test_auroc'] if metrics['final_test_auroc'] is not None else 'n/a'} "
                f"aulc_acc={f'{aulc_acc:.3f}' if aulc_acc is not None else 'n/a'} "
                f"aulc_auroc={f'{aulc_auc:.3f}' if aulc_auc is not None else 'n/a'}"
            )
    else:
        for acq in payload["acquisitions"]:
            metrics = payload["aggregate_final"][acq]
            aulc = payload["aggregate_aulc"][acq]
            acc = metrics["final_test_accuracy"]
            auc = metrics["final_test_auroc"]
            queried = metrics["final_queried_labels"]
            aulc_acc = aulc["aulc_test_accuracy_normalized"]
            aulc_auc = aulc["aulc_test_auroc_normalized"]
            acc_text = (
                "n/a"
                if acc["mean"] is None
                else f"{acc['mean']:.3f} +/- {acc['stderr']:.3f} (sd={acc['stddev']:.3f}, n={acc['support']})"
            )
            auc_text = (
                "n/a"
                if auc["mean"] is None
                else f"{auc['mean']:.3f} +/- {auc['stderr']:.3f} (sd={auc['stddev']:.3f}, n={auc['support']})"
            )
            queried_text = (
                "n/a"
                if queried["mean"] is None
                else f"{queried['mean']:.1f} (sd={queried['stddev']:.2f}, n={queried['support']})"
            )
            aulc_acc_text = (
                "n/a"
                if aulc_acc["mean"] is None
                else f"{aulc_acc['mean']:.3f} +/- {aulc_acc['stderr']:.3f} (sd={aulc_acc['stddev']:.3f}, n={aulc_acc['support']})"
            )
            aulc_auc_text = (
                "n/a"
                if aulc_auc["mean"] is None
                else f"{aulc_auc['mean']:.3f} +/- {aulc_auc['stderr']:.3f} (sd={aulc_auc['stddev']:.3f}, n={aulc_auc['support']})"
            )
            typer.echo(
                f"{acq}: queried={queried_text} "
                f"acc={acc_text} "
                f"auroc={auc_text} "
                f"aulc_acc={aulc_acc_text} "
                f"aulc_auroc={aulc_auc_text}"
            )
    typer.echo(f"Wrote report: {output_json}")


@app.command("plot-active-learning-figures")
def plot_active_learning_figures_cmd(
    comparison_json: Annotated[
        Path,
        typer.Option(
            help=(
                "Path to comparison JSON produced by compare-active-learning-acquisitions."
            )
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where Figure 1/2/3 files will be written."),
    ] = Path("results/figures"),
) -> None:
    """Generate three high-quality figures from active-learning comparison results."""
    from latent_dynamics.learning_experiment import generate_comparison_figures

    if not comparison_json.exists():
        raise typer.BadParameter(f"comparison_json does not exist: {comparison_json}")

    outputs = generate_comparison_figures(
        comparison_json=comparison_json,
        output_dir=output_dir,
    )
    for fig_name, paths in outputs.items():
        typer.echo(f"{fig_name}:")
        typer.echo(f"  png: {paths['png']}")
        typer.echo(f"  pdf: {paths['pdf']}")


@app.command("run-contrastive-probe")
def run_contrastive_probe_cmd(
    activations_root: Annotated[
        Path,
        typer.Option(
            help=(
                "Activation root or layer leaf. For multi-layer runs, point to a "
                "directory containing layer_* subdirectories."
            )
        ),
    ] = Path("activations/wildjailbreak/gemma3_4b"),
    layer: Annotated[
        List[int],
        typer.Option(
            help="Layer index (repeat for multiple layers: --layer 7 --layer 34)."
        ),
    ] = [],
    pooling: Annotated[
        PoolingMode,
        typer.Option(help="Token pooling method for trajectory -> feature vector."),
    ] = PoolingMode.last,
    label_field: Annotated[
        str,
        typer.Option(
            help=(
                "Metadata label field for downstream binary probe "
                "(default: judge_unsafe_labels). "
                "Use 'auto' to try judge_unsafe_labels, labels, then judge_compliance_labels."
            )
        ),
    ] = "judge_unsafe_labels",
    prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Limit examples by prompt group from metadata example_metadata.data_type. "
                "Use all, vanilla, or adversarial."
            )
        ),
    ] = PromptSubsetMode.all,
    train_prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Subset used for train+val split construction. "
                "Applied after --prompt-subset filtering."
            )
        ),
    ] = PromptSubsetMode.all,
    test_prompt_subset: Annotated[
        PromptSubsetMode,
        typer.Option(
            help=(
                "Subset used for final held-out test split. "
                "Applied after --prompt-subset filtering."
            )
        ),
    ] = PromptSubsetMode.all,
    max_examples: Annotated[
        Optional[int],
        typer.Option(
            help="Optional cap on number of examples (uses first N examples)."
        ),
    ] = None,
    test_size: Annotated[
        float,
        typer.Option(help="Held-out test split fraction."),
    ] = 0.2,
    val_size: Annotated[
        float,
        typer.Option(help="Validation split fraction of the train+val pool."),
    ] = 0.1,
    hidden_dim: Annotated[
        int,
        typer.Option(help="Hidden dimension of the contrastive MLP."),
    ] = 256,
    embedding_dim: Annotated[
        int,
        typer.Option(help="Output embedding dimension of the contrastive MLP."),
    ] = 64,
    dropout: Annotated[
        float,
        typer.Option(help="Dropout rate in the contrastive MLP."),
    ] = 0.1,
    temperature: Annotated[
        float,
        typer.Option(help="Temperature for supervised contrastive loss."),
    ] = 0.07,
    learning_rate: Annotated[
        float,
        typer.Option(help="AdamW learning rate for contrastive encoder."),
    ] = 1e-3,
    weight_decay: Annotated[
        float,
        typer.Option(help="AdamW weight decay for contrastive encoder."),
    ] = 1e-4,
    batch_size: Annotated[
        int,
        typer.Option(help="Minibatch size for contrastive training."),
    ] = 128,
    max_epochs: Annotated[
        int,
        typer.Option(help="Maximum number of epochs for contrastive training."),
    ] = 100,
    patience: Annotated[
        int,
        typer.Option(help="Early-stopping patience on validation loss."),
    ] = 10,
    seed: Annotated[
        int,
        typer.Option(help="Random seed for split construction and model training."),
    ] = 42,
    output_json: Annotated[
        Path,
        typer.Option(help="Where to write the contrastive-probe report JSON."),
    ] = Path("results/contrastive_probe_report.json"),
) -> None:
    """Train a supervised-contrastive head on frozen activations and compare against a raw logistic baseline."""
    from latent_dynamics.learning_experiment import (
        ContrastiveProbeConfig,
        load_activation_feature_bundle,
        run_contrastive_probe_experiment,
    )

    selected_layers = layer if layer else None
    bundle = load_activation_feature_bundle(
        root_or_leaf=activations_root,
        layers=selected_layers,
        pooling=pooling.value,
        label_field=label_field,
        prompt_subset=prompt_subset.value,
        max_examples=max_examples,
    )
    group_counts: str | None = None
    if bundle.prompt_groups is not None:
        unique_groups, group_sizes = np.unique(bundle.prompt_groups, return_counts=True)
        group_counts = ", ".join(
            f"{str(g)}={int(c)}"
            for g, c in zip(unique_groups.tolist(), group_sizes.tolist(), strict=False)
        )
    typer.echo(
        f"Loaded activation features: n={bundle.labels.shape[0]} "
        f"layers={bundle.layers} label_field={bundle.label_field} "
        f"prompt_subset={bundle.prompt_subset} "
        f"train_prompt_subset={train_prompt_subset.value} "
        f"test_prompt_subset={test_prompt_subset.value}"
    )
    if group_counts is not None:
        typer.echo(f"Prompt groups in selection: {group_counts}")

    cfg = ContrastiveProbeConfig(
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
        temperature=temperature,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        train_prompt_subset=train_prompt_subset.value,
        test_prompt_subset=test_prompt_subset.value,
    )
    result = run_contrastive_probe_experiment(bundle=bundle, config=cfg)
    payload = result.to_dict()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    for layer_key in payload["per_layer"]:
        layer_payload = payload["per_layer"][layer_key]
        raw_metrics = layer_payload["raw_probe"]
        contrastive_metrics = layer_payload["contrastive_probe"]
        typer.echo(
            f"layer={layer_key}: "
            f"raw_acc={raw_metrics['accuracy']:.3f} "
            f"raw_auroc={raw_metrics['auroc'] if raw_metrics['auroc'] is not None else 'n/a'} "
            f"contrastive_acc={contrastive_metrics['accuracy']:.3f} "
            f"contrastive_auroc={contrastive_metrics['auroc'] if contrastive_metrics['auroc'] is not None else 'n/a'}"
        )
    typer.echo(f"Wrote report: {output_json}")


@app.command("run-safety-pipeline")
def run_safety_pipeline(
    activations: Annotated[
        Path,
        typer.Option(
            help="Local activations leaf directory for train/calib/test split."
        ),
    ],
    shifted_activations: Annotated[
        Optional[Path],
        typer.Option(
            help="Optional shifted-domain activations for generator-shift AUROC."
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Output directory for markdown/latex tables and report."),
    ] = Path("experiments/outputs"),
    model: Annotated[
        Optional[ModelKey],
        typer.Option(
            help="Optional model key sanity check against activations metadata."
        ),
    ] = None,
    model_output: Annotated[
        Optional[Path],
        typer.Option(help="Optional trust-region model path (.pkl)."),
    ] = None,
    model_path: Annotated[
        Optional[Path],
        typer.Option(help="Existing trust-region model path (.pkl) for --only-drift."),
    ] = None,
    only_fit: Annotated[
        bool,
        typer.Option(help="Only fit trust-region model and stop."),
    ] = False,
    only_drift: Annotated[
        bool,
        typer.Option(help="Only compute drift metrics from existing model (.pkl)."),
    ] = False,
    plot_drift: Annotated[
        bool,
        typer.Option(help="Save exit-time histogram and boundary overlay plots."),
    ] = False,
    real_sap: Annotated[
        bool,
        typer.Option(help="Run real SaP baseline by cloning/invoking SafetyPolytope."),
    ] = False,
    sap_repo_path: Annotated[
        Optional[Path],
        typer.Option(help="Optional local SaP repo path (used with --real-sap)."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed for split (70/15/15).")] = 42,
) -> None:
    """Run safety pipeline: trust region, baselines, drift, and table exports."""
    from latent_dynamics.experiments.m2_m3_pipeline import (
        PipelineConfig,
        compute_drift_only,
        fit_trust_region_only,
        run_pipeline,
    )

    if only_fit and only_drift:
        raise typer.BadParameter("Use only one of --only-fit or --only-drift.")
    if only_drift and model_path is None:
        raise typer.BadParameter("--only-drift requires --model-path.")

    if only_fit:
        report = fit_trust_region_only(
            activations=activations,
            model_output=model_output,
            seed=seed,
        )
        typer.echo(json.dumps(report, indent=2, default=str))
        return

    if only_drift:
        report = compute_drift_only(
            activations=activations,
            model_path=model_path,
            seed=seed,
        )
        typer.echo(json.dumps(report, indent=2, default=str))
        return

    cfg = PipelineConfig(
        activations=activations,
        shifted_activations=shifted_activations,
        output_dir=output_dir,
        model_output=model_output,
        seed=seed,
        sap_repo_path=sap_repo_path,
        real_sap=real_sap,
        plot_drift=plot_drift,
    )
    report = run_pipeline(cfg)

    if model is not None:
        src_model = report.get("source_config", {}).get("model_key")
        if src_model is not None and src_model != model.value:
            typer.echo(
                f"Warning: provided --model={model.value} but activations model={src_model}.",
                err=True,
            )

    typer.echo(json.dumps(report, indent=2, default=str))
    typer.echo(f"Wrote report: {output_dir / 'milestone23_report.json'}")


@app.command("fit-trust-region")
def fit_trust_region_cmd(
    activations: Annotated[
        Path,
        typer.Option(help="Local activations leaf directory."),
    ],
    model_output: Annotated[
        Optional[Path],
        typer.Option(help="Optional output path for trust-region model (.pkl)."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed for split (70/15/15).")] = 42,
) -> None:
    """Fit and save trust-region model only."""
    from latent_dynamics.experiments.m2_m3_pipeline import fit_trust_region_only

    report = fit_trust_region_only(
        activations=activations,
        model_output=model_output,
        seed=seed,
    )
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("compute-drift")
def compute_drift_cmd(
    activations: Annotated[
        Path,
        typer.Option(help="Local activations leaf directory."),
    ],
    model_path: Annotated[
        Path,
        typer.Option(help="Path to trust-region model (.pkl)."),
    ],
    output_json: Annotated[
        Optional[Path],
        typer.Option(help="Optional output JSON path for drift report."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed for split (70/15/15).")] = 42,
) -> None:
    """Compute drift metrics from existing trust-region model."""
    from latent_dynamics.experiments.m2_m3_pipeline import compute_drift_only

    report = compute_drift_only(
        activations=activations,
        model_path=model_path,
        seed=seed,
    )
    text = json.dumps(report, indent=2, default=str)
    typer.echo(text)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text)
        typer.echo(f"Wrote report to {output_json}")


@app.command("milestone23")
def milestone23_backward_compat(
    activations: Annotated[
        Path,
        typer.Option(
            help="Local activations leaf directory for train/calib/test split."
        ),
    ],
    shifted_activations: Annotated[
        Optional[Path],
        typer.Option(
            help="Optional shifted-domain activations for generator-shift AUROC."
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Output directory for markdown/latex tables and report."),
    ] = Path("experiments/outputs"),
    model: Annotated[
        Optional[ModelKey],
        typer.Option(
            help="Optional model key sanity check against activations metadata."
        ),
    ] = None,
    model_output: Annotated[
        Optional[Path],
        typer.Option(help="Optional trust-region model path (.pkl)."),
    ] = None,
    model_path: Annotated[
        Optional[Path],
        typer.Option(help="Existing trust-region model path (.pkl) for --only-drift."),
    ] = None,
    only_fit: Annotated[
        bool,
        typer.Option(help="Only fit trust-region model and stop."),
    ] = False,
    only_drift: Annotated[
        bool,
        typer.Option(help="Only compute drift metrics from existing model (.pkl)."),
    ] = False,
    plot_drift: Annotated[
        bool,
        typer.Option(help="Save exit-time histogram and boundary overlay plots."),
    ] = False,
    real_sap: Annotated[
        bool,
        typer.Option(help="Run real SaP baseline by cloning/invoking SafetyPolytope."),
    ] = False,
    sap_repo_path: Annotated[
        Optional[Path],
        typer.Option(help="Optional local SaP repo path for external baseline run."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed for split (70/15/15).")] = 42,
) -> None:
    """Backward-compatible alias for run-safety-pipeline."""
    run_safety_pipeline(
        activations=activations,
        shifted_activations=shifted_activations,
        output_dir=output_dir,
        model=model,
        model_output=model_output,
        model_path=model_path,
        only_fit=only_fit,
        only_drift=only_drift,
        plot_drift=plot_drift,
        real_sap=real_sap,
        sap_repo_path=sap_repo_path,
        seed=seed,
    )


@app.command()
def list_models() -> None:
    """List available model keys."""
    for key, spec in MODEL_REGISTRY.items():
        typer.echo(f"  {key:20s}  {spec['hf_id']}")


@app.command()
def list_datasets() -> None:
    """List available dataset keys."""
    for key, spec in DATASET_REGISTRY.items():
        source = spec.path or "built-in"
        typer.echo(f"  {key:20s}  {source}")


if __name__ == "__main__":
    app()
