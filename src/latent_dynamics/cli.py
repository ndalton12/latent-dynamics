from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

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


@app.command()
def extract(
    model: Annotated[
        ModelKey, typer.Option(help="Model key from registry.")
    ] = ModelKey.gemma3_4b,
    dataset: Annotated[
        DatasetKey, typer.Option(help="Dataset key from registry.")
    ] = DatasetKey.toy_contrastive,
    layer: Annotated[
        Optional[list[int]],
        typer.Option(help="Layer index (repeat for multiple: --layer 5 --layer 10)."),
    ] = None,
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
    from latent_dynamics.activations import extract_multi_layer_trajectories
    from latent_dynamics.config import RunConfig
    from latent_dynamics.data import load_examples, prepare_text_and_labels
    from latent_dynamics.hub import (
        activation_subpath,
        save_activations,
    )
    from latent_dynamics.hub import (
        push_to_hub as _push,
    )
    from latent_dynamics.judge import SafetyJudge, judge_prompt_generations
    from latent_dynamics.models import load_model_and_tokenizer, resolve_device

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
    result = extract_multi_layer_trajectories(
        mdl,
        tokenizer,
        texts,
        layer_indices=layer_indices,
        cfg=cfg,
    )
    extracted_layers = sorted(result.per_layer.keys())
    typer.echo(f"Extracted {len(texts)} trajectories x {len(extracted_layers)} layers.")

    extra_metadata: dict[str, object] = {}
    if metadata is not None:
        extra_metadata["example_metadata"] = metadata

    if judge_generations:
        typer.echo(
            f"Judging {len(texts)} generations with {DEFAULT_GENERATION_JUDGE_MODEL}..."
        )
        judge = SafetyJudge(
            model=DEFAULT_GENERATION_JUDGE_MODEL,
            max_concurrency=12,
            batch_size=32,
            requests_per_minute=120,
            show_progress=True,
        )
        judge_results = judge_prompt_generations(
            prompts=texts,
            generations=result.generated_texts,
            judge=judge,
            cache=None,
        )
        unsafe_labels = [int(r.unsafe) for r in judge_results]
        compliance_labels = [int(r.compliance) for r in judge_results]
        confidences = [float(r.confidence) for r in judge_results]
        n_unsafe = sum(unsafe_labels)
        n_compliance = sum(compliance_labels)
        typer.echo(
            f"Judge summary: unsafe={n_unsafe} safe={len(judge_results) - n_unsafe} "
            f"compliance={n_compliance} refusal={len(judge_results) - n_compliance}"
        )
        extra_metadata["judge_model"] = DEFAULT_GENERATION_JUDGE_MODEL
        extra_metadata["judge_unsafe_labels"] = unsafe_labels
        extra_metadata["judge_compliance_labels"] = compliance_labels
        extra_metadata["judge_confidences"] = confidences

    for li in extracted_layers:
        sub = activation_subpath(cfg.dataset_key, cfg.model_key, li)
        layer_dir = output / sub
        layer_cfg = RunConfig(**{**cfg.__dict__, "layer_idx": li})
        save_activations(
            layer_dir,
            result.per_layer[li],
            texts,
            labels,
            result.token_texts,
            layer_cfg,
            generated_texts=result.generated_texts,
            extra_metadata=(extra_metadata if extra_metadata else None),
        )
        typer.echo(f"  Saved layer {li} -> {layer_dir}")

    if push_to_hub:
        url = _push(output, push_to_hub)
        typer.echo(f"Pushed to {url}")


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
