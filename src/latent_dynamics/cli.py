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
    max_new_tokens: Annotated[
        int, typer.Option(help="Tokens to generate (requires --use-generate).")
    ] = 128,
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
) -> None:
    """Extract hidden-state trajectories from a model and dataset, save to disk.

    Activations are stored under a structured path:
      {output}/{dataset}/{model}/layer_{N}/
    Each layer directory contains a metadata.json that records the layer index,
    input prompts, and generated text (when using --use-generate) so that
    activations can be matched back to their source data.
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
    from latent_dynamics.models import load_model_and_tokenizer, resolve_device

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
    )

    typer.echo(f"Device: {cfg.device}")
    typer.echo(f"Model:  {cfg.model_key}")
    typer.echo(f"Dataset: {cfg.dataset_key} (max_samples={cfg.max_samples})")
    typer.echo(f"Layers:  {'all' if layer_indices is None else layer_indices}")

    ds, spec = load_examples(cfg.dataset_key, cfg.max_samples)
    texts, labels = prepare_text_and_labels(
        ds,
        text_field=spec.text_field,
        label_field=spec.label_field,
        label_fn=spec.label_fn,
    )

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
        )
        typer.echo(f"  Saved layer {li} -> {layer_dir}")

    if push_to_hub:
        url = _push(output, push_to_hub)
        typer.echo(f"Pushed to {url}")


@app.command()
def analyze(
    activations: Annotated[
        Optional[Path],
        typer.Option(
            help="Local activations directory (leaf path with metadata.json)."
        ),
    ] = None,
    from_hub: Annotated[
        Optional[str],
        typer.Option(help="HuggingFace Hub repo id to pull activations from."),
    ] = None,
    hub_dataset: Annotated[
        Optional[str],
        typer.Option(help="Dataset key for Hub path resolution (with --from-hub)."),
    ] = None,
    hub_model: Annotated[
        Optional[str],
        typer.Option(help="Model key for Hub path resolution (with --from-hub)."),
    ] = None,
    hub_layer: Annotated[
        Optional[int],
        typer.Option(help="Layer index for Hub path resolution (with --from-hub)."),
    ] = None,
    direction: Annotated[
        DirectionMethod, typer.Option(help="Direction method for LAT scan.")
    ] = DirectionMethod.probe_weight,
    pooling: Annotated[
        PoolingMode, typer.Option(help="Trajectory pooling mode for feature matrix.")
    ] = PoolingMode.last,
    test_size: Annotated[float, typer.Option(help="Fraction for test split.")] = 0.25,
    calib_size: Annotated[
        float, typer.Option(help="Fraction for calibration split.")
    ] = 0.25,
    trust_region: Annotated[
        bool,
        typer.Option(
            "--trust-region/--no-trust-region",
            help="Compute and plot trust-region drift.",
        ),
    ] = True,
    save_plots: Annotated[
        Optional[Path], typer.Option(help="Directory to save HTML plots.")
    ] = None,
    max_traces: Annotated[int, typer.Option(help="Max traces in each plot.")] = 16,
    random_state: Annotated[int, typer.Option(help="Random seed.")] = 7,
) -> None:
    """Analyze stored activation trajectories: train probe, LAT scan, visualize.

    Use --activations to point to a local leaf directory, or --from-hub with
    --hub-dataset, --hub-model, --hub-split, --hub-layer to pull a specific
    slice from HuggingFace Hub.
    """
    from latent_dynamics.activations import build_feature_matrix
    from latent_dynamics.analysis import (
        drift_curve,
        fit_trust_region,
        lat_scan,
        make_direction,
        train_linear_probe,
    )
    from latent_dynamics.hub import (
        activation_subpath,
        load_activations,
        pull_from_hub,
    )
    from latent_dynamics.viz import plot_drift_curves, plot_lat_scans

    if activations is None and from_hub is None:
        typer.echo("Provide --activations or --from-hub.", err=True)
        raise typer.Exit(code=1)

    if from_hub and activations is None:
        for name, val in [
            ("--hub-dataset", hub_dataset),
            ("--hub-model", hub_model),
            ("--hub-layer", hub_layer),
        ]:
            if val is None:
                typer.echo(f"{name} is required when using --from-hub.", err=True)
                raise typer.Exit(code=1)

        assert hub_dataset and hub_model and hub_layer is not None
        sub = activation_subpath(hub_dataset, hub_model, hub_layer)
        activations = Path(f".cache/hub/{from_hub.replace('/', '__')}") / sub
        typer.echo(f"Pulling from Hub: {from_hub} / {sub}")
        pull_from_hub(from_hub, activations, path_in_repo=str(sub))

    assert activations is not None
    trajectories, texts, labels, token_texts, _generated, cfg = load_activations(
        activations
    )
    typer.echo(f"Loaded {len(trajectories)} trajectories from {activations}")
    typer.echo(
        f"  model={cfg.model_key}  dataset={cfg.dataset_key}  layer={cfg.layer_idx}"
    )

    if labels is None:
        typer.echo(
            "No labels found -- cannot train probe or compute directions.", err=True
        )
        raise typer.Exit(code=1)

    X = build_feature_matrix(trajectories, pooling.value)
    typer.echo(f"Feature matrix shape: {X.shape}")

    probe, (X_train, y_train), (X_calib, y_calib), (X_test, y_test), metrics = (
        train_linear_probe(X, labels, test_size, calib_size, random_state)
    )
    typer.echo(
        f"Probe accuracy: {metrics['accuracy']:.3f}  ROC-AUC: {metrics['roc_auc']:.3f}"
    )
    typer.echo(metrics["report"])

    dir_vec = make_direction(direction.value, X_train, y_train, probe)
    scans = lat_scan(trajectories, dir_vec)

    scan_path = (save_plots / "lat_scans") if save_plots else None
    fig_scan = plot_lat_scans(
        scans,
        labels,
        max_traces=max_traces,
        title=f"LAT scan — {cfg.model_key} / {cfg.dataset_key} / L{cfg.layer_idx} / {direction.value}",
        save_path=scan_path,
    )
    if save_plots:
        typer.echo(f"Saved LAT scan plot to {scan_path}.html")
    else:
        fig_scan.show()

    if trust_region:
        try:
            safe_center, tau = fit_trust_region(X_calib, y_calib)
        except ValueError as e:
            typer.echo(f"Skipping trust region: {e}", err=True)
            return

        drifts = [drift_curve(t, safe_center) for t in trajectories]

        drift_path = (save_plots / "drift_curves") if save_plots else None
        fig_drift = plot_drift_curves(
            drifts,
            labels,
            tau,
            max_traces=max_traces,
            title=f"Trust-region drift — {cfg.model_key} / {cfg.dataset_key} / L{cfg.layer_idx}",
            save_path=drift_path,
        )
        if save_plots:
            typer.echo(f"Saved drift plot to {drift_path}.html")
        else:
            fig_drift.show()


@app.command("qd-active")
def qd_active(
    model: Annotated[
        ModelKey, typer.Option(help="Target model key for rollout latent extraction.")
    ] = ModelKey.gemma3_4b,
    layer: Annotated[
        int, typer.Option(help="Layer index for latent trajectories.")
    ] = 5,
    label_budget: Annotated[int, typer.Option(help="Total labeling budget.")] = 1000,
    warm_start_labels: Annotated[
        int, typer.Option(help="Initial labeled pool size before active learning.")
    ] = 128,
    batch_size: Annotated[
        int, typer.Option(help="Labels acquired per iteration.")
    ] = 32,
    candidate_pool_size: Annotated[
        int, typer.Option(help="Candidate prompts generated per iteration.")
    ] = 128,
    rollout_batch_size: Annotated[
        int, typer.Option(help="Prompt rollout extraction batch size.")
    ] = 16,
    proxy_model: Annotated[
        str, typer.Option(help="Proxy model used by flashlite for mutation/judging.")
    ] = "gpt-5-mini",
    proxy_batch_size: Annotated[
        int, typer.Option(help="flashlite request batch size per complete_many call.")
    ] = 32,
    template_dir: Annotated[
        Optional[Path],
        typer.Option(help="Optional flashlite Jinja template directory override."),
    ] = None,
    output_root: Annotated[
        Path, typer.Option(help="Directory root for run artifacts.")
    ] = Path(".cache/qd_active"),
    output_json: Annotated[
        Optional[Path], typer.Option(help="Optional path to write final JSON report.")
    ] = None,
    device: Annotated[
        Optional[str], typer.Option(help="Device override (cuda/mps/cpu).")
    ] = None,
    show_progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Enable or disable tqdm progress bars.",
        ),
    ] = True,
    random_state: Annotated[int, typer.Option(help="Random seed.")] = 7,
) -> None:
    """Run QD-driven active boundary learning with Bayesian logistic probes."""
    from latent_dynamics.experiments.qd_active_boundary_experiment import (
        ExperimentConfig,
        run_experiment,
    )

    cfg = ExperimentConfig(
        model_key=model.value,
        layer_idx=layer,
        label_budget=label_budget,
        warm_start_labels=warm_start_labels,
        batch_size=batch_size,
        candidate_pool_size=candidate_pool_size,
        rollout_batch_size=rollout_batch_size,
        proxy_model=proxy_model,
        proxy_batch_size=proxy_batch_size,
        template_dir=template_dir,
        output_root=output_root,
        output_json=output_json,
        device=device,
        show_progress=show_progress,
        random_state=random_state,
    )
    report = run_experiment(cfg)
    text = json.dumps(report, indent=2, default=str)
    typer.echo(text)
    if output_json is not None:
        typer.echo(f"Wrote report to {output_json}")


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
