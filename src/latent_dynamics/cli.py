from __future__ import annotations

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
    split: Annotated[str, typer.Option(help="Dataset split.")] = "train",
    layer: Annotated[
        Optional[list[int]],
        typer.Option(help="Layer index (repeat for multiple: --layer 5 --layer 10)."),
    ] = None,
    max_samples: Annotated[int, typer.Option(help="Maximum number of examples.")] = 120,
    max_length: Annotated[
        int, typer.Option(help="Maximum token sequence length.")
    ] = 256,
    output: Annotated[
        Path, typer.Option(help="Root output directory. Activations are saved under {dataset}/{split}/{model}/layer_{N}/.")
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
) -> None:
    """Extract hidden-state trajectories from a model and dataset, save to disk.

    Activations are stored under a structured path:
      {output}/{dataset}/{split}/{model}/layer_{N}/
    This layout is mirrored when pushing to HuggingFace Hub.
    """
    from latent_dynamics.activations import extract_multi_layer_trajectories
    from latent_dynamics.config import RunConfig
    from latent_dynamics.data import load_examples, prepare_text_and_labels
    from latent_dynamics.hub import (
        activation_subpath,
        push_to_hub as _push,
        save_activations,
    )
    from latent_dynamics.models import load_model_and_tokenizer, resolve_device

    layer_indices = layer if layer else [5]

    cfg = RunConfig(
        model_key=model.value,
        dataset_key=dataset.value,
        split=split,
        max_samples=max_samples,
        max_length=max_length,
        layer_idx=layer_indices[0],
        device=resolve_device(device),
        use_generate=use_generate,
        max_new_tokens=max_new_tokens,
        include_prompt_in_trajectory=include_prompt,
    )

    typer.echo(f"Device: {cfg.device}")
    typer.echo(f"Model:  {cfg.model_key}")
    typer.echo(
        f"Dataset: {cfg.dataset_key} (split={cfg.split}, max_samples={cfg.max_samples})"
    )
    typer.echo(f"Layers:  {layer_indices}")

    ds, spec = load_examples(cfg.dataset_key, cfg.split, cfg.max_samples)
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
    mdl, tokenizer = load_model_and_tokenizer(cfg.model_key, cfg.device)

    typer.echo(f"Extracting trajectories for layers {layer_indices} (single forward pass)...")
    per_layer, token_texts = extract_multi_layer_trajectories(
        mdl, tokenizer, texts,
        layer_indices=layer_indices,
        max_length=cfg.max_length,
        device=cfg.device,
        cfg=cfg,
    )
    typer.echo(f"Extracted {len(texts)} trajectories x {len(layer_indices)} layers.")

    for li in layer_indices:
        sub = activation_subpath(cfg.dataset_key, cfg.split, cfg.model_key, li)
        layer_dir = output / sub
        layer_cfg = RunConfig(**{**cfg.__dict__, "layer_idx": li})
        save_activations(layer_dir, per_layer[li], texts, labels, token_texts, layer_cfg)
        typer.echo(f"  Saved layer {li} -> {layer_dir}")

    if push_to_hub:
        url = _push(output, push_to_hub)
        typer.echo(f"Pushed to {url}")


@app.command()
def analyze(
    activations: Annotated[
        Optional[Path],
        typer.Option(help="Local activations directory (leaf path with metadata.json)."),
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
    hub_split: Annotated[
        Optional[str],
        typer.Option(help="Split for Hub path resolution (with --from-hub)."),
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
            ("--hub-split", hub_split),
            ("--hub-layer", hub_layer),
        ]:
            if val is None:
                typer.echo(f"{name} is required when using --from-hub.", err=True)
                raise typer.Exit(code=1)

        assert hub_dataset and hub_model and hub_split and hub_layer is not None
        sub = activation_subpath(hub_dataset, hub_split, hub_model, hub_layer)
        activations = Path(f".cache/hub/{from_hub.replace('/', '__')}") / sub
        typer.echo(f"Pulling from Hub: {from_hub} / {sub}")
        pull_from_hub(from_hub, activations, path_in_repo=str(sub))

    assert activations is not None
    trajectories, texts, labels, token_texts, cfg = load_activations(activations)
    typer.echo(f"Loaded {len(trajectories)} trajectories from {activations}")
    typer.echo(f"  model={cfg.model_key}  dataset={cfg.dataset_key}  layer={cfg.layer_idx}")

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
