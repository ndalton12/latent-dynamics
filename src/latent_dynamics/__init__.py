"""Latent trajectory analysis for language models."""

from latent_dynamics.activations import (
    build_feature_matrix,
    extract_hidden_trajectories,
    extract_multi_layer_trajectories,
    pool_trajectory,
)
from latent_dynamics.analysis import (
    drift_curve,
    fit_trust_region,
    lat_scan,
    make_direction,
    train_linear_probe,
)
from latent_dynamics.config import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    RunConfig,
)
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.hub import (
    activation_subpath,
    load_activations,
    pull_from_hub,
    push_to_hub,
    save_activations,
)
from latent_dynamics.models import load_model_and_tokenizer, resolve_device
from latent_dynamics.viz import plot_drift_curves, plot_lat_scans

__all__ = [
    "RunConfig",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "activation_subpath",
    "build_feature_matrix",
    "drift_curve",
    "extract_hidden_trajectories",
    "extract_multi_layer_trajectories",
    "fit_trust_region",
    "lat_scan",
    "load_activations",
    "load_examples",
    "load_model_and_tokenizer",
    "make_direction",
    "plot_drift_curves",
    "plot_lat_scans",
    "pool_trajectory",
    "prepare_text_and_labels",
    "pull_from_hub",
    "push_to_hub",
    "resolve_device",
    "save_activations",
    "train_linear_probe",
]
