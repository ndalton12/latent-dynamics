"""Latent trajectory analysis for language models."""

from latent_dynamics.activations import (
    ExtractionResult,
    build_feature_matrix,
    extract_hidden_trajectories,
    extract_multi_layer_trajectories,
    pool_trajectory,
)
from latent_dynamics.config import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    RunConfig,
)
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.drift import (
    compute_drift_metrics,
    evaluate_generator_shift,
    first_exit_time,
)
from latent_dynamics.dynamics import (
    TrustRegionModel,
    baseline_brt_align_scores,
    baseline_nglare_scores,
    baseline_sap_proxy_scores,
    fit_trust_region_model,
    save_trust_region_model,
    split_indices_70_15_15,
    trust_region_scores,
)
from latent_dynamics.hub import (
    activation_subpath,
    load_activations,
    pull_from_hub,
    push_to_hub,
    save_activations,
)
from latent_dynamics.judge import (
    JudgeCache,
    JudgeResult,
    SafetyJudge,
    judge_cache_key,
    judge_texts,
    stable_text_hash,
)
from latent_dynamics.models import load_model_and_tokenizer, resolve_device
from latent_dynamics.viz import (
    plot_drift_curves,
    plot_layer_lat_heatmap,
    plot_pca_subspace,
)

__all__ = [
    "RunConfig",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "activation_subpath",
    "build_feature_matrix",
    "drift_curve",
    "first_exit_time",
    "compute_drift_metrics",
    "evaluate_generator_shift",
    "ExtractionResult",
    "extract_hidden_trajectories",
    "extract_multi_layer_trajectories",
    "fit_trust_region",
    "fit_trust_region_model",
    "lat_scan",
    "TrustRegionModel",
    "baseline_brt_align_scores",
    "baseline_nglare_scores",
    "baseline_sap_proxy_scores",
    "load_activations",
    "load_examples",
    "load_model_and_tokenizer",
    "make_direction",
    "plot_drift_curves",
    "plot_lat_scans",
    "plot_layer_lat_heatmap",
    "plot_pca_subspace",
    "pool_trajectory",
    "prepare_text_and_labels",
    "pull_from_hub",
    "push_to_hub",
    "resolve_device",
    "save_activations",
    "save_trust_region_model",
    "split_indices_70_15_15",
    "trust_region_scores",
    "train_linear_probe",
    "BenignManifold",
    "compute_prefix_signatures",
    "compute_turning_angles",
    "fit_benign_manifold",
    "fit_trajectory_pca",
    "reduce_trajectories",
    "signature_prefix_score_map",
    "turning_angle_score_map",
    "JudgeCache",
    "JudgeResult",
    "SafetyJudge",
    "judge_cache_key",
    "judge_texts",
    "stable_text_hash",
]
