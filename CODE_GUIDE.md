# Latent Dynamics Code Guide

This guide covers the main code paths in `src/latent_dynamics`.

Excluded on purpose:
- `src/latent_dynamics/experiments/`
- `src/latent_dynamics/dynamics.py`
- `src/latent_dynamics/drift.py`

## 1. High-level flow

Typical extraction + analysis flow:
1. `cli.py::extract(...)`
2. `data.py::load_examples(...)` + `prepare_text_and_labels(...)`
3. `models.py::load_model_and_tokenizer(...)`
4. `activations.py::extract_multi_layer_trajectories(...)`
5. `hub.py::save_activations(...)`
6. Optional judging with `judge.py::SafetyJudge` + `judge_prompt_generations(...)`
7. Optional analysis/visualization via `rep_eng.py` and `viz.py`

## 2. Entry points

## `src/latent_dynamics/__main__.py`
- Calls the Typer app from `cli.py`.
- Running `python -m latent_dynamics` is equivalent to CLI usage.

## `src/latent_dynamics/cli.py`
Primary CLI surface.

Key commands:
- `extract(...)`
  - Main extraction pipeline.
  - Supports single/multi-layer extraction, generation mode, optional 4-bit loading.
  - New optional judging toggle: `--judge-generations`.
  - If enabled, generation outputs are judged with `gpt-5-mini` and labels are saved in metadata.
- `run_safety_pipeline(...)`, `fit_trust_region_cmd(...)`, `compute_drift_cmd(...)`, `milestone23_backward_compat(...)`
  - Pipeline wrappers (internals live in excluded modules/experiments).
- `list_models()` and `list_datasets()`
  - Print registries from `config.py`.

Important constant:
- `DEFAULT_GENERATION_JUDGE_MODEL = "gpt-5-mini"`

## 3. Config and data loading

## `src/latent_dynamics/config.py`
Core configuration and registries.

Key items:
- `RunConfig`
  - Runtime settings for extraction and downstream analysis.
  - Includes generation settings (`use_generate`, `max_new_tokens`, `temperature`, etc.).
- `MODEL_REGISTRY`
  - Maps model key to HF id and dtype.
- `DATASET_REGISTRY` + `DatasetSpec`
  - Defines dataset source path/split and labeling strategy.
- `DEFAULT_LAYERS`
  - Default layer index per model family.

## `src/latent_dynamics/data.py`
Dataset ingestion and preprocessing.

Key functions:
- `load_examples(dataset_key, max_samples=None, stratify_labels=False)`
  - Loads dataset from HF or built-in toy dataset.
  - Optional random or stratified downsampling.
- `prepare_text_and_labels(ds, text_field, label_field=None, label_fn=None)`
  - Extracts text list and optional label array.

Supporting helpers:
- `_balanced_sample_by_label(...)` for stratified sample selection.

## 4. Model loading

## `src/latent_dynamics/models.py`
Model/device utilities.

Key functions:
- `resolve_device(preferred=None)`
  - Chooses `cuda`, then `mps`, then `cpu` unless overridden.
- `load_model_and_tokenizer(model_key, device, load_in_4bit=False)`
  - Loads tokenizer + causal LM.
  - Optional 4-bit path with bitsandbytes (CUDA only).

## 5. Activation extraction

## `src/latent_dynamics/activations.py`
Hidden-state trajectory extraction logic.

Key dataclass:
- `ExtractionResult`
  - `per_layer`: `dict[layer_idx, list[np.ndarray]]`
  - `token_texts`, `input_prompts`, `generated_texts`

Key functions:
- `extract_multi_layer_trajectories(model, tokenizer, texts, layer_indices, cfg)`
  - Main extraction function.
  - Handles batched/single-pass inference and optional generation mode.
- `extract_hidden_trajectories(...)`
  - Single-layer wrapper over `extract_multi_layer_trajectories`.
- `generate_full_sequence(...)`
  - Central generation helper used during extraction.
- `pool_trajectory(traj, mode="last")`
  - Pools token-level trajectory into one vector.
- `build_feature_matrix(trajectories, pooling)`
  - Applies pooling across all trajectories into `(N, D)` matrix.

## 6. Activation persistence and Hub IO

## `src/latent_dynamics/hub.py`
Storage format and Hugging Face helpers.

Key functions:
- `activation_subpath(dataset_key, model_key, layer_idx)`
  - Canonical `dataset/model/layer_N` path.
- `save_activations(...)`
  - Saves tensors to `trajectories.safetensors` and metadata to `metadata.json`.
  - Supports `extra_metadata` for additional metadata fields.
- `load_activations(input_dir)`
  - Loads trajectories + metadata and reconstructs `RunConfig`.
- `push_to_hub(...)` / `pull_from_hub(...)`
  - Upload/download helper functions for dataset repos.

## 7. Shared activation path utilities

## `src/latent_dynamics/utils.py`
Small shared helpers used by multiple modules.

Key functions:
- `is_activation_leaf(path)`
  - Checks for `metadata.json` + `trajectories.safetensors`.
- `resolve_activation_leaf(root_or_leaf)`
  - Resolves a directory to an activation leaf.
- `load_activation_bundle(...)`
  - Unified loader for local path or Hub path.
  - Returns trajectories, text fields, labels, config, and resolved leaf path.

## 8. Judging

## `src/latent_dynamics/judge.py`
LLM-as-judge utilities with structured outputs.

Key types:
- `JudgeResult`
  - `unsafe`, `compliance`, `confidence`, `rationale`
- `JudgeCache`
  - JSON cache keyed by hash(prompt + completion)

Key functions/classes:
- `SafetyJudge`
  - Wraps Flashlite with templated prompt `prompts/judge/safety_judge.jinja`.
  - Uses Pydantic structured output for robust response parsing.
- `judge_texts(pairs, judge, cache=None)`
  - Batch judging with optional caching.
- `judge_prompt_generations(prompts, generations, judge, cache=None)`
  - Convenience for activation metadata-aligned prompt/generation judging.
- `judge_activation_metadata(metadata, judge, cache=None)`
  - Accepts metadata dict directly and judges from it.
- `stable_text_hash(...)` and `judge_cache_key(...)`
  - Stable cache-key helpers.

CLI smoke-test support exists under `if __name__ == "__main__":`.

## 9. Representation engineering baselines

## `src/latent_dynamics/rep_eng.py`
Concept-direction baseline readers.

Core abstractions:
- `Reader` (abstract base)
  - `train(...)` and `predict(...)`

Concrete readers:
- `DifferenceInMeanReader`
- `PCAReader`
- `LinearProbeReader`

Shared internal helpers:
- `_to_feature_matrix(...)`
- `_as_binary_labels(...)`

Script mode:
- Has a smoke test that loads activations (via `utils.load_activation_bundle`) and reports baseline metrics.

## 10. Visualization

## `src/latent_dynamics/viz.py`
Visualization and layer-scan utilities.

Core plotting functions:
- `plot_concept_direction_over_time(...)`
- `plot_pca_subspace(...)`
- `plot_layer_lat_heatmap(...)`
- `plot_drift_curves(...)` (function exists here, details rely on excluded drift concepts)

Layer-scan helpers:
- `compute_layer_lat_scans(...)`
- `LayerScanResult`
- `_load_multi_layer_bundle(...)`

Script mode:
- Can run layerwise LAT scan end-to-end and write output figures.

## 11. Public package exports

## `src/latent_dynamics/__init__.py`
- Re-exports many symbols as package-level API.
- Intended as convenience import surface (`from latent_dynamics import ...`).

## 12. Useful extension points

If you are adding features, these are the easiest hooks:
- New model: `MODEL_REGISTRY` in `config.py`
- New dataset: `DATASET_REGISTRY` + optional label function in `config.py`
- New metadata fields: `extra_metadata` in `hub.save_activations(...)`
- New judge schema/logic: Pydantic model + prompt template in `judge.py` and `prompts/judge/`
- New reader baseline: subclass `Reader` in `rep_eng.py`
- New plots: add function in `viz.py` and expose from CLI or script mode
