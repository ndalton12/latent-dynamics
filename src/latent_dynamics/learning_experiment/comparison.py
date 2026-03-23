from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from latent_dynamics.learning_experiment.active_learning import (
    ActiveLearningConfig,
    ActiveLearningResult,
    make_active_learning_split,
    run_active_learning_experiment_with_split,
)
from latent_dynamics.learning_experiment.data import ActivationFeatureBundle


@dataclass
class AcquisitionComparisonResult:
    base_config: dict[str, Any]
    acquisitions: list[str]
    shared_split: dict[str, Any]
    layer_metrics: dict[str, Any]
    runs: dict[str, dict[str, Any]]
    summary: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiSeedAcquisitionComparisonResult:
    base_config: dict[str, Any]
    acquisitions: list[str]
    seeds: list[int]
    layer_metrics: dict[str, Any]
    per_seed: dict[str, dict[str, Any]]
    aggregate_final: dict[str, dict[str, dict[str, float | int | None]]]
    aggregate_aulc: dict[str, dict[str, dict[str, float | int | None]]]
    aggregate_by_queried_labels: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "mean": None,
            "stddev": None,
            "stderr": None,
            "support": 0,
        }

    support = len(values)
    mean = float(sum(values) / support)
    if support <= 1:
        stddev = 0.0
        stderr = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in values) / (support - 1)
        stddev = float(math.sqrt(variance))
        stderr = float(stddev / math.sqrt(support))

    return {
        "mean": mean,
        "stddev": stddev,
        "stderr": stderr,
        "support": support,
    }


def _collect_single_seed_layer_metrics(
    runs: dict[str, dict[str, Any]],
    acquisitions: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for acquisition in acquisitions:
        out[acquisition] = runs[acquisition].get("layer_metrics")
    return out


def _aggregate_layer_metrics(
    per_seed: dict[str, dict[str, Any]],
    acquisitions: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for acquisition in acquisitions:
        auroc_values_by_layer: dict[str, list[float]] = {}
        pair_values: dict[
            tuple[int, int], dict[str, list[float]]
        ] = {}

        for payload in per_seed.values():
            run = payload["runs"][acquisition]
            layer_metrics = run.get("layer_metrics")
            if not isinstance(layer_metrics, dict):
                continue

            auroc_by_layer = layer_metrics.get("test_auroc_by_layer", {})
            if isinstance(auroc_by_layer, dict):
                for layer_key, value in auroc_by_layer.items():
                    if value is None:
                        continue
                    auroc_values_by_layer.setdefault(str(layer_key), []).append(
                        float(value)
                    )

            pair_list = layer_metrics.get("disagreement_by_layer_pair", [])
            if isinstance(pair_list, list):
                for row in pair_list:
                    if not isinstance(row, dict):
                        continue
                    layer_a = int(row["layer_a"])
                    layer_b = int(row["layer_b"])
                    key = (layer_a, layer_b)
                    bucket = pair_values.setdefault(
                        key,
                        {
                            "prediction_disagreement_rate": [],
                            "mean_abs_probability_diff": [],
                            "support_test_examples": [],
                        },
                    )
                    bucket["prediction_disagreement_rate"].append(
                        float(row["prediction_disagreement_rate"])
                    )
                    bucket["mean_abs_probability_diff"].append(
                        float(row["mean_abs_probability_diff"])
                    )
                    bucket["support_test_examples"].append(
                        float(row["support_test_examples"])
                    )

        auroc_agg = {
            layer_key: _compute_stats(values)
            for layer_key, values in sorted(auroc_values_by_layer.items(), key=lambda x: int(x[0]))
        }
        pair_agg: list[dict[str, Any]] = []
        for (layer_a, layer_b), values in sorted(pair_values.items()):
            pair_agg.append(
                {
                    "layer_a": layer_a,
                    "layer_b": layer_b,
                    "prediction_disagreement_rate": _compute_stats(
                        values["prediction_disagreement_rate"]
                    ),
                    "mean_abs_probability_diff": _compute_stats(
                        values["mean_abs_probability_diff"]
                    ),
                    "support_test_examples": _compute_stats(
                        values["support_test_examples"]
                    ),
                }
            )

        out[acquisition] = {
            "test_auroc_by_layer": auroc_agg,
            "disagreement_by_layer_pair": pair_agg,
        }

    return out


def _compute_aulc_for_metric(
    rounds: list[dict[str, Any]],
    metric_key: str,
) -> tuple[float | None, float | None]:
    pairs: list[tuple[float, float]] = []
    for row in rounds:
        value = row.get(metric_key)
        if value is None:
            continue
        pairs.append((float(row["queried_labels"]), float(value)))

    if not pairs:
        return None, None

    pairs.sort(key=lambda item: item[0])
    x = [item[0] for item in pairs]
    y = [item[1] for item in pairs]

    if len(x) == 1:
        # With a single point there is no horizontal span; use point value as normalized.
        return 0.0, float(y[0])

    area = 0.0
    for i in range(1, len(x)):
        width = x[i] - x[i - 1]
        area += 0.5 * (y[i] + y[i - 1]) * width

    span = x[-1] - x[0]
    if span <= 0:
        normalized = float(y[-1])
    else:
        normalized = float(area / span)

    return float(area), normalized


def _compute_aulc_summary(
    rounds: list[dict[str, Any]],
) -> dict[str, float | None]:
    acc_aulc, acc_aulc_norm = _compute_aulc_for_metric(
        rounds=rounds,
        metric_key="test_accuracy",
    )
    auroc_aulc, auroc_aulc_norm = _compute_aulc_for_metric(
        rounds=rounds,
        metric_key="test_auroc",
    )
    return {
        "aulc_test_accuracy": acc_aulc,
        "aulc_test_accuracy_normalized": acc_aulc_norm,
        "aulc_test_auroc": auroc_aulc,
        "aulc_test_auroc_normalized": auroc_aulc_norm,
    }


def compare_acquisition_functions(
    bundle: ActivationFeatureBundle,
    base_config: ActiveLearningConfig,
    acquisitions: list[str],
) -> AcquisitionComparisonResult:
    if not acquisitions:
        raise ValueError("acquisitions cannot be empty.")

    deduped_acquisitions = list(dict.fromkeys(acquisitions))
    split = make_active_learning_split(
        bundle=bundle,
        initial_labeled_size=base_config.initial_labeled_size,
        test_size=base_config.test_size,
        random_state=base_config.random_state,
        train_prompt_subset=base_config.train_prompt_subset,
        test_prompt_subset=base_config.test_prompt_subset,
    )

    runs: dict[str, dict[str, Any]] = {}
    summary: dict[str, dict[str, float | int | None]] = {}

    for acquisition in deduped_acquisitions:
        cfg = replace(base_config, acquisition=acquisition)
        result: ActiveLearningResult = run_active_learning_experiment_with_split(
            bundle=bundle,
            config=cfg,
            split=split,
        )
        payload = result.to_dict()
        runs[acquisition] = payload

        final_round = payload["rounds"][-1]
        summary_row: dict[str, Any] = {
            "final_queried_labels": int(final_round["queried_labels"]),
            "final_test_accuracy": float(final_round["test_accuracy"]),
            "final_test_auroc": (
                None
                if final_round["test_auroc"] is None
                else float(final_round["test_auroc"])
            ),
        }
        summary_row.update(_compute_aulc_summary(payload["rounds"]))
        stability_summary = payload.get("ranking_stability_summary")
        if stability_summary is not None:
            summary_row["ranking_stability_summary"] = stability_summary
        summary[acquisition] = summary_row

    shared_split = {
        "initial_labeled_indices": split.initial_labeled_indices.tolist(),
        "unlabeled_pool_indices": split.unlabeled_pool_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
        "initial_labeled_size": int(split.initial_labeled_indices.shape[0]),
        "unlabeled_pool_size": int(split.unlabeled_pool_indices.shape[0]),
        "test_size": int(split.test_indices.shape[0]),
        "train_prompt_subset": base_config.train_prompt_subset,
        "test_prompt_subset": base_config.test_prompt_subset,
    }
    layer_metrics = _collect_single_seed_layer_metrics(
        runs=runs,
        acquisitions=deduped_acquisitions,
    )

    return AcquisitionComparisonResult(
        base_config=asdict(base_config),
        acquisitions=deduped_acquisitions,
        shared_split=shared_split,
        layer_metrics=layer_metrics,
        runs=runs,
        summary=summary,
    )


def _aggregate_final_metrics(
    per_seed: dict[str, dict[str, Any]],
    acquisitions: list[str],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for acquisition in acquisitions:
        queried_values: list[float] = []
        accuracy_values: list[float] = []
        auroc_values: list[float] = []
        for payload in per_seed.values():
            run = payload["runs"][acquisition]
            final_round = run["rounds"][-1]
            queried_values.append(float(final_round["queried_labels"]))
            accuracy_values.append(float(final_round["test_accuracy"]))
            auroc = final_round["test_auroc"]
            if auroc is not None:
                auroc_values.append(float(auroc))

        out[acquisition] = {
            "final_queried_labels": _compute_stats(queried_values),
            "final_test_accuracy": _compute_stats(accuracy_values),
            "final_test_auroc": _compute_stats(auroc_values),
        }
    return out


def _aggregate_aulc_metrics(
    per_seed: dict[str, dict[str, Any]],
    acquisitions: list[str],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for acquisition in acquisitions:
        acc_aulc_values: list[float] = []
        acc_aulc_norm_values: list[float] = []
        auroc_aulc_values: list[float] = []
        auroc_aulc_norm_values: list[float] = []

        for payload in per_seed.values():
            run = payload["runs"][acquisition]
            aulc = _compute_aulc_summary(run["rounds"])

            if aulc["aulc_test_accuracy"] is not None:
                acc_aulc_values.append(float(aulc["aulc_test_accuracy"]))
            if aulc["aulc_test_accuracy_normalized"] is not None:
                acc_aulc_norm_values.append(float(aulc["aulc_test_accuracy_normalized"]))
            if aulc["aulc_test_auroc"] is not None:
                auroc_aulc_values.append(float(aulc["aulc_test_auroc"]))
            if aulc["aulc_test_auroc_normalized"] is not None:
                auroc_aulc_norm_values.append(float(aulc["aulc_test_auroc_normalized"]))

        out[acquisition] = {
            "aulc_test_accuracy": _compute_stats(acc_aulc_values),
            "aulc_test_accuracy_normalized": _compute_stats(acc_aulc_norm_values),
            "aulc_test_auroc": _compute_stats(auroc_aulc_values),
            "aulc_test_auroc_normalized": _compute_stats(auroc_aulc_norm_values),
        }
    return out


def _aggregate_by_queried_labels(
    per_seed: dict[str, dict[str, Any]],
    acquisitions: list[str],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}

    for acquisition in acquisitions:
        by_q: dict[int, dict[str, list[float]]] = {}
        for payload in per_seed.values():
            rounds = payload["runs"][acquisition]["rounds"]
            for row in rounds:
                q = int(row["queried_labels"])
                bucket = by_q.setdefault(
                    q,
                    {
                        "test_accuracy": [],
                        "test_auroc": [],
                        "test_positive_rate": [],
                    },
                )
                bucket["test_accuracy"].append(float(row["test_accuracy"]))
                bucket["test_positive_rate"].append(float(row["test_positive_rate"]))
                if row["test_auroc"] is not None:
                    bucket["test_auroc"].append(float(row["test_auroc"]))

        aggregated_rows: list[dict[str, Any]] = []
        for q in sorted(by_q):
            bucket = by_q[q]
            aggregated_rows.append(
                {
                    "queried_labels": q,
                    "test_accuracy": _compute_stats(bucket["test_accuracy"]),
                    "test_auroc": _compute_stats(bucket["test_auroc"]),
                    "test_positive_rate": _compute_stats(bucket["test_positive_rate"]),
                }
            )
        out[acquisition] = aggregated_rows

    return out


def compare_acquisition_functions_multi_seed(
    bundle: ActivationFeatureBundle,
    base_config: ActiveLearningConfig,
    acquisitions: list[str],
    seeds: list[int],
) -> MultiSeedAcquisitionComparisonResult:
    if not acquisitions:
        raise ValueError("acquisitions cannot be empty.")
    if not seeds:
        raise ValueError("seeds cannot be empty.")

    deduped_acquisitions = list(dict.fromkeys(acquisitions))
    deduped_seeds = list(dict.fromkeys(int(s) for s in seeds))

    per_seed: dict[str, dict[str, Any]] = {}
    for seed in deduped_seeds:
        cfg = replace(base_config, random_state=seed)
        single_seed_result = compare_acquisition_functions(
            bundle=bundle,
            base_config=cfg,
            acquisitions=deduped_acquisitions,
        )
        per_seed[str(seed)] = single_seed_result.to_dict()

    aggregate_final = _aggregate_final_metrics(
        per_seed=per_seed,
        acquisitions=deduped_acquisitions,
    )
    aggregate_aulc = _aggregate_aulc_metrics(
        per_seed=per_seed,
        acquisitions=deduped_acquisitions,
    )
    layer_metrics_per_seed = {
        seed: payload.get("layer_metrics", {})
        for seed, payload in per_seed.items()
    }
    layer_metrics_aggregate = _aggregate_layer_metrics(
        per_seed=per_seed,
        acquisitions=deduped_acquisitions,
    )
    aggregate_by_queried_labels = _aggregate_by_queried_labels(
        per_seed=per_seed,
        acquisitions=deduped_acquisitions,
    )

    return MultiSeedAcquisitionComparisonResult(
        base_config=asdict(base_config),
        acquisitions=deduped_acquisitions,
        seeds=deduped_seeds,
        layer_metrics={
            "per_seed": layer_metrics_per_seed,
            "aggregate": layer_metrics_aggregate,
        },
        per_seed=per_seed,
        aggregate_final=aggregate_final,
        aggregate_aulc=aggregate_aulc,
        aggregate_by_queried_labels=aggregate_by_queried_labels,
    )
