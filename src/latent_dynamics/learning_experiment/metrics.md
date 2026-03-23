# Learning Experiment Metrics

This document describes the metrics produced by the active-learning experiments in this folder.

## Per-round metrics

Each round logs these values on the untouched test set:

- `queried_labels`: cumulative number of labels acquired so far.
- `test_accuracy`: fraction of correct binary predictions.
- `test_auroc`: area under ROC curve from model scores (can be `null` if test set has only one class).
- `test_positive_rate`: fraction of positive labels in the test split.
- `subgroup_metrics` (optional): same metrics within prompt groups (for example `vanilla` / `adversarial`).

## Final metrics (single seed)

For each acquisition strategy, the comparison summary reports:

- `final_queried_labels`
- `final_test_accuracy`
- `final_test_auroc`

These are taken from the last round in the run.

## AULC

`AULC` = Area Under the Learning Curve.  
The x-axis is `queried_labels`; the y-axis is a test metric (accuracy or AUROC).

### AULC computation

For a metric `m`, collect points from rounds:

- `x_i = queried_labels_i`
- `y_i = m_i` (skip rows where `m_i` is `null`)

Sort points by `x_i`, then compute trapezoidal area:

`AULC = sum_{i=1..k-1} 0.5 * (y_i + y_{i-1}) * (x_i - x_{i-1})`

If only one valid point exists, raw AULC is set to `0.0` (no horizontal span).

### AULC normalization

Let `span = x_last - x_first`.

- If `span > 0`:  
  `AULC_normalized = AULC / span`
- If `span <= 0`:  
  `AULC_normalized = y_last`

Interpretation: normalized AULC is the average metric value across the queried-label range, making runs with same x-range directly comparable.

### Reported AULC fields

Per acquisition (single seed summary):

- `aulc_test_accuracy`
- `aulc_test_accuracy_normalized`
- `aulc_test_auroc`
- `aulc_test_auroc_normalized`

## Dynamic ranking-stability metrics

These are computed for dynamic acquisition methods (`dynamic_*`) only.
For non-dynamic methods they are `null` / absent in summaries.

Per round (`rounds[*].ranking_stability`):

- `top_k_overlap_at_k`:
  overlap between consecutive rounds' ranked top-k sets on the shared pool.
  - `k = min(batch_size, |shared_pool_t|, |shared_pool_{t-1}|)`
  - value in `[0, 1]`
- `spearman_rho_shared_pool`:
  Spearman rank correlation between consecutive-round rankings over shared pool items.
- `score_drift_mae_shared_pool`:
  mean absolute change in acquisition score over shared pool items.
  - score is rule-dependent:
    - `uncertainty`: `-|decision_score|`
    - `disagreement`: variance of per-layer probabilities
    - `uncertainty_diversity`: base uncertainty score (`-|decision_score|`)
- `selected_set_turnover`:
  fraction of newly selected items relative to previous round's selected batch:
  - `1 - |B_t ∩ B_{t-1}| / |B_t|`
  - first queried round has `null` turnover.
- `k_for_top_k_overlap`: effective `k` used for top-k overlap.
- `shared_pool_size`: number of items shared between consecutive rounds.

Run-level summary (`ranking_stability_summary`):

- `mean_top_k_overlap_at_k`
- `mean_spearman_rho_shared_pool`
- `mean_score_drift_mae_shared_pool`
- `mean_selected_set_turnover`
- corresponding `support_*` counts for each metric.

## Multi-seed aggregation metrics

When running comparison with multiple seeds, aggregate stats are reported for:

- final metrics (`aggregate_final`)
- AULC metrics (`aggregate_aulc`)
- each queried-label point on the learning curve (`aggregate_by_queried_labels`)

For a list of values `v_1 ... v_n`:

- `mean = (1/n) * sum(v_i)`
- `stddev` = sample standard deviation (denominator `n - 1`), `0.0` if `n <= 1`
- `stderr = stddev / sqrt(n)`, `0.0` if `n <= 1`
- `support = n` (number of seeds contributing non-null values)

Note: for AUROC-like fields, seeds where AUROC is `null` are excluded from that AUROC aggregate support.

## Layer metrics (separate root key)

Comparison outputs include a top-level `layer_metrics` key.

### Single-seed comparison

`layer_metrics[acquisition]` contains:

- `test_auroc_by_layer`: AUROC on test set for each layer probe.
- `disagreement_by_layer_pair`: list over layer pairs with:
  - `prediction_disagreement_rate`: fraction of test examples where pair predictions differ
  - `mean_abs_probability_diff`: mean absolute difference in predicted positive probability
  - `support_test_examples`: test-set size used

### Multi-seed comparison

Top-level structure:

- `layer_metrics.per_seed[seed][acquisition]`: raw per-seed layer metrics
- `layer_metrics.aggregate[acquisition]`: aggregated stats

Aggregated stats use the same `mean/stddev/stderr/support` convention as other aggregates:

- `test_auroc_by_layer[layer] -> stats`
- `disagreement_by_layer_pair[*].prediction_disagreement_rate -> stats`
- `disagreement_by_layer_pair[*].mean_abs_probability_diff -> stats`
- `disagreement_by_layer_pair[*].support_test_examples -> stats`

## Notes

- Final metrics can look similar across methods at large budgets; AULC is often more informative because it rewards faster gains at lower label counts.
- In comparison mode, all acquisition methods share the same split per seed (`L0`, `U0`, `test`) so comparisons are controlled within each seed.
