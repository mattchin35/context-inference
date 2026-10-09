# Inter-regional neural regression

This package runs reproducible, bidirectional PFC/HPC prediction analyses from
one validated metadata-v2 neural session. The stable workflow supports held-out
OLS for direct units and fold-local regional PCs plus target-wise held-out
Poisson GLMs for direct-unit counts. Independent descriptive stages fit
in-sample linear and Poisson Granger-style restricted/full comparisons without
requiring CV folds or their corresponding CV stages.

## Module map

- `configuration.py`: immutable scientific configuration and explicit PFC/HPC
  population contracts.
- `preparation.py`: Pynapple spike-count tensors, scientific masks, block folds,
  windows, and within-trial history matrices.
- `pca.py`: training-fold-only regional standardization and unwhitened PCA.
- `linear.py`: unpenalized OLS, held-out scores, and complete-fold summaries.
- `poisson.py`: target-wise unpenalized Poisson fitting, deviance scoring, and
  derived matched OLS/Poisson count-MSE comparisons.
- `granger.py`: matched-row nested-fit checks plus linear log residual-variance
  ratios and Poisson likelihood/deviance improvements.
- `pipeline.py`: in-memory preparation and bidirectional unit/PC orchestration.
- `records.py`: exact array and seven-table result schemas.
- `persistence.py`: content fingerprints, manifests, trusted result loading, and
  atomic immutable run directories.
- `plotting.py`: saved-table-only light-mode figures.
- `resource_usage.py`: append-only process/cgroup telemetry and completion
  summaries for long offline runs.
- `run_session.py` and `run_batch.py`: offline composition roots. Batch workers
  receive configuration paths and remain session-separated.

The dependency direction is configuration/records, then preparation and model
helpers, then pipeline, persistence/plotting, and finally command or webapp
presentation. The webapp imports saved results only and never starts analyses.

## Axes and units

- Regional count tensors: `(trial, whole_window_bin, unit)`, values are integer
  spike counts per configured bin.
- PCA fitting pool: training `(trial, whole_window_bin)` axes are flattened to
  observations; unit columns retain qualified IDs.
- History rows: `(trial_row, target_bin_position)` identities are preserved and
  fingerprinted. Feature order is most-recent lag first, then stable feature
  order.
- Times and bin edges are seconds relative to the configured alignment event.
- R-squared and its increment are dimensionless. Unit MSE is squared spike
  count per bin; PC MSE is squared PCA-score units.
- Poisson deviance and deviance explained are dimensionless. Poisson expected
  values are counts per bin, and Poisson MSE is squared counts per bin.
- Linear Granger magnitude is the dimensionless in-sample
  `log(SSE_restricted / SSE_full)`. Poisson Granger-style output retains the
  dimensionless likelihood ratio and reports its per-row mean deviance
  improvement. These scales are not interchangeable with each other or with
  held-out CV increments.

PCA uses training means, population standard deviations (`ddof=0`), full SVD,
no whitening, and no random generator. The same regional fold transforms are
used across directions, conditions, and windows. Descriptive all-data PCA is a
separate scope and is rejected by cross-validation.

## Commands

Start from
`docs/examples/neural_analysis/interregional_regression_config.json.example`,
replace its absolute metadata path and probe IDs, and keep outputs outside the
Git checkout.

```bash
uv run python -m src.neural_analysis.interregional.run_session dry-run \
  --config /path/to/session/interregional_regression_config.json

uv run python -m src.neural_analysis.interregional.run_session new \
  --config /path/to/session/interregional_regression_config.json

uv run python -m src.neural_analysis.interregional.run_session new \
  --config /path/to/session/interregional_regression_config.json --rerun
```

Batch lists contain one configuration path per nonblank, non-comment line:

```bash
uv run python -m src.neural_analysis.interregional.run_batch dry-run \
  --config-list /path/to/interregional_configs.txt

uv run python -m src.neural_analysis.interregional.run_batch new \
  --config-list /path/to/interregional_configs.txt --workers 4
```

`dry-run` validates metadata, population selections, paths, versions, and byte
estimates without hashing large inputs, fitting, or creating output paths.
`new` requires tracked-clean code and no untracked Python beneath
`src/neural_analysis`, hashes every consumed file, and reuses a validated
matching final run unless `--rerun` is supplied.

The configuration's `analyses` list independently selects `ols_cv`,
`poisson_cv`, `linear_granger`, and `poisson_granger`. Granger-only runs use all
scientifically eligible condition rows, including eligible trials whose block
is missing, and do not construct folds. Linear PC Granger fits one separate
session-wide descriptive PCA basis; it never reuses fold-local CV transforms.

The cluster wrapper runs the same session command in the frozen offline
environment; it does not implement a separate computation path:

```bash
sbatch src/shell_scripts/interregional_regression_slurm.sh new \
  --config /absolute/cluster/path/interregional_regression_config.json
```

## Immutable output

The default location is `<session_root>/analysis_runs`. Work remains in a
hidden `.incomplete` directory until configuration, manifest, pure result,
scripts, log, summary, and figures validate. One same-parent atomic rename then
publishes the timestamped fingerprint directory. Failures retain `run.log`,
`failure.json`, and any completed telemetry records under the `.incomplete`
name and are never resumed or displayed.

`resource_trace.jsonl` is appended throughout execution. Every record contains
schema version, sequence, UTC and monotonic elapsed time, PID, process CPU and
peak RSS, and cgroup-v2 current/peak/limit/OOM fields when the operating system
provides them. A heartbeat is written every 30 seconds. Additional progress
records identify actual stage boundaries, every OLS/Poisson analysis cell, the
exact cell matrix shapes and byte sizes, and the first, every 25th, and final
completed Poisson target. These records allow an interrupted run's last active
cell and target range to be matched to its memory trajectory without per-fit
I/O. Missing or unlimited resource values are JSON `null`.

`resource_summary.json` is written atomically only after ordinary completion.
It contains the observed record count and maximum process/cgroup memory fields.
An OOM-killed run is therefore expected to retain a parseable trace without a
summary. Telemetry is execution metadata and does not enter the scientific
configuration, run fingerprint, result schema, folds, fits, or scores.

The Streamlit option **Inter-regional regression results** discovers only
validated final runs in the default session directory. It provides display-only
selectors and shows saved configuration, code/input identity, unavailable rows,
PCA fits, tables, and figures. Held-out CV and descriptive in-sample Granger
results have separate evaluation scopes and figures. It has no run, resume, or
recompute action.

Held-out directional scores are predictive summaries. Granger magnitudes are
descriptive in-sample nested-model improvements, not significance tests or
evidence of mechanistic causality. The first-pass coverage contract assumes
that loaded aligned spikes cover every requested trial window.
