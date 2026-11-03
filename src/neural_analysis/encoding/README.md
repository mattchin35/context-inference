# Context encoding on fixed regional PCs

This package fits true task context to individual neural time bins pooled across
eligible choice-centered windows. It uses independent PFC/HPC descriptive PCAs,
all numerical nonzero PCs, and native signed or two-indicator context coding.
The scientific contract is `docs/single_unit_ridge_spec_v1.md`; execution gates
and the tests-first record are in `docs/single_unit_plan.md`.

## Inputs

Copy `docs/examples/neural_analysis/context_encoding_config.json` beside a
session's `neural_session.json` and edit its relative input paths. The augmented
trial CSV and feature-parameter JSON must already exist. This analysis does not
generate behavioral features or refit behavioral models. The source and augmented
trial tables must describe the same complete chronological trial sequence.

Version-3 metadata supplies the declared probe roles when `probe_roles` is null.
For version-2 metadata, specify a nonempty mapping such as
`{"PFC": "ProbeA", "HPC": "ProbeB"}`. A single-region mapping is allowed.
Manual-alignment sessions are deferred: aligned archives need `irig_utc_unix`
recording coverage, independently for each region. Spike extrema do not establish
coverage. Existing good/mua unit and channel-quality selection rules apply.

## Commands

Run commands from the repository with the existing UV environment:

```bash
uv run python -m src.neural_analysis.encoding.run_session dry-run --config /path/to/session/encoding.json
uv run python -m src.neural_analysis.encoding.run_session new --config /path/to/session/encoding.json
uv run python -m src.neural_analysis.encoding.run_session status --run-directory /path/to/session/analysis_runs/context_encoding_RUN
uv run python -m src.neural_analysis.encoding.run_session resume --run-directory /path/to/session/analysis_runs/context_encoding_RUN
```

`new` reuses a completed identical run. If matching work is partial, it reports
that directory for explicit `resume`. `new --rerun` creates another timestamped
run and preserves previous products. Published compatible preparation can be
shared across runs with different model penalties. Resume uses saved inputs once
preparation has been published; changes to numerical code require a new run.
Before preparation was published, original input identities must still match.

For batches, create a text file listing one configuration per noncomment line,
relative to that list file or absolute:

```bash
uv run python -m src.neural_analysis.encoding.run_batch dry-run --config-list /path/to/configs.txt
uv run python -m src.neural_analysis.encoding.run_batch new --config-list /path/to/configs.txt --workers 2
```

Batch execution uses separate session processes, capped by session and CPU count.
It rejects two configurations targeting the same canonical session. The command
boundary limits nested BLAS threads before numerical imports. Dry-run uses small
tables/metadata and archive member names; it does not load spikes, fit PCA/models,
start workers, or write outputs. Neural coverage and PC counts remain pending.

Experimental execution follows a separate gate: synthetic validation first, then
one user-designated session and user inspection, followed by authorized batches.

## Scientific and saved-data contracts

- Activity windows are `[-2,+2)` seconds relative to choice. Supported bin widths
  are 500 (default), 100, 50, and 20 milliseconds. Before/after split at zero.
- Unit counts have axes `(window,bin,unit)`; unwhitened PC scores have axes
  `(window,bin,original_PC)`. Eligible regression matrices are `X(N,P)` and
  `Y(N,K)`, retaining original trial/bin/time identities. Absolute times are UTC
  Unix seconds; elapsed zero is the first unfiltered trial start.
- Manual-reward, no-choice, and dark/invalid entering-context trials are excluded
  before PCA. Each region also requires complete four-second recording coverage.
  Later predictor missingness does not change the descriptive PCA pool.
- `choice_updated` assigns `state_int[i]` before choice and the immediate
  unfiltered `state_int[i+1]` after choice. A missing/dark/invalid adjacent state,
  including the absent terminal state, makes post bins unavailable. `trial_window`
  holds the entering state throughout. Logged events are diagnostic only.
- Ridge minimizes summed SSE plus `alpha * sum(slopes**2)`, with an unpenalized
  intercept and no predictor/response scaler. `matched_context` uses alpha/2 for
  the two-indicator fit; `same_numeric` uses alpha unchanged. OLS uses centered
  minimum-norm slopes with both encodings and has no penalty.
- Slopes have axes `(predictor,response)` and intercepts `(response,)`. Reference
  coefficients use all eligible rows. Display scaling uses
  `beta * SD(X) / SD(Y)` on those same rows and never rescales activity.
- Five deterministic `cur_block` folds are assigned per region before model
  masking. All bins from a window keep its fold. Held-out scores are conditional
  on fixed descriptive session PCA, which includes test-block neural activity.
  MSE and R-squared pool all held-out bins; negative R-squared is retained.
  Missing folds leave primary scores unavailable while preserving reference and
  completed-fold results. Zero-SST responses retain MSE with unavailable R-squared.

Preparation is saved once under session-local `processed/context_encoding`.
Timestamped `analysis_runs/context_encoding_*` directories contain configuration,
progress, logs, script copies, a summary, and named cell products. NPZ products
use named arrays plus a dictionary named `meta`, including source/code identities,
version, parameters, units/axes, and randomness provenance. Completed scientific
unavailability is finished work; it is not repeatedly refit during resume.
Experimental outputs are rejected inside the source repository.

The initial implementation covers PC ridge/OLS inspection. Production unit fits,
Poisson, significance testing, automatic alpha tuning, and additional predictors
remain outside this delivery.
