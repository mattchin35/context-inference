# Joint encoding implementation validation - 2026-11-13

Implemented the approved [joint-model plan](single_unit_joint_plan_v2.md).
The current [scientific specification](single_unit_ridge_spec_v1.md) and
[usage guide](../src/neural_analysis/encoding/README.md) describe the delivered
model. Experimental numerical processing remains pending the one-session gate.

## Delivered behavior

Unit rates in Hz and all nonzero fixed regional PCs use the same eligible
choice-aligned [-1,+1) bins and block folds. Entering context/current choice are
fixed across the whole window; six chosen-side outcome pulses start at choice.
Both context encodings, ridge/OLS, feature-specific matched penalties, raw
reward-minus-omission contrasts and fold support are saved. Units work without
PCA. Current-version saved inspection, source-free resume, orphan-cell recovery,
single/batch commands, exports and measured performance reporting are retained.
Old encoding configurations/products are explicitly rejected or skipped without
migration or modification. Manual-alignment sessions remain deferred.

## Synthetic scientific validation

Seed **20261113** generated real spike timestamps and normal source files, then
ran the production single-session pipeline. Each population contains context,
choice, left/right reward, left/right omission and left/right incorrect units,
a generic post-choice unit, seeded noisy units, a constant unit and a silent
unit. Counts use four 500-ms bins per window. Independent PFC/HPC IRIG coverage
leaves 72/64 windows in the balanced example. Both regions retain 19 units and
16 numerically nonzero PCs from 17 variable inputs.

All 20 cells in each balanced/imbalanced run were checked against independently
constructed X, generated count-to-Hz responses, explicit saved PCA projections,
and an augmented least-squares penalty oracle. Reference and every training
fold's coefficients, contrasts and held-out predictions agree to 1e-10 absolute
tolerance. Exact OLS units recover their specified 4-Hz context/choice and
10-Hz condition effects. Constant/silent controls retain MSE and unavailable
R-squared. Additional end-to-end variants validate missing left omissions and
an entirely unavailable PCA without losing unit results.

| PFC signed-context generic post-choice unit | Balanced trials | Unequal trials |
| --- | ---: | ---: |
| Left/right reward trial counts | 12 / 12 | 24 / 18 |
| Left/right omission trial counts | 12 / 12 | 6 / 6 |
| OLS left/right reward-minus-omission (Hz) | approximately 0 / 0 | approximately 0 / 0 |
| Ridge alpha=1 left/right contrast (Hz) | approximately 0 / 0 | 0.326535 / 0.285678 |

The nonzero imbalanced ridge contrasts agree with the independent penalty
oracle. They demonstrate unequal shrinkage in an equal-response unit, not
reward-specific biology. The exact OLS unit responses' largest pooled held-out
MSE was below 2e-28 Hz squared in these fixtures. No significance test was run.

## Saved UI and visual checks

AppTest forbids live input loading or regression during saved controls. Tests
cover original nonmonotonic unit IDs, PC12, native units, sorting, supported raw
contrasts, prediction scope, absent PCA, exact bin edges and blank gaps, partial
runs, retired-run omission, immutable source arrays and exports.

A real browser inspected the saved synthetic run through the production webapp:

- Unit coefficients/activity share identical original IDs and row order.
- All six outcome tracks are absent before choice; unit activity uses Hz.
- Dragging the neural panel pans all elapsed axes together. The categorical
  coefficient columns remain unchanged. The panned ticks (40/60/80 s) survive
  standardized coefficient display, switching to PCs and selecting PC12.
- Raw contrast sorting preserves original PC IDs on both heatmaps, while the
  PC12 prediction trace retains its identity and signed PC-score scale.
- A discovered colorbar/predictor-label overlap was reproduced in a regression
  test and corrected. Browser reload verified clear labels, activity colorbars
  aligned with the neural panel, and the coefficient scale below its panel.
- Both unit and PC PNGs were visually checked. PNG tests verify opaque white
  backgrounds, readable output dimensions, exact scope/ID/unit captions and
  refusal to overwrite existing exports.

The default general PSTH route initially reported a missing `led_on_time` field
in this deliberately minimal synthetic source. Selecting saved Context encoding
results worked; the saved encoding route requires no such live raster column.
No unrelated PSTH code was changed.

## Measured synthetic performance

One balanced pytest-process run, with unspecified filesystem-cache state:

| Measurement | Value |
| --- | ---: |
| Common preparation | 0.0763 s |
| Preparation publication | 0.00706 s |
| Aggregate reference/fold fitting, prediction and scoring (20 cells) | 0.0487 s |
| Aggregate cell publication | 0.0434 s |
| Saved preparation + representative-cell loading | 0.0153 s |
| Four representative figures | 0.335 s |
| Figure serialization | 0.0186 s |
| Four serialized figure payloads | 361,349 bytes |
| Shared preparation + 20 compressed model cells | 607,593 bytes |
| Peak process-lifetime RSS | 616,153,088 bytes |

The benchmark loads preparation once and four representative cells (one per
region/representation), never all model cells together. RSS includes imports,
pytest and earlier process activity; it is not incremental analysis memory.
These tiny synthetic measurements do not establish CT026 or 20-ms performance.

## CT026 readiness

Read-only dry-run used the existing CT026 2026-08-04 input locators with the new
version, both representations, and the obsolete timing/window/timeline fields
removed. Metadata path:

```text
/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260804_latent_inference/neural_session.json
```

Its augmented source is
`processed/CT026_2026-08-04_120843_augmented_trials.csv`; feature parameters are
`processed/trial_feature_params.json`. Declared version-3 probe roles apply.
The review config is staged locally at
`.pytest_cache/ct026_joint_encoding_review.json`, with absolute verified input
paths. It can be inspected without changing the existing session config.

Dry-run: **716 baseline trials, 20 planned cells, no input blockers, no context
event diagnostics**. Neural counts and exact regional coverage remain pending.
No raw spike arrays were loaded and no CT026 scientific products, configuration
or prior results were changed. Real numerical execution and subsequent user
inspection remain the next gate before any wider experimental batch.

## Regression results

Focused scientific, saved-view and directly affected timeline checks passed
333 tests, with one frontend check initially skipped without Node. The bundled
Node runtime subsequently passed both frontend tests. A final integrated run
and a broad neural regression pass are recorded below after completion.

Final focused command (bundled `NODE_BINARY` set) passed **335 tests in 49.42 s**:

```bash
uv run --frozen --no-sync python -m pytest src/tests/neural_analysis/encoding \
  src/tests/neural_analysis/test_session_timeline.py \
  src/tests/neural_analysis/test_session_timeline_io.py \
  src/tests/neural_analysis/test_session_timeline_scripts.py \
  src/tests/neural_analysis/test_timeline_plotting.py \
  src/tests/neural_analysis/test_timeline_chart.py \
  src/tests/neural_analysis/test_session_timeline_webapp.py -q
```

Broad neural pass used `--import-mode=importlib` because default pytest import
mode collides on repeated test filenames across existing test directories.
It completed with **5,054 passed and 12 failed in 537.46 s**. Failures are outside
the changed encoding/timeline paths:

- Existing interregional SLURM test expects job name `interregional_poisson`;
  the unchanged script uses `interregional_regression`.
- Existing Open Ephys synchronization test expects a `sample_rate_hz` call
  keyword that the pre-existing dirty synchronization implementation does not
  supply. That user edit was preserved.
- Ten existing LFP PPC spawn tests fail after an unrelated event-view AppTest
  replaces `sys.modules['__main__']` with its generated script. Workers then
  encounter `NameError: __args`. The encoding test harness restores its own main
  module; this unrelated event-view harness was not modified. The ten PPC tests
  were rerun together in a fresh process to classify this order-dependent issue.

The failing tests and LFP implementation were unchanged by this work. No
unrelated code or tests were repaired to make the broad pass appear green.
`ruff check` for encoding/the saved view and `git diff --check` passed.

The isolated PPC rerun passed **11 tests (229 deselected) in 22.28 s**, including
all ten broad-run PPC failures and the serial worker-count control. The two
persistent unrelated SLURM/synchronization failures remain documented above.
