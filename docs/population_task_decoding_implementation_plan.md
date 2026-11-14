# Population task decoding: implementation plan

Status: D1-D7 implemented and synthetically validated, 2026-11-14.
Implementation was approved on that date. Experimental execution remains a
separate user-designated-session gate. See the execution record below and
[usage instructions](../src/neural_analysis/population_decoding/README.md).
Scientific authority: [population_task_decoding_spec_v1.md](population_task_decoding_spec_v1.md).
[AGENTS.md](../AGENTS.md) and [SoftwareDesign.md](SoftwareDesign.md) govern
readability, data contracts, tests-first commits, and experimental run gates.

## 1. Deliverable and webapp placement

Add **Population task decoding** as a separate entry in the existing webapp's
**Plot view** dropdown, beside **Single unit selectivity**. Selecting it opens
its own routed view within the app. It is not a tab or mode inside the single
unit view and does not require another webapp process or browser window.

Proposed v1 interaction:

- Select a saved run, PFC/HPC, raw units/PCs, and a target model.
- Inspect actual labels and held-out probabilities on real session time,
  using the existing timeline navigation, event overlays, and gap handling.
- Inspect pooled scores, confusion matrices, class counts, fold diagnostics,
  and explicit reasons for unavailable fits or metrics.
- View results without mounted raw spike files. Changing controls never fits
  a model. Computation uses a separate CLI; if no results exist, explain how
  to produce them.

Use a new `webapp/population_decoding_view.py` renderer and widget-key prefix.
Add `PLOT_VIEW_POPULATION_DECODING = "Population task decoding"` in
`webapp/session_inputs.py`, its availability rule, and early dispatch in
`webapp/app.py` before live spike/unit controls. Add the corresponding metadata
requirement explanation for the legacy launch path.

Preserve the existing **Single unit selectivity** route, saved products,
controls, and within-unit display z-score option. Use separate result discovery,
cache keys, and session state for population decoding.

## 2. Scientific scope carried into implementation

Each region and representation has five target models:

| Model | Classes | Fitting and held-out scoring bins |
| --- | --- | --- |
| Context | Right=0, left=1 | `[-1,+1)` s |
| Current choice | Right=0, left=1 | `[-1,+1)` s |
| Outcome | Left reward, right reward, left no reward, right no reward, codes 0..3 in that order | `[0,+1)` s |
| Left choice switch | Current left followed immediately by right; indicator 0/1 | `[-1,+1)` s |
| Right choice switch | Current right followed immediately by left; indicator 0/1 | `[-1,+1)` s |

Labels remain fixed across each supervised interval. Outcomes use one joint
multinomial classifier; incorrect unrewarded trials are no-reward trials.
Switch labels use the immediately following unfiltered trial, never the next
retained neural trial. Invalid/manual-reward successors and the final trial
have unavailable switch labels. Other valid transitions are negatives; do not
add side-restricted switch/stay models.

Use independent PFC/HPC populations, good/mua and channel-quality selection,
IRIG-backed complete two-second coverage, and the agreed baseline exclusions.
Manual alignment remains deferred. Missing reward affects only outcomes;
missing successor affects only switches. Correctness and behavior-model value
columns are not prerequisites for these labels.

Default bins are 500 ms, with 100/50/20 ms supported. Pool bin vectors within
the supervised interval. Fit fixed L2 logistic regression, `C=1.0`, an
unpenalized intercept, and no tuning or class balancing. Each trial contributes
total training weight one, distributed equally over its supervised bins.

Fit scaling and PCA only on the exact eligible training bins for that
region/target/fold. Compare standardized direct unit features with all
numerically nonzero, unwhitened PCs. This PC comparison is largely a rotation
of the same features under isotropic L2 regularization; nearly identical
predictions are expected and are not evidence of a failed implementation.

## 3. Architecture and audited reuse

Use one shallow package, `src/neural_analysis/population_decoding/`, with
functions and a few data-only records. Do not introduce an estimator registry,
formula language, general workflow engine, or inheritance hierarchy.

| New module | Responsibility and main interfaces |
| --- | --- |
| `configuration.py` | Small `PopulationDecodingConfig`; JSON loading/initialization; paths, probe roles, bin width, versions, fixed scientific defaults. |
| `targets.py` | Validate chronological source/augmented table agreement on used fields; `build_population_targets` from source columns; target values, validity, reasons, class order. |
| `preparation.py` | Load sources, establish baseline/global block folds, prepare regional activity and bin-row mappings; `PreparedPopulationRegion` contains rates and identity, not fitted PCA. |
| `features.py` | `fit_unit_scaling`, `transform_unit_activity`, `fit_fold_pca`, `transform_pc_activity`; one training-fitted unit scaler shared by both representations, followed by sklearn PCA on standardized vectors. Save array-only transforms and unit/rank masks. |
| `modeling.py` | Fit weighted binary/multinomial readouts; predict ordered probabilities/log probabilities from saved numeric parameters. |
| `scoring.py` | Pooled and per-fold metrics, class support, conditional outcome summaries, switch prevalence/baseline. |
| `pipeline.py` | Explicit region/target/fold/representation loops; combine eligible rows, call pure computations, publish completed results. |
| `results.py` | Preparation/result publication, named-array loading, discovery, compatibility checks, and run status. |
| `plotting.py` | Saved-result probability timelines, label strips, confusion/performance panels, and PNG exports; no fitting. |
| `run_session.py`, `run_batch.py` | Assemble dependencies, single-session CLI, session-level workers, logs, summaries, resume. |
| `README.md`, `__init__.py` | Usage and package marker. |

Define records with the module owning their contract: configuration, prepared
region, unit scaling/PCA transform, and classifier parameters. Computational
helpers receive arrays/columns and explicit scalars; whole tables are reserved
for loading, validation, and row mapping. CLI/UI assemble concrete dependencies.
Do not pass a large mutable session object through numeric helpers.

The following public seams were inspected:

| Existing code | Reuse and boundary |
| --- | --- |
| `session_metadata.py` | Resolve metadata and declared regional probe roles; do not infer anatomy. |
| `encoding/predictors.py:select_baseline_trials` | Reuse the baseline mask and original row/trial identities. Its separate joint-model validator requires correctness, and its omission classifier is too narrow; do not reuse those for the new targets. |
| `population/session_timeline.py` | Reuse `choice_bin_edges`, complete-window coverage selection, and `bin_choice_counts`, explicitly passing `window_s=(-1,1)`. Preserve other callers' defaults. |
| `population/session_timeline_io.py` and `task_decoding/activity.py` | Reuse regional spike/unit loading and trusted coverage inspection. Reject unsupported manual alignment; do not estimate coverage from spike extrema. |
| `encoding/preparation.py:build_bin_rows` | Reuse explicit trial/bin/time geometry with the new shared fold map. Its regional preparation/PCA/fold selection is not compatible with this protocol. |
| `interregional/preparation.py` | Reference for deterministic `GroupKFold`. Its public fold builder assumes the unfiltered table; use a small baseline-pool wrapper around the public sklearn splitter and map results back to original rows. |
| `interregional/pca.py` | Reference for full-SVD, unwhitened PCA. Its helper bundles count-domain scaling with PCA, so it does not accept the shared standardized vectors required here. Call sklearn PCA directly after the new package's single scaling step; leave the existing helper unchanged. |
| `task_decoding/pooled/decoding.py:stack_training_window` | Reuse trial-major/bin-major flattening and `1/B` weights, including integer outcome codes. |
| `task_decoding/session_prediction/modeling.py` | Reference for logistic controls and convergence handling. Its public fitter flattens coefficients and keeps one intercept, so it cannot represent four-class parameters. Leave it unchanged. |
| `population/timeline_plotting.py`, `webapp/timeline_view.py`, `webapp/timeline_chart.py` | Reuse actual-time/gap geometry, behavior/event overlays, navigation, and persistent chart ranges. |

Existing session-prediction and cross-condition pipelines assume fixed
session-wide PC axes and mostly binary outputs. Their fitting pipelines and
saved schemas are not suitable substitutes. Reuse compatible public functions,
not private helpers or entire pipelines with extra mode flags. Avoid shared
numerical refactors that would invalidate existing encoding run identities.

Dependencies are already installed: standard library, NumPy, pandas, SciPy,
scikit-learn, Pynapple, Plotly, Matplotlib, Streamlit, and pytest. No new package
is proposed. Installed sklearn 1.8.0 APIs were inspected; follow the existing
`lbfgs`, `l1_ratio=0.0`, `tol=1e-4`, `max_iter=5000` controls. Use sklearn PCA
with `svd_solver="full"` and `whiten=False` on the standardized training
matrix. Verify any API not covered by existing usage before implementation.

## 4. Data contracts and computation order

Let N be complete source trials, W regional covered baseline windows, B bins
in two seconds, U input units, P training-variable units, K nonzero PCs, and
C target classes. Every public function documents dtype, shapes/axes, physical
units, return values, and missingness. Inputs are not modified.

| Boundary | Contract |
| --- | --- |
| Target construction | Source columns `(N,)` -> integer targets `(N,5)`, Boolean validity and reason arrays `(N,5)`, fixed class metadata. Unknown is `-1` plus invalid, never a negative class. |
| Fold assignment | Baseline eligibility/block identity `(N,)` -> fold IDs `(N,)`, values 0..4 or unavailable `-1`, and reasons. Assignment precedes regional and target masks. |
| Regional preparation | Counts `(W,B,U)` in spikes/bin -> rates `(W,B,U)` in Hz by division by bin duration in seconds. Preserve ordered unit IDs, original row positions `(W,)`, relative bin edges `(B+1,)`, elapsed/UTC times, and coverage. |
| Unit scaling | Only eligible training rates `(T_train,B_scope,U)` plus unit IDs -> means/SDs in Hz, retained-unit mask, and standardized unit vectors. One fitted scaler supplies both representations. No held-out input is accepted by fitting functions. |
| PCA | Standardized training vectors `(M,P)` -> components `(K,P)`, PCA centering offset `(P,)`, singular values, and explained variance. Values are dimensionless; no unit scaling or selection is refitted. |
| Transform application | Rates `(...,U)` plus saved transform and matching unit order -> standardized units `(...,P)` or PC scores `(...,K)`. No re-estimation or PC whitening. |
| Classifier | `X(M,F)`, labels `(M,)`, weights `(M,)`, ordered classes `(C,)` -> coefficients `(L,F)`, intercepts `(L,)`, status/reason; L=1 for binary, L=4 for outcome. Coefficients map feature units to log odds/logits. |
| Prediction | Saved parameters and `X(M,F)` -> probabilities/log probabilities `(M,C)` in fixed class order; probabilities are dimensionless. Failed or unsupported rows remain unavailable, not zero. |
| Scoring | Ordered labels/probabilities, validity, row identities -> metrics, counts, and explicit metric-specific reasons. |

1. Validate source/augmented chronology and fields used by this analysis;
   construct all targets on the full table. Do not require correctness or
   unrelated fitted behavior values.
2. Apply baseline eligibility and establish one deterministic five-fold block
   assignment from that pool. Keep all rows/bins of a block together and map
   folds back to original row positions. Never redraw folds for class support.
   Missing block identity leaves affected rows without OOF predictions;
   fewer than five usable blocks makes the five-fold protocol unavailable.
3. Load/bin each region once and save its complete eligible-window rates.
   Regional coverage changes retained rows, not the global fold assignment.
4. For a target/fold, select its eligible training trials and supervised bins.
   Learn means/population SDs (`ddof=0`) and omit training-constant units,
   recording IDs/reasons. This is the single owner of unit scaling and
   selection for both representations. All-constant inputs make the fit
   unavailable.
5. Flatten the standardized variable-unit training vectors to `(M,P)` and
   pass that matrix directly to sklearn PCA with full SVD and no whitening.
   Retain singular values above
   `max(M,P) * eps(float64) * largest_singular_value`; record the rule and ranks.
   Save the PCA centering offset and retained components so numeric parameters
   reproduce its transform. PCA does not estimate another unit scaler or mask.
6. Fit both readouts on matched rows. The raw-unit readout uses the shared
   standardized vectors; the PC readout uses their PCA scores. Apply the
   saved unit scaler unchanged to held-out rates, then apply the saved PCA
   transform for the PC readout. A PCA-specific failure must not discard an
   otherwise valid raw-unit fit.
7. Collect one OOF prediction per eligible supervised bin. Compute primary
   metrics from pooled OOF rows, not equal averages of fold scores. Incomplete
   required predictions leave primary scores unavailable while preserving
   completed folds. Do not replace missing OOF values with an all-data fit.

Save balanced accuracy and log loss as primary scores; also ordinary accuracy,
fixed-order confusion counts, per-class precision/recall, ROC-AUC, average
precision, and support/prevalence. Class-dependent metrics with missing test
classes get explicit reasons without discarding valid probabilities/log loss.
Training must contain both binary classes or all four outcome classes.

For outcomes, additionally summarize reward/no-reward discrimination within
actual left/right choices by normalizing that side's two probabilities from
the same four-class model. Use stable log probabilities where needed. For
switches, show positive/negative counts and an always-negative baseline.
Do not add significance tests or prospective/causal interpretation.

## 5. Saved products, CLI, and inspection

Use a new `population-task-decoding-v1` analysis identifier and independent
schema version. Save shared activity in Hz once, rather than duplicate it for
raw-unit and PC models. Do not save live sklearn estimator objects.

Proposed session-local layout:

```text
processed/population_task_decoding/<preparation_id>/preparation.npz
analysis_runs/population_task_decoding_<date>_<time>_<run_id>/
    config.json, progress.json, run.log, summary.md, performance.json
    run_session.py, run_batch.py
    cells/<region>_<target>.npz
    figures/
```

Each paired region/target file contains both representations, five fold
results, pooled OOF predictions, and metrics. Two regions produce ten paired
files, twenty region/target/representation comparisons, and at most 100
classifier fits. Use explicit per-fold arrays for variable PC ranks; never
pretend fold PC axes are interchangeable.

NPZs use named arrays and a `meta` dictionary: generating function, analysis
and schema versions, parameters, deterministic/random-seed provenance, class
order, units/axes, source/code/runtime identities, train/test row mappings,
folds, transforms, unit/rank masks, counts, fit statuses, and reasons. Include
behavior/events needed for offline inspection. Stored rates plus bin width
document the original count-to-Hz conversion.

Use ordinary sibling temporary files and rename to publish a completed file,
then update small JSON progress. A published compatible file is adopted after
an interruption before the progress update. Resume skips completed paired
files; interruption inside one file redoes that file only. Do not overwrite
incompatible/corrupt files or prior results after code/parameter changes.
No per-fold checkpoint framework is needed.

V1 reuses only this decoder's own compatible saved preparation, matching its
version, source identity, unit selection/order, trial mapping, IRIG coverage,
window, and bin geometry. First preparation uses the existing source loaders
and binning functions. Support for importing other analyses' count archives
is deferred until measured preparation costs justify a format-specific
adapter. Do not add an older-archive input option or compatibility layer in v1.

Provide the existing-style CLI operations: `init-config`, `dry-run`, `new`,
`new --rerun`, `status`, and `resume`. Configuration initialization locates
inputs; it does not inherit another analysis's labels or PCA. Dry-run performs
read-only path/metadata/compatibility checks, without binning or fitting.
After preparation publication, resume and viewing can use saved data without
raw sources. Log parameters, session IDs, warnings/errors, timings, and sizes;
copy main/batch scripts and write a readable run summary.

The saved-results view shows:

- Context/choice: complementary left/right probability traces.
- Outcomes: four probabilities only on `[0,1)`; pre-choice is blank.
- Switches: the selected directional event's probability across `[-1,1)`,
  labeled as a next-chosen-port change attached to the current trial.
- Actual labels, real elapsed time/gaps, common probability limits `[0,1]`,
  class support, pooled scores, and fold/unavailability diagnostics.

Use existing pan/zoom preservation. Keep categorical confusion axes out of
time-axis range propagation. If coefficient details are exposed, require an
explicit fold and identify that fold's unit/PC basis; no averaged PC weight
or fixed-session-PC contribution plot. PNG exports use readable labels,
opaque light backgrounds by default, and captions stating scope and results.

## 6. Tests written before implementation

Tests live under `src/tests/neural_analysis/population_decoding/`, with small
fixtures and function-based tests. For each phase: write the tests, demonstrate
RED for the missing behavior, commit tests only, implement GREEN, then refactor
with tests passing. Record commands/results and separate test/source commits.
Never weaken a test to accommodate a different scientific implementation.

### D1. Configuration, labels, and global folds

Write `test_configuration.py` and `test_targets.py` first:

- Fixed classes/scopes/versions, all supported bin widths, and invalid configs.
- Broad no-reward labels including incorrect choices; side uses action, not
  context; unknown reward affects only outcomes.
- Both switch directions, all other valid transitions zero, invalid/manual/
  no-choice successors and terminal rows unavailable; no skipping filtered rows.
- Irregular original trial IDs, chronological/source mismatch rejection,
  target-specific missingness, and no correctness/model-value requirement.
- Deterministic shared block map before regional/target filtering, all bins of
  a block together, and explicit missing-block/insufficient-block behavior.

### D2. Activity, scaling, and PCA

Write `test_preparation.py` and `test_features.py` first:

- Exact two-second geometry, zero edge, counts-to-Hz units, original row/unit
  mappings, independent regional coverage, and manual-alignment rejection.
- This decoder's compatible preparation can be reused; mismatched geometry,
  source, or selection is rejected. Existing timeline defaults stay intact.
- Held-out perturbations cannot alter training means/scales, retained units,
  PCA rank, or basis. Outcome preprocessing sees post-choice training bins only.
- Training-constant/silent units excluded even if they vary at test time;
  all-constant pools unavailable; all nonzero PCs retained, no hidden cap.
- Both representations use the same fitted scaler, retained units, and
  standardized training vectors; PCA does not refit unit scaling or selection.
- Transforms require matching unit order; saved unit scaling, PCA centering,
  and components reproduce library transformations without saved estimators.

### D3. Classifiers and scoring

Write `test_modeling.py` and `test_scoring.py` first:

- Weighted binary and four-class fits against an independent numeric oracle;
  coefficient/intercept shapes, class order, probability sums, stable extremes,
  binary 0.5 threshold, and multiclass argmax.
- Per-trial weight sums are one at every resolution and target scope.
- Raw/all-PC linear predictions agree within solver tolerance on controlled
  data; no whitening or PC-score rescaling.
- Missing training class/nonconvergence unavailable; missing test class keeps
  valid predictions/log loss and invalidates only affected metrics.
- Pooled OOF scores differ correctly from equal fold averages; incomplete OOF
  cannot report a complete primary score. Average precision uses that definition.
- Conditional left/right reward summaries use the same four-class fit;
  rare switches report prevalence and the always-negative baseline.

### D4. Pipeline, persistence, and resume

Write `test_pipeline.py` and `test_results.py` first:

- Correct five-target/two-representation scope and maximum fit count; shared
  training transforms; variable fold ranks saved with correct identity/units.
- PCA failure preserves raw-unit results; one unavailable target/fold does not
  erase other results. Input tables/arrays are unchanged.
- Named-array round trips, source-free loading/resume, compatible completed
  files skipped, publication/progress interruption adoption, and partial-cell redo.
- Changed source/code/runtime/parameters reject reuse; incompatible/corrupt
  products are not overwritten and prior runs remain discoverable.

### D5. Entry points and session workers

Write `test_run_session.py` and `test_run_batch.py` first:

- Initialization, dry-run without computation/output writes, new/reuse/rerun,
  status/resume, logs, script copies, summary, and measured performance fields.
- Actual subprocess execution on two synthetic sessions; CPU-capped session
  worker count, duplicate-session rejection, and no within-session worker pool.

### D6. Separate webapp view and saved plots

Write `test_webapp.py` and `test_plotting.py` first:

- Both dropdown entries exist and dispatch to distinct renderers; the new
  saved-only route precedes live source controls and works with raw files absent.
- Independent widget state/caches; selector changes cannot fit/load raw spikes;
  missing runs and partial/unavailable results have useful messages.
- Correct probability/class labels, blank pre-choice outcomes, current-trial
  switch placement, real-time gaps, pan preservation, and categorical axes.
- Fold-specific coefficient identity if displayed; PNG background/caption and
  non-overwriting export behavior. Existing single-unit view/z-score tests pass.

### D7. Full synthetic validation and integration

Write `test_synthetic.py` first, then exercise the entire saved-results path:

- Known context, current choice, side-specific reward/no-reward and next-choice
  signals, including incorrect unrewarded trials and distinct context/action.
- Rare switches, absent classes, constants/noise units, truncated regional
  coverage, invalid successor trials, and target-specific missingness.
- A side-only signal can predict four outcome states partially but cannot
  establish within-side reward discrimination. Generic pre/post timing alone
  cannot solve an outcome decoder trained only post-choice.
- End-to-end OOF leakage checks, raw/all-PC agreement, publication/resume,
  source-free viewer loading, dropdown navigation, and actual rendered charts.

Run focused tests after each phase with `uv run`; run related existing encoding,
timeline, PCA, and webapp tests after integration. Record any pre-existing suite
failures separately. Synthetic fit thresholds use controlled signal/noise and
numeric oracles rather than arbitrary claims about biological selectivity.

## 7. Performance and storage

Bin once per region; keep one shared native-rate tensor rather than one per
representation. Fit at most one target/fold's feature matrices at a time, share
its transform across both readouts, and release temporary matrices afterward.
Do not retain all training predictions or duplicate full activity per fold.

For W windows and U units in total, float64 rates require `8 * W * B * U`
bytes. As a sizing example, W=696 and U=141+286 require approximately 9.1 MiB
at 500 ms (B=4), or 226.7 MiB at 20 ms (B=100), before working arrays,
decompression, spike loading, and SVD workspace. These are array-size estimates,
not measured end-to-end memory or a new experimental fit.

At most 50 region/target/fold PCA fits accompany 100 readout fits. Full-PC
comparisons retain potentially large bases and provide no dimensionality
reduction guarantee. Prefer clear library operations before optimization;
measure preparation, transforms, fitting, saving, peak memory, and archive sizes
on synthetic data and the later authorized session.

Parallelize batches across sessions with available-CPU worker defaults/caps,
subject to measured RAM/I/O capacity and an explicit worker override. Set
numerical-library thread limits before imports in worker entry points to avoid
nested BLAS parallelism. No within-session parallelism, new compiled dependency,
or major optimization is proposed.

## 8. Approval and delivery gates

1. Review/approve this draft before source implementation. The saved-only CLI
   plus separate dropdown view is the proposed v1 compute/view interaction.
2. Complete D1-D7 with recorded tests-first commits, synthetic results, usage
   examples, and a browser check of both distinct views.
3. Obtain a user-designated experimental session for this new analysis, run it
   once, and present saved probabilities/scores for inspection. Prior 8/28
   single-unit fits do not constitute a population-decoder result or authorize
   this new experimental run.
4. After user inspection/approval, run further designated sessions through the
   batch entry point. Do not infer a wider session list.

No additional scientific decision is currently required. Report contradictory
source evidence discovered during implementation rather than silently changing
label definitions, eligibility, folds, or timing.

## 9. Execution record

- Existing single-unit baseline: 271 tests passed.
- D1 RED: missing population-decoding package; tests committed as `7dd4372`.
  A pandas string-assignment fixture error was corrected in `1d4c05b`.
- D1 GREEN: 20 configuration/target/fold tests passed.
- D2 RED tests: `f581d24`; GREEN source: `8b1ab60`; 28 cumulative tests passed.
- D3 RED tests: `8d43c80`; the rotation fixture was corrected to use the
  specified standardized unit vectors in `0af0b8e`, preserving its assertion
  tolerance. GREEN source: `df1ec0b`; 15 model/scoring tests passed.
- D4 RED tests: `cd488aa`; GREEN source: `1e7c4c1`; 49 cumulative tests passed.
- D5 RED tests: `789d731`; GREEN source: `76c3845`; 56 cumulative tests passed,
  including two actual spawned session workers and source-free resume.
- D6 RED tests: `3dbe8c3`; GREEN source: `7ef47f5`; 61 cumulative tests passed.
- D7 scientific/provenance/axis tests: `d4c7c85`; full Poisson-spike source
  validation: `e2e62f8`. Browser review prompted compact outcome summaries,
  explicit time-axis metadata, and logged fold reasons, tested in `529846b`.
  A broad-suite Streamlit main-module leak was reproduced in `e67ed11` and
  isolated inside this package's test fixtures, without changing other analyses.
- Final focused integration: 317 tests passed: 75 new population-decoding
  tests, two Node-backed timeline-component tests, and 240 existing PPC
  parallel/runtime tests in a fresh process. Single-unit regression checks
  also passed in the combined 342-test check before the final additions.
- Broad neural-analysis check before the last integration repairs: 5,135
  passed, 13 failed, one skipped. Eleven failures involved leaked Streamlit
  `__main__` state in spawned-worker tests (including the new batch test);
  affected parallel/runtime tests pass in the focused run above. The other
  two failures are unchanged SLURM job-name and Open Ephys `sample_rate`
  expectations outside this implementation. The Node skip was covered by
  rerunning with the installed Node binary. This is not an all-green broad
  suite claim; no unrelated source/test failures were edited away.

### Synthetic validation evidence

Seed 81 generated 240 trials with mixed context/action/reward and both choice
switches, across 20 behavioral blocks. Seed 20261114 generated actual Poisson
spike archives for two eight-unit probes. The raw loaders, good/mua/channel
selection, IRIG coverage, binning, all five models, both representations,
publication and saved viewing were exercised at 500-ms resolution. Pooled
held-out balanced accuracies ranged from 0.9446 to 0.9954 across those controlled
signals; raw-unit and all-PC predictions agreed. These are artificial-data
correctness checks, not experimental findings or significance tests.

Side-only and generic post-choice controls did not establish within-side reward
discrimination or solve the post-only outcome task. Missing labels, training
class absence, constants, regional coverage, saved-data corruption, partial
resume, code/runtime mismatch and held-out preprocessing leakage are covered.
Measured timings and peak process RSS are saved by the synthetic entry-point
runs; interpreter/test-process overhead is included in RSS. Production-session
performance remains unmeasured.

No experimental session has been run for this new analysis. User session
designation and inspection remain the next gate before any wider batch.
