# Condition Prediction Generalization: Implementation Handoff

Status: reviewed draft, 2026-11-03. Scientific choices are confirmed.
Implementation, delegation, and experimental preparation have not started.

Read the [scientific spec](condition_prediction_generalization_spec_v1.md),
[SoftwareDesign.md](SoftwareDesign.md), and repository `AGENTS.md` before
implementation. The spec owns scientific meaning; this plan owns file scope,
tests, delivery order, and live status. New user decisions supersede both.

## 0. Live implementation handoff

Update the snapshot and ledger at each RED/GREEN/review boundary and before
stopping. Append command/commit evidence to the log; do not replace it with
chat summaries. This follows the existing
[pooled-decoding handoff](task_variable_condition_generalization_implementation_plan.md)
and [regression handoff](neural_regression_plan.md) without adding runtime
checkpointing to this analysis.

### 0.1 Current snapshot

| Field | Current value |
| --- | --- |
| Date / branch / inspected HEAD | 2026-11-03 / `refactor` / `d435b42046e96d529b1f75116c79a2be8c72cb97`. Recheck on resumption. |
| Phase / authorization | WP0 audit complete; ready for handoff. This request authorizes plan revision only. Start WP1 when the user instructs implementation. |
| Scientific settings | Confirmed; see Section 2. No outstanding scientific question. |
| Source/tests | No new feature source/test directories, RED/GREEN commits, or implementation tests exist. |
| Documentation | This plan and the scientific spec are untracked at the inspected baseline; no feature commit is claimed. |
| Existing user changes | Staged: added `docs/session_timeline_pca_spec_v1.md`. Unstaged tracked: `src/main.py`, `src/neural_analysis/sync_ephys.py`. Preserve these and unrelated untracked files. |
| Team / writer | No worker launched; no implementation writer assigned. Planned Sol lead plus at most one Luna worker, Section 12. |
| Running analysis | None started by this feature; no experimental output exists. |
| Latest verification | Documentation checks passed: ASCII, four links, fences, whitespace, sections 0-12, nine ledger entries, confirmed settings, and current reused APIs/gates. No implementation tests run. |
| Next action | Hand off the reviewed documents. Once implementation is authorized, Sol writes WP1 tests and establishes RED before production code. |

### 0.2 Package ledger

Fill evidence as work proceeds. Worker completion is not Sol acceptance;
experimental visual acceptance belongs to the user.

| Package | Writer | State | RED commit / command / result | GREEN commit / checks / review | Next gate |
| --- | --- | --- | --- | --- | --- |
| WP0 documentation | Lead | Reviewed, ready for handoff | Not applicable | Document checks passed; no commit | Receiving task begins WP1 on implementation instruction. |
| WP1 settings/alignment/folds | Sol | Not started | None | None | Tests first on implementation authorization. |
| WP2 models/contributions | Sol | Not started | None | None | Accepted WP1 contracts. |
| WP3a persistence/identity | Sol | Not started | None | None | WP2 result records. |
| WP3b runner/CLI/batch | Luna | Not started | None | None | Frozen WP3a API. |
| WP4a plotting/geometry | Sol | Not started | None | None | Synthetic WP3 round trip. |
| WP4b webapp integration | Luna | Not started | None | None | Frozen WP4a display API. |
| WP5a usage documentation | Luna | Not started | Not applicable | None | Integrated synthetic GREEN. |
| WP5b CT026 inspection | Sol + user | Not started | Not applicable | No run | Authorized implementation and synthetic gates. |

### 0.3 Resume and interruption procedure

1. Read the spec, this plan, repository instructions, and latest log entry.
   Confirm existing authorization; do not request the same approval again.
2. Inspect branch/HEAD, staged and unstaged diffs, and owned files against the
   snapshot. Preserve partial work. Clarify unexpected edits/uncertain ownership.
3. Inspect any recorded live agent, command, or analysis job before replacing
   it. Lost output does not establish process death; avoid duplicate writers/runs.
4. Verify recorded commits and evidence. Reproduce a bounded test only when its
   result is missing, interrupted, or invalidated by subsequent relevant edits.
5. Resume the unfinished gate: write tests -> verify RED -> commit tests ->
   implement -> verify GREEN -> review. Do not restart completed packages or
   rerun experimental preparation merely to recover coding context.
6. Update the snapshot with owner, file scope, phase, and exact next command
   before assigning work. Record tool session/PID/log path for running commands.

### 0.4 Evidence log

The lead updates current state and appends dated records after decisions,
RED/GREEN commits, reviews, worker replacement, and experimental validation.
Correct mistaken historical claims with an appended correction.

```text
YYYY-MM-DD HH:MM TZ - package / phase
Authorization; actual lead/worker model, effort, agent ID, writable files:
Start/end HEAD; relevant worktree state:
Exact RED/GREEN/review commands, counts, warnings, commits:
Interface/scientific changes or unresolved findings:
Running job/process, session/output/log paths; measured resources or none:
Next exact action, owner, and any missing authority:
```

### 2026-11-03 - WP0 / live-handoff and agent-assignment revision

- Authorization: user requested resumable implementation documentation and
  economical Sol/Luna task divisions. Scientific choices were already confirmed.
- Work performed: inspected the existing pooled-decoding/neural-regression
  handoff patterns and LFP single-writer/review rules; adapted them to this
  plan. Assigned responsibilities without launching a worker.
- Baseline: branch `refactor`, HEAD
  `d435b42046e96d529b1f75116c79a2be8c72cb97`; preserve the staged timeline
  specification and the tracked/untracked user work listed in 0.1.
- Implementation/RED/GREEN/commits: none. No source/test edits, model fitting,
  cluster work, or experimental output was produced.
- Documentation verification: `uv run python -c` assertions passed for ASCII,
  four existing document links, balanced code fences, trailing whitespace,
  all nine ledger entries, Sol/Luna assignments, and the confirmed scientific
  settings. `git status --short` confirmed that the two feature documents
  remain untracked; `git diff --cached --stat` confirmed that only the
  pre-existing timeline specification is staged. No files were staged or
  committed during this revision.
- Next action: user review of the updated plan; after implementation
  authorization, Sol begins WP1 tests under the single-writer policy.

### 2026-11-03 - WP0 / standalone handoff audit

- Authorization: user requested another review for completeness, correctness,
  and conciseness before implementation in another task.
- Findings addressed: explicit configuration/role precedence; raw-source-free
  display of relocated results; missing-fold indexing guards; constant-label
  ridge fits; independent numerical test expectations; stable contribution
  scales; actual source snapshots and concrete test commands.
- Code inspection also found strict missing-block rejection in the reused
  target loader and a raw-behavior availability gate before timeline dispatch.
  The narrow compatibility changes and tests below are required to honor the
  already confirmed missing-block and saved-display contracts.
- Condensed repeated handoff/orchestration prose while preserving the earlier
  evidence entry. Scientific choices and package/model assignments are unchanged.
- Verification: `uv run python -c` assertions passed for ASCII, four document
  links, code fences, trailing whitespace, sections 0-12, all nine ledger
  entries, required corrections/settings, and referenced existing APIs/gates.
  No pytest suite was run for this documentation-only review.
- Conciseness: reduced the prior 809-line plan by approximately 23 percent,
  retaining the original handoff evidence and adding the identified fixes.
- `git status --short` and `git diff --cached --stat` confirm the same unrelated
  staged/unstaged tracked changes; both feature documents remain untracked.
  No source/tests, experimental work, workers, staging, or commits.
- Next action: hand off this reviewed plan/spec; the receiving task starts
  WP1 RED when instructed to implement.

## 1. Outcome and boundaries

Add individual choice-window predictions beneath the existing session events,
using saved session-wide regional PCs. Fit correct-rewarded readouts once per
held-out block group, predict all valid choice conditions, and overlay actual
trial values. PC activity/contribution heatmaps are optional.

Use functions, narrow array contracts, and data-only records. Separate input
loading, fitting, persistence, plotting, and UI. Reuse existing libraries and
timeline navigation. Do not introduce estimator/track frameworks, runtime
checkpoint services, new dependencies, or cluster resource changes.

The existing decoder's elastic-net models, pooled `full` window, and saved
schemas retain their meanings. Place new code in
`src/neural_analysis/task_decoding/session_prediction/`. One necessary exception
is the backward-compatible missing-block option in existing `targets.py`
(Section 4). Preserve default behavior and existing result readability. That
file is in the old decoder's source fingerprint: old in-progress runs still
need their original scientific checkout for strict resume; do not claim that
changing it leaves old execution identities unchanged.

## 2. Confirmed scientific settings

| Item | Contract |
| --- | --- |
| Targets, canonical order | `current_state`, `current_action`, `qlearning_relative_value`, `forgetting_q_relative_value`, `hmm_signed_belief`, `hmm_decay_signed_belief`, `relative_doubt`. Exact source columns/orientations are in the spec. |
| Regions / features | Separate PFC and HPC models; all saved session PCs, unwhitened, with no further PC or target scaling. No combined model. |
| Bins | Default 500 ms; 100 ms selectable. Retain existing 50/20 ms support. Each resolution uses its matching session-PCA archive. |
| Training / prediction | Pool `[-1, +1)` choice-relative training bins from correct-rewarded trials; predict each `[-2, +2)` choice window; retain intertrial gaps. |
| Models | Continuous ridge `alpha=1`; binary L2 logistic `C=1`; unpenalized intercepts. No tuning. |
| Folds | One deterministic five-fold behavioral-block map, shared across targets and regions, before condition/target filtering. |
| Missing/insufficient blocks | Missing-block choices keep behavior/actual values but no held-out prediction. Fewer than five blocks stops preparation explicitly. |
| Display | Actual-value overlay, optional raw/weighted PC rows, original/contribution order fixed across navigation. Start inside **Session timeline and PCA**; reconsider a separate view if crowded. |
| Validation | CT026 2026-08-04, initially at 500 ms, after implementation/synthetic gates. |

For each group, train on correct-rewarded trials from the other four and
predict all valid windows in that held-out group. Groups may contain blocks
from different session times; training can include later blocks. Each choice
uses exactly its held-out model, not an ensemble containing models trained
on that choice. There are at most `7 * 2 * 5 = 70` fits.

Interpretation to retain in captions/summary:

- PCA/scaling use the whole session, including held-out neural observations.
  Only supervised fitting is block-held-out; this is not a fully held-out
  preprocessing pipeline or a future-only forecast.
- State/action labels normally coincide on correct-rewarded trials. Identical
  eligible training rows/settings should yield identical models, even when
  actual destination labels differ.
- With a full orthonormal basis, isotropic L2 on all unwhitened PCs is L2 on
  the same standardized unit activity in rotated coordinates, not dimensional
  reduction. PC signs are arbitrary; signed weight-times-score contributions
  are invariant to consistently reversing a component's sign.

## 3. Modules, interfaces, and ownership

New package files and their tests reside under `task_decoding/session_prediction/`
in the source and `src/tests/neural_analysis/` trees respectively.

| Owner / package | Source responsibility | New tests |
| --- | --- | --- |
| Sol / WP1 | `config.py`, `preparation.py`, package `__init__.py`, and narrow existing `task_decoding/targets.py` option: settings, aligned targets, regional maps, folds. | `test_config.py`, `test_preparation.py`, existing `task_decoding/test_targets.py` regressions |
| Sol / WP2 | `modeling.py`: data-only result records, weighted fits, predictions, contribution summaries. | `test_modeling.py`, `test_contributions.py` |
| Sol / WP3a | `results.py`: NPZ/config validation, identity, save/load/discovery. | `test_results.py` |
| Luna / WP3b | `run_session.py`, `run_batch.py`: preparation/reuse composition, logs/summary, CLI, thin session-list wrapper. | `test_scripts.py` |
| Sol / WP4a | `plotting.py` and localized changes to `population/timeline_plotting.py`: prediction rows, safe contribution arrays, PC ordering/geometry. | `test_plotting.py`, bounded existing timeline regressions |
| Luna / WP4b | `webapp/session_prediction_view.py`, localized `timeline_view.py` and `session_inputs.py` integration: saved discovery/availability, explicit Prepare, controls/cache. | `src/tests/neural_analysis/test_session_prediction_webapp.py`, existing timeline routing regressions |
| Luna / WP5a | Relevant `SESSION_TIMELINE.md`, neural-analysis README, and task-decoding README sections. | Documentation checks |
| Sol / WP5b | Authorized single-session validation and handoff evidence; user inspects output. | Existing numerical reconstruction checks |

Core interfaces to settle in WP1/WP2: `prepare_prediction_inputs(...)`,
`fit_region_prediction_models(...)`, `predict_region_windows(...)`; WP3 freezes
save/load/find signatures before runner delegation. Record inputs, shapes,
units, and return values in docstrings. Numerical functions accept prepared
arrays, not session objects or UI state.

Keep new plotting helpers independent of `timeline_plotting.py` to prevent
circular imports. A small data-only prediction-display record may group the
saved result, selected target, and heatmap mode. Extend existing plot APIs
with optional defaults; callers without predictions retain current behavior.

## 4. Configuration, inputs, and row alignment

Default optional control file: `session_prediction_config.json` beside metadata.
Its initial allowed fields/defaults are:

```json
{
  "base_config_path": "task_decoding_config.json",
  "bin_width_ms": 500,
  "ridge_alpha": 1.0,
  "logistic_C": 1.0
}
```

Absence uses these defaults; an existing malformed file/unknown key is an
error. Penalties must be finite positive scalars, bin width one of the four
supported values. Targets, all-PC policy, five folds, and window bounds are
fixed for this revision. Resolve input locators relative to the control file,
contained in the metadata session. No metadata schema change is needed.

Precedence: fixed scientific contract -> default/control-file settings ->
explicit GUI bin/probe selection. Overrides apply only on Prepare and are
saved as effective configuration. A saved result's recorded settings drive
its display; current widgets must not relabel it as another resolution/basis.
The metadata-only CLI uses version-3 regional role defaults. With version-2
metadata lacking roles, explain that explicit existing GUI probe selection
or metadata role setup is needed; do not guess probe/anatomical mappings.

During preparation, load the base `TaskDecodingConfig`, check that it belongs
to the selected session, and make a local `dataclasses.replace` selecting the
seven targets and `choice_time`. Call public `build_target_table()` and
`build_condition_masks()`. Use its augmented CSV/feature-parameter inputs,
not its estimator, PC counts, condition subset, bin width, or bilateral
coverage requirement. Missing prerequisites produce an actionable error;
preparation does not fit behavioral models or run the old neural decoder.

Compatibility change required: `validate_augmented_trials()` and
`build_target_table()` currently require every block ID. Add keyword-only
`allow_missing_blocks=False` to both and forward it to their shared block
validator. Default callers retain their current strict behavior. The new
analysis passes True; present block labels must still form contiguous
segments when missing entries are ignored. Do not drop/renumber source rows,
fabricate block IDs, or duplicate the target derivations in a new loader.

Align the augmented table and source timeline by original row position,
trial ID, action, and choice timestamp. IDs/positions must agree exactly;
compare UTC seconds with `rtol=0` and a small explicit absolute tolerance
(e.g. 1 microsecond for CSV round trips), never a relative tolerance on epoch
seconds. Reject reordered or mismatched inputs. Build targets before filtering.

Reuse `interregional.preparation.build_block_fold_assignment()` on the full
chronological augmented table. It already supplies normalized identities,
five deterministic `GroupKFold(shuffle=False)` folds, and missing-block handling.
First map project-missing string sentinels to NaN in a local grouping copy
using existing `is_present_value()` semantics; the fold helper checks actual
nulls and must not treat a sentinel string as another behavioral block.
Map its rows to timeline choices and regional `choice_positions` explicitly.
Do not reuse the old target-dependent scorer's split validation: requiring
valid variable test labels/both test classes would suppress usable predictions.

Regional coverage remains independent. Missing actual targets exclude that
target's training rows but not otherwise predictable held-out windows.

## 5. Fitting and contribution contracts

Use N shared timeline choices, T=7 targets, F=5 folds, B prediction bins, and
K_r all saved PCs in region r. Regional scores are float64 `(W_r,B,K_r)`;
`choice_positions` is int64 `(W_r,)` into N. Targets/validity have shape `(N,T)`,
baseline/correct-rewarded/fold IDs `(N,)`, edges `(B+1,)` seconds. Fold `-1`
means no held-out assignment. Scores retain source units and are never mutated.

Select bins fully inside `[-1,+1)` and verify those boundaries exist. Counts
are training/prediction: 500 ms `4/8`, 100 ms `20/40`, 50 ms `40/80`, 20 ms
`100/200`. For a target/region/fold, intersect regional coverage, baseline,
correct-rewarded membership, valid target, and nonnegative fold different
from the held-out fold. Reuse `pooled.decoding.stack_training_window()`:
trial-major rows, repeated trial-i target, weights `1 / training_bin_count`.

Installed scikit-learn 1.8.0 source/signatures were inspected during planning;
recheck the installed API before implementation if the environment changed:

- `Ridge(alpha=1.0, fit_intercept=True, solver="svd")`, weighted fit.
- `LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", fit_intercept=True,
  class_weight=None, tol=1e-4, max_iter=5000)`, weighted fit. Omit the deprecated
  `penalty` parameter. Use no class balancing/calibration or solver fallback.

Their objectives, with an unpenalized intercept, are:

```text
ridge: sum_n w_n * (y_n - b - X_n beta)^2 + alpha * ||beta||^2
logistic: sum_n w_n * binary_log_loss(y_n, b + X_n beta) + ||beta||^2/(2*C)
```

Each trial has total weight one regardless of bin count. Continuous targets
stay in native units. A nonempty finite continuous training set is sufficient
for a ridge fit; constant training targets are valid and can yield a constant
intercept-only prediction. Do not import target-variance or two-test-trial
requirements from R2 scoring. Binary training must contain both classes.
Save independent training-trial/stacked-row counts so weak sampling is visible.

Treat empty training, single-class binary training, nonconvergence, numerical
fit failure, or nonfinite parameters as model-unavailable with a readable
reason. Continue other fits. Invalid input structure raises before fitting;
do not catch all errors indiscriminately. Check finite outputs and `[0,1]`
binary probabilities. Unexpected nonfinite predictions invalidate that
model's destination windows, with a recorded reason rather than a fallback.

Predict all B bins of each assigned held-out window even when its actual
value is missing or constant. Binary output is class-1 (left) probability;
continuous output is `b + sum_j beta_j*z_j`. Logistic contributions add to
logit, then `scipy.special.expit` gives probability. Do not clip ridge outputs.

For each fitted target/region/fold, compute `s_j = abs(beta_j) * std(z_j)` on
its pooled supervised training rows (`ddof=0`). Save the equal mean across
valid folds, contributing fold count, and descending PC order with original
rank breaking ties. This measures varying additive contributions, not unique
importance or an additive share of output variance. Mean offsets remain in
the original affine prediction; the ranking describes variation only.

## 6. Saved arrays, identity, and publication

Analysis version `condition-session-prediction-v1`; NPZ schema 1. Use named
arrays and dictionary-valued `meta`, following trusted-project timeline IO.
Do not pickle estimator instances. Choices retain the source structured table.

| Shared name | Dtype / shape | Meaning |
| --- | --- | --- |
| `choices` | Structured, N rows | Original choice identity, action, UTC/elapsed time, reward when present. |
| `target_names`, `target_families` | Unicode `(T,)` | Canonical order/families. |
| `target_values`, `target_valid` | float64/bool `(N,T)` | Native actual values/validity. |
| `baseline_valid`, `correct_rewarded` | bool `(N,)` | Independent input masks. |
| `block_values_json`, `fold_ids` | Unicode/int64 `(N,)` | Canonical block identity; fold `-1` for missing. |
| `bin_edges_s`, `training_bin_mask` | float64 `(B+1,)`, bool `(B,)` | Exact grid and supervised subset. |
| `meta` | Dictionary | Versions, configuration, provenance, units/axes, status, timing. |

For each source-available region, prefix the following with `pfc_` or `hpc_`:

| Suffix | Dtype / shape | Meaning |
| --- | --- | --- |
| `choice_positions`, `pc_ids` | int64 `(W_r,)`, `(K_r,)` | Choice map and original one-based PC ranks. |
| `predictions` | float64 `(W_r,B,T)` | Held-out values; NaN where unavailable. |
| `prediction_reason` | Unicode `(W_r,T)` | Empty if valid, otherwise missing-block/model/output reason. |
| `coefficients`, `intercepts` | float64 `(T,F,K_r)`, `(T,F)` | PC-scale coefficients, native/logit intercepts. |
| `fit_status`, `fit_reason` | Unicode `(T,F)` | `ok`/`unavailable` and readable reasons. |
| `training_choice_mask` | bool `(T,F,N)` | Exact choices used to train each model. |
| `training_trial_count`, `training_row_count` | int64 `(T,F)` | Independent trial / stacked row counts. |
| `training_pc_sd`, `contribution_strength` | float64 `(T,F,K_r)` | Training SD and absolute-coefficient-times-SD. |
| `mean_contribution_strength`, `contribution_pc_order` | float64/int64 `(T,K_r)` | Mean strength and ordered original PC IDs. |
| `contributing_fold_count`, `contribution_color_limit` | int64/float64 `(T,)` | Valid ranking folds; fixed positive weighted-heatmap limit. |

Unavailable models have NaN parameters/summaries and explicit reasons; counts
remain factual. If no fold fits, mean strength is NaN and display order falls
back to original PC order. Missing regions omit regional arrays and report
why in `meta.regions`; behavior remains displayable. Validate shapes, indices,
masks, finite/NaN conventions, probabilities, fold exclusion, and counts.

`meta.source_pca` records session-relative archive path, configuration ID,
size and SHA-256. Record target source identities, class-1 labels, units,
window/penalty/solver/weighting definitions, and numerical library versions.
Store source/code identities for preparation reuse; include the actual
computational modules/reused scientific dependencies, exclude display-only
files and unrelated metadata. Git HEAD/absolute execution paths are provenance,
not reasons to reject an otherwise valid saved display. Different numerical
runtime settings can distinguish new fits without preventing old result loading.

Reference the original PCA archive rather than copying its scores into every
prediction result. A relocated session must include that archive at the same
relative location; verify its bytes once per cached load. Discovery reads
small run/config metadata, then loads/validates only the selected result/source.
Do not parse raw behavioral configs or inspect spikes for ordinary display.

Output directory under the session, outside the repository:

```text
analysis_runs/condition_session_prediction_<date>_<time>_<identity>/
    config.json     results.npz     summary.md     run.log
    run_session.py  run_batch.py    source_snapshots/
```

Capture the new computational modules used, not just the thin runner, in
`source_snapshots/`; record reused-helper source hashes and code revision.
Write config/log/summary and validate arrays before atomic publication of
`results.npz` using a temporary file in that directory. Discovery excludes
incomplete directories. Reuse completed matching identity by default; explicit
rerun creates another directory. No receipt registry, locks service, automatic
retry, or runtime checkpoint engine. An interrupted numerical fit is restarted
on a later explicit preparation; compatible source PCA is still reused.

## 7. Timeline integration and portable saved display

Add a compact prediction control group within the existing timeline view:
**Prepare predictions**, **Recompute predictions**, saved result/target
selection, and PC mode/order/count controls. Prepare computes all seven
specified targets for the selected source configuration. A deliberate Prepare
can reuse/prepare missing PCA through the existing runner; widget changes
alone never fit. Preserve existing PCA preparation and optional speed controls.

Important existing-code gap: `timeline_view.py` currently computes current
raw-source configuration before loading a timeline. Its identity contains
absolute paths, so copied archives may not match. WP4b must discover/select
saved predictions before that preparation-only validation, then load the
exact PCA archive pinned by the selected result. Validate session ID,
recorded identities, resolution, and region assignments from saved products.
Do not silently choose a different basis or run legacy reward enrichment from
raw tables for this saved path. Use rewards already saved in prediction
choices; absent optional legacy values remain unknown.

Also update the earlier `session_inputs.metadata_view_availability()` gate:
allow this view when a completed saved prediction is discoverable even if raw
behavior is missing. Use small discovery metadata, not NPZ array loading.
Without a saved result, retain ordinary source availability behavior. Test
the actual app dispatch as well as direct renderer calls so this gate cannot
silently defeat the portable saved path.

The saved display path must work with metadata, prediction run, and referenced
PCA archive even if raw neural/behavior sources and original cluster paths
are absent. Show the archive's actual settings. Keep the existing current-input
preparation path intact when creating a new configuration; a selected archived
result does not authorize recomputation against its old absolute paths.
Optional unprepared speed/window summaries must not block saved predictions.

Rows: choices/events/context/optional speed, PFC prediction, HPC prediction,
optional regional PC heatmaps, existing navigator. Show one target at a time.
Actual trial-i values repeat within its window. Use post-step bin segments
and explicit breaks: no line stretches across intertrial gaps. Missing target,
missing prediction, and a genuine zero remain distinct. Binary axes show
left probability `[0,1]`; continuous axes show native units. Hover retains
trial/time/condition/fold and PC/weight/contribution details.

For weighted rows, initialize NaN and first mask `fold_id >= 0`, valid fit,
and an empty prediction reason. Only then index that fold's coefficient
array: NumPy `[-1]` must never select
the last model for a missing-block choice. For valid rows use
`contribution[w,b,j] = score[w,b,pc_id[j]-1] * beta[target,fold[w],j]`.
All-PC contributions plus intercept reconstruct output/logit; a visible subset
need not reconstruct the whole prediction. No full target/fold/window/PC
contribution tensor is stored.

Raw mode optionally uses existing display-only PC z-scores; weighted mode
retains target/logit units. Contribution order is global for each target/region;
keep original PC IDs in labels. Original order uses rank ranges; strength
order shows top N. Use a zero-centered weighted color scale fixed across
viewports/visible-row changes: persist the maximum absolute valid contribution
over all predictor PCs/destination windows, or 1 if all are zero/unavailable.
Compute it by fold/PC or bounded chunks, without a full contribution tensor.

Reuse the existing chart component/navigation. Target/PC/mode changes reset
vertical ranges only; Jump/start/duration explicitly update horizontal range.
Keep all-session navigation reachable with bounded neural loading at fine
resolution. Inspect the integrated controls for crowding before adding a view.

## 8. Runner and batch behavior

Proposed runner boundary:
`prepare_session_predictions(metadata_path, *, config_path=None,
probe_roles=None, bin_width_ms=None, rerun=False, dry_run=False)`.
GUI selections are explicit overrides; metadata-only CLI uses Section 4 defaults.
Both call this same composition root. Ordinary commands remain:

```bash
uv run python -m src.neural_analysis.task_decoding.session_prediction.run_session \
  --session-metadata "$METADATA" --dry-run
uv run python -m src.neural_analysis.task_decoding.session_prediction.run_session \
  --session-metadata "$METADATA"
```

Optional `--config` selects another controls file; `--rerun` creates a new run.
Dry-run reads small controls/tables and source metadata, reports grids,
regions, model counts/expected array bytes, and any missing PCA prerequisite;
no source PCA fitting, spikes, writes, or submission. Never call the old
neural decoder to prepare these targets. GUI computation is explicit with
status/log output, not triggered by a dropdown.

The batch wrapper takes metadata paths and calls the same runner, parallel
across sessions using the existing timeline batch worker-count convention.
Do not add model-level pools or a new memory scheduler. No batch is run during
this feature's initial validation.

## 9. Test-first work packages and acceptance

For code packages: write meaningful tests -> establish RED -> commit tests ->
implement -> focused GREEN -> review -> update Section 0. Sol commits only
owned paths and preserves unrelated staged files. Internal gates within an
approved scope do not require repeated user permission.

- **WP1 (Sol):** exact seven-target orientation; mismatched/reordered source
  rows and absolute UTC tolerance; config precedence/defaults; full-table
  block exclusion shared across targets/regions; missing/insufficient blocks;
  missing-block keyword preserves existing strict target validation and
  normalizes string sentinels for grouping; per-region coverage and all grids.
- **WP2 (Sol):** training masks/row weights and no held-out labels in fits;
  compare ridge coefficients/intercept to an independent weighted normal-
  equation oracle (ridge shrinks, so do not demand exact unpenalized recovery);
  oriented logistic probabilities; constant continuous training labels remain
  valid; empty/single-class/nonconverged fits; missing destination labels;
  identical state/action fits; unchanged PCA; contribution reconstruction,
  scale-aware ranking/ties, missing-fold guard and global color limits.
- **WP3a (Sol):** all schema/identity invariants and save/load round trip;
  missing/corrupt referenced PCA; nonfinite outputs and probability bounds;
  matching reuse versus scientific changes; relocation; partial publication;
  no invalidation from display-only changes/unrelated metadata edits.
- **WP3b (Luna, Sol review):** synthetic end-to-end prepare/reuse/rerun;
  dry-run writes/fits nothing; absent PCA prepared only on explicit action;
  metadata/probe/bin precedence; no duplicate estimator logic; thin batch
  delegation; source snapshots and logs. Add required stubs only after RED
  tests are committed, not before.
- **WP4a (Sol):** real bin edges/gaps, zeros versus unavailable, actual overlays,
  unequal regional coverage, fold-specific weighted rows and original PC IDs;
  saved-target units, fixed order/limits, no fitting during plotting; existing
  plot callers retain default behavior.
- **WP4b (Luna, Sol review):** actual Streamlit saved path with removed raw
  inputs and relocated archive, including availability/app dispatch, not just
  mocked loader tests; explicit Prepare;
  saved target/event/PC changes cause no fitting/writes; persistent pan/Jump;
  missing optional tracks; defaults without prediction controls remain usable.
- **WP5a (Luna, Sol review):** update usage/copy-back requirements and scientific
  interpretation without deleting existing instructions.
- **WP5b (Sol + user):** after synthetic gates and authorization, prepare CT026
  2026-08-04 at 500 ms; verify original-row alignment, fold training counts,
  no block overlap, and selected saved predictions against reconstruction.
  Inspect both regions and binary/continuous targets, contribution ordering,
  gaps, and control crowding; user accepts visual/scientific results.

First RED command after creating WP1 tests (paths are intentionally new):

```bash
uv run pytest -q src/tests/neural_analysis/task_decoding/session_prediction/test_config.py \
  src/tests/neural_analysis/task_decoding/session_prediction/test_preparation.py
```

At integration, run the new package and directly affected existing interfaces:

```bash
uv run pytest -q src/tests/neural_analysis/task_decoding/session_prediction \
  src/tests/neural_analysis/task_decoding/test_targets.py \
  src/tests/neural_analysis/test_session_prediction_webapp.py \
  src/tests/neural_analysis/test_session_timeline.py \
  src/tests/neural_analysis/test_session_timeline_io.py \
  src/tests/neural_analysis/test_timeline_plotting.py \
  src/tests/neural_analysis/test_session_timeline_webapp.py \
  src/tests/neural_analysis/test_session_timeline_scripts.py \
  src/tests/neural_analysis/test_timeline_chart.py
```

Use `uv run` for Python commands and check changed Python with Ruff if available,
plus changed-path whitespace checks. Broaden tests only for additional changed
interfaces/failures; do not run the entire neural suite at every package gate.
A new browser inspection verifies frontend pan/zoom that Python tests cannot
establish. Record exact outcomes and warnings, never anticipated pass counts.

## 10. Performance and one-session validation

Prediction arrays for 1,000 windows per region and seven targets use about
0.85 MiB at 500 ms, 4.27 MiB at 100 ms, 21.36 MiB at 20 ms (float64, two
regions). These are estimates, not peak-memory/runtime measurements. The main
costs can be full PCA loading/preparation, a stacked training matrix, repeated
SVD fits, and browser payloads. Release temporary matrices between models;
optimize/reuse factorizations only after measurement and correctness checks.

Reuse compatible source PCA; form only the visible target/PC contributions.
Report fit count, preparation/load/fit/save/render times, peak process RSS,
NPZ and chart payload bytes, warnings, and exact run/log paths. At 100 ms the
existing option must pass synthetic/UI tests; initial experimental validation
is 500 ms, not an automatic second-resolution or multi-session run.

Designated root:
`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260804_latent_inference`.
Use its existing `neural_session.json`; do not rewrite session metadata or old
outputs to make validation pass. If preparation is long, run it through the
established external execution workflow and record the process/job identity;
await completion outside an active coding turn. No new Slurm resources are
implied. User inspection precedes processing further sessions.

## 11. Implementation authorization

Scientific choices and agent divisions are confirmed. This audit edits only
documentation. The receiving task begins WP1 when the user instructs it to
implement this plan; no additional per-package approval loop is required.
Code completion does not authorize new sessions, cluster submissions, resource
changes, or transfers. Record authorized scope and experimental-run state in
Section 0 rather than inferring them from a worker report.

## 12. Sol/Luna execution and handoff rules

- **Sol lead (`gpt-6.1-sol`, high):** implements WP1/WP2/WP3a/WP4a, reviews
  Luna's integration, owns scientific correctness, commits, and live records;
  runs authorized WP5b validation. No second Sol/reviewer agent is needed.
- **One optional Luna (`gpt-6-luna`, medium):** WP3b/WP4b/WP5a after Sol freezes
  their contracts. Reuse it sequentially; Sol may do tiny tasks directly when
  delegation would cost more than the work. No worker-created agents.
- **Maximum two active agents including lead**, ordinarily one for scientific
  work. The role labels do not change the current chat's model. Record actual
  model/effort/agent ID; report unavailability instead of silently substituting.

Dispatch Luna with package/RED-or-GREEN phase, exact writable files from
Section 3, frozen signatures/acceptance conditions, focused command, HEAD,
and user changes to preserve. Luna returns changed files, actual checks,
findings, and next action; it does not expand scientific/file scope.

Luna writes tests first, then Sol verifies RED and commits them before a GREEN
assignment. Sol reviews source/results before acceptance. Record Sol's own
scientific work as self-review backed by synthetic/oracle checks, not an
independent reviewer. Scientific issues found in a Luna task return to Sol.

Only the lead stages/commits. Keep the worker idle before reviewing/committing
its changes. Use explicit paths and path-limited commits when unrelated files
are already staged. Never amend/reset/revert user work or discard partial
feature edits. No two agents edit the same files concurrently; end/narrow a
worker assignment before Sol writes there.

Order: `WP1 -> WP2 -> WP3a -> WP3b/review -> WP4a -> WP4b/review -> WP5a/review
-> WP5b/user inspection`. Update Section 0 after each gate and interruption.
Tests/commits plus this document are sufficient coordination; no per-file
locking system, speculative workers, or parallel worktrees are needed.
