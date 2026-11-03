# Condition Prediction Generalization: Implementation Handoff

Status: implemented and numerically validated, 2026-11-03. Scientific choices are confirmed.
WP1-WP5a complete. Designated CT026 preparation/browser inspection complete;
the user's visual/scientific acceptance is pending. No experimental batch run.

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
| Date / branch / inspected HEAD | 2026-11-03 / `refactor` / `f7cd9c8` (final display fix; CT026 computation used `2c39850`). Recheck on resumption. |
| Phase / authorization | User-authorized implementation and designated single-session validation complete. User inspection pending; no broader sessions, batch, transfer or cluster operation. |
| Scientific settings | Confirmed; see Section 2. No outstanding scientific question. |
| Source/tests | session_prediction package and localized target/timeline/webapp integration complete; 173 planned integration tests pass. |
| Documentation | Reviewed local spec/plan revisions were preserved and committed with feature work. Final validation evidence appended below; feature commits have not been pushed. |
| Existing user changes | Staged: added `docs/session_timeline_pca_spec_v1.md`. Unstaged tracked: `src/main.py`, `src/neural_analysis/sync_ephys.py`. Preserve these and unrelated untracked files. |
| Team / writer | Current chat lead is the sole writer; no worker launched. Runtime model/effort identity is not exposed to the agent. |
| Running analysis | No analysis running. CT026 result published; local saved-result preview at `http://127.0.0.1:8517`, exec session `99148`, watcher disabled. Earlier preview sessions stopped. |
| Latest verification | 173 passed in 8.66 s with bundled Node; Ruff/whitespace clean. CT026: 70/70 fits, 715/715 windows in each region, exact continuous reconstruction, no missing predictions. |
| Next action | User inspects CT026 preview/summary. Resume from saved products; no rerun or broader session processing without a new request. |

### 0.2 Package ledger

Fill evidence as work proceeds. Worker completion is not Sol acceptance;
experimental visual acceptance belongs to the user.

| Package | Writer | State | RED commit / command / result | GREEN commit / checks / review | Next gate |
| --- | --- | --- | --- | --- | --- |
| WP0 documentation | Lead | Feedback revision verified | Not applicable | Baseline `9e46967`; checked review corrections uncommitted | Receiving task begins WP1 on implementation instruction. |
| WP1 settings/alignment/folds | Lead | Accepted | `fa67c05`; 2 expected collection errors | `0a8c65f`; 64 passed with legacy targets | WP2. |
| WP2 models/contributions | Lead | Accepted | `d8561d4`; 2 expected import errors | `3f70f6b`; 74 passed; Ruff clean | WP3a. |
| WP3a persistence/identity | Lead | Accepted | `17d94ef`; 1 expected import error | `6e4afe5`; 86 integrated tests passed | WP3b. |
| WP3b runner/CLI/batch | Lead | Accepted | `62a5ae2`; 1 expected import error | `6e3fc86`; 98 integrated tests passed | WP4a. |
| WP4a plotting/geometry | Lead | Accepted | `91da846`; expected import error | `435fd69`; follow-up hover test `0f77f29`; all 172 integrated tests pass | Browser checks completed. |
| WP4b webapp integration | Lead | Accepted | `1aa86bd`, `0f77f29`, `87ef8b3`; final control test `1cce9ad` | `2c39850`, `f7cd9c8`; 173 tests pass; saved-only browser display/pan checked | User inspection. |
| WP5a usage documentation | Lead | Complete | Not applicable | `2c39850`: timeline guide + neural/task-decoding README updates | User usage/review. |
| WP5b CT026 inspection | Lead + user | Numerical/browser validation complete; user acceptance pending | Not applicable | 70 fits; 715 windows/region; exact reconstruction; performance below | User inspects saved preview/summary. |

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

### 2026-11-03 - WP0 / implementation-chat review follow-up

- Baseline: spec/plan committed at `9e46967` and pushed to `origin/refactor`;
  refreshed the live snapshot without rewriting older untracked-state records.
- User decisions: continue other targets when a continuous column is entirely
  missing; save/show every applicable canonical condition membership.
- Plan corrections: strict-default opt-in empty-target support; saved condition
  arrays and tests; sequential heatmap display coordinates preserving PC IDs;
  explicit full-base-config prerequisite; distinction between full NPZ memory
  loading and viewport-limited chart/contribution work.
- Read-only source inspection confirmed the validator, heatmap coordinates,
  config prerequisites, and loader behavior. Source/test directories for this
  feature are still absent. No worker, fitting, commit, or push in this revision.
- Verification: `uv run python -c` assertions passed for ASCII, ten local
  document links, code fences/whitespace, all five corrections, current
  committed-baseline wording, canonical six-condition order, all sections,
  and the nine-row ledger. Changed-document `git diff --check` passed;
  `git status --short` confirms only the two feature documents changed by
  this revision alongside the preserved user edits. No pytest suite was run.
- Next action: implementation task uses these updated documents for WP1 RED;
  these feedback changes are not yet part of the pushed baseline commit.

### 2026-11-03 - WP0 / PC indexing and fixed-penalty clarification

- Corrected displayed contributions to look up scores and coefficient columns
  by the same original PC ID. Added a nonmonotonic-selection test with distinct
  coefficients and explicit expected values, alongside geometry/label checks.
- Aligned the control file with the confirmed specification: alpha=1 and C=1
  are constants, recorded in result provenance but not user configuration.
  Removed the penalty input fields and added their rejection to WP1 tests.
- Module boundaries and Sol/Luna ownership are unchanged. No source/test edits,
  computation, delegation, staging, commit, or push in this revision.
- Verification: `uv run python -c` assertions passed for ten document links,
  ASCII/whitespace/fences, the two-field JSON controls, constant-penalty wording,
  original-PC weight lookup, reordered contribution values, and ledger rows.
  Changed-document `git diff --check` passed. No pytest suite was run.
- Next action: hand off the checked local updates, then begin WP1 only on
  implementation instruction. Follow-up edits remain uncommitted/unpushed.

### 2026-11-03 - WP1 / RED and GREEN

- User authorized implementation through as much of the plan as practical.
  Current chat lead is the sole writer; model/effort identity is not exposed.
- RED: `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync --offline pytest -q
  src/tests/neural_analysis/task_decoding/session_prediction/test_config.py
  src/tests/neural_analysis/task_decoding/session_prediction/test_preparation.py`
  produced two expected missing-package collection errors. Tests committed as
  `fa67c05` before source implementation. Default uv cache was read-only;
  subsequent commands use the ignored project pytest cache.
- GREEN: same uv prefix, `pytest -q src/tests/neural_analysis/task_decoding/session_prediction
  src/tests/neural_analysis/task_decoding/test_targets.py`: 64 passed in 0.60 s.
  Source review checked row/fold/region mappings and strict-default preservation.
- No experimental work, worker, or background analysis. Next: source commit,
  WP2 numerical/oracle tests. Existing unrelated staged and unstaged work preserved.

### 2026-11-03 - WP2 / RED, GREEN and self-review

- RED: same offline uv/cache prefix as WP1, `pytest -q
  src/tests/neural_analysis/task_decoding/session_prediction/test_modeling.py
  src/tests/neural_analysis/task_decoding/session_prediction/test_contributions.py`:
  two expected import errors before source. Tests committed as `d8561d4`.
- GREEN/refactor: `pytest -q src/tests/neural_analysis/task_decoding/session_prediction
  src/tests/neural_analysis/task_decoding/test_targets.py`: 74 passed, latest 1.04 s.
  Ridge checked against independent weighted affine normal equations; logistic
  objective gradient checked; reconstruction uses original reordered PC IDs.
  Reviewed missing-block guards, target/region independence and source immutability.
- Rechecked installed Ridge/LogisticRegression signatures and pooled stacking
  source. Ruff on all new source/tests plus targets.py passed; whitespace clean.
- No independent reviewer, experimental run or worker. Next: commit numerical
  source and begin WP3a saved-result/identity/relocation tests.

### 2026-11-03 - WP3a / RED, GREEN and fixture correction

- RED: offline uv/cache prefix, `pytest -q
  src/tests/neural_analysis/task_decoding/session_prediction/test_results.py`:
  expected missing-results import error; tests committed as `17d94ef`.
- A genuine fixture error surfaced during GREEN: the synthetic event table
  had no columns, unlike the real timeline loader's empty event schema.
  Corrected only the fixture to retain Time/Event/Note/elapsed_s columns;
  no production workaround or requirement change. Persistence tests: 12 passed.
- Implemented exact-PCA content identity, named-array round trips, small JSON
  discovery, relocation without raw files, and scientific array validation.
  No estimator instances or redundant scores/contribution tensor stored.
- Ruff passed. Integrated WP1-WP3a plus existing target tests checked before
  source commit. Next: runner composition, dry-run and synthetic pipeline tests.

### 2026-11-03 - WP3b / synthetic runner and CLI

- Lead implemented this bounded package directly; no worker was launched.
  RED: offline uv/cache prefix, `pytest -q
  src/tests/neural_analysis/task_decoding/session_prediction/test_scripts.py`:
  one expected missing-runner import error. Tests committed as `62a5ae2`.
- GREEN: 5 runner tests passed in 0.74 s. Actual seeded spike files, metadata,
  six behavioral blocks and full seven-target fitting exercised at 500 and
  100 ms. Verified reuse, immutable reruns, independent regional selection,
  missing PCA preparation only on explicit action, snapshots, CLI and batch dry-run.
- Reused the existing timeline worker-count/session-list conventions. No model
  pools, scheduler, background process, experimental session or cluster work.
  Scientific source identity includes the actual runner as well as numerical
  modules/reused dependencies. Ruff passed before final integration check.
- Integrated command: same uv prefix, `pytest -q
  src/tests/neural_analysis/task_decoding/session_prediction
  src/tests/neural_analysis/task_decoding/test_targets.py
  src/tests/neural_analysis/test_session_timeline_scripts.py --tb=short`.
  Next: commit this package, then plot/geometry RED tests.

### 2026-11-03 - WP4a GREEN / WP4b RED

- WP3b integrated check completed: 98 passed in 3.52 s; source `6e3fc86`.
- WP4a RED test commit `91da846`; expected missing plotting import.
  GREEN command (same offline uv/cache prefix): `pytest -q
  src/tests/neural_analysis/task_decoding/session_prediction/test_plotting.py
  src/tests/neural_analysis/test_timeline_plotting.py --tb=short`: 34 passed.
  Ruff clean. Optional existing plot argument preserves default callers;
  new helpers do not import timeline plotting, preventing a circular dependency.
- WP4b RED: `pytest -q src/tests/neural_analysis/test_session_prediction_webapp.py
  --tb=short`: 2 expected failures, missing controls module and old availability
  rejection of a relocated saved-only session. No production UI change yet.
- Next: commit plotting source, commit WP4b tests, implement saved-first routing.
  Lead remains sole writer; no experimental analysis or cluster operation.

### 2026-11-03 - WP4b/WP5a / integrated GREEN and browser inspection

- Lead performed UI integration directly; no worker. WP4b RED tests committed
  `1aa86bd` before UI production code. Saved-first selection bypasses current
  raw-input validation and metadata availability gating only for saved products.
- Review refinements used RED tests before fixes: `0f77f29` (2 failures: heatmap
  prediction/actual hover and original weighted PC ranges), `87ef8b3` (1 failure:
  misleading raw-score color caption). All now GREEN. Compact dry-run reporting
  test `1a1dc7c` initially failed for missing unit_count, then reporting was fixed.
- Planned Section 9 integration command, with `UV_CACHE_DIR=.pytest_cache/uv`,
  offline/frozen/no-sync uv and `NODE_BINARY=/home/matt/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node`:
  172 passed in 8.36 s. Initial run without NODE_BINARY was 171 passed/1 skipped;
  bundled Node resolved the skip. Ruff clean on all changed Python source/tests.
- Browser inspection: loopback Streamlit at port 8517, exec session 43533,
  synthetic saved-only metadata `/tmp/pytest-of-matt/pytest-221/test_relocated_saved_display_a0/copied/neural_session.json`.
  Sandbox blocked the initial socket; an approved loopback-only escalation
  started the server. Browser showed PFC/HPC binary and continuous tracks,
  weighted PC rows/identities, real gaps and fixed color scales. Panned time
  ticks 20-70 s persisted across target and PC-mode/order changes. Proof image:
  `.pytest_cache/session_prediction_browser.png` (ignored synthetic artifact).
- Usage docs updated in SESSION_TIMELINE.md and the neural/task-decoding READMEs.
  No new dependencies, framework, scheduler or storage mechanism added.
- CT026 read-only CLI dry-run passed: 715 choices, 4 training/8 prediction bins,
  at most 70 models, 640640 prediction bytes upper bound, both metadata roles,
  existing source PCA `session_timeline_pca_20261030_115229_118901_269a76fce972be62/results.npz`.
  No experimental fit/write yet. Next: final source commit, designated validation.

### 2026-11-03 - WP5b / CT026 preparation and numerical/browser validation

- Single designated session only, at 500 ms, after implementation approval and
  synthetic/browser gates. Session root:
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260804_latent_inference`.
- Exact run: `analysis_runs/condition_session_prediction_20261103_141009_556738_3a20d0dc3d27c60f2466/`.
  Computational HEAD `2c39850`; the later UI-only fix does not invalidate its
  scientific identity. Existing PCA and metadata were reused without changes.
- Command: `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync --offline
  python -m src.neural_analysis.task_decoding.session_prediction.run_session
  --session-metadata <session-root>/neural_session.json`. Approved sandbox
  escalation permitted the new session-local output directory. Exec `5891`
  exited 0; actual elapsed 5.622 s. All 70 models fitted, no unavailable fits.
- PFC: 715 windows, 174 saved predictor PCs; HPC: 715 windows, 316 PCs.
  Predictions per region: (715,8,7). For all targets/regions, training trials
  across folds: [225,233,223,232,243]; four bins/trial. Saved loader validated
  counts, masks, block exclusion, class probabilities and coefficient reconstruction.
  Additional evaluation of saved continuous affine outputs had maximum absolute
  reconstruction error 0.0. State/action prediction difference was 0.0,
  consistent with identical eligible labels. No missing prediction bins.
- Run log: PFC fitting 1.551 s, HPC fitting 3.737 s; save/validation 0.062 s.
  Saved NPZ 865355 bytes. Peak process RSS 382390272 bytes (process lifetime).
  Separate warm/unspecified-filesystem-cache read: 0.135 s. Figure construction /
  serialization / JSON bytes: current_action hidden PCs 1.303 s / 0.279 s /
  4553578; relative_doubt weighted top 10 PCs 1.641 s / 1.226 s / 29735129.
  These measurements do not establish browser timings or other-session budgets.
- A second ordinary preparation request reused this same immutable result in
  0.402 s; no second fit/run directory. No batch, cluster job or transfer.
- Real browser displayed CT026 binary/continuous panels, actual overlays,
  optional saved speed/context, and true gaps. Inspection found a real UI
  mismatch: hidden PC rows still offered attached window-count controls.
  RED test `1cce9ad` failed as expected; minimal fix committed as `f7cd9c8`.
  Final Section 9 integration command with bundled NODE_BINARY: 173 passed in
  8.66 s. Ruff and changed-path whitespace checks passed.
- Initial preview emitted machine inotify-limit warnings; subsequent preview
  commands use `--server.fileWatcherType=none` without changing machine settings.
  Matplotlib's unwritable default cache warning was avoided on later commands
  with `MPLCONFIGDIR=.pytest_cache/matplotlib`; no external library changed.
- Final preview uses ignored `.pytest_cache/session_prediction_preview.py`,
  calling the actual timeline renderer directly to avoid unrelated raw-spike
  startup in the full app. Loopback port 8517, exec session 99148; no analysis
  running. Final CT026 screenshot: `.pytest_cache/ct026_session_prediction.png`
  (ignored preview artifact); browser tab retained for user inspection.
  User visual/scientific acceptance remains outstanding. Final spec/plan ASCII,
  local-link, code-fence and whitespace checks passed. Unrelated user work remains
  staged/unstaged exactly as before; no feature commits were pushed.

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
| All-missing continuous target | Retain that target as unavailable and continue the others; missing columns or malformed present values are input errors. |
| Regions / features | Separate PFC and HPC models; all saved session PCs, unwhitened, with no further PC or target scaling. No combined model. |
| Bins | Default 500 ms; 100 ms selectable. Retain existing 50/20 ms support. Each resolution uses its matching session-PCA archive. |
| Training / prediction | Pool `[-1, +1)` choice-relative training bins from correct-rewarded trials; predict each `[-2, +2)` choice window; retain intertrial gaps. |
| Models | Continuous ridge `alpha=1`; binary L2 logistic `C=1`; unpenalized intercepts. No tuning. |
| Folds | One deterministic five-fold behavioral-block map, shared across targets and regions, before condition/target filtering. |
| Missing/insufficient blocks | Missing-block choices keep behavior/actual values but no held-out prediction. Fewer than five blocks stops preparation explicitly. |
| Display | Actual-value overlay, optional raw/weighted PC rows, original/contribution order fixed across navigation. Start inside **Session timeline and PCA**; reconsider a separate view if crowded. |
| Condition hover | Save/display all canonical condition memberships, preserving overlaps. |
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
  "bin_width_ms": 500
}
```

Absence uses these defaults; an existing malformed file/unknown key is an
error. Bin width must be one of the four supported values. Define internal
constants `RIDGE_ALPHA=1.0` and `LOGISTIC_C=1.0`; reject penalty keys, including
`ridge_alpha` and `logistic_C`, as unknown input fields. There are no JSON,
CLI, GUI, or runner penalty overrides. Still record the constants in effective
run configuration/metadata for provenance. Targets, all-PC policy, five folds,
and window bounds are also fixed. Resolve locators relative to the control
file, contained in the metadata session. No metadata schema change is needed.

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

This is a deliberate compatibility shortcut: the base loader still requires
both regional configuration blocks and validates old decoder settings and
output paths before replacement. A malformed unused setting can therefore
block preparation. Report the base-config prerequisite clearly; do not
silently repair it or broaden this work into a config-loader refactor. Saved
display does not require that config or any of these preparation prerequisites.

Compatibility change required: `validate_augmented_trials()` and
`build_target_table()` currently require every block ID. Add keyword-only
`allow_missing_blocks=False` to both and forward it to their shared block
validator. Default callers retain their current strict behavior. The new
analysis passes True; present block labels must still form contiguous
segments when missing entries are ignored. Do not drop/renumber source rows,
fabricate block IDs, or duplicate the target derivations in a new loader.
Also add keyword-only `allow_empty_numerical_targets=False` to both public
functions. Only this new analysis enables it, alongside missing-block support.
It bypasses only the requirement for one finite numerical target value;
malformed present values and missing columns still fail. Preserve and test
both strict defaults for existing decoder callers.

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
Build all masks from existing `CONDITION_IDENTIFIERS` on the complete table,
then map them to choices. Save them without inferring condition from reward
alone. `correct_rewarded` must equal its column in the saved membership matrix.

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

If a continuous source column has no usable values, keep its coordinate in
T=7, actual values NaN, validity false, models/predictions unavailable, and
reason `target_unavailable_no_values`. Skip its fits and continue other targets.
Expose this target's status in the UI; do not silently remove it from selectors.

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
| `condition_names`, `condition_masks` | Unicode `(C,)`, bool `(N,C)` | All six canonical conditions in existing order; overlapping membership. |
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
Check `C=6`, canonical condition order, choice-row alignment, and equality of
the standalone correct-rewarded mask with its membership column. Record
all-missing target reasons in `meta.target_availability` as well as model rows.

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
the last model for a missing-block choice. Select scores and weights by the
same original PC IDs. For each displayed PC, map its ID to the source-score
axis and the saved coefficient-axis `pc_ids`; never use display position j
directly as the coefficient index:

```text
original_id = displayed_pc_ids[j]
score_index = original_id - 1
weight_index = index of original_id in saved model pc_ids
contribution[w,b,j] = score[w,b,score_index] * beta[target,fold[w],weight_index]
```

In the full canonical model order both indices equal `original_id - 1`.
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

Remap the shared heatmap helper's original-PC `y` values at the prediction
plot boundary to sequential display centers `0..P-1`; use original PC IDs for
tick text and hover. Preserve the shared helper's existing public contract.
An order `[7,2,11]` must appear as three equally spaced rows in that exact
top-to-bottom order. Test the constructed/rendered figure, including y-axis
orientation and tick/hover labels, rather than testing selected arrays alone.
List all true saved condition memberships in hover; do not collapse them to
one mutually exclusive condition or reopen the raw behavioral table.

Reuse the existing chart component/navigation. Target/PC/mode changes reset
vertical ranges only; Jump/start/duration explicitly update horizontal range.
Keep all-session navigation reachable with viewport-limited contribution
construction and chart payloads at fine resolution. The current NPZ loader
still loads full regional score arrays into memory before filtering; this is
not partial or bounded NPZ memory loading. Measure source-load RSS before
considering another storage mechanism. Inspect control crowding before adding a view.

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
  normalizes string sentinels for grouping; empty-target keyword preserves
  strict defaults, permits all-missing continuous columns, and still rejects
  malformed present values; canonical overlapping masks/row alignment;
  rejection of penalty configuration keys; explicit base-config prerequisite
  failures; per-region coverage and all grids.
- **WP2 (Sol):** training masks/row weights and no held-out labels in fits;
  compare ridge coefficients/intercept to an independent weighted normal-
  equation oracle (ridge shrinks, so do not demand exact unpenalized recovery);
  oriented logistic probabilities; constant continuous training labels remain
  valid; all-missing HMM target keeps its unavailable coordinate while the
  other six run; empty/single-class/nonconverged fits; missing destination labels;
  identical state/action fits; unchanged PCA; contribution reconstruction,
  scale-aware ranking/ties, missing-fold guard and global color limits.
- **WP3a (Sol):** all schema/identity invariants and save/load round trip;
  missing/corrupt referenced PCA; nonfinite outputs and probability bounds;
  matching reuse versus scientific changes; relocation; partial publication;
  persisted overlapping condition masks and target availability; no
  invalidation from display-only changes/unrelated metadata edits.
- **WP3b (Luna, Sol review):** synthetic end-to-end prepare/reuse/rerun;
  dry-run writes/fits nothing; absent PCA prepared only on explicit action;
  metadata/probe/bin precedence; no duplicate estimator logic; thin batch
  delegation; source snapshots and logs. Add required stubs only after RED
  tests are committed, not before.
- **WP4a (Sol):** real bin edges/gaps, zeros versus unavailable, actual overlays,
  unequal regional coverage, fold-specific weighted rows and original PC IDs;
  rendered nonmonotonic `[7,2,11]` order has equal row spacing and correct
  tick/hover identities, including a browser check where needed. Use distinct
  coefficients and assert values too: `z7=2, z2=3, z11=5` and
  `beta7=-7, beta2=2, beta11=11` must give displayed contributions
  `[-14,6,55]`, not weights selected by display-row indices;
  saved-target units, fixed order/limits, no fitting during plotting; existing
  plot callers retain default behavior.
- **WP4b (Luna, Sol review):** actual Streamlit saved path with removed raw
  inputs and relocated archive, including availability/app dispatch, not just
  mocked loader tests; explicit Prepare;
  saved target/event/PC changes cause no fitting/writes; persistent pan/Jump;
  full overlapping condition labels without raw sources, unavailable-target
  explanation, and no fit attempted for the empty target;
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
SVD fits, and browser payloads. Viewport filtering reduces browser/contribution
work, not the existing full NPZ score-array allocation. Release temporary matrices between models;
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
