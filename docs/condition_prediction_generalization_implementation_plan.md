# Condition Generalization and Mixed-Outcome Decoding: Implementation Handoff

Status: revision-4 implementation reference updated 2026-11-09; new work is
planned, not implemented. The Full-window seven-target generalization and
two-target mixed baseline, including performance/laterality, is complete
(last recorded integration gate: 237 tests). WP11-WP15 specify Before/After,
actual reward and split-by-outcome scoring. Original experimental acceptance
gates remain recorded; this planning request runs no analyses. Keep this filename.
Start with Section 0.1, the contracts in Sections 1-8, then WP11-WP15 in
Section 15. Section 0.4 preserves chronological evidence; Sections 13-14 are
short completed-baseline summaries, not new work instructions.

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
| Date / branch / inspected HEAD | 2026-11-09 / `refactor` / `04ca54d`. Earlier computational commits remain in the evidence log; recheck checkout state on resumption. |
| Phase / authorization | User authorized spec/plan corrections for the scaffolder default-target gap and explicit Sol orchestration with at most one active worker. Scientific choices are confirmed; WP11-WP15 code work awaits implementation instruction. No experimental processing, batch, transfer or cluster operation in this review. |
| Scientific settings | Three matched choice-aligned windows: Full [-1,+1), Before [-1,0), After [0,+1); prediction stays [-2,+2). Generalization retains seven targets; new mixed products add current_reward as target three. Both splits/outcomes and both lateralities are summarized. |
| Source/tests | Baseline implementation complete; last recorded gate 237 passed in 17.73 s. New tests/source not started; no pytest gate rerun during planning. |
| Documentation | Spec revision 4 is confirmed. This plan now specifies localized changes, v1/v2/v3 compatibility, test-first WP11-WP15 and the next handoff gate. Documentation edits are uncommitted. |
| Existing user changes | Unstaged tracked: `src/main.py`, `src/neural_analysis/sync_ephys.py`; revision-4 spec edits are already present. Preserve these and unrelated untracked files. No staged file was reported at inspection. |
| Team / writer | Current documentation chat is the sole writer; no worker launched. Implementation assignments: Sol orchestrator; at most one active worker, with bounded Luna integration/usage scope in Section 12. Runtime model/effort identity is not exposed to the agent. |
| Running analysis | No process/worker launched by this planning revision. Prior preview liveness was not checked; do not infer it from the old log. |
| Latest verification | Review identified the additional single_session_setup default-target consumer; WP11 and integration gates now include its regression. Document checks and evidence below cover the correction and explicit assignments. Historical baseline remains 237 tests/browser checks; no new code gate was run. |
| Next action | Receiving implementation chat reads the entry points above. On implementation instruction, write WP11 tests, verify RED and commit tests before source. Preserve completed work and old products. |

### 0.2 Package ledger

Fill evidence as work proceeds. Worker completion is not Sol acceptance;
experimental visual acceptance belongs to the user.

| Package | Writer | State | RED commit / command / result | GREEN commit / checks / review | Next gate |
| --- | --- | --- | --- | --- | --- |
| WP0 documentation | Lead | Complete | Not applicable | Original baseline `9e46967`; completed record `d2bdd43`; extension implementation evidence below | User review. |
| WP1 settings/alignment/folds | Lead | Accepted | `fa67c05`; 2 expected collection errors | `0a8c65f`; 64 passed with legacy targets | WP2. |
| WP2 models/contributions | Lead | Accepted | `d8561d4`; 2 expected import errors | `3f70f6b`; 74 passed; Ruff clean | WP3a. |
| WP3a persistence/identity | Lead | Accepted | `17d94ef`; 1 expected import error | `6e4afe5`; 86 integrated tests passed | WP3b. |
| WP3b runner/CLI/batch | Lead | Accepted | `62a5ae2`; 1 expected import error | `6e3fc86`; 98 integrated tests passed | WP4a. |
| WP4a plotting/geometry | Lead | Accepted | `91da846`; expected import error | `435fd69`; follow-up hover test `0f77f29`; all 172 integrated tests pass | Browser checks completed. |
| WP4b webapp integration | Lead | Accepted | `1aa86bd`, `0f77f29`, `87ef8b3`; final control test `1cce9ad` | `2c39850`, `f7cd9c8`; 173 tests pass; saved-only browser display/pan checked | User inspection. |
| WP5a usage documentation | Lead | Complete | Not applicable | `2c39850`: timeline guide + neural/task-decoding README updates | User usage/review. |
| WP5b CT026 inspection | Lead + user | Numerical/browser validation complete; user acceptance pending | Not applicable | 70 fits; 715 windows/region; exact reconstruction; performance below | User inspects saved preview/summary. |
| WP6 mode/config/alignment extension | Lead | GREEN | `2922a09`; 11 failed / 23 passed as expected | `5a555c1`; 76 focused tests; 66 prediction tests; changed-file Ruff clean | Complete. |
| WP7 models/persistence extension | Lead | GREEN | `8a599c1`; 6 failed / 23 setup errors / 10 passed as expected | `1c81c84`; 39 scientific/persistence tests; 125 with package/target regressions | Complete. |
| WP8 runner/inspection extension | Lead | GREEN | `4c6ee75` runner; `1a54eb0` UI/chart; `9732b6d` counts; `01b0cc1` actual chart position | Runner `1a466c9`; UI `5b72147`; 216 integration tests and real browser checks | Complete. |
| WP9 extension docs/validation | Lead + user | Docs/synthetic/browser complete; experimental acceptance pending | Required code RED gates complete | Updated usage guides; 216 passed; screenshots below | Separately requested CT026 mixed preparation and user inspection. |
| WP10 performance/laterality | Lead | GREEN | `c381dda` scoring collection error; `beb4770` three UI failures; `05d0011` continuous caption failure | `86b7131` scoring; `c315ef4` UI; 237 integrated tests, Ruff and synthetic browser checks | Complete; no fits/archive migration required. |
| WP11 windows/reward inputs | Sol orchestrator, direct writer | Planned | Not started | Not started | Fixed-window controls, shared target registration, preserved parser/scaffolder defaults, aligned preparation; Section 15. |
| WP12 versioned persistence | Sol orchestrator, direct writer | Planned | Not started | Not started | v3 publication and read-only v1/v2 compatibility. |
| WP13 split/outcome scoring | Sol orchestrator, direct writer | Planned | Not started | Not started | Matched-window scores, exact split/outcome/side masks and reward counts. |
| WP14 runner/view integration | Sol orchestrator; optional single Luna worker | Planned | Not started | Not started | One window per preparation, saved selection, labels and anchored navigation; Sol reviews/accepts. |
| WP15 docs/integration/acceptance | Sol orchestrator + user; optional same Luna worker for usage docs | Planned | Not started | Not started | Sol owns synthetic/browser checks; user inspects explicitly authorized experimental validation. |

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

### 2026-11-03 - WP0 / mixed-outcome scientific and implementation reference

- Authorization: user requested updating these existing spec/plan documents;
  no source/test implementation or new experimental preparation in this turn.
  Both additional classifiers (state and action) were explicitly confirmed.
- Baseline inspected: `refactor` at `d2bdd43`; local `origin/refactor` agrees.
  Existing staged timeline spec and unstaged main/sync edits are preserved.
- Earlier read-only inspection of the saved CT026 2026-08-04 result found
  identical state/action training masks, coefficients and predictions in both
  regions. Correct-rewarded: 289 trials, 0 state/action disagreements;
  correct omission: 68, 0 disagreements; incorrect: 358, all disagree.
  These observations motivate the additional mode; no new fits were used.
- Updated the scientific contract and Sections 1-8 in place, with exact mode
  names, target sets, canonical membership, unchanged weighting/windows/folds,
  new-schema identity and read-only v1 loading. Added WP6-WP9 with file/API
  contracts, RED-before-code tests, ownership, UI labeling and acceptance gates.
  Kept original implementation/validation evidence and output paths intact.
- Verification: `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync --offline
  python -` document assertions passed for ASCII, ten local links, unique
  section numbering, balanced fences/whitespace, both mode/label contracts,
  version compatibility, all four extension packages and original evidence.
  Initial assertions identified an unnamed selected-mask field in the spec;
  it now explicitly names `training_condition_mask`. Changed-document
  `git diff --check` passed. No pytest suite, worker, computation, staging,
  commit or push was performed for this documentation revision.
- Next: user review; on implementation instruction start WP6 tests and its
  focused RED command in Section 13. Do not restart completed WP1-WP5.

### 2026-11-03 - WP0 / implementation-review integration corrections

- Checked feedback against the actual timeline renderer, navigation controls,
  persistent chart Python/JS, and numerical model source. The controls' `None`
  return currently triggers raw loading; navigation keys/layout revision
  currently include PCA configuration. Both require the localized WP8 changes
  now specified in Section 7 and its RED tests.
- Corrected the prior factual error: public `predict_region_windows()` already
  exists in `modeling.py` and is called by the fitter. Preserve its signature
  and prediction/reason contracts; do not add or duplicate that API.
- Added the explicit saved-anchor return state, including prepared-but-hidden
  and unprepared-mode switches without raw sources. Split session/time-origin
  navigation from saved-basis/model controls and vertical revisions; require
  cross-configuration 500-ms/100-ms tests and actual browser-range verification.
- Both documents now disclose that mixed training changes sample size and
  effective regularization relative to average loss at fixed C. The existing
  summed objective, equal trial weights, penalties and two-target scope remain.
- Verification: the same offline `uv run ... python -` document assertions
  passed for ASCII, ten local links, sections/fences/whitespace, mode/schema
  contracts and historical evidence; additional assertions checked all four
  feedback corrections and removed the inaccurate API claim. Changed-document
  `git diff --check` passed. No source/tests, pytest, fitting, browser session,
  worker, staging, commit or push in this documentation-only follow-up.
- Next action remains WP6 RED on implementation instruction; do not restart
  original work or require a new framework to address these integration gaps.

### 2026-11-03 - WP0 / approved bounded-loading navigation exception

- User approved the review's bounded-loading exception and requested preparing
  the plan for implementation. Updated only this plan and its scientific spec.
- Preserve browser pan within destination loaded bounds; otherwise reset to
  the server control-selected interval and explain the behavior in the caption.
  WP8 now includes the local chart guard, synthetic 500-ms/20-ms switch tests,
  and a browser check. No pan callback or new loading mechanism is required.
- Verification: offline/frozen/no-sync `uv run python -` assertions passed for
  ASCII, balanced fences, local links, and the bounded-navigation contract;
  changed-document `git diff --check` passed. No source/test edits, pytest,
  experimental preparation, staging, commit, or push in this documentation turn.
- Reference ready; next action on implementation instruction is WP6 RED,
  followed by WP7-WP9. Experimental preparation remains separately scoped.

### 2026-11-03 - WP6-WP9 / mixed-outcome implementation and synthetic acceptance

- Authorization: user instructed implementation of the reviewed plan. Sole
  writer; no delegated worker or new experimental session run. Original
  correct-rewarded results and unrelated staged/unstaged work were preserved.
- WP6: tests committed `2922a09` after 11 expected failures/23 passes in
  config/preparation. GREEN `5a555c1`: fixed two-mode controls, selected targets
  before shared validation, factual correct-rewarded plus separate training
  union, identical full-table folds. Focused targets/config/preparation 76
  passed; prediction package 66 passed; Ruff clean.
- WP7: RED `8a599c1` (6 failed, 23 setup errors, 10 passed) required explicit
  membership, seeded independent state/action signals (seed 318), weighted
  logistic gradient, additive contributions, and versioned products. GREEN
  `1c81c84`: optional keyword-only fitter mask, v2/schema-2 publication, strict
  mode/target/condition/fit-mask validation, read-only exact v1 compatibility.
  Scientific/persistence tests 39 passed; package plus old targets 125 passed.
  The v1 fixture constructs original array names without new membership/mode
  fields and verifies original identity, predictions and unchanged archive bytes.
- WP8 runner RED `4c6ee75`: 5 failed/4 passed. GREEN `1a466c9`: mode overrides
  through session/batch/CLI, actual target-dependent counts/bytes, matching-PCA
  reuse, separate immutable products and factual mode summaries/logs. All nine
  runner tests passed, including 500/100-ms mode preparation and reuse.
- UI/plot/chart RED `1a54eb0`: five plot failures and five UI/JS failures.
  Additional RED `9732b6d` corrected binary-only class-count requirements and
  approved output names. Real browser inspection then exposed a mount-position
  reset that stable keys/revision tests alone missed: mixed captions shifted
  the chart from tree position 7 to 8. RED `01b0cc1` captured this failure.
  GREEN `5b72147` groups variable controls/captions in one fixed container,
  retains one saved anchor path, scopes horizontal navigation to session/origin,
  labels modes/training clearly and bounds the browser range transfer locally.
- Final Section 9 integration command used
  `MPLCONFIGDIR=.pytest_cache/matplotlib`, `UV_CACHE_DIR=.pytest_cache/uv`,
  `uv run --frozen --no-sync --offline pytest -q` and
  `NODE_BINARY=/home/matt/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node`:
  216 passed in 14.43 s. Changed Python Ruff and whitespace checks passed.
  No new dependency, external-library edit, or prediction-storage framework.
- Browser: loopback Streamlit port 8518, synthetic saved-only fixture copies
  in ignored `.pytest_cache/mixed_preview_two_modes` and `mixed_preview_bounded`.
  Both regions' predictions and weighted PC rows had readable identities,
  log-odds units and true gaps. Actual panned 40-90 s ticks survived a 500-ms
  generalization -> 100-ms mixed switch and addition of weighted rows. At
  20 ms, an in-bounds panned range survived; a 90-140 s range outside loaded
  -1..121 s reset to the control-selected 1..61 s. No browser pan callbacks.
  Proof: `.pytest_cache/mixed_prediction_browser.png` (ignored synthetic image).
- WP9 usage docs now cover both modes, controls/CLI overrides, v1/v2 copy-back,
  scientific limits, anchored unprepared/hidden views and bounded navigation.
  ASCII, fences, local-link and changed-path whitespace assertions passed.
  Temporary synthetic preview stopped after inspection; original 8517 preview
  untouched. No experimental fit, batch, transfer, push or acceptance claim.
- Next: user review; when separately requested, prepare only the designated
  500-ms CT026 mixed product and obtain visual/scientific acceptance. Do not
  repeat completed implementation or infer authorization for broader processing.

### 2026-11-03 - WP10 / trial-mean training/test performance and laterality

- User requested accuracy and left/right comparisons for both modes, then
  selected trial means over `[-1,+1)` and both actual choice and recorded
  state/context splits. Continued the proposed small scoring/UI implementation;
  no experimental preparation or additional scientific model was requested.
- RED `c381dda`: new scoring module absent (one expected collection error).
  GREEN `86b7131`: 18 scoring tests passed; prediction package plus existing
  target-loader regressions 152 passed. Oracles cover probability-before-class
  averaging, independent fold reconstruction, unique held-out counts, exact
  outcome/side masks, negative/undefined R2, native RMSE, and unavailable groups.
- Training reconstruction uses the exact saved training mask and coefficients,
  processing one bin's window/PC slice at a time. It applies sigmoid before
  averaging for binary models. Training sets overlap and remain per-fold;
  pooled held-out metrics count each eligible trial once per region. No fits,
  source-array changes, NPZ writes or schema/fit-identity changes.
- UI RED `beb4770`: three expected missing-panel/cache failures. Caption RED
  `05d0011` required continuous targets to avoid binary threshold/class advice.
  GREEN `c315ef4`: saved-only tests pass for both modes, correct units/labels,
  outcome selection, small-summary caching and fixed chart position/revision.
  Changing populations/PC displays reuses scores; changing target computes the
  new target once. Synthetic archived files remain byte-unchanged, with no raw
  inputs or preparation calls during display.
- Full Section 9 integration gate with offline/frozen/no-sync uv, bundled
  NODE_BINARY and MPLCONFIGDIR=.pytest_cache/matplotlib: 237 passed in 17.73 s.
  Changed Python Ruff and whitespace checks passed. No dependency added.
- Browser used loopback port 8518 and ignored saved-only synthetic fixture
  copies in `.pytest_cache/performance_preview_*`. Verified readable binary
  outcome scores, per-fold training/held-out metrics, both left/right grouping
  tables/counts and percentage-point differences. Continuous generalization
  displayed negative R2 and native-unit RMSE. Screenshot:
  `.pytest_cache/prediction_laterality_browser.png` (synthetic values only).
  Temporary server/tab stopped after inspection; original port 8517 untouched.
- Spec/plan and three existing usage guides updated; document checks verify
  ASCII, local links, fenced blocks and scoring contracts. Unrelated user work
  remains staged/unstaged as before. No new experimental processing or push.

### 2026-11-09 - WP0 / revision-4 modular expansion plan

- Authorization: user requested updating this implementation reference after
  confirming the revision-4 spec; no implementation or experimental run in
  this turn. Re-read SoftwareDesign.md, current preparation/fitting/persistence/
  scoring/UI code and shared target construction. Baseline is `refactor` at
  `04ca54d`; preserve existing spec edits and main/sync user changes.
- Design: reuse fixed session PCA and numerical fit/prediction functions.
  Add three fixed supervised-window masks, one actual-reward target and
  split/outcome/side records in the existing modules. No new package,
  dependencies, window-specific estimators, storage engine or scheduler.
- Practical source findings addressed: shared target identifiers currently
  double as the old omitted-target defaults, so adding reward needs an
  explicit preserved default set. Saved validation currently resolves mixed
  targets only through today's helper; v3 must coexist with legacy T=2 Full
  products via a small version-local contract. Scoring already accepts a saved
  mask but returns held-out-only outcome/side rows; expand those records and
  reuse per-fold reconstructed means rather than adding another scorer.
- Updated active contracts, v1/v2/v3 matrix, four-field controls, record/API
  scope, ownership and WP11-WP15 RED-first gates. Retained original evidence
  exactly; completed baseline references are marked historical. Synced the
  spec's implementation-plan pointer without changing its confirmed decisions.
- Verification: offline/frozen/no-sync `uv run ... python -` document checks
  passed for ASCII, ten local links, section numbering/fences/whitespace,
  controls JSON, source/test paths, version contracts, all five new packages
  and unchanged original evidence. Changed-document `git diff --check` passed.
  No pytest, source/test edit, worker, staging, commit, push or analysis run.
- Next: user reviews this plan; on implementation instruction write WP11
  tests, run its focused RED command, commit tests, then implement. Continue
  through the approved scope without repeated per-package approval.

### 2026-11-09 - WP0 / handoff-readiness audit

- Reviewed spec and active plan against current preparation/model, shared target
  registry, result validation, scoring and viewer code. No scientific choice
  remains unresolved; the expansion stays within the existing module boundaries.
- Clarified training-score availability per examined model, independent of a
  trial's own held-out-model failure; added the corresponding WP13 case.
  Preserved ordinary-accuracy percentage-point gap presentation, per-side
  balanced accuracy and valid training-versus-pooled fold selection.
- Included fitter/contribution tests in WP11's actual first gate. Consolidated
  superseded package/agent instructions into short baseline summaries; all
  prior chronological evidence is byte-for-byte retained. Plan was reduced
  from 1841 to 1560 lines before this audit entry; added an explicit read order.
- Documentation assertions passed for ASCII, ten links, all section/fence/
  whitespace checks, source/test paths, fixed controls, bin-count grids,
  v1/v2/v3 compatibility and revised scoring gates. No pytest or real data run.
  Changed-document whitespace checks passed; no source/test edits, worker,
  staging, commit or push. Next gate is WP11 RED on implementation instruction.

### 2026-11-09 - WP0 / scaffolder-default correction and assignment clarification

- User authorized documentation corrections only; implementation remains a
  later step. Review found that `single_session_setup._task_payload()` also
  uses `TARGET_IDENTIFIERS`, so changing only the parser default would silently
  add reward to newly generated old-decoder configs.
- Added the spec invariant, WP11 source/test scope and regression gates:
  parser and scaffolder both use preserved `DEFAULT_TARGET_IDENTIFIERS`.
  Existing explicit selections and existing config bytes remain unchanged.
- Clarified Sol as orchestrator/direct scientific writer for WP11-WP13 and
  acceptance owner for all packages. At most one active worker; the optional
  Luna worker handles bounded WP14 integration or WP15 usage docs sequentially.
- Verification: documentation checks cover ASCII, local links, fences,
  scaffolder source/test paths, both test commands and assignment consistency;
  changed-document whitespace checks passed. No source/test edits, pytest,
  worker, experimental run, staging or commit. No further blocker found.
- Next: stop after documentation review; WP11 RED awaits implementation instruction.

## 1. Outcome and boundaries

Retain individual choice-window predictions beneath the existing session
events, using saved session-wide regional PCs. Expose two fixed analysis modes:
seven-target correct-rewarded generalization and three-target mixed-outcome
state/action/actual-reward decoding. Each has Full, Before and After variants.
Fit once per selected window/target/region/held-out block group,
predict across valid choice conditions, and overlay actual trial values.
PC activity/contribution heatmaps remain optional. Reuse the existing
numerical fitter/predictor with a selected bin mask and target columns;
no new estimator, analysis package, window-specific subclass or PCA method.

Use functions, narrow array contracts, and data-only records. Separate input
loading, fitting, persistence, plotting, and UI. Reuse existing libraries and
timeline navigation. Do not introduce estimator/track frameworks, runtime
checkpoint services, new dependencies, or cluster resource changes.

The existing decoder's elastic-net models, pooled `full` window, and saved
schemas retain their meanings. Extend the existing modules in
`src/neural_analysis/task_decoding/session_prediction/`. One necessary exception
in the completed original work was the backward-compatible missing-block and
empty-target support in existing `targets.py` (Section 4). This extension adds
one shared current-reward definition/encoding, while preserving old default
target sets and validation rules. Preserve existing result readability. That
file is in the old decoder's source fingerprint: old in-progress runs still
need their original scientific checkout for strict resume; do not claim that
changing it leaves old execution identities unchanged.

## 2. Confirmed scientific settings

| Item | Contract |
| --- | --- |
| Generalization mode | ID `correct_rewarded_generalization`; label **Correct-rewarded generalization**; canonical training mask `correct_rewarded`. Default for old controls/calls/results. |
| Generalization targets, canonical order | `current_state`, `current_action`, `qlearning_relative_value`, `forgetting_q_relative_value`, `hmm_signed_belief`, `hmm_decay_signed_belief`, `relative_doubt`. Exact source columns/orientations are in the spec. |
| Mixed mode | ID `mixed_outcome_decoding`; label **Mixed-outcome decoding**; training membership is canonical `correct_rewarded OR omission OR incorrect`. `omission` means correct omission. |
| Mixed targets, new products | `current_state`, `current_action`, `current_reward`; class 1 is left, left, rewarded respectively. Legacy v2 mixed products retain their original two targets. |
| Supervised window | `full`: [-1,+1), default; `before`: [-1,0); `after`: [0,+1), all choice-aligned. Each is a separate fit and matched scoring interval. |
| Actual reward | Trial-i reward flag, 0=unrewarded/1=rewarded, manual/no-choice exclusions unchanged. No previous-trial shift, correctness substitution, model expectation or third analysis mode. |
| All-missing continuous target | Retain that target as unavailable and continue the others; missing columns or malformed present values are input errors. |
| Regions / features | Separate PFC and HPC models; all saved session PCs, unwhitened, with no further PC or target scaling. No combined model. |
| Bins | Default 500 ms; 100 ms selectable. Retain existing 50/20 ms support. Each resolution uses its matching session-PCA archive. |
| Training / prediction | Pool the selected Full/Before/After bins; predict each complete [-2,+2) choice window and retain gaps. Time zero belongs to After. No cross-window scoring matrix. |
| Sampling / weighting | All eligible training trials, each total weight one. No additional train fraction, subsampling, class/outcome balancing, or condition-count multiplication. |
| Models | Continuous ridge `alpha=1`; binary L2 logistic `C=1`; unpenalized intercepts. No tuning. |
| Folds | One deterministic five-fold behavioral-block map, shared across windows, targets and regions, before condition/target filtering. |
| Summaries | Both training and held-out, by outcome and both choice/state lateralities; exact training masks, per-fold training, unique pooled held-out observations. |
| Missing/insufficient blocks | Missing-block choices keep behavior/actual values but no held-out prediction. Fewer than five blocks stops preparation explicitly. |
| Display | Existing **Analysis** plus **Training/scoring window**; mode/window-filtered runs and actual saved targets. Explicit target/window/population labels. Existing timeline and PC views remain. |
| Condition hover | Save/display all canonical condition memberships, preserving overlaps. |
| Validation | CT026 2026-08-04, initially at 500 ms, after implementation/synthetic gates. |

For each group, train on the selected mode's eligible trials from the other four and
predict all valid windows in that held-out group. Groups may contain blocks
from different session times; training can include later blocks. Each choice
uses exactly its held-out model, not an ensemble containing models trained
on that choice. The full-table block map is identical across modes when the
input table is unchanged. Generalization has at most `7 * 2 * 5 = 70` fits;
expanded mixed decoding has `3 * 2 * 5 = 30`, per selected window. Each run
contains one mode/window; Prepare does not automatically fit all three windows
or the other mode. All variants reuse the matching saved PCA and trial folds.

Interpretation to retain in captions/summary:

- PCA/scaling use the whole session, including held-out neural observations.
  Only supervised fitting is block-held-out; this is not a fully held-out
  preprocessing pipeline or a future-only forecast.
- State/action labels normally coincide on correct-rewarded trials. Identical
  eligible training rows/settings should yield identical models, even when
  actual destination labels differ. This is a correct-rewarded reference
  projection, not independently identified subjective state or belief.
- Mixed state decoding targets recorded task state; mixed action decoding
  targets choice. Incorrect trials allow their labels to differ. This need
  not produce different outputs on every dataset or identify causal coding.
- Before reward decoding predicts eventual recorded reward, not subjective
  expectation. After is post-choice, not necessarily post-delivery. Reward
  overlays are trial labels, not delivery-event timing.
- With a full orthonormal basis, isotropic L2 on all unwhitened PCs is L2 on
  the same standardized unit activity in rotated coordinates, not dimensional
  reduction. PC signs are arbitrary; signed weight-times-score contributions
  are invariant to consistently reversing a component's sign.

## 3. Modules, interfaces, and ownership

Existing package files and tests reside under `task_decoding/session_prediction/`
in the source and `src/tests/neural_analysis/` trees respectively. The table
records the pending revision-4 ownership. Completed baseline references remain
in Sections 13-14; Section 15 gives the new test-first gates.

| Owner / package | Source responsibility | New tests |
| --- | --- | --- |
| Sol orchestrator, direct writer / WP11 | Existing prediction `config.py`, `preparation.py`; localized shared `task_decoding/config.py` and `targets.py` registration/encoding; `single_session_setup.py` default-target selection. | `test_config.py`, `test_preparation.py`, shared target/config regressions, `test_single_session_setup.py`, bounded fitter/contribution regressions |
| Sol orchestrator, direct writer / WP12 | `results.py`: versioned metadata, exact mask validation, identity, discovery/load/save. | `test_results.py` |
| Sol orchestrator, direct writer / WP13 | `scoring.py`: one mean per trial on the saved mask; split/outcome/side records and class counts. | `test_scoring.py` |
| Sol orchestrator; optional single Luna worker / WP14 | Existing runners, `plotting.py`, `webapp/session_prediction_view.py` and narrow timeline/availability integration if required; Sol reviews/accepts. | `test_scripts.py`, `test_plotting.py`, `test_session_prediction_webapp.py` and directly affected timeline/chart tests |
| Sol orchestrator; optional same Luna worker for usage docs / WP15 | Existing usage guides; Sol owns integration/browser inspection and live handoff; user inspects any authorized real result. | Existing integration command and numerical reconstruction checks |

Implemented core interfaces are `prepare_prediction_inputs(...)`,
`fit_region_prediction_models(...)`, and public `predict_region_windows(...)`,
plus save/load/discovery and runner functions. The fitter already calls the
prediction helper, which returns predictions `(W,B,T)` and reasons `(W,T)`
from scores, folds and fitted parameters without fitting. Preserve that
public helper and its existing signature; no new prediction API is needed.
Section 15 gives backward-compatible extensions. Record inputs, shapes,
units, and returns in docstrings. Numerical functions accept prepared arrays,
not session objects or UI state.

Keep new plotting helpers independent of `timeline_plotting.py` to prevent
circular imports. A small data-only prediction-display record may group the
saved result, selected target, and heatmap mode. Extend existing plot APIs
with optional defaults; callers without predictions retain current behavior.

## 4. Configuration, inputs, and row alignment

Default optional control file: `session_prediction_config.json` beside metadata.
Its final allowed fields/defaults for the extension are:

```json
{
  "base_config_path": "task_decoding_config.json",
  "bin_width_ms": 500,
  "analysis_mode": "correct_rewarded_generalization",
  "training_window": "full"
}
```

Absence uses these defaults; an existing malformed file/unknown key is an
error. An omitted `analysis_mode` retains correct-rewarded generalization;
the only other accepted value is `mixed_outcome_decoding`. Reject unknown
modes, arbitrary condition lists, and target-set overrides. Bin width must
be one of the four supported values. Retain internal
constants `RIDGE_ALPHA=1.0` and `LOGISTIC_C=1.0`; reject penalty keys, including
`ridge_alpha` and `logistic_C`, as unknown input fields. There are no JSON,
CLI, GUI, or runner penalty overrides. Still record the constants in effective
run configuration/metadata for provenance. `training_window` accepts only
`full`, `before`, `after`; missing controls retain Full. No arbitrary bounds
or separate scoring-window knob. All-PC policy and five folds remain fixed.
Resolve locators relative to the control
file, contained in the metadata session. No metadata schema change is needed.

Precedence: fixed scientific contract -> default/control-file settings ->
explicit runner/CLI/GUI mode/window/bin/probe selection. Overrides apply only on
Prepare and are saved as effective configuration. A saved result's settings drive
its display; current widgets must not relabel it as another resolution/basis.
The metadata-only CLI uses version-3 regional role defaults. With version-2
metadata lacking roles, explain that explicit existing GUI probe selection
or metadata role setup is needed; do not guess probe/anatomical mappings.

During preparation, load the base `TaskDecodingConfig`, check that it belongs
to the selected session, and make a local `dataclasses.replace` selecting the
chosen mode's ordered target set and `choice_time`. Call public
`build_target_table()` and `build_condition_masks()`. Use its augmented CSV/feature-parameter inputs,
not its estimator, PC counts, condition subset, bin width, or bilateral
coverage requirement. Missing prerequisites produce an actionable error;
preparation does not fit behavioral models or run the old neural decoder.

This is a deliberate compatibility shortcut: the base loader still requires
both regional configuration blocks and validates old decoder settings and
output paths before replacement. A malformed unused setting can therefore
block preparation. Report the base-config prerequisite clearly; do not
silently repair it or broaden this work into a config-loader refactor. Saved
display does not require that config or any of these preparation prerequisites.

Register `current_reward` once in the existing shared target definitions:
categorical, source `reward`, derivation `source`, positive label `rewarded`,
class labels {0: unrewarded, 1: rewarded}. Reuse binary validation/encoding
alongside `current_action_is_correct`; do not duplicate chronological loading.
The shared parser currently uses `TARGET_IDENTIFIERS` as its omitted-target
default. Preserve that old tuple as `DEFAULT_TARGET_IDENTIFIERS`, append reward
to the recognized identifiers, and retain the old default in `_parse_target_names()`.
`single_session_setup._task_payload()` also writes `list(TARGET_IDENTIFIERS)`
into new task-decoding configs. Switch its import/payload to
`DEFAULT_TARGET_IDENTIFIERS` too, and test the generated config's exact original
target order. Preserve existing files through the setup command's current
exclusive-write behavior; do not add a migration or new setup option.
Old default decoding runs must not silently acquire a reward target.
Existing explicit target selections and class meanings remain unchanged.
Adding shared scientific source changes can change strict execution fingerprints;
legacy saved display still uses its recorded products, not current source hashes.

Original compatibility change, already implemented: `validate_augmented_trials()`
and `build_target_table()` have keyword-only `allow_missing_blocks=False`,
forwarded to their shared block validator. Default callers retain their current
strict behavior. Both prediction modes pass True; present block labels must still form contiguous
segments when missing entries are ignored. Do not drop/renumber source rows,
fabricate block IDs, or duplicate the target derivations in a new loader.
The implemented keyword-only `allow_empty_numerical_targets=False` also
remains on both functions. Session prediction enables it alongside
missing-block support.
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
Add a separate bool `(N,)` `training_condition_mask`: that column for
generalization, or the union of the three primary outcomes for mixed decoding.
Keep `correct_rewarded` factual in both modes. No row is duplicated because
it is also switch/stay. New mixed preparation constructs only its three target columns;
it must not acquire continuous-target-column requirements from the original
seven-target preparation (the full base-config prerequisite above remains).

Keep the window mapping in prediction `config.py` as three fixed numeric
pairs. Extend `training_bin_mask(edges, *, training_window="full")` to select
complete bins inside that pair, checking exact boundaries with absolute
tolerance and no interpolation. Append `training_window="full"` to the
data-only `PredictionInputs` record, preserving existing construction calls.
Add the same keyword default to `prepare_prediction_inputs()`; extend controls
loading with an optional `training_window=None` override. Models receive the
resulting bool `(B,)` mask, not a mode/window control flag. The existing fitter,
`predict_region_windows()` and `build_pc_contributions()` need no new numerical API.
Append the same Full default to `SessionPredictionConfig`. Update any existing
binary docstrings that assume class 1 always means left; the 0/1 numerical
interface remains, while interpretation comes from target metadata.
New preparation uses seven or three targets; legacy loading supplies its
actual saved two-target tuple explicitly under Section 6's version contract.

## 5. Fitting and contribution contracts

Use N shared timeline choices, T=7 generalization or T=3 new mixed targets
(legacy mixed T=2 only on the saved read path),
F=5 folds, B prediction bins, and K_r all saved PCs in region r.
Regional scores are float64 `(W_r,B,K_r)`;
`choice_positions` is int64 `(W_r,)` into N. Targets/validity have shape `(N,T)`,
baseline/correct-rewarded/training-condition/fold IDs `(N,)`, and edges
`(B+1,)` seconds. Fold `-1` means no held-out assignment. Scores retain source
units and are never mutated.

Select bins fully inside Full [-1,+1), Before [-1,0), or After [0,+1), with
zero in After only. Full/Before/After/prediction bin counts are 500 ms
`4/2/2/8`, 100 ms `20/10/10/40`, 50 ms `40/20/20/80`, 20 ms `100/50/50/200`.
For a target/region/fold, intersect regional coverage, baseline,
selected training-condition membership, valid target, and nonnegative fold
different from the held-out fold. Reuse `pooled.decoding.stack_training_window()`:
trial-major rows, repeated trial-i target, weights `1 / training_bin_count`.
At 500 ms these are 1/4 for Full, 1/2 for either half: every trial still has
weight one. Complete [-2,+2) neural eligibility, labels and folds are shared,
so merely changing the supervised window does not alter trial membership.
Do not change these weights to compensate for unequal class/outcome counts.
An absent outcome in a training fold is not itself a fit failure or permission
to import held-out rows. Retain the existing both-binary-classes requirement.

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

For logistic models, let `S = sum_n w_n = training_trial_count`. Dividing
the objective by S gives weighted average loss plus `||beta||^2/(2*C*S)`.
Mixed-outcome fits normally contain more trials, so fixed C gives weaker
regularization relative to average loss. Mode differences reflect outcome
composition, sample size, and that relative penalty scale; they are not a
composition-only comparison. Keep C=1 and the confirmed weights. Do not add
C rescaling, matched subsampling or balancing; report factual training counts
and this limitation in interpretation/captions/summary.

Each trial has total weight one regardless of bin count. Continuous targets
stay in native units. A nonempty finite continuous training set is sufficient
for a ridge fit; constant training targets are valid and can yield a constant
intercept-only prediction. Do not import target-variance or two-test-trial
requirements from R2 scoring. Binary training must contain both classes.
Save independent training-trial/stacked-row counts so weak sampling is visible.

If a continuous source column has no usable values, keep its coordinate in
the generalization target axis, actual values NaN, validity false, unavailable
models/predictions, and reason `target_unavailable_no_values`. Skip its fits
and continue other targets.
Expose this target's status in the UI; do not silently remove it from selectors.

Treat empty training, single-class binary training, nonconvergence, numerical
fit failure, or nonfinite parameters as model-unavailable with a readable
reason. Continue other fits. Invalid input structure raises before fitting;
do not catch all errors indiscriminately. Check finite outputs and `[0,1]`
binary probabilities. Unexpected nonfinite predictions invalidate that
model's destination windows, with a recorded reason rather than a fallback.

Predict all B bins of each assigned held-out window even when its actual
value is missing or constant. Binary output is class-1 probability (left for
state/action, rewarded for reward);
continuous output is `b + sum_j beta_j*z_j`. Logistic contributions add to
logit, then `scipy.special.expit` gives probability. Do not clip ridge outputs.

For each fitted target/region/fold, compute `s_j = abs(beta_j) * std(z_j)` on
its pooled supervised training rows (`ddof=0`). Save the equal mean across
valid folds, contributing fold count, and descending PC order with original
rank breaking ties. This measures varying additive contributions, not unique
importance or an additive share of output variance. Mean offsets remain in
the original affine prediction; the ranking describes variation only.

## 6. Saved arrays, identity, and publication

New writes use analysis version `condition-session-prediction-v3`; NPZ schema 3.
Continue reading the exact v1/schema-1 and v2/schema-2 pairs as below. Use named
arrays and dictionary-valued `meta`, following trusted-project timeline IO.
Do not pickle estimator instances. Choices retain the source structured table.

| Shared name | Dtype / shape | Meaning |
| --- | --- | --- |
| `choices` | Structured, N rows | Original choice identity, action, UTC/elapsed time, reward when present. |
| `target_names`, `target_families` | Unicode `(T,)` | Exact ordered target set/families for the saved mode. |
| `target_values`, `target_valid` | float64/bool `(N,T)` | Native actual values/validity. |
| `baseline_valid`, `correct_rewarded` | bool `(N,)` | Independent input masks. |
| `training_condition_mask` | bool `(N,)` | Selected mode's canonical membership before baseline/target/region/fold filtering; schema 2. |
| `condition_names`, `condition_masks` | Unicode `(C,)`, bool `(N,C)` | All six canonical conditions in existing order; overlapping membership. |
| `block_values_json`, `fold_ids` | Unicode/int64 `(N,)` | Canonical block identity; fold `-1` for missing. |
| `bin_edges_s`, `training_bin_mask` | float64 `(B+1,)`, bool `(B,)` | Exact prediction grid and selected fit/scoring subset. No separate scoring mask is needed. |
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
Validate T against the exact version/mode contract below, not against an
arbitrary target list or only today's `MIXED_TARGET_NAMES`. Reconstruct
`training_condition_mask` from canonical memberships,
then validate every regional `(target,fold,choice)` training mask against it.
Also compare the saved bin mask to `training_bin_mask(bin_edges_s,
training_window=...)`; bin edges, recorded bin width, selected bounds and
prediction extent must agree. Do not trust a valid-shaped but wrong window mask.

Schema-3 `meta.effective_config` must include `analysis_mode`, its exact
`target_names`, and `training_condition_names`: `["correct_rewarded"]` or
`["correct_rewarded", "omission", "incorrect"]` in this order. Include these
scientific choices in the identity, alongside the existing source/library
fingerprints. Presentation strings are derived from the mode, not editable
scientific settings. Counts/array-size estimates must use actual T.
Require `training_window` (`full`/`before`/`after`), numeric `training_window_s`
and equal `scoring_window_s`, plus unchanged `prediction_window_s=[-2,2]`.
Bounds are derived constants, not independently configurable. Document the
selected window in provenance; retain existing array names/axis meanings and
the dated output directory pattern.
New categorical units describe P(class 1), with its actual per-target label
in `target_metadata`; do not keep the old global "class 1 = left" statement
for reward. Preserve legacy metadata under its original two-class meanings.

The complete compatibility table is deliberately small:

| Version / schema | Mode | Exact targets | Supervised window |
| --- | --- | --- | --- |
| v1 / 1 | Generalization only | Original seven | Full |
| v2 / 2 | Generalization | Original seven | Full |
| v2 / 2 | Mixed | State/action only, T=2 | Full |
| v3 / 3 | Generalization | Original seven | Full, Before, After |
| v3 / 3 | Mixed | State/action/reward, T=3 | Full, Before, After |

Legacy compatibility remains read-only:

- Recognize v1/schema-1 as correct-rewarded generalization, require its exact
  seven targets, and derive the missing training-condition mask from its saved
  `correct_rewarded`. Normalize its missing mode/condition names as currently
  implemented and validate against the canonical membership column.
- For v2 retain its explicit mode/condition metadata and exact old target set.
  For v1/v2 normalize `training_window="full"` and matched scoring bounds in
  memory only. Verify original numeric training/prediction bounds and mask
  match Full; reject contradictory legacy window declarations rather than
  silently rewriting them. Missing v1 selected membership is derived as
  before; v2's stored selected membership remains required.
- Retain original identities/versions/bytes. Do not compare old code hashes to
  current source, rewrite products, or require raw sources. Old mixed T=2
  remains displayable/scorable; it contains no reward model to synthesize.
- Require every new field and exact T for v3. Reject unknown version/schema
  pairs, mode/target/window/bounds contradictions and wrong masks. New products
  never accept a two-target mixed tuple merely because legacy products do.
- Keep version resolution local to `results.py`, with one small helper for
  expected saved target tuples and normalization, shared by discovery/load/
  metadata validation. Preserve `validate_prediction_result(inputs, regions)`:
  structural checks can recognize the two finite mixed tuples (legacy T=2
  only at Full); publication/metadata checks enforce the exact schema. Do not
  put a saved-schema flag into the fitter or broaden to arbitrary subsets.
  Both saver and loader run array and metadata validation.
- Change discovery, loading, validation and reuse together. A different
  mode/window/target schema cannot satisfy the new identity. Explicit Prepare
  can produce v3; ordinary saved display never migrates or reruns old products.

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

Extend the existing prediction control group with **Analysis**, ahead of
**Prepare predictions**, **Recompute predictions**, mode-filtered saved
result/target selection, and existing PC mode/order/count controls. Prepare
computes only the selected mode/window's targets for the selected source configuration.
A deliberate Prepare
can reuse/prepare missing PCA through the existing runner; widget changes
alone never fit. Preserve existing PCA preparation and optional speed controls.

Use labels **Correct-rewarded generalization** and **Mixed-outcome decoding**.
Within those modes, use **State generalization** / **Action generalization**
and **State decoding** / **Action decoding** / **Reward decoding**, respectively. Regional track
titles append `(correct-rewarded training)` or `(mixed-outcome training)`.
Captions name all three mixed outcomes and retain the offline/PCA disclosure.
Summaries, hover and model details must identify mode and training pool.
Add **Training/scoring window** with **Full [-1,+1) s**, **Before [-1,0) s**
and **After [0,+1) s**, default Full. Filter saved runs by mode/window pair;
reflect actual saved target names, not the current three-target default.
Include the saved window in run labels, captions, track titles and model details.
Use target metadata for positive labels: reward uses `P(rewarded)`, not
`P(left)`; actual reward overlays are constant trial labels, not timed delivery.
For a legacy mixed run, show its two targets with a legacy-scope caption and
explicit Prepare action to add reward; do not claim reward is prepared.
Model details show per-fold total, outcome and binary-class training counts,
derived from saved masks/targets rather than another persisted tensor. Counts
expose sampling imbalance; they do not trigger automatic balancing.
Saved-run labels include mode, actual bin width and date/run identity; use
`PredictionRunOption.display_label` rather than discarding it for a bare path.
Keep these display helpers outside scientific configuration to avoid making a
text-label change invalidate fits. Do not introduce an estimator/view registry.

Use mode/window-scoped run/target widget keys or reset incompatible values before
widget creation. A selected mode with no saved run shows an unprepared
message; it must not display the other mode with a new label or auto-fit it.
For an initial saved-only open without previous selections, choose an
available mode/window pair if the default pair has no saved product. This
initializes selectors; it never overrides a later explicit unprepared choice.
Recompute availability is based on that mode's runs. Loading a saved run
drives labels from the saved mode, never from a stale preparation widget.

**Unprepared/hidden display contract:** retain the existing small data-only
`PredictionViewState` in `session_prediction_view.py`. It carries selected mode/preparation status,
an optional validated saved-result anchor, and an optional `PredictionDisplay`.
Append selected `training_window="full"` to the existing `PredictionViewState`
record and `PredictionRunOption`, preserving older construction calls.
Update its sole production caller, `render_session_timeline_view()`, only as
needed for the added window. Keep all variable controls/captions inside the
existing fixed Streamlit container so new widgets never move/remount the chart.
Keep only the anchor path in session state; use the existing cached loader
rather than storing another large PCA/model copy there.

- Prepared and shown: anchor and prediction display refer to the selected run.
- Prepared but hidden: keep its saved timeline/PCA anchor, display no predictions.
- Selected mode/window unprepared: retain the last valid saved anchor for this session,
  even if from the other mode. Show an explicit unprepared message and the
  background's actual source/bin settings; suppress prediction tracks, actual
  target overlays and weighted contributions. Ordinary raw PC activity can
  remain visible. Do not present the anchor's other-mode predictions as selected.
- With a valid saved anchor, route through its pinned timeline/PCA, regardless
  of whether a prediction overlay exists. Do not call raw configuration,
  behavior/spike loaders, reward enrichment, or preparation on this path.
  Only the absence of any usable saved anchor permits the existing raw path.

This is a local UI record and one retained path, not a new state-management
framework. Keep the chart mounted and the saved behavior/PCA visible during
the unprepared switch; a caption followed by early return is insufficient.

**Navigation contract:** preserve the implemented split of state namespaces.
Use session/time-origin scope for duration/start/Jump values and the explicit
navigation revision. Keep source-PC ranges, scaling/color controls, window
summaries and vertical-axis revisions scoped to the actual saved PCA basis;
prediction-specific vertical state also identifies its mode/model/target.
Update the footer's selected-trial lookup to use the navigation scope.

In `population/timeline_plotting.py`, separate horizontal identity from basis
identity. Layout/time-axis `uirevision` uses session/time origin plus explicit
navigation revision, never PCA configuration ID, prediction run, mode, window or bin
width. Keep the chart component key stable for the session; its existing JS
already transfers the actual browser time range to new matched axes when
that revision is unchanged. Preserve the requested range across a 500-ms
generalization to 100-ms mixed-run switch, while new PC controls/vertical
axes use their own basis/model identities. Do not re-key/remount the chart
or introduce browser-to-Python pan callbacks. Reset on a different session/time
origin; clamp if changed session bounds require it.

Bounded neural loading is the user-approved exception to range preservation.
It derives its loaded interval from the server's navigation controls, which
do not track browser-only pan. Preserve the actual browser range only if it
fits wholly inside the destination loaded interval; otherwise reset to the
control-selected visible interval. For example, browser pan at 1000-1060 s
with controls at 0-60 s cannot survive a switch to 20-ms data loading 0-120 s.
The bounded-loading caption must explain this reset behavior and identify the
loaded interval. This applies to run/mode switches and enabling bounded loading.
Retain the existing destination-bounds check in `webapp/timeline_chart.js`;
it explicitly applies the chosen range to matched time
axes so unchanged `uirevision` cannot retain an out-of-bounds range. Preserve
session-scoped navigation keys and the stable chart component. No pan callback,
automatic interval fetching, or additional loading framework is needed.
Keep two regional tracks, not four simultaneous comparisons or a new page.

Original saved-first integration is already implemented: discover/select
saved predictions before preparation-only raw-source validation, then load the
exact PCA archive pinned by the selected result. Validate session ID,
recorded identities, resolution, and region assignments from saved products.
Do not silently choose a different basis or run legacy reward enrichment from
raw tables for this saved path. Use rewards already saved in prediction
choices; absent optional legacy values remain unknown.

Preserve the implemented `session_inputs.metadata_view_availability()` gate:
allow this view when a completed saved prediction of either supported mode is discoverable even if raw
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
class-1 probability `[0,1]` with its actual target label; continuous axes show native units. Hover retains
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

Reuse the existing chart component/navigation with the horizontal/basis split
above. Mode/window/run/configuration changes preserve horizontal range subject to
the bounded-loading exception; appropriate vertical ranges reset.
Jump/start/duration explicitly update horizontal range.
Keep all-session navigation reachable with viewport-limited contribution
construction and chart payloads at fine resolution. The current NPZ loader
still loads full regional score arrays into memory before filtering; this is
not partial or bounded NPZ memory loading. Measure source-load RSS before
considering another storage mechanism. Inspect control crowding before adding a view.

**Performance records:** extend existing `scoring.py` and the same expander.
Set `SCORING_VERSION="trial-mean-performance-v2"`. Preserve public
`trial_metrics()` inputs/metric return conventions and the four report keys
`folds`, `held_out`, `laterality`, `gaps`; expand records, not storage systems.

- Compute the held-out trial mean once per region/target from the saved bin
  mask. Reconstruct each fold's training mean once using its exact saved
  mask/coefficients and that same interval. Sigmoid precedes binary averaging;
  every trial supplies one probability/value, not one independent score per bin.
  Training availability is based on the examined fold's fit and finite
  reconstructed values, not on a trial's own held-out model/reason. Failure of
  one held-out model must not remove its trials from other valid training fits.
- `folds` contains each fold's training and held-out rows by population.
  `held_out` retains pooled rows with each eligible trial counted once. Exact
  training intersections produce empty omission/incorrect rows in generalization;
  other-fold untrained choices never become training observations.
- `laterality`/`gaps` carry `split` (training/held_out), `fold` (0..4 or None
  for pooled held-out), population, choice/state grouping and factual counts.
  Keep per-fold side detail and pooled held-out side views. Never pool training
  sets across folds. Missing side labels exclude only the affected grouping.
- Binary performance records add integer `class_0_trials`/`class_1_trials`
  from their scored labels. Keep these outside the public metric helper's
  old return contract. Class counts sum to `n_trials`; side labels and reward
  class labels remain independent. Unavailable fits have no scored trials
  and explicit status/reasons; model details retain eligible training counts.
- Gap calculation enumerates actual metrics (accuracy/balanced accuracy or
  R2/RMSE), never all numeric fields: counts are not quantities to subtract as
  performance. Keep left/right trial counts factual. Single-class groups have
  accuracy but undefined balanced accuracy, and empty groups never become zero
  success. Existing negative/undefined R2 and native RMSE rules remain.
- Extend the existing formatter with generic split/fold/class-count handling,
  using target metadata for Right/Left versus Unrewarded/Rewarded. Retain percent
  versus percentage-point units. Preserve the existing binary gap display:
  ordinary-accuracy difference in pp, with balanced accuracy available in the
  per-side table; do not add a new balanced-accuracy gap column. Keep existing
  metric fields in report records rather than changing their public meaning.
  Use split/outcome/fold controls in the same panel. Held-out fold choices allow
  pooled or 0..4; training allows 0..4 only, resetting pooled to fold 0 when
  switching to training. Retain pooled held-out and per-fold training/test detail.
- Cache selected-target summaries by result/source path/stats, saved window
  and scoring version. Do not fit, write or reread raw tables during display,
  and do not repeat neural matrix products for every outcome/side selection.

## 8. Runner and batch behavior

Proposed runner boundary:
`prepare_session_predictions(metadata_path, *, config_path=None,
probe_roles=None, bin_width_ms=None, analysis_mode=None, training_window=None,
rerun=False, dry_run=False)`.
GUI selections are explicit overrides; metadata-only CLI uses Section 4 defaults.
Both call this same composition root. Ordinary commands remain:

```bash
uv run python -m src.neural_analysis.task_decoding.session_prediction.run_session \
  --session-metadata "$METADATA" --dry-run
uv run python -m src.neural_analysis.task_decoding.session_prediction.run_session \
  --session-metadata "$METADATA"
```

Optional `--config` selects another controls file; `--rerun` creates a new run.
Add optional `--analysis-mode` with only the two fixed identifiers; omission
uses the controls-file mode or the original generalization default. Users
can set `analysis_mode` once in the session controls and keep the metadata-only
call. An explicit one-off mixed preparation is:

```bash
uv run python -m src.neural_analysis.task_decoding.session_prediction.run_session \
  --session-metadata "$METADATA" --analysis-mode mixed_outcome_decoding
```

Add optional `--training-window {full,before,after}` to session and batch
runners; forward `training_window=None` to preserve each session's controls/
Full default. Metadata-only calls stay simple. A one-off half-window request:

```bash
uv run python -m src.neural_analysis.task_decoding.session_prediction.run_session \
  --session-metadata "$METADATA" --analysis-mode mixed_outcome_decoding \
  --training-window before
```

This new option describes the planned interface, not a command available
before implementation. Prepare resolves only that window's mask, fits the
existing model functions and records exact fit/scoring bounds. Keep existing
PCA prerequisite handling and source/target validation; add no automatic
three-window loop. Logs/summaries use saved window and class meaning rather
than literal [-1,+1) or "left" for every target.

Dry-run reads small controls/tables and source metadata, reports grids,
mode/window/bounds/targets/canonical training-condition names, candidate outcome counts,
regions, model counts/expected array bytes, and any missing PCA prerequisite;
no source PCA fitting, spikes, writes, or submission. Never call the old
neural decoder to prepare these targets. GUI computation is explicit with
status/log output, not triggered by a dropdown.

The batch wrapper takes metadata paths and calls the same runner, parallel
across sessions using the existing timeline batch worker-count convention.
Do not add model-level pools or a new memory scheduler. No batch is run during
this feature's initial validation.
Forward an optional batch `analysis_mode`/`--analysis-mode` override unchanged;
without it, each session uses its own controls/default. Do not add a separate
mixed-outcome runner or automatically prepare both modes.

## 9. Test-first work packages and acceptance

WP1-WP10 are completed baseline work; keep their evidence in Section 0.
Pending tests/file ownership are in Section 15. Do not restart old packages.

For each approved code package: write meaningful tests -> establish RED ->
commit tests -> implement -> focused GREEN -> review -> update Section 0.
Use explicit paths/path-limited commits and preserve unrelated staged work.
The first RED command is in WP11; its numerical checks are part of that gate.

At integration, run the prediction package and directly affected interfaces:

```bash
uv run pytest -q src/tests/neural_analysis/task_decoding/session_prediction \
  src/tests/neural_analysis/task_decoding/test_config.py \
  src/tests/neural_analysis/task_decoding/test_targets.py \
  src/tests/neural_analysis/test_single_session_setup.py \
  src/tests/neural_analysis/test_session_prediction_webapp.py \
  src/tests/neural_analysis/test_session_timeline.py \
  src/tests/neural_analysis/test_session_timeline_io.py \
  src/tests/neural_analysis/test_timeline_plotting.py \
  src/tests/neural_analysis/test_session_timeline_webapp.py \
  src/tests/neural_analysis/test_session_timeline_scripts.py \
  src/tests/neural_analysis/test_timeline_chart.py
```

Use `uv run`; the known workstation setup uses ignored project caches
(`UV_CACHE_DIR=.pytest_cache/uv`, `MPLCONFIGDIR=.pytest_cache/matplotlib`)
and an available `NODE_BINARY` for frontend checks, as recorded in Section 0.4.
Confirm runtime paths on resumption. Report skipped Node checks explicitly;
they do not establish frontend acceptance. Check changed Python with Ruff
and changed-path whitespace. Broaden tests only for new affected interfaces
or failures; no routine full neural suite.

A synthetic saved-only browser inspection verifies actual pan, chart position
and mode/window/target transitions that Python tests alone cannot establish.
Record exact outcomes/warnings rather than anticipated pass counts.

## 10. Performance and one-session validation

Prediction arrays for 1,000 windows per region and seven targets use about
0.85 MiB at 500 ms, 4.27 MiB at 100 ms, 21.36 MiB at 20 ms (float64, two
regions). The new three-target mixed product uses about 0.37, 1.83 and 9.16 MiB at
those resolutions. These are array-size estimates, not peak-memory/runtime measurements. The main
costs can be full PCA loading/preparation, a stacked training matrix, repeated
SVD fits, and browser payloads. Viewport filtering reduces browser/contribution
work, not the existing full NPZ score-array allocation. Release temporary matrices between models;
optimize/reuse factorizations only after measurement and correctness checks.

Reuse compatible source PCA; form only the visible target/PC contributions.
Report fit count, preparation/load/fit/save/render times, peak process RSS,
NPZ and chart payload bytes, warnings, and exact run/log paths. At 100 ms the
existing option must pass synthetic/UI tests; initial experimental validation
is 500 ms, not an automatic second-resolution or multi-session run.
For expanded mixed decoding, measure the 30 classifier fits per selected window and source
loading separately; the original 5.622-second seven-target preparation is
evidence for that run, not a promised mixed-mode runtime. Do not load both
large PCA copies or fit other modes/windows merely to offer selectors. The
full prediction tensor B is unchanged for Before/After, so smaller training
windows reduce stacked fit rows, not saved prediction coverage. Reconstruct
one training mean vector per target/region/fold and reuse it across outcome/
side filters; do not repeat matrix products for each row of a summary table.

Designated root:
`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260804_latent_inference`.
Use its existing `neural_session.json`; do not rewrite session metadata or old
outputs to make validation pass. If preparation is long, run it through the
established external execution workflow and record the process/job identity;
await completion outside an active coding turn. No new Slurm resources are
implied. User inspection precedes processing further sessions.

## 11. Implementation authorization

The earlier implementation authority covered the completed Full-window
baseline. The current request expands the revision-4 implementation plan;
it does not begin code or experimental processing. On implementation
instruction, proceed through WP11-WP15 without repeated package-level approval.
Record whether that instruction includes designated CT026 validation; when
included, do not ask for the same permission again. No broader sessions,
batch, cluster submission/resource change or transfer is implicit. Record
actual authority in Section 0; historical experimental authority does not
authorize a new run by itself.

## 12. Sol orchestrator and single-worker handoff rules

- One Sol orchestrator (`gpt-6.1-sol`, high) owns package sequencing, contracts,
  review/acceptance, commits and the live handoff. Sol directly writes WP11-WP13
  scientific/config/version/scoring changes and owns WP15 integration/browser
  verification and any explicitly authorized experimental validation.
- At most one active worker at a time, in addition to Sol. One optional Luna
  worker (`gpt-6-luna`, medium) handles a bounded WP14 runner/UI assignment or
  WP15 usage docs after contracts freeze; reuse it sequentially. Sol may do
  these small tasks directly. Maximum two active agents including orchestrator;
  no worker-created agents, overlapping worker tasks or extra review agent.
- Record actual model/effort/agent identity; role labels do not change the
  current chat's model. No agent is launched for this documentation audit.

Order: `WP11 -> WP12 -> WP13 -> WP14 -> WP15`. Freeze input/version/scoring
contracts before integration delegation. Give any worker its exact files,
RED/GREEN phase, acceptance tests/command, HEAD and user changes to preserve.
The worker writes tests; lead verifies RED and commits them before GREEN.
Lead reviews science/source/results and commits only owned paths. Worker
reports do not authorize new experiments or substitute for user acceptance.

Keep one writer per file, idle the worker before review/commit, and record
findings/next action at each gate or interruption. Use path-limited commits;
never amend/reset/revert user work. Existing tests, commits and this handoff
are sufficient coordination; no new worktree or locking system is needed.

## 13. Completed mixed-outcome baseline reference

WP6-WP8 implemented the two-mode Full-window baseline; WP9 completed usage,
synthetic and browser gates, with experimental acceptance separately pending.
That baseline writes v2: generalization T=7, mixed T=2, five block folds,
existing saved-PCA reuse, anchored portable display and session navigation.

Commit/test/browser evidence is retained verbatim in Sections 0.2 and 0.4.
Current source is the implementation reference; Sections 1-12 and 15 define
the revision-4 changes. These completed packages are not new work instructions.

## 14. Completed WP10 scoring reference

WP10 added pure saved-model trial-mean scoring over Full [-1,+1), binary
accuracy/balanced accuracy, numerical R2/RMSE, per-fold training/test rows and
pooled held-out outcome/choice/state laterality. Existing report keys are
`folds`, `held_out`, `laterality`, `gaps`; public `trial_metrics()` stays.

The recorded 237-test/browser gate and commits remain in Section 0.
WP13 below extends this scorer to saved windows and both splits by outcome.
Use current source and the new contracts rather than repeating the old
Full-only/held-out-only implementation instructions.

## 15. Revision 4: window/reward expansion and handoff gates

This is the pending implementation sequence. The design follows
SoftwareDesign.md's function-first composition and KISS/YAGNI guidance:
configuration selects fixed targets/bounds; preparation builds arrays/masks;
the existing fitter consumes them; persistence owns version compatibility;
scoring and UI consume saved products. No new package, dependency, estimator
factory, per-window model class, generic result-migration engine or scheduler.
Only input/preparation/persistence/scoring/integration seams change. In
particular, do not rewrite the already generic fitter or prediction helper.

### WP11 - fixed windows and actual-reward preparation (Sol orchestrator, direct writer)

Write tests first, establish RED and commit tests before source changes:

- Prediction `test_config.py`: absent controls/old JSON retain Full; accept
  exactly three window IDs; config versus explicit-override precedence;
  reject unknown windows, arbitrary numeric bounds and penalty overrides.
- Shared `task_decoding/test_config.py` and `test_targets.py`: current reward
  registration, exact 0/1 class orientation and same-row alignment; reward
  disagrees with correctness on a correct omission; reward differs from the
  previous trial. Validate missing/malformed values, manual/no-choice baseline
  exclusions, and unchanged omitted/default/explicit old target selections.
- `test_single_session_setup.py`: newly scaffolded task-decoding configs retain
  the exact original default target order without reward. Retain the existing
  regression for byte-for-byte preservation of files on repeated setup.
- `test_preparation.py`: all three masks on 500/100/50/20-ms saved grids;
  time-zero partition; 4/2/2 bins at 500 ms; seven generalization and three mixed
  targets; unchanged PCA scores, complete-window trial eligibility and folds
  across windows. Missing reward never becomes zero or suppresses an otherwise
  valid other target; canonical membership remains independently authoritative.
- Bounded `test_modeling.py`/`test_contributions.py` additions: unchanged Full
  behavior, 1/4 versus 1/2 weights at 500 ms, trial total one, no held-out rows
  in fits, full B-bin predictions for all windows, and reconstruction/ranking
  from the actual selected training rows. Use independent weighted affine/
  logistic objective checks already present; do not weaken their tolerances.
- Seed a synthetic example with separate state/action/reward signals and both
  reward classes in training folds. Test reward recovery and a separate
  constant/missing-reward case where only that target is unavailable. Update
  fixtures for the genuinely changed T=3 contract; preserve legacy T=2 fixtures
  separately rather than changing expectations merely to pass tests.

Implement the small shared target registration, preserved parser/scaffolder
defaults and Section 4's fixed window controls/mask selection. Keep the existing
fitter/predictor signatures and computations. An empty/single-class reward fit uses the existing local failure
path; a preparation-level target-availability reason can explain absent labels.
No new fit fallback or binary-target scaling is needed.

First focused gate after writing these tests:

```bash
uv run pytest -q src/tests/neural_analysis/task_decoding/session_prediction/test_config.py \
  src/tests/neural_analysis/task_decoding/session_prediction/test_preparation.py \
  src/tests/neural_analysis/task_decoding/session_prediction/test_modeling.py \
  src/tests/neural_analysis/task_decoding/session_prediction/test_contributions.py \
  src/tests/neural_analysis/task_decoding/test_config.py \
  src/tests/neural_analysis/task_decoding/test_targets.py \
  src/tests/neural_analysis/test_single_session_setup.py
```

### WP12 - strict v3 products and legacy saved display (Sol orchestrator, direct writer)

Add/commit RED tests in `test_results.py` before implementing Section 6:

- Round-trip each mode/window with unchanged prediction axes/units, exact
  mask/bounds and all new reward metadata. New mixed products have exactly
  three targets; new generalization products exactly seven.
- Equal source data but different windows give different identities; target
  scope and scoring bounds cannot be silently substituted. Reject a mask
  selecting the wrong half despite correct length, mismatched bounds/bin width,
  unknown versions or v3 fields/targets missing from metadata/NPZ.
- Load faithful synthetic v1 generalization, v2 generalization, and v2 mixed
  archives that contain no new window field. Verify Full normalization in
  memory, actual target tuples, original identities/bytes and source-PCA pinning.
  A v3 product relabeled as v2 is not adequate legacy coverage.
- Reject legacy half-window declarations and reward-target injection, and v3
  mixed archives containing only state/action. Discovery, loading and explicit
  preparation reuse must agree about the version/mode/window contract.
- Relocate products without raw files and score/display legacy two-target
  mixed results under Full. No rewards are synthesized and no migration writes.

Implement one version-resolution path in existing `results.py`; keep shared
array names and numerical data records. Append only the window field/default
where needed. Preserve the public structural validator and enforce exact
versioned target/window rules at metadata/publication boundaries. Freeze the
v3 contract before runner/UI work; review original-public-interface callers.

### WP13 - matched-window split/outcome/laterality scores (Sol orchestrator, direct writer)

First add/commit failing tests in `test_scoring.py`:

- Probability means precede thresholds in Full/Before/After. Deliberately
  different outer/opposite-half bins do not affect a selected window's score.
  Continuous means/RMSE retain units and negative/undefined R2 handling.
- Independently reconstruct each fold's training means from saved coefficients
  and exact masks. Score every outcome intersection for training and held-out;
  generalization training omissions/incorrect have zero scored rows and no
  metric. Never label untrained other-fold rows as training.
- Per-fold held-out and pooled totals agree, without duplicate trials in each
  region/population. Overlapping training folds remain separate. Cover
  unavailable fits, missing targets/folds/side labels and regional coverage.
  Include a case where one fold cannot fit but other folds fit: trials whose
  own held-out model failed remain in other models' exact training-score masks.
- Choice and state/context side masks differ on incorrect trials. Verify each
  split/outcome/fold's side metrics/counts, pooled held-out laterality and
  left-minus-right units against direct small-array oracles.
- Reward class counts sum to scored trials; correct-rewarded and omission
  groups have one class and ordinary accuracy but no balanced accuracy.
  Wrong counts must not become side-performance differences. Preserve the
  public `trial_metrics()` return contract and source/archive bytes.

Implement Section 7's record expansion in the existing scorer. Reconstruct a
training mean vector once per fold, then use direct Boolean masks for outcome/
side reporting. Retain four report lists, small cached summaries and no raw
reads/refits. Use explicit metric-name selection for differences. Update the
old callers/tests for the added split/fold selectors rather than creating a
second scoring pipeline.

### WP14 - runner and existing timeline inspection (Sol orchestrator; optional single Luna worker)

Commit RED tests before each runner/UI source change:

- `test_scripts.py`: config/session/batch/CLI window precedence and forwarding;
  metadata-only Full defaults; selected-window dry-run masks/bounds/counts;
  synthetic prepare/reuse/immutable rerun for both modes and half windows.
  All choices reuse matching PCA; neither other windows nor other modes fit.
- `test_plotting.py` and directly affected timeline tests: Before/After keep
  full [-2,+2) curves and gaps, exact saved-window captions, actual reward 0/1
  overlays, `P(rewarded)` and positive-class hover, and log-odds contributions.
  Existing Full and legacy two-target displays stay usable.
- Actual Streamlit `test_session_prediction_webapp.py`: filter mode/window
  pairs, initialize from an available saved pair, reflect saved target sets,
  and explain missing reward in legacy products. Selecting Before/After does
  not crop/relabel a Full model or fit from a widget change.
- With raw sources removed, switch from a prepared pair to an unprepared
  window and back; keep the saved behavior/PCA anchor, suppress all prediction-
  derived overlays, retain the browser chart position/time range and preserve
  the approved bounded-loading exception. No new pan callback or JS subsystem.
- Performance controls show both splits/outcomes and both side groupings,
  with per-fold training detail, pooled held-out views and correct class/count/
  percentage labels. Cache keys include the saved window and scoring version;
  changing filters/viewport/PC count never fits or recomputes neural products.
  Test binary reward and continuous targets, including empty/undefined groups.

Update existing runner kwargs/CLI, metadata, summary and snapshot plumbing.
Add the window selector inside the fixed controls container and extend the
existing view-state/discovery records. Keep version handling in persistence,
numeric mask selection in preparation, and target-dependent probability labels
in display helpers. The two analysis names and existing navigation/component
stay. Lead reviews all scientific/version/window plumbing before accepting any
Luna integration.

### WP15 - usage, bounded integration and acceptance (Sol orchestrator + user; optional Luna usage docs)

Update existing usage guides without deleting commands or copy-back guidance.
Explain selected-window preparation and inspection, Full defaults and legacy
targets, reward interpretation, matched scoring, empty training strata and
single-class outcome scores. Keep calls metadata-only when controls are saved.

Run the Section 9 integration command plus shared config tests, changed-file
Ruff/whitespace checks and one synthetic saved-only browser inspection. Cover
both regions, all windows, reward/continuous targets, mode/window switching,
laterality, chart position and actual pan. Record real commands/counts/warnings
and RED/GREEN commits in Section 0; the historical 237 tests are baseline evidence.
No full neural suite, new dependency or routine fine-resolution real rerun.

If the implementation instruction includes designated CT026 2026-08-04
validation, use only the authorized 500-ms variants. Preserve the existing
Full generalization result; prepare the new Before/After generalization and
requested mixed reward variants without automatically rerunning that baseline.
Verify saved masks/weights/folds, class orientation and affine/logit prediction
reconstruction, then user inspection. Record runtime/RSS/output identities and
paths. No new sessions, batch, cluster job, transfer or resource adjustment is
implied. If experimental validation is not authorized, finish synthetic/code
gates and record that remaining acceptance gate explicitly.
