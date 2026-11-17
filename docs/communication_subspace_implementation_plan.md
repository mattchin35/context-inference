# Communication Subspace: Implementation Plan and Live Handoff

Status: Implementation approved, 2026-11-16; WP1/WP2 tests in progress.
Experimental execution awaits a designated session after synthetic acceptance.

The [scientific spec](communication_subspace_spec_v1.md) owns scientific scope.
This plan applies [SoftwareDesign.md](SoftwareDesign.md) and
[AGENTS.md](../AGENTS.md) to the inspected codebase. Approval of this plan also
accepts the proposed technical defaults explicitly listed below.

## 0. Live implementation handoff

Start here when resuming. This follows the existing
[CCA handoff](session_timeline_cca_implementation_plan.md#0-live-implementation-handoff)
and [prediction handoff](condition_prediction_generalization_implementation_plan.md#0-live-implementation-handoff).
The orchestrator updates this section at assignment, RED/test commit,
GREEN/review, ownership transfer, interruption, and experimental-data gates.
This is an implementation record, not runtime checkpoint/resume machinery.

### 0.1 Current snapshot

| Field | Current value |
| --- | --- |
| Observation / checkout | 2026-11-16; branch `refactor`; inspected HEAD `c702784`. Recheck on resume; this is not a reset target. |
| Authorization | User: "yep yep, go ahead with implementation". Plan/defaults and implementation approved; no experimental session designated. |
| Phase | WP1/WP2 tests-first work dispatched; other packages pending dependencies. |
| Orchestrator | Planned Astra lead; owns scientific acceptance, integration, Git and this handoff. Current documentation writer is `/root`; runtime model/effort not independently recorded. Do not infer actual model from planned role. |
| Workers / ownership | `/root/scientific` and `/root/inputs`, both `gpt-6-sol`, active on WP1/WP2 tests only until test commits. Exact scopes in Section 0.3. |
| Feature files | `docs/communication_subspace_spec_v1.md` and this plan; no feature source/tests implemented. No feature commit recorded. |
| Existing worktree | Eleven pre-existing modified tracked files in prediction/spec/view code, `src/main.py`, `src/neural_analysis/SESSION_TIMELINE.md`, and `src/neural_analysis/sync_ephys.py`; unrelated untracked files also exist. Preserve all. Record exact owned paths/diffs before implementation. |
| Latest checks | Documentation inspection only. No RED/GREEN, synthetic, browser or experimental checks run for this feature. |
| Live commands / artifacts | No feature worker, test, preview server or analysis started; no result archives. This does not assert that unrelated project processes are stopped. |
| Pending decisions | Designated experimental session after synthetic acceptance. |
| Next action / owner | Root reviews WP1/WP2 contracts/RED, commits tests, then authorizes GREEN. |

### 0.2 Work-package ledger

States: not started, RED, tests committed, GREEN pending review, accepted,
interrupted, or waiting for a named decision. Worker completion is not Astra
acceptance. Record actual agent IDs at dispatch, not just planned model names.

| Package | Planned owner | State / next gate | RED command/result and test commit | GREEN checks/commit and Astra review |
| --- | --- | --- | --- | --- |
| WP0 spec/plan/handoff | Astra orchestrator | Approved | Not applicable | Documentation only |
| WP1 model/scoring | `/root/scientific` Sol | Tests in progress | Pending | None |
| WP2 inputs/splits | `/root/inputs` Sol | Tests in progress | Pending | None |
| WP3 CV composition | Sol scientific | Not started; WP1/WP2 acceptance | None | None |
| WP4 results/runner | Sol input | Not started; WP3 and runner ownership transfer | None | None |
| WP5a plots | Sol scientific | Not started; accepted result contract | None | None |
| WP5b webapp | Luna UI + Astra routing | Not started; WP4/WP5a acceptance | None | None |
| WP6a synthetic acceptance | Sol scientific + Astra acceptance | Not started; integrated pipeline | None | None |
| WP6b usage docs | Luna UI + Astra review | Not started; accepted behavior/evidence | Not applicable | None |
| WP7 designated session | Astra + user inspection | Not started; synthetic gate and session designation | As needed for discovered defects | None |

### 0.3 Active work and interruption checkpoint

Active assignments (no worker Git writes):

- `/root/scientific`, `gpt-6-sol`: WP1 tests `test_modeling.py` and
  `test_scoring.py`; future GREEN ownership `modeling.py`, `scoring.py`,
  `__init__.py`. Must report RED and wait for root test commit.
- `/root/inputs`, `gpt-6-sol`: WP2 tests `test_preparation.py`,
  `test_validation_splits.py`, `conftest.py`; future GREEN ownership
  `configuration.py`, `preparation.py`. Must report RED and wait for root commit.
- Root owns this plan, Git, integration and later shared routing. No experimental
  process or preview started. Source paths are in the new communication_subspace
  package; test paths in its matching test directory.

At each dispatch or interruption, update each record with:

- Package, actual agent ID/model/effort when known, state, and exact writable
  files; list shared files reserved to Astra or another worker.
- Last accepted interface/commit; test-only commit; uncommitted owned changes;
  current RED/GREEN/review gate and known failures.
- Running command, working directory, tool session ID/PID, log/output paths,
  and whether work is running, stopped, finished, or unverified. For previews,
  record port/browser target; for analyses, record session/configuration/run ID.
- Next exact action or command, responsible owner, dependencies, and missing
  authority, if any. Distinguish requested cancellation from confirmed stop.

Before a planned interruption, request a worker checkpoint and obtain its
status. If that is impossible, mark status unverified rather than assume the
worker stopped. Preserve partial source and failing tests; do not reset files,
discard output, or claim a partial run is complete. A documentation checkpoint
does not require committing unfinished implementation.

For worker replacement, first verify the old writer is stopped or has relinquished
the exact scope, record its final diff/evidence, then transfer ownership. In
particular, never overlap WP3/WP4 writes to `run_session.py` or concurrent writes
to `conftest.py`. A new orchestrator must reconcile live workers before dispatch.

### 0.4 Resume procedure

1. Read repository instructions, the spec, this snapshot/ledger, Section 8,
   and the latest evidence entry. Preserve existing authorization; do not ask
   again for an action already approved. Do not infer approval from a planned task.
2. Verify actual branch/HEAD, staged/unstaged diffs, untracked feature files,
   and owned scopes against the checkpoint. Stop to clarify unexpected edits
   or ownership ambiguity; preserve unrelated and partial work.
3. Inspect recorded agents/processes/logs before replacing a worker or restarting
   a command. Lost tool output or chat context does not prove process death.
   Avoid duplicate analysis runs and competing writers.
4. Verify the recorded test commit and evidence. Repeat only interrupted,
   missing, or invalidated checks. Resume the unfinished TDD gate: tests -> RED
   -> Astra test-only commit -> authorized GREEN -> review. Do not backfill a
   supposed RED result after implementation already exists.
5. Confirm file ownership and next action in Section 0.3, then resume. Completed
   packages and saved experimental runs are not rerun merely to recover context.
   Preserve incomplete-run logs and apply the documented publication policy.
6. Update snapshot/ledger and append evidence after the next boundary. Scientific
   changes require updating the spec/plan and recording their authority before
   dependent implementation proceeds.

### 0.5 Append-only evidence log

Keep current state in Sections 0.1-0.3 and chronological evidence here. Append
corrections to mistaken historical claims rather than silently rewriting them.

```text
YYYY-MM-DD HH:MM TZ - package / phase
Authority and decisions:
Actual orchestrator/worker IDs, model/effort if known, writable scope:
Start/end HEAD; owned staged/unstaged files; ownership transfers:
Exact RED/GREEN/review commands, outcomes, warnings, test/implementation commits:
Scientific/interface findings and unresolved risks:
Live commands/agents, tool sessions/PIDs, logs, previews/results, or none:
Next exact action, owner, and remaining gate:
```

2026-11-16 - WP0 / live handoff initialized. User requested an interruption and
implementation-transfer record. Read CCA/prediction handoff precedents and
inspected branch `refactor`, HEAD `c702784`, and tracked worktree status.
Changed this plan only; no workers, code, tests, analysis or commits initiated.
Next gate: implementation approval, then Astra-owned contract/TDD dispatch.

## 1. Delivery and design decisions

Deliver a dedicated metadata-session **Communication subspace** webapp view,
a callable/CLI single-session runner, saved numerical results, light PNG plots,
and a short usage guide. Implement the four prediction directions, matched
random populations, five draws, five chronological outer folds, inner ridge
tuning, and raw choice-window activity exactly as specified.

Apply SoftwareDesign.md:

- Use small pure functions for sampling, splitting, preprocessing, fitting,
  scoring, and summarization. Use dataclasses only for coherent records.
- Pass arrays and small records into computations, not session objects or
  Streamlit state. Session/path resolution belongs at the preparation boundary.
- Keep filesystem writes and logging in persistence/runner code. Plotting takes
  saved results and never fits or loads raw spikes.
- Use NumPy arrays, pandas tables, native Pynapple groups, and a typed progress
  callback. No estimator framework or abstract interfaces are needed.
- Add no dependency. Reuse NumPy, SciPy, pandas, Pynapple, Matplotlib, Streamlit,
  and pytest already declared in `pyproject.toml`.
- Preserve existing public interfaces. Do not refactor CCA, PCA, interregional
  regression, synchronization, or metadata schemas as part of this feature.
- All new functions document purpose, input/output types, axes, units, behavior,
  and relevant limitations. Public computations precede grouped local helpers.

Batch processing, background jobs, and publication-level inference are excluded.

## 2. Code inspection and reuse map

Paths below are relative to the repository root. These are inspected project
interfaces, not hypothetical external APIs.

| Existing code | Reuse decision |
| --- | --- |
| `src/neural_analysis/session_metadata.py`: `load_session_metadata`, `resolve_session_metadata`, `resolve_probe_sources` | Reuse metadata resolution at the input boundary. No schema change. |
| `population/run_session_timeline.py`: `resolve_timeline_probe_roles` | Reuse metadata/default versus explicit role resolution. Add feature-local validation requiring exactly PFC/HPC and distinct configured probes. Empty or partial roles never enable computation. |
| `population/session_timeline_io.py`: `select_timeline_population` | Reuse metadata-only selection with explicit `population_quality="good"`, `missing_label_policy="exclude"`. This uses manual Phy `group`, good inside-brain channels, metadata restrictions, and stable cluster ordering; no KSLabel fallback. |
| Same module: `load_timeline_population` | Reuse synchronized spike loading, selected cluster validation, probe-qualified unit IDs, Pynapple groups, and trusted coverage. |
| `task_decoding/activity.py`: `inspect_probe_coverage` | Use the public coverage routine already used by timeline input preparation; do not infer coverage from first/last observed spikes. |
| `population/session_timeline.py`: `prepare_timeline_choices` | Reuse completed ordinary choice selection, original row positions, UTC origin, action/reward conventions. It does not preserve every optional trial-table column. |
| Same module: `select_complete_choice_windows`, `choice_bin_edges`, `bin_choice_counts` | Reuse common-window selection, exact supported widths, and unsmoothed counts. Add an adapter-level overlap contract test before relying on multi-window binning. |
| `population/pca.py`: `build_trial_unit_rate_tensor_pynapple` | Existing underlying binner returns Hz; `bin_choice_counts` multiplies by width to restore counts. Preserve that conversion and axis order. |
| `population/session_cca_io.py` | Follow pairing and persistence patterns; do not call its loaders/configuration directly, which carry CCA-specific semantics. Reuse lower-level public functions, not CCA records/private helpers. |
| `interregional/linear.py` | Do not reuse its OLS estimator: it rejects rank-deficient designs and is unregularized. Its restricted/full scoring and population-quartile summaries also differ from this spec. Avoid an adapter that hides those semantic differences. |
| `population/run_session_cca.py`, `webapp/cca_view.py` | Follow explicit run/reuse/recompute, logging, script snapshot, saved-result controls and caching. No private-helper imports or batch/benchmark machinery. |
| `webapp/session_inputs.py` | Add the view option and allow entry to its controls before generic raw-source prerequisites. Run readiness belongs to the view and is scoped to the selected PFC/HPC pair. |
| `webapp/app.py`: `_start_metadata_webapp`, manual-path branch of `main` | Dispatch before active-probe/trial controls and return. Manual-path launch explains the metadata requirement before signal loading. |
| `src/tests/neural_analysis/session_cca/` | Follow synthetic-fixture and AppTest patterns; use a feature-local fixture with sufficient units and choices for CV. |

### Two input details to handle locally

1. The existing Pynapple binner constructs an `IntervalSet` from all windows.
   Verify installed Pynapple behavior for overlapping/touching intervals before
   implementation. Test exact shape, identity, and half-open endpoints. If
   windows cannot safely be batched, call `bin_choice_counts` once per choice
   and concatenate in original order; keep the existing batched call for safe
   separated windows. This preserves all eligible choices without changing the
   shared binner. Do not switch silently to the NumPy histogram implementation:
   that implementation includes the final right edge of its last bin.
2. Read the trial CSV once in preparation and call `prepare_timeline_choices`.
   Preserve original selected trial rows separately using `row_position`, so
   optional context/outcome columns survive without inventing aliases or changing
   the shared choice-table schema. This analysis does not need event CSV loading;
   missing or malformed optional events should not block it.

## 3. Proposed module boundaries and APIs

New package: `src/neural_analysis/communication_subspace/`, with a minimal
`__init__.py`. All names below are proposed project APIs; implementation must
verify any newly used library API against installed source or official docs.

| Module | Responsibility and proposed functions/records |
| --- | --- |
| `configuration.py` | `CommunicationConfig`, `validate_configuration`; fixed scientific constants, user bin width/seed, canonical configuration serialization. No data loading. |
| `preparation.py` | `PreparedInputs`, `inspect_session_inputs`, `load_session_inputs`, `draw_populations`, `build_validation_splits`; source inspection/loading and separately testable pure sampling/split functions. |
| `modeling.py` | `TrainingTransform`, `RidgeDecomposition`, `fit_training_transform`, `transform_counts`, `fit_ridge_decomposition`, `predict_rank`, `select_alphas`; numerical functions with no filesystem/UI dependency. |
| `scoring.py` | `score_target_predictions`, `summarize_fold_scores`, `summarize_draw_scores`; explicit validity masks and descriptive summaries. |
| `results.py` | `CommunicationResult`, `ResultInfo`, `save_result`, `load_result`, `discover_results`; result records, schema checks, metadata and named-array persistence. |
| `run_session.py` | `compute_communication_subspace`, `run_session`, `main`; pure composition of prepared inputs separated from side-effecting run/reuse/log/save orchestration. Return a small `RunReport` to the UI/CLI. |
| `plotting.py` | `plot_rank_curves`, `save_rank_figure`; figure construction from results and explicit export, no refitting. |
| `webapp/communication_subspace_view.py` | `render_communication_subspace_view`, cached saved loading, readiness and run controls. No scientific calculations. |

Keep records close to their owning modules; avoid additional registries.

Dependency flow: metadata/loaders -> preparation -> computation -> results;
runner assembles these and persistence; plotting consumes results; webapp calls
runner and saved-load/plot functions. Numerical modules do not import webapp,
metadata, persistence, or run-control flags.

### Key data contracts

- Prepared counts: two float64 arrays `(choice, bin, unit_region)`, spikes/bin;
  unit IDs are ordered probe-qualified strings. Shared choice indices and bin
  edges prove pairing; choice times are UTC seconds.
- Population draws: ordered integer indices `(draw, n)` for each region/role,
  with corresponding IDs retained. Explicit seed; no global RNG state.
- Splits: a tuple of five small records containing outer train/test choice
  indices, inner train/validation indices, and purged positions. All indices
  refer to the common choice axis, never flattened observation IDs.
- Model inputs: float64 `(observation, unit)`, choice-major/bin-major order.
  Training transforms retain means/SDs in counts/bin and variable masks.
- Model fit: standardized coefficients `(source_variable, target_variable)`;
  standardized output dimensionless; inverse transform returns counts/bin.
- Fold score table: one row per draw/direction/fold/model/rank/target, including
  SSE, SST, R2, selected alpha, status/reason and stable identity columns.
  Use `model="ridge"` with no requested rank for unrestricted models; do not
  pretend the full baseline is a reduced-rank fit with rank zero.
- Fit metadata and compact factors live separately from score tables. Repeated
  arrays are not embedded into every table row.

## 4. Numerical and validation procedure

1. Determine `n = floor(min(N_PFC, N_HPC)/2)` from eligible good-unit counts.
   Require n >= 2. Generate five deterministic draws with n disjoint source and
   n target units in each region. Draws are fixed before CV, independent of
   observed activity. Preserve all four pairing relationships from the spec.
2. Split common choices chronologically into five nearly equal blocks, assigning
   remainder choices to the earliest blocks. For each outer test block, purge
   training intervals that
   overlap any test interval. For half-open windows, touching endpoints alone
   are not overlap. Check physical intervals, not just trial identifiers.
3. Select the latest `ceil(0.2 * outer_training_choice_count)` choices for inner
   validation. Purge inner-training overlaps with that block. Require at least
   two choices in each train/validation/test set after purging. Use the same
   splits for all draws, comparisons, ranks, and unrestricted models. Outer
   training can contain choices before and after the test block: this estimates
   held-out session prediction, not prospective forecasting. Reject an invalid
   shared split plan before fitting, rather than silently skipping folds.
4. Fit inner-training transforms with `ddof=0`. Constant source columns are
   omitted, constant target columns get mean-only predictions. No replacement
   neurons and no test-derived selection. Compute rank limits from training
   information only. Repeat transforms independently for the outer refit.
   For each training partition, the candidate rank cap is
   `min(numerical_rank(X_standardized), number_of_variable_targets)`.
   Do not cap ranks by observed prediction performance or numerical rank of
   X.T@Y: zero cross-covariance can legitimately produce zero coefficients
   at any allowed rank because the constraint is rank(B) <= r, not rank(B)=r.
5. For each of 13 alphas `10**[-6,...,6]`, minimize
   `||Y-XB||_F^2/N + alpha*||B||_F^2` with an unpenalized intercept represented
   by training centering. A stable reference formulation is:
   `A=X.T@X + N*alpha*I`, `A=L@L.T`, `C=X.T@Y`, `M=solve(L,C)`;
   right singular vectors of M give the penalized predictive target directions.
   Solve A B_ridge = C and construct B_r = B_ridge V_r V_r.T.
   Never form an explicit inverse. The independent reference uses
   `X_aug=[X; sqrt(N*alpha)*I]`, `Y_aug=[Y; 0]`: OLS followed by projection
   onto the right singular vectors of its augmented fitted responses. This
   checks the penalized solution without repeating its Cholesky construction.
6. Score ranks 1..n where supported and unrestricted ridge on inner validation.
   Select alpha independently per rank/model by mean per-target R2 using the
   same valid target mask, determined only by positive validation SST.
   Nonfinite predictions are fit failures, not reasons to drop a difficult
   target. Proposed tie tolerance: absolute 1e-12, favor larger alpha. Save grid
   scores, selected alphas, boundary selections and statuses.
7. Refit only needed alphas on purged outer-training observations. Score on the
   untouched outer test block. Use the minimum of inner/outer supported ranks
   for displayed feasible ranks; do not silently clamp requested ranks.
   A vanished training predictor or unidentifiable rank is recorded explicitly.
8. Compute per-target R2 as in the spec, then mean per-target fold score, equal
   mean of five complete folds per draw, and mean over complete draws. Undefined
   scores are not zero. Keep negative scores and target-count diagnostics.
9. Show the unrestricted mean over complete baseline draws as a horizontal
   reference. At ranks with fewer contributing draws, mark the RRR mean as
   incomplete and show the baseline mean over those same draws as a separate
   reference marker. Save the exact contributing draw IDs, not only counts.
   Use the intersection of complete RRR and baseline draws for both paired
   markers (RRR and baseline), omitting them when the intersection is empty;
   never interpret unmatched means as a performance difference. Individual draw
   curves remain visible, with no confidence-interval interpretation.

Technical defaults proposed for approval: 100 ms initial bin width, seed 0,
n >= 2, 20% inner chronological validation, 13-alpha grid, 1e-12 tie tolerance,
and standard float64 SVD rank tolerance `max(shape)*eps*largest_singular_value`.
Detect exactly zero variance separately; the numerical rank tolerance must not
become an undocumented firing-rate cutoff. If all inner R2 scores are undefined,
mark tuning unavailable rather than choose an arbitrary penalty.

Unrestricted ridge accepts collinear predictors. At fixed alpha and the full
algebraic cap, RRR must reproduce ridge to numerical tolerance; a tolerance-based
cap must account for discarded singular directions in that assertion. For a
well-conditioned fixture, maximal-rank tuning must agree with ridge as well.

Failure policy: malformed/nonfinite inputs and invalid shared splits fail the
run. No variable training source/target, undefined tuning scores, or numerical
fit failure yield explicit draw/direction/fold statuses; never replace them with
zero scores. An alpha candidate with a numerical failure is excluded and logged;
all candidates failing makes tuning unavailable. No complete curve is reported
without five valid folds. Expected unsupported ranks are not run failures.
A successful run has a valid rank-1 curve and unrestricted baseline for all four
directions and all five draws. Otherwise persist diagnostics as `partial` (some
valid scores) or `unavailable` (none); neither qualifies for automatic reuse as
a successful computation. Unexpected programming errors propagate with logs.

## 5. Persistence and run lifecycle

Use this concrete session-local layout, outside the repository:

```text
<session_root>/communication_subspace/
    results/<run_id>.npz
    runs/communication_subspace_<date>_<time>_<identity>/
        config.json
        run_session.py
        run.log
        summary.md
        rank_curves.png
```

`run_id` identifies the timestamped run, including explicit recomputations.
Use America/New_York for readable directory stamps and UTC for metadata/logs.
Record result/run-directory paths relative to the session root for discovery.
No fake batch-script copy; summary states single-session execution only.

Follow the existing trusted local NPZ convention: named numeric arrays, record
arrays for tables, and a dictionary named `meta` (object loading is therefore
restricted to trusted project artifacts). Version the schema and restore dtypes
explicitly. Validate table identity axes and factor shapes when loading.

Save enough to reconstruct every outer fitted prediction from prepared inputs:
training transforms/masks; selected alpha per model/rank; one ridge coefficient
matrix and ordered target basis per actually selected alpha, fold, draw and
direction. Reuse prefixes of each basis across ranks. Do not store a separate
dense coefficient matrix or prediction tensor for every rank. Save inner grid
scores and split provenance, not all inner candidate models.

Configuration identity includes scientific parameters, unit/channel selection,
probe roles, source identities, schema/analysis version, relevant dependency
versions, and a code fingerprint. Include UTC-alignment file/coverage, cluster
assignments, manual
labels, channel quality, trial table, and metadata. Save Git HEAD and dirty-state
information, but also fingerprint the feature and directly used analysis source
files: HEAD alone cannot distinguish dirty-code changes. Do not make unrelated
documentation edits invalidate a numerical result. For large scientific source
files follow existing path/size/mtime identity conventions, explicitly documented
as change detection rather than full content hashing.

Run sequence: validate configuration/source metadata -> find compatible saved
result unless rerun -> load counts -> compute -> write figure/summary/log ->
save final NPZ through a temporary sibling and rename. Recheck relevant source
and code identities before publication. Refuse existing final paths. Discovery
lists finalized NPZ files with their scientific status; partial diagnostic
results are distinct from interrupted writes, which are never listed. Retain
interrupted run folders for logs; no resume or transaction framework.

Saved-result discovery/loading must work without source spikes or behavior.
Reusing a result for a new requested computation still requires checking current
configuration/source identity. Distinguish that action from simply viewing an
older saved result. Show warnings when skipping malformed saved artifacts.

## 6. Webapp integration and outputs

Existing source changes should be limited to:

- `webapp/session_inputs.py`: option constant and entry to this view before
  generic source gates, including when roles are unset or sources are missing.
- `webapp/app.py`: import/metadata dispatch and early manual-path explanation.
- `src/neural_analysis/README.md`: short feature/usage entry.

Other behavior lives in the new package/view. Add a local package README rather
than expanding unrelated CCA documentation. Existing package exports need not
be changed unless an inspected caller actually requires them.

The view discovers saved results first, then offers region roles, bin width,
and seed. Prefill roles from metadata; reject partial or identical assignments.
Always allow access to these controls and historical results. Disable Run and
Recompute until the selected pair passes readiness, regardless of unrelated
probes. Show good-unit counts, matched n, common choices, split feasibility, and
specific failures using metadata/trials/coverage only. Raw spikes load on Run;
optional event/LFP inputs are irrelevant.

Run uses compatible reuse; Recompute always creates a new result. The view calls
the same runner as the CLI, with a small progress callback invoked per draw/fold.
Do not call Streamlit inside numerical functions. Failures preserve the selected
saved plot and provide an explicit reason.

Two light Matplotlib panels show mean rank curves and faint individual draws,
within/cross labels, unrestricted-ridge references, and R2=0. Captions explain
raw activity, simultaneous prediction, CV, and incomplete support. Save the same
figure as PNG and close figure resources after rendering/export. Diagnostics
include unit IDs/counts, fold/purge counts, valid targets/draws, and selected
alphas. Cache saved loads by path, size, and mtime; control changes cannot trigger
a fit or relabel saved settings. A result with no valid curves still renders its
diagnostic explanation instead of failing during plotting.

CLI: `uv run python -m src.neural_analysis.communication_subspace.run_session`
with `--session-metadata`, optional paired `--pfc-probe`/`--hpc-probe`,
`--bin-width-ms {100,500}`, `--seed`, `--rerun`, and `--dry-run`.
Dry-run reports readiness/configuration without loading spike arrays or writing
outputs. Return nonzero for failed/unavailable/partial runs, with the result or
log path when present. Use the same readiness and runner functions as the UI.

## 7. Tests first and work packages

For each package: write tests -> run and record meaningful RED -> commit only
those tests -> implement GREEN -> inspect/refactor -> run appropriate focused
regressions. Missing-module RED is acceptable initially, but numerical oracle
tests must exercise actual mathematics at GREEN. Do not commit unrelated staged
changes. No production scaffold is needed before the first tests.

| Package | Tests committed before implementation | Acceptance |
| --- | --- | --- |
| WP1 numerical model/scoring | Independent ridge/augmented-design RRR oracle; rank bounds including zero cross-covariance; maximal-rank equivalence; inverse scaling; collinear/constant units; negative/undefined R2; nonfinite predictions fail rather than drop targets; ties | Pure functions pass; no files/UI imports; inputs unmodified. |
| WP2 preparation/splits | Good-only selection and missing labels; duplicate cluster numbers across probes; matching/draw reproducibility; all four population identities; common coverage; 40/8 bins and edge spikes; overlapping windows; chronological whole-choice folds; inner/outer purging; insufficient data | Prepared counts and provenance are exact; no shared-loader API changes. |
| WP3 CV composition | Outer-test perturbation cannot change that fold's transform, alpha or fit; same folds across ranks/draws; inner and refit constant/rank changes; full-rank tuning equivalence; complete-fold aggregation and unequal draw-support diagnostics | A deterministic synthetic prepared session yields all four curves with auditable statuses. |
| WP4 results/runner | Round trip/replay; source/code/version invalidation; reuse bypasses fits; no overwrite; interrupted writes ignored; partial/unavailable status; source-free viewing; CLI/dry-run; logging/output-root guard | Callable/CLI runner writes and reopens immutable outputs. |
| WP5 plots/webapp | Paired support markers; two light panels/captions; empty-result display; unset roles accessible; third missing probe irrelevant; disabled Run on missing region; no display-triggered fits; saved setting identity; manual-launch explanation | PNG and AppTest checks pass; existing routes still work. |
| WP6 synthetic acceptance/docs | Seeded low-rank and higher-rank predictive examples; independent-population null; complete synthetic on-disk pipeline at both widths; reopen after raw-source removal | Review actual PNG and webapp display; write usage/limitations; record tests and measured runtime/memory. |
| WP7 designated experimental session | No new scientific tests unless a defect is found; use established checks on one user-designated session | User inspects output before any additional sessions. Not authorized yet. |

Test location: `src/tests/neural_analysis/communication_subspace/` with a local
`conftest.py` for fixtures; `test_modeling.py`, `test_scoring.py`,
`test_preparation.py`, `test_validation.py`, `test_results.py`, `test_run_session.py`,
`test_plotting.py`, `test_webapp.py`, and `test_synthetic_integration.py` as needed.
Keep test files aligned with responsibilities rather than creating one per helper.

Use deterministic analytic fixtures for linear algebra and seeded nonnegative
spike fixtures for integration. Low-/high-rank/null tests use documented broad
tolerances, not exact noisy-rank recovery or universally negative null R2.

Primary commands after implementation is authorized:

```bash
uv run pytest src/tests/neural_analysis/communication_subspace/test_modeling.py src/tests/neural_analysis/communication_subspace/test_scoring.py
uv run pytest src/tests/neural_analysis/communication_subspace
uv run pytest src/tests/neural_analysis/session_cca/test_webapp.py src/tests/neural_analysis/session_cca/test_webapp_navigation.py src/tests/neural_analysis/test_webapp_package.py src/tests/neural_analysis/test_session_timeline_webapp.py
```

Add binning/selection regressions if shared behavior changes. Record actual
RED/GREEN commands, results, and commits. No tests ran during this plan review.

## 8. Orchestrator and worker assignments

Use an **Astra orchestrator**, **Sol scientific/engineering workers**, and a
**Luna presentation/documentation worker**. These are planned assignments, not
claims about agents already running. No workers are launched by this planning
update; implementation approval remains the next gate.

### Ownership and acceptance

| Package | Worker and writable scope | Astra orchestrator responsibility |
| --- | --- | --- |
| WP1 numerical model/scoring | Sol scientific worker: `modeling.py`, `scoring.py`, their tests | Independently verify the penalized-RRR derivation and test oracle before GREEN; inspect rank, scaling, constants, masks and maximal-rank equivalence. Passing worker tests alone is insufficient. |
| WP2 preparation/splits | Sol input worker: `configuration.py`, `preparation.py`, preparation/split tests and initial local `conftest.py` | Approve data contracts, inspect actual loader/API behavior, audit trial identity, half-open bins, overlap handling, seeded matching and train/test purging. |
| WP3 CV composition | Sol scientific worker after WP1/WP2 acceptance: computational portion of `run_session.py`, `test_validation.py` | Trace one fold end to end; verify outer-test perturbation invariance, inner-only tuning, refit transforms, support masks and aggregation. Own cross-module contract changes. |
| WP4 results/runner | Sol input worker after WP3 handoff: `results.py`, side-effecting runner/CLI portion of `run_session.py`, result/runner tests | Approve archive schema and configuration identity; review replay, stale reuse, partial results, source-free loading and no-overwrite behavior. Sole write ownership of `run_session.py` transfers from WP3. |
| WP5a scientific plots | Sol scientific worker: `plotting.py`, `test_plotting.py` | Review actual PNGs and paired-support semantics, baseline markers, negative R2, labels and captions. Plotting includes scientific aggregation semantics, so it is not a Luna task. |
| WP5b webapp controls | Luna UI worker after runner/result/plot APIs are stable: `webapp/communication_subspace_view.py`, `test_webapp.py` | Supply exact callable contracts; review readiness, saved-result identity, no implicit fits and failure display. Astra alone edits shared `webapp/app.py` and `webapp/session_inputs.py` routing. Escalate nonlocal integration problems to Sol. |
| WP6a synthetic acceptance | Sol scientific worker: `test_synthetic_integration.py`, fixture extensions after ownership transfer, measured runtime report | Approve signal/noise construction and tolerances independently; execute final acceptance, inspect browser/PNG results, and decide whether remaining limitations permit a real-session run. |
| WP6b usage documentation | Luna UI worker: new package README and proposed usage text | Verify scientific claims, commands and exclusions; Astra applies the shared `src/neural_analysis/README.md` change. |
| WP7 designated session | Astra owns the run and interpretation; Sol may diagnose a narrowly assigned defect | Confirm session designation, inspect inputs/results, present measured performance and limitations, and wait for user inspection before broader execution. Luna does not interpret scientific results. |

Workers write tests first, demonstrate RED, and hand the exact test paths and
failure evidence to Astra. **Astra owns all Git staging/commits**, including the
required test-only commit before authorizing that package's GREEN work. Workers
must not change tests to accommodate implementation without documenting a genuine
test error or changed requirement and obtaining orchestrator review.

### Dispatch sequence and shared-file discipline

1. Astra freezes the small configuration/input/model contracts and assigns
   exclusive file ownership. WP1 and WP2 may run concurrently after that point.
   WP2 owns `conftest.py`; WP1 uses its own analytic fixtures until handoff.
2. Accept WP1/WP2, then dispatch WP3. Do not overlap WP3 and WP4 edits to
   `run_session.py`; use an explicit handoff even if workers are idle.
3. Once WP3 fixes the result contract, WP4 and WP5a may proceed in parallel.
   WP5a consumes the agreed contract and may not edit `results.py`.
4. After WP4/WP5a acceptance, dispatch WP5b. WP6a may run concurrently with
   bounded UI work if fixtures and source files have separate owners. Luna can
   draft WP6b only against accepted behavior and actual validation evidence.
5. Astra integrates shared routing/docs, runs relevant regressions and final
   synthetic/browser acceptance, then handles the designated-session gate.

Use at most three workers alongside Astra; two workers usually suffice. Idle
slots are preferable to concurrent changes in the same files. Each dispatch
includes the spec/plan, accepted interfaces, exact writable paths, tests-first
gate, dependencies, and completion criteria. Workers report commands, RED/GREEN
evidence, changed files, assumptions, and unresolved concerns. Astra updates the
handoff and explicitly accepts each package; worker completion is not acceptance.

### When to use an Astra worker

**None is required by the current plan.** Sol can implement the specified linear
algebra and validation with independent Astra review. Luna is limited to tasks
whose scientific behavior has already been fixed.

Escalate a bounded scientific question to an Astra worker only if an independent
oracle disagrees with the implementation and the discrepancy remains unresolved,
or observed data force a new statistical/design decision outside the spec.
First let the orchestrator investigate; a second Astra is useful for an
independent derivation or adversarial scientific review, not routine coding.
Report the reason and proposed scope to the user before changing the staffing
plan. Do not respond to ordinary implementation difficulty by automatically
escalating models or broadening the analysis.

## 9. Performance and practical limits

There are 100 outer comparison tasks (5 draws x 5 folds x 4 directions).
Thirteen inner alpha candidates and up to n ranks make independent repeated
fitting expensive. Compute training transforms and cross-products once per
partition/comparison; one factorization/decomposition per alpha supports every
rank. Refit only unique selected alphas on outer training. Accumulate predictions
rank by rank or in bounded arrays; do not allocate
`(all_ranks, all_observations, all_targets)` by default.

Keep two common count tensors, index selected populations when needed, and bin
only once per run. Raw count storage is approximately
`8 * choices * bins * (N_PFC + N_HPC)` bytes before model workspaces. The 100 ms
setting has five times the observations of 500 ms. Report estimated arrays and
actual timings; do not invent an expected runtime before measuring synthetic
data and the designated session.

Use bounded serial execution initially, native optimized linear algebra, and
brief progress updates. Do not introduce rank-level multiprocessing or Numba.
Overlap-safe per-choice binning may be slower; measure it before optimizing.
If runtime is unsuitable for an explicit Streamlit action, report measured
costs and reconsider with the user rather than silently adding workers or
reducing the scientific scope.

## 10. Approval gates

Section 0 is the authoritative live handoff; Section 8 defines team ownership.
Implementation requires approval of this plan/defaults. After synthetic
acceptance, obtain a user-designated session and user inspection of its output
before broader execution. Prior CCA-session authorization does not transfer.
