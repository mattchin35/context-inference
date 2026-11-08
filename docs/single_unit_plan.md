# PC-First Context Encoding Implementation Plan

Status: Implementation approved 2026-11-02. P1-P6 are complete;
P7 awaits a user-designated experimental session and authorization.
No experimental run is authorized by implementation approval.
The execution record below tracks tests-first commits and validation gates.

Scientific authority:
[single_unit_ridge_spec_v1.md](single_unit_ridge_spec_v1.md).
[single_unit_spec_v6.md](single_unit_spec_v6.md) is historical reference, not the
response, estimator, or inference contract for this work.
[../AGENTS.md](../AGENTS.md) and [SoftwareDesign.md](SoftwareDesign.md) govern
approval, readability, data contracts, tests-first commits, and run gates.

### Receiving-chat checklist

1. Read the ridge spec, this plan, AGENTS.md, and SoftwareDesign.md. Use v6 only
   as historical reference; do not restore its trial totals, rank rejection,
   Poisson requirement, or significance workflow.
2. Inspect current HEAD and the full staged/unstaged/untracked state. Planning
   baseline HEAD was `1db6b57`; do not reset to it. Preserve unrelated changes.
   At review, this plan and the ridge spec are **untracked**, so they must be
   retained in the same checkout or explicitly transferred to another checkout.
3. Confirm implementation authorization. When authorized, reproduce the focused
   existing timeline baseline, then start P1 with tests. No new scientific
   clarification remains; new contradictory source evidence must be reported.
4. Use the sequential orchestrator/worker division in section 10. Record each
   completed RED/GREEN gate and commit before advancing. Real-session selection
   and authorization remain a separate P7 gate.

## 1. Deliverable and Agreed Scope

Deliver offline context-to-PC regression and saved-result inspection: pooled
bin observations in before/after/whole choice windows, both context encodings
and timing modes, ridge and both OLS baselines, saved coefficients/predictions,
and a coefficient-orderable activity timeline. The default is [-2,+2) s with
500-ms bins; support existing 100/50/20-ms bins. Trial-start alignment is deferred.

PFC/HPC are independent, use full regional coverage and existing channel/good+mua
selection, and exclude manual-reward/no-choice/dark trials before PCA. Fit all
nonzero PCs on fixed descriptive session axes. Native predictor coding and an
unpenalized intercept are required; ridge defaults to alpha 1.0, with optional
context-only matched shrinkage. Use five-fold block-grouped pooled MSE/R-squared.

Production unit fitting, Poisson, new behavioral features, significance tests,
and local-PCA analyses are subsequent work. Existing NumPy, pandas, SciPy,
scikit-learn, Pynapple, Plotly, Streamlit, and Matplotlib suffice; add no dependency
or general estimator/workflow framework.

User clarification: manual-alignment sessions are deferred. This implementation
requires IRIG-backed recording coverage (`irig_utc_unix`) and reports unsupported
alignment inputs during validation/dry-run; it never infers coverage from spikes.

## 2. Current-Code Audit and Reuse

The following modules and their direct tests were inspected for this draft.
Their existing public behavior and historical saved products must be preserved.

| Existing code | Reuse | Boundary or mismatch |
| --- | --- | --- |
| `session_metadata.py` | Resolve session paths and declared PFC/HPC probe roles. | Version-2 sessions need explicit roles; do not guess probe anatomy. |
| `population/session_timeline_io.py` | Public probe loading, good/mua selection, trusted coverage, and trusted saved timeline loading. | Existing choice preparation does not explicitly filter dark-state trials. Reuse a basis only when its actual pool matches this spec. |
| `population/session_timeline.py` | Choice-window binning, coverage selection, fixed-basis PCA, and result records. | Filter baseline trials first. Existing PCA can include numerical zero-variance components; select nonzero ranks for regression without renumbering axes. |
| `interregional/pca.py` | Established unit standardization and full, unwhitened regional PCA. | Keep a descriptive session basis fixed during regression CV. |
| `interregional/preparation.py` | Reference for ordinary deterministic GroupKFold over `cur_block`. | Its fold helper expects a complete source table. Use the same public sklearn splitter on this analysis's regional eligible pool, not that helper's different eligibility contract. |
| `interregional/linear.py` | Reference for existing conventions and tests. | Its rank rejection, result tables, and fold-average aggregation do not implement this ridge/minimum-norm/pooled-score contract. |
| `population/timeline_plotting.py` | Actual-time bin geometry, explicit gaps, events, and fixed PC identities. | The heatmap helper currently takes contiguous PC ranges; add a small optional explicit-PC-order input. |
| `webapp/timeline_chart.py` and `.js` | Persistent chart and browser range preservation. | JS currently treats every x-axis as time. A categorical coefficient axis needs an explicit exclusion from that range propagation. |
| `webapp/app.py`, `session_inputs.py` | Existing early routing for saved-result views. | New encoding view must load results before any live spike/PCA controls and work without raw spikes being mounted. |

### Why strict PCA-pool compatibility matters

Filtering regression rows cannot remove their prior influence on PCA axes.
The existing timeline may include dark-state choices. Compare actual trial/unit
IDs, bin edges, source settings, scaling, and full-basis availability. Reuse a
matching complete basis; otherwise prepare one compatible basis with existing
functions and save it separately. Display that same basis in the encoding view.
Do not overwrite the original or equate PC12 from different bases.

## 3. Code Organization

Use one shallow scientific package, plus one Streamlit presentation module:

```text
src/neural_analysis/encoding/
    __init__.py            # minimal package marker
    configuration.py       # small run configuration and JSON loading
    predictors.py          # pure context timing/coding and baseline masks
    preparation.py         # source loading, compatible activity/PCA, bin rows
    modeling.py            # multi-response ridge/OLS, CV, scores, scaling
    pipeline.py            # explicit ordering of prepared analysis cells
    results.py             # saved preparation/cells, progress, read-only loading
    plotting.py            # row ordering, coupled Plotly panels, PNG summaries
    run_session.py         # offline single-session CLI
    run_batch.py           # session-list CLI and process workers

src/neural_analysis/webapp/encoding_views.py
src/tests/neural_analysis/encoding/
```

Do not create empty future-unit or Poisson modules. Use functions and ordinary
arrays/tables. A few small data-only records are justified: `EncodingConfig`,
`PreparedRegion`, and `LinearFit`; define them with the module that owns their
contract rather than adding a separate class hierarchy or schema package.

### Responsibilities and proposed public functions

| Module | Main functions and purpose |
| --- | --- |
| `configuration.py` | `load_encoding_config`: resolve input paths and derived output locations; validate supported timing, windows, encodings, bin widths, models, and positive alpha. |
| `predictors.py` | `select_baseline_trials`, `build_context_values`, `encode_context`: operate on complete chronological behavior, retain pre/post validity, and produce native model columns. This is the small future predictor replacement boundary. |
| `preparation.py` | `load_behavior_inputs`, `prepare_regional_activity`, `build_bin_rows`: match original trial identity, reuse compatible saved PCA or existing loaders/binning/PCA, and flatten activity with explicit row metadata. |
| `modeling.py` | `fit_ridge_responses`, `fit_ols_responses`, `predict_responses`, `evaluate_block_cv`, `standardize_coefficients`: pure numeric operations on X/Y and explicit fold IDs. |
| `pipeline.py` | `run_prepared_encoding`: loop over region/timing/window/encoding/model cells, save each completed cell, and skip compatible completed work. No Streamlit or raw source discovery. |
| `results.py` | `save_preparation`, `save_cell`, `load_encoding_run`, `inspect_run_status`: ordinary named-array NPZs, small JSON progress, and basic shape/identity/version checks. |
| `plotting.py` | `order_responses`, `plot_encoding_timeline`, `plot_selected_prediction`, `save_inspection_figures`: consume prepared/saved results, never fit. |
| `run_session.py` | `main` and `run_encoding_session`: assemble concrete dependencies, logs, preparation, analysis, saving, and offline figure generation. |
| `run_batch.py` | `main` and `run_encoding_batch`: explicit session configuration list, dry run, CPU-capped session workers. |
| `encoding_views.py` | `render_encoding_view`: select saved run/configuration and display rows, scale, sorting predictor, viewport, and selected PC. |

Dependency direction: configuration/predictors and existing source utilities;
then preparation/modeling; then pipeline/results; finally CLI and UI. Pure
functions receive the columns/arrays they need. Only source-loading or whole-table
validation functions accept complete tables. Document dtype, shape, axes, units,
return values, and missingness at every function boundary.

### Small changes to existing code

1. Extend `build_timeline_heatmap_data` with an optional explicit original-PC
   selection/order. Preserve its current range-based defaults and all callers.
2. Declare the primary time axis and time-axis list in the new figure metadata.
   Make the chart renderer read/preserve ranges using those declarations.
   Existing figures without it retain their existing behavior. The coefficient
   axis must remain categorical and fixed, not receive elapsed-second bounds.
3. Add one saved-result view label/availability entry and early dispatch in
   `session_inputs.py` / `app.py`. Reuse the current chart renderer and navigation
   helpers; do not import fitting or preparation into the saved-result UI.

Avoid a general timeline rewrite or extracting a new plotting framework. Use
existing public event/time helpers; if a trace helper must be shared, expose one
small helper at its current owner instead of copying the rendering logic.

## 4. Inputs, Preparation, and Array Contracts

### Portable run configuration

Use a small JSON file beside the session metadata, with paths relative to that
configuration file. Provide one editable example under
`docs/examples/neural_analysis/context_encoding_config.json`.

Inputs: neural metadata, the existing augmented-trial CSV, its
`trial_feature_params.json`, optional explicit probe roles, and optionally an
existing timeline result to consider for reuse. Scientific defaults are 500 ms,
PCs, both timing modes, all three fit windows, both encodings, ridge and OLS,
alpha 1.0, and five block folds. The normal penalty mode is `same_numeric`;
`matched_context` is an explicit optional request. The example can request both
for inspection while the UI initially selects `same_numeric`.

Freeze these configuration names in P1; omit a general formula/schema system:

```json
{
  "session_metadata": "neural_session.json",
  "augmented_trials": "processed/<session>_augmented_trials.csv",
  "feature_parameters": "processed/trial_feature_params.json",
  "probe_roles": null,
  "timeline_result": null,
  "bin_width_ms": 500,
  "windows": ["before", "after", "whole"],
  "timing_modes": ["choice_updated", "trial_window"],
  "encodings": ["signed_context", "separate_context"],
  "estimators": ["ridge", "ols"],
  "alpha": 1.0,
  "penalty_modes": ["same_numeric"]
}
```

`probe_roles=null` uses declared version-3 roles; version-2 metadata requires an
explicit PFC/HPC mapping. `separate_context` means the two indicator columns.
Add `matched_context` to the penalty-mode list to request that inspection.
`same_numeric` uses alpha for both encodings; `matched_context` uses alpha for
signed context and alpha/2 for the two indicators.
OLS has no penalty mode and is not recomputed for each requested ridge mode.
Validate unknown fields and unsupported values with concise errors. The
alignment, five folds, all-nonzero-PC rule, and output locations are fixed by
this first-pass contract rather than additional configuration knobs.

Only context is implemented initially. Reject unsupported predictor identifiers
with a useful message rather than silently substituting a feature or pretending
the entire old catalog is implemented.

### Preparation sequence

1. Read the complete chronological augmented table and source behavior/events.
   Match trial IDs and timestamps; preserve source row positions independently
   of DataFrame index labels. IDs need not start at zero. Reject ambiguous joins
   rather than align two files merely by their current filtered row number.
   Source times are UTC seconds; elapsed zero is the first unfiltered trial start.
2. On the complete unfiltered trial sequence, set pre-choice context to
   `state_int[i]` and post-choice context to `state_int[i+1]`. Use the immediate
   next row even if that row is later excluded from regression; never shift the
   already filtered table or skip to a later retained choice. Missing/invalid or
   dark next state makes the post bins unavailable. A manual/no-choice next row
   does not erase a known context state. Terminal post bins are unavailable:
   no logged-event fallback or carry-forward inference in this first version.
   This post-bin unavailability applies to `choice_updated`; `trial_window`
   still uses its valid entering-trial value throughout.
3. Apply baseline trial exclusions before PCA preparation. Intersect each
   region independently with complete recording coverage. Predictor missingness
   must not change this descriptive PCA pool.
4. Reuse an exactly compatible fixed basis, or call the existing binning/PCA
   functions on this pool. Preserve saved PC signs/ranks and unit scales. Select
   nonzero components without refitting or rotating the basis. With M pooled
   observations, U retained units, and saved eigenvalues e, calculate
   s=sqrt(max(e,0)*(M-1)); retain s > max(M,U)*eps(float64)*max(s). An all-zero
   spectrum makes that region unavailable. This is a numerical-rank check,
   not a variance-fraction cutoff; preserve original PC ranks.
5. Build full-window bin rows and their two context-timing variants. Time zero
   is a bin boundary: a post-choice update applies at zero. Window-fixed mode
   repeats the entering context across the whole window. Retain one shared
   full-window activity array per region and slice it for the three fit windows.
6. Per region, run `GroupKFold(n_splits=5, shuffle=False)` once on its baseline-
   valid, full-coverage window rows and recorded `cur_block`, then repeat each
   window's fold ID across its bins. Reuse this map across its encodings,
   estimators, windows, and timing modes; do not search for easier splits after
   further masking. Regions need not share a fold map. Assign post-choice bins
   to their source window's block, not the next context's block. Fewer than five
   usable groups, missing required block IDs, or an empty requested fold makes
   CV unavailable while preserving valid reference fits.

Recorded events are diagnostic and remain at their original timestamps in the
viewer. A context-entry log slightly after a choice is not a conflict with the
intentional choice-time update. Compare state order, not timestamp equality;
report discrepancies for inspection without using events to override the
adjacent-trial values or masks. Conflicting trial identities/states between the
two trial tables are input errors, not an event-based repair opportunity.
Timing modes may have different valid-bin counts; retain those masks/counts and
do not describe their scores as paired on identical rows unless they are.

An empty regional pool, no variable units/PCs, or an empty model cell is marked
unavailable with a reason. Continue other valid regions/cells; do not fabricate
zero results or abort the session for ordinary scientific unavailability.

### Core arrays

| Object | Shape / meaning |
| --- | --- |
| Regional PC scores | `(W, B, K)` float64; fixed unwhitened scores. W is regional windows, B is 8 by default, K is nonzero fitted PC count. |
| Bin-row table | N = W*B rows, retaining original trial ID/row, block, bin position, absolute/elapsed/choice-relative edges, and eligibility reasons. |
| Predictor matrix X | `(N_cell, P)` float64, without an explicit ones column; P=1 signed or P=2 indicators. Intercept is separate. |
| Response matrix Y | `(N_cell, K)` float64 scores; the selected fit window and predictor-valid mask choose N_cell rows from the shared pool. |
| Slopes / intercept | `(P, K)` / `(K,)`; normalize library output orientation once and document it. |
| Fold slopes / intercepts | `(5, P, K)` / `(5, K)`, with fold validity and reasons. |
| Reference / held-out predictions | Each `(N_cell, K)`, with row identity and distinct scope labels. Missing held-out values remain unavailable. |
| Display slopes | `(P, K)` using beta*SD(X)/SD(Y) on reference-fit rows; raw slopes always retained. Undefined response SD gives unavailable display values. |

Store each original PC rank explicitly. Sorting changes an index selection,
not the arrays' scientific identity. Use stable ties and place unavailable
strengths after finite values. An activity value of zero is never missing data.

## 5. Estimators and Evaluation

Use existing scikit-learn rather than implementing a new optimizer:

- `Ridge(alpha=effective_alpha, fit_intercept=True, solver="svd")` on dense
  native X and matrix Y. The small SVD is deterministic, accepts the redundant
  indicator design, and fits all PC responses together.
- `LinearRegression(fit_intercept=True)` for both OLS encodings. Its dense
  centered least-squares path provides the required minimum-norm slope
  convention; do not append a ones column and minimize intercept norm too.
- No predictor scaler, response z-scoring, regularization tuning, positive-slope
  constraint, or automatic predictor removal.

Verified in the planning environment's installed sklearn `_ridge.py` / `_base.py`:
summed SSE plus alpha*L2, dense matrix responses, centered intercept handling,
and the least-squares path. Tests must confirm these contracts in the receiving
runtime. Use the locked existing dependencies; do not edit site-packages.

For each cell: fit reference coefficients once, then fit the five training
folds and fill one out-of-fold prediction per requested bin. Pool held-out
squared errors for MSE and use pooled held-out SST for R-squared. Retain negative
R-squared. Zero SST makes R-squared unavailable while leaving MSE defined.
Require all requested folds/rows for primary scores, preserving reference and
completed-fold results when that requirement fails.

Both OLS encodings must predict equivalently on matched data. Matched-context
ridge must also agree across encodings with the specified penalty mapping;
same-numeric ridge is intentionally allowed to differ. These are useful
scientific correctness checks, not reasons to collapse the requested encodings.

Constant predictors or one training context class are not rank-failure reasons:
use the specified ridge/centered minimum-norm fit and record the limitation.
Do not import v6's positive-residual-DF or constant-response selectivity rules.
For constant Y, coefficients/intercepts and MSE can be defined while standardized
coefficients and R-squared are unavailable. Nonfinite input/output is invalid.

## 6. Saved Progress and Execution

Use ordinary session-local outputs, for example:

```text
<session>/processed/context_encoding/<run_id>/
    preparation.npz
    cells/<cell_id>.npz

<session>/analysis_runs/context_encoding_<date>_<time>_<id>/
    config.json
    progress.json
    run.log
    summary.md
    run_session.py
    run_batch.py
    figures/*.png
```

The preparation stores the fixed basis/scores, source identity, choices/events,
bin rows, predictor values, PC/unit IDs, and block mapping once. A cell stores
reference/fold coefficients, both prediction scopes, scores/counts/status, and
its selected row indices. Do not copy the activity tensor into every cell.
The run index names completed cell files; it need not duplicate their arrays in
a second giant aggregate NPZ.

Keep a preparation identity separate from the model-cell settings: alpha,
estimator, and coefficient display do not change PCA inputs. A later run with
another penalty can reference compatible immutable preparation instead of
binning/refitting PCA again. Store paths relative to the session where practical
and record the referenced preparation identity. This needs only ordinary saved
metadata and file lookup, not a general cache-management layer.

Use named arrays and a dictionary called `meta`, following the existing trusted
project NPZ convention. Include generator, analysis/schema version, config,
code revision/source identity, runtime versions, units/axes, timing, actual
penalties, PCA identity, and eligible counts. Numerical analysis is deterministic;
record that no random draws are used. Synthetic tests use explicit saved seeds.

Use canonical JSON hashes for preparation/model identities, including relevant
settings, source records, analysis version, and scoped numerical-code identity.
Record small input-file hashes and existing large-source path/size/mtime records;
do not hash entire spike recordings for every cell. Hash the small numerical
source files needed by each stage, not unrelated repository edits. Changing
alpha/encoding does not change the PCA identity; a changed numerical algorithm
does. Display settings are excluded from scientific identity.

Save config/input identities before preparation begins. Publish an NPZ through
a temporary sibling before updating the index. On resume,
validate any existing named cell even if an interrupted index update omitted it;
adopt a compatible published file instead of failing on its existence or fitting
it again. Ignore unpublished temporary files. Changed/corrupt published data is
reported, not silently overwritten. Keep execution progress separate from
scientific validity: a completed cell can have valid reference coefficients and
unavailable CV. Such a cell is finished work, not an endlessly retried task.

| CLI | Behavior |
| --- | --- |
| `dry-run --config FILE` | Read metadata/small tables; report input blockers, behavioral counts, and planned cells. Neural coverage/PC counts remain pending unless compatible preparation exists. No spikes, PCA, workers, or writes. |
| `new --config FILE [--rerun]` | Reuse a completed identical run by default. If an identical partial run exists, identify its directory for explicit resume. `--rerun` creates a new timestamped run. |
| `resume --run-directory DIR` | Require matching numerical code; otherwise start a new run. Use saved config/preparation, without reloading changed live features. Before preparation was published, require original input identities to match. Skip complete cells and recompute an unfinished cell. |
| `status --run-directory DIR` | Read progress/metadata without spikes or fitting. |

Preserve old results on input/config/code changes. Outputs must remain inside
the source session and outside the repository; reject an invalid output root.
The webapp loads completed cells from partial runs without requiring live raw
files. Basic shape/identity checks and ordinary files suffice; no locks, PPC
state machine, generic checkpoint engine, or Slurm integration.

Batch execution uses one process per session, defaulting to available CPUs
capped by session count, with an explicit worker override. Reuse the existing
path-list and worker-count helpers where their contracts apply. Limit nested
BLAS work before numerical imports at the CLI boundary. Reject duplicate
canonical sessions in a batch; use one writer per run. Batch CLI is tested on
synthetic sessions; real batch execution remains gated on the accepted single
session and practical RAM/I/O measurements.

## 7. Inspection View and Export

Add a metadata-driven saved-results view named `Context encoding results`.
Build one coupled figure with choices/predictor tracks, a categorical
coefficient strip/matrix beside the regional activity heatmap, and matched row
positions. Every time axis shares elapsed seconds; the coefficient axis does
not. Start with the existing 60-second viewport and ten displayed rows where
available, with access to all fitted PCs. These are display defaults only.

Use consecutive display-row positions with original-PC tick/hover labels (or
an explicitly ordered categorical y-axis). Numeric coordinates [12,2,7] alone
do not guarantee that a heatmap will display PC12/PC2/PC7 in that order. Both
coefficient and activity panels must share the chosen display-row coordinates.

Expose original order versus selected-coefficient magnitude, raw versus
standardized coefficients, region, row count/range, fit configuration, and one
selected response. Sort within region using reference-fit coefficients over
the whole session. Preserve original PC labels, signs, variance percentages,
and compatible scores when panning or reordering. Show actual predictor values
used for the fit separately from the original logged-context event track.

Use actual bin edges, blank inter-window gaps, separate labeled coefficient
and activity color scales, and stable activity limits. Mark the before/after
bins used by the selected fit while showing the full activity window. Keep
missing effects/scores/predictions visibly distinct from zero. PC-SD coefficient
display must not silently standardize the activity heatmap.

Put one compact observed/predicted trace in a detail panel, selecting held-out
or reference predictions with explicit labels. No many-PC curve overlay,
interpolation across gaps, or added smoothing. Read saved predictions only.

Export light-mode PNG coefficient summaries and a bounded example timeline /
selected-response trace with readable fonts and captions. Use Matplotlib for
static summaries so PNG export does not require a new Plotly image dependency.
The main scientific inspection is the interactive saved-result view; exported
figures must identify the same model, PCs, time interval, and score scope.

## 8. Tests Written Before Each Implementation Phase

Every source phase follows RED -> tests-only commit -> GREEN -> refactor.
Confirm failures reflect the requested missing behavior, not faulty fixtures.
Do not stage user/pre-existing files. Commit implementation separately after
the focused and directly affected regression tests pass. Test helpers and
fixtures also document their input/output shape and units.

| Phase | Tests committed first | Implementation / completion gate |
| --- | --- | --- |
| P1: configuration and predictors | `test_configuration.py`, `test_predictors.py`: defaults/paths, invalid settings, native codes, both timing modes, unfiltered adjacency (including an excluded next row with known context), terminal post-bin unavailability, baseline exclusions, event diagnostic lag, and mismatched trial tables. | Pure configuration/predictors; no feature generation or event fallback. |
| P2: preparation and fixed PCA | `test_preparation.py`: 8/40/80/200 bins, time-zero boundary, irregular IDs, independent coverage, strict PCA pool, exact reuse/mismatch rebuild, numerical zero-PC exclusion with original ranks, fixed basis across modes/windows, and regional folds unaffected by excluded/other-region-only groups. | Existing bin/PCA functions composed with the new pool and row map. |
| P3: ridge, both OLS encodings, CV | `test_modeling.py`: analytic ridge objective; intercept offset invariance; minimum-norm OLS/equal opposite slopes and equivalent predictions; matched ridge equality; same-alpha ridge difference; matrix-response equivalence; raw/SD display; one-class/constant predictors; block integrity; one OOF value/bin; unequal-fold pooled scores; negative R-squared; zero SST; unavailable CV keeps reference fits. | Both encodings GREEN using public estimators and ordinary grouped CV. |
| P4: pipeline, persistence, main/batch | `test_results.py`, `test_pipeline.py`, `test_run_session.py`, `test_run_batch.py`: round trip and row/scope identity; interrupted resume skips complete cells; crash after NPZ publish/before index reconciles; completed-but-CV-unavailable cells do not rerun; changed input/code/config creates new identity; default reuse and no overwrite; no raw loading on status/reuse/dry-run/UI load; independent regions; script/log outputs; duplicate-session rejection and worker cap. | Offline resumable coarse cells, CLIs, and ordinary files. |
| P5: plotting and saved UI | `test_plotting.py`, `test_webapp.py`, existing timeline tests: nonmonotonic PC12/PC2/PC7 order checks rendered row coordinates and original labels/hover; coupled row order; raw/SD toggle only changes display/order; gaps/zeros/unavailable; fixed activity scale; prediction-scope labels; no fits/raw loads on UI changes; browser time axes remain numeric and coefficient axis categorical. | Coupled saved view, selected-PC trace, light PNGs; old viewer stays compatible. |
| P6: synthetic end-to-end and measurement | `test_synthetic.py`: seeded nonoverlapping choices with known context changes, two independent neural populations, all requested configurations, terminal/missing cases, full save/load/UI path, known coefficients and expected encoding equivalences. | Inspect synthetic figures, measure preparation/fit/save/load/serialization and memory, and finish documentation. No invented runtime targets. |
| P7: one-session scientific inspection | No experimental run until the user names a session and authorizes this gate. Use the completed synthetic suite and an explicit dry run first. | User inspects PC coefficient ordering, assigned context, predictions, and timing-mode differences. Wider session batches require acceptance and separate authorization. |

## 9. Verification and Performance

Before implementation, reproduce the existing focused timeline/plotting/renderer
baseline and record any unrelated failures without broadening scope. During
each phase run `uv run pytest src/tests/neural_analysis/encoding/<focused_file>`
and directly affected existing suites. For integration run the complete new
encoding test directory plus:

```bash
uv run pytest src/tests/neural_analysis/encoding
uv run pytest src/tests/neural_analysis/test_session_timeline.py src/tests/neural_analysis/test_session_timeline_io.py src/tests/neural_analysis/test_session_timeline_scripts.py
uv run pytest src/tests/neural_analysis/test_timeline_plotting.py src/tests/neural_analysis/test_timeline_chart.py src/tests/neural_analysis/test_session_timeline_webapp.py
```

Add relevant metadata/webapp routing and full neural-suite regression once the
integration is GREEN, not repeatedly after unchanged checks. Verify actual
browser behavior on the synthetic result: pan, reorder, toggle coefficient
scale, change the selected response, and confirm time/row alignment and a
categorical coefficient axis. App tests alone do not prove browser pan state.

The initial designs have only one or two predictor columns. Fit all PC responses
in one call per cell/fold, prepare each regional PCA once, reuse fold assignment,
and avoid recomputing identical signed-context fits just to label a matched
comparison. This is ordinary library reuse, not a custom optimized solver.

Process and save one cell at a time. Keep the shared activity once; do not retain
every cell's large prediction arrays simultaneously or save training predictions
for every fold. Measure compressed output size as well as RAM. At 500 ms there
are 8 bins per window; 20 ms increases row count 25-fold, so keep bounded-view
loading available. Do not build a dense session-time grid full of gap values.

Report session/window/unit/PC counts, bin width, preparation/PCA time, aggregate
fit time, save/load time, plot serialization size/time, and peak process memory.
Do not promise responsiveness or request cluster infrastructure without a
measured reason. Profiling-driven major optimization requires a later decision.

## 10. Orchestrator and Worker Assignments

Use **one orchestrator and at most one active coding worker at a time** through
the sequential phases in section 8. Reuse worker contexts; no subworkers or
per-file agents. Luna is optional, and the orchestrator can do small routine
tasks directly. These agent limits do not change session-process parallelism.
This is a delegation plan; no workers have been started.

| Role | Model / reasoning | Responsibility |
| --- | --- | --- |
| Orchestrator | `gpt-6.1-sol`, `high` | Own scope, architecture, independent review, integration, commits, and user communication. Lead P6's synthetic/browser audit and the separately authorized P7 run. |
| Scientific worker | `gpt-6.1-sol`, `high` | Own P1-P3; P4 pipeline/results and scientific persistence tests; P5 scaling/order, plotting, original-PC support, and time-axis renderer correction. |
| Optional routine worker | `gpt-6-luna`, `medium` | After scientific interfaces pass review, own P4 CLI wrappers/example config or P5 mechanical Streamlit controls/routing. Documentation can use `low` for literal edits. |

Keep timing, eligibility, units/axes, model mathematics, evaluation, and plotted
data alignment with Sol, even for short diffs. Return an expanding Luna task to
Sol; do not downgrade science work if Sol is unavailable. Reserve `xhigh` for
a concrete unresolved scientific/integration problem. Check model availability
when dispatching. These starting recommendations follow OpenAI's
[model-selection guidance](https://developers.openai.com/api/docs/guides/model-selection)
and
[reasoning-effort guidance](https://developers.openai.com/api/docs/guides/deployment-checklist).
They are not measured performance claims; reproduced evidence determines
acceptance.

For each assignment:

1. Orchestrator specifies owned files, contracts/spec sections, tests, and scope.
   Only one editor touches a shared file; workers do not alter Git state or
   dependencies.
2. Worker writes tests, reports a meaningful RED command/output and test diff,
   then pauses. Orchestrator checks the oracle, reproduces RED, and commits
   **only those tests**, preserving unrelated staging.
3. Worker implements, runs focused GREEN/regressions, and reports files/results
   and unresolved issues. Orchestrator inspects the diff, independently checks
   section 8's invariants, reruns checks, and alone commits implementation.
4. Record the gate before assigning dependent work. On interruption, inspect
   files and reproduce the last gate; do not reset or restart completed work.

If the orchestrator implements a substantive scientific change, use the existing
Sol worker for focused review. No permanent reviewer team or agent framework.

## 11. Review and Execution Record

2026-11-02 review: clarified adjacent-trial/terminal context, regional fold
construction, constant-design handling, immutable resume, and actual display-row
geometry; synchronized the active spec and removed duplicate delegation text.
No scientific decision remains open. Documentation only; no tests or analysis
were run. The unrelated staged/modified timeline spec, `src/main.py`, and
`src/neural_analysis/sync_ephys.py` were preserved.

Implementation approval: granted by the user in this chat on 2026-11-02.
P1-P7 execution records follow below. After each gate, append
short dated records with files, RED/GREEN commands, tests/implementation commits,
outcomes, performance measurements where relevant, and next gate. The real
session and authorization remain a separate P7 decision.

2026-11-02 implementation start: user approved proceeding through the plan;
manual-alignment sessions are deferred. Initial timeline baseline command:
`UV_CACHE_DIR="$PWD/.pytest_cache/uv" uv run --frozen --no-sync pytest`
with the six section-9 existing timeline/IO/scripts/plotting/chart/webapp files.
Result: 67 passed, 1 skipped (Node not on PATH); the bundled Node runtime is
available for the renderer check. The local UV cache avoids the read-only
default cache; no dependencies were changed. Existing staging is preserved.

P1 RED: new configuration/predictor tests independently reproduced two expected
missing-package import errors. Tests-only commit `424fd5c`; GREEN authorized.
The bundled-Node renderer baseline passed both tests (no remaining baseline skips).

P1 GREEN: implementation commit `e895288` adds configuration and pure predictors.
87 focused tests passed; independent P1 plus metadata run passed 120 tests.
Worker's directly related parser/spike-behavior/metadata run passed 70 tests.
Existing `test_project_utils.py` could not collect because it unconditionally
imports unavailable `autograd`; no dependency or unrelated test was changed.
P2 tests-first work authorized after source review.

P2 RED: expected missing-preparation import error independently reproduced.
Tests-only commit `949d7e2` adds a seeded two-probe fixture and preparation
contracts. Implementation is authorized; regional scientific unavailability
must retain valid activity/reference-fit opportunities in other cells.

P2 GREEN: independently passed 149 tests across all encoding tests and existing
timeline, IO, scripts, and interregional PCA. Genuine test-oracle corrections
(`93366e7`) fixed the existing loader's canonical channel-quality filename and
the omitted-HPC expectation in a PFC-only run. Scientific assertions remain.
Implementation/example commit `eed7c1d` adds fixed regional preparation, IRIG
coverage enforcement, original-PC numerical rank selection, and row/fold maps.
The example JSON selections were verified against the frozen P1 constants.
P3 tests-first work authorized after source review.

P3 RED: expected missing-modeling import error independently reproduced.
Tests-only commit `f19d476` specifies analytic ridge/minimum-norm OLS,
matrix-response fits, matched/native penalty comparisons, per-response constant
handling, complete pooled five-fold scoring, partial-result preservation, and
encoding/penalty propagation through every fold. GREEN implementation authorized.

P3 review found missing fold-level R-squared reasons and unassigned (-1) rows
entering training pools. Added regression tests, independently reproduced five
failures, and committed tests first (`953f5c2`). Source fixes retain unassigned
rows in reference fits only and separate metric reasons from valid finite fits.
P3 GREEN: 43 modeling tests; independently passed 199 encoding/direct-regression
tests. Implementation commit `1e7b513`. The installed Ridge single-response
coefficient squeeze is normalized explicitly to (predictor,response).
P4 persistence/pipeline/CLI tests-first work authorized after source review.

P4 split ownership: the existing scientific worker owns results/pipeline and
their tests; the orchestrator owns the routine session/batch command wrappers
and their tests. No shared source file has multiple editors. CLI tests-only
commit `2cd16fa` and persistence/pipeline tests-only commit `682fd94` each followed
independently reproduced missing-module RED. GREEN source work is in progress.

P4 review RED: CLI tests `a8ad96b` reproduced duplicate-alias dry-run counts
(72 rather than 60 unique cells) and missing explicit worker spawn context.
Fixes passed all 17 command tests, including actual two-session synthetic batch
execution. Standard-library multiprocessing APIs were checked in installed source.
Persistence tests `085bd5c` independently reproduced 18 failures for runtime
identity, missing/corrupt published preparation, metadata axes, nonfinite CV
claims, duplicate completed-cell reads, and reference-cell failure handling.
Those source corrections are in progress under the tests-first gate.

P4 logging RED (`3591613`) required per-cell settings/counts/status/errors and
elapsed seconds. P4 GREEN: independently passed 257 encoding/direct-regression
tests without warnings. Implementation commit `7374f4a` adds ordinary NPZ/JSON
products, aliases, immutable preparation, crash reconciliation, runtime/code
validation, read-only status/loading, offline session/batch commands, and README.
The scientific worker reviewed the orchestrator's CLI integration. All unrelated
staging and source modifications remain untouched. No experimental data ran.
P5 tests-first plotting/saved-UI work authorized after source review.

P5 RED: UI tests `b63a529` first failed on the missing saved view; plotting/export
tests `ca4491f` independently failed on missing plots/exports, explicit PC order
and categorical-axis time propagation (7 old-suite failures). Review tests
`dfe9074` reproduced saved-X display, reference-export scope, bounded loading
and initial-penalty selection gaps before source fixes. P5 GREEN: 318 combined
tests including Node passed; independent focused run passed 71. Both orchestrator
and worker visually inspected PNGs for aligned time columns, readable unclipped
captions, native signs/scales and original PC labels. Implementation `a131991`
adds the coupled figure, saved-only view/routing/navigation, original-PC helper,
time-axis renderer metadata, and scope-aware light PNG exports. The old viewer's
public defaults remain compatible. Scientific outputs stayed session-local.
P6 seeded end-to-end validation, browser audit and measurement work is authorized.

### 2026-11-02: P6 synthetic validation and delivery

P6 tests-only commit `6c41d77` adds a seeded two-region, 32-unfiltered-trial
end-to-end fixture. Three scientific checks passed against independent
closed-form context-group regression calculations, including all 60 distinct
cells and five folds, encoding/penalty equivalences, original PC12 inspection,
saved-only UI use, and PNG exports. Two measurement checks first failed for the
missing saved-run benchmark/report. The completed synthetic suite passed all
five checks. Source and documentation commit: `d435b42`.

Independent integration command (existing frozen UV environment, bundled Node):
`pytest src/tests/neural_analysis/encoding` plus the six timeline test files
listed in section 9 passed 312 tests in 27.84 seconds. Browser review added
tests-first corrections: shared time-axis pixel domains (`b8513d7` tests,
`9c3dfa4` source), repeated LED legend entries and coefficient-scale placement
(`e627268`, `b52fb99` tests; `d435b42` source). The final focused plotting run
passed 13 tests. Browser inspection verified real mouse panning, identical time
alignment, persistent pan across response reordering and coefficient-scale
changes, matching categorical coefficient/activity rows, original PC12 trace
selection, fixed neural color scale, visible gaps and native-predictor hover.

The full neural suite passed 3,415 tests and failed 12 (50 warnings). Ten LFP
multiprocessing failures followed earlier tests replacing the process main
module with a temporary renderer script (`__args` NameError in spawned children).
Both affected LFP files passed all 240 tests in a fresh `python -m pytest`
process. The remaining two failures concern unchanged existing contracts:
the interregional SLURM job-name expectation and the sync workflow's absent
`sample_rate_hz` keyword. No unrelated source or tests were changed to hide them.

Measured synthetic run: seed 20261103, 500-ms bins, PFC 28 windows and HPC 26
windows, eight bins/window, 17 input/16 retained units and 16 fitted PCs per
region. One observed run recorded preparation 0.0663 s, aggregate fitting
0.1266 s, preparation publication 0.0063 s, cell publication 0.0838 s, saved
loading 0.0141 s, representative figure construction 0.0951 s, and serialization
0.0095 s / 194,526 UTF-8 bytes. Compressed preparation plus cells occupied
757,047 bytes. Peak process RSS was 611,094,528 bytes, including the surrounding
pytest process/imports/prior work; it is not incremental analysis memory.
Filesystem cache state was unspecified. These small synthetic measurements
do not establish real-session performance. Each executed new/resumed run
writes `performance.json`; saved-only `benchmark_encoding_run` can repeat the
load/plot measurements without fitting. Independent worker review found no
measurement correctness blocker.

Temporary synthetic browser data (not experimental output):
`/tmp/context_encoding_validation_20261103_0113/context_encoding_scientific0/session`.
P7 remains gated on the user selecting and authorizing one experimental session.
Manual-alignment sessions remain deferred. Pre-existing staged, modified and
untracked files remain preserved; this pre-existing untracked plan/spec are not
included in implementation commits.
