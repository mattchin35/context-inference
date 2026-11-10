# Cross-Condition Generalization: Implementation Plan and Handoff

Status: revised implementation draft aligned with spec revision 9, 2026-11-10.
Includes Sol orchestrator / sequential Sol and Luna worker assignments.
Implementation authorized by the user on 2026-11-10; P1-P4 complete, P5a GREEN in progress.

Read the [scientific spec](cross_condition_generalization_spec_v1.md),
[SoftwareDesign.md](SoftwareDesign.md), and repository `AGENTS.md`.
The spec owns scientific meaning; this plan owns file scope, contracts,
test-first delivery, and evidence. The
[session-prediction implementation](condition_prediction_generalization_implementation_plan.md)
is a completed reference, not work to resume.

## 0. Live handoff

At RED, test-commit, GREEN/review and stop boundaries, update this snapshot and
ledger with exact commands/results, commits and any live process/log IDs.

| Field | Current snapshot |
| --- | --- |
| Branch / inspected HEAD | `refactor` / `03ffd66`; recheck on resumption. |
| Authorization | User approved implementation on 2026-11-10. Synthetic validation is included; experimental preparation remains separately gated. |
| Existing tracked edits | `src/main.py`, `src/neural_analysis/sync_ephys.py`; preserve both and unrelated untracked files. |
| Feature documents | Approved spec/plan and P1 evidence tracked in `5e1d49e`; subsequent ledger edits belong to the orchestrator. |
| Running work / owner | `/root/scientific_worker`: P5a GREEN single-session runner; no experimental processes/jobs. |
| Implementation team | Orchestrator owns review/commits/handoff. Scientific worker `gpt-6.1-sol`, high; at most one worker active. |
| Confirmed inputs | Version-3 metadata only; ignore blocks; normalize numeric masks locally; validate context after baseline exclusions and exclude dark state 2. |
| Reuse / availability | Reuse confidently compatible PCA. Report each unavailable region's reason without crashing, including when both are unavailable. |
| Confirmed preparation | One call prepares all sixteen transfers, three windows and two weightings at one bin width. Explicit Prepare/Recompute controls in the summary view. |
| Next action | Review P5a GREEN, then dispatch sequential Luna P5b RED. |

| Package | Assigned writer / owner | State | RED / test commit | GREEN / review / implementation commit |
| --- | --- | --- | --- | --- |
| P0 spec/plan | Sol orchestrator | Reviewed draft | Not applicable | Documentation checks only |
| P1 public fit helper | Sol worker | Complete | RED: 2 expected missing-public-helper failures; 173 passed. Tests: `4e1b478`. | 175 passed; implementation `c08967a`; numerical body unchanged. |
| P2 controls/cells/alignment | Sol worker | Complete | 2 expected absent-package collection errors; tests `1150ea6`. | Combined 319 passed; post-refactor P2 89 passed; implementation `aa70f22`. |
| P3 fitting/scoring | Sol worker | Complete | 2 expected absent-modeling collection errors; tests `fc80d35`. | 178 passed; implementation `3ef2862`; all 192 oracle fits and 1920 summary rows checked. |
| P4 saved products | Sol worker | Complete | Initial RED `256f0e2`; review-regression RED `f31446e` (4 failed, 3 passed). | Final gate 243 passed; implementation commit recorded below. |
| P5a single-session runner | Sol worker | GREEN in progress | Expected missing-runner import error; tests `03ffd66`. | Joint script-copy gate awaits P5b. |
| P5b batch wrapper | Luna worker | Not started | Record here | Record here |
| P6 summary webapp | Sol worker | Not started | Record here | Record here |
| P7a synthetic integration/browser | Sol orchestrator | Not started | Prior package tests | Record here |
| P7b usage documentation | Luna worker | Not started | Not applicable | Document checks / review |
| P7c final acceptance/handoff | Sol orchestrator | Not started | Prior package tests | Record here |
| P8 experimental acceptance | Sol orchestrator + user | Separately gated | Not applicable | Record only after authorization |

Before resuming, compare HEAD/owned-file diffs and inspect any recorded live
command. Follow `AGENTS.md` for unexpected edits; preserve partial work.

### 0.1 Orchestrator and worker protocol

- **Sol orchestrator** (`gpt-6.1-sol`, high): owns sequencing, scientific
  contracts, acceptance cases, diff/test review, commits and this handoff.
  Reviews all target/cell definitions, exclusions, weights, score semantics,
  PCA reuse and saved identities, including work delegated to Sol.
- **Sol workers** (`gpt-6.1-sol`, high): implement scientific packages and their
  tests in the ledger order. This includes persistence and UI selectors because
  incorrect axes, source identities or displayed roles can change interpretation.
- **Luna workers** (`gpt-6-luna`, medium): implement the thin batch wrapper and
  usage docs after the relevant Sol contracts pass review. They consume existing
  APIs and approved scientific wording; scientific questions return to Sol.
- At most **one active worker in addition to the orchestrator**. Reuse workers
  sequentially where practical; idle/stop the current worker before dispatching
  another. No worker-created agents, concurrent worker assignments or extra
  review agent. These agent limits do not change session-level analysis workers.
- One writer per assigned file. The orchestrator does not edit worker-owned
  files during an assignment; review/commit follows the worker's return to idle.
  Use the shared checkout and path-limited commits; no extra worktree/locking
  framework. Record actual agent/model/effort, phase and owned files in the ledger;
  a role label does not change the current chat's model.

Execution order after implementation approval:
`P1 -> P2 -> P3 -> P4 -> P5a -> P5b -> P6 -> P7a -> P7b -> P7c`.
P8 remains separately authorized. Sol reviews each gate before the next starts.
P5a/P5b are sequential assignments within one runner gate: the session runner's
batch-script snapshot and combined GREEN checks finish after P5b.

For every code assignment:

1. Sol dispatches exact files, input/output contracts, acceptance tests/command,
   current HEAD, user edits to preserve, and a **RED-only** task. Worker writes
   tests, demonstrates the intended failure and returns evidence, then idles.
2. Sol verifies RED and commits only the owned tests. Only then does Sol resume
   that worker for GREEN within the same approved scope.
3. Worker implements, runs the assigned checks and reports changed files,
   commands/results and unresolved issues, then idles. Sol reviews science and
   diffs, verifies the package gate, commits owned implementation and updates
   the ledger. Corrections return to the same worker before advancing.

Worker handoffs include the active phase, test/implementation commits, live
process/log IDs and next action. Documentation-only assignments use document
checks and Sol review. Worker reports do not authorize experiments or replace
user scientific acceptance. No workers or implementation are started by this plan edit.

## 1. Scope and architecture

Implement the four targets and sixteen directed splits in spec Sections 3-6.
Use fixed session-wide PFC/HPC PCs, all source-cell trials, complementary-cell
tests, and Full/Before/After matched training/scoring windows. Both equal-trial
and equal-cell fits are saved together: at most
`16 * 3 * 2 * 2 = 192` fits per bin width.
The PCA retains its existing complete `[-2,+2)` source windows; only the
supervised fitting/scoring bins are selected for this analysis.

Correct outcomes remain combined for fitting and separate only in summaries.
Use version-3 metadata and ignore blocks entirely. Full supervised windows do
not overlap in this task; add no overlap handling. Scope excludes training
fractions, continuous targets, tuning, full-window decoder traces, timeline
integration, contribution heatmaps and new statistics. Existing analyses retain
their numerical behavior and cluster resources remain unchanged.

Use functions and small records with explicit array contracts; separate
preparation, computation, persistence and presentation. Existing NumPy, pandas,
SciPy, scikit-learn, Streamlit and PCA/Pynapple dependencies suffice.

### 1.1 Reuse actual existing functionality

| Existing owner | Reuse / limitation |
| --- | --- |
| `task_decoding/config.py`, `targets.py` | Existing four target definitions/encodings; add a default-off `ignore_blocks` opt-in to target construction/validation. |
| `task_decoding/conditions.py` | Canonical masks on a locally normalized numeric table. Actual reward uses its recorded flag; canonical `rewarded` means correct-rewarded. |
| `session_prediction/preparation.py:training_bin_mask`, `session_prediction/config.py` | Exact supervised-window masks/bounds. Do not call its fold-assignment preparation or mode-specific config loader. |
| `pooled/decoding.py:stack_training_window` | Trial-major/bin-major stacking and 1/B row weights; multiply by each trial's total weight. |
| `session_prediction/modeling.py:_fit_one_model` | Existing fixed logistic/ridge fitting. Expose this function as `fit_readout`, without moving or changing its numerical implementation. |
| `session_prediction/scoring.py:trial_metrics` | Primitive trial accuracy/balanced accuracy; its full report builder is fold-specific and is not reused. |
| `session_prediction/results.py:file_identity` | Portable relative-path/size/SHA-256 provenance for contained files. |
| `population/run_session_timeline.py`, `session_timeline_io.py` | Existing exact/registered lookup, regional compatibility checks, missing-PCA preparation and loading. Regional compatibility lookup reads scores; reserve it for explicit preparation. |
| `webapp/condition_generalization_views.py`, `session_inputs.py`, `app.py` | Summary conventions and early metadata dispatch; the existing pooled view stays separate. |

Retain `load_task_decoding_config` for input location/target contracts; locally
replace its targets/alignment with the four requested targets and `choice_time`.
Its feature-parameter JSON and both regional config prerequisites remain.
Document these prerequisites rather than refactoring the config loader.

`allow_missing_blocks=True` still checks block contiguity and is insufficient.
Add keyword-only `ignore_blocks=False` to `build_target_table` and
`validate_augmented_trials`. True removes `cur_block` from required columns,
skips block validation and returns missing `block_id` values without reading
source block contents. The new analysis opts in; existing callers keep their
defaults. No synthetic block labels or fold assignments are needed.

### 1.2 File boundaries

Create `src/neural_analysis/task_decoding/cross_condition/`:

| Module | Responsibility / public entry points |
| --- | --- |
| `config.py` | Fixed target/split registry, small controls record, `load_cross_condition_config`. |
| `preparation.py` | `TransferInputs`, `prepare_transfer_inputs`; aligned labels, cell masks and conflict audit; no I/O. |
| `modeling.py` | `RegionTransferResult`, `training_trial_weights`, `fit_region_transfers`; array-only computations. |
| `scoring.py` | `summarize_transfers`; small role/outcome records using `trial_metrics`. |
| `results.py` | `CrossConditionResult`, identity, save/load, discovery and exact-identity lookup. |
| `run_session.py` | `prepare_cross_condition_generalization`, CLI and run documents. |
| `run_batch.py` | Thin `run_transfer_batch` session-level wrapper. |
| `__init__.py` | Package purpose; no import-time work. |

Add `webapp/cross_condition_view.py` with
`render_cross_condition_view(st_module, session) -> None`. Tables/captions
satisfy the summary request.

Outside these files, only:

- Rename the existing private fitting helper and its internal call in
  `session_prediction/modeling.py`; document positive row weights and generic
  class-1 labels. Keep numerical behavior; existing provenance includes this file.
- Add the narrow `ignore_blocks` opt-in in `task_decoding/targets.py`, including
  required-column selection and unused `block_id` output as described above.
- Add the new view's constant, availability and early dispatch in
  `webapp/session_inputs.py` / `webapp/app.py`.
- Add usage documentation and README links, preserving current instructions.

No new estimator/reuse framework, metadata migration, backward-compatibility
path, scheduler or automatic session-setup job is needed.

## 2. Preparation and data contract

Axes: J original behavioral rows; N saved ordinary choices; T=4 targets in
order `current_state`, `current_action`, `current_reward`,
`current_action_is_correct`; S=16 splits; H=3 windows (Full/Before/After);
G=2 weightings (equal-trial/equal-cell). Regional scores have shape `(W,B,K)`,
with ordered integer `choice_positions(W,)` mapping into N.
Targets/probabilities/weights are float64, masks Boolean, and maps/counts integer.

Times are choice-relative or UTC Unix seconds. Scores retain unwhitened
standardized-unit PC units. Targets/probabilities/weights are dimensionless;
coefficients are log-odds per PC-score unit and intercepts log-odds. Document
types, shapes, units and returns for every function.

`prepare_transfer_inputs(trial_df, choices, bin_edges_s, base_config)`
takes chronological augmented trials, a choice DataFrame and a common second
grid `(B+1,)`; it does not need a whole PCA/session object:

1. Preserve all chronological rows. Establish ordinary-choice membership with
   existing manual-reward/no-choice/alignment exclusions. Work on a local copy;
   context outside that membership is not validated (use missing context on the
   loader copy). Build the four targets with `ignore_blocks=True`, retaining
   existing validation of other labels. Among ordinary choices, exclude dark
   state 2 from all targets and report its count; remaining context must be 0/1.
2. Normalize the relevant state/action/correct/reward/manual-flag columns on the
   local mask-building copy. Numeric strings must match numeric labels; preserve
   project missing/no-choice semantics and reject malformed present values
   rather than silently coercing them to missing. The original table is unchanged.
3. Match augmented chronology to supplied choices using original row position,
   trial ID, action and choice time (absolute 1-us tolerance, relative zero).
   Preserve row identities and reject mismatches.
4. Reuse canonical masks on the normalized copy and apply baseline/dark filters.
   For recorded binary state/action/correctness, conflict is
   `correct != (state == action)`. Exclude conflicts from this analysis and
   retain count, original IDs/positions, recorded values and reason. Do not
   edit the behavioral table or refit PCA.
5. Build the spec's cells explicitly: Context uses context/correctness;
   Choice and Value use action/correctness; Reward uses action/actual reward.
   Correct combines correct-rewarded/omission; actual unrewarded combines
   omissions/incorrect-unrewarded. Reward and Value label omissions 0 and 1,
   respectively. Other target validity follows the existing loader.
6. Use sixteen simple `TransferDefinition` records with stable IDs, target,
   source/test cell names and labels. Validate disjoint source/test cells and
   opposite classes within each pair; no expression parser.
7. Reuse exact Full/Before/After bin masks. At 500 ms counts are 4/2/2, at
   100 ms 20/10/10; zero belongs to After. No resampling or PCA changes.

`TransferInputs` holds targets/validity `(N,T)`, baseline/conflict masks
`(N,)`, canonical conditions `(N,C)`, separate source/test cell arrays
`(N,S,2)` (last axis actual class 0/1), window masks `(H,B)`, edges, choices,
dark-state count, conflict records and target/split metadata.
Regional scores remain separate. Reuse this preparation for dry-run and real
execution: current behavioral choices supply dry-run inputs; actual saved PCA
choices supply the real alignment check. The existing
`prepare_timeline_choices` builds choice coordinates without loading events.

## 3. Fitting and scoring

### 3.1 Expose the existing fit helper

Rename `_fit_one_model(matrix, labels, weights, family)` to `fit_readout` and
update its sole caller. New fitting uses `family="categorical"`, finite `X(M,K)`,
binary `y(M,)` and positive row weights `(M,)`. Return coefficient `(K,)`, scalar
intercept and reason; unavailable fits retain None/NaN/reason conventions.

Keep the current logistic settings exactly: C=1, L2/l1_ratio=0, lbfgs,
unpenalized intercept, no class weights, tol=1e-4, max_iter=5000.
Keep ridge/failure behavior unchanged. Verify the installed API/source before editing.

### 3.2 Regional computations

`fit_region_transfers(scores, choice_positions, inputs)` fits all H/G/S
combinations on original K PCs:

1. Intersect cells with baseline, required labels, conflict exclusion and
   regional coverage. Do not filter by blocks.
2. After those intersections, let source counts be n0/n1 and Ntrain their sum.
   Equal-trial total weight is 1; equal-cell is `Ntrain/(2*nk)` per trial.
   Divide each trial's weight across B_selected bins via the stacking helper.
   Both schemes sum to Ntrain when both source cells exist; do not double
   balance with class weights or balance outcomes inside a combined cell.
3. Fit one selected training design at a time. Evaluate source/test trials
   only within the selected bins, apply `expit` before averaging, and retain
   one mean probability per trial. No outputs outside the interval.
4. Missing source classes or existing numerical/convergence failures make
   only that model unavailable. Record the reason and NaN parameters/outputs.
   A nonfinite affine/output result follows the same convention. Do not
   borrow test trials, clip invalid outputs or change another model's fit.

Return regional `choice_positions(W,)`, coefficients `(H,G,S,K)`,
intercept/status/reason `(H,G,S)`,
trial probabilities `(H,G,S,W)`, exact source/test masks `(S,W)`,
source trial weights `(G,S,W)`, eligible cell counts
`(S,2,2)` (source/test role, then actual class 0/1), and original one-based
PC IDs `(K,)`.
Frozen field names: `choice_positions`, `pc_ids`, `coefficients`, `intercepts`,
`fit_status`, `fit_reason`, `trial_probabilities`, `source_mask`, `test_mask`,
`source_trial_weights`, `eligible_cell_counts`. `training_trial_weights` takes
bool source cells `(W,2)` and one weighting ID, returning float64 totals `(W,)`.
Weights are zero off source; probabilities are NaN off eligible source/test
or for unavailable fits. Complete PCA coverage makes membership identical
across supervised windows. No fold axis or neural-score copies are saved.

For an empty source class, equal-cell weights are zero with an unavailable
reason; equal-trial weights still describe available source trials. Weight
normalization checks apply to valid two-cell pools, not empty ones. A later
numerical failure does not erase eligibility counts or valid trial weights.
Represent an unavailable region by its reason, without calling the fit function
or fabricating a PC axis. Save/report either or both regions' unavailability
without crashing.

### 3.3 Summary records

`summarize_transfers(inputs, regions, unavailable_regions)` returns target,
split, window, weighting, region, role, population, counts, accuracy, balanced accuracy and
availability reason. Roles are `training` and `complementary_test`.
Intersect every summary with its exact role; never pool source and test.

Population lists:

- Context/Choice/Value: all eligible, correct combined, correct-rewarded,
  correct-omission, incorrect.
- Reward: all eligible, actual rewarded, actual unrewarded combined,
  correct-omission, incorrect-unrewarded. Use reward=1 for actual rewarded,
  not the canonical `rewarded` alias.

Pass only finite trial means to `trial_metrics`; use equal evaluation weight
per trial regardless of fitting scheme. Report eligible count separately from
scored `n_trials` and actual class-0/class-1 counts. Unavailable models have zero
scored trials and unavailable metrics with a fit reason; retain eligible
class/cell counts in fit details.

Use mean P(class 1) >=0.5. Empty metrics stay unavailable; single-class rows have
accuracy but no balanced accuracy. Preserve below-chance results. Do not
aggregate directions or add statistics for overlapping populations.
`unavailable_regions` maps region names to reasons; show those regions with
unavailable metrics rather than omitting them or inventing coverage counts.
Frozen row keys: `target_name`, `split_id`, `window`, `weighting`, `region`,
`role`, `population`, `eligible_trials`, `n_trials`, `class_0_trials`,
`class_1_trials`, `accuracy`, `balanced_accuracy`, `reason`. Class counts are
scored counts. `single_class` explains unavailable balanced accuracy without
suppressing ordinary accuracy; unknown regional coverage has `eligible_trials=None`.

## 4. Saved products and provenance

Publish a new `cross-condition-generalization-v1` / schema-1 bundle containing
all S/H/G combinations at one bin width: named NPZ arrays, dictionary-valued
`meta` and discovery `config.json`. Include preparation/regional arrays,
labels/cells/roles, grids/bins, weights, exclusions, units, generator and
analysis/library/source/code provenance. Record runtime, peak RSS, Git
revision/status and `randomness_used=False`, `random_seed=None`.

`CrossConditionResult` holds prepared inputs, regional results, unavailable-region
reasons, source references and metadata; it contains no neural-score copies.
Frozen constructor: `(inputs, regions, unavailable_regions, meta)`.
`build_transfer_metadata(inputs, session_root, source_pcas, target_sources,
effective_config, *, session_id)` reads only small source-PCA metadata plus file
identities. `meta["source_pcas"]` keys distinct relative archive paths;
`meta["region_sources"]` maps regions to those paths. Raw metadata provenance is
separate from fitted identity. `save_transfer_result(path, result)` and
`load_transfer_result(path)` publish/load the self-contained packet;
`discover_transfer_runs(session_root)` and `find_saved_transfer(session_root, identity)`
support selection and exact reuse.

Fitting identity covers regional PCA hashes/configuration IDs, augmented/base inputs,
roles/bin width, split/target/window/weight/conflict/source-test definitions and
actual computational dependencies. Store a region-to-source map supporting
shared or separate archives. Reuse `file_identity`; hash each distinct source
once. Exclude machine prefixes, timestamps and presentation/metric code from
fitting identity. Whole-file hashes suffice; no column-level hash system.

Validate normal saved-data contracts: versions, labels, axes, grid, masks/maps,
role disjointness, counts/weights, parameter availability and probability range.
Failed models have NaN parameters/probabilities, with reasons and retained
eligibility counts. Match manifest/archive identity; ignore incomplete runs
during discovery and explain invalid selected runs. Use the existing pattern
of temporary NPZ plus atomic rename as final publication; never overwrite.

Load/score using only the saved run; do not open raw files, base config or PCA.
Cache summaries by result stats/scoring version. Numerical reconstruction
against PCA belongs in computation tests, not publication or viewing.

## 5. Single-session and batch entry points

`prepare_cross_condition_generalization(metadata_path, *, config_path=None,
bin_width_ms=None, rerun=False, dry_run=False) -> TransferRunReport`
owns orchestration. Return status (prepared/reused/dry-run), result path/identity,
elapsed seconds and small planning details.

Optional session-local `cross_condition_config.json` has only
`base_config_path` (default `task_decoding_config.json`) and
`bin_width_ms` (default 500; existing 100/50/20 supported). Resolve contained
relative paths and require the base config to reference the selected metadata.
Require version-3 metadata and its analysis roles. With default control files,
the CLI needs only the metadata path; no version-2 support or role guessing.

Order:

1. Validate small config/behavior inputs. Use `timeline_configuration` and
   `find_saved_timeline` for metadata-only exact/registered discovery. Dry-run
   reports cells/conflicts, source status and maximum fit count, then returns
   without neural loading, writes, fits or workers. If regional compatibility
   still requires array verification, say so rather than declaring PCA absent.
2. On explicit preparation, reuse exact/registered sources first. For unresolved
   regions, use `find_compatible_regional_timeline`; unrelated event/probe config
   differences must not force a new basis. Explicitly use the session-prediction
   policy in lookup/preparation: `population_quality="good_mua"`,
   `missing_label_policy="mua"`; the regional finder's default differs. Reuse
   existing receipts where applicable; add no migration/compatibility subsystem.
3. Prepare only regions without a verified reusable basis, in at most one
   `run_session_timeline` call. Preserve already reused regions. Record source
   load/preparation failures per region and continue; never substitute an
   unverified basis. Both regions unavailable is a reportable result, not a crash.
4. Build identity from small inputs and selected source identities. When rerun
   is false, return a validated exact existing bundle before further score loads
   or fitting. On the exact/registered path this precedes all score loading;
   the regional compatibility finder may already have read arrays to verify reuse.
5. Otherwise load each distinct selected archive into the fitting stage once,
   align its choices, fit sequentially and save. Reuse scores across all fits.
   Transfer rerun creates a new dated result while reusing compatible PCA.

Write under
`session/analysis_runs/cross_condition_generalization_<date>_<time>_<id>/`:
results/config, `run.log`, `summary.md`, main/batch script copies and source
snapshots including reused scientific modules. Experimental outputs stay
outside the repository. Log source/fit/save durations, counts and fit reasons,
PCA reuse/preparation and peak RSS. Record dirty code provenance; do not
require a clean checkout. Discovery considers a run complete only after NPZ
publication; no checkpoint/recovery service is needed.

Planned commands, available only after implementation:

```bash
uv run --frozen python -m src.neural_analysis.task_decoding.cross_condition.run_session \
  --session-metadata "$METADATA" --dry-run

uv run --frozen python -m src.neural_analysis.task_decoding.cross_condition.run_session \
  --session-metadata "$METADATA"
```

The thin batch runner reuses `read_session_list` / `choose_worker_count`,
parallelizes sessions only and returns reports in input order. Dry-run is
sequential. Keep the existing CPU/session-capped worker default and lower-worker
override; no batch/cluster execution or resource changes are authorized now.

## 6. Summary webapp

Add **Cross-condition generalization** as a distinct metadata-session view.
Route before generic raw/population loading; allow empty/saved-only sessions.
Do not alter the old condition-generalization view or session timeline.

Select saved run, target, supervised window, weighting and role; show all four
directions with PFC/HPC metrics/counts. Offer combined/outcome rows through a
simple selector or grouping. Details show fit availability, eligible cell
counts and excluded conflict IDs. Labels identify the actual positive class.
Always identify unavailable regions and their reasons, including when neither
has results; region unavailability must not terminate the view.

Explicit **Prepare/Recompute** uses metadata roles and selected preparation bin
width to create the full bundle. Explain missing-PCA preparation. Changing
saved-run/window/weight/outcome selectors only reads small saved products.
Saved settings remain authoritative; preparation controls do not reinterpret
an existing result. On preparation failure, keep the previous saved selection.

Cache result/report by path, file stats and scoring version. Captions distinguish
in-sample training from complementary-condition tests, explain single-class
metrics and fixed session-wide preprocessing. No timeline navigation,
heatmaps, chart JavaScript or block-held-out/online claims.

## 7. Test-first delivery

For each package: tests -> confirm/record RED -> commit only owned tests ->
implementation -> GREEN/review -> update Section 0. Follow `AGENTS.md`;
do not change tests to accommodate incorrect code. New tests live in
`src/tests/neural_analysis/task_decoding/cross_condition/`, except the existing
helper regression tests and new webapp test below. Use seeded synthetic fixtures,
not experimental files or another package's conftest fixture.

### P1. Existing helper exposure

Add tests in `session_prediction/test_modeling.py` for public `fit_readout`:
binary coefficients/intercept match an independent sklearn fit with nonuniform
weights; old regional predictions/ridge behavior remain unchanged. Confirm
expected RED, commit tests, then rename the helper/internal call only.
Run `uv run --frozen pytest src/tests/neural_analysis/task_decoding/session_prediction -q`.

### P2. Controls, masks and alignment

Write `test_config.py` / `test_preparation.py` and fixtures first.
Hand-enumerate actual trial IDs for every source/test pair; do not derive
expected masks with the registry being tested. Cover equivalent numeric/string
labels and flags, missing sentinels, input immutability, baseline exclusions,
dark-state exclusion, conflicts, original-row alignment/timestamp tolerance,
missing context on excluded versus ordinary choices, and malformed label errors.
Assert identical outputs with absent, missing, noncontiguous or changed block
labels; shared loader defaults must retain existing block checks. Cover exact
bin masks and omission labels (Reward=0, Value=1), including right omissions.
Test version-3 metadata defaults, contained locators and invalid widths; reject
other metadata versions without adding compatibility paths.
Run these tests plus shared `test_targets.py` / `test_conditions.py`.

### P3. Numerical fitting and summaries

Write `test_modeling.py` / `test_scoring.py` first. Verify exact bin/trial
weights and Ntrain totals, equality for balanced counts, unequal-count oracle
fits, post-regional-coverage normalization, pooled-bin ordering, PC scale and
intercept. Test a missing source class/local fit failure, independent region
maps, and retained eligibility versus zero scored trials for an unavailable fit.

Use handcrafted probabilities distinguishing mean probability from
sigmoid(mean logit)/majority vote; outside bins must not affect matched scores.
Compare role/outcome rows against independent masks and direct metric values,
including empty/single-class groups. Test controlled transfer reversals and
Reward/Value omission-label differences. Do not require different targets to
produce different coefficients. Run new numerical tests plus existing
session-prediction modeling/scoring tests.

### P4. Saved products

Write `test_results.py` first: round-trip axes/meta, meaningful identity changes,
path relocation, exact labels/grids/maps/weights, failed-model conventions,
incomplete discovery and no overwrite. A copied run must load/score with raw,
base-config and PCA files removed and fitting/PCA loading disabled.
Run new persistence tests and old session-prediction result tests.

### P5. Runners: Sol single-session, then Luna batch

**P5a, Sol worker:** own source
`src/neural_analysis/task_decoding/cross_condition/run_session.py` and tests
`src/tests/neural_analysis/task_decoding/cross_condition/test_scripts.py`.
Freeze the runner's signature/report contract after Sol orchestrator review.
Write tests first:
metadata-only call; all 192 combinations when two
regions are available; exact/registered and confidently compatible regional reuse,
including unrelated configuration changes; consistent population policies;
at most one preparation of missing regions; exact reuse before unnecessary PCA
loads; new-directory rerun without PCA recomputation; no-write/no-neural-load
dry-run; source maps for shared/separate archives; required run documents; and
one/both regions unavailable without crashing. Use synthetic files/fakes, never
Slurm submissions. Run new/old prediction script tests.

**P5b, Luna worker:** after P5a API/diff review, own only source
`src/neural_analysis/task_decoding/cross_condition/run_batch.py` and tests
`src/tests/neural_analysis/task_decoding/cross_condition/test_run_batch.py`.
Write tests first
for session-level scheduling, argument forwarding, report order, worker limits
and sequential no-write dry-run. Call the reviewed single-session runner;
do not duplicate preparation, identity, eligibility or scoring logic. Sol reviews
that boundary and runs batch plus single-session regression tests, including
run-document/script-copy checks, before declaring the full P5 gate GREEN.

### P6. Summary view

Write `src/tests/neural_analysis/test_cross_condition_webapp.py` first:
actual app dispatch with raw/PCA sources absent; all targets/directions;
correct saved-weight selection; no fitting on selectors; explicit full-bundle
preparation; saved-bin labels; source/test and single-class captions; fit/eligible
counts and conflicts; one/both regions unavailable with reasons; retained
selection on preparation error.
Run with existing condition-generalization, session-prediction,
session-timeline and webapp-package tests.

### P7. Integration, documentation and final handoff

**P7a, Sol orchestrator:** after individual gates, run:

```bash
uv run --frozen pytest \
  src/tests/neural_analysis/task_decoding \
  src/tests/neural_analysis/test_cross_condition_webapp.py \
  src/tests/neural_analysis/test_condition_generalization_webapp.py \
  src/tests/neural_analysis/test_session_prediction_webapp.py \
  src/tests/neural_analysis/test_session_timeline_webapp.py \
  src/tests/neural_analysis/test_webapp_package.py -q
```

Run available changed-file lint/format checks and `git diff --check`.
Record actual commands/counts; broaden tests only for a relevant unresolved
change/failure. Inspect a seeded synthetic bundle in the actual webapp across
targets, windows, weights, directions and outcome summaries, including a
saved-only copy. Verify explicit preparation and read-only selector behavior.

Record measured synthetic runtime/RSS and freeze the demonstrated commands and
scientific captions before the documentation assignment.

**P7b, Luna worker:** own `src/neural_analysis/task_decoding/cross_condition/README.md`,
plus usage/link edits in `src/neural_analysis/README.md` and
`src/neural_analysis/task_decoding/README.md`.
Document reviewed prerequisites, commands, 100-ms controls and saved-only
copying/viewing. Use Sol-approved interpretation text and measured results;
do not invent scientific claims or performance guarantees. Preserve existing
usage docs and check links/commands against implemented entry points.

**P7c, Sol orchestrator:** review Luna's commands, scientific wording and file
diffs; run documentation/whitespace checks and resolve remaining findings.
Record actual package evidence, commits, runtime/RSS, authorization and next
action in Section 0. Repeat code tests only if subsequent code changes or an
unresolved failure justify it; docs-only edits do not repeat completed analyses.

### P8. Experimental acceptance, separately authorized

After explicit authorization, use a user-designated session and verify existing
PCA before preparation. Run long work in a durable process/job, recording
command/job/log IDs. Await completion without duplicate submissions or changes
to cluster resources.

Check behavioral cell counts/conflicts, selected-bin numerical reconstruction
and all four targets/weightings. User scientific/visual acceptance precedes
batch experimental use. Synthetic success is not experimental acceptance.

## 8. Performance and completion

Reuse loaded PCA scores across all fits. The regional compatibility finder may
read archives during verification; avoid unnecessary reloads but do not build
a new reuse/cache framework to eliminate that read. NPZ loading reads all regions.
Keep one selected training design at a time; save trial means. Probability
storage per region is `3*2*16*W*8 = 768*W` bytes before compression, plus small
masks/weights/coefficients. Measure runtime rather than promise a bound.

Fit sequentially within a session. Batch workers multiply PCA RAM use; document
the lower-worker option and measure before experimental batching. No new storage
backend, within-session scheduler, GPU/numba path or recovery system is needed.

P1-P7 completion requires passed gates, saved-only inspection, unchanged existing
numerics, usage documentation and an accurate handoff. P8 needs separate approval.

### Review record: 2026-11-10, spec revision 9

Resolved all six review points: numeric mask normalization, confident PCA reuse,
blocks fully ignored, context checked after baseline exclusions, version-3
metadata only, and region unavailability displayed without crashing. Corrected
the shared-loader scope, regional source maps, reuse/loading guarantees and tests.
Full supervised windows do not overlap; no overlap handling is planned.
Documentation checks passed: sixteen complementary two-class transfers, local
links, Markdown fences, ASCII and whitespace. No pytest or analysis was run.
This revision changes only the spec and plan; implementation remains unstarted.

Assignment update: Sol owns orchestration and scientific packages; Luna owns
the reviewed thin batch wrapper and usage docs. At most one worker runs at a
time, with RED/test-commit/GREEN/review handoffs. No worker was launched.

## 9. Implementation evidence

### P1 RED / test commit, 2026-11-10

User authorized P1-P7 implementation; P8 remains gated. Sol worker
`/root/scientific_worker` (`gpt-6.1-sol`, high) changed only
`src/tests/neural_analysis/task_decoding/session_prediction/test_modeling.py`.

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/session_prediction -q`
returned 2 expected missing-`fit_readout` failures and 173 passes in 13.16 s.
Orchestrator independently reran the two new public-helper tests: 2 expected
AttributeErrors, 20 deselected. Test-only commit: `4e1b478`.
Installed sklearn 1.8.0 source/API verified. No experimental files/jobs touched.
P1 GREEN is assigned to the same worker; source/test processes have finished.

### P1 GREEN / review, 2026-11-10

The same full session-prediction command returned 175 passes in 13.47 s.
Orchestrator reviewed the diff: only public rename/caller/docstring changed;
estimator body/settings and failures are unchanged. Implementation commit:
`c08967a`. `git diff --check` passed; user-owned hashes unchanged. P2 RED-only
tests assigned to the same Sol worker; no live analysis/job remains.

### P2 RED / test commit, 2026-11-10

Sol worker wrote self-contained controls/preparation fixtures and all sixteen
literal cell-ID expectations. Orchestrator reviewed the cells and corrected one
test-contract error before implementation: dark state excludes common analysis
eligibility, retaining recorded target values/validity.

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition/test_config.py src/tests/neural_analysis/task_decoding/cross_condition/test_preparation.py -q`
returned 2 expected absent-package collection errors (worker 0.13 s;
orchestrator 0.18 s). Existing shared targets/conditions baseline: 55 passed.
Test-only commit: `1150ea6`. P2 GREEN assigned to the same Sol worker;
no experimental processing, commits of unrelated paths or live analysis jobs.

### P2 GREEN / review, 2026-11-10

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition/test_config.py src/tests/neural_analysis/task_decoding/cross_condition/test_preparation.py src/tests/neural_analysis/task_decoding/test_targets.py src/tests/neural_analysis/task_decoding/test_conditions.py src/tests/neural_analysis/task_decoding/session_prediction -q`
returned 319 passes in 14.97 s. Reviewed all sixteen registry/cell definitions,
numeric-mask semantics, context/baseline ordering, raw audits and default-off
shared block opt-in. Narrowed a helper to its four actual data inputs during
REFACTOR; the P2-only command then passed 89 tests in 1.59 s. No test changes
after commit. Implementation: `aa70f22`; whitespace and user hashes unchanged.
P3 RED-only assigned to the same Sol worker; no live analysis/job remains.

### P3 RED / test commit, 2026-11-10

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition/test_modeling.py src/tests/neural_analysis/task_decoding/cross_condition/test_scoring.py -q`
returned 2 expected missing-modeling collection errors (worker 0.16 s;
orchestrator 0.13 s). Reviewed independent all-192 fit oracles, explicit
row/bin weights, role/outcome metrics, probability averaging, controlled
reversals, unavailable fits/regions and retained eligible versus scored counts.
Test-only commit: `fc80d35`. Frozen field contracts are recorded in Sections
3.2-3.3. P3 GREEN assigned to the same Sol worker; no experimental processing.

### P3 GREEN / review, 2026-11-10

New numerical/scoring tests: 39 passed in 6.22 s. Full assigned gate:
`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition src/tests/neural_analysis/task_decoding/session_prediction/test_modeling.py src/tests/neural_analysis/task_decoding/session_prediction/test_scoring.py -q`
returned 178 passes in 11.40 s. Orchestrator reviewed weighting/coverage,
trial-major stacking, selected-bin mean-expit outputs, local fit failures and
role/outcome metrics; clarified only a docstring's off-source missing-label
contract. No test changes. Implementation: `3ef2862`; whitespace/user hashes
unchanged. P4 RED-only assigned to the same Sol worker; no live analysis/job.

### P4 RED / test commit, 2026-11-10

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition/test_results.py -q`
returned 1 expected absent-results import error (worker 0.16 s; orchestrator
0.14 s). Reviewed real-fit round trips, shared/separate PCA references,
metadata-only identities, saved-only relocation with original sources removed,
normal saved-contract violations, interrupted publication and no-overwrite tests.
Frozen APIs are recorded in Section 4. Tests: `256f0e2`. P4 GREEN assigned
to the same Sol worker; no experimental processing or live analysis/job.

### P4 review regressions, 2026-11-10

Initial GREEN: persistence-only 63 passed in 19.71 s; cross-condition plus
old prediction-result tests 236 passed in 29.82 s. Review found mislabeled
PCA-source interval metadata, existing-manifest overwrite and silent coercion
of a noninteger/nonscalar dark-exclusion count. Fixed-setting validation was
already strengthened during worker self-review.

Orchestrator added tests before fixes:
`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition/test_results.py -q -k 'distinguishes_pca or fixed_scientific or existing_manifest or integer_scalar'`
returned 4 expected failures, 3 passes, 63 deselected in 2.50 s. Test-only
commit: `f31446e`. Same Sol worker resumes source-only fixes and final gate.

### P4 GREEN / resumed review, 2026-11-10

After interruption, the orchestrator verified existing user-file hashes and
HEAD, reviewed the preserved fixes, and confirmed no prior test process remained.
The final gate was rerun because its completion receipt was lost:
`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition src/tests/neural_analysis/task_decoding/session_prediction/test_results.py -q`
returned 243 passes in 31.15 s. PCA coverage metadata, manifest protection and
integer-scalar exclusion validation are correct; no tests were changed.
Implementation commit: `0664d85`. P5a RED-only assigned to the same Sol worker.
No experimental processing or analysis jobs were launched.

### P5a RED / test commit, 2026-11-10

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition/test_scripts.py -q`
returned one expected missing-runner import error (worker 0.17 s; orchestrator
0.13 s). Reviewed real seeded PCA fixtures, exact/registered/regional reuse,
consistent population policies, dry-run isolation, early reuse, source loading
and failures, rerun and source snapshots. Tests committed as `03ffd66`; same
Sol worker authorized for source-only GREEN. The actual main/batch copy
integration case remains an explicit unmet dependency until sequential P5b,
without a skip.

Frozen report details: session_id, bin_width_ms, probe_roles, choice_count,
target_names, window_bin_counts, max_model_count, dark_state_count,
conflict_count/conflict_trial_ids, global pre-coverage cell_counts, and
source_status mapping region to status/result_path/reason. Details are JSON-safe;
actual regional counts remain in saved models. Metadata-only dry-run source
status distinguishes exact_or_registered from compatibility_pending.
