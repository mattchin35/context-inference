# Population decoding: within-side switch-versus-stay extension

Status: S1-S5 implemented, 2026-11-14. The user approved this plan and requested
implementation. The execution record below tracks validation and the designated
CT026 inspection run; human inspection precedes any wider batch.

Scientific baseline: [population_task_decoding_spec_v1.md](population_task_decoding_spec_v1.md).
Existing delivery: [population_task_decoding_implementation_plan.md](population_task_decoding_implementation_plan.md).
[AGENTS.md](../AGENTS.md) and [SoftwareDesign.md](SoftwareDesign.md) govern
approval, explicit contracts, readable functions, and tests-first commits.

## 1. Deliverable and scope

Add two independently fitted binary targets to the existing **Population task
decoding** view's **Target** dropdown. Preserve the existing directional switch
targets and their all-trial interpretation.

| New model ID | Eligible current trials | Class 0: stay | Class 1: switch |
| --- | --- | --- | --- |
| `left_switch_vs_stay` | Current animal choice is left | Left -> left | Left -> right |
| `right_switch_vs_stay` | Current animal choice is right | Right -> right | Right -> left |

Opposite-side trials are excluded from each new target, not labeled class 0.
There is one switch-versus-stay classifier per current side, not separate
switch and stay classifiers. Display `P(switch)`; `P(stay)` is its complement.

Retain the agreed `[-1,+1)` second window, default 500-ms bins with existing
100/50/20-ms support, both units and PCs, independent PFC/HPC fits, fixed L2
logistic `C=1`, per-trial training weight one, and no balancing or tuning.
The existing context, choice, outcome, and directional-switch definitions do
not change. The Single unit selectivity view remains separate and unchanged.

The successor remains the immediately following unfiltered trial and must have
a valid animal choice. No-choice/manual-reward successors and terminal trials
are unavailable. Do not skip excluded source rows to find a later choice.
Future context/reward/neural
coverage is not required to identify a valid successor action.

Holding the current side fixed controls the direct current-port distinction.
It does not control reward, context, history, or trial difficulty. The full
window includes post-choice activity. Conditioning on the recorded current
choice, including for pre-choice bins, is an offline analysis rather than a
deployment-time prediction before that choice is observed.
With current side fixed, switch versus stay is also a next-chosen-port label:
right versus left within current-left trials, and left versus right within
current-right trials. This analysis cannot by itself distinguish upcoming-
choice coding from a switching process.

## 2. Comparison that answers the user's question

For each region, representation, and current side, display three scopes:

| Comparison row | Training trials | Held-out scoring trials |
| --- | --- | --- |
| Existing directional decoder: all trials | All original eligible trials | All original eligible trials |
| Existing directional decoder: matching side | Same existing all-trial fit | Only matching-current-side eligible trials |
| New switch-versus-stay decoder | Only matching-current-side eligible training trials | Exactly the same matching-side held-out trials as the preceding row |

The second row is rescoring, not refitting. It separates changes in evaluation
population from changes in the trained decoder. Preserve the first row as the
existing headline result; do not replace it with the restricted score.

Compare the second and third rows on identical original row positions, bin
indices, held-out fold IDs, and binary labels. Both models use their actual
saved OOF probabilities. Do not renormalize the old directional probability
using a separately decoded choice probability, recalibrate it, or optimize
thresholds on these test trials. Binary decisions still use probability 0.5.

Their probability questions differ: the old head reads a directional event
across all choices; the new head reads switching conditional on the known
current side. A raw difference in headline accuracy across different scopes
would mix class prevalence and eligibility effects. Even the matched comparison
also changes training sample size, scaling, and possibly retained units/PC rank;
it is a descriptive comparison of readouts, not a causal information test.
With fixed `C` and total weight one per training trial, the smaller training
pool also changes the penalty's weight relative to the data. Keep the agreed
`C=1` rather than silently introducing a penalty-matching adjustment.

Use the existing balanced accuracy, log loss, accuracy, confusion, per-class
precision/recall, ROC-AUC and average-precision calculations. Show trial counts,
bin counts, positive prevalence, and the corresponding baseline. The restricted
models' negative baseline is **always stay**; the all-trial head's negative
class remains **not this directional switch**. Do not call that entire negative
class "stay."

Required matched rows are established before checking prediction availability.
Do not silently retain only rows where both models happened to succeed. If a
new fold is unfit, keep the reference's matched score when valid, mark the new
primary score/comparison difference unavailable, and retain valid fold
diagnostics. Do not redraw folds, drop difficult rows, or substitute full-data
predictions. Show descriptive metric differences only when both complete
matched scores are defined; add no significance tests or confidence claims.

## 3. Minimal architecture and reuse

Extend the existing package rather than create another decoder or framework.
No dependency additions, estimator changes, generalized condition registry,
formula system, or broad shared-module refactor are needed.

An important distinction keeps this small: the new models reuse recorded
labels but have different eligibility. Keep the existing five source-label
columns and their public `TARGET_NAMES`/`CLASS_NAMES` meanings unchanged.
Introduce an explicit seven-entry `MODEL_TARGET_NAMES` for fitted models.

| Module | Planned change |
| --- | --- |
| `targets.py` | Add `MODEL_TARGET_NAMES`, a small data-only `ModelTarget`, and `build_model_target`. Existing five models map directly to their source column; the two new models reuse the corresponding switch column with a current-side validity mask and stay/switch class names. |
| `pipeline.py` | Make `fit_target` select rows/classes from `ModelTarget`. Add a small pure `build_switch_comparison` function for strict row/bin/fold/label alignment and comparison metadata. Preserve existing numeric fitting calls. |
| `scoring.py` | Reuse `score_predictions` on the explicitly selected reference and new predictions, pooled and per fold. Add a narrow helper only if needed to avoid repeating this same calculation; no alternate metrics implementation. |
| `configuration.py` | Declare the extended analysis version and fixed seven-model set. New fits use that version; no new scientific tuning controls. |
| `results.py` | Read the existing and extended analysis versions; preserve five-column preparation axes. Validate the added model-scope/comparison metadata and store it with the new paired cells. |
| `run_session.py` | Create the expanded manifest, derive counts from it, and attach comparisons after the existing reference cell is published. Add an optional initializer output path for creating a new config without replacing the existing one. |
| `run_batch.py` | Keep session-level scheduling; inherit extended single-session execution. No within-session workers. |
| `plotting.py` | Recognize the new binary readouts explicitly, show positive switch probability on eligible windows, and include the conditioning side in captions. |
| `webapp/population_decoding_view.py` | Offer targets declared by the selected run, add the two new entries for extended runs, and render a compact three-row comparison table. |
| README/spec/plan documents | Explain versioned execution, eligibility, matched comparison, interpretation, and validation evidence. |

Keep computational functions pure and typed. Pass source columns, prepared
label records, and arrays rather than a session object. The runner owns file
loading/publication; the viewer loads saved products and never fits or rescales.
The existing feature/scoring libraries already provide the required methods.
Verify any newly used library API from installed source or official docs.

## 4. Data contracts

N denotes complete chronological source trials; W a target's regional eligible
windows; B its supervised bins; U selected units; P training-variable units;
K numerical nonzero PCs. Physical activity remains Hz; standardized unit/PC
inputs, labels and probabilities are dimensionless; times are seconds.

| Boundary | Contract |
| --- | --- |
| Existing prepared labels | `PopulationTargets.values/valid/reasons` retain `(N,5)` axes and original source target order. No `(N,7)` rewrite or duplication of neural activity is required. |
| `build_model_target(name, targets)` | Return `ModelTarget`: labels `(N,)`, validity `(N,)`, reasons `(N,)`, ordered class names, source target ID, and optional conditioning side. Unknown/excluded labels are `-1`; source arrays are unchanged. |
| New eligibility | Existing switch validity AND known current animal choice AND current side equals the requested side. Baseline and regional coverage are then intersected in the existing pipeline. |
| Regional cell | Retain original row positions `(W,)`, bin indices `(B,)`, labels/folds `(W,)`, and paired probabilities/log probabilities `(W,B,2)`. New class order is `stay=0,switch=1`. |
| Unit/PC transforms | Fit only the target's eligible training rates `(W_train,B,U)`; use one scaler/mask for both representations and PCA of those same standardized vectors. Apply unchanged to held-out matching-side rows. |
| `build_switch_comparison(current_choices, reference_cell, new_cell)` | Current choices are `(N,)`; saved-cell records carry original row/bin/fold/label mappings. Return the reference-window index map `(W,)` and pooled/fold comparison metadata. Raise on structural misalignment rather than silently intersect. |
| Comparison metadata | Per representation, preserve reference all-trial scores, reference matched-side scores, new within-side scores, required/predicted support, provenance, and defined descriptive differences/reasons. No duplicated probability tensors. |

For model-target reasons, distinguish an opposite current side from an unknown
successor or invalid current choice. Preserve the original unfiltered trial ID
and row position throughout. Membership uses row positions; irregular trial IDs
are not reinterpreted as contiguous array indices.

## 5. Training and publication sequence

1. Prepare/validate the same complete source tables, source-label arrays,
   baseline pool, trusted regional windows and native Hz arrays as today.
2. Establish the global five-fold block assignment once from the full baseline
   pool, before new side masks. Do not construct new folds from each side.
3. Fit the existing five targets with their existing labels, eligibility,
   bins, weights, model parameters and ordering. Their scientific computation
   remains unchanged under the extended run version.
4. For each new target/fold, select matching-side eligible training and test
   windows. Both classes are still required in training. A constant unit on
   this restricted training pool is omitted even if it varied across all trials.
   All-constant inputs or missing classes remain unavailable with reasons.
5. Reuse the existing scaler/PCA/logistic/OOF functions on those arrays. Never
   reuse the old all-trial target's scaler or PCA as the new target transform.
   Trial weight is still one, distributed as `1/B` per supervised bin.
6. Before publishing a new paired target cell, the runner loads its already
   published all-trial reference cell from the same run and checks run identity.
   The pure comparison builder selects the reference's matching-side rows and
   verifies exact equality to the new cell's required rows/bins/folds/labels.
7. Score both sets of matched OOF predictions. Store the reference-window
   index mapping as a named array in the new cell, and the comparison summaries
   and reference identity/path in `meta['switch_comparison']`.
8. Publish the new paired cell atomically, then update progress using the
   current file-based completion procedure. New model IDs follow the five
   reference models in execution order, so dependencies are explicit.

Comparisons travel with the four new region/target cells; there is no separate
comparison job, cache tree, or checkpoint state. Interruption before publication
redos at most that paired cell. A published cell includes its comparison, so
the existing orphan-publication adoption remains sufficient. Missing/corrupt
reference files produce explicit errors, not substituted data or overwritten
results. No comparison is silently attached to a different run.

## 6. Versions, old runs, and entry points

Use `population-task-decoding-v2` for new extended fits. The named-array
geometry can retain schema version 1 because source-label axes stay `(N,5)`;
the analysis version distinguishes the seven-model scientific profile.
Readers explicitly accept v1/v2 with that schema and reject unknown versions.
Do not introduce a generic migration/version-negotiation framework.

Old v1 runs remain readable in the viewer and through read-only status/loading.
They are not rewritten, relabeled, supplemented in place, or resumed under the
changed numerical code. Their dropdown exposes their original five targets;
an extended run exposes seven. Derive these options from the saved manifest,
not solely from the currently installed global model-name list.

New execution requires a v2 config. Keep the initializer's existing default
filename for new sessions and add optional `--output`/`output_path` so CT026
can create `population_decoding_switch_vs_stay.json` beside its existing
`population_decoding.json`. The latter and its v1 manifest remain intact.
Attempting new/resume work with a legacy profile should explain how to create
a new config/run; read-only status/viewing must continue to work.

Use the same `new`, `dry-run`, `status`, `resume`, `--rerun`, and batch entry
points. New manifests declare model IDs explicitly. Replace the current hard-
coded 10-cell summary/100-fit plan counts with counts derived from the manifest
and fixed representations/folds: two-region extended runs have 14 paired cells,
28 region/target/representation combinations, and at most 140 classifier fits.
Existing manifests still describe ten cells correctly.

Run/preparation identities include the new code/profile and retain existing
source/runtime provenance. Prepare into the new immutable location using the
existing source loaders; do not add a v1 cache-import adapter for this extension.
Within each run, activity is prepared once and shared by all seven models.
Continue saving logs, script copies, measured performance, progress and summary
in session-local run directories; no experimental products enter Git.

## 7. Viewer and human-readable output

Keep the existing Plot view entry and independent Single unit selectivity view.
Inside Population task decoding:

- Add readable labels "Switch vs stay: current left" and "Switch vs stay:
  current right," backed by the explicit new model IDs.
- Show actual stay/switch labels and OOF `P(switch)` across eligible current-side
  `[-1,+1)` windows. Leave opposite-side and invalid windows blank; preserve
  session elapsed time, gaps, and the common viewport when switching targets.
- Show the conditioning side, eligible trial/class counts, and always-stay
  baseline near the new model's scores.
- Show the three-row comparison table with score scope, trial/bin support,
  balanced accuracy, log loss, ROC-AUC and average precision. Clearly distinguish
  all-trial versus matched-side evaluation. Retain metric/fold-specific reasons.
- Make the same comparison accessible when selecting the corresponding existing
  directional target in a v2 run. While its companion cell is pending, display
  the existing result and a short comparison-pending message.
- For a v1 run, retain its current results and explain that conditional models
  require an extended run; do not show fabricated missing target files.
- Keep computation out of the UI. Loading, changing selectors, and exporting
  consume saved products only, including when raw sources are absent.

Reuse the existing probability chart, navigation and PNG export. Captions name
the current-side restriction and OOF scope. Keep fold diagnostics behind the
existing expander. No extra Plot view, overlay mode, plotting dependency,
coefficient contribution analysis, or new theme controls are needed.

## 8. Tests written first and implementation phases

Before each phase, write pytest tests, demonstrate RED, commit tests alone,
then implement GREEN and refactor with tests passing. Record commands/commits.
Adjust an existing expectation only for this approved extension or a genuine
test error; retain v1-specific regression expectations in dedicated fixtures.

### S1. Model-label eligibility and unchanged source contracts

Extend `test_targets.py`/`test_pipeline.py` first:

- Exact left/right switch-versus-stay definitions for all four valid transitions.
- Opposite-side rows invalid with `-1`, never stay/zero; original source arrays
  and five target definitions unchanged and unmodified.
- Immediate invalid/manual/no-choice successors, unknown flags, terminal rows,
  irregular IDs, and valid successors with missing future reward/context.
- Unknown current reward does not remove an otherwise valid switch target.
- One shared block map before side selection; no side-specific split redraws.

### S2. Restricted training and matched OOF comparisons

Extend feature/model/pipeline/scoring tests first:

- New scaler/mask/PCA uses only matching-side training bins; perturb opposite-
  side or held-out activity and assert no change to that fold's new transforms
  and parameters. All-trial reference transforms remain independent.
- Training-constant units on one side are omitted in both representations;
  full-PC retention/rank, weighting, class order and solver settings are intact.
- A deliberately permuted row/bin/fold/label mapping is rejected. Missing
  predictions do not shrink the required comparison scope or become zeros.
- Reference matched scores equal an independent direct rescore of its saved
  OOF rows; reference all-trial metrics/probabilities remain unchanged.
- Missing training/test classes, empty-side folds and incomplete OOF produce
  the existing metric-specific reasons without a new split strategy.
- Pooled scores use all required matching bins, not an equal fold mean;
  comparison differences are absent when either required score is undefined.

### S3. Persistence, execution, and v1 viewing compatibility

Extend result/config/runner/batch tests first:

- Minimal frozen v1 fixtures still load with five source columns and five
  model options; their file bytes remain unchanged. Unknown versions fail.
- New v2 config/manifest/cells retain source-label geometry, explicit model
  IDs, scope metadata, reference mapping and comparison provenance.
- A complete two-region run publishes 14 paired cells, including both new
  representations, with no separate comparison checkpoint tree.
- Initializer output override creates a new config without replacing v1;
  legacy execution/resume gives a clear message while read-only status works.
- Source-free resume, compatible completed-cell skipping, interruption before
  publication, orphan adoption, and corrupt/missing reference behavior.
- Dry-run derives counts without neural loading or output writes; existing
  session worker limits/duplicate checks and one-thread numerical workers hold.

### S4. Viewer and exports

Extend webapp/plot tests first:

- Seven target choices for v2, five for v1, and the same independent Plot view
  routes. Existing context/choice/outcome/directional controls remain usable.
- New target label strips/probabilities are blank on opposite-side windows;
  actual-time gaps, overlapping windows, viewport and chart identity persist.
- Comparison table labels all three train/test scopes and shows correct side
  counts, baseline, scores and unavailability. Partial companions are explicit.
- Raw-free viewing/selector changes perform no source reads, fits or transforms.
- PNG captions identify current side, target window and OOF scope; existing
  non-overwrite/background/readability checks continue to pass.

### S5. Synthetic end-to-end validation and designated-session inspection

Extend synthetic tests before final integration:

- Reuse known switch-signal populations: new models recover switching within
  each side, for units and PCs, using the same global held-out blocks.
- In a current-side-only signal plus independent noise control, the all-trial
  directional head can discriminate partly through current side, while matched
  evaluation and new within-side models do not invent switching information.
  Include noise features so an all-constant fit is not mistaken for chance.
- Include rare switches, constant units, missing successor/reward data,
  one-side class absence and different regional coverage.
- Exercise raw synthetic spike loading through all seven models, publication,
  source-free reload/resume and saved comparison rendering. No labels enter
  actual decoder features; using labels to generate artificial spikes is a
  separate validation fixture operation.
- Run focused population tests, related existing timeline/encoding/UI checks,
  and a browser check of v1/v2 saved views. Keep the collection-time pytest
  main-module isolation so Streamlit tests cannot corrupt spawned workers.

After implementation authorization and synthetic checks, use the already
designated CT026 2026-08-28 session for the extended comparison run. Preserve
the existing v1 run and report both matched and full-scope metrics. This plan
turn starts no experimental fit. Human inspection precedes any wider batch.

## 9. Feasibility, performance, and acceptance

Read-only inspection of the existing CT026 8/28 v1 results found the same
covered eligibility for both regions:

| New target | Trials | Switch | Stay |
| --- | ---: | ---: | ---: |
| Current left | 300 | 125 | 175 |
| Current right | 392 | 123 | 269 |

Both classes are present in every relevant training fold. This is feasibility
evidence for that saved session, not a new model fit or a reason to alter folds
for future sessions.

The full extension adds at most 40 readout fits and 20 PCA fits to the existing
100-readout/50-PCA run. New fits use smaller current-side pools. Share one
regional Hz tensor across targets and one scaler per target/fold across its
two representations. Comparisons reuse existing probabilities and store only
small mappings/summaries. Process one paired target/fold at a time; retain
session-level batch parallelism. No new optimization/dependency is proposed.

Measure actual times/memory/output sizes on synthetic data and CT026 rather
than assume a proportional runtime increase. With all nonzero PCs retained,
the raw/PC comparison still mainly rotates the same standardized features.

Acceptance requires: unchanged original target semantics and readable v1
results; correct side restriction; no held-out/opposite-side preprocessing
leakage; exact matched comparison rows; explicit unavailable scopes; successful
synthetic and CT026 saved-result inspection; and no overwritten older outputs.
The extended experimental result remains subject to the user's inspection.

## 10. Execution record

Implementation was authorized on 2026-11-14 and followed tests-first phases:

| Phase | Tests commit / RED evidence | GREEN implementation |
| --- | --- | --- |
| S1-S2 | `2ce576a`: missing seven-model API, side-restricted fitting and exact OOF comparison contracts | `a45d708`: explicit ModelTarget, independent side-only training transforms, strict comparison builder; 26 focused tests passed |
| S3 | `7fbce6f`: four failures for v2 output selection, legacy execution rules and saved comparisons | `5190eed`: fourteen-cell execution, companion comparisons and v1/v2 readers; 92 cumulative tests passed |
| S4 | `ee83a7b`: missing new dropdown entries and undesired P(stay) trace | `8d46402`: manifest-driven options, compact three-scope comparison and conditional P(switch); 95 cumulative tests passed |
| S5 | `0baa2ac`: three failures accepting missing/incorrect comparison summaries; synthetic controls added | `e605d5f`: saved scope/count/class validation; 101 cumulative tests passed |
| Recovery regression | `fe2e795`: simulated interruption before conditional publication, then source-free resume | Existing runner behavior passed; published reference cells remain byte-identical, only nine missing pairs fit |

Final focused integration command:

```bash
NODE_BINARY=/usr/lib/chatgpt/resources/cua_node/bin/node \
UV_CACHE_DIR="$PWD/.pytest_cache/uv" uv run --frozen --no-sync pytest -q \
  src/tests/neural_analysis/population_decoding \
  src/tests/neural_analysis/encoding \
  src/tests/neural_analysis/test_timeline_plotting.py \
  src/tests/neural_analysis/test_timeline_chart.py \
  src/tests/neural_analysis/test_session_timeline_webapp.py \
  src/tests/neural_analysis/test_session_metadata.py \
  src/tests/neural_analysis/test_psth_webapp.py --import-mode=importlib --tb=short
```

Result: **512 passed in 63.89 s**, including 102 population-decoding tests.
The existing broad-suite limitations recorded in the original plan are not
claimed resolved; no unrelated analysis source/tests were changed.

The raw Poisson-spike synthetic pipeline now publishes all seven models in
both regions/representations and requires balanced accuracy above 0.8 for each
planted signal. Within-side switch signals are recovered. The current-side-only
plus independent-noise control produces directional all-trial discrimination,
but matched-side and new within-side AUCs remain between 0.3 and 0.7. Tests
also cover training-constant units, one-side class absence, missing required
predictions, transform perturbations, malformed saved maps/scopes, and legacy
source-free viewing. No test labels enter actual experimental features.

### CT026 2026-08-28 v2 inspection

Run: `population_task_decoding_20261114_192507_406224_a176e2c37f26`.
Session-local `inspection.md` contains full three-scope balanced accuracy and
matched log-loss tables for both regions/representations. The run completed
14 paired cells and 140 successful fold fits with complete required OOF
predictions. It used 696 baseline windows, 141 PFC units and 286 HPC units at
500 ms. Matching-side pools were exactly 300 left (125 switch/175 stay) and
392 right (123 switch/269 stay), sharing the original block map.

All 20 fingerprinted legacy files, including the v1 config/preparation/run,
remain byte-identical. The original five models in v2 exactly reproduce v1's
saved probabilities (maximum absolute difference zero). Input/code/runtime
identities were verified after fitting. New configuration:
`population_decoding_switch_vs_stay.json`; the original config is retained.

Descriptive unit-input balanced accuracy (original matching-side / new): PFC
left 0.5766 / 0.5779; PFC right 0.5828 / 0.6386; HPC left 0.5057 / 0.5276;
HPC right 0.5014 / 0.5356. Full-PC readouts were similar. These observations
are not significance tests or proof of a distinct switch-intention signal.

Measured v2 execution was 10.13 s including 3.45 s preparation, peak process
RSS 685,895,680 bytes, preparation archive 4,279,616 bytes and paired result
archives 30,613,723 bytes. These are session-specific measurements.

Browser inspection on a temporary loopback server verified seven v2 options,
P(switch) with opposite-side windows blank, exact three-row counts/scores, five
preserved v1 options and their legacy explanation, and the separate Single
unit selectivity route. The temporary server was stopped. Streamlit logged
local inotify watcher-limit warnings; result rendering completed normally.
Human inspection is pending. No wider experimental batch was run.
