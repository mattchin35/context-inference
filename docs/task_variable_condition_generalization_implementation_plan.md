# Pooled-Window Task-Variable Decoding and Condition-Generalization Plan

**Status:** Revision-1 documentation and PW-WP1 through PW-WP5 were approved
on 2026-10-11. The PW-WP6 CT026 run is scientifically complete and validated,
and the transfer-figure layout correction is implemented and validated. User
review of the corrected preview and a decision about the incomplete full-run
telemetry remain open. PW-WP7 and later work packages and cluster work are not
authorized.

**Scientific contract:**
`docs/task_variable_condition_generalization_spec_v1.md`.

**Reference contracts:** `docs/task_variable_spec_v5.md`,
`docs/task_variable_spec_v6.md`, `docs/task_variable_spec_v7.md`, and
`docs/task_variable_implementation_plan.md` describe the implemented
condition-specific per-time-bin decoder. Their scientific meaning and saved
results do not change.

## 1. Live implementation handoff

Update this section at every work-package boundary and before handing work to
another task. Do not replace prior commit or test evidence with only a prose
summary.

| Field | Current value |
| --- | --- |
| Snapshot date | 2026-10-11 |
| Branch | `refactor` |
| PW-WP1 implementation HEAD | `65f8a2e5eb04071dbecf26debc03e334a6625fd2` |
| PW-WP2 implementation HEAD | `2ffb1ab94323db375d5c8dd5154c770a9809ba66` |
| PW-WP3 implementation HEAD | `bf431956e2e23128f8cb8817c47a6c1adde518da` |
| PW-WP4 validation HEAD | `a99630f` |
| PW-WP5 implementation HEAD | `1ff6b13d309bc85828bb547be387cd67b6c9ac9b` |
| PW-WP5 provenance correction HEAD | `715ab95` |
| PW-WP6 reporting correction HEAD | `ed09fe7` |
| Approved documentation commit | `7df0100` (`updating task variable plans for generalized encoding analysis`) |
| Current work package | PW-WP6: CT026 pre-choice benchmark |
| Current phase | Experimental run terminally valid; corrected saved-result preview awaiting user review before a separately approved telemetry rerun |
| Completed work packages | PW-WP0 documentation; PW-WP1 configuration and master folds; PW-WP2 pooled scientific core; PW-WP3 result schema and synchronous computation; PW-WP4 synthetic scientific gate; PW-WP5 reporting, lifecycle, resources, usage, and runtime-provenance correction |
| Active implementation files | None; immutable CT026 run `analysis_runs/pooled_task_variable_decoding_20261012T011808Z` remains unchanged outside the repository |
| Latest validation | 2026-10-12: plotting correction focused `4 passed`, pooled `68 passed`, full `946 passed, 27 warnings`; all 54 actual CT026 layouts pass renderer containment/non-overlap checks; six external previews were rendered with all 91 source-run files unchanged |
| Open blockers | User inspection of corrected preview; exact full-run CPU/peak-RSS telemetry was lost during operator recovery and requires a separately approved fresh benchmark |
| Exact next action | Present the corrected preview; if the user approves it and authorizes the rerun, launch exactly one detached fresh benchmark and await it using read-only durable status only |

The repository contains unrelated untracked files. They are outside this
analysis and must not be added, changed, or removed. Before each implementation
work package, record any new relevant worktree changes and coordinate rather
than overwriting concurrent work.

### 1.1 Work-package ledger

Add one row when a phase completes. A work package is complete only after its
focused tests, the full existing task-decoding suite, independent review, and
user approval all pass.

| Work package | RED commit | GREEN/refactor commit | Focused tests | Full regression | Review and approval | State |
| --- | --- | --- | --- | --- | --- | --- |
| PW-WP0 documentation | Not applicable | `7df0100` | Documentation checks passed 2026-10-11 | Not applicable | User approved 2026-10-11 | Complete |
| PW-WP1 configuration and folds | `ebc8cc8`, `7aec299` | `65f8a2e` | `20 passed` | `890 passed, 27 warnings` | User approved by authorizing PW-WP2 on 2026-10-11 | Complete |
| PW-WP2 scientific core | `2c00910` | `2ffb1ab` | `128 passed, 2 warnings` | `912 passed, 27 warnings` | User approved by authorizing PW-WP3 on 2026-10-11 | Complete |
| PW-WP3 result schema, checkpoints, and synchronous computation | `46db250`, `0d95c95` | `bf43195` | `11 passed`; combined boundary `425 passed` | `925 passed, 27 warnings` | User approved by authorizing PW-WP4 on 2026-10-11 | Complete |
| PW-WP4 synthetic scientific gate | `a99630f` | Not required; PW-WP3 passed the new gate unchanged | `3 passed`; pooled suite `50 passed` | `928 passed, 27 warnings` | User approved by authorizing PW-WP5 on 2026-10-11 | Complete |
| PW-WP5 reporting, lifecycle, resources, and usage | `c29f259`, `718a3dc`, `1e2e9ed`, `48bcfa6`, `15e923a`; fixture correction `9333520` | `1ff6b13`, `715ab95` | Pooled suite `65 passed`; focused Ruff passed | `943 passed, 27 warnings` | User approved entry to PW-WP6 on 2026-10-11; corrective provenance review passed | Complete |
| PW-WP6 CT026 benchmark | Reporting correction `3342579`; test-fixture correction `5049e47` | Reporting correction `ed09fe7` | Plotting `4 passed`; pooled `68 passed`; 54/54 actual layouts pass geometry audit | `946 passed, 27 warnings` | Corrected preview and fresh telemetry decision await user review | Awaiting review |
| PW-WP7 optional descriptive comparison | - | - | - | - | - | Deferred |
| PW-WP8 cluster and batch execution | - | - | - | - | - | Deferred |
| PW-WP9 additional windows and experimental batch | - | - | - | - | - | Deferred |

### 1.2 Handoff update requirements

Every handoff must record:

- branch, HEAD, and relevant clean/dirty paths;
- the current work package and RED/GREEN/refactor/review phase;
- exact commits, test commands, pass/fail counts, and unresolved failures;
- files changed and any public interface change;
- decisions made, open decisions, blockers, and user approvals;
- synthetic fixture or experimental run directories, without committing data;
- measured elapsed time, CPU time, peak RSS, and output size when available;
- the exact next safe action and actions still requiring external approval.

### 1.3 PW-WP0 validation evidence

No Python tests were run because PW-WP0 changes documentation only. The
following read-only checks passed on 2026-10-10:

```text
git diff --check
rg scans for unresolved weighting, dense-prediction, comparison, module, and
status language
ASCII scan of both changed Markdown files
existence checks for every referenced existing document and reused module
even code-fence count in both changed Markdown files
installed scikit-learn source/API characterization for LogisticRegression and
ElasticNet sample-weight behavior
```

The formatting, ASCII, fence-balance, stale-contract, and required-clause
checks were rerun successfully on 2026-10-11 after the prepared-run,
dependency-scope, reporting-provenance, resource-schema, parent-initializer,
and synthetic-oracle corrections.

The 2026-10-10 external review corrections also fixed the canonical condition
overlap description, froze the complete-pipeline interpretation and limits of
the native-minus-transfer gap, added exact state propagation and numerical
eligibility, isolated pooled source scope, resolved configuration
serialization/identity, replaced repeated evaluation records with dense
arrays, reduced required figures to `3*W*T`, simplified resource evidence, and
moved the synchronous synthetic gate ahead of reporting infrastructure.

The installed scikit-learn version is 1.8.0; both estimator `fit` signatures
accept `sample_weight`. On a 120-row, six-feature synthetic fixture with NumPy
seed `20261010`, strict controls (`max_iter=10000`, `tol=1e-10`) gave maximum
repeated-versus-single coefficient/intercept deltas of `2.17e-10` for SAGA
logistic regression and `1.67e-16` for ElasticNet. Under the project's frozen
production controls, the coefficient/intercept deltas were
`2.39e-4`/`1.15e-4` for SAGA and at most `8.33e-17` for ElasticNet. Section 7
therefore separates exact weight-plumbing tests from explicitly toleranced
numerical characterization.

The approved documentation is committed at `7df0100`; the PW-WP1 handoff
update is committed at `3ea2f1c`.

### 1.4 PW-WP2 implementation evidence

PW-WP2 followed tests-first commits. Commit `2c00910` added the scientific-core
contracts without production changes. The focused RED command was:

```text
uv run pytest -q src/tests/neural_analysis/task_decoding/test_modeling.py \
  src/tests/neural_analysis/task_decoding/pooled/test_decoding.py
```

It produced the expected `22 failed, 106 passed, 2 warnings`; every failure
was a missing PW-WP2 API or enum. Commit `2ffb1ab` implemented the minimal
fit/evaluate seam and pooled core. The same focused command then produced
`128 passed, 2 warnings`. The full regression command:

```text
uv run pytest -q src/tests/neural_analysis/task_decoding
```

produced `912 passed, 27 warnings in 139.47s`. The 27 warnings are the same
existing Pynapple bin-boundary, grouped-split class-count, and empty plotting
slice warnings present before PW-WP2. `uv run ruff check` on the four changed
source/test files, syntax compilation, an AST docstring-presence scan, and
`git diff --check` also passed.

The public change in `modeling.py` is limited to closed fit/evaluation failure
enums, frozen runtime fit/evaluation containers, and reusable
`fit_estimator`/`evaluate_fitted_estimator` functions. Existing
`fit_and_score` now delegates to them with no sample-weight keyword and retains
its legacy result/reason contracts. `pooled/decoding.py` adds stacking, exactly
two anatomical transform fits per valid condition/fold cell, weighted model
fitting, frozen through-time evaluation, reference-model reuse, and exact
native-minus-transfer gaps. No result persistence, pipeline lifecycle,
plotting, experimental-data execution, dependency, or new module was added.

No synthetic or experimental run directory was created. Elapsed focused and
full test times were 10.38 s and 139.47 s; CPU time, peak RSS, and output size
were not collected because PW-WP2 does not run the resource-tracked pipeline.
The relevant source/test paths are clean at implementation commit `2ffb1ab`;
unrelated pre-existing untracked repository files remain untouched. The exact
next safe action is independent scientific/API review and user approval of
PW-WP2. PW-WP3 result-schema or persistence work remains unauthorized.

### 1.5 PW-WP3 implementation evidence

PW-WP3 followed tests-first commits. Commit `46db250` added the result-schema,
checkpoint, fingerprint, cleanliness, synchronous-computation, and resume
contracts without production changes. The RED run failed during collection
because `pooled.results` and `pooled.pipeline` did not exist. Commit `0d95c95`
corrected only a genuine test expectation error: model, evaluation, and gap
shape tuples had omitted their region axis. No scientific requirement changed.

Commit `bf43195` implemented the exact schema-1 primitive arrays, safe
non-pickle atomic checkpoints and final result loading, explicit pooled source
identity, the narrow legacy nested-package cleanliness correction, target
audit projection, checkpoint-to-dense status/reason propagation, and the one
synchronous computation/resume path. It also added checkpoint identity fields
to `PooledWindowTargetResult`. It did not add lifecycle state, reports,
plotting, a CLI, resource telemetry, detached execution, or prediction arrays.

The PW-WP3 contract command:

```text
uv run pytest -q \
  src/tests/neural_analysis/task_decoding/pooled/test_results.py \
  src/tests/neural_analysis/task_decoding/pooled/test_pipeline.py
```

produced `11 passed`. The broader focused command over the complete pooled
package plus legacy result persistence produced `425 passed in 6.05s`. The
full relevant regression command:

```text
uv run pytest -q src/tests/neural_analysis/task_decoding
```

produced `925 passed, 27 warnings in 124.27s`. The warnings are the same
existing Pynapple bin-boundary, grouped-split class-count, and empty plotting
slice warnings. Focused Ruff and `git diff --check` passed.

A repository-wide `uv run pytest -q` attempt was also made and stopped during
collection with 20 unrelated environment errors: `autograd` is absent and a
local `statsmodels` module shadows the external package needed by interregional
tests. PW-WP3 introduced neither dependency and the task-decoding boundary is
fully green, so no out-of-scope dependency or import changes were made.

No synthetic or experimental run directory was created. CPU time, peak RSS,
and output size were not collected because PW-WP3 tests only small synthetic
in-memory fixtures. Unrelated pre-existing workspace changes remain untouched.
The exact next safe action is independent schema/source/API review and user
approval of PW-WP3. PW-WP4 synthetic scientific-gate work remains unauthorized.

### 1.6 PW-WP4 synthetic validation evidence

Commit `a99630f` added tests 23-25 in `pooled/test_synthetic.py` before any
PW-WP4 production edit. The seeded fixture uses 24 trials in 12 intact
behavioral blocks, 40 canonical 100 ms bins, two selected units per region,
three fixed outer folds, and fixed seed `20261011`. Tests inject this already
prepared scientific input only at the session-loading seam, then exercise the
public `compute_or_resume_pooled_results` path through model fitting, dense
assembly, immutable checkpoints, final publication, and strict reload.

The synthetic gate covers stable within-condition decoding, stable transfer,
sign-changing/rotating code, destination remapping, absent destination code,
delayed shared code, constant-reference unavailability, overlapping
conditions, and all three training windows. The rotating case uses the
existing `modeling.fit_and_score` only as a test-side independent-bin oracle.
Persisted native-minus-transfer gaps are checked directly. The multi-window
fixture produces distinct models and proves that adding post-choice/full
windows leaves the pre-choice result unchanged.

The first complete run of the new tests passed unchanged against the committed
PW-WP3 implementation. The assertions were then strengthened to inspect saved
gap arrays and genuinely different rotating-window models; they again passed:

```text
uv run pytest -q \
  src/tests/neural_analysis/task_decoding/pooled/test_synthetic.py
```

produced `3 passed in 32.65s`. Because the new scientific contracts exposed no
production defect, PW-WP4 has no GREEN production commit. Adding a synthetic
production path or otherwise changing already-correct code would violate the
single-path and minimal-scope decisions.

The full-window fixture recorded exactly 36 estimator calls:
`2 conditions * 1 target * 3 folds * 3 regions * 2 representations`, independent
of its 40 evaluation bins. The interrupted two-window run preserved the first
checkpoint byte-for-byte, restored it, computed only remaining cells, and
matched one-shot arrays and canonical metadata exactly. Every unavailable cell
in the constant-reference case had an explicit compact reason; stable runs had
no unavailable scientific status.

The complete pooled suite produced `50 passed in 32.99s`. The full relevant
regression command produced `928 passed, 27 warnings in 154.64s`; warnings are
the unchanged Pynapple bin-boundary, grouped-split class-count, and empty
plotting-slice warnings. Focused Ruff and `git diff --check` passed. No
experimental session, report, lifecycle feature, plotting, CLI, dependency,
or production module was added. At that boundary, the exact next safe action
was independent scientific review and user approval of PW-WP4 before PW-WP5.

### 1.7 PW-WP5 implementation evidence

PW-WP5 followed tests-first commits. Commit `c29f259` added tests 26-32 for
saved-result-only figures, exact dry-run counts/bytes, immutable preparation,
interruption and resume, strict lean resource evidence, CLI delegation, and
the portable example. The initial focused RED run produced `8 failed, 16
passed`; every failure named an absent PW-WP5 module or API. Commit `9333520`
corrected one genuine fixture error discovered during GREEN: the lifecycle
test had accidentally invoked a one-spike Pynapple fixture with empty time
support. It now injects a schema-valid computation context at the existing
scientific-loading seam; the separate synchronous and synthetic tests still
exercise real pooled computation and checkpoint resume.

The ownership/failure audit added three further RED-first regressions. Commit
`718a3dc` proves that a losing local writer cannot mutate an active run's
state, log, or resource evidence. Commit `1e2e9ed` requires an input-manifest
failure to publish failed resource evidence without accepting a checkpoint.
Commit `48bcfa6` requires a detached child to wait for its matching parent
receipt before execution, preventing a late receipt from invalidating final
output-byte accounting. Each focused test failed for the intended missing or
unsafe behavior before the production correction.

Commit `1ff6b13` implements the four reviewed public lifecycle APIs:

```text
plan_pooled_decoding_session(config_path)
prepare_pooled_decoding_run(config_path)
run_prepared_pooled_decoding(run_directory)
inspect_pooled_decoding_status(run_directory)
```

Planning reads small metadata/CSV/IRIG inputs only and reports exact `G`,
requested transform/model/evaluation counts, tensor bytes, and planned major
array bytes. Preparation writes canonical config/input/source identity,
launch/main/plotting snapshots with hashes and byte sizes, command files,
lifecycle state, and initial lean resource evidence before neural loading.
Execution revalidates reconstructed paths and the live input manifest before
checkpoint acceptance, uses the single PW-WP3 synchronous computation path,
preserves exact checkpoints on interruption, and publishes terminal state only
after strict result, summary, figure, checkpoint, resource, and output-size
validation. Exact completed reentry and active-owner rejection are read-only.

`pooled/plotting.py` loads only validated `results.npz` content and creates
exactly native, transfer, and gap PNGs for each window/target cell. The six
region-by-representation panels identify metrics and conditions/pairs, mark
the training interval, label unavailable cells, distinguish in-window from
out-of-window evaluation, and state the complete-pipeline/remapping limits.
No spikes, fitted estimators, dense predictions, or new dependency enter the
reporting path. The thin CLI provides `dry-run`, foreground or detached `new`,
exact `resume`, and read-only `status`; no Slurm, batch, webapp, or generic
workflow layer was added.

The final automated gates were:

```text
uv run ruff check src/neural_analysis/task_decoding/pooled \
  src/tests/neural_analysis/task_decoding/pooled
# All checks passed

uv run pytest -q src/tests/neural_analysis/task_decoding/pooled
# 60 passed in 40.39s

uv run pytest -q src/tests/neural_analysis/task_decoding
# 938 passed, 27 warnings in 153.91s
```

The warnings are the unchanged Pynapple bin-boundary, grouped-split
class-count, and empty plotting-slice warnings. `git diff --check` also passed.
At that PW-WP5 boundary, no experimental session was loaded, no durable
analysis run directory was created, and no measured CT026 resource envelope
existed; lifecycle tests used pytest temporary directories. The exact next
safe action was independent review and user inspection of PW-WP5. Preparing or
running the CT026 benchmark remained gated on explicit PW-WP5 approval.

After approval to enter PW-WP6, preflight found two narrow contract gaps before
experimental use: the pooled CLI imported numerical libraries before applying
the inherited one-thread benchmark limits, and pooled results omitted required
software-version provenance. Test-first commit `15e923a` produced six intended
failures with 18 passing focused tests. Commit `715ab95` lazily imports the
pooled pipeline after setting OpenMP, MKL, and OpenBLAS to one thread; freezes
runtime versions and thread limits at preparation; rejects drift before neural
loading; and stores validated software versions in result metadata. No new
dependency or shared execution framework was introduced.

The corrected gates produced `24 passed` for the focused result/pipeline tests,
`65 passed in 41.84s` for the complete pooled suite, and `943 passed, 27
warnings in 153.19s` for the full task-decoding suite. Focused Ruff and `git
diff --check` passed. The warning categories remained unchanged.

### 1.8 PW-WP6 CT026 dry-run evidence

The approved metadata-only dry-run used session
`CT026_2026-08-03_111938`, base configuration
`task_decoding_config_v3_full_conditions.json`, and session-local overlay
`pooled_task_decoding_config_pw6_pre_choice.json`. It selected only
`pre_choice`, all six conditions, all 18 targets, five outer folds, three
region configurations, and both representations. The scientific source was
clean. Its source fingerprint was
`31f015ed7f0ff16601d2e8e914e5228cc14bfccfdf447e193097c7c715479080`, and
the planned run fingerprint was
`e59243b033139f3628485174c7dea1ef1b665eabbf63787f67aa42db0cfc98b2`.

The exact dry-run envelope was 650 complete-table trials, 646 common bilateral
tensor trials, 160 PFC units, 309 HPC units, 40 evaluation bins, 1,440
categorical fits, 1,800 numerical fits, 1,080 regional transform fits, and
343,200 evaluation-metric cells. Tensor allocation was 96,951,680 bytes and
planned major-array allocation was 122,745,320 bytes. No spike array was
loaded, no estimator was fit, and no run directory or scientific result was
created. The next action is explicit user approval or rejection of the exact
experimental `new` execution; the dry-run does not authorize it.

### 1.9 PW-WP6 CT026 execution evidence and open review

The user approved one detached execution of the exact dry-run identity. It
created immutable run directory
`analysis_runs/pooled_task_variable_decoding_20261012T011808Z`. The detached
worker was not visible from a later sandbox PID namespace. Mistaking that
namespace isolation for process death, the operator removed its live guard and
started an attached resume. Both deterministic workers computed the first
target; the detached owner published the checkpoint first, and the attached
worker correctly failed rather than overwrite it. The original worker then
completed all 18 checkpoints, `results.npz`, the summary, and 54 figures. It
failed only at final byte accounting because its guard had been removed. One
exact checkpoint-only resume restored all 18 checkpoints and terminally
published the run with lifecycle `complete`, `final_results_published=true`,
resource status `complete`, and `result_valid=true`. No checkpoint or
scientific result was overwritten.

Predicted and requested counts reconciled exactly: 3,240 model cells, 1,080
transform cells, and 343,200 evaluation-metric cells. Of these, 3,090 models,
1,030 transforms, and 314,400 evaluation-metric cells were valid. There were
zero convergence warnings and zero invalid converged-model cells. The 150
unavailable models and 50 unavailable transforms were explicit
`training_class_unavailable` cells confined to `current_action_is_correct` in
`correct_rewarded`, `omission`, and `incorrect`, plus
`next_choice_switch_stay` in `switch` and `stay`. The 28,800 unavailable
evaluation cells were the corresponding training-class failures plus
destination test-class failures for `next_choice_switch_stay`. All numerical
models were valid.

The final validated output records 10,319,237 checkpoint bytes, 57,222,180
contract-counted output bytes, a 3,254,408-byte `results.npz`, and 43,518,826
bytes across 54 PNGs. Primary-worker log timestamps bound the original work
from 01:18:09Z through completed reporting at 01:45:28Z, or about 1,639 seconds;
the last checkpoint appeared about 1,604 seconds after start and reporting
then took about 35 seconds. These are filesystem/log elapsed measurements, not
the overwritten resource sidecar's stage timings.

The checkpoint-only terminal resume replaced the original worker's resource
sidecar. Therefore its saved 2.22-second wall time, 2.20 seconds of CPU, and
360,177,664-byte peak RSS describe finalization only, not the scientific
computation. The duplicate first-target process separately measured 203.87
seconds wall time, 203.83 seconds CPU, and 869,347,328-byte peak RSS while
loading the complete session tensors and computing one categorical target;
that peak is a measured lower bound for this execution path, not an exact
full-run peak. Exact full-run CPU time and peak RSS are unavailable. Obtaining
them requires a separately approved fresh immutable benchmark rather than
altering this completed run.

The 54 PNG files are structurally valid at 2,100 by 1,540 pixels. Visual
inspection found that transfer-figure legends overflow their panels and
collide with lower panels and the caption. Native and gap figures are readable,
but the immutable completed run retains its originally generated figures.

The approved test-first reporting correction used RED commit `3342579`, the
test-fixture correction `5049e47`, and GREEN commit `ed09fe7`. It replaces six
per-panel legends with one deduplicated legend in a reserved right column and
reserves a separate wrapped-caption band without changing any saved result,
curve, coordinate, metric, or scientific computation. The focused plotting
suite passed 4 tests, the pooled suite passed 68 tests, and the complete
task-decoding suite passed 946 tests with the same 27 warnings. Renderer-level
checks on all 54 actual CT026 layouts found every legend and caption contained
inside the figure and disjoint from every panel and from each other.

Six representative corrected figures were rendered from the validated saved
result outside the completed run. Recursive size, modification-time, and
SHA-256 snapshots confirmed that all 91 original run files were unchanged.
PW-WP6 remains awaiting user inspection of that preview and separate approval
for a fresh detached benchmark to recover exact full-run CPU and peak-RSS
telemetry. Do not enter later work packages or silently rerun CT026.

## 2. Fixed revision-1 decisions

- The reference training condition is exactly `correct_rewarded`.
- Canonical windows are `pre_choice` [-0.5 s, 0 s), `post_choice`
  [0 s, +0.5 s), and `full` [-2 s, +2 s).
- `pre_choice` is the default and first experimental benchmark.
- A run uses one explicit nonempty subset of those windows. Runtime never
  changes the selection.
- Every configured condition has a native pooled decoder evaluated across the
  complete time axis.
- The correct-rewarded decoder is evaluated on every configured destination.
- Native-minus-transfer gaps intentionally compare two complete predictive
  pipelines with separately fitted training-condition scaling, feature
  retention/PCA, regularization scale, and estimator coefficients. They test
  whole-pipeline transfer and must not be interpreted as isolated readout
  weight changes.
- One estimator is fit per window/condition/target/fold/region/representation,
  never per evaluation time or destination.
- Revision 1 uses fixed regularization only.
- Revision 1 introduces no new dependency.
- Every stacked row has sample weight `1 / n_training_bins`, so each trial has
  total weight one across any supported window. Every row from one trial
  remains in the same block-grouped fold.
- Trials, not stacked rows, are the independent sampling units.
- Revision 1 saves fold metrics and model/transform provenance, but not dense
  per-trial predictions.
- Revision 1 does not calculate a paired time-specific-minus-pooled metric.
  A later, read-only descriptive overlay may compare independently completed
  analyses without refitting either one.

## 3. Minimal architecture and reuse boundary

Keep the workflow in the small
`src/neural_analysis/task_decoding/pooled` subpackage with a separate analysis
identity. This prevents new pooled modules from entering the existing
decoder's top-level `task_decoding/*.py` fingerprint glob. Follow the current
config/modeling/results/pipeline/plotting separation, but do not add an
independent split module, batch module, web-app route, or generic framework
before a concrete need is demonstrated.

### 3.1 New files required for the validated single-session workflow

| File | Purpose |
| --- | --- |
| `pooled/__init__.py` | Mark the package boundary; contain no scientific logic. |
| `pooled/config.py` | Validate a small pooled-analysis overlay and resolve the existing base task-decoding configuration. |
| `pooled/decoding.py` | Assemble pooled features, fit one fixed model, evaluate it through time, and construct native/transfer/gap arrays. |
| `pooled/results.py` | Define, save, load, fingerprint, and strictly validate schema-1 checkpoints and final results. |
| `pooled/pipeline.py` | Plan, prepare, execute, resume, and inspect one session while reusing current lifecycle helpers where extraction is safe. |
| `pooled/plotting.py` | Load saved pooled results and create the required native, transfer, and gap figures. |
| `pooled/run_session.py` | Thin `dry-run`, `new`, `resume`, and `status` CLI; no scientific logic. |

Six focused modules plus the package initializer are justified by separate
data contracts. Do not create
classes that only wrap functions, introduce a shared plugin system, or refactor
the current decoder speculatively.

### 3.2 Existing code reused unchanged where possible

- `config.py`: `TaskDecodingConfig`, target/condition constants, frozen
  estimator controls, and `load_task_decoding_config`.
- `targets.py`: `validate_augmented_trials` and `build_target_table`.
- `conditions.py`: `build_condition_masks`.
- `activity.py`: coverage inspection, memory estimates, region loading,
  `build_session_rate_tensors`, and target-mask projection.
- `modeling.py`: `GroupedSplit`, `SplitPlan`, `make_outer_splits`,
  `RegionTransform`, `fit_region_transform`, `build_region_features`,
  `make_estimator`, `classification_metrics`, and `regression_r2`. PW-WP2
  first extracts a small fitted-estimator/evaluation seam from the existing
  `fit_and_score` implementation because pooled evaluation must reuse one fit
  many times. Existing `fit_and_score` delegates to that seam and retains its
  current public behavior and byte contracts.
- `results.py`: input-manifest helpers and current top-level fingerprint
  enumeration, not its existing result schema. Narrow its cleanliness check
  from every descendant Python path to the same direct top-level paths, so a
  dirty nested pooled module neither changes nor blocks the existing analysis;
  protect that one-line scope correction with regression tests.
- `resource_usage.py`: retain the current schema, validator, and tracker
  unchanged. The pooled pipeline may reuse the public RSS-normalization helper,
  but its lean sidecar contract remains local to the pooled workflow.
- Current execution-guard, detached-launch, atomic-write, and status behavior
  may be extracted into small shared helpers only if focused regression tests
  first prove that the existing public commands and byte contracts are
  unchanged.

The current private `pipeline.py` functions are not a public reuse contract.
Copying a large private function or making the existing pipeline branch on an
analysis mode is disallowed. A small shared extraction is allowed only when it
removes literal duplication and preserves both callers clearly.

### 3.3 Explicitly deferred files and features

- No `pooled_decoding_splits.py`; use `modeling.make_outer_splits`.
- No pooled batch or Slurm code before PW-WP8.
- No web-app integration in revision 1.
- No automatic loader for current per-time-bin results before PW-WP7.
- No tuned regularization, train-time x test-time matrix, arbitrary windows,
  inferential statistics, or within-session multiprocessing.

PW-WP5 also adds the portable overlay
`docs/examples/neural_analysis/pooled_task_decoding_config.json`, whose
`base_config_path` names the sibling base example with all six conditions and
all 18 current targets explicit, and updates
`src/neural_analysis/task_decoding/README.md` with
the pooled single-session commands and result-location contract. These are
required usage deliverables, not new architecture.

## 4. Configuration contract

Avoid duplicating the established task-decoding parser. A pooled configuration
contains exactly:

```json
{
  "base_config_path": "task_decoding_config.json",
  "training_window_names": ["pre_choice"]
}
```

`base_config_path` is required and resolves relative to the pooled config.
The referenced base configuration supplies session/input paths, selected
conditions and targets, bin width, PCA counts, fixed estimator controls, fold
count, trusted bounds, and output root. It must have `alignment ==
"choice_time"` and `regularization_mode == "fixed"`.

Omitting `training_window_names` selects `pre_choice`. Reject unknown top-level
fields, missing/invalid base configurations, empty/duplicate/unknown windows,
and windows whose edges do not coincide with the base time-bin edges. Select a
bin only when its complete half-open `[edge_i, edge_{i+1})` interval is contained
in the named half-open window. Restore accepted windows to canonical order.

`PooledDecodingConfig` is a frozen dataclass with:

```text
base_config: TaskDecodingConfig
base_config_path: pathlib.Path
training_window_names: tuple[str, ...]
condition_names: tuple[str, ...]
```

`condition_names` is the base condition subset with `correct_rewarded` added
in canonical order when absent. The existing base default remains `all`; it is
not reinterpreted as all six conditions. The complete pooled example must list
all six explicitly in the base configuration.

Its public functions are:

```text
load_pooled_decoding_config(path: Path | str) -> PooledDecodingConfig
resolve_training_windows(config: PooledDecodingConfig,
                         time_bin_edges_s: np.ndarray) -> tuple[TrainingWindow, ...]
scientific_pooled_config_payload(
    config: PooledDecodingConfig,
    training_windows: Sequence[TrainingWindow],
) -> dict[str, object]
load_prepared_pooled_decoding_config(
    run_directory: Path | str,
) -> tuple[PooledDecodingConfig, tuple[TrainingWindow, ...]]
```

`TrainingWindow` records `name`, `(start_s, stop_s)`, and a one-dimensional
ascending integer `bin_indices` array. Bounds and time axes are seconds
relative to choice. The scientific payload embeds the canonical base
scientific payload, effective `condition_names`, fixed
`reference_condition`, and resolved window definitions; a change to any of
them changes the run fingerprint. Planning derives the canonical time-bin
edges from the validated base bounds and bin width, resolves windows first,
and passes those immutable definitions to the serializer. Execution validates
the loaded tensor axes against them.

Canonical resolved configuration content, not JSON formatting or the original
configuration pathname, determines scientific identity. Preparation copies
the submitted overlay and referenced base JSON to
`pooled_config.input.json` and `base_config.input.json` for audit, even when
either source lies outside the session root. These byte copies are not
scientific inputs and are not added to the session-contained input manifest;
execution and resume use the immutable resolved `config.json`, not mutable
external configuration files.

Preparation stores the absolute resolved session root under the exact
`execution.json` key `session_root` as execution-only provenance; it is
excluded from scientific/config identity.
`load_prepared_pooled_decoding_config` reads only `config.json`,
`execution.json`, and the run-directory location. It reconstructs every
session-relative scientific path below that saved root, reconstructs
`output_root` as `run_directory.parent`, uses
`run_directory/base_config.input.json` only as the inert `base_config_path`
audit location, rebuilds `TaskDecodingConfig`, `PooledDecodingConfig`, and the
resolved `TrainingWindow` records, and validates all frozen controls and
canonical ordering. It never parses either audit JSON copy. Missing or
nonabsolute roots, path escape, payload drift, and invalid reconstructed
configuration fail before neural data are loaded.

After reconstruction, `run_prepared_pooled_decoding` recomputes and validates
the saved input manifest against the reconstructed absolute paths before
loading large arrays or accepting a checkpoint. Thus moving or formatting the
original submitted configurations cannot affect resume, while moved, missing,
or changed scientific inputs are rejected.

## 5. Scientific core API and data flow

Use functions and small immutable records. Each production function must have
a docstring stating input types, shapes, axis order, units, returns, and
scientific unavailable states.

### 5.1 Public scientific functions

On the full chronological axis, construct
`master_eligible = baseline_eligible & target_eligible &
bilateral_full_window_coverage`. Preserve `target_eligible` separately. Assign
the one saved ineligibility reason with exact precedence
`baseline_ineligible`, then `target_ineligible`, then
`neural_window_unavailable`; an eligible row has reason index `-1`.
`trial_row_indices` contains exactly the baseline-valid rows with bilateral
coverage, so only target eligibility can exclude a row after projection to
the common tensor axis.

```text
make_master_outer_split(target_values: np.ndarray,
                        block_ids: np.ndarray,
                        master_eligible: np.ndarray,
                        *, target_family: str,
                        fold_count: int) -> modeling.SplitPlan
```

Inputs are one-dimensional arrays on the common bilateral neural-tensor row
axis `(M,)`, after the pipeline projects full-table values through
`SessionRateTensors.trial_row_indices`. The function calls the existing grouped
splitter on eligible tensor rows and projects its fold IDs back to length `M`
with `-1` for target-ineligible tensor rows. Every split index is a tensor-row
position. It never regenerates folds after condition filtering.

```text
stack_training_window(feature_tensor: np.ndarray,
                      trial_targets: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

`feature_tensor` has shape `(training_trial, training_time, feature)`;
`trial_targets` has shape `(training_trial,)`. Return features with shape
`(training_trial * training_time, feature)`, repeated targets with shape
`(training_trial * training_time,)`, and float64 sample weights of the same
length. Ordering is trial-major then time-major. Every weight is
`1 / training_time`, so weights sum to one within each trial. Empty axes,
nonfinite values, or misaligned shapes raise `ValueError`.

```text
fit_pooled_transforms(
    pfc_training_rates_hz: np.ndarray,
    hpc_training_rates_hz: np.ndarray,
    training_bin_indices: np.ndarray,
    *,
    pfc_pc_count: int,
    hpc_pc_count: int,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
) -> PooledTransforms
```

Both rate tensors are finite Hz arrays shaped `(training_trial,
evaluation_time, regional_unit)` on the same trial/time axes; selected bin
indices are distinct, ascending, and in range. The function fits exactly one
PFC and one HPC training-only transform for the current
window/condition/target/fold cell. `PooledTransforms` is frozen and records
both transform states, stable unit identities, requested/effective counts,
and declared compact availability codes. It is reused by all three region
configurations and both representations.

PW-WP2 first adds these minimal reusable seams to `modeling.py`:

```text
fit_estimator(estimator: object,
              train_features: np.ndarray,
              train_targets: np.ndarray,
              *, target_family: str,
              sample_weight: np.ndarray | None = None) -> FittedEstimator
evaluate_fitted_estimator(fitted: FittedEstimator,
                          held_out_features: np.ndarray,
                          held_out_targets: np.ndarray,
                          *, target_family: str) -> FitEvaluation
```

Fit inputs are finite `(training_row, feature)` float matrices and aligned
`(training_row,)` targets. A supplied weight vector is finite, nonnegative,
one-dimensional, aligned to training rows, and has positive sum; it is passed
unchanged to exactly one estimator `fit` call. When it is `None`, the helper
omits the keyword so legacy estimator call behavior is unchanged. Evaluation
inputs are a finite `(held_out_trial, feature)` matrix with the fitted feature
width and an aligned `(held_out_trial,)` target vector.

`modeling.py` owns closed `FitFailureKind` and `EvaluationFailureKind` string
enums. The exact fit members are `convergence_warning`,
`invalid_coefficient_or_intercept_shape`,
`nonfinite_coefficient_or_intercept`, `invalid_class_structure`, and
`ambiguous_positive_class`. The exact evaluation members are
`invalid_probability_shape`, `nonfinite_probability`,
`positive_score_alignment_invalid`,
`categorical_test_classes_invalid`, `invalid_numerical_prediction`, and
`invalid_regression_metric`. These fine-grained semantic states distinguish
every reviewed branch in current `fit_and_score`; they are not serialized
pooled reason codes.

`FittedEstimator` is a frozen runtime container with the sklearn estimator,
coefficients, intercept, class orientation, convergence status, and optional
`FitFailureKind`; unavailable fits have no estimator. `FitEvaluation` carries
metrics, the categorical positive-score vector/class when applicable, or one
`EvaluationFailureKind`. The evaluation helper predicts and scores without
fitting. Existing `fit_and_score` delegates to these two functions with
`sample_weight=None`; its adapter retains the current up-front validation of
both partitions and their shared feature width before any estimator call. It
maps each kind through this exhaustive legacy table:

| Failure member | Exact legacy `FitResult.reason` |
| --- | --- |
| `convergence_warning` | `convergence_warning` |
| `invalid_coefficient_or_intercept_shape` | `invalid coefficient or intercept shape` |
| `nonfinite_coefficient_or_intercept` | `non-finite coefficient or intercept` |
| `invalid_class_structure` | `categorical estimator must expose binary classes including positive class 1` |
| `invalid_probability_shape` | `invalid probability shape` |
| `nonfinite_probability` | `non-finite probability or score` |
| `ambiguous_positive_class` | `positive class is ambiguous` |
| `positive_score_alignment_invalid` | `positive_scores must be finite and align with target_values.` |
| `categorical_test_classes_invalid` | `Categorical metrics require held-out values from both classes.` |
| `invalid_numerical_prediction` | `non-finite prediction or score` |
| `invalid_regression_metric` | `constant, singleton, or non-finite held-out numerical target` |

The adapter also preserves the current invalid NaN fields and convergence
status. The pooled adapter maps the same kinds through the separate exhaustive
table in Section 6.4. Unknown enum values fail rather than falling through.
This avoids prose parsing and a dependency from generic `modeling.py` back to
pooled modules. Unexpected estimator or programming exceptions still
propagate.

```text
fit_pooled_model(
    transforms: PooledTransforms,
    pfc_training_rates_hz: np.ndarray,
    hpc_training_rates_hz: np.ndarray,
    training_targets: np.ndarray,
    training_bin_indices: np.ndarray,
    *,
    target_identifier: str,
    target_family: str,
    region_configuration: str,
    representation: str,
) -> FittedPooledModel
```

The function applies the already-fitted transforms, builds one requested
region/representation tensor, stacks it, passes the returned sample weights to
`modeling.fit_estimator`, and constructs a durable `PooledModelRecord` with
coefficients, intercept, independent-trial and stacked-row counts, feature
identities, estimator parameters, status, and compact reason. It never fits a
transform and never scores evaluation data.

```text
evaluate_pooled_model(
    model: FittedPooledModel,
    pfc_test_rates_hz: np.ndarray,
    hpc_test_rates_hz: np.ndarray,
    test_targets: np.ndarray,
    evaluation_time_centers_s: np.ndarray,
) -> PooledEvaluationResult
```

Held-out tensors have shape `(test_trial, evaluation_time, regional_unit)` in
Hz; targets have shape `(test_trial,)`; centers have shape
`(evaluation_time,)` in choice-relative seconds. The fitted transform and
estimator are applied unchanged at each time. The immutable return contains
dense `(evaluation_time, metric)` status, reason, and score arrays plus the
single test-trial count and categorical class counts when applicable. It does
not repeat test indices per time or metric. Pair/fold membership is exactly
reconstructable from saved condition masks, target fold IDs, and
`trial_row_indices`. The function never refits, recenters, rescales, or
recalibrates.

```text
decode_pooled_window_target(
    *,
    training_window: TrainingWindow,
    target_identifier: str,
    target_family: str,
    target_values: np.ndarray,
    master_eligible: np.ndarray,
    block_ids: np.ndarray,
    condition_masks: Mapping[str, np.ndarray],
    outer_split: modeling.SplitPlan,
    pfc_rate_tensor_hz: np.ndarray,
    hpc_rate_tensor_hz: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
    evaluation_time_centers_s: np.ndarray,
    pfc_pc_count: int,
    hpc_pc_count: int,
) -> PooledWindowTargetResult
```

The one-dimensional target, eligibility, block, and condition arrays have
shape `(M,)`, and both Hz tensors have shape
`(M, evaluation_time, regional_unit)` on exactly the same common tensor-row
axis. This function handles one window-target checkpoint cell, orchestrates
every native model, reuses each correct-rewarded model for all destination
evaluations, and computes fold-level native-minus-transfer gaps only from
scores on identical destination rows. `PooledWindowTargetResult` contains the
complete dense model, evaluation, gap, transform-status/count, and fold
arrays for that cell. It does not materialize a Python record for every
time/metric cell. Shape/identity violations raise; declared scientific
insufficiency remains in the returned result. The pipeline retains the
full-table `(N,)` audit arrays and `trial_row_indices (M,)` separately for
final assembly.

### 5.2 Required computation order

For each target:

1. Construct full-table target/eligibility/condition audit arrays, then project
   target values, valid masks, conditions, and block identities through
   `trial_row_indices` onto the common tensor-row axis.
2. Construct one target-specific master block-fold map on that tensor-row axis.
3. For each training window, condition, and fold, select outer-training tensor
   rows.
4. Fit one PFC and one HPC transform from those trials and selected window bins.
5. Reuse those frozen transforms to build all three region configurations and
   both representations.
6. Stack equal-count bins with `1 / n_training_bins` row weights and fit one
   estimator for each region/representation.
7. Evaluate the frozen native model at every time in its own held-out condition.
8. Reuse the correct-rewarded model on each held-out destination condition.
9. Score native and reference predictions on exactly the same destination
   tensor rows and compute native-minus-transfer gaps. The valid
   correct-rewarded self-gap is exactly zero.
10. Save the complete `PooledWindowTargetResult` checkpoint, then continue to
    the next window-target cell.
11. Preserve scientific unavailable cells; propagate programming, schema, and
    I/O errors.

`correct_rewarded -> correct_rewarded` is fit and stored once. Adding a
destination adds evaluation calls, not reference fits.

Before invalidities, the exact regional-transform fit count is
`W * C * T * F * 2`: two anatomical regions per
window/condition/target/fold, independent of region configuration,
representation, evaluation time, and destination.

## 6. Frozen result and checkpoint contract

Use pooled analysis version `pooled-task-variable-decoding-v1` and result
schema version `1`. Final output is `results.npz` plus the same small JSON
sidecar categories used by the existing workflow. NPZ loading always uses
`allow_pickle=False`.

### 6.1 Run-directory layout

```text
analysis_runs/
  pooled_task_variable_decoding_<UTC timestamp>/
    config.json
    pooled_config.input.json     # exact submitted overlay; audit only
    base_config.input.json       # exact referenced base JSON; audit only
    input_manifest.json
    scientific_source.json
    execution.json
    run_state.json
    resume_command.txt
    status_command.txt
    run.log
    console.log                 # detached local runs only
    local_launch.json           # latest detached launch receipt only
    execution_guard.json        # present only while an invocation owns the run
    resource_usage.json
    results.npz                 # present only after atomic final publication
    summary.md                  # present only for a complete run
    run_session.py              # immutable launch-module snapshot
    pipeline.py                 # immutable main-analysis snapshot
    plotting.py                 # immutable reporting-module snapshot
    checkpoints/
      <window>__<target>.npz
    figures/
      native__<window>__<target>.png
      transfer__<window>__<target>.png
      gap__<window>__<target>.png
```

`config.json` is the canonical resolved scientific payload and embeds the
resolved base payload. The two `*.input.json` files preserve exact submitted
bytes for audit but do not determine identity and are not reread during
execution. A checkpoint key is exactly `<training_window>::<target>`. Filename
labels use the existing strict filename-safe identifier rule. One checkpoint
is complete only when it holds all configured conditions, folds, regions,
representations, evaluation pairs, and times for that cell.

Required figure count is `3*W*T`: 54 PNGs for the first one-window/18-target
benchmark and 162 for all three windows. More granular views are optional and
generated only on request.

The launch, main, and plotting source snapshots are copied at preparation,
hashed and named in execution metadata, and validated byte-for-byte before
execution, resume, and complete reentry. PW-WP8 defines the batch run-directory
additions only when the batch launcher exists; the single-session schema
contains no future placeholder.

### 6.2 Coordinate arrays

The result declares these axes and coordinate arrays:

| Axis | Coordinate | Meaning |
| --- | --- | --- |
| `window` (`W`) | `training_window_names` | Canonical selected windows |
| `condition` (`C`) | `condition_names` | Configured native/destination conditions; reference is always present |
| `target` (`T`) | `target_names` | Configured target order |
| `fold` (`F`) | `fold_ids` | Zero-based requested outer folds |
| `region` (`R=3`) | `region_names` | `PFC`, `HPC`, `PFC+HPC` |
| `source_region` (`A=2`) | `source_region_names` | Anatomical transform order `PFC`, `HPC` |
| `representation` (`P=2`) | `representation_names` | `pca`, `units` |
| `time` (`Q`) | `evaluation_time_centers_s` | Choice-relative seconds |
| `metric` (`K=3`) | `metric_names` | `balanced_accuracy`, `auc`, `r2` |
| `pair` (`E=2C-1`) | `evaluation_train_condition`, `evaluation_test_condition` | Unique reference-transfer and native pairs |
| `trial` (`N`) | existing safe trial identity arrays | Complete chronological trial table |
| `tensor_trial` (`M`) | `trial_row_indices` | Common bilateral neural-tensor rows mapped to `trial` |
| `feature` (`G`) | padded model-feature slot | `G=max(1, selected PFC units + selected HPC units, requested PFC PCs + requested HPC PCs)` |

UTF coordinate widths are derived before allocation and validated without
truncation. Selected source-unit counts are available from dry-run metadata,
so `G` and exact result bytes do not depend on fitting or large neural-array
loading. Fold-specific feature removal or reduced PCA counts are represented
by `model_feature_active`; they never resize `G`. Time and rate metadata state
seconds and Hz explicitly.

The NPZ also contains the one-dimensional coordinate arrays
`status_names`, `reason_code_names`, and `convergence_status_names` in the
orders declared below. Pair coordinates are the two length-`E` arrays
`evaluation_train_condition` and `evaluation_test_condition`. Pair order is
all `correct_rewarded -> destination` pairs in `condition_names` order,
followed by every non-reference `condition -> itself` pair in that same order.
The duplicate reference self-pair is omitted.

### 6.3 Required scientific arrays

Status arrays are `uint8`: `0=not_applicable`, `1=valid`, and `2=unavailable`.
Reason arrays are `int16` indices into `reason_code_names`; `-1` means no
reason. Scores, target values, coefficients, intercepts, and time coordinates
are `float64`; unavailable or padded float entries are NaN. Counts are
`int32`, masks are Boolean, fold maps are `int16`, and trial positions/row
indices are `int64`; integer padding is `-1` unless an array says otherwise.
Convergence arrays are `uint8` indices into `not_run`, `converged`, `warning`,
and `invalid` in that order. Categorical positive-class coordinates are
`int8`. String-coordinate Unicode widths are derived and validated before
allocation.

| Array | Shape | Contract |
| --- | --- | --- |
| `target_values` | `(T,N)` | Native target units; NaN only where target-ineligible |
| `target_eligible` | `(T,N)` | Target-specific eligibility |
| `master_eligible` | `(T,N)` | Baseline eligibility AND target eligibility AND bilateral full-window coverage |
| `eligibility_reason_index` | `(T,N)` | First failing reason in baseline, target, neural precedence; `-1` when eligible |
| `condition_masks` | `(C,N)` | Full-table canonical masks |
| `trial_row_indices` | `(M,)` | Full-table positions with baseline eligibility and bilateral full-window coverage |
| `outer_fold_ids` | `(T,M)` | Fold IDs or `-1` for target-ineligible tensor rows |
| `training_window_bin_mask` | `(W,Q)` | Exact selected training bins |
| `transform_status` | `(W,C,T,F,A)` | One shared regional-transform availability state |
| `transform_reason_index` | `(W,C,T,F,A)` | Strict compact transform reason code |
| `transform_input_unit_count` | `(W,C,T,F,A)` | Selected source units before training-only removal |
| `transform_usable_unit_count` | `(W,C,T,F,A)` | Nonconstant finite training units |
| `transform_component_limit` | `(W,C,T,F,A)` | Deterministic PCA component limit |
| `transform_requested_pc_count` | `(W,C,T,F,A)` | Configured regional PC count |
| `transform_effective_pc_count` | `(W,C,T,F,A)` | Fitted retained PC count |
| `model_status` | `(W,C,T,F,R,P)` | Model availability |
| `model_reason_index` | `(W,C,T,F,R,P)` | Strict compact invalidity code |
| `model_train_trial_count` | `(W,C,T,F,R,P)` | Independent training trials |
| `model_stacked_row_count` | `(W,C,T,F,R,P)` | Trials times selected bins |
| `model_requested_feature_count` | `(W,C,T,F,R,P)` | Width before training-only feature removal |
| `model_feature_count` | `(W,C,T,F,R,P)` | Effective fitted width |
| `model_convergence_status` | `(W,C,T,F,R,P)` | Index into declared convergence statuses |
| `model_intercept` | `(W,C,T,F,R,P)` | Fitted estimator intercept |
| `model_coefficients` | `(W,C,T,F,R,P,G)` | Coefficients on saved feature scale |
| `model_feature_active` | `(W,C,T,F,R,P,G)` | Features present in the fitted model |
| `evaluation_status` | `(W,E,T,F,R,P,Q,K)` | Fold-level metric status |
| `evaluation_reason_index` | `(W,E,T,F,R,P,Q,K)` | Strict compact reason code |
| `evaluation_scores` | `(W,E,T,F,R,P,Q,K)` | Fold-level held-out metrics |
| `evaluation_test_trial_count` | `(W,E,T,F)` | Destination test-trial count |
| `gap_status` | `(W,C,T,F,R,P,Q,K)` | Native-minus-transfer availability |
| `gap_reason_index` | `(W,C,T,F,R,P,Q,K)` | `source_score_unavailable` or no reason |
| `gap_scores` | `(W,C,T,F,R,P,Q,K)` | Native minus reference-transfer score |

The following non-score arrays are also required:

| Array | Shape | Contract |
| --- | --- | --- |
| `time_bin_edges_s` | `(Q+1,)` | Choice-relative seconds |
| `training_window_start_s`, `training_window_stop_s` | `(W,)` | Exact half-open bounds in seconds |
| `trial_positions` | `(N,)`, `int64` | Complete-table row positions |
| `trial_ids` | `(N,)`, `int64` | Stable complete-table trial IDs |
| `behavioral_block_payload` | `(B,)`, `uint8` | Packed exact behavioral-block identity bytes |
| `behavioral_block_offsets` | `(N+1,)`, `int64` | Byte offsets delimiting each trial's block identity |
| `target_family_names`, `target_unit_names` | `(T,)`, `(T,)` | Family and physical/dimensionless target units |
| `target_positive_class` | `(T,)` | `1` for categorical targets, `-1` for numerical targets |
| `coefficient_feature_ids` | `(R,P,G)` | Stable direct-unit or regional-PC identity per padded feature slot |
| `coefficient_feature_regions` | `(R,P,G)` | Anatomical source of each feature slot |
| `coefficient_feature_present` | `(R,P,G)` | Non-padding feature slots in the run-wide union |
| `unit_selection_rules_json` | `(2,)` | Canonical PFC/HPC source-selection provenance |
| `model_train_class_counts` | `(W,C,T,F,2)` | Class 0/1 counts, `-1` for numerical targets |
| `evaluation_test_class_counts` | `(W,E,T,F,2)` | Class 0/1 counts, `-1` for numerical targets |
| `target_estimator_class_names` | `(T,)` | Concrete fixed estimator class |
| `target_estimator_parameters_json` | `(T,)` | Canonical fixed-control JSON |

`B` is the packed behavioral-block payload byte count, not a scientific
coordinate or result axis.

Metadata records software versions, random seed, source fingerprint, run
fingerprint, input manifest identity, stable regional source-unit identities,
coefficient scales/units, the fixed `1 / n_training_bins` weighting policy,
and the exact axis list for every array. Fitted
scaler/PCA matrices are not saved: immutable inputs, configuration, folds,
source fingerprint, selection rules, and model records are the reproduction
contract, matching the current task decoder's level of persisted transform
provenance. `pooled/results.py` owns a pooled fingerprint function with an
explicit fixed scientific source list: `pooled/__init__.py`,
`pooled/config.py`, `pooled/decoding.py`, `pooled/results.py`, and
`pooled/pipeline.py`; the parent initializer
`src/neural_analysis/task_decoding/__init__.py`; the reused top-level
`activity.py`, `conditions.py`, `config.py`, `modeling.py`, `results.py`, and
`targets.py`; the condition-mask implementation
`src/neural_analysis/spike_behavior/trials.py`; and every path in the existing
`_DIRECT_RUNTIME_PATHS`. The parent initializer is explicit because importing
the pooled package executes it. `trials.py` is explicit because
`conditions.py` imports `make_trial_type_masks` from it and the existing direct
dependency list does not include it. Reporting-only `pooled/plotting.py` and
the thin `pooled/run_session.py` CLI are excluded from scientific identity but
are hashed and snapshotted in execution provenance; telemetry-only
`resource_usage.py` remains outside scientific identity. The fingerprint
function does not glob the package. An unrelated new file therefore cannot
silently enter either analysis identity.
The existing top-level fingerprint function remains unchanged and excludes
the nested pooled package by construction. Its companion cleanliness check is
made consistent with that direct top-level scope; the pooled validator
separately requires every explicitly listed pooled dependency to be clean and
tracked. The existing identity correctly changes if a reused top-level
scientific file such as `modeling.py` changes. The array names, dtypes, and
axes in this section are exact; implementations must not factor, rename, or
omit them because values happen to repeat across axes.

Do not save per-trial predictions in revision 1. AUC is computed in memory
from held-out positive-class probabilities. Reproduction is provided by fold
identities, coefficients, intercepts, estimator controls, source/config
fingerprints, and immutable inputs.

### 6.4 Compact unavailable vocabulary

Only the following codes may be serialized in revision 1:

```text
baseline_ineligible
target_ineligible
neural_window_unavailable
target_outer_unavailable
empty_training_subset
empty_test_subset
training_class_unavailable
test_class_unavailable
training_target_unavailable
test_target_unavailable
pfc_transform_unavailable
hpc_transform_unavailable
regional_transform_unavailable
no_features
fit_convergence_failure
outer_estimator_failure
invalid_probability
invalid_prediction
invalid_metric
source_score_unavailable
```

`pooled.decoding.POOLED_REASON_CODES` is the sole ordered vocabulary used by
producers and validators; `reason_code_names` serializes that exact order.

Pooled scientific functions choose these codes at the decision point or map a
closed modeling failure enum through the exact adapter below. They never infer
codes from substrings or arbitrary exception prose. The required producer
contract is:

| Decision point | Allowed compact code |
| --- | --- |
| Full-table baseline/target construction | `baseline_ineligible`, `target_ineligible` |
| Bilateral coverage projection | `neural_window_unavailable` |
| Reviewed unavailable `SplitPlan` | `target_outer_unavailable` |
| Condition/fold subset validation | `empty_training_subset`, `empty_test_subset`, `training_class_unavailable`, `test_class_unavailable`, `training_target_unavailable`, `test_target_unavailable` |
| Standalone regional transform | `pfc_transform_unavailable`, `hpc_transform_unavailable` |
| Combined regional feature construction | `regional_transform_unavailable`, `no_features` |
| Estimator fit validation | `fit_convergence_failure`, `outer_estimator_failure` |
| Prediction/metric validation | `invalid_probability`, `invalid_prediction`, `invalid_metric` |
| Paired gap construction | `source_score_unavailable` |

The exhaustive modeling-to-pooled adapter is:

| Modeling failure member | Pooled code |
| --- | --- |
| `convergence_warning` | `fit_convergence_failure` |
| `invalid_coefficient_or_intercept_shape`, `nonfinite_coefficient_or_intercept`, `invalid_class_structure`, `ambiguous_positive_class` | `outer_estimator_failure` |
| `invalid_probability_shape`, `nonfinite_probability` | `invalid_probability` |
| `positive_score_alignment_invalid`, `categorical_test_classes_invalid`, `invalid_regression_metric` | `invalid_metric` |
| `invalid_numerical_prediction` | `invalid_prediction` |

`make_master_outer_split` uses the existing `SplitPlan.is_available` state and
emits `target_outer_unavailable` without parsing its human-readable detail.
`fit_pooled_transforms` similarly uses reviewed transform availability and the
known anatomical context, not reason prose. The pooled estimator adapter owns
the table above; generic `modeling.py` does not import pooled code. The legacy
adapter separately maps every enum member to the exact current
human-readable `fit_and_score` reason and preserves its invalid coefficient,
intercept, metric, positive-score, and convergence outputs. Unknown enum
members, strings, states, exceptions, schema changes, and I/O errors fail the
run instead of being relabeled unavailable. The validator rejects undeclared
or over-width codes; every declared code must fit NumPy `U32` without
truncation.

### 6.5 Exact status and reason propagation

The status values have one interpretation everywhere. `valid` requires a
reason index of `-1`. `unavailable` requires one declared reason. Float output
for an unavailable cell is NaN and its active-feature mask is false.
`not_applicable` is used only for metric-family mismatch: `r2` for categorical
targets and `balanced_accuracy`/`auc` for numerical targets. Those metric cells
in both evaluation and gap arrays have reason `-1` and NaN scores; transform
and model cells are never `not_applicable`. The propagation table below
therefore applies only to metrics applicable to the target family.

Counts use observed values whenever the relevant subset or operation is
known. An observed empty subset is count zero. A count that cannot be known
because an upstream stage was not reached is `-1`, never zero. Requested
regional unit/PC counts and training/test trial counts are populated whenever
their defining inputs are available. For a transform not attempted,
usable-unit, component-limit, and effective-PC counts are `-1`; after an
attempt, its observed counts are retained even if the transform is
unavailable. Model stacked-row and effective-feature counts are `-1` until
those stages are reached. Categorical class counts use observed class-0/class-1
counts; numerical class-count arrays are always `(-1, -1)`. Model convergence
is `not_run` before an estimator attempt, `converged` for a valid fit,
`warning` for `fit_convergence_failure`, and `invalid` for another attempted
but unavailable fit.

Numerical training and test subsets are valid only with at least two
independent finite trials and at least two distinct finite target values.
Categorical training and test subsets are valid only when nonempty and both
classes 0 and 1 are present. The following table is exhaustive:

| Earliest outcome | Transform cell | Dependent model cell | Applicable evaluation cells | Gap cell |
| --- | --- | --- | --- | --- |
| Target outer split unavailable | `unavailable`, `target_outer_unavailable` | copy the same status/reason | copy the same status/reason | `unavailable`, `source_score_unavailable` |
| Training subset empty or target-invalid before transform fitting | `unavailable`, exact training-subset reason | copy the same status/reason | copy the same status/reason | `unavailable`, `source_score_unavailable` |
| One regional transform unavailable | failing anatomical transform uses its exact PFC/HPC code; the other retains its own state | PFC/HPC-only models copy their source code; a combined model uses `regional_transform_unavailable` | copy the model status/reason | `unavailable`, `source_score_unavailable` |
| Transforms valid but requested model has no usable features | `valid`, no reason | `unavailable`, `no_features` | copy `no_features` | `unavailable`, `source_score_unavailable` |
| Estimator fit unavailable | `valid`, no reason | `unavailable`, exact mapped fit code | copy the model status/reason | `unavailable`, `source_score_unavailable` |
| Model valid but destination test subset empty or target-invalid | unchanged | `valid`, no reason | `unavailable`, exact test-subset reason | `unavailable`, `source_score_unavailable` |
| Prediction or metric unavailable | unchanged | `valid`, no reason | `unavailable`, exact mapped evaluation code | `unavailable`, `source_score_unavailable` |
| Score valid | `valid`, no reason | `valid`, no reason | `valid`, no reason | `valid`, no reason if both source scores are valid |

Upstream reasons are copied without reinterpretation except for the two
explicit aggregation boundaries in the table: combined-region transform
failure becomes `regional_transform_unavailable`, and any missing gap source
becomes `source_score_unavailable`. Evaluation cells inherit the exact model
reason when the model is unavailable. A valid correct-rewarded self-gap is
exactly zero; if its shared source evaluation is unavailable, it follows the
ordinary unavailable-gap rule.

### 6.6 Persistence and lifecycle APIs

```text
save_pooled_checkpoint(path: Path, *, checkpoint_key: str,
                       run_fingerprint: str,
                       result: PooledWindowTargetResult) -> None
load_pooled_checkpoint(path: Path, *, expected_checkpoint_key: str,
                       expected_run_fingerprint: str) -> PooledWindowTargetResult
save_pooled_run(run_directory: Path, *, arrays: Mapping[str, np.ndarray],
                metadata: Mapping[str, object]) -> Path
load_pooled_run(run_directory: Path, *,
                expected_run_fingerprint: str | None = None
                ) -> dict[str, object]
```

Savers validate before one atomic publication. Loaders use
`allow_pickle=False`, validate the complete schema, and return no fitted
scikit-learn objects.

The scientific stage also exposes one synchronous computation API used by the
synthetic gate before lifecycle, detached execution, or plotting is
implemented:

```text
compute_or_resume_pooled_results(
    config: PooledDecodingConfig,
    training_windows: Sequence[TrainingWindow],
    work_directory: Path,
) -> pathlib.Path
```

It loads the configured session inputs, writes or restores fingerprinted
window-target checkpoints under the supplied working directory, and atomically
publishes validated `results.npz`. It does not create terminal lifecycle state,
reports, or figures. The later full lifecycle calls this same function; there
is no second scientific path.

```text
plan_pooled_decoding_session(config_path: Path | str) -> dict[str, object]
prepare_pooled_decoding_run(config_path: Path | str) -> pathlib.Path
run_prepared_pooled_decoding(run_directory: Path | str) -> None
inspect_pooled_decoding_status(run_directory: Path | str) -> dict[str, object]
```

Planning reads small metadata and reports exact dimensions/resources without
loading spikes or fitting. Preparation writes the immutable identity and
lifecycle sidecars. `run_prepared_pooled_decoding` calls the synchronous
computation API, then adds reporting and terminal publication; it is not a
second compute path. Those two execution functions are the only functions that
load full neural arrays or fit. Status reads only small sidecars and validates
a completed result without mutating it.
`pooled.run_session.main(argv)` only parses CLI arguments and delegates to
these functions.

### 6.7 Lean pooled resource-usage contract

The current `RESOURCE_USAGE_SCHEMA_VERSION`, `ResourceUsageTracker`, and
`validate_resource_usage_payload` remain unchanged for existing task-decoding
runs. The pooled pipeline records its smaller schema-1 sidecar locally, using
standard elapsed/CPU clocks, `resource.getrusage`, and the existing public RSS
normalizer. Do not add a second tracker class or nested per-operation timing
framework before the first benchmark demonstrates a need.

The pooled resource envelope has exactly these integer fields:

```text
full_trial_count
tensor_trial_count
pfc_unit_count
hpc_unit_count
time_bin_count
training_window_count
condition_count
evaluation_pair_count
target_count
outer_fold_count
coefficient_feature_capacity
categorical_fit_count
numerical_fit_count
transform_fit_count
evaluation_metric_cell_count
tensor_allocation_bytes
planned_major_array_bytes
```

Family fit counts may individually be zero; all dimensions and byte counts are
nonnegative, required dimensions are positive, and categorical plus numerical
fit count is positive. The envelope comes from dry run and is identical across
every resume.

The pooled payload has exactly `schema_version`, `analysis_type` (fixed to
`"pooled_task_variable_decoding"`), `measurement_method`, `status`,
`peak_rss_bytes`, `wall_time_seconds`, `user_cpu_seconds`,
`system_cpu_seconds`, `cpu_efficiency`, `resource_envelope`,
`completed_checkpoint_keys`, `restored_checkpoint_keys`,
`stage_timing_seconds`, `result_counts`, `checkpoint_size_bytes`, and
`output_size_bytes`. Stage timings have exactly `input_loading`, `modeling`,
`final_assembly`, and `reporting`; they are aggregate diagnostics, not an
operation ledger. `schema_version` is integer `1`; `measurement_method` is
exactly `"resource.getrusage"`; and `status` is exactly one of `running`,
`failed`, `interrupted`, or `complete`.

`result_counts` has exactly these nonnegative integer keys:

```text
requested_transform_cells
valid_transform_cells
unavailable_transform_cells
requested_model_cells
valid_model_cells
unavailable_model_cells
requested_evaluation_metric_cells
valid_evaluation_metric_cells
unavailable_evaluation_metric_cells
```

Requested counts describe the immutable full run and match the envelope in
every status: transforms equal `transform_fit_count`, models equal
`categorical_fit_count + numerical_fit_count`, and evaluations equal
`evaluation_metric_cell_count`. Valid and unavailable counts describe
validated completed checkpoints only. Their sum is at most the requested count for a running,
failed, or interrupted payload and equals it for `complete`. Metric-family
`not_applicable` padding is excluded from all three evaluation counts.

A checkpoint key is the canonical `<training_window>::<target>` identity. A
complete payload contains every and only the canonical `W * T` completed keys
in pipeline order; restored keys are an ordered subset of completed keys. A
noncomplete payload may contain any canonical ordered subset.

`peak_rss_bytes`, `checkpoint_size_bytes`, and `output_size_bytes` are
nonnegative integers in every status. Wall/user/system seconds are
nonnegative finite numbers. `cpu_efficiency` is the nonnegative finite
`(user + system) / wall` ratio when wall time is positive and JSON `null` only
when it is zero. Each stage-timing value is either a
nonnegative finite number or `null` when that stage was not entered; all four
are non-null for `complete`. Times and process usage cover only the current
process invocation and are not fabricated for work performed by an earlier
process. `checkpoint_size_bytes` is the current sum of validated checkpoint
files. `output_size_bytes` is the current sum of all regular files below the
run directory except `resource_usage.json` and its atomic-publication
temporaries, so writing the sidecar cannot change its own value. For
`complete`, it is measured only after all other final artifacts publish.

This evidence is sufficient to choose the first cluster memory/time envelope.
Finer per-target or per-operation telemetry is deferred until the CT026
benchmark shows a concrete diagnostic need.

## 7. Tests committed before production

Tests use pytest and are committed in a RED commit before the corresponding
production code. Do not weaken a test after RED unless the requirement changed
or the test is demonstrably wrong.

### 7.1 Proposed test files

| File | Scope |
| --- | --- |
| `src/tests/neural_analysis/task_decoding/pooled/test_config.py` | Overlay configuration, canonical windows, serialization, identity |
| `src/tests/neural_analysis/task_decoding/pooled/test_decoding.py` | Master folds, stacking, model fitting, reuse, evaluation, gaps, invalidity |
| `src/tests/neural_analysis/task_decoding/test_modeling.py` | Regression coverage for extracted fit/evaluate seams and unchanged `fit_and_score` behavior |
| `src/tests/neural_analysis/task_decoding/pooled/test_results.py` | Exact arrays, dtypes, axes, reason vocabulary, checkpoints, safe load/save |
| `src/tests/neural_analysis/task_decoding/pooled/test_pipeline.py` | Synchronous computation, planning, prepare/run/resume/status, ownership, atomic completion, lean resource sidecar |
| `src/tests/neural_analysis/task_decoding/pooled/test_plotting.py` | Saved-result-only compact figures and captions |
| `src/tests/neural_analysis/task_decoding/pooled/test_synthetic.py` | Seeded public-entry-point scientific cases |

### 7.2 Required tests

Configuration and folds:

1. Omitted windows select only `pre_choice`; subsets restore canonical order.
2. Unknown fields, invalid base mode/alignment, empty/duplicate/unknown or
   misaligned windows fail.
3. Window resolution precedes serialization; canonical serialization embeds
   the base payload, reference condition, exact bounds/bin indices, and
   analysis version. Semantically identical source JSON with different paths
   or formatting has one identity; exact source bytes are preserved only in
   the two audit copies. A prepared loader reconstructs both config objects and
   windows solely from canonical `config.json`, absolute execution-only
   `session_root`, and `run_directory.parent`; it never parses the audit
   copies and rejects missing or escaping reconstructed paths.
4. Full-table values, masks, and blocks project through `trial_row_indices`
   without reordering; master eligibility is the exact baseline/target/
   bilateral intersection; rows failing multiple predicates retain the
   independent target mask and use the fixed reason precedence; one
   deterministic `(M,)` target fold map is shared by every condition, and
   behavioral blocks never cross train/test.
5. Condition-specific insufficiency is unavailable without fold regeneration,
   seed search, or trial-wise fallback.

Scientific core:

6. Stacking is exactly trial-major/time-major; repeated targets align; every
   row weight is `1 / n_training_bins`; each trial's weights sum to one; and
   the weight vector is passed to each estimator fit.
7. A spy estimator receives the exact row-weight vector. A seeded
   strict-control characterization uses `atol=1e-8` for repeated-versus-single
   coefficients/intercepts; a separate frozen-production-control smoke uses
   `atol=1e-3` for SAGA and `atol=1e-12` for ElasticNet. These are explicit
   characterization tolerances, not `COEFFICIENT_TOLERANCE` or a scientific
   equality claim.
8. Scaling/PCA use outer-training condition/window rows only and remain frozen
   for every time and destination. Exactly two regional transforms are fit per
   valid window/condition/target/fold cell, independent of region,
   representation, time, and destination counts.
9. PFC, HPC, and combined direct/PCA feature IDs, shapes, units, and coefficient
   positions are stable.
10. Estimator call count is one per model identity and is independent of time
   count and destination count.
11. Every native model is evaluated at every time, and each reference model is
    reused across destinations.
12. Transfer and native metrics use identical destination fold rows; gaps are
    exactly native minus transfer, including exact zero for valid
    correct-rewarded self-comparisons.
13. Categorical positive-class orientation and numerical native units match
    the current decoder.
14. Each reviewed scientific decision emits only its declared compact code;
    every `FitFailureKind` and `EvaluationFailureKind` member has the exact
    exhaustive mapping in Section 6.4; every code fits `U32`; no
    substring/prose inference exists; unknown states and unexpected exceptions
    propagate.
15. Extracted fit/evaluate seams preserve every existing `fit_and_score`
    output, including exact human-readable invalidity reasons, NaN fields,
    positive scores, metrics, class orientation, and convergence status.

Results:

16. Every required array has the exact declared axes, shape, dtype, units,
    padding, sentinel, and coordinate range, including `(B,) uint8` block
    payload and `(N+1,) int64` block offsets.
17. `target_values`/condition/eligibility arrays use full-table `N`,
    eligibility reasons follow the frozen precedence, `trial_row_indices` maps
    baseline-valid bilateral rows exactly, and `outer_fold_ids` uses `(T,M)`
    with `-1` only for target-ineligible common rows.
18. Model and pair identities are complete and unique; reference models are
    stored once.
19. Dense per-trial prediction arrays are absent.
20. A checkpoint contains exactly one window-target cell and cannot be reused
    across a different window, target, condition set, source, config, schema,
    or code fingerprint.
21. Every row of the exact Section 6.5 propagation table is exercised,
    including before-fit failures, copied upstream reasons, combined-region
    replacement, metric-family `not_applicable` padding, observed-zero versus
    unknown-count sentinels, and exact categorical/numerical subset rules.
22. Synchronous one-shot and interrupted/resumed computation produce
    byte-identical scientific arrays and canonical metadata. The pooled source
    fingerprint changes for every explicitly scoped dependency, including
    `task_decoding/__init__.py` and `spike_behavior/trials.py`, and ignores an
    unrelated new file; adding or editing the nested package neither enters nor
    blocks the existing top-level fingerprint/cleanliness scope, while dirty or
    untracked pooled dependencies block a pooled run.

Scientific fixtures, before reporting/lifecycle work:

23. Seeded stable, rotating/sign-changing, stable-transfer, remapped,
    absent-code, delayed-code, constant-reference, overlapping-condition, and
    multi-window fixtures produce every pooled outcome through the synchronous
    public computation path. For the rotating/sign-changing case only, the
    independent-time-bin decodability oracle calls the existing
    `modeling.fit_and_score` separately at each bin on the same seeded fixture;
    it is test-only, is not stored as a pooled result, and does not create a
    second pooled production path.
24. The full-window fixture fits once per model identity, not once per time.
25. One-shot and interrupted/resumed synthetic runs have no unexplained
    unavailable cells and preserve exact checkpoints and complete validated
    `results.npz`.

Reporting, lifecycle, resources, and usage:

26. Completion creates exactly three multi-panel PNGs per window/target:
    native, transfer, and gap. Panels/captions identify condition/pair, region,
    representation, metric, time units, the training interval, unavailable
    cells, and in-window versus out-of-window evaluation. The gap caption says
    it compares complete pipelines and cannot isolate estimator weights.
27. Plotting reads saved results only and never loads spikes or fits models.
28. Interrupted execution preserves complete checkpoints; exact resume skips
    them; terminal publication occurs only after validated results, summary,
    and required figures exist; reconstructed paths are revalidated against
    the input manifest before any checkpoint is accepted; complete reentry is
    read-only; launch snapshots match the launch/main/plotting source hashes
    and bytes recorded at preparation.
29. Dry run loads no large spike arrays or fits; it computes exact `G`,
    requested aggregate counts, and planned major-array bytes from small
    metadata.
30. The lean Section 6.7 resource payload accepts only the frozen schema,
    method, statuses, exact `result_counts` keys, canonical completed/restored
    `W*T` checkpoint identities, and declared null/nonnegative sentinels.
    Complete and partial count reconciliation is exact; restored work never
    receives fabricated timing; output size excludes the sidecar and its
    temporaries; the existing resource schema and tracker remain unchanged.
31. Existing per-time-bin runs remain readable and are never discovered as
    pooled runs; log and summary record the goal, session, parameters, scripts,
    warnings/errors, elapsed time, and the comparison's scientific limits.
32. The portable example resolves to all six conditions and all 18 targets;
    README commands name `src.neural_analysis.task_decoding.pooled.run_session`
    and the immutable run directory.

Every work package must also run all existing tests under
`src/tests/neural_analysis/task_decoding` to protect the current workflow.

## 8. Work packages and gates

### PW-WP0: documentation

- Deliverables: this plan, the companion specification, and necessary
  clarifications in existing reference documents.
- Validation: link/path checks, forbidden-open-decision search, terminology
  consistency, `git diff --check`, and human review.
- Gate: user approves the revised contract before any test or code change.

### PW-WP1: configuration and master folds

- RED: tests 1-5 in `pooled/test_config.py` and `pooled/test_decoding.py`.
- GREEN: `pooled/config.py`, including canonical prepared-run reconstruction,
  plus the minimal master-fold helper.
- Gate: focused tests, existing task-decoding tests, independent leakage/API
  review, user approval, and ledger update.

### PW-WP2: pooled scientific core

- RED: tests 6-15 and focused seeded unit fixtures, including regression tests
  for the existing `fit_and_score` API.
- GREEN: the minimal fit/evaluate extraction in `modeling.py` and
  `pooled/decoding.py`; reuse current transforms, estimators, and metrics
  without changing existing public behavior.
- Gate: exact estimator and transform call counts, verified weighting
  semantics, no evaluation refit/recalibration, scientific review, full
  regression, and ledger update.

### PW-WP3: result schema, checkpoints, and synchronous computation

- RED: tests 16-22 in `pooled/test_results.py` and
  `pooled/test_pipeline.py`.
- GREEN: `pooled/results.py`, schema-1 validation, exact status propagation,
  explicit source fingerprinting, the minimal existing-cleanliness scope fix,
  immutable atomic checkpoint save/load, and the synchronous
  preparation/computation path with no prediction arrays.
- Gate: independent byte/axis/sentinel/reason-code/source-scope review,
  one-shot/resume equality, full regression, and ledger update.

### PW-WP4: synthetic end-to-end scientific gate

- RED: tests 23-25 through the synchronous public entry point.
- GREEN: only fixes required by those scientific contracts; no reporting,
  detached execution, telemetry expansion, or new feature scope.
- Gate: deterministic complete validated scientific results, no unexplained
  unavailable cells, exact fit counts, one-shot/resume equality, independent
  scientific review, user inspection, and ledger update.

### PW-WP5: reporting, lifecycle, resources, and usage documentation

- RED: tests 26-32 in `pooled/test_plotting.py`, `pooled/test_pipeline.py`, and
  focused configuration/documentation checks.
- GREEN: `pooled/plotting.py`, `pooled/pipeline.py`, and thin
  `pooled/run_session.py` with compact multi-panel reporting, dry-run, `new`,
  `resume`, local detached execution, `status`, and the lean resource sidecar;
  revalidate reconstructed inputs before checkpoint acceptance; integrate
  validated summary/PNG rendering before terminal completion; add
  `docs/examples/neural_analysis/pooled_task_decoding_config.json`, and the
  pooled usage section in `src/neural_analysis/task_decoding/README.md`.
- Gate: exactly `3*W*T` required PNGs, explicit complete-pipeline limitation,
  ownership/failure-preservation review, atomic completion, portable examples,
  full regression, user inspection, and ledger update.

### PW-WP6: CT026 pre-choice benchmark

- Prepare an immutable all-target/all-condition pre-choice configuration.
- Run dry-run first, then one user-approved experimental session.
- Record predicted versus actual fits/evaluations/bytes, elapsed and CPU time,
  peak RSS, stage timings, invalidity, checkpoint bytes, and final output size.
- Gate: user inspects scientific results before any new feature, window, or
  multi-session execution is authorized.

### PW-WP7: optional descriptive comparison

- Enter only if the user wants a consolidated existing-versus-pooled figure
  after PW-WP6.
- Compare saved complete-fold means only after validating common session,
  target, condition, time, region, representation, and metric coordinates plus
  matching neural-source/unit-selection identity. Require and label both
  transform-training provenances. Label curves as independently
  cross-validated and descriptive.
- State that the existing per-time-bin analysis fits its scaler/PCA basis over
  the complete analysis window, while pre/post pooled models fit transforms in
  their named windows. Thus those overlays compare preprocessing as well as
  readout constraints; `full` is the closest transform-matched comparison.
- Do not calculate paired differences, modify either result, load spikes, or
  rerun either analysis.
- This work package is not required for a valid pooled analysis.

### PW-WP8: cluster and batch execution

- Enter only after PW-WP6 supplies a measured resource envelope and the user
  approves cluster implementation.
- Reuse or minimally generalize reviewed current Slurm/lifecycle behavior.
- Add dry-run and session-manifest tests before code. Default parallelism is
  across sessions, one session per worker/job-array element.
- A synthetic scheduler smoke precedes experimental submission. Submission and
  retrieval are separately approved external actions.

### PW-WP9: additional windows and experimental batch

- Run post-choice/full only with an explicit immutable configuration approved
  after PW-WP6.
- Verify adding windows does not change pre-choice scientific arrays.
- Batch only a reviewed session manifest inside the measured resource
  envelope; larger sessions require a new benchmark.

## 9. Performance policy

Before scientific invalidities, one window with all six conditions, 18
targets, five folds, three region configurations, and two representations has:

```text
1 * 6 * 18 * 5 * 3 * 2 = 3,240 model fits
```

All three windows have 9,720 fits. Destination transfer adds evaluation, not
reference fits. At 100 ms, pre/post use five stacked rows per trial and full
uses 40. These are dimensions, not runtime forecasts.

Transform fits number `W * C * T * F * 2` before invalidities because PFC and
HPC transforms are shared across regions and representations. Row weights keep
each training trial's total loss weight at one across five-bin and forty-bin
windows.

Dry run reports model/evaluation/transform counts, stacked shapes, selected
source-unit counts, exact `G`, and exact planned major-array bytes without
loading spikes. Initial runtime instrumentation records only aggregate input
loading, modeling, final assembly, and reporting times plus process wall/CPU/
peak-RSS and checkpoint/final-output sizes. Add finer timing only when the
first benchmark identifies a bottleneck that the aggregate evidence cannot
localize.

Prioritize correctness and clarity. Do not add Numba, Cython, a new estimator,
within-session multiprocessing, sparse/on-disk schemas, or caching beyond the
declared reference-model reuse unless profiling after PW-WP6 identifies a real
bottleneck and the user approves a revised plan.

## 10. Implementation discipline

For every code work package:

1. Re-read the surrounding production files and all callers before editing.
2. Verify any newly used library API from installed source or official
   documentation; record the evidence in the handoff.
3. Write and commit focused failing tests.
4. Record the expected RED failure and obtain the required review/approval.
5. Implement the smallest change that satisfies the approved contract.
6. Run focused tests, the full existing task-decoding suite, and `git diff
   --check`.
7. Refactor only while tests remain green.
8. Record commits, commands, results, review, approval, and exact next action
   in Section 1.

Implementation approval for one work package does not authorize a later work
package, experimental data access, cluster synchronization/submission, or
multi-session execution.

## 11. Immediate next action

Complete PW-WP0 validation and user review. If approved, begin PW-WP1 by
writing the listed configuration and master-fold tests and committing the RED
state. Do not begin production implementation directly from documentation
approval.
