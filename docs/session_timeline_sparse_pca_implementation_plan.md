# Session timeline: targeted SparsePCA addition

Status: implemented and validated; viewer inspection finalized 2026-11-15.
The user revised the component count to a cap of 30 after the first
full-count CT026 fit produced individual-neuron components. Adjustable alpha
initially 1 and no variance summaries remain approved. The completed full-count
product is retained separately; new preparations use the 30-component cap.

## 1. Scope and intended behavior

Add one **PCA / SparsePCA** selection to the current session timeline viewer.
Ordinary PCA remains the default. SparsePCA is an alternative preparation of
the same neural observations, displayed using the same heatmap machinery.

Reuse the current metadata, population selection, completed choices, full-window
coverage rules, [-2,+2) s bins, independent regional fits, first-trial time zero,
context/reward markers, event rows, signed speed, navigation, bounded neural
loading, color controls, and display-only component z-scoring.

The only new scientific user control is SparsePCA's alpha. Component count is
fixed to a cap of 30 per region, not another control in this addition.
The same alpha applies to both selected regions. PCA parameters and behavior
remain as they are today.

Do not add automatic hyperparameter search, another decomposition framework,
new metadata fields, new dependencies, a new chart component, or another batch
runner. SparsePCA prediction models and replacement variance metrics are outside
this addition. Existing condition predictions remain available in PCA mode.

## 2. Scientific contract

For each region independently:

1. Start with the existing float64 spike counts `(W,B,U)`: complete eligible
   choice windows, time bins, and selected units, respectively.
2. Pool window/bin observations into `(N,U)`, where `N=W*B`, preserving order.
3. Use the existing session-wide unit mean and population SD (`ddof=0`). Omit
   constant units exactly as ordinary PCA does. Keep original, retained, and
   omitted qualified unit IDs in their original order.
4. Fit on standardized counts `(N,U_retained)`, with
   `K=min(30,N,U_retained)`. Fit all eligible prepared observations, independently
   of the current viewport or displayed component range.
5. Obtain scores through sklearn's `fit_transform()` (equivalently fit followed
   by transform), not multiplication by `components_.T`. Its transform uses
   ridge regression because sparse components need not be orthogonal.
6. Reshape scores back to float64 `(W,B,K)` with the same choice/bin order.
   Preserve sklearn's returned component order and signs. Display stable IDs
   `SC1`, `SC2`, etc.; these IDs do not imply a variance ranking.

SparsePCA makes neuron loadings sparse; it does not require component activity
to be sparse over time. The initial full-count CT026 fit produced individual-
neuron directions, motivating the user's subsequent 30-component cap. The cap
does not itself guarantee that every component combines multiple neurons.

### Fixed estimator settings

The installed scikit-learn 1.8.0 signature and source were inspected. Use the
actual `sklearn.decomposition.SparsePCA` class with explicit settings:

| Setting | Initial value / policy |
| --- | --- |
| `n_components` | Regional capped count `min(30,N,U_retained)` |
| `alpha` | User selection; default 1.0; finite and nonnegative |
| `ridge_alpha` | 0.01 |
| `method` | `lars` |
| `max_iter` | 1000 |
| `tol` | 1e-8 |
| `random_state` | 0 |
| `n_jobs` | 1; preserve existing parallelism across sessions |
| Initialization | sklearn defaults; no custom warm start |

These fixed settings are recorded, not additional viewer controls. Alpha zero
removes the sparsity penalty; do not describe it as equivalent to ordinary PCA.
Alpha 1 is a starting value, not a scientifically tuned optimum. Its effect
can differ with observation count/bin width.

Record `mean_` (sklearn's additional centering), `components_`, `n_iter_`, and
`error_`, together with preprocessing and actual settings. Do not promise
component correspondence across alpha values, bin widths, or library versions.

Nonfinite inputs/outputs and no variable units make that region unavailable,
using the existing regional error behavior. Preserve zero-loading components
and their IDs rather than silently removing them. Record zero-component IDs
and show a short diagnostic if present; existing z-scoring maps constants to
zero. Ensure the heatmap has a usable color range even for all-zero scores.

If the returned iteration count reaches the configured limit, keep a finite
result but flag **iteration limit reached; convergence unconfirmed** in the
saved diagnostics and viewer. Do not equate reaching the limit with a proven
converged fit or automatically rerun with different settings. Preserve relevant
sklearn warnings in the run log.

## 3. Small implementation boundaries

One new production module is sufficient. Existing public names and default
calls remain supported; no general estimator registry, plugin system, or class
hierarchy is needed.

| File | Targeted change |
| --- | --- |
| `population/timeline_sparse_pca.py` (new) | Parameter validation and one `fit_sparse_timeline_activity(...)` computation over prepared count arrays; no file or UI operations. |
| `population/session_timeline.py` | Add a small data-only `SparseTimelineTransform`; allow the existing regional result's transform to be either ordinary or sparse. Preserve current public result-container names. |
| `interregional/pca.py` | Extract only the existing mean/SD/constant-unit standardization block into a shared function. Ordinary PCA calls the same arithmetic through that function; fitting and projection remain unchanged. |
| `population/session_timeline_io.py` | Sparse-specific configuration and named-array serialization/loading branches; keep ordinary PCA and legacy identity paths intact. |
| `population/run_session_timeline.py` | Select the fitter at the preparation boundary; add CLI options and algorithm-aware summary text/discovery. Reuse current source loading, run directories, logging, and reports. |
| `population/run_session_timeline_batch.py` | Forward the same preparation options through the existing session workers. |
| `population/timeline_plotting.py` | Reuse heatmaps; use sparse labels and omit PCA-only hover percentages. |
| `webapp/timeline_view.py` | Algorithm/alpha controls, correct saved-result selection, sparse diagnostics, and summary/prediction gating. |
| `population/window_variance.py`, `population/run_window_variance.py` | Small explicit guards against computing PCA variance contributions for a SparsePCA result. No change to ordinary Stage 2 calculations. |

All new/modified functions document input types, shapes/axes, physical units,
and returned data. Keep the new fitter readable and procedural, following
docs/SoftwareDesign.md.

### Shared standardization helper

Extract the existing numerical block rather than maintaining two versions of
unit scaling. Its input is the float64 observation matrix `(N,U)` in counts
per bin. Return standardized observations `(N,U_retained)`, the retained-unit
mask `(U,)`, and mean/SD vectors `(U_retained,)`. Preserve the existing handling
of nonfinite statistics, constant units, and population-SD arithmetic.

This is the only planned change inside the interregional PCA module. Verify
that ordinary PCA means, scales, loadings, variances, and scores are unchanged
on representative fixtures. No changes to RegionalPCATransform's PCA-specific
validation or to interregional analyses are needed.

### Sparse result record

`SparseTimelineTransform` holds input/retained/omitted unit IDs, count means and
SDs, sklearn's additional center, `(K,U_retained)` components, observation count,
settings, iteration count, and objective history. It has no invented
`explained_variance` or `explained_variance_ratio` fields.

Reuse `RegionalTimelinePCA` and `SessionTimelineResult` as the timeline's existing
array/table envelopes. Retain direct standardized-unit session/window variance
totals where the envelope already requires them; these remain valid descriptive
input quantities, but are not attributed to sparse components or displayed as
SparsePCA coverage. Update the documented score contract to distinguish ordinary
PCA projections from sparse ridge coefficients. Avoid renaming public records
or changing existing positional constructors.

## 4. Saved products and compatibility

Keep ordinary PCA's analysis version, schema, default configuration dictionary,
configuration digest, NPZ layout, and existing archive lookup behavior unchanged.
Missing algorithm metadata in an old ordinary result means PCA; interpreting
that default does not require modifying its metadata or rewriting the file.

SparsePCA uses `session-timeline-sparse-pca-v1` with the existing named-array
NPZ envelope (`schema_version=1`). Its configuration adds the algorithm,
30-component cap, complete estimator settings, seed, and installed sklearn
version before calculating its identity. Include all existing source/population,
choice-window, and bin identities as before. Ignore no scientific setting when
finding a matching sparse cache.

Reuse the current `session_timeline_pca_<date>_<time>_<id>` run-directory layout
to avoid another discovery tree. Config/summary/viewer labels explicitly identify
SparsePCA. Different algorithm/alpha settings have different IDs and new paths.
The historical directory prefix alone does not identify the algorithm. The
current sparse identity includes `component_count_policy="capped"` and
`requested_components=30`; the initial `"full"` products keep their own IDs
and remain readable without conversion. This is a changed parameter, not a
change to sklearn's algorithm or to the ordinary-PCA version.

The serializer currently hardcodes ordinary analysis version and `random_seed:
None`; explicitly branch those fields for SparsePCA. The loader accepts only
the two documented version/schema combinations and creates the corresponding
transform record. Do not weaken the ordinary transform checks to admit sparse
loadings. Sparse loads validate their own finite arrays, dimensions, ID order,
positive preprocessing scales, and settings; they do not require orthogonality.

Store named arrays for scores, loadings, count means/scales, sklearn center, and
objective history, alongside the existing behavior/choice/bin arrays. Metadata
records generator, analysis version, seed, parameters, software/code version,
units/axis conventions, source identities, regional diagnostics, and timings.
Sparse score units are ridge coefficients for standardized-unit activity;
component SD z-scores are dimensionless display values. Do not save fake PCA
variance arrays or pickle sklearn estimator objects.

Keep existing temporary-save/no-overwrite, explicit-rerun, and dry-run behavior.
Legacy PCA registration and compatible regional PCA searches remain PCA-only.
Existing PCA, Stage 2, prediction, and speed products must retain their paths,
bytes, mtimes, and ordinary lookup results.

## 5. Preparation interfaces

Extend `run_session_timeline(...)` and `timeline_configuration(...)` with
keyword-only algorithm/sparse-alpha inputs, defaulting to ordinary PCA. Keep
the same report type. Select ordinary versus sparse computation at the runner
boundary after the common population/choice selection; call existing binning
functions rather than duplicating bin semantics.

Add `--algorithm {pca,sparse_pca}` and `--sparse-alpha` to the current single and
batch CLIs. Example:

```bash
uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/neural_session.json \
  --algorithm sparse_pca --sparse-alpha 1 --bin-width-ms 500 --dry-run
```

The same command without `--dry-run` prepares/reuses the selected sparse result.
`--rerun` creates another timestamped result. Explicit sparse-only options with
ordinary PCA or `--speed` are input errors rather than silently ignored settings.
SparsePCA plus `--window-summary` is an explicit unsupported combination.
Direct Stage 2 Python calls also reject sparse inputs before writing a result;
the UI hiding alone is insufficient protection.

Dry-run inspects only the same metadata/small source identities as today, with
no spike loading, decomposition, or output directory. Batch preparation retains
the existing session-level workers and uses one SparsePCA job per worker.

## 6. Viewer behavior

Place the algorithm selector beside the existing preparation selections, inside
the stable controls container. In sparse mode show one numeric alpha input,
initially 1.0, minimum 0. Changing algorithm or alpha selects a configuration;
only Prepare/Recompute fits it. A missing sparse cache shows behavior and an
unprepared status, not scores from another alpha or ordinary PCA.

Reuse component selection with labels such as `PFC sparse components` and
heatmap rows `SC1`, `SC2`, etc. Keep the same initial visible count, colors,
manual limits, heatmap gaps, and prepared full-session standardization for the
optional component z-score. Use `component SD` in sparse color legends.

In sparse mode omit:

- Session 80/90/95% coverage counts and percent-variance hover fields.
- Stage 2 prepare/recompute/visibility controls, count rows, and contribution
  tables. Display one concise explanation that variance summaries require
  ordinary PCA.
- The condition-prediction preparation/overlay block. Existing saved ordinary
  prediction models cannot be applied to sparse scores.

The last point requires gating before calling `render_prediction_controls()`:
its current saved-anchor behavior can replace the timeline basis even when
prediction display is hidden. In sparse mode do not load or honor that anchor.
Preserve the saved ordinary-PCA selection in session state for returning to PCA
mode; no prediction-model refactor is needed.

Preserve the single existing chart component, controls container, session/time
navigation keys, and time-axis UI revision. Component/configuration changes
may update vertical ranges while keeping the panned time interval when it fits
the destination loaded bounds, using the chart's existing behavior. Jump and
explicit time controls continue to request a new interval.

Retain ordinary heatmap custom-data shape for compatibility; sparse entries
that would otherwise hold percent variance are absent values, never invented
percentages. Select the appropriate hover template so those entries are not
shown. Reuse the existing generic window-heatmap and z-score helpers.

## 7. Tests before implementation

Write focused tests, observe RED, and commit tests before the implementation
for each work package. Use pytest functions and existing fixtures, plus a
small new `test_timeline_sparse_pca.py`; extend current I/O, runner, plotting,
and webapp tests rather than building another testing harness.

1. **Ordinary preprocessing regression:** extraction preserves means, SDs,
   retained IDs/order, PCA loadings/scores/variance, and constant-unit behavior.
2. **Sparse estimator agreement:** a seeded small count tensor matches an
   independently constructed installed sklearn fit/transform using explicit
   standardization and identical settings. Verify `(W,B,K)` order, the cap of
   30, smaller observation/unit bounds, and loading legacy full-count products.
3. **Correct score method:** a nonorthogonal fixture agrees with sklearn's
   ridge transform and differs from a simple loading dot product.
4. **Scientific edge cases:** repeatability, constant units, no usable units,
   zero components, finite-alpha validation including zero, nonfinite input,
   stored center/settings, and an iteration-limit diagnostic. Do not assert a
   particular sparsity level or component identity on arbitrary random data.
5. **Persistence:** exact named-array/diagnostic round trip for sparse results;
   ordinary/legacy loads remain valid. Reject mismatched shapes/versions.
6. **Cache separation:** alpha/algorithm changes select another result; ordinary
   default digests and registered archive lookup are unchanged. Reuse avoids
   fitting; explicit rerun preserves prior output; dry-run avoids neural reads.
7. **CLI/batch:** new options reach the sparse fitter; speed and ordinary paths
   retain their current calls. Unsupported combinations fail. Exercise the
   batch path on small synthetic sessions only.
8. **Plotting:** reuse actual-time edges, gaps, score values, bounds, colors,
   and component z-scoring; sparse labels/hover omit all variance percentages.
   Input scientific arrays remain unchanged by display operations.
9. **Viewer isolation:** selected alpha controls lookup/preparation; no implicit
   fit. Sparse mode never invokes the prediction-anchor path or Stage 2 controls;
   ordinary mode still does. Missing sparse data shows behavior rather than
   an unrelated saved basis.
10. **Stage 2 guard:** direct sparse summary calls fail explicitly, while all
    ordinary calculations and saved-sidecar selections continue to pass.
11. **Navigation regression:** stable component key/container and time revision
    survive algorithm, alpha, component-range, event, and z-score changes.
    Include the existing Node chart lifecycle test and actual browser panning.

Run the focused suite first, then related interregional PCA, timeline/legacy
identity, Stage 2, speed, and prediction-view regressions. Use `uv run` for all
local Python commands, Ruff on changed Python paths, and diff whitespace checks
on this addition. Do not change unrelated dirty files or commit their changes.

## 8. Performance and designated-session validation

Use vectorized existing NumPy operations and sklearn. No optimization framework,
resampling, subsampling, minibatch substitute, GPU, or extra decomposition cache
is part of this addition. Sparse fitting is expected to be the expensive step;
measure rather than promise a preparation time.

After synthetic correctness, validate CT026 2026-08-04 at 500 ms, alpha 1, up to 30 components,
for the existing selected regional populations. Reuse the existing metadata
and speed cache; do not alter experimental inputs. Before/after checks cover
the paths, stats, and hashes of the original ordinary PCA/Stage 2 products.

Record actual eligible W/B/U/K, sparse loading zero fraction, zero-component
IDs, objective history, iteration count, warnings, standardization/fit/transform
and total preparation time, saved size/load time, chart construction/JSON size,
and process peak RSS with its lifetime limitation. Use existing timing/benchmark
boundaries; separate fit and transform timing if needed rather than estimating
their individual costs from a combined `fit_transform()` duration.

Browser-check sparse selection and cache reuse, negative/zero scores, rank
selection, z-scoring, alpha changes, context/reward/events/speed alignment,
panning, Jump, bounded loading, missing summaries, and return to ordinary PCA
with saved predictions. Keep the current light theme. Save a labeled PNG and
short inspection notes beside the new result.

All output stays in a new timestamped directory under the session's existing
analysis_runs, with results.npz, configuration, log, main/batch script copies,
and summary. The summary states purpose, session, invocation, settings, units,
diagnostics, and runtime; no statistical tests are implied. No old result is
overwritten. User inspection precedes any experimental batch.

Do not automatically launch the full 20-ms preparation after the first run.
Review the measured 500-ms fit/iteration/memory behavior first. If capped
SparsePCA is impractical, report that limitation instead of silently reducing
K, coarsening data, or switching to MiniBatchSparsePCA. The eventual plotting
path continues to use prepared arrays and existing bounded loading.

## 9. Implementation sequence and completion

1. Commit RED tests for shared preprocessing and sparse computation; implement
   the shared arithmetic extraction, sparse record, and new fitter.
2. Commit RED storage/cache/runner tests; implement sparse-only identities,
   save/load, single/batch CLI forwarding, diagnostics, and Stage 2 guards.
3. Commit RED plotting/viewer tests; add controls, labels, and the narrow
   summary/prediction exclusions. Keep the existing chart implementation.
4. Run integrated regressions and the synthetic pipeline; inspect the designated
   CT026 run and record performance, screenshots, and original-result integrity.
5. Add concise SparsePCA usage to SESSION_TIMELINE.md and its README reference.
   Record measured limitations and completed validation in this plan.

Completion means both algorithms load their own matching products; ordinary
PCA remains compatible; SparsePCA scores agree with sklearn and display on the
same timeline; PCA-only metrics/models cannot attach to sparse scores; tests
pass; and usage plus single-session validation are available for review.

References checked against installed scikit-learn 1.8.0:

- SparsePCA API: https://scikit-learn.org/1.8/modules/generated/sklearn.decomposition.SparsePCA.html
- Formulation: https://scikit-learn.org/1.8/modules/decomposition.html#sparse-principal-components-analysis-sparsepca-and-minibatchsparsepca
- Local implementation: .venv/lib/python3.12/site-packages/sklearn/decomposition/_sparse_pca.py

## 10. Completed implementation and validation

The addition uses one new production module, the shared arithmetic extraction,
and the existing timeline storage/runner/viewer paths. No dependency, metadata
schema, chart component, or replacement variance metric was added. Sparse
fitting is explicit; display changes use saved scores. The component cap is
fixed at 30, with fewer components only when observation/unit bounds require it.

### Test-driven work packages

RED tests were committed before implementation:

- `c4a9f9a`: preprocessing/scientific scores; GREEN in `12512fe`.
- `a5186ca`: persistence, identity, CLI/batch, and Stage 2 isolation; GREEN in `4ba59ea`.
- `146f52e`: plotting/viewer behavior and prediction-anchor exclusion; GREEN in `8553c19`.
- `c9e14f0`: the user's revised 30-component cap and legacy full-product loading;
  GREEN in `b66b46b`.

One genuine test correction allowed 1e-12 numerical tolerance against an
independent sklearn reference: different NumPy array layouts produced a maximum
2.22e-16 rounding difference. Ordinary PCA regression checks remain exact;
repeated identical sparse calls remain exactly repeatable in the tested setup.

The integrated run passed **364 tests** across sparse computation/storage/UI,
ordinary/interregional PCA, legacy timeline identity, Stage 2, speed, the Node
chart lifecycle, and condition-prediction preparation/models/results/plotting.
Two warnings came from the existing fork-based synthetic batch executor under
Python 3.12. Changed Python paths pass Ruff. No experimental batch was run.

### CT026 products and source integrity

Designated session: CT026 2026-08-04. The completed capped result is:

```text
analysis_runs/session_timeline_pca_20261114_234820_260110_d7104f373744628c/results.npz
```

It contains `(716,8,30)` scores for each region, 500-ms bins, alpha 1, seed 0,
174 retained PFC units and 316 retained HPC units (one constant HPC unit omitted).
The initial full-count product remains separately readable at
`session_timeline_pca_20261114_233545_548420_a1345a13844a2f47/results.npz`.

The processed behavior CSVs had been regenerated before this task. The current
source pool has 716 choices; the original October PCA products had 715 and
already-stale source identities. Validation therefore used current raw spike
counts, independent manual unit scaling, and sklearn's ridge solve. Both
regions' saved scores matched that independent ridge calculation exactly.
This checks score correctness, not optimization convergence.

Before/after size, mtime, and SHA-256 checks preserved all four original
500/20-ms PCA/Stage 2 NPZ archives, and their ordinary lookup status remained
unchanged. Current source changes were not bypassed by registering/relabeling
an old product. The existing speed implementation had advanced to v3 before
this task; its matching cache was prepared explicitly through the existing
Prepare speed button for inspection, preserving the older v2 product.

### Capped-fit diagnostics and measurements

| Quantity | PFC | HPC |
| --- | --- | --- |
| Fitted components | 30 | 30 |
| Retained units | 174 | 316 |
| Fit time | 101.56 s | 199.07 s |
| Transform time | 0.0048 s | 0.0068 s |
| Iterations | 1000 | 1000 |
| Zero loading fraction | 51.44% | 44.93% |
| Entirely zero components | None | None |
| Optimization status | Iteration limit; convergence unconfirmed | Iteration limit; convergence unconfirmed |

The smaller basis now combines neurons rather than reproducing a full set of
single-neuron axes. Both fits exhausted the approved iteration budget. Finite
coefficients remain available for inspection, with explicit viewer warnings
and saved diagnostics; neither is described as a certified converged optimum.
No tolerance/budget change or automatic retry was performed.

Total preparation took 304.26 s. The NPZ is 3,177,429 bytes and loaded in
0.034-0.035 s. The two score tensors occupy 2,749,440 bytes combined. A full
session figure with 10 visible components per region and default event rows
took 0.968 s to construct and 0.658 s to serialize, producing 14,475,722 bytes
of JSON without optional speed. Process peak RSS was 0.771 GB during
preparation and 0.790 GB across the validation/benchmark lifetime, including
imports and independent count reconstruction. These are single local calls,
filesystem cache state unspecified, excluding browser latency. Sklearn and
BLAS used one job/thread; the environment reported 24 CPUs.

Validation JSON, benchmarks, code/runner copies, source snapshots, and
`implementation_provenance.json` remain beside the capped result. Preparation
started while the cap edits were being committed; the immutable NPZ records
the then-current HEAD. The provenance supplement identifies validated commit
`b66b46b` and source copies, while saved estimator settings explicitly record
the actual 30-component fit. No NPZ provenance was rewritten.

### Browser inspection

The production viewer selected the capped archive and showed SC labels,
component-SD legends, context/reward markers, and signed speed. It omitted PCA
variance summaries and prediction controls. Native pan moved the shared ticks
from 10-60 to 30-80 s; a rank change and an alpha round trip preserved that
interval. Unprepared alpha 0.3 showed behavior/speed without unrelated scores
or a new preparation. Ordinary PCA restored its saved prediction anchor;
returning to SparsePCA correctly bypassed that anchor.

Bounded loading preserved the panned interval. Jump to trial 4 explicitly
changed Window start to -0.002 s and the loaded interval to [-0.002,119.998] s.
Both convergence warnings were present. `ct026_sparse30_preview.png`,
`viewer_inspection.md`, and `viewer_validation.json` provide review artifacts
beside the prepared result. Usage is documented in SESSION_TIMELINE.md and
linked from the neural-analysis README. No finer-bin sparse preparation or
experimental batch was started.
