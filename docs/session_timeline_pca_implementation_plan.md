# Session Timeline and PCA Viewer Implementation Plan

Status: Stage 1 code implemented and verified on synthetic data on 2026-10-30.
The designated real-session inspection remains pending. Stage 2 has not
started and still follows the Stage 1 inspection gate.

Scientific specification: `docs/session_timeline_pca_spec_v1.md`.
Design guidance: `docs/SoftwareDesign.md` and repository `AGENTS.md`.
The user clarifications recorded below resolve the input questions raised
during review. They take precedence over the earlier review's suggestions.

## Stage 1 implementation record

- Implementation commit: `1bee80d`.
- Tests-first commits: `19987ca` (inputs/PCA), `723d628` (geometry, scripts,
  routing), `6efedfb` (Streamlit reruns), and `04e1ccc` (ordering/measurements).
  RED failures were observed before the corresponding implementations.
- `d2922bb` corrected a genuine synthetic-fixture error: the existing loader
  requires the established `channel_quality.csv` filename. Scientific test
  expectations were not relaxed.
- Final focused and related regression run: 207 passed, with four existing
  Pynapple warnings in existing population plotting tests. The command covered
  the five new test files and the existing population PCA, interregional PCA,
  population/webapp packages, PSTH webapp, session metadata, and quickstart.
- Ruff checks passed for all new source and test modules. New source and usage
  documentation use ASCII. Changed-path whitespace checks passed.
- Synthetic CLI preparation/reuse, explicit rerun, batch preparation, and
  benchmark output were exercised. Browser checks covered both populations,
  real gaps, cell hover, shared native panning, event toggles, and PC selection
  beyond the initial ten rows. A separate synthetic case used 18 units per
  population; 95-percent coverage required 12 PFC PCs and 13 HPC PCs while the
  initial heatmaps displayed ten, confirming independent footer coverage.
- Initial Streamlit 1.58 browser behavior: sidebar display changes restored the
  explicit Window start/duration viewport after native panning. Numerical
  products are reused. This limitation is stated in the app and usage guide;
  no custom JavaScript component was added initially. The approved viewport
  follow-up below supersedes this limitation.
- Usage documentation: `src/neural_analysis/SESSION_TIMELINE.md`, linked from
  the neural-analysis README.
- Approved follow-up: choices-row context overlay from logged patch/dark entry
  events, with default-on visibility control and active context in choice hover.
  Tests-first commit `5356911` produced eight expected RED failures; the
  updated combined suite passes 215 tests with the same four existing warnings.
  This display-only addition reuses existing archives and requires no PCA
  recomputation or saved-result format change.
- Approved follow-up: optional live per-PC z-scoring using each population's
  full prepared choice-window/bin pool (population SD, `ddof=0`). Only selected
  rows are transformed; saved scores, loadings, and variance summaries retain
  their original meaning. Heatmap units and manual limits reflect the mode.
  Tests-first commits `2d9683d` and `36f2d04` observed RED before implementation;
  the combined regression suite passes 225 tests with four existing warnings.
  Synthetic normalization timings for 200,000 bins: median 37.7 ms for 10 PCs
  and 669.7 ms for 100 PCs across three calls, excluding figure/browser work.
  The option requires no archive, preparation, or dependency changes.
- Approved viewport follow-up: keep a stable Streamlit v2 component and update
  its Plotly element with `Plotly.react`. PC-rank, z-score, and display changes
  preserve mouse-pan and range-slider time windows. Explicit time controls and
  Jump change the navigation revision, including repeated/clamped requests.
  Vertical-axis revisions are separate so PC/track changes reset row ranges.
  The wrapper preserves the existing toolbar, including fullscreen, and uses
  the installed Plotly JS locally without new dependencies or archive changes.
  Tests-first commit `0e3ee29` observed seven RED failures; additional frontend
  and axis regressions were tested before their fixes. Browser checks covered
  pan/range-slider then rank changes, z-scoring, PC order, fullscreen, Jump,
  and repeated Jump after panning away. Panning remains entirely browser-local.
  Final combined regression: 230 passed, with the same four existing Pynapple
  warnings; Ruff and the offline dependency-lock check passed.

Small synthetic benchmark examples (three choices, two units/PCs per region):

| Bin width | NPZ load | Figure construction | JSON serialization | JSON bytes |
| --- | --- | --- | --- | --- |
| 500 ms | 0.00168 s | 0.13167 s | 0.00212 s | 27,736 |
| 20 ms | 0.00174 s | 0.08934 s | 0.00909 s | 351,852 |

These are measured smoke checks, not representative-session benchmarks or
latency guarantees. Filesystem cache state is unspecified; import/warmup
effects can dominate small fixtures. Generated benchmark JSON additionally
records preparation timings, dimensions, available memory, machine/CPU, and
process-lifetime peak RSS. No experimental session or experimental batch has
been run. Supply the first designated metadata path for the scientific
inspection and representative-session performance check.

## 1. Agreed behavior

| Topic | Decision |
| --- | --- |
| Time origin | First trial start, determined before selecting choices or a viewport. |
| Choices | Completed ordinary left/right choices with finite choice timestamps, including incorrect and unrewarded choices. |
| Nonchoices | Exclude experimenter rewards and `no_choice` rows from choice markers and neural windows. Their available behavioral events can still be shown. |
| End of file | Use the processed trial table. Do not reconstruct the unfinished active trial from event rows, and do not drop the last saved trial merely because it is the last row. |
| Populations | One population per probe, restricted by metadata `unit_channels` when supplied. Do not split a probe into populations based on its named LFP sites. |
| Channel selection | Existing good-channel and inside-brain filtering, intersected with metadata channel restrictions. |
| Unit selection | Existing Phy manual `group` labels, retaining `good` and `mua`, including the existing normalization of missing quality labels. No automatic switch to `KSLabel`. |
| Coverage | Existing IRIG-derived recording bounds and choice timestamps establish complete windows. No artifact masks, additional artifact screening, or new coverage framework. |
| Behavioral events | Existing trial-start, choice, left/right lick, and LED event definitions and timestamp conventions. No new event schema. |
| Staging | Deliver and inspect Stage 1 before implementing Stage 2. Stage 2 is required; Stage 3 is optional and needs a separate decision. |

The viewer is exploratory and processes one simultaneously recorded session
at a time. Neural activity is shown only inside `[-2, +2)` second choice
windows. Actual elapsed time and the gaps between windows are preserved.
The nonoverlapping-window assumption is accepted as specified; do not add
overlap detection or exclusion policies.

## 2. Design approach

Use short, typed functions and a few data-only dataclasses. Scientific
functions accept arrays, series, or tables whose alignment matters; they do
not load files, access Streamlit, or construct session metadata. Loading,
saving, plotting, and orchestration have separate owners.

Keep new code within the existing shallow `population` and `webapp`
packages. Do not add a visualization plugin system, inheritance hierarchy,
factory registry, generic pipeline engine, or a new metadata schema.
No broad refactor of the existing app or PCA workflows is needed.

### Proposed files and responsibilities

All paths in this table are relative to `src/neural_analysis/`.

| File | Responsibility and principal functions |
| --- | --- |
| `population/session_timeline.py` | Data-only result records and pure calculations: `prepare_timeline_choices`, `select_complete_choice_windows`, `compute_timeline_pca`, `summarize_session_variance`, and subsequently `summarize_window_variance`. |
| `population/session_timeline_io.py` | Metadata-based source loading, existing channel/unit selection, source identity, and named-array persistence: `load_timeline_behavior`, `load_timeline_population`, `save_timeline_result`, `load_timeline_result`, and subsequently save/load of window summaries. |
| `population/timeline_plotting.py` | Pure Plotly figure construction: `build_timeline_heatmap_data` and `plot_session_timeline`. Accept computed results and explicit display settings. |
| `population/run_session_timeline.py` | Single-session entry point and `run_session_timeline`: assemble inputs, reuse matching saved results, run preparation, save outputs, and write the run log/summary. |
| `population/run_session_timeline_batch.py` | Small batch entry point: read a list of metadata paths and call the single-session function in session-level workers. |
| `webapp/timeline_view.py` | `render_session_timeline_view`, narrowly scoped caching, navigation controls, per-region PC controls, event toggles, and footer/readout rendering. |

Existing files receive localized edits:

- `webapp/session_inputs.py`: add the view label and its behavior-first
  availability rule.
- `webapp/app.py`: dispatch to the new view before the existing active-probe,
  trial-filter, and raster controls.
- `pyproject.toml` and `uv.lock`: add Plotly.
- `README.md` within `src/neural_analysis`: document preparation and viewing.

The documented `psth_webapp.py` entry point remains the entry point. The new
view uses the metadata route. The manual-path app continues to offer its
existing views and gives the existing metadata launch guidance for this view;
there is no second implementation of population resolution.

These are responsibility boundaries, not a requirement to split small
functions further. Helpers stay private in their owning module. Every
function documents input types, shapes, axes, units, and returns. Avoid
convenience lambdas, nested helpers, generic `utils` modules, and unnecessary
configuration classes.

### Reuse instead of replacement

| Existing owner | Reuse |
| --- | --- |
| `session_metadata.py` | Load/resolve metadata, probe sources, `unit_channels`, and existing PFC/HPC probe-role assignments. |
| `spike_behavior/loading.py` | Sorter/aligned-spike loading, spike/cluster correspondence checks, channel-quality loading and selection, `filter_cluster_metadata`, and Pynapple spike groups. |
| `behavior_analysis/project_utils.py` | Existing experimenter-reward flag normalization and missing-value conventions. |
| `population/pca.py` | `build_trial_unit_rate_tensor_pynapple` for existing unsmoothed, choice-aligned binning. |
| `interregional/pca.py` | Public `fit_regional_pca` and `transform_regional_activity` functions. They already retain the required scaling, loadings, and spectrum and omit constant units. |
| `task_decoding/activity.py` | `inspect_probe_coverage` for existing trusted IRIG bounds. Do not call the bilateral trial-selection/tensor workflow. |
| Existing webapp and analysis outputs | Follow existing metadata launch, session-local run-directory, log, and named-array conventions. Do not copy another analysis's complete workflow infrastructure. |

Direct reuse of the regional PCA primitives is a deliberate, narrow import
dependency. It is simpler than copying their implementation or moving their
public result class and risking existing serialized results. Do not call
interregional paired fitting, cross-validation, regression, or decoding.
Existing PCA interfaces and result formats remain unchanged.

## 3. Inputs and data contracts

### Behavior and population loading

1. Load the processed trial table from the metadata path. Determine the
   first trial start before filtering. Preserve original row positions and
   `cur_trial` identities; they are not interchangeable.
2. Select rows with finite `choice_time`, ordinary action `0` (right) or `1`
   (left), and ordinary non-experimenter choice status using the existing
   flag conventions. Do not add reward, correctness, block, model-target,
   or decoding eligibility requirements.
3. Load optional events from the metadata event table. Reuse `Time`, `Event`,
   and existing `Note` information and event names. Missing event input does
   not prevent displaying choices or neural activity.
4. Resolve each selected probe independently. Select channels with the
   existing good/inside-brain functions, apply `unit_channels`, and select
   manual Phy good/mua clusters in stable cluster-ID order.
5. Keep probe-qualified unit IDs such as `ProbeA:7`; integer cluster IDs
   remain the keys used for the probe's Pynapple group.
6. Read the existing IRIG coverage bounds. A region includes a choice window
   only when both window edges fall within its recording bounds. Valid
   silence is a zero count, not a missing window. Do not infer recording
   bounds from spike extrema.

Metadata version 3 probe-role assignments provide defaults. Where a supported
metadata file lacks assignments, show explicit optional PFC/HPC probe
selectors using the existing probe keys, without guessing anatomy from names.
Selecting a probe identifies the collection; its sites do not create extra
populations. Distinct panels cannot silently use the same probe twice.

An unavailable probe, missing required source, no selected units, no complete
windows, or no variable units makes that population unavailable with a short
reason. Continue with other valid populations and behavior. Catch expected
input/file errors at this boundary; do not disguise programming errors as
missing data. An archive without the expected IRIG information is reported
unavailable rather than starting another synchronization workflow.

### Array conventions

| Product | Shape and meaning |
| --- | --- |
| Choice IDs, row positions, actions, times | One-dimensional arrays with one entry per completed eligible behavioral choice. |
| Regional eligible positions | One-dimensional indices into the common choice table. Different regions may include different choices. |
| Choice-window edges | `(B + 1,)`, relative seconds; absolute edges are the actual choice timestamp plus these edges. |
| Binned activity | `(W, B, U)`, choice window, time bin, selected unit; unsmoothed spike counts per bin. |
| Fitting observations | `(W * B, U)`, pooled choice-window observations; unit order unchanged. |
| PCA loadings | `(K, U_retained)`, component by nonconstant retained unit. |
| PC scores | `(W, B, K)`, ordinary signed, unwhitened scores. |
| Session spectrum | `(K,)`, sample explained variances and their total-variance ratios. |
| Window variance totals | `(W,)`, sample variance summed across the same session-standardized retained units. |
| Stage 2 summaries | One count/status/total per regional window; ranked session-PC identities and contributions have shape `(W, K)`. |

Use float64 for calculations and scores. Preserve absolute UTC seconds and
the first-trial origin in saved metadata; plot elapsed seconds obtained by
subtracting that origin. Relative edges remain in seconds. Windows before
elapsed zero are valid when recording coverage includes them; the behavior
origin is not a new neural exclusion boundary.

Use two Stage 1 data-only records: `RegionalTimelinePCA` and
`SessionTimelineResult`. The regional record holds eligibility, scores,
the existing fitted transform, bin edges, variance totals, and color limits.
The session record holds identity/origin, aligned choice/event tables, regional
results, unavailable reasons, and analysis metadata. Stage 2 adds a small
`WindowVarianceSummary` record. No record manages loading, plotting, or jobs.

## 4. Stage 1 calculations

For each selected population and bin width:

1. Select complete regional windows from the common choice table.
2. Use the existing Pynapple rate builder with `choice_time` and `[-2, +2)`.
   Convert returned Hz to counts by multiplying by the known bin width, as
   existing interregional preparation already does. Document the conversion;
   do not write a second binner or add smoothing.
3. Verify the expected `B`: 8, 40, 80, or 200 for 500, 100, 50, or 20 ms.
   Use explicit edges, not rounded choice timestamps or an all-session grid.
4. Fit the existing regional PCA primitive on all complete windows with a
   request large enough for the full available basis, capped by observations
   and units. It uses population SD (`ddof=0`) for unit z-scoring, full SVD,
   and `whiten=False`. No randomness is used.
5. Preserve the fitted transform and project every included window into it.
   Keep PC order and signs exactly as fitted. Do not refit to show more rows.
6. Compute total session variance directly in the standardized unit space
   using sample variance (`ddof=1`). Use the same convention for explained
   variances and the leading 80/90/95 percent threshold counts.
7. Retain the full available fitted spectrum/loadings and, initially, all
   scores. Numerically null axes may be identified with a documented standard
   machine-precision rank tolerance; do not introduce a scientific variance
   cutoff or remove axes separately for individual choices.
8. During projection, save the within-window standardized-unit variance totals
   needed as Stage 2 denominators. This is a small reduction over data already
   in memory, allowing Stage 2 to reuse exact totals without keeping another
   large raw activity tensor. Stage 1 does not calculate or display local
   participation counts.
9. Set each population's fixed symmetric color range from the maximum absolute
   score over its complete configuration, including nonvisible PCs. Display
   the limits. A manual symmetric range may be offered, with saturation
   explicitly labeled; no automatic percentile clipping is needed initially.

Constant units are omitted once per population/configuration and reported.
One variable unit can produce a one-PC display; an all-constant population is
unavailable. No post-PCA z-score, whitening, row scaling, or trace smoothing
is introduced. Production metadata records that the computation is
deterministic and has no random seed; all synthetic generators have explicit
saved seeds.

## 5. Stage 1 figure and navigation

### Figure construction

Construct one Plotly figure with matched numeric elapsed-time x-axes:

1. Left/right choice lanes at actual choice times.
2. Independently toggled trial-start, left-lick, right-lick, and LED tracks.
3. PFC heatmap when available.
4. HPC heatmap when available.
5. A compact range navigator; the variance footer is rendered below the chart.

Each neural heatmap uses `PC x temporal cell` values and explicit time edges.
Between successive windows, insert one missing-data column spanning the
actual gap, rather than allocating bins throughout the gap. Touching windows
need no gap cell. Include unavailable windows as blanks; never extend a
neighboring score through them. Set `zsmooth=False`, `connectgaps=False`,
and disable hover over gap cells. Use PC1 at the top and preserve session order.

Heatmap hover identifies population/probe, PC rank and session variance
percentage, trial ID, absolute and elapsed bin interval, choice-relative
interval, and raw score. A zero score has the center color; missing cells
show the background. Keep hover metadata compact and measure its contribution
to payload size instead of preformatting large repeated strings.

Use the existing LED streams to pair verified on/off transitions. Preserve
the original times in hover, including when an interval crosses the visible
boundary. Show unpaired transitions without inventing an interval endpoint.
Do not infer separate LED identities if the source does not contain them.

### Controls and state

- Default bin size: 500 ms. All four specified sizes remain available.
- Initial visible duration: 60 seconds, starting near the first choice.
- Explicit time-position, duration, and jump-to-original-trial controls.
- Per-population PC rank range, initially the first ten available PCs.
- Event-track toggles and readable panel heights. PC rank controls provide
  access to many rows without requiring an enormous page.
- Native horizontal pan and zoom. Lock the neural y-axes against accidental
  drag navigation, and let the mouse wheel scroll the page normally.
- Fixed regional colorbars and a footer reporting leading 80/90/95 percent
  counts, unit counts, eligible/total choice counts, bin size, and fitting scope.

Start with the full selected-PC session payload and native range navigation.
If measurement shows that serialization or browser rendering is too heavy,
load only windows intersecting an explicitly selected interval. Keep a compact
full-session behavioral overview and use Streamlit time/jump controls to load
another interval. Panning within the loaded interval remains native.

A native Plotly range slider cannot fetch unloaded data through Streamlit's
selection interface. Do not promise drag-triggered server loading. Use a
stable chart key and `uirevision` where effective, but test actual viewport
preservation in installed Streamlit. An explicit time or trial jump must
deliberately update the viewport. Display-only changes must preserve numerical
results even if a frontend limitation requires restoring the last
control-defined viewport.

The default is an opaque light background with readable labels and captions.
Confirm light/dark preference at the real-session inspection before producing
exported figures, following the repository plotting guidelines. A small-PC
trace view and export tooling are not needed to complete the viewer.

## 6. Caching, persistence, and entry points

### Analysis identity

Use one straightforward identity for the scientific configuration:

- analysis version;
- session and resolved metadata identity;
- selected probe IDs and ordered unit IDs;
- fixed channel/unit selection rules and metadata restrictions;
- choice-window definition and bin width;
- consumed source paths, file sizes, and nanosecond modification times.

Include the trial/event tables, alignment archives, spike assignments,
cluster metadata, and channel-quality files as applicable. Check source
identity before invoking a cached loader; calling the old path-only caches
would otherwise conceal changed files. Reuse the existing low-level readers
within the new keyed loading path instead of changing every existing cache.

PC display ranges, viewport, event visibility, and manual display color range
are not analysis inputs. Changing those settings must neither fit PCA nor
invalidate numerical products. A changed source, bin width, or population
selects another analysis. Record code revision and library versions with the
result; bump the analysis version when calculation changes would invalidate
saved products. Do not introduce whole-repository hashing or a provenance
service.

### Saved products

Use a session-local directory under the original data directory:

```text
<session_root>/analysis_runs/
  session_timeline_pca_<date>_<time>_<configuration_id>/
    results.npz
    config.json
    run.log
    summary.md
    run_session.py
    run_batch.py
```

This keeps numerical results and human-readable outputs in a session-local
analysis directory outside the source repository. The script files are copies
of the actual main and batch entry points used. Do not write experimental
outputs into repository `reports/` or commit them. Reject an experimental
output destination inside the source repository; synthetic tests use pytest's
temporary directories.

`results.npz` contains named arrays, using population prefixes for regional
arrays, and a dictionary named `meta` following existing local NPZ conventions.
Save choice/event columns, time origin, eligibility, IDs, relative bin edges,
unit scaling, components, spectrum, scores, variance totals, and color limits.
The metadata includes generator, analysis version, parameters, units/axes,
source identities, code/library versions, unavailable reasons, and random
seed information. Loading reconstructs the result records without fitting.

The single-session function searches for a completed matching result before
loading raw spikes. By default, reuse it. Explicit rerun creates a new
timestamped directory and never overwrites a previous result. An ordinary
interrupted save must not be mistaken for a completed result; write the final
NPZ through a temporary sibling and rename after success. No checkpoint or
transaction framework is needed.

Stage 2 saves a separate named-array summary associated with the exact Stage 1
result and its basis. Keep the Stage 1 archive unchanged. Use a timestamped
Stage 2 subdirectory containing its summary NPZ, log, short summary, and the
entry scripts used. Reuse a matching completed summary on subsequent opens;
changed metric code/version creates another summary. No raw unit tensor is
needed when complete scores and saved unit-variance totals are present.

### Main script, batch script, and UI

The main entry point exposes ordinary prepare/load, dry-run, and explicit
rerun behavior. Stage 2 subsequently adds an explicit window-summary action
that loads a saved Stage 1 result. These run controls belong at the boundary,
not inside numerical functions.

The batch script accepts a text list of session metadata paths and the same
analysis settings, and delegates to the single-session function. Parallelize
across sessions with `ProcessPoolExecutor`; default to available CPU cores,
capped by session count. Provide `--workers` and a dry-run. Review memory and
storage limits before a real batch; use fewer workers when required. Do not
add within-session worker orchestration or a scheduler.

In Streamlit, an explicit Prepare action creates a missing configuration;
otherwise load matching saved products. Reuse the same preparation function
as the CLI. Cache loaded numerical products using source/result identity and
bounded cache entries. Keep figure construction separate. Event toggles and
navigation must not write new analysis runs. Clearing the app's in-memory
cache does not remove saved results.

Proposed usage, to be documented when these entry points exist:

```bash
uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/session/neural_session.json --bin-width-ms 500

uv run streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata /path/to/session/neural_session.json

uv run -m src.neural_analysis.population.run_session_timeline_batch \
  --session-list /path/to/session_list.txt --bin-width-ms 500 --dry-run
```

Probe arguments are only needed where metadata role assignments are absent.
Do not add a second required scientific configuration file.

## 7. Stage 2: fixed-session-axis window participation

Implement only after the Stage 1 inspection gate, but complete it as part of
this plan. Stage 2 is not an unspecified future enhancement.

For each eligible regional window:

1. Compute sample variance of every available session-PC score along time.
2. Use the saved total sample variance across session-standardized units as
   the denominator. Never substitute a retained-score total or the global
   explained-variance ratios.
3. Sort window contributions in descending order, preserving the original
   session-PC identities. Break exact ties by session rank for reproducibility.
4. Find the minimum count reaching 95 percent and retain the ranked identities,
   individual contributions, and cumulative contributions.
5. Mark zero-total-variance, insufficient-bin, and missing-data windows
   unavailable. Do not emit zero as a dimensionality result.
6. Record basis completeness. The normal implementation uses the full fitted
   basis. If incomplete products are encountered, never renormalize them or
   label a restricted count as the exact minimum over all session axes. If
   their contributions sum to less than 95 percent, report "95% not reached
   by retained basis". Even when they reach the threshold, label the count
   restricted rather than exact over the complete basis.

Use direct NumPy variance, sorting, and cumulative sums. Keep sample-variance
normalization consistent and use a documented numerical tolerance for
conservation and threshold comparisons. Do not form per-choice covariance
matrices, run new PCAs, whiten, or standardize again.

Add a compact point track at choice times for each available region and a
selected-trial readout with the count, contributing PC identities and fractions,
total variance, bin count, unit count, and completeness. No connecting signal
is drawn across gaps. Use the existing jump-to-trial selection for the readout;
the metric does not depend on custom heatmap click events.

The main heatmap keeps its original session ordering. The count may exceed
`B - 1`; do not impose the local-refit rank limit. Show the specification's
descriptive interpretation, including that this measures variation around
the within-window mean rather than sustained mean displacement.

## 8. Tests written before implementation

Use pytest functions and fixtures. Put new tests in:

- `src/tests/neural_analysis/test_session_timeline.py`;
- `src/tests/neural_analysis/test_session_timeline_io.py`;
- `src/tests/neural_analysis/test_timeline_plotting.py`;
- `src/tests/neural_analysis/test_session_timeline_scripts.py`;
- `src/tests/neural_analysis/test_session_timeline_webapp.py`.

Do not create a new validation framework. Test observable scientific and UI
contracts, rather than private helper names or source-text formatting.

### Inputs and selection

- First-trial time zero survives choice filtering and viewport changes.
- Right/left codes, numeric/string flag normalization, original trial IDs,
  and row positions remain correct. Incorrect/unrewarded choices are included;
  experimenter rewards and nonchoices are excluded.
- A choice found only in raw end-of-file events is not synthesized, while a
  completed last saved trial is retained.
- All eligible channels of a probe contribute to one population; named sites
  do not partition it. Good/inside-brain, `unit_channels`, and manual good/mua
  rules produce the same selection as existing functions.
- Identical cluster numbers on different probes remain distinct.
- IRIG bounds exclude only incomplete regional windows; valid silent windows
  remain zero-count data. One unavailable region does not remove the other.
- Missing optional event input permits a choices/neural display.

### Binning and Stage 1 mathematics

- All four widths produce exactly 8/40/80/200 bins.
- Boundary spikes at window start, internal edges, and final excluded edge
  follow the half-open convention, including realistic UTC timestamps and
  choice times not on a session bin grid.
- An independent small expected-count fixture verifies count/rate conversion,
  axis order, and no temporal smoothing. Do not use the existing NumPy binner
  unmodified as the endpoint oracle, since its final-edge behavior differs.
- Hand-checkable data verify pooled unit mean/SD, constant-unit exclusion,
  full unwhitened projection, and the one-variable-unit case.
- Full-spectrum explained variances and direct standardized-unit total
  variance give the expected 80/90/95 percent counts without renormalization.
- Changing displayed rows or viewport cannot change fit inputs, scores,
  signs, total variance, or footer counts.
- Saved within-window unit totals agree with direct standardized activity
  and with complete-basis score variance sums.

### Geometry, navigation, persistence, and orchestration

- Choices at 100 and 114 seconds have windows at 98-102 and 112-116 seconds,
  with an explicit ten-second blank interval and correct cell edges/hover.
- Zero scores and missing windows are represented differently; PC1 is first,
  row identities are stable, and color limits do not depend on visible rows.
- Event toggles preserve numerical products; verified LED intervals retain
  original endpoints and unpaired transitions do not acquire invented bounds.
- Jump controls use original trial IDs and reach choices outside the initial
  viewport. Windows intersecting viewport boundaries retain their true edges.
- The new route is available with behavior and partial neural input, bypasses
  the existing single-probe/trial filters, and preserves other view routes.
- NPZ round trips preserve arrays, IDs, axes, scaling, basis, and metadata.
- A matching result is reused without binning/fitting; changed scientific
  settings or source identity invalidate it; display settings do not.
- Explicit rerun creates a new result; existing outputs are not overwritten;
  incomplete saves are not loaded as completed analyses.
- Main and batch functions share the same single-session computation. Dry-run
  does not compute or write results; workers are capped by sessions/CPUs and
  accept an explicit limit.
- A small seeded synthetic two-probe session exercises preparation, saving,
  reloading, figure construction, missing-region behavior, and cache reuse.

### Stage 2 mathematics

- Complete-basis PC variance sums conserve direct unit-space total variance.
- Contributions on PC2/PC7/PC12 return count 3 and those identities without
  changing heatmap order.
- A rank-one diagonal trajectory spread over many fixed axes can require
  more than `B - 1` fixed axes; no clipping is applied.
- A sustained nonzero score with no within-window variation is unavailable
  when total variance is zero, not high participation from mean energy.
- Incomplete bases and zero/insufficient-data cases receive explicit statuses.
- Metric counts/contributions are unchanged by displayed-PC changes.
- Stage 2 reuses saved basis/scores/totals without fitting or loading raw spikes.

### TDD and regression workflow

For each implementation increment: add relevant tests, run them to establish
the expected RED failures, commit tests only, implement, rerun to GREEN, and
refactor while tests remain passing. Record commands and outcomes in the
implementation handoff. Do not change tests merely to obtain GREEN.
Commit only the intended paths: the user's already staged specification and
other pre-existing changes must not enter test or implementation commits.

Run the new focused tests throughout. At the integrated Stage 1 and Stage 2
boundaries, also run the existing population PCA, interregional PCA,
population-package, webapp-package, PSTH webapp, session-metadata, and neural
quickstart tests. Run additional existing suites only when an edit touches
their responsibilities or an observed failure warrants it.

The browser inspection is separate from figure-object tests: native pan,
range navigation, rerun persistence, hover readability, and responsiveness
must actually be checked in the app.

## 9. Implementation sequence and acceptance gates

| Step | Deliverable | Tests first and completion evidence |
| --- | --- | --- |
| 1. Inputs | Choice table, fixed origin, existing metadata population selection, IRIG completeness, independent unavailability. | Input/selection tests RED then GREEN; verify no new scientific selection policy. |
| 2. Stage 1 preparation | Existing binning/PCA reuse, complete basis and scores, footer thresholds, exact window variance totals, result records. | Binning/PCA numerical tests RED then GREEN; existing PCA regression tests pass. |
| 3. Viewer | Add Plotly, pure figure construction, navigation/PC/event controls, new app route. | Figure and route tests RED then GREEN; inspect a synthetic session in the browser. |
| 4. Reproducible execution | Persistence, cache reuse, main/batch entry points, dry-run, logging, script copies, usage documentation. | I/O/script/cache tests RED then GREEN; complete seeded synthetic end-to-end run. |
| 5. Stage 1 inspection | Run one user-designated real session, inspect behavior/neural alignment and navigation, and measure 500/20-ms preparation and display. | Save run evidence and benchmark table; user inspects and approves Stage 1 before Step 6. |
| 6. Stage 2 completion | Fixed-axis window summary, point/readout display, saved summaries, incremental timing/memory measurements. | Stage 2 tests RED then GREEN; synthetic and designated-session evaluation; document local/cluster decision and final limitations. |

Plan approval authorizes the agreed implementation sequence; it does not
remove the explicit Stage 1 inspection gate. The implementation plan is not
itself approval to run experimental batches. After the synthetic pipeline,
use only the designated real session until its output is accepted. A larger
benchmark session and any multi-session execution require designation of the
sessions of interest. Do not add a new approval ceremony at every small code
increment.

## 10. Performance measurements and decisions

Prioritize correctness and readable NumPy/library calls. Initially retain the
full basis and scores because that is the simplest exact implementation.
Keep per-window arrays; do not build a dense neural grid over session gaps.
Remember that 20-ms bins have 25 times the observations of 500-ms bins.

On the inspection workstation, record session/choice/window/unit/PC counts,
bin width, CPU, available memory, major array sizes, elapsed times, and peak
process memory. Separate:

1. source loading and binning;
2. standardization and session PCA;
3. projection;
4. Stage 2 variance reduction/sorting/counting;
5. cache load and warm display-only updates;
6. figure construction, serialized payload size, and browser interaction.

Use a small standalone benchmark action in the main entry point, with standard
library timing and process-memory measurements. Distinguish measured peak RSS
from calculated array sizes; do not present either as the other. Test 500 and
20 ms on the designated session and a larger available designated session
when practical. Intermediate resolutions remain supported without exhaustive
hardware benchmarking.

If memory is a demonstrated limitation, first project/reduce windows in
ordinary batches and avoid keeping duplicate standardized/raw arrays. If
warm cache copying is expensive, adjust cache ownership at the view boundary
without changing scientific products. Do not change PCA to an approximate or
incremental estimator as a hidden optimization.

If the bottleneck is browser payload, use bounded time loading and PC ranges.
If numerical CPU or memory remains limiting after simple reuse, document the
measurements and propose existing cluster batch execution of the same products.
No new cluster scheduler, live service, GPU, Numba, or distributed framework is
part of this plan. Major optimization or cluster additions need a subsequent
decision based on measurements.

## 11. Dependencies and API checks

Add only Plotly as a new runtime dependency, using `uv add plotly` and the
resulting lockfile. Install it before figure-test RED runs so failures concern
missing behavior rather than an unprepared environment. Do not add a niche
plotting component, schema package, profiling dependency, or browser-test
framework. Use `uv run` for all Python and pytest commands.

Review already established the installed Streamlit version as 1.58.0 and
scikit-learn as 1.8.0. Recheck the actual environment at implementation start.
Verify the installed Plotly heatmap edge coordinates, gap/smoothing controls,
matched axes, range slider, hover custom data, and `uirevision` behavior.
Streamlit's native chart event interface supplies selections, not arbitrary
pan/zoom ranges. API and browser checks must reflect installed versions.

Useful official references:

- [Streamlit Plotly chart](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart)
- [Plotly heatmaps](https://plotly.com/python/reference/heatmap/)
- [Plotly range sliders](https://plotly.com/python/range-slider/)
- [Plotly axis matching](https://plotly.com/python/reference/layout/xaxis/#layout-xaxis-matches)
- [Plotly UI revision](https://plotly.com/python/reference/layout/#layout-uirevision)
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching)

## 12. Optional Stage 3 and completion

Stage 3 is not implemented by this plan. If explicitly selected after Stage 2
evaluation, fit a new within-window PCA to the full included unit activity
using the same session unit scaling, subtract within-window means, and find
95-percent local variance coverage. Keep its results separate and show the
rank ceiling `min(U, B - 1)`. The mandatory heatmaps remain on session axes.
Do not add component matching, local-axis heatmaps, or infer independent
biological computations from either count. Rebin the existing sources if this
optional analysis is selected; do not retain a second large raw activity
tensor in Stage 1 solely for a possible Stage 3.

Stages 1 and 2 are complete when the specified viewer and metric work on
synthetic data and the designated session, relevant regression tests pass,
saved products can be reloaded without recomputation, actual browser behavior
has been inspected, and local performance evidence is recorded. Report any
measured rendering limits or incomplete data explicitly.

Remaining practical input: designate the first real session's metadata path
before Step 5. The scientific input conventions are settled; do not reopen
them through artifact-mask requirements, event reclassification, or additional
population-selection machinery.
