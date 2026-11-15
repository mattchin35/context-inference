# Event sequences: SparsePCA and manual color limits

Status: implementation plan for review, 2026-11-15. Follow AGENTS.md and
docs/SoftwareDesign.md; implementation awaits approval.

This extends docs/event_sequence_heatmap_spec_v1.md and its implementation
plan with a third neural source and the user's selected symmetric color limit.
Existing trial/sequence definitions and ordinary unit/PCA behavior are retained.

## 1. Outcome and scope

Both event-sequence tools will offer these neural sources:

- Single unit: existing manually good unit firing rate.
- Regional PCs: existing ordinary session PCA.
- Regional sparse components: the existing session-timeline SparsePCA method.

Sparse preparation uses up to 30 components per region and adjustable alpha,
initially 1, as in the timeline viewer. The event tool displays one selected
component at a time. Both tools use the same fixed prepared source across all
central-trial conditions, sides, sequence families, orientations, and lengths.

Add an optional manual color limit L for every source type:

| Display | Manual limits |
| --- | --- |
| Raw ordinary PCA scores | [-L,+L] |
| Raw sparse coefficients | [-L,+L] |
| Any available z-scored source | [-L,+L], in source SD units |
| Raw single-unit firing rate | [0,L], in Hz |

L must be a finite positive number. All panels/segments in a figure share the
same limits, and PNG export uses the identical range. Automatic limits remain
the default. Manual limits affect color saturation and legends only: no array
clipping, normalization change, trial exclusion, fitting, or source rewrite.

No new dependencies, estimator implementation, behavioral selector, chart
component, metadata field, or batch framework is needed. Do not add per-panel
scaling, arbitrary lower/upper bounds, automatic parameter tuning, or motif-
specific decomposition. The existing light plotting theme remains in use.

## 2. Scientific/data contracts

### Sparse computation

Use the existing `population/timeline_sparse_pca.py` through the current
session-timeline preparation runner. Do not implement another SparsePCA fit in
the event package or call ordinary PCA first to obtain sparse components.

The upstream computation remains:

1. Select the requested region/probe and current manual-quality population.
2. Use all ordinary completed choices with full regional recording coverage,
   regardless of outcome or event-sequence membership.
3. Bin unsmoothed counts into existing [-2,+2) s windows at the chosen width.
4. Standardize units once across all eligible window/bin observations, using
   the existing means and population SDs; omit constant units consistently.
5. Fit `K=min(30,W*B,U_retained)` sparse components. Use the existing sklearn
   ridge transform for float64 scores `(W,B,K)`, not a loading dot product.

Keep the timeline's actual fixed estimator settings: ridge alpha 0.01, LARS,
1000 maximum iterations, tolerance 1e-8, seed 0, and one sklearn job. Alpha is
finite/nonnegative and user-adjustable. Settings/cap/version come from existing
code rather than another copied set of defaults.

One selected component yields `(W,B)` signed ridge coefficients. Display IDs
are SC1, SC2, etc., preserving the upstream component order and signs. They
are not explained-variance ranks. Sparse neuron weights do not require sparse
activity over time. No PCA variance percentages or participation counts are
introduced into these tools.

The current timeline cap-30 CT026 fits reached the iteration limit. Event
views must retain and display that warning, including when the fit is reused.
Finite coefficients remain inspectable; do not label them as certified
converged results or silently increase the iteration budget.

### Population conventions

Preserve the existing event-source population selector: manual good + MUA
(default) or manual good only, on good inside-brain channels intersected with
metadata unit_channels. Event preparation excludes missing manual labels;
single-unit requests remain manual-good only. Do not change these conventions.

The timeline's older default missing-label fallback can select a different
population. Therefore an existing timeline file is reusable only if its actual
ordered input unit IDs/channels, sources, coverage-eligible choice pool, bin
geometry, and sparse settings match the event request. A shared probe name,
displayed unit count, or matching retained units alone is insufficient: input
IDs include units later omitted as constant.

### Sequence selection and display normalization

Keep all original row positions, trial identities, coverage rules, missing-
condition handling, chronological ordering, and additional-selector behavior.
Central panels and multi-trial sequences assemble the same saved component
values; an episode with an uncovered segment remains excluded as today.

Optional z-scoring uses the selected source's mean and population SD over all
unique eligible session choice-window bins, before selecting motifs. Repeated
appearances of one trial in different episodes never change that reference.
Zero-SD sources retain valid raw activity and disabled z-scoring. Entirely zero
sparse components keep their IDs and their existing availability diagnostics.

Shapes and units remain explicit:

- Upstream scores: float64 `(choice_window,time_bin,component)`.
- Selected source: float64 `(choice_window,time_bin)` with original row map.
- Central logical panel: `(matched_trial,time_bin)`.
- Sequence logical panel: `(episode,trial_segment,time_bin)`.
- Relative time: choice-relative seconds, with the existing display separators.
- Raw unit activity: Hz; raw sparse activity: ridge coefficients in
  standardized-unit activity space; z-scores: dimensionless source SD units.

## 3. Targeted modules and interfaces

Extend the existing modules with small explicit branches. No new production
module, source registry, class hierarchy, or alternate saved-product tree is
planned. Preserve current public calls and default arguments.

| Module | Targeted responsibility |
| --- | --- |
| `event_sequences/run_session.py` | Extend EventSequenceRequest with sparse source kind and sparse-only alpha; resolve identity, reuse/prepare the upstream sparse basis, and save the usual event packet. |
| `population/session_timeline_io.py` | Extend regional compatible-source lookup with optional algorithm/alpha selection and sparse-specific validation, preserving ordinary defaults. |
| `event_sequences/io.py` | Recognize sparse event metadata, bind the exact sparse archive, and validate component/unit/choice/bin contracts without copying score tensors. |
| `event_sequences/activity.py` | Select/label sparse components and their units using existing row mapping; preserve common scaling and assembly. |
| `event_sequences/plotting.py` | Add optional manual color-limit preparation and sparse labels/diagnostics to the shared display/view payload. Render/export the same arrays and limits. |
| `webapp/event_sequence_views.py` | New source/alpha selection, sparse component selector, warning/status text, manual color controls, and correct saved-source matching. |
| `event_sequences/run_batch.py` | Reuse its existing request-driven session workers. Update documentation/types only if needed; no new scheduler or duplicate preparation path. |

Document every new/modified function's input types, shapes/axes, units, and
return values. Keep source computation separate from matching and plotting.

### Request/source selection

Add `source_kind="sparse_pc"` alongside the existing `unit` and `pc` values.
Use the existing region, probe override, bin width, and population controls.
Add keyword-only `sparse_alpha: float | None = None` to EventSequenceRequest:
None resolves to 1 for sparse requests; sparse-only settings on unit/ordinary
PC requests are rejected rather than ignored.

Preserve existing resolved dictionaries/digests for unit and ordinary PCA;
do not insert new default None fields into their canonical configurations.
Sparse configurations include the decomposition version, actual estimator
settings/seed, cap 30, alpha, and sklearn version, reusing existing parameter
helpers. Rank/component selection and all color settings remain display-only.

Keep public `pc_rank` arguments usable to avoid a needless interface rename;
document them as one-based component indices for sparse sources. Internally,
select_pc_activity can handle the corresponding saved transform with explicit
source kind, SC label, and ridge-coefficient units. Expand SourceActivity's
allowed kind and signed-value rules without duplicating its row-map logic.

## 4. Compatible upstream reuse

Extend `find_compatible_regional_timeline(...)` with keyword-only
`algorithm="pca"` and `sparse_alpha=None`. Existing callers, including
cross-condition decoding, must continue to request ordinary PCA by default.

Retain the current metadata-first candidate scan and shared scientific source,
trial-pool, and geometry checks. Verify the candidate's canonical digest using
its complete original configuration, including the sparse-specific fields;
only then compare the requested region. A two-region archive may supply a
matching single requested region without requiring an identical other probe.

For ordinary PCA, retain full-basis, orthogonality, explained-variance, units,
and legacy-registration checks. Do not relax these to admit sparse archives.
For sparse requests, check the sparse analysis version, capped policy/settings,
actual ordered input/retained/omitted units, expected K, finite `(W,B,K)` scores,
recorded scaling, and sparse units/axes. Ordinary or incompatible full-count
sparse products cannot substitute for the requested capped fit.

If no validated compatible sparse basis exists, the explicit event Prepare
action calls run_session_timeline for only the requested region with
algorithm="sparse_pca", the selected alpha, and the existing strict event
population policy. Preserve the current forced-new-upstream fallback when
compatible lookup fails, rather than trusting metadata-only reuse of a rejected
candidate. Never overwrite a previous upstream result.

Display controls never call this lookup/preparation path. Normal event-product
reuse keeps its exact original upstream reference even if a newer fit appears.
Event Recompute creates a new event run and can reuse a compatible upstream
basis; it does not itself request a new sparse fit.

## 5. Event persistence and saved-only viewing

Use the existing `event_sequence_heatmaps_<date>_<time>_<id>` layout. Sparse
event products use analysis version `event-sequence-sparse-heatmaps-v1` with
schema 1 and the existing packet structure. Unit/ordinary-PC products keep
their current analysis version, schema, cache identities, and loading rules.

Retain the `pca_reference` field name as the shared decomposition reference,
avoiding a migration or duplicate reference. It binds the exact session-relative
path, byte size, SHA-256, configuration ID, region, and probe. Sparse products
save choices, row maps, bins, matches, and metadata; they do not duplicate the
upstream `(W,B,K)` score tensor or pickle an estimator.

Add an explicit sparse branch to metadata validation and archive binding.
Require agreement between event source kind/settings and the actual upstream
SparseTimelineTransform, cap/count, input IDs, score units, and saved choice
identity. Cross-algorithm substitution must fail even when array dimensions
coincidentally match. Preserve exact-byte and path-containment checks.

Sparse provenance includes upstream seed/settings, iteration count/limit and
zero-component diagnostics, algorithm/code/library versions, and units/axes.
Distinguish stochastic upstream fitting from deterministic event matching;
do not retain an unqualified claim that the sparse pipeline used no randomness.
Convergence status is taken from the bound fit, not inferred from event counts.

Saved display must continue to work without current CSV, spikes, quality files,
or behavior-model configuration: load only the event packet and its exact
referenced decomposition. Missing/modified references report unavailable data,
without a fallback fit, silent rebinding, or NPZ rewrite. Additional motif
matches remain runtime derivations using the current selector-version cache key.

## 6. Manual color-limit behavior

Extend `prepare_display_source(..., color_limit: float | None = None)` with an
optional trailing keyword. None keeps the exact existing automatic convention:
all eligible source bins determine the range, independent of displayed motifs.
For a supplied finite positive L, set [-L,+L] for signed/z-scored activity or
[0,L] for raw Hz after any requested display z-scoring. Reject zero, negative,
nonfinite, boolean, and nonnumeric inputs with a useful message.

Keep this override in display preparation, not the scientific request or
stored event packet. Propagate limits and auto/manual mode into the common view
payload/caption. Existing Plotly shared coloraxis and Matplotlib vmin/vmax then
use the same range and existing palettes. Do not add a second independent
export-normalization path or call np.clip on activity arrays.

UI controls, after selecting a prepared source/component and scaling:

- **Manual color limit** checkbox, initially off.
- When enabled, one positive numeric L, initialized from that source's current
  automatic maximum magnitude; label its current score/Hz/SD units.
- Raw Hz explains that its range is 0 to L; signed/SD activity explains -L to L.
- Unavailable sources disable the numeric control. Constant raw sources remain
  viewable with a valid range and their existing constant-activity message.

Keep manual values/toggle state in one plain, non-widget session-state
preferences dictionary, keyed by session, saved source configuration,
component/unit, and raw versus z-scored mode. Temporary widget keys alone are
insufficient: Streamlit can remove them when a source/scaling control disappears.
Restore the selected source's remembered preferences before rendering controls,
then save valid returned settings to that dictionary; do not overwrite an
already-rendered widget's state. This adds no on-disk preference storage.
The preference key must exclude which event
tool, side, outcome, motif family, direction, or episode length is displayed.
Thus both tools and all subsets of one selected source share its chosen range,
while incompatible units/scales do not inherit one another's values.

The caption reports auto/manual bounds and units, noting saturation outside a
manual range. Hover retains actual original or z-scored values, including values
beyond L. Export PNG consumes that exact view, including its source/configuration,
SC identity, scaling, limits, and convergence warning. Existing explicit export
and unique-path/no-overwrite behavior remains in place.

## 7. Preparation and viewer interactions

Expose Regional sparse components in Neural source. Reuse the regional source
configuration/browsing flow with sparse alpha and clear saved labels including
algorithm and alpha. Saved sparse and ordinary products are distinct choices;
all request reconstruction and matching helpers include sparse settings.

Sparse Prepare/Recompute is explicit, as for regional PCs; only the existing
single-unit selection retains its current automatic preparation behavior.
An unprepared sparse alpha/population/bin configuration shows an explanation
and Prepare, not a differently configured saved heatmap. Changing SC index,
z-score, limits, or motif never creates another event run or upstream fit.

Use **Sparse component** for the one-based selector and SC labels in captions
and hover. Respect the actual available K, including K=1 or zero-loading rows.
Surface upstream iteration-limit warnings in both tools and PNG captions.
Keep the central six-panel layout, sequence-local time coordinates, separator
geometry, source availability, behavioral counts and export buttons unchanged.

The existing callable main/request and batch entry points remain the preparation
interfaces. Document a Sparse EventSequenceRequest example; do not invent another
CLI when this package currently exposes a callable request API.

## 8. Tests written before implementation

For each package below: write tests, observe RED, commit the tests, implement
GREEN, then refactor with tests passing. Use pytest functions and current
synthetic fixtures; extend existing files rather than another test harness.

1. **Requests and identity:** sparse defaults/alpha validation; same region/bin/
   quality rules; sparse settings distinguish identities; existing unit/PCA
   configuration dictionaries and digests remain unchanged; rank/L stay absent.
2. **Compatible reuse:** a matching region in a two-region capped sparse archive
   is reused without spikes or fitting. Wrong alpha, cap, algorithm, input unit
   IDs, constants, source identity, coverage pool or bins rejects reuse. Unrelated
   probe/event-display changes do not invalidate the requested regional basis.
3. **Default callers:** ordinary event and cross-condition searches cannot return
   sparse archives; legacy ordinary PCA validation remains strict.
4. **Explicit preparation:** no compatible sparse fit creates exactly one
   requested regional upstream fit and an event product. Repeated Prepare reuses
   the pinned result; event Recompute creates a new event run without unnecessary
   upstream fitting. Dry-run performs no neural reads/writes/workers.
5. **Selection/math:** selected SC values and original row/bin mapping exactly
   equal the referenced timeline tensor. Session z-scoring uses all unique
   eligible bins and stays unchanged across motif/panel selection. Signed and
   zero-component behavior remains correct; no second estimator implementation.
6. **Persistence/offline display:** sparse packet round trip, exact upstream
   binding, units/seed/diagnostics, no score duplication, relocated contained
   references, and no raw-input reads during display. Reject changed hashes,
   malformed mappings/units/settings, and cross-algorithm references. Preserve
   ordinary packets, paths, bytes and registrations.
7. **Manual color math:** all source/scaling combinations produce the specified
   ranges; automatic defaults and constant/unavailable behavior are unchanged.
   Invalid L fails; values, masks, means/SDs and counts are never changed.
8. **Rendering/export:** every central panel and sequence segment uses the same
   limits in Plotly and Matplotlib. Saturation preserves hover values; captions
   identify bounds/mode/units and sparse warning. PNG stays opaque and refuses
   overwrite, including empty/n=1/additional-family cases.
9. **Actual Streamlit controls:** source/alpha matching, explicit-only sparse
   preparation, saved offline browsing, SC selector, warning display, manual
   toggle and shared raw/z-score state across both tools. Patch preparation,
   binning and fitting to fail during live display changes.
10. **Batch and regression:** reuse existing synthetic request batch/spawn tests,
    with sparse requests supported. Run all event-sequence/additional-selector/
    unit-selector/webapp tests, shared timeline sparse/storage/identity tests,
    and ordinary cross-condition regional-lookup consumers.

Use uv run for all Python commands and Ruff on changed Python files. Check the
addition's whitespace and preserve unrelated working changes. No test is changed
merely to accommodate implementation; document genuine fixture corrections.

## 9. Validation, performance, and completion

No experimental preparation occurs during planning. After approval and passing
synthetic tests, inspect CT026 2026-08-04 at 500 ms, alpha 1, cap 30, initially
one region and the existing good+MUA selection. Check whether its previously
prepared capped timeline is actually compatible before allowing a new fit.
Record reused/prepared status and reason; do not force incompatible reuse.

Exercise both central and multi-trial tools, including the added motif families,
SC selection, session z-scoring, small and large manual limits, empty panels,
auto restoration, cross-tool state, raw-unit/ordinary-PC regression, convergence
warnings, and PNG export. Save a labeled PNG/browser preview and short inspection
notes for user review. The current CT026 upstream cap-30 fit is explicitly
convergence-unconfirmed; do not present event displays as resolving that issue.

Measure new preparation versus reuse, first saved load/reference validation,
component selection, color-only rerender, Plotly construction/serialization,
and PNG export separately. Record W/B/U/K, file/array/payload sizes, machine and
process-lifetime peak memory with its limitation. Archive hashing and tensor
loading should remain behind the existing saved-data cache, not recur solely
because L changed. Basic scaling of one `(W,B)` source should remain inexpensive;
no performance guarantee is inferred without measurements.

Reuse existing session-level spawn workers, with one sklearn job per session.
No experimental batch or finer-bin SparsePCA run is part of initial validation.
No resampling, fit subsampling, alternative solver, iteration-budget change,
or major optimization is introduced. Report a measured bottleneck before
proposing a separate optimization.

New outputs stay under the original session analysis_runs in new timestamped
directories, with named-array NPZ/meta, parameters, versions, source identities,
units/axes, runtime/log, main/batch snapshots, and a human-readable summary.
Preserve existing event, timeline and reference archives. Add usage to
event-sequence documentation/README and record validation in this plan, without
including unrelated user edits in commits.

Implementation order:

1. RED/GREEN regional sparse compatibility and request/preparation selection.
2. RED/GREEN saved sparse references, selected activity/scaling and diagnostics.
3. RED/GREEN pure color override and browser/PNG rendering agreement.
4. RED/GREEN UI controls, saved-source matching, cross-tool state and batch
   regressions; then integrated tests and the designated-session inspection.
5. Usage/validation documentation and review artifacts.

Complete when both tools display the intended prepared SC values, fitting is
explicit and reusable, old sources still work, one shared manual L reaches both
renderers without scientific changes, tests pass, and the single-session result
is ready for user inspection. Implementation remains pending plan approval.
