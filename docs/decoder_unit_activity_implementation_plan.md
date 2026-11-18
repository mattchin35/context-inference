# Decoder-weight unit activity: implementation plan

Status: implemented and verified on 2026-11-15; CT026 output awaits user inspection
before any wider experimental batch.

Follow [AGENTS.md](../AGENTS.md) and [SoftwareDesign.md](SoftwareDesign.md).
Scientific sources: [population decoding](population_task_decoding_spec_v1.md)
and [within-side switching](population_switch_vs_stay_implementation_plan.md).
This version includes the corrections from both design reviews.

## 1. Purpose, confirmed choices, and scope

Select units by their population-decoder weights and show their actual firing
activity around successive choices. Confirmed choices:

- Actual elapsed session time, chronological choice windows, and real gaps.
- Within-side switch-versus-stay weights for L->R and R->L.
- One display group per unit, chosen by its strongest relative category rank.
- Coefficients available in all five folds; expected sign in at least four.
- Up to 10 units per group by default, with an adjustable positive integer cap.
- Per-unit display z-scores with a Hz toggle, and light-mode plots.

Use saved `population_decoding` direct-unit results independently for PFC/HPC.
The source models already use L2 logistic regression (C=1, l1_ratio=0, lbfgs).
Saved coefficients, fold masks/scalers, and regional Hz activity supply the
inputs. Existing decoder products and fit/resume identities remain unchanged.

The source defines the available window and resolution: [-1,+1) seconds and its
saved bin width. A 500-ms run has only four bins per choice window. Show the bin
width beside the chart and in exports; preserve native bin edges and values.
Zooming does not increase temporal resolution. Finer display requires a
different prepared source, which this feature does not create.

PC back-projection, new fits, raw spike loading, continuous between-choice
activity, resampling/smoothing, PSTHs, and significance testing are outside this
version. The algorithms and overlap layout below were approved when the user
requested implementation. The user subsequently designated the existing CT026
2026-08-28 v2 decoder run for the first experimental inspection.

## 2. Groups and coefficient meanings

Keep this display order. Binary beta is the coefficient for saved class 1
versus class 0. For outcome class k, define
`contrast[k] = beta[k] - mean(beta[other three classes])`.

| Group | Saved model | Oriented fold weight |
| --- | --- | --- |
| Left context | `context` | +beta; right=0, left=1 |
| Right context | `context` | -beta |
| Left choice | `choice` | +beta; right=0, left=1 |
| Right choice | `choice` | -beta |
| Left reward | `outcome` | contrast[0] |
| Right reward | `outcome` | contrast[1] |
| Left no reward | `outcome` | contrast[2] |
| Right no reward | `outcome` | contrast[3] |
| L->R switch | `left_switch_vs_stay` | +beta; stay=0, switch=1 on current-left trials |
| R->L switch | `right_switch_vs_stay` | +beta; stay=0, switch=1 on current-right trials |

Validate saved class codes/names and conditioning side before using these rules.
Weights describe logit or relative-logit changes per corresponding training-fold
unit SD, rather than per Hz. Fold-specific SDs differ, so the averaged weights
are a descriptive summary, not the coefficients of a new common model.

The outcome contrast is invariant to a shared shift of class coefficients. It
does not remove chosen-side information or define a one-versus-rest classifier.
Outcome groups can therefore contain side-related units. Alongside four-class
scores, display the saved within-actual-side reward/no-reward balanced accuracy,
log loss, reward ROC-AUC, and unavailable reasons. Label support separately as
eligible trials and required evaluation bins: derive per-class trial counts
from the cell's original row positions and one-label-per-trial vector; the saved
score's `class_counts` counts required bins, including bins without predictions.
Include required/predicted bin counts when predictions are incomplete. Preserve
the saved pooled out-of-fold scores and their availability rules. These
are population diagnostics, not tests of an individual unit's reward selectivity.

Groups describe decoder weights. Correlated features share weights, and high
relative ranks can occur in uninformative models. Four-of-five sign agreement
uses overlapping training sets and is not a confidence interval. Display the
existing model performance without introducing an automatic performance cutoff.

## 3. Selection algorithm

Apply these steps independently to one saved run and region.

1. **Restore unit identity.** Validate the source contracts in section 6. Expand
   each fold's retained-feature coefficients onto the original ordered unit
   axis. Omitted constants and unavailable fits receive NaN with reasons.
   Construct oriented weights with axes (5 folds, 10 groups, U units).
2. **Summarize.** Only group/unit pairs with five usable coefficients receive
   a ranking mean. Compute their arithmetic mean and strictly positive fold
   count. Candidates require mean > 0 and positive count >= 4; zero is not a
   positive sign. Preserve all five weights, counts, and exclusions.
3. **Rank within categories.** Rank signed means across all five-fold-usable
   units in each group, before the sign filter. Use ascending average ranks
   for exact ties. For M rankable units and rank r, percentile = (r - 0.5)/M.
   M=0 gives unavailable ranks; M=1 gives percentile 0.5.
4. **Assign uniquely.** Among a unit's candidate categories, choose its highest
   percentile. Exact ties use positive-fold count, then the fixed group order.
   Record ties and the runner-up category/percentile. Units without candidates
   remain unassigned.
5. **Cap after assignment.** Within each assigned group, sort by descending
   percentile, then unit ID for exact ties; display at most the requested cap.
   Do not move units to second-best categories to fill groups. Empty groups
   retain a summary entry with counts/reasons.

Percentiles implement relative rank; raw magnitudes across classifiers are not
compared. Assignment considers all supported groups before display filtering.
Changing cap, scale, or viewport preserves each unit's assigned category.
Different target training pools are joined on unit identity, not trial rows.

## 4. Activity and temporal display

Display actual rates or display z-scores, without multiplying by weights.
Use the same regional covered baseline windows for every selected unit,
including windows ineligible for an individual target when measured activity
exists. Outcome units show pre-choice activity, and switch units show choices
to either side. Captions distinguish the displayed [-1,+1) interval from the
outcome model's [0,+1) supervised interval.

Normalize each selected unit over all its saved regional window/bin
observations using its mean and population SD (ddof=0), before viewport
filtering. Reuse `zscore_window_activity`; save means/SDs and constant flags.
Overlapping windows retain the existing observation weighting, including
repeated time coverage. In Hz mode, preserve all measured rates, including
zero and nonzero constants. In z-score mode, zero Hz for a variable unit maps
to `-mean/SD`, and a constant unit maps to zero. Measured zeros remain valid
observations in both modes; only missing data and display gaps/padding are NaN.

Use a shared color range across groups: symmetric around zero for z-scores,
starting at zero for Hz. Automatic limits use the full selected-unit pool and
remain fixed during panning. A changed selected set may change those limits;
record the limits and scale. Color saturation must not alter saved values.
The subsequently approved color-control extension adds Auto/Custom mode for
z-scores: one positive finite SD limit sets a shared symmetric range. It is
applied after cached activity preparation, preserving values and normalization.
PNG exports and optional `--unit-sd-limit` CLI/batch arguments use the same
range, recorded with effective limits in export settings. Hz remains automatic.

Reuse choice/context/event tracks, elapsed-time navigation, and original
trial/bin identities. Unit labels appear once within their assigned group.
Hover shows unit/group, weight summary, trial ID, native bin width, and original
choice-relative and elapsed bin bounds. Other-category weights and ties appear
in an inspection table.

### Overlap policy: separate sublanes

Never paint one choice window over another. Assign the full regional window
pool to temporal lanes in chronological order, using the lowest-numbered lane
whose previous window ends at or before the new window starts. Touching
half-open windows do not overlap; create a new lane only when all existing
lanes are occupied. Lane assignment precedes viewport filtering.

If L lanes are needed, give each unit row L labeled sublanes. A source window
is drawn only in its assigned sublane, with exact temporal edges. The unit
keeps one label and one group membership; row height scales with L for
readability. Empty lane intervals stay blank. Explain the sublanes and report
the overlap count in the caption.

Skip rendering lanes with no windows in the loaded viewport. For each remaining
lane l, reuse the existing nonoverlapping geometry helper to obtain native x
edges (M_l+1,) and values (S,M_l), including gap columns. For zero-based selected
unit index s and lane l, define display row `r = s*L + l`. Pad each lane's values
to (S*L,M_l): populate only rows assigned to l and leave other rows NaN. Both
renderers use explicit y edges 0..S*L, with row 0 at the top. Label unit s once
at `s*L + L/2`; put group separators at unit boundaries multiplied by L.

Render one heatmap per lane using its native x edges, the common y edges, and
shared color limits. Padding stays transparent and has no hover entries, so
traces cannot cover another lane's data. Preserve the row-to-unit/lane mapping
in hover/export data. This adds display rows without changing the source
(W,B,S) activity axes. Preview and PNG use the same layout.

## 5. Architecture and data contracts

Create one shallow sibling package, `src/neural_analysis/decoder_unit_activity/`.
The decoder fingerprints every Python file in `population_decoding/`; keeping
this derived feature separate avoids changing numerical fit/resume identities.

| Module | Responsibility and principal interfaces |
| --- | --- |
| `selection.py` | Pure `build_group_weights(cells, unit_ids)`, `summarize_group_weights(weights, group_ids, unit_ids)`, and `assign_unit_groups(summary, max_units_per_group)`. Return arrays and labeled summary/display tables. |
| `results.py` | `load_decoder_inputs(run_directory, region)`, `save_unit_selection(path, arrays, meta)`, `load_unit_selection(path)`: source validation and persistence using public source readers. |
| `plotting.py` | `build_unit_activity_data(...)`, `plot_unit_activity(...)`, `export_unit_activity_png(...)`: prepare shared values/geometry and render both formats. |
| `run_session.py` | Main script: assemble one source run/region, compute or reuse and save selection, then reload it for rendering; support dry-run and plot-from-selection. |
| `run_batch.py` | Batch script: validate section 7's request list and invoke the same single-session orchestration. |
| `__init__.py`, `README.md` | Package marker and usage/interpretation instructions. |
| `webapp/decoder_unit_activity_view.py` | Controls, summaries, and chart assembly. Existing `population_decoding_view.py` gains a subview selector and dispatch. |

Use small functions and ordinary arrays/DataFrames; document types, shapes,
units, missingness, and return values. Numeric helpers receive their required
arrays/axes explicitly. Reuse public source loaders, timeline behavior/geometry/
normalization helpers, navigation, and chart rendering. Preserve public
interfaces and use the existing dependencies; verify unfamiliar APIs before use.

Let F=5, G=10, U=regional units, W=covered windows, B=native bins, S=selected units.

| Boundary | Contract |
| --- | --- |
| Activity | Float (W,B,U), Hz; IDs (U,), original row positions (W,), relative edges (B+1,) seconds. |
| Group weights | Float (F,G,U), logit/relative-logit per training unit SD; NaN plus reasons for unavailable entries. |
| Summary | Means/percentiles (G,U), available/positive counts (G,U), candidate mask (G,U). Undefined values remain NaN. |
| Membership | Assigned group indices (U,), -1 unassigned; ordered display unit/group indices (S,), S <= min(U,G*cap). Preserve region/probe/unit identity. |
| Display values | Means/SDs (S,) Hz, constant flags (S,), values (W,B,S) in Hz or dimensionless SDs, before lane padding. |
| Heatmap geometry | Lane IDs (W,); per lane l, x edges (M_l+1,), padded values (S*L,M_l), shared y edges (S*L+1,), and explicit row-to-unit/lane and column-to-trial/bin maps. Padding/gaps are NaN; source values retain their units. |

## 6. Source validation and availability

The new loader owns coefficient/fold validation; the current public source
readers do not validate that entire contract.

- Match model/preparation identities, region, ordered unit IDs, class
  definitions, bin width/edges, and target supervised intervals. Check the
  preparation's probe ID against the source config's region/probe mapping.
- For a present region and published model, require exactly five units-fit
  records with unique integer fold IDs {0,1,2,3,4}. Index by ID, not list position.
  Missing/duplicate/unknown IDs are malformed data, not unavailable fits.
- For each available fold, require a Boolean retained mask (U,), finite
  coefficients (1,P) for binary or (4,P) for outcomes, and P equal to mask.sum().
  Validate finite training means/SD arrays (U,) in Hz and strictly positive
  retained-unit SDs.
  An explicitly unavailable fold may lack those arrays and retains its reason.
- If the region is absent from the preparation with a recorded unavailable
  reason, return region-unavailable before coefficient processing. The UI shows
  that reason; CLI returns an unavailable report without creating a selection.
  An absent region without an explanation, or contradictory available-fit data,
  is a malformed source.
- V1 supplies eight context/choice/outcome groups; its two within-side switch
  groups stay unavailable. Never substitute all-trial switch weights.
- In v2, unpublished required cells are pending: defer selection/export until
  they are published. Published unavailable folds produce empty affected groups.
  Corrupt files and inconsistent identities raise errors rather than becoming
  missing weights.

## 7. UI, persistence, and execution

Inside **Population task decoding**, retain **Probability timelines** as the
default and add **Decoder-weight unit activity**. Share run/region navigation;
the new view exposes cap and z-score/Hz controls. Show source bin width, group
counts/reasons, and source performance including section 2's outcome diagnostics.
Place fold weights, alternate memberships, and ties in an expander. Empty
selections show a message. Preserve viewport during cap/scale changes.

Selectors may compute cached summaries of saved data. Export explicitly saves
the selection/settings, reloads them, and renders with matching preview values,
order, lanes, limits, viewport, and captions.

Save intermediate arrays at
`<session>/processed/decoder_unit_activity/<selection_id>/selection.npz`.
Include weights, summaries, counts, masks, membership/display indices, labeled
axes, normalization statistics, and a `meta` dictionary serialized as JSON in
the NPZ and restored on load. Reference the existing activity archive rather
than duplicating it.

Bind selection identity to source IDs/content hashes, region, rules/cap, and
selection/normalization code version. Metadata includes generator, analysis
version `decoder-unit-activity-v1`, parameters, units/axes, source definitions/
performance/exclusions, runtime versions, randomness_used=false, and
random_seed=null. Use session-relative references where possible and validate
referenced files on reload.

Human-readable outputs go in
`analysis_runs/decoder_unit_activity_YYYYMMDD_HHMMSS_<id>/`: PNG,
`unit_groups.csv`, settings/selection reference, summary, log, and copies of
main/batch scripts and directly used new modules. Record goal, session/source,
scripts, counts, interpretation, warnings, runtime, and failures. Captions
include native resolution, windows, normalization, limits, selection rule, and
overlap layout. Reuse a validated identical intermediate; changed inputs select
a new one. Renders use fresh timestamped directories without overwriting files.

The single-session script takes a source run and one region. Batch input is a
JSON array with one entry per session:

```json
[
  {"source_run": "analysis_runs/population_task_decoding_RUN", "regions": ["PFC", "HPC"]}
]
```

Resolve relative source paths against the request file directory. Each entry
has exactly those two fields; regions is a nonempty unique subset of PFC/HPC.
Require source runs under their session's analysis_runs directory and reject
duplicate session roots. Cap/scale options apply to the invocation. Process
regions sequentially within a session and parallelize across sessions with the
existing CPU/session worker cap and an override for RAM/I/O constraints.

Dry-run reports planned sources, availability, dimensions, outputs, and workers
without writes. Batch reports unavailable/pending/failed requests explicitly,
continues independent sessions, and returns partial status when any requested
region cannot complete. No new checkpoint or resume framework is needed.

## 8. Tests-first phases and acceptance

For each phase: write pytest tests, confirm RED, commit tests alone, implement
GREEN, then refactor. Use `uv run`; preserve unrelated work. These tests will
be written before implementation:

| Phase | Tests and independent checks |
| --- | --- |
| U1: coefficient mapping | Exact binary signs, four outcome contrasts, switch directions, class/side checks, retained-mask restoration, axis identities, and input nonmutation using handcrafted coefficients. |
| U2: selection | Five-fold means, four-positive versus three-positive, zero/negative means, unavailable coefficients, M=0/1, average-rank ties/tie-breaks, unique assignment, empty groups, and assignment before cap with no backfilling. |
| U3: source/persistence/scripts | Missing region with/without reason; reordered/duplicate/missing fold IDs; malformed masks/coefficients/scalers; v1 eight groups; v2 pending/unavailable cases; source identity mismatch; round trips, corruption, reuse/non-overwrite, dry-run no writes, JSON batch validation, partial batches, logs/script snapshots, and unchanged decoder bytes/code identity. |
| U4: activity/exports | Known Hz values, irregular IDs and row mapping, native resolution/elapsed edges/gaps, outcome pre-choice and both-side switch activity, variable-unit zero Hz -> -mean/SD, nonzero constants preserved in Hz and zero in z-score mode, and valid zeros distinct from NaN. Test two units/two lanes with exact padded rows/y edges, transparent padding, one label per unit, touching windows, no occlusion, stable lanes during panning, shared limits, opaque white PNG, and Plotly/PNG geometry agreement. |
| U5: webapp | Raw sources removed; model fits, raw spike loading, and preparation forbidden while saved-result reads remain allowed. Exercise subview/run/region/cap/scale/export; resolution, within-side outcome diagnostics/reasons, separately labeled trial/bin support, identical trial counts across different bin widths, cached summaries, viewport retention, empty regions/selections, and existing routes. |
| U6: synthetic integration | Seeded positive/negative/shared-category signals plus constants; saved fit -> selection -> save/reload -> render/export; source-free reproduction and batch failures. Include a side-only signal with balanced reward/no-reward control to verify that outcome diagnostics expose side-driven decoding without claiming reward selectivity. |

Use handcrafted expectations for exact ranks/membership; fitted synthetic
checks verify recovery of strong planted signals without brittle ordering of
correlated units. Run the new tests, existing population-decoding tests, and
directly affected timeline/chart/routing checks. Record RED/GREEN evidence and
measured runtime, memory, and output sizes.

Coefficient storage is O(F*G*U), ranking O(G*U*log U), and display normalization
O(W*B*S). Cache small summaries of all five models together: the existing
four-entry cell cache would otherwise churn. Reuse the preparation and release
unused probability arrays; current NPZ loading may still read all regional
units. Restrict heatmap/hover payloads to selected units and the loaded viewport;
panning reuses ranks, normalization, and global lanes. Sublane padding adds
O(S*L*sum_l M_l) displayed cells, including gaps; account for this allocation
when measuring memory. Measure before optimizing.

Acceptance requires correct mapping/eligibility; one group per displayed unit;
inspectable alternatives; assignment independent of cap; native units/resolution
and faithful, non-occluding time geometry; reproducible source-free exports;
unchanged decoder products; and passing focused tests.

After implementation approval and synthetic success, inspect one designated
session's existing saved run. Report group support/ties, source and within-side
outcome performance, overview/detail plots, and measured costs. User inspection
precedes any wider batch.

## 9. Execution record

Implementation was authorized on 2026-11-15. Tests were committed before the
corresponding implementation; unrelated worktree changes were preserved.

| Work | Tests/RED evidence | GREEN implementation |
| --- | --- | --- |
| U1-U2 signed weights and grouping | bd84ada; missing package during collection | 2b5bc1b; 25 passed |
| U3-U4 sources, persistence, native activity and lanes | 10e0998; missing results/plotting modules | 01db14b; 39 cumulative tests passed |
| Recorded context/LED export | 983f729/a854912; fixture corrected for frozen record, then missing interval color reproduced | 01db14b; public context labels/colors reused |
| U5-U6 CLI, batch, UI, planted signals and side-only control | 152b4ed; missing runners and Display selector | c7b2d3b; CLI/batch/UI and synthetic checks pass |
| Cached viewport slicing | 35edd45; missing viewport helper | c7b2d3b; normalization/global lanes reused during panning |
| PNG colorbar spacing | 8711908; exported colorbar overlapped plot bounds | c7b2d3b; explicit colorbar axes beyond the heatmap |
| Repository-local synthetic timeline outputs | 44d596f; existing production guard rejected fixtures | cacc289; extract unchanged production guard, mock only fixture-session paths |
| Invalid inputs, incomplete dry-run, tall captions | 0839b94/378bac6; targeted failures | e64fcfc; validations/status/caption fixes |
| Narrow preview labels and captions | 1d008fc/63b020b; bounding-box failures | cb451cd; separated labels and heatmap-local colorbar |
| Hundred-unit PNG text, absent-region contradictions, unknown rewards | 56b410a/bb1338e/c16d29f; targeted failures | e0303d9; physical row spacing, source checks, unknown-reward crosses |

Final verification: **264 passed in 37.36 seconds**, covering the feature,
population decoding, timeline plotting/chart/routing, and PSTH webapp tests.
The bundled Node binary enabled the frontend check. All generated synthetic
fixtures stayed inside `.pytest_cache/`; the user rejected external temporary
test files. Production output-directory restrictions remain enforced.

The actual CT026 browser view was inspected with cap/scale controls, labels,
captions, diagnostics, and export. Its temporary preview server was stopped.
Final PFC/HPC PNGs were visually inspected after the hundred-unit spacing fix.

The designated source is
`CT026_20260828_latent_inference/analysis_runs/population_task_decoding_20261114_192507_406224_a176e2c37f26`
under `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/`.
Both regions have 696 windows and native 500-ms bins (four bins per choice).
PFC selects 100 unique units from 141; HPC selects 100 from 286, with 10 in each
group. PFC has 9 assignment ties and HPC 2 across original units before capping.
Detail figures cover 0-99.083 seconds around the first context transition.

Final overview/detail run directories under that session's `analysis_runs/`:

- PFC overview: `decoder_unit_activity_20261115_171533_451520_526d72b3c8a2`
- PFC detail: `decoder_unit_activity_20261115_171534_627396_526d72b3c8a2`
- HPC overview: `decoder_unit_activity_20261115_171536_192265_e23c2a91c23f`
- HPC detail: `decoder_unit_activity_20261115_171537_633439_e23c2a91c23f`

Each contains `activity.png`, memberships, settings, logs, performance, and
module/script snapshots. The PFC overview also contains the combined
`inspection.md` and `inspection.json`, including saved model diagnostics.
Render times were 0.42-0.78 seconds from preloaded inputs; PNGs were 549-660 kB.
Peak process RSS reached 673 MB cumulatively across four renders, including
imports and prior renders. Global color limits make many values faint; no
temporal resolution was added. HPC left-side reward discrimination was near
chance (balanced accuracy 0.497), reinforcing the descriptive interpretation.

All 23 original source-run/preparation files retained their SHA-256 hashes.
Decoder fit code identity remains
`9e214d4ceaca5d085438b97eac278785a211df480505c7fd57b1d51cdb4b2379`.
No raw recordings were loaded, no experimental models were refitted, and no
wider experimental batch was run.

### Approved symmetric SD color control

The user approved this extension on 2026-11-15. Tests were committed first in
`005896f`: nine new cases failed on the missing helper, option, and UI control.
After implementation, all nine passed; the complete affected suite then passed
**273 tests in 39.52 seconds**. Coverage checks symmetric Plotly/PNG limits,
invalid values, unchanged activity/statistics/cache, stable limits when panning,
Auto restoration, UI export, CLI/batch forwarding, and saved export settings.

Z-score mode now exposes **SD color range -> Custom -> Symmetric limit (SD)**,
defaulting to 3 SD when Custom is selected. Auto remains the initial setting.
One shallow mapping copy changes color limits without copying activity arrays
or repeating normalization; values outside the range saturate only in color.
Single-session/plot/batch commands accept optional `--unit-sd-limit`. Exports
record the requested limit and effective range. No dependencies were added.
The user authorized finishing while leaving the concurrent, unrelated change
in `event_sequences/selection.py` untouched.
