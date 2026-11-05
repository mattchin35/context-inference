# Event-Sequence Heatmap Implementation Handoff

Status: implementation and designated CT026 single-session validation complete,
2026-11-05. Reviewed outputs await user inspection. No experimental batch run.

Authority: [spec](event_sequence_heatmap_spec_v1.md), [AGENTS.md](../AGENTS.md),
and [SoftwareDesign.md](SoftwareDesign.md). The spec owns scientific meaning;
this plan owns delivery, assignments, and live evidence. Its handoff format
follows the [phase-synchrony plan](session_timeline_phase_synchrony_implementation_plan.md#0-live-implementation-handoff).

Later user decisions refine the spec's population wording: good + MUA means
explicit manual `group=good` or `group=mua`, excluding blank/missing labels.
It does not inherit the existing timeline loader's missing-label-to-MUA
fallback. Existing timeline callers retain their current defaults.

## 0. Live handoff

Update the snapshot/ledger and append Section 7 evidence at RED, tests-only
commit, GREEN, review, interruption, and user-inspection boundaries.

### Current snapshot

| Field | Current value |
| --- | --- |
| Date / branch / inspected HEAD | 2026-11-05 / `refactor` / source accepted at `5ecdcbf`; final handoff commit follows. Recheck; do not reset to this observation. |
| Authority / next action | Implementation and designated-session validation are complete. User inspects reviewed figures/app before any experimental multi-session batch. |
| Feature files | Feature source/tests and this handoff are committed. The user's spec remains staged with unstaged refinements, preserved exactly. Transfer those actual user files if changing checkouts. |
| Other work to preserve | Staged PCA spec, single-unit plan, ridge spec; staged/unstaged phase-synchrony spec; unstaged `src/main.py`, `src/neural_analysis/sync_ephys.py`; unrelated untracked files. |
| Team / running work | Lead runtime identity is not exposed. Both workers idle. CT026 inspection app runs on loopback 127.0.0.1:8516, tool session 84354; no computations or experimental batches are running. |
| Verification | Latest scoped regression: 593 passed, no warnings. Synthetic sparse/dense and actual CT026 browser inspection accepted; 26 revised exports passed text-bounds and immutable-NPZ checks. |

### Package ledger

Proceed in order; only the Sol lead accepts completion. Record exact evidence
instead of advancing on a worker's completion message alone.

| Package | Owner | State | RED commit / command / result | GREEN commit / review |
| --- | --- | --- | --- | --- |
| WP0 spec/handoff | Lead | Revised plan approved; baseline passed | Documentation/baseline only | Implementation authorized |
| WP1 selection/context capture | Sol worker | Accepted | `d000950`; missing feature + 2 parser failures reproduced | `b556b82`; lead reviewed; 261 passed |
| WP2 population quality/PCA compatibility | Sol worker | Accepted | `121e747`; 55 failed on absent policy/lookup APIs | `790d3de`; lead reviewed; 387 passed |
| WP3 activity/coverage/scaling | Sol worker | Accepted | `ef5ff7b`; missing module; `a4fd9df` identity regressions | `3569bb0`; lead reviewed; 454 passed |
| WP4a preparation/persistence | Sol worker | Accepted | `c6c274f`; missing IO module reproduced | `176593c`; lead reviewed; 508 passed |
| WP4b callable batch wrapper | Luna | Accepted | `7ac4430`; 12 missing-callable failures; `608f50c` spawn guard | `c3893e6`; lead reviewed; 520 passed |
| WP5a scientific plots/export | Sol worker + lead | Accepted | `9ee9c61`; missing module; `caa8e6d` coverage annotation regression | `d4a010c` + subsequent caption fix; lead review, 554 combined / 36 focused passed |
| WP5b app wiring/usage docs | Luna + lead | Accepted | `ca8e213`, `e1ebea8`; missing app and live-state regressions | `1c8a515`, `4da688b`; 593 combined passed |
| WP6 synthetic/browser/regression audit | Sol lead | Accepted | `2df8a34`, `049ff5c`, `6bf8714`, `1cbb5f2`, `ca45558` display regressions | Six full-pipeline cases; 20 benchmark configurations; sparse/dense browser, physical PNG text-bounds; 593 combined passed |
| WP7 designated-session inspection | Sol lead + user | Computation and lead QA complete; awaiting user inspection | Static real-layout regressions `ca45558` | `5ecdcbf`; 12 configurations / 26 reviewed exports, no batch |

### Receiving and resuming

1. Read the spec, repository rules, this plan, and latest Section 7 entry.
   Confirm existing authorization once; record later approval without asking
   again at every package. Confirm the requested Sol lead and actual available
   model IDs; role labels do not change a chat's model.
2. Inspect branch/HEAD, staged/unstaged diffs, untracked files, and any recorded
   worker/command sessions. Preserve partial work; stop for unexpected changes.
   Do not create duplicate writers or rerun completed analysis to recover context.
3. Run the baseline below after initial approval. Subsequently reproduce checks
   only when evidence is missing, interrupted, or invalidated by relevant edits.
4. Resume the first unfinished gate using Section 6. Before stopping, record
   owner, model/effort/agent ID when known, writable files, exact commands and
   commits, running session/PID/log/output paths, and next action.

Initial baseline from the repository root; existing files only:

```bash
UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync --offline pytest -q \
  src/tests/neural_analysis/test_session_timeline.py \
  src/tests/neural_analysis/test_session_timeline_io.py \
  src/tests/neural_analysis/test_session_timeline_scripts.py \
  src/tests/neural_analysis/test_timeline_scientific_identity.py \
  src/tests/neural_analysis/test_timeline_plotting.py \
  src/tests/neural_analysis/test_session_timeline_webapp.py \
  src/tests/mouse_behavior_preprocessing/test_process_behavior_log.py
```

Record pre-existing failures. An environment failure or nonexistent test path
is not meaningful feature RED evidence.

## 1. Scope and architecture

Deliver the two agreed single-session app views, explicit preparation,
interactive hover, and light-mode PNG export. Implement 500/100-ms bins first;
expose 50/20-ms options after synthetic correctness/performance checks. The
optional one-panel central display is deferred; implement the six-panel grid.

The webapp is the user-facing preparation interface. Defer the public CLI,
its option matrix, and the JSON batch-request format. Keep callable main/batch
preparation functions for reproducibility and future approved batches; they
use the same computation as the app. Add no event-sequence settings to
`neural_session.json`: it continues to supply paths and probe-role assignments,
while app controls select the source, population, and bin width. Do not eagerly
prepare every unit, population mode, or bin width.

Keep one shallow `src/neural_analysis/event_sequences/` package. Use functions
and existing arrays/tables; add a data-only record only when it clarifies an
actual contract. No strategy classes, generalized pipeline, scheduler, or
runtime checkpoint/resume system is needed.

| Module | Responsibility / proposed functions |
| --- | --- |
| `selection.py` | Whole-table validation, `select_central_sequences`, `select_ideal_transitions`; retain original adjacency and identities, including empty conditions. |
| `activity.py` | `extract_unit_activity`, `select_pc_activity`, coverage selection, session scaling, and heatmap assembly; pure array operations and existing binning. |
| `io.py` | Metadata/source loading and result save/load/discovery; no orchestration or plotting. |
| `plotting.py` | Plotly app figures and Matplotlib PNGs consuming the same prepared values, layout, labels, and limits. |
| `run_session.py` | Callable `prepare_event_sequences` orchestrates selection, source preparation/reuse, and publication for the app and batch wrapper; no public CLI in this delivery. |
| `run_batch.py` | Callable per-session preparation wrapper with session-level processes and ordered results; no command-line or JSON request parser in this delivery. |

Keep `__init__.py` minimal. Add `webapp/event_sequence_views.py` for both views;
make bounded routing/availability edits in `webapp/app.py` and
`webapp/session_inputs.py`. No LFP samples or behavioral model fits are needed.
Every new function documents input/return types, shapes, axes, units, and
missingness; deterministic functions receive only their required data.

## 2. Scientific boundaries and existing-code reuse

### Behavior and identities

Read the full processed trial table. Validate required columns separately for
each tool: absent context/block fields disable Part II without blocking Part I.
Require a manual-reward flag column or its supported alias; do not inherit the
existing helper's missing-column default of all-zero flags. Normalize numeric
strings/missing sentinels without silently sorting or resetting source identity.

Preserve zero-based source position, original trial ID, and DataFrame label
separately. Invalid rows and gaps in trial IDs break adjacency. Require outcomes
only where the motif uses them; the next trial in Part I has unconstrained
correctness/reward. Compare actions directly, since existing switch/stay masks
exclude rewarded central trials. Part I can cross block boundaries.

Part II uses the last correct-rewarded old-block trial immediately before an
opposite active context begins, followed by exactly x incorrect old-side
repeats and an immediately rewarded/correct opposite-side choice, all in the
new block. Match on behavior before applying neural coverage. The parser
snapshots block/context at `trial_start`; current tests do not establish switch
boundary correctness. Add characterization tests for old/new trial attribution,
context-entry ordering relative to trial starts, and equal-timestamp ordering.
Downstream selection never rereads logs. Report a concrete parser defect before
expanding source scope. `RPi4_behavior_boxes` is read-only if inspected.

### Activity and PCA

Reuse `population/session_timeline.py` choice-window coverage/binning and fixed
regional PCA; use underlying spike loaders and channel-quality functions.
Units require manual Phy `group=good`; PCs offer `good_mua` (default) or `good`.
Both inherit good inside-brain channels and metadata channel restrictions.
For this feature both PC modes require explicit manual labels: normalize case
and surrounding whitespace, but exclude real missing values and blank/string
missing sentinels. Never substitute `KSLabel` or convert an unknown label to
MUA. Unit selection uses the same explicit-label rule with `good` only.

Add keyword-only population-quality and missing-label policy selection to the
public timeline configuration, population loader, and session runner. Propagate
both through `_prepare_session` to actual loading, not just the cache key.
Preserve existing defaults, their configuration identities, legacy registration,
and default encoding, prediction, variance, and timeline callers. New strict
policies have distinct identities, including strict good + MUA versus the legacy
fallback policy. Verify actual fitted input units against the declared selection.

Current timeline lookup matches the entire PFC/HPC configuration. Add bounded
regional compatibility checks so a selected region can reuse its exact basis
from a two-region archive. Verify the requested probe/acquisition, relevant
source identities and channel restrictions, explicit-label unit selection,
complete choice pool, bin geometry, full basis, and score/transform contracts.
An unrelated probe must not invalidate an otherwise compatible selected region.
A legacy/default archive is eligible only if its actual input unit IDs exactly
equal the strict expected selection and the remaining checks pass. Do not
relabel or modify its original identity; record the strict request and exact
source archive separately. Require legacy registration where existing rules
require it; do not infer compatibility from array shapes or session name alone.
Reject candidates missing the requested regional scores, even if their metadata
configuration matches. Keep ordinary timeline lookup/default reuse unchanged.

Fit PCA on all ordinary complete choice windows, independently of motifs.
Map scores through `regional.choice_positions` to the shared choices table,
then through `choices.row_position` to the full trial table. Verify original
IDs/timestamps/actions before joining; these are not interchangeable indices.

For the selected source, form `(W,B)` values across all eligible complete
windows: unsmoothed Hz for units, saved unwhitened scores for PCs. Bin only the
requested unit using the existing half-open convention; do not use the older
inclusive-final-edge unit utility unchanged. Require full coverage for Part I's
central window or every Part II segment; do not infer coverage from spikes.
Use existing trusted-coverage inspection; unavailable bounds disable that
source rather than trigger guessed coverage or a new alignment mode.

Compute session mean, population SD, and color limits from that source's full
`(W,B)` matrix, before motif selection. Do not reuse the timeline's regional
`color_limit`, which spans all PCs. Zero-SD z-scoring is unavailable; raw silence
is valid. Display matrices are `(matched_trial,B)` or logically
`(episode,x+2,B)`. Separator gaps are display-only and independent of bin width.

## 3. App preparation, saved display, and persistence

A request selects session metadata, source, probe/region, unit ID for a unit,
PC population mode for a PC, and bin width. PC rank/outcome/side/direction/x and
z-scoring are display settings. Use metadata PFC/HPC roles for PCs with an
explicit probe override; unit selection always identifies its metadata probe.

The app offers Prepare for a missing source configuration and explicit Recompute
for a fresh run. Source/population/bin-width changes select a compatible saved
product when present; an uncached configuration shows its readiness and awaits
Prepare. Show progress while preparation runs. Once loaded, outcome, side,
direction, motif length, PC rank, and raw/z-scored mode change live without
spike loading, binning, fitting, or result publication. PNG export remains an
explicit display action. Do not add command-line selection flags or metadata
fields to resolve the deferred CLI's source-selection questions.

The callable batch wrapper accepts explicit per-session preparation requests
using the same source contracts as `prepare_event_sequences`; define that small
Python input contract in WP4 tests. Unit IDs must be explicit per session, not
inferred as corresponding neurons. Reject duplicate session requests. Reuse
the existing worker-count policy and keep dry-run planning free of spike loads,
fits, writes, and worker startup; do not build a configuration framework.

Publish a timestamped session-local
`analysis_runs/event_sequence_heatmaps_<date>_<time>/` with named-array
`results.npz` and `meta`, copied main/batch scripts, `summary.md`, `run.log`, and
optional `figures/`. Save behavior-match tables, original identities, coverage
maps/counts, and selected-unit rates where applicable. PC products reference
the exact session-timeline archive with a session-relative path and saved
identity; do not duplicate its full score tensor.

Before WP4 GREEN, commit round-trip tests defining the named-array schema:
saved trial/row labels and original identities, all 24 central conditions and
8 transition conditions (including correctly shaped empty matches), behavioral
counts, source coverage maps/exclusions, and unit activity `(W,B)` where used.
Define the mappings from source rows to eligible activity windows explicitly.
For PC products, save the shared/regional choice mappings and verify a
session-relative PCA reference by configuration identity, byte size, and content
hash. Reuse existing portable-reference conventions where suitable. Saved
labels and availability reasons must support display without current CSVs.

Reuse requires matching analysis version, parameters, source identities, and
unit or PCA identity. Save Git revision as provenance; use the analysis version
to invalidate changed computations rather than every unrelated Git commit.
Metadata records generator, versions, parameters, quality policy, units/axes,
and random seed or explicit absence of randomness. Never overwrite results.
A valid final NPZ marks publication; interrupted/incomplete directories are
not reusable. An explicit rerun creates a fresh directory; no stage recovery
state machine or analysis-job ownership system is required.

Separate saved display from preparation readiness. Saved views read their
saved labels/mappings, unit arrays or exact referenced PCA, without current
raw spikes, quality tables, or behavior CSVs. Validate saved shapes/identities
and report a missing referenced PCA explicitly. Preparation checks current
sources; saved display does not reconstruct a current-source configuration.
Discover saved runs before checking current-source readiness or constructing
a current timeline configuration. Keep saved runs selectable when source files
are unavailable. Do not apply the app's all-probes-ready gate to these views.

Display selectors never implicitly fit/save. Offer Prepare for a missing source
configuration. Keep chronological rows and every requested empty panel; report
behavioral matches separately from coverage exclusions. Use the common six-panel
order and the spec's hover/caption fields. Part II displays one direction/x,
with local choice-time zero lines and fixed separators. Plotly and Matplotlib
receive the same values/limits; PNG export needs no new rendering dependency.

## 4. Tests-first delivery and file ownership

New tests: `src/tests/neural_analysis/event_sequences/`; app tests:
`src/tests/neural_analysis/test_event_sequence_webapp.py`. Extend existing
parser and timeline tests for their shared changes. Use seeded fixtures with
nontrivial trial IDs and unrelated DataFrame labels. Each package's tests
precede its source change, with the RED/commit/GREEN protocol in Section 6.

1. **WP1 / Sol: `selection.py`, `test_selection.py`, parser tests.** Cover all
   24 central conditions and 8 transition motifs; rewarded-central switches,
   unconstrained next outcomes, permitted block crossings, invalid/manual rows,
   missing flags/fields, strings/sentinels, ID gaps, boundaries, and n=0/1.
   Reject false triggers, extra/missing repeats, premature/opposite omissions,
   changed intervening blocks, and dark/unknown contexts. Verify missing context
   leaves Part I usable; parser events label the correct old/new rows, including
   context/trial-start ordering and timestamp ties.
2. **WP2 / Sol: timeline IO/runner and their existing tests.** Test conflicting
   manual/automatic quality labels, missing/blank label exclusion in both strict
   modes, unchanged legacy fallback/default identities, channel restrictions,
   actual strict good-only/good + MUA loading/fitting, constants, unavailable
   populations, and distinct strict identities. Test selected-region reuse
   from a two-region archive, irrelevant-probe changes, legacy exact-selection
   compatibility, and rejection of fallback-selected extra units or missing
   regional scores. Include a runner test proving its saved transform uses the
   units its configuration declares.
3. **WP3 / Sol: `activity.py`, `test_activity.py`.** Hand-calculated counts at
   real UTC verify 8/40/80/200 bins, Hz, inclusion at -2 s, exclusion at +2 s,
   and edge allocation. Test covered silence, source-specific coverage,
   central-only/all-segment exclusions, both levels of PC index mapping, and
   all-session z-scores/limits. An unrelated large-amplitude PC must not change
   the selected PC's scale. Test zero SD and constant raw activity.
4. **WP4a / Sol: `io.py`, preparation core in `run_session.py`, IO/preparation
   tests.** Define the saved schema before source edits. Round-trip empty and
   populated arrays/metadata, verify exact PCA bytes and relative references,
   immutable reuse, version/source invalidation, incomplete/missing products,
   independent sources, and unit preparation without PCA. After accepting
   these contracts, **WP4b / Luna** owns `run_batch.py` and `test_run_batch.py`:
   dry-run loads no spikes, fits/writes nothing and starts no processes; test
   Python request validation, explicit unit choices, duplicates, worker limits,
   and a small synthetic parallel batch. No CLI/JSON parser tests are needed.
5. **WP5a / Sol: `plotting.py`, `test_plotting.py`.** Verify panel positions,
   chronological identities, signed values/limits, segment geometry, zero
   lines/gaps, hover/captions, n=0/1, and PNG/interactive numeric agreement.
   **WP5b / Luna:** new view module, bounded app routing/availability edits,
   app tests and usage docs. Prove display selectors do not load spikes, bin,
   fit, or publish; cached configurations load without Prepare, uncached ones
   await it, and only Prepare/Recompute invokes computation. Test saved-first
   discovery/display after raw files are unavailable, cache reuse/invalidation,
   and that one missing probe or Part II field does not disable valid views.
   Inspect rendered PNGs and app layout.
6. **WP6 / Sol lead: `test_synthetic_integration.py` and acceptance evidence.**
   Exercise both tools, both PC modes, unit-only preparation, reload/export,
   and z-scoring end to end. Run relevant timeline IO/identity/scripts/plots/UI,
   encoding preparation/results, prediction preparation, variance, and parser
   regressions. Measure fine-width behavior. Broaden checks only for new edits,
   failures, or unresolved concerns.

## 5. Dependencies, performance, and experimental gate

Use existing NumPy, pandas, Pynapple, scikit-learn, Plotly, Streamlit,
Matplotlib, pytest, and standard library; no new dependency. Verify new library
APIs in installed source/official docs. Run Python through `uv run`.

Existing [timeline measurements](../src/neural_analysis/SESSION_TIMELINE.md#measured-stage-2-performance)
support local app preparation: 715 complete choices, 174 PFC and 316 HPC
units/PCs on the i9-12900 workstation took 3.723 s at 500 ms and 27.042 s at
20 ms for both regional PCAs including output writing. Saved loads took
0.090 s and about 1.6-1.7 s respectively. These are historical measurements of
the existing population policy, not timings for this feature, strict labels,
good-only mode, or 100-ms bins; do not present them as latency guarantees.

Reuse prepared data/PCA and send only the selected unit/PC to the browser.
Cache loaded results and, where repeated unit preparation would otherwise
reload the same probe, reuse loaded spike inputs keyed by relevant source
identities. Keep cached inputs read-only and avoid retaining all configurations
or duplicate full tensors. A source change invalidates its cached inputs.
Loading may still read probe-wide spikes or decompress a whole regional score
tensor; do not promise streaming or add a storage rewrite without measurements.

Measure first preparation separately from cached preparation/reload and live
display changes. Record loading/binning/PCA/publication/figure/serialization
times, payload size, process peak RSS, and browser responsiveness. Use a seeded
synthetic case comparable to the recorded 715-choice, 174/316-unit session;
state actual retained units/PCs and machine details. Verify 500/100 ms first,
then evaluate 50/20 ms before exposing them. Retain all matched rows and assess
single-source payloads rather than extrapolating the old multi-PC timeline's
large browser payload. Resource or responsiveness problems require review
before adding optimization or enabling fine bins. A public CLI can be proposed
later if measured preparation costs make advance preparation useful; it is
not needed to complete this delivery.

Callable batch preparation defaults to available CPUs capped by session count,
with a worker override; measured RAM or I/O limits can justify fewer processes.
Workers use a spawn context to avoid inheriting a threaded parent's runtime;
this changes neither session-level parallelism nor the worker-count policy.
Agent limits in Section 6 are separate.

Implementation approval covers tests-first code and synthetic validation.
WP7 uses the user-designated CT026 session of 2026-08-28, with metadata at
`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260828_latent_inference/neural_session.json`.
Run it only after the synthetic gates pass; user inspection of that output
precedes any experimental batch.

## 6. Sol/Luna execution rules

- **Sol orchestrator: `gpt-6.1-sol`, high.** Own assignments, scientific
  acceptance, integration, Git operations, user communication, live state, WP6
  and authorized WP7. Record actual runtime identity when exposed.
- **Sol scientific worker: `gpt-6.1-sol`, high.** WP1-WP3, WP4a, WP5a and
  bounded scientific fixes/review. **Luna: `gpt-6-luna`, medium.** WP4b/WP5b
  callable batch wrapper, widgets, routing, and docs after Sol accepts
  scientific interfaces.
- At most **one active worker plus the orchestrator**, including review work.
  Reuse contexts sequentially; no subworkers or extra reviewer team. If Luna's
  scope becomes scientific, return it to Sol. Check model availability; do not
  silently substitute another family. Current assignments are in Section 0.

Section 4 defines writable scope. Only one editor owns a file at a time;
WP4a owns `run_session.py` and releases its interface before WP4b starts the
batch wrapper. The lead alone updates this ledger and stages/commits explicit
paths, preserving unrelated staging. Do not amend/reset/revert user work.

For each assignment, supply package/phase, HEAD, exact writable files, work to
preserve, spec/array contracts, test oracles, and focused `uv run` command:

1. Worker writes tests only, demonstrates meaningful RED, returns the diff and
   exact command/results, then stops editing. Lead checks the oracle,
   reproduces RED, and commits only those tests.
2. Lead assigns GREEN. Worker implements and runs focused/regression checks,
   reports evidence, and becomes idle. Lead reviews the actual diff and
   scientific contracts, verifies checks, commits, and records acceptance.
3. A new defect gets a committed failing regression test before its fix. If
   the lead implements substantive science, reuse the Sol worker for focused
   review; do not describe self-review as independent review.

Order: WP1 -> WP2 -> WP3 -> WP4a -> WP4b -> WP5a -> WP5b -> WP6 -> WP7.
These are internal acceptance gates after implementation approval; no repeated
user approval is needed for each code package.

## 7. Review and execution record

Append date, package/phase, authorization, actual model/effort/agent ID and
active state, writable files, start/end HEAD, exact commands/counts/warnings,
RED/GREEN commits and reviewer, findings/resources, any process/session/log/
output paths, and next action. Preserve history; append corrections to mistakes.

### 2026-11-05 - WP0 / live-handoff and assignment revision

- User requested a Sol orchestrator, Sol workers for hard scientific tasks,
  Luna for simpler tasks, and minimal active workers. Incorporated those
  assignments and the maximum of one active worker alongside the lead.
- Added the live snapshot, sequential package ledger, receiving/interruption
  procedure, bounded baseline, file ownership, RED/GREEN review/commit protocol,
  and append-only evidence record, using the existing handoff plans as format
  references.
- Inspected branch `refactor`, HEAD `ed359a3`, and the relevant dirty state.
  This plan is untracked; the staged spec has unstaged refinements. Preserve
  both working files and all unrelated changes when handing off.
- Documentation-only revision. No worker, pytest run, source/test change,
  staging, commit, experimental analysis, or model-setting change occurred.
- Next action: user review of the revised plan. Upon implementation approval,
  the receiving Sol orchestrator follows Section 0, records the baseline,
  and assigns WP1 RED to one Sol scientific worker.

### 2026-11-05 - WP0 / correctness, completeness, and conciseness review

- Checked the spec, SoftwareDesign.md, and actual timeline configuration,
  loading, PCA mapping, publication, availability, and batch code.
- Clarified end-to-end quality-mode propagation, missing flag/context handling,
  selected-PC scaling, saved-only display, relative PCA references, publication,
  analysis-version reuse, and the batch request interface.
- Consolidated duplicate scope, ownership, handoff, and TDD instructions. Kept
  the live ledger, Sol/Luna assignments, tests-first commits, and session gate.
- Only this plan was edited; no workers, pytest, implementation, staging,
  commits, or experimental analysis. Next: user review/implementation approval.

### 2026-11-05 - WP0 / app preparation and explicit quality-label revision

- User required explicit manual good/mua labels, questioned the need for a
  CLI, and approved updating this plan after reviewing existing PCA timings.
  This authorizes documentation changes only; code approval remains pending.
- Made the webapp the preparation interface, with explicit Prepare/Recompute,
  cached source configurations, and live display selectors. Deferred the
  public CLI/JSON batch format and removed their flag-selection questions.
  Retained callable single-session/batch functions and script snapshots for
  repository reproducibility requirements. No metadata schema expansion.
- Required missing/blank manual-label exclusion in both new PC modes, while
  preserving existing timeline defaults and identities. Added explicit
  selected-region compatibility checks and exact-population legacy reuse.
- Added parser ordering/timestamp characterization, saved-first discovery,
  tests-first archive contracts, cache checks, and separate first-use/live
  performance measurements. Recorded prior 3.723-s/27.042-s PCA preparations
  as evidence for the app workflow, with their scientific/hardware limits.
- Inspected relevant code and prior timing records; branch/HEAD remain
  `refactor`/`ed359a3`. Only this plan was edited; no agents, code/test edits,
  test runs, staging, commits, or experimental computations. Existing dirty
  files and staged content were preserved.
- Documentation checks: `rg -n '[[:blank:]]+$'
  docs/event_sequence_heatmap_implementation_plan.md` found no trailing
  whitespace; `git diff --no-index --check /dev/null
  docs/event_sequence_heatmap_implementation_plan.md` reported no whitespace
  errors. Repository-wide `git diff --check` reported the pre-existing blank
  line at EOF in `src/neural_analysis/sync_ephys.py:243`; left unchanged.
- Next: user review/implementation approval, then the recorded baseline and
  WP1 RED. A new designated experimental session is still required for WP7.

### 2026-11-05 - Implementation authorization / baseline / WP1 RED assignment

- User instructed "start implementation" after the revised plan. Authorized
  scope includes all tests-first packages and synthetic validation; WP7 still
  needs a user-designated session. Branch/HEAD inspected: `refactor` /
  `ed359a3634929420c172be9b63024f0d063395a9`; existing dirty state matches the
  review and remains preserved.
- Exact baseline: `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync
  --offline pytest -q src/tests/neural_analysis/test_session_timeline.py
  src/tests/neural_analysis/test_session_timeline_io.py
  src/tests/neural_analysis/test_session_timeline_scripts.py
  src/tests/neural_analysis/test_timeline_scientific_identity.py
  src/tests/neural_analysis/test_timeline_plotting.py
  src/tests/neural_analysis/test_session_timeline_webapp.py
  src/tests/mouse_behavior_preprocessing/test_process_behavior_log.py`.
  Result: 94 passed in 7.15 s, no warnings. Tool session 69880 completed.
- Worker `/root/scientific_worker` requested as `gpt-6.1-sol`, high, fresh
  context; lead runtime model/effort is not exposed. Only worker writable
  files: new `src/tests/neural_analysis/event_sequences/test_selection.py`
  and `src/tests/mouse_behavior_preprocessing/test_event_sequence_context.py`.
  WP1 RED only: worker must return API/oracles/command evidence and stop before
  source edits; lead reviews, reproduces RED, and commits the tests first.

### 2026-11-05 - WP7 session designation

- User designated "ct026 8/28/2026". Located the exact session metadata at
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260828_latent_inference/neural_session.json`
  and read its source declarations: session `CT026_2026-08-28_134630`, PFC
  `ProbeA`, HPC `ProbeB`, schema 3, Open Ephys. No raw arrays were loaded and
  no experimental computation started. WP7 remains after synthetic acceptance;
  experimental batches remain gated on user inspection.

### 2026-11-05 - WP1 RED/commit/GREEN/review; WP2 RED assignment

- Worker `/root/scientific_worker`, requested Sol/high, wrote only new
  `src/tests/neural_analysis/event_sequences/test_selection.py` and
  `src/tests/mouse_behavior_preprocessing/test_event_sequence_context.py`.
  Lead reviewed all 24/8 condition oracles and identity/adjacency tests.
- RED reproduced with `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync
  --offline pytest -q src/tests/neural_analysis/event_sequences/test_selection.py
  src/tests/mouse_behavior_preprocessing/test_event_sequence_context.py`:
  missing feature import, one collection error in 0.22 s. Parser file alone:
  2 failed, 8 passed in 0.24 s; default timestamp sort reverses tied context
  entry/trial-start order. Tests-only commit `d000950` precedes source edits.
- GREEN writable files: new `event_sequences/__init__.py`, `selection.py`,
  and a bounded stable-Time-sort correction in
  `src/mouse_behavior_preprocessing/process_behavior_log.py`. No test edits.
  Worker focused selection/context/existing parser: 179 passed in 1.26 s;
  seven-path baseline: 94 passed in 6.84 s; local whitespace checks clean.
- Lead reviewed actual source and reproduced combined checks with the same
  `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync --offline pytest -q`
  prefix, the two WP1 test paths, and all seven baseline paths: 261 passed in
  7.27 s, no warnings. Tool session 43536 completed. GREEN commit `b556b82`.
  Commits used explicit `git commit --only` paths; unrelated staged docs remain.
- Same worker assigned WP2 RED only. Writable new
  `src/tests/neural_analysis/event_sequences/test_population_sources.py`;
  frozen quality/missing-label keywords, public metadata selection, and
  selected-region archive compatibility contracts are to be tested before
  shared timeline edits. No other active worker or experimental computation.

### 2026-11-05 - WP2 RED review/commit and GREEN assignment

- Same Sol/high worker wrote only `test_population_sources.py`; lead reviewed
  all 55 cases, including independently reconstructed old identity, actual
  selected-unit scaling, requested-region compatibility, corrupt-array
  rejection, and strict-population/legacy receipt rules.
- Lead RED: `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync --offline
  pytest -q --tb=short
  src/tests/neural_analysis/event_sequences/test_population_sources.py`:
  55 failed in 3.04 s, missing policy keywords/public selectors/lookup only;
  no fixture errors. Tool session 17982 completed. Tests-only commit `121e747`.
- GREEN assigned to `/root/scientific_worker`. Writable source files only:
  `population/session_timeline_io.py` and `population/run_session_timeline.py`.
  Preserve default identities/callers; shared code does not depend on the new
  viewer package. Finder checks trial-based pools and relevant inputs without
  reading raw spikes or optional event-display inputs. Tests remain unchanged.

### 2026-11-05 - WP2 GREEN/review/commit; WP3 RED assignment

- Worker implemented only the two authorized timeline files. Focused command
  from the WP2 RED entry: 55 passed in 3.34 s. All-empty manual-label columns
  needed local object dtype; degenerate spike trains needed explicit Pynapple
  support including trusted bounds and all selected spikes. Existing project
  binning/normalization remains in use; no dependency edits or test changes.
- Lead reviewed policy propagation, exact default identity preservation,
  metadata-first candidate filtering, receipt validation, source selection,
  full choice/PC axes and scaling. Worker inspected installed Pynapple
  constructor signatures before adding explicit support.
- Worker combined regression: 387 passed in 12.68 s. Lead independently ran
  `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync --offline pytest -q
  src/tests/neural_analysis/event_sequences
  src/tests/mouse_behavior_preprocessing/test_event_sequence_context.py
  src/tests/neural_analysis/test_session_timeline.py
  src/tests/neural_analysis/test_session_timeline_io.py
  src/tests/neural_analysis/test_session_timeline_scripts.py
  src/tests/neural_analysis/test_timeline_scientific_identity.py
  src/tests/neural_analysis/test_timeline_plotting.py
  src/tests/neural_analysis/test_session_timeline_webapp.py
  src/tests/mouse_behavior_preprocessing/test_process_behavior_log.py
  src/tests/neural_analysis/encoding/test_preparation.py
  src/tests/neural_analysis/encoding/test_results.py
  src/tests/neural_analysis/task_decoding/session_prediction/test_preparation.py
  src/tests/neural_analysis/test_window_variance.py
  src/tests/neural_analysis/test_window_variance_io.py
  src/tests/neural_analysis/test_window_variance_view.py`: 387 passed in 12.76 s,
  no warnings. Session 1006 completed. Local whitespace checks clean.
- GREEN commit `790d3de`; prior RED commit `121e747`. Only explicit source paths
  committed. User's staged documentation hashes were independently unchanged.
- Same worker assigned WP3 RED; writable new `event_sequences/test_activity.py`
  under tests only. No other worker or experimental computation is active.

### 2026-11-05 - WP3 RED review/commit and GREEN assignment

- Same Sol/high worker wrote only new
  `src/tests/neural_analysis/event_sequences/test_activity.py`. Lead accepted
  a data-only SourceActivity and functions for selected-unit extraction,
  PC selection, full-session scaling, and coverage-aware array assembly.
- Before committing, lead corrected one genuine count-oracle error: zero
  eligible windows with trusted coverage exclude requested matches for
  incomplete coverage; zero-SD scaling alone does not exclude covered rows.
  Replaced a nested convenience spy with Mock; no source existed at this point.
- Lead RED reproduced: `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync
  --offline pytest -q src/tests/neural_analysis/event_sequences/test_activity.py`:
  one missing-activity-module collection error in 0.25 s. Tests-only commit
  `ef5ff7b` precedes source changes.
- GREEN assigned to `/root/scientific_worker`; only new
  `src/neural_analysis/event_sequences/activity.py` is writable. No committed
  test changes, other workers, or experimental computation are authorized here.

### 2026-11-05 - WP3 initial GREEN review / trial-identity regression RED

- Initial activity implementation passed 56 focused tests in 1.38 s and 372
  event/baseline checks in 9.73 s (worker). Lead source review found a genuine
  omitted case: paired missing trial IDs must break motif adjacency, not
  invalidate otherwise ordinary complete PC reference windows. Regional reuse
  also rejected paired missing and equivalent numeric-string IDs.
- Worker wrote regressions only; activity source stayed unchanged. Lead
  reviewed and ran `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync
  --offline pytest -q --tb=no --no-summary
  src/tests/neural_analysis/event_sequences/test_activity.py
  src/tests/neural_analysis/event_sequences/test_population_sources.py
  -k 'matching_missing_trial_ids or trial_id_comparison_preserves or regional_reuse_preserves_missing'`:
  5 failed, 6 passed, 111 deselected in 1.65 s, on the WP3 working candidate.
  Malformed-ID rejection controls already passed. Session 6276 completed.
- Regression tests-only commit `a4fd9df` precedes fixes. Same worker assigned
  bounded GREEN changes in `event_sequences/activity.py`,
  `population/session_timeline_io.py`, and pure `population/session_timeline.py`
  for one shared semantic identity comparator. Recognized missing sentinels
  and numeric strings compare consistently; distinct malformed tokens must
  remain different. Ordinary eligibility, motif adjacency, default identities,
  and committed test expectations remain unchanged. No experimental work.

### 2026-11-05 - WP3 GREEN acceptance / WP4a RED assignment

- Worker identity regression command from the previous entry: 11 passed,
  111 deselected in 1.56 s. Full combined regression: 454 passed in 13.21 s.
- Lead reviewed SourceActivity array/shape validation, read-only copies,
  selected-unit half-open UTC binning, two-level PC identity joins, session
  scaling, and coverage-first assembly. Shared pure trial_identities_match
  preserves numeric/missing semantics without merging distinct malformed IDs.
- Lead independently ran the exact full command in the WP2 GREEN entry,
  whose event test directory now includes WP3 and identity regressions:
  454 passed in 13.32 s, no warnings. Session 14051 completed. Source whitespace
  checks clean. GREEN commit `3569bb0` contains only activity.py and the two
  bounded shared-timeline changes; tests were committed first.
- Same Sol/high worker assigned WP4a RED. Writable new tests only:
  `event_sequences/test_io.py`, `test_preparation.py`, and optional `conftest.py`
  under tests. Scientific interfaces and saved schema are to be reviewed before
  implementation; no public CLI, metadata expansion, or batch implementation.

### 2026-11-05 - WP4a contracts / RED review / commit / GREEN assignment

- Lead accepted a data-only EventSequenceRequest and plain prepared packet.
  IO saves all conditions/identities/counts/maps and unit rates; PC scores stay
  in an exact relative archive reference, with only runtime root/timeline
  binding on load. Raw probe arrays have an immutable two-entry source-keyed
  cache. Unit requests resolve quality to good; PCs default to strict good/mua.
- Lead reviewed the three new test files and independently checked the seeded
  fixture's central positions 1/3/4, transition [1,2,3], and covered positions
  1..7. Tests cover saved-only/relocated loading, exact PCA bytes before decode,
  source/version invalidation, no-overwrite publication, script copies,
  empty-spike good units, missing Part II fields, independent probes, dry-run
  no-work guarantees, and cache reuse/eviction/source invalidation.
- Lead RED: `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync --offline
  pytest -q src/tests/neural_analysis/event_sequences/test_io.py
  src/tests/neural_analysis/event_sequences/test_preparation.py`: two missing-IO
  collection errors in 0.14 s; session 99099 completed. Tests-only commit
  `c6c274f` precedes source changes.
- Same worker assigned GREEN in new `event_sequences/io.py`, `run_session.py`,
  and a docstring-only `run_batch.py` skeleton, solely so preparation tests can
  verify actual script snapshot bytes. Batch computation still waits for Luna's
  WP4b RED tests; no experimental outputs before the complete delivery gates.
  This avoids snapshot fallback branches or test-only production locators.

### 2026-11-05 - WP4a GREEN/review/commit / WP4b RED assignment

- Worker focused IO/preparation command: 54 passed in 4.59 s. Ordinary default
  combined collection exposed duplicate bare `test_preparation` module names.
  Lead reproduced with `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync
  --offline pytest -q src/tests/neural_analysis/event_sequences
  src/tests/neural_analysis/encoding/test_preparation.py
  src/tests/neural_analysis/encoding/test_results.py`: one import mismatch in
  0.15 s, session 19624 completed. Added only a test-package docstring marker
  `event_sequences/__init__.py` under tests, separately committed `d2bcb08`.
  No global pytest import-mode change or existing test edits.
- Lead reviewed all source: request validation, strict unit identity, empty
  assigned spike trains, unit-only preparation, exact PCA reuse/reference,
  packet maps/counts/coverage/config checks, immutable bounded cache,
  saved-only discovery/loading, logs/provenance and script publication.
  Canonical config identifies scientific source inputs; separate exact PCA
  reference binds immutable path/config/size/SHA256, preserved on normal reuse.
- Runtime provenance includes Python/library versions, explicit no randomness,
  small-input/reuse/coverage/unit-loading/binning or PCA lookup/prep/load/hash
  times. Final publication/elapsed is logged/reported after save; NPZ is never
  rewritten. Malformed pickle/NPZ discovery is handled as an ordinary bad file.
- Worker ordinary combined command from WP2 GREEN: 508 passed in 17.79 s.
  Lead independently ran that same exact command: 508 passed in 17.66 s,
  no warnings, session 32886 completed. Local source whitespace checks clean.
  GREEN commit `176593c`; source files io.py (411 lines), run_session.py (344),
  and docstring-only run_batch.py (6). No experimental output.
- Scientific worker released source ownership and is idle. Fresh worker
  `/root/interface_worker`, requested `gpt-6-luna`, medium, assigned WP4b RED
  only; writable new `event_sequences/test_run_batch.py` under tests. Frozen
  callable request API and existing worker policy are reused, with no CLI/JSON
  parser, metadata changes, or scientific work. Only one worker is active.

### 2026-11-05 - WP4b RED review/corrections/commit and GREEN assignment

- Luna wrote new test_run_batch.py only. Lead review found genuine test setup
  errors hidden by the missing callable: stale A_8 versus A:8 identity, two
  supposedly independent requests sharing one root, and a MagicMock worker
  count assertion on an unset attribute. Corrected these before committing;
  separately exercised normalized-path and alternate-filename root aliases.
  Replaced nested convenience classes/functions with standard mocks and one
  module-level completed-Future helper; clarified function contracts.
- Lead final RED: `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync
  --offline pytest -q --tb=no --no-summary
  src/tests/neural_analysis/event_sequences/test_run_batch.py`: 12 failed in
  0.10 s, missing run_batch callable. Session 7798 completed. New-file whitespace
  check clean after removing a blank EOF line. Tests-only commit `7ac4430`.
- Luna GREEN writable source: `event_sequences/run_batch.py` only. No scientific
  or metadata changes, no CLI, no committed test edits. Reuse shared worker
  policy and single-session API; preserve session-level processes and order.
  Sol worker remains idle; no experimental batch is authorized before WP7
  inspection, and only synthetic batch fixtures run in this package.

### 2026-11-05 - WP4b initial GREEN / spawn regression / acceptance

- Initial wrapper passed 12 focused and 520 combined checks, but added two
  Python 3.12 warnings about forking a multithreaded parent and possible child
  deadlocks. Lead reproduced 12 passed / 2 warnings in 0.26 s, session 43753.
  Verified installed ProcessPoolExecutor signature accepts mp_context and
  multiprocessing.get_context('spawn') returns a spawn context.
- Before the fix, Luna added the internal spawn-context assertion and a narrow
  warning-to-error filter for the actual synthetic parallel case. Lead reviewed
  and reproduced `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync
  --offline pytest -q --tb=short
  src/tests/neural_analysis/event_sequences/test_run_batch.py`: 1 failed,
  11 passed in 0.33 s (missing mp_context), session 88120. Warning reproduction
  itself depends on parent thread state; the deterministic context assertion
  is the regression. Tests-only commit `608f50c` precedes the source fix.
- Luna added only explicit spawned workers and an intent comment. Focused:
  12 passed in 1.72 s; combined: 520 passed in 19.30 s, no warnings. Lead
  reviewed and independently ran the full command recorded in WP2 GREEN,
  now including all event tests: 520 passed in 19.37 s, no warnings, session
  3984 completed. Scoped source whitespace check clean. GREEN commit `c3893e6`.
- Luna released source ownership and is idle. Sol scientific context resumed
  for WP5a RED only, writable new `event_sequences/test_plotting.py` under tests.
  Shared display payload and plotting/export contracts will be reviewed before
  source implementation. No experimental computation or batch has run.

### 2026-11-05 - WP5a plotting RED acceptance

- Sol added tests for one immutable selected-source display payload, all motif
  conditions, fixed session limits, signed values, hover identities/local time,
  fixed display separators, empty/constant/unavailable data, and PNG publication.
  Lead requested actual two-row chronology cases for both tools so renderer
  orientation is tested beyond singleton panels. Added original central rows
  1/5 and episodes [0,1,2]/[4,5,6], with manual separators and distinct IDs.
- Lead reviewed fixtures and tests and independently reproduced RED using
  `MPLCONFIGDIR=.pytest_cache/event_sequence_matplotlib UV_CACHE_DIR=.pytest_cache/uv
  uv run --frozen --no-sync --offline pytest -q
  src/tests/neural_analysis/event_sequences/test_plotting.py`: one missing
  plotting module collection error in 0.20 s. No plotting source exists yet.
  Matplotlib cache stays inside the ignored project test cache.
- User designated CT026 August 28, 2026 for the eventual single-session gate.
  Metadata resolves to CT026_20260828_latent_inference/neural_session.json;
  no experimental computation has run. Batch remains gated on user inspection.

### 2026-11-05 - WP5a GREEN / visual regression / WP5b handoff

- Sol delivered plotting.py only: one source scaling payload, common view
  arrays/hover/geometry, and thin Plotly/Matplotlib renderers plus explicit PNG.
  Lead reviewed the full file and inspected central PC z-score and transition
  raw-unit x4 PNGs. Source maps/labels, signed colors, fixed gaps/local ticks,
  captions and oldest-first axes are coherent. Worker: 34 focused passed;
  lead standard combined command: 554 passed in 20.78 s, no warnings,
  session 38438 completed. GREEN commit d4a010c.
- Lead found partial-coverage counts overlaid real activity. Requested and
  inspected a two-match/one-exclusion PNG confirming obscured bins/axis text.
  Added both-tool regressions: 2 failed / 34 deselected in 0.52 s. Committed
  RED caa8e6d before modifying source. Fix 0338277 puts counts in panel titles
  above populated activity and retains centered empty-panel explanations.
  Focused: 36 passed in 2.31 s. Inspected the corrected central PNG: clear bins,
  readable two-line title. Scoped whitespace check clean.
- QA artifacts are ignored under .pytest_cache/event_sequence_visual_review/:
  wp5a_central_pc_zscore.png, wp5a_transition_unit_x4_100ms.png,
  wp5a_partial_coverage_central.png and wp5a_partial_coverage_fixed.png.
  Sol released source ownership and is idle. Luna resumed WP5b RED only,
  writable new src/tests/neural_analysis/test_event_sequence_webapp.py.
- Lead added WP6 backend acceptance tests while Luna owns UI RED, with no
  production edits: all 24/8 motifs, seeded full-session UTC spikes, both widths
  500/100 ms, unit/good-PC/good-MUA-PC requests, metadata regional roles,
  immutable reuse and exact PCA bytes, removal of current raw/CSV/quality
  followed by saved reload, both tools, selected-source SD and explicit PNG.
  Existing APIs satisfied all six cases immediately (acceptance-only, no new
  implementation or claimed RED): 6 passed in 2.40 s, session 75098 completed.
  Tests-only commit recorded in Git. No experimental analysis has run.

### 2026-11-05 - WP6 seeded backend performance experiment

- Lead ran a reproducible 715-complete-choice synthetic session, seed 20261105,
  with every motif/manual separators, trusted IRIG bounds, uniform 5-Hz spikes,
  174 PFC / 316 HPC explicit good+MUA units and 87/158 good-only units. All
  retained PCs equal those counts. Machine: i9-12900, 24 logical CPUs, 62.5 GiB,
  Linux, Python 3.12.12. Raw fixture timestamps are uncompressed NPZ, assignments
  NPY. These timings describe this synthetic workload, not experimental data.
- Internal harness/report/log/script snapshot are ignored under
  .pytest_cache/event_sequence_benchmark_20261105/. Temporary session metadata:
  /tmp/event_sequence_benchmark_20261105_x8wcad4c/neural_session.json.
  Each real preparation also saved its normal session-local run/log/scripts.
  Initial harness fixture omitted mandatory x_um/y_um quality columns; fixed
  that fixture error before measuring and removed only its own failed temp
  directory. No product code was changed. Main command used uv frozen/no-sync/
  offline Python plus the usual MPLCONFIGDIR/UV_CACHE_DIR and completed via
  session 16820. All 20 source/width preparations succeeded.

| Cold preparation seconds | 500 ms | 100 ms | 50 ms | 20 ms |
| --- | ---: | ---: | ---: | ---: |
| One unit | 0.276 | 0.150 | 0.155 | 0.161 |
| PFC good+MUA | 1.731 | 3.030 | 5.885 | 14.549 |
| HPC good+MUA | 4.621 | 7.645 | 14.678 | 29.198 |
| PFC good only | 0.792 | 1.384 | 2.071 | 4.428 |
| HPC good only | 1.733 | 2.998 | 5.600 | 9.162 |

- Event saved-load latency ranged 0.028-1.316 s. Once loaded, selected source
  scaling plus both view builders took 0.002-0.014 s, figure creation 0.019-0.078 s
  and JSON serialization 0.002-0.030 s. Single-source figure payloads ranged
  44-978 kB, preserving all matched rows (48 central rows / six x4 episodes).
  Cumulative process high-water RSS, including raw fixture construction, peaked
  at about 2.4 GiB; this is not an isolated per-configuration memory estimate.
- Results support app-first preparation with visible progress; finer full PCA
  preparation can take tens of seconds while live selection stays small.
  50/20-ms preparation controls remain gated on actual browser checks. Luna
  still owns WP5b RED, with lead-requested oracle corrections; no experimental
  analysis or batch has run.

### 2026-11-05 - WP5b final RED acceptance / dense-display browser fixture

- Lead corrected app-test oracles before committing: bad-channel A:9 changed
  to valid B:7; saved switching now actually selects unit then PC; preparation
  uses wrapped real reports, not impossible prepared/None paths; export tests
  click twice and inspect figures/; exact PCA cache tests mutate same-size bytes.
  Added offline/source-work guards, transitions, zero-spike good-unit handling,
  one-ready-probe availability, shared source/rank across tools, K=1 clamping,
  and visible preparation failures. Lead verified installed Streamlit slider
  source rejects min==max, removed the erroneous K=1 slider requirement, and
  tests instead require actual PC1 in the rendered caption.
- Lead final command with ordinary MPLCONFIGDIR/UV_CACHE_DIR and uv frozen/
  no-sync/offline pytest -q --tb=no --show-capture=no
  src/tests/neural_analysis/test_event_sequence_webapp.py: 16 failed in 2.80 s,
  absent options/dispatch/availability/view module, no fixture failures. RED
  commit ca8e213. Luna GREEN owns new webapp/event_sequence_views.py, bounded
  app.py/session_inputs.py and README.md; committed tests/science immutable.
- To avoid judging fine bins only on balanced motifs, lead also measured a
  pure dense display with 715 rewarded left choices and 713 matching rows.
  JSON sizes at 500/100/50/20 ms: 0.51/2.74/5.51/13.31 MB. View+figure creation
  stayed below 0.15 s and serialization below 0.25 s; browser behavior remains
  unmeasured. Results: dense_display_stress.json in the ignored benchmark cache.
  Added a normal saved-unit synthetic stress session using the same immutable
  seeded raw spikes and its own dense behavior table, all four widths, for
  actual app browser inspection. Its path/report is dense_session.json.
  These are synthetic tests only; no experimental computation has run.
- Dense fixture creation initially tried source symlinks; existing metadata
  containment validation correctly rejected them. The harness now copies its
  own seeded raw files into a separate contained temporary session, preserving
  the production path rule. Normal unit preparation succeeded for all four
  widths (session 85582 completed); metadata for browser QA is
  /tmp/event_sequence_dense_20261105_r5oh4918/neural_session.json.

### 2026-11-05 - WP5b helper repair / source-review regressions

- Luna correctly identified an AppTest helper error after the new module
  existed: from_function serializes function source without module globals.
  Lead added local load_webapp_session imports to both wrappers; this is a
  genuine test repair, not relaxed product behavior. No production workaround.
- Lead reviewed the new source draft and added tests before three fixes:
  available scaling uses an empty-string reason (not None), so the draft
  disabled valid z-scoring; an open new-source editor reread current quality
  CSV on live motif changes; absent legacy probe roles defaulted to the first
  probe rather than awaiting an explicit override. Strengthened valid z-score
  assertions to verify enabled checkbox and actual rendered SD caption.
- Focused command with standard environment/uv flags, pytest -q --tb=short
  --show-capture=no test_event_sequence_webapp.py -k
  'display_selectors_do_not or open_source_editor or legacy_metadata':
  3 failed / 15 deselected in 2.39 s, session 78318 completed. Lead committed
  these tests/helper repair before authorizing source fixes. Other GREEN work
  stays with Luna; scientific modules remain untouched.

### 2026-11-05 - WP5b GREEN acceptance / browser layout regression

- Lead reproduced three AppTest ElementList concatenation TypeErrors in 2.29 s
  (session 52253); converted collections to ordinary lists without relaxing any
  assertions, commit 24e67a1. This unmasked a real missing Part II UI notice:
  one failed / two passed in 2.30 s. Luna added the saved reason outside the plot.
- Luna source work: 18 focused passed, standard subset 580 passed in 26.65 s.
  Lead reviewed full new module and bounded routing/availability/README diffs,
  independently ran the complete planned subset plus new app/integration tests:
  580 passed in 26.54 s, no warnings, session 65420 completed. Commit 1c8a515.
  Current metadata selectors are cached by small-source stats; saved packets by
  event/PCA stats (two entries). No new CLI or metadata fields. Luna released
  source ownership and is idle. An unnecessary worker full-repository collection
  hit 49 unrelated environment errors (autograd/statsmodels); left untouched.
- Loopback synthetic browser QA uses 127.0.0.1:8516, original tool session 97769
  stopped after source updates; log is under the ignored benchmark cache.
  Default legacy raster failed on missing LED columns in the synthetic fixture;
  selecting the dedicated event route works and avoids those unrelated controls.
- Actual 1280x720 browser inspection exposed overlapping central panel titles
  and a horizontal long PC colorbar title squeezing plot width. Lead committed
  RED 2df8a34: one failed / 36 deselected in 0.35 s, then wrapped panel/segment
  labels, kept the exact units beside a vertical colorbar, and separated global
  transition title from segment labels. Plotting+app tests: 55 passed in 6.50 s,
  session 95848 completed. Browser reinspection and source commit remain next.

### 2026-11-05 - WP6 browser/fine-bin acceptance and designated-session start

- Browser reinspection confirmed the title/unit fix, then exposed right-panel
  y labels touching left data and a transition caption touching its time label.
  Lead reproduced/committed the spacing regression 049ff5c (one failed in
  0.44 s) and caption regression 6bf8714 (one failed in 0.47 s), before each
  source fix. Panel headings/segment labels now wrap, PC units remain exact
  alongside the vertical colorbar, grid spacing reserves trial labels, and
  interactive captions wrap separately from static PNG captions.
- Actual balanced 20-ms browser: initial saved-view controls ready in 1.240 s;
  z-score response 355 ms, PC1->PC2 324 ms. Source config/PCA hash unchanged;
  switching tools retained PC2 and SD units. Six-segment x4 layout inspected.
  The browser's SVG text wait helper sometimes timed out despite visible text;
  those failed measurement attempts are not product-latency evidence.
- Dense 20-ms browser: all 713 central rows retained, initial saved-view controls
  ready in 1.465 s, actual chart subsequently inspected. Z-score response through
  native click to attached SVG unit label: 873 ms. JSON stress was 13.31 MB.
  Balanced/dense services were loopback only and have now been stopped.
  Idle dense tool session 60152 had exited; replacement 21875 was tested and
  stopped. Balanced final service 18099 stopped. Temporary error tab was not
  used to bypass browser policy; a fresh ordinary localhost tab completed QA.
- Lead also added empty-panel text wrapping and fine-bin explicit-Prepare
  tests: 3 failed / 55 deselected in 1.21 s before source, commit 1cbb5f2.
  After correctness/resource/browser acceptance, exposed optional 50/20-ms
  preparation and documented measured synthetic timings. Static central and
  x4 transition PNGs were regenerated and visually inspected under ignored
  event_sequence_visual_review/wp6_*_responsive.png. No label/caption overlap.
- Full scoped regression after final changes: 584 passed in 28.73 s, no
  warnings, session 79874 completed. Scoped whitespace clean. Source commit
  4da688b. WP1-WP6 accepted; both workers idle.
- WP7 single-session validation launched via tool session 32561, source script
  .pytest_cache/event_sequence_validation_20261105_143800/main.py and run.log.
  It prepares only CT026 August 28: first explicit manual-good unit per metadata
  PFC/HPC role, both PC modes, 500/100 ms; no fine-bin experimental preparation
  or cross-session batch. Normal run directories and a timestamped validation
  report/PNG directory are additive within that session. User inspection still
  gates any experimental batch. No user input is needed to finish this gate.

### 2026-11-05 - WP7 computation complete / real-PNG review findings

- Designated-session script completed successfully, session 32561: 12 explicit
  source configurations (ProbeA:0 / ProbeB:0, both regional PC modes, 500/100 ms),
  695 eligible ordinary complete windows. Strict good+MUA PFC/HPC input/PC
  counts: 141/286; strict good only: 125/245. No coverage exclusions in the
  inspected conditions; left correct-rewarded central grid has 194 rows, most
  populated transition is left->right x1 with 25 episodes.
- Additive report/scripts/log/measurements/26 initial PNGs:
  /home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260828_latent_inference/analysis_runs/event_sequence_validation_20261105_143831_589625.
  100-ms new good+MUA preparation: PFC 1.961 s, HPC 4.633 s; saved loads 0.172/
  0.289 s. Existing compatible 500-ms good+MUA bases were reused, so their
  0.30-s event preparation must not be described as a cold PCA fit. Both unit
  sources are explicit manual-good; no experimental batch was run.
- Lead inspected actual central PC SD, transition PC SD, unit raw and HPC
  good-only raw PNGs. Values/identities/counts/gaps/limits are coherent. Two
  presentation defects became clear with real identities/counts: long static
  captions clip in 12-inch transition exports; twelve tick labels can overlap
  when a central panel has 17 rows. Before source changes, lead added physical
  Matplotlib canvas/text-bounds regressions (width and time-label separation,
  and neighboring trial-label bounding boxes). Tests are currently running
  in session 93230. Fix presentation only, preserve all computed NPZs and
  prior PNGs; publish revised exports additively for final user inspection.

### 2026-11-05 - Final static regressions / reviewed exports / handoff

- Physical regression RED: five failed / four passed / 38 deselected in 1.14 s,
  session 93230. Long 12-inch captions exceeded the canvas by up to 45 pixels;
  neighboring tick labels overlapped for 17-row panels. Tests commit ca45558.
  Caption wrapping now follows canvas width and uses reserved bottom space;
  uniformly spaced integer tick strides retain all data and avoid adjacent
  crowded labels. Initial narrower wrapping exposed insufficient caption-height
  clearance (four focused failures); reserved transition bottom margin fixed it.
- Final focused plotting/app: 67 passed in 7.98 s, session 23474. Final full
  planned subset: 593 passed in 28.27 s, no warnings, session 28723. Scoped
  whitespace check clean. Presentation-only source commit 5ecdcbf.
- Revised 26 PNGs from the same 12 saved products were published in a NEW
  review directory, preserving initial PNGs and every NPZ/PCA archive. Each
  export checked caption canvas/axis-label bounds, neighboring trial label
  bounds, and unchanged source NPZ SHA256. Re-export completed via session
  95294; script/log in the ignored validation cache and copied into the report.
  Final report and figures:
  /home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260828_latent_inference/analysis_runs/event_sequence_validation_20261105_143831_589625/render_review_20261105_144750_670251/summary.md.
  Lead inspected revised actual PFC central SD and transition SD PNGs: complete
  captions, readable trial IDs, consistent source limits and local segment axes.
- Actual CT026 browser verified metadata/source identity, strict 141-PC PFC
  good+MUA basis, PC1 at 100 ms, live source SD display and 194 central rows.
  App remains on 127.0.0.1:8516, tool session 84354; its log is
  .pytest_cache/event_sequence_validation_20261105_143800/webapp.log.
  Temporary synthetic services are stopped. No experimental batch was run.
- Final Git audit reproduced all five original staged document blob hashes
  unchanged. The original dirty main.py/sync_ephys.py/spec refinements remain.
  Range whitespace audit found only one own test EOF blank, removed without
  changing assertions; the unrelated pre-existing sync_ephys.py EOF is untouched.
  Feature source/tests and this handoff are ready for review. User inspection
  is the only remaining gate before any multi-session experimental analysis.
