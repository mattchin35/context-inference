# Session timeline: optional signed treadmill speed

Status: full implementation plan for review, 2026-10-30. No speed implementation
or experimental metadata changes have been made. Follow `AGENTS.md` and
`docs/SoftwareDesign.md`; implementation starts after this plan is approved.

## 1. Outcome and confirmed decisions

Add one optional, synchronized speed row to the existing Session timeline and
PCA viewer, immediately below the selected event rows and above regional PCs.

- Preserve the signed `runSpeed` values exactly as logged, in mm/s, as confirmed
  by the user. No absolute value, sign inversion, smoothing, or PC normalization.
- Prepare speed explicitly once, with a separate button or CLI action, then
  load the saved result. This lifecycle was selected by the user during planning.
- Use the same first-trial time zero and shared time navigation as other rows.
- Speed is continuous session behavior, including intervals between choices;
  do not restrict it to the neural choice windows.
- One speed cache is shared by all probe assignments and neural bin widths.
- Existing PCA scores, Stage 2 counts, and saved archives remain unchanged.
- Keep speed optional. Missing or unprepared speed must not disable other rows.
- Keep the viewer's current light theme and existing chart component.

Initial experimental validation is CT026 2026-08-04 only. Other webapp views,
speed decoding targets, analog voltage processing, and experimental batches
are outside the initial rollout.

## 2. Evidence and existing code to reuse

Designated source:

```text
/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260804_latent_inference/
  teensy/CoolTerm Capture (Untitled_0) 2026-08-04 12-08-14-010.txt
```

Read-only inspection found a 10,149,386-byte log with 171,837 records:
54,733 each of distance, DAC value, and speed, plus 7,638 sync transitions.
Speed includes 49,265 positive, 1,539 negative, and 3,929 zero values, spanning
-198.72 to 526.86 mm/s. Preserve these values; their range is not an outlier rule.

The existing reader parsed the complete file in 0.765 s in one measurement.
The existing NeuroKairos decoder read 63 valid frames, beginning at
2026-08-04 16:09:00 UTC and ending at 17:11:00 UTC, with no repeated sync states
or conversion warnings. These frame times are not the file's coverage bounds.
No end-to-end speed alignment has yet been validated.

| Existing module | Reuse and limits |
| --- | --- |
| `mouse_behavior_preprocessing/process_treadmill.py` | Reference for timer semantics, signed `runSpeed` extraction, and post-step plotting. Keep its legacy public functions and CT014 workflow intact. Its parser rejects current `...Us` labels/time-only prefixes, and its decoder assumes standard-decisecond IRIG. Do not wire it directly into the viewer. |
| `irig_tools/irig_serial_io.py` | Import-safe, tested parsing; converts microseconds to seconds once. Extend it minimally for IRIG-only processing that does not require a PC calendar date. |
| `irig_tools/irig_core.py` | Use `decode_irig_pulses(..., irig_format="neurokairos")` and its decoded frame anchors. Do not duplicate bit classification or frame decoding. |
| `neural_analysis/synchronization/manual.py` | Reuse `make_timebase_mapper` for piecewise linear clock-to-UTC conversion, including its documented endpoint extrapolation. Apply recorded-data bounds at the caller. |
| `neural_analysis/session_metadata.py` | Existing contained session-relative path resolution and optional-input conventions. `treadmill_file` is currently an always-None property. |
| `population/timeline_plotting.py`, `webapp/timeline_view.py`, `webapp/timeline_chart.js` | Existing row geometry, caching, matched axes, bounded neural loading, and persistent browser navigation. |

The installed Plotly `Scattergl` line API was inspected: `shape="hv"` supports
the required horizontal-then-vertical step trace. No new plotting dependency
or frontend component is needed. Existing parser/decoder/metadata tests passed
33 tests during inspection; this does not replace the new tests below.

## 3. Metadata change

Add optional `behavior.treadmill` to schema versions 2 and 3. It is a
session-relative text-file path or null; omission, null, and the existing empty
optional-path convention mean no source. No schema-version bump or new
scientific configuration object is needed for this additive optional field.

CT026's behavior block becomes:

```json
{
  "trials": "processed/CT026_2026-08-04_120843_trials.csv",
  "events": "processed/CT026_2026-08-04_120843_events.csv",
  "treadmill": "teensy/CoolTerm Capture (Untitled_0) 2026-08-04 12-08-14-010.txt"
}
```

Implementation changes:

- Add a defaulted `treadmill: str | None = None` field to `BehaviorSources`.
- Replace the always-None `ResolvedBehaviorSources.treadmill_file` property
  with a defaulted `Path | None` field of the same public name. Existing
  two-argument construction remains valid.
- Extend parsing, allowed keys, resolution, CLI skeleton, and metadata docs.
- Use the existing containment checks; reject absolute paths and traversal.
- An absent file makes speed unavailable. It does not make ordinary webapp,
  power, synchrony, or spike-phase inputs invalid.
- Do not add timezone, date, calibration, probe, or bin settings for speed.
  Absolute dates come from the encoded IRIG frames, not the filename or PC.

Do not edit CT026's JSON until the cache preservation work in Section 7 has
been implemented and applied to its existing results. Back up the JSON with
a timestamped filename before the actual update; preserve unrelated fields.

## 4. Scientific data and time contracts

### 4.1 Parsing

Add an explicit `pc_timestamp_format="ignore"` mode to the existing serial
parser/reader. In this mode, strip an optional tab-separated PC prefix and
parse the serial message through the existing payload parser. Leave PC fields
empty and require no reference date. Existing datetime, time-only, and None
modes keep their existing behavior and requirements.

This avoids inventing a reference date simply to parse a time-only prefix.
PC receipt times remain irrelevant to the speed alignment; the CT026 log
contains batched PC timestamps for distinct microcontroller event times.

The parser's `time_value` column is already float64 seconds. Do not pass it
through the legacy microsecond conversion again. Recognize the existing
`syncIntervalUs`/`syncInterval` and
`timeSincePinChangeUs`/`timeSincePinChange` label pairs with their existing
microsecond-on-disk convention. Unsupported timing semantics produce a clear
input error rather than a guessed conversion.

Blank lines may be skipped. Malformed nonblank records must identify the
source line and stop preparation rather than silently remove timing records.
Other well-formed payload labels may remain in the parsed table; only sync
and speed records are required for the scientific output.

### 4.2 Reconstruct the microcontroller clock

Let `d[j]` be each sync record's interval from the preceding transition, in
seconds. Define `edge_time[j] = sum(d[0:j+1])`. Clock zero is therefore the
transition preceding the first recorded sync transition, not session zero.

For each speed record after edge j:

```text
speed_clock_time = edge_time[j] + timeSincePinChange
```

For leading speed records before the first recorded edge, their
`timeSincePinChange` is relative to the same preceding-transition clock zero.
Retain them when consistent with the first recorded interval. This uses the
first sync interval to place leading data; do not drop these records merely
because a complete IRIG frame occurs later in the file.

Do not cumulatively sum every message: distance, DAC, and speed rows repeat
the same offset, and the offsets are not intervals between messages.
Document the resulting `(N,)` source-clock axis in seconds.

Validate finite, nonnegative intervals/offsets and timer ordering. Different
speeds at equal timestamps retain source order; the last recorded value is
the state after that instant. Do not average or silently deduplicate samples.
Reject clock resets or contradictory edge/offset timing with a useful error;
do not add automatic epoch stitching or reset repair.

### 4.3 Decode and map to UTC

1. Pair alternating HIGH/LOW sync records into pulses using their reconstructed
   source-clock times. Require a closing LOW for each decoded pulse; a trailing
   unpaired HIGH is not a fabricated completed pulse.
2. Call the current IRIG core with measured pulse widths, source onset times,
   and original parsed-row indices. Preserve its invalid-pulse diagnostics.
3. Use the frame table's `observed_start_time_s` and `decoded_utc_seconds`
   as source/target anchors. Do not treat the decoder's one-second-per-bit
   assignment as an independently measured speed clock.
4. Require at least two finite, strictly increasing frame anchors in each
   clock domain. Build the existing linear timebase mapper and map the speed
   records' source-clock times to UTC Unix seconds.
5. Endpoint extrapolation is allowed only to actual reconstructed records
   before/after the first/last frame anchor, following existing alignment
   conventions. Do not synthesize samples outside the recorded log.

An incomplete initial/final frame is ordinary. Bad pulse diagnostics do not
automatically destroy an otherwise usable log; frame-anchor absence or
inconsistent timing does. Do not manufacture zeros for lost data or add
arbitrary speed masks, drift cutoffs, or smoothing. Record anchor intervals,
clock scale, and decoded/source time ranges for inspection. Duplicate sync
states are an explicit preparation error in this first implementation; the
CT026 file has none. Existing parser transition-repair behavior is unchanged.

The initial speed workflow explicitly uses NeuroKairos. Keep legacy format
functions usable, but do not silently try multiple formats and choose whichever
produces dates. Supporting another speed-log protocol would be a later explicit
decision, not another mandatory metadata field now.

### 4.4 Output record

Use a small data-only `TreadmillSpeed` record with native NumPy arrays:

| Field | Type, shape, units |
| --- | --- |
| `utc_s` | float64 `(S,)`, nondecreasing UTC Unix seconds for speed records |
| `speed_mm_s` | float64 `(S,)`, signed logged mm/s; same sample order as `utc_s` |
| `source_rows` | int64 `(S,)`, original positions in the parsed message table |
| `frame_clock_s` | float64 `(A,)`, strictly increasing source-clock seconds |
| `frame_utc_s` | float64 `(A,)`, strictly increasing UTC Unix seconds |
| `meta` | dictionary of diagnostics, units, version, and source identity |

S is the aligned speed sample count; A is the valid frame-anchor count. Keep
all finite speed values, including zero and negative values. Nonfinite speed
payloads are input errors rather than values to plot or replace with zero.
No new trial/bin/PC axes are introduced. These simple arrays are directly
usable by Plotly and can be used to construct a Pynapple time series later.

## 5. Modules and interfaces

Keep two focused new modules rather than a new package or class hierarchy.
Document input types, axes, units, returns, and errors on every function.

| Module | Responsibility and planned interface |
| --- | --- |
| `mouse_behavior_preprocessing/treadmill_speed.py` (new) | `TreadmillSpeed`, `reconstruct_treadmill_clock(parsed)`, and `align_treadmill_speed(parsed)`. Computation receives the parsed table, uses existing IRIG/time-mapping functions, and returns arrays/diagnostics. No filesystem writes, metadata discovery, or plotting. |
| `neural_analysis/session_speed.py` (new) | `speed_configuration(session)`, `find_saved_speed(session_root, configuration_id)`, `save_speed_result(path, result)`, `load_speed_result(path)`, and `run_session_speed(metadata_path, *, rerun=False, dry_run=False)`. Own metadata/source loading, persistence, logging, and the explicit preparation boundary. |
| `irig_tools/irig_serial_io.py` | Minimal ignore-prefix parsing mode; retain existing modes and public behavior. |
| `neural_analysis/session_metadata.py`, `session_metadata_cli.py` | Optional path support, contained resolution, and skeleton. |
| `population/session_timeline_io.py`, `run_session_timeline.py` | Scoped PCA configuration identity and safe reuse of already prepared archives, as specified below. |
| `population/run_session_timeline.py`, `run_session_timeline_batch.py` | Add a mutually exclusive `--speed` action that calls the independent speed runner. Reuse existing main/batch boundary and session-worker controls. |
| `population/timeline_plotting.py` | Add a trailing optional `speed: TreadmillSpeed | None = None` argument and one plotting-only helper. Existing call sites remain valid. |
| `webapp/timeline_view.py` | Explicit preparation controls, optional speed loading/cache, Show speed toggle, and status/caption text. |

No changes to Stage 1/2 numerical functions, archive schemas, or the speed's
raw text file. No importing the exploratory `irig_tools/treadmill_serial.py`,
which executes a hardcoded workflow on import. Do not refactor all legacy
treadmill code or add an analog fallback as part of this work.

The runner returns a small data-only `SpeedRunReport`: status
prepared/reused/dry-run, result path (None for dry-run), configuration ID,
and elapsed seconds. Expected source/input failures raise a descriptive
OSError or ValueError for the existing CLI/UI boundary to report. Keep this
record local to the speed workflow rather than importing the neural runner
solely for its report class.

## 6. Preparation, persistence, and CLI

### Independent saved product

Speed cache identity contains the resolved text path, byte size, mtime_ns,
session identity, speed analysis version, parser/timer/IRIG conventions,
signed-mm/s convention, and time-mapping method. It excludes raw metadata
file timestamps, neural files, probes, PC ranges, trial tables, and bin size.
Changing a trial-table origin therefore does not require decoding UTC speed
again; the viewer subtracts its current common origin at display time.

Use `treadmill-speed-v1` as the initial analysis version. Preserve prior
results when source identity, version, or parameters change. `--rerun` creates
a new timestamped run even for the same identity. `--dry-run` reads metadata
and file stat only, without parsing, decoding, or creating directories.
Changes to parsing, timing, mapping, or unit semantics must bump this analysis
version; unrelated display-code changes do not force speed recomputation.

```text
<session>/analysis_runs/treadmill_speed_<date>_<time>_<configuration_id>/
  results.npz
  config.json
  run.log
  summary.md
  run_session.py
  run_batch.py
```

Store the named arrays above and a dictionary named `meta`. Metadata includes
generator, analysis version, code revision, parameters, units/axes, source file
identity, timing diagnostics, sample/anchor counts, runtime/library versions,
and `random_seed: null` because no random operations occur. Include seconds
for parsing, clock reconstruction/decoding/mapping, writing, and total action.

Use the existing project's simple temporary-file/final-rename convention so
partial NPZ files are not discovered as completed results. Never overwrite
an existing result. Script copies are the actual existing main/batch entry
points; the summary records their `--speed` invocation, session, source,
goal, valid time extent, and any excluded incomplete pulse. No statistical
tests are performed by this preparation. Refuse outputs inside the repository.

### Commands

```bash
uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/neural_session.json --speed --dry-run

uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/neural_session.json --speed

uv run -m src.neural_analysis.population.run_session_timeline_batch \
  --session-list /path/to/sessions.txt --speed --dry-run
```

`--speed` and `--window-summary` are mutually exclusive. The speed action does
not load spikes or fit PCA, even when no neural result exists. Probe and neural
bin arguments do not configure speed; document this in CLI help. Keep main/
batch's existing error conventions, worker cap, and across-session processes.
An explicitly requested speed preparation with a missing/invalid source gives
an actionable error. In the viewer that error affects speed only.

## 7. Preserve prepared PCA and Stage 2 results

The current Stage 1 identity includes raw `neural_session.json` size/mtime.
Simply adding the optional path would hide existing prepared results. Do not
solve this by fitting PCA again, changing archive metadata, copying large
arrays, or weakening all cache checks.

### New scientific identity

Make the PCA identity depend on the actual resolved inputs it uses:

- analysis and configuration-identity versions, session ID, role/probe keys;
- selected probes' acquisition families and effective channel/unit selection;
- bin width, window, PCA/scaling/quality conventions;
- trial/event tables and neural source paths/sizes/mtime_ns.

Exclude the treadmill path and whole metadata-file timestamp. Keep metadata
path/stat as provenance outside the scientific digest. Other changes in the
actual PCA inputs must still invalidate reuse. Stage 2 remains tied to its
exact Stage 1 archive path/stat, without any new compatibility rule there.

### Existing archives: establish identity before changing metadata

Add one small, tested registration helper, used during the rollout before
editing CT026's metadata:

1. While the original JSON is unchanged, verify a completed legacy archive
   exactly matches the existing v1 configuration, including that JSON's
   recorded file identity and all other source/parameter checks.
2. Compute the new scientific configuration from those same resolved inputs.
3. Write `scientific_identity.json` beside that archive, recording the new
   configuration ID/configuration, legacy ID, and original result path/stat.
   This is supplemental provenance; the NPZ, its path, and its mtime do not change.
   Create the record exclusively; reuse an identical record, and reject a
   conflicting existing record rather than overwrite it.
4. Extend `find_saved_timeline` to try normal exact matches first, then these
   registered records. Require the identity, analysis/schema versions, and
   original result stat still to match; the result must be in that same run.

Keep the existing lookup signature. The viewer, Stage 1 runner, and Stage 2
metadata entry point already share it and should return the original archive.
Do not perform registration or any other writes on display reruns.

Register both CT026 500-ms and 20-ms runs before adding `behavior.treadmill`.
An unregistered legacy archive whose metadata has already changed is not
automatically trusted. Do not match by session name, array shape, or units
alone. Document the one-time registration order for other existing sessions.
This small supplementary record is justified specifically by preserving
immutable legacy results with insufficient stored semantic metadata.

## 8. Viewer behavior

- **Prepare speed**: enabled when a declared source exists; prepare/reuse via
  the same runner as the CLI. It never prepares neural data.
- **Recompute speed**: explicitly create another run for an existing prepared
  configuration. No automatic decoding on page load, pan, or display changes.
- **Show speed**: default on when a matching saved speed result is available;
  otherwise disabled with a short missing/unprepared/stale explanation.
- Cache NPZ loading by path, bytes, and mtime_ns with bounded entries, following
  the existing `st.cache_data` convention. Source changes select another cache.
- Keep preparation status and captions in stable Streamlit slots/containers so
  toggling speed does not remount the chart, following the Stage 2 finding.

Plot one `Scattergl` trace with `mode="lines"`, `line.shape="hv"`, and no
connection across explicit unavailable values. The line represents each
logged value held until the next logged update, including recorded zero
states. It ends at the last speed timestamp; do not append a synthetic zero
or indefinitely extend the last value. Equal-time updates preserve file order.

Use elapsed seconds `utc_s - result.time_origin_s`, float64 `(S,)`. Label the
row **Signed speed (mm/s)**, show a zero reference, and include both signs in
the y range. Use one fixed full-session speed range with a small display
margin, independent of PC ranks, PC z-scoring, viewport, and bin width. Hover
shows signed mm/s, elapsed seconds, UTC seconds, and parsed source-row position.

Send the complete speed series initially. CT026's two primary float64 arrays
occupy about 0.876 MB, unlike its large neural heatmaps. No averaging,
decimation, or viewport-dependent speed resampling is justified yet. Existing
bounded-neural loading continues to constrain native panning when enabled;
do not change that behavior or crop speed to the neural choice windows.

Keep the existing session-time navigator extent from trials/events/neural
windows; a recorder that started early or stopped late should not unexpectedly
expand that established session view. Speed is clipped visually by the shared
axes, retaining its original saved UTC samples.

Add the speed row to the vertical layout revision while keeping the shared
time navigation revision stable. Reuse the existing browser range synchronization
when rows are inserted/removed. No new JavaScript pan callbacks are planned.
Speed works in behavior-only sessions as well as sessions with one/two probes.

The caption explains that signed speeds are logged values held until the next
update, aligned using Teensy timing and decoded IRIG. Speed between choices is
measured behavior; blank neural gaps do not imply absent speed.

## 9. Tests written before implementation

For each work package: write tests, observe RED, commit those tests alone,
then implement GREEN and refactor without weakening expectations. Do not
include the user's already staged specification or other dirty files.

### Metadata and compatibility

Extend `test_session_metadata.py` and `test_session_metadata_cli.py`:

- v2/v3 omission, null, empty, and relative text path; old constructors work;
- absolute/traversing paths rejected; directories rejected as source files;
- missing optional treadmill does not disable existing action availability;
- skeleton/docs use the optional field without requiring schema migration.

Extend timeline I/O/script/webapp tests:

- supplementary identity can be registered only against an exact old match;
- both old bin-width archives retain their paths, bytes, mtimes, and Stage 2
  sidecar selection after only adding/changing/removing the treadmill field;
- neural quality, acquisition, probe/channel/unit selection, trials/events,
  source files, or bin changes still invalidate the scientific identity;
- stale/mismatched registration and unregistered edited metadata are rejected;
- no metadata-registration writes, spike loading, fitting, or Stage 2 reduction
  occur during display changes or speed preparation.

### Serial parsing and time reconstruction

Extend `test_irig_serial_io.py`; add `test_treadmill_speed.py`:

- ignore-prefix mode accepts full/time-only/no PC prefixes without reference
  date; existing parser modes and microsecond conversion remain unchanged;
- blank lines, malformed records, both timing label spellings, signed/zero
  values, and irrelevant payload labels have documented behavior;
- hand-computed clocks with sync intervals plus repeated message offsets,
  leading records, equal timestamps, incomplete boundary pulses, and resets;
- nonfinite/negative timing, contradictory offsets, and duplicate sync states
  produce explicit errors, without guessed time or silent repair;
- seeded synthetic NeuroKairos messages with known absolute UTC, nonzero
  microcontroller drift, misleading PC receipt times, and irregular speed
  updates recover the expected UTC and unchanged mm/s values;
- initial/final incomplete frames work; insufficient/nonmonotonic anchors
  fail; invalid pulse diagnostics are retained; extrapolation stays within
  actual record extent, without generated speed samples.

### Persistence, runner, and plot/UI

Add `test_session_speed.py` and extend plotting/script/webapp/frontend tests:

- named-array round trip preserves values, signs, zeros, order, axes, units,
  anchors, version, provenance, and diagnostics;
- matching reuse, explicit rerun, source/version invalidation, no overwrite,
  no partial-result discovery, and stat-only dry-run;
- main/batch `--speed`, mutual exclusion, and synthetic across-session run;
- row placement below events, true elapsed times, correct step geometry,
  negative y values, hover units, boundary behavior, and no choice-window cropping;
- default visibility, missing/unprepared/stale states, explicit-only preparation,
  behavior-only operation, and cached reload after reopening the viewer;
- PC ranks/z-scoring, events, speed visibility, and bounded-neural loading do
  not alter speed arrays or any neural numerical products;
- stable component key and mount position, plus real browser pan/range-slider
  persistence when speed is hidden/restored; Jump still explicitly navigates.

Use small exact fixtures and the existing synthetic-session fixtures. Run
related metadata, serial/core IRIG, manual-mapper, timeline/Stage 2, and webapp
regressions at the integrated boundary. Include the Node chart lifecycle test.
Use `uv run` for every Python command and check changed paths with Ruff.

## 10. Implementation sequence and gates

| Work package | Deliverable and gate |
| --- | --- |
| 1. Metadata | RED tests committed; optional field, resolution/skeleton/docs GREEN. No experimental JSON edit yet. |
| 2. PCA identity | RED reuse/invalidation tests committed; scoped identity and legacy registration/lookup GREEN. Prove both Stage 2 products are still selected without numerical work. |
| 3. Timing | RED parser/clock/seeded alignment tests committed; small scientific module GREEN using existing IRIG and mapping APIs. |
| 4. Saved speed | RED I/O/main/batch tests committed; independent versioned NPZ and explicit CLI preparation GREEN on synthetic sessions. |
| 5. Viewer | RED row/UI/cache/mount tests committed; speed controls and plotting GREEN, with no new frontend component. |
| 6. CT026 | Register original neural identities, back up/update only the treadmill metadata field, dry-run, prepare one speed result, and inspect the actual viewer at 500 and 20 ms. |
| 7. Documentation | Update usage, metadata README, and implementation record with measured timings, validation results, and limitations. User inspects the new experimental output before any experimental batch. |

No analysis output, experimental source, or existing metadata file is written
during the current planning task. Necessary output-directory access will be
handled through the existing execution permissions during implementation.

## 11. CT026 validation and performance

Before the metadata edit, record the original PCA and Stage 2 paths, sizes,
mtimes, and file hashes for both configurations. After registration, metadata
editing, speed preparation, and viewer changes, verify all four NPZ archives
remain byte-identical and are still selected by ordinary loading.

Validate the speed result against the parsed source: 54,733 speed records
unless an explicitly reported input problem is discovered, unchanged ordered
mm/s values, expected positive/negative/zero counts, monotonic UTC times,
63 frame anchors if the same complete frames are accepted, and decoded date
2026-08-04. Inspect mapping of frame-start source times to encoded UTC,
source-clock scale/interval diagnostics, and actual speed/session overlap.
Do not use PC receipt timing as the correctness oracle.

Browser-check rest, forward/backward values, sharp changes, speed between
choices, missing-data boundaries, hover, shared pan, range slider, Jump,
row visibility, PC rank/z-score changes, and both bin widths. Save a labeled
PNG/report in the speed run directory for review. Preserve the current
context/reward/PC/count rows.

Measure parsing, clock/IRIG/mapping, NPZ write/load, figure construction/JSON
with and without speed, and browser responsiveness separately. Record input
dimensions, machine, available memory, major array bytes, and process peak RSS
with its process-lifetime limitation. The prior 0.765 s measured parsing only;
do not quote it as total preparation time or a latency guarantee.

No new dependencies, cluster work, GPU, JIT, resampling framework, or major
optimization is planned. If actual speed rendering is unexpectedly expensive,
report the measured cause before changing the scientific/display sampling.

## 12. Acceptance

The feature is complete when the optional path is documented, signed-mm/s
speed is prepared once and reused at every neural bin width, the row is
correctly aligned and navigable with the other rows, existing PCA/Stage 2
archives remain usable and unchanged, tests pass, and CT026 output plus
measured performance are available for the user's scientific inspection.
