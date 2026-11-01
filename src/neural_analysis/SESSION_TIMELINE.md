# Session timeline and PCA viewer

Stage 1 displays completed choices, behavioral events, and regional population
PC scores on one elapsed-session timeline. Time zero is the first trial start.
Each neural window covers `[-2, +2)` seconds around an actual choice; the real
gaps between choices stay blank. Choices outside a probe's recording coverage
remain visible without neural measurements from that probe.

Preparation runs once for a session, bin size, and population selection. It
saves the fitted basis and all scores. Viewing, panning, trial jumps, event
toggles, PC-row changes, and display-only PC z-scoring reuse those products.
Changing bin size selects a different configuration and can change the PC axes.

## Inputs

Use the existing `neural_session.json` in the session's data directory. See
the [metadata quickstart](README.md#create-and-validate-session-metadata) for
the supported source layouts. No additional configuration file is required.

- The processed trial table supplies completed choices, original `cur_trial`
  IDs, `start_time`, `choice_time`, and action `0=right`, `1=left`. Current and
  existing legacy experimenter-reward flags follow the normal project rules.
  Incorrect and unrewarded choices are included. Experimenter rewards,
  nonchoices, and the unsaved active end-of-file trial are excluded.
  The optional `reward` column uses `1=rewarded`, `0=unrewarded`; missing or
  other values stay unknown. Correctness is not used as a reward substitute.
- Optional event CSVs use the existing `Time`, `Event`, and `Note` columns:
  `trial_start`, `left_entry`, `right_entry`, LED on/off transitions, and
  `enter_left_patch`/`enter_right_patch`/`enter_dark_period` context entries.
- Each probe supplies the existing aligned spike archive with
  `spike_utc_unix` and `irig_utc_unix`, sorter `spike_clusters.npy`, manually
  curated `cluster_info.tsv`, and `channel_quality.csv` or `.json`.
- One population includes all quality-selected channels on a probe. Channels
  must be good and inside brain, and optional metadata `unit_channels`
  further restricts them. Clusters use manual Phy `group` values `good` and
  `mua`, following existing normalization. Named LFP sites do not split this
  population. LFP data are not required by the viewer.

Version-3 `analysis.probe_roles` supplies PFC/HPC defaults. With version-2
metadata, select the probe keys explicitly in the app or command. Assigning
one probe is supported; the other panel is marked unavailable. An unassigned
configuration can display behavior alone.

## Prepare before opening the viewer

From the repository root, install the locked dependencies if needed:

```bash
uv sync --locked
```

For version-3 metadata with existing PFC/HPC roles:

```bash
uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/session/neural_session.json
```

The default bin width is **500 ms**. Supported widths are 500, 100, 50, and
20 ms, producing 8, 40, 80, and 200 bins per window respectively.

For explicit probe assignments, using the exact keys in the metadata:

```bash
uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/session/neural_session.json \
  --pfc-probe ProbeA --hpc-probe ProbeB --bin-width-ms 100
```

Omit either probe argument to prepare just the specified population. When no
probe arguments are supplied, existing version-3 metadata roles are used;
version-2 metadata has no automatic anatomical assignment.

Add `--dry-run` to inspect the configuration without loading spikes or writing
outputs. A matching completed result is reused by default. Add `--rerun` only
to recompute explicitly; this creates a new timestamped directory and leaves
the old result intact.

## Open and navigate

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata /path/to/session/neural_session.json
```

1. Select **Session timeline and PCA** in **Plot view**.
2. Check the PFC/HPC probe selections and bin width. The viewer automatically
   loads matching prepared products. If none exist, behavior is displayed and
   **Prepare** computes/saves the selected configuration. **Recompute** creates
   a new run when explicitly needed.
3. Set **Visible duration (s)** and **Window start (s)**, or select an original
   trial ID and click **Jump**. The initial duration is 60 seconds.
4. Drag horizontally inside the figure to pan. The bottom session navigator
   changes the visible time range. The mouse wheel scrolls the page.
5. Select a PC rank range independently for each population. The initial
   selection is the first ten available PCs. Change optional event checkboxes
   to show trial starts, licks, and LED intervals/transitions.
6. Optionally enable **Z-score each PC** to inspect each PC's variability on
   its own scale. This defaults off and works with existing prepared results;
   no recomputation or archive update is needed.

**Show active context** is on by default when logged context entries are
available. Faint shading behind the choices indicates blue for left active,
red for right active, and gray for dark periods. The shading uses the actual
context-entry timestamps and continues through gaps between choices until
the next entry or session end. It affects only the choices row.

Choice markers show actual reward outcomes: **large filled triangles**
(12 px) are rewarded and **small open triangles** (7 px) are unrewarded.
Left/right colors and lanes stay the same, including for correct choices
without reward. Gray crosses indicate unknown reward status. Hover includes
the outcome alongside the trial ID, time, and active context.

New prepared results retain reward flags in the choice table. When an older
result omits them, the metadata-driven viewer uses a cached behavior-table
read and checks choice identities before attaching outcomes in memory. This
does not rewrite the archive or require PCA recomputation. A standalone old
archive without its behavior source displays unknown reward status.

Time before the first logged context entry remains unshaded and is labeled
**Unknown** in choice hover. If context entries are absent, the checkbox is
disabled and the viewer reports context unavailable. Context is not inferred
from chosen side, correctness, LEDs, or model estimates. Turning off shading
keeps context in choice hover. Existing prepared results already contain the
event stream, so enabling this overlay does not require PCA recomputation.

The chart axes stay synchronized during native panning. Changing PC ranks,
z-scoring, colors, or event visibility preserves the current time window,
including a window reached by mouse panning or the bottom session navigator.
PC rows still update to the selected ranks in their original order.

**Window start**, **Visible duration**, and **Jump** explicitly request a new
time window. Clicking Jump again restores its target even if you have panned
away since the previous click. Changing the session, probe assignment, or bin
width starts a new configuration/view. Reloading the page also starts a new
view; browser navigation is not written into the saved analysis.

The sidebar time fields retain the last explicit navigation request; mouse
panning and the bottom navigator operate locally in the browser. They do not
rerun Python or update those fields. Numerical results and variance coverage
remain unchanged.

For large figures, enable **Load only visible neural interval**. It loads the
window specified by the time controls plus neighboring time on either side
and retains a compact full-session navigator. Panning is confined to the loaded
interval. Use Window start or Jump to load another part of the session;
dragging the native range slider does not request new data from Python.

The chart defaults to a light background. Hover a choice for trial ID, side,
time, reward outcome, and true active context from the logged transitions.
Hover a heatmap cell for region/probe, PC identity and session variance percentage, original
trial ID, elapsed/UTC/choice-relative bin intervals, and the signed score (or
per-PC z-score when enabled). LED intervals use verified on/off pairs;
unpaired transitions remain markers without invented endpoints.

## Read the PCA display

Each population is fitted separately using all of its complete choice windows.
Units are z-scored once across pooled window/bin observations. Constant units
are omitted. Saved PCA uses full SVD, no whitening, and no post-PCA score
scaling. The PC rows stay in session explained-variance order.

By default, colors are signed and centered at zero, using fixed regional limits
calculated from all scores in the configuration. Gray center-color cells are
measured near-zero scores; blank gaps have no neural measurements. The optional manual
color-range control changes display limits only and labels saturation. It
does not change scores or normalize rows.

With **Z-score each PC**, the display uses `(score - mean) / SD` separately
for each PC in each population. The reference includes all that population's
prepared choice-window bins across the session, with population SD (`ddof=0`).
It excludes gaps. Constant PCs display zero. The same bin keeps the same
z-score when you pan, change the selected PC rows, or load only the visible
neural interval.

Heatmap colorbars and hover values then use **PC SD** units. Automatic symmetric
limits span all prepared z-scores for the selected PC rows, so they stay fixed
while navigating; selecting different rows can change those limits. Manual
limits use SD units, initially +/-3, and are separate from raw-score limits.
The option changes only heatmap values, color limits, and their labels. It
preserves timing, gaps, context, PC order, loadings, and the original variance
percentages. Low-variance PCs become as visually prominent as leading PCs,
including any noise they contain.

Z-scoring runs live on only the selected PC rows, without new dependencies or
saved arrays. A synthetic benchmark of 1,000 windows with 200 bins each measured
about 38 ms for 10 PCs and 670 ms for 100 PCs (median of three calls). These
measure normalization only, excluding result loading, figure construction, and
browser rendering; they are not real-session latency guarantees.

The footer shows the smallest leading-PC counts covering **80%, 90%, and
95%** of total variance in each population's complete fitting dataset. It also
reports unit/window counts and omitted constants. Displaying fewer PCs does
not change the denominator or footer. Regions can have different eligible
windows and unit counts, so their summaries describe their own fitting data.

These PCs describe pooled choice-window activity, not a decision-specific or
noise-corrected latent dimensionality. Signs and identities apply to the
saved configuration and need not match another session or bin width. Stage 2
adds the separate per-choice summary described below.

## Stage 2: fixed-axis variance within each choice

With Stage 1 prepared, click **Prepare window summary**. This loads the saved
scores and direct unit-variance totals; it does not load spikes, bin again,
or fit another PCA. A matching saved summary is reused on later opens.
**Recompute window summary** explicitly creates another timestamped sidecar.

**Show fixed-axis window counts** adds a purple point row below each regional
heatmap. Each point is the smallest number of original session axes needed
to account for 95% of that choice window's variance. Points occur at actual
choice times, with no connecting lines. Hover shows the count, original trial
ID, time, status, total variance, bin count, unit count, and basis completeness.
Showing/hiding these rows preserves the current panned time interval.

Select an original trial in **Jump to trial** to show its contribution tables
below the figure. Clicking **Jump** also centers the timeline; selecting alone
updates the readout. Each region reports the count, total variance, bins,
units, and completeness. The table lists the selected original **Session PC**
identities in decreasing within-window contribution order, their fractions,
and cumulative fractions through the 95% threshold. The heatmaps keep their
original session PC order. Their rank ranges, z-scoring, and color settings
do not change these summaries.

For every PC, the numerator is sample variance across the window's time bins
(`ddof=1`). The denominator is the saved sum of sample variances of all units
after their original session-wide standardization. Scores are centered within
each window by the variance calculation. Sustained mean displacement alone
therefore contributes no variance. This is fixed-axis participation, not a
new local PCA or a noise-corrected dimensionality estimate. Correlated session
axes can all contribute, so the count can exceed `bins - 1`.

Exactly constant score columns have zero variance, without an added variance
cutoff. Sorting breaks ties by original session PC rank. Complete bases must
conserve the direct unit total within `1e-10` on the PC/unit variance ratio;
the 95% comparison allows `1e-12` fractional roundoff. Both tolerances are saved.
Zero-total, nonfinite, insufficient-bin, and inconsistent windows have no
count point; their reason is available in the trial readout. An incomplete
basis is never renormalized: a threshold-reaching count is labeled
**restricted**, and failure to reach 95% is reported explicitly. All saved axes
are considered, including those outside the displayed range.

The CLI uses the same explicit action on an already prepared configuration:

```bash
uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/session/neural_session.json \
  --bin-width-ms 500 --window-summary
```

The existing probe arguments, `--dry-run`, `--rerun`, and `--benchmark` also
apply. A missing Stage 1 configuration raises an error instead of implicitly
fitting it. For an approved session list, the batch command accepts
`--window-summary` and parallelizes across sessions using the existing workers.
Inspect a single experimental session before running a batch.

Python callers can use `run_window_variance.run_window_variance(stage1_path)`,
load its result with `window_variance_io.load_window_variance`, and pass
`window_variance=summary.regions` to `plot_session_timeline`.

## Measured Stage 2 performance

CT026 2026-08-04 was checked at 500 and 20 ms on an Intel i9-12900 workstation
(24 logical CPUs, 62.5 GiB RAM, about 26 GiB available during measurement).
Both regions had 715 complete windows: PFC 174 units/PCs, HPC 316 units/PCs.
All windows passed variance conservation and the minimal-count threshold
check; maximum conservation error was below `1.9e-15`. No experimental batch
was run. User inspection of these new descriptive outputs remains appropriate.

| Bin width / payload scope | Stage 1 load | Summary load | Stage 2 calculation | Figure | JSON | Payload |
| --- | --- | --- | --- | --- | --- | --- |
| 500 ms, full session | 0.090 s | 0.034 s | 0.026 s | 1.057 s | 0.698 s | 16.1 MB |
| 20 ms, elapsed 0-125.757 s | 1.722 s | 0.034 s | 0.318 s | 1.046 s | 0.660 s | 14.6 MB |
| 20 ms, full session | 1.637 s | 0.032 s | 0.313 s | 4.125 s | 16.506 s | 406.2 MB |

Figures in this table use ten PCs per region and existing default event tracks.
The complete Stage 2 calculation always uses all 715 windows and all axes,
even with bounded heatmaps. Warm z-scored figure construction/serialization
measured 0.935/0.688 s (500 ms full) and 1.001/0.594 s (20 ms bounded).
These are individual measurements, with filesystem cache state unspecified,
and exclude browser rendering. They are not latency guarantees.

The accepted 500-ms Stage 1 preparation took 3.723 s including output writing;
the separately prepared 20-ms configuration took 27.042 s. The recorded regional
source loading/binning/PCA/projection sums were 1.791/0.299/0.850/0.152 s at
500 ms and 2.219/2.894/6.571/2.044 s at 20 ms. Output compression and other
orchestration account for the remaining preparation time. This work is shared
Stage 1 preparation, separate from Stage 2.

The complete score arrays occupy 22.4 MB at 500 ms and 560.6 MB at 20 ms.
Saved summaries are about 8.5 MB at either width. Benchmark process peak RSS
was 0.565 GB (500 ms full), 1.858 GB (20 ms bounded), and 4.562 GB (20 ms full);
these are process-lifetime peaks including imports, score loads, and plotting,
not incremental metric memory. The validation script deliberately loads a
second copy while measuring the load operation. Stage 1 preparation peaks
were separately recorded as 0.695 and 2.568 GB.

These measurements support keeping Stage 2 local. The full 20-ms browser
payload exceeds Streamlit's usual message limit, so **Load only visible neural
interval defaults on at 20 ms**. Use Window start or Jump to load other intervals;
native panning stays inside the loaded interval. Window-count points remain
session-wide. Wider loaded intervals or many displayed PCs can still produce
large payloads. A cluster would not reduce that browser payload.

Browser inspection covered both count rows, contribution tables, point hover,
shared panning, row visibility, rank/z-score changes, and 20-ms bounded loading.
Detailed validation JSON, script copies, and logs remain beside each summary
in the CT026 data directory.

## Optional signed treadmill speed

Add a session-relative Teensy log path to the existing `behavior` object:

```json
"treadmill": "teensy/CoolTerm Capture (Untitled_0) 2026-08-04 12-08-14-010.txt"
```

The field is optional in metadata versions 2 and 3. Omitted, null, or empty
means no speed source. All other metadata fields retain their existing meaning.
The current speed workflow decodes NeuroKairos IRIG from the log; it does not
require a PC timestamp date, timezone, calibration field, probe, or neural bin.

**For existing prepared PCA results, register their scientific identity before
editing the original metadata JSON.** The original archive fingerprint included
that whole JSON file's stat. While the original sources still match, use:

```python
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.neural_analysis.population.session_timeline_io import register_timeline_scientific_identity

session = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
register_timeline_scientific_identity(session, original_results_npz)
```

Repeat for each existing bin-width result to retain. This writes only
`scientific_identity.json` beside the existing archive. It validates the old
configuration first and leaves NPZ bytes, paths, and mtimes unchanged, preserving
their Stage 2 summaries. New configurations use actual resolved scientific
inputs instead of the whole JSON timestamp, so optional speed-path edits do
not invalidate them. A changed neural/behavior source or population still does.
An already edited metadata file cannot silently authorize an old unregistered
archive. Back up metadata before editing it.

Speed is prepared independently of neural data:

```bash
uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/session/neural_session.json --speed --dry-run

uv run -m src.neural_analysis.population.run_session_timeline \
  --session-metadata /path/to/session/neural_session.json --speed
```

Or click **Prepare speed** in the viewer. It parses/aligns once and saves the
result. A matching result is reused; **Recompute speed** or `--speed --rerun`
creates a new timestamped run. `--speed` and `--window-summary` are mutually
exclusive. Probe and neural bin arguments do not configure speed. The existing
batch entry point accepts `--speed`, with the usual dry-run and worker controls;
inspect a designated experimental session before running an experimental batch.

**Show speed** is on by default when a matching result is available. The row
sits below events and above PCs, and works with behavior alone or either/both
neural populations. A missing, unprepared, stale, or invalid speed source affects
only this optional row; ordinary neural preparation remains independent.
Display interactions never read the raw log, decode IRIG, or prepare speed.

The row shows **signed logged mm/s**, including negative values and actual zero
updates. It holds each value until the next recorded update using horizontal
steps. It does not take absolute values, invert direction, smooth, resample,
or insert artificial zeros/endpoints. Speed remains visible between choices
when its measurements cover those intervals. Retained samples keep their source
order and equal-time values. The line ends at the last speed timestamp.

Version `treadmill-speed-v2` handles one specific logging ambiguity: a group of
reset offsets printed immediately before its sync record. It discards the
ambiguous speed updates only when retained readings on either side are less
than 50 ms apart in both source-clock and UTC time. It does not assign a guessed
timestamp. The step plot holds the previous reading until the next retained
update. Other resets, unsupported timing, longer gaps, or missing neighbors
remain explicit preparation errors. `meta.rejected_speed_records` records each
omitted parsed row, logged value, neighboring rows, and both intervals.

The source-clock reconstruction uses sync intervals plus offsets from the latest
sync edge; it never sums every message or relies on batched PC receipt times.
Current IRIG frames provide absolute UTC anchors. The viewer then subtracts
the common first-trial origin. The speed y range uses the full prepared series
and is independent of PC selection, PC z-scoring, time window, and neural bins.
Hover shows signed mm/s, elapsed/UTC seconds, and the parsed source row.

The complete speed series is sent to the browser; no choice-window or bounded
neural filtering is applied to it. The existing neural loading bounds still
limit native panning when that option is enabled. Showing/hiding speed preserves
the current panned time interval, and Jump/time controls request a new interval
as before. The navigator retains the existing session extent.

Outputs are separate from all PCA runs:

```text
analysis_runs/treadmill_speed_<date>_<time>_<configuration_id>/
  results.npz
  config.json
  run.log
  summary.md
  run_session.py
  run_batch.py
  measurements.json
```

The NPZ has named `(speed_sample,)` arrays `utc_s`, `speed_mm_s`, and
`source_rows`, plus `(decoded_frame,)` arrays `frame_clock_s` and `frame_utc_s`.
Its `meta` dictionary records version, source identity, units, axes, generator,
code revision, parameters, and timing diagnostics. The source log's path, size,
mtime, and analysis version control reuse; trial origin, probe assignments,
and neural bin width do not. Logs and measurements separate parsing, alignment,
writing, and total time. No statistical tests are part of speed preparation.

CT026 2026-08-04 was prepared with 54,731 of 54,733 readings and all 63 decoded
frame anchors. The two omitted speed rows are zero-based parsed positions
110224 and 144284. Their retained UTC intervals are 36.646 ms and 8.142 ms.
The signed range remains -198.72 to 526.86 mm/s. Preparation took 0.36 s
(0.25 s parsing, 0.05 s alignment); the saved 414-kB NPZ loaded in about 3 ms.
Both original PCA archives and their Stage 2 summaries remain byte-identical
and reusable after adding the speed metadata.

One measured viewer comparison used 10 PCs per region, the full 500-ms result,
and bounded 20-ms neural loading. Speed added about 2.42 MB to either chart's
JSON; figure construction plus serialization took about 1.5-1.8 s with or
without speed. These are single local calls, excluding browser latency and
with unspecified filesystem cache state, rather than a latency guarantee.
Validation JSON, benchmark script/measurements, and the rejection ledger are
saved in the speed run directory.

## Outputs, measurements, and batch preparation

Outputs stay under the original session data directory:

```text
analysis_runs/session_timeline_pca_<date>_<time>_<configuration_id>/
  results.npz
  config.json
  run.log
  summary.md
  run_session.py
  run_batch.py
  window_variance_<date>_<time>_<configuration_id>/
    results.npz
    config.json
    run.log
    summary.md
    run_session.py
    run_batch.py
```

The named-array NPZ retains choices/events, origin, regional eligibility,
qualified unit IDs, bin edges, scaling, full loadings/spectrum/scores, and a
dictionary called `meta` with parameters, units, axes, source identities,
versions, and timings. Script copies and the summary/log make the run easy to
inspect. Experimental outputs are refused inside the source repository.
Clearing the app's in-memory cache does not delete prepared results.

Stage 2 sidecars contain named arrays for original-order sample variances,
sorted original PC IDs/fractions/cumulative fractions, counts/statuses,
direct totals, regional choice positions, bin edges, and unit IDs. Their
`meta` dictionary records the metric version, generator, parameters, units,
axis conventions, code revision, and exact Stage 1 path/size/mtime identity.
Changing that source identity or the metric version invalidates reuse.
No Stage 1 archive is modified.

Add `--benchmark` to the preparation command to save a timestamped benchmark
JSON beside the result. It separately measures NPZ loading, default figure
construction, JSON serialization/payload size, and process peak memory, and
includes the recorded loading/binning/PCA/projection timings. Filesystem cache
state is unspecified; peak RSS includes the process lifetime. Browser
responsiveness still needs inspection. Smaller synthetic sessions do not
establish performance on a real session.

For an accepted collection of sessions, make a text file with one metadata
path per line; relative paths are relative to that list file. Blank lines and
lines starting with `#` are ignored. Inspect the batch first:

```bash
uv run -m src.neural_analysis.population.run_session_timeline_batch \
  --session-list /path/to/sessions.txt --bin-width-ms 500 --dry-run
```

Remove `--dry-run` to prepare. Workers run across sessions, defaulting to
available CPU cores capped by session count. Set `--workers 1` or another
smaller count when RAM or disk throughput limits concurrency. Version-2
sessions can use the same explicit `--pfc-probe`/`--hpc-probe` arguments when
their metadata keys match across the list. Inspect one designated real
session before running an experimental batch.

Python callers can use `run_session_timeline`,
`session_timeline_io.load_timeline_result`, and
`timeline_plotting.plot_session_timeline` for the same separate preparation,
load, and figure-building steps used by the app. Pass `zscore_pcs=True` to
`plot_session_timeline` for the same optional display normalization.

The webapp uses a small Streamlit v2 chart component to keep the same Plotly
element alive during display updates. It uses Plotly's installed JavaScript
bundle locally, with no CDN or additional dependency. The existing figure
builder supplies the chart; no saved-result format changes are needed.
Frontend lifecycle tests run through pytest when Node is on PATH;
alternatively set `NODE_BINARY` to its executable.
