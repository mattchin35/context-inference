# Session timeline and PCA viewer

Stage 1 displays completed choices, behavioral events, and regional population
PC scores on one elapsed-session timeline. Time zero is the first trial start.
Each neural window covers `[-2, +2)` seconds around an actual choice; the real
gaps between choices stay blank. Choices outside a probe's recording coverage
remain visible without neural measurements from that probe.

Preparation runs once for a session, bin size, and population selection. It
saves the fitted basis and all scores. Viewing, panning, trial jumps, event
toggles, and PC-row changes reuse those products. Changing bin size selects a
different configuration and can change the PC axes.

## Inputs

Use the existing `neural_session.json` in the session's data directory. See
the [metadata quickstart](README.md#create-and-validate-session-metadata) for
the supported source layouts. No additional configuration file is required.

- The processed trial table supplies completed choices, original `cur_trial`
  IDs, `start_time`, `choice_time`, and action `0=right`, `1=left`. Current and
  existing legacy experimenter-reward flags follow the normal project rules.
  Incorrect and unrewarded choices are included. Experimenter rewards,
  nonchoices, and the unsaved active end-of-file trial are excluded.
- Optional event CSVs use the existing `Time`, `Event`, and `Note` columns:
  `trial_start`, `left_entry`, `right_entry`, and LED on/off transitions.
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

The chart axes stay synchronized during native panning. In installed
Streamlit 1.58, changing sidebar display controls restores the explicit
Window start/duration viewport, including after a native mouse pan. Use those
controls or Jump when you want that viewport to survive subsequent reruns.
Numerical results and the variance footer remain unchanged.

For large figures, enable **Load only visible neural interval**. It loads the
current viewport plus neighboring time on either side and retains a compact
full-session navigator. Panning is confined to the loaded interval. Use Window
start or Jump to load another part of the session; dragging the native range
slider does not request new data from Python.

The chart defaults to a light background. Hover a choice for trial ID, side,
and time. Hover a heatmap cell for region/probe, PC identity and session
variance percentage, original trial ID, elapsed/UTC/choice-relative bin
intervals, and the signed score. LED intervals use verified on/off pairs;
unpaired transitions remain markers without invented endpoints.

## Read the PCA display

Each population is fitted separately using all of its complete choice windows.
Units are z-scored once across pooled window/bin observations. Constant units
are omitted. PCA uses full SVD, no whitening, and no post-PCA score scaling.
The PC rows stay in session explained-variance order.

Colors are signed and centered at zero, using fixed regional limits calculated
from all scores in the configuration. Gray center-color cells are measured
near-zero scores; blank gaps have no neural measurements. The optional manual
color-range control changes display limits only and labels saturation. It
does not change scores or normalize rows.

The footer shows the smallest leading-PC counts covering **80%, 90%, and
95%** of total variance in each population's complete fitting dataset. It also
reports unit/window counts and omitted constants. Displaying fewer PCs does
not change the denominator or footer. Regions can have different eligible
windows and unit counts, so their summaries describe their own fitting data.

These PCs describe pooled choice-window activity, not a decision-specific or
noise-corrected latent dimensionality. Signs and identities apply to the
saved configuration and need not match another session or bin width. Stage 2
per-choice variance-participation counts are not part of this Stage 1 view.

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
```

The named-array NPZ retains choices/events, origin, regional eligibility,
qualified unit IDs, bin edges, scaling, full loadings/spectrum/scores, and a
dictionary called `meta` with parameters, units, axes, source identities,
versions, and timings. Script copies and the summary/log make the run easy to
inspect. Experimental outputs are refused inside the source repository.
Clearing the app's in-memory cache does not delete prepared results.

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
load, and figure-building steps used by the app.
