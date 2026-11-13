# Session timeline and CCA

This view displays fixed session-wide canonical coordinates for simultaneously
recorded PFC and HPC unit populations. It uses the same completed choices,
behavioral events, spike quality selection, and elapsed-session clock as the
[PCA timeline](SESSION_TIMELINE.md). It does not require a prepared PCA result.

## Prepare and open

Launch the existing app with the session's metadata:

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- --session-metadata /path/to/session/neural_session.json
```

Select **Session timeline and CCA**. Assign distinct PFC/HPC probes and choose
the bin width. Version-3 metadata provides regional defaults; version 2 needs
explicit selections. **Prepare** creates or reuses the selected configuration.
**Recompute** explicitly creates a new run and preserves earlier outputs.

The default is 500-ms bins. Available widths are 500, 100, 50, and 20 ms, giving
8, 40, 80, or 200 bins in each `[-2,+2)` choice window. Counts are unsmoothed;
each bin is positioned relative to the actual choice time.

Preparation uses good inside-brain channels, metadata channel restrictions,
and the existing manual Phy good/mua policy. The historical missing-label-to-MUA
rule is recorded explicitly with the selected unit IDs. Both regions must have
complete recording coverage for a window to enter the paired model. All choices
remain in the behavior track; a window missing either region has no canonical
activity displayed in either region.

For terminal preparation, inspect the planned configuration first:

```bash
uv run -m src.neural_analysis.population.run_session_cca --session-metadata /path/to/session/neural_session.json --bin-width-ms 500 --dry-run
```

Remove `--dry-run` to prepare. Version-2 sessions use explicit `--pfc-probe` and
`--hpc-probe` keys. `--rerun` requests a new run; `--benchmark` records measured
load/figure/serialization costs alongside preparation measurements.

## Navigate and read the display

Both heatmaps use the same canonical-pair ranks. Initially, they show the first
ten available pairs, with an overlaid PFC/HPC trace row for CC1. Change the rank
range to see other pairs, or select a pair for the trace row. Traces use the
same saved scores as the heatmaps and break between choice windows. Hide the
trace row when only the overview is needed.

Each region's canonical score has mean zero and sample SD one across all fitted
session bins. A value of +2 means two session SDs above that component's mean;
it is not a firing rate. Scores are not renormalized for individual trials or
the visible interval. Heatmaps share one fixed symmetric color scale. Manual
limits affect display saturation only. A measured zero is distinct from a
blank interval with no displayed neural measurements.

Use **Visible duration**, **Window start**, and **Jump** to navigate. Dragging
inside the chart pans its synchronized time axes; the wheel scrolls the page.
Display changes preserve the current panned interval. Real gaps between choices
are retained.

At 20 ms, bounded neural loading defaults on. Explicit time controls load another
interval; browser panning stays within the currently loaded bounds and does not
request new data from Python. Bounded plotting leaves fitting, normalization,
correlations, and color limits unchanged. Loading a compressed result can still
require the full regional score arrays in memory.

Optional behavioral tracks and logged active-context shading follow the PCA
timeline conventions. Hover identifies the original trial, region/probe, pair,
actual bin interval, canonical score, and session correlation. The summary lists
paired-window/observation counts, unit counts, and numerical ranks; the full
correlation spectrum is available separately from the plotted row selection.

## Optional treadmill speed

Declare the session-relative log as `behavior.treadmill` in
`neural_session.json`, then click **Prepare speed**. **Show speed** adds an
IRIG-aligned, signed **mm/s** step row below events and above the heatmaps.
Matching saved speed is reused; **Recompute speed** creates a new speed run.
Speed preparation and display do not refit or change saved CCA results.

The row preserves actual update times, signed values, and activity between
choices. It uses the whole prepared series and fixed full-series y limits,
including during bounded neural loading. Hiding/showing it preserves pan.
Missing, stale or invalid speed leaves CCA available with an explanatory
status. Raw logs are parsed only on explicit preparation.

Speed version `treadmill-speed-v3` treats consecutive equal sync states as
redundant pulse reports. Every logged interval still contributes to elapsed
time; only pulse pairing ignores those reports. Their parsed positions and
times are recorded as `redundant_sync_records`. The existing policy for short
ambiguous speed offsets remains separately recorded as `rejected_speed_records`.
Use **Prepare speed** to create a v3 cache for an older prepared session.

CT026 validation retained 29,714 readings and 63 UTC frame anchors. One
redundant LOW retained the full 0.799985-second low phase. One separately
ambiguous speed reading was excluded across a 4.223-ms interval under the
existing policy. Source-clock and retained-value audits passed, and both CCA
archive hashes and modification times were unchanged after browser checks.
The speed run contains `cca_speed_validation.json`, its audit script,
`cca_speed_display_validation.json`, and `viewer_preview.jpg`:

```text
analysis_runs/treadmill_speed_20261113_152337_327562_320dffb85af6bb7c/
```

## Interpretation

PFC CC1 and HPC CC1 are different regional weighted combinations chosen to have
high correlation. Pairs are ordered by their session-wide correlation, rather
than explained regional variance. There is no PCA-style variance-coverage count.

These correlations are fitted and evaluated on this same session. Shared task
responses are retained, and pooled time bins are dependent. The view describes
association; it does not establish held-out generalization or causal direction.
Weakly correlated pairs can look as prominent as strong pairs because both use
unit-SD scores. Signs and axes belong to the saved configuration and need not
match another session, bin width, or independently fitted result.

The numerical fit retains all supported directions after removing constant
units and numerical null directions. It does not use a PCA variance cutoff or
regularization. Sample-rank saturation can force correlations to one; rank
diagnostics identify that condition. An unavailable or failed fit has an explicit
reason and must not be interpreted as zero coupling.

## Saved results and batches

Each completed paired model is stored beneath its source session:

```text
analysis_runs/session_timeline_cca_<date>_<time>_<configuration_id>/
  results.npz
  config.json
  run.log
  summary.md
  run_session.py
  run_batch.py
  measurements.json
```

The NPZ preserves behavior, origin, common/regional eligibility, unit identities,
bin edges, preprocessing, coefficients/offsets, canonical scores, correlations,
numerical diagnostics, and a `meta` dictionary describing versions, units, axes,
parameters, and source identities. Display interactions do not refit or rewrite
these products. Scientific input/settings changes select a different identity.

Saved viewing identifies the actual saved configuration. A historical result
does not establish compatibility with changed current sources; fresh preparation
checks the current inputs. A copied session must retain valid metadata and its
completed CCA result directory. Saved viewing works when raw trial, event, and
spike files are absent; preparation is disabled with a missing-source reason.

For an approved session collection, use one metadata path per session-list line:

```bash
uv run -m src.neural_analysis.population.run_session_cca_batch --session-list /path/to/sessions.txt --bin-width-ms 500 --dry-run
```

Relative metadata paths are resolved against the list file. Remove `--dry-run`
to process; default process count is available CPU cores capped by session count.
Use `--workers` to reduce concurrency when memory or disk throughput is limiting.
One failed session is reported without preventing the others from finishing.
Inspect one designated experimental session before processing a wider batch.

## Validation and measured performance

Validation on 2026-11-13 included an independent finite-sample correlation
oracle, dependent/constant units, rank-one cases, all four bin grids, immutable
archive/reuse checks, real spawned-session execution, and actual browser
hover/pan/rank/pair/trace/Jump/bounded-loading checks. Copied raw-absent results
are covered by actual Streamlit tests. The combined affected suite passed 360
tests; one existing Node-dependent frontend unit test was skipped because Node
was unavailable. Browser checks exercised the actual chart separately.
After the final plotting fixes, the CCA/chart suite passed 164 tests with that
same one skip.

The designated CT026 2026-08-28 session had 696 common complete windows, 141
PFC units and 286 HPC units. Both regional unit sets were nonconstant, with
numerical ranks 141/286 and all 141 pairs retained. Full-array projection errors
were below 2e-14; covariance errors were below 4e-14. Neither fit had
rank-forced unit correlations. This checks numerical consistency, not held-out
or biological validity.

Single-run measurements on the 24-CPU Linux workstation are below. MB/GB use
decimal bytes. Filesystem cache state was unspecified; these are observations,
not performance guarantees.

| Measurement | 500 ms | 20 ms |
| --- | ---: | ---: |
| Preparation, including publication | 3.41 s | 18.07 s |
| Saved NPZ | 12.99 MB | 302.62 MB |
| Saved result load | 0.058 s | 0.993 s |
| Full-session plot JSON | 16.66 MB | 443.72 MB |
| Default viewer JSON, with optional events | 17.61 MB | 18.68 MB |
| Peak process RSS through full benchmark | 0.72 GB | 4.35 GB |

These original CCA measurements exclude the optional speed row. With CT026's
complete speed series, default viewer JSON was 18.91/19.99 MB at 500/20 ms;
construction took 1.38/1.13 s and serialization 0.84/0.87 s, excluding browser
rendering. Speed preparation took 0.21 s. These are single local observations.

Full-session JSON uses the plotter's first ten pairs and CC1, without optional
events. The actual default viewer includes all four available event tracks;
at 20 ms its first 60-second viewport loads neural windows intersecting
[-1.348,118.652] elapsed seconds. Bounded plotting reduced the fine-bin plot
JSON to 17.74 MB before optional events. The full saved 314-MB score arrays
still load, so bounded plotting reduces transport rather than archive memory.
Actual viewer figure construction/serialization took 1.25/0.73 s at 500 ms and
1.06/0.83 s at 20 ms. These timings exclude browser rendering.

The session's completed run directories are:

```text
analysis_runs/session_timeline_cca_20261113_143835_549829_31e43944ef338a61/
analysis_runs/session_timeline_cca_20261113_143906_219681_fb80ed228f329ade/
```

Each includes immutable scientific results, preparation measurements, a full
benchmark JSON, and `validation_audit.json` with a copy of the audit script.
`viewer_preview.jpg` records the verified display for each bin width.
Inspect these two results before processing additional experimental sessions.
The [live handoff](../../docs/session_timeline_cca_implementation_plan.md)
records exact commands, commits, and validation evidence.
