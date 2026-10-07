# Neural analysis quickstart

Neural-session metadata is a small JSON file that tells the existing analysis
code where one session's behavior, LFP, synchronization, and spike sources are.
It contains paths and labels only; scientific settings remain in the analysis
code and explicit run commands.

Start from the closest editable example:

- `docs/examples/neural_analysis/open_ephys_session.json`
- `docs/examples/neural_analysis/spikeglx_session.json`
- `docs/examples/neural_analysis/differently_named_session.json`

Copy the example to the session root as `neural_session.json`, then replace its
relative paths. Each probe is one complete record containing that probe's
files, LFP sites, and optional channel restriction. The probe dictionary key is
the only probe label.

## Create and validate session metadata

Create the editable skeleton in an existing session directory:

```bash
uv run python -m src.neural_analysis.session_metadata_cli create \
  --session-root /path/to/session
```

Edit `/path/to/session/neural_session.json`. All paths inside the file are
relative to that session directory. The creator does not search for recordings
or replace an existing file.

The file has these user-edited sections:

| Section or field | Meaning |
| --- | --- |
| `session` | The single session name used in displays and saved outputs |
| `acquisition` | One session-wide value: `open_ephys` or `spikeglx` |
| `behavior` | Required trial table and optional event table |
| `probes` | One complete record per probe: paths, sites, and optional unit channels |
| `site_pairs` | Site-name pairs available to synchrony views |
| optional `cache` | The session-wide reusable LFP-summary cache |

For each probe, `lfp`, `alignment`, `sorter`, `quality`, and `sites` are
required. `alignment` is the one aligned file used for synchronization and
spikes. `sites` maps each site name directly to its zero-based saved LFP
channel. Optional `unit_channels` restricts population analysis to those
zero-based channels; omit it to use every quality-approved inside-brain
channel. The fixed
`spike_times.npy`, `spike_clusters.npy`, and `cluster_info.tsv` children are
resolved directly beneath `sorter`. The LFP sidecar is inferred from
`acquisition`: `lfp_preprocessing.json` beside Open Ephys LFP data, or the
matching `.meta` file for SpikeGLX. No alternative filenames are searched for.

Validate all currently supported action categories:

```bash
uv run python -m src.neural_analysis.session_metadata_cli validate \
  --metadata /path/to/session/neural_session.json
```

To require one action to be ready, add `--action webapp`, `--action power`,
`--action synchrony`, or `--action spike-phase`. A requested unavailable action
returns a nonzero exit code and lists the missing inputs. General validation
lists all four categories and permits an intentionally incomplete skeleton.

Schema version 2 accepts the existing Open Ephys and SpikeGLX source layouts.
It never loads LFP, synchronization, or spike arrays during metadata validation.
Python callers use `load_session_metadata`, `resolve_session_metadata`,
`resolve_probe_sources`, and `validate_session_for_action` from
`src.neural_analysis.session_metadata`. These return frozen records containing
labels, zero-based channel indices, and filesystem paths; they return no signal
arrays and perform no unit conversion.

## Launch the webapp

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata /path/to/session/neural_session.json
```

The app gets the session name, probe sources, LFP sites and pairs, implicit
per-probe populations, behavior tables, and optional summary cache from that
one file. Metadata validation and resolution do not load numerical arrays.
The default Unit raster/PSTH view is spike-dependent, so its initial render
loads the selected population's sorter and aligned-spike arrays. Other views
retain their existing view-specific loading behavior; selecting a cached
summary keeps the existing saved provenance. A missing optional source
disables the affected view and explains which source is unavailable.

## Run Spike-phase/PPC computation

### Local preview

Check a new request locally without loading numerical arrays or starting the
computation:

```bash
uv run python -m src.neural_analysis.lfp_spike_phase_launcher new \
  --session-metadata /path/to/session/neural_session.json \
  --probe rear-probe \
  --cache-directory /path/to/session/processed/lfp-summary-cache \
  --shuffles 100 \
  --workers 8 \
  --dry-run
```

Remove `--dry-run` for the existing 100-shuffle preview. The command keeps the
probe, shuffle count, worker count, cache directory, and optional analysis root
explicit. Metadata supplies only session identity and input paths. The legacy
`--session-path` route remains available for the reviewed CT026 workflow.

Run the local preview only after its dry run succeeds:

```bash
uv run python -m src.neural_analysis.lfp_spike_phase_launcher new \
  --session-metadata /path/to/session/neural_session.json \
  --probe rear-probe \
  --cache-directory /path/to/session/processed/lfp-summary-cache \
  --shuffles 100 \
  --workers 8
```

The existing final intent remains explicit: use `--shuffles 1000 --final-run`.
This is the same computation with the reviewed final shuffle count, not a new
analysis mode.

### Slurm

For Slurm, submit the same arguments through the existing wrapper:

```bash
sbatch src/shell_scripts/hpc_ppc.sh new \
  --session-metadata /path/to/session/neural_session.json \
  --probe rear-probe \
  --cache-directory /path/to/session/processed/lfp-summary-cache \
  --shuffles 100 \
  --workers 8
```

Submit from a tracked-clean repository root. The wrapper fixes the Slurm task
to eight CPUs and requires `--workers 8`; it forwards the scientific arguments
to the same Python launcher.

### Resume and recover

These commands use the saved run configuration and do not reread live session
metadata:

```bash
uv run python -m src.neural_analysis.lfp_spike_phase_launcher resume \
  --run-directory /path/to/session/analysis_runs/exact-run-directory
```

If computation completed but report publication failed:

```bash
uv run python -m src.neural_analysis.lfp_spike_phase_launcher recover-report \
  --run-directory /path/to/session/analysis_runs/exact-run-directory
```

To rebuild only the report from a completed run:

```bash
uv run python -m src.neural_analysis.lfp_spike_phase_launcher rerender-report \
  --run-directory /path/to/session/analysis_runs/exact-run-directory
```

## Where outputs go

- `processed/lfp_summary_cache*` holds reusable component NPZ files and their
  manifest. It stays under the session, outside Git.
- `processed/lfp_summary_work` holds current prepared/checkpoint work managed
  by the existing computation.
- `analysis_runs/<timestamped-run>` holds launcher state, configuration,
  source identity, logs, summary, and the human-readable report.
- The exact `resume` command is printed and saved in an interrupted run.

Never point a new run at the protected historical `processed/lfp_summary_cache`
unless it is already the intended compatible input. Use an explicit new cache
directory for a new source/configuration identity.

## Python API

The small public surface for new-session use is:

- `load_session_metadata(path)` parses the version-2 JSON file;
- `resolve_session_metadata(metadata, path)` resolves relative paths beneath
  the metadata file's session directory;
- `validate_session_for_action(session, action)` reports missing ordinary
  inputs without loading arrays;
- `build_lfp_summary_config(session, request)` builds the existing Power and
  Synchrony configuration from metadata; and
- `build_metadata_spike_phase_config(...)` adds one selected probe population and
  explicit cache/shuffle/worker choices for the existing Spike-phase pipeline.

The dataclass and function docstrings specify path types, zero-based channel
axes, seconds, Hz, and uV. Scientific calculations, plot definitions, cache
formats, and scheduler behavior are unchanged by the metadata layer.

## Task-variable decoding

Task-variable decoding is an offline, single-session analysis. It constructs
categorical and numerical behavioral targets, aligns them to PFC and HPC firing
rates, and compares PFC, HPC, and PFC+HPC decoders using both regional PCA and
direct-unit features. Start with `regularization_mode: "fixed"`; tuned mode is
substantially more expensive and should be used only after a representative
fixed-mode benchmark.

### Prepare the session inputs

Each session needs four small control files before neural arrays are loaded:

1. `neural_session.json`, using the metadata format described above. It must
   contain two distinct probe IDs selected as PFC and HPC in the decoding
   configuration.
2. The augmented chronological trial CSV produced by the normal behavior
   processing workflow. It must include `cur_trial`, `cur_block`, `action`,
   `experimenter_reward_given`, the selected alignment column (`choice_time` or
   `start_time` in UTC Unix seconds), and the source column for every selected
   target.
3. The behavior workflow's trial-feature parameter JSON. The pipeline copies
   its exact bytes into the run directory for provenance.
4. A task-decoding JSON configuration copied from the portable example.

Copy the example into the session root, then edit its relative paths, probe
IDs, unit-selection rules, targets, and scientific settings:

```bash
cp docs/examples/neural_analysis/task_decoding_config.json \
  /path/to/session/task_decoding_config.json
```

All input paths are resolved relative to the configuration. They must remain
inside the metadata session root. `output_root` is also session-relative and
must name a separate child directory outside the Git checkout. An empty
`channel_ids` list means that the configured channel labels, inside-brain rule,
and cluster groups select units; a nonempty list adds an explicit zero-based
channel restriction. An empty `trusted_utc_bounds` object uses ordinary
alignment coverage. Add a probe's `[start, end]` UTC-second bounds only when
those bounds have been independently established.

The normal behavior-processing path should create the complete augmented
table. There is one narrow migration exception: if `rewards_in_block` is the
sole missing column, first make a timestamped backup beside the CSV, then run
the tested behavior-side backfill:

```bash
cp --preserve=all /path/to/augmented_trials.csv \
  /path/to/augmented_trials.csv.backup-YYYYMMDDTHHMMSSZ

uv run python -c 'from pathlib import Path; from src.behavior_analysis.gather_trial_features import backfill_rewards_in_block_csv; backfill_rewards_in_block_csv(Path("/path/to/augmented_trials.csv"))'
```

Confirm the backup exists before invoking the Python command. The helper
refuses an existing destination column and atomically replaces the CSV only
after round-trip validation. If any other required column is absent, use
normal behavior regeneration. Do not hand-edit the CSV, and do not rerun
unrelated models merely to add this derived column.

### Inspect and start one session

Run the bounded dry run first. It validates small metadata and table inputs,
reports target/work/tensor dimensions and source sizes, and does not load
sorter spike arrays or fit models:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session dry-run \
  --config /path/to/session/task_decoding_config.json
```

A persistent `new` or `resume` requires the scoped scientific source files to
be tracked and clean. Keep a deliberately small run in the foreground with:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session new \
  --config /path/to/session/task_decoding_config.json
```

For an unattended local run, use the detached form:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session new \
  --config /path/to/session/task_decoding_config.json --detach
```

The parent performs bounded preparation, prints the immutable run directory,
and returns only after publishing the child ownership receipt. It also prints
the exact status and resume commands. After that message, it is safe to close
the terminal and close the Codex task; neither must remain resident. The child
writes its console stream to `console.log` and durable stage messages to
`run.log`. Do not poll it. Inspect it once later with the printed one-shot
command:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session status \
  --run-directory /path/to/session/analysis_runs/task_variable_decoding_<timestamp>
```

`status` is read-only and never polls a process or scheduler. Add
`--verify-results` to validate a completed publication. You can also inspect
`run_state.json`, `run.log`, and `console.log` directly after the one-shot
status check. An interrupted or failed prepared run resumes from valid
target-local checkpoints with:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session resume \
  --run-directory /path/to/session/analysis_runs/task_variable_decoding_<timestamp> \
  --detach
```

Only one matching owner may execute a run directory. A live receipt or guard
causes another resume to refuse rather than create a second writer. Sending a
normal termination or interrupt to a foreground worker records durable
interrupted state and releases its guard; use `resume` afterward.

A `new` request whose complete fingerprint already has a valid completed run
prints that directory and skips computation. Use `--rerun` only when you
intentionally want another immutable timestamped directory with the same
scientific identity:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session new \
  --config /path/to/session/task_decoding_config.json --rerun --detach
```

### Run a local batch

Create a UTF-8 text file with one configuration path per line. Blank lines and
lines beginning with `#` are ignored; relative entries resolve from the list
file. Preview the ordered sessions and resource calculation with:

```bash
uv run python -m src.neural_analysis.task_decoding.run_batch dry-run \
  --config-list /path/to/task_decoding_configs.txt
```

Launch the batch in the foreground with:

```bash
uv run python -m src.neural_analysis.task_decoding.run_batch new \
  --config-list /path/to/task_decoding_configs.txt
```

Without measured evidence, the safe effective default is one worker even if
more CPUs are available or `--workers` is supplied. A compatible completed
resource run may later be supplied explicitly:

```bash
uv run python -m src.neural_analysis.task_decoding.run_batch dry-run \
  --config-list /path/to/task_decoding_configs.txt \
  --workers 2 \
  --resource-run-directory /path/to/completed/measured-run
```

The evidence is not discovered automatically. It must contain the atomic
`resource_usage.json` produced by an authorized measurement workflow and must
match the scientific source, runtime, platform, thread limits, analysis mode,
and a resource envelope that is not exceeded by any planned session. The
runner then caps cross-session workers by the request, CPU count, session
count, measured peak RSS, and 50% of current `MemAvailable`. Parallelism is
across sessions only; each session fixes OMP, MKL, and OpenBLAS to one thread.

### Outputs and saved-result view

The default run parent is the session's `analysis_runs` directory. Each run is
a direct child named `task_variable_decoding_<timestamp>` (with a collision
suffix when necessary). Important members are:

| Path | Content |
| --- | --- |
| `run_state.json` | Durable lifecycle, current stage, warnings, and completed targets |
| `execution.json`, `local_launch.json` | Requested/actual owner and detached receipt |
| `run.log`, `console.log` | Stage log and detached stdout/stderr |
| `resume_command.txt`, `status_command.txt` | Exact shell-safe follow-up commands |
| `checkpoints/` | Fingerprint-bound, target-local primitive NPZ checkpoints |
| `results.npz` | Final validated non-pickle arrays and JSON metadata |
| `summary.md` | Session, settings, availability, timing, and output summary |
| `figures/` | `categorical_balanced_accuracy.png`, `categorical_auc.png`, and/or `numerical_r2.png` |

Open the existing metadata-driven webapp and select **Task-variable decoding
results**:

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata /path/to/session/neural_session.json
```

The view defaults to the session-relative locator `analysis_runs`. If the
configuration used a nondefault `output_root`, enter that relative locator in
the view's **Results root (relative to session)** field. The selector considers
only validated, completed direct-child runs. All controls load and re-render
saved arrays; they never refit a decoder or load raw spikes.

### Fixed, tuned, and benchmark planning

Fixed mode performs one outer fit for each target x fold x time x region x
representation cell. Tuned mode evaluates 15 candidates in each of three inner
folds and then refits the selected candidate: 46 fits per outer cell instead of
one with the default grid. That is a large tuned-mode work-count warning, not a
minor option change. Memory is driven mainly by the common PFC and HPC rate
tensors and coefficient capacity, while runtime scales with eligible targets,
time bins, folds, regions, representations, and tuning fits.

Before a full session or any cluster proposal, make a separate fixed-mode
configuration containing exactly one representative categorical target and
one representative numerical target, for example `current_action` and
`relative_doubt`. Keep the intended bin width and unit selections. Run its
dry-run, review the reported tensor allocation and categorical/numerical fit
counts, then run that bounded configuration in the foreground only after the
benchmark is explicitly authorized. Record:

- run fingerprint, commit/source fingerprint, Python/library versions,
  platform, CPU count, and one-thread environment;
- full and common-neural trial counts, PFC/HPC units, time bins, fold counts,
  coefficient capacity, and fit counts from the resource envelope;
- `stage_timing_seconds` and `total_timing_seconds` from `results.npz` and
  `summary.md`;
- wall-clock time, peak RSS in bytes measured with `resource.getrusage`, and
  the measurement method; and
- success, warnings, unavailable targets, and checkpoint reuse.

Project runtime by scaling the measured modeling time by the planned full fit
count divided by the measured fit count; treat that as a planning estimate,
not a guarantee. Use measured peak RSS plus the full dry-run tensor allocation
to choose whether the full job fits safely under the local memory budget. Do
not hand-author `resource_usage.json`; the gated WP10 measurement path owns its
atomic schema. Until that evidence exists, batch execution remains at one
worker.

Single-session Slurm submission is intentionally unavailable pending WP11.
That later gate must freeze the exact commit and offline environment, safe
input/result transfer, submit/resume/status commands, and resource request
from WP10 measurements. Do not invent an `sbatch` command from the local CLI.
Slurm array batching is separately deferred until WP13.
