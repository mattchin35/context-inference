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
one file. Startup reads metadata and checks paths but does not
load LFP or spike arrays. Selecting a spike-dependent view loads only its
selected population; selecting a cached summary keeps the existing saved
provenance. A missing optional source disables the affected view and explains
which source is unavailable.

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
