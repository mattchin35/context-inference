# Neural analysis quickstart

Neural-session metadata is a small JSON file that tells the existing analysis
code where one session's behavior, LFP, synchronization, and spike sources are.
It contains paths and labels only; scientific settings remain in the analysis
code and explicit run commands.

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
| `subject_id`, `session_id`, optional date/label | Display and saved session identity |
| `behavior` | Behavior directory, trial table, and optional event/treadmill files |
| `probes` | Stable hardware ID plus explicit Open Ephys or SpikeGLX source files |
| `sites` | Display label, probe reference, and zero-based saved LFP channel |
| `site_pairs` | Existing site-ID pairs available to synchrony views |
| `channel_groups` | Anatomical label, probe reference, and zero-based channel indices |
| `populations` | Stable label linking one probe to one anatomical channel group |
| optional cache/snapshot directory | Existing approved saved output for cached views |

For each probe, `lfp_file`, `lfp_metadata_file`, `synchronization_file`,
`aligned_spike_file`, and optional `channel_quality_file` name files;
`sorter_directory` names the existing sorter output directory. The fixed
`spike_times.npy`, `spike_clusters.npy`, and `cluster_info.tsv` children are
resolved directly beneath that directory. No alternative filenames are
searched for.

Validate all currently supported action categories:

```bash
uv run python -m src.neural_analysis.session_metadata_cli validate \
  --metadata /path/to/session/neural_session.json
```

To require one action to be ready, add `--action webapp`, `--action power`,
`--action synchrony`, or `--action spike-phase`. A requested unavailable action
returns a nonzero exit code and lists the missing inputs. General validation
lists all four categories and permits an intentionally incomplete skeleton.

The initial schema accepts the existing Open Ephys and SpikeGLX source layouts.
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

The app gets the subject/session labels, probe sources, LFP sites and pairs,
anatomical channel groups, populations, behavior tables, and optional summary
cache from that one file. Startup reads metadata and checks paths but does not
load LFP or spike arrays. Selecting a spike-dependent view loads only its
selected population; selecting a cached summary keeps the existing saved
provenance. A missing optional source disables the affected view and explains
which source is unavailable.

The large-computation command will be added in U3 and is not claimed as
metadata-driven until that milestone is implemented and tested.
