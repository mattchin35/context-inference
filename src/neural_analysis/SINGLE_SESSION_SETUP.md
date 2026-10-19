# Prepare one neural-analysis session

The normal workflow has one input: `neural_session.json`. Schema version 3
stores the stable PFC, HPC, and PPC probe roles, the three PPC cache paths, and
the preview/final shuffle counts. Cache paths are relative direct children of
`processed`; machine-specific session roots do not belong in metadata.

The workflow covers task-variable decoding, pooled condition generalization,
interregional regression, and spike-phase/PPC. LFP Power and Synchrony are PPC
prerequisites and are prepared by the same command; they are not separate
user-facing runs here.

## 1. Add the version-3 analysis block

Keep the existing behavior, probe, site, and site-pair fields. Change
`schema_version` to `"3"`, remove the old top-level `cache` field, and add:

```json
"analysis": {
  "probe_roles": {
    "pfc": "ProbeA",
    "hpc": "ProbeB",
    "ppc": "ProbeB"
  },
  "ppc": {
    "prerequisite_cache": "processed/ppc_prerequisites_open_ephys_v1",
    "preview_cache": "processed/ppc_prerequisites_preview",
    "final_cache": "processed/ppc_prerequisites_final",
    "preview_shuffles": 100,
    "final_shuffles": 1000
  }
}
```

PFC and HPC must be different declared probes. PPC may be the HPC probe. The
three cache paths must be distinct. Schema version 2 remains readable by the
webapp and legacy explicit commands, but the simplified workflow requires
version 3 because it does not guess probe roles or cache intent.

## 2. Initialize and check control files

For the CT026 2026-08-03 session:

```bash
SESSION="$HOME/contextProjectData/CT026/CT026_20260803_latent_inference"
METADATA="$SESSION/neural_session.json"

uv run --frozen python -m src.neural_analysis.single_session_setup init \
  --session-metadata "$METADATA"

uv run --frozen python -m src.neural_analysis.single_session_setup check \
  --session-metadata "$METADATA"
```

`init` creates only missing task-decoding, pooled-decoding, and interregional
configuration files. It never replaces an existing control file. Review those
scientific configurations before running their analyses. Their session and
output paths are relative, so the complete session tree can be synchronized to
another host without editing paths.

`check` reads small control files and named filesystem paths. It does not load
neural arrays, run an analysis, or submit a job. It always prints labeled local
and cluster instructions for all four analysis families. A blocked analysis
keeps its instructions visible beneath its blockers, while the command returns
status 2 until every analysis is ready.

## 3. Task-variable decoding

Validate the single-session task-variable configuration without running the
analysis:

```bash
uv run --frozen python -m src.neural_analysis.task_decoding.run_session dry-run \
  --config "$SESSION/task_decoding_config.json"
```

Submit the existing condition-validation resource profile:

```bash
bash src/shell_scripts/task_variable_decoding_slurm.sh submit-new \
  --config "$SESSION/task_decoding_config.json" \
  --resource-profile condition_validation
```

The wrapper owns resource selection, immutable preparation, submission, and
the submission receipt. It does not depend on any PPC job.

## 4. Condition generalization

Validate the pooled multi-window configuration:

```bash
uv run --frozen python -m src.neural_analysis.task_decoding.pooled.run_session dry-run \
  --config "$SESSION/pooled_task_decoding_config.json"
```

Submit the existing pooled runner:

```bash
bash src/shell_scripts/pooled_task_variable_decoding_slurm.sh submit-new \
  --config "$SESSION/pooled_task_decoding_config.json"
```

Condition generalization uses the task-decoding configuration as its declared
base configuration, but it is a separate run and does not require the ordinary
task-variable job to finish first.

## 5. Interregional regression

Validate the regression configuration:

```bash
uv run --frozen python -m src.neural_analysis.interregional.run_session dry-run \
  --config "$SESSION/interregional_regression_config.json"
```

Submit the existing Slurm runner:

```bash
sbatch src/shell_scripts/interregional_regression_slurm.sh new \
  --config "$SESSION/interregional_regression_config.json"
```

This job is independent of task decoding, condition generalization, and PPC.

## 6. PPC preparation, preview, and final

PPC is the only family with a separate prerequisite job. The `prepare` command
does not run task-variable decoding, condition generalization, interregional
regression, or PPC itself. It prepares Power, Synchrony, and reusable phase.

### Prepare prerequisites

On a workstation, run the preparation command directly:

```bash
uv run --frozen python -m src.neural_analysis.single_session_setup prepare \
  --session-metadata "$METADATA"
```

On a workstation with no prerequisite cache, `prepare` calls the existing
Power implementation, prepares phase once, passes that exact phase object to
the existing Synchrony implementation, and publishes independent preview and
final prerequisite caches. It does not compute PPC.

Do not run that computational command on a cluster login node. Submit the
tracked preparation wrapper from the repository root instead:

```bash
sbatch src/shell_scripts/hpc_lfp_prepare.sh \
  --session-metadata "$METADATA"
```

The wrapper owns the reviewed `normal` partition, eight-CPU, 32-GiB, and
24-hour request. Its log is
`$HOME/logs/lfp_prepare_<job-id>.log`. PPC preview and final resources remain
owned separately by `hpc_ppc.sh` on `unlimited`.

If the metadata-declared prerequisite cache was copied with `rsync -a`, the
cluster preparation first uses the existing provenance/header adoption path
for Power and Synchrony, then creates or reuses the cluster-local prepared-phase
cache. It does not recompute compatible transferred Power or Synchrony.

The command is conservative and idempotent. Complete compatible targets and a
warm phase cache are reused. A partial preview/final pair, unexpected source
members, or stale/running/failed component state is rejected without
overwriting it. A successful invocation writes a timestamped preparation
receipt under `analysis_runs`.

### Preview and final runs

Dry-run the preview first:

```bash
uv run --frozen python -m src.neural_analysis.lfp_spike_phase_launcher preview \
  --session-metadata "$METADATA" \
  --dry-run
```

The actual cluster jobs are:

```bash
sbatch src/shell_scripts/hpc_ppc.sh preview \
  --session-metadata "$METADATA"

sbatch src/shell_scripts/hpc_ppc.sh final \
  --session-metadata "$METADATA"
```

`preview` and `final` read the probe, cache, and shuffle settings from metadata.
They require the preparation products and refuse unsafe retained work. The
legacy explicit `new --probe --cache-directory --shuffles --workers` launcher
route remains available for schema-version-2 and historical workflows.

The PPC Slurm wrapper resource directives are unchanged: the existing
partition, CPU, memory, wall-time, signal, logging, and mail requirements still
apply. `init`, `check`, and the Python `prepare` command do not themselves
submit a Slurm job; cluster preparation is explicit through
`hpc_lfp_prepare.sh`.

## 7. Synchronizing to the cluster

Use `rsync -a` so the source fingerprint policy retains sizes and nanosecond
modification times. Copy the complete session metadata and declared input
tree, especially the LFP sidecars, alignment files, sorter/curation products,
channel-quality files, and the metadata-declared prerequisite cache when it was
prepared on the workstation. Do not use deletion flags.

Normal aliases above the session directory, including a symlinked cluster
`$HOME`, are accepted and canonicalized. The session directory itself,
`neural_session.json`, `processed`, cache directories, and cache members must
be real nonsymlink objects.

Resume, report-recovery, and rerender commands remain documented in the main
[`README.md`](README.md). Task-decoding and interregional details remain in
their package-specific README files.

## 8. Copy cluster outputs back to the workstation

After the cluster jobs are complete, define the matching session locations.
This example uses CT026 2026-08-04:

```bash
LOCAL_SESSION="/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260804_latent_inference"
REMOTE_SESSION="mchin1@mchin1.hpc.einsteinmed.edu:~/contextProjectData/CT026/CT026_20260804_latent_inference"
```

Preview every proposed transfer first:

```bash
rsync -a --dry-run --itemize-changes --relative \
  "$REMOTE_SESSION/./analysis_runs/" \
  "$REMOTE_SESSION/./processed/ppc_prerequisites_preview/" \
  "$REMOTE_SESSION/./processed/ppc_prerequisites_final/" \
  "$REMOTE_SESSION/./processed/ppc_prerequisites_adoption.json" \
  "$LOCAL_SESSION/"
```

If the dry-run lists only the expected new or updated outputs, perform the
copy:

```bash
rsync -a --info=progress2 --relative \
  "$REMOTE_SESSION/./analysis_runs/" \
  "$REMOTE_SESSION/./processed/ppc_prerequisites_preview/" \
  "$REMOTE_SESSION/./processed/ppc_prerequisites_final/" \
  "$REMOTE_SESSION/./processed/ppc_prerequisites_adoption.json" \
  "$LOCAL_SESSION/"
```

`analysis_runs` covers task-variable decoding, condition generalization,
interregional regression, PPC reports, and preparation receipts. The two PPC
cache directories contain the completed preview and final components; the
adoption receipt preserves their cluster provenance. The command has no
deletion option, so local files absent from the cluster remain untouched.

The metadata-driven webapp needs no path-rewrite or adoption command after this
copy. **Cached LFP Summary** reads the metadata-declared preview/final locations
through its read-only `session cache` source. It prefers a fully completed final
cache, then a fully completed preview; a still-running final cache is shown as
partial and does not fabricate `spike_phase.npz`. **Inter-regional regression
results** accepts a copied finalized run when the local metadata file matches
the `session_metadata` size and SHA-256 saved in its input manifest. The saved
absolute cluster paths remain provenance and should not be edited.

`processed/lfp_summary_work` is intentionally excluded. It is a large,
machine-local prepared-phase/checkpoint cache rather than a portable result.
The detailed task-decoding and pooled-decoding sections in the main README
retain their stricter per-run staging and validation procedures when an
individual immutable run must be verified before publication.
