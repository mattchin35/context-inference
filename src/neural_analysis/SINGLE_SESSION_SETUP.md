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
neural arrays, run an analysis, or submit a job. When ready, it prints the
existing commands for all four analysis families.

## 3. Prepare PPC prerequisites

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

## 4. Preview and final PPC runs

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

## 5. Synchronizing to the cluster

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
