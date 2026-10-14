# Prepare one neural-analysis session

`single_session_setup` is a preparation-only command for one session. It can
create missing editable JSON control files, inspect ordinary input paths, and
print commands for later use. It does not preprocess data, load neural arrays,
run an analysis, create an analysis run, or submit a Slurm job.

The supported analysis families are:

- task-variable decoding, including all configured task conditions;
- pooled-window condition generalization;
- fast-first interregional regression (`ols_cv` and `linear_granger`);
- spike-phase/PPC.

LFP power and LFP synchrony are not part of this workflow.

## 1. Scaffold editable control files

Start with an existing session directory and its exact PFC and HPC metadata
probe IDs:

```bash
uv run python -m src.neural_analysis.single_session_setup init \
  --session-root '/path/to/CT026/2025-08-04' \
  --pfc-probe ProbeA \
  --hpc-probe ProbeB
```

The command creates only files that are missing:

- `neural_session.json`;
- `task_decoding_config.json`;
- `pooled_task_decoding_config.json`;
- `interregional_regression_config.json`.

Existing files are reported as `preserved` and are never overwritten. The
three canonical pooled training windows are included by default. Select a
smaller canonical subset with `--training-windows` if that is the intended
scientific analysis.

The generated files are editable templates, not inferred scientific truth.
Review probe roles, paths, regions, channel filters, target/condition lists,
and output locations before continuing. The generated interregional metadata
and output paths are relative to its configuration file, so synchronizing the
complete session tree does not require a cluster-specific rewrite. Existing
absolute-path configurations remain accepted for backward compatibility.

## 2. Supply preprocessing products

This command does not create missing preprocessing products. The session must
already contain the products named by its metadata and configurations:

- augmented chronological trial CSV;
- trial-feature parameter JSON;
- aligned probe synchronization files;
- sorter directories with spike times, spike clusters, and cluster metadata;
- channel-quality tables;
- LFP file and preprocessing sidecar for the PPC probe;
- an existing Power/Synchrony-only PPC source cache directly below
  `session/processed`;
- declared LFP sites and at least one site pair, as required by the existing
  spike-phase launcher.

If these are absent, the readiness report remains blocked and names them. It
does not attempt to repair, align, infer, or generate them.

## 3. Check readiness and print commands

```bash
uv run python -m src.neural_analysis.single_session_setup check \
  --session-root '/path/to/CT026/2025-08-04' \
  --pfc-probe ProbeA \
  --hpc-probe ProbeB \
  --ppc-probe ProbeB \
  --ppc-cache-directory '/path/to/CT026/2025-08-04/processed/ppc-cache'
```

Omit `--ppc-cache-directory` only when `neural_session.json` already names the
intended nonlegacy cache. The report prints `READY` or `BLOCKED` independently
for each analysis. A missing PPC-only LFP input does not block task decoding or
interregional regression.

When all four families are ready, the command prints shell-quoted commands for
later explicit use:

- each existing local preflight/dry-run;
- each existing local analysis command;
- each existing cluster wrapper command;
- the one-time cluster cache-adoption command;
- both the 100-shuffle PPC preview and explicit 1,000-shuffle final PPC run.

The preview commands always use `processed/ppc_prerequisites_preview`; final
commands always use `processed/ppc_prerequisites_final`. The provided
`--ppc-cache-directory` is the rsynced source cache, not either run cache.

Printing these commands does not execute them. Existing launchers and Slurm
wrappers retain their own reviewed computation, lifecycle, and resource
requirements unchanged.

## 4. Cluster path preparation

Synchronize the session non-destructively with `rsync -a`, excluding existing
analysis runs, and use a tracked-clean cluster checkout. Then run `check`
against the cluster-visible session path. Every launcher validates paths in the
environment where it will run.

For the CT026 2026-08-03 example, the complete PPC preparation step on the
cluster is:

```bash
SESSION="$HOME/contextProjectData/CT026/CT026_20260803_latent_inference"

uv run --frozen python -m src.neural_analysis.lfp_summary_cache_adoption \
  --session-metadata "$SESSION/neural_session.json" \
  --probe ProbeB \
  --source-cache "$SESSION/processed/ppc_prerequisites_open_ephys_v1"
```

This command only validates, byte-copies, and rebinds the prerequisite cache.
It does not compute Power, Synchrony, phase, or PPC and does not submit Slurm.
It prints the next preview and final dry-run/submission commands, each using its
own cache. The existing `hpc_ppc.sh` resource requirements remain unchanged.

A symlinked cluster `$HOME` or another alias above the session directory is
accepted. Adoption resolves that alias once and uses canonical paths for every
read and write. The session directory itself and its metadata, `processed`,
cache directories, and cache members must remain nonsymlink objects. No
`readlink -f` step is required with the current command; it remains a valid
workaround for an older checkout.

Do not add deletion to the synchronization command. Prefer the generated
relative interregional paths; older absolute-path configurations must be made
valid for the destination before submission. The detailed sync, submission,
status, resume, and result-return procedures remain in the main
[`README.md`](README.md), the task-decoding
[`README.md`](task_decoding/README.md), and the interregional
[`README.md`](interregional/README.md).
