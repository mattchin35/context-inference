# Single-Node Cluster Spike-Phase Computation: Technical Specification

## 1. Purpose and status

This document specifies a portable way to run the existing LFP Spike-phase
analysis on one SLURM-managed cluster node and later inspect the returned cache
in the local web application. The cluster is an offload host: it provides more
cores, a longer uninterrupted runtime, and keeps the local workstation free.
It is not a distributed-computing design.

This is a technical design specification, not an implementation plan. It
defines required behavior, data contracts, safety properties, and known gaps.
It does not assign code changes, define a commit sequence, list RED/GREEN test
steps, select final SLURM resources, or authorize an experimental-data run.
Those decisions belong in a later implementation plan.

The design is deliberately narrow:

- One session is processed by one submitted job.
- Only the `spike_phase` component is computed and returned.
- One Python driver invokes the existing production analysis pipeline.
- One shell script requests a single node and invokes that driver.
- Data staging to the cluster and result transfer back are manual operations.
- A Conda environment is used on the cluster; `uv` is not required there.
- A returned cache is selected explicitly in the local web application.

The design should leave clear seams for other components or batch submission
later, but those capabilities must not complicate the first working path.

## 2. Decisions already made

The following are design inputs, not open questions:

1. The user stages required session data to the cluster. Preprocessed data that
   is created locally is synchronized to the cluster before submission.
2. The cache must be portable between different absolute cluster and local
   paths.
3. The Python driver is session-generic, rather than CT026-specific.
4. The first version computes only Spike-phase and returns only a Spike-phase
   component.
5. Scientific settings are supplied in a reviewed configuration file.
6. The job uses an exact Git checkout, identified by commit.
7. Cluster dependencies are provided through a small Conda environment.
8. Work storage may be large; persistent cluster storage is available.
9. Completed results are normally returned with `rsync`.
10. Interrupted or failed jobs are resubmitted manually.
11. CPU, memory, and wall-time requests will be chosen from benchmarks rather
    than fixed prematurely in this specification.
12. The web application must support selecting a returned cache without
    replacing the current default cache.
13. Git, Conda, `rsync`, `squeue`, and `sacct` are available on the cluster.
14. Cluster results live under the session's
    `processed/lfp_summary_cache` directory.

## 3. Scope and non-goals

### 3.1 In scope

The cluster path covers:

- Validating a session-relative Spike-phase configuration.
- Validating that all configured inputs are present and unchanged.
- Materializing cluster-local absolute paths only for file access.
- Preparing LFP phase and trial-local spikes through production adapters.
- Computing the configured observed PPC and trial-shuffle inference.
- Using the existing bounded process-worker and checkpoint facilities.
- Atomically writing and validating a final `spike_phase.npz` component.
- Recording scientific configuration, input identity, code identity, execution
  details, progress, warnings, and terminal state.
- Safely resuming work after a manually resubmitted job.
- Transferring a self-contained completed cache back to the local session.
- Explicitly selecting and validating that cache in the web application.

### 3.2 Out of scope

The first version does not include:

- Multi-node computation, MPI, Dask, Ray, or a custom distributed reducer.
- SLURM job arrays or automatic division across sessions.
- Automatic upload, download, or remote submission from the web application.
- Automatic resubmission after timeout, preemption, or failure.
- Power or Synchrony computation on the cluster.
- Merging a returned Spike-phase file into an existing mixed-component cache.
- Replacing an existing local cache automatically.
- A general workflow manager or database of cluster jobs.
- Changing the PPC estimator, null model, phase transform, array axes, units,
  missing-value rules, randomization, or FDR correction.
- Choosing final worker, memory, or wall-time defaults before representative
  optimized benchmarks exist.

The performance changes described in `docs/ppc_speedup.md` are logically
separate. The cluster path must run the production implementation that exists
at its recorded Git commit; it must not carry an alternate numerical
implementation.

## 4. High-level operating model

The intended flow is:

```text
local authoritative session
        |
        | manual input sync
        v
cluster session snapshot
        |
        | one SLURM allocation, one Python parent, bounded process workers
        v
cluster returned-cache directory
        |
        | manual rsync into a temporary local incoming directory
        v
local validation and atomic directory promotion
        |
        | explicit webapp selection
        v
cached Spike-phase inspection
```

The local session remains the authoritative long-term copy. The cluster copy is
an execution snapshot. The job never reaches back to the local workstation and
the web application never reads an actively changing remote directory.

The shell layer owns scheduler and environment concerns. The Python layer owns
configuration validation, input identity, analysis execution, cache writing,
and structured run metadata. The numerical pipeline continues to own phase and
PPC semantics. These boundaries prevent cluster-specific details from leaking
into scientific calculations.

## 5. Session and directory model

### 5.1 Session root

Every invocation has exactly one `session_root`. It is the directory that
contains the session's configured raw and preprocessed inputs and its
`processed` directory. Its absolute value will differ between the workstation
and cluster and is therefore execution metadata, not scientific identity.

All portable source paths must be relative to `session_root`. A portable path:

- must not be absolute;
- must not contain `..` path traversal;
- must resolve inside `session_root` after symbolic links are resolved; and
- uses POSIX-style relative spelling in serialized metadata.

The `session_id` is also checked. A matching directory layout alone is not
enough to treat results from another session as compatible.

### 5.2 Final cache location

To satisfy both simple session-local storage and safe cache selection, a
cluster result must use a distinct run directory below:

```text
<session_root>/processed/lfp_summary_cache/cluster_runs/<run_id>/
```

Writing the returned component directly over
`processed/lfp_summary_cache/manifest.json` or
`processed/lfp_summary_cache/spike_phase.npz` is forbidden. Direct replacement
would conflict with the requirement that the default cache remain untouched
and would expose the web application to partially transferred files.

`run_id` must be human-readable and collision-resistant. It should contain the
analysis name, a UTC creation timestamp, and a short prefix of the portable
scientific fingerprint, for example:

```text
spike_phase_20260909T213306Z_a1b2c3d4
```

The complete fingerprint remains in metadata; the short form is only for
navigation. A completed run directory is immutable. A repeated computation
creates a new run directory rather than overwriting an earlier result.

### 5.3 Persistent work location

Prepared phase arrays, PPC edge statistics, checkpoints, and locks are work
artifacts rather than returned inspection data. Their authoritative location
must be persistent cluster storage, conceptually:

```text
<session_root>/processed/lfp_summary_work/ppc/<run_fingerprint>/
```

They must not exist only in `$SLURM_TMPDIR`, `/tmp`, or another node-local
ephemeral directory. Node-local storage may be used as a disposable read cache
only if loss of that copy does not impair restart. Work artifacts are not
normally synchronized back to the workstation.

The final cache location and work location are separate contracts. A future
implementation must not infer the persistent work root merely from a nested
returned-cache parent if that inference produces the wrong directory.

## 6. Portable scientific configuration

### 6.1 Configuration responsibilities

The reviewed configuration file describes what is computed. At minimum it
must express all Spike-phase-relevant fields already represented by the
analysis contracts:

- session ID;
- LFP sites, stable site labels, acquisition formats, saved channel indices,
  sample rates, voltage units, LFP paths, and aligned-sync paths;
- trial-table path and choice/context/exclusion filters;
- event alignment and exact half-open whole/before/after windows in seconds;
- unit population, probe identity, source paths, selected channels, quality
  settings, and stable unit IDs;
- phase frequency axis in Hz, Morlet parameters, output sample rate, notch
  settings, phase validity thresholds, and block duration;
- PPC epochs, computability and reliability thresholds, shuffle count, FDR
  settings, phase-bin edges, and random seed; and
- schema and analysis version identifiers.

Paths in this file are session-relative. The file must not contain cluster
usernames or absolute `/gs/...` paths, local `/home/...` paths, Conda paths,
SLURM log paths, worker counts, memory requests, or wall-time requests.

The configuration may use the complete existing LFP summary model internally,
but fields that apply only to Power or Synchrony must neither change the
Spike-phase scientific fingerprint nor cause those components to run.

### 6.2 Scientific versus execution settings

Scientific settings determine whether two results represent the same analysis.
Execution settings determine how the same result is obtained.

Scientific identity includes the session, source roles and logical paths,
trial selection, site and unit selection, phase transform, PPC rules, shuffle
count, and seed. Execution identity includes:

- physical session and repository roots;
- output and work roots;
- SLURM job and node identities;
- worker count;
- unit, shuffle, and trial-edge block sizes;
- checkpoint retention and progress frequency;
- thread-control environment variables; and
- start, stop, and elapsed times.

Execution settings must be recorded in provenance but excluded from the final
Spike-phase scientific fingerprint. Changing worker count or checkpoint block
size must not make scientifically identical output appear stale.

### 6.3 Runtime path materialization

The driver resolves each logical source path against the physical
`session_root` only after validating containment. The resulting absolute paths
are used to open files but are not fed into portable scientific identity.

This distinction is a required change from the current cache assumptions. The
current `component_fingerprint` includes absolute session and source paths, and
the current `fingerprint_source_files` mapping is keyed by resolved absolute
paths. A cache produced under `/gs/...` would therefore be reported stale under
`/home/...` even if every byte were copied correctly. Merely adding a portable
sidecar file would not fix this. Both cache production and cache assessment
must recognize a versioned session-relative identity scheme.

Legacy local caches may retain their existing absolute-path identity. The
manifest must identify which scheme it uses so that portable and legacy
comparisons are never mixed silently.

### 6.4 Unsupported scientific options

A configuration value must not be accepted merely because it can be parsed or
fingerprinted. In particular, any amplitude-threshold option that is not yet
applied by the production numerical runtime must be rejected at preflight when
non-default. Otherwise the cache would claim to represent a setting that was
not actually used. This restriction can be removed only when the production
analysis supports and tests that setting.

## 7. Portable input-snapshot identity

### 7.1 Purpose

The input snapshot answers two questions:

1. Did the cluster compute against the staged files that the reviewed
   configuration identified?
2. After the result returns, do the corresponding files in the local session
   still represent that same input snapshot?

It is an integrity and staleness check, not a substitute for archival storage
or a cryptographic hash of terabyte-scale recordings.

### 7.2 Snapshot entries

Every source used by Spike-phase must have a deterministic entry containing:

- its semantic role, such as trial table, site LFP, aligned sync, sorter source,
  or aligned spike source;
- its normalized session-relative path;
- whether it is a regular file or directory;
- byte size for a file;
- modification time with the finest available resolution;
- SHA-256 for small regular files; and
- for directory inputs, a deterministic inventory of relevant contained files
  using the same relative-entry rules.

The threshold that defines a small file must be simple, deterministic, and
recorded in the snapshot. A single default such as 64 MiB is preferable to a
large role-specific policy. Full hashing is not required for large raw LFP
files. Their portable identity is relative path, size, and preserved
modification time.

For hashed files, path and size must match. A modification-time mismatch should
trigger content-hash verification rather than automatically rejecting
identical content copied by a filesystem with different timestamp precision.
For un-hashed large files, size and modification time are necessarily the
authoritative inexpensive identity. This leaves a documented limitation: a
large file changed in place while preserving both size and modification time
will not be detected.

### 7.3 Snapshot timing

The cluster driver must build or verify the snapshot during preflight and
verify it again immediately before publishing a completed cache. If any input
changes during the run, the job fails without a completion marker. This avoids
publishing a cache assembled from inconsistent versions of a source file.

The local cache loader resolves the recorded relative paths under the active
local `session_root` and performs the same comparison. It never requires the
cluster absolute path to exist.

### 7.4 Input synchronization expectations

Input synchronization is intentionally manual, but the operating contract is:

- all configured inputs are synchronized before submission;
- modification times are preserved where possible, such as with an
  archive-style `rsync` transfer;
- an incomplete or still-changing preprocessing output is not submitted;
- the cluster preflight is the final authority on missing or mismatched files;
  and
- destructive broad synchronization, especially `rsync --delete` against an
  entire `processed` tree, is not part of this workflow.

The input sync and result sync are separate operations with separate source and
destination scopes. Returning one cache directory must not overwrite newer
locally preprocessed inputs.

## 8. Python driver contract

### 8.1 Role

The Python driver is a thin, noninteractive adapter around the production
Spike-phase pipeline. It must use the production dependency factory and
`compute_spike_phase_component`; it must not duplicate phase preparation, PPC,
shuffle generation, FDR correction, payload assembly, or manifest writing.

The driver owns:

- loading and validating portable configuration;
- binding one physical session root;
- constructing the portable input snapshot;
- selecting a new or resumable run;
- binding execution settings;
- reporting structured progress;
- invoking exactly one Spike-phase computation;
- validating the committed result by reopening it without pickle support;
- writing provenance and the human-readable summary; and
- publishing a final completion marker only after all required artifacts are
  complete.

It must not import or launch Streamlit and must not require an interactive
display.

### 8.2 Minimal invocation inputs

The user-facing inputs should remain small:

- physical session root;
- reviewed portable configuration file;
- optional run directory or run ID when explicitly resuming;
- worker count, normally derived from the SLURM allocation; and
- an explicit resume/recovery choice when incomplete work exists.

The final output base is derived from the session root as
`processed/lfp_summary_cache/cluster_runs`. Work paths are derived from the
session root as described above. The driver should not require the user to
repeat these routine paths on every invocation.

A read-only preflight or dry-run mode is desirable because it can report the
resolved session, scientific fingerprint, source inventory, unit count,
shuffle count, output path, work path, and requested workers without loading
raw LFP or creating final output. It is not a separate analysis mode.

### 8.3 Exit behavior

The process returns zero only when:

- computation reports `spike_phase` complete;
- the final component and manifest reopen and validate;
- required run metadata is written;
- input identity still matches preflight; and
- the completion marker is durably published.

Configuration errors, missing inputs, stale input snapshots, unavailable
scientific options, lock conflicts, numerical failures, invalid arrays, failed
metadata publication, and external interruption must produce a nonzero exit.
An existing completed run is not silently recomputed or overwritten.

### 8.4 Progress

Progress output must be useful in both a terminal and a SLURM log. Each record
should include UTC time, run ID, component, stage, completed and total work
when known, elapsed time, ETA when supportable, and a short message. Progress
must be flushed promptly. It must not serialize phase arrays, spikes, or other
large data into logs.

Warnings remain warnings only when they cannot invalidate the scientific
component. A post-commit checkpoint-cleanup warning, for example, may be
recoverable; an input change or malformed component is not.

## 9. SLURM shell contract

### 9.1 Allocation model

The shell script requests one node, one SLURM task, and multiple CPUs for that
task:

```text
--nodes=1
--ntasks=1
--cpus-per-task=<N>
```

The Python parent process then owns a bounded process pool on that node. The
script must not use `-n 16` to mean threads while also declaring
`--ntasks=1`; under SLURM, `-n` conventionally requests tasks. This ambiguity
in the example `hpc_postprocess.sh` must not be copied.

Memory and wall time are per-job resource requests selected from measurement.
The first path does not request multiple nodes and does not launch one Python
interpreter per CPU.

### 9.2 Shell responsibilities

The shell script supplies only cluster concerns:

- partition/account/QOS and notification settings required by the site;
- one-node CPU, memory, and wall-time requests;
- stable SLURM stdout/stderr locations;
- exact repository checkout location;
- Conda environment activation;
- physical session and configuration paths;
- worker count consistent with the allocation;
- thread-pool limits; and
- the call to the Python driver with proper quoting and exit propagation.

It should fail on unset variables and failed commands. A final `echo "Done"`
must not mask a failed Python process. It should not rely on an interactive
`.bashrc` to happen to activate Conda or set `PYTHONPATH`. Conda's shell hook
should be initialized explicitly, the repository root should be selected
explicitly, and the exact Python executable/environment should be logged.

The repository must be at the reviewed Git commit before computation. The job
records the requested commit and actual `HEAD` and fails if they differ. A
dirty checkout should normally be rejected because a commit hash alone would
not identify the executed code.

### 9.3 Nested numerical threads

Each process worker performs NumPy/SciPy work. BLAS and OpenMP libraries must
not independently create many threads inside every worker. The shell layer
sets the applicable OpenMP, MKL, OpenBLAS, and similar thread counts to one
unless a benchmark justifies another value. The driver verifies that the PPC
worker count does not exceed `SLURM_CPUS_PER_TASK`.

This is essential: requesting 32 CPUs and then running 32 worker processes each
with 32 BLAS threads can make the job slower and less stable while violating
the intended allocation.

### 9.4 Signals and scheduler provenance

The Python process must receive scheduler termination signals so it can stop
submitting work, cancel pending futures where possible, and leave checkpoints
consistent. Whether the site prefers a direct Python launch or `srun` for the
single task may remain a small site-specific shell detail, but the selected
method must propagate the Python exit status.

The run records at least `SLURM_JOB_ID`, array identifiers if ever present,
node list, allocated CPUs, requested memory, partition, and time limit when
SLURM exposes them.

## 10. Conda environment contract

The later implementation should include a small environment file for the
cluster. It must use Python 3.11 or newer and pin the versions that were
actually tested. The compute path directly needs standard scientific packages
including NumPy, Pandas, SciPy, and Pynapple.

The current import graph is slightly broader than the numerical ideal. It also
loads Matplotlib and scikit-learn through project modules, and the SpikeGLX
reader currently imports Matplotlib and `tkinter` at module import time. Until
those imports are safely decoupled, the cluster environment must include those
runtime requirements even though the job does not plot or open a GUI.
`MPLBACKEND=Agg` should be set for headless execution.

Pynapple may be installed in the Conda environment through Conda or a `pip`
section, depending on the cluster channels, but its tested version must be
pinned. The environment must also contain versions of SciPy and NumPy that
provide the exact APIs already used by the production pipeline. The job should
not install or upgrade packages at runtime.

Streamlit, Marimo, Dynamax, Seaborn, Formulaic, and Statsmodels are not needed
merely to compute Spike-phase unless the project packaging/import graph forces
them. Installing the full project dependency set is easier but less minimal
and creates more opportunities for cluster solver problems. The preferred
compute environment contains only the tested transitive requirements of this
path plus a lightweight test/smoke-test facility.

The project checkout may be made importable through a controlled editable
installation without dependency resolution or by running from the exact
repository root. An arbitrary inherited `PYTHONPATH` is not part of the
reproducibility contract.

## 11. Computation and publication lifecycle

### 11.1 States

A run has the following conceptual states:

```text
created -> preflight_validated -> running -> component_committed
        -> bundle_validated -> complete
```

It may enter `failed` or `interrupted` before `complete`. Work checkpoints may
remain after either state. Only `complete` is eligible for transfer and webapp
selection.

### 11.2 Preflight

Before expensive work, the driver verifies:

- configuration schema and all numerical invariants;
- session ID and containment of every logical path;
- existence and readability of required source files;
- absence of unsupported non-default settings;
- exact clean Git commit;
- Python and package versions;
- allocated CPUs versus requested workers;
- writability and free space of persistent work and final-cache parents;
- portable scientific fingerprint and input snapshot;
- final run-directory collision status; and
- lock/resume status for matching work.

Preflight errors must be specific. For example, "trial table missing at logical
path X under session root Y" is preferable to a later generic loader error.

### 11.3 Production computation

The driver constructs the production Spike-phase dependencies and invokes the
component pipeline once. Scientific inputs and array contracts are unchanged
from local execution. Process parallelism remains bounded by the execution
configuration and node allocation. Checkpoints and prepared phase data are
identified by their scientific and representation fingerprints, so a restart
cannot accidentally reuse incompatible work.

The final cache is written through the existing manifest-last component
transaction. Named NPZ arrays, axis metadata, units, NaN semantics, and
`allow_pickle=False` validation remain mandatory.

### 11.4 Bundle publication

The component manifest is the commit point for the scientific NPZ transaction,
but the returned run directory also contains provenance and transfer metadata.
Therefore the directory needs a final small completion marker written after:

1. the component manifest and NPZ validate;
2. final input-snapshot verification succeeds;
3. provenance, configuration, logs, and summary are closed;
4. checksums for immutable returned files are written; and
5. required files are flushed to persistent storage.

The completion marker is written last and includes the run ID, full scientific
fingerprint, completion time, and checksum-manifest identity. Nothing covered
by the checksum manifest may change after this marker is written.

A directory without a valid completion marker is incomplete even if it happens
to contain `spike_phase.npz`. The web application and result-transfer procedure
must ignore it by default.

## 12. Returned-cache contents

A completed run directory contains at least:

```text
<run_id>/
    manifest.json
    spike_phase.npz
    scientific_config.json
    input_snapshot.json
    provenance.json
    checksums.json
    summary.md
    run.log
    run_spike_phase.py
    hpc_spike_phase.sh
    COMPLETE.json
```

The last two script names represent immutable copies of the Python entry point
and submitted shell script used for that run. Their maintained source may have
different names, but the returned copies make the analysis run self-contained.

### 12.1 `manifest.json` and `spike_phase.npz`

These use the existing safe LFP-summary component schema. The manifest contains
exactly one complete `spike_phase` component entry. Power and Synchrony are
absent, not marked failed and not represented by placeholders.

The Spike-phase entry retains the production array schema, axes, physical
units, missingness rules, configuration snapshot, analysis version, portable
scientific fingerprint, portable source identity, seed, and completion time.
The NPZ contains named, non-object arrays and loads with
`allow_pickle=False`.

### 12.2 `scientific_config.json`

This is the exact reviewed portable configuration used by the run, serialized
deterministically. It contains logical paths and scientific settings, not
physical cluster roots or scheduler resources.

### 12.3 `input_snapshot.json`

This contains the session-relative source inventory and hashes described in
Section 7. It records the identity policy and small-file threshold so the local
validator can reproduce the comparison.

### 12.4 `provenance.json`

Provenance includes:

- run and session IDs;
- requested and actual Git commit and clean/dirty state;
- repository origin when available without credentials;
- Python executable and version;
- installed versions of numerical dependencies;
- Conda environment name or prefix identifier;
- physical cluster session, repository, output, and work paths;
- host and SLURM allocation metadata;
- execution-only PPC settings and thread limits;
- start, completion, and elapsed times;
- terminal state and warnings; and
- the full scientific and work fingerprints.

Absolute cluster paths belong here because they describe execution. They are
not used for local cache compatibility.

### 12.5 `summary.md` and `run.log`

The summary states the goal, session, component, selected conditions/sites/
epochs/units, shuffle count and seed, scripts used, Git commit, runtime,
warnings, and output files. Because the PPC pipeline performs inferential
tests, it should also provide a restrained scientific inventory of computable,
reliable, and FDR-significant results by relevant grouping. It must not invent
a biological conclusion merely to make an operational run report sound
scientific.

The log records configuration, progress, warnings, errors, and timing in a
human-readable form. It contains no raw signals or giant serialized arrays.
The SLURM stdout/stderr log may be maintained separately by the scheduler; the
returned `run.log` is the stable run-level record.

### 12.6 Checksums and completion

`checksums.json` contains SHA-256 and byte size for every immutable returned
file it covers, excluding itself and the completion marker to avoid recursive
identity. `COMPLETE.json` identifies `checksums.json` by SHA-256 and is the
bundle-level publication point.

## 13. Resume and lock safety

### 13.1 Normal manual resubmission

After a timeout, preemption, node failure, or interrupted connection, the user
submits a new one-node job with the same session, scientific configuration,
and explicit resume target. The driver revalidates configuration, source
snapshot, code compatibility, work fingerprint, and every checkpoint before
reuse. Invalid or mismatched work is rejected rather than partially reused.

Resume applies to execution work, not completed final cache replacement. If a
valid completed cache already exists, the driver reports it and exits without
overwriting it.

### 13.2 Cross-node stale locks

The current PPC work-cache recovery proves liveness only for a PID on the same
hostname. That is intentionally insufficient for SLURM resubmission because a
new job may land on another node, where the old PID cannot be interpreted.

Cluster lock ownership therefore needs, at minimum:

- lock schema and scope;
- work/run fingerprint;
- hostname and PID;
- SLURM cluster identity when available;
- SLURM job ID; and
- creation/update time.

A different-node job must never delete a lock merely because the recorded PID
does not exist locally. Recovery is allowed only after the operator explicitly
requests resume/recovery and SLURM state confirms through `squeue`/`sacct` that
the recorded job is no longer running or pending. If scheduler state is
unavailable or ambiguous, recovery fails safely and tells the user what to
inspect. Lock removal is narrowly scoped to the validated fingerprinted work
directory.

Manual resubmission does not mean manual editing of checkpoint metadata or
blind deletion of a whole work tree.

### 13.3 Incompatible restart

Changes to scientific parameters, source identity, representation schema,
analysis version, or relevant code create an incompatible work identity. They
must start separate work. Execution-only changes such as worker count may
resume when the checkpoint format explicitly permits it.

## 14. Result transfer back to the workstation

Only a completed run directory is transferred. A safe manual return procedure
has three phases:

1. Copy the exact cluster run directory to a uniquely named local incoming
   directory under the same session's `processed/lfp_summary_cache` area.
2. Validate `COMPLETE.json`, the checksum manifest, every covered file, the
   safe manifest/NPZ schema, session ID, portable configuration, and local
   source snapshot.
3. Rename the validated incoming directory atomically to its final
   `cluster_runs/<run_id>` name on the local filesystem.

The user should not rsync directly into a directory currently selected by the
web application. Interrupted transfers remain visibly incomplete and are not
selectable. Re-running rsync into the incoming staging directory is safe.

The result transfer does not include the cluster work cache by default. It also
does not copy files into the default cache root, merge manifests, or delete
other local results.

## 15. Web-application cache selection

### 15.1 Selection behavior

The web application must allow an explicit cache directory to be selected for
the active session. The simple primary interface should enumerate the default
cache and completed returned-cache directories under:

```text
<active_session>/processed/lfp_summary_cache/cluster_runs/
```

A manual directory entry or browse control may support valid cache directories
copied elsewhere, but it receives the same validation. Selection is local UI
state. It does not move files, rewrite manifests, merge components, or change
the configured default cache.

The UI should display enough identity to avoid confusing similar runs: run ID,
completion time, Git commit, shuffle count, selected trial filters, worker
count, runtime, and compatibility state.

### 15.2 Validation before use

Before a returned cache becomes active, the loader verifies:

- valid bundle completion marker and file checksums;
- manifest and component filenames are safe local basenames;
- session ID matches the active session;
- identity scheme and schema versions are supported;
- exactly the intended Spike-phase component exists and is complete;
- NPZ arrays satisfy named-axis and dtype contracts with pickle disabled;
- portable scientific configuration matches the active Spike-phase settings;
- relative source paths resolve inside the active session root; and
- local source identity matches the returned input snapshot.

The cache is classified as compatible, stale, incomplete, or failed with
specific reasons. A stale cache may be inspectable only through an explicit
diagnostic path; it must not be presented as the current compatible result.

Because the returned cache is Spike-phase-only, Power and Synchrony controls
should appear unavailable for that selection. Missing components are not
errors. The application must not silently fall back to Power or Synchrony from
another cache, since that would create an unrecorded mixed result.

### 15.3 Current compatibility gap

The current cache assessment compares absolute-path-based configuration and
source fingerprints. The current production web dependency surface also does
not yet provide all returned Spike-phase selection behavior. Consequently,
copying a cluster NPZ and manifest into place is not by itself sufficient. The
portable identity and selectable-cache contracts in this specification must be
supported before the web application can call such a returned cache
compatible.

## 16. Failure handling

Expected terminal categories include:

- `configuration_error`: invalid schema, parameter, or unsupported option;
- `input_error`: missing, unreadable, escaping, or changed source;
- `environment_error`: wrong Git commit, dirty checkout, missing package, or
  incompatible runtime;
- `resource_error`: allocation mismatch, storage exhaustion, or memory failure;
- `lock_error`: live, ambiguous, or incompatible work ownership;
- `compute_error`: preparation, interpolation, PPC, shuffle, or payload failure;
- `validation_error`: malformed manifest/NPZ or failed post-write reopening;
- `interrupted`: scheduler or user termination; and
- `publication_error`: failure writing immutable metadata, checksums, or the
  final completion marker.

Failures are recorded with stage, type, concise message, and traceback in the
run log/provenance where available. They must not create `COMPLETE.json`.
Existing complete caches are never deleted or downgraded by a failed new run.

An out-of-memory kill may prevent Python from writing a final error record, so
the absence of a completion marker plus SLURM accounting remains meaningful.
The next preflight should report any resumable work it finds.

## 17. Resource-selection policy

Final resource numbers are deliberately deferred. They should be based on a
representative benchmark of the optimized production computation, not the
current conservative serial projection alone.

The benchmark must distinguish:

- cold phase preparation from warm prepared-phase reuse;
- parent, individual worker, and aggregate proportional memory;
- unit/site/condition/epoch work actually exposed to concurrent workers;
- storage for prepared phase and checkpoints;
- 100-shuffle preview from 1,000-shuffle final inference; and
- scaling at 1, 2, 4, 8, and larger sensible worker counts on the chosen node.

The requested CPUs should be at or modestly above the chosen worker count to
allow the parent process and I/O. Requested memory should cover measured peak
aggregate memory plus a safety margin, not multiply one process's RSS blindly
when read-only memory-mapped pages are shared. Wall time should cover measured
runtime plus scheduler-safe margin and still rely on checkpoints for
interruption recovery.

The shell script should expose a few clearly labeled site-specific resource
values rather than embedding assumptions throughout Python code.

## 18. Security and data-integrity requirements

Even though this is a trusted research workflow, cache loading and lock
recovery must retain existing defensive properties:

- no pickle-bearing NumPy archives;
- no object-dtype arrays;
- no manifest-controlled path traversal;
- no source path escaping the active session root;
- no following a substituted symbolic-link lock during recovery;
- no broad recursive deletion for stale-work recovery;
- atomic same-filesystem replacement for final files;
- manifest-last component commit and completion-last bundle publication;
- deterministic random seeds saved in scientific metadata;
- explicit axes, units, sample rates, time-window conventions, and NaN
  semantics; and
- no secret tokens or credentials in returned provenance or repository-origin
  strings.

## 19. Deliberate expansion seams

The design permits, but does not initially implement:

- other LFP-summary components using the same portable session identity;
- a session list and SLURM array with one independent session per task;
- automated submit/status/sync commands around the same driver;
- a registry of recent cache runs;
- cluster-side resource presets for node classes;
- retention policies for old checkpoints; and
- richer scientific run summaries.

Expansion should reuse the portable configuration, snapshot, provenance,
publication, and cache-selection contracts. It should not turn the initial
Spike-phase driver into a speculative generic workflow engine.

## 20. Conformance criteria

A cluster path conforms to this specification only if all of the following are
true:

1. The same reviewed session-relative scientific configuration can run under
   different local and cluster absolute roots.
2. A one-node, one-task SLURM job can use multiple bounded Python workers
   without nested-thread oversubscription.
3. The production Spike-phase pipeline, rather than a duplicated cluster
   calculation, creates the component.
4. Only `spike_phase` is computed and represented in the returned manifest.
5. Scientific fingerprints and local compatibility do not depend on cluster
   absolute paths or worker count.
6. All actual inputs are snapshotted before computation and rechecked before
   publication.
7. A terminated job leaves no valid completion marker but can retain validated
   persistent work for manual resubmission.
8. Cross-node lock recovery proves the old SLURM job is inactive and never
   relies on a foreign PID alone.
9. A completed directory is immutable, checksummed, self-describing, and
   contains the scripts and summary for the run.
10. An interrupted rsync cannot become a selectable complete cache.
11. The local web application can select the returned cache, validate it
    against the active session, and leave the default cache untouched.
12. Older completed results are never overwritten when parameters, inputs,
    code, or execution dates change.
13. Failures produce a nonzero process status and preserve existing complete
    caches.

## 21. Remaining non-blocking decisions

No further scientific or architectural answer is required before writing a
later implementation plan. That plan will still need to choose measured
defaults and cluster-specific labels for:

- partition, account/QOS, notification address, and log parent;
- exact repository and session roots on the cluster;
- Conda environment name and pinned tested package versions;
- default CPUs, memory, and wall time after benchmarking; and
- the small-file hashing threshold if 64 MiB is not suitable.

These are deployment values, not reasons to weaken the portable single-node
design.
