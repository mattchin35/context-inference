# Neural Analysis Refactor Execution Log

## Authority and safety

- Implementation was authorized by the user on 2026-09-23.
- The implementation plan at `docs/neural_analysis_refactor_plan.md` is the
  authoritative ongoing handoff and package sequence.
- Scientific recomputation, cache mutation, artifact replacement, cluster
  submission, and Git push remain separately gated by the plan.
- All Python commands use `uv run`. Tests and implementation are committed
  separately under RED-GREEN-REFACTOR.

## Starting checkpoint

- Branch: `refactor`.
- Pushed planning baseline: `2245475aceadb841d0b13374fddb0a7f4758e71d`
  (`neural analysis refactor prep`), initially equal to `origin/refactor`.
- Implementation-status documentation commit: `d4b1c30` (`docs: start neural
  refactor implementation`).
- Tracked worktree at the planning baseline: clean.
- Complete untracked inventory: 4,077 paths; SHA-256 of
  `git status --porcelain=v1 --untracked-files=all` output:
  `f2970c827607dbc2fec45f0d625d46b3e5ebb76119fbeb911e96924e2655052b`.
- Normal untracked inventory: 168 entries; SHA-256 of
  `git status --porcelain=v1 --untracked-files=normal` output:
  `7c8d98eff71dfcdfcd4da07c71487fbf2fbae6ad4d9d245e7f686936c957a4f5`.
- The untracked inventory is pre-existing user-owned work and remains excluded
  from refactor staging and commits. Fourteen untracked files are under
  `src/tests/neural_analysis`; none is in the NR0 focused command. They may be
  collected by directory-wide baseline runs, but NR0 will not edit or claim
  them as package tests.

## Agent policy and runtime verification

- Lead: user-assigned Sol/high orchestrator.
- NR0 writer: `gpt-5.6-terra`, `xhigh`, one continuing agent for tests and
  implementation.
- NR0 gate reviewer: fresh `gpt-5.6-sol`, `xhigh`, read-only, after each stable
  RED or GREEN diff.
- Optional scout: `gpt-5.6-terra`, `medium`, read-only.
- Child model/effort override requests were accepted by runtime metadata.
- Three initial Sol/high read-only audit assignments were interrupted before
  accepting work product after the lead re-read the stricter role topology in
  Section 2.6. They made no file edits. All subsequent assignments follow the
  exact role topology above.

## Baseline gate

Status: complete. No NR0 test or source edit had started when these baselines
were run.

Required commands:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_open_ephys_behavior_integration.py \
  src/tests/neural_analysis/test_lfp_loading.py \
  src/tests/neural_analysis/test_lfp_spike_phase_launcher.py \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/test_lfp_summary_io.py \
  src/tests/neural_analysis/test_lfp_summary_pipeline.py \
  src/tests/neural_analysis/test_lfp_summary_plotting.py \
  src/tests/neural_analysis/test_lfp_summary_preparation.py \
  src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_webapp.py \
  src/tests/neural_analysis/test_lfp_summary_work_cache.py \
  src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py \
  src/tests/neural_analysis/test_psth_webapp.py

uv run pytest -q -p no:cacheprovider src/tests/neural_analysis

uv run pytest -q -p no:cacheprovider
```

Results:

- Focused NR0 command: 555 passed, 3 warning instances, 35.05 seconds. The
  warnings were two all-NaN plotting warnings and the intentional duplicate NPZ
  member warning.
- Complete `src/tests/neural_analysis` command: 1,246 passed, 22 warning
  instances, 157.52 seconds. Warning categories were the existing
  multiprocessing-fork deprecation, all-NaN plotting, intentional duplicate
  NPZ member, and Pynapple empty-epoch/divide-by-zero warnings.
- Complete repository command: collection stopped with one unrelated error in
  2.90 seconds because the pre-existing untracked
  `src/tests/behavior_analysis/test_project_utils.py` imports unavailable
  `autograd`. The file and dependency are outside NR0. This is accepted as a
  frozen user-owned baseline failure; NR0 will neither edit the test nor add the
  missing dependency.
- Pytest initially could not create a lock in `/home/matt/.cache/uv` under the
  filesystem sandbox. The same commands were rerun with approved access to the
  existing uv cache; no dependency installation or repository write occurred.

Read-only representative CT026 and legacy-artifact check:

- Both ProbeA and ProbeB `lfp_preprocessing.json` files exist, are ASCII/JSON
  readable, and expose the reviewed `float32`, 2,500 Hz, 384-channel,
  time-major layout plus the nested affine scaling object.
- Both corresponding `lfp.dat` files exist at 14,338,163,712 bytes. No binary
  samples were loaded during the baseline check.
- The five-file local final `cache_snapshot` has the expected recorded sizes.
  Calling the existing public `validate_cache_snapshot` read-only returned
  `valid: Valid committed snapshot.`
- The first validator import command lacked `src` on `sys.path` and failed with
  `ModuleNotFoundError`; the corrected command added only the in-process import
  path and passed. Neither command modified the snapshot.
- Tracked status remained clean and both untracked inventory hashes remained
  exactly equal to the starting checkpoint after all baselines.

## Package ledger

| Package | State | Test commit | Implementation commit | Evidence and next gate |
| --- | --- | --- | --- | --- |
| NR0 | Complete and approved | `e728cea`, `2b497e4` | `177a8d6` | Final focused 704 passed; neural suite 1,395 passed; fresh Sol review approved; closure `a94559d`. |
| NR1 | In progress; dry run approved | - | - | Metadata/path/resource evidence passed fresh Sol review. Small numerical windows require separate user approval. |
| NR2 | Pending | - | - | Requires approved NR1 corrected baseline. |
| NR3 | Pending | - | - | Sequential after NR2. |
| NR4 | Pending | - | - | Sequential after NR3. |
| NR5 | Pending | - | - | Sequential after NR4. |
| NR6 | Pending | - | - | Requires NR2-NR5 and NR0 semantics identity. |
| NR7 | Pending | - | - | Requires NR2-NR6. |
| NR8 | Pending | - | - | Requires NR6-NR7. |
| NR9 | Pending | - | - | Requires NR3-NR5 and NR8. |
| NR10 | Pending | - | - | Requires NR2-NR9. |
| NR11 | Pending | - | - | Requires NR3-NR4 and NR8. |
| NR12 | Pending | - | - | Requires NR3-NR6 and NR11. |
| NR13 | Pending | - | - | Requires NR4-NR5 and NR9. |
| NR14 | Pending | - | - | Requires NR11-NR13. |
| NR15 | Pending | - | - | Requires NR7 and NR11-NR14. |
| NR16 | Pending | - | - | Requires NR8-NR15. |
| NR17 | Pending | - | - | Requires NR7, NR15, and NR16. |
| NR18 | Pending | - | - | Requires NR0-NR17 completion and approval gates. |

## NR0 record

### Tests-only phase

- Assigned writer: `/root/nr0_writer`, `gpt-5.6-terra`, `xhigh`; the same
  writer continues through tests and implementation.
- Allowed files: exact NR0 allowlist in plan Section 4.1.
- Source-allowlist correction: added `src/neural_analysis/lfp_summary_plotting.py`
  on 2026-09-23 because Section 4.3 item 30 already requires corrected plot
  unit labels and the existing hard-coded labels cannot be changed from any
  originally allowlisted source. Scope is labels only, not plot numerics.
- First tests draft: blocked by the first Sol review because several failures
  were malformed or vacuous and production/cache/legacy routes were missing.
- Revised RED command/result: exact 14-file Section 4.4 command independently
  reproduced by the lead: 60 failed, 567 passed, 3 warnings in 34.54 seconds.
  There were no collection, import, or fixture failures.
- Second Sol test-design review: blocked. Required repairs are: prove semantic
  identity reaches and invalidates every Open Ephys keyed exploratory cache;
  test truthful exploratory labels and saved provenance plus SpikeGLX
  non-regression; require both configured and observed mismatch values; freeze
  the complete pre-NR0 source identity; cover preparation validity/RMS/peak-to-
  peak axes; and prevent absent semantics on SpikeGLX snapshots from being
  labeled Open Ephys legacy-unscaled.
- Revision-two RED command/result: exact 14-file Section 4.4 command
  independently reproduced by the lead: 71 failed, 571 passed, 3 warnings in
  35.74 seconds. No collection, import, or fixture failures occurred.
- Third Sol test-design review: blocked. Required repairs are: verify saved
  semantics/digest/unit provenance for relative-phase, phase-clustering, and
  spike-phase-locking as well as Hilbert; expose legacy semantics through the
  public receipt-validated cache UI and reject unvalidated fallback; cover
  literal dtype aliases and invalid sample/channel/count metadata; freeze one
  exact mismatch diagnostic across trial and continuous routes; remove the
  invalid assumption that unequal Python objects have unequal hashes; and
  assert cached trace/phase units plus the `uV^2/Hz` plotting contract.
- Revision-three RED command/result: exact 14-file Section 4.4 command
  independently reproduced by the lead: 101 failed, 578 passed, 3 warnings in
  36.50 seconds. No collection, import, or fixture failures occurred.
- Fourth Sol test-design review: blocked. One test incorrectly required linear
  `uV^2/Hz` on a normalized-dB PSD axis; plan item 30 now separates linear
  cache/schema units from normalized display units. Remaining repairs must
  enforce in-place affine operations, make the legacy stale fixture initially
  compatible except for semantics, freeze exact corrected component identity,
  vary only semantics in work-cache checks, exercise semantics-version changes
  through every exploratory cache, and complete malformed primitive/unit/layout
  validation plus exact sidecar-digest assertions.
- Revision-four RED command/result: exact 14-file Section 4.4 command
  independently reproduced by the lead: 114 failed, 582 passed, 3 warnings in
  36.85 seconds. No collection, import, or fixture failures occurred.
- Fifth Sol test-design review: blocked. Repair an impossible pre-edit digest
  assertion, allow `.astype(float, copy=True)` followed by in-place operators,
  resolve the production generic PSD schema with cached `site_voltage_units`
  rather than a test-local schema, use the real corrected component fingerprint
  for legacy staleness, and compare PPC null summaries plus stable cache axes.
- Revision-five RED command/result: exact 14-file Section 4.4 command
  independently reproduced by the lead: 115 failed, 580 passed, 3 warnings in
  38.40 seconds. The sixth review found two hidden GREEN defects: a tracking
  ndarray rejected its own final non-affine assertion, and a pure-sine
  presession reference made near-zero Welch bins numerically unstable.
- Revision-six RED command/result: exact command independently reproduced: 115
  failed, 580 passed, 3 warnings in 38.21 seconds. The seventh review found one
  arithmetic typo in the tracking fixture and approved all other contracts
  under an in-memory intended affine implementation.
- Final RED command/result: exact command independently reproduced after the
  surgical arithmetic fix: 115 failed, 580 passed, 3 warnings in 37.43 seconds.
  There were no collection, import, or fixture failures.
- Final Sol test-design review: fresh reviewer `/root/nr0_test_reviewer_8`,
  `gpt-5.6-sol`, `xhigh`: APPROVE. Scope was exactly 12 allowlisted tracked
  test files, staging was empty, and `git diff --check` passed.
- Tests-only commit: `e728cea` (`test: define NR0 Open Ephys scaling
  contract`).

### Implementation phase

- First GREEN command/result: exact 14-file Section 4.4 command independently
  passed 695 tests with 3 warnings in 37.28 seconds.
- First affected/full-suite result: complete `src/tests/neural_analysis` passed
  1,386 tests with 22 warnings in 180.02 seconds.
- First Sol implementation review: blocked despite GREEN. The source must build
  and reuse exact tokens at actual view boundaries, bind supplied tokens to the
  requested paths, save provenance from the computation token, stream sidecar
  SHA-256, own the semantics identifier once at the adapter boundary, correct
  all affected data contracts, and centralize trial/continuous metadata
  compatibility validation with saved-channel bounds.
- Second GREEN command/result: the lead independently ran the exact 14-file
  Section 4.4 command: 695 passed, 3 warnings in 38.14 seconds.
- Second affected-suite result: complete `src/tests/neural_analysis`: 1,386
  passed, 22 warnings in 159.78 seconds.
- Complete-repository result: collection still stops only at the frozen
  unrelated untracked `test_project_utils.py` import of unavailable
  `autograd`: 1 error and 1 warning in 2.00 seconds.
- Second fresh Sol implementation review: blocked. Saved exploratory semantics
  still came from the global constant rather than the exact compute token;
  multi-site production compatibility checks could occur after an earlier
  site's numerical/work-cache I/O; present-but-malformed saved semantics were
  coerced rather than rejected; and several new or modified function contracts
  remained incomplete.
- Third-candidate GREEN command/result: the lead independently ran the exact
  14-file Section 4.4 command: 695 passed, 3 warnings in 36.72 seconds; complete
  `src/tests/neural_analysis`: 1,386 passed, 22 warnings in 158.97 seconds.
- Third fresh Sol implementation review: blocked. A present but unknown saved
  semantics string was still accepted; an unusable aligned-sync path on a
  later production Open Ephys site could be rejected only after prior
  numerical or work-cache I/O; and the changed metadata/fingerprint public
  contracts remained incomplete. The first two missing cases now require a
  narrow reviewed tests-first addendum before source correction.
- Addendum RED command/result: the lead ran the three affected files: 7 failed,
  129 passed in 6.64 seconds. The failures were exactly three trial preflight
  order cases, three phase work-cache/preflight order cases, and one unknown
  present-semantics case; injected-seam and current/legacy controls passed.
- Addendum Sol test-design review: fresh reviewer
  `/root/nr0_addendum_test_reviewer_2`, `gpt-5.6-sol`, `xhigh`: APPROVE. The
  source diff remained frozen, error contracts and unreachable numerical/cache
  seams were explicit, and `git diff --check` passed.
- Addendum tests-only commit: `2b497e4` (`test: close NR0 fail-closed preflight
  gaps`).
- Final addendum GREEN: 136 passed in 6.95 seconds.
- Final GREEN command/result: exact 14-file Section 4.4 command: 704 passed,
  3 warnings in 37.19 seconds.
- Final affected/full-suite result: complete `src/tests/neural_analysis`:
  1,395 passed, 22 warnings in 159.73 seconds.
- Final complete-repository result: collection stopped only at the frozen
  unrelated untracked `test_project_utils.py` import of unavailable
  `autograd`: 1 error and 1 warning in 1.99 seconds.
- Final Sol implementation review: fresh reviewer `/root/nr0_impl_reviewer_4`,
  `gpt-5.6-sol`, `xhigh`: APPROVE with no P0-P3 findings. Scope was exactly
  the eight source files, staging was empty, and `git diff --check` passed.
- Implementation commit: `177a8d6` (`fix: scale Open Ephys LFP values from
  metadata`).
- Performance evidence: a synthetic 32-channel, 300,000-sample float32 binary
  compared commit `e728cea` with the current bounded 200,000-sample
  single-channel read over 25 warm repetitions. Median time changed from
  1.380 ms to 1.494 ms (1.083x); traced peak allocation changed from 1,613,183
  to 1,613,064 bytes (1.000x). The corrected first value was exactly 10.125 for
  stored 1.25, gain 2.5, and offset 7.0. Shape and 2,500 Hz rate were unchanged.
- Accepted residual risk: token construction and later source reads are not one
  atomic filesystem operation, so concurrent external source mutation could
  create a token/read TOCTOU mismatch. NR0's operating contract treats inputs
  as immutable during computation. Atomic snapshots or post-read identity
  revalidation are deferred unless a later package adds concurrent-mutation
  support.

## NR1-NR18 records

NR1 dry run was authorized by the user on 2026-09-23 with these exact paths:

- session: `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference`
- corrected cache: `<session>/processed/lfp_summary_cache_open_ephys_affine_uV_v1_2026-09-23T09-49-58Z`
- analysis run: `<session>/analysis_runs/ct026_nr1_open_ephys_affine_uV_v1_2026-09-23T09-49-58Z`
- report: `<run>/report`
- temporary profiling: `<run>/profiling_tmp`

All destinations were absent before the gate. The approved action could create
dry-run scripts/configuration/identity/preflight/log/summary evidence only
inside `<run>`. It could not load numerical LFP windows, compute components,
create or mutate the corrected cache, create report/profiling outputs, alter
legacy artifacts, submit cluster work, or push.

### NR1 metadata/path/resource dry run

Status: complete and approved on 2026-09-23. The exact command was:

```text
uv run /home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_nr1_open_ephys_affine_uV_v1_2026-09-23T09-49-58Z/dry_run.py
```

The final run directory contains `dry_run.py`, `configuration.json`,
`source_identity.json`, `preflight.json`, `cache_state.json`,
`resource_bounds.json`, `incident.json`, `run.log`, `run_summary.md`, and
`file_inventory.json`. There are no unlisted directories, numerical arrays,
plots, report artifacts, profiling artifacts, or temporary files. The corrected
cache, report, and profiling destinations were absent before and after. The
repository remained tracked-clean on `refactor` at
`181b82b796f2c68d7126267975951e84c25db291`; no stage, commit, push, or cluster
action occurred during the run.

Resolved evidence:

- Both Open Ephys sources are float32, time-major/channel-interleaved,
  384-channel, 9,334,742-sample, 2,500-Hz files. Each is 14,338,163,712 bytes;
  gain is `0.1949999928 uV/value`, offset is zero, physical units are uV, and
  the affine conversion declaration is present. Both sidecars have SHA-256
  `38f0f874988bbd71d25f66c82b4282146b738af640a4b3f1808045b681650706`.
- The trial table has 427 rows. Sites are PFC channel 5, HPC1 channel 222, and
  HPC2 channel 14; all three configured pairs and the -2/0/+2-second windows
  are preserved. Theta is 6-10 Hz; gamma is 30-80 Hz excluding 58-62 Hz.
- Every site records `open_ephys_affine_uV_v1`. Corrected fingerprints are
  Power `9e4df59438cebb9aaeab0b6dd10dfd3872d0c00e265f988b1df862a7f4578df4`,
  Synchrony
  `7b89286d914cd3a53ab649dfceff813fda312c85933b75c91772e936353ed909`,
  and 100-shuffle ProbeB Spike-phase
  `e31ebb2c6f708c0fd048489faf1696aef037b24be13b53946869b196eda901f9`.
- The metadata-only ProbeB population contains 273 ordered units and 383
  selected channels. Its stable-unit SHA-256 is
  `eb3a9cc6cacb9dbbc4fc4f82b195ebbc64b0870f89e47c1065ca1535e55902ff`.
- The preview records 100 shuffles, seed zero, eight workers, unit/shuffle/
  trial-edge blocks 8/25/64, checkpointing, prepared-phase caching, a 2-GiB
  per-worker cap, and a 12-GiB aggregate cap. The conservative dry-run estimate
  is 427 trials x 3 sites x 50 frequencies x 2,000 samples x 9 bytes =
  1,152,900,000 bytes. Exact PPC planning remains unavailable without
  unauthorized phase validity and spike geometry.
- Corrected-cache components are expected missing. Legacy Power, Synchrony,
  and Spike-phase are expected stale by explicitly labeled manifest-only
  fingerprint/source-identity comparison; the corrected pass did not open
  component NPZ files.

The first completed evidence attempt was rejected during fresh review. It used
`assess_component_status`, which materialized legacy component NPZ members
read-only and advanced their access times to approximately 07:22:48-49 EDT.
It did not change content or modification times. The final runner removes that
route, and `incident.json`, `preflight.json`, `run.log`, and `run_summary.md`
retain the incident rather than concealing it.

Final evidence SHA-256 values:

```text
cache_state.json      7a8cf35ff14f696d19bf53fe51d87ece8378cc50fae3f0582b2eb9fa8f4cb7ef
configuration.json    fc83c1707d952efd46eff7a301b4ff50a813cdd2b116bb583ab4f03ad9210c50
dry_run.py            98f37ea59a569a34fbbd911819307e252114cbb5266d67bbb6b117dfdc52e765
file_inventory.json   5f1838bd1c98cce67e0d80b49c59ef63bdf914de2cf326590d66f53de1ed0643
incident.json         e8efc5e7f201db553b63c500b3fa8fc2aac60851125e8d4bbd9b0f629520e134
preflight.json        c74065795a53431874ebd83c9cef1e9ba7a644da3ff8c506dc868fa5fb9bd320
resource_bounds.json  2e6bac0063b5613f3548cec8efae37f1ec4b9a2ad475919db2351802f193d4ef
run.log               17efbb3f15c1bd2c7a6b558e95411dea3bf7dad3c2f661c6fc1b50e959c2356b
run_summary.md        45d3080b051478f11db391e01e352dd6505029dad805dab186018dba33ee7e06
source_identity.json  dd18ce8845c914b6ae73a7c4d337167c24279ce912ae405fdd7d09dcf12c4b0d
```

Fresh reviewer `/root/nr1_dryrun_reviewer` (`gpt-5.6-sol`, `xhigh`) approved
the corrected evidence with no P0-P3 findings. The next ordered action is the
small representative-window affine check in plan Section 5.2 item 2. It is a
numerical read and remains unauthorized. NR2-NR18 are not started.
