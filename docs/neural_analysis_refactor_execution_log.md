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
| NR1 | Synchrony approved; NR1C-C correction planned | - | - | Revised Synchrony is user-approved. Push and first checkout update reached `c859afe`; NR1E stopped before evidence/transfer because valid retained prepared-phase work exposed an overbroad launcher root-existence guard. NR1C-C requires user approval before tests-first implementation. |
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

### NR1 Synchrony chart and cluster-preview plan revision

On 2026-09-23 the user withheld Synchrony visual approval after observing that
the PFC theta-whole ITPC circles lie below the vertical intervals. Inspection
established that the circles are observed plug-in ITPC estimates and the lines
are 2.5/97.5 percentile-bootstrap bounds, not medians/IQRs. The nonlinear
with-replacement resampling distribution is shifted upward in all nine saved
conditions; the legacy artifact has the same behavior. No code or artifact was
changed. Plan Section 5.5 now freezes the exact analysis/display contract,
interpretation, alternative future decisions, and required simulation tests.

The user also selected byte-preserving reuse of the corrected Power and
Synchrony components for an eventual cluster 100-shuffle ProbeB preview and
requires that job to run without Codex monitoring. A literal whole-cache copy
is insufficient because the manifest fingerprints resolved absolute source
paths. Plan Section 5.6 therefore requires an explicit safe launcher cache
target, exact source-equivalence hashing, byte-identical NPZ transfer,
cluster-bound manifest rebinding with retained producer provenance, atomic
publication, prerequisite validation, isolated new-run safety checks, and a
durable submit-and-disconnect Slurm handoff. This documentation update does not
authorize implementation, push, transfer, cache mutation, or submission.

The user then selected the presentation-only resolution for both ITPC and ISPC
band summaries because both use the same phase-clustering, trial-resampling,
time-frequency averaging, and percentile calculation after their respective
single-site versus relative-phase inputs. Plan Section 5.5 now freezes a
horizontal display with a prominent observed-estimate circle and a slightly
offset bootstrap box showing actual Q25/median/Q75 plus capped unchanged
2.5/97.5 percentile whiskers, exact trial counts, a two-part legend, and no
confidence/null/significance language. The bootstrap draws are not persisted
in the approved Synchrony cache, so the interior quantiles cannot be inferred
or produced by a cache-only rerender. The implementation must version the
Synchrony payload, preserve Power, and remain tests-first; a new Synchrony
artifact and report require separate real-data authorization and user visual
approval before cluster transfer. No implementation or artifact mutation
occurred in this documentation-only step; source work remains behind the
plan-approval and tests-first gate.

The subsequent documentation audit found that the generic Section 2.6 NR1 row
still described only a command/evidence runner and therefore did not allocate
the new tests/source work safely. The plan now defines five explicit sequential
subpackages: NR1P (Terra/xhigh presentation/schema writer with mandatory fresh
Sol/xhigh RED and GREEN reviews), NR1V (Terra/high real-data evidence runner
with fresh Sol/xhigh artifact review), NR1C-A (Terra/xhigh atomic relocation
writer), NR1C-B (separate Terra/xhigh launcher-target writer), and NR1E
(Terra/high external cluster evidence runner). The lead remains Sol/high and
alone edits documentation, stages, commits, communicates with the user, and
decides gates. NR1E separates read-only preflight, user-approved transfer, Sol
review, and separately approved single submission; the runner stops after the
job id and never monitors. This worker-specification update makes no source,
test, data, cache, cluster, push, or submission change.

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
small representative-window affine check in plan Section 5.2 item 2.

### NR1 small-window authorization

The user authorized this bounded numerical read on 2026-09-23. Evidence is
restricted to
`<session>/analysis_runs/ct026_nr1_small_window_open_ephys_affine_uV_v1_2026-09-23T12-04-46Z`.
For PFC channel 5, HPC1 channel 222, and HPC2 channel 14, the check reads only
three 1-second/2,500-sample windows at the start, midpoint, and final valid
second. It compares direct stored float32 values, the public loader's physical
uV output, and the authoritative per-channel affine formula and records bounded
timing/allocation evidence. Sync/spike arrays, scientific kernels, caches,
reports, profiling, legacy mutation, cluster work, and Git push remain out of
scope.

### NR1 small-window result

Status: complete and approved on 2026-09-23. The exact command was:

```text
uv run /home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_nr1_small_window_open_ephys_affine_uV_v1_2026-09-23T12-04-46Z/small_window.py
```

The runner performed exactly nine bounded public reads: three configured sites
times `[0,2500)`, centered `[4666121,4668621)`, and final
`[9332242,9334742)` windows. Raw arrays were `(2500,)` float32 stored values;
public outputs were `(2500,)` float64 uV at 2,500 Hz. Every selected channel
used gain `0.1949999928 uV/value` and zero offset. All nine outputs were exactly
equal to `raw.astype(float) * gain + offset`, including
`array_equal=True`, `allclose(rtol=0, atol=0)`, zero maximum absolute error, and
zero maximum relative error.

Scalar population statistics:

| Site | Window | Raw min | Raw max | Raw mean | Raw SD | uV min | uV max | uV mean | uV SD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PFC | start | -1883.9042 | 1481.1938 | 32.2544 | 680.2274 | -367.3613 | 288.8328 | 6.2896 | 132.6443 |
| PFC | midpoint | -1584.6124 | 1599.3354 | 21.9784 | 583.9948 | -308.9994 | 311.8704 | 4.2858 | 113.8790 |
| PFC | final | -1817.8446 | 1737.9828 | -30.7842 | 598.7260 | -354.4797 | 338.9066 | -6.0029 | 116.7516 |
| HPC1 | start | -1352.0294 | 1584.6024 | 52.8167 | 519.2606 | -263.6457 | 308.9975 | 10.2993 | 101.2558 |
| HPC1 | midpoint | -1497.8121 | 1746.2561 | 75.8701 | 640.4283 | -292.0734 | 340.5199 | 14.7947 | 124.8835 |
| HPC1 | final | -1258.1469 | 1400.8303 | -7.2143 | 473.6809 | -245.3386 | 273.1619 | -1.4068 | 92.3678 |
| HPC2 | start | -889.3183 | 795.0800 | -7.7922 | 324.0905 | -173.4171 | 155.0406 | -1.5195 | 63.1976 |
| HPC2 | midpoint | -1105.2030 | 924.4885 | 49.6060 | 421.5107 | -215.5146 | 180.2753 | 9.6732 | 82.1946 |
| HPC2 | final | -915.3886 | 1093.4213 | -3.3301 | 371.6038 | -178.5008 | 213.2171 | -0.6494 | 72.4627 |

Public per-window reads took 1,676,658-4,304,161 ns with
212,179-214,419 traced Python bytes. These are bounded call measurements, not a
full-component runtime or process-RSS claim. Exact output equality establishes
agreement with the same parsed sidecar gain and operation ordering; it does not
independently establish that the declared decimal calibration is scientifically
correct.

The evidence directory contains only `small_window.py`, `preflight.json`,
`source_identity.json`, `results.json`, `run.log`, `run_summary.md`, and
`file_inventory.json`. Arrays were retained only in memory. Corrected
cache/report/profiling targets remained absent; legacy directory stat identity
was unchanged; no legacy component NPZ, sync, spike, cache, component, kernel,
report, profiling, cluster, stage, commit, or push route ran. Repository state
remained tracked-clean at
`f0736f546c2ae7e93722770705edd204cdbee7d5`.

Final evidence SHA-256 values:

```text
file_inventory.json   028b010b9246e860e12724245a1213c851970abb72cb4821d423edec293b1667
preflight.json        1870bae4cb93a4f311d6ce2761056759f524d3ace7af6be8d2605c53b91b8421
results.json          b67c002a449912cbeaf352c89bbb0cfb3f31d92300fe1b6cbea3644d9deaf1a3
run.log               8112496653a56f03a7fd07aca937f0ffd91ad8dc9671bcfe3cff4fc65414170d
run_summary.md        60b010303b90cd611b92985081ceb272d89ae2ea7273bdf027d47ee619a13276
small_window.py       ea2c000da5bb762e7c4ffb44fd1acc380848388ed41446519a05de19fa5d5640
source_identity.json  22e212e535bbc28becaaf1ed5a880964ce56be893bbff9e3a50d078c85104d97
```

Fresh reviewer `/root/nr1_window_reviewer` (`gpt-5.6-sol`, `xhigh`) approved
with no P0-P3 findings. Corrected Power in plan Section 5.2 item 3 is the next
ordered action.

### NR1 corrected-Power authorization

The user authorized corrected Power on 2026-09-23. The only scientific output
location is the reserved corrected cache
`<session>/processed/lfp_summary_cache_open_ephys_affine_uV_v1_2026-09-23T09-49-58Z`;
it may receive only the Power component and its manifest. Runner, identity,
configuration, log, comparison, and Power-only report evidence is restricted to
`<session>/analysis_runs/ct026_nr1_power_open_ephys_affine_uV_v1_2026-09-23T12-23-53Z`.
The legacy cache may be loaded only for read-only Power comparison. The gate
must compare axes, trials, validity, references/traces, linear PSDs, band power,
normalized dB, reports/plots, runtime, memory, and cache provenance. Synchrony,
Spike-phase, legacy mutation, cluster work, and Git push remain prohibited.
At authorization, result and review were pending. The completed Power result
is recorded below. NR2-NR18 are not started.

#### First Synchrony attempt and retry boundary

The first production attempt under
`ct026_nr1_synchrony_open_ephys_affine_uV_v1_2026-09-23T13-17-26Z` did not
publish Synchrony. The command tool yielded after 31 seconds and then returned
empty output without exit metadata or a Python traceback. No process remained.
The run contains only `run_synchrony.py`, `configuration.json`, and
`preflight.json`; no report, log, postflight, or scientific artifact exists.
The corrected cache still contains only the approved manifest and byte-identical
Power component. No kernel OOM event was available, so SIGKILL/OOM is plausible
but unproven. Available RAM after the event was approximately 34 GB, while swap
was nearly exhausted. The historical approved Synchrony run took 642.487
seconds, so a 31-second silent return is not a successful computation.

The failed run is immutable evidence and must not be reused. The same unchanged
scientific Synchrony settings will be retried at
`<session>/analysis_runs/ct026_nr1_synchrony_open_ephys_affine_uV_v1_retry_2026-09-23T13-23-01Z`.
The retry adds complete Git/hash/source preconditions, durable progress and
exception evidence before compute, resource snapshots, and a terminal session
that is explicitly polled through completion. This is an operational retry
within the approved Synchrony scope; no scientific setting is reduced. At this
checkpoint, result and review remained pending; the completed result follows.

### NR1 corrected-Power result

Status: scientifically, operationally, and visually approved by the user.
Production command:

```text
uv run /home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_nr1_power_open_ephys_affine_uV_v1_2026-09-23T12-23-53Z/run_power.py
```

The corrected cache contains exactly two files:

```text
manifest.json  14,726 bytes      4218cb430f2a17f7bb4ea44248005d98e431c95f891ff9763e444b9c641b73dc
power.npz      78,680,638 bytes  164114d606cc204ff29ad173a686f0be66b54c2b944ed1f15008b2fa85367355
```

The component independently reloads as compatible, contains the expected 30
arrays, and records corrected fingerprint
`9e4df59438cebb9aaeab0b6dd10dfd3872d0c00e265f988b1df862a7f4578df4`,
both sidecars, and `open_ephys_affine_uV_v1` for all three sites. No corrected
Synchrony or Spike-phase component or work artifact exists.

All shapes/dtypes, axes, trial indices, nine condition memberships/counts,
objective/site/PSD validity, exclusions, reference availability, and finite
support match the legacy Power component. Source traces scale by the common
gain; linear PSD, references, and band power scale by gain squared. Normalized
band powers differ by at most `1.70e-13 dB`.

The initial fixed comparison (`rtol=1e-9`, `atol=1e-10`) stopped on only two
full-resolution arrays:

- `normalized_psd_session_db`: 3,316 cells; maximum `3.344045609310342e-7 dB`;
- `normalized_psd_presession_db`: 1,597 cells; maximum
  `3.345628485362795e-7 dB`.

The original `comparison.json` remains failed and unchanged. A read-only
diagnostic established identical finite masks, gain-squared proportional PSDs
and references, and legacy gain-cancellation reconstruction error no larger
than `1.4210854715202004e-14 dB`. The maxima occur at 1,250 Hz where PSD support
is approximately `1.014e-15`; below 100 Hz, maxima are `6.31e-12 dB` session
and `5.34e-12 dB` presession. Median errors are about `3e-14 dB`.

Fresh review required and then verified a separate a priori acceptance rule,
without rewriting the original result:

```text
B = (10 / ln(10)) * (abs(ln(P1/P0)) + abs(ln(R1/R0)))
    + 32 * eps64 * max(1, abs(D0), abs(D1))
fixed ceiling: atol = 1e-6 dB, rtol = 0
```

Observed/bound maxima were `3.3440456093e-7 / 3.3440486933e-7 dB` for session
and `3.3456284854e-7 / 3.3456320204e-7 dB` for presession, with zero pointwise
bound or fixed-ceiling violations. This identifies floating-point re-execution
before the ratio/log operation rather than a scientific or support change.

Power ran in 42.56690235005226 seconds with 591,646,720-byte peak RSS, versus
legacy 10.4942 seconds and 586,289,152 bytes: 4.056x wall time and +0.914% RSS.
The component size is exactly unchanged. A single cold/cache-sensitive run
cannot attribute the runtime difference to affine conversion; it remains
unexplained variance but is operationally acceptable at under one minute.

The report contains 12 validated PNGs with the expected names and dimensions,
427 trials, 424 valid trials per site, three `missing_choice_time` exclusions,
and no new warning. Cache and report hashes are recursively anchored in
`file_inventory.json`; corrected cache hashes are separately anchored in
`artifact_inventory.json`. Evidence chronology retains the initial hard stop,
the diagnostic, the separate acceptance, and packaging repair. Legacy content
hashes and mtimes are unchanged; repository state remained tracked-clean at
`d78db2429a2a06abab8850ca8cecfce2e6a0e22d`.

Fresh reviewer `/root/nr1_power_reviewer` (`gpt-5.6-sol`, `xhigh`) approved the
scientific, safety, and evidence gate with no P0-P3 findings. User visual
approval followed on 2026-09-23; the 12 Power PNGs are accepted.

### NR1 corrected-Synchrony authorization

The user authorized corrected Synchrony on 2026-09-23. The approved Power
component and manifest identity must be preserved while the existing corrected
cache receives only `synchrony.npz` and its atomic manifest update. Runner,
identity, configuration, log, comparison, and Synchrony-only report evidence is
restricted to
`<session>/analysis_runs/ct026_nr1_synchrony_open_ephys_affine_uV_v1_2026-09-23T13-17-26Z`.
The legacy cache is read-only. The gate compares phase-derived arrays,
bootstrap schedules/results, validity/support, source and filtered traces,
exemplars, report metadata/plots, runtime, and memory. Spike-phase, approved
Power mutation, legacy mutation, cluster work, and Git push remain prohibited.
At authorization, result and review were pending. The completed result follows;
NR2-NR18 are not started.

### NR1 corrected-Synchrony result

Status: scientifically and operationally approved after fresh Sol numerical,
provenance, and systematic visual reviews; explicit user visual approval is
pending. The first run directory remains immutable failed operational evidence.
The successful retry and all evidence are at:

```text
/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_nr1_synchrony_open_ephys_affine_uV_v1_retry_2026-09-23T13-23-01Z
```

Production command:

```text
uv run /home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_nr1_synchrony_open_ephys_affine_uV_v1_retry_2026-09-23T13-23-01Z/run_synchrony_retry.py
```

The corrected cache contains exactly three files and no Spike-phase or work
artifact:

```text
manifest.json   25,489 bytes      f0301f3006be225e4c55142e68a3024ea06ddd8bb90542a6704d11e703a57ba3
power.npz       78,680,638 bytes  164114d606cc204ff29ad173a686f0be66b54c2b944ed1f15008b2fa85367355
synchrony.npz  153,572,688 bytes  fc4d742ec42dfd00281e8759b15545c4e9488a34e85c38cd26a6e30469145983
```

The production calculation and report completed, then the predeclared strict
comparison stopped on nine arrays. Its original `comparison.json`,
`failure.json`, and failed `postflight.json` remain unchanged. All 38 arrays
have identical schema, shape, dtype, and finite masks; 29 passed the original
rules, including every exact identity, mask, count, axis, validity, and support
field. `source_trace` follows the strict gain rule. `band_filtered_trace`
differs from `float32(legacy * gain)` by at most one float32 ULP. The largest
Hilbert phase difference is `2.9802322388e-8 rad`; ITPC and ISPC differ by at
most `1.8747214725e-7` and `2.1686773694e-7`; frequency-resolved PLV differs by
at most `1.5274714604e-8`.

Fresh review approved a separate fixed policy without changing the failed
comparison: exact fields remain exact; source trace retains the original
`atol=1e-10, rtol=1e-9`; filtered trace uses `atol=0,
rtol=2*float32_eps`; Hilbert phase uses wrapped `atol=1e-6 rad`; all
dimensionless and bootstrap summaries use `atol=1e-6, rtol=0`; ISPC and PLV
complex coherence vectors use `atol=1e-6, rtol=0` plus a raw wrapped-angle
ceiling of `1e-3 rad`. All checks pass. The maximum ISPC angle change,
`1.0875650435e-4 rad` (`0.00623 degrees`), occurs at resultant magnitude
`9.10e-5`; maximum ISPC complex-vector error is `2.2379020101e-7`. The maximum
PLV angle change is `1.7986499481e-6 rad` at PLV `0.00204`; maximum PLV
complex-vector error is `1.5280284020e-8`. This is the expected conditioning
of angles near zero resultant magnitude, not a validity or phase reversal.

Thirteen of 108 PLV exemplar selections switch. Every switch is between the
same two adjacent order statistics at an exact 25th/75th-percentile `.5` rank;
legacy and corrected distances from the interpolated target differ by no more
than `1.11e-16`. Selection pools, rank order, validity, and the other 95
exemplars are unchanged. This satisfies the separately reviewed exact
midpoint-bracket rule while preserving the changed trial identities for visual
review.

Synchrony ran in `728.9850418729475 s` with `4,286,124,032` bytes peak RSS,
versus legacy `642.487 s` and unavailable peak memory. The +13.46% wall time is
acceptable for this cold single-run gate; RSS is below the configured 12-GiB
aggregate limit. Component size is unchanged. The report has the same 252 PNG
filenames, trial/exclusion counts, zero unstable summaries, and no warnings.
Thirty-four image-dimension differences are exemplar-only tight-layout changes
of 3-8 vertical pixels. A fresh reviewer systematically inspected all 252
figures and approved with no P0-P3 findings; this does not replace user visual
approval.

Acceptance packaging preserves the original hard stop and uses separate
`diagnostic.json` and `acceptance.json`. A first packaging invocation failed
before atomic diagnostic output on NumPy-bool JSON serialization. The script
was edited in place, one fixed revision packaged the acceptance, and a later
incident/log revision refreshed it. The pre-fix and intermediate bytes/hashes
were not preserved. `packaging_chronology.json` discloses this irrecoverable
ancillary limitation and maps all three invocations; final revision B is
preserved at SHA-256
`df3e58eb2891ce4b1ec558139f5473374740b059f0a6f3f76fe153f1d8193dd7`.
The exact production runner and scientific artifacts are preserved. A
dedicated provenance-only closer refreshed the 276-file recursive inventory
without invoking the packager again. Fresh re-review accepted the disclosure
and full package with no P0-P3 finding. Legacy hashes and mtimes, approved Power
bytes, report bytes, repository HEAD `19f9ef866ae9381db22836cfcb8fb6bbc46d466c`,
and tracked-clean state remained unchanged.

### NR1 Synchrony user-review finding and Spike-preview execution decision

The user withheld Synchrony approval after noticing that every observed point
in `PFC_theta_whole_itpc_band_summary.png` lies below its vertical interval.
The lines are saved percentile-bootstrap 95% intervals, not IQRs, and the
circles are observed nonlinear ITPC estimates rather than bootstrap medians.
The saved `(estimate, low, high)` triples are:

```text
correct_rewarded  (0.093031498, 0.101026567, 0.141250198)
omission          (0.138474499, 0.160156721, 0.233108471)
incorrect         (0.063974960, 0.073807686, 0.097182355)
switch            (0.112607792, 0.132281939, 0.179805459)
stay              (0.064153998, 0.075357745, 0.100829593)
omission_switch   (0.290252871, 0.320827530, 0.488008435)
omission_stay     (0.179708280, 0.205423319, 0.288580083)
incorrect_switch  (0.131302948, 0.147979124, 0.206079952)
incorrect_stay    (0.066730630, 0.080087825, 0.106781842)
```

The implementation recomputes the nonnegative vector-magnitude statistic for
each with-replacement trial resample before taking percentile bounds. Duplicate
trials reduce effective directional diversity and produce upward finite-sample
bias, so a percentile interval need not contain the original estimate. The
plotting test explicitly freezes that behavior, and the legacy figure has the
same geometry. This explains the result but does not settle whether its
statistical method or presentation should change. No scientific or plot code
was changed, and Synchrony user approval remains pending.

The user requires the eventual 100-shuffle ProbeB preview to run on Slurm
without Codex monitoring. The existing planned execution path is the standalone
launcher through `src/shell_scripts/hpc_ppc.sh`: one task, eight CPUs/workers,
32 GiB, 72 hours, `--shuffles 100`, no `--final-run`, timestamped state/log,
exact resume command, persistent checkpoints, and terminal report validation.
Codex may inspect persistent state later when asked but must not poll the job.

Submission is neither safe nor authorized yet. The launcher exposes session,
analysis-root, probe, shuffle, and worker arguments but no cache-output
argument; its CT026 builder targets `processed/lfp_summary_cache`, the protected
legacy cache. The branch is also 25 commits ahead of `origin/refactor` at local
HEAD `058665443b54550e7ca585c012d212f0dff55266`, so cluster execution requires a
separately approved push and exact clean checkout. Before proposing submission,
add and test an explicit corrected-cache target and choose a reviewed way to
establish the approved corrected Power/Synchrony state on the cluster (verified
copy or separately authorized reproduction). No push, transfer, or Slurm
submission occurred.

### NR1P Synchrony presentation implementation

Status: complete and approved in code; no real-data artifact was created or
changed. The next step is the separately authorized NR1V versioned Synchrony
run and user visual review.

The user selected a presentation-only revision for both ITPC and ISPC. The
observed phase-clustering estimate, trial selection, validity masks, seed,
1,000 with-replacement resamples, time-frequency averaging, and existing
2.5th/97.5th percentile endpoints remain unchanged. The implementation now
computes Q25, median, and Q75 from the same finite scalar bootstrap draws using
one explicit linear-percentile operation, persists those six small metric
arrays plus the two exact per-band/epoch selected-trial-count arrays, and does
not persist the bootstrap draws.

The shared renderer now uses horizontal condition rows. Each row labels the
condition and exact saved trial count, shows the observed estimate as a filled
circle, and offsets an unnotched Q25-Q75 box with its actual median and capped
2.5th/97.5th percentile whiskers. The figure legend distinguishes the observed
estimate from the bootstrap resampling distribution and uses no confidence,
null, or significance claim. Report and webapp paths consume only the saved
cache arrays. A code-owned Synchrony-only identity key
`synchrony_payload_contract_version` has value
`synchrony-bootstrap-quantiles-counts-v1`; Power and Spike-phase identities are
unchanged. The new expected Synchrony fingerprints are
`4f2754235f64490fca168107d5e34e005bcd3f0e50cd9b000721f2c79bd4b706`
for the frozen SpikeGLX configuration and
`18eeb15ac4450408bd676bdcdf3450f7afb9f95fd31fc66cb6a1f6d8fc7b77a5`
for the frozen one-site corrected Open Ephys unit-test configuration. These are
fixture-specific identities, not universal format identities. The later full
live CT026 configuration correctly resolves to
`12e5f78a347e7cc2a210172dd08bfc59c9152a79e58464cd5ba6fe5d80049e29`.

Tests were written and committed before source implementation. The initial RED
package was committed as `d5d9e2b`. Fresh Sol/xhigh test-design review found
gaps in exact counts, fingerprint identity, payload validation, NaN handling,
artist semantics, caller wiring, and legacy-receipt isolation; the same
Terra/xhigh writer corrected them before approval. During implementation
review, tests-first corrections added fail-closed count/quantile validation and
the condition-only webapp label (`5ae2040`), exact unsigned-count preservation
(`c0129fe`), and a one-line wrapping-only harness correction (`0b8e2d7`). Each
behavioral correction reproduced genuine RED before its source fix.

The reviewed source implementation is commit `ad8e598`. Final lead and worker
verification was:

```text
uv run pytest -q -p no:cacheprovider <nine NR1P files>
269 passed, 2 warnings in 19.75s

uv run pytest -q -p no:cacheprovider src/tests/neural_analysis
1415 passed, 22 warnings in 166.76s

uv run python -m py_compile <seven NR1P source modules>
passed
```

The warnings are the known PPC all-NaN-slice, multiprocessing fork, duplicate
ZIP-entry fixture, and Pynapple empty-epoch warnings. `git diff --check` passed,
and each test/source commit stayed within its Section 5.5 allowlist. No CT026
array, existing cache, report, external analysis directory, cluster checkout,
or Slurm state was accessed or mutated during NR1P. No dependency or
configuration file changed.

Orchestration followed the frozen division: lead Sol/high; one Terra/xhigh
writer for tests and source; a fresh Sol/xhigh read-only test-design reviewer;
and a different fresh Sol/xhigh read-only implementation reviewer. Reviewer
write restrictions were procedural; repository state checks confirmed that
they made no edits. The lead alone staged and committed reviewed paths. All
NR1P agents are complete. Git push remains a separate user authorization gate.

### NR1C-A corrected-cache relocation implementation

Status: complete and approved in code on 2026-09-23. No CT026 path, cluster,
network transfer, scientific kernel, cache artifact, or external run directory
was accessed or changed. Git push remains separately gated.

The initial tests-first contracts were committed as `e940671` and hardened as
`72028ad`. They freeze the pure manifest-rebinding API, bounded ZIP/NPY header
validation, backward-compatible header-only public component status, exact
source/configuration equivalence, nonsymlink containment, streamed hashing,
byte-preserving Power/Synchrony transfer, external ASCII receipt, executable
CLI, atomic sibling publication, and source/legacy preservation. Focused RED
was genuine: the existing IO tests passed and every new failure was an absent
planned interface or relocation module.

Implementation review exposed three concurrency/provenance boundaries, each
handled tests-first before its source correction:

- `6e243c3` requires receipt-failure rollback to preserve a destination whose
  same-name component was concurrently replaced;
- `f10b7bd` requires the embedded source-producer manifest and its SHA-256 to
  describe one coherent bounded byte snapshot; and
- `b994495` covers invalid and schema-valid replacements in the interval
  immediately after atomic publication, requiring fail-closed behavior,
  preservation of the changed destination, and no success receipt.

The reviewed implementation is `e1d99c3`. `lfp_summary_io.py` now provides the
pure rebinding helper and bounded header-only NPZ validation without loading
numerical arrays. `lfp_summary_cache_relocation.py` provides the immutable
request/runtime/result contracts, strict root/path/source equivalence checks,
streamed byte copies and hashes, destination-bound manifest construction,
prepublication receipt preparation, atomic cache publication, public
header-only validation, and receipt commit. Producer-manifest content and
digest are derived from one bounded ASCII read. Cache ownership is captured
from staging before rename using directory device/inode plus exact identities
for `manifest.json`, `power.npz`, and `synchrony.npz`; it is checked immediately
after publication and immediately before receipt commit. Failure rollback
removes only that unchanged publication and preserves the whole destination if
any concurrent actor changed it.

Final verification was:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_summary_io.py \
  src/tests/neural_analysis/test_lfp_summary_cache_relocation.py
107 passed in 2.31s

uv run pytest -q -p no:cacheprovider src/tests/neural_analysis
1503 passed, 22 warnings in 169.10s

uv run python -m py_compile \
  src/neural_analysis/lfp_summary_io.py \
  src/neural_analysis/lfp_summary_cache_relocation.py
passed
```

`git diff --check` passed. Source scope was exactly the two Section 5.6 NR1C-A
files; tests were committed separately before implementation. Fresh Sol/xhigh
test-design and implementation review approved the final state with no P0-P3
findings. NR1C-B launcher-target/preflight work is now unblocked, but cluster
transfer, push, and Slurm submission remain unauthorized.

### NR1C-B explicit launcher cache target

Status: complete and approved in code on 2026-09-23. No CT026 path, cache,
scientific kernel, cluster, network transfer, or Slurm state was accessed or
changed. The existing shell wrapper was read and exercised only through a
temporary synthetic Git repository.

The tests-first launcher contract was committed as `0d056fa`. It requires an
explicit new-run cache target, full durable identity propagation through every
resume/recovery path, direct-child/nonsymlink containment, protected-legacy and
alias rejection, exact prerequisite inventory, public header-only component
status, absent Spike phase and derived work root, builder immutability, dry-run
nonmutation, and exact eight-worker preview argument forwarding through the
unchanged Slurm wrapper. Initial RED was 45 failures and five intentional
existing boundary passes. Fresh Sol/xhigh test-design review approved the
contract after adding arbitrary nondefault-cache recovery, no-array-load,
corrupt-header, nested/symlink, exact-inventory, and dynamic-wrapper coverage.

Once the parser accepted the new argument, two dangling-symlink fixtures and an
older second-new-run subcase surfaced as genuine test isolation errors. The
reviewed correction `4646b5d` creates each real backing cache before its alias
and isolates the older recovery subcase from the first run's intentionally
retained work root. It does not weaken launcher safety.

The reviewed implementation is `c844adc`. `new` requires
`--cache-directory`, validates it as an existing nonsymlink direct child of the
selected session's `processed` directory, and immutably replaces only
`LFPSummaryConfig.output_directory`. Before trial loading or run-directory
creation it rejects the protected legacy cache, aliases/nesting/outside paths,
unexpected or unsafe members, non-compatible Power/Synchrony, a present Spike
component, and the exact derived PPC work root. Prerequisite NPZ validation
uses `assess_component_status(..., validate_headers_only=True)` and never loads
numerical arrays. Configuration, identity, paths, preflight, state, summary,
ordinary resume, report recovery, and rerender recover the exact saved target;
non-new CLI modes expose no replacement option. `src/shell_scripts/hpc_ppc.sh`
was not changed.

Final verification was:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_spike_phase_launcher.py
50 passed in 1.75s

uv run pytest -q -p no:cacheprovider src/tests/neural_analysis
1530 passed, 22 warnings in 169.42s

uv run python -m py_compile \
  src/neural_analysis/lfp_spike_phase_launcher.py
passed
```

`git diff --check` passed. Final implementation scope was exactly
`src/neural_analysis/lfp_spike_phase_launcher.py`; its tests were committed
separately before source. Fresh Sol/xhigh implementation review reported no
P0-P3 findings. NR1V real-data presentation evidence, Git push, cluster
relocation, dry-run evidence, and Slurm submission remain separate user gates.

### NR1V versioned Synchrony artifact and presentation correction

Status: the scientific cache is complete and approved; the first immutable
report is preserved but rejected for observed-marker/box overlap. A reviewed
presentation-only source correction is committed, and a new immutable
cache-only rerender remains separately user-gated.

The user authorized these exact local destinations:

- cache:
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/processed/lfp_summary_cache_open_ephys_affine_uV_v1_synchrony_bootstrap_quantiles_counts_v1_2026-09-23T22-09-50Z`;
- run:
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_nr1v_synchrony_bootstrap_quantiles_counts_v1_2026-09-23T22-09-50Z`;
- report: `<run>/report`.

Preparation stopped before invocation when the draft runner incorrectly used
the frozen one-site Open Ephys unit-test fingerprint `18eeb15...` as a live
CT026 identity. Read-only review independently reconstructed old CT026
fingerprint `7b89286d914cd3a53ab649dfceff813fda312c85933b75c91772e936353ed909`
and new fingerprint
`12e5f78a347e7cc2a210172dd08bfc59c9152a79e58464cd5ba6fe5d80049e29`,
with only `synchrony_payload_contract_version` added. Static review also fixed
the approved source-report leaf, pinned source-manifest hash/schema, removed a
preparatory bytecode file, and made workflow metadata non-transient before the
first invocation. `preparation_incident.json` preserves the draft hashes and
corrections. Corrected runner SHA-256 is
`c24a99bbacfe90c44c210a4136180e6c6e5b02082d6884f77f2bae3ba7249531`.
A fresh Sol/xhigh static reviewer approved with no P0-P3 findings.

The sole production invocation at pushed tracked-clean HEAD `3a50de3` completed
with exit code zero in 704.424 seconds and 4,288,274,432-byte peak RSS. Power
remained byte-identical at SHA-256
`164114d606cc204ff29ad173a686f0be66b54c2b944ed1f15008b2fa85367355`;
new Synchrony SHA-256 is
`d9673f2183dcbb2eac7d8ec2be80916840815c727f1431b8fdf65b41d95cb86f`.
The cache has exactly manifest, Power, and Synchrony members. Public component
status passed; cache and report manifest snapshots agree and contain no pending
metadata. All 38 prior Synchrony arrays, including NaN locations, endpoints,
validity, phase, estimates, and exemplars, are exactly equal. Exactly eight
quantile/count arrays were added; quantiles are finite and ordered, counts are
nonnegative `int32`, and instability is exactly `count < 10`. Source cache/run
inventories remained unchanged. No Spike phase, PPC, cluster action, retry, or
repository mutation occurred.

The first report preserved all 252 filenames; 216 non-band PNGs are
byte-identical to the approved source report. All 36 revised band summaries
were inspected. Counts, order, actual medians/quartiles/endpoints, caps,
legends, wording, and clipping passed. Fresh review nevertheless found the
filled observed-circle footprint overprinted the bootstrap box in 12 of 18
ISPC summaries when the observed estimate lay inside Q25-Q75 (82 of 162 ISPC
cells). The scientific cache requires no recomputation, but the first report is
not visually approved and must remain immutable.

The correction followed tests-first review. Commit `88d7826` adds a
post-render display-coordinate regression including marker/box strokes across
all nine rows. Commit `389e4fd` changes only presentation geometry: box center
offset `+0.30` rows and box height `0.20` rows, retaining the exact `s=64`
filled observed circles and every scientific/presentation meaning. Fresh
Sol/xhigh review found no P0-P3 issue and independently verified positive,
DPI-invariant clearance for 1, 2, 9, and 20 rows plus PDF/SVG export. Final
gates were:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_summary_plotting.py
29 passed, 2 warnings

uv run pytest -q -p no:cacheprovider <nine NR1P files>
279 passed, 2 warnings

uv run pytest -q -p no:cacheprovider src/tests/neural_analysis
1531 passed, 22 warnings in 171.22s
```

The user then authorized exactly one immutable cache-only rerender at:

```text
/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_nr1v_synchrony_presentation_rerender_2026-09-23T23-30-02Z
```

Fresh Sol/xhigh static review approved runner SHA-256
`678b2b110ca897cc1d696159b78cf53269a26d374a5289f31006f057315aa273`
before invocation. The runner called the production cache-only renderer once
at exact tracked-clean HEAD `f8ecbf0fab09c0023160727df58c859e6aea5299`.
It completed with exit code zero in 74.825914 seconds with 685,510,656-byte
process peak RSS. No retry, raw-data loader, scientific computation, cache
write, work/PPC write, cluster action, or prior-report overwrite occurred.

The accepted cache retained exactly three members and unchanged SHA-256s:

```text
manifest.json  7967c14b77adf9b0041d6bd32593a0d06df3615a9289cd025d7289d927bf9986
power.npz      164114d606cc204ff29ad173a686f0be66b54c2b944ed1f15008b2fa85367355
synchrony.npz  d9673f2183dcbb2eac7d8ec2be80916840815c727f1431b8fdf65b41d95cb86f
```

The sole new report leaf is:

```text
<run>/report/CT026_2026-08-01_130853_lfp_synchrony_validation_2026-09-23T23-30-02Z
```

It contains exactly 257 direct files: 252 PNGs plus
`configuration.json`, `manifest.json`, `run.log`, `run_summary.md`, and
`source_identifiers.json`. All 216 non-band PNGs are byte-identical to both
preserved reports. Exactly all 36 band summaries changed from the rejected
geometry. Fresh Sol/xhigh artifact review inspected all 18 ITPC and 18 ISPC
summaries and reported no P0-P3 finding. Its independent pixel audit covered
all 324 condition rows, found zero marker/box overlaps, and measured at least
seven clear pixel rows in every row. The accepted cache, both prior reports,
and Git identity remained unchanged.

The runner's terminal status was
`rendered_once_pending_fresh_review_and_user_visual_approval`. Fresh review
then completed, and the user explicitly visually approved the revised report
on 2026-09-23 with "Looks good, keep going." The NR1V cache is now the sole
Section 5.6 transfer source and must not be recomputed. NR1E cannot begin until
the user separately authorizes the required Git push; transfer, cluster
checkout mutation, dry run, Spike-phase execution, and Slurm submission remain
unapproved.

### NR1E pushed-checkout preflight and NR1C-C blocker

The user separately authorized the required Git push. Exactly the five reviewed
commits from `88d7826` through `c859afe` were pushed with:

```text
git push origin refactor
```

Local `HEAD` and `origin/refactor` then both resolved to exact
`c859afe7d08235e4454fa15858ed8e02f6ce6feb`, with tracked files clean and all
pre-existing user-owned untracked files unchanged.

The user next separately authorized the cluster checkout update and read-only
preflight. The cluster checkout at
`/gs/gsfs0/home/mchin1/context-inference` was tracked-clean on `refactor` at
`02b0f2c`. A safe fetch required fetched `origin/refactor` to equal exact
`c859afe`; a fast-forward then succeeded. Pre-existing untracked Python cache
and egg-info entries were preserved. No reset, clean, or unrelated mutation
occurred.

Preflight stopped before creating its proposed evidence directory
`analysis_runs/ct026_nr1e_cluster_checkout_preflight_2026-09-24T02-01-13Z`.
The planned corrected-cache destination was absent, but the launcher derives
the shared work path `processed/lfp_summary_work` and rejects its mere
existence. Read-only inspection found a valid historical post-success state:

- one empty real `ppc/` directory;
- one complete fingerprint-named prepared-phase representation with exactly
  `metadata.json`, `complete.json`, `axes.npz`, `valid.npy`, and `phase.npy`,
  totaling 1,153,088,565 bytes;
- no PPC run child, lock, symlink, or unexpected member; and
- a completed historical 1,000-shuffle launcher record showing successful
  cleanup of its exact PPC run directory.

No numerical `.npy` or `.npz` payload was opened. No evidence directory,
corrected cache, work, report, transfer, launcher dry run, or Slurm artifact was
created. Fresh Sol/xhigh diagnosis classified the launcher check as an
overbroad P1: successful execution intentionally retains fingerprinted
prepared phase while deleting the exact PPC run. Rejecting the container itself
therefore permanently prevents any later `new` invocation for the session and
contradicts the plan's active-PPC protection.

The recommended NR1C-C correction is tests-first and launcher-only. It permits
complete lock-free prepared representations plus an absent or empty real
`ppc/` container, rejects any PPC child or unsafe/malformed work structure, and
never opens numerical work payloads. Historical work must not be deleted,
archived, moved, or rewritten. NR1C-C implementation, its later push/checkout,
cache relocation, launcher dry run, and Slurm submission remain separately
gated.
