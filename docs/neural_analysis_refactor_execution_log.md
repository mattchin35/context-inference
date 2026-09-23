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
| NR0 | Tests-only revision | - | - | Second Sol review blocked; revise cache, legacy, label, and preparation contracts, then repeat RED/review. |
| NR1 | Pending | - | - | Requires accepted NR0 and separate real-data gates. |
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
- Final GREEN command/result: pending after source-only corrections.
- Final affected/full-suite results: pending.
- Final Sol implementation review: pending.
- Implementation commit: pending.
- Performance evidence: pending.
- Unresolved risks: none recorded yet.

## NR1-NR18 records

Not started. Add exact commands, results, commits, review findings, performance
evidence, authorization state, and next gate before each package handoff.
