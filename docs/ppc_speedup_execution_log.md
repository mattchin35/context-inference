# Exact PPC speedup execution log

This file is the persistent execution record required by
`docs/ppc_speedup_plan.md`. It records implementation authority, package gates,
agent assignments, commits, verification, and benchmark authority without
changing the scientific specification.

## Baseline

- Plan: `docs/ppc_speedup_plan.md`, audited 2026-09-10.
- Pre-documentation scientific implementation HEAD: `31a5c54` on `refactor`.
- Documentation-baseline commit: `73e2d9f` (`docs: finalize exact PPC speedup
  handoff`).
- Plan clarification approved 2026-09-10: strict shared PPC/histogram
  exact-sample semantics; named conservative private-peak memory accounting;
  actual derived schedule identity; context-managed S0 normal shutdown; and a
  Sol-`high` lead with `xhigh` reserved for S1-S4/S7 writers and reviewers. The
  clarification also freezes `KernelAllocationEstimate`, adds a 12 GiB planned
  aggregate-array preflight, and requires new grouped-worker RED evidence in S7
  rather than reusing S0's single-job regressions. Clarification commit:
  `20202a4` (`docs: clarify PPC speedup execution plan`).
- Expected S0 RED checkpoint: 2 failed and 12 passed in
  `src/tests/neural_analysis/test_lfp_summary_ppc_parallel.py`.
- Implementation authorization: on 2026-09-10 the user explicitly authorized
  the S0-S7 source/test implementation sequence and its bounded Sol/Terra
  delegation after confirming the lead Sol is set to `high`. Temporary usage-
  window interruptions must follow the plan's recovery contract and do not
  waive any RED/GREEN or review gate.
- CT026 authorization: no new CT026 computation is authorized. Metadata-only
  inspection and every later work-only or scientific run remain subject to the
  gates in the plan.

Unrelated pre-existing worktree changes were present when this baseline was
prepared. They are outside the PPC plan, are not included in its documentation
commits, and must remain untouched.

## Package records

Fill each record before its package-record commit. Do not delete fields that do
not apply; write `not applicable` and explain why.

### S0 - Restore worker checkpoint to GREEN

- Status: complete.
- Authorization: user authorization recorded 2026-09-10; no CT026 work or
  production worker benchmark authorized.
- Starting HEAD and worktree inventory: `bd735acfbba8808ffd50dd9ad660d34d8b5888ef`;
  complete `git status --short -z` inventory contained 172 entries with SHA-256
  `58b0b546eb35ee92598775b40b93bf4984619153eafc8d07b9a353187441fc47`.
  The only tracked pre-existing modification was `src/main.py`; all remaining
  entries were unrelated untracked user files/directories. The lead will
  compare the live inventory before accepting the worker diff.
- Lead Sol model/effort: `gpt-5.6-sol`, `high`, confirmed by the user; write
  authority restricted to review, staging, commits, and this execution log.
- Terra worker model/effort, permissions, and file scope: `gpt-5.6-terra`,
  `high`, write-enabled only for
  `src/neural_analysis/lfp_summary_ppc_runtime.py`; the existing parallel test
  file is read-only.
- Optional Terra scout model/effort, permissions, and findings: not applicable;
  no scout was needed.
- Sol reviewer model/effort, permissions, and disposition: `gpt-5.6-sol`,
  `high`, procedurally read-only; PASS with no blocking findings after an
  independent stable-diff audit.
- Read-only enforcement or procedural restriction: agent prompts provide
  procedural file restrictions because spawned agents share the worktree.
- Test-only commit and RED evidence: existing commit `9fe7f1f`; baseline 2
  failed and 12 passed.
- Implementation commit and GREEN evidence: `e28d8b5` (`fix: complete PPC
  worker shutdown contract`). Lead-focused verification passed 14 parallel
  tests; the combined Spike-PPC/runtime/work-cache/pipeline set passed 121.
- Full neural-suite evidence: 750 passed with 19 known warnings (16 Pynapple
  warnings and three multiprocessing-fork deprecation warnings).
- Numerical/interface review: the diff changed only executor lifecycle in
  `_run_parallel_worker_batches`; spawn context, bounded canonical submission,
  prior-yield resumability, public interfaces, numerical code, schedules,
  checkpoint schema, and final publication remained unchanged. The unrelated
  worktree fingerprint remained exactly
  `58b0b546eb35ee92598775b40b93bf4984619153eafc8d07b9a353187441fc47`.
- Unresolved risks: no blocking risk. Early consumer-driven generator closure
  still uses the context manager's default wait behavior, and exceptional
  cleanup performs an idempotent second shutdown during context exit; both are
  outside the narrow S0 result-failure contract and were accepted by the Sol
  reviewer.
- CT026 work performed: none permitted.

### S1 - Uniform-grid geometry and segmented edge kernel

- Status: complete.
- Authorization and starting HEAD: user authorization recorded 2026-09-10;
  `98e6ef6f278b77e861e58129e3d6fef6f1aef2d9`. The unrelated worktree inventory
  remained 172 entries with SHA-256
  `58b0b546eb35ee92598775b40b93bf4984619153eafc8d07b9a353187441fc47`.
- Agent assignments, permissions, and file scopes: lead `gpt-5.6-sol` `high`;
  writer `gpt-5.6-terra` `xhigh`, initially write-enabled only for
  `src/tests/neural_analysis/test_lfp_summary_ppc_kernel.py` and, if required,
  focused additions to `test_spike_lfp_summary.py`; independent reviewer
  `gpt-5.6-sol` `xhigh`, procedurally read-only. No scout assigned.
- Test-design review, test-only commit, and RED evidence: independent Sol
  `xhigh` gate PASS after corrections for interpolation weights, adjacent-source
  validity, empty groups, stable identities, ownership, and isolated checked
  arithmetic. Lead reproduced the intended collection RED (`1 error`, `29
  deselected`) because `lfp_summary_ppc_kernel` did not yet exist. Test-only
  commit `9705200` (`test: define optimized PPC kernel contracts`).
- Post-RED test corrections: corrected the half-open-boundary offset fixture and
  replaced NumPy 2.4.3's unsupported array-valued `assert_allclose` tolerance
  with an equivalent explicit per-cell bound in test-only commit `585c1e7`.
  Added reviewed RED coverage for element-level immutability and signed-int64
  identity range failures in test-only commit `513decf`.
- Allocation clarification: independent source review found that the initial
  formulas omitted retained NumPy identity arrays. The plan now charges 8 bytes
  per source geometry and 16 bytes per edge. Clarification commit `8aaff3c` and
  matching RED test-only commit `254629a` preceded the estimator change.
- Implementation commit and GREEN evidence: `1088dac` (`feat: add segmented
  PPC kernel`). Complete kernel plus edge-oracle suites: `78 passed`; directly
  affected parallel/PPC-runtime/payload-runtime suites: `46 passed`.
- Full neural-suite and independent Sol gate evidence: `794 passed`, with the
  same 19 known warnings (16 Pynapple zero-duration/rate warnings and three
  multiprocessing `fork` deprecation warnings). Independent Sol `xhigh` final
  gate PASS after memory-lifetime, ownership, immutability, checked-range, and
  identity-accounting corrections.
- Numerical/interface review and unresolved risks: strict canonical equality,
  half-open before/after assignment, target-specific validity, two-neighbor
  interpolation, complex128 interpolation, complex64 normalized samples,
  complex128 reductions, stable unsorted identities, output axes, and checked
  allocation contracts match the plan and oracle. The per-cell NumPy temporary
  lifetime remains below the conservative 51-byte allowance. Python
  edge/unit/segment loops are an explicit S6 profiling question, not an S1
  correctness blocker; Python object overhead and bounded ufunc iterator
  buffers remain covered by process headroom and the later measured-memory
  gate.
- CT026 work performed: none permitted.

### S2 - Observed reuse and whole-from-halves numerics

- Status: complete.
- Authorization and starting HEAD: covered by the user's 2026-09-10
  implementation authorization; `53f81983d19e8d0d4f7f21472e6d0a36911f6fe2`.
  During S2 the user confirmed that the newly observed
  `docs/SoftwareDesign.md` modification was theirs; it and all other unrelated
  changes remained untouched. The resulting unrelated-worktree inventory is
  SHA-256
  `888f0459d557e862e4a5a1fb6b3b8cb3bc18477c8b3ddb8de09910e0b5724bcc`.
- Agent assignments, permissions, and file scopes: lead `gpt-5.6-sol` `high`;
  one retained writer `gpt-5.6-terra` `xhigh`, sequentially write-enabled for
  the focused test file and then the kernel source; independent reviewer
  `gpt-5.6-sol` `xhigh`, procedurally read-only. No scout assigned.
- Test-design review, test-only commit, and RED evidence: the initial API tests
  were committed in `1b39c45`; independent review then found that pooled-only
  observed statistics could not support cross-condition reuse and that
  complex128 angle promotion changed legacy boundary bins. Corrected stable-
  trial tests passed the independent design gate and were committed in
  `fb0ba35`; the final histogram-validation correction was committed in
  `006bb24`. Root reproduced the staged RED sequence, ending with 3 expected
  failures, 83 passes, and 45 deselections for the scoped selection.
- Implementation commit and GREEN evidence: `d8d1a9d`. Root obtained 86 passed
  and 45 deselected for the focused S2 selection, 165 passed for the complete
  kernel plus WP5B oracle files, and 46 passed for the PPC parallel, PPC
  runtime, and payload-runtime suites.
- Full neural-suite and independent Sol gate evidence: 881 passed with 19 known
  warnings. The final independent `gpt-5.6-sol` `xhigh` implementation gate
  returned PASS after verifying the stable-trial mapping, one-pass reuse,
  ordered and empty memberships, whole contributor unions, legacy histogram
  behavior, ownership, and the unchanged S1 51-byte temporary bound.
- Numerical/interface review and unresolved risks: observed sums, counts, and
  representative histograms are retained by stable physical trial before
  condition aggregation. Whole metrics add before/after complex sums and
  counts rather than averaging half PPC. Histogram sampling uses the S1 exact
  canonical-grid acceptance rule but preserves current `np.angle(complex64)`
  bin behavior; valid nominal boundary samples can therefore fall outside
  float64 `[-pi, pi]` edges, so public validation permits histogram totals below
  (never above) selected-frequency valid counts. S3 must include the retained
  per-trial observed arrays in parent-memory accounting.
- CT026 work performed: none permitted.

### S3 - Stable job planner and cross-condition edge union

- Status: complete.
- Authorization and starting HEAD: covered by the user's 2026-09-10
  implementation authorization and repeated instruction to continue;
  `b5fb87748e17863b563e2f474abc0708e80d6003`.
- Agent assignments, permissions, and file scopes: lead `gpt-5.6-sol` `high`;
  one `gpt-5.6-terra` `xhigh` writer, initially write-enabled only for focused
  S3 runtime/model tests; independent `gpt-5.6-sol` `xhigh` reviewer,
  procedurally read-only. No scout assigned.
- Test-design review, test-only commit, and RED evidence: the independently
  reviewed base planner/allocation contract was committed in `7f54204`; a
  genuine one-line edge-identity accounting error was corrected separately in
  `af7b6d4`. Source-gate regressions for aggregate-aware batching, nonnegative
  stable identities, and direct immutable construction were committed in
  `f78ddba`; exact schedule/seed/union/count/metric/batch coherence and early
  allocation rejection in `5ae1819`; and generator-buffer reuse plus exact
  integer schedule-shape identity in `607ccbd`. Each test revision passed an
  independent Sol design gate before commit. The final focused RED was three
  expected failures with 101 passes and 83 deselections.
- Implementation commit and GREEN evidence: `706f6b0` (`feat: plan grouped PPC
  execution`). The final focused planner/allocation selection passed 104 tests
  with 83 deselections; the complete PPC runtime/model files passed 187 tests.
- Full neural-suite and independent Sol gate evidence: the writer and root each
  obtained 985 passing tests with the same 19 known warnings after the final
  trusted-path memory correction. The independent `gpt-5.6-sol` `xhigh` gate
  returned PASS after verifying schedule-buffer transfer, allocation-free
  trusted construction, full public validation, preflight ordering, metadata
  identity, and preservation of the legacy single-job executor.
- Numerical/interface review and unresolved risks: stable physical trial rows
  are nonnegative and condition-local schedules translate to a sorted,
  site-qualified edge union. Jobs are site-major then condition-major then
  epoch-major and retain the exact legacy derived seeds and row-wise
  derangements. Full-shuffle accumulators, retained per-trial observed arrays,
  separate observed/null gather lifetimes, full parent summaries, site/block
  worker summaries, planner arrays, and one shared mmap are all explicitly
  included in checked int64 process/aggregate estimates. Both the 2 GiB
  process limit and 12 GiB aggregate limit drive deterministic condition
  batching, and rejection precedes final stable-map materialization. The
  estimate intentionally covers NumPy arrays rather than Python object/process
  overhead; the reserved headroom and later measured-memory gate remain
  required. S3 plans but does not execute grouped work; S4 owns that boundary.
- CT026 work performed: none permitted.

### S4 - Grouped serial executor and checkpoints

- Status: complete.
- Authorization and starting HEAD: covered by the user's 2026-09-10
  implementation authorization and repeated instruction to continue;
  `8b451f3`.
- Agent assignments, permissions, and file scopes: lead `gpt-5.6-sol` `high`;
  one retained `gpt-5.6-terra` `xhigh` writer, initially write-enabled only for
  focused grouped-runtime/work-cache tests; independent `gpt-5.6-sol` `xhigh`
  reviewer, procedurally read-only. No scout assigned.
- Test-design review, test-only commit, and RED evidence: `cf5dba6` (`test:
  define grouped PPC serial execution`). The final tests-only diff passed the
  independent Sol `xhigh` design gate. Root reproduced 19 expected failures,
  27 passes, and 126 deselections in the grouped/component-plan/checkpoint
  selection, plus 12 expected failures, 55 passes, and 136 deselections in the
  allocation selection. Failures were confined to the intentionally absent S4
  executor, checkpoint streaming/no-copy, and checkpoint-memory contracts.
  The first GREEN source review exposed unaccounted phase selection, planning,
  edge-input, batching, null-finalization, and checkpoint-repair lifetimes.
  Corrective RED coverage passed a second independent test-design gate and was
  committed in `3669c7d` (`test: cover grouped PPC memory safety`) before those
  source corrections began. Subsequent independently reviewed test-only commits
  were `1c76681` (bounded null finalization), `5cefb02` (nondegenerate
  inferential fixtures), `c40a69e` (memory-safe identity paths), `70b4d02`
  (per-schema resume bounds and final validation), `9e720ac` (fixed S1/S2
  representative axis), and `4167a22` (execution-peak and progress contracts).
  The final focused RED contained six expected failures covering fixed planned
  edge partitions, interval-bounded progress, rank validation, scalar
  checkpoint bounds, and interpolation-heavy geometry construction.
- Implementation commit and GREEN evidence: `0155587` (`feat: execute grouped
  PPC serially`). Root reproduced six passing final focused regressions and 430
  passing tests across the PPC kernel, runtime, work cache, and models. The
  non-CT026 LFP-summary group passed 547 tests. Compile and diff checks were
  clean.
- Full neural-suite and independent Sol gate evidence: root obtained 1,067
  passing tests with 19 known warnings. The final independent `gpt-5.6-sol`
  `xhigh` source gate returned PASS on frozen scoped source hash
  `2ea5e02fd867bb82e9c6ded2ac643f0cc84ad3b2ff8443c118c27dcc2ac4c63e`.
  The gate verified planner/executor edge-group coherence, eligibility bypass,
  the geometry construction peak, exact checkpoint schemas and transaction
  behavior, complete work identity, fixed two-band representative histograms,
  raw comparison semantics, and preservation of the legacy single-job
  executor.
- Numerical/interface review and unresolved risks:
  - On 2026-09-14, the user approved treating exact mathematical
    observed/null PPC ties as deterministic summation-order-dependent cases.
    The optimized implementation keeps the raw ``null >= observed`` rule and
    adds no comparison tolerance. Inferential-equivalence fixtures will use
    deterministic trial-distinct spike patterns with at least a ``1e-5``
    comparison margin. This resolves the artificial identical-train fixture
    that changed an exceedance decision by one-to-three binary64 ULPs without
    weakening exact non-tied count, p/q, or significance checks.
  - The serial grouped executor translates stable condition schedules into a
    site-qualified physical-edge union, reuses one observed trial-statistics
    pass, composes whole from before/after sufficient statistics, consumes null
    work in planner-bounded edge groups, and publishes parent-only atomic
    site/unit checkpoints. Every resumed block is validated against its own
    exact schema and byte ceiling.
  - Planned limits cover concurrently live NumPy arrays. Python objects,
    interpreter/process overhead, and filesystem buffers remain outside the
    exact accounting and are reserved by the documented headroom plus the S8
    measured-memory gate. Grouped execution is intentionally serial in S4;
    S7 owns process-worker rebinding after the S6 profile.
- CT026 work performed: none permitted.

### S5 - Payload integration and histogram reuse

- Status: complete.
- Authorization and starting HEAD: covered by the user's 2026-09-10
  implementation authorization and repeated instruction to continue;
  `3aa8e37`.
- Agent assignments, permissions, and file scopes: lead `gpt-5.6-sol` `high`;
  one `gpt-5.6-terra` `high` writer, sequentially write-enabled for the three
  focused S5 test files and then `lfp_summary_runtime.py`; independent
  `gpt-5.6-sol` `high` reviewer, procedurally read-only. No scout assigned.
- Test-only commit and RED evidence: `569dfad` (`test: define grouped PPC
  payload integration`). Independent review required replacing conflicting
  legacy-call expectations, adding a production cache/reload/plot path,
  adversarial exact-grid sampling, nontrivial two-unit exemplars, real
  cold/failure/resume behavior, and two-site seed/order coverage before PASS.
  Root reproduced five expected failures, four passes, and seven deselections;
  every failure was confined to the old legacy executor/per-job progress path.
  A compact sparse-frequency fixture was then identified as violating the
  canonical contiguous 2-Hz band-mean contract; the independently reviewed
  correction was committed as `9881f91` (`test: use canonical PPC payload
  frequency grid`).
- Implementation commit and GREEN evidence: `87ba842` (`feat: integrate
  grouped PPC payload`). The final focused S5 selection passed nine tests with
  seven deselections, and the affected runtime/spike/synthetic/pipeline files
  passed 27 tests. Compile and diff checks were clean.
- Full neural-suite and independent Sol gate evidence: root obtained 1,071
  passing tests with 19 known warnings. The final independent `gpt-5.6-sol`
  `high` gate returned PASS on scoped source hash
  `df4163e7375940e8ccc60e64aa6471af0d583eedb4639cb6bfd107fea67a36f1`.
- Numerical/interface review and unresolved risks: the payload makes exactly
  one grouped executor call, validates and owns every full summary array,
  renames the fixed two-representative histogram without resampling phase, and
  derives exemplar trial counts from spike times only. Stable seeds, schedule
  identity, public axes/schema/dtypes, q/significance, traces, packed spikes,
  exemplars, progress, and manifest-last transaction behavior are preserved.
  Exact-grid/near-grid behavior now comes solely from the S1/S2 grouped kernel.
  Failed final writes retain resumable grouped work; successful commit cleanup
  targets only the exact completed grouped run. The payload continues to use
  the canonical contiguous 2-Hz band-mean helper; no sparse-grid semantics were
  added. Three dead legacy per-job assembly helpers were removed.
- CT026 work performed: none permitted.

### S6 - Profiling and schedule-union cost model

- Status: tests-only contract in progress.
- Authorization and starting HEAD: covered by the user's 2026-09-10
  implementation authorization and repeated instruction to continue;
  `cb96495`. No CT026 data access, work-only timing, or scientific execution is
  authorized in this package.
- Agent assignments, permissions, and file scopes: lead `gpt-5.6-sol` `high`;
  one `gpt-5.6-terra` `high` writer, initially write-enabled only for the four
  profile/adapter/runner test modules; independent `gpt-5.6-sol` `high`
  reviewer, procedurally read-only. Source scope includes the profile and
  adapter/runner modules plus one narrowly bounded private histogram-timing
  seam extraction in `lfp_summary_ppc_runtime.py`. That extraction may not
  change scientific behavior, axes, memory ownership, or checkpoints; all
  other grouped-runtime changes remain S7 work.
- Test-only commit and RED evidence:
- Implementation commit and GREEN evidence:
- Full neural-suite and independent Sol gate evidence:
- Synthetic and metadata-only evidence:
- Numerical/interface review and unresolved risks:
- CT026 work performed: metadata-only inspection unless separately authorized.

### S7 - Rebind parallel workers to the grouped engine

- Status: not started.
- Authorization and starting HEAD:
- Agent assignments, permissions, and file scopes:
- Test-design review, test-only commit, and RED evidence:
- Implementation commit and GREEN evidence:
- Full neural-suite and independent Sol gate evidence:
- Worker invariance, memory, and failure review:
- Unresolved risks:
- CT026 work performed: none unless separately authorized.

### S8 - Authorized representative benchmark and decision gate

- Status: not started; requires separate CT026 authorization.
- Authorization and starting HEAD:
- Lead Sol, Terra command runner, and Sol reviewer assignments:
- Exact authorized commands and data scope:
- Correctness comparison:
- Stage timings, union ratios, and throughput:
- Planned allocation and measured RSS/PSS:
- Selected worker count and rationale:
- Revised 100- and 1,000-shuffle projections:
- Independent Sol gate disposition and unresolved risks:
- Source or test changes: none permitted.
