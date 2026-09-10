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

- Status: not started.
- Authorization and starting HEAD:
- Agent assignments, permissions, and file scopes:
- Test-design review, test-only commit, and RED evidence:
- Implementation commit and GREEN evidence:
- Full neural-suite and independent Sol gate evidence:
- Numerical/interface review and unresolved risks:
- CT026 work performed: none permitted.

### S2 - Observed reuse and whole-from-halves numerics

- Status: not started.
- Authorization and starting HEAD:
- Agent assignments, permissions, and file scopes:
- Test-design review, test-only commit, and RED evidence:
- Implementation commit and GREEN evidence:
- Full neural-suite and independent Sol gate evidence:
- Numerical/interface review and unresolved risks:
- CT026 work performed: none permitted.

### S3 - Stable job planner and cross-condition edge union

- Status: not started.
- Authorization and starting HEAD:
- Agent assignments, permissions, and file scopes:
- Test-design review, test-only commit, and RED evidence:
- Implementation commit and GREEN evidence:
- Full neural-suite and independent Sol gate evidence:
- Numerical/interface review and unresolved risks:
- CT026 work performed: none permitted.

### S4 - Grouped serial executor and checkpoints

- Status: not started.
- Authorization and starting HEAD:
- Agent assignments, permissions, and file scopes:
- Test-design review, test-only commit, and RED evidence:
- Implementation commit and GREEN evidence:
- Full neural-suite and independent Sol gate evidence:
- Numerical/interface review and unresolved risks:
- CT026 work performed: none permitted.

### S5 - Payload integration and histogram reuse

- Status: not started.
- Authorization and starting HEAD:
- Agent assignments, permissions, and file scopes:
- Test-only commit and RED evidence:
- Implementation commit and GREEN evidence:
- Full neural-suite and independent Sol gate evidence:
- Numerical/interface review and unresolved risks:
- CT026 work performed: none permitted.

### S6 - Profiling and schedule-union cost model

- Status: not started.
- Authorization and starting HEAD:
- Agent assignments, permissions, and file scopes:
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
