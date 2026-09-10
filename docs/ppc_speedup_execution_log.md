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
- Expected S0 RED checkpoint: 2 failed and 12 passed in
  `src/tests/neural_analysis/test_lfp_summary_ppc_parallel.py`.
- Implementation authorization: not yet recorded. This documentation handoff
  does not authorize source or test changes.
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

- Status: not started; implementation authorization not yet recorded.
- Authorization:
- Starting HEAD and worktree inventory:
- Lead Sol model/effort:
- Terra worker model/effort, permissions, and file scope:
- Optional Terra scout model/effort, permissions, and findings:
- Sol reviewer model/effort, permissions, and disposition:
- Read-only enforcement or procedural restriction:
- Test-only commit and RED evidence: existing commit `9fe7f1f`; baseline 2
  failed and 12 passed.
- Implementation commit and GREEN evidence:
- Full neural-suite evidence:
- Numerical/interface review:
- Unresolved risks:
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
