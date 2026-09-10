# Exact PPC speedup implementation plan

Status: implementation plan approved by the user on 2026-09-09. This approval
is documentation-only: it does not authorize source changes, CT026 computation,
or the 100/1,000-shuffle Spike-phase runs. The Sol-orchestrator/Terra-worker
execution specification in Section 3.1 was added at the user's request on the
same date, audited against the current host on 2026-09-10, and is subject to
those same authorization boundaries.

This plan turns `docs/ppc_speedup.md` into test-first work packages. It is kept
separate from `docs/Tasks_neural.md` because the change is a substantial,
focused replacement for the remaining WP5C-5 sequencing. After approval,
`Tasks_neural.md` should receive only a short status/link update and retain the
broader LFP-summary roadmap.

## 1. Current state and recommendation

Repository state re-inspected on 2026-09-10:

- HEAD is `31a5c54` on `refactor`.
- The accepted serial sufficient-statistic, prepared-phase cache, restart, and
  checkpoint path exists in `spike_lfp_summary.py`,
  `lfp_summary_ppc_runtime.py`, `lfp_summary_work_cache.py`, and
  `lfp_summary_runtime.py`.
- The current component builder still executes PPC independently for every
  condition, site, and epoch.
- The current edge kernel loops over edges, then units, then calls
  `numpy.interp` twice for every frequency. It repeats the same source-spike
  neighbor search for every target trial.
- Observed execution constructs `numpy.where(valid, phase, 0j)` repeatedly and
  samples same-trial edges a second time solely to count contributing trials.
- The payload builder separately resamples observed phases for every
  unit/condition/site/epoch to construct the representative histograms.
- Whole, before, and after epochs currently run as independent jobs.
- Condition-local schedules are not translated into a cross-condition union,
  so overlapping conditions recompute identical physical edges.
- The serial CT026 work-only profile measured 66.80 seconds for three units at
  100 shuffles and projected roughly 47-51 hours for the full component.
- The worker RED checkpoint remains genuine at HEAD. The focused command
  `uv run pytest -q -p no:cacheprovider
  src/tests/neural_analysis/test_lfp_summary_ppc_parallel.py` currently reports
  2 failed and 12 passed. The remaining failures concern normal executor
  shutdown and cancellation of the future whose `result()` raised.

Recommendation: finish the small existing worker-contract GREEN correction so
the repository has a green baseline, but do not run the planned 1/2/4/8-worker
production benchmark on the current kernel. First remove the measured serial
redundancy. The grouped algorithm changes the useful worker task boundary, so
benchmarking the old task boundary would spend CT026 time on an implementation
that is expected to be replaced.

## 2. Scope and non-goals

The implementation will preserve:

- unweighted frequency-resolved PPC;
- the current complex interpolation and validity support rule;
- complex64 normalized samples followed by complex128 sums;
- the configured half-open before/after partition and its whole-window union
  (approved defaults: before `[-2, 0)`, after `[0, 2)`, whole `[-2, 2)`);
- condition-specific deterministic trial derangements and existing seeds;
- observed results below the permutation reliability threshold;
- the 50-valid-spike and two-contributing-trial inference gates;
- plus-one p-values, `ddof=0` null standard deviation, explicit linear
  percentiles, and frequency-wise BH correction;
- the final `spike_phase.npz` schema and public pipeline entry points;
- exact-fingerprint restart, parent-only checkpoint publication, and
  manifest-last final commit.

This work will not change the estimator, conditions, frequency grid, epochs,
shuffle counts, amplitude policy, final cache schema, or plotting semantics.
It will not add Numba, Cython, Zarr, HDF5, Dask, or another dependency. Cluster
offload and path portability remain later work.

## 3. Decisions requiring approval

The choices below were approved on 2026-09-09. Implementation still requires a
separate user request; approval of this plan alone is not execution authority.

1. **Add a grouped production executor and preserve the single-job executor.**
   Keep `execute_ppc_blocks(...)` as the frozen single-condition/site/epoch
   equivalence and profiling boundary. Add a grouped component executor for the
   optimized production path instead of stretching the single-job return type
   into condition/site/epoch axes.
2. **Use search-once geometry, then direct gathers.** Build neighbor indices and
   weights once from the canonical stored time coordinate and source spikes.
   Use exact equality to the canonical grid for exact-sample classification,
   matching the current WP5B reference. Validate uniform spacing against
   `config.phase.output_rate_hz` (500 Hz for the approved default), but do not
   derive boundary classification solely from floating-point division by `dt`.
3. **Preserve the existing epoch-specific schedules.** Before, after, and whole
   keep the current `_ppc_schedule_seed(...)` results. A whole schedule consumes
   the sum of before/after edge statistics for each edge in that same whole
   schedule. Standalone before/after schedules consume only their own segment.
4. **Require exact inferential decisions.** Identities, schedules, validity,
   counts, eligibility, exceedance counts, permutation counts, p-values,
   q-values, and significance flags must match the reference. Floating metrics
   may use the tight tolerances in Section 7. A disagreement caused by a null
   draw tied within tolerance of observed PPC is a review failure, not an
   automatically accepted significance change.
5. **Use an explicit planned-allocation limit.** Add execution-only integer
   `maximum_worker_allocation_bytes = 2 * 1024**3` (2 GiB). It limits
   planned private NumPy accumulators per worker, does not enter scientific
   component identity, and does not claim to measure shared mmap residency.
   Aggregate system memory remains governed by the existing 16 GiB benchmark
   gate.
6. **Require separate CT026 authorization.** Synthetic tests and metadata-only
   schedule-union measurement are authorized by implementation approval. Any
   new work-only CT026 timing run still requires explicit execution approval;
   neither a full preview nor a 1,000-shuffle run is implied.

### 3.1 Codex implementation team and model policy

This subsection is part of the approved implementation method. When the user
separately authorizes implementation, that authorization includes the bounded
delegation described here; it does not authorize implementation, CT026 work, or
benchmarking by itself. Do not ask the user to choose models again unless the
named model/effort combination is unavailable or the requested scope changes.

The model names below refer to Codex agents, not to the Python worker processes
implemented and benchmarked in S0, S7, and S8. In this document:

- **Lead Sol orchestrator** means the primary/root Codex agent using
  `gpt-5.6-sol` that coordinates the complete implementation and owns gate
  decisions.
- **Sol gate reviewer** means a separate `gpt-5.6-sol` Codex agent that audits
  a stable test or implementation diff but has no orchestration or write
  authority.
- **Terra package worker** means the single `gpt-5.6-terra` Codex agent that
  performs the bounded test and implementation edits assigned for one active
  package.
- **Terra scout** means an optional, read-only `gpt-5.6-terra` Codex agent used
  only for a bounded evidence-gathering task.
- **PPC process worker** means a local Python multiprocessing process in the
  scientific implementation. Codex-agent counts and PPC process-worker counts
  are unrelated and must be reported separately.

The exact model IDs and effort values were rechecked on 2026-09-10 against the
official [Sol model](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
[Terra model](https://developers.openai.com/api/docs/models/gpt-5.6-terra), and
[Codex subagents](https://developers.openai.com/codex/agent-configuration/subagents/)
documentation and against the current host capability metadata. The public
model API and the Codex team runtime do not necessarily expose identical effort
menus, so the table records the values currently accepted for spawned agents
on this host. Recheck host support at the start of implementation because it is
account- and release-dependent. Never silently substitute another model or
effort.

| Codex model | Intended role in this plan | Spawn efforts exposed on the checked host |
| --- | --- | --- |
| `gpt-5.6-sol` | Lead orchestration, scientific judgment, final package gates, and difficult discrepancy resolution | `low`, `medium`, `high`, `xhigh`, `max`, `ultra` |
| `gpt-5.6-terra` | Bounded codebase exploration, test authoring, implementation, test triage, and evidence collection | `low`, `medium`, `high`, `xhigh`, `max`, `ultra` |

Codex UI text may call `xhigh` **Extra High**. Agent configuration and explicit
spawn requests must use the machine value `xhigh`. Do not use `none` or
`minimal` for either named model in this plan.

#### Fixed roles and authority

1. **Lead Sol orchestrator: `gpt-5.6-sol`, `xhigh`.** One primary/root agent
   owns requirements, package ordering, user communication, authorization
   gates, worktree inspection, worker prompts, RED/GREEN verification, and git
   commits. It is the only agent allowed to declare a package complete. It must
   independently inspect diffs and test output rather than accepting a worker's
   summary as proof. It has repository write authority for staging the reviewed
   package paths and creating the required test-only, implementation, and
   documentation-only package-record commits, but it does not author or repair
   package source/tests; accepted edit findings return to the Terra package
   worker.
2. **Sol gate reviewer: `gpt-5.6-sol`, `high` or `max` as assigned in the table
   below.** This is a fresh, read-only agent used at a test-design or completed-
   package gate. It looks for scientific drift, invalid tests, interface or data-
   contract breaks, concurrency faults, and missing verification. It reports to
   the lead and does not edit, commit, spawn agents, or broaden scope.
3. **Terra package worker: `gpt-5.6-terra`, `high` or `xhigh` as assigned
   below.** Exactly one write-enabled Terra agent owns the bounded files for the
   active package. The same agent writes tests, stops for the externally
   verified RED/commit gate, and resumes only after the lead explicitly assigns
   implementation. It never commits, modifies tests merely to make them pass,
   runs unauthorized CT026 work, edits files outside its assignment, or spawns
   another agent.
4. **Optional Terra scout: `gpt-5.6-terra`, `medium`, read-only.** The lead may
   use one scout for a genuinely independent, evidence-heavy task such as usage
   mapping, installed-library API inspection, or failure-log triage. A scout
   returns file/line references and uncertainties, makes no edits, and must
   finish before its findings become implementation requirements. Do not create
   a scout for work the package worker or lead can do directly with little
   context cost.

The lead stays at `xhigh`; it is not necessary to restart the root thread for a
package-specific `max` gate. Instead, the independent Sol reviewer receives
`max` for the high-risk packages listed below. This keeps routine coordination
responsive while adding deeper review exactly where numerical, transaction, or
concurrency mistakes would be expensive.

`low` is not used because even the read-only tasks require scientific-repository
context. `medium` is restricted to optional, narrowly scoped Terra scouting.
Terra `high` is the default for clear implementation packages, and Terra
`xhigh` is required for numerical kernels, cross-condition identity, restart,
or concurrency work. Sol `max` is a review/escalation setting, not the default.
`ultra` is not assigned anywhere in this plan. Do not treat the ChatGPT Ultra
intelligence level, its proactive-delegation behavior, and a host-exposed
`reasoning_effort="ultra"` value as interchangeable. If `max` cannot resolve a
scientific or architectural discrepancy, stop and ask the user; any move to
Ultra or change in agent topology requires a documented plan revision.

#### Concurrency, workspace, and ownership rules

- Use at most four active Codex agents including the lead: one lead Sol, one
  write-enabled Terra package worker, one read-only Sol gate reviewer, and at
  most one optional read-only Terra scout. If the runtime exposes fewer slots,
  reduce read-only concurrency; never replace the lead or writer.
- Only one agent may edit the shared worktree at a time. Work packages S0-S8
  remain sequential. Read-only scouting and review may overlap only when their
  questions are independent and they do not inspect a diff that is still
  changing.
- "Read-only" means no file, index, commit, cache, generated artifact, or
  external-state mutation. Prefer an enforced read-only custom-agent sandbox
  when the runtime exposes one. If the spawn interface exposes only inherited
  permissions, the prompt restriction is procedural rather than a security
  boundary; the lead must record that fact and verify the worktree before and
  after the assignment.
- All agents share the same worktree. The lead records `git status --short` and
  the active HEAD before each assignment and checks them again after it. If any
  unassigned or unexpected change appears, all work stops immediately and the
  user decides how to proceed. No agent may reset, revert, clean, amend, or
  overwrite unrelated user changes.
- The Terra writer receives an explicit file allowlist. If it discovers that a
  necessary edit falls outside that list or changes a public interface, it must
  stop and return the evidence to the lead for replanning; it must not expand
  its own assignment.
- The lead performs all git commits. A Terra worker's completion means only that
  its bounded edit and local evidence are ready for lead review.
- The lead stages only the reviewed package paths, inspects the staged diff, and
  confirms that unrelated pre-existing changes remain unstaged before each
  commit. Commit authority does not relax the one-editor rule: the Terra worker
  must be idle before the lead stages or commits.
- Before S0, the lead requires a committed documentation baseline containing
  this plan and its `docs/Tasks_neural.md` handoff. It creates
  `docs/ppc_speedup_execution_log.md` in that documentation-only baseline; this
  is the persistent package record referenced below. Documentation changes must
  never be swept into a test-only or implementation commit. Unrelated
  pre-existing worktree changes need not be cleaned, but must be inventoried and
  left untouched.
- Subagents may not create nested agents. The lead alone spawns, follows up, or
  interrupts agents, and closes completed threads when the runtime exposes that
  action. It waits for all requested reports before a gate decision.

#### Spawn and handoff contract

Before source work begins, the lead must confirm from session/host metadata,
not model self-report, that its active configuration is `gpt-5.6-sol` with
`xhigh` reasoning and that explicit `gpt-5.6-terra`/effort overrides are
available for children. If the runtime does not expose enough metadata to make
that confirmation, or either check fails, stop and ask the user whether to
reconfigure or revise the plan. Do not silently let Terra inherit Sol, let Sol
perform the Terra assignment, or replace either model with the family alias,
Astra, Luna, or another available model.

Every delegated prompt must be self-contained and state all of the following:

- package ID, role, exact model, exact reasoning effort, and whether the task is
  read-only or write-enabled;
- goal, accepted decisions from Sections 2-3, exact file allowlist, forbidden
  files/actions, and the public interfaces/data contracts that must remain
  unchanged;
- tests to write or run, the expected RED failure, and the exact stopping gate;
- applicable numerical, memory, authorization, and TDD requirements;
- required return format: files inspected/changed, concise findings or diff
  summary, commands and outcomes, expected versus actual RED/GREEN evidence,
  unresolved risks, and confirmation that no commit or out-of-scope edit was
  made.

When the agent API exposes `spawn_agent`, use explicit overrides. A Terra S1
assignment, for example, uses `model="gpt-5.6-terra"`,
`reasoning_effort="xhigh"`, a unique task name such as `s1_kernel_worker`, and
a bounded context fork such as `fork_turns="3"`. The corresponding S1 gate
uses `model="gpt-5.6-sol"`, `reasoning_effort="max"`, and a distinct task name;
scouts always use Terra `medium`. Do not use a full-history
`fork_turns="all"` when changing model or effort: current team runtimes require
full-history children to inherit the parent settings. The prompt must therefore
carry all essential constraints rather than depending on inherited chat history.
If the active runtime uses different field names, select the documented exposed
equivalents for model and reasoning. Record any returned configuration metadata;
if the spawn call rejects an override or reports a different configuration, the
agent must not edit and the lead must stop for user direction.

Use follow-up messages to move the same Terra package worker across its TDD
stops; do not start a replacement implementation agent merely to avoid a clean
handoff. A follow-up must state which gate the lead verified, the new allowed
action, and any review findings. Interrupt an agent only for wrong scope,
unsafe behavior, unexpected worktree changes, or a superseding user request.

#### Package-specific agent assignments

| Package | Terra assignment | Independent Sol gate | Required emphasis |
| --- | --- | --- | --- |
| S0 | `high` | `high` | Narrow executor shutdown/cancellation correction; tests already provide RED. |
| S1 | `xhigh` | `max` | Interpolation geometry, exact-sample boundaries, dtypes, axes, and segmented sums. |
| S2 | `xhigh` | `max` | Whole-from-halves scientific equivalence, cross terms, eligibility, and histogram semantics. |
| S3 | `xhigh` | `max` | Stable physical identities, union planning, deterministic batching, and allocation arithmetic. |
| S4 | `xhigh` | `max` | Grouped execution, bounded memory, checkpoint fingerprints, resume, and transaction failure behavior. |
| S5 | `high` | `high` | Minimal payload-loop integration while preserving schema, seeds, exemplar behavior, and public entry points. |
| S6 | `high` | `high` | Scalar-only profiling, metadata-only CT026 inspection, and no unauthorized scientific computation. |
| S7 | `xhigh` | `max` | Spawn/failure/cancellation semantics, shared mmap behavior, deterministic parent publication, and memory safety. |
| S8 | `high`, command execution and evidence collection only | `max`, benchmark interpretation and recommendation | Respect separate authorization; no source edits; select worker count only from measured correctness, throughput, and memory evidence. |

Every S1-S7 package receives an independent Sol review of the stable GREEN
diff. A pre-commit test-design review is additionally mandatory for S1-S4 and
S7, whose omitted cases could permit numerical, identity, restart, or
concurrency drift; it is optional for S5-S6 when the lead identifies comparable
risk. The same reviewer may perform both gates if it retains no write access.
S0 uses only the final gate because its RED tests already exist. S8 has no Terra
package writer: its Terra assignment may execute only the explicitly authorized
benchmark commands and collect raw outputs, with no repository edits. The lead
and `max` Sol reviewer interpret the stable S8 evidence independently.

#### Mandatory per-package agent/TDD sequence

1. The lead Sol re-reads the relevant source, tests, usages, package scope, and
   current worktree state. It resolves ambiguity with the user before edits.
2. If useful, a medium Terra scout returns bounded evidence. The lead decides
   what, if anything, becomes part of the worker assignment.
3. The assigned Terra writer adds only the package's tests, runs the focused
   command if allowed, records the genuine expected RED, and stops. S0 starts at
   its already committed RED state.
4. The lead inspects the test diff, confirms that failure is caused by missing
   required behavior rather than a broken fixture, reruns RED independently,
   obtains the assigned Sol test-design gate when required, and creates the
   test-only commit.
5. The lead follows up with the same Terra writer to authorize implementation.
   The worker makes the smallest in-scope source change, runs focused tests to
   GREEN, reports evidence, and stops without committing.
6. The lead independently inspects the complete diff and runs the focused and
   affected suites. The assigned Sol gate reviewer audits requirements,
   scientific equivalence, contracts, tests, and failure behavior from a
   read-only state.
7. The Terra writer addresses only specific accepted review findings. The lead
   repeats verification, runs the full neural suite required by Section 10, and
   creates the separate implementation commit only when every gate is green.
8. The lead records package commit IDs, model/effort assignments, tests and
   outcomes, unresolved risks, and any authorized benchmark evidence before
   starting the next package, then commits that execution-log update separately
   from the test-only and implementation commits.

#### Interruption and recovery contract

- An interrupted agent report is never a completed gate. The lead records the
  last independently verified state, including HEAD, `git status --short`, the
  exact diff, commands already run, and whether RED tests or implementation
  changes are uncommitted.
- If a Terra package worker becomes unavailable because of a usage limit or
  runtime failure, the lead first re-audits the bounded diff and reproduces the
  last claimed test result. It may then spawn one replacement Terra worker with
  the same model, effort, role, file allowlist, and stopping gate. The
  replacement prompt must include the recovery record and must not rely on the
  interrupted worker's unverified summary. A different model or expanded scope
  requires user approval.
- If a Sol reviewer is interrupted, none of its partial findings count. Spawn a
  fresh reviewer with the same model/effort only after the relevant diff is
  stable, and rerun that gate from the beginning.
- If the lead/root thread is interrupted, a replacement lead must satisfy the
  same Sol/`xhigh` requirement, re-read this plan and the package record, inspect
  HEAD plus staged and unstaged changes, and independently establish the last
  completed gate before delegating further work. It must not infer authorization
  or RED/GREEN status from an incomplete chat.
- If ownership of an existing edit cannot be established after those checks,
  stop without modifying or committing it and ask the user how to proceed.

Agent agreement is not evidence of correctness. Conflicting reports, uncertain
library behavior, near-threshold numerical changes, or a failure that cannot be
reproduced by the lead are escalation conditions: pause the package, preserve
the RED test and worktree, and ask the user for a scientific or scope decision.

## 4. Architecture

### 4.1 Module responsibilities

`spike_lfp_summary.py`

- Remains the scientific reference for PPC, schedule generation, null
  summaries, and BH correction.
- Retains the current public functions and `EdgeSufficientStatistics` contract.
- May receive one small shared helper for converting complex sums/counts to
  PPC, so the formula is not reimplemented in multiple modules.
- The existing `compute_trial_shuffle_ppc(...)` remains the oracle for small
  equivalence tests.

New `lfp_summary_ppc_kernel.py`

- Owns the optimized NumPy-only interpolation and segmented edge reduction.
- Has no file I/O, multiprocessing, checkpoints, pipeline imports, or
  Streamlit imports.
- Builds source-spike interpolation geometry once per bounded unit block.
- Reduces one bounded physical edge block into before/after complex sums and
  counts while vectorizing across frequency and concatenated units.
- Optionally returns representative-frequency observed histogram counts for
  same-trial edges; it never retains all sampled phases.
- Exposes internal `build_source_trial_spike_geometry(...)` and
  `compute_segmented_edge_statistics(...)` functions returning the frozen
  contracts below. Their implementation signatures may use keyword-only
  arguments, but must accept the configured time grid/rate, stable source-trial
  identity, configured segment bounds, bounded unit/trial spikes, explicit
  phase validity, frequency axis, and bounded stable physical edges without
  importing pipeline objects.

`lfp_summary_ppc_runtime.py`

- Keeps `execute_ppc_blocks(...)` for reference compatibility.
- Adds deterministic job planning, local-to-stable trial translation,
  cross-condition edge unions, bounded job accumulators, checkpoint assembly,
  and the grouped production executor.
- Exposes internal `plan_grouped_ppc_component(...)` and
  `execute_grouped_ppc_component(...) -> PPCComponentExecutionResult`. The
  planner is pure and performs no phase sampling or file I/O. The executor
  receives `config`, `execution`, complete prepared phase/spike inputs,
  `work_root`, and an optional progress callback; it constructs the unchanged
  schedules internally rather than accepting a replacement schedule.
- Processes one site and one unit block at a time.
- Workers, when re-enabled, receive unit-local spikes and open one shared,
  read-only prepared-phase representation.

`lfp_summary_runtime.py`

- Replaces the condition/site/epoch call loop in
  `_build_spike_phase_payload(...)` with one grouped executor call.
- Continues to own final public payload construction, exemplar selection,
  final component validation, and post-commit cleanup.
- Stops resampling all 50 frequencies for representative histograms; it
  consumes histogram counts produced from observed same-trial sampling.

`lfp_summary_models.py`

- Adds only the approved execution-only memory-limit field.
- Scientific fingerprints remain unchanged when execution settings change.

`lfp_summary_work_cache.py`

- Reuses the established atomic checkpoint primitives.
- Validates the new grouped checkpoint axes and complete execution-plan
  fingerprint; work artifacts remain incompatible with final components.

`lfp_summary_ppc_profile.py` and CT026 profile adapter/runner modules

- Add grouped-kernel stage timings and metadata-only schedule-union metrics.
- Keep profile products scalar/metadata-only and work-only.

### 4.2 New internal data contracts

Every dataclass/function must document types, shapes, axes, units, missingness,
returns, and failure behavior.

`SourceTrialSpikeGeometry` (frozen dataclass)

- Represents one physical source trial for one bounded unit block.
- `source_trial_index`: int64 scalar stable trial-table row identity.
- `unit_ids`: tuple of unique stable unit IDs in the bounded block, defining the
  unit order used by `group_offsets`.
- `group_offsets`: int64 shape `(unit * 2 + 1,)`; contiguous groups are ordered
  unit-major, then before/after.
- `left_index`, `right_index`: int64 shape `(spike,)` into the canonical phase
  time axis. They are safe, clamped gather indices for every spike; values for
  an outside-support spike are ignored because `inside_support` is false.
- `right_weight`: float64 shape `(spike,)`; exact samples have weight zero and
  identical left/right indices. Outside-support weights are zero and ignored.
- `inside_support`, `exact_sample`: Boolean shape `(spike,)`.
- Spike order within each unit/segment is preserved. No physical units apply to
  indices/offsets; weights are dimensionless.

`SegmentedEdgeStatistics` (frozen dataclass)

- Stable source/target physical trial identities: int64 shape `(edge,)`.
- `phase_vector_sum`: complex128 shape `(edge, unit, segment=2, frequency)`.
- `valid_spike_count`: int64 with identical axes.
- Segment order is exactly `before`, `after`; whole is never stored as an
  independently sampled segment.
- No per-spike phase array survives the reducer.

`PPCJobPlan` (frozen dataclass)

- Identifies one condition/site/epoch result cell.
- Stores condition/site/epoch positions and names, stable selected trial rows,
  the unchanged local derangement schedule, its stable physical-edge mapping,
  and which segment expression (`before`, `after`, or `before + after`) it
  consumes.
- Condition-local position zero is never used as a cross-condition edge key.

`PPCComponentExecutionResult` (frozen dataclass)

- Returns summary arrays on
  `(unit, condition, site, epoch, frequency)` axes and representative histogram
  counts on `(unit, condition, site, epoch, band, phase_bin)` axes.
- Contains exact completed/resumed site-and-unit-block identities and work
  directories. A block identity includes the stable site ID and half-open unit
  bounds; unit-block identity alone is insufficient because sites are processed
  separately.
- Contains no full-session null tensor or prepared phase tensor.

### 4.3 Grouped serial algorithm

For each site:

1. Build every condition/epoch schedule exactly as today.
2. Translate every schedule edge from condition-local positions to stable
   physical trial rows and calculate independent-edge count, union-edge count,
   saturation, reuse ratio, and theoretical projected accumulator bytes. These
   schedule metrics remain reportable even when observed eligibility later
   permits work to be skipped.
3. For each bounded unit block, classify all trial-local spikes into the two
   configured disjoint half-open segments and build interpolation geometry once.
4. Process the required same-trial edges first. From that single bounded pass,
   derive observed before/after sums, counts, and contributing-trial counts;
   compose whole metrics from the two segments; and build the representative
   histograms at the configured frequencies nearest 8 and 40 Hz. A trial with
   a positive count in either segment counts once toward whole eligibility.
5. Use those observed results to determine the exact null-eligible
   unit/frequency cells for each job. A job with no eligible cell in the current
   unit block requests no shuffled edge reduction; an inactive cell in an
   otherwise active job receives no shuffle accumulation. Whole eligibility
   activates both segment statistics even when the standalone halves are
   ineligible.
6. Form the active stable physical shuffled-edge union. Allocate bounded
   complex128/int64 accumulators and temporary float64 PPC draws only for the
   current active job batch. If their planned peak exceeds the approved limit,
   split jobs into deterministic condition batches and report that batching
   reduces cross-condition reuse.
7. Process the active shuffled-edge union in bounded edge blocks. The kernel
   gathers real/imaginary coefficients across all frequencies, applies target-
   specific validity, normalizes in complex64, and reduces immediately to
   unit/segment sums and counts.
8. Apply each edge block immediately to all eligible condition/epoch consumers,
   preserving each unchanged schedule, then discard it. Never materialize the
   complete edge table unless a later measured cost-model change is separately
   approved.
9. For shuffled accumulation, a before or after job consumes its segment. A
   whole job adds both segment statistics for each edge before accumulating the
   unchanged whole schedule row.
10. Convert the current unit block's accumulators to PPC draws, compute exact
    null summaries, discard draws, then apply BH after all frequencies for the
    unit/job spectrum are available.
11. Parent code writes a complete grouped site-and-unit-block checkpoint in
    canonical site/unit order. The final component and manifest remain owned by
    the existing pipeline transaction.

## 5. Dependencies

- Existing: Python standard library, NumPy, SciPy, pytest.
- No new runtime or development dependency is proposed.
- Before first implementation use, verify the installed NumPy APIs used for
  indexed gathering, cumulative/segmented reduction, and explicit percentile
  behavior against the installed package source or official documentation.
- Continue to use `uv` and `uv run` for all Python commands.

## 6. Tests that must be written before implementation

The following is the complete test inventory. Each package in Section 8 commits
its tests and records genuine RED before its source implementation.

### Geometry and interpolation

- Reject a nonuniform grid or one inconsistent with the configured output rate
  and canonical event grid before geometry construction; accept a valid
  non-500-Hz configuration and reproduce the reference on its canonical grid.
- Exact first/interior/last samples use one sample and do not require an
  adjacent valid neighbor.
- Between-sample spikes require both neighbors valid for each target trial and
  frequency.
- Outside-support, nonfinite, zero-magnitude, and zero-valued interpolation
  results remain invalid.
- Spikes at the configured shared half-epoch boundary (zero in the approved
  default) belong only to the after segment; a valid nonzero-boundary
  configuration follows the same rule.
- Search-once geometry reproduces current `compute_edge_sufficient_statistics`
  counts and normalized complex64-before-complex128 sums.
- Neighbor-search call count depends on source unit/trial spikes, not the
  number of target edges, conditions, schedules, or shuffle blocks.
- Kernel output is invariant to unit and edge block sizes.

### Observed and whole composition

- One same-trial edge pass yields pooled sums, counts, and contributing-trial
  counts; no second trial-count pass occurs.
- No full phase-mask `numpy.where` copy is constructed inside a unit/trial loop.
- Before plus after sufficient statistics reproduce direct whole observed PPC,
  resultant length, preferred phase, counts, computability, and reliability.
- Cross-before/after phase cancellation is retained; averaging half PPC values
  fails the fixture and composed sufficient statistics pass it.
- A trial contributing to both halves counts once for whole eligibility.
- A unit below 50 spikes in each half but at or above 50 in whole remains whole-
  eligible.
- Whole shuffled draws equal the WP5B direct-whole reference when the same
  whole schedule is supplied.
- Existing distinct before/after/whole schedule seeds are unchanged.

### Cross-condition planning and reduction

- Local trial positions from different conditions translate to correct stable
  physical identities.
- Nested and partially overlapping conditions request one stable edge union
  without enlarging any condition's trial pool.
- A physical edge is sampled once per site/unit block/segment even when several
  condition schedules consume it.
- Independent execution, grouped execution, scheduled-edge mode, and forced
  complete-pair reference mode agree for small seeded fixtures.
- Planner edge counts, union counts, saturation, reuse ratio, and memory-byte
  estimates match hand-calculated examples at 100 and 1,000 shuffles. Peak
  accumulator accounting includes simultaneously resident complex sums, int64
  counts, float64 PPC draws, and per-job accumulator scratch.
- Deterministic condition batching under the allocation limit changes neither
  output order nor values.
- The allocation limit defaults to exactly 2 GiB, round-trips as an
  execution-only integer, rejects Boolean/nonpositive values, and does not
  alter any final scientific component fingerprint.
- Empty/single-trial conditions retain observed semantics and do not request an
  invalid derangement.

### Runtime, restart, and integration

- Grouped summaries have exact final unit/condition/site/epoch/frequency axes.
- Checkpoint fingerprints bind condition memberships, stable trial rows, every
  schedule, segment definitions, phase/spike identity, kernel version, and
  execution settings.
- Grouped checkpoint IDs and axes bind the stable site identity as well as the
  half-open unit block; canonical resume order is site-major then unit-major.
- Cold and resumed grouped runs are equivalent; corrupt/orphan/mismatched
  blocks recompute without damaging valid siblings.
- Only the parent writes checkpoints, progress, final components, and the
  manifest.
- Failure never publishes a partial `spike_phase.npz`.
- Execution-only memory/block/worker settings do not stale a scientific
  component.
- Inference-ineligible unit/frequency entries do not request shuffled edge
  accumulation, including whole-eligible/half-ineligible mixed cases.
- Representative histograms match the current payload builder without its
  repeated 50-frequency resampling.
- Existing final component schema, exemplar selection, post-commit cleanup, and
  pipeline entry points remain unchanged.
- Seeded synthetic end-to-end cache and plotting integration remains green.

### Parallelism and performance instrumentation

- Finish the existing bounded-spawn executor tests: successful shutdown,
  cancellation including the failed future, ordered results, and resumable
  prior yields.
- Worker counts 1/2/4/8 preserve all exact fields and tolerance-governed
  floating fields.
- Workers open one read-only prepared phase representation and do not serialize
  or privately copy the full tensor.
- Submission remains bounded and parent checkpoint order remains canonical.
- Grouped profiler reports geometry, observed, union-edge reduction, shuffle
  aggregation, null summary, histogram, checkpoint, and total times.
- Profile metadata reports private planned allocation, process RSS, aggregate
  RSS/PSS where available, mmap cache size, edge reuse, and throughput.

## 7. Numerical equivalence policy

Exact equality is required for:

- axes, shapes, dtypes, unit/site/condition/epoch/trial identities;
- schedule rows and schedule seeds;
- exact/between/outside classification and phase-validity masks;
- valid-spike and contributing-trial counts;
- computable, reliable, null-eligible, and significant flags;
- null exceedance and permutation counts;
- p-values and q-values derived from those exact counts.

Initial floating tolerances, subject to approval:

- normalized complex samples: `rtol=0`, `atol=2e-7` after complex64 casting;
- PPC, resultant length, null mean/std/percentiles: `rtol=1e-6`,
  `atol=1e-7`, with matching NaN positions;
- preferred phase: circular absolute difference at most `1e-6` radians;
- complex sums: `rtol=1e-6`, `atol=max(2e-7, 2e-7 * valid_count)` to account
  for a changed but deterministic complex64 summation grouping.

Tests must also use adversarial invalid-support and epoch-boundary fixtures so
these tolerances cannot conceal a lost/doubled spike. If floating differences
change an exceedance comparison or FDR decision, the package fails review even
when each raw value falls within tolerance. The implementation must either
restore the reference ordering or present the near-tie case for a separate
scientific decision.

## 8. Sequential test-first work packages

All packages use one write-enabled Terra implementer at a time because the
principal files overlap. The Sol gate reviewer and optional Terra scout remain
read-only; the lead Sol has orchestration, verification, package-record, and
staging/commit authority, but no package source/test edit role, as specified in
Section 3.1. Every test-only commit precedes its implementation commit. No
package may start from a red baseline except S0, whose RED tests are already
intentionally committed.

### S0 - Restore the existing worker checkpoint to GREEN

Files:

- `src/neural_analysis/lfp_summary_ppc_runtime.py`
- existing `src/tests/neural_analysis/test_lfp_summary_ppc_parallel.py`

Tests already committed first in `9fe7f1f`; current RED is 2 failed, 12 passed.
Implement explicit normal shutdown and cancel the failed current future as well
as still-pending futures. Run the focused file, affected PPC runtime tests, then
the full neural suite. Do not run a production worker benchmark.

### S1 - Uniform-grid geometry and segmented edge kernel

Files:

- new `src/neural_analysis/lfp_summary_ppc_kernel.py`
- new `src/tests/neural_analysis/test_lfp_summary_ppc_kernel.py`
- minimal shared-formula addition in `spike_lfp_summary.py` and focused tests if
  needed

Write all geometry/interpolation tests first. Implement immutable geometry and
segmented edge statistics without runtime or cache changes.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_ppc_kernel.py src/tests/neural_analysis/test_spike_lfp_summary.py -k "geometry or segmented_edge or sufficient"`

### S2 - Observed reuse and whole-from-halves numerics

Files:

- `lfp_summary_ppc_kernel.py` and its test file
- focused additions to `test_lfp_summary_ppc_runtime.py` only if the assertion
  belongs to runtime dispatch

Write the observed single-pass, half-open boundary, cross-term, whole
eligibility, histogram additivity, and direct-whole equivalence tests first.
Implement reducers that return sufficient statistics, not final payloads.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_ppc_kernel.py -k "observed or whole or histogram"`

### S3 - Stable job planner and cross-condition edge union

Files:

- `lfp_summary_ppc_runtime.py`
- `test_lfp_summary_ppc_runtime.py`
- `lfp_summary_models.py` and `test_lfp_summary_models.py` for the approved
  allocation limit

Write planner, stable-identity, union, memory-estimate, and deterministic-
batching tests first. This package builds plans only; it does not replace the
production component loop.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py src/tests/neural_analysis/test_lfp_summary_models.py -k "plan or edge_union or allocation"`

### S4 - Grouped serial executor and checkpoints

Files:

- `lfp_summary_ppc_runtime.py`
- `lfp_summary_work_cache.py` only if validation cannot remain in runtime
- focused runtime/work-cache tests

Write grouped-reference equivalence, bounded-consumption, eligibility bypass,
progress, checkpoint, resume, corruption, and failure tests first. Implement
the grouped serial engine while leaving `execute_ppc_blocks(...)` intact.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py src/tests/neural_analysis/test_lfp_summary_work_cache.py -k "grouped or component_plan or checkpoint or resume"`

### S5 - Payload integration and removal of duplicate histogram sampling

Files:

- `lfp_summary_runtime.py`
- `test_lfp_summary_runtime.py`
- `test_lfp_summary_synthetic_integration.py`

Write final-axis/schema, histogram equivalence, unchanged seed, unchanged
exemplar, transaction, and end-to-end synthetic tests first. Replace only the
PPC job loop and duplicate observed histogram sampling. Preserve public builder
signatures.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_runtime.py src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py -k "spike_phase or ppc"`

### S6 - Profiling and schedule-union cost model

Files:

- `lfp_summary_ppc_profile.py` and its tests
- the smallest necessary CT026 profile adapter/runner changes and their
  adapter, runner, and production-binding tests

Write scalar stage-accounting, union-metric, and memory-accounting tests first.
Run synthetic microbenchmarks only during implementation. Present the cheap
metadata-only CT026 100/1,000-shuffle union calculation before requesting any
new production timing run.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_ppc_profile.py src/tests/neural_analysis/test_lfp_summary_ct026_profile_adapter.py src/tests/neural_analysis/test_lfp_summary_ct026_profile_runner.py src/tests/neural_analysis/test_lfp_summary_ct026_profile_production_binding.py`

### S7 - Rebind parallel workers to the grouped engine

Files:

- `lfp_summary_ppc_runtime.py`
- `test_lfp_summary_ppc_parallel.py`

Only after S6 shows the optimized serial stage profile, write worker-invariance,
shared-mmap, aggregate-memory, ordered-checkpoint, and failure tests first.
The existing spawn/bounded-window semantics remain binding. Implement workers
across independent unit blocks, not conditions or sites.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_ppc_parallel.py src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py`

### S8 - Authorized representative benchmark and decision gate

No source implementation belongs in this package. After explicit authorization:

1. Run the work-only low/median/high representative CT026 job at 100 shuffles
   with one worker and warm prepared phase.
2. Compare old accepted serial versus grouped serial stage timings and outputs.
3. If parallelism remains worthwhile, run 1/2/4/8 workers on enough unit blocks
   to occupy every requested worker.
4. Choose the smallest worker count whose median throughput is within 10% of
   the best measured throughput and whose aggregate memory stays below 16 GiB.
5. Re-project 100- and 1,000-shuffle complete-component runtime from measured
   union edges; do not reuse the old 47-51/90-100 hour projections.
6. Present results for review. WP5C-6 preview remains separately gated.

## 9. Performance and memory acceptance gates

- Correctness and exact inferential decisions are mandatory regardless of
  speed.
- Synthetic kernel benchmarks use fixed seeded data, warm inputs, at least five
  measured repeats after warm-up, and median plus range; a single best timing
  is not accepted.
- Each accepted serial optimization must either improve its targeted measured
  stage beyond run-to-run noise or provide necessary architecture for a later
  measured optimization. Revert/defer complexity that does neither.
- Report geometry-build time separately so reuse is not hidden in edge timing.
- Report independent versus union edge counts; do not infer reuse from condition
  names.
- Temporary edge arrays remain proportional to
  `edge_block * unit_block * 2 segments * frequency`.
- Shuffle accumulators and temporary PPC draws remain proportional to the
  current bounded unit/job batch, never the complete unit population. The
  2 GiB allocation gate applies to their concurrently resident planned peak,
  not to the sum/count arrays in isolation.
- At CT026 defaults, 27 jobs (9 conditions x 3 epochs), 1,000 shuffles, 8 units,
  and 50 frequencies require about 247 MiB for complex128 sums plus int64 counts
  for one site. This is a lower bound because it excludes float64 PPC draws and
  scratch. The planner must calculate the exact planned peak instead of relying
  on this estimate; measured aggregate RSS/PSS remains the broader safety gate.
- Prepared phase remains read-only/memory-mapped. Aggregate RSS/PSS is measured;
  per-process high-water RSS is not multiplied or treated as shared without
  evidence.
- A full off-diagonal edge-statistic table remains disallowed by default. Any
  future adaptive complete-table mode needs separate tests, a measured cost
  model, and proof that it fits the approved allocation bound.

## 10. Verification after every implementation package

1. Test-only commit exists and recorded genuine RED.
2. Implementation is a separate commit touching only assigned files.
3. Focused tests pass.
4. Complete `test_spike_lfp_summary.py`, PPC runtime, work-cache, runtime, and
   pipeline tests pass when affected.
5. Full command passes before the next package:

   `uv run pytest -q -p no:cacheprovider src/tests/neural_analysis`

6. Exact/tolerance equivalence is recorded, including near-threshold checks.
7. Functions and dataclasses document types, axes, shapes, units, returns, and
   failure behavior.
8. No external library, final schema, scientific fingerprint, or unrelated
   worktree file changed.
9. No CT026 scientific component or manifest was written by tests, profiling,
   or merge activity.
10. The package record names the lead Sol, Terra worker (or S8 command runner),
    optional scout, and Sol reviewer model/effort values; confirms their write
    permissions and file scopes; states whether read-only roles had enforced
    sandboxes or procedural restrictions; and includes the independent Sol gate
    disposition.
11. The lead independently reproduced the reported RED/GREEN results and
    confirmed that every spawned agent finished and is no longer running before
    the next package started; completed threads are closed when the runtime
    exposes that action.
12. The package's `docs/ppc_speedup_execution_log.md` entry is complete and its
    documentation-only commit is recorded before the next package starts.

## 11. Documentation handoff before implementation

The corresponding `docs/Tasks_neural.md` handoff has been drafted. Before S0,
commit it with this final plan as the committed documentation baseline required
by Section 3.1, and create `docs/ppc_speedup_execution_log.md` with:

- the pre-documentation scientific implementation HEAD;
- the documentation-baseline commit identity. Because a Git commit cannot
  contain its own hash, the baseline scaffold may mark this as `pending (this
  commit)`; record the exact hash in an immediate documentation-only follow-up
  commit before S0 edits begin;
- the implementation authorization and any separately authorized CT026 scope;
- one S0-S8 entry containing the package record required by Section 10.

Verify that the `Tasks_neural.md` handoff continues to:

- record the actual S0 RED state at the approved HEAD;
- link this document as the detailed PPC speedup execution plan;
- replace the immediate 1/2/4/8 benchmark instruction with S0-S8 sequencing;
- describe S0 as the next planned package pending separate implementation
  authorization, and remove any stale wording that calls WP5C-5 already
  authorized for implementation;
- keep WP5C-6, WP10-WP13, the preview gate, and the separate 1,000-shuffle
  authorization unchanged.

Completion of S8 should append measured timings, union ratios, selected worker
count, memory evidence, and revised projections to both documents. Historical
measurements remain labeled as measurements of the prior kernel.
