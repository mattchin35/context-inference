# Exact PPC speedup implementation plan

Status: complete. The implementation plan was approved by the user on
2026-09-09, S0-S7 were completed by 2026-09-18, and the separately authorized
work-only S8 CT026 benchmark completed on 2026-09-18. S8 selected eight workers
for the benchmarked CT026 production workload. It did not publish a scientific
component, manifest, preview, or `spike_phase.npz`. The separate 100-shuffle
preview and 1,000-shuffle final scientific runs remain unstarted and require
their own launch decisions. The approved S6 metadata-only 100/1,000-shuffle
schedule-union calculation was completed without reading phase values or spike
trains. The
Sol-orchestrator/Terra-worker
execution specification in Section 3.1 was added at the user's request on the
same date, audited against the current host on 2026-09-10, and is subject to
those same CT026 authorization boundaries. On 2026-09-10 the user approved plan-only
clarifications that unify PPC/histogram exact-sample semantics, make planned
private-memory accounting explicit, bind actual derived schedule identities,
clarify S0 executor lifecycle, reserve `xhigh` reasoning for S1-S4 and S7,
freeze the kernel allocation-estimator interface, add a 12 GiB aggregate
preflight, and distinguish S7 grouped-worker RED tests from existing single-job
regressions. At that time those clarifications did not authorize implementation
or CT026 computation; the later S0-S8 authorizations and completion records
supersede that historical gate.

This plan turns `docs/ppc_speedup.md` into test-first work packages. It is kept
separate from `docs/Tasks_neural.md` because the change is a substantial,
focused replacement for the remaining WP5C-5 sequencing. After approval,
`Tasks_neural.md` should receive only a short status/link update and retain the
broader LFP-summary roadmap.

## 1. Current state and recommendation

Historical starting state, inspected on 2026-09-10:

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

Completion update, 2026-09-18:

- S0-S7 implemented the grouped exact executor, one-pass payload integration,
  bounded resumable checkpoints, and invariant 1/2/4/8-worker execution.
- S8 benchmarked 64 rate-stratified units in eight unit blocks for condition
  `incorrect`, site `PFC`, all three epochs, 249 trials, 50 frequencies, and
  100 shuffles. Twelve fresh runs were measured without checkpoint resumes.
- Median wall times for 1/2/4/8 workers were 820.833, 471.556, 293.419, and
  211.397 seconds. Median scheduled-edge throughput was 91.005, 158.412,
  254.585, and 353.363 edges/s. Median aggregate PSS was 1.718, 2.291, 2.646,
  and 3.352 GiB.
- Eight workers were the smallest count within 90 percent of the best measured
  throughput while remaining well below the 16 GiB aggregate-memory gate. The
  library-wide default remains conservative; eight workers are the preferred
  CT026 production configuration, subject to the existing exact preflight.
- Exact benchmark-workload plans contain 74,700 scheduled, 61,586 independent,
  and 43,301 site-qualified union edges at 100 shuffles; at 1,000 shuffles the
  counts are 747,000, 182,056, and 61,752.
- The selected eight-worker 1,000-shuffle engineering estimate is 26.4-35.2
  minutes for this representative benchmark only. Full 427-trial/273-unit plan
  attempts were stopped after 10 minutes at 100 shuffles and 30 minutes at
  1,000 shuffles. No complete-component runtime is claimed from the
  representative estimate.
- The retained evidence is under
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_ppc_s8_2026-09-18T15-06-33Z`.

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

## 3. Approved decisions

The initial choices below were approved on 2026-09-09 and the explicit
clarifications recorded in this revision were approved on 2026-09-10.
At approval time, implementation required a separate user request. Those
requests were subsequently granted for S0-S8; the rule remains that a plan
alone is not execution authority for the pending preview or final run.

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
   This exact canonical-grid rule also governs representative histograms. The
   current histogram helper's `isclose(..., atol=1e-12)` behavior is an
   inconsistency to correct, not a second sampling contract to preserve.
3. **Preserve the existing epoch-specific schedules.** Before, after, and whole
   keep the current `_ppc_schedule_seed(...)` results. A whole schedule consumes
   the sum of before/after edge statistics for each edge in that same whole
   schedule. Standalone before/after schedules consume only their own segment.
4. **Require exact inferential decisions outside mathematical ties.**
   Identities, schedules, validity, counts, eligibility, permutation counts,
   and non-tied exceedance, p-value, q-value, and significance decisions must
   match the reference. Floating metrics may use the tight tolerances in
   Section 7. The optimized source-wise reduction is deterministic but need
   not reproduce the legacy concatenate-and-sum bit pattern when observed and
   shuffled PPC are mathematically tied. No tolerance is added to the
   production comparison: each implementation continues to apply raw
   ``null_ppc >= observed_ppc`` to its own deterministic values. This narrow
   exception was approved on 2026-09-14 after an identical-spike-train fixture
   exposed a one-to-three-ULP ordering difference.
5. **Use an explicit planned-allocation limit.** Add execution-only integer
   `maximum_worker_allocation_bytes = 2 * 1024**3` (2 GiB). It limits
   the conservatively estimated concurrently live private NumPy arrays in any
   process: the serial or parallel parent as well as each active worker. It
   does not enter scientific component identity and does not claim to measure
   shared mmap residency. The planner reports retained planning and summary
   arrays, observed and null computation stages, and their conservative
   lifetime-based private peaks separately. Also add execution-only integer
   `maximum_aggregate_allocation_bytes = 12 * 1024**3` (12 GiB) as a
   conservative preflight ceiling for planned arrays across the parent,
   active workers, and one shared prepared-worker input mmap set. For serial
   execution, the existing `shared_phase_mmap_bytes` field remains the phase
   tensor plus validity tensor. For grouped parallel execution, that legacy-
   named field is the exact sum of six read-only arrays counted once: phase,
   phase validity, relative time, stable trial rows, packed spike times, and
   int64 unit-by-trial spike offsets. The remaining 4 GiB of the 16 GiB
   measured gate is reserved for Python/process overhead, filesystem buffers,
   and estimation error. Neither execution field enters scientific component
   identity.
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

1. **Lead Sol orchestrator: `gpt-5.6-sol`, `high`.** One primary/root agent
   owns requirements, package ordering, user communication, authorization
   gates, worktree inspection, worker prompts, RED/GREEN verification, and git
   commits. It is the only agent allowed to declare a package complete. It must
   independently inspect diffs and test output rather than accepting a worker's
   summary as proof. It has repository write authority for staging the reviewed
   package paths and creating the required test-only, implementation, and
   documentation-only package-record commits, but it does not author or repair
   package source/tests; accepted edit findings return to the Terra package
   worker.
2. **Sol gate reviewer: `gpt-5.6-sol`, `high` or `xhigh` as assigned in the table
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

The lead stays at `high`; routine coordination, worktree checks, test execution,
commits, and package-log updates do not require `xhigh`. The independent Sol
reviewer receives `xhigh` only for the high-risk packages listed below. This
adds deeper review where numerical, identity, transaction, or concurrency
mistakes would be expensive without making it the default for the complete
implementation.

`low` is not used because even the read-only tasks require scientific-repository
context. `medium` is restricted to optional, narrowly scoped Terra scouting.
Terra `high` is the default for clear implementation packages, and Terra
`xhigh` is required for numerical kernels, cross-condition identity, restart,
or concurrency work. Neither `max` nor `ultra` is assigned anywhere in this
plan. Do not treat the ChatGPT Ultra
intelligence level, its proactive-delegation behavior, and a host-exposed
`reasoning_effort="ultra"` value as interchangeable. If a scientific or
architectural discrepancy cannot be resolved at the assigned level, stop and
ask the user; any move to `max` or `ultra`, or any change in agent topology,
requires a documented plan revision and explicit approval.

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
`high` reasoning and that explicit `gpt-5.6-terra`/effort overrides are
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
uses `model="gpt-5.6-sol"`, `reasoning_effort="xhigh"`, and a distinct task name;
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
| S1 | `xhigh` | `xhigh` | Interpolation geometry, exact-sample boundaries, dtypes, axes, and segmented sums. |
| S2 | `xhigh` | `xhigh` | Whole-from-halves scientific equivalence, cross terms, eligibility, and histogram semantics. |
| S3 | `xhigh` | `xhigh` | Stable physical identities, union planning, deterministic batching, and allocation arithmetic. |
| S4 | `xhigh` | `xhigh` | Grouped execution, bounded memory, checkpoint fingerprints, resume, and transaction failure behavior. |
| S5 | `high` | `high` | Minimal payload-loop integration while preserving schema, seeds, exemplar behavior, and public entry points. |
| S6 | `high` | `high` | Scalar-only profiling, metadata-only CT026 inspection, and no unauthorized scientific computation. |
| S7 | `xhigh` | `xhigh` | Spawn/failure/cancellation semantics, shared mmap behavior, deterministic parent publication, and memory safety. |
| S8 | `high`, command execution and evidence collection only | `high`, benchmark interpretation and recommendation | Respect separate authorization; no source edits; select worker count only from measured correctness, throughput, and memory evidence. |

Every S1-S7 package receives an independent Sol review of the stable GREEN
diff. A pre-commit test-design review is additionally mandatory for S1-S4 and
S7, whose omitted cases could permit numerical, identity, restart, or
concurrency drift; it is optional for S5-S6 when the lead identifies comparable
risk. The same reviewer may perform both gates if it retains no write access.
S0 uses only the final gate because its RED tests already exist. S8 has no Terra
package writer: its Terra assignment may execute only the explicitly authorized
benchmark commands and collect raw outputs, with no repository edits. The lead
and `high` Sol reviewer interpret the stable S8 evidence independently.

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
  same Sol/`high` requirement, re-read this plan and the package record, inspect
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
- Exposes pure
  `estimate_segmented_kernel_allocation(*, source_trial_spike_count,
  edge_source_trial_position, frequency_count) -> KernelAllocationEstimate`
  so the runtime planner does not guess kernel temporaries.
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
- Combines the kernel byte estimate with explicitly shaped job accumulators and
  calculation scratch to return `PPCAllocationEstimate`, enforce the 2 GiB
  process-private limit, and preflight the 12 GiB aggregate-array ceiling.
- Exposes internal `plan_grouped_ppc_component(...)` and
  `execute_grouped_ppc_component(...) -> PPCComponentExecutionResult`. The
  planner is pure and performs no phase sampling or file I/O. The executor
  receives `config`, `execution`, complete prepared phase/spike inputs,
  `work_root`, and an optional progress callback; it constructs the unchanged
  schedules internally rather than accepting a replacement schedule.
- Processes one site and one unit block at a time.
- Workers, when re-enabled, open one shared, read-only prepared-worker input
  set: phase, validity, relative time, stable trial rows, packed spike times,
  and unit-by-trial spike offsets. Tasks carry only bounded site/unit identity
  and plan data, not private copies of those arrays.

`lfp_summary_runtime.py`

- Replaces the condition/site/epoch call loop in
  `_build_spike_phase_payload(...)` with one grouped executor call.
- Continues to own final public payload construction, exemplar selection,
  final component validation, and post-commit cleanup.
- Stops resampling all 50 frequencies for representative histograms; it
  consumes histogram counts produced from observed same-trial sampling.

`lfp_summary_models.py`

- Adds only the two approved execution-only worker and aggregate memory-limit
  fields.
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

`KernelAllocationEstimate` (frozen dataclass)

- Is returned by `estimate_segmented_kernel_allocation(...)` and reports
  nonnegative integer `geometry_bytes`, `segmented_edge_statistics_bytes`,
  `gather_temporary_bytes`, and `planned_kernel_peak_bytes`.
- `source_trial_spike_count` is an integer, nonnegative array with axes
  `(source_trial, unit, segment=2)` in before/after order.
  `edge_source_trial_position` is an integer shape `(edge_in_block,)` index
  into its source-trial axis, and `frequency_count` is a positive integer.
- The estimate counts geometry retained for the bounded unit block and every
  named gather/interpolation/validity array concurrently required by the
  supplied edge block. The kernel documents each temporary's dtype, shape, and
  lifetime; the estimator uses `dtype.itemsize * product(shape)` rather than an
  unexplained multiplier.
- Geometry accounting includes the owned int64 scalar physical source-trial
  identity for every source geometry, in addition to spike interpolation arrays
  and group offsets. Segmented-statistics accounting includes the owned int64
  source and target identity vectors in addition to sums and counts. Therefore,
  for `S` source geometries and `E` edges,
  `geometry_bytes` includes `8 * S`, and
  `segmented_edge_statistics_bytes` equals
  `48 * E * unit_count * frequency_count + 16 * E`.
- `planned_kernel_peak_bytes` is the maximum of documented concurrently live
  kernel array groups. Invalid axes/counts and checked-arithmetic overflow raise
  `ValueError`; the helper performs no allocation, sampling, or I/O.

`PPCJobPlan` (frozen dataclass)

- Identifies one condition/site/epoch result cell.
- Stores condition/site/epoch positions and names, stable selected trial rows,
  the unchanged local derangement schedule, its stable physical-edge mapping,
  and which segment expression (`before`, `after`, or `before + after`) it
  consumes.
- Stores `base_ppc_seed` separately from the actual derived `schedule_seed`,
  plus the condition/site/epoch derivation identities, schedule shape, and exact
  schedule fingerprint. Grouped checkpoint identity binds all of these values.
- Condition-local position zero is never used as a cross-condition edge key.
- Public construction validates exact epoch/derivation/seed provenance, true
  row-wise derangements, stable-edge translation, union positions, schedule
  shape, and schedule fingerprint. Public input arrays are defensively copied
  and made read-only. The planner's trusted construction path instead transfers
  already-owned generator/mapping buffers without schedule-sized copies or
  validation temporaries after allocation preflight.

`PPCComponentPlan` (frozen dataclass)

- Stores jobs in canonical site-major, condition-major, then epoch-major order.
- Stores the sorted unique site-qualified physical-edge union as matching
  int64 site/source/target vectors; an edge shared across sites remains two
  physical work items.
- Stores deterministic per-site condition batches and exact scheduled,
  independent, and union edge counts plus dimensionless saturation and reuse
  metrics. Empty/singleton plans use finite zero metrics.
- Public construction validates that the union, counts, metrics, and batch
  coverage are derived exactly from its jobs and owns read-only array inputs.

`PPCAllocationEstimate` (frozen dataclass)

- Reports nonnegative integer byte counts for `job_accumulator_bytes`,
  `observed_trial_statistics_bytes`, `observed_gather_temporary_bytes`,
  `kernel_working_bytes`, `geometry_bytes`, `planner_array_bytes`,
  `condition_membership_bytes`, `source_trial_spike_count_bytes`,
  `planning_working_bytes`, `planned_planning_private_bytes`,
  `summary_assembly_bytes`, `worker_plan_bytes`, `worker_summary_bytes`,
  `checkpoint_block_bytes`,
  `planned_computation_private_bytes`, `planned_parent_private_bytes`,
  `planned_worker_private_bytes`, `shared_phase_mmap_bytes`, and
  `planned_aggregate_array_bytes`, plus nonnegative integer
  `active_worker_count`. `kernel_working_bytes` contains segmented-edge and
  null-gather temporaries but excludes separately reported retained geometry.
  `shared_phase_mmap_bytes` retains its public name for compatibility. It is
  phase plus validity for serial execution; for grouped parallel execution it
  is the exact six-array shared worker-input set defined in Section 3, counted
  once in the aggregate rather than once per worker.
- Job-accumulator accounting includes, for every concurrently resident
  job/shuffle/unit/frequency cell, complex128 vector sums, int64 counts, float64
  PPC draws, and one float64 calculation scratch array: 40 bytes per cell with
  the approved dtypes.
- Segmented edge statistics contribute 48 bytes per
  edge/unit/frequency cell (two segments times one complex128 sum and one int64
  count), plus 16 bytes per edge for the owned int64 source/target identities.
  Geometry includes an owned int64 source identity per source geometry, two
  int64 indices, one float64 weight, two Boolean masks per spike, and int64
  group offsets. Any additional gather temporary is named and calculated from
  its documented dtype and shape rather than hidden in a multiplier.
- Observed-stage memory is retained geometry plus per-stable-trial observed
  sums/counts/histograms and observed-source gather temporaries. Null-stage
  memory is retained geometry plus full-shuffle job accumulators and bounded
  kernel working arrays. `planned_computation_private_bytes` is the larger of
  those stages, never their sum.
- Planning has a separate lifetime from summary execution. Let `M` be the
  Boolean `(trial, condition)` gated-membership bytes, `Q = 16 * trial * unit`
  be the full int64 before/after source-count table, `D` be the retained int64
  selected-row and schedule draft bytes, and `P` be final
  `planner_array_bytes`. The planner centrally derives a conservative
  construction bound `A = 8 * (2*T + T*(T-1)) + 16*T*B`, where `T` is the
  full-trial count and `B` is the bounded unit-block size. Thus
  `planning_working_bytes = D + A` and
  `planned_planning_private_bytes = M + Q + max(P, D + A)`. Callers cannot
  underreport this bound. Scalar preflight occurs before membership, count,
  or schedule allocation; exact final-plan preflight occurs before stable maps
  are materialized. Membership and count tables are released before steady
  summary execution.
- Parent summary assembly uses all component units and jobs. Worker summary
  retention uses the current unit block and every result job at that site,
  while active-job counts size only the current condition batch's null
  accumulators. Parent and worker plan arrays are reported separately.
  `checkpoint_block_bytes` contains one site/unit-block summary plus the compact
  int64 site index and half-open unit bounds stored with that checkpoint.
- The private process peaks are sums of arrays whose lifetimes overlap and
  maxima across mutually exclusive planning, observed/null, and checkpoint
  stages. The serial steady-parent stage is `P + full summary +
  max(computation, checkpoint block)`. The parallel steady-parent stage is
  `P + full summary + checkpoint block`, while each active worker includes its
  site plan, site/unit-block result, and computation peak. Reported parent
  private memory is the larger of `planned_planning_private_bytes` and the
  applicable steady-parent stage.
- Checkpoints are published synchronously through an explicitly trusted
  no-copy writer path that accepts only read-only, non-object arrays. The
  existing public/default writer behavior remains defensive-copy. Resume loads
  and merges one owned, read-only checkpoint block at a time, releasing it
  before the next block, so neither cold publication nor warm resume retains a
  second checkpoint-sized array set outside `checkpoint_block_bytes`.
- `planned_worker_private_bytes` is the maximum private peak among tasks in the
  active submission window. `active_worker_count` is the smaller of requested
  workers and pending unit blocks; idle requested workers are neither charged
  nor spawned.
- `planned_aggregate_array_bytes` equals shared mmap bytes counted once plus
  the larger of the planning stage alone or the parallel/serial steady-parent
  stage plus `active_worker_count * planned_worker_private_bytes`. Planning
  never overlaps active workers. Shared mmap residency is not multiplied by
  worker count. Checked integer arithmetic rejects overflow.
- The planner rejects or deterministically reduces condition batches when a
  process or the aggregate plan would exceed its limit. Both constraints
  participate in batching; rejection occurs only when a singleton condition
  batch still fails. Preflight occurs before final stable-edge maps are
  materialized and before any process is spawned. The planner never silently
  lowers the requested worker count.

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
   histograms at the configured frequencies nearest 8 and 40 Hz using the same
   exact canonical-grid classification as PPC. A trial with a positive count in
   either segment counts once toward whole eligibility.
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
- PPC and representative histograms make identical exact/between/outside
  decisions. Spikes one representable float above or below a grid point, and a
  spike within `1e-12` but not exactly equal to it, remain between-sample cases.
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
- `estimate_segmented_kernel_allocation(...)` matches hand-calculated geometry,
  segmented-edge, named gather-temporary, and lifetime-peak bytes without
  allocating arrays; malformed counts/axes, Boolean integer arguments, and
  checked-arithmetic overflow fail with `ValueError`. The hand calculation
  explicitly includes the owned int64 scalar identity for every source
  geometry and both owned int64 edge-identity vectors.

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
  allocation tests verify every named category in `KernelAllocationEstimate`
  and `PPCAllocationEstimate`, the 40-byte job-cell accounting, the 48-byte
  segmented-edge-cell accounting, geometry arrays, explicitly named gather
  temporaries, lifetime-based process and aggregate peaks, shared mmap counted
  once, and checked-overflow behavior.
- Deterministic condition batching under the allocation limit changes neither
  output order nor values.
- The worker and aggregate allocation limits default to exactly 2 GiB and
  12 GiB, respectively; both round-trip as execution-only integers, reject
  Boolean/nonpositive values, and do not alter any final scientific component
  fingerprint. Deterministic batching satisfies the process limit; an unsafe
  requested worker count fails preflight before process creation.
- Empty/single-trial conditions retain observed semantics and do not request an
  invalid derangement.

### Runtime, restart, and integration

- Grouped summaries have exact final unit/condition/site/epoch/frequency axes.
- Checkpoint fingerprints bind condition memberships, stable trial rows, every
  schedule, segment definitions, phase/spike identity, kernel version, and
  execution settings.
- Every grouped job records the base PPC seed, actual derived schedule seed,
  derivation identities, schedule shape, and exact schedule fingerprint. Jobs
  that share a base seed but differ in derived schedule identity cannot resume
  one another.
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
- Representative histograms exactly match the current payload builder for
  inputs unaffected by its legacy near-grid `isclose` discrepancy. Adversarial
  near-grid inputs intentionally follow the S1 exact canonical-grid contract,
  and PPC/histogram accepted-sample counts agree.
- Existing final component schema, exemplar selection, post-commit cleanup, and
  pipeline entry points remain unchanged.
- Seeded synthetic end-to-end cache and plotting integration remains green.

### Parallelism and performance instrumentation

- Finish the existing bounded-spawn executor tests: successful shutdown,
  cancellation including the failed future, ordered results, and resumable
  prior yields.
- Worker counts 1/2/4/8 preserve all exact fields and tolerance-governed
  floating fields.
- Workers open one read-only prepared worker-input set and do not serialize or
  privately copy its six full arrays. The auxiliary parallel arrays are
  materialized by bounded direct mmap writes only after scalar preflight; they
  are not created for serial execution.
- Planned aggregate bytes count that shared mmap set once and combine it with
  the parent and every active worker. Unsafe worker counts fail before planner
  arrays, mmap materialization, or spawn, and
  measured aggregate RSS/PSS must still remain below 16 GiB.
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
these tolerances cannot conceal a lost/doubled spike. Inferential-equivalence
fixtures use deterministic, trial-distinct spike-time patterns and require a
minimum absolute null-versus-observed margin of ``1e-5`` for every comparison
whose exact decision is asserted. If floating differences change an
exceedance comparison or FDR decision outside that margin, the package fails
review even when each raw value falls within tolerance. For an exact
mathematical tie, the deterministic reduction order may select a different
side of the raw comparison than the legacy implementation; tests must identify
that case as a tie rather than conceal it with a production tolerance. This
tie policy was separately approved on 2026-09-14. A future change to make ties
canonical would be a scientific estimator change requiring separate approval.

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
Use the executor context manager for normal shutdown. Retain the current future
until `result()` succeeds; on failure, cancel that failed current future and all
submitted pending futures, request `shutdown(wait=True, cancel_futures=True)`,
and re-raise the original exception without submitting more work. Run the
focused file, affected PPC runtime tests, then the full neural suite. Do not run
a production worker benchmark or change the already-committed RED tests merely
to accommodate the current source.

### S1 - Uniform-grid geometry and segmented edge kernel

Files:

- new `src/neural_analysis/lfp_summary_ppc_kernel.py`
- new `src/tests/neural_analysis/test_lfp_summary_ppc_kernel.py`
- minimal shared-formula addition in `spike_lfp_summary.py` and focused tests if
  needed

Write all geometry/interpolation tests first, including exact, adjacent
representable-float, and within-`1e-12` nonexact cases shared by PPC and
representative histograms. Implement immutable geometry, segmented edge
statistics, `KernelAllocationEstimate`, and the frozen pure allocation-estimator
interface without runtime or cache changes.

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
  worker and aggregate allocation limits

Write planner, stable-identity, union, memory-estimate, and deterministic-
batching tests first. Bind the base seed, actual derived schedule seed,
derivation identities, schedule shape, and schedule fingerprint. Test every
named allocation category and conservative lifetime peak rather than only a
single total. This package builds plans only; it does not replace the production
component loop.

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

Write final-axis/schema, histogram equivalence, shared exact-sample semantics,
unchanged seed, unchanged exemplar, transaction, and end-to-end synthetic tests
first. Legacy-equivalence fixtures exclude the documented near-grid discrepancy;
adversarial near-grid fixtures require PPC and histogram counts to follow the S1
contract. Replace only the PPC job loop and duplicate observed histogram
sampling. Preserve public builder signatures.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_runtime.py src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py -k "spike_phase or ppc"`

### S6 - Profiling and schedule-union cost model

Files:

- `lfp_summary_ppc_profile.py` and its tests
- `lfp_summary_ppc_runtime.py` only for extracting a private
  `_record_grouped_representative_histogram(...)` seam around the existing
  representative-histogram update. The extraction must not change numerical
  behavior, axes, allocation ownership, or checkpoint semantics; all other
  grouped-runtime changes remain S7 scope.
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
The existing single-job `execute_ppc_blocks(...)` worker tests are frozen green
regressions after S0 and do not count as S7 RED evidence. S7 must add grouped-
executor tests whose genuine RED is caused by the grouped engine lacking
parallel dispatch, shared grouped inputs, aggregate preflight, or grouped
checkpoint ordering. The existing spawn/bounded-window semantics remain
binding. Implement workers across independent unit blocks, not conditions or
sites.

RED/GREEN command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_ppc_parallel.py src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py`

### S8 - Authorized representative benchmark and decision gate

Status: complete on 2026-09-18. No source implementation belonged to this
package. After explicit authorization, S8:

1. Run the work-only low/median/high representative CT026 job at 100 shuffles
   with one worker and warm prepared phase.
2. Compare old accepted serial versus grouped serial stage timings and outputs.
3. If parallelism remains worthwhile, run 1/2/4/8 workers on enough unit blocks
   to occupy every requested worker that passes the 12 GiB planned aggregate
   preflight.
4. Choose the smallest worker count whose median throughput is within 10% of
   the best measured throughput and whose aggregate memory stays below 16 GiB.
5. Re-project 100- and 1,000-shuffle complete-component runtime from measured
   union edges; do not reuse the old 47-51/90-100 hour projections.
6. Presented the results for review. WP5C-6 preview remains separately gated.

The benchmark used a realistic eight-block, 64-unit rate-stratified workload.
Scientific comparison against the accepted legacy overlap passed. Worker-count
plans and outputs were invariant: identities, schedules, counts, masks,
decisions, p-values, significance, and representative histograms agreed
exactly; tolerance-governed floating outputs agreed under the frozen policy.
The legacy q-value comparison contained 51 binary64 differences no larger than
`1.1102230246251565e-16` absolute and caused no tolerance-level mismatch or
decision change.

Eight workers are the preferred CT026 production setting because they were the
only measured count within 10 percent of the best throughput and their maximum
sampled aggregate PSS remained below 16 GiB. This does not change the universal
library default and does not bypass exact runtime preflight. The 100-shuffle
preview and 1,000-shuffle final run must be launched separately; neither is an
implicit continuation of S8.

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
  2 GiB process allocation gate applies to the parent or worker private peak,
  covering the explicitly named concurrently live accumulator, kernel-working,
  geometry, and gather arrays.
- Before serial execution or worker spawn, the planner verifies that shared
  mmap bytes counted once plus the parent-private peak plus every active worker
  peak do not exceed 12 GiB. Failure is side-effect-free and does not silently
  change worker count. Measured aggregate RSS/PSS must remain below 16 GiB,
  leaving at least 4 GiB beyond planned arrays for runtime overhead and error.
- At CT026 defaults, 27 jobs (9 conditions x 3 epochs), 1,000 shuffles, 8 units,
  and 50 frequencies require about 247 MiB for complex128 sums plus int64 counts
  for one site. This is a lower bound because it excludes float64 PPC draws and
  scratch. The planner must calculate the exact planned peak instead of relying
  on this estimate; measured aggregate RSS/PSS remains the broader safety gate.
- Prepared phase remains read-only/memory-mapped. Aggregate RSS/PSS is measured;
  per-process high-water RSS is not multiplied or treated as shared without
  evidence. Conversely, private worker allocation is multiplied by active
  worker count in preflight even when phase storage is shared.
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

## 11. Historical documentation handoff before implementation

This pre-S0 handoff is complete and retained for audit. The original handoff
and execution-log scaffold are committed in `73e2d9f`,
with its exact identity recorded in `a1d23f3`. Before S0:

- Commit the 2026-09-10 plan clarification, synchronized `Tasks_neural.md`, and
  execution-log note as documentation only. Record that clarification commit's
  exact hash in an immediate documentation-only follow-up commit; neither
  commit may include source, tests, generated files, or unrelated worktree
  changes.
- Record the active HEAD and complete `git status --short` inventory in the S0
  package record before assigning an editor. Treat the existing `src/main.py`
  modification and untracked files as unrelated user work; do not clean, stage,
  overwrite, or summarize them away. Stop if the inventory changes unexpectedly.
- Confirm from session/host metadata that the lead is `gpt-5.6-sol` at `high`
  and the exact Terra/Sol effort overrides in Section 3.1 are available.
- Record explicit implementation authorization separately from any CT026
  authorization. Documentation approval alone still does not authorize S0.
- Retain one S0-S8 execution-log entry containing every package record required
  by Section 10.

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

## 12. Post-S8 integration handoff

S8 closes the PPC performance redesign. Subsequent packages must consume the
grouped executor rather than reopen its scientific or worker architecture.

- CT026 production configuration should request eight workers. The general
  `PPCExecutionConfig` default remains conservative, and exact preflight may
  reject an unsafe requested count but must not silently substitute another.
- The approved post-S8 sequence is WP10 composed production dependencies, WP12
  complete PPC plots/reporting, the standalone launcher, and then the
  user-invoked 100-shuffle preview. WP11 follows the preview infrastructure and
  is not required for that standalone preview.
- WP11 supports exactly one active ProbeA or ProbeB population per run. Both
  choices use good/MUA units on good inside-brain channels; a combined
  cross-probe population is out of scope.
- WP13 is an optional absolute-amplitude feature, not a PPC optimization. The
  current CT026 preview has an empty absolute-threshold list, so deferring WP13
  leaves that default result unchanged. Every nonempty request must be rejected
  before computation until WP13 is implemented; warning and continuing is not
  permitted.
- WP5C-6 and the later 1,000-shuffle run require a tested standalone resumable
  command-line launcher. The launcher runs outside Codex, records a timestamped
  run directory and log, supports dry-run/new/resume modes, prints its exact
  resume command, and preserves work without publishing a false final marker
  after interruption.
- The launcher creates its analysis-run identity and resume command before full
  planning. Deterministic planning is not checkpointed and may repeat after an
  interruption. Compatible prepared-phase and PPC work still resumes by exact
  identity after it exists.
- Retain exact PPC work through component, manifest, plot, detailed report,
  log, and summary validation. Exact-fingerprint cleanup is the last successful
  step; a report failure preserves resumable work.
- The launcher must run 100 and 1,000 shuffles as separate explicit commands
  and directories. It must never promote a completed preview automatically;
  1,000-shuffle execution requires an additional explicit final-run flag.
- Full-component planner time is an unresolved operational measurement. Record
  it separately during the 100-shuffle preview and do not substitute the S8
  64-unit 26.4-35.2 minute estimate for the complete 273-unit run.
