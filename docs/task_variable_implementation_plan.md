# Task-Variable Decoding Implementation Plan

**Status:** Implementation authorized; WP1-WP10D are complete, including the
single-session and batch CLIs, saved-result webapp, portable example, user/
maintainer documentation, seeded synthetic end-to-end gate, and the separately
approved CT026 `rewards_in_block` table preparation. WP10B measured the bounded
CT026 workload and exposed categorical convergence failures under the original
100-iteration ceiling. The user approved a local revision-6 repair and local
validation, then selected cluster execution as a required implementation path
because approximately 15 minutes per session does not scale well across many
sessions. WP11 single-session Slurm support and WP13 bounded per-session job
arrays are therefore required packages. Experimental-data transfer and real
scheduler submissions retain their separate exact-action gates below.

**Scientific contract:** `docs/task_variable_spec_v5.md` plus the active
normative amendment `docs/task_variable_spec_v6.md`.

## Live handoff snapshot

**Snapshot date:** 2026-10-07

**Current phase:** WP0-WP11 are complete. The first v2 bounded run proved the
convergence repair but exposed a competing-resume lifecycle defect during
diagnosis. The approved repair is committed through `77ff461`, pushed, and
tracked-clean. The clean bounded v2 rerun
`task_variable_decoding_2026-10-07T19-29-08Z` passed its one-shot lifecycle,
result, and convergence validation. The approved local eight-target
categorical stress gate also passed with all 9,600 cells valid. WP10 is now
complete. The exact WP11 single-session Slurm resource block and tests-first
architecture are implemented: RED is `a3dcdb9`, its handoff is `e91fa5f`, and
GREEN production is `b9d8645`. WP12 one-session cluster validation and WP13
per-session job arrays follow. WP12's deterministic synthetic fixture has been
transferred, passed matching local/cluster dry runs, and completed as Slurm job
`30984881`. The generated-bytecode source-gate repair is RED `849c7ea` and
GREEN `db50e1a`; the post-smoke accounting/summary repair is covered by tests
`86fc413` and `5386a99` with GREEN `a1763bc`. Repaired one-shot status and
independent result validation accepted the immutable synthetic run; acceptance
is recorded at `484e784`. The full fixed-mode CT026 config, grouped input
transfer, and matching local/cluster dry runs are now complete. The separately
approved full CT026 run was accepted as Slurm job `30985392` at exact commit
`c086bdf`. WP12 now waits for one later user-requested completion inspection;
there is no monitoring or automatic resume.
WP4 grouped-modeling tests are
committed at `6ee6355`, with fixed-mode supplements at `c85a00c` and
`fab38b8`, and tuned-provenance coverage at `c1b183c`. Fixed-mode production
is committed at `a8787aa`; complete tuned production and provenance are
committed at `b844f26` after independent approval. WP5A base saved-results
tests are committed at `4a5664a`; independently approved dynamic-schema,
semantic, and frozen-control corrections are committed at `bc628c4` and
`2c5f086`. Real-state RED coverage is committed at `dedde7d`; later transform,
shared-split, exact-PCA, and sparse-tuning corrections are committed through
`c7c2912`. WP5A production is committed at `1bd0f69` after independent
approval. The independently approved WP5B lifecycle/CLI RED suite is committed
at `449f7c9`, its production-review supplement is committed at `4215c82`, the
final tests-first corrections are committed through `77f2cc8`, and WP5B GREEN
is committed at `8b9232d` after independent approval. The WP6 tests-first
contracts are committed through `827989c`, and WP6 GREEN is committed at
`7d6b4a8`. WP7 plotting/webapp contracts and review supplements are committed
through `0c6ce07`, and WP7 GREEN is committed at `ef7ad55`. WP8 tests are
committed through `a1c1f5c`, the dry-run JSON correction is `b2b3dde`, and the
documentation/example GREEN commit is `8b92c4f`. No further experimental-data
mutation, transfer, or real cluster action is authorized by this planning
decision. The already launched bounded WP10 rerun may finish, and cluster
source/tests may begin only after the local gates and the tests-first plan
below.
WP9's characterization-only synthetic integration gate is committed at
`9ea0109`. WP9A's permission regression and fix are committed at `b322427` and
`cdb04d9`; the CT026 table and its exact backup passed post-write validation.
WP10A resource-measurement RED contracts are committed at `fb0278a`, the
schema-consistency supplement at `cc0cf7d`, and GREEN production at `1153115`.
The WP10B bounded v1 run completed, validated, and recorded 2,400 estimator
calls, 203.715 seconds wall time, 984,047,616 bytes peak RSS, and 53,932,914
output bytes. It also recorded 687/1,200 unavailable `current_action` cells,
all caused by convergence warnings at the frozen 100-iteration ceiling.

**Repository state at this snapshot:**

- branch: `refactor`;
- exact CT026 scheduled execution commit and cluster HEAD:
  `c086bdf55e08d062bec0842ac453af8dd0d29710`;
- prior local documentation handoff commits: `39ba0db`, `3104c59`, `38b2854`,
  `fda0399`, `ceb9b28`, and `5c97d9a`;
- WP4 is complete and its handoff is committed at `01f8b0f`; WP5A tests and
  production are complete through `1bd0f69`, including the final sparse tuned
  transform-evidence review; WP5B single-session lifecycle/CLI production is
  complete through independently approved GREEN commit `8b9232d`; WP6 batch
  planning/execution is complete through GREEN commit `7d6b4a8`; WP7 plotting
  and the read-only saved-results webapp view are complete through GREEN commit
  `ef7ad55`; WP8 documentation and portable examples are complete through
  GREEN commit `8b92c4f`; WP9 synthetic integration is complete through
  characterization commit `9ea0109`; WP9A table preparation is complete after
  permission-preservation commits `b322427` and `cdb04d9`; WP10A measurement
  instrumentation is complete through GREEN `1153115`; and
- all pre-existing untracked files remain outside this plan's ownership.

**Completed planning evidence:**

- `src/neural_analysis`, `docs/SoftwareDesign.md`, and revision 4 were audited;
- revision 5 records the resolved scientific/data contracts;
- CT026 2026-08-03 input presence and small-table/alignment metadata were
  inspected read-only;
- the implementation architecture, tests-first sequence, saved-run contract,
  integrated webapp view, and benchmark strategy are drafted below; and
- the offline contract now includes unattended local launch, one-shot
  read-only status inspection, and an unattended single-session Slurm path. It
  explicitly does not require active Codex monitoring or polling; and
- cluster planning uses `src/shell_scripts/hpc_ppc.sh` as the site
  execution reference, manual non-destructive `rsync` for data/results, and
  benchmark-derived task-decoding resources rather than copied PPC values; and
- a second correctness pass resolved tuned-transform reuse, error boundaries,
  portable path/source identity, comparable benchmark threading, and
  measurement-based memory/resource decisions; and
- a third pass cross-checked the revised contracts against the current loaders,
  augmentation helpers, run lifecycle, and resource handoff; and
- a fourth readiness correction synchronized the committed handoff, reused the
  existing cluster-metadata filter, froze target/fold vocabularies and the PCA
  component-limit rule, and completed temporary-file cleanup semantics; and
- a fifth readiness correction froze exact configured channel-ID validation,
  added the missing cleanup-failure test, and locked the valid one-usable-unit
  PCA edge against the older helper's stricter minimum.

**Next exact action:** leave job `30985392` unattended. In a later
user-requested task, run the recorded one-shot status command and inspect the
durable state/log/result once. Do not poll, automatically resume, or update the
cluster checkout away from the job's exact commit while execution may depend
on it.

### Authority order

When resuming, use this order:

1. `docs/task_variable_spec_v5.md` owns the base scientific definitions and
   defaults; `docs/task_variable_spec_v6.md` is its active normative amendment.
2. This document owns implementation order, file ownership, tests, agent
   assignments, documentation deliverables, and live status.
3. `AGENTS.md` and `docs/SoftwareDesign.md` govern TDD, readability, data
   contracts, dependencies, and user approval.
4. The top live snapshot and package records in this document own execution
   state. Historical chat summaries are not a substitute.
5. Git history and independently reproduced commands are evidence; an agent
   summary alone is not.

### Work-package state

| Package | State at snapshot | Next gate |
| --- | --- | --- |
| WP0 documentation approval | Complete at pushed `b69c70a` | None |
| WP1 behavioral feature | Complete: RED `acf24ae`, GREEN `8ea4d24` | None; CT026 backfill remains separately gated at WP9A |
| WP2 configuration and targets | Complete: RED from `3ccae6c`, review tests through `f13d31b`, GREEN `2899502` | None |
| WP3 activity loading and coverage | Complete: RED `6e45ebf`, GREEN `d668250` | None |
| WP4 grouped modeling | Complete: fixed GREEN `a8787aa`, tuned GREEN `b844f26` | None |
| WP5 results and session pipeline | Complete: WP5A GREEN `1bd0f69`; WP5B GREEN `8b9232d` after tests through `77f2cc8` | None |
| WP6 batch runner | Complete: RED `9727ec8` plus reviews through `827989c`; GREEN `7d6b4a8` | None |
| WP7 plotting and webapp | Complete: RED `c027f2c` plus reviews through `0c6ce07`; GREEN `ef7ad55` | None |
| WP8 documentation and examples | Complete: RED `c1cda3b`, corrections through `a1c1f5c`, JSON fix `b2b3dde`, GREEN `8b92c4f` | None |
| WP9 synthetic integration | Complete: characterization-only test gate `9ea0109` | None |
| WP9A CT026 augmented-table preparation | Complete: mode RED `b322427`, fix `cdb04d9`, validated backup and table | None |
| WP10A measurement instrumentation | Complete: RED `fb0278a`, supplement `cc0cf7d`, GREEN `1153115` | None |
| WP10B CT026 preflight/benchmark | Complete: v1 run validated; 2,400 calls measured; 687 categorical convergence failures exposed | Superseded by revision-6 evidence |
| WP10C revision-6 convergence repair | Complete: RED `ba47795`, GREEN `6f9c721`; first v2 run made all 2,400 cells valid | Clean lifecycle evidence remains the WP10 gate |
| WP10D competing-owner lifecycle repair and local acceptance | Complete: lifecycle repair through `77ff461`; clean 2,400-cell bounded run and 9,600-cell categorical stress run accepted | None |
| WP11 single-session cluster path | Complete: RED `a3dcdb9`, handoff `e91fa5f`, GREEN `b9d8645`; mocked and local tests only | None; real execution belongs to WP12 |
| WP12 CT026 one-session cluster validation | In progress: full CT026 job `30985392` accepted at exact commit `c086bdf` after matching dry runs | One later user-requested status/log/result inspection; no monitoring |
| WP13 bounded cluster batch array | Required after WP12 acceptance; source not started | Implement tests-first per-session concurrency; authorize any real array separately |

### Resume checklist

A new or returning Sol supervisor must:

1. read revision 5, its revision-6 amendment, this complete plan, `AGENTS.md`,
   `docs/SoftwareDesign.md`, and the most recent package record;
2. record current branch, HEAD, `git status --short`, staged diff, and unstaged
   diff without modifying either;
3. distinguish user/pre-existing changes from package-owned changes and stop
   for user direction if ownership is uncertain;
4. reproduce the last claimed RED/GREEN command before accepting it as a gate;
5. verify the current user authorization, especially for implementation,
   CT026 computation, or Slurm use;
6. resume at the first incomplete gate, not at the beginning of a completed
   package; and
7. update this snapshot before delegating or editing.

If a Terra worker or Sol reviewer is interrupted, its partial report is not a
completed gate. The supervisor re-audits the bounded diff and reproduces the
last claimed command before resuming the same worker or assigning a replacement.
Unknown edits are preserved and escalated; they are never reset or absorbed
into a package.

### Live update record

After every tests-only commit, implementation commit, documentation gate,
benchmark, interruption, or user approval, the Sol supervisor updates this
section and adds a concise package record containing:

- date/time, package and gate;
- supervisor, Terra worker, and independent reviewer model/effort;
- branch, starting/ending HEAD, and worktree ownership;
- exact files changed;
- exact commands and RED/GREEN outcomes;
- commit IDs, if any;
- scientific/configuration decisions and unresolved risks;
- real-data or external actions taken (normally none);
- output/run paths and measured performance, when authorized; and
- the one exact next action and authorization required.

Do not rewrite earlier package evidence to make a later run look cleaner.
Correct errors with a dated correction note.

Use this template for each appended record:

```text
#### YYYY-MM-DD HH:MM - WPx gate
- State:
- Authorization:
- Sol / Terra / reviewer:
- Start HEAD / end HEAD:
- Owned files:
- RED command and result:
- GREEN/regression commands and results:
- Commits:
- Real-data or external actions:
- Findings and unresolved risks:
- Exact next action:
```

#### 2026-10-06 - WP0 readiness review

- State: implementation plan ready for user review; WP0 remains open until the
  accepted documentation is committed and the user separately requests
  implementation.
- Authorization: documentation changes only.
- Sol / Terra / reviewer: primary Codex review; no Terra worker or independent
  implementation reviewer was used.
- Start HEAD / end HEAD: `740aceac8b63d01e9529bb7c8842b8afe29221db` /
  `740aceac8b63d01e9529bb7c8842b8afe29221db`.
- Owned files: `docs/task_variable_implementation_plan.md` only. Concurrent
  neural-regression documentation changes belong to another user task and were
  excluded.
- RED command and result: not applicable; no tests or implementation were
  authorized.
- GREEN/regression commands and results: documentation-only checks passed:
  `git diff --check -- docs/task_variable_implementation_plan.md`; Markdown
  fence count was even; revision-5 specification remained unchanged.
- Commits: none; the WP0 plan revision is an uncommitted working-tree change.
- Real-data or external actions: none.
- Findings and unresolved risks: the readiness findings were resolved by
  freezing CT026 augmented-table preparation, neural tensor row mapping,
  scientific/execution and file/code identity, shared prepared-run execution,
  single-writer resume behavior, atomic publication, inner grouped CV tests,
  memory/resource thresholds, and non-destructive transfer scope. No known
  implementation-contract blocker remains.
- Exact next action: user reviews the plan; if accepted, commit the WP0
  documentation and record its HEAD before any separately authorized WP1 work.

#### 2026-10-06 - WP0 second readiness correction

- State: the specification and plan were corrected after a second
  correctness/completeness/conciseness audit; WP0 remains documentation only.
- Authorization: documentation changes only.
- Sol / Terra / reviewer: primary Codex review; no worker was used.
- Start HEAD / end HEAD: `740aceac8b63d01e9529bb7c8842b8afe29221db` /
  `740aceac8b63d01e9529bb7c8842b8afe29221db`.
- Owned files: `docs/task_variable_spec_v5.md` and this plan only.
- RED command and result: not applicable; no implementation was authorized.
- GREEN/regression commands and results: `git diff --check --
  docs/task_variable_spec_v5.md docs/task_variable_implementation_plan.md`
  passed; both Markdown fence counts were even; stale superseded terms were
  searched explicitly.
- Commits: none.
- Real-data or external actions: none.
- Findings and unresolved risks: corrected per-fold transform reuse in tuned
  CV, declared-versus-unexpected failure handling, single-thread benchmark
  identity, session-root path containment, scoped scientific-source identity,
  fixed estimator defaults, and measured rather than guessed resource gates.
- Exact next action: user reviews the corrected documents, then separately
  approves implementation if satisfied.

#### 2026-10-06 - WP0 third readiness correction

- State: third correctness/completeness/conciseness correction complete. A
  fresh pass after the final correction found no additional issue; WP0 is
  ready for user review and remains documentation only.
- Authorization: documentation changes only.
- Sol / Terra / reviewer: primary Codex review; no worker was used.
- Start HEAD / end HEAD: `787aadbc2843c9ec2e8b03410f8a201f19f67c37` /
  `787aadbc2843c9ec2e8b03410f8a201f19f67c37`.
- Owned files: `docs/task_variable_spec_v5.md` and this plan only. Concurrent
  neural-regression documentation remains owned by another user task.
- RED command and result: not applicable; no implementation was authorized.
- GREEN/regression commands and results: `git diff --check --
  docs/task_variable_spec_v5.md docs/task_variable_implementation_plan.md`
  passed; Markdown fence counts were even (2 and 28); `test -f`/`rg` checks
  found every referenced existing source file and function; the documented
  fixed, tuned, and bounded fit counts independently recomputed as 21,600,
  993,600, and 2,400; and `git rev-parse HEAD` matched
  `git rev-parse origin/refactor` at the recorded commit.
- Commits: the previous task-variable documentation is present in
  `787aadbc2843c9ec2e8b03410f8a201f19f67c37`; the current correction is
  uncommitted.
- Real-data or external actions: read-only inspection confirmed the designated
  CT026 table's categorical encodings and the two aligned archives' member
  names. No table, neural array, source code, test, benchmark, transfer, or
  scheduler state was changed.
- Findings and unresolved risks: corrected safe one-column CSV publication,
  exact binary/no-choice validation, mechanical IRIG/manual coverage
  selection, verbatim feature-parameter provenance, tensor-row identity,
  custom-results-root discovery, output/Git containment, validated reuse of
  completed runs, resource-envelope detail, and scoped dirty-source gates. No
  known correctness, completeness, concision, or implementation-contract
  blocker remains after the final fresh pass.
- Exact next action: user reviews the corrected documents. If accepted, commit
  them and record the new HEAD; implementation still requires a separate
  explicit request.

#### 2026-10-06 - WP0 fourth readiness correction

- State: implementation-chat findings resolved; a fresh consistency pass found
  no additional blocker. WP0 remains documentation only and ready for review.
- Authorization: documentation changes only.
- Sol / Terra / reviewer: primary Codex review of the reported findings; no
  worker was used.
- Start HEAD / end HEAD: `ed9a79b3d98505c757c75f93061e3f08b91fb360` /
  `ed9a79b3d98505c757c75f93061e3f08b91fb360`.
- Owned files: `docs/task_variable_spec_v5.md` and this plan only. Both were
  tracked and clean at the start of this correction.
- RED command and result: not applicable; no implementation was authorized.
- GREEN/regression commands and results: `git diff --check --
  docs/task_variable_spec_v5.md docs/task_variable_implementation_plan.md`
  passed; Markdown fence counts were even (2 and 30); extracted specification
  and plan target vocabularies each contained the same 18 identifiers; stale
  searches confirmed that no proposed unit-selector extension or ambiguous
  rank contract remains; and HEAD matched `origin/refactor` at the recorded
  commit.
- Commits: `ed9a79b3d98505c757c75f93061e3f08b91fb360` contains both documents
  through the prior correction; this fourth correction is uncommitted.
- Real-data or external actions: none.
- Findings and unresolved risks: corrected the stale handoff, replaced the
  proposed unit-selector API change with existing `filter_cluster_metadata`
  reuse, froze all target identifiers and legal fold counts, defined the PCA
  dimensional component limit, and required temporary-file cleanup on every
  backfill path. No known blocker remains from the reported findings or the
  subsequent consistency pass.
- Exact next action: user reviews this correction. If accepted, commit the two
  documents and record the new HEAD before separately authorized WP1 work.

#### 2026-10-06 - WP0 fifth readiness correction

- State: final TDD completeness findings resolved; WP0 remains documentation
  only and ready for user review/commit.
- Authorization: the user explicitly requested resolution of the reported TDD
  issues and required work to stop before implementation.
- Sol / Terra / reviewer: primary Codex documentation correction; no worker was
  used.
- Start HEAD / end HEAD: `ed9a79b3d98505c757c75f93061e3f08b91fb360` /
  `ed9a79b3d98505c757c75f93061e3f08b91fb360`.
- Owned files: `docs/task_variable_spec_v5.md` and this plan only. Existing
  uncommitted readiness corrections in both files were preserved.
- RED command and result: not applicable; no tests or implementation were
  authorized.
- GREEN/regression commands and results: documentation-only checks passed:
  `git diff --check -- docs/task_variable_spec_v5.md
  docs/task_variable_implementation_plan.md`; Markdown fence counts remained
  even (2 and 30); the target vocabularies remained the same ordered 18
  identifiers; and source inspection reconfirmed both the legacy PCA helper's
  two-unit minimum and the configurable cluster filter's integer coercion.
- Commits: none; these documentation changes are intentionally left for the
  user's requested commit/push.
- Real-data or external actions: none.
- Findings and unresolved risks: explicit channel restrictions now reject
  lossy type coercion and normalize deterministically; the backfill tests now
  inject cleanup failure; and a one-usable-unit fold is explicitly valid and
  tested without routing through the incompatible legacy PCA fitter. No known
  TDD readiness issue remains.
- Exact next action: the user reviews, commits, and pushes the documentation.
  Implementation remains separately authorized and has not begun.

#### 2026-10-06 - WP0 approval and WP1 start

- State: WP0 complete at the pushed documentation commit; WP1 tests-only RED
  phase is the single active implementation package.
- Authorization: the user reported the documents pushed and explicitly asked
  implementation to begin with this plan kept current. This does not authorize
  CT026 mutation/computation, benchmarking, transfer, or scheduler actions.
- Sol / Terra / reviewer: primary Sol supervisor; the bounded WP1 Terra worker
  has not yet been assigned.
- Start HEAD / end HEAD: `b69c70a618284aef86f73b8f3d12ac8df2dbea9f` /
  `b69c70a618284aef86f73b8f3d12ac8df2dbea9f` before this handoff commit.
- Owned files: this plan only. The many pre-existing untracked files remain
  outside task-variable ownership.
- RED command and result: pending the WP1 tests-only assignment.
- GREEN/regression commands and results: not applicable; production work has
  not begun.
- Commits: pushed WP0 authority is
  `b69c70a618284aef86f73b8f3d12ac8df2dbea9f`; this live handoff update will be
  committed separately before worker assignment.
- Real-data or external actions: none.
- Findings and unresolved risks: HEAD equals `origin/refactor`; tracked files
  were clean at WP1 start; expected historical untracked files are preserved.
- Exact next action: commit this handoff, assign only the WP1 behavior tests,
  reproduce genuine RED, and commit those tests before implementation.

#### 2026-10-06 - WP1 tests-only RED gate

- State: WP1 RED complete and committed; bounded GREEN implementation is next.
- Authorization: implementation is authorized, but this gate changed tests
  only and performed no experimental-data or external action.
- Sol / Terra / reviewer: primary Sol supervisor / `gpt-5.6-terra` high tests
  worker / Sol independent reproduction; no additional reviewer required for
  WP1.
- Start HEAD / end HEAD: `39ba0db0fbbaa73845bbd2780ae8428d4ff0488e` /
  `acf24ae334ba518a0beab020f7b40a895bbecb6b`.
- Owned files: modified
  `src/tests/behavior_analysis/test_session_analysis.py`; created
  `src/tests/behavior_analysis/test_task_decoding_backfill.py`. The originally
  planned `test_gather_trial_features.py` was already a user-owned untracked
  file, so it was preserved and the focused backfill tests use a new file.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/behavior_analysis/test_session_analysis.py
  src/tests/behavior_analysis/test_task_decoding_backfill.py -q -k
  'rewards_in_block or strict_decision_variable or backfill'` produced 25
  expected WP1 failures, 7 passes, and 77 deselections. Failures cover the
  missing helper/backfill, permissive parser, and non-default-index behavior.
- GREEN/regression commands and results: not applicable before implementation.
  The pre-existing unfiltered session-analysis baseline remains 74 passes and
  3 unrelated failures caused by its existing `model_matrix` test stub.
- Commits: tests-only commit
  `acf24ae334ba518a0beab020f7b40a895bbecb6b`.
- Real-data or external actions: none.
- Findings and unresolved risks: tests freeze entering-trial reward counts,
  strict finite parsing, index preservation, shared-helper reuse, verified
  atomic publication, no-write failure behavior, and visible cleanup failure.
  No production file changed during RED.
- Exact next action: the same worker implements only the two WP1 behavior
  modules, then reports focused GREEN without committing.

#### 2026-10-06 - WP1 GREEN gate

- State: WP1 implementation complete; WP2 tests-only RED is next.
- Authorization: project implementation was authorized. The work remained
  limited to source code and tests; CT026 mutation, neural loading,
  benchmarking, transfer, and scheduler actions remain unauthorized.
- Sol / Terra / reviewer: primary Sol supervisor / the same
  `gpt-5.6-terra` high worker / Sol independent diff review and test
  reproduction.
- Start HEAD / end HEAD: `3104c59` / implementation commit
  `8ea4d24cb4aba2e5718ccd15a40c35ae0944f5ba`.
- Owned files: modified `src/behavior_analysis/session_analysis.py` and
  `src/behavior_analysis/gather_trial_features.py`. The committed WP1 tests
  remained unchanged. All pre-existing untracked files were preserved.
- RED command and result: the committed WP1 RED evidence remains 25 expected
  failures, 7 passes, and 77 deselections from the preceding record.
- GREEN/regression commands and results: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/behavior_analysis/test_session_analysis.py
  src/tests/behavior_analysis/test_task_decoding_backfill.py -q -k
  'rewards_in_block or strict_decision_variable or backfill'` passed with 32
  tests and 77 deselections. The same two files with the three independently
  established baseline `formulaic.model_matrix`-stub failures deselected
  passed with 106 tests, 3 deselections, and one expected warning.
  `git diff --check` also passed.
- Commits: tests-only `acf24ae`; implementation `8ea4d24`.
- Real-data or external actions: none. In particular, the CT026 augmented CSV
  was not opened for writing and the backfill below was not executed.
- Findings and unresolved risks: the general augmentation now emits an
  entering-trial integer `rewards_in_block` series while preserving the input
  index; the shared parser rejects fractional/non-finite choices and
  non-finite rewards; non-default indices no longer break decision-variable
  counting; and the migration validates a unique sibling temporary CSV before
  `os.replace`, with cleanup failures surfaced. The exact proposed WP9A
  invocation from the repository root is:

  ```bash
  uv run python -c 'from pathlib import Path; from src.behavior_analysis.gather_trial_features import backfill_rewards_in_block_csv; backfill_rewards_in_block_csv(Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/processed/CT026_2026-08-03_111938_augmented_trials.csv"))'
  ```

  It is a proposal only and requires the WP9A prerequisite, backup, and
  explicit user approval before execution.
- Exact next action: commit this documentation-only GREEN handoff, then write
  only the WP2 config/target tests, reproduce RED, and commit those tests
  before any WP2 implementation.

#### 2026-10-06 - WP2 tests-only RED gate

- State: WP2 RED complete and committed; bounded GREEN implementation is next.
- Authorization: implementation is authorized, but this gate changed tests
  only and performed no experimental-data or external action.
- Sol / Terra / reviewer: primary Sol supervisor / the same
  `gpt-5.6-terra` high tests worker / Sol test-design correction and
  independent RED reproduction; no additional reviewer is required for WP2.
- Start HEAD / end HEAD: `38b2854a334fb4a6bd2373264c0d86f8fa6a8cfc`
  / `3ccae6c65912eba6dd696a18ee393ab45db1bee5`.
- Owned files: created
  `src/tests/neural_analysis/task_decoding/__init__.py`, `test_config.py`, and
  `test_targets.py`. All pre-existing untracked files were preserved.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/task_decoding/test_config.py
  src/tests/neural_analysis/task_decoding/test_targets.py -q` produced two
  collection errors, both the expected missing
  `src.neural_analysis.task_decoding` production package. Both test modules
  separately passed `uv run python -m py_compile`.
- GREEN/regression commands and results: not applicable before implementation.
- Commits: tests-only commit
  `3ccae6c65912eba6dd696a18ee393ab45db1bee5`.
- Real-data or external actions: none.
- Findings and unresolved risks: Sol corrected one repeated-fixture directory
  bug and added missing approved edge cases before committing. Because the
  prose freezes semantics but not every JSON key spelling, the initial tests
  use direct snake-case dataclass-aligned keys: `session_metadata_path`,
  `augmented_trial_path`, `trial_feature_parameter_path`, `pfc_region`,
  `hpc_region`, `alignment`, `bin_width_ms`, `pfc_pc_count`, `hpc_pc_count`,
  `target_names`, `regularization_mode`, fold counts,
  `trusted_utc_bounds`, and `output_root`. No aliases or extra user knobs are
  accepted. Path/source-manifest hashing remains owned by later `results.py`;
  WP2 covers portable configured paths, containment, overlap, and scientific
  payload separation only.
- Exact next action: the same worker implements only the new package marker,
  `config.py`, and `targets.py`, then reports focused GREEN without committing.

#### 2026-10-06 - WP2 GREEN gate

- State: WP2 implementation complete; WP3 tests-only RED is next.
- Authorization: project implementation was authorized. Work remained limited
  to source and synthetic temporary test fixtures; no experimental-data,
  decoding, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / the same
  `gpt-5.6-terra` high worker / Sol independent contract and source review.
- Start HEAD / end HEAD: `fda0399` / implementation commit
  `289950223a4a383e459cdfce17c0d4417b180510`.
- Owned files: created `src/neural_analysis/task_decoding/__init__.py`,
  `config.py`, and `targets.py`; review-only test corrections modified the two
  committed WP2 test files. All pre-existing untracked files were preserved.
- RED command and result: the initial committed gate produced two expected
  missing-package collection errors. Independent source review then added
  three tests-only correction commits before production was finalized:
  `d3cd613` produced 6 failures and 112 passes, `67a18cf` produced 2 failures
  and 116 passes, and `f13d31b` produced 4 failures and 118 passes. These froze
  canonical metadata naming, external-config output defaults, immutable
  scientific bounds, categorical label/derivation metadata, estimator seed
  provenance, strict validation error boundaries, output-directory kind, and
  finite experimenter flags.
- GREEN/regression commands and results: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/task_decoding/test_config.py
  src/tests/neural_analysis/task_decoding/test_targets.py -q` passed 122 tests.
  `env UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/test_session_metadata.py -q` passed 17 tests.
  `uv run python -m py_compile` for all three new modules, `git diff --check`,
  and the new-source line-length check also passed.
- Commits: initial tests `3ccae6c`; review regression tests `d3cd613`,
  `67a18cf`, and `f13d31b`; implementation `2899502`.
- Real-data or external actions: none.
- Findings and unresolved risks: the installed scikit-learn API was verified
  read-only as version 1.8.0 before implementation; its LogisticRegression,
  ElasticNet, and PCA signatures match the recorded controls, and no
  deprecated logistic `penalty` value is emitted. The minimal JSON keys listed
  in the RED record are now the initial exact interface; no aliases or extra
  user knobs are accepted. Configuration/path validation opens only the three
  small declared inputs. Neural metadata membership, sorter/alignment files,
  channel/unit selection, and coverage remain WP3 responsibilities, and input
  identity hashing remains owned by later `results.py`.
- Exact next action: commit this documentation-only GREEN handoff, then write
  only the WP3 activity/unit/coverage tests, reproduce RED, and commit those
  tests before any WP3 implementation.

#### 2026-10-06 - WP3 RED gate

- State: WP3 activity/unit/coverage tests are committed RED; implementation is
  next.
- Authorization: project implementation was authorized. Work remained limited
  to source-controlled synthetic tests; no experimental-data, decoding,
  benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / the same Terra high worker /
  Sol independent test-contract and existing-loader review.
- Start HEAD / end HEAD: `ceb9b28` / tests-only commit
  `6e45ebfb17074a332f41502dbf48e1fc756f3d61`.
- Owned files: created only
  `src/tests/neural_analysis/task_decoding/test_activity.py`; all pre-existing
  untracked files were preserved.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/task_decoding/test_activity.py -q` stops at one
  collection error solely because
  `src.neural_analysis.task_decoding.activity` does not yet exist. The test
  file separately passes `uv run python -m py_compile` and the staged diff
  passed `git diff --cached --check`.
- GREEN/regression commands and results: not applicable before implementation.
- Commits: tests-only commit `6e45ebf`.
- Real-data or external actions: none.
- Findings and unresolved risks: review replaced a parallel probe-path record
  with the existing `ResolvedProbeSources` contract, corrected the on-disk
  channel-quality fixture schema, and added independent label/inside-brain
  filters, duplicate cluster IDs, bilateral full-window coverage, missing and
  non-finite trusted bounds, dry-run array-read sentinels, source-size versus
  tensor-memory accounting, applicable memory-source selection, and the
  unknown-budget hard stop. The tests freeze small helper interfaces needed by
  later preflight code but do not authorize opening a real neural archive.
- Exact next action: commit this documentation-only RED handoff, then have the
  same worker implement only
  `src/neural_analysis/task_decoding/activity.py` and report focused GREEN
  without committing.

#### 2026-10-06 - WP3 GREEN gate

- State: WP3 activity loading and coverage are complete; WP4 tests and their
  independent numerical-design review are next.
- Authorization: project implementation was authorized. Work used only
  source-controlled code and pytest temporary synthetic fixtures; no
  experimental-data, decoding, benchmark, transfer, or scheduler action was
  performed.
- Sol / Terra / reviewer: primary Sol supervisor / the same Terra high worker /
  Sol production, loader-API, and scientific-contract review.
- Start HEAD / end HEAD: RED handoff `5c97d9a` / implementation commit
  `d6682504f5f42f1a46e40838597201619a446f77`.
- Owned files: created only
  `src/neural_analysis/task_decoding/activity.py`; the committed WP3 test file
  was not changed during GREEN and all pre-existing untracked files were
  preserved.
- RED command and result: the committed `test_activity.py` suite stopped at
  one collection error because `activity.py` did not exist.
- GREEN/regression commands and results: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/task_decoding -q` passed 143 tests. The affected
  channel-quality/population-PCA/session-metadata group passed 57 tests, and
  the affected unit-loading/spike-Pynapple/package group passed 90 tests.
  `uv run python -m py_compile` for `activity.py`, `config.py`, and
  `targets.py`, the staged whitespace check, and the new-source line-length
  check also passed. Pynapple emitted only its existing tiny-fixture interval
  and zero-duration time-support warnings.
- Commits: tests-only `6e45ebf`; implementation `d668250`.
- Real-data or external actions: none.
- Findings and unresolved risks: the synthetic representative tensors are
  PFC `(3, 4, 2)` and HPC `(3, 4, 2)` on common trial/time axes. The dry
  allocation inputs are 3 tensor trials, 4 bins, 2 PFC units, and 2 HPC units,
  yielding exactly 384 float64 tensor bytes; eight source-file sizes remain
  separate I/O/provenance facts. Injected budgets verified local-only 1000
  bytes, scheduled minimum 800 bytes, login-approved 900 bytes, the inclusive
  400-of-800-byte threshold, the 401-byte rejection, and unknown-budget
  rejection. No claim about real-session memory or runtime is made.
- Exact next action: commit this documentation-only GREEN handoff, then write
  only WP4 grouped-modeling RED tests with Terra xhigh. An independent Sol
  xhigh review must approve leakage, fit-reuse, validity, component-limit, and
  coefficient coverage before the tests-only commit.

#### 2026-10-06 - WP4 RED gate

- State: WP4 split, preprocessing, estimator, metric, aggregation, and
  coefficient tests are committed RED; fixed-mode implementation is next.
- Authorization: project implementation was authorized. Work remained limited
  to source-controlled synthetic tests; no experimental-data, decoding,
  benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / Terra xhigh tests worker /
  independent Sol xhigh numerical and leakage review.
- Start HEAD / end HEAD: `deda1c6` / tests-only commit
  `6ee63556944bf50f48a4fb51b6934136260fec30`.
- Owned files: created only
  `src/tests/neural_analysis/task_decoding/test_modeling.py`; all pre-existing
  untracked files were preserved.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/task_decoding/test_modeling.py -q` stops at one
  collection error solely because
  `src.neural_analysis.task_decoding.modeling` does not yet exist. The test
  file separately passes `uv run python -m py_compile`; the staged diff passed
  `git diff --cached --check`; and all lines are at most 100 characters.
- GREEN/regression commands and results: not applicable before implementation.
- Commits: tests-only `6ee6355`.
- Real-data or external actions: none.
- Findings and unresolved risks: two independent audit rounds closed leakage
  and vacuity gaps before approval. The committed contract checks exact
  installed grouped-fold assignments, outer and inner row identities, shared
  transform and split reuse, exact 15-candidate grids and tie order, one-fold
  candidate invalidation, complete Cartesian result records, positive-class
  probability orientation, complete-fold aggregation, non-finite and warning
  boundaries, stable feature ordering, and direct-unit coefficient metadata.
  The installed API remains scikit-learn 1.8.0; logistic construction must omit
  the deprecated `penalty` argument. Tuned mode is tested but remains
  deliberately gated until fixed mode is independently verified.
- Exact next action: commit this documentation-only RED handoff, then have the
  Terra xhigh worker implement fixed mode and shared helpers without changing
  the committed tests. Independent Sol xhigh review must approve that slice
  before the separate tuned-mode follow-up.

#### 2026-10-06 - WP4 fixed-mode supplemental RED gate

- State: the first fixed-mode implementation passed its focused tests but was
  rejected by independent production review. Supplemental audit-boundary tests
  are committed RED; production correction is in progress.
- Authorization: project implementation remains authorized. Work used only
  source-controlled code and deterministic synthetic tests; no experimental
  data, decoding run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh worker /
  independent Sol xhigh production and supplemental-test review.
- Start HEAD / end HEAD: WP4 RED handoff `f064f9f` / supplemental tests-only
  commit `c85a00c7a84ce53d989925dcec1278b87bcbbebb`.
- Owned files: the commit changes only
  `src/tests/neural_analysis/task_decoding/test_modeling.py`. The bounded
  uncommitted production draft remains limited to `modeling.py` and the
  approved shared-control refactor in `config.py`; it was not staged with the
  tests.
- RED command and result: the nine supplemental test nodes plus the legal
  float-coded categorical case collect as 29 cases: 25 fail for missing
  production contracts and four valid controls pass. The passing controls are
  outer fold counts three/five, inner fold count three, and categorical
  floating values exactly equal to zero/one. Test pycompile, 100-column, and
  whitespace checks pass.
- GREEN/regression commands and results: before audit, the initial production
  draft passed 24 fixed/shared modeling cases, 80 config cases, and 143 other
  task-decoding cases. These results are not a GREEN gate because the
  independent audit found missing contracts.
- Commits: base tests `6ee6355`; base RED handoff `f064f9f`; supplemental
  tests-only `c85a00c`.
- Real-data or external actions: none.
- Findings and unresolved risks: the rejected draft discarded outer-split
  failure reasons, accepted unapproved fold counts and coercible inner row
  indices, dropped target/inner-fold audit identity, permitted arbitrary
  estimator-control overrides, and allowed ambiguous categorical/unit
  identities. The independently approved supplemental tests cover each issue.
  Tuned mode still exits immediately and remains outside this correction.
- Exact next action: commit this documentation-only supplemental RED handoff,
  correct only the fixed/shared production files without changing tests, then
  repeat focused/regression commands and independent Sol xhigh review.

#### 2026-10-06 - WP4 fixed-mode boundary-order RED correction

- State: the first supplemental production correction passed 53 non-tuned
  modeling cases but was rejected on three remaining validation-order edges.
  A second supplemental tests-only slice is committed RED.
- Authorization: project implementation remains authorized. Work used only
  source-controlled code and synthetic arrays; no experimental data, decoding
  run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh worker /
  independent Sol xhigh production and test-design review.
- Start HEAD / end HEAD: first supplemental handoff `11a4e35` / second
  supplemental tests-only commit
  `fab38b8c6f333f2e149dddfec43df04d6fb474aa`.
- Owned files: the commit adds 100 lines only to
  `src/tests/neural_analysis/task_decoding/test_modeling.py`. The bounded
  production draft remains unstaged in `modeling.py` and `config.py`.
- RED command and result: four new test nodes collect as 17 cases: 14 fail and
  three valid controls pass. The failures prove malformed scalar/matrix inner
  indices are mislabeled as scientific missingness, invalid regional PC counts
  can reach outer splitting, and scalar strings can be iterated into fake unit
  IDs. The passing controls are a genuinely empty one-dimensional integer
  selection and scalar-byte rejection for both regions. Test pycompile,
  100-column, and whitespace checks pass.
- GREEN/regression commands and results: the prior corrected draft passed 29
  first-supplemental cases, 24 original fixed/shared cases, 80 config cases,
  and 63 activity/target cases. Full modeling passed 53 and retained exactly
  seven deliberately deferred tuned failures. These are not yet a GREEN gate
  because the independent review rejected the remaining edges.
- Commits: second supplemental tests-only `fab38b8`; preceding supplemental
  tests-only `c85a00c` and handoff `11a4e35`.
- Real-data or external actions: none.
- Findings and unresolved risks: production must distinguish malformed inner
  index shape from a valid empty subset, validate both PC counts before any
  outer-split unavailable return, reject scalar text/byte unit-ID containers,
  and correct the documented invalid-coefficient shape. Tuned mode remains
  outside this correction.
- Exact next action: commit this documentation-only RED correction, make only
  the bounded production fixes without changing tests, then repeat all fixed
  and regression gates plus independent Sol xhigh review.

#### 2026-10-06 - WP4 fixed-mode GREEN gate

- State: fixed-mode grouped modeling and shared preprocessing/scoring helpers
  are GREEN and independently approved. Optional tuned mode is next.
- Authorization: project implementation remains authorized. Only
  source-controlled code and deterministic synthetic tests were used; no
  experimental data, decoding run, benchmark, transfer, or scheduler action
  was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh worker /
  independent Sol xhigh scientific, numerical, leakage, and boundary review.
- Start HEAD / end HEAD: second RED handoff `b92446c` / fixed implementation
  `a8787aaa4ce172e48746690e05e9ab73f3d635f8`.
- Owned files: created
  `src/neural_analysis/task_decoding/modeling.py` and modified
  `src/neural_analysis/task_decoding/config.py` to expose one immutable
  estimator/PCA control mapping used by both execution and serialization.
  Committed tests were not changed during the GREEN correction.
- RED command and result: the base suite originally failed collection because
  `modeling.py` was absent. The two independently reviewed supplements then
  exposed 25 and 14 production failures respectively, with their valid-control
  cases already passing.
- GREEN/regression commands and results: all 70 committed non-tuned modeling
  cases pass: 24 original fixed/shared, 29 first-supplemental, and 17
  second-supplemental cases. The 80 config tests and 63 activity/target tests
  pass; the latter retain three existing tiny-epoch warnings. The full
  modeling module passes 70 cases and has exactly seven expected tuned-only
  failures: six immediate `NotImplementedError` paths and the absent tuned
  selection helper. Source pycompile, 100-column, and staged whitespace checks
  pass.
- Commits: base tests `6ee6355`; supplemental tests `c85a00c` and `fab38b8`;
  fixed production `a8787aa`.
- Real-data or external actions: none.
- Findings and unresolved risks: fixed mode fits one PFC and one HPC transform
  per outer fold, reuses them across time/region/representation cells, and
  emits the complete record Cartesian product. Frozen scikit-learn 1.8.0
  controls omit deprecated logistic `penalty`; estimator overrides are limited
  to family-specific tuning parameters. Scientific unavailable states retain
  reasons, while malformed configuration, shapes, and identities raise. The
  independent reviewer found no remaining fixed-mode production issue. The
  only open WP4 behavior is the already-committed tuned contract.
- Exact next action: commit this documentation-only GREEN handoff, then have
  the same Terra xhigh worker implement tuned mode without changing tests.
  Independent Sol xhigh review must approve the complete WP4 diff.

#### 2026-10-06 - WP4 tuned-mode provenance RED gate

- State: tuned scientific/orchestration behavior is GREEN, but final audit
  rejected missing selection-row provenance. Two supplemental provenance tests
  are committed RED; the bounded correction is in progress.
- Authorization: project implementation remains authorized. Work used only
  source-controlled code and synthetic arrays; no experimental data, decoding
  run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh worker /
  independent Sol xhigh tuned leakage, numerical, and audit-schema review.
- Start HEAD / end HEAD: fixed GREEN handoff `44b6343` / tuned-provenance
  tests-only commit `c1b183c57f3aac37b26ab80d7af03c07441647cc`.
- Owned files: the commit adds 95 lines only to
  `src/tests/neural_analysis/task_decoding/test_modeling.py`. The tuned source
  draft remains unstaged in `modeling.py` and `config.py`.
- RED command and result: the two new focused tests both fail only because
  tuned `FoldRecord.inner_selection_indices` is empty instead of the global
  outer-training row positions. Before that final assertion, they pass the
  complete valid 15-by-3 and unavailable 15-by-0 candidate audit mapping
  checks, selected-index alignment, fixed-mode empty-index control, and the
  no-fit inner-unavailable control. Seventy-seven other modeling cases were
  deselected. Test pycompile, 100-column, and whitespace checks pass.
- GREEN/regression commands and results: before the audit rejection, the full
  task-decoding suite passed 220 tests with five existing fixture/split
  warnings. Tuned evidence was 24 transform fits for both one and two time
  bins, 828/1656 estimator fits respectively, and three inner-split calls.
  Those results are not the final GREEN gate until row provenance is fixed and
  independently re-reviewed.
- Commits: fixed production `a8787aa`; tuned-provenance tests-only `c1b183c`.
- Real-data or external actions: none.
- Findings and unresolved risks: tuned outer-test exclusion, exact inner plans,
  preprocessing reuse, candidate invalidation/ties, unavailable Cartesian
  records, selected outer refit, grid order, and fixed regressions were
  approved. Production must populate each tuned record with the outer-training
  global selection universe and update stale fixed-only docstrings.
- Exact next action: commit this documentation-only tuned RED handoff, make
  only the provenance/documentation correction without changing tests, then
  rerun all WP4 gates and obtain independent Sol xhigh approval.

#### 2026-10-06 - WP4 tuned-mode GREEN gate

- State: WP4 grouped modeling is complete. Fixed and tuned modes, shared
  transforms, split audit records, tuning audits, and selection-row provenance
  are GREEN and independently approved.
- Authorization: project implementation remains authorized. Work used only
  source-controlled code and deterministic synthetic arrays; no experimental
  data, decoding run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh worker /
  independent Sol xhigh final production and provenance review.
- Start HEAD / end HEAD: tuned-provenance handoff `8812a58` / tuned production
  `b844f26cf3ad81ac6e5ae77e64b3041e72f586e2`.
- Owned files: modified
  `src/neural_analysis/task_decoding/config.py` and
  `src/neural_analysis/task_decoding/modeling.py`. Committed tests were not
  changed during GREEN implementation.
- RED command and result: the two final focused provenance tests failed only
  because tuned records held empty selection arrays instead of defensive
  global outer-training row copies; 77 other modeling cases were deselected.
- GREEN/regression commands and results: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/task_decoding -q` passed 222 tests with five
  existing tiny-epoch/class-frequency warnings in 9.28 seconds. The final two
  provenance tests pass; source pycompile and `git diff --check` pass. The
  independent reviewer additionally reproduced 159 modeling/config cases with
  only two expected class-frequency warnings. Tuned fit-count evidence remains
  24 transform fits for both one and two time bins, 828/1656 estimator fits,
  and three inner-split calls.
- Commits: base tests `6ee6355`; fixed supplements `c85a00c` and `fab38b8`;
  fixed production `a8787aa`; tuned-provenance tests `c1b183c`; tuned
  production `b844f26`.
- Real-data or external actions: none.
- Findings and unresolved risks: installed scikit-learn 1.8.0 behavior is
  frozen in configuration and exercised without deprecated logistic
  parameters. Tuned candidate order, invalidation, tie selection, nested
  leakage boundaries, complete unavailable records, selected outer refits, and
  global selection-row identity were approved. Fixed records intentionally
  retain empty inner-selection arrays. No unresolved WP4 scientific or
  numerical issue remains.
- Exact next action: commit this documentation-only GREEN handoff, then begin
  WP5A by writing bounded `results.py` RED tests for the frozen saved schema,
  input/scoped-source identity, checkpoints, and atomic result publication.
  Obtain independent tests-only approval before implementation.

#### 2026-10-06 - WP5A saved-results RED gate

- State: WP5A saved-result persistence, scientific/input identity, target
  checkpoint, and atomic-publication contracts are committed RED. Production
  `results.py` is intentionally absent.
- Authorization: project implementation remains authorized. Work used only
  deterministic temporary files and synthetic arrays; no experimental data,
  decoding run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh worker /
  independent Sol xhigh result-schema, portability, scientific-identity, and
  atomicity review.
- Start HEAD / end HEAD: WP4 GREEN handoff `01f8b0f` / WP5A tests-only
  `4a5664af49a782639e5303847c66a8ce2a748f73`.
- Owned files: created only
  `src/tests/neural_analysis/task_decoding/test_results.py`. No production,
  configuration, pipeline, CLI, documentation, or data file changed in the
  tests-only commit.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/task_decoding/test_results.py -q` fails collection
  only because `src.neural_analysis.task_decoding.results` does not exist; no
  test executes. The module pycompiles, contains no line over 100 columns, and
  passes whitespace checks.
- GREEN/regression commands and results: not applicable before implementation.
- Commits: WP5A tests-only `4a5664a`.
- Real-data or external actions: none.
- Findings and unresolved risks: four tests-only review cycles corrected
  candidate-audit collisions/domain, impossible fold counts, group leakage,
  feature-map ambiguity, incomplete axes/units/schema validation, manifest
  boundary mocks, source-scope gaps, and false atomic-publication assumptions.
  The approved schema contains 48 primitive arrays and 143 expected fixed,
  tuned, invalid-schema, identity, checkpoint, and atomicity cases. Arbitrary
  mixed-type block labels use canonical UTF-8 bytes plus int64 offsets rather
  than pickle or fixed-width text. Prepared sidecars remain after a failed
  final NPZ publication so the run remains resumable. WP5B lifecycle,
  discovery, summaries, figures, detachment, status, and CLI behavior remain
  explicitly outside this slice.
- Exact next action: commit this documentation-only RED handoff, implement only
  `src/neural_analysis/task_decoding/results.py` without changing committed
  tests, run WP5A plus affected task-decoding regressions, and obtain
  independent Sol xhigh production approval before WP5A GREEN.

#### 2026-10-07 - WP5A supplemental saved-results RED correction

- State: the first `results.py` candidate passed the original 143 tests but was
  rejected by independent production review. The missing dynamic and semantic
  contracts are now committed as a supplemental RED gate; production remains
  uncommitted.
- Authorization: project implementation remains authorized. Work used only
  deterministic temporary files and synthetic arrays; no experimental data,
  decoding run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh worker /
  independent Sol xhigh tests-only reviewer across correction cycles.
- Start HEAD / end HEAD: original WP5A RED handoff `87fe0b1` / supplemental
  tests-only commit `bc628c4`.
- Owned files: modified only
  `src/tests/neural_analysis/task_decoding/test_results.py`. The existing
  untracked `src/neural_analysis/task_decoding/results.py` stayed byte-identical
  throughout the RED correction, with SHA-256
  `5799a44418c12777fe054a238994e4503a7ca9bd54165bab1693d7cd60f20a38`.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/task_decoding/test_results.py -q` executes 302
  cases and reports 152 failed / 150 passed against the unchanged production
  candidate. The 143 original baseline cases remain GREEN; seven supplemental
  docstring controls already pass. Pycompile, the 100-column check, and
  `git diff --check` pass.
- Commits: base tests `4a5664a`; original RED handoff `87fe0b1`; supplemental
  tests `bc628c4`; frozen fixed-control correction `2c5f086`.
- Real-data or external actions: none.
- Findings and unresolved risks: review exposed hard-coded fixture dimensions,
  a discarded run fingerprint, common-row/full-row leakage indexing, incomplete
  tuning-audit and schema semantics, noncanonical block acceptance, binary
  over-hashing, overwriteable immutable artifacts, incomplete scoped-source
  tracking, and incomplete data-contract docstrings. The correction suite now
  covers dynamic one-target/five-fold and genuinely unavailable-inner-plan
  records, configuration-coherent time bins, exact axes/units/provenance,
  causal eligibility/class/fit/feature/candidate sentinels, canonical UTF-8
  blocks, immutable publication, and save-before-publication validation. A
  final audit corrected the numerical fixed-control fixture from `alpha=0.1`
  to the specification-owned `alpha=1.0` and added causal validation for both
  family strengths, `l1_ratio`, canonical JSON, and fixed-mode selection
  sentinels.
- Exact next action: modify only
  `src/neural_analysis/task_decoding/results.py` until all 302 result tests and
  the full task-decoding regression suite pass without changing committed
  tests, then obtain independent Sol xhigh production approval before the
  WP5A GREEN commit and handoff.

#### 2026-10-07 - WP5A real-model-state RED gate

- State: independent review of the 302-case GREEN candidate found that the
  persistence validator could not represent several valid WP4 modeling
  results and accepted paired corrupt states. The corrected 351-case result
  contract is committed RED and independently approved; production remains
  uncommitted.
- Authorization: project implementation remains authorized. Work used only
  deterministic temporary files and synthetic arrays; no experimental data,
  decoding run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh tests
  worker / independent Sol xhigh tests-only reviewer across correction cycles.
- Start HEAD / end HEAD: prior grouped-fixture correction `f59b360` / final
  tests-only commit `dedde7d`.
- Owned files: modified only
  `src/tests/neural_analysis/task_decoding/test_results.py`. The untracked
  `src/neural_analysis/task_decoding/results.py` production candidate remained
  unchanged during this RED gate, with reviewed SHA-256
  `7bc43dc75872ecbfc0900a97c84f23141306cfdffb9307a102979ad447629b41`.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest -q
  src/tests/neural_analysis/task_decoding/test_results.py -k wp5a` reports 43
  failed / 2 passed / 306 deselected. The complete file reports 256 failed /
  95 passed. All new saved-run failures first reach the intentional 48-versus-
  50 primitive schema gate; checkpoint failures remain directly causal.
  Pycompile, the 100-column check, and `git diff --check` pass.
- Commits: final real-state tests `dedde7d`; earlier WP5A test history remains
  `4a5664a`, `bc628c4`, and `2c5f086`.
- Real-data or external actions: none.
- Findings and unresolved risks: lossless persistence requires target-axis
  `target_status` and `target_unavailable_reasons` arrays, increasing the
  unreleased result schema from 48 to 50 primitives. Tests now cover real
  categorical and numerical outer-split unavailability, categorical inner
  class-coverage failure despite sufficient group count, selection retained
  across later outer failures, fold-local feature width, nonzero effective
  width after post-transform failure, partition-specific class coverage,
  invalid unavailable-plan audits, exact target metadata/mappings, canonical
  standalone and combined feature maps, and safe checkpoint metadata. Valid
  special states are reloaded and compared so coercive serialization cannot
  pass vacuously.
- Exact next action: implement only recognition and basic consistency of the
  two target-status arrays, rerun the focused tests to expose deeper causal
  failures, then implement the remaining validation and round-trip behavior
  without changing committed tests. Obtain independent production approval
  before the WP5A GREEN commit and handoff.

#### 2026-10-07 - WP5A fold-transform and split-causality RED correction

- State: the 351-case result suite reached GREEN, but independent production
  review found that it still admitted transform states WP4 cannot emit and did
  not establish outer-split causality. Corrected tests are committed under an
  explicitly staged RED gate; production remains uncommitted.
- Authorization: project implementation remains authorized. Work used only
  deterministic temporary files and synthetic arrays; no experimental data,
  decoding run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh tests
  worker / independent Sol xhigh modeling-to-persistence reviewer.
- Start HEAD / end HEAD: prior RED handoff `ed4c4ac` / corrected tests-only
  commit `7f8373b`.
- Owned files: modified only
  `src/tests/neural_analysis/task_decoding/test_results.py`. The untracked
  `src/neural_analysis/task_decoding/results.py` candidate remained byte-
  identical during this correction, with SHA-256
  `857b453e55871b1d951c6237d3156284df752a52b2ff1eea41a7201b5aa5ca37`.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest -q
  src/tests/neural_analysis/task_decoding/test_results.py` reports 10 failed /
  348 passed. Four parameterized outer-unavailable cases and one numeric-label
  positive fail only because production incorrectly requires the full U128
  target reason to equal the compact U32 cell code. Five causal negatives are
  direct missing-rejection failures; the splittable-target negative is
  intentionally masked until the reason/code separation lands. Pycompile, the
  100-column check, and `git diff --check` pass.
- Commits: transform/split correction `7f8373b`; preceding real-state tests
  `dedde7d`.
- Real-data or external actions: none.
- Findings and unresolved risks: one regional transform is reused across all
  time bins and standalone/combined representations for each target/outer
  fold. Direct-unit masks and effective widths therefore cannot vary by time
  or disagree with the corresponding combined segment, and retained PCA
  components must be a prefix. A tuned cell with no valid candidate has zero
  effective features. Target availability must agree with a real outer split
  computed from decoded scalar JSON block labels. Full target-level splitter
  reasons remain lossless in U128; synthesized fit and candidate audit cells
  use the exact compact code `target_outer_unavailable`.
- Exact next action: implement only full target reason versus compact cell-code
  separation, reproduce the sixth causal RED failure, then correct the three
  remaining validator boundaries without changing committed tests. Obtain a
  fresh independent production approval before the WP5A GREEN commit.

#### 2026-10-07 - WP5A shared-split and exact-transform RED correction

- State: the 358-case result suite reached GREEN, but independent production
  review found that persistence used a private block-label adapter that could
  disagree with WP4, and that transform validation did not yet encode all
  regional availability, PCA-width, and tuned cross-cell reuse invariants. The
  corrected tests are committed under an independently approved staged RED
  gate; production remains uncommitted.
- Authorization: project implementation remains authorized. Work used only
  deterministic temporary files and synthetic arrays; no experimental data,
  decoding run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh tests
  worker / independent Sol xhigh modeling-to-persistence reviewer.
- Start HEAD / end HEAD: preceding documentation handoff `856e5d8` / corrected
  tests-only commit `3b69f97`.
- Owned files: modified only
  `src/tests/neural_analysis/task_decoding/test_modeling.py` and
  `src/tests/neural_analysis/task_decoding/test_results.py`. The untracked
  `src/neural_analysis/task_decoding/results.py` candidate remained byte-
  identical throughout, with SHA-256
  `294c866fd9f942035164946391c171868296890eb81b8b55a34521da7c34820e`
  and Git blob ID `a6de8c34115742deefd30b096942fddbc6269b8c`.
- RED commands and results: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest -q
  src/tests/neural_analysis/task_decoding/test_modeling.py` reports 16 failed /
  80 passed with two existing sklearn warnings; all failures are the four
  scalar collision pairs crossed with categorical/numerical and outer/inner
  grouped splits. `env UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run
  pytest -q src/tests/neural_analysis/task_decoding/test_results.py` reports
  three failed / 360 passed. The result failures are the valid regional-
  unavailable state rejected by an invalid sum rule, the valid one-unit PFC
  state rejected by a global rather than per-region combined-PCA prefix rule,
  and a staggered tuned standalone/combined contradiction that is not rejected.
  Pycompile, the 100-column check, and `git diff --check` pass.
- Commits: shared-split and exact-transform tests `3b69f97`; preceding
  transform/split correction `7f8373b`.
- Real-data or external actions: none.
- Findings and unresolved risks: block identities are defined by scalar type
  and value, so integer, string, floating-point, and Boolean aliases must stay
  distinct while homogeneous numerical labels retain numerical ordering. One
  regional transform governs both PCA and direct-unit representations. A
  failed required region makes its standalone and combined cells unavailable,
  but does not invalidate the other standalone region. Combined PCA masks use
  separate regional prefixes, and each attempted regional PCA width is
  `min(requested PCs, usable direct units, training trials * time bins)`.
  Tuned transform evidence must reconcile selected standalone and combined
  cells even when those selections occur at different time bins.
- Exact next action: implement the single shared type-preserving group-label
  normalizer in `modeling.py` and route decoded result scalars through it. Then
  fix the two valid persistence states first so the staged cross-representation
  and exact-PCA-count negatives become causal, implement those checks plus
  staggered selected-cell reconciliation, run both full modules and the full
  task-decoding suite, and obtain fresh independent production approval before
  the WP5A GREEN commit.

#### 2026-10-07 - WP5A saved-results GREEN handoff

- State: WP5A is complete. The saved-result schema, portable input and scoped
  source identities, run fingerprints, target checkpoints, atomic sidecars and
  final NPZ publication, and strict loader validation are GREEN. Grouped split
  semantics and persisted fold-transform evidence now use the same scientific
  contracts as WP4.
- Authorization: project implementation remains authorized. Work used only
  deterministic temporary files and synthetic arrays; no experimental data,
  decoding run, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / same Terra xhigh production
  worker / independent Sol xhigh modeling-to-persistence reviewer.
- Start HEAD / end HEAD: shared-split RED handoff `646e49b` / WP5A GREEN
  production commit `1bd0f69`.
- Owned files: added
  `src/neural_analysis/task_decoding/results.py` and modified
  `src/neural_analysis/task_decoding/modeling.py`. Two genuine stale or missing
  test contracts found during production review were corrected first in
  `src/tests/neural_analysis/task_decoding/test_results.py` and committed
  separately at `072b921` and `c7c2912`.
- GREEN/regression commands and results: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest -q
  src/tests/neural_analysis/task_decoding/test_modeling.py` passed 96 tests with
  two expected sklearn class-size warnings. The result suite passed 366 tests.
  `env UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest -q
  src/tests/neural_analysis/task_decoding` passed 605 tests with five existing
  dependency warnings: three Pynapple interval-rounding warnings and the same
  two sklearn class-size warnings. Pycompile, the 100-column check,
  `git diff --check`, and generated-bytecode cleanup passed.
- Commits: exact regional-PCA test correction `072b921`; sparse tuned-transform
  RED `c7c2912`; independently approved WP5A production `1bd0f69`.
- Real-data or external actions: none.
- Findings and unresolved risks: one shared type-preserving sortable identity
  now governs categorical/numerical outer and inner grouped splitting while
  retaining homogeneous numeric ordering. Saved-result validation derives
  underlying PFC/HPC transform evidence from both standalone cells and valid
  combined segments, including sparse tuned selections; it does not invent
  segment masks from post-estimator failures. Regional availability, combined
  counts, mask identity, and exact PCA component counts are checked against
  reusable direct-unit and training-observation evidence. No WP5A blocker
  remains after the final independent review.
- Exact next action: begin WP5B with tests only. Freeze the shared
  prepare/revalidate/execute contract, immutable run directory and lifecycle,
  checkpoint resume and unexpected-failure behavior, durable logs and summary,
  single-writer guard, standard-library detached launcher, and one-shot
  read-only status interface. Commit an independently approved RED suite before
  implementing `pipeline.py` or `run_session.py`.

#### 2026-10-07 - WP5B single-session lifecycle RED gate

- State: the independently approved WP5B RED contract is committed. It freezes
  bounded planning and preparation, exact portable identity revalidation,
  immutable run discovery, fixed and tuned target checkpoints, result assembly,
  lifecycle/failure/interruption state, single-writer ownership, detached local
  launch, read-only status, and the session CLI.
- Authorization: project implementation remains authorized. Tests use only
  deterministic temporary session trees, synthetic rate tensors, and bounded
  child processes; no experimental data, decoding run, benchmark, transfer, or
  scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / Terra xhigh tests workers /
  independent Sol xhigh tests reviewer. Three rejected drafts were corrected
  before the final approval; collection RED alone was not accepted as evidence.
- Start HEAD / end HEAD: WP5A handoff `14c38c2` / WP5B tests-only commit
  `449f7c9`.
- Owned file: added
  `src/tests/neural_analysis/task_decoding/test_pipeline.py`. No production,
  configuration, data, or external files changed.
- RED command and result: `env UV_CACHE_DIR=/tmp/context-inference-uv-cache
  uv run pytest src/tests/neural_analysis/task_decoding/test_pipeline.py -q`
  fails during collection only because
  `src.neural_analysis.task_decoding.pipeline` and `run_session` do not yet
  exist. This is the intended pre-implementation failure. Pycompile, the
  100-column check, staged/no-index whitespace checks, and generated-bytecode
  cleanup passed.
- Contract decisions: the prepared session command files shell-quote immutable
  paths; `run_batch.py` snapshotting remains deferred to WP6. WP5B owns only the
  minimal categorical balanced-accuracy/AUC and numerical R2 PNGs required by
  the completion gate; the full plotting API, styling, coefficient plots, and
  webapp remain WP7. Late reporting/final-state failures must resume without
  replacing an already valid immutable `results.npz`.
- Findings and unresolved risks: the approved synthetic fixture resolves real
  session metadata, covers all 12 trials bilaterally, produces one available
  144-record categorical result plus one genuine constant-target unavailable
  result, and round-trips through the released WP5A validator. Fixed and tuned
  checkpoint recovery, PID/start-token and Slurm ownership, receipt-before-
  success ordering, abrupt detached death, atomic retry, and lazy thread-limit
  import order are causally tested. No WP5B test-design blocker remains.
- Exact next action: implement `pipeline.py` and `run_session.py` only after this
  tests-only commit. Run the focused lifecycle suite, all task-decoding tests,
  pycompile, line-length and diff checks, then obtain independent production
  approval before the WP5B GREEN commit.

#### 2026-10-07 - WP5B lifecycle and scientific-integrity RED supplement

- State: an independent review of the first source implementation found
  uncovered ownership, resume-history, result-provenance, reporting, scheduler,
  and completion-validation defects. The corrected supplemental RED contract
  is independently approved and committed.
- Authorization: project implementation remains authorized. All new coverage
  uses deterministic temporary directories, synthetic tables/tensors, mocked
  scheduler responses, and bounded child-process fakes. No experimental data,
  real decoding, benchmark, transfer, or scheduler action was performed.
- Sol / Terra / reviewer: primary Sol supervisor / Terra xhigh tests worker /
  independent Sol xhigh tests reviewer. Four review passes corrected receipt
  atomicity, WP4-emittable transform/reason fixtures, causal candidate-code
  corruption, exact execution timing/history, and no-selection estimator
  provenance before approval.
- Start HEAD / end HEAD: WP5B documentation handoff `9b4837c` / supplemental
  tests-only commit `4215c82`.
- Owned files: modified
  `src/tests/neural_analysis/task_decoding/test_pipeline.py` and
  `src/tests/neural_analysis/task_decoding/test_results.py`. The existing
  uncommitted source implementation was frozen during this tests-first gate;
  no production, configuration, data, or external file was changed by the
  supplement.
- RED command and result: `env UV_CACHE_DIR=/tmp/context-inference-uv-cache
  uv run pytest -q src/tests/neural_analysis/task_decoding/test_pipeline.py
  src/tests/neural_analysis/task_decoding/test_results.py -k wp5b_` reports
  31 failed, 3 passed, and 445 deselected. Every failure is a causal missing
  production boundary. The complementary legacy selector reports 445 passed
  and 34 deselected. Pycompile, the 100-column audit, whitespace checks, and
  generated-bytecode cleanup passed.
- Contract decisions: detached receipt-to-guard handoff must never create an
  unowned interval; terminal publication retains the guard; matching complete
  runs are no-op reentries; Slurm uncertainty refuses takeover after bounded
  lookup. Result assembly receives the resolved session ID explicitly, keeps
  target-invalid and not-common rows distinct, derives direct feature axes
  from stable selected metadata, maps exact modeling reasons to whitelisted
  compact codes, and stores canonical selection-rule JSON at a content-derived
  Unicode width rather than an arbitrary cap. Completion requires six real
  region-by-representation heatmap panels per present metric family, accurate
  launch/timing provenance, and decodable PNGs.
- Findings and unresolved risks: the base source passed its original 79-test
  lifecycle suite but was not acceptable under the supplemental scientific and
  concurrency contract. The first GREEN step must add the explicit session-ID
  seam and dynamic selection-rule validation so the remaining causal tests are
  no longer masked. The direct feature, compact-reason, checkpoint preflight,
  owner-history, summary/figure, and scheduler fixes then remain required.
- Exact next action: correct only `pipeline.py`, `run_session.py`, and
  `results.py`; rerun the supplemental selector after the schema seam, then run
  the complete pipeline/results and full task-decoding suites. Obtain a fresh
  independent production review before any WP5B GREEN commit.

#### 2026-10-07 - WP5B single-session lifecycle GREEN gate

- State: WP5B is complete and independently approved. The package now owns
  bounded preparation, immutable discovery, exact identity revalidation,
  resumable target checkpoints, guarded foreground/detached execution,
  portable result assembly, completion summaries/figures, and read-only status.
- Authorization: implementation used only deterministic temporary session
  trees, synthetic arrays/tables, and bounded process fakes. No experimental
  data, decoding run, benchmark, transfer, or scheduler action was performed.
- Start HEAD / end HEAD: corrected fixture handoff `957563d` / independently
  approved WP5B GREEN production commit `8b9232d`.
- Owned production files: added
  `src/neural_analysis/task_decoding/pipeline.py` and `run_session.py`, and
  modified the bounded schema validation in `results.py`. Final tests-first
  contracts and fixture corrections were committed separately at `8f6c012`,
  `2bd2666`, `eb3df73`, and `77f2cc8`.
- RED evidence: the final focused contract initially reported 27 failures and
  12 passes, covering source snapshots, explicit session identity, closed
  reason-code mapping/storage, receipt ordering, and complete-directory
  validation. Independent review then added causal RED coverage for adversarial
  regional prose, inconsistent terminal publication, and whitespace identity.
- GREEN/regression evidence: the focused final-review selector passed 36 tests.
  The complete pipeline/results pair passed 517 tests with one expected
  all-NaN fixture warning. The final full task-decoding package passed 760 tests
  with six known warnings: three Pynapple interval-rounding warnings, two
  sklearn small-class warnings, and the same all-NaN expected-value warning.
  `git diff --check` passed.
- Contract decisions: detached children wait for the matching atomic
  PID/start-token receipt before execution; complete discovery validates the
  summary and every applicable decodable PNG; result assembly requires the
  resolved nonblank session ID; saved invalidities use an exact reviewed compact
  vocabulary; and preparation snapshots both `run_session.py` and `pipeline.py`.
- Independent review: the first production pass found permissive regional
  reason parsing, inconsistent complete-state reentry, and whitespace session
  identity. All three were corrected test-first. Re-review reported no P0-P2
  findings or blockers.
- Exact next action: propose the WP6 batch-runner implementation plan, including
  architecture, dependencies, tests to write first, and session-level parallel
  performance limits. Obtain user approval before writing WP6 code.

#### 2026-10-07 - WP6 batch-runner GREEN gate

- State: WP6 is complete. The standard-library batch entrypoint supports
  ordered `dry-run` and foreground `new`, explicit reruns, independent session
  outcomes, one-worker safe execution, and evidence-gated process parallelism.
- Authorization: implementation and verification used only deterministic
  temporary files, synthetic plans, process fakes, and a two-process failure
  smoke against empty temporary run directories. No experimental data,
  decoding run, benchmark, transfer, or scheduler action was performed.
- Start HEAD / end HEAD: WP5B handoff `1341060` / WP6 GREEN production commit
  `7d6b4a8`.
- Owned files: added `src/neural_analysis/task_decoding/run_batch.py`; extended
  bounded activity dry-run dimensions in `activity.py`; and extended
  `pipeline.py` planning, batch provenance, and immutable source snapshots.
  Tests were committed first at `9727ec8`, with evidence-fingerprint and
  admission-edge supplements at `a49a74f` and `827989c`.
- RED evidence: the base batch suite initially failed collection because
  `run_batch.py` did not exist. Separate causal checks failed for missing
  activity dimensions, resource-envelope identity, evidence fingerprint,
  one-family zero fit counts, full printed memory calculation, and the exact
  `run_batch.py` snapshot.
- GREEN/regression evidence: the final batch suite passed 22 tests; the full
  activity/pipeline/batch group passed 190 tests with four known warnings; and
  the full task-decoding package passed 782 tests with six known warnings.
  `git diff --check`, the 100-column audit, source compilation, both CLI help
  paths, and a real two-worker process-executor failure smoke passed.
- Contract decisions: config-list-relative paths are resolved portably and
  duplicates are rejected; `new --rerun` mirrors the single-session runner;
  no evidence means one worker regardless of request; parallel admission uses
  explicit complete evidence, an exact run fingerprint, matching scientific/
  runtime/platform/thread identities, a non-exceeded resource envelope, and
  measured peak RSS unchanged under CPU/session/50%-MemAvailable caps.
- Resource measurement boundary: WP6 defines and validates the atomic
  `resource_usage.json` evidence schema but does not fabricate measurements.
  Parallel use therefore remains unavailable until an authorized benchmark
  records `resource.getrusage` peak RSS for a completed compatible run.
- Review findings: a manual post-GREEN review added causal coverage for the
  absent-family zero-fit case, printed memory/CPU calculation, and direct
  process-executor routing. No remaining correctness blocker was found.
- Exact next action: propose the WP7 plotting and saved-results webapp plan,
  including light-mode rendering, saved-loader-only routing, tests to write
  first, and shared `webapp/app.py` ownership. Obtain user approval before code.

#### 2026-10-07 - WP7 plotting and integrated webapp GREEN gate

- State: WP7 is complete. Saved results now produce readable six-panel
  categorical balanced-accuracy/AUC and numerical R2 heatmaps through one
  plotting owner. Direct-unit selections provide per-fold coefficients,
  median/IQR, selection frequency, nonzero-fit summaries, explicit excluded/
  unavailable states, and scientifically bounded captions. The existing
  metadata webapp has one early-routed read-only task-variable results view.
- Authorization: the user approved the bounded WP7 plan before tests. Work
  used only synthetic arrays, temporary saved-run fixtures, and rendered
  temporary PNGs. No experimental data, decoding run, benchmark, transfer,
  scheduler action, or long local computation was performed.
- Start HEAD / end HEAD: pushed WP6 handoff `5fb25ef` / WP7 GREEN production
  commit `ef7ad55`.
- Owned files: added `src/neural_analysis/task_decoding/plotting.py` and
  `src/neural_analysis/webapp/task_decoding_views.py`; replaced the private
  pipeline renderer with the shared saved-result plotting owner; and extended
  `webapp/session_inputs.py` plus the early route in `webapp/app.py`.
- Tests-first history: base RED contracts are committed at `c027f2c`, with
  empty-run guidance at `b3fff29`. Genuine test correction `b098021` replaced
  an ambiguous NumPy RGB comparison. Supplemental tests committed before the
  final production commit cover canonical `run_state.json` and symlink
  containment (`6c2eb45`), zero-unit regions (`db91e54`), colorbar/caption
  layout (`2dd91d6`), signed-time titles (`aa8d287`), and causal completed-run
  selector rendering (`0c6ce07`).
- RED evidence: the base focused command reported 10 expected failures and 12
  existing passes because the plotting/view modules, view constant, and early
  route did not exist. Review REDs separately reproduced the canonical-state
  mismatch, the zero-unit coefficient-table failure, overlapping colorbar,
  and ambiguous negative-time title.
- GREEN/regression evidence: the final focused plotting/webapp contract passed
  26 tests; the full task-decoding package passed 789 tests with six known
  synthetic-fixture/library warnings; and the affected legacy/canonical
  webapp suites passed 84 tests. Source compilation, whitespace checks, the
  new-file 100-column audit, and installed Streamlit signature verification
  passed.
- Visual review: synthetic categorical/numerical PNGs were inspected at native
  resolution. The first pass exposed a colorbar overlap and unreliable caption
  wrapping; the test-first correction allocates a dedicated colorbar axis,
  wraps captions, and suppresses redundant right-column target labels. A
  second visual pass found the heatmaps and coefficient summary readable on
  opaque white backgrounds.
- Contract decisions: result discovery reads only direct children of the
  contained session-relative root, rejects symlink escapes, ignores hidden,
  incomplete, and loader-invalid runs, and uses canonical complete/final-
  publication state. The view remains selectable without raw sources; all
  selectors load/re-render validated saved arrays and never import fitting,
  pipeline, or activity modules. Legacy manual-path launches receive concise
  metadata-session guidance rather than an unrestricted filesystem browser.
- Exact next action: propose the WP8 user/maintainer documentation and portable
  example plan from Sections 4.4, 10.9, and the WP8 package gate. Obtain user
  approval before writing documentation tests or artifacts.

#### 2026-10-07 - WP8 documentation and portable example GREEN gate

- State: WP8 is complete. The scientist quickstart now covers portable input
  preparation, the narrow behavior-side backfill, local and batch commands,
  unattended ownership, output discovery, the saved-result webapp, fixed/
  tuned work, benchmark planning, and the explicit WP11/WP13 cluster gates.
  The package README maps every Python file, data/result contracts, leakage
  boundaries, lifecycle guarantees, tests, extension steps, and non-goals.
- Authorization: the user approved the bounded WP8 tests-first plan. Work used
  repository text, small temporary files, and one temporary synthetic session.
  No experimental data, CT026 mutation, real benchmark, transfer, scheduler
  action, or persistent analysis output was used.
- Start HEAD / end HEAD: pushed WP7 handoff `3281d01` / WP8 documentation
  GREEN `8b92c4f` (this handoff commit follows separately).
- Owned files: added
  `src/tests/neural_analysis/test_task_decoding_documentation.py`,
  `docs/examples/neural_analysis/task_decoding_config.json`, and
  `src/neural_analysis/task_decoding/README.md`; extended
  `src/neural_analysis/README.md`; and made one bounded JSON-boundary change in
  `pipeline.py` so dry-run source-size mapping keys are strings.
- RED command and result: `env
  UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run pytest
  src/tests/neural_analysis/test_task_decoding_documentation.py -q` reported
  six expected failures and two passes. Five failures identified the missing
  WP8 artifacts. The sixth causally exposed the existing real dry-run CLI
  failure: `json.dumps(..., default=str)` did not convert `Path` mapping keys.
- GREEN/regression commands and results: the focused WP8 suite passed 8 tests.
  The full task-decoding package plus WP8 contracts passed 797 tests with the
  same six known synthetic/library warnings in 84.78 seconds. Both exact CLI
  help commands passed. Real single-session and batch dry runs against one
  temporary fixture returned valid JSON; the batch reported one effective
  worker without evidence. Source compilation and `git diff --check` passed.
- Commits: tests-only `c1cda3b`; genuine Markdown-whitespace test correction
  `52729fc`; test formatting `a1c1f5c`; dry-run JSON fix `b2b3dde`; and
  documentation/example GREEN `8b92c4f`.
- Real-data or external actions: none. The bounded detached command created a
  temporary run beneath `/tmp/context-inference-wp8-vF9gCf`, printed its
  receipt/log and exact resume/status commands, and returned. One later status
  call reported the durable initialized state, dead receipt, and recovery
  command without polling.
- Findings and unresolved risks: the command sandbox reaped its detached child
  when the parent tool invocation ended, so this environment cannot prove
  terminal-independent child completion even though the receipt/status/
  recovery lifecycle behaved correctly and focused lifecycle tests pass. A
  foreground recovery attempt reached neural loading but the planning-only
  fixture's two isolated spikes produced an empty Pynapple time-support union;
  no production change was made to disguise that inadequate execution
  fixture. WP9 must supply the seeded, fully executable synthetic session that
  tests end-to-end completion. Operational Slurm instructions remain
  intentionally absent pending WP11, and array commands remain gated by WP13.
- Exact next action: propose the WP9 seeded synthetic end-to-end test plan,
  including expected signal timing, all region/representation/target-family
  coverage, one unavailable target, saved reload/plots, and leakage checks.
  Obtain user approval before writing WP9 tests. Do not execute CT026 or a real
  benchmark.

#### 2026-10-07 - WP9 synthetic integration characterization gate

- State: WP9 is complete as a user-approved characterization-only gate. The
  final seven scientific integration contracts pass the existing production
  pipeline, so no production edit was justified or made.
- Authorization: the user approved the bounded fixed-mode WP9 plan and then
  explicitly approved characterization-only completion after the valid final
  tests produced no genuine production RED. Work used only seeded synthetic
  data below pytest-owned `/tmp` directories. No experimental data, CT026
  preparation, benchmark, transfer, scheduler action, or persistent analysis
  output was used.
- Start HEAD / end HEAD: WP8 handoff `36cbe76` / WP9 tests-only commit
  `9ea0109` (this documentation handoff follows separately). The remote remains
  `3281d01bd4cf65e381e986cda8ad50c4db360383` because GitHub rejected the prior
  pushes with a server-side internal error; local implementation work remains
  seven commits ahead before this handoff.
- Owned file: added
  `src/tests/neural_analysis/task_decoding/test_synthetic_integration.py`.
  No production, configuration, dependency, experimental-data, or external
  file changed.
- TDD/characterization evidence: the first test draft reported two failures
  and five passes, but both failures were test-fixture defects rather than
  production defects: the held-out-offset fixture advanced the shared random
  stream and therefore changed training spikes, and an assertion incorrectly
  required every estimator cell to converge instead of accepting reviewed
  cell-local scientific unavailability. The fixture was corrected with an
  independent offset random stream and the assertion was aligned with the
  saved validity contract. The resulting seven scientifically valid tests all
  passed current production. The user approved stopping rather than inventing
  a production failure merely to manufacture a RED/GREEN pair.
- Synthetic fixture identity: seed `20261007`; 72 trials in six chronological
  blocks; two probes with four selected units per region; choice alignment;
  eight 500 ms bins across the frozen [-2, 2] s window; fixed regularization;
  three outer folds; available categorical `current_action`; available
  numerical `relative_doubt`; and grouped-impossible categorical
  `current_state`. Seeded Poisson spikes include a local action signal in bin
  four. A second session modifies only fold-zero held-out spikes using its own
  random stream.
- Gate coverage: the tests exercise real metadata and aligned-spike loading,
  Pynapple rate binning, all PFC/HPC/combined by PCA/direct-unit cells,
  cell-local unavailable fits, target checkpoints, immutable saved results,
  summaries, all three required PNGs, validated reload without decoding,
  coefficient summaries, exact scientific rerun determinism, the expected
  signal interval, and an end-to-end held-out leakage control.
- GREEN/regression evidence: the focused WP9 file passed 7 tests in 4.55
  seconds. The full task-decoding package plus documentation contracts passed
  804 tests with six known warnings in 86.87 seconds. Reused metadata,
  spike-loading, population-PCA, saved-result webapp, cross-session plotting,
  behavior backfill, and session-analysis regressions passed 400 tests with 23
  known warnings in 10.76 seconds. Compilation, the 100-column audit, and
  `git diff --check` passed. A separate pre-existing untracked legacy
  `test_project_utils.py` could not collect because undeclared optional
  `autograd` is absent; it is not part of the tracked WP9 surface and no
  dependency was added to mask that repository issue.
- Visual review: the seeded categorical balanced-accuracy and numerical R2
  PNGs were inspected at original resolution. All six panel titles, axes,
  target labels, descriptive-reference colorbars, unavailable-cell shading,
  and scientific captions are readable on opaque white backgrounds. A
  lower-detail combined preview briefly appeared to omit the numerical
  first-row titles; original-resolution inspection and rendered artist bounds
  confirmed that this was preview scaling rather than a saved-figure defect.
- Proposed CT026 WP10 preflight command, not executed: after separately
  approved WP9A has produced and validated the complete augmented table, use a
  proposed immutable config at
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/task_decoding_config_wp10_bounded.json`.
  Its bounded scientific payload is choice alignment, 100 ms bins, ten PCs per
  region, ProbeA as PFC, ProbeB as HPC, fixed regularization, five outer folds,
  three inactive inner folds, and exactly `current_action` plus
  `relative_doubt`. From the tracked-clean repository root, the exact proposed
  read-only command is:

  ```bash
  env UV_CACHE_DIR=/tmp/context-inference-uv-cache \
    MPLCONFIGDIR=/tmp/context-inference-matplotlib \
    uv run python -m src.neural_analysis.task_decoding.run_session dry-run \
    --config "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/task_decoding_config_wp10_bounded.json"
  ```

  Creating that config, running this command, loading CT026 neural arrays, and
  launching the 2,400-fit benchmark all remain unperformed and require their
  later package approvals.
- Exact next action: request explicit WP9A approval for the already recorded
  timestamped-backup and one-file `backfill_rewards_in_block_csv` procedure.
  Stop on any pre-write mismatch. Do not start the proposed WP10 dry run or
  benchmark under WP9A authority.

#### 2026-10-07 - WP9A CT026 augmented-table preparation GREEN gate

- State: WP9A is complete. The separately approved CT026 table preparation
  created one recoverable backup and atomically appended the validated integer
  `rewards_in_block` column. No neural array was opened, no decoder or
  benchmark ran, and no transfer or scheduler action occurred.
- Authorization: after WP9 was pushed, the user approved the exact source,
  fixed backup path, backfill invocation, validation procedure, and stop
  conditions. The first preflight stopped without writing when it found both a
  LibreOffice lock and a source mode that the helper did not yet preserve. The
  user closed the CSV and separately approved the tests-first permission fix
  and stale-lock removal if needed. LibreOffice removed the lock on close; the
  exact removal command found no path and deleted nothing.
- Source-control commits: tests-only permission regression `b322427`; minimal
  helper fix `cdb04d9` (this documentation handoff follows separately). The
  fix records the source Unix permission bits and applies them to the validated
  sibling temporary CSV before `os.replace`; ownership already matched the
  executing user and parent directory.
- RED/GREEN evidence: the new causal test failed because the atomic helper
  published mode `0600` instead of source mode `0664`. After the production
  fix, that test passed; the focused behavior selector passed 33 tests with 77
  deselections; and the final backfill plus task-target suites passed 50 tests.
  The broader two-file behavior run passed 107 tests and retained exactly the
  three documented `formulaic.model_matrix`-stub baseline failures, with two
  warnings. Source compilation and `git diff --check` passed.
- Approved input:
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/processed/CT026_2026-08-03_111938_augmented_trials.csv`.
  Its pre-write identity was 399,173 bytes, SHA-256
  `51d69a39483b454b31bbafa508d0d1726c48c1f7f7fb3c144faccf11ea7e1702`,
  mode `0664`, owner/group `1000:1000`, 650 rows, and 75 columns.
- Recoverable backup:
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/processed/CT026_2026-08-03_111938_augmented_trials.csv.pre-wp9a-20261007T172053Z.bak`.
  It is a separate inode and retains the exact pre-write hash, byte count,
  permissions, ownership, and modification time.
- Exact executed mutation: from the repository root, after backup verification,
  WP9A ran:

  ```bash
  uv run python -c 'from pathlib import Path; from src.behavior_analysis.gather_trial_features import backfill_rewards_in_block_csv; backfill_rewards_in_block_csv(Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/processed/CT026_2026-08-03_111938_augmented_trials.csv"))'
  ```

- Output identity and validation: the augmented CSV is now 400,416 bytes with
  SHA-256
  `a0cf46386f55b0b7134cdfb67101d496b7ca17c33d7b57a38e645bbd04118383`,
  mode `0664`, and owner/group `1000:1000`. It has 650 rows and 76 columns;
  `rewards_in_block` is the sole appended column, has integer dtype and range
  0 through 9, and exactly matches an independent recomputation from the
  backup. Every prior loaded column/value and row order is identical to the
  backup. Validation of the complete canonical task-decoding target set at
  choice alignment passed. No sibling temporary file remains.
- Bounded write-set evidence: the source replacement and declared backup are
  the only WP9A-created or changed experimental-data paths. Metadata checks for
  all 17 other pre-existing direct children matched the preflight snapshot,
  and no unexpected directory entry remained. The feature-parameter file was
  unchanged at SHA-256
  `4df6dc6df127040be311b1e65caaadd9b79de55d9528bd10cefbf62fe480f9c1`.
- Proposed WP10 command, not executed: create only after separate WP10 approval
  the bounded config
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/task_decoding_config_wp10_bounded.json`,
  then run from the tracked-clean repository root:

  ```bash
  env UV_CACHE_DIR=/tmp/context-inference-uv-cache \
    MPLCONFIGDIR=/tmp/context-inference-matplotlib \
    uv run python -m src.neural_analysis.task_decoding.run_session dry-run \
    --config "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/task_decoding_config_wp10_bounded.json"
  ```

  The proposed config selects ProbeA/PFC and ProbeB/HPC, choice alignment,
  100 ms bins, ten PCs per region, fixed regularization, five outer folds,
  three inactive inner folds, and only `current_action` plus `relative_doubt`.
  Config creation, dry run, neural loading, detached smoke, and benchmark all
  remain unperformed.
- Exact next action: propose the complete WP10 command sequence, identity and
  resource-measurement contract, detached ownership behavior, and stop
  conditions. Obtain explicit user approval before creating the bounded config
  or reading CT026 neural inputs.

#### 2026-10-07 - WP10A resource-measurement instrumentation GREEN gate

- State: WP10A is complete. The pipeline now produces validated atomic
  `resource_usage.json` snapshots after every target checkpoint and on
  completion, failure, or interruption. No CT026 neural array was opened and
  the bounded real-session benchmark has not started.
- Authorization: the user approved the WP10A/WP10B plan, including tests-first
  measurement instrumentation, a later external bounded config, read-only
  dry-run stop conditions, one synthetic detached smoke, and the 2,400-fit
  local fixed benchmark. Full 18-target decoding, tuned benchmarking, transfer,
  and Slurm work remain outside this authorization.
- Sol / Terra / reviewer: primary Codex implemented and reviewed this bounded
  package directly; no subagent was used. The final suite and strict schema
  consistency checks provide the executable gate for this slice.
- Start HEAD / end HEAD: pushed WP9A handoff
  `7c59e862019696d85a91e7639375521cf65effc5a` / WP10A production
  `115311551acbac4ab32d77b86dbbe5667b2bdaba` before this documentation-only
  handoff.
- Owned files: added
  `src/neural_analysis/task_decoding/resource_usage.py` and its focused test;
  extended `activity.py`, `modeling.py`, `pipeline.py`, `run_batch.py`, the
  activity/modeling/pipeline/batch/synthetic/documentation tests, and both
  neural-analysis READMEs. Pre-existing untracked files remain untouched.
- RED commands and results: the initial focused contract run reported ten
  intended failures and nine passes: missing measurement module, missing dry
  row/fold diagnostics, unsupported timing callback, and acceptance of partial
  batch evidence. A later schema supplement independently failed on a negative
  nested estimator count before the validator was tightened.
- GREEN/regression commands and results: the focused WP10 selection passed 59
  tests in 16.72 seconds. After schema tightening, resource/batch/detached
  tests passed 29 tests and the seeded integration file passed eight tests.
  The final complete task-decoding plus documentation suite passed 816 tests
  with the same six known warnings in 90.08 seconds. Package compilation and
  `git diff --check` passed.
- Commits: initial RED `fb0278a`; evidence-consistency tests supplement
  `cc0cf7d`; GREEN production and maintainer/scientist documentation `1153115`.
- Measurement contract: Linux `ru_maxrss` is normalized from KiB to bytes;
  wall/user/system CPU totals accumulate across resume while peak RSS uses the
  maximum. Target records separate split, PFC/HPC transform, feature-building,
  and estimator timing by region/representation. Restored checkpoints without
  historical evidence are explicitly timing-unavailable rather than assigned
  fabricated values. Completed evidence must cover the exact target envelope
  and its nested/aggregate fit counts must agree before batch admission.
- Dry-run contract: `ActivityDryRunReport` now exposes bilateral full-table row
  positions. Planning reports, in configured order, target family, common
  eligible row count, block count, categorical class counts or numerical
  range, and grouped outer-plan availability/reason without reading sorter
  spike arrays or fitting models.
- Synthetic/detached evidence: three real seeded foreground runs published
  complete measurement evidence, and an actual Linux detached child published
  the same atomic schema through the production launcher. These were pytest
  temporary outputs only.
- Real-data or external actions: none. No CT026 config was created, no CT026
  neural source was opened, and no benchmark, transfer, network, or scheduler
  action occurred.
- Findings and unresolved risks: the new evidence measures the exact executing
  process and is suitable for the bounded local projection, but it does not
  make a full-runtime projection by itself. WP10B must still verify both target
  split diagnostics, then record observed target-family throughput, CPU
  efficiency, peak RSS, and output bytes. The local commits are four commits
  ahead only after this handoff and must be pushed before persistent evidence
  is launched.
- Exact next action: commit this handoff, have the user push the four WP10A
  commits, verify `HEAD == origin/refactor`, then create the approved bounded
  CT026 config and run only the recorded dry-run command. Stop on any target
  diagnostic failure rather than substituting a target.

#### 2026-10-07 - WP10B evidence and WP10C revision-6 convergence repair

- State: the bounded CT026 v1 benchmark completed and passed saved-result
  validation, but its categorical invalidity failed the scientific readiness
  gate. The approved WP10C source repair is locally complete; replacement v2
  real-session computation has not started.
- Authorization: after reviewing the benchmark, the user explicitly chose to
  fix convergence locally before any cluster work and approved
  `max_iter=5000`, analysis v2, tests-first implementation, a repeated bounded
  run, and a later local eight-target categorical stress run. WP11 remains
  unauthorized.
- Sol / Terra / reviewer: primary Codex implemented and reviewed the bounded
  change directly; no subagent was used.
- Start HEAD / end HEAD: pushed WP10A handoff `b327d36` / WP10C production
  `6f9c721` before this documentation handoff.
- Owned files: tests extended
  `test_config.py`, `test_results.py`, and
  `test_task_decoding_documentation.py`; production changed only
  `config.py`; documentation added `task_variable_spec_v6.md` and updated this
  plan plus the package README. Pre-existing untracked files remain untouched.
- RED command and result: the four-test focused run produced the intended
  three failures and one pass in 5.1 seconds. Failures were exactly v1 instead
  of v2, the old 100-iteration scientific payload, and the absent revision-6
  amendment. The existing convergence-warning-invalidity test passed.
- GREEN/regression commands and results: the same focused selection passed
  four tests; complete configuration/modeling/results/documentation coverage
  passed 561 tests with two known warnings; the full task-decoding plus
  documentation suite passed 819 tests with six known warnings in 137.26
  seconds. Package compilation and `git diff --check` passed.
- Commits: RED `ba47795`; GREEN `6f9c721`.
- V1 benchmark evidence: run
  `task_variable_decoding_2026-10-07T18-15-16Z` completed 2,400 estimator calls
  in 203.715 seconds wall time with 984,047,616 bytes peak RSS, 0.995 CPU
  efficiency, and 53,932,914 output bytes. All 1,200 numerical cells were
  valid. Only 513/1,200 categorical cells were valid: all 600 direct-unit
  cells and 87 PCA cells recorded `fit_convergence_failure` under
  `max_iter=100`.
- Scientific decision: revision 5 remains the immutable base contract;
  revision 6 is a narrow normative amendment. It changes only the analysis
  identity to `task-variable-decoding-v2` and the logistic ceiling to 5,000.
  SAGA, tolerance, seed, scaling, PCA, folds, regularization, metrics, and the
  rule that any convergence warning invalidates a fit are unchanged. V1 runs
  remain readable but cannot be reused as v2.
- Real-data or external actions: this package inspected the already completed
  v1 run read-only. No new CT026 neural computation, input mutation, transfer,
  network operation, or scheduler action occurred during WP10C implementation.
- Exact next action: commit this handoff and push the RED, GREEN, and handoff
  commits. After `HEAD == origin/refactor` and tracked cleanliness are proven,
  repeat the bounded dry run and launch a new detached v2 run from an ordinary
  terminal. Stop if any of its 2,400 cells is invalid; otherwise compare the
  overlapping v1/v2 valid cells and proceed to the separately approved local
  eight-target categorical stress configuration.

#### 2026-10-07 - WP10C v2 evidence and WP10D ownership lifecycle repair

- State: revision 6 solved the categorical convergence failure. The first v2
  run is scientifically valid but is not the final operational benchmark
  because a rejected competing resume wrote a false `last_error` while the
  original owner remained active. The tests-first lifecycle repair is locally
  complete; a clean replacement run has not started.
- Authorization: the user approved local lifecycle debugging, a clean bounded
  rerun, and continued deferral of cluster work.
- Sol / Terra / reviewer: primary Codex implemented and reviewed this narrow
  package directly; no subagent was used.
- Start HEAD / end HEAD: pushed revision-6 handoff `aade4cd` / lifecycle GREEN
  `4f131f7` before this documentation handoff.
- Owned files: `test_pipeline.py` and `pipeline.py`; this handoff updates only
  the implementation plan. Pre-existing untracked files remain untouched.
- RED command and result: the focused selection passed three existing
  post-ownership failure controls and failed four new cases as intended: live
  receipt and live guard contenders mutated `run_state.json`, an atomic guard
  race surfaced raw `FileExistsError` after writing contender provenance, and
  successful completion retained a stale mid-run error. A supplement also
  froze byte-for-byte preservation of the winner's `execution.json` during an
  atomic claim loss.
- GREEN/regression commands and results: the focused ownership matrix passed
  17 tests; the complete pipeline module passed 156 tests with one known
  warning; the full task-decoding plus documentation suite passed 823 tests
  with six known warnings in 162.28 seconds. Package compilation and
  `git diff --check` passed.
- Commits: lifecycle RED `c1a64aa`; winner-provenance test supplement
  `b201711`; GREEN `4f131f7`.
- Implementation: `_ExecutionOwnershipConflict` distinguishes rejection by a
  live/foreign owner from a failure owned by the current invocation. A worker
  now claims the exclusive guard before publishing its PID/start token, so a
  losing contender cannot overwrite winner provenance. Ownership conflicts
  propagate without changing lifecycle/resource state. Genuine errors after
  claim retain the existing failed/interrupted publication and guard release.
  Valid final publication explicitly clears `last_error`.
- V2 scientific evidence: run
  `task_variable_decoding_2026-10-07T18-35-21Z` passed full result validation
  and produced 2,400/2,400 valid cells. All 687 formerly unavailable
  `current_action` cells converged. The 513 formerly valid categorical cells
  had exactly equal scores, coefficients, intercepts, and NaN patterns under
  v1/v2; the complete numerical target output was exactly equal. Stable
  target, row, fold, count, and feature arrays were also exactly equal.
- V2 measured resources: 896.584 seconds wall time, 803.954 seconds user CPU,
  90.038 seconds system CPU, 0.997 CPU efficiency, 984,301,568 bytes peak RSS,
  and 62,590,932 output bytes. `current_action` used 813.427 seconds and
  `relative_doubt` 77.107 seconds.
- Operational contamination: sandbox PID visibility initially classified the
  user-launched owner as dead. The escalated competing resume correctly found
  the live guard, but the old broad error handler marked the shared run failed;
  the original owner later completed and published valid results while the
  false `last_error` remained. No result array or scientific input was changed.
- Real-data or external actions: the approved v2 bounded run read CT026 neural
  inputs and wrote only its timestamped analysis directory. Subsequent work
  read the two immutable benchmark runs for comparison. No behavior input,
  transfer, network operation, or scheduler action occurred.
- Exact next action: commit this handoff and push all four WP10D commits. After
  exact remote equality and tracked cleanliness, repeat the bounded dry run
  and launch one new v2 run. Accept it only if lifecycle is complete,
  `final_results_published` is true, `last_error` is empty, result validation
  is valid, and all 2,400 cells are valid. Then plan the already approved local
  eight-target categorical stress run; do not begin WP11.

#### 2026-10-07 - Cluster throughput implementation decision

- State: the clean bounded v2 rerun was accepted from an ordinary terminal as
  `task_variable_decoding_2026-10-07T19-29-08Z` and may finish while this
  documentation-only update is made. Its one-shot completion inspection and
  the local categorical stress gate remain pending.
- Authorization: the user selected cluster execution as a required
  implementation path. This supersedes the earlier WP10B/WP10D records that
  deferred or left WP11 unauthorized; those records remain above as historical
  evidence of the decisions in force at their respective times.
- Throughput decision: approximately 15 minutes is acceptable for one session
  but undesirable when repeated serially over many sessions. A single-session
  Slurm wrapper solves unattended execution and establishes a measured cluster
  resource profile, but by itself does not shorten a multi-session batch.
  Therefore both WP11 and the later WP13 per-session job array are required.
- Concurrency architecture: WP13 uses one independently prepared and resumable
  run per array element, with an explicit resource-evidence admission check and
  concurrency cap. It does not place the existing local `run_batch` process
  inside one large allocation, share mutable result state, or introduce
  within-session parallel fitting.
- Remaining approvals: this decision authorizes planning the required source
  packages; it does not authorize an external transfer, synthetic scheduler
  smoke, CT026 submission, or real multi-session array. Exact resource
  directives and the tests-first WP11 implementation plan are approved after
  the clean benchmark and categorical stress evidence. Every external action
  retains the bounded approval gates in WP12 and WP13.
- Performance evidence: the first scientifically valid v2 run measured
  896.584 seconds wall time, 0.997 CPU efficiency, and 984,301,568 bytes peak
  RSS. These values justify cross-session concurrency but are not yet the
  finalized Slurm request because the clean lifecycle run and categorical
  stress evidence are still pending.
- Repository/external effects: this record changes documentation only. It does
  not alter the active benchmark, scientific source, experimental inputs, or
  any local or remote scheduler state.
- Exact next action: validate the accepted clean run once it finishes, perform
  the approved categorical stress run, and then propose the exact WP11 Slurm
  directives, owned files, RED tests, and validation commands for user
  approval before implementation.

#### 2026-10-07 - WP10D clean bounded v2 acceptance

- State: the replacement run
  `task_variable_decoding_2026-10-07T19-29-08Z` completed and passed the clean
  bounded revision-6 acceptance gate.
- One-shot validation: `status --verify-results` reported lifecycle
  `complete`, `final_results_published=true`, empty `last_error`, valid result
  validation, both expected completed targets, and no warnings. Direct NPZ
  inspection found 2,400/2,400 `fit_status` cells equal to `valid` and no
  nonempty fit-reason code.
- Resource evidence: the run requested and measured 2,400 estimator calls,
  completed in 1,008.113 seconds wall time, used 874.181 seconds user CPU and
  120.207 seconds system CPU (0.986 CPU efficiency), peaked at 983,080,960
  bytes RSS, and wrote 62,591,329 bytes. Activity construction took 4.420
  seconds and modeling took 999.851 seconds.
- Target evidence: `current_action` completed 1,200/1,200 valid cells in
  902.309 seconds; `relative_doubt` completed 1,200/1,200 valid cells in
  97.134 seconds. The categorical workload remains the dominant resource and
  wall-time driver.
- Inputs/outputs: the inspection was read-only and changed neither the run nor
  experimental inputs. The immutable accepted run remains at
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/analysis_runs/task_variable_decoding_2026-10-07T19-29-08Z`.
- Gate decision: WP10D is accepted. The remaining pre-WP11 scientific gate is
  the already approved local run containing exactly the eight canonical
  categorical targets, with every other bounded configuration setting held
  fixed. Its dry run must accept all targets and grouped folds; its completed
  result must contain no convergence-warning cell.
- Exact next action: push the documentation/evidence commits, prove exact
  remote equality, create the stress configuration beside the bounded config,
  run its read-only dry run, and present the exact detached launch command.

#### 2026-10-07 - WP10 categorical stress acceptance and WP11 resource proposal

- State: the exact eight-target categorical stress run
  `task_variable_decoding_2026-10-07T19-50-38Z` completed and passed the final
  pre-WP11 scientific gate. WP10 is complete; WP11 source has not started.
- One-shot validation: `status --verify-results` reported lifecycle
  `complete`, published valid results, empty `last_error`, all eight expected
  completed targets, and no warnings. Direct NPZ inspection found
  9,600/9,600 `fit_status` cells equal to `valid` and no nonempty reason code.
- Determinism: every saved target-dependent scientific array for the shared
  `current_action` target was byte-for-byte/value-for-value equal to the clean
  bounded run, including scores, fold assignments, eligibility, coefficients,
  intercepts, counts, statuses, and fixed parameters.
- Measured resources: wall time was 6,352.307 seconds, user CPU 6,139.717
  seconds, system CPU 200.960 seconds, CPU efficiency 0.998, peak RSS
  978,534,400 bytes, and output size 293,248,510 bytes. Modeling used
  6,346.161 seconds. Per-target totals ranged from 735.519 to 854.896 seconds.
- Full fixed-mode projection: retain the measured complete eight-categorical
  total and add ten times the measured 97.134-second representative numerical
  target, yielding 7,323.649 seconds (2.034 hours). For the first-job
  conservative bound, use eight times the slower independently measured
  902.309-second categorical target plus ten numerical targets and bounded
  overhead: 8,199.818 seconds (2.278 hours). Twice that bound rounded upward
  to an hour gives a five-hour request.
- Memory projection: the measured eight-target result arrays contain
  43,232,976 target-dependent bytes. Scaling their exact per-target capacity to
  all 18 targets adds 54,041,220 bytes to the measured peak, for a projected
  peak of 1,032,575,620 bytes. The Section 12.6 candidates are 1,548,863,430
  bytes at 1.5 times peak, 3,180,059,268 bytes at peak plus 2 GiB, and
  193,903,360 bytes at twice the exact tensor allocation. The maximum rounds
  upward to a 3 GiB Slurm request.
- Proposed frozen directives: partition `unlimited`, job name
  `task_decoding`, one task, one CPU per task, `--mem=3G`,
  `--time=05:00:00`, `--signal=B:TERM@300`, log
  `/gs/gsfs0/users/mchin1/logs/task_decoding_%j.log`, mail type `ALL`, and mail
  user `matthew.chin@einsteinmed.edu`. OMP, MKL, and OpenBLAS remain one thread.
- WP11 architecture: add a thin self-submitting
  `src/shell_scripts/task_variable_decoding_slurm.sh`; add one standard-library
  `task_decoding/slurm.py` operational module for immutable submission receipts,
  `sbatch`/one-shot `sacct` interaction, effective Slurm/cgroup memory limits,
  and JSON-safe status; extend `run_session.py` only for explicit Slurm
  preparation and TERM-to-durable-interruption handling; keep bounded atomic
  lifecycle transitions in `pipeline.py`; and update both neural READMEs.
  Scientific configuration, fitting, checkpointing, and results remain in the
  existing runner and are not duplicated.
- Dependencies: no Python dependency is added. Runtime tools are Bash, Git,
  existing `uv`, Slurm `sbatch`/`sacct`, and documented manual `rsync`.
- RED ownership: create `test_slurm.py` for the Section 10.11 scheduler,
  repository/environment, memory, receipt-race, status, and transfer-contract
  cases; extend `test_pipeline.py` for signal durability and the private
  prepare/execute seam; extend `test_task_decoding_documentation.py` for exact
  commands and safe rsync prose. Tests use fake commands and temporary data;
  they never invoke Slurm, SSH, rsync, network access, or experimental data.
- Performance boundary: WP11 remains one CPU and one session per allocation.
  It adds no within-session process/thread parallelism. Cross-session
  concurrency remains required WP13 work after one cluster session validates
  the wrapper and measured resource profile.
- External effects: acceptance inspection was read-only. No transfer,
  scheduler, SSH, network, or new computation occurred. The new stress config
  and run remain beside the CT026 session; repository source is unchanged.
- Exact next action: obtain user approval of the directives and tests-first
  architecture, commit RED tests, reproduce the intended failures, and stop
  for the tests-only gate before implementing production.

#### 2026-10-07 - WP11 single-session Slurm RED gate

- Authorization: the user approved the `unlimited` / one-task / one-CPU /
  3-GiB / five-hour resource block and the tests-first WP11 architecture.
- State: tests-only RED is committed at `a3dcdb9`; no WP11 production or
  documentation implementation has started.
- Owned tests: new `test_slurm.py` exercises the Bash wrapper through fake
  `uv`, repository/source gates, exact public/private forwarding, frozen
  resources, submission success/failure/race behavior, one-query read-only
  accounting, exact resume, and login/active-allocation memory limits.
  `test_pipeline.py` adds explicit Slurm preparation, TERM translation, and
  pre-spike active-memory integration contracts.
  `test_task_decoding_documentation.py` adds exact commands, offline setup,
  safe rsync, hidden incoming validation, and resource documentation.
- RED result: the focused WP11 selection failed 20 tests as intended. Causes
  were exclusively the absent wrapper and `slurm.py`, absent
  `--execution-mode slurm` private preparation, absent SIGTERM bridge and
  active-allocation guard, and the intentionally stale WP8 cluster prose.
- Existing controls: the pre-existing pipeline and documentation tests passed
  165 tests with one known empty-slice plotting warning; three new WP11 tests
  were excluded from that control command. Test modules compile, and
  `git diff --check` passed.
- Safety: fake executables and temporary repositories were used. No real
  `sbatch`, `sacct`, SSH, rsync, network, cluster, or experimental-data action
  occurred.
- Exact next action: after user approval of the tests-only commit, implement
  the bounded WP11 production files and documentation, then run focused and
  affected regression suites. Stop before any real cluster action.

#### 2026-10-07 - WP11 single-session Slurm GREEN gate

- Authorization: the user confirmed the tests-only push and approved WP11
  production implementation. No real scheduler or transfer action was
  authorized.
- State: production is committed at `b9d8645`. The implementation adds the
  executable `task_variable_decoding_slurm.sh`, standard-library `slurm.py`,
  explicit Slurm preparation, durable TERM translation, an active
  allocation/request memory guard before spike loading, and scientist/
  maintainer operating instructions.
- Source behavior: the login branch checks the exact repository root,
  tracked-clean state, tracked direct dependencies, and untracked package
  Python; it invokes only frozen/no-sync/offline `uv`. Submission records the
  current commit before one `sbatch` call and exports it so the scheduled
  branch refuses a checkout that moved while queued. The scheduled branch can
  only execute one exact prepared directory. Status performs one read-only
  `sacct` query. Submission never retries or selects a latest run.
- Lifecycle behavior: `submission-pending` is durable before `sbatch`;
  submission failure becomes `failed`; success writes only the atomic Slurm
  receipt so a fast job retains ownership of later state. Slurm TERM becomes
  the pipeline's existing `KeyboardInterrupt` boundary and preserves valid
  checkpoints. Effective memory is the conservative minimum of available,
  cgroup, and reviewed 3-GiB limits, with the existing 50-percent tensor guard.
- Verification: the 20 focused WP11 contracts passed. The complete affected
  runner/pipeline/documentation selection passed 185 tests with one known
  empty-slice plotting warning. `test_results.py` passed 373 tests, and the
  complete task-decoding package passed 833 tests with six known warnings.
  Bash syntax, Python compilation, and cached-diff whitespace checks passed.
- Start HEAD / end HEAD: `e91fa5f` / `b9d8645` before this documentation
  handoff. Only the six WP11 production/documentation files were committed;
  historical untracked workspace files were untouched.
- Real-data or external actions: none. No real `sbatch`, `sacct`, SSH, rsync,
  network access, cluster computation, or experimental computation occurred.
- Exact next action: push the GREEN and handoff commits. Then plan the
  separately approved WP12 synthetic smoke using exact cluster paths and
  command; do not transfer or submit until that action is approved.

#### 2026-10-07 - WP12 synthetic preflight and bytecode repair

- Authorization: the user approved read-only HPC preflight, creation and
  transfer of the deterministic synthetic fixture, local/cluster dry runs, the
  exact synthetic submission, and the tests-first repair after that submission
  attempt stopped before `sbatch`.
- Preflight: the canonical checkout is
  `/gs/gsfs0/home/mchin1/context-inference`, branch `refactor`, initially clean
  at pushed `4238d52`. `uv`, `sbatch`, `sacct`, `rsync`, the executable wrapper,
  frozen environment, writable log directory, and active `unlimited` partition
  were present. Cluster Python is 3.14.7 versus local 3.12.12; locked numerical
  and scientific package versions match.
- Fixture: seed `20261007` base session was generated below a unique local
  `/tmp` directory and transferred outside Git to
  `/gs/gsfs0/users/mchin1/task_decoding_smoke/wp12-seed-20261007-4238d52/`.
  All 17 files matched aggregate SHA-256
  `2bdf2f9c74d04d5821b5ba3f2b0374a447b91cf0856a8911d48edc6816298ff2`.
  Local and cluster dry runs matched: 72 trials, 3 targets, 432 outer cells,
  36,864 tensor bytes, source fingerprint
  `a3d504862ba519906ac41c57f15894381ed3df45676c23dd8dd17c4d707b52a6`,
  expected unavailable `current_state`, and available `current_action` plus
  `relative_doubt`.
- Submission outcome: the exact approved `submit-new` entered Python but
  preparation rejected package `__pycache__/*.pyc` as untracked scientific
  source. Read-only confirmation found no prepared run, submission receipt, or
  Slurm job. Ten generated `.pyc` files are the only remote package artifacts;
  tracked state remains clean.
- Repair: tests-only RED `849c7ea` produced seven intended failures while
  retaining genuine untracked `.py` rejection. GREEN `db50e1a` makes both
  source gates distinguish `.py` from `.pyc` and exports
  `PYTHONDONTWRITEBYTECODE=1` before either wrapper Python path.
- Verification: 12 focused repair contracts passed; complete Slurm/results
  suites passed 390 tests; the full task-decoding package passed 834 tests with
  the same six known warnings. Bash syntax and diff checks passed.
- External effects: SSH preflight, one new synthetic transfer, cluster dry-run,
  and one failed-before-scheduler submission attempt occurred. No experimental
  data moved, no immutable run was prepared, and no Slurm job was submitted.
- Exact next action: push the repair/handoff, remove only the generated remote
  `task_decoding/__pycache__`, update the clean cluster checkout to the exact
  pushed commit, reverify, and request renewed approval for one submission.

#### 2026-10-07 - WP12 synthetic smoke completion and reporting repair

- Authorization and execution: after the bytecode repair was pushed and the
  cluster checkout was clean at `618a29a`, the user approved one corrected
  synthetic submission. Slurm accepted job `30984881` for prepared run
  `/gs/gsfs0/home/mchin1/task_decoding_smoke/wp12-seed-20261007-4238d52/base-session/analysis_runs/task_variable_decoding_2026-10-07T23-55-04Z`.
- Scientific result: the run completed with `final_results_published=true`,
  no warning or error, all expected files, three valid PNGs, and valid result
  revalidation. It requested 432 outer cells: 288 were valid and the 144
  `current_state` cells were unavailable exactly as declared by the synthetic
  fixture; `current_action` and `relative_doubt` each contributed 144 valid
  cells.
- Durable resource evidence: `resource_usage.json` records 5.112 seconds wall
  time, 1.824 seconds user CPU, 0.250 seconds system CPU, 339,750,912 bytes peak
  RSS, 2,422,271 output bytes, and 288 measured estimator calls. The empty
  scheduler log is consistent with a successful job that emitted no standard
  stream output.
- Scheduler-accounting limitation: the required one-shot exact-root `sacct`
  query returned no row even though cluster Slurm is configured with
  `accounting_storage/slurmdbd`; the completed job had already left `squeue`.
  The old status command raised and hid the otherwise valid durable completion
  state. Consequently the synthetic result is scientifically valid, but
  cluster accounting fields such as Slurm `MaxRSS` are unavailable for this
  evidence run and must not be fabricated.
- Summary defect: immutable `summary.md` reported the earlier running snapshot
  (290,217,984 bytes peak RSS and pre-figure CPU values), while the terminal
  structured resource evidence correctly included figure generation. The
  summary is therefore not authoritative for this run's resource totals.
- Approved repair: tests-only commits `86fc413` and `5386a99` freeze graceful
  empty/failed-accounting fallback, successful-accounting compatibility,
  post-figure complete evidence in the summary, and the corresponding
  late-publication/resume order. Production commit `a1763bc` returns durable
  pipeline status plus `scheduler=null` and `scheduler_error` when the one-shot
  accounting query is unavailable, and renders figures before taking the
  terminal resource snapshot supplied to `summary.md`.
- Verification: the focused GREEN selection passed seven tests; complete
  pipeline/Slurm regression passed 178 tests with one known warning; and the
  full task-decoding package passed 837 tests with the same six known warnings.
  No scientific setting, result schema, or scheduler resource directive
  changed.
- External effects and immutability: inspection was read-only. The completed
  synthetic run is not rewritten or rerun merely to update its immutable
  summary; its `resource_usage.json` remains the authoritative resource record.
- Post-repair cluster validation: the four repair/handoff commits were pushed,
  generated repository bytecode/egg metadata was removed with explicit user
  approval, and the otherwise clean cluster checkout fast-forwarded to exact
  commit `34da429`. The repaired wrapper status returned the durable completed
  state, all three targets, no warnings, `scheduler=null`, and
  `scheduler_error="sacct returned no job-level record for 30984881."` without
  raising or writing. A separate `run_session status --verify-results` returned
  `result_validation="valid"`; the checkout remained clean afterward.
- Gate decision: the WP12 synthetic scheduler/environment/logging smoke is
  accepted without rerunning its immutable computation. Missing Slurm
  accounting remains an explicit site/accounting limitation, not a failed or
  inferred job state. Future inspections retain the same one-query boundary
  and use structured in-run resource evidence when no scheduler row exists.
- Exact next action: present the separate WP12 CT026 input-transfer, cluster
  dry-run, and one-session submission plan for user approval. Do not infer
  authorization for experimental transfer or computation from synthetic-smoke
  acceptance.

#### 2026-10-08 - WP12 full CT026 transfer and dry-run gate

- Authorization: the user approved a full 18-target fixed-mode configuration,
  preferred all cluster data grouped below the actual mirrored session root,
  and separately approved the input transfer plus cluster dry run. Slurm
  submission remains unapproved at this gate.
- Configuration: created
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference/task_decoding_config_wp12_full_fixed.json`
  with SHA-256
  `cf682e381480ece5f00e63f7957baa47c5d5f24b0e833cc180fe745431c2032f`,
  mode `0664`, and owner/group `1000:1000`. It differs from the accepted WP10
  bounded config only by selecting the canonical ordered 18-target vocabulary.
- Grouped destination: reused the pre-existing mirrored session at
  `/gs/gsfs0/users/mchin1/contextProjectData/CT026/CT026_20260803_latent_inference`.
  No new task-decoding data root was created. Future session inputs should use
  the same `contextProjectData/<animal>/<session>` hierarchy; run directories
  remain inside each session's `analysis_runs`.
- Transfer decision: a non-writing whole-session preview found 9,825 changed
  regular files and 4,417,205,615 bytes, almost entirely unrelated ephys
  products. Size was not a cluster constraint, but the validated pipeline
  source contract required only 12 files. The user approved that selective
  source/config transfer. Checksum mode transferred seven genuinely changed or
  absent files totaling 235,576,718 bytes, performed no deletion, and left
  identical already-present files in place. A post-transfer checksum preview
  reported zero created, deleted, or transferred files across the 12-file
  274,713,572-byte source set.
- Matching dry runs: local Python 3.12.12 and cluster Python 3.14.7 both
  reported clean scientific source fingerprint
  `d18b07fd1985cda2a8c1b3cd787b4f8e6aec99b6f07dc6fdaa693cf92c8fb8be`,
  identical package versions, exact source sizes, ordered targets, and target
  diagnostics. Both resource envelopes contain 650 full trials, 646 tensor
  trials, 160 PFC units, 309 HPC units, 40 bins, 18 targets, five outer folds,
  9,600 categorical fits, 12,000 numerical fits, 469 direct-unit coefficient
  capacity, and 96,951,680 tensor bytes. Every target has an available grouped
  outer split.
- Read-only final checks: the cluster checkout remained clean at `54ee2ca`,
  and the mirrored session has no `analysis_runs` directory. No prepared run,
  submission receipt, scheduler job, or neural result was created.
- Performance/resource boundary: the reviewed request remains partition
  `unlimited`, one task, one CPU, 3 GiB, five hours, five-minute TERM notice,
  one numerical thread, and the established private log path. The fixed-mode
  projection remains approximately 2.03 hours nominal and 2.28 hours at the
  conservative bound; the request retains the approved margin.
- Submission receipt: with separate explicit user approval, the wrapper
  accepted Slurm job `30985392` at
  `2026-10-08T00:34:16Z` from exact commit
  `c086bdf55e08d062bec0842ac453af8dd0d29710`. The immutable run directory is
  `/gs/gsfs0/home/mchin1/contextProjectData/CT026/CT026_20260803_latent_inference/analysis_runs/task_variable_decoding_2026-10-08T00-34-15Z`;
  its scheduler log is
  `/gs/gsfs0/users/mchin1/logs/task_decoding_30985392.log`.
- Exact next action: leave the submitted run unattended and do not change the
  cluster checkout while the exact-commit job may run. In a later
  user-requested task, issue the receipt's one-shot status command and inspect
  durable state/log/results once. Do not poll or resume automatically.

## 1. Objective

Implement a readable, single-session task-variable decoding pipeline that:

1. consumes the existing augmented behavioral table and aligned spikes;
2. constructs deterministic categorical and numerical targets;
3. compares PFC, HPC, and PFC + HPC using regional PCA or direct units;
4. evaluates elastic-net decoders with grouped, leakage-safe CV;
5. saves complete offline results and provenance;
6. displays saved results in one read-only view inside the existing Streamlit
   webapp; and
7. reports realistic runtime and memory expectations before routine use.

The implementation should be direct scientific Python. It should use small
functions, explicit arrays and tables, frozen configuration dataclasses, and
docstrings that state types, shapes, axes, units, and return values. It should
not introduce a framework, plugin system, estimator hierarchy, or generalized
workflow engine.

## 2. Design constraints

### 2.1 Simplicity and readability

- Prefer module-level functions to behavior-heavy classes.
- Use dataclasses only for configuration and compact data records.
- Pass narrow arrays/series to computational helpers when dataframe-level row
  alignment is not their responsibility.
- Keep transformations visible: trial selection, rate binning, scaling, PCA,
  fitting, scoring, and aggregation should be separately testable steps.
- Reuse existing loaders and rate-tensor code when their contracts match.
- Avoid generic `utils.py` or `helpers.py` modules.

### 2.2 Scope boundaries

The first implementation will not:

- modify the neural-session metadata schema;
- refactor or replace the existing exploratory PCA decoder;
- support multiple probes per brain region;
- pool units across sessions;
- compute permutation significance or decoding onset;
- standardize numerical targets;
- tune PC counts, bin widths, class weights, or thresholds;
- run decoding inside Streamlit;
- add a second viewer application;
- add a new dependency; or
- perform major performance optimization before profiling.

### 2.3 Required workflow

Every implementation package follows Red-Green-Refactor:

1. write focused tests;
2. run them and record the expected failure;
3. commit the tests before implementation;
4. implement the smallest coherent package;
5. run focused and affected regression tests; and
6. refactor only while tests remain green.

No production analysis is run until unit tests and a synthetic end-to-end
pipeline are green. After the separately approved WP9A behavior-table
preparation, CT026 neural work begins with a read-only preflight and bounded
benchmark, not a full tuned analysis.

## 3. Audited codebase baseline

### 3.1 Reusable code

| Existing code | Planned use |
| --- | --- |
| `src/neural_analysis/session_metadata.py` | Load and resolve authoritative session/probe paths. Do not duplicate metadata parsing. |
| `spike_behavior.loading.load_sorter_metadata` | Load spike-cluster assignments and curated cluster metadata. |
| `spike_behavior.loading.load_aligned_spikes` | Load aligned UTC spike timestamps. Extend validation outside this function rather than changing its public contract unnecessarily. |
| `spike_behavior.loading.validate_aligned_spike_inputs` | Confirm spike/cluster one-to-one length. |
| `spike_behavior.loading.build_spike_tsgroup` | Construct Pynapple unit spike series. |
| `spike_behavior.loading.load_channel_quality` and `select_channels_from_quality` | Load channel metadata and apply existing quality/inside-brain selection. |
| `spike_behavior.loading.filter_cluster_metadata` | Apply configured cluster-quality groups to selected channels and retain sorted cluster/channel/normalized-quality metadata. Reuse it unchanged. |
| `spike_behavior.loading.select_units_by_channels` | Existing fixed `good`/`mua` convenience API for older callers. Leave it unchanged; the new decoder uses `filter_cluster_metadata`. |
| `population.pca.build_trial_unit_rate_tensor` | Produce trial x time x unit unsmoothed firing rates in Hz. |
| `behavior_analysis.session_analysis.make_augmented_trial_df` | Add the general `rewards_in_block` feature at the existing augmentation boundary. |
| `behavior_analysis.project_utils.is_present_value`, `is_zero_flag`, and `make_no_choice_action_mask` | Reuse project missing-sentinel, manual-flag, and no-choice semantics in target validation instead of defining decoder-only variants. |
| `webapp.session_inputs` | Own the new view name and availability entry. |
| `webapp.app._start_metadata_webapp` | Existing early view-selection boundary. Route saved decoding results before population controls and raw spike loading. |

The Pynapple-backed population rate-tensor function is the primary binning
path. The NumPy implementation remains useful as a test/reference path; the new
pipeline should not fork another spike-binning implementation.

### 3.2 Code not reused as the decoding core

`src/neural_analysis/population/decoding.py` implements a different,
exploratory/limited analysis: choice-only windows, ungrouped stratified CV,
accuracy/permutation scoring, and one-probe PCA controls. Its model-fitting
functions do not meet revision-5 targets, grouping, metrics, regional, or
saved-result contracts.

`src/neural_analysis/population/pca.py::fit_population_pca` is also not the
fold-local modeling interface. It requires at least two units, whereas
revision 5 explicitly permits one usable training unit and one retained
component. The new modeling package owns its fold-local scikit-learn PCA fit;
only the existing dimensional-cap formula is shared.

Do not stretch those functions with many flags. Leave their public behavior
unchanged and build the revision-5 pipeline in a focused package. Shared rate
binning is reused at the lower boundary.

### 3.3 Resolved inconsistencies from revision 4

| Revision-4 ambiguity or mismatch | Resolution in revision 5 and this plan |
| --- | --- |
| Metadata may point to raw trials | Configuration explicitly names augmented CSV and feature-parameter JSON. |
| Missing number-of-rewards-in-block target | Add `rewards_in_block` to general behavior augmentation. |
| HMM columns say `logodds` | Source code returns tanh-transformed signed belief; keep column names but correct scientific labels. |
| State parser supports dark state 2 | Binary current-state decoding accepts only states 0 and 1. |
| Region identity could be inferred from probe names | Require explicit PFC/HPC probe and channel-selection configuration. |
| Cluster IDs can collide across probes | Persist `probe_id:cluster_id` identities. |
| Trial and alignment eligibility were incomplete | Define baseline, target-specific, matched-trial, and full-window coverage masks. |
| PC count could be silently reduced | Use the exact dimensional component cap and record requested/effective count plus visible cap status. |
| Solver warnings could be ignored | Treat nonconvergence as an invalid candidate/fold. |
| Separate result viewer was proposed | Use one integrated, read-only existing-webapp view. |
| Runtime was unknown | Add bounded synthetic and CT026 benchmark gates. |

## 4. Proposed package structure

### 4.1 Behavior change

`src/behavior_analysis/session_analysis.py`

- Add a small, independently tested function that computes
  `rewards_in_block` from `cur_block`, `action`,
  `reward`, and normalized experimenter-reward flags.
- Reuse `should_skip_decision_variable_update(...)` and
  `parse_decision_variable_update_values(...)` so manual/no-choice and
  malformed-value semantics are not duplicated.
- Strengthen that shared parser to require finite rewards and exact 0/1
  actions; the current `int(float(action))` path otherwise accepts fractional
  actions, and non-finite rewards can be silently treated as unrewarded.
- Call it from `make_augmented_trial_df`.
- The helper preserves row count, row order, index, and every existing column;
  integration appends one public column without changing
  `make_augmented_trial_df`'s existing experimenter-reward alias normalization
  or other outputs.

`src/behavior_analysis/gather_trial_features.py`

- Add one thin, tested `backfill_rewards_in_block_csv(path) -> None` migration
  function for already saved augmented tables. It loads with the project's CSV
  sentinel convention, requires the canonical `cur_block`, `action`, `reward`,
  and `experimenter_reward_given` source columns, refuses a table that already
  has the destination column, and calls the same general helper. It writes a
  uniquely named sibling temporary CSV, reloads it with the same convention,
  verifies that its row order and every prior loaded column/value are unchanged
  and only the new column was added, and only then publishes with `os.replace`.
  A missing source or failed round-trip check leaves the original file
  untouched. Use `try`/`finally` around temporary-file creation, serialization,
  reload, and validation. Remove the sibling temporary file if it still exists
  after any success or failure path, and surface a cleanup failure rather than
  silently leaving an undeclared artifact.
- It does not recompute model-derived features, rewrite
  `trial_feature_params.json`, create a decoder-specific table, or contain a
  second counting implementation. The top-level neural README documents the
  exact `uv run` invocation only as recovery for an older table that fails
  lightweight validation solely because `rewards_in_block` is absent. Tables
  with other schema gaps must use the normal behavior-processing path.

The helper should iterate once in chronological row order. A dataframe groupby
expression is not preferred if it obscures the entering-trial update order or
manual/no-choice behavior.

### 4.2 New analysis package

Create `src/neural_analysis/task_decoding/` with:

| Module | Responsibility | Principal public interface |
| --- | --- | --- |
| `__init__.py` | Mark the focused package boundary without broad re-exports. | Package marker only |
| `config.py` | Frozen user settings, analysis version, region definitions, target/fold vocabularies, defaults, scientific/execution separation, JSON serialization, and lightweight value validation. | `ANALYSIS_VERSION`, `TARGET_IDENTIFIERS`, `TaskDecodingConfig`, `RegionConfig`, `load_task_decoding_config(...)`, `scientific_config_payload(...)` |
| `targets.py` | Augmented-table validation, chronological shifted targets, source mappings, and target-specific eligibility. | `build_target_table(...)`, `validate_augmented_trials(...)` |
| `activity.py` | Probe loading, channel/unit selection, trusted coverage, matched trial windows, rate tensors, stable feature identities. | `load_region_activity(...)`, `build_session_rate_tensors(...)` |
| `modeling.py` | Grouped splits, fold-local standardization/PCA, fixed/tuned elastic-net fits, metrics, coefficients, and validity. | `make_outer_splits(...)`, `decode_target(...)` |
| `results.py` | Input and scientific-source manifests, in-memory result record, NPZ/config/log save-load contract, completion state, and compatible-run discovery. | `build_input_manifest(...)`, `scientific_source_fingerprint(...)`, `save_task_decoding_run(...)`, `load_task_decoding_run(...)` |
| `plotting.py` | Light-mode heatmaps, coefficient summaries, captions, and PNG export from saved results. | `plot_decoding_heatmap(...)`, `plot_unit_coefficients(...)` |
| `pipeline.py` | Assemble preparation, loading, targets, activity, decoding, checkpoints, logging, and reporting for one session. | `plan_task_decoding_session(...)`, `prepare_task_decoding_run(...)`, `run_prepared_task_decoding(...)` |
| `run_session.py` | Standard-library-only single-session entrypoint that sets numerical thread limits before lazy pipeline imports; provides `dry-run`, `new`, `resume`, and read-only `status`, with optional unattended detachment. | `main(...)` |
| `run_batch.py` | Standard-library-only session-list entrypoint with the same early thread limits and lazy imports; provides `dry-run` and `new` with session-level parallelism only. | `main(...)` |
| `README.md` | File-by-file ownership, dependency direction, public entry points, array/result contracts, and developer extension notes. | Documentation only |

No inheritance or Protocol is needed initially: there is one loader path and
one model family per target family. Add an abstraction only if a second real
implementation creates a concrete need.

Every initial run computes all six region/representation combinations: PFC,
HPC, and PFC + HPC crossed with PCA and direct units. The webapp selectors only
choose which saved result to display; they do not control computation.

### 4.3 Existing webapp integration

Add `src/neural_analysis/webapp/task_decoding_views.py`.

This module:

- lists direct child run directories under the selected results root, default
  `<session_root>/analysis_runs`;
- accepts one optional session-relative results-root locator for a run that
  used a nondefault configured output root, rejects paths outside the selected
  metadata session, and never recursively scans the full session tree;
- ignores hidden incoming transfers and any run not both marked complete and
  valid under the saved-result loader;
- loads only saved result/config files;
- renders selectors, heatmaps, fold details, and coefficients; and
- contains no estimator or raw-spike loading call.

Add one `PLOT_VIEW_TASK_DECODING` constant and option in
`webapp/session_inputs.py`. In `webapp/app.py`, route this
view immediately after the metadata session and view are selected, before
`_metadata_population_controls` and
`load_metadata_viewer_data_cached`.

The initial result view is metadata-session-only. The legacy manual-path route
does not have a reliable session root/run directory contract and should show a
short message directing the user to launch with `neural_session.json`
rather than adding an unrestricted filesystem browser.

For a metadata session, keep the view selectable even when no completed run
exists or raw behavior/spike sources are currently unavailable. The view itself
then explains that no saved run is available. This requires a small explicit
availability case rather than letting the existing generic spike-readiness
rule classify it as a live-computation view.

### 4.4 Required user and developer documentation

Documentation is a required implementation deliverable, not cleanup after the
scientific code is complete.

Update `src/neural_analysis/README.md` with a short
**Task-variable decoding** section that explains:

- required metadata, augmented-trial, feature-parameter, and configuration
  files;
- how the normal behavior-processing path creates the augmented table; when
  `rewards_in_block` is its only schema gap, how to add that column with the
  tested behavior-side backfill after making a timestamped backup; and why any
  other schema gap requires normal behavior regeneration rather than
  hand-editing the CSV or rerunning unrelated models unnecessarily;
- how to copy and edit the example configuration;
- the exact local `dry-run`, foreground and detached `new`, detached
  `resume`, and read-only `status` commands;
- how to start an unattended run, close the terminal/Codex task, and inspect
  its durable state and logs once later without polling;
- how to perform a batch dry run and batch launch, including the one-worker
  default and optional explicit measured-run evidence for safe parallelism;
- where run directories, checkpoints, logs, NPZ results, summaries, and PNGs
  are written;
- how to open the saved results in the existing webapp, including the
  session-relative locator for a nondefault output root;
- the difference between fixed and tuned mode, including the large tuned-mode
  work-count warning;
- the bounded two-target benchmark recipe, recorded timing/memory fields, and
  how to use its projection before choosing a full local or Slurm run;
- how matching completed runs are skipped and how an explicit rerun creates a
  new immutable directory;
- the single-session Slurm submit/resume/status path, exact-commit/offline
  environment gates, safe input/result rsync, and finalized resource block
  implemented by WP11; and
- the array batch command after the required, separately gated WP13 package.

Create `src/neural_analysis/task_decoding/README.md` with:

- one concise paragraph describing the package and its dependency direction;
- a table describing every package Python file and its public entry points,
  plus the new webapp view, optional Slurm wrapper, and example configuration;
- input dataframe columns and rate/result array shapes, axes, and units;
- configuration and result-schema summaries;
- which existing modules are deliberately reused;
- how target eligibility, fold validity, leakage prevention, and checkpoints
  work;
- how the common neural-eligible tensor rows map back to the complete target
  table and why invalid alignment rows are not zero-filled;
- how portable relative paths, cluster execution paths, termination, returned
  results, single-writer guards, atomic publication, and resource provenance
  work;
- a map from package files to their focused tests;
- a short recipe for adding a future target without changing unrelated
  modules; and
- explicit non-goals so a future maintainer does not merge this work into the
  older exploratory decoder.

Add `docs/examples/neural_analysis/task_decoding_config.json` as a
portable template. It uses obvious placeholder paths and probe IDs, contains
all required fields, and is accepted by the same config loader used for real
runs. Do not put CT026 absolute paths in the reusable example.

The top-level README is for scientists running the analysis. The in-package
README is for maintainers reviewing or extending it. Neither should duplicate
the complete scientific specification.

### 4.5 Straightforward offline command surface

The single-session CLI is the primary offline interface. A normal local
workflow is shown below. `dry-run` may be used during development, but a
persistent `new` or `resume` requires the scoped scientific source files in
Section 8.1 to be tracked and clean.

```bash
cp docs/examples/neural_analysis/task_decoding_config.json \
  /path/to/session/task_decoding_config.json

uv run python -m src.neural_analysis.task_decoding.run_session dry-run \
  --config /path/to/session/task_decoding_config.json

uv run python -m src.neural_analysis.task_decoding.run_session new \
  --config /path/to/session/task_decoding_config.json --detach

uv run python -m src.neural_analysis.task_decoding.run_session status \
  --run-directory /path/to/session/analysis_runs/task_variable_decoding_<timestamp>
```

Omit `--detach` when an intentionally small run should remain in the
foreground. A detached `new` launch must do only bounded setup in the calling
process: validate small inputs, create the immutable run directory, write the
saved configuration/manifests/state and exact follow-up commands, start one
detached child with its console streams redirected into the run directory, and
return. It prints the run directory, local PID, log paths, and exact status and
resume commands. Before printing success, it atomically writes a separate
`local_launch.json` receipt with the child PID and process-start token so an
immediate resume cannot race the child's guard creation or overwrite
child-owned execution state. It does not poll the child.

The matching unattended resume command is:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session resume \
  --run-directory /path/to/session/analysis_runs/task_variable_decoding_<timestamp> \
  --detach
```

The entry modules set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1` before lazily importing NumPy/scikit-learn or the
pipeline. They record these values in execution provenance. The Slurm wrapper
sets the same values explicitly, so the initial local benchmark and one-CPU
cluster projection have the same threading identity.

The detached child uses the same foreground pipeline, saved configuration,
and target checkpoints as an ordinary invocation. Use the Python standard
library process launcher with a new process session and explicit file handles;
do not require `nohup`, a terminal multiplexer, a notebook, Streamlit, or a
resident Codex task.

Preparation and execution have one internal contract shared by all launch
paths:

- `prepare_task_decoding_run(config_path, rerun, execution_mode) -> Path`
  performs bounded validation, creates the immutable run directory, and saves
  the configuration, manifests, initial state, and exact follow-up commands;
- `run_prepared_task_decoding(run_directory) -> None` loads only that saved
  configuration, claims the run's single-writer guard, and executes or resumes
  the foreground scientific pipeline; and
- local foreground `new`, the detached child, and a Slurm compute job all call
  these same functions rather than reconstructing scientific settings.

`run_session.py` may expose private `_prepare` and `_execute-prepared`
subcommands for the shell wrapper and detached child. They are tested internal
interfaces, not additional scientist-facing workflows and are not advertised
in the user quickstart.

`status` is a one-shot, read-only inspection. It reports the saved lifecycle,
current stage, completed/total targets, start/update/end times, last warning or
error, result completeness, local PID or Slurm job identity when present, and
the tail location of durable logs. It neither loads spike arrays nor fits,
resumes, kills, or continuously watches anything. PID liveness is advisory:
after an abrupt process or machine failure, saved checkpoints and files are
authoritative, and the user must explicitly run the exact resume command.
Default status reads only small state/manifest files. An explicit
`--verify-results` option may additionally validate the saved NPZ schema
for a completed or returned run; it still never accesses source spikes or
recomputes results.

This is also the Codex operating contract. Codex may start an authorized
detached run, report the path and commands, and end its task. The user can ask
Codex later for one fresh status/log inspection. No recurring wait, background
agent, or token-consuming polling is part of the computation plan.

The separate batch interface remains:

```bash
uv run python -m src.neural_analysis.task_decoding.run_batch dry-run \
  --config-list /path/to/task_decoding_configs.txt

uv run python -m src.neural_analysis.task_decoding.run_batch new \
  --config-list /path/to/task_decoding_configs.txt
```

Without explicit measured resource evidence, `run_batch new` caps execution at
one session at a time even when a larger `--workers` value is requested. To
permit parallel sessions, add
`--resource-run-directory /path/to/completed/measured/run`. That directory,
not an implicitly discovered "latest" run, must contain a complete run with
measured peak RSS. Its analysis version, scoped source, dependency versions,
platform/architecture, regularization mode, and thread limits must match, and
each planned session's trial, regional-unit, time-bin, result-array, and fit
dimensions, including categorical and numerical fit counts separately, must be
no larger than the measured envelope. For an admitted session, use the measured
peak unchanged as its conservative memory estimate;
do not invent a scaling law. Cap workers by requested workers, available CPUs,
and the number whose summed estimates stay within 50% of `MemAvailable`.
Record the evidence run and calculation. An explicitly supplied but invalid
evidence run is an error; omitting it intentionally selects the one-worker
safe path. If `--workers` is omitted, the requested count is the available CPU
count before these caps. One session failure does not cancel independent
sessions already running; after all launched work ends, the batch command
prints every session outcome and returns nonzero if any session failed.

After a matching completed evidence run exists, an explicit parallel launch
is:

```bash
uv run python -m src.neural_analysis.task_decoding.run_batch new \
  --config-list /path/to/task_decoding_configs.txt --workers 4 \
  --resource-run-directory /path/to/completed/measured/run
```

`dry-run` validates metadata, configuration, selected-target columns, feature
parameters, output paths, small `cluster_info.tsv`/channel metadata, and
planned work count without calling `load_sorter_metadata`, loading
`spike_clusters.npy` or the large `spike_utc_unix` members, or fitting models.
It inspects aligned-archive member names. A present `irig_utc_unix` member is
the authoritative coverage source and is read and validated; configured bounds
for that probe are rejected rather than ignored. When the member is absent,
dry run requires the configured trusted bounds. It never substitutes
first/last spike times. `new` creates the run directory and resume record
before large-array loading.
`resume` accepts only the exact run directory and saved configuration; it does
not reconstruct settings from the current command line.

The batch config list is a UTF-8 text file with one configuration path per
nonblank, non-comment line. Batch resume is deliberately per-session through
the printed single-session resume commands; do not add a second batch
checkpoint format. Do not add detached batch supervision during the first
implementation: prove the simpler single-session unattended path on CT026
before deciding whether batch detachment is actually needed.

For a foreground run, the CLI returns nonzero on invalid configuration or
failed scientific execution. A detached `new` or `resume` exit code reports
only whether preparation and child launch succeeded; later scientific success
or failure is read from `status`, durable state, and logs. Likewise, a Slurm
submission exit code reports only preparation and scheduler acceptance. Every
mode prints a short actionable error plus the run directory/resume command when
one exists. Do not require a notebook or Streamlit to start offline
computation.

### 4.6 Routine single-session cluster path

WP10 first measures the bounded fixed-mode workload locally and projects the
full default fixed run. The user selected WP11 as a required implementation
package because even a locally practical single-session runtime compounds over
many sessions. Cluster use does not require proof that local execution is
impossible, and there is no automatic duration threshold. WP11 first proves
one session; WP13 supplies the cross-session concurrency that addresses batch
throughput.

Add one thin, self-submitting Slurm wrapper:

`src/shell_scripts/task_variable_decoding_slurm.sh`

Run these commands on the cluster login node from the tracked-clean repository
root, after the session/configuration has been transferred:

```bash
bash src/shell_scripts/task_variable_decoding_slurm.sh submit-new \
  --config /cluster/session/task_decoding_config.json

bash src/shell_scripts/task_variable_decoding_slurm.sh submit-resume \
  --run-directory /cluster/session/analysis_runs/task_variable_decoding_<timestamp>

bash src/shell_scripts/task_variable_decoding_slurm.sh status \
  --run-directory /cluster/session/analysis_runs/task_variable_decoding_<timestamp>
```

`src/shell_scripts/hpc_ppc.sh` is the site-specific reference, not a
script to modify or call. Reuse its reviewed operational pattern:

- partition `unlimited`, one task, a descriptive job name, five-minute
  `TERM` notice, the existing private Slurm log directory, and existing mail
  settings;
- resolution of `SLURM_SUBMIT_DIR` to the exact Git root, rejection of
  tracked-dirty or wrong-root checkouts, and logging of the actual commit;
- `uv run --frozen --no-sync --offline` so compute jobs never install,
  upgrade, or download dependencies;
- explicit `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `OPENBLAS_NUM_THREADS` limits; and
- `exec` of the Python process so scheduler signals and exit codes propagate.

Login-node setup checks out and records the exact user-approved pushed commit.
The wrapper rejects all tracked changes plus untracked Python files inside the
task-decoding package; every explicit dependency file in the scoped source
list must be tracked. Unrelated untracked files elsewhere in this historically
dirty repository do not block submission. The compute job does not fetch.
Before submission, verify the prepared frozen environment and record the same
runtime package versions captured by `execution.json`; any needed
`uv sync --frozen` happens on the login node before the offline job.

Do not copy PPC's eight CPUs, 32 GB, or 72-hour request without evidence.
Those values serve only as a known high-resource reference. Task decoding has
no initial within-session process pool, so the expected starting CPU request
is one; request more only after a measured and separately approved parallel or
threaded path exists. After WP10, Sol proposes the exact memory and wall-time
values, the user approves them, and the tests freeze all top-of-script Slurm
fields before wrapper implementation.

The self-submitting script has two explicit branches: login-node commands may
prepare/status/submit, while the scheduled private execution branch may only
`exec` `_execute-prepared` and can never call `sbatch`. The layer contains no
scientific defaults or duplicate configuration parsing. `submit-new` runs the
lightweight cluster dry run,
calls the runner's private `_prepare` interface to create one immutable cluster
run directory, submits that exact directory to `_execute-prepared`, and
returns. `submit-resume` submits the same `_execute-prepared` interface for one
existing run directory. Both private modes call the public Python preparation
and execution functions above. Preparation is an implementation detail rather
than a fifth scientist-facing Python workflow.

Immediately before `sbatch`, atomically set lifecycle to
`submission-pending`. On submission failure, set `failed`. On success, write
`slurm_submission.json` atomically with the job ID,
submission time, requested partition/tasks/CPUs/memory/time, code commit,
scheduler log path, and exact status/resume commands. The wrapper prints the
same receipt and returns immediately, but does not write `run_state.json`
after `sbatch`; the possibly fast-starting compute process owns the next state
transition. This prevents a late submitter write from overwriting a `running`
or terminal compute-owned state. The wrapper never selects a latest run,
submits a replacement job, or retries automatically.

Cluster `status` reads small run files and performs at most one
`sacct` query. It reports `State`, `Elapsed`, `TotalCPU`, `AllocCPUS`,
`MaxRSS`, `ReqMem`, `Timelimit`, and `ExitCode` when available, alongside the
pipeline state. It does not write them back into the completed run, guess which
state is authoritative, or poll for a transition. The later benchmark handoff
records this one-shot output with the resource decision.

The Python pipeline handles `SIGTERM`/`SIGINT` by stopping at the
safest available boundary, flushing logs and state, and preserving completed
target checkpoints. A hard timeout or out-of-memory kill may prevent that
handler from running; later status must expose that discrepancy without
declaring completion. Resume remains an explicit separately authorized
submission against the exact cluster run directory.

If benchmarking identifies a need for within-session parallel fitting, stop
and revise this plan. A scheduler provides unattended wall time and memory; it
does not itself justify a second numerical implementation. No `sbatch`
command is run without explicit approval. Once submitted, neither the user nor
Codex remains connected or polls the job.

### 4.7 Manual rsync and portable run contract

The workstation session is the authoritative long-term copy. The cluster holds
an execution copy with the same directory structure below the session root;
absolute workstation and cluster prefixes may differ. Configuration paths are
therefore saved in portable session-relative form. Resolved absolute paths are
execution provenance only and must not enter the scientific fingerprint or be
required to view returned saved results.

Input transfer remains an explicit, simple operation documented with concrete
site paths, following this pattern:

```bash
rsync -a --info=progress2 --exclude='/analysis_runs/' /local/session/ \
  user@cluster:/cluster/session/
```

This example assumes the documented default output root
`<session_root>/analysis_runs`. If a different contained output root is
configured, substitute its validated session-relative path in both the
`--exclude` rule and result-return paths. Dry run prints that exact relative
path. Never run input synchronization without excluding the active output
root.

The workstation copy is authoritative, so updating its corresponding cluster
input files is intentional. Excluding the configured output root is mandatory:
input synchronization must never overwrite cluster run state, checkpoints,
logs, or results. First run the same command with `-n` added when the
destination has not been inspected recently. Do not use broad `--delete`, and
do not submit
while local preprocessing files are changing. After transfer, the cluster
`dry-run` revalidates the configuration, required augmented columns, source
identities, probe files, output path, planned work count, and exact tensor
bytes against 50% of the approved Slurm memory request before `sbatch`. The Git
repository is handled separately: the cluster uses a tracked-clean checkout of
the exact pushed commit rather than an rsynced dirty source tree, and its uv
environment is prepared explicitly on the login node before offline jobs run.

Only a terminally complete run directory is returned. Transfer it first to a
hidden, uniquely named directory on the same local filesystem:

```bash
rsync -a --info=progress2 \
  user@cluster:/cluster/session/analysis_runs/<run_id>/ \
  /local/session/analysis_runs/.incoming-<run_id>/
```

Run the ordinary read-only `status`/result loader against the incoming
copy using `status --verify-results`. It must validate the complete
state and loadable result schema without requiring cluster source paths or
spike arrays. Only then rename the incoming directory atomically to its final
`<run_id>` name; the final target must not already exist. Interrupted
transfers remain hidden and can be safely repeated; they are never presented
by the webapp as completed runs. Input synchronization and result return are
separate commands, and returning results never overwrites preprocessing inputs
or an existing immutable run.

The top-level neural README owns the exact site commands and the distinction
between local and cluster paths. The in-package README owns portability,
fingerprint, termination, and resume semantics. No automatic SSH, rsync,
remote deployment, or general workflow manager belongs in the first version.

### 4.8 Bounded cluster batch-array follow-up

Cluster batch-array support is required WP13 work and remains deliberately the
last implementation job. It starts only after the single-session CT026 result
is accepted and the single-session Slurm resource profile is finalized from
measured cluster usage. It is not part of WP11 or the first scientific cluster
run because array concurrency should reuse a proven single-session execution
unit rather than debug scheduler and scientific behavior simultaneously.

The intended extension is mechanically narrow:

- one Slurm array element per session configuration, using the same
  single-session wrapper and Python runner;
- an explicit `--resource-run-directory` naming the completed measured cluster
  run that owns the approved per-element resource block and admission envelope;
- one immutable mapping/receipt from array index to configuration, run
  directory, job ID, and log;
- one explicit array concurrency cap;
- independent run state, checkpoints, result files, and exact resume commands
  for every session;
- no invocation of the local `run_batch` process inside one large
  allocation; Slurm owns cross-session parallelism;
- no within-session parallelism or shared mutable batch state;
- one-shot aggregate status only; and
- manual resubmission of an exact failed session, never automatic array-wide
  retry.

WP13 begins with mocked array tests and a two-session synthetic smoke test. Its
source implementation no longer needs a new decision about whether arrays are
in scope, but a real multi-session array remains separately authorized. A
later session may reuse the finalized resource block only when it matches the
completed measured cluster run's
analysis/source/environment/platform/mode/threading identity and none of its
trial, regional-unit, time-bin, result-array, or categorical and numerical
fit-count dimensions exceeds that evidence run. The dry run reports those
comparisons. Otherwise stop for a new bounded benchmark and resource decision;
do not extrapolate a scaling law from CT026.

## 5. Configuration contract

`TaskDecodingConfig` contains explicit scientific settings plus one execution
setting. Keep one typed configuration object for a simple user experience,
but classify its fields explicitly so execution-host choices do not alter
scientific identity.

Scientific fields are:

- session metadata path;
- augmented-trial path;
- trial-feature-parameter path;
- PFC and HPC `RegionConfig` records;
- alignment;
- bin width;
- requested PFC/HPC PC counts;
- `target_names`;
- fixed or tuned regularization mode;
- `outer_fold_count` and `inner_fold_count`;
- optional trusted UTC bounds per manually aligned probe.

Freeze the public, case-sensitive `target_names` vocabulary in this canonical
order:

```text
current_state
current_action
current_action_is_correct
previous_action
previous_action_was_rewarded
next_action
current_choice_switch_stay
next_choice_switch_stay
consecutive_omissions
consecutive_rewards
session_trial_index
trial_index_in_block
rewards_in_block
qlearning_relative_value
forgetting_q_relative_value
hmm_signed_belief
hmm_decay_signed_belief
relative_doubt
```

The default is the complete list. A supplied list must be nonempty and
duplicate-free; display-label and source-column aliases are rejected. Normalize
accepted subsets to this canonical order so equivalent subsets have identical
target axes and scientific fingerprints. `outer_fold_count` accepts only
integer `5` (default) or `3`; `inner_fold_count` accepts only integer `3` and
remains inactive in fixed mode. Reject booleans, fractional values, and numeric
strings rather than coercing them.

The configuration's execution-only field is the output root. Batch worker
count belongs only to the batch command's `--workers` option because it spans
sessions rather than describing any one session. It never changes a session's
scientific fingerprint or saved scientific configuration. Resolved absolute
paths, local/detached/Slurm mode, PIDs/job IDs, Slurm resources, log paths, and
plot display selections are also execution provenance rather than scientific
settings.

`scientific_config_payload(config) -> dict[str, object]` is the single owner of
this separation. It returns a JSON-serializable, deterministically ordered
mapping of the scientific fields with portable paths plus the code-owned
analysis version and frozen estimator/PCA controls. Fingerprinting, matching,
tests, and result provenance use this function rather than maintaining parallel
include/exclude lists.

`RegionConfig` should contain:

- canonical display region, exactly `PFC` or `HPC`;
- metadata probe ID;
- channel labels;
- `inside_brain` requirement;
- cluster groups; and
- optional explicit channel IDs, applied as an additional intersection after
  the quality/inside-brain rules and the probe metadata's optional
  `unit_channels` restriction.

Explicit configured channel IDs must be a duplicate-free sequence of
nonnegative JSON integers. Reject booleans, floats (including integer-valued
floats), numeric strings, and negative values rather than relying on the
existing downstream `dtype=int` coercion. Normalize an accepted restriction
into ascending order before serialization and scientific fingerprinting.

Avoid a dictionary of arbitrary settings. Typed fields make the scientific
choices discoverable and testable.

Configurable defaults match revision 5: the complete canonical target list,
choice alignment, 100 ms, 10 PCs per region, fixed regularization,
`outer_fold_count=5`, and inactive `inner_fold_count=3`.
The [-2, 2] s window, seed 0, `1e-8` coefficient tolerance, 15-candidate tuning
grid, and exact LogisticRegression, ElasticNet, and PCA controls are code
constants recorded in the scientific payload; the initial JSON does not expose
extra knobs for them. Changing one requires an analysis-version bump.

Configuration validation checks values and cross-field consistency without
opening large arrays. Input/path validation is a separate pipeline stage.
Augmented-table validation requires the shared identity/baseline columns, the
selected alignment column, and only the source columns needed by the selected
targets. It also requires finite integer-valued `cur_trial` values equal to the
complete zero-based row sequence, nonmissing `cur_block` values, and one
contiguous segment per block label. Missing-sentinel `cur_block` values are
treated as missing. Required numeric columns reject
malformed/non-finite values that remain present under the existing project
missing-sentinel rules and, for `action`, after recognized no-choice labels are
excluded; the selected alignment and every selected numeric target need at
least one finite value. One validation error reports every missing required
column together.

Resolve relative JSON paths against the configuration file's parent directory.
The resolved `neural_session.json` parent is the canonical `session_root`.
Require the metadata file, augmented-trial CSV, feature-parameter JSON, every
required sorter/alignment/channel-quality input for the two configured probes,
and output root to be inside that root; reject required data or output paths
that escape it. LFP files in session metadata are not decoding inputs and do
not gate this analysis. Require the output root to be a proper, dedicated
subdirectory: it must not equal the session root, contain a required input, or
contain the configuration file, or sit inside an explicitly required input
directory such as a sorter directory. Preparation also rejects an output root
inside the Git checkout that owns the executing task-decoding source. The
configuration file itself may live elsewhere, although the documented
workflow places it in the session root.
Save execution-host absolute paths only as provenance. Scientific
configuration and input identities use canonical paths relative to
`session_root`, so copying the same tree under a different workstation or
cluster prefix preserves identity.

Use one small, documented file-identity policy:

- every identity, including `neural_session.json`, records the canonical
  session-relative path, file size, and
  whole-second modification time;
- small structured inputs (JSON, CSV, TSV, and text files no larger than 64
  MiB) additionally record a streaming SHA-256 digest;
- large binary spike/alignment arrays use path, size, and modification time so
  lightweight dry-run does not scan their entire contents; `rsync -a` is
  required to preserve these fields between workstation and cluster; and
- a directory identity is the sorted list of the explicit files the loader
  will open, never a hash of an unspecified directory tree.

This policy protects against ordinary replacement or preprocessing changes; it
is not intended as an adversarial integrity system. Save the identity method
and values in `input_manifest.json` and use exactly the same records locally
and on the cluster.

`build_input_manifest(...) -> dict[str, object]` is the single owner of this
policy and accepts the explicit resolved files plus their portable paths; the
CLI, fingerprinting code, cluster preflight, and tests must not recreate file
identity rules independently.

## 6. Data flow and array contracts

```
neural_session.json + explicit augmented table/config
        |
        +--> validated target table and target-specific masks
        |
        +--> explicit PFC/HPC probe and unit selection
        |
        +--> trusted alignment coverage intersection
        |
        +--> PFC and HPC rate tensors
             shape: (trial, time_bin, unit), units: Hz
        |
        +--> target/fold-local training transforms
             pooled unit z-score
             optional separate regional PCA
        |
        +--> one elastic-net decoder per target/time/region/representation/fold
        |
        +--> fold results, coefficients, failures, timings, provenance
        |
        +--> versioned NPZ + config + log + summary + light-mode PNGs
        |
        +--> read-only existing-webapp view
```

### 6.1 Trial table

`build_target_table(...)` returns a dataframe with one row per
original trial and:

- a zero-based `row_position` in chronological table order, independent of the
  caller's pandas index labels;
- `trial_id` copied from `cur_trial`;
- `block_id` copied from `cur_block`;
- one canonical numeric-encoding column per target; numerical targets retain
  their stored native values and units without normalization or
  standardization;
- target-valid boolean columns; and
- baseline validity/reason fields.

Shifted targets are built before baseline filtering. The function must not
modify its input dataframe. Rate-tensor calls use a RangeIndex working view and
`row_position`; they never pass arbitrary source index labels to the existing
`.loc`-based tensor builder.

### 6.2 Rate tensors

PFC and HPC tensors use:

- axis 0: the same ordered common neural-eligible trial subset for both
  regions, accompanied by `trial_row_indices` mapping each tensor row to the
  zero-based target-table `row_position` and stable `cur_trial` identity;
- axis 1: common event-relative time-bin centers;
- axis 2: region-specific stable units; and
- values: float firing rates in Hz.

Do not build separate rate tensors for targets. Build each regional tensor once
per alignment/bin-width run over rows passing the target-independent baseline:
valid animal choice, no manual reward, present block/alignment, and a complete
window on every configured probe. Then project each target's additional
eligibility mask through `trial_row_indices`. Rows excluded from the tensor are
never represented as zero-rate observations. Preserve full-original-row
eligibility and outer-fold arrays separately in results so trial matching
remains inspectable.

If channel/unit selection yields zero units for PFC or HPC, input validation
fails before tensor allocation. Do not treat a missing configured population
as a fold-level unavailable feature set or let a combined model silently
collapse to the surviving region.

The initial implementation may hold both regional tensors in memory. Dry run
reports their exact float64 allocation as
`n_tensor_trials * n_time_bins * (n_pfc_units + n_hpc_units) * 8`, and reports
source file sizes separately as I/O/provenance facts. Compressed or on-disk
file sizes are not RAM estimates. The only pre-benchmark hard stop is when the
exact tensor allocation alone exceeds 50% of the applicable memory budget, or
that budget cannot be determined. Locally the budget is Linux `MemAvailable`.
In a scheduled Slurm process it is the lower of `MemAvailable` and the parsed
scheduler/cgroup allocation limit; login-side submission also uses the
approved `--mem` request. These sources are injectable in tests. Otherwise do
not claim that dry run predicts peak RSS. The bounded CT026 benchmark's
measured peak RSS is authoritative for later local concurrency and Slurm
sizing; do not add disk-backed arrays or streaming without measured need and a
revised plan.

### 6.3 Feature preprocessing

Within one target and outer fold:

- determine constant/unavailable units from the outer training observations;
- pool training trials and time bins for regional mean/scale;
- after removing those unit columns, set each region's PCA component limit to
  `min(n_usable_units, n_training_trials * n_time_bins)` and its effective
  count to the smaller of that limit and the requested count; do not call
  `matrix_rank` or introduce a singular-value tolerance;
- treat exactly one usable unit as a valid one-component PCA input; only zero
  usable units is the unavailable feature-set case;
- reuse that transform across all time-bin decoders in the fold;
- fit one PCA per selected region and fold when PCA is requested;
- reuse PFC/HPC transforms for standalone and combined results; and
- apply unchanged transforms to the outer test rows.

Tuned mode repeats these steps within each inner training split. Do not reuse
outer-training transforms inside inner validation. Within one target and outer
fold, fit each outer regional transform once and reuse it across all time bins,
representations, and standalone/combined region results. Within each inner
fold, fit each regional transform once on that inner-training subset and reuse
it across every time bin and all 15 candidates. Candidate selection is
time-bin/region/representation-specific, but transform fitting is not.

### 6.4 Fits and scores

One fit record is identified by:

```
target
alignment
bin_width
time_bin
region_configuration
representation
outer_fold
regularization_mode
```

It records status/reason, train/test counts, class counts when applicable,
requested/effective feature counts, estimator parameters, convergence status,
metrics, intercept, and coefficients with stable feature identities.

Classification balanced accuracy and AUC come from one set of held-out
predictions. A metric toggle never creates another fit.

## 7. Split and model implementation details

### 7.1 Outer splits

Build one split assignment per target from its matched eligible rows:

- categorical: `StratifiedGroupKFold`;
- numerical: `GroupKFold`;
- groups: `cur_block`;
- deterministic, no shuffle; and
- validate all categorical train/test class sets after splitting.

Persist an array shaped `(n_targets, n_original_trials)` containing
outer-fold IDs and `-1` for ineligible rows. This makes matching and
reproducibility directly inspectable.

### 7.2 Fixed mode

For each target/fold:

1. fit training-only regional transform(s);
2. transform train/test rate tensors;
3. fit each time-bin decoder at the fixed settings;
4. catch convergence warnings as invalid fits;
5. compute held-out metric(s); and
6. retain coefficients and fold diagnostics.

### 7.3 Tuned mode

For each target and outer fold:

1. create three grouped inner folds from only the outer-training rows, using
   deterministic `StratifiedGroupKFold(shuffle=False)` for categorical targets
   and deterministic `GroupKFold(shuffle=False)` for numerical targets;
2. initialize compact candidate score/status accumulators for every
   time-bin/region/representation cell;
3. for each inner fold, fit the PFC/HPC training transforms once, evaluate all
   cells and 15 candidates from those transformed train/validation tensors,
   append only scores/status, and release the transformed tensors before the
   next inner fold;
4. invalidate a candidate if any required inner fit/metric is invalid, then
   select the highest mean inner balanced accuracy or $R^2$ for each cell,
   resolving exact ties by declared candidate order; and
5. fit the PFC/HPC transforms once on all outer-training rows, reuse them for
   every outer cell, refit each selected estimator, and evaluate once on the
   outer-test rows.

Keep tuned mode mechanically explicit rather than hiding it inside a generic
search object: the pipeline must fit PCA inside each inner training split and
retain failure reasons and selected settings. Validate that blocks never cross
inner train/validation partitions and that both partitions contain both classes
for categorical targets. If a valid three-fold inner split cannot be
constructed, all tuned cells for that target/outer-fold are unavailable rather
than falling back to trial-wise splitting or a different fold count.

### 7.4 Complete-fold aggregation

Aggregate after all individual folds are stored. A summary cell is available
only if every requested fold is valid and finite. Unavailable cells preserve
surviving fold records but have no primary mean.

Only declared scientific invalidities become unavailable cells and allow
unrelated work to continue: invalid grouped folds/class coverage, a constant
target, no usable fold-training features after constant-feature removal, a
convergence warning, or a non-finite fit/score. Missing required inputs or zero
session-level units in a configured region are preflight errors, not scientific
missingness. If one component region has no usable fold-training features, its
standalone and combined cells are unavailable; the combined cell never falls
back to the surviving region.
Unexpected programming, schema, or I/O exceptions fail the run, flush state
and logs, preserve already-published target checkpoints, and return nonzero.
They are never caught by a broad exception handler and relabeled as missing
scientific results.

## 8. Saved-run layout and schema

Each execution creates an immutable session-local directory:

```
<output_root>/  # documented default: <session_root>/analysis_runs
    task_variable_decoding_<YYYY-MM-DDTHH-MM-SSZ>/
        config.json
        trial_feature_params.json  # exact copied input bytes
        input_manifest.json
        run_state.json
        resume_command.txt
        status_command.txt
        execution.json
        local_launch.json      # latest detached-local launch receipt only
        execution_guard.json  # present only while an execution owns the run
        results.npz
        run.log
        console.log             # detached local runs only
        summary.md
        run_session.py
        run_batch.py
        slurm_submission.json   # only when submitted through WP11
        checkpoints/
        figures/
```

Create the run directory with an exclusive `mkdir`. If two preparations share
the same second, append the first available zero-padded numeric suffix; never
reuse or clear an existing path.

The script files are snapshots of the two launch modules used for the run.
Human-readable plots and report material remain in the run directory. No output
is written into the Git repository.

### 8.1 Identity and rerun behavior

A stable run fingerprint is computed from:

- analysis version and a scoped scientific-source fingerprint;
- normalized scientific configuration with portable relative paths, not
  workstation or cluster root prefixes;
- session ID;
- `neural_session.json`, augmented-table, and feature-parameter identities;
- portable alignment/sorter/quality identities using the Section 5 policy.

The normalized scientific configuration contains only the scientific fields
listed in Section 5. Output roots, batch worker requests, resolved absolute path
prefixes, foreground/detached/Slurm mode, scheduler resources, logs, and display
choices are excluded. `scientific_source_fingerprint(...)` hashes
repository-relative paths and contents for every `.py` file in
`src/neural_analysis/task_decoding/`, the
direct runtime dependencies
`src/__init__.py`, `src/neural_analysis/__init__.py`,
`src/neural_analysis/session_metadata.py`,
`src/neural_analysis/spike_behavior/__init__.py`,
`src/neural_analysis/spike_behavior/loading.py`,
`src/neural_analysis/population/__init__.py`, and
`src/neural_analysis/population/pca.py`, plus
`src/behavior_analysis/__init__.py`,
`src/behavior_analysis/project_utils.py`, `pyproject.toml`, and `uv.lock`.
The initial list is explicit and reviewed whenever a new direct dependency is
introduced. A local persistent `new`/`resume` requires only these relevant
tracked files to be clean, rejects untracked Python inside the task-decoding
package, and requires each explicit outside dependency file to be tracked;
unrelated documentation or source changes do not block it. The full Git HEAD
and dirty summary are recorded as execution provenance, not scientific identity.
`dry-run` may report relevant source dirtiness but performs no run preparation.
The cluster wrapper retains the stronger whole-checkout tracked-clean,
exact-pushed-commit gate for operational reproducibility.

Resume compares the saved scoped source fingerprint, analysis version,
scientific configuration, and input manifest. It does not require the same Git
HEAD when unrelated repository files changed. Any scientific-code change must
also bump `ANALYSIS_VERSION`; tests enforce that the version and scoped source
identity are both saved, while review/commit discipline enforces the bump.

The default single-session runner deterministically lists every directory with
a matching fingerprint. A completed match requires both lifecycle `complete`
and successful directory-level result validation. If any valid completed match
exists, the runner skips and reports it. If none does but one or more matches
are nonterminal, interrupted, failed, or marked complete with invalid/missing
artifacts, it refuses `new` and prints every state plus the applicable
status/resume or explicit-rerun command. It never chooses an implicit "latest"
run or mutates a corrupt one. `--rerun` explicitly creates a new timestamped
directory without overwriting an old run. A changed version, scoped source,
input, or scientific configuration creates a different fingerprint.

Use small JSON manifests and ordinary NPZ files. No database, lock service, or
content-addressed object store is needed.

### 8.2 Launch, interruption, and resume state

`new` writes the run directory, immutable saved configuration, an exact
byte-for-byte copy of `trial_feature_params.json`, input/code identity, initial
`run_state.json`, and exact
`resume_command.txt` and `status_command.txt` before loading large
spike arrays. `execution.json` records foreground/detached/Slurm mode
and the local PID or scheduler identity as advisory execution metadata. It also
records host, OS/platform, machine architecture, available CPU count, explicit
thread limits, and Python, NumPy, SciPy, pandas, scikit-learn, Pynapple, and
Matplotlib versions. The scientific fingerprint's `uv.lock` identity remains
the environment contract. Batch-launched runs also record the
requested/effective worker count and, when parallel execution is admitted, the
exact measured evidence-run directory, fingerprint, envelope, peak RSS, and
memory calculation. Run state
uses the small lifecycle `initialized`,
`submission-pending`, `preflight_complete`, `running`, `interrupted`, `failed`,
and `complete`.

Only one process may execute a run directory. Before entering the foreground
pipeline, atomically create a small `execution_guard.json` containing mode,
host, PID, process-start token, start time, and Slurm job ID when applicable.
`resume` refuses when the recorded same-host process identity is alive or the
recorded Slurm job is pending or running. A dead/reused local process identity
or terminal scheduler state is a stale guard that may be replaced only after
this bounded liveness check is recorded. An unresolvable foreign-host guard
stops with an actionable message rather than guessing. Remove the guard on
orderly exit; retain enough execution history in `execution.json` for
diagnosis. This is a single atomic file, not a lock service or heartbeat
system.

During detached launch, the live child identity in `local_launch.json` owns the
run even before the child finishes creating `execution_guard.json`. A child
that dies during that handoff leaves an `initialized` run that is resumable
only after the saved process identity is verified dead. The receipt is not a
second lock and never changes lifecycle state; it prevents parent/child file
writer races while reusing the same bounded liveness check.

The state file includes started, last-updated, and completed timestamps;
current stage; completed and total target names; the last error/warning; and
whether final results were published. Update it at stage transitions and
completed-target boundaries. Do not add a heartbeat daemon or a monitoring
database. A long target can legitimately leave the timestamp unchanged, so
`status` reports facts rather than declaring a run stale from elapsed time
alone.

Update the state file atomically at package-defined boundaries and flush the
log before returning a failure/interruption code. A resumed run trusts only its
saved configuration/fingerprint and valid target checkpoints. It must not use
new command-line scientific overrides or infer the latest run directory.
Before claiming the execution guard, every prepared execution or resume
rebuilds the lightweight input manifest and verifies it against the saved
manifest, then verifies the scoped scientific-source fingerprint and analysis
version and the saved Python/numerical-library version mapping. Any mismatch
fails before large-array loading and requires a new run; queued or detached
work must not silently consume changed inputs or environments after
preparation.

The summary and live handoff record distinguish an interrupted resumable run
from a completed result. A missing final `results.npz` is never
presented as complete. `console.log` captures detached-local standard streams;
Slurm standard streams remain in the recorded site scheduler log.
`run.log` is the pipeline log and is flushed at stage/target boundaries so
later inspection does not depend on the original terminal or Codex task.

### 8.3 Checkpoint boundary

The restart boundary is one completed target. Write a target checkpoint only
after all requested region/representation/time/fold cells for that target have
finished or been recorded unavailable. Resume only checkpoints whose complete
fingerprint matches.

This limits lost work without adding per-fit transaction machinery. The final
`results.npz` is written only after combining target checkpoints. Write every
checkpoint, final NPZ, summary, and required default PNG to a uniquely named
temporary file in its destination directory, flush/close it, and publish it
with `os.replace`. Set `final_results_published=true` and lifecycle
`complete` only after `results.npz`, `summary.md`, and the required default
heatmaps for every target family present have all been published and the
directory-level loader has validated the saved configuration, manifests, exact
feature-parameter copy, NPZ, and required figures.
Categorical-only or numerical-only runs do not create or require an empty
other-family figure. Optional coefficient figures are not part of the
completion gate. A failure during final reporting remains resumable from target
checkpoints and must never expose a false complete run.

### 8.4 NPZ arrays

The exact schema is frozen by round-trip tests before implementation. It should
use named arrays and include at least:

- target, region, representation, metric, time, fold, and feature labels;
- time-bin edges/centers in seconds;
- dense fold score arrays with NaN for non-applicable/invalid metrics;
- fit status and compact reason-code arrays;
- requested/effective feature counts;
- train/test and class counts;
- full-table row positions, trial IDs, block IDs, the ordered
  `trial_row_indices` mapping from common neural-tensor rows to full-table
  positions, encoded target values, eligibility masks/reasons, and outer-fold
  assignments;
- for tuned runs, inner-fold assignments plus compact candidate-by-inner-fold
  score/status arrays and the selected candidate index for each outer cell;
- direct-unit and PC coefficients plus fitted intercepts, grouped by region
  configuration;
- selected fixed/tuned parameters;
- stable unit/feature identities;
- stage timings; and
- a dictionary named `meta` containing version, seed, units, axes,
  paths, parameters, provenance, and warning text.

The directory-level loader validates required supporting files and verifies
the copied feature-parameter file against its small-file SHA-256 input-manifest
identity. It then validates the NPZ `meta` schema/version before exposing
result arrays. The NPZ is a trusted local analysis artifact, matching the
project's existing metadata convention.

`summary.md` records the analysis goal, analysis name, UTC date/time, included
session, exact launch mode/command and scripts actually run, the snapshotted
main and batch scripts, scientific configuration, eligibility/fold coverage,
warnings and unavailable results, stage and total timings, and a concise
scientific description of the saved outputs. It does not claim a scientific
conclusion that the saved metrics do not support.

If dense coefficient arrays would waste unreasonable space after real
preflight, use parallel long-form primitive arrays inside the same NPZ. Do not
switch to a new storage dependency.

## 9. Plotting and webapp plan

### 9.1 Offline figures

Generate from saved results when that target family is present:

1. categorical heatmap for balanced accuracy;
2. categorical heatmap for ROC AUC;
3. numerical $R^2$ heatmap; and
4. optional unit-coefficient figure(s) generated on demand from a selected
   saved direct-unit result.

Plots use opaque white backgrounds, black text/axes, readable fonts, complete
labels, and captions. Unavailable cells use a mask/color distinct from chance
or zero. Each default metric figure uses a fixed 3-region by
2-representation panel layout, with target rows and time-bin columns, so it
shows all six computed combinations without an arbitrary default selection.

The pipeline does not generate every combination as a PNG by default. Saved
arrays support interactive inspection; default report figures should remain a
small, declared set. On-demand coefficient display/exports are presentation
choices, not scientific configuration fields or run-completion requirements.

### 9.2 Integrated read-only view

The new existing-webapp view provides:

- a session-relative completed-results-root locator, defaulting to
  `analysis_runs`, that is validated inside the selected metadata session;
- saved-run selector, completion/fingerprint/config summary;
- region and representation selectors;
- alignment/bin-width information from the run, not recomputation controls;
- categorical target heatmap with metric toggle;
- numerical heatmap;
- fold score/count/failure table for a selected cell;
- coefficient plot/table for a selected direct-unit cell; and
- links or displayed paths to run summary and PNGs.

It does not expose a Compute button. If no compatible completed run exists, it
shows the expected session-local location and CLI invocation pattern.

## Implementation orchestration - Sol supervisor and Terra workers

These assignments organize future user-approved implementation. They do not
authorize source edits, tests, CT026 computation, or Slurm actions by
themselves.

### Roles

**Lead Sol supervisor - `gpt-5.6-sol`, high reasoning**

- owns authorization, scientific contracts, package order, shared files,
  commits, and the live handoff;
- gives each worker an exact allowlist, tests, commands, and stop gate; and
- independently reproduces RED/GREEN and marks gates complete.

**Terra package worker - `gpt-5.6-terra`, high by default and xhigh
where assigned**

- receives one bounded tests-only or implementation-only task and edits only
  its allowlist;
- stops after genuine RED, then implements only after Sol verifies and commits
  the tests; and
- makes no commits, real-data/Slurm actions, or nested assignments and returns
  exact files, commands/results, and unresolved risks.

**Independent Sol gate reviewer - `gpt-5.6-sol`, high or xhigh as
assigned**

- reviews a stable tests-only design or stable GREEN diff read-only;
- checks scientific drift, leakage, array axes/units, validity policy,
  checkpoint identity, failure handling, and missing tests;
- does not edit, commit, or broaden scope; and
- is mandatory for WP4, WP5, WP9, WP10 interpretation, WP11, and WP13.

Recheck model and effort availability at implementation start. Do not silently
substitute another model or effort. If the named configuration is unavailable,
stop and ask the user to revise the agent plan.

### Shared-worktree and concurrency rules

- Use at most one write-enabled Terra worker at a time.
- Sol may run independent read-only review concurrently only while the diff is
  stable.
- No two agents edit the same source, test, README, fixture, package
  initializer, webapp router, or plan document concurrently.
- Sol records HEAD and `git status --short` before and after every
  assignment.
- Unexpected or unowned changes stop the package immediately.
- The worker's allowlist must name exact files or a narrow responsibility
  group. Needing another file is a stop-and-replan condition.
- The Sol supervisor performs commits only while the Terra writer is idle.
- Subagents do not spawn subagents. The supervisor alone assigns, follows up,
  interrupts, or replaces workers.

### Mandatory package sequence

For each implementation package:

1. Sol re-reads the package's existing files, callers, tests, plan scope, and
   current worktree.
2. Sol sends a self-contained tests-only assignment to the Terra worker with
   model/effort, file allowlist, required tests, expected RED, forbidden
   actions, and return format.
3. Terra writes tests, runs the focused command, reports RED, and stops.
4. Sol inspects the test diff, reproduces RED, obtains the assigned independent
   test-design review when required, and commits tests only.
5. Sol follows up with the same Terra worker to authorize the bounded
   implementation.
6. Terra implements the smallest in-scope change, runs focused tests to GREEN,
   reports, and stops without committing.
7. Sol reviews the diff and runs focused plus affected regression suites. The
   independent Sol reviewer performs the assigned stable-diff gate.
8. Accepted findings return to the same Terra worker. Sol reruns verification,
   commits implementation separately, and updates the live handoff/package
   record.

Documentation-only updates are separate commits from tests and implementation.
The README/example package may characterize and test already-stable interfaces,
but must not silently redesign them.

Every Terra assignment repeats its package ID/model/effort, objective, exact
allowlist, contract, commands, expected RED/GREEN stop, prohibited actions,
current HEAD/ownership, and required return evidence. An interrupted report is
not a gate; Sol re-audits the diff and reruns the last command before resuming.
The detailed Sol/Terra breakdown and independent-review gates live once, in the
WP0-WP13 descriptions below.

## 10. Tests to write before implementation

### 10.1 Behavioral feature tests

Add to `src/tests/behavior_analysis/test_session_analysis.py`:

1. `rewards_in_block` is recorded before the current trial update.
2. Counts reset at every block change, including when the first new-block row
   is manual/no-choice.
3. Positive numeric rewards increment; zero rewards do not.
4. Numeric-string actions/rewards are handled consistently.
5. Manual rewards and no-choice rows within an unchanged block neither
   increment nor reset.
6. Fractional/non-finite actions, non-finite/malformed rewards on valid animal
   choices, and missing block identities raise clear errors; strengthening the
   shared parser retains its existing valid-value behavior.
7. On a normalized fixture, `make_augmented_trial_df` preserves row count,
   order, index, and prior output columns/values while adding the new column;
   legacy experimenter-reward alias behavior remains unchanged.

Add a focused test in
`src/tests/behavior_analysis/test_gather_trial_features.py`: the file backfill
uses the same helper, requires the four canonical source columns, reloads and
validates its temporary serialization before publication, atomically adds only
`rewards_in_block`, preserves row order and all prior loaded values/columns,
and leaves the original untouched if a source is missing or the round-trip
check fails. Successful and injected serialization/validation failures leave
no sibling temporary file. It leaves the feature-parameter file and unrelated
artifacts untouched; an existing destination column is a clear no-write error.
Inject a temporary-file removal failure separately and assert that it is
raised with the temporary path visible. If cleanup follows an earlier
serialization or validation failure, preserve that original failure as
exception context; do not claim the temporary artifact was removed when the
operating system refused its removal.

### 10.2 Config and input tests

Create `src/tests/neural_analysis/task_decoding/test_config.py` and
`test_targets.py`:

1. Defaults exactly match revision 5, including the ordered 18-identifier
   target vocabulary and `outer_fold_count=5`, `inner_fold_count=3`.
2. Region mappings require distinct configured probes and recognized regions.
   Explicit channel restrictions accept only duplicate-free nonnegative JSON
   integers and normalize them into ascending order; booleans, floats
   (including integer-valued floats), numeric strings, negative values, and
   duplicates fail before the downstream cluster filter can coerce them.
3. Every canonical target identifier maps to its frozen family, display label,
   source/derivation, and order; equivalent subsets normalize to canonical
   order. Empty, duplicate, wrong-case, display-label, source-column, or unknown
   names fail. Outer counts other than integer 5 or 3 and inner counts other
   than integer 3 fail, including booleans, fractional values, and numeric
   strings. Invalid alignment, bin size, PC counts, bounds, and frozen
   window/seed/tolerance/tuning-grid overrides also fail clearly.
4. Missing augmented columns are reported together; a selected subset requires
   shared baseline columns, its selected alignment, and only its target
   sources. Choice-aligned runs do not require `start_time`.
5. Empty, nonnumeric, non-finite, fractional, non-zero-based, duplicated, or
   gapped `cur_trial` sequences; missing `cur_block` values; noncontiguous reuse
   of a block label; and malformed/non-finite present numeric values fail,
   while established project missing sentinels remain row-level missingness.
6. Feature-parameter JSON must be an object.
7. HMM display labels describe signed belief, while source columns remain
   unchanged.
8. Present `action`/`correct` labels other than exact numeric 0/1 and
   `state_int` labels outside 0/1/known dark-state 2 fail rather than becoming
   extra classes; recognized missing sentinels remain row-ineligible.
9. Paths resolve from the config parent to the canonical metadata-parent
   session root; required inputs and output root outside it are rejected, while
   moving the whole tree preserves portable identities. Output root equal to
   the session root or overlapping a required input path/directory or the
   configuration file is rejected, as is an output root inside the executing
   source-code Git checkout.
10. Scientific fingerprint payload includes every scientific field but excludes
   output root, batch worker requests, execution mode, scheduler resources,
   resolved root prefix, and display choices.
11. The frozen window, seed, coefficient tolerance, tuning grid,
    LogisticRegression, ElasticNet, and PCA controls match revision 5, are
    recorded, and are not extra initial JSON knobs.

### 10.3 Target and eligibility tests

1. Previous/next targets are constructed before filtering.
2. Manual/no-choice adjacent rows are not bridged.
3. Switch/stay derivations match explicit action sequences.
4. Dark state 2 is excluded only from binary current-state eligibility.
5. Target-specific missingness does not leak into unrelated masks.
6. Baseline manual/no-choice/current-alignment exclusions are correct.
7. Only exact numeric 0/1 actions are valid choices; unrecognized action and
   categorical labels never become classes through truncation or truthiness.
8. PFC/HPC/combined representations receive identical target trial rows.
9. Class labels and positive-class mappings are persisted.
10. Numerical target values equal their source values exactly after numeric
   coercion; no normalization or standardization is applied.

### 10.4 Activity and unit-selection tests

Create `test_activity.py`:

1. Explicit PFC/HPC probe mapping is used instead of probe-name inference.
2. Channel selection combines channel-quality label, inside-brain status,
   optional metadata `unit_channels`, and optional additional config channels
   by intersection, and persists the rules and selected IDs.
3. Cluster selection passes configured groups to the existing
   `filter_cluster_metadata`, retains its cluster/channel/normalized-quality
   columns and stable sorted order, and leaves `select_units_by_channels`
   unchanged for existing callers.
4. Stable unit IDs include probe IDs and preserve axis order.
5. Spike and cluster length mismatch fails.
6. Duplicate selected cluster IDs, selected IDs absent from spike-cluster
   assignments, and non-finite selected spike timestamps fail as input errors.
7. A present `irig_utc_unix` member must be one-dimensional, nonempty, and
   finite and supplies standard alignment coverage; configured bounds for that
   probe are rejected rather than ignored or allowed to mask a malformed
   member.
8. Absence of `irig_utc_unix` requires finite ordered trusted bounds, matching
   the current manual-alignment archive; first/last spikes are never used.
9. Dry validation rejects an aligned archive missing `spike_utc_unix` without
   loading that large member.
10. Full event window, not only the alignment timestamp, must be covered on both
   probes; failure on either removes the row from every region configuration.
11. Both regional tensors share trial/time axes and retain Hz units.
12. Existing Pynapple and NumPy reference binning agree on a small fixture.
13. Manual/no-choice/missing-alignment/uncovered original rows are absent from
    tensors rather than represented as zeros, and `trial_row_indices` maps
    every tensor row back to zero-based table position and `cur_trial` identity
    even when the input dataframe has non-default index labels.
14. Projecting different target masks through `trial_row_indices` preserves
    matched PFC/HPC/combined trial order.
15. Dry run reports exact tensor allocation bytes and source file sizes
    separately; it may read `irig_utc_unix` coverage but not
    `spike_utc_unix`, and only tensor bytes participate in the pre-benchmark
    50%-of-budget guard. Local, Slurm allocation/cgroup, and approved
    login-side memory sources choose the documented applicable minimum.
16. Zero selected channels/units in either configured region is a preflight
    error; combined decoding never collapses to one surviving region.

### 10.5 Split, preprocessing, and model tests

Create `test_modeling.py`:

1. Blocks never cross train/test folds.
2. Categorical folds contain both classes or return a declared unavailable
   result.
3. Numerical folds do not discretize targets.
4. Outer assignments are reused across time/region/representation and between
   fixed/tuned runs with otherwise identical scientific inputs.
5. Scaling statistics come only from training rows.
6. PCA directions and the dimensional component limit come only from training
   rows; the limit is exactly
   `min(n_usable_units, n_training_trials * n_time_bins)` with no
   `matrix_rank` or singular-value tolerance.
7. PFC + HPC PC features are concatenated separate regional projections.
8. Direct combined features preserve regional/stable-unit order.
9. Requested PC count is capped visibly and deterministically by that exact
   component-limit formula. A fold with exactly one usable unit remains valid,
   returns one component, and does not call the legacy
   `fit_population_pca` helper that requires at least two units; zero usable
   units remains unavailable.
10. Fixed mode performs outer evaluation without inner fits.
11. Tuned mode never passes outer-test rows into inner splitting,
    preprocessing, or selection.
12. Inner folds use the declared grouped splitter, never split a block, and
    validate categorical train/validation class coverage; one assignment is
    reused across all candidates, time bins, regions, and representations for
    that target/outer fold.
13. Failure to construct a valid three-fold inner split makes the tuned cell
    unavailable without changing fold count or falling back to trial-wise
    splitting.
14. Instrumented transform fit counts are invariant to time-bin and candidate
    count: one PFC/HPC transform per outer fold and one per inner fold, reused
    across representations and standalone/combined results.
15. Candidate tie resolution follows declared order.
16. Balanced accuracy uses threshold 0.5 and AUC uses the same fit's scores
    oriented to the saved positive class.
17. $R^2$ uses native targets and returns unavailable for constant/singleton
    test targets.
18. Negative $R^2$ and below-chance categorical scores remain valid.
19. A convergence warning invalidates the fit; a converged all-zero solution
    remains valid.
20. If one region has no usable fold-training features, its standalone and
    combined cells are unavailable while the other standalone region may
    proceed.
21. Coefficient signs, scales, selection frequencies, and `1e-8` threshold are
    correct; fitted intercepts round-trip but are excluded from feature counts.
22. Primary means require all requested valid outer folds.
23. Results are deterministic for the fixed seed.

Include a direct installed-API test for the supported scikit-learn 1.8
logistic configuration so implementation does not depend on the deprecated
`penalty` argument.

### 10.6 Result I/O and restart tests

Create `test_results.py` and `test_pipeline.py`:

1. NPZ round trip preserves arrays, full-table trial/block/target identity,
   the common neural-tensor row mapping, eligibility reasons, labels, axes,
   units, and metadata.
2. The named `meta` dictionary preserves required provenance.
3. The run directory preserves an exact byte-for-byte copy of the manifested
   `trial_feature_params.json`, including after portable result transfer; a
   missing or altered copy fails directory-level result validation.
4. Schema/version mismatch fails clearly.
5. Tuned-result round trip preserves inner assignments, every candidate's
   fold scores/status, deterministic order, and selected index; fixed runs mark
   these arrays not applicable without fabricating inner results.
6. The input manifest includes `neural_session.json`; small structured inputs
   use SHA-256 identities and large binaries use portable path/size/mtime
   without reading their contents.
7. Scoped scientific-source identity changes when a relevant package/direct
   dependency (including `project_utils.py`) or environment lock changes, but
   not for an unrelated document; relevant dirty/untracked source blocks
   persistent local preparation.
8. Fingerprints change with scientific input/config/analysis-version/scoped
   source changes, remain stable across execution-only/unrelated-repository
   changes and root-prefix moves, and use the documented input identities.
9. An identical run causes a default skip only when lifecycle and
   directory-level result validation both pass; otherwise every incomplete,
   failed, or falsely-complete/corrupt match is listed with exact recovery or
   explicit-rerun guidance, and no implicit latest run is chosen or mutated.
10. Explicit rerun creates a new path without overwrite; same-second run-ID
   collisions are resolved by exclusive creation and a deterministic suffix.
11. `new` writes saved config, run state, and exact resume/status
   commands before an injected large-array/long-running stage.
12. Matching target checkpoints resume; mismatched checkpoints or a different
   scoped source fingerprint/analysis version are rejected.
13. Interrupted/failed foreground states return nonzero, preserve
   logs/checkpoints, and do not publish a complete result; detached/submission
   exit codes report launch acceptance rather than later scientific outcome.
14. Each declared scientific invalidity is recorded unavailable without
   aborting unrelated cells, time bins, or targets; an injected unexpected
   programming, schema, or I/O exception fails the run, preserves completed
   checkpoints, returns nonzero, and is not masked as unavailable.
15. Log and summary contain the analysis goal/name/date, session, snapshotted
   scripts, parameters, eligibility/folds, warnings, scientific output
   description, and timing information.
16. Dry-run loads metadata/small tables and directly inspects small cluster and
   channel metadata without calling `load_sorter_metadata`, loading
   `spike_clusters.npy`/`spike_utc_unix`, or fitting models; ordinary IRIG
   coverage may read only `irig_utc_unix`.
17. Changing any manifested input after preparation makes foreground,
   detached, and mocked Slurm execution fail before large-array loading;
   changing an injected runtime package version does the same, while unchanged
   inputs/environments resume normally.
18. Detached `new` and `resume` return after bounded setup, redirect
   child output, atomically save the child PID/start token in the separate
   launch receipt before returning, prevent immediate-resume and
   parent/child-writer races, and run the same foreground pipeline without a
   shell-dependent scientific path.
19. One-shot `status` reads only small saved files, reports progress and
   result completeness, and never loads spikes, fits, resumes, or polls.
20. A simulated abrupt child death leaves checkpoints resumable and is
   reported without automatic restart or a false `complete` state.
21. A Linux integration smoke test confirms that a detached synthetic child
   survives the launcher process and completes with no open terminal. Before
   relying on Codex as the launcher, repeat that smoke through the actual
   Codex command environment because a host may clean up child processes.
22. `status --verify-results` additionally validates a completed NPZ
   through the saved-result loader without opening source data or mutating the
   run.
23. Foreground, detached, and mocked Slurm paths use the same prepare/execute
   functions and the private CLI bridge does not re-parse scientific settings.
24. A second execution refuses a matching live same-host process identity or
   pending/running Slurm job; a dead or PID-reused stale guard can be replaced
   without a lock service.
25. Checkpoints, NPZ, summary, applicable-family PNGs, and state are atomically
   published, and `complete` is written last only after required artifacts
   validate.
26. Session and batch entrypoints set and record one-thread OpenMP/MKL/OpenBLAS
   values before lazily importing numerical modules; local benchmark and
   mocked Slurm execution identities match.
27. Execution provenance records the host/platform/architecture/CPU/thread
   identity and declared runtime package versions.

### 10.7 Batch-runner tests

Create `test_run_batch.py`:

1. The config list ignores blank/comment lines and preserves declared session
   order.
2. Dry-run reports every session, planned fits, exact tensor bytes, and selected
   worker count without loading `spike_utc_unix` or fitting; it may read the
   small IRIG coverage member.
3. Without `--resource-run-directory`, batch execution is capped at one
   worker. Explicit evidence must be a complete measured run with matching
   analysis/source/environment/platform/mode/threading and an envelope at
   least as large as every admitted trial/unit/bin/result/fit dimension. An
   invalid explicit directory errors rather than falling back silently.
4. With valid evidence, each admitted session uses the measured peak RSS
   unchanged, and default/explicit workers are capped by requested workers,
   available CPUs, and 50% of injected `MemAvailable`; the evidence identity,
   envelope, and calculation are saved.
5. One session whose exact tensor allocation fails the single-session guard is
   rejected rather than launched with one forced worker.
6. Sessions use independent run directories/state and invoke the same
   single-session preparation/execution path without within-session
   parallelism.
7. One session failure does not cancel independent launched sessions; the
   final summary lists all outcomes and the batch exit code is nonzero.

### 10.8 Plot and webapp tests

Create `test_plotting.py` and extend
`src/tests/neural_analysis/test_webapp_package.py`:

1. Heatmaps preserve negative and below-reference values.
2. Unavailable cells are visually/data-wise distinct.
3. Reference values, labels, captions, trial counts, and fold coverage appear.
4. Saved PNGs have opaque light backgrounds, facet all six
   region/representation combinations, and are required only for target
   families present in the run.
5. Coefficient plots distinguish zero, unavailable, and excluded features.
6. The new view has one canonical module owner.
7. Metadata availability keeps the result view selectable without raw-spike
   readiness and presents the no-saved-run state cleanly.
8. Default discovery inspects only direct children of `analysis_runs`; an
   optional nondefault results root is session-relative and contained, while
   traversal outside the selected session and recursive whole-session scans
   are rejected.
9. Hidden incoming directories and incomplete/corrupt runs are never listed.
10. The app routes the decoding-results view before population controls and raw
   spike loading.
11. The view loads a saved fixture and never invokes computation.
12. Metric/selector changes only re-render saved data.

### 10.9 Documentation and command tests

Create `test_task_decoding_documentation.py`:

1. The reusable example configuration loads and validates without CT026 or
   another machine-specific absolute path.
2. Every Python file in `task_decoding`, including `__init__.py`, is described
   in the in-package README; the README also describes the new webapp view,
   example configuration, and Slurm wrapper when present.
3. The top-level neural README contains local dry-run, foreground/detached
   new-run, detached resume, one-shot status, batch, output-location, and
   webapp instructions.
4. `run_session --help` and `run_batch --help` exit
   successfully and show the documented modes.
5. The documented single-session dry-run works against a small temporary
   metadata/config fixture without loading spike arrays.
6. Exact resume and status commands are written before an injected long-running
   stage.
7. README result paths and CLI mode names match implementation constants rather
   than describing obsolete paths.
8. The quickstart explicitly permits the terminal or Codex task to close after
   detached launch and documents later one-shot state/log inspection.
9. The quickstart limits the tested one-column backfill to a table whose sole
   schema gap is `rewards_in_block`, requires a backup, directs other schema
   gaps to normal behavior regeneration, and rejects hand-editing or unrelated
   model reruns.
10. Batch documentation names the explicit measured resource-run option and
    does not imply that a profile is discovered automatically.

Documentation tests should validate stable commands and file ownership, not
word-for-word prose.

### 10.10 Synthetic integration tests

Create `test_synthetic_integration.py`:

1. A seeded two-region synthetic session completes end to end.
2. Injected time-local signal produces its strongest decoding in the expected
   interval without requiring an exact score.
3. All six region/representation combinations are computed, and PFC-only,
   HPC-only, and combined axes are correct.
4. Both target families and both representations complete.
5. A deliberately impossible grouped categorical target is unavailable while
   other targets finish.
6. Saved results reload and reproduce plotted summaries.
7. A held-out-only offset does not influence training scaling/PCA, serving as
   an end-to-end leakage regression.

### 10.11 Single-session cluster-path tests

Write these in WP11 only after the clean WP10 evidence, categorical stress
evidence, and user approval of the exact benchmark-derived resource directives
and tests-first implementation plan. The cluster package itself is required;
this gate resolves its concrete resource and interface details:

1. `submit-new` runs a lightweight cluster dry run, prepares one exact
   cluster run directory through the Python runner, and submits that directory
   to the same foreground pipeline without changing scientific settings.
2. `submit-resume` forwards one exact saved cluster run directory
   without re-parsing or overriding its scientific configuration.
3. Unknown modes, missing paths, wrong repository roots, tracked-dirty
   checkouts, untracked Python inside the task-decoding package, and untracked
   explicit dependency files return nonzero before launch; unrelated untracked
   files do not.
4. The top-of-script partition/task/CPU/memory/time/signal/log/mail fields match
   the user-approved WP10 resource record. Tests do not assume PPC's eight
   CPUs, 32 GB, or 72 hours.
5. Login dry run rejects an approved memory request whose 50% tensor guard
   fails; scheduled execution uses the lower of mocked `MemAvailable` and
   Slurm/cgroup memory limits rather than whole-node memory.
6. The scheduled private branch never invokes `sbatch`; it uses the exact
   submitted commit, explicit thread limits, and
   `uv run --frozen --no-sync --offline`; arguments, exit status, and
   scheduler signals reach the Python process.
7. A simulated `SIGTERM` leaves no false complete result, flushes
   state/logs, and preserves only valid completed-target checkpoints.
8. Resume requires one exact run directory and never searches for a latest
   run.
9. A mocked submission failure is returned, recorded in the prepared run,
   and never triggers automatic resubmission.
10. A mocked immediately-starting job cannot have its `running` state regressed
   by the submitter; scheduler streams use the recorded Slurm log while
   `console.log` remains local-detach-only.
11. Cluster `status` combines saved pipeline state with at most one
   mocked `sacct` lookup, reports the required accounting fields, and
   never polls or mutates the run.
12. Moving the same session/config tree between local and cluster root prefixes
    preserves the scientific fingerprint while recording both resolved
    execution paths.
13. A completed returned fixture loads from `.incoming-<run_id>`
    without cluster source access; an incomplete/corrupt fixture fails before
    final promotion.
14. Documentation contains non-destructive input/result `rsync`
    commands, excludes the configured output root from input synchronization,
    prohibits broad `--delete`, uses an incoming directory, and explains
    exact-commit/offline-environment preparation.
15. No test invokes a real scheduler, SSH, rsync, network, or scientific data.

### 10.12 Bounded cluster batch-array tests

Write these in required WP13 after WP12 acceptance and finalization of the
single-session cluster resource profile:

1. A config list produces one immutable array-index-to-session/run mapping.
2. Each array element invokes the proven single-session cluster path with its
   exact prepared run directory and no shared mutable scientific state.
3. The documented concurrency cap is forwarded exactly.
4. One element's failure does not alter another element's state or outputs.
5. Aggregate status performs one bounded scheduler query and read-only
   per-session status checks; it never polls.
6. Resume instructions name individual failed run directories and never
   resubmit the whole array automatically.
7. A dry run rejects sessions that do not match the measured resource-run
   identity/envelope required by Section 4.8 and reports every admission
   comparison.
8. A two-session synthetic array smoke passes with a fake scheduler; no test
   submits a real job or touches experimental data.

## 11. Work packages and gates

### WP0: Documentation approval

- Sol finalizes revision 5, this plan, the live snapshot, documentation
  deliverables, agent map, and interruption rules.
- No Terra worker, tests, source edits, or data runs.
- Gate: user accepts the documents and separately requests implementation.
- Handoff: record the documentation commit/HEAD and set WP1 as the single next
  package.

### WP1: General behavioral feature

- Terra high writes RED tests for `rewards_in_block` and stops.
- Sol verifies/commits tests, then the same Terra worker implements the additive
  augmented-table feature and the bounded exact-0/1/finite-value correction to
  the shared decision-variable parser, plus the thin atomic backfill in
  `gather_trial_features.py`.
- Sol runs focused and affected behavior tests.
- Gate: new behavior processing emits the column, and the backfill changes only
  that column in an older table without touching feature parameters or leaving
  a temporary artifact on success or injected failure.
- Handoff: record test/implementation commits and the exact proposed backfill
  invocation for later CT026 preparation; do not run it on CT026 here.

### WP2: Configuration and targets

- Terra high writes RED config, scientific/execution payload, validation,
  target, and eligibility tests.
- After Sol's tests-only commit, Terra implements `config.py` and
  `targets.py`.
- Gate: all target definitions and masks are inspectable without neural data.
- Handoff: freeze the config fields/example schema needed by later CLI and
  documentation work.

### WP3: Activity loading and coverage

- Terra high writes RED activity/unit/coverage tests.
- After Sol's tests-only commit, Terra implements explicit region loading,
  stable IDs, trusted bounds, matched trial windows, and regional tensors using
  existing loaders/binning and the existing configurable
  `filter_cluster_metadata` API. It does not modify
  `select_units_by_channels`.
- Gate: shapes, axes, Hz units, coverage, original-row mapping, exact tensor
  allocation guard,
  and unit identities are verified; invalid alignment rows cannot appear as
  zero-filled neural observations.
- Handoff: record representative synthetic tensor shapes and memory-estimate
  inputs; no CT026 arrays are loaded.

### WP4: Grouped modeling

- Terra xhigh writes RED split/preprocessing/model tests.
- An independent Sol xhigh reviewer audits leakage and validity coverage before
  the tests-only commit.
- Terra implements fixed mode first. Sol verifies it before a separate
  follow-up authorizes optional tuned mode.
- Gate: leakage and transform fit-count tests, frozen estimator/PCA controls,
  invalid-fold/error-boundary policy, coefficient contract, and
  deterministic results are green; independent Sol xhigh review finds no
  unresolved scientific/numerical issue.
- Handoff: record exact installed scikit-learn API/version and measured
  synthetic fit counts.

### WP5: Results and single-session pipeline

- Use two bounded RED/commit/GREEN slices rather than one oversized worker
  assignment. WP5A covers NPZ/schema, input/scoped-source identity,
  checkpoints, and atomic result publication in `results.py`. WP5B then covers
  `pipeline.py` and the session CLI: prepare/revalidate/execute,
  failure boundaries, logging/summary, single-writer state, interruption,
  exact resume, optional detachment, and read-only status.
- Terra xhigh handles each slice separately. Independent Sol xhigh review
  approves each tests-only design and each stable GREEN diff before proceeding.
- Gate: synthetic small target runs resume and round-trip; a detached fixture
  returns immediately, completes without its launcher, and is inspectable once
  later without computation or polling.
- Handoff: freeze CLI modes, output paths, schema version, detachment behavior,
  and resume/status semantics before batch, webapp, or README work.

### WP6: Batch runner

- Terra high writes RED dry-run, skip/rerun, and session-isolation tests.
- After Sol's tests-only commit, Terra implements the session-list runner with
  parallelism across sessions only.
- Without explicit completed measured evidence, cap batch execution at one
  worker. Parallel execution requires a
  `--resource-run-directory` satisfying Section 4.5; use its peak RSS unchanged
  for every admitted session, then cap by available CPUs/requested workers and
  50% of `MemAvailable`. A session whose exact tensor allocation fails Section
  6.2 still stops before allocation. Print and save the evidence identity,
  envelope, worker count, and calculation.
- Gate: no within-session parallelism and no shared mutable session state.
- Handoff: record batch input format and exact documented commands.

### WP7: Plotting and integrated webapp view

- Terra high writes RED plotting/routing/read-only tests.
- Sol freezes shared `session_inputs.py`/`app.py` ownership
  and commits tests.
- Terra implements light-mode plotting and the saved-result view; only Sol or
  that one assigned worker edits shared webapp routing files.
- Gate: selecting the view never loads raw spikes or calls fitting code.
- Handoff: freeze user-visible view names, selectors, and no-result messages.

### WP8: User and maintainer documentation

- After WP1-WP7 interfaces are stable, Terra high first writes the stable
  documentation/command tests in Section 10.9 and stops at RED. Sol commits
  them, then the same worker updates `src/neural_analysis/README.md`, creates
  `src/neural_analysis/task_decoding/README.md`, and creates the portable
  example configuration. Documentation changes remain a separate GREEN
  commit.
- Sol runs every documented `--help` and dry-run command against a
  temporary fixture, exercises detached start plus one later status check, and
  checks every Python file is described.
- Gate: a scientist can configure, dry-run, start and leave a run, inspect its
  state/log once later, resume it, locate outputs, and open results without
  reading implementation source; a maintainer can identify every file's role
  from the package README.
- Handoff: record exact commands and note whether cluster documentation remains
  intentionally absent pending WP10.

### WP9: Synthetic full pipeline

- Terra xhigh writes RED end-to-end synthetic tests and, after the tests-only
  gate, makes only bounded integration fixes.
- Sol runs all focused task-decoding tests and affected neural/behavior
  regressions.
- Independent Sol xhigh review audits scientific signal timing, leakage,
  failures, saved outputs, and documentation consistency.
- Gate: deterministic saved output and plots from both target families,
  representations, and regions.
- Handoff: record test counts/runtime, output fixture identity, and the exact
  proposed CT026 preflight commands without executing them.

### WP9A: CT026 augmented-table preparation

- This package requires separate explicit user approval for the exact tested
  behavior-side backfill invocation recorded by WP1. It authorizes one
  augmented-table rewrite only, not neural loading, model/feature
  recomputation, benchmarking, or decoding.
- Sol first re-runs lightweight validation. If the column now exists, compare
  it read-only with the general helper: an exact match records a no-write skip,
  while a mismatch stops for user direction. Do not invoke a backfill that
  would overwrite an existing column.
- Sol records the augmented-table and feature-parameter identities and confirms
  that the backfill command's sole persistent write is replacement of the
  augmented CSV. As a separate, declared safety action, Sol makes the approved
  timestamped backup of that CSV beside it before invoking the helper.
- Terra high acts as a command runner only: run the approved
  `backfill_rewards_in_block_csv` invocation, record its exit status, output
  identity, and log, then stop. It never hand-edits the CSV or invokes the
  broader behavior/model workflow.
- Sol performs read-only post-write validation: all revision-5 required columns
  exist; row count, order, and every previous loaded column/value match the
  pre-run table; the feature-parameter identity is unchanged; no file changed
  beyond the target replacement and declared backup; and the new column passes
  the WP1 semantic checks.
- Gate: the regenerated CT026 table passes validation and the user is shown the
  backup/output identities. A mismatch stops before WP10 and does not trigger
  an ad hoc repair.
- Handoff: record the approved command, the helper's one-file persistent write
  set, the separately created backup path, old/new file identities, validation
  output, and the exact proposed read-only WP10 dry-run.

### WP10: CT026 2026-08-03 preflight and benchmark

- This package requires separate explicit user approval for the exact CT026
  commands.
- WP9A must already have produced and validated the complete augmented table;
  WP10 remains read-only with respect to behavior inputs.
- Before real data, repeat the detached synthetic smoke through the intended
  launcher. If the Codex execution host cleans up detached children, use the
  same documented command from the user's ordinary terminal; do not keep a
  Codex task alive as a workaround.
- Terra high acts as a command runner only: read-only validation, dry-run, and
  the approved bounded representative fixed-mode benchmark. It starts the
  benchmark detached if it is not expected to finish promptly, records the run
  directory and follow-up commands, and stops; it makes no source edits,
  parameter changes, or monitoring loop.
- Sol confirms selected channels/units, eligible trials, block/class coverage,
  tensor memory, exact model-fit count, one-thread execution identity, stage
  timings, and peak memory.
- An independent Sol xhigh reviewer checks raw evidence and the local/full/tuned
  runtime projection.
- Gate: after the process has finished, one explicit later task reads
  `status`, logs, and saved results once. The user reviews the evidence
  and approves the exact resource proposal for the already selected WP11
  path. Local execution remains available for individual sessions, but it is
  no longer an alternative that can remove WP11 or WP13 from scope. There is
  no automatic threshold and no implicit long run.
- Handoff: update the live snapshot with exact config/run directory, launch and
  status commands, logs, fit counts, stage timings, peak RSS, output size,
  the full fixed-mode projection, CPU efficiency, and the proposed Slurm CPU,
  memory, and wall-time request with its safety factors and explicit evidence
  envelope. Record the tuned fit count and that no tuned runtime projection is
  valid until its separately approved representative subset is measured.

### WP11: Single-session cluster execution and transfer

- This is a required implementation package after the clean WP10 benchmark and
  categorical stress gates. The user has selected the architecture in
  principle; Sol must still present the exact resource directives, owned
  files, RED tests, and verification commands for approval before writing
  source.
- Sol records the benchmark-derived resource proposal. Use one CPU unless
  measured evidence justifies more. Proposed memory is the largest of 1.5
  times projected peak RSS, projected peak plus 2 GiB, or twice the exact
  tensor allocation, rounded upward; proposed wall time is twice the
  conservative projected full-run duration, rounded upward to an hour. These
  are first-job safety rules, not permanent defaults, and the user approves the
  exact final directives before tests are written.
- Terra high writes the Section 10.11 mocked wrapper, portability, signal,
  transfer-validation, and documentation tests and stops at RED. After Sol's
  tests-only gate, the same worker implements the thin wrapper and bounded
  Python state/termination support.
- Independent Sol xhigh review checks portable fingerprints, exact-run resume,
  resource visibility, repository/environment identity, one-shot scheduler
  accounting, input/result transfer safety, no automatic resubmission, and no
  duplicated scientific settings.
- Gate: local and cluster dry-run fixtures plus mocked wrapper/transfer tests
  pass; exact one-command `submit-new`, `submit-resume`, and
  `status` instructions and manual rsync round trip are documented. No
  experimental transfer or real job is part of this gate.
- Handoff: record wrapper/resource directives, transfer roots, exact pushed
  commit, environment gate commands, and proposed synthetic Slurm smoke. A real
  transfer/submission remains unperformed until explicitly approved.

### WP12: One-session scientific inspection

- This required cluster-validation package needs explicit approval of the
  exact fixed-mode CT026 configuration, input rsync, and Slurm command.
- First submit the separately approved tiny synthetic Slurm smoke proposed by
  WP11, end the initiating task, and inspect it once later. A
  wrapper/environment/logging failure stops before any CT026 transfer or
  submission; a successful smoke does not itself authorize the experimental
  job.
- Terra high acts as command runner only. It launches the approved Slurm job,
  records the receipt/run directory, and stops. Sol does not monitor or alter
  parameters mid-run.
- In a later user-requested task, Terra or Sol performs one read-only
  status/log/result inspection. If the run is interrupted, resumption is a new
  explicit action using the saved exact command; it is never automatic.
- For a cluster run, first execute the separately approved input rsync and
  cluster dry run. After completion, inspect `sacct` once, record actual
  `Elapsed`, `TotalCPU`, and `MaxRSS`, then separately approve the
  exact result-return rsync into a hidden local incoming directory. Validate
  before atomic promotion.
- Compare requested versus actual resources and freeze the normal
  single-session resource block and exact completed evidence run for later
  sessions. Do not reduce safety margins or generalize beyond the observed
  workload without recording the decision.
- User inspects saved heatmaps, fold coverage, warnings, and coefficients.
- Only after user approval does WP13 begin implementing the batch-array path.
  That approval accepts the single-session evidence; it does not authorize a
  real multi-session submission.
- Handoff: record immutable run paths, configuration, commit, launch and status
  commands, transfer commands, requested versus measured timing/CPU/memory,
  finalized resource block/evidence run, warnings/errors, and user acceptance.
  Do not treat completion as permission for other sessions.

### WP13: Bounded cluster batch array

- This required final package starts only after WP12 scientific acceptance,
  finalized single-session cluster resources, and completed documentation.
  No new decision about whether to implement arrays is needed; the user still
  approves the exact tests-first implementation plan before source work and
  every real array submission separately.
- Terra high writes the Section 10.12 mocked array and two-session synthetic
  tests, stops at RED, and later implements only the narrow submission/status
  extension after Sol commits the tests.
- The array mapping is immutable, concurrency is explicit, every element uses
  the proven single-session runner, and failed sessions resume individually.
- Independent Sol xhigh review checks isolation, resource-profile admission,
  argument quoting, one-shot status, and the absence of array-wide automatic
  retries.
- Gate: focused tests and a fake-scheduler two-session smoke pass; user and
  maintainer documentation includes exact batch commands and recovery. No real
  array is part of the implementation gate.
- Handoff: present the exact proposed real config list, concurrency cap,
  transfer plan, and resource estimate for separate authorization.

## 12. Performance plan

### 12.1 Reuse before optimization

The first implementation should:

- load each probe once;
- build one regional tensor per alignment/bin-width run;
- build one outer split assignment per target;
- fit one regional fold transform and reuse it across time bins;
- reuse separate PFC/HPC PCA projections in the combined analysis;
- compute balanced accuracy and AUC from one classifier fit; and
- checkpoint at target boundaries.

Do not add Numba, Cython, multiprocessing within a session, memory mapping, or
GPU code without a measured bottleneck and user approval.

### 12.2 Work-count warning

The default target set has 8 categorical and 10 numerical targets. At 100 ms
over four seconds there are 40 time bins.

Fixed mode requests:

```
18 targets x 40 bins x 3 region configurations
x 2 representations x 5 outer folds = 21,600 outer fits
```

Tuned mode evaluates 15 candidates in 3 inner folds and then refits once for
each outer fold:

```
21,600 x (15 x 3 + 1) = 993,600 estimator fits
```

This is the principal runtime risk. Tuned mode must remain optional and should
not be presented as an ordinary interactive choice. Before any full tuned run,
measure a representative subset and report the projected duration.

### 12.3 Benchmark stages

Record wall time with `time.perf_counter`. On the planned Linux hosts,
record and normalize peak RSS with the standard-library
`resource.getrusage` API; during WP11/WP12, also preserve Slurm's
reported `MaxRSS` for comparison. No profiling dependency is needed for
this first benchmark. Also record process user/system CPU time so CPU
utilization can distinguish a long single-core workload from a genuinely
multi-core one. Run the local projection benchmark with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`, matching the initial one-CPU Slurm job, and record
those values. Record:

1. configuration/metadata/table validation;
2. sorter/alignment loading and unit selection;
3. PFC and HPC rate-tensor construction;
4. target/split construction;
5. fold preprocessing/PCA;
6. estimator fitting/scoring by target family and representation;
7. result serialization;
8. plotting/report generation; and
9. total wall time, user/system CPU time, and peak resident memory.

Report number of trials, blocks, units, time bins, requested/valid folds, model
fits, invalid cells, and output size so timing is interpretable.

Persist these measurements in structured result metadata as well as
`summary.md`; do not rely on a terminal transcript. For detached work,
flush the stage timing and peak-RSS-so-far at every target checkpoint. The
measurement approach must be the same in foreground, detached local, and Slurm
execution so the comparison is meaningful. A Slurm result additionally records
its requested scheduler resources and job ID; the later read-only one-shot
accounting output is retained in the live benchmark handoff rather than
mutating the completed run.

### 12.4 Benchmark tiers

1. **Synthetic microbenchmark:** catches pathological overhead and produces a
   stable regression fixture; it is not a production-time estimate.
2. **CT026 bounded benchmark:** choice alignment, 100 ms, fixed mode, canonical
   categorical target `current_action` and canonical numerical target
   `relative_doubt`, both representations and all regions. This is
   2,400 requested outer fits and
   includes exact loading/binning once, so it exercises the real session path
   rather than only timing estimators in isolation. Launch it detached if it is
   not expected to finish promptly. The dry run must first confirm that both
   targets have sufficient eligible rows and valid grouped folds; if either
   fails, stop and choose a replacement explicitly rather than silently
   changing the benchmark.
3. **One-shot benchmark inspection:** after launch, the initiating user or
   Codex task ends. In a later task, run `status` once and read the
   completed log/result metadata. If it is still running, report that fact and
   stop; do not begin a watch loop.
4. **Projection:** separate fixed loading/tensor/serialization costs from model
   costs. Project the 8 categorical and 10 numerical targets from their own
   observed per-fit/per-target timings, scale saved-result size by target count,
   and calculate exact planned tensor/result-array bytes. For the same session,
   project peak RSS as measured bounded-benchmark peak plus only positive
   growth in planned in-memory result arrays; the full tensor is already
   present in the bounded benchmark. For a later session, reuse evidence only
   when analysis/source/environment/platform/mode/threading identity matches
   and every resource-driving dimension, including categorical and numerical
   fit counts separately, is within the measured envelope; use the measured
   peak RSS unchanged rather than scaling it. Otherwise run another bounded
   benchmark. Report timing estimates and a conservative range because
   eligibility and convergence affect cost; do not multiply total bounded wall
   time by nine.
5. **Full default benchmark/run:** only after the user authorizes the exact
   local or cluster path. A local run uses `new --detach`; a cluster run
   uses the approved rsync, dry-run, and `submit-new` sequence. End the
   initiating task and inspect once later. Compare measured versus projected
   wall time, CPU time, peak RSS, fit throughput, and output size.
6. **Tuned estimate:** representative subset only unless separately authorized;
   measure its peak RSS as well as fit time because inner-fold preprocessing
   workspace is not inferred from the fixed-mode profile.
7. **Local/cluster resource decision:** Sol reports ordinary local practicality
   and proposes cluster resources. The cluster implementation path is selected
   because serial per-session runtime compounds over a multi-session dataset;
   the decision does not itself authorize transfer or submission. Local
   execution remains supported for bounded validation and individual sessions.

### 12.5 Unattended local-to-cluster decision path

Follow WP9A-WP12: validate/regenerate the table when needed, dry-run, launch the
bounded local benchmark and leave it unattended, inspect its durable state
once later, then implement and validate the selected Slurm path. Each real
cluster action remains explicitly authorized. The Slurm branch uses the
Section 4.7 transfer and Section 4.6 wrapper; returned results are validated in
the hidden incoming directory before promotion. Local and cluster execution
use the same prepared-run, checkpoint, log, and result contracts, and neither
polls, retries, nor resumes automatically. After one cluster session is
accepted, WP13 adds explicitly capped cross-session job-array concurrency.

### 12.6 Slurm resource selection and finalization

`hpc_ppc.sh` proves the site accepts partition `unlimited`, one
task, explicit CPU/memory/time requests, five-minute termination notice, the
private log directory, and existing mail settings. Its PPC-specific values are
not task-decoding defaults.

After the bounded local benchmark, Sol produces a resource table containing:

- requested fit count and measured fit throughput by target family;
- full-run conservative wall-time range;
- measured and projected peak RSS, including full tensor and scaled result
  storage;
- user plus system CPU time divided by wall time;
- proposed tasks/CPUs, memory, wall time, partition, signal notice, and log
  path; and
- the safety-factor calculation and benchmark session/config identity.

For the first task-decoding cluster job, request one task and one CPU unless
measurement supports a revised parallel plan. Propose memory as the largest of
1.5 times projected peak RSS, projected peak plus 2 GiB, or twice the exact
tensor allocation, rounded upward. Propose wall time as twice the upper-bound
full-run projection, rounded upward to a whole hour. If these requests exceed
site limits, stop for a new plan; do not reduce them silently.

After the first completed cluster run, compare the request with Slurm
`Elapsed`, `TotalCPU`, `AllocCPUS`, `MaxRSS`,
`ReqMem`, and `Timelimit`. Record and user-approve the resulting
normal single-session resource block and its exact completed evidence run. A
later session can reuse it only under the identity and no-larger-envelope rule
in Section 4.8. Otherwise stop for another bounded benchmark and explicit
estimate. WP13 uses the approved per-session block for each admitted element
rather than creating a larger shared allocation.

## 13. CT026 2026-08-03 validation fixture

Planned session root:

```
/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference
```

Planned inputs:

- metadata: `neural_session.json`;
- augmented trials:
  `processed/CT026_2026-08-03_111938_augmented_trials.csv`;
- feature parameters: `processed/trial_feature_params.json`;
- PFC: `ProbeA`;
- HPC: `ProbeB`.

Read-only audit observations:

- 650 augmented rows;
- 646 valid animal choices;
- 83 behavioral blocks;
- both requested alignment columns are present;
- both probes have 3,780 finite IRIG values with common bounds
  1785770350.0 to 1785774129.0 UTC seconds;
- ProbeA has 132 good, 72 MUA, and 49 noise clusters before channel filters;
- ProbeB has 229 good, 81 MUA, and 119 noise clusters before channel filters;
- the augmented table contains all revision-5 source columns except the new
  `rewards_in_block` column; and
- metadata currently points to the raw trial CSV, which is why the augmented
  path is explicit in the decoding configuration.

WP1 implements the general feature but does not touch CT026. In separately
approved WP9A, create the missing column with the tested behavior-side backfill
recorded by WP1, with the recoverable backup and post-write checks defined
there. Do not hand-edit the experimental CSV or rerun the broader behavior/model
workflow.

These counts are preflight facts, not authorization to mutate the table or run
decoding.

## 14. Dependencies and API verification

Use existing project dependencies only:

- Python standard library;
- NumPy;
- SciPy;
- pandas;
- scikit-learn;
- Pynapple;
- Matplotlib;
- Streamlit; and
- pytest.

All local Python commands use `uv run`. Before implementation, verify
the exact installed APIs from package source or official documentation. The
audit found scikit-learn 1.8.0 with:

- `ElasticNet(alpha=..., l1_ratio=...)`;
- `LogisticRegression(solver="saga", C=..., l1_ratio=...)`;
- deterministic `GroupKFold(..., shuffle=False)`; and
- deterministic `StratifiedGroupKFold(..., shuffle=False)`.

Pin behavior in tests rather than relying on memory of older scikit-learn
interfaces.

## 15. Risks and deliberate trade-offs

| Risk | Planned response |
| --- | --- |
| Fixed/tuned fit cost is large | Reuse fold transforms, benchmark fixed mode, measure a tuned subset before projection, and checkpoint targets. |
| Leakage or invalid grouped folds distort results | Fit preprocessing only within each training subset; test fold membership, transform call counts, and declared invalidity behavior. |
| Dry-run memory guesses are misleading | Guard only exact tensor allocation against the applicable local/job/request budget; use measured CT026 peak RSS for concurrency and Slurm sizing. |
| Scientific identity changes across hosts or unrelated edits | Use session-relative input identities and scoped scientific-source hashes; keep host paths/full Git state as provenance. |
| A long run is interrupted or resumed twice | Publish atomically, checkpoint targets, keep durable state/logs, and enforce one bounded execution guard without a service. |
| Unexpected defects are hidden as unavailable science | Limit unavailable results to enumerated invalidities; fail and preserve checkpoints on unexpected exceptions. |
| The saved-results UI recomputes | Route before raw-data loading and test a saved-loader-only view. |
| Cluster execution diverges or transfer exposes partial state | Reuse the prepared Python run, require exact clean pushed code, keep rsync explicit, and validate hidden incoming results before promotion. |
| Handoff or concurrent editing loses TDD state | Keep dated package records, one Terra writer, separate RED/GREEN commits, and Sol reproduction of every gate. |

## 16. Approval checklist

Before implementation, confirm:

- revision 5 is the scientific authority and WP0 remains documentation only
  until a separate implementation request;
- `rewards_in_block` is an additive general augmented-table column with the
  frozen entering-trial semantics;
- target eligibility, explicit probe/coverage rules, common tensor-row mapping,
  native numerical units, and all six region/representation outputs match the
  specification;
- fold-local transform reuse and declared-versus-unexpected error boundaries
  are covered by tests before modeling code;
- session-root containment, portable input identities, scoped source identity,
  and analysis-version rules are frozen;
- dry-run guards exact tensor bytes only against the applicable local or Slurm
  memory budget, the local benchmark uses recorded one-thread limits, and
  measured peak RSS/runtime drive local and Slurm sizing;
- one prepared-run path provides immutable outputs, atomic target checkpoints,
  detached launch, exact resume, and one-shot status without active monitoring;
- the only UI is an integrated saved-results view, with scientist quickstart,
  maintainer file map, and portable example configuration;
- Sol/Terra work follows the documented RED-commit-GREEN sequence and updates
  this live handoff at every gate; and
- CT026 table preparation, benchmark/full run, cluster transfer/submission, and
  final batch-array follow-up each retain their separate user gates; a chosen
  cluster path also passes a separately approved synthetic scheduler smoke
  before experimental submission.

Approval of this document should be followed by a separate implementation
request. Until then, no code, tests, augmented data, or neural results should be
changed.
