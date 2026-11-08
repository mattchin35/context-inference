# Trial-policy Dynamax analysis: implementation plan and live handoff

Status: WP0-WP6 complete. WP7 ran CT026 2026-08-28 successfully and awaits user inspection; Bayesian/WSLS identifiability limitation recorded. No batch authorized.

This plan implements [trial_dynamax_hmm_spec.md](trial_dynamax_hmm_spec.md) using the principles in [SoftwareDesign.md](SoftwareDesign.md). The specification owns scientific decisions; this document describes modules, interfaces, tests, integration, and delivery order.

## 0. Live implementation handoff

This is the durable handoff for a DIFFERENT implementing agent/chat. The planning author is not the implementation owner. Follow [AGENTS.md](../AGENTS.md) and the live-handoff format used in [event_sequence_heatmap_implementation_plan.md](event_sequence_heatmap_implementation_plan.md#0-live-handoff). Update this snapshot and ledger at RED, tests-only commit, GREEN, review, interruption, and user-inspection boundaries; append exact evidence to Section 11. Do not treat this document as an executable checkpoint system.

### Current snapshot

| Field | Current value |
|---|---|
| Inspected checkout | 2026-11-07; branch `refactor`; starting HEAD `9ffe30e859b95e5fdeb49b90892abd05154b5ea7`. Observation only: recheck, never reset to it. |
| Authority | Implementation authorized, then user: "you can try CT026 8/28". This authorizes only the resolved `CT026_2026-08-28_134630` single-session validation; no batch authorized. |
| Implementation owner | Lead `/root`; sequential WP1 Sol-high, WP3/WP4 Luna-medium workers complete; independent WP6 Sol-high audit complete. No active worker/job. |
| Feature state | Four feature modules, Bayesian guard, and tests committed. Main integration is implemented/tested but remains uncommitted alongside the pre-existing main migration; feature-only patch is preserved in ignored `.cache/trial_policy_implementation/trial_policy_main_changes.patch`. |
| Dependencies not recoverable from HEAD alone | Pre-existing main/block migration in dirty `src/main.py`; untracked `exemplar_hmm_results.py` and `exemplar_hmm_plotting.py`. Preserve actual working files. `block_regression.py` and `exemplar_hmm_modeling.py` are tracked. |
| Other work | Pre-existing `sync_ephys.py` edit and untracked files retained. Unexpected `docs/SAC_behavior_prep.md` entered commit `de94775`; user explicitly instructed ignoring that document and continuing. Leave its current changes and commit intact. |
| Verification | Final suites: 268 numerical/plot/feature/block + 89 legacy main/window + 11 new workflow = 368 passed. Fresh main import excludes `ssm`; synthetic single/joint/four-adapter figures saved and visually reviewed. |
| Exact next action | User inspects the CT026 August 28 diagnostic and identical Bayesian/WSLS-policy finding before deciding any subsequent candidate/configuration or batch work. Preserve dirty-main dependencies. |
| Experimental validation | `CT026_2026-08-28_134630`: 698 rows, 695 comparisons, three unknown-flag exclusions; saved/visually inspected outputs and preservation checks pass. User inspection pending; no batch run. |

### Package ledger

States advance only after the receiving lead reviews the actual diff and evidence. A worker completion message is not acceptance. Record actual model/effort/agent ID when assigned; model names below are planned roles, not claims about a chat's runtime.

| Package | Planned owner | Current state / next gate | RED command/result + test commit | GREEN/review + source commit or uncommitted paths |
|---|---|---|---|---|
| WP0 baseline/contracts | Lead `/root` | Complete | Existing tests only | 101 + 89 passed; no source edits |
| WP1 causal adapters | `/root/wp1_adapters`, Sol high | Accepted | `2360445`, correction `a1cb550` | 146 passed; `16b8a42` |
| WP2 likelihood/HMM | Lead `/root` | Accepted | 26 failed, `5e6447a` | 26 passed; `9700ff7` |
| WP3 persistence/reporting | `/root/wp3_results`, Luna medium; lead review | Accepted | `fd310d1`, `96c56d2`, `acc8b5f`, `1d1f0b5` | 29 current cases pass; source `82bc108`, `cd5ef8f`, `c66fcae` |
| WP4 saved-result figures | `/root/wp4_plotting`, Luna medium; lead review | Accepted; actual PNGs reviewed | `fa496e5`; fixture correction `d45e2ec`; audit `8e24b2f` | 10 current cases pass; source `9753c34`, `c66fcae`, `8c41826` |
| WP5 main/batch integration | Lead `/root`; WP6 independent Sol review | Accepted in working tree | `3575d6b`, `0537290`, `1d1f0b5`; scoped legacy fixtures | 11 new + 89 legacy cases pass; main source intentionally uncommitted |
| WP6 synthetic/regression audit | `/root/wp6_audit`, Sol high; lead | Complete | Two plotting findings fixed tests-first; import fixture corrected | 368 final tests passed; synthetic artifacts and figure evidence below |
| WP7 designated-session inspection | Lead `/root` + user | Run complete; user inspection pending | User-designated CT026 8/28 | Numerical/preservation/visual checks pass; identical Bayesian/WSLS recommendations on 695 comparisons; no batch authorized |

### Receiving/resuming procedure

1. Read this snapshot, latest Section 11 entry, spec, AGENTS.md, and SoftwareDesign.md. Record later implementation approval once; do not re-request it at every package.
2. Inspect actual HEAD, branch, full status, staged/unstaged diffs, untracked dependencies, and recorded worker/command sessions. Preserve partial work. In a new checkout, transfer the relevant working files; a checkout of the recorded HEAD alone is insufficient. Stop for unexpected concurrent edits.
3. Inspect the first unfinished package's files and evidence. Resume its unfinished RED/GREEN/review gate rather than restarting completed work or running nonexistent tests. Recheck completed tests only when evidence is missing or relevant edits invalidate it.
4. Before dispatch, record package, actual worker identity, writable files from Section 9, test command, and expected deliverable. Only the lead handles Git and this handoff. Maximum one active worker; do not assume old agent IDs remain usable after a chat handoff.
5. Before stopping, update the ledger and append commands/results, test/source commits or uncommitted file paths, outstanding findings, active tool session/PID/log/artifact paths (or `none`), and one exact next action. Preserve failures; do not replace evidence with an unsupported 'passed'.

Initial read-only commands, separately from the repository root:

```bash
git status --short
git diff --stat
git diff --cached --stat
git rev-parse HEAD
```

WP0 baseline after authorization (existing paths; do not infer feature RED from an environment/collection failure):

```bash
UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync pytest -q src/tests/behavior_analysis/test_trial_features.py src/tests/behavior_analysis/test_fixed_exemplar_block_states.py src/tests/behavior_analysis/test_exemplar_hmm_results.py src/tests/behavior_analysis/test_exemplar_hmm_plotting.py
UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync pytest -q src/tests/test_main_dynamax_migration.py src/tests/test_main_choice_windows.py src/tests/test_main_multisession.py
```

Record exact results and any pre-existing failures. Never reuse historical block-migration pass counts as evidence for this implementation.

## 1. Outcome and boundaries

Add trialwise comparison of three existing causal policies: forgetting Q-learning, the full Bayesian `HMM_rel_value_logodds` policy, and WSLS. Convert their greedy recommendations to Bernoulli choice likelihoods with shared opposite-action error probability 0.10. Save independent comparisons and, by default, fixed-transition Dynamax smoothed probabilities with trial stay probability 0.90.

The normal single-session and full multisession workflows will invoke the analysis. Each source session has independent policy initialization and HMM inference. Cropped inputs initialize at their first retained row. All input rows remain on the HMM axis, including excluded observations. Existing block analysis retains its current behavior, including continuous multisession block smoothing.

The reward-decay HMM and history ideal observer are required subsequent additions. Round one provides the extension point and verifies a synthetic fourth candidate, but does not advertise those two production adapters as implemented. No learning-rule changes, emission fitting, parameter searches, residualized policy regressors, or legacy trial GLM-HMM execution are included.

Subsequent-round status (2026-11-08): those adapters and standard Q-learning are
now implemented under the approved
[candidate expansion plan](trial_policy_candidate_expansion_plan.md). All six
are defaults, subsets remain selectable, and the new membership grids preserve
ties with one row per model. The linked handoff records the v2 validation and
artifact paths; the first-round implementation history below remains unchanged.

## 2. Existing code to reuse and localized changes

| Current location | Use in this implementation |
|---|---|
| `src/behavior_analysis/trial_features.py` | Reuse the three pre-choice feature functions. Add a small impossible/non-finite outcome-normalization check inside the existing full Bayesian update. |
| `src/behavior_analysis/gather_trial_features.py` | Use `TaskParams` and its effective parameter values; do not substitute simulator defaults. |
| `src/behavior_analysis/project_utils.py` | Reuse flag alias normalization and missing-value conventions. Validate exact action codes at the new boundary. |
| `src/behavior_analysis/exemplar_hmm_modeling.py` | Reuse `build_sticky_transition_matrix` and `smooth_fixed_exemplar_likelihoods_with_dynamax`; ignore its argmax hard labels. |
| `src/behavior_analysis/exemplar_hmm_results.py` | Follow named NPZ/meta, cache identity, archival, and run-record conventions. Reuse public export preservation where appropriate. Do not route trial arrays through the block-specific serializer. |
| `src/main.py` | Integrate after block output attachment, pass source-specific parameters/paths, and orchestrate session workers and plotting. |
| Existing block/choice-window tests | Regression coverage for preservation, collection-only behavior, workflow contracts, and output locations. |

Do not extract a general HMM framework or refactor block persistence merely to share a few lines. Preserve existing calls to the Bayesian function by adding one optional keyword-only `validate_outcome_evidence=False` argument. The new trial adapter enables it; existing callers retain their behavior. This narrowly scoped validation switch avoids making an unrelated upstream feature calculation fail before the trial stage. It leaves valid-input arithmetic unchanged and raises ordinary `ValueError` on impossible evidence when enabled. Check all callers and test both value modes before making this localized change.

## 3. Proposed architecture

Use four focused function modules under `src/behavior_analysis`. Public boundaries are specified below; private helpers may be simplified without changing these contracts.

| Module | Responsibility and principal functions |
|---|---|
| `trial_policy_models.py` | Validate/normalize one session's history, resolve ordered candidate definitions, and replay existing policy functions. `prepare_trial_policy_history`, `resolve_trial_policy_candidates`, `replay_trial_policy_candidates`. |
| `trial_policy_analysis.py` | Convert scores to probabilities, calculate likelihoods and independent assignments, and smooth full-length arrays. `compute_trial_policy_comparison`, `smooth_trial_policy_comparison`, `assign_unique_probability_maxima`. |
| `trial_policy_results.py` | Cache lookup, save/load numerical artifacts and metadata, attach results to current trial tables, and write concise run records. `find_trial_policy_cache`, `save_trial_policy_results`, `load_trial_policy_results`, `attach_trial_policy_outputs`. |
| `trial_policy_plotting.py` | Render single-session and multisession diagnostics from loaded artifacts. `plot_trial_policy_diagnostic`, `plot_multisession_trial_policy_diagnostic`. |

Dependency direction is models -> analysis -> results; plotting consumes saved result data. `main.py` assembles paths, parameters, candidate selection, and run controls. Numerical helpers receive arrays/scalars; dataframe helpers are limited to validation and row-aligned attachment. None of the new computational modules imports `main.py`, `ssm`, or the legacy trial modeling module.

Use small data-only records where they improve clarity: an immutable candidate definition (ID, label, adapter, parameter description), prepared history, and a numerical result. No behavior-heavy result classes, inheritance, mutable global registries, factories of factories, or plugin discovery. Candidate definitions are an ordinary ordered collection; the analysis consumes their arrays without branching on model names.

### Data contracts

- One input table contains `T` chronological retained rows from one source session. Arrays for action, reward, eligibility, row position, and identifiers have shape `(T,)`. Validate session membership and alignment; never sort or drop rows silently.
- Raw actions are dimensionless `0=Right`, `1=Left`. Reward is the existing binary delivery indicator, not physical water volume. Missing rewards remain missing. Original columns are not mutated.
- Candidate adapters return left-positive signed scores and prediction availability with shape `(T,)`. Parameters are explicit, and all candidates share the approved history eligibility. Scores are dimensionless; FQL values use the existing reward-indicator scale.
- Stacked scores, probabilities, likelihoods, residuals, and winner masks use `(T, K)`: rows are retained trial positions, columns are the ordered candidate IDs. Likelihoods are natural log probabilities. Residuals, if retained, are `y_right - p_right` and dimensionless.
- Validity, state, and tie arrays use `(T,)`; transition matrix `(K, K)`; initial distribution `(K,)`. Save state indices with candidate IDs. Use missing-capable representations for independent state/tie values on invalid rows; masks make unavailable predictions distinct from indifferent models.
- HMM fields are explicitly absent when smoothing is disabled. Empty and all-invalid status handling follows the spec. Dataframe attachment preserves row order/index and unrelated columns; persisted identifiers permit checked reattachment to freshly loaded tables.
- Every new/modified function documents input types, shapes, conventions, units, and return values. No global JAX precision change is permitted.

### Public boundaries and saved representation

Use these stable IDs/order: `("forgetting_q_learning", "bayesian_hmm", "win_stay_lose_switch")`. Adapters call `forgetting_qlearning_relative_value`, `hmm_relative_value(value_mode="bayesian_log_odds", validate_outcome_evidence=True)`, and `win_stay_lose_switch_regressor`, respectively. All functions are in `trial_features.py`. Only selected adapters execute. Adapter callables stay in code; metadata stores plain IDs, effective parameter dictionaries, and provenance strings.

FQL maps `TaskParams.FQL_decay` to `decay`, uses `reward_update_rate=1-decay`, and does not use the fast-learn field. Bayesian maps `state_transition_prob`, `active_reward_probability`, `inactive_reward_probability`, `correct_reward_size`, `incorrect_reward_size`, and `tanh_scale` directly. All adapters receive normalized action/reward/manual flags and the two-choice convention; WSLS has no fitted parameters. Save derived effective values such as the FQL update rate explicitly.

The following are interface contracts, not code to implement during planning. Named records below are data-only; private helper names may vary.

```text
prepare_trial_policy_history(trial_df, *, session_id) -> PreparedTrialHistory
resolve_trial_policy_candidates(model_ids) -> tuple[TrialPolicyCandidate, ...]
replay_trial_policy_candidates(history, candidates, feature_params)
    -> (scores[T,K], prediction_valid[T,K])
compute_trial_policy_comparison(history, scores, prediction_valid, model_ids, *,
    opposite_action_error_prob=0.10, probability_tie_atol=1e-6) -> TrialPolicyResult
smooth_trial_policy_comparison(result, *, stay_prob=0.90,
    probability_tie_atol=1e-6) -> TrialPolicyResult
assign_unique_probability_maxima(probabilities, valid_rows, *, atol=1e-6)
    -> (best_model_mask[T,K], states[T], tie_flags[T])
find_trial_policy_cache(processed_data_path, signature) -> Path | None
save_trial_policy_results(result, *, processed_data_path, metadata) -> Path
load_trial_policy_results(path) -> (TrialPolicyResult, metadata_dict)
attach_trial_policy_outputs(trial_df, result, *, metadata) -> new trial_df
write_trial_policy_export(trial_df, *, export_directory, numerical_path,
    metadata, current_csv_path=None) -> Path
plot_trial_policy_diagnostic(result, metadata, *, output_path) -> Path
plot_multisession_trial_policy_diagnostic(results, metadata_records, *, output_path) -> Path
```

`PreparedTrialHistory` contains normalized actions/rewards `(T,)`, adapter-local skip flags, scoring/update eligibility and reason arrays, original trial/block IDs, session ID and zero-based retained row positions. Missing rewards are NaN; missing/no-choice actions never enter choice likelihoods. `TrialPolicyCandidate` contains ID, label and ordinary replay function. `TrialPolicyResult` contains the named numerical arrays below and `status`; computation neither reads/writes files nor emits global warnings. Smoothing returns a new result with HMM fields, without mutating independent outputs.

Use the explicit full session ID from the runner and validate any input `session_ID`/`source_session_id` against it. `trial_id` comes from `cur_trial`; `block_id` comes from `source_cur_block` on concatenated input, otherwise `cur_block`. Nonempty inputs lacking these columns raise a clear error; empty inputs may produce empty identifier arrays. Unicode IDs are display/validation values, not a promise of mixed-type CSV round trips. Checked attachment uses session ID, retained row position, and matching validated input identity; preserve the current table's original typed columns in memory.

`metadata` carries `input_digest` and the original source flag-column schema. Compute the digest from normalized actions, rewards, flags/eligibility, original identifiers, retained positions, initialization/crop convention, AND the original flag-column schema before alias normalization. Represent that schema as a fixed-order list of present names from `("give_reward", "experimenter_reward_given")`: empty, legacy-only, current-only, or both. Schema-only changes invalidate the digest/cache even when normalized values are identical; dataframe column reordering does not. Before attachment, recompute this identity from the current table using the same preparation rules and compare it to the saved digest; matching actions and IDs alone are insufficient. This check is separate from the broader cache signature, which also includes model parameters and code. The loader returns the required metadata with the result; fresh computation passes its original prepared metadata. Never accept a caller's asserted current digest without checking the current history.

| Named NPZ arrays / result fields | Dtype, shape, missing convention |
|---|---|
| `model_ids`; `session_id` | Unicode `(K,)`; Unicode scalar |
| `trial_id`, `block_id`, `row_position` | Unicode `(T,)` display IDs; int64 `(T,)` retained positions. Keep original typed IDs in current CSVs. |
| `observed_action` | float64 `(T,)`, validated raw `0/1`, NaN for no choice. Save this so plots can show observations without rereading mutable source CSVs. |
| `comparison_valid`, `history_update_valid` | Boolean `(T,)` |
| `comparison_exclusion_reason`, `history_update_exclusion_reason` | Unicode `(T,)`; empty string means included. For multiple reasons, join stable reason names with `;`. |
| `prediction_valid`, `model_indifferent` | Boolean `(T,K)`; indifference is false for unavailable predictions |
| `greedy_choice` | int8 `(T,K)`; raw `0/1`, `-1` for indifference/unavailable (distinguished by masks) |
| `p_right`, `log_likelihoods`, `choice_residuals`, `independent_probs` | float64 `(T,K)`, NaN when unavailable; include residuals in round one since they are a simple saved subtraction |
| `independent_best_model_mask` | Boolean `(T,K)`; all false on invalid comparisons |
| `independent_state`, `independent_tie` | int64 `(T,)`, `-1` for no unique state; int8 `(T,)`, `-1` unavailable, `0` unique, `1` tied |
| `initial_probs`, `transition_matrix`, `hmm_smoothed_probs` | float64 `(K,)`, `(K,K)`, `(T,K)`; preserve the actual Dynamax numerical precision in metadata |
| `hmm_smoothed_best_model_mask`, `hmm_smoothed_state`, `hmm_smoothed_tie` | Boolean `(T,K)`; int64 `(T,)` with `-1` for ties; Boolean `(T,)`. Invalid comparison rows may still have HMM assignments. |

HMM fields are `None` in an unsmoothed in-memory result and omitted from its NPZ. With smoothing enabled, an empty input has `(0,K)`/`(0,)` outputs and configured prior/transition arrays without a Dynamax call. All-invalid nonempty input follows the spec's uniform/prior-only status, including the one-candidate exception. Store `status` and all scalar settings in `meta`, using the existing trusted-project NPZ metadata convention; do not serialize adapter callables or live JAX/model objects.

Current-table columns use `trial_policy_` plus these explicit names: `comparison_valid`, `history_update_valid`, the two reason fields, `independent_state`, `independent_tie`, `hmm_smoothed_state`, `hmm_smoothed_tie`; per-model fields use `<field>_<model_id>` with the exact array field names above. Convert negative missing sentinels to pandas nullable values in CSV exports. Keep state-to-ID mapping in metadata and add `trial_policy_independent_model`/`trial_policy_hmm_smoothed_model` nullable strings for readability. Existing action and ID/position columns are preserved, not duplicated under this prefix.

### Policy replay and simple edge handling

Normalize supported missing sentinels and flag aliases once. Unknown values in an existing manual-reward flag column exclude both scoring and history updates. An absent flag column follows the existing animal-controlled compatibility assumption and records it. Pass normalized adapter-local arrays to existing functions; do not change the legacy feature columns or shared eligibility rules.

Parsing is based on inspected inputs and the existing producer/loader. Accept finite exact numeric `0/1` actions, rewards and flags, including equivalent CSV strings and integer-valued float representations; do not truncate fractional values. Actions also accept the existing `None`/`no_choice`/empty and real missing-value sentinels as no choice. Reward/flag missingness follows `project_utils.MISSING_STRINGS` and real missing values. Preserve native boolean flags' existing numeric `0/1` behavior, but add no textual `true/false`, `yes/no`, or `left/right` aliases. Textual side names are invalid actions; unrecognized flag values follow the documented unknown-flag exclusion and warning rule. Normalize numeric/missing values source-locally before checking legacy/current alias agreement. Do not broaden shared legacy parsers.

For a valid choice with a missing reward, the existing functions supply a pre-choice score and skip the update. Preserve that behavior, including freezing forgetting/hazard advancement and WSLS history. Do not turn the row into a no-choice row, which would incorrectly remove its current score. Missing candidate predictions affect the common scoring mask, not other candidates' otherwise valid history updates.

The full Bayesian function checks its existing outcome-normalization mass before division and reports its row position and offending observation/parameters. The adapter adds session/trial/model context using normal exception handling. Do not build a separate replay to validate another replay. Use exact-zero score neutrality for the initial adapters and the spec's separate `1e-6` probability maximum-tie tolerance.

Return ordinary diagnostic counts/row flags from computation. The workflow emits one concise warning per condition/session, with a short ID preview; full row flags remain in the numerical output. Workers do not install global warning capture or shared logging handlers. Errors propagate with context through existing reporting; there is no automatic repair, fallback policy, retry loop, or custom exception hierarchy.

Create the parent workflow record before parameter/history validation: single-session records use `<processed_data_path>/trial_policy_hmm/runs/<timestamp>/`; multisession records use `<cross_session_analysis>/trial_policy_hmm/<timestamp>/`. Both are independent of numerical signatures. Write `run.log` and the ordinary run metadata there, initially recording the requested session(s)/options, then link numerical artifacts and figures as they become available. Report early validation errors to the console and this same log with source-session context. No separate error-record framework or assumption of an existing global main.py logger is needed.

## 4. Numerical sequence

1. Prepare a single session's history and effective parameters. Resolve a nonempty candidate selection, rejecting duplicate/unknown IDs.
2. Replay the three existing policies from the first retained row. Initial zero scores map to indifference. No reset occurs at task block boundaries.
3. Convert finite scores to greedy choices and `p_right` using the shared error probability. Never use the repeat-on-tie agreement helper.
4. Build `comparison_valid` from animal-choice eligibility and availability of every selected candidate prediction. Score only these rows with `scipy.stats.bernoulli.logpmf` after converting `y_right = 1 - action`.
5. Normalize valid-row likelihoods with `scipy.special.softmax(axis=1)`. Derive maximum masks, nullable unique winners, and tie flags.
6. If smoothing is enabled, copy the likelihood matrix and replace entire invalid rows with zeros. Call the existing generic Dynamax smoother on all `T` rows. Derive trial-specific tie-aware assignments from its probabilities.
7. Handle empty/all-invalid inputs explicitly. Attach numerical results by checked row identity and save them before plotting.

Independent likelihood outputs remain missing on invalid rows; the zero-filled HMM input is a separate array. A single model's missing prediction never receives a favorable neutral likelihood while other models are scored. The simple conditional likelihood model remains retrospective after smoothing; no new predictive-performance metric is introduced.

Validate the constructed float64 probabilities as well as `0 < error_prob < 0.5`: both `error_prob` and `1-error_prob` must remain strictly inside `(0,1)`. Reject settings such as `1e-20`, whose complement rounds to one, with a clear parameter error. Do not clip probabilities silently or feed resulting `-inf` likelihoods into the finite-matrix smoother.

### Approved numerical reuse boundary

The receiving chat reproduced a correctness failure in Dynamax 1.0.2's `hmm_smoother` at `stay_prob=1`: 12 observations with likelihoods `[.9,.1]` followed by 13 with `[.1,.9]` should give `[.1,.9]` at every row, but early rows return `[1,0]`. Source inspection confirms an `isclose(predicted_probability, 0)` cutoff. A planning probe reproduced this; `hmm_two_filter_smoother` fixes that example but fails a longer 60/61 reversal through underflow. Float64 alone does not remove the original cutoff.

User approved the bounded range for round one: for `K>1`, the requested finite float64 `stay_prob` must satisfy the inclusive bounds `1e-6 <= stay_prob <= 1-(K-1)*1e-6`; retain the explicit one-state case `A=[[1]]`. These requested-value bounds are authoritative and receive NO tolerance. The mathematical transition-entry floor is `1e-6`; derived matrix entries may fall short only by `transition_roundoff_atol=1e-12` (absolute, relative tolerance zero). Apply this small allowance to the constructed float64 matrix and its JAX dtype conversion, not to the user's requested range. Do not clip/renormalize entries to force the floor. This accepts the three-state upper endpoint's approximately `9.999999999732445e-7` off-diagonal entries while keeping the numerical margin above Dynamax's cutoff.

Test both exact bounds and the immediately outside float64 values obtained with `np.nextafter(lower, -np.inf)` and `np.nextafter(upper, np.inf)` for K=2,3,4. Requested stay=0/1 for K>1 must fail before Dynamax. Compare long evidence reversals at both bounds and default .90 to a stable test-only log-domain reference; require finite normalized probabilities and use `atol=2e-6, rtol=0` for float32 posterior/reference comparisons. This test tolerance is distinct from the `1e-6` probability maximum-tie rule and the `1e-12` matrix roundoff allowance. Keep the existing smoother and block workflow unchanged; no installed-package patch or alternate production inference implementation is required. This supersedes the former unrestricted `[0,1]` trial contract.

## 5. Parameters, persistence, and figures

### Parameter resolution and cache identity

In single-session runs, pass the effective `TaskParams` returned by current feature collection. For each multisession source, precedence is: an explicit complete `TaskParams` override keyed by full source-session ID, otherwise that source's `trial_feature_params.json`, otherwise documented `TaskParams()` defaults. An override replaces saved parameters rather than merging with them. For partial legacy JSON, fill missing fields from documented defaults and record/warn which candidate-relevant fields defaulted. Malformed JSON or invalid supplied values raise a contextual error; they do not trigger silent fallback. Ignore fields unrelated to selected candidates when forming the numerical parameter signature, while retaining the parameter source in provenance. Never borrow another session's object or infer old column provenance from JSON presence.

For round one, replay selected existing feature functions from normalized history on every numerical cache miss. Do not reuse old score columns. Reuse the new numerical cache only when its input/parameter/code identity matches; no trusted/untrusted feature framework is needed.

Cache identity includes the schema-bearing `input_digest` defined above, effective candidate parameters, ordered IDs, error rate, smoothing settings and numerical tolerances, analysis/source version, and relevant package versions. Absent versus present-all-zero flags and legacy/current alias renaming therefore cause cache misses; do not reuse old arrays and merely replace their schema provenance. Preserve exact effective parameters in `meta`. Deterministic analysis records `random_seed=None`. Unrelated derived dataframe columns should not invalidate numerical inference or be restored from stale cache snapshots.

Store named arrays at:

`<processed_data_path>/trial_policy_hmm/<analysis_version>/<signature>/<timestamp>/results.npz`

Cache discovery inspects only the matching signature directory and selects the newest compatible run containing both `results.npz` and its completion marker. Write that marker only after arrays and metadata are saved; incomplete/error or export-only directories are not numerical cache hits. Timestamps and runtime/log/export paths are excluded from the signature. A rerun creates a new timestamp under the same signature. On a cache hit, keep the numerical NPZ immutable and create another timestamp directory under the same signature for the current `augmented_trials.csv` snapshot and run record referencing that NPZ; do not duplicate the arrays. Never restore a cached dataframe as the current table. Use existing named NPZ/meta conventions, not a new cache service.

The parent calls `write_trial_policy_export` after checked attachment: it writes the current snapshot and run record to the chosen new export directory and, when requested, archives/replaces `current_csv_path`. A fresh numerical run can use its own directory for the first export. Numerical completion means arrays are reusable; later export/plot errors remain explicit workflow failures without invalidating correct saved arrays. Refresh only `trial_policy_` columns, clearing stale candidate/HMM fields; preserve `inherited_block_*`, existing features and choice-window outputs. Never rewrite block CSVs or numerical artifacts.

Implementation clarification: `numerical_path=None` is reserved for a joint CSV export. Its manifest links the individual source NPZs; it does not claim a single joint numerical artifact. Single-source exports always reference their actual numerical path.

### Saved-result plotting and run records

Single-session figures go under `<session.figure_path>/trial_policy_hmm/<timestamp>/`. Multisession PNGs go directly into the existing `cross_session_analysis` folder with mouse/session scope and timestamp in their filenames, matching the established cross-session figure placement. Companion run records live beneath `cross_session_analysis/trial_policy_hmm/<timestamp>/` and link the figure paths.

Record the analysis goal, included sessions, source artifacts, parameters, runtime, warnings/errors, and status in the existing JSON/log/Markdown style. Include actual main/batch source snapshots following the block pipeline's convention: `main.py` currently contains both entry points. Avoid inventing additional executable scripts solely for bookkeeping. Numerical NPZ files remain in original session processed directories; no experimental outputs go into the repository.

Render the five specified panels from loaded artifacts, using stable colors keyed by model ID and actual source block boundaries. Smoothing disabled produces the first three panels. Show all tied winners, distinguish skipped rows, and do not jitter overlapping values. Multisession figures use an ordered concatenated display axis with local identifiers and visible session boundaries; traces break at session boundaries. Use white backgrounds, readable fonts, PNG export, and an explanatory caption. No occupancy statistics or prediction-coverage figure is added.

For the choice-probability panel, plot actual choices at `1 - observed_action`, with `0=Left` and `1=Right` axis labels, so the markers share the `p_right` convention. Keep saved raw action codes unchanged. Do not show unavailable choices as zero/Left.

## 6. Main-workflow integration

Expose flat trial-specific options alongside the current block options in `SingleSessionAnalysisConfig` and the full multisession entry point:

| Proposed option | Default |
|---|---|
| `run_trial_policy_analysis` | `True` |
| `trial_policy_model_ids` | `("forgetting_q_learning", "bayesian_hmm", "win_stay_lose_switch")` |
| `trial_policy_error_prob` | `0.10` |
| `trial_policy_stay_prob` | `0.90` |
| `trial_policy_smoothing` | `True` |
| `rerun_trial_policy_analysis` | `False` |
| `trial_policy_n_jobs` (multisession batch boundary) | Available CPUs, capped at source-session count |

Keep probability-tie tolerance in explicit numerical parameters/metadata with the spec default; it need not become a prominent user-facing workflow setting. Legacy trial state-count/seed/GLM options remain inactive and must not silently control the new model.

### Single session

Add `run_dynamax_trial_analysis` as the normal integration point. Call it after `run_dynamax_block_analysis`, passing the latter's returned current trial table and the effective feature parameters. This order matters: the block runner writes the augmented-trial CSV. The trial stage subsequently attaches its own columns and updates that CSV, preserving block inheritance. Plot from the saved trial artifact and leave existing summary grids intact.

Its main.py contract is `run_dynamax_trial_analysis(trial_df, session, *, feature_params, model_ids=DEFAULT_MODEL_IDS, error_prob=0.10, stay_prob=0.90, smoothing=True, rerun=False, plot=True) -> (current_trial_df, result, numerical_path)`. `session` supplies existing `processed_data_path`, `figure_path`, and `sess_id_full`. Validate identifiers before attachment. This orchestrator performs parent-owned cache lookup, computation if needed, save/current export, and saved-result plotting; low-level computation has no run-control flags.

When trial analysis is disabled, skip its computation and figures. In a full workflow that refreshes current exports, remove only old `trial_policy_` output columns before the final current-table write, so historical assignments are not presented as freshly analyzed results. Existing upfront archival preserves the earlier exports and numerical artifacts. If trial analysis fails, report it clearly; previously completed block outputs remain available and no successful trial artifact is claimed. Clear stale trial-owned columns from the table passed to the block writer before a new trial run, so a failed trial run cannot leave old assignments attached to refreshed data. This is simple column ownership, not a new migration mechanism.

### Multiple sessions

Extend `run_multisession_analysis`/its caller with explicit source-session information and trial options, preserving its existing four-item return structure. `main_multisession` already has the `saved_sessions` list; forward it so each source's paths and parameters are available. Trial outputs populate the returned current trial dataframe; the legacy model-selection placeholders remain unchanged.

Add keyword-only `source_sessions: Sequence[SavedSessionAnalysis] | None = None` and `trial_feature_params_by_session: Mapping[str, TaskParams] | None = None` alongside the options above. If trial analysis is enabled, require source metadata matching the ordered concatenation manifest, including empty sources; otherwise these arguments are unnecessary. Prepare trial-policy history from each original `SavedSessionAnalysis.augmented_trial_df` before concatenation can erase its schema. Retain each source's original flag-column presence/aliases and input digest with its prepared history. Match results by source ID and retained source-row position, never by potentially repeated `cur_trial` alone.

An absent flag column is scientifically different from missing entries in a present column. Normalize legacy/current aliases within each source, not on a heterogeneous concatenated table. A column introduced by another session must not make the originally flag-less session ineligible, nor create a false alias conflict. For joint reattachment, project each source's current joint rows through its recorded original flag schema, restore local block IDs from `source_cur_block`, and recompute its input identity. Introduced alias columns must be missing on that source's rows; unexpected nonmissing values are an identity mismatch, not values to discard. Check original source inputs against the joint view; do not blindly attach stale source results to changed joint actions/rewards/flags. This is source-local preparation and a few schema fields, not a dataframe schema framework.

Here the projection schema comes from the CURRENT un-concatenated source table. Compare it to saved metadata before reusing results; never take an old cache's schema as authority for dropping current columns. This preserves the schema-only invalidation rule during joint reattachment.

Run the existing continuous block stage once. Compute trial results independently per source session, attach by `source_session_id` plus `source_trial_row`, and then update the joint current trial export. Preserve original local trial/block IDs even though concatenation remaps display block IDs. Per-source trial artifacts go to source processed directories; the joint runner need not rewrite source conventional CSVs or source block models. An ordered manifest links the source artifacts to the joint display/export, including empty sources.

For per-source exports made by the joint runner, snapshot only the source-local analysis input columns and new trial outputs; do not copy joint block assignments into source artifacts. Pass `current_csv_path=None` for these exports. The final joint export retains the full current joint table and its block assignments. Single-session exports retain the full current single-session table.

Before the continuous block writer, strip stale `trial_policy_` columns from the JOINT working table too, whether trial inference is enabled or disabled. Leave source tables/files and block columns untouched. On a source trial failure, report that source and propagate the error; do not publish a successful/partial joint trial export or figure. Already saved successful per-source artifacts remain reusable. The joint current CSV remains the fresh block-stage output without stale trial assignments. No rollback, automatic retry, or partial-success aggregation is needed.

Direct callers enabling trial analysis must provide source metadata/effective parameter context rather than guessed paths from a synthetic session ID. Existing callers intentionally doing block-only work can disable the new stage. Update dependent call sites and fixtures accordingly; retain all original block assertions. Collection-only and agreement-only early-return modes must still skip trial inference and trial plotting. Collection-only retains its existing role of copying source tables, including any previously saved source columns, without claiming new trial inference; the full workflow owns stale-column removal. Do not change `main_multisession`'s existing default collection-only setting.

### Scope of missing-data support

The new trial runner supports the documented missing-reward behavior. Some older upstream session-analysis counters reject missing rewards before this runner is reached. Test the new runner on such input, and document this existing full-pipeline limitation. Do not silently expand this task into rewriting all upstream counters or weakening their validation. Normal full-workflow integration tests use valid synthetic upstream data.

## 7. Tests to write before implementation

Use function-based pytest tests and small fixtures, including a hand-checked chronological history, two sessions sharing local trial IDs, an interior excluded row, and small arrays of predetermined candidate scores. Seed any synthetic random generator and retain the seed in synthetic artifact metadata.

| Test file / area | Required assertions |
|---|---|
| `test_trial_policy_models.py` | Known pre-choice FQL/Bayesian/WSLS values; neutral first row; actual mouse-history updates; no true-context dependency; no block reset; cropped initialization; current/future perturbations leave earlier recommendations unchanged; exact action validation; tiny nonzero FQL and positive scaling retain directions. |
| `test_trial_policy_bayesian_validation.py` (new) | Enabled evidence validation raises on impossible/non-finite normalization with row context; supported binary histories retain existing numerical outputs in both value modes; default legacy calls retain their behavior. No separate validation replay. Keep these tests in a new owned file rather than committing the pre-existing untracked `test_trial_features.py`. |
| `test_trial_policy_models.py` missing data | Manual/no-choice/unknown flags freeze histories; valid choice plus missing reward retains its pre-choice score and freezes updates; absent flag-column convention; unavailable predictions remain distinguishable from zero scores. |
| `test_trial_policy_analysis.py` | Exact Bernoulli values for matching/opposite/indifferent choices; common validity mask; independent winner masks and nullable ties; invalid error/stay parameters; candidate permutation invariance. |
| `test_trial_policy_analysis.py` smoothing | Compare real Dynamax output to an enumerated short-path reference with an interior neutral row; identical policies give uniform probabilities; uniform transition rows recover independent probabilities; session isolation; posterior labels preserve ties. |
| `test_trial_policy_analysis.py` shapes | Zero rows, one row, all-invalid rows, one candidate, and a synthetic fourth candidate; smoothing-disabled output contract; no NaNs enter Dynamax. |
| `test_trial_policy_results.py` | NPZ/meta round trip; current-table reattachment; original IDs/order; changed inputs, crop, parameters, model order, source version, or smoothing invalidate cache; unchanged inputs hit cache; rerun preserves old artifacts; stale trial fields removed while unrelated/block fields remain. |
| `test_trial_policy_results.py` reporting | One clear warning per condition/session, counts and short ID previews, complete saved flags, no clean-session warning spam, failed calculation recorded without a success artifact. |
| `test_trial_policy_plotting.py` | Five/three panels, all tied winners, excluded-row cues, stable model colors, correct block/session boundaries, no cross-session connecting lines, white backgrounds/captions, one/four candidate support, saved-result plotting with inference disabled by a test stub. |
| `src/tests/test_main_trial_policy.py` | Real small synthetic single-session runner; full-workflow call/save order preserves inherited block and fresh feature columns; independently computed source sessions equal multisession numerical results; serial/parallel equivalence; correct figure and artifact paths. |
| Main integration modes | Trial disabled and failed runs do not leave stale assignments in refreshed exports; smoothing disabled clears stale HMM fields; collection-only and agreement-only retain their early returns; source-specific parameters; legacy options remain inactive; fresh-process `src.main` import without `ssm`. |
| Feedback regressions | Heterogeneous absent/present/legacy/current flag columns remain source-local; changed rewards or flags with unchanged actions/IDs reject attachment; pre-signature errors reach the defined parent log; `error_prob=1e-20` is rejected; the approved transition range and long evidence reversals have numerical regression coverage. |
| Bound/schema refinements | Exact transition bounds pass; adjacent outside float64 requests fail without tolerance; derived-matrix roundoff does not reject allowed requests. Absent to all-zero flag column, legacy to current alias, and adding/removing a matching second alias change input/cache identity despite identical numerical histories; column reordering alone does not. Check standalone and joint reattachment using the current source schema. |
| Existing regressions | Fixed-exemplar block arrays/exports/plots, main Dynamax migration, trial features, and choice-window integration remain correct. |

Write and commit each phase's tests before its production changes. Verify RED fails for the intended missing behavior rather than an unrelated environment failure. Then implement GREEN and refactor with those tests passing. Do not change expected science to make a test pass. When old workflow fixtures lack action/reward fields because they tested only block behavior, explicitly disable the new stage or add valid trial fixtures; keep their original assertions and add independent tests of the new default.

Record the baseline before editing. Some existing tests use global module stubs and earlier runs identified unrelated failures; run affected tests in clean processes as needed and report baseline failures separately rather than hiding them in broad skips. Use `uv run` for Python/pytest. No tests, code changes, or commits are part of creating this plan.

## 8. Performance and dependencies

Reuse NumPy, pandas, SciPy, JAX/Dynamax, Matplotlib, and the standard library already present in the uv environment. Verify any newly used API against installed source or official documentation during implementation. No new dependency is planned.

Policy replay and likelihood construction cost approximately `O(T*K)`; dense HMM inference is `O(T*K^2)` with small `K`. Store full `(T,K)` outputs as required. Vectorize likelihoods and winner masks; keep existing readable policy loops. Do not add JIT kernels, padding schemes, multiprocessing frameworks, or profiling-driven optimizations without a demonstrated need.

For the new multisession trial batch, follow the existing choice-window numerical-worker/parent-plotting pattern with `ThreadPoolExecutor`: default workers are `min(available CPUs, number of sessions)`, with an explicit positive `n_jobs` override and a serial path. Threads avoid copying a JAX runtime into every process. The sequence is explicit: parent prepares metadata/signatures and loads cache hits; submit only cache misses to pure per-session computation; parent consumes results in source order, saves new artifacts, attaches current outputs, logs, and plots. Do not call a combined persistence runner inside workers or serialize all computation in the parent. Do not nest session pools inside the existing single-session batch runner.

The current full single-session batch is serial and updates shared mouse-level summaries. Do not broadly parallelize that entire legacy workflow as part of this addition. New trial-only multisession numerical work is the session-parallel unit. Test two workers against serial output before enabling the default. If actual memory use or runtime makes the default unsuitable, report the measured constraint and lower concurrency through the existing option rather than adding scheduling machinery.

## 9. Delivery sequence and worker division

Use Sol at medium reasoning for implementation orchestration and review, with at most one active worker at a time. Each worker receives bounded file ownership and the approved contracts. The orchestrator reviews the diff and test evidence before assigning the next phase. Scientific phases use Sol; mechanical phases use Luna with Sol review. This is a proposed execution allocation, not implementation authorization.

Section 0 is the single package-status/assignment ledger; the table below owns file boundaries and acceptance. Sol directly reviews scientific invariants and shared-file integration even when Luna implements a package. Escalate unresolved scientific logic to Sol. No automatic experimental batch run follows test completion.

### Writable paths and package acceptance

All paths below are relative to the repository. Workers may read related modules; they may write only their assigned package files. The receiving lead owns this handoff, Git operations, and shared-file changes outside the assignment. Do not commit pre-existing main/block migration edits or the user's staged specs as part of a tests-only commit. New owned test files allow isolated RED commits; where an existing shared file needs edits, preserve its prior diff and document the new hunks.

| Package | Writable production paths | Tests and acceptance evidence |
|---|---|---|
| WP0 | This handoff only | Record baseline commands/results, actual dirty dependencies, authorization and assigned owner; no production edits. |
| WP1 | `src/behavior_analysis/trial_policy_models.py`, minimal guarded update in `trial_features.py` | New `src/tests/behavior_analysis/test_trial_policy_models.py` and `test_trial_policy_bayesian_validation.py`; causal/missing-history contracts and legacy default parity pass. |
| WP2 | `src/behavior_analysis/trial_policy_analysis.py` | New `src/tests/behavior_analysis/test_trial_policy_analysis.py`; hand calculations, enumerated HMM reference, full-row spacing, ties, K=1/4 cases pass. |
| WP3 | `src/behavior_analysis/trial_policy_results.py` | New `src/tests/behavior_analysis/test_trial_policy_results.py`; exact schema, cache/current-table distinction, preservation and reporting pass. |
| WP4 | `src/behavior_analysis/trial_policy_plotting.py` | New `src/tests/behavior_analysis/test_trial_policy_plotting.py`; automated contracts plus actual PNG review, artifact paths recorded. |
| WP5 | Trial-owned changes in `src/main.py`; only necessary fixes to new modules | New `src/tests/test_main_trial_policy.py`; bounded fixture/call-site updates in existing main tests; all modes/save ordering, source alignment, batch failure and serial/parallel equivalence pass. Sol reviews shared-file hunks before acceptance. |
| WP6 | This handoff and narrowly assigned corrections to owned files | New regression tests precede fixes; record final targeted suites, fresh import evidence, synthetic NPZ/PNG paths and visual review. |
| WP7 | This handoff; outputs only in designated session/figure directories | Record authorized exact session, commands, outputs, user inspection outcome and whether any later batch is authorized. No new science inferred from validation. |

After each test file is written, run it with `UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync pytest -q <actual-test-path>`. Record its meaningful RED and tests-only commit BEFORE editing production. Re-run for GREEN and add the related existing regressions listed in Section 7. The lead reviews the actual diff and artifacts, updates the package ledger, and only then dispatches the next worker. This plan names all files; nonexistent future tests are not baseline commands.

If an implementation interface must change, update its tests and this handoff deliberately, explaining whether the change corrects a genuine contract error or follows a new user decision. Do not weaken tests to bypass a failing requirement. Source may remain uncommitted between gates if that is the current authorization; record exact dirty paths rather than falsely claiming a durable source commit.

## 10. Acceptance and experimental rollout

Round one is complete when the new stage runs through both normal workflows, produces correctly aligned saved arrays and readable figures, preserves existing block/choice-window outputs, passes the focused numerical and integration tests, and supports a synthetic fourth adapter without changes to common comparison/storage/plotting logic.

Run the synthetic end-to-end validation in an ignored project-local test/cache location. Inspect actual generated PNGs, including ties, excluded rows, and two sessions. Then report code changes, tests, known limitations, and output locations. Ask for a user-designated experimental session only when ready for that validation stage; a dataset choice is not needed to implement the code. Run that session, let the user inspect its outputs, and only then run the agreed experimental batch.

Later implementation of the required decay-HMM and observer adapters adds their definitions, effective parameter mappings, and causal/neutrality tests. It must use their raw pre-choice values, preserve the same history contract, and record the expanded candidate set because model probabilities and transition destinations change with `K`.

## 11. Append-only evidence and handoff record

### 2026-11-07 - Planning and independent review; implementation not started

- Authority: user requested a draft plan, then required a live handoff for a different implementing agent and a correctness/completeness/concision review. No implementation authorization was given.
- Reviewed against the finalized scientific spec, SoftwareDesign.md, current policy functions, persistence and main workflow, and the event-sequence/phase-synchrony handoff examples. A separate read-only reviewer checked the implementation plan; no worker wrote source/tests.
- Corrections from review: pinned model IDs/adapters and parameter mappings, public contracts, nullable NPZ/CSV representation (including observed actions for saved-result plotting), parameter precedence, parent cache lookup versus worker computation, stale joint-column removal, failure semantics, and cache-hit export identity. Kept ordinary functions, four modules and one optional validation guard; added no recovery framework.
- Final read-only review found no remaining material scientific/integration contradiction. Follow-up clarifications fixed identifier sources and position-based attachment, early-validation logging, the maximum-assignment helper contract, and raw-action versus plotted-choice encoding. Removed the duplicate delivery-phase table; Section 0 owns live status and Section 9 owns writable files/acceptance.
- Planning files: this handoff and the previously finalized spec must travel with the actual dirty block-migration dependencies. This revision changes only the handoff. No feature tests, source implementation, commits, baseline test runs, experimental analysis or plot artifacts were created.
- Documentation validation: `git diff --no-index --check /dev/null docs/trial_dynamax_hmm_implementation_plan.md` emitted no whitespace diagnostics (exit 1 denotes the new-file difference). No Python tests run; receiving lead must record actual test evidence at WP0 rather than inheriting an assumed clean baseline.
- Running implementation workers/jobs: none. Exact next action: user reviews this revised handoff; after authorization, a different receiving lead starts WP0.

### Entry template for the receiving lead

For each gate, append: date/time; package/gate and authorization; actual owner/model/effort/agent ID; starting HEAD; owned files and prior dirty state; exact commands with exit status and pass/fail counts; RED tests-only commit; GREEN source commit or uncommitted paths; reviewed numerical/figure evidence; findings and unresolved limitations; active process/tool session/PID/log/output paths (or none); last durable artifact; exact next action. Update Section 0 to match. Do not overwrite earlier failure or review records.

### Implementation-chat feedback review - pre-approval revision

- Corrected source-schema preservation, metadata-backed attachment checks, concrete pre-signature run-log paths, and representable error-probability validation. Added corresponding test requirements.
- Reproduced the reported smoother failure with small in-memory probes and inspected installed Dynamax source. The two-filter alternative also fails a longer reversal; it is not an accepted replacement. Transition-range choice and exact input parsing are pending user response.
- No production code, tests, or installed packages changed. An initial uv inspection failed on its default read-only cache; rerunning with `UV_CACHE_DIR=.cache/uv` succeeded. An explicit `compute_trans_probs=False` probe hit the installed JIT wrapper's traced-boolean error; successful probes used the default call, matching the existing project helper.
- Per user instruction, the Section 0 checkout/status snapshot is deliberately NOT refreshed during plan revision. Refresh it last, after plan approval and the user's Git push, before implementation handoff. Its historical HEAD/dirty-state observations must not be treated as current checkout facts in the meantime.

### Approved transition bound and input-format inspection

- User approved the bounded transition range and asked that parsing follow actual inputs rather than a preference choice. The approval closes the transition-range question above; the earlier pending entry is historical.
- Read-only CSV field inspection covered 14 CT014 source sessions (6,799 rows: seven legacy `give_reward` files, seven current-flag files) and 30 CT026 source sessions (20,145 rows, current flag). Paths were `<data-root>/<mouse>/*/processed/*_augmented_trials.csv`, with data root `/home/matt/Documents/EXPERIMENTS/contextProjectData`. Observed actions were `0`, `1`, `None`, `no_choice`; flags and rewards were `0`, `1`, and, in CT026, `None`. No textual sides or boolean strings occurred.
- Also inspected `data/processed/CT001/CT001_2024-08-29_choices.csv`, `data/processed/HD005/HD005_2024-09-07_choices.csv`, and two stored HMM simulation CSVs under `data/processed/sample_agents`: numeric action/reward codes and no manual-flag column. These support the existing absent-column convention.
- Producer `process_behavior_log.choice_event_summary` emits integer action/reward/flag codes and `no_choice`; normal main.py loaders use `na_filter=False`, preserving string missing sentinels. Add fixtures for observed representations, both flag aliases, absent flags, and pandas numeric/missing equivalents; do not add speculative textual aliases.
- Inspection used stdlib CSV counters through `UV_CACHE_DIR=.cache/uv uv run --no-sync python -B`; no model computation, experimental analysis outputs, implementation edits, or tests were run. Only plan/spec decisions were updated. The handoff snapshot remains deferred until approval/push as requested.

### Transition-roundoff and schema-identity review

- Made inclusive requested stay bounds authoritative with no tolerance; specified a derived-matrix-only absolute allowance of `1e-12`. Added exact-bound and immediately-outside tests. Specified float32 posterior/reference test tolerance `atol=2e-6, rtol=0`, separate from scientific tie labeling.
- Made the original per-source flag-column presence/alias list part of `input_digest` and hence cache identity. Added schema-only invalidation tests and a column-order invariance test; cached metadata must not override the current source schema during joint reattachment.
- External implementation-chat review supplied by the user reports finite results for 121-row evidence reversals at both bounds and default .90, K=2,3,4, with maximum probability error approximately `7.9e-7` versus a stable log-domain reference. This is reported review evidence, not a test suite run or reproduction by this planning chat. The implementing agent must add and run the regressions before WP2 acceptance.
- Documentation-only revision. Section 0 remains unchanged until the requested final snapshot refresh after approval/push.

### 2026-11-07 - WP0 authorized implementation baseline

- Authority: user confirmed plan pushed and instructed implementation. Starting HEAD `9ffe30e859b95e5fdeb49b90892abd05154b5ea7`, branch `refactor`; index clean. Lead `/root`; one approved worker `/root/wp1_adapters` (gpt-6.1-sol high) owns WP1 test files, then its two production files after RED commit confirmation. No experimental data processing authorized.
- Existing dirty tracked paths: `src/main.py` (532 insertions, 192 deletions), `src/neural_analysis/sync_ephys.py` (one insertion). Relevant untracked dependencies: `exemplar_hmm_results.py`, `exemplar_hmm_plotting.py`, `expectant_switching_selectivity.py`, and existing trial-feature/main simulation tests; numerous unrelated untracked artifacts remain untouched. Starting main/source diff preserved under ignored `.cache/trial_policy_implementation/`.
- Baseline command: `UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync pytest -q src/tests/behavior_analysis/test_trial_features.py src/tests/behavior_analysis/test_fixed_exemplar_block_states.py src/tests/behavior_analysis/test_exemplar_hmm_results.py src/tests/behavior_analysis/test_exemplar_hmm_plotting.py` -> exit 0, 101 passed, one existing TFP/JAX deprecation warning, 3.99 s.
- Baseline command: `UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync pytest -q src/tests/test_main_dynamax_migration.py src/tests/test_main_choice_windows.py src/tests/test_main_multisession.py` -> exit 0, 89 passed, same deprecation warning, 7.27 s. Separate processes avoid existing global-stub contamination.
- Baseline tool sessions 38419 and 46734 completed. No active analysis process. No source edits or commits yet. Next gate: review WP1 RED tests and make the tests-only commit before implementation.

### 2026-11-07 - WP1 RED and tests-only commit

- Owner `/root/wp1_adapters`, gpt-6.1-sol high; lead reviewed hand calculations, exclusion/history and selection contracts. Worker wrote only the two new owned tests. No production edits before commit.
- `UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync pytest -q src/tests/behavior_analysis/test_trial_policy_models.py src/tests/behavior_analysis/test_trial_policy_bayesian_validation.py` -> exit 1, 89 failed in 1.55 s. Collection succeeded; failures are missing trial-policy module (76 cases) and missing opt-in Bayesian keyword (13 cases).
- RED tests-only commit `2360445` contains exactly those two files (364 lines). Lead authorized WP1 GREEN after commit. Root drafted the next phase's numerical tests in its new owned file, unrun and uncommitted pending the WP1 dependency.
- Next action: worker implements source-local replay and the guarded Bayesian validation; no experimental runs.

### 2026-11-07 - WP1 first GREEN review and WP2 RED preparation

- WP1 owned tests: 89 passed in 0.98 s; three expected warnings from deliberately nonfinite direct legacy validation probes. Existing `test_trial_features.py test_gather_trial_features.py`: 55 passed in 1.00 s. Root reviewed actual source and requested one small regression: unexpected nonfinite first-round computed values on usable rows must raise rather than become unavailable. Worker is adding that test before its fix; WP1 not yet accepted.
- Root-authored `test_trial_policy_analysis.py` uses hand Bernoulli values, enumerated short HMM paths, a stable log-domain long-sequence reference, excluded-row spacing, ties, K=1-4 and bounded transitions. Exact standard environment command with that path -> exit 1, 26 failed in 0.38 s, all on missing `trial_policy_analysis` import inside collected tests. WP1 history fixtures already prepare successfully. Tests-only numerical commit recorded by lead; no numerical production file yet.
- Root also drafted WP3 tests in its new owned test file; unrun/uncommitted pending numerical implementation. No experimental outputs.

### 2026-11-07 - WP1 calculation-error regression RED

- Added two cases proving NaN/inf FQL calculations on a usable row must raise with session/original trial/model context. Standard command with `test_trial_policy_models.py -k unexpected_nonfinite` -> exit 1, 2 failed, 76 deselected, 0.93 s, intended DID NOT RAISE.
- Tests-only correction commit `a1cb550`; worker authorized to fix afterward. WP2 tests-only RED commit is `5e6447a`. No unrelated files staged.

### 2026-11-07 - WP1 accepted; WP2 GREEN

- WP1 final command: standard environment `pytest -q` with owned model/Bayesian tests and existing trial-feature/gather tests -> exit 0, 146 passed, three expected deliberate nonfinite legacy-probe warnings, 1.15 s. Reviewed actual source and accepted; source commit `16b8a42`. First-round unexpected calculation failures now raise, while explicit synthetic-adapter unavailability remains supported.
- WP2 owner lead `/root`. Tests-only RED `5e6447a` preceded production. Standard environment `pytest -q src/tests/behavior_analysis/test_trial_policy_analysis.py` -> exit 0, 26 passed, one existing TFP/JAX warning, 2.19 s. Real Dynamax matches enumerated/long log-domain references, preserves full row spacing and ties, supports K=1-4, accepts exact bounds and rejects adjacent outside values. Numerical source committed after GREEN.
- `git diff --check` reports only the pre-existing unrelated `sync_ephys.py:243` blank EOF; trial-owned paths clean. No experimental outputs. Next: WP3 RED tests and owned persistence implementation.

### 2026-11-07 - WP3 initial GREEN and review corrections

- WP3 tests-only RED: standard environment `pytest -q src/tests/behavior_analysis/test_trial_policy_results.py --tb=short` -> 25 failed, missing module, 0.97 s; committed `fd310d1`. Worker `/root/wp3_results`, gpt-6-luna medium, wrote only its assigned new results module. Initial GREEN: 25 passed.
- Root source review identified real omissions: code digest joined behavior filenames under `src/` and hashed only main; HMM per-model column names lost their prefix; greedy -1 stayed numeric in CSV attachment; NPZ meta was JSON scalar rather than required dictionary. Four failing assertions committed before correction as `96c56d2`: 4 failed, 23 passed, 2.20 s. Worker correcting these plus function-contract/type-hint quality; no numerical science changed.
- WP4 tests drafted and meaningful RED verified: standard environment `pytest -q src/tests/behavior_analysis/test_trial_policy_plotting.py --tb=short` -> 9 failed on missing plot module, 0.35 s. No plotting production code yet. WP5 tests drafted; RED command running separately. No experimental data processed.

### 2026-11-07 - WP3 accepted and remaining RED gates

- WP3 correction GREEN: 27 passed after `96c56d2`; source accepted/committed `82bc108`. Review also found the unknown-flag warning mask used bitwise AND with alias count, suppressing warnings when both aliases existed. Regression RED: 1 failed/27 deselected, 0.29 s; tests commit `acc8b5f` preceded simple mask correction. Final persistence suite: 28 passed, one existing warning, 2.15 s; fix `cd5ef8f`.
- WP4 tests-only commit `fa496e5`, nine meaningful missing-module failures; worker `/root/wp4_plotting` (gpt-6-luna medium) owns only plotting source. WP5 tests-only `3575d6b`: standard command `pytest -q src/tests/test_main_trial_policy.py --tb=short` -> nine intended failures (missing runner/config/default trial behavior), 2.57 s. Main source still untouched since starting snapshot.
- Existing block-only window fixture explicitly disables trial analysis (`acc8b5f`), retaining original assertions. Analogous one-line fixture update to tracked `test_main_dynamax_migration.py` pending stage; prior version preserved in ignored starting snapshot. No experimental analysis or outputs.

### 2026-11-07 - WP4 GREEN and WP5 integration

- WP4 worker `/root/wp4_plotting` (gpt-6-luna medium) wrote only `trial_policy_plotting.py`. Original capture fixture wrapped `Figure.savefig` before pyplot initialized; Matplotlib rejected its local qualname. Genuine fixture correction `d45e2ec` imports pyplot before wrapping; no assertions weakened and no production compatibility shim retained. Nine plotting cases then passed; source accepted as `9753c34`. Root inspected real two-source PNG under `.cache/trial_policy_implementation/wp4_review/`.
- WP5 lead `/root` implemented shared main changes directly against the preserved dirty starting main, keeping the feature delta separate. Added config/default runner, source-specific parameter resolution and checked joint projection, parent-only cache/I/O, pure numerical thread workers, full-workflow stale-column clearing, and saved-artifact figures. Return structure and collection/agreement early returns remain intact.
- Additional source-parameter/empty-source tests RED: two failures, nine deselected, 2.51 s; tests-only `0537290`. Complete new main suite GREEN: 11 passed, 7.43 s. After replacing an opaque work tuple with a small data-only source record, 11 passed, 7.60 s. No source commit for main: its pre-existing 532-insertion/192-deletion migration remains intertwined and must not be committed incidentally.
- Reporting review regression RED: nine failures/40 deselected, 4.69 s, covering actual joint-source artifact links, options retained on early failure, and excluded-row caption meaning. Tests-only `1d1f0b5` preceded fixes. `write_trial_policy_export` now accepts None for a joint export, preserving source NPZ links without mislabeling a JSON manifest as numerical data. Targeted results/plot/main GREEN: 49 passed, 10.60 s; source correction `c66fcae`.
- Existing main/window suites passed 89 tests, 7.06 s; combined numerical/feature/block suites passed 266, 8.14 s at this gate. Warnings are existing TFP/JAX deprecation and three deliberate nonfinite legacy guard probes, not normal-input warnings.

### 2026-11-07 - WP6 independent audit, final tests and synthetic artifacts

- Independent read-only owner `/root/wp6_audit`, gpt-6.1-sol high, reviewed spec, all new modules, Bayesian guard and feature-only main delta. Independently ran 117 model/numerical/Bayesian cases. Found no material scientific/integration defect; two real plotting issues remained: normal 30-source caption overlap and HMM interpretation printed when smoothing was disabled.
- Plot regression RED: four failures/six passed, 5.17 s. Test assertion previously requiring retrospective wording even without smoothing was a genuine test error; corrected it to require disabled-smoothing wording. Added real 30-source rendered caption-bbox regression. Tests-only `8e24b2f` preceded conditional caption and measured-height layout fix. GREEN: ten passed, 5.66 s; source `8c41826`.
- Final combined command: `UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync pytest -q src/tests/behavior_analysis/test_trial_policy_models.py src/tests/behavior_analysis/test_trial_policy_bayesian_validation.py src/tests/behavior_analysis/test_trial_policy_analysis.py src/tests/behavior_analysis/test_trial_policy_results.py src/tests/behavior_analysis/test_trial_policy_plotting.py src/tests/behavior_analysis/test_trial_features.py src/tests/behavior_analysis/test_gather_trial_features.py src/tests/behavior_analysis/test_fixed_exemplar_block_states.py src/tests/behavior_analysis/test_exemplar_hmm_results.py src/tests/behavior_analysis/test_exemplar_hmm_plotting.py --tb=short` -> exit 0, 268 passed, four known/intentional warnings, 10.27 s.
- Final new-main command: same environment `pytest -q src/tests/test_main_trial_policy.py --tb=short` -> exit 0, 11 passed, one existing deprecation warning, 8.42 s.
- The final legacy-main run exposed a genuine dynamic import-fixture omission after the new data-only record used forward annotations: 71 failed, six passed, 12 errors because the fixture executed a module without registering it in `sys.modules`. Verified Python's installed importlib source registers modules before execution; corrected that single fixture operation without changing assertions or adding a production workaround. Correction commit `de94775`. Final exact legacy command: same environment `pytest -q src/tests/test_main_dynamax_migration.py src/tests/test_main_choice_windows.py src/tests/test_main_multisession.py --tb=short` -> exit 0, 89 passed, one existing warning, 7.23 s. Final total: 368 passed across these three suites.
- Fresh standalone command: same uv environment `python -B` importing `src.main` and asserting `ssm` absent from `sys.modules` -> exit 0. Trial-owned `git diff --check` passed. No full unrelated repository test sweep or experimental data processing was claimed.
- Deterministic synthetic driver: `UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync python .cache/trial_policy_implementation/validate_synthetic.py` -> exit 0. Review directory `.cache/trial_policy_implementation/synthetic_2026-11-07_23-16-31/` contains exact driver copy, `summary.md`, `validation_manifest.json`, source-local versioned NPZs and current CSVs, source/joint run logs and actual main/batch snapshots. No RNG used; metadata records seed None.
- The synthetic run exercised real block/trial inference, two sessions sharing local IDs, no-choice/missing reward/manual/unknown-flag rows, absent/legacy source schemas, immutable cache reattachment, threaded source inference, and a fourth neutral adapter. Joint marginals exactly matched independently computed source marginals. Missing-data warnings appeared once per condition/source per invocation.
- Final saved-result-only re-rendering produced `final_figures/SYNTH_A.png`, `SYNTH_B.png`, `synthetic_fourth.png`, `multisession.png`, and `thirty_source_layout.png` in that review directory. Root visually inspected final two-source and 30-source PNGs: legible caption separated from axes; exclusions/ties and reset boundaries visible; no cross-source trace connections. All numerical artifacts remain in source processed subdirectories.
- Main implementation is a 402-insertion/two-deletion feature delta against `.cache/trial_policy_implementation/starting_main.py`; exact isolated diff is `.cache/trial_policy_implementation/trial_policy_main_changes.patch`. The working `src/main.py` must accompany this branch/handoff because its pre-existing migration and new integration are deliberately uncommitted. Original `sync_ephys.py` change and unrelated untracked files were not modified.
- Interruption: unexpected staged `docs/SAC_behavior_prep.md` entered `de94775` together with the fixture fix. Lead stopped; user explicitly replied "ignore that document and keep going". Commit and current document edits are left intact; no history rewrite performed. Future commits use explicit path ownership.
- All test/tool sessions and workers complete; no active process/log tail to resume. WP0-WP6 accepted. Exact next action: user selects the full ID/path of one experimental session for WP7 validation, then inspects its outputs before any batch. No session or batch authorization is inferred from implementation completion.

### 2026-11-07 - WP7 CT026 August 28 experimental validation

- Authority: user "you can try CT026 8/28". Resolved the existing session uniquely to `CT026_2026-08-28_134630` under `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260828_latent_inference/`. Lead `/root`; no worker delegation, upstream recomputation, fitting, code edits or experimental batch.
- Read the existing augmented-trial CSV with `na_filter=False`; explicit parameters came from that source's `processed/trial_feature_params.json` (SHA-256 `961411b66878ef1f91de681732678793d375a70da185ab49aabd47a76eb884c7`). Effective first-round parameters match approved defaults: FQL decay .7/update .3, Bayesian switch .2/active .8/inactive 0/reward sizes 1 and 0/tanh scale 1; error .10, trial stay .90, smoothing enabled. Retained original IDs run from 0 through 697; initialization is first retained row.
- Preflight found 698 retained rows, 695 eligible comparisons/updates, no existing `trial_policy_` columns. Unknown manual flags on original trials 136, 436 and 578 exclude scoring and history updates while retaining all HMM time steps. No missing-reward continuation warning on this session.
- Exact driver: `UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib MPLBACKEND=Agg uv run --frozen --no-sync python -B .cache/trial_policy_implementation/run_ct026_2026_08_28.py` -> exit 0. Explicitly authorized session-directory writes ran through sandbox escalation. JAX used CPU because CUDA-enabled jaxlib is absent; no dependency was changed.
- Numerical artifact: `processed/trial_policy_hmm/trial_policy_hmm_v1/350250c87646836d339a54d6e13583d0a248cf09cbe200d2c8ea5166792342b8/2026-11-08_04-27-34_445475_UTC/results.npz` beneath that session directory. Status `ok`, likelihood/posterior shape (698,3), finite normalized smoothed probabilities, common excluded likelihood rows unavailable, no independent winners on excluded rows, and saved-array round trip verified.
- Current augmented CSV refreshed with only new trial-owned columns. Original column values/order/index were checked unchanged in memory. Original CSV bytes were verified in `processed/trial_policy_hmm/archives/33a0fc8628bb56df422d4ec78217248b745fbe681dd61641a9f2936d9ae8b719/CT026_2026-08-28_134630_augmented_trials.csv`. Block-performance and saved parameter file byte hashes are unchanged. Existing upstream/block outputs were retained.
- Figure and human review directory: `figures/trial_policy_hmm/2026-11-08_04-27-33_791289_UTC/` containing `trial_policy_diagnostic.png`, `validation_report.json`, `validation_summary.md`, `validation_driver.py`, and actual `main.py`/`batch.py` snapshots. Root visually inspected the real 698-row five-panel PNG: dense expected overlapping greedy traces; preserved tied winners; block boundaries and excluded-row cues; readable caption. Parent log/metadata: `processed/trial_policy_hmm/runs/2026-11-08_04-27-33_791289_UTC/`.
- Scientific identifiability finding: saved Bayesian-HMM and WSLS p_right columns are identical on all 695 eligible comparisons, with zero disagreements. Maximum smoothed probability difference is 1.4901161193847656e-7 (below tie tolerance). Independent comparisons contain 603 tied rows and smoothed assignments 522 tied rows; labels preserve them. This configuration cannot distinguish those two policies on this session; each still receives a separate equal prior share. Recorded in the validation report/summary; no candidate was silently removed or retuned.
- All tool sessions (36435 preflight, 47409 execution, 58841 identifiability check, 30331 report update) complete. Existing unrelated source/document edits remain untouched. No tests rerun because no production code changed. Exact next gate is user inspection of this session's output and the policy-identifiability issue; no later session/batch authorization inferred.
