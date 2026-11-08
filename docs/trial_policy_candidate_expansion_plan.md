# Trial-policy candidate expansion plan

Status: implemented and validated on 2026-11-08 after user approval. This extends the completed
first round in `trial_dynamax_hmm_implementation_plan.md` and follows the causal
history and probability contracts in `trial_dynamax_hmm_spec.md`.

## Scope and decisions

Add standard Q-learning, reward-decay HMM log odds, and the existing mouse-history
ideal observer to the trial-policy comparison. Reuse the existing learning rules,
likelihood calculation, Dynamax smoothing, and source-session workflow. There is
no parameter fitting or simulated agent history in this extension.

The user approved all six candidates as the default, retaining explicit subsets.
Keep the original three in their existing order and append the additions.

The user also requires a general tie display whose size does not depend on the
number of possible tied-model combinations. The proposed layout below uses one
row per model, not one category per observed or possible combination. The user
subsequently approved implementation of the complete expansion.

The archive/discovery prerequisite is implemented in the working tree. Tests
were written, confirmed failing, and committed first (`f5ca3fe`). Discovery now
checks only `*/processed/<session>_augmented_trials.csv` below the mouse root.
Future archives use `.archived.csv`, and trial-policy snapshots use
`trial_policy_export.csv`. Existing archives remain intact and count toward
content deduplication. All 139 affected workflow/persistence tests passed; a
read-only check resolves CT026 2026-08-28 correctly with its old archive present.
This fix does not implement the candidate expansion.

## Candidate definitions

All scores describe the state before the current trial outcome. Actions remain
`0=Right, 1=Left`; a positive score favors Left. Scores are dimensionless because
rewards are binary delivery indicators, not water volumes.

| Stable ID | Existing computation | Effective parameters |
|---|---|---|
| `q_learning` | `trial_features.qlearning_relative_value`: `Q_left - Q_right` | `QL_learning_rate` (currently 0.3), `n_actions=2` |
| `reward_decay_hmm` | `trial_features.hmm_relative_value_reward_decay`: `tanh(tanh_scale * B_prior)` | Existing Bayesian task-belief parameters plus `hmm_reward_decay_lambda` (currently 0.1) |
| `ideal_observer` | Reward-decay HMM score minus raw relative doubt | Same decay-HMM parameters plus `omission_lam` (currently 0.5) |

Standard Q-learning updates only the chosen action:
`Q[action] += learning_rate * (reward - Q[action])`. Its unchosen action does
not undergo the forgetting used by the existing FQL candidate.

The decay HMM updates from rewards using Bayes. On omissions it uses
`B_post = (1 - lambda_decay) * B_prior`; it then applies the task-context hazard
transition. That task-context transition is separate from the outer HMM's
transition between candidate behavioral policies.

For the ideal observer, maintain pre-choice right/left counterfactual omission
counts, initialized to zero. An omission increments the chosen side; a reward
clears both counts. Read out
`doubt = (1 - exp(-omission_lam * left_count)) -
(1 - exp(-omission_lam * right_count))` before updating those counts. Subtract
this raw doubt from the pre-choice decay-HMM score, without residualization,
normalization, or an additional tanh.

"Ideal observer" retains the repository's existing name for this composite
policy; this extension does not establish that it is an optimal policy.

Use explicit `TaskParams`, then the existing saved-session parameter/default
resolution in normal workflows. Do not substitute generic function defaults
(Q-learning 0.1 and decay HMM 0.2 differ from `TaskParams`). Preserve each source's
parameter provenance and defaulted-field warnings.

## Architecture and localized changes

### 1. Existing feature arithmetic: `trial_features.py`

Keep the existing decay-HMM replay and public return contract. Address symmetric
beliefs inside that replay: explicitly equal right/left belief components must
produce zero log odds, including after epsilon normalization. Use the belief
components to identify equality; do not impose a near-zero threshold on scores.
Retain existing non-neutral calculations unless a test demonstrates a necessary
correction. This fixes the real case where omission-only history drifts into a
tiny rightward score that would otherwise become a 90/10 prediction.

Add an opt-in outcome-evidence check for the decay HMM, analogous to the existing
full Bayesian check. Validate positive finite normalization mass at its existing
reward update and report retained row, observation, and parameters on failure.
Do not perform a second replay to validate it. Preserve legacy behavior with
validation disabled, apart from the explicitly tested neutral-belief correction.

### 2. Candidate adapters: `trial_policy_models.py`

Add ordinary adapter functions and three candidate records. Use explicit stable
IDs in adapter/parameter lookup; do not make their identity depend on positions
in the default tuple. Preserve original IDs and relative ordering.

The Q-learning adapter calls the existing feature function. A small shared
decay-HMM helper calls the existing feature with outcome validation enabled.
Both the standalone decay candidate and the ideal observer use this helper.
Replaying the small decay component twice when both are selected is acceptable;
do not introduce a dependency graph or component cache for this extension.

The observer replays doubt with the existing
`CounterfactualDoubtState`, `update_counterfactual_doubt`, and raw doubt readout
in `behavior_modeling/counterfactual_doubt.py`. A short chronological loop reads
the pre-choice value, then updates only on `history_update_valid` rows. It uses
the prepared history rather than the whole session table. Do not call the
high-level observer helper on historical CSVs: that helper may reuse stale
features and derive parameters from task columns.

All adapters return aligned float64 scores and boolean prediction availability,
each `(T,)`. Manual, no-choice, and unknown-flag rows skip prediction and freeze
history. A usable choice with missing reward still receives a pre-choice score
but freezes both belief and omission state. Crops and separate sessions initialize
at their first retained row. True context and block boundaries do not drive or
reset policy state.

Validate only the selected candidates' effective parameters: Q-learning rate
and decay fraction in `[0, 1]`, positive finite doubt rate, and the existing
two-action/Bayesian parameter contracts. Genuine score zero maps to indifference;
unexpected nonfinite computation raises an error with session/trial/model context.

### 3. Provenance and integration

In `trial_policy_results.py`, record each candidate's indifference rule rather
than claiming a single global exact-zero rule without explaining the decay
belief correction. Include the shared counterfactual-doubt source in the code
digest. Use a new analysis version for the changed numerical/provenance behavior,
preserving existing artifacts and their loader compatibility.

Extend effective parameter metadata and existing source-local parameter checks.
Keep `(T, K)` likelihood/posterior arrays and the existing result schema. Model
selection, ordering, and parameters remain part of cache identity. No likelihood
or Dynamax rewrite is expected; test the existing code with six candidates.

Propagate the approved default candidate set through existing single-session and
multisession options in `main.py`; retain explicit subsets and smoothing-off runs.
Do not rerun upstream feature collection or block analysis solely to add candidates.

Equal prior mass remains per selected candidate. On CT026 8/28 the full Bayesian
and WSLS candidates made identical predictions on all eligible rows. Adding
candidates changes their individual probabilities; do not silently merge models,
drop candidates, or reinterpret individual posterior probability as policy-family
support. Keep the configured outer stay probability unchanged and save the
resulting six-state matrix.

### 4. General tie display

Use explicit, distinct colors and short labels for the new IDs. Replace the
winner-marker panels with a trial-by-model membership grid: horizontal position
is trial, and each model has exactly one labeled row. Fill a cell in that
model's color whenever it attains the maximum. One filled cell in a trial column
means a unique winner; multiple filled cells mean a tie; all rows filled means
an all-model tie. Unselected cells are white, and unavailable comparisons are
gray. No combination-specific category, color, or legend entry is needed.

Render contiguous cells rather than overlapping point markers so long sessions
remain readable. The display height grows only with the number of candidates.
Apply the same layout separately to independent likelihood winners and
retrospective smoothed-marginal winners. Retain the existing probability traces
and caption explaining how to read ties and excluded trials.

Plot only saved arrays, retain source boundaries and the existing light PNG
style, and inspect a full-length example. Keep existing three-candidate artifacts
plottable. This change affects display only; preserve the existing winner masks,
probability tolerances, and nullable state labels.

## Tests written and committed before implementation

Use pytest through `uv run`; confirm meaningful failures, then commit tests before
production edits. Change existing expected defaults/names only where the approved
requirements change. Add tests to the existing suites rather than a new framework.

1. Hand-calculated Q-learning pre-choice values, chosen-only updates, and explicit
   `TaskParams` rate; distinguish its recommendations from FQL on a short history.
2. Decay-HMM reward updates, omission log-odds decay, and subsequent hazard update;
   neutral omission-only history remains exactly indifferent. Include full decay
   (`lambda=1`), symmetric hazard, and mirrored left/right histories.
3. Impossible rewarded observations raise from the existing update with row and
   parameter context; validation remains opt-in for legacy callers. Valid
   non-neutral feature values retain their previous numerical behavior.
4. Hand-calculated observer omission counts and signed scores, including reward
   reset of both counters. Verify score equals decay component minus raw doubt,
   and that doubt can reverse a recommendation. Compare ordinary complete-history
   output with existing feature composition, allowing only the neutral correction.
5. Parameterized new-adapter causality, cropped/session initialization, excluded
   row freeze, missing-reward prediction/freeze, and immunity to stale features,
   future observations, and true task-context columns.
6. Six-candidate and selected-subset likelihoods, tie masks, smoothing, and candidate
   permutation consistency. Test genuine indifference separately from missingness;
   preserve the existing tiny nonzero FQL direction test.
7. NPZ round trip, ordered candidate IDs, effective parameters, per-candidate
   indifference provenance, cache invalidation for changed relevant parameters,
   and safe loading of original three-candidate artifacts.
8. Single/multisession integration with the approved defaults and explicit subsets;
   per-source parameters and histories remain independent and unrelated table
   columns survive attachment. Confirm cache reuse avoids numerical recomputation.
9. Six-candidate plot labels/colors and full tie masks; verify membership cells
   for unique, partial-tie, all-tie, and excluded trials from saved arrays,
   including source boundaries. Assert one row per candidate regardless of the
   number of tie combinations. Render three- and six-candidate examples.

Run the focused suites first, then the existing trial-feature, observer,
trial-policy, and main workflow regressions affected by these changes.

## Dependencies, performance, and validation

No new packages. Reuse NumPy/SciPy, existing counter helpers, and installed
Dynamax through the current smoother. Adapter replay is linear in retained
trials per candidate; six-state smoothing uses the existing implementation.
Retain current session-level parallelism and avoid nested workers or optimization
work without evidence of a bottleneck.

After unit tests, run a small synthetic full-pipeline example where the new
candidates disagree, plus neutral and skipped-row examples. Then rerun only
the user-designated CT026 2026-08-28 session in a new versioned output location,
preserving previous results. Check eligible-row counts, posterior normalization,
pairwise prediction agreement, and figure readability. Save the parameters,
version, scripts, log, and a short scientific summary using the existing run
record machinery. User inspection precedes any broader experimental batch.

## Completion and validation handoff

Implementation is complete in `trial_policy_models.py`, `trial_features.py`,
`trial_policy_results.py`, and `trial_policy_plotting.py`. Existing main workflow
defaults inherit the expanded `DEFAULT_MODEL_IDS`; explicit subsets still work.
No likelihood or smoother implementation changed, and no dependencies were added.

Tests were written and confirmed failing before production edits, then committed
as `4a8a0ff`. The final affected checks passed across separate pytest processes:

- 288 policy, feature, observer, persistence, and plotting tests.
- 92 existing main workflow tests (including the corrected archive-name expectation).
- 13 trial-policy single/multisession workflow tests.
- Total: 393 passing tests. Existing warnings are the JAX/TFP deprecation and
  intentionally invalid-input probes in the original Bayesian guard tests.

The synthetic pipeline used a deterministic 140-trial history with manual,
no-choice, unknown-flag, and missing-reward examples: 137 eligible comparisons,
136 history updates, and 70 eligible rows where candidate predictions differed.
The saved PNG was inspected for legible labels, membership grids, and caption.
Its reproducible driver is `.cache/trial_policy_implementation/validate_expanded_policies.py`.

The only experimental run was the user-designated CT026 2026-08-28 session:

- 698 retained trials; 695 eligible comparisons and history updates. Unknown
  manual flags on original trial IDs 136, 436, and 578 freeze replay and exclude
  comparison evidence while retaining all HMM positions.
- Candidate predictions differed on 293 eligible trials. Independent likelihood
  maxima were tied on 694 of 695 eligible trials; smoothed marginal maxima were
  tied on 405 of all 698 retained rows. These have different denominators because
  HMM assignments also exist on excluded rows.
- Full Bayesian HMM and WSLS predictions still agree on every eligible trial.
  Their separate equal-prior shares remain visible. This analysis cannot identify
  one of these two policies over the other on this history with these parameters.
- All posterior rows are finite and normalize within the established tolerance.
  NPZ round trip, non-policy table columns, prior numerical artifacts, block CSV,
  and saved trial-feature parameters were checked. The prior current CSV was
  preserved before refresh. No upstream analysis or experimental batch was run.
- The full-length PNG was inspected. Rows 3 and 5 have six fixed labeled model
  rows; ties use multiple colored cells in the same trial column. They require
  no additional rows or legend entries for new combinations.

Experimental figure and report directory:

`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260828_latent_inference/figures/trial_policy_hmm/2026-11-08_05-23-56_881415_UTC/`

It contains `trial_policy_diagnostic.png`, `validation_report.json`,
`validation_summary.md`, `validation_driver.py`, `main.py`, `batch.py`, and
`run.log`. The report records exact effective parameters, parameter-file source,
pairwise prediction agreement, preservation checks, and artifact links.

Numerical artifact:

`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260828_latent_inference/processed/trial_policy_hmm/trial_policy_hmm_v2/770e7032d6c3bbb8bcc5d9ed2176e2842446444873e551967f3f7ecffa866192/2026-11-08_05-23-57_589877_UTC/results.npz`

Existing three-candidate v1 artifacts remain loadable and were not overwritten.
Main integration and the block-stage modules retain their existing worktree
changes; unrelated user staging was left untouched. User inspection of the new
figure remains the gate before any broader experimental analysis.
