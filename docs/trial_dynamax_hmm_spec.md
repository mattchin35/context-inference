# Codex specification: trialwise comparison of fixed behavioral policies

## Goal and scope

Extend the **existing block-level fixed-exemplar/Dynamax analysis** to operate on individual trials.

Each candidate model uses the mouse’s actual preceding choices and rewards to recommend a greedy Left or Right choice. Convert that recommendation into a probabilistic policy using a common error rate, calculate the likelihood of the mouse’s observed choice, and optionally smooth the model assignments using the existing Dynamax infrastructure.

**Approved first-round candidates, in fixed order:**

| Stable model ID | Existing pre-choice feature | Existing implementation |
|---|---|---|
| `forgetting_q_learning` | `FQlearning_rel_value` | `trial_features.forgetting_qlearning_relative_value` |
| `bayesian_hmm` | `HMM_rel_value_logodds` | `trial_features.hmm_relative_value`, with `value_mode="bayesian_log_odds"` |
| `win_stay_lose_switch` | `wsls_regressor` | `trial_features.win_stay_lose_switch_regressor` |

The Bayesian candidate updates beliefs from both rewards and omissions. Its internal two-context belief model is distinct from the outer HMM over behavioral policies.

**Required subsequent round:** make the existing reward-decay HMM (`HMM_rel_value_logodds_decay`) and mouse-history ideal observer (`observer_value`, reward-decay HMM minus relative doubt) available as additional selectable candidates. They are not first-round deliverables, but adding them must not require rewriting likelihood calculation, smoothing, persistence, or plotting.

**Subsequent-round status (2026-11-08):** implemented, together with standard
Q-learning. All six candidates are now the approved defaults, with selectable
subsets. The original three retain their order; `q_learning`, `reward_decay_hmm`,
and `ideal_observer` are appended. See
[the expansion plan and validation handoff](trial_policy_candidate_expansion_plan.md)
for parameters, the belief-symmetry correction, general tie grids, and v2 results.

**Immediate-persistence extension (2026-11-11):** `immediate_persistence` is the
seventh default candidate, appended after the six above. It repeats the previous
eligible actual mouse choice regardless of reward, using the same shared
previous-choice computation as the existing benchmark model. It is indifferent
before any eligible choice. This differs from the exponentially weighted
`perseveration_regressor`, which averages multiple earlier choices. The adapter
has no task-feature parameters and uses the existing shared choice error rate.
Missing rewards do not prevent its choice-memory update; exclusions such as
manual rewards, invalid switches/no-choice attempts, and unknown manual flags
still freeze it. This choice-only exception is approved explicitly and saved in
the v4 metadata. The outer HMM's policy-state persistence is a separate concept.

Those future adapters must use the raw causal policy values. Session-wide residualized columns such as `HMM_decay_res` and `rel_hazard_res` can depend on later observations and are not substitutes for these policies.

Use a small ordered collection of candidate definitions and ordinary adapter functions. Each adapter receives one chronological session's observed history and explicit parameters, and returns aligned left-positive pre-choice scores and prediction availability, each with shape `(n_trials,)`. Its definition records its stable ID, parameter provenance, and indifference rule. Common stages operate on `(n_trials, n_models)` arrays and model labels; they must not assume exactly three models. Avoid a new agent class hierarchy or plugin framework.

Use one FQL candidate initially. The current standard and fast-learn FQL features have the same decay and differ only in positive reward-update scaling from zero initialization; they have the same greedy policy. Duplicating that policy would change prior mass without adding distinguishable predictions.

This task does **not** require new behavioral-model fitting, learning new HMM emissions, or simulated choices/rewards.

## 1. Inspect and cross-check the existing implementation first

Before implementing, inspect:

- **The existing block-level Dynamax pipeline:** likelihood-array layout, transition-matrix construction, smoother calls, state/model labels, outputs, plotting, and tests.
- **Existing Left/Right conventions:** raw action codes, signed-value conventions, and whether a positive model score favors Left or Right.
- **Existing FQ-learning, Bayesian, and WSLS implementations:** parameter sources, initialization, greedy-choice helpers, tie handling, and whether stored model values describe the state **before or after** the current trial’s outcome.

Reuse these implementations and shared helpers wherever appropriate. Do not create a parallel HMM implementation or redefine the behavioral models.

Check disagreements between this specification and the code explicitly. Ask only questions that remain consequential after inspection; otherwise document the resolved conventions briefly.

**Keep the current block-level analysis unchanged.**

Code inspection resolved the conventions below. Reuse the existing functions rather than rewriting their learning rules:

- Raw actions are `0=Right`, `1=Left`; positive signed values favor Left.
- All three selected feature functions record values before the current outcome update. Do not shift their columns.
- Q values initialize to zero; Bayesian context beliefs initialize symmetrically; WSLS has no prior choice/outcome and initially returns zero.
- WSLS uses the last usable actual mouse choice and outcome, staying after positive reward and switching otherwise.
- The existing agreement-analysis helper repeats its previous recommendation on exact ties, initially Right. This new trial analysis deliberately uses indifference instead; existing agreement outputs retain their current behavior.
- The generic fixed-exemplar smoother accepts arbitrary finite log likelihoods, but its hard states use `argmax`. Reuse its probabilities, initial distribution, and transition matrix; derive trial-specific tie-aware labels separately. Do not reuse the block wrapper that drops invalid rows or its Gaussian likelihood calculation.

### Parameter and reward contract

Use explicit session trial-feature parameters (`gather_trial_features.TaskParams`), saving their exact effective values and source. Current defaults include FQL decay `0.7`, reward-update rate `1 - decay`, Bayesian context-switch probability `0.2`, active-side reward probability `0.8`, inactive-side reward probability `0`, and `tanh_scale=1`. These are assumed policy parameters, not estimates from this analysis. Do not silently substitute generic simulator defaults or actual task-context labels.

The normal processed experimental table encodes reward delivery as `0` or `1`, with default reward sizes `correct_reward_size=1` and `incorrect_reward_size=0`; reward is not water volume. Preserve this convention. Validate actions as exact finite `0` or `1` after parsing supported representations; do not truncate fractional actions to integers. Validate finite supported rewards and parameter compatibility. Unsupported nonbinary reward input requires an explicit compatible model contract rather than silent binarization.

Input inspection confirms numeric codes, CSV numeric strings, `None`/`no_choice` sentinels, both legacy/current manual-flag column names, and flag-less legacy/simulated tables. Accept exact numeric `0/1` values and equivalent numeric strings/float representations. Reuse existing missing-value conventions (`project_utils.MISSING_STRINGS`, real missing values, and action no-choice labels). Preserve native boolean flags' numeric `0/1` behavior; do not add textual `true/false`, `yes/no`, or `left/right` aliases. Textual side names are invalid actions; unrecognized flag values use the unknown-flag exclusion/logging rule. Interpret aliases within each source before heterogeneous concatenation can introduce missing columns.

The Bayesian implementation uses exact reward-size support and normalizes outcome probabilities with an epsilon. An impossible observation can otherwise collapse beliefs to zero and appear to yield a neutral score. Detect impossible/non-finite belief updates and report a calculation error with session/trial/model context; never reinterpret them as genuine indifference. Default parameters support ordinary binary observations. Missing outcomes follow the explicit skip-update rule below and are distinct from impossible supplied outcomes.

Keep this edge-case handling small: validate inputs once at the trial-analysis boundary and check outcome normalization at the existing Bayesian update. Use an ordinary descriptive exception to stop the affected session's trial analysis when the update is impossible. Report the observation and relevant parameters so the user can correct the input/configuration. Do not add a second Bayesian replay for validation, custom exception hierarchies, fallback models, automatic repair, retries, or a separate recovery framework. Use the existing workflow's error reporting.

## 2. Trial alignment and history replay

For trial `t`, every candidate must recommend its action using only the mouse’s history **before trial `t`**.

Processing order:

```text
1. Use mouse choices/rewards through trial t−1.
2. Obtain each candidate’s greedy recommendation for trial t.
3. Score the mouse’s actual choice on trial t.
4. Update every candidate using the mouse’s actual choice and reward on trial t.
```

If suitable precomputed model outputs already exist, reuse them only after verifying row alignment, causal calculation, effective parameters, and compatibility with the current input history and eligibility rules. A historical CSV accompanied by `trial_feature_params.json` alone does not establish this: those files do not bind values to an input/code digest. Values generated in the current workflow with known parameters may be reused. Otherwise replay the existing functions with explicit parameters and record that provenance; do not infer old parameters from column names. Do not shift already pre-choice quantities.

All models continue processing the actual mouse history throughout the session, regardless of the inferred behavioral state. They do not simulate their own histories or stop updating when another model is favored.

Do not expose the true current context or future observations to a candidate that is supposed to infer context. True context may be used for plotting. Preserve the existing causal model initialization; do not add resets at true block boundaries.

The HMM states represent **candidate behavioral policies**, not the task’s Left/Right contexts.

Replay independently by source session before combining results. A missing prediction from one candidate does not stop other candidates from updating on an otherwise usable observed outcome. Never use the independent winner or smoothed state to gate any candidate's history updates.

Existing `min_time`/`max_time` preprocessing can trim a session before feature generation. Current feature collection initializes at the first retained input row; reusing values computed before trimming instead retains earlier history and is not equivalent. Save the analyzed range and initialization convention. Within the supplied table, preserve chronological row order and original identifiers; do not infer missing observations from gaps in numeric trial IDs.

**Approved cropping behavior:** for intentionally cropped sessions, preserve current first-retained-row initialization in round one. Do not replay the omitted earlier history or reuse precomputed scores that include it. Full-session inputs are unaffected. Record the retained range and initialization convention; do not claim that a cropped replay uses unobserved earlier history.

## 3. Convert greedy recommendations to choice probabilities

Use one configurable parameter shared by all models:

```python
opposite_action_error_prob = 0.10
```

Interpretation:

```text
Follow greedy recommendation: probability 0.90
Choose the opposite side:      probability 0.10
```

This is **not** a 10% chance of choosing randomly between both sides.

For the likelihood calculation only, convert the existing action codes to:

```text
y_right = 1 for an observed Right choice
y_right = 0 for an observed Left choice
```

Do not change the codebase’s original action representation.

For valid raw actions, `y_right = 1 - action`. The conversion must happen only after action validation.

For each trial and model:

| Greedy recommendation | `p_right` |
|---|---:|
| Right | `1 - opposite_action_error_prob` |
| Left | `opposite_action_error_prob` |
| Genuine indifference between sides | `0.5` |

A genuine model tie is represented as `p_right=0.5`, with no greedy side. Do not use the existing repeat-on-tie agreement helper or invent an initial Right choice. For the three first-round adapters, use exact zero of their existing signed score as the indifference rule. Record this rule explicitly.

Do not apply a blanket absolute near-zero threshold to signed scores: repeated forgetting can make a nonzero FQL score arbitrarily small without changing its greedy direction. Positive rescaling must not change its recommendations. This is separate from the numerical tolerance for comparing posterior probabilities below. The later decay-HMM adapter must address roundoff around symmetric belief using its belief representation; a global score deadband is not an acceptable substitute.

Missing data or failed model calculations are **not** genuine indifference; handle those separately.

Validate `0 < opposite_action_error_prob < 0.5`. Do not fit model-specific error probabilities in this initial implementation.

Also require both constructed float64 probabilities `opposite_action_error_prob` and `1 - opposite_action_error_prob` to lie strictly inside `(0, 1)`. Reject values such as `1e-20`, whose complement rounds to one, with a clear parameter error; do not silently clip them or generate infinite log likelihoods.

## 4. Calculate trial-by-model likelihoods

Construct:

```text
p_right.shape = (n_trials, n_models)
model_names   = fixed, ordered model labels
```

For observed choice `y` and model probability `p`, use the Bernoulli log-likelihood:

\[
LL = y\log(p)+(1-y)\log(1-p).
\]

Use SciPy’s `bernoulli.logpmf`, which evaluates the log probability mass of a binary observation. [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html)

```python
from scipy.stats import bernoulli

# On valid rows:
log_likelihoods = bernoulli.logpmf(
    y_right[:, None],
    p=p_right,
)
```

With the default error probability:

```text
Mouse agrees with greedy recommendation:    log(0.9)
Mouse disagrees with greedy recommendation: log(0.1)
Model is indifferent:                       log(0.5)
```

Optionally retain descriptive choice residuals:

```python
choice_residuals = y_right[:, None] - p_right
```

**Do not use Gaussian residual likelihoods, exemplar standard deviations, or normalized TTS residuals here.** The observation likelihood is Bernoulli.

## 5. Independent model comparison: preserve ties

Compute independent probabilities using an equal prior over candidate models:

```python
from scipy.special import softmax

independent_probs = softmax(log_likelihoods, axis=1)
```

Here softmax normalizes the exponentiated log-likelihoods over models. [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html)

**Ties are expected and important.** With a common error probability, models recommending the same action receive identical likelihoods on that trial.

For each valid trial, record which models attain the maximum likelihood. Assign a single independent state only when the maximum is unique:

```text
Unique maximum:
    independent_state = winning model
    independent_tie = False

Multiple maxima:
    independent_state = missing / ambiguous
    independent_tie = True
    preserve the tied-model mask or labels
```

Do not silently select the first model, randomly resolve ties, or carry forward the previous independent assignment. There is no shared existing behavioral tie-tolerance constant. Define and save a probability-space maximum-tie tolerance, initially `probability_tie_atol=1e-6`, `rtol=0`, applied to independent and smoothed probabilities. A model is a tied maximum when its probability is within this tolerance of that row's maximum. This accommodates the existing JAX float32 computation; it is not a scientific confidence threshold or a tolerance on behavioral scores. Test model-order invariance and symmetric cases without changing global JAX precision for the block workflow.

When every model recommends the same side, the trial provides no evidence distinguishing those models.

These probabilities are conditional on the selected candidate set and equal model prior. With three decisive binary policies, at least two recommend the same side, so independent ties are expected. Adding candidates in later rounds changes normalized probabilities and transition destinations; preserve candidate IDs/order in results and do not compare state indices across different candidate sets.

## 6. HMM smoothing with fixed policy likelihoods

Reuse the existing block-level inference helper if it accepts arbitrary time-by-state log-likelihoods. Otherwise use the low-level Dynamax function:

```python
import jax.numpy as jnp
from dynamax.hidden_markov_model.inference import hmm_smoother

posterior = hmm_smoother(
    initial_distribution=jnp.asarray(initial_probs),
    transition_matrix=jnp.asarray(transition_matrix),
    log_likelihoods=jnp.asarray(log_likelihoods_for_hmm),
)

smoothed_probs = np.asarray(posterior.smoothed_probs)
```

Dynamax accepts the supplied log-likelihood matrix directly; no Gaussian emission model or emission fitting is needed. [ProbML](https://probml.github.io/dynamax/_modules/dynamax/hidden_markov_model/inference.html)

### Transition settings

Reuse the transition-matrix builder, but give this analysis its **own trialwise configuration**:

```python
trial_stay_prob = 0.90  # Initial configurable setting, not an estimated optimum.
```

For `K > 1`:

```text
A[i, i] = trial_stay_prob
A[i, j] = (1 - trial_stay_prob) / (K - 1), for i != j
initial_probs[i] = 1 / K
```

For `K == 1`, use `A = [[1.0]]` and `initial_probs = [1.0]`.

Do not automatically inherit a blockwise stay probability: transitions now occur between **trials**, not blocks. For `K > 1`, `0.90` implies an expected state duration of 10 retained input trial rows, including manual/no-choice and other invalid rows. This is a modeling setting, not a result or ten valid animal choices. For `K == 1`, there is no state switching.

For round one, the user approved a numerical support bound: for `K > 1`, finite float64 `trial_stay_prob` must satisfy the inclusive bounds `1e-6 <= trial_stay_prob <= 1-(K-1)*1e-6`. These requested-value bounds are authoritative; apply no tolerance to them. The mathematical transition-entry floor is `1e-6`, but checking the derived matrix allows absolute roundoff `1e-12` (relative tolerance zero), including after JAX dtype conversion. Do not clip or otherwise alter the matrix to force the floor. Reject out-of-range requested settings with an explanatory error. The one-state case remains `A=[[1]]`. This replaces the former unrestricted `[0,1]` contract: installed Dynamax 1.0.2 can incorrectly discard small positive predicted probabilities in its backward pass, including at stay probability one. Keep the default .90, test exact bounds and immediately outside float64 values, and do not modify the block workflow or installed dependency to support excluded settings.

Validate long evidence reversals against a stable test-only log-domain reference for K=2,3,4 at both bounds and default .90. Require finite normalized posteriors; float32 posterior/reference comparisons use `atol=2e-6, rtol=0`. This validation tolerance does not change the separate probability maximum-tie rule.

Keep transition probabilities fixed in this first implementation. Do not add transition optimization unless it is separately requested.

### State outputs

Save all `smoothed_probs`. Derive a hard label from the largest smoothed probability, preserving ties as ambiguous.

Call this the **smoothed marginal state assignment**, not the Viterbi sequence. Smoothing uses the whole session, including later observations; it is retrospective state estimation rather than next-choice prediction. [ProbML](https://probml.github.io/dynamax/_modules/dynamax/hidden_markov_model/inference.html)

Run each session separately. Do not reset the behavioral-state HMM at task block boundaries or connect transitions across sessions.

Smoothing is enabled by default. If explicitly disabled at the workflow boundary, still save independent results and omit the two HMM panels; do not synthesize HMM outputs. A unique maximum does not imply strong evidence: always retain the full probabilities. Smoothed assignments can be driven by neighboring observations and the persistence prior even when the current row does not discriminate policies. They are descriptive conditional policy assignments, not proof of a unique behavioral mechanism or an out-of-sample prediction score.

## 7. Initialization and missing observations

Use animal-choice eligibility: a valid Left/Right action on a non-manual trial, with explicit handling of unknown flags below. Preserve original trial identifiers. Do not reuse the legacy GLM-HMM eligibility mask: it requires previous-action/predictor fields and can wrongly remove initialized first trials.

For initial trials without sufficient history, use the existing model’s causal initialization. A genuinely unspecified starting preference may be represented as `p_right = 0.5`; do not invent a previous choice or reward.

For missing observed choices or unavailable model predictions:

```text
comparison_valid = observed animal choice is eligible
                   AND all selected models have valid probabilities
```

Use this common mask so models are compared on the same observations.

Maintain a separate `history_update_valid` flag for complete usable choice/outcome
pairs. Choice-only persistence uses the prepared history's animal-choice
eligibility even when that outcome-update flag is False:

- Known manual-reward or no-choice rows: no choice likelihood, no behavioral-model update. Retain the HMM time step. Existing Q forgetting, Bayesian hazard advancement, and WSLS memory all freeze on these rows.
- Eligible choice with a missing reward: score the current pre-choice recommendation if available for every model; skip the entire update for outcome-based candidates, including forgetting/hazard advancement. Do not fabricate a zero reward. WSLS retains the last usable choice/outcome pair. Immediate persistence remembers the observed choice for its next recommendation.
- Unknown/missing values in an existing manual-reward flag column: exclude from scoring and history updates and log the condition. If the entire flag column is absent, retain the project's compatibility convention of treating rows as animal-controlled, and record that assumption. Normalize the legacy flag-column alias with the existing helper.
- A missing candidate prediction invalidates the comparison row for every candidate. It does not by itself invalidate an otherwise usable outcome update.
- Unsupported/malformed supplied values and unexpected numerical failures must be reported explicitly, not silently converted into ordinary missingness or indifference.

Save exclusion/update flags or reason codes and log counts plus affected original trial IDs for missing rewards and unknown flags. Skipping a missing outcome is an explicit carry-forward approximation for outcome-based policies: subsequent recommendations condition on available usable history, not the unknowable complete mouse history. Do not impute outcomes or add a latent missing-history model. All candidates share animal-choice exclusions, with the approved missing-reward exception for immediate persistence. Save `candidate_history_update_rules` and `history_update_valid_definition` so that `history_update_valid=False` is not interpreted as freezing this choice-only model. Unverified cached features with different flag handling must be replayed. Do not alter shared legacy analysis behavior to implement this policy.

Make this visible with one concise warning per condition per session in the existing console/run log, stating the count, affected trial IDs, and action taken. For many affected trials, show a short ID preview and keep the full row flags in saved results. Missing-reward messages distinguish skipped outcome-based updates from persistence remembering the observed choice when selected. Missing rewards follow this documented continuation rule without user prompts. Use ordinary masks, the saved flags, and existing logging; do not introduce per-trial warning spam, extra diagnostic files, or a missing-data subsystem. Clean sessions need no edge-case warnings.

**Do not delete interior invalid trials and concatenate the remaining rows before smoothing**, because that changes the spacing of HMM transitions.

Instead:

```text
Original likelihood output:
    invalid comparison rows = missing

Separate HMM input:
    invalid comparison rows = 0.0 for every model
```

The all-zero log-likelihood row supplies no state-discriminating observation evidence while retaining that time step.

Independent outputs remain missing on invalid rows. The HMM may still return probabilities inferred from neighboring trials; retain the validity flag and distinguish these rows in plots. A missing model prediction must never receive a neutral likelihood by itself while other models are penalized.

If the established block implementation handles gaps differently, identify the difference rather than copying it without checking the consequences.

The current block wrapper compresses its valid rows. The trial analysis must instead pass one HMM row for every retained input row. These are deliberately different gap policies. Neutral HMM rows still permit transitions, while behavioral-model history freezes on skipped outcomes.

For zero input rows, return aligned empty arrays and an `empty` status without calling Dynamax. For a nonempty session with no valid comparisons, save `no_valid_comparisons`, keep independent outputs unavailable, and retain uniform prior-only HMM probabilities when smoothing is enabled. With multiple candidates, these are tied and have no hard assignment. With one candidate, probability one is structurally unavoidable and must still be marked as having no comparison evidence. Do not present either case as successful behavioral identification.

## 8. Outputs

Match the existing block-analysis naming and storage style. Save arrays or equivalent columns for:

```text
session_id
trial_id
block_id
comparison_valid
history_update_valid
comparison_exclusion_reason
history_update_exclusion_reason

greedy_choice_by_model
model_indifferent_by_model
prediction_valid_by_model
p_right_by_model
log_likelihood_by_model
choice_residual_by_model              # Optional

independent_prob_by_model
independent_best_model_mask
independent_state                    # Nullable
independent_tie

hmm_smoothed_prob_by_model
hmm_smoothed_best_model_mask
hmm_smoothed_state                   # Nullable
hmm_smoothed_tie
```

Save model order, existing model-parameter provenance, error probability, transition matrix, and initial probabilities with the results using the project’s existing metadata mechanism.

`greedy_choice_by_model` retains raw action codes and is missing for both indifference and unavailable predictions; the two explicit masks distinguish these conditions. Independent probabilities/states/tie flags are unavailable on invalid comparison rows, and their best-model mask is all false. HMM probabilities and assignments on invalid rows may be retained, but must be identified as inferred without a current comparison observation. Persist both independent and HMM tied-winner masks so ambiguity is inspectable.

Map generic identifiers to existing `session_ID`/full session ID, `cur_trial`, and original `cur_block`. For concatenated input, use `source_session_id`, `source_trial_row`, and `source_cur_block` where provided; preserve local trial IDs and row positions. Use a dedicated trial-policy output prefix and do not overwrite inherited block-strategy/model/probability columns.

Use named NPZ arrays with `meta`, a trial-analysis version, exact effective candidate parameters, parameter source, input/code provenance, candidate order, eligibility/neutrality/tie rules, analyzed range, status, and library versions. This inference is deterministic and requires no artificial random tie breaking or random seed. Follow existing cache invalidation and preservation conventions: changed parameters, candidate set, inputs, or code must not silently reuse or overwrite prior results. Numerical results remain in session processed-data subdirectories, separate from figure generation.

Do not introduce a separate metadata framework.

Include original source flag-column presence and alias names in the input digest/cache identity, using a fixed-order list drawn from `give_reward` and `experimenter_reward_given` before alias normalization. Absent flags, a present all-zero column, and equivalent alias renaming must have distinct identities even if normalized values match. Schema changes invalidate cached results; ordinary dataframe column reordering does not. Checked reattachment must compare current source schema as well as history values, rather than replacing provenance on cached arrays.

### Main-workflow integration

Enable the trial analysis by default in both normal single-session and multisession workflows, with its own enable, smoothing, stay-probability, error-probability, candidate-selection, and rerun settings. Retain a way to disable trial analysis independently of block analysis. The first round exposes only the three implemented candidates; future IDs must not be accepted as working implementations before their adapters exist.

Run each source session independently in multisession mode, using its own effective feature parameters and initialization. Concatenate aligned results only after per-session inference. Preserve collection-only behavior: it must not trigger model inference. Do not revive legacy `ssm` imports or trial GLM-HMM fitting. Keep the existing block workflow, including its continuous multisession smoothing, functioning independently.

Follow existing session-level batch processing and parameter-aware cache reuse. Save single-session diagnostics under the session figure directory and multisession diagnostics under the established `cross_session_analysis` figure folder. For a multisession display, mark session boundaries and break state traces at those boundaries; joining visual axes must not imply cross-session inference. Do not add an aggregate occupancy/statistical comparison or a prediction-coverage panel in this round.

## 9. Diagnostic figure

Follow the existing block-level diagnostic style, using trial position on a shared x-axis. Include:

| Panel | Content |
|---|---|
| Choice probabilities | Each model’s `p_right`, with actual mouse choices shown as Left/Right markers. |
| Log-likelihoods | Each model’s trialwise log-likelihood. |
| Independent assignment | Unique best model or all tied winners; do not draw arbitrary tie-broken labels. |
| HMM probabilities | Smoothed probability of each model, preferably a heatmap or existing probability display. |
| HMM assignment | Smoothed marginal state labels, with ambiguous/missing cases distinguished. |

Keep model colors/order consistent. Mark actual task block boundaries for orientation only.

Use the approved light mode: opaque white background, black text/axes, readable fonts, and PNG export. Include a caption identifying candidates, shared error rate, trial stay probability, tie handling, session resets, and the retrospective interpretation. Visibly distinguish excluded observations and model/HMM ambiguity. Figures must load saved numerical results so plotting does not rerun inference.

Overlapping probability and likelihood traces are expected in this greedy-plus-error version; do not jitter their numerical values.

## 10. Focused validation tests

Test the following without building a new testing framework:

| Test | Expected result |
|---|---|
| Model recommends Right | `p_right = 0.9`; observed Right scores `log(0.9)`, observed Left scores `log(0.1)`. |
| Model recommends Left | Reverse the probabilities and scores above. |
| Genuine model indifference | Either observed choice scores `log(0.5)`. |
| Two models recommend Right and mouse chooses Right | They tie; no arbitrary independent winner. |
| All models have identical probabilities | Independent probabilities are uniform. With uniform initialization and symmetric transitions, HMM probabilities also remain uniform. |
| Every transition-matrix row is uniform, with uniform initialization | Smoothed probabilities match independent probabilities on valid rows. |
| Current eligible choice/reward or future observations are changed | Pre-choice recommendations for current and earlier trials remain unchanged, with fixed model parameters and eligibility. Changing current eligibility may change output availability. Smoothed probabilities may change retrospectively. |
| Missing interior observation | Trial remains in the HMM sequence; no NaNs enter Dynamax and no model receives a missing-data advantage. |
| Separate sessions | No candidate history or HMM transition is carried across session boundaries. Changing one session cannot change another session's numerical outputs. |
| Neutral initialization and first valid trial | All initial candidates use their existing neutral initialization; first trials are not removed for missing previous action. |
| Tiny nonzero FQL scores and positive rescaling | Greedy directions remain unchanged; there is no blanket near-zero indifference threshold. |
| Unsupported reward / impossible Bayesian update | An explicit error identifies session, trial, and model; no zero-belief collapse is reported as indifference. |
| Missing reward with valid choice | Current choice can be scored; all candidate outcome updates freeze; flags and logs identify the approximation. |
| Manual/no-choice/unknown-flag rows | Approved common eligibility and skip-update rules hold; original HMM row spacing remains intact. |
| Missing prediction from one candidate | Every model's comparison likelihood is missing for that row; valid mouse outcomes still update the other candidates. |
| Candidate-order permutation | Probabilities and tied-winner masks permute consistently; no scientific result depends on arbitrary list order. |
| Configurable candidate collection | Computation and plotting support one, two, three, and four candidates (using a synthetic fourth adapter in tests) without hard-coded state assumptions; unimplemented production candidates fail clearly. |
| Empty, all-invalid, and single-row inputs | Defined statuses, aligned shapes, and correctly distinguished prior-only/absent outputs. |
| Cached values / changed parameters | Unverified stored features are replayed; input, parameter, model-order, and analysis-version changes invalidate numerical cache reuse. |
| Workflow and plotting integration | Single- and multisession entry points, disabled smoothing, collection-only mode, output paths, saved-result plotting, and preservation of block outputs follow this spec. |

Verify Left/Right conversion and WSLS timing against a short hand-checked sequence. Confirm that the existing block-level pipeline still passes its tests.

Include a small enumerated-path HMM reference check for a short synthetic sequence with an interior neutral row, so validation checks posterior values as well as array length. The uniform-transition test uses `trial_stay_prob=1/K` for `K>1`. The uniform-posterior test requires identical candidate likelihoods at every row of the session; agreement at a single row alone does not require a uniform smoothed posterior there.

Implementation follows the project's tests-first workflow: list focused tests in the implementation plan, write and commit failing tests before implementation, verify RED/GREEN, and run a synthetic end-to-end pipeline before a user-designated experimental session. Experimental batch rollout follows user inspection of that session's outputs. No experimental data processing is authorized by finalizing this specification alone.

**Implementation boundary:** this is a fixed-policy agreement/likelihood analysis with temporal smoothing. Do not add logistic-weight fitting, new learning rules, parameter searches, or GLM-HMM emission fitting in this task.
