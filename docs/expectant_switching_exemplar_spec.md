# Expectant-switching exemplars: implementation-planning specification

## 1. Goal and scope

Plan two modular, trial-level exemplar choice models:

1. **Simple probe + persistence:** repeat the previous choice, with an opposing, constant-strength probe signal after reward.
2. **Full expectancy + persistence + doubt:** repeat the previous choice, with a reward-count-dependent expectant-switch signal and the existing accumulating doubt signal.

The target behavior includes repeated multi-trial excursions to the incorrect side, returns to the correct side, and subsequent excursions **without a real context change**. An excursion must not be forced to end after one omission.

This is a planning specification, not a request for an immediate implementation or a new model-fitting framework. The source-code locations below come from a supplied Codex summary; they have not been inspected in this chat. Verify their current definitions and conventions before proposing changes.

## 2. Conventions and variables

The equations below use **positive = left-favoring** and **negative = right-favoring**. These are illustrative conventions, not permission to assume that every existing function follows them.

**Codex must verify choice encoding, predictor signs, reward encoding, and pre-trial/post-trial alignment in the existing code.** In particular, an existing quantity called “doubt” may point toward the doubted side rather than toward the preferred next choice; it may require a sign flip before addition.

| Symbol / suggested name | Meaning |
|---|---|
| `t` | Trial being predicted; only earlier observations may influence its prediction. |
| `c_prev` | Previous valid choice: +1 left, -1 right in these equations; 0 before any valid choice. |
| `r_prev` | Previous valid reward indicator: 1 rewarded, 0 omitted; initialized to 0. |
| `confirmed_side` | Side of the most recent observed reward; unknown before the first reward. This is not the true hidden context. |
| `reward_count`, k | Rewards in the current reward-confirmed-side episode, including the latest observed reward. Initialized to 0. |
| `threshold`, k0 | Reward count where expectancy reaches 0.5; not a hard onset threshold. |
| `scale`, s | Positive scale controlling the tanh transition width. |
| `E_t` | Nonnegative expectancy strength for trial t, calculated from the stored count. |
| `P_t` | Persistence predictor, equal to `c_prev`. |
| `G_t` | Constant post-reward probe predictor, `-c_prev * r_prev`. |
| `X_E,t` | Count-dependent expectant-switch predictor, `G_t * E_t`. |
| `D_t` | Existing doubt predictor, aligned to trial t and oriented so positive favors left. |
| `w_P`, `w_R`, `w_E`, `w_D` | Nonnegative exemplar weights for persistence, constant probing, expectancy, and doubt. Zero enables ablation. |
| `S_t` | Unbounded weighted decision drive. |
| `V_t` | Bounded model value, `tanh(S_t)`, in [-1, 1]. |

Use symmetric left/right parameters initially. Directionality comes from the signed choice, not separate left and right fitted models.

## 3. New expectancy module

### 3.1 Expectancy function

Use the shifted/scaled tanh discussed in planning:

\[
E_t=E(k_{t-1})=\frac{1+\tanh((k_{t-1}-k_0)/s)}{2},\qquad s>0.
\]

Use **k0 = 3 and s = 1 as the initial illustrative configuration**, matching the latest plots; expose both as configuration, not hardcoded or fitted scientific constants.

The directional signal is:

\[
X_{E,t}=-c_{t-1}r_{t-1}E(k_{t-1}).
\]

`E` is a magnitude in [0, 1]; `X_E` is signed and bounded in [-1, 1]. After a left reward it favors right; after a right reward it favors left. After an omission it is exactly zero, even though the internal reward-count memory survives.

Do **not** replace `E` with plain `tanh((k-k0)/s)`: its negative values below threshold would actively favor staying, which is a different hypothesis. The specified function approaches zero at low counts but is not exactly zero below threshold.

### 3.2 Reward-count increment/reset rules

Maintain one **reward-confirmed episode counter**, not true block age and not two lifetime side totals.

| Newly observed event | Update to `confirmed_side` | Update to `reward_count` |
|---|---|---|
| Start of independent session | Unknown | 0 |
| First observed reward, on side A | A | 1 |
| Reward on the same confirmed side A | Keep A | Add 1 |
| Omission on confirmed side A | Keep A | No change |
| Omission on opposite side B | Keep A | No change |
| Choice changes without reward | No reward-confirmed change | No reset |
| Reward on opposite side B | B | Reset to 1, including that reward |
| Experimenter-labelled context switch | No model update from this label | No model update from this label |

After **every valid completed trial**, separately set `c_prev` to its observed choice and `r_prev` to its observed binary outcome.

Important consequences:

- An unrewarded excursion does not erase the count accumulated on the confirmed side.
- Returning and receiving reward on that same confirmed side continues the old count.
- A reward on the opposite side starts a new confirmed-side episode, even if this was only a single-trial visit.
- Returning later to a formerly confirmed side and obtaining reward starts another episode at 1; do not revive its historical count.
- There is no count decay or omission-driven count decrement in this initial model.

The gate is based on the **immediately preceding valid trial's reward**, not “whether this side has ever been rewarded.” Multiple omissions therefore preserve `k` but keep `X_E` at zero.

## 4. The two exemplar compositions

### 4.1 Simple probe + previous-choice persistence

\[
P_t=c_{t-1},\qquad G_t=-c_{t-1}r_{t-1}
\]

\[
S_{\mathrm{simple},t}=w_PP_t+w_RG_t
\]

\[
V_{\mathrm{simple},t}=\tanh(S_{\mathrm{simple},t}).
\]

This model has no reward-count dependence and no doubt contribution. It does not need to maintain a reward counter.

After reward, its drive is `c_prev * (w_P - w_R)`. After omission, its drive is `c_prev * w_P`. Reward reduces or reverses persistence; omission itself supplies no additional switching pressure.

This is a constant post-reward probe tendency, not automatically probability matching to the true task-switch hazard. Do not set its probe probability equal to the task hazard without an explicit separate configuration decision.

### 4.2 Full expectancy + previous-choice persistence + existing doubt

\[
S_{\mathrm{full},t}=w_PP_t+w_EX_{E,t}+w_DD_t
\]

\[
V_{\mathrm{full},t}=\tanh(S_{\mathrm{full},t}).
\]

The roles are distinct: persistence sustains a run, expectancy promotes post-reward departure, and doubt promotes departure from a side accumulating omission evidence.

Use immediate previous choice for the initial persistence module, **not the existing exponentially weighted choice-history regressor by default**. A long memory can still favor the old side immediately after an excursion begins; that would be a different comparator.

No extra belief/HMM term, separate latent strategy state, or side-bias intercept is required initially. Weight values have not been selected; expose them explicitly and propose documented exemplar settings after checking the existing doubt scale and behavior. Do not silently assign “fitted” or “optimal” values.

## 5. Signed value and probabilistic choice

Retain both `S_t` and `V_t` if useful for diagnostics. A sum of bounded predictors is not itself bounded; the outer tanh defines the requested bounded model output.

For the sign convention in this specification, use:

\[
P(L_t)=\frac{1+V_t}{2},\qquad P(R_t)=\frac{1-V_t}{2}.
\]

This equals `sigmoid(2*S_t)` for the left probability. If integrating with an existing shared choice-policy API, preserve the intended probability mapping and document any equivalent parameterization. Do not add another softmax to `V_t` and assume it is unchanged.

**Probabilistic choice is essential for the intended simple comparator.** With positive persistence, its probability of leaving an omission run is:

\[
P(\text{switch after omission})=\frac{1-\tanh(w_P)}{2}.
\]

That probability is constant across the run. A greedy version would keep repeating the incorrect side while omissions continue. The full model can instead change its departure probability as doubt accumulates.

Greedy choices may remain available for existing agreement plots, but must not be the only behavioral interpretation or validation of these models. Avoid systematic left selection on exact ties; reuse an appropriate existing tie policy.

## 6. Reuse the existing doubt model exactly

Inspect the existing accumulating doubt implementation, beginning with `relative_doubt_index` and its use in the existing observer/composite models. Confirm that this is the intended doubt component rather than selecting a different omission regressor based only on its name.

Reuse its underlying function/state-transition logic and existing parameters. Do not implement a second, approximately equivalent doubt algorithm inside the new exemplar.

The adapter's contract is:

- Supply choices and outcomes in the encoding the existing doubt code expects.
- Produce a pre-trial `D_t` using only observations through trial t-1.
- Orient its sign to the composite's preferred-choice convention.
- Preserve the existing reward, omission, choice-change, counterfactual-update, saturation, decay, and reset rules.
- Initialize/reset it using its existing independent-session initialization.

**Do not impose “clear all doubt on reward” merely because that was an earlier conceptual example.** Its actual reset behavior is not established by the supplied code summary. Codex's plan must explicitly document what the current implementation does after reward, omission, side change, and session reset, and whether those rules support repeated excursions.

Do not use `doubt_perseveration_value` as the doubt input if it already includes persistence; doing so would double-count that component. Do not per-session normalize or otherwise change the existing doubt signal merely to make weights convenient.

If the existing doubt rules strongly suppress later excursions, report that finding and its implications for weight selection. Any change to those rules is a separately named model variant, not an invisible change to this one.

## 7. Modular design and trial ordering

Separate the following responsibilities, following the codebase's existing functional or class-based style:

| Component | Responsibility |
|---|---|
| Immediate-choice persistence | Return the previous-choice signal. |
| Post-reward probe gate | Return the signed signal `-c_prev * r_prev`. |
| Expectancy memory/curve | Maintain reward-confirmed count and produce `E(k)`; multiply by the probe gate to produce `X_E`. |
| Existing doubt adapter | Expose the existing doubt signal with verified timing and sign. |
| Exemplar composition/readout | Combine named signals with weights and produce bounded value/probabilities. |

The new expectancy module must not own or rewrite the doubt module. The full exemplar should compose them. The simple exemplar should use the same persistence and probe-gate definitions, without a hidden separate implementation of their logic.

Use one source of truth for state updates shared by batch feature generation and any executable agent. Avoid duplicating algorithms in `trial_features.py` and an agent class. A large refactor of unrelated models is not required.

For each valid trial:

1. Read state summarizing history through t-1.
2. Compute predictors, weighted drive, value, and choice probabilities for t.
3. Observe the actual mouse choice/outcome, or sample an agent choice and obtain its outcome in simulation.
4. Update the model states once, making them ready to predict t+1.

A reward earned on t may update the predictor for t+1, never the prediction already assigned to t. All modules must observe the same history without double-updating shared counters.

Reset all module state at independent-session boundaries. If processing one continuous session in chunks, preserve state across chunks. Do not reset at analysis blocks or true hidden-context transitions.

For invalid/missing trials, reuse the pipeline's established validity and continuity policy and document it. Never silently treat missing reward as an omission or feed no-choice sentinels as a side. Count updates require a valid choice/outcome pair. Preserve row alignment and validity masks in outputs; make explicit whether “previous trial” means previous valid trial across excluded rows.

## 8. Worked counter example

Using k0 = 3 and s = 1, all entries below describe state **after observing the listed event**, and `X_E` is the signal for the next trial.

| Observed history/event | Confirmed side | k | Next-trial X_E |
|---|---:|---:|---:|
| Session start | Unknown | 0 | 0 |
| First left reward | Left | 1 | -0.018 |
| Second left reward | Left | 2 | -0.119 |
| Third left reward | Left | 3 | -0.500 |
| Right omission | Left | 3 | 0 |
| Two more right omissions | Left | 3 | 0 after each omission |
| Return to left and receive reward | Left | 4 | -0.881 |
| Another left reward | Left | 5 | -0.982 |
| Subsequently obtain a right reward | Right | 1 | +0.018 |

During the right-omission run, persistence points right and the existing doubt module evolves according to its own rules. The model is not instructed to return on any particular omission count. Elevated expectancy memory is available again when reward resumes on the same confirmed side.

## 9. Outputs and checks

Suggested diagnostic outputs, with naming adapted to the existing registry:

`previous_choice`, `reward_triggered_probe`, `expectancy_reward_count`, `expectancy_strength`, `expectant_switch`, `doubt_choice_signal`, plus each exemplar's decision drive, signed value, and left-choice probability. Keep the confirmed-side state available for debugging. Counts are diagnostic state, not bounded directional predictors.

At minimum, plan tests for:

- **Signs and symmetry:** mirroring left/right negates signed outputs and swaps choice probabilities.
- **Gating:** no reward on the preceding valid trial implies zero probe and expectancy-switch signals.
- **Counter rules:** omissions preserve count; same-confirmed-side rewards increment it; opposite-side rewards reset it to 1; a choice change alone never resets it.
- **No leakage:** changing trial t or later observations cannot change the prediction for t; hidden-context labels never enter predictor updates.
- **Timing/reset:** correct first-trial state, independent-session reset, and equivalent chunked versus whole-session replay.
- **Doubt parity:** adapter outputs match the existing doubt implementation, apart from an explicitly documented sign convention/alignment conversion.
- **Modularity:** with a test expectancy magnitude fixed at 1, doubt weight zero, and `w_E = w_R`, the full composition matches the simple composition.
- **Behavioral checks:** the simple model has constant departure probability during omission runs; the full model can show doubt-dependent returns and repeated excursions under documented parameters. Do not require exactly three incorrect trials or assume every parameter setting produces the target pattern.

When simulating, generate predictors from the agent's own choices/outcomes. For closed-loop simulation of the actual task, use the task's reward-triggered context-transition rules; replay against a fixed mouse-derived context sequence is a different evaluation. Mouse-history prediction and autonomous simulation should be labelled separately.

Compare post-reward excursion frequency, incorrect-side run lengths, return intervals, and recurrence as a function of accumulated rewards—not just total greedy choice agreement.

## 10. Scope boundaries and Codex planning deliverable

Implement the saturating expectancy curve first. Keep the curve replaceable for a later nonsaturating version, but do not silently add one now. An unbounded future expectancy would also make its raw directional predictor unbounded; the outer tanh can still bound the final model value.

The two starting exemplars differ in both count dependence and doubt. Their comparison does not isolate expectancy alone. The modular design should make a future **constant probe + persistence + the same doubt** control straightforward without requiring it in the first implementation.

Candidate integration points from the supplied summary are `trial_features.py`, `gather_trial_features.py`, `session_analysis.py`, the trial predictor registry in `trial_state_space_modeling.py`, and the existing executable-agent infrastructure in the neighboring `behavior_modeling` package. Inspect these before deciding where new code belongs; do not assume the legacy `model_agents.py` is the preferred home.

Before implementation, Codex should return a plan identifying the exact reusable doubt function and its update/reset rules; verified sign/row conventions; module and file boundaries; proposed configuration/presets; required output/registry changes; and tests. Identify incompatibilities with this specification explicitly rather than silently changing either the new model or the existing doubt algorithm.
