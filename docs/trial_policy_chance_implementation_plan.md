# Constant-chance baseline for the trial-policy HMM

Status: implemented and validated, 2026-11-17, after user approval. `chance` is
the ninth default candidate. The sections below retain the approved plan;
completion and validation are recorded at the end.

## 1. Objective and approved behavior

Give the trial-policy comparison a neutral candidate that assigns equal
probability to the two choices. This provides a reference for stretches that
the directional candidates predict poorly. Its posterior is conditional on the
candidate set, fixed readouts, and HMM transition assumptions; it is not a direct
estimate of the fraction of intrinsically random animal behavior.

For every comparison-eligible trial:

```text
P(Left) = P(Right) = 0.5
log P(observed choice) = -ln(2)
```

The model has no learning, history, fitted parameters, temperature, side bias,
or sampled choices. Its emission probabilities are constant; its HMM posterior
can vary as evidence and neighboring trials change.

Append `chance` to the current ordered defaults:

```text
forgetting_q_learning
bayesian_hmm
win_stay_lose_switch
q_learning
ideal_observer
perseveration
win_switch_lose_stay
home_referenced_expectancy_doubt
chance
```

The first eight IDs retain their order, parameter values, and choice readouts.
Selectable subsets, including the original eight-model set and chance alone,
remain supported. Home E+D keeps its current parameters.

## 2. Data contract and eligibility

Reuse `PreparedTrialHistory` and the existing trial-by-candidate arrays.

| Quantity | Contract |
|---|---|
| Adapter inputs | One chronological `PreparedTrialHistory` with aligned `(T,)` arrays, plus the existing unused `TaskParams` argument |
| Action convention | `0=Right`, `1=Left`; no action-code conversion in the adapter |
| Adapter scores | `float64[T]`, dimensionless zero on eligible choices; NaN on excluded attempts |
| Adapter availability | `bool[T]`, equal to `history.comparison_valid` |
| Choice probabilities | Dimensionless `(T,K)` arrays; the chance column is exactly 0.5 wherever available |
| Scored evidence | Natural-log `(T,K)` likelihoods; chance is `-ln(2)` on the common comparison mask |
| History rule | Metadata value `none`; there is no state to update |
| Random seed | `None`, because replay and inference are deterministic |

Missing rewards do not prevent a current choice prediction. Manual rewards,
unknown manual flags, no-choice attempts, and explicit invalid switches retain
the current exclusions. An unavailable prediction from another selected model
still invalidates the common comparison row for all models.

Excluded comparison rows retain NaN likelihoods in saved arrays. The existing
smoother converts the entire row to zero log evidence for every state. Chance
must not contribute `-ln(2)` on those excluded rows, which would penalize it
relative to states receiving no evidence.

Zero scores use the existing diagnostic conventions: `model_indifferent=True`
and `greedy_choice=-1` on available rows. Chance never invents a greedy Left or
Right choice. The shared `history_update_valid` array continues to describe
usable observed choice/outcome pairs; candidate metadata explains that chance
does not use it for updates.

## 3. Production changes

### A. Candidate registration: `src/behavior_analysis/trial_policy_models.py`

- Append `chance` to `DEFAULT_MODEL_IDS`.
- Add a small `_replay_chance` adapter returning aligned zero scores and the
  existing animal-choice eligibility mask. Use ordinary NumPy allocation;
  no new class or model framework is needed.
- Register it with label `Chance`, empty `parameter_names`,
  `history_update_rule="none"`, `choice_rule="constant_half"`, and no temperature.
- Extend `effective_trial_policy_choice_parameters` to return only
  `{"rule": "constant_half"}` for this candidate. Its effective feature
  parameter dictionary is empty.
- Keep temperature overrides restricted to graded candidates, so a temperature
  override for `chance` raises the existing contextual validation error.
- Preserve current validation of shared run options, including the global error
  argument; valid error settings have no effect on chance's probabilities.
- Update relevant docstrings to include the constant choice rule.

The existing benchmark `_replay_chance` already specifies 50/50 predictions.
Use that definition as the reference and verify parity through its public
benchmark specification. Its completed-observation wrapper is not required for
the trial adapter, whose eligibility includes missing rewards. A constant-array
adapter avoids constructing or replaying the entire benchmark collection.

### B. Probability conversion: `src/behavior_analysis/trial_policy_analysis.py`

- Add one `constant_half` branch in `_choice_probability_arrays`.
- On available rows, fill probability 0.5 and both choice log probabilities
  with `-ln(2)` directly. Preserve unavailable NaNs.
- Reuse the existing observed-action likelihood selection, independent
  normalization, tied-maximum handling, and Dynamax smoothing.
- Preserve the existing low-level `choice_parameters=None` behavior for legacy
  callers. The normal workflow supplies the explicit candidate choice records.
- Keep all array shapes, units, and original trial identifiers unchanged.

### C. HMM integration

The smoother already accepts arbitrary candidate counts. With nine defaults:

```text
initial state probability = 1/9 for each candidate
diagonal transition probability = 0.9
each off-diagonal transition probability = 0.1/8 = 0.0125
```

Chance receives the same prior and stay probability as every other state.
Retain source-session resets, retrospective smoothing, and the existing single-
state, empty-input, all-invalid, and smoothing-disabled behavior. No fitting or
special chance-state transition logic is introduced.

### D. Metadata/cache: `src/behavior_analysis/trial_policy_results.py`

- Bump `ANALYSIS_VERSION` from `trial_policy_hmm_v8` to `trial_policy_hmm_v9`.
- Save empty feature parameters, the explicit `constant_half` choice rule, and
  the `none` history rule through the existing candidate metadata mappings.
- Extend the human-readable probability definition to mention constant 50/50
  candidates without assigning them an error rate or temperature.
- Keep the NPZ schema and loader compatible with previous versions. The changed
  source digest, version, and selected model IDs distinguish new cache entries.
- Use the existing immutable numerical artifacts, CSV archive naming, current
  export attachment, run records, logs, and source snapshots.

The existing source-digest list already includes the modules being changed;
no additional provenance mechanism is needed.

### E. Figures: `src/behavior_analysis/trial_policy_plotting.py`

- Add a stable, distinct color and a `Chance` display label.
- Add caption text stating that chance assigns 0.5 to either side. It must not
  appear in the graded-temperature or discrete-error lists.
- Reuse the existing five panels and model-membership grids. Chance adds one
  model row to the independent and HMM winner grids; ties remain multiple
  filled cells in a trial column.
- Retain light-mode PNGs, readable labels, source/block boundary conventions,
  excluded-row cues, and plotting from saved arrays only.
- Preserve rendering of older artifacts whose metadata lacks the new rule.

### F. Main/workflow and documentation

`SingleSessionAnalysisConfig`, the single runner, and multisession defaults
already import `DEFAULT_MODEL_IDS`. No main.py implementation edit is expected.
Check these paths through integration tests, including explicit subsets and
serial/threaded source replay.

Update `docs/trial_dynamax_hmm_spec.md` and
`docs/trial_policy_candidate_expansion_plan.md` with the ninth default, constant
readout, v9 output version, and validation handoff. Existing 20-choice agreement
analyses remain unchanged: chance has no single greedy action to score there.

## 4. Tests to write before production changes

Add `src/tests/behavior_analysis/test_trial_policy_chance.py` for focused cases
and extend the existing result, plotting, and main workflow tests where needed.

| Test group | Required behavior |
|---|---|
| Defaults and subsets | Chance is ninth; the first eight preserve order; chance-only and original-eight selections work |
| Constant predictions | Both actions, varied past histories and rewards, and multiple valid error settings always give 0.5 and `-ln(2)` |
| Reference parity | Match the existing benchmark chance probabilities on common eligible completed observations |
| Eligibility | Preserve missing-reward predictions; exclude manual/no-choice/unknown-flag/invalid-switch attempts; preserve the input table |
| No history | Fresh sessions/crops and empty inputs work; changing preceding actions or outcomes cannot change a valid chance prediction |
| Configuration | Empty feature parameters, constant choice metadata, `none` history rule, and rejected chance-temperature overrides |
| Common scoring mask | Another candidate's unavailable prediction still excludes every likelihood; excluded rows give zero HMM evidence for every state |
| Numerical inference | Nine-state rows normalize, transitions have the specified entries, and all-neutral candidate evidence preserves ties |
| Synthetic chance support | A deterministic alternating-choice stretch favors chance over confident fixed-Left/fixed-Right comparators under the existing sticky HMM |
| Compatibility | Original eight candidates have identical scores/probabilities/likelihoods with chance added; older NPZs and explicit historical subsets remain loadable |
| Persistence/cache | Round-trip the chance column and rule metadata; distinguish v9/default candidate signatures from prior outputs |
| Saved-result plotting | Nine labeled rows, distinct chance color, accurate constant caption, preserved ties/exclusions, no inference during plotting |
| Main integration | Default nine-model single/multisession outputs, selected subsets, cache reuse, empty sources, and serial/threaded equivalence |

Update eight-default assertions because the requirement changed. Keep tests
specifically checking historical eight-model behavior on an explicit eight-ID
subset. Where a plotting fixture appends an optional candidate to the defaults,
size its arrays for the resulting ten IDs rather than assuming nine.

Do not test for a predetermined real-session chance occupancy. Assignment
fractions are descriptive results, not correctness criteria.

## 5. Implementation sequence and verification

1. Recheck the worktree; preserve unrelated edits and existing staging.
2. Write the new tests and the necessary default-count fixture updates.
3. Run the focused tests and confirm failures for the missing candidate/rule.
4. Commit only those tests before modifying production code.
5. Implement the small adapter, probability branch, metadata/caption changes,
   and default registration. Keep existing public call signatures.
6. Run the focused tests, then all affected trial-policy and main-workflow
   regressions. Use `uv run` for every local Python command. Run the main trial
   workflow module in its own pytest process if needed to avoid legacy stubs.
7. Run a deterministic synthetic full-pipeline example through the existing
   single-session runner. Verify numerical save/load, eligible/excluded rows,
   posterior normalization, and the nine-row PNG visually.
8. Run only the designated CT016 May 29 session, inspect its figure, and record
   the result. Broader experimental runs follow user inspection.

Use existing pytest, NumPy/SciPy, and installed Dynamax/JAX. Chance adds one
linear-size emission column; the existing smoother grows from eight to nine
states. No new dependency, profiling project, session-worker change, or nested
parallelism is needed.

## 6. CT016 May 29 validation and acceptance

Designated source:

```text
Session: CT016_2026-05-29_132812
Root: /home/matt/Documents/EXPERIMENTS/contextProjectData/CT016/CT016_20260529_latent_inference
Input: processed/CT016_2026-05-29_132812_augmented_trials.csv
Parameters: processed/trial_feature_params.json
```

Use the current input and resolved saved parameters. On an unchanged source,
compare the new result with the matching v8 artifact: all eight existing score,
availability, choice-probability, and log-likelihood columns must be unchanged.
Check the input digest before making that comparison; record changed inputs
rather than treating them as an implementation regression.

Accept the run when:

- Chance is 0.5 on every available prediction and `-ln(2)` on every common
  scored choice, with unchanged exclusions and retained row identifiers.
- Every smoothed posterior row is finite and normalized within the current
  tolerance, and NPZ round-trip values agree.
- The previous current CSV is archived using the existing distinct filename;
  prior numerical artifacts, non-policy columns, block output, and parameter
  files are preserved using the established validation checks.
- The new PNG is readable and clearly describes constant chance probabilities
  and retrospective model assignment.
- The report records effective candidate/readout parameters, source input and
  code provenance, deterministic seed `None`, sessions, runtime, and checks.

Save numerical results under the session's
`processed/trial_policy_hmm/trial_policy_hmm_v9/<signature>/<timestamp>/`.
Keep plots, a short scientific summary, the validation driver, main/batch
snapshots, and log in the existing timestamped trial-policy figure/run
directories. Report chance posterior means and maximum-membership fractions
separately so a winning-state fraction is not presented as posterior certainty.

The final handoff includes the scoped commits, test results, new CT016 figure
and numerical paths, preservation checks, and any observed limitations. This
work implements the approved baseline; it does not establish which choices
were truly random or calibrate the other candidate policies.

## 7. Completion and validation handoff

Tests were written, confirmed failing (18 focused cases), and committed first
as `9a275a9`. The completed implementation passed 326 policy/model/plotting tests
and 22 main trial-workflow tests (348 distinct tests). Existing warnings are the
JAX/TFP deprecation and intentional invalid-input Bayesian test cases.

The deterministic synthetic pipeline passed with 140 retained rows and 137
eligible comparisons. CT016 May 29 passed with all 858 choices eligible.
Chance has exactly .5 choice probability and `-ln(2)` likelihood on scored rows.
All eight previous models' scores, availability, probabilities, and likelihoods
are exactly unchanged from the matching v8 artifact. NPZ round trips, finite
normalized posteriors, prior current-CSV archival, non-policy columns, nine prior
numerical artifacts, block output, and parameter-file preservation were checked.
The user's main.py diff was also checked unchanged.

The inspected PNG, report, reproducible driver, main/batch snapshots, and log are
in this directory (UTC timestamp, November 17 local date):

`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT016/CT016_20260529_latent_inference/figures/trial_policy_hmm/2026-11-18_04-46-08_024690_UTC/`

Chance's mean smoothed posterior is .3838; its largest-posterior membership
fraction is .5524. These distinct measures reproduce the earlier in-memory
diagnostic, without treating either as an estimate of inherently random choices.
No experimental batch was run. User inspection remains the next step before a
broader experimental analysis.
