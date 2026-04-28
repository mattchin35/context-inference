# Experimenter Reward Handling

Experimenter-given reward trials currently overload the `action` column in a way
that can leak manual reward side information into downstream choice-history
analyses.

Current preprocessing behavior in
`src/mouse_behavior_preprocessing/process_behavior_log.py`:

- `giving_reward_left_patch` is saved as `action = 1`, `correct = 0`,
  `give_reward = 1`.
- `giving_reward_right_patch` is saved as `action = 0`, `correct = 0`,
  `give_reward = 1`.
- Each trial also stores the current block state in `state` and `state_int`, but
  downstream code does not appear to use block state to recover the manual reward
  side.

The problem is that `action` should mean animal choice, not reward-delivery
side. A cleaner data contract would be:

- `action = 0` or `1`: animal chose right or left.
- `action = "no_choice"`: no animal choice occurred before the trial was
  finalized.
- `correct = 0`: acceptable for no-choice or experimenter-reward trials, because
  `correct` is functioning as a boolean for whether the animal made a correct
  choice.
- `reward = 1`: reward was delivered.
- `give_reward = 1`: reward was delivered by experimenter/manual override.

If the side of the manual reward matters later, it should be stored separately,
for example as `experimenter_reward_side = 0`, `1`, or `"None"`. It should not
be encoded in `action`.

Downstream behavior observed during the audit:

- `src/behavior_analysis/session_analysis.py::get_block_switches` detects
  `give_reward` and replaces that trial's `action` with the previous or next
  action before counting switches, so it does not preserve manual reward side.
- `src/behavior_analysis/session_analysis.py::count_decision_variables` skips
  `give_reward` trials entirely for decision-variable updates.
- `src/behavior_analysis/simulate_priors.py::collect_agent_performance` skips
  current `give_reward` trials before using `action`.
- `src/behavior_analysis/trial_features.py` normalizes `give_reward` into a skip
  mask, so model-value updates skip those trials.
- `src/neural_analysis/spike_behavior_pynapple.py::make_trial_type_masks` treats
  nonzero `give_reward` as invalid for neural trial-condition masks.

The main leak is `prev_action` in
`src/behavior_analysis/session_analysis.py::make_augmented_trial_df`. It blindly
shifts `trial_df["action"]`, so the trial after an experimenter reward can inherit
the manual reward side as if it were the animal's previous choice. Then
`src/behavior_analysis/trial_state_space_modeling.py::prepare_trial_glm_hmm_data`
excludes rows where the current trial has `give_reward == 1`, but it does not
exclude rows whose `prev_action` came from a previous experimenter-reward trial.

Deferred refactor:

- Change preprocessing so experimenter reward trials save `action = "no_choice"`
  rather than a left/right side.
- Add a separate manual-reward side column only if that side is needed.
- Update downstream skip logic so `"no_choice"` and previous `give_reward` trials
  cannot enter action-history predictors such as `prev_action`.
- Add tests around preprocessing, `make_augmented_trial_df`, trial feature
  generation, and trial GLM-HMM preparation before making the data-contract
  change.
