# Experimenter Reward Handling

Experimenter-given reward trials previously overloaded the `action` column in a
way that could leak manual reward side information into downstream
choice-history analyses.

Previous preprocessing behavior in
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
- `experimenter_reward_given = 1`: reward was delivered by
  experimenter/manual override.

If the side of the manual reward matters later, it should be stored separately,
for example as `experimenter_reward_side = 0`, `1`, or `"None"`. It should not
be encoded in `action`.

Downstream behavior observed during the audit:

- `src/behavior_analysis/session_analysis.py::get_block_switches` detects
  `experimenter_reward_given` and replaces skipped trial actions with nearby
  animal choices before counting switches, so it does not preserve manual reward
  side.
- `src/behavior_analysis/session_analysis.py::count_decision_variables` skips
  `experimenter_reward_given` trials entirely for decision-variable updates.
- `src/behavior_analysis/simulate_priors.py::collect_agent_performance` skips
  current experimenter-reward trials before using `action`.
- `src/behavior_analysis/trial_features.py` uses `experimenter_reward_given` in
  its skip mask, so model-value updates skip those trials.
- `src/neural_analysis/spike_behavior_pynapple.py::make_trial_type_masks` treats
  nonzero `experimenter_reward_given` as invalid for neural trial-condition
  masks.

New preprocessing and downstream behavior-analysis code now use
`experimenter_reward_given` instead of `give_reward`, and manual reward rows are
saved with `action = "no_choice"`. Old CSVs with `give_reward` are normalized at
load boundaries. The remaining deferred question is whether a separate
`experimenter_reward_side` column is needed for future analyses.

# Session Stats Renaming

Session-level regression statistics in `session_performance` were renamed from
generic names such as `slope`, `intercept`, `r_value`, and `p_value` to names
that identify the independent variable:

- `prev_consecutive_rewards_slope`
- `prev_consecutive_rewards_intercept`
- `prev_consecutive_rewards_r_value`
- `prev_consecutive_rewards_p_value`
- `prev_n_correct_slope`
- `prev_n_correct_intercept`
- `prev_n_correct_r_value`
- `prev_n_correct_p_value`

The session-level block count was also renamed from `n_switches` to `n_blocks`.
This avoids confusing the number of task blocks with mouse choice-switching
behavior.

General analysis functions are expected to work with the new names. Plotting and
notebook code that reads older multisession CSVs may still need updates if it
expects the generic stat columns. Plotting code can temporarily tolerate legacy
`n_switches` for old CSVs, but new analysis outputs should save `n_blocks`.

# State-space session dependency injection

`src/behavior_analysis/block_state_space_modeling.py` and
`src/behavior_analysis/trial_state_space_modeling.py` still pass full session
objects into lower-level fitting and plotting functions in places where only a
few fields are needed. This makes demo code awkward and hides function data
contracts.

Deferred refactor:

- Keep full `Session` dependencies in pipeline functions such as
  `run_block_modeling()` and `run_trial_modeling()`.
- Refactor lower-level fitting helpers to accept explicit paths, session ids,
  and data inputs instead of a full session object.
- Preserve dependency injection: functions should receive the concrete data they
  need rather than reaching through broad session structures.
