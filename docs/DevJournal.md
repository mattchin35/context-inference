# 2026/03/18

Implemented the new behavior-modeling agents and the supporting controller workflow needed to inspect and reproduce their behavior. This included adding an `HMMRewardDecay` agent and a composite `HMMRewardDecayRelativeDoubt` agent that keeps separate belief and doubt signals while combining them into the policy value, extending the controller so runs can be plotted directly from the output dataframe, refactoring controller output handling so run execution, saving, and plotting are managed separately and more readably, and adding optional deterministic seeding so the same task, agent, and seed reproduce the same run.

- Added agent-level internal-value outputs including combined value, HMM value, and doubt value.
- Added dataframe-based plotting for actions, rewards, and model values from controller runs.
- Added documentation updates across `PRD.md`, `ImplementationDetails.md`, and `README.md` to capture the new model descriptions, reproducibility behavior, and controller plotting workflow.

# 2026/04/28

Started cleaning the behavior-analysis path used by `src/main.py`, with the immediate goal of reducing duplicated preprocessing and removing stale module dependencies. The raw behavior preprocessing path is now routed through `mouse_behavior_preprocessing.process_behavior_log`, and `main.py` has an explicit `preprocess_raw_session` flag so preprocessing is no longer controlled by commenting or uncommenting a function call.

- Removed active dependencies on `session_overview`, `block_analysis`, old `fileIO`, and `context_switch_analysis` from the current behavior-analysis path. Historical modules were moved under `src/old` by the user rather than being kept as active dependencies.
- Copied legacy context-switch helper logic directly into `src/behavior_modeling/rnn_analysis/behavior_preprocess.py` so the RNN analysis code no longer imports `behavior_analysis.context_switch_analysis`.
- Documented the experimenter-reward action leak in `docs/RepoRefactors.md`. Current preprocessing still encodes manual reward side in `action`, but downstream code mostly treats `give_reward` as an exclusion flag. The main unresolved issue is that `prev_action` can inherit manual reward side on the following trial.
- Refactored `process_behavior_log.iterate_trials` into helper functions with a `TrialParserState` dataclass, `ChoiceSide` and `TaskState` enums, and a centralized `MISSING_VALUE = "None"` sentinel.
- Added a `missing_value` argument to `iterate_trials` and `make_trial_df`, but it currently asserts that only string `"None"` is supported because downstream CSV loading and analysis code depend on that sentinel.
- Preserved the existing trial-finalization rule: a trial is saved only when a later `trial_start` begins the next trial. Trials cut off by `enter_dark_period` should not be saved because they have no behavioral meaning.
- Added focused pytest coverage for the behavior-log parser and ran related regression tests for `main.py` and session analysis.

# 2026/04/29

Continued refactoring the behavior-analysis path used by `src/main.py`, mostly
inside `src/behavior_analysis/session_analysis.py`. The focus was readability
and reducing duplicated/ambiguous flow while preserving the existing analysis
outputs.

- Added a `run_session_analysis` flag in `main.py`, mirroring the
  preprocessing flag, so session analysis can be explicitly rerun or loaded.
- Split `session_analysis.analyze_session` into smaller helpers for augmented
  trial creation, block-performance summaries, and session-performance
  summaries while keeping `analyze_session` as a convenience wrapper.
- Renamed session-level regression statistics in `session_performance` from
  generic `slope`, `intercept`, `r_value`, and `p_value` columns to columns that
  name the independent variable, including `prev_consecutive_rewards_*` and
  `prev_n_correct_*`.
- Renamed the session-level block count from `n_switches` to `n_blocks` for new
  outputs. Plotting code now tolerates legacy `n_switches` when reading old CSVs.
- Refactored `run_analysis` and `save_analysis` so saving no longer reloads the
  same CSVs immediately afterward. `save_analysis` now returns the in-memory
  session or multisession dataframe and verifies saved CSVs exist and are
  non-empty.
- Split trials-to-correct summary logic into readable helpers for numeric
  conversion, non-final missing-block detection, warning metadata, and mean
  calculation by block type.
- Refactored `count_decision_variables` around a `DecisionVariableState`
  dataclass and helper functions that make the append-before-update contract
  explicit: each row stores pre-current-trial history, skipped manual/no-choice
  trials do not update counters, and valid trials update reward-history,
  choice-value, and counterfactual counters.
- Added focused pytest coverage around the session-analysis helper functions and
  reran related tests for session analysis, performance plotting, and main
  wrapper behavior.

# 2026/05/06

Refactored and bug-fixed the state-space modeling workflow for block-level
LM-HMMs and trial-level GLM-HMMs. The priority was fixing behavior and data
contracts before doing broader readability refactors.

- In `src/behavior_analysis/block_state_space_modeling.py`, added a shared
  valid-block-history mask so block LM-HMM fitting, hardcoded inspection labels,
  and presentation plotting use the same criteria for valid block outcomes and
  previous-block history.
- Clarified the block strategy inheritance flow: hardcoded strategy labels are
  kept only for inspection, while trial-level inheritance propagates the HMM
  `inferred_strategy` and block bias using explicit
  `inherited_block_strategy` and `inherited_block_bias` columns.
- Centralized block model pickle saving with `save_block_model_dict()` so
  `run_block_modeling()` owns the saved `{sess_id_full}_block_statedict.pkl`
  artifact and `map_block_states()` only computes and returns model results.
- In `src/behavior_analysis/trial_state_space_modeling.py`, updated trial
  GLM-HMM preparation to use robust missing-value checks for `prev_action` and
  inherited block strategy labels.
- Added `project_utils.is_zero_flag()` and used it for `give_reward` filtering
  so CSV-loaded string flags such as `"0"` and `"1"` are handled correctly, and
  unexpected present nonnumeric flag values raise a `ValueError`.
- Centralized trial model pickle saving with `save_trial_model_dict()` so
  `run_trial_modeling()` owns the saved `{sess_id_full}_trial_statedict.pkl`
  artifact and `map_trial_states()` no longer writes a duplicate/current-working-
  directory pickle.
- Added tests for block and trial model save ownership, robust missing/flag
  handling, valid-mask behavior, and state-space preparation helpers.
