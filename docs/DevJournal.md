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
