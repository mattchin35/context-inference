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

# 2026/05/08

Continued cleanup of the block and trial state-space modeling workflow, focused
on reproducibility, clearer user-facing settings, and fixing remaining hidden
trial GLM-HMM data-contract issues.

- Added explicit random-seed handling for block LM-HMM and trial GLM-HMM
  workflows. Seeds are now set from `main.py`, split into deterministic child
  seeds for MLE/MAP fits and IC/CV restarts, and stored in model-selection or
  model-output dictionaries.
- Removed the import-time `np.random.seed(0)` from
  `trial_state_space_modeling.py`. HMM construction and fitting are now wrapped
  by a temporary seed helper because the local `ssm` fork draws random
  transition and observation parameters during `ssm.HMM(...)` construction.
- Made trial GLM-HMM predictor selection explicit in `main.py` via
  `trial_glm_predictor_columns`. Added `time_to_choice` as a valid selectable
  predictor while keeping it out of the module default predictor set.
- Added `make_valid_trial_glm_hmm_mask()` so trial GLM-HMM preparation excludes
  rows with missing `action`, selected predictors, manual rewards, missing
  previous action, or missing inherited block strategy when required. Present
  nonnumeric predictor values still fail loudly during numeric casting.
- Made `mle_trial_states()` explicitly require `inherited_block_strategy`,
  matching the current block/trial comparison diagnostics and replacing a hidden
  `KeyError` with the existing clean missing-column `ValueError`.
- Saved the MLE and MAP block/trial state-comparison plots with explicit
  filenames instead of creating figures without writing them.
- Added focused pytest coverage for seed helpers, trial predictor selection,
  trial GLM-HMM valid-mask behavior, inherited-strategy requirements, and saved
  comparison plots. The full behavior-analysis suite passed after these changes.

# 2026/06/13

Added the first pass of a local Streamlit unit raster/PSTH browser for neural
session inspection. The goal is to browse one selected unit and trial subset at
a time instead of generating an unmanageable number of static spike plots.

- Added `src/neural_analysis/psth_webapp.py` as the user-facing Streamlit entry
  point. It loads one session at a time, lets the user edit session/sorter/aligned
  spike paths, select CT014 channel presets or custom channels, filter units by
  quality label, select a cluster id, and choose condition/action/alignment/page
  settings.
- Added `unit_spike_loading.py` for testable loading support, channel-list
  parsing, CT014 channel presets, cluster metadata filtering, and single-unit
  spike extraction from a Pynapple `TsGroup`.
- Added `unit_spike_plotting.py` for testable Matplotlib raster/PSTH plotting.
  Rasters show the current page of chronologically ordered trials, while PSTHs
  use all filtered trials and are normalized to firing rate in Hz.
- Added plot saving to the session `figures/unit_spike_viewer/` folder with
  filenames that encode session, region, unit, condition, action, alignment, and
  raster page.
- Added `streamlit` to the project dependencies and focused pytest coverage for
  the new loading and plotting helpers. Updated cross-session plotting tests so
  they monkeypatch script-level region/date controls instead of depending on
  editable analysis defaults.

# 2026/06/17

Expanded the local Streamlit spike viewer so it can inspect richer single-trial
and single-unit neural activity without requiring large batches of static plots.
The main goal was to keep the app useful for partially processed sessions while
making probe, LFP, and plot-type choices explicit in the UI.

- Added probe-aware path handling in `psth_webapp.py`, with separate HPC/V1 and
  PFC sorter, aligned-spike, and LFP path inputs. Region selection routes spike
  loading through the appropriate probe, while optional LFP plotting can choose
  directly between the user-entered HPC/V1 and PFC LFP files.
- Added optional single-trial LFP plotting above the combined
  spikes/licks/choices raster. LFP loading is lazy behind a toggle, uses the
  selected saved channel, decodes IRIG sync from digital line 6, applies
  SpikeGLX gain correction, and falls back to raster-only plotting if the LFP
  file or metadata is unavailable.
- Added `lfp_loading.py` for testable LFP metadata loading, sync decoding,
  behavior-time-to-sample mapping, saved-channel extraction, and microvolt
  conversion. The loader documents saved-channel indexing, sample units, time
  units, and output array shapes.
- Extended the single-trial view with a combined behavior/spike raster and
  population PSTH, including both visible-page and all-selected-unit PSTH
  options. Left/right licks are plotted on separate rows, left/right choices are
  drawn on the matching lick rows, and LED onset spans the behavior rows.
- Added alternative single-unit binned firing-rate views using full 100 ms bins
  by default: one view overlays faded per-trial rate traces with the mean, and
  the other plots mean firing rate with a mean +/- standard-deviation band.
  These views reuse the same unit, trial-filter, alignment, and time-window
  controls as the raster/PSTH view and use all filtered trials rather than the
  current raster page.
- Fixed the Streamlit metadata display path so mixed numeric/string unit
  metadata, including manual quality labels such as `mua`, are converted to
  display-safe strings before rendering through `st.dataframe`.
- Added focused pytest coverage for LFP loading helpers, probe/path selection,
  metadata display formatting, single-trial combined plotting, binned
  firing-rate computation and plotting, and plot filenames for alternative
  single-unit views.

# 2026/06/25

Expanded behavior-analysis diagnostics for multisession and single-session
block HMM workflows. The goal was to make block-level strategy, bias, and
session-quality patterns easier to inspect without changing the underlying HMM
analysis assumptions.

- Extended block predicted-state plotting with configurable right-axis traces.
  MLE/MAP block HMM plots can now show no secondary trace, `min_value_bias`, or
  `prev_n_rewarded`, with axis scaling appropriate to the selected trace.
- Added optional session-boundary support and plot-tuning controls for
  multisession block/trial HMM plots, including thinner line widths, wider
  figures, and readable boundary labels placed below the axes.
- Added `trial_input_source` handling for multisession trial HMM workflows so
  trial IC/modeling can load a saved augmented trial CSV with inherited block
  modeling results, instead of requiring block modeling to run first.
- Added cross-session side-specific trials-to-correct quality summaries. The
  multisession workflow now saves side-level summary and raw block-point CSVs
  and plots completion fraction, median trials-to-correct with Q1-Q3 spread,
  raw block points, and explicit no-correct block markers for left and right
  rewarded blocks.
- Added a single-session block quality summary plot. It shows a block timeline
  with numeric trials-to-correct, no-correct side blocks, and dark periods, plus
  a left/right side summary with raw points, medians, and Q1-Q3 intervals.
- Added a block bias quadrant plot that places each block by inference bias
  flag and RL bias flag, with deterministic jitter and block-type colors.
- Added optional sliding block-regression diagnostics underneath block HMM
  predicted-state plots. The diagnostic uses valid block rows, regresses
  `trials_to_correct` on `prev_n_rewarded` in 10-block windows stepped by 5
  blocks, and plots `prev_n_rewarded_weight` on the left axis with
  `window_intercept` on the right axis.
- Added additive RL status columns to saved block-performance tables without
  changing the existing `bias_rl` and `bias_inf` columns. For RL status only,
  blocks with `prev_n_correct == 0` use an effective previous-correct value of
  1, giving that condition a one-trial grace period. New columns include
  `rl_effective_prev_n_correct`, `rl_thresh`, `rl_thresh_flag`,
  `bias_rl_status_value`, and `rl_status`.
- Added focused pytest coverage for the new plotting, multisession summary,
  sliding-regression, and RL-status helpers. The focused HMM/plot/main test
  sets passed. The session-analysis test file could not be run in this
  environment because `formulaic` and `statsmodels` were unavailable, but the
  touched file compiled and the RL-status helper behavior was checked with
  those unavailable imports stubbed.

# 2026/06/26

Added several block-level performance diagnostics for single-session and
multisession behavior analysis. The goal was to separate early block exploration
from later block performance, make across-session block quality easier to scan,
and expose mouse agreement with the greedy mouse-history ideal observer.

- Added post-first-correct block metrics to saved `block_performance` tables:
  `first_correct_trial_in_block`, `n_trials_after_first_correct`, and
  `percent_correct_after_first_correct`. Single-session plots now show this
  metric by block, with distinct markers for blocks where no correct choice
  occurred.
- Added multisession percent-correct-after-first-correct summaries and raw
  block-point CSVs. The plot shows overall, left, and right median/Q1-Q3
  traces, raw block points, and no-correct markers while excluding dark periods.
- Added multisession block-switch quality summaries using raw `n_switches`.
  The plot shows overall, left, and right median/Q1-Q3 traces with underlying
  raw block points.
- Generalized the learning-curve input so `main_multisession()` now requests
  `prev_n_rewarded_slope`. The code fails clearly if the requested regressor
  column is missing from the overall-performance CSV, which should catch stale
  session summaries.
- Added block-level `block_history_ideal_mouse_agreement`, computed as the
  fraction of valid behavioral-choice trials where the mouse action matches the
  greedy mouse-history ideal-observer action. A new single-session plot shows
  this by block, with distinct markers for blocks that have no valid ideal
  comparisons.
- Added focused tests for the new plotting and multisession dataframe helpers.
  The focused performance-plot and multisession test sets passed, and the
  session-analysis metric was smoke-tested with unavailable stats imports
  stubbed because `formulaic` is not installed in this environment.

# 2026/06/28

Added reusable block-HMM outputs and expanded mouse-agent agreement diagnostics
for single-session and multisession behavior checks. The goal was to make
exploratory plotting faster while comparing mouse behavior against several
candidate model strategies without rerunning expensive HMM fits unnecessarily.

- Added a file-presence block-HMM reuse flag for single-session and
  multisession workflows. When `skip_block_hmm_if_existing` is enabled, the
  workflow reuses existing `{sess_id_full}_block_statedict.pkl`,
  `{sess_id_full}_block_performance.csv`, and
  `{sess_id_full}_augmented_trials.csv`; if any required file is missing, the
  block HMM runs normally and prints the missing paths.
- Added blockwise mouse-agent agreement metrics for Q-learning, forgetting
  Q-learning, HMM log-odds, HMM log-odds with decay, and the mouse-history
  ideal observer. Greedy actions use the existing ideal-observer sign
  convention and repeat-on-tie rule, so exact zero values repeat the previous
  greedy action.
- Added a single-session mouse-agent agreement figure. The top panel shows
  blockwise agreement lines for `QL`, `FQL`, `HMM`, `HMM decay`, and `Ideal`;
  the bottom panel shows raw block values with median and Q1-Q3 summaries for
  each agent.
- Added a standalone multisession mouse-agent agreement quality plot with
  median, Q1-Q3, and raw block points across sessions. The multisession
  workflow now saves `{mouse}_agent_mouse_agreement_summary.csv`,
  `{mouse}_agent_mouse_agreement_block_points.csv`, and
  `{mouse}_agent-mouse-agreement-quality.png`.
- Made the multisession agent-agreement prep robust to older block CSVs. If
  saved block-performance files lack the new agreement columns, the workflow
  regenerates them from saved augmented trial features; if those trial feature
  columns are also missing, it fails loudly and asks for single-session trial
  feature collection to be rerun.
- Updated the agent-agreement plots to use short visible labels and an explicit
  bolder palette, avoiding weak default colors such as yellow. The same palette
  is shared by the single-session and multisession agent-agreement plots.
- Added focused pytest coverage for HMM output reuse, agent-agreement trial and
  block summaries, robust multisession regeneration, and the new
  single-session/multisession agent-agreement plots. The affected
  multisession and performance-plot test modules passed, along with compile
  checks for touched modules.

# 2026/06/30

Expanded the single-session summary CSV metrics so session-level behavior
quality can be scanned without manually recomputing values from trial and block
tables.

- Added signed side-bias metrics to the session summary:
  `bias_oracle`, `bias_ideal`, and `raw_side_bias`. These use valid behavioral
  choice rows, with `bias_ideal` restricted to rows with a valid ideal-agent
  greedy choice.
- Added block-derived session summary metrics for trials to switch,
  post-first-correct correctness, and mouse-history ideal-agent agreement:
  `median_TTS`, `q3_TTS`, `frac_blocks_TTS_gt_5`,
  `median_post_switch_correct`, `q1_post_switch_correct`,
  `frac_blocks_post_switch_correct_lt_0p7`, `median_ideal_agreement`,
  `q1_ideal_agreement`, and `frac_blocks_ideal_agreement_lt_0p6`.
- Added post-first-correct ideal-agent agreement summaries:
  `median_post_switch_ideal_agreement` and
  `q1_post_switch_ideal_agreement`. These include the first correct choice and
  exclude blocks without eligible valid comparisons.
- Increased the visible slope label size on the single-session
  trials-to-switch regression figures.
- Added focused tests for the new session summary helpers and the slope-label
  size check.
