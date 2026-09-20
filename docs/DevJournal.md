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

# 2026/07/23

Expanded the block-level behavior-analysis pipeline so saved CSVs are easier to
inspect in R while carrying richer block outcome, residual-model, and exemplar-
model information through the single-session, multisession, and cross-mouse
workflows. The main design goal was to keep Python as the canonical computation
path, while making the resulting tables self-describing enough that exploratory
R analysis can load them directly and select interpretable subsets or model
summaries without recomputing hidden state.

- Refactored R source loading for scripts under `src/Rcode` so preprocessing
  helpers can be sourced relative to the R project root instead of the current
  working directory. This made `inspect_trials.R` and `inspect_blocks.R` more
  robust when run from RStudio's Source button or from `Rscript`.
- Updated trial preprocessing to match the current augmented-trial schema. The
  R cleanup path now handles the new model-value columns and current behavioral
  choice conventions, and trial-index flags such as `block_entry_trial`,
  `first_switch_in_block`, `switch_trial`, `stay_trial`, `explore_trial`, and
  `block_entry_explore_trial` are generated from the augmented trial dataframe
  and ported to leave-stay trial-value CSVs for consistent filtering.
- Added detailed block-level switch and transition metrics to the Python block
  performance tables. New metrics include TTS/no-switch handling,
  post-switch correctness, transition width, reversion choices/events,
  terminal omission streaks, previous-block reward load, session-relative TTS
  residuals, within-session TTS percentile ranks, and QC flags for no-switch or
  very short post-switch blocks. These metrics preserve the existing behavioral
  side conventions and retain the first block behavior expected by the older
  pipeline.
- Updated R block preprocessing for current single-session, multisession, and
  cross-mouse block-performance CSVs. The cleaner now treats centered reward
  columns (`prev_rewards_session_centered`, `prev_rewards_mouse_centered`, and
  `prev_rewards_global_centered`), `block_side_code`, residual model columns,
  exemplar residual columns, and mouse/session metadata columns with the correct
  numeric or factor types. Filtering of missing TTS remains opt-in so no-switch
  blocks can be inspected instead of silently dropped.
- Added regularized block residual models for session-level block performance.
  The canonical Python path and the R exploratory helper now fit lasso and
  elastic-net versions of both `trials_to_correct ~ prev_n_rewarded * side` and
  `trials_to_correct ~ prev_n_rewarded + side`. Outputs include per-block raw
  residuals, session-level residual spread summaries, and per-session model
  summary CSVs. The original interaction-model residual column names are kept
  as aliases, while formula-specific names make additive and interaction models
  explicit.
- Made the residual model summaries directly interpretable by saving
  right-side and left-side intercepts/slopes as derived columns, plus
  `average_intercept` and `average_reward_slope`. The raw right-reference model
  terms are still preserved, but the derived columns make it easier to plot
  side-specific reward-load effects without re-deriving contrasts in R.
- Expanded residual spread summaries beyond MAD/IQR/RMSE by adding sample
  standard deviation (`residual_sd`). The Python implementation uses
  `np.std(..., ddof=1)` to match R's `sd()`, and returns the project missing
  sentinel when fewer than two residuals are available. The new metric now
  propagates into session summaries and residual-model summary CSVs.
- Added mouse-level and cross-mouse collection for residual model summary CSVs,
  separate from the existing `overall_performance.csv`. Mouse-level outputs are
  saved under each mouse's `cross_session_analysis` directory, and cross-mouse
  outputs are saved under `cross_mouse_analysis`. Interaction and additive
  formulas are collected into separate formula-specific CSVs so downstream
  plots do not accidentally mix model families.
- Added code-defined block exemplar models as a separate concept from fitted
  residual models. `block_exemplar_models.py` now defines editable exemplar
  specs with `name`, `intercept`, `reward_slope`, `residual_sd`, and
  `description`. For each block, the pipeline saves only the raw residual and
  normalized residual for each exemplar, using
  `trials_to_correct - (intercept + reward_slope * prev_n_rewarded)` and
  division by the exemplar residual SD. Predictions and absolute residuals are
  intentionally not saved to keep the block CSV focused.
- Saved exemplar model parameter CSV and JSON files for each session so the
  exact code-defined exemplar parameters used for a processed session can be
  inspected later. The exemplar parameters are not added to
  `overall_performance.csv`; they are block-level residual diagnostics plus
  session-local parameter metadata.
- Added R exploratory examples in `inspect_blocks.R` for loading CT024
  cross-session residual-model summary CSVs, filtering to one model family such
  as elastic net with `lambda.min`, and plotting right/left or average reward
  slopes over training day. The intended workflow is now to load precomputed
  Python outputs in R, filter by `model_formula`, `model_type`, and
  `lambda_choice`, and inspect the saved interpretable columns directly.
- Added focused Python and R tests across the new behavior. The test coverage
  now includes residual model fitting and summary rows, formula-specific CSV
  saving and collection, R/Python residual-spread parity, R block-preprocessing
  coercion for new schemas, exemplar model validation and residual calculation,
  save-analysis integration, and parameter CSV/JSON writing.

# 2026/07/27

Added first-pass Dynamax-based HMM tooling for block exemplar model diagnostics.
The main goal was to classify blocks against the already defined exemplar
models without fitting new emission models, so the resulting states remain
interpretable as exemplar names rather than arbitrary latent Gaussian clusters.

- Added `dynamax` as a project dependency and verified the local HMM API,
  including `GaussianHMM` and low-level `hmm_smoother` support.
- Added an exploratory fitted Gaussian-HMM path using the four normalized
  exemplar residual columns as a 4D emission vector. This path was kept
  available as example/inspection code, but it is not the preferred modeling
  workflow because its hidden states are learned clusters rather than fixed
  exemplar identities.
- Added the fixed-exemplar likelihood workflow in
  `exemplar_hmm_modeling.py`. It loads the saved exemplar parameter CSV for
  model names and residual spreads, uses the existing raw residual columns to
  compute Gaussian log likelihoods, reads the existing normalized residual
  columns for output and plotting, computes independent likelihood-normalized
  probabilities, and smooths the fixed likelihoods with Dynamax
  `hmm_smoother`.
- Saved interpretable block-level outputs including
  `exemplar_z_residual_<model>`, `exemplar_log_likelihood_<model>`,
  `exemplar_independent_prob_<model>`,
  `exemplar_hmm_smoothed_prob_<model>`, `exemplar_ll_argmax_model`, and
  `exemplar_hmm_smoothed_model`.
- Added diagnostic plots with four panels: normalized residual traces, fixed
  exemplar log likelihoods, independent likelihood argmax assignments, and
  HMM-smoothed exemplar assignments. The categorical panels use exemplar model
  names as y-axis labels.
- Added one-mouse multisession support for cross-session block-performance
  CSVs, with a hardcoded `main_multisession()` entry point and optional
  session-boundary markers derived from `source_date` or `source_session_id`
  after valid-row filtering.
- Added focused pytest coverage for fixed likelihood scoring, sticky
  transition matrix construction, Dynamax smoothing, output-column creation,
  diagnostic plotting, loadable pickle payloads, and multisession boundary
  handling.

# 2026/08/25

Reached the user-approved Power inspection gate for the new cached LFP summary
pipeline and prepared the next Synchrony implementation stage. The governing
documents are `docs/Tasks_neural.md` and `docs/webappDesign.md`; their plan was
approved before implementation. Work is on branch `refactor`, and the user
pushed the branch after approving the revised CT026 Power figures.

Completed implementation state:

- Restored the neural-analysis baseline and removed the duplicate public
  Hilbert plotting definition.
- Added immutable configuration/fingerprint models, safe NPZ/JSON cache I/O,
  component schemas, shared trial/LFP preparation, numerical Power,
  Synchrony, and spike-phase summary functions, pure Matplotlib plotting,
  component pipeline orchestration, and the Streamlit route boundary.
- Added the production Power runtime and CT026 validation runner. CT026 uses
  session `CT026_2026-08-01_130853`, PFC ProbeA channel 5, HPC1 ProbeB channel
  222, and HPC2 ProbeB channel 14. Open Ephys trial loading preserves exact
  fractional alignment and the cached 500 Hz source traces use anti-aliased
  resampling.
- The current Power cache is under the session's
  `processed/lfp_summary_cache` directory. Its main arrays have 3 sites, 427
  trials, 3 epochs, a canonical 2 Hz PSD grid, and exact 500 Hz `[-2, 2)`
  source traces. Three trial rows (114, 229, and 284) are invalid because
  `choice_time` is missing; no amplitude-based exclusion was silently applied.
- The user requested Tukey box plots instead of median/IQR bars and separate
  figures for the base five conditions versus omission/incorrect
  subdivisions. This is implemented in commits `716723f`, `1d8ec4b`, and
  `a89d245`. Box plots use 1.5-IQR whiskers and hide outlier markers. PSD and
  band captions state that the condition masks overlap. The cached web Power
  preview defaults to the base five conditions.
- The approved cache-only CT026 report is
  `analysis_runs/CT026_2026-08-01_130853_lfp_power_validation_2026-08-25T16-17-40Z`
  beneath the session root. It contains 12 visually inspected PNGs: base and
  subdivision PSD and band-power plots for PFC, HPC1, and HPC2. Its metadata
  records `power_recomputed=false`; it reused the compatible cache and carried
  forward the measured 10.494185874 second runtime and 586289152 byte peak
  memory value.
- The latest complete neural-analysis command was
  `UV_CACHE_DIR=/tmp/context_inference_uv_cache MPLCONFIGDIR=/tmp/mpl uv run
  pytest -q -p no:cacheprovider src/tests/neural_analysis`, which passed 597
  tests with 16 pre-existing Pynapple empty-epoch/divide-by-zero warnings.

The next approved gate is production Synchrony on the same CT026 session. The
low-level numerical code and cache schema exist, but the production bridge is
not implemented: `lfp_summary_runtime.make_power_pipeline_dependencies()`
still installs explicit unsupported phase/Synchrony seams, and
`lfp_summary_webapp.make_production_summary_dependencies()` still reports
production Synchrony as unavailable. Another agent should resume with strict
test-first work rather than running CT026 immediately:

1. Add and commit RED tests for a production phase-preparation object,
   Synchrony payload builder/factory, web action, and immutable cache-only
   Synchrony validation report. Tests must use injected tiny loaders and must
   prove Synchrony does not recompute Power or invoke spike-phase work.
2. Reuse `lfp_phase_clustering.compute_site_phase_trial_tensor()` for bounded
   continuous transforms (120 second cores with Morlet edge padding), and keep
   the configured 2-100 Hz, 2 Hz-spaced phase frequencies. Interpolate complex
   real/imaginary coefficients onto the exact 500 Hz grid and normalize; never
   interpolate wrapped phase angles.
3. Preserve the full 427-row trial axis with per-site validity. Do not use
   `combine_phase_trial_tensors()`, because its all-site intersection would let
   a missing HPC2 trial incorrectly remove an otherwise valid PFC-HPC1 trial.
   ITPC is site-specific and ISPC/PLV validity is pair-specific.
4. Assemble every array in `SYNCHRONY_ARRAY_SCHEMA`, including ITPC/ISPC maps
   and counts, signed A-minus-B offsets, direct whole/before/after PLV and 80
   percent coverage diagnostics, disjoint gamma handling, bootstrap summaries,
   and cached source/band/Hilbert exemplar traces. Wavelet tensors must remain
   temporary and must not be written to the cache.
5. Bootstrap ITPC/ISPC by resampling selected trial positions within each
   condition and recomputing the scalar band/window phase-clustering statistic.
   Do not substitute a bootstrap over already-aggregated values. Use exactly
   the configured seeded 1,000 resamples and retain under-10 instability flags.
6. Add a timestamped Synchrony validation report outside the repository with
   maps plus effective counts, theta/gamma summaries with confidence intervals,
   PLV distributions/coverage, deterministic high/low exemplars, manifest and
   configuration snapshots, log, source identifiers, runtime, peak memory,
   cache/component sizes, exclusions, and warnings. Run it only after all
   focused and full tests pass, then pause for user inspection before any
   spike-phase preview.

Performance needs explicit attention. A dense temporary phase tensor for 3
sites x 50 frequencies x 427 trials x 2000 samples is about 1.0 GB as
`complex64`, before its validity mask and temporary reductions. Keep continuous
wavelet calculation block-bounded, avoid accidental `complex128`/`float64`
copies of the entire tensor, and report observed memory/time rather than adding
an arbitrary cutoff. The 1,000-resample bootstrap may dominate runtime and
should be implemented in bounded chunks while preserving the exact statistic.

Worktree caution: `docs/Tasks_neural.md` and `docs/webappDesign.md` were already
modified by the user and must not be reverted. The repository also contains
many unrelated untracked files. Stage only files belonging to the current LFP
summary task. Two Terra Medium handoff attempts on 2026/08/25 ended before
editing because their service usage limit was reached, so there are no partial
Synchrony test or source changes to recover from those attempts.

Synchrony continuation completed later on 2026/08/25:

- Added and committed the production full-trial phase preparation, exact
  Synchrony payload, Synchrony-only pipeline factory, seeded trial-resampling
  bootstrap, and CT026 cache-backed validation/report runner. The main commits
  are `a432aa1`, `ea542d1`, `fa19ad0`, and `f312957`; subsequent test-first
  plotting/report fixes end at `cc2fd81`.
- Phase preparation uses bounded continuous transforms, preserves complex64
  storage on `(site, frequency, trial, time)`, retains the full 427-row trial
  axis, and computes site- and pair-specific validity without an all-site
  intersection. The cached component contains no wavelet tensor.
- The CT026 Synchrony computation completed successfully and atomically added
  `synchrony.npz` beside the existing Power cache. The Synchrony component is
  153572688 bytes; the complete generic cache is 232275359 bytes. Three trials
  are unavailable at every site because their alignment event is missing,
  yielding site exclusion counts `[3, 3, 3]` and pair exclusion counts
  `[3, 3, 3]`. No ITPC/ISPC band summary has fewer than 10 contributing trials.
- Observed compute time was approximately 642.487 seconds (10.7 minutes),
  derived from the run start and manifest completion timestamps. The process
  peak RSS was not captured before a post-compute plotting exception, so the
  final report explicitly marks peak memory unavailable instead of claiming a
  zero-byte measurement. The nine-filter linear projection is about 5782.383
  seconds and 1382154192 component bytes.
- Real-data plotting exposed three issues that synthetic tests had missed:
  percentile bootstrap intervals need not contain their point estimate; a
  caption must not enumerate hundreds of per-trial sample counts; and combining
  3 sites, 3 pairs, and 9 conditions makes a band-summary plot unreadable.
  Corrective RED tests were committed first. Intervals are now drawn as direct
  ranges, PLV captions show per-epoch count/min/median/max diagnostics, reports
  publish from a hidden staging directory only after all artifacts succeed,
  and band summaries are split into per-site ITPC and per-pair ISPC figures.
- The final report to inspect is
  `analysis_runs/CT026_2026-08-01_130853_lfp_synchrony_validation_2026-08-25T23-37-18Z`
  beneath the CT026 session root. It contains 252 PNGs plus manifest,
  configuration, source-identifier, log, and Markdown snapshots. Earlier
  Synchrony report directories from this date are partial or superseded and
  should not be used for approval.
- The final full neural-analysis command passed 609 tests with the same 16
  pre-existing Pynapple empty-epoch/divide-by-zero warnings. The next required
  action is user inspection/approval of the final Synchrony report. Do not run
  the Spike-phase preview until that approval. Production webapp dependency
  wiring for Synchrony remains a later integration task; the validation runner
  and component cache are complete and independently callable.

# 2026/08/26

Refined the CT026 ITPC/ISPC map presentation after user inspection. The
effective-trial heatmaps were numerically constant and therefore uninformative:
their intended purpose was to expose frequency/time-specific missingness, but
all retained CT026 trials contribute at every displayed bin.

- Replaced the second effective-count heatmap with a compact annotation on the
  ITPC/ISPC map. Constant coverage is shown as
  `Effective/total displayed: effective/total`; variable coverage is shown as
  `Effective range/total displayed: minimum-maximum/total`.
- The denominator is the condition/filter trial count before site or pair
  validity. The numerator remains the actual contributor count from the cached
  frequency-by-time count array. Invalid, fractional, negative, or
  greater-than-total counts fail clearly.
- Added and committed RED tests for constant/ranged counts, validation-runner
  propagation, removal of the count axis, and sufficient one-panel caption
  spacing. Implementation commits are `3bf3938` and `811e47d`.
- The full neural-analysis suite still passes 609 tests with the same 16
  pre-existing Pynapple warnings. No numerical cache or phase transform was
  recomputed.
- The latest report for approval is
  `analysis_runs/CT026_2026-08-01_130853_lfp_synchrony_validation_2026-08-26T16-57-33Z`
  beneath the CT026 session root. Earlier Synchrony reports are superseded.
  Continue to pause before the Spike-phase preview until the user approves
  this report.

# 2026/08/27

Audited the complete `src/neural_analysis` module inventory, the current LFP
summary implementation, its tests, `docs/Tasks_neural.md`, and
`docs/webappDesign.md` before continuing experimental validation. No source or
cache files were changed during the audit.

Current verified development state:

- Power remains implemented, cached, and user-approved for session
  `CT026_2026-08-01_130853`. The approved report remains
  `analysis_runs/CT026_2026-08-01_130853_lfp_power_validation_2026-08-25T16-17-40Z`.
- Production Synchrony calculation and cache-backed reporting are implemented.
  The current Synchrony cache remains compatible, and the latest report remains
  `analysis_runs/CT026_2026-08-01_130853_lfp_synchrony_validation_2026-08-26T16-57-33Z`.
  The development record does not yet contain an explicit user approval of
  that report, so approval must be confirmed rather than inferred before a
  production Spike-phase preview is executed.
- The production Spike-phase runtime bridge, 100-shuffle CT026 preview
  configuration, active ProbeB unit-population selection, cache-only plotting,
  and immutable preview-report writer are now implemented in the nine local
  commits after `origin/refactor`, ending at commit `c036d62`.
- The Spike-phase implementation has not yet been run on CT026. The generic
  cache currently contains only `power.npz`, `synchrony.npz`, and
  `manifest.json`; there is no `spike_phase.npz` and no Spike-phase preview
  analysis-run directory. The final 1,000-shuffle run has therefore not been
  attempted.
- The complete neural-analysis test command was rerun at current HEAD:
  `UV_CACHE_DIR=/tmp/context_inference_uv_cache MPLCONFIGDIR=/tmp/mpl uv run
  pytest -q -p no:cacheprovider src/tests/neural_analysis`. It passed 617 tests
  with the same 16 pre-existing Pynapple empty-epoch/divide-by-zero warnings.
  No neural tests were skipped or marked xfail.
- Branch `refactor` is nine commits ahead of `origin/refactor`. The existing
  user modifications to `docs/Tasks_neural.md` and `docs/webappDesign.md`, and
  unrelated untracked files, remain untouched.

Important incomplete integration work:

- The Streamlit production dependency factory still wires only Power.
  Synchrony, Spike phase, and Compute All deliberately return unavailable, and
  cached plotting supports only the limited Power preview. The main webapp
  route does not yet provide an active unit population to the summary view.
- No composed production dependency bundle currently supports Compute All.
  The component-specific runtime factories intentionally reject operations
  belonging to the other components.
- Pipeline progress callbacks are not connected to Streamlit progress output.
- The first-pass Spike preview report renders four representative cached PPC
  figures. It does not yet provide the complete planned high/low percentile
  exemplar set, detailed run log, exclusion/warning summary, or nine-filter
  performance projection.
- Configured per-site absolute amplitude thresholds are validated and
  fingerprinted but are not applied by production phase preparation. PPC
  `worker_count` and `chunk_size` are also validated configuration fields but
  are not used by the production PPC calculation.
- Real-session Spike-phase wall time, peak memory, cache size, and unit/spike
  reliability counts remain unknown. The passing synthetic and injected tests
  do not substitute for the gated CT026 100-shuffle run.

The next safe milestone is to confirm user approval of the latest Synchrony
report, then run only the CT026 100-shuffle Spike-phase preview for the approved
active ProbeB population. The resulting cache, plots, counts, exclusions,
runtime, memory use, and storage must be inspected before authorizing the final
1,000-shuffle calculation or broader webapp integration.

# 2026/08/27 - Corrective handoff

This entry appends a correction without rewriting the historical entry above.
The earlier instruction to run the CT026 100-shuffle Spike-phase preview next is
superseded. The user approved both the latest Synchrony report and the revised
ordering in which WP5C is completed and benchmarked before that preview.

- Current branch/HEAD is `refactor` at
  `c036d62b9567db3bd66ce8ad84d75a089a335ad8`, nine commits ahead of
  `origin/refactor` at `e580252`.
- The full neural suite at this HEAD is 617 passed with no skipped or xfailed
  tests and the same 16 known Pynapple empty-epoch/divide-by-zero warnings.
- The user explicitly approved the final Synchrony report
  `analysis_runs/CT026_2026-08-01_130853_lfp_synchrony_validation_2026-08-26T16-57-33Z`
  on 2026-08-27.
- WP5C remains unimplemented. Its scientific design and its position before the
  preview are approved, but its proposed internal execution contracts and
  implementation remain unauthorized.
- The CT026 generic cache contains `power.npz`, `synchrony.npz`, and
  `manifest.json`. It contains no `spike_phase.npz`, and no CT026 Spike-phase
  preview report directory exists.
- The exact next gate is user review and explicit approval of the WP5C-0
  contracts in `docs/Tasks_neural.md` Sections 2.22-2.26. Only after that gate
  may WP5C-1 write its test-only commit and record RED. No CT026 computation is
  authorized by contract approval alone.

This journal is historical. `docs/Tasks_neural.md` is the authoritative source
for current execution state, contracts, gates, package ownership, and remaining
work; later work must not recover sequencing from an older journal entry.

# 2026/09/20 - ProbeB preview result and recovery/cluster handoff

The first local CT026 ProbeB 100-shuffle launcher run ended after successful
numerical publication but before report publication. No source or numerical
parameter was changed during diagnosis.

- Run directory:
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/CT026_2026-08-01_130853_spike_phase_ProbeB_preview_2026-09-18T20-36-32Z`.
- All 105 grouped blocks completed. The launcher reached
  `component_complete`, and the generic manifest marks `spike_phase.npz`
  complete. The component is 211876438 bytes and represents 273 ProbeB units,
  427 trials, nine overlapping conditions, three sites, three epochs, and 50
  frequencies.
- The launcher then failed with `bottom cannot be >= top`. The population plot
  caption serialized complete `9 x 50` eligible and total count matrices. It
  produced 4789 characters, 36 wrapped lines, and a requested subplot bottom
  margin of 1.71. This is a deterministic report-layout failure rather than a
  PPC or cache failure.
- The report parent contains no published report. Success cleanup did not run,
  and the exact 1.3 GB PPC work directory remains available. The supplied
  ordinary resume command would repeat the same error at the original code
  commit; a new plotting commit would be rejected by ordinary exact-commit
  resume.
- Measured component time was 11719.959 seconds: 115.086 seconds phase
  preparation, 3653.792 seconds planning, and 7940.727 seconds grouped
  execution. Peak process-tree PSS was 4146417664 bytes and RSS was 7690588160
  bytes. The earlier 45-120 minute projection for 1,000 shuffles is retired.
- Preview diagnostics include 1080900 computable cells, 914700 reliable/null-
  eligible cells, and 167860 significant eligible cells. With 100 shuffles the
  smallest attainable p/q value is `1/101`; these counts are validation
  evidence rather than the final scientific result.

The user approved a documentation-first R1 package. The population view will
show three heatmaps: reliable-unit median PPC, fraction of eligible units
passing FDR, and exact eligible-unit count. An explicit `recover-report`
launcher command will preserve strict ordinary resume identity while allowing
only report generation under a later clean descendant commit. It must validate
the original component/configuration/population/sources, record computation and
report commits separately, make every numerical seam unreachable, validate the
report, and clean only the saved exact target last.

After R1 report inspection, work moves directly to C1 cluster execution before
WP11. The pushed baseline is `ce37409`, which tracks `pyproject.toml`, `uv.lock`,
and the sample `src/shell_scripts/hpc_ppc.sh`; uv is installed on cluster
access. The user selected Slurm partition `unlimited` and an initial 72-hour
request. The production wrapper will use one task, eight CPUs, 32 GB, a
five-minute SIGTERM warning, offline/no-sync/frozen uv execution, and the
existing launcher. A harmless uv/Python hello-world check precedes project
imports, wrapper implementation, or scientific access.

Numerical caches will remain on the cluster. The approved inspection topology
runs Streamlit beside the cache and binds it to loopback; a local browser uses
SSH port forwarding. This avoids SSHFS latency and absolute-path remapping,
while transferring only figures and scalar UI output. Immutable reports may be
copied locally. `docs/Tasks_neural.md` contains the authoritative R1/C1 tests,
state rules, commands, resource policy, and execution gates.
