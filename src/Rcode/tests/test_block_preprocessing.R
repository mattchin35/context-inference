get_test_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    stop("Could not determine test script path from commandArgs().")
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE)
}

# Build a small current-schema block dataframe for cleanup tests.
#
# Returns:
#   data.frame with shape (3 blocks, n_columns). Values intentionally use CSV
#   string conventions from current block-performance outputs, including
#   "None" missing sentinels and "True"/"False" flags.
make_current_block_dataframe <- function() {
  data.frame(
    block_ix = c("0", "1", "2"),
    block_type = c("right_uncued", "left_uncued", "right_uncued"),
    trials_to_correct = c("1", "None", "3"),
    first_correct_trial_in_block = c("1", "None", "3"),
    n_trials_after_first_correct = c("2", "None", "4"),
    percent_correct_after_first_correct = c("1.0", "None", "0.75"),
    block_history_ideal_mouse_agreement = c("0.5", "0.25", "None"),
    prev_consecutive_rewards = c("0", "2", "4"),
    prev_consecutive_rewards_memory = c("0", "1", "3"),
    prev_n_correct = c("0", "2", "3"),
    prev_n_rewarded = c("0.0", "2.0", "1.0"),
    prev_rewards_session_centered = c("-1.0", "1.0", "None"),
    prev_rewards_mouse_centered = c("-2.0", "0.0", "2.0"),
    prev_rewards_global_centered = c("-3.0", "None", "3.0"),
    block_side_code = c("-0.5", "0.5", "None"),
    previous_block_length = c("None", "5", "6"),
    previous_block_reward_fraction = c("None", "0.4", "0.5"),
    no_switch = c("False", "True", "False"),
    transition_width = c("0", "None", "5"),
    reversion_choices = c("0", "None", "3"),
    reversion_events = c("0", "None", "2"),
    terminal_omission_streak_inclusive = c("1", "None", "4"),
    short_or_low_postswitch_trials = c("True", "True", "False"),
    n_switches = c("1", "0", "2"),
    n_explore_trials = c("0", "0", "1"),
    normalized_switches = c("0.5", "0.0", "0.25"),
    confusion_flag = c("False", "False", "True"),
    n_correct = c("2", "0", "5"),
    percent_correct = c("0.66", "0.0", "0.83"),
    n_rewarded = c("1.0", "0.0", "4.0"),
    mean_choice_time = c("0.44", "0.55", "0.66"),
    median_choice_time = c("0.45", "0.56", "0.67"),
    std_choice_time = c("0.10", "0.20", "0.30"),
    session_ID = c("CT024_2026-06-09", "CT024_2026-06-09", "CT024_2026-06-09"),
    predicted_TTS_from_prev_rewards = c("1.5", "None", "2.5"),
    residual_TTS = c("-0.5", "None", "0.5"),
    TTS_percentile_within_session = c("0.25", "None", "1.0"),
    bias_rl = c("1.0", "None", "0.0"),
    bias_inf = c("-0.6", "None", "-0.1"),
    min_value_bias = c("-0.6", "None", "-0.1"),
    bias_rl_flag = c("True", "None", "False"),
    bias_inf_flag = c("False", "None", "False"),
    bias_full_flag = c("False", "None", "False"),
    rl_effective_prev_n_correct = c("1", "None", "3"),
    rl_thresh = c("0", "None", "1"),
    rl_thresh_flag = c("True", "None", "True"),
    bias_rl_status_value = c("0.0", "None", "0.25"),
    rl_status = c("valid_rl", "None", "below_rl_threshold"),
    prev_n_omissions = c("0", "1", "2"),
    prev_consecutive_omissions = c("0", "1", "2"),
    qlearning_mouse_agreement = c("0.66", "None", "0.75"),
    fql_mouse_agreement = c("0.66", "None", "0.75"),
    hmm_logodds_mouse_agreement = c("0.66", "None", "0.75"),
    hmm_logodds_decay_mouse_agreement = c("0.66", "None", "0.75"),
    perseveration_mouse_agreement = c("0.33", "None", "0.25"),
    doubt_perseveration_mouse_agreement = c("0.33", "None", "0.25"),
    wsls_mouse_agreement = c("0.66", "None", "0.75"),
    observer_mouse_agreement = c("0.66", "None", "0.75"),
    n_explore_runs = c("0", "0", "1"),
    lasso_lambda_min_residual_TTS = c("0.1", "None", "-0.1"),
    lasso_lambda_1se_residual_TTS = c("0.2", "None", "-0.2"),
    elastic_net_lambda_min_residual_TTS = c("0.3", "None", "-0.3"),
    elastic_net_lambda_1se_residual_TTS = c("0.4", "None", "-0.4"),
    lasso_rewards_x_side_lambda_min_residual_TTS = c("0.5", "None", "-0.5"),
    lasso_rewards_x_side_lambda_1se_residual_TTS = c("0.6", "None", "-0.6"),
    elastic_net_rewards_x_side_lambda_min_residual_TTS = c("0.7", "None", "-0.7"),
    elastic_net_rewards_x_side_lambda_1se_residual_TTS = c("0.8", "None", "-0.8"),
    lasso_rewards_plus_side_lambda_min_residual_TTS = c("0.9", "None", "-0.9"),
    lasso_rewards_plus_side_lambda_1se_residual_TTS = c("1.0", "None", "-1.0"),
    elastic_net_rewards_plus_side_lambda_min_residual_TTS = c("1.1", "None", "-1.1"),
    elastic_net_rewards_plus_side_lambda_1se_residual_TTS = c("1.2", "None", "-1.2"),
    mouse = c("CT024", "CT024", "CT024"),
    source_mouse = c("CT024", "CT024", "CT024"),
    session_id = c(
      "CT024_2026-06-09_143852",
      "CT024_2026-06-09_143852",
      "CT024_2026-06-09_143852"
    ),
    source_session_id = c(
      "CT024_2026-06-09_143852",
      "CT024_2026-06-09_143852",
      "CT024_2026-06-09_143852"
    ),
    date = c("2026-06-09", "2026-06-09", "2026-06-09"),
    source_date = c("2026-06-09", "2026-06-09", "2026-06-09"),
    stringsAsFactors = FALSE
  )
}

test_script_path <- get_test_script_path()
rcode_root <- normalizePath(
  file.path(dirname(test_script_path), ".."),
  winslash = "/",
  mustWork = TRUE
)

block_env <- new.env(parent = globalenv())
source(file.path(rcode_root, "preprocessing", "block_preprocessing.R"), local = block_env)
stopifnot(exists("clean_block_dataframe", envir = block_env, inherits = FALSE))

block_df <- make_current_block_dataframe()
cleaned_df <- block_env$clean_block_dataframe(block_df)

stopifnot(nrow(cleaned_df) == 3)
stopifnot(!"inferred_strategy" %in% names(cleaned_df))
stopifnot(!"declared_strategy" %in% names(cleaned_df))
stopifnot(!"cur_strategy_slope" %in% names(cleaned_df))
stopifnot(is.na(cleaned_df$trials_to_correct[2]))
stopifnot(identical(cleaned_df$no_switch, c(FALSE, TRUE, FALSE)))
stopifnot(identical(cleaned_df$short_or_low_postswitch_trials, c(TRUE, TRUE, FALSE)))
stopifnot(is.na(cleaned_df$bias_rl_flag[2]))

numeric_columns <- c(
  "block_ix",
  "trials_to_correct",
  "first_correct_trial_in_block",
  "n_trials_after_first_correct",
  "percent_correct_after_first_correct",
  "block_history_ideal_mouse_agreement",
  "prev_consecutive_rewards",
  "prev_consecutive_rewards_memory",
  "prev_n_correct",
  "prev_n_rewarded",
  "prev_rewards_session_centered",
  "prev_rewards_mouse_centered",
  "prev_rewards_global_centered",
  "block_side_code",
  "previous_block_length",
  "previous_block_reward_fraction",
  "transition_width",
  "reversion_choices",
  "reversion_events",
  "terminal_omission_streak_inclusive",
  "n_switches",
  "n_explore_trials",
  "normalized_switches",
  "n_correct",
  "percent_correct",
  "n_rewarded",
  "mean_choice_time",
  "median_choice_time",
  "std_choice_time",
  "predicted_TTS_from_prev_rewards",
  "residual_TTS",
  "TTS_percentile_within_session",
  "bias_rl",
  "bias_inf",
  "min_value_bias",
  "rl_effective_prev_n_correct",
  "rl_thresh",
  "bias_rl_status_value",
  "prev_n_omissions",
  "prev_consecutive_omissions",
  "qlearning_mouse_agreement",
  "fql_mouse_agreement",
  "hmm_logodds_mouse_agreement",
  "hmm_logodds_decay_mouse_agreement",
  "perseveration_mouse_agreement",
  "doubt_perseveration_mouse_agreement",
  "wsls_mouse_agreement",
  "observer_mouse_agreement",
  "n_explore_runs",
  "lasso_lambda_min_residual_TTS",
  "lasso_lambda_1se_residual_TTS",
  "elastic_net_lambda_min_residual_TTS",
  "elastic_net_lambda_1se_residual_TTS",
  "lasso_rewards_x_side_lambda_min_residual_TTS",
  "lasso_rewards_x_side_lambda_1se_residual_TTS",
  "elastic_net_rewards_x_side_lambda_min_residual_TTS",
  "elastic_net_rewards_x_side_lambda_1se_residual_TTS",
  "lasso_rewards_plus_side_lambda_min_residual_TTS",
  "lasso_rewards_plus_side_lambda_1se_residual_TTS",
  "elastic_net_rewards_plus_side_lambda_min_residual_TTS",
  "elastic_net_rewards_plus_side_lambda_1se_residual_TTS"
)
for (column_name in numeric_columns) {
  stopifnot(is.numeric(cleaned_df[[column_name]]))
}

factor_columns <- c(
  "block_type",
  "session_ID",
  "rl_status",
  "mouse",
  "source_mouse",
  "session_id",
  "source_session_id",
  "date",
  "source_date"
)
for (column_name in factor_columns) {
  stopifnot(is.factor(cleaned_df[[column_name]]))
}

logical_columns <- c(
  "no_switch",
  "short_or_low_postswitch_trials",
  "confusion_flag",
  "bias_rl_flag",
  "bias_inf_flag",
  "bias_full_flag",
  "rl_thresh_flag"
)
for (column_name in logical_columns) {
  stopifnot(is.logical(cleaned_df[[column_name]]))
}

filtered_df <- block_env$clean_block_dataframe(block_df, drop_missing_tts = TRUE)
stopifnot(nrow(filtered_df) == 2)
stopifnot(!any(is.na(filtered_df$trials_to_correct)))

hmm_block_df <- block_df
hmm_block_df$inferred_strategy <- c("0", "1", "None")
hmm_block_df$declared_strategy <- c("Inference", "Qlearning", "None")
hmm_block_df$hardcoded_strategy <- c("Inference", "None", "Qlearning")
hmm_block_df$cur_strategy_slope <- c("0.25", "0.75", "None")
hmm_cleaned_df <- block_env$clean_block_dataframe(hmm_block_df)
stopifnot(is.factor(hmm_cleaned_df$inferred_strategy))
stopifnot(is.factor(hmm_cleaned_df$declared_strategy))
stopifnot(is.factor(hmm_cleaned_df$hardcoded_strategy))
stopifnot(is.numeric(hmm_cleaned_df$cur_strategy_slope))
stopifnot(is.na(hmm_cleaned_df$cur_strategy_slope[3]))

cat("All block preprocessing tests passed.\n")
