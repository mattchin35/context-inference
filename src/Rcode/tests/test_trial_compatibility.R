get_test_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    stop("Could not determine test script path from commandArgs().")
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE)
}

test_script_path <- get_test_script_path()
rcode_root <- normalizePath(
  file.path(dirname(test_script_path), ".."),
  winslash = "/",
  mustWork = TRUE
)

library(dplyr)

trial_env <- new.env(parent = globalenv())
source(file.path(rcode_root, "preprocessing", "trial_preprocessing.R"), local = trial_env)

stopifnot(exists("cleanup_trial_dataframe", envir = trial_env, inherits = FALSE))
stopifnot(exists("add_trial_column_compatibility", envir = trial_env, inherits = FALSE))

new_schema_trial_df <- data.frame(
  state = c("0", "0", "1", "1"),
  state_int = c("0", "0", "1", "1"),
  action = c("0", "1", "1", "None"),
  correct = c("1", "0", "1", "0"),
  reward = c("0", "1", "1", "0"),
  session_ID = c("sess", "sess", "sess", "sess"),
  block_type = c("A", "A", "B", "B"),
  prev_action = c("None", "0", "1", "0"),
  prev_reward = c("None", "0", "None", "1"),
  active_stimulus = c("left", "right", "left", "right"),
  block_stimulus = c("left", "right", "left", "right"),
  reward_time = c("None", "2.0", "3.0", "4.0"),
  negative_value = c("0", "-1", "0", "-1"),
  consecutive_rewards_memory = c("0", "1", "1", "2"),
  consecutive_omissions = c(0, 1, 2, 3),
  consecutive_omissions_memory = c(0, 1, 1, 2),
  consecutive_rewards = c("0", "1", "2", "3"),
  left_value = c("0.1", "0.2", "0.3", "0.4"),
  right_value = c("0.4", "0.3", "0.2", "0.1"),
  relative_value = c("-0.3", "-0.1", "0.1", "0.3"),
  left_omissions = c("0", "1", "2", "3"),
  right_omissions = c("3", "2", "1", "0"),
  relative_omissions = c("-3", "-1", "1", "3"),
  left_cf_value = c("0.0", "0.1", "0.2", "0.3"),
  right_cf_value = c("0.3", "0.2", "0.1", "0.0"),
  relative_cf_value = c("-0.3", "-0.1", "0.1", "0.3"),
  left_cf_omissions = c("0", "1", "2", "3"),
  right_cf_omissions = c("3", "2", "1", "0"),
  relative_cf_omissions = c("-3", "-1", "1", "3"),
  left_monotonic_cf_value = c("0.0", "0.2", "0.4", "0.6"),
  right_monotonic_cf_value = c("0.6", "0.4", "0.2", "0.0"),
  relative_monotonic_cf_value = c("-0.6", "-0.2", "0.2", "0.6"),
  Qlearning_rel_value = c("0.1", "0.2", "0.3", "0.4"),
  FQlearning_rel_value = c("0.0", "0.5", "None", "0.8"),
  FQlearning_rel_value_fast_learn = c("0.0", "0.6", "0.7", "0.9"),
  HMM_rel_value_logodds = c("-0.1", "0.1", "0.2", "0.3"),
  HMM_rel_value_logodds_decay = c("-0.2", "0.2", "0.4", "0.6"),
  relative_omissions_index = c("-0.5", "0.0", "0.5", "0.8"),
  relative_doubt_index = c("-0.2", "0.1", "0.6", "0.7"),
  relative_hazard_index = c("0.2", "0.3", "0.4", "0.5"),
  perseveration_regressor = c("0", "0.25", "0.5", "0.75"),
  doubt_perseveration_value = c("0.0", "0.025", "0.3", "0.525"),
  signed_omission_regressor = c("-0.5", "0.0", "0.5", "0.8"),
  observer_value = c("0.0", "0.1", "0.2", "0.3"),
  HMM_decay_res = c("-0.01", "0.02", "0.03", "0.04"),
  rel_hazard_res = c("0.11", "0.12", "0.13", "0.14"),
  inherited_block_strategy = c("1", "1", "2", "2"),
  inherited_block_bias = c("False", "False", "True", "True"),
  prev_correct = c("FALSE", "TRUE", "TRUE", "FALSE"),
  block_entry_trial = c("TRUE", "FALSE", "TRUE", "FALSE"),
  switch_trial = c("FALSE", "TRUE", "FALSE", "FALSE"),
  stay_trial = c("FALSE", "FALSE", "TRUE", "FALSE"),
  first_switch_in_block = c("FALSE", "TRUE", "FALSE", "FALSE"),
  explore_trial = c("FALSE", "FALSE", "FALSE", "FALSE"),
  block_entry_explore_trial = c("FALSE", "FALSE", "FALSE", "FALSE"),
  explore_run_start = c("FALSE", "TRUE", "FALSE", "FALSE"),
  explore_run_trial = c("FALSE", "TRUE", "TRUE", "FALSE"),
  explore_run_return = c("FALSE", "FALSE", "TRUE", "FALSE"),
  explore_run_id = c("None", "7", "7", "None"),
  explore_run_length = c("None", "2", "2", "None"),
  stringsAsFactors = FALSE
)

compat_df <- trial_env$add_trial_column_compatibility(new_schema_trial_df)

stopifnot(all(c("consecutive_failures", "consecutive_failures_memory") %in% names(compat_df)))
stopifnot(identical(compat_df$consecutive_failures, compat_df$consecutive_omissions))
stopifnot(identical(
  compat_df$consecutive_failures_memory,
  compat_df$consecutive_omissions_memory
))
stopifnot(!"FQlearning_prob_left_approx" %in% names(compat_df))
stopifnot(!"HMM_prob_left_logodds_decay_approx" %in% names(compat_df))

cleaned_df <- trial_env$cleanup_trial_dataframe(new_schema_trial_df, include_model_regressors = TRUE)

stopifnot(nrow(cleaned_df) == 1)
stopifnot(identical(as.character(cleaned_df$action), "1"))
stopifnot(identical(as.character(cleaned_df$prev_action), "0"))
stopifnot(identical(as.character(cleaned_df$prev_reward), "0"))

numeric_columns <- c(
  "negative_value",
  "consecutive_rewards_memory",
  "consecutive_omissions_memory",
  "consecutive_rewards",
  "consecutive_omissions",
  "left_value",
  "right_value",
  "relative_value",
  "left_omissions",
  "right_omissions",
  "relative_omissions",
  "left_cf_value",
  "right_cf_value",
  "relative_cf_value",
  "left_cf_omissions",
  "right_cf_omissions",
  "relative_cf_omissions",
  "left_monotonic_cf_value",
  "right_monotonic_cf_value",
  "relative_monotonic_cf_value",
  "Qlearning_rel_value",
  "FQlearning_rel_value",
  "FQlearning_rel_value_fast_learn",
  "HMM_rel_value_logodds",
  "HMM_rel_value_logodds_decay",
  "relative_omissions_index",
  "relative_doubt_index",
  "relative_hazard_index",
  "perseveration_regressor",
  "doubt_perseveration_value",
  "signed_omission_regressor",
  "observer_value",
  "HMM_decay_res",
  "rel_hazard_res",
  "explore_run_id",
  "explore_run_length"
)
for (column_name in numeric_columns) {
  stopifnot(is.numeric(cleaned_df[[column_name]]))
}

factor_columns <- c(
  "state",
  "state_int",
  "action",
  "correct",
  "reward",
  "session_ID",
  "block_type",
  "prev_action",
  "prev_reward",
  "inherited_block_strategy",
  "inherited_block_bias"
)
for (column_name in factor_columns) {
  stopifnot(is.factor(cleaned_df[[column_name]]))
}

flag_columns <- c(
  "prev_correct",
  "block_entry_trial",
  "switch_trial",
  "stay_trial",
  "first_switch_in_block",
  "explore_trial",
  "block_entry_explore_trial",
  "explore_run_start",
  "explore_run_trial",
  "explore_run_return"
)
for (column_name in flag_columns) {
  stopifnot(is.logical(cleaned_df[[column_name]]))
}
stopifnot(identical(cleaned_df$switch_trial, TRUE))
stopifnot(identical(cleaned_df$block_entry_trial, FALSE))

removed_columns <- c(
  "Qlearning_prob_left",
  "FQlearning_prob_left",
  "HMM_prob_left",
  "HMM_rel_value",
  "RFLR_rel_value",
  "RFLR_prob_left",
  "inferred_strategy"
)
stopifnot(!any(removed_columns %in% names(cleaned_df)))

minimal_new_schema_df <- new_schema_trial_df[, setdiff(
  names(new_schema_trial_df),
  c(
    "relative_hazard_index",
    "observer_value",
    "HMM_decay_res",
    "rel_hazard_res",
    "inherited_block_strategy",
    "inherited_block_bias"
  )
)]
minimal_cleaned_df <- trial_env$cleanup_trial_dataframe(
  minimal_new_schema_df,
  include_model_regressors = TRUE
)
stopifnot(nrow(minimal_cleaned_df) == 1)
stopifnot(!"relative_hazard_index" %in% names(minimal_cleaned_df))
stopifnot(!"inherited_block_strategy" %in% names(minimal_cleaned_df))

cat("All trial compatibility tests passed.\n")
