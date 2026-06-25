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

stopifnot(exists("cleanup_leave_stay_trial_dataframe", envir = trial_env, inherits = FALSE))
stopifnot(exists("preprocess_leave_stay_trial_dataframe", envir = trial_env, inherits = FALSE))

leave_stay_df <- data.frame(
  state = c("0", "0", "1", "1"),
  state_int = c("0", "0", "1", "1"),
  cur_trial = c("0", "1", "2", "3"),
  cur_trial_in_block = c("0", "1", "0", "1"),
  cur_block = c("0", "0", "1", "1"),
  raw_action = c("1", "1", "0", "0"),
  action = c("None", "1", "0", "1"),
  correct = c("1", "1", "0", "1"),
  reward = c("0", "1", "0", "1"),
  experimenter_reward_given = c("0", "0", "0", "0"),
  session_ID = c("sess", "sess", "sess", "sess"),
  block_type = c("A", "A", "B", "B"),
  time_to_choice = c("0.5", "0.6", "0.7", "0.8"),
  prev_action = c("None", "1", "0", "0"),
  prev_reward = c("None", "1", "0", "1"),
  inherited_block_strategy = c("None", "1", "1", "2"),
  inherited_block_bias = c("None", "False", "False", "True"),
  Qlearning_rel_value_prev_action_side = c("None", "0.2", "0.3", "-0.4"),
  FQlearning_rel_value_prev_action_side = c("None", "0.4", "0.6", "-0.8"),
  FQlearning_rel_value_fast_learn_prev_action_side = c("None", "0.5", "0.7", "-0.9"),
  HMM_rel_value_logodds_prev_action_side = c("None", "0.6", "0.8", "-1.0"),
  HMM_rel_value_logodds_decay_prev_action_side = c("None", "0.7", "0.9", "-0.1"),
  relative_omissions_index_prev_action_side = c("None", "0.15", "0.25", "-0.35"),
  signed_omission_regressor_prev_action_side = c("None", "0.16", "0.26", "-0.36"),
  relative_doubt_index_prev_action_side = c("None", "-0.2", "-0.3", "0.4"),
  relative_hazard_index_prev_action_side = c("None", "-0.5", "-0.75", "1.0"),
  perseveration_regressor_prev_action_side = c("None", "0.2", "0.4", "-0.6"),
  observer_value_prev_action_side = c("None", "0.9", "1.2", "-0.5"),
  HMM_decay_res_prev_action_side = c("None", "0.21", "0.31", "-0.41"),
  rel_hazard_res_prev_action_side = c("None", "0.22", "0.32", "-0.42"),
  stringsAsFactors = FALSE
)

cleaned_df <- trial_env$cleanup_leave_stay_trial_dataframe(
  leave_stay_df,
  include_model_regressors = TRUE
)

stopifnot(nrow(cleaned_df) == 3)
stopifnot(identical(as.character(cleaned_df$action), c("1", "0", "1")))
stopifnot(identical(as.character(cleaned_df$raw_action), c("1", "0", "0")))
stopifnot(identical(as.character(cleaned_df$prev_action), c("1", "0", "0")))
stopifnot(identical(as.character(cleaned_df$prev_reward), c("1", "0", "1")))

numeric_columns <- c(
  "cur_trial",
  "cur_trial_in_block",
  "cur_block",
  "time_to_choice",
  "Qlearning_rel_value_prev_action_side",
  "FQlearning_rel_value_prev_action_side",
  "FQlearning_rel_value_fast_learn_prev_action_side",
  "HMM_rel_value_logodds_prev_action_side",
  "HMM_rel_value_logodds_decay_prev_action_side",
  "relative_omissions_index_prev_action_side",
  "signed_omission_regressor_prev_action_side",
  "relative_doubt_index_prev_action_side",
  "relative_hazard_index_prev_action_side",
  "perseveration_regressor_prev_action_side",
  "observer_value_prev_action_side",
  "HMM_decay_res_prev_action_side",
  "rel_hazard_res_prev_action_side"
)
for (column_name in numeric_columns) {
  stopifnot(is.numeric(cleaned_df[[column_name]]))
}

factor_columns <- c(
  "state",
  "state_int",
  "raw_action",
  "action",
  "correct",
  "reward",
  "experimenter_reward_given",
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

minimal_leave_stay_df <- leave_stay_df[, setdiff(
  names(leave_stay_df),
  c(
    "HMM_decay_res_prev_action_side",
    "rel_hazard_res_prev_action_side",
    "inherited_block_strategy",
    "inherited_block_bias"
  )
)]
minimal_cleaned_df <- trial_env$cleanup_leave_stay_trial_dataframe(
  minimal_leave_stay_df,
  include_model_regressors = TRUE
)
stopifnot(nrow(minimal_cleaned_df) == 3)
stopifnot(!"HMM_decay_res_prev_action_side" %in% names(minimal_cleaned_df))
stopifnot(!"inherited_block_strategy" %in% names(minimal_cleaned_df))

preprocessed <- trial_env$preprocess_leave_stay_trial_dataframe(
  leave_stay_df,
  include_model_regressors = TRUE
)
stopifnot(all(c("leave_stay_df", "clean_df") %in% names(preprocessed)))
stopifnot(nrow(preprocessed$leave_stay_df) == 3)
stopifnot(nrow(preprocessed$clean_df) == 3)

cat("All leave-stay preprocessing tests passed.\n")
