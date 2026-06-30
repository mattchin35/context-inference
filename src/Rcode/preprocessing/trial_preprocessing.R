# Preprocessing pipelines for trial-level analysis data

bootstrap_get_script_path <- function() {
  frame_indices <- rev(seq_len(sys.nframe()))
  
  for (frame_index in frame_indices) {
    frame_file <- sys.frame(frame_index)$ofile
    if (!is.null(frame_file)) {
      return(normalizePath(frame_file, winslash = "/", mustWork = TRUE))
    }
  }
  
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE))
  }
  
  stop("Could not determine the current script path for source bootstrap.")
}

bootstrap_find_rcode_root <- function(start_path) {
  current_path <- dirname(normalizePath(start_path, winslash = "/", mustWork = TRUE))
  
  repeat {
    if (file.exists(file.path(current_path, "Rcode.Rproj"))) {
      return(current_path)
    }
    
    parent_path <- dirname(current_path)
    if (identical(parent_path, current_path)) {
      stop("Could not locate src/Rcode from script path: ", start_path)
    }
    current_path <- parent_path
  }
}

source_utils_path <- file.path(
  bootstrap_find_rcode_root(bootstrap_get_script_path()),
  "source_utils.R"
)
source(source_utils_path, local = TRUE)

source_rcode("preprocessing/common_preprocessing.R")

# Convert a trialwise left-minus-right regressor into an approximate left-choice
# probability without claiming exact equivalence to the original agent policy.
#
# Args:
#   signed_value: numeric vector of length n_trials. Positive values favor left,
#     negative values favor right, and 0 is neutral. Units: signed regressor.
#
# Returns:
#   numeric vector of length n_trials with values in [0, 1]. Units: approximate
#   left-choice probability.
signed_value_to_linear_probability <- function(signed_value) {
  numeric_value <- suppressWarnings(as.numeric(signed_value))
  clamped_value <- pmax(pmin(numeric_value, 1), -1)
  (clamped_value / 2) + 0.5
}

# Convert tanh(log-odds)-style belief regressors back into approximate left-side
# probabilities.
#
# Args:
#   signed_belief: numeric vector of length n_trials. Expected range is [-1, 1],
#     where positive values favor left. Units: tanh-scaled signed belief.
#   tanh_scale: numeric scalar. Scale factor applied before tanh in the source
#     feature computation. Units: dimensionless.
#
# Returns:
#   numeric vector of length n_trials with values in [0, 1]. Units: approximate
#   left-belief probability.
signed_belief_to_probability <- function(signed_belief, tanh_scale = 1) {
  numeric_value <- suppressWarnings(as.numeric(signed_belief))
  bounded_value <- pmax(pmin(numeric_value, 1 - 1e-8), -1 + 1e-8)
  stats::plogis(atanh(bounded_value) / tanh_scale)
}

# Add compatibility columns so legacy R analyses can run against the current
# trial-feature schema while keeping approximate reconstructions explicit.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Trialwise features loaded
#     from `*_augmented_trials.csv`.
#
# Returns:
#   data.frame with shape (n_trials, n_columns + n_added_columns). Existing
#   columns are preserved; newly added columns use explicit compatibility names
#   or legacy aliases when the mapping is a pure rename.
add_trial_column_compatibility <- function(df) {
  compatible_df <- df
  
  if (!"consecutive_failures" %in% names(compatible_df) &&
      "consecutive_omissions" %in% names(compatible_df)) {
    compatible_df[["consecutive_failures"]] <- compatible_df[["consecutive_omissions"]]
  }
  
  if (!"consecutive_failures_memory" %in% names(compatible_df) &&
      "consecutive_omissions_memory" %in% names(compatible_df)) {
    compatible_df[["consecutive_failures_memory"]] <- compatible_df[["consecutive_omissions_memory"]]
  }

  compatible_df
}

# Return the subset of requested columns that are present in a dataframe.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Trialwise table.
#   columns: character vector. Candidate column names.
#
# Returns:
#   character vector. Names from `columns` that exist in `df`, preserving input
#   order.
existing_columns <- function(df, columns) {
  columns[columns %in% names(df)]
}

# Convert project missing-value sentinels to NA in columns that exist.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Trialwise table.
#   columns: character vector. Candidate column names to clean.
#
# Returns:
#   data.frame with the same shape as `df`. Existing requested columns have
#   string missing-value sentinels replaced by NA.
convert_existing_none_to_na <- function(df, columns) {
  convertNoneToNA(df, existing_columns(df, columns))
}

# Coerce existing columns to numeric without changing row order.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Trialwise table.
#   columns: character vector. Candidate numeric column names. Units are the
#     source feature units documented by the augmented trial CSV generator.
#
# Returns:
#   data.frame with the same shape as `df`. Existing requested columns are
#   numeric vectors; absent columns are ignored.
coerce_existing_numeric_columns <- function(df, columns) {
  for (col in existing_columns(df, columns)) {
    df[[col]] <- suppressWarnings(as.numeric(unlist(df[[col]])))
  }
  df
}

# Coerce existing columns to factors without changing row order.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Trialwise table.
#   columns: character vector. Candidate categorical column names.
#
# Returns:
#   data.frame with the same shape as `df`. Existing requested columns are
#   factors; absent columns are ignored.
factor_existing_columns <- function(df, columns) {
  for (col in existing_columns(df, columns)) {
    df[[col]] <- as.factor(df[[col]])
  }
  df
}

# Coerce existing flag columns to logical values.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Trialwise table.
#   columns: character vector. Candidate logical flag column names. Accepted
#     true values are TRUE, "true", "t", "1", and "yes"; accepted false values
#     are FALSE, "false", "f", "0", and "no". Missing values remain NA.
#
# Returns:
#   data.frame with the same shape as `df`. Existing requested columns are
#   logical vectors; absent columns are ignored.
coerce_existing_logical_columns <- function(df, columns) {
  for (col in existing_columns(df, columns)) {
    if (is.logical(df[[col]])) {
      next
    }
    
    text_values <- tolower(trimws(as.character(df[[col]])))
    logical_values <- rep(NA, length(text_values))
    logical_values[text_values %in% c("true", "t", "1", "yes")] <- TRUE
    logical_values[text_values %in% c("false", "f", "0", "no")] <- FALSE
    df[[col]] <- as.logical(logical_values)
  }
  df
}

# Clean an augmented trial dataframe for R visualization/modeling.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Rows are behavioral trials.
#     Expected new-schema core columns include `action`, `prev_action`, and
#     `prev_reward`. Optional numeric predictors may include model values,
#     hazard/doubt regressors, residualized regressors, and observer values.
#     Actions are coded 0=right and 1=left. Model/regressor columns are
#     unitless left-minus-right scalar values unless otherwise documented by
#     the Python feature generator.
#   include_model_regressors: logical scalar. If TRUE, coerce optional model and
#     engineered regressor columns to numeric when they exist.
#
# Returns:
#   data.frame with shape (n_valid_trials, n_columns + n_compatibility_columns).
#   Rows missing `action`, `prev_action`, or `prev_reward` are dropped. Existing
#   numeric predictors are numeric; existing categorical columns are factors.
cleanup_trial_dataframe <- function(df, include_model_regressors = FALSE) {
  required_non_missing_columns <- c("action", "prev_action", "prev_reward")
  base_numeric_columns <- c(
    "cur_trial", "cur_trial_in_block", "cur_block",
    "reward_time", "start_time", "choice_time", "led_on_time", "led_off_time",
    "trial_time_since_start", "choice_time_since_start", "time_to_choice",
    "p_active_rew", "p_inactive_rew", "p_switch"
  )
  history_numeric_columns <- c(
    "negative_value",
    "consecutive_rewards_memory",
    "consecutive_omissions_memory",
    "consecutive_rewards",
    "consecutive_omissions",
    "consecutive_failures_memory",
    "consecutive_failures",
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
    "relative_monotonic_cf_value"
  )
  model_numeric_columns <- c(
    "Qlearning_rel_value",
    "FQlearning_rel_value",
    "FQlearning_rel_value_fast_learn",
    "HMM_rel_value_logodds",
    "HMM_rel_value_logodds_decay",
    "relative_omissions_index",
    "signed_omission_regressor",
    "relative_doubt_index",
    "relative_hazard_index",
    "perseveration_regressor",
    "doubt_perseveration_value",
    "wsls_regressor",
    "observer_value",
    "HMM_decay_res",
    "rel_hazard_res"
  )
  run_numeric_columns <- c("explore_run_id", "explore_run_length")
  factor_columns <- c(
    "state", "state_int", "action", "correct", "reward", "session_ID",
    "block_type", "prev_action", "prev_reward", "inherited_block_strategy",
    "inherited_block_bias"
  )
  trial_type_flag_columns <- c(
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
  sentinel_columns <- unique(c(
    required_non_missing_columns,
    "active_stimulus", "block_stimulus",
    base_numeric_columns,
    history_numeric_columns,
    model_numeric_columns,
    run_numeric_columns,
    factor_columns,
    trial_type_flag_columns
  ))
  
  missing_required_columns <- setdiff(required_non_missing_columns, names(df))
  if (length(missing_required_columns) > 0) {
    stop(
      "Trial dataframe is missing required cleanup columns: ",
      paste(missing_required_columns, collapse = ", ")
    )
  }
  
  df <- convert_existing_none_to_na(df, sentinel_columns)
  df <- removeNARows(df, required_non_missing_columns)
  df <- add_trial_column_compatibility(df)
  
  df <- coerce_existing_numeric_columns(df, base_numeric_columns)
  df <- coerce_existing_numeric_columns(df, history_numeric_columns)
  df <- coerce_existing_numeric_columns(df, run_numeric_columns)
  
  if (include_model_regressors) {
    df <- coerce_existing_numeric_columns(df, model_numeric_columns)
  }
  
  df <- coerce_existing_logical_columns(df, trial_type_flag_columns)
  df <- factor_existing_columns(df, factor_columns)
  return(df)
}

# Clean a leave-stay trial dataframe for R visualization/modeling.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Rows are behavioral trials
#     loaded from `*_leave_stay_trial_values.csv`. Required columns are
#     `action`, `prev_action`, and `prev_reward`. Here, `action` is a stay/leave
#     outcome coded 1=stay and 0=leave/switch; `raw_action` preserves the
#     original left/right side choice when present. Predictors ending in
#     `_prev_action_side` use the previous-action convention: positive values
#     favor staying with the previous action and negative values favor switching
#     away from it.
#   include_model_regressors: logical scalar. If TRUE, coerce optional
#     side-equivalent model and engineered regressor columns to numeric when
#     they exist.
#
# Returns:
#   data.frame with shape (n_valid_trials, n_columns). Rows missing `action`,
#   `prev_action`, or `prev_reward` are dropped. Existing side-equivalent
#   predictors are numeric; existing categorical columns are factors.
cleanup_leave_stay_trial_dataframe <- function(df, include_model_regressors = FALSE) {
  required_non_missing_columns <- c("action", "prev_action", "prev_reward")
  base_numeric_columns <- c(
    "cur_trial", "cur_trial_in_block", "cur_block", "time_to_choice"
  )
  side_equivalent_numeric_columns <- c(
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
    "doubt_perseveration_value_prev_action_side",
    "observer_value_prev_action_side",
    "HMM_decay_res_prev_action_side",
    "rel_hazard_res_prev_action_side"
  )
  run_numeric_columns <- c("explore_run_id", "explore_run_length")
  factor_columns <- c(
    "state", "state_int", "raw_action", "action", "correct", "reward",
    "experimenter_reward_given", "session_ID", "block_type", "prev_action",
    "prev_reward", "inherited_block_strategy", "inherited_block_bias"
  )
  trial_type_flag_columns <- c(
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
  sentinel_columns <- unique(c(
    required_non_missing_columns,
    base_numeric_columns,
    side_equivalent_numeric_columns,
    run_numeric_columns,
    factor_columns,
    trial_type_flag_columns
  ))
  
  missing_required_columns <- setdiff(required_non_missing_columns, names(df))
  if (length(missing_required_columns) > 0) {
    stop(
      "Leave-stay dataframe is missing required cleanup columns: ",
      paste(missing_required_columns, collapse = ", ")
    )
  }
  
  df <- convert_existing_none_to_na(df, sentinel_columns)
  df <- removeNARows(df, required_non_missing_columns)
  df <- coerce_existing_numeric_columns(df, base_numeric_columns)
  df <- coerce_existing_numeric_columns(df, run_numeric_columns)
  
  if (include_model_regressors) {
    df <- coerce_existing_numeric_columns(df, side_equivalent_numeric_columns)
  }
  
  df <- coerce_existing_logical_columns(df, trial_type_flag_columns)
  df <- factor_existing_columns(df, factor_columns)
  return(df)
}

curate_trial_analysis_dataframe <- function(df,
                                            drop_columns = c("state", "state_int", "block_type",
                                                             "active_stimulus", "block_stimulus", "reward_time",
                                                             "start_time", "choice_time", "led_on_time",
                                                             "led_off_time", "p_active_rew", "p_inactive_rew",
                                                             "p_switch", "session_ID")) {
  existing_drop_columns <- intersect(drop_columns, names(df))
  
  if (length(existing_drop_columns) == 0) {
    return(df)
  }
  
  curated_df <- dplyr::select(df, -dplyr::all_of(existing_drop_columns))
  return(curated_df)
}

curate_trial_analysis_dataframe_for_random_forest <- function(df,
                                            drop_columns = c("state", "state_int", "block_type", "cur_block", "cur_trial", "cur_trial_in_block",
                                                             "experimenter_reward_given", "trial_time_since_start", "choice_time_since_start", "time_to_choice",
                                                             "active_stimulus", "block_stimulus", "reward_time",
                                                             "start_time", "choice_time", "led_on_time",
                                                             "led_off_time", "p_active_rew", "p_inactive_rew",
                                                             "stay_trial", "switch_trial", "explore_trial", "reward", "correct",
                                                             "block_entry_trial", "first_switch_in_block", "block_entry_explore_trial",
                                                             "inherited_block_strategy",
                                                             "p_switch", "session_ID")) {
  existing_drop_columns <- intersect(drop_columns, names(df))

  if (length(existing_drop_columns) == 0) {
    return(df)
  }

  curated_df <- dplyr::select(df, -dplyr::all_of(existing_drop_columns))
  return(curated_df)
}

preprocess_trial_dataframe <- function(df, include_model_regressors = FALSE, drop_columns = NULL) {
  cleaned_df <- cleanup_trial_dataframe(df, include_model_regressors = include_model_regressors)
  
  if (is.null(drop_columns)) {
    curated_df <- curate_trial_analysis_dataframe(cleaned_df)
  } else {
    curated_df <- curate_trial_analysis_dataframe(cleaned_df, drop_columns = drop_columns)
  }
  
  return(list(trial_df = cleaned_df, clean_df = curated_df))
}

# Preprocess a leave-stay dataframe and return both cleaned and curated tables.
#
# Args:
#   df: data.frame with shape (n_trials, n_columns). Leave-stay trial table
#     loaded from `*_leave_stay_trial_values.csv`.
#   include_model_regressors: logical scalar. If TRUE, side-equivalent model
#     regressors are coerced to numeric when present.
#   drop_columns: character vector or NULL. Columns to remove from the curated
#     output. If NULL, the standard trial curation drop list is used.
#
# Returns:
#   named list with:
#     - leave_stay_df: cleaned data.frame with shape
#       (n_valid_trials, n_columns)
#     - clean_df: curated data.frame with selected metadata columns dropped
preprocess_leave_stay_trial_dataframe <- function(df, include_model_regressors = FALSE, drop_columns = NULL) {
  cleaned_df <- cleanup_leave_stay_trial_dataframe(df, include_model_regressors = include_model_regressors)
  
  if (is.null(drop_columns)) {
    curated_df <- curate_trial_analysis_dataframe(cleaned_df)
  } else {
    curated_df <- curate_trial_analysis_dataframe(cleaned_df, drop_columns = drop_columns)
  }
  
  return(list(leave_stay_df = cleaned_df, clean_df = curated_df))
}
