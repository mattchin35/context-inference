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
  
  if (!"Qlearning_prob_left_approx" %in% names(compatible_df) &&
      "Qlearning_rel_value" %in% names(compatible_df)) {
    compatible_df[["Qlearning_prob_left_approx"]] <- signed_value_to_linear_probability(
      compatible_df[["Qlearning_rel_value"]]
    )
  }
  
  if (!"FQlearning_prob_left_approx" %in% names(compatible_df) &&
      "FQlearning_rel_value" %in% names(compatible_df)) {
    compatible_df[["FQlearning_prob_left_approx"]] <- signed_value_to_linear_probability(
      compatible_df[["FQlearning_rel_value"]]
    )
  }
  
  if (!"HMM_prob_left_logodds_approx" %in% names(compatible_df) &&
      "HMM_rel_value_logodds" %in% names(compatible_df)) {
    compatible_df[["HMM_prob_left_logodds_approx"]] <- signed_belief_to_probability(
      compatible_df[["HMM_rel_value_logodds"]]
    )
  }
  
  if (!"HMM_prob_left_logodds_decay_approx" %in% names(compatible_df) &&
      "HMM_rel_value_logodds_decay" %in% names(compatible_df)) {
    compatible_df[["HMM_prob_left_logodds_decay_approx"]] <- signed_belief_to_probability(
      compatible_df[["HMM_rel_value_logodds_decay"]]
    )
  }
  
  compatible_df
}

cleanup_trial_dataframe <- function(df, include_model_regressors = FALSE) {
  df <- convertNoneToNA(df, c("action", "active_stimulus", "block_stimulus", "reward_time"))
  df <- convertNoneToNA(df, c("Qlearning_prob_left", "Qlearning_rel_value", "Qlearning_greedy_action",
                              "FQlearning_prob_left", "FQlearning_rel_value", "FQlearning_greedy_action"))
  df <- convertNoneToNA(df, c("RFLR_prob_left", "RFLR_rel_value", "RFLR_greedy_action",
                              "HMM_prob_left", "HMM_rel_value", "HMM_greedy_action"))
  df <- convertNoneToNA(df, c("HMM_rel_value_logodds", "HMM_rel_value_logodds_decay",
                              "Qlearning_prob_left_approx", "FQlearning_prob_left_approx",
                              "HMM_prob_left_logodds_approx", "HMM_prob_left_logodds_decay_approx",
                              "relative_omissions_index", "relative_doubt_index", "relative_hazard_index",
                              # "HMM_rel_value_logodds_decay",
                              "HMM_decay_res", "rel_hazard_res",))
  df <- convertNoneToNA(df, c("prev_action", "prev_reward"))
  df <- removeNARows(df, "action")
  df <- convertNoneToNA(df, c("inferred_strategy"))
  df <- removeNARows(df, "inferred_strategy")
  df <- add_trial_column_compatibility(df)
  
  df["reward_time"] <- as.numeric(unlist(df["reward_time"]))
  df["prev_action"] <- as.numeric(unlist(df["prev_action"]))
  df["prev_reward"] <- as.numeric(unlist(df["prev_reward"]))
  
  if (include_model_regressors) {
    numeric_cols <- intersect(
      c("HMM_rel_value", "HMM_prob_left", "RFLR_rel_value", "RFLR_prob_left",
        "Qlearning_rel_value", "Qlearning_prob_left", "FQlearning_rel_value",
        "FQlearning_prob_left", "HMM_rel_value_logodds", "HMM_rel_value_logodds_decay",
        "Qlearning_prob_left_approx", "FQlearning_prob_left_approx",
        "HMM_prob_left_logodds_approx", "HMM_prob_left_logodds_decay_approx",
        "relative_omissions_index", "relative_doubt_index"),
      names(df)
    )
    for (col in numeric_cols) {
      df[col] <- as.numeric(unlist(df[col]))
    }
    
    df <- df %>%
      dplyr::mutate(across(dplyr::all_of(intersect(
        c("state", "state_int", "action", "correct", "session_ID", "block_type", "prev_action",
          "prev_reward", "Qlearning_greedy_action", "FQlearning_greedy_action",
          "RFLR_greedy_action", "HMM_greedy_action", "inherited_strategy",
          "inherited_bias_flag", "inferred_strategy"),
        names(df)
      )), as.factor))
  } else {
    df <- df %>%
      dplyr::mutate(across(dplyr::all_of(intersect(
        c("state", "state_int", "action", "correct", "session_ID", "block_type", "prev_action",
          "prev_reward", "inherited_strategy", "inherited_bias_flag", "inferred_strategy"),
        names(df)
      )), as.factor))
  }
  
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

preprocess_trial_dataframe <- function(df, include_model_regressors = FALSE, drop_columns = NULL) {
  cleaned_df <- cleanup_trial_dataframe(df, include_model_regressors = include_model_regressors)
  
  if (is.null(drop_columns)) {
    curated_df <- curate_trial_analysis_dataframe(cleaned_df)
  } else {
    curated_df <- curate_trial_analysis_dataframe(cleaned_df, drop_columns = drop_columns)
  }
  
  return(list(trial_df = cleaned_df, clean_df = curated_df))
}
