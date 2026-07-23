# Preprocessing pipelines for block-level analysis data

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

# Return candidate column names that exist in a block dataframe.
#
# Args:
#   df: data.frame with shape (n_blocks, n_columns). Block-level session table.
#   columns: character vector. Candidate column names.
#
# Returns:
#   character vector. Existing column names, preserving candidate order.
existing_block_columns <- function(df, columns) {
  columns[columns %in% names(df)]
}

# Convert project missing-value sentinels to NA in existing block columns.
#
# Args:
#   df: data.frame with shape (n_blocks, n_columns). Block-level session table.
#   columns: character vector. Candidate columns to clean.
#
# Returns:
#   data.frame with the same shape as `df`. Existing requested columns have
#   string missing-value sentinels replaced by NA.
convert_existing_block_none_to_na <- function(df, columns) {
  convertNoneToNA(df, existing_block_columns(df, columns))
}

# Coerce existing block columns to numeric values without changing row order.
#
# Args:
#   df: data.frame with shape (n_blocks, n_columns). Block-level session table.
#   columns: character vector. Candidate numeric columns. Counts are in trials;
#     fractions and agreement columns are unitless.
#
# Returns:
#   data.frame with the same shape as `df`. Existing requested columns are
#   numeric vectors; absent columns are ignored.
coerce_existing_block_numeric_columns <- function(df, columns) {
  for (col in existing_block_columns(df, columns)) {
    df[[col]] <- suppressWarnings(as.numeric(unlist(df[[col]])))
  }
  df
}

# Coerce existing block columns to factors without changing row order.
#
# Args:
#   df: data.frame with shape (n_blocks, n_columns). Block-level session table.
#   columns: character vector. Candidate categorical columns.
#
# Returns:
#   data.frame with the same shape as `df`. Existing requested columns are
#   factors; absent columns are ignored.
factor_existing_block_columns <- function(df, columns) {
  for (col in existing_block_columns(df, columns)) {
    df[[col]] <- as.factor(df[[col]])
  }
  df
}

# Coerce existing block flag columns to logical values.
#
# Args:
#   df: data.frame with shape (n_blocks, n_columns). Block-level session table.
#   columns: character vector. Candidate logical flag columns. Accepted true
#     values are TRUE, "true", "t", "1", and "yes"; accepted false values are
#     FALSE, "false", "f", "0", and "no". Missing values remain NA.
#
# Returns:
#   data.frame with the same shape as `df`. Existing requested columns are
#   logical vectors; absent columns are ignored.
coerce_existing_block_logical_columns <- function(df, columns) {
  for (col in existing_block_columns(df, columns)) {
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

# Clean a block-performance dataframe for R visualization/modeling.
#
# Args:
#   df: data.frame with shape (n_blocks, n_columns). Rows are block-level
#     metrics loaded from `*_block_performance.csv`. Current main-workflow
#     columns include TTS, transition metrics, bias/status metrics, omission
#     history, and agent agreement fractions. Optional post-HMM columns include
#     `inferred_strategy`, `declared_strategy`, `hardcoded_strategy`, and
#     `cur_strategy_slope`.
#   drop_missing_tts: logical scalar. If TRUE, remove rows missing
#     `trials_to_correct` or `prev_n_correct` when those columns exist. If
#     FALSE, preserve no-switch blocks and other missing-TTS rows for
#     inspection.
#
# Returns:
#   data.frame with shape (n_blocks, n_columns) unless `drop_missing_tts` drops
#   rows. Existing numeric metrics are numeric, existing flag columns are
#   logical, and existing categorical columns are factors. Missing-value
#   sentinels such as "None" are converted to NA in known block columns.
clean_block_dataframe <- function(df, drop_missing_tts = FALSE) {
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
    "cur_strategy_slope",
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
  logical_columns <- c(
    "no_switch",
    "short_or_low_postswitch_trials",
    "confusion_flag",
    "bias_rl_flag",
    "bias_inf_flag",
    "bias_full_flag",
    "rl_thresh_flag"
  )
  factor_columns <- c(
    "block_type",
    "session_ID",
    "rl_status",
    "mouse",
    "source_mouse",
    "session_id",
    "source_session_id",
    "date",
    "source_date",
    "inferred_strategy",
    "declared_strategy",
    "hardcoded_strategy"
  )
  exemplar_numeric_columns <- grep(
    "_exemplar_(normalized_)?residual_TTS$",
    names(df),
    value = TRUE
  )
  numeric_columns <- unique(c(numeric_columns, exemplar_numeric_columns))
  sentinel_columns <- unique(c(numeric_columns, logical_columns, factor_columns))
  
  df <- convert_existing_block_none_to_na(df, sentinel_columns)
  
  if (isTRUE(drop_missing_tts)) {
    filter_columns <- existing_block_columns(
      df,
      c("prev_n_correct", "trials_to_correct")
    )
    if (length(filter_columns) > 0) {
      df <- removeNARows(df, filter_columns)
    }
  }
  
  df <- coerce_existing_block_numeric_columns(df, numeric_columns)
  df <- coerce_existing_block_logical_columns(df, logical_columns)
  df <- factor_existing_block_columns(df, factor_columns)
  return(df)
}
