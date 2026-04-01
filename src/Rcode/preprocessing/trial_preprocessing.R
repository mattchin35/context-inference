# Preprocessing pipelines for trial-level analysis data

source("preprocessing/common_preprocessing.R")

cleanup_trial_dataframe <- function(df, include_model_regressors = FALSE) {
  df <- convertNoneToNA(df, c("action", "active_stimulus", "block_stimulus", "reward_time"))
  df <- convertNoneToNA(df, c("Qlearning_prob_left", "Qlearning_rel_value", "Qlearning_greedy_action",
                              "FQlearning_prob_left", "FQlearning_rel_value", "FQlearning_greedy_action"))
  df <- convertNoneToNA(df, c("RFLR_prob_left", "RFLR_rel_value", "RFLR_greedy_action",
                              "HMM_prob_left", "HMM_rel_value", "HMM_greedy_action"))
  df <- convertNoneToNA(df, c("prev_action", "prev_reward"))
  df <- removeNARows(df, "action")
  df <- convertNoneToNA(df, c("inferred_strategy"))
  df <- removeNARows(df, "inferred_strategy")
  
  df["reward_time"] <- as.numeric(unlist(df["reward_time"]))
  df["prev_action"] <- as.numeric(unlist(df["prev_action"]))
  df["prev_reward"] <- as.numeric(unlist(df["prev_reward"]))
  
  if (include_model_regressors) {
    numeric_cols <- c("HMM_rel_value", "HMM_prob_left", "RFLR_rel_value", "RFLR_prob_left",
                      "Qlearning_rel_value", "Qlearning_prob_left", "FQlearning_rel_value",
                      "FQlearning_prob_left")
    for (col in numeric_cols) {
      df[col] <- as.numeric(unlist(df[col]))
    }
    
    df <- df %>%
      dplyr::mutate(across(c(state, state_int, action, correct, session_ID, block_type, prev_action,
                             prev_reward, Qlearning_greedy_action, FQlearning_greedy_action,
                             RFLR_greedy_action, HMM_greedy_action, inherited_strategy,
                             inherited_bias_flag, inferred_strategy), as.factor))
  } else {
    df <- df %>%
      dplyr::mutate(across(c(state, state_int, action, correct, session_ID, block_type, prev_action,
                             prev_reward, inherited_strategy, inherited_bias_flag,
                             inferred_strategy), as.factor))
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