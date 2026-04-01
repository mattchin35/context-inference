# Preprocessing pipelines for block-level analysis data

source("preprocessing/common_preprocessing.R")

clean_block_dataframe <- function(df) {
  df <- convertNoneToNA(df, c("trials_to_correct", "prev_consecutive_rewards",
                              "prev_consecutive_rewards_memory", "prev_n_correct",
                              "prev_n_rewarded", "bias_rl", "bias_inf", "bias_rl_flag",
                              "bias_inf_flag", "bias_full_flag", "inferred_strategy",
                              "declared_strategy"))
  df <- removeNARows(df, c("prev_n_correct", "trials_to_correct"))
  df <- df %>%
    dplyr::mutate(across(c(block_type, confusion_flag, session_ID, bias_rl_flag,
                           bias_inf_flag, bias_full_flag, inferred_strategy,
                           declared_strategy, cur_strategy_slope), as.factor))
  
  df["trials_to_correct"] <- as.numeric(unlist(df["trials_to_correct"]))
  df["prev_consecutive_rewards"] <- as.numeric(unlist(df["prev_consecutive_rewards"]))
  df["prev_consecutive_rewards_memory"] <- as.numeric(unlist(df["prev_consecutive_rewards_memory"]))
  df["prev_n_rewarded"] <- as.numeric(unlist(df["prev_n_rewarded"]))
  df["bias_rl"] <- as.numeric(unlist(df["bias_rl"]))
  df["bias_inf"] <- as.numeric(unlist(df["bias_inf"]))
  df["prev_n_correct"] <- as.numeric(unlist(df["prev_n_correct"]))
  
  return(df)
}