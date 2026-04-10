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
