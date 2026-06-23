require(flexplot)
require(ggplot2)
require(tidyverse)
require(dplyr)
require(tidyverse)
require(cowplot)
require(lme4)

#require(randomForestSRC)  #not sure about flexplot compatibility
#require(randomForest)
#library(varPro)

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

source_rcode("preprocessing/block_preprocessing.R")
source_rcode("preprocessing/trial_preprocessing.R")

###############################################################################
# Actual processing and analysis of trial data
###############################################################################

trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT016/CT016_20260529_latent_inference/processed/CT016_2026-05-29_132812_augmented_trials.csv",
                     header = TRUE)#,
trial_df <- cleanup_trial_dataframe(trial_df, include_model_regressors = TRUE)
clean_df <- curate_trial_analysis_dataframe(trial_df)


flexplot(FQlearning_rel_value ~ HMM_rel_value_logodds_decay + relative_hazard_index, data=clean_df)
flexplot(FQlearning_rel_value ~ HMM_decay_res + rel_hazard_res, data=clean_df)




