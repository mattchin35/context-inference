require(flexplot)
require(ggplot2)
require(tidyverse)
require(dplyr)
require(tidyverse)
require(cowplot)
require(lme4)

require(randomForestSRC)  #not sure about flexplot compatibility
#require(randomForest)
library(varPro)

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

#trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT020/CT020_20260611_latent_inference/processed/CT020_2026-06-11_151738_augmented_trials.csv",
#                     header = TRUE)#,
trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT016/CT016_20260529_latent_inference/processed/CT016_2026-05-29_132812_augmented_trials.csv",
                     header = TRUE)#,
trial_df <- cleanup_trial_dataframe(trial_df, include_model_regressors = TRUE)
#trial_df_no_giveaways <- curate_trial_analysis_dataframe(trial_df)
trial_df_rf <- curate_trial_analysis_dataframe_for_random_forest(trial_df)
first_switch <- trial_df %>% dplyr::filter(first_switch_in_block)

flexplot(FQlearning_rel_value ~ HMM_rel_value_logodds_decay, data=first_switch)
flexplot(action ~ signed_omission_regressor, data=trial_df)
flexplot(action ~ signed_omission_regressor | HMM_rel_value_logodds_decay + perseveration_regressor + relative_hazard_index, data=trial_df)
flexplot(action ~ signed_omission_regressor | FQlearning_rel_value + perseveration_regressor, data=trial_df)
 #flexplot(FQlearning_rel_value ~ HMM_decay_res + rel_hazard_res, data=clean_df)

trial_df_rf_subset <- trial_df_rf %>% select(#-signed_omission_regressor, 
                                             -negative_value,
                                             -consecutive_rewards_memory,-consecutive_omissions_memory,-consecutive_rewards	,-consecutive_omissions,
                                             -left_value,-right_value,-relative_value,-left_omissions,-right_omissions,-relative_omissions,-left_cf_value,-right_cf_value,-relative_cf_value,-left_cf_omissions,-right_cf_omissions,-relative_cf_omissions,-left_monotonic_cf_value,-right_monotonic_cf_value,-relative_monotonic_cf_value)

rf_action_subset = rfsrc(action ~ ., data=trial_df_rf_subset)
print(rf_action_subset)
varfit_action_subset = varpro(action ~ ., data=trial_df_rf_subset)
importance(varfit_action_subset, plot.it = TRUE)

rf_action = rfsrc(action ~ ., data=trial_df_rf)
print(rf_action)
varfit_action = varpro(action ~ ., data=trial_df_rf)
importance(varfit_action, plot.it = TRUE)
varfit_action_imp = importance(varfit_action)



#leave_stay_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT020/CT020_20260611_latent_inference/processed/CT020_2026-06-11_151738_leave_stay_trial_values.csv",
                     #header = TRUE)#,
leave_stay_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT016/CT016_20260529_latent_inference/processed/CT016_2026-05-29_132812_leave_stay_trial_values.csv",
                     header = TRUE)#,
leave_stay_df <- cleanup_leave_stay_trial_dataframe(leave_stay_df, include_model_regressors = TRUE)
#leave_stay_df <- curate_trial_analysis_dataframe(leave_stay_df)
first_switch_leave_stay <- leave_stay_df %>% filter(explore_trial)


flexplot(Qlearning_rel_value_prev_action_side ~ 1, data=leave_stay_df)
flexplot(FQlearning_rel_value_prev_action_side ~ 1, data=leave_stay_df)
flexplot(HMM_rel_value_logodds_prev_action_side ~ 1, data=leave_stay_df)
flexplot(HMM_rel_value_logodds_decay_prev_action_side ~ 1, data=leave_stay_df)
flexplot(relative_doubt_index_prev_action_side ~ 1, data=leave_stay_df)
flexplot(observer_value_prev_action_side ~ 1, data=leave_stay_df)
flexplot(HMM_decay_res_prev_action_side ~ 1, data=leave_stay_df)
flexplot(relative_hazard_index_prev_action_side ~ 1, data=leave_stay_df)



