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

#trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260610_latent_inference/processed/CT024_2026-06-10_150729_augmented_trials.csv",
#                     header = TRUE)#,
trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260609_latent_inference/processed/CT024_2026-06-09_143852_augmented_trials.csv",
                     header = TRUE)#,
#trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260529_latent_inference/processed/CT024_2026-05-29_143237_augmented_trials.csv",
                     #header = TRUE)#,
#trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260520_latent_inference/processed/CT024_2026-05-20_183627_augmented_trials.csv",
 #                    header = TRUE)#,
#trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT021/CT021_20260519_latent_inference/processed/CT021_2026-05-19_130244_augmented_trials.csv",
#                     header = TRUE)#,
#trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT016/CT016_20260529_latent_inference/processed/CT016_2026-05-29_132812_augmented_trials.csv",
                     #header = TRUE)#,
trial_df <- cleanup_trial_dataframe(trial_df, include_model_regressors = TRUE)
#trial_df_no_giveaways <- curate_trial_analysis_dataframe(trial_df)
trial_df_rf <- curate_trial_analysis_dataframe_for_random_forest(trial_df)
first_switch <- trial_df %>% dplyr::filter(first_switch_in_block)

#flexplot(FQlearning_rel_value ~ HMM_rel_value_logodds_decay, data=first_switch)
flexplot(observer_value ~ 1, data=trial_df)
flexplot(action ~ FQlearning_rel_value  + relative_doubt_index | HMM_decay_res + perseveration_regressor, data=trial_df, method="logistic")
flexplot(action ~ HMM_rel_value_logodds_decay + relative_doubt_index | perseveration_regressor, data=trial_df, method="logistic")

flexplot(action ~ signed_omission_regressor | HMM_rel_value_logodds_decay, data=trial_df, method="logistic")
flexplot(action ~ signed_omission_regressor | perseveration_regressor, data=trial_df)
flexplot(action ~ signed_omission_regressor | Qlearning_rel_value + relative_doubt_index, data=trial_df)
flexplot(action ~ signed_omission_regressor | HMM_rel_value_logodds_decay + relative_doubt_index + perseveration_regressor, data=trial_df)
flexplot(action ~ signed_omission_regressor | HMM_decay_res + , data=trial_df)
flexplot(action ~ signed_omission_regressor + FQlearning_rel_value | HMM_decay_res + relative_doubt_index, data=trial_df)
 #flexplot(FQlearning_rel_value ~ HMM_decay_res + rel_hazard_res, data=clean_df)

full = glm(action ~ signed_omission_regressor + HMM_rel_value_logodds_decay + relative_doubt_index + perseveration_regressor + relative_hazard_index, data=trial_df, family=binomial)
reduced = glm(action ~ signed_omission_regressor + HMM_rel_value_logodds_decay + relative_doubt_index + perseveration_regressor, data=trial_df, family=binomial)
compare.fits(action ~ signed_omission_regressor | perseveration_regressor + relative_doubt_index, data=trial_df, model1=full, model2=reduced)
model.comparison(full, reduced)

ideal = glm(action ~ HMM_rel_value_logodds_decay + relative_doubt_index + perseveration_regressor, data=trial_df, family=binomial)
fql_doubt = glm(action ~ FQlearning_rel_value + relative_doubt_index + perseveration_regressor, data=trial_df, family=binomial)
ql = glm(action ~ Qlearning_rel_value + perseveration_regressor, data=trial_df, family=binomial)
hmm = glm(action ~ HMM_rel_value_logodds + perseveration_regressor, data=trial_df, family=binomial)
hmm_decay = glm(action ~ HMM_rel_value_logodds_decay + perseveration_regressor, data=trial_df, family=binomial)
compare.fits(action ~ perseveration_regressor, data=trial_df, model1=full, model2=reduced)

model.comparison(ideal, hmm)
model.comparison(ideal, ql)
model.comparison(hmm_decay, hmm)
model.comparison(ideal, fql_doubt)
model.comparison(fql_doubt, ql)



flexplot(HMM_rel_value_logodds_decay ~ 1, data=trial_df_rf)
trial_df_rf_subset <- trial_df_rf %>% select(-signed_omission_regressor, 
                                             -negative_value,
                                             -consecutive_rewards_memory,-consecutive_omissions_memory,-consecutive_rewards	,-consecutive_omissions, -consecutive_failures_memory,-consecutive_failures,
                                             -left_value,-right_value,-relative_value,-left_omissions,-right_omissions,-relative_omissions,-left_cf_value,-right_cf_value,-relative_cf_value,-left_cf_omissions,-right_cf_omissions,-relative_cf_omissions,-left_monotonic_cf_value,-right_monotonic_cf_value,-relative_monotonic_cf_value,
                                             #-perseveration_regressor,
                                             -Qlearning_rel_value,
                                             -FQlearning_rel_value_fast_learn,
                                             #-FQlearning_rel_value,
                                             -rel_hazard_res,
                                             -HMM_decay_res,
                                             -HMM_rel_value_logodds,
                                             #-HMM_rel_value_logodds_decay,
                                             #-inherited_block_bias,
                                             -observer_value,
                                             -prev_action,
                                             -prev_reward,
                                             #-relative_hazard_index,
                                             #-relative_doubt_index,
                                             -relative_omissions_index,
                                             -prev_correct,
                                             -explore_run_trial,
                                             -explore_run_start,
                                             -explore_run_return,
                                             -explore_run_id,
                                             -explore_run_length,
                                             )

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



