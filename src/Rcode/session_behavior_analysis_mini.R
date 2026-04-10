#require(tidyverse)
require(dplyr)
require(tidyverse)
require(flexplot)
require(cowplot)
require(lme4)
require(ggplot2)
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

source_rcode("preprocessing/trial_preprocessing.R")
source_rcode("preprocessing/block_preprocessing.R")

trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/processed/CT014_2025-12-16_153200_augmented_trials.csv", 
                     header = TRUE)#,
#stringsAsFactors = FALSE)  # Don't convert strings to factors

trial_df <- cleanup_trial_dataframe(trial_df, include_model_regressors = TRUE)
clean_df <- curate_trial_analysis_dataframe(trial_df)


set.seed(1)
#rf_model = randomforestthing(action~., data=trial_df)
rf_model = rfsrc(action ~ ., data=clean_df)
varfit = varpro(action ~ ., data=clean_df)
imp = importance(varfit)
print(rf_model)
importance(varfit, plot.it = TRUE)

# iteratively removing things so all important predictors are left acroos all strategies
set.seed(1)
small_df <- clean_df %>% select(-cur_trial, -cur_trial_in_block, -trial_time_since_start, -choice_time_since_start)
small_df <- small_df %>% select(-reward)
small_df <- small_df %>% select(-correct)
#small_df <- removeRowsByColumn(small_df, "correct", function(x) x == 1)
#small_df <- removeRowsByColumn(small_df, "reward", function(x) x == 1)
#small_df <- small_df %>% select(-RFLR_greedy_action, -FQlearning_greedy_action, -HMM_greedy_action, -Qlearning_greedy_action)
#small_df <- clean_df %>% select(-RFLR_rel_value, -FQlearning_rel_value, -HMM_rel_value, -Qlearning_rel_value)
#small_df <- clean_df %>% select(-RFLR_prob_left, -FQlearning_prob_left, -HMM_prob_left, -Qlearning_prob_left)
small_df <- small_df %>% select(-consecutive_rewards_memory, -consecutive_failures_memory, -consecutive_failures, -consecutive_rewards)
small_df <- small_df %>% select(-time_to_choice, -negative_value, -cur_block)
small_df <- small_df %>% select(-RFLR_prob_left, -RFLR_rel_value, -RFLR_greedy_action) 
small_df <- small_df %>% select(-FQlearning_prob_left, -FQlearning_rel_value, -FQlearning_greedy_action)
small_df <- small_df %>% select(-Qlearning_prob_left, -Qlearning_rel_value, -Qlearning_greedy_action)
small_df <- small_df %>% select(-HMM_prob_left, -HMM_rel_value, -HMM_greedy_action)
small_df <- small_df %>% select(-inherited_strategy,-inherited_bias_flag)
#small_df <- small_df %>% select(-inferred_strategy)
#small_df <- small_df %>% select(-prev_action, -prev_reward)
#small_df <- small_df %>% select(-FQlearning_rel_value, -HMM_rel_value)#, -HMM_prob_left, -FQlearning_prob_left)
#small_df <- small_df %>% select(-HMM_prob_left, -FQlearning_prob_left)
small_df <- small_df %>% select(-relative_value, -relative_nonneg_value, -relative_omissions)
#small_df <- small_df %>% select(-right_omissions, -left_omissions)
#small_df <- small_df %>% select(-right_value, -left_value)
small_df <- small_df %>% select(-right_nonneg_value, -left_nonneg_value)
#print(small_df)

rf_small = rfsrc(action ~ ., data=small_df)
varfit_small = varpro(action ~ ., data=small_df)
imp_small = importance(varfit_small)
print(rf_small)
importance(varfit_small, plot.it = TRUE)

#small_df <- small_df %>% select(-RFLR_greedy_action, -HMM_greedy_action, -Qlearning_greedy_action, -FQlearning_greedy_action)
#small_df <- small_df %>% select(-consecutive_rewards_memory, -HMM_rel_value)
#small_df <- small_df %>% select(-consecutive_failures, -consecutive_rewards, -time_to_choice)
#small_df <- small_df %>% select(-RFLR_prob_left, -RFLR_rel_value, -Qlearning_prob_left, -Qlearning_rel_value)
#small_df <- small_df %>% select(-Qlearning_rel_value, -HMM_rel_value, -consecutive_rewards_memory, -consecutive_failures)c

# strategies: -HMM_prob_left, -HMM_rel_value, -Qlearning_rel_value, -Qlearning_prob_left, -RFLR_rel_value, -RFLR_prob_left, -FQlearning_prob_left, -FQlearning_rel_value, -correct, -reward)
# decision variables: negative_value, consecutive_rewards_memory, consecutive_failures_memory, consecutive_rewards, consecutive_failures 
# seemingly unimportant attributes (for now at least) -led_on_time, -led_off_time, -p_active_rew, -p_inactive_rew, -p_switch, -session_ID) # removing all of these stays fine
# trial/block qualities: (-cur_trial, -cur_trial_in_block, -cur_block, -start_time, -choice_time, -reward_time, -time_to_choice, -reward_time))

# try another way!
hmm_df <- clean_df %>% select(-Qlearning_rel_value, -Qlearning_prob_left, -RFLR_rel_value, -RFLR_prob_left, -FQlearning_prob_left, -FQlearning_rel_value, -correct, -reward)
hmm_df <- hmm_df %>% select(-led_on_time, -led_off_time, -p_active_rew, -p_inactive_rew, -p_switch, -session_ID, -reward_time) # remove all of these on principle or because they should be unimportant for now
#hmm_df <- hmm_df %>% select(-cur_trial, -cur_trial_in_block, -cur_block, -start_time, -choice_time, -time_to_choice)
rf_hmm = rfsrc(action ~ ., data=hmm_df)
print(rf_hmm)
varfit_hmm = varpro(action ~ ., data=hmm_df)
imp_hmm = importance(varfit_hmm)
importance(varfit_hmm, plot.it = TRUE)

### try some logistic relationships now! See how you compare to the RF plot, and why you might need some state/strategy predictors
flexplot(action ~ HMM_prob_left | HMM_rel_value, method="logistic", data=clean_df, jitter=c(.01,.1)) # ghost.line="gray"
flexplot(action ~ FQlearning_prob_left + FQlearning_rel_value, method="logistic", data=clean_df, jitter=c(.01,.1)) # ghost.line="gray"
flexplot(action ~ HMM_prob_left | HMM_rel_value + FQlearning_prob_left , method="logistic", data=clean_df, jitter=c(.01,.1)) # ghost.line="gray"
#flexplot(action ~ HMM_greedy_action + correct, data=small_df, jitter=c(.1,.1))

full = glm(action ~ HMM_rel_value * correct + HMM_prob_left * correct + FQlearning_rel_value * correct + FQlearning_prob_left * correct, data=clean_df, family=binomial)
#reduced = glm(action ~ HMM_rel_value * correct + HMM_prob_left * correct, data=clean_df, family=binomial)
#compare.fits(action ~ HMM_rel_value | correct, data=clean_df, model1=full, model2=reduced)
reduced = glm(action ~ FQlearning_rel_value * correct + FQlearning_prob_left * correct, data=clean_df, family=binomial)
compare.fits(action ~ FQlearning_rel_value | correct, data=clean_df, model1=full, model2=reduced)
model.comparison(full, reduced)
#estimates(full)

# compare HMM and FQL predictors to only using HMM or FQL
full_multimodel = glm(action ~ HMM_rel_value * HMM_prob_left + FQlearning_rel_value * FQlearning_prob_left, data=clean_df, family=binomial)
#reduced_hmm = glm(action ~ HMM_rel_value * HMM_prob_left, data=clean_df, family=binomial)
#compare.fits(action ~ HMM_rel_value | HMM_prob_left, data=clean_df, model1=full, model2=reduced_hmm)
#model.comparison(full, reduced_hmm)
reduced_fql = glm(action ~ FQlearning_rel_value * FQlearning_prob_left, data=clean_df, family=binomial)
compare.fits(action ~ FQlearning_prob_left , data=clean_df, model1=full_multimodel, model2=reduced_fql)
model.comparison(full_multimodel, reduced_fql)

# compare QL and FQL
full_multimodel = glm(action ~ Qlearning_rel_value * Qlearning_prob_left + FQlearning_rel_value * FQlearning_prob_left, data=clean_df, family=binomial)
reduced_ql = glm(action ~ Qlearning_rel_value * Qlearning_prob_left, data=clean_df, family=binomial)
reduced_fql = glm(action ~ FQlearning_rel_value * FQlearning_prob_left, data=clean_df, family=binomial)
compare.fits(action ~ FQlearning_prob_left | FQlearning_rel_value, data=clean_df, model1=full, model2=reduced_fql)
model.comparison(full, reduced_fql)


mod_hmm = glm(action ~ HMM_rel_value * correct + HMM_prob_left * correct, data=clean_df, family=binomial)
mod_fql = glm(action ~FQlearning_rel_value * correct + FQlearning_prob_left * correct, data=clean_df, family=binomial)
#compare.fits(action ~ HMM_rel_value | correct, data=clean_df, model1=mod_hmm, model2=mod_fql)
model.comparison(mod_hmm, mod_fql)

mod_hmm = glm(action ~ HMM_rel_value * HMM_prob_left, data=clean_df, family=binomial)
mod_fql = glm(action ~FQlearning_rel_value * FQlearning_prob_left, data=clean_df, family=binomial)
#compare.fits(action ~ HMM_rel_value | correct, data=clean_df, model1=mod_hmm, model2=mod_fql)
model.comparison(mod_hmm, mod_fql)

#compare.fits(action ~ .)

# try some RFs where you only have HMM or only have FQL or only have Logistic
FQL_df <- small_df %>% select(-Qlearning_rel_value, -Qlearning_prob_left, -RFLR_rel_value, -RFLR_prob_left, -HMM_prob_left, -HMM_rel_value) 
rf_fql = rfsrc(action ~ ., data=FQL_df)
print(rf_fql)
varfit_fql = varpro(action ~ ., data=FQL_df)
imp_fql = importance(varfit_fql)
importance(varfit_fql, plot.it = TRUE)

#* relative_omissions + prev_action + prev_reward
### GLIM for Generalized mixed model. WOW STATS!!
flexplot(relative_omissions ~ 1, data=clean_df)
flexplot(action ~ relative_value | inferred_strategy, data=clean_df, method="logistic")
control = glmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_NELDERMEAD"))
control = glmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_BOBYQA"))
control = glmerControl(optimizer = "Nelder_Mead")
mixmod_trials = glmer(action ~ relative_value + relative_omissions + prev_action + prev_reward +
                        (relative_value + relative_omissions | inferred_strategy), 
                      data=clean_df, family="binomial", control=control)
allFit(mixmod_trials)
visualize(mixmod_trials)
estimates(mixmod_trials)







##############################################################
# inspect and analyze blocks
##############################################################

block_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/processed/CT014_2025-12-16_153200_block_performance.csv", 
                     header = TRUE)#,
#stringsAsFactors = FALSE)  # Don't convert strings to factors
block_df <- clean_block_dataframe(block_df)

flexplot(trials_to_correct ~  prev_n_rewarded | bias_full_flag + inferred_strategy,  data=block_df)
flexplot(trials_to_correct ~  prev_n_rewarded | declared_strategy,  data=block_df)
#flexplot(trials_to_correct ~ prev_n_correct + inferred_strategy, data=block_df)
flexplot(trials_to_correct ~ prev_n_rewarded | inferred_strategy, data=block_df)
#flexplot(trials_to_correct ~ prev_n_rewarded + declared_strategy, data=block_df)
#flexplot(trials_to_correct ~ prev_n_rewarded + inferred_strategy, data=block_df)

flexplot(trials_to_correct ~ prev_n_correct + bias_full_flag, data=block_df)
#flexplot(trials_to_correct ~ prev_n_correct  bias_full_flag, data=block_df)
#flexplot(trials_to_correct ~ prev_n_rewarded | bias_full_flag, data=block_df)
flexplot(trials_to_correct ~ prev_n_rewarded | declared_strategy, data=block_df)
flexplot(trials_to_correct ~ block_type, data=block_df)
#flexplot(trials_to_correct ~ prev_n_rewarded + prev_n_correct | bias_full_flag, data=block_df)
#flexplot(trials_to_correct ~ prev_consecutive_rewards, data=block_df)
#flexplot(trials_to_correct ~ prev_consecutive_rewards_memory, data=block_df)

mod_rewards_full = lm(trials_to_correct ~ prev_n_rewarded * declared_strategy, data = block_df)
mod_rewards = lm(trials_to_correct ~ prev_n_rewarded + bias_full_flag, data = block_df)
mod_rew_minimal = lm(trials_to_correct ~ prev_n_rewarded, data = block_df)
compare.fits(trials_to_correct ~ prev_n_rewarded | bias_full_flag, data=block_df, model1=mod_rewards_full, model2=mod_rewards)
model.comparison(mod_rewards_full, mod_rewards)
visualize(mod_rewards_full)
estimates(mod_rewards_full)
compare.fits(trials_to_correct ~ prev_n_rewarded | bias_full_flag, data=block_df, model1=mod_rewards_full, model2=mod_rew_minimal)
model.comparison(mod_rewards_full, mod_rew_minimal)

mod_correct = lm(trials_to_correct ~ prev_n_correct * bias_full_flag, data = block_df)
visualize(mod_correct)

### compare the different behavior states/strategies #############################
mod_fullest = lm(trials_to_correct ~ prev_n_rewarded * inferred_strategy * bias_full_flag, data = block_df)
#mod_fullest = lm(trials_to_correct ~ prev_n_rewarded * inferred_strategy + prev_n_rewarded * bias_full_flag, data = block_df)
mod_inferred = lm(trials_to_correct ~ prev_n_rewarded * inferred_strategy, data = block_df)
mod_declared = lm(trials_to_correct ~ prev_n_rewarded * declared_strategy, data = block_df)
mod_bias = lm(trials_to_correct ~ prev_n_rewarded * bias_full_flag, data = block_df)
visualize(mod_fullest)
visualize(mod_inferred)
estimates(mod_inferred)
visualize(mod_declared)
estimates(mod_declared)
visualize(mod_bias)
estimates(mod_bias)

#compare.fits(trials_to_correct ~ prev_n_rewarded, data=block_df, model1=mod_inferred, model2=mod_declared)
#model.comparison(mod_inferred, mod_declared)
#compare.fits(trials_to_correct ~ prev_n_rewarded, data=block_df, model1=mod_declared, model2=mod_bias)
#model.comparison(mod_declared, mod_bias)
compare.fits(trials_to_correct ~ prev_n_rewarded| bias_full_flag + inferred_strategy, data=block_df, model1=mod_inferred, model2=mod_bias)
model.comparison(mod_inferred, mod_bias)

compare.fits(trials_to_correct ~ prev_n_rewarded | bias_full_flag + inferred_strategy , data=block_df, model1=mod_fullest, model2=mod_bias)
model.comparison(mod_fullest, mod_bias)
compare.fits(trials_to_correct ~ prev_n_rewarded | bias_full_flag + inferred_strategy, data=block_df, model1=mod_fullest, model2=mod_inferred)
model.comparison(mod_fullest, mod_inferred)

##########################################33

# compare a simple flat fit to a linear model like RL

# apparently I need to put everything into long format
mixmod_hmm = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded + bias_full_flag | inferred_strategy), data = block_df)
mixmod_full = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | bias_full_flag + inferred_strategy), data = block_df)
mixmod_rew = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | bias_full_flag), data = block_df)
mixmod_declared = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | declared_strategy), data = block_df)
mixmod_inferred = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | inferred_strategy), data = block_df)
#mixmod_rew = lmer(trials_to_correct ~ prev_n_rewarded + (1 | bias_full_flag), data = block_df)
visualize(mixmod_hmm, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded + bias_full_flag | inferred_strategy)
visualize(mixmod_full, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded + bias_full_flag + inferred_strategy)
summary(mixmod_rew)
visualize(mixmod_rew, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded | bias_full_flag)
visualize(mixmod_declared, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded | declared_strategy)
summary(mixmod_declared)
visualize(mixmod_inferred, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded | inferred_strategy)

cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded | inferred_strategy, object = mixmod_inferred)
#cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded | bias_full_flag, object = mixmod_rew, adjust="all")
allFit(mixmod_hmm)
control = lmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_NELDERMEAD"))
mod_hmm = lmer(trials_to_correct ~ prev_n_rewarded + 
                 (prev_n_rewarded | inferred_strategy), 
               data = block_df, control=control)
mod_bias = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | bias_full_flag), data = block_df, control=control)

mod_hmm_full = lmer(trials_to_correct ~ prev_n_rewarded + bias_full_flag + 
                      (prev_n_rewarded | inferred_strategy), 
                    data = block_df, control=control)
visualize(mod_hmm, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded | inferred_strategy)
visualize(mod_hmm_full, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded + bias_full_flag | inferred_strategy)
model.comparison(mod_hmm_full, mod_hmm)

#mixmod_smooth = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | bias_full_flag), 
#                     data = block_df %>% mutate(prev_n_rewarded = scale(prev_n_rewarded),
#                                                trials_to_correct = scale(trials_to_correct)))

small_blocks <- block_df %>% select(-block_ix,-block_type,-prev_consecutive_rewards,-prev_consecutive_rewards_memory,-prev_n_correct,-normalized_switches,-confusion_flag,
                                    -n_correct,-percent_correct,-n_rewarded,-mean_choice_time,
                                    -median_choice_time,-std_choice_time,-session_ID,-bias_rl,
                                    -bias_inf,-bias_rl_flag,-bias_inf_flag,-declared_strategy)
rf_blocks = rfsrc(trials_to_correct ~ ., data=small_blocks)
print(rf_blocks)
varfit_blocks = varpro(trials_to_correct ~ ., data=small_blocks)
imp_blocks = importance(varfit_blocks)
importance(varfit_blocks, plot.it = TRUE)
