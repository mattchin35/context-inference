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
# trial_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251202/processed/CT014_2025-12-02_151940_augmented_trials.csv", 
#                      header = TRUE)#,
#stringsAsFactors = FALSE)  # Don't convert strings to factors
trial_df <- cleanup_trial_dataframe(trial_df, include_model_regressors = TRUE)
clean_df <- curate_trial_analysis_dataframe(trial_df)

# Run a random-forest / VarPro comparison for a named predictor set.
#
# Args:
#   data_frame: data.frame with shape (n_trials, n_columns). Trialwise analysis
#     table containing an `action` column and all predictors listed below.
#   predictors: character vector of predictor column names. Each predictor is a
#     trialwise scalar regressor aligned to the rows in `data_frame`.
#   label: character scalar used for printed output.
#
# Returns:
#   named list with `rf_model` and `varfit` objects.
run_choice_importance <- function(data_frame, predictors, label) {
  analysis_columns <- c("action", predictors)
  model_df <- data_frame %>% dplyr::select(dplyr::all_of(analysis_columns))
  varpro_df <- model_df
  
  for (predictor_name in predictors) {
    if (is.factor(varpro_df[[predictor_name]])) {
      varpro_df[[predictor_name]] <- as.numeric(as.character(varpro_df[[predictor_name]]))
    }
  }
  
  model_formula <- stats::as.formula(
    paste("action ~", paste(predictors, collapse = " + "))
  )
  
  cat("\n###", label, "###\n")
  rf_model <- rfsrc(model_formula, data = model_df)
  varfit <- varpro(model_formula, data = varpro_df)
  print(rf_model)
  importance(varfit, plot.it = TRUE)
  
  list(rf_model = rf_model, varfit = varfit)
}

legacy_value_predictors <- c(
  "relative_value", "relative_omissions", "relative_doubt_index",
  "prev_action", "prev_reward"
)
legacy_cf_predictors <- c(
  "relative_cf_value", "relative_omissions", "relative_doubt_index",
  "prev_action", "prev_reward"
)
current_model_predictors <- c(
  "FQlearning_rel_value", "HMM_rel_value_logodds", "HMM_rel_value_logodds_decay",
  "relative_omissions", "relative_doubt_index", "prev_action", "prev_reward"
)

flexplot(relative_value ~ 1, data = clean_df)
flexplot(relative_cf_value ~ 1, data = clean_df)
flexplot(FQlearning_rel_value ~ 1, data = clean_df)
flexplot(HMM_rel_value_logodds ~ 1, data = clean_df)
flexplot(HMM_rel_value_logodds_decay ~ 1, data = clean_df)

legacy_value_models <- run_choice_importance(clean_df, legacy_value_predictors, "Legacy relative_value predictors")
legacy_cf_models <- run_choice_importance(clean_df, legacy_cf_predictors, "Legacy relative_cf_value predictors")
current_model_models <- run_choice_importance(clean_df, current_model_predictors, "Current model-value predictors")

flexplot(
  action ~ FQlearning_prob_left_approx + FQlearning_rel_value,
  method = "logistic",
  data = clean_df,
  jitter = c(.01, .1)
)
flexplot(
  action ~ HMM_prob_left_logodds_approx | HMM_rel_value_logodds,
  method = "logistic",
  data = clean_df,
  jitter = c(.01, .1)
)
flexplot(
  action ~ HMM_prob_left_logodds_decay_approx | HMM_rel_value_logodds_decay,
  method = "logistic",
  data = clean_df,
  jitter = c(.01, .1)
)

full_current <- glm(
  action ~ FQlearning_rel_value + HMM_rel_value_logodds + HMM_rel_value_logodds_decay +
    relative_omissions + relative_doubt_index + prev_action + prev_reward,
  data = clean_df,
  family = binomial
)
reduced_fql <- glm(
  action ~ FQlearning_rel_value + relative_omissions + relative_doubt_index + prev_action + prev_reward,
  data = clean_df,
  family = binomial
)
reduced_hmm_logodds <- glm(
  action ~ HMM_rel_value_logodds + relative_omissions + relative_doubt_index + prev_action + prev_reward,
  data = clean_df,
  family = binomial
)
reduced_hmm_decay <- glm(
  action ~ HMM_rel_value_logodds_decay + relative_omissions + relative_doubt_index + prev_action + prev_reward,
  data = clean_df,
  family = binomial
)

model.comparison(full_current, reduced_fql)
model.comparison(full_current, reduced_hmm_logodds)
model.comparison(full_current, reduced_hmm_decay)

flexplot(
  action ~ relative_value + relative_omissions | inferred_strategy + prev_action,
  data = clean_df,
  method = "logistic"
)
flexplot(
  action ~ relative_cf_value + relative_doubt_index | inferred_strategy + prev_action,
  data = clean_df,
  method = "logistic"
)

control <- glmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_BOBYQA"))
mixmod_relative_value <- glmer(
  action ~ relative_value + relative_omissions + prev_action * prev_reward +
    (relative_value + relative_omissions | inferred_strategy),
  data = clean_df,
  family = "binomial",
  control = control
)
mixmod_relative_cf <- glmer(
  action ~ relative_cf_value + relative_doubt_index + prev_action * prev_reward +
    (relative_cf_value + relative_doubt_index | inferred_strategy),
  data = clean_df,
  family = "binomial",
  control = control
)

strategy_sample_size <- safe_group_sample_size(clean_df$inferred_strategy, requested_size = 3L)
visualize(mixmod_relative_value, sample = strategy_sample_size)
visualize(mixmod_relative_cf, sample = strategy_sample_size)
estimates(mixmod_relative_value)
estimates(mixmod_relative_cf)
model.comparison(mixmod_relative_value, mixmod_relative_cf)

pred_probs <- predict(mixmod_relative_cf, type = "response")
predictions <- ifelse(pred_probs > 0.5, 1, 0)
mean(predictions == clean_df$action) * 100

##############################################################
# inspect and analyze blocks
##############################################################

block_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/processed/CT014_2025-12-16_153200_block_performance.csv",
                     header = TRUE)#,
# block_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251202/processed/CT014_2025-12-02_151940_block_performance.csv", 
#                      header = TRUE)#,
#stringsAsFactors = FALSE)  # Don't convert strings to factors
block_df <- clean_block_dataframe(block_df)

flexplot(trials_to_correct ~  prev_n_rewarded  | inferred_strategy,  data=block_df, method='lm') + theme(
  panel.grid.major = element_blank(), panel.grid.minor = element_blank()   # Remove minor grid lines
) + #xlab("Trials to correct") + ylab("Previous rewards") + 
  labs(
    x = "", 
    y = "Previous rewards",
  ) + 
  facet_grid( ~ inferred_strategy,
              labeller = labeller(inferred_strategy = c("0" = "RL",
                                                "1" = "Inf"
                                                ))) +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16, angle = 90),
    strip.text = element_text(size = 16)
    )

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


### compare the different behavior states/strategies ###########################
mod_fullest = lm(trials_to_correct ~ prev_n_rewarded * inferred_strategy * bias_full_flag, data = block_df)  # use this one!!!
#mod_fullest = lm(trials_to_correct ~ prev_n_rewarded * inferred_strategy + prev_n_rewarded * bias_full_flag, data = block_df)
mod_inferred = lm(trials_to_correct ~ prev_n_rewarded * inferred_strategy, data = block_df)
# mod_declared = lm(trials_to_correct ~ prev_n_rewarded * declared_strategy, data = block_df)
# mod_bias = lm(trials_to_correct ~ prev_n_rewarded * bias_full_flag, data = block_df)
visualize(mod_fullest)
visualize(mod_inferred)
estimates(mod_inferred)

#compare.fits(trials_to_correct ~ prev_n_rewarded, data=block_df, model1=mod_inferred, model2=mod_declared)
#model.comparison(mod_inferred, mod_declared)
#compare.fits(trials_to_correct ~ prev_n_rewarded, data=block_df, model1=mod_declared, model2=mod_bias)
#model.comparison(mod_declared, mod_bias)
compare.fits(trials_to_correct ~ prev_n_rewarded | bias_full_flag + inferred_strategy, data=block_df, model1=mod_fullest, model2=mod_inferred)
model.comparison(mod_fullest, mod_inferred)

#################################################################

# compare a simple flat fit to a linear model like RL

# apparently I need to put everything into long format
# mixmod_hmm = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded + bias_full_flag | inferred_strategy), data = block_df)
# mixmod_full = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | bias_full_flag + inferred_strategy), data = block_df)
# mixmod_rew = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | bias_full_flag), data = block_df)
# mixmod_declared = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | declared_strategy), data = block_df)
# mixmod_inferred = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | inferred_strategy), data = block_df)
#mixmod_rew = lmer(trials_to_correct ~ prev_n_rewarded + (1 | bias_full_flag), data = block_df)
control = lmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_NELDERMEAD"))
mod_rewards = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | inferred_strategy),
               data = block_df, control=control)
# mod_bias = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | bias_full_flag), data = block_df, control=control)

# this one is not supported by AIC/BIC/Bayes, even tho it looks like a better fit?
mod_blocks_fullest = lmer(trials_to_correct ~ prev_n_rewarded + bias_full_flag + (prev_n_rewarded + bias_full_flag | inferred_strategy),
                    data = block_df, control=control)

# this one is supported over the one without the block bias flag, and seems preferable to the extra bias random effect
mod_blocks = lmer(trials_to_correct ~ prev_n_rewarded + bias_full_flag + (prev_n_rewarded | inferred_strategy), 
                    data = block_df, control=control)

visualize(mod_blocks_fullest, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded | bias_full_flag + inferred_strategy)
visualize(mod_blocks, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded + bias_full_flag | inferred_strategy)
model.comparison(mod_blocks_fullest, mod_blocks)
model.comparison(mod_blocks, mod_rewards)
estimates(mod_blocks_fullest)

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
