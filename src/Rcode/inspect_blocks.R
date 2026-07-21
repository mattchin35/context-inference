require(flexplot)
require(ggplot2)
require(tidyverse)
require(dplyr)
require(tidyverse)
require(cowplot)
require(lme4)
require(glmnet)

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

################################################################################

fit_glmnet_session <- function(df_session, predictor_cols, y_col,
                               alpha = 0.5,
                               lambda_choice = "lambda.1se") {
  X <- as.matrix(df_session[, predictor_cols])
  y <- df_session[[y_col]]
  
  cv_fit <- cv.glmnet(
    x = X,
    y = y,
    family = "gaussian",
    alpha = alpha,
    standardize = TRUE
  )
  
  pred <- predict(cv_fit, newx = X, s = lambda_choice)
  resid <- y - as.numeric(pred)
  
  list(
    fit = cv_fit,
    predictions = as.numeric(pred),
    residuals = resid,
    resid_summary = tibble::tibble(
      resid_sd = sd(resid, na.rm = TRUE),
      resid_rmse = sqrt(mean(resid^2, na.rm = TRUE)),
      resid_mad = mad(resid, na.rm = TRUE),
      lambda = if (lambda_choice == "lambda.min") {
        cv_fit$lambda.min
      } else {
        cv_fit$lambda.1se
      }
    ),
    coefficients = as.matrix(coef(cv_fit, s = lambda_choice))
  )
}

################################################################################

inf_session <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260609_latent_inference/processed/CT024_2026-06-09_143852_block_performance.csv", 
                header = TRUE)
inf_session <- clean_block_dataframe(inf_session, drop_missing_tts=TRUE)
bias_session <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260611_latent_inference/processed/CT024_2026-06-11_142023_block_performance.csv", 
                        header = TRUE)
bias_session <- clean_block_dataframe(bias_session, drop_missing_tts=TRUE)
rl_session <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260520_latent_inference/processed/CT024_2026-05-20_183627_block_performance.csv", 
                         header = TRUE)
rl_session <- clean_block_dataframe(rl_session, drop_missing_tts=TRUE)
uncorr_session <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260529_latent_inference/processed/CT024_2026-05-29_143237_block_performance.csv", 
                      header = TRUE)
uncorr_session <- clean_block_dataframe(uncorr_session, drop_missing_tts=TRUE)
naive_session <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT021/CT021_20260519_latent_inference/processed/CT021_2026-05-19_130244_block_performance.csv", 
                           header = TRUE)
naive_session <- clean_block_dataframe(naive_session, drop_missing_tts=TRUE)

flexplot(previous_block_length ~ 1, data=rl_session)
flexplot(trials_to_correct ~ previous_block_length, data=rl_session)
flexplot(trials_to_correct ~ prev_n_correct, data=rl_session)
flexplot(trials_to_correct ~ prev_n_rewarded, data=rl_session)
flexplot(block_side_code ~ 1, data=inf_session)

flexplot(trials_to_correct ~ prev_n_rewarded + block_type, data=inf_session, method='lm') + ggtitle("Expert inference session")
flexplot(trials_to_correct ~ prev_n_rewarded | block_side_code, data=inf_session, method='lm') + ggtitle("Expert inference session")
summary(lm(trials_to_correct ~ prev_n_rewarded + block_side_code, data=inf_session))

#flexplot(trials_to_correct ~ previous_block_length + block_type, data=inf_session, method='lm') + ggtitle("Expert inference session")
flexplot(trials_to_correct ~ prev_n_rewarded + block_type, data=bias_session, method='lm') + ggtitle("Biased inference session")
#flexplot(trials_to_correct ~ previous_block_length + block_type, data=bias_session, method='lm') + ggtitle("Biased inference session")
flexplot(trials_to_correct ~ prev_n_rewarded + block_type, data=rl_session, method='lm') + ggtitle("Non-expert RL session")
#flexplot(trials_to_correct ~ previous_block_length + block_type, data=rl_session, method='lm') + ggtitle("Non-expert RL session")
flexplot(trials_to_correct ~ prev_n_rewarded + block_type, data=uncorr_session, method='lm') + ggtitle("Naive uncorrelated session")
#flexplot(trials_to_correct ~ previous_block_length + block_type, data=uncorr_session, method='lm') + ggtitle("Naive uncorrelated session")
flexplot(trials_to_correct ~ prev_n_rewarded + block_type, data=uncorr_session, method='lm') + ggtitle("Naive uncorrelated session")
flexplot(trials_to_correct ~ prev_n_rewarded + block_type, data=naive_session, method='lm') + ggtitle("Fully naive session")

# a quick flexplot analysis shows that the FLAT model is better for the expert inference session!
mod_inf_full = lm(trials_to_correct ~ prev_n_rewarded * block_type, data=inf_session)
mod_inf_reduced = lm(trials_to_correct ~ prev_n_rewarded + block_type, data=inf_session)
mod_inf_reward = lm(trials_to_correct ~ prev_n_rewarded, data=inf_session)
mod_inf_flat = lm(trials_to_correct ~ 1, data=inf_session)
visualize(mod_inf_minimal)
compare.fits(trials_to_correct ~ prev_n_rewarded | block_type, data=inf_session, model1=mod_inf_full, model2=mod_inf_reduced)
model.comparison(mod_inf_full, mod_inf_reduced)
model.comparison(mod_inf_full, mod_inf_flat)
model.comparison(mod_inf_reduced, mod_inf_reward)
model.comparison(mod_inf_reward, mod_inf_flat)
summary(mod_inf_flat)

# a quick flexplot analysis shows that the BLOCKTYPE model is better for the bias inference session!
mod_bias_full = lm(trials_to_correct ~ prev_n_rewarded * block_type, data=bias_session)
mod_bias_reduced = lm(trials_to_correct ~ prev_n_rewarded + block_type, data=bias_session)
mod_bias_blocktype = lm(trials_to_correct ~ block_type, data=bias_session)
mod_bias_minimal = lm(trials_to_correct ~ prev_n_rewarded, data=bias_session)
mod_bias_flat = lm(trials_to_correct ~ 1, data=bias_session)
visualize(mod_bias_full)
visualize(mod_bias_reduced)
compare.fits(trials_to_correct ~ prev_n_rewarded | block_type, data=bias_session, model1=mod_bias_full, model2=mod_bias_reduced)
model.comparison(mod_bias_full, mod_bias_reduced)
model.comparison(mod_bias_full, mod_bias_blocktype)
model.comparison(mod_bias_reduced, mod_bias_minimal)
model.comparison(mod_bias_reduced, mod_bias_blocktype)
model.comparison(mod_bias_blocktype, mod_bias_flat)
summary(mod_bias_blocktype)

# flexplot analysis shows that the REWARD model does fine for this RL session
mod_rl_full = lm(trials_to_correct ~ prev_n_rewarded * block_type, data=rl_session)
mod_rl_reduced = lm(trials_to_correct ~ prev_n_rewarded + block_type, data=rl_session)
mod_rl_reward = lm(trials_to_correct ~ prev_n_rewarded, data=rl_session)
mod_rl_blocktype = lm(trials_to_correct ~ block_type, data=rl_session)
mod_rl_flat = lm(trials_to_correct ~ 1, data=rl_session)
visualize(mod_rl_minimal)
compare.fits(trials_to_correct ~ prev_n_rewarded | block_type, data=rl_session, model1=mod_rl_full, model2=mod_rl_reduced)
model.comparison(mod_rl_full, mod_rl_reduced)
model.comparison(mod_rl_reduced, mod_rl_reward)
model.comparison(mod_rl_reward, mod_rl_flat)
model.comparison(mod_rl_reward, mod_rl_blocktype)
summary(mod_rl_reward)

# flexplot analysis shows that the FLAT model does fine for this uncorrelated session
mod_uncorr_full = lm(trials_to_correct ~ prev_n_rewarded * block_type, data=uncorr_session)
mod_uncorr_reduced = lm(trials_to_correct ~ prev_n_rewarded + block_type, data=uncorr_session)
mod_uncorr_reward = lm(trials_to_correct ~ prev_n_rewarded, data=uncorr_session)
mod_uncorr_flat = lm(trials_to_correct ~ 1, data=uncorr_session)
visualize(mod_uncorr_minimal)
compare.fits(trials_to_correct ~ prev_n_rewarded | block_type, data=uncorr_session, model1=mod_uncorr_full, model2=mod_uncorr_reduced)
model.comparison(mod_uncorr_full, mod_uncorr_reduced)
model.comparison(mod_uncorr_reduced, mod_uncorr_reward)
model.comparison(mod_uncorr_reward, mod_uncorr_flat)
summary(mod_uncorr_flat)

# Use of glmnet for regularization and feature selection
model_df <- bias_session %>%
  dplyr::select(trials_to_correct, prev_n_rewarded, block_type) %>%
  tidyr::drop_na()
X <- model.matrix(
  trials_to_correct ~ prev_n_rewarded * block_type,
  data = model_df
)[, -1, drop = FALSE]
y <- model_df$trials_to_correct

set.seed(1)
cv_elastic <- cv.glmnet(
  x = X,
  y = y,
  alpha = 0.5,
  family = "gaussian",
  standardize = TRUE
)

## Inspect the coefficients at the optimal lambda values. If a coefficient is present in lambda min and 1se, it's very real. 
## If only present in min, it's probably real but worth thinking about. If absent in both, it's probably noise.
elastic_coefs_min <- coef(cv_elastic, s = "lambda.min")
elastic_coefs_1se <- coef(cv_elastic, s = "lambda.1se")
elastic_coefs_min
elastic_coefs_1se

pred_min <- predict(cv_fit, newx = X, s = "lambda.min")
resid_min <- y - as.numeric(pred_min)

pred_1se <- predict(cv_fit, newx = X, s = "lambda.1se")
resid_1se <- y - as.numeric(pred_1se)

df_session <- df_session %>%
  mutate(
    fitted_glmnet_min = as.numeric(pred_min),
    resid_glmnet_min = y - fitted_glmnet_min,
    fitted_glmnet_1se = as.numeric(pred_1se),
    resid_glmnet_1se = y - fitted_glmnet_1se
  )

# lasso 
lasso_inf <- glmnet(X, y, alpha = 1)
cv_lasso <- cv.glmnet(
  x = X,
  y = y,
  alpha = 1,
  family = "gaussian",
  standardize = TRUE
)
coef(cv_lasso, s = "lambda.min")
coef(cv_lasso, s = "lambda.1se")

# Coefficient Inspection
## Convert the sparse matrix to a readable dataframe:
coef_df <- as.matrix(coef(cv_elastic, s = "lambda.min")) %>%
as.data.frame() %>%
tibble::rownames_to_column("term")

names(coef_df)[2] <- "coefficient"
coef_df

## Inspect coefficients at weaker penalties too:
coef(cv_elastic, s = 0.01)
coef(cv_elastic, s = 0.001)
## Plot the CV curve and coefficient path:
plot(cv_elastic)

elastic_fit <- glmnet(X, y, alpha = 0.5, family = "gaussian")
plot(elastic_fit, xvar = "lambda", label = TRUE)


# Mixed model approach. Need to take all the sessions stacked together, and use the session id 
ct024_multisession <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/cross_session_analysis/CT024_multisession_block_performance.csv", 
                        header = TRUE)
ct024_multisession <- clean_block_dataframe(ct024_multisession, drop_missing_tts=TRUE)
ct021_multisession <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT021/cross_session_analysis/CT021_multisession_block_performance.csv", 
                               header = TRUE)
ct021_multisession <- clean_block_dataframe(ct021_multisession, drop_missing_tts=TRUE)
single_mouse_multisession <- ct024_multisession

mixed_complete_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded * block_type +
    (prev_n_rewarded * block_type | source_session_id),
  data = single_mouse_multisession
)
mixed_blocklen_fit <- lmer(
  trials_to_correct ~ previous_block_length * block_type +
    (previous_block_length * block_type | source_session_id),
  data = single_mouse_multisession
)
mixed_ncorrect_fit <- lmer(
  trials_to_correct ~ prev_n_correct * block_type +
    (prev_n_correct * block_type | source_session_id),
  data = single_mouse_multisession
)
mixed_session_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded + block_type +
    (prev_n_rewarded * block_type | source_session_id),
  data = single_mouse_multisession
)
mixed_full_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded * block_type +
    (prev_n_rewarded + block_type | source_session_id),
  data = single_mouse_multisession
)
mixed_full_nointeraction_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded + block_type +
    (prev_n_rewarded + block_type | source_session_id),
  data = single_mouse_multisession
)
mixed_blocklen_nointeraction_fit <- lmer(
  trials_to_correct ~ previous_block_length + block_type +
    (previous_block_length + block_type | source_session_id),
  data = single_mouse_multisession
)
mixed_rew_blocklen_nointeraction_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded + previous_block_length + block_type +
    (prev_n_rewarded + previous_block_length + block_type | source_session_id),
  data = single_mouse_multisession
)
mixed_small_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded + block_type +
    (prev_n_rewarded | source_session_id),
  data = single_mouse_multisession
)
visualize(mixed_session_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded)
flexplot(trials_to_correct ~ prev_n_rewarded + source_session_id, data=single_mouse_multisession, method='lm') + ggtitle("Single mouse multi-session analysis")

visualize(mixed_complete_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded)
visualize(mixed_full_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded + block_type)
visualize(mixed_full_nointeraction_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded | block_type)
visualize(mixed_blocklen_nointeraction_fit, plot="model", formula = trials_to_correct ~ previous_block_length | block_type)

compare.fits(trials_to_correct ~ prev_n_rewarded, data=ct024_multisession, model1=mixed_full_fit, model2=mixed_full_nointeraction_fit)
model.comparison(mixed_complete_fit, mixed_session_fit)
model.comparison(mixed_session_fit, mixed_full_fit)
model.comparison(mixed_session_fit, mixed_full_nointeraction_fit)
model.comparison(mixed_complete_fit, mixed_full_fit)
model.comparison(mixed_complete_fit, mixed_full_nointeraction_fit)
model.comparison(mixed_full_fit, mixed_full_nointeraction_fit)
model.comparison(mixed_full_nointeraction_fit, mixed_small_fit)
model.comparison(mixed_complete_fit, mixed_blocklen_fit)

model.comparison(mixed_rew_blocklen_nointeraction_fit, mixed_blocklen_nointeraction_fit)
model.comparison(mixed_rew_blocklen_nointeraction_fit, mixed_full_nointeraction_fit)
model.comparison(mixed_blocklen_nointeraction_fit, mixed_full_nointeraction_fit)

# session interaction, full variables >/< no interaction, full variables
# full interactions >/< no interactions, full variables
# no interaction, full variables > fixed interaction, full variables
# full variables >> less variables
summary(mixed_session_fit)
coef(mixed_session_fit)$source_session_id
summary(mixed_complete_fit)
coef(mixed_complete_fit)$source_session_id
summary(mixed_full_nointeraction_fit)
coef(mixed_full_nointeraction_fit)$source_session_id
cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded, object=mixed_session_fit)
cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded + block_type, data = ct024_multisession, random = ~(prev_n_rewarded * block_type | source_session_id))


#### regularization example code, do not run - only for reference
# Fit Regularization
# alpha = 1 specifies Lasso
# alpha = 0 specifies Ridge
lasso_model <- glmnet(X, y, alpha = 1)
ridge_model <- glmnet(X, y, alpha = 0)

# Perform cross-validation to find the optimal penalty strength (lambda)
cv_lasso <- cv.glmnet(X, y, alpha = 1)
best_lambda_lasso <- cv_lasso$lambda.min
lasso_coefs <- coef(cv_lasso, s = "lambda.min")
print("Lasso Coefficients (Notice some are absolute zero):")
print(lasso_coefs)

cv_ridge <- cv.glmnet(X, y, alpha = 0)
best_lambda_ridge <- cv_ridge$lambda.min
print("Ridge Coefficients (Shrunk but none are zero):")
print(ridge_coefs)

# Make Predictions on New Data
pred_lasso <- predict(cv_lasso, newx = new_data, s = "lambda.min")
pred_ridge <- predict(cv_ridge, newx = new_data, s = "lambda.min")

####

# multi-mouse multisession!
multimouse_df <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/cross_mouse_analysis/cross_mouse_multisession_block_performance.csv", 
                               header = TRUE)
multimouse_df <- clean_block_dataframe(multimouse_df, drop_missing_tts=TRUE)
multimouse_df <- multimouse_df %>%
  mutate(
    prev_rewards_z = as.numeric(scale(multimouse_df$prev_n_rewarded)),
    TTS_z = as.numeric(scale(multimouse_df$trials_to_correct))
  )

flexplot(TTS_z ~ 1, data=multimouse_df)
flexplot(trials_to_correct ~ 1, data=multimouse_df)
flexplot(prev_rewards_z ~ 1, data=multimouse_df)
flexplot(prev_n_rewarded ~ 1, data=multimouse_df)
flexplot(block_type ~ 1, data=multimouse_df)

multimouse_allinteraction_nested_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded * block_type +
    (prev_n_rewarded * block_type | mouse/source_session_id),
  data = multimouse_df
)
multimouse_allinteraction_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded * block_type +
    (prev_n_rewarded * block_type | source_session_id),
  data = multimouse_df
)
multimouse_blocklen_fit <- lmer(
  trials_to_correct ~ previous_block_length * block_type +
    (previous_block_length * block_type | source_session_id),
  data = multimouse_df
)
multimouse_sessioninteraction_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded + block_type +
    (prev_n_rewarded * block_type | source_session_id),
  data = multimouse_df
)
multimouse_sessioninteraction_nested_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded + block_type +
    (prev_n_rewarded * block_type | mouse/source_session_id),
  data = multimouse_df
)
multimouse_allfixedinteraction_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded * block_type +
    (prev_n_rewarded + block_type | source_session_id),
  data = multimouse_df
)
multimouse_nointeraction_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded + block_type +
    (prev_n_rewarded + block_type | source_session_id),
  data = multimouse_df
)
multimouse_nointeraction_nested_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded + block_type +
    (prev_n_rewarded + block_type | mouse/source_session_id),
  data = multimouse_df
)
multimouse_small_fit <- lmer(
  trials_to_correct ~ prev_n_rewarded +
    (1 | source_session_id),
  data = multimouse_df
)

flexplot(TTS_z ~ prev_rewards_z + block_type, data=multimouse_df, method='lm') + ggtitle("Multi-mouse multi-session analysis")
flexplot(trials_to_correct ~ prev_n_rewarded + mouse, data=multimouse_df, method='lm') + ggtitle("Multi-mouse multi-session analysis")
flexplot(trials_to_correct ~ prev_n_rewarded + source_session_id, data=multimouse_df, method='lm') + ggtitle("Multi-mouse multi-session analysis") + 
  theme(legend.position = 'none')

visualize(multimouse_sessioninteraction_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded)
visualize(multimouse_sessioninteraction_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded, sample=50)
visualize(multimouse_allinteration_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded)
visualize(multimouse_allfixedinteraction_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded)
visualize(multimouse_nointeraction_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded)
visualize(multimouse_small_fit, plot="model", formula = trials_to_correct ~ prev_n_rewarded)

compare.fits(trials_to_correct ~ prev_n_rewarded, data=ct024_multisession, model1=mixed_full_fit, model2=mixed_full_nointeraction_fit)
model.comparison(multimouse_allinteraction_nested_fit, multimouse_sessioninteraction_nested_fit)
model.comparison(multimouse_sessioninteraction_nested_fit, multimouse_nointeraction_nested_fit)

model.comparison(multimouse_allinteraction_nested_fit, multimouse_allinteraction_fit)
model.comparison(multimouse_sessioninteraction_nested_fit, multimouse_sessioninteraction_fit)
model.comparison(multimouse_nointeraction_nested_fit, multimouse_nointeraction_fit)

model.comparison(multimouse_allinteraction_fit, multimouse_blocklen_fit)

model.comparison(multimouse_allinteraction_fit, multimouse_sessioninteraction_fit)
#model.comparison(multimouse_sessioninteraction_fit, mixed_full_fit)
model.comparison(multimouse_sessioninteraction_fit, multimouse_nointeraction_fit)
#model.comparison(multimouse_allinteration_fit, mixed_full_fit)
model.comparison(multimouse_allinteration_fit, multimouse_nointeraction_fit)
#model.comparison(mixed_full_fit, multimouse_nointeraction_fit)
model.comparison(multimouse_nointeraction_fit, mixed_small_fit)

# session interaction, full variables >/< no interaction, full variables
# full interactions >/< no interactions, full variables
# no interaction, full variables > fixed interaction, full variables
# full variables >> less variables
summary(mixed_session_fit)
coef(mixed_session_fit)$source_session_id
summary(mixed_complete_fit)
coef(mixed_complete_fit)$source_session_id
summary(mixed_full_nointeraction_fit)
coef(mixed_full_nointeraction_fit)$source_session_id
cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded, object=multimouse_nointeraction_fit)
cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded, object=multimouse_sessioninteraction_fit)
cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded + block_type, data = ct024_multisession, random = ~(prev_n_rewarded * block_type | source_session_id))


###################################################
###################################################

ct024_cross_session_path <- "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/cross_session_analysis"

ct024_resid_x_side <- read.csv(
  file.path(
    ct024_cross_session_path,
    "CT024_block_residual_model_summary_rewards_x_side.csv"
  ),
  header = TRUE,
  na.strings = c("None", "NA", "")
)

ct024_resid_plus_side <- read.csv(
  file.path(
    ct024_cross_session_path,
    "CT024_block_residual_model_summary_rewards_plus_side.csv"
  ),
  header = TRUE,
  na.strings = c("None", "NA", "")
)

ct024_resid_models <- dplyr::bind_rows(
  ct024_resid_x_side,
  ct024_resid_plus_side
)

# Inspect the easy plotting columns
ct024_resid_models %>%
  dplyr::select(
    training_day,
    date,
    session_id,
    model_formula,
    model_type,
    lambda_choice,
    right_intercept,
    left_intercept,
    right_reward_slope,
    left_reward_slope,
    residual_rmse,
    residual_mad_scaled
  )

ct024_elastic_min <- ct024_resid_models %>%
  dplyr::filter(
    model_type == "elastic_net",
    lambda_choice == "lambda.min"
  )
ct024_elastic_min_x_side <- ct024_resid_models %>%
  dplyr::filter(
    model_formula == "rewards_x_side",
    model_type == "elastic_net",
    lambda_choice == "lambda.min"
  )
ct024_elastic_min_plus_side <- ct024_resid_models %>%
  dplyr::filter(
    model_formula == "rewards_plus_side",
    model_type == "elastic_net",
    lambda_choice == "lambda.min"
  )
flexplot(left_intercept ~ 1, data=ct024_elastic_min_plus_side)
flexplot(right_intercept ~ 1, data=ct024_elastic_min_plus_side)
flexplot(left_reward_slope ~ 1, data=ct024_elastic_min_plus_side)
flexplot(right_reward_slope ~ 1, data=ct024_elastic_min_plus_side)
flexplot(side_intercept_delta_left_minus_right ~ 1, data=ct024_elastic_min_plus_side)
#flexplot(side_reward_slope_delta_left_minus_right ~ 1, data=ct024_elastic_min_plus_side)
flexplot(residual_iqr ~ 1, data=ct024_elastic_min_plus_side, bins=20)
flexplot(residual_rmse ~ 1, data=ct024_elastic_min_plus_side, bins=20)


flexplot(left_intercept ~ 1, data=ct024_elastic_min_x_side)
flexplot(right_intercept ~ 1, data=ct024_elastic_min_x_side)
flexplot(left_reward_slope ~ 1, data=ct024_elastic_min_x_side)
flexplot(right_reward_slope ~ 1, data=ct024_elastic_min_x_side)
flexplot(side_intercept_delta_left_minus_right ~ 1, data=ct024_elastic_min_x_side)
flexplot(side_reward_slope_delta_left_minus_right ~ 1, data=ct024_elastic_min_x_side)
flexplot(residual_iqr ~ 1, data=ct024_elastic_min_x_side)
flexplot(residual_rmse ~ 1, data=ct024_elastic_min_x_side)


