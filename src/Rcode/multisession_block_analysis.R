# multisession analysis!!#
require(tidyverse)
require(dplyr)
require(tidyverse)
require(flexplot)
require(cowplot)
require(lme4)
require(ggplot2)
require(randomForestSRC)  #not sure about flexplot compatibility
#require(randomForest)
library(varPro)

source("preprocessing/block_preprocessing.R")



##############################################################
# inspect and analyze blocks
##############################################################

df1 <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/processed/CT014_2025-12-16_153200_block_performance.csv", 
                     header = TRUE)#,
df2 <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251202/processed/CT014_2025-12-02_151940_block_performance.csv", 
                     header = TRUE)#,
df3 <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251204/processed/CT014_2025-12-04_123418_block_performance.csv", 
                header = TRUE)#,
df4 <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251211_latentInference/processed/CT014_2025-12-11_134311_block_performance.csv", 
                header = TRUE)#,
df5 <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference/processed/CT014_2025-12-23_163505_block_performance.csv", 
                header = TRUE)#,

df1 <- clean_block_dataframe(df1)
df2 <- clean_block_dataframe(df2)
df3 <- clean_block_dataframe(df3)
df4 <- clean_block_dataframe(df4)
df5 <- clean_block_dataframe(df5)


# Usage
stack_df <- stack_dataframes(df1, df2, df3,df4,df5)#, .id = "source")

flexplot(cur_strategy_slope ~ 1,  data=stack_df)
flexplot(trials_to_correct ~ prev_n_rewarded + cur_strategy_slope | declared_strategy,  data=stack_df)#, method='lm')

mod_full = lm(trials_to_correct ~ prev_n_rewarded * bias_full_flag * cur_strategy_slope, data=stack_df)
visualize(mod_full)


mod_rewards = lm(trials_to_correct ~ prev_n_rewarded + bias_full_flag, data = block_df)
# mod_rew_minimal = lm(trials_to_correct ~ prev_n_rewarded, data = block_df)
# compare.fits(trials_to_correct ~ prev_n_rewarded | bias_full_flag, data=block_df, model1=mod_rewards_full, model2=mod_rewards)
# model.comparison(mod_rewards_full, mod_rewards)
# 
# estimates(mod_rewards_full)
# compare.fits(trials_to_correct ~ prev_n_rewarded | bias_full_flag, data=block_df, model1=mod_rewards_full, model2=mod_rew_minimal)
# model.comparison(mod_rewards_full, mod_rew_minimal)
# 
# mod_correct = lm(trials_to_correct ~ prev_n_correct * bias_full_flag, data = block_df)
# visualize(mod_correct)

### compare the different behavior states/strategies ###########################
mod_fullest = lm(trials_to_correct ~ prev_n_rewarded * inferred_strategy * bias_full_flag, data = block_df)
mod_inferred = lm(trials_to_correct ~ prev_n_rewarded * inferred_strategy, data = block_df)
visualize(mod_fullest)
visualize(mod_inferred)
estimates(mod_inferred)

compare.fits(trials_to_correct ~ prev_n_rewarded | bias_full_flag + inferred_strategy, data=block_df, model1=mod_fullest, model2=mod_inferred)
model.comparison(mod_fullest, mod_inferred)

##########################################

# compare a simple flat fit to a linear model like RL

# apparently I need to put everything into long format

# control = lmerControl(optimizer = "nloptwrap", optCtrl = list(algorithm = "NLOPT_LN_NELDERMEAD"))
control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun=2e5))
mod_blocks = lmer(trials_to_correct ~ prev_n_rewarded + bias_full_flag +
                    (prev_n_rewarded | cur_strategy_slope),#  declared_strategy),
                  data = stack_df, control=control)
mod_blocks_layered = lmer(trials_to_correct ~ prev_n_rewarded + bias_full_flag +
                    (prev_n_rewarded | declared_strategy / cur_strategy_slope), 
                  data = stack_df, control=control)
model.comparison(mod_blocks_layered, mod_blocks)


allFit(mod_blocks)
visualize(mod_blocks)

cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded, object = mod_blocks, adjust="all")
#cluster_adjusted_scatter(trials_to_correct ~ prev_n_rewarded | bias_full_flag, object = mixmod_rew, adjust="all")
# allFit(mixmod_hmm)

# mod_rewards = lmer(trials_to_correct ~ prev_n_rewarded + (prev_n_rewarded | inferred_strategy),
#                    data = block_df, control=control)

visualize(mod_blocks, plot = 'model', formula = trials_to_correct ~ prev_n_rewarded | declared_strategy + cur_strategy_slope)

# model.comparison(mod_blocks_fullest, mod_blocks)
# model.comparison(mod_blocks, mod_rewards)
# estimates(mod_blocks_fullest)


# small_blocks <- block_df %>% select(-block_ix,-block_type,-prev_consecutive_rewards,-prev_consecutive_rewards_memory,-prev_n_correct,-normalized_switches,-confusion_flag,
#                                     -n_correct,-percent_correct,-n_rewarded,-mean_choice_time,
#                                     -median_choice_time,-std_choice_time,-session_ID,-bias_rl,
#                                     -bias_inf,-bias_rl_flag,-bias_inf_flag,-declared_strategy)
# rf_blocks = rfsrc(trials_to_correct ~ ., data=small_blocks)
# print(rf_blocks)
# varfit_blocks = varpro(trials_to_correct ~ ., data=small_blocks)
# imp_blocks = importance(varfit_blocks)
# importance(varfit_blocks, plot.it = TRUE)

