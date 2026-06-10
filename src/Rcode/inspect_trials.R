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

#source("preprocessing/block_preprocessing.R")
source("preprocessing/trial_preprocessing.R")

df1 <- read.csv("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT016/CT016_20260511_latent_inference/processed/CT016_2026-05-11_124709_augmented_trials.csv",
                header = TRUE)#,


