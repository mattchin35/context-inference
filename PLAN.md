# Current status

Mouse behavior regressors are available but the code is messy. Model agents can be run, saved, and plotted.
Neural data can be collected into 500 ms bins around choice times for simple decoding of context.

# Current goals
I need to make sure model selection works, show that it works on some simulated models, and use the models to 
analyze some sample mouse behavior sessions. I need to make plots from these analyses to prepare for a presentation.

## Validate the model agents 
1. Implement switching between model agent strategies during model runs
2. Make plots corresponding to each model agent, and make plots of switching strategies within a run

## Validate the GLM-HMM and LM-HMM implementations
1. Finish implementing cross-validation and BIC for both LM-HMM and GLM-HMM 
2. Use model runs with strategy switching to validate the LM-HMM and GLM-HMM implementations,
showing that they can distinguish the behaviors and that the combined HMM-decay-doubt agent can be distinguished from
the simpler HMM and Forgetting Q-learning agents
3. Use the validated FQL, HMM, and HMM-decay-doubt regressors to analyze 3 mouse behavior sessions, extracting the 
regressor weights and showing that they correspond to describable strategies in each session.

## Prepare behavior for a presentation
I will prepare a presentation using three example mouse behavior sessions. In particular, I will use CT014_20251205_latentInference,
CT014_20251216_latentInference, and CT014_20251223_latentInference.

1. Show for each session that I have selected a reasonable number of LM-HMM and GLM-HMM states based on the BIC and
cross-validation results, using the plots of these results to justify my choice of state number for each session.
2. Show that the LM-HMM has found distinguishable block strategies in each session, roughly 
corresponding to inference and reinforcement learning strategies based on the LM-HMM fit weights.
3. Show that the GLM-HMM has found distinguishable and meaningful states that  roughly correspond to my agent strategies 
based on the GLM-HMM fit weights.

## Integrate neural analyses with simple behavior analyses
For the 3 behavior sessions above, I will need to do crude analyses of my behavior corresponding to analyzing the 
500 ms periods directly before and directly after a choice is made. The analysis techniques may not improve for now - 
any significant changes will likely be related to refactoring or integrating the code with pynapple.

TODO: make a plan for neural behavior. Right now I need to finish up the model implementation, validation, and 
analysis of mouse behavior sessions.

## Prepare neural analysis for the presentation
TODO: this plan will be made in tandem with the above neural analysis plan 
