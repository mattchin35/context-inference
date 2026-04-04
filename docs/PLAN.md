# Current status

Mouse behavior regressors are available but the code is messy. Model agents can be run, saved, and plotted.
Neural data can be collected into 500 ms bins around choice times for simple decoding of context.

# Current goals
I need to make sure model selection works, show that it works on some simulated models, and use the models to 
analyze some sample mouse behavior sessions. I need to make plots from these analyses to prepare for a presentation.

## Validate the model agents 
1. Implement switching between model agent strategies during model runs
   - done for agents with a hard switch (mode 1), decaying memory for unused agents (mode 2), and parallel execution (mode 3)
2. Make plots corresponding to each model agent, and make plots of switching strategies within a run
3. Implement switching mode 3 for now as parallel model execution:
   - all selected models run in parallel on the same trial stream
   - the active model for that trial supplies the recorded value, action probabilities, and action output
   - all models update from the executed action and observed reward, even if they would not have sampled that action 
themselves
   - only the active model's outputs are recorded in the main run dataframe
4. Revisit a true shared-state mode 3 later, after the shared latent state is specified more clearly

## Validate the GLM-HMM and LM-HMM implementations
1. Finish implementing single-session model selection with AIC/BIC for both LM-HMM and GLM-HMM 
2. Use model runs with strategy switching to validate the LM-HMM and GLM-HMM implementations, showing that they can 
distinguish the behaviors and that the combined HMM-decay-doubt agent can be distinguished from the simpler HMM and 
Forgetting Q-learning agents.
   - Note: upon attempting to validate the HMMs, it seems that distinguishing them is not so easy. I will skip the 
computational validation step for now, especially as we may not even stick with this modeling strategy long-term.
3. Use the validated FQL, HMM, and HMM-decay-doubt regressors to analyze 3 mouse behavior sessions, extracting the 
regressor weights and showing that they correspond to describable strategies in each session.

## Prepare behavior for a presentation
I will prepare a presentation using three example mouse behavior sessions. In particular, I will use 
CT014_20251205_latentInference, CT014_20251216_latentInference, and CT014_20251223_latentInference.

1. Show for each session that I have selected a reasonable number of LM-HMM and GLM-HMM states based on the AIC and BIC.
2. Show that using more states results in new states that do not learn anything (i.e. have very low weights for all regressors)
3. Show that the LM-HMM has found distinguishable block strategies in each session, roughly 
corresponding to inference and reinforcement learning strategies based on the LM-HMM fit weights.
4. Show that the GLM-HMM has found distinguishable and meaningful states, particularly showing that later in training
the GLM-HMM starts to weight the doubt regressor more.

It might also be good to show that spread of trials to switch alongside the LM-HMM results, instead of just 
the correlation weights, so the scale is clear.

## Integrate neural analyses with simple behavior analyses
For the 3 behavior sessions above, I will need to do crude analyses of my behavior corresponding to analyzing the 
500 ms periods directly before and directly after a choice is made. The analysis techniques may not improve for now - 
any significant changes will likely be related to refactoring or integrating the code with pynapple.

1. Neural data must have UTC timestamps for alignment with behavior 
2. HPC and V1 spikes must be chosen by electrode sites, and binned into 500 ms bins around choice times.
3. Pre-existing analyses should be replicated. That is, 
   - Train decoder on correct choices with rewards given in the 500 ms before choice time; 
test on held-out correct choices in the training condition,
correct choices with rewards given in the 500 ms after choice time, 
incorrect choices in the 500 ms before AND after choice time,
and correct choices with rewards withheld (omission trials) in the 500 ms before AND after choice time.
4. For my only new analysis, I should incorporate lab meeting feedback: separate unrewarded trials into 
those that lead to a behavior switch and those that don’t
   - Can you see a sufficient change in belief to switch vs an insufficient change?
   - So categorize unrewarded trials into those that are followed by a switch in behavior and those that are not
   - Assess the change in HPC decoding of context in the 500 ms before and after choice time each category

## Prepare neural analysis for the presentation
1. For each of the 3 sessions, make the plots showing the decoding performance before and after choice time for the different trial types.
Make sure each session's data are seen grayed with an overall trend in black
2. For the new analysis, show a plot for each condition (next trial switch vs no switch) as in 1
3. Repeat the above 2 for putative V1 units
4. Ideally repeat 1+2 for mPFC units; we'll see if this happens

