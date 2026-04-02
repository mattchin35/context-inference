# Trial conditions for neural data analysis

Only trials with trial_df['give_reward'] = 0 should be selected, for every condition.
give_reward refers to experimenter-given rewards, indicating that the trial is invalid for analysis.

Base trial conditions:
- training/correct-rewarded: correct choices with rewards given
- omission trials (correct choices with rewards withheld)
- incorrect choices
- unrewarded-switch trials (unrewarded trials, whether correct or incorrect, that lead to an action switch)
- unrewarded-stay trials (unrewarded trials, whether correct or incorrect, that do not lead to an action switch)

Combinations of trial conditions:
- omission-switch trials: from the unrewarded-switch trials, select only those that are omission trials (correct choices with rewards withheld)
- omission-stay trials: from the unrewarded-stay trials, select only those that are omission trials (correct choices with rewards withheld)
- incorrect-switch trials: from the unrewarded-switch trials, select only those that are incorrect choices
- incorrect-stay trials: from the unrewarded-stay trials, select only those that are incorrect choices
These should be easily doable as AND combinations of the base conditions.

## Switch and stay trial definitions
For trials prior to the last trial, if the next trial's choice is different from the current trial's choice, 
then the current trial is a switch trial. If the next trial's choice is the same as the current trial's choice, 
then the current trial is a stay trial.

# Classifier bins
Options should remain available to bin spikes around the trial start and choice time.
Reward delivery should be instantaneous after choice, so choice time is effectively reward time.

# Decodability analysis
Cross-validated decoding performance should be assessed for each trial condition, and for each time bin (before and after choice time).
This is a simple check of whether the correct state can be decoded at all from the neural data, not a check for the 
currently encoded state. Previously implemented into cv_decode_only, using sklearn's permutation_test_score on a 
logistic regression classifier. I might as well keep the permutation scores as well, displaying them as 
a simple mean(SCORES) +/- std(SCORES). This should also be renamed to something a little more descriptive, 
like cv_decodeability_score.

# Correct-rewarded decoding performance
Previously, I tested decoder performance as follows:
1. A decoder is trained on the correct-rewarded trials before choice time, and tested against a set of held-out trials. 
It is then compared against a distribution of decoders trained on the same data but with the trial labels shuffled to 
create a null distribution. That decoder is passed onto downstream analyses. (shuffle_decode_only)
2. The decoder from step 1 is then tested on the other desired trial conditions in the before and after choice time bins,
only using sklearn's accuracy_score.

This way of doing things is simple, but it could be mildly improved by running the entire process multiple times 
so that each accuracy score has a distribution of scores from multiple decoders. Then the result would not be 
so bound to the particular random state of the original decoder instantiation. 

# Plotting
Many of the previous plots can be repeated. The new trial conditions are just a new set of trials with accuracy scores
to check and plot. The main change will be for repeating the decoder runs: the new plots would show each session as 
a distribution of scores instead of as a single point, so that I would have (n_sessions * n_decoders) points instead of 
just n_sessions points. It might be nice to show each session's distribution of scores in a different color,
with the overall mean across sessions in black.
