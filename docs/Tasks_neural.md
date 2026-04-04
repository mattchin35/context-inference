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

In this refactoring, I will train the decoder in step on correct-rewarded trials AFTER choice time.
I will also improve the result by running the entire process multiple times so that each accuracy score has a 
distribution of scores from multiple decoders. Then the result will not be bound to the particular random state 
of the original decoder instantiation. I will run 5 decoders per session to start; this can be adjusted as needed.


# Saved items
Data from decodability analysis and from the correct-rewarded decoding performance analysis should be saved as .csv files.
Decodability analysis will be one csv file per session, and these will have to be pulled across sessions for plotting.
Decodability analysis will save the cv_score, p_value, and the score mean and std for each session, trial condition, and time bin.
These values will have to be pulled across sessions to be shown in a single plot.

Correct-rewarded decoding performance analysis will save from each decoder run the test accuracy for each 
trial condition and time bin. The mean and std of the test accuracy across decoders will be computed on the fly
after the csv file is loaded.
These will also be saved as one csv file.

## Specs
Per session csvs:
- state_decodability_analysis.csv: each row represents one trial condition, with columns for 
  - cv_score_before
  - cv_score_after
  - p_value_before
  - p_value_after
  - score_mean_before
  - score_mean_after
  - score_std_before
  - score_std_after
- correct_rewarded_state_decoding_performance.csv: each row represents one decoder, with columns representing the test accuracy for each trial condition and time bin. Columns will be:
  - test_accuracy_before for each condition
  - test_accuracy_after for each condition
  - in the correct_rewarded row, there should be 2 additional columns train_acc and shuffle_p from the decoder training step
The mean and std will be computed from the loaded csv.

Cross-session csvs:
- state_decodability_analysis_cross_session.csv: each row represents one session and one trial condition
  - columns: session, trial_condition, cv_score_before, cv_score_after, p_value_before, p_value_after
- correct_rewarded_state_decoding_performance_cross_session.csv: each row represents one session and one trial condition.
  - columns: session, trial_condition, decoder_run_index, test_accuracy_before, test_accuracy_after

# Plotting
Decodability values will have to be pulled across sessions, and the cv_score and p_value will be plotted. Each
plot will show the before and after choice time bins for each trial condition, plotting all sessions together.

Correct-rewarded decoding performance will also be pulled across sessions, and will be plotted twice:
- once for each session, showing the distribution of test accuracy scores across decoders for each trial condition before 
and after choice time.
- across sessions, showing the mean test accuracy scores across decoders for each trial condition before and after choice time.

This is probably an opportunity for mixed-model style analysis, with a super-plot showing all of the individual decoder 
runs for each session in one faded color per session, with one bolder color for the session mean, 
and then the overall mean in black. I would have (n_sessions * n_decoders) points instead of 
just n_sessions points. 

The previous plotstyle will generally be repeated, with the before/after for each decoder connected by a line. 
The new trial conditions are just a new set of accuracy scores to plot. 

Plots should all go into the "figures" folder for each session, or the cross_session_analysis folder for the mouse's 
cross-session plots.
