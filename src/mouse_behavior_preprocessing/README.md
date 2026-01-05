This folder should contain all files for preprocessing (not analyzing!) raw behavior from a mouse session.

The output for each session should be

# ANALYSIS PREP
- A logging file with superfluous elements removed (done)
- IRIG-aligned mouse speed (mouse speed readings are aligned to Unix time - not yet available as interpolation function)
- interpolated mouse speed based on mouse speed alignment (needed for pynapple/neural analysis integration, for speed and isrunning)
  - probably will have to be able to compute this on demand - given a point or interval of time, what is the speed then
- A nice pandas dataframe with each trial's action, correctness, reward outcome, time, prior choice, decision variables, prior-rewarded, pupil size, treadmill speed, inter-choice interval, time from LED light-onset
- pynapple datasets for integration with neural data (or general use, if there are other useful features): licks or lick rates, LED times, trials+choices, speed, reward times

I think main.py, fileIO.py, and session_overview.py are my starting points from the behavior_analysis folder 

In a different file (the behavior analysis folder), clean up the following:
# SESSION OVERVIEW
- behavior rasters
- choice rasters
- performance in blocks
- across days, the trials-to-switch regression

Later this will be integrated with spiking and LFPs.
