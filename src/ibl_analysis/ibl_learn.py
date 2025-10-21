import numpy as np
import pandas as pd
from pathlib import Path
import scipy as sp
import sklearn
import matplotlib.pyplot as plt
from icecream import ic
import pickle as pkl
from one.api import ONE
import brainbox.behavior.wheel as wh
import seaborn as sns
from brainbox.io.one import load_wheel_reaction_times
from ibllib.io.extractors.ephys_fpga import extract_wheel_moves
from ibllib.io.extractors.training_wheel import extract_first_movement_times
import one.alf.io as alfio
from brainbox.behavior.training import compute_performance
from brainbox.behavior.training import plot_psychometric
from brainbox.task.trials import find_trial_ids
from brainbox.io.one import load_iti
from ibllib.pipes.dynamic_pipeline import get_trials_tasks
from ibllib.qc import task_metrics

"""
Tutorials at
https://docs.internationalbrainlab.org/notebooks_external/loading_trials_data.html
https://docs.internationalbrainlab.org/notebooks_external/docs_wheel_moves.html
"""

one = ONE()
sns.set_style('whitegrid')

device_info = ('The wheel diameter is {} cm and the number of ticks is {} per revolution'.format(wh.WHEEL_DIAMETER, wh.ENC_RES))
print(device_info)

experiment_folder = Path('/home/matt/Downloads/ONE/openalyx.internationalbrainlab.org')
lab = 'angelakilab'
subject = 'NYU-11'
date = '2020-02-18'
session = '001'
base_path = experiment_folder / lab / 'Subjects' / subject / date / session / 'alf'
assert base_path.exists(), "Path does not exist: {}".format(base_path)

eid = Path('/home/matt/Downloads/ONE/openalyx.internationalbrainlab.org/angelakilab/Subjects/NYU-11/2020-02-18/001')

# qc = task_metrics.TaskQC(eid)
# outcome, results = qc.run(update=False)
# print(f'QC_status: {outcome}')
# print(f'Individual QC values:')
# ic(results)

trials = one.load_object(eid, 'trials')
trials['iti'] = load_iti(trials)
print(trials.to_df().iloc[:5, -5:])

fig, ax = plot_psychometric(trials)

# compute performance
performance, contrasts, n_contrasts = compute_performance(trials)
# compute performance expressed as probability of choosing right
performance, contrasts, n_contrasts = compute_performance(trials, prob_right=True)
# compute performance during 0.8 biased block
performance, contrasts, n_contrasts = compute_performance(trials, block=0.8)

# find index for stim right trials ordered by trial number
trial_id, _ = find_trial_ids(trials, side='right', choice='all', order='trial num')
# find index for correct, stim left, 100% contrast trials ordered by reaction time
trial_id, _ = find_trial_ids(trials, side='left', choice='correct', contrast=[1], order='reaction time')
# find index for correct trials ordered by trial number sorted by stimulus side
trial_id, _ = find_trial_ids(trials, side='left', choice='correct', order='reaction time', sort='side')


wheel_posn = np.load(base_path / '_ibl_wheel.position.npy')
wheel_timestamps = np.load(base_path / '_ibl_wheel.timestamps.npy')
wheelmove_intervals = np.load(base_path / '_ibl_wheelMoves.intervals.npy')
wheelmove_amplitudes = np.load(base_path / '_ibl_wheelMoves.peakAmplitude.npy')

pos, t = wh.interpolate_position(wheel_timestamps, wheel_posn)
ic(pos, t)
ic(t[1]-t[0])  # should be 0.001 sec for 1000 Hz

# Convert the pos threshold defaults from samples to correct unit
thresholds_cm = wh.samples_to_cm(np.array([8, 1.5]), resolution=wh.ENC_RES)
thresholds = wh.cm_to_rad(thresholds_cm)

### RESPONSE TIMES ###
wheel_sample_freq = 1000
vel, acc = wh.velocity_filtered(pos, wheel_sample_freq)
rt = load_wheel_reaction_times(eid)
ts = wh.get_movement_onset(wheelmove_intervals, trials.response_times)
# The time from final movement onset to response threshold
movement_response_times = trials.response_times - ts

idx = 15 # trial index
mask = np.logical_and(trials['goCue_times'][idx] < t, t < trials['feedback_times'][idx])
f, ax = plt.subplots()
plt.plot(t[mask], pos[mask])
plt.axvline(x=ts[idx], label='movement onset', linestyle='--');
plt.axvline(x=trials.response_times[idx], label='response threshold', linestyle=':');
plt.legend()

### REACTION TIMES VS CONTRAST ###
# Replace nans with zeros
trials.contrastRight[np.isnan(trials.contrastRight)] = 0
trials.contrastLeft[np.isnan(trials.contrastLeft)] = 0
contrast = trials.contrastRight - trials.contrastLeft
mean_rt = [np.nanmean(rt[contrast == c]) for c in set(contrast)]

# RT may be nan if there were no detected movements, or if the goCue or stimOn times were nan
xdata = np.unique(contrast)
f = plt.figure(figsize=(4, 3))  # Some sort of strange behaviour in this cell's output
plt.plot(xdata, mean_rt)
plt.xlabel('contrast')
plt.ylabel('mean rt / s')
plt.ylim(bottom=0)
plt.title('Mean reaction time vs contrast')

wheel_moves = one.load_object(eid, 'wheelMoves', collection='alf')
firstMove_times, is_final_movement, ids = extract_first_movement_times(wheel_moves, trials)

# Plot the interpolated data points
sec = 5  # Number of seconds to plot
f, ax = plt.subplots()
mask = t < (t[0] + sec)
plt.plot(t[mask], pos[mask], '.', markeredgecolor='lightgrey', markersize=1)
# Plot the original data
mask = wheel_timestamps < (wheel_timestamps[0] + sec)
plt.plot(wheel_timestamps[mask], wheel_posn[mask], 'r+', markersize=6)
plt.xlabel('time / sec')
plt.ylabel('position / rad')
plt.title('Wheel position')
plt.box(on=None)

threshold_deg = 35 # visual degrees
gain = 4  # deg / mm
threshold_rad = wh.cm_to_rad(1e-1) * (threshold_deg / gain)  # rad
print('The wheel must be turned ~%.1f rad to move the stimulus to threshold' % threshold_rad)

n = 5  # trial number
start, end = trials['intervals'][n,]  # trial intervals
intervals = wheel_moves['intervals']  # movement onsets and offsets

# Find direction changes for a given trial
mask = np.logical_and(intervals[:,0] > start, intervals[:,0] < end)
change_times, idx, = wh.direction_changes(t, vel, intervals[mask])

plt.figure()
mask = np.logical_and(t > start, t < end)  # trial intervals mask
plt.plot(t[mask], pos[mask], 'k')  # plot wheel trace for trial
for i in np.concatenate(change_times):
    plt.axvline(x=i, color='k', linestyle=':')
plt.title('Trial #%s direction changes' % n)
plt.xlabel('time / s')
plt.ylabel('position / rad')

mask = t < (t[0] + sec)
onsets, offsets, peak_amp, peak_vel_times = wh.movements(t[mask], pos[mask], pos_thresh=thresholds[0], pos_thresh_onset=thresholds[0], make_plots=True)  # this makes its own new figure

# plt.show()


lick_times = np.load(base_path / 'licks.times.npy')
gocue_times = np.load(base_path / '_ibl_trials.goCueTrigger_times.npy')
stimoff_times = np.load(base_path / '_ibl_trials.stimOff_times.npy')

probe00_path = base_path / 'probe00'
pykilosort_date = '#2024-05-06#'
electrode_sites = np.load(probe00_path / 'electrodeSites.brainLocationIds_ccf_2017.npy')
electrode_localcoords = np.load(probe00_path / 'electrodeSites.localCoordinates.npy')
electrode_mlapdv = np.load(probe00_path / 'electrodeSites.mlapdv.npy')

spike_times = np.load(probe00_path / 'pykilosort' / pykilosort_date / 'spikes.times.npy')
spike_clusters = np.load(probe00_path / 'pykilosort' / pykilosort_date / 'spikes.clusters.npy')
cluster_channels = np.load(probe00_path / 'pykilosort' / pykilosort_date / 'clusters.channels.npy')

ic(spike_times.shape, spike_times.dtype)
ic(spike_clusters.shape, spike_clusters.dtype)
ic(np.unique(spike_clusters))
ic(cluster_channels, cluster_channels.shape)

# processed_data_path = experiment_folder / 'processed_data'

# current_mouse = 'CT010'
# current_date_behavior = '2025-08-21'
# current_date_ephys = ''.join(current_date_behavior.split('-'))  # remove dashes for ephys folder name
# sess_timestamp = '123406'
# sess_id = current_mouse + '_' + current_date_behavior
# sess_id_full = current_mouse + '_' + current_date_behavior + '_' + sess_timestamp
# recording_path = ephys_data_path.joinpath('{}_{}_catgt/catgt_run0_g0/run0_g0_imec0'.format(current_mouse, current_date_ephys))
# recording_path = ephys_data_path.joinpath('{}_{}_catgt/catgt_run0_g0'.format(current_mouse, current_date_ephys))
# sorter_output = recording_path / 'run0_g0_imec0/Kilosort2.5_2025-08-29_145109'

# behavior_session_path = behavior_data_path / sess_id_full
# session_info_path = behavior_session_path / '{}_session_info.pkl'.format(sess_id_full)
# output_path = processed_data_path / current_mouse / sess_id_full

