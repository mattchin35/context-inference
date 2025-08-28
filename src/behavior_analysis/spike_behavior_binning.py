import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import readSGLX
from icecream import ic
import time
import pickle as pkl


experiment_folder = Path('/home/matt/Documents/EXPERIMENTS/')
behavior_data_path = experiment_folder / 'raw_behavior_data'
ephys_data_path = experiment_folder / 'raw_ephys_data'
processed_data_path = experiment_folder / 'processed_data'

current_mouse = 'CT010'
current_date_behavior = '2025-08-15'
current_date_ephys = ''.join(current_date_behavior.split('-'))  # remove dashes for ephys folder name
sess_timestamp = '125111'
sess_id = current_mouse + '_' + current_date_behavior
sess_id_full = current_mouse + '_' + current_date_behavior + '_' + sess_timestamp

behavior_session_path = behavior_data_path / sess_id_full
session_info_path = behavior_session_path / '{}_session_info.pkl'.format(sess_id_full)
output_path = processed_data_path / current_mouse / sess_id_full

with open(output_path / '{}.csv'.format(sess_id_full), 'rb') as f:
    event_df = pd.read_csv(f, sep=',')
with open(session_info_path, 'rb') as f:
    session_info = pkl.load(f)
with open(output_path / '{}_trials.pkl'.format(sess_id_full), 'rb') as f:
    trial_df = pkl.load(f)

spike_times = np.load(output_path / '{}_aligned_spike_times.npy'.format(sess_id_full), allow_pickle=True)
spike_clusters = np.load(output_path / '{}_aligned_spike_clusters.npy'.format(sess_id_full), allow_pickle=True)
cluster_dict = np.load(output_path / '{}_cluster_dict.npy'.format(sess_id_full), allow_pickle=True).item()
spike_df = pd.DataFrame({'spike_times': spike_times, 'spike_clusters': spike_clusters})
# clustered_spikes = spike_df.groupby('spike_clusters')
# sorted_spike_df = spike_df.sort_values(['spike_clusters', 'spike_times'])
cluster_ids = np.unique(spike_clusters)
n_clusters = cluster_ids.size

ic(event_df)
# ic(session_info)
ic(trial_df.iloc[0])
ic(spike_times)
ic(spike_clusters)

# lick_df = event_df[(event_df['Event'] == 'left_entry') | (event_df['Event'] == 'right_entry')]

"""
TODO 
- bin spikes for each unit (50 ms) and licks 
- gather spikes and licks around trials 
- plot spikes and licks around trials 
- decode choice from spikes within 50 ms bins 
- decode context from spikes within 50 ms bins 
"""


def bin_spikes_to_trial(trial_start: float, trial_end: float, spike_times: np.ndarray, cluster_ids: np.ndarray,
                        bin_size: float = 0.1, pre_time: float = 2.0, post_time: float = 2.0) -> [np.ndarray, np.ndarray]:
    bin_st = trial_start - pre_time
    bin_end = trial_end + post_time
    n_bins = int((bin_end - bin_st) / bin_size)
    binned_spikes = np.zeros((n_clusters, n_bins))
    bin_edges = np.arange(bin_st, bin_end, bin_size)
    if bin_edges.size < n_bins+1:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_size)

    for i, cluster in enumerate(cluster_ids):
        spikes = spike_times[spike_clusters == cluster]
        try:
            binned_spikes[i], _ = np.histogram(spikes, bins=bin_edges)
        except ValueError as e:
            print(f"Error binning spikes for cluster {cluster}: {e}")
            return

    return binned_spikes, bin_edges


def bin_licks_to_trial(trial_start: float, trial_end: float, event_df: pd.DataFrame,
              pre_time: float = 2.0, post_time: float = 2.0, bin_size: float = 0.1) -> [np.ndarray, np.ndarray]:
    bin_st = trial_start - pre_time
    bin_end = trial_end + post_time
    n_bins = int((bin_end - bin_st) / bin_size)
    bin_edges = np.arange(bin_st, bin_end, bin_size)
    if bin_edges.size < n_bins+1:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_size)
    binned_licks = np.zeros((n_bins, 2))  # last dimension for left and right licks; 0=right, 1=left

    left_licks = event_df[event_df['Event'] == 'left_entry']['Time']
    right_licks = event_df[event_df['Event'] == 'right_entry']['Time']
    binned_licks[:, 0], _ = np.histogram(right_licks, bins=bin_edges)
    binned_licks[:, 1], _ = np.histogram(left_licks, bins=bin_edges)
    return binned_licks, bin_edges


def decode_from_spikes(binned_spikes: np.ndarray, bin_value: np.ndarray) -> [object, float]:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score

    nanmask = np.isnan(bin_value)
    if np.any(nanmask):
        print("Warning: NaN values found in bin_value. Removing corresponding spike bins.")
        binned_spikes = binned_spikes[:, ~nanmask]
        bin_value = bin_value[~nanmask]

    X_train, X_test, y_train, y_test = train_test_split(binned_spikes.T, bin_value, test_size=0.2, random_state=42)

    # classifier = LinearRegression()
    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    np.random.shuffle(y_train)  # Shuffle fit
    shuffle_clf = LogisticRegression(max_iter=10000)
    shuffle_clf.fit(X_train, y_train)
    shuffle_y_pred = shuffle_clf.predict(X_test)
    shuffle_accuracy = accuracy_score(y_test, shuffle_y_pred)
    ic(accuracy, shuffle_accuracy)
    return classifier, accuracy, binned_spikes, bin_value


def plot_trial(trial_start: float, choice_t: float=None, reward_t: float=None,
              spike_times: np.ndarray=None, cluster_ids: np.ndarray=None, lick_times: np.ndarray=None,
              pre_time: float = 2.0, post_time: float = 2.0, bin_size: float=.1) -> None:
    """
    Plot a trial with licks as rasters and spikes as bins.
    """
    f, ax = plt.figure(figsize=(10, 6))
    plt.axvline(trial_start, 0, 1, color='k', label='Trial Start', linewidth=2, alpha=0.5, linestyle='--')
    if choice_t:
        plt.axvline(choice_t, 0, 1, color='b', label='Choice', linewidth=2, alpha=0.5)
    if reward_t:
        plt.axvline(reward_t, 0, 1, color='g', label='Reward', linewidth=2, alpha=0.5)
    if reward_t:
        trial_end = reward_t
    elif choice_t:
        trial_end = choice_t
    else:
        trial_end = trial_start

    if lick_times:
        ax.eventplot(lick_times, linelengths=0.6, linewidths=.2, color='black')

    if spike_times is not None and cluster_ids is not None:
        binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, spike_times, cluster_ids, bin_size, pre_time, post_time)
        for unit in range(binned_spikes.shape[0]):
            plt.plot(bin_edges[:-1] + bin_size / 2, binned_spikes[unit], label='Unit {}'.format(cluster_ids[unit]))

    plt.xlim(trial_start-pre_time, trial_end + post_time)
    # plt.ylim(0.5, len(spikes_around_trials[unit_id]) + 0.5)
    plt.xlabel('Time (s) ')
    plt.ylabel('Trial')
    # plt.title(f'Raster plot for unit {unit_id}')
    plt.title('Trial plot: {}'.format(sess_id_full))
    plt.show()

# lists of dictionaries for each trial's bins
p = output_path / ('{}_spikes_trial_binned.pkl'.format(sess_id_full))
with p.open('rb') as f:
    spikes_trial_binned = pkl.load(f)

p = output_path / ('{}_licks_trial_binned.pkl'.format(sess_id_full))
with p.open('rb') as f:
    licks_trial_binned = pkl.load(f)


# spikes_trial_binned = []
# licks_trial_binned = []
#
# for i, trial in trial_df.iterrows():
#     trial_start = trial['start_time']
#     if not np.isnan(trial['reward_time']):
#         trial_end = trial['reward_time']
#     elif not np.isnan(trial['choice_time']):
#         trial_end = trial['choice_time']
#     else:
#         trial_end = trial_start
#
#     binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, spike_times, cluster_ids,
#                                                    bin_size=0.05, pre_time=2.0, post_time=2.0)
#     bin_states = np.ones(binned_spikes.shape[1]) * trial['state_int']  # 0=right active, 1=left active
#     bin_choices = np.ones(binned_spikes.shape[1]) * trial['action']  # 0=right, 1=left
#     spikes_trial_binned.append(dict(trial_ix=i, binned_spikes=binned_spikes, bin_edges=bin_edges,
#                                     bin_states=bin_states, bin_choices=bin_choices))
#
#     binned_licks, lick_bin_edges = bin_licks_to_trial(trial_start, trial_end, event_df,
#                                                       pre_time=2.0, post_time=2.0, bin_size=0.1)
#     bin_states = np.ones(binned_licks.shape[1]) * trial['state_int']
#     bin_choices = np.ones(binned_licks.shape[1]) * trial['action']
#     licks_trial_binned.append(dict(trial_ix=i, binned_licks=binned_licks, bin_edges=lick_bin_edges,
#                                    bin_states=bin_states, bin_choices=bin_choices))
#
# p = output_path / ('{}_spikes_trial_binned.pkl'.format(sess_id_full))
# with p.open('wb') as f:
#     pkl.dump(spikes_trial_binned, f)
#
# p = output_path / ('{}_licks_trial_binned.pkl'.format(sess_id_full))
# with p.open('wb') as f:
#     pkl.dump(licks_trial_binned, f)


# logistic regression
# all_spike_bins = np.concatenate([trial['binned_spikes'] for trial in spikes_trial_binned], axis=1)
# all_state_bins = np.concatenate([trial['bin_states'] for trial in spikes_trial_binned])
# all_choice_bins = np.concatenate([trial['bin_choices'] for trial in spikes_trial_binned])
# classifier, accuracy, _, _ = decode_from_spikes(all_spike_bins, all_choice_bins)
# print(f"Choice decoding accuracy: {accuracy:.2f}")
# classifier, accuracy, _, _ = decode_from_spikes(all_spike_bins, all_state_bins)
# print(f"State decoding accuracy: {accuracy:.2f}")

training_ix = (trial_df['reward'] == 1)  & (trial_df['correct'] == 1)
incorrect_ix = (~np.isnan(trial_df['action'])) & (trial_df['correct'] == 0)
withheld_ix = (trial_df['correct'] == 1) & (trial_df['reward'] == 0)

# bins for the max probability period after choice
train_spike_bins, train_state_bins, train_choice_bins = [], [], []
incorrect_spike_bins, incorrect_state_bins, incorrect_choice_bins = [], [], []
withheld_spike_bins, withheld_state_bins, withheld_choice_bins = [], [], []

for ix in np.where(training_ix)[0]:
    # trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['choice_time'] + 0) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['choice_time'] + 1.5))
    trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['reward_time'] + 0) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['reward_time'] + 2))
    _spike_bins = spikes_trial_binned[ix]['binned_spikes'][:, trial_bin_ix[:-1]]
    _state_bins = spikes_trial_binned[ix]['bin_states'][trial_bin_ix[:-1]]
    _choice_bins = spikes_trial_binned[ix]['bin_choices'][trial_bin_ix[:-1]]

    train_spike_bins.append(_spike_bins)
    train_state_bins.append(_state_bins)
    train_choice_bins.append(_choice_bins)

    # plot_trial(trial_start=trial_df.iloc[ix]['start_time'],
    #            choice_t=trial_df.iloc[ix]['choice_time'],
    #            reward_t=trial_df.iloc[ix]['reward_time'],
    #            spike_times=spike_times,
    #            cluster_ids=cluster_ids,
    #            lick_times=event_df[(event_df['Event'] == 'left_entry') | (event_df['Event'] == 'right_entry')]['Time'].values,
    #            pre_time=2.0, post_time=2.0, bin_size=0.05)

for ix in np.where(incorrect_ix)[0]:
    trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['choice_time'] + 0) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['choice_time'] + 1.5))
    _spike_bins = spikes_trial_binned[ix]['binned_spikes'][:, trial_bin_ix[:-1]]
    _state_bins = spikes_trial_binned[ix]['bin_states'][trial_bin_ix[:-1]]
    _choice_bins = spikes_trial_binned[ix]['bin_choices'][trial_bin_ix[:-1]]

    incorrect_spike_bins.append(_spike_bins)
    incorrect_state_bins.append(_state_bins)
    incorrect_choice_bins.append(_choice_bins)

for ix in np.where(withheld_ix)[0]:
    trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['choice_time'] + 0) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['choice_time'] + 1.5))
    _spike_bins = spikes_trial_binned[ix]['binned_spikes'][:, trial_bin_ix[:-1]]
    _state_bins = spikes_trial_binned[ix]['bin_states'][trial_bin_ix[:-1]]
    _choice_bins = spikes_trial_binned[ix]['bin_choices'][trial_bin_ix[:-1]]

    withheld_spike_bins.append(_spike_bins)
    withheld_state_bins.append(_state_bins)
    withheld_choice_bins.append(_choice_bins)

train_spike_bins = np.concatenate(train_spike_bins, axis=1)
train_state_bins = np.concatenate(train_state_bins)
train_choice_bins = np.concatenate(train_choice_bins)

choice_classifier, accuracy, _, _ = decode_from_spikes(train_spike_bins, train_choice_bins)
print(f"Choice decoding accuracy: {accuracy:.2f}")
state_classifier, accuracy, _, _ = decode_from_spikes(train_spike_bins, train_state_bins)
print(f"State decoding accuracy: {accuracy:.2f}")

