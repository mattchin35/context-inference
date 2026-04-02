import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from pathlib import Path
import src.external_tools.readSGLX as readSGLX
import time
from icecream import ic
import pickle as pkl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score, StratifiedKFold, StratifiedShuffleSplit


experiment_folder = Path('/home/matt/Documents/EXPERIMENTS/')
# experiment_folder = Path('C:/Users/mattc/EinsteinMed Dropbox/Matthew Chin/phd_data/remotework/EXPERIMENTS/')
behavior_data_path = experiment_folder / 'raw_behavior_data'
ephys_data_path = experiment_folder / 'processed_ephys_data'
processed_data_path = experiment_folder / 'processed_data'

current_mouse = 'CT010'
current_date_behavior = '2025-08-18'
current_date_ephys = ''.join(current_date_behavior.split('-'))  # remove dashes for ephys folder name
sess_timestamp = '124026'
sess_id = current_mouse + '_' + current_date_behavior
sess_id_full = current_mouse + '_' + current_date_behavior + '_' + sess_timestamp
# recording_path = ephys_data_path.joinpath('{}_{}_catgt/catgt_run0_g0/run0_g0_imec0'.format(current_mouse, current_date_ephys))
recording_path = ephys_data_path.joinpath('{}_{}_catgt/catgt_run0_g0'.format(current_mouse, current_date_ephys))
sorter_output = recording_path / 'run0_g0_imec0' / 'Kilosort2.5_2025-08-29_173906'

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
# cluster_dict = np.load(output_path / '{}_cluster_dict.npy'.format(sess_id_full), allow_pickle=True).item()
spike_df = pd.DataFrame({'spike_times': spike_times, 'spike_clusters': spike_clusters})
cluster_info = pd.read_csv(sorter_output / 'cluster_info.tsv', sep='\t')
# clustered_spikes = spike_df.groupby('spike_clusters')
# sorted_spike_df = spike_df.sort_values(['spike_clusters', 'spike_times'])
cluster_ids = np.unique(spike_clusters)
n_clusters = cluster_ids.size

# spike_channels = np.array([cluster_info['ch'][spike_clusters[ix]] for ix in range(spike_times.size)])
df=cluster_info.set_index('cluster_id')
spike_channels = np.array([df['ch'][cl] for cl in spike_clusters])

channel_positions = sorter_output / 'channel_positions.npy'
channel_positions = np.load(channel_positions, allow_pickle=True)  # ks2.5
geometric_sort_imec0 = np.argsort(channel_positions[:,1])
sorted_coords_imec0 = channel_positions[geometric_sort_imec0]

ch_hpc = np.arange(192, 240)  # channels 192-239 are HPC
ch_md = geometric_sort_imec0[:50]
ch_v1 = geometric_sort_imec0[-50:]

# gather spikes by channel group
hpc_spikes = np.zeros_like(spike_times, dtype=bool)
md_spikes = np.zeros_like(spike_times, dtype=bool)
v1_spikes = np.zeros_like(spike_times, dtype=bool)
for i, ch in enumerate(ch_hpc):
    hpc_spikes = hpc_spikes | (spike_channels == ch)
for ch in ch_md:
    md_spikes = md_spikes | (spike_channels == ch)
for ch in ch_v1:
    v1_spikes = v1_spikes | (spike_channels == ch)

hpc_spike_times = spike_times[hpc_spikes]
hpc_spike_clusters = spike_clusters[hpc_spikes]
hpc_cluster_ids = np.unique(hpc_spike_clusters)
md_spike_times = spike_times[md_spikes]
md_spike_clusters = spike_clusters[md_spikes]
md_cluster_ids = np.unique(md_spike_clusters)
v1_spike_times = spike_times[v1_spikes]
v1_spike_clusters = spike_clusters[v1_spikes]
v1_cluster_ids = np.unique(v1_spike_clusters)

# lick_df = event_df[(event_df['Event'] == 'left_entry') | (event_df['Event'] == 'right_entry')]

def bin_spikes_to_trial(trial_start: float, trial_end: float, spike_times: np.ndarray, spike_clusters: np.ndarray, cluster_ids: np.ndarray,
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


def decode_from_spikes(binned_spikes: np.ndarray, bin_value: np.ndarray, label='') -> [object, float]:
    nanmask = np.isnan(bin_value)
    if np.any(nanmask):
        print("Warning: NaN values found in bin_value. Removing corresponding spike bins.")
        binned_spikes = binned_spikes[:, ~nanmask]
        bin_value = bin_value[~nanmask]

    X_train, X_test, y_train, y_test = train_test_split(binned_spikes.T, bin_value, test_size=0.2, random_state=42)

    # clf = LogisticRegression(max_iter=10000)
    clf = LogisticRegression(
        solver="saga",
        penalty="l1",
        max_iter=10000,
        random_state=42,
    )
    # scores = cross_val_score(clf, X_train, y_train, cv=5)

    score, permutation_scores, pvalue = permutation_test_score(clf, binned_spikes.T, bin_value, scoring="accuracy", cv=5, n_permutations=100)
    ic(score, pvalue)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    ic(train_accuracy, test_accuracy, r2)  # , np.mean(scores), np.std(scores))

    n_shuffles = 1000
    shuffle_accuracies = []
    shuffle_r2 = []
    for i in range(n_shuffles):
        np.random.shuffle(y_train)
        shuffle_clf = LogisticRegression(max_iter=10000)
        shuffle_clf.fit(X_train, y_train)
        shuffle_y_pred = shuffle_clf.predict(X_test)
        shuffle_accuracies.append(accuracy_score(y_test, shuffle_y_pred))
        shuffle_r2.append(sklearn.metrics.r2_score(y_test, shuffle_y_pred))
        # ic(accuracy, shuffle_accuracy)
    shuffle_pval = np.sum(np.array(shuffle_accuracies) >= test_accuracy) / n_shuffles
    # shuffle_vals = dict(accuracy_mean=np.mean(shuffle_accuracies), accuracy_std=np.std(shuffle_accuracies),)
    ic(shuffle_pval, np.mean(shuffle_accuracies), np.std(shuffle_accuracies))
    return (clf, dict(train_accuracy=train_accuracy, test_accuracy=test_accuracy, r2=r2, cv_score=score, cv_pvalue=pvalue,
                      shuffle_pvalue=shuffle_pval, shuffle_acc=np.mean(shuffle_accuracies), shuffle_acc_std=np.std(shuffle_accuracies), label=''),
            binned_spikes, bin_value) #, shuffle_vals

def shuffle_decode_only(binned_spikes: np.ndarray, bin_value: np.ndarray, label='') -> [object, float]:
    nanmask = np.isnan(bin_value)
    if np.any(nanmask):
        print("Warning: NaN values found in bin_value. Removing corresponding spike bins.")
        binned_spikes = binned_spikes[:, ~nanmask]
        bin_value = bin_value[~nanmask]

    X_train, X_test, y_train, y_test = train_test_split(binned_spikes.T, bin_value, test_size=0.2, random_state=42)

    # clf = LogisticRegression(max_iter=10000)
    clf = LogisticRegression(
        solver="saga",
        penalty="l1",
        max_iter=10000,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    ic(train_accuracy, test_accuracy, r2)  # , np.mean(scores), np.std(scores))

    n_shuffles = 1000
    shuffle_accuracies = []
    shuffle_r2 = []
    for i in range(n_shuffles):
        np.random.shuffle(y_train)
        shuffle_clf = LogisticRegression(max_iter=10000)
        shuffle_clf.fit(X_train, y_train)
        shuffle_y_pred = shuffle_clf.predict(X_test)
        shuffle_accuracies.append(accuracy_score(y_test, shuffle_y_pred))
        shuffle_r2.append(sklearn.metrics.r2_score(y_test, shuffle_y_pred))
        # ic(accuracy, shuffle_accuracy)

    shuffle_pval = np.sum(np.array(shuffle_accuracies) >= test_accuracy) / n_shuffles
    plt.hist(shuffle_accuracies, bins=10)
    plt.axvline(test_accuracy, color='k', linestyle='dashed', linewidth=2)
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(sess_id_full)
    plt.text(0.8, 0.8, 'p={:.3f}'.format(shuffle_pval), transform=ax.transAxes, fontsize=12)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    plt.figsize=(4, 3)
    plt.savefig(output_path / 'shuffle_decode_{}_{}.png'.format(sess_id_full, label))
    # plt.show()
    ic(shuffle_pval, np.mean(shuffle_accuracies), np.std(shuffle_accuracies))
    return (clf, dict(train_accuracy=train_accuracy, test_accuracy=test_accuracy, r2=r2,
                      shuffle_pvalue=shuffle_pval, shuffle_acc=np.mean(shuffle_accuracies),
                      shuffle_acc_std=np.std(shuffle_accuracies), label=''))


def cv_decode_only(binned_spikes: np.ndarray, bin_value: np.ndarray, label='') -> dict:
    nanmask = np.isnan(bin_value)
    if np.any(nanmask):
        print("Warning: NaN values found in bin_value. Removing corresponding spike bins.")
        binned_spikes = binned_spikes[:, ~nanmask]
        bin_value = bin_value[~nanmask]

    # clf = LogisticRegression(max_iter=10000)
    clf = LogisticRegression(
        solver="saga",
        penalty="l1",
        max_iter=10000,
        random_state=42,
    )

    # scores = cross_val_score(clf, X_train, y_train, cv=5)
    score, permutation_scores, pvalue = permutation_test_score(clf, binned_spikes.T, bin_value, scoring="accuracy", cv=5, n_permutations=100)
    ic(score, pvalue)
    return dict(cv_score=score, cv_pvalue=pvalue, label=label)


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


def bin_spikes(spike_times, spike_clusters, cluster_ids, bin_size=0.5, pre_time=2.0, post_time=2.0):
    spikes_trial_binned = []
    licks_trial_binned = []
    for i, trial in trial_df.iterrows():
        trial_start = trial['start_time']
        if not np.isnan(trial['reward_time']):
            trial_end = trial['reward_time']
        elif not np.isnan(trial['choice_time']):
            trial_end = trial['choice_time']
        else:
            trial_end = trial_start

        binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, spike_times, spike_clusters, cluster_ids,
                                                       bin_size=bin_size, pre_time=pre_time, post_time=post_time)
        bin_states = np.ones(binned_spikes.shape[1]) * trial['state_int']  # 0=right active, 1=left active
        bin_choices = np.ones(binned_spikes.shape[1]) * trial['action']  # 0=right, 1=left
        spikes_trial_binned.append(dict(trial_ix=i, binned_spikes=binned_spikes, bin_edges=bin_edges,
                                        bin_states=bin_states, bin_choices=bin_choices))

        # binned_licks, lick_bin_edges = bin_licks_to_trial(trial_start, trial_end, event_df,
        #                                                   pre_time=2.0, post_time=2.0, bin_size=0.1)
        # bin_states = np.ones(binned_licks.shape[1]) * trial['state_int']
        # bin_choices = np.ones(binned_licks.shape[1]) * trial['action']
        # licks_trial_binned.append(dict(trial_ix=i, binned_licks=binned_licks, bin_edges=lick_bin_edges,
        #                                bin_states=bin_states, bin_choices=bin_choices))

    return spikes_trial_binned


def make_classifier_bins(spikes_trial_binned, trial_df, trial_ix, event='choice_time', bounds=(-0.5, 0.0)):
    assert event in ['start_time', 'choice_time', 'reward_time'], "Event must be 'start_time', 'choice_time', or 'reward_time'"

    spike_bins, state_bins, choice_bins = [], [], []
    for ix in np.where(trial_ix)[0]:
        trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix][event] + bounds[0]) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix][event] + bounds[1]))
        # trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['start_time'] - .5) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['start_time'] + 0))
        # trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['choice_time'] - .5) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['choice_time'] + 0))
        # trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['reward_time'] + 0) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['reward_time'] + .5))
        _spike_bins = spikes_trial_binned[ix]['binned_spikes'][:, trial_bin_ix[:-1]]  # exclude last bin edge which is the end of the last bin
        _state_bins = spikes_trial_binned[ix]['bin_states'][trial_bin_ix[:-1]]
        _choice_bins = spikes_trial_binned[ix]['bin_choices'][trial_bin_ix[:-1]]

        spike_bins.append(_spike_bins)
        state_bins.append(_state_bins)
        choice_bins.append(_choice_bins)

    spike_bins = np.concatenate(spike_bins, axis=1)
    state_bins = np.concatenate(state_bins)
    choice_bins = np.concatenate(choice_bins)
    return spike_bins, state_bins, choice_bins


def decode_with_cv(main_binned_spikes: np.ndarray, main_bin_value: np.ndarray, prediction: ((np.ndarray, np.ndarray, str))=(), label='') -> [object, float]:
    nanmask = np.isnan(main_bin_value)
    if np.any(nanmask):
        print("Warning: NaN values found in bin_value. Removing corresponding spike bins.")
        binned_spikes = main_binned_spikes[:, ~nanmask]
        main_bin_value = main_bin_value[~nanmask]

    skf = StratifiedKFold(n_splits=5)#, shuffle=True, random_state=42)
    if prediction:
        pred_dict = {label: [] for _, _, label in prediction}

    train_acc, test_acc, r2s = [], [], []
    for i, (ix_train, ix_test) in enumerate(skf.split(main_binned_spikes.T, main_bin_value)):
        xtrain = main_binned_spikes.T[ix_train]
        xtest = main_binned_spikes.T[ix_test]
        ytrain = main_bin_value[ix_train]
        ytest = main_bin_value[ix_test]

        clf = LogisticRegression(
            solver="saga",
            penalty="l1",
            max_iter=10000,
            random_state=42,
        )

        clf.fit(xtrain, ytrain)
        y_pred = clf.predict(xtest)
        train_accuracy = accuracy_score(ytrain, clf.predict(xtrain))
        test_accuracy = accuracy_score(ytest, y_pred)
        r2 = sklearn.metrics.r2_score(ytest, y_pred)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        r2s.append(r2)
        ic(train_accuracy, test_accuracy, r2)#, np.mean(scores), np.std(scores))

        if prediction:
            for spk, val, label in prediction:
                acc = accuracy_score(val, clf.predict(spk.T))
                pred_dict[label].append(acc)

    ic(np.mean(train_acc), np.std(train_acc), np.mean(test_acc), np.std(test_acc), np.mean(r2s), np.std(r2s))
    if prediction:
        avg_pred_dict = {label: np.mean(accs) for label, accs in pred_dict.items()}
        std_pred_dict = {label: np.std(accs) for label, accs in pred_dict.items()}
        ic(avg_pred_dict)

    return


training_ix = (trial_df['reward'] == 1)  & (trial_df['correct'] == 1)
incorrect_ix = (~np.isnan(trial_df['action'])) & (trial_df['correct'] == 0)
withheld_ix = (trial_df['correct'] == 1) & (trial_df['reward'] == 0)

# spikes_trial_binned = []
# licks_trial_binned = []
# for i, trial in trial_df.iterrows():
#     trial_start = trial['start_time']
#     if not np.isnan(trial['reward_time']):
#         trial_end = trial['reward_time']
#     elif not np.isnan(trial['choice_time']):
#         trial_end = trial['choice_time']
#     else:
#         trial_end = trial_start
#
#     binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, hpc_spike_times, hpc_spike_clusters, hpc_cluster_ids,
#                                                    bin_size=0.5, pre_time=2.0, post_time=2.0)
#     # binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, v1_spike_times, v1_spike_clusters, v1_cluster_ids,
#     #                                                bin_size=0.05, pre_time=2.0, post_time=2.0)
#     # binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, md_spike_times, md_spike_clusters, md_cluster_ids,
#     #                                                bin_size=0.05, pre_time=2.0, post_time=2.0)
#     # binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, spike_times, cluster_ids,
#     #                                                bin_size=0.05, pre_time=2.0, post_time=2.0)
#     bin_states = np.ones(binned_spikes.shape[1]) * trial['state_int']  # 0=right active, 1=left active
#     bin_choices = np.ones(binned_spikes.shape[1]) * trial['action']  # 0=right, 1=left
#     spikes_trial_binned.append(dict(trial_ix=i, binned_spikes=binned_spikes, bin_edges=bin_edges,
#                                     bin_states=bin_states, bin_choices=bin_choices))
#
#     # binned_licks, lick_bin_edges = bin_licks_to_trial(trial_start, trial_end, event_df,
#     #                                                   pre_time=2.0, post_time=2.0, bin_size=0.1)
#     # bin_states = np.ones(binned_licks.shape[1]) * trial['state_int']
#     # bin_choices = np.ones(binned_licks.shape[1]) * trial['action']
#     # licks_trial_binned.append(dict(trial_ix=i, binned_licks=binned_licks, bin_edges=lick_bin_edges,
#     #                                bin_states=bin_states, bin_choices=bin_choices))
#
# # save trial bins
# # p = output_path / ('{}_spikes_trial_binned.pkl'.format(sess_id_full))
# # with p.open('wb') as f:
# #     pkl.dump(spikes_trial_binned, f)
# #
# # p = output_path / ('{}_licks_trial_binned.pkl'.format(sess_id_full))
# # with p.open('wb') as f:
# #     pkl.dump(licks_trial_binned, f)
# #
# # # load trial bins if already calculated
# # p = output_path / ('{}_spikes_trial_binned.pkl'.format(sess_id_full))
# # with p.open('rb') as f:
# #     spikes_trial_binned = pkl.load(f)
# #
# # p = output_path / ('{}_licks_trial_binned.pkl'.format(sess_id_full))
# # with p.open('rb') as f:
# #     licks_trial_binned = pkl.load(f)
#
#
# # logistic regression
# # all_spike_bins = np.concatenate([trial['binned_spikes'] for trial in spikes_trial_binned], axis=1)
# # all_state_bins = np.concatenate([trial['bin_states'] for trial in spikes_trial_binned])
# # all_choice_bins = np.concatenate([trial['bin_choices'] for trial in spikes_trial_binned])
# # classifier, accuracy, _, _ = decode_from_spikes(all_spike_bins, all_choice_bins)
# # print(f"Choice decoding accuracy: {accuracy:.2f}")
# # classifier, accuracy, _, _ = decode_from_spikes(all_spike_bins, all_state_bins)
# # print(f"State decoding accuracy: {accuracy:.2f}")
#

#
# # bins for the max probability period after choice
# train_spike_bins, train_state_bins, train_choice_bins = [], [], []
# incorrect_spike_bins, incorrect_state_bins, incorrect_choice_bins = [], [], []
# withheld_spike_bins, withheld_state_bins, withheld_choice_bins = [], [], []
#
#
# for ix in np.where(training_ix)[0]:
#     # trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['start_time'] - .5) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['start_time'] + 0))
#     trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['choice_time'] - .5) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['choice_time'] + 0))
#     # trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['reward_time'] + 0) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['reward_time'] + .5))
#     _spike_bins = spikes_trial_binned[ix]['binned_spikes'][:, trial_bin_ix[:-1]]
#     _state_bins = spikes_trial_binned[ix]['bin_states'][trial_bin_ix[:-1]]
#     _choice_bins = spikes_trial_binned[ix]['bin_choices'][trial_bin_ix[:-1]]
#
#     train_spike_bins.append(_spike_bins)
#     train_state_bins.append(_state_bins)
#     train_choice_bins.append(_choice_bins)
#
#     # plot_trial(trial_start=trial_df.iloc[ix]['start_time'],
#     #            choice_t=trial_df.iloc[ix]['choice_time'],
#     #            reward_t=trial_df.iloc[ix]['reward_time'],
#     #            spike_times=spike_times,
#     #            cluster_ids=cluster_ids,
#     #            lick_times=event_df[(event_df['Event'] == 'left_entry') | (event_df['Event'] == 'right_entry')]['Time'].values,
#     #            pre_time=2.0, post_time=2.0, bin_size=0.05)
#
# for ix in np.where(incorrect_ix)[0]:
#     trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['choice_time'] + 0) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['choice_time'] + 1.5))
#     _spike_bins = spikes_trial_binned[ix]['binned_spikes'][:, trial_bin_ix[:-1]]
#     _state_bins = spikes_trial_binned[ix]['bin_states'][trial_bin_ix[:-1]]
#     _choice_bins = spikes_trial_binned[ix]['bin_choices'][trial_bin_ix[:-1]]
#
#     incorrect_spike_bins.append(_spike_bins)
#     incorrect_state_bins.append(_state_bins)
#     incorrect_choice_bins.append(_choice_bins)
#
# for ix in np.where(withheld_ix)[0]:
#     trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['choice_time'] + 0) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['choice_time'] + 1.5))
#     _spike_bins = spikes_trial_binned[ix]['binned_spikes'][:, trial_bin_ix[:-1]]
#     _state_bins = spikes_trial_binned[ix]['bin_states'][trial_bin_ix[:-1]]
#     _choice_bins = spikes_trial_binned[ix]['bin_choices'][trial_bin_ix[:-1]]
#
#     withheld_spike_bins.append(_spike_bins)
#     withheld_state_bins.append(_state_bins)
#     withheld_choice_bins.append(_choice_bins)
#
# train_spike_bins = np.concatenate(train_spike_bins, axis=1)
# train_state_bins = np.concatenate(train_state_bins)
# train_choice_bins = np.concatenate(train_choice_bins)

hpc_spikes_trial_binned = bin_spikes(hpc_spike_times, hpc_spike_clusters, hpc_cluster_ids, bin_size=0.5, pre_time=2.0, post_time=2.0)
hpc_correct_spike_bins_pre, hpc_correct_state_bins_pre, hpc_correct_choice_bins_pre = make_classifier_bins(hpc_spikes_trial_binned, trial_df, training_ix, event='choice_time', bounds=(-0.5, 0.0))
hpc_incorrect_spike_bins_pre, hpc_incorrect_state_bins_pre, hpc_incorrect_choice_bins_pre = make_classifier_bins(hpc_spikes_trial_binned, trial_df, incorrect_ix, event='choice_time', bounds=(-0.5, 0.0))
hpc_withheld_spike_bins_pre, hpc_withheld_state_bins_pre, hpc_withheld_choice_bins_pre = make_classifier_bins(hpc_spikes_trial_binned, trial_df, withheld_ix, event='choice_time', bounds=(-.5, 0))
hpc_correct_spike_bins_post, hpc_correct_state_bins_post, hpc_correct_choice_bins_post = make_classifier_bins(hpc_spikes_trial_binned, trial_df, training_ix, event='choice_time', bounds=(0, 0.5))
hpc_incorrect_spike_bins_post, hpc_incorrect_state_bins_post, hpc_incorrect_choice_bins_post = make_classifier_bins(hpc_spikes_trial_binned, trial_df, incorrect_ix, event='choice_time', bounds=(0, 0.5))
hpc_withheld_spike_bins_post, hpc_withheld_state_bins_post, hpc_withheld_choice_bins_post = make_classifier_bins(hpc_spikes_trial_binned, trial_df, withheld_ix, event='choice_time', bounds=(0, 0.5))

### GATHER CROSS-VALIDATED RESULTS #################################
# cv_dicts = []
# ic('gather CV results')
# cv_result = cv_decode_only(hpc_correct_spike_bins_pre, hpc_correct_state_bins_pre)
# cv_result['sess_id'] = sess_id_full
# cv_result['region'] = 'HPC'
# cv_result['trial_type'] = 'correct'
# cv_result['event'] = 'choice_time'
# cv_result['target'] = 'state'
# cv_result['bounds'] = (-0.5, 0.0)
# cv_result['n_units'] = hpc_correct_spike_bins_pre.shape[0]
# cv_dicts.append(cv_result)
#
# cv_result = cv_decode_only(hpc_correct_spike_bins_post, hpc_correct_state_bins_post)
# cv_result['sess_id'] = sess_id_full
# cv_result['region'] = 'HPC'
# cv_result['trial_type'] = 'correct'
# cv_result['event'] = 'choice_time'
# cv_result['target'] = 'state'
# cv_result['bounds'] = (0.0, 0.5)
# cv_result['n_units'] = hpc_correct_spike_bins_pre.shape[0]
# cv_dicts.append(cv_result)
#
# cv_result = cv_decode_only(hpc_incorrect_spike_bins_pre, hpc_incorrect_state_bins_pre)
# cv_result['sess_id'] = sess_id_full
# cv_result['region'] = 'HPC'
# cv_result['trial_type'] = 'incorrect'
# cv_result['event'] = 'choice_time'
# cv_result['target'] = 'state'
# cv_result['bounds'] = (-0.5, 0.0)
# cv_result['n_units'] = hpc_correct_spike_bins_pre.shape[0]
# cv_dicts.append(cv_result)
#
# cv_result = cv_decode_only(hpc_incorrect_spike_bins_post, hpc_incorrect_state_bins_post)
# cv_result['sess_id'] = sess_id_full
# cv_result['region'] = 'HPC'
# cv_result['trial_type'] = 'incorrect'
# cv_result['event'] = 'choice_time'
# cv_result['target'] = 'state'
# cv_result['bounds'] = (0, 0.5)
# cv_result['n_units'] = hpc_correct_spike_bins_pre.shape[0]
# cv_dicts.append(cv_result)
#
# cv_result = cv_decode_only(hpc_withheld_spike_bins_pre, hpc_withheld_state_bins_pre)
# cv_result['sess_id'] = sess_id_full
# cv_result['region'] = 'HPC'
# cv_result['trial_type'] = 'withheld'
# cv_result['event'] = 'choice_time'
# cv_result['target'] = 'state'
# cv_result['bounds'] = (-0.5, 0.0)
# cv_result['n_units'] = hpc_correct_spike_bins_pre.shape[0]
# cv_dicts.append(cv_result)
#
# cv_result = cv_decode_only(hpc_withheld_spike_bins_post, hpc_withheld_state_bins_post)
# cv_result['sess_id'] = sess_id_full
# cv_result['region'] = 'HPC'
# cv_result['trial_type'] = 'withheld'
# cv_result['event'] = 'choice_time'
# cv_result['target'] = 'state'
# cv_result['bounds'] = (0, 0.5)
# cv_result['n_units'] = hpc_correct_spike_bins_pre.shape[0]
# cv_dicts.append(cv_result)
#
# # save results to dataframe
# results_df = pd.DataFrame(cv_dicts)
# results_df_path = processed_data_path / current_mouse / 'HPC_decoding_CV_results.csv'
# if results_df_path.exists():
#     results_df.to_csv(results_df_path, mode='a', header=False, index=False)
# else:
#     results_df.to_csv(results_df_path, index=False)
#
# exit()

# ic('Training on correct trials; decoding on other target-variables with cross-validation')
# decode_with_cv(hpc_correct_spike_bins_pre, hpc_correct_state_bins_pre,[(hpc_correct_spike_bins_post, hpc_correct_state_bins_post, 'HPC correct post-choice'),
#                                                                        (hpc_incorrect_spike_bins_pre, hpc_incorrect_state_bins_pre, 'HPC incorrect pre-choice'),
#                                                                        (hpc_incorrect_spike_bins_post, hpc_incorrect_state_bins_post, 'HPC incorrect post-choice'),
#                                                                        (hpc_withheld_spike_bins_pre, hpc_withheld_state_bins_pre, 'HPC withheld pre-choice'),
#                                                                        (hpc_withheld_spike_bins_post, hpc_withheld_state_bins_post, 'HPC withheld post-choice')],
#                )
#
# v1_spikes_trial_binned = bin_spikes(v1_spike_times, v1_spike_clusters, v1_cluster_ids, bin_size=0.5, pre_time=2.0, post_time=2.0)
# v1_train_spike_bins, v1_train_state_bins, v1_train_choice_bins = make_classifier_bins(v1_spikes_trial_binned, trial_df, training_ix)


### GATHER SHUFFLE DECODER RESULTS #################################
# ic('decoding choice from HPC')
df_dicts = []

# choice_classifier, results_dict, _, _ = decode_from_spikes(hpc_correct_spike_bins_pre, hpc_correct_choice_bins_pre)
# choice_classifier, results_dict = shuffle_decode_only(hpc_correct_spike_bins_pre, hpc_correct_choice_bins_pre)
# results_dict['trial_type'] = 'correct'
# results_dict['event'] = 'choice_time'
# results_dict['target'] = 'choice'
# results_dict['bounds'] = (-0.5, 0.0)
# results_dict['n_units'] = hpc_correct_spike_bins_pre.shape[0]
# results_dict['region'] = 'HPC'
# results_dict['sess_id'] = sess_id_full
# results_dict['training'] = True
# df_dicts.append(results_dict)
# ic(f"HPC choice decoding accuracy (0-0.5s pre-choice): {results_dict['test_accuracy']:.2f}")

ic('training decoder on HPC state')
state_classifier, results_dict = shuffle_decode_only(hpc_correct_spike_bins_pre, hpc_correct_state_bins_pre)
results_dict['trial_type'] = 'correct'
results_dict['event'] = 'choice_time'
results_dict['target'] = 'state'
results_dict['bounds'] = (-0.5, 0.0)
results_dict['n_units'] = hpc_correct_spike_bins_pre.shape[0]
results_dict['region'] = 'HPC'
results_dict['sess_id'] = sess_id_full
results_dict['training'] = True
df_dicts.append(results_dict)
# ic(f"HPC state decoding accuracy (0-0.5s pre-choice): {results_dict['test_accuracy']:.2f}")

ic('Testing decoder on other time periods and trial types')
accuracy = accuracy_score(hpc_correct_state_bins_post, state_classifier.predict(hpc_correct_spike_bins_post.T))
results_dict = {}
results_dict['trial_type'] = 'correct'
results_dict['event'] = 'choice_time'
results_dict['target'] = 'state'
results_dict['bounds'] = (0, 0.5)
results_dict['n_units'] = hpc_correct_spike_bins_post.shape[0]
results_dict['region'] = 'HPC'
results_dict['sess_id'] = sess_id_full
results_dict['training'] = False
results_dict['test_accuracy'] = accuracy
df_dicts.append(results_dict)
# ic("HPC state decoding accuracy (0-0.5s post-choice): {:.2f}".format(results_dict['test_accuracy']))

# accuracy = accuracy_score(hpc_correct_state_bins_pre, state_classifier.predict(hpc_train_spike_bins.T))
# results_dict = {}
# results_dict['trial_type'] = 'correct'
# results_dict['event'] = 'choice_time'
# results_dict['target'] = 'choice'
# results_dict['bounds'] = (0, 0.5)
# results_dict['n_units'] = hpc_train_spike_bins.shape[0]
# results_dict['region'] = 'HPC'
# results_dict['sess_id'] = sess_id_full
# results_dict['training'] = False
# results_dict['test_accuracy'] = accuracy
# df_dicts.append(results_dict)
# ic("HPC choice decoding accuracy (0-0.5s post-choice): {:.2f}".format(accuracy))
#
#
accuracy = accuracy_score(hpc_incorrect_state_bins_pre, state_classifier.predict(hpc_incorrect_spike_bins_pre.T))
results_dict = {}
results_dict['trial_type'] = 'incorrect'
results_dict['event'] = 'choice_time'
results_dict['target'] = 'state'
results_dict['bounds'] = (-0.5, 0.0)
results_dict['n_units'] = hpc_incorrect_spike_bins_pre.shape[0]
results_dict['region'] = 'HPC'
results_dict['sess_id'] = sess_id_full
results_dict['training'] = False
results_dict['test_accuracy'] = accuracy
df_dicts.append(results_dict)

# ic("HPC state decoding accuracy (-0.5-0s pre-choice, incorrect trials): {:.2f}".format(results_dict['test_accuracy']))
# accuracy = accuracy_score(hpc_train_choice_bins, choice_classifier.predict(hpc_train_spike_bins.T))
# results_dict = {}
# results_dict['trial_type'] = 'incorrect'
# results_dict['event'] = 'choice_time'
# results_dict['target'] = 'choice'
# results_dict['bounds'] = (-0.5, 0.0)
# results_dict['n_units'] = hpc_train_spike_bins.shape[0]
# results_dict['region'] = 'HPC'
# results_dict['sess_id'] = sess_id_full
# results_dict['training'] = False
# results_dict['test_accuracy'] = accuracy
# df_dicts.append(results_dict)
# ic("HPC choice decoding accuracy (-0.5-0s pre-choice, incorrect trials): {:.2f}".format(accuracy))

accuracy = accuracy_score(hpc_incorrect_state_bins_post, state_classifier.predict(hpc_incorrect_spike_bins_post.T))
results_dict = {}
results_dict['trial_type'] = 'incorrect'
results_dict['event'] = 'choice_time'
results_dict['target'] = 'state'
results_dict['bounds'] = (0, 0.5)
results_dict['n_units'] = hpc_incorrect_spike_bins_post.shape[0]
results_dict['region'] = 'HPC'
results_dict['sess_id'] = sess_id_full
results_dict['training'] = False
results_dict['test_accuracy'] = accuracy
df_dicts.append(results_dict)
# ic("HPC state decoding accuracy (0-0.5s post-choice, incorrect trials): {:.2f}".format(results_dict['test_accuracy']))

# accuracy = accuracy_score(hpc_train_choice_bins, choice_classifier.predict(hpc_train_spike_bins.T))
# results_dict = {}
# results_dict['trial_type'] = 'incorrect'
# results_dict['event'] = 'choice_time'
# results_dict['target'] = 'choice'
# results_dict['bounds'] = (0, 0.5)
# results_dict['n_units'] = hpc_train_spike_bins.shape[0]
# results_dict['region'] = 'HPC'
# results_dict['sess_id'] = sess_id_full
# results_dict['training'] = False
# results_dict['test_accuracy'] = accuracy
# df_dicts.append(results_dict)
# ic("HPC choice decoding accuracy (0-0.5s post-choice, incorrect trials): {:.2f}".format(accuracy))


accuracy = accuracy_score(hpc_withheld_state_bins_pre, state_classifier.predict(hpc_withheld_spike_bins_pre.T))
results_dict = {}
results_dict['trial_type'] = 'withheld'
results_dict['event'] = 'choice_time'
results_dict['target'] = 'state'
results_dict['bounds'] = (-0.5, 0.0)
results_dict['n_units'] = hpc_withheld_spike_bins_pre.shape[0]
results_dict['region'] = 'HPC'
results_dict['sess_id'] = sess_id_full
results_dict['training'] = False
results_dict['test_accuracy'] = accuracy
df_dicts.append(results_dict)
# ic("HPC state decoding accuracy (-0.5-0s pre-choice, withheld trials): {:.2f}".format(results_dict['test_accuracy']))
# accuracy = accuracy_score(hpc_train_choice_bins, choice_classifier.predict(hpc_train_spike_bins.T))
# results_dict = {}
# results_dict['trial_type'] = 'withheld'
# results_dict['event'] = 'choice_time'
# results_dict['target'] = 'choice'
# results_dict['bounds'] = (-0.5, 0.0)
# results_dict['n_units'] = hpc_train_spike_bins.shape[0]
# results_dict['region'] = 'HPC'
# results_dict['sess_id'] = sess_id_full
# results_dict['training'] = False
# results_dict['test_accuracy'] = accuracy
# df_dicts.append(results_dict)
# ic("HPC choice decoding accuracy (-0.5-0s pre-choice, withheld trials): {:.2f}".format(accuracy))


accuracy = accuracy_score(hpc_withheld_state_bins_post, state_classifier.predict(hpc_withheld_spike_bins_post.T))
results_dict = {}
results_dict['trial_type'] = 'withheld'
results_dict['event'] = 'choice_time'
results_dict['target'] = 'state'
results_dict['bounds'] = (0, 0.5)
results_dict['n_units'] = hpc_withheld_spike_bins_pre.shape[0]
results_dict['region'] = 'HPC'
results_dict['sess_id'] = sess_id_full
results_dict['training'] = False
results_dict['test_accuracy'] = accuracy
df_dicts.append(results_dict)
# ic("HPC state decoding accuracy (0-0.5s post-choice, withheld trials): {:.2f}".format(results_dict['test_accuracy']))

# accuracy = accuracy_score(hpc_withheld_state_bins_post, choice_classifier.predict(hpc_train_spike_bins.T))
# results_dict = {}
# results_dict['trial_type'] = 'withheld'
# results_dict['event'] = 'choice_time'
# results_dict['target'] = 'choice'
# results_dict['bounds'] = (0, 0.5)
# results_dict['n_units'] = hpc_train_spike_bins.shape[0]
# results_dict['region'] = 'HPC'
# results_dict['sess_id'] = sess_id_full
# results_dict['training'] = False
# results_dict['test_accuracy'] = accuracy
# df_dicts.append(results_dict)
# ic("HPC choice decoding accuracy (0-0.5s post-choice, withheld trials): {:.2f}".format(accuracy))

# save results to dataframe
results_df = pd.DataFrame(df_dicts)
results_df_path = processed_data_path / current_mouse / 'HPC_decoding_shuffle_results.csv'
if results_df_path.exists():
    results_df.to_csv(results_df_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(results_df_path, index=False)


# choice_classifier, (accuracy, r2), _, _ = decode_from_spikes(v1_train_spike_bins, v1_train_choice_bins)
# ic(f"V1 choice decoding accuracy: {accuracy:.2f}")
# state_classifier, (accuracy, r2), _, _ = decode_from_spikes(v1_train_spike_bins, v1_train_state_bins)
# ic(f"V1 state decoding accuracy: {accuracy:.2f}")

