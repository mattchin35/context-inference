import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from icecream import ic
from pathlib import Path
import pickle as pkl
from one.api import ONE
from brainbox.io.one import load_iti
import brainbox.behavior.wheel as wh
from brainbox.io.one import load_wheel_reaction_times
from ibllib.io.extractors.ephys_fpga import extract_wheel_moves
from ibllib.io.extractors.training_wheel import extract_first_movement_times
from ibllib.pipes.dynamic_pipeline import get_trials_tasks
from brainbox.behavior.training import compute_performance
from brainbox.behavior.training import plot_psychometric
from brainbox.task.trials import find_trial_ids
import one.alf.io as alfio

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.multiclass import OneVsRestClassifier
import timeit


def bin_spikes_to_trial(trial_start: float, trial_end: float, spike_times: np.ndarray, spike_clusters: np.ndarray, cluster_ids: np.ndarray,
                        bin_size: float = 0.1, pre_time: float = 2.0, post_time: float = 2.0) -> [np.ndarray, np.ndarray]:
    bin_st = trial_start - pre_time
    bin_end = trial_end + post_time
    try:
        n_bins = int((bin_end - bin_st) / bin_size)
    except ValueError as e:
        print(f"Error calculating number of bins: {e}")
        return
    binned_spikes = np.zeros((cluster_ids.size, n_bins))
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


# def decode_from_spikes(binned_spikes: np.ndarray, bin_value: np.ndarray) -> [object, float]:
#     nanmask = np.isnan(bin_value)
#     if np.any(nanmask):
#         print("Warning: NaN values found in bin_value. Removing corresponding spike bins.")
#         binned_spikes = binned_spikes[:, ~nanmask]
#         bin_value = bin_value[~nanmask]
#
#     X_train, X_test, y_train, y_test = train_test_split(binned_spikes.T, bin_value, test_size=0.2, random_state=42)
#
#     # classifier = LinearRegression()
#     classifier = LogisticRegression(max_iter=10000)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     r2 = sklearn.metrics.r2_score(y_test, y_pred)
#
#     np.random.shuffle(y_train)  # Shuffle fit
#     shuffle_clf = LogisticRegression(max_iter=10000)
#     shuffle_clf.fit(X_train, y_train)
#     shuffle_y_pred = shuffle_clf.predict(X_test)
#     shuffle_accuracy = accuracy_score(y_test, shuffle_y_pred)
#     ic(accuracy, shuffle_accuracy)
#     return classifier, (accuracy, r2), binned_spikes, bin_value


def decode_from_spikes(binned_spikes: np.ndarray, bin_value: np.ndarray) -> [object, float]:
    nanmask = np.isnan(bin_value)
    if np.any(nanmask):
        print("Warning: NaN values found in bin_value. Removing corresponding spike bins.")
        binned_spikes = binned_spikes[:, ~nanmask]
        bin_value = bin_value[~nanmask]

    X_train, X_test, y_train, y_test = train_test_split(binned_spikes.T, bin_value, test_size=0.2, random_state=42)
    clf = LogisticRegression(
        solver="saga",
        penalty="l1",
        max_iter=10000,
        random_state=42,
    )
    # classifier = LogisticRegression(max_iter=10000)
    # classifier = OneVsRestClassifier(classifier)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    ic(train_accuracy, test_accuracy, r2)

    cv_score, permutation_scores, pvalue = permutation_test_score(
        clf, X_train, y_train, scoring="accuracy", cv=5, n_permutations=500
    )
    ic(cv_score, pvalue)

    # np.random.shuffle(y_train)  # Shuffle fit
    # shuffle_clf = LogisticRegression(max_iter=10000)
    # shuffle_clf.fit(X_train, y_train)
    # shuffle_y_pred = shuffle_clf.predict(X_test)
    # shuffle_accuracy = accuracy_score(y_test, shuffle_y_pred)
    # r2_shuffle = sklearn.metrics.r2_score(y_test, y_pred)
    # ic(accuracy, shuffle_accuracy, r2, r2_shuffle)
    return clf, (test_accuracy, r2), binned_spikes, bin_value


one = ONE()
sns.set_style('whitegrid')

experiment_folder = Path('/home/matt/Downloads/ONE/openalyx.internationalbrainlab.org')
lab = 'churchlandlab_ucla'
subject = 'UCLA033'
date = '2022-02-15'
session = '001'
eid = experiment_folder / lab / 'Subjects' / subject / date / session
# eid = Path('/home/matt/Downloads/ONE/openalyx.internationalbrainlab.org/angelakilab/Subjects/NYU-11/2020-02-18/001')
base_path = eid / 'alf'
assert base_path.exists(), "Path does not exist: {}".format(base_path)

experiment_folder = Path('/home/matt/Documents/EXPERIMENTS/')
# experiment_folder = Path('C:/Users/mattc/EinsteinMed Dropbox/Matthew Chin/phd_data/remotework/EXPERIMENTS/')
processed_data_path = experiment_folder / 'processed_data'
output_path = processed_data_path / lab / 'Subjects' / subject / date / session
if not output_path.exists():
    output_path.mkdir(parents=True, exist_ok=True)


trials = one.load_object(eid, 'trials')
trials['iti'] = load_iti(trials)
trials.contrastRight[np.isnan(trials.contrastRight)] = 0
trials.contrastLeft[np.isnan(trials.contrastLeft)] = 0
trials['contrast'] = trials.contrastRight - trials.contrastLeft

choice_types = np.unique(trials['choice'])
choice_classes = {choice: i for i, choice in enumerate(choice_types)}
trials['choice_class'] = np.array([choice_classes[choice] for choice in trials['choice']])

block_types = np.unique(trials['probabilityLeft'])
block_classes = {block: i for i, block in enumerate(block_types)}
trials['block_class'] = np.array([block_classes[block] for block in trials['probabilityLeft']])


trial_df = trials.to_df()
ic(trial_df.iloc[:5, -5:])


probe00_path = base_path / 'probe00'
pykilosort_date = '#2024-05-06#'
electrode_sites = np.load(probe00_path / 'electrodeSites.brainLocationIds_ccf_2017.npy')
electrode_localcoords = np.load(probe00_path / 'electrodeSites.localCoordinates.npy')
electrode_mlapdv = np.load(probe00_path / 'electrodeSites.mlapdv.npy')

sorter_output = probe00_path / 'pykilosort' / pykilosort_date
spike_times = np.load(sorter_output / 'spikes.times.npy')
spike_clusters = np.load(sorter_output / 'spikes.clusters.npy')
cluster_channels = np.load(sorter_output / 'clusters.channels.npy')
# cluster_info = pd.read_csv(sorter_output / 'cluster_info.tsv', sep='\t')
# df=cluster_info.set_index('cluster_id')
# spike_channels = np.array([df['ch'][cl] for cl in spike_clusters])
spike_channels = np.array([cluster_channels[cl] for cl in spike_clusters])

# ic(spike_times.shape, spike_times.dtype)
# ic(spike_clusters.shape, spike_clusters.dtype)
# ic(np.unique(spike_clusters))
# ic(cluster_channels, cluster_channels.shape)

# gather CA1 clusters  - 382 CA1, 463 CA3
ca1_channels = np.arange(384)[electrode_sites == 382]
ca1_clusters = [chan in ca1_channels for chan in cluster_channels]
ca1_cluster_ids = np.where(ca1_clusters)[0]
ca1_spikes_mask = np.zeros_like(spike_times, dtype=bool)

v1_channels = np.arange(384)[electrode_sites == 385]
v1_clusters = [chan in ca1_channels for chan in cluster_channels]
v1_cluster_ids = np.where(ca1_clusters)[0]
v1_spikes_mask = np.zeros_like(spike_times, dtype=bool)

# for i, cl in enumerate(spike_clusters):
#     ca1_spikes_mask[i] = cl in ca1_cluster_ids
for ch in ca1_channels:
    ca1_spikes_mask = ca1_spikes_mask | (spike_channels == ch)
for ch in v1_channels:
    v1_spikes_mask = v1_spikes_mask | (spike_channels == ch)

ca1_spike_times = spike_times[ca1_spikes_mask]
ca1_spike_clusters = spike_clusters[ca1_spikes_mask]
ca1_cluster_ids = np.unique(ca1_spike_clusters)

v1_spike_times = spike_times[v1_spikes_mask]
v1_spike_clusters = spike_clusters[v1_spikes_mask]
v1_cluster_ids = np.unique(v1_spike_clusters)

calculate_bins = False
if calculate_bins:
    print("Calculating trial binned spikes and licks...")
    spikes_trial_binned = []
    licks_trial_binned = []
    for i, trial in trial_df.iterrows():
        trial_start = trial['goCueTrigger_times']
        if not np.isnan(trial['feedback_times']):
            trial_end = trial['feedback_times']
        elif not np.isnan(trial['response_times']):
            trial_end = trial['response_times']
        else:
            trial_end = trial_start

        binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, ca1_spike_times, ca1_spike_clusters, ca1_cluster_ids,
                                                       bin_size=.5, pre_time=1.0, post_time=1.0)
        # binned_spikes, bin_edges = bin_spikes_to_trial(trial_start, trial_end, v1_spike_times, v1_spike_clusters,
        #                                                v1_cluster_ids, bin_size=.5, pre_time=1.0, post_time=1.0)

        # two "states" for ibl date - block probability and trial contrast
        bin_states = np.ones(binned_spikes.shape[1]) * trial['block_class']
        # bin_states = np.ones(binned_spikes.shape[1]) * trial['probabilityLeft']
        # bin_stimulusSide = np.ones(binned_spikes.shape[1]) * np.abs(trial['contrast'])
        bin_contrast = np.ones(binned_spikes.shape[1]) * np.sign(trial['contrast'])
        # bin_choices = np.ones(binned_spikes.shape[1]) * trial['choice_class']
        bin_choices = np.ones(binned_spikes.shape[1]) * trial['choice']
        bin_feedback = np.ones(binned_spikes.shape[1]) * trial['feedbackType']
        spikes_trial_binned.append(dict(trial_ix=i, binned_spikes=binned_spikes, bin_edges=bin_edges,
                                        bin_states=bin_states, bin_choices=bin_choices, bin_contrast=bin_contrast,
                                        bin_feedback=bin_feedback))

        # binned_licks, lick_bin_edges = bin_licks_to_trial(trial_start, trial_end, event_df,
        #                                                   pre_time=2.0, post_time=2.0, bin_size=0.1)
        # bin_states = np.ones(binned_licks.shape[1]) * trial['state_int']
        # bin_choices = np.ones(binned_licks.shape[1]) * trial['action']
        # licks_trial_binned.append(dict(trial_ix=i, binned_licks=binned_licks, bin_edges=lick_bin_edges,
        #                                bin_states=bin_states, bin_choices=bin_choices))

    # save trial bins
    p = output_path / 'spikes_trial_binned.pkl'
    with p.open('wb') as f:
        pkl.dump(spikes_trial_binned, f)

    p = output_path / 'licks_trial_binned.pkl'
    with p.open('wb') as f:
        pkl.dump(licks_trial_binned, f)

else:
    print("Loading trial binned spikes and licks...")
    p = output_path / 'spikes_trial_binned.pkl'
    with p.open('rb') as f:
        spikes_trial_binned = pkl.load(f)

    p = output_path / 'licks_trial_binned.pkl'
    with p.open('rb') as f:
        licks_trial_binned = pkl.load(f)

# inclusion criteria
# valid trials: choice made, feedback times exist, stimOn and firstMovement times exist
# reaction time between .8 and 2 sec
# training trials: valid trials, correct, easy (100% contrast)

valid_ix = (trials['choice'] != 0) & ~np.isnan(trials['feedback_times']) & ~np.isnan(trial_df['stimOn_times']) & ~np.isnan(trial_df['firstMovement_times'])
valid_ix = valid_ix & (trial_df['firstMovement_times'] - trial_df['stimOn_times'] <= 2) & (trial_df['firstMovement_times'] - trial_df['stimOn_times'] >= .08)
training_ix = (trial_df['feedbackType'] == 1) & (trial_df['contrast'].abs() == 1) & ((trials['probabilityLeft'] == 0.8) | (trials['probabilityLeft'] == 0.2)) & valid_ix
train_spike_bins, train_state_bins, train_choice_bins = [], [], []
train_stimside_bins, train_feedback_bins = [], []
for ix in np.where(training_ix)[0]:
    trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['stimOn_times']-.5) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['stimOn_times'] + 0))
    # trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['response_times'] - .5) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['response_times'] + .5))
    # trial_bin_ix = (spikes_trial_binned[ix]['bin_edges'] >= trial_df.iloc[ix]['feedback_times'] + -.25) & (spikes_trial_binned[ix]['bin_edges'] < (trial_df.iloc[ix]['feedback_times'] + .5))
    _spike_bins = spikes_trial_binned[ix]['binned_spikes'][:, trial_bin_ix[:-1]]

    _state_bins = spikes_trial_binned[ix]['bin_states'][trial_bin_ix[:-1]]
    _choice_bins = spikes_trial_binned[ix]['bin_choices'][trial_bin_ix[:-1]]
    _contrast_bins = spikes_trial_binned[ix]['bin_contrast'][trial_bin_ix[:-1]]
    _feedback_bins = spikes_trial_binned[ix]['bin_feedback'][trial_bin_ix[:-1]]

    train_spike_bins.append(_spike_bins)
    train_state_bins.append(_state_bins)
    train_choice_bins.append(_choice_bins)
    train_stimside_bins.append(_contrast_bins)
    train_feedback_bins.append(_feedback_bins)

ic(training_ix.sum())
train_spike_bins = np.concatenate(train_spike_bins, axis=1)
train_state_bins = np.concatenate(train_state_bins)
train_choice_bins = np.concatenate(train_choice_bins)
train_stimside_bins = np.concatenate(train_stimside_bins)
train_feedback_bins = np.concatenate(train_feedback_bins)

print("Beginning decoding...")
choice_classifier, (accuracy, r2), _, _ = decode_from_spikes(train_spike_bins, train_choice_bins)
print(f"Choice decoding accuracy: {accuracy:.2f}, R2: {r2:.2f}")
state_classifier, (accuracy, r2), _, _ = decode_from_spikes(train_spike_bins, train_state_bins)
print(f"State decoding accuracy: {accuracy:.2f}, R2: {r2:.2f}")
contrast_classifier, (accuracy, r2), _, _ = decode_from_spikes(train_spike_bins, train_stimside_bins)
print(f"Contrast decoding accuracy: {accuracy:.2f}, R2: {r2:.2f}")
# feedback_classifier, (accuracy, r2), _, _ = decode_from_spikes(train_spike_bins, train_feedback_bins)
# print(f"Feedback decoding accuracy: {accuracy:.2f}, R2: {r2:.2f}")

