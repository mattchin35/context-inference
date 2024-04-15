import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fileIO
import re
from typing import Iterable
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import pickle as pkl
import collect_events
import raster_plots as rp
import session_overview


"""
This was code to analyze changes from block to block, but it needs to be refactored.
"""


def _plot_bins(bin_counts, bin_range, bin_centers, transition_list,
               fig_path, fig_name, fig_format='svg'):
    # colors = ['red', 'green', 'red', 'purple', 'orange']  # Add more colors if needed
    colors = ['cyan', 'darkgreen', 'lightgray', 'plum', 'indigo']  # Add more colors if needed
    cmap = {transition: color for transition, color in zip(transition_list, colors)}

    fig, ax = plt.subplots(figsize=(12, 6))

    patches = []
    for transition in transition_list:
        color = cmap[transition]
        patch = mpatches.Patch(color=color, alpha=0.2, label=transition)
        patches.append(patch)

    plot_outputs = ['left_entry', 'right_entry']
    plt_options = (('Left choice', '-', 'k'), ('Right choice', '--', 'gray'))
    for j, o in enumerate(plot_outputs):
        y = bin_counts[o].values
        ax.plot(bin_centers, y, linewidth=1, label=plt_options[j][0], linestyle=plt_options[j][1],
        # ax.plot(bin_counts.loc['intervals', i], y, linewidth=1, label=plt_options[j][0], linestyle=plt_options[j][1],
                color=plt_options[j][2])

    for i in range(bin_counts.shape[0]):
        if bin_range[0] <= i < bin_range[1]:
            # ax.axvspan(bins[i], bins[i + 1], facecolor=cmap[bin_type[i]], alpha=0.075)
            ax.axvspan(bin_counts.loc['intervals', i].left, bin_counts.loc['intervals', i].right,
                       facecolor=cmap[bin_counts.loc['bin_type', i]], alpha=0.075)

    plt.title(f"Lick Choice Summary", fontsize=16)
    plt.xlabel('Time(seconds)', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    plt.xlim(bin_centers[0], bin_centers[-1])
    plt.ylim(bottom=0)
    plt.legend(fancybox=False, fontsize=16)
    plt.tight_layout()
    # plt.show()

    savename = os.path.join(fig_path, '{}.{}'.format(fig_name, fig_format))
    fig.savefig(savename, format=fig_format)


def plot_binned_behavior(session_df: pd.DataFrame, fig_path, fig_name, bin_range=(0, np.inf), plot=False):

    event_list = ['left_entry', 'right_entry']
    event_list = rp.get_choice_events(session_df, event_list)

    transition_list = ['enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval', 'enter_ContextC1',
                       'enter_ContextC2']

    transition_ix = (session_df['Event'] == 'enter_ContextA') | (session_df['Event'] == 'enter_ContextB') | \
         (session_df['Event'] == 'enter_ContextC1') | (session_df['Event'] == 'enter_ContextC2') | \
         (session_df['Event'] == 'enter_intercontext_interval')

    session_df = session_df.round({'Time': 3})
    sorted_sess = session_df.sort_values(by=['Time'])
    sorted_transitions = session_df[transition_ix].sort_values(by=['Time'])

    bins = sorted_transitions['Time'].values
    bin_type = sorted_transitions['Event'].values[:-1]  # exclude the last bin - it's an ending intercontext interval
    bin_centers = (sorted_transitions[:-1]['Time'].values + sorted_transitions[1:]['Time'].values) / 2
    bin_counts = OrderedDict()

    session_df['Bin'] = np.zeros(session_df['Time'].size)
    for event in event_list:
        if event in session_df['Event'].values:
            ix = session_df['Event'] == event
            b = pd.cut(session_df.loc[ix, 'Time'], bins)
            session_df.loc[ix, 'Bin'] = b  # this is saving the interval of each event
            counts = session_df.loc[ix, 'Time'].groupby(b).count()
            bin_counts[event] = counts.values

    # roundbins = np.round(bins, decimals=3)
    intervals = []
    for j, b in enumerate(bin_type):
        iv = pd.Interval(bins[j], bins[j + 1], closed='right')
        intervals.append(iv)

    bin_counts['bin_type'] = bin_type
    bin_counts['intervals'] = intervals
    bin_counts = pd.DataFrame(bin_counts)

    if plot:
        # colors = ['red', 'green', 'red', 'purple', 'orange']  # Add more colors if needed
        colors = ['cyan', 'darkgreen', 'lightgray', 'plum', 'indigo']  # Add more colors if needed
        cmap = {transition: color for transition, color in zip(transition_list, colors)}

        fig, ax = plt.subplots(figsize=(12, 6))

        patches = []
        for transition in transition_list:
            color = cmap[transition]
            patch = mpatches.Patch(color=color, alpha=0.2, label=transition)
            patches.append(patch)

        plot_outputs = ['left_entry', 'right_entry']
        plt_options = (('Left choice', '-', 'k'), ('Right choice', '--', 'gray'))
        for j, o in enumerate(plot_outputs):
            y = bin_counts[o].values
            ax.plot(bin_centers, y, linewidth=1, label=plt_options[j][0], linestyle=plt_options[j][1],
                    color=plt_options[j][2])

        for i in range(len(bins) - 1):
            if bin_range[0] <= i < bin_range[1]:
                ax.axvspan(bins[i], bins[i + 1], facecolor=cmap[bin_type[i]], alpha=0.075)

        plt.title(f"Lick Choice Summary", fontsize=16)
        plt.xlabel('Time(seconds)', fontsize=14)
        plt.ylabel('Count', fontsize=14)

        plt.xlim(bin_centers[0], bin_centers[-1])
        plt.ylim(bottom=0)
        plt.legend(fancybox=False, fontsize=16)
        plt.tight_layout()
        # plt.show()

        fig_format = 'svg'
        savename = os.path.join(fig_path, '{}.{}'.format(fig_name, fig_format))
        fig.savefig(savename, format=fig_format)

    return bin_counts, session_df, bins


"""
Collect analyses of overall behavior within a block and overall within a session (i.e. no within-block trial analyses).

Overall within session

Between sessions / summarize session
1. Number and percent of correct blocks (70% correct choices??)
2. Number and percent of correct blocks by block type

Trials within session
1. Trials to correct side first
2. Trials to correct side consistently 
"""


def block_analysis(bin_counts, session_df, sess_ID, plot=False):
    # left vs right
    # get context blocks (i.e. no intercontext intervals)
    block_ix = (bin_counts['bin_type'] == 'enter_ContextA') | (bin_counts['bin_type'] == 'enter_ContextB') | \
               (bin_counts['bin_type'] == 'enter_ContextC1') | (bin_counts['bin_type'] == 'enter_ContextC2')

    # block_ix = block_ix.values
    blocks = bin_counts[block_ix]
    nblocks = np.sum(block_ix.values)
    # criterion = bins['bin_type'].map(lambda x: x in ['enter_ContextA', 'enter_ContextB'])
    # blocks = bins[criterion]

    Aix = (blocks['bin_type'] == 'enter_ContextA').values
    Bix = (blocks['bin_type'] == 'enter_ContextB').values
    C1ix = (blocks['bin_type'] == 'enter_ContextC1').values
    C2ix = (blocks['bin_type'] == 'enter_ContextC2').values

    norm = blocks['left_entry'].values + blocks['right_entry'].values
    active_ix = norm != 0

    ### BLOCKS OVERALL WITHIN SESSION ###
    # 1. Overall licks to each side in block (bin counts)
    # 2. "Choices" correct (licks after cue signal) in each block // Reward deliveries in block(these are the same for deterministic rewards)
    # 3. Choices per block (i.e. task engagement)
    # 4. Block correct?  (70 % correct choices??)
    # as -1 to 1
    # ix = (blocks['left_entry'].values[active_ix] - blocks['right_entry'].values[active_ix]) / norm[active_ix]
    percent = blocks['right_entry'].values[active_ix] / norm[active_ix]
    percent_nan = np.zeros(nblocks)
    percent_nan[active_ix] = percent
    percent_nan[~active_ix] = np.nan

    if plot:
        f, ax = plt.subplots()
        X = np.arange(nblocks)
        ax.scatter(X[active_ix], percent)
        plt.yticks([0,.5,1], ['100% Left', 'Split', '100% Right'])
        plt.title('Percent Left vs Right licks per block')
    # ax[1].scatter(X, percent_nan)
    # ax[1].plot(block_ix, np.ma.masked_where(norm == 0, ))

    # do a bunch of regex stuff here for the whole fxn
    events = session_df['Event'].unique()

    # get the Rall, Lall ix
    pump_events = [e for e in events if re.fullmatch('pump.*', e)]
    pump1_events = [e for e in pump_events if re.fullmatch('pump1.*', e)]
    pump1_rewards = [e for e in pump1_events if not re.fullmatch('pump1_reward_0', e)]
    pump2_events = [e for e in events if re.fullmatch('pump2.*', e)]
    pump2_rewards = [e for e in pump2_events if not re.fullmatch('pump2_reward_0', e)]

    all_rewards = np.sum(np.array([blocks[e] for e in pump_events]), axis=0)
    active_ix = all_rewards > 0
    # get percent correct for L and R
    Lreward_perc = np.sum(np.array([blocks[e] for e in pump1_rewards]), axis=0)
    Lreward_perc[active_ix] = Lreward_perc[active_ix] / all_rewards[active_ix]
    Rreward_perc = np.sum(np.array([blocks[e] for e in pump2_rewards]), axis=0)
    Rreward_perc[active_ix] = Rreward_perc[active_ix] / all_rewards[active_ix]

    # correct vs incorrect choices/reward deliveries AND overall block correct
    thresh = .7
    correct = np.zeros(nblocks)
    correct[Aix] = Rreward_perc[Aix]
    correct[Bix] = Lreward_perc[Bix]
    correct[C1ix] = Rreward_perc[C1ix]
    correct[C2ix] = Lreward_perc[C2ix]
    block_correct = correct > thresh

    if plot:
        fCorr, axCorr = plt.subplots()
        axCorr.scatter(X, correct)
        axCorr.plot(X, block_correct, 'k--')
        plt.title('Percent correct choices/reward delivery')
        plt.ylabel('% Correct')

        # Choices per block/task engagement
        fChoice, axChoice = plt.subplots()
        plt.plot(X, all_rewards)
        plt.title('Task engagement - Number of "choices" made')

    block_summary = dict(percent_right_lick=percent, percent_block_correct=correct, binary_block_correct=block_correct, thresh=thresh,
         n_choices=all_rewards)

    ### SUMMARIZE SESSION ###
    # 1. Number and percent of correct blocks (70 % correct choices??)
    # 2. Number and percent of correct blocks by block type
    # contexts which are not present will return nan
    if block_correct.size > 0:
        overall_correct = np.sum(block_correct) / block_correct.size
    else:
        overall_correct = np.nan

    if correct[Aix].size > 0:
        A_correct = np.sum(correct[Aix]) / correct[Aix].size
    else:
        A_correct = np.nan

    if correct[Bix].size > 0:
        B_correct = np.sum(correct[Bix]) / correct[Bix].size
    else:
        B_correct = np.nan

    if correct[C1ix].size > 0:
        C1_correct = np.sum(correct[C1ix]) / correct[C1ix].size
    else:
        C1_correct = np.nan

    if correct[C2ix].size > 0:
        C2_correct = np.sum(correct[C2ix]) / correct[C2ix].size
    else:
        C2_correct = np.nan

    session_summary = dict(A_correct=A_correct, B_correct=B_correct, C1_correct=C1_correct, C2_correct=C2_correct,
                           overall_correct=overall_correct)
    session_summary = pd.DataFrame(data=session_summary, index=[sess_ID])

    # A is rightlick; C1 is too
    # pump2 is right (so A=pump2=right)

    ### WITHIN BLOCKS ###
    ### Trials to correct side first
    # for current setup, this works - will modify for probabilistic
    trials_to_correct, trials_to_thresh = [], []
    bin_types = blocks['bin_type'].values
    for j, iv in enumerate(blocks['intervals']):
        Rreward_times = session_df.loc[(session_df['Event'] == 'pump1_reward_3') & (session_df['Bin'] == iv), 'Time']
        Rfail_times = session_df.loc[(session_df['Event'] == 'pump1_reward_0') & (session_df['Bin'] == iv), 'Time']
        Lreward_times = session_df.loc[(session_df['Event'] == 'pump2_reward_3') & (session_df['Bin'] == iv), 'Time']
        Lfail_times = session_df.loc[(session_df['Event'] == 'pump2_reward_0') & (session_df['Bin'] == iv), 'Time']

        Rall = np.sort(np.concatenate([Rreward_times, Rfail_times]))
        Lall = np.sort(np.concatenate([Lreward_times, Lfail_times]))
        Tix = np.argsort(np.concatenate([Rall, Lall]))
        T = np.concatenate([Rall, Lall])[Tix]

        R = np.ones(len(Rall))
        L = np.zeros(len(Lall))
        Tbinary = np.concatenate([R, L])[Tix]

        ttc = np.nan
        if (bin_types[j] == 'enter_ContextA') | (bin_types[j] == 'enter_ContextC1'):
            # list choices as correct or incorrect
            choices_correct = Tbinary == 1
            if Rall.size > 0:
                ttc = np.sum(T < Rall[0])

        elif (bin_types[j] == 'enter_ContextB') | (bin_types[j] == 'enter_ContextC2'):
            choices_correct = Tbinary == 0
            if Lall.size > 0:
                ttc = np.sum(T < Lall[0])

        else:
            raise NameError('faulty context type found at block {}'.format(j))

        # moving average choice?
        # filtersize = 3
        # sp.ndimage.uniform_filter1d(Tbinary, filtersize)

        # forward average
        forward_avg = np.zeros(Tbinary.size)
        tt_thresh = np.nan
        for i in range(Tbinary.size):
            forward_avg = np.sum(choices_correct[i:]) / choices_correct[i:].size
            if forward_avg > thresh:
                tt_thresh = i

        # Rall.size / (Rall.size + Lall.size)
        trials_to_correct.append(ttc)
        trials_to_thresh.append(tt_thresh)

    trials_to_correct = np.array(trials_to_correct)
    trials_to_thresh = np.array(trials_to_thresh)
    trial_summary = dict(trials_to_correct=trials_to_correct, trials_to_thresh=trials_to_thresh)

    if plot:
        fTTC, axTTC = plt.subplots()
        plt.scatter(np.arange(trials_to_correct.size), trials_to_correct, label='trials to correct')
        plt.scatter(np.arange(trials_to_thresh.size), trials_to_thresh, label='trials to threshold')
        plt.title('Trials to correct choice')
        plt.legend()

        plt.show()

    return block_summary, session_summary, trial_summary


def compare_sessions(session_list, plot_path, plot_name):
    """
    1. collect a number of sessions, load their session summaries
    session_list is a list of (mouse, date) pairs
    2. use the session summaries to plot blocks across days
    """

    summaries = []
    dates = []
    water = []
    for mouse, date in session_list:
        sess_ID = mouse + '-' + date
        df = fileIO.load(mouse, date, load_cleaned=False)
        bin_counts, session_df, bins = plot_binned_behavior(df, plot_path, 'sample_bin_plot', plot=False)
        _, session_summary, _ = block_analysis(bin_counts, session_df, sess_ID)
        w = session_overview.total_water_delivery(df)
        summaries.append(session_summary)
        dates.append(date)
        water.append(w)

    summaries = pd.concat(summaries, axis=0)
    water = pd.concat(water, axis=0)
    print(summaries)

    X = np.arange(len(session_list))
    f, ax = plt.subplots(2, 1, figsize=(12, 6))
    plt.sca(ax[0])
    plt.plot(X, summaries['A_correct'], label='A')
    plt.plot(X, summaries['B_correct'], label='B')
    plt.plot(X, summaries['C1_correct'], label='C1')
    plt.plot(X, summaries['C2_correct'], label='C2')
    plt.plot(X, summaries['overall_correct'], label='Overall')
    plt.title('Percent Correct by Context Type')
    plt.xticks(ticks=X, labels=dates)
    plt.ylim([0,1])
    plt.xlabel('Session Date')
    plt.ylabel('Percent Correct')
    plt.legend()

    plt.sca(ax[1])
    plt.plot(X, water.loc[water['water_source'] == 'right water', 'amount'], label='right rewards')
    plt.plot(X, water.loc[water['water_source'] == 'left water', 'amount'], label='left rewards')
    plt.plot(X, water.loc[water['water_source'] == 'total water', 'amount'], label='total rewards')
    plt.title('Water rewards')
    plt.xticks(ticks=X, labels=dates)
    plt.xlabel('Session Date')
    plt.ylabel('Water vol (uL)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    fig_format = 'svg'
    savename = os.path.join(plot_path, '{}.{}'.format(plot_name, fig_format))
    f.savefig(savename, format=fig_format)
