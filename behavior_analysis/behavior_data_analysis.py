import pandas as pd
import os
import numpy as np
import scipy as sp
from tkinter import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fileIO
# from IPython.display import display, HTML
# pd.set_option('display.max_rows', None) #these four lines permit printing of the entire data frame
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
# display(HTML("<style>.container { width:100% !important; }</style>"))

"""
Read and analyze ABC1C2 data as provided by Mitch. Not using a GUI to make the plots here - hardcoding only.

1. plot behavior as mice enter and exit blocks
2. set a correct criteria, plot a. trials to first correct choice b. trials to criterion (over a moving window) c. number of correct choices within a block
That should be complicated enough for now....
"""


def generate_session_raster(session, fig_name, fig_path, timespan=None):
    """
    Create a raster plot from a single session.
    session is a dictionary of pandas dataframes.
    """
    if timespan is None:
        timespan = (0, np.inf)

    # Concatenate the dataframes in the specific_subsets dictionary into a single dataframe
    cleaned_df = pd.concat(session.values())
    cleaned_df = cleaned_df[cleaned_df['Output'] != 'exit_standby']
    events = np.unique(cleaned_df['Output'].values)

    # behavior_outputs = ['left_entry', 'right_entry', 'pump1_reward_3', 'pump2_reward_3']
    event_list = ['left_entry', 'right_entry',
                  'pump1_reward_0', 'pump1_reward_3',
                  'pump2_reward_0', 'pump2_reward_3',
                  'enter_intercontext_interval']  # ITI is intertrial, in the same block; intercontext is between blocks

    event_labels = ['Left lick', 'Right lick',
                    'left reward 0', 'left reward 3',
                    'right reward 0', 'right reward 3',
                    'enter intercontext interval']

    if 'enter_ContextA' and 'enter_ContextC1' in events:
        session_type = 'full'
        event_list += ['enter_ContextA', 'enter_ContextB', 'enter_ContextC1', 'enter_ContextC2']
        event_labels += ['enter Context A', 'enter Context B', 'enter Context C1', 'enter Context C2']
    elif 'enter_ContextA' in events:
        session_type = 'AB'
        event_list += ['enter_ContextA', 'enter_ContextB']
        event_labels += ['enter Context A', 'enter Context B']
    else:  # 'enter_ContextC1' in events:
        session_type = 'C1C2'
        event_list += ['enter_ContextC1', 'enter_ContextC2']
        event_labels += ['enter Context C1', 'enter Context C2']

    # output_list = ['left_entry', 'right_entry', 'pump1_reward_0', 'pump1_reward_3', 'pump2_reward_0',
    #                'pump2_reward_3', 'ITI', 'enter_intercontext_interval',
    #                'enter_ContextC1', 'enter_ContextC2', 'enter_ContextA', 'enter_ContextB']

    arrays_dict = {}
    for e in event_list:
        sub_df = cleaned_df[cleaned_df['Output'] == e]
        if e in ['enter_ContextA', 'enter_ContextB', 'enter_ContextC1', 'enter_ContextC2'] and sub_df.empty:
            continue

        arrays_dict[e + '_timestamp_array'] = np.array(sub_df['Time'])

    abridged_arrays_dict = {}
    for key, value in arrays_dict.items():
        abridged_array = value[(value >= timespan[0]) & (value <= timespan[1])]
        abridged_arrays_dict[key + '_abridged'] = abridged_array

    array_of_timestamp_arrays = np.array(list(abridged_arrays_dict.values()), dtype=object)

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(18)
    ax.eventplot(array_of_timestamp_arrays, linelengths=0.6, linewidths=0.15, color='black')
    ax.set_title('Behavioral Raster Plot', fontsize=24)
    plt.xlabel("Time (seconds)", fontsize=24)
    ax.set_yticks(np.arange(len(event_list)))
    ax.set_yticklabels(event_labels, fontsize=20)
    plt.xticks(fontsize=20)
    plt.tight_layout()

    fig_format = 'svg'
    savename = os.path.join(fig_path, '{}.{}'.format(fig_name, fig_format))
    fig.savefig(savename, format=fig_format)
    # plt.show()


def plot_binned_behavior(session, fig_name, fig_path, bin_range=None, plot=False):
    """
    session is a dictionary of pandas dataframes.
    """
    if bin_range is None:
        bin_range = (0, np.inf)
        # bin_range = [0, 30]

    output_list = ['left_entry', 'right_entry', 'pump1_reward_0', 'pump1_reward_3',
                   'pump2_reward_0', 'pump2_reward_3', 'current_ITI_2', 'current_ITI_3', 'current_ITI_4']

    transition_list = ['enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval', 'enter_ContextC1',
                       'enter_ContextC2']

    # Generate bins based on the timestamps of the transition list
    bins = []
    bin_type = []
    for transition in transition_list:
        if transition in session.keys():
            timestamps = session[transition]['Time'].values
            bins.extend(timestamps)  # collect the timestamps for every transition into one list
            bin_type.extend([transition] * len(timestamps))

    bins, bin_type = np.array(bins), np.array(bin_type)
    ix = np.argsort(bins)  # make sure the bins are in ascending order
    bins = bins[ix]
    bins = np.round(bins, decimals=3)
    bin_type = bin_type[ix][:-1]  # exclude the last bin - it's an ending intercontext interval
    bin_centers = (bins[:-1] + bins[1:])/2

    # Iterate over the output list
    bin_dict = dict(bin_type=bin_type)#, bin_bounds=np.around(intervals, decimals=3))
    for output in output_list:
        if output in session.keys():
            # Bin the 'Time' values of the output
            # session[output]['Bin'] = pd.cut(session[output]['Time'], bins, include_lowest=True)
            session[output]['Bin'] = pd.cut(session[output]['Time'], bins)

            # Count the number of 'Time' values in each bin
            counts = session[output].groupby('Bin').count()
            bin_dict[output] = counts['Time'].values

    intervals = []
    # roundbins = np.round(bins, decimals=3)
    for j, b in enumerate(bin_type):
        iv = pd.Interval(bins[j], bins[j + 1], closed='right')
        intervals.append(iv)
    bin_dict['intervals'] = intervals

    bin_df = pd.DataFrame(bin_dict)
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
        plt_options = (('Left lick', '-', 'k'), ('Right lick', '--', 'gray'))
        for j, o in enumerate(plot_outputs):
            y = bin_df[o].values
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

    return bin_df, session, bins


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


def block_analysis(bin_df, session, sess_ID):
    # left vs right
    # get context blocks (i.e. no intercontext intervals)
    block_ix = (bin_df['bin_type'] == 'enter_ContextA') | (bin_df['bin_type'] == 'enter_ContextB') | \
               (bin_df['bin_type'] == 'enter_ContextC1') | (bin_df['bin_type'] == 'enter_ContextC2')

    # block_ix = block_ix.values
    blocks = bin_df[block_ix]
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

    f, ax = plt.subplots()

    X = np.arange(nblocks)
    ax.scatter(X[active_ix], percent)
    plt.yticks([0,.5,1], ['100% Left', 'Split', '100% Right'])
    plt.title('Percent Left vs Right licks per block')
    # ax[1].scatter(X, percent_nan)
    # ax[1].plot(block_ix, np.ma.masked_where(norm == 0, ))

    # correct vs incorrect choices/reward deliveries AND overall block correct
    all_rewards = (blocks['pump1_reward_3'] + blocks['pump2_reward_3'] + blocks['pump1_reward_0'] + blocks['pump1_reward_0']).values
    active_ix = all_rewards > 0
    Lreward = np.zeros_like(all_rewards)
    Lreward[active_ix] = blocks['pump1_reward_3'].values[active_ix] / all_rewards[active_ix]
    Rreward = np.zeros_like(all_rewards)
    Rreward[active_ix] = blocks['pump2_reward_3'].values[active_ix] / all_rewards[active_ix]

    thresh = .7
    correct = np.zeros(nblocks)
    correct[Aix] = Rreward[Aix]
    correct[Bix] = Lreward[Bix]
    correct[C1ix] = Lreward[C1ix]
    correct[C2ix] = Lreward[C2ix]
    block_correct = correct > thresh

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
    overall_correct = np.sum(block_correct) / block_correct.size
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

    session_summary = dict(A_correct=A_correct, B_correct=B_correct, C1_correct=C1_correct, C2_correct=C2_correct, overall_correct=overall_correct)
    session_summary = pd.DataFrame(data=session_summary, index=[sess_ID])

    # A is rightlick
    # pump2 is right (so A=pump2=right)

    ### WITHIN BLOCKS ###
    ### Trials to correct side first
    # for current setup, this works - will modify for probabilistic
    trials_to_correct, trials_to_thresh = [], []
    bin_types = blocks['bin_type'].values
    for j, iv in enumerate(blocks['intervals']):
        Rreward_ix = session['pump1_reward_3']['Bin'] == iv
        Rreward = session['pump1_reward_3']['Time'][Rreward_ix].values

        Rfail_ix = session['pump1_reward_0']['Bin'] == iv
        Rfail = session['pump1_reward_0']['Time'][Rfail_ix].values

        Lreward_ix = session['pump2_reward_3']['Bin'] == iv
        Lreward = session['pump2_reward_3']['Time'][Lreward_ix].values

        Lfail_ix = session['pump2_reward_0']['Bin'] == iv
        Lfail = session['pump2_reward_0']['Time'][Lfail_ix].values

        Rall = np.sort(np.concatenate([Rreward, Rfail]))
        Lall = np.sort(np.concatenate([Lreward, Lfail]))
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

    fTTC, axTTC = plt.subplots()
    plt.scatter(np.arange(trials_to_correct.size), trials_to_correct, label='trials to correct')
    plt.scatter(np.arange(trials_to_thresh.size), trials_to_thresh, label='trials to threshold')
    plt.title('Trials to correct choice')
    plt.legend()

    plt.show()
    return block_summary, session_summary, trial_summary


def compare_sessions(session_list):
    """
    1. collect a number of sessions, load their session summaries
    session_list is a list of (mouse, date) pairs
    2. use the session summaries to plot blocks across days
    """

    summaries = []
    for mouse, date in session_list:
        sess_ID = mouse + '-' + date
        data_dict = fileIO.load(mouse, date)
        bin_df, session, bins = plot_binned_behavior(data_dict[mouse][0], 'sample_bin_plot', plot_path)
        _, session_summary, _ = block_analysis(bin_df, session, sess_ID)
        summaries.append(session_summary)

    summaries = pd.concat(summaries, axis=0)


    pass


if __name__ == '__main__':

    current_mouse = 'MF06'
    current_date = '2023-06-13'
    plot_path = '../figures'
    sess_ID = current_mouse + '-' + current_date

    # retrieves a dictionary - call the mouse name to load the data
    data_dict = fileIO.load(current_mouse, current_date)
    # generate_session_raster(data_dict[current_mouse][0], 'sample_raster', plot_path)
    bin_df, session, bins = plot_binned_behavior(data_dict[current_mouse][0], 'sample_bin_plot', plot_path)
    block_analysis(bin_df, session, sess_ID)


