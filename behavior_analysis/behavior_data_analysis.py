import pandas as pd
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fileIO
import re
from typing import Iterable
    

"""
Read and analyze ABC1C2 data as provided by Mitch. Not using a GUI to make the plots here - hardcoding only.

1. plot behavior as mice enter and exit blocks
2. set a correct criteria, plot a. trials to first correct choice b. trials to criterion (over a moving window) c. number of correct choices within a block
That should be complicated enough for now....
"""


def generate_event_array(df: pd.DataFrame, events: list, timespan: tuple) -> np.ndarray:
    event_array = []
    for e in events:
        event_df = df.loc[df['Event'] == e, 'Time']
        ix = (event_df >= timespan[0]) & (event_df <= timespan[1])
        event_array.append(event_df[ix].values)
    return np.array(event_array, dtype=object)


def plot_event_raster(array: np.ndarray, labels: Iterable[str]) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(18)
    ax.eventplot(array, linelengths=0.6, linewidths=0.15, color='black')
    ax.set_title('Behavioral Raster Plot', fontsize=24)
    plt.xlabel("Time (seconds)", fontsize=24)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=20)
    # ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=20)  # not possible in earlier matplotlib versions
    plt.xticks(fontsize=20)
    plt.tight_layout()
    return fig


def get_pump_events(df: pd.DataFrame, event_list: list = []) -> list:
    unique_events = np.unique(df['Event'].values)
    for e in unique_events:
        if re.fullmatch('pump.*', e):
            event_list.append(e)

    return event_list


def get_context_events(df: pd.DataFrame, event_list: list = []) -> list:
    unique_events = np.unique(df['Event'].values)
    for e in unique_events:
        if re.fullmatch('enter_Context.*', e):
            event_list.append(e)

    return event_list


def make_event_labels(events: list) -> list:
    event_labels = [s.replace('_', ' ') for s in events]
    event_labels = [s.replace('pump1', 'right') for s in event_labels]
    event_labels = [s.replace('pump2', 'left') for s in event_labels]
    return event_labels


def generate_session_raster(session_df: pd.DataFrame, fig_name: str, plot_path: str, timespan: tuple = (0, np.inf)) \
        -> None:
    """
    Create a raster plot from a single behavior session.
    """
    cleaned_df = session_df[session_df['Event'] != 'exit_standby']

    event_list = ['left_entry', 'right_entry', 'enter_intercontext_interval']
    event_list = get_pump_events(cleaned_df, event_list)
    event_list = get_context_events(cleaned_df, event_list)
    event_labels = make_event_labels(event_list)
    event_array = generate_event_array(cleaned_df, event_list, timespan)

    raster_figure = plot_event_raster(event_array, event_labels)

    # plt.show()
    fig_format = 'svg'
    savename = os.path.join(plot_path, '{}.{}'.format(fig_name, fig_format))
    raster_figure.savefig(savename, format=fig_format)
    plt.close('all')


def plot_binned_behavior(session_df, fig_path, fig_name, bin_range=None, plot=False):
    """
    session is a dictionary of pandas dataframes.
    """
    if bin_range is None:
        bin_range = (0, np.inf)
        # bin_range = [0, 30]

    event_list = ['left_entry', 'right_entry']
    event_list = get_pump_events(session_df, event_list)
        # , 'pump1_reward_0', 'pump1_reward_3', 'pump2_reward_0', 'pump2_reward_3'] #, 'current_ITI_2', 'current_ITI_3', 'current_ITI_4']

    # unique_events = session_df['Event'].unique()
    # # regex search for the available pump options
    # for e in unique_events:
    #     if re.fullmatch('pump.*', e):
    #         event_list.append(e)

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
    bin_counts = dict()

    session_df['Bin'] = np.zeros(session_df['Time'].size)
    for event in event_list:
        if event in session_df['Event'].values:
            ix = session_df['Event'] == event
            b = pd.cut(session_df.loc[ix, 'Time'], bins)
            session_df.loc[ix, 'Bin'] = b
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
        plt_options = (('Left lick', '-', 'k'), ('Right lick', '--', 'gray'))
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
        df = fileIO.load(mouse, date, reprocess=True)
        bin_counts, session_df, bins = plot_binned_behavior(df, plot_path, 'sample_bin_plot', plot=False)
        _, session_summary, _ = block_analysis(bin_counts, session_df, sess_ID)
        w = count_water(df)
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


def count_water(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all water delivery events to determine mouse water intake.
    """

    # get dataframe indices for all reward events
    reward_event_types = get_pump_events(session_df)
    ix = np.zeros(session_df.shape[0]).astype(bool)
    for e in reward_event_types:
        ix = ix | (session_df['Event'] == e)

    # create lists of reward events of each type
    reward_events = session_df.loc[ix, 'Event'].values
    pump1_regex = re.compile("pump1.*")
    pump2_regex = re.compile("pump2.*")
    pump1_list = list(filter(pump1_regex.match, reward_events))  # Read Note below
    pump2_list = list(filter(pump2_regex.match, reward_events))  # Read Note below

    # add up events to determine total rewards
    pump1_water = np.sum([float(re.findall(r'(\d+)', a)[1]) for a in pump1_list])
    pump2_water = np.sum([float(re.findall(r'(\d+)', a)[1]) for a in pump2_list])
    total_water = pump1_water + pump2_water
    source = ['right water', 'left water', 'total water']
    amount = [pump1_water, pump2_water, total_water]

    session_water = pd.DataFrame(dict(water_source=source, amount=amount))
    return session_water


if __name__ == '__main__':
    plot_path = '../figures'

    """Analyze the data from a single mouse session"""
    current_mouse = 'HD005'
    current_date = '2023-08-29'
    sess_ID = current_mouse + '-' + current_date

    # df = fileIO.load(current_mouse, current_date, reprocess=False)
    # generate_session_raster(df, sess_ID + '_raster', plot_path)
    # water = count_water(df)
    # bin_counts, session_df, bins = plot_binned_behavior(df,  plot_path, sess_ID + '_bin_plot', plot=False)
    # block_analysis(bin_counts, session_df, sess_ID)

    """Analyze data of a mouse across several sessions"""
    sess_list = [('MF24', '2023-07-14'),
                 ('MF24', '2023-07-17'),
                 ('MF24', '2023-07-18'),
                 ('MF24', '2023-07-19')]
    compare_sessions(sess_list, plot_path, plot_name='MF24_sample')


