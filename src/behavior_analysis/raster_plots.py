import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import src.behavior_analysis.fileIO as fileIO
from typing import Iterable
from pathlib import Path
from icecream import ic
import src.behavior_analysis.collect_events as collect_events
import re


def plot_event_raster(array: np.ndarray, labels: Iterable[str]) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(18)
    ax.eventplot(array, linelengths=0.6, linewidths=0.1, color='black')
    # ax.set_title('Behavioral Raster Plot', fontsize=24)
    plt.xlabel("Time (seconds)", fontsize=24)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=20)
    # ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=20)  # not possible in earlier matplotlib versions
    plt.xticks(fontsize=20)
    plt.tight_layout()
    return fig, ax


def generate_session_raster(session_df: pd.DataFrame, fig_name: str, plot_path: str, timespan: tuple = (0, np.inf),
                            fig_format: str = 'png') -> None:
    """
    Create a raster plot from a single behavior session.
    """
    cleaned_df = session_df[session_df['Event'] != 'exit_standby']
    if timespan[1] == np.inf:
        timespan = (0, np.amax(cleaned_df['Time']))

    event_list = ['left_entry', 'right_entry']
    event_list = collect_events.get_choice_events(cleaned_df, event_list)
    event_list = collect_events.get_context_events(cleaned_df, event_list)
    event_list += ['giving_reward']
    event_list += ['stimulus_A_on', 'stimulus_B_on']
    event_labels = np.array(collect_events.make_event_labels(event_list))
    event_array = collect_events.generate_event_array(cleaned_df, event_list, timespan)

    # remove events to simplify visualization
    ix_no_reward = [i for i, e in enumerate(event_list) if re.findall('reward_amount: 0', e)]  # remove reward 0
    ix_wrong_choice = [i for i, e in enumerate(event_list) if re.findall('wrong_choice', e)]
    ix_remove = np.concatenate((ix_no_reward, ix_wrong_choice))

    event_labels = np.delete(event_labels, ix_remove)
    event_array = np.delete(event_array, ix_remove, axis=0)
    raster_figure, raster_ax = plot_event_raster(event_array, event_labels)

    # plt.show()
    savename = os.path.join(plot_path, '{}.{}'.format(fig_name, fig_format))
    raster_figure.savefig(savename, format=fig_format)
    plt.close('all')
    print("Saved raster plot as {}".format(savename))


def colorblock_raster(session_df: pd.DataFrame, fig_name: str, plot_path: str, timespan: tuple = (0, np.inf),
                      fig_format: str = 'png') -> None:
    """
    Create a raster plot from a single behavior session.
    """
    color_dict = {'right active': 'darkred', 'left active': 'blue', 'dark period': 'black'}
    # 'right': 'cyan', 'left': 'darkgreen', 0: 'cyan', 1: 'darkkhaki'}
    state_dict = {0: 'right', 1: 'left'}

    cleaned_df = session_df[session_df['Event'] != 'exit_standby']
    if timespan[1] == np.inf:
        timespan = (0, np.amax(cleaned_df['Time']))

    event_list = ['left_entry', 'right_entry', 'enter_left_patch', 'enter_right_patch', 'enter_dark_period']
    event_labels = ['Left\nlick', 'Right\nlick', 'enter left patch', 'enter right patch', 'enter dark period']
    event_array = collect_events.generate_event_array(cleaned_df, event_list, timespan)
    end_time = np.amax([np.amax([np.amax(e) for e in event_array]), timespan[1]])

    states = ['left active'] * len(event_array[2]) + ['right active'] * len(event_array[3]) + ['dark period'] * len(event_array[4])
    state_times = np.concatenate(event_array[2:])
    state_sort_ix = np.argsort(state_times)
    states = np.array(states)[state_sort_ix]
    state_times = state_times[state_sort_ix]
    unique_states = np.unique(states).tolist()

    # stimulus plotting
    stimulus_times = cleaned_df[cleaned_df['Event'].isin(['stimulus_A_on', 'stimulus_B_on'])]
    stim_st = stimulus_times.Time.values
    stim_end = stim_st + 10  # 10 seconds of stimulus, hardcoded - eventually want to use session_info

    raster_figure, ax = plot_event_raster(event_array[:2], event_labels[:2])

    # color blocks
    state_alpha = 0.15
    for i in range(states.size - 1):
        if states[i] in unique_states:
            plt.axvspan(state_times[i], state_times[i + 1], color=color_dict[states[i]], alpha=state_alpha,
                        label=states[i])
            unique_states.remove(states[i])
        else:
            plt.axvspan(state_times[i], state_times[i + 1], color=color_dict[states[i]], alpha=state_alpha)
    plt.axvspan(state_times[-1], end_time, color=color_dict[states[-1]], alpha=state_alpha)
    plt.xlim(timespan)

    for i in range(stim_st.size):
        plt.axvspan(stim_st[i], stim_end[i], color='k', alpha=.2)

    # Add legend
    patches = [mpatches.Patch(color=color_dict[k], label=k) for k in color_dict.keys()]
    plt.legend(handles=patches, fontsize=20, fancybox=False)
    plt.ylim(-.4, 1.4)
    plt.xlabel("Time (seconds)", fontsize=40)
    ax.tick_params(axis='x', which='major', labelsize=25)
    ax.tick_params(axis='y', which='major', labelsize=45)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)

    raster_figure.tight_layout()
    # plt.show()
    savename = os.path.join(plot_path, '{}.png'.format(fig_name, 'png'))
    raster_figure.savefig(savename, format='png')

    # savename = os.path.join(plot_path, '{}.svg'.format(fig_name, 'svg'))
    # raster_figure.savefig(savename, format='svg')

    plt.close('all')
    print("Saved raster plot as {}".format(savename))


if __name__ == '__main__':
    # data_path = Path('../../data/raw/Mitch_behavior')
    data_path = Path('../../data/processed') / 'cleaned_sessions'
    figure_path = Path('../../reports/figures')
    # plot_path = Path('../../reports/figures/F31_Apr2024')

    """Analyze the data from a single mouse session"""
    mouse = 'CT002'  # 'MF23'
    date = '2024-08-26'  # '2023-08-18'
    sess_ID = mouse + '_' + date

    # df = fileIO.load(sess_ID, data_path, load_cleaned=False)
    df = fileIO.load_cleaned_data(sess_ID, data_path)
    print(df)
    # generate_session_raster(df, sess_ID + '_raster', plot_path)
    plot_path = figure_path / mouse
    # plot_path = figure_path / sess_ID
    if not plot_path.exists():
        plot_path.mkdir()

    # generate_session_raster(df, sess_ID + '_raster', plot_path)
    colorblock_raster(df, sess_ID + '_context_raster', plot_path)
