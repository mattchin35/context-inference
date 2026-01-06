import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Iterable
from pathlib import Path
from icecream import ic
import fileIO
import collect_events
import re
import pickle as pkl


def plot_event_raster(array: np.ndarray, labels: Iterable[str], linewidths=.75) -> plt.Figure:
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(18)
    ax.eventplot(array, linelengths=0.6, linewidths=linewidths, color='black')
    # ax.set_title('Behavioral Raster Plot', fontsize=24)
    plt.xlabel("Time (seconds)", fontsize=24)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=20)
    # ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=20)  # not possible in earlier matplotlib versions
    plt.xticks(fontsize=20)
    plt.tight_layout()
    return fig, ax


def generate_session_raster(event_df: pd.DataFrame, session_info: dict,
                            fig_name: str, plot_path: str, timespan: tuple = (0, np.inf),
                            fig_format: str = 'png') -> None:
    assert fig_format in ['png', 'pdf', 'svg', 'jpg'], "Provide a valid figure format. Choose from png, pdf, svg, jpg"
    event_df = event_df[event_df['Event'] != 'exit_standby']
    if timespan[1] == np.inf:
        timespan = (np.amin(event_df['Time']), np.amax(event_df['Time']))

    event_list = ['left_entry', 'right_entry']
    event_list = collect_events.get_choice_events(event_df, event_list)
    event_list = collect_events.get_context_events(event_df, event_list)
    event_list = collect_events.get_reward_events(event_df, event_list)
    # event_list += ['stimulus_A_on', 'stimulus_B_on']

    event_labels = np.array(collect_events.make_event_labels(event_list, session_info))
    event_array = collect_events.generate_event_array(event_df, event_list, timespan)

    # remove events to simplify visualization
    ix_no_reward = [i for i, e in enumerate(event_list) if re.findall('reward_amount: 0', e)]  # remove reward 0
    ix_wrong_choice = [i for i, e in enumerate(event_list) if re.findall('wrong_choice', e)]
    ix_trial_t = [i for i, e in enumerate(event_list) if re.findall('trial_', e)]
    ix_remove = np.concatenate((ix_no_reward, ix_wrong_choice, ix_trial_t)).astype(int)

    event_labels = np.delete(event_labels, ix_remove)
    event_array = np.delete(event_array, ix_remove, axis=0)
    raster_figure, raster_ax = plot_event_raster(event_array, event_labels)

    # plt.show()
    savename = os.path.join(plot_path, '{}.{}'.format(fig_name, fig_format))
    raster_figure.savefig(savename, format=fig_format)
    print("Saved raster plot as {}".format(savename))
    plt.close('all')


def colorblock_raster(event_df: pd.DataFrame, fig_name: str, plot_path: str, session_info: dict,
                      timespan: tuple = (0, np.inf), fig_format: str = 'png', plot_choices=False) -> None:
    """
    Create a raster plot from a single behavior session.
    """
    color_dict = {'right active': 'darkred', 'left active': 'blue', 'dark period': 'black'}
    # 'right': 'cyan', 'left': 'darkgreen', 0: 'cyan', 1: 'darkkhaki'}
    state_dict = {0: 'right', 1: 'left'}

    cleaned_df = event_df[event_df['Event'] != 'exit_standby']
    if timespan[1] == np.inf:
        timespan = (np.amin(cleaned_df['Time']), np.amax(cleaned_df['Time']))

    if plot_choices:
        event_list = ['correct_choice_left_patch', 'wrong_choice_right_patch', 'correct_choice_right_patch',
                      'wrong_choice_left_patch', 'enter_left_patch', 'enter_right_patch']
        if session_info['use_dark_period']:
            event_list += ['enter_dark_period']
        event_array = collect_events.generate_event_array(cleaned_df, event_list, timespan)

        left_choice = np.concatenate([event_array[0], event_array[1]])
        right_choice = np.concatenate([event_array[2], event_array[3]])
        event_array = [left_choice, right_choice, event_array[4], event_array[5]]
        event_labels = ['Left\nchoice', 'Right\nchoice', 'enter left patch', 'enter right patch']
        if session_info['use_dark_period']:
            event_array += [event_array[6]]
            event_labels += ['enter dark period']
        linewidths = 0.3

    else:
        event_list = ['left_entry', 'right_entry', 'enter_left_patch', 'enter_right_patch']
        event_labels = ['Left\nlick', 'Right\nlick', 'enter left patch', 'enter right patch']
        if session_info['use_dark_period']:
            event_list += ['enter dark period']
            event_labels += ['enter dark period']

        event_array = collect_events.generate_event_array(cleaned_df, event_list, timespan)
        linewidths = 0.2

    end_time = np.amax([np.amax([np.amax(e) for e in event_array]), timespan[1]])

    states = ['left active'] * len(event_array[2]) + ['right active'] * len(event_array[3])
    if session_info['use_dark_period']:
        states + ['dark period'] * len(event_array[4])

    state_times = np.concatenate(event_array[2:])
    state_sort_ix = np.argsort(state_times)
    states = np.array(states)[state_sort_ix]
    state_times = state_times[state_sort_ix]
    unique_states = np.unique(states).tolist()

    # stimulus plotting
    stimulus_times = cleaned_df[cleaned_df['Event'].isin(['stimulus_A_on', 'stimulus_B_on'])]
    stim_st = stimulus_times.Time.values
    stim_end = stim_st + 10  # 10 seconds of stimulus, hardcoded - eventually want to use session_info

    raster_figure, ax = plot_event_raster(event_array[:2], event_labels[:2], linewidths=linewidths)

    # color blocks
    state_alpha = 0.15
    for i in range(states.size - 1):
        plt.axvspan(state_times[i], state_times[i + 1], color=color_dict[states[i]], alpha=state_alpha)
        # if states[i] in unique_states:
        #     plt.axvspan(state_times[i], state_times[i + 1], color=color_dict[states[i]], alpha=state_alpha,
        #                 label=states[i])
        #     unique_states.remove(states[i])
        # else:
        #     plt.axvspan(state_times[i], state_times[i + 1], color=color_dict[states[i]], alpha=state_alpha)
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
    savename = os.path.join(plot_path, '{}.{}'.format(fig_name, 'png'))
    raster_figure.savefig(savename, format='png')
    # savename = os.path.join(plot_path, '{}.{}'.format(fig_name, 'svg'))
    # raster_figure.savefig(savename, format='svg')

    # savename = os.path.join(plot_path, '{}.svg'.format(fig_name, 'svg'))
    # raster_figure.savefig(savename, format='svg')

    plt.close('all')
    print("Saved raster plot as {}".format(savename))


def main1():
    colorblock_raster(trial_df, sess_id_full + '_lick_context_raster', figure_path, plot_choices=False)
    colorblock_raster(trial_df, sess_id_full + '_choice_context_raster', figure_path, plot_choices=True)


def plot_session():
    data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    raw_behavior_folder = data_home / 'rpi/CT014_2025-12-16_153200'
    processed_data_path = data_home / 'processed'
    figure_path = data_home / 'figures'

    mouse = 'CT014'
    date = '2025-12-16'
    timestamp = '153200'
    sess_id = mouse + '_' + date
    sess_id_full = mouse + '_' + date + '_' + timestamp
    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',')
    session_log = raw_behavior_folder / '{}.log'.format(sess_id_full)
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    if not figure_path.exists():
        figure_path.mkdir()

    generate_session_raster(event_df, session_info=session_info, fig_name=sess_id_full + '_raster', plot_path=figure_path, fig_format='png')
    colorblock_raster(event_df, sess_id_full + '_lick_context_raster', session_info=session_info, plot_path=figure_path, plot_choices=False)
    colorblock_raster(event_df, sess_id_full + '_choice_context_raster', session_info=session_info, plot_path=figure_path, plot_choices=True)


if __name__ == '__main__':
    plot_session()

