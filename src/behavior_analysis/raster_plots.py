import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Iterable
from pathlib import Path
from src.behavior_analysis import collect_events
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
    event_df = event_df[event_df['Event'] != 'exit_standby']  #potentially not needed
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


def _max_event_time(event_array: Iterable[np.ndarray], timespan: tuple | np.ndarray) -> float:
    """Return max plotted event time while ignoring empty event arrays.

    Parameters
    ----------
    event_array : Iterable[np.ndarray]
        One-dimensional event-time arrays grouped by raster row. Each row stores
        event times in seconds and may be empty.
    timespan : tuple or np.ndarray
        Two-element plot window in seconds.

    Returns
    -------
    float
        Maximum event time in seconds, or `timespan[1]` when all event rows are
        empty.
    """
    maxima = [float(np.amax(events)) for events in event_array if np.asarray(events).size > 0]
    maxima.append(float(timespan[1]))
    return float(np.amax(maxima))


def _state_events_from_colorblock_array(
    event_array: list[np.ndarray],
    use_dark_period: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted block-state labels and entry times for color spans.

    Parameters
    ----------
    event_array : list[np.ndarray]
        Raw event arrays from `collect_events.generate_event_array`. Expected
        row order is left-patch entries, right-patch entries, and optionally
        dark-period entries.
    use_dark_period : bool
        Whether the third row contains dark-period entries.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        State labels and entry times sorted by time. Times are in seconds.
    """
    state_arrays = event_array[2:]
    states = ['left active'] * len(state_arrays[0]) + ['right active'] * len(state_arrays[1])
    if use_dark_period:
        states += ['dark period'] * len(state_arrays[2])

    state_times = np.concatenate(state_arrays)
    if state_times.size == 0:
        raise ValueError("colorblock_raster requires at least one state-entry event.")

    state_sort_ix = np.argsort(state_times)
    return np.array(states)[state_sort_ix], state_times[state_sort_ix]


def _prepare_colorblock_raster_data(
    event_df: pd.DataFrame,
    session_info: dict,
    timespan: tuple = (0, np.inf),
    plot_choices: bool = False,
) -> dict:
    """Prepare event arrays and block spans for a colorblock raster.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event table with shape `(n_events, n_columns)`. Required columns are
        `Event` and `Time`; times are in seconds.
    session_info : dict
        Session metadata. Required key is `use_dark_period`, a bool indicating
        whether dark-period state entries should be included.
    timespan : tuple, default=(0, np.inf)
        Two-element time window in seconds. `np.inf` as the right edge expands
        to the event table's maximum time.
    plot_choices : bool, default=False
        If True, plot left/right choices instead of left/right licks.

    Returns
    -------
    dict
        Prepared arrays and labels. Event arrays are one-dimensional numpy
        arrays of event times in seconds, grouped by raster row. State-entry
        arrays are used to draw context-colored spans.
    """
    cleaned_df = event_df[event_df['Event'] != 'exit_standby']
    if timespan[1] == np.inf:
        timespan = np.array([np.amin(cleaned_df['Time']), np.amax(cleaned_df['Time'])])

    if plot_choices:
        event_list = ['correct_choice_left_patch', 'wrong_choice_right_patch', 'correct_choice_right_patch',
                      'wrong_choice_left_patch', 'enter_left_patch', 'enter_right_patch']
        if session_info['use_dark_period']:
            event_list += ['enter_dark_period']
        raw_event_array = collect_events.generate_event_array(cleaned_df, event_list, timespan)

        left_choice = np.concatenate([raw_event_array[0], raw_event_array[1]])
        right_choice = np.concatenate([raw_event_array[2], raw_event_array[3]])
        event_array = [left_choice, right_choice, raw_event_array[4], raw_event_array[5]]
        event_labels = ['Left\nchoice', 'Right\nchoice', 'enter left patch', 'enter right patch']
        if session_info['use_dark_period']:
            event_array += [raw_event_array[6]]
            event_labels += ['enter dark period']
        linewidths = 0.3

    else:
        event_list = ['left_entry', 'right_entry', 'enter_left_patch', 'enter_right_patch']
        event_labels = ['Left\nlick', 'Right\nlick', 'enter left patch', 'enter right patch']
        if session_info['use_dark_period']:
            event_list += ['enter_dark_period']
            event_labels += ['enter dark period']

        event_array = collect_events.generate_event_array(cleaned_df, event_list, timespan)
        linewidths = 0.2

    end_time = _max_event_time(event_array, timespan)
    states, state_times = _state_events_from_colorblock_array(
        event_array,
        use_dark_period=session_info['use_dark_period'],
    )

    stimulus_times = cleaned_df[cleaned_df['Event'].isin(['stimulus_A_on', 'stimulus_B_on'])]
    stim_st = stimulus_times.Time.values
    stim_end = stim_st + 10

    return {
        "timespan": timespan,
        "event_array": event_array,
        "event_labels": event_labels,
        "linewidths": linewidths,
        "states": states,
        "state_times": state_times,
        "unique_states": np.unique(states).tolist(),
        "end_time": end_time,
        "stim_st": stim_st,
        "stim_end": stim_end,
    }


def plot_colorblock_raster_on_ax(
    ax: plt.Axes,
    event_df: pd.DataFrame,
    session_info: dict,
    timespan: tuple = (0, np.inf),
    plot_choices: bool = False,
    axis_label_size: float = 10,
    tick_label_size: float = 8,
    legend_font_size: float = 7,
) -> plt.Axes:
    """Plot a colorblock lick/choice raster on a caller-owned axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the raster events and context spans.
    event_df : pd.DataFrame
        Event table with shape `(n_events, n_columns)`. Required columns are
        `Event` and `Time`; times are in seconds.
    session_info : dict
        Session metadata. Required key is `use_dark_period`, a bool indicating
        whether dark-period state entries should be included.
    timespan : tuple, default=(0, np.inf)
        Two-element time window in seconds. `np.inf` as the right edge expands
        to the event table's maximum time.
    plot_choices : bool, default=False
        If True, plot left/right choices instead of left/right licks.
    axis_label_size : float, default=10
        Font size for axis labels.
    tick_label_size : float, default=8
        Font size for x/y tick labels.
    legend_font_size : float, default=7
        Font size for the context legend.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in, with raster events and colored context spans.
    """
    color_dict = {'right active': 'darkred', 'left active': 'blue', 'dark period': 'black'}
    prepared = _prepare_colorblock_raster_data(
        event_df=event_df,
        session_info=session_info,
        timespan=timespan,
        plot_choices=plot_choices,
    )

    event_array = prepared["event_array"]
    event_labels = prepared["event_labels"]
    states = prepared["states"]
    state_times = prepared["state_times"]
    end_time = prepared["end_time"]
    state_alpha = 0.15

    ax.eventplot(
        event_array[:2],
        linelengths=0.6,
        linewidths=prepared["linewidths"],
        color='black',
    )
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(event_labels[:2], fontsize=tick_label_size)

    for i in range(states.size - 1):
        ax.axvspan(state_times[i], state_times[i + 1], color=color_dict[states[i]], alpha=state_alpha)
    ax.axvspan(state_times[-1], end_time, color=color_dict[states[-1]], alpha=state_alpha)

    for stim_start, stim_stop in zip(prepared["stim_st"], prepared["stim_end"]):
        ax.axvspan(stim_start, stim_stop, color='k', alpha=.2)

    patches = [
        mpatches.Patch(color=color_dict[state], label=state)
        for state in color_dict
        if state in prepared["unique_states"]
    ]
    if patches:
        plt.sca(ax)
        plt.legend(handles=patches, fontsize=legend_font_size, fancybox=False, frameon=False)
    ax.set_xlim(prepared["timespan"])
    ax.set_ylim(-.4, 1.4)
    ax.set_xlabel("Time (seconds)", fontsize=axis_label_size)
    ax.tick_params(axis='x', which='major', labelsize=tick_label_size)
    ax.tick_params(axis='y', which='major', labelsize=tick_label_size)
    return ax


def colorblock_raster(event_df: pd.DataFrame, fig_name: str, plot_path: str, session_info: dict,
                      timespan: tuple = (0, np.inf), fig_format: str = 'png', plot_choices=False):
    """
    Create a raster plot from a single behavior session.
    """
    raster_figure, ax = plt.subplots(1, 1)
    raster_figure.set_figheight(10)
    raster_figure.set_figwidth(18)
    plot_colorblock_raster_on_ax(
        ax=ax,
        event_df=event_df,
        session_info=session_info,
        timespan=timespan,
        plot_choices=plot_choices,
        axis_label_size=40,
        tick_label_size=25,
        legend_font_size=20,
    )
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

