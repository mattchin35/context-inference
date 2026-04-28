import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import src.behavior_analysis.fileIO as fileIO
from pathlib import Path
from icecream import ic
import src.behavior_analysis.raster_plots as rp
import src.behavior_analysis.collect_events as collect_events


"""
TODO:
- Represent licks as rewarded or unrewarded by color or size
- Alternately, don't represent rewards at all, just licks 

Color scheme
Right context: blue
Left context: green
Lick: black
"""


color_dict = {'right active': 'darkred', 'left active': 'blue'}
              # 'right': 'cyan', 'left': 'darkgreen', 0: 'cyan', 1: 'darkkhaki'}
state_dict = {0: 'right', 1: 'left'}


def generate_session_raster(session_df: pd.DataFrame, fig_name: str, plot_path: str, timespan: tuple = (0, np.inf),
                            fig_format: str = 'png') -> None:
    """
    Create a raster plot from a single behavior session.
    """
    cleaned_df = session_df[session_df['Event'] != 'exit_standby']
    if timespan[1] == np.inf:
        timespan = (0, np.amax(cleaned_df['Time']))

    event_list = ['left_entry', 'right_entry', 'enter_left_patch', 'enter_right_patch']
    event_labels = ['Left\nlick', 'Right\nlick', 'enter left patch', 'enter right patch']
    event_array = collect_events.generate_event_array(cleaned_df, event_list, timespan)
    end_time = np.amax([np.amax([np.amax(e) for e in event_array]), timespan[1]])

    states = ['left active'] * len(event_array[2]) + ['right active'] * len(event_array[3])
    state_times = np.concatenate(event_array[2:])
    state_sort_ix = np.argsort(state_times)
    states = np.array(states)[state_sort_ix]
    state_times = state_times[state_sort_ix]
    unique_states = np.unique(states).tolist()

    raster_figure, ax = rp.plot_event_raster(event_array[:2], event_labels[:2])

    # color blocks
    for i in range(states.size - 1):
        if states[i] in unique_states:
            plt.axvspan(state_times[i], state_times[i + 1], color=color_dict[states[i]], alpha=.2, label=states[i])
            unique_states.remove(states[i])
        else:
            plt.axvspan(state_times[i], state_times[i + 1], color=color_dict[states[i]], alpha=.2)
    plt.axvspan(state_times[-1], end_time, color=color_dict[states[-1]], alpha=.2)
    plt.xlim(timespan)

    # Add legend
    patches = [mpatches.Patch(color=color_dict[k], label=k) for k in color_dict.keys()]
    # plt.legend(handles=patches, fontsize=20, fancybox=False)
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

    savename = os.path.join(plot_path, '{}.svg'.format(fig_name, 'svg'))
    raster_figure.savefig(savename, format='svg')

    plt.close('all')
    print("Saved raster plot as {}".format(savename))


if __name__ == '__main__':
    plot_path = Path('../../reports/figures/F31_Apr2024')
    processed_path = Path('../../data/processed')

    """Analyze the data from a single mouse session"""
    mouse = 'MF24'  # 'MF03'
    date = '2023-10-13'  # '2023-10-03'
    sess_ID = mouse + '_' + date

    # data_path = Path('../../data/raw/Mitch_behavior')
    # df = fileIO.load_raw_data(sess_ID, data_path, processed_path)

    # data_path = Path('../../data/processed')
    df = fileIO.load_cleaned_data(sess_ID, processed_path)

    generate_session_raster(df, sess_ID + '_raster', plot_path, timespan=(0, 1000))
