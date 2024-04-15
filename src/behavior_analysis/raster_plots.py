import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import fileIO
from typing import Iterable
from pathlib import Path
from icecream import ic
import collect_events


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
    event_labels = collect_events.make_event_labels(event_list)
    event_array = collect_events.generate_event_array(cleaned_df, event_list, timespan)

    raster_figure = plot_event_raster(event_array, event_labels)

    # plt.show()
    savename = os.path.join(plot_path, '{}.{}'.format(fig_name, fig_format))
    raster_figure.savefig(savename, format=fig_format)
    plt.close('all')
    print("Saved raster plot as {}".format(savename))


if __name__ == '__main__':
    # data_path = Path('../../data/raw/Mitch_behavior')
    data_path = Path('../../data/processed')
    plot_path = Path('../../reports/figures/F31_Apr2024')

    """Analyze the data from a single mouse session"""
    mouse = 'MF03'  # 'MF23'
    date = '2023-10-03'  # '2023-08-18'
    sess_ID = mouse + '_' + date

    # df = fileIO.load(sess_ID, data_path, load_cleaned=False)
    df = fileIO.load_cleaned_data(sess_ID, data_path)
    print(df)
    generate_session_raster(df, sess_ID + '_raster', plot_path)


