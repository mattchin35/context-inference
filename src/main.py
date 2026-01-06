# import session_overview
# import raster_plots
# import fileIO
import re
from behavior_analysis import raster_plots, session_analysis
from mouse_behavior_preprocessing import process_behavior_log
import pickle as pkl
from pathlib import Path
import pandas as pd
from dataclasses import dataclass


def preprocess_session_log(raw_behavior_folder: str, processed_data_path: str, sess_id_full: str):
    session_log = raw_behavior_folder / '{}.log'.format(sess_id_full)
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    event_df = process_behavior_log.process_file(save_directory=processed_data_path,
                                                         file_path=raw_behavior_folder / session_log)
    trial_df = process_behavior_log.make_trial_df(cleaned_data=event_df,
                                                         session_id=sess_id_full,
                                                         save_name=sess_id_full + '_trials',
                                                         output_path=processed_data_path,
                                                         session_info=session_info)
    water = process_behavior_log.calculate_water_delivery(event_df, session_info)
    return trial_df, water


def plot_session(raw_behavior_folder: str, processed_data_path: str, figure_path: str, sess_id_full: str):
    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)

    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    if not figure_path.exists():
        figure_path.mkdir()

    raster_plots.generate_session_raster(event_df, session_info=session_info, fig_name=sess_id_full + '_raster',
                            plot_path=figure_path, fig_format='png')
    raster_plots.colorblock_raster(event_df, sess_id_full + '_lick_context_raster', session_info=session_info, plot_path=figure_path,
                      plot_choices=False)
    raster_plots.colorblock_raster(event_df, sess_id_full + '_choice_context_raster', session_info=session_info,
                      plot_path=figure_path, plot_choices=True)


def analyze_session(trial_df: pd.DataFrame, sess_id_full: str, within_session_data_path: str, multi_session_save_path: str):
    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    assert match, "session id does not have correct format"

    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
    else:
        print("Double-check the session name!")
        return

    session_performance, block_performance, augmented_trial_df = session_analysis.analyze_session(trial_df, mouse=mouse, date=date)

    # rewards, mean, std, sem = summarize_block_switches(block_performance, min_counts=0)
    slope, intercept, r_value, p_value = session_analysis.session_stats(dependent_var=block_performance['trials_to_correct'],
                                                       independent_var=block_performance['consecutive_rewards'])
    session_performance['slope'] = slope
    session_performance['intercept'] = intercept
    session_performance['r_value'] = r_value
    session_performance['p_value'] = p_value
    session_performance['n_switches'] = block_performance.shape[0]
    session_analysis.save_analysis(session_performance, block_performance, augmented_trial_df,
                  sess_id=sess_id_full, session_save_path=within_session_data_path, overall_save_path=multi_session_save_path)


def main():
    """Analyze a single session from start to finish"""

    # prep data selection
    multi_session_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')

    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)

    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
        sess_id_abbreviated = mouse + '_' + date
    else:
        print("Double-check the session name!")
        return

    # trial_df, water = preprocess_session_log(raw_behavior_folder=raw_behavior_folder, processed_data_path=processed_data_path, sess_id_full=sess_id_full)

    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',')
    plot_session(raw_behavior_folder=raw_behavior_folder, processed_data_path=processed_data_path,
                 figure_path=figure_path, sess_id_full=sess_id_full)
    analyze_session(trial_df, sess_id_full,
                    within_session_data_path=processed_data_path, multi_session_save_path=multi_session_save_path)



if __name__ == '__main__':
    main()

