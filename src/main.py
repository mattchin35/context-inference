# import session_overview
# import session_analysis
# import raster_plots
# import fileIO
import behavior_analysis
import mouse_behavior_preprocessing
import pickle as pkl
from pathlib import Path
import pandas as pd


def process_behavior_log(raw_behavior_folder: str, processed_data_path: str, sess_id_full: str):
    session_log = raw_behavior_folder / '{}.log'.format(sess_id_full)
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    event_df = mouse_behavior_preprocessing.process_file(save_directory=processed_data_path,
                                                         file_path=raw_behavior_folder / session_log)
    trial_df = mouse_behavior_preprocessing.make_trial_df(cleaned_data=event_df,
                                                         session_id=sess_id_full,
                                                         save_name=sess_id_full + '_trials',
                                                         output_path=processed_data_path,
                                                         session_info=session_info)
    water = mouse_behavior_preprocessing.calculate_water_delivery(event_df, session_info)
    return trial_df, water


def plot_session(raw_behavior_folder: str, processed_data_path: str, figure_path: str, sess_id_full: str):
    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',')
    session_log = raw_behavior_folder / '{}.log'.format(sess_id_full)
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)

    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    if not figure_path.exists():
        figure_path.mkdir()

    behavior_analysis.generate_session_raster(event_df, session_info=session_info, fig_name=sess_id_full + '_raster',
                            plot_path=figure_path, fig_format='png')
    # colorblock_raster(trial_df, sess_id_full + '_lick_context_raster', figure_path, plot_choices=False)
    # colorblock_raster(trial_df, sess_id_full + '_choice_context_raster', figure_path, plot_choices=True)


def main():
    data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    raw_behavior_folder = data_home / 'rpi/CT014_2025-12-16_153200'
    processed_data_path = data_home / 'processed'
    figure_path = data_home / 'figures'

    current_mouse = 'CT014'
    current_date = '2025-12-16'
    sess_timestamp = '153200'
    sess_id = current_mouse + '_' + current_date
    sess_id_full = current_mouse + '_' + current_date + '_' + sess_timestamp
    process_behavior_log(raw_behavior_folder=raw_behavior_folder, processed_data_path=processed_data_path, sess_id_full=sess_id_full)
    plot_session(raw_behavior_folder=raw_behavior_folder, processed_data_path=processed_data_path,
                 figure_path=figure_path, sess_id_full=sess_id_full)


if __name__ == '__main__':
    main()

