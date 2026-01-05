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
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',')

    if not figure_path.exists():
        figure_path.mkdir()

    raster_plots.generate_session_raster(trial_df, sess_id_full + '_raster', figure_path)
    raster_plots.colorblock_raster(trial_df, sess_id_full + '_lick_context_raster', figure_path, plot_choices=False)
    raster_plots.colorblock_raster(trial_df, sess_id_full + '_choice_context_raster', figure_path, plot_choices=True)
    session_performance, block_performance, choices_df = session_analysis.analyze_session(trial_df, mouse=mouse, date=date)

    # rewards, mean, std, sem = summarize_block_switches(block_performance, min_counts=0)
    slope, intercept, r_value, p_value = session_analysis.session_stats(dependent_var=block_performance['trials_to_correct'],
                                                       independent_var=block_performance['consecutive_rewards'])
    session_performance['slope'] = slope
    session_performance['intercept'] = intercept
    session_performance['r_value'] = r_value
    session_performance['p_value'] = p_value
    session_performance['n_switches'] = block_performance.shape[0]

    # session_analysis.save_analysis(session_performance, block_performance, choices_df, m, d, processed_data_path)
    # session_analysis.save_analysis(session_performance, block_performance, choices_df, sess_id_full,
    #                                session_save_path=output_path,
    #                                overall_save_path=processed_data_path)


def main():
    raw_data_path = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/rpi/CT014_2025-12-16_153200')
    processed_data_path = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/processed')

    current_mouse = 'CT014'
    current_date = '2025-12-16'
    sess_timestamp = '153200'
    sess_id = current_mouse + '_' + current_date
    sess_id_full = current_mouse + '_' + current_date + '_' + sess_timestamp



if __name__ == '__main__':
    plot_session()

