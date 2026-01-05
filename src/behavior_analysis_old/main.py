import session_overview
import session_analysis
import raster_plots
import fileIO
from pathlib import Path
import pickle as pkl
import pandas as pd


def process_sessions():
    """Prepare raw behavior data files for analysis."""

    # experiment_folder = Path('/home/matt/Documents/EXPERIMENTS/')
    experiment_folder = Path('C:/Users/mattc/EinsteinMed Dropbox/Matthew Chin/phd_data/remotework/EXPERIMENTS/')
    raw_data_path = experiment_folder / 'raw_behavior_data'
    processed_data_path = experiment_folder / 'processed_data'
    # plot_path = processed_data_path / 'figures'
    # plot_path = Path('../../reports/figures')

    current_mouse = 'CT010'
    current_date = '2025-08-15'
    sess_timestamp = '125111'
    sess_id = current_mouse + '_' + current_date
    sess_id_full = current_mouse + '_' + current_date + '_' + sess_timestamp

    session_folder = raw_data_path / sess_id_full
    session_log = '{}.log'.format(sess_id_full)
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)

    output_path = processed_data_path / current_mouse / sess_id_full

    # plot_path = Path('../../reports/figures/')
    # # raw_data_path = Path('../../data/raw/')
    # raw_data_path = Path('C:/Users/mattc/EinsteinMed Dropbox/Matthew Chin/LabComputerShare/RPi_transfer')
    # processed_data_path = Path('../../data/processed/')
    
    """Analyze the data from a single mouse session"""
    # F31_Apr2024 - used MF24 from 2023-10-04 to 2023-10-13. The behavior and analysis code has changed a lot since then.
    # mice = ['MF24'] * 7
    # dates = ['2023-10-04', '2023-10-05', '2023-10-06', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13']

    # dates = ['2024-08-19', '2024-08-21', '2024-08-22', '2024-08-23', '2024-08-26', '2024-08-28', '2024-08-29', '2024-08-30']
    dates = ['2024-09-06', '2024-09-07', '2024-09-10']
    mice = len(dates) * ['CT001']

    # mice = ['HD005', 'HD006', 'CT002', 'CT003', 'CT001', 'CT005']
    # dates = len(mice) * ['2024-09-10']

    # mice = ['CT010'] * 4
    # dates = ['2025-08-15', '2025-08-18', '2025-08-20', '2025-08-21']
    # timestamps = ['125111', '124026', '142917', '123406']
    mice = ['CT010'] 
    dates = ['2025-08-15']
    timestamps = ['125111']

    for m, d, t in zip(mice, dates, timestamps):
        sess_id = m + '_' + d
        sess_id_full = m + '_' + d + '_' + t
        output_path = processed_data_path / current_mouse / sess_id_full
        if not output_path.exists():
            output_path.mkdir()

        session_folder = raw_data_path / sess_id_full
        session_log = '{}.log'.format(sess_id_full)
        session_info_path = '{}_session_info.pkl'.format(sess_id_full)
        with open(session_folder / session_info_path, 'rb') as f:
            session_info = pkl.load(f)

        # event_df = fileIO.process_file(save_directory=output_path, file_path=session_folder / session_log)
        event_df = pd.read_csv(output_path / (sess_id_full + '.csv'), sep=',')

        trial_df = session_overview.make_trial_df(cleaned_data=event_df,
                                                  session_id=sess_id_full,
                                                  save_name=sess_id_full + '_trials',
                                                  output_path=output_path,
                                                  session_info=session_info)
        # with open(output_path / '{}_trials.pkl'.format(sess_id_full), 'rb') as f:
        #     trial_df = pkl.load(f)
        
        mouse_plot_path = output_path / 'raster_plots'
        if not mouse_plot_path.exists():
            mouse_plot_path.mkdir()

        raster_plots.generate_session_raster(event_df, sess_id_full + '_raster', mouse_plot_path)
        raster_plots.colorblock_raster(event_df, sess_id_full + '_lick_context_raster', mouse_plot_path, plot_choices=False)
        raster_plots.colorblock_raster(event_df, sess_id_full + '_choice_context_raster', mouse_plot_path, plot_choices=True)

        water = session_overview.total_water_delivery(event_df, session_info)
        session_performance, block_performance, choices_df = session_analysis.analyze_session(trial_df, mouse=m, date=d)

        # rewards, mean, std, sem = summarize_block_switches(block_performance, min_counts=0)
        slope, intercept, r_value, p_value = session_analysis.session_stats(dependent_var=block_performance['trials_to_correct'],
                                                           independent_var=block_performance['consecutive_rewards'])
        session_performance['slope'] = slope
        session_performance['intercept'] = intercept
        session_performance['r_value'] = r_value
        session_performance['p_value'] = p_value
        session_performance['n_switches'] = block_performance.shape[0]

        # session_analysis.save_analysis(session_performance, block_performance, choices_df, m, d, processed_data_path)
        session_analysis.save_analysis(session_performance, block_performance, choices_df, sess_id_full,
                                       session_save_path=output_path,
                                       overall_save_path=processed_data_path / current_mouse)


if __name__ == '__main__':
    process_sessions()

