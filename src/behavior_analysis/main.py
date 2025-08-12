import session_overview
import session_analysis
import raster_plots
import fileIO
from pathlib import Path


def process_sessions():
    """Prepare raw behavior data files for analysis."""

    plot_path = Path('../../reports/figures/')
    # raw_data_path = Path('../../data/raw/')
    raw_data_path = Path('C:/Users/mattc/EinsteinMed Dropbox/Matthew Chin/LabComputerShare/RPi_transfer')
    processed_data_path = Path('../../data/processed/')
    
    """Analyze the data from a single mouse session"""
    # F31_Apr2024 - used MF24 from 2023-10-04 to 2023-10-13. The behavior and analysis code has changed a lot since then.
    # mice = ['MF24'] * 7
    # dates = ['2023-10-04', '2023-10-05', '2023-10-06', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13']

    # dates = ['2024-08-19', '2024-08-21', '2024-08-22', '2024-08-23', '2024-08-26', '2024-08-28', '2024-08-29', '2024-08-30']
    dates = ['2024-09-06', '2024-09-07', '2024-09-10']
    mice = len(dates) * ['CT001']

    # mice = ['HD005', 'HD006', 'CT002', 'CT003', 'CT001', 'CT005']
    # dates = len(mice) * ['2024-09-10']

    # mice = ['TM001']
    # dates = ['2024-01-01']

    for m, d in zip(mice, dates):
        sess_id = m + '_' + d
        # cleaned_data = fileIO.load_raw_data(sess_id, raw_data_path, processed_data_path / 'cleaned_sessions')
        cleaned_data = fileIO.load_cleaned_data(sess_id, processed_data_path / 'cleaned_sessions')

        # generate_session_raster(df, sess_ID + '_raster', plot_path)
        mouse_plot_path = plot_path / m / 'raster_plots'
        if not mouse_plot_path.exists():
            mouse_plot_path.mkdir()

        # raster_plots.generate_session_raster(cleaned_data, sess_id + '_raster', mouse_plot_path)
        # raster_plots.colorblock_raster(cleaned_data, sess_id + '_context_raster', mouse_plot_path, plot_choices=False)
        # raster_plots.colorblock_raster(cleaned_data, sess_id + '_choice_context_raster', mouse_plot_path, plot_choices=True)

        event_df = session_overview.make_event_df(cleaned_data=cleaned_data, session_id=sess_id,
                                                  save_name='{}_events'.format(sess_id),
                                                  output_path=processed_data_path / m)

        water = session_overview.total_water_delivery(cleaned_data)

        session_performance, block_performance, choices_df = session_analysis.analyze_session(event_df, mouse=m, date=d)

        # rewards, mean, std, sem = summarize_block_switches(block_performance, min_counts=0)
        slope, intercept, r_value, p_value = session_analysis.session_stats(dependent_var=block_performance['trials_to_correct'],
                                                           independent_var=block_performance['consecutive_rewards'])
        session_performance['slope'] = slope
        session_performance['intercept'] = intercept
        session_performance['r_value'] = r_value
        session_performance['p_value'] = p_value
        session_performance['n_switches'] = block_performance.shape[0]

        session_analysis.save_analysis(session_performance, block_performance, choices_df, m, d, processed_data_path)


if __name__ == '__main__':
    process_sessions()
