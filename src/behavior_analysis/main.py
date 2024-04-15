import session_overview
import fileIO
from pathlib import Path

"""Run these files to prepare raw behavior data files for analysis."""


def main_single_session():
    plot_path = Path('../../reports/figures')
    raw_data_path = Path('../../data/raw/Mitch_behavior')
    processed_data_path = Path('../../data/processed')

    """Analyze the data from a single mouse session"""
    current_mouse = 'MF03'  # 'MF23'
    current_date = '2023-10-03'  # '2023-08-18'
    sess_ID = current_mouse + '_' + current_date

    cleaned_data = fileIO.load_raw_data(sess_ID, raw_data_path, processed_data_path)
    # cleaned_data = fileIO.load_cleaned_data(sess_ID, processed_data_path)
    event_df = session_overview.make_event_df(cleaned_data=cleaned_data, session_id=sess_ID,
                     save_name=sess_ID + '_performance', output_path=processed_data_path)


def main_multiple_sessions():
    plot_path = Path('../../reports/figures/Mitch_behavior')
    raw_data_path = Path('../../data/raw/Mitch_behavior')
    processed_data_path = Path('../../data/processed/Mitch_behavior')

    """Analyze the data from a single mouse session"""
    mice = ['MF24'] * 7
    dates = ['2023-10-04', '2023-10-05', '2023-10-06', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13']
    sess_ids = [m + '_' + d for m, d in zip(mice, dates)]

    for id in sess_ids:
        cleaned_data = fileIO.load_raw_data(id, raw_data_path, processed_data_path)
        # cleaned_data = fileIO.load_cleaned_data(id, processed_data_path)
        event_df = session_overview.make_event_df(cleaned_data=cleaned_data, session_id=id,
                                                  save_name=id + '_events', output_path=processed_data_path)


if __name__ == '__main__':
    # main_single_session()
    main_multiple_sessions()
