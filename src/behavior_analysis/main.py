import session_overview
import fileIO
from pathlib import Path


def process_sessions():
    """Prepare raw behavior data files for analysis."""

    plot_path = Path('../../reports/figures/')
    raw_data_path = Path('../../data/raw/')
    # raw_data_path = Path('C:/Users/mattc/EinsteinMed/Dropbox/Matthew Chin/LabComputerShare/RPi_transfer')
    processed_data_path = Path('../../data/processed/')
    
    """Analyze the data from a single mouse session"""
    # mice = ['MF24'] * 7
    # dates = ['2023-10-04', '2023-10-05', '2023-10-06', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13']

    mice = ['CT002', 'CT002']
    dates = ['2024-08-23', '2024-08-26']
    sess_ids = [m + '_' + d for m, d in zip(mice, dates)]

    for id in sess_ids:
        cleaned_data = fileIO.load_raw_data(id, raw_data_path, processed_data_path / 'cleaned_sessions')
        # cleaned_data = fileIO.load_cleaned_data(id, processed_data_path / 'cleaned_sessions')
        event_df = session_overview.make_event_df(cleaned_data=cleaned_data, session_id=id,
                                                  save_name=id + '_events',
                                                  output_path=processed_data_path / 'session_events')
        # event_df = session_overview.make_event_df(cleaned_data=cleaned_data, session_id=session_id,
        #                                           save_name=session_id + '_performance',
        #                                           output_path=processed_data_path)
        water = session_overview.total_water_delivery(cleaned_data)


def main():
    process_sessions()


if __name__ == '__main__':
    main()
