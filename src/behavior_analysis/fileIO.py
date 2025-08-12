import pandas as pd
import os
import re
from pathlib import Path
from typing import List, Any, Optional

"""
Functions for loading behavior and preprocessing behavior data files.
"""


def process_file(save_directory: Path, file_path: str, filter_events: Optional[List[str]] = []) -> pd.DataFrame:
    # Read the file into a DataFrame, skipping the first row
    # df = pd.read_csv(file_path, sep=';', skiprows=1, on_bad_lines='skip', usecols=[1, 3])
    df = pd.read_csv(file_path, sep=';', header=None, on_bad_lines='skip', usecols=[1, 3])
    # df = pd.read_csv(file_path, sep=';', header=None, on_bad_lines='skip', usecols=[1, 3, 4])

    # Rename the columns
    df.columns = ['Time', 'Event']
    # df.columns = ['Time', 'Event', 'Reward']

    # Subtract 'exit_standby' time value from every element
    if 'exit_standby' in df['Event'].values:
        exit_standby_time = df.loc[df['Event'] == 'exit_standby', 'Time'].values[0]
    else:
        exit_standby_time = df['Time'].min()
    df['Time'] = df['Time'] - exit_standby_time

    # Remove any rows with negative 'Time' values
    df = df[df['Time'] >= 0]

    # You can use the str.replace() function to replace messy event names. Ideally, just avoid using messy names.
    # df = df.replace({'Event': r'.*ITI.*'}, {'Event': 'ITI'}, regex=True)

    # Get unique values from the 2nd column
    unique_events = df['Event'].unique().tolist()

    # Subset the DataFrame based on the elements of the 2nd column
    subsets = {}
    for e in unique_events:
        subsets[e] = df[df['Event'] == e]

    # Extract specific keys from the subsets dictionary
    # will need to adjust this as events change and for the infinite options of reward_X that are possible
    # if filter_events:
    #     for e in unique_events:
    #         if re.fullmatch('pump.*', e):
    #             filter_events.append(e)
    #
    #     subsets = {key: subsets[key] for key in filter_events if key in subsets}

    # specific_keys = ['exit_standby', 'left_entry', 'right_entry',
    #                  'ITI', 'enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval',
    #                  'enter_ContextC1', 'enter_ContextC2']

    # Filter the subsets dictionary to include only specific keys
    # These are now dataframes of times corresponding to each output type

    # Save the specific subsets dictionary to a new DataFrame
    df_new = pd.concat(subsets.values())
    df_new.columns = ['Time', 'Event']
    # df_new.columns = ['Time', 'Event', 'Reward']
    # df_new = df_new.set_index('Event')

    # Save the new DataFrame to a CSV file
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # new_file_path = os.path.join(save_directory, 'cleaned_' + os.path.basename(file_path))
    new_file_path = save_directory / ('cleaned_' + Path(file_path).name)
    df_new.to_csv(new_file_path, index=False)
    # df.to_csv(new_file_path, index=False)
    print('cleaned_file_created')
    return df_new


def find_files(search_dir: Path, session_ID: str) -> List[Path]:
    # match_files = list(search_dir.glob('*{}*.log'.format(session_ID)))  # non-recursive search
    match_files = list(search_dir.glob('**/*{}*.log'.format(session_ID)))  # recursive search
    return match_files


def load_raw_data(session_ID: str, data_path: Path, save_path: Path) -> pd.DataFrame:
    file = find_session(session_ID, data_path)
    print('Processing raw file')
    df = process_file(save_path, file_path=file)
    return df


def load_cleaned_data(session_id: str, data_path: Path):
    file = find_session(session_id, data_path)
    print('Loading cleaned file')
    df = pd.read_csv(file, sep=',')
    return df


def find_session(session_id: str, data_path: Path) -> Path:
    file = find_files(data_path, session_id)
    assert file, "File for {} not found!!!".format(session_id)
    assert len(file) == 1, "Multiple files for {} found!!!".format(session_id)
    file = file[0]
    return file


if __name__ == '__main__':
    mouse = 'MF23'
    date = '2023-08-18'
    sess_ID = mouse + '_' + date

    data_path = Path('../../data/raw/Mitch_behavior')
    save_path = Path('../../data/processed')
    df = load_raw_data(sess_ID, data_path, save_path)
    print(df)
