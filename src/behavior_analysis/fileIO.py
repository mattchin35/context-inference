import pandas as pd
import os
import re
from pathlib import Path
from typing import List, Any, Optional

"""
Functions for loading behavior data files.
"""


def process_file(save_directory: str, file_path: str, filter_events: Optional[List[str]] = []) -> pd.DataFrame:
    # Read the file into a DataFrame, skipping the first row
    df = pd.read_csv(file_path, sep=';', skiprows=1, on_bad_lines='skip', usecols=[1, 3])

    # Rename the columns
    df.columns = ['Time', 'Event']

    # Subtract 'exit_standby' time value from every element
    if 'exit_standby' in df['Event'].values:
        exit_standby_time = df.loc[df['Event'] == 'exit_standby', 'Time'].values[0]
    else:
        exit_standby_time = df['Time'].min()
    df['Time'] = df['Time'] - exit_standby_time

    # Remove rows with negative 'Time' values
    df = df[df['Time'] >= 0]

    # You can use the str.replace() function
    df = df.replace({'Event': r'.*ITI.*'}, {'Event': 'ITI'}, regex=True)

    # Get unique values from the 2nd column
    unique_events = df['Event'].unique().tolist()

    # Subset the DataFrame based on the elements of the 2nd column
    subsets = {}
    for e in unique_events:
        subsets[e] = df[df['Event'] == e]

    # Extract specific keys from the subsets dictionary
    # will need to adjust this as events change and for the infinite options of reward_X that are possible
    if filter_events:
        for e in unique_events:
            if re.fullmatch('pump.*', e):
                filter_events.append(e)

        subsets = {key: subsets[key] for key in filter_events if key in subsets}

    # specific_keys = ['exit_standby', 'left_entry', 'right_entry',
    #                  'ITI', 'enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval',
    #                  'enter_ContextC1', 'enter_ContextC2']

    # Filter the subsets dictionary to include only specific keys
    # These are now dataframes of times corresponding to each output type

    # Save the specific subsets dictionary to a new DataFrame
    df_new = pd.concat(subsets.values())
    df_new.columns = ['Time', 'Event']
    # df_new = df_new.set_index('Event')

    # Save the new DataFrame to a CSV file
    new_file_path = os.path.join(save_directory, 'cleaned_' + os.path.basename(file_path))
    df_new.to_csv(new_file_path, index=False)
    print('cleaned_file_created')
    return df_new


def find_files(search_dir: list, mouse, date) -> List[Path]:
    start_path = Path.cwd()
    match_files = []
    for d in search_dir:
        # Reset the working directory at the start of each iteration
        os.chdir(start_path)
        for root, dirs, files in os.walk(d):
            for f in files:
                if re.fullmatch('.*{}.*{}.*.log'.format(mouse, date), f):
                    match_files.append(Path(root) / f)

    os.chdir(start_path)
    return match_files


def separate_session_paths(files: List[Path]) -> List[Path]:
    raw_file, cleaned_file = None, None
    for i, filepath in enumerate(files):
        if re.fullmatch('cleaned_.*', filepath.name):
            cleaned_file = filepath.resolve()
            files.pop(i)

    raw_file = files[0].resolve()
    return [raw_file, cleaned_file]


def load(mouse: str, date: str, load_cleaned: bool = True):
    """
    Load the behavior data for a particular mouse on a particular day.
    a larger structure.
    """
    directories = ["../behavior_data"]
    files = find_files(directories, mouse, date)
    assert files, "File not found!!!"

    raw_file, clean_file = separate_session_paths(files)
    if load_cleaned and bool(clean_file):
        df = pd.read_csv(clean_file, sep=',')
    else:
        print('Processing raw file')
        df = process_file('../behavior_data', raw_file)

    return df


if __name__ == '__main__':
    current_mouse = 'MF23'
    current_date = '2023-08-18'
    df = load(current_mouse, current_date, load_cleaned=False)
    # print(df)