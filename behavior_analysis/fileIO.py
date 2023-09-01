import pandas as pd
import os
import numpy as np
import re

"""
Utils for behavior data IO.
"""


def process_file(directory, file_path):
    os.chdir(directory)

    # Read the file into a DataFrame, skipping the first row
    df = pd.read_csv(file_path, sep=';', skiprows=1, on_bad_lines='skip', usecols=[1, 3])

    # Rename the columns
    df.columns = ['Time', 'Event']

    # Subtract 'exit_standby' time value from every element
    exit_standby_time = df.loc[df['Event'] == 'exit_standby', 'Time'].values[0]
    df['Time'] = df['Time'] - exit_standby_time

    # Remove rows with negative 'Time' values
    df = df[df['Time'] >= 0]

    # You can use the str.replace() function
    df = df.replace({'Event': r'.*ITI.*'}, {'Event': 'ITI'}, regex=True)

    # Get unique values from the 2nd column
    unique_values = df['Event'].unique().tolist()

    # Subset the DataFrame based on the elements of the 2nd column
    subsets = {}
    for value in unique_values:
        subsets[value] = df[df['Event'] == value]

    # Extract specific keys from the subsets dictionary
    # will need to adjust this for the infinite options of reward_X that are possible in the settings
    """
    TODO - do a regex search for any pump1_ or pump2_ keys
    """

    specific_keys = ['exit_standby', 'left_entry', 'right_entry',
                     'ITI', 'enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval',
                     'enter_ContextC1', 'enter_ContextC2']
    # 'pump1_reward_0', 'pump1_reward_3',
    # 'pump2_reward_0', 'pump2_reward_3',

    for key in unique_values:
        if re.fullmatch('pump.*', key):
            specific_keys.append(key)

    # Filter the subsets dictionary to include only specific keys
    # These are now dataframes of times corresponding to each output type
    specific_subsets = {key: subsets[key] for key in specific_keys if key in subsets}

    # Save the specific subsets dictionary to a new DataFrame
    df_new = pd.concat(specific_subsets.values())
    df_new.columns = ['Time', 'Event']
    # df_new = df_new.set_index('Event')

    # Save the new DataFrame to a CSV file
    new_file_path = os.path.join(directory, 'cleaned_' + os.path.basename(file_path))
    df_new.to_csv(new_file_path, index=False)
    print('cleaned_file_created')

    # Turn the new dataframe into a form to work with

    return df_new


def load(mouse, date, reprocess=False):
    """
    Load the behavior data for a particular mouse on a particular day.
    a larger structure.
    """
    directories = ["../behavior_data"]
    load_dict = {mouse: []}
    original_directory = os.getcwd()

    files = []
    found = False
    # Get all mouse+date-matching filenames
    for directory in directories:
        # Reset the working directory at the start of each iteration
        os.chdir(original_directory)
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                # # If it's a log file
                # prefix = filename[:4]  # Get the first four characters of the filename
                # if prefix in load_dict.keys():
                if re.fullmatch('.*{}.*{}.*.log'.format(mouse, date), filename):
                    files.append((root, filename))
                    found = True

    assert found is True, "File not found!!!"

    # first check for a cleaned file
    found = False
    for i, (root, fname) in enumerate(files):
        if re.fullmatch('cleaned_.*', fname):
            if reprocess is False:  # supposedly I should make a program just to load the clean csv and call that from elsewhere
                found = True
                df = pd.read_csv(os.path.join(root, fname), sep=',')
                break
            else:
                files.pop(i)
                break

    # then check for the unclean file
    if not found:
    # if found:
        root, fname = files[0]  # there should only be one file if a "cleaned" file wasn't found
        df = pd.read_csv(os.path.join(root, fname), sep=';', skiprows=1, on_bad_lines='skip', usecols=[1, 3])

        print(f'Processing file: {fname}')
        df = process_file(root, fname)

    for key, value in load_dict.items():
        print(f'For {key}, processed {len(value)} files.')

    return df