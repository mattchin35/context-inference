import pandas as pd
import os
import numpy as np

"""
Utils for behavior data IO.
"""


def process_file(directory, file_path):
    os.chdir(directory)

    # Read the file into a DataFrame, skipping the first row
    df = pd.read_csv(file_path, sep=';', skiprows=1, on_bad_lines='skip', usecols=[1, 3])

    # Rename the columns
    df.columns = ['Time', 'Output']

    # Subtract 'exit_standby' time value from every element
    exit_standby_time = df.loc[df['Output'] == 'exit_standby', 'Time'].values[0]
    df['Time'] = df['Time'] - exit_standby_time

    # Remove rows with negative 'Time' values
    df = df[df['Time'] >= 0]

    # You can use the str.replace() function
    df = df.replace({'Output': r'.*ITI.*'}, {'Output': 'ITI'}, regex=True)

    # Get unique values from the 2nd column
    unique_values = df['Output'].unique().tolist()

    # Subset the DataFrame based on the elements of the 2nd column
    subsets = {}
    for value in unique_values:
        subsets[value] = df[df['Output'] == value]

    # Extract specific keys from the subsets dictionary
    specific_keys = ['exit_standby', 'left_entry', 'right_entry', 'pump1_reward_0', 'pump1_reward_3',
                     'pump2_reward_0', 'pump2_reward_3', 'ITI',
                     'enter_ContextA', 'enter_ContextB', 'enter_intercontext_interval',
                     'enter_ContextC1', 'enter_ContextC2']

    # Filter the subsets dictionary to include only specific keys
    specific_subsets = {key: subsets[key] for key in specific_keys if key in subsets}

    # Save the specific subsets dictionary to a new DataFrame
    df_new = pd.concat(specific_subsets.values())

    # Save the new DataFrame to a CSV file
    new_file_path = os.path.join(directory, 'cleaned_' + os.path.basename(file_path))

    if os.path.exists(new_file_path):
        print('file_already_exists')
    else:
        df_new.to_csv(new_file_path, index=False)
        print('cleaned_file_created')

    return specific_subsets


def load(mouse, date):
    """
    Load the behavior data for a particular mouse on a particular day.
    For the current mouse, returns a list of dictionaries of the data for each file read, each file representing a single session.
    """
    directories = ["../Mitch_behavior"]
    load_dict = {mouse: []}
    original_directory = os.getcwd()

    # Iterate through directories
    for directory in directories:
        # Reset the working directory at the start of each iteration
        os.chdir(original_directory)

        # Traverse directory, and for each file
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                # If it's a log file
                prefix = filename[:4]  # Get the first four characters of the filename
                if prefix in load_dict.keys():
                    if filename.endswith('.log') and date in filename and not 'cleaned' in filename:
                        print(f'Processing file: {filename}')
                        subdata = process_file(dirpath, filename)
                        load_dict[prefix].append(subdata)

    for key, value in load_dict.items():
        print(f'For {key}, processed {len(value)} files.')

    return load_dict