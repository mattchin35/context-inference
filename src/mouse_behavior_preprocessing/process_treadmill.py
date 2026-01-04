import pandas as pd
import numpy as np
import re
from typing import Optional
from pathlib import Path
import pickle as pkl
import collect_events
from collections import OrderedDict
import pynapple as nap


def read_treadmill_file(treadmill_file: str) -> list:
    try:
        with open(treadmill_file, 'r') as file:
            filesize = treadmill_file.stat().st_size
            print(f"File size: {filesize} bytes")
            lines = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Error: The file '{treadmill_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return lines


def interpret_treadmill_messages(msg_list: list) -> pd.DataFrame:
    valid_event_labels = ['Fdistance', 'Bdistance', 'distance', 'dacval', 'runSpeed', 'syncPinState']
    valid_time_labels = ['syncInterval', 'timeSincePinChange']
    log_dicts = []

    for ix, msg in enumerate(msg_list):
        # sometimes there are blank lines
        if msg == '':
            continue

        # some messages have the PC timestamp and some do not
        if re.match('.*\t.*', msg):
            pc_timestamp, serial_msg = msg.split('\t')
            pc_date, pc_time = pc_timestamp.split(' ')
        else:
            serial_msg = msg
            pc_date = ''
            pc_time = ''

        # some messages will not come through completely, and will be lost. Check for a full enough message, which has 2
        # label-value pairs separated by : within the pair and ; between/after the pairs
        if not re.match('.*:.*;.*:.*;', serial_msg):
            print('Message: "', serial_msg, '" was sent incomplete from line {}, skipping'.format(ix))
            continue

        event_msg, time_msg, _ = serial_msg.split(';')
        event_label, event_value = event_msg.split(':')
        time_label, time_value = time_msg.split(':')
        if event_label not in valid_event_labels:
            print(event_label, 'is not a valid event label from line {}, skipping'.format(ix))
            continue
        if not (event_value in ['HIGH', 'LOW'] or re.search(r'\d+', event_value)):
            print(event_value, 'is not a valid event value from line {}, skipping'.format(ix))
            continue
        if time_label not in valid_time_labels:
            print(time_label, 'is not a valid label from line {}, skipping'.format(ix))
            continue
        if not re.search(r'\d+', time_value):
            print(time_value, 'is not a valid time value from line {}, skipping'.format(ix))

        log_dicts.append(dict(pc_date=pc_date, pc_time=pc_time,
                              event_label=event_label, event_value=event_value,
                              time_label=time_label, time_value=time_value))

    df = pd.DataFrame(log_dicts)
    return df


def main():
    # hardcoding the input/output paths for data intake, it's too messy and variable to do it "automatically"
    treadmill_file = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/teensy/cleaned_treadmill_2025-12-16 15-31-28-903.txt')
    processed_data_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/processed')

    current_mouse = 'CT014'
    current_date = '2025-12-16'
    sess_timestamp = '153200'
    sess_id = current_mouse + '_' + current_date
    sess_id_full = current_mouse + '_' + current_date + '_' + sess_timestamp

    messages = read_treadmill_file(treadmill_file)
    treadmill_df = interpret_treadmill_messages(messages)

    p = processed_data_path / (sess_id_full + '_treadmill.csv')
    treadmill_df.to_csv(p, index=False)


if __name__ == '__main__':
    main()
