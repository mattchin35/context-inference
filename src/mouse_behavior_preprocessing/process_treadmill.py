import pandas as pd
import numpy as np
import re
from icecream import ic
from pathlib import Path
from datetime import datetime, timezone
from src.irig_tools import irig_h_gpio as irig
import pynapple as nap
from typing import Optional, Any


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


def decode_irig_bits(irig_bits: np.array) -> list[tuple[float, float]]:
    irig_frames = []
    tracking_start = None
    frame_ix = []

    for i in range(120):
        if irig_bits[i] == 'P' and irig_bits[i + 1] == 'P':
            tracking_start = i + 1
            frame_ix.append(tracking_start)
            break

    if tracking_start is None:
        raise ValueError('No starting position marker found.')

    frame = []
    for i in range(tracking_start, irig_bits.size - 1):
        frame.append(irig_bits[i])
        if irig_bits[i] == 'P' and irig_bits[i + 1] == 'P':
            irig_frames.append(frame)
            frame_ix.append(i+1)
            frame = []

    if len(frame_ix) > len(irig_frames):
        frame_ix = frame_ix[:-1]

    # remove timecodes with bad lengths
    # irig_frames = [frame for frame in irig_frames if len(frame) == 60]
    good_frames, good_frame_ix = [], []
    for frame, ix in zip(irig_frames, frame_ix):
        if len(frame) == 60:
            good_frames.append(frame)
            good_frame_ix.append(ix)
    irig_frames = good_frames
    frame_ix = good_frame_ix

    # irig.irig_h_to_datetime(irig_frames[0])
    posix_decoded = [irig.irig_h_to_posix(frame) for frame in irig_frames]
    datetime_decoded = [irig.irig_h_to_datetime(frame) for frame in irig_frames]

    # Handle invalid timecodes
    # decoded = [item for item in decoded if item is not None]

    print(f'List spliced! Splices: {len(posix_decoded)}')
    return posix_decoded, datetime_decoded, frame_ix


def calculate_cumuluative_times(treadmill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the cumulative time for each message based on the Teensy-encoded interval times.
    Not synced to Unix time in this function.
    """
    treadmill_df['time_value'] = treadmill_df['time_value'].astype(float) / 1e6  # convert to seconds
    last_irigtime = 0
    cumulative_time = np.zeros_like(treadmill_df['time_value'])
    first_irig_ix = np.where(treadmill_df['event_label'] == 'syncPinState')[0][0]
    for ix in range(first_irig_ix, treadmill_df.shape[0]):
        cumulative_time[ix] = last_irigtime + treadmill_df['time_value'].iloc[ix]
        if treadmill_df.iloc[ix]['event_label'] == 'syncPinState':
            last_irigtime = cumulative_time[ix]
    cumulative_time[:first_irig_ix] = treadmill_df['time_value'].iloc[:first_irig_ix]
    treadmill_df['cumulative_time'] = cumulative_time
    return treadmill_df


def decode_irig_times(treadmill_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the Unix/UTC times for all events, combining the IRIG signals with the Teensy encoded times.
    Use the IRIG pulses to get the pulse times and interpolate the Teensy signals to the corresponding values,
    as with an ephys sync line.
    """
    irig_ix = treadmill_df['event_label'] == 'syncPinState'
    irig_bit_ix = irig_ix & (treadmill_df['event_value'] == 'HIGH')

    # syncInterval tells you the time since the last IRIG pulse, not the length of the pulse corresponding to the current time
    irig_bit_lengths = np.round(treadmill_df['time_value'][irig_ix].to_numpy(), 1)
    irig_bit_lengths[:-1] = irig_bit_lengths[1:]
    irig_bit_lengths[-1] = np.nan # the last bit doesn't have a recorded pulse length - the last pulse length finishes after recording stops!

    irig_bits = np.ones(irig_bit_lengths.shape[0], dtype=object) * np.nan
    irig_bits[(irig_bit_lengths == .2)] = False
    irig_bits[(irig_bit_lengths == .5)] = True
    irig_bits[(irig_bit_lengths == .8)] = 'P'

    irig_signal_values = treadmill_df['event_value'][irig_ix].to_numpy()
    irig_signal_values[irig_signal_values == 'HIGH'] = 1
    irig_signal_values[irig_signal_values == 'LOW'] = 0
    irig_signal_values = irig_signal_values.astype(bool)

    irig_bits_fordf = np.ones(treadmill_df.shape[0], dtype=object) * np.nan
    irig_bits_fordf[irig_ix] = irig_bits
    irig_bits_fordf[~irig_bit_ix] = np.nan
    treadmill_df['irig_bit'] = irig_bits_fordf
    irig_bits = irig_bits[irig_signal_values]

    posix_decoded, datetime_decoded, frameix_decoded = decode_irig_bits(irig_bits)
    unix_time = np.zeros_like(irig_bits, dtype=float)
    unix_time[frameix_decoded] = posix_decoded
    ic(posix_decoded)
    ic(datetime_decoded)

    # get the unix times for every time corresponding to an irig pulse onset/offset
    # can either use integer increases based on precise-second IRIG pulse assumption or Teensy-based interval messages

    # Integer assumption
    unix_time[:frameix_decoded[0]] = posix_decoded[0] - np.arange(frameix_decoded[0], 0, -1) # before first timecode
    for ix, frame in enumerate(frameix_decoded[:-1]):
        unix_time[frameix_decoded[ix]:frameix_decoded[ix + 1]] = posix_decoded[ix] + np.arange(frameix_decoded[ix + 1] - frameix_decoded[ix]) # should be 0-59
    unix_time[frameix_decoded[-1]:] = posix_decoded[-1] + np.arange(irig_bits.size - frameix_decoded[-1])

    # Teensy-based intervals - would need to call dataframe irig_ix indices, not frame_decoded intervals
    # unix_time[:frameix_decoded[0]] = posix_decoded[0] - (treadmill_df['cumulative_time'].iloc[frameix_decoded[0]] - treadmill_df['cumulative_time'].iloc[0:frameix_decoded[0]])
    # for ix, frame in enumerate(frameix_decoded[:-1]):
    #     unix_time[frameix_decoded[ix]:frameix_decoded[ix + 1]] = unix_time[frameix_decoded[ix]] + treadmill_df['cumulative_time'].iloc[frameix_decoded[ix]:frameix_decoded[ix + 1]] - treadmill_df['cumulative_time'].iloc[frameix_decoded[ix]]
    # unix_time[frameix_decoded[-1]:] = posix_decoded[-1] + treadmill_df['cumulative_time'].iloc[frameix_decoded[-1]:] - treadmill_df['cumulative_time'].iloc[frameix_decoded[-1]]

    unix_time_fordf = np.ones(treadmill_df.shape[0]) * np.nan
    unix_time_fordf[irig_bit_ix] = unix_time

    # get all the other unix times based on the cumsum strategy from before. I should convert this to interpolation
    first_unix_ix = np.where(~np.isnan(unix_time_fordf))[0][0]
    unix_time_fordf[:first_unix_ix] = unix_time[0] - (
                treadmill_df['cumulative_time'].iloc[first_unix_ix] - treadmill_df['cumulative_time'].iloc[:first_unix_ix][::-1])
    last_unixtime = unix_time_fordf[first_unix_ix]
    for ix in range(first_unix_ix, treadmill_df.shape[0]):
        if irig_bit_ix[ix]:
            last_unixtime = unix_time_fordf[ix]
        else:
            unix_time_fordf[ix] = last_unixtime + treadmill_df['time_value'].iloc[ix]

    datetime_for_df = [datetime.fromtimestamp(t) for t in unix_time_fordf[~np.isnan(unix_time_fordf)]]
    treadmill_df['unix_time'] = unix_time_fordf
    treadmill_df['datetime'] = datetime_for_df
    return treadmill_df


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
    treadmill_df = calculate_cumuluative_times(treadmill_df)
    treadmill_df = decode_irig_times(treadmill_df)

    p = processed_data_path / (sess_id_full + '_treadmill.csv')
    treadmill_df.to_csv(p, index=False)


if __name__ == '__main__':
    main()

# after running the code, do a couple of sanity checks. If your treadmill file comes with datetime stamps included,
# the IRIG timecodes should correspond to 4 hours ahead on the same day (i.e. US EST recordings are 4 hours ahead of UTC).
