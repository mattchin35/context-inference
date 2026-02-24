import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from icecream import ic
from pathlib import Path
from datetime import datetime, timezone
from src.irig_tools import irig_h_gpio as irig
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


def decode_irig_times(treadmill_df: pd.DataFrame, interpolate_irig=True) -> pd.DataFrame:
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

    if interpolate_irig:
        ## interpolation strategy - use irig bit indices as the known points and interpolate the unix times for all other points based on the cumulative times
        unix_time_fordf = np.ones(treadmill_df.shape[0]) * np.nan
        unix_time_fordf[irig_bit_ix] = unix_time

        # Interpolate Unix time from IRIG-anchored points in cumulative-time space.
        known_ix = np.where(~np.isnan(unix_time_fordf))[0]
        known_unix = unix_time_fordf[known_ix]
        cumulative_time = treadmill_df['cumulative_time'].to_numpy(dtype=float)
        known_cumulative = cumulative_time[known_ix]

        if known_ix.size < 2:
            raise ValueError("Need at least two IRIG anchor points to interpolate unix times.")

        unix_time_fordf = np.interp(cumulative_time, known_cumulative, known_unix)

        # np.interp clamps outside bounds; replace with linear extrapolation at both ends.
        left_mask = cumulative_time < known_cumulative[0]
        right_mask = cumulative_time > known_cumulative[-1]

        left_den = known_cumulative[1] - known_cumulative[0]
        right_den = known_cumulative[-1] - known_cumulative[-2]
        left_slope = (known_unix[1] - known_unix[0]) / left_den if left_den != 0 else 1.0
        right_slope = (known_unix[-1] - known_unix[-2]) / right_den if right_den != 0 else 1.0

        unix_time_fordf[left_mask] = known_unix[0] + (cumulative_time[left_mask] - known_cumulative[0]) * left_slope
        unix_time_fordf[right_mask] = known_unix[-1] + (cumulative_time[right_mask] - known_cumulative[-1]) * right_slope

        # Keep anchors exact after interpolation/extrapolation.
        unix_time_fordf[known_ix] = known_unix

    else:
        # get all the other unix times based on the cumsum strategy from before, adding cumulative times to the last known unix time at each irig bit
        unix_time_fordf = np.ones(treadmill_df.shape[0]) * np.nan
        unix_time_fordf[irig_bit_ix] = unix_time

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


def gather_runspeed(treadmill_df: pd.DataFrame, plot=False) -> pd.DataFrame:
    """
    Pull the run speeds from the treadmill dataframe, alongside the times they occur.
    """
    ix = treadmill_df['event_label'] == 'runSpeed'
    runspeed_df = treadmill_df.loc[ix, ['event_label', 'unix_time', 'event_value']].copy()
    runspeed_df = runspeed_df.sort_values('unix_time').reset_index(drop=True)
    runspeed_df['utc_datetime'] = [
        datetime.fromtimestamp(t) if pd.notna(t) else pd.NaT
        for t in pd.to_numeric(runspeed_df['unix_time'], errors='coerce')
    ]

    if plot:
        plt.figure(figsize=(10, 4))
        # plt.plot(runspeed_df['unix_time'], pd.to_numeric(runspeed_df['event_value'], errors='coerce'), drawstyle='steps-post')
        plt.plot(runspeed_df['utc_datetime'], pd.to_numeric(runspeed_df['event_value'], errors='coerce'), drawstyle='steps-post')
        plt.xlabel('Unix Time (s)')
        plt.ylabel('Run Speed')
        plt.title('Run Speed Over Time in Steps')
        plt.show()

    return runspeed_df


def gather_runspeed_with_buffer(treadmill_df: pd.DataFrame, plot=False) -> pd.DataFrame:
    """
    Pull the run speeds from the treadmill dataframe, alongside the times they occur. Adds a zero runspeed before
    after long periods at rest to allow for linear interpolation. But stepwise interpolation may be doable and better
    """
    ix = treadmill_df['event_label'] == 'runSpeed'
    runspeed_df = treadmill_df.loc[ix, ['event_label', 'unix_time', 'event_value']].copy()
    runspeed_df = runspeed_df.sort_values('unix_time').reset_index(drop=True)
    unchanged_runspeed_df = runspeed_df.copy(deep=True)  # for sanity checking later, to make sure we aren't changing any values with the insertion of zero points

    if runspeed_df.empty:
        return runspeed_df

    event_values = pd.to_numeric(runspeed_df['event_value'], errors='coerce')
    run_times = runspeed_df['unix_time'].to_numpy(dtype=float)
    zero_gap_threshold = 0.002  # 2 ms
    insert_offset = 0.001  # 1 ms

    inserted_rows = []
    for i in range(len(runspeed_df) - 1):
        if event_values.iloc[i] != 0:
            continue

        next_time = run_times[i + 1]
        current_time = run_times[i]
        if (next_time - current_time) > zero_gap_threshold:
            inserted_rows.append(
                {
                    'event_label': 'runSpeed',
                    'unix_time': next_time - insert_offset,
                    'event_value': 0,
                }
            )

    if inserted_rows:
        runspeed_df = pd.concat([runspeed_df, pd.DataFrame(inserted_rows)], ignore_index=True)
        runspeed_df = runspeed_df.sort_values('unix_time').reset_index(drop=True)

    runspeed_df['utc_datetime'] = [
        datetime.fromtimestamp(t) if pd.notna(t) else pd.NaT
        for t in pd.to_numeric(runspeed_df['unix_time'], errors='coerce')
    ]

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(unchanged_runspeed_df['unix_time'], pd.to_numeric(unchanged_runspeed_df['event_value'], errors='coerce'), label='Original', drawstyle='steps-post')
        plt.plot(runspeed_df['unix_time'], pd.to_numeric(runspeed_df['event_value'], errors='coerce'), label='With Inserted Zeros', drawstyle='steps-post')
        plt.xlabel('Unix Time (s)')
        plt.ylabel('Run Speed')
        plt.title('Run Speed Over Time with Inserted Zero Points')
        plt.legend()
        plt.show()

    return runspeed_df


def main():
    # hardcoding the input/output paths for data intake, it's too messy and variable to do it "automatically"
    treadmill_file = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/teensy/cleaned_treadmill_2025-12-16 15-31-28-903.txt')
    processed_data_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/processed')

    current_mouse = 'CT014'
    current_date = '2025-12-16'
    sess_timestamp = '153200'
    sess_id = current_mouse + '_' + current_date
    sess_id_full = current_mouse + '_' + current_date + '_' + sess_timestamp

    # messages = read_treadmill_file(treadmill_file)
    # treadmill_df = interpret_treadmill_messages(messages)
    # treadmill_df = calculate_cumuluative_times(treadmill_df)
    # treadmill_df = decode_irig_times(treadmill_df)

    p = processed_data_path / (sess_id_full + '_treadmill.csv')
    # treadmill_df.to_csv(p, index=False)

    treadmill_df = pd.read_csv(p)
    runspeed_df = gather_runspeed(treadmill_df, plot=True)


if __name__ == '__main__':
    main()

# after running the code, do a couple of sanity checks. If your treadmill file comes with datetime stamps included,
# the IRIG timecodes should correspond to 4 hours ahead on the same day (i.e. US EST recordings are 4 hours ahead of UTC).
