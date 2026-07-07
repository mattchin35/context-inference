import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from icecream import ic
import irig_h_gpio as irig
from typing import Generator, List, Optional, Tuple, Literal
from datetime import datetime, timezone


def decode_irig_bits(irig_bits: np.array) -> List[Tuple[float, float]]:
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
    good_frames, good_frameix = [], []
    for frame, ix in zip(irig_frames, frame_ix):
        if len(frame) == 60:
            good_frames.append(frame)
            good_frameix.append(ix)
    irig_frames = good_frames
    frame_ix = good_frameix

    # irig.irig_h_to_datetime(irig_frames[0])
    posix_decoded = [irig.irig_h_to_posix(frame) for frame in irig_frames]
    datetime_decoded = [irig.irig_h_to_datetime(frame) for frame in irig_frames]

    # Handle invalid timecodes
    # decoded = [item for item in decoded if item is not None]

    print(f'List spliced! Splices: {len(posix_decoded)}')
    return posix_decoded, datetime_decoded, frame_ix


# file_path = Path.home() / 'Documents' / 'ephys_transfer' / 'treadmill_20251008' / 'CoolTerm Capture (Untitled_0) 2025-10-21 12-36-20-447.txt'
file_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/test_runs/irig_test_20260707/CoolTerm Capture (Untitled_0) 2026-07-07 12-06-51-676.txt')
# df = pd.read_csv(filepath, sep=';', header=None, on_bad_lines='skip', usecols=[1, 2, 4])

log_labels = ['Fdistance', 'Bdistance', 'dacval', 'runSpeed']
irig_labels = ['syncPinState']
log_dicts = []

pc_timestamp_present = True
lines = []
try:
    with open(file_path, 'r') as file:
        filesize = file_path.stat().st_size
        print(f"File size: {filesize} bytes")
        lines = [line.strip() for line in file]
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

for line in lines:
    if line == '':
        continue
    if pc_timestamp_present:
        pc_timestamp, serial_msg = line.split('\t')
        pc_date, pc_time = pc_timestamp.split(' ')
    else:
        serial_msg = line
        pc_date = ''
        pc_time = ''

    words = serial_msg.split(';')
    label, value = words[0].split(':')
    time_label, time_value = words[1].split(':')
    log_dicts.append(dict(pc_date=pc_date, pc_time=pc_time, label=label, value=value, time_label=time_label, time_value=time_value))

df = pd.DataFrame(log_dicts)

# df['value'] = df['value'] == 'LOW'  # flip values to make Low = True, High = False - not sure why raw data is inverted
df['time_value'] = df['time_value'].astype(float) / 1e6  # convert to seconds
irig_ix = df['label'] == 'syncPinState'
irig_cumulative_time = df[irig_ix]['time_value'].cumsum() - df[irig_ix]['time_value'].iloc[0]
# df['cumulative_time'] = (df['time_value'].cumsum() - df['time_value'].iloc[0]) #/ 1e6  # convert to seconds
cumulative_time = (df['time_value'].cumsum() - df['time_value'].iloc[0]) #/ 1e6  # convert to seconds
last_cumtime = 0
first_irig_ix = np.where(df['label'] == 'syncPinState')[0][0]
for ix in range(first_irig_ix, df.shape[0]):
    cumulative_time[ix] = last_cumtime + df['time_value'].iloc[ix]
    if df.iloc[ix]['label'] == 'syncPinState':
        last_cumtime = cumulative_time[ix]
cumulative_time[:first_irig_ix] = df['time_value'].iloc[:first_irig_ix]
# for ix in range(first_irig_ix):
    # cumulative_time[ix] = cumulative_time[first_irig_ix] - df['time_value'].iloc[ix]
df['cumulative_time'] = cumulative_time

# endtime = 30
# time_ix = df['cumulative_time'] < endtime
# plt.plot(df['cumulative_time'][time_ix & irig_ix], df['value'][time_ix & irig_ix], drawstyle='steps')
# plt.show()


irig_bit_ix = irig_ix & (df['value'] == 'HIGH')
irig_bit_lengths = np.round(df['time_value'][irig_ix].to_numpy(),1)
irig_bit_lengths[:-1] = irig_bit_lengths[1:]
irig_bit_lengths[-1] = np.nan

irig_bits = np.ones(irig_bit_lengths.shape[0], dtype=object) * np.nan
irig_bits[(irig_bit_lengths == .2)] = False
irig_bits[(irig_bit_lengths == .5)] = True
irig_bits[(irig_bit_lengths == .8)] = 'P'

irig_signal_values = df['value'][irig_ix].to_numpy()
irig_signal_values[irig_signal_values == 'HIGH'] = 1
irig_signal_values[irig_signal_values == 'LOW'] = 0
irig_signal_values = irig_signal_values.astype(bool)

irig_bits_fordf = np.ones(df.shape[0], dtype=object) * np.nan
irig_bits_fordf[irig_ix] = irig_bits
irig_bits_fordf[~irig_bit_ix] = np.nan
df['irig_bit'] = irig_bits_fordf
irig_bits = irig_bits[irig_signal_values]

t_decoded, datetime_decoded, frameix_decoded = decode_irig_bits(irig_bits)
unix_time = np.zeros_like(irig_bits, dtype=float)
unix_time[frameix_decoded] = t_decoded

ic(t_decoded)
irig_bit_cumtime = df['cumulative_time'][irig_bit_ix].to_numpy()
# pre_t = t_decoded[0] - (frameix_decoded[0] - np.array(range(frameix_decoded[0])))
unix_time[:frameix_decoded[0]] = t_decoded[0] - (frameix_decoded[0] - np.array(range(frameix_decoded[0])))
for ix, frame in enumerate(frameix_decoded[:-1]):
    unix_time[frameix_decoded[ix]:frameix_decoded[ix+1]] = t_decoded[ix] + np.array(range(frameix_decoded[ix+1]-frameix_decoded[ix]))
unix_time[frameix_decoded[-1]:] = t_decoded[-1] + np.array(range(irig_bits.size - frameix_decoded[-1]))

unix_time_fordf = np.ones(df.shape[0]) * np.nan
unix_time_fordf[irig_bit_ix] = unix_time

first_unix_ix = np.where(~np.isnan(unix_time_fordf))[0][0]
unix_time_fordf[:first_unix_ix] = unix_time[0] - (df['cumulative_time'].iloc[first_unix_ix] - df['cumulative_time'].iloc[:first_unix_ix][::-1])
last_unixtime = unix_time_fordf[first_unix_ix]
last_unixix = first_unix_ix
for ix in range(first_unix_ix, df.shape[0]):
    if irig_bit_ix[ix]:
        last_unixtime = unix_time_fordf[ix]
        last_unixix = ix
    else:
        unix_time_fordf[ix] = last_unixtime + (df['cumulative_time'].iloc[ix] - df['cumulative_time'].iloc[last_unixix])
df['unix_time'] = unix_time_fordf
ic(unix_time_fordf)
print([datetime.fromtimestamp(t) for t in unix_time_fordf[~np.isnan(unix_time_fordf)][:10]])
