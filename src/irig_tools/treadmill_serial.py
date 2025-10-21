import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from icecream import ic
import irig_h_gpio as irig
from typing import Generator, List, Optional, Tuple, Literal
from datetime import datetime, timezone


file_path = Path.home() / 'Documents/ephys_transfer' / 'treadmill_20251008' / 'CoolTerm Capture (Untitled_0) 2025-10-21 12-36-20-447.txt'
# df = pd.read_csv(filepath, sep=';', header=None, on_bad_lines='skip', usecols=[1, 2, 4])

log_labels = ['Fdistance', 'Bdistance', 'dacval', 'runSpeed']
irig_labels = ['syncPinState']
log_dicts = []

pc_timestamp_present = False

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

df['value'] = df['value'] == 'LOW'  # flip values to make Low = True, High = False - not sure why raw data is inverted
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

endtime = 30
time_ix = df['cumulative_time'] < endtime
plt.plot(df['cumulative_time'][time_ix & irig_ix], df['value'][time_ix & irig_ix], drawstyle='steps')
plt.show()

"""
TODO:
- convert syncpinstates to irig bits 
- convert irig codes to unix timestamps
- obtain event times 
"""

# df[df['label']=='syncPinState']['value'] = ~df[df['label']=='syncPinState']['value']
irig_ix = (df['label'] == 'syncPinState') & (df['value'] == True)
# irig_bit_lengths = np.round(df[irig_ix]['time_value']/1e6,1)
irig_bit_lengths = np.round(df['time_value']/1e6,1)
irig_bits = np.ones(irig_ix.size, dtype=object) * np.nan


# assigning bits uses pulse lengths of HIGH states
irig_bits[(irig_bit_lengths == .2) & irig_ix] = False
irig_bits[(irig_bit_lengths == .5) & irig_ix] = True
irig_bits[(irig_bit_lengths == .8) & irig_ix] = 'P'
df['irig_bit'] = irig_bits
ic(df)

# IRIG_BIT = Literal[True,False,'P'] # type for IRIG-H bits
# irig_pulse_lengths = np.array([.2,.5,.8])
# tol = 2e3  # tolerance for matching times in milliseconds


def decode_irig_bits(irig_bits: np.array) -> List[Tuple[float, float]]:
    irig_frames = []
    tracking_start = None
    frame_ix = []

    for i in range(120):
        if irig_bits[i] == 'P' and irig_bits[i + 1] == 'P':
            tracking_start = i + 1
            frame_ix.append(i)
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

    # for i in range(starting_index, len(irig_bits) - 60, 60):
    #     spliced.append((irig_h_to_posix([t[0] for t in irig_bits[i:i+60]]), irig_bits[i][1]))

    # remove timecodes with bad lengths
    irig_frames = [frame for frame in irig_frames if len(frame) == 60]
    # irig.irig_h_to_datetime(irig_frames[0])
    decoded = [irig.irig_h_to_posix(frame) for frame in irig_frames]
    decoded = [irig.irig_h_to_datetime(frame) for frame in irig_frames]

    # Handle invalid timecodes
    decoded = [item for item in decoded if item is not None]

    print(f'List spliced! Splices: {len(decoded)}')
    return decoded, frame_ix

# for val in df[irig_ix]['time_value']:

t_decoded, frame_decoded = decode_irig_bits(irig_bits[irig_ix])
ic(t_decoded)
ic