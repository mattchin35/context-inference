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

    # remove timecodes with bad lengths
    irig_frames = [frame for frame in irig_frames if len(frame) == 60]
    print('irig frame 0:',
          irig_frames[0])
    ic(irig.irig_h_to_datetime(irig_frames[0]))
    decoded = [irig.irig_h_to_posix(frame) for frame in irig_frames]

    # Handle invalid timecodes
    decoded = [item for item in decoded if item is not None]

    print(f'List spliced! Splices: {len(decoded)}')
    return decoded, frame_ix

# file_path = Path.home() / 'Documents/EXPERIMENTS/raw_behavior_data' / 'test-mouse_2025-10-14_205950/test-mouse_2025-10-14_205950_cam0_flipper.csv'
file_path = Path('/home/matt/Documents/irig_mods/irig_signal_timestamps_2025-10-22_19-28-12.csv')
df = pd.read_csv(file_path, sep=',', on_bad_lines='skip', usecols=[0,1])


# df['cumulative_time'] = df['time.time()'] - df['time.time()'].iloc[0]
# df['interval_time'] = df['time.time()'].diff().fillna(0)#.round(1)
df['cumulative_time'] = df['Signal times'] - df['Signal times'].iloc[0]
df['interval_time'] = df['Signal times'].diff().fillna(0)#.round(1)
irig_ix = np.zeros(df.shape[0], dtype=bool)
irig_pulselen = np.ones(df.shape[0]) * np.nan
for ix in range(0, df.iloc[:-1].shape[0]):
    # curstate = df['Input State'].iloc[ix]
    # nextstate = df['Input State'].iloc[ix+1]
    curstate = df['Signal values'].iloc[ix]
    nextstate = df['Signal values'].iloc[ix + 1]
    if curstate == 1 and nextstate == 0:
        irig_pulselen[ix] = df['interval_time'].iloc[ix+1]
        irig_ix[ix] = True
# for ix in range(1, df['Input State'].iloc[1:].shape[0]):
#     curstate = df['Input State'].iloc[ix]
    # prevstate = df['Input State'].iloc[ix-1]
    # if curstate == 0 and prevstate == 1:
        # irig_bits.append(df['interval_time'].iloc[ix])
        # irig_ix[ix] = True

# df['irig_bit'] = irig_bits
# ic(df)
# ic(irig_bits)

irig_pulselen = np.round(irig_pulselen[~np.isnan(irig_pulselen)],1)
# remove last pulse length since it is incomplete
irig_pulselen = irig_pulselen[:-1]
irig_ix[-2] = False
assert np.isin(irig_pulselen, [.2, .5, .8]).all(), "Unexpected pulse lengths found!!!"
irig_bits = np.zeros(irig_pulselen.size, dtype=object)
irig_bits[irig_pulselen == .2] = False
irig_bits[irig_pulselen == .5] = True
irig_bits[irig_pulselen == .8] = 'P'
# ic(irig_bits)
# ic(df[irig_ix]['time.time()'])
# ic(df[irig_ix]['Signal times'])


# irig_bits[(df['interval_time'] == .2) & irig_ix] = True
# irig_bits[(df['interval_time'] == .5) & irig_ix] = False
# irig_bits[(df['interval_time'] == .8) & irig_ix] = 'P'
# irig_ix = df['interval_time'].isin([.2, .5, .8])
# df['irig_bit'] = irig_bits
# irig_bits = irig_bits[irig_ix]

decoded, frame_ix = decode_irig_bits(irig_bits)
ic([datetime.fromtimestamp(t) for t in decoded])

"""
irig intervals are from the onset (rising edge) of a pulse to its falling edge
"""

