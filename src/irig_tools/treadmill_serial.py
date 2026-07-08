import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from icecream import ic
from src.irig_tools import irig_core
from src.irig_tools import decode_irig_logic_csv
from src.irig_tools import irig_serial_io
from typing import Generator, List, Optional, Tuple, Literal
from datetime import datetime, timezone


def decode_standard_decisecond_frame(frame: list[object]) -> tuple[float | None, datetime | None]:
    """Decode one legacy standard-decisecond IRIG frame.

    Args:
        frame: IRIG frame bits with shape ``(60,)``.

    Returns:
        tuple[float | None, datetime | None]: UTC Unix seconds and UTC-aware
        datetime. Both are ``None`` if the frame is invalid.
    """
    try:
        frame_fields = irig_core.decode_irig_frame(
            frame,
            irig_format="standard_decisecond",
        )
        decoded_unix = float(frame_fields["decoded_utc_seconds"])
        if not np.isfinite(decoded_unix):
            return None, None
        return decoded_unix, datetime.fromtimestamp(decoded_unix, tz=timezone.utc)
    except ValueError:
        return None, None


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

    decoded_pairs = [decode_standard_decisecond_frame(frame) for frame in irig_frames]
    posix_decoded = [decoded_pair[0] for decoded_pair in decoded_pairs]
    datetime_decoded = [decoded_pair[1] for decoded_pair in decoded_pairs]

    # Handle invalid timecodes
    # decoded = [item for item in decoded if item is not None]

    print(f'List spliced! Splices: {len(posix_decoded)}')
    return posix_decoded, datetime_decoded, frame_ix


def treadmill_log_to_transition_df(
        parsed_df: pd.DataFrame,
        local_timezone: str = "America/New_York") -> pd.DataFrame:
    """Convert parsed CoolTerm sync rows into the desktop IRIG transition format.

    Args:
        parsed_df: Parsed serial log table. Required columns are ``label``,
            ``value``, ``time_value``, ``pc_date``, and ``pc_time``. The
            ``time_value`` column must have shape ``(n_rows,)`` in seconds and
            represents the serial interval attached to each transition row.
        local_timezone: IANA timezone name for interpreting PC timestamp text.

    Returns:
        pd.DataFrame: Transition table with one row per sync transition and
        required desktop decoder columns ``utc_seconds``, ``local_datetime``,
        and ``pin_state``. ``utc_seconds`` is anchored to the first PC timestamp
        but subsequent timing uses serial intervals, not PC timestamps.
    """
    return irig_serial_io.treadmill_log_to_transition_df(
        parsed_df=parsed_df,
        local_timezone=local_timezone,
    )


# file_path = Path.home() / 'Documents' / 'ephys_transfer' / 'treadmill_20251008' / 'CoolTerm Capture (Untitled_0) 2025-10-21 12-36-20-447.txt'
file_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/test_runs/irig_test_20260707/CoolTerm Capture (Untitled_0) 2026-07-07 12-06-51-676.txt')
# df = pd.read_csv(filepath, sep=';', header=None, on_bad_lines='skip', usecols=[1, 2, 4])

log_labels = ['Fdistance', 'Bdistance', 'dacval', 'runSpeed']
irig_labels = ['syncPinState']

pc_timestamp_present = True
try:
    filesize = file_path.stat().st_size
    print(f"File size: {filesize} bytes")
    df = irig_serial_io.load_coolterm_serial_txt(
        file_path,
        pc_timestamp_present=pc_timestamp_present,
    )
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    df = pd.DataFrame()
except Exception as e:
    print(f"An error occurred: {e}")
    df = pd.DataFrame()

transition_df = treadmill_log_to_transition_df(df)
pulse_df = decode_irig_logic_csv.extract_irig_pulses_from_transitions(transition_df)
pulse_debug_df = decode_irig_logic_csv.build_pulse_debug_table(pulse_df)
frame_debug_df = decode_irig_logic_csv.build_frame_debug_table(pulse_df)

print("NeuroKairos-compatible transition rows:")
print(
    transition_df.loc[
        :,
        ['utc_seconds', 'local_datetime', 'pin_state', 'pc_local_datetime', 'pc_vs_reconstructed_error_s'],
    ].head(10).to_string(index=False)
)
print("NeuroKairos-compatible pulse rows:")
print(
    pulse_debug_df.loc[
        :,
        ['rising_edge_ix', 'rising_edge_utc_seconds', 'pulse_width_s', 'irig_bit'],
    ].head(10).to_string(index=False)
)
print("NeuroKairos-compatible decoded frames:")
if frame_debug_df.empty:
    print("No valid 60-pulse NeuroKairos frames found.")
else:
    print(
        frame_debug_df.loc[
            :,
            [
                'frame_ix',
                'observed_start_utc_seconds',
                'observed_start_local_datetime',
                'decoded_utc_seconds',
                'decoded_datetime_utc',
                'decoded_stratum',
                'decoded_dispersion_bucket',
            ],
        ].head(10).to_string(index=False)
    )

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
print([datetime.fromtimestamp(t, tz=timezone.utc) for t in unix_time_fordf[~np.isnan(unix_time_fordf)][:10]])
