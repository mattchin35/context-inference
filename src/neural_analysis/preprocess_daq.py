import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import src.external_tools.readSGLX as readSGLX
from src.irig_tools import irig_h_gpio as irig
from icecream import ic
import time
import numpy.typing as npt
import re
import pickle as pkl


def read_digital_lines(binaryFilePath: str, digitalWord: int, digitalLines: list[int]) -> [np.ndarray, int]:
    meta = readSGLX.readMeta(binaryFilePath)
    sampleRate = readSGLX.SampRate(meta)
    nChan = int(meta['nSavedChans'])
    nSamples = int(int(meta['fileSizeBytes']) / (2 * nChan))

    firstSamp = 0
    lastSamp = nSamples - 1  # sample ix is 0-indexed but ExtracDigital is inclusive

    rawData = readSGLX.makeMemMapRaw(binaryFilePath, meta)
    digArray = readSGLX.ExtractDigital(rawData, firstSamp, lastSamp, digitalWord, digitalLines, meta)
    digArray = np.squeeze(digArray)
    return digArray, sampleRate


def volts2speed(mVolts: np.ndarray) -> np.ndarray:
    """
    Calculate the speed of the mouse based on the treadmill data.
    :param mVolts: Treadmill voltage signal in mV
    """
    max_dacval = 2.5 / 3.3 * 4095  # 2.5V is the max voltage for the treadmill
    dacval = mVolts * 4095 / 3300  # convert mV to DAC value
    vel = (dacval / max_dacval - .5) * 2 * 1000  # in mm/s
    return vel


def find_signal_edges(binary_signal: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Find rising (positive threshold crossing) and falling edge sample indices.
    """
    sig = np.asarray(binary_signal, dtype=bool)
    rising_ix = np.flatnonzero(~sig[:-1] & sig[1:]) + 1
    falling_ix = np.flatnonzero(sig[:-1] & ~sig[1:]) + 1

    if sig.size and sig[0]:
        rising_ix = np.insert(rising_ix, 0, 0)
    return rising_ix, falling_ix


def pulse_lengths_from_edges(rising_ix: np.ndarray, falling_ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pair each rising edge to the next falling edge and return pulse lengths in samples.
    """
    if rising_ix.size == 0 or falling_ix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    next_fall_pos = np.searchsorted(falling_ix, rising_ix, side='right')
    valid = next_fall_pos < falling_ix.size
    if not np.any(valid):
        return np.array([], dtype=int), np.array([], dtype=float)

    paired_rising = rising_ix[valid]
    paired_falling = falling_ix[next_fall_pos[valid]]
    lengths = (paired_falling - paired_rising).astype(float)
    return paired_rising, lengths


def classify_irig_h_pulses(
    pulse_lengths_samples: np.ndarray,
    sample_rate: float,
    bit_period_s: float = 1.0) -> np.ndarray:
    """
    Classify IRIG-H pulse lengths into False(0), True(1), or 'P' marker.
    """
    samples_per_bit = sample_rate * bit_period_s
    p_thresh = 0.75 * samples_per_bit
    one_thresh = 0.45 * samples_per_bit
    zero_thresh = 0.05 * samples_per_bit

    bits = np.full(pulse_lengths_samples.shape, None, dtype=object)
    bits[pulse_lengths_samples > p_thresh] = 'P'
    bits[(pulse_lengths_samples > one_thresh) & (pulse_lengths_samples <= p_thresh)] = True
    bits[(pulse_lengths_samples > zero_thresh) & (pulse_lengths_samples <= one_thresh)] = False
    return bits


def _irig_frame_to_utc_unix(frame_bits: list[object]) -> Optional[float]:
    """
    Convert one IRIG-H frame to unix time assuming the encoded clock is UTC.
    """
    decoded_dt = irig.irig_h_to_datetime(frame_bits)
    if decoded_dt is None:
        return None
    decoded_dt_utc = decoded_dt.replace(tzinfo=timezone.utc)
    return float(decoded_dt_utc.timestamp())


def decode_irig_h_frame_anchors(irig_bits: np.ndarray) -> list[tuple[int, float]]:
    """
    Decode IRIG-H frame starts from bit labels and return (frame_start_bit_ix, frame_start_unix).
    """
    if irig_bits.size < 122:
        return []

    tracking_start = None
    scan_max = min(120, irig_bits.size - 1)
    for i in range(scan_max):
        if irig_bits[i] == 'P' and irig_bits[i + 1] == 'P':
            tracking_start = i + 1
            break
    if tracking_start is None:
        return []

    frame_bits = []
    frame_ix = []
    anchors: list[tuple[int, float]] = []
    for i in range(tracking_start, irig_bits.size - 1):
        frame_bits.append(irig_bits[i])
        frame_ix.append(i)
        if irig_bits[i] == 'P' and irig_bits[i + 1] == 'P':
            if len(frame_bits) == 60:
                posix = _irig_frame_to_utc_unix(frame_bits)
                if posix is not None:
                    anchors.append((frame_ix[0], float(posix)))
            frame_bits = []
            frame_ix = []
    return anchors


def assign_utc_to_irig_bits(n_bits: int, frame_anchors: list[tuple[int, float]]) -> np.ndarray:
    """
    Assign UTC unix seconds to each IRIG bit index using decoded frame anchors.
    """
    unix_time = np.full(n_bits, np.nan, dtype=float)
    if n_bits == 0 or len(frame_anchors) == 0:
        return unix_time

    anchor_ix = np.array([a[0] for a in frame_anchors], dtype=int)
    anchor_unix = np.array([a[1] for a in frame_anchors], dtype=float)
    keep = np.concatenate(([True], np.diff(anchor_ix) > 0))
    anchor_ix = anchor_ix[keep]
    anchor_unix = anchor_unix[keep]
    if anchor_ix.size == 0:
        return unix_time

    first_ix = anchor_ix[0]
    unix_time[first_ix:] = anchor_unix[0] + np.arange(n_bits - first_ix, dtype=float)
    unix_time[:first_ix] = anchor_unix[0] - np.arange(first_ix, 0, -1, dtype=float)

    for k in range(1, anchor_ix.size):
        start = anchor_ix[k]
        unix_time[start:] = anchor_unix[k] + np.arange(n_bits - start, dtype=float)
    return unix_time


def decode_daq_irig_h_crossings(
    daq_irig: npt.ArrayLike,
    sample_rate: float,
    bit_period_s: float = 1.0,
    align_first_crossing_unix: Optional[float] = None,
) -> pd.DataFrame:
    """
    Decode IRIG-H from digital DAQ line and return UTC for each positive threshold crossing.
    """
    rising_ix, falling_ix = find_signal_edges(daq_irig)
    paired_rising_ix, pulse_lengths = pulse_lengths_from_edges(rising_ix, falling_ix)
    irig_bits = classify_irig_h_pulses(pulse_lengths, sample_rate=sample_rate, bit_period_s=bit_period_s)

    valid_mask = np.array([bit is not None for bit in irig_bits], dtype=bool)
    valid_bits = irig_bits[valid_mask]
    anchors = decode_irig_h_frame_anchors(valid_bits)
    valid_unix = assign_utc_to_irig_bits(valid_bits.size, anchors)

    all_unix = np.full(irig_bits.shape[0], np.nan, dtype=float)
    all_unix[valid_mask] = valid_unix

    if align_first_crossing_unix is not None and np.isfinite(align_first_crossing_unix):
        finite_ix = np.where(np.isfinite(all_unix))[0]
        if finite_ix.size > 0:
            shift_s = float(align_first_crossing_unix) - float(all_unix[finite_ix[0]])
            all_unix = all_unix + shift_s

    crossing_time_s = paired_rising_ix / float(sample_rate)
    utc_datetime = [datetime.fromtimestamp(t, tz=timezone.utc) if np.isfinite(t) else pd.NaT for t in all_unix]
    return pd.DataFrame(
        {
            'sample_ix': paired_rising_ix,
            'recording_time_s': crossing_time_s,
            'pulse_len_samples': pulse_lengths,
            'irig_bit': irig_bits,
            'utc_unix': all_unix,
            'utc_datetime': utc_datetime,
        }
    )


def _interpolate_with_linear_extrapolation(
    x_query: np.ndarray,
    x_known: np.ndarray,
    y_known: np.ndarray,
) -> np.ndarray:
    """
    Interpolate y(x) for query points, with linear extrapolation beyond bounds.
    """
    y = np.interp(x_query.astype(float), x_known.astype(float), y_known.astype(float))
    if x_known.size < 2:
        return y

    left_mask = x_query < x_known[0]
    right_mask = x_query > x_known[-1]

    left_dx = float(x_known[1] - x_known[0])
    right_dx = float(x_known[-1] - x_known[-2])
    left_slope = (y_known[1] - y_known[0]) / left_dx if left_dx != 0 else 0.0
    right_slope = (y_known[-1] - y_known[-2]) / right_dx if right_dx != 0 else 0.0

    y[left_mask] = y_known[0] + (x_query[left_mask] - x_known[0]) * left_slope
    y[right_mask] = y_known[-1] + (x_query[right_mask] - x_known[-1]) * right_slope
    return y


def map_digital_crossings_to_utc(
    digital_signal: npt.ArrayLike,
    sample_rate: float,
    irig_crossings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map positive crossings on a digital line to UTC using IRIG-derived crossing times.
    """
    rising_ix, _ = find_signal_edges(digital_signal)

    known = irig_crossings_df[np.isfinite(irig_crossings_df['utc_unix'])][['sample_ix', 'utc_unix']].copy()
    known = known.drop_duplicates(subset='sample_ix').sort_values('sample_ix')
    if known.shape[0] < 2:
        raise ValueError('Need at least two finite IRIG UTC points to map digital crossings.')

    utc_unix = _interpolate_with_linear_extrapolation(
        x_query=rising_ix.astype(float),
        x_known=known['sample_ix'].to_numpy(dtype=float),
        y_known=known['utc_unix'].to_numpy(dtype=float),
    )
    utc_datetime = [datetime.fromtimestamp(t, tz=timezone.utc) for t in utc_unix]
    return pd.DataFrame(
        {
            'sample_ix': rising_ix,
            'recording_time_s': rising_ix / float(sample_rate),
            'utc_unix': utc_unix,
            'utc_datetime': utc_datetime,
        }
    )


def map_sample_indices_to_utc(sample_ix: np.ndarray, irig_crossings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map arbitrary sample indices to UTC using IRIG-derived sample->UTC mapping.
    """
    known = irig_crossings_df[np.isfinite(irig_crossings_df['utc_unix'])][['sample_ix', 'utc_unix']].copy()
    known = known.drop_duplicates(subset='sample_ix').sort_values('sample_ix')
    if known.shape[0] < 2:
        raise ValueError('Need at least two finite IRIG UTC points to map sample indices.')

    utc_unix = _interpolate_with_linear_extrapolation(
        x_query=sample_ix.astype(float),
        x_known=known['sample_ix'].to_numpy(dtype=float),
        y_known=known['utc_unix'].to_numpy(dtype=float),
    )
    utc_datetime = [datetime.fromtimestamp(t, tz=timezone.utc) for t in utc_unix]
    return pd.DataFrame({'sample_ix': sample_ix, 'utc_unix': utc_unix, 'utc_datetime': utc_datetime})


def get_flipper_events(data: np.ndarray, sample_rate: float, threshold: float = 0.5, debounce=.0002):
    boolean_signal = data.astype(bool)
    pos_ix, neg_ix = find_signal_edges(boolean_signal)
    crossing_ix = np.sort(np.concatenate((pos_ix, neg_ix)))

    if crossing_ix.size == 0:
        return dict(
            crossing_ix=np.array([], dtype=int),
            crossing_t=np.array([], dtype=float),
            pos_ix=np.array([], dtype=int),
            pos_t=np.array([], dtype=float),
            neg_ix=np.array([], dtype=int),
            neg_t=np.array([], dtype=float),
        )

    # Remove very short toggles as noise.
    crossing_t = crossing_ix / float(sample_rate)
    noise_length = 0.001  # seconds
    keep = np.ones(crossing_ix.shape[0], dtype=bool)
    i = 0
    while i < crossing_ix.size - 1:
        if (crossing_t[i + 1] - crossing_t[i]) < noise_length:
            keep[i] = False
            keep[i + 1] = False
            i += 2
        else:
            i += 1

    crossing_ix = crossing_ix[keep]
    crossing_t = crossing_ix / float(sample_rate)
    pos_ix = pos_ix[np.isin(pos_ix, crossing_ix)]
    neg_ix = neg_ix[np.isin(neg_ix, crossing_ix)]
    pos_t = pos_ix / float(sample_rate)
    neg_t = neg_ix / float(sample_rate)

    events = dict(
        crossing_ix=crossing_ix,
        crossing_t=crossing_t,
        pos_ix=pos_ix,
        pos_t=pos_t,
        neg_ix=neg_ix,
        neg_t=neg_t,
    )
    return events


def decode_flipper_barcodes(flipper_signal: np.ndarray, sample_rate: float=25000, barcode_present=True, plot=False):
    events = get_flipper_events(flipper_signal, sample_rate)
    if plot:
        f, ax = plt.subplots()
        x = np.arange(flipper_signal.size) / sample_rate
        start_time = 3600
        end_time = 3700  # seconds
        plt.plot(x[start_time*int(sample_rate):end_time*int(sample_rate)], flipper_signal[start_time*int(sample_rate):end_time*int(sample_rate)])
        plt.title('Flipper signal')
        plt.show()

    if not barcode_present:
        events['flipper_pos_t'] = events['pos_t']
        events['flipper_neg_t'] = events['neg_t']
        return events, (events['crossing_t'][0], events['crossing_t'][-1]), None

    nbits = 32
    wrapper_bit_time = .01  # seconds
    barcode_bit_time = .03  # seconds

    wrapper_time = 3 * wrapper_bit_time  # Off-On-Off
    barcode_time = nbits * barcode_bit_time  # 32 bits
    total_barcode_time = barcode_time + 2 * wrapper_time

    # Tolerance conversions
    tolerance = .2  # % tolerance - so for a duration of 10 and 10% tolerance, 9-11 is acceptable
    min_wrap_duration = wrapper_bit_time - wrapper_bit_time * tolerance
    max_wrap_duration = wrapper_bit_time + wrapper_bit_time * tolerance
    min_bar_duration = barcode_bit_time - barcode_bit_time * tolerance
    max_bar_duration = barcode_bit_time + barcode_bit_time * tolerance
    # sample_conversion = 1000 / expected_sample_rate  # Convert sampling rate to msec

    # ic(events['crossing_t'])
    # events = remove_signal_noise(flipper_signal, events, noise_length=.001)
    ic(events['crossing_t'])

    wrapper_t = []
    pulse_times = np.diff(events['crossing_t'])
    pulse_time_match = (pulse_times > min_wrap_duration) & (pulse_times < max_wrap_duration)
    pulse_direction_match = flipper_signal[events['crossing_ix']][:-1].astype(bool)
    wrapper_ix = pulse_time_match & pulse_direction_match
    assert np.sum(wrapper_ix) == 4, "File is missing barcodes or barcode wrappers"
    wrapper_pulse_t = events['crossing_t'][:-1][wrapper_ix]
    ic(wrapper_pulse_t)
    # plt.show()

    barcode_st = [wrapper_pulse_t[0] + 2*wrapper_bit_time, wrapper_pulse_t[2] + 2*wrapper_bit_time]
    barcode_end = [wrapper_pulse_t[1] - wrapper_bit_time, wrapper_pulse_t[3] - wrapper_bit_time]

    flipper_bounds = [wrapper_pulse_t[1] + max_wrap_duration, wrapper_pulse_t[2] - max_wrap_duration]

    flipper_pos_mask = (events['pos_t'] > flipper_bounds[0]) & (events['pos_t'] < flipper_bounds[1])
    flipper_neg_mask = (events['neg_t'] > flipper_bounds[0]) & (events['neg_t'] < flipper_bounds[1])
    events['flipper_pos_t'] = events['pos_t'][flipper_pos_mask]
    events['flipper_neg_t'] = events['neg_t'][flipper_neg_mask]

    events['barcode_start_t'] = np.array(barcode_st, dtype=float)
    events['barcode_end_t'] = np.array(barcode_end, dtype=float)
    events['barcode_start_ix'] = np.round(events['barcode_start_t'] * sample_rate).astype(int)
    events['barcode_end_ix'] = np.round(events['barcode_end_t'] * sample_rate).astype(int)

    # decode the barcodes
    signals_barcodes = []
    for start, end in zip(barcode_st, barcode_end):
        on_times = (events['pos_t'] > start) & (events['pos_t'] < end)
        off_times = (events['neg_t'] > start) & (events['neg_t'] < end)
        cur_time = start
        bits = np.zeros((nbits,))
        interbit_ON = False  # Changes to "True" during multiple ON bars

        for bit in range(0, nbits):
            next_on = events['pos_t'][on_times][0] if np.any(on_times) else end #start + total_barcode_time
            next_off = events['neg_t'][off_times][0] if np.any(off_times) else end #start + total_barcode_time

            if (cur_time - barcode_bit_time*tolerance) <= next_on <= (cur_time + barcode_bit_time*tolerance):
                bits[bit] = 1
                interbit_ON = True
            elif (cur_time - barcode_bit_time*tolerance) <= next_off <= (cur_time + barcode_bit_time*tolerance):
                interbit_ON = False  # bits default to 0
            elif interbit_ON:
                bits[bit] = 1
            else:
                pass  # bits default to 0

            cur_time += barcode_bit_time
            on_times = (events['pos_t'] > cur_time) & on_times
            off_times = (events['neg_t'] > cur_time) & off_times

        barcode = 0
        for bit in range(0, nbits):
            barcode += bits[bit] * pow(2, nbits-1 - bit)

        signals_barcodes.append(int(barcode))

    signals_barcodes_bitwise = (format(int(signals_barcodes[0]), '0' + str(32) + 'b'), format(int(signals_barcodes[1]), '0' + str(32) + 'b'))
    return events, flipper_bounds, (signals_barcodes, signals_barcodes_bitwise)


def get_flipper_barcode_presence_times_utc(
    daq_flipper: np.ndarray,
    sample_rate: float,
    irig_crossings_df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return UTC times when the first and second barcodes are present on the flipper signal.
    """
    flipper_events, _, _ = decode_flipper_barcodes(
        daq_flipper,
        sample_rate=sample_rate,
        barcode_present=True,
        plot=False,
    )
    barcode_times_df = map_sample_indices_to_utc(flipper_events['barcode_start_ix'], irig_crossings_df)
    return barcode_times_df.iloc[0]['utc_datetime'], barcode_times_df.iloc[1]['utc_datetime']


def main():
    ### Behavior paths ###
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    raw_ephys_folder = session_data_home / 'ephys/raw/run0_g0'
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)

    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
        sess_id_abbreviated = mouse + '_' + date
    else:
        print("Double-check the session name!")
        return

    session_info_path = raw_behavior_folder / '{}_session_info.pkl'.format(sess_id_full)
    assert session_info_path.exists(), "session_info at {} not found!".format(session_info_path)
    with open(session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    # ni file
    ni_file = raw_ephys_folder.joinpath('run0_g0_t0.nidq.bin')
    ni_word = 0
    ni_lines = [0, 1, 2, 3]  # ephys sync/IRIG, flipper, left, right
    daq_lines, daq_srate = read_digital_lines(ni_file, ni_word, ni_lines)
    daq_irig = daq_lines[0]
    daq_flipper = daq_lines[1]
    daq_leftlick = daq_lines[2]
    daq_rightlick = daq_lines[3]
    irig_crossings_df = decode_daq_irig_h_crossings(
        daq_irig,
        daq_srate,
        align_first_crossing_unix=None,
    )
    ic(irig_crossings_df.head())
    leftlick_crossings_df = map_digital_crossings_to_utc(daq_leftlick, daq_srate, irig_crossings_df)
    rightlick_crossings_df = map_digital_crossings_to_utc(daq_rightlick, daq_srate, irig_crossings_df)
    ic(leftlick_crossings_df.head())
    ic(rightlick_crossings_df.head())

    flipper_events, flipper_bounds, flipper_barcodes = decode_flipper_barcodes(
        daq_flipper,
        sample_rate=daq_srate,
        barcode_present=True,
        plot=False,
    )
    first_barcode_present_utc, second_barcode_present_utc = get_flipper_barcode_presence_times_utc(
        daq_flipper,
        daq_srate,
        irig_crossings_df,
    )
    ic(flipper_barcodes)
    ic(first_barcode_present_utc, second_barcode_present_utc)

    # ni analog treadmill signal
    # tStart = 0
    # tEnd = 15
    # dataType = 'A'  # 'A' for analog, 'D' for digital data
    # chanList = [0]
    # meta = readSGLX.readMeta(ni_file)
    # # Rate = readSGLX.SampRate(meta)
    # firstSamp = int(daq_srate * tStart)
    # lastSamp = int(daq_srate * tEnd)
    # # array of times for plot
    # tDat = np.arange(firstSamp, lastSamp + 1, dtype='uint64')
    # tDat = 1000 * tDat / daq_srate  # plot time axis in msec
    # rawData = readSGLX.makeMemMapRaw(ni_file, meta)
    # selectData = rawData[chanList, firstSamp:lastSamp + 1]
    # MN, MA, XA, DW = readSGLX.ChannelCountsNI(meta)
    # ic("NI channel counts:", MN, MA, XA, DW)
    # # apply gain correction and convert to mV
    # convData = 1e3 * readSGLX.GainCorrectNI(selectData, chanList, meta)
    # convData = np.squeeze(convData)
    # speed = volts2speed(convData)  # in mm/s


if __name__ == "__main__":
    main()
