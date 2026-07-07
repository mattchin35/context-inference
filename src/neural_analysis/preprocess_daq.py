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
from scipy.signal import savgol_filter, medfilt
from src.neural_analysis import modified_sinc_smoother as mss
from src.neural_analysis.irig_sync_utils import (
    assign_utc_to_irig_bits as shared_assign_utc_to_irig_bits,
    classify_irig_h_pulses as shared_classify_irig_h_pulses,
    decode_irig_h_frame_anchors as shared_decode_irig_h_frame_anchors,
    decode_sync_line_to_irig_utc as shared_decode_sync_line_to_irig_utc,
    find_signal_edges as shared_find_signal_edges,
    interpolate_with_linear_extrapolation as shared_interpolate_with_linear_extrapolation,
    map_digital_rising_edges_to_utc as shared_map_digital_rising_edges_to_utc,
    map_sample_indices_to_utc as shared_map_sample_indices_to_utc,
    pulse_lengths_from_edges as shared_pulse_lengths_from_edges,
)
from src.neural_analysis.spikeglx_sync_io import read_digital_lines as shared_read_digital_lines
import pywt
from scipy import signal


def read_digital_lines(binaryFilePath: str, digitalWord: int, digitalLines: list[int]) -> [np.ndarray, int]:
    """Read one or more SpikeGLX digital lines.

    Args:
        binaryFilePath: Path to a SpikeGLX ``.bin`` file.
        digitalWord: Digital word index used by ``ExtractDigital``.
        digitalLines: Line indices within ``digitalWord``.

    Returns:
        tuple[np.ndarray, int]:
            - Digital signal array with shape ``(n_lines, n_samples)`` or
              ``(n_samples,)`` stored as bool.
            - Sampling rate in Hz.
    """
    return shared_read_digital_lines(binaryFilePath, digitalWord, digitalLines)


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
    """Find rising and falling edge sample indices for a digital signal."""
    return shared_find_signal_edges(binary_signal)


def pulse_lengths_from_edges(rising_ix: np.ndarray, falling_ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair each rising edge to the next falling edge and return pulse lengths."""
    return shared_pulse_lengths_from_edges(rising_ix, falling_ix)


def classify_irig_h_pulses(
    pulse_lengths_samples: np.ndarray,
    sample_rate: float,
    bit_period_s: float = 1.0) -> np.ndarray:
    """Classify IRIG-H pulse lengths into False, True, or position markers."""
    return shared_classify_irig_h_pulses(pulse_lengths_samples, sample_rate, bit_period_s)


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
    """Decode IRIG-H frame starts from bit labels."""
    return shared_decode_irig_h_frame_anchors(irig_bits)


def assign_utc_to_irig_bits(n_bits: int, frame_anchors: list[tuple[int, float]]) -> np.ndarray:
    """Assign UTC unix seconds to each IRIG bit index using frame anchors."""
    return shared_assign_utc_to_irig_bits(n_bits, frame_anchors)


def decode_daq_irig_h_crossings(
    daq_irig: npt.ArrayLike,
    sample_rate: float,
    bit_period_s: float = 1.0,
    align_first_crossing_unix: Optional[float] = None,
) -> pd.DataFrame:
    """Decode IRIG-H from a digital DAQ line and return UTC for pulse onsets."""
    return shared_decode_sync_line_to_irig_utc(
        sync_signal=daq_irig,
        sample_rate_hz=sample_rate,
        bit_period_s=bit_period_s,
        align_first_rising_edge_unix=align_first_crossing_unix,
    )


def _interpolate_with_linear_extrapolation(
    x_query: np.ndarray,
    x_known: np.ndarray,
    y_known: np.ndarray,
) -> np.ndarray:
    """Interpolate y(x) for query points, with linear extrapolation beyond bounds."""
    return shared_interpolate_with_linear_extrapolation(x_query, x_known, y_known)


def map_digital_crossings_to_utc(
    digital_signal: npt.ArrayLike,
    sample_rate: float,
    irig_crossings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map positive crossings on a digital line to UTC using IRIG anchors."""
    return shared_map_digital_rising_edges_to_utc(
        digital_signal=digital_signal,
        sample_rate_hz=sample_rate,
        irig_df=irig_crossings_df,
    )


def map_sample_indices_to_utc(sample_ix: npt.ArrayLike, irig_crossings_df: pd.DataFrame) -> pd.DataFrame:
    """Map an arbitrary collection of sample indices to UTC using IRIG anchors."""
    return shared_map_sample_indices_to_utc(sample_ix, irig_crossings_df)


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
    # ic(events['crossing_t'])

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


def analyze_treadmill_signal(treadmill_analog: np.ndarray, sample_rate: float):
    signal = np.asarray(treadmill_analog, dtype=float).squeeze()
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError('treadmill_analog must be a 1D signal with at least 2 samples.')

    # Remove DC to make narrow-band noise easier to spot.
    signal = signal - np.nanmedian(signal)

    nperseg = int(min(4096, signal.size))
    if nperseg < 128:
        nperseg = signal.size
    noverlap = int(nperseg // 2)

    f_spec, t_spec, sxx = sp.signal.spectrogram(
        signal,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        scaling='density',
        mode='psd',
    )
    f_per, pxx = sp.signal.periodogram(
        signal,
        fs=sample_rate,
        window='hann',
        detrend='constant',
        scaling='density',
    )
    f_welch, pxx_welch = sp.signal.welch(
        signal,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        scaling='density',
    )

    sxx_db = 10 * np.log10(sxx + 1e-20)
    pxx_db = 10 * np.log10(pxx + 1e-20)
    pxx_welch_db = 10 * np.log10(pxx_welch + 1e-20)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    pcm = axes[0].pcolormesh(t_spec, f_spec, sxx_db, shading='auto')
    axes[0].set_title('Treadmill Analog Spectrogram')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(pcm, ax=axes[0], label='Power/Frequency (dB)')

    axes[1].plot(f_per, pxx_db, lw=1.0)
    axes[1].set_title('Treadmill Analog Periodogram')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power/Frequency (dB)')
    axes[1].grid(alpha=0.3)

    axes[2].plot(f_welch, pxx_welch_db, lw=1.0, color='tab:orange')
    axes[2].set_title('Treadmill Analog Welch PSD')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Power/Frequency (dB)')
    axes[2].grid(alpha=0.3)

    plt.show()


def wavelet_hard_threshold(signal: np.ndarray, wavelet: str = 'sym8', level: int = 4) -> np.ndarray:
    """
    Removes baseline noise while preserving sharp peaks using Wavelet Hard Thresholding.
    """
    # 1. Decompose the signal into wavelet coefficients
    # 'per' mode handles the boundaries cleanly
    coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)

    # 2. Estimate the noise floor
    # We look at the finest detail coefficients (the last array in coeffs)
    # Median Absolute Deviation (MAD) is the standard robust estimator for this
    detail_coeffs = coeffs[-1]
    mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
    sigma = mad / 0.6745  # 0.6745 relates the median to the standard deviation

    # 3. Calculate the Universal Threshold (Donoho-Johnstone)
    # This mathematically calculates the maximum expected height of pure white noise
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # 4. Apply the HARD threshold
    # We keep coeffs[0] (the low-frequency baseline approximation) untouched
    denoised_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        # 'hard' mode sets values below threshold to 0, and leaves others completely unchanged
        thresholded_array = pywt.threshold(coeffs[i], value=threshold, mode='hard')
        denoised_coeffs.append(thresholded_array)

    # 5. Reconstruct the clean signal
    clean_signal = pywt.waverec(denoised_coeffs, wavelet, mode='per')

    # Ensure the output length matches the input length exactly
    return clean_signal[:len(signal)]


def robust_spectral_subtraction(noisy_sig: np.ndarray, noise_ref: np.ndarray, fs: int = 25000):
    """
    Subtracts noise even if noise_ref and noisy_sig are different lengths.
    """
    # 1. Estimate the average noise magnitude spectrum using Welch
    # nperseg should be roughly the length of your sharpest peak
    freqs, noise_psd = signal.welch(noise_ref, fs=fs, nperseg=1024)
    noise_mag_template = np.sqrt(noise_psd)

    # 2. Use STFT to process the noisy signal in blocks
    f, t, Zxx = signal.stft(noisy_sig, fs=fs, nperseg=1024)

    # 3. Perform the subtraction on the magnitude
    # Zxx is complex; we subtract from the absolute magnitude
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Subtract noise template (broadcasting across time)
    # We use a 'noise_reduction_factor' (beta) to tune the aggressiveness
    beta = 1.5
    clean_mag = np.maximum(magnitude - beta * noise_mag_template[:, np.newaxis], 0)

    # 4. Reconstruct and Inverse STFT
    Zxx_clean = clean_mag * np.exp(1j * phase)
    _, cleaned_sig = signal.istft(Zxx_clean, fs=fs)

    return cleaned_sig


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
    t_start = 0
    t_end = 60*10
    data_type = 'A'  # 'A' for analog, 'D' for digital data
    chan_list = [0]  # must be a list for readSGLX functions
    metadata = readSGLX.readMeta(ni_file)
    # Rate = readSGLX.SampRate(meta)
    first_samp = int(daq_srate * t_start)
    last_samp = int(daq_srate * t_end)
    # sample indices for analog segment and their UTC times
    sample_ix = np.arange(first_samp, last_samp + 1, dtype='uint64')
    sample_utc_df = map_sample_indices_to_utc(sample_ix, irig_crossings_df)

    nidaq_analog = readSGLX.makeMemMapRaw(ni_file, metadata)
    treadmill_analog = nidaq_analog[chan_list, first_samp:last_samp + 1]
    MN, MA, XA, DW = readSGLX.ChannelCountsNI(metadata)
    ic("NI channel counts:", MN, MA, XA, DW)
    # apply gain correction and convert to mV
    treadmill_analog = 1e3 * readSGLX.GainCorrectNI(treadmill_analog, chan_list, metadata)
    treadmill_analog = np.squeeze(treadmill_analog)  # treadmill signal should now be 1D array of mV values
    # treadmill_analog -= np.nanmedian(treadmill_analog)  # remove DC offset
    # savgol_result = savgol_filter(treadmill_analog, window_length=21, polyorder=3)
    # ms_result = mss.smooth_ms(treadmill_analog, deg=6, m=15)
    # median_result = medfilt(treadmill_analog, kernel_size=11)

    speed_raw = volts2speed(treadmill_analog) # in mm/s
    speed_raw -= np.nanmedian(speed_raw)  # remove DC offset
    speed_savgol = volts2speed(savgol_result)
    speed_savgol -= np.nanmedian(speed_savgol)
    # speed_ms = volts2speed(ms_result)
    # speed_median = volts2speed(median_result)

    # 2. Define the noise floor
    # (Assume the first 1000 samples are just idle background noise)
    idle_noise = speed_savgol[0:1024*5]
    # noise_mean = np.mean(idle_noise)
    # noise_std = np.std(idle_noise)

    # Set threshold to 4 standard deviations above the noise mean
    # threshold = noise_mean + (5 * noise_std)

    # 3. Apply the Noise Gate (Force everything below threshold to 0)
    # gated_signal = np.where(np.abs(speed_savgol) < threshold, 0, speed_savgol)
    # gated_signal = np.where(np.abs(speed_savgol) < 35, 0, speed_savgol) ## this will work but it's not ideal

    # ic(np.mean(treadmill_analog), np.median(treadmill_analog), np.std(treadmill_analog))
    # ic(np.mean(speed_raw), np.median(speed_raw), np.std(speed_raw))
    # ic(np.mean(speed_savgol), np.median(speed_savgol), np.std(speed_savgol))
    # ic(np.mean(speed_ms), np.median(speed_ms), np.std(speed_ms))

    # Filter the signal
    # 'sym8' (Symlets) or 'db4' (Daubechies) are excellent wavelets for sharp spikes
    # filtered_signal = wavelet_hard_threshold(speed_raw, wavelet='sym8', level=5)

    # plt.plot(sample_utc_df['utc_datetime'], speed)
    plt.plot(sample_utc_df['utc_datetime'], speed_raw, label='Raw signal')
    plt.plot(sample_utc_df['utc_datetime'], speed_savgol, label='Savitzky-Golay')
    plt.plot(sample_utc_df['utc_datetime'], gated_signal, label='Savitzky-Golay with noise gate')
    # plt.plot(sample_utc_df['utc_datetime'], filtered_signal, label="Wavelet Output (Flat baseline, sharp peaks)")
    # plt.plot(sample_utc_df['utc_datetime'], speed_ms, label='Modified Sinc')
    # plt.plot(sample_utc_df['utc_datetime'], speed_median, label='Median Filter')
    plt.xlabel('UTC time')
    plt.ylabel('Treadmill speed (mm/s)')
    plt.legend()
    plt.show()

    # analyze_treadmill_signal(treadmill_analog, daq_srate)


if __name__ == "__main__":
    main()
