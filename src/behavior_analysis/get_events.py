import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import readSGLX
from icecream import ic
import time


class DAQ:

    def __init__(self, sync_lines: dict, sample_rate: float):
        self.sync_lines = sync_lines
        self.sample_rate = sample_rate
        self.interp_fn = {}

    def sync(self, sync_from: np.ndarray, sync_to: np.ndarray, desc: str = ''):
        interp_fn = sp.interpolate.interp1d(sync_from, sync_to, kind='linear', fill_value='extrapolate')
        self.interp_fn[desc] = interp_fn

    def sync_data(self, description: str, data):
        sync_fn = self.interp_fn[description]
        sync_data = sync_fn(data)
        return sync_data

    def ix_to_time(self, ix: np.ndarray):
        return ix / self.sample_rate

    def time_to_ix(self, time: np.ndarray):
        return time * self.sample_rate


def get_events(data: np.ndarray, sample_rate: float, threshold: float = 0.5, debounce=.0002):
    # crossings = np.diff(data > threshold, prepend=False)
    boolean_signal = data.astype(bool)
    if len(data.shape) == 2:
        crossings = np.diff(np.concatenate((boolean_signal[:, 0].reshape(-1, 1), boolean_signal), axis=1), axis=1)
        crossing_ix = np.transpose(np.nonzero(crossings))
    elif len(data.shape) == 1:
        crossings = np.diff(boolean_signal, prepend=boolean_signal[0], axis=-1)
        crossing_ix = np.nonzero(crossings)[0]
    else:
        raise ValueError("Data must be 1D or 2D array")

    if len(crossing_ix) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    positive_crossings_ix = crossing_ix[data[crossing_ix] > threshold]
    negative_crossings_ix = crossing_ix[data[crossing_ix] < threshold]
    # ic(positive_crossings_ix / sample_rate, negative_crossings_ix / sample_rate)
    pos_ix, pos_t = get_positive_crossings(positive_crossings_ix, negative_crossings_ix, sample_rate, debounce)
    neg_ix, neg_t = get_negative_crossings(positive_crossings_ix, negative_crossings_ix, sample_rate, debounce)
    return pos_ix, pos_t, neg_ix, neg_t


def get_positive_crossings(pos_ix: np.ndarray, neg_ix: np.ndarray, sample_rate: float, debounce: float) -> [np.ndarray, np.ndarray]:
    neg_ix = neg_ix[neg_ix > pos_ix[0]]
    pos_t, neg_t = pos_ix / sample_rate, neg_ix / sample_rate
    for p, n in zip(pos_t, neg_t):
        if n - p < debounce:
            pos_ix = pos_ix[pos_t != p]
            pos_t = pos_t[pos_t != p]
            neg_ix = neg_ix[neg_t != n]
            neg_t = neg_t[neg_t != n]
    return pos_ix, pos_t


def get_negative_crossings(pos_ix: np.ndarray, neg_ix: np.ndarray, sample_rate: float, debounce=float) -> [np.ndarray, np.ndarray]:
    pos_ix = pos_ix[pos_ix > neg_ix[0]]
    pos_t, neg_t = pos_ix / sample_rate, neg_ix / sample_rate
    for p, n in zip(pos_t, neg_t):
        if p - n < debounce:
            pos_ix = pos_ix[pos_t != p]
            pos_t = pos_t[pos_t != p]
            neg_ix = neg_ix[neg_t != n]
            neg_t = neg_t[neg_t != n]
    return neg_ix, neg_t


def run_file(binaryFilePath: str, digitalWord: int, digitalLines: list[int], tag: str=None, debounce=.0002, plot=False) -> [np.ndarray, np.ndarray, np.ndarray]:
    meta = readSGLX.readMeta(binaryFilePath)
    sampleRate = readSGLX.SampRate(meta)
    nChan = int(meta['nSavedChans'])
    nSamples = int(int(meta['fileSizeBytes']) / (2 * nChan))

    firstSamp = 0
    lastSamp = nSamples - 1  # sample ix is 0-indexed but ExtracDigital is inclusive

    # firstSamp = int(sampleRate * 6*60)
    # lastSamp = int(sampleRate * (6*60 + 10)) - 1
    rawData = readSGLX.makeMemMapRaw(binaryFilePath, meta)
    digArray = readSGLX.ExtractDigital(rawData, firstSamp, lastSamp, digitalWord, digitalLines, meta)
    digArray = np.squeeze(digArray)

    pos_ix, pos_t, neg_ix, neg_t = get_events(digArray, sampleRate, debounce=debounce)
    event_ix, event_t = np.sort(np.concatenate((pos_ix, neg_ix))), np.sort(np.concatenate((pos_t, neg_t)))

    if plot:
        plotFirstSamp = int((6*60+4) * sampleRate)
        plotLastSamp = int(sampleRate * 1) + plotFirstSamp -1
        plt.plot(np.arange(plotLastSamp-plotFirstSamp+1), digArray[plotFirstSamp:plotLastSamp+1])
        plt.xticks(np.linspace(0, plotLastSamp-plotFirstSamp, 6), np.round(np.linspace(plotFirstSamp/sampleRate, plotLastSamp/sampleRate, 6), 2))
        events_window = event_ix[(event_ix > plotFirstSamp) & (event_ix < plotLastSamp)]
        ic(events_window / sampleRate)
        if tag:
            plt.title(tag)
        plt.show()

    return (event_ix, event_t), (pos_ix, pos_t), (neg_ix, neg_t), sampleRate


def sync_behavior(imec_file: Path, ni_file: Path, debounce=.0002) -> [np.ndarray, np.ndarray]:
    """
    Synchronize NI file to a probe. Extracts sync lines, flipper, and left/right events.
    :param imec_file:
    :param ni_file:
    :param debounce:
    :return:
    """
    imec_word = 0
    imec_line = [6]  # [0, 1, 6]
    imec_events, imec_pos, imec_neg, imec_srate = run_file(binaryFilePath=imec_file, digitalWord=imec_word,
                                                           digitalLines=imec_line, tag='imec')

    # run ni file
    ni_word = 0
    ni_ephys, ni_ephys_pos, ni_ephys_neg, ni_srate = run_file(binaryFilePath=ni_file, digitalWord=ni_word,
                                                              digitalLines=[0], tag='ni_ephys')
    ni_flip, ni_flip_pos, ni_flip_neg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[1],
                                                    tag='ni_flip')
    ni_Levents, ni_Lpos, ni_Lneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[2],
                                               tag='ni_L', plot=False, debounce=debounce)
    ni_Revents, ni_Rpos, ni_Rneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[3],
                                               tag='ni_R', plot=False, debounce=debounce)

    imec_synclines = {'imec_ix': imec_pos[0], 'imec_t': imec_pos[1]}
    imec = DAQ(sync_lines=imec_synclines, sample_rate=imec_srate)

    ni_synclines = {'ephys_ix': ni_ephys_pos[0], 'ephys_t': ni_ephys_pos[1], 'flipper_ix': ni_flip_pos[0],
                    'flipper_t': ni_flip_pos[1]}
    ni = DAQ(sync_lines=ni_synclines, sample_rate=ni_srate)

    ni.sync(sync_from=ni.sync_lines['ephys_ix'], sync_to=imec.sync_lines['imec_ix'], desc='ephys_ix')
    ni.sync(sync_from=ni.sync_lines['ephys_t'], sync_to=imec.sync_lines['imec_t'], desc='ephys_t')

    # synchronize flipper events
    flipper_ix = ni.sync_data(description='ephys_ix', data=ni_flip_pos[0])
    flipper_t = ni.ix_to_time(flipper_ix)

    # synchronize left/right events
    Lonset_ix = ni.sync_data(description='ephys_ix', data=ni_Lneg[0])
    Ronset_ix = ni.sync_data(description='ephys_ix', data=ni_Rneg[0])
    Loffset_ix = ni.sync_data(description='ephys_ix', data=ni_Lpos[0])
    Roffset_ix = ni.sync_data(description='ephys_ix', data=ni_Rpos[0])

    Lonset_t = ni.ix_to_time(Lonset_ix)
    Ronset_t = ni.ix_to_time(Ronset_ix)
    Loffset_t = ni.ix_to_time(Loffset_ix)
    Roffset_t = ni.ix_to_time(Roffset_ix)

    ix_sync_dict = dict(flipper_ix=flipper_ix, L_onset_ix=Lonset_ix, R_onset_ix=Ronset_ix,
                        L_offset_ix=Loffset_ix, R_offset_ix=Roffset_ix)
    t_sync_dict = dict(flipper_t=flipper_t, L_onset_t=Lonset_t, R_onset_t=Ronset_t,
                       L_offset_t=Loffset_t, R_offset_t=Roffset_t)

    return ix_sync_dict, t_sync_dict#, imec, ni


def get_flipper_events(data: np.ndarray, sample_rate: float, threshold: float = 0.5, debounce=.0002):
    boolean_signal = data.astype(bool)
    crossings = np.diff(boolean_signal, prepend=boolean_signal[0])
    crossing_ix = np.nonzero(crossings)[0]
    crossing_t = crossing_ix / sample_rate
    if len(crossing_ix) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    positive_crossings_ix = crossing_ix[boolean_signal[crossing_ix]]
    negative_crossings_ix = crossing_ix[~boolean_signal[crossing_ix]]
    # ic(positive_crossings_ix / sample_rate, negative_crossings_ix / sample_rate)
    pos_ix, pos_t = get_positive_crossings(positive_crossings_ix, negative_crossings_ix, sample_rate, debounce)
    neg_ix, neg_t = get_negative_crossings(positive_crossings_ix, negative_crossings_ix, sample_rate, debounce)

    # remove noise events (anything under 1 ms)
    noise_length = .001  # seconds
    noise_ix, noise_times = [], []
    i = 0
    while i < len(crossing_ix) - 1:
        if (crossing_t[i + 1] - crossing_t[i]) < noise_length:
            noise_ix.append(i)
            noise_ix.append(i + 1)
            noise_times.append(crossing_t[i])
            noise_times.append(crossing_t[i + 1])
            i += 2
        else:
            i += 1

    crossing_ix = np.delete(crossing_ix, noise_ix)
    crossing_t = np.delete(crossing_t, noise_ix)
    pos_mask = np.isin(pos_ix, crossing_ix)
    neg_mask = np.isin(neg_ix, crossing_ix)
    pos_ix = pos_ix[pos_mask]
    pos_t = pos_t[pos_mask]
    neg_ix = neg_ix[neg_mask]
    neg_t = neg_t[neg_mask]

    events = dict(crossing_ix=crossing_ix, crossing_t=crossing_t,
                  pos_ix=pos_ix, pos_t=pos_t,
                  neg_ix=neg_ix, neg_t=neg_t)
    return events


def remove_signal_noise(signal: np.ndarray, events: dict, noise_length: float = .001):
    noise_ix, noise_times = [], []
    i = 0
    while i < len(events['crossing_ix']) - 1:
        if (events['crossing_t'][i + 1] - events['crossing_t'][i]) < noise_length:
            noise_ix.append(i)
            noise_ix.append(i + 1)
            noise_times.append(events['crossing_t'][i])
            noise_times.append(events['crossing_t'][i + 1])
            i += 2
        else:
            i += 1

    # while i < len(noise_ix) - 1:
    #     signal[events['crossing_ix'][noise_ix[i]]:events['crossing_ix'][noise_ix[i+1]]] = signal[events['crossing_ix'][noise_ix]-1]
    #     i+=2

    # for i, n in enumerate(noise_ix[:-1]):
    #     signal[events['crossing_ix'][n]] = signal[events['crossing_ix'][n-1]]

    events['crossing_ix'] = np.delete(events['crossing_ix'], noise_ix)
    events['crossing_t'] = np.delete(events['crossing_t'], noise_ix)
    pos_mask = np.isin(events['pos_ix'], events['crossing_ix'])
    neg_mask = np.isin(events['neg_ix'], events['crossing_ix'])
    events['pos_ix'] = events['pos_ix'][pos_mask]
    events['pos_t'] = events['pos_t'][pos_mask]
    events['neg_ix'] = events['neg_ix'][neg_mask]
    events['neg_t'] = events['neg_t'][neg_mask]
    return events#, signal


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


def sync_flipper(daq_events: np.ndarray, flipper_events: np.ndarray, sample_rate: float) -> DAQ:
    # sync the Pi flipper stamps events with the DAQ
    assert daq_events.size == flipper_events.size, "DAQ and flipper events must be the same size"

    ni_synclines = {'flipper_t': daq_events}
    ni = DAQ(sync_lines=ni_synclines, sample_rate=sample_rate)

    ni.sync(sync_from=ni.sync_lines['flipper_t'], sync_to=flipper_events, desc='flipper_t')
    # daq_RT = ni.sync_data(description='flipper_t', data=None)
    return ni


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


def sync_all_devices(imec_file: Path, ni_file: Path, flipper_csv: str, debounce=.0002) -> None:
    # Read imec and NI files
    # imec_events, imec_pos, imec_neg, imec_srate = run_file(binaryFilePath=imec_file, digitalWord=0, digitalLines=[6], tag='imec', debounce=debounce)
    ni_events, ni_pos, ni_neg, ni_srate = run_file(binaryFilePath=ni_file, digitalWord=0, digitalLines=[0], tag='ni_ephys', debounce=debounce)

    # Decode flipper barcodes
    flipper_events = decode_flipper_barcodes(ni_events[0], sample_rate=ni_srate)

    # Read flipper CSV
    df = pd.read_csv(flipper_csv)
    positive_flip_timestamps = df[df['pin_state'] == 1]['time.time()'].to_numpy()
    negative_flip_timestamps = df[df['pin_state'] == 0]['time.time()'].to_numpy()

    # Sync flipper timestamps with DAQ events
    ni = sync_flipper(flipper_events['flipper_pos_t'], positive_flip_timestamps, sample_rate=ni_srate)

    # Print results for verification
    ic(ni.sync_lines)


def match_timestamps(A: np.ndarray, B: np.ndarray, tolerance: float = 0.001) -> [np.ndarray, np.ndarray]:
    """
    Use the intervals between timestamps to match timestamps from two arrays within a specified tolerance.
    :param A: First array of timestamps.
    :param B: Second array of timestamps.
    :param tolerance: Tolerance in seconds for matching.
    :return: arrays of matching timestamps.
    """
    assert np.size(B) <= np.size(A), "Array B must be the same size or smaller than Array A"
    A_mask = np.zeros_like(A, dtype=bool)
    B_mask = np.zeros_like(B, dtype=bool)
    A_diffs = np.diff(A, prepend=A[0])
    B_diffs = np.diff(B, prepend=B[0])

    j = 0
    matched_indices = []
    for i, b in enumerate(B_diffs):
        while j < A_diffs.size:
            a = A_diffs[j]
            if np.abs(b - a) <= tolerance:
                matched_indices.append((j, i))
                A_mask[j] = True
                B_mask[i] = True
                j += 1
                break
            else:
                j += 1

    ix = np.array(matched_indices)
    A = A[ix[:, 0]]
    B = B[ix[:, 1]]
    return A, B


def main_debug():
    raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    processed_data_root = Path.home().joinpath('Documents', 'processed_ephys')

    # run imec file
    # recording_path = raw_data_root.joinpath('CT009_current_20250302/run1_g0')
    # imec_file = recording_path.joinpath('run1_g0_imec0/run1_g0_t0.imec0.ap.bin')
    imec_word = 0
    imec_line = [6]  # [0, 1, 6]

    # imec_events, imec_pos, imec_neg, imec_srate = run_file(binaryFilePath=imec_file, digitalWord=imec_word, digitalLines=imec_line, tag='imec')
    # ic(imec_events[1])

    # run ni file
    # ni_file = recording_path.joinpath('run1_g0_t0.nidq.bin')
    ni_file = raw_data_root / 'flipper_20250805' / 'run1_g0' / 'run1_g0_t0.nidq.bin'
    ni_word = 0
    ni_lines = [0,1,2,3,4,5]   # ephys sync, flipper, left, right, treadmill, IRIG

    flipper_csv = '/home/matt/Documents/RPi_transfer/test-mouse_2025-08-05_182704/test-mouse_2025-08-05_182704_flipper_output.csv'

    ic(ni_file)
    daq_lines, daq_srate = read_digital_lines(ni_file, ni_word, ni_lines)
    daq_ephys = daq_lines[0]
    daq_flipper = daq_lines[1]
    daq_left = daq_lines[2]
    daq_right = daq_lines[3]
    daq_treadmill = daq_lines[4]
    flipper_events = decode_flipper_barcodes(daq_flipper, sample_rate=daq_srate, barcode_present=True)

    df = pd.read_csv(flipper_csv)
    positive_flip_timestamps = df[df['pin_state'] == 1]['time.time()'].to_numpy()
    negative_flip_timestamps = df[df['pin_state'] == 0]['time.time()'].to_numpy()
    ni = sync_flipper(flipper_events['flipper_pos_t'], positive_flip_timestamps, sample_rate=daq_srate)
    daq_pos_flipperT = ni.sync_data(description='flipper_t', data=flipper_events['flipper_pos_t'])
    daq_neg_flipperT = ni.sync_data(description='flipper_t', data=flipper_events['flipper_neg_t'])
    # ic(daq_pos_flipperT, positive_flip_timestamps)
    # ic(daq_neg_flipperT, negative_flip_timestamps)
    np.set_printoptions(precision=12)
    print('negative_flip_timestamps', negative_flip_timestamps)
    print(daq_neg_flipperT)

    # ni_ephys, ni_ephys_pos, ni_ephys_neg, ni_srate = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[0], tag='ni_ephys')
    # ni_flip, ni_flip_pos, ni_flip_neg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[1], tag='ni_flip')
    # ni_Levents, ni_Lpos, ni_Lneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[2], tag='ni_L')
    # ni_Revents, ni_Rpos, ni_Rneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[3], tag='ni_R')
    # ic(ni_Lneg[1])
    # ic(ni_Rneg[1])

    # imec_synclines = {'imec_ix': imec_pos[0], 'imec_t': imec_pos[1]}
    # imec = sync.DAQ(sync_lines=imec_synclines, sample_rate=imec_srate)

    # ni_synclines = {'ephys_ix': ni_ephys_pos[0], 'ephys_t': ni_ephys_pos[1], 'flipper_ix': ni_flip_pos[0], 'flipper_t': ni_flip_pos[1]}
    # ni = sync.DAQ(sync_lines=ni_synclines, sample_rate=ni_srate)

    # ni.sync(sync_from=ni.sync_lines['ephys_ix'], sync_to=imec.sync_lines['imec_ix'], desc='ephys_ix')

    # Lsync = ni.sync_data(description='ephys_ix', data=ni_Levents[0])
    # Rsync = ni.sync_data(description='ephys_ix', data=ni_Revents[0])

    # ic(imec_pos, ni)
    # ic(Lsync.astype(np.int32), Rsync)


def speed():
    """
    Calculate the speed of the mouse based on the treadmill data.
    :return: Speed in ???
    """
    # Placeholder for speed calculation logic
    dacval = volts * 3.3/4095
    maxdacval = 2.5 / 3.3 * 4095  # 2.5V is the max voltage for the treadmill
    analog_speed = (dacval / maxdacval - .5) * 2 * 1000
    pass


def main():
    experiment_folder = Path('/home/matt/Documents/EXPERIMENTS/')
    behavior_data_root = experiment_folder.joinpath('raw_behavior_data')
    ephys_data_root = experiment_folder.joinpath('raw_ephys_data')
    # raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    processed_data_root = experiment_folder / 'processed_data'

    current_mouse = 'CT010'
    current_date_behavior = '2025-08-15'
    current_date_ephys = ''.join(current_date_behavior.split('-'))  # remove dashes for ephys folder name
    sess_timestamp = '125111'
    # sess_id = current_mouse + '_' + current_date
    sess_id_full = current_mouse + '_' + current_date_behavior + '_' + sess_timestamp
    output_path = processed_data_root / current_mouse / sess_id_full

    # imec file
    # recording_path = ephys_data_root.joinpath('CT011_20250624_catgt/catgt_run0_g0/run0_g0_imec0')
    recording_path = ephys_data_root.joinpath('{}_{}_catgt/catgt_run0_g0/run0_g0_imec0'.format(current_mouse, current_date_ephys))
    ap_file = recording_path.joinpath('run0_g0_tcat.imec0.ap.bin')
    lfp_file = recording_path / 'run0_g0_tcat.imec0.lf.bin'
    imec_word = 0
    imec_line = [6]  # [0, 1, 6]
    imec_syncline, imec_srate = read_digital_lines(ap_file, imec_word, imec_line)
    imec_syncline_events = get_flipper_events(imec_syncline, sample_rate=imec_srate)

    sorter_output = recording_path / 'kilosort4_2025-08-18_170053'
    spike_clusters = sorter_output / 'spike_clusters.npy'
    spike_times = sorter_output / 'spike_times.npy'
    spike_clusters = np.load(spike_clusters, allow_pickle=True)
    spike_times = np.load(spike_times, allow_pickle=True) / imec_srate  # raw times are in SAMPLES, must be converted to seconds

    # ni file
    ni_file = ephys_data_root.joinpath('{}_{}_catgt/catgt_run0_g0/run0_g0_t0.nidq.bin'.format(current_mouse, current_date_ephys))
    ni_word = 0
    ni_lines = [0, 1, 2, 3, 4, 5]  # ephys sync, flipper, left, right, treadmill, IRIG
    daq_lines, daq_srate = read_digital_lines(ni_file, ni_word, ni_lines)
    daq_ephys = daq_lines[0]
    daq_flipper = daq_lines[1]

    ni_ephys_events = get_flipper_events(daq_ephys, sample_rate=daq_srate)
    # ni_flipper_events = get_flipper_events(daq_flipper, sample_rate=daq_srate)
    ni_flipper_events, flipper_bounds, (barcodes, barcodes_bitwise) = decode_flipper_barcodes(daq_flipper, sample_rate=daq_srate, barcode_present=True, plot=False)
    ic(barcodes)
    ic(barcodes_bitwise)

    # imec_to_ni = sp.interpolate.interp1d(imec_syncline_events['pos_t'], ni_ephys_events['pos_t'], kind='linear', fill_value='extrapolate')

    # flipper_csv = '/home/matt/Documents/RPi_transfer/CT011_2025-06-24_180128/CT011_2025-06-24_180128_flipper_output.csv'
    flipper_csv = behavior_data_root / sess_id_full / '{}_flipper_output.csv'.format(sess_id_full)

    pi_flipper_df = pd.read_csv(flipper_csv)
    positive_flip_timestamps = pi_flipper_df[pi_flipper_df['pin_state'] == 1]['time.time()'].to_numpy()
    negative_flip_timestamps = pi_flipper_df[pi_flipper_df['pin_state'] == 0]['time.time()'].to_numpy()
    ni_flipper_events['flipper_pos_t'], positive_flip_timestamps = match_timestamps(ni_flipper_events['flipper_pos_t'], positive_flip_timestamps)
    ni_flipper_events['flipper_neg_t'], negative_flip_timestamps = match_timestamps(ni_flipper_events['flipper_neg_t'], negative_flip_timestamps)

    ni_to_pi = sp.interpolate.interp1d(ni_flipper_events['flipper_pos_t'], positive_flip_timestamps, kind='linear', fill_value='extrapolate')
    imec_to_ni = sp.interpolate.interp1d(imec_syncline_events['pos_t'], ni_ephys_events['pos_t'], kind='linear', fill_value='extrapolate')
    imec_to_pi = lambda x: ni_to_pi(imec_to_ni(x))

    daq_neg_flipperT = ni_to_pi(ni_flipper_events['flipper_neg_t'])

    imec_middle_eventix = (imec_syncline_events['neg_t'] > 1000) & (imec_syncline_events['neg_t'] < 2000)
    imec_middle_event_t = imec_syncline_events['neg_t'][imec_middle_eventix]
    imec_RT = imec_to_pi(imec_middle_event_t)
    time.gmtime(imec_RT[0])

    np.set_printoptions(precision=12)
    print('negative_flip_timestamps')
    # ic(daq_pos_flipperT, positive_flip_timestamps)
    ic(daq_neg_flipperT, negative_flip_timestamps)
    # print(negative_flip_timestamps)
    # print(daq_neg_flipperT)

    spike_times_ni = imec_to_ni(spike_times)
    spike_times_mask = (spike_times_ni > flipper_bounds[0]) & (spike_times_ni < flipper_bounds[1])
    aligned_spike_times_bounded = imec_to_pi(spike_times[spike_times_mask])
    aligned_spike_clusters_bounded = spike_clusters[spike_times_mask]

    cluster_ids = np.unique(aligned_spike_clusters_bounded)
    cluster_dict = {}
    for id in cluster_ids:
        cluster_dict[id] = aligned_spike_times_bounded[aligned_spike_clusters_bounded == id]

    # ic(cluster_dict)

    # Save the aligned spike times and clusters
    aligned_spike_times_file = output_path / '{}_aligned_spike_times.npy'.format(sess_id_full)
    aligned_spike_clusters_file = output_path / '{}_aligned_spike_clusters.npy'.format(sess_id_full)
    cluster_dict_file = output_path / '{}_cluster_dict.npy'.format(sess_id_full)
    np.save(aligned_spike_times_file, aligned_spike_times_bounded)
    np.save(aligned_spike_clusters_file, aligned_spike_clusters_bounded)
    np.save(cluster_dict_file, cluster_dict)


if __name__ == '__main__':
    main()

