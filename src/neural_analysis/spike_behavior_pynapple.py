import re
import time
from pathlib import Path
import pickle as pkl
from dataclasses import dataclass

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score, StratifiedKFold, StratifiedShuffleSplit
import pynapple as nap
import src.external_tools.readSGLX as readSGLX


@dataclass
class Session:
    multi_session_save_path = Path.home()
    session_data_home = Path.home()
    sess_id_full = 'mouseid_YYYY-MM-DD_hhmmss'
    sess_id_abbreviated = 'mouseid_abbreviated'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'
    mouse = 'test-mouse'
    date = '1970-01-01'
    timestamp = '000000'
    session_info_fname = '{}_session_info.pkl'.format(sess_id_full)
    session_info = None
    pfc_spike_path = None
    hpc_spike_path = None


def main():
    multi_session_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    pfc_spike_path = session_data_home / 'ephys/catgt/catgt_run0_g0/run0_g0_imec0/Kilosort2.5.2_2026-03-18_115111'
    hpc_spike_path = session_data_home / 'ephys/catgt/catgt_run0_g0/run0_g0_imec1/Kilosort2.5.2_2026-03-18_122341'
    assert pfc_spike_path.exists(), "PFC spikes at at {} not found!".format(hpc_spike_path)
    assert hpc_spike_path.exists(), "HPC spikes at at {} not found!".format(hpc_spike_path)

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

    sess = Session()
    sess.multi_session_save_path = multi_session_save_path
    sess.session_data_home = session_data_home
    sess.sess_id_full = sess_id_full
    sess.sess_id_abbreviated = sess_id_abbreviated
    sess.raw_behavior_folder = raw_behavior_folder
    sess.processed_data_path = processed_data_path
    sess.figure_path = figure_path
    sess.mouse = mouse
    sess.date = date
    sess.timestamp = timestamp
    sess.session_info_fname = session_info_path
    sess.session_info = session_info

    # load processed raw data
    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',')
    block_performance_df = pd.read_csv(processed_data_path / (sess_id_full + '_block_performance.csv'), sep=',', na_filter=False)
    augmented_trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), sep=',', na_filter=False)
    multisession_df = pd.read_csv(multi_session_save_path / (mouse + '_overall_performance.csv'), sep=',', na_filter=False)


if __name__ == "__main__":
    main()