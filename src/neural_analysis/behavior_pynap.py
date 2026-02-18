"""
Prepare behavior trials as pynapple objects. Neural data will be added later.
"""

import pickle as pkl
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from pandas.core.sample import process_sampling_size

import src.mouse_behavior_preprocessing.process_treadmill as process_treadmill


def main():
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
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

    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',')
    treadmill_df = pd.read_csv(processed_data_path / (sess_id_full + '_treadmill.csv'))
    runspeed_df = process_treadmill.gather_runspeed(treadmill_df, plot=False)


if __name__ == "__main__":
    main()
