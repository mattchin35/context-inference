import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import pickle as pkl


def plot_hist():
    pass


def main():
    session_data_home = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
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

    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    augmented_trial_df = pd.read_csv(augmented_trial_df_path, sep=',', na_filter=False)

    val = augmented_trial_df['RFLR_prob_left']
    val = val[val != 'None'].values.astype(float)

    # reward_time = augmented_trial_df['reward_time']
    # reward_time = reward_time[reward_time != 'None'].values.astype(float)
    # HMM_rel_val = augmented_trial_df['HMM_rel_value']
    # HMM_rel_val = HMM_rel_val[HMM_rel_val != 'None'].values.astype(float)
    # HMM_prob_left = augmented_trial_df['HMM_prob_left']
    # HMM_prob_left = HMM_prob_left[HMM_prob_left != 'None'].values.astype(float)

    sns.histplot(data=val)
    plt.show()
    pass


if __name__ == '__main__':
    main()

