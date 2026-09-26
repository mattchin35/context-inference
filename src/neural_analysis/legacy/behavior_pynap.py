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
from typing import Any
from pandas.core.sample import process_sampling_size

import src.mouse_behavior_preprocessing.process_treadmill as process_treadmill


def _validate_event_df(event_df: pd.DataFrame) -> None:
    required_cols = {'Time', 'Event'}
    missing = required_cols - set(event_df.columns)
    if missing:
        raise ValueError(f"event_df is missing required columns: {sorted(missing)}")


def _ts_from_events(event_df: pd.DataFrame, event_mask: pd.Series) -> nap.Ts:
    times = pd.to_numeric(event_df.loc[event_mask, 'Time'], errors='coerce').dropna().to_numpy(dtype=float)
    if times.size:
        times = np.sort(times)
    return nap.Ts(t=times)


def prepare_trial_starts(event_df: pd.DataFrame) -> nap.Ts:
    _validate_event_df(event_df)
    return _ts_from_events(event_df, event_df['Event'] == 'trial_start')


def prepare_led_on(event_df: pd.DataFrame) -> nap.Ts:
    _validate_event_df(event_df)
    return _ts_from_events(event_df, event_df['Event'] == 'LED_on')


def prepare_led_off(event_df: pd.DataFrame) -> nap.Ts:
    _validate_event_df(event_df)
    return _ts_from_events(event_df, event_df['Event'] == 'LED_off')


def prepare_choices(event_df: pd.DataFrame) -> nap.TsGroup:
    _validate_event_df(event_df)
    choice_pat = r'^(correct|wrong)_choice_(left|right)_patch$'
    choice_events = event_df.loc[event_df['Event'].str.match(choice_pat, na=False), 'Event'].unique().tolist()
    choice_events = sorted(choice_events)
    return nap.TsGroup({event_name: _ts_from_events(event_df, event_df['Event'] == event_name) for event_name in choice_events})


def prepare_lick_times(event_df: pd.DataFrame) -> nap.TsGroup:
    _validate_event_df(event_df)
    lick_events = ['left_entry', 'right_entry']
    return nap.TsGroup({event_name: _ts_from_events(event_df, event_df['Event'] == event_name) for event_name in lick_events})


def prepare_reward_deliveries(event_df: pd.DataFrame) -> nap.TsGroup:
    _validate_event_df(event_df)
    reward_pat = r'^pump\d+_reward'
    reward_mask = event_df['Event'].str.match(reward_pat, na=False)
    reward_events = event_df.loc[reward_mask, 'Event'].unique().tolist()
    reward_events = sorted(reward_events)
    return nap.TsGroup({event_name: _ts_from_events(event_df, event_df['Event'] == event_name) for event_name in reward_events})


def build_behavior_pynapple(event_df: pd.DataFrame) -> dict[str, Any]:
    """
    Build pynapple behavior structures from event_df.
    """
    _validate_event_df(event_df)
    return {
        'trial_starts': prepare_trial_starts(event_df),
        'led_on': prepare_led_on(event_df),
        'led_off': prepare_led_off(event_df),
        'choices': prepare_choices(event_df),
        'lick_times': prepare_lick_times(event_df),
        'reward_deliveries': prepare_reward_deliveries(event_df),
    }


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
    behavior_nap = build_behavior_pynapple(event_df)


if __name__ == "__main__":
    main()
