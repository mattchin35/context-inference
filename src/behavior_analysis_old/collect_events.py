import pandas as pd
import numpy as np
import re


def generate_event_array(df: pd.DataFrame, events: list, timespan: tuple) -> np.ndarray:
    event_array = []
    for e in events:
        event_df = df.loc[df['Event'] == e, 'Time']
        ix = (event_df >= timespan[0]) & (event_df <= timespan[1])
        event_array.append(event_df[ix].values)

    return np.array(event_array, dtype=object)


def get_choice_events(df: pd.DataFrame, event_list: list=[]) -> list:
    unique_events = np.unique(df['Event'].values)
    for e in unique_events:
        # if re.fullmatch('pump.*', e):
        if re.fullmatch('.*choice.*', e):
            event_list.append(e)

        elif re.fullmatch('giving_reward.*', e):
            event_list.append(e)

    return event_list


def get_context_events(df: pd.DataFrame, event_list: list=[]) -> list:
    unique_events = np.unique(df['Event'].values)
    for e in unique_events:
        if re.fullmatch('enter_.*', e):
        # if re.fullmatch('enter_.*', e) or re.fullmatch('exit_.*', e):
            event_list.append(e)

        elif re.fullmatch('stimulus_.*_.*', e):
            event_list.append(e)

        elif re.fullmatch('trial_.*', e):
            event_list.append(e)

    return event_list


def get_reward_events(df: pd.DataFrame, event_list: list=[]) -> list:
    unique_events = np.unique(df['Event'].values)
    for e in unique_events:
        if re.fullmatch('pump.*', e):
            event_list.append(e)

    return event_list


def get_stimulus_events(df: pd.DataFrame, event_list: list=[]) -> list:
    unique_events = np.unique(df['Event'].values)
    for e in unique_events:
        if re.fullmatch('LED_.*', e):
            event_list.append(e)

        # elif re.fullmatch('stimulus_.*', e):
        #     event_list.append(e)
        
    return event_list


def make_event_labels(events: list) -> list:
    event_labels = [s.replace('_', ' ') for s in events]
    event_labels = [s.replace('pump1', 'left') for s in event_labels]
    event_labels = [s.replace('pump2', 'right') for s in event_labels]
    return event_labels

