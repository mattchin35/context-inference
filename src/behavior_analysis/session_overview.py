import pandas as pd
import numpy as np
import fileIO
import re
from typing import Tuple
from pathlib import Path
import pickle as pkl
import collect_events

"""
Functions to prepare a single behavior session for analysis.
"""


def total_water_delivery(session_df: pd.DataFrame) -> pd.DataFrame:
    reward_event_types = collect_events.get_choice_events(session_df)
    ix = np.zeros(session_df.shape[0]).astype(bool)
    for e in reward_event_types:
        ix = ix | (session_df['Event'] == e)

    # create lists of reward events of each type
    reward_events = session_df.loc[ix, 'Event'].values
    pump1_regex = re.compile("pump1.*")
    pump2_regex = re.compile("pump2.*")
    pump1_list = list(filter(pump1_regex.match, reward_events))  # Read Note below
    pump2_list = list(filter(pump2_regex.match, reward_events))  # Read Note below

    # add up events to determine total rewards
    pump1_water = np.sum([float(re.findall(r'(\d+)', a)[1]) for a in pump1_list])
    pump2_water = np.sum([float(re.findall(r'(\d+)', a)[1]) for a in pump2_list])
    total_water = pump1_water + pump2_water
    source = ['right water', 'left water', 'total water']
    amount = [pump1_water, pump2_water, total_water]

    session_water = pd.DataFrame(dict(water_source=source, amount=amount))
    return session_water


def choice_event_summary(event) -> Tuple[int, int, int]:
    # LEFT CHOICES
    if event == 'wrong_choice_right_patch':
        action = 1
        correct = 0
        reward = 0
    elif event == 'pump2_reward_0':
        action = 1
        correct = 1
        reward = 0
    elif re.fullmatch('pump2.*', event) or event == 'correct_choice_right_patch':
        action = 1
        correct = 1
        reward = 1

    # RIGHT CHOICES
    elif event == 'wrong_choice_left_patch':
        action = 0
        correct = 0
        reward = 0
    elif event == 'pump1_reward_0':
        action = 0
        correct = 1
        reward = 0
    elif re.fullmatch('pump1.*', event) or event == 'correct_choice_left_patch':
        action = 0
        correct = 1
        reward = 1

    else:
        raise NameError('Unrecognized choice event: {}'.format(event))

    return action, correct, reward


def iterate_trials(raw_data: pd.DataFrame, context_events: list, choice_events: list) -> pd.DataFrame:
    """
    Iterate through the raw data to obtain the full description of each trial.
    """
    cur_state = -1
    cur_block = -1
    cur_trial_in_block = -1
    cur_trial = -1
    _action = -1
    _correct = -1
    _reward = 0

    states = []
    trials = []
    trials_in_block = []
    blocks = []
    actions = []
    correct = []
    rewards = []

    # go through each event and build up lists of the above variables
    for i, e in enumerate(raw_data['Event'].values):
        if e in context_events:  # update the background conditions, but do not append to lists
            cur_state = e
            cur_block += 1
            cur_trial_in_block = -1

        elif e in choice_events:  # update the trial and append to lists
            cur_trial += 1
            cur_trial_in_block += 1
            action, _correct, _reward = choice_event_summary(e)

            states.append(cur_state)
            trials.append(cur_trial)
            trials_in_block.append(cur_trial_in_block)
            blocks.append(cur_trial_in_block)
            actions.append(action)
            correct.append(_correct)
            rewards.append(_reward)

    assert -1 not in actions, "Action list contains -1s, which means there are unaccounted for events."

    states = np.array(states)
    states_int = np.zeros(states.size)
    states_int[states == 'enter_right_patch'] = 0
    states_int[states == 'enter_left_patch'] = 1
    states_int.astype(int)

    event_df = pd.DataFrame({'state': states, 'state_int': states_int, 'cur_trial': trials,
                             'cur_trial_in_block': trials_in_block, 'cur_block': blocks, 'action': actions,
                             'correct': correct, 'reward': rewards})
    return event_df


def make_event_df(cleaned_data: pd.DataFrame, session_id: str, save_name: str, output_path: Path) -> pd.DataFrame:
    """
    Prepare an event dataframe with describing each trial of a session.
    Should be compatible with computational agents.
    Collect state, cur_trial, cur_block, cur_trial_in_block,
    p_active_rew, p_inactive_rew,
    stimulus, action, correct, reward, session_ID
    """
    df = cleaned_data.sort_values(by=['Time'])
    # unique_events = df['Event'].unique()
    context_events = collect_events.get_context_events(df)
    choice_events = collect_events.get_choice_events(df)

    # start from the first known context entry; throw out everything before that
    ix = np.zeros(df.shape[0]).astype(bool)
    for e in context_events:
        ix = ix | (df['Event'] == e)

    start_time = df.loc[ix, 'Time'].values[0]
    df = df.loc[df['Time'] >= start_time]

    p_active_rew, p_inactive_rew = .9, 0
    event_df = iterate_trials(df, context_events, choice_events)

    # add in variables which are constant across all trials or will be updated as the behavior task is updated
    n_trials = len(event_df['state'])
    stimulus = [None] * n_trials  # np.zeros(n_trials) - 1 # no stimulus
    p_active_rew = np.ones(n_trials) * p_active_rew
    p_inactive_rew = np.ones(n_trials) * p_inactive_rew
    session_id = [session_id] * n_trials

    event_df['p_active_rew'] = p_active_rew
    event_df['p_inactive_rew'] = p_inactive_rew
    event_df['stimulus'] = stimulus
    event_df['session_ID'] = session_id

    p = output_path / (save_name + '.pkl')
    with p.open('wb') as f:
        pkl.dump(event_df, f)

    print("[***] Experiment saved as: {}".format(p.name))
    return event_df


if __name__ == '__main__':
    plot_path = Path('../../reports/figures')
    data_path = Path('../../data/processed')

    """Analyze the data from a single mouse session"""
    current_mouse = 'MF03'  # 'MF23'
    current_date = '2023-10-02'  # '2023-08-18'
    sess_ID = current_mouse + '_' + current_date

    cleaned_data = fileIO.load_cleaned_data(sess_ID, data_path)

    # rp.generate_session_raster(df, sess_ID + '_raster', plot_path)
    # water = total_water_delivery(df)
    # print(water)

    event_df = make_event_df(cleaned_data=cleaned_data, session_id=sess_ID,
                             save_name=sess_ID + '_performance', output_path=data_path)
    # bin_counts, session_df, bins = plot_binned_behavior(df,  plot_path, sess_ID + '_bin_plot', plot=True)
    # block_analysis(bin_counts, session_df, sess_ID)

    """Analyze data of a mouse across several sessions"""
    # sess_list = [('MF24', '2023-07-14'),
    #              ('MF24', '2023-07-17'),
    #              ('MF24', '2023-07-18'),
    #              ('MF24', '2023-07-19')]
    # compare_sessions(sess_list, plot_path, plot_name='MF24_sample')
