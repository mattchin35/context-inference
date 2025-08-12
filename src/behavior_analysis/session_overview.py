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

states = ['right', 'left']
state_dict = {s: i for i, s in enumerate(states)}  # i.e. [0 right, 1 left]

# pump1 = left, pump2 = right


def total_water_delivery(session_df: pd.DataFrame) -> pd.DataFrame:
    """This fxn tries to calculate water delivery based on an older version of the event dataframe"""
    reward_event_types = collect_events.get_reward_events(session_df)
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
    # pump1_water = np.sum([float(re.findall(r'(\d+)', a)[1]) for a in pump1_list])
    # pump2_water = np.sum([float(re.findall(r'(\d+)', a)[1]) for a in pump2_list])
    pump1_water = np.sum([int(re.findall(r'reward_amount: (\d+)', a)[0]) for a in pump1_list])
    pump2_water = np.sum([int(re.findall(r'reward_amount: (\d+)', a)[0]) for a in pump2_list])
    total_water = pump1_water + pump2_water
    source = ['left water', 'right water', 'total water']
    amount = [pump1_water, pump2_water, total_water]

    session_water = pd.DataFrame(dict(water_source=source, amount=amount))
    return session_water


def choice_event_summary(event, nearby_events) -> Tuple[int, int, int]:

    ### NOT IN USE ###
    # LEFT CHOICES
    # if event == 'wrong_choice_right_patch':
    #     action = 1
    #     correct = 0
    #     reward = 0
    # elif event == 'pump2_reward_0':
    #     action = 1
    #     correct = 1
    #     reward = 0
    # elif re.fullmatch('pump2.*', event) or event == 'correct_choice_right_patch':
    #     action = 1
    #     correct = 1
    #     reward = 1
    #
    # # RIGHT CHOICES
    # elif event == 'wrong_choice_left_patch':
    #     action = 0
    #     correct = 0
    #     reward = 0
    # elif event == 'pump1_reward_0':
    #     action = 0
    #     correct = 1
    #     reward = 0
    # elif re.fullmatch('pump1.*', event) or event == 'correct_choice_left_patch':
    #     action = 0
    #     correct = 1
    #     reward = 1
    ### END NOT IN USE ###

    ## FOR GOOD CODE AFTER 9/2/24
    # LEFT CHOICES
    if event == 'wrong_choice_right_patch':
        action = 1
        correct = 0
        reward = 0
    elif event == 'correct_choice_left_patch':
        action = 1
        correct = 1
        reward = 0
        for ne in nearby_events:
            try:
                if int(re.findall(r'reward_amount: (\d+)', ne)[0]) > 0:
                    reward = 1
                    break
            except IndexError:
                continue

    # RIGHT CHOICES
    elif event == 'wrong_choice_left_patch':
        action = 0
        correct = 0
        reward = 0
    elif event == 'correct_choice_right_patch':
        action = 0
        correct = 1
        reward = 0
        for ne in nearby_events:
            try:
                if int(re.findall(r'reward_amount: (\d+)', ne)[0]) > 0:
                    reward = 1
                    break
            except IndexError:
                continue

    ## FOR BAD CODE BEFORE 9/2/24
    # LEFT CHOICES
    # if event == 'wrong_choice_right_patch':
    #     action = 0
    #     correct = 0
    #     reward = 0
    # elif event == 'correct_choice_right_patch':
    #     action = 0
    #     correct = 1
    #     reward = 0
    #     for ne in nearby_events:
    #         try:
    #             if int(re.findall(r'reward_amount: (\d+)', ne)[0]) > 0:
    #                 reward = 1
    #                 break
    #         except IndexError:
    #             continue
    # elif event == 'correct_choice_left_patch':
    #     action = 1
    #     correct = 1
    #     reward = 0
    #     for ne in nearby_events:
    #         try:
    #             if int(re.findall(r'reward_amount: (\d+)', ne)[0]) > 0:
    #                 reward = 1
    #                 break
    #         except IndexError:
    #             continue
    # elif event == 'wrong_choice_left_patch':
    #     action = 1
    #     correct = 0
    #     reward = 0

    else:
        raise NameError('Unrecognized choice event: {}'.format(event))

    return action, correct, reward


def iterate_trials(raw_data: pd.DataFrame, context_events: list, choice_events: list, reward_events: list) -> pd.DataFrame:
    """
    Iterate through the events to obtain full descriptions of each trial.
    """
    cur_state = None
    cur_state_int = -1
    cur_block = -1
    cur_trial_in_block = -1
    cur_trial = -1
    cur_stimulus = None
    block_stimulus = None
    _action = -1
    _correct = -1
    _reward = 0

    states = []
    trials = []
    trials_in_block = []
    blocks = []
    states_int = []
    actions = []
    correct = []
    rewards = []
    active_stimuli = []
    block_stimuli = []
    time = []

    # go through each event and build up lists of the above variables
    for i, e in enumerate(raw_data['Event'].values):
        cur_time = raw_data['Time'].values[i]

        if e in context_events:  # update the background conditions, but do not append to lists
            if e == 'enter_right_patch':
                cur_state = 'right_patch'
                cur_state_int = 0
                cur_block += 1
                cur_trial_in_block = -1
            elif e == 'enter_left_patch':
                cur_state = 'left_patch'
                cur_state_int = 1
                cur_block += 1
                cur_trial_in_block = -1
            elif e == 'enter_dark_period':
                cur_state = 'dark_period'
                cur_state_int = 2
                cur_stimulus = None
                block_stimulus = None
            elif e == 'stimulus_A_on':
                cur_stimulus = 'A'
                block_stimulus = 'A'
            elif e == 'stimulus_A_off':
                cur_stimulus = None
                # block stimulus remains A
            elif e == 'stimulus_B_on':
                cur_stimulus = 'B'
                block_stimulus = 'B'
            elif e == 'stimulus_B_off':
                cur_stimulus = None
                # block stimulus remains B
            elif e == 'stimulus_C_on':
                pass  # for C, I might just leave the stimulus as None
                # cur_stimulus = 'C'
                # block_stimulus = 'C'
            else:
                raise NameError('Unrecognized context event: {}'.format(e))

        elif e in choice_events:  # update the trial and append to lists
            cur_trial += 1
            cur_trial_in_block += 1
            nearby_events = (raw_data['Time'] > cur_time) & (raw_data['Time'] < cur_time + .5)
            nearby_events = raw_data.loc[nearby_events, 'Event'].values
            _action, _correct, _reward = choice_event_summary(e, nearby_events)

            states.append(cur_state)
            states_int.append(cur_state_int)
            trials.append(cur_trial)
            trials_in_block.append(cur_trial_in_block)
            blocks.append(cur_block)
            actions.append(_action)
            correct.append(_correct)
            rewards.append(_reward)
            active_stimuli.append(cur_stimulus)
            block_stimuli.append(block_stimulus)
            time.append(cur_time)

    assert -1 not in actions, "Action list contains -1s, which means there are unaccounted for events."

    states = np.array(states)
    states_int = np.array(states_int)
    event_df = pd.DataFrame({'state': states, 'state_int': states_int, 'cur_trial': trials,
                             'cur_trial_in_block': trials_in_block, 'cur_block': blocks, 'action': actions,
                             'correct': correct, 'reward': rewards,
                             'active_stimulus': active_stimuli, 'block_stimulus': block_stimuli,
                             'time': time})
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
    context_events = collect_events.get_context_events(df)
    choice_events = collect_events.get_choice_events(df)
    reward_events = collect_events.get_reward_events(df)

    # start from the first known context entry; throw out everything before that
    ix = np.zeros(df.shape[0]).astype(bool)
    for e in context_events:
        ix = ix | (df['Event'] == e)

    start_time = df.loc[ix, 'Time'].values[0]
    df = df.loc[df['Time'] >= start_time]

    p_active_rew, p_inactive_rew = .9, 0  # will want to load from session_info in the future
    event_df = iterate_trials(df, context_events, choice_events, reward_events)

    # add in variables which are constant across all trials or will be updated as the behavior task is updated
    n_trials = len(event_df['state'])
    p_active_rew = np.ones(n_trials) * p_active_rew
    p_inactive_rew = np.ones(n_trials) * p_inactive_rew
    session_id = [session_id] * n_trials

    event_df['p_active_rew'] = p_active_rew
    event_df['p_inactive_rew'] = p_inactive_rew
    event_df['session_ID'] = session_id

    if not output_path.exists():
        output_path.mkdir()

    p = output_path / (save_name + '.pkl')
    with p.open('wb') as f:
        pkl.dump(event_df, f)

    print("[***] Experiment saved as: {}".format(p.name))
    return event_df


def load_event_df(session_id: str, data_path: Path) -> pd.DataFrame:
    p = data_path / (session_id + '_events.pkl')
    with p.open('rb') as f:
        event_df = pkl.load(f)
    return event_df


if __name__ == '__main__':
    plot_path = Path('../../reports/figures')
    data_path = Path('../../data/processed')
    
    """Analyze the data from a single mouse session"""
    current_mouse = 'HD005'  # 'MF23'
    current_date = '2024-08-26'  # '2023-08-18'
    sess_ID = current_mouse + '_' + current_date
    # cleaned_data = fileIO.load_raw_data(sess_ID, data_path / 'cleaned_sessions')
    cleaned_data = fileIO.load_cleaned_data(sess_ID, data_path / 'cleaned_sessions')
    event_df = make_event_df(cleaned_data=cleaned_data, session_id=sess_ID,
                              save_name=sess_ID + '_events',
                              output_path=data_path / current_mouse)
    # some old events files came from here with the suffix _performance

    event_df = load_event_df(sess_ID, data_path / current_mouse)
    water = total_water_delivery(cleaned_data)
    print(water)

    # bin_counts, session_df, bins = plot_binned_behavior(df,  plot_path, sess_ID + '_bin_plot', plot=True)
    # block_analysis(bin_counts, session_df, sess_ID)

    """Analyze data of a mouse across several sessions"""
    # sess_list = [('MF24', '2023-07-14'),
    #              ('MF24', '2023-07-17'),
    #              ('MF24', '2023-07-18'),
    #              ('MF24', '2023-07-19')]
    # compare_sessions(sess_list, plot_path, plot_name='MF24_sample')

