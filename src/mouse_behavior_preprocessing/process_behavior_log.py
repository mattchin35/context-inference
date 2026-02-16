import pandas as pd
import numpy as np
import re
from typing import Optional
from pathlib import Path
import pickle as pkl
from src.behavior_analysis import collect_events
from collections import OrderedDict


"""
Functions to prepare a single behavior session for analysis.
"""

states = ['right', 'left']
state_dict = {s: i for i, s in enumerate(states)}  # i.e. [0 right, 1 left]

# pump1 = left, pump2 = right
def calculate_water_delivery(session_df: pd.DataFrame, session_info: dict) -> pd.DataFrame:
    reward_event_types = collect_events.get_reward_events(session_df)
    ix = np.zeros(session_df.shape[0]).astype(bool)
    for e in reward_event_types:
        ix = ix | (session_df['Event'] == e)

    # create lists of reward events of each type
    if session_info['ephys_rig']:
        right_pump_regex = re.compile("pump3.*")
        left_pump_regex = re.compile("pump2.*")
    else:
        right_pump_regex = re.compile("pump2.*")
        left_pump_regex = re.compile("pump1.*")
    reward_events = session_df.loc[ix, 'Event'].values

    right_list = list(filter(right_pump_regex.match, reward_events))
    left_list = list(filter(left_pump_regex.match, reward_events))

    right_water = np.sum([int(re.findall(r'reward_amount: (\d+)', a)[0]) for a in right_list])
    left_water = np.sum([int(re.findall(r'reward_amount: (\d+)', a)[0]) for a in left_list])
    total_water = right_water + left_water

    source = ['left water', 'right water', 'total water']
    amount = [left_water, right_water, total_water]
    session_water = pd.DataFrame(dict(water_source=source, amount=amount))
    return session_water


def process_file(save_directory: Path, file_path: str, filter_events: Optional[list[str]] = []) -> pd.DataFrame:
    # Read the file into a DataFrame, skipping the first row
    df = pd.read_csv(file_path, sep=';', header=None, on_bad_lines='skip', usecols=[1, 3, 4])

    # Rename the columns
    df.columns = ['Time', 'Event', 'Note']

    # Subtract 'exit_standby' time value from every element
    if 'exit_standby' in df['Event'].values:
        exit_standby_time = df.loc[df['Event'] == 'exit_standby', 'Time'].values[0]
    else:
        exit_standby_time = df['Time'].min()

    # Remove any rows with times before session start
    df = df[df['Time'] - exit_standby_time >= 0]  # use this to keep unix times - needed for synchronization with ephys + treadmill
    df.reset_index(inplace=True, drop=True)
    assert df.iloc[0]['Event'] == 'exit_standby', "First event should be 'exit_standby'"
    df = df[df['Time'] - exit_standby_time > 0]  # once you have the relevant events, throw out exit-standby to keep in-session bits

    # Could replace event names or extract specific keys to filter events dictionary - deleted here, but available
    # in "behavior_analysis_old/fileIO" if I need that functionality back

    # Save the new DataFrame to a CSV file
    if not save_directory.exists():
        save_directory.mkdir(parents=True)

    new_file_path = save_directory / (Path(file_path).stem + '_events.csv')
    df.to_csv(new_file_path, index=False)
    print('cleaned_file_created')
    return df


def choice_event_summary(event: str, reward_note: str) -> tuple[int, int, int, int]:
    # LEFT CHOICES
    if event == 'wrong_choice_right_patch':
        action = 1  # state_dict['left']
        correct = 0
        reward = 0
        give_reward = 0
    elif event == 'correct_choice_left_patch':
        action = 1  # state_dict['left']
        correct = 1
        reward = 0
        give_reward = 0

    # RIGHT CHOICES
    elif event == 'wrong_choice_left_patch':
        action = 0  # state_dict['right']
        correct = 0
        reward = 0
        give_reward = 0
    elif event == 'correct_choice_right_patch':
        action = 0  # state_dict['right']
        correct = 1
        reward = 0
        give_reward = 0

    elif event == 'giving_reward_left_patch':
        # action = 'None'
        action = 1
        correct = 0
        reward = 1
        give_reward = 1
    elif event == 'giving_reward_right_patch':
        # action = 'None'
        action = 0
        correct = 0
        reward = 1
        give_reward = 1

    else:
        raise NameError('Unrecognized choice event: {}'.format(event))

    if reward_note == 'reward_True':
        reward = 1
    elif reward_note == 'reward_False':
        pass
    else:
        raise NameError('Unrecognized reward outcome note: {}'.format(reward_note))

    return action, correct, reward, give_reward


def iterate_trials(raw_data: pd.DataFrame, context_events: list, choice_events: list, reward_events: list,
                   stimulus_events:list) -> pd.DataFrame:
    """
    Iterate through the events to obtain full descriptions of each trial.
    Refactor this later to separate the different if/else cases for code cleanliness.
    """
    cur_state = 'None'
    cur_state_int = -1
    cur_block = -1
    cur_trial_in_block = -1
    cur_trial = -1
    cur_stimulus = 'None'
    block_stimulus = 'None'
    _action = 'None'
    _correct = 0
    _reward = 0

    trial_dict = None
    trial_list = []
    # refactor to have OrderedDict out here with the above qualities, updates below are applied to
    # the OrderedDict entries

    for i, e in enumerate(raw_data['Event'].values):
        cur_time = raw_data['Time'].values[i]

        if e in context_events:  # update the background conditions, but do not append to lists
            # refactor as if in context events, call context_event_handler(event) that returns trial_dict, trial_list
            if e == 'enter_right_patch':
                cur_state = 'right_patch'
                cur_state_int = 0
                cur_block += 1
                cur_trial_in_block = -1
                cur_stimulus = 'None'
                block_stimulus = 'None'
            elif e == 'enter_left_patch':
                cur_state = 'left_patch'
                cur_state_int = 1
                cur_block += 1
                cur_trial_in_block = -1
                cur_stimulus = 'None'
                block_stimulus = 'None'
            elif e == 'enter_dark_period':
                cur_state = 'dark_period'
                cur_state_int = 2
                cur_stimulus = 'None'
                block_stimulus = 'None'
            elif e == 'trial_start':
                if trial_dict is not None:
                    trial_list.append(trial_dict)

                cur_trial += 1
                cur_trial_in_block += 1
                # trial_dict = OrderedDict(state=cur_state, state_int=cur_state_int,
                #                          cur_trial=cur_trial, cur_trial_in_block=cur_trial_in_block,
                #                          cur_block=cur_block, action=None, correct=None, reward=None,
                #                          active_stimulus=cur_stimulus, block_stimulus=block_stimulus,
                #                          start_time=cur_time, choice_time=None, reward_time=None,
                #                          led_on_time=None, led_off_time=None)
                trial_dict = OrderedDict(state=cur_state, state_int=cur_state_int,
                                         cur_trial=cur_trial, cur_trial_in_block=cur_trial_in_block, cur_block=cur_block,
                                         action='None', correct='None', reward='None', give_reward='None',
                                         active_stimulus=cur_stimulus, block_stimulus=block_stimulus,
                                         start_time=cur_time, choice_time='None', reward_time='None',
                                         led_on_time='None', led_off_time='None')

            elif e == 'trial_stop':
                pass

            else:
                raise NameError('Unrecognized context event: {}'.format(e))

        elif e in choice_events:  # update the trial and append to lists
            trial_dict['choice_time'] = cur_time

            ## FOR GOOD CODE AFTER 9/2/24
            # LEFT CHOICES
            if e == 'wrong_choice_right_patch':
                _action = 1 #state_dict['left']
                _correct = 0
                _reward = 0
                _give_reward = 0
            elif e == 'correct_choice_left_patch':
                _action = 1 #state_dict['left']
                _correct = 1
                _reward = 0
                _give_reward = 0

            # RIGHT CHOICES
            elif e == 'wrong_choice_left_patch':
                _action = 0  #state_dict['right']
                _correct = 0
                _reward = 0
                _give_reward = 0
            elif e == 'correct_choice_right_patch':
                _action = 0  #state_dict['right']
                _correct = 1
                _reward = 0
                _give_reward = 0

            elif e == 'giving_reward_left_patch':
                # _action = 'None'
                _action = 1
                _correct = 0
                _reward = 1
                _give_reward = 1
            elif e == 'giving_reward_right_patch':
                # _action = 'None'
                _action = 0
                _correct = 0
                _reward = 1
                _give_reward = 1

            else:
                raise NameError('Unrecognized choice event: {}'.format(e))

            reward_outcome = raw_data['Note'].values[i]
            if reward_outcome == 'reward_True':
                _reward = 1
            elif reward_outcome == 'reward_False':
                _reward = 0
            else:
                raise NameError('Unrecognized reward outcome note: {}'.format(reward_outcome))

            trial_dict['action'] = _action
            trial_dict['correct'] = _correct
            trial_dict['reward'] = _reward
            trial_dict['give_reward'] = _give_reward

        elif e in reward_events:
            if re.fullmatch('pump.*', e):
                trial_dict['reward_time'] = cur_time

        elif e in stimulus_events:
            if e == 'stimulus_A_on':
                cur_stimulus = 'A'
                block_stimulus = 'A'
            elif e == 'stimulus_A_off':
                cur_stimulus = 'None'
                # block stimulus remains A
            elif e == 'stimulus_B_on':
                cur_stimulus = 'B'
                block_stimulus = 'B'
            elif e == 'stimulus_B_off':
                cur_stimulus = 'None'
                # block stimulus remains B
            elif e == 'stimulus_C_on':
                pass  # for C, I might just leave the stimulus as None
                # cur_stimulus = 'C'
                # block_stimulus = 'C'
            elif e == 'stimulus_C_off':
                pass
            elif e == 'LED_on':  # use this when LED logs are saved - they better be!! (from 8/13/25 forwards)
                trial_dict['led_on_time'] = cur_time
            elif e == 'LED_off':
                trial_dict['led_off_time'] = cur_time

    # assert -1 not in actions, "Action list contains -1s, which means there are unaccounted for events."
    # alternately, use -1 to represent trials to skip - noise, given rewards, etc. NONE doesn't show up in spreadsheets
    event_df = pd.DataFrame(trial_list)
    return event_df


def make_trial_df(cleaned_data: pd.DataFrame, session_id: str, save_name: str, output_path: Path, session_info: dict,
                  min_time: float=0, max_time: float=np.inf) -> pd.DataFrame:
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
    stimulus_events = collect_events.get_stimulus_events(df)

    # start from the first known context entry; throw out everything before that. I think this is unnecessary now
    # ix = np.zeros(df.shape[0]).astype(bool)
    # for e in context_events:
    #     ix = ix | (df['Event'] == e)
    #
    # start_time = df.loc[ix, 'Time'].values[0]
    # df = df.loc[df['Time'] >= start_time]

    p_active_rew, p_inactive_rew = session_info['correct_reward_probability'], session_info['incorrect_reward_probability']
    trial_df = iterate_trials(df, context_events, choice_events, reward_events, stimulus_events)
    st_time = trial_df['start_time'].values[0]
    trial_df['trial_time_since_start'] = trial_df['start_time'] - st_time
    trial_df['choice_time_since_start'] = trial_df['choice_time'] - st_time

    # add in variables which are constant across all trials or will be updated as the behavior task is updated
    n_trials = len(trial_df['state'])
    trial_df['p_active_rew'] = np.ones(n_trials) * p_active_rew
    trial_df['p_inactive_rew'] = np.ones(n_trials) * p_inactive_rew
    trial_df['p_switch'] = np.ones(n_trials) * session_info['switch_probability']
    trial_df['session_ID'] = [session_id] * n_trials

    if min_time > 0:
        trial_df = trial_df[trial_df['trial_time_since_start'] > min_time]
    if max_time < np.inf:
        trial_df = trial_df[trial_df['trial_time_since_start'] < max_time]

    trial_df.reset_index(drop=True, inplace=True)

    if not output_path.exists():
        output_path.mkdir()

    p = output_path / (save_name + '.csv')
    trial_df.to_csv(p, index=False)
    # p = output_path / (save_name + '.pkl')
    # with p.open('wb') as f:
    #     pkl.dump(event_df, f)

    print("[***] Experiment saved as: {}".format(p.name))
    return trial_df


def main():
    # hardcoding the input/output paths for data intake, it's too messy and variable to do it "automatically"
    raw_data_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/rpi/CT014_2025-12-16_153200')
    processed_data_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/processed')

    current_mouse = 'CT014'
    current_date = '2025-12-16'
    sess_timestamp = '153200'
    sess_id = current_mouse + '_' + current_date
    sess_id_full = current_mouse + '_' + current_date + '_' + sess_timestamp

    session_folder = raw_data_path
    session_log = raw_data_path / '{}.log'.format(sess_id_full)
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)

    event_df = process_file(save_directory=processed_data_path, file_path=session_folder / session_log)
    with open(session_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    trial_df = make_trial_df(cleaned_data=event_df,
                             session_id=sess_id_full,
                             save_name=sess_id_full + '_trials',
                             output_path=processed_data_path,
                             session_info=session_info)
    water = calculate_water_delivery(event_df, session_info)
    print(water)


if __name__ == '__main__':
    main()

