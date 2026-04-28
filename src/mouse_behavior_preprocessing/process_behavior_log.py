import pandas as pd
import numpy as np
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
from pathlib import Path
import pickle as pkl
from src.behavior_analysis import collect_events
from collections import OrderedDict


"""
Functions to prepare a single behavior session for analysis.
"""

MISSING_VALUE = "None"
NO_CHOICE_ACTION = "no_choice"


class ChoiceSide(IntEnum):
    """Integer codes for animal choice side in processed trial tables."""

    RIGHT = 0
    LEFT = 1


class TaskState(IntEnum):
    """Integer codes for task block state in processed trial tables."""

    RIGHT_PATCH = 0
    LEFT_PATCH = 1
    DARK_PERIOD = 2


states = ['right', 'left']
state_dict = {
    'right': int(ChoiceSide.RIGHT),
    'left': int(ChoiceSide.LEFT),
}  # i.e. [0 right, 1 left]


@dataclass
class TrialParserState:
    """Mutable state used while converting event rows into trial rows.

    Attributes
    ----------
    current_state : str
        Current task state label. Unitless categorical string.
    current_state_int : int
        Current task state code. Values follow `TaskState`, or -1 before the
        first known state.
    current_block : int
        Zero-based block index. Starts at -1 before the first block entry event.
    current_trial_in_block : int
        Zero-based trial index within the current block. Starts at -1 before the
        first trial in each block.
    current_trial : int
        Zero-based session trial index. Starts at -1 before the first trial.
    current_stimulus : str
        Current active stimulus label, or the configured missing-value string.
    block_stimulus : str
        Stimulus assigned to the current block, or the configured missing-value
        string.
    active_trial : OrderedDict or None
        Trial row currently being populated. None when no trial has started.
    completed_trials : list[OrderedDict]
        Completed trial rows accumulated in session order.
    """

    current_state: str = MISSING_VALUE
    current_state_int: int = -1
    current_block: int = -1
    current_trial_in_block: int = -1
    current_trial: int = -1
    current_stimulus: str = MISSING_VALUE
    block_stimulus: str = MISSING_VALUE
    active_trial: OrderedDict | None = None
    completed_trials: list[OrderedDict] = field(default_factory=list)


def _assert_supported_missing_value(missing_value: str) -> None:
    """Validate the missing-value sentinel used in processed trial tables.

    Parameters
    ----------
    missing_value : str
        Missing-value sentinel to write into trial dataframe fields. Currently
        only the literal string `"None"` is supported because downstream CSV
        readers and analysis code check for this value explicitly.

    Returns
    -------
    None
        Raises AssertionError when a non-supported sentinel is requested.
    """
    assert missing_value == MISSING_VALUE, (
        "Only string 'None' is currently supported for CSV/downstream compatibility. "
        "Python None or np.nan support should be added in a later compatibility refactor."
    )


def _new_trial_dict(state: TrialParserState, start_time: float, missing_value: str) -> OrderedDict:
    """Create an initialized processed-trial row.

    Parameters
    ----------
    state : TrialParserState
        Parser state after trial counters have been advanced for the new trial.
    start_time : float
        Trial start timestamp in seconds, matching the raw event table timebase.
    missing_value : str
        Missing-value sentinel for unset trial fields. Currently must be
        `"None"`.

    Returns
    -------
    OrderedDict
        Trial row with fixed column order. Scalar fields describe one trial;
        times are in seconds.
    """
    return OrderedDict(
        state=state.current_state,
        state_int=state.current_state_int,
        cur_trial=state.current_trial,
        cur_trial_in_block=state.current_trial_in_block,
        cur_block=state.current_block,
        action=missing_value,
        correct=missing_value,
        reward=missing_value,
        give_reward=missing_value,
        active_stimulus=state.current_stimulus,
        block_stimulus=state.block_stimulus,
        start_time=start_time,
        choice_time=missing_value,
        reward_time=missing_value,
        led_on_time=missing_value,
        led_off_time=missing_value,
    )


def _start_new_trial(state: TrialParserState, start_time: float, missing_value: str) -> None:
    """Finalize the previous trial and start a new active trial.

    Parameters
    ----------
    state : TrialParserState
        Mutable parser state. Updated in place.
    start_time : float
        Trial start timestamp in seconds, matching the raw event table timebase.
    missing_value : str
        Missing-value sentinel for unset trial fields. Currently must be
        `"None"`.

    Returns
    -------
    None
        The previous active trial, if present, is appended to
        `state.completed_trials`; `state.active_trial` is replaced by the new
        initialized trial row.
    """
    if state.active_trial is not None:
        state.completed_trials.append(state.active_trial)

    state.current_trial += 1
    state.current_trial_in_block += 1
    state.active_trial = _new_trial_dict(state, start_time, missing_value)


def _require_active_trial(state: TrialParserState, event: str) -> OrderedDict:
    """Return the active trial row or raise if an event appears outside a trial.

    Parameters
    ----------
    state : TrialParserState
        Mutable parser state for the current event stream.
    event : str
        Raw event name being handled. Unitless categorical string.

    Returns
    -------
    OrderedDict
        Active trial row to update.
    """
    if state.active_trial is None:
        raise RuntimeError(f"Event {event!r} occurred before any trial_start event.")
    return state.active_trial


def _handle_context_event(event: str, current_time: float, state: TrialParserState, missing_value: str) -> None:
    """Update parser state for context and trial-boundary events.

    Parameters
    ----------
    event : str
        Raw context event name.
    current_time : float
        Event timestamp in seconds. Used as the trial start time for
        `trial_start` events.
    state : TrialParserState
        Mutable parser state. Updated in place.
    missing_value : str
        Missing-value sentinel for unset trial fields. Currently must be
        `"None"`.

    Returns
    -------
    None
        Updates task context or starts a new active trial.
    """
    if event == 'enter_right_patch':
        state.current_state = 'right_patch'
        state.current_state_int = int(TaskState.RIGHT_PATCH)
        state.current_block += 1
        state.current_trial_in_block = -1
        state.current_stimulus = missing_value
        state.block_stimulus = missing_value
    elif event == 'enter_left_patch':
        state.current_state = 'left_patch'
        state.current_state_int = int(TaskState.LEFT_PATCH)
        state.current_block += 1
        state.current_trial_in_block = -1
        state.current_stimulus = missing_value
        state.block_stimulus = missing_value
    elif event == 'enter_dark_period':
        state.current_state = 'dark_period'
        state.current_state_int = int(TaskState.DARK_PERIOD)
        state.current_stimulus = missing_value
        state.block_stimulus = missing_value
    elif event == 'trial_start':
        _start_new_trial(state, current_time, missing_value)
    elif event == 'trial_stop':
        pass
    else:
        raise NameError('Unrecognized context event: {}'.format(event))


def _handle_choice_event(event: str, reward_note: str, current_time: float, state: TrialParserState) -> None:
    """Update the active trial from a choice or experimenter-reward event.

    Parameters
    ----------
    event : str
        Raw choice-like event name.
    reward_note : str
        Raw reward outcome note. Expected values are `reward_True` or
        `reward_False`.
    current_time : float
        Event timestamp in seconds, saved as `choice_time`.
    state : TrialParserState
        Mutable parser state. The active trial is updated in place.

    Returns
    -------
    None
        The active trial receives `choice_time`, `action`, `correct`, `reward`,
        and `give_reward` values.
    """
    trial = _require_active_trial(state, event)
    action, correct, reward, give_reward = choice_event_summary(event, reward_note)
    trial['choice_time'] = current_time
    trial['action'] = action
    trial['correct'] = correct
    trial['reward'] = reward
    trial['give_reward'] = give_reward


def _handle_reward_event(event: str, current_time: float, state: TrialParserState) -> None:
    """Update reward timing for the active trial.

    Parameters
    ----------
    event : str
        Raw reward event name. Pump events are expected to match `pump.*`.
    current_time : float
        Event timestamp in seconds, saved as `reward_time`.
    state : TrialParserState
        Mutable parser state. The active trial is updated in place.

    Returns
    -------
    None
        Saves reward delivery time when the event is a pump event.
    """
    if re.fullmatch('pump.*', event):
        trial = _require_active_trial(state, event)
        trial['reward_time'] = current_time


def _handle_stimulus_event(event: str, current_time: float, state: TrialParserState, missing_value: str) -> None:
    """Update active stimulus, block stimulus, or LED timing.

    Parameters
    ----------
    event : str
        Raw stimulus or LED event name.
    current_time : float
        Event timestamp in seconds. Used for LED on/off fields.
    state : TrialParserState
        Mutable parser state. Updated in place.
    missing_value : str
        Missing-value sentinel for unset stimulus fields. Currently must be
        `"None"`.

    Returns
    -------
    None
        Updates current stimulus context or LED timing on the active trial.
    """
    if event == 'stimulus_A_on':
        state.current_stimulus = 'A'
        state.block_stimulus = 'A'
    elif event == 'stimulus_A_off':
        state.current_stimulus = missing_value
        # block stimulus remains A
    elif event == 'stimulus_B_on':
        state.current_stimulus = 'B'
        state.block_stimulus = 'B'
    elif event == 'stimulus_B_off':
        state.current_stimulus = missing_value
        # block stimulus remains B
    elif event == 'stimulus_C_on':
        state.current_stimulus = missing_value
        state.block_stimulus = 'C'
    elif event == 'stimulus_C_off':
        pass
    elif event == 'LED_on':
        trial = _require_active_trial(state, event)
        trial['led_on_time'] = current_time
    elif event == 'LED_off':
        trial = _require_active_trial(state, event)
        trial['led_off_time'] = current_time
    else:
        raise NameError('Unrecognized stimulus event: {}'.format(event))


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

    # Could replace event names or extract specific keys to filter events dictionary. That functionality was deprecated
    # and deleted, but it is available in "behavior_analysis_old/fileIO" if desired. The filter_events input is kept
    # as a reminder.

    # Save the new DataFrame to a CSV file
    if not save_directory.exists():
        save_directory.mkdir(parents=True)

    new_file_path = save_directory / (Path(file_path).stem + '_events.csv')
    df.to_csv(new_file_path, index=False)
    print('cleaned_file_created')
    return df


def choice_event_summary(event: str, reward_note: str) -> tuple[int, int, int, int]:
    """Summarize one choice-like event into trial outcome fields.

    Parameters
    ----------
    event : str
        Raw event name for an animal choice or experimenter reward event.
        Supported animal choices are correct/wrong left/right patch events.
        Supported experimenter reward events are `giving_reward_left_patch` and
        `giving_reward_right_patch`.
    reward_note : str
        Raw reward outcome note. Expected values are `reward_True` or
        `reward_False`.

    Returns
    -------
    tuple[int, int, int, int]
        `(action, correct, reward, give_reward)` for one trial. `action` uses
        `ChoiceSide` codes, where 0 is right and 1 is left. `correct`, `reward`,
        and `give_reward` are binary 0/1 flags. Experimenter rewards currently
        preserve the legacy side-as-action behavior for downstream
        compatibility.
    """
    # LEFT CHOICES
    if event == 'wrong_choice_right_patch':
        action = int(ChoiceSide.LEFT)
        correct = 0
        give_reward = 0
    elif event == 'correct_choice_left_patch':
        action = int(ChoiceSide.LEFT)
        correct = 1
        give_reward = 0

    # RIGHT CHOICES
    elif event == 'wrong_choice_left_patch':
        action = int(ChoiceSide.RIGHT)
        correct = 0
        give_reward = 0
    elif event == 'correct_choice_right_patch':
        action = int(ChoiceSide.RIGHT)
        correct = 1
        give_reward = 0

    elif event == 'giving_reward_left_patch':
        # action = 'None'
        action = int(ChoiceSide.LEFT)
        correct = 0
        give_reward = 1
    elif event == 'giving_reward_right_patch':
        # action = 'None'
        action = int(ChoiceSide.RIGHT)
        correct = 0
        give_reward = 1

    else:
        raise NameError('Unrecognized choice event: {}'.format(event))

    if reward_note == 'reward_True':
        reward = 1
    elif reward_note == 'reward_False':
        reward = 0
    else:
        raise NameError('Unrecognized reward outcome note: {}'.format(reward_note))

    return action, correct, reward, give_reward


def iterate_trials(raw_data: pd.DataFrame, context_events: list, choice_events: list, reward_events: list,
                   stimulus_events: list, missing_value: str = MISSING_VALUE) -> pd.DataFrame:
    """Convert a raw event table into one processed row per completed trial.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw event table with shape `(n_events, n_columns)`. Required columns are
        `Time` in seconds, `Event` as raw event-name strings, and `Note` as raw
        event-note strings.
    context_events : list
        Event names that update task context or mark trial boundaries.
    choice_events : list
        Event names that encode animal choices or experimenter rewards.
    reward_events : list
        Event names that encode pump reward delivery.
    stimulus_events : list
        Event names that encode stimulus or LED state changes.
    missing_value : str, default="None"
        Missing-value sentinel for unset trial fields. Only string `"None"` is
        currently supported for CSV/downstream compatibility.

    Returns
    -------
    pd.DataFrame
        Trial table with shape `(n_completed_trials, n_trial_columns)`. Time
        columns are in seconds. Trials are finalized only when a later
        `trial_start` event begins the next trial, preserving legacy behavior.
    """
    _assert_supported_missing_value(missing_value)
    parser_state = TrialParserState(
        current_state=missing_value,
        current_stimulus=missing_value,
        block_stimulus=missing_value,
    )

    for i, event in enumerate(raw_data['Event'].values):
        current_time = raw_data['Time'].values[i]

        # Stimulus events are checked before context events because
        # collect_events.get_context_events also returns stimulus_* names.
        if event in choice_events:
            _handle_choice_event(event, raw_data['Note'].values[i], current_time, parser_state)
        elif event in reward_events:
            _handle_reward_event(event, current_time, parser_state)
        elif event in stimulus_events:
            _handle_stimulus_event(event, current_time, parser_state, missing_value)
        elif event in context_events:
            _handle_context_event(event, current_time, parser_state, missing_value)

    # The active trial is intentionally not appended at end-of-file. Trials are
    # saved only when another trial starts, matching the original parser.
    return pd.DataFrame(parser_state.completed_trials)


def make_trial_df(cleaned_data: pd.DataFrame, session_id: str, save_name: str, output_path: Path, session_info: dict,
                  min_time: float = 0, max_time: float = np.inf,
                  missing_value: str = MISSING_VALUE) -> pd.DataFrame:
    """Prepare a processed trial dataframe from cleaned behavior events.

    Parameters
    ----------
    cleaned_data : pd.DataFrame
        Cleaned event table with shape `(n_events, n_columns)`. Required columns
        are `Time` in seconds, `Event` as raw event-name strings, and `Note` as
        raw event-note strings.
    session_id : str
        Session identifier copied into the `session_ID` column.
    save_name : str
        Output CSV stem, without `.csv`.
    output_path : Path
        Directory where the processed trial CSV is saved.
    session_info : dict
        Session metadata containing `correct_reward_probability`,
        `incorrect_reward_probability`, and `switch_probability`.
    min_time : float, default=0
        Minimum `trial_time_since_start` in seconds to retain.
    max_time : float, default=np.inf
        Maximum `trial_time_since_start` in seconds to retain.
    missing_value : str, default="None"
        Missing-value sentinel for unset trial fields. Only string `"None"` is
        currently supported for CSV/downstream compatibility.

    Returns
    -------
    pd.DataFrame
        Trial table with one row per completed trial. Time columns are in
        seconds. Additional columns include reward probabilities, switch
        probability, and session identifier.
    """
    _assert_supported_missing_value(missing_value)
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

    p_active_rew = session_info['correct_reward_probability']
    p_inactive_rew = session_info['incorrect_reward_probability']
    trial_df = iterate_trials(
        df,
        context_events,
        choice_events,
        reward_events,
        stimulus_events,
        missing_value=missing_value,
    )
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
    trial_df.to_csv(p, index=False, na_rep=missing_value)
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
