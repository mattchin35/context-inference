# import session_overview
# import raster_plots
# import fileIO
import numpy as np
import re
from behavior_analysis import raster_plots, session_analysis, simulate_priors
from mouse_behavior_preprocessing import process_behavior_log
from behavior_analysis import block_state_space_modeling as bssm
from behavior_analysis import trial_state_space_modeling as tssm
import src.state_space_modeling.utilplot as utilplot
import pickle as pkl
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import seaborn as sns
from matplotlib.colors import ListedColormap


sns.set_style("white")
sns.set_context("talk")
color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]
colors = sns.xkcd_palette(color_names)
cmap = ListedColormap(colors)


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


def preprocess_session_log(raw_behavior_folder: str, processed_data_path: str, sess_id_full: str, min_time: float=0, max_time: float=np.inf):
    session_log = raw_behavior_folder / '{}.log'.format(sess_id_full)
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    event_df = process_behavior_log.process_file(save_directory=processed_data_path,
                                                 file_path=raw_behavior_folder / session_log)
    trial_df = process_behavior_log.make_trial_df(cleaned_data=event_df,
                                                 session_id=sess_id_full,
                                                 save_name=sess_id_full + '_trials',
                                                 output_path=processed_data_path,
                                                 session_info=session_info,
                                                  min_time=min_time, max_time=max_time)
    water = process_behavior_log.calculate_water_delivery(event_df, session_info)
    return trial_df, event_df, water


def plot_session(event_df: pd.DataFrame, session_info: dict, raw_behavior_folder: str, processed_data_path: str, figure_path: str, sess_id_full: str):
    # event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    # session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    # with open(raw_behavior_folder / session_info_path, 'rb') as f:
    #     session_info = pkl.load(f)

    if not figure_path.exists():
        figure_path.mkdir()

    event_df['Time'] -= event_df['Time'].min()
    raster_plots.generate_session_raster(event_df, session_info=session_info, fig_name=sess_id_full + '_raster',
                                         plot_path=figure_path, fig_format='png')
    raster_plots.colorblock_raster(event_df, sess_id_full + '_lick_context_raster', session_info=session_info,
                                   plot_path=figure_path, plot_choices=False)
    raster_plots.colorblock_raster(event_df, sess_id_full + '_choice_context_raster', session_info=session_info,
                                   plot_path=figure_path, plot_choices=True)


def analyze_session(trial_df: pd.DataFrame, sess_id_full: str, within_session_data_path: str, multi_session_save_path: str):
    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    assert match, "session id does not have correct format"

    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
    else:
        print("Double-check the session name!")
        return

    session_performance, block_performance, augmented_trial_df = session_analysis.analyze_session(trial_df, mouse=mouse, date=date)

    # rewards, mean, std, sem = summarize_block_switches(block_performance, min_counts=0)
    if block_performance['trials_to_correct'].iloc[-1] == 'None':
        ix_valid = block_performance.shape[0] - 1
        slope, intercept, r_value, p_value = session_analysis.session_stats(
            dependent_var=block_performance['trials_to_correct'].iloc[:ix_valid].astype(int),
            independent_var=block_performance['prev_consecutive_rewards'].iloc[:ix_valid])
    else:
        slope, intercept, r_value, p_value = session_analysis.session_stats(
            dependent_var=block_performance['trials_to_correct'],
            independent_var=block_performance['prev_consecutive_rewards'])

    session_performance['slope'] = slope
    session_performance['intercept'] = intercept
    session_performance['r_value'] = r_value
    session_performance['p_value'] = p_value
    session_performance['n_switches'] = block_performance.shape[0]
    session_analysis.save_analysis(session_performance, block_performance, augmented_trial_df,
                                   sess_id=sess_id_full,
                                   session_save_path=within_session_data_path,
                                   overall_save_path=multi_session_save_path)


def block_hmm_model(block_performance: pd.DataFrame, trial_df: pd.DataFrame, sess_id_full: str,
                      processed_data_path: Path, figure_path: Path,):
    map_model_dict, block_performance = bssm.map_block_states(block_performance, figure_path, sess_id_full, plot=True)
    map_savename = processed_data_path / (sess_id_full + '_block_map_statedict.pkl')
    with open(map_savename, 'wb') as file:
        pkl.dump(map_model_dict, file)

    block_performance = bssm.declare_inferred_strategy(block_performance)
    block_performance.to_csv(processed_data_path / (sess_id_full + '_block_performance.csv'), index=False)

    augmented_trial_df = bssm.trials_inherit_strategy(block_performance, trial_df)
    augmented_trial_df.to_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), index=False)

# def run_performance_collection():
#     """
#     For the stats plots, need to collect a forgetting Q-learning model with decay .6, stickiness 0, temp .3;
#     2. A HMM with Psw .1, stickiness 0, temp 1
#     """
#     params = TaskParams()
#
#     # Task general params
#     params.p_cue = 0
#     params.state_transition_prob = .2
#     params.active_reward_probability = .8
#     params.inactive_reward_probability = 0
#     params.correct_reward_size = 1
#     params.incorrect_reward_size = 0
#
#     # Reinforcement learning parameters
#     params.greedy_action_selection = True
#     params.greedy_epsilon = 1
#     params.QL_learning_rate = .1  # for standard Q-learning agent
#     params.FQL_decay = .9  # for forgetting Q-learning agent
#
#     # Forgetting Q-learning/RFLR parameters...may need to fit this for best fit
#     params.stickiness = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
#     params.weight_reward_history = 2  # default 2. Weight on reward history. "beta" in Beron PNAS 2022
#     params.weight_decay = 1.5  # default 1.5. Reward history decay parameter (higher is faster decay, lower has longer "memory"). "tau" in Beron PNAS 2022
#     params.action_temperature = 1  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value
#
#     agent = model_agents.Qlearning(params)
#     action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
#     augmented_trial_df['Qlearning_prob_left'] = action_dist['p_left']
#     augmented_trial_df['Qlearning_rel_value'] = rel_value
#     augmented_trial_df['Qlearning_greedy_action'] = actions
#
#     agent = model_agents.ForgettingQlearning(params)
#     action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
#     augmented_trial_df['FQlearning_prob_left'] = action_dist['p_left']
#     augmented_trial_df['FQlearning_rel_value'] = rel_value
#     augmented_trial_df['FQlearning_greedy_action'] = actions
#
#     agent = model_agents.Logistic(params)
#     action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
#     augmented_trial_df['RFLR_prob_left'] = action_dist['p_left']
#     augmented_trial_df['RFLR_rel_value'] = rel_value
#     augmented_trial_df['RFLR_greedy_action'] = actions
#
#     agent = model_agents.HMM(params)
#     action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
#     augmented_trial_df['HMM_prob_left'] = action_dist['p_left']
#     augmented_trial_df['HMM_rel_value'] = rel_value
#     augmented_trial_df['HMM_greedy_action'] = actions
#
#     augmented_trial_df.to_csv(augmented_trial_df_path, index=False)


# def performance_plots_single_session():
#     multisession_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
#     session_data_home = Path(
#         '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
#     sess_id_full = 'CT014_2025-12-16_153200'
#     processed_data_path = session_data_home / 'processed'
#     session_figure_path = session_data_home / 'figures'
#
#     pattern = r'(\w+)_([\d\-]+)_(\d+)'
#     match = re.search(pattern, sess_id_full)
#
#     if match:
#         mouse, date, timestamp = match.groups()
#         print(f"Mouse id: {mouse}")  # abc123
#         print(f"Date: {date}")  # YYYY-MM-DD
#         print(f"Time: {timestamp}")  # HHMMSS
#         sess_id_abbreviated = mouse + '_' + date
#     else:
#         print("Double-check the session name!")
#         return
#
#     multisession_df, block_performance, augmented_trial_df = session_analysis.load_analysis(sess_id_full,
#                                                                                             session_data_folder=processed_data_path,
#                                                                                             multisession_data_folder=multisession_save_path)
#     overall_df = multisession_df[~multisession_df['date'].isna()]
#     plot_session_correct(block_performance, session_figure_path, sess_id_full)
#     plot_session_trials_to_correct(block_performance, session_figure_path, sess_id_full)
#     plot_session_nswitches(block_performance, session_figure_path, sess_id_full)
#     # plot_multisession_correct(overall_df, mouse_plot_path, figure_id=mouse)
#     # plot_multisession_trials_to_correct(overall_df, mouse_plot_path, figure_id=mouse)
#
#     slope = multisession_df.loc[multisession_df['date'] == date, 'slope'].values[0]
#     intercept = multisession_df.loc[multisession_df['date'] == date, 'intercept'].values[0]
#     scatter_trials_to_correct(block_performance, slope=slope, intercept=intercept,
#                               plot_path=session_figure_path, figure_id=sess_id_full)
#     plot_learning_curve(multisession_df['slope'], multisession_df['n_switches'], figure_id=mouse,
#                         plot_path=multisession_save_path,
#                         dates=multisession_df['date'].values)


def main():
    """Analyze a single session from start to finish"""
    # prep data selection

    multi_session_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference')
    sess_id_full = 'CT014_2025-12-16_153200'
    # session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251204')
    # sess_id_full = 'CT014_2025-12-04_123418'
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

    # preprocess behavior log and save
    # trial_df, event_df, water = preprocess_session_log(raw_behavior_folder=raw_behavior_folder, processed_data_path=processed_data_path, sess_id_full=sess_id_full,
    #                                                    min_time=0, max_time=np.inf)

    # load processed raw data
    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',')

    # plot_session(event_df, session_info,
    #              raw_behavior_folder=raw_behavior_folder, processed_data_path=processed_data_path,
    #              figure_path=figure_path, sess_id_full=sess_id_full)

    # analyze trials and save
    # augmented_trial_df, block_performance, multisession_df = session_analysis.run_analysis(trial_df, session=sess)
    multisession_df, block_performance, augmented_trial_df = session_analysis.load_analysis(sess_id_full,
                                                                                            session_data_folder=processed_data_path,
                                                                                            multisession_data_folder=multi_session_save_path)

    model_selection = bssm.run_information_criteria(block_performance, session=sess, prior_alpha=1, prior_sigma=1)
    # block_performance, augmented_trial_df = bssm.run_block_modeling(block_performance, augmented_trial_df, session=sess, num_states=1,
    #                                                                 prior_alpha=1, prior_sigma=1)
    # block_model_dict_path = processed_data_path / (sess_id_full + '_block_statedict.pkl')
    # with open(block_model_dict_path, 'rb') as file:
    #     block_model_dict = pkl.load(file)

    # model_selection = tssm.run_information_criteria(augmented_trial_df, session=sess, algorithm='MAP', prior_alpha=1, prior_sigma=1)
    # augmented_trial_df = tssm.run_trial_modeling(augmented_trial_df, session=sess, num_states=2, prior_alpha=1, prior_sigma=1)


def presentation_plots(block_df: pd.DataFrame, trial_df: pd.DataFrame):
    ix_valid = (block_df['trials_to_correct'] != 'None') & (block_df['prev_n_correct'] != 'None')
    df = block_df[ix_valid]

    consecutive_rewards = df['prev_consecutive_rewards'].to_numpy().reshape(-1, 1).astype(int)
    prev_rewards = df['prev_n_rewarded'].to_numpy().reshape(-1, 1).astype(int)
    # prev_correct = df['prev_n_correct'].to_numpy().reshape(-1, 1).astype(int)
    trials_to_correct = df['trials_to_correct'].to_numpy().reshape(-1, 1).astype(int)
    bias_flag = df['bias_full_flag'].to_numpy().reshape(-1, 1) == 'True'
    predictors = np.concatenate([prev_rewards, bias_flag], axis=1)
    pred_labels = ['previous rewards', 'block bias flag']

    utilplot.plot_postprob_obs_for_presentation(block_model_dict['map']['posterior_probs'], trials_to_correct, predictors, map_hmm, colors, cmap, predictor_labels = pred_labels)


if __name__ == '__main__':
    main()

