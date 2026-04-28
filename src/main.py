import json
import numpy as np
import re
from behavior_analysis import performance_plots
from behavior_analysis import raster_plots, session_analysis, simulate_priors
from mouse_behavior_preprocessing import process_behavior_log
from behavior_analysis import block_state_space_modeling as bssm
from behavior_analysis import trial_state_space_modeling as tssm
from behavior_analysis import gather_trial_features as gtf
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


def build_simulation_session(simulated_run_dir: Path, sess_id: str, session_info: dict) -> Session:
    """Build a Session object for a saved switched-run analysis directory.

    Parameters
    ----------
    simulated_run_dir : Path
        Directory containing one saved simulated run and all analysis outputs.
    sess_id : str
        Full simulated run identifier with prefix, date, and time embedded in
        the filename stem.
    session_info : dict
        JSON metadata loaded from the saved switched-run params file.

    Returns
    -------
    Session
        Session metadata configured for simulation analysis. Multisession
        saving is disabled, and all outputs are written back into
        `simulated_run_dir`.
    """
    pattern = r'(.+?)_(\d{4}-\d{2}-\d{2})_(\d{6})'
    match = re.search(pattern, sess_id)
    if match is None:
        raise ValueError(f"simulated session id does not match expected pattern: {sess_id}")

    prefix, date, timestamp = match.groups()
    sess = Session()
    sess.multi_session_save_path = None
    sess.session_data_home = simulated_run_dir
    sess.sess_id_full = sess_id
    sess.sess_id_abbreviated = f"{prefix}_{date}_{timestamp}"
    sess.raw_behavior_folder = simulated_run_dir
    sess.processed_data_path = simulated_run_dir
    sess.figure_path = simulated_run_dir
    sess.mouse = prefix
    sess.date = date
    sess.timestamp = timestamp
    sess.session_info_fname = simulated_run_dir / f"{sess_id}_params.json"
    sess.session_info = session_info
    return sess


def preprocess_session_log(
    raw_behavior_folder: Path,
    processed_data_path: Path,
    sess_id_full: str,
    min_time: float = 0,
    max_time: float = np.inf,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess one raw behavior log into event and trial tables.

    Parameters
    ----------
    raw_behavior_folder : Path
        Directory containing `{sess_id_full}.log` and
        `{sess_id_full}_session_info.pkl`.
    processed_data_path : Path
        Directory where processed event and trial CSV files are saved.
    sess_id_full : str
        Full session identifier formatted as `mouse_YYYY-MM-DD_HHMMSS`.
    min_time : float, default=0
        Minimum trial time since session start, in seconds, retained in the
        saved trial table.
    max_time : float, default=np.inf
        Maximum trial time since session start, in seconds, retained in the
        saved trial table.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - trial table, shape `(n_trials, n_trial_columns)`
        - event table, shape `(n_events, 3)` with columns `Time`, `Event`,
          and `Note`; `Time` remains in log-file seconds
        - water summary table, shape `(3, 2)`, with delivered amounts in the
          task log reward units
    """
    session_log = raw_behavior_folder / '{}.log'.format(sess_id_full)
    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    event_df = process_behavior_log.process_file(save_directory=processed_data_path,
                                                 file_path=session_log)
    trial_df = process_behavior_log.make_trial_df(cleaned_data=event_df,
                                                 session_id=sess_id_full,
                                                 save_name=sess_id_full + '_trials',
                                                 output_path=processed_data_path,
                                                 session_info=session_info,
                                                  min_time=min_time, max_time=max_time)
    water = process_behavior_log.calculate_water_delivery(event_df, session_info)
    return trial_df, event_df, water


def load_or_preprocess_session(
    raw_behavior_folder: Path,
    processed_data_path: Path,
    sess_id_full: str,
    preprocess_raw_session: bool = False,
    min_time: float = 0,
    max_time: float = np.inf,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load processed behavior tables, or regenerate them from the raw log.

    Parameters
    ----------
    raw_behavior_folder : Path
        Directory containing raw behavior log inputs when
        `preprocess_raw_session` is True.
    processed_data_path : Path
        Directory containing or receiving `{sess_id_full}_events.csv` and
        `{sess_id_full}_trials.csv`.
    sess_id_full : str
        Full session identifier formatted as `mouse_YYYY-MM-DD_HHMMSS`.
    preprocess_raw_session : bool, default=False
        If True, regenerate processed event and trial tables from the raw log.
        If False, load existing processed CSV files.
    min_time : float, default=0
        Minimum trial time since session start, in seconds, used only during
        preprocessing.
    max_time : float, default=np.inf
        Maximum trial time since session start, in seconds, used only during
        preprocessing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame or None]
        - trial table, shape `(n_trials, n_trial_columns)`
        - event table, shape `(n_events, n_event_columns)`
        - water summary table when preprocessing was run, otherwise None
    """
    if preprocess_raw_session:
        return preprocess_session_log(
            raw_behavior_folder=raw_behavior_folder,
            processed_data_path=processed_data_path,
            sess_id_full=sess_id_full,
            min_time=min_time,
            max_time=max_time,
        )

    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',')
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',')
    return trial_df, event_df, None


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


def performance_plots_single_session():
    multisession_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
    session_data_home = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    processed_data_path = session_data_home / 'processed'
    session_figure_path = session_data_home / 'figures'

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

    multisession_df, block_performance, augmented_trial_df = session_analysis.load_analysis(sess_id_full,
                                                                                            session_data_folder=processed_data_path,
                                                                                            multisession_data_folder=multisession_save_path)
    overall_df = multisession_df[~multisession_df['date'].isna()]
    plot_session_correct(block_performance, session_figure_path, sess_id_full)
    plot_session_trials_to_correct(block_performance, session_figure_path, sess_id_full)
    plot_session_nswitches(block_performance, session_figure_path, sess_id_full)
    # plot_multisession_correct(overall_df, mouse_plot_path, figure_id=mouse)
    # plot_multisession_trials_to_correct(overall_df, mouse_plot_path, figure_id=mouse)

    slope = multisession_df.loc[multisession_df['date'] == date, 'slope'].values[0]
    intercept = multisession_df.loc[multisession_df['date'] == date, 'intercept'].values[0]
    scatter_trials_to_correct(block_performance, slope=slope, intercept=intercept,
                              plot_path=session_figure_path, figure_id=sess_id_full)
    plot_learning_curve(multisession_df['slope'], multisession_df['n_switches'], figure_id=mouse,
                        plot_path=multisession_save_path,
                        dates=multisession_df['date'].values)


def main_simulation():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    switching_run_data_dir = PROJECT_ROOT / "data/processed/switching_agents"

    sess_id = 'sample_switch_mode_1_2026-03-26_141941_agent-sample_switch_mode_1_pRew-0.8_pInc-0_pSwitch-0.2_nTrials-400'
    simulated_run_dir = switching_run_data_dir / 'sample_switch_mode_1' / sess_id

    trial_df_path = simulated_run_dir / '{}.csv'.format(sess_id)
    session_info_path = simulated_run_dir / '{}_params.json'.format(sess_id)
    assert simulated_run_dir.exists(), "simulated_run_dir at {} not found!".format(simulated_run_dir)
    assert trial_df_path.exists(), "trial_df at {} not found!".format(trial_df_path)
    assert session_info_path.exists(), "session_info at {} not found!".format(session_info_path)
    trial_df = pd.read_csv(trial_df_path, sep=',')
    with open(session_info_path, 'r') as file:
        session_info = json.load(file)

    sess = build_simulation_session(simulated_run_dir, sess_id, session_info)

    # analyze trials and save
    augmented_trial_df, block_performance, multisession_df = session_analysis.run_analysis(trial_df, session=sess)

    ### Gather trial features ###
    augmented_trial_df, task_params = gtf.collect_and_save_trial_features(
        augmented_trial_df,
        processed_data_path=sess.processed_data_path,
        sess_id_full=sess.sess_id_full,
    )

    block_model_selection = bssm.run_information_criteria(block_performance, session=sess, algorithm='MLE',
                                                    prior_alpha=1, prior_sigma=1)
    # cv_model_selection = bssm.run_cross_validation(block_performance, session=sess, algorithm='MLE',
    #                                                prior_alpha=1, prior_sigma=1, n_runs=5, n_folds=2)

    # block_performance, augmented_trial_df = bssm.run_block_modeling(block_performance, augmented_trial_df, session=sess, num_states=2,
    #                                                                 prior_alpha=1, prior_sigma=1)
    # block_model_dict_path = sess.processed_data_path / (sess.sess_id_full + '_block_statedict.pkl')
    # with open(block_model_dict_path, 'rb') as file:
    #     block_model_dict = pkl.load(file)

    ### trial state space modeling ###
    # trial_model_selection = tssm.run_information_criteria(augmented_trial_df, session=sess, algorithm='MLE', prior_alpha=1, prior_sigma=1)
    # augmented_trial_df = tssm.run_trial_modeling(augmented_trial_df, session=sess, num_states=2, prior_alpha=1,
    #                                              prior_sigma=1)


def main_mouse():
    """Analyze a single behavior session from start to finish."""

    ### USER FLAGS - CHOOSE THESE FOR EACH RUN ###
    preprocess_raw_session = False

    ### HARDCODED DATA PATHS - CHOOSE THESE FOR EACH RUN ###
    multi_session_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
    # session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251221_latentInference')
    # sess_id_full = 'CT014_2025-12-21_165755'
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251204')
    sess_id_full = 'CT014_2025-12-04_123418'


    ### everything below this should be edited so it doesn't have to be commented in or out or have hardcodes changed ###
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

    trial_df, event_df, water = load_or_preprocess_session(
        raw_behavior_folder=raw_behavior_folder,
        processed_data_path=processed_data_path,
        sess_id_full=sess_id_full,
        preprocess_raw_session=preprocess_raw_session,
        min_time=0,
        max_time=np.inf,
    )
    if water is not None:
        print(f"Preprocessed session log for {sess_id_full}. Water delivered: {water}")

    plot_session(event_df, session_info,
                 raw_behavior_folder=raw_behavior_folder, processed_data_path=processed_data_path,
                 figure_path=figure_path, sess_id_full=sess_id_full)

    # analyze trials and save
    augmented_trial_df, block_performance, multisession_df = session_analysis.run_analysis(trial_df, session=sess)
    multisession_df, block_performance, augmented_trial_df = session_analysis.load_analysis(sess_id_full,
                                                                                            session_data_folder=processed_data_path,
                                                                                            multisession_data_folder=multi_session_save_path)

    ### plot single session performance ###
    performance_plots.plot_session_correct(block_performance, sess.figure_path, sess_id_full)
    performance_plots.plot_session_trials_to_correct(block_performance, sess.figure_path, sess_id_full)
    performance_plots.plot_session_nswitches(block_performance, sess.figure_path, sess_id_full)

    slope = multisession_df.loc[multisession_df['date'] == date, 'slope'].values[0]
    intercept = multisession_df.loc[multisession_df['date'] == date, 'intercept'].values[0]
    performance_plots.scatter_trials_to_correct(block_performance, slope=slope, intercept=intercept,
                              plot_path=sess.figure_path, figure_id=sess_id_full,
                              title=sess_id_abbreviated)

    ### Gather trial features ###
    augmented_trial_df, task_params = gtf.collect_and_save_trial_features(
        augmented_trial_df,
        processed_data_path=processed_data_path,
        sess_id_full=sess_id_full,
    )

    # block_model_selection = bssm.run_information_criteria(block_performance, session=sess, algorithm='MLE',
    #                                                 prior_alpha=1, prior_sigma=1)

    # cv_model_selection = bssm.run_cross_validation(block_performance, session=sess, algorithm='MLE',
    #                                                prior_alpha=1, prior_sigma=1, n_runs=5, n_folds=2)

    # block_performance, augmented_trial_df = bssm.run_block_modeling(block_performance, augmented_trial_df, session=sess, num_states=2,
    #                                                                 prior_alpha=1, prior_sigma=1)

    # block_model_dict_path = processed_data_path / (sess_id_full + '_block_statedict.pkl')
    # with open(block_model_dict_path, 'rb') as file:
    #     block_model_dict = pkl.load(file)

    ### trial state space modeling ###
    # trial_model_selection = tssm.run_information_criteria(augmented_trial_df, session=sess, algorithm='MLE', prior_alpha=1, prior_sigma=1)
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
    main_mouse()
    # main_simulation()
