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
from typing import Sequence
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


@dataclass
class SavedSessionAnalysis:
    """Saved single-session analysis tables loaded from CSV.

    Attributes
    ----------
    session : Session
        Session metadata. The `sess_id_full`, `date`, `processed_data_path`,
        and `figure_path` fields identify the saved CSV source.
    block_performance : pd.DataFrame
        Blockwise table, shape `(n_blocks, n_block_columns)`. `block_ix`
        identifies within-session block ids before concatenation.
    augmented_trial_df : pd.DataFrame
        Trialwise table, shape `(n_trials, n_trial_columns)`. `cur_block`
        identifies each trial's within-session block before concatenation.
    """
    session: Session
    block_performance: pd.DataFrame
    augmented_trial_df: pd.DataFrame


@dataclass
class ConcatenatedSessionAnalysis:
    """Continuous multisession table set built from saved per-session CSVs.

    Attributes
    ----------
    block_performance : pd.DataFrame
        Concatenated block table, shape `(sum(n_blocks), n_block_columns)`.
        `block_ix` is recoded to a continuous zero-based axis across sessions;
        `source_block_ix` stores the original within-session block id.
    augmented_trial_df : pd.DataFrame
        Concatenated trial table, shape `(sum(n_trials), n_trial_columns)`.
        `cur_block` is recoded to match the concatenated block axis;
        `source_cur_block` stores the original within-session block id.
    block_session_lengths : np.ndarray
        Number of block rows contributed by each source session, shape
        `(n_sessions,)`, in concatenation order.
    trial_session_lengths : np.ndarray
        Number of trial rows contributed by each source session, shape
        `(n_sessions,)`, in concatenation order.
    """
    block_performance: pd.DataFrame
    augmented_trial_df: pd.DataFrame
    block_session_lengths: np.ndarray
    trial_session_lengths: np.ndarray


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

    event_df = pd.read_csv(processed_data_path / (sess_id_full + '_events.csv'), sep=',', na_filter=False)
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',', na_filter=False)
    return trial_df, event_df, None


def load_or_run_session_analysis(
    trial_df: pd.DataFrame,
    session: Session,
    run_session_analysis: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run session analysis or load existing analysis outputs.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_trial_columns)`. Used only when
        `run_session_analysis` is True.
    session : Session
        Session metadata containing `sess_id_full`, `processed_data_path`, and
        `multi_session_save_path`.
    run_session_analysis : bool, default=False
        If True, recompute and save session-analysis outputs. If False, load
        existing saved analysis outputs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `(augmented_trial_df, block_performance, multisession_df)`. The load
        path normalizes `session_analysis.load_analysis`, which returns
        `(multisession_df, block_performance, augmented_trial_df)`.
    """
    if run_session_analysis:
        return session_analysis.run_analysis(trial_df, session=session)

    multisession_df, block_performance, augmented_trial_df = session_analysis.load_analysis(
        session.sess_id_full,
        session_data_folder=session.processed_data_path,
        multisession_data_folder=session.multi_session_save_path,
    )
    return augmented_trial_df, block_performance, multisession_df


def find_saved_session_by_date(
    mouse: str,
    date: str,
    session_data_root: Path,
    multi_session_save_path: Path,
) -> Session:
    """Resolve one saved session analysis from mouse/date identifiers.

    Parameters
    ----------
    mouse : str
        Mouse identifier used as the first field in saved session ids.
    date : str
        Session date formatted as `YYYY-MM-DD`.
    session_data_root : Path
        Mouse-level data directory searched recursively for
        `{mouse}_{date}_*_augmented_trials.csv`.
    multi_session_save_path : Path
        Cross-session output directory assigned to the returned session
        metadata.

    Returns
    -------
    Session
        Session metadata for the single matching saved session. Paths point to
        the existing processed and figure directories for that session.

    Raises
    ------
    FileNotFoundError
        If no matching saved augmented-trials CSV is found.
    ValueError
        If more than one saved session matches the date, or if the matching
        filename does not follow `mouse_YYYY-MM-DD_HHMMSS`.
    """
    augmented_trial_paths = sorted(
        session_data_root.rglob(f"{mouse}_{date}_*_augmented_trials.csv")
    )
    if len(augmented_trial_paths) == 0:
        raise FileNotFoundError(
            f"No saved augmented-trials CSV found for {mouse} on {date} under {session_data_root}."
        )
    if len(augmented_trial_paths) > 1:
        matched_paths = "\n".join(str(path) for path in augmented_trial_paths)
        raise ValueError(
            f"Expected one saved session for {mouse} on {date}, found {len(augmented_trial_paths)}:\n"
            f"{matched_paths}"
        )

    augmented_trial_path = augmented_trial_paths[0]
    suffix = "_augmented_trials.csv"
    sess_id_full = augmented_trial_path.name[:-len(suffix)]
    match = re.search(r"(.+?)_(\d{4}-\d{2}-\d{2})_(\d{6})", sess_id_full)
    if match is None:
        raise ValueError(f"Saved session filename does not match expected pattern: {augmented_trial_path.name}")

    parsed_mouse, parsed_date, timestamp = match.groups()
    if parsed_mouse != mouse or parsed_date != date:
        raise ValueError(
            f"Resolved session id {sess_id_full} does not match requested mouse/date {mouse}/{date}."
        )

    processed_data_path = augmented_trial_path.parent
    session_data_home = processed_data_path.parent
    sess = Session()
    sess.multi_session_save_path = multi_session_save_path
    sess.session_data_home = session_data_home
    sess.sess_id_full = sess_id_full
    sess.sess_id_abbreviated = f"{mouse}_{date}"
    sess.raw_behavior_folder = session_data_home / "rpi" / sess_id_full
    sess.processed_data_path = processed_data_path
    sess.figure_path = session_data_home / "figures"
    sess.mouse = mouse
    sess.date = date
    sess.timestamp = timestamp
    sess.session_info_fname = sess.raw_behavior_folder / f"{sess_id_full}_session_info.pkl"
    sess.session_info = None
    return sess


def load_saved_session_analysis(session: Session) -> SavedSessionAnalysis:
    """Load saved block and augmented-trial CSVs for one session.

    Parameters
    ----------
    session : Session
        Session metadata with `sess_id_full` and `processed_data_path`.

    Returns
    -------
    SavedSessionAnalysis
        Loaded block table and trial table. Dataframes preserve literal
        `"None"` missing-value sentinels via `na_filter=False`.
    """
    block_path = session.processed_data_path / f"{session.sess_id_full}_block_performance.csv"
    augmented_trial_path = session.processed_data_path / f"{session.sess_id_full}_augmented_trials.csv"
    block_performance = pd.read_csv(block_path, sep=",", na_filter=False)
    augmented_trial_df = pd.read_csv(augmented_trial_path, sep=",", na_filter=False)
    return SavedSessionAnalysis(
        session=session,
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
    )


def _offset_numeric_column(values: pd.Series, offset: int, column_name: str) -> pd.Series:
    """Add an integer offset to an id column while preserving row order.

    Parameters
    ----------
    values : pd.Series
        Integer-like id column with shape `(n_rows,)`.
    offset : int
        Non-negative integer added to each id value.
    column_name : str
        Column name used in validation error messages.

    Returns
    -------
    pd.Series
        Offset integer ids with shape `(n_rows,)`.
    """
    numeric_values = pd.to_numeric(values, errors="raise")
    if np.any(np.isnan(numeric_values)):
        raise ValueError(f"{column_name} contains NaN and cannot be offset.")
    return numeric_values.astype(int) + offset


def concatenate_saved_sessions(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> ConcatenatedSessionAnalysis:
    """Concatenate saved sessions into one simple continuous session.

    This is intentionally a simplistic continuous concatenation: block ids and
    trial block ids are shifted onto one long axis, but already-computed
    within-session histories and model-value features are not recomputed across
    overnight boundaries.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Loaded sessions in desired concatenation order. Each block table must
        contain `block_ix`; each trial table must contain `cur_block`.

    Returns
    -------
    ConcatenatedSessionAnalysis
        Concatenated block and trial tables with explicit source-session
        metadata and per-session lengths.
    """
    if len(saved_sessions) == 0:
        raise ValueError("saved_sessions must contain at least one session.")

    concatenated_blocks = []
    concatenated_trials = []
    block_session_lengths = []
    trial_session_lengths = []
    block_offset = 0

    for session_index, saved_session in enumerate(saved_sessions):
        session = saved_session.session
        block_df = saved_session.block_performance.copy()
        trial_df = saved_session.augmented_trial_df.copy()

        if "block_ix" not in block_df.columns:
            raise ValueError(f"{session.sess_id_full} block_performance is missing 'block_ix'.")
        if "cur_block" not in trial_df.columns:
            raise ValueError(f"{session.sess_id_full} augmented_trial_df is missing 'cur_block'.")

        block_df["source_session_id"] = session.sess_id_full
        block_df["source_date"] = session.date
        block_df["source_session_index"] = session_index
        block_df["source_block_ix"] = block_df["block_ix"]
        block_df["block_ix"] = _offset_numeric_column(block_df["block_ix"], block_offset, "block_ix")
        block_id_map = dict(zip(block_df["source_block_ix"], block_df["block_ix"]))

        trial_df["source_session_id"] = session.sess_id_full
        trial_df["source_date"] = session.date
        trial_df["source_session_index"] = session_index
        trial_df["source_trial_row"] = np.arange(trial_df.shape[0], dtype=int)
        trial_df["source_cur_block"] = trial_df["cur_block"]
        remapped_cur_block = trial_df["source_cur_block"].map(block_id_map)
        if remapped_cur_block.isna().any():
            missing_blocks = sorted(trial_df.loc[remapped_cur_block.isna(), "source_cur_block"].unique())
            raise ValueError(
                f"{session.sess_id_full} trial rows reference blocks not present in block_performance: "
                f"{missing_blocks}"
            )
        trial_df["cur_block"] = remapped_cur_block.astype(int)

        concatenated_blocks.append(block_df)
        concatenated_trials.append(trial_df)
        block_session_lengths.append(block_df.shape[0])
        trial_session_lengths.append(trial_df.shape[0])
        block_offset += block_df.shape[0]

    return ConcatenatedSessionAnalysis(
        block_performance=pd.concat(concatenated_blocks, axis=0, ignore_index=True),
        augmented_trial_df=pd.concat(concatenated_trials, axis=0, ignore_index=True),
        block_session_lengths=np.asarray(block_session_lengths, dtype=int),
        trial_session_lengths=np.asarray(trial_session_lengths, dtype=int),
    )


def plot_mouse_learning_curve(
    mouse: str,
    multi_session_save_path: Path,
) -> pd.DataFrame:
    """Plot the learning curve from all dates in a mouse summary CSV.

    Parameters
    ----------
    mouse : str
        Mouse identifier used in `{mouse}_overall_performance.csv`.
    multi_session_save_path : Path
        Directory containing the mouse-level cross-session summary CSV and
        receiving the learning-curve figure.

    Returns
    -------
    pd.DataFrame
        Multisession summary table sorted by `date`, shape
        `(n_sessions, n_summary_columns)`.
    """
    multisession_summary_path = multi_session_save_path / f"{mouse}_overall_performance.csv"
    multisession_df = pd.read_csv(multisession_summary_path, sep=",", na_filter=False)
    multisession_df = multisession_df[~multisession_df["date"].isna()].copy()
    multisession_df.sort_values(by="date", inplace=True)
    multisession_df.reset_index(drop=True, inplace=True)

    block_count_col = performance_plots.get_session_block_count_column(multisession_df)
    learning_regressor = getattr(
        performance_plots,
        "DEFAULT_LEARNING_REGRESSOR",
        "prev_consecutive_rewards",
    )
    learning_column = f"{learning_regressor}_slope"
    performance_plots.plot_learning_curve(
        multisession_df[learning_column],
        multisession_df[block_count_col],
        figure_id=mouse,
        plot_path=multi_session_save_path,
        dates=multisession_df["date"].values,
    )
    return multisession_df


def build_multisession_session(
    mouse: str,
    multi_session_save_path: Path,
    sess_id_full: str | None = None,
) -> Session:
    """Build synthetic session metadata for concatenated multisession outputs.

    Parameters
    ----------
    mouse : str
        Mouse identifier. Used in the default synthetic session id.
    multi_session_save_path : Path
        Cross-session output directory used for processed files and figures.
    sess_id_full : str or None, default=None
        Synthetic session id. If omitted, `{mouse}_multisession` is used.

    Returns
    -------
    Session
        Session metadata whose processed and figure paths both point to
        `multi_session_save_path`.
    """
    if sess_id_full is None:
        sess_id_full = f"{mouse}_multisession"

    sess = Session()
    sess.multi_session_save_path = multi_session_save_path
    sess.session_data_home = multi_session_save_path
    sess.sess_id_full = sess_id_full
    sess.sess_id_abbreviated = sess_id_full
    sess.raw_behavior_folder = multi_session_save_path
    sess.processed_data_path = multi_session_save_path
    sess.figure_path = multi_session_save_path
    sess.mouse = mouse
    sess.date = "multisession"
    sess.timestamp = "000000"
    sess.session_info_fname = multi_session_save_path / f"{sess_id_full}_session_info.pkl"
    sess.session_info = {
        "analysis_type": "continuous_multisession_concatenation",
        "history_note": (
            "Saved within-session history columns are preserved; histories are "
            "not recomputed across session boundaries."
        ),
    }
    return sess


def save_concatenated_multisession_inputs(
    concatenated: ConcatenatedSessionAnalysis,
    session: Session,
) -> None:
    """Save concatenated pre-modeling tables for inspection.

    Parameters
    ----------
    concatenated : ConcatenatedSessionAnalysis
        Concatenated block and trial tables.
    session : Session
        Synthetic multisession metadata with `processed_data_path` and
        `sess_id_full`.

    Returns
    -------
    None
        Writes `{sess_id_full}_block_performance.csv` and
        `{sess_id_full}_augmented_trials.csv` before HMM modeling modifies
        them.
    """
    session.processed_data_path.mkdir(parents=True, exist_ok=True)
    session.figure_path.mkdir(parents=True, exist_ok=True)
    block_path = session.processed_data_path / f"{session.sess_id_full}_block_performance.csv"
    trial_path = session.processed_data_path / f"{session.sess_id_full}_augmented_trials.csv"
    concatenated.block_performance.to_csv(block_path, index=False, na_rep="None")
    concatenated.augmented_trial_df.to_csv(trial_path, index=False, na_rep="None")


def run_multisession_analysis(
    concatenated: ConcatenatedSessionAnalysis,
    session: Session,
    block_num_states: int = 2,
    trial_num_states: int = 2,
    prior_alpha: float = 1,
    prior_sigma: float = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run block and trial HMM modeling on concatenated session tables.

    Parameters
    ----------
    concatenated : ConcatenatedSessionAnalysis
        Continuous concatenation output. Block rows use a continuous
        zero-based `block_ix`; trial rows use matching continuous `cur_block`.
    session : Session
        Synthetic multisession metadata controlling output filenames and
        directories.
    block_num_states : int, default=2
        Number of hidden states for block LM-HMM modeling.
    trial_num_states : int, default=2
        Number of hidden states for trial GLM-HMM modeling.
    prior_alpha : float, default=1
        Sticky-transition prior alpha passed to MAP HMM workflows.
    prior_sigma : float, default=1
        Observation prior sigma passed to MAP HMM workflows.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Modeled `(block_performance, augmented_trial_df)` tables. The trial
        table includes block-state inheritance generated by the block model.
    """
    save_concatenated_multisession_inputs(concatenated, session)
    modeled_block_df, modeled_trial_df = bssm.run_block_modeling(
        concatenated.block_performance,
        concatenated.augmented_trial_df,
        session=session,
        num_states=block_num_states,
        prior_alpha=prior_alpha,
        prior_sigma=prior_sigma,
    )
    modeled_trial_df = tssm.run_trial_modeling(
        modeled_trial_df,
        session=session,
        num_states=trial_num_states,
        prior_alpha=prior_alpha,
        prior_sigma=prior_sigma,
    )
    return modeled_block_df, modeled_trial_df


def plot_session(event_df: pd.DataFrame, session_info: dict, figure_path: Path, sess_id_full: str):
    if not figure_path.exists():
        figure_path.mkdir()

    event_df['Time'] -= event_df['Time'].min()
    raster_plots.generate_session_raster(event_df, session_info=session_info, fig_name=sess_id_full + '_raster',
                                         plot_path=figure_path, fig_format='png')
    raster_plots.colorblock_raster(event_df, sess_id_full + '_lick_context_raster', session_info=session_info,
                                   plot_path=figure_path, plot_choices=False)
    raster_plots.colorblock_raster(event_df, sess_id_full + '_choice_context_raster', session_info=session_info,
                                   plot_path=figure_path, plot_choices=True)


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


def main_multisession():
    """Analyze selected saved sessions as one continuous multisession table."""
    mouse = 'CT014'
    session_data_root = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014')
    multi_session_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
    dates = ['2025-12-05', '2025-12-16', '2025-12-23']

    plot_mouse_learning_curve(mouse=mouse, multi_session_save_path=multi_session_save_path)

    sessions = [
        find_saved_session_by_date(
            mouse=mouse,
            date=date,
            session_data_root=session_data_root,
            multi_session_save_path=multi_session_save_path,
        )
        for date in dates
    ]
    saved_sessions = [load_saved_session_analysis(session) for session in sessions]
    concatenated = concatenate_saved_sessions(saved_sessions)
    multisession = build_multisession_session(
        mouse=mouse,
        multi_session_save_path=multi_session_save_path,
        sess_id_full=f'{mouse}_multisession',
    )
    run_multisession_analysis(
        concatenated=concatenated,
        session=multisession,
        block_num_states=2,
        trial_num_states=2,
        prior_alpha=1,
        prior_sigma=1,
    )





def main_mouse():
    """Analyze a single behavior session from start to finish."""

    ### USER FLAGS - CHOOSE THESE FOR EACH RUN ###
    preprocess_raw_session = False
    run_session_analysis = True

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

    plot_session(event_df, session_info, figure_path=figure_path, sess_id_full=sess_id_full)

    # analyze trials and save, or load existing analysis outputs
    augmented_trial_df, block_performance, multisession_df = load_or_run_session_analysis(
        trial_df=trial_df,
        session=sess,
        run_session_analysis=run_session_analysis,
    )

    ### plot single session performance ###
    performance_plots.plot_session_correct(block_performance, sess.figure_path, sess_id_full)
    performance_plots.plot_session_trials_to_correct(block_performance, sess.figure_path, sess_id_full)
    performance_plots.plot_session_nswitches(block_performance, sess.figure_path, sess_id_full)

    slope = multisession_df.loc[
        multisession_df['date'] == date,
        'prev_consecutive_rewards_slope',
    ].values[0]
    intercept = multisession_df.loc[
        multisession_df['date'] == date,
        'prev_consecutive_rewards_intercept',
    ].values[0]
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
    ix_valid = bssm.make_valid_block_history_mask(block_df)
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
