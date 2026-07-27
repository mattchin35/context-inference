import json
import numpy as np
import re
import warnings
from behavior_analysis import performance_plots
from behavior_analysis import plot_model_values, raster_plots, session_analysis, simulate_priors
from mouse_behavior_preprocessing import process_behavior_log
from behavior_analysis import block_state_space_modeling as bssm
from behavior_analysis import trial_state_space_modeling as tssm
from behavior_analysis import gather_trial_features as gtf
from src.behavior_analysis import switch_persistence
from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    normalize_experimenter_reward_column,
)
# import src.state_space_modeling.utilplot as utilplot
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

AGENT_MOUSE_AGREEMENT_COLUMNS = {
    "QL": "qlearning_mouse_agreement",
    "FQL": "fql_mouse_agreement",
    "HMM": "hmm_logodds_mouse_agreement",
    "HMM decay": "hmm_logodds_decay_mouse_agreement",
    "Persev": "perseveration_mouse_agreement",
    "Doubt+P": "doubt_perseveration_mouse_agreement",
    "WSLS": "wsls_mouse_agreement",
    "Ideal": "observer_mouse_agreement",
}

CROSS_MOUSE_SESSION_METRIC_SPECS = {
    "median_TTS": {
        "label": "Median Trials to Correct",
    },
    "median_post_switch_correct": {
        "label": "Median Post-First-Correct Accuracy",
        "ylim": (0, 1),
    },
    "median_ideal_agreement": {
        "label": "Median Ideal-Agent Agreement",
        "ylim": (0, 1),
    },
}

SESSION_SUMMARY_METRIC_FAMILIES = {
    "tts": {
        "metrics": {
            "median_TTS": "median TTS",
            "q3_TTS": "Q3 TTS",
            "frac_blocks_TTS_gt_5": "fraction TTS > 5",
        },
        "ylabel": "TTS summary",
    },
    "post_switch_correct": {
        "metrics": {
            "median_post_switch_correct": "median",
            "q1_post_switch_correct": "Q1",
            "frac_blocks_post_switch_correct_lt_0p7": "fraction < 0.7",
        },
        "ylabel": "Post-switch correctness",
        "ylim": (0, 1),
    },
    "ideal_agreement": {
        "metrics": {
            "median_ideal_agreement": "median",
            "q1_ideal_agreement": "Q1",
            "frac_blocks_ideal_agreement_lt_0p6": "fraction < 0.6",
        },
        "ylabel": "Ideal agreement",
        "ylim": (0, 1),
    },
}


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


@dataclass
class SingleSessionAnalysisConfig:
    """Configuration for one complete single-session analysis run.

    Attributes
    ----------
    preprocess_raw_session : bool
        If True, regenerate processed event/trial CSVs from the raw behavior
        log. If False, load existing processed CSVs.
    run_session_analysis : bool
        If True, recompute and save session-analysis CSVs. If False, load
        existing analysis outputs.
    ideal_observer_n_replays : int
        Number of fixed replay samples used by ideal-observer summaries.
    ideal_observer_seed : int or None
        Random seed for fixed replay ideal-observer summaries.
    block_modeling_states : int
        Number of HMM states used for block modeling.
    trial_modeling_states : int
        Number of HMM states reserved for trial modeling when enabled.
    min_time : float
        Minimum trial time since session start, in seconds, retained during
        preprocessing.
    max_time : float
        Maximum trial time since session start, in seconds, retained during
        preprocessing.
    scatter_regressor : str
        Block-performance regressor column used in trials-to-correct scatter
        plots. Units depend on the selected column.
    block_hmm_random_seed : int
        Random seed for block HMM modeling.
    skip_block_hmm_if_existing : bool
        If True, reuse saved block HMM outputs when the model pickle, modeled
        block table, and inherited-strategy trial table are all present.
    trial_hmm_random_seed : int
        Random seed reserved for trial HMM modeling when enabled.
    block_secondary_trace : str
        Secondary trace mode passed to block predicted-state plots.
    plot_sliding_regression : bool
        If True, add sliding block-regression panels to block HMM plots.
    sliding_regression_window_size : int
        Number of valid blocks per sliding-regression window.
    sliding_regression_step_size : int
        Step size between sliding-regression windows, in valid blocks.
    trial_glm_predictor_columns : tuple[str, ...]
        Trial-level predictor columns reserved for trial HMM modeling.
    max_explore_run_length : int
        Maximum number of exploratory-side valid choice trials allowed in one
        explore run during trial-feature collection.
    min_explore_run_length_to_count : int
        Minimum exploratory-side run length required for an explore run to be
        counted in block summaries and plots.
    """
    preprocess_raw_session: bool = True
    run_session_analysis: bool = True
    ideal_observer_n_replays: int = 100
    ideal_observer_seed: int | None = 12345
    block_modeling_states: int = 2
    trial_modeling_states: int = 3
    min_time: float = 0
    max_time: float = np.inf
    scatter_regressor: str = "prev_n_rewarded"
    block_hmm_random_seed: int = 1001
    skip_block_hmm_if_existing: bool = False
    trial_hmm_random_seed: int = 2001
    block_secondary_trace: str = "prev_n_rewarded"
    plot_sliding_regression: bool = True
    sliding_regression_window_size: int = 10
    sliding_regression_step_size: int = 5
    max_explore_run_length: int = 5
    min_explore_run_length_to_count: int = 1
    trial_glm_predictor_columns: tuple[str, ...] = (
        "FQlearning_rel_value",
        "HMM_decay_res",
        "rel_hazard_res",
        "relative_doubt_index",
        "perseveration_regressor",
    )


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
    trial_df = normalize_experimenter_reward_column(trial_df)
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
    trial_df = normalize_experimenter_reward_column(trial_df)
    return trial_df, event_df, None


def load_or_run_session_analysis(
    trial_df: pd.DataFrame,
    session: Session,
    run_session_analysis: bool = False,
    ideal_observer_n_replays: int = 100,
    ideal_observer_seed: int | None = 12345,
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
    ideal_observer_n_replays : int, default=100
        Number of fixed-state ideal-observer replay samples used when
        `run_session_analysis` is True.
    ideal_observer_seed : int or None, default=12345
        Seed used for fixed-state ideal-observer replay sampling when
        `run_session_analysis` is True.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `(augmented_trial_df, block_performance, multisession_df)`. The load
        path normalizes `session_analysis.load_analysis`, which returns
        `(multisession_df, block_performance, augmented_trial_df)`.
    """
    if run_session_analysis:
        augmented_trial_df, block_performance, multisession_df = session_analysis.run_analysis(
            trial_df,
            session=session,
            ideal_observer_n_replays=ideal_observer_n_replays,
            ideal_observer_seed=ideal_observer_seed,
        )
        block_performance = add_block_explore_counts_from_trials(
            block_performance,
            augmented_trial_df,
        )
        return augmented_trial_df, block_performance, multisession_df

    multisession_df, block_performance, augmented_trial_df = session_analysis.load_analysis(
        session.sess_id_full,
        session_data_folder=session.processed_data_path,
        multisession_data_folder=session.multi_session_save_path,
    )
    block_performance = add_block_explore_counts_from_trials(
        block_performance,
        augmented_trial_df,
    )
    return augmented_trial_df, block_performance, multisession_df


def plot_mouse_history_ideal_observer_for_session(
    augmented_trial_df: pd.DataFrame,
    session: Session,
):
    """Plot mouse-history ideal-observer values for one analyzed session.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        `state`, `action`, `reward`, `p_active_rew`, and `p_switch`; rows
        without mouse choices or with experimenter rewards are excluded by the
        observer helper.
    session : Session
        Session metadata with `figure_path` and `sess_id_full` attributes.

    Returns
    -------
    tuple[pd.DataFrame, pathlib.Path]
        Observer run dataframe and saved PNG path.
    """
    return plot_model_values.plot_mouse_history_ideal_observer_values(
        trial_df=augmented_trial_df,
        figure_path=session.figure_path,
        sess_id_full=session.sess_id_full,
    )


def save_and_plot_switch_persistence_for_session(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
    session: Session,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Save and plot post-context-switch persistence for one session.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`, including
        `block_ix` and previous-block metrics when available.
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`, including
        `cur_block`, `state`, and `action`.
    session : Session
        Session metadata with `processed_data_path`, `figure_path`, and
        `sess_id_full` attributes.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pathlib.Path]
        `(detail_df, summary_df, plot_path)`. The detail table has one row per
        post-switch trial, and the summary table has one row per switch group
        and valid post-switch choice index.
    """
    detail_df, summary_df = switch_persistence.save_switch_persistence_outputs(
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
        processed_data_path=session.processed_data_path,
        sess_id_full=session.sess_id_full,
    )
    plot_path = performance_plots.plot_switch_persistence_summary(
        summary_df=summary_df,
        plot_path=session.figure_path,
        figure_id=session.sess_id_full,
    )
    return detail_df, summary_df, plot_path


def save_and_plot_post_first_correct_accuracy_for_session(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
    session: Session,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Save and plot post-first-correct accuracy for one session.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`, including
        `block_ix` and previous-block metrics when available.
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`, including
        `cur_block`, `state`, `action`, and `correct`.
    session : Session
        Session metadata with `processed_data_path`, `figure_path`, and
        `sess_id_full` attributes.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pathlib.Path]
        `(detail_df, summary_df, plot_path)`. The detail table has one row per
        block trial, and the summary table has one row per correct-side group
        and valid post-first-correct choice index.
    """
    detail_df, summary_df = switch_persistence.save_post_first_correct_accuracy_outputs(
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
        processed_data_path=session.processed_data_path,
        sess_id_full=session.sess_id_full,
    )
    plot_path = performance_plots.plot_post_first_correct_accuracy_summary(
        summary_df=summary_df,
        plot_path=session.figure_path,
        figure_id=session.sess_id_full,
    )
    return detail_df, summary_df, plot_path


def ensure_session_side_bias_metrics(
    multisession_df: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
    session: Session,
) -> pd.DataFrame:
    """Return a session-summary table containing side-bias metrics.

    Parameters
    ----------
    multisession_df : pd.DataFrame
        Mouse-level summary table with shape `(n_sessions, n_columns)`.
        Expected to include a `date` column when multiple sessions are present.
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`, including
        `state`, `action`, and `observer_value`.
    session : Session
        Session metadata with `date` and `sess_id_full` fields. `date` is used
        to select the summary row when possible.

    Returns
    -------
    pd.DataFrame
        Copy of `multisession_df` with the side-bias metric columns populated
        for the current session row. Existing columns are preserved.
    """
    metric_values = session_analysis.compute_session_side_bias_metrics(augmented_trial_df)
    updated_summary = multisession_df.copy()
    if updated_summary.empty:
        updated_summary = pd.DataFrame(index=[0])

    if "date" in updated_summary.columns:
        session_rows = updated_summary["date"].astype(str).eq(str(session.date))
        if not session_rows.any():
            session_row_index = updated_summary.index[0]
        else:
            session_row_index = updated_summary.index[session_rows][0]
    else:
        session_row_index = updated_summary.index[0]

    for column_name, value in metric_values.items():
        updated_summary.loc[session_row_index, column_name] = value
    return updated_summary


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


def get_block_hmm_output_paths(session: Session) -> dict[str, Path]:
    """Return the saved-output paths required to reuse block HMM results.

    Parameters
    ----------
    session : Session
        Session metadata with `processed_data_path` and `sess_id_full` fields.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping with keys `model_dict`, `block_performance`, and
        `augmented_trials`. CSV paths contain modeled block outputs and
        block-inherited trial labels; the pickle path contains the fitted HMM.
    """
    processed_data_path = Path(session.processed_data_path)
    sess_id_full = session.sess_id_full
    return {
        "model_dict": processed_data_path / f"{sess_id_full}_block_statedict.pkl",
        "block_performance": processed_data_path / f"{sess_id_full}_block_performance.csv",
        "augmented_trials": processed_data_path / f"{sess_id_full}_augmented_trials.csv",
    }


def missing_block_hmm_output_paths(session: Session) -> list[Path]:
    """List saved block HMM outputs that are absent from disk.

    Parameters
    ----------
    session : Session
        Session metadata with `processed_data_path` and `sess_id_full` fields.

    Returns
    -------
    list[pathlib.Path]
        Required output paths that do not exist. An empty list means the saved
        block HMM run can be reused by file-presence checks.
    """
    return [
        output_path
        for output_path in get_block_hmm_output_paths(session).values()
        if not output_path.exists()
    ]


def block_hmm_outputs_exist(session: Session) -> bool:
    """Check whether all saved block HMM outputs needed for reuse exist.

    Parameters
    ----------
    session : Session
        Session metadata with `processed_data_path` and `sess_id_full` fields.

    Returns
    -------
    bool
        True when the fitted model pickle, modeled block CSV, and inherited
        trial CSV are all present.
    """
    return len(missing_block_hmm_output_paths(session)) == 0


def load_saved_block_hmm_outputs(session: Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved block HMM-modeled block and trial tables.

    Parameters
    ----------
    session : Session
        Session metadata with `processed_data_path` and `sess_id_full` fields.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        `(block_performance, augmented_trial_df)` loaded from saved CSVs. The
        block table has shape `(n_blocks, n_block_columns)` and the trial table
        has shape `(n_trials, n_trial_columns)`.

    Raises
    ------
    FileNotFoundError
        If any required block HMM output is missing.
    """
    missing_paths = missing_block_hmm_output_paths(session)
    if missing_paths:
        missing_summary = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Cannot reuse saved block HMM outputs for {session.sess_id_full}; "
            f"missing required files:\n{missing_summary}"
        )

    output_paths = get_block_hmm_output_paths(session)
    block_performance = pd.read_csv(output_paths["block_performance"])
    augmented_trial_df = pd.read_csv(output_paths["augmented_trials"])
    return block_performance, augmented_trial_df


def find_raw_session_by_date(
    mouse: str,
    date: str,
    session_data_root: Path,
    task_tag: str,
    multi_session_save_path: Path,
    behavior_timestamp: str | None = None,
) -> Session:
    """Resolve one raw behavior session from a mouse/date pair.

    Parameters
    ----------
    mouse : str
        Mouse identifier used in the session folder and raw timestamp folder.
    date : str
        Session date formatted as `YYYY-MM-DD`.
    session_data_root : pathlib.Path
        Mouse-level data directory containing exact task-tagged session
        folders such as `{mouse}_YYYYMMDD_{task_tag}`.
    task_tag : str
        Exact task tag used in the session folder name. This is not globbed.
    multi_session_save_path : pathlib.Path
        Cross-session output directory assigned to returned session metadata.
    behavior_timestamp : str or None, optional
        Explicit behavior timestamp formatted as `HHMMSS`. If provided, the
        exact timestamp folder is loaded even when other timestamp folders
        exist for the date. If None, exactly one timestamp folder must exist.

    Returns
    -------
    Session
        Session metadata for the single matching raw timestamp folder. The
        session-info pickle is loaded into `session_info`.

    Raises
    ------
    FileNotFoundError
        If the exact task-tagged session folder, `rpi` folder, timestamp
        folder, or session-info pickle is missing.
    ValueError
        If more than one timestamp folder matches the requested date, or the
        resolved folder name does not follow `mouse_YYYY-MM-DD_HHMMSS`.
    """
    date_no_dash = date.replace("-", "")
    session_data_home = session_data_root / f"{mouse}_{date_no_dash}_{task_tag}"
    if not session_data_home.exists():
        raise FileNotFoundError(
            f"No exact task-tagged session folder found for {mouse} on {date}: "
            f"{session_data_home}"
        )

    rpi_path = session_data_home / "rpi"
    if not rpi_path.exists():
        raise FileNotFoundError(f"No rpi folder found for {mouse} on {date}: {rpi_path}")

    session_name_pattern = re.compile(rf"^{re.escape(mouse)}_{re.escape(date)}_(\d{{6}})$")
    if behavior_timestamp is None:
        timestamp_folders = sorted(
            folder for folder in rpi_path.glob(f"{mouse}_{date}_*")
            if folder.is_dir() and session_name_pattern.match(folder.name) is not None
        )
        if len(timestamp_folders) == 0:
            raise FileNotFoundError(
                f"No raw timestamp folder found for {mouse} on {date} under {rpi_path}."
            )
        if len(timestamp_folders) > 1:
            matched_paths = "\n".join(str(path) for path in timestamp_folders)
            raise ValueError(
                f"Expected one raw timestamp folder for {mouse} on {date}, "
                f"found {len(timestamp_folders)}:\n{matched_paths}"
            )
        raw_behavior_folder = timestamp_folders[0]
    else:
        raw_behavior_folder = rpi_path / f"{mouse}_{date}_{behavior_timestamp}"
        if not raw_behavior_folder.exists():
            raise FileNotFoundError(
                f"No raw timestamp folder found for {mouse} on {date} with "
                f"timestamp {behavior_timestamp}: {raw_behavior_folder}"
            )

    match = session_name_pattern.match(raw_behavior_folder.name)
    if match is None:
        raise ValueError(
            f"Raw timestamp folder does not match expected session pattern: "
            f"{raw_behavior_folder.name}"
        )

    timestamp = match.group(1)
    sess_id_full = raw_behavior_folder.name
    session_info_path = raw_behavior_folder / f"{sess_id_full}_session_info.pkl"
    if not session_info_path.exists():
        raise FileNotFoundError(
            f"No session-info pickle found for {mouse} on {date}: {session_info_path}"
        )
    with open(session_info_path, "rb") as file:
        session_info = pkl.load(file)

    sess = Session()
    sess.multi_session_save_path = multi_session_save_path
    sess.session_data_home = session_data_home
    sess.sess_id_full = sess_id_full
    sess.sess_id_abbreviated = f"{mouse}_{date}"
    sess.raw_behavior_folder = raw_behavior_folder
    sess.processed_data_path = session_data_home / "processed"
    sess.figure_path = session_data_home / "figures"
    sess.mouse = mouse
    sess.date = date
    sess.timestamp = timestamp
    sess.session_info_fname = session_info_path
    sess.session_info = session_info
    return sess


def find_raw_session_dates_for_task_tag(
    mouse: str,
    session_data_root: Path,
    task_tag: str,
) -> list[str]:
    """Find session dates for one mouse and task tag under a mouse data root.

    Parameters
    ----------
    mouse : str
        Mouse identifier expected at the start of each session folder name.
    session_data_root : pathlib.Path
        Mouse-level data directory. Only direct child directories are scanned;
        discovery is not recursive.
    task_tag : str
        Exact task tag expected at the end of each session folder name.

    Returns
    -------
    list[str]
        Sorted session dates formatted as `YYYY-MM-DD`. Dates are parsed from
        direct child folders named `{mouse}_YYYYMMDD_{task_tag}`.

    Raises
    ------
    FileNotFoundError
        If no exact task-tagged session folders are found under
        `session_data_root`.
    """
    session_folder_pattern = re.compile(
        rf"^{re.escape(mouse)}_(\d{{8}})_{re.escape(task_tag)}$"
    )
    dates = []
    for child_path in session_data_root.iterdir():
        if not child_path.is_dir():
            continue
        match = session_folder_pattern.match(child_path.name)
        if match is None:
            continue
        raw_date = match.group(1)
        dates.append(f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}")

    if not dates:
        raise FileNotFoundError(
            f"No session folders matching {mouse}_YYYYMMDD_{task_tag} found under "
            f"{session_data_root}."
        )
    return sorted(dates)


def resolve_multisession_dates(
    mouse: str,
    session_data_root: Path,
    task_tag: str,
    explicit_dates: Sequence[str],
    use_all_dates_for_task_tag: bool,
    require_saved_outputs: bool = True,
    multi_session_save_path: Path | None = None,
) -> list[str]:
    """Choose multisession dates from an explicit list or task-tag discovery.

    Parameters
    ----------
    mouse : str
        Mouse identifier used in session folder and saved CSV names.
    session_data_root : pathlib.Path
        Mouse-level data directory. Task-tag discovery scans only direct child
        folders named `{mouse}_YYYYMMDD_{task_tag}`.
    task_tag : str
        Exact task tag used for automatic date discovery.
    explicit_dates : Sequence[str]
        User-provided dates formatted as `YYYY-MM-DD`. These are returned
        unchanged when `use_all_dates_for_task_tag` is False.
    use_all_dates_for_task_tag : bool
        If True, ignore `explicit_dates` and discover all dates matching
        `task_tag` under `session_data_root`.
    require_saved_outputs : bool, default=True
        If True and date discovery is enabled, require each discovered date to
        have exactly one saved augmented-trials CSV usable by multisession
        analysis. Explicit dates are not validated here.
    multi_session_save_path : pathlib.Path or None, optional
        Cross-session output directory passed to saved-session validation. If
        None, defaults to `session_data_root / "cross_session_analysis"`.

    Returns
    -------
    list[str]
        Dates formatted as `YYYY-MM-DD`, either preserving `explicit_dates`
        order or sorted by discovered session folder date.
    """
    if not use_all_dates_for_task_tag:
        return list(explicit_dates)

    resolved_dates = find_raw_session_dates_for_task_tag(
        mouse=mouse,
        session_data_root=session_data_root,
        task_tag=task_tag,
    )
    if not require_saved_outputs:
        return resolved_dates

    validation_save_path = (
        multi_session_save_path
        if multi_session_save_path is not None
        else session_data_root / "cross_session_analysis"
    )
    for date in resolved_dates:
        try:
            find_saved_session_by_date(
                mouse=mouse,
                date=date,
                session_data_root=session_data_root,
                multi_session_save_path=validation_save_path,
            )
        except (FileNotFoundError, ValueError) as error:
            raise type(error)(
                f"Discovered multisession date {date} cannot be used because "
                f"saved analysis outputs are missing or ambiguous: {error}"
            ) from error

    return resolved_dates


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
    block_performance = add_block_explore_counts_from_trials(
        block_performance,
        augmented_trial_df,
    )
    block_performance = add_block_explore_run_counts_from_trials(
        block_performance,
        augmented_trial_df,
    )
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


def _mouse_from_session_metadata(session) -> str:
    """Return the mouse identifier from a session-like object.

    Parameters
    ----------
    session : object
        Session-like object with `mouse` and/or `sess_id_full` attributes.
        `sess_id_full` is expected to contain `{mouse}_{YYYY-MM-DD}_{HHMMSS}`.

    Returns
    -------
    str
        Mouse identifier.

    Raises
    ------
    ValueError
        If neither `mouse` nor a parseable `sess_id_full` is available.
    """
    mouse = getattr(session, "mouse", None)
    if mouse is not None:
        return str(mouse)

    sess_id_full = getattr(session, "sess_id_full", None)
    if sess_id_full is None:
        raise ValueError("Session metadata is missing both 'mouse' and 'sess_id_full'.")
    match = re.search(r"(.+?)_\d{4}-\d{2}-\d{2}_\d{6}", str(sess_id_full))
    if match is None:
        raise ValueError(f"Cannot parse mouse from session id: {sess_id_full}")
    return match.group(1)


def _add_mouse_metadata_columns(
    table: pd.DataFrame,
    mouse: str,
    source_mouse: str | None = None,
) -> pd.DataFrame:
    """Attach mouse identifiers to a block or trial table.

    Parameters
    ----------
    table : pandas.DataFrame
        Analysis table with shape `(n_rows, n_columns)`.
    mouse : str
        Mouse identifier for the row in the current analysis context.
    source_mouse : str or None, default=None
        Original mouse identifier before concatenation. None uses `mouse`.

    Returns
    -------
    pandas.DataFrame
        Copy of `table` with `mouse` and `source_mouse` columns. Existing
        columns with these names are overwritten with the supplied metadata.
    """
    table_with_metadata = table.copy()
    source_mouse = mouse if source_mouse is None else source_mouse
    metadata_values = {"mouse": mouse, "source_mouse": source_mouse}
    for column_name, column_value in metadata_values.items():
        table_with_metadata[column_name] = column_value
    leading_columns = [column for column in metadata_values if column in table_with_metadata.columns]
    remaining_columns = [
        column for column in table_with_metadata.columns if column not in leading_columns
    ]
    return table_with_metadata.loc[:, leading_columns + remaining_columns]


def _add_single_session_metadata_columns(
    table: pd.DataFrame,
    session: Session,
) -> pd.DataFrame:
    """Attach mouse, session, and date identifiers to a single-session table.

    Parameters
    ----------
    table : pandas.DataFrame
        Analysis table with shape `(n_rows, n_columns)`.
    session : Session
        Session metadata object with `mouse`, `sess_id_full`, and `date`
        attributes.

    Returns
    -------
    pandas.DataFrame
        Copy of `table` with `mouse`, `session_id`, and `date` columns.
        Existing columns with these names are overwritten.
    """
    table_with_metadata = table.copy()
    metadata_values = {
        "mouse": str(session.mouse),
        "session_id": str(session.sess_id_full),
        "date": str(session.date),
    }
    for column_name, column_value in metadata_values.items():
        table_with_metadata[column_name] = column_value
    leading_columns = [column for column in metadata_values if column in table_with_metadata.columns]
    remaining_columns = [
        column for column in table_with_metadata.columns if column not in leading_columns
    ]
    return table_with_metadata.loc[:, leading_columns + remaining_columns]


def _require_columns(table: pd.DataFrame, required_columns: Sequence[str], context: str) -> None:
    """Raise a clear error when an analysis table lacks required columns.

    Parameters
    ----------
    table : pandas.DataFrame
        Analysis table with shape `(n_rows, n_columns)`.
    required_columns : Sequence[str]
        Column names required by the caller, checked in the supplied order.
    context : str
        Human-readable source description used in the error message.

    Returns
    -------
    None
        Returns None when all required columns are present.

    Raises
    ------
    ValueError
        If any required column is absent.
    """
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"{context} is missing required columns: {missing_columns}")


def _add_centered_prev_rewards_column(
    table: pd.DataFrame,
    output_column: str,
) -> pd.DataFrame:
    """Add previous-reward counts centered over all numeric rows in a table.

    Parameters
    ----------
    table : pandas.DataFrame
        Block table with shape `(n_blocks, n_columns)`. Must contain
        `prev_n_rewarded`, measured in rewarded trials in the previous block.
    output_column : str
        Name of the centered output column. Values are in rewarded trials
        relative to the mean of all numeric `prev_n_rewarded` rows.

    Returns
    -------
    pandas.DataFrame
        Copy of `table` with `output_column` added. Nonnumeric source rows are
        assigned the string sentinel `"None"`.
    """
    _require_columns(table, ["prev_n_rewarded"], "block table")
    output_df = table.copy()
    prev_rewards = pd.to_numeric(output_df["prev_n_rewarded"], errors="coerce")
    numeric_rows = prev_rewards.notna()
    centered_values = np.full(output_df.shape[0], "None", dtype=object)
    if numeric_rows.any():
        centered_values[numeric_rows.to_numpy()] = (
            prev_rewards.loc[numeric_rows] - prev_rewards.loc[numeric_rows].mean()
        ).to_numpy(dtype=float)
    output_df[output_column] = centered_values
    return output_df


def _require_session_block_collection_columns(block_performance: pd.DataFrame, context: str) -> None:
    """Require single-session block variables needed for multisession collection.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Single-session block table with shape `(n_blocks, n_columns)`.
    context : str
        Human-readable source description used in the error message.

    Returns
    -------
    None
        Returns None when all required collection columns are present.
    """
    _require_columns(
        block_performance,
        ["prev_rewards_session_centered", "block_side_code"],
        context,
    )


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
        source_mouse = _mouse_from_session_metadata(session)
        block_df = saved_session.block_performance.copy()
        trial_df = saved_session.augmented_trial_df.copy()
        block_df = _add_mouse_metadata_columns(
            block_df,
            mouse=source_mouse,
            source_mouse=source_mouse,
        )
        trial_df = _add_mouse_metadata_columns(
            trial_df,
            mouse=source_mouse,
            source_mouse=source_mouse,
        )

        if "block_ix" not in block_df.columns:
            raise ValueError(f"{session.sess_id_full} block_performance is missing 'block_ix'.")
        if "cur_block" not in trial_df.columns:
            raise ValueError(f"{session.sess_id_full} augmented_trial_df is missing 'cur_block'.")
        _require_session_block_collection_columns(
            block_df,
            context=f"{session.sess_id_full} block_performance",
        )

        raw_trial_block_ids = pd.Series(
            sorted(pd.to_numeric(trial_df["cur_block"], errors="raise").astype(int).unique())
        )
        if raw_trial_block_ids.shape[0] != block_df.shape[0]:
            raise ValueError(
                f"{session.sess_id_full} has {raw_trial_block_ids.shape[0]} trial blocks but "
                f"{block_df.shape[0]} block rows; cannot align raw trial block ids to block_performance."
            )

        block_df["source_session_id"] = session.sess_id_full
        block_df["source_date"] = session.date
        block_df["source_session_index"] = session_index
        block_df["source_block_ix"] = raw_trial_block_ids.to_numpy()
        block_df["source_block_row"] = block_df["block_ix"]
        block_df["block_ix"] = np.arange(block_df.shape[0], dtype=int) + block_offset
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

    concatenated_block_performance = pd.concat(concatenated_blocks, axis=0, ignore_index=True)
    concatenated_block_performance = _add_centered_prev_rewards_column(
        concatenated_block_performance,
        output_column="prev_rewards_mouse_centered",
    )

    return ConcatenatedSessionAnalysis(
        block_performance=concatenated_block_performance,
        augmented_trial_df=pd.concat(concatenated_trials, axis=0, ignore_index=True),
        block_session_lengths=np.asarray(block_session_lengths, dtype=int),
        trial_session_lengths=np.asarray(trial_session_lengths, dtype=int),
    )


def prepare_learning_curve_data(
    mouse: str,
    multi_session_save_path: Path,
    learning_regressor: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load numeric mouse learning-curve arrays from the summary CSV.

    Parameters
    ----------
    mouse : str
        Mouse identifier used in `{mouse}_overall_performance.csv`.
    multi_session_save_path : Path
        Directory containing the mouse-level cross-session summary CSV.
    learning_regressor : str or None, default=None
        Regression predictor prefix to plot. The function reads
        `{learning_regressor}_slope` from the summary CSV. If None, uses
        `performance_plots.DEFAULT_LEARNING_REGRESSOR`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - regression coefficients, shape `(n_valid_sessions,)`, numeric slope
          values for the requested learning regressor
        - block counts, shape `(n_valid_sessions,)`, numeric session block
          counts
        - dates, shape `(n_valid_sessions,)`, date labels sorted ascending

        Rows with nonnumeric slope values such as the string `"None"` are
        dropped before plotting.
    """
    multisession_summary_path = multi_session_save_path / f"{mouse}_overall_performance.csv"
    multisession_df = pd.read_csv(multisession_summary_path, sep=",", na_filter=False)
    multisession_df.sort_values(by="date", inplace=True)
    multisession_df.reset_index(drop=True, inplace=True)

    if hasattr(performance_plots, "get_session_block_count_column"):
        block_count_col = performance_plots.get_session_block_count_column(multisession_df)
    elif "n_blocks" in multisession_df.columns:
        block_count_col = "n_blocks"
    elif "n_switches" in multisession_df.columns:
        block_count_col = "n_switches"
    else:
        raise ValueError("multisession summary must contain 'n_blocks' or legacy 'n_switches'.")
    if learning_regressor is None:
        learning_regressor = getattr(
            performance_plots,
            "DEFAULT_LEARNING_REGRESSOR",
            "prev_consecutive_rewards",
        )
    learning_column = f"{learning_regressor}_slope"
    if learning_column not in multisession_df.columns:
        raise ValueError(
            f"overall performance summary is missing '{learning_column}'. "
            "Rerun session analyses to calculate this learning regressor."
        )

    multisession_df[learning_column] = pd.to_numeric(
        multisession_df[learning_column],
        errors="coerce",
    )
    multisession_df[block_count_col] = pd.to_numeric(
        multisession_df[block_count_col],
        errors="coerce",
    )
    valid_learning_rows = multisession_df[learning_column].notna()
    plot_df = multisession_df[valid_learning_rows].copy()
    return (
        plot_df[learning_column].to_numpy(dtype=float),
        plot_df[block_count_col].to_numpy(),
        plot_df["date"].to_numpy(),
    )


def discover_mice_with_overall_performance(data_root: Path) -> list[str]:
    """Find mouse folders with saved cross-session overall-performance summaries.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory with shape `(n_mouse_folders,)` at its first level. Each
        candidate mouse folder is expected to contain
        `cross_session_analysis/{mouse}_overall_performance.csv`.

    Returns
    -------
    list[str]
        Sorted mouse identifiers whose expected summary CSV exists. Folder
        names are used as mouse identifiers.
    """
    discovered_mice = []
    for mouse_path in sorted(Path(data_root).iterdir()):
        if not mouse_path.is_dir():
            continue
        mouse = mouse_path.name
        summary_path = mouse_path / "cross_session_analysis" / f"{mouse}_overall_performance.csv"
        if summary_path.exists():
            discovered_mice.append(mouse)
    return discovered_mice


def multisession_block_performance_path(data_root: Path, mouse: str) -> Path:
    """Build the expected saved multisession block-performance CSV path.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse.
    mouse : str
        Mouse identifier. The expected file is
        `{data_root}/{mouse}/cross_session_analysis/{mouse}_multisession_block_performance.csv`.

    Returns
    -------
    pathlib.Path
        Expected multisession block-performance CSV path.
    """
    return (
        Path(data_root)
        / mouse
        / "cross_session_analysis"
        / f"{mouse}_multisession_block_performance.csv"
    )


def _fill_missing_metadata_column(
    table: pd.DataFrame,
    column_name: str,
    fill_value: str,
) -> pd.DataFrame:
    """Fill absent or sentinel metadata values with a known identifier.

    Parameters
    ----------
    table : pandas.DataFrame
        Analysis table with shape `(n_rows, n_columns)`.
    column_name : str
        Metadata column to add or patch.
    fill_value : str
        Value used where `column_name` is absent, empty, or the string `"None"`.

    Returns
    -------
    pandas.DataFrame
        Copy of `table` with `column_name` present and missing metadata filled.
    """
    table_with_metadata = table.copy()
    if column_name not in table_with_metadata.columns:
        table_with_metadata[column_name] = fill_value
        return table_with_metadata

    missing_values = table_with_metadata[column_name].isna() | table_with_metadata[column_name].isin(
        ["", "None"]
    )
    table_with_metadata.loc[missing_values, column_name] = fill_value
    return table_with_metadata


def load_cross_mouse_multisession_block_performance(
    data_root: Path,
    mice: Sequence[str],
) -> pd.DataFrame:
    """Load saved multisession block-performance tables across mice.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse. Each selected mouse must
        contain `cross_session_analysis/{mouse}_multisession_block_performance.csv`.
    mice : Sequence[str]
        Mouse identifiers to load in the requested order.

    Returns
    -------
    pandas.DataFrame
        Concatenated block-performance table with shape
        `(n_total_blocks, n_columns)`. Legacy CSVs without `mouse` or
        `source_mouse` columns are filled from the selected mouse name.

    Raises
    ------
    FileNotFoundError
        If any selected mouse lacks its multisession block-performance CSV.
    """
    mouse_frames = []
    for mouse_order, mouse in enumerate(mice):
        block_path = multisession_block_performance_path(data_root, mouse)
        if not block_path.exists():
            raise FileNotFoundError(
                f"{mouse} is missing multisession block performance: {block_path}"
            )

        mouse_df = pd.read_csv(block_path, sep=",", na_filter=False)
        _require_columns(
            mouse_df,
            [
                "prev_rewards_session_centered",
                "prev_rewards_mouse_centered",
                "block_side_code",
                "prev_n_rewarded",
            ],
            context=f"{mouse} multisession block performance",
        )
        mouse_df = _fill_missing_metadata_column(mouse_df, "mouse", mouse)
        mouse_df = _fill_missing_metadata_column(mouse_df, "source_mouse", mouse)
        if "session_id" not in mouse_df.columns and "source_session_id" in mouse_df.columns:
            mouse_df["session_id"] = mouse_df["source_session_id"]
        elif "session_id" in mouse_df.columns and "source_session_id" in mouse_df.columns:
            missing_session_id = mouse_df["session_id"].isna() | mouse_df["session_id"].isin(
                ["", "None"]
            )
            mouse_df.loc[missing_session_id, "session_id"] = mouse_df.loc[
                missing_session_id,
                "source_session_id",
            ]
        mouse_df["mouse_order"] = mouse_order
        leading_columns = [
            column
            for column in ["mouse", "source_mouse", "mouse_order", "session_id"]
            if column in mouse_df.columns
        ]
        remaining_columns = [column for column in mouse_df.columns if column not in leading_columns]
        mouse_frames.append(mouse_df.loc[:, leading_columns + remaining_columns])

    if not mouse_frames:
        return pd.DataFrame(
            columns=[
                "mouse",
                "source_mouse",
                "mouse_order",
                "session_id",
                "prev_rewards_global_centered",
            ]
        )
    cross_mouse_df = pd.concat(mouse_frames, axis=0, ignore_index=True)
    return _add_centered_prev_rewards_column(
        cross_mouse_df,
        output_column="prev_rewards_global_centered",
    )


def collect_cross_mouse_multisession_block_performance(
    data_root: Path,
    output_path: Path,
    mice: Sequence[str],
    output_filename: str = "cross_mouse_multisession_block_performance.csv",
) -> Path:
    """Save a combined cross-mouse multisession block-performance CSV.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse.
    output_path : pathlib.Path
        Directory where the combined CSV is saved.
    mice : Sequence[str]
        Mouse identifiers to include.
    output_filename : str, default="cross_mouse_multisession_block_performance.csv"
        Name of the combined output CSV.

    Returns
    -------
    pathlib.Path
        Path to the saved combined CSV.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cross_mouse_df = load_cross_mouse_multisession_block_performance(
        data_root=data_root,
        mice=mice,
    )
    output_csv_path = output_path / output_filename
    cross_mouse_df.to_csv(output_csv_path, index=False, na_rep="None")
    return output_csv_path


BLOCK_RESIDUAL_MODEL_SUMMARY_COLUMNS = (
    "session_id",
    "mouse",
    "date",
    "model_formula",
    "model_type",
    "lambda_choice",
    "lambda_value",
    "alpha",
    "n_valid_blocks",
    "coefficient_intercept",
    "coefficient_prev_n_rewarded",
    "coefficient_block_side_left",
    "coefficient_prev_n_rewarded_block_side_left",
    "right_intercept",
    "left_intercept",
    "average_intercept",
    "right_reward_slope",
    "left_reward_slope",
    "average_reward_slope",
    "side_intercept_delta_left_minus_right",
    "side_reward_slope_delta_left_minus_right",
    "residual_mad_raw",
    "residual_mad_scaled",
    "residual_iqr",
    "residual_rmse",
    "residual_sd",
)
BLOCK_RESIDUAL_MODEL_FORMULAS = ("rewards_x_side", "rewards_plus_side")


def _block_residual_model_summary_filename(owner_id: str, model_formula: str) -> str:
    """Build a formula-specific residual-model summary filename."""
    if model_formula not in BLOCK_RESIDUAL_MODEL_FORMULAS:
        raise ValueError(f"Unknown block residual model formula: {model_formula}")
    return f"{owner_id}_block_residual_model_summary_{model_formula}.csv"


def block_residual_model_summary_path_for_session(
    session: Session,
    model_formula: str = "rewards_x_side",
) -> Path:
    """Build the residual-model summary CSV path for one saved session.

    Parameters
    ----------
    session : Session
        Single-session metadata with `processed_data_path` and `sess_id_full`.
    model_formula : str, default="rewards_x_side"
        Formula-specific residual model summary to load.

    Returns
    -------
    pathlib.Path
        Path to the formula-specific summary inside the session processed-data
        directory.
    """
    return (
        Path(session.processed_data_path)
        / _block_residual_model_summary_filename(session.sess_id_full, model_formula)
    )


def load_session_block_residual_model_summary(
    session: Session,
    model_formula: str = "rewards_x_side",
) -> pd.DataFrame:
    """Load and validate one per-session residual-model summary CSV.

    Parameters
    ----------
    session : Session
        Single-session metadata with `processed_data_path`, `sess_id_full`,
        `mouse`, and `date` attributes.
    model_formula : str, default="rewards_x_side"
        Formula-specific residual model summary to load.

    Returns
    -------
    pandas.DataFrame
        Residual-model summary table with one row per model/lambda pair. String
        missing-value sentinels are preserved.
    """
    summary_path = block_residual_model_summary_path_for_session(
        session,
        model_formula=model_formula,
    )
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing block residual model summary: {summary_path}")

    summary_df = pd.read_csv(summary_path, sep=",", na_filter=False)
    _require_columns(
        summary_df,
        BLOCK_RESIDUAL_MODEL_SUMMARY_COLUMNS,
        context=f"{session.sess_id_full} block residual model summary",
    )
    return summary_df


def collect_mouse_block_residual_model_summaries(
    saved_sessions: Sequence[SavedSessionAnalysis],
    output_path: Path,
    mouse: str,
    model_formula: str = "rewards_x_side",
) -> Path:
    """Collect per-session residual-model summaries for one mouse.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved sessions in the desired mouse-level collection set. Each session
        must have a `{sess_id_full}_block_residual_model_summary.csv` in its
        processed-data directory.
    output_path : pathlib.Path
        Mouse-level cross-session output directory.
    mouse : str
        Mouse identifier used in the output file name and metadata columns.
    model_formula : str, default="rewards_x_side"
        Formula-specific residual model summary to collect.

    Returns
    -------
    pathlib.Path
        Saved formula-specific mouse-level summary path.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    sorted_sessions = sorted(saved_sessions, key=lambda saved: str(saved.session.date))

    for training_day, saved_session in enumerate(sorted_sessions, start=1):
        session = saved_session.session
        session_df = load_session_block_residual_model_summary(
            session,
            model_formula=model_formula,
        )
        session_df = session_df.copy()
        session_df["mouse"] = mouse
        session_df["date"] = str(session.date)
        session_df["session_id"] = str(session.sess_id_full)
        session_df["model_formula"] = model_formula
        session_df["training_day"] = training_day
        rows.append(session_df)

    if rows:
        mouse_summary_df = pd.concat(rows, axis=0, ignore_index=True)
        mouse_summary_df.sort_values(
            by=["date", "session_id", "model_formula", "model_type", "lambda_choice"],
            inplace=True,
        )
        mouse_summary_df.reset_index(drop=True, inplace=True)
    else:
        mouse_summary_df = pd.DataFrame(
            columns=[*BLOCK_RESIDUAL_MODEL_SUMMARY_COLUMNS, "training_day"]
        )

    output_csv_path = output_path / _block_residual_model_summary_filename(
        mouse,
        model_formula,
    )
    mouse_summary_df.to_csv(output_csv_path, index=False, na_rep="None")
    return output_csv_path


def block_residual_model_summary_path(
    data_root: Path,
    mouse: str,
    model_formula: str = "rewards_x_side",
) -> Path:
    """Build the mouse-level residual-model summary path.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse.
    mouse : str
        Mouse identifier.
    model_formula : str, default="rewards_x_side"
        Formula-specific residual model summary to load.

    Returns
    -------
    pathlib.Path
        Formula-specific mouse-level summary path.
    """
    return (
        Path(data_root)
        / mouse
        / "cross_session_analysis"
        / _block_residual_model_summary_filename(mouse, model_formula)
    )


def load_cross_mouse_block_residual_model_summaries(
    data_root: Path,
    mice: Sequence[str],
    model_formula: str = "rewards_x_side",
) -> pd.DataFrame:
    """Load mouse-level residual-model summaries across mice.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse.
    mice : Sequence[str]
        Mouse identifiers to include in the combined table.
    model_formula : str, default="rewards_x_side"
        Formula-specific residual model summary to load.

    Returns
    -------
    pandas.DataFrame
        Combined residual-model summary rows with `source_mouse` and
        `mouse_order` columns added.
    """
    frames = []
    for mouse_order, mouse in enumerate(mice):
        summary_path = block_residual_model_summary_path(
            data_root,
            mouse,
            model_formula=model_formula,
        )
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{mouse} is missing block residual model summary: {summary_path}"
            )

        summary_df = pd.read_csv(summary_path, sep=",", na_filter=False)
        _require_columns(
            summary_df,
            [*BLOCK_RESIDUAL_MODEL_SUMMARY_COLUMNS, "training_day"],
            context=f"{mouse} block residual model summary",
        )
        summary_df = summary_df.copy()
        summary_df["model_formula"] = model_formula
        summary_df["source_mouse"] = mouse
        summary_df["mouse_order"] = mouse_order
        frames.append(summary_df)

    if not frames:
        return pd.DataFrame(
            columns=[
                *BLOCK_RESIDUAL_MODEL_SUMMARY_COLUMNS,
                "training_day",
                "source_mouse",
                "mouse_order",
            ]
        )
    return pd.concat(frames, axis=0, ignore_index=True)


def collect_cross_mouse_block_residual_model_summaries(
    data_root: Path,
    output_path: Path,
    mice: Sequence[str],
    model_formula: str = "rewards_x_side",
    model_formulas: Sequence[str] | None = None,
    output_filename: str | None = None,
) -> Path | dict[str, Path]:
    """Save a combined cross-mouse residual-model summary CSV.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse.
    output_path : pathlib.Path
        Directory where the combined CSV is saved.
    mice : Sequence[str]
        Mouse identifiers to include.
    model_formula : str, default="rewards_x_side"
        Formula-specific residual model summary to collect when
        `model_formulas` is None.
    model_formulas : Sequence[str] or None, default=None
        If provided, collect one output CSV per formula and return a mapping.
    output_filename : str or None, default=None
        Optional name of the combined output CSV for single-formula collection.
        None uses the formula-specific convention.

    Returns
    -------
    pathlib.Path or dict[str, pathlib.Path]
        Path to the saved cross-mouse summary CSV for single-formula
        collection, or a mapping from formula identifier to saved path.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if model_formulas is not None:
        return {
            formula: collect_cross_mouse_block_residual_model_summaries(
                data_root=data_root,
                output_path=output_path,
                mice=mice,
                model_formula=formula,
            )
            for formula in model_formulas
        }

    cross_mouse_df = load_cross_mouse_block_residual_model_summaries(
        data_root=data_root,
        mice=mice,
        model_formula=model_formula,
    )
    if output_filename is None:
        output_filename = f"cross_mouse_block_residual_model_summary_{model_formula}.csv"
    output_csv_path = output_path / output_filename
    cross_mouse_df.to_csv(output_csv_path, index=False, na_rep="None")
    return output_csv_path


def load_cross_mouse_learning_curve_data(
    data_root: Path,
    mice: Sequence[str],
    learning_regressor: str = "prev_n_rewarded",
) -> pd.DataFrame:
    """Load long-form learning-curve rows across mice.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse. Each mouse folder must
        contain `cross_session_analysis/{mouse}_overall_performance.csv`.
    mice : Sequence[str]
        Mouse identifiers to load. Each identifier is matched to a same-named
        child folder under `data_root`.
    learning_regressor : str, default="prev_n_rewarded"
        Regression predictor prefix. The loader reads
        `{learning_regressor}_slope`, in trials-to-switch per predictor unit.

    Returns
    -------
    pd.DataFrame
        Long-form dataframe with shape `(n_valid_mouse_sessions, 5)` and
        columns `mouse`, `date`, `training_day`, `learning_regressor`, and
        `slope`. `training_day` is assigned after sorting all sessions for a
        mouse by date and before invalid slopes are dropped, so invalid rows
        leave gaps in the x-axis rather than renumbering later sessions.
    """
    data_root = Path(data_root)
    learning_column = f"{learning_regressor}_slope"
    mouse_frames = []

    for mouse in mice:
        summary_path = data_root / mouse / "cross_session_analysis" / f"{mouse}_overall_performance.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing overall performance summary for {mouse}: {summary_path}")

        mouse_df = pd.read_csv(summary_path, sep=",", na_filter=False)
        if "date" not in mouse_df.columns:
            raise ValueError(f"{summary_path} is missing required 'date' column.")
        if learning_column not in mouse_df.columns:
            raise ValueError(
                f"{mouse} overall performance summary is missing '{learning_column}'. "
                "Rerun session analyses to calculate this learning regressor."
            )

        mouse_df = mouse_df.sort_values(by="date").reset_index(drop=True)
        mouse_df["training_day"] = np.arange(1, mouse_df.shape[0] + 1, dtype=int)
        mouse_df["slope"] = pd.to_numeric(mouse_df[learning_column], errors="coerce")
        valid_mouse_df = mouse_df[mouse_df["slope"].notna()].copy()
        if valid_mouse_df.empty:
            continue
        valid_mouse_df["mouse"] = mouse
        valid_mouse_df["learning_regressor"] = learning_regressor
        mouse_frames.append(
            valid_mouse_df.loc[
                :,
                ["mouse", "date", "training_day", "learning_regressor", "slope"],
            ]
        )

    if not mouse_frames:
        return pd.DataFrame(
            columns=["mouse", "date", "training_day", "learning_regressor", "slope"]
        )
    return pd.concat(mouse_frames, axis=0, ignore_index=True)


def load_cross_mouse_session_metric_data(
    data_root: Path,
    mice: Sequence[str],
    metric_column: str,
    metric_label: str,
) -> pd.DataFrame:
    """Load long-form session metric rows across mice.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse. Each mouse folder must
        contain `cross_session_analysis/{mouse}_overall_performance.csv`.
    mice : Sequence[str]
        Mouse identifiers to load. Each identifier is matched to a same-named
        child folder under `data_root`.
    metric_column : str
        Column name to read from each overall-performance CSV. Values are
        coerced to numeric; string sentinels such as `"None"` are dropped after
        training-day assignment.
    metric_label : str
        Human-readable label describing `metric_column` and its units.

    Returns
    -------
    pd.DataFrame
        Long-form dataframe with shape `(n_valid_mouse_sessions, 6)` and
        columns `mouse`, `date`, `training_day`, `metric_column`,
        `metric_label`, and `metric_value`. `training_day` is assigned after
        sorting all sessions for a mouse by date and before invalid metric rows
        are dropped, matching the cross-mouse learning-curve convention.
    """
    data_root = Path(data_root)
    mouse_frames = []

    for mouse in mice:
        summary_path = data_root / mouse / "cross_session_analysis" / f"{mouse}_overall_performance.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing overall performance summary for {mouse}: {summary_path}")

        mouse_df = pd.read_csv(summary_path, sep=",", na_filter=False)
        if "date" not in mouse_df.columns:
            raise ValueError(f"{summary_path} is missing required 'date' column.")
        if metric_column not in mouse_df.columns:
            raise ValueError(
                f"{mouse} overall performance summary is missing '{metric_column}'. "
                "Rerun session analyses to calculate this session metric."
            )

        mouse_df = mouse_df.sort_values(by="date").reset_index(drop=True)
        mouse_df["training_day"] = np.arange(1, mouse_df.shape[0] + 1, dtype=int)
        mouse_df["metric_value"] = pd.to_numeric(mouse_df[metric_column], errors="coerce")
        valid_mouse_df = mouse_df[mouse_df["metric_value"].notna()].copy()
        if valid_mouse_df.empty:
            continue
        valid_mouse_df["mouse"] = mouse
        valid_mouse_df["metric_column"] = metric_column
        valid_mouse_df["metric_label"] = metric_label
        mouse_frames.append(
            valid_mouse_df.loc[
                :,
                [
                    "mouse",
                    "date",
                    "training_day",
                    "metric_column",
                    "metric_label",
                    "metric_value",
                ],
            ]
        )

    columns = [
        "mouse",
        "date",
        "training_day",
        "metric_column",
        "metric_label",
        "metric_value",
    ]
    if not mouse_frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(mouse_frames, axis=0, ignore_index=True)


def run_cross_mouse_session_metric_curves(
    data_root: Path,
    output_path: Path,
    mice: Sequence[str],
    metric_specs: dict[str, dict[str, object]] | None = None,
    figure_id: str = "cross_mouse",
) -> None:
    """Save cross-mouse session metric CSVs and plots.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse. Each mouse folder must
        contain `cross_session_analysis/{mouse}_overall_performance.csv`.
    output_path : pathlib.Path
        Directory where long-form CSVs and PNG plots are saved.
    mice : Sequence[str]
        Mouse identifiers to include in each metric curve.
    metric_specs : dict[str, dict[str, object]] or None, default=None
        Mapping from overall-performance metric column to plot settings. Each
        setting dictionary must include `label` and may include `ylim` as a
        `(low, high)` tuple in metric units. None uses
        `CROSS_MOUSE_SESSION_METRIC_SPECS`.
    figure_id : str, default="cross_mouse"
        Figure identifier used as the filename prefix and plot title prefix.

    Returns
    -------
    None
        Writes one long-form CSV and one PNG plot per requested metric.
    """
    if metric_specs is None:
        metric_specs = CROSS_MOUSE_SESSION_METRIC_SPECS
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for metric_column, metric_spec in metric_specs.items():
        metric_label = str(metric_spec["label"])
        cross_mouse_df = load_cross_mouse_session_metric_data(
            data_root=data_root,
            mice=mice,
            metric_column=metric_column,
            metric_label=metric_label,
        )
        if cross_mouse_df.empty:
            raise ValueError(
                f"No valid {metric_column} rows found for selected mice: {list(mice)}"
            )

        csv_path = output_path / f"{figure_id}_{metric_column}_session_metric_curve_data.csv"
        cross_mouse_df.to_csv(csv_path, index=False, na_rep="None")
        performance_plots.plot_cross_mouse_session_metric_curve(
            cross_mouse_df=cross_mouse_df,
            plot_path=output_path,
            figure_id=figure_id,
            metric_column=metric_column,
            metric_label=metric_label,
            ylim=metric_spec.get("ylim"),
        )


def run_cross_mouse_learning_curve(
    data_root: Path,
    output_path: Path,
    mice: Sequence[str],
    learning_regressor: str = "prev_n_rewarded",
    figure_id: str = "cross_mouse",
) -> None:
    """Save the cross-mouse learning-curve CSV and plot.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse. Each mouse folder must
        contain `cross_session_analysis/{mouse}_overall_performance.csv`.
    output_path : pathlib.Path
        Directory where the long-form CSV and PNG plot are saved.
    mice : Sequence[str]
        Mouse identifiers to include.
    learning_regressor : str, default="prev_n_rewarded"
        Regression predictor prefix. The loader reads
        `{learning_regressor}_slope`.
    figure_id : str, default="cross_mouse"
        Figure identifier used as the filename prefix and plot title prefix.

    Returns
    -------
    None
        Writes the learning-curve CSV and PNG plot.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cross_mouse_df = load_cross_mouse_learning_curve_data(
        data_root=data_root,
        mice=mice,
        learning_regressor=learning_regressor,
    )
    if cross_mouse_df.empty:
        raise ValueError(
            f"No valid {learning_regressor}_slope rows found for selected mice: {list(mice)}"
        )

    csv_path = output_path / f"{figure_id}_{learning_regressor}_learning_curve_data.csv"
    cross_mouse_df.to_csv(csv_path, index=False, na_rep="None")
    performance_plots.plot_cross_mouse_learning_curve(
        cross_mouse_df=cross_mouse_df,
        plot_path=output_path,
        figure_id=figure_id,
        learning_regressor=learning_regressor,
    )


def load_overall_performance_summary(
    mouse: str,
    multi_session_save_path: Path,
) -> pd.DataFrame:
    """Load the mouse-level overall-performance summary CSV.

    Parameters
    ----------
    mouse : str
        Mouse identifier used in `{mouse}_overall_performance.csv`.
    multi_session_save_path : pathlib.Path
        Directory containing the mouse-level cross-session summary CSV.

    Returns
    -------
    pd.DataFrame
        Overall-performance dataframe with shape `(n_sessions, n_columns)`,
        sorted by `date`. Literal `"None"` sentinels are preserved so plotting
        helpers can decide which metrics are valid.
    """
    multisession_summary_path = multi_session_save_path / f"{mouse}_overall_performance.csv"
    overall_df = pd.read_csv(multisession_summary_path, sep=",", na_filter=False)
    overall_df.sort_values(by="date", inplace=True)
    overall_df.reset_index(drop=True, inplace=True)
    return overall_df


def prepare_session_trials_to_correct_summary(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Summarize block trials-to-correct distributions for each session.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Each `block_performance`
        table has shape `(n_blocks, n_block_columns)` and must contain
        `trials_to_correct`, where values are counts in trials from a block
        transition to correct responding. Literal `"None"` and other
        nonnumeric values are treated as missing.

    Returns
    -------
    pd.DataFrame
        Session summary table with shape `(n_valid_sessions, 6)`. Columns are
        `date`, `session_id`, `n_valid_blocks`, `trials_to_correct_q1`,
        `trials_to_correct_median`, and `trials_to_correct_q3`. Quartiles and
        medians are in trials. Sessions with no numeric values are skipped
        after emitting a `UserWarning`.
    """
    columns = [
        "date",
        "session_id",
        "n_valid_blocks",
        "trials_to_correct_q1",
        "trials_to_correct_median",
        "trials_to_correct_q3",
    ]
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        block_performance = saved_session.block_performance
        if "trials_to_correct" not in block_performance.columns:
            raise ValueError(
                f"{session.sess_id_full} block_performance is missing 'trials_to_correct'."
            )

        trials_to_correct = pd.to_numeric(
            block_performance["trials_to_correct"],
            errors="coerce",
        ).dropna()
        if trials_to_correct.empty:
            warnings.warn(
                f"Skipping {session.sess_id_full} in trials-to-correct session summary "
                "because it has no valid numeric trials_to_correct values.",
                UserWarning,
                stacklevel=2,
            )
            continue

        quartiles = trials_to_correct.quantile([0.25, 0.5, 0.75])
        rows.append(
            {
                "date": session.date,
                "session_id": session.sess_id_full,
                "n_valid_blocks": int(trials_to_correct.shape[0]),
                "trials_to_correct_q1": float(quartiles.loc[0.25]),
                "trials_to_correct_median": float(quartiles.loc[0.5]),
                "trials_to_correct_q3": float(quartiles.loc[0.75]),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _get_rewarded_side_for_block_type(block_type: str) -> str | None:
    """Return the rewarded side represented by one block type.

    Parameters
    ----------
    block_type : str
        Block label from `block_performance`, such as `"left_cued"`,
        `"right_uncued"`, or `"dark period"`.

    Returns
    -------
    str or None
        `"left"` for left rewarded blocks, `"right"` for right rewarded
        blocks, and None for non-side blocks such as dark periods.
    """
    if str(block_type).startswith("left_"):
        return "left"
    if str(block_type).startswith("right_"):
        return "right"
    return None


def _count_explore_flags(explore_values: pd.Series) -> int:
    """Count truthy explore-trial flags in one trial vector.

    Parameters
    ----------
    explore_values : pd.Series
        Explore flags with shape `(n_trials,)`. Values may be booleans,
        numeric 0/1, or CSV-loaded strings such as `"true"` and `"1"`.

    Returns
    -------
    int
        Number of truthy explore flags, in trials.
    """
    text_values = explore_values.astype(str).str.strip().str.lower()
    true_text_mask = text_values.isin(["true", "1", "1.0"])
    numeric_values = pd.to_numeric(explore_values, errors="coerce")
    positive_numeric_mask = numeric_values.fillna(0) > 0
    return int((true_text_mask | positive_numeric_mask).sum())


def add_block_explore_counts_from_trials(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return block performance with `n_explore_trials` filled when possible.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Block table with shape `(n_blocks, n_block_columns)`. When trial-level
        `explore_trial` tags are available, `n_explore_trials` is derived from
        those tags even if a saved block column already exists. If trial-level
        tags are unavailable, existing `n_explore_trials` values are preserved.
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_trial_columns)`. Required column
        is `cur_block` when `explore_trial` is available. Optional
        `explore_trial` is used to derive counts;
        missing `explore_trial` is treated as zero explore trials for backward
        compatibility with older saved analyses.

    Returns
    -------
    pd.DataFrame
        Copy of `block_performance` with an `n_explore_trials` column. Counts
        are integers in trials.
    """
    output_df = block_performance.copy()

    if "explore_trial" not in augmented_trial_df.columns:
        if "n_explore_trials" in output_df.columns:
            return output_df
        output_df["n_explore_trials"] = 0
        return output_df

    if "cur_block" not in augmented_trial_df.columns:
        raise ValueError(
            "Cannot derive n_explore_trials because augmented_trial_df is missing "
            "columns: ['cur_block']"
        )

    raw_block_ids = np.unique(augmented_trial_df["cur_block"])
    if raw_block_ids.shape[0] != output_df.shape[0]:
        raise ValueError(
            "Cannot derive n_explore_trials because trial cur_block values do not "
            "align one-to-one with block_performance rows."
        )

    explore_counts = []
    for raw_block_id in raw_block_ids:
        block_trials = augmented_trial_df[augmented_trial_df["cur_block"] == raw_block_id]
        explore_counts.append(_count_explore_flags(block_trials["explore_trial"]))

    output_df["n_explore_trials"] = explore_counts
    return output_df


def add_block_explore_run_counts_from_trials(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
    min_explore_run_length_to_count: int = 1,
) -> pd.DataFrame:
    """Return block performance with `n_explore_runs` filled when possible.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Block table with shape `(n_blocks, n_block_columns)`. When trial-level
        `explore_run_start` tags are available, `n_explore_runs` is derived
        from those tags even if a saved block column already exists. If
        trial-level tags are unavailable, existing `n_explore_runs` values are
        preserved.
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_trial_columns)`. Required column
        is `cur_block` when `explore_run_start` is available. Optional
        `explore_run_start` is used to derive counts; missing tags are treated
        as zero explore runs for backward compatibility with older saved
        analyses.
    min_explore_run_length_to_count : int, default=1
        Minimum exploratory-side run length required for a run to be counted.
        This filters summary/plot counts only; it does not change the
        trial-level run detection.

    Returns
    -------
    pd.DataFrame
        Copy of `block_performance` with an `n_explore_runs` column. Counts are
        integers in runs.
    """
    if min_explore_run_length_to_count < 1:
        raise ValueError("min_explore_run_length_to_count must be at least 1.")

    output_df = block_performance.copy()

    if "explore_run_start" not in augmented_trial_df.columns:
        if "n_explore_runs" in output_df.columns:
            return output_df
        output_df["n_explore_runs"] = 0
        return output_df

    if "cur_block" not in augmented_trial_df.columns:
        raise ValueError(
            "Cannot derive n_explore_runs because augmented_trial_df is missing "
            "columns: ['cur_block']"
        )

    raw_block_ids = np.unique(augmented_trial_df["cur_block"])
    if raw_block_ids.shape[0] != output_df.shape[0]:
        raise ValueError(
            "Cannot derive n_explore_runs because trial cur_block values do not "
            "align one-to-one with block_performance rows."
        )

    if min_explore_run_length_to_count > 1 and "explore_run_length" not in augmented_trial_df.columns:
        raise ValueError(
            "Cannot filter n_explore_runs by minimum length because "
            "augmented_trial_df is missing 'explore_run_length'."
        )

    explore_run_counts = []
    for raw_block_id in raw_block_ids:
        block_trials = augmented_trial_df[augmented_trial_df["cur_block"] == raw_block_id]
        if min_explore_run_length_to_count == 1:
            explore_run_counts.append(_count_explore_flags(block_trials["explore_run_start"]))
            continue

        start_values = block_trials["explore_run_start"]
        start_text = start_values.astype(str).str.strip().str.lower()
        start_mask = start_text.isin(["true", "1", "1.0"])
        start_numeric = pd.to_numeric(start_values, errors="coerce")
        start_mask = start_mask | (start_numeric.fillna(0) > 0)
        run_lengths = pd.to_numeric(block_trials["explore_run_length"], errors="coerce")
        explore_run_counts.append(int((start_mask & (run_lengths >= min_explore_run_length_to_count)).sum()))

    output_df["n_explore_runs"] = explore_run_counts
    return output_df


def prepare_side_trials_to_correct_block_points(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Build raw side-specific block points for cross-session quality plots.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Each `block_performance`
        table has shape `(n_blocks, n_block_columns)` and must contain
        `block_type` and `trials_to_correct`. `block_ix` is preserved when
        available and otherwise replaced by the row index.

    Returns
    -------
    pd.DataFrame
        Raw side-block table with shape `(n_side_blocks, 7)`. Columns are
        `date`, `session_id`, `rewarded_side`, `block_ix`, `block_type`,
        `trials_to_correct_numeric`, and `no_correct_choice`. Trial counts are
        in trials; `no_correct_choice` is True when `trials_to_correct` cannot
        be interpreted numerically for a left/right block.
    """
    columns = [
        "date",
        "session_id",
        "rewarded_side",
        "block_ix",
        "block_type",
        "trials_to_correct_numeric",
        "no_correct_choice",
    ]
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        block_performance = saved_session.block_performance
        required_columns = {"block_type", "trials_to_correct"}
        missing_columns = sorted(required_columns.difference(block_performance.columns))
        if missing_columns:
            raise ValueError(
                f"{session.sess_id_full} block_performance is missing columns: {missing_columns}"
            )

        if "block_ix" in block_performance.columns:
            block_ids = block_performance["block_ix"].tolist()
        else:
            block_ids = block_performance.index.tolist()

        trials_to_correct_numeric = pd.to_numeric(
            block_performance["trials_to_correct"],
            errors="coerce",
        )
        for row_position, (_row_index, row) in enumerate(block_performance.iterrows()):
            rewarded_side = _get_rewarded_side_for_block_type(row["block_type"])
            if rewarded_side is None:
                continue

            numeric_value = trials_to_correct_numeric.iloc[row_position]
            base_row = {
                "date": session.date,
                "session_id": session.sess_id_full,
                "block_ix": block_ids[row_position],
                "block_type": row["block_type"],
                "trials_to_correct_numeric": float(numeric_value)
                if not pd.isna(numeric_value)
                else np.nan,
                "no_correct_choice": bool(pd.isna(numeric_value)),
            }
            rows.append(base_row | {"rewarded_side": "overall"})
            rows.append(base_row | {"rewarded_side": rewarded_side})

    return pd.DataFrame(rows, columns=columns)


def prepare_side_trials_to_correct_summary(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Summarize completion and trials-to-correct distributions by rewarded side.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Left cued/uncued blocks are
        pooled as `"left"` and right cued/uncued blocks are pooled as
        `"right"`. Non-side blocks such as dark periods are excluded.

    Returns
    -------
    pd.DataFrame
        Side-specific session summary with one row per session and rewarded
        side present in the data. Columns include block counts,
        `completion_fraction`, and `trials_to_correct_q1`,
        `trials_to_correct_median`, and `trials_to_correct_q3`. Trial-count
        summaries are in trials and are NaN when a side has no numeric
        trials-to-correct values.
    """
    columns = [
        "date",
        "session_id",
        "rewarded_side",
        "n_blocks",
        "n_valid_blocks",
        "n_no_correct_blocks",
        "completion_fraction",
        "trials_to_correct_q1",
        "trials_to_correct_median",
        "trials_to_correct_q3",
    ]
    block_points = prepare_side_trials_to_correct_block_points(saved_sessions)
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        session_points = block_points[block_points["session_id"] == session.sess_id_full]
        for rewarded_side in ("overall", "left", "right"):
            side_points = session_points[session_points["rewarded_side"] == rewarded_side]
            if side_points.empty:
                continue

            numeric_trials = side_points["trials_to_correct_numeric"].dropna()
            if numeric_trials.empty:
                q1 = median = q3 = np.nan
            else:
                quartiles = numeric_trials.quantile([0.25, 0.5, 0.75])
                q1 = float(quartiles.loc[0.25])
                median = float(quartiles.loc[0.5])
                q3 = float(quartiles.loc[0.75])

            n_blocks = int(side_points.shape[0])
            n_valid_blocks = int(numeric_trials.shape[0])
            rows.append(
                {
                    "date": session.date,
                    "session_id": session.sess_id_full,
                    "rewarded_side": rewarded_side,
                    "n_blocks": n_blocks,
                    "n_valid_blocks": n_valid_blocks,
                    "n_no_correct_blocks": int(side_points["no_correct_choice"].sum()),
                    "completion_fraction": n_valid_blocks / n_blocks,
                    "trials_to_correct_q1": q1,
                    "trials_to_correct_median": median,
                    "trials_to_correct_q3": q3,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def ensure_block_agent_mouse_agreement_columns(
    saved_session: SavedSessionAnalysis,
) -> SavedSessionAnalysis:
    """Return a saved session with blockwise mouse-agent agreement columns.

    Parameters
    ----------
    saved_session : SavedSessionAnalysis
        Saved single-session tables. `block_performance` has shape
        `(n_blocks, n_block_columns)` and `augmented_trial_df` has shape
        `(n_trials, n_trial_columns)`.

    Returns
    -------
    SavedSessionAnalysis
        Session wrapper whose block table contains all columns listed in
        `AGENT_MOUSE_AGREEMENT_COLUMNS`.

    Raises
    ------
    ValueError
        If the block columns are missing and the saved augmented trial table
        does not contain the trial features needed to regenerate them.
    """
    missing_columns = [
        column
        for column in AGENT_MOUSE_AGREEMENT_COLUMNS.values()
        if column not in saved_session.block_performance.columns
    ]
    if not missing_columns:
        return saved_session

    try:
        block_performance = session_analysis.add_block_agent_mouse_agreement_columns(
            saved_session.block_performance,
            saved_session.augmented_trial_df,
        )
    except ValueError as exc:
        raise ValueError(
            f"{saved_session.session.sess_id_full} is missing block agent-agreement "
            "columns and they could not be regenerated from augmented trial features. "
            "Rerun single-session trial feature collection for this session."
        ) from exc

    return SavedSessionAnalysis(
        session=saved_session.session,
        block_performance=block_performance,
        augmented_trial_df=saved_session.augmented_trial_df,
    )


def prepare_agent_mouse_agreement_block_points(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Build raw block-level mouse-agent agreement points across sessions.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Missing block agreement
        columns are regenerated from each session's saved augmented trial table
        when possible.

    Returns
    -------
    pd.DataFrame
        Long-form table with shape `(n_valid_block_agent_rows, 6)`. Columns are
        `date`, `session_id`, `agent`, `block_ix`, `block_type`, and
        `agreement`. Agreement values are fractions in [0, 1].
    """
    columns = ["date", "session_id", "agent", "block_ix", "block_type", "agreement"]
    rows = []

    for raw_saved_session in saved_sessions:
        saved_session = ensure_block_agent_mouse_agreement_columns(raw_saved_session)
        session = saved_session.session
        block_performance = saved_session.block_performance
        required_columns = {"block_type", *AGENT_MOUSE_AGREEMENT_COLUMNS.values()}
        missing_columns = sorted(required_columns.difference(block_performance.columns))
        if missing_columns:
            raise ValueError(
                f"{session.sess_id_full} block_performance is missing columns: {missing_columns}"
            )

        if "block_ix" in block_performance.columns:
            block_ids = block_performance["block_ix"].tolist()
        else:
            block_ids = block_performance.index.tolist()

        numeric_agreements = {
            agent: pd.to_numeric(block_performance[column], errors="coerce")
            for agent, column in AGENT_MOUSE_AGREEMENT_COLUMNS.items()
        }
        for row_position, (_row_index, row) in enumerate(block_performance.iterrows()):
            for agent, agreement_values in numeric_agreements.items():
                agreement_value = agreement_values.iloc[row_position]
                if pd.isna(agreement_value):
                    continue
                rows.append(
                    {
                        "date": session.date,
                        "session_id": session.sess_id_full,
                        "agent": agent,
                        "block_ix": block_ids[row_position],
                        "block_type": row["block_type"],
                        "agreement": float(agreement_value),
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def prepare_agent_mouse_agreement_summary(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Summarize mouse-agent agreement distributions per session and agent.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per session and agent. Columns include
        `date`, `session_id`, `agent`, `n_blocks`, `agreement_q1`,
        `agreement_median`, and `agreement_q3`.
    """
    columns = [
        "date",
        "session_id",
        "agent",
        "n_blocks",
        "agreement_q1",
        "agreement_median",
        "agreement_q3",
    ]
    block_points = prepare_agent_mouse_agreement_block_points(saved_sessions)
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        session_points = block_points[block_points["session_id"] == session.sess_id_full]
        for agent in AGENT_MOUSE_AGREEMENT_COLUMNS.keys():
            agent_points = session_points[session_points["agent"] == agent]
            if agent_points.empty:
                continue

            agreement_values = agent_points["agreement"].dropna()
            quartiles = agreement_values.quantile([0.25, 0.5, 0.75])
            rows.append(
                {
                    "date": session.date,
                    "session_id": session.sess_id_full,
                    "agent": agent,
                    "n_blocks": int(agreement_values.shape[0]),
                    "agreement_q1": float(quartiles.loc[0.25]),
                    "agreement_median": float(quartiles.loc[0.5]),
                    "agreement_q3": float(quartiles.loc[0.75]),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def prepare_block_switch_block_points(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Build raw block-switch points for cross-session quality plots.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Each `block_performance`
        table has shape `(n_blocks, n_block_columns)` and must contain
        `block_type` and `n_switches`. `n_switches` is the raw count of
        within-block adjacent choice transitions where the action changed.

    Returns
    -------
    pd.DataFrame
        Raw switch table with shape `(n_points, 6)`. Columns are `date`,
        `session_id`, `switch_group`, `block_ix`, `block_type`, and
        `n_switches`. Each valid block contributes one `"overall"` row; left
        and right blocks also contribute one side-specific row.
    """
    columns = [
        "date",
        "session_id",
        "switch_group",
        "block_ix",
        "block_type",
        "n_switches",
    ]
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        block_performance = saved_session.block_performance
        required_columns = {"block_type", "n_switches"}
        missing_columns = sorted(required_columns.difference(block_performance.columns))
        if missing_columns:
            raise ValueError(
                f"{session.sess_id_full} block_performance is missing columns: {missing_columns}"
            )

        if "block_ix" in block_performance.columns:
            block_ids = block_performance["block_ix"].tolist()
        else:
            block_ids = block_performance.index.tolist()

        n_switches = pd.to_numeric(block_performance["n_switches"], errors="coerce")
        for row_position, (_row_index, row) in enumerate(block_performance.iterrows()):
            n_switch_value = n_switches.iloc[row_position]
            if pd.isna(n_switch_value):
                continue

            base_row = {
                "date": session.date,
                "session_id": session.sess_id_full,
                "block_ix": block_ids[row_position],
                "block_type": row["block_type"],
                "n_switches": float(n_switch_value),
            }
            rows.append(base_row | {"switch_group": "overall"})

            rewarded_side = _get_rewarded_side_for_block_type(row["block_type"])
            if rewarded_side is not None:
                rows.append(base_row | {"switch_group": rewarded_side})

    return pd.DataFrame(rows, columns=columns)


def prepare_block_switch_summary(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Summarize raw block-switch distributions by session and switch group.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Non-side blocks contribute to
        the `"overall"` distribution only; left/right blocks contribute to both
        `"overall"` and their side-specific distribution.

    Returns
    -------
    pd.DataFrame
        Switch summary table with one row per session/group present in the
        data. Columns include `date`, `session_id`, `switch_group`, `n_blocks`,
        `n_switches_q1`, `n_switches_median`, and `n_switches_q3`. Switch
        counts are raw within-block choice-switch counts.
    """
    columns = [
        "date",
        "session_id",
        "switch_group",
        "n_blocks",
        "n_switches_q1",
        "n_switches_median",
        "n_switches_q3",
    ]
    block_points = prepare_block_switch_block_points(saved_sessions)
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        session_points = block_points[block_points["session_id"] == session.sess_id_full]
        for switch_group in ("overall", "left", "right"):
            group_points = session_points[session_points["switch_group"] == switch_group]
            if group_points.empty:
                continue

            switch_values = group_points["n_switches"].dropna()
            quartiles = switch_values.quantile([0.25, 0.5, 0.75])
            rows.append(
                {
                    "date": session.date,
                    "session_id": session.sess_id_full,
                    "switch_group": switch_group,
                    "n_blocks": int(switch_values.shape[0]),
                    "n_switches_q1": float(quartiles.loc[0.25]),
                    "n_switches_median": float(quartiles.loc[0.5]),
                    "n_switches_q3": float(quartiles.loc[0.75]),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def prepare_block_explore_block_points(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Build raw block explore-trial points for cross-session quality plots.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Each `block_performance`
        table has shape `(n_blocks, n_block_columns)` and must contain
        `block_type` and `n_explore_trials`. `n_explore_trials` is the number
        of rewarded-switch explore trials in that block.

    Returns
    -------
    pd.DataFrame
        Raw explore table with shape `(n_points, 6)`. Columns are `date`,
        `session_id`, `explore_group`, `block_ix`, `block_type`, and
        `n_explore_trials`. Each valid block contributes one `"overall"` row;
        left and right blocks also contribute one side-specific row.
    """
    columns = [
        "date",
        "session_id",
        "explore_group",
        "block_ix",
        "block_type",
        "n_explore_trials",
    ]
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        block_performance = add_block_explore_counts_from_trials(
            saved_session.block_performance,
            saved_session.augmented_trial_df,
        )
        required_columns = {"block_type", "n_explore_trials"}
        missing_columns = sorted(required_columns.difference(block_performance.columns))
        if missing_columns:
            raise ValueError(
                f"{session.sess_id_full} block_performance is missing columns: {missing_columns}"
            )

        if "block_ix" in block_performance.columns:
            block_ids = block_performance["block_ix"].tolist()
        else:
            block_ids = block_performance.index.tolist()

        n_explore_trials = pd.to_numeric(block_performance["n_explore_trials"], errors="coerce")
        for row_position, (_row_index, row) in enumerate(block_performance.iterrows()):
            rewarded_side = _get_rewarded_side_for_block_type(row["block_type"])
            if rewarded_side is None:
                continue

            n_explore_value = n_explore_trials.iloc[row_position]
            if pd.isna(n_explore_value):
                continue

            base_row = {
                "date": session.date,
                "session_id": session.sess_id_full,
                "block_ix": block_ids[row_position],
                "block_type": row["block_type"],
                "n_explore_trials": float(n_explore_value),
            }
            rows.append(base_row | {"explore_group": "overall"})
            rows.append(base_row | {"explore_group": rewarded_side})

    return pd.DataFrame(rows, columns=columns)


def prepare_block_explore_summary(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Summarize raw block explore-trial distributions by session and group.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Non-side blocks contribute to
        the `"overall"` distribution only; left/right blocks contribute to both
        `"overall"` and their side-specific distribution.

    Returns
    -------
    pd.DataFrame
        Explore summary table with one row per session/group present in the
        data. Columns include `date`, `session_id`, `explore_group`,
        `n_blocks`, `n_explore_trials_q1`, `n_explore_trials_median`, and
        `n_explore_trials_q3`. Counts are raw explore trials per block.
    """
    columns = [
        "date",
        "session_id",
        "explore_group",
        "n_blocks",
        "n_explore_trials_q1",
        "n_explore_trials_median",
        "n_explore_trials_q3",
    ]
    block_points = prepare_block_explore_block_points(saved_sessions)
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        session_points = block_points[block_points["session_id"] == session.sess_id_full]
        for explore_group in ("overall", "left", "right"):
            group_points = session_points[session_points["explore_group"] == explore_group]
            if group_points.empty:
                continue

            explore_values = group_points["n_explore_trials"].dropna()
            quartiles = explore_values.quantile([0.25, 0.5, 0.75])
            rows.append(
                {
                    "date": session.date,
                    "session_id": session.sess_id_full,
                    "explore_group": explore_group,
                    "n_blocks": int(explore_values.shape[0]),
                    "n_explore_trials_q1": float(quartiles.loc[0.25]),
                    "n_explore_trials_median": float(quartiles.loc[0.5]),
                    "n_explore_trials_q3": float(quartiles.loc[0.75]),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def prepare_block_explore_run_block_points(
    saved_sessions: Sequence[SavedSessionAnalysis],
    min_explore_run_length_to_count: int = 1,
) -> pd.DataFrame:
    """Build raw block explore-run points for cross-session quality plots.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Each session supplies a
        block-performance table with shape `(n_blocks, n_block_columns)` and an
        augmented-trial table with shape `(n_trials, n_trial_columns)`.
        `n_explore_runs` is the number of valid exploratory leave-return runs
        in each block. If missing from the block table, it is derived from
        trial-level `explore_run_start` tags when available.
    min_explore_run_length_to_count : int, default=1
        Minimum exploratory-side run length required for a run to be counted.

    Returns
    -------
    pd.DataFrame
        Raw block table with one `"overall"` row for every valid block and one
        side-specific row for every left/right block. Columns include `date`,
        `session_id`, `explore_group`, `block_ix`, `block_type`, and
        `n_explore_runs`. Counts are in runs per block.
    """
    columns = [
        "date",
        "session_id",
        "explore_group",
        "block_ix",
        "block_type",
        "n_explore_runs",
    ]
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        block_performance = add_block_explore_run_counts_from_trials(
            saved_session.block_performance,
            saved_session.augmented_trial_df,
            min_explore_run_length_to_count=min_explore_run_length_to_count,
        )
        required_columns = {"block_type", "n_explore_runs"}
        missing_columns = sorted(required_columns.difference(block_performance.columns))
        if missing_columns:
            raise ValueError(
                f"{session.sess_id_full} block_performance is missing columns: {missing_columns}"
            )

        if "block_ix" in block_performance.columns:
            block_ids = block_performance["block_ix"].tolist()
        else:
            block_ids = block_performance.index.tolist()

        n_explore_runs = pd.to_numeric(block_performance["n_explore_runs"], errors="coerce")
        for row_position, (_row_index, row) in enumerate(block_performance.iterrows()):
            rewarded_side = _get_rewarded_side_for_block_type(row["block_type"])
            if rewarded_side is None:
                continue

            n_explore_value = n_explore_runs.iloc[row_position]
            if pd.isna(n_explore_value):
                continue

            base_row = {
                "date": session.date,
                "session_id": session.sess_id_full,
                "block_ix": block_ids[row_position],
                "block_type": row["block_type"],
                "n_explore_runs": float(n_explore_value),
            }
            rows.append(base_row | {"explore_group": "overall"})
            rows.append(base_row | {"explore_group": rewarded_side})

    return pd.DataFrame(rows, columns=columns)


def prepare_block_explore_run_summary(
    saved_sessions: Sequence[SavedSessionAnalysis],
    min_explore_run_length_to_count: int = 1,
) -> pd.DataFrame:
    """Summarize block explore-run distributions by session and side group.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Non-side blocks contribute to
        the `"overall"` distribution only; left/right blocks contribute to both
        `"overall"` and their side-specific distribution.
    min_explore_run_length_to_count : int, default=1
        Minimum exploratory-side run length required for a run to be counted.

    Returns
    -------
    pd.DataFrame
        Explore-run summary table with one row per session/group present in
        the data. Columns include `date`, `session_id`, `explore_group`,
        `n_blocks`, `n_explore_runs_q1`, `n_explore_runs_median`, and
        `n_explore_runs_q3`. Counts are runs per block.
    """
    columns = [
        "date",
        "session_id",
        "explore_group",
        "n_blocks",
        "n_explore_runs_q1",
        "n_explore_runs_median",
        "n_explore_runs_q3",
    ]
    block_points = prepare_block_explore_run_block_points(
        saved_sessions,
        min_explore_run_length_to_count=min_explore_run_length_to_count,
    )
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        session_points = block_points[block_points["session_id"] == session.sess_id_full]
        for explore_group in ("overall", "left", "right"):
            group_points = session_points[session_points["explore_group"] == explore_group]
            if group_points.empty:
                continue

            explore_values = group_points["n_explore_runs"].dropna()
            quartiles = explore_values.quantile([0.25, 0.5, 0.75])
            rows.append(
                {
                    "date": session.date,
                    "session_id": session.sess_id_full,
                    "explore_group": explore_group,
                    "n_blocks": int(explore_values.shape[0]),
                    "n_explore_runs_q1": float(quartiles.loc[0.25]),
                    "n_explore_runs_median": float(quartiles.loc[0.5]),
                    "n_explore_runs_q3": float(quartiles.loc[0.75]),
                }
            )

    return pd.DataFrame(rows, columns=columns)


def prepare_correct_after_first_block_points(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Build raw post-first-correct accuracy points for cross-session plots.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Each `block_performance`
        table has shape `(n_blocks, n_block_columns)` and must contain
        `block_type` and `percent_correct_after_first_correct`. Values are
        fractions from 0 to 1 for side blocks with a correct choice; missing
        sentinels mark side blocks with no correct choice. Non-side blocks
        such as dark periods are excluded.

    Returns
    -------
    pd.DataFrame
        Raw side-block table with shape `(n_points, 8)`. Columns are `date`,
        `session_id`, `correct_after_first_group`, `rewarded_side`,
        `block_ix`, `block_type`, `percent_correct_after_first_numeric`, and
        `no_correct_choice`. Each side block contributes one `"overall"` row
        and one side-specific row.
    """
    columns = [
        "date",
        "session_id",
        "correct_after_first_group",
        "rewarded_side",
        "block_ix",
        "block_type",
        "percent_correct_after_first_numeric",
        "no_correct_choice",
    ]
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        block_performance = saved_session.block_performance
        required_columns = {"block_type", "percent_correct_after_first_correct"}
        missing_columns = sorted(required_columns.difference(block_performance.columns))
        if missing_columns:
            raise ValueError(
                f"{session.sess_id_full} block_performance is missing columns: {missing_columns}"
            )

        if "block_ix" in block_performance.columns:
            block_ids = block_performance["block_ix"].tolist()
        else:
            block_ids = block_performance.index.tolist()

        percent_correct = pd.to_numeric(
            block_performance["percent_correct_after_first_correct"],
            errors="coerce",
        )
        for row_position, (_row_index, row) in enumerate(block_performance.iterrows()):
            rewarded_side = _get_rewarded_side_for_block_type(row["block_type"])
            if rewarded_side is None:
                continue

            numeric_value = percent_correct.iloc[row_position]
            base_row = {
                "date": session.date,
                "session_id": session.sess_id_full,
                "rewarded_side": rewarded_side,
                "block_ix": block_ids[row_position],
                "block_type": row["block_type"],
                "percent_correct_after_first_numeric": float(numeric_value)
                if not pd.isna(numeric_value)
                else np.nan,
                "no_correct_choice": bool(pd.isna(numeric_value)),
            }
            rows.append(base_row | {"correct_after_first_group": "overall"})
            rows.append(base_row | {"correct_after_first_group": rewarded_side})

    return pd.DataFrame(rows, columns=columns)


def prepare_correct_after_first_summary(
    saved_sessions: Sequence[SavedSessionAnalysis],
) -> pd.DataFrame:
    """Summarize post-first-correct accuracy by session and side group.

    Parameters
    ----------
    saved_sessions : Sequence[SavedSessionAnalysis]
        Saved session analyses in plotting order. Left and right task blocks
        are included in the `"overall"` distribution and their side-specific
        distribution. Dark periods and other non-side blocks are excluded.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per session/group present in the data.
        Columns include block counts, no-correct block counts, and
        `percent_correct_after_first_q1`, `percent_correct_after_first_median`,
        and `percent_correct_after_first_q3`. Percent-correct values are
        fractions from 0 to 1 and are NaN when a group has no valid blocks.
    """
    columns = [
        "date",
        "session_id",
        "correct_after_first_group",
        "n_blocks",
        "n_valid_blocks",
        "n_no_correct_blocks",
        "percent_correct_after_first_q1",
        "percent_correct_after_first_median",
        "percent_correct_after_first_q3",
    ]
    block_points = prepare_correct_after_first_block_points(saved_sessions)
    rows = []

    for saved_session in saved_sessions:
        session = saved_session.session
        session_points = block_points[block_points["session_id"] == session.sess_id_full]
        for correct_after_first_group in ("overall", "left", "right"):
            group_points = session_points[
                session_points["correct_after_first_group"] == correct_after_first_group
            ]
            if group_points.empty:
                continue

            numeric_percent = group_points["percent_correct_after_first_numeric"].dropna()
            if numeric_percent.empty:
                q1 = median = q3 = np.nan
            else:
                quartiles = numeric_percent.quantile([0.25, 0.5, 0.75])
                q1 = float(quartiles.loc[0.25])
                median = float(quartiles.loc[0.5])
                q3 = float(quartiles.loc[0.75])

            rows.append(
                {
                    "date": session.date,
                    "session_id": session.sess_id_full,
                    "correct_after_first_group": correct_after_first_group,
                    "n_blocks": int(group_points.shape[0]),
                    "n_valid_blocks": int(numeric_percent.shape[0]),
                    "n_no_correct_blocks": int(group_points["no_correct_choice"].sum()),
                    "percent_correct_after_first_q1": q1,
                    "percent_correct_after_first_median": median,
                    "percent_correct_after_first_q3": q3,
                }
            )

    return pd.DataFrame(rows, columns=columns)


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


def load_saved_multisession_augmented_trials(
    session: Session,
    predictor_columns: tuple[str, ...],
    require_inherited_strategy: bool = True,
) -> pd.DataFrame:
    """Load a saved multisession trial table for trial GLM-HMM analysis.

    Parameters
    ----------
    session : Session
        Synthetic multisession metadata. The loader reads
        `{sess_id_full}_augmented_trials.csv` from `processed_data_path`.
    predictor_columns : tuple[str, ...]
        Trial-level predictor columns required by the GLM-HMM input matrix.
        Columns are not reshaped here; validation only checks that the saved
        trial table contains them.
    require_inherited_strategy : bool, default=True
        Whether to require block-model inheritance columns. Trial MAP modeling
        requires `inherited_block_strategy`; `inherited_block_bias` is also
        required so the loaded table matches block-modeled augmented trials.

    Returns
    -------
    pd.DataFrame
        Saved trial table with shape `(n_trials, n_columns)`, loaded with
        `na_filter=False` so string missing-value sentinels are preserved.
    """
    if isinstance(predictor_columns, str):
        predictor_columns = (predictor_columns,)

    trial_path = session.processed_data_path / f"{session.sess_id_full}_augmented_trials.csv"
    if not trial_path.exists():
        raise FileNotFoundError(f"Saved multisession trial table not found: {trial_path}")

    trial_df = pd.read_csv(trial_path, sep=",", na_filter=False)
    trial_df = normalize_experimenter_reward_column(trial_df)
    required_columns = {
        "prev_action",
        EXPERIMENTER_REWARD_GIVEN_COLUMN,
        "action",
        *predictor_columns,
    }
    if require_inherited_strategy:
        required_columns.update(
            {
                "inherited_block_strategy",
                "inherited_block_bias",
            }
        )

    missing_columns = sorted(required_columns.difference(trial_df.columns))
    if missing_columns:
        missing_summary = ", ".join(missing_columns)
        raise ValueError(
            "Saved multisession trial table is missing required columns: "
            f"{missing_summary}"
        )
    return trial_df


def run_multisession_analysis(
    concatenated: ConcatenatedSessionAnalysis,
    session: Session,
    block_num_states: int = 2,
    trial_num_states: int = 2,
    prior_alpha: float = 1,
    prior_sigma: float = 1,
    block_random_seed: int | None = None,
    trial_random_seed: int | None = None,
    trial_predictor_columns: tuple[str, ...] | None = None,
    skip_block_hmm_if_existing: bool = False,
    block_predicted_state_line_width: float | None = None,
    block_state_plot_figsize: tuple[float, float] | None = None,
    block_secondary_trace: str = "none",
    plot_sliding_regression: bool = False,
    sliding_regression_window_size: int = 10,
    sliding_regression_step_size: int = 5,
    trial_input_source: str = "from_block_modeling",
    trial_state_plot_line_width: float | None = None,
    trial_state_plot_figsize: tuple[float, float] | None = None,
) -> tuple[dict | None, dict, pd.DataFrame, pd.DataFrame]:
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
    block_random_seed : int or None, default=None
        Base seed passed to block LM-HMM information-criteria and final-model
        workflows. None preserves stochastic current behavior.
    trial_random_seed : int or None, default=None
        Base seed passed to trial GLM-HMM information-criteria and final-model
        workflows. None preserves stochastic current behavior.
    trial_predictor_columns : tuple[str, ...] or None, default=None
        Trial-level predictor columns used for GLM-HMM information criteria and
        final modeling. Input matrix columns follow this order, with the bias
        column appended internally by `trial_state_space_modeling`. If None,
        the trial modeling module default predictor tuple is used.
    skip_block_hmm_if_existing : bool, default=False
        If True and the fitted model pickle, modeled block CSV, and
        inherited-strategy trial CSV are all present for `session`, skip block
        HMM fitting and load the saved block/trial CSVs instead.
    block_predicted_state_line_width : float or None, default=None
        Optional linewidth for block predicted-state summary traces. None
        preserves the existing plotting defaults.
    block_state_plot_figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches for block predicted-state
        summary traces. None preserves the existing plotting defaults.
    block_secondary_trace : str, default="none"
        Optional right-axis trace for block predicted-state plots. Supported
        values are `"none"`, `"min_value_bias"`, `"bias_rl"`, and
        `"prev_n_rewarded"`.
    plot_sliding_regression : bool, default=False
        Whether block predicted-state plots include a third subplot with
        10-block sliding regressions of trials-to-correct on `prev_n_rewarded`.
    sliding_regression_window_size : int, default=10
        Number of valid blocks in each sliding-regression window.
    sliding_regression_step_size : int, default=5
        Number of valid blocks between successive sliding-regression windows.
    trial_input_source : str, default="from_block_modeling"
        Source of the trial table passed to trial GLM-HMM workflows:
        `"from_block_modeling"` runs block modeling first, while
        `"saved_augmented_trials"` loads the saved multisession augmented trial
        CSV and skips block modeling.
    trial_state_plot_line_width : float or None, default=None
        Optional linewidth for trial predicted-state and state-summary traces.
        None preserves the existing plotting defaults.
    trial_state_plot_figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches for trial predicted-state and
        state-summary figures. None preserves the existing plotting defaults.

    Returns
    -------
    tuple[dict or None, dict, pd.DataFrame, pd.DataFrame]
        `(block_model_selection, trial_model_selection, block_performance,
        augmented_trial_df)`. `block_model_selection` is None when
        `trial_input_source="saved_augmented_trials"` because block IC/modeling
        is skipped. The trial table includes block-state inheritance generated
        by the block model or loaded from the saved augmented trial CSV.
    """
    if trial_predictor_columns is None:
        trial_predictor_columns = tssm.DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS

    valid_trial_input_sources = {"from_block_modeling", "saved_augmented_trials"}
    if trial_input_source not in valid_trial_input_sources:
        valid_summary = ", ".join(sorted(valid_trial_input_sources))
        raise ValueError(
            f"trial_input_source must be one of: {valid_summary}. "
            f"Received {trial_input_source!r}."
        )

    block_model_selection = None
    trial_model_selection = None
    if trial_input_source == "from_block_modeling":
        # block_model_selection = bssm.run_information_criteria(
        #     concatenated.block_performance,
        #     session=session,
        #     algorithm='MLE',
        #     prior_alpha=prior_alpha,
        #     prior_sigma=prior_sigma,
        #     random_seed=block_random_seed,
        # )
        missing_hmm_outputs = missing_block_hmm_output_paths(session)
        if skip_block_hmm_if_existing and not missing_hmm_outputs:
            print(
                f"Skipping block HMM for {session.sess_id_full}; "
                "saved block HMM outputs already exist."
            )
            modeled_block_df, modeled_trial_df = load_saved_block_hmm_outputs(session)
        else:
            if skip_block_hmm_if_existing:
                missing_summary = "\n".join(str(path) for path in missing_hmm_outputs)
                print(
                    f"Block HMM skip requested for {session.sess_id_full}, but required "
                    f"outputs are missing. Running block HMM.\n{missing_summary}"
                )
            save_concatenated_multisession_inputs(concatenated, session)
            modeled_block_df, modeled_trial_df = bssm.run_block_modeling(
                concatenated.block_performance,
                concatenated.augmented_trial_df,
                session=session,
                num_states=block_num_states,
                prior_alpha=prior_alpha,
                prior_sigma=prior_sigma,
                random_seed=block_random_seed,
                predicted_state_line_width=block_predicted_state_line_width,
                state_plot_figsize=block_state_plot_figsize,
                block_secondary_trace=block_secondary_trace,
                plot_sliding_regression=plot_sliding_regression,
                sliding_regression_window_size=sliding_regression_window_size,
                sliding_regression_step_size=sliding_regression_step_size,
            )
    else:
        modeled_block_df = concatenated.block_performance
        modeled_trial_df = load_saved_multisession_augmented_trials(
            session=session,
            predictor_columns=trial_predictor_columns,
            require_inherited_strategy=True,
        )
    # trial_model_selection = tssm.run_information_criteria(
    #     modeled_trial_df,
    #     session=session,
    #     algorithm='MLE',
    #     prior_alpha=prior_alpha,
    #     prior_sigma=prior_sigma,
    #     predictor_columns=trial_predictor_columns,
    #     random_seed=trial_random_seed,
    # )
    # modeled_trial_df = tssm.run_trial_modeling(
    #     modeled_trial_df,
    #     session=session,
    #     num_states=trial_num_states,
    #     prior_alpha=prior_alpha,
    #     prior_sigma=prior_sigma,
    #     predictor_columns=trial_predictor_columns,
    #     random_seed=trial_random_seed,
    #     state_plot_line_width=trial_state_plot_line_width,
    #     state_plot_figsize=trial_state_plot_figsize,
    # )
    return block_model_selection, trial_model_selection, modeled_block_df, modeled_trial_df


def run_or_collect_multisession_analysis(
    concatenated: ConcatenatedSessionAnalysis,
    session: Session,
    collection_only: bool = False,
    **modeling_kwargs,
) -> tuple | None:
    """Either save refreshed multisession CSVs or run full multisession modeling.

    Parameters
    ----------
    concatenated : ConcatenatedSessionAnalysis
        Concatenated block and trial tables. Block table shape is
        `(n_blocks, n_block_columns)` and trial table shape is
        `(n_trials, n_trial_columns)`. Existing within-session history columns
        are preserved.
    session : Session
        Synthetic multisession metadata with `processed_data_path`,
        `figure_path`, and `sess_id_full` attributes.
    collection_only : bool, default=False
        If True, overwrite `{sess_id_full}_block_performance.csv` and
        `{sess_id_full}_augmented_trials.csv` from `concatenated` and skip all
        HMM work. If False, forward to `run_multisession_analysis`.
    **modeling_kwargs
        Keyword arguments forwarded unchanged to `run_multisession_analysis`
        when `collection_only` is False.

    Returns
    -------
    tuple or None
        Returns None in collection-only mode. Otherwise returns
        `(block_model_selection, trial_model_selection, modeled_block_df,
        modeled_trial_df)` from `run_multisession_analysis`.
    """
    if collection_only:
        save_concatenated_multisession_inputs(concatenated, session)
        print(
            "WARNING: collection-only mode overwrote multisession block/trial CSVs "
            f"for {session.sess_id_full} without rerunning HMM. Existing HMM "
            "pickle files and plots may no longer match these CSVs. Disable "
            "skip_block_hmm_if_existing or rerun HMM before interpreting HMM outputs."
        )
        return None

    return run_multisession_analysis(
        concatenated=concatenated,
        session=session,
        **modeling_kwargs,
    )


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

    block_hmm_random_seed = 1001

    block_model_selection = bssm.run_information_criteria(block_performance, session=sess, algorithm='MLE',
                                                    prior_alpha=1, prior_sigma=1,
                                                    random_seed=block_hmm_random_seed)
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


def main_multisession(multisession_collection_only: bool = True):
    """Analyze selected saved sessions as one continuous multisession table.

    Parameters
    ----------
    multisession_collection_only : bool, default=True
        If True, overwrite the saved multisession block/trial CSVs from the
        current single-session CSVs and return before plotting or HMM-related
        work. If False, run the full multisession plotting and modeling flow.
    """
    mouse = 'CT024'
    session_data_root = Path(f'/home/matt/Documents/EXPERIMENTS/contextProjectData/{mouse}')
    multi_session_save_path = Path(f'/home/matt/Documents/EXPERIMENTS/contextProjectData/{mouse}/cross_session_analysis')
    task_tag = "latent_inference"
    use_all_dates_for_task_tag = True
    min_explore_run_length_to_count = 2
    dates = ['2026-04-17', '2026-04-20',
             '2026-04-21', '2026-04-22', '2026-04-23', '2026-04-24', '2026-04-27', '2026-04-28',
             '2026-05-01', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08', '2026-05-09', '2026-05-10',
             '2026-05-11', '2026-05-12', '2026-05-13', '2026-05-14', '2026-05-15', '2026-05-17',
             '2026-05-18',
             '2026-05-19', '2026-05-20',
             '2026-05-21', '2026-05-22', '2026-05-23', '2026-05-24', '2026-05-25',
             # '2026-05-26',
             '2026-05-27', '2026-05-28', '2026-05-29',]
    #
    # dates = [#'2026-04-17', '2026-04-20',
    #          # '#2026-04-21', '2026-04-23', '2026-04-24', '2026-04-27', '2026-04-28',
    #          # '2026-05-01', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08', '2026-05-08', '2026-05-09', '2026-05-10',
    #          # '2026-05-11', '2026-05-12', '2026-05-13', '2026-05-14', '2026-05-15', '2026-05-17',
    #         # '2026-05-18', '
    #     '2026-05-19', '2026-05-20',
    #     '2026-05-21', '2026-05-22', '2026-05-24', '2026-05-25',
    #     '2026-05-26', '2026-05-27', '2026-05-28', '2026-05-29',
    #     '2026-05-31', '2026-06-01',
    #     '2026-06-02',
    #     '2026-06-03', '2026-06-04', '2026-06-05',
    #     '2026-06-07', '2026-06-08', '2026-06-09',
    #     '2026-06-10', '2026-06-11'
    # ]
    dates = resolve_multisession_dates(
        mouse=mouse,
        session_data_root=session_data_root,
        task_tag=task_tag,
        explicit_dates=dates,
        use_all_dates_for_task_tag=use_all_dates_for_task_tag,
        require_saved_outputs=True,
        multi_session_save_path=multi_session_save_path,
    )
    block_hmm_random_seed = 1001
    trial_hmm_random_seed = 2001
    skip_block_hmm_if_existing = True
    block_predicted_state_line_width = 0.8
    block_state_plot_figsize = (18, 6)
    block_secondary_trace = "none"
    plot_sliding_regression = True
    sliding_regression_window_size = 10
    sliding_regression_step_size = 5
    block_hmm_state_marker_mode = "alpha"  # default, markersize, or alpha
    plot_agent_agreement_raw_blocks = False
    learning_regressor = "prev_n_rewarded"
    trials_to_correct_display_cap = 25
    trial_state_plot_line_width = 0.5
    trial_state_plot_figsize = (18, 6)
    trial_input_source = "from_block_modeling"  # either from_block_modeling or saved_augmented_trials
    trial_glm_predictor_columns = (
        "FQlearning_rel_value",
        "HMM_rel_value_logodds_decay",
        "relative_doubt_index",
        "perseveration_regressor",
        "time_to_choice"
    )

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
    if all(hasattr(saved_session.session, "processed_data_path") for saved_session in saved_sessions):
        for model_formula in BLOCK_RESIDUAL_MODEL_FORMULAS:
            collect_mouse_block_residual_model_summaries(
                saved_sessions=saved_sessions,
                output_path=multi_session_save_path,
                mouse=mouse,
                model_formula=model_formula,
            )
    concatenated = concatenate_saved_sessions(saved_sessions)
    multisession = build_multisession_session(
        mouse=mouse,
        multi_session_save_path=multi_session_save_path,
        sess_id_full=f'{mouse}_multisession',
    )
    if multisession_collection_only:
        run_or_collect_multisession_analysis(
            concatenated=concatenated,
            session=multisession,
            collection_only=True,
        )
        return

    learning_coefficients, learning_block_counts, learning_dates = prepare_learning_curve_data(
        mouse=mouse,
        multi_session_save_path=multi_session_save_path,
        learning_regressor=learning_regressor,
    )
    performance_plots.plot_learning_curve(
        learning_coefficients,
        learning_block_counts,
        figure_id=mouse,
        plot_path=multi_session_save_path,
        dates=learning_dates,
    )
    overall_performance_df = load_overall_performance_summary(
        mouse=mouse,
        multi_session_save_path=multi_session_save_path,
    )
    performance_plots.plot_multisession_oracle_behavior(
        overall_df=overall_performance_df,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )
    performance_plots.plot_multisession_ideal_observer_behavior(
        overall_df=overall_performance_df,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )

    block_hmm_state_features = bssm.collect_block_hmm_state_features_for_sessions(
        sessions=sessions,
        output_path=multi_session_save_path,
        mouse=mouse,
    )
    performance_plots.plot_block_hmm_state_feature_scatter(
        state_features_df=block_hmm_state_features,
        plot_path=multi_session_save_path,
        figure_id=mouse,
        fit_type="map",
        x_column="bias",
        y_column="prev_n_rewarded_weight",
        size_column="state_block_count",
        marker_mode=block_hmm_state_marker_mode,
    )
    agent_mouse_agreement_summary = prepare_agent_mouse_agreement_summary(saved_sessions)
    agent_mouse_agreement_block_points = prepare_agent_mouse_agreement_block_points(saved_sessions)
    agent_mouse_agreement_summary.to_csv(
        multi_session_save_path / f"{mouse}_agent_mouse_agreement_summary.csv",
        index=False,
        na_rep="None",
    )
    agent_mouse_agreement_block_points.to_csv(
        multi_session_save_path / f"{mouse}_agent_mouse_agreement_block_points.csv",
        index=False,
        na_rep="None",
    )
    performance_plots.plot_multisession_agent_mouse_agreement_quality(
        summary_df=agent_mouse_agreement_summary,
        block_points_df=agent_mouse_agreement_block_points,
        plot_path=multi_session_save_path,
        figure_id=mouse,
        show_raw_blocks=plot_agent_agreement_raw_blocks,
    )
    trials_to_correct_summary = prepare_session_trials_to_correct_summary(saved_sessions)
    performance_plots.plot_trials_to_correct_session_summary(
        summary_df=trials_to_correct_summary,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )
    side_trials_to_correct_summary = prepare_side_trials_to_correct_summary(saved_sessions)
    side_trials_to_correct_block_points = prepare_side_trials_to_correct_block_points(saved_sessions)
    side_trials_to_correct_summary.to_csv(
        multi_session_save_path / f"{mouse}_side_trials_to_correct_summary.csv",
        index=False,
        na_rep="None",
    )
    side_trials_to_correct_block_points.to_csv(
        multi_session_save_path / f"{mouse}_side_trials_to_correct_block_points.csv",
        index=False,
        na_rep="None",
    )
    performance_plots.plot_side_trials_to_correct_quality(
        summary_df=side_trials_to_correct_summary,
        block_points_df=side_trials_to_correct_block_points,
        plot_path=multi_session_save_path,
        figure_id=mouse,
        trial_display_cap=trials_to_correct_display_cap,
    )
    correct_after_first_summary = prepare_correct_after_first_summary(saved_sessions)
    correct_after_first_block_points = prepare_correct_after_first_block_points(saved_sessions)
    correct_after_first_summary.to_csv(
        multi_session_save_path / f"{mouse}_correct_after_first_summary.csv",
        index=False,
        na_rep="None",
    )
    correct_after_first_block_points.to_csv(
        multi_session_save_path / f"{mouse}_correct_after_first_block_points.csv",
        index=False,
        na_rep="None",
    )
    performance_plots.plot_correct_after_first_session_summary(
        summary_df=correct_after_first_summary,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )
    performance_plots.plot_multisession_correct_after_first_quality(
        summary_df=correct_after_first_summary,
        block_points_df=correct_after_first_block_points,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )
    block_switch_summary = prepare_block_switch_summary(saved_sessions)
    block_switch_block_points = prepare_block_switch_block_points(saved_sessions)
    block_switch_summary.to_csv(
        multi_session_save_path / f"{mouse}_block_switch_summary.csv",
        index=False,
        na_rep="None",
    )
    block_switch_block_points.to_csv(
        multi_session_save_path / f"{mouse}_block_switch_block_points.csv",
        index=False,
        na_rep="None",
    )
    performance_plots.plot_multisession_block_switches_quality(
        summary_df=block_switch_summary,
        block_points_df=block_switch_block_points,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )
    block_explore_summary = prepare_block_explore_summary(saved_sessions)
    block_explore_block_points = prepare_block_explore_block_points(saved_sessions)
    block_explore_summary.to_csv(
        multi_session_save_path / f"{mouse}_block_explore_summary.csv",
        index=False,
        na_rep="None",
    )
    block_explore_block_points.to_csv(
        multi_session_save_path / f"{mouse}_block_explore_block_points.csv",
        index=False,
        na_rep="None",
    )
    performance_plots.plot_multisession_block_explore_quality(
        summary_df=block_explore_summary,
        block_points_df=block_explore_block_points,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )
    block_explore_run_summary = prepare_block_explore_run_summary(
        saved_sessions,
        min_explore_run_length_to_count=min_explore_run_length_to_count,
    )
    block_explore_run_block_points = prepare_block_explore_run_block_points(
        saved_sessions,
        min_explore_run_length_to_count=min_explore_run_length_to_count,
    )
    block_explore_run_summary.to_csv(
        multi_session_save_path / f"{mouse}_block_explore_run_summary.csv",
        index=False,
        na_rep="None",
    )
    block_explore_run_block_points.to_csv(
        multi_session_save_path / f"{mouse}_block_explore_run_block_points.csv",
        index=False,
        na_rep="None",
    )
    performance_plots.plot_multisession_block_explore_run_quality(
        summary_df=block_explore_run_summary,
        block_points_df=block_explore_run_block_points,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )
    for family_name, family_spec in SESSION_SUMMARY_METRIC_FAMILIES.items():
        performance_plots.plot_session_summary_metric_family(
            overall_df=overall_performance_df,
            plot_path=multi_session_save_path,
            figure_id=mouse,
            family_name=family_name,
            metric_specs=family_spec["metrics"],
            y_label=family_spec["ylabel"],
            ylim=family_spec.get("ylim"),
        )
    multisession_result = run_or_collect_multisession_analysis(
        concatenated=concatenated,
        session=multisession,
        collection_only=multisession_collection_only,
        block_num_states=3,
        trial_num_states=3,
        prior_alpha=1,
        prior_sigma=1,
        block_random_seed=block_hmm_random_seed,
        trial_random_seed=trial_hmm_random_seed,
        trial_predictor_columns=trial_glm_predictor_columns,
        skip_block_hmm_if_existing=skip_block_hmm_if_existing,
        block_predicted_state_line_width=block_predicted_state_line_width,
        block_state_plot_figsize=block_state_plot_figsize,
        block_secondary_trace=block_secondary_trace,
        plot_sliding_regression=plot_sliding_regression,
        sliding_regression_window_size=sliding_regression_window_size,
        sliding_regression_step_size=sliding_regression_step_size,
        trial_input_source=trial_input_source,
        trial_state_plot_line_width=trial_state_plot_line_width,
        trial_state_plot_figsize=trial_state_plot_figsize,
    )
    if multisession_result is None:
        return
    _block_selection, _trial_selection, modeled_block_df, _modeled_trial_df = multisession_result
    performance_plots.plot_multisession_summary_grid(
        block_performance=modeled_block_df,
        side_trials_to_correct_summary=side_trials_to_correct_summary,
        side_trials_to_correct_block_points=side_trials_to_correct_block_points,
        correct_after_first_summary=correct_after_first_summary,
        correct_after_first_block_points=correct_after_first_block_points,
        block_switch_summary=block_switch_summary,
        block_switch_block_points=block_switch_block_points,
        overall_df=overall_performance_df,
        plot_path=multi_session_save_path,
        figure_id=mouse,
        block_model_session_id=multisession.sess_id_full,
        processed_data_path=multi_session_save_path,
        sliding_regression_window_size=sliding_regression_window_size,
        sliding_regression_step_size=sliding_regression_step_size,
        trial_display_cap=trials_to_correct_display_cap,
    )


def run_single_session_workflow(
    sess: Session,
    config: SingleSessionAnalysisConfig,
) -> None:
    """Run the full single-session behavior and HMM analysis workflow.

    Parameters
    ----------
    sess : Session
        Fully resolved single-session metadata. Required fields include
        `raw_behavior_folder`, `processed_data_path`, `figure_path`,
        `sess_id_full`, `sess_id_abbreviated`, `date`, and `session_info`.
        `session_info` may be None when `session_info_fname` points to a valid
        pickle file.
    config : SingleSessionAnalysisConfig
        Analysis parameters controlling preprocessing, saved-analysis loading,
        plotting, and HMM modeling.

    Returns
    -------
    None
        Saves plots, CSVs, and model outputs into the session's configured
        processed/figure directories.
    """
    if sess.session_info is None:
        with open(sess.session_info_fname, "rb") as file:
            sess.session_info = pkl.load(file)

    trial_df, event_df, water = load_or_preprocess_session(
        raw_behavior_folder=sess.raw_behavior_folder,
        processed_data_path=sess.processed_data_path,
        sess_id_full=sess.sess_id_full,
        preprocess_raw_session=config.preprocess_raw_session,
        min_time=config.min_time,
        max_time=config.max_time,
    )
    if water is not None:
        print(f"Preprocessed session log for {sess.sess_id_full}. Water delivered: {water}")

    plot_session(
        event_df,
        sess.session_info,
        figure_path=sess.figure_path,
        sess_id_full=sess.sess_id_full,
    )

    augmented_trial_df, block_performance, multisession_df = load_or_run_session_analysis(
        trial_df=trial_df,
        session=sess,
        run_session_analysis=config.run_session_analysis,
        ideal_observer_n_replays=config.ideal_observer_n_replays,
        ideal_observer_seed=config.ideal_observer_seed,
    )
    multisession_df = ensure_session_side_bias_metrics(
        multisession_df=multisession_df,
        augmented_trial_df=augmented_trial_df,
        session=sess,
    )

    performance_plots.plot_session_correct(block_performance, sess.figure_path, sess.sess_id_full)
    performance_plots.plot_session_correct_after_first_correct(block_performance, sess.figure_path, sess.sess_id_full)
    performance_plots.plot_session_history_ideal_mouse_agreement(block_performance, sess.figure_path, sess.sess_id_full)
    performance_plots.plot_session_trials_to_correct(block_performance, sess.figure_path, sess.sess_id_full)
    performance_plots.plot_session_nswitches(block_performance, sess.figure_path, sess.sess_id_full)
    performance_plots.plot_block_bias_quadrants(
        block_performance,
        sess.figure_path,
        sess.sess_id_full,
        title=sess.sess_id_abbreviated,
    )
    performance_plots.plot_session_block_trials_to_correct_summary(
        block_performance,
        sess.figure_path,
        sess.sess_id_full,
    )
    performance_plots.plot_session_zero_trials_to_correct_fraction(
        block_performance,
        sess.figure_path,
        sess.sess_id_full,
    )
    session_summary_for_plot = multisession_df
    if "date" in multisession_df.columns:
        matching_session_summary = multisession_df[
            multisession_df["date"].astype(str).eq(str(sess.date))
        ]
        if not matching_session_summary.empty:
            session_summary_for_plot = matching_session_summary
    performance_plots.plot_session_side_bias_ratios(
        session_summary=session_summary_for_plot,
        plot_path=sess.figure_path,
        sess_id_full=sess.sess_id_full,
    )
    performance_plots.plot_session_signed_side_bias(
        session_summary=session_summary_for_plot,
        plot_path=sess.figure_path,
        sess_id_full=sess.sess_id_full,
    )
    save_and_plot_switch_persistence_for_session(
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
        session=sess,
    )
    save_and_plot_post_first_correct_accuracy_for_session(
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
        session=sess,
    )

    slope = multisession_df.loc[
        multisession_df["date"] == sess.date,
        f"{config.scatter_regressor}_slope",
    ].values[0]
    intercept = multisession_df.loc[
        multisession_df["date"] == sess.date,
        f"{config.scatter_regressor}_intercept",
    ].values[0]
    performance_plots.scatter_trials_to_correct(
        block_performance,
        slope=slope,
        intercept=intercept,
        plot_path=sess.figure_path,
        figure_id=sess.sess_id_full,
        title=sess.sess_id_abbreviated,
        regressor_column=config.scatter_regressor,
    )

    augmented_trial_df, _task_params = gtf.collect_and_save_trial_features(
        augmented_trial_df,
        processed_data_path=sess.processed_data_path,
        sess_id_full=sess.sess_id_full,
        max_explore_run_length=config.max_explore_run_length,
    )
    block_performance = session_analysis.add_block_agent_mouse_agreement_columns(
        block_performance,
        augmented_trial_df,
    )
    block_performance = add_block_explore_counts_from_trials(
        block_performance,
        augmented_trial_df,
    )
    block_performance = add_block_explore_run_counts_from_trials(
        block_performance,
        augmented_trial_df,
        min_explore_run_length_to_count=config.min_explore_run_length_to_count,
    )
    block_performance = session_analysis.add_session_block_collection_variables(block_performance)
    block_performance = _add_single_session_metadata_columns(block_performance, sess)
    block_performance.to_csv(
        sess.processed_data_path / f"{sess.sess_id_full}_block_performance.csv",
        index=False,
        na_rep="None",
    )
    performance_plots.plot_session_explore_trials(block_performance, sess.figure_path, sess.sess_id_full)
    performance_plots.plot_session_explore_runs(block_performance, sess.figure_path, sess.sess_id_full)
    performance_plots.plot_session_agent_mouse_agreement(block_performance, sess.figure_path, sess.sess_id_full)
    plot_mouse_history_ideal_observer_for_session(
        augmented_trial_df=augmented_trial_df,
        session=sess,
    )

    missing_hmm_outputs = missing_block_hmm_output_paths(sess)
    if config.skip_block_hmm_if_existing and not missing_hmm_outputs:
        print(
            f"Skipping block HMM for {sess.sess_id_full}; "
            "saved block HMM outputs already exist."
        )
        block_performance, augmented_trial_df = load_saved_block_hmm_outputs(sess)
    else:
        if config.skip_block_hmm_if_existing:
            missing_summary = "\n".join(str(path) for path in missing_hmm_outputs)
            print(
                f"Block HMM skip requested for {sess.sess_id_full}, but required "
                f"outputs are missing. Running block HMM.\n{missing_summary}"
            )
        block_performance, augmented_trial_df = bssm.run_block_modeling(
            block_performance,
            augmented_trial_df,
            session=sess,
            num_states=config.block_modeling_states,
            prior_alpha=1,
            prior_sigma=1,
            random_seed=config.block_hmm_random_seed,
            block_secondary_trace=config.block_secondary_trace,
            plot_sliding_regression=config.plot_sliding_regression,
            sliding_regression_window_size=config.sliding_regression_window_size,
            sliding_regression_step_size=config.sliding_regression_step_size,
        )

    performance_plots.plot_single_session_summary_grid(
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
        event_df=event_df,
        session_info=sess.session_info,
        plot_path=sess.figure_path,
        figure_id=sess.sess_id_full,
        title=sess.sess_id_abbreviated,
        scatter_slope=slope,
        scatter_intercept=intercept,
        scatter_regressor=config.scatter_regressor,
        processed_data_path=sess.processed_data_path,
        sliding_regression_window_size=config.sliding_regression_window_size,
        sliding_regression_step_size=config.sliding_regression_step_size,
    )


def run_single_session_batch(
    mouse: str,
    dates: Sequence[str],
    task_tag: str,
    session_data_root: Path,
    multi_session_save_path: Path,
    config: SingleSessionAnalysisConfig,
    runner=run_single_session_workflow,
) -> list[Session]:
    """Run the single-session workflow for multiple dates of one mouse.

    Parameters
    ----------
    mouse : str
        Mouse identifier used in raw session folder names.
    dates : Sequence[str]
        Session dates formatted as `YYYY-MM-DD`, processed in the given order.
    task_tag : str
        Exact task tag used in `{mouse}_YYYYMMDD_{task_tag}` folders.
    session_data_root : pathlib.Path
        Mouse-level data directory containing task-tagged session folders.
    multi_session_save_path : pathlib.Path
        Cross-session output directory assigned to each resolved session
        metadata. This function does not concatenate sessions.
    config : SingleSessionAnalysisConfig
        Analysis parameters passed unchanged to `runner`.
    runner : callable, default=run_single_session_workflow
        Dependency-injected single-session runner. It must accept
        `(session, config)` and return None.

    Returns
    -------
    list[Session]
        Resolved sessions in processing order.
    """
    resolved_sessions = []
    for date in dates:
        sess = find_raw_session_by_date(
            mouse=mouse,
            date=date,
            session_data_root=session_data_root,
            task_tag=task_tag,
            multi_session_save_path=multi_session_save_path,
        )
        resolved_sessions.append(sess)
        runner(sess, config)
    return resolved_sessions


def main_mouse():
    """Analyze a single behavior session from start to finish."""
    mouse = "CT019"
    date = "2026-06-04"
    behavior_timestamp = 121125  # Set to "HHMMSS" to choose one session on ambiguous dates.
    task_tag = "latent_inference"
    session_data_root = Path(f"/home/matt/Documents/EXPERIMENTS/contextProjectData/{mouse}")
    multi_session_save_path = session_data_root / "cross_session_analysis"
    config = SingleSessionAnalysisConfig(
        preprocess_raw_session=True,
        run_session_analysis=True,
        ideal_observer_n_replays=100,
        ideal_observer_seed=12345,
        block_modeling_states=2,
        trial_modeling_states=3,
        scatter_regressor="prev_n_rewarded",
        block_secondary_trace="prev_n_rewarded",
        skip_block_hmm_if_existing=True,
        max_explore_run_length=5,
        min_explore_run_length_to_count=2,
    )

    sess = find_raw_session_by_date(
        mouse=mouse,
        date=date,
        session_data_root=session_data_root,
        task_tag=task_tag,
        multi_session_save_path=multi_session_save_path,
        behavior_timestamp=behavior_timestamp,
    )
    run_single_session_workflow(sess, config)


def main_mouse_batch():
    """Run ordinary single-session analysis for several dates of one mouse."""
    mouse = "CT024"
    use_all_dates_for_task_tag = True
    dates = [#'2026-04-17', '2026-04-20',
        #'2026-04-21', '2026-04-22', '2026-04-23', '2026-04-24', '2026-04-27', '2026-04-28',
       # '2026-05-01', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08', '2026-05-09', '2026-05-10',
       # '2026-05-11', '2026-05-12', '2026-05-13', '2026-05-14', '2026-05-15', '2026-05-17',
        #'2026-05-18',
        # '2026-05-19', '2026-05-20',
        # '2026-05-21', '2026-05-22', '2026-05-23', '2026-05-24', '2026-05-25',
        # '2026-05-26',
        # '2026-05-27', '2026-05-28', '2026-05-29',
        # '2026-05-31',
        # '2026-06-01', '2026-06-02', '2026-06-03', '2026-06-04',
        # '2026-06-05',
        # '2026-06-07', '2026-06-08', '2026-06-09',
        '2026-06-10', '2026-06-11'
    ]
    task_tag = "latent_inference"
    session_data_root = Path(f"/home/matt/Documents/EXPERIMENTS/contextProjectData/{mouse}")
    multi_session_save_path = session_data_root / "cross_session_analysis"
    if use_all_dates_for_task_tag:
        dates = find_raw_session_dates_for_task_tag(
            mouse=mouse,
            session_data_root=session_data_root,
            task_tag=task_tag,
        )

    config = SingleSessionAnalysisConfig(
        preprocess_raw_session=False,
        run_session_analysis=True,
        ideal_observer_n_replays=100,
        ideal_observer_seed=12345,
        block_modeling_states=2,
        trial_modeling_states=3,
        scatter_regressor="prev_n_rewarded",
        block_secondary_trace="prev_n_rewarded",
        skip_block_hmm_if_existing=True,
        max_explore_run_length=5,
        min_explore_run_length_to_count=2,
    )

    run_single_session_batch(
        mouse=mouse,
        dates=dates,
        task_tag=task_tag,
        session_data_root=session_data_root,
        multi_session_save_path=multi_session_save_path,
        config=config,
    )


def main_cross_mouse_metrics(
    data_root: Path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData"),
    use_discovered_mice: bool = False,
    mice: Sequence[str] | None = None,
    learning_regressor: str = "prev_n_rewarded",
) -> None:
    """Plot all configured cross-mouse metrics from saved summaries.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse and the
        `cross_mouse_analysis` output directory.
    use_discovered_mice : bool, default=False
        If True, discover mouse folders containing
        `{mouse}_overall_performance.csv` files. If False, use `mice`.
    mice : Sequence[str] or None, default=None
        Explicit mouse identifiers to include when `use_discovered_mice` is
        False. None uses the current hardcoded cross-mouse cohort.
    learning_regressor : str, default="prev_n_rewarded"
        Regression predictor prefix for the learning-curve plot.

    Returns
    -------
    None
        Writes all configured cross-mouse CSVs and PNG plots.
    """
    output_path = Path(data_root) / "cross_mouse_analysis"
    if mice is None:
        mice = [
            "CT016", "CT017", "CT019", "CT020", "CT021", "CT022", "CT023", "CT024", "CT025",
        ]

    selected_mice = discover_mice_with_overall_performance(data_root) if use_discovered_mice else list(mice)
    if not selected_mice:
        raise ValueError("No mice selected for cross-mouse metric plotting.")

    run_cross_mouse_learning_curve(
        data_root=data_root,
        output_path=output_path,
        mice=selected_mice,
        learning_regressor=learning_regressor,
        figure_id="cross_mouse",
    )
    run_cross_mouse_session_metric_curves(
        data_root=data_root,
        output_path=output_path,
        mice=selected_mice,
        metric_specs=CROSS_MOUSE_SESSION_METRIC_SPECS,
        figure_id="cross_mouse",
    )
    collect_cross_mouse_multisession_block_performance(
        data_root=data_root,
        output_path=output_path,
        mice=selected_mice,
    )
    collect_cross_mouse_block_residual_model_summaries(
        data_root=data_root,
        output_path=output_path,
        mice=selected_mice,
        model_formulas=BLOCK_RESIDUAL_MODEL_FORMULAS,
    )


def main_cross_mouse_multisession_block_performance(
    data_root: Path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData"),
    mice: Sequence[str] | None = None,
) -> Path:
    """Collect saved multisession block-performance CSVs across mice.

    Parameters
    ----------
    data_root : pathlib.Path
        Directory containing one folder per mouse and the
        `cross_mouse_analysis` output directory.
    mice : Sequence[str] or None, default=None
        Explicit mouse identifiers to include. None uses the current hardcoded
        cross-mouse cohort.

    Returns
    -------
    pathlib.Path
        Path to the saved combined block-performance CSV.
    """
    if mice is None:
        mice = [
            "CT016", "CT017", "CT019", "CT020", "CT021", "CT022", "CT023", "CT024", "CT025",
        ]
    output_path = Path(data_root) / "cross_mouse_analysis"
    return collect_cross_mouse_multisession_block_performance(
        data_root=data_root,
        output_path=output_path,
        mice=list(mice),
    )


# def presentation_plots(block_df: pd.DataFrame, trial_df: pd.DataFrame):
#     ix_valid = bssm.make_valid_block_history_mask(block_df)
#     df = block_df[ix_valid]
#
#     consecutive_rewards = df['prev_consecutive_rewards'].to_numpy().reshape(-1, 1).astype(int)
#     prev_rewards = df['prev_n_rewarded'].to_numpy().reshape(-1, 1).astype(int)
#     # prev_correct = df['prev_n_correct'].to_numpy().reshape(-1, 1).astype(int)
#     trials_to_correct = df['trials_to_correct'].to_numpy().reshape(-1, 1).astype(int)
#     bias_flag = df['bias_full_flag'].to_numpy().reshape(-1, 1) == 'True'
#     predictors = np.concatenate([prev_rewards, bias_flag], axis=1)
#     pred_labels = ['previous rewards', 'block bias flag']
#
#     utilplot.plot_postprob_obs_for_presentation(block_model_dict['map']['posterior_probs'], trials_to_correct, predictors, map_hmm, colors, cmap, predictor_labels = pred_labels)


if __name__ == '__main__':
    # main_mouse()
    # main_mouse_batch()
    main_multisession(multisession_collection_only=True)
    # main_simulation()
    # main_cross_mouse_metrics()
