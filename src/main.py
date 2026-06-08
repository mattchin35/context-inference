import json
import numpy as np
import re
import warnings
from behavior_analysis import performance_plots
from behavior_analysis import raster_plots, session_analysis, simulate_priors
from mouse_behavior_preprocessing import process_behavior_log
from behavior_analysis import block_state_space_modeling as bssm
from behavior_analysis import trial_state_space_modeling as tssm
from behavior_analysis import gather_trial_features as gtf
from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    normalize_experimenter_reward_column,
)
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
        return session_analysis.run_analysis(
            trial_df,
            session=session,
            ideal_observer_n_replays=ideal_observer_n_replays,
            ideal_observer_seed=ideal_observer_seed,
        )

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

    return ConcatenatedSessionAnalysis(
        block_performance=pd.concat(concatenated_blocks, axis=0, ignore_index=True),
        augmented_trial_df=pd.concat(concatenated_trials, axis=0, ignore_index=True),
        block_session_lengths=np.asarray(block_session_lengths, dtype=int),
        trial_session_lengths=np.asarray(trial_session_lengths, dtype=int),
    )


def prepare_learning_curve_data(
    mouse: str,
    multi_session_save_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load numeric mouse learning-curve arrays from the summary CSV.

    Parameters
    ----------
    mouse : str
        Mouse identifier used in `{mouse}_overall_performance.csv`.
    multi_session_save_path : Path
        Directory containing the mouse-level cross-session summary CSV.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - regression coefficients, shape `(n_valid_sessions,)`, numeric slope
          values for `DEFAULT_LEARNING_REGRESSOR`
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
    learning_regressor = getattr(
        performance_plots,
        "DEFAULT_LEARNING_REGRESSOR",
        "prev_consecutive_rewards",
    )
    learning_column = f"{learning_regressor}_slope"
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
    block_predicted_state_line_width: float | None = None,
    block_state_plot_figsize: tuple[float, float] | None = None,
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
    block_predicted_state_line_width : float or None, default=None
        Optional linewidth for block predicted-state summary traces. None
        preserves the existing plotting defaults.
    block_state_plot_figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches for block predicted-state
        summary traces. None preserves the existing plotting defaults.
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


def main_multisession():
    """Analyze selected saved sessions as one continuous multisession table."""
    mouse = 'CT017'
    session_data_root = Path(f'/home/matt/Documents/EXPERIMENTS/contextProjectData/{mouse}')
    multi_session_save_path = Path(f'/home/matt/Documents/EXPERIMENTS/contextProjectData/{mouse}/cross_session_analysis')
    dates = ['2026-04-16', '2026-04-17', '2026-04-20',
             '2026-04-21', '2026-04-23', '2026-04-24', '2026-04-27', '2026-04-28', '2026-04-29', '2026-04-30',
             '2026-05-01', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08', '2026-05-08', '2026-05-09', '2026-05-10',
             '2026-05-11',
             '2026-05-12', '2026-05-13', '2026-05-14', '2026-05-15', '2026-05-17', '2026-05-18',
             '2026-05-19', '2026-05-20', '2026-05-21', '2026-05-22', '2026-05-23', '2026-05-24',
             '2026-05-25', '2026-05-26', '2026-05-27', '2026-05-28', '2026-05-29', ]
    block_hmm_random_seed = 1001
    trial_hmm_random_seed = 2001
    block_predicted_state_line_width = 0.8
    block_state_plot_figsize = (18, 6)
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

    learning_coefficients, learning_block_counts, learning_dates = prepare_learning_curve_data(
        mouse=mouse,
        multi_session_save_path=multi_session_save_path,
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
    trials_to_correct_summary = prepare_session_trials_to_correct_summary(saved_sessions)
    performance_plots.plot_trials_to_correct_session_summary(
        summary_df=trials_to_correct_summary,
        plot_path=multi_session_save_path,
        figure_id=mouse,
    )
    concatenated = concatenate_saved_sessions(saved_sessions)
    multisession = build_multisession_session(
        mouse=mouse,
        multi_session_save_path=multi_session_save_path,
        sess_id_full=f'{mouse}_multisession',
    )
    run_multisession_analysis(
        concatenated=concatenated,
        session=multisession,
        block_num_states=4,
        trial_num_states=3,
        prior_alpha=1,
        prior_sigma=1,
        block_random_seed=block_hmm_random_seed,
        trial_random_seed=trial_hmm_random_seed,
        trial_predictor_columns=trial_glm_predictor_columns,
        block_predicted_state_line_width=block_predicted_state_line_width,
        block_state_plot_figsize=block_state_plot_figsize,
        trial_input_source=trial_input_source,
        trial_state_plot_line_width=trial_state_plot_line_width,
        trial_state_plot_figsize=trial_state_plot_figsize,
    )


def main_mouse():
    """Analyze a single behavior session from start to finish."""

    ### USER FLAGS - CHOOSE THESE FOR EACH RUN ###
    preprocess_raw_session = False
    run_session_analysis = True
    ideal_observer_n_replays = 100
    ideal_observer_seed = 12345

    ### HARDCODED DATA PATHS - CHOOSE THESE FOR EACH RUN ###
    mouse = 'CT014'
    date = '2025-12-16'
    behavior_timestamp = '153200'
    date_no_dash = date.replace('-', '')

    multi_session_save_path = Path(f'/home/matt/Documents/EXPERIMENTS/contextProjectData/{mouse}/cross_session_analysis')
    session_data_home = Path(f'/home/matt/Documents/EXPERIMENTS/contextProjectData/{mouse}/{mouse}_{date_no_dash}_latentInference')
    sess_id_full = f'{mouse}_{date}_{behavior_timestamp}'
    sess_id_abbreviated = mouse + '_' + date

    ### everything below this should be edited so it doesn't have to be commented in or out or have hardcodes changed ###
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)

    if match:
        mouse, date, timestamp = match.groups()
        print(f"Found session for mouse id: {mouse}, date: {date}, behavior timestamp: {timestamp}")
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
        ideal_observer_n_replays=ideal_observer_n_replays,
        ideal_observer_seed=ideal_observer_seed,
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

    block_hmm_random_seed = 1001
    trial_hmm_random_seed = 2001
    trial_glm_predictor_columns = (
        "FQlearning_rel_value",
        "HMM_rel_value_logodds_decay",
        "relative_doubt_index",
        "perseveration_regressor",
        # "relative_cf_value",
        # "time_to_choice"
    )

    # prefer use of the information criteria for model selection, but here is how you'd use CV
    # cv_model_selection = bssm.run_cross_validation(block_performance, session=sess, algorithm='MLE',
    #                                                prior_alpha=1, prior_sigma=1, n_runs=5, n_folds=2,
    #                                                random_seed=block_hmm_random_seed)

    # IC
    # block_model_selection = bssm.run_information_criteria(block_performance, session=sess, algorithm='MLE',
    #                                                 prior_alpha=1, prior_sigma=1, max_states=5,
    #                                                 random_seed=block_hmm_random_seed)
    #
    # block_performance, augmented_trial_df = bssm.run_block_modeling(block_performance, augmented_trial_df, session=sess,
    #                                                                 num_states=2,
    #                                                                 prior_alpha=1, prior_sigma=1,
    #                                                                 random_seed=block_hmm_random_seed)

    # load a saved block model instead of running it
    # block_model_dict_path = processed_data_path / (sess_id_full + '_block_statedict.pkl')
    # with open(block_model_dict_path, 'rb') as file:
    #     block_model_dict = pkl.load(file)

    ### trial state space modeling ###
    # trial_model_selection = tssm.run_information_criteria(
    #     augmented_trial_df,
    #     session=sess,
    #     algorithm='MLE',
    #     prior_alpha=1,
    #     prior_sigma=1,
    #     predictor_columns=trial_glm_predictor_columns,
    #     random_seed=trial_hmm_random_seed,
    # )
    # augmented_trial_df = tssm.run_trial_modeling(
    #     augmented_trial_df,
    #     session=sess,
    #     num_states=2,
    #     prior_alpha=1,
    #     prior_sigma=1,
    #     predictor_columns=trial_glm_predictor_columns,
    #     random_seed=trial_hmm_random_seed,
    # )


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
    # main_multisession()
    # main_simulation()
