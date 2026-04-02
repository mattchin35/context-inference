"""
Prepare HPC trial-aligned spike and lick bins using Pynapple time-series objects.
"""

import pickle as pkl
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pynapple as nap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score, train_test_split

import src.external_tools.readSGLX as readSGLX
import src.external_tools.get_brain_channels as get_brain_channels

@dataclass
class Session:
    """
    Session metadata and file locations.

    Attributes
    ----------
    multi_session_save_path : Path
        Directory path for cross-session summary files. Units: filesystem path.
    session_data_home : Path
        Root directory for one experimental session. Units: filesystem path.
    sess_id_full : str
        Session identifier formatted as ``mouse_YYYY-MM-DD_hhmmss``.
    sess_id_abbreviated : str
        Session identifier formatted as ``mouse_YYYY-MM-DD``.
    raw_behavior_folder : Path
        Directory containing raw Raspberry Pi behavior files. Units: filesystem path.
    processed_data_path : Path
        Directory containing processed CSV outputs. Units: filesystem path.
    figure_path : Path
        Directory for diagnostic figures. Units: filesystem path.
    mouse : str
        Mouse identifier. Units: not applicable.
    date : str
        Session date in ``YYYY-MM-DD`` format. Units: not applicable.
    timestamp : str
        Session time in ``hhmmss`` format. Units: not applicable.
    session_info_path : Path | None
        Path to the pickled session metadata file. Units: filesystem path.
    session_info : Any
        Session metadata loaded from pickle. Type depends on upstream preprocessing.
    pfc_spike_path : Path | None
        Sorter output directory for the PFC probe. Units: filesystem path.
    hpc_spike_path : Path | None
        Sorter output directory for the HPC probe. Units: filesystem path.
    """

    multi_session_save_path: Path = Path.home()
    session_data_home: Path = Path.home()
    sess_id_full: str = "mouseid_YYYY-MM-DD_hhmmss"
    sess_id_abbreviated: str = "mouseid_YYYY-MM-DD"
    raw_behavior_folder: Path = Path.home()
    processed_data_path: Path = Path.home()
    figure_path: Path = Path.home()
    mouse: str = "test-mouse"
    date: str = "1970-01-01"
    timestamp: str = "000000"
    session_info_path: Path | None = None
    session_info: Any = None
    pfc_spike_path: Path | None = None
    hpc_spike_path: Path | None = None


def parse_session_id(sess_id_full: str) -> tuple[str, str, str]:
    """
    Parse a full session identifier.

    Parameters
    ----------
    sess_id_full : str
        Session identifier with format ``mouse_YYYY-MM-DD_hhmmss``. Units: not applicable.

    Returns
    -------
    tuple[str, str, str]
        ``(mouse, date, timestamp)`` strings with no unit conversion.
    """

    match = re.search(r"(\w+)_([\d\-]+)_(\d+)", sess_id_full)
    if match is None:
        raise ValueError(f"Session id {sess_id_full!r} does not match 'mouse_YYYY-MM-DD_hhmmss'.")

    return match.groups()


def load_session_tables(session: Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed behavior tables for one session.

    Parameters
    ----------
    session : Session
        Session metadata with processed-data paths. Units: filesystem paths.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(event_df, trial_df)`` where rows are events/trials and time columns are in seconds.
    """

    event_df = pd.read_csv(session.processed_data_path / f"{session.sess_id_full}_events.csv", sep=",")
    trial_df = pd.read_csv(session.processed_data_path / f"{session.sess_id_full}_trials.csv", sep=",")
    return event_df, trial_df


def load_session_info(session_info_path: Path) -> Any:
    """
    Load serialized session metadata.

    Parameters
    ----------
    session_info_path : Path
        Path to a pickle file containing session metadata. Units: filesystem path.

    Returns
    -------
    Any
        Deserialized metadata object from upstream preprocessing.
    """

    with session_info_path.open("rb") as file_handle:
        return pkl.load(file_handle)


def load_sorter_spikes(sorter_output_path: Path, ap_bin_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load Kilosort spike arrays and convert times from samples to seconds.

    Parameters
    ----------
    sorter_output_path : Path
        Path to a sorter output directory containing ``spike_times.npy``,
        ``spike_clusters.npy``, and ``cluster_info.tsv``. Units: filesystem path.
    ap_bin_path : Path
        Path to the corresponding ``*.ap.bin`` file used to recover sample rate.
        Units: filesystem path.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        ``(spike_times_seconds, spike_clusters, cluster_info)`` where:
        ``spike_times_seconds`` has shape ``(n_spikes,)`` and units seconds,
        ``spike_clusters`` has shape ``(n_spikes,)`` and contains integer cluster ids,
        ``cluster_info`` has one row per cluster with sorter metadata.
    """

    imec_meta = readSGLX.readMeta(ap_bin_path)
    imec_sample_rate = float(readSGLX.SampRate(imec_meta))

    spike_times = np.squeeze(np.load(sorter_output_path / "spike_times.npy", allow_pickle=True)).astype(float)
    spike_times_seconds = spike_times / imec_sample_rate
    spike_clusters = np.load(sorter_output_path / "spike_clusters.npy", allow_pickle=True).astype(int)
    cluster_info = pd.read_csv(sorter_output_path / "cluster_info.tsv", sep="\t")
    return spike_times_seconds, spike_clusters, cluster_info


def load_aligned_spikes():
    pass


def build_spike_tsgroup(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    cluster_ids: np.ndarray | None = None,
) -> nap.TsGroup:
    """
    Build a Pynapple spike group from flat spike arrays.

    Parameters
    ----------
    spike_times : np.ndarray
        One-dimensional array of spike timestamps with shape ``(n_spikes,)`` and units seconds.
    spike_clusters : np.ndarray
        One-dimensional array of cluster ids with shape ``(n_spikes,)``. Each entry labels
        the corresponding spike in ``spike_times``.
    cluster_ids : np.ndarray | None, optional
        One-dimensional array of cluster ids to include with shape ``(n_units,)``.
        If ``None``, all unique cluster ids in ``spike_clusters`` are used.

    Returns
    -------
    nap.TsGroup
        Pynapple ``TsGroup`` keyed by integer cluster id. Each value is a sorted ``nap.Ts``
        of spike timestamps in seconds.
    """

    if spike_times.ndim != 1 or spike_clusters.ndim != 1:
        raise ValueError("spike_times and spike_clusters must both be one-dimensional arrays.")
    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("spike_times and spike_clusters must have the same length.")

    if cluster_ids is None:
        cluster_ids = np.unique(spike_clusters)

    spike_group_data = {}
    for cluster_id in np.asarray(cluster_ids, dtype=int):
        cluster_spike_times = np.sort(spike_times[spike_clusters == cluster_id].astype(float))
        spike_group_data[int(cluster_id)] = nap.Ts(t=cluster_spike_times)

    return nap.TsGroup(spike_group_data)


def build_lick_time_dict(event_df: pd.DataFrame) -> dict[str, nap.Ts]:
    """
    Build named lick event series from the event table.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event table with columns ``Time`` and ``Event``. ``Time`` values are in seconds.

    Returns
    -------
    dict[str, nap.Ts]
        Dictionary with keys ``"right_entry"`` and ``"left_entry"``. Each value is a
        one-dimensional ``nap.Ts`` of lick timestamps in seconds.
    """

    required_columns = {"Time", "Event"}
    missing_columns = required_columns - set(event_df.columns)
    if missing_columns:
        raise ValueError(f"event_df is missing required columns: {sorted(missing_columns)}")

    lick_time_dict: dict[str, nap.Ts] = {}
    for event_name in ("right_entry", "left_entry"):
        event_times = (
            pd.to_numeric(event_df.loc[event_df["Event"] == event_name, "Time"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        lick_time_dict[event_name] = nap.Ts(t=np.sort(event_times))

    return lick_time_dict


def normalize_region_channels(region_channels: np.ndarray | list[int]) -> np.ndarray:
    """
    Normalize a user-provided channel specification for one brain region.

    Parameters
    ----------
    region_channels : np.ndarray | list[int]
        One-dimensional list-like collection of channel indices with shape ``(n_channels,)``.
        Channel ids are integer labels with no physical-unit conversion.

    Returns
    -------
    np.ndarray
        One-dimensional integer array with shape ``(n_channels,)`` containing the requested
        region channels.
    """

    normalized_channels = np.asarray(region_channels, dtype=int)
    if normalized_channels.ndim != 1:
        raise ValueError("region_channels must be one-dimensional.")
    if normalized_channels.size == 0:
        raise ValueError("region_channels must contain at least one channel.")
    return normalized_channels


def select_units_by_channels(cluster_info: pd.DataFrame, region_channels: np.ndarray | list[int]) -> np.ndarray:
    """
    Select cluster ids assigned to a user-provided set of channels.

    Parameters
    ----------
    cluster_info : pd.DataFrame
        Cluster metadata table with columns ``cluster_id``, ``ch``, and ``group``. ``ch`` is an
        integer channel index with no physical-unit conversion. ``group`` is a cluster-quality
        label, and only ``"good"`` and ``"mua"`` clusters are retained.
    region_channels : np.ndarray | list[int]
        One-dimensional list-like collection of channel indices with shape ``(n_channels,)``.
        Channel ids are integer labels with no physical-unit conversion.

    Returns
    -------
    np.ndarray
        One-dimensional integer array with shape ``(n_region_units,)`` containing cluster ids
        whose main channel matches one of the requested channels and whose quality label is
        ``"good"`` or ``"mua"``.
    """

    required_columns = {"cluster_id", "ch", "group"}
    missing_columns = required_columns - set(cluster_info.columns)
    if missing_columns:
        raise ValueError(f"cluster_info is missing required columns: {sorted(missing_columns)}")

    normalized_channels = normalize_region_channels(region_channels)
    normalized_groups = cluster_info["group"].astype(str).str.strip().str.lower()
    selected_mask = cluster_info["ch"].isin(normalized_channels) & normalized_groups.isin({"good", "mua"})
    return cluster_info.loc[selected_mask, "cluster_id"].to_numpy(dtype=int)


def resolve_trial_end(trial_row: pd.Series) -> float:
    """
    Resolve a trial end time using the legacy reward-choice-start priority order.

    Parameters
    ----------
    trial_row : pd.Series
        Trial metadata row containing ``start_time``, ``choice_time``, and ``reward_time``.
        Time values are scalar floats in seconds.

    Returns
    -------
    float
        Trial end time in seconds.
    """

    reward_time = float(trial_row["reward_time"]) if pd.notna(trial_row["reward_time"]) else np.nan
    choice_time = float(trial_row["choice_time"]) if pd.notna(trial_row["choice_time"]) else np.nan
    start_time = float(trial_row["start_time"])

    if not np.isnan(reward_time):
        return reward_time
    if not np.isnan(choice_time):
        return choice_time
    return start_time


def build_bin_edges(
    trial_start: float,
    trial_end: float,
    bin_size: float,
    pre_time: float,
    post_time: float,
) -> np.ndarray:
    """
    Construct trial-relative bin edges using the legacy floor-division convention.

    Parameters
    ----------
    trial_start : float
        Trial start time in seconds.
    trial_end : float
        Trial end time in seconds.
    bin_size : float
        Width of each time bin in seconds.
    pre_time : float
        Time before trial start to include in seconds.
    post_time : float
        Time after trial end to include in seconds.

    Returns
    -------
    np.ndarray
        One-dimensional float array of bin edges with shape ``(n_bins + 1,)`` and units seconds.
    """

    bin_start = trial_start - pre_time
    bin_end = trial_end + post_time
    n_bins = int((bin_end - bin_start) / bin_size)
    bin_edges = np.arange(bin_start, bin_end, bin_size, dtype=float)
    if bin_edges.size < n_bins + 1:
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_size)
    return bin_edges


def _count_tsgroup_in_bins(
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    bin_edges: np.ndarray,
    bin_size: float,
) -> np.ndarray:
    """
    Count spikes for selected units within one trial interval.

    Parameters
    ----------
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike timestamps are in seconds.
    cluster_ids : np.ndarray
        One-dimensional integer array with shape ``(n_units,)`` listing the unit order
        for the returned count matrix.
    bin_edges : np.ndarray
        One-dimensional float array of bin edges with shape ``(n_bins + 1,)`` and units seconds.
    bin_size : float
        Width of each time bin in seconds.

    Returns
    -------
    np.ndarray
        Two-dimensional count array with shape ``(n_units, n_bins)``. Values are spike counts per bin.
    """

    cluster_ids = np.asarray(cluster_ids, dtype=int)
    if cluster_ids.size == 0:
        return np.zeros((0, max(bin_edges.size - 1, 0)), dtype=float)

    restricted_group = nap.TsGroup({int(cluster_id): spike_group[int(cluster_id)] for cluster_id in cluster_ids})
    interval = nap.IntervalSet(start=[bin_edges[0]], end=[bin_edges[-1]])
    binned_counts = restricted_group.count(bin_size=bin_size, ep=interval)
    return binned_counts.values.T.astype(float)


def _count_ts_in_bins(
    event_times: nap.Ts,
    bin_edges: np.ndarray,
    bin_size: float,
) -> np.ndarray:
    """
    Count one event stream within one trial interval.

    Parameters
    ----------
    event_times : nap.Ts
        One-dimensional event timestamp series with units seconds.
    bin_edges : np.ndarray
        One-dimensional float array of bin edges with shape ``(n_bins + 1,)`` and units seconds.
    bin_size : float
        Width of each time bin in seconds.

    Returns
    -------
    np.ndarray
        One-dimensional count array with shape ``(n_bins,)``. Values are event counts per bin.
    """

    interval = nap.IntervalSet(start=[bin_edges[0]], end=[bin_edges[-1]])
    return event_times.count(bin_size=bin_size, ep=interval).values.astype(float)


def bin_spikes_to_trial_pynapple(
    trial_start: float,
    trial_end: float,
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    bin_size: float = 0.1,
    pre_time: float = 2.0,
    post_time: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin selected spike trains around one trial.

    Parameters
    ----------
    trial_start : float
        Trial start time in seconds.
    trial_end : float
        Trial end time in seconds.
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike timestamps are in seconds.
    cluster_ids : np.ndarray
        One-dimensional integer array with shape ``(n_units,)`` defining row order in the
        returned count matrix.
    bin_size : float, optional
        Width of each time bin in seconds.
    pre_time : float, optional
        Time before trial start to include in seconds.
    post_time : float, optional
        Time after trial end to include in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(binned_spikes, bin_edges)`` where ``binned_spikes`` has shape ``(n_units, n_bins)``
        and contains spike counts, and ``bin_edges`` has shape ``(n_bins + 1,)`` with units seconds.
    """

    bin_edges = build_bin_edges(
        trial_start=trial_start,
        trial_end=trial_end,
        bin_size=bin_size,
        pre_time=pre_time,
        post_time=post_time,
    )
    binned_spikes = _count_tsgroup_in_bins(
        spike_group=spike_group,
        cluster_ids=cluster_ids,
        bin_edges=bin_edges,
        bin_size=bin_size,
    )
    return binned_spikes, bin_edges


def bin_licks_to_trial_pynapple(
    trial_start: float,
    trial_end: float,
    lick_times: Mapping[str, nap.Ts],
    bin_size: float = 0.1,
    pre_time: float = 2.0,
    post_time: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin right and left lick events around one trial.

    Parameters
    ----------
    trial_start : float
        Trial start time in seconds.
    trial_end : float
        Trial end time in seconds.
    lick_times : Mapping[str, nap.Ts]
        Mapping containing ``"right_entry"`` and ``"left_entry"`` keys. Each value is a
        one-dimensional ``nap.Ts`` of lick timestamps in seconds.
    bin_size : float, optional
        Width of each time bin in seconds.
    pre_time : float, optional
        Time before trial start to include in seconds.
    post_time : float, optional
        Time after trial end to include in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(binned_licks, bin_edges)`` where ``binned_licks`` has shape ``(n_bins, 2)``
        and columns are ordered as ``[right, left]`` counts, and ``bin_edges`` has shape
        ``(n_bins + 1,)`` with units seconds.
    """

    if "right_entry" not in lick_times or "left_entry" not in lick_times:
        raise ValueError("lick_times must contain 'right_entry' and 'left_entry' keys.")

    bin_edges = build_bin_edges(
        trial_start=trial_start,
        trial_end=trial_end,
        bin_size=bin_size,
        pre_time=pre_time,
        post_time=post_time,
    )
    binned_licks = np.column_stack(
        [
            _count_ts_in_bins(lick_times["right_entry"], bin_edges, bin_size),
            _count_ts_in_bins(lick_times["left_entry"], bin_edges, bin_size),
        ]
    ).astype(float)
    return binned_licks, bin_edges


def bin_region_trials(
    trial_df: pd.DataFrame,
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    bin_size: float = 0.5,
    pre_time: float = 2.0,
    post_time: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Bin region-selected spikes for every trial while preserving the legacy output structure.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are ``start_time``,
        ``choice_time``, ``reward_time``, ``state_int``, and ``action``.
        Time columns are in seconds.
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike timestamps are in seconds.
    cluster_ids : np.ndarray
        One-dimensional integer array with shape ``(n_units,)`` defining row order in each
        trial's ``binned_spikes`` matrix.
    bin_size : float, optional
        Width of each time bin in seconds.
    pre_time : float, optional
        Time before trial start to include in seconds.
    post_time : float, optional
        Time after trial end to include in seconds.

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per trial. Each dictionary contains:
        ``trial_ix`` (int),
        ``binned_spikes`` (``np.ndarray`` with shape ``(n_units, n_bins)``),
        ``bin_edges`` (``np.ndarray`` with shape ``(n_bins + 1,)`` in seconds),
        ``bin_states`` (``np.ndarray`` with shape ``(n_bins,)``),
        ``bin_choices`` (``np.ndarray`` with shape ``(n_bins,)``).
    """

    required_columns = {"start_time", "choice_time", "reward_time", "state_int", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")

    spikes_trial_binned: list[dict[str, Any]] = []
    for trial_ix, trial_row in trial_df.iterrows():
        trial_start = float(trial_row["start_time"])
        trial_end = resolve_trial_end(trial_row)
        binned_spikes, bin_edges = bin_spikes_to_trial_pynapple(
            trial_start=trial_start,
            trial_end=trial_end,
            spike_group=spike_group,
            cluster_ids=cluster_ids,
            bin_size=bin_size,
            pre_time=pre_time,
            post_time=post_time,
        )
        n_bins = binned_spikes.shape[1]
        spikes_trial_binned.append(
            {
                "trial_ix": int(trial_ix),
                "binned_spikes": binned_spikes,
                "bin_edges": bin_edges,
                "bin_states": np.full(n_bins, float(trial_row["state_int"]), dtype=float),
                "bin_choices": np.full(n_bins, float(trial_row["action"]), dtype=float),
            }
        )

    return spikes_trial_binned


def make_trial_type_masks(trial_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Build boolean trial masks for simple neural-behavior analyses.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are ``give_reward``, ``correct``,
        ``reward``, and ``action``. ``give_reward`` is an experimenter-override flag where
        nonzero values mark trials invalid for neural analysis. ``correct`` and ``reward``
        are scalar trial outcomes with no unit conversion. ``action`` is a scalar choice label.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary of boolean masks indexed like ``trial_df``. Keys are:
        ``valid``,
        ``correct_rewarded``,
        ``rewarded``,
        ``incorrect``,
        ``omission``,
        ``switch``,
        ``stay``,
        ``omission_switch``,
        ``omission_stay``,
        ``incorrect_switch``,
        ``incorrect_stay``.
    """

    required_columns = {"give_reward", "correct", "reward", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")

    valid = trial_df["give_reward"].eq(0)
    correct_rewarded = valid & trial_df["correct"].eq(1) & trial_df["reward"].eq(1)
    incorrect = valid & trial_df["correct"].eq(0) & trial_df["action"].notna()
    omission = valid & trial_df["correct"].eq(1) & trial_df["reward"].eq(0)

    current_unrewarded = valid & trial_df["reward"].eq(0)
    current_action_valid = trial_df["action"].notna()
    next_valid = trial_df["give_reward"].shift(-1).eq(0).fillna(False)
    next_action = trial_df["action"].shift(-1)
    next_action_valid = next_action.notna()
    comparable_next_trial = current_unrewarded & current_action_valid & next_valid & next_action_valid
    switch = comparable_next_trial & next_action.ne(trial_df["action"])
    stay = comparable_next_trial & next_action.eq(trial_df["action"])

    masks = {
        "valid": valid,
        "correct_rewarded": correct_rewarded,
        "rewarded": correct_rewarded,
        "incorrect": incorrect,
        "omission": omission,
        "switch": switch,
        "stay": stay,
    }
    masks["omission_switch"] = masks["omission"] & masks["switch"]
    masks["omission_stay"] = masks["omission"] & masks["stay"]
    masks["incorrect_switch"] = masks["incorrect"] & masks["switch"]
    masks["incorrect_stay"] = masks["incorrect"] & masks["stay"]
    return masks


def summarize_trial_masks(
    trial_masks: Mapping[str, pd.Series],
    condition_names: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Summarize selected trial masks in a user-specified condition order.

    Parameters
    ----------
    trial_masks : Mapping[str, pd.Series]
        Mapping from condition name to boolean trial mask. Each mask must be one-dimensional and
        indexed like the trial table.
    condition_names : list[str]
        Ordered list of condition names to summarize.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        ``(summary_df, condition_trial_indices)`` where ``summary_df`` has one row per requested
        condition with columns ``condition`` and ``n_trials``, and ``condition_trial_indices``
        maps each condition name to a one-dimensional integer array of selected trial indices.
    """

    summary_rows: list[dict[str, Any]] = []
    condition_trial_indices: dict[str, np.ndarray] = {}
    for condition_name in condition_names:
        if condition_name not in trial_masks:
            raise ValueError(f"Requested condition {condition_name!r} is not present in trial_masks.")
        trial_indices = np.flatnonzero(np.asarray(trial_masks[condition_name], dtype=bool))
        condition_trial_indices[condition_name] = trial_indices
        summary_rows.append(
            {
                "condition": condition_name,
                "n_trials": int(trial_indices.size),
            }
        )

    return pd.DataFrame(summary_rows), condition_trial_indices


def make_classifier_bins(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    trial_mask: pd.Series | np.ndarray,
    event: str = "choice_time",
    bounds: tuple[float, float] = (-0.5, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract classifier inputs from pre-binned trial spike counts.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial. Each dictionary must
        contain ``binned_spikes`` with shape ``(n_units, n_bins)``, ``bin_edges`` with shape
        ``(n_bins + 1,)`` in seconds, ``bin_states`` with shape ``(n_bins,)``, and
        ``bin_choices`` with shape ``(n_bins,)``.
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required event columns are ``start_time`` and/or
        ``choice_time`` depending on ``event``. Time values are in seconds.
    trial_mask : pd.Series | np.ndarray
        One-dimensional boolean selector with shape ``(n_trials,)`` indicating which trials to
        include in the extracted classifier bins.
    event : str, optional
        Alignment event name. Supported values are ``"start_time"`` and ``"choice_time"``.
    bounds : tuple[float, float], optional
        Time window relative to ``event`` in seconds as ``(start_offset, end_offset)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(spike_bins, state_bins, choice_bins)`` where ``spike_bins`` has shape
        ``(n_units, n_selected_bins)``, and ``state_bins`` and ``choice_bins`` have shape
        ``(n_selected_bins,)``.
    """

    if event not in {"start_time", "choice_time"}:
        raise ValueError("event must be 'start_time' or 'choice_time'.")
    if len(bounds) != 2:
        raise ValueError("bounds must contain exactly two values.")
    if len(region_trial_binned) != len(trial_df):
        raise ValueError("region_trial_binned and trial_df must have the same number of trials.")

    trial_mask_array = np.asarray(trial_mask, dtype=bool)
    if trial_mask_array.ndim != 1 or trial_mask_array.shape[0] != len(trial_df):
        raise ValueError("trial_mask must be a one-dimensional boolean selector matching trial_df.")

    selected_trial_indices = np.flatnonzero(trial_mask_array)
    if selected_trial_indices.size == 0:
        raise ValueError("No trials were selected for classifier bin extraction.")

    spike_bins = []
    state_bins = []
    choice_bins = []
    for trial_index in selected_trial_indices:
        event_time = trial_df.iloc[trial_index][event]
        if pd.isna(event_time):
            raise ValueError(f"Selected trial {trial_index} is missing {event}.")

        bin_edges = np.asarray(region_trial_binned[trial_index]["bin_edges"], dtype=float)
        trial_bin_mask = (
            (bin_edges >= float(event_time) + bounds[0]) &
            (bin_edges < float(event_time) + bounds[1])
        )[:-1]
        if not np.any(trial_bin_mask):
            raise ValueError(
                f"Selected trial {trial_index} has no bins in bounds {bounds} around {event}."
            )

        spike_bins.append(np.asarray(region_trial_binned[trial_index]["binned_spikes"], dtype=float)[:, trial_bin_mask])
        state_bins.append(np.asarray(region_trial_binned[trial_index]["bin_states"], dtype=float)[trial_bin_mask])
        choice_bins.append(np.asarray(region_trial_binned[trial_index]["bin_choices"], dtype=float)[trial_bin_mask])

    return (
        np.concatenate(spike_bins, axis=1),
        np.concatenate(state_bins),
        np.concatenate(choice_bins),
    )


def collect_condition_classifier_bins(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    trial_masks: Mapping[str, pd.Series],
    condition_names: list[str],
    windows: Mapping[str, tuple[float, float]],
    event: str = "choice_time",
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Collect classifier-ready bins for ordered condition/window combinations.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial.
    trial_df : pd.DataFrame
        Trial table with one row per trial. Must contain the requested alignment event column.
    trial_masks : Mapping[str, pd.Series]
        Mapping from condition name to boolean trial masks indexed like ``trial_df``.
    condition_names : list[str]
        Ordered list of condition names to extract.
    windows : Mapping[str, tuple[float, float]]
        Mapping from window label to relative event bounds in seconds.
    event : str, optional
        Alignment event name passed through to ``make_classifier_bins``.

    Returns
    -------
    dict[tuple[str, str], dict[str, Any]]
        Dictionary keyed by ``(condition_name, window_name)``. Each value contains either:
        ``status == "ok"`` with ``spike_bins``, ``state_bins``, and ``choice_bins``,
        or ``status == "failed"`` with a short failure reason.
    """

    collected_bins: dict[tuple[str, str], dict[str, Any]] = {}
    for condition_name in condition_names:
        if condition_name not in trial_masks:
            raise ValueError(f"Requested condition {condition_name!r} is not present in trial_masks.")

        for window_name, bounds in windows.items():
            key = (condition_name, window_name)
            try:
                spike_bins, state_bins, choice_bins = make_classifier_bins(
                    region_trial_binned=region_trial_binned,
                    trial_df=trial_df,
                    trial_mask=trial_masks[condition_name],
                    event=event,
                    bounds=bounds,
                )
            except ValueError as error:
                collected_bins[key] = {
                    "status": "failed",
                    "reason": str(error),
                    "event": event,
                    "bounds": bounds,
                }
                continue

            collected_bins[key] = {
                "status": "ok",
                "event": event,
                "bounds": bounds,
                "spike_bins": spike_bins,
                "state_bins": state_bins,
                "choice_bins": choice_bins,
            }

    return collected_bins


def get_decode_target(trial_df: pd.DataFrame, target: str = "state_int") -> np.ndarray:
    """
    Extract one supported decoding target from the trial table.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Must contain the requested target column.
    target : str, optional
        Name of the decode target column. Supported values are ``"state_int"`` and ``"action"``.

    Returns
    -------
    np.ndarray
        One-dimensional float array with shape ``(n_trials,)`` containing the requested target values.
    """

    if target not in {"state_int", "action"}:
        raise ValueError("target must be 'state_int' or 'action'.")
    if target not in trial_df.columns:
        raise ValueError(f"trial_df is missing requested target column {target!r}.")
    return pd.to_numeric(trial_df[target], errors="coerce").to_numpy(dtype=float)


def _prepare_decode_inputs(
    binned_spikes: np.ndarray,
    target_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Filter decode inputs and summarize class counts.

    Parameters
    ----------
    binned_spikes : np.ndarray
        Spike-count matrix with shape ``(n_units, n_samples)``.
    target_values : np.ndarray
        One-dimensional target array with shape ``(n_samples,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, int]]
        ``(filtered_spikes, filtered_targets, summary)`` where ``filtered_spikes`` retains shape
        ``(n_units, n_valid_samples)``, ``filtered_targets`` has shape ``(n_valid_samples,)``,
        and ``summary`` contains integer counts for ``n_samples`` and ``n_classes``.
    """

    filtered_targets = np.asarray(target_values, dtype=float)
    filtered_spikes = np.asarray(binned_spikes, dtype=float)
    if filtered_spikes.ndim != 2:
        raise ValueError("binned_spikes must be a two-dimensional array with shape (n_units, n_samples).")
    if filtered_targets.ndim != 1:
        raise ValueError("target_values must be a one-dimensional array.")
    if filtered_spikes.shape[1] != filtered_targets.shape[0]:
        raise ValueError("binned_spikes and target_values must agree on sample count.")

    valid_target_mask = ~np.isnan(filtered_targets)
    filtered_spikes = filtered_spikes[:, valid_target_mask]
    filtered_targets = filtered_targets[valid_target_mask]
    summary = {
        "n_samples": int(filtered_targets.shape[0]),
        "n_classes": int(np.unique(filtered_targets).size),
    }
    return filtered_spikes, filtered_targets, summary


def _failed_decode_result(reason: str, label: str = "", **extra_fields: Any) -> dict[str, Any]:
    """
    Build a standardized failed-decoding result dictionary.

    Parameters
    ----------
    reason : str
        Short machine-readable failure reason.
    label : str, optional
        Optional human-readable label for the attempted decode.
    **extra_fields : Any
        Additional key-value pairs to include in the result dictionary.

    Returns
    -------
    dict[str, Any]
        Result dictionary with ``status == "failed"`` and the supplied metadata.
    """

    result = {"status": "failed", "reason": reason, "label": label}
    result.update(extra_fields)
    return result


def cv_decodeability_score(
    binned_spikes: np.ndarray,
    target_values: np.ndarray,
    cv: int = 5,
    n_permutations: int = 100,
    random_state: int = 42,
    label: str = "",
) -> dict[str, Any]:
    """
    Compute a cross-validated decodeability score with a permutation-test null.

    Parameters
    ----------
    binned_spikes : np.ndarray
        Spike-count matrix with shape ``(n_units, n_samples)``.
    target_values : np.ndarray
        One-dimensional target array with shape ``(n_samples,)``. Values are class labels.
    cv : int, optional
        Number of cross-validation folds.
    n_permutations : int, optional
        Number of label permutations used for ``permutation_test_score``.
    random_state : int, optional
        Random seed passed to the permutation test and classifier.
    label : str, optional
        Human-readable label describing the decodeability run.

    Returns
    -------
    dict[str, Any]
        Result dictionary containing decodeability metrics or an explicit failure reason.
    """

    filtered_spikes, filtered_targets, summary = _prepare_decode_inputs(binned_spikes, target_values)
    result_base = {"label": label, **summary}
    if summary["n_samples"] == 0:
        return _failed_decode_result("no_valid_samples", **result_base)
    if summary["n_classes"] < 2:
        return _failed_decode_result("insufficient_classes", **result_base)

    class_counts = np.unique(filtered_targets, return_counts=True)[1]
    if summary["n_samples"] < cv or int(class_counts.min()) < cv:
        return _failed_decode_result("insufficient_samples", cv=cv, **result_base)

    classifier = LogisticRegression(
        solver="saga",
        # penalty="l1",
        l1_ratio=.5,  # L1 penalty is l1_ratio=1; for L2 set to 0; for elastic net set to .5
        max_iter=10000,
        random_state=random_state,
    )
    cv_score, permutation_scores, cv_pvalue = permutation_test_score(
        classifier,
        filtered_spikes.T,
        filtered_targets,
        scoring="accuracy",
        cv=cv,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return {
        "status": "ok",
        "reason": "",
        "label": label,
        "n_samples": summary["n_samples"],
        "n_classes": summary["n_classes"],
        "cv_score": float(cv_score),
        "cv_pvalue": float(cv_pvalue),
        "permutation_scores": np.asarray(permutation_scores, dtype=float),
        "permutation_score_mean": float(np.mean(permutation_scores)),
        "permutation_score_std": float(np.std(permutation_scores)),
    }


def summarize_decoding_results(
    results_df: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    """
    Select a compact, ordered subset of decoding-result columns for display.

    Parameters
    ----------
    results_df : pd.DataFrame
        Decoding-results table with at least ``condition``, ``window``, ``status``, and ``reason`` columns.
    value_columns : list[str]
        Ordered list of metric columns to include after the core display columns.

    Returns
    -------
    pd.DataFrame
        A shallow copy of ``results_df`` containing only the requested display columns in a stable order.
    """

    required_columns = {"condition", "window", "status", "reason"}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        raise ValueError(f"results_df is missing required columns: {sorted(missing_columns)}")

    selected_columns = ["condition", "window", "status", "reason", *value_columns]
    for column_name in value_columns:
        if column_name not in results_df.columns:
            raise ValueError(f"results_df is missing requested value column {column_name!r}.")

    return results_df.loc[:, selected_columns].copy()


def train_single_decoder_with_shuffle_null(
    binned_spikes: np.ndarray,
    target_values: np.ndarray,
    test_size: float = 0.2,
    n_shuffles: int = 1000,
    random_state: int = 42,
    label: str = "",
) -> tuple[LogisticRegression | None, dict[str, Any]]:
    """
    Train one classifier and compare its held-out score against shuffled-label null fits.

    Parameters
    ----------
    binned_spikes : np.ndarray
        Spike-count matrix with shape ``(n_units, n_samples)``.
    target_values : np.ndarray
        One-dimensional target array with shape ``(n_samples,)``. Values are class labels.
    test_size : float, optional
        Fraction of samples reserved for held-out evaluation.
    n_shuffles : int, optional
        Number of shuffled-label null fits on the training split.
    random_state : int, optional
        Random seed passed to the train-test split and primary classifier.
    label : str, optional
        Human-readable label describing the training run.

    Returns
    -------
    tuple[LogisticRegression | None, dict[str, Any]]
        Fitted classifier and a metrics dictionary, or ``(None, failed_result)`` if fitting is not possible.
    """

    filtered_spikes, filtered_targets, summary = _prepare_decode_inputs(binned_spikes, target_values)
    result_base = {"label": label, **summary}
    if summary["n_samples"] == 0:
        return None, _failed_decode_result("no_valid_samples", **result_base)
    if summary["n_classes"] < 2:
        return None, _failed_decode_result("insufficient_classes", **result_base)
    if summary["n_samples"] < 2:
        return None, _failed_decode_result("insufficient_samples", test_size=test_size, **result_base)

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            filtered_spikes.T,
            filtered_targets,
            test_size=test_size,
            random_state=random_state,
        )
    except ValueError:
        return None, _failed_decode_result("insufficient_samples", test_size=test_size, **result_base)

    if np.unique(y_train).size < 2:
        return None, _failed_decode_result("insufficient_classes_after_split", test_size=test_size, **result_base)

    classifier = LogisticRegression(
        solver="saga",
        # penalty="l1",
        l1_ratio=1,  # L1 penalty is l1_ratio=1; for L2 set to 0; for elastic net set to .5
        max_iter=10000,
        random_state=random_state,
    )
    classifier.fit(x_train, y_train)
    train_accuracy = accuracy_score(y_train, classifier.predict(x_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(x_test))

    shuffle_accuracies = []
    for shuffle_index in range(n_shuffles):
        shuffled_targets = np.array(y_train, copy=True)
        rng = np.random.default_rng(random_state + shuffle_index)
        rng.shuffle(shuffled_targets)
        if np.unique(shuffled_targets).size < 2:
            continue
        shuffle_classifier = LogisticRegression(max_iter=10000)
        shuffle_classifier.fit(x_train, shuffled_targets)
        shuffle_accuracies.append(accuracy_score(y_test, shuffle_classifier.predict(x_test)))

    shuffle_accuracy_array = np.asarray(shuffle_accuracies, dtype=float)
    shuffle_pvalue = float(np.mean(shuffle_accuracy_array >= test_accuracy)) if shuffle_accuracy_array.size else np.nan
    return classifier, {
        "status": "ok",
        "reason": "",
        "label": label,
        "n_samples": summary["n_samples"],
        "n_classes": summary["n_classes"],
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "shuffle_pvalue": shuffle_pvalue,
        "shuffle_accuracy_mean": float(np.mean(shuffle_accuracy_array)) if shuffle_accuracy_array.size else np.nan,
        "shuffle_accuracy_std": float(np.std(shuffle_accuracy_array)) if shuffle_accuracy_array.size else np.nan,
        "n_shuffles": int(n_shuffles),
    }


def evaluate_decoder_on_condition(
    classifier: LogisticRegression | None,
    binned_spikes: np.ndarray,
    target_values: np.ndarray,
    label: str = "",
) -> dict[str, Any]:
    """
    Evaluate a fitted classifier on one condition-specific dataset.

    Parameters
    ----------
    classifier : LogisticRegression | None
        Fitted classifier from ``train_single_decoder_with_shuffle_null``.
    binned_spikes : np.ndarray
        Spike-count matrix with shape ``(n_units, n_samples)``.
    target_values : np.ndarray
        One-dimensional target array with shape ``(n_samples,)``. Values are class labels.
    label : str, optional
        Human-readable label for the evaluation set.

    Returns
    -------
    dict[str, Any]
        Evaluation result dictionary containing held-dataset accuracy or an explicit failure reason.
    """

    filtered_spikes, filtered_targets, summary = _prepare_decode_inputs(binned_spikes, target_values)
    result_base = {"label": label, **summary}
    if classifier is None:
        return _failed_decode_result("missing_classifier", **result_base)
    if summary["n_samples"] == 0:
        return _failed_decode_result("no_valid_samples", **result_base)
    if summary["n_classes"] < 2:
        return _failed_decode_result("insufficient_classes", **result_base)

    predictions = classifier.predict(filtered_spikes.T)
    return {
        "status": "ok",
        "reason": "",
        "label": label,
        "n_samples": summary["n_samples"],
        "n_classes": summary["n_classes"],
        "test_accuracy": float(accuracy_score(filtered_targets, predictions)),
    }


def run_base_condition_decoding(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    target: str = "state_int",
    cv: int = 5,
    n_permutations: int = 100,
    n_shuffles: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run the simple base-condition decoding workflow around choice time.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial.
    trial_df : pd.DataFrame
        Trial table containing mask columns, event times, and decode targets.
    target : str, optional
        Decode target column name. Supported values are ``"state_int"`` and ``"action"``.
    cv : int, optional
        Number of folds used for decodeability scoring.
    n_permutations : int, optional
        Number of permutations used for cross-validated decodeability scoring.
    n_shuffles : int, optional
        Number of shuffled-label null fits for the single trained decoder.
    random_state : int, optional
        Base random seed used for all stochastic decoding steps.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the fitted classifier, training metrics, decodeability results,
        and generalization results for base trial conditions and pre/post choice windows.
    """

    get_decode_target(trial_df, target=target)
    trial_masks = make_trial_type_masks(trial_df)
    base_conditions = ["correct_rewarded", "incorrect", "omission", "switch", "stay"]
    windows = {
        "pre_choice": (-0.5, 0.0),
        "post_choice": (0.0, 0.5),
    }

    decodeability_rows: list[dict[str, Any]] = []
    extracted_bins: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    for condition in base_conditions:
        for window_name, bounds in windows.items():
            condition_label = f"{condition}_{window_name}"
            try:
                spike_bins, state_bins, choice_bins = make_classifier_bins(
                    region_trial_binned=region_trial_binned,
                    trial_df=trial_df,
                    trial_mask=trial_masks[condition],
                    event="choice_time",
                    bounds=bounds,
                )
            except ValueError as error:
                extracted_bins[(condition, window_name)] = None
                decodeability_rows.append(
                    {
                        "condition": condition,
                        "window": window_name,
                        "target": target,
                        **_failed_decode_result("no_selected_bins", label=condition_label, error=str(error)),
                    }
                )
                continue

            condition_targets = state_bins if target == "state_int" else choice_bins
            extracted_bins[(condition, window_name)] = (spike_bins, condition_targets, np.array([]))
            decodeability_result = cv_decodeability_score(
                binned_spikes=spike_bins,
                target_values=condition_targets,
                cv=cv,
                n_permutations=n_permutations,
                random_state=random_state,
                label=condition_label,
            )
            decodeability_rows.append(
                {
                    "condition": condition,
                    "window": window_name,
                    "target": target,
                    **decodeability_result,
                }
            )

    train_key = ("correct_rewarded", "pre_choice")
    train_bins = extracted_bins.get(train_key)
    if train_bins is None:
        classifier = None
        training_result = _failed_decode_result(
            "missing_training_bins",
            label="correct_rewarded_pre_choice",
            condition="correct_rewarded",
            window="pre_choice",
            target=target,
        )
    else:
        classifier, training_metrics = train_single_decoder_with_shuffle_null(
            binned_spikes=train_bins[0],
            target_values=train_bins[1],
            n_shuffles=n_shuffles,
            random_state=random_state,
            label="correct_rewarded_pre_choice",
        )
        training_result = {
            "condition": "correct_rewarded",
            "window": "pre_choice",
            "target": target,
            **training_metrics,
        }

    generalization_rows: list[dict[str, Any]] = []
    for condition in base_conditions:
        for window_name in windows:
            if condition == "correct_rewarded" and window_name == "pre_choice":
                generalization_rows.append(
                    {
                        "condition": condition,
                        "window": window_name,
                        "target": target,
                        **{key: value for key, value in training_result.items() if key not in {"shuffle_accuracy_mean", "shuffle_accuracy_std", "shuffle_pvalue", "n_shuffles"}},
                        "test_accuracy": training_result.get("test_accuracy", np.nan),
                    }
                )
                continue

            condition_bins = extracted_bins.get((condition, window_name))
            if condition_bins is None:
                evaluation_result = _failed_decode_result(
                    "no_selected_bins",
                    label=f"{condition}_{window_name}",
                )
            else:
                evaluation_result = evaluate_decoder_on_condition(
                    classifier=classifier,
                    binned_spikes=condition_bins[0],
                    target_values=condition_bins[1],
                    label=f"{condition}_{window_name}",
                )
            generalization_rows.append(
                {
                    "condition": condition,
                    "window": window_name,
                    "target": target,
                    **evaluation_result,
                }
            )

    return {
        "classifier": classifier,
        "training_result": training_result,
        "decodeability_results": pd.DataFrame(decodeability_rows),
        "generalization_results": pd.DataFrame(generalization_rows),
    }


def main() -> None:
    """
    Run one hard-coded session example for user-defined region spike binning.

    The script loads processed behavior tables, loads sorter spikes from the current
    example HPC probe, builds a Pynapple spike group, bins region-selected spikes by trial, and
    prints a short summary. Times are handled in seconds throughout.
    """

    multi_session_save_path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis")
    session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference")
    sess_id_full = "CT014_2025-12-16_153200"
    raw_behavior_folder = session_data_home / "rpi" / sess_id_full
    processed_data_path = session_data_home / "processed"
    figure_path = session_data_home / "figures"

    pfc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/Kilosort2.5.2_2026-03-19_180103"
    pfc_ap_bin_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/run0_g0_tcat.imec0.ap.bin"
    hpc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/Kilosort2.5.2_2026-03-19_183540"
    hpc_ap_bin_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/run0_g0_tcat.imec1.ap.bin"

    aligned_spike_path = session_data_home / 'ephys/aligned/aligned_imec'
    aligned_pfc_spike_path = aligned_spike_path / 'imec0_sync.npz'
    aligned_hpc_spike_path = aligned_spike_path / 'imec1_sync.npz'

    probe_json_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/probe_json.json"

    if not hpc_spike_path.exists():
        raise FileNotFoundError(f"HPC sorter output not found at {hpc_spike_path}")
    if not hpc_ap_bin_path.exists():
        raise FileNotFoundError(f"HPC AP binary not found at {hpc_ap_bin_path}")
    assert probe_json_path.exists(), f"Probe JSON file not found at {probe_json_path}"
    assert aligned_pfc_spike_path.exists(), f"Aligned PFC spike file not found at {aligned_hpc_spike_path}"
    assert aligned_hpc_spike_path.exists(), f"Aligned HPC spike file not found at {aligned_hpc_spike_path}"

    # probe_sites = get_brain_channels.load_probe_json(probe_json_path)
    # brain_channels, shank_sites = get_brain_channels.get_brain_and_shank_sites(
    #     probe_sites,
    #     sort_order="shallow_to_deep",
    # )

    # aligned_pfc_spikes = np.load(aligned_pfc_spike_path)
    aligned_hpc_spikes = np.load(aligned_hpc_spike_path)

    decode_target = "state_int"
    region_name = "HPC"
    # print(", ".join(map(str, "your_array")))
    hpc_channels_shank0 = np.array([
        4, 3, 2, 1, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181,
        180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167,
        166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153,
        152, 151, 150, 149, 148, 147, 146, 145, 96, 95, 94, 93, 92, 91, 90, 89,
        88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71,
        70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53,
        52, 51, 50, 49, 384, 383, 382, 381
    ])
    hpc_channels_shank3 = np.array([
        255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242,
        241, 336, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326, 325, 324,
        323, 322, 321, 320, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310,
        309, 308, 307, 306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296,
        295, 294, 293, 292, 291, 290, 289, 240, 239, 238, 237, 236, 235, 234,
        233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220,
        219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206,
        205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 144,
        143, 142, 141
    ])
    hpc_channels = np.concatenate([hpc_channels_shank0, hpc_channels_shank3])

    # region_name = "V1"
    v1_channels_shank0 = np.array([
        138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125,
        124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
        110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97,
        48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
        14, 13, 12, 11, 10, 9, 8, 7, 6, 5
    ])
    v1_channels_shank3 = np.array([
        78, 377, 376, 375, 374, 373, 372, 371, 370, 369, 368, 367, 366, 365,
        364, 363, 362, 361, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351,
        350, 349, 348, 347, 346, 345, 344, 343, 342, 341, 340, 339, 338, 337,
        288, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275,
        274, 273, 272, 271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261,
        260, 259, 258, 257, 256
    ])
    v1_channels = np.concatenate([v1_channels_shank0, v1_channels_shank3])

    region_channels = normalize_region_channels(hpc_channels)

    mouse, date, timestamp = parse_session_id(sess_id_full)
    session_info_path = raw_behavior_folder / f"{sess_id_full}_session_info.pkl"

    session = Session(
        multi_session_save_path=multi_session_save_path,
        session_data_home=session_data_home,
        sess_id_full=sess_id_full,
        sess_id_abbreviated=f"{mouse}_{date}",
        raw_behavior_folder=raw_behavior_folder,
        processed_data_path=processed_data_path,
        figure_path=figure_path,
        mouse=mouse,
        date=date,
        timestamp=timestamp,
        session_info_path=session_info_path,
        session_info=load_session_info(session_info_path) if session_info_path.exists() else None,
        pfc_spike_path=pfc_spike_path,
        hpc_spike_path=hpc_spike_path,
    )

    event_df, trial_df = load_session_tables(session)
    lick_times = build_lick_time_dict(event_df)
    spike_times, spike_clusters, cluster_info = load_sorter_spikes(
        sorter_output_path=session.hpc_spike_path,
        ap_bin_path=hpc_ap_bin_path,
    )

    region_cluster_ids = select_units_by_channels(cluster_info, region_channels=region_channels)
    if region_cluster_ids.size == 0:
        raise RuntimeError(
            f"No {region_name} units were found in the configured sorter output for channels {region_channels.tolist()}."
        )

    region_spike_group = build_spike_tsgroup(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        cluster_ids=region_cluster_ids,
    )
    region_spike_bins = bin_region_trials(
        trial_df=trial_df,
        spike_group=region_spike_group,
        cluster_ids=region_cluster_ids,
        bin_size=0.5,
        pre_time=2.0,
        post_time=2.0,
    )
    trial_masks = make_trial_type_masks(trial_df)
    condition_names = [
        "correct_rewarded",
        "incorrect",
        "omission",
        "switch",
        "stay",
        "omission_switch",
        "omission_stay",
        "incorrect_switch",
        "incorrect_stay",
    ]
    choice_windows = {
        "pre_choice": (-0.5, 0.0),
        "post_choice": (0.0, 0.5),
    }
    mask_summary_df, condition_trial_indices = summarize_trial_masks(
        trial_masks=trial_masks,
        condition_names=condition_names,
    )
    classifier_bins_by_condition = collect_condition_classifier_bins(
        region_trial_binned=region_spike_bins,
        trial_df=trial_df,
        trial_masks=trial_masks,
        condition_names=condition_names,
        windows=choice_windows,
        event="choice_time",
    )
    decoding_results = run_base_condition_decoding(
        region_trial_binned=region_spike_bins,
        trial_df=trial_df,
        target=decode_target,
        cv=5,
        n_permutations=100,
        n_shuffles=1000,
        random_state=0,
    )

    first_trial_end = resolve_trial_end(trial_df.iloc[0])
    first_trial_licks, _ = bin_licks_to_trial_pynapple(
        trial_start=float(trial_df.iloc[0]["start_time"]),
        trial_end=first_trial_end,
        lick_times=lick_times,
        bin_size=0.5,
        pre_time=2.0,
        post_time=2.0,
    )

    print(f"Session: {session.sess_id_full}")
    print(f"Region: {region_name}")
    print(f"Region units: {region_cluster_ids.size}")
    print(f"Trials binned: {len(region_spike_bins)}")
    print(f"First trial spike-bin shape: {region_spike_bins[0]['binned_spikes'].shape}")
    print(f"First trial lick-bin shape: {first_trial_licks.shape}")
    print(f"Decode target: {decode_target}")
    print("Trial condition counts:")
    for _, summary_row in mask_summary_df.iterrows():
        condition_name = str(summary_row["condition"])
        condition_count = int(summary_row["n_trials"])
        trial_indices = condition_trial_indices[condition_name].tolist()
        print(f"  {condition_name}: {condition_count} trials {trial_indices}")

    print("Classifier bin summary:")
    for condition_name in condition_names:
        for window_name in choice_windows:
            collected_entry = classifier_bins_by_condition[(condition_name, window_name)]
            if collected_entry["status"] == "ok":
                print(
                    f"  {condition_name} {window_name}: spike_bins {collected_entry['spike_bins'].shape}, "
                    f"labels {collected_entry['state_bins'].shape[0]}"
                )
            else:
                print(f"  {condition_name} {window_name}: failed ({collected_entry['reason']})")
    print("CV decodeability summary:")
    cv_summary_df = summarize_decoding_results(
        decoding_results["decodeability_results"],
        value_columns=["cv_score", "cv_pvalue"],
    )
    for _, summary_row in cv_summary_df.iterrows():
        if summary_row["status"] == "ok":
            print(
                f"  {summary_row['condition']} {summary_row['window']}: "
                f"cv_score={summary_row['cv_score']:.3f}, cv_pvalue={summary_row['cv_pvalue']:.3f}"
            )
        else:
            print(
                f"  {summary_row['condition']} {summary_row['window']}: "
                f"failed ({summary_row['reason']})"
            )
    print("One-shot decoder summary:")
    training_result = decoding_results["training_result"]
    if training_result["status"] == "ok":
        print(
            "  trained on correct_rewarded pre_choice: "
            f"train_acc={training_result['train_accuracy']:.3f}, "
            f"test_acc={training_result['test_accuracy']:.3f}, "
            f"shuffle_p={training_result['shuffle_pvalue']:.3f}"
        )
    else:
        print(f"  training failed ({training_result['reason']})")
    generalization_summary_df = summarize_decoding_results(
        decoding_results["generalization_results"],
        value_columns=["test_accuracy"],
    )
    for _, summary_row in generalization_summary_df.iterrows():
        if summary_row["status"] == "ok":
            print(
                f"  {summary_row['condition']} {summary_row['window']}: "
                f"accuracy={summary_row['test_accuracy']:.3f}"
            )
        else:
            print(
                f"  {summary_row['condition']} {summary_row['window']}: "
                f"failed ({summary_row['reason']})"
            )


if __name__ == "__main__":
    main()
