"""
Prepare HPC trial-aligned spike and lick bins using Pynapple time-series objects.
"""

import pickle as pkl
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score, train_test_split
from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    normalize_experimenter_reward_column,
)


SUPPORTED_ANALYSIS_REGIONS = {"HPC", "V1", "PFC"}


def normalize_region_name_for_filename(region_name: str) -> str:
    """
    Normalize a brain-region label for analysis filenames.

    Parameters
    ----------
    region_name : str
        Brain-region label. Supported values are ``"HPC"``, ``"V1"``, and
        ``"PFC"``, case-insensitive.

    Returns
    -------
    str
        Canonical region label for filenames.
    """

    normalized_region = str(region_name).strip().upper()
    if normalized_region not in SUPPORTED_ANALYSIS_REGIONS:
        raise ValueError(
            f"Unsupported region {region_name!r}. Expected one of {sorted(SUPPORTED_ANALYSIS_REGIONS)}."
        )
    return normalized_region


def make_region_analysis_filename(region_name: str, base_filename: str) -> str:
    """
    Prefix an analysis filename with a canonical brain-region label.

    Parameters
    ----------
    region_name : str
        Brain-region label used to namespace per-session outputs.
    base_filename : str
        Base analysis filename including extension.

    Returns
    -------
    str
        Region-prefixed filename, e.g. ``"HPC_state_decodability_analysis.csv"``.
    """

    return f"{normalize_region_name_for_filename(region_name)}_{base_filename}"


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


def load_sorter_metadata(sorter_output_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load sorter cluster assignments and cluster metadata for aligned-spike analyses.

    Parameters
    ----------
    sorter_output_path : Path
        Path to a sorter output directory containing ``spike_clusters.npy`` and ``cluster_info.tsv``.
        Units: filesystem path.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        ``(spike_clusters, cluster_info)`` where:
        ``spike_clusters`` has shape ``(n_spikes,)`` and contains integer cluster ids aligned
        one-to-one with an external aligned spike-time array,
        ``cluster_info`` has one row per cluster with sorter metadata.
    """

    sorter_path = Path(sorter_output_path)
    spike_clusters = np.load(sorter_path / "spike_clusters.npy", allow_pickle=True).astype(int)
    cluster_info_path = sorter_path / "cluster_info.tsv"
    if not cluster_info_path.exists():
        raise FileNotFoundError(
            "Missing cluster_info.tsv: manual spike curation in Phy must be performed before "
            f"behavior integration, or its output is missing: {cluster_info_path}"
        )
    cluster_info = pd.read_csv(cluster_info_path, sep="\t")
    return spike_clusters, cluster_info


def load_aligned_spikes(aligned_spike_path: Path) -> np.ndarray:
    """
    Load aligned spike timestamps for behavior-aligned neural analyses.

    Parameters
    ----------
    aligned_spike_path : Path
        Path to an ``.npz`` file containing behavior-aligned spike timestamps.
        This file must include a ``spike_utc_unix`` array.

    Returns
    -------
    np.ndarray
        One-dimensional float array with shape ``(n_spikes,)`` containing aligned spike timestamps
        in UTC Unix time, units seconds.
    """

    aligned_spike_file = np.load(aligned_spike_path)
    if "spike_utc_unix" not in aligned_spike_file:
        raise ValueError(
            f"Aligned spike file {aligned_spike_path} is missing required 'spike_utc_unix' array."
        )

    aligned_spike_times = np.asarray(aligned_spike_file["spike_utc_unix"], dtype=float)
    if aligned_spike_times.ndim != 1:
        raise ValueError("Aligned spike times must be one-dimensional.")
    return aligned_spike_times


def validate_aligned_spike_inputs(aligned_spike_times: np.ndarray, spike_clusters: np.ndarray) -> None:
    """
    Validate that aligned spike times and sorter cluster ids describe the same spike sequence.

    Parameters
    ----------
    aligned_spike_times : np.ndarray
        One-dimensional aligned spike-time array with shape ``(n_spikes,)`` and units seconds.
    spike_clusters : np.ndarray
        One-dimensional cluster-id array with shape ``(n_spikes,)`` aligned spike-by-spike to
        ``aligned_spike_times``.

    Returns
    -------
    None
        This function returns nothing and raises on invalid inputs.
    """

    if aligned_spike_times.ndim != 1 or spike_clusters.ndim != 1:
        raise ValueError("aligned_spike_times and spike_clusters must both be one-dimensional arrays.")
    if aligned_spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("aligned_spike_times and spike_clusters must have the same length.")


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


def _transform_plot_times(
    times: np.ndarray,
    time_mode: str,
    reference_time: float,
    session_start_time: float | None,
) -> np.ndarray:
    """
    Transform timestamps for trial-inspection plots.

    Parameters
    ----------
    times : np.ndarray
        One-dimensional float array with shape ``(n_times,)`` containing timestamps in UTC Unix
        seconds.
    time_mode : str
        Plotting time base. Supported values are ``"utc"``, ``"session"``, and ``"event"``.
    reference_time : float
        Reference event timestamp for the selected trial, in UTC Unix seconds.
    session_start_time : float | None
        Session start timestamp in UTC Unix seconds. Required when ``time_mode == "session"``.

    Returns
    -------
    np.ndarray
        One-dimensional float array with shape ``(n_times,)``. Units are seconds in the requested
        plotting frame.
    """

    if time_mode == "utc":
        return times
    if time_mode == "session":
        if session_start_time is None:
            raise ValueError("session_start_time is required when time_mode='session'.")
        return times - float(session_start_time)
    if time_mode == "event":
        return times - reference_time
    raise ValueError("time_mode must be one of {'utc', 'session', 'event'}.")


def plot_trial_raster(
    region_spike_group: nap.TsGroup,
    trial_df: pd.DataFrame,
    trial_ix: int,
    lick_times: Mapping[str, nap.Ts] | None = None,
    event: str = "choice_time",
    time_mode: str = "utc",
    session_start_time: float | None = None,
    pre_time: float = 2.0,
    post_time: float = 2.0,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one trial's spikes and licks in an interactive two-panel raster view.

    Parameters
    ----------
    region_spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike timestamps must be in UTC Unix
        seconds and all units share the same time base.
    trial_df : pd.DataFrame
        Trial table with one row per trial. The selected trial must contain the requested
        alignment event column. Event timestamps are UTC Unix seconds.
    trial_ix : int
        Integer row index into ``trial_df`` selecting one trial to display.
    lick_times : Mapping[str, nap.Ts] | None, optional
        Mapping containing optional ``"right_entry"`` and ``"left_entry"`` lick series. Each
        series is one-dimensional and uses UTC Unix seconds.
    event : str, optional
        Alignment event column name. Supported values are ``"choice_time"`` and ``"start_time"``.
    time_mode : str, optional
        X-axis time base. Supported values are ``"utc"``, ``"session"``, and ``"event"``.
    session_start_time : float | None, optional
        Session start timestamp in UTC Unix seconds. Required when ``time_mode == "session"``.
    pre_time : float, optional
        Seconds before the alignment event to include in the plot window.
    post_time : float, optional
        Seconds after the alignment event to include in the plot window.
    show : bool, optional
        If ``True``, display the figure immediately with ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        ``(figure, axes)`` where ``axes`` has shape ``(2,)``. ``axes[0]`` is the spike raster axis
        and ``axes[1]`` is the lick raster axis.
    """

    if event not in {"choice_time", "start_time"}:
        raise ValueError("event must be 'choice_time' or 'start_time'.")
    if time_mode not in {"utc", "session", "event"}:
        raise ValueError("time_mode must be one of {'utc', 'session', 'event'}.")
    if trial_ix not in trial_df.index:
        raise ValueError(f"trial_ix {trial_ix} is not present in trial_df.")
    if time_mode == "session" and session_start_time is None:
        raise ValueError("session_start_time is required when time_mode='session'.")

    trial_row = trial_df.loc[trial_ix]
    reference_time = pd.to_numeric(pd.Series([trial_row[event]]), errors="coerce").iloc[0]
    if pd.isna(reference_time):
        raise ValueError(f"Trial {trial_ix} has no valid {event} timestamp.")
    reference_time = float(reference_time)

    window_start = reference_time - float(pre_time)
    window_end = reference_time + float(post_time)
    plot_interval = nap.IntervalSet(start=[window_start], end=[window_end])

    restricted_spike_group = region_spike_group.restrict(plot_interval)
    cluster_ids = list(restricted_spike_group.keys())
    if cluster_ids:
        spike_tsd = restricted_spike_group.to_tsd(np.arange(len(cluster_ids), dtype=float))
        spike_x = _transform_plot_times(
            times=np.asarray(spike_tsd.index.to_numpy(), dtype=float),
            time_mode=time_mode,
            reference_time=reference_time,
            session_start_time=session_start_time,
        )
        spike_y = spike_tsd.values.astype(float)
    else:
        spike_x = np.array([], dtype=float)
        spike_y = np.array([], dtype=float)

    if time_mode == "event":
        reference_x = 0.0
    else:
        reference_x = float(
            _transform_plot_times(
                times=np.array([reference_time], dtype=float),
                time_mode=time_mode,
                reference_time=reference_time,
                session_start_time=session_start_time,
            )[0]
        )

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), height_ratios=[3, 1])
    spike_axis, lick_axis = axes

    spike_axis.scatter(spike_x, spike_y, marker="|", color="black", s=80)
    spike_axis.axvline(reference_x, color="tab:red", linestyle="--", linewidth=1.5)
    spike_axis.set_ylabel("Unit")
    spike_axis.set_title(f"Trial {trial_ix} aligned to {event}")
    if cluster_ids:
        spike_axis.set_yticks(np.arange(len(cluster_ids), dtype=float))
        spike_axis.set_yticklabels([str(cluster_id) for cluster_id in cluster_ids])

    lick_axis.axvline(reference_x, color="tab:red", linestyle="--", linewidth=1.5)
    lick_axis.set_ylabel("Lick")

    lick_positions = {"right_entry": 1.0, "left_entry": 0.0}
    lick_colors = {"right_entry": "tab:blue", "left_entry": "tab:orange"}
    if lick_times is not None:
        for lick_name in ("right_entry", "left_entry"):
            if lick_name not in lick_times:
                continue
            restricted_licks = lick_times[lick_name].restrict(plot_interval)
            transformed_licks = _transform_plot_times(
                times=np.asarray(restricted_licks.index.to_numpy(), dtype=float),
                time_mode=time_mode,
                reference_time=reference_time,
                session_start_time=session_start_time,
            )
            lick_axis.scatter(
                transformed_licks,
                np.full(transformed_licks.shape, lick_positions[lick_name], dtype=float),
                marker="|",
                color=lick_colors[lick_name],
                s=120,
                label=lick_name,
            )

    lick_axis.set_yticks([0.0, 1.0])
    lick_axis.set_yticklabels(["left", "right"])
    if lick_times is not None:
        handles, labels = lick_axis.get_legend_handles_labels()
        if handles:
            lick_axis.legend(loc="upper right")

    if time_mode == "utc":
        lick_axis.set_xlabel("Time (UTC Unix s)")
    elif time_mode == "session":
        lick_axis.set_xlabel("Time Since Session Start (s)")
    else:
        lick_axis.set_xlabel(f"Time From {event} (s)")

    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes


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


def select_units_by_channels(
    cluster_info: pd.DataFrame,
    region_channels: np.ndarray | list[int],
    default_group: str = "mua",
    quality_column: str = "group",
) -> np.ndarray:
    """
    Select cluster ids assigned to a user-provided set of channels.

    Parameters
    ----------
    cluster_info : pd.DataFrame
        Cluster metadata table with columns ``cluster_id``, ``ch``, and
        ``quality_column``. ``ch`` is an integer channel index with no
        physical-unit conversion. ``quality_column`` is a cluster-quality label,
        and only ``"good"`` and ``"mua"`` clusters are retained.
    region_channels : np.ndarray | list[int]
        One-dimensional list-like collection of channel indices with shape ``(n_channels,)``.
        Channel ids are integer labels with no physical-unit conversion.
    default_group : str, default="mua"
        Label used when the quality column contains missing values or missing
        sentinels such as ``"nan"``. Units: not applicable.
    quality_column : str, default="group"
        Name of the cluster-quality column to use, such as ``"group"`` for
        manual Phy labels or ``"KSLabel"`` for Kilosort labels. Units: not
        applicable.

    Returns
    -------
    np.ndarray
        One-dimensional integer array with shape ``(n_region_units,)`` containing cluster ids
        whose main channel matches one of the requested channels and whose quality label is
        ``"good"`` or ``"mua"``.
    """

    required_columns = {"cluster_id", "ch", str(quality_column)}
    missing_columns = required_columns - set(cluster_info.columns)
    if missing_columns:
        raise ValueError(f"cluster_info is missing required columns: {sorted(missing_columns)}")

    normalized_channels = normalize_region_channels(region_channels)
    normalized_groups = cluster_info[str(quality_column)].copy()
    normalized_groups.loc[normalized_groups.isna()] = default_group
    normalized_groups = normalized_groups.astype(str).str.strip().str.lower()
    missing_sentinels = {"", "nan", "none"}
    normalized_groups.loc[normalized_groups.isin(missing_sentinels)] = default_group
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
        Trial table with one row per trial. Required columns are
        ``experimenter_reward_given``, ``correct``, ``reward``, and ``action``.
        Legacy ``give_reward`` columns are accepted and normalized at this
        boundary. Nonzero experimenter-reward values mark trials invalid for
        neural analysis. ``correct`` and ``reward`` are scalar trial outcomes
        with no unit conversion. ``action`` is a scalar choice label.

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

    trial_df = normalize_experimenter_reward_column(trial_df)
    required_columns = {EXPERIMENTER_REWARD_GIVEN_COLUMN, "correct", "reward", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")

    valid = trial_df[EXPERIMENTER_REWARD_GIVEN_COLUMN].eq(0)
    correct_rewarded = valid & trial_df["correct"].eq(1) & trial_df["reward"].eq(1)
    incorrect = valid & trial_df["correct"].eq(0) & trial_df["action"].notna()
    omission = valid & trial_df["correct"].eq(1) & trial_df["reward"].eq(0)

    current_unrewarded = valid & trial_df["reward"].eq(0)
    current_action_valid = trial_df["action"].notna()
    next_valid = trial_df[EXPERIMENTER_REWARD_GIVEN_COLUMN].shift(-1).eq(0).fillna(False)
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


def _collect_base_condition_decode_inputs(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    target: str,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None]:
    """
    Extract base-condition classifier bins for repeated decode workflows.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial.
    trial_df : pd.DataFrame
        Trial table with mask columns, event times, and decode targets.
    target : str
        Decode target column name. Supported values are ``"state_int"`` and ``"action"``.

    Returns
    -------
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None]
        Mapping from ``(condition, window_name)`` to ``(spike_bins, target_values)``.
        Missing or empty selections map to ``None``.
    """

    get_decode_target(trial_df, target=target)
    trial_masks = make_trial_type_masks(trial_df)
    base_conditions = ["correct_rewarded", "incorrect", "omission", "switch", "stay"]
    windows = {
        "pre_choice": (-0.5, 0.0),
        "post_choice": (0.0, 0.5),
    }

    extracted_bins: dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None] = {}
    for condition in base_conditions:
        for window_name, bounds in windows.items():
            try:
                spike_bins, state_bins, choice_bins = make_classifier_bins(
                    region_trial_binned=region_trial_binned,
                    trial_df=trial_df,
                    trial_mask=trial_masks[condition],
                    event="choice_time",
                    bounds=bounds,
                )
            except ValueError:
                extracted_bins[(condition, window_name)] = None
                continue

            target_values = state_bins if target == "state_int" else choice_bins
            extracted_bins[(condition, window_name)] = (spike_bins, target_values)
    return extracted_bins


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

    base_conditions = ["correct_rewarded", "incorrect", "omission", "switch", "stay"]
    windows = {
        "pre_choice": (-0.5, 0.0),
        "post_choice": (0.0, 0.5),
    }
    extracted_bin_inputs = _collect_base_condition_decode_inputs(
        region_trial_binned=region_trial_binned,
        trial_df=trial_df,
        target=target,
    )

    decodeability_rows: list[dict[str, Any]] = []
    for condition in base_conditions:
        for window_name in windows:
            condition_label = f"{condition}_{window_name}"
            extracted_bin_entry = extracted_bin_inputs[(condition, window_name)]
            if extracted_bin_entry is None:
                decodeability_rows.append(
                    {
                        "condition": condition,
                        "window": window_name,
                        "target": target,
                        **_failed_decode_result("no_selected_bins", label=condition_label),
                    }
                )
                continue

            spike_bins, condition_targets = extracted_bin_entry
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

    train_key = ("correct_rewarded", "post_choice")
    train_bins = extracted_bin_inputs.get(train_key)
    if train_bins is None:
        classifier = None
        training_result = _failed_decode_result(
            "missing_training_bins",
            label="correct_rewarded_post_choice",
            condition="correct_rewarded",
            window="post_choice",
            target=target,
        )
    else:
        classifier, training_metrics = train_single_decoder_with_shuffle_null(
            binned_spikes=train_bins[0],
            target_values=train_bins[1],
            n_shuffles=n_shuffles,
            random_state=random_state,
            label="correct_rewarded_post_choice",
        )
        training_result = {
            "condition": "correct_rewarded",
            "window": "post_choice",
            "target": target,
            **training_metrics,
        }

    generalization_rows: list[dict[str, Any]] = []
    for condition in base_conditions:
        for window_name in windows:
            if condition == "correct_rewarded" and window_name == "post_choice":
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

            condition_bins = extracted_bin_inputs.get((condition, window_name))
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


def run_repeated_correct_rewarded_decoder(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    target: str = "state_int",
    n_decoder_runs: int = 5,
    n_shuffles: int = 1000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Train multiple correct-rewarded post-choice decoders and evaluate each across base conditions.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial.
    trial_df : pd.DataFrame
        Trial table with mask columns, event times, and decode targets.
    target : str, optional
        Decode target column name. Supported values are ``"state_int"`` and ``"action"``.
    n_decoder_runs : int, optional
        Number of independently seeded decoder fits to run.
    n_shuffles : int, optional
        Number of shuffled-label null fits per decoder run.
    random_state : int, optional
        Base random seed. Each decoder run uses ``random_state + decoder_run``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with ``training_results`` and ``generalization_results`` tables.
    """

    base_conditions = ["correct_rewarded", "incorrect", "omission", "switch", "stay"]
    windows = ["pre_choice", "post_choice"]
    extracted_bin_inputs = _collect_base_condition_decode_inputs(
        region_trial_binned=region_trial_binned,
        trial_df=trial_df,
        target=target,
    )

    training_rows: list[dict[str, Any]] = []
    generalization_rows: list[dict[str, Any]] = []
    for decoder_run in range(int(n_decoder_runs)):
        run_seed = int(random_state) + decoder_run
        train_key = ("correct_rewarded", "post_choice")
        train_bins = extracted_bin_inputs.get(train_key)
        if train_bins is None:
            classifier = None
            training_result = _failed_decode_result(
                "missing_training_bins",
                label="correct_rewarded_post_choice",
                training_condition="correct_rewarded",
                training_window="post_choice",
            )
        else:
            classifier, training_metrics = train_single_decoder_with_shuffle_null(
                binned_spikes=train_bins[0],
                target_values=train_bins[1],
                n_shuffles=n_shuffles,
                random_state=run_seed,
                label="correct_rewarded_post_choice",
            )
            training_result = {
                "training_condition": "correct_rewarded",
                "training_window": "post_choice",
                **training_metrics,
            }

        training_rows.append(
            {
                "decoder_run": decoder_run,
                "target": target,
                **training_result,
            }
        )

        for condition in base_conditions:
            for window_name in windows:
                condition_bins = extracted_bin_inputs.get((condition, window_name))
                if condition == "correct_rewarded" and window_name == "post_choice":
                    evaluation_result = {
                        "status": training_result["status"],
                        "reason": training_result["reason"],
                        "label": f"{condition}_{window_name}",
                        "n_samples": training_result.get("n_samples", 0),
                        "n_classes": training_result.get("n_classes", 0),
                        "test_accuracy": training_result.get("test_accuracy", np.nan),
                    }
                elif condition_bins is None:
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
                        "decoder_run": decoder_run,
                        "target": target,
                        "condition": condition,
                        "window": window_name,
                        **evaluation_result,
                    }
                )

    return {
        "training_results": pd.DataFrame(training_rows),
        "generalization_results": pd.DataFrame(generalization_rows),
    }


def build_state_decodability_session_table(
    session: Session,
    region_name: str,
    decodeability_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape state decodeability results into one row per condition with paired pre/post columns.

    Parameters
    ----------
    session : Session
        Session metadata providing the session identifier, mouse, and date strings.
    region_name : str
        Human-readable region label for the decoded units.
    decodeability_results : pd.DataFrame
        Long-form decodeability results from ``run_base_condition_decoding``. Required columns are
        ``condition``, ``window``, ``status``, ``reason``, ``n_samples``, ``n_classes``,
        ``cv_score``, ``cv_pvalue``, ``permutation_score_mean``, and ``permutation_score_std``.

    Returns
    -------
    pd.DataFrame
        Wide per-session state decodeability table with one row per condition and paired
        ``*_pre`` / ``*_post`` columns for the before- and after-choice bins.
    """

    required_columns = {
        "condition",
        "window",
        "status",
        "reason",
        "n_samples",
        "n_classes",
        "cv_score",
        "cv_pvalue",
        "permutation_score_mean",
        "permutation_score_std",
    }
    missing_columns = required_columns - set(decodeability_results.columns)
    if missing_columns:
        raise ValueError(f"decodeability_results is missing required columns: {sorted(missing_columns)}")

    row_order = list(dict.fromkeys(decodeability_results["condition"].tolist()))
    session_rows: list[dict[str, Any]] = []
    for condition_name in row_order:
        condition_rows = decodeability_results.loc[decodeability_results["condition"] == condition_name]
        pre_row = condition_rows.loc[condition_rows["window"] == "pre_choice"]
        post_row = condition_rows.loc[condition_rows["window"] == "post_choice"]
        if pre_row.shape[0] != 1 or post_row.shape[0] != 1:
            raise ValueError(f"Condition {condition_name!r} must have exactly one pre_choice and one post_choice row.")

        pre_result = pre_row.iloc[0]
        post_result = post_row.iloc[0]
        session_rows.append(
            {
                "session_id": session.sess_id_full,
                "mouse": session.mouse,
                "date": session.date,
                "region": region_name,
                "condition": condition_name,
                "status_pre": pre_result["status"],
                "reason_pre": pre_result["reason"],
                "n_trials_pre": pre_result["n_samples"],
                "n_classes_pre": pre_result["n_classes"],
                "cv_score_pre": pre_result["cv_score"],
                "p_value_pre": pre_result["cv_pvalue"],
                "score_mean_pre": pre_result["permutation_score_mean"],
                "score_std_pre": pre_result["permutation_score_std"],
                "status_post": post_result["status"],
                "reason_post": post_result["reason"],
                "n_trials_post": post_result["n_samples"],
                "n_classes_post": post_result["n_classes"],
                "cv_score_post": post_result["cv_score"],
                "p_value_post": post_result["cv_pvalue"],
                "score_mean_post": post_result["permutation_score_mean"],
                "score_std_post": post_result["permutation_score_std"],
            }
        )

    return pd.DataFrame(session_rows)


def save_state_decodability_session_csv(
    output_dir: Path | str,
    state_decodability_table: pd.DataFrame,
    region_name: str,
) -> Path:
    """
    Save the per-session state decodeability table to a region-specific CSV.

    Parameters
    ----------
    output_dir : Path | str
        Directory where the CSV should be written.
    state_decodability_table : pd.DataFrame
        Wide per-session state decodeability table with one row per condition.
    region_name : str
        Brain-region label used to namespace the saved CSV filename.

    Returns
    -------
    Path
        Saved CSV path.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / make_region_analysis_filename(region_name, "state_decodability_analysis.csv")
    state_decodability_table.to_csv(csv_path, index=False)
    return csv_path


def build_correct_rewarded_decoding_performance_session_table(
    session: Session,
    region_name: str,
    training_results: pd.DataFrame,
    generalization_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape repeated decoder outputs into one row per decoder run with all condition results.

    Parameters
    ----------
    session : Session
        Session metadata providing the session identifier, mouse, and date strings.
    region_name : str
        Human-readable region label for the decoded units.
    training_results : pd.DataFrame
        One row per decoder run with training metrics.
    generalization_results : pd.DataFrame
        One row per ``decoder_run x condition x window`` with evaluation metrics.

    Returns
    -------
    pd.DataFrame
        Wide per-session decoder-performance table with one row per decoder run and paired
        before/after columns for each base condition.
    """

    required_training_columns = {
        "decoder_run",
        "training_condition",
        "training_window",
        "train_accuracy",
        "test_accuracy",
        "shuffle_pvalue",
        "shuffle_accuracy_mean",
        "shuffle_accuracy_std",
    }
    missing_training = required_training_columns - set(training_results.columns)
    if missing_training:
        raise ValueError(f"training_results is missing required columns: {sorted(missing_training)}")

    required_generalization_columns = {
        "decoder_run",
        "condition",
        "window",
        "status",
        "reason",
        "test_accuracy",
        "n_samples",
        "n_classes",
    }
    missing_generalization = required_generalization_columns - set(generalization_results.columns)
    if missing_generalization:
        raise ValueError(
            f"generalization_results is missing required columns: {sorted(missing_generalization)}"
        )

    condition_order = list(dict.fromkeys(generalization_results["condition"].tolist()))
    decoder_rows: list[dict[str, Any]] = []
    for _, training_row in training_results.sort_values("decoder_run").iterrows():
        decoder_run = int(training_row["decoder_run"])
        row = {
            "session_id": session.sess_id_full,
            "mouse": session.mouse,
            "date": session.date,
            "region": region_name,
            "decoder_run": decoder_run,
            "training_condition": training_row["training_condition"],
            "training_window": training_row["training_window"],
            "train_accuracy": training_row.get("train_accuracy", np.nan),
            "heldout_test_accuracy": training_row.get("test_accuracy", np.nan),
            "shuffle_pvalue": training_row.get("shuffle_pvalue", np.nan),
            "shuffle_accuracy_mean": training_row.get("shuffle_accuracy_mean", np.nan),
            "shuffle_accuracy_std": training_row.get("shuffle_accuracy_std", np.nan),
        }
        decoder_generalization = generalization_results.loc[
            generalization_results["decoder_run"] == decoder_run
        ]
        for condition_name in condition_order:
            for window_name, suffix in (("pre_choice", "pre"), ("post_choice", "post")):
                matching_rows = decoder_generalization.loc[
                    (decoder_generalization["condition"] == condition_name)
                    & (decoder_generalization["window"] == window_name)
                ]
                if matching_rows.shape[0] != 1:
                    raise ValueError(
                        f"Decoder run {decoder_run} condition {condition_name!r} must have exactly one {window_name} row."
                    )
                result_row = matching_rows.iloc[0]
                row[f"{condition_name}_status_{suffix}"] = result_row["status"]
                row[f"{condition_name}_reason_{suffix}"] = result_row["reason"]
                row[f"{condition_name}_test_accuracy_{suffix}"] = result_row["test_accuracy"]
                row[f"{condition_name}_n_trials_{suffix}"] = result_row["n_samples"]
                row[f"{condition_name}_n_classes_{suffix}"] = result_row["n_classes"]
        decoder_rows.append(row)

    return pd.DataFrame(decoder_rows)


def save_correct_rewarded_decoding_performance_session_csv(
    output_dir: Path | str,
    decoding_performance_table: pd.DataFrame,
    region_name: str,
) -> Path:
    """
    Save the repeated decoder-performance table to a region-specific CSV.

    Parameters
    ----------
    output_dir : Path | str
        Directory where the CSV should be written.
    decoding_performance_table : pd.DataFrame
        Wide per-session decoder-performance table with one row per decoder run.
    region_name : str
        Brain-region label used to namespace the saved CSV filename.

    Returns
    -------
    Path
        Saved CSV path.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / make_region_analysis_filename(region_name, "correct_rewarded_decoding_performance.csv")
    decoding_performance_table.to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    """
    Run one hard-coded session example for user-defined region spike binning.

    The script loads processed behavior tables, loads aligned spike times plus sorter cluster
    metadata for the current example HPC probe, builds a Pynapple spike group, bins
    region-selected spikes by trial, and prints a short summary. Times are handled in seconds throughout.
    """

    multi_session_save_path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis")
    session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference")
    sess_id_full = "CT014_2025-12-23_163505"
    pfc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/Kilosort2.5.2_2026-03-19_180103/sorting_mchin_20260330"
    hpc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/Kilosort2.5.2_2026-03-19_183540/sorting_mchin_20260331"

    raw_behavior_folder = session_data_home / "rpi" / sess_id_full
    processed_data_path = session_data_home / "processed"
    figure_path = session_data_home / "figures"

    aligned_spike_path = session_data_home / 'ephys/aligned/aligned_imec'
    aligned_pfc_spike_path = aligned_spike_path / 'imec0_sync.npz'
    aligned_hpc_spike_path = aligned_spike_path / 'imec1_sync.npz'

    probe_json_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/probe_json.json"

    if not hpc_spike_path.exists():
        raise FileNotFoundError(f"HPC sorter output not found at {hpc_spike_path}")
    assert probe_json_path.exists(), f"Probe JSON file not found at {probe_json_path}"
    assert aligned_pfc_spike_path.exists(), f"Aligned PFC spike file not found at {aligned_pfc_spike_path}"
    assert aligned_hpc_spike_path.exists(), f"Aligned HPC spike file not found at {aligned_hpc_spike_path}"

    decode_target = "action"  # state_int or action
    n_decoder_runs = 5
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

    _pfc_channels = ("55  54  53  52  51  50  49  48 383 382 381 380 379 378 377 376 375 374 "
                    "373 372 371 370 369 368 367 366 365 364 363 362 361 360 359 358 357 356 "
                    "355 354 353 352 351 350 349 348 347 346 345 344 343 342 341 340 339 338 "
                    "337 336 287 286 285 284 283 282 281 280 279 278 277 276 275 274 273 272 "
                    "271 270 269 268 267 266 265 264 263 262 261 260 259 258 257 256 255 254 "
                    "253 252 251 250 249 248 247 246 245 244 243 242 241 240 335 334 333 332 "
                    "331 330 329 328 327 326 325 324 323 322 321 320 319 318 317 316 315 314 "
                    "313 312 311 310 309 308 307 306 305 304 303 302 301 300 299 298 297 296 "
                    "295 294 293 292 291 290 289 288 239 238 237 236 235 234 233 232 231 230 "
                    "229 228 227 226 225 224 223 222 221 220 219 218 217 216 215 214 213 212 "
                    "211 210 209 208 207 206 205 204 203 202 201 200 199 198 197 196 195 194 "
                    "193 192 143 142 141 140 139 138 137 136 135 134 133 132 131 130 129 128 "
                    "127 126 125 124 123 122 121 120 119 118 117 116 115 114 113 112 111 110 "
                    "109 108 107 106 105 104 103 102 101 100  99  98  97  96  47  46  45  44 "
                    "43  42  41  40  39  38  37  36  35  34  33  32  31  30  29  28  27  26 "
                    "25  24  23  22  21  20  19  18  17  16  15  14  13  12  11  10 9  8 "
                    "7   6   5   4   3   2   1   0")
    pfc_channels = np.sort(np.array(list(map(int, [val for val in _pfc_channels.split(" ") if val != '']))))

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
        session_info=load_session_info(session_info_path)
        if session_info_path.exists()
        else None,
        pfc_spike_path=pfc_spike_path,
        hpc_spike_path=hpc_spike_path,
    )

    region_name = "PFC"
    if region_name in SUPPORTED_ANALYSIS_REGIONS:
        if region_name == "HPC":
            region_channels = normalize_region_channels(hpc_channels)
            sorter_output_path = session.hpc_spike_path
            spike_path = aligned_hpc_spike_path
        elif region_name == "V1":
            region_channels = normalize_region_channels(v1_channels)
            sorter_output_path = session.hpc_spike_path
            spike_path = aligned_hpc_spike_path
        elif region_name == "PFC":
            region_channels = normalize_region_channels(pfc_channels)
            sorter_output_path = session.pfc_spike_path
            spike_path = aligned_pfc_spike_path
    else:
        exit("Region not supported")

    event_df, trial_df = load_session_tables(session)
    lick_times = build_lick_time_dict(event_df)
    aligned_spike_times = load_aligned_spikes(spike_path)
    spike_clusters, cluster_info = load_sorter_metadata(
        sorter_output_path=sorter_output_path,
    )
    validate_aligned_spike_inputs(
        aligned_spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
    )

    region_cluster_ids = select_units_by_channels(cluster_info, region_channels=region_channels)
    if region_cluster_ids.size == 0:
        raise RuntimeError(
            f"No {region_name} units were found in the configured sorter output for channels {region_channels.tolist()}."
        )

    region_spike_group = build_spike_tsgroup(
        spike_times=aligned_spike_times,
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
        random_state=42,
    )
    repeated_decoding_csv_path: Path | None = processed_data_path
    if decode_target == "state_int":
        repeated_decoding_results = run_repeated_correct_rewarded_decoder(
            region_trial_binned=region_spike_bins,
            trial_df=trial_df,
            target=decode_target,
            n_decoder_runs=n_decoder_runs,
            n_shuffles=1000,
            random_state=42,
        )
        repeated_decoding_table = build_correct_rewarded_decoding_performance_session_table(
            session=session,
            region_name=region_name,
            training_results=repeated_decoding_results["training_results"],
            generalization_results=repeated_decoding_results["generalization_results"],
        )
        repeated_decoding_csv_path = save_correct_rewarded_decoding_performance_session_csv(
            output_dir=session.processed_data_path,
            decoding_performance_table=repeated_decoding_table,
            region_name=region_name,
        )
    state_decodability_csv_path: Path | None = processed_data_path
    if decode_target == "state_int":
        state_decodability_table = build_state_decodability_session_table(
            session=session,
            region_name=region_name,
            decodeability_results=decoding_results["decodeability_results"],
        )
        state_decodability_csv_path = save_state_decodability_session_csv(
            output_dir=session.processed_data_path,
            state_decodability_table=state_decodability_table,
            region_name=region_name,
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
    if state_decodability_csv_path is not None:
        print(f"State decodability CSV: {state_decodability_csv_path}")
    if repeated_decoding_csv_path is not None:
        print(f"Repeated decoder performance CSV: {repeated_decoding_csv_path}")
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
            "  trained on correct_rewarded post_choice: "
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

    plot_trial_ix: int | None = None
    plot_event = "choice_time"
    plot_time_mode = "session"  # utc, event, or session
    plot_pre_time = 2.0
    plot_post_time = 2.0
    plot_session_start_time: float | None = trial_df['trial_time_since_start'].min() if plot_time_mode == "session" else None
    if plot_trial_ix is not None:
        plot_trial_raster(
            region_spike_group=region_spike_group,
            trial_df=trial_df,
            trial_ix=plot_trial_ix,
            lick_times=lick_times,
            event=plot_event,
            time_mode=plot_time_mode,
            session_start_time=plot_session_start_time,
            pre_time=plot_pre_time,
            post_time=plot_post_time,
        )


if __name__ == "__main__":
    main()
