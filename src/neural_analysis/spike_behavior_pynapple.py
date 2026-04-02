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

import src.external_tools.readSGLX as readSGLX


HPC_CHANNEL_START = 192
HPC_CHANNEL_STOP = 240


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


def select_hpc_units(cluster_info: pd.DataFrame) -> np.ndarray:
    """
    Select cluster ids assigned to the legacy HPC channel range.

    Parameters
    ----------
    cluster_info : pd.DataFrame
        Cluster metadata table with columns ``cluster_id`` and ``ch``. ``ch`` is an integer
        channel index with no physical-unit conversion.

    Returns
    -------
    np.ndarray
        One-dimensional integer array with shape ``(n_hpc_units,)`` containing cluster ids
        whose main channel lies in the inclusive range 192-239.
    """

    required_columns = {"cluster_id", "ch"}
    missing_columns = required_columns - set(cluster_info.columns)
    if missing_columns:
        raise ValueError(f"cluster_info is missing required columns: {sorted(missing_columns)}")

    hpc_mask = cluster_info["ch"].between(HPC_CHANNEL_START, HPC_CHANNEL_STOP - 1)
    return cluster_info.loc[hpc_mask, "cluster_id"].to_numpy(dtype=int)


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


def bin_hpc_trials(
    trial_df: pd.DataFrame,
    spike_group: nap.TsGroup,
    cluster_ids: np.ndarray,
    bin_size: float = 0.5,
    pre_time: float = 2.0,
    post_time: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Bin HPC spikes for every trial while preserving the legacy output structure.

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


def main() -> None:
    """
    Run one hard-coded session example for HPC spike binning.

    The script loads processed behavior tables, loads sorter spikes from the current
    example HPC probe, builds a Pynapple spike group, bins HPC spikes by trial, and
    prints a short summary. Times are handled in seconds throughout.
    """

    multi_session_save_path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis")
    session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference")
    sess_id_full = "CT014_2025-12-16_153200"
    raw_behavior_folder = session_data_home / "rpi" / sess_id_full
    processed_data_path = session_data_home / "processed"
    figure_path = session_data_home / "figures"

    pfc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/Kilosort2.5.2_2026-03-18_115111"
    hpc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/Kilosort2.5.2_2026-03-18_122341"
    hpc_ap_bin_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_tcat.imec1.ap.bin"

    if not hpc_spike_path.exists():
        raise FileNotFoundError(f"HPC sorter output not found at {hpc_spike_path}")
    if not hpc_ap_bin_path.exists():
        raise FileNotFoundError(f"HPC AP binary not found at {hpc_ap_bin_path}")

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

    hpc_cluster_ids = select_hpc_units(cluster_info)
    if hpc_cluster_ids.size == 0:
        raise RuntimeError(
            "No HPC units were found in the configured sorter output using the legacy channel range 192-239."
        )

    hpc_spike_group = build_spike_tsgroup(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        cluster_ids=hpc_cluster_ids,
    )
    hpc_spike_bins = bin_hpc_trials(
        trial_df=trial_df,
        spike_group=hpc_spike_group,
        cluster_ids=hpc_cluster_ids,
        bin_size=0.5,
        pre_time=2.0,
        post_time=2.0,
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
    print(f"HPC units: {hpc_cluster_ids.size}")
    print(f"Trials binned: {len(hpc_spike_bins)}")
    print(f"First trial spike-bin shape: {hpc_spike_bins[0]['binned_spikes'].shape}")
    print(f"First trial lick-bin shape: {first_trial_licks.shape}")


if __name__ == "__main__":
    main()
