from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis import spike_behavior_pynapple as sbp


DEFAULT_SESSION_DATA_HOME = Path(
    "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference"
)
DEFAULT_SESSION_ID = "CT014_2025-12-23_163505"
DEFAULT_HPC_SORTER_OUTPUT_PATH = (
    DEFAULT_SESSION_DATA_HOME
    / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/Kilosort2.5.2_2026-03-19_183540/sorting_mchin_20260331"
)
DEFAULT_ALIGNED_SPIKE_PATH = DEFAULT_SESSION_DATA_HOME / "ephys" / "imec1_sync.npz"


def get_ct014_region_channel_presets() -> dict[str, np.ndarray]:
    """
    Return CT014-specific channel presets for the initial webapp workflow.

    Parameters
    ----------
    None
        This helper takes no inputs.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from region label to one-dimensional channel arrays. Channel
        values are integer channel ids with no unit conversion. These presets
        are intended as editable defaults, not universal atlas definitions.
    """

    hpc_channels_shank0 = np.array(
        [
            4, 3, 2, 1, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181,
            180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167,
            166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153,
            152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141,
        ],
        dtype=int,
    )
    hpc_channels_shank3 = np.array(
        [
            251, 252, 253, 254, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16,
        ],
        dtype=int,
    )
    v1_channels_shank0 = np.array(
        [
            138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125,
            124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
            110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96,
            95, 94, 93, 92, 91, 90, 89, 88,
        ],
        dtype=int,
    )
    v1_channels_shank3 = np.array(
        [
            259, 260, 261, 262, 263, 132, 131, 130, 129, 128, 127, 126, 125, 124,
            123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110,
            109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95,
            94, 93, 92, 91, 90, 89, 88, 260, 259, 258, 257, 256,
        ],
        dtype=int,
    )
    return {
        "HPC": np.concatenate([hpc_channels_shank0, hpc_channels_shank3]).astype(int),
        "V1": np.concatenate([v1_channels_shank0, v1_channels_shank3]).astype(int),
        "Custom": np.array([], dtype=int),
    }


def parse_channel_list(channel_text: str) -> np.ndarray:
    """
    Parse editable channel text into integer channel ids.

    Parameters
    ----------
    channel_text : str
        Text containing integer channel ids separated by commas, whitespace, or
        newlines. Units are channel ids, not physical distance.

    Returns
    -------
    np.ndarray
        One-dimensional integer array with shape ``(n_channels,)``.
    """

    normalized_text = str(channel_text).replace(",", " ")
    tokens = [token for token in normalized_text.split() if token]
    if not tokens:
        return np.array([], dtype=int)
    try:
        return np.asarray([int(token) for token in tokens], dtype=int)
    except ValueError as error:
        raise ValueError("Channel list must contain only integer channel ids.") from error


def format_channel_list(channels: np.ndarray | list[int]) -> str:
    """
    Format channel ids for display in an editable text area.

    Parameters
    ----------
    channels : np.ndarray | list[int]
        One-dimensional channel ids with shape ``(n_channels,)``.

    Returns
    -------
    str
        Comma-separated channel list.
    """

    normalized_channels = np.asarray(channels, dtype=int).reshape(-1)
    return ", ".join(str(channel) for channel in normalized_channels)


def normalize_quality_labels(quality_values: pd.Series, default_group: str = "mua") -> pd.Series:
    """
    Normalize sorter quality labels for filtering and display.

    Parameters
    ----------
    quality_values : pd.Series
        One-dimensional cluster-quality labels with shape ``(n_clusters,)``.
    default_group : str, default="mua"
        Label used when the sorter table contains a string ``"nan"`` group.

    Returns
    -------
    pd.Series
        Lowercase quality labels aligned to ``quality_values``.
    """

    quality_labels = quality_values.astype(str).str.strip().str.lower()
    quality_labels.loc[quality_labels == "nan"] = default_group
    return quality_labels


def filter_cluster_metadata(
    cluster_info: pd.DataFrame,
    region_channels: np.ndarray | list[int],
    quality_labels: tuple[str, ...] | list[str] | None = ("good", "mua"),
    default_group: str = "mua",
) -> pd.DataFrame:
    """
    Select unit metadata by channel and optional quality label.

    Parameters
    ----------
    cluster_info : pd.DataFrame
        Sorter cluster table with columns ``cluster_id``, ``ch``, and ``group``.
        Rows correspond to clusters; channel ids are integer sorter channels.
    region_channels : np.ndarray | list[int]
        One-dimensional channel ids with shape ``(n_channels,)`` used to select
        units in the active region.
    quality_labels : tuple[str, ...] | list[str] | None, default=("good", "mua")
        Quality labels to include. ``None`` disables quality filtering.
    default_group : str, default="mua"
        Label used when the sorter table contains a string ``"nan"`` group.

    Returns
    -------
    pd.DataFrame
        Filtered cluster metadata sorted by ``cluster_id``. Includes an added
        ``quality_label`` column with normalized quality strings.
    """

    required_columns = {"cluster_id", "ch", "group"}
    missing_columns = required_columns - set(cluster_info.columns)
    if missing_columns:
        raise ValueError(f"cluster_info is missing required columns: {sorted(missing_columns)}")

    normalized_channels = np.asarray(region_channels, dtype=int).reshape(-1)
    cluster_metadata = cluster_info.copy()
    cluster_metadata["quality_label"] = normalize_quality_labels(
        cluster_metadata["group"],
        default_group=default_group,
    )
    selected_mask = cluster_metadata["ch"].isin(normalized_channels)
    if quality_labels is not None:
        normalized_quality = {str(label).strip().lower() for label in quality_labels}
        selected_mask = selected_mask & cluster_metadata["quality_label"].isin(normalized_quality)
    return cluster_metadata.loc[selected_mask].sort_values("cluster_id").reset_index(drop=True)


def build_session_from_inputs(
    session_data_home: Path | str,
    sess_id_full: str,
    sorter_output_path: Path | str,
    aligned_spike_path: Path | str,
) -> sbp.Session:
    """
    Build session metadata from webapp path inputs.

    Parameters
    ----------
    session_data_home : Path | str
        Root directory for one session.
    sess_id_full : str
        Session id formatted as ``mouse_YYYY-MM-DD_hhmmss``.
    sorter_output_path : Path | str
        Active sorter-output directory.
    aligned_spike_path : Path | str
        Active aligned-spike ``.npz`` file.

    Returns
    -------
    sbp.Session
        Session metadata object with standard behavior-analysis paths.
    """

    session_home = Path(session_data_home)
    mouse, date, timestamp = sbp.parse_session_id(sess_id_full)
    session = sbp.Session(
        session_data_home=session_home,
        sess_id_full=sess_id_full,
        sess_id_abbreviated=f"{mouse}_{date}",
        raw_behavior_folder=session_home / "rpi" / sess_id_full,
        processed_data_path=session_home / "processed",
        figure_path=session_home / "figures",
        mouse=mouse,
        date=date,
        timestamp=timestamp,
        hpc_spike_path=Path(sorter_output_path),
    )
    session.session_info_path = session.raw_behavior_folder / f"{sess_id_full}_session_info.pkl"
    session.session_info = {"aligned_spike_path": str(aligned_spike_path)}
    return session


def load_viewer_data(
    session_data_home: Path | str,
    sess_id_full: str,
    sorter_output_path: Path | str,
    aligned_spike_path: Path | str,
) -> dict[str, Any]:
    """
    Load one session's behavior tables, sorter metadata, and spike group.

    Parameters
    ----------
    session_data_home : Path | str
        Root directory for one session.
    sess_id_full : str
        Session id formatted as ``mouse_YYYY-MM-DD_hhmmss``.
    sorter_output_path : Path | str
        Sorter output directory containing ``spike_clusters.npy`` and
        ``cluster_info.tsv``.
    aligned_spike_path : Path | str
        Aligned spike ``.npz`` file containing ``spike_utc_unix`` in seconds.

    Returns
    -------
    dict[str, Any]
        Loaded objects with keys ``session``, ``event_df``, ``trial_df``,
        ``cluster_info``, ``spike_group``, and ``aligned_spike_times``.
    """

    session = build_session_from_inputs(
        session_data_home=session_data_home,
        sess_id_full=sess_id_full,
        sorter_output_path=sorter_output_path,
        aligned_spike_path=aligned_spike_path,
    )
    event_df, trial_df = sbp.load_session_tables(session)
    spike_clusters, cluster_info = sbp.load_sorter_metadata(Path(sorter_output_path))
    aligned_spike_times = sbp.load_aligned_spikes(Path(aligned_spike_path))
    sbp.validate_aligned_spike_inputs(aligned_spike_times, spike_clusters)
    spike_group = sbp.build_spike_tsgroup(
        spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
    )
    return {
        "session": session,
        "event_df": event_df,
        "trial_df": trial_df,
        "cluster_info": cluster_info,
        "spike_group": spike_group,
        "aligned_spike_times": aligned_spike_times,
    }


def get_unit_spike_times(spike_group: nap.TsGroup, unit_id: int) -> np.ndarray:
    """
    Extract one unit's spike times from a Pynapple spike group.

    Parameters
    ----------
    spike_group : nap.TsGroup
        Pynapple spike group keyed by cluster id. Times are in seconds.
    unit_id : int
        Cluster id to extract.

    Returns
    -------
    np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    """

    if int(unit_id) not in spike_group:
        raise ValueError(f"Unit {unit_id} is not present in the spike group.")
    return np.asarray(spike_group[int(unit_id)].index.to_numpy(), dtype=float)
