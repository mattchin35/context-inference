from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
DEFAULT_PFC_SORTER_OUTPUT_PATH = (
    DEFAULT_SESSION_DATA_HOME
    / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/Kilosort2.5.2_2026-03-19_180103/sorting_mchin_20260330"
)
DEFAULT_ALIGNED_SPIKE_DIR = DEFAULT_SESSION_DATA_HOME / "ephys" / "aligned" / "aligned_imec"
DEFAULT_HPC_V1_ALIGNED_SPIKE_PATH = DEFAULT_ALIGNED_SPIKE_DIR / "imec1_sync.npz"
DEFAULT_PFC_ALIGNED_SPIKE_PATH = DEFAULT_ALIGNED_SPIKE_DIR / "imec0_sync.npz"
DEFAULT_HPC_V1_LFP_PATH = DEFAULT_SESSION_DATA_HOME / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/run0_g0_t0.imec1.lf.bin"
DEFAULT_PFC_LFP_PATH = DEFAULT_SESSION_DATA_HOME / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/run0_g0_t0.imec0.lf.bin"
PROBE_LABEL_HPC_V1 = "HPC/V1"
PROBE_LABEL_PFC = "PFC"
SUPPORTED_PROBE_LABELS = (PROBE_LABEL_HPC_V1, PROBE_LABEL_PFC)
REGION_TO_PROBE_LABEL = {
    "HPC": PROBE_LABEL_HPC_V1,
    "V1": PROBE_LABEL_HPC_V1,
    "PFC": PROBE_LABEL_PFC,
}


@dataclass(frozen=True)
class ProbeDataPaths:
    """
    Filesystem inputs for one probe data source.

    Parameters
    ----------
    label : str
        Human-readable probe label, such as ``"HPC/V1"`` or ``"PFC"``.
    sorter_output_path : Path | str | None
        Sorter output directory containing ``spike_clusters.npy`` and
        ``cluster_info.tsv``. Units are filesystem path coordinates.
    aligned_spike_path : Path | str | None
        Aligned spike ``.npz`` file containing ``spike_utc_unix`` in seconds.
    lfp_path : Path | str | None
        LFP ``.lf.bin`` file path. Optional for spike-only plots.
    """

    label: str
    sorter_output_path: Path | str | None
    aligned_spike_path: Path | str | None
    lfp_path: Path | str | None

    def __post_init__(self) -> None:
        """Normalize nonempty path strings to ``Path`` objects."""

        object.__setattr__(self, "sorter_output_path", _normalize_optional_path(self.sorter_output_path))
        object.__setattr__(self, "aligned_spike_path", _normalize_optional_path(self.aligned_spike_path))
        object.__setattr__(self, "lfp_path", _normalize_optional_path(self.lfp_path))


def _normalize_optional_path(path_value: Path | str | None) -> Path | None:
    """
    Normalize optional path input from text boxes or tests.

    Parameters
    ----------
    path_value : Path | str | None
        Filesystem path value. Empty strings are treated as missing values.

    Returns
    -------
    Path | None
        Expanded ``Path`` object for nonempty inputs, otherwise ``None``.
    """

    if path_value is None:
        return None
    if str(path_value).strip() == "":
        return None
    return Path(path_value).expanduser()


def list_path_browser_entries(
    current_path: Path | str,
    file_suffix: str | None = None,
) -> dict[str, list[Path]]:
    """
    List one directory for the Streamlit server-side path browser.

    Parameters
    ----------
    current_path : Path | str
        Directory to list on the server filesystem.
    file_suffix : str | None, optional
        Optional file suffix filter, such as ``".npz"``. ``None`` returns all
        non-hidden files.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary with sorted ``directories`` and ``files`` path lists.
    """

    directory_path = Path(current_path).expanduser()
    if not directory_path.exists():
        raise FileNotFoundError(f"Path browser directory does not exist: {directory_path}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path browser path is not a directory: {directory_path}")

    directories: list[Path] = []
    files: list[Path] = []
    for child_path in directory_path.iterdir():
        if child_path.name.startswith("."):
            continue
        if child_path.is_dir():
            directories.append(child_path)
        elif child_path.is_file() and (file_suffix is None or child_path.name.endswith(file_suffix)):
            files.append(child_path)

    return {
        "directories": sorted(directories, key=lambda path: path.name.lower()),
        "files": sorted(files, key=lambda path: path.name.lower()),
    }


def resolve_channel_quality_path(channel_quality_path_or_probe_dir: Path | str) -> Path:
    """
    Resolve a channel-quality file from either a file path or probe directory.

    Parameters
    ----------
    channel_quality_path_or_probe_dir : Path | str
        Either a direct path to ``channel_quality.csv`` or
        ``channel_quality.json``, or a probe-derived directory containing one
        of those files. Units: filesystem path.

    Returns
    -------
    Path
        Existing channel-quality file path. CSV is preferred over JSON when a
        directory contains both files.
    """

    input_path = Path(channel_quality_path_or_probe_dir).expanduser()
    if input_path.is_dir():
        for filename in ("channel_quality.csv", "channel_quality.json"):
            candidate_path = input_path / filename
            if candidate_path.exists():
                return candidate_path
        raise FileNotFoundError(f"No channel_quality.csv or channel_quality.json found in {input_path}.")
    if input_path.exists():
        if input_path.name not in {"channel_quality.csv", "channel_quality.json"}:
            raise ValueError(f"Unsupported channel-quality filename: {input_path.name}")
        return input_path
    raise FileNotFoundError(f"Channel-quality path does not exist: {input_path}")


def _coerce_bool_series(values: pd.Series, column_name: str) -> pd.Series:
    """
    Convert a mixed-type boolean-like column to real booleans.

    Parameters
    ----------
    values : pd.Series
        One-dimensional column with shape ``(n_channels,)``. Values may be
        booleans, 0/1 integers, or true/false strings.
    column_name : str
        Column name used in validation error messages. Units: not applicable.

    Returns
    -------
    pd.Series
        Boolean series with shape ``(n_channels,)`` aligned to ``values``.
    """

    normalized_values = values.astype(str).str.strip().str.lower()
    bool_values = normalized_values.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    if bool_values.isna().any():
        invalid_values = sorted(normalized_values.loc[bool_values.isna()].unique().tolist())
        raise ValueError(f"Column {column_name!r} contains non-boolean values: {invalid_values}")
    return bool_values.astype(bool)


def _normalize_channel_quality_dataframe(channel_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw channel-quality rows to the webapp channel contract.

    Parameters
    ----------
    channel_quality : pd.DataFrame
        Raw channel-quality table with shape ``(n_channels, n_columns)``.
        Required columns are ``channel_id``, ``label``, ``inside_brain``,
        ``x_um``, and ``y_um``. ``is_good`` is optional and inferred from
        ``label == "good"`` when absent.

    Returns
    -------
    pd.DataFrame
        Normalized table sorted by ``ch`` with columns ``ch``, ``channel_id``,
        ``label``, ``is_good``, ``inside_brain``, ``x_um``, and ``y_um``.
        ``ch`` is a zero-based integer channel id; coordinates are in
        micrometers.
    """

    required_columns = {"channel_id", "label", "inside_brain", "x_um", "y_um"}
    missing_columns = required_columns - set(channel_quality.columns)
    if missing_columns:
        raise ValueError(f"channel_quality is missing required columns: {sorted(missing_columns)}")

    normalized_quality = channel_quality.copy()
    normalized_quality["channel_id"] = normalized_quality["channel_id"].astype(str).str.strip()
    parsed_channel = normalized_quality["channel_id"].str.extract(r"(\d+)", expand=False)
    if parsed_channel.isna().any():
        invalid_channel_ids = normalized_quality.loc[parsed_channel.isna(), "channel_id"].tolist()
        raise ValueError(f"Could not parse integer channel ids from channel_quality: {invalid_channel_ids}")
    normalized_quality["ch"] = parsed_channel.astype(int)
    if normalized_quality["ch"].duplicated().any():
        duplicate_channels = sorted(normalized_quality.loc[normalized_quality["ch"].duplicated(), "ch"].unique().tolist())
        raise ValueError(f"Duplicate channel ids in channel_quality: {duplicate_channels}")

    normalized_quality["label"] = normalized_quality["label"].astype(str).str.strip().str.lower()
    if "is_good" in normalized_quality.columns:
        normalized_quality["is_good"] = _coerce_bool_series(normalized_quality["is_good"], "is_good")
    else:
        normalized_quality["is_good"] = normalized_quality["label"].eq("good")
    normalized_quality["inside_brain"] = _coerce_bool_series(normalized_quality["inside_brain"], "inside_brain")
    normalized_quality["x_um"] = pd.to_numeric(normalized_quality["x_um"], errors="raise").astype(float)
    normalized_quality["y_um"] = pd.to_numeric(normalized_quality["y_um"], errors="raise").astype(float)

    output_columns = ["ch", "channel_id", "label", "is_good", "inside_brain", "x_um", "y_um"]
    return normalized_quality.loc[:, output_columns].sort_values("ch").reset_index(drop=True)


def load_channel_quality(channel_quality_path_or_probe_dir: Path | str) -> pd.DataFrame:
    """
    Load and normalize channel-quality metadata for one probe.

    Parameters
    ----------
    channel_quality_path_or_probe_dir : Path | str
        Either a direct path to ``channel_quality.csv`` or
        ``channel_quality.json``, or a probe-derived directory containing one
        of those files. Units: filesystem path.

    Returns
    -------
    pd.DataFrame
        Normalized channel-quality dataframe with one row per channel. Columns
        are ``ch`` as zero-based channel index, ``channel_id``, ``label``,
        ``is_good``, ``inside_brain``, ``x_um``, and ``y_um``. Coordinates are
        in micrometers.
    """

    channel_quality_path = resolve_channel_quality_path(channel_quality_path_or_probe_dir)
    if channel_quality_path.suffix.lower() == ".csv":
        raw_channel_quality = pd.read_csv(channel_quality_path)
    elif channel_quality_path.suffix.lower() == ".json":
        with channel_quality_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if isinstance(payload, dict) and "channels" in payload:
            raw_channel_quality = pd.DataFrame(payload["channels"])
        elif isinstance(payload, list):
            raw_channel_quality = pd.DataFrame(payload)
        else:
            raw_channel_quality = pd.DataFrame(payload)
    else:
        raise ValueError(f"Unsupported channel-quality file type: {channel_quality_path.suffix}")
    return _normalize_channel_quality_dataframe(raw_channel_quality)


def select_channels_from_quality(
    channel_quality: pd.DataFrame,
    require_inside_brain: bool = True,
    labels: tuple[str, ...] | list[str] | None = ("good",),
) -> np.ndarray:
    """
    Select channel ids from normalized channel-quality metadata.

    Parameters
    ----------
    channel_quality : pd.DataFrame
        Normalized channel-quality table with shape ``(n_channels, n_columns)``.
        Required columns are ``ch``, ``label``, and ``inside_brain``.
    require_inside_brain : bool, default=True
        If ``True``, only channels with ``inside_brain == True`` are retained.
    labels : tuple[str, ...] | list[str] | None, default=("good",)
        Channel labels to retain. Matching is case-insensitive. ``None``
        disables label filtering.

    Returns
    -------
    np.ndarray
        Sorted one-dimensional integer array with shape ``(n_selected_channels,)``
        containing zero-based channel ids.
    """

    required_columns = {"ch", "label", "inside_brain"}
    missing_columns = required_columns - set(channel_quality.columns)
    if missing_columns:
        raise ValueError(f"channel_quality is missing required columns: {sorted(missing_columns)}")

    selected_mask = pd.Series(True, index=channel_quality.index)
    if require_inside_brain:
        selected_mask = selected_mask & _coerce_bool_series(channel_quality["inside_brain"], "inside_brain")
    if labels is not None:
        normalized_labels = {str(label).strip().lower() for label in labels}
        normalized_channel_labels = channel_quality["label"].astype(str).str.strip().str.lower()
        selected_mask = selected_mask & normalized_channel_labels.isin(normalized_labels)
    return np.sort(channel_quality.loc[selected_mask, "ch"].to_numpy(dtype=int))


def infer_probe_derived_dir(
    sorter_output_path: Path | str | None,
    lfp_path: Path | str | None,
) -> Path | None:
    """
    Infer the probe-derived directory from active sorter and LFP paths.

    Parameters
    ----------
    sorter_output_path : Path | str | None
        Sorter output directory, typically ``.../ProbeA/kilosort4``. Units:
        filesystem path.
    lfp_path : Path | str | None
        LFP binary path, typically ``.../ProbeA/lfp.dat``. Units: filesystem
        path.

    Returns
    -------
    Path | None
        Inferred probe-derived directory, such as ``.../ProbeA``. Returns
        ``None`` when neither input path is available.
    """

    sorter_path = _normalize_optional_path(sorter_output_path)
    if sorter_path is not None:
        if sorter_path.name == "kilosort4":
            return sorter_path.parent
        return sorter_path

    normalized_lfp_path = _normalize_optional_path(lfp_path)
    if normalized_lfp_path is not None:
        return normalized_lfp_path.parent
    return None


def get_probe_label_for_region(region_name: str, custom_probe_label: str | None = None) -> str:
    """
    Resolve the probe data source for a selected region.

    Parameters
    ----------
    region_name : str
        Region preset label. Supported named regions are ``"HPC"``, ``"V1"``,
        ``"PFC"``, and ``"Custom"``.
    custom_probe_label : str | None, optional
        Explicit probe source for ``"Custom"`` channel selections. Supported
        values are ``"HPC/V1"`` and ``"PFC"``.

    Returns
    -------
    str
        Canonical probe label, either ``"HPC/V1"`` or ``"PFC"``.
    """

    normalized_region = str(region_name).strip().upper()
    if normalized_region == "CUSTOM":
        if custom_probe_label is None or str(custom_probe_label).strip() == "":
            raise ValueError("Custom region selections require an explicit custom_probe_label.")
        normalized_probe = str(custom_probe_label).strip().upper()
        for supported_probe_label in SUPPORTED_PROBE_LABELS:
            if normalized_probe == supported_probe_label.upper():
                return supported_probe_label
        raise ValueError(f"Unsupported custom probe label {custom_probe_label!r}.")
    if normalized_region not in REGION_TO_PROBE_LABEL:
        raise ValueError(
            f"Unsupported region {region_name!r}. Expected one of "
            f"{sorted([*REGION_TO_PROBE_LABEL, 'Custom'])}."
        )
    return REGION_TO_PROBE_LABEL[normalized_region]


def validate_probe_paths_for_spike_plot(probe_paths: ProbeDataPaths) -> None:
    """
    Validate active probe paths needed for spike raster/PSTH plotting.

    Parameters
    ----------
    probe_paths : ProbeDataPaths
        Active probe paths. ``sorter_output_path`` must be an existing
        directory and ``aligned_spike_path`` must be an existing file.
        ``lfp_path`` is intentionally not required for spike-only plots.

    Returns
    -------
    None
        Raises a clear exception if active spike inputs are missing.
    """

    if probe_paths.sorter_output_path is None:
        raise ValueError(f"{probe_paths.label} sorter output path is required for spike plots.")
    if not probe_paths.sorter_output_path.exists() or not probe_paths.sorter_output_path.is_dir():
        raise FileNotFoundError(f"{probe_paths.label} sorter output path does not exist: {probe_paths.sorter_output_path}")
    if probe_paths.aligned_spike_path is None:
        raise ValueError(f"{probe_paths.label} aligned spike path is required for spike plots.")
    if not probe_paths.aligned_spike_path.exists() or not probe_paths.aligned_spike_path.is_file():
        raise FileNotFoundError(f"{probe_paths.label} aligned spike path does not exist: {probe_paths.aligned_spike_path}")


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
            152, 151, 150, 149, 148, 147, 146, 145, 96, 95, 94, 93, 92, 91, 90, 89,
            88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71,
            70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53,
            52, 51, 50, 49, 383, 382, 381,
        ],
        dtype=int,
    )
    hpc_channels_shank3 = np.array(
        [
            255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242,
            241, 336, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326, 325, 324,
            323, 322, 321, 320, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310,
            309, 308, 307, 306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296,
            295, 294, 293, 292, 291, 290, 289, 240, 239, 238, 237, 236, 235, 234,
            233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220,
            219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206,
            205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 144,
            143, 142, 141,
        ],
        dtype=int,
    )
    v1_channels_shank0 = np.array(
        [
            138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125,
            124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
            110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97,
            48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
            14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
        ],
        dtype=int,
    )
    v1_channels_shank3 = np.array(
        [
            78, 377, 376, 375, 374, 373, 372, 371, 370, 369, 368, 367, 366, 365,
            364, 363, 362, 361, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351,
            350, 349, 348, 347, 346, 345, 344, 343, 342, 341, 340, 339, 338, 337,
            288, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275,
            274, 273, 272, 271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261,
            260, 259, 258, 257, 256,
        ],
        dtype=int,
    )
    pfc_channels = np.array(
        [
            55, 54, 53, 52, 51, 50, 49, 48, 383, 382, 381, 380, 379, 378, 377,
            376, 375, 374, 373, 372, 371, 370, 369, 368, 367, 366, 365, 364,
            363, 362, 361, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351,
            350, 349, 348, 347, 346, 345, 344, 343, 342, 341, 340, 339, 338,
            337, 336, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277,
            276, 275, 274, 273, 272, 271, 270, 269, 268, 267, 266, 265, 264,
            263, 262, 261, 260, 259, 258, 257, 256, 255, 254, 253, 252, 251,
            250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 335, 334,
            333, 332, 331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321,
            320, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310, 309, 308,
            307, 306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296, 295,
            294, 293, 292, 291, 290, 289, 288, 239, 238, 237, 236, 235, 234,
            233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221,
            220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208,
            207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195,
            194, 193, 192, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134,
            133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121,
            120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108,
            107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 47, 46,
            45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30,
            29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
            13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        ],
        dtype=int,
    )
    return {
        "HPC": np.concatenate([hpc_channels_shank0, hpc_channels_shank3]).astype(int),
        "V1": np.concatenate([v1_channels_shank0, v1_channels_shank3]).astype(int),
        "PFC": np.sort(pfc_channels).astype(int),
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

    quality_labels = quality_values.copy()
    quality_labels.loc[quality_labels.isna()] = default_group
    quality_labels = quality_labels.astype(str).str.strip().str.lower()
    missing_sentinels = {"", "nan", "none"}
    quality_labels.loc[quality_labels.isin(missing_sentinels)] = default_group
    return quality_labels


def filter_cluster_metadata(
    cluster_info: pd.DataFrame,
    region_channels: np.ndarray | list[int],
    quality_labels: tuple[str, ...] | list[str] | None = ("good", "mua"),
    default_group: str = "mua",
    quality_column: str = "group",
) -> pd.DataFrame:
    """
    Select unit metadata by channel and optional quality label.

    Parameters
    ----------
    cluster_info : pd.DataFrame
        Sorter cluster table with columns ``cluster_id``, ``ch``, and
        ``quality_column``. Rows correspond to clusters; channel ids are
        integer sorter channels.
    region_channels : np.ndarray | list[int]
        One-dimensional channel ids with shape ``(n_channels,)`` used to select
        units in the active region.
    quality_labels : tuple[str, ...] | list[str] | None, default=("good", "mua")
        Quality labels to include. ``None`` disables quality filtering.
    default_group : str, default="mua"
        Label used when the sorter table contains a string ``"nan"`` group.
    quality_column : str, default="group"
        Name of the cluster-quality column to filter, such as ``"group"`` for
        manual Phy labels or ``"KSLabel"`` for Kilosort labels.

    Returns
    -------
    pd.DataFrame
        Filtered cluster metadata sorted by ``cluster_id``. Includes an added
        ``quality_label`` column with normalized quality strings.
    """

    required_columns = {"cluster_id", "ch", str(quality_column)}
    missing_columns = required_columns - set(cluster_info.columns)
    if missing_columns:
        raise ValueError(f"cluster_info is missing required columns: {sorted(missing_columns)}")

    normalized_channels = np.asarray(region_channels, dtype=int).reshape(-1)
    cluster_metadata = cluster_info.copy()
    cluster_metadata["quality_label"] = normalize_quality_labels(
        cluster_metadata[str(quality_column)],
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


def build_session_from_probe(
    session_data_home: Path | str,
    sess_id_full: str,
    probe_paths: ProbeDataPaths,
) -> sbp.Session:
    """
    Build session metadata using the active probe path bundle.

    Parameters
    ----------
    session_data_home : Path | str
        Root directory for one session.
    sess_id_full : str
        Session id formatted as ``mouse_YYYY-MM-DD_hhmmss``.
    probe_paths : ProbeDataPaths
        Active probe paths. Spike paths are filesystem coordinates; aligned
        spike times loaded from ``aligned_spike_path`` are in seconds.

    Returns
    -------
    sbp.Session
        Session metadata object with behavior paths and the active probe sorter
        path assigned to the corresponding session probe field.
    """

    session_home = Path(session_data_home)
    mouse, date, timestamp = sbp.parse_session_id(sess_id_full)
    hpc_spike_path = probe_paths.sorter_output_path if probe_paths.label == PROBE_LABEL_HPC_V1 else None
    pfc_spike_path = probe_paths.sorter_output_path if probe_paths.label == PROBE_LABEL_PFC else None
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
        pfc_spike_path=pfc_spike_path,
        hpc_spike_path=hpc_spike_path,
    )
    session.session_info_path = session.raw_behavior_folder / f"{sess_id_full}_session_info.pkl"
    session.session_info = {
        "active_probe_label": probe_paths.label,
        "aligned_spike_path": str(probe_paths.aligned_spike_path),
        "lfp_path": str(probe_paths.lfp_path) if probe_paths.lfp_path is not None else None,
    }
    return session


def load_viewer_data_for_probe(
    session_data_home: Path | str,
    sess_id_full: str,
    probe_paths: ProbeDataPaths,
) -> dict[str, Any]:
    """
    Load one session's behavior and active-probe spike data for the webapp.

    Parameters
    ----------
    session_data_home : Path | str
        Root directory for one session.
    sess_id_full : str
        Session id formatted as ``mouse_YYYY-MM-DD_hhmmss``.
    probe_paths : ProbeDataPaths
        Active probe paths. For spike raster/PSTH plots, ``sorter_output_path``
        must be a sorter directory and ``aligned_spike_path`` must be an
        aligned spike ``.npz`` file with spike times in seconds.

    Returns
    -------
    dict[str, Any]
        Loaded objects with keys ``session``, ``event_df``, ``trial_df``,
        ``cluster_info``, ``spike_group``, ``aligned_spike_times``, and
        ``active_probe``.
    """

    validate_probe_paths_for_spike_plot(probe_paths)
    session = build_session_from_probe(
        session_data_home=session_data_home,
        sess_id_full=sess_id_full,
        probe_paths=probe_paths,
    )
    event_df, trial_df = sbp.load_session_tables(session)
    spike_clusters, cluster_info = sbp.load_sorter_metadata(probe_paths.sorter_output_path)
    aligned_spike_times = sbp.load_aligned_spikes(probe_paths.aligned_spike_path)
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
        "active_probe": probe_paths,
    }


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

    probe_paths = ProbeDataPaths(
        label=PROBE_LABEL_HPC_V1,
        sorter_output_path=sorter_output_path,
        aligned_spike_path=aligned_spike_path,
        lfp_path=None,
    )
    return load_viewer_data_for_probe(
        session_data_home=session_data_home,
        sess_id_full=sess_id_full,
        probe_paths=probe_paths,
    )


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
