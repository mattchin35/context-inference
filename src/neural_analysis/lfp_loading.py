from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap

import src.external_tools.readSGLX as readSGLX
from src.neural_analysis import ephys_sync_utils


OPEN_EPHYS_AFFINE_UV_SEMANTICS = "open_ephys_affine_uV_v1"


def load_lfp_metadata(lfp_path: Path | str) -> dict:
    """
    Load SpikeGLX metadata for one LFP binary file.

    Parameters
    ----------
    lfp_path : Path | str
        Path to a SpikeGLX ``.lf.bin`` file. The matching ``.meta`` file must
        live next to it.

    Returns
    -------
    dict
        SpikeGLX metadata dictionary. Values are strings as returned by the
        local ``readSGLX.readMeta`` helper.
    """

    lfp_file = Path(lfp_path)
    meta = readSGLX.readMeta(lfp_file)
    if not meta:
        raise FileNotFoundError(f"Could not load SpikeGLX metadata for {lfp_file}.")
    return meta


def load_open_ephys_lfp_metadata(lfp_path: Path | str) -> dict:
    """Load and fail-closed normalize metadata for one derived Open Ephys LFP binary.

    Parameters
    ----------
    lfp_path : Path | str
        Path to a derived Open Ephys ``lfp.dat`` file. The sibling metadata
        file must be named ``lfp_preprocessing.json``. Units: filesystem path.

    Returns
    -------
    dict
        A normalized JSON-compatible mapping with no numerical trace arrays.
        ``output_binary`` exactly matches the selected filename;
        ``sampling_frequency_hz`` is a finite positive float in Hz;
        ``num_channels`` is a positive saved-channel count; ``num_segments``
        is exactly one; and ``num_samples_by_segment`` is retained as a
        one-entry count list while derived ``num_samples`` is a positive
        sample count. ``dtype`` is exactly ``"float32"`` and
        ``binary_layout`` is exactly ``"time_major_channel_interleaved"``,
        so raw stored values have logical shape ``(sample, saved_channel)``.
        ``channel_ids_in_binary_order`` is a nonempty string list of shape
        ``(num_channels,)``. ``lfp_binary_scaling`` retains the observed
        ``data_units``, ``has_scaleable_traces``, channel-id, gain, offset,
        unit, export-scale, and conversion fields; each gain/offset/unit
        vector has shape ``(num_channels,)``, gains are finite positive
        multipliers to uV, offsets are finite uV, and units are explicitly
        ``"uV"``. No scaling is applied by this metadata loader.

    Raises
    ------
    FileNotFoundError
        If the selected LFP binary or its sibling
        ``lfp_preprocessing.json`` file is absent.
    OSError
        If either file cannot be opened, read, or statted.
    json.JSONDecodeError
        If the sidecar is not valid JSON.
    ValueError
        If mandatory fields, primitive types, literal contracts, channel
        order/vector shapes, physical units, scaling values, or the exact
        binary byte size disagree with the supported one-segment float32
        contract.
    """

    lfp_file = Path(lfp_path)
    if not lfp_file.exists():
        raise FileNotFoundError(f"Open Ephys LFP file was not found: {lfp_file}")
    metadata_path = lfp_file.parent / "lfp_preprocessing.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Open Ephys LFP metadata file was not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)

    required_fields = {
        "output_binary",
        "sampling_frequency_hz",
        "num_channels",
        "num_segments",
        "num_samples_by_segment",
        "dtype",
        "binary_layout",
        "channel_ids_in_binary_order",
        "lfp_binary_scaling",
    }
    missing_fields = required_fields - set(metadata)
    if missing_fields:
        raise ValueError(f"Open Ephys LFP metadata is missing required fields: {sorted(missing_fields)}")
    if metadata["output_binary"] != lfp_file.name:
        raise ValueError("Open Ephys LFP metadata output_binary does not match the selected binary.")
    if metadata["dtype"] != "float32":
        raise ValueError("Open Ephys LFP metadata dtype must be exactly 'float32'.")
    if metadata["binary_layout"] != "time_major_channel_interleaved":
        raise ValueError(
            "Open Ephys LFP metadata binary_layout is incompatible with the output_binary/"
            "num_segments/num_samples_by_segment contract."
        )
    sampling_frequency_hz = _positive_finite_number(
        metadata["sampling_frequency_hz"], "sampling_frequency_hz"
    )
    num_channels = _positive_integer(metadata["num_channels"], "num_channels")
    num_segments = _positive_integer(metadata["num_segments"], "num_segments")
    if num_segments != 1:
        raise ValueError("Open Ephys LFP metadata num_segments must be exactly one.")
    sample_counts_value = metadata["num_samples_by_segment"]
    if not isinstance(sample_counts_value, list) or len(sample_counts_value) != 1:
        raise ValueError("Open Ephys LFP metadata num_samples_by_segment must contain one sample count.")
    sample_counts = [_positive_integer(sample_counts_value[0], "num_samples_by_segment")]
    channel_ids = metadata["channel_ids_in_binary_order"]
    if (
        not isinstance(channel_ids, list)
        or len(channel_ids) != num_channels
        or any(not isinstance(channel_id, str) or not channel_id for channel_id in channel_ids)
    ):
        raise ValueError("Open Ephys LFP metadata channel_ids_in_binary_order is invalid for num_channels.")
    _validate_open_ephys_scaling(metadata["lfp_binary_scaling"], channel_ids, num_channels)
    normalized_metadata = dict(metadata)
    normalized_metadata["sampling_frequency_hz"] = sampling_frequency_hz
    normalized_metadata["num_channels"] = num_channels
    normalized_metadata["num_segments"] = num_segments
    normalized_metadata["num_samples"] = int(sample_counts[0])
    normalized_metadata["dtype"] = "float32"
    normalized_metadata["binary_layout"] = "time_major_channel_interleaved"

    dtype = np.dtype(normalized_metadata["dtype"])
    expected_bytes = normalized_metadata["num_samples"] * normalized_metadata["num_channels"] * dtype.itemsize
    if lfp_file.exists() and lfp_file.stat().st_size != expected_bytes:
        raise ValueError(
            f"Open Ephys LFP file size does not match metadata: expected {expected_bytes} bytes, "
            f"found {lfp_file.stat().st_size} bytes."
        )
    return normalized_metadata


def _positive_finite_number(value: object, field_name: str) -> float:
    """Validate one positive finite JSON physical scalar.

    Parameters
    ----------
    value : object
        JSON-decoded candidate numeric scalar. Booleans, strings, arrays,
        nonfinite values, zero, and negative values are invalid.
    field_name : str
        Metadata-field label used only in the failure message. It has no
        physical unit or array axis.

    Returns
    -------
    float
        Finite positive scalar preserving the caller-defined physical unit,
        such as Hz. It has no array axis.

    Raises
    ------
    ValueError
        If ``value`` is not a non-boolean finite positive JSON number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value <= 0:
        raise ValueError(f"Open Ephys LFP metadata {field_name} must be a finite positive number.")
    return float(value)


def _positive_integer(value: object, field_name: str) -> int:
    """Validate one positive JSON integer count.

    Parameters
    ----------
    value : object
        JSON-decoded candidate count. Only non-boolean Python integers greater
        than zero are accepted; floats and strings are rejected.
    field_name : str
        Metadata-field label used only in the failure message. It has no
        physical unit or array axis.

    Returns
    -------
    int
        Positive count with no physical unit and no array axis.

    Raises
    ------
    ValueError
        If ``value`` is not a non-boolean positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Open Ephys LFP metadata {field_name} must be a positive integer.")
    return value


def _validate_open_ephys_scaling(
    scaling: object,
    channel_ids: list[object],
    num_channels: int,
) -> None:
    """Fail closed unless one sidecar has the full observed affine-uV contract.

    Parameters
    ----------
    scaling : object
        JSON-decoded ``lfp_binary_scaling`` candidate. It must be a mapping
        describing canonical float32 stored values and one affine conversion
        to physical uV per saved-channel axis entry.
    channel_ids : list[object]
        Ordered sidecar channel identifiers for the saved-channel axis. They
        are categorical labels with shape ``(num_channels,)`` and no unit.
    num_channels : int
        Positive saved-channel-axis length. It has no physical unit.

    Returns
    -------
    None
        The input mapping is not mutated. No binary samples, synchronization,
        cache, filtering, interpolation, or numerical analysis is read.

    Raises
    ------
    ValueError
        If any required observed literal, channel-axis length/order, finite
        gain/offset, or explicit per-channel uV unit is absent or invalid.
    """
    if not isinstance(scaling, dict):
        raise ValueError("Open Ephys LFP metadata lfp_binary_scaling must be an object.")
    required_fields = {
        "data_units",
        "has_scaleable_traces",
        "channel_ids",
        "gain_to_uV_by_channel",
        "offset_to_uV_by_channel",
        "physical_unit_by_channel",
        "export_scale_factor",
        "conversion",
    }
    missing_fields = required_fields - set(scaling)
    if missing_fields:
        raise ValueError(f"Open Ephys LFP scaling is missing required fields: {sorted(missing_fields)}")
    if scaling["data_units"] != "unscaled_binary_values":
        raise ValueError("Open Ephys LFP scaling data_units is unsupported.")
    if scaling["has_scaleable_traces"] is not True:
        raise ValueError("Open Ephys LFP scaling has_scaleable_traces must be true.")
    if (
        isinstance(scaling["export_scale_factor"], bool)
        or not isinstance(scaling["export_scale_factor"], (int, float))
        or scaling["export_scale_factor"] != 1.0
    ):
        raise ValueError("Open Ephys LFP scaling export_scale_factor must be exactly 1.0.")
    if scaling["conversion"] != "trace_uV = trace_value * gain_to_uV + offset_to_uV":
        raise ValueError("Open Ephys LFP scaling conversion is unsupported.")
    for field_name in (
        "channel_ids",
        "gain_to_uV_by_channel",
        "offset_to_uV_by_channel",
        "physical_unit_by_channel",
    ):
        values = scaling[field_name]
        if not isinstance(values, list) or len(values) != num_channels:
            raise ValueError(f"Open Ephys LFP scaling {field_name} must match num_channels.")
    if scaling["channel_ids"] != channel_ids:
        raise ValueError("Open Ephys LFP scaling channel_ids must match channel_ids_in_binary_order.")
    if any(unit != "uV" for unit in scaling["physical_unit_by_channel"]):
        raise ValueError("Open Ephys LFP scaling physical_unit_by_channel must be explicit uV.")
    for gain in scaling["gain_to_uV_by_channel"]:
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not isfinite(float(gain)) or gain <= 0:
            raise ValueError("Open Ephys LFP scaling gain_to_uV_by_channel must be finite and positive.")
    for offset in scaling["offset_to_uV_by_channel"]:
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not isfinite(float(offset)):
            raise ValueError("Open Ephys LFP scaling offset_to_uV_by_channel must be finite.")


def sha256_file_content(path: Path | str, *, chunk_size_bytes: int = 1024 * 1024) -> str:
    """Return a streamed SHA-256 digest for one source-sidecar file.

    Parameters
    ----------
    path : pathlib.Path or str
        Existing file to hash. Units: filesystem path; contents are arbitrary
        bytes and are never interpreted as a numerical array.
    chunk_size_bytes : int, default=1048576
        Positive byte count read per iteration. It bounds temporary memory and
        does not change the returned digest.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest of every byte in ``path``. The
        return has no physical unit or array axis.

    Raises
    ------
    ValueError
        If ``chunk_size_bytes`` is not a positive integer.
    OSError
        If the source file cannot be opened or read.
    """
    if isinstance(chunk_size_bytes, bool) or not isinstance(chunk_size_bytes, int) or chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be a positive integer")
    digest = sha256()
    with Path(path).open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_open_ephys_site_metadata(
    site_id: str,
    saved_channel_index: int,
    configured_sample_rate_hz: float,
    configured_voltage_unit: str,
    metadata: dict[str, object],
) -> None:
    """Require one configured Open Ephys site to match normalized sidecar metadata.

    Parameters
    ----------
    site_id : str
        Stable human-readable site identifier. It has no physical unit.
    saved_channel_index : int
        Zero-based selected channel in the saved binary channel axis.
    configured_sample_rate_hz : float
        Positive configured LFP sampling rate in Hz.
    configured_voltage_unit : str
        Configured physical voltage unit for the selected channel.
    metadata : dict[str, object]
        Normalized Open Ephys sidecar metadata. ``num_channels`` defines the
        saved-channel axis and ``physical_unit_by_channel`` is aligned to it.

    Returns
    -------
    None
        Validation is non-mutating and performs no sync, binary-trace, cache,
        interpolation, filtering, or numerical analysis I/O.

    Raises
    ------
    ValueError
        If the selected channel is out of bounds or the configured rate/unit
        disagrees with authoritative normalized sidecar values.
    """
    num_channels = metadata.get("num_channels")
    if isinstance(saved_channel_index, bool) or not isinstance(saved_channel_index, int):
        raise ValueError(f"Open Ephys site {site_id}: saved_channel_index must be an integer")
    if isinstance(num_channels, bool) or not isinstance(num_channels, int) or num_channels <= 0:
        raise ValueError(f"Open Ephys site {site_id}: normalized metadata num_channels is invalid")
    if saved_channel_index < 0 or saved_channel_index >= num_channels:
        raise ValueError(
            f"Open Ephys site {site_id}: saved_channel_index={saved_channel_index} "
            f"is outside authoritative num_channels={num_channels}"
        )
    authoritative_rate_hz = metadata.get("sampling_frequency_hz")
    if (
        isinstance(configured_sample_rate_hz, bool)
        or not isinstance(configured_sample_rate_hz, (int, float))
        or not isfinite(float(configured_sample_rate_hz))
        or configured_sample_rate_hz <= 0
        or isinstance(authoritative_rate_hz, bool)
        or not isinstance(authoritative_rate_hz, (int, float))
        or not isfinite(float(authoritative_rate_hz))
        or authoritative_rate_hz <= 0
    ):
        raise ValueError(f"Open Ephys site {site_id}: sample_rate_hz must be finite and positive")
    configured_rate_hz = float(configured_sample_rate_hz)
    authoritative_rate = float(authoritative_rate_hz)
    if configured_rate_hz != authoritative_rate:
        raise ValueError(
            f"Open Ephys site {site_id}: configured sample_rate_hz={configured_rate_hz} "
            f"does not match authoritative sample_rate_hz={authoritative_rate}"
        )
    scaling = metadata.get("lfp_binary_scaling")
    if not isinstance(scaling, dict):
        raise ValueError(f"Open Ephys site {site_id}: normalized scaling metadata is invalid")
    physical_units = scaling.get("physical_unit_by_channel")
    if not isinstance(physical_units, list) or len(physical_units) != num_channels:
        raise ValueError(f"Open Ephys site {site_id}: normalized physical_unit_by_channel is invalid")
    authoritative_unit = physical_units[saved_channel_index]
    if not isinstance(configured_voltage_unit, str) or not isinstance(authoritative_unit, str):
        raise ValueError(f"Open Ephys site {site_id}: voltage_unit metadata is invalid")
    if configured_voltage_unit != authoritative_unit:
        raise ValueError(
            f"Open Ephys site {site_id}: configured voltage_unit={configured_voltage_unit} "
            f"does not match authoritative voltage_unit={authoritative_unit}"
        )


def validate_lfp_saved_channel(meta: dict, saved_channel_index: int) -> None:
    """
    Validate a saved-channel row index for one LFP binary file.

    Parameters
    ----------
    meta : dict
        SpikeGLX metadata containing ``nSavedChans``.
    saved_channel_index : int
        Zero-based saved channel index into the binary file rows. This is not
        necessarily the original/acquired channel id.

    Returns
    -------
    None
        Raises ``ValueError`` if the saved channel index is outside the binary
        row range.
    """

    if "nSavedChans" not in meta:
        raise ValueError("LFP metadata is missing nSavedChans.")
    n_saved_channels = int(meta["nSavedChans"])
    if int(saved_channel_index) < 0:
        raise ValueError("LFP saved channel index must be nonnegative.")
    if int(saved_channel_index) >= n_saved_channels:
        raise ValueError(
            f"LFP saved channel index {saved_channel_index} must be less than nSavedChans={n_saved_channels}."
        )


def read_open_ephys_lfp_channel_window(
    lfp_path: Path | str,
    saved_channel_index: int,
    start_sample: int,
    stop_sample: int,
) -> tuple[np.ndarray, float]:
    """
    Read one channel window from a derived Open Ephys LFP binary.

    Parameters
    ----------
    lfp_path : Path | str
        Path to a derived Open Ephys ``lfp.dat`` file. Units: filesystem path.
    saved_channel_index : int
        Zero-based saved channel index into the second axis of the binary
        layout. Units: channel index.
    start_sample : int
        Inclusive LFP start sample. Units: LFP samples.
    stop_sample : int
        Exclusive LFP stop sample. Units: LFP samples.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(lfp_values, sample_rate_hz)``. ``lfp_values`` has shape
        ``(stop_sample - start_sample,)`` in physical microvolts (uV).
        ``sample_rate_hz`` is the LFP sampling frequency in Hz.
    """

    lfp_file = Path(lfp_path)
    metadata = load_open_ephys_lfp_metadata(lfp_file)
    n_channels = int(metadata["num_channels"])
    n_samples = int(metadata["num_samples"])
    if int(saved_channel_index) < 0:
        raise ValueError("LFP saved channel index must be nonnegative.")
    if int(saved_channel_index) >= n_channels:
        raise ValueError(f"LFP saved channel index {saved_channel_index} must be less than num_channels={n_channels}.")
    if int(start_sample) < 0:
        raise ValueError("start_sample must be nonnegative.")
    if int(stop_sample) <= int(start_sample):
        raise ValueError("stop_sample must be greater than start_sample.")
    if int(stop_sample) > n_samples:
        raise ValueError(f"stop_sample {stop_sample} exceeds Open Ephys LFP sample count {n_samples}.")

    lfp_memmap = np.memmap(
        lfp_file,
        dtype=np.dtype(metadata["dtype"]),
        mode="r",
        shape=(n_samples, n_channels),
    )
    lfp_values = np.asarray(
        lfp_memmap[int(start_sample):int(stop_sample), int(saved_channel_index)],
        dtype=float,
    ).reshape(-1)
    scaling = metadata["lfp_binary_scaling"]
    gain_to_uv = float(scaling["gain_to_uV_by_channel"][int(saved_channel_index)])
    offset_to_uv = float(scaling["offset_to_uV_by_channel"][int(saved_channel_index)])
    # The selected float window is the only affine destination buffer.
    lfp_values *= gain_to_uv
    lfp_values += offset_to_uv
    return lfp_values, float(metadata["sampling_frequency_hz"])


def decode_lfp_sync(
    lfp_path: Path | str,
    digital_word: int = 0,
    irig_line: int = 6,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
) -> tuple[pd.DataFrame, float]:
    """
    Decode IRIG timestamps from the digital sync line in an LFP file.

    Parameters
    ----------
    lfp_path : Path | str
        SpikeGLX ``.lf.bin`` file path.
    digital_word : int, default=0
        Digital word index in the saved binary stream.
    irig_line : int, default=6
        Digital line carrying IRIG-H.
    bit_period_s : float, default=1.0
        IRIG-H bit period in seconds.
    utc_offset_hours : float, default=0.0
        Constant offset applied to decoded UTC timestamps.

    Returns
    -------
    tuple[pd.DataFrame, float]
        Decoded IRIG dataframe and LFP sample rate in Hz.
    """

    return ephys_sync_utils.decode_binary_file_irig_utc(
        binary_file=Path(lfp_path),
        digital_word=int(digital_word),
        irig_line=int(irig_line),
        bit_period_s=float(bit_period_s),
        utc_offset_hours=float(utc_offset_hours),
    )


def build_open_ephys_lfp_irig_df(
    aligned_sync_npz_path: Path | str,
    lfp_sample_rate_hz: float,
    continuous_sample_rate_hz: float | None = None,
    utc_offset_hours: float = 0.0,
) -> pd.DataFrame:
    """
    Convert Open Ephys AP/global IRIG anchors into derived LFP sample anchors.

    Parameters
    ----------
    aligned_sync_npz_path : Path | str
        Path to a probe sync ``.npz`` produced by
        ``sync_open_ephys_kilosort_spikes_to_utc``. Required arrays are
        ``irig_sample_ix`` in Open Ephys global AP/continuous samples and
        ``irig_utc_unix`` in UTC Unix seconds. The ``meta`` object should
        include ``continuous_start_sample_ix`` in global AP samples.
    lfp_sample_rate_hz : float
        LFP sampling rate in Hz.
    continuous_sample_rate_hz : float | None, optional
        AP/continuous sampling rate in Hz. If ``None``, the rate is estimated
        from finite IRIG sample/time anchor slopes.
    utc_offset_hours : float, default=0.0
        Constant offset added to UTC timestamps in hours.

    Returns
    -------
    pd.DataFrame
        IRIG anchor dataframe with one row per finite anchor. ``sample_ix`` is
        the corresponding derived LFP sample coordinate in samples, ``utc_unix``
        is UTC Unix time in seconds, and ``utc_datetime`` is timezone-aware UTC.
    """

    if float(lfp_sample_rate_hz) <= 0:
        raise ValueError("lfp_sample_rate_hz must be positive.")
    sync_file = np.load(Path(aligned_sync_npz_path), allow_pickle=True)
    required_arrays = {"irig_sample_ix", "irig_utc_unix", "meta"}
    missing_arrays = required_arrays - set(sync_file.files)
    if missing_arrays:
        raise ValueError(f"Open Ephys sync file is missing required arrays: {sorted(missing_arrays)}")

    irig_sample_ix = np.asarray(sync_file["irig_sample_ix"], dtype=float).reshape(-1)
    irig_utc_unix = np.asarray(sync_file["irig_utc_unix"], dtype=float).reshape(-1)
    if irig_sample_ix.shape != irig_utc_unix.shape:
        raise ValueError("irig_sample_ix and irig_utc_unix must have the same one-dimensional shape.")

    meta_value = sync_file["meta"]
    meta = meta_value.item() if getattr(meta_value, "shape", None) == () else dict(meta_value.tolist())
    continuous_start_sample_ix = float(meta.get("continuous_start_sample_ix", 0.0))

    finite_mask = np.isfinite(irig_sample_ix) & np.isfinite(irig_utc_unix)
    if finite_mask.sum() < 2:
        raise ValueError("At least two finite Open Ephys IRIG anchors are required for LFP alignment.")
    anchor_df = pd.DataFrame(
        {
            "global_sample_ix": irig_sample_ix[finite_mask],
            "utc_unix": irig_utc_unix[finite_mask],
        }
    ).sort_values("utc_unix", kind="mergesort")

    if continuous_sample_rate_hz is None:
        sample_diffs = np.diff(anchor_df["global_sample_ix"].to_numpy(dtype=float))
        utc_diffs = np.diff(anchor_df["utc_unix"].to_numpy(dtype=float))
        valid_rate_mask = (sample_diffs > 0) & (utc_diffs > 0)
        if not valid_rate_mask.any():
            raise ValueError("Could not infer continuous sample rate from Open Ephys IRIG anchors.")
        continuous_sample_rate_hz = float(np.median(sample_diffs[valid_rate_mask] / utc_diffs[valid_rate_mask]))
    if float(continuous_sample_rate_hz) <= 0:
        raise ValueError("continuous_sample_rate_hz must be positive.")

    lfp_sample_ix = (
        (anchor_df["global_sample_ix"].to_numpy(dtype=float) - continuous_start_sample_ix)
        * float(lfp_sample_rate_hz)
        / float(continuous_sample_rate_hz)
    )
    utc_unix = anchor_df["utc_unix"].to_numpy(dtype=float) + float(utc_offset_hours) * 3600.0
    return pd.DataFrame(
        {
            "sample_ix": lfp_sample_ix,
            "utc_unix": utc_unix,
            "utc_datetime": [
                datetime.fromtimestamp(unix_time, tz=timezone.utc) if np.isfinite(unix_time) else pd.NaT
                for unix_time in utc_unix
            ],
        }
    )


def map_lfp_time_window_to_samples(
    alignment_time_s: float,
    window: tuple[float, float],
    lfp_irig_df: pd.DataFrame,
    sample_rate_hz: float,
) -> tuple[np.ndarray, int, int]:
    """
    Convert a behavior-aligned time window to LFP sample indices.

    Parameters
    ----------
    alignment_time_s : float
        Absolute alignment timestamp in seconds, using the same UTC-like time
        base as decoded LFP IRIG timestamps.
    window : tuple[float, float]
        Relative window bounds in seconds around ``alignment_time_s``.
    lfp_irig_df : pd.DataFrame
        IRIG dataframe with ``sample_ix`` in samples and ``utc_unix`` in
        seconds.
    sample_rate_hz : float
        LFP sampling rate in Hz.

    Returns
    -------
    tuple[np.ndarray, int, int]
        ``(relative_time_s, start_sample, stop_sample)``. ``relative_time_s``
        has shape ``(stop_sample - start_sample,)`` in seconds. ``stop_sample``
        is exclusive.
    """

    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    if float(sample_rate_hz) <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    required_columns = {"sample_ix", "utc_unix"}
    missing_columns = required_columns - set(lfp_irig_df.columns)
    if missing_columns:
        raise ValueError(f"lfp_irig_df is missing required columns: {sorted(missing_columns)}")

    sync_points = lfp_irig_df.loc[:, ["sample_ix", "utc_unix"]].copy()
    sync_points["sample_ix"] = pd.to_numeric(sync_points["sample_ix"], errors="coerce")
    sync_points["utc_unix"] = pd.to_numeric(sync_points["utc_unix"], errors="coerce")
    sync_points = sync_points.replace([np.inf, -np.inf], np.nan).dropna().sort_values("utc_unix")
    if sync_points.shape[0] < 2:
        raise ValueError("At least two finite LFP IRIG sync points are required.")

    utc_unix = sync_points["utc_unix"].to_numpy(dtype=float)
    sample_ix = sync_points["sample_ix"].to_numpy(dtype=float)
    if float(alignment_time_s) < utc_unix[0] or float(alignment_time_s) > utc_unix[-1]:
        raise ValueError("alignment_time_s is outside the decoded LFP sync range.")

    alignment_sample = float(np.interp(float(alignment_time_s), utc_unix, sample_ix))
    sample_tolerance = 1e-9
    start_sample = int(np.floor(alignment_sample + float(window[0]) * float(sample_rate_hz) + sample_tolerance))
    stop_sample = int(np.ceil(alignment_sample + float(window[1]) * float(sample_rate_hz) - sample_tolerance))
    if start_sample < 0:
        raise ValueError("Requested LFP window starts before sample 0.")
    if stop_sample <= start_sample:
        raise ValueError("Requested LFP window contains no samples.")

    relative_time_s = (np.arange(start_sample, stop_sample, dtype=float) - alignment_sample) / float(sample_rate_hz)
    return relative_time_s, start_sample, stop_sample


def read_lfp_saved_channel_window(
    lfp_path: Path | str,
    saved_channel_index: int,
    start_sample: int,
    stop_sample: int,
) -> tuple[np.ndarray, float]:
    """
    Read a gain-corrected LFP segment for one saved channel.

    Parameters
    ----------
    lfp_path : Path | str
        SpikeGLX ``.lf.bin`` file path.
    saved_channel_index : int
        Zero-based saved channel index into the binary file rows.
    start_sample : int
        Inclusive start sample index.
    stop_sample : int
        Exclusive stop sample index.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(lfp_uv, sample_rate_hz)`` where ``lfp_uv`` has shape
        ``(stop_sample - start_sample,)`` in microvolts.
    """

    lfp_file = Path(lfp_path)
    meta = load_lfp_metadata(lfp_file)
    validate_lfp_saved_channel(meta, saved_channel_index=int(saved_channel_index))
    sample_rate_hz = float(readSGLX.SampRate(meta))
    n_saved_channels = int(meta["nSavedChans"])
    n_samples = int(int(meta["fileSizeBytes"]) / (2 * n_saved_channels))
    if int(start_sample) < 0:
        raise ValueError("start_sample must be nonnegative.")
    if int(stop_sample) <= int(start_sample):
        raise ValueError("stop_sample must be greater than start_sample.")
    if int(stop_sample) > n_samples:
        raise ValueError(f"stop_sample {stop_sample} exceeds LFP sample count {n_samples}.")

    raw_data = readSGLX.makeMemMapRaw(lfp_file, meta)
    channel_list = [int(saved_channel_index)]
    raw_channel_window = np.asarray(raw_data[channel_list, int(start_sample):int(stop_sample)], dtype=np.int16)
    gain_corrected_volts = readSGLX.GainCorrectIM(raw_channel_window, channel_list, meta)
    lfp_uv = np.asarray(gain_corrected_volts, dtype=float).reshape(-1) * 1e6
    return lfp_uv, sample_rate_hz


def filter_lfp_trace(
    relative_time_s: np.ndarray,
    lfp_uv: np.ndarray,
    sample_rate_hz: float,
    frequency_band_hz: tuple[float, float] | None,
) -> np.ndarray:
    """
    Optionally bandpass-filter one LFP trace with Pynapple.

    Parameters
    ----------
    relative_time_s : np.ndarray
        One-dimensional time vector with shape ``(n_samples,)`` in seconds.
        Times are relative to the trial alignment event.
    lfp_uv : np.ndarray
        One-dimensional LFP voltage vector with shape ``(n_samples,)`` in
        microvolts. Values should already be gain-corrected.
    sample_rate_hz : float
        LFP sample rate in Hz.
    frequency_band_hz : tuple[float, float] | None
        Bandpass cutoff frequencies in Hz as ``(low_hz, high_hz)``. If
        ``None``, the gain-corrected signal is returned unfiltered.

    Returns
    -------
    np.ndarray
        One-dimensional LFP vector with shape ``(n_samples,)`` in microvolts.
        The default path is unfiltered; band-limited paths use
        ``nap.apply_bandpass_filter(..., mode="butter")``.
    """

    relative_time_s = np.asarray(relative_time_s, dtype=float).reshape(-1)
    lfp_uv = np.asarray(lfp_uv, dtype=float).reshape(-1)
    if relative_time_s.shape != lfp_uv.shape:
        raise ValueError("relative_time_s and lfp_uv must have the same one-dimensional shape.")
    if float(sample_rate_hz) <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if frequency_band_hz is None:
        return lfp_uv

    if len(frequency_band_hz) != 2:
        raise ValueError("frequency_band_hz must be None or a two-value tuple.")
    low_hz, high_hz = (float(frequency_band_hz[0]), float(frequency_band_hz[1]))
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("frequency_band_hz must satisfy 0 < low_hz < high_hz.")

    lfp_tsd = nap.Tsd(t=relative_time_s, d=lfp_uv, time_units="s")
    filtered_tsd = nap.apply_bandpass_filter(
        lfp_tsd,
        (low_hz, high_hz),
        fs=float(sample_rate_hz),
        mode="butter",
    )
    return np.asarray(filtered_tsd.values, dtype=float).reshape(-1)


def load_trial_lfp_trace(
    lfp_path: Path | str,
    saved_channel_index: int,
    alignment_time_s: float,
    window: tuple[float, float],
    lfp_irig_df: pd.DataFrame,
    sample_rate_hz: float,
    frequency_band_hz: tuple[float, float] | None = None,
    filter_padding_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one trial-aligned LFP trace from a saved-channel row.

    Parameters
    ----------
    lfp_path : Path | str
        SpikeGLX ``.lf.bin`` file path.
    saved_channel_index : int
        Zero-based saved channel index into the binary file rows.
    alignment_time_s : float
        Absolute alignment timestamp in seconds.
    window : tuple[float, float]
        Relative window bounds in seconds around ``alignment_time_s``.
    lfp_irig_df : pd.DataFrame
        IRIG dataframe with ``sample_ix`` in samples and ``utc_unix`` in
        seconds.
    sample_rate_hz : float
        LFP sampling rate in Hz.
    frequency_band_hz : tuple[float, float] | None, default=None
        Optional bandpass cutoff frequencies in Hz as ``(low_hz, high_hz)``.
        ``None`` returns the gain-corrected, unfiltered trace.
    filter_padding_s : float, default=1.0
        Seconds added to both sides of the requested window before filtering.
        The returned arrays are trimmed back to ``window`` after filtering.
        This parameter is ignored when ``frequency_band_hz`` is ``None``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(relative_time_s, lfp_uv)``. Both arrays have shape ``(n_samples,)``.
        Time is in seconds relative to alignment; LFP is in microvolts.
    """

    relative_time_s, start_sample, stop_sample = map_lfp_time_window_to_samples(
        alignment_time_s=alignment_time_s,
        window=window,
        lfp_irig_df=lfp_irig_df,
        sample_rate_hz=sample_rate_hz,
    )
    read_start_sample = start_sample
    read_stop_sample = stop_sample
    read_relative_time_s = relative_time_s
    if frequency_band_hz is not None:
        if float(filter_padding_s) < 0:
            raise ValueError("filter_padding_s must be nonnegative.")
        padded_window = (
            float(window[0]) - float(filter_padding_s),
            float(window[1]) + float(filter_padding_s),
        )
        read_relative_time_s, read_start_sample, read_stop_sample = map_lfp_time_window_to_samples(
            alignment_time_s=alignment_time_s,
            window=padded_window,
            lfp_irig_df=lfp_irig_df,
            sample_rate_hz=sample_rate_hz,
        )
    lfp_uv, file_sample_rate_hz = read_lfp_saved_channel_window(
        lfp_path=lfp_path,
        saved_channel_index=int(saved_channel_index),
        start_sample=read_start_sample,
        stop_sample=read_stop_sample,
    )
    if not np.isclose(float(file_sample_rate_hz), float(sample_rate_hz)):
        raise ValueError(
            f"LFP sync sample rate {sample_rate_hz} does not match file metadata sample rate {file_sample_rate_hz}."
        )
    if lfp_uv.shape[0] != read_relative_time_s.shape[0]:
        raise ValueError("Loaded LFP trace length does not match the requested sample window.")
    if frequency_band_hz is not None:
        filtered_lfp_uv = filter_lfp_trace(
            relative_time_s=read_relative_time_s,
            lfp_uv=lfp_uv,
            sample_rate_hz=float(sample_rate_hz),
            frequency_band_hz=frequency_band_hz,
        )
        trim_start = int(start_sample) - int(read_start_sample)
        trim_stop = trim_start + int(stop_sample) - int(start_sample)
        lfp_uv = filtered_lfp_uv[trim_start:trim_stop]
        if lfp_uv.shape[0] != relative_time_s.shape[0]:
            raise ValueError("Filtered LFP trace length does not match the requested sample window.")
    return relative_time_s, lfp_uv


def load_trial_lfp_trace_with_sample_rate(
    lfp_path: Path | str,
    saved_channel_index: int,
    alignment_time_s: float,
    window: tuple[float, float],
    lfp_irig_df: pd.DataFrame,
    sample_rate_hz: float,
    frequency_band_hz: tuple[float, float] | None = None,
    filter_padding_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load one SpikeGLX trial LFP trace and retain its sample rate.

    Parameters
    ----------
    lfp_path : Path | str
        SpikeGLX ``.lf.bin`` file path.
    saved_channel_index : int
        Zero-based saved-channel row index.
    alignment_time_s : float
        Absolute alignment timestamp in seconds.
    window : tuple[float, float]
        Relative window bounds in seconds around ``alignment_time_s``.
    lfp_irig_df : pd.DataFrame
        IRIG anchors with ``sample_ix`` in samples and ``utc_unix`` in seconds.
    sample_rate_hz : float
        LFP sample rate in Hz.
    frequency_band_hz : tuple[float, float] | None, default=None
        Optional bandpass frequencies in Hz. ``None`` returns unfiltered data.
    filter_padding_s : float, default=1.0
        Filtering padding on each side in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(relative_time_s, lfp_uv, sample_rate_hz)``. The arrays have shape
        ``(n_samples,)``; time is in seconds and LFP is in microvolts.
    """

    relative_time_s, lfp_uv = load_trial_lfp_trace(
        lfp_path=lfp_path,
        saved_channel_index=int(saved_channel_index),
        alignment_time_s=float(alignment_time_s),
        window=window,
        lfp_irig_df=lfp_irig_df,
        sample_rate_hz=float(sample_rate_hz),
        frequency_band_hz=frequency_band_hz,
        filter_padding_s=float(filter_padding_s),
    )
    return relative_time_s, lfp_uv, float(sample_rate_hz)


def load_open_ephys_trial_lfp_trace(
    lfp_path: Path | str,
    aligned_sync_npz_path: Path | str,
    saved_channel_index: int,
    alignment_time_s: float,
    window: tuple[float, float],
    continuous_sample_rate_hz: float | None = None,
    utc_offset_hours: float = 0.0,
    frequency_band_hz: tuple[float, float] | None = None,
    filter_padding_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one trial-aligned trace from a derived Open Ephys LFP file.

    Parameters
    ----------
    lfp_path : Path | str
        Path to a derived Open Ephys ``lfp.dat`` file. Units: filesystem path.
    aligned_sync_npz_path : Path | str
        Path to the matching Open Ephys probe sync ``.npz``. IRIG anchors in
        this file are in Open Ephys global AP/continuous samples and UTC Unix
        seconds.
    saved_channel_index : int
        Zero-based saved channel index into the LFP binary columns. Units:
        channel index.
    alignment_time_s : float
        Absolute trial alignment timestamp in UTC Unix seconds.
    window : tuple[float, float]
        Relative window bounds in seconds around ``alignment_time_s``.
    continuous_sample_rate_hz : float | None, optional
        AP/continuous sampling rate in Hz. ``None`` estimates the rate from
        finite IRIG anchors in ``aligned_sync_npz_path``.
    utc_offset_hours : float, default=0.0
        Constant offset added to sync timestamps in hours before trial mapping.
    frequency_band_hz : tuple[float, float] | None, default=None
        Optional bandpass cutoff frequencies in Hz as ``(low_hz, high_hz)``.
        ``None`` returns the unfiltered derived LFP values.
    filter_padding_s : float, default=1.0
        Seconds added to both sides of the requested window before filtering.
        The returned arrays are trimmed back to ``window`` after filtering.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(relative_time_s, lfp_values)``. Both arrays have shape
        ``(n_samples,)``. Time is in seconds relative to alignment; LFP values
        are physical uV after exactly one affine conversion from canonical
        float32 ``lfp.dat`` storage. The returned working array remains the
        reader's floating-point copy and preserves the requested time grid.
    """

    metadata = load_open_ephys_lfp_metadata(lfp_path)
    sample_rate_hz = float(metadata["sampling_frequency_hz"])
    lfp_irig_df = build_open_ephys_lfp_irig_df(
        aligned_sync_npz_path=aligned_sync_npz_path,
        lfp_sample_rate_hz=sample_rate_hz,
        continuous_sample_rate_hz=continuous_sample_rate_hz,
        utc_offset_hours=utc_offset_hours,
    )
    relative_time_s, start_sample, stop_sample = map_lfp_time_window_to_samples(
        alignment_time_s=alignment_time_s,
        window=window,
        lfp_irig_df=lfp_irig_df,
        sample_rate_hz=sample_rate_hz,
    )

    read_start_sample = start_sample
    read_stop_sample = stop_sample
    read_relative_time_s = relative_time_s
    if frequency_band_hz is not None:
        if float(filter_padding_s) < 0:
            raise ValueError("filter_padding_s must be nonnegative.")
        padded_window = (
            float(window[0]) - float(filter_padding_s),
            float(window[1]) + float(filter_padding_s),
        )
        read_relative_time_s, read_start_sample, read_stop_sample = map_lfp_time_window_to_samples(
            alignment_time_s=alignment_time_s,
            window=padded_window,
            lfp_irig_df=lfp_irig_df,
            sample_rate_hz=sample_rate_hz,
        )

    lfp_values, file_sample_rate_hz = read_open_ephys_lfp_channel_window(
        lfp_path=lfp_path,
        saved_channel_index=int(saved_channel_index),
        start_sample=read_start_sample,
        stop_sample=read_stop_sample,
    )
    if not np.isclose(float(file_sample_rate_hz), sample_rate_hz):
        raise ValueError(
            f"LFP sync sample rate {sample_rate_hz} does not match file metadata sample rate {file_sample_rate_hz}."
        )
    if lfp_values.shape[0] != read_relative_time_s.shape[0]:
        raise ValueError("Loaded Open Ephys LFP trace length does not match the requested sample window.")
    if frequency_band_hz is not None:
        filtered_lfp_values = filter_lfp_trace(
            relative_time_s=read_relative_time_s,
            lfp_uv=lfp_values,
            sample_rate_hz=sample_rate_hz,
            frequency_band_hz=frequency_band_hz,
        )
        trim_start = int(start_sample) - int(read_start_sample)
        trim_stop = trim_start + int(stop_sample) - int(start_sample)
        lfp_values = filtered_lfp_values[trim_start:trim_stop]
        if lfp_values.shape[0] != relative_time_s.shape[0]:
            raise ValueError("Filtered Open Ephys LFP trace length does not match the requested sample window.")
    return relative_time_s, lfp_values
