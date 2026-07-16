"""
This code should aim for manual synchronization between probe sync lines and a DAQ, and between electrical
lines (probe/DAQ) to a behavior reference.

behavior timestamps and behavior flipper have utc times, DAQ/imec only have rising/falling
edges. Use the edges to align to the flipper utc and get spike/sample times from there. Flipper barcodes can be used
to identify the session start/end locations in the electrical signals.

"""

import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal
import warnings
import re
import pickle as pkl
from src.irig_tools.irig_sync_utils import find_signal_edges
from src.neural_analysis.ephys_sync_utils import save_stream_sync_npz
from src.neural_analysis.spikeglx_sync_io import read_digital_lines


BoundsPolicy = Literal["reject", "clip", "warn", "allow"]
SpikeTimeUnits = Literal["samples", "seconds"]


@dataclass(frozen=True)
class FlipperUtcMapper:
    """Map NI DAQ times to behavior flipper UTC times.

    Attributes:
        ni_to_utc:
            Callable mapping NI DAQ times in seconds, shape ``(n_times,)``,
            to Unix UTC seconds, shape ``(n_times,)``.
        matched_ni_pos_t:
            NI DAQ positive flipper edge times in seconds, shape
            ``(n_matched_edges,)``.
        matched_behavior_pos_utc:
            Behavior positive flipper edge timestamps as Unix UTC seconds,
            shape ``(n_matched_edges,)``.
        bounds_s:
            Inclusive barcode-derived NI DAQ session bounds in seconds as
            ``(start_s, end_s)``. ``None`` means no bounds are enforced.
        bounds_policy:
            How calls to :meth:`map_ni_times_to_utc` handle times outside
            ``bounds_s``.
        negative_edge_residual_s:
            Validation residuals in seconds for matched negative flipper edges,
            computed as mapped NI UTC minus behavior UTC. Empty when no
            negative-edge validation was possible.
    """

    ni_to_utc: Callable[[npt.ArrayLike], np.ndarray]
    matched_ni_pos_t: np.ndarray
    matched_behavior_pos_utc: np.ndarray
    bounds_s: tuple[float, float] | None
    bounds_policy: BoundsPolicy
    negative_edge_residual_s: np.ndarray

    def map_ni_times_to_utc(
        self,
        ni_times_s: npt.ArrayLike,
        bounds_policy: BoundsPolicy | None = None,
    ) -> np.ndarray:
        """Map NI DAQ times in seconds to Unix UTC seconds.

        Args:
            ni_times_s:
                NI DAQ times in seconds, shape ``(n_times,)``.
            bounds_policy:
                Optional override for this mapper's bounds behavior. Accepted
                values are ``"reject"``, ``"clip"``, ``"warn"``, and
                ``"allow"``.

        Returns:
            Unix UTC seconds, shape ``(n_kept_times,)``. The shape matches the
            input except when ``bounds_policy="clip"``, which removes
            out-of-bounds times.
        """
        policy = self.bounds_policy if bounds_policy is None else bounds_policy
        bounded_times = apply_bounds_policy(ni_times_s, self.bounds_s, policy)
        return np.asarray(self.ni_to_utc(bounded_times), dtype=float)


def load_behavior_flipper_timestamps(flipper_csv: Path | str) -> dict[str, np.ndarray | pd.DataFrame]:
    """Load behavior-side flipper UTC timestamps from the Raspberry Pi CSV.

    Args:
        flipper_csv:
            Path to a behavior flipper CSV. The file must contain a
            ``pin_state`` column with binary flipper state values and a
            ``time.time()`` column containing Unix UTC seconds. Rows are
            interpreted in file order.

    Returns:
        Dictionary with:
            ``data_frame``:
                Full parsed CSV as a ``pandas.DataFrame`` with the required
                columns preserved.
            ``positive_utc_seconds``:
                Unix UTC seconds for rows where ``pin_state == 1``, shape
                ``(n_positive_edges,)``.
            ``negative_utc_seconds``:
                Unix UTC seconds for rows where ``pin_state == 0``, shape
                ``(n_negative_edges,)``.

    Raises:
        ValueError:
            If required columns are missing, timestamps are non-monotonic, or
            ``pin_state`` contains values other than 0 and 1.
    """
    flipper_df = pd.read_csv(flipper_csv)
    required_columns = {"pin_state", "time.time()"}
    missing_columns = required_columns.difference(flipper_df.columns)
    if missing_columns:
        raise ValueError(
            f"Behavior flipper CSV is missing required column(s): {sorted(missing_columns)}"
        )

    pin_state = flipper_df["pin_state"].to_numpy()
    if not np.isin(pin_state, [0, 1]).all():
        raise ValueError("Behavior flipper pin_state values must be binary 0/1 values.")

    utc_seconds = flipper_df["time.time()"].to_numpy(dtype=float)
    if utc_seconds.size > 1 and np.any(np.diff(utc_seconds) < 0):
        raise ValueError("Behavior flipper time.time() values must be monotonic increasing.")

    return {
        "data_frame": flipper_df,
        "positive_utc_seconds": utc_seconds[pin_state == 1],
        "negative_utc_seconds": utc_seconds[pin_state == 0],
    }


def match_timestamps_by_intervals(
    source_times_s: npt.ArrayLike,
    target_times_s: npt.ArrayLike,
    tolerance_s: float = 0.001,
    min_matches: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Match a target event sequence inside a longer source sequence by intervals.

    Args:
        source_times_s:
            Source event times in seconds, shape ``(n_source_events,)``. For
            manual session sync this is usually NI DAQ flipper times.
        target_times_s:
            Target event times in seconds or Unix UTC seconds, shape
            ``(n_target_events,)``. For manual session sync this is usually
            behavior flipper timestamps.
        tolerance_s:
            Maximum absolute interval mismatch in seconds.
        min_matches:
            Minimum number of matched events required to accept a match.

    Returns:
        Tuple ``(matched_source_times_s, matched_target_times_s)`` with both
        arrays shaped ``(n_target_events,)``.

    Raises:
        ValueError:
            If the target sequence cannot be matched contiguously inside the
            source sequence.
    """
    source_times = np.asarray(source_times_s, dtype=float)
    target_times = np.asarray(target_times_s, dtype=float)
    if source_times.ndim != 1 or target_times.ndim != 1:
        raise ValueError("source_times_s and target_times_s must be 1D arrays.")
    if target_times.size < min_matches:
        raise ValueError(
            f"At least {min_matches} target timestamps are required; got {target_times.size}."
        )
    if source_times.size < target_times.size:
        raise ValueError("Source timestamps must contain at least as many events as target timestamps.")

    target_intervals = np.diff(target_times)
    for start_ix in range(0, source_times.size - target_times.size + 1):
        candidate = source_times[start_ix:start_ix + target_times.size]
        candidate_intervals = np.diff(candidate)
        if np.all(np.abs(candidate_intervals - target_intervals) <= tolerance_s):
            return candidate, target_times

    raise ValueError(
        "No matching timestamp sequence found between source and target intervals."
    )


def make_timebase_mapper(
    source_times_s: npt.ArrayLike,
    target_times_s: npt.ArrayLike,
) -> Callable[[npt.ArrayLike], np.ndarray]:
    """Create a linear timebase mapper from source seconds to target seconds.

    Args:
        source_times_s:
            Source clock event times in seconds, shape ``(n_events,)``.
        target_times_s:
            Target clock event times in seconds or Unix UTC seconds, shape
            ``(n_events,)``.

    Returns:
        Callable that maps source times in seconds, shape ``(n_query,)``, to
        target times with the same shape. Values outside the anchor range are
        linearly extrapolated; callers should apply bounds checks before using
        extrapolated values.
    """
    source_times = np.asarray(source_times_s, dtype=float)
    target_times = np.asarray(target_times_s, dtype=float)
    if source_times.ndim != 1 or target_times.ndim != 1:
        raise ValueError("source_times_s and target_times_s must be 1D arrays.")
    if source_times.size != target_times.size:
        raise ValueError("source_times_s and target_times_s must have the same length.")
    if source_times.size < 2:
        raise ValueError("At least two timebase anchors are required.")
    if np.any(np.diff(source_times) <= 0):
        raise ValueError("source_times_s must be strictly increasing.")

    interpolator = interp1d(
        source_times,
        target_times,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )

    def mapper(query_times_s: npt.ArrayLike) -> np.ndarray:
        """Map source query times in seconds to target times."""
        return np.asarray(interpolator(np.asarray(query_times_s, dtype=float)), dtype=float)

    return mapper


def apply_bounds_policy(
    times_s: npt.ArrayLike,
    bounds_s: tuple[float, float] | None,
    policy: BoundsPolicy = "reject",
) -> np.ndarray:
    """Apply barcode-session bounds to times in an electrical recording.

    Args:
        times_s:
            Event or sample times in seconds, shape ``(n_times,)``.
        bounds_s:
            Inclusive barcode-derived bounds in seconds as ``(start_s, end_s)``.
            ``None`` means the input is returned unchanged.
        policy:
            Bounds behavior. ``"reject"`` raises on out-of-bounds values,
            ``"clip"`` removes them, ``"warn"`` keeps them with a warning, and
            ``"allow"`` keeps them silently.

    Returns:
        Times in seconds after policy application, shape ``(n_kept_times,)``.
        The output shape differs from input only for ``policy="clip"``.
    """
    times = np.asarray(times_s, dtype=float)
    if bounds_s is None:
        return times
    if policy not in {"reject", "clip", "warn", "allow"}:
        raise ValueError(f"Unknown bounds policy: {policy}")

    start_s, end_s = bounds_s
    outside_bounds = (times < start_s) | (times > end_s)
    if not np.any(outside_bounds) or policy == "allow":
        return times
    if policy == "clip":
        return times[~outside_bounds]

    message = (
        f"{int(np.sum(outside_bounds))} time value(s) are outside barcode bounds "
        f"[{start_s}, {end_s}] seconds."
    )
    if policy == "reject":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return times


def build_flipper_utc_mapper(
    daq_flipper_events: dict[str, np.ndarray],
    behavior_flipper_times: dict[str, np.ndarray | pd.DataFrame],
    bounds_s: tuple[float, float] | None,
    bounds_policy: BoundsPolicy = "reject",
    tolerance_s: float = 0.001,
) -> FlipperUtcMapper:
    """Build an NI DAQ to UTC mapper from positive behavior flipper edges.

    Args:
        daq_flipper_events:
            Event dictionary from :func:`decode_flipper_barcodes`. It must
            contain ``flipper_pos_t`` and may contain ``flipper_neg_t``. Times
            are NI DAQ seconds.
        behavior_flipper_times:
            Dictionary from :func:`load_behavior_flipper_timestamps` containing
            behavior positive and negative flipper timestamps as Unix UTC
            seconds.
        bounds_s:
            Inclusive barcode-derived NI DAQ bounds in seconds as
            ``(start_s, end_s)``. ``None`` disables bounds checks.
        bounds_policy:
            Default bounds behavior for the returned mapper.
        tolerance_s:
            Maximum allowed interval mismatch in seconds during event matching.

    Returns:
        :class:`FlipperUtcMapper` that maps NI DAQ seconds to Unix UTC seconds.
    """
    daq_positive_t = np.asarray(daq_flipper_events["flipper_pos_t"], dtype=float)
    behavior_positive_utc = np.asarray(
        behavior_flipper_times["positive_utc_seconds"],
        dtype=float,
    )
    matched_daq_pos_t, matched_behavior_pos_utc = match_timestamps_by_intervals(
        daq_positive_t,
        behavior_positive_utc,
        tolerance_s=tolerance_s,
    )
    ni_to_utc = make_timebase_mapper(matched_daq_pos_t, matched_behavior_pos_utc)

    negative_residual_s = np.array([], dtype=float)
    daq_negative_t = np.asarray(daq_flipper_events.get("flipper_neg_t", []), dtype=float)
    behavior_negative_utc = np.asarray(
        behavior_flipper_times.get("negative_utc_seconds", []),
        dtype=float,
    )
    if daq_negative_t.size >= 2 and behavior_negative_utc.size >= 2:
        matched_daq_neg_t, matched_behavior_neg_utc = match_timestamps_by_intervals(
            daq_negative_t,
            behavior_negative_utc,
            tolerance_s=tolerance_s,
        )
        negative_residual_s = ni_to_utc(matched_daq_neg_t) - matched_behavior_neg_utc

    return FlipperUtcMapper(
        ni_to_utc=ni_to_utc,
        matched_ni_pos_t=matched_daq_pos_t,
        matched_behavior_pos_utc=matched_behavior_pos_utc,
        bounds_s=bounds_s,
        bounds_policy=bounds_policy,
        negative_edge_residual_s=negative_residual_s,
    )


def map_ni_times_to_utc(
    ni_times_s: npt.ArrayLike,
    ni_to_utc: FlipperUtcMapper | Callable[[npt.ArrayLike], np.ndarray],
    bounds_s: tuple[float, float] | None = None,
    bounds_policy: BoundsPolicy = "reject",
) -> np.ndarray:
    """Map NI DAQ times in seconds to behavior UTC seconds.

    Args:
        ni_times_s:
            NI DAQ times in seconds, shape ``(n_times,)``.
        ni_to_utc:
            Either a :class:`FlipperUtcMapper` or a callable mapping NI seconds
            to Unix UTC seconds.
        bounds_s:
            Inclusive barcode-derived NI DAQ bounds in seconds. Ignored when
            ``ni_to_utc`` is a :class:`FlipperUtcMapper`.
        bounds_policy:
            How to handle out-of-bounds times.

    Returns:
        Unix UTC seconds, shape ``(n_kept_times,)``.
    """
    if isinstance(ni_to_utc, FlipperUtcMapper):
        return ni_to_utc.map_ni_times_to_utc(ni_times_s, bounds_policy=bounds_policy)

    bounded_times = apply_bounds_policy(ni_times_s, bounds_s, bounds_policy)
    return np.asarray(ni_to_utc(bounded_times), dtype=float)


def map_imec_samples_to_utc(
    sample_ix: npt.ArrayLike,
    imec_sample_rate: float,
    imec_to_ni: Callable[[npt.ArrayLike], np.ndarray],
    ni_to_utc: FlipperUtcMapper | Callable[[npt.ArrayLike], np.ndarray],
    bounds_s: tuple[float, float] | None = None,
    bounds_policy: BoundsPolicy = "reject",
) -> np.ndarray:
    """Map imec sample indices to behavior UTC seconds through NI time.

    Args:
        sample_ix:
            imec sample indices, shape ``(n_samples,)``. Units are samples.
        imec_sample_rate:
            imec sampling rate in Hz.
        imec_to_ni:
            Callable mapping imec seconds to NI DAQ seconds.
        ni_to_utc:
            Either a :class:`FlipperUtcMapper` or a callable mapping NI seconds
            to Unix UTC seconds.
        bounds_s:
            Inclusive barcode-derived NI DAQ bounds in seconds. Ignored when
            ``ni_to_utc`` is a :class:`FlipperUtcMapper`.
        bounds_policy:
            How to handle NI times outside barcode bounds.

    Returns:
        Unix UTC seconds, shape ``(n_kept_samples,)``.
    """
    sample_indices = np.asarray(sample_ix, dtype=float)
    imec_times_s = sample_indices / float(imec_sample_rate)
    ni_times_s = np.asarray(imec_to_ni(imec_times_s), dtype=float)
    return map_ni_times_to_utc(
        ni_times_s,
        ni_to_utc,
        bounds_s=bounds_s,
        bounds_policy=bounds_policy,
    )


def write_manual_alignment_note(output_root: Path | str) -> Path:
    """Write a visible note identifying manual behavior-flipper synchronization.

    Args:
        output_root:
            Root aligned-output directory. The note is written to
            ``output_root / "time_adjustment_note.txt"``.

    Returns:
        Path to the written note file.
    """
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    note_path = output_path / "time_adjustment_note.txt"
    note_lines = [
        "UTC hour adjustment: 0",
        f"Processed at: {datetime.now().astimezone().isoformat()}",
        "Synchronization method: manual line sync to the behavior flipper timestamps",
    ]
    note_path.write_text("\n".join(note_lines) + "\n")
    return note_path


def load_spike_times_npy(spike_times_npy: Path | str) -> np.ndarray:
    """Load Kilosort/Phy spike times from a ``spike_times.npy`` file.

    Args:
        spike_times_npy:
            Path to a NumPy file containing spike times. Values may be IMEC AP
            sample indices or IMEC seconds, depending on the later
            ``spike_time_units`` argument.

    Returns:
        One-dimensional array with shape ``(n_spikes,)``.
    """
    return np.asarray(np.load(Path(spike_times_npy), allow_pickle=True)).reshape(-1)


def load_spike_clusters_npy(spike_clusters_npy: Path | str) -> np.ndarray:
    """Load Kilosort/Phy cluster ids from a ``spike_clusters.npy`` file.

    Args:
        spike_clusters_npy:
            Path to a NumPy file containing cluster ids, shape compatible with
            ``spike_times.npy``.

    Returns:
        One-dimensional integer array with shape ``(n_spikes,)``.
    """
    return np.asarray(np.load(Path(spike_clusters_npy), allow_pickle=True), dtype=np.int64).reshape(-1)


def validate_spike_times_and_clusters(
    spike_times: npt.ArrayLike,
    spike_clusters: npt.ArrayLike,
) -> None:
    """Validate that spike times and cluster ids are aligned one-to-one.

    Args:
        spike_times:
            One-dimensional spike-time array, shape ``(n_spikes,)``.
        spike_clusters:
            One-dimensional cluster-id array, shape ``(n_spikes,)``.

    Returns:
        None. Raises on invalid shape or length.
    """
    spike_times_array = np.asarray(spike_times)
    spike_clusters_array = np.asarray(spike_clusters)
    if spike_times_array.ndim != 1 or spike_clusters_array.ndim != 1:
        raise ValueError("spike_times and spike_clusters must both be one-dimensional arrays.")
    if spike_times_array.shape[0] != spike_clusters_array.shape[0]:
        raise ValueError("spike_times and spike_clusters must have the same length.")


def convert_spike_times_to_imec_seconds(
    spike_times: npt.ArrayLike,
    imec_sample_rate: float,
    spike_time_units: SpikeTimeUnits = "samples",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert spike times into IMEC recording seconds.

    Args:
        spike_times:
            One-dimensional spike-time array, shape ``(n_spikes,)``. Units are
            either IMEC AP samples or IMEC seconds.
        imec_sample_rate:
            IMEC AP sampling rate in Hz.
        spike_time_units:
            ``"samples"`` when ``spike_times`` contains IMEC AP sample indices,
            or ``"seconds"`` when it already contains IMEC seconds.

    Returns:
        tuple[np.ndarray, np.ndarray | None]:
            - Spike times in IMEC seconds, shape ``(n_spikes,)``.
            - Spike sample indices in samples, shape ``(n_spikes,)``, when the
              input units are samples; otherwise ``None``.
    """
    spike_time_array = np.asarray(spike_times)
    if spike_time_array.ndim != 1:
        raise ValueError("spike_times must be one-dimensional.")
    if float(imec_sample_rate) <= 0:
        raise ValueError("imec_sample_rate must be positive.")

    if spike_time_units == "samples":
        if not np.all(np.isfinite(spike_time_array.astype(float))):
            raise ValueError("Spike sample indices must be finite.")
        rounded_samples = np.rint(spike_time_array.astype(float))
        if not np.allclose(spike_time_array.astype(float), rounded_samples):
            raise ValueError("Spike sample indices must be integer-valued when spike_time_units='samples'.")
        spike_sample_ix = rounded_samples.astype(np.int64)
        return spike_sample_ix.astype(float) / float(imec_sample_rate), spike_sample_ix
    if spike_time_units == "seconds":
        spike_imec_time_s = spike_time_array.astype(float)
        if not np.all(np.isfinite(spike_imec_time_s)):
            raise ValueError("Spike times in seconds must be finite.")
        return spike_imec_time_s, None

    raise ValueError("spike_time_units must be either 'samples' or 'seconds'.")


def _manual_bounds_keep_mask(
    times_s: np.ndarray,
    bounds_s: tuple[float, float] | None,
    bounds_policy: BoundsPolicy,
) -> np.ndarray:
    """Return a keep mask for manual-sync bounds handling."""
    keep_mask = np.ones(times_s.shape[0], dtype=bool)
    if bounds_s is None or bounds_policy == "allow":
        return keep_mask
    start_s, end_s = bounds_s
    outside_bounds = (times_s < start_s) | (times_s > end_s)
    if not np.any(outside_bounds):
        return keep_mask
    message = (
        f"{int(np.sum(outside_bounds))} spike time value(s) are outside barcode bounds "
        f"[{start_s}, {end_s}] NI seconds."
    )
    if bounds_policy == "reject":
        raise ValueError(message)
    if bounds_policy == "clip":
        return ~outside_bounds
    if bounds_policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return keep_mask
    raise ValueError(f"Unknown bounds policy: {bounds_policy}")


def map_imec_spikes_to_utc(
    spike_times: npt.ArrayLike,
    spike_clusters: npt.ArrayLike,
    imec_sample_rate: float,
    imec_to_ni: Callable[[npt.ArrayLike], np.ndarray],
    ni_to_utc: FlipperUtcMapper | Callable[[npt.ArrayLike], np.ndarray],
    spike_time_units: SpikeTimeUnits = "samples",
    bounds_policy: BoundsPolicy = "reject",
    bounds_s: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Map Kilosort/Phy spike times to behavior UTC using manual line sync.

    Args:
        spike_times:
            Spike times, shape ``(n_spikes,)``. Units are controlled by
            ``spike_time_units``.
        spike_clusters:
            Cluster ids aligned one-to-one with ``spike_times``, shape
            ``(n_spikes,)``.
        imec_sample_rate:
            IMEC AP sampling rate in Hz.
        imec_to_ni:
            Callable mapping IMEC seconds to NI seconds.
        ni_to_utc:
            Manual flipper UTC mapper or callable mapping NI seconds to Unix
            UTC seconds.
        spike_time_units:
            ``"samples"`` for Kilosort sample indices or ``"seconds"`` for
            pre-converted IMEC seconds.
        bounds_policy:
            How to handle spikes whose NI time is outside barcode bounds.
        bounds_s:
            Optional barcode bounds in NI seconds for callable ``ni_to_utc``.
            Ignored when ``ni_to_utc`` is a :class:`FlipperUtcMapper`.

    Returns:
        DataFrame with one row per kept spike. Columns include ``spike_ix``,
        ``cluster_id``, ``spike_imec_time_s``, ``spike_ni_time_s``,
        ``spike_utc_unix``, ``spike_utc_datetime``, and ``spike_sample_ix`` when
        input units are samples.
    """
    spike_time_array = np.asarray(spike_times)
    spike_cluster_array = np.asarray(spike_clusters, dtype=np.int64)
    validate_spike_times_and_clusters(spike_time_array.reshape(-1), spike_cluster_array.reshape(-1))
    spike_time_array = spike_time_array.reshape(-1)
    spike_cluster_array = spike_cluster_array.reshape(-1)

    spike_imec_time_s, spike_sample_ix = convert_spike_times_to_imec_seconds(
        spike_time_array,
        imec_sample_rate=imec_sample_rate,
        spike_time_units=spike_time_units,
    )
    spike_ni_time_s = np.asarray(imec_to_ni(spike_imec_time_s), dtype=float)

    mapper_bounds = ni_to_utc.bounds_s if isinstance(ni_to_utc, FlipperUtcMapper) else bounds_s
    keep_mask = _manual_bounds_keep_mask(spike_ni_time_s, mapper_bounds, bounds_policy)
    kept_ni_times_s = spike_ni_time_s[keep_mask]
    if isinstance(ni_to_utc, FlipperUtcMapper):
        spike_utc_unix = ni_to_utc.map_ni_times_to_utc(kept_ni_times_s, bounds_policy="allow")
    else:
        spike_utc_unix = np.asarray(ni_to_utc(kept_ni_times_s), dtype=float)

    spike_df = pd.DataFrame(
        {
            "spike_ix": np.flatnonzero(keep_mask).astype(np.int64),
            "cluster_id": spike_cluster_array[keep_mask].astype(np.int64),
            "spike_imec_time_s": spike_imec_time_s[keep_mask],
            "spike_ni_time_s": kept_ni_times_s,
            "spike_utc_unix": spike_utc_unix,
            "spike_utc_datetime": pd.to_datetime(spike_utc_unix, unit="s", utc=True),
        }
    )
    if spike_sample_ix is not None:
        spike_df["spike_sample_ix"] = spike_sample_ix[keep_mask]
    return spike_df


def sync_imec_spikes_to_manual_utc(
    spike_times_npy: Path | str,
    spike_clusters_npy: Path | str,
    imec_sample_rate: float,
    imec_to_ni: Callable[[npt.ArrayLike], np.ndarray],
    ni_to_utc: FlipperUtcMapper | Callable[[npt.ArrayLike], np.ndarray],
    output_file: Path | str | None = None,
    spike_time_units: SpikeTimeUnits = "samples",
    bounds_policy: BoundsPolicy = "reject",
    bounds_s: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Load Kilosort/Phy spikes and save manual-flipper UTC alignment.

    Args:
        spike_times_npy:
            Path to ``spike_times.npy``.
        spike_clusters_npy:
            Path to ``spike_clusters.npy`` aligned one-to-one to
            ``spike_times.npy``.
        imec_sample_rate:
            IMEC AP sampling rate in Hz.
        imec_to_ni:
            Callable mapping IMEC seconds to NI seconds.
        ni_to_utc:
            Manual flipper UTC mapper or callable mapping NI seconds to Unix
            UTC seconds.
        output_file:
            Optional output ``.npz`` path. Uses the same downstream-compatible
            ``spike_utc_unix`` key as the IRIG workflow.
        spike_time_units:
            ``"samples"`` for Kilosort sample indices or ``"seconds"`` for
            pre-converted IMEC seconds.
        bounds_policy:
            How to handle spikes whose NI time is outside barcode bounds.
        bounds_s:
            Optional barcode bounds in NI seconds for callable ``ni_to_utc``.

    Returns:
        DataFrame with mapped spike times and cluster ids.
    """
    spike_times = load_spike_times_npy(spike_times_npy)
    spike_clusters = load_spike_clusters_npy(spike_clusters_npy)
    spike_df = map_imec_spikes_to_utc(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        imec_sample_rate=imec_sample_rate,
        imec_to_ni=imec_to_ni,
        ni_to_utc=ni_to_utc,
        spike_time_units=spike_time_units,
        bounds_policy=bounds_policy,
        bounds_s=bounds_s,
    )

    if output_file is not None:
        arrays = {
            "spike_utc_unix": spike_df["spike_utc_unix"].to_numpy(dtype=float),
            "spike_clusters": spike_df["cluster_id"].to_numpy(dtype=np.int64),
            "spike_imec_time_s": spike_df["spike_imec_time_s"].to_numpy(dtype=float),
            "spike_ni_time_s": spike_df["spike_ni_time_s"].to_numpy(dtype=float),
            "spike_ix": spike_df["spike_ix"].to_numpy(dtype=np.int64),
        }
        if "spike_sample_ix" in spike_df:
            arrays["spike_sample_ix"] = spike_df["spike_sample_ix"].to_numpy(dtype=np.int64)
        meta = {
            "generator": "sync_imec_spikes_to_manual_utc",
            "analysis_version": "0.1.0",
            "sync_method": "manual_line_sync_to_behavior_flipper",
            "utc_reference": "behavior_flipper_time.time()",
            "utc_offset_hours": 0.0,
            "spike_times_file": str(spike_times_npy),
            "spike_clusters_file": str(spike_clusters_npy),
            "imec_sample_rate_hz": float(imec_sample_rate),
            "spike_time_units": spike_time_units,
            "bounds_policy": bounds_policy,
            "units": {
                "spike_sample_ix": "samples",
                "spike_imec_time_s": "seconds",
                "spike_ni_time_s": "seconds",
                "spike_utc_unix": "seconds",
            },
            "axis_convention": "1D spike arrays aligned spike-by-spike to spike_clusters",
        }
        save_stream_sync_npz(output_file=output_file, arrays=arrays, meta=meta)

    return spike_df


def build_imec_to_ni_mapper(
    imec_sync_signal: npt.ArrayLike,
    imec_sample_rate: float,
    ni_ephys_sync_signal: npt.ArrayLike,
    ni_sample_rate: float,
    tolerance_s: float = 0.001,
) -> Callable[[npt.ArrayLike], np.ndarray]:
    """Build an imec-seconds to NI-seconds mapper from shared sync pulses.

    Args:
        imec_sync_signal:
            Digital imec sync line, shape ``(n_imec_samples,)``.
        imec_sample_rate:
            imec sampling rate in Hz.
        ni_ephys_sync_signal:
            Digital NI ephys sync line, shape ``(n_ni_samples,)``.
        ni_sample_rate:
            NI sampling rate in Hz.
        tolerance_s:
            Maximum allowed interval mismatch in seconds during pulse matching.

    Returns:
        Callable mapping imec times in seconds to NI DAQ times in seconds.
    """
    imec_events = get_flipper_events(np.asarray(imec_sync_signal), imec_sample_rate)
    ni_events = get_flipper_events(np.asarray(ni_ephys_sync_signal), ni_sample_rate)
    if imec_events["pos_t"].size >= ni_events["pos_t"].size:
        matched_imec_t, matched_ni_t = match_timestamps_by_intervals(
            imec_events["pos_t"],
            ni_events["pos_t"],
            tolerance_s=tolerance_s,
        )
    else:
        matched_ni_t, matched_imec_t = match_timestamps_by_intervals(
            ni_events["pos_t"],
            imec_events["pos_t"],
            tolerance_s=tolerance_s,
        )
    return make_timebase_mapper(matched_imec_t, matched_ni_t)


def map_ni_digital_rising_edges_to_utc(
    digital_signal: npt.ArrayLike,
    sample_rate: float,
    ni_to_utc: FlipperUtcMapper,
    bounds_policy: BoundsPolicy = "reject",
) -> pd.DataFrame:
    """Map rising edges from an NI digital signal to behavior UTC seconds.

    Args:
        digital_signal:
            One-dimensional NI digital signal, shape ``(n_samples,)``.
        sample_rate:
            NI sampling rate in Hz.
        ni_to_utc:
            Mapper created from behavior flipper timestamps.
        bounds_policy:
            How to handle rising-edge times outside barcode bounds.

    Returns:
        DataFrame with columns:
            ``sample_ix``:
                Rising-edge NI sample indices, units of samples.
            ``time_s``:
                Rising-edge NI times in seconds.
            ``utc_seconds``:
                Behavior-referenced Unix UTC seconds.
            ``utc_datetime``:
                UTC timestamps as timezone-aware ``pandas.Timestamp`` values.
    """
    rising_ix, _ = find_signal_edges(np.asarray(digital_signal, dtype=bool))
    rising_t = rising_ix / float(sample_rate)
    bounded_t = apply_bounds_policy(rising_t, ni_to_utc.bounds_s, bounds_policy)
    keep_mask = np.isin(rising_t, bounded_t)
    utc_seconds = ni_to_utc.map_ni_times_to_utc(rising_t[keep_mask], bounds_policy=bounds_policy)
    return pd.DataFrame(
        {
            "sample_ix": rising_ix[keep_mask],
            "time_s": rising_t[keep_mask],
            "utc_seconds": utc_seconds,
            "utc_datetime": pd.to_datetime(utc_seconds, unit="s", utc=True),
        }
    )


def get_flipper_events(data: np.ndarray, sample_rate: float, debounce: float = 0.002):
    """Extract debounced rising and falling edges from a digital flipper signal.

    Args:
        data:
            One-dimensional digital flipper signal, shape ``(n_samples,)``.
            Values are cast to bool.
        sample_rate:
            Sampling rate in Hz.
        debounce:
            Minimum allowed interval between paired crossings in seconds.
            Crossing pairs closer than this are treated as noise and removed.

    Returns:
        Dictionary containing crossing, positive-edge, and negative-edge sample
        indices and times. Indices are in samples and times are in seconds.
    """
    boolean_signal = data.astype(bool)
    pos_ix, neg_ix = find_signal_edges(boolean_signal)
    crossing_ix = np.sort(np.concatenate((pos_ix, neg_ix)))

    if crossing_ix.size == 0:
        return dict(
            crossing_ix=np.array([], dtype=int),
            crossing_t=np.array([], dtype=float),
            pos_ix=np.array([], dtype=int),
            pos_t=np.array([], dtype=float),
            neg_ix=np.array([], dtype=int),
            neg_t=np.array([], dtype=float),
        )

    crossing_t = crossing_ix / float(sample_rate)
    keep = np.ones(crossing_ix.shape[0], dtype=bool)
    i = 0
    while i < crossing_ix.size - 1:
        if (crossing_t[i + 1] - crossing_t[i]) < debounce:
            keep[i] = False
            keep[i + 1] = False
            i += 2
        else:
            i += 1

    crossing_ix = crossing_ix[keep]
    crossing_t = crossing_ix / float(sample_rate)
    pos_ix = pos_ix[np.isin(pos_ix, crossing_ix)]
    neg_ix = neg_ix[np.isin(neg_ix, crossing_ix)]
    pos_t = pos_ix / float(sample_rate)
    neg_t = neg_ix / float(sample_rate)

    events = dict(
        crossing_ix=crossing_ix,
        crossing_t=crossing_t,
        pos_ix=pos_ix,
        pos_t=pos_t,
        neg_ix=neg_ix,
        neg_t=neg_t,
    )
    return events


def decode_flipper_barcodes(flipper_signal: np.ndarray, sample_rate: float=25000, barcode_present=True, plot=False):
    events = get_flipper_events(flipper_signal, sample_rate)
    if plot:
        from matplotlib import pyplot as plt

        f, ax = plt.subplots()
        x = np.arange(flipper_signal.size) / sample_rate
        start_time = 3600
        end_time = 3700  # seconds
        plt.plot(x[start_time*int(sample_rate):end_time*int(sample_rate)], flipper_signal[start_time*int(sample_rate):end_time*int(sample_rate)])
        plt.title('Flipper signal')
        plt.show()

    if not barcode_present:
        events['flipper_pos_t'] = events['pos_t']
        events['flipper_neg_t'] = events['neg_t']
        return events, (events['crossing_t'][0], events['crossing_t'][-1]), None

    nbits = 32
    wrapper_bit_time = .01  # seconds
    barcode_bit_time = .03  # seconds

    wrapper_time = 3 * wrapper_bit_time  # Off-On-Off
    barcode_time = nbits * barcode_bit_time  # 32 bits
    total_barcode_time = barcode_time + 2 * wrapper_time

    # Tolerance conversions
    tolerance = .2  # % tolerance - so for a duration of 10 and 10% tolerance, 9-11 is acceptable
    min_wrap_duration = wrapper_bit_time - wrapper_bit_time * tolerance
    max_wrap_duration = wrapper_bit_time + wrapper_bit_time * tolerance
    min_bar_duration = barcode_bit_time - barcode_bit_time * tolerance
    max_bar_duration = barcode_bit_time + barcode_bit_time * tolerance
    # sample_conversion = 1000 / expected_sample_rate  # Convert sampling rate to msec

    # events = remove_signal_noise(flipper_signal, events, noise_length=.001)

    wrapper_t = []
    pulse_times = np.diff(events['crossing_t'])
    pulse_time_match = (pulse_times > min_wrap_duration) & (pulse_times < max_wrap_duration)
    pulse_direction_match = flipper_signal[events['crossing_ix']][:-1].astype(bool)
    wrapper_ix = pulse_time_match & pulse_direction_match
    assert np.sum(wrapper_ix) == 4, "File is missing barcodes or barcode wrappers"
    wrapper_pulse_t = events['crossing_t'][:-1][wrapper_ix]
    # plt.show()

    barcode_st = [wrapper_pulse_t[0] + 2*wrapper_bit_time, wrapper_pulse_t[2] + 2*wrapper_bit_time]
    barcode_end = [wrapper_pulse_t[1] - wrapper_bit_time, wrapper_pulse_t[3] - wrapper_bit_time]

    flipper_bounds = [wrapper_pulse_t[1] + max_wrap_duration, wrapper_pulse_t[2] - max_wrap_duration]

    flipper_pos_mask = (events['pos_t'] > flipper_bounds[0]) & (events['pos_t'] < flipper_bounds[1])
    flipper_neg_mask = (events['neg_t'] > flipper_bounds[0]) & (events['neg_t'] < flipper_bounds[1])
    events['flipper_pos_t'] = events['pos_t'][flipper_pos_mask]
    events['flipper_neg_t'] = events['neg_t'][flipper_neg_mask]

    events['barcode_start_t'] = np.array(barcode_st, dtype=float)
    events['barcode_end_t'] = np.array(barcode_end, dtype=float)
    events['barcode_start_ix'] = np.round(events['barcode_start_t'] * sample_rate).astype(int)
    events['barcode_end_ix'] = np.round(events['barcode_end_t'] * sample_rate).astype(int)

    # decode the barcodes
    signals_barcodes = []
    for start, end in zip(barcode_st, barcode_end):
        on_times = (events['pos_t'] > start) & (events['pos_t'] < end)
        off_times = (events['neg_t'] > start) & (events['neg_t'] < end)
        cur_time = start
        bits = np.zeros((nbits,))
        interbit_ON = False  # Changes to "True" during multiple ON bars

        for bit in range(0, nbits):
            next_on = events['pos_t'][on_times][0] if np.any(on_times) else end #start + total_barcode_time
            next_off = events['neg_t'][off_times][0] if np.any(off_times) else end #start + total_barcode_time

            if (cur_time - barcode_bit_time*tolerance) <= next_on <= (cur_time + barcode_bit_time*tolerance):
                bits[bit] = 1
                interbit_ON = True
            elif (cur_time - barcode_bit_time*tolerance) <= next_off <= (cur_time + barcode_bit_time*tolerance):
                interbit_ON = False  # bits default to 0
            elif interbit_ON:
                bits[bit] = 1
            else:
                pass  # bits default to 0

            cur_time += barcode_bit_time
            on_times = (events['pos_t'] > cur_time) & on_times
            off_times = (events['neg_t'] > cur_time) & off_times

        barcode = 0
        for bit in range(0, nbits):
            barcode += bits[bit] * pow(2, nbits-1 - bit)

        signals_barcodes.append(int(barcode))

    signals_barcodes_bitwise = (format(int(signals_barcodes[0]), '0' + str(32) + 'b'), format(int(signals_barcodes[1]), '0' + str(32) + 'b'))
    return events, flipper_bounds, (signals_barcodes, signals_barcodes_bitwise)


def main():
    ### Behavior paths ###
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    raw_ephys_folder = session_data_home / 'ephys/raw/run0_g0'
    output_root = session_data_home / "ephys" / "aligned"
    imec_output_dir = output_root / "aligned_imec"
    sorting_output_name = "Kilosort2.5.2_YYYY-MM-DD_HHMMSS"

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)

    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
    else:
        print("Double-check the session name!")
        return

    session_info_path = raw_behavior_folder / '{}_session_info.pkl'.format(sess_id_full)
    assert session_info_path.exists(), "session_info at {} not found!".format(session_info_path)
    with open(session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    write_manual_alignment_note(output_root)

    behavior_flipper_csv = raw_behavior_folder / f'{sess_id_full}_flipper_output.csv'
    behavior_flipper_times = load_behavior_flipper_timestamps(behavior_flipper_csv)

    # ni file
    ni_file = raw_ephys_folder.joinpath('run0_g0_t0.nidq.bin')
    ni_word = 0
    ni_lines = [0, 1, 2, 3]  # ephys sync/IRIG, flipper, left, right
    daq_lines, daq_srate = read_digital_lines(ni_file, ni_word, ni_lines)
    daq_ephys_sync = daq_lines[0]
    daq_flipper = daq_lines[1]
    daq_leftlick = daq_lines[2]
    daq_rightlick = daq_lines[3]

    flipper_events, flipper_bounds, flipper_barcodes = decode_flipper_barcodes(
        daq_flipper,
        sample_rate=daq_srate,
        barcode_present=True,
        plot=False,
    )
    ni_to_utc = build_flipper_utc_mapper(
        daq_flipper_events=flipper_events,
        behavior_flipper_times=behavior_flipper_times,
        bounds_s=(float(flipper_bounds[0]), float(flipper_bounds[1])),
        bounds_policy="reject",
    )
    print(f"Flipper barcodes: {flipper_barcodes}")
    print(f"Negative-edge residuals (s): {ni_to_utc.negative_edge_residual_s}")

    leftlick_crossings_df = map_ni_digital_rising_edges_to_utc(
        daq_leftlick,
        daq_srate,
        ni_to_utc,
        bounds_policy="reject",
    )
    rightlick_crossings_df = map_ni_digital_rising_edges_to_utc(
        daq_rightlick,
        daq_srate,
        ni_to_utc,
        bounds_policy="reject",
    )
    print("Left lick UTC crossings:")
    print(leftlick_crossings_df.head())
    print("Right lick UTC crossings:")
    print(rightlick_crossings_df.head())

    imec_file = raw_ephys_folder / 'run0_g0_imec0' / 'run0_g0_t0.imec0.ap.bin'
    imec_to_ni = None
    spike_df = None
    if imec_file.exists():
        imec_syncline, imec_srate = read_digital_lines(imec_file, 0, [6])
        imec_to_ni = build_imec_to_ni_mapper(
            imec_sync_signal=imec_syncline,
            imec_sample_rate=imec_srate,
            ni_ephys_sync_signal=daq_ephys_sync,
            ni_sample_rate=daq_srate,
        )
        sorting_output = raw_ephys_folder / 'run0_g0_imec0' / sorting_output_name
        spike_times_file = sorting_output / "spike_times.npy"
        spike_clusters_file = sorting_output / "spike_clusters.npy"
        if spike_times_file.exists() and spike_clusters_file.exists():
            spike_df = sync_imec_spikes_to_manual_utc(
                spike_times_npy=spike_times_file,
                spike_clusters_npy=spike_clusters_file,
                imec_sample_rate=imec_srate,
                imec_to_ni=imec_to_ni,
                ni_to_utc=ni_to_utc,
                output_file=imec_output_dir / "imec0_sync.npz",
                spike_time_units="samples",
                bounds_policy="reject",
            )
            print(f"Mapped {spike_df.shape[0]} imec0 spikes using manual behavior-flipper sync.")
        else:
            print(f"Spike files not found under sorter output: {sorting_output}")

    return {
        "session_info": session_info,
        "flipper_barcodes": flipper_barcodes,
        "flipper_bounds_s": flipper_bounds,
        "ni_to_utc": ni_to_utc,
        "imec_to_ni": imec_to_ni,
        "spike_df": spike_df,
        "leftlick_crossings_df": leftlick_crossings_df,
        "rightlick_crossings_df": rightlick_crossings_df,
    }


if __name__ == "__main__":
    main()
