"""Signed speed from serial offsets and encoded IRIG frames; no I/O or plotting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.irig_tools import irig_core
from src.neural_analysis.synchronization.manual import make_timebase_mapper


@dataclass(frozen=True)
class TreadmillSpeed:
    """Aligned signed speed: S samples and A decoded frame anchors.

    utc_s (S,) is float64 UTC Unix seconds, nondecreasing in source order;
    speed_mm_s (S,) is unchanged signed logged mm/s, including zero. source_rows
    (S,) is int64 parsed-table row position. frame_clock_s/frame_utc_s (A,)
    are source-clock/UTC seconds. meta records units, timing, and diagnostics.
    Equal timestamps retain all original values; no speed samples are inserted.
    """

    utc_s: np.ndarray
    speed_mm_s: np.ndarray
    source_rows: np.ndarray
    frame_clock_s: np.ndarray
    frame_utc_s: np.ndarray
    meta: dict


def reconstruct_treadmill_clock(parsed: pd.DataFrame) -> np.ndarray:
    """Return source-clock seconds (N,) for the N parsed serial messages.

    Required columns: label/value/time_label strings and time_value float
    SECONDS, already converted by irig_serial_io. Sync rows contain intervals
    from the preceding edge; other rows contain offsets from the latest edge.
    Clock zero is the edge preceding the first recorded sync. Leading offsets
    use that same zero. Reject unsupported semantics, resets, or nonfinite/
    negative timing. Input rows/order/values are never modified.
    """
    required = {"label", "value", "time_label", "time_value"}
    if parsed.empty or not required.issubset(parsed.columns):
        raise ValueError("Treadmill input needs nonempty parsed serial records.")
    sync = parsed.label.to_numpy() == "syncPinState"
    if not sync.any():
        raise ValueError("No syncPinState records in treadmill log.")
    times = parsed.time_value.to_numpy(dtype=np.float64)
    labels = parsed.time_label.to_numpy()
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError(
            "Treadmill intervals/offsets must be finite and nonnegative seconds."
        )
    supported = np.where(
        sync,
        np.isin(labels, ("syncIntervalUs", "syncInterval")),
        np.isin(labels, ("timeSincePinChangeUs", "timeSincePinChange")),
    )
    if not supported.all():
        raise ValueError(
            f"Unsupported timing label at parsed rows {np.flatnonzero(~supported).tolist()}."
        )
    # Only edge intervals advance the base clock; payload offsets repeat it.
    edges = np.cumsum(np.where(sync, times, 0.0))
    clock = edges + np.where(sync, 0.0, times)
    if (np.diff(clock) < -1e-9).any():
        row = int(np.flatnonzero(np.diff(clock) < -1e-9)[0] + 1)
        raise ValueError(f"Contradictory offset or timer reset at parsed row {row}.")
    return clock


def align_treadmill_speed(parsed: pd.DataFrame) -> TreadmillSpeed:
    """Align original signed runSpeed payloads using NeuroKairos frame UTC anchors.

    parsed follows reconstruct_treadmill_clock's contract; time_value is seconds
    and runSpeed values are mm/s strings/numbers. Return UTC/value/source-row
    arrays (S,) and frame anchors (A,), preserving zeros, negatives, duplicates,
    and leading samples. At least two increasing frame anchors are required.
    Mapping extrapolates only to actual logged samples at the boundaries.
    PC receipt timestamps, choice windows, and neural bin widths are unused.
    """
    clock = reconstruct_treadmill_clock(parsed)
    labels = parsed.label.to_numpy()
    sync_rows = np.flatnonzero(labels == "syncPinState")
    states = parsed.value.to_numpy()[sync_rows]
    if not np.isin(states, ("HIGH", "LOW")).all() or (states[1:] == states[:-1]).any():
        raise ValueError(
            "Sync transitions must alternate HIGH/LOW without duplicate states."
        )
    # An initial LOW and final HIGH have no completed pulse partner.
    start = 1 if states[0] == "LOW" else 0
    stop = len(states) - (1 if states[-1] == "HIGH" else 0)
    rising = sync_rows[start:stop:2]
    falling = sync_rows[start + 1 : stop : 2]
    if len(rising) == 0:
        raise ValueError("No complete sync pulses in treadmill log.")
    widths = clock[falling] - clock[rising]
    if (widths <= 0).any():
        raise ValueError("Sync pulse widths must be positive.")
    decoded = irig_core.decode_irig_pulses(
        widths,
        source_onset_time_s=clock[rising],
        source_onset_ix=rising,
        irig_format="neurokairos",
        warn_invalid=False,
    )
    frames = decoded.frame_df
    if frames.empty or len(frames) < 2:
        raise ValueError("At least two valid NeuroKairos frame anchors are required.")
    frame_clock = frames.observed_start_time_s.to_numpy(dtype=np.float64)
    frame_utc = frames.decoded_utc_seconds.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(frame_clock).all()
        or not np.isfinite(frame_utc).all()
        or (np.diff(frame_clock) <= 0).any()
        or (np.diff(frame_utc) <= 0).any()
    ):
        raise ValueError(
            "IRIG frame anchors must be finite and strictly increasing in both clocks."
        )
    speed_rows = np.flatnonzero(labels == "runSpeed")
    speeds = pd.to_numeric(parsed.value.iloc[speed_rows], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if len(speeds) == 0 or not np.isfinite(speeds).all():
        raise ValueError("Treadmill log needs finite signed runSpeed values.")
    mapper = make_timebase_mapper(frame_clock, frame_utc)
    utc = mapper(clock[speed_rows])
    if not np.isfinite(utc).all() or (np.diff(utc) < 0).any():
        raise ValueError("Aligned speed timestamps must be finite and nondecreasing.")
    meta = {
        "generator": "mouse_behavior_preprocessing.treadmill_speed.align_treadmill_speed",
        "units": {
            "utc_s": "UTC Unix seconds",
            "speed_mm_s": "signed mm/s",
            "frame_clock_s": "source-clock seconds",
            "frame_utc_s": "UTC Unix seconds",
            "source_rows": "dimensionless parsed-row positions",
        },
        "axes": {
            "utc_s": ["speed_sample"],
            "speed_mm_s": ["speed_sample"],
            "frame_clock_s": ["decoded_frame"],
            "frame_utc_s": ["decoded_frame"],
            "source_rows": ["speed_sample"],
        },
        "sampling": "irregular logged updates; no resampling",
        "parsed_records": len(parsed),
        "speed_samples": len(speeds),
        "frame_anchors": len(frames),
        "incomplete_boundary_pulses": start + (len(states) - stop),
        "invalid_pulse_indices": decoded.invalid_pulse_ix.tolist(),
        "source_extent_s": [float(clock[0]), float(clock[-1])],
        "speed_utc_extent_s": [float(utc[0]), float(utc[-1])],
        "clock_seconds_per_utc_second": float(
            (frame_clock[-1] - frame_clock[0]) / (frame_utc[-1] - frame_utc[0])
        ),
        "frame_source_intervals_s": np.diff(frame_clock).tolist(),
        "frame_utc_intervals_s": np.diff(frame_utc).tolist(),
    }
    return TreadmillSpeed(utc, speeds, speed_rows, frame_clock, frame_utc, meta)
