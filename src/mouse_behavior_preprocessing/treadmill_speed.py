"""Signed speed from serial offsets and encoded IRIG frames; no I/O or plotting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.irig_tools import irig_core
from src.neural_analysis.synchronization.manual import make_timebase_mapper

MAX_REJECTED_INTERVAL_S = 0.05


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
    use that same zero. A reset-offset suffix immediately before a printed
    sync edge becomes NaN only when retained speed neighbors are under 50 ms
    apart. Other resets, unsupported or nonfinite/negative timing raise.
    Input rows/order/values are never modified; no timestamp is reassigned.
    """
    clock, _ = _clock_and_rejections(parsed)
    return clock


def _clock_and_rejections(parsed: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    """Return (N,) source seconds/NaNs and a ledger of short-gap rejected speeds.

    Parsed input follows reconstruct_treadmill_clock. Only a decreasing-offset
    suffix directly before the next sync record is eligible; every omitted
    speed requires retained neighbors less than MAX_REJECTED_INTERVAL_S apart.
    Sync rows stay intact. Ledger row indices are zero-based parsed positions.
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
    sync_rows = np.flatnonzero(sync)
    groups = []
    for row in np.flatnonzero(np.diff(clock) < -1e-9) + 1:
        following = sync_rows[sync_rows > row]
        if sync[row] or not len(following):
            raise ValueError(
                f"Contradictory offset or timer reset at parsed row {row}."
            )
        edge = int(following[0])
        suffix = times[row:edge]
        if (np.diff(suffix) < -1e-9).any() or suffix.max() >= times[row - 1]:
            raise ValueError(f"Ambiguous timer reset at parsed row {row}.")
        groups.append((int(row), edge))
    # Leave unknown times absent rather than attach them to a guessed sync edge.
    for row, edge in groups:
        clock[row:edge] = np.nan
    retained_clock = clock[np.isfinite(clock)]
    if (np.diff(retained_clock) < -1e-9).any():
        raise ValueError("Contradictory retained treadmill clock.")
    speed = parsed.label.to_numpy() == "runSpeed"
    retained_rows = np.flatnonzero(speed & np.isfinite(clock))
    rejected = []
    for row, edge in groups:
        before = retained_rows[retained_rows < row]
        after = retained_rows[retained_rows > edge]
        if not len(before) or not len(after):
            raise ValueError(
                "Speed rejection requires retained readings on both sides."
            )
        previous, following = int(before[-1]), int(after[0])
        interval = float(clock[following] - clock[previous])
        if not 0 <= interval < MAX_REJECTED_INTERVAL_S:
            raise ValueError("Speed rejection interval must be less than 50 ms.")
        for rejected_row in np.flatnonzero(speed[row:edge]) + row:
            rejected.append(
                {
                    "source_row": int(rejected_row),
                    "logged_speed_mm_s": float(parsed.value.iloc[rejected_row]),
                    "offset_s": float(times[rejected_row]),
                    "following_sync_row": edge,
                    "previous_retained_row": previous,
                    "next_retained_row": following,
                    "retained_interval_s": interval,
                    "reason": "offset_reset_before_sync_record",
                }
            )
    return clock, rejected


def align_treadmill_speed(parsed: pd.DataFrame) -> TreadmillSpeed:
    """Align original signed runSpeed payloads using NeuroKairos frame UTC anchors.

    parsed follows reconstruct_treadmill_clock's contract; time_value is seconds
    and runSpeed values are mm/s strings/numbers. Return UTC/value/source-row
    arrays (S,) and frame anchors (A,), preserving zeros, negatives, duplicates,
    and leading samples. Only the documented short-gap ambiguous speed records
    are omitted, with their original positions/values recorded in meta.
    At least two increasing frame anchors are required.
    Mapping extrapolates only to actual logged samples at the boundaries.
    PC receipt timestamps, choice windows, and neural bin widths are unused.
    """
    clock, rejected = _clock_and_rejections(parsed)
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
    valid = np.isfinite(clock[speed_rows])
    speed_rows, speeds = speed_rows[valid], speeds[valid]
    mapper = make_timebase_mapper(frame_clock, frame_utc)
    for record in rejected:
        bounds = mapper(
            clock[[record["previous_retained_row"], record["next_retained_row"]]]
        )
        record["retained_interval_utc_s"] = float(bounds[1] - bounds[0])
        if not 0 <= record["retained_interval_utc_s"] < MAX_REJECTED_INTERVAL_S:
            raise ValueError(
                "Aligned speed rejection interval must be less than 50 ms."
            )
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
        "rejected_speed_rows": [record["source_row"] for record in rejected],
        "rejected_speed_records": rejected,
        "ambiguous_source_rows": np.flatnonzero(~np.isfinite(clock)).tolist(),
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
