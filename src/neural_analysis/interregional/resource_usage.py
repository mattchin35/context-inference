"""Durable, low-overhead resource telemetry for long inter-regional runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import threading
import time
from types import TracebackType
from typing import Any


_SCHEMA_VERSION = "1"
_CGROUP_KEYS = (
    "cgroup_memory_current_bytes",
    "cgroup_memory_peak_bytes",
    "cgroup_memory_limit_bytes",
    "cgroup_memory_events_high",
    "cgroup_memory_events_max",
    "cgroup_memory_events_oom",
    "cgroup_memory_events_oom_kill",
)


def linux_peak_rss_bytes(ru_maxrss_kib: float) -> int:
    """Convert Linux ``ru_maxrss`` from KiB to a nonnegative byte count.

    Parameters
    ----------
    ru_maxrss_kib : float
        Process peak resident-set size in Linux kibibytes.

    Returns
    -------
    int
        Process peak resident-set size in bytes.
    """
    value = float(ru_maxrss_kib)
    if not math.isfinite(value) or value < 0:
        raise ValueError("ru_maxrss must be finite nonnegative KiB.")
    return int(value * 1024)


def _read_nonnegative_integer(path: Path, *, unlimited: bool = False) -> int | None:
    """Return one nonnegative integer file value or null when unavailable.

    ``path`` is a filesystem coordinate. Returned integers retain the file's
    units; ``unlimited=True`` also maps the cgroup token ``max`` to null.
    """
    try:
        text = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if unlimited and text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def read_cgroup_v2_memory(cgroup_directory: Path | None) -> dict[str, int | None]:
    """Read memory usage and OOM counters from one cgroup-v2 directory.

    Parameters
    ----------
    cgroup_directory : pathlib.Path or None
        Process cgroup directory. Memory quantities are stored in bytes and
        event fields are unitless counters. Missing data are returned as null.

    Returns
    -------
    dict[str, int or None]
        Stable memory-current, peak, limit, high, max, OOM, and OOM-kill keys.
    """
    values = dict.fromkeys(_CGROUP_KEYS)
    if cgroup_directory is None:
        return values
    directory = Path(cgroup_directory)
    values.update(
        {
            "cgroup_memory_current_bytes": _read_nonnegative_integer(
                directory / "memory.current"
            ),
            "cgroup_memory_peak_bytes": _read_nonnegative_integer(
                directory / "memory.peak"
            ),
            "cgroup_memory_limit_bytes": _read_nonnegative_integer(
                directory / "memory.max", unlimited=True
            ),
        }
    )
    try:
        event_lines = (directory / "memory.events").read_text(
            encoding="ascii"
        ).splitlines()
    except OSError:
        event_lines = []
    parsed_events: dict[str, int | None] = {}
    for line in event_lines:
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            value = int(fields[1])
        except ValueError:
            value = -1
        parsed_events[fields[0]] = value if value >= 0 else None
    for event_name in ("high", "max", "oom", "oom_kill"):
        values[f"cgroup_memory_events_{event_name}"] = parsed_events.get(event_name)
    return values


def resolve_cgroup_v2_directory(
    proc_cgroup_path: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Path | None:
    """Resolve this process's unified cgroup-v2 filesystem directory.

    Parameters
    ----------
    proc_cgroup_path : pathlib.Path
        Text file containing Linux process cgroup membership.
    cgroup_root : pathlib.Path
        Mounted cgroup-v2 filesystem root.

    Returns
    -------
    pathlib.Path or None
        Existing process cgroup directory, or null when no unified membership
        can be resolved. Paths contain no physical-unit values.
    """
    try:
        lines = Path(proc_cgroup_path).read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    root = Path(cgroup_root).resolve()
    for line in lines:
        fields = line.split(":", maxsplit=2)
        if len(fields) != 3 or fields[0] != "0" or fields[1] != "":
            continue
        relative = Path(fields[2].lstrip("/"))
        candidate = (root / relative).resolve()
        if candidate != root and not candidate.is_relative_to(root):
            return None
        return candidate if candidate.is_dir() else None
    return None


def sample_resources(cgroup_directory: Path | None) -> dict[str, int | float | None]:
    """Return one process and cgroup resource snapshot.

    Parameters
    ----------
    cgroup_directory : pathlib.Path or None
        Resolved cgroup-v2 directory. Memory fields are bytes; CPU fields are
        seconds; event values are unitless cumulative counters.

    Returns
    -------
    dict[str, int or float or None]
        Process CPU, process peak RSS, and cgroup-v2 memory measurements.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_cpu_user_seconds": float(usage.ru_utime),
        "process_cpu_system_seconds": float(usage.ru_stime),
        "process_peak_rss_bytes": linux_peak_rss_bytes(usage.ru_maxrss),
        **read_cgroup_v2_memory(cgroup_directory),
    }


class ResourceMonitor:
    """Append resource/progress snapshots and summarize a completed run.

    ``resource_trace.jsonl`` is appended one complete UTF-8 JSON record at a
    time so records already handed to the kernel survive abrupt process death.
    Heartbeat intervals and elapsed/CPU values are seconds; memory values are
    bytes. The monitor owns no scientific arrays.
    """

    def __init__(
        self,
        trace_path: Path,
        summary_path: Path,
        *,
        interval_seconds: float = 30.0,
        cgroup_directory: Path | None = None,
    ) -> None:
        """Configure paths, heartbeat seconds, and optional cgroup directory."""
        interval = float(interval_seconds)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval_seconds must be finite and positive.")
        self.trace_path = Path(trace_path)
        self.summary_path = Path(summary_path)
        self.interval_seconds = interval
        self.cgroup_directory = (
            resolve_cgroup_v2_directory()
            if cgroup_directory is None
            else Path(cgroup_directory)
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._file_descriptor: int | None = None
        self._started_monotonic: float | None = None
        self._started_utc: str | None = None
        self._sequence = 0
        self._max_current_bytes: int | None = None
        self._max_peak_bytes: int | None = None
        self._max_process_peak_bytes: int | None = None

    @property
    def is_running(self) -> bool:
        """Return whether the heartbeat thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Open the append-only trace and start one daemon heartbeat thread."""
        if self._file_descriptor is not None:
            raise RuntimeError("ResourceMonitor is already started.")
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_descriptor = os.open(
            self.trace_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        self._started_monotonic = time.monotonic()
        self._started_utc = _utc_now()
        self._stop_event.clear()
        self._append_record(
            "resource",
            resource_event="monitor_start",
            heartbeat_interval_seconds=self.interval_seconds,
            cgroup_directory=(
                str(self.cgroup_directory)
                if self.cgroup_directory is not None
                else None
            ),
        )
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="interregional-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def record_progress(self, progress_event: str, **details: Any) -> None:
        """Append one progress event with an immediate resource snapshot.

        Parameters
        ----------
        progress_event : str
            Stable execution-boundary name.
        **details : object
            JSON-compatible scalar/list metadata. Shapes are integer lists,
            byte fields are integer bytes, and row/target counts are unitless.

        Returns
        -------
        None
            The trace is changed in place; scientific data are not returned.
        """
        if not isinstance(progress_event, str) or not progress_event:
            raise ValueError("progress_event must be a nonempty string.")
        reserved = {
            "schema_version",
            "sequence",
            "record_kind",
            "progress_event",
            "timestamp_utc",
            "elapsed_seconds",
            "pid",
        }
        if reserved & set(details):
            raise ValueError("Progress details contain reserved telemetry keys.")
        self._append_record(
            "progress", progress_event=progress_event, **details
        )

    def stop(self, *, completed: bool) -> None:
        """Stop heartbeats and optionally write an atomic completion summary.

        Parameters
        ----------
        completed : bool
            True only after every run artifact is produced normally. A false
            value closes the trace without creating a summary.

        Returns
        -------
        None
            The heartbeat thread and trace descriptor are closed.
        """
        if self._file_descriptor is None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
            if self._thread.is_alive():
                raise RuntimeError("Resource monitor heartbeat thread did not stop.")
        self._thread = None
        self._append_record("resource", resource_event="monitor_stop")
        descriptor = self._file_descriptor
        self._file_descriptor = None
        os.close(descriptor)
        if completed:
            self._write_summary()

    def __enter__(self) -> ResourceMonitor:
        """Start monitoring and return this context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Stop monitoring and summarize only an exception-free context."""
        self.stop(completed=exception_type is None)
        return False

    # Helpers

    def _heartbeat_loop(self) -> None:
        """Append snapshots until the stop event interrupts the interval wait."""
        while not self._stop_event.wait(self.interval_seconds):
            self._append_record("resource", resource_event="heartbeat")

    def _append_record(self, record_kind: str, **details: Any) -> None:
        """Build and append one complete JSON record under the sequence lock."""
        if self._file_descriptor is None or self._started_monotonic is None:
            raise RuntimeError("ResourceMonitor must be started before recording.")
        snapshot = sample_resources(self.cgroup_directory)
        with self._lock:
            record = {
                "schema_version": _SCHEMA_VERSION,
                "sequence": self._sequence,
                "record_kind": record_kind,
                "timestamp_utc": _utc_now(),
                "elapsed_seconds": time.monotonic() - self._started_monotonic,
                "pid": os.getpid(),
                **snapshot,
                **details,
            }
            encoded = (
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(self._file_descriptor, remaining)
                if written <= 0:
                    raise OSError("Unable to append resource trace record.")
                remaining = remaining[written:]
            self._sequence += 1
            self._update_maxima(record)

    def _update_maxima(self, record: dict[str, Any]) -> None:
        """Update byte maxima from one successfully appended record."""
        pairs = (
            ("cgroup_memory_current_bytes", "_max_current_bytes"),
            ("cgroup_memory_peak_bytes", "_max_peak_bytes"),
            ("process_peak_rss_bytes", "_max_process_peak_bytes"),
        )
        for key, attribute in pairs:
            value = record.get(key)
            if isinstance(value, int):
                prior = getattr(self, attribute)
                setattr(self, attribute, value if prior is None else max(prior, value))

    def _write_summary(self) -> None:
        """Atomically write aggregate telemetry after ordinary completion."""
        assert self._started_monotonic is not None
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "completed": True,
            "started_at_utc": self._started_utc,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - self._started_monotonic,
            "record_count": self._sequence,
            "max_process_peak_rss_bytes": self._max_process_peak_bytes,
            "max_cgroup_memory_current_bytes": self._max_current_bytes,
            "max_cgroup_memory_peak_bytes": self._max_peak_bytes,
        }
        temporary = self.summary_path.with_name(self.summary_path.name + ".tmp")
        with temporary.open("w", encoding="ascii") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.summary_path)


def _utc_now() -> str:
    """Return an RFC-3339 UTC timestamp with a trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
