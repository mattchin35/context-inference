"""Contracts for durable inter-regional execution resource telemetry."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.neural_analysis.interregional.resource_usage import (
    ResourceMonitor,
    linux_peak_rss_bytes,
    read_cgroup_v2_memory,
    resolve_cgroup_v2_directory,
)


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    """Return complete JSON objects from one UTF-8 JSONL filesystem path.

    Parameters
    ----------
    path : pathlib.Path
        Existing trace path. File size is measured in bytes.

    Returns
    -------
    list[dict[str, object]]
        Ordered JSON objects; no scientific arrays or physical-unit axes are
        transformed.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_linux_peak_rss_is_normalized_from_kibibytes_to_bytes() -> None:
    """Linux ru_maxrss values become exact nonnegative integer byte counts."""
    assert linux_peak_rss_bytes(123) == 123 * 1024
    assert linux_peak_rss_bytes(0) == 0
    for invalid in (-1, math.inf, math.nan):
        with pytest.raises(ValueError, match="finite nonnegative"):
            linux_peak_rss_bytes(invalid)


def test_cgroup_v2_memory_files_and_events_are_parsed(tmp_path: Path) -> None:
    """Current, peak, limit, and OOM counters retain their documented units."""
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("1024\n", encoding="ascii")
    (cgroup / "memory.peak").write_text("4096\n", encoding="ascii")
    (cgroup / "memory.max").write_text("max\n", encoding="ascii")
    (cgroup / "memory.events").write_text(
        "low 0\nhigh 3\nmax 4\noom 2\noom_kill 1\n",
        encoding="ascii",
    )

    values = read_cgroup_v2_memory(cgroup)

    assert values == {
        "cgroup_memory_current_bytes": 1024,
        "cgroup_memory_peak_bytes": 4096,
        "cgroup_memory_limit_bytes": None,
        "cgroup_memory_events_high": 3,
        "cgroup_memory_events_max": 4,
        "cgroup_memory_events_oom": 2,
        "cgroup_memory_events_oom_kill": 1,
    }


def test_cgroup_v2_missing_malformed_and_unlimited_values_are_explicit(
    tmp_path: Path,
) -> None:
    """Unavailable cgroup values become null rather than invented measurements."""
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("not-a-number\n", encoding="ascii")
    (cgroup / "memory.max").write_text("max\n", encoding="ascii")
    (cgroup / "memory.events").write_text("oom invalid\noom_kill -1\n", encoding="ascii")

    values = read_cgroup_v2_memory(cgroup)

    assert set(values) == {
        "cgroup_memory_current_bytes",
        "cgroup_memory_peak_bytes",
        "cgroup_memory_limit_bytes",
        "cgroup_memory_events_high",
        "cgroup_memory_events_max",
        "cgroup_memory_events_oom",
        "cgroup_memory_events_oom_kill",
    }
    assert all(value is None for value in values.values())


def test_current_cgroup_v2_directory_resolves_process_membership(tmp_path: Path) -> None:
    """The unified hierarchy path is resolved beneath its supplied mount root."""
    proc_cgroup = tmp_path / "proc-self-cgroup"
    proc_cgroup.write_text("0::/slurm/job_123/step_batch\n", encoding="ascii")
    cgroup_root = tmp_path / "sys-fs-cgroup"
    expected = cgroup_root / "slurm" / "job_123" / "step_batch"
    expected.mkdir(parents=True)

    resolved = resolve_cgroup_v2_directory(proc_cgroup, cgroup_root)

    assert resolved == expected


def test_monitor_writes_ordered_trace_and_atomic_completion_summary(
    tmp_path: Path,
) -> None:
    """Synchronous progress samples and final summary use a stable JSON schema."""
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    for filename, value in (
        ("memory.current", "2048\n"),
        ("memory.peak", "8192\n"),
        ("memory.max", "16384\n"),
        ("memory.events", "high 0\nmax 0\noom 0\noom_kill 0\n"),
    ):
        (cgroup / filename).write_text(value, encoding="ascii")
    trace = tmp_path / "resource_trace.jsonl"
    summary = tmp_path / "resource_summary.json"
    monitor = ResourceMonitor(
        trace,
        summary,
        interval_seconds=60.0,
        cgroup_directory=cgroup,
    )

    monitor.start()
    monitor.record_progress(
        "analysis_cell_start",
        stage="poisson_cv",
        direction="HPC_to_PFC",
        fold_id=2,
    )
    monitor.stop(completed=True)

    records = _read_json_lines(trace)
    assert [record["sequence"] for record in records] == list(range(len(records)))
    assert records[0]["record_kind"] == "resource"
    progress = next(record for record in records if record["record_kind"] == "progress")
    assert progress["progress_event"] == "analysis_cell_start"
    assert progress["direction"] == "HPC_to_PFC"
    assert progress["fold_id"] == 2
    for record in records:
        assert record["schema_version"] == "1"
        assert record["timestamp_utc"].endswith("Z")
        for key, value in record.items():
            if key.endswith(("_bytes", "_seconds")) or key in {
                "sequence",
                "pid",
                "cgroup_memory_events_high",
                "cgroup_memory_events_max",
                "cgroup_memory_events_oom",
                "cgroup_memory_events_oom_kill",
            }:
                assert value is None or (math.isfinite(value) and value >= 0)
    assert not monitor.is_running
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["completed"] is True
    assert payload["record_count"] == len(records)
    assert payload["max_cgroup_memory_current_bytes"] == 2048
    assert payload["max_cgroup_memory_peak_bytes"] == 8192
    assert not (tmp_path / "resource_summary.json.tmp").exists()


def test_monitor_exception_stops_thread_and_preserves_trace(tmp_path: Path) -> None:
    """An exceptional context has durable records but no completion summary."""
    trace = tmp_path / "resource_trace.jsonl"
    summary = tmp_path / "resource_summary.json"
    monitor = ResourceMonitor(trace, summary, interval_seconds=0.02)

    with pytest.raises(RuntimeError, match="synthetic"):
        with monitor:
            monitor.record_progress("stage_start", stage="poisson_cv")
            raise RuntimeError("synthetic")

    assert not monitor.is_running
    assert len(_read_json_lines(trace)) >= 2
    assert not summary.exists()


def test_sigkill_leaves_complete_parseable_trace_records(tmp_path: Path) -> None:
    """Previously appended heartbeat lines survive abrupt subprocess termination."""
    trace = tmp_path / "resource_trace.jsonl"
    summary = tmp_path / "resource_summary.json"
    ready = tmp_path / "ready"
    script = "\n".join(
        (
            "import pathlib, time",
            "from src.neural_analysis.interregional.resource_usage import ResourceMonitor",
            f"monitor = ResourceMonitor(pathlib.Path({str(trace)!r}), pathlib.Path({str(summary)!r}), interval_seconds=0.02)",
            "monitor.start()",
            f"pathlib.Path({str(ready)!r}).write_text('ready', encoding='ascii')",
            "while True:",
            "    time.sleep(1.0)",
        )
    )
    process = subprocess.Popen((sys.executable, "-c", script), cwd=Path.cwd())
    deadline = time.monotonic() + 5.0
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.is_file()
    time.sleep(0.08)

    process.kill()
    process.wait(timeout=5.0)

    records = _read_json_lines(trace)
    assert len(records) >= 2
    assert [record["sequence"] for record in records] == list(range(len(records)))
    assert all(record["record_kind"] == "resource" for record in records)
    assert not summary.exists()
