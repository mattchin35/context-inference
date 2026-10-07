"""WP6 contracts for bounded task-decoding batch planning and execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from src.neural_analysis.task_decoding import run_batch


_THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def write_config_list(tmp_path: Path, names: tuple[str, ...]) -> Path:
    """Write an ordered UTF-8 batch list of paths relative to the list file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory receiving the list and empty config files.
    names : tuple[str, ...]
        Relative configuration filenames in declared session order.

    Returns
    -------
    pathlib.Path
        Config-list path containing comments and blank lines around ``names``.
    """
    for name in names:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    lines = ["# ordered sessions", "", *(f"  {name}  " for name in names)]
    config_list = tmp_path / "configs.txt"
    config_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_list


def resource_envelope(*, tensor_bytes: int = 100) -> dict[str, int]:
    """Return one internally coherent synthetic resource envelope.

    Parameters
    ----------
    tensor_bytes : int, default=100
        Exact simultaneous float64 regional-tensor allocation in bytes.

    Returns
    -------
    dict[str, int]
        Dimensionless/session-size drivers plus the byte-valued tensor member.
    """
    return {
        "full_trial_count": 12,
        "tensor_trial_count": 10,
        "pfc_unit_count": 2,
        "hpc_unit_count": 2,
        "time_bin_count": 8,
        "target_count": 2,
        "outer_fold_count": 3,
        "inner_fold_count": 3,
        "coefficient_feature_capacity": 4,
        "categorical_fit_count": 144,
        "numerical_fit_count": 144,
        "tensor_allocation_bytes": tensor_bytes,
    }


def session_plan(config_path: Path, *, tensor_bytes: int = 100) -> dict[str, object]:
    """Build one bounded fake returned by ``plan_task_decoding_session``.

    Parameters
    ----------
    config_path : pathlib.Path
        Existing synthetic config path used for stable session identity.
    tensor_bytes : int, default=100
        Exact regional-tensor allocation in bytes.

    Returns
    -------
    dict[str, object]
        JSON-safe dry-run plan with identity and resource-envelope facts.
    """
    return {
        "config_path": str(config_path),
        "session_id": config_path.stem,
        "target_names": ("current_action", "relative_doubt"),
        "fit_count": 288,
        "tensor_allocation_bytes": tensor_bytes,
        "source_file_sizes_bytes": {},
        "source_cleanliness": "clean",
        "analysis_version": "task-variable-decoding-v1",
        "scientific_source_fingerprint": "source-fingerprint",
        "runtime_versions": {"python": "3.12"},
        "platform": "Linux-test",
        "architecture": "x86_64",
        "thread_limits": dict(_THREAD_LIMITS),
        "regularization_mode": "fixed",
        "resource_envelope": resource_envelope(tensor_bytes=tensor_bytes),
    }


class FakePipeline:
    """Small pipeline seam recording batch planning and execution calls."""

    def __init__(self, plans: dict[Path, dict[str, object]]) -> None:
        """Initialize path-keyed plans and empty lifecycle call records.

        Parameters
        ----------
        plans : dict[pathlib.Path, dict[str, object]]
            Exact resolved config paths and bounded plans returned for them.
        """
        self.plans = plans
        self.planned: list[Path] = []
        self.prepared: list[tuple[Path, bool, str]] = []
        self.executed: list[Path] = []
        self.batch_provenance: list[tuple[Path, dict[str, object]]] = []
        self.completed: set[Path] = set()
        self.fail_execution_for: set[Path] = set()

    def plan_task_decoding_session(self, config_path: Path | str) -> dict[str, object]:
        """Return one bounded plan without scientific array loading."""
        resolved = Path(config_path).resolve()
        self.planned.append(resolved)
        return dict(self.plans[resolved])

    def prepare_task_decoding_run(
        self,
        config_path: Path | str,
        rerun: bool,
        execution_mode: str,
    ) -> Path:
        """Create one fake independent run directory for a config."""
        resolved = Path(config_path).resolve()
        self.prepared.append((resolved, rerun, execution_mode))
        run_directory = resolved.parent / f"run-{resolved.stem}"
        run_directory.mkdir(exist_ok=True)
        return run_directory

    def _validate_complete_prepared_run(self, run_directory: Path) -> bool:
        """Return whether a fake run was declared complete by its test."""
        return Path(run_directory) in self.completed

    def _record_batch_execution_provenance(
        self,
        run_directory: Path,
        batch_provenance: dict[str, object],
    ) -> None:
        """Record the exact per-run worker and evidence calculation."""
        self.batch_provenance.append((Path(run_directory), dict(batch_provenance)))

    def run_prepared_task_decoding(self, run_directory: Path) -> None:
        """Record execution or raise one injected independent failure."""
        directory = Path(run_directory)
        self.executed.append(directory)
        if directory in self.fail_execution_for:
            raise RuntimeError(f"synthetic failure for {directory.name}")


def write_resource_evidence(
    run_directory: Path,
    *,
    peak_rss_bytes: int = 200,
    envelope: dict[str, int] | None = None,
    source_fingerprint: str = "source-fingerprint",
) -> None:
    """Write one explicit completed measured-run evidence fixture.

    Parameters
    ----------
    run_directory : pathlib.Path
        Evidence directory receiving execution/source/config/resource sidecars.
    peak_rss_bytes : int, default=200
        Measured peak resident memory in bytes.
    envelope : dict[str, int] or None, default=None
        Measured session envelope; the default uses :func:`resource_envelope`.
    source_fingerprint : str, default="source-fingerprint"
        Scoped scientific-source identity recorded by the evidence run.

    Returns
    -------
    None
        Writes only small JSON files; no result arrays or source data are read.
    """
    run_directory.mkdir()
    (run_directory / "config.json").write_text(
        json.dumps(
            {
                "analysis_version": "task-variable-decoding-v1",
                "regularization_mode": "fixed",
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "scientific_source.json").write_text(
        json.dumps({"fingerprint": source_fingerprint}),
        encoding="utf-8",
    )
    (run_directory / "execution.json").write_text(
        json.dumps(
            {
                "runtime_versions": {"python": "3.12"},
                "platform": "Linux-test",
                "architecture": "x86_64",
                "thread_limits": _THREAD_LIMITS,
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "resource_usage.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "measurement_method": "resource.getrusage",
                "peak_rss_bytes": peak_rss_bytes,
                "resource_envelope": envelope or resource_envelope(),
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "run_fingerprint.txt").write_text(
        "measured-run-fingerprint\n",
        encoding="ascii",
    )


def parse_stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    """Decode the batch CLI's sole JSON stdout record."""
    output = capsys.readouterr().out
    return json.loads(output)


def test_config_list_ignores_comments_resolves_relative_paths_and_preserves_order(tmp_path):
    """Config-list parsing is portable, deterministic, and list-relative."""
    config_list = write_config_list(tmp_path, ("b.json", "a.json"))
    assert run_batch.read_config_list(config_list) == (
        (tmp_path / "b.json").resolve(),
        (tmp_path / "a.json").resolve(),
    )


def test_config_list_rejects_duplicate_resolved_paths(tmp_path):
    """Aliased duplicate sessions cannot be launched twice accidentally."""
    config = tmp_path / "session.json"
    config.write_text("{}\n", encoding="utf-8")
    config_list = tmp_path / "configs.txt"
    config_list.write_text("session.json\n./session.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        run_batch.read_config_list(config_list)


def test_resource_envelope_allows_zero_fits_for_an_absent_target_family():
    """A categorical-only or numerical-only run has one legitimate zero fit count."""
    envelope = resource_envelope()
    envelope["numerical_fit_count"] = 0
    assert run_batch._validate_resource_envelope(envelope, "test") == envelope


def test_dry_run_reports_ordered_sessions_and_safe_one_worker_without_evidence(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Dry-run is bounded and defaults to one session without measured evidence."""
    config_list = write_config_list(tmp_path, ("first.json", "second.json"))
    configs = run_batch.read_config_list(config_list)
    pipeline = FakePipeline({path: session_plan(path) for path in configs})
    monkeypatch.setattr(run_batch, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(run_batch, "_read_mem_available_bytes", lambda: 10_000)
    monkeypatch.setattr(run_batch.os, "cpu_count", lambda: 8)

    assert run_batch.main(["dry-run", "--config-list", str(config_list), "--workers", "6"]) == 0
    report = parse_stdout_json(capsys)
    assert report["mode"] == "dry-run"
    assert report["requested_workers"] == 6
    assert report["effective_workers"] == 1
    assert report["resource_evidence"] is None
    assert [item["config_path"] for item in report["sessions"]] == [
        str(path) for path in configs
    ]
    assert [item["fit_count"] for item in report["sessions"]] == [288, 288]
    assert [item["tensor_allocation_bytes"] for item in report["sessions"]] == [100, 100]
    assert pipeline.prepared == []
    assert pipeline.executed == []


@pytest.mark.parametrize(
    ("damage", "match"),
    (
        ("incomplete", "complete|evidence"),
        ("missing_usage", "resource|usage|evidence"),
        ("missing_fingerprint", "fingerprint|evidence"),
        ("source_mismatch", "source|identity|evidence"),
        ("runtime_mismatch", "runtime|identity|evidence"),
        ("platform_mismatch", "platform|identity|evidence"),
        ("thread_mismatch", "thread|identity|evidence"),
        ("mode_mismatch", "mode|regularization|identity|evidence"),
        ("envelope_too_small", "envelope|exceed|evidence"),
    ),
)
def test_explicit_invalid_resource_evidence_errors_without_fallback(
    monkeypatch,
    tmp_path,
    capsys,
    damage,
    match,
):
    """An explicit invalid evidence directory never silently selects one worker."""
    config_list = write_config_list(tmp_path, ("session.json",))
    config = run_batch.read_config_list(config_list)[0]
    pipeline = FakePipeline({config: session_plan(config)})
    evidence = tmp_path / "evidence"
    envelope = resource_envelope()
    if damage == "envelope_too_small":
        envelope["tensor_trial_count"] = 1
    write_resource_evidence(
        evidence,
        envelope=envelope,
        source_fingerprint=("different" if damage == "source_mismatch" else "source-fingerprint"),
    )
    pipeline.completed.add(evidence)
    if damage == "incomplete":
        pipeline.completed.clear()
    elif damage == "missing_usage":
        (evidence / "resource_usage.json").unlink()
    elif damage == "missing_fingerprint":
        (evidence / "run_fingerprint.txt").unlink()
    elif damage in {"runtime_mismatch", "platform_mismatch", "thread_mismatch"}:
        execution_path = evidence / "execution.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if damage == "runtime_mismatch":
            execution["runtime_versions"] = {"python": "different"}
        elif damage == "platform_mismatch":
            execution["platform"] = "different"
        else:
            execution["thread_limits"] = {**_THREAD_LIMITS, "OMP_NUM_THREADS": "2"}
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
    elif damage == "mode_mismatch":
        config_path = evidence / "config.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload["regularization_mode"] = "tuned"
        config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.setattr(run_batch, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(run_batch, "_read_mem_available_bytes", lambda: 10_000)
    monkeypatch.setattr(run_batch.os, "cpu_count", lambda: 8)

    assert run_batch.main(
        [
            "dry-run",
            "--config-list",
            str(config_list),
            "--resource-run-directory",
            str(evidence),
        ]
    ) == 1
    assert re.search(match, capsys.readouterr().err, flags=re.IGNORECASE)
    assert pipeline.prepared == []


@pytest.mark.parametrize(
    ("worker_arguments", "expected_requested"),
    (((), 8), (("--workers", "6"), 6)),
)
def test_valid_evidence_caps_workers_by_cpu_memory_and_session_count(
    monkeypatch,
    tmp_path,
    capsys,
    worker_arguments,
    expected_requested,
):
    """Measured peak RSS is reused unchanged under all concurrency caps."""
    config_list = write_config_list(tmp_path, ("one.json", "two.json", "three.json"))
    configs = run_batch.read_config_list(config_list)
    pipeline = FakePipeline({path: session_plan(path) for path in configs})
    evidence = tmp_path / "evidence"
    write_resource_evidence(evidence, peak_rss_bytes=200)
    pipeline.completed.add(evidence)
    monkeypatch.setattr(run_batch, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(run_batch, "_read_mem_available_bytes", lambda: 1_000)
    monkeypatch.setattr(run_batch.os, "cpu_count", lambda: 8)

    arguments = [
        "dry-run",
        "--config-list",
        str(config_list),
        *worker_arguments,
        "--resource-run-directory",
        str(evidence),
    ]
    assert run_batch.main(arguments) == 0
    report = parse_stdout_json(capsys)
    assert report["requested_workers"] == expected_requested
    assert report["effective_workers"] == 2
    assert report["memory_limit_bytes"] == 500
    assert report["per_session_peak_rss_bytes"] == 200
    assert report["resource_evidence"]["run_directory"] == str(evidence.resolve())
    assert report["resource_evidence"]["run_fingerprint"] == "measured-run-fingerprint"


def test_parallel_new_records_evidence_and_passes_only_capped_session_workers(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Parallel admission is explicit and every run records the same calculation."""
    config_list = write_config_list(tmp_path, ("one.json", "two.json", "three.json"))
    configs = run_batch.read_config_list(config_list)
    pipeline = FakePipeline({path: session_plan(path) for path in configs})
    evidence = tmp_path / "evidence"
    write_resource_evidence(evidence, peak_rss_bytes=200)
    pipeline.completed.add(evidence)
    captured: dict[str, object] = {}

    def execute_without_processes(run_directories, *, worker_count, pipeline):
        """Capture the admitted process count without starting test subprocesses."""
        captured["run_directories"] = tuple(run_directories)
        captured["worker_count"] = worker_count
        return [
            {"outcome": "complete", "error": ""}
            for _run_directory in run_directories
        ]

    monkeypatch.setattr(run_batch, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(run_batch, "_read_mem_available_bytes", lambda: 1_000)
    monkeypatch.setattr(run_batch.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(run_batch, "_execute_sessions", execute_without_processes)

    assert run_batch.main(
        [
            "new",
            "--config-list",
            str(config_list),
            "--workers",
            "6",
            "--resource-run-directory",
            str(evidence),
        ]
    ) == 0
    report = parse_stdout_json(capsys)
    assert captured["worker_count"] == 2
    assert captured["run_directories"] == tuple(
        tmp_path / f"run-{name}" for name in ("one", "two", "three")
    )
    assert len(pipeline.batch_provenance) == 3
    for _run_directory, provenance in pipeline.batch_provenance:
        assert provenance["effective_workers"] == 2
        assert provenance["resource_evidence"]["run_directory"] == str(evidence.resolve())
        assert provenance["per_session_peak_rss_bytes"] == 200
    assert [item["outcome"] for item in report["sessions"]] == [
        "complete",
        "complete",
        "complete",
    ]


def test_tensor_guard_rejects_one_session_before_preparation(monkeypatch, tmp_path, capsys):
    """Exact tensor bytes above half MemAvailable stop before any run is created."""
    config_list = write_config_list(tmp_path, ("large.json",))
    config = run_batch.read_config_list(config_list)[0]
    pipeline = FakePipeline({config: session_plan(config, tensor_bytes=501)})
    monkeypatch.setattr(run_batch, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(run_batch, "_read_mem_available_bytes", lambda: 1_000)
    monkeypatch.setattr(run_batch.os, "cpu_count", lambda: 8)

    assert run_batch.main(["new", "--config-list", str(config_list)]) == 1
    assert "tensor" in capsys.readouterr().err.lower()
    assert pipeline.prepared == []
    assert pipeline.executed == []


def test_new_uses_independent_single_session_seams_and_records_batch_provenance(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Batch new prepares and executes each session through the WP5B seams."""
    config_list = write_config_list(tmp_path, ("one.json", "two.json"))
    configs = run_batch.read_config_list(config_list)
    pipeline = FakePipeline({path: session_plan(path) for path in configs})
    monkeypatch.setattr(run_batch, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(run_batch, "_read_mem_available_bytes", lambda: 10_000)
    monkeypatch.setattr(run_batch.os, "cpu_count", lambda: 8)

    assert run_batch.main(["new", "--config-list", str(config_list)]) == 0
    report = parse_stdout_json(capsys)
    assert pipeline.prepared == [
        (configs[0], False, "foreground"),
        (configs[1], False, "foreground"),
    ]
    assert pipeline.executed == [tmp_path / "run-one", tmp_path / "run-two"]
    assert len({path for path, _payload in pipeline.batch_provenance}) == 2
    assert all(payload["effective_workers"] == 1 for _path, payload in pipeline.batch_provenance)
    assert report["available_cpu_count"] == 8
    assert report["available_memory_bytes"] == 10_000
    assert report["memory_limit_bytes"] == 5_000
    assert [item["outcome"] for item in report["sessions"]] == ["complete", "complete"]


def test_new_skips_completed_matches_and_rerun_is_explicit(monkeypatch, tmp_path, capsys):
    """Default batch new skips validated results while --rerun creates fresh runs."""
    config_list = write_config_list(tmp_path, ("session.json",))
    config = run_batch.read_config_list(config_list)[0]
    pipeline = FakePipeline({config: session_plan(config)})
    completed = tmp_path / "run-session"
    completed.mkdir()
    pipeline.completed.add(completed)
    monkeypatch.setattr(run_batch, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(run_batch, "_read_mem_available_bytes", lambda: 10_000)

    assert run_batch.main(["new", "--config-list", str(config_list)]) == 0
    default_report = parse_stdout_json(capsys)
    assert default_report["sessions"][0]["outcome"] == "skipped_complete"
    assert pipeline.executed == []

    pipeline.completed.clear()
    assert run_batch.main(["new", "--config-list", str(config_list), "--rerun"]) == 0
    rerun_report = parse_stdout_json(capsys)
    assert pipeline.prepared[-1] == (config, True, "foreground")
    assert rerun_report["sessions"][0]["outcome"] == "complete"


def test_one_session_failure_does_not_cancel_later_sessions(monkeypatch, tmp_path, capsys):
    """Independent session outcomes remain ordered and aggregate failure is nonzero."""
    config_list = write_config_list(tmp_path, ("one.json", "two.json", "three.json"))
    configs = run_batch.read_config_list(config_list)
    pipeline = FakePipeline({path: session_plan(path) for path in configs})
    pipeline.fail_execution_for.add(tmp_path / "run-two")
    monkeypatch.setattr(run_batch, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(run_batch, "_read_mem_available_bytes", lambda: 10_000)

    assert run_batch.main(["new", "--config-list", str(config_list)]) == 1
    report = parse_stdout_json(capsys)
    assert pipeline.executed == [
        tmp_path / "run-one",
        tmp_path / "run-two",
        tmp_path / "run-three",
    ]
    assert [item["outcome"] for item in report["sessions"]] == [
        "complete",
        "failed",
        "complete",
    ]
    assert "synthetic failure" in report["sessions"][1]["error"]


def test_parallel_execution_uses_only_the_admitted_process_count(monkeypatch, tmp_path):
    """More than one admitted session uses processes, never an in-process thread pool."""
    submitted: list[tuple[object, str]] = []
    executor_counts: list[int] = []

    class FakeFuture:
        """Immediate fake process future returning one completed session."""

        def result(self):
            """Return one dimensionless completion outcome."""
            return {"outcome": "complete", "error": ""}

    class FakeExecutor:
        """Context-managed process-executor stand-in recording submissions."""

        def __init__(self, *, max_workers):
            executor_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def submit(self, function, run_directory):
            submitted.append((function, run_directory))
            return FakeFuture()

    class FailIfUsedPipeline:
        """Reject accidental parent-process scientific execution."""

        @staticmethod
        def run_prepared_task_decoding(_run_directory):
            raise AssertionError("parallel sessions executed in the parent process")

    run_directories = (tmp_path / "run-one", tmp_path / "run-two", tmp_path / "run-three")
    monkeypatch.setattr(run_batch, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(run_batch, "as_completed", lambda futures: tuple(futures))
    outcomes = run_batch._execute_sessions(
        run_directories,
        worker_count=2,
        pipeline=FailIfUsedPipeline(),
    )
    assert executor_counts == [2]
    assert submitted == [
        (run_batch._execute_prepared_worker, str(run_directory))
        for run_directory in run_directories
    ]
    assert outcomes == [
        {"outcome": "complete", "error": ""},
        {"outcome": "complete", "error": ""},
        {"outcome": "complete", "error": ""},
    ]


def test_batch_entry_sets_thread_limits_before_lazy_pipeline_import():
    """A fresh batch process freezes numerical thread limits before pipeline import."""
    script = """
import builtins
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.pop(key, None)
seen = []
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('numpy', 'scipy', 'sklearn', 'src.neural_analysis.task_decoding.pipeline')):
        assert all(os.environ.get(key) == '1' for key in
                   ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
        seen.append(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from src.neural_analysis.task_decoding import run_batch
assert all(os.environ.get(key) == '1' for key in
           ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
run_batch._pipeline_module()
assert any('pipeline' in name for name in seen)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
