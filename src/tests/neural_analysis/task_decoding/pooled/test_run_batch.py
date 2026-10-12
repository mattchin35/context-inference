"""WP8 contracts for bounded pooled Slurm array planning and execution."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess

import pytest


MODULE_NAME = "src.neural_analysis.task_decoding.pooled.run_batch"


def _batch_module():
    """Import and return the pooled batch module inside an individual test.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.pooled.run_batch``.
    """
    return importlib.import_module(MODULE_NAME)


def _write_config_list(tmp_path: Path, count: int = 3) -> tuple[Path, tuple[Path, ...]]:
    """Create an ordered portable config list and small config files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory.
    count : int, default=3
        Positive number of session configuration files.

    Returns
    -------
    tuple[pathlib.Path, tuple[pathlib.Path, ...]]
        Config-list path and resolved configuration paths in declared order.
    """
    paths = tuple(tmp_path / f"session-{index}" / "pooled.json" for index in range(count))
    for path in paths:
        path.parent.mkdir()
        path.write_text("{}\n", encoding="ascii")
    config_list = tmp_path / "batch" / "sessions.txt"
    config_list.parent.mkdir()
    config_list.write_text(
        "\n".join(os.path.relpath(path, config_list.parent) for path in paths) + "\n",
        encoding="utf-8",
    )
    return config_list, tuple(path.resolve() for path in paths)


def _plan(config_path: Path, tensor_bytes: int = 100) -> dict[str, object]:
    """Return one minimal bounded pooled session plan.

    Parameters
    ----------
    config_path : pathlib.Path
        Resolved pooled overlay path.
    tensor_bytes : int, default=100
        Exact planned simultaneous rate-tensor allocation in bytes.

    Returns
    -------
    dict[str, object]
        Session identity, source identity, and positive resource envelope.
    """
    return {
        "config_path": str(config_path),
        "session_id": config_path.parent.name,
        "analysis_version": "pooled-task-variable-decoding-v1",
        "scientific_source_fingerprint": "source-fingerprint",
        "resource_envelope": {
            "full_trial_count": 12,
            "tensor_trial_count": 12,
            "pfc_unit_count": 2,
            "hpc_unit_count": 2,
            "time_bin_count": 8,
            "training_window_count": 3,
            "condition_count": 2,
            "evaluation_pair_count": 3,
            "target_count": 2,
            "outer_fold_count": 3,
            "coefficient_feature_capacity": 4,
            "categorical_fit_count": 108,
            "numerical_fit_count": 108,
            "transform_fit_count": 72,
            "evaluation_metric_cell_count": 3888,
            "tensor_allocation_bytes": tensor_bytes,
            "planned_major_array_bytes": 1000,
        },
    }


def _write_evidence(tmp_path: Path) -> Path:
    """Create a complete measured evidence directory for admission tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory.

    Returns
    -------
    pathlib.Path
        Evidence directory containing only bounded JSON sidecars.
    """
    directory = tmp_path / "evidence"
    directory.mkdir()
    evidence = _plan(tmp_path / "evidence-config.json", tensor_bytes=200)
    (directory / "run_state.json").write_text(
        json.dumps({"lifecycle": "complete", "final_results_published": True}),
        encoding="utf-8",
    )
    (directory / "scientific_source.json").write_text(
        json.dumps({"fingerprint": evidence["scientific_source_fingerprint"]}),
        encoding="utf-8",
    )
    (directory / "config.json").write_text(
        json.dumps({"analysis_version": evidence["analysis_version"]}),
        encoding="utf-8",
    )
    (directory / "resource_usage.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "peak_rss_bytes": 1_107_640_320,
                "resource_envelope": evidence["resource_envelope"],
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_config_list_is_ordered_portable_and_rejects_duplicates(tmp_path: Path) -> None:
    """Manifest inputs resolve relative paths once and forbid duplicate sessions."""
    run_batch = _batch_module()
    config_list, expected = _write_config_list(tmp_path)

    assert run_batch.read_config_list(config_list) == expected

    duplicate = tmp_path / "duplicates.txt"
    duplicate.write_text(f"{expected[0]}\n{expected[0]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        run_batch.read_config_list(duplicate)


def test_batch_plan_is_bounded_and_admitted_by_completed_evidence(
    tmp_path: Path,
) -> None:
    """Planning reads small plans only and compares every envelope dimension."""
    run_batch = _batch_module()
    config_list, config_paths = _write_config_list(tmp_path, count=2)
    evidence = _write_evidence(tmp_path)
    calls: list[Path] = []

    class FakePipeline:
        """Return bounded plans without scientific loading or fitting."""

        @staticmethod
        def plan_pooled_decoding_session(path: Path) -> dict[str, object]:
            """Record and return one small plan."""
            calls.append(Path(path))
            return _plan(Path(path))

    batch_plan = run_batch.plan_batch(
        config_list_path=config_list,
        resource_run_directory=evidence,
        array_concurrency=2,
        pipeline=FakePipeline,
    )

    assert calls == list(config_paths)
    assert batch_plan["array_concurrency"] == 2
    assert batch_plan["session_count"] == 2
    assert batch_plan["resource_evidence"]["peak_rss_bytes"] == 1_107_640_320
    assert tuple(item["config_path"] for item in batch_plan["sessions"]) == tuple(
        str(path) for path in config_paths
    )


def test_batch_plan_rejects_any_session_larger_than_evidence(tmp_path: Path) -> None:
    """A larger scientific envelope requires a new measured benchmark."""
    run_batch = _batch_module()
    config_list, _config_paths = _write_config_list(tmp_path, count=1)
    evidence = _write_evidence(tmp_path)

    class OversizedPipeline:
        """Return one plan exceeding the evidence tensor dimension."""

        @staticmethod
        def plan_pooled_decoding_session(path: Path) -> dict[str, object]:
            """Return an otherwise matching oversized plan."""
            return _plan(Path(path), tensor_bytes=201)

    with pytest.raises(ValueError, match="exceeds|benchmark|evidence"):
        run_batch.plan_batch(
            config_list_path=config_list,
            resource_run_directory=evidence,
            array_concurrency=1,
            pipeline=OversizedPipeline,
        )


def test_preparation_writes_immutable_index_to_exact_run_mapping(
    tmp_path: Path,
) -> None:
    """Each array index owns one exact independently resumable prepared run."""
    run_batch = _batch_module()
    config_list, config_paths = _write_config_list(tmp_path, count=3)
    evidence = _write_evidence(tmp_path)

    class FakePipeline:
        """Prepare deterministic run directories without scientific work."""

        @staticmethod
        def plan_pooled_decoding_session(path: Path) -> dict[str, object]:
            """Return one matching bounded plan."""
            return _plan(Path(path))

        @staticmethod
        def prepare_pooled_decoding_run(path: Path, *, execution_mode: str) -> Path:
            """Create and return one exact synthetic run directory."""
            assert execution_mode == "slurm"
            directory = Path(path).parent / "analysis_runs" / "prepared"
            directory.mkdir(parents=True)
            return directory

    batch_plan = run_batch.plan_batch(
        config_list_path=config_list,
        resource_run_directory=evidence,
        array_concurrency=2,
        pipeline=FakePipeline,
    )
    prepared = run_batch.prepare_array_batch(
        batch_plan=batch_plan,
        config_list_path=config_list,
        batch_root=config_list.parent / "pooled_batch_runs",
        pipeline=FakePipeline,
        timestamp="20261012T120000Z",
    )

    manifest_path = Path(prepared["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["array_concurrency"] == 2
    assert [item["array_index"] for item in manifest["sessions"]] == [0, 1, 2]
    assert [item["config_path"] for item in manifest["sessions"]] == [
        str(path) for path in config_paths
    ]
    assert len({item["run_directory"] for item in manifest["sessions"]}) == 3


def test_array_submission_uses_one_capped_sbatch_and_records_each_index(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """One array submission maps job/index receipts without retries."""
    run_batch = _batch_module()
    config_list, _config_paths = _write_config_list(tmp_path, count=3)
    batch_directory = config_list.parent / "pooled_batch_runs" / "batch"
    batch_directory.mkdir(parents=True)
    sessions = []
    for index in range(3):
        run_directory = tmp_path / f"run-{index}"
        run_directory.mkdir()
        (run_directory / "run_state.json").write_text(
            json.dumps({"lifecycle": "prepared", "error": None}),
            encoding="utf-8",
        )
        sessions.append(
            {
                "array_index": index,
                "config_path": f"/cluster/session-{index}/pooled.json",
                "run_directory": str(run_directory),
            }
        )
    manifest_path = batch_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps({"array_concurrency": 2, "sessions": sessions}),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def accept(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Capture one scheduler command and return one array job ID."""
        commands.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "77777;einstein\n", "")

    monkeypatch.setattr(run_batch, "_run_command", accept)
    monkeypatch.setattr(run_batch, "_repository_commit", lambda: "approved-commit")

    receipt = run_batch.submit_array_batch(manifest_path)

    assert len(commands) == 1
    assert "--array=0-2%2" in commands[0]
    assert commands[0][-3:] == ["_execute-array", "--manifest", str(manifest_path)]
    assert receipt["job_id"] == "77777"
    for item in sessions:
        saved = json.loads(
            (Path(item["run_directory"]) / "slurm_submission.json").read_text()
        )
        assert saved["job_id"] == "77777"
        assert saved["array_index"] == item["array_index"]


def test_array_element_executes_only_its_exact_manifest_run(
    tmp_path: Path,
) -> None:
    """A scheduled array element cannot search for or execute a sibling run."""
    run_batch = _batch_module()
    manifest_path = tmp_path / "manifest.json"
    runs = tuple(tmp_path / f"run-{index}" for index in range(2))
    manifest_path.write_text(
        json.dumps(
            {
                "array_concurrency": 1,
                "sessions": [
                    {
                        "array_index": index,
                        "config_path": f"/cluster/session-{index}/pooled.json",
                        "run_directory": str(run_directory),
                    }
                    for index, run_directory in enumerate(runs)
                ],
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    class FakePipeline:
        """Record the exact run selected by one array index."""

        @staticmethod
        def run_prepared_pooled_decoding(directory: Path) -> None:
            """Record one prepared-run execution request."""
            calls.append(Path(directory))

    run_batch.execute_array_element(
        manifest_path,
        array_task_id=1,
        pipeline=FakePipeline,
    )
    assert calls == [runs[1]]

    with pytest.raises(ValueError, match="index|array"):
        run_batch.execute_array_element(
            manifest_path,
            array_task_id=2,
            pipeline=FakePipeline,
        )
