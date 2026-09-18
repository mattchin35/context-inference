"""RED contracts for the standalone resumable Spike-phase launcher."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import fcntl
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from src.neural_analysis.lfp_spike_phase_validation import (
    build_ct026_spike_phase_preview_config,
)
from src.neural_analysis.lfp_summary_models import UnitPopulationConfig
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult


def _launcher() -> object:
    """Import the launcher inside tests so an absent module is a normal RED."""
    return importlib.import_module("src.neural_analysis.lfp_spike_phase_launcher")


def _population(probe: str = "ProbeB") -> UnitPopulationConfig:
    """Return one explicit probe-qualified two-unit population."""
    return UnitPopulationConfig(
        label=f"CT026 {probe} active",
        probe_label=probe,
        sorter_path=Path(f"{probe}/kilosort4"),
        aligned_spike_path=Path(f"{probe}_sync.npz"),
        selected_channels=(7, 11),
        quality_settings=(
            ("channel_quality", "good"),
            ("inside_brain", "true"),
            ("unit_quality", "good,mua"),
        ),
        stable_unit_ids=(f"{probe}:2", f"{probe}:9"),
    )


class _MemorySample:
    """Mutable scalar memory record populated by an injected context manager."""

    process_rss_bytes = 1024
    process_tree_rss_bytes = 2048
    process_tree_pss_bytes = 1536
    provenance = "synthetic process tree"
    sampling_interval_seconds = 0.1
    maximum_child_process_count = 3


def _dependencies(
    tmp_path: Path,
    calls: list[str],
    terminal: list[str],
    *,
    compute: Callable[..., ComponentRunResult] | None = None,
    component_compatible: bool = False,
    cleanup: Callable[[object], None] | None = None,
) -> object:
    """Return deterministic high-level launcher seams with no raw-data access."""
    launcher = _launcher()

    def build_config(
        session_path: Path,
        population: UnitPopulationConfig,
        shuffle_count: int,
        worker_count: int,
    ) -> object:
        """Build the reviewed CT026 configuration with explicit execution values."""
        config = build_ct026_spike_phase_preview_config(session_path, population)
        return replace(
            config,
            ppc=replace(config.ppc, shuffle_count=shuffle_count),
            ppc_execution=replace(
                config.ppc_execution,
                worker_count=worker_count,
                checkpoint_enabled=True,
                checkpoint_retention="incomplete_only",
            ),
        )

    def default_compute(
        config: object,
        progress_callback: Callable[[object], None],
        cleanup_preparation_observer: Callable[[tuple[object, ...]], None],
    ) -> ComponentRunResult:
        """Persist one exact cleanup target before returning a completed component."""
        calls.append("compute")
        work = config.output_directory.parent / "lfp_summary_work" / "ppc" / ("a" * 64)
        work.mkdir(parents=True, exist_ok=True)
        (work / "metadata.json").write_text(
            json.dumps({"run_fingerprint": "a" * 64}),
            encoding="ascii",
        )
        target = launcher.PPCWorkCleanupTarget(work, "a" * 64)
        cleanup_preparation_observer((target,))
        return ComponentRunResult(
            "spike_phase",
            "complete",
            "complete_spike_phase",
            None,
            {"components": {"spike_phase": {}}},
            deferred_cleanup=lambda: calls.append("callback_cleanup"),
            deferred_cleanup_targets=(target,),
            execution_metadata={
                "ppc_planning_seconds": 1.25,
                "grouped_execution_seconds": 2.5,
                "requested_worker_count": config.ppc_execution.worker_count,
                "planner_active_worker_count": 3,
                "planned_parent_private_bytes": 100,
                "planned_worker_private_bytes": 50,
                "planned_aggregate_array_bytes": 250,
                "shared_phase_mmap_bytes": 75,
                "completed_block_count": 2,
                "resumed_block_count": 0,
                "run_fingerprint": "a" * 64,
                "run_directory": str(work),
            },
        )

    def publish_report(**kwargs: object) -> object:
        """Publish one already-validated fake report beneath the launcher run."""
        calls.append("report")
        parent = Path(kwargs["run_parent"])
        report_directory = parent / "synthetic_report"
        report_directory.mkdir(parents=True, exist_ok=False)
        report_path = report_directory / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": "spike_phase_report.v1",
                    "run_kind": kwargs["run_kind"],
                    "shuffle_count": kwargs["config"].ppc.shuffle_count,
                }
            )
            + "\n",
            encoding="ascii",
        )
        return SimpleNamespace(
            run_directory=report_directory,
            report_path=report_path,
            deferred_cleanup=kwargs.get("deferred_cleanup"),
        )

    def default_cleanup(target: object) -> None:
        """Remove one synthetic exact work directory and record ordering."""
        calls.append("cleanup_target")
        for child in target.run_directory.iterdir():
            child.unlink()
        target.run_directory.rmdir()

    return launcher.LauncherDependencies(
        load_active_population=lambda session, probe: _population(probe),
        build_config=build_config,
        load_trial_count=lambda config: 12,
        repository_state=lambda: launcher.RepositoryState(
            repository_root=tmp_path / "repo",
            git_commit="b" * 40,
            tracked_clean=True,
        ),
        source_fingerprints=lambda config: {"trial_table": {"size_bytes": 10}},
        component_is_compatible=lambda config: component_compatible,
        compute_component=compute or default_compute,
        validate_component=lambda config: calls.append("validate_component"),
        publish_report=publish_report,
        cleanup_target=cleanup or default_cleanup,
        now_utc=iter(
            (
                "2026-09-18T20-00-00Z",
                *[f"2026-09-18T20-00-{index:02d}Z" for index in range(1, 40)],
            )
        ).__next__,
        monotonic_seconds=iter(float(index) for index in range(100)).__next__,
        process_tree_sampler=lambda: nullcontext(_MemorySample()),
        terminal_write=terminal.append,
    )


def test_parser_requires_explicit_probe_and_final_confirmation() -> None:
    """CLI accepts only explicit one-probe preview/final and exact resume modes."""
    launcher = _launcher()
    preview = launcher.parse_launcher_command(
        [
            "new",
            "--session-path",
            "/data/CT026",
            "--probe",
            "ProbeA",
            "--shuffles",
            "100",
        ]
    )
    assert preview.mode == "new"
    assert preview.probe_label == "ProbeA"
    assert preview.worker_count == 8
    assert preview.shuffle_count == 100
    assert preview.final_run is False

    final = launcher.parse_launcher_command(
        [
            "new",
            "--session-path",
            "/data/CT026",
            "--probe",
            "ProbeB",
            "--shuffles",
            "1000",
            "--final-run",
            "--workers",
            "4",
        ]
    )
    assert final.final_run is True and final.worker_count == 4
    resume = launcher.parse_launcher_command(
        ["resume", "--run-directory", "/runs/exact"]
    )
    assert resume.mode == "resume" and resume.run_directory == Path("/runs/exact")

    invalid = (
        ["new", "--session-path", "/data/CT026", "--shuffles", "100"],
        [
            "new", "--session-path", "/data/CT026", "--probe", "ProbeB",
            "--shuffles", "1000",
        ],
        [
            "new", "--session-path", "/data/CT026", "--probe", "ProbeB",
            "--shuffles", "100", "--final-run",
        ],
    )
    for arguments in invalid:
        with pytest.raises(SystemExit):
            launcher.parse_launcher_command(arguments)


def test_dry_run_writes_evidence_without_scientific_work(tmp_path: Path) -> None:
    """Metadata preflight is terminal and never invokes computation or reporting."""
    launcher = _launcher()
    calls: list[str] = []
    terminal: list[str] = []
    dependencies = _dependencies(tmp_path, calls, terminal)
    command = launcher.parse_launcher_command(
        [
            "new",
            "--session-path",
            str(tmp_path / "CT026"),
            "--probe",
            "ProbeB",
            "--shuffles",
            "100",
            "--dry-run",
        ]
    )

    result = launcher.run_launcher(command, dependencies)

    assert result.exit_code == 0 and result.status == "preflight_complete"
    assert calls == []
    assert result.run_directory.is_dir()
    state = json.loads((result.run_directory / "launcher_state.json").read_text())
    preflight = json.loads((result.run_directory / "preflight.json").read_text())
    assert state["resume_command"] is None
    assert preflight["schema_version"] == "spike_phase_preflight.v1"
    assert preflight["exact_plan_available"] is False
    assert preflight["ppc_planning_seconds"] is None
    assert preflight["planned_ppc_allocation_bytes"] is None
    assert not (tmp_path / "CT026" / "processed" / "lfp_summary_cache").exists()


def test_identity_and_resume_command_exist_before_interrupted_planning(
    tmp_path: Path,
) -> None:
    """A planning interruption leaves atomic state/log and an exact resume command."""
    launcher = _launcher()
    calls: list[str] = []
    terminal: list[str] = []

    def interrupt(
        config: object,
        progress_callback: Callable[[object], None],
        cleanup_preparation_observer: Callable[[tuple[object, ...]], None],
    ) -> ComponentRunResult:
        """Verify durable initialization before simulating Ctrl-C in planning."""
        del progress_callback, cleanup_preparation_observer
        run_directories = list((tmp_path / "runs").iterdir())
        assert len(run_directories) == 1
        run = run_directories[0]
        for name in (
            "launcher_state.json",
            "configuration.json",
            "source_identity.json",
            "preflight.json",
            "run.log",
            "run_summary.md",
        ):
            assert (run / name).is_file()
        state = json.loads((run / "launcher_state.json").read_text())
        assert state["resume_command"].startswith("uv run python -m ")
        raise KeyboardInterrupt("synthetic planning interruption")

    dependencies = _dependencies(tmp_path, calls, terminal, compute=interrupt)
    command = launcher.parse_launcher_command(
        [
            "new", "--session-path", str(tmp_path / "CT026"),
            "--analysis-root", str(tmp_path / "runs"),
            "--probe", "ProbeB", "--shuffles", "100",
        ]
    )
    result = launcher.run_launcher(command, dependencies)

    assert result.exit_code == 130 and result.status == "interrupted"
    state = json.loads((result.run_directory / "launcher_state.json").read_text())
    assert state["active_status"] == "interrupted"
    assert "component_complete" not in state["completed_stages"]
    assert any("resume" in line.lower() for line in terminal)


def test_success_validates_then_cleans_exact_target(tmp_path: Path) -> None:
    """Cleanup is the last successful action after report and launcher validation."""
    launcher = _launcher()
    calls: list[str] = []
    terminal: list[str] = []
    dependencies = _dependencies(tmp_path, calls, terminal)
    command = launcher.parse_launcher_command(
        [
            "new", "--session-path", str(tmp_path / "CT026"),
            "--analysis-root", str(tmp_path / "runs"),
            "--probe", "ProbeA", "--shuffles", "100",
        ]
    )

    result = launcher.run_launcher(command, dependencies)

    assert result.exit_code == 0 and result.status == "complete"
    assert calls.index("compute") < calls.index("validate_component")
    assert calls.index("validate_component") < calls.index("report")
    assert calls[-1] == "cleanup_target"
    state = json.loads((result.run_directory / "launcher_state.json").read_text())
    assert state["completed_stages"][-3:] == [
        "launcher_artifacts_validated",
        "cleanup_complete",
        "complete",
    ]
    assert state["identity"]["probe_label"] == "ProbeA"
    assert state["measurements"]["ppc_planning_seconds"] == 1.25
    assert state["measurements"]["process_tree_pss_bytes"] == 1536


def test_cleanup_failure_resumes_without_recomputing_or_rerendering(
    tmp_path: Path,
) -> None:
    """A cleanup-only resume revalidates artifacts and retries the exact target."""
    launcher = _launcher()
    first_calls: list[str] = []
    terminal: list[str] = []

    def fail_cleanup(target: object) -> None:
        """Leave exact work in place on the first cleanup attempt."""
        first_calls.append("cleanup_failed")
        raise OSError("synthetic cleanup failure")

    first_dependencies = _dependencies(
        tmp_path,
        first_calls,
        terminal,
        cleanup=fail_cleanup,
    )
    command = launcher.parse_launcher_command(
        [
            "new", "--session-path", str(tmp_path / "CT026"),
            "--analysis-root", str(tmp_path / "runs"),
            "--probe", "ProbeB", "--shuffles", "100",
        ]
    )
    failed = launcher.run_launcher(command, first_dependencies)
    assert failed.exit_code == 1 and failed.status == "cleanup_failed"

    resume_calls: list[str] = []

    def successful_cleanup(target: object) -> None:
        """Remove only the exact synthetic target during cleanup-only resume."""
        resume_calls.append("cleanup_target")
        for child in target.run_directory.iterdir():
            child.unlink()
        target.run_directory.rmdir()

    resumed_dependencies = _dependencies(
        tmp_path,
        resume_calls,
        terminal,
        component_compatible=True,
        compute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cleanup-only resume recomputed")
        ),
        cleanup=successful_cleanup,
    )
    resume = launcher.parse_launcher_command(
        ["resume", "--run-directory", str(failed.run_directory)]
    )
    completed = launcher.run_launcher(resume, resumed_dependencies)

    assert completed.exit_code == 0 and completed.status == "complete"
    assert "compute" not in resume_calls and "report" not in resume_calls
    assert resume_calls == ["validate_component", "cleanup_target"]


def test_dirty_tracked_checkout_fails_before_run_directory(tmp_path: Path) -> None:
    """Production new runs reject unidentified tracked source changes."""
    launcher = _launcher()
    calls: list[str] = []
    terminal: list[str] = []
    dependencies = replace(
        _dependencies(tmp_path, calls, terminal),
        repository_state=lambda: launcher.RepositoryState(
            tmp_path / "repo",
            "b" * 40,
            False,
        ),
    )
    command = launcher.parse_launcher_command(
        [
            "new", "--session-path", str(tmp_path / "CT026"),
            "--analysis-root", str(tmp_path / "runs"),
            "--probe", "ProbeB", "--shuffles", "100",
        ]
    )

    result = launcher.run_launcher(command, dependencies)

    assert result.exit_code == 2 and result.run_directory is None
    assert not (tmp_path / "runs").exists()
    assert calls == []


def test_resume_rejects_tampered_source_artifact_before_component_access(
    tmp_path: Path,
) -> None:
    """Saved source evidence is part of resume identity, not decorative output."""
    launcher = _launcher()
    first_calls: list[str] = []
    terminal: list[str] = []
    failed = launcher.run_launcher(
        launcher.parse_launcher_command(
            [
                "new", "--session-path", str(tmp_path / "CT026"),
                "--analysis-root", str(tmp_path / "runs"),
                "--probe", "ProbeB", "--shuffles", "100",
            ]
        ),
        _dependencies(
            tmp_path,
            first_calls,
            terminal,
            cleanup=lambda _target: (_ for _ in ()).throw(OSError("keep work")),
        ),
    )
    source_path = failed.run_directory / "source_identity.json"
    source = json.loads(source_path.read_text(encoding="ascii"))
    source["repository"]["git_commit"] = "c" * 40
    source_path.write_text(json.dumps(source) + "\n", encoding="ascii")
    resume_calls: list[str] = []

    resumed = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["resume", "--run-directory", str(failed.run_directory)]
        ),
        _dependencies(tmp_path, resume_calls, terminal, component_compatible=True),
    )

    assert resumed.exit_code != 0
    assert resume_calls == []


def test_live_launcher_lock_fails_without_mutating_owner_state(tmp_path: Path) -> None:
    """A competing resume cannot relabel or append to the lock owner's run."""
    launcher = _launcher()
    first_calls: list[str] = []
    terminal: list[str] = []
    failed = launcher.run_launcher(
        launcher.parse_launcher_command(
            [
                "new", "--session-path", str(tmp_path / "CT026"),
                "--analysis-root", str(tmp_path / "runs"),
                "--probe", "ProbeB", "--shuffles", "100",
            ]
        ),
        _dependencies(
            tmp_path,
            first_calls,
            terminal,
            cleanup=lambda _target: (_ for _ in ()).throw(OSError("keep work")),
        ),
    )
    state_path = failed.run_directory / "launcher_state.json"
    log_path = failed.run_directory / "run.log"
    before_state = state_path.read_bytes()
    before_log = log_path.read_bytes()
    with (failed.run_directory / "launcher.lock").open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        competing = launcher.run_launcher(
            launcher.parse_launcher_command(
                ["resume", "--run-directory", str(failed.run_directory)]
            ),
            _dependencies(tmp_path, [], terminal, component_compatible=True),
        )

    assert competing.exit_code == 1
    assert state_path.read_bytes() == before_state
    assert log_path.read_bytes() == before_log
