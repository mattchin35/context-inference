"""Documented command entry point for resumable Spike-phase runs.

Argument parsing and top-level mode dispatch intentionally remain here. The
launcher-specific filesystem, lifecycle, execution, and resource helpers live
in :mod:`src.neural_analysis.lfp_summary.launcher_runtime`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from src.neural_analysis.lfp_summary import launcher_runtime as _runtime


_MODULE = "src.neural_analysis.lfp_spike_phase_launcher"

LauncherCommand = _runtime.LauncherCommand
RepositoryState = _runtime.RepositoryState
LauncherDependencies = _runtime.LauncherDependencies
LauncherRunResult = _runtime.LauncherRunResult
PPCWorkCleanupTarget = _runtime.PPCWorkCleanupTarget

_LauncherSignal = _runtime._LauncherSignal
_LauncherLockError = _runtime._LauncherLockError
_prepare_new_run = _runtime._prepare_new_run
_prepare_resume = _runtime._prepare_resume
_prepare_report_recovery = _runtime._prepare_report_recovery
_load_resume_state = _runtime._load_resume_state
_load_completed_state = _runtime._load_completed_state
_rerender_completed_report = _runtime._rerender_completed_report
_execute_locked_run = _runtime._execute_locked_run
_complete_report_recovery = _runtime._complete_report_recovery
_record_report_recovery_failure = _runtime._record_report_recovery_failure
_record_report_rerender_failure = _runtime._record_report_rerender_failure
_record_terminal_failure = _runtime._record_terminal_failure
_resume_from_state = _runtime._resume_from_state
_terminal = _runtime._terminal
_launcher_lock = _runtime._launcher_lock
_launcher_signal_handlers = _runtime._launcher_signal_handlers
make_production_launcher_dependencies = _runtime.make_production_launcher_dependencies


def __getattr__(name: str) -> object:
    """Forward legacy private helper access to the canonical runtime owner."""
    return getattr(_runtime, name)


def parse_launcher_command(argv: Sequence[str] | None = None) -> LauncherCommand:
    """Parse one explicit new, resume, recovery, or rerender request.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command arguments excluding the Python module name. ``None`` uses
        ``sys.argv[1:]``.

    Returns
    -------
    LauncherCommand
        Validated categorical/path/count values. No filesystem access occurs.
    """
    parser = argparse.ArgumentParser(prog=f"python -m {_MODULE}")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    new = subparsers.add_parser("new")
    session_source = new.add_mutually_exclusive_group(required=True)
    session_source.add_argument("--session-path", type=Path)
    session_source.add_argument("--session-metadata", type=Path)
    new.add_argument("--probe", dest="probe_label", required=True)
    new.add_argument(
        "--shuffles",
        dest="shuffle_count",
        type=int,
        choices=(100, 1000),
        required=True,
    )
    new.add_argument("--workers", dest="worker_count", type=int, default=8)
    new.add_argument("--analysis-root", type=Path)
    new.add_argument("--cache-directory", type=Path, required=True)
    new.add_argument("--dry-run", action="store_true")
    new.add_argument("--final-run", action="store_true")
    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-directory", type=Path, required=True)
    recover_report = subparsers.add_parser("recover-report")
    recover_report.add_argument("--run-directory", type=Path, required=True)
    rerender_report = subparsers.add_parser("rerender-report")
    rerender_report.add_argument("--run-directory", type=Path, required=True)
    values = parser.parse_args(list(argv) if argv is not None else None)
    if values.mode in {"resume", "recover-report", "rerender-report"}:
        return LauncherCommand(mode=values.mode, run_directory=values.run_directory)
    if values.worker_count < 1:
        parser.error("--workers must be a positive integer")
    if values.shuffle_count == 1000 and not values.final_run:
        parser.error("1,000 shuffles requires --final-run")
    if values.shuffle_count == 100 and values.final_run:
        parser.error("--final-run is valid only with 1,000 shuffles")
    analysis_root = values.analysis_root or (
        values.session_path / "analysis_runs"
        if values.session_path is not None
        else None
    )
    return LauncherCommand(
        mode="new",
        session_path=values.session_path,
        session_metadata=values.session_metadata,
        probe_label=values.probe_label,
        shuffle_count=values.shuffle_count,
        worker_count=values.worker_count,
        analysis_root=analysis_root,
        cache_directory=values.cache_directory,
        dry_run=bool(values.dry_run),
        final_run=bool(values.final_run),
    )


def run_launcher(
    command: LauncherCommand,
    dependencies: LauncherDependencies,
) -> LauncherRunResult:
    """Run or resume one launcher transaction without hiding failures.

    Parameters
    ----------
    command : LauncherCommand
        Validated new/resume values from :func:`parse_launcher_command`.
    dependencies : LauncherDependencies
        Injected scalar identity, numerical component, report, cleanup, clock,
        process-memory, and terminal seams.

    Returns
    -------
    LauncherRunResult
        Exit/status/path metadata. Numerical arrays never cross this boundary.
    """
    run_directory: Path | None = None
    state: dict[str, object] | None = None
    try:
        if command.mode == "new":
            config, run_directory, state = _prepare_new_run(command, dependencies)
            if bool(state["dry_run"]):
                return LauncherRunResult(0, "preflight_complete", run_directory, None)
            with _launcher_lock(run_directory, state, dependencies):
                return _execute_locked_run(config, run_directory, state, dependencies)
        if command.mode == "resume":
            run_directory, state = _load_resume_state(command)
            with _launcher_lock(run_directory, state, dependencies):
                config, run_directory, state = _prepare_resume(
                    command,
                    dependencies,
                    run_directory,
                    state,
                )
                return _execute_locked_run(config, run_directory, state, dependencies)
        if command.mode == "recover-report":
            run_directory, state = _load_resume_state(command)
            with _launcher_lock(run_directory, state, dependencies):
                config, run_directory, state = _prepare_report_recovery(
                    dependencies,
                    run_directory,
                    state,
                )
                result = _execute_locked_run(
                    config,
                    run_directory,
                    state,
                    dependencies,
                    component_already_validated=True,
                    report_recovery=True,
                )
                _complete_report_recovery(run_directory, state, dependencies)
                return result
        if command.mode == "rerender-report":
            run_directory, state = _load_completed_state(command)
            with _launcher_lock(run_directory, state, dependencies):
                return _rerender_completed_report(
                    dependencies,
                    run_directory,
                    state,
                )
        raise ValueError(
            "launcher mode must be new, resume, recover-report, or rerender-report"
        )
    except _LauncherLockError as error:
        _terminal(dependencies, f"Launcher failed: {error}")
        return LauncherRunResult(
            1,
            "failed",
            run_directory,
            _resume_from_state(state),
        )
    except _LauncherSignal as error:
        exit_code = 128 + error.signum
        if run_directory is not None and state is not None:
            if command.mode == "rerender-report":
                _record_report_rerender_failure(
                    run_directory,
                    state,
                    str(error),
                    dependencies,
                )
                _terminal(dependencies, f"Launcher interrupted: {error}")
            else:
                _record_report_recovery_failure(
                    command,
                    run_directory,
                    state,
                    str(error),
                    dependencies,
                )
                _record_terminal_failure(
                    run_directory,
                    state,
                    "interrupted",
                    str(error),
                    dependencies,
                )
        return LauncherRunResult(
            exit_code,
            "interrupted",
            run_directory,
            _resume_from_state(state),
        )
    except KeyboardInterrupt as error:
        if run_directory is not None and state is not None:
            message = str(error) or "keyboard interrupt"
            if command.mode == "rerender-report":
                _record_report_rerender_failure(
                    run_directory,
                    state,
                    message,
                    dependencies,
                )
                _terminal(dependencies, f"Launcher interrupted: {message}")
            else:
                _record_report_recovery_failure(
                    command,
                    run_directory,
                    state,
                    message,
                    dependencies,
                )
                _record_terminal_failure(
                    run_directory,
                    state,
                    "interrupted",
                    message,
                    dependencies,
                )
        return LauncherRunResult(
            130,
            "interrupted",
            run_directory,
            _resume_from_state(state),
        )
    except (ValueError, FileExistsError) as error:
        if run_directory is not None and state is not None:
            if command.mode == "rerender-report":
                _record_report_rerender_failure(
                    run_directory,
                    state,
                    str(error),
                    dependencies,
                )
                _terminal(dependencies, f"Launcher rejected request: {error}")
            else:
                _record_report_recovery_failure(
                    command,
                    run_directory,
                    state,
                    str(error),
                    dependencies,
                )
                _record_terminal_failure(
                    run_directory,
                    state,
                    "failed",
                    str(error),
                    dependencies,
                )
        else:
            _terminal(dependencies, f"Launcher rejected request: {error}")
        return LauncherRunResult(2, "failed", run_directory, _resume_from_state(state))
    except (ArithmeticError, KeyError, OSError, RuntimeError) as error:
        status = (
            "cleanup_failed"
            if state is not None and state.get("active_status") == "cleaning"
            else "failed"
        )
        if run_directory is not None and state is not None:
            if command.mode == "rerender-report":
                _record_report_rerender_failure(
                    run_directory,
                    state,
                    str(error),
                    dependencies,
                )
                _terminal(dependencies, f"Launcher failed: {error}")
            else:
                _record_report_recovery_failure(
                    command,
                    run_directory,
                    state,
                    str(error),
                    dependencies,
                )
                _record_terminal_failure(
                    run_directory,
                    state,
                    status,
                    str(error),
                    dependencies,
                )
        else:
            _terminal(dependencies, f"Launcher {status}: {error}")
        return LauncherRunResult(1, status, run_directory, _resume_from_state(state))
    except Exception as error:
        if run_directory is not None and state is not None:
            message = f"{type(error).__name__}: {error}"
            if command.mode == "rerender-report":
                _record_report_rerender_failure(
                    run_directory,
                    state,
                    message,
                    dependencies,
                )
                _terminal(dependencies, f"Launcher failed: {message}")
            else:
                _record_report_recovery_failure(
                    command,
                    run_directory,
                    state,
                    message,
                    dependencies,
                )
                _record_terminal_failure(
                    run_directory,
                    state,
                    "failed",
                    message,
                    dependencies,
                )
        else:
            _terminal(dependencies, f"Launcher failed: {type(error).__name__}: {error}")
        return LauncherRunResult(1, "failed", run_directory, _resume_from_state(state))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the production CLI and return its process exit code.

    Parameters
    ----------
    argv : Sequence[str] or None
        Arguments excluding the module name, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Zero for complete/preflight success, 130-style for first-signal
        interruption, two for rejected input, and one for runtime failure.
    """
    command = parse_launcher_command(argv)
    try:
        with _launcher_signal_handlers():
            result = run_launcher(
                command,
                make_production_launcher_dependencies(),
            )
        return result.exit_code
    except _LauncherSignal as error:
        return 128 + error.signum
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
