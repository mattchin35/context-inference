"""Standalone resumable orchestration for CT026 Spike-phase runs.

The launcher owns scalar identity, state, logging, report publication, and the
final exact work cleanup. Numerical preparation and PPC remain in the existing
dependency-injected runtime and pipeline modules.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
from tempfile import NamedTemporaryFile
import threading
import time
from typing import Callable, Iterator, Mapping, Sequence

from src.neural_analysis.lfp_summary.models import (
    LFPSummaryConfig,
    ProgressEvent,
    UnitPopulationConfig,
    canonical_config_json,
    component_fingerprint,
    fingerprint_source_files,
    lfp_summary_config_from_json,
    validate_lfp_summary_config,
)
from src.neural_analysis.lfp_summary.cache import (
    assess_component_status,
    load_or_initialize_manifest,
)
from src.neural_analysis.lfp_summary.pipeline import (
    ComponentRunResult,
    PPCWorkCleanupTarget,
)
from src.neural_analysis.lfp_summary_runtime import _work_fingerprint


_MODULE = "src.neural_analysis.lfp_spike_phase_launcher"
_STATE_SCHEMA = "spike_phase_launcher_state.v1"
_PREFLIGHT_SCHEMA = "spike_phase_preflight.v1"
_REPORT_RECOVERY_SCHEMA = "spike_phase_report_recovery.v1"
_REPORT_RERENDER_SCHEMA = "spike_phase_report_rerender.v1"
_RUN_NAME = re.compile(
    r"^[A-Za-z0-9_.-]+_spike_phase_[A-Za-z0-9_.-]+_"
    r"(preview|final|dry_run)_[A-Za-z0-9_.:-]+$"
)
_PREREQUISITE_CACHE_MEMBERS = frozenset(
    {"manifest.json", "power.npz", "synchrony.npz"}
)
_LEGACY_CACHE_DIRECTORY_NAME = "lfp_summary_cache"
_WORK_ROOT_MEMBERS = frozenset({"prepared_phase", "ppc"})
_PREPARED_REPRESENTATION_MEMBERS = frozenset(
    {"metadata.json", "complete.json", "axes.npz", "valid.npy", "phase.npy"}
)
_PREPARED_METADATA_KEYS = frozenset(
    {
        "generator",
        "analysis_version",
        "schema_version",
        "source_fingerprint",
        "scientific_fingerprint",
        "representation_fingerprint",
        "execution_settings",
        "axes",
        "shapes",
        "dtypes",
        "units",
    }
)
_PREPARED_REPRESENTATION_IDENTITY_KEYS = (
    "generator",
    "analysis_version",
    "source_fingerprint",
    "scientific_fingerprint",
    "axes",
    "shapes",
    "dtypes",
)
_PREPARED_GENERATOR = "prepare_phase_run"
_PREPARED_ANALYSIS_VERSION = "prepared-phase-cache-v1"
_PREPARED_EXECUTION_SETTINGS = {"phase_storage": "complex64/bool"}
_PREPARED_AXES = ["site", "frequency", "trial", "time"]
_PREPARED_DTYPES = {
    "phase": "complex64",
    "valid": "bool",
    "site_trial_valid": "bool",
    "site_trial_exclusion_reason": "<U32",
}
_PREPARED_UNITS = {
    "phase": "dimensionless",
    "relative_time_s": "s",
    "frequency_hz": "Hz",
}
_WORK_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_MAX_RETAINED_IDENTITY_JSON_BYTES = 64 * 1024
_PREREQUISITE_COMPONENT_STATES = (
    ("power", "compatible"),
    ("synchrony", "compatible"),
    ("spike_phase", "missing"),
)
_STAGES = (
    "initialized",
    "preflight_complete",
    "cleanup_prepared",
    "component_complete",
    "report_complete",
    "launcher_artifacts_validated",
    "cleanup_complete",
    "complete",
)


@dataclass(frozen=True)
class LauncherCommand:
    """Validated command-line values with no numerical signal arrays.

    Paths are filesystem identities. Worker/shuffle values are positive counts
    without physical units. Fields not applicable to one mode are ``None``.
    """

    mode: str
    session_path: Path | None = None
    session_metadata: Path | None = None
    probe_label: str | None = None
    shuffle_count: int | None = None
    worker_count: int | None = None
    analysis_root: Path | None = None
    cache_directory: Path | None = None
    dry_run: bool = False
    final_run: bool = False
    run_directory: Path | None = None


@dataclass(frozen=True)
class RepositoryState:
    """Exact tracked-code identity for a launcher invocation."""

    repository_root: Path
    git_commit: str
    tracked_clean: bool

    def __post_init__(self) -> None:
        """Validate repository path, SHA-1 identity, and Boolean cleanliness."""
        if re.fullmatch(r"[0-9a-f]{40}", self.git_commit) is None:
            raise ValueError("git_commit must be 40 lowercase hexadecimal characters")
        if not isinstance(self.tracked_clean, bool):
            raise ValueError("tracked_clean must be Boolean")
        object.__setattr__(self, "repository_root", Path(self.repository_root))


@dataclass(frozen=True)
class LauncherDependencies:
    """Injected metadata, compute, report, cleanup, clock, and output seams."""

    load_active_population: Callable[[Path, str], UnitPopulationConfig]
    build_config: Callable[[Path, UnitPopulationConfig, int, int], LFPSummaryConfig]
    load_trial_count: Callable[[LFPSummaryConfig], int]
    repository_state: Callable[[], RepositoryState]
    source_fingerprints: Callable[[LFPSummaryConfig], Mapping[str, object]]
    component_is_compatible: Callable[[LFPSummaryConfig], bool]
    compute_component: Callable[..., ComponentRunResult]
    validate_component: Callable[[LFPSummaryConfig], None]
    publish_report: Callable[..., object]
    cleanup_target: Callable[[PPCWorkCleanupTarget], None]
    now_utc: Callable[[], str]
    monotonic_seconds: Callable[[], float]
    process_tree_sampler: Callable[[], AbstractContextManager[object]]
    terminal_write: Callable[[str], None]
    repository_commit_is_ancestor: Callable[[str, str], bool] | None = None
    load_metadata_config: Callable[
        [Path, str, Path, int, int], LFPSummaryConfig
    ] | None = None


@dataclass(frozen=True)
class LauncherRunResult:
    """Scalar terminal outcome for one new or resumed launcher invocation."""

    exit_code: int
    status: str
    run_directory: Path | None
    resume_command: str | None


class _LauncherSignal(BaseException):
    """Internal first-signal interruption carrying the POSIX signal number."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


class _LauncherLockError(RuntimeError):
    """A competing process owns the exact launcher run lock."""


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
    new.add_argument("--shuffles", dest="shuffle_count", type=int, choices=(100, 1000), required=True)
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
    analysis_root = (
        values.analysis_root
        or (values.session_path / "analysis_runs" if values.session_path is not None else None)
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
            prepared = _prepare_new_run(command, dependencies)
            config, run_directory, state = prepared
            if bool(state["dry_run"]):
                return LauncherRunResult(0, "preflight_complete", run_directory, None)
            with _launcher_lock(run_directory, state, dependencies):
                return _execute_locked_run(config, run_directory, state, dependencies)
        elif command.mode == "resume":
            run_directory, state = _load_resume_state(command)
            with _launcher_lock(run_directory, state, dependencies):
                config, run_directory, state = _prepare_resume(
                    command,
                    dependencies,
                    run_directory,
                    state,
                )
                return _execute_locked_run(config, run_directory, state, dependencies)
        elif command.mode == "recover-report":
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
        elif command.mode == "rerender-report":
            run_directory, state = _load_completed_state(command)
            with _launcher_lock(run_directory, state, dependencies):
                return _rerender_completed_report(
                    dependencies,
                    run_directory,
                    state,
                )
        else:
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
        status = "cleanup_failed" if state is not None and state.get("active_status") == "cleaning" else "failed"
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
                _record_terminal_failure(run_directory, state, status, str(error), dependencies)
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


def _prepare_new_run(
    command: LauncherCommand,
    dependencies: LauncherDependencies,
) -> tuple[LFPSummaryConfig, Path, dict[str, object]]:
    """Validate identity and atomically initialize one new run directory."""
    assert command.session_path is not None or command.session_metadata is not None
    assert command.probe_label is not None
    assert command.shuffle_count is not None
    assert command.worker_count is not None
    assert command.cache_directory is not None
    repository = dependencies.repository_state()
    if not repository.tracked_clean:
        raise ValueError("launcher requires a clean tracked Git checkout")
    if command.session_metadata is not None:
        if dependencies.load_metadata_config is None:
            raise ValueError("metadata-driven launcher configuration is unavailable")
        config = dependencies.load_metadata_config(
            Path(command.session_metadata),
            command.probe_label,
            Path(command.cache_directory),
            command.shuffle_count,
            command.worker_count,
        )
    else:
        assert command.session_path is not None
        population = dependencies.load_active_population(
            Path(command.session_path),
            command.probe_label,
        )
        config = dependencies.build_config(
            Path(command.session_path),
            population,
            command.shuffle_count,
            command.worker_count,
        )
    cache_directory = _validated_new_cache_directory(
        Path(command.cache_directory),
        Path(config.session_path),
    )
    # The creator supplies scientific settings; the explicit CLI target alone
    # replaces its protected default cache location.
    config = replace(config, output_directory=cache_directory)
    _validate_launcher_config(config, command)
    _validate_new_cache_preflight(config)
    source_fingerprints = dict(dependencies.source_fingerprints(config))
    trial_count = dependencies.load_trial_count(config)
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count < 1:
        raise ValueError("load_trial_count must return a positive integer")
    timestamp = dependencies.now_utc()
    run_kind = "dry_run" if command.dry_run else ("final" if command.final_run else "preview")
    requested_analysis_root = command.analysis_root or config.session_path / "analysis_runs"
    analysis_root = _validated_analysis_root(Path(requested_analysis_root))
    run_name = f"{config.session_id}_spike_phase_{command.probe_label}_{run_kind}_{timestamp}"
    if _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("launcher timestamp/session produced an unsafe run name")
    run_directory = analysis_root / run_name
    if run_directory.exists() or run_directory.is_symlink():
        raise FileExistsError(f"launcher run already exists: {run_directory}")
    analysis_root.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(exist_ok=False)
    resume_command = None if command.dry_run else _resume_command(run_directory)
    identity = _identity_mapping(
        config,
        command,
        repository,
        source_fingerprints,
    )
    preflight = _preflight_mapping(config, trial_count, run_directory, command)
    now = dependencies.now_utc()
    state: dict[str, object] = {
        "schema_version": _STATE_SCHEMA,
        "run_id": run_name,
        "created_utc": now,
        "updated_utc": now,
        "run_kind": run_kind,
        "dry_run": bool(command.dry_run),
        "active_status": "preflight_complete" if command.dry_run else "initialized",
        "completed_stages": ["initialized", "preflight_complete"],
        "resume_command": resume_command,
        "identity": identity,
        "paths": _paths_mapping(config, analysis_root, run_directory),
        "measurements": {},
        "cleanup_request": [],
        "report_directory": None,
        "warnings": [],
        "error": None,
    }
    _write_json_atomic(run_directory / "configuration.json", json.loads(canonical_config_json(config)))
    _write_json_atomic(
        run_directory / "source_identity.json",
        {
            "repository": {
                "root": str(repository.repository_root.resolve(strict=False)),
                "git_commit": repository.git_commit,
                "tracked_clean": repository.tracked_clean,
            },
            "source_fingerprints": source_fingerprints,
            "source_fingerprint_sha256": identity["source_fingerprint_sha256"],
        },
    )
    _write_json_atomic(run_directory / "preflight.json", preflight)
    _write_json_atomic(run_directory / "launcher_state.json", state)
    _append_log(run_directory, _log_event(state, "initialized", "launcher identity created"))
    _write_summary(run_directory, state)
    if resume_command is not None:
        _terminal(dependencies, f"Run directory: {run_directory}")
        _terminal(dependencies, f"Resume command: {resume_command}")
    else:
        _terminal(dependencies, f"Dry-run evidence directory: {run_directory}")
    return config, run_directory, state


def _prepare_resume(
    command: LauncherCommand,
    dependencies: LauncherDependencies,
    run_directory: Path,
    state: dict[str, object],
) -> tuple[LFPSummaryConfig, Path, dict[str, object]]:
    """Reload exact state and reject every identity mismatch before work."""
    del command
    identity = state["identity"]
    assert isinstance(identity, dict)
    repository = dependencies.repository_state()
    required_commit = _required_resume_git_commit(state)
    if not repository.tracked_clean or repository.git_commit != required_commit:
        raise ValueError("resume Git identity does not match saved clean checkout")
    config = _rebuild_saved_config(
        dependencies,
        run_directory,
        state,
        repository,
    )
    _terminal(dependencies, f"Run directory: {run_directory}")
    _terminal(dependencies, f"Resume command: {state['resume_command']}")
    _append_log(
        run_directory,
        _log_event(state, "resume", "validated exact resume identity"),
    )
    return config, run_directory, state


def _rebuild_saved_config(
    dependencies: LauncherDependencies,
    run_directory: Path,
    state: Mapping[str, object],
    repository: RepositoryState,
) -> LFPSummaryConfig:
    """Rebuild and validate the saved configuration and source identity.

    Parameters
    ----------
    dependencies : LauncherDependencies
        Metadata-only configuration and fingerprint seams.
    run_directory : pathlib.Path
        Absolute launcher directory containing immutable identity artifacts.
    state : Mapping[str, object]
        Validated scalar launcher state; no numerical arrays are accepted.
    repository : RepositoryState
        Current clean checkout. Its path must match the computation checkout;
        its commit may be a separately authorized report commit.

    Returns
    -------
    LFPSummaryConfig
        Rebuilt configuration exactly matching the saved computation identity.
    """
    identity = state["identity"]
    assert isinstance(identity, dict)
    session_path = Path(str(identity["session_path"]))
    probe_label = str(identity["probe_label"])
    metadata_path = identity.get("session_metadata")
    if metadata_path is not None:
        snapshot_path = run_directory / "configuration.json"
        config = lfp_summary_config_from_json(snapshot_path.read_text(encoding="ascii"))
    else:
        population = dependencies.load_active_population(session_path, probe_label)
        config = dependencies.build_config(
            session_path,
            population,
            int(identity["shuffle_count"]),
            int(identity["requested_worker_count"]),
        )
    cache_directory = _saved_cache_directory(identity)
    config = replace(config, output_directory=cache_directory)
    rebuilt_command = LauncherCommand(
        mode="new",
        session_path=session_path,
        session_metadata=Path(str(metadata_path)) if metadata_path is not None else None,
        probe_label=probe_label,
        shuffle_count=int(identity["shuffle_count"]),
        worker_count=int(identity["requested_worker_count"]),
        analysis_root=run_directory.parent,
        cache_directory=cache_directory,
        dry_run=False,
        final_run=bool(identity["final_run"]),
    )
    _validate_launcher_config(config, rebuilt_command)
    sources = dict(dependencies.source_fingerprints(config))
    computation_repository = RepositoryState(
        repository.repository_root,
        str(identity["git_commit"]),
        True,
    )
    rebuilt_identity = _identity_mapping(
        config,
        rebuilt_command,
        computation_repository,
        sources,
    )
    if rebuilt_identity != identity:
        raise ValueError("resume source/config/population identity does not match saved run")
    _validate_configuration_snapshot(run_directory, config)
    expected_paths = _paths_mapping(config, run_directory.parent, run_directory)
    if state["paths"] != expected_paths:
        raise ValueError("resume path identity does not match saved run")
    _validate_launcher_identity_artifacts(run_directory, state)
    return config


def _required_resume_git_commit(state: Mapping[str, object]) -> str:
    """Return the exact commit required for the unfinished launcher stage."""
    identity = state.get("identity")
    measurements = state.get("measurements")
    completed = state.get("completed_stages")
    if not isinstance(identity, Mapping) or not isinstance(measurements, Mapping):
        raise ValueError("launcher identity/measurements are unavailable")
    report_commit = measurements.get("report_git_commit")
    if (
        isinstance(completed, list)
        and "report_complete" in completed
        and isinstance(report_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", report_commit) is not None
    ):
        return report_commit
    computation_commit = identity.get("git_commit")
    if not isinstance(computation_commit, str):
        raise ValueError("launcher computation Git identity is unavailable")
    return computation_commit


def _prepare_report_recovery(
    dependencies: LauncherDependencies,
    run_directory: Path,
    state: dict[str, object],
) -> tuple[LFPSummaryConfig, Path, dict[str, object]]:
    """Authorize report-only reuse of one exact completed component.

    All configuration, population, source, path, and component checks finish
    before report state is changed. Numerical preparation and PPC callbacks are
    not reachable from this function.
    """
    completed = state["completed_stages"]
    assert isinstance(completed, list)
    if completed != list(_STAGES[:4]):
        raise ValueError(
            "recover-report requires an incomplete run ending at component_complete"
        )
    if state.get("report_directory") is not None:
        raise ValueError("recover-report requires an absent published report")
    cleanup_request = state.get("cleanup_request")
    if not isinstance(cleanup_request, list) or not cleanup_request:
        raise ValueError("recover-report requires a saved cleanup target")
    identity = state["identity"]
    assert isinstance(identity, dict)
    repository = dependencies.repository_state()
    computation_commit = str(identity["git_commit"])
    if not repository.tracked_clean:
        raise ValueError("recover-report requires a clean tracked Git checkout")
    if repository.git_commit == computation_commit:
        raise ValueError("use ordinary resume at the saved computation commit")
    is_ancestor = dependencies.repository_commit_is_ancestor
    if is_ancestor is None or not is_ancestor(
        computation_commit,
        repository.git_commit,
    ):
        raise ValueError("report checkout must descend from the computation commit")
    config = _rebuild_saved_config(
        dependencies,
        run_directory,
        state,
        repository,
    )
    dependencies.validate_component(config)

    audit_path = run_directory / "report_recovery.json"
    original_error = state.get("error")
    if audit_path.is_file():
        prior = _load_report_recovery_audit(audit_path, state)
        original_error = prior["original_error"]
    if not isinstance(original_error, str) or not original_error:
        raise ValueError("recover-report requires the original report error")
    now = dependencies.now_utc()
    audit = {
        "schema_version": _REPORT_RECOVERY_SCHEMA,
        "run_id": state["run_id"],
        "status": "initialized",
        "computation_git_commit": computation_commit,
        "report_git_commit": repository.git_commit,
        "original_error": original_error,
        "recovery_utc": now,
        "updated_utc": now,
        "component_reused": True,
        "error": None,
    }
    measurements = state["measurements"]
    assert isinstance(measurements, dict)
    measurements.update(
        {
            "component_reused": True,
            "computation_git_commit": computation_commit,
            "report_git_commit": repository.git_commit,
            "report_recovery_utc": now,
            "report_recovery_status": "initialized",
        }
    )
    state["active_status"] = "report_recovery"
    state["error"] = None
    _write_json_atomic(audit_path, audit)
    _persist_state(run_directory, state, dependencies)
    _append_log(
        run_directory,
        _log_event(state, "report_recovery", "validated report-only recovery identity"),
    )
    _write_summary(run_directory, state)
    _terminal(dependencies, f"Run directory: {run_directory}")
    _terminal(dependencies, f"Resume command: {state['resume_command']}")
    return config, run_directory, state


def _load_resume_state(
    command: LauncherCommand,
) -> tuple[Path, dict[str, object]]:
    """Load only the path-bound state needed to acquire its advisory lock."""
    if command.run_directory is None:
        raise ValueError("resume requires run_directory")
    requested = Path(command.run_directory)
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("resume run_directory is not a valid launcher directory")
    run_directory = requested.resolve(strict=True)
    if _RUN_NAME.fullmatch(run_directory.name) is None:
        raise ValueError("resume run_directory is not a valid launcher directory")
    state = _load_state(run_directory / "launcher_state.json")
    if bool(state["dry_run"]):
        raise ValueError("dry-run evidence directories are not resumable")
    if state["active_status"] == "complete":
        raise ValueError("completed launcher runs are not resumable")
    paths = state["paths"]
    assert isinstance(paths, dict)
    if paths.get("run_directory") != str(run_directory):
        raise ValueError("resume directory does not match saved path identity")
    return run_directory, state


def _load_completed_state(
    command: LauncherCommand,
) -> tuple[Path, dict[str, object]]:
    """Load one path-bound completed launcher state for report rerendering.

    Parameters
    ----------
    command : LauncherCommand
        A ``rerender-report`` command with one existing launcher directory.

    Returns
    -------
    tuple[pathlib.Path, dict[str, object]]
        Resolved launcher directory and validated scalar state mapping.

    Raises
    ------
    ValueError
        If the path is unsafe or the run is not exactly complete.
    """
    if command.run_directory is None:
        raise ValueError("rerender-report requires run_directory")
    requested = Path(command.run_directory)
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("rerender run_directory is not a valid launcher directory")
    run_directory = requested.resolve(strict=True)
    if _RUN_NAME.fullmatch(run_directory.name) is None:
        raise ValueError("rerender run_directory is not a valid launcher directory")
    state = _load_state(run_directory / "launcher_state.json")
    if (
        bool(state["dry_run"])
        or state["active_status"] != "complete"
        or state["completed_stages"] != list(_STAGES)
    ):
        raise ValueError("rerender-report requires a completed launcher run")
    paths = state["paths"]
    assert isinstance(paths, dict)
    if paths.get("run_directory") != str(run_directory):
        raise ValueError("rerender directory does not match saved path identity")
    return run_directory, state


def _rerender_completed_report(
    dependencies: LauncherDependencies,
    run_directory: Path,
    state: dict[str, object],
) -> LauncherRunResult:
    """Publish a new report from one immutable completed numerical cache.

    Parameters
    ----------
    dependencies : LauncherDependencies
        Metadata, component-validation, report, Git, clock, and terminal seams.
        Compute and cleanup seams are intentionally unreachable.
    run_directory : pathlib.Path
        Resolved completed launcher directory containing scalar provenance.
    state : dict[str, object]
        Validated completed launcher state. It is not persisted until the new
        report has been published and validated.

    Returns
    -------
    LauncherRunResult
        Complete result pointing to the unchanged launcher run directory.

    Raises
    ------
    ValueError
        If Git, saved identity, component, prior report, or new report fails
        closed validation.
    """
    repository = dependencies.repository_state()
    previous_report_commit = _previous_report_git_commit(state)
    if not repository.tracked_clean:
        raise ValueError("rerender-report requires a clean tracked Git checkout")
    if repository.git_commit == previous_report_commit:
        raise ValueError("rerender-report requires a newer report commit")
    is_ancestor = dependencies.repository_commit_is_ancestor
    if is_ancestor is None or not is_ancestor(
        previous_report_commit,
        repository.git_commit,
    ):
        raise ValueError("rerender report checkout must descend from the prior report commit")

    config = _rebuild_saved_config(
        dependencies,
        run_directory,
        state,
        repository,
    )
    _validate_launcher_artifacts(run_directory, state)
    dependencies.validate_component(config)

    identity = state["identity"]
    paths = state["paths"]
    assert isinstance(identity, dict)
    assert isinstance(paths, dict)
    previous_report_directory = str(state["report_directory"])
    now = dependencies.now_utc()
    audit: dict[str, object] = {
        "schema_version": _REPORT_RERENDER_SCHEMA,
        "run_id": state["run_id"],
        "status": "initialized",
        "computation_git_commit": identity["git_commit"],
        "previous_report_git_commit": previous_report_commit,
        "rerender_git_commit": repository.git_commit,
        "previous_report_directory": previous_report_directory,
        "new_report_directory": None,
        "rerender_utc": now,
        "updated_utc": now,
        "component_reused": True,
        "error": None,
    }
    audit_path = run_directory / "report_rerender.json"
    _write_json_atomic(audit_path, audit)

    report_started = _clock(dependencies.monotonic_seconds)
    report_parent = Path(str(paths["report_parent"]))
    report_parent.mkdir(parents=True, exist_ok=True)
    report_result = dependencies.publish_report(
        config=config,
        run_parent=report_parent,
        run_kind=str(state["run_kind"]),
        component_wall_time_s=float(
            _measurement(state, "component_total_seconds", 0.0)
        ),
        component_peak_memory_bytes=int(
            _measurement(state, "process_rss_bytes", 0)
        ),
        report_measurements=_report_measurements(config, state),
        deferred_cleanup=None,
    )
    new_report_directory = str(
        Path(report_result.run_directory).resolve(strict=False)
    )

    candidate_state = json.loads(json.dumps(state, allow_nan=False))
    candidate_state["report_directory"] = new_report_directory
    candidate_state["active_status"] = "complete"
    candidate_state["error"] = None
    measurements = candidate_state["measurements"]
    assert isinstance(measurements, dict)
    legacy_child_count = measurements.pop("measured_active_worker_count", None)
    if (
        "maximum_child_process_count" not in measurements
        and legacy_child_count is not None
    ):
        measurements["maximum_child_process_count"] = legacy_child_count
    measurements.update(
        {
            "report_rerender_git_commit": repository.git_commit,
            "report_rerender_utc": now,
            "report_rerender_seconds": max(
                0.0,
                _clock(dependencies.monotonic_seconds) - report_started,
            ),
        }
    )
    _validate_existing_report(candidate_state)

    audit["status"] = "validated"
    audit["new_report_directory"] = new_report_directory
    audit["updated_utc"] = dependencies.now_utc()
    _validate_report_rerender_audit(audit, candidate_state)
    _write_json_atomic(audit_path, audit)
    _persist_state(run_directory, candidate_state, dependencies)

    audit["status"] = "complete"
    audit["updated_utc"] = dependencies.now_utc()
    _validate_report_rerender_audit(audit, candidate_state)
    _write_json_atomic(audit_path, audit)
    _append_log(
        run_directory,
        _log_event(
            candidate_state,
            "report_rerender_complete",
            "completed cache-only immutable report rerender",
        ),
    )
    _write_summary(run_directory, candidate_state)
    _validate_launcher_artifacts(run_directory, candidate_state)
    _terminal(dependencies, f"Run directory: {run_directory}")
    _terminal(dependencies, f"New report directory: {new_report_directory}")
    return LauncherRunResult(
        0,
        "complete",
        run_directory,
        _resume_from_state(candidate_state),
    )


def _previous_report_git_commit(state: Mapping[str, object]) -> str:
    """Return the commit that produced the completed run's current report.

    Parameters
    ----------
    state : Mapping[str, object]
        Validated completed launcher state with scalar measurement provenance.

    Returns
    -------
    str
        Forty-character lowercase Git commit for the current report.
    """
    identity = state.get("identity")
    measurements = state.get("measurements")
    if not isinstance(identity, Mapping) or not isinstance(measurements, Mapping):
        raise ValueError("launcher report Git provenance is unavailable")
    value = (
        measurements.get("report_rerender_git_commit")
        or measurements.get("report_git_commit")
        or identity.get("git_commit")
    )
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("launcher report Git provenance is invalid")
    return value


def _validate_launcher_config(config: LFPSummaryConfig, command: LauncherCommand) -> None:
    """Validate fixed CT026 launcher scientific and execution policies."""
    validate_lfp_summary_config(config)
    if config.unit_population is None or config.unit_population.probe_label != command.probe_label:
        raise ValueError("launcher configuration does not match explicit probe")
    if config.ppc.shuffle_count != command.shuffle_count:
        raise ValueError("launcher configuration does not match requested shuffles")
    if config.ppc_execution.worker_count != command.worker_count:
        raise ValueError("launcher configuration does not match requested workers")
    if config.phase.absolute_amplitude_thresholds:
        raise ValueError("absolute amplitude thresholds are unsupported until WP13")
    if not config.ppc_execution.checkpoint_enabled or config.ppc_execution.checkpoint_retention != "incomplete_only":
        raise ValueError("launcher requires incomplete_only PPC checkpoints")


def _validated_analysis_root(path: Path) -> Path:
    """Return a resolved nonsymlink analysis root path."""
    if path.exists() and path.is_symlink():
        raise ValueError("analysis_root must not be a symbolic link")
    return path.resolve(strict=False)


def _validated_new_cache_directory(cache_directory: Path, session_path: Path) -> Path:
    """Resolve one safe direct child of the selected session's processed path.

    Parameters
    ----------
    cache_directory : pathlib.Path
        User-supplied corrected-cache directory. It must exist, be a real
        directory, and be a direct child of ``session_path / 'processed'``.
    session_path : pathlib.Path
        Selected session directory. Its path and the target path may not cross
        symbolic links or lexical parent-directory aliases.

    Returns
    -------
    pathlib.Path
        Absolute, normalized cache directory used in every saved artifact.
    """
    resolved_session = _resolved_nonsymlink_path(session_path, "session_path")
    resolved_cache = _resolved_nonsymlink_path(cache_directory, "cache_directory")
    processed_directory = resolved_session / "processed"
    if processed_directory.is_symlink():
        raise ValueError("session processed directory must not be a symbolic link")
    resolved_processed = processed_directory.resolve(strict=False)
    if resolved_cache.name == _LEGACY_CACHE_DIRECTORY_NAME:
        raise ValueError("the legacy lfp_summary_cache directory is protected")
    if resolved_cache.parent != resolved_processed:
        raise ValueError(
            "cache_directory must be a direct child of the selected session processed directory"
        )
    if not resolved_cache.is_dir():
        raise ValueError("cache_directory must be an existing directory")
    return resolved_cache


def _resolved_nonsymlink_path(path: Path, label: str) -> Path:
    """Return a normalized path after rejecting aliases and symlink components.

    Parameters
    ----------
    path : pathlib.Path
        Absolute or current-directory-relative filesystem identity.
    label : str
        Human-readable argument name included in validation errors.

    Returns
    -------
    pathlib.Path
        Absolute normalized path. This helper does not create filesystem
        entries and accepts a final path that does not yet exist.
    """
    requested = Path(path)
    if ".." in requested.parts:
        raise ValueError(f"{label} must not contain a parent-directory alias")
    absolute_path = requested if requested.is_absolute() else Path.cwd() / requested
    current = Path(absolute_path.anchor)
    for part in absolute_path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symbolic link")
    return absolute_path.resolve(strict=False)


def _validate_new_cache_preflight(config: LFPSummaryConfig) -> None:
    """Validate immutable prerequisite cache evidence without loading arrays.

    Parameters
    ----------
    config : LFPSummaryConfig
        Fully validated active configuration whose output directory is the
        selected corrected cache. No trial table, phase array, PPC work, or
        launcher run directory is opened or created by this check.

    Returns
    -------
    None
        The cache is only read through manifest JSON and public NPZ header
        checks. Any incompatible prerequisite raises ``ValueError``.
    """
    cache_directory = config.output_directory
    work_root = cache_directory.parent / "lfp_summary_work"
    _validate_retained_work_root(work_root, config)
    members = tuple(cache_directory.iterdir())
    member_names = {member.name for member in members}
    if member_names != _PREREQUISITE_CACHE_MEMBERS:
        raise ValueError(
            "corrected cache must contain exactly manifest.json, power.npz, and synchrony.npz"
        )
    for member in members:
        if member.is_symlink() or not member.is_file():
            raise ValueError("corrected cache members must be regular non-symbolic-link files")
    manifest = load_or_initialize_manifest(cache_directory, config)
    for component, expected_status in _PREREQUISITE_COMPONENT_STATES:
        status = assess_component_status(
            cache_directory,
            component,
            config,
            manifest,
            validate_headers_only=True,
        )
        if status.status != expected_status:
            differences = "; ".join(status.differences)
            raise ValueError(
                f"{component} prerequisite must be {expected_status}, "
                f"found {status.status}: {differences or 'no details'}"
            )


def _validate_retained_work_root(work_root: Path, config: LFPSummaryConfig) -> None:
    """Classify optional retained work through stable directory descriptors.

    Parameters
    ----------
    work_root : pathlib.Path
        Exact session-local ``lfp_summary_work`` path. An absent path is valid;
        an existing path must retain a stable real-directory identity while it
        is inspected. No numerical array payload, shape, axis, or unit data is
        opened by this preflight classifier.
    config : LFPSummaryConfig
        Validated active configuration. Only ``schema_version`` is used to
        validate retained metadata generated by the current writer contract.

    Returns
    -------
    None
        The existing tree is read-only. Only absent/empty safe layouts and
        complete lock-free retained prepared representations are accepted.

    Raises
    ------
    ValueError
        If an entry is linked, malformed, substituted, or structurally unsafe.
    FileExistsError
        If the exact ``ppc`` container has any child, preserving it for resume.
    """
    root_status = _lstat_or_none(work_root)
    if root_status is None:
        return
    with _opened_anchored_directory(
        work_root,
        "work root",
        expected_status=root_status,
    ) as work_root_fd:
        initial_root_status = os.fstat(work_root_fd)
        root_members = _stable_directory_member_names(work_root_fd, "work root")
        if set(root_members) - _WORK_ROOT_MEMBERS:
            raise ValueError("work root contains unexpected members")
        root_member_statuses = _anchored_member_statuses(
            work_root_fd,
            root_members,
            "work root member",
        )

        if "prepared_phase" in root_members:
            prepared_path = work_root / "prepared_phase"
            with _opened_anchored_directory(
                prepared_path,
                "prepared_phase container",
                parent_descriptor=work_root_fd,
                member_name="prepared_phase",
                expected_status=root_member_statuses["prepared_phase"],
            ) as prepared_fd:
                _validate_retained_prepared_phase(
                    prepared_path,
                    prepared_fd,
                    config,
                )

        if "ppc" in root_members:
            ppc_path = work_root / "ppc"
            with _opened_anchored_directory(
                ppc_path,
                "ppc container",
                parent_descriptor=work_root_fd,
                member_name="ppc",
                expected_status=root_member_statuses["ppc"],
            ) as ppc_fd:
                if _stable_directory_member_names(ppc_fd, "ppc container"):
                    raise FileExistsError(f"pre-existing PPC work: {ppc_path}")

            # A new PPC child can appear after the first descriptor closes.
            # Reopen the same anchored entry before accepting the work root.
            with _opened_anchored_directory(
                ppc_path,
                "ppc container",
                parent_descriptor=work_root_fd,
                member_name="ppc",
                expected_status=root_member_statuses["ppc"],
            ) as final_ppc_fd:
                if _stable_directory_member_names(final_ppc_fd, "ppc container"):
                    raise FileExistsError(f"pre-existing PPC work: {ppc_path}")

        final_root_members = _stable_directory_member_names(work_root_fd, "work root")
        if final_root_members != root_members:
            raise ValueError("work root inventory changed during preflight")
        _require_unchanged_anchored_members(
            work_root_fd,
            root_member_statuses,
            "work root member",
        )
        _require_same_directory_inventory(
            initial_root_status,
            os.fstat(work_root_fd),
            "work root",
        )
        _require_stable_directory_path(work_root, work_root_fd, "work root")


def _validate_retained_prepared_phase(
    prepared_path: Path,
    prepared_descriptor: int,
    config: LFPSummaryConfig,
) -> None:
    """Validate prepared representations using descriptors and small JSON only.

    Parameters
    ----------
    prepared_path : pathlib.Path
        Display path for the already-open ``prepared_phase`` directory. It is
        rechecked against ``prepared_descriptor`` before return.
    prepared_descriptor : int
        Open read-only directory descriptor anchored to ``prepared_path``.
        Direct children are opened relative to this descriptor, never through
        an unverified replacement path.
    config : LFPSummaryConfig
        Active configuration supplying the canonical metadata schema version.
        No trial table or numerical phase data is loaded.

    Returns
    -------
    None
        Every accepted child has an exact writer metadata schema, a matching
        representation fingerprint, and a one-field completion certificate.

    Raises
    ------
    ValueError
        If an entry changes identity, inventory, metadata, or completion while
        preflight is in progress.
    """
    initial_prepared_status = os.fstat(prepared_descriptor)
    representation_names = _stable_directory_member_names(
        prepared_descriptor,
        "prepared_phase container",
    )
    for fingerprint in representation_names:
        if _WORK_FINGERPRINT.fullmatch(fingerprint) is None:
            raise ValueError("prepared representation name is not a fingerprint")
    representation_statuses = _anchored_member_statuses(
        prepared_descriptor,
        representation_names,
        "prepared representation",
    )
    for fingerprint in representation_names:
        representation_path = prepared_path / fingerprint
        with _opened_anchored_directory(
            representation_path,
            "prepared representation",
            parent_descriptor=prepared_descriptor,
            member_name=fingerprint,
            expected_status=representation_statuses[fingerprint],
        ) as representation_fd:
            _validate_retained_representation(
                representation_path,
                representation_fd,
                fingerprint,
                config,
            )

    final_representation_names = _stable_directory_member_names(
        prepared_descriptor,
        "prepared_phase container",
    )
    if final_representation_names != representation_names:
        raise ValueError("prepared_phase inventory changed during preflight")
    _require_unchanged_anchored_members(
        prepared_descriptor,
        representation_statuses,
        "prepared representation",
    )
    _require_same_directory_inventory(
        initial_prepared_status,
        os.fstat(prepared_descriptor),
        "prepared_phase container",
    )
    _require_stable_directory_path(
        prepared_path,
        prepared_descriptor,
        "prepared_phase container",
    )


def _validate_retained_representation(
    representation_path: Path,
    representation_descriptor: int,
    fingerprint: str,
    config: LFPSummaryConfig,
) -> None:
    """Validate one retained representation without opening its array payloads.

    Parameters
    ----------
    representation_path : pathlib.Path
        Display path for an already-open fingerprint-named representation.
    representation_descriptor : int
        Stable read-only descriptor for ``representation_path``. Named members
        are statted/opened relative to it, preventing replacement-path use.
    fingerprint : str
        Lowercase 64-hex representation directory name. It must match the
        canonical hash recomputed from stored metadata.
    config : LFPSummaryConfig
        Active configuration supplying the exact writer schema version.

    Returns
    -------
    None
        Only ``metadata.json`` and ``complete.json`` bytes are read, each under
        64 KiB. ``phase.npy``, ``valid.npy``, and ``axes.npz`` stay unopened.

    Raises
    ------
    ValueError
        If the member inventory/type/identity changes or violates the writer
        contract.
    """
    initial_representation_status = os.fstat(representation_descriptor)
    if not stat.S_ISDIR(initial_representation_status.st_mode):
        raise ValueError("prepared representation descriptor is not a directory")
    member_names = _stable_directory_member_names(
        representation_descriptor,
        "prepared representation",
    )
    if set(member_names) != _PREPARED_REPRESENTATION_MEMBERS:
        raise ValueError("prepared representation members are incomplete or unexpected")
    initial_member_statuses: dict[str, os.stat_result] = {}
    for member_name in member_names:
        member_path = representation_path / member_name
        expected_status = _lstat_or_none(member_path)
        if expected_status is None:
            raise ValueError(f"prepared representation member is missing: {member_path}")
        _require_regular_nonsymlink_file(
            member_path,
            expected_status,
            "prepared representation member",
        )
        descriptor_status = _stat_member_without_following(
            representation_descriptor,
            member_name,
            "prepared representation member",
        )
        _require_same_entry(
            expected_status,
            descriptor_status,
            "prepared representation member",
        )
        initial_member_statuses[member_name] = descriptor_status

    metadata = _read_bounded_identity_json(
        representation_descriptor,
        "metadata.json",
        representation_path / "metadata.json",
    )
    completion = _read_bounded_identity_json(
        representation_descriptor,
        "complete.json",
        representation_path / "complete.json",
    )
    _validate_prepared_metadata(metadata, fingerprint, config.schema_version)
    _validate_prepared_completion(completion, fingerprint)

    final_member_names = _stable_directory_member_names(
        representation_descriptor,
        "prepared representation",
    )
    if final_member_names != member_names:
        raise ValueError("prepared representation inventory changed during preflight")
    for member_name, initial_status in initial_member_statuses.items():
        final_status = _stat_member_without_following(
            representation_descriptor,
            member_name,
            "prepared representation member",
        )
        _require_same_entry(
            initial_status,
            final_status,
            "prepared representation member",
        )
    _require_same_directory_inventory(
        initial_representation_status,
        os.fstat(representation_descriptor),
        "prepared representation",
    )
    _require_stable_directory_path(
        representation_path,
        representation_descriptor,
        "prepared representation",
    )


@contextmanager
def _opened_anchored_directory(
    path: Path,
    label: str,
    *,
    parent_descriptor: int | None = None,
    member_name: str | None = None,
    expected_status: os.stat_result | None = None,
) -> Iterator[int]:
    """Open one real directory and bind it to its checked ``lstat`` identity.

    Parameters
    ----------
    path : pathlib.Path
        Expected directory path used solely for link-preserving identity checks
        and diagnostics. It must not be trusted after its status is read.
    label : str
        Human-readable role for a validation error.
    parent_descriptor : int or None, default=None
        Stable parent directory descriptor. When supplied, ``member_name`` is
        opened relative to it; otherwise ``path`` is opened directly.
    member_name : str or None, default=None
        Direct child name to open relative to ``parent_descriptor``. It has no
        array, axis, shape, or physical-unit content.
    expected_status : os.stat_result or None, default=None
        Previously captured nofollow status for ``path``. Supplying it binds
        the open operation to an earlier existence check instead of performing
        a second path lookup after a possible replacement race.

    Yields
    ------
    int
        Open read-only directory descriptor whose ``fstat`` identity exactly
        matches the preceding nonsymlink ``lstat`` result.

    Raises
    ------
    ValueError
        If the path is missing, linked, non-directory, substituted, or cannot
        be opened with no-follow semantics.
    """
    status = expected_status if expected_status is not None else _lstat_or_none(path)
    if status is None:
        raise ValueError(f"{label} is missing: {path}")
    _require_real_directory(path, status, label)
    if parent_descriptor is None:
        if member_name is not None:
            raise ValueError("member_name requires a parent directory descriptor")
        open_target: str | Path = path
    else:
        if not isinstance(member_name, str) or not member_name:
            raise ValueError("anchored child directory requires a member name")
        open_target = member_name
    flags = _directory_open_flags()
    try:
        descriptor = os.open(open_target, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError(f"cannot safely open {label}: {path}") from error
    try:
        actual_status = os.fstat(descriptor)
        _require_real_directory(path, actual_status, label)
        _require_same_entry(status, actual_status, label)
        yield descriptor
    finally:
        os.close(descriptor)


def _directory_open_flags() -> int:
    """Return mandatory no-follow read-only flags for directory inspection.

    Returns
    -------
    int
        POSIX ``os.open`` flags for a read-only directory descriptor with close
        on exec and symbolic-link following disabled. No data arrays or units
        are involved.

    Raises
    ------
    ValueError
        If the host lacks mandatory directory or no-follow support, because
        preflight must fail closed rather than use an unsafe fallback.
    """
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise ValueError("safe retained-work inspection requires O_NOFOLLOW and O_DIRECTORY")
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _file_open_flags() -> int:
    """Return mandatory no-follow nonblocking flags for identity JSON reads.

    Returns
    -------
    int
        POSIX ``os.open`` flags that prevent link following and ensure a raced
        FIFO cannot block this metadata-only preflight. No numerical payload is
        opened with these flags.

    Raises
    ------
    ValueError
        If the host lacks no-follow support and cannot fail safely.
    """
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ValueError("safe retained-work inspection requires O_NOFOLLOW")
    return os.O_RDONLY | no_follow | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _stable_directory_member_names(descriptor: int, label: str) -> tuple[str, ...]:
    """Return a stable sorted direct inventory from an open directory descriptor.

    Parameters
    ----------
    descriptor : int
        Open stable directory descriptor. Its contents are listed directly;
        no replacement path, numerical payload, array axis, or unit is read.
    label : str
        Human-readable role for a validation error.

    Returns
    -------
    tuple[str, ...]
        Sorted direct child basenames after matching before/after directory
        metadata confirms that the inventory did not change while listing.

    Raises
    ------
    ValueError
        If descriptor type or directory inventory metadata changes during the
        listing, or the inventory cannot be listed.
    """
    before = os.fstat(descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} descriptor is not a directory")
    try:
        member_names = tuple(sorted(os.listdir(descriptor)))
    except OSError as error:
        raise ValueError(f"cannot inspect {label} inventory") from error
    after = os.fstat(descriptor)
    _require_same_directory_inventory(before, after, label)
    return member_names


def _read_bounded_identity_json(
    parent_descriptor: int,
    member_name: str,
    path: Path,
) -> object:
    """Read one anchored JSON identity file through a capped no-follow descriptor.

    Parameters
    ----------
    parent_descriptor : int
        Stable descriptor for the prepared representation containing the JSON.
    member_name : str
        Exact direct filename, ``metadata.json`` or ``complete.json``.
    path : pathlib.Path
        Display path whose immediate pre-open ``lstat`` identity must match the
        descriptor opened relative to ``parent_descriptor``. No numerical
        arrays, axes, shapes, or units are read.

    Returns
    -------
    object
        Parsed UTF-8 JSON value from at most 64 KiB. The caller validates its
        schema and fingerprint before treating it as a retained identity.

    Raises
    ------
    ValueError
        If the entry is linked, nonregular, substituted, oversized, changed
        while read, malformed, or cannot be safely opened without blocking.
    """
    expected_status = _lstat_or_none(path)
    if expected_status is None:
        raise ValueError(f"retained identity file is missing: {path}")
    _require_regular_nonsymlink_file(path, expected_status, "retained identity file")
    if expected_status.st_size > _MAX_RETAINED_IDENTITY_JSON_BYTES:
        raise ValueError(f"retained identity JSON exceeds 64 KiB: {path}")
    try:
        descriptor = os.open(
            member_name,
            _file_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ValueError(f"cannot safely open retained identity JSON: {path}") from error
    try:
        opened_status = os.fstat(descriptor)
        _require_regular_nonsymlink_file(
            path,
            opened_status,
            "retained identity file",
        )
        _require_same_entry(expected_status, opened_status, "retained identity file")
        if opened_status.st_size > _MAX_RETAINED_IDENTITY_JSON_BYTES:
            raise ValueError(f"retained identity JSON exceeds 64 KiB: {path}")
        encoded = _read_descriptor_at_most(
            descriptor,
            _MAX_RETAINED_IDENTITY_JSON_BYTES,
            path,
        )
        final_status = os.fstat(descriptor)
        _require_same_entry(opened_status, final_status, "retained identity file")
        if final_status.st_size != opened_status.st_size or len(encoded) != final_status.st_size:
            raise ValueError(f"retained identity JSON changed while read: {path}")
        current_status = _stat_member_without_following(
            parent_descriptor,
            member_name,
            "retained identity file",
        )
        _require_same_entry(
            final_status,
            current_status,
            "retained identity file",
        )
    finally:
        os.close(descriptor)
    try:
        return json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"retained identity JSON is malformed: {path}") from error


def _read_descriptor_at_most(descriptor: int, maximum_bytes: int, path: Path) -> bytes:
    """Read at most one bounded regular-file payload from an open descriptor.

    Parameters
    ----------
    descriptor : int
        Open nonblocking descriptor for a regular metadata file. It never
        identifies a numerical work array.
    maximum_bytes : int
        Nonnegative maximum allowed payload length in bytes. The helper reads
        one additional byte to detect growth beyond this ceiling.
    path : pathlib.Path
        Display path for a validation error only.

    Returns
    -------
    bytes
        Entire metadata payload when it is no longer than ``maximum_bytes``.

    Raises
    ------
    ValueError
        If the read fails or exceeds the fixed ceiling.
    """
    try:
        encoded = os.read(descriptor, maximum_bytes + 1)
    except OSError as error:
        raise ValueError(f"cannot read retained identity JSON: {path}") from error
    if len(encoded) > maximum_bytes:
        raise ValueError(f"retained identity JSON exceeds 64 KiB: {path}")
    return encoded


def _validate_prepared_metadata(
    record: object,
    expected_fingerprint: str,
    expected_schema_version: str,
) -> None:
    """Require the exact current writer metadata schema and self-hash.

    Parameters
    ----------
    record : object
        Parsed bounded JSON metadata. It may contain scalar/list/mapping JSON
        values but never numerical payload arrays.
    expected_fingerprint : str
        Fingerprint from the anchored representation directory name.
    expected_schema_version : str
        Current configuration schema identifier that the writer records.

    Returns
    -------
    None
        The metadata is inspected only; no filesystem or numerical data is
        mutated.

    Raises
    ------
    ValueError
        If keys, canonical values, identifier formats, shape/dtype/unit schema,
        or the real writer representation fingerprint are invalid.
    """
    if not isinstance(record, dict) or set(record) != _PREPARED_METADATA_KEYS:
        raise ValueError("prepared metadata has an invalid key set")
    if (
        record.get("generator") != _PREPARED_GENERATOR
        or record.get("analysis_version") != _PREPARED_ANALYSIS_VERSION
        or record.get("schema_version") != expected_schema_version
        or record.get("execution_settings") != _PREPARED_EXECUTION_SETTINGS
        or record.get("axes") != _PREPARED_AXES
        or record.get("dtypes") != _PREPARED_DTYPES
        or record.get("units") != _PREPARED_UNITS
    ):
        raise ValueError("prepared metadata differs from the current writer schema")
    for key in (
        "generator",
        "analysis_version",
        "schema_version",
        "source_fingerprint",
        "scientific_fingerprint",
        "representation_fingerprint",
    ):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError("prepared metadata identifier type is invalid")
    if (
        _WORK_FINGERPRINT.fullmatch(record["source_fingerprint"]) is None
        or _WORK_FINGERPRINT.fullmatch(record["scientific_fingerprint"]) is None
        or _WORK_FINGERPRINT.fullmatch(record["representation_fingerprint"]) is None
    ):
        raise ValueError("prepared metadata fingerprint format is invalid")
    _validate_prepared_shape_schema(record["shapes"])
    identity = {
        key: record[key] for key in _PREPARED_REPRESENTATION_IDENTITY_KEYS
    }
    if _work_fingerprint(identity) != record["representation_fingerprint"]:
        raise ValueError("prepared metadata representation fingerprint is invalid")
    if record["representation_fingerprint"] != expected_fingerprint:
        raise ValueError("prepared metadata fingerprint differs from its directory")


def _validate_prepared_shape_schema(shapes: object) -> None:
    """Validate writer-declared prepared array shape metadata without arrays.

    Parameters
    ----------
    shapes : object
        JSON shape mapping from retained metadata. Expected arrays use
        `(site, frequency, trial, time)` and `(site, trial)` axes, but no array
        payload, unit conversion, or axis values are opened.

    Returns
    -------
    None
        The mapping is inspected without mutation.

    Raises
    ------
    ValueError
        If keys, integer dimensions, or writer-required axis relationships are
        malformed.
    """
    expected_keys = {
        "phase",
        "valid",
        "site_trial_valid",
        "site_trial_exclusion_reason",
    }
    if not isinstance(shapes, dict) or set(shapes) != expected_keys:
        raise ValueError("prepared metadata shape schema is invalid")
    phase = shapes["phase"]
    valid = shapes["valid"]
    site_trial_valid = shapes["site_trial_valid"]
    exclusion_reason = shapes["site_trial_exclusion_reason"]
    if not _nonnegative_integer_list(phase, 4):
        raise ValueError("prepared phase shape is invalid")
    if valid != phase or site_trial_valid != [phase[0], phase[2]]:
        raise ValueError("prepared validity shape differs from phase shape")
    if exclusion_reason != [phase[0], phase[2]]:
        raise ValueError("prepared exclusion-reason shape differs from phase shape")


def _nonnegative_integer_list(value: object, length: int) -> bool:
    """Return whether a JSON list has one fixed count of nonnegative integers.

    Parameters
    ----------
    value : object
        JSON value expected to be a finite list of count dimensions. It has no
        physical units and is never converted into an array.
    length : int
        Required number of dimensions.

    Returns
    -------
    bool
        ``True`` only for exactly ``length`` non-Boolean nonnegative integers.
    """
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value)
    )


def _validate_prepared_completion(record: object, expected_fingerprint: str) -> None:
    """Require the exact one-field completion certificate from the writer.

    Parameters
    ----------
    record : object
        Parsed bounded completion JSON. It contains no numerical array data.
    expected_fingerprint : str
        Representation directory identity that the sole certificate field must
        exactly equal.

    Returns
    -------
    None
        The record is read-only.

    Raises
    ------
    ValueError
        If completion is not exactly one matching fingerprint field.
    """
    if record != {"representation_fingerprint": expected_fingerprint}:
        raise ValueError("prepared completion certificate is invalid")


def _lstat_or_none(path: Path) -> os.stat_result | None:
    """Return link-preserving status metadata for one path without opening it.

    Parameters
    ----------
    path : pathlib.Path
        One direct filesystem entry. Its data payload, numerical arrays, axes,
        shapes, and units remain unread.

    Returns
    -------
    os.stat_result or None
        Nofollow status metadata for an existing entry, or ``None`` when the
        exact path is absent.

    Raises
    ------
    OSError
        If metadata cannot be inspected.
    """
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _require_real_directory(path: Path, status: os.stat_result, label: str) -> None:
    """Require status metadata to describe a real nonsymlink directory.

    Parameters
    ----------
    path : pathlib.Path
        Display path for an error only; it is never reopened by this helper.
    status : os.stat_result
        Nofollow ``lstat`` or descriptor ``fstat`` metadata with no numerical
        content, axes, shapes, or units.
    label : str
        Human-readable entry role for a validation error.

    Returns
    -------
    None
        The filesystem is not mutated.

    Raises
    ------
    ValueError
        If the entry is linked or not a directory.
    """
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{label} must be a real non-symbolic-link directory: {path}")


def _require_regular_nonsymlink_file(
    path: Path,
    status: os.stat_result,
    label: str,
) -> None:
    """Require status metadata to describe a regular nonsymlink identity file.

    Parameters
    ----------
    path : pathlib.Path
        Display path for an error only; array payload bytes remain unopened.
    status : os.stat_result
        Nofollow ``lstat`` or descriptor ``fstat`` metadata.
    label : str
        Human-readable entry role for a validation error.

    Returns
    -------
    None
        The filesystem is not mutated.

    Raises
    ------
    ValueError
        If the entry is linked or not a regular file.
    """
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular non-symbolic-link file: {path}")


def _stat_member_without_following(
    parent_descriptor: int,
    member_name: str,
    label: str,
) -> os.stat_result:
    """Return nofollow direct-child status relative to a stable directory fd.

    Parameters
    ----------
    parent_descriptor : int
        Open stable directory descriptor containing ``member_name``.
    member_name : str
        Direct child basename with no array payload content.
    label : str
        Human-readable entry role for an I/O error.

    Returns
    -------
    os.stat_result
        Nofollow child status metadata, never child bytes or numerical arrays.

    Raises
    ------
    ValueError
        If the child cannot be statted relative to the anchored directory.
    """
    try:
        return os.stat(member_name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {member_name}") from error


def _anchored_member_statuses(
    parent_descriptor: int,
    member_names: Sequence[str],
    label: str,
) -> dict[str, os.stat_result]:
    """Capture nofollow identities for direct children of an anchored directory.

    Parameters
    ----------
    parent_descriptor : int
        Open stable parent directory descriptor whose direct children are
        classified without opening their payloads.
    member_names : collections.abc.Sequence[str]
        Exact direct child basenames from a stable parent inventory.
    label : str
        Human-readable role supplied to filesystem validation errors.

    Returns
    -------
    dict[str, os.stat_result]
        Nofollow device, inode, type, and size metadata keyed by each direct
        child basename. No child bytes, arrays, axes, or units are read.
    """
    return {
        member_name: _stat_member_without_following(
            parent_descriptor,
            member_name,
            label,
        )
        for member_name in member_names
    }


def _require_unchanged_anchored_members(
    parent_descriptor: int,
    initial_statuses: Mapping[str, os.stat_result],
    label: str,
) -> None:
    """Require every direct child to retain its anchored nofollow identity.

    Parameters
    ----------
    parent_descriptor : int
        Still-open stable parent directory descriptor used for final direct
        child status checks after nested child descriptors have closed.
    initial_statuses : collections.abc.Mapping[str, os.stat_result]
        Initial nofollow status records keyed by direct child basename. The
        records contain no payload values or physical units.
    label : str
        Human-readable role supplied to replacement validation errors.

    Returns
    -------
    None
        The hierarchy stays read-only. Every current child must match the
        original device, inode, type, and size metadata.

    Raises
    ------
    ValueError
        If a child is missing, linked, substituted, changes type, or changes
        size before the parent directory is accepted.
    """
    for member_name, initial_status in initial_statuses.items():
        final_status = _stat_member_without_following(
            parent_descriptor,
            member_name,
            label,
        )
        _require_same_entry(initial_status, final_status, label)


def _require_same_entry(
    expected: os.stat_result,
    actual: os.stat_result,
    label: str,
) -> None:
    """Require two status records to identify the same filesystem entry.

    Parameters
    ----------
    expected, actual : os.stat_result
        Nofollow or descriptor status records containing identity/type/size
        metadata only. No numerical values, axes, shapes, or units are read.
    label : str
        Human-readable role for a validation error.

    Returns
    -------
    None
        No entry is changed.

    Raises
    ------
    ValueError
        If device, inode, file type, or size changed between inspections.
    """
    if (
        expected.st_dev != actual.st_dev
        or expected.st_ino != actual.st_ino
        or stat.S_IFMT(expected.st_mode) != stat.S_IFMT(actual.st_mode)
        or expected.st_size != actual.st_size
    ):
        raise ValueError(f"{label} changed identity during preflight")


def _require_stable_directory_path(path: Path, descriptor: int, label: str) -> None:
    """Require a directory pathname to still identify its anchored descriptor.

    Parameters
    ----------
    path : pathlib.Path
        Directory path rechecked after descriptor-based inspection.
    descriptor : int
        Open anchored directory descriptor whose identity must match ``path``.
    label : str
        Human-readable role for a validation error.

    Returns
    -------
    None
        The path and descriptor remain read-only.

    Raises
    ------
    ValueError
        If the path was removed, linked, substituted, or changed type.
    """
    descriptor_status = os.fstat(descriptor)
    current_status = _lstat_or_none(path)
    if current_status is None:
        raise ValueError(f"{label} disappeared during preflight")
    _require_real_directory(path, current_status, label)
    _require_same_entry(current_status, descriptor_status, label)

    # Recheck after comparing the original pathname status to the fd. A
    # same-name directory replacement can occur in that narrow gap.
    final_status = _lstat_or_none(path)
    if final_status is None:
        raise ValueError(f"{label} disappeared during preflight")
    _require_real_directory(path, final_status, label)
    _require_same_entry(final_status, descriptor_status, label)


def _require_same_directory_inventory(
    before: os.stat_result,
    after: os.stat_result,
    label: str,
) -> None:
    """Require directory identity and mutation metadata to remain unchanged.

    Parameters
    ----------
    before, after : os.stat_result
        ``fstat`` records from one open directory descriptor before and after
        listing direct children. They contain no child payload or numerical
        array contents.
    label : str
        Human-readable directory role for a validation error.

    Returns
    -------
    None
        The directory is not modified.

    Raises
    ------
    ValueError
        If identity, type, size, link count, modification, or change time
        differs while its inventory is read.
    """
    _require_same_entry(before, after, label)
    if (
        before.st_nlink != after.st_nlink
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError(f"{label} inventory changed during preflight")


def _saved_cache_directory(identity: Mapping[str, object]) -> Path:
    """Return the immutable cache target recorded by an earlier new run.

    Parameters
    ----------
    identity : Mapping[str, object]
        Durable launcher identity JSON containing the resolved cache path.

    Returns
    -------
    pathlib.Path
        Exact absolute cache path persisted by the originating new run.
    """
    value = identity.get("cache_directory")
    if not isinstance(value, str) or not value:
        raise ValueError("saved launcher identity lacks cache_directory")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve(strict=False)) != value:
        raise ValueError("saved launcher cache_directory is not a resolved absolute path")
    return path


def _identity_mapping(
    config: LFPSummaryConfig,
    command: LauncherCommand,
    repository: RepositoryState,
    source_fingerprints: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact JSON-safe run identity without numerical arrays."""
    population = config.unit_population
    assert population is not None
    unit_json = json.dumps(list(population.stable_unit_ids), separators=(",", ":"))
    source_json = json.dumps(dict(source_fingerprints), sort_keys=True, separators=(",", ":"), allow_nan=False)
    config_json = canonical_config_json(config)
    identity = {
        "session_path": str(config.session_path.resolve(strict=False)),
        "cache_directory": str(config.output_directory.resolve(strict=False)),
        "repository_path": str(repository.repository_root.resolve(strict=False)),
        "session_id": config.session_id,
        "probe_label": population.probe_label,
        "stable_unit_ids": list(population.stable_unit_ids),
        "stable_unit_ids_sha256": sha256(unit_json.encode("ascii")).hexdigest(),
        "shuffle_count": int(config.ppc.shuffle_count),
        "final_run": bool(command.final_run),
        "requested_worker_count": int(config.ppc_execution.worker_count),
        "configuration_sha256": sha256(config_json.encode("ascii")).hexdigest(),
        "component_fingerprint": component_fingerprint("spike_phase", config),
        "source_fingerprints": dict(source_fingerprints),
        "source_fingerprint_sha256": sha256(source_json.encode("ascii")).hexdigest(),
        "git_commit": repository.git_commit,
        "tracked_clean": repository.tracked_clean,
    }
    if command.session_metadata is not None:
        identity["session_metadata"] = str(command.session_metadata.resolve(strict=False))
    return identity


def _paths_mapping(
    config: LFPSummaryConfig,
    analysis_root: Path,
    run_directory: Path,
) -> dict[str, str]:
    """Return resolved launcher/cache/work/report paths without creating work."""
    return {
        "analysis_root": str(analysis_root),
        "run_directory": str(run_directory),
        "output_directory": str(config.output_directory.resolve(strict=False)),
        "work_root": str((config.output_directory.parent / "lfp_summary_work").resolve(strict=False)),
        "report_parent": str((run_directory / "report").resolve(strict=False)),
    }


def _preflight_mapping(
    config: LFPSummaryConfig,
    trial_count: int,
    run_directory: Path,
    command: LauncherCommand,
) -> dict[str, object]:
    """Return conservative metadata-only preflight counts and byte bounds."""
    population = config.unit_population
    assert population is not None
    time_count = int(
        round(
            (config.analysis_windows.whole_stop_s - config.analysis_windows.whole_start_s)
            * config.phase.output_rate_hz
        )
    )
    phase_cell_count = (
        len(config.sites)
        * len(config.phase.frequency_hz)
        * trial_count
        * time_count
    )
    unit_blocks = math.ceil(
        len(population.stable_unit_ids) / config.ppc_execution.unit_block_size
    )
    maximum_active_workers = min(config.ppc_execution.worker_count, unit_blocks)
    paths = _paths_mapping(config, run_directory.parent, run_directory)
    return {
        "schema_version": _PREFLIGHT_SCHEMA,
        "session_id": config.session_id,
        "probe_label": population.probe_label,
        "run_kind": "dry_run" if command.dry_run else ("final" if command.final_run else "preview"),
        "shuffle_count": int(config.ppc.shuffle_count),
        "unit_count": len(population.stable_unit_ids),
        "trial_count": int(trial_count),
        "site_count": len(config.sites),
        "frequency_count": len(config.phase.frequency_hz),
        "time_count": time_count,
        "requested_worker_count": int(config.ppc_execution.worker_count),
        "conservative_maximum_active_worker_count": maximum_active_workers,
        "maximum_worker_allocation_bytes": int(config.ppc_execution.maximum_worker_allocation_bytes),
        "maximum_aggregate_allocation_bytes": int(config.ppc_execution.maximum_aggregate_allocation_bytes),
        "estimated_phase_and_validity_bytes": phase_cell_count * 9,
        "exact_plan_available": False,
        "exact_plan_unavailable_reason": "phase-derived site validity and schedules require scientific execution",
        "cache_directory": str(config.output_directory.resolve(strict=False)),
        "ppc_planning_seconds": None,
        "planned_ppc_allocation_bytes": None,
        "maximum_child_process_count": None,
        "prepared_phase_cache_state": None,
        "paths": paths,
    }


def _execute_locked_run(
    config: LFPSummaryConfig,
    run_directory: Path,
    state: dict[str, object],
    dependencies: LauncherDependencies,
    *,
    component_already_validated: bool = False,
    report_recovery: bool = False,
) -> LauncherRunResult:
    """Complete unfinished component/report/cleanup stages under one run lock.

    ``component_already_validated`` is true only after explicit report recovery
    has validated the immutable component. ``report_recovery`` adds the audit
    validation gate immediately before cleanup; neither flag changes numerics.
    """
    completed = state["completed_stages"]
    assert isinstance(completed, list)
    if "launcher_artifacts_validated" in completed:
        dependencies.validate_component(config)
        _validate_existing_report(state)
        _validate_launcher_artifacts(run_directory, state)
        return _cleanup_and_complete(config, run_directory, state, dependencies)

    component_result: ComponentRunResult | None = None
    component_started = _clock(dependencies.monotonic_seconds)
    if "component_complete" not in completed:
        if dependencies.component_is_compatible(config):
            if "cleanup_prepared" not in completed:
                state["cleanup_request"] = []
                _complete_stage(run_directory, state, "cleanup_prepared", dependencies)
            dependencies.validate_component(config)
            measurements = state["measurements"]
            assert isinstance(measurements, dict)
            measurements["component_reused"] = True
        else:
            state["active_status"] = "preparing_phase"
            _persist_state(run_directory, state, dependencies)

            def observe_cleanup(targets: tuple[PPCWorkCleanupTarget, ...]) -> None:
                """Persist exact cleanup identities before manifest-last commit."""
                state["cleanup_request"] = [
                    {
                        "run_directory": str(target.run_directory.resolve(strict=False)),
                        "run_fingerprint": target.run_fingerprint,
                    }
                    for target in targets
                ]
                if "cleanup_prepared" not in completed:
                    _complete_stage(run_directory, state, "cleanup_prepared", dependencies)

            progress = _progress_log_adapter(run_directory, state, dependencies)
            with dependencies.process_tree_sampler() as memory_sample:
                component_result = dependencies.compute_component(
                    config,
                    progress,
                    observe_cleanup,
                )
            if component_result.state != "complete":
                raise RuntimeError(component_result.error or "Spike-phase component failed")
            if "cleanup_prepared" not in completed:
                observe_cleanup(tuple(component_result.deferred_cleanup_targets))
            _merge_execution_measurements(
                state,
                component_result,
                memory_sample,
                _clock(dependencies.monotonic_seconds) - component_started,
            )
            dependencies.validate_component(config)
        _complete_stage(run_directory, state, "component_complete", dependencies)
    elif not component_already_validated:
        dependencies.validate_component(config)

    if "report_complete" not in completed:
        state["active_status"] = "reporting"
        _persist_state(run_directory, state, dependencies)
        report_started = _clock(dependencies.monotonic_seconds)
        paths = state["paths"]
        assert isinstance(paths, dict)
        report_parent = Path(str(paths["report_parent"]))
        report_parent.mkdir(parents=True, exist_ok=True)
        measurements = _report_measurements(config, state)
        report_result = dependencies.publish_report(
            config=config,
            run_parent=report_parent,
            run_kind=str(state["run_kind"]),
            component_wall_time_s=float(
                _measurement(state, "component_total_seconds", 0.0)
            ),
            component_peak_memory_bytes=int(
                _measurement(state, "process_rss_bytes", 0)
            ),
            report_measurements=measurements,
            deferred_cleanup=(
                component_result.deferred_cleanup
                if component_result is not None
                else None
            ),
        )
        state["report_directory"] = str(Path(report_result.run_directory).resolve(strict=False))
        state_measurements = state["measurements"]
        assert isinstance(state_measurements, dict)
        state_measurements["report_seconds"] = max(
            0.0,
            _clock(dependencies.monotonic_seconds) - report_started,
        )
        _complete_stage(run_directory, state, "report_complete", dependencies)
    else:
        _validate_existing_report(state)

    _write_summary(run_directory, state)
    _validate_launcher_artifacts(run_directory, state)
    _complete_stage(run_directory, state, "launcher_artifacts_validated", dependencies)
    if report_recovery:
        _mark_report_recovery_validated(run_directory, state, dependencies)
        _validate_launcher_artifacts(run_directory, state)
    return _cleanup_and_complete(config, run_directory, state, dependencies)


def _cleanup_and_complete(
    config: LFPSummaryConfig,
    run_directory: Path,
    state: dict[str, object],
    dependencies: LauncherDependencies,
) -> LauncherRunResult:
    """Validate and remove only saved exact targets, then mark completion."""
    completed = state["completed_stages"]
    assert isinstance(completed, list)
    state["active_status"] = "cleaning"
    _persist_state(run_directory, state, dependencies)
    for target in _cleanup_targets_from_state(config, state):
        if target.run_directory.exists():
            dependencies.cleanup_target(target)
        if target.run_directory.exists():
            raise OSError(f"cleanup target still exists: {target.run_directory}")
    if "cleanup_complete" not in completed:
        _complete_stage(run_directory, state, "cleanup_complete", dependencies)
    if "complete" not in completed:
        _complete_stage(run_directory, state, "complete", dependencies)
    state["active_status"] = "complete"
    state["error"] = None
    _persist_state(run_directory, state, dependencies)
    _append_log(run_directory, _log_event(state, "complete", "launcher run complete"))
    _write_summary(run_directory, state)
    return LauncherRunResult(
        0,
        "complete",
        run_directory,
        str(state["resume_command"]),
    )


def _merge_execution_measurements(
    state: dict[str, object],
    result: ComponentRunResult,
    memory_sample: object,
    component_seconds: float,
) -> None:
    """Copy validated scalar runtime/memory values into launcher state."""
    measurements = state["measurements"]
    assert isinstance(measurements, dict)
    measurements.update(dict(result.execution_metadata))
    measurements["component_total_seconds"] = max(0.0, float(component_seconds))
    for source_name, target_name in (
        ("process_rss_bytes", "process_rss_bytes"),
        ("process_tree_rss_bytes", "process_tree_rss_bytes"),
        ("process_tree_pss_bytes", "process_tree_pss_bytes"),
        ("provenance", "memory_provenance"),
        ("sampling_interval_seconds", "memory_sampling_interval_seconds"),
        ("maximum_child_process_count", "maximum_child_process_count"),
    ):
        measurements[target_name] = getattr(memory_sample, source_name, None)
    if any(
        measurements.get(name) is None
        for name in (
            "process_rss_bytes",
            "process_tree_rss_bytes",
            "process_tree_pss_bytes",
            "memory_provenance",
        )
    ):
        warnings = state["warnings"]
        assert isinstance(warnings, list)
        warning = "one or more Linux process-tree memory measurements are unavailable"
        if warning not in warnings:
            warnings.append(warning)


def _report_measurements(
    config: LFPSummaryConfig,
    state: Mapping[str, object],
) -> object:
    """Build WP12 scalar report measurements from launcher state."""
    from src.neural_analysis.lfp_spike_phase_validation import (
        SpikePhaseFilterBenchmark,
        SpikePhaseReportMeasurements,
    )

    values = state["measurements"]
    assert isinstance(values, Mapping)
    final_component = _file_size(config.output_directory / "spike_phase.npz")
    final_cache = _directory_size(config.output_directory)
    intermediate = sum(
        _directory_size(target.run_directory)
        for target in _cleanup_targets_from_state(config, state)
    )
    component_seconds = _optional_float(values.get("component_total_seconds"))
    benchmark = None
    if component_seconds is not None:
        benchmark = SpikePhaseFilterBenchmark(
            source_label="launcher active filter run",
            wall_time_s=component_seconds,
            final_cache_size_bytes=final_cache,
            intermediate_cache_size_bytes=intermediate,
        )
    return SpikePhaseReportMeasurements(
        requested_worker_count=config.ppc_execution.worker_count,
        phase_preparation_seconds=_optional_float(values.get("phase_preparation_seconds")),
        ppc_planning_seconds=_optional_float(values.get("ppc_planning_seconds")),
        grouped_execution_seconds=_optional_float(values.get("grouped_execution_seconds")),
        component_total_seconds=component_seconds,
        planned_worker_count=_optional_int(values.get("planner_active_worker_count")),
        active_worker_count=_optional_int(values.get("scientific_active_worker_count")),
        prepared_phase_cache_state=_optional_string(values.get("prepared_phase_cache_state")),
        peak_process_rss_bytes=_optional_int(values.get("process_rss_bytes")),
        peak_process_tree_rss_bytes=_optional_int(values.get("process_tree_rss_bytes")),
        peak_process_tree_pss_bytes=_optional_int(values.get("process_tree_pss_bytes")),
        memory_provenance=_optional_string(values.get("memory_provenance")),
        final_component_size_bytes=final_component,
        final_cache_size_bytes=final_cache,
        intermediate_work_size_bytes=intermediate,
        warnings=tuple(str(value) for value in state.get("warnings", [])),
        exclusions=(),
        benchmark=benchmark,
    )


def _cleanup_targets_from_state(
    config: LFPSummaryConfig,
    state: Mapping[str, object],
) -> tuple[PPCWorkCleanupTarget, ...]:
    """Rebuild and path-validate exact cleanup targets from scalar state."""
    raw_targets = state.get("cleanup_request")
    if not isinstance(raw_targets, list):
        raise ValueError("cleanup_request must be a JSON list")
    expected_parent = (
        config.output_directory.parent / "lfp_summary_work" / "ppc"
    ).resolve(strict=False)
    targets: list[PPCWorkCleanupTarget] = []
    for raw in raw_targets:
        if not isinstance(raw, dict) or set(raw) != {"run_directory", "run_fingerprint"}:
            raise ValueError("cleanup target state is malformed")
        path = Path(str(raw["run_directory"]))
        if path.is_symlink():
            raise ValueError("cleanup target must not be a symbolic link")
        resolved = path.resolve(strict=False)
        target = PPCWorkCleanupTarget(resolved, str(raw["run_fingerprint"]))
        if resolved.parent != expected_parent:
            raise ValueError("cleanup target is outside the configured PPC work parent")
        if resolved.exists():
            metadata_path = resolved / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="ascii"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("cleanup target metadata is unavailable") from error
            if not isinstance(metadata, dict) or metadata.get("run_fingerprint") != target.run_fingerprint:
                raise ValueError("cleanup target metadata fingerprint mismatch")
        targets.append(target)
    return tuple(targets)


def _progress_log_adapter(
    run_directory: Path,
    state: Mapping[str, object],
    dependencies: LauncherDependencies,
) -> Callable[[ProgressEvent], None]:
    """Return a progress callback that validates, logs, and prints scalars."""
    prior: dict[tuple[str, str, str | None], int] = {}

    def update(event: ProgressEvent) -> None:
        """Write one monotonic framework-independent progress event."""
        key = (event.component, event.stage, event.job_id)
        previous = prior.get(key, -1)
        if event.completed_count < previous:
            raise ValueError("progress completed_count must not decrease")
        prior[key] = event.completed_count
        record = {
            "event": "progress",
            "utc": dependencies.now_utc(),
            "run_id": state["run_id"],
            "component": event.component,
            "stage": event.stage,
            "completed_count": event.completed_count,
            "total_count": event.total_count,
            "message": event.message,
            "elapsed_seconds": event.elapsed_seconds,
            "eta_seconds": event.eta_seconds,
            "job_id": event.job_id,
        }
        _append_log(run_directory, record)
        total = "?" if event.total_count is None else str(event.total_count)
        timing = ""
        if event.elapsed_seconds is not None:
            timing += f" elapsed={event.elapsed_seconds:.3f}s"
        if event.eta_seconds is not None:
            timing += f" eta={event.eta_seconds:.3f}s"
        _terminal(
            dependencies,
            f"{record['utc']} {state['run_id']} {event.component}/{event.stage} "
            f"{event.completed_count}/{total}{timing} {event.message}",
        )

    return update


def _complete_stage(
    run_directory: Path,
    state: dict[str, object],
    stage: str,
    dependencies: LauncherDependencies,
) -> None:
    """Append one exact next stage and atomically persist launcher state."""
    completed = state["completed_stages"]
    assert isinstance(completed, list)
    expected = _STAGES[len(completed)]
    if stage != expected:
        raise ValueError(f"expected launcher stage {expected!r}, got {stage!r}")
    completed.append(stage)
    state["active_status"] = stage
    _persist_state(run_directory, state, dependencies)
    _append_log(run_directory, _log_event(state, stage, f"completed {stage}"))


def _persist_state(
    run_directory: Path,
    state: dict[str, object],
    dependencies: LauncherDependencies,
) -> None:
    """Update UTC and atomically replace state without changing stage order."""
    state["updated_utc"] = dependencies.now_utc()
    _validate_state(state)
    _write_json_atomic(run_directory / "launcher_state.json", state)


def _load_report_recovery_audit(
    path: Path,
    state: Mapping[str, object],
) -> dict[str, object]:
    """Load and validate scalar report-recovery provenance.

    Parameters
    ----------
    path : pathlib.Path
        Audit JSON path inside the launcher run directory.
    state : Mapping[str, object]
        Validated launcher state used for run and computation identities.

    Returns
    -------
    dict[str, object]
        JSON-safe scalar audit mapping. No numerical arrays are permitted.
    """
    try:
        audit = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("report_recovery.json is unavailable") from error
    required = {
        "schema_version",
        "run_id",
        "status",
        "computation_git_commit",
        "report_git_commit",
        "original_error",
        "recovery_utc",
        "updated_utc",
        "component_reused",
        "error",
    }
    identity = state.get("identity")
    measurements = state.get("measurements")
    if (
        not isinstance(audit, dict)
        or set(audit) != required
        or audit.get("schema_version") != _REPORT_RECOVERY_SCHEMA
        or audit.get("run_id") != state.get("run_id")
        or not isinstance(identity, Mapping)
        or audit.get("computation_git_commit") != identity.get("git_commit")
        or audit.get("component_reused") is not True
        or audit.get("status")
        not in {"initialized", "validated", "failed", "complete"}
        or not isinstance(audit.get("original_error"), str)
        or not audit.get("original_error")
        or re.fullmatch(r"[0-9a-f]{40}", str(audit.get("report_git_commit"))) is None
    ):
        raise ValueError("report_recovery.json is incompatible with launcher state")
    if (
        isinstance(measurements, Mapping)
        and measurements.get("report_git_commit") is not None
    ):
        if audit["report_git_commit"] != measurements["report_git_commit"]:
            raise ValueError("report recovery commit disagrees with launcher state")
    error = audit.get("error")
    if error is not None and (not isinstance(error, str) or not error):
        raise ValueError("report recovery error must be null or nonempty text")
    return audit


def _mark_report_recovery_validated(
    run_directory: Path,
    state: dict[str, object],
    dependencies: LauncherDependencies,
) -> None:
    """Record validated report publication before exact work cleanup."""
    path = run_directory / "report_recovery.json"
    audit = _load_report_recovery_audit(path, state)
    audit["status"] = "validated"
    audit["error"] = None
    audit["updated_utc"] = dependencies.now_utc()
    measurements = state["measurements"]
    assert isinstance(measurements, dict)
    measurements["report_recovery_status"] = "validated"
    _write_json_atomic(path, audit)
    _persist_state(run_directory, state, dependencies)
    _append_log(
        run_directory,
        _log_event(
            state,
            "report_recovery_validated",
            "report artifacts validated before cleanup",
        ),
    )
    _write_summary(run_directory, state)


def _complete_report_recovery(
    run_directory: Path,
    state: dict[str, object],
    dependencies: LauncherDependencies,
) -> None:
    """Finalize report-recovery provenance after launcher completion."""
    path = run_directory / "report_recovery.json"
    audit = _load_report_recovery_audit(path, state)
    audit["status"] = "complete"
    audit["error"] = None
    audit["updated_utc"] = dependencies.now_utc()
    measurements = state["measurements"]
    assert isinstance(measurements, dict)
    measurements["report_recovery_status"] = "complete"
    _write_json_atomic(path, audit)
    _persist_state(run_directory, state, dependencies)
    _append_log(
        run_directory,
        _log_event(state, "report_recovery_complete", "report-only recovery complete"),
    )
    _write_summary(run_directory, state)


def _record_report_recovery_failure(
    command: LauncherCommand,
    run_directory: Path,
    state: dict[str, object],
    message: str,
    dependencies: LauncherDependencies,
) -> None:
    """Record a failed started recovery while preserving its original error."""
    path = run_directory / "report_recovery.json"
    if command.mode != "recover-report" or not path.is_file():
        return
    audit = _load_report_recovery_audit(path, state)
    audit["status"] = "failed"
    audit["error"] = message or "report recovery failed"
    audit["updated_utc"] = dependencies.now_utc()
    measurements = state["measurements"]
    assert isinstance(measurements, dict)
    measurements["report_recovery_status"] = "failed"
    _write_json_atomic(path, audit)


def _validate_report_rerender_audit(
    audit: Mapping[str, object],
    state: Mapping[str, object],
) -> None:
    """Validate one scalar completed-report rerender audit mapping.

    Parameters
    ----------
    audit : Mapping[str, object]
        JSON-safe rerender audit with Git commits and immutable report paths.
    state : Mapping[str, object]
        Validated launcher state supplying run and computation identity.

    Returns
    -------
    None
        The mappings are validated without mutation.

    Raises
    ------
    ValueError
        If fields, identities, paths, status, or error semantics disagree.
    """
    required = {
        "schema_version",
        "run_id",
        "status",
        "computation_git_commit",
        "previous_report_git_commit",
        "rerender_git_commit",
        "previous_report_directory",
        "new_report_directory",
        "rerender_utc",
        "updated_utc",
        "component_reused",
        "error",
    }
    identity = state.get("identity")
    status = audit.get("status")
    if (
        set(audit) != required
        or audit.get("schema_version") != _REPORT_RERENDER_SCHEMA
        or audit.get("run_id") != state.get("run_id")
        or not isinstance(identity, Mapping)
        or audit.get("computation_git_commit") != identity.get("git_commit")
        or audit.get("component_reused") is not True
        or status not in {"initialized", "validated", "failed", "complete"}
    ):
        raise ValueError("report_rerender.json is incompatible with launcher state")
    for name in (
        "computation_git_commit",
        "previous_report_git_commit",
        "rerender_git_commit",
    ):
        if re.fullmatch(r"[0-9a-f]{40}", str(audit.get(name))) is None:
            raise ValueError("report rerender Git commit is invalid")
    if not isinstance(audit.get("previous_report_directory"), str):
        raise ValueError("report rerender previous report path is invalid")
    new_report = audit.get("new_report_directory")
    if status in {"validated", "complete"}:
        if not isinstance(new_report, str) or state.get("report_directory") != new_report:
            raise ValueError("report rerender new report path disagrees with launcher state")
    elif new_report is not None and not isinstance(new_report, str):
        raise ValueError("report rerender new report path is invalid")
    error = audit.get("error")
    if status == "failed":
        if not isinstance(error, str) or not error:
            raise ValueError("failed report rerender requires a nonempty error")
    elif error is not None:
        raise ValueError("successful report rerender error must be null")


def _load_report_rerender_audit(
    path: Path,
    state: Mapping[str, object],
) -> dict[str, object]:
    """Load and validate one completed-report rerender audit.

    Parameters
    ----------
    path : pathlib.Path
        Audit JSON path inside the launcher run directory.
    state : Mapping[str, object]
        Validated launcher state supplying scalar identity.

    Returns
    -------
    dict[str, object]
        Validated JSON-safe audit mapping.
    """
    try:
        audit = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("report_rerender.json is unavailable") from error
    if not isinstance(audit, dict):
        raise ValueError("report_rerender.json must contain one mapping")
    _validate_report_rerender_audit(audit, state)
    measurements = state.get("measurements")
    if isinstance(measurements, Mapping):
        rerender_commit = measurements.get("report_rerender_git_commit")
        if (
            rerender_commit is not None
            and audit.get("rerender_git_commit") != rerender_commit
        ):
            raise ValueError("report rerender commit disagrees with launcher state")
    return audit


def _record_report_rerender_failure(
    run_directory: Path,
    state: Mapping[str, object],
    message: str,
    dependencies: LauncherDependencies,
) -> None:
    """Record a started rerender failure without changing completed state.

    Parameters
    ----------
    run_directory : pathlib.Path
        Completed launcher directory containing an optional initialized audit.
    state : Mapping[str, object]
        Original completed launcher state, retained byte-for-byte on disk.
    message : str
        Nonempty failure description.
    dependencies : LauncherDependencies
        Clock seam used for the audit timestamp.

    Returns
    -------
    None
        Only an already-created rerender audit may be atomically updated.
    """
    path = run_directory / "report_rerender.json"
    if not path.is_file():
        return
    try:
        audit = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if (
        not isinstance(audit, dict)
        or audit.get("status") not in {"initialized", "validated"}
    ):
        return
    audit["status"] = "failed"
    audit["error"] = message or "report rerender failed"
    audit["updated_utc"] = dependencies.now_utc()
    _validate_report_rerender_audit(audit, state)
    _write_json_atomic(path, audit)


def _record_terminal_failure(
    run_directory: Path,
    state: dict[str, object],
    status: str,
    message: str,
    dependencies: LauncherDependencies,
) -> None:
    """Persist one incomplete terminal status and exact resume information."""
    state["active_status"] = status
    state["error"] = message or status
    _persist_state(run_directory, state, dependencies)
    _append_log(run_directory, _log_event(state, status, state["error"]))
    _write_summary(run_directory, state)
    _terminal(dependencies, f"Launcher {status}: {state['error']}")
    resume = _resume_from_state(state)
    if resume is not None:
        _terminal(dependencies, f"Resume command: {resume}")


def _load_state(path: Path) -> dict[str, object]:
    """Load and validate one ASCII launcher-state mapping."""
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("unable to load valid launcher_state.json") from error
    if not isinstance(value, dict):
        raise ValueError("launcher state must be a JSON mapping")
    _validate_state(value)
    return value


def _validate_state(state: Mapping[str, object]) -> None:
    """Validate fixed state keys and ordered durable-stage prefix."""
    required = {
        "schema_version", "run_id", "created_utc", "updated_utc", "run_kind",
        "dry_run", "active_status", "completed_stages", "resume_command",
        "identity", "paths", "measurements", "cleanup_request",
        "report_directory", "warnings", "error",
    }
    if set(state) != required or state.get("schema_version") != _STATE_SCHEMA:
        raise ValueError("launcher state schema/fields are incompatible")
    completed = state.get("completed_stages")
    if not isinstance(completed, list) or completed != list(_STAGES[: len(completed)]):
        raise ValueError("launcher completed stages must be an exact ordered prefix")
    if not isinstance(state.get("identity"), dict) or not isinstance(state.get("paths"), dict):
        raise ValueError("launcher state identity/paths must be mappings")
    if not isinstance(state.get("measurements"), dict) or not isinstance(state.get("cleanup_request"), list):
        raise ValueError("launcher measurements/cleanup request have invalid types")
    if not isinstance(state.get("warnings"), list):
        raise ValueError("launcher warnings must be a list")


def _validate_configuration_snapshot(
    run_directory: Path,
    config: LFPSummaryConfig,
) -> None:
    """Require saved canonical configuration equality on resume."""
    try:
        saved = json.loads((run_directory / "configuration.json").read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("saved launcher configuration is unavailable") from error
    if saved != json.loads(canonical_config_json(config)):
        raise ValueError("saved launcher configuration does not match resume config")


def _validate_existing_report(state: Mapping[str, object]) -> None:
    """Require the saved report directory and machine report to remain readable."""
    raw = state.get("report_directory")
    if not isinstance(raw, str) or not raw:
        raise ValueError("launcher state has no published report directory")
    directory = Path(raw)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("published report directory is unavailable")
    paths = state.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("launcher paths are unavailable")
    report_parent = Path(str(paths.get("report_parent"))).resolve(strict=False)
    if directory.resolve(strict=True).parent != report_parent:
        raise ValueError("published report directory is outside the launcher report parent")
    try:
        report = json.loads((directory / "report.json").read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("published report.json is unavailable") from error
    if not isinstance(report, dict) or report.get("schema_version") != "spike_phase_report.v1":
        raise ValueError("published launcher report schema is incompatible")


def _validate_launcher_artifacts(
    run_directory: Path,
    state: Mapping[str, object],
) -> None:
    """Reload fixed JSON/log/summary artifacts before cleanup authorization."""
    _validate_launcher_identity_artifacts(run_directory, state)
    for name in (
        "launcher_state.json",
        "configuration.json",
    ):
        value = json.loads((run_directory / name).read_text(encoding="ascii"))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain one JSON mapping")
    lines = (run_directory / "run.log").read_text(encoding="ascii").splitlines()
    if not lines:
        raise ValueError("run.log must contain JSON Lines events")
    for line in lines:
        if not isinstance(json.loads(line), dict):
            raise ValueError("every run.log line must be a JSON mapping")
    summary = (run_directory / "run_summary.md").read_text(encoding="ascii")
    if not summary.strip():
        raise ValueError("run_summary.md must be nonempty")
    saved = _load_state(run_directory / "launcher_state.json")
    if saved != dict(state):
        raise ValueError("in-memory and persisted launcher state disagree")
    measurements = state.get("measurements")
    assert isinstance(measurements, Mapping)
    recovery_path = run_directory / "report_recovery.json"
    if "report_git_commit" in measurements or recovery_path.exists():
        _load_report_recovery_audit(recovery_path, state)
    rerender_path = run_directory / "report_rerender.json"
    if "report_rerender_git_commit" in measurements or rerender_path.exists():
        _load_report_rerender_audit(rerender_path, state)
    _validate_existing_report(state)


def _validate_launcher_identity_artifacts(
    run_directory: Path,
    state: Mapping[str, object],
) -> None:
    """Validate saved source and preflight evidence against durable identity."""
    identity = state.get("identity")
    paths = state.get("paths")
    if not isinstance(identity, Mapping) or not isinstance(paths, Mapping):
        raise ValueError("launcher identity/path state is unavailable")
    try:
        source = json.loads(
            (run_directory / "source_identity.json").read_text(encoding="ascii")
        )
        preflight = json.loads(
            (run_directory / "preflight.json").read_text(encoding="ascii")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("launcher identity evidence is unavailable") from error
    expected_source = {
        "repository": {
            "root": identity.get("repository_path"),
            "git_commit": identity.get("git_commit"),
            "tracked_clean": identity.get("tracked_clean"),
        },
        "source_fingerprints": identity.get("source_fingerprints"),
        "source_fingerprint_sha256": identity.get("source_fingerprint_sha256"),
    }
    if source != expected_source:
        raise ValueError("source_identity.json does not match durable identity")
    cache_directory = identity.get("cache_directory")
    if (
        not isinstance(cache_directory, str)
        or paths.get("output_directory") != cache_directory
        or not isinstance(preflight, dict)
        or preflight.get("schema_version") != _PREFLIGHT_SCHEMA
        or preflight.get("session_id") != identity.get("session_id")
        or preflight.get("probe_label") != identity.get("probe_label")
        or preflight.get("shuffle_count") != identity.get("shuffle_count")
        or preflight.get("cache_directory") != cache_directory
        or preflight.get("paths") != paths
    ):
        raise ValueError("preflight.json does not match durable identity")


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write deterministic strict ASCII JSON with a trailing newline."""
    encoded = json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="ascii",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically write one nonempty ASCII text artifact."""
    encoded = text.encode("ascii", errors="backslashreplace").decode("ascii")
    if not encoded.strip():
        raise ValueError("launcher text artifact must be nonempty")
    with NamedTemporaryFile(
        "w",
        encoding="ascii",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _append_log(run_directory: Path, record: Mapping[str, object]) -> None:
    """Append and fsync one strict ASCII JSON Lines event."""
    encoded = json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n"
    with (run_directory / "run.log").open("a", encoding="ascii") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _log_event(
    state: Mapping[str, object],
    event: str,
    message: object,
) -> dict[str, object]:
    """Return one scalar launcher log record."""
    return {
        "utc": state["updated_utc"],
        "run_id": state["run_id"],
        "event": event,
        "status": state["active_status"],
        "message": str(message),
    }


def _write_summary(run_directory: Path, state: Mapping[str, object]) -> None:
    """Write a concise ASCII operational summary with no numerical arrays."""
    identity = state["identity"]
    assert isinstance(identity, Mapping)
    measurements = state["measurements"]
    assert isinstance(measurements, Mapping)
    text = (
        "# CT026 Spike-phase launcher run\n\n"
        f"Run: {state['run_id']}\n\n"
        f"Status: {state['active_status']}\n\n"
        f"Session: {identity['session_id']}\n\n"
        f"Probe: {identity['probe_label']}\n\n"
        f"Shuffles: {identity['shuffle_count']}\n\n"
        f"Requested workers: {identity['requested_worker_count']}\n\n"
        f"Cache directory: {identity['cache_directory']}\n\n"
        f"Completed stages: {state['completed_stages']}\n\n"
        f"Measurements: {json.dumps(dict(measurements), sort_keys=True, allow_nan=False)}\n\n"
        f"Warnings: {state['warnings']}\n\n"
        f"Error: {state['error'] if state['error'] is not None else 'unavailable'}\n\n"
        f"Resume command: {state['resume_command'] if state['resume_command'] is not None else 'unavailable'}\n"
    )
    _write_text_atomic(run_directory / "run_summary.md", text)


def _resume_command(run_directory: Path) -> str:
    """Return one shell-quoted absolute resume command."""
    return shlex.join(
        (
            "uv",
            "run",
            "python",
            "-m",
            _MODULE,
            "resume",
            "--run-directory",
            str(run_directory.resolve(strict=False)),
        )
    )


def _resume_from_state(state: Mapping[str, object] | None) -> str | None:
    """Return a validated saved resume string when available."""
    if state is None:
        return None
    value = state.get("resume_command")
    return value if isinstance(value, str) and value else None


def _clock(clock: Callable[[], float]) -> float:
    """Read one finite monotonic seconds value."""
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("monotonic clock must return finite seconds")
    return float(value)


def _measurement(
    state: Mapping[str, object],
    name: str,
    unavailable: object,
) -> object:
    """Return one saved scalar measurement or its explicit fallback."""
    values = state.get("measurements")
    if not isinstance(values, Mapping):
        raise ValueError("launcher measurements must be a mapping")
    value = values.get(name)
    return unavailable if value is None else value


def _optional_float(value: object) -> float | None:
    """Return a finite nonnegative float or preserve unavailable ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("measurement must be finite nonnegative seconds")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("measurement must be finite nonnegative seconds")
    return result


def _optional_int(value: object) -> int | None:
    """Return a nonnegative integer or preserve unavailable ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("measurement must be a nonnegative integer")
    return value


def _optional_string(value: object) -> str | None:
    """Return one nonempty categorical string or preserve unavailable ``None``."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("measurement must be a nonempty string")
    return value


def _file_size(path: Path) -> int:
    """Return one regular file size in bytes, or measured zero when absent."""
    return path.stat().st_size if path.is_file() else 0


def _directory_size(path: Path) -> int:
    """Return recursive regular-file bytes, or measured zero when absent."""
    if not path.is_dir():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _terminal(dependencies: LauncherDependencies, message: str) -> None:
    """Write one flushed-line-equivalent terminal message through its seam."""
    dependencies.terminal_write(str(message))


@contextmanager
def _launcher_lock(
    run_directory: Path,
    state: Mapping[str, object],
    dependencies: LauncherDependencies,
) -> Iterator[None]:
    """Hold one nonblocking advisory lock for the exact launcher directory."""
    import fcntl

    path = run_directory / "launcher.lock"
    with path.open("a+", encoding="ascii") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise _LauncherLockError("launcher run is already locked") from error
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "run_id": state["run_id"],
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "acquired_utc": dependencies.now_utc(),
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class _LinuxProcessTreeSampler:
    """Best-effort standard-library peak RSS/PSS sampler for one process tree."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        """Configure a positive polling interval in seconds."""
        if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError("memory sampling interval must be positive seconds")
        self.sampling_interval_seconds = float(interval_seconds)
        self.process_rss_bytes: int | None = None
        self.process_tree_rss_bytes: int | None = None
        self.process_tree_pss_bytes: int | None = None
        self.maximum_child_process_count: int | None = None
        self.provenance: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_LinuxProcessTreeSampler":
        """Start daemon polling and return this mutable scalar record."""
        self._sample()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop polling, take one final sample, and retain peak scalar values."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sampling_interval_seconds * 4.0))
        self._sample()

    def _poll(self) -> None:
        """Poll until the owning context requests shutdown."""
        while not self._stop.wait(self.sampling_interval_seconds):
            self._sample()

    def _sample(self) -> None:
        """Update peaks from deduplicated Linux procfs process identities."""
        try:
            pids = _process_tree_pids(os.getpid())
            samples = {pid: _proc_memory_bytes(pid) for pid in pids}
            usable = {pid: value for pid, value in samples.items() if value is not None}
            if not usable:
                return
            root = usable.get(os.getpid())
            if root is not None:
                self.process_rss_bytes = max(self.process_rss_bytes or 0, root[0])
            tree_rss = sum(value[0] for value in usable.values())
            tree_pss_values = [value[1] for value in usable.values()]
            self.process_tree_rss_bytes = max(self.process_tree_rss_bytes or 0, tree_rss)
            if all(value is not None for value in tree_pss_values):
                tree_pss = sum(int(value) for value in tree_pss_values)
                self.process_tree_pss_bytes = max(self.process_tree_pss_bytes or 0, tree_pss)
            child_count = max(0, len(usable) - 1)
            self.maximum_child_process_count = max(self.maximum_child_process_count or 0, child_count)
            self.provenance = "Linux /proc smaps_rollup process-tree sampler"
        except (OSError, UnicodeDecodeError, ValueError):
            return


def _process_tree_pids(root_pid: int) -> set[int]:
    """Return recursively discovered Linux child pids including the root."""
    discovered = {root_pid}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            children = children_path.read_text(encoding="ascii").split()
        except OSError:
            continue
        for raw in children:
            child = int(raw)
            if child not in discovered:
                discovered.add(child)
                pending.append(child)
    return discovered


def _proc_memory_bytes(pid: int) -> tuple[int, int | None] | None:
    """Read one process RSS and optional PSS from Linux ``smaps_rollup``."""
    try:
        lines = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in lines:
        if line.startswith(("Rss:", "Pss:")):
            name, raw, unit = line.split()[:3]
            if unit != "kB":
                raise ValueError("unexpected procfs memory unit")
            values[name.rstrip(":")] = int(raw) * 1024
    if "Rss" not in values:
        return None
    return values["Rss"], values.get("Pss")


def make_production_launcher_dependencies() -> LauncherDependencies:
    """Bind direct-path and session-metadata runtime, report, and cleanup seams.

    Returns
    -------
    LauncherDependencies
        Production callables. The population seam reads only the explicitly
        selected probe, while numerical LFP/PPC work remains inside the
        existing Spike-phase pipeline.
    """
    import pandas as pd

    from src.neural_analysis.lfp_spike_phase_validation import (
        build_ct026_active_population,
        build_ct026_spike_phase_preview_config,
        make_production_spike_phase_preview_dependencies,
        render_cached_spike_phase_report,
    )
    from src.neural_analysis.lfp_summary.cache import load_component_arrays
    from src.neural_analysis.lfp_summary.pipeline import (
        compute_spike_phase_component,
    )
    from src.neural_analysis.lfp_summary_runtime import (
        load_configured_trial_table,
        make_spike_phase_pipeline_dependencies,
    )
    from src.neural_analysis.lfp_summary.work_cache import cleanup_ppc_run
    from src.neural_analysis.lfp_summary.session import (
        build_metadata_spike_phase_config,
    )
    from src.neural_analysis.session_metadata import (
        load_session_metadata,
        resolve_session_metadata,
        validate_session_for_action,
    )
    from src.neural_analysis.spike_behavior.loading import load_channel_quality, load_sorter_metadata

    pipeline = make_spike_phase_pipeline_dependencies(
        trial_table_loader=load_configured_trial_table,
    )
    report_dependencies = make_production_spike_phase_preview_dependencies()

    def load_population(
        session_path: Path,
        probe_label: str,
    ) -> UnitPopulationConfig:
        """Load one probe's quality-filtered units from sorter metadata."""
        return build_ct026_active_population(
            session_path,
            probe_label,
            lambda sorter: load_sorter_metadata(sorter)[1],
            lambda sorter: load_channel_quality(sorter.parent),
        )

    def build_config(
        session_path: Path,
        population: UnitPopulationConfig,
        shuffle_count: int,
        worker_count: int,
    ) -> LFPSummaryConfig:
        """Build CT026 settings with explicit count-only execution overrides."""
        base = build_ct026_spike_phase_preview_config(session_path, population)
        return replace(
            base,
            ppc=replace(base.ppc, shuffle_count=shuffle_count),
            ppc_execution=replace(
                base.ppc_execution,
                worker_count=worker_count,
                checkpoint_enabled=True,
                checkpoint_retention="incomplete_only",
            ),
        )

    def load_metadata_config(
        metadata_path: Path,
        probe_label: str,
        cache_directory: Path,
        shuffle_count: int,
        worker_count: int,
    ) -> LFPSummaryConfig:
        """Build one existing Spike-phase request from explicit session metadata."""
        metadata = load_session_metadata(metadata_path)
        session = resolve_session_metadata(metadata, metadata_path)
        availability = validate_session_for_action(session, "spike-phase")
        if not availability.available:
            raise ValueError(
                "session metadata is missing Spike-phase inputs: "
                + ", ".join(availability.missing_inputs)
            )
        return build_metadata_spike_phase_config(
            session,
            probe_id=probe_label,
            cache_directory=cache_directory,
            shuffle_count=shuffle_count,
            worker_count=worker_count,
            cluster_metadata_loader=lambda sorter: pd.read_csv(
                sorter / "cluster_info.tsv",
                sep="\t",
            ),
            channel_metadata_loader=load_channel_quality,
        )

    def repository_state() -> RepositoryState:
        """Read the containing repository's commit and tracked cleanliness."""
        repository_root = Path(__file__).resolve().parents[2]
        commit = subprocess.run(
            ("git", "-C", str(repository_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked = subprocess.run(
            (
                "git", "-C", str(repository_root), "status", "--porcelain",
                "--untracked-files=no",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return RepositoryState(repository_root, commit, not tracked.strip())

    def repository_commit_is_ancestor(original: str, current: str) -> bool:
        """Return whether ``original`` is a Git ancestor of ``current``."""
        repository_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            (
                "git",
                "-C",
                str(repository_root),
                "merge-base",
                "--is-ancestor",
                original,
                current,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(
                "unable to establish report checkout ancestry: "
                + result.stderr.strip()
            )
        return result.returncode == 0

    def component_is_compatible(config: LFPSummaryConfig) -> bool:
        """Return whether the exact Spike component is complete and compatible."""
        manifest = load_or_initialize_manifest(config.output_directory, config)
        status = assess_component_status(
            config.output_directory,
            "spike_phase",
            config,
            manifest,
        )
        return status.status == "compatible"

    def validate_component(config: LFPSummaryConfig) -> None:
        """Load and validate the exact compatible Spike component arrays."""
        manifest = load_or_initialize_manifest(config.output_directory, config)
        status = assess_component_status(
            config.output_directory,
            "spike_phase",
            config,
            manifest,
        )
        if status.status != "compatible":
            raise ValueError(
                "Spike-phase component is not compatible: "
                + "; ".join(status.differences)
            )
        load_component_arrays(
            config.output_directory / "spike_phase.npz",
            manifest,
            "spike_phase",
        )

    def compute_component(
        config: LFPSummaryConfig,
        progress_callback: Callable[[ProgressEvent], None],
        cleanup_preparation_observer: Callable[
            [tuple[PPCWorkCleanupTarget, ...]], None
        ],
    ) -> ComponentRunResult:
        """Compute only Spike phase and defer exact PPC work cleanup."""
        return compute_spike_phase_component(
            config,
            pipeline,
            progress_callback,
            defer_post_commit_cleanup=True,
            cleanup_preparation_observer=cleanup_preparation_observer,
        )

    def cleanup_target(target: PPCWorkCleanupTarget) -> None:
        """Delete one fingerprint-validated PPC work directory."""
        cleanup_ppc_run(target.run_directory, target.run_fingerprint)

    return LauncherDependencies(
        load_active_population=load_population,
        build_config=build_config,
        load_trial_count=lambda config: len(load_configured_trial_table(config)),
        repository_state=repository_state,
        source_fingerprints=lambda config: fingerprint_source_files(
            config,
            component="spike_phase",
        ),
        component_is_compatible=component_is_compatible,
        compute_component=compute_component,
        validate_component=validate_component,
        publish_report=lambda **kwargs: render_cached_spike_phase_report(
            dependencies=report_dependencies,
            **kwargs,
        ),
        cleanup_target=cleanup_target,
        now_utc=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H-%M-%SZ"
        ),
        monotonic_seconds=time.monotonic,
        process_tree_sampler=_LinuxProcessTreeSampler,
        terminal_write=lambda message: print(message, flush=True),
        repository_commit_is_ancestor=repository_commit_is_ancestor,
        load_metadata_config=load_metadata_config,
    )


@contextmanager
def _launcher_signal_handlers() -> Iterator[None]:
    """Translate the first SIGINT/SIGTERM into a persisted launcher stop."""
    handled = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in handled}
    received = 0

    def handle(signum: int, _frame: object) -> None:
        """Raise on the first signal and force the default action thereafter."""
        nonlocal received
        received += 1
        if received == 1:
            raise _LauncherSignal(signum)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        for signum in handled:
            signal.signal(signum, handle)
        yield
    finally:
        for signum, prior in previous.items():
            signal.signal(signum, prior)


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
