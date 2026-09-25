"""RED contracts for the standalone resumable Spike-phase launcher."""

from __future__ import annotations

import builtins
from contextlib import nullcontext
from dataclasses import replace
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from src.neural_analysis.lfp_spike_phase_validation import (
    build_ct026_spike_phase_preview_config,
)
from src.neural_analysis import lfp_summary_io, lfp_summary_runtime
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    UnitPopulationConfig,
    canonical_config_json,
    component_fingerprint,
    fingerprint_source_files,
    source_value_semantics,
)
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult


_MAX_RETAINED_IDENTITY_JSON_BYTES = 64 * 1024
_PREPARED_REPRESENTATION_IDENTITY_KEYS = (
    "generator",
    "analysis_version",
    "source_fingerprint",
    "scientific_fingerprint",
    "axes",
    "shapes",
    "dtypes",
)


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


def _corrected_cache_directory(tmp_path: Path) -> Path:
    """Return the sole ordinary synthetic corrected cache target.

    The path is a direct child of the selected session's ``processed``
    directory. It intentionally differs from the protected legacy cache name.
    """
    return tmp_path / "CT026" / "processed" / "corrected_lfp_summary_cache"


def _new_command_arguments(
    tmp_path: Path,
    *,
    probe: str = "ProbeB",
    shuffle_count: int = 100,
    worker_count: int | None = None,
    analysis_root: Path | None = None,
    cache_directory: Path | None = None,
    dry_run: bool = False,
    final_run: bool = False,
) -> list[str]:
    """Build one explicit new-run CLI request with a cache target.

    Parameters use paths and count-only launcher values. The helper creates no
    directory or numerical data; fixtures create the compatibility cache
    separately when a run must reach preflight.
    """
    arguments = [
        "new",
        "--session-path",
        str(tmp_path / "CT026"),
        "--cache-directory",
        str(cache_directory or _corrected_cache_directory(tmp_path)),
        "--probe",
        probe,
        "--shuffles",
        str(shuffle_count),
    ]
    if worker_count is not None:
        arguments.extend(("--workers", str(worker_count)))
    if analysis_root is not None:
        arguments.extend(("--analysis-root", str(analysis_root)))
    if dry_run:
        arguments.append("--dry-run")
    if final_run:
        arguments.append("--final-run")
    return arguments


def _cache_config(
    tmp_path: Path,
    cache_directory: Path,
    *,
    probe: str = "ProbeB",
    shuffle_count: int = 100,
    worker_count: int = 8,
) -> LFPSummaryConfig:
    """Build the exact immutable configuration expected for a cache target.

    No input recording exists in this synthetic fixture. Missing inputs are
    represented by the public source-fingerprint contract, so no signal array
    is opened while a Power/Synchrony prerequisite is assembled.
    """
    config = build_ct026_spike_phase_preview_config(
        tmp_path / "CT026",
        _population(probe),
    )
    return replace(
        config,
        output_directory=cache_directory,
        ppc=replace(config.ppc, shuffle_count=shuffle_count),
        ppc_execution=replace(
            config.ppc_execution,
            worker_count=worker_count,
            checkpoint_enabled=True,
            checkpoint_retention="incomplete_only",
        ),
    )


def _write_prerequisite_cache(
    tmp_path: Path,
    cache_directory: Path | None = None,
    *,
    power_state: str = "complete",
    synchrony_state: str = "complete",
    include_spike_phase: bool = False,
    unexpected_member: str | None = None,
) -> LFPSummaryConfig:
    """Write a minimal complete Power/Synchrony cache using public contracts.

    The tiny scalar NPZ files stand in only for cache schema/header validation.
    They contain no experimental samples, phase values, PPC results, or report
    output. Every manifest fingerprint is generated from the immutable active
    configuration supplied to the launcher.
    """
    target = cache_directory or _corrected_cache_directory(tmp_path)
    config = _cache_config(tmp_path, target)
    target.mkdir(parents=True, exist_ok=False)
    schema = {"values": {"axes": ["sample"], "units": "dimensionless"}}
    components: dict[str, object] = {}
    for component, state in (
        ("power", power_state),
        ("synchrony", synchrony_state),
    ):
        components[component] = {
            "file_name": f"{component}.npz",
            "state": state,
            "configuration_fingerprint": component_fingerprint(component, config),
            "configuration_snapshot": json.loads(canonical_config_json(config)),
            "source_fingerprints": fingerprint_source_files(config, component),
            "source_value_semantics": source_value_semantics(config),
            "array_schema": schema,
        }
        np.savez(target / f"{component}.npz", values=np.array((1.0,)))
    if include_spike_phase:
        components["spike_phase"] = {
            "file_name": "spike_phase.npz",
            "state": "complete",
            "configuration_fingerprint": component_fingerprint("spike_phase", config),
            "configuration_snapshot": json.loads(canonical_config_json(config)),
            "source_fingerprints": fingerprint_source_files(config, "spike_phase"),
            "source_value_semantics": source_value_semantics(config),
            "array_schema": schema,
        }
        np.savez(target / "spike_phase.npz", values=np.array((1.0,)))
    manifest = {
        "schema_version": config.schema_version,
        "session_id": config.session_id,
        "configuration": json.loads(canonical_config_json(config)),
        "components": components,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="ascii",
    )
    if unexpected_member is not None:
        (target / unexpected_member).write_text("unexpected\n", encoding="ascii")
    return config


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
    report: Callable[..., object] | None = None,
    built_configurations: list[LFPSummaryConfig] | None = None,
) -> object:
    """Return deterministic high-level launcher seams with no raw-data access."""
    launcher = _launcher()
    default_cache = _corrected_cache_directory(tmp_path)
    if not default_cache.exists():
        _write_prerequisite_cache(tmp_path, default_cache)

    def build_config(
        session_path: Path,
        population: UnitPopulationConfig,
        shuffle_count: int,
        worker_count: int,
    ) -> object:
        """Build the reviewed CT026 configuration with explicit execution values."""
        config = build_ct026_spike_phase_preview_config(session_path, population)
        configured = replace(
            config,
            ppc=replace(config.ppc, shuffle_count=shuffle_count),
            ppc_execution=replace(
                config.ppc_execution,
                worker_count=worker_count,
                checkpoint_enabled=True,
                checkpoint_retention="incomplete_only",
            ),
        )
        if built_configurations is not None:
            built_configurations.append(configured)
        return configured

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
        publish_report=report or publish_report,
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


def _prepared_phase_metadata(representation_fingerprint: str) -> dict[str, object]:
    """Return a metadata-only retained prepared-phase identity record.

    The launcher preflight may inspect this JSON record but must not inspect
    the synthetic numerical payload files stored beside it.
    """
    return {
        "generator": "synthetic_launcher_test",
        "analysis_version": "test",
        "schema_version": "prepared_phase_cache.v1",
        "source_fingerprint": "a" * 64,
        "scientific_fingerprint": "b" * 64,
        "representation_fingerprint": representation_fingerprint,
        "execution_settings": {},
        "axes": ["site", "frequency", "trial", "time"],
        "shapes": {},
        "dtypes": {},
        "units": {},
    }


def _write_retained_prepared_phase(
    work_root: Path,
    representation_fingerprint: str,
    *,
    metadata: object | None = None,
    completion: object | None = None,
) -> Path:
    """Write one metadata-valid retained work representation without arrays.

    The ``.npy`` and ``.npz`` payloads deliberately contain opaque ASCII
    sentinels rather than numerical data. A launcher preflight that tries to
    open them therefore violates this test fixture's contract immediately.
    """
    representation = (
        work_root / "prepared_phase" / representation_fingerprint
    )
    representation.mkdir(parents=True)
    representation_metadata = (
        _prepared_phase_metadata(representation_fingerprint)
        if metadata is None
        else metadata
    )
    representation_completion = (
        {"representation_fingerprint": representation_fingerprint}
        if completion is None
        else completion
    )
    (representation / "metadata.json").write_text(
        json.dumps(representation_metadata, sort_keys=True) + "\n",
        encoding="ascii",
    )
    (representation / "complete.json").write_text(
        json.dumps(representation_completion, sort_keys=True) + "\n",
        encoding="ascii",
    )
    (representation / "axes.npz").write_bytes(b"retained synthetic axes\n")
    (representation / "valid.npy").write_bytes(b"retained synthetic validity\n")
    (representation / "phase.npy").write_bytes(b"retained synthetic phase\n")
    return representation


def _authentic_prepared_phase_metadata(
    config: LFPSummaryConfig,
    *,
    trial_offset: int = 0,
) -> dict[str, object]:
    """Build real runtime metadata for a small synthetic prepared trial axis.

    This calls the pure runtime metadata builder and its fingerprint algorithm;
    it never creates a phase array or reads experimental data. The two trial
    positions and two finite alignment times are scalar test identities only.
    """
    trial_indices = np.asarray((trial_offset, trial_offset + 1), dtype=np.int64)
    alignment_times_s = np.asarray(
        (float(trial_offset) + 1.0, float(trial_offset) + 2.0),
        dtype=np.float64,
    )
    metadata = lfp_summary_runtime._prepared_phase_work_metadata(
        config,
        trial_indices,
        alignment_times_s,
    )
    representation_identity = {
        key: metadata[key] for key in _PREPARED_REPRESENTATION_IDENTITY_KEYS
    }
    assert metadata["representation_fingerprint"] == (
        lfp_summary_runtime._work_fingerprint(representation_identity)
    )
    return metadata


def _write_authentic_retained_prepared_phase(
    work_root: Path,
    config: LFPSummaryConfig,
    *,
    trial_offset: int = 0,
    metadata: dict[str, object] | None = None,
    completion: object | None = None,
) -> Path:
    """Write authentic metadata with opaque retained numerical sentinels.

    The metadata follows the live writer/runtime contract exactly. The three
    numerical member files remain non-array ASCII sentinels because launcher
    preflight must classify their paths without opening their contents.
    """
    authentic_metadata = (
        _authentic_prepared_phase_metadata(config, trial_offset=trial_offset)
        if metadata is None
        else metadata
    )
    fingerprint = authentic_metadata["representation_fingerprint"]
    assert isinstance(fingerprint, str)
    return _write_retained_prepared_phase(
        work_root,
        fingerprint,
        metadata=authentic_metadata,
        completion=completion,
    )


def _install_lstat_replacement_race(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    occurrence: int,
    replacement: Callable[[], None],
) -> None:
    """Replace one entry after its chosen ``lstat`` result is returned.

    The hook simulates a deterministic filesystem race without naming a
    launcher helper. A safe preflight must reject rather than inspect the
    replacement, load trials, or create a run directory.
    """
    real_lstat = Path.lstat
    observed = 0

    def raced_lstat(path: Path) -> os.stat_result:
        """Return original metadata, then atomically perform the test race."""
        nonlocal observed
        status = real_lstat(path)
        if path == target:
            observed += 1
            if observed == occurrence:
                replacement()
        return status

    monkeypatch.setattr(Path, "lstat", raced_lstat)


def _install_descriptor_close_replacement_race(
    monkeypatch: pytest.MonkeyPatch,
    closed_entry_status: os.stat_result,
    replacement: Callable[[], None],
) -> dict[str, bool]:
    """Replace an entry immediately after its anchored descriptor closes.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patch controller used to replace the standard :func:`os.close` seam.
    closed_entry_status : os.stat_result
        Pre-race device and inode identity of the directory whose descriptor
        close should trigger the replacement. No directory contents are read.
    replacement : collections.abc.Callable[[], None]
        Atomic same-name replacement action performed after the original
        descriptor has closed.

    Returns
    -------
    dict[str, bool]
        Mutable ``{"triggered": bool}`` record proving the close boundary was
        reached during the launcher invocation.
    """
    real_close = os.close
    race_state = {"triggered": False}

    def raced_close(descriptor: int) -> None:
        """Close first, then replace only the matching closed directory."""
        try:
            status = os.fstat(descriptor)
        except OSError:
            real_close(descriptor)
            return
        real_close(descriptor)
        if (
            not race_state["triggered"]
            and status.st_dev == closed_entry_status.st_dev
            and status.st_ino == closed_entry_status.st_ino
        ):
            replacement()
            race_state["triggered"] = True

    monkeypatch.setattr(os, "close", raced_close)
    return race_state


def _forbid_raced_identity_path_opening(
    monkeypatch: pytest.MonkeyPatch,
    identity_path: Path,
) -> None:
    """Prevent an unsafe high-level read after an identity path is replaced."""
    real_path_open = Path.open

    def guarded_path_open(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Fail deterministically if preflight reopens the raced identity path."""
        if path == identity_path:
            raise AssertionError("preflight attempted to read a raced identity path")
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)


def _work_tree_snapshot(work_root: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    """Return a deterministic directory inventory and regular-file bytes."""
    directories: list[str] = []
    files: dict[str, bytes] = {}
    for path in sorted(work_root.rglob("*")):
        relative = str(path.relative_to(work_root))
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files[relative] = path.read_bytes()
    return tuple(directories), files


def _preflight_only_dependencies(
    tmp_path: Path,
    calls: list[str],
    terminal: list[str],
    *,
    label: str,
) -> object:
    """Return seams that fail if unsafe work reaches trial loading or a run."""
    return replace(
        _dependencies(tmp_path, calls, terminal),
        load_trial_count=lambda _config: (_ for _ in ()).throw(
            AssertionError(f"{label} work reached trial loading")
        ),
        compute_component=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"{label} work reached numerical computation")
        ),
        publish_report=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"{label} work reached report rendering")
        ),
    )


def _forbid_retained_payload_opening(
    monkeypatch: pytest.MonkeyPatch,
    launcher: object,
    representations: tuple[Path, ...],
) -> None:
    """Fail if preflight opens any synthetic retained numerical payload."""
    numerical_paths = {
        representation / filename
        for representation in representations
        for filename in ("phase.npy", "valid.npy", "axes.npz")
    }
    real_builtin_open = builtins.open
    real_path_open = Path.open

    def forbid_payload_open(
        file: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Delegate every non-payload open to the standard implementation."""
        if isinstance(file, (str, os.PathLike)) and Path(file) in numerical_paths:
            raise AssertionError("launcher preflight opened retained numerical payload")
        return real_builtin_open(file, *args, **kwargs)

    def forbid_payload_path_open(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Protect ``Path.read_*`` calls in addition to builtin ``open``."""
        if path in numerical_paths:
            raise AssertionError("launcher preflight opened retained numerical payload")
        return real_path_open(path, *args, **kwargs)

    def forbid_array_loading(*_args: object, **_kwargs: object) -> object:
        """Fail if preflight delegates retained work to an array loader."""
        raise AssertionError("launcher preflight loaded retained numerical arrays")

    monkeypatch.setattr(builtins, "open", forbid_payload_open)
    monkeypatch.setattr(Path, "open", forbid_payload_path_open)
    monkeypatch.setattr(np, "load", forbid_array_loading)
    monkeypatch.setattr(lfp_summary_io, "load_component_arrays", forbid_array_loading)
    monkeypatch.setattr(
        launcher,
        "load_component_arrays",
        forbid_array_loading,
        raising=False,
    )


def test_parser_requires_explicit_cache_probe_and_final_confirmation() -> None:
    """CLI accepts only explicit corrected-cache preview/final requests."""
    launcher = _launcher()
    preview = launcher.parse_launcher_command(
        [
            "new",
            "--session-path",
            "/data/CT026",
            "--cache-directory",
            "/data/CT026/processed/corrected-cache",
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
    assert preview.cache_directory == Path("/data/CT026/processed/corrected-cache")

    final = launcher.parse_launcher_command(
        [
            "new",
            "--session-path",
            "/data/CT026",
            "--cache-directory",
            "/data/CT026/processed/corrected-cache-v2",
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
    recovery = launcher.parse_launcher_command(
        ["recover-report", "--run-directory", "/runs/failed-report"]
    )
    assert recovery.mode == "recover-report"
    assert recovery.run_directory == Path("/runs/failed-report")
    rerender = launcher.parse_launcher_command(
        ["rerender-report", "--run-directory", "/runs/completed-report"]
    )
    assert rerender.mode == "rerender-report"
    assert rerender.run_directory == Path("/runs/completed-report")

    invalid = (
        ["new", "--session-path", "/data/CT026", "--shuffles", "100"],
        [
            "new", "--session-path", "/data/CT026", "--probe", "ProbeB",
            "--shuffles", "100",
        ],
        [
            "new", "--session-path", "/data/CT026", "--probe", "ProbeB",
            "--cache-directory", "/data/CT026/processed/corrected-cache",
            "--shuffles", "1000",
        ],
        [
            "new", "--session-path", "/data/CT026", "--probe", "ProbeB",
            "--cache-directory", "/data/CT026/processed/corrected-cache",
            "--shuffles", "100", "--final-run",
        ],
    )
    for arguments in invalid:
        with pytest.raises(SystemExit):
            launcher.parse_launcher_command(arguments)


def test_parser_accepts_metadata_as_new_run_session_source() -> None:
    """A new run accepts one metadata file and arbitrary stable probe ID."""
    launcher = _launcher()

    command = launcher.parse_launcher_command(
        [
            "new",
            "--session-metadata",
            "/data/mouse-z/neural_session.json",
            "--cache-directory",
            "/data/mouse-z/processed/summary-cache",
            "--probe",
            "rear-probe",
            "--shuffles",
            "100",
        ]
    )

    assert command.session_path is None
    assert command.session_metadata == Path("/data/mouse-z/neural_session.json")
    assert command.probe_label == "rear-probe"
    assert command.analysis_root is None


def test_parser_requires_exactly_one_new_run_session_source() -> None:
    """Metadata and the legacy session directory are mutually exclusive."""
    launcher = _launcher()
    common = [
        "--cache-directory",
        "/data/mouse-z/processed/summary-cache",
        "--probe",
        "rear-probe",
        "--shuffles",
        "100",
    ]

    with pytest.raises(SystemExit):
        launcher.parse_launcher_command(["new", *common])
    with pytest.raises(SystemExit):
        launcher.parse_launcher_command(
            [
                "new",
                "--session-path",
                "/data/mouse-z",
                "--session-metadata",
                "/data/mouse-z/neural_session.json",
                *common,
            ]
        )


@pytest.mark.parametrize("mode", ("resume", "recover-report", "rerender-report"))
def test_parser_rejects_cache_replacement_for_every_recovery_mode(mode: str) -> None:
    """Only a new run may name a cache; recovery is bound to saved identity."""
    launcher = _launcher()

    with pytest.raises(SystemExit):
        launcher.parse_launcher_command(
            [
                mode,
                "--run-directory",
                "/runs/exact",
                "--cache-directory",
                "/data/CT026/processed/replacement",
            ]
        )


def test_dry_run_writes_evidence_without_scientific_work(tmp_path: Path) -> None:
    """Metadata preflight is terminal and never invokes computation or reporting."""
    launcher = _launcher()
    calls: list[str] = []
    terminal: list[str] = []
    dependencies = _dependencies(tmp_path, calls, terminal)
    cache_directory = _corrected_cache_directory(tmp_path).resolve()
    cache_before = {
        child.name: child.read_bytes()
        for child in cache_directory.iterdir()
        if child.is_file()
    }
    assert set(cache_before) == {"manifest.json", "power.npz", "synchrony.npz"}
    assert not (cache_directory.parent / "lfp_summary_work").exists()
    command = launcher.parse_launcher_command(
        _new_command_arguments(tmp_path, dry_run=True)
    )

    result = launcher.run_launcher(command, dependencies)

    assert result.exit_code == 0 and result.status == "preflight_complete"
    assert calls == []
    assert result.run_directory.is_dir()
    state = json.loads((result.run_directory / "launcher_state.json").read_text())
    configuration = json.loads((result.run_directory / "configuration.json").read_text())
    preflight = json.loads((result.run_directory / "preflight.json").read_text())
    summary = (result.run_directory / "run_summary.md").read_text(encoding="ascii")
    cache_after = {
        child.name: child.read_bytes()
        for child in cache_directory.iterdir()
        if child.is_file()
    }
    assert state["resume_command"] is None
    assert configuration["output_directory"] == str(cache_directory)
    assert state["identity"]["cache_directory"] == str(cache_directory)
    assert state["paths"]["output_directory"] == str(cache_directory)
    assert preflight["cache_directory"] == str(cache_directory)
    assert preflight["paths"]["output_directory"] == str(cache_directory)
    assert f"Cache directory: {cache_directory}" in summary
    assert preflight["schema_version"] == "spike_phase_preflight.v1"
    assert preflight["exact_plan_available"] is False
    assert preflight["ppc_planning_seconds"] is None
    assert preflight["planned_ppc_allocation_bytes"] is None
    assert cache_after == cache_before
    assert not (cache_directory / "spike_phase.npz").exists()
    assert not (cache_directory.parent / "lfp_summary_work").exists()
    assert not (result.run_directory / "report").exists()
    assert list(result.run_directory.parent.iterdir()) == [result.run_directory]
    assert not (tmp_path / "CT026" / "processed" / "lfp_summary_cache").exists()


def test_new_run_immutably_replaces_only_the_builder_cache_directory(
    tmp_path: Path,
) -> None:
    """The caller-selected target replaces only frozen ``output_directory``."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "arbitrary-corrected-cache"
    _write_prerequisite_cache(tmp_path, target)
    calls: list[str] = []
    terminal: list[str] = []
    builder_results: list[LFPSummaryConfig] = []
    dependencies = _dependencies(
        tmp_path,
        calls,
        terminal,
        built_configurations=builder_results,
    )

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=tmp_path / "runs",
                dry_run=True,
            )
        ),
        dependencies,
    )

    assert result.exit_code == 0 and result.status == "preflight_complete"
    assert len(builder_results) == 1
    original = builder_results[0]
    expected = replace(original, output_directory=target.resolve())
    saved = json.loads(
        (result.run_directory / "configuration.json").read_text(encoding="ascii")
    )
    assert saved == json.loads(canonical_config_json(expected))
    assert original.output_directory == tmp_path / "CT026" / "processed" / "lfp_summary_cache"
    assert original != expected


def test_preflight_ignores_unrelated_empty_work_roots(tmp_path: Path) -> None:
    """Only the exact derived work root blocks a new run; unrelated roots do not."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "second-corrected-cache"
    _write_prerequisite_cache(tmp_path, target)
    (tmp_path / "unrelated" / "lfp_summary_work").mkdir(parents=True)
    (target.parent / "other_lfp_summary_work").mkdir()
    calls: list[str] = []
    terminal: list[str] = []

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=tmp_path / "runs",
                dry_run=True,
            )
        ),
        _dependencies(tmp_path, calls, terminal),
    )

    assert result.exit_code == 0 and result.status == "preflight_complete"
    assert calls == []


def test_preflight_uses_public_header_only_component_status_without_array_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Power/Synchrony/Spike checks inspect only public archive headers."""
    launcher = _launcher()
    calls: list[str] = []
    terminal: list[str] = []
    dependencies = _dependencies(tmp_path, calls, terminal)
    seen_status_calls: list[tuple[str, bool]] = []
    real_assess = lfp_summary_io.assess_component_status

    def record_status(
        cache_directory: Path,
        component: str,
        config: LFPSummaryConfig,
        manifest: object,
        **kwargs: object,
    ) -> object:
        """Record the public status mode while retaining real header validation."""
        seen_status_calls.append((component, bool(kwargs.get("validate_headers_only"))))
        return real_assess(cache_directory, component, config, manifest, **kwargs)

    def forbid_array_loading(*_args: object, **_kwargs: object) -> object:
        """Fail if a cache prerequisite opens an NPZ numerical array collection."""
        raise AssertionError("launcher preflight must not load component arrays")

    monkeypatch.setattr(lfp_summary_io, "assess_component_status", record_status)
    monkeypatch.setattr(
        launcher,
        "assess_component_status",
        record_status,
        raising=False,
    )
    monkeypatch.setattr(lfp_summary_io, "load_component_arrays", forbid_array_loading)
    monkeypatch.setattr(
        launcher,
        "load_component_arrays",
        forbid_array_loading,
        raising=False,
    )
    monkeypatch.setattr(np, "load", forbid_array_loading)

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(tmp_path, dry_run=True)
        ),
        dependencies,
    )

    assert result.exit_code == 0 and result.status == "preflight_complete"
    assert seen_status_calls == [
        ("power", True),
        ("synchrony", True),
        ("spike_phase", True),
    ]
    assert calls == []


def test_preflight_permits_retained_prepared_phase_and_empty_ppc_without_opening_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-success retained representation permits an immutable dry run."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "retained-work-cache"
    config = _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    representation = _write_authentic_retained_prepared_phase(work_root, config)
    (work_root / "ppc").mkdir()
    before = _work_tree_snapshot(work_root)
    calls: list[str] = []
    terminal: list[str] = []

    # The open guards apply only to launcher preflight. The subsequent snapshot
    # intentionally reads the fixture bytes to prove the work stayed unchanged.
    with monkeypatch.context() as guarded_monkeypatch:
        _forbid_retained_payload_opening(
            guarded_monkeypatch,
            launcher,
            (representation,),
        )
        result = launcher.run_launcher(
            launcher.parse_launcher_command(
                _new_command_arguments(
                    tmp_path,
                    cache_directory=target,
                    analysis_root=tmp_path / "runs-retained-work",
                    dry_run=True,
                )
            ),
            _dependencies(tmp_path, calls, terminal),
        )

    assert result.exit_code == 0 and result.status == "preflight_complete"
    assert calls == []
    assert _work_tree_snapshot(work_root) == before


def test_preflight_permits_multiple_complete_retained_prepared_representations(
    tmp_path: Path,
) -> None:
    """Preflight classifies every safe retained representation as inert work."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "multiple-retained-work-cache"
    config = _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    _write_authentic_retained_prepared_phase(work_root, config)
    _write_authentic_retained_prepared_phase(
        work_root,
        config,
        trial_offset=2,
    )
    calls: list[str] = []
    terminal: list[str] = []

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=tmp_path / "runs-multiple-retained-work",
                dry_run=True,
            )
        ),
        _dependencies(tmp_path, calls, terminal),
    )

    assert result.exit_code == 0 and result.status == "preflight_complete"
    assert calls == []


@pytest.mark.parametrize(
    "containers",
    ((), ("prepared_phase",), ("ppc",), ("prepared_phase", "ppc")),
)
def test_preflight_permits_existing_empty_safe_work_layout(
    tmp_path: Path,
    containers: tuple[str, ...],
) -> None:
    """An empty shared root and its optional empty containers are inert."""
    launcher = _launcher()
    layout_name = "-".join(containers) or "root-only"
    target = tmp_path / "CT026" / "processed" / f"empty-work-{layout_name}"
    _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    work_root.mkdir()
    for container in containers:
        (work_root / container).mkdir()
    calls: list[str] = []
    terminal: list[str] = []

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=tmp_path / f"runs-empty-work-{layout_name}",
                dry_run=True,
            )
        ),
        _dependencies(tmp_path, calls, terminal),
    )

    assert result.exit_code == 0 and result.status == "preflight_complete"
    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    (
        "source-identity",
        "scientific-identity",
        "representation-input",
        "missing-schema-key",
        "extra-schema-key",
        "wrong-generator-type",
        "wrong-schema-version",
        "wrong-execution-settings-type",
        "wrong-axes",
        "wrong-shapes-type",
        "wrong-dtypes-type",
        "wrong-units-type",
    ),
)
def test_preflight_rejects_noncanonical_authentic_prepared_metadata_before_trial_loading(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Retained metadata must remain an exact self-authenticating writer record."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"metadata-{mutation}-cache"
    config = _write_prerequisite_cache(tmp_path, target)
    metadata = _authentic_prepared_phase_metadata(config)
    fingerprint = metadata["representation_fingerprint"]
    assert isinstance(fingerprint, str)
    if mutation == "source-identity":
        metadata["source_fingerprint"] = "d" * 64
    elif mutation == "scientific-identity":
        metadata["scientific_fingerprint"] = "e" * 64
    elif mutation == "representation-input":
        metadata["generator"] = "different_phase_generator"
    elif mutation == "missing-schema-key":
        del metadata["units"]
    elif mutation == "extra-schema-key":
        metadata["unexpected"] = "not writer metadata"
    elif mutation == "wrong-generator-type":
        metadata["generator"] = 1
    elif mutation == "wrong-schema-version":
        metadata["schema_version"] = "unsupported-schema"
    elif mutation == "wrong-execution-settings-type":
        metadata["execution_settings"] = []
    elif mutation == "wrong-axes":
        metadata["axes"] = ["site", "trial"]
    elif mutation == "wrong-shapes-type":
        metadata["shapes"] = []
    elif mutation == "wrong-dtypes-type":
        metadata["dtypes"] = []
    elif mutation == "wrong-units-type":
        metadata["units"] = []
    else:
        raise AssertionError(f"unknown metadata mutation: {mutation}")
    assert metadata["representation_fingerprint"] == fingerprint
    _write_authentic_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        config,
        metadata=metadata,
    )
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-metadata-{mutation}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=mutation),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


@pytest.mark.parametrize("completion_case", ("missing", "extra"))
def test_preflight_requires_exact_one_field_authentic_prepared_completion(
    tmp_path: Path,
    completion_case: str,
) -> None:
    """Completion certifies only the matching representation fingerprint."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"completion-{completion_case}-cache"
    config = _write_prerequisite_cache(tmp_path, target)
    metadata = _authentic_prepared_phase_metadata(config)
    fingerprint = metadata["representation_fingerprint"]
    assert isinstance(fingerprint, str)
    completion: dict[str, object]
    if completion_case == "missing":
        completion = {}
    else:
        completion = {
            "representation_fingerprint": fingerprint,
            "unexpected": "not a completion certificate",
        }
    _write_authentic_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        config,
        metadata=metadata,
        completion=completion,
    )
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-completion-{completion_case}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(
            tmp_path,
            calls,
            terminal,
            label=completion_case,
        ),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


@pytest.mark.parametrize("replacement_kind", ("symlink", "fifo", "external-file"))
def test_preflight_fails_closed_when_identity_file_changes_after_status_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    """Identity races cannot read a replacement, block, or reach trial loading."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"identity-race-{replacement_kind}"
    config = _write_prerequisite_cache(tmp_path, target)
    representation = _write_authentic_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        config,
    )
    identity_path = representation / "metadata.json"
    external_path = tmp_path / f"external-{replacement_kind}-metadata.json"
    if replacement_kind != "fifo":
        external_path.write_bytes(identity_path.read_bytes())

    def replace_identity() -> None:
        """Substitute the inspected identity entry before its content read."""
        identity_path.unlink()
        if replacement_kind == "symlink":
            identity_path.symlink_to(external_path)
        elif replacement_kind == "fifo":
            os.mkfifo(identity_path)
        else:
            external_path.replace(identity_path)

    _install_lstat_replacement_race(
        monkeypatch,
        identity_path,
        occurrence=2,
        replacement=replace_identity,
    )
    _forbid_raced_identity_path_opening(monkeypatch, identity_path)
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-identity-race-{replacement_kind}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(
            tmp_path,
            calls,
            terminal,
            label=replacement_kind,
        ),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


def test_preflight_fails_closed_when_work_root_changes_before_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory identity must remain stable from status check through listing."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "work-root-directory-race"
    config = _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    work_root.mkdir()
    original_root = tmp_path / "original-work-root"
    external_root = tmp_path / "external-work-root"
    external_representation = _write_authentic_retained_prepared_phase(
        external_root,
        config,
    )
    original_inventory = _work_tree_snapshot(work_root)
    replacement_identity = (
        work_root
        / "prepared_phase"
        / external_representation.name
        / "metadata.json"
    )

    def replace_work_root() -> None:
        """Swap a safe empty root for an external valid-looking directory."""
        work_root.rename(original_root)
        external_root.rename(work_root)

    _install_lstat_replacement_race(
        monkeypatch,
        work_root,
        occurrence=1,
        replacement=replace_work_root,
    )
    _forbid_raced_identity_path_opening(monkeypatch, replacement_identity)
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / "runs-work-root-directory-race"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label="work-root-race"),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert _work_tree_snapshot(original_root) == original_inventory


def test_preflight_fails_closed_when_prepared_child_changes_before_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained representation cannot be substituted after its status check."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "prepared-child-directory-race"
    config = _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    representation = _write_authentic_retained_prepared_phase(work_root, config)
    original_representation = tmp_path / "original-prepared-representation"
    external_root = tmp_path / "external-prepared-root"
    external_representation = _write_authentic_retained_prepared_phase(
        external_root,
        config,
    )
    original_inventory = _work_tree_snapshot(representation)

    def replace_representation() -> None:
        """Swap the inspected child for a matching external directory."""
        representation.rename(original_representation)
        external_representation.rename(representation)

    _install_lstat_replacement_race(
        monkeypatch,
        representation,
        occurrence=1,
        replacement=replace_representation,
    )
    _forbid_raced_identity_path_opening(
        monkeypatch,
        representation / "metadata.json",
    )
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / "runs-prepared-child-directory-race"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(
            tmp_path,
            calls,
            terminal,
            label="prepared-child-race",
        ),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert _work_tree_snapshot(original_representation) == original_inventory


def _forbid_numerical_member_descriptor_opening(
    monkeypatch: pytest.MonkeyPatch,
    representations: tuple[Path, ...],
) -> None:
    """Fail if preflight opens any listed representation's numerical payload.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patch controller used to replace :func:`os.open` for one test.
    representations : tuple[pathlib.Path, ...]
        Absolute retained-representation directories whose ``axes.npz``,
        ``valid.npy``, and ``phase.npy`` members must remain opaque.

    Returns
    -------
    None
        Installs a guard that rejects both directory-descriptor basenames and
        absolute member paths, independent of the requested ``os.open`` mode.
    """
    real_open = os.open
    numerical_member_names = {"axes.npz", "valid.npy", "phase.npy"}
    protected_numerical_paths = {
        os.fspath(representation / member_name)
        for representation in representations
        for member_name in numerical_member_names
    }

    def guarded_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        """Delegate metadata opens while refusing numerical-member descriptors."""
        path_name = os.fsdecode(os.fspath(path))
        if (
            path_name in protected_numerical_paths
            or (dir_fd is not None and path_name in numerical_member_names)
        ):
            raise AssertionError("launcher preflight opened a numerical member")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", guarded_open)


@pytest.mark.parametrize(
    ("identity_name", "read_occurrence"),
    (("metadata.json", 1), ("complete.json", 2)),
)
def test_preflight_fails_closed_when_identity_is_replaced_after_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_name: str,
    read_occurrence: int,
) -> None:
    """A same-name identity replacement cannot certify the old descriptor data."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"post-read-{identity_name}"
    config = _write_prerequisite_cache(tmp_path, target)
    representation = _write_authentic_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        config,
    )
    identity_path = representation / identity_name
    replacement = tmp_path / f"replacement-{identity_name}"
    replacement.write_bytes(identity_path.read_bytes())
    replacement_inode = replacement.lstat().st_ino
    real_read = os.read
    observed_reads = 0

    def raced_read(descriptor: int, size: int) -> bytes:
        """Return the original descriptor bytes, then atomically replace its name."""
        nonlocal observed_reads
        encoded = real_read(descriptor, size)
        observed_reads += 1
        if observed_reads == read_occurrence:
            replacement.replace(identity_path)
        return encoded

    monkeypatch.setattr(os, "read", raced_read)
    _forbid_retained_payload_opening(monkeypatch, launcher, (representation,))
    _forbid_numerical_member_descriptor_opening(monkeypatch, (representation,))
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-post-read-{identity_name}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(
            tmp_path,
            calls,
            terminal,
            label=identity_name,
        ),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert identity_path.lstat().st_ino == replacement_inode


@pytest.mark.parametrize("member_name", ("axes.npz", "valid.npy", "phase.npy"))
def test_preflight_fails_closed_when_numerical_member_is_replaced_after_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_name: str,
) -> None:
    """An opaque numerical member cannot change after preflight classifies it."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"post-stat-{member_name}"
    config = _write_prerequisite_cache(tmp_path, target)
    representation = _write_authentic_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        config,
    )
    member_path = representation / member_name
    replacement = tmp_path / f"replacement-{member_name}"
    replacement.write_bytes(member_path.read_bytes())
    replacement_inode = replacement.lstat().st_ino
    real_stat = os.stat
    replaced = False

    def raced_stat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        """Return the original anchored status, then replace the same entry."""
        nonlocal replaced
        status = real_stat(path, *args, **kwargs)
        if (
            not replaced
            and path == member_name
            and kwargs.get("dir_fd") is not None
            and kwargs.get("follow_symlinks") is False
        ):
            replacement.replace(member_path)
            replaced = True
        return status

    monkeypatch.setattr(os, "stat", raced_stat)
    _forbid_retained_payload_opening(monkeypatch, launcher, (representation,))
    _forbid_numerical_member_descriptor_opening(monkeypatch, (representation,))
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-post-stat-{member_name}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(
            tmp_path,
            calls,
            terminal,
            label=member_name,
        ),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert replaced is True
    assert member_path.lstat().st_ino == replacement_inode


def test_preflight_fails_closed_when_representation_is_replaced_after_final_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final representation path check cannot certify a replaced directory."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "post-final-representation"
    config = _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    representation = _write_authentic_retained_prepared_phase(work_root, config)
    original_representation = tmp_path / "post-final-original-representation"
    external_root = tmp_path / "post-final-external-root"
    external_representation = _write_authentic_retained_prepared_phase(
        external_root,
        config,
    )
    original_inventory = _work_tree_snapshot(representation)
    replacement_inventory = _work_tree_snapshot(external_representation)
    original_inode = representation.lstat().st_ino
    replacement_inode = external_representation.lstat().st_ino

    def replace_after_final_status() -> None:
        """Swap the name only after its final old-directory status is returned."""
        representation.rename(original_representation)
        external_representation.rename(representation)

    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / "runs-post-final-representation"

    with monkeypatch.context() as guarded_monkeypatch:
        _install_lstat_replacement_race(
            guarded_monkeypatch,
            representation,
            occurrence=2,
            replacement=replace_after_final_status,
        )
        _forbid_retained_payload_opening(
            guarded_monkeypatch,
            launcher,
            (representation, original_representation, external_representation),
        )
        _forbid_numerical_member_descriptor_opening(
            guarded_monkeypatch,
            (representation, original_representation, external_representation),
        )
        result = launcher.run_launcher(
            launcher.parse_launcher_command(
                _new_command_arguments(
                    tmp_path,
                    cache_directory=target,
                    analysis_root=analysis_root,
                )
            ),
            _preflight_only_dependencies(
                tmp_path,
                calls,
                terminal,
                label="post-final-representation",
            ),
        )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert _work_tree_snapshot(original_representation) == original_inventory
    assert original_representation.lstat().st_ino == original_inode
    assert _work_tree_snapshot(representation) == replacement_inventory
    assert representation.lstat().st_ino == replacement_inode


def test_preflight_fails_closed_when_representation_is_replaced_after_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid-looking representation cannot change after its descriptor closes."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "post-close-representation"
    config = _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    representation = _write_authentic_retained_prepared_phase(work_root, config)
    original_representation = tmp_path / "post-close-original-representation"
    replacement_work_root = tmp_path / "post-close-replacement-work-root"
    replacement_representation = _write_authentic_retained_prepared_phase(
        replacement_work_root,
        config,
    )
    assert replacement_representation.name == representation.name
    original_inventory = _work_tree_snapshot(representation)
    replacement_inventory = _work_tree_snapshot(replacement_representation)
    original_inode = representation.lstat().st_ino
    replacement_inode = replacement_representation.lstat().st_ino

    def replace_after_representation_close() -> None:
        """Replace the validated name only after its anchored fd is closed."""
        representation.rename(original_representation)
        replacement_representation.rename(representation)

    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / "runs-post-close-representation"
    with monkeypatch.context() as guarded_monkeypatch:
        race_state = _install_descriptor_close_replacement_race(
            guarded_monkeypatch,
            representation.lstat(),
            replace_after_representation_close,
        )
        _forbid_retained_payload_opening(
            guarded_monkeypatch,
            launcher,
            (representation, original_representation, replacement_representation),
        )
        _forbid_numerical_member_descriptor_opening(
            guarded_monkeypatch,
            (representation, original_representation, replacement_representation),
        )
        result = launcher.run_launcher(
            launcher.parse_launcher_command(
                _new_command_arguments(
                    tmp_path,
                    cache_directory=target,
                    analysis_root=analysis_root,
                )
            ),
            _preflight_only_dependencies(
                tmp_path,
                calls,
                terminal,
                label="post-close-representation",
            ),
        )

    assert race_state["triggered"] is True
    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert _work_tree_snapshot(original_representation) == original_inventory
    assert original_representation.lstat().st_ino == original_inode
    assert _work_tree_snapshot(representation) == replacement_inventory
    assert representation.lstat().st_ino == replacement_inode


def test_preflight_fails_closed_when_prepared_container_is_replaced_after_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prepared container cannot change after its descriptor closes."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "post-close-prepared"
    config = _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    representation = _write_authentic_retained_prepared_phase(work_root, config)
    prepared_path = representation.parent
    original_prepared = tmp_path / "post-close-original-prepared"
    replacement_prepared = tmp_path / "post-close-replacement-prepared"
    replacement_prepared.mkdir()
    original_inventory = _work_tree_snapshot(prepared_path)
    replacement_inventory = _work_tree_snapshot(replacement_prepared)
    original_inode = prepared_path.lstat().st_ino
    replacement_inode = replacement_prepared.lstat().st_ino

    def replace_after_prepared_close() -> None:
        """Replace the validated container after its anchored fd is closed."""
        prepared_path.rename(original_prepared)
        replacement_prepared.rename(prepared_path)

    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / "runs-post-close-prepared"
    with monkeypatch.context() as guarded_monkeypatch:
        race_state = _install_descriptor_close_replacement_race(
            guarded_monkeypatch,
            prepared_path.lstat(),
            replace_after_prepared_close,
        )
        _forbid_retained_payload_opening(
            guarded_monkeypatch,
            launcher,
            (representation, original_prepared / representation.name),
        )
        _forbid_numerical_member_descriptor_opening(
            guarded_monkeypatch,
            (representation, original_prepared / representation.name),
        )
        result = launcher.run_launcher(
            launcher.parse_launcher_command(
                _new_command_arguments(
                    tmp_path,
                    cache_directory=target,
                    analysis_root=analysis_root,
                )
            ),
            _preflight_only_dependencies(
                tmp_path,
                calls,
                terminal,
                label="post-close-prepared",
            ),
        )

    assert race_state["triggered"] is True
    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert _work_tree_snapshot(original_prepared) == original_inventory
    assert original_prepared.lstat().st_ino == original_inode
    assert _work_tree_snapshot(prepared_path) == replacement_inventory
    assert prepared_path.lstat().st_ino == replacement_inode


def test_preflight_fails_closed_when_empty_ppc_is_replaced_after_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty PPC container cannot change after its descriptor closes."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "post-close-ppc"
    _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    ppc_path = work_root / "ppc"
    ppc_path.mkdir(parents=True)
    original_ppc = tmp_path / "post-close-original-ppc"
    replacement_ppc = tmp_path / "post-close-replacement-ppc"
    replacement_ppc.mkdir()
    original_inventory = _work_tree_snapshot(ppc_path)
    replacement_inventory = _work_tree_snapshot(replacement_ppc)
    original_inode = ppc_path.lstat().st_ino
    replacement_inode = replacement_ppc.lstat().st_ino

    def replace_after_ppc_close() -> None:
        """Replace the validated empty container after its fd is closed."""
        ppc_path.rename(original_ppc)
        replacement_ppc.rename(ppc_path)

    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / "runs-post-close-ppc"
    with monkeypatch.context() as guarded_monkeypatch:
        race_state = _install_descriptor_close_replacement_race(
            guarded_monkeypatch,
            ppc_path.lstat(),
            replace_after_ppc_close,
        )
        result = launcher.run_launcher(
            launcher.parse_launcher_command(
                _new_command_arguments(
                    tmp_path,
                    cache_directory=target,
                    analysis_root=analysis_root,
                )
            ),
            _preflight_only_dependencies(
                tmp_path,
                calls,
                terminal,
                label="post-close-ppc",
            ),
        )

    assert race_state["triggered"] is True
    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert _work_tree_snapshot(original_ppc) == original_inventory
    assert original_ppc.lstat().st_ino == original_inode
    assert _work_tree_snapshot(ppc_path) == replacement_inventory
    assert ppc_path.lstat().st_ino == replacement_inode


def test_preflight_fails_closed_when_ppc_child_is_inserted_after_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty PPC container cannot gain a child after its descriptor closes."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / "post-close-ppc-child"
    _write_prerequisite_cache(tmp_path, target)
    work_root = target.parent / "lfp_summary_work"
    ppc_path = work_root / "ppc"
    ppc_path.mkdir(parents=True)
    ppc_inode = ppc_path.lstat().st_ino
    inserted_child = ppc_path / "late-child"
    inserted_bytes = b"late PPC child\n"

    def insert_child_after_ppc_close() -> None:
        """Create work only after the empty PPC descriptor has closed."""
        inserted_child.write_bytes(inserted_bytes)

    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / "runs-post-close-ppc-child"
    with monkeypatch.context() as guarded_monkeypatch:
        race_state = _install_descriptor_close_replacement_race(
            guarded_monkeypatch,
            ppc_path.lstat(),
            insert_child_after_ppc_close,
        )
        result = launcher.run_launcher(
            launcher.parse_launcher_command(
                _new_command_arguments(
                    tmp_path,
                    cache_directory=target,
                    analysis_root=analysis_root,
                )
            ),
            _preflight_only_dependencies(
                tmp_path,
                calls,
                terminal,
                label="post-close-ppc-child",
            ),
        )

    assert race_state["triggered"] is True
    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []
    assert ppc_path.lstat().st_ino == ppc_inode
    assert inserted_child.read_bytes() == inserted_bytes


@pytest.mark.parametrize("ppc_state", ("complete", "incomplete", "locked", "malformed"))
def test_preflight_rejects_any_retained_ppc_child_before_trial_loading(
    tmp_path: Path,
    ppc_state: str,
) -> None:
    """Any PPC child is active/resumable work and blocks a new launcher run."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"ppc-{ppc_state}-cache"
    _write_prerequisite_cache(tmp_path, target)
    run_fingerprint = "f" * 64
    run_directory = target.parent / "lfp_summary_work" / "ppc" / run_fingerprint
    run_directory.mkdir(parents=True)
    if ppc_state == "complete":
        (run_directory / "metadata.json").write_text(
            json.dumps({"run_fingerprint": run_fingerprint}) + "\n",
            encoding="ascii",
        )
        blocks = run_directory / "blocks"
        blocks.mkdir()
        (blocks / "summary.npz").write_bytes(b"synthetic PPC result\n")
        (blocks / "summary.complete.json").write_text(
            json.dumps(
                {"block_id": "summary", "run_fingerprint": run_fingerprint}
            )
            + "\n",
            encoding="ascii",
        )
    elif ppc_state == "incomplete":
        (run_directory / "metadata.json").write_text(
            json.dumps({"run_fingerprint": run_fingerprint}) + "\n",
            encoding="ascii",
        )
    elif ppc_state == "locked":
        (run_directory / "executor.lock").write_text("active\n", encoding="ascii")
    else:
        (run_directory / "metadata.json").write_text("{\n", encoding="ascii")
    calls: list[str] = []
    terminal: list[str] = []

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=tmp_path / f"runs-ppc-{ppc_state}",
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=ppc_state),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not (tmp_path / f"runs-ppc-{ppc_state}").exists()
    assert calls == []


@pytest.mark.parametrize("entry_kind", ("regular-file", "symlink"))
def test_preflight_rejects_non_directory_or_linked_ppc_child_before_trial_loading(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    """A PPC container must have no entries, including files and links."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"ppc-{entry_kind}-cache"
    _write_prerequisite_cache(tmp_path, target)
    ppc = target.parent / "lfp_summary_work" / "ppc"
    ppc.mkdir(parents=True)
    child = ppc / "unsafe-entry"
    if entry_kind == "regular-file":
        child.write_text("unexpected PPC entry\n", encoding="ascii")
    else:
        backing = tmp_path / "unsafe-ppc-entry-backing"
        backing.write_text("unexpected PPC entry\n", encoding="ascii")
        try:
            child.symlink_to(backing)
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-ppc-{entry_kind}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=entry_kind),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


def _unsafe_work_root(
    tmp_path: Path,
    target: Path,
    label: str,
) -> None:
    """Create one unsafe shared-work-root layout for preflight rejection."""
    work_root = target.parent / "lfp_summary_work"
    fingerprint = "c" * 64
    if label == "root-file":
        work_root.write_text("not a directory\n", encoding="ascii")
    elif label == "root-symlink":
        backing = tmp_path / "unsafe-root-backing"
        backing.mkdir()
        try:
            work_root.symlink_to(backing, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
    elif label == "prepared-container-symlink":
        work_root.mkdir()
        backing = tmp_path / "unsafe-prepared-backing"
        backing.mkdir()
        try:
            (work_root / "prepared_phase").symlink_to(
                backing,
                target_is_directory=True,
            )
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
    elif label == "ppc-container-symlink":
        work_root.mkdir()
        backing = tmp_path / "unsafe-ppc-backing"
        backing.mkdir()
        try:
            (work_root / "ppc").symlink_to(backing, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
    elif label == "representation-symlink":
        backing_root = tmp_path / "unsafe-representation-backing"
        backing = _write_retained_prepared_phase(backing_root, fingerprint)
        prepared = work_root / "prepared_phase"
        prepared.mkdir(parents=True)
        try:
            (prepared / fingerprint).symlink_to(backing, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
    elif label == "representation-file":
        prepared = work_root / "prepared_phase"
        prepared.mkdir(parents=True)
        (prepared / fingerprint).write_text("not a representation directory\n", encoding="ascii")
    elif label == "prepared-container-file":
        work_root.mkdir()
        (work_root / "prepared_phase").write_text("not a directory\n", encoding="ascii")
    elif label == "ppc-container-file":
        work_root.mkdir()
        (work_root / "ppc").write_text("not a directory\n", encoding="ascii")
    elif label == "prepared-lock":
        representation = _write_retained_prepared_phase(work_root, fingerprint)
        (representation / "writer.lock").write_text("active\n", encoding="ascii")
    elif label == "incomplete-representation":
        representation = _write_retained_prepared_phase(work_root, fingerprint)
        (representation / "valid.npy").unlink()
    elif label == "malformed-metadata":
        representation = _write_retained_prepared_phase(work_root, fingerprint)
        (representation / "metadata.json").write_text("{\n", encoding="ascii")
    elif label == "malformed-complete":
        representation = _write_retained_prepared_phase(work_root, fingerprint)
        (representation / "complete.json").write_text("{\n", encoding="ascii")
    elif label == "inconsistent-complete":
        representation = _write_retained_prepared_phase(work_root, fingerprint)
        (representation / "complete.json").write_text(
            json.dumps({"representation_fingerprint": "d" * 64}) + "\n",
            encoding="ascii",
        )
    elif label == "inconsistent-metadata":
        representation = _write_retained_prepared_phase(work_root, fingerprint)
        inconsistent_metadata = _prepared_phase_metadata(fingerprint)
        inconsistent_metadata["representation_fingerprint"] = "d" * 64
        (representation / "metadata.json").write_text(
            json.dumps(inconsistent_metadata, sort_keys=True) + "\n",
            encoding="ascii",
        )
    elif label == "unexpected-root-member":
        work_root.mkdir()
        (work_root / "leftover.txt").write_text("unexpected\n", encoding="ascii")
    elif label == "unexpected-prepared-member":
        prepared = work_root / "prepared_phase"
        prepared.mkdir(parents=True)
        (prepared / "leftover.txt").write_text("unexpected\n", encoding="ascii")
    elif label == "unexpected-representation-member":
        representation = _write_retained_prepared_phase(work_root, fingerprint)
        (representation / "leftover.txt").write_text("unexpected\n", encoding="ascii")
    elif label == "nonfingerprint-representation":
        _write_retained_prepared_phase(work_root, "not-a-fingerprint")
    else:
        raise AssertionError(f"unknown unsafe work layout: {label}")


@pytest.mark.parametrize(
    "label",
    (
        "root-file",
        "root-symlink",
        "prepared-container-symlink",
        "ppc-container-symlink",
        "representation-symlink",
        "representation-file",
        "prepared-container-file",
        "ppc-container-file",
        "prepared-lock",
        "incomplete-representation",
        "malformed-metadata",
        "malformed-complete",
        "inconsistent-complete",
        "inconsistent-metadata",
        "unexpected-root-member",
        "unexpected-prepared-member",
        "unexpected-representation-member",
        "nonfingerprint-representation",
    ),
)
def test_preflight_rejects_unsafe_retained_work_layout_before_trial_loading(
    tmp_path: Path,
    label: str,
) -> None:
    """Only inert complete prepared work may coexist with a new run."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"unsafe-work-{label}"
    _write_prerequisite_cache(tmp_path, target)
    _unsafe_work_root(tmp_path, target, label)
    calls: list[str] = []
    terminal: list[str] = []

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=tmp_path / f"runs-unsafe-work-{label}",
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=label),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not (tmp_path / f"runs-unsafe-work-{label}").exists()
    assert calls == []


@pytest.mark.parametrize(
    "member_name",
    ("metadata.json", "complete.json", "axes.npz", "valid.npy", "phase.npy"),
)
def test_preflight_rejects_missing_prepared_representation_member_before_trial_loading(
    tmp_path: Path,
    member_name: str,
) -> None:
    """Every retained representation member is required before reuse is considered."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"missing-retained-{member_name}"
    _write_prerequisite_cache(tmp_path, target)
    representation = _write_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        "c" * 64,
    )
    (representation / member_name).unlink()
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-missing-retained-{member_name}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=member_name),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


@pytest.mark.parametrize(
    "member_name",
    ("metadata.json", "complete.json", "axes.npz", "valid.npy", "phase.npy"),
)
def test_preflight_rejects_symlinked_prepared_representation_member_before_trial_loading(
    tmp_path: Path,
    member_name: str,
) -> None:
    """Retained representation files are regular files, never link aliases."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"symlink-retained-{member_name}"
    _write_prerequisite_cache(tmp_path, target)
    representation = _write_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        "c" * 64,
    )
    backing = tmp_path / f"symlinked-{member_name}"
    backing.write_bytes((representation / member_name).read_bytes())
    (representation / member_name).unlink()
    try:
        (representation / member_name).symlink_to(backing)
    except OSError as error:
        pytest.skip(f"test filesystem does not support symlinks: {error}")
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-symlink-retained-{member_name}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=member_name),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


@pytest.mark.parametrize(
    "member_name",
    ("metadata.json", "complete.json", "axes.npz", "valid.npy", "phase.npy"),
)
def test_preflight_rejects_directory_replacing_prepared_representation_member(
    tmp_path: Path,
    member_name: str,
) -> None:
    """Every expected retained representation member is a regular file."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"directory-retained-{member_name}"
    _write_prerequisite_cache(tmp_path, target)
    representation = _write_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        "c" * 64,
    )
    (representation / member_name).unlink()
    (representation / member_name).mkdir()
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-directory-retained-{member_name}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=member_name),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


@pytest.mark.parametrize("identity_name", ("metadata.json", "complete.json"))
def test_preflight_rejects_oversized_retained_identity_json_before_trial_loading(
    tmp_path: Path,
    identity_name: str,
) -> None:
    """Retained identity records have a bounded metadata-only read budget."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"oversized-{identity_name}"
    _write_prerequisite_cache(tmp_path, target)
    representation = _write_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        "c" * 64,
    )
    identity_path = representation / identity_name
    identity_path.write_bytes(
        identity_path.read_bytes().rstrip() + b" " * _MAX_RETAINED_IDENTITY_JSON_BYTES
    )
    assert identity_path.stat().st_size > _MAX_RETAINED_IDENTITY_JSON_BYTES
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-oversized-{identity_name}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=identity_name),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


@pytest.mark.parametrize("identity_name", ("metadata.json", "complete.json"))
@pytest.mark.parametrize(
    "record_case",
    ("json-list", "json-scalar", "missing-fingerprint", "wrong-fingerprint", "nonstring-fingerprint"),
)
def test_preflight_rejects_nonidentity_retained_json_before_trial_loading(
    tmp_path: Path,
    identity_name: str,
    record_case: str,
) -> None:
    """Both retained identity records must be mappings with the path fingerprint."""
    launcher = _launcher()
    target = (
        tmp_path
        / "CT026"
        / "processed"
        / f"invalid-{identity_name}-{record_case}"
    )
    _write_prerequisite_cache(tmp_path, target)
    fingerprint = "c" * 64
    representation = _write_retained_prepared_phase(
        target.parent / "lfp_summary_work",
        fingerprint,
    )
    record: object
    if record_case == "json-list":
        record = []
    elif record_case == "json-scalar":
        record = 1
    elif identity_name == "metadata.json":
        record = _prepared_phase_metadata(fingerprint)
        if record_case == "missing-fingerprint":
            del record["representation_fingerprint"]
        elif record_case == "wrong-fingerprint":
            record["representation_fingerprint"] = "d" * 64
        else:
            record["representation_fingerprint"] = 1
    else:
        record = {"representation_fingerprint": fingerprint}
        if record_case == "missing-fingerprint":
            record = {}
        elif record_case == "wrong-fingerprint":
            record = {"representation_fingerprint": "d" * 64}
        else:
            record = {"representation_fingerprint": 1}
    (representation / identity_name).write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="ascii",
    )
    calls: list[str] = []
    terminal: list[str] = []
    analysis_root = tmp_path / f"runs-invalid-{identity_name}-{record_case}"

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=analysis_root,
            )
        ),
        _preflight_only_dependencies(tmp_path, calls, terminal, label=record_case),
    )

    assert result.exit_code == 2 and result.run_directory is None
    assert not analysis_root.exists()
    assert calls == []


@pytest.mark.parametrize(
    "label",
    (
        "missing",
        "legacy",
        "outside",
        "nested",
        "path-alias",
        "symlink",
        "component-symlink",
        "processed-symlink",
        "stale",
        "running",
        "failed",
        "spike-phase",
        "unexpected-member",
        "unexpected-directory",
        "unexpected-symlink",
    ),
)
def test_preflight_rejects_unsafe_or_nonprerequisite_cache_before_trial_loading(
    tmp_path: Path,
    label: str,
) -> None:
    """Invalid cache targets cannot create a run or reach a trial/phase loader."""
    launcher = _launcher()
    processed = tmp_path / "CT026" / "processed"
    target = processed / f"{label}-cache"
    if label == "missing":
        pass
    elif label == "legacy":
        target = processed / "lfp_summary_cache"
        _write_prerequisite_cache(tmp_path, target)
    elif label == "outside":
        target = tmp_path / "outside-session" / "corrected-cache"
        _write_prerequisite_cache(tmp_path, target)
    elif label == "nested":
        target = processed / "nested" / "corrected-cache"
        _write_prerequisite_cache(tmp_path, target)
    elif label == "path-alias":
        real_target = processed / "path-alias-cache"
        _write_prerequisite_cache(tmp_path, real_target)
        target = processed / ".." / "processed" / "path-alias-cache"
    elif label == "symlink":
        backing = processed / "symlink-backing"
        _write_prerequisite_cache(tmp_path, backing)
        target = processed / "symlink-cache"
        try:
            target.symlink_to(backing, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
    elif label == "component-symlink":
        linked_parent = processed / "linked-parent"
        backing_parent = tmp_path / "outside-linked-parent"
        _write_prerequisite_cache(
            tmp_path,
            backing_parent / "corrected-cache",
        )
        processed.mkdir(parents=True)
        try:
            linked_parent.symlink_to(backing_parent, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
        target = linked_parent / "corrected-cache"
    elif label == "processed-symlink":
        backing_processed = tmp_path / "outside-processed"
        _write_prerequisite_cache(
            tmp_path,
            backing_processed / "corrected-cache",
        )
        processed.parent.mkdir(parents=True)
        try:
            processed.symlink_to(backing_processed, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
        target = processed / "corrected-cache"
    elif label == "stale":
        _write_prerequisite_cache(tmp_path, target)
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        manifest["components"]["power"]["configuration_fingerprint"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="ascii")
    elif label == "running":
        _write_prerequisite_cache(tmp_path, target, synchrony_state="running")
    elif label == "failed":
        _write_prerequisite_cache(tmp_path, target, power_state="failed")
    elif label == "spike-phase":
        _write_prerequisite_cache(tmp_path, target, include_spike_phase=True)
    elif label == "unexpected-member":
        _write_prerequisite_cache(tmp_path, target, unexpected_member="extra.txt")
    elif label == "unexpected-directory":
        _write_prerequisite_cache(tmp_path, target)
        (target / "extra-directory").mkdir()
    elif label == "unexpected-symlink":
        _write_prerequisite_cache(tmp_path, target)
        try:
            (target / "extra-link").symlink_to(target / "power.npz")
        except OSError as error:
            pytest.skip(f"test filesystem does not support symlinks: {error}")
    else:
        raise AssertionError(f"unknown unsafe cache fixture: {label}")
    calls: list[str] = []
    terminal: list[str] = []
    dependencies = replace(
        _dependencies(tmp_path, calls, terminal),
        load_trial_count=lambda _config: (_ for _ in ()).throw(
            AssertionError(f"{label} cache reached trial loading")
        ),
        compute_component=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"{label} cache reached numerical computation")
        ),
        publish_report=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"{label} cache reached report rendering")
        ),
    )
    command = launcher.parse_launcher_command(
        _new_command_arguments(
            tmp_path,
            cache_directory=target,
            analysis_root=tmp_path / f"runs-{label}",
        )
    )

    result = launcher.run_launcher(command, dependencies)

    assert result.exit_code == 2
    assert result.run_directory is None
    assert not (tmp_path / f"runs-{label}").exists()
    assert calls == []


def test_preflight_rejects_missing_component_members_before_trial_loading(
    tmp_path: Path,
) -> None:
    """Both completed Power and Synchrony archive members are mandatory."""
    launcher = _launcher()
    for component in ("power", "synchrony"):
        target = tmp_path / "CT026" / "processed" / f"missing-{component}"
        _write_prerequisite_cache(tmp_path, target)
        (target / f"{component}.npz").unlink()
        calls: list[str] = []
        terminal: list[str] = []
        dependencies = replace(
            _dependencies(tmp_path, calls, terminal),
            load_trial_count=lambda _config: (_ for _ in ()).throw(
                AssertionError("missing prerequisite reached trial loading")
            ),
        )
        result = launcher.run_launcher(
            launcher.parse_launcher_command(
                _new_command_arguments(
                    tmp_path,
                    cache_directory=target,
                    analysis_root=tmp_path / f"runs-missing-{component}",
                )
            ),
            dependencies,
        )

        assert result.exit_code == 2
        assert result.run_directory is None
        assert calls == []


@pytest.mark.parametrize("damage", ("truncated", "object-dtype"))
def test_preflight_rejects_unsafe_component_headers_before_trial_loading(
    tmp_path: Path,
    damage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupt or pickle-capable prerequisites fail before any numerical work."""
    launcher = _launcher()
    target = tmp_path / "CT026" / "processed" / f"unsafe-{damage}"
    _write_prerequisite_cache(tmp_path, target)
    power_path = target / "power.npz"
    if damage == "truncated":
        power_path.write_bytes(b"not an NPZ archive")
    else:
        np.savez(power_path, values=np.array(("unsafe",), dtype=object))
    calls: list[str] = []
    terminal: list[str] = []

    def forbid_array_loading(*_args: object, **_kwargs: object) -> object:
        """Fail if corrupt-cache preflight attempts a numerical array load."""
        raise AssertionError("corrupt cache preflight loaded an array")

    monkeypatch.setattr(np, "load", forbid_array_loading)
    monkeypatch.setattr(lfp_summary_io, "load_component_arrays", forbid_array_loading)
    monkeypatch.setattr(
        launcher,
        "load_component_arrays",
        forbid_array_loading,
        raising=False,
    )
    dependencies = replace(
        _dependencies(tmp_path, calls, terminal),
        load_trial_count=lambda _config: (_ for _ in ()).throw(
            AssertionError("unsafe cache reached trial loading")
        ),
    )

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=target,
                analysis_root=tmp_path / f"runs-unsafe-{damage}",
            )
        ),
        dependencies,
    )

    assert result.exit_code == 2
    assert result.run_directory is None
    assert calls == []


def test_hpc_wrapper_keeps_the_exact_eight_worker_preview_forwarding_boundary() -> None:
    """The reviewed Slurm wrapper remains a thin eight-CPU argument forwarder."""
    repository_root = Path(__file__).resolve().parents[3]
    wrapper = repository_root / "src" / "shell_scripts" / "hpc_ppc.sh"
    text = wrapper.read_text(encoding="ascii")

    assert "#SBATCH --cpus-per-task=8" in text
    assert '"$SLURM_CPUS_PER_TASK" -ne 8' in text
    assert '"$explicit_worker_count" -ne "$SLURM_CPUS_PER_TASK"' in text
    assert 'python -m src.neural_analysis.lfp_spike_phase_launcher "$@"' in text


def _wrapper_test_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Create a clean temporary checkout containing only the reviewed wrapper.

    The fake checkout exists solely to exercise argument forwarding. It does
    not import project code, create a cache, or submit a Slurm job.
    """
    repository = tmp_path / "wrapper-repository"
    wrapper = repository / "src" / "shell_scripts" / "hpc_ppc.sh"
    launcher = repository / "src" / "neural_analysis" / "lfp_spike_phase_launcher.py"
    wrapper.parent.mkdir(parents=True)
    launcher.parent.mkdir(parents=True)
    tracked_wrapper = Path(__file__).resolve().parents[3] / "src" / "shell_scripts" / "hpc_ppc.sh"
    shutil.copy2(tracked_wrapper, wrapper)
    launcher.write_text('"""Synthetic wrapper identity target."""\n', encoding="ascii")
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'launcher-wrapper-test'\nversion = '0.0.0'\n",
        encoding="ascii",
    )
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "launcher-wrapper@example.invalid"),
        ("git", "config", "user.name", "Launcher Wrapper Test"),
        ("git", "add", "."),
        ("git", "commit", "-q", "-m", "wrapper fixture"),
    ):
        subprocess.run(command, cwd=repository, check=True)
    return repository, wrapper


def _fake_uv_for_wrapper(tmp_path: Path) -> tuple[Path, Path]:
    """Create an executable fake ``uv`` that records its exact arguments."""
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    capture = tmp_path / "forwarded-argv.bin"
    executable = fake_bin / "uv"
    executable.write_text(
        """#!/bin/bash
set -eu
if [[ "${1-}" == "--version" ]]; then
    printf 'uv 0.test\\n'
    exit 0
fi
printf '%s\\0' "$@" > "$PPC_TEST_CAPTURE"
""",
        encoding="ascii",
    )
    executable.chmod(0o755)
    return fake_bin, capture


def test_hpc_wrapper_forwards_the_exact_corrected_cache_preview_argv(
    tmp_path: Path,
) -> None:
    """The reviewed wrapper preserves the bounded preview command verbatim."""
    repository, wrapper = _wrapper_test_repository(tmp_path)
    fake_bin, capture = _fake_uv_for_wrapper(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "PPC_TEST_CAPTURE": str(capture),
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_SUBMIT_DIR": str(repository),
        }
    )
    preview_arguments = (
        "new",
        "--session-path",
        "/cluster/CT026",
        "--cache-directory",
        "/cluster/CT026/processed/corrected-cache",
        "--probe",
        "ProbeB",
        "--shuffles",
        "100",
        "--workers",
        "8",
    )

    result = subprocess.run(
        ("bash", str(wrapper), *preview_arguments),
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    forwarded = capture.read_bytes().decode("ascii").rstrip("\0").split("\0")
    assert forwarded == [
        "run",
        "--frozen",
        "--no-sync",
        "--offline",
        "python",
        "-m",
        "src.neural_analysis.lfp_spike_phase_launcher",
        *preview_arguments,
    ]
    assert "--final-run" not in forwarded


def test_identity_and_resume_command_exist_before_interrupted_planning(
    tmp_path: Path,
) -> None:
    """A planning interruption leaves atomic state/log and an exact resume command."""
    launcher = _launcher()
    calls: list[str] = []
    terminal: list[str] = []
    cache_directory = tmp_path / "CT026" / "processed" / "interrupted-nondefault-cache"
    _write_prerequisite_cache(tmp_path, cache_directory)

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
        assert state["identity"]["cache_directory"] == str(
            cache_directory.resolve()
        )
        raise KeyboardInterrupt("synthetic planning interruption")

    dependencies = _dependencies(tmp_path, calls, terminal, compute=interrupt)
    command = launcher.parse_launcher_command(
        _new_command_arguments(
            tmp_path,
            cache_directory=cache_directory,
            analysis_root=tmp_path / "runs",
        )
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
        _new_command_arguments(
            tmp_path,
            probe="ProbeA",
            analysis_root=tmp_path / "runs",
        )
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
    cache_directory = tmp_path / "CT026" / "processed" / "resume-nondefault-cache"
    _write_prerequisite_cache(tmp_path, cache_directory)

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
        _new_command_arguments(
            tmp_path,
            cache_directory=cache_directory,
            analysis_root=tmp_path / "runs",
        )
    )
    failed = launcher.run_launcher(command, first_dependencies)
    assert failed.exit_code == 1 and failed.status == "cleanup_failed"
    saved_configuration = json.loads(
        (failed.run_directory / "configuration.json").read_text(encoding="ascii")
    )
    assert saved_configuration["output_directory"] == str(
        cache_directory.resolve()
    )

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
        _new_command_arguments(tmp_path, analysis_root=tmp_path / "runs")
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
            _new_command_arguments(tmp_path, analysis_root=tmp_path / "runs")
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


def test_resume_rejects_mixed_open_ephys_value_semantics_before_component_access(
    tmp_path: Path,
) -> None:
    """A corrected checkpoint cannot be resumed against an explicitly legacy source identity."""
    launcher = _launcher()
    terminal: list[str] = []
    first_dependencies = _dependencies(
        tmp_path,
        [],
        terminal,
        cleanup=lambda _target: (_ for _ in ()).throw(OSError("retain work for resume")),
    )
    first_dependencies = replace(
        first_dependencies,
        source_fingerprints=lambda _config: {
            "lfp.dat": {"value_semantics": "open_ephys_affine_uV_v1"}
        },
    )
    failed = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(tmp_path, analysis_root=tmp_path / "runs")
        ),
        first_dependencies,
    )
    assert failed.status == "cleanup_failed"

    resume_calls: list[str] = []
    resumed_dependencies = _dependencies(
        tmp_path,
        resume_calls,
        terminal,
        component_compatible=True,
    )
    resumed_dependencies = replace(
        resumed_dependencies,
        source_fingerprints=lambda _config: {
            # Live identities retain the historical absence; only receipt-validated
            # cache inspection renders that absence as "legacy-unscaled".
            "lfp.dat": {}
        },
    )

    resumed = launcher.run_launcher(
        launcher.parse_launcher_command(["resume", "--run-directory", str(failed.run_directory)]),
        resumed_dependencies,
    )

    assert resumed.exit_code == 2
    assert resume_calls == []


def test_live_launcher_lock_fails_without_mutating_owner_state(tmp_path: Path) -> None:
    """A competing resume cannot relabel or append to the lock owner's run."""
    launcher = _launcher()
    first_calls: list[str] = []
    terminal: list[str] = []
    failed = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(tmp_path, analysis_root=tmp_path / "runs")
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


def _component_complete_report_failure(
    tmp_path: Path,
    terminal: list[str],
) -> tuple[object, object, Path]:
    """Create one original-commit run that fails only during report rendering."""
    launcher = _launcher()
    calls: list[str] = []
    cache_directory = tmp_path / "CT026" / "processed" / "report-recovery-nondefault-cache"
    _write_prerequisite_cache(tmp_path, cache_directory)

    def fail_report(**_: object) -> object:
        """Fail after component publication and before report publication."""
        calls.append("report_failed")
        raise RuntimeError("synthetic report layout failure")

    dependencies = _dependencies(
        tmp_path,
        calls,
        terminal,
        report=fail_report,
    )
    command = launcher.parse_launcher_command(
        _new_command_arguments(
            tmp_path,
            cache_directory=cache_directory,
            analysis_root=tmp_path / "runs",
        )
    )
    failed = launcher.run_launcher(command, dependencies)
    state = json.loads((failed.run_directory / "launcher_state.json").read_text())
    target = Path(state["cleanup_request"][0]["run_directory"])

    assert failed.exit_code == 1 and failed.status == "failed"
    assert state["completed_stages"][-1] == "component_complete"
    assert state["report_directory"] is None
    assert state["identity"]["cache_directory"] == str(cache_directory.resolve())
    assert target.is_dir()
    return launcher, failed, target


def _report_recovery_dependencies(
    tmp_path: Path,
    calls: list[str],
    terminal: list[str],
    *,
    report: Callable[..., object] | None = None,
    cleanup: Callable[[object], None] | None = None,
    tracked_clean: bool = True,
    is_ancestor: bool = True,
) -> object:
    """Return later-commit seams that make every computation call fail."""
    base = _dependencies(
        tmp_path,
        calls,
        terminal,
        compute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("report recovery reached numerical computation")
        ),
        component_compatible=True,
        cleanup=cleanup,
        report=report,
    )
    values = {
        name: getattr(base, name)
        for name in base.__dataclass_fields__
    }
    values.update(
        repository_state=lambda: _launcher().RepositoryState(
            tmp_path / "repo",
            "c" * 40,
            tracked_clean,
        ),
        repository_commit_is_ancestor=lambda original, current: (
            is_ancestor and original == "b" * 40 and current == "c" * 40
        ),
    )
    return SimpleNamespace(**values)


def test_report_recovery_reuses_component_records_code_and_cleans_last(
    tmp_path: Path,
) -> None:
    """Explicit recovery renders under a descendant commit without recomputing."""
    terminal: list[str] = []
    launcher, failed, target = _component_complete_report_failure(tmp_path, terminal)
    calls: list[str] = []
    dependencies = _report_recovery_dependencies(tmp_path, calls, terminal)

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["recover-report", "--run-directory", str(failed.run_directory)]
        ),
        dependencies,
    )

    assert result.exit_code == 0 and result.status == "complete"
    assert calls == ["validate_component", "report", "cleanup_target"]
    assert not target.exists()
    state = json.loads((failed.run_directory / "launcher_state.json").read_text())
    recovery = json.loads((failed.run_directory / "report_recovery.json").read_text())
    assert state["identity"]["git_commit"] == "b" * 40
    assert state["measurements"]["component_reused"] is True
    assert state["measurements"]["report_git_commit"] == "c" * 40
    assert recovery["schema_version"] == "spike_phase_report_recovery.v1"
    assert recovery["status"] == "complete"
    assert recovery["computation_git_commit"] == "b" * 40
    assert recovery["report_git_commit"] == "c" * 40
    assert recovery["original_error"] == "synthetic report layout failure"


def test_report_recovery_failure_retains_exact_work_and_component_stage(
    tmp_path: Path,
) -> None:
    """A second report failure must remain retryable and preserve PPC work."""
    terminal: list[str] = []
    launcher, failed, target = _component_complete_report_failure(tmp_path, terminal)
    calls: list[str] = []

    def fail_again(**_: object) -> object:
        """Fail the later report implementation before publication."""
        calls.append("report_failed_again")
        raise RuntimeError("second synthetic report failure")

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["recover-report", "--run-directory", str(failed.run_directory)]
        ),
        _report_recovery_dependencies(
            tmp_path,
            calls,
            terminal,
            report=fail_again,
        ),
    )

    assert result.exit_code == 1 and result.status == "failed"
    assert calls == ["validate_component", "report_failed_again"]
    assert target.is_dir()
    state = json.loads((failed.run_directory / "launcher_state.json").read_text())
    recovery = json.loads((failed.run_directory / "report_recovery.json").read_text())
    assert state["completed_stages"][-1] == "component_complete"
    assert state["report_directory"] is None
    assert recovery["status"] == "failed"
    assert recovery["error"] == "second synthetic report failure"


def test_recovered_cleanup_resume_requires_exact_report_commit(
    tmp_path: Path,
) -> None:
    """Cleanup-only resume binds to report code while preserving compute identity."""
    terminal: list[str] = []
    launcher, failed, target = _component_complete_report_failure(tmp_path, terminal)
    recovery_calls: list[str] = []

    def fail_cleanup(_: object) -> None:
        """Leave the exact target after a successfully recovered report."""
        recovery_calls.append("cleanup_failed")
        raise OSError("synthetic recovered cleanup failure")

    recovered = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["recover-report", "--run-directory", str(failed.run_directory)]
        ),
        _report_recovery_dependencies(
            tmp_path,
            recovery_calls,
            terminal,
            cleanup=fail_cleanup,
        ),
    )
    assert recovered.status == "cleanup_failed" and target.is_dir()

    wrong_calls: list[str] = []
    wrong_commit = _dependencies(
        tmp_path,
        wrong_calls,
        terminal,
        component_compatible=True,
    )
    rejected = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["resume", "--run-directory", str(failed.run_directory)]
        ),
        wrong_commit,
    )
    assert rejected.exit_code == 2
    assert wrong_calls == []

    resume_calls: list[str] = []
    completed = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["resume", "--run-directory", str(failed.run_directory)]
        ),
        _report_recovery_dependencies(tmp_path, resume_calls, terminal),
    )
    assert completed.exit_code == 0 and completed.status == "complete"
    assert resume_calls == ["validate_component", "cleanup_target"]
    assert not target.exists()


@pytest.mark.parametrize(
    ("tracked_clean", "is_ancestor", "message"),
    (
        (False, True, "clean"),
        (True, False, "descend"),
    ),
)
def test_report_recovery_rejects_untrusted_report_checkout_before_component_access(
    tmp_path: Path,
    tracked_clean: bool,
    is_ancestor: bool,
    message: str,
) -> None:
    """Dirty or unrelated report code cannot inspect or mutate a completed component."""
    terminal: list[str] = []
    launcher, failed, target = _component_complete_report_failure(tmp_path, terminal)
    calls: list[str] = []
    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["recover-report", "--run-directory", str(failed.run_directory)]
        ),
        _report_recovery_dependencies(
            tmp_path,
            calls,
            terminal,
            tracked_clean=tracked_clean,
            is_ancestor=is_ancestor,
        ),
    )

    assert result.exit_code == 2
    assert message in json.loads(
        (failed.run_directory / "launcher_state.json").read_text()
    )["error"]
    assert calls == []
    assert target.is_dir()


def test_report_recovery_honors_existing_launcher_lock(tmp_path: Path) -> None:
    """A competing recovery cannot mutate the failed run owner's artifacts."""
    terminal: list[str] = []
    launcher, failed, _ = _component_complete_report_failure(tmp_path, terminal)
    state_path = failed.run_directory / "launcher_state.json"
    log_path = failed.run_directory / "run.log"
    before_state = state_path.read_bytes()
    before_log = log_path.read_bytes()
    with (failed.run_directory / "launcher.lock").open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        competing = launcher.run_launcher(
            launcher.parse_launcher_command(
                ["recover-report", "--run-directory", str(failed.run_directory)]
            ),
            _report_recovery_dependencies(tmp_path, [], terminal),
        )

    assert competing.exit_code == 1
    assert state_path.read_bytes() == before_state
    assert log_path.read_bytes() == before_log


def test_report_recovery_rejects_same_commit_and_precomponent_stage(
    tmp_path: Path,
) -> None:
    """Recovery is not an alias for ordinary resume or interrupted computation."""
    terminal: list[str] = []
    launcher, failed, _ = _component_complete_report_failure(tmp_path, terminal)
    same_commit = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["recover-report", "--run-directory", str(failed.run_directory)]
        ),
        _dependencies(tmp_path, [], terminal, component_compatible=True),
    )
    assert same_commit.exit_code == 2
    assert "ordinary resume" in json.loads(
        (failed.run_directory / "launcher_state.json").read_text()
    )["error"]

    def interrupt(*_: object, **__: object) -> ComponentRunResult:
        """Leave a second run before component completion."""
        raise KeyboardInterrupt("synthetic interruption")

    precomponent_root = tmp_path / "precomponent-root"
    interrupted = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                precomponent_root,
                analysis_root=precomponent_root / "other-runs",
            )
        ),
        _dependencies(precomponent_root, [], terminal, compute=interrupt),
    )
    assert interrupted.exit_code == 130
    assert interrupted.status == "interrupted"
    assert interrupted.run_directory is not None
    rejected = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["recover-report", "--run-directory", str(interrupted.run_directory)]
        ),
        _report_recovery_dependencies(precomponent_root, [], terminal),
    )
    assert rejected.exit_code == 2
    assert "component_complete" in json.loads(
        (interrupted.run_directory / "launcher_state.json").read_text()
    )["error"]


def test_report_recovery_rejects_changed_sources_and_incompatible_component(
    tmp_path: Path,
) -> None:
    """Recovery fails closed before plotting when data or cache identity changes."""
    terminal: list[str] = []
    launcher, failed, target = _component_complete_report_failure(tmp_path, terminal)
    source_calls: list[str] = []
    changed_sources = _report_recovery_dependencies(
        tmp_path,
        source_calls,
        terminal,
    )
    changed_sources.source_fingerprints = lambda _config: {
        "trial_table": {"size_bytes": 11}
    }
    source_result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["recover-report", "--run-directory", str(failed.run_directory)]
        ),
        changed_sources,
    )
    assert source_result.exit_code == 2
    assert source_calls == [] and target.is_dir()

    component_calls: list[str] = []
    incompatible = _report_recovery_dependencies(
        tmp_path,
        component_calls,
        terminal,
    )

    def reject_component(_: object) -> None:
        """Reject a missing or stale committed component before reporting."""
        component_calls.append("validate_component")
        raise ValueError("incompatible committed component")

    incompatible.validate_component = reject_component
    component_result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["recover-report", "--run-directory", str(failed.run_directory)]
        ),
        incompatible,
    )
    assert component_result.exit_code == 2
    assert component_calls == ["validate_component"]
    assert target.is_dir()


def _completed_launcher_run(
    tmp_path: Path,
    terminal: list[str],
) -> tuple[object, object, Path]:
    """Create one fully completed original-commit run and immutable report."""
    launcher = _launcher()
    cache_directory = tmp_path / "CT026" / "processed" / "rerender-nondefault-cache"
    _write_prerequisite_cache(tmp_path, cache_directory)
    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            _new_command_arguments(
                tmp_path,
                cache_directory=cache_directory,
                analysis_root=tmp_path / "runs",
            )
        ),
        _dependencies(tmp_path, [], terminal),
    )
    state = json.loads((result.run_directory / "launcher_state.json").read_text())
    report_directory = Path(state["report_directory"])
    assert result.status == "complete"
    assert state["identity"]["cache_directory"] == str(cache_directory.resolve())
    assert report_directory.is_dir()
    return launcher, result, report_directory


def _rerender_dependencies(
    tmp_path: Path,
    calls: list[str],
    terminal: list[str],
    *,
    report: Callable[..., object] | None = None,
    tracked_clean: bool = True,
    is_ancestor: bool = True,
) -> object:
    """Return later-commit report seams that forbid compute and cleanup."""

    def publish_rerender(**kwargs: object) -> object:
        """Publish one distinct synthetic report and capture its measurements."""
        calls.append("report")
        calls.append(
            f"active={kwargs['report_measurements'].active_worker_count}"
        )
        parent = Path(kwargs["run_parent"])
        report_directory = parent / "synthetic_rerendered_report"
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
            deferred_cleanup=None,
        )

    base = _dependencies(
        tmp_path,
        calls,
        terminal,
        compute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("report rerender reached numerical computation")
        ),
        component_compatible=True,
        cleanup=lambda _target: (_ for _ in ()).throw(
            AssertionError("report rerender reached cleanup")
        ),
        report=report or publish_rerender,
    )
    values = {name: getattr(base, name) for name in base.__dataclass_fields__}
    values.update(
        repository_state=lambda: _launcher().RepositoryState(
            tmp_path / "repo",
            "c" * 40,
            tracked_clean,
        ),
        repository_commit_is_ancestor=lambda original, current: (
            is_ancestor and original == "b" * 40 and current == "c" * 40
        ),
    )
    return SimpleNamespace(**values)


def test_completed_report_rerender_preserves_prior_report_and_never_computes_or_cleans(
    tmp_path: Path,
) -> None:
    """A clean descendant publishes and validates one new immutable report."""
    terminal: list[str] = []
    launcher, completed, original_report = _completed_launcher_run(tmp_path, terminal)
    calls: list[str] = []

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["rerender-report", "--run-directory", str(completed.run_directory)]
        ),
        _rerender_dependencies(tmp_path, calls, terminal),
    )

    assert result.exit_code == 0 and result.status == "complete"
    assert calls == ["validate_component", "report", "active=None"]
    assert original_report.is_dir()
    state = json.loads((completed.run_directory / "launcher_state.json").read_text())
    new_report = Path(state["report_directory"])
    assert new_report.is_dir() and new_report != original_report
    assert state["completed_stages"][-1] == "complete"
    assert state["active_status"] == "complete" and state["error"] is None
    assert state["measurements"]["maximum_child_process_count"] == 3
    assert "measured_active_worker_count" not in state["measurements"]
    audit = json.loads(
        (completed.run_directory / "report_rerender.json").read_text()
    )
    assert audit["schema_version"] == "spike_phase_report_rerender.v1"
    assert audit["status"] == "complete"
    assert audit["computation_git_commit"] == "b" * 40
    assert audit["previous_report_git_commit"] == "b" * 40
    assert audit["rerender_git_commit"] == "c" * 40
    assert audit["previous_report_directory"] == str(original_report)
    assert audit["new_report_directory"] == str(new_report)


def test_completed_report_rerender_failure_preserves_state_and_prior_report(
    tmp_path: Path,
) -> None:
    """A failed new publication records its audit without damaging completion."""
    terminal: list[str] = []
    launcher, completed, original_report = _completed_launcher_run(tmp_path, terminal)
    state_path = completed.run_directory / "launcher_state.json"
    before_state = state_path.read_bytes()
    calls: list[str] = []

    def fail_report(**_: object) -> object:
        """Fail the new immutable report before publication."""
        calls.append("report_failed")
        raise RuntimeError("synthetic rerender failure")

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["rerender-report", "--run-directory", str(completed.run_directory)]
        ),
        _rerender_dependencies(tmp_path, calls, terminal, report=fail_report),
    )

    assert result.exit_code == 1 and result.status == "failed"
    assert calls == ["validate_component", "report_failed"]
    assert state_path.read_bytes() == before_state
    assert original_report.is_dir()
    audit = json.loads(
        (completed.run_directory / "report_rerender.json").read_text()
    )
    assert audit["status"] == "failed"
    assert audit["error"] == "synthetic rerender failure"


@pytest.mark.parametrize(
    ("tracked_clean", "is_ancestor", "message"),
    (
        (False, True, "clean"),
        (True, False, "descend"),
    ),
)
def test_completed_report_rerender_rejects_untrusted_checkout_without_mutation(
    tmp_path: Path,
    tracked_clean: bool,
    is_ancestor: bool,
    message: str,
) -> None:
    """Dirty or unrelated report code cannot alter a completed launcher run."""
    terminal: list[str] = []
    launcher, completed, original_report = _completed_launcher_run(tmp_path, terminal)
    state_path = completed.run_directory / "launcher_state.json"
    before_state = state_path.read_bytes()
    calls: list[str] = []

    result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["rerender-report", "--run-directory", str(completed.run_directory)]
        ),
        _rerender_dependencies(
            tmp_path,
            calls,
            terminal,
            tracked_clean=tracked_clean,
            is_ancestor=is_ancestor,
        ),
    )

    assert result.exit_code == 2
    assert message in terminal[-1]
    assert calls == []
    assert state_path.read_bytes() == before_state
    assert original_report.is_dir()


def test_completed_report_rerender_rejects_same_commit_incomplete_run_and_lock(
    tmp_path: Path,
) -> None:
    """Rerender is neither an identical-report alias nor an incomplete resume."""
    terminal: list[str] = []
    launcher, completed, _ = _completed_launcher_run(tmp_path, terminal)
    state_path = completed.run_directory / "launcher_state.json"
    before_state = state_path.read_bytes()
    same = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["rerender-report", "--run-directory", str(completed.run_directory)]
        ),
        _dependencies(tmp_path, [], terminal, component_compatible=True),
    )
    assert same.exit_code == 2 and "newer" in terminal[-1]
    assert state_path.read_bytes() == before_state

    with (completed.run_directory / "launcher.lock").open("a+") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = launcher.run_launcher(
            launcher.parse_launcher_command(
                ["rerender-report", "--run-directory", str(completed.run_directory)]
            ),
            _rerender_dependencies(tmp_path, [], terminal),
        )
    assert locked.exit_code == 1
    assert state_path.read_bytes() == before_state

    _, incomplete, _ = _component_complete_report_failure(
        tmp_path / "incomplete",
        terminal,
    )
    incomplete_state = incomplete.run_directory / "launcher_state.json"
    before_incomplete = incomplete_state.read_bytes()
    rejected = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["rerender-report", "--run-directory", str(incomplete.run_directory)]
        ),
        _rerender_dependencies(tmp_path, [], terminal),
    )
    assert rejected.exit_code == 2 and "completed" in terminal[-1]
    assert incomplete_state.read_bytes() == before_incomplete


def test_completed_report_rerender_rejects_changed_sources_and_component(
    tmp_path: Path,
) -> None:
    """Rerender fails closed before publication when immutable inputs differ."""
    terminal: list[str] = []
    launcher, completed, original_report = _completed_launcher_run(tmp_path, terminal)
    state_path = completed.run_directory / "launcher_state.json"
    before_state = state_path.read_bytes()
    changed_calls: list[str] = []
    changed = _rerender_dependencies(tmp_path, changed_calls, terminal)
    changed.source_fingerprints = lambda _config: {
        "trial_table": {"size_bytes": 11}
    }

    changed_result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["rerender-report", "--run-directory", str(completed.run_directory)]
        ),
        changed,
    )

    assert changed_result.exit_code == 2
    assert changed_calls == []
    assert state_path.read_bytes() == before_state
    assert original_report.is_dir()

    component_calls: list[str] = []
    incompatible = _rerender_dependencies(tmp_path, component_calls, terminal)

    def reject_component(_: object) -> None:
        """Reject a stale component before any new report publication."""
        component_calls.append("validate_component")
        raise ValueError("incompatible committed component")

    incompatible.validate_component = reject_component
    component_result = launcher.run_launcher(
        launcher.parse_launcher_command(
            ["rerender-report", "--run-directory", str(completed.run_directory)]
        ),
        incompatible,
    )
    assert component_result.exit_code == 2
    assert component_calls == ["validate_component"]
    assert state_path.read_bytes() == before_state
    assert original_report.is_dir()
