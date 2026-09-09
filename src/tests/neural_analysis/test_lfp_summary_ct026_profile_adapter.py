"""RED contracts for the production-bound, work-only CT026 PPC profiler."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.neural_analysis.lfp_summary_ct026_profile_adapter import (
    CT026PPCProfileAdapterDependencies,
    run_ct026_ppc_profile_adapter,
)
from src.neural_analysis.lfp_summary_ct026_profile_locks import acquire_ct026_profile_run_lock
from src.neural_analysis.lfp_summary_ppc_profile import RepresentativePPCProfileJob


def _representative_job() -> RepresentativePPCProfileJob:
    """Return scalar CT026 selection metadata without scientific arrays."""
    return RepresentativePPCProfileJob(
        condition_name="right_rewarded",
        site_id="PFC",
        epoch_name="whole",
        epoch_bounds_s=(-2.0, 2.0),
        trial_indices=tuple(range(1000, 1249)),
        trial_count=249,
        site_valid_trial_count=247,
        unit_count=6,
        frequency_count=50,
        population_id="CT026 ProbeB active",
        low_unit_id="ProbeB:11",
        median_unit_id="ProbeB:22",
        high_unit_id="ProbeB:33",
        low_spike_count=100,
        median_spike_count=500,
        high_spike_count=900,
        low_spike_rate_hz=0.1,
        median_spike_rate_hz=0.5,
        high_spike_rate_hz=0.9,
    )


def test_ct026_adapter_binds_cold_warm_selection_and_isolated_ordered_jobs(
    tmp_path: Path,
) -> None:
    """The adapter delegates orchestration while preserving exact job identity.

    No real CT026 inputs are opened.  The injected generic runner exercises the
    adapter's callbacks in its documented cold, warm, selection, then ordered
    low/median/high/combined order.  The adapter must use one phase work path
    for cache-backed cold and warm preparation, prepare spikes only from warm
    mmap-backed phase, construct job slices using the exact stable trial IDs,
    and send every scenario through the injected isolated-child seam.  Scalar
    phase preparation timings are distinct from each child PPC executor timing.
    """
    calls: list[tuple[str, object]] = []
    clock_values = iter((10.0, 13.0, 20.0, 22.5))
    representative = _representative_job()

    def build_config(session_path: Path, population: object) -> object:
        calls.append(("config", (session_path, population)))
        return "ct026-config"

    def load_population(session_path: Path) -> object:
        calls.append(("population", session_path))
        return "probe-b-population"

    def prepare_phase(config: object, *, work_cache_root: Path) -> object:
        calls.append(("phase", (config, work_cache_root)))
        return f"phase-{len([call for call in calls if call[0] == 'phase'])}"

    def prepare_spikes(config: object, phase: object) -> object:
        calls.append(("spikes", (config, phase)))
        return "prepared-spikes"

    def select_job(config: object, phase: object, spikes: object) -> RepresentativePPCProfileJob:
        calls.append(("select", (config, phase, spikes)))
        return representative

    def slice_job(
        *, config: object, profile_job: RepresentativePPCProfileJob, scenario: str,
        unit_ids: tuple[str, ...], phase: object, spikes: object,
    ) -> object:
        calls.append(("slice", (config, scenario, unit_ids, profile_job, phase, spikes)))
        return SimpleNamespace(
            scenario=scenario,
            trial_indices=profile_job.trial_indices,
            unit_ids=unit_ids,
        )

    def run_child(*, job: object, shuffle_count: int, work_root: Path) -> dict[str, object]:
        calls.append(("child", (job.scenario, shuffle_count, work_root, job.trial_indices)))
        return {
            "executor_total_elapsed_seconds": 4.0,
            "executor_peak_memory_bytes": 8192,
            "ru_maxrss": 8,
            "ru_maxrss_unit": "KiB",
        }

    @contextmanager
    def acquire_run_lock(*, analysis_root: Path, identity: dict[str, str], run_directory: Path | None):
        calls.append(("run-lock-enter", (analysis_root, identity, run_directory)))
        try:
            yield
        finally:
            calls.append(("run-lock-exit", None))

    def generic_runner(**kwargs: object) -> object:
        calls.append(("generic", kwargs))
        assert calls[-2][0] == "run-lock-enter"
        prepare = kwargs["prepare_phase"]
        profile = kwargs["profile_job"]
        phase_cold = prepare(warm=False)
        phase_warm = prepare(warm=True)
        spikes = kwargs["select_spikes"](phase_warm)
        assert phase_cold != phase_warm
        for scenario in ("low", "median", "high", "combined"):
            metrics = profile(
                scenario=scenario,
                phase=phase_warm,
                spikes=spikes,
                shuffle_count=100,
                work_root=tmp_path / "generic-work" / scenario,
            )
            assert metrics["phase_cold_seconds"] == 3.0
            assert metrics["phase_warm_seconds"] == 2.5
            assert metrics["peak_memory_source"] == "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss_kib"
            assert metrics["config_fingerprint"] == "config-id"
            assert metrics["source_fingerprint"] == "source-id"
            assert metrics["git_fingerprint"] == "commit:abc123+dirty-tree:sha256:deadbeef"
        return SimpleNamespace(profiles={})

    dependencies = CT026PPCProfileAdapterDependencies(
        build_config=build_config,
        load_active_population=load_population,
        prepare_phase=prepare_phase,
        prepare_spikes=prepare_spikes,
        select_profile_job=select_job,
        slice_profile_job=slice_job,
        run_isolated_profile_job=run_child,
        run_profile=generic_runner,
        config_fingerprint=lambda _: "config-id",
        source_fingerprint=lambda _: "source-id",
        git_fingerprint=lambda: "commit:abc123+dirty-tree:sha256:deadbeef",
        monotonic_seconds=lambda: next(clock_values),
        acquire_run_lock=acquire_run_lock,
        recover_profile_work=lambda **_: (_ for _ in ()).throw(AssertionError("not resuming")),
    )
    result = run_ct026_ppc_profile_adapter(
        session_path=tmp_path / "CT026_2026-08-01_130853",
        analysis_root=tmp_path / "analysis-runs",
        dependencies=dependencies,
    )

    assert result.profiles == {}
    generic_kwargs = next(value for name, value in calls if name == "generic")
    assert generic_kwargs["config"] == "ct026-config"
    assert generic_kwargs["config_fingerprint"] == "config-id"
    assert generic_kwargs["source_fingerprint"] == "source-id"
    assert generic_kwargs["git_fingerprint"] == "commit:abc123+dirty-tree:sha256:deadbeef"
    assert calls[-1] == ("run-lock-exit", None)
    phase_paths = [value[1] for name, value in calls if name == "phase"]
    assert phase_paths[0] == phase_paths[1]
    assert phase_paths[0].name == "phase"
    assert [value[1] for name, value in calls if name == "slice"] == [
        "low", "median", "high", "combined",
    ]
    expected_units = {
        "low": ("ProbeB:11",),
        "median": ("ProbeB:22",),
        "high": ("ProbeB:33",),
        "combined": ("ProbeB:11", "ProbeB:22", "ProbeB:33"),
    }
    for _name, value in [call for call in calls if call[0] == "slice"]:
        assert value[0] == "ct026-config"
        assert value[2] == expected_units[value[1]]
        assert value[3] is representative
        assert value[4] == "phase-2"
        assert value[5] == "prepared-spikes"
    for _name, value in [call for call in calls if call[0] == "child"]:
        assert value[1] == 100
        assert value[3] == representative.trial_indices
        assert value[2].name in {"low", "median", "high", "combined"}
        assert "manifest" not in str(value[2])


def test_ct026_adapter_forwards_exact_resume_and_never_constructs_preview_artifacts(
    tmp_path: Path,
) -> None:
    """Resume is owned by the generic runner, with no final-cache escape hatch.

    The adapter supplies an existing run directory unchanged to the generic
    interruption-safe runner.  It provides only preparation, selection, and
    isolated profiling callbacks: neither a final payload builder nor a
    manifest writer is part of the adapter dependency contract.
    """
    captured: dict[str, object] = {}

    events: list[str] = []

    @contextmanager
    def acquire_run_lock(**_: object):
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")

    def recover_profile_work(*, work_root: Path, **_: object) -> None:
        assert work_root.name == "low"
        events.append("recover")

    def generic_runner(**kwargs: object) -> object:
        captured.update(kwargs)
        metrics = kwargs["profile_job"](
            scenario="low", phase="warm", spikes="spikes", shuffle_count=100,
            work_root=tmp_path / "analysis-runs" / "ct026_ppc_profile_2026-09-09T12-00-00Z" / "work" / "low",
        )
        assert metrics["total_elapsed_seconds"] == 1.0
        return SimpleNamespace(profiles={"low": {"total_elapsed_seconds": 1.0}})

    dependencies = CT026PPCProfileAdapterDependencies(
        build_config=lambda _session, _population: "config",
        load_active_population=lambda _session: "population",
        prepare_phase=lambda _config, **_: "phase",
        prepare_spikes=lambda _config, _phase: "spikes",
        select_profile_job=lambda _config, _phase, _spikes: _representative_job(),
        slice_profile_job=lambda **_: "slice",
        run_isolated_profile_job=lambda **_: (events.append("child") or {"total_elapsed_seconds": 1.0}),
        run_profile=generic_runner,
        config_fingerprint=lambda _: "config-id",
        source_fingerprint=lambda _: "source-id",
        git_fingerprint=lambda: "git-id",
        monotonic_seconds=lambda: 0.0,
        acquire_run_lock=acquire_run_lock,
        recover_profile_work=recover_profile_work,
    )
    resume_directory = tmp_path / "analysis-runs" / "ct026_ppc_profile_2026-09-09T12-00-00Z"
    result = run_ct026_ppc_profile_adapter(
        session_path=tmp_path / "CT026_2026-08-01_130853",
        analysis_root=tmp_path / "analysis-runs",
        dependencies=dependencies,
        run_directory=resume_directory,
    )

    assert result.profiles == {"low": {"total_elapsed_seconds": 1.0}}
    assert captured["run_directory"] == resume_directory
    assert captured["analysis_root"] == tmp_path / "analysis-runs"
    assert captured["config_fingerprint"] == "config-id"
    assert events == ["lock-enter", "recover", "child", "lock-exit"]
    assert "write_component_transaction" not in captured
    assert "build_spike_phase_payload" not in captured


def test_ct026_profile_run_lock_preserves_live_foreign_and_mismatched_locks(
    tmp_path: Path,
) -> None:
    """Only a validated dead same-host exact lock is recoverable on resume."""
    run_directory = tmp_path / "ct026_ppc_profile_2026-09-09T12-00-00Z"
    run_directory.mkdir()
    identity = {
        "config_fingerprint": "config-id",
        "source_fingerprint": "source-id",
        "git_fingerprint": "git-id",
    }
    lock_path = run_directory / ".ct026-profile.lock"
    with acquire_ct026_profile_run_lock(
        run_directory, identity, pid=42, hostname="host-a", process_exists=lambda pid: pid == 42,
    ):
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        assert metadata == {"hostname": "host-a", "identity": identity, "pid": 42}

    for owner, expected_error in (
        ({"hostname": "host-a", "identity": identity, "pid": 42}, "live"),
        ({"hostname": "host-b", "identity": identity, "pid": 42}, "foreign"),
        ({"hostname": "host-a", "identity": {**identity, "git_fingerprint": "other"}, "pid": 42}, "identity"),
    ):
        lock_path.write_text(json.dumps(owner), encoding="ascii")
        error_type = ValueError if expected_error == "identity" else RuntimeError
        with pytest.raises(error_type, match=expected_error):
            with acquire_ct026_profile_run_lock(
                run_directory, identity, pid=42, hostname="host-a",
                process_exists=lambda _pid: expected_error == "live",
            ):
                pass
        assert lock_path.is_file()

    lock_path.write_text(
        json.dumps({"hostname": "host-a", "identity": identity, "pid": 42}), encoding="ascii",
    )
    with acquire_ct026_profile_run_lock(
        run_directory, identity, pid=42, hostname="host-a", process_exists=lambda _pid: False,
    ):
        assert lock_path.is_file()
    assert not lock_path.exists()
