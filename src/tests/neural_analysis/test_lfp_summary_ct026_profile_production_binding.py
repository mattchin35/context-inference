"""RED contracts for concrete, nonpublishing CT026 PPC profile bindings."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_ct026_profile_adapter as adapter
from src.neural_analysis.lfp_summary_models import UnitPopulationConfig, default_lfp_summary_config
from src.neural_analysis.lfp_summary_preparation import PreparedTrials, TrialRelativeSpikeTrains
from src.neural_analysis.lfp_summary_ppc_profile import RepresentativePPCProfileJob


def _job() -> RepresentativePPCProfileJob:
    """Return one scalar representative descriptor with exact stable identities."""
    return RepresentativePPCProfileJob(
        "condition-a", "PFC", "whole", (-2.0, 2.0), (10, 20), 2, 2, 3, 5,
        "ProbeB active", "ProbeB:1", "ProbeB:2", "ProbeB:3", 1, 2, 3, .125, .25, .375,
    )


def test_production_dependencies_bind_existing_ct026_population_and_runtime_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The factory uses existing metadata/loaders and fixes serial 100 shuffles.

    This test replaces every data-facing function: it neither opens CT026 nor
    computes phase/PPC.  In particular, sorter metadata's second return value
    and channel quality from the sorter parent are passed to the existing active
    population builder, then the existing preview-config builder receives that
    population.  Phase and spike preparation bind their established runtime
    loaders and a caller-owned shared work cache.
    """
    calls: list[tuple[str, object]] = []
    sorter = tmp_path / "ephys/derived/probeB/kilosort4"
    population = SimpleNamespace(stable_unit_ids=("ProbeB:1",))
    base = default_lfp_summary_config()

    monkeypatch.setattr(adapter.spike_behavior_pynapple, "load_sorter_metadata", lambda path: ("clusters", {"table": path}))
    monkeypatch.setattr(adapter.unit_spike_loading, "load_channel_quality", lambda path: calls.append(("quality", path)) or {"channels": path})
    monkeypatch.setattr(adapter, "_ct026_probe_b_sorter", lambda _session: sorter)
    def build_population(session: Path, cluster_loader: object, channel_loader: object) -> object:
        """Exercise the established loader-callable population-builder contract."""
        clusters = cluster_loader(sorter)
        channels = channel_loader(sorter)
        calls.append(("population", (session, clusters, channels)))
        return population

    monkeypatch.setattr(adapter, "build_ct026_default_active_population", build_population)
    monkeypatch.setattr(adapter, "build_ct026_spike_phase_preview_config", lambda session, selected: calls.append(("config", (session, selected))) or base)
    monkeypatch.setattr(adapter.lfp_summary_runtime, "prepare_phase_run", lambda config, loader, **kwargs: calls.append(("phase", (config, loader, kwargs))) or "phase")
    monkeypatch.setattr(adapter.lfp_summary_runtime, "prepare_spike_run", lambda config, phase, loader: calls.append(("spikes", (config, phase, loader))) or "spikes")

    dependencies = adapter.make_production_ct026_profile_dependencies(work_cache_root=tmp_path / "shared-work")
    loaded_population = dependencies.load_active_population(tmp_path / "CT026")
    config = dependencies.build_config(tmp_path / "CT026", loaded_population)
    dependencies.prepare_phase(config, work_cache_root=tmp_path / "run-work")
    dependencies.prepare_spikes(config, "phase")

    assert calls[:2] == [
        ("quality", sorter.parent),
        ("population", (tmp_path / "CT026", {"table": sorter}, {"channels": sorter.parent})),
    ]
    assert calls[2] == ("config", (tmp_path / "CT026", population))
    assert config.ppc.shuffle_count == 100
    assert config.ppc_execution.worker_count == 1
    assert calls[3][1][1] is adapter.lfp_summary_runtime.load_configured_trial_table
    assert calls[3][1][2]["work_cache_root"] == tmp_path / "run-work"
    assert calls[4][1][2] is adapter.lfp_summary_runtime.load_configured_unit_spikes


def test_recovery_only_targets_exact_runtime_locks_and_git_ignores_untracked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Recovery delegates exact runtime paths; Git identity excludes unrelated files."""
    recovered: list[Path] = []
    monkeypatch.setattr(adapter.lfp_summary_work_cache, "recover_stale_lock", lambda path, fingerprint, **_: recovered.append(path))
    (tmp_path / "ppc/run-a").mkdir(parents=True)
    (tmp_path / "ppc/run-b").mkdir()
    for directory in (tmp_path / "ppc/run-a", tmp_path / "ppc/run-b"):
        (directory / "executor.lock").write_text("owned", encoding="ascii")
        (directory / "writer.lock").write_text("owned", encoding="ascii")
    adapter.recover_ct026_profile_work(work_root=tmp_path, identity={"config_fingerprint": "c", "source_fingerprint": "s", "git_fingerprint": "g"})
    assert recovered == [
        tmp_path / "ppc/run-a/executor.lock", tmp_path / "ppc/run-a/writer.lock",
        tmp_path / "ppc/run-b/executor.lock", tmp_path / "ppc/run-b/writer.lock",
    ]
    first = adapter.production_git_fingerprint(repository_root=tmp_path, head_reader=lambda: "abc", tracked_source_reader=lambda: {"src/neural_analysis/a.py": "x"})
    same = adapter.production_git_fingerprint(repository_root=tmp_path, head_reader=lambda: "abc", tracked_source_reader=lambda: {"src/neural_analysis/a.py": "x"})
    changed_head = adapter.production_git_fingerprint(repository_root=tmp_path, head_reader=lambda: "def", tracked_source_reader=lambda: {"src/neural_analysis/a.py": "x"})
    changed_source = adapter.production_git_fingerprint(repository_root=tmp_path, head_reader=lambda: "abc", tracked_source_reader=lambda: {"src/neural_analysis/a.py": "y"})
    assert first == same
    assert first != changed_head and first != changed_source


def test_recovery_rejects_symlinked_fingerprint_directories(tmp_path: Path) -> None:
    """Recovery never follows a fingerprint-directory symlink outside work."""
    external = tmp_path / "external"
    external.mkdir()
    external_lock = external / "executor.lock"
    external_lock.write_text("owned", encoding="ascii")
    ppc_root = tmp_path / "work" / "ppc"
    ppc_root.mkdir(parents=True)
    (ppc_root / "linked-run").symlink_to(external, target_is_directory=True)

    adapter.recover_ct026_profile_work(
        work_root=tmp_path / "work",
        identity={
            "config_fingerprint": "c",
            "source_fingerprint": "s",
            "git_fingerprint": "g",
        },
    )

    assert external_lock.read_text(encoding="ascii") == "owned"


def test_default_child_launcher_runs_fresh_profile_worker_without_final_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The default child path is fresh-process scalar profiling, not a raising stub."""
    job = SimpleNamespace(config="config", prepared_phase="phase", prepared_spikes="spikes")
    monkeypatch.setattr(adapter, "_profile_child_worker", lambda payload: {"total_elapsed_seconds": 1.0, "ru_maxrss": 4, "ru_maxrss_unit": "KiB", "child_pid": 999, "edge_call_count": 2})
    monkeypatch.setattr(adapter.os, "getpid", lambda: 111)
    metrics = adapter.run_isolated_ct026_profile_job(job=job, shuffle_count=100, work_root=tmp_path / "work")
    assert metrics["peak_memory_bytes"] == 4096
    assert metrics["child_pid"] != 111
    assert metrics["edge_call_count"] == 2
    assert not list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("manifest.json"))


def test_child_memory_sample_reports_process_rss_without_inventing_aggregate_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux ``ru_maxrss`` is child-process RSS, not an aggregate measurement."""
    monkeypatch.setattr(adapter, "_self_rss_bytes", lambda: 4096)

    sample = adapter._child_memory_sample()

    assert sample == {
        "peak_process_rss_bytes": 4096,
        "peak_aggregate_rss_bytes": None,
        "peak_aggregate_pss_bytes": None,
        "memory_source": (
            "resource.getrusage(RUSAGE_SELF).ru_maxrss_kib; "
            "aggregate_rss_pss_unavailable"
        ),
    }


def test_profile_child_binds_grouped_serial_profiler_and_forwards_only_scalars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The S6 child uses the production grouped profiler, never the legacy job path.

    This is a fully injected child boundary: real runtime validators accept the
    tiny complete prepared phase/spike records, while no CT026 path is opened.
    The grouped profiler returns every public scalar field; a second invocation
    proves that a private plan is rejected rather than silently filtered.
    """
    defaults = default_lfp_summary_config()
    config = replace(
        defaults,
        unit_population=UnitPopulationConfig(
            label="synthetic-profile-population",
            probe_label="PFC",
            sorter_path=None,
            aligned_spike_path=None,
            selected_channels=(0,),
            quality_settings=(),
            stable_unit_ids=("PFC:1", "PFC:2", "PFC:3"),
        ),
        phase=replace(defaults.phase, frequency_hz=(8.0, 40.0)),
        ppc=replace(defaults.ppc, shuffle_count=100),
        ppc_execution=replace(defaults.ppc_execution, worker_count=1),
    )
    trial_indices = np.arange(1000, 1260, dtype=np.int64)
    trial_count = trial_indices.size
    selected_trial_indices = trial_indices[:249]
    time_s = np.array([-2.0, 0.0], dtype=float)
    prepared_trials = PreparedTrials(
        condition_names=("condition-a", "condition-b"),
        condition_membership=np.column_stack(
            (
                np.arange(trial_count) < 249,
                np.arange(trial_count) >= 249,
            )
        ),
        filter_membership=np.ones(trial_count, dtype=bool),
        user_excluded=np.zeros(trial_count, dtype=bool),
        objective_valid=np.ones(trial_count, dtype=bool),
        objective_exclusion_reason=np.full(trial_count, "", dtype="<U1"),
        user_exclusion_reason=np.full(trial_count, "", dtype="<U1"),
        site_validity={
            site.stable_id: (
                np.arange(trial_count) < 255 if site.stable_id == "PFC" else np.ones(trial_count, dtype=bool)
            )
            for site in config.sites
        },
        pair_validity={pair: np.ones(trial_count, dtype=bool) for pair in config.site_pairs},
    )
    prepared_phase = adapter.lfp_summary_runtime.PreparedPhaseRun(
        trial_indices=trial_indices,
        alignment_times_s=np.arange(trial_count, dtype=float),
        prepared_trials=prepared_trials,
        phase_tensor=np.ones((len(config.sites), 2, trial_count, 2), dtype=np.complex64),
        phase_valid=np.ones((len(config.sites), 2, trial_count, 2), dtype=bool),
        relative_time_s=time_s,
        site_valid=np.vstack(
            (
                np.arange(trial_count) < 255,
                np.ones(trial_count, dtype=bool),
                np.ones(trial_count, dtype=bool),
            )
        ),
        pair_valid=np.ones((len(config.site_pairs), trial_count), dtype=bool),
        source_trace=np.zeros((len(config.sites), trial_count, 2), dtype=float),
    )
    prepared_spikes = adapter.lfp_summary_runtime.PreparedSpikeRun(
        unit_ids=("PFC:1", "PFC:2", "PFC:3"),
        population_ids=("synthetic-profile-population",),
        trial_spike_trains=tuple(
            TrialRelativeSpikeTrains(
                unit_id=unit_id,
                relative_spike_times=tuple(
                    np.array([-1.0, 1.0]) for _ in range(trial_count)
                ),
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            )
            for unit_id in ("PFC:1", "PFC:2", "PFC:3")
        ),
    )
    adapter.lfp_summary_runtime._validate_prepared_phase_run(config, prepared_phase)
    adapter.lfp_summary_runtime._validate_prepared_spike_run(
        config, prepared_phase, prepared_spikes
    )
    profile_job = RepresentativePPCProfileJob(
        condition_name="condition-a",
        site_id="PFC",
        epoch_name="whole",
        epoch_bounds_s=(-2.0, 2.0),
        trial_indices=tuple(int(value) for value in selected_trial_indices),
        trial_count=249,
        site_valid_trial_count=249,
        unit_count=3,
        frequency_count=2,
        population_id="synthetic-profile-population",
        low_unit_id="PFC:1",
        median_unit_id="PFC:2",
        high_unit_id="PFC:3",
        low_spike_count=498,
        median_spike_count=498,
        high_spike_count=498,
        low_spike_rate_hz=0.5,
        median_spike_rate_hz=0.5,
        high_spike_rate_hz=0.5,
    )
    job = adapter.slice_ct026_profile_job(
        config=config,
        profile_job=profile_job,
        scenario="median",
        unit_ids=("PFC:2",),
        phase=prepared_phase,
        spikes=prepared_spikes,
    )
    assert job.prepared_phase.trial_indices.tolist() == selected_trial_indices.tolist()
    assert job.prepared_phase.phase_tensor.shape == (len(config.sites), 2, 249, 2)
    assert job.prepared_spikes.unit_ids == ("PFC:2",)
    assert job.config.unit_population.stable_unit_ids == ("PFC:2",)
    adapter.lfp_summary_runtime._validate_prepared_phase_run(job.config, job.prepared_phase)
    adapter.lfp_summary_runtime._validate_prepared_spike_run(
        job.config, job.prepared_phase, job.prepared_spikes
    )
    expected_shared_phase_mmap_bytes = (
        job.prepared_phase.phase_tensor.nbytes
        + job.prepared_phase.phase_valid.nbytes
    )
    calls: list[dict[str, object]] = []

    def grouped_profile(**kwargs: object) -> object:
        """Derive every union scalar from the supplied real base/projection plans."""
        calls.append(kwargs)
        base_plan = kwargs["component_plan"]
        projections = dict(kwargs["projection_plans"])
        plan_100 = projections[100]
        plan_1000 = projections[1000]
        return SimpleNamespace(
            schema_version="grouped_ppc_profile_result.v1",
            profile_kind="grouped_serial_ppc",
            run_fingerprint="e" * 64,
            geometry_build_seconds=0.1,
            observed_reduction_seconds=0.2,
            union_edge_reduction_seconds=0.3,
            shuffle_aggregation_seconds=0.4,
            null_summarization_seconds=0.5,
            representative_histogram_seconds=0.6,
            checkpoint_overhead_seconds=0.7,
            total_elapsed_seconds=1.5,
            throughput_scheduled_edge_per_second=base_plan.scheduled_edge_count / 1.5,
            scheduled_edge_count=base_plan.scheduled_edge_count,
            independent_edge_count=base_plan.independent_edge_count,
            unique_site_qualified_union_edge_count=base_plan.union_edge_count,
            edge_union_saturation=base_plan.edge_union_saturation,
            edge_reuse_ratio=base_plan.edge_reuse_ratio,
            planned_parent_private_bytes=base_plan.allocation_estimate.planned_parent_private_bytes,
            planned_worker_private_bytes=base_plan.allocation_estimate.planned_worker_private_bytes,
            shared_phase_mmap_bytes=base_plan.allocation_estimate.shared_phase_mmap_bytes,
            planned_aggregate_array_bytes=base_plan.allocation_estimate.planned_aggregate_array_bytes,
            measured_peak_process_rss_bytes=6144,
            measured_peak_aggregate_rss_bytes=None,
            measured_peak_aggregate_pss_bytes=None,
            measured_memory_source="resource.getrusage(RUSAGE_SELF).ru_maxrss_kib; aggregate_rss_pss_unavailable",
            projection_100_scheduled_edge_count=plan_100.scheduled_edge_count,
            projection_100_independent_edge_count=plan_100.independent_edge_count,
            projection_100_unique_site_qualified_union_edge_count=plan_100.union_edge_count,
            projection_100_edge_union_saturation=plan_100.edge_union_saturation,
            projection_100_edge_reuse_ratio=plan_100.edge_reuse_ratio,
            projection_1000_scheduled_edge_count=plan_1000.scheduled_edge_count,
            projection_1000_independent_edge_count=plan_1000.independent_edge_count,
            projection_1000_unique_site_qualified_union_edge_count=plan_1000.union_edge_count,
            projection_1000_edge_union_saturation=plan_1000.edge_union_saturation,
            projection_1000_edge_reuse_ratio=plan_1000.edge_reuse_ratio,
        )

    monkeypatch.setattr(adapter, "profile_grouped_ppc_component", grouped_profile)
    monkeypatch.setattr(
        adapter,
        "profile_production_ppc_job",
        lambda **_: (_ for _ in ()).throw(AssertionError("legacy profiler called")),
    )

    metrics = adapter._profile_child_worker(
        {"job": job, "shuffle_count": 100, "work_root": tmp_path / "work"}
    )

    assert len(calls) == 1
    assert calls[0]["config"] is job.config
    assert calls[0]["execution"] is job.execution
    assert calls[0]["prepared_phase"] is job.prepared_phase
    assert calls[0]["prepared_spikes"] is job.prepared_spikes
    assert calls[0]["work_root"] == tmp_path / "work"
    assert "schedule" not in calls[0]
    base_plan = calls[0]["component_plan"]
    assert tuple(count for count, _plan in calls[0]["projection_plans"]) == (100, 1000)
    assert base_plan.allocation_estimate.shared_phase_mmap_bytes == expected_shared_phase_mmap_bytes
    assert calls[0]["projection_plans"][0][1].allocation_estimate.shared_phase_mmap_bytes == expected_shared_phase_mmap_bytes
    assert metrics["schema_version"] == "grouped_ppc_profile_result.v1"
    assert metrics["profile_kind"] == "grouped_serial_ppc"
    assert metrics["run_fingerprint"] == "e" * 64
    assert metrics["scheduled_edge_count"] == base_plan.scheduled_edge_count
    assert metrics["independent_edge_count"] == base_plan.independent_edge_count
    assert metrics["unique_site_qualified_union_edge_count"] == base_plan.union_edge_count
    assert metrics["projection_100_scheduled_edge_count"] == calls[0]["projection_plans"][0][1].scheduled_edge_count
    assert metrics["projection_1000_scheduled_edge_count"] == calls[0]["projection_plans"][1][1].scheduled_edge_count
    assert metrics["measured_peak_aggregate_rss_bytes"] is None
    assert metrics["measured_peak_aggregate_pss_bytes"] is None
    assert metrics["scheduled_edge_count"] == metrics["projection_100_scheduled_edge_count"]
    assert metrics["projection_1000_scheduled_edge_count"] == 10 * metrics["scheduled_edge_count"]
    assert set(metrics) == {
        "schema_version", "profile_kind", "run_fingerprint",
        "geometry_build_seconds", "observed_reduction_seconds",
        "union_edge_reduction_seconds", "shuffle_aggregation_seconds",
        "null_summarization_seconds", "representative_histogram_seconds",
        "checkpoint_overhead_seconds", "total_elapsed_seconds",
        "throughput_scheduled_edge_per_second", "scheduled_edge_count",
        "independent_edge_count", "unique_site_qualified_union_edge_count",
        "edge_union_saturation", "edge_reuse_ratio",
        "planned_parent_private_bytes", "planned_worker_private_bytes",
        "shared_phase_mmap_bytes", "planned_aggregate_array_bytes",
        "measured_peak_process_rss_bytes", "measured_peak_aggregate_rss_bytes",
        "measured_peak_aggregate_pss_bytes", "measured_memory_source",
        "projection_100_scheduled_edge_count", "projection_100_independent_edge_count",
        "projection_100_unique_site_qualified_union_edge_count",
        "projection_100_edge_union_saturation", "projection_100_edge_reuse_ratio",
        "projection_1000_scheduled_edge_count", "projection_1000_independent_edge_count",
        "projection_1000_unique_site_qualified_union_edge_count",
        "projection_1000_edge_union_saturation", "projection_1000_edge_reuse_ratio",
        "ru_maxrss", "ru_maxrss_unit", "child_pid",
    }
    assert all(not isinstance(value, np.ndarray) for value in metrics.values())

    def grouped_profile_with_private_plan(**kwargs: object) -> object:
        """Attempt to cross the child boundary with a private plan object."""
        return SimpleNamespace(**vars(grouped_profile(**kwargs)), private_plan=object())

    monkeypatch.setattr(adapter, "profile_grouped_ppc_component", grouped_profile_with_private_plan)
    with pytest.raises(ValueError, match="scalar"):
        adapter._profile_child_worker(
            {"job": job, "shuffle_count": 100, "work_root": tmp_path / "work"}
        )


def test_default_child_launcher_bounds_timeout_and_handles_pipe_eof() -> None:
    """A hung or abruptly exited child cannot hang the resumable parent run."""
    started = time.monotonic()
    timed_out = adapter._default_child_launcher(
        lambda _: (time.sleep(10.0) or {}),
        {},
        timeout_seconds=0.01,
    )
    assert time.monotonic() - started < 2.0
    assert timed_out["exit_code"] != 0
    assert "timeout" in timed_out["error"]

    exited = adapter._default_child_launcher(
        lambda _: os._exit(7),
        {},
        timeout_seconds=1.0,
    )
    assert exited["exit_code"] != 0


def test_production_run_lock_does_not_steal_another_live_local_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Production liveness delegates to OS PID checks rather than self-PID equality."""
    seen: list[int] = []
    monkeypatch.setattr(adapter.os, "kill", lambda pid, _signal: seen.append(pid))
    assert adapter._production_process_exists(4242) is True
    assert seen == [4242]


def test_production_dependency_lock_accepts_runner_positional_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The concrete lock seam accepts the runner's two positional arguments."""
    captured: list[tuple[Path, dict[str, str]]] = []
    monkeypatch.setattr(
        adapter,
        "acquire_ct026_profile_run_lock",
        lambda directory, identity, **_: (
            captured.append((directory, identity)) or nullcontext()
        ),
    )
    dependencies = adapter.make_production_ct026_profile_dependencies(
        work_cache_root=tmp_path / "unused-default"
    )
    run_directory = tmp_path / "ct026_ppc_profile_test"
    run_directory.mkdir()
    identity = {
        "config_fingerprint": "config-id",
        "source_fingerprint": "source-id",
        "git_fingerprint": "git-id",
    }

    with dependencies.acquire_run_lock(run_directory, identity):
        pass

    assert captured == [(run_directory, identity)]


def test_production_phase_uses_callback_work_root_and_slice_rejects_nonexact_trials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Cold/warm cache location is runner-owned and production slice requires 249 rows."""
    captured: list[Path] = []
    monkeypatch.setattr(adapter, "build_ct026_default_active_population", lambda *_: "population")
    monkeypatch.setattr(adapter, "build_ct026_spike_phase_preview_config", lambda *_: default_lfp_summary_config())
    monkeypatch.setattr(adapter.spike_behavior_pynapple, "load_sorter_metadata", lambda *_: (None, None))
    monkeypatch.setattr(adapter.unit_spike_loading, "load_channel_quality", lambda *_: None)
    monkeypatch.setattr(adapter.lfp_summary_runtime, "prepare_phase_run", lambda *_args, **kwargs: captured.append(kwargs["work_cache_root"]) or "phase")
    dependencies = adapter.make_production_ct026_profile_dependencies(work_cache_root=tmp_path / "default")
    dependencies.prepare_phase(default_lfp_summary_config(), work_cache_root=tmp_path / "run-specific-phase")
    assert captured == [tmp_path / "run-specific-phase"]

    job = _job()
    phase = SimpleNamespace(trial_indices=np.array([10, 20], dtype=np.int64), site_valid=np.ones((3, 2), dtype=bool), phase_tensor=np.ones((3, 5, 2, 2), dtype=np.complex64), phase_valid=np.ones((3, 5, 2, 2), dtype=bool), prepared_trials=SimpleNamespace(condition_names=("condition-a",), condition_membership=np.ones((2, 1), dtype=bool)), relative_time_s=np.array([-2., 0.]))
    spikes = SimpleNamespace(unit_ids=("ProbeB:1",), population_ids=("ProbeB active",), trial_spike_trains=(SimpleNamespace(unit_id="ProbeB:1", relative_spike_times=(np.array([0.]), np.array([0.])), overlap_trial_indices=np.array([], dtype=np.int64)),))
    with pytest.raises(ValueError, match="249"):
        adapter.slice_ct026_profile_job(config=default_lfp_summary_config(), profile_job=job, scenario="low", unit_ids=("ProbeB:1",), phase=phase, spikes=spikes)


def test_local_process_exists_handles_permission_and_dead_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """OS liveness treats permission as live and ProcessLookup as dead."""
    monkeypatch.setattr(adapter.os, "kill", lambda *_: (_ for _ in ()).throw(PermissionError()))
    assert adapter._local_process_exists(7) is True
    monkeypatch.setattr(adapter.os, "kill", lambda *_: (_ for _ in ()).throw(ProcessLookupError()))
    assert adapter._local_process_exists(7) is False
