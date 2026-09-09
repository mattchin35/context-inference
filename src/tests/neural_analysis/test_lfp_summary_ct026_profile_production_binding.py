"""RED contracts for concrete, nonpublishing CT026 PPC profile bindings."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_ct026_profile_adapter as adapter
from src.neural_analysis.lfp_summary_models import default_lfp_summary_config
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
        channels = channel_loader(sorter.parent)
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
    assert calls[3][1][2]["work_cache_root"] == tmp_path / "shared-work"
    assert calls[4][1][2] is adapter.lfp_summary_runtime.load_configured_unit_spikes


def test_slice_and_isolated_child_are_exact_scalar_work_only_seams(tmp_path: Path) -> None:
    """Slices preserve trials/site/whole epoch/overlap and child returns scalars only."""
    job = _job()
    phase = SimpleNamespace(
        trial_indices=np.array([10, 20, 30], dtype=np.int64),
        site_valid=np.array([[True, True, False]]),
        phase_tensor=np.ones((1, 5, 3, 2), dtype=np.complex64),
        phase_valid=np.ones((1, 5, 3, 2), dtype=bool),
    )
    spikes = SimpleNamespace(
        unit_ids=("ProbeB:1", "ProbeB:2", "ProbeB:3"),
        trial_spike_trains=tuple(
            SimpleNamespace(unit_id=unit, relative_spike_times=(np.array([-2., 0., 2.]),) * 3, overlap_trial_indices=np.array([20, 30], dtype=np.int64))
            for unit in ("ProbeB:1", "ProbeB:2", "ProbeB:3")
        ),
    )
    config = default_lfp_summary_config()
    sliced = adapter.slice_ct026_profile_job(config=config, profile_job=job, scenario="combined", unit_ids=("ProbeB:1", "ProbeB:2", "ProbeB:3"), phase=phase, spikes=spikes)
    assert sliced.scenario == "combined" and sliced.unit_ids == ("ProbeB:1", "ProbeB:2", "ProbeB:3")
    assert sliced.trial_indices == (10, 20) and sliced.site_id == "PFC" and sliced.epoch_bounds_s == (-2.0, 2.0)
    assert sliced.overlap_trial_indices == (20,)
    assert sliced.phase_tensor.shape == (1, 5, 2, 2)
    expected_seed = adapter.lfp_summary_runtime._ppc_schedule_seed(config, 0, 0, 0)
    expected_schedule = adapter.lfp_summary_runtime._shared_derangement_schedule(2, 100, expected_seed)
    np.testing.assert_array_equal(sliced.schedule, expected_schedule)

    launches: list[object] = []
    metrics = adapter.run_isolated_ct026_profile_job(
        job=sliced, shuffle_count=100, work_root=tmp_path / "work",
        process_launcher=lambda target, payload: launches.append((target, payload)) or {"total_elapsed_seconds": 2.0, "ru_maxrss": 7, "ru_maxrss_unit": "KiB"},
    )
    assert launches and metrics == {"total_elapsed_seconds": 2.0, "peak_memory_bytes": 7168, "peak_memory_source": "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss_kib"}
    with pytest.raises(RuntimeError, match="exit 2"):
        adapter.run_isolated_ct026_profile_job(job=sliced, shuffle_count=100, work_root=tmp_path / "work", process_launcher=lambda *_: {"exit_code": 2, "error": "child failed"})


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
