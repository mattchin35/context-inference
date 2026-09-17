"""RED contracts for interruption-safe, nonpublishing CT026 PPC profiling runs."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from typing import Iterator

import pytest

from src.neural_analysis.lfp_summary_ct026_profile_runner import run_ct026_ppc_profile


def _phase_metrics(*, cache_hit: bool, elapsed_seconds: float) -> dict[str, object]:
    """Return complete scalar phase-preparation metrics in seconds and bytes."""
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": 256,
        "peak_memory_source": "child_rusage_self",
        "cache_hit": cache_hit,
    }


def test_ct026_profile_runner_persists_atomic_stages_and_scalar_report_only(
    tmp_path: Path,
) -> None:
    """A completed injected run records cold/warm/selection/jobs without publication.

    The runner accepts caller-owned configuration and identity fingerprints plus
    injected phase preparation, spike selection, and scalar profile functions.
    It writes ``state.json`` atomically after ``phase_cold``, ``phase_warm``,
    ``selection``, and each ordered 100-shuffle scenario (low, median, high,
    combined). The timestamped analysis directory contains scalar JSON and a
    Markdown report only: no component payload, final builder, or manifest.
    """
    calls: list[tuple[str, object]] = []
    phase_roots: list[Path] = []

    def prepare_phase(*, warm: bool, phase_work_root: Path) -> tuple[str, dict[str, object]]:
        """Return injected cold/warm phase tokens and scalar child metrics."""
        calls.append(("phase", warm))
        phase_roots.append(phase_work_root)
        phase = "warm-phase" if warm else "cold-phase"
        metrics = _phase_metrics(cache_hit=warm, elapsed_seconds=0.5 if warm else 2.0)
        return phase, metrics

    def select_spikes(phase: str) -> str:
        """Return an injected selected-spike token tied to warm prepared phase."""
        calls.append(("selection", phase))
        return "spikes"

    def profile_job(*, scenario: str, phase: str, spikes: str, shuffle_count: int, work_root: Path) -> dict[str, float]:
        """Return scalar synthetic metrics for one named 100-shuffle scenario."""
        calls.append(("job", scenario))
        assert phase == "warm-phase" and spikes == "spikes"
        assert shuffle_count == 100
        assert work_root.name == scenario
        return {"total_elapsed_seconds": 1.0, "peak_memory_bytes": 128.0}

    @contextmanager
    def acquire_run_lock(run_directory: Path, identity: dict[str, str]) -> Iterator[None]:
        """Require the resolved exact run directory and hold through report writes."""
        assert run_directory == run_directory.resolve()
        assert run_directory.is_dir()
        assert not (run_directory / "state.json").exists()
        assert identity["source_fingerprint"] == "source-a"
        calls.append(("lock", "enter"))
        try:
            yield
        finally:
            assert (run_directory / "state.json").is_file()
            assert (run_directory / "profile.json").is_file()
            assert (run_directory / "summary.md").is_file()
            calls.append(("lock", "exit"))

    result = run_ct026_ppc_profile(
        config={"session": "CT026"},
        config_fingerprint="config-a",
        source_fingerprint="source-a",
        git_fingerprint="git-a",
        analysis_root=tmp_path,
        prepare_phase=prepare_phase,
        select_spikes=select_spikes,
        profile_job=profile_job,
        acquire_run_lock=acquire_run_lock,
        timestamp_factory=lambda: "2026-09-09T12-00-00Z",
    )

    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["identity"] == {
        "config_fingerprint": "config-a", "source_fingerprint": "source-a", "git_fingerprint": "git-a",
    }
    assert state["completed_stages"] == [
        "phase_cold", "phase_warm", "selection", "low", "median", "high", "combined",
    ]
    assert state["phase_profiles"] == {
        "cold": _phase_metrics(cache_hit=False, elapsed_seconds=2.0),
        "warm": _phase_metrics(cache_hit=True, elapsed_seconds=0.5),
        "resume_warm": None,
    }
    profile = json.loads((result.run_directory / "profile.json").read_text(encoding="utf-8"))
    assert profile["phase_profiles"] == state["phase_profiles"]
    assert phase_roots == [
        result.run_directory / "work" / "phase",
        result.run_directory / "work" / "phase",
    ]
    assert calls[0] == ("lock", "enter")
    assert calls[-1] == ("lock", "exit")
    assert [call for call in calls if call[0] == "job"] == [
        ("job", "low"), ("job", "median"), ("job", "high"), ("job", "combined"),
    ]
    assert (result.run_directory / "profile.json").is_file()
    assert (result.run_directory / "summary.md").is_file()
    assert not list(result.run_directory.rglob("manifest.json"))
    assert not list(result.run_directory.rglob("component.json"))
    assert not list(result.run_directory.rglob("*.npz"))
    assert not list(result.run_directory.glob("*.tmp"))


def test_ct026_profile_runner_resumes_exact_identity_and_preserves_interrupted_work(
    tmp_path: Path,
) -> None:
    """Interrupted work remains, while exact resume rehydrates required inputs.

    The injected median scenario creates a work marker then raises. Its marker
    must remain. An exact-identity rerun reloads only the warm prepared phase,
    reconstructs spike selection, and resumes median/high/combined without cold
    phase work or the completed low scenario. A source, config, or Git
    fingerprint mismatch raises before invoking any injected operation.
    """
    first_calls: list[str] = []

    def prepare_phase(*, warm: bool, phase_work_root: Path) -> tuple[str, dict[str, object]]:
        """Record preparation and return a phase token plus scalar metrics."""
        assert phase_work_root.name == "phase"
        first_calls.append(f"phase:{warm}")
        elapsed = 0.5 if warm else 2.0
        return "phase", _phase_metrics(cache_hit=warm, elapsed_seconds=elapsed)

    def select_spikes(_: str) -> str:
        """Record selection after warm preparation."""
        first_calls.append("selection")
        return "spikes"

    def interrupted_job(*, scenario: str, work_root: Path, **_: object) -> dict[str, float]:
        """Complete low, then leave median work intact while raising interruption."""
        first_calls.append(scenario)
        work_root.mkdir(parents=True, exist_ok=True)
        (work_root / "checkpoint-marker").write_text("keep", encoding="ascii")
        if scenario == "median":
            raise KeyboardInterrupt("stop after median checkpoint")
        return {"total_elapsed_seconds": 1.0, "peak_memory_bytes": 1.0}

    with pytest.raises(KeyboardInterrupt, match="median"):
        run_ct026_ppc_profile(
            config={"session": "CT026"}, config_fingerprint="config-a",
            source_fingerprint="source-a", git_fingerprint="git-a", analysis_root=tmp_path,
            prepare_phase=prepare_phase, select_spikes=select_spikes,
            profile_job=interrupted_job, timestamp_factory=lambda: "2026-09-09T12-00-00Z",
            acquire_run_lock=lambda *_: nullcontext(),
        )
    run_directory = tmp_path / "ct026_ppc_profile_2026-09-09T12-00-00Z"
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["completed_stages"] == ["phase_cold", "phase_warm", "selection", "low"]
    assert (run_directory / "work" / "median" / "checkpoint-marker").is_file()

    resumed_calls: list[str] = []
    hydration_calls: list[str] = []

    def rehydrate_phase(
        *,
        warm: bool,
        phase_work_root: Path,
    ) -> tuple[str, dict[str, object]]:
        """Reload the prepared phase mmap while forbidding cold recomputation."""
        assert warm is True
        assert phase_work_root == run_directory / "work" / "phase"
        hydration_calls.append("phase:True")
        return "rehydrated-phase", _phase_metrics(cache_hit=True, elapsed_seconds=0.25)

    def reconstruct_spikes(phase: str) -> str:
        """Rebuild deterministic spike metadata from the rehydrated phase."""
        assert phase == "rehydrated-phase"
        hydration_calls.append("selection")
        return "rehydrated-spikes"

    def resumed_job(*, scenario: str, phase: str, spikes: str, **_: object) -> dict[str, float]:
        """Profile only unfinished scenarios using reconstructed inputs."""
        assert phase == "rehydrated-phase"
        assert spikes == "rehydrated-spikes"
        resumed_calls.append(scenario)
        return {"total_elapsed_seconds": 1.0, "peak_memory_bytes": 1.0}

    result = run_ct026_ppc_profile(
        config={"session": "CT026"}, config_fingerprint="config-a",
        source_fingerprint="source-a", git_fingerprint="git-a", analysis_root=tmp_path,
        run_directory=run_directory,
        prepare_phase=rehydrate_phase,
        select_spikes=reconstruct_spikes,
        profile_job=resumed_job,
        acquire_run_lock=lambda *_: nullcontext(),
    )
    assert result.run_directory == run_directory
    assert hydration_calls == ["phase:True", "selection"]
    assert resumed_calls == ["median", "high", "combined"]
    resumed_state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert resumed_state["phase_profiles"] == {
        "cold": _phase_metrics(cache_hit=False, elapsed_seconds=2.0),
        "warm": _phase_metrics(cache_hit=True, elapsed_seconds=0.5),
        "resume_warm": _phase_metrics(cache_hit=True, elapsed_seconds=0.25),
    }

    mismatched_lock_events: list[str] = []

    @contextmanager
    def acquire_mismatched_resume_lock(
        locked_directory: Path,
        identity: dict[str, str],
    ) -> Iterator[None]:
        """Prove saved identity is inspected only while the exact run is locked."""
        assert locked_directory == run_directory.resolve()
        assert identity["source_fingerprint"] == "changed"
        mismatched_lock_events.append("enter")
        try:
            yield
        finally:
            mismatched_lock_events.append("exit")

    with pytest.raises(ValueError, match="source_fingerprint"):
        run_ct026_ppc_profile(
            config={"session": "CT026"}, config_fingerprint="config-a",
            source_fingerprint="changed", git_fingerprint="git-a", analysis_root=tmp_path,
            run_directory=run_directory,
            prepare_phase=lambda **_: (_ for _ in ()).throw(AssertionError("called")),
            select_spikes=lambda _: (_ for _ in ()).throw(AssertionError("called")),
            profile_job=lambda **_: (_ for _ in ()).throw(AssertionError("called")),
            acquire_run_lock=acquire_mismatched_resume_lock,
        )
    assert mismatched_lock_events == ["enter", "exit"]


@pytest.mark.parametrize("corruption", ("missing", "mismatched"))
def test_ct026_profile_runner_rejects_invalid_persisted_phase_profiles(
    tmp_path: Path,
    corruption: str,
) -> None:
    """Resume rejects incomplete or cache-inconsistent phase metrics before callbacks."""
    run_ct026_ppc_profile(
        config={"session": "CT026"},
        config_fingerprint="config-a",
        source_fingerprint="source-a",
        git_fingerprint="git-a",
        analysis_root=tmp_path,
        prepare_phase=lambda *, warm, phase_work_root: (
            "phase",
            _phase_metrics(cache_hit=warm, elapsed_seconds=1.0),
        ),
        select_spikes=lambda _: "spikes",
        profile_job=lambda **_: {
            "total_elapsed_seconds": 1.0,
            "peak_memory_bytes": 1.0,
        },
        acquire_run_lock=lambda *_: nullcontext(),
        timestamp_factory=lambda: "2026-09-09T13-00-00Z",
    )
    run_directory = tmp_path / "ct026_ppc_profile_2026-09-09T13-00-00Z"
    state_path = run_directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if corruption == "missing":
        del state["phase_profiles"]["cold"]["peak_memory_source"]
    else:
        state["phase_profiles"]["warm"]["cache_hit"] = False
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="phase_profiles"):
        run_ct026_ppc_profile(
            config={"session": "CT026"},
            config_fingerprint="config-a",
            source_fingerprint="source-a",
            git_fingerprint="git-a",
            analysis_root=tmp_path,
            run_directory=run_directory,
            prepare_phase=lambda **_: (_ for _ in ()).throw(AssertionError("called")),
            select_spikes=lambda _: (_ for _ in ()).throw(AssertionError("called")),
            profile_job=lambda **_: (_ for _ in ()).throw(AssertionError("called")),
            acquire_run_lock=lambda *_: nullcontext(),
        )


def test_ct026_grouped_profile_document_records_schema_and_scalar_provenance(
    tmp_path: Path,
) -> None:
    """S6 persists an explicit grouped-profile schema without retaining arrays.

    The injected scenarios supply only scalar grouped-profile accounting and a
    deterministic grouped run fingerprint. The final report must state the
    profile-product schema/version and provenance separately from the runner
    identity, preserving the existing scalar state/resume contract while
    making grouped 100/1000-shuffle cost reports self-describing.
    """
    def prepare_phase(*, warm: bool, phase_work_root: Path) -> tuple[str, dict[str, object]]:
        """Return a caller token and complete scalar cold/warm metrics."""
        del phase_work_root
        return "phase", _phase_metrics(cache_hit=warm, elapsed_seconds=1.0)

    def grouped_profile_job(*, scenario: str, **_: object) -> dict[str, object]:
        """Return scalar-only grouped accounting for one runner scenario."""
        return {
            "schema_version": "grouped_ppc_profile_result.v1",
            "profile_kind": "grouped_serial_ppc",
            "run_fingerprint": "a" * 64,
            "geometry_build_seconds": 0.1,
            "observed_reduction_seconds": 0.2,
            "union_edge_reduction_seconds": 0.3,
            "shuffle_aggregation_seconds": 0.4,
            "null_summarization_seconds": 0.5,
            "representative_histogram_seconds": 0.6,
            "checkpoint_overhead_seconds": 0.7,
            "total_elapsed_seconds": 1.0,
            "throughput_scheduled_edge_per_second": 2400.0,
            "scheduled_edge_count": 2400,
            "independent_edge_count": 24,
            "unique_site_qualified_union_edge_count": 8,
            "edge_union_saturation": 0.5,
            "edge_reuse_ratio": 3.0,
            "planned_parent_private_bytes": 4096,
            "planned_worker_private_bytes": 2048,
            "shared_phase_mmap_bytes": 4096,
            "planned_aggregate_array_bytes": 8192,
            "measured_peak_process_rss_bytes": 6144,
            "measured_peak_aggregate_rss_bytes": 7168,
            "measured_peak_aggregate_pss_bytes": 6656,
            "measured_memory_source": "injected_rss_pss_sampler",
            "projection_100_scheduled_edge_count": 2400,
            "projection_100_independent_edge_count": 24,
            "projection_100_unique_site_qualified_union_edge_count": 8,
            "projection_100_edge_union_saturation": 0.5,
            "projection_100_edge_reuse_ratio": 3.0,
            "projection_1000_scheduled_edge_count": 24000,
            "projection_1000_independent_edge_count": 48,
            "projection_1000_unique_site_qualified_union_edge_count": 16,
            "projection_1000_edge_union_saturation": 1.0,
            "projection_1000_edge_reuse_ratio": 3.0,
        }

    result = run_ct026_ppc_profile(
        config={"session": "synthetic"},
        config_fingerprint="config-s6",
        source_fingerprint="source-s6",
        git_fingerprint="git-s6",
        analysis_root=tmp_path,
        prepare_phase=prepare_phase,
        select_spikes=lambda _: "spikes",
        profile_job=grouped_profile_job,
        acquire_run_lock=lambda *_: nullcontext(),
        timestamp_factory=lambda: "2026-09-16T12-00-00Z",
    )

    document = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert document["profile_schema_version"] == "ct026_grouped_ppc_profile.v1"
    assert document["profile_provenance"] == {
        "profile_kind": "grouped_serial_ppc",
        "scenario_metric_schema_version": "grouped_ppc_profile_result.v1",
        "projection_shuffle_counts": "100,1000",
    }
    assert document["identity"] == {
        "config_fingerprint": "config-s6",
        "source_fingerprint": "source-s6",
        "git_fingerprint": "git-s6",
    }
    assert all(
        not isinstance(value, (dict, list))
        for metrics in document["profiles"].values()
        for value in metrics.values()
    )
    assert set(document["profiles"]["low"]) == set(grouped_profile_job(scenario="low"))
    assert document["profiles"]["low"]["projection_100_scheduled_edge_count"] == 2400
    assert document["profiles"]["low"]["projection_1000_scheduled_edge_count"] == 24000
    assert document["profiles"]["low"]["scheduled_edge_count"] == document["profiles"]["low"]["projection_100_scheduled_edge_count"]
    assert (
        document["profiles"]["low"]["projection_1000_scheduled_edge_count"]
        == 10 * document["profiles"]["low"]["projection_100_scheduled_edge_count"]
    )
    assert document["profiles"]["low"]["projection_1000_independent_edge_count"] == 48
    assert document["profiles"]["low"]["projection_1000_unique_site_qualified_union_edge_count"] == 16

    for case_index, (field_name, invalid_value) in enumerate((
        ("schema_version", "other-schema"),
        ("profile_kind", "legacy_serial_ppc"),
        ("run_fingerprint", "g" * 64),
        ("run_fingerprint", "A" * 64),
        ("projection_1000_scheduled_edge_count", 999),
        ("projection_1000_edge_union_saturation", 1.1),
        ("projection_1000_edge_reuse_ratio", 2.0),
        ("throughput_scheduled_edge_per_second", 1.0),
    )):
        def mismatched_profile_job(*, scenario: str, **_: object) -> dict[str, object]:
            """Return one otherwise-valid grouped metric map with one mismatch."""
            metrics = grouped_profile_job(scenario=scenario)
            metrics[field_name] = invalid_value
            return metrics

        with pytest.raises(ValueError, match=field_name):
            run_ct026_ppc_profile(
                config={"session": "synthetic"},
                config_fingerprint="config-s6",
                source_fingerprint="source-s6",
                git_fingerprint="git-s6",
                analysis_root=tmp_path / f"invalid-{case_index}-{field_name}",
                prepare_phase=prepare_phase,
                select_spikes=lambda _: "spikes",
                profile_job=mismatched_profile_job,
                acquire_run_lock=lambda *_: nullcontext(),
                timestamp_factory=lambda: "2026-09-16T12-00-00Z",
            )

    for field_name, invalid_value in (
        ("edge_union_saturation", 1.1),
        ("edge_reuse_ratio", 2.0),
        ("unique_site_qualified_union_edge_count", 25),
    ):
        def paired_base_projection_corruption(
            *, scenario: str, **_: object
        ) -> dict[str, object]:
            """Corrupt matching base/100 fields so equality alone cannot reject it."""
            metrics = grouped_profile_job(scenario=scenario)
            metrics[field_name] = invalid_value
            metrics[f"projection_100_{field_name}"] = invalid_value
            return metrics

        with pytest.raises(ValueError, match=field_name):
            run_ct026_ppc_profile(
                config={"session": "synthetic"},
                config_fingerprint="config-s6",
                source_fingerprint="source-s6",
                git_fingerprint="git-s6",
                analysis_root=tmp_path / f"paired-invalid-{field_name}",
                prepare_phase=prepare_phase,
                select_spikes=lambda _: "spikes",
                profile_job=paired_base_projection_corruption,
                acquire_run_lock=lambda *_: nullcontext(),
                timestamp_factory=lambda: "2026-09-16T12-00-00Z",
            )
