"""RED contracts for interruption-safe, nonpublishing CT026 PPC profiling runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.neural_analysis.lfp_summary_ct026_profile_runner import run_ct026_ppc_profile


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

    def prepare_phase(*, warm: bool) -> str:
        """Return injected cold/warm phase tokens without a real transform."""
        calls.append(("phase", warm))
        return "warm-phase" if warm else "cold-phase"

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

    result = run_ct026_ppc_profile(
        config={"session": "CT026"},
        config_fingerprint="config-a",
        source_fingerprint="source-a",
        git_fingerprint="git-a",
        analysis_root=tmp_path,
        prepare_phase=prepare_phase,
        select_spikes=select_spikes,
        profile_job=profile_job,
        timestamp_factory=lambda: "2026-09-09T12-00-00Z",
    )

    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["identity"] == {
        "config_fingerprint": "config-a", "source_fingerprint": "source-a", "git_fingerprint": "git-a",
    }
    assert state["completed_stages"] == [
        "phase_cold", "phase_warm", "selection", "low", "median", "high", "combined",
    ]
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
    """An interrupted scenario leaves work/state and an exact rerun skips completed stages.

    The injected median scenario creates a work marker then raises. Its marker
    must remain. An exact-identity rerun resumes only median/high/combined,
    rather than redoing phase preparation, selection, or low. A source, config,
    or Git fingerprint mismatch raises before invoking any injected operation.
    """
    first_calls: list[str] = []

    def prepare_phase(*, warm: bool) -> str:
        """Record phase preparation and return a deterministic token."""
        first_calls.append(f"phase:{warm}")
        return "phase"

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
        )
    run_directory = tmp_path / "ct026_ppc_profile_2026-09-09T12-00-00Z"
    state = json.loads((run_directory / "state.json").read_text(encoding="utf-8"))
    assert state["completed_stages"] == ["phase_cold", "phase_warm", "selection", "low"]
    assert (run_directory / "work" / "median" / "checkpoint-marker").is_file()

    resumed_calls: list[str] = []
    result = run_ct026_ppc_profile(
        config={"session": "CT026"}, config_fingerprint="config-a",
        source_fingerprint="source-a", git_fingerprint="git-a", analysis_root=tmp_path,
        run_directory=run_directory,
        prepare_phase=lambda **_: (_ for _ in ()).throw(AssertionError("phase repeated")),
        select_spikes=lambda _: (_ for _ in ()).throw(AssertionError("selection repeated")),
        profile_job=lambda *, scenario, **_: resumed_calls.append(scenario) or {"total_elapsed_seconds": 1.0, "peak_memory_bytes": 1.0},
    )
    assert result.run_directory == run_directory
    assert resumed_calls == ["median", "high", "combined"]

    with pytest.raises(ValueError, match="source_fingerprint"):
        run_ct026_ppc_profile(
            config={"session": "CT026"}, config_fingerprint="config-a",
            source_fingerprint="changed", git_fingerprint="git-a", analysis_root=tmp_path,
            run_directory=run_directory,
            prepare_phase=lambda **_: (_ for _ in ()).throw(AssertionError("called")),
            select_spikes=lambda _: (_ for _ in ()).throw(AssertionError("called")),
            profile_job=lambda **_: (_ for _ in ()).throw(AssertionError("called")),
        )
