"""Explicit-run and readiness contracts for the communication subspace view."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.neural_analysis.communication_subspace import preparation
from src.neural_analysis.communication_subspace import run_session as runner
from src.neural_analysis.communication_subspace import results
from src.neural_analysis.communication_subspace import plotting


def _render(metadata_path: str) -> None:
    """Render the isolated view for an existing metadata JSON path."""
    import streamlit as st

    from src.neural_analysis.webapp.communication_subspace_view import (
        render_communication_subspace_view,
    )
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    render_communication_subspace_view(st, load_webapp_session(metadata_path))


def _assign_probe_roles(app_test) -> None:
    """Select distinct metadata probes for PFC and HPC in an AppTest instance."""
    next(widget for widget in app_test.selectbox if widget.label == "PFC probe").select("A").run()
    next(widget for widget in app_test.selectbox if widget.label == "HPC probe").select("B").run()


def test_view_inspects_metadata_but_waits_for_explicit_run(
    synthetic_communication_session: Path, monkeypatch,
):
    """Rendering and changing controls inspect readiness without loading spikes or fitting."""
    load_inputs_calls = []
    run_calls = []

    def forbid_load(*args, **kwargs):
        """Reject any raw-input load; return type is intentionally unreachable."""
        load_inputs_calls.append((args, kwargs))
        raise AssertionError("Display must not load spike arrays")

    def record_run(config, **kwargs):
        """Record an explicit request and return a valid no-result run report."""
        run_calls.append((config, kwargs))
        return SimpleNamespace(
            status="dry-run", result_path=None, run_directory=None,
            configuration_id="synthetic", elapsed_s=0.0,
        )

    monkeypatch.setattr(preparation, "load_session_inputs", forbid_load)
    monkeypatch.setattr(runner, "run_session", record_run)
    at = AppTest.from_function(_render, args=(str(synthetic_communication_session),)).run()
    assert not at.exception
    assert not run_calls and not load_inputs_calls
    assert next(button for button in at.button if button.label == "Run").disabled

    width = next(widget for widget in at.selectbox if widget.label == "Bin width (ms)")
    width.select(500).run()
    assert not at.exception
    assert not run_calls and not load_inputs_calls

    _assign_probe_roles(at)
    assert not next(button for button in at.button if button.label == "Run").disabled
    next(button for button in at.button if button.label == "Run").click().run()
    assert not at.exception
    assert len(run_calls) == 1
    assert run_calls[0][0].bin_width_ms == 500
    assert not load_inputs_calls


def _install_saved_result(monkeypatch, tmp_path):
    """Install a minimal immutable saved product and a real Matplotlib figure."""
    import matplotlib.pyplot as plt

    saved_configuration = {
        "configuration_id": "saved-500ms", "session_id": "synthetic-communication",
        "bin_width_ms": 500, "seed": 7, "window_s": [-2.0, 2.0],
    }
    result = SimpleNamespace(
        session_id="synthetic-communication", time_origin_s=0.0,
        choices=pd.DataFrame({"choice_time": [10.0]}),
        selected_trials=pd.DataFrame({"cur_trial": [100]}),
        choice_positions=[0], unit_ids={"PFC": ("A:0",), "HPC": ("B:0",)},
        bin_edges_s=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        probe_roles={"PFC": "A", "HPC": "B"},
        computation=SimpleNamespace(
            fold_scores=pd.DataFrame(), inner_scores=pd.DataFrame(),
            draw_scores=pd.DataFrame(), curve_scores=pd.DataFrame(),
            status="success",
        ),
        meta={"configuration": saved_configuration},
    )
    result_path = tmp_path / "saved-result.npz"
    result_path.write_bytes(b"synthetic archived result")
    info = SimpleNamespace(
        path=result_path, configuration=saved_configuration,
        completed_utc="2026-11-16T00:00:00Z", run_id="saved-run",
    )
    monkeypatch.setattr(results, "discover_results", Mock(return_value=(info,)))
    monkeypatch.setattr(results, "load_result", Mock(return_value=result))
    monkeypatch.setattr(plotting, "plot_rank_curves", Mock(side_effect=plt.figure))
    return result, info


def test_saved_result_displays_without_raw_sources_and_keeps_saved_caption(
    synthetic_communication_session: Path, monkeypatch, tmp_path,
):
    """Saved curves remain viewable without spike files and retain their saved width label."""
    _install_saved_result(monkeypatch, tmp_path)
    monkeypatch.setattr(preparation, "inspect_session_inputs", Mock(
        side_effect=FileNotFoundError("missing spike source")))
    monkeypatch.setattr(runner, "run_session", Mock(
        side_effect=AssertionError("Saved view must not run analysis")))
    (synthetic_communication_session.parent / "B" / "spike_clusters.npy").unlink()
    at = AppTest.from_function(_render, args=(str(synthetic_communication_session),)).run()
    assert not at.exception
    assert any("500 ms" in item.value for item in at.caption + at.markdown)
    assert next(button for button in at.button if button.label == "Run").disabled

    next(widget for widget in at.selectbox if widget.label == "Bin width (ms)").select(100).run()
    assert not at.exception
    assert any("500 ms" in item.value for item in at.caption + at.markdown)


def test_recompute_is_explicit_and_passes_rerun_true(
    synthetic_communication_session: Path, monkeypatch, tmp_path,
):
    """The explicit Recompute action requests a fresh immutable run."""
    _install_saved_result(monkeypatch, tmp_path)
    calls = []

    def record_run(config, **kwargs):
        """Record configuration and rerun flag, then return a completed report."""
        calls.append((config, kwargs))
        return SimpleNamespace(
            status="prepared", result_path=Path("saved-result.npz"),
            run_directory=Path("run"), configuration_id="new", elapsed_s=1.0,
        )

    monkeypatch.setattr(runner, "run_session", record_run)
    at = AppTest.from_function(_render, args=(str(synthetic_communication_session),)).run()
    assert not at.exception
    _assign_probe_roles(at)
    next(button for button in at.button if button.label == "Recompute").click().run()
    assert not at.exception
    assert len(calls) == 1 and calls[0][1]["rerun"] is True


def test_failed_recompute_preserves_saved_result(
    synthetic_communication_session: Path, monkeypatch, tmp_path,
):
    """A failed explicit recompute reports its error while retaining the prior saved curve."""
    _install_saved_result(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "run_session", Mock(side_effect=ValueError("synthetic failure")))
    plot_calls = plotting.plot_rank_curves
    at = AppTest.from_function(_render, args=(str(synthetic_communication_session),)).run()
    assert not at.exception
    next(button for button in at.button if button.label == "Recompute").click().run()
    assert not at.exception
    assert any("synthetic failure" in item.value for item in at.error)
    assert plot_calls.call_count >= 2


def test_missing_population_sources_disable_run_without_raw_loading(
    synthetic_communication_session: Path, monkeypatch,
):
    """A missing regional spike source yields unavailable readiness and disabled actions."""
    hpc_spike_file = synthetic_communication_session.parent / "B" / "spike_clusters.npy"
    hpc_spike_file.unlink()
    run_calls = []

    def record_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        raise AssertionError("Unavailable inputs must not reach the runner")

    monkeypatch.setattr(runner, "run_session", record_run)
    at = AppTest.from_function(_render, args=(str(synthetic_communication_session),)).run()
    assert not at.exception
    run_button = next(button for button in at.button if button.label == "Run")
    assert run_button.disabled
    assert not run_calls
    assert any("missing" in element.value.lower() for element in at.warning + at.error + at.info)
