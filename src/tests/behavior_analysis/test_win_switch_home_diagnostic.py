"""Saved-only Win-switch/Home diagnostic contracts, in original trial units."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import performance_plots as plots
from src.tests.behavior_analysis.test_candidate_expectant_integration import (
    _candidate_trial_table, gtf, main,
)


@pytest.fixture
def trials():
    """Four chronological rows; the manually rewarded second row is excluded."""
    table = gtf.collect_candidate_expectant_switching_features(
        _candidate_trial_table(), gtf.TaskParams(),
    )
    table = table.drop(columns=[c for c in table if c.startswith("probe_")])
    table["wsls_regressor"] = [0, None, 1, -1]
    table.index = [10, 20, 30, 40]
    return table


def test_preparation_preserves_rows_and_uses_canonical_inverse(trials):
    from src.behavior_analysis.win_switch_home import prepare_win_switch_home_data

    original = trials.copy(deep=True)
    prepared, spans, provenance = prepare_win_switch_home_data(trials)
    pd.testing.assert_frame_equal(trials, original)
    assert prepared.index.tolist() == [10, 20, 30, 40]
    np.testing.assert_allclose(prepared.win_switch_lose_stay_regressor,
                               [0, np.nan, -1, 1], equal_nan=True)
    assert spans.trial_start.tolist() == [-0.5, 1.5]
    assert spans.win_switch_lose_stay_mouse_agreement.tolist() == [0, 1]
    assert provenance["win_switch_lose_stay_regressor"]["source"] == "wsls_regressor"
    replayed, _, provenance = prepare_win_switch_home_data(trials.drop(columns="wsls_regressor"))
    np.testing.assert_allclose(replayed.win_switch_lose_stay_regressor,
                               [0, np.nan, -1, -1], equal_nan=True)
    assert "trial_features" in provenance["win_switch_lose_stay_regressor"]["source"]


def test_plot_has_only_requested_models_and_home_components(trials, monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(plots, "save_performance_figure", lambda fig, path: captured.append(fig))
    path = plots.plot_win_switch_home_diagnostics(trials, tmp_path, "synthetic")
    fig = captured[0]
    assert path.name == "synthetic_win_switch_home_diagnostic.png"
    lines = {line.get_label(): line for line in fig.axes[0].lines}
    assert {"Win-switch/lose-stay value", "Home E+D value", "History ideal value"} <= lines.keys()
    assert not any("Probe" in name for name in lines)
    np.testing.assert_allclose(lines["Win-switch/lose-stay value"].get_ydata(),
                               [0, np.nan, -1, 1], equal_nan=True)
    assert {line.get_label() for line in fig.axes[2].lines if not line.get_label().startswith("_")} == {
        "Home expectancy", "Home-model doubt"}
    assert "different scales" in " ".join(text.get_text() for text in fig.texts)
    plt.close(fig)


def test_saved_export_roundtrip_cache_and_missing_columns(trials, tmp_path):
    from src.behavior_analysis.win_switch_home import load_win_switch_home_data

    session = type("Session", (), dict(processed_data_path=tmp_path / "processed",
        figure_path=tmp_path / "figures", sess_id_full="synthetic"))()
    session.processed_data_path.mkdir()
    csv = session.processed_data_path / "synthetic_augmented_trials.csv"
    trials.to_csv(csv, index=False)
    before = csv.read_bytes()
    paths = main.save_and_plot_win_switch_home_for_session(session)
    prepared, spans, meta = load_win_switch_home_data(paths["results"])
    assert meta["analysis_version"] and meta["input_digest"]
    assert meta["parameters"] is None
    assert len(prepared) == 4 and len(spans) == 2
    assert paths["plot"].exists()
    saved = {key: path.stat().st_mtime_ns for key, path in paths.items()}
    assert main.save_and_plot_win_switch_home_for_session(session) == paths
    assert saved == {key: path.stat().st_mtime_ns for key, path in paths.items()}
    assert csv.read_bytes() == before
    with pytest.raises(ValueError, match="observer_value"):
        main.save_and_plot_win_switch_home_for_session(session, trials.drop(columns="observer_value"))


def test_normal_workflow_adds_output_and_retains_existing(trials, monkeypatch, tmp_path):
    monkeypatch.setattr(main.plot_model_values, "plot_mouse_history_candidate_expectant_switching_values",
                        lambda **kwargs: (trials, tmp_path / "old_values.png"))
    monkeypatch.setattr(main.performance_plots, "plot_candidate_expectant_switching_diagnostics",
                        lambda *args, **kwargs: tmp_path / "old_diagnostic.png")
    monkeypatch.setattr(main, "save_and_plot_win_switch_home_for_session",
                        lambda *args, **kwargs: {"plot": tmp_path / "new.png"})
    session = type("Session", (), dict(figure_path=tmp_path, sess_id_full="synthetic"))()
    result = main.plot_expectant_switching_for_session(trials, session, gtf.TaskParams(), include_legacy=False)
    assert result == {"candidate_value_plot": tmp_path / "old_values.png",
                      "candidate_diagnostic_plot": tmp_path / "old_diagnostic.png",
                      "win_switch_home_diagnostic_plot": tmp_path / "new.png"}


def test_quick_entry_uses_saved_session_resolver(monkeypatch, tmp_path):
    calls = []
    session = object()
    monkeypatch.setattr(main, "find_saved_session_by_date", lambda **kwargs: calls.append(kwargs) or session)
    monkeypatch.setattr(main, "save_and_plot_win_switch_home_for_session",
                        lambda selected: calls.append(selected) or {"plot": tmp_path / "new.png"})
    main.main_mouse_win_switch_home_diagnostic()
    assert calls[0]["mouse"] == "CT016" and calls[0]["date"] == "2026-05-29"
    assert calls[1] is session


def test_ties_keep_history_across_context_blocks(trials):
    """A zero score repeats the preceding greedy choice, including across blocks."""
    trials["wsls_regressor"] = [-1, None, 0, 0]
    trials["observer_value"] = [1, None, 0, 0]
    from src.behavior_analysis.win_switch_home import prepare_win_switch_home_data

    _, spans, _ = prepare_win_switch_home_data(trials)
    assert spans.win_switch_lose_stay_mouse_agreement.tolist() == [1, 0.5]
    assert spans.observer_mouse_agreement.tolist() == [1, 0.5]


def test_missing_inputs_fail_before_creating_outputs(trials, tmp_path):
    session = type("Session", (), dict(processed_data_path=tmp_path / "processed",
        figure_path=tmp_path / "figures", sess_id_full="synthetic"))()
    with pytest.raises(ValueError, match="home_referenced_expectancy_doubt_expectancy_choice"):
        main.save_and_plot_win_switch_home_for_session(
            session, trials.drop(columns="home_referenced_expectancy_doubt_expectancy_choice"))
    assert not session.processed_data_path.exists()
    assert not session.figure_path.exists()


def test_new_data_and_rerun_preserve_previous_results(trials, tmp_path):
    session = type("Session", (), dict(processed_data_path=tmp_path / "processed",
        figure_path=tmp_path / "figures", sess_id_full="synthetic"))()
    first = main.save_and_plot_win_switch_home_for_session(session, trials, feature_parameters={"home_weight": 1})
    original = {key: path.read_bytes() for key, path in first.items()}
    second = main.save_and_plot_win_switch_home_for_session(session, trials, feature_parameters={"home_weight": 2})
    assert first["results"] != second["results"]
    rerun = main.save_and_plot_win_switch_home_for_session(session, trials, feature_parameters={"home_weight": 1}, rerun=True)
    assert rerun["results"] != first["results"]
    assert original == {key: path.read_bytes() for key, path in first.items()}
