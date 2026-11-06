"""Persistence and plotting contracts for additive choice-window analysis."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import choice_window_results as results


@pytest.fixture
def trial_df():
    """Return 23 chronological right choices and one agent, shape (23, 4)."""
    return pd.DataFrame({
        "action": np.zeros(23), "reward": np.ones(23),
        "cur_block": np.repeat([0, 1, 2], [7, 7, 9]),
        "Qlearning_rel_value": np.full(23, -1.0),
    })


def test_save_roundtrip_cache_and_provenance(tmp_path, trial_df, monkeypatch):
    """Named NPZ arrays retain metadata and matching results skip scoring."""
    block_path = tmp_path / "session_block_performance.csv"
    block_path.write_text("existing context output\n")
    table, path = results.save_choice_window_results(trial_df, tmp_path, "session")
    restored, metadata = results.load_choice_window_results(path)
    pd.testing.assert_frame_equal(table, restored)
    assert metadata["analysis_version"]
    assert metadata["random_seed"] is None
    assert metadata["discarded_choices"] == {"10": 3, "20": 3}
    assert metadata["generator"]
    assert metadata["parameters"]["window_sizes"] == [10, 20]
    assert metadata["context_transition_trial_positions"] == [7, 14]
    assert metadata["analysis_source_digest"]
    assert (path.parent / "run.log").exists()
    assert (path.parent / "main.py").exists()
    with np.load(path, allow_pickle=True) as saved:
        assert "meta" in saved.files
        assert saved["valid_choice_trial_positions"].tolist() == list(range(23))
    monkeypatch.setattr(results, "compute_trial_agent_agreements", fail_if_scored)
    cached, cached_path = results.save_choice_window_results(trial_df, tmp_path, "session")
    assert cached_path == path
    pd.testing.assert_frame_equal(table, cached)
    assert block_path.read_text() == "existing context output\n"


def fail_if_scored(*args, **kwargs):
    """Fail if matching cached data is needlessly rescored; inputs arbitrary."""
    pytest.fail("Matching cached results should be loaded")


def test_changed_inputs_parameters_and_reruns_preserve_outputs(tmp_path, trial_df):
    """Input, parameter, and explicit rerun changes produce separate artifacts."""
    _, first = results.save_choice_window_results(trial_df, tmp_path, "session")
    original = first.read_bytes()
    trial_df.loc[0, "Qlearning_rel_value"] = 1.0
    _, changed = results.save_choice_window_results(trial_df, tmp_path, "session")
    _, sized = results.save_choice_window_results(trial_df, tmp_path, "session", (10,))
    _, rerun = results.save_choice_window_results(trial_df, tmp_path, "session", rerun=True)
    _, params = results.save_choice_window_results(
        trial_df, tmp_path, "session", feature_parameters={"alpha": 0.2}
    )
    assert len({first, changed, sized, rerun, params}) == 5
    assert first.read_bytes() == original


def test_empty_artifact_retains_schema(tmp_path, trial_df):
    """Short sessions save usable empty results, including requested sizes."""
    table, path = results.save_choice_window_results(trial_df.iloc[:3], tmp_path, "session")
    loaded, metadata = results.load_choice_window_results(path)
    pd.testing.assert_frame_equal(table, loaded)
    assert loaded.empty
    assert metadata["discarded_choices"] == {"10": 3, "20": 3}


def test_analysis_version_changes_invalidate_cache(tmp_path, trial_df, monkeypatch):
    """A new analysis version cannot reuse or overwrite an older result."""
    _, first = results.save_choice_window_results(trial_df, tmp_path, "session")
    monkeypatch.setattr(results, "ANALYSIS_VERSION", "test-new-version")
    _, second = results.save_choice_window_results(trial_df, tmp_path, "session")
    assert second != first
    assert first.exists()
    assert results.load_choice_window_results(second)[1]["analysis_version"] == "test-new-version"


def test_plot_saved_results_uses_trial_coordinates_and_counts(tmp_path, trial_df, monkeypatch):
    """Plotting uses saved values, labels denominators, and supports empty data."""
    from src.behavior_analysis.performance_plots import plot_session_choice_window_agreement
    import matplotlib.pyplot as plt

    table, path = results.save_choice_window_results(trial_df, tmp_path, "session")
    loaded, metadata = results.load_choice_window_results(path)
    captured = []
    original_savefig = plt.Figure.savefig

    def capture(figure, *args, **kwargs):
        """Capture figure objects then save using matplotlib's original API."""
        captured.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    output = plot_session_choice_window_agreement(loaded, metadata, tmp_path, "session")
    assert output.exists()
    assert output.stat().st_size > 0
    assert captured[0].get_facecolor() == (1, 1, 1, 1)
    assert captured[0].axes[0].lines[0].get_xdata().tolist() == [4.5, 14.5]
    assert "comparison" in captured[0].axes[1].get_ylabel().lower()
    empty = plot_session_choice_window_agreement(table.iloc[:0], metadata, tmp_path, "empty")
    assert empty.exists()
