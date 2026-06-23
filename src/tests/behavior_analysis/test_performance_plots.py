import importlib
import sys
from types import ModuleType
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
session_analysis = ModuleType("src.behavior_analysis.session_analysis")


def _import_performance_plots():
    """Import the plotting module after providing its legacy dependency alias."""
    sys.modules["src.behavior_analysis.session_analysis"] = session_analysis
    sys.modules["session_analysis"] = session_analysis
    return importlib.import_module("src.behavior_analysis.performance_plots")


def _capture_plot_calls(monkeypatch):
    """Record every Axes.plot call as simple x/y/label triples."""
    from matplotlib.axes import Axes

    plot_calls = []

    def fake_plot(self, x, y, *args, **kwargs):
        plot_calls.append(
            {
                "x": np.asarray(x).tolist(),
                "y": np.asarray(y).tolist(),
                "label": kwargs.get("label"),
            }
        )
        return []

    monkeypatch.setattr(Axes, "plot", fake_plot)
    return plot_calls


def _capture_fill_between_calls(monkeypatch):
    """Record every Axes.fill_between call as simple x/y-bound triples."""
    from matplotlib.axes import Axes

    fill_between_calls = []

    def fake_fill_between(self, x, y1, y2, *args, **kwargs):
        fill_between_calls.append(
            {
                "x": np.asarray(x).tolist(),
                "y1": np.asarray(y1).tolist(),
                "y2": np.asarray(y2).tolist(),
                "label": kwargs.get("label"),
            }
        )
        return None

    monkeypatch.setattr(Axes, "fill_between", fake_fill_between)
    return fill_between_calls


def test_save_performance_figure_uses_tight_bbox_and_closes_only_saved_figure(monkeypatch, tmp_path: Path):
    """The shared save helper should improve text clipping without closing unrelated figures."""
    performance_plots = _import_performance_plots()
    saved = {}

    fig_to_save, _ = plt.subplots()
    other_fig, _ = plt.subplots()

    def fake_savefig(path, **kwargs):
        saved["path"] = path
        saved["kwargs"] = kwargs

    monkeypatch.setattr(fig_to_save, "savefig", fake_savefig)

    save_path = tmp_path / "saved.png"
    performance_plots.save_performance_figure(fig_to_save, save_path)

    assert saved["path"] == save_path
    assert saved["kwargs"]["format"] == "png"
    assert saved["kwargs"]["dpi"] == 300
    assert saved["kwargs"]["bbox_inches"] == "tight"
    assert not plt.fignum_exists(fig_to_save.number)
    assert plt.fignum_exists(other_fig.number)
    plt.close(other_fig)


def test_plot_session_correct_skips_missing_percent_correct_values(monkeypatch, tmp_path: Path):
    """Block correctness plots should skip CSV missing strings before plotting."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": ["right_cued", "left_cued", "right_cued"],
            "percent_correct": [1.0, "None", "0.5"],
        }
    )

    performance_plots.plot_session_correct(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_correct",
    )

    assert [call["label"] for call in plot_calls] == ["right_cued"]
    assert plot_calls[0]["x"] == [0, 2]
    assert plot_calls[0]["y"] == [1.0, 0.5]


def test_plot_session_trials_to_correct_skips_missing_values_with_gaps(monkeypatch, tmp_path: Path):
    """Valid points should keep their original block indices after missing rows are omitted."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": [
                "right_cued",
                "left_cued",
                "right_cued",
                "left_uncued",
                "right_cued",
            ],
            "trials_to_correct": [1, None, 3, 2, 4],
        }
    )

    performance_plots.plot_session_trials_to_correct(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_a",
    )

    right_cued_call = next(call for call in plot_calls if call["label"] == "right cued")
    left_uncued_call = next(call for call in plot_calls if call["label"] == "left uncued")

    assert right_cued_call["x"] == [0, 2, 4]
    assert right_cued_call["y"] == [1, 3, 4]
    assert left_uncued_call["x"] == [3]
    assert left_uncued_call["y"] == [2]


def test_plot_session_trials_to_correct_supports_all_missing_encodings(monkeypatch, tmp_path: Path):
    """String, Python, and NaN missing values should all be excluded from plotted points."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": [
                "right_cued",
                "left_cued",
                "right_uncued",
                "left_uncued",
                "right_cued",
            ],
            "trials_to_correct": [1, "None", np.nan, None, 5],
        }
    )

    performance_plots.plot_session_trials_to_correct(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_b",
    )

    assert [call["label"] for call in plot_calls] == ["right cued"]
    assert plot_calls[0]["x"] == [0, 4]
    assert plot_calls[0]["y"] == [1, 5]


def test_plot_session_trials_to_correct_uses_block_specific_y_values(monkeypatch, tmp_path: Path):
    """Each block type should receive only its own trials-to-correct values."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": [
                "right_cued",
                "left_uncued",
                "right_cued",
                "left_uncued",
            ],
            "trials_to_correct": [1, 2, 7, 8],
        }
    )

    performance_plots.plot_session_trials_to_correct(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_c",
    )

    right_cued_call = next(call for call in plot_calls if call["label"] == "right cued")
    left_uncued_call = next(call for call in plot_calls if call["label"] == "left uncued")

    assert right_cued_call["x"] == [0, 2]
    assert right_cued_call["y"] == [1, 7]
    assert left_uncued_call["x"] == [1, 3]
    assert left_uncued_call["y"] == [2, 8]


def test_plot_session_trials_to_correct_skips_block_types_without_valid_points(monkeypatch, tmp_path: Path):
    """A block type with only missing entries should not generate an empty plot call."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": ["left_cued", "left_cued", "right_uncued"],
            "trials_to_correct": [None, "None", 6],
        }
    )

    performance_plots.plot_session_trials_to_correct(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_d",
    )

    assert [call["label"] for call in plot_calls] == ["right uncued"]
    assert plot_calls[0]["x"] == [2]
    assert plot_calls[0]["y"] == [6]


def test_plot_trials_to_correct_session_summary_draws_median_and_iqr(monkeypatch, tmp_path: Path):
    """Session summary plots should match learning-curve spacing with an IQR band."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-23"],
            "trials_to_correct_q1": [1.5, 2.0],
            "trials_to_correct_median": [3.0, 4.0],
            "trials_to_correct_q3": [4.5, 6.0],
        }
    )

    performance_plots.plot_trials_to_correct_session_summary(
        summary_df=summary_df,
        plot_path=tmp_path,
        figure_id="CT014",
    )

    assert plot_calls[0]["x"] == [1, 2]
    assert plot_calls[0]["y"] == [3.0, 4.0]
    assert fill_between_calls[0]["x"] == [1, 2]
    assert fill_between_calls[0]["y1"] == [1.5, 2.0]
    assert fill_between_calls[0]["y2"] == [4.5, 6.0]
    assert (tmp_path / "CT014_trials-to-correct-session-summary.png").exists()


def test_plot_multisession_oracle_behavior_saves_accuracy_and_reward_figures(tmp_path: Path):
    """Oracle summaries should be plotted as equally spaced session trajectories."""
    performance_plots = _import_performance_plots()
    overall_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-08"],
            "oracle_choice_accuracy": [0.6, 0.75],
            "oracle_reward_fraction": [0.8, 0.9],
            "oracle_reward_difference": [-2.0, -1.0],
        }
    )

    performance_plots.plot_multisession_oracle_behavior(
        overall_df=overall_df,
        plot_path=tmp_path,
        figure_id="CT014",
    )

    assert (tmp_path / "CT014_oracle-choice-accuracy.png").exists()
    assert (tmp_path / "CT014_oracle-reward-collection.png").exists()


def test_plot_multisession_oracle_behavior_requires_rerun_columns(tmp_path: Path):
    """Stale overall-performance CSVs should fail with a rerun instruction."""
    performance_plots = _import_performance_plots()
    overall_df = pd.DataFrame(
        {
            "date": ["2025-12-05"],
            "overall_correct": [0.6],
        }
    )

    with pytest.raises(ValueError, match="rerun session analyses"):
        performance_plots.plot_multisession_oracle_behavior(
            overall_df=overall_df,
            plot_path=tmp_path,
            figure_id="CT014",
        )


def test_plot_multisession_ideal_observer_behavior_saves_expected_figures(tmp_path: Path):
    """Ideal-observer summaries should be plotted across equally spaced sessions."""
    performance_plots = _import_performance_plots()
    overall_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-08"],
            "actual_reward_collected": [20.0, 25.0],
            "history_ideal_oracle_accuracy": [0.7, 0.8],
            "history_ideal_mouse_agreement": [0.6, 0.65],
            "history_ideal_reward_fraction": [0.75, 0.82],
            "history_ideal_expected_reward": [26.7, 30.5],
            "fixed_replay_ideal_reward_q1": [23.0, 27.0],
            "fixed_replay_ideal_reward_median": [25.0, 29.0],
            "fixed_replay_ideal_reward_q3": [28.0, 31.0],
            "fixed_replay_ideal_oracle_accuracy_q1": [0.65, 0.72],
            "fixed_replay_ideal_oracle_accuracy_median": [0.7, 0.78],
            "fixed_replay_ideal_oracle_accuracy_q3": [0.76, 0.84],
        }
    )

    performance_plots.plot_multisession_ideal_observer_behavior(
        overall_df=overall_df,
        plot_path=tmp_path,
        figure_id="CT014",
    )

    assert (tmp_path / "CT014_ideal-observer-choices.png").exists()
    assert (tmp_path / "CT014_ideal-observer-reward.png").exists()


def test_plot_multisession_ideal_observer_behavior_requires_rerun_columns(tmp_path: Path):
    """Stale overall-performance CSVs should fail clearly for ideal plots."""
    performance_plots = _import_performance_plots()
    overall_df = pd.DataFrame(
        {
            "date": ["2025-12-05"],
            "overall_correct": [0.6],
        }
    )

    with pytest.raises(ValueError, match="rerun session analyses"):
        performance_plots.plot_multisession_ideal_observer_behavior(
            overall_df=overall_df,
            plot_path=tmp_path,
            figure_id="CT014",
        )


def test_get_session_block_count_column_prefers_n_blocks():
    """New multisession summaries should use `n_blocks` as the block-count column."""
    performance_plots = _import_performance_plots()
    multisession_df = pd.DataFrame({"n_blocks": [4], "n_switches": [99]})

    block_count_col = performance_plots.get_session_block_count_column(multisession_df)

    assert block_count_col == "n_blocks"


def test_get_session_block_count_column_tolerates_legacy_n_switches():
    """Old multisession summaries should still plot when only `n_switches` exists."""
    performance_plots = _import_performance_plots()
    multisession_df = pd.DataFrame({"n_switches": [4]})

    block_count_col = performance_plots.get_session_block_count_column(multisession_df)

    assert block_count_col == "n_switches"


def _make_scatter_block_performance() -> pd.DataFrame:
    """Build minimal block data for trials-to-correct scatter plots."""
    return pd.DataFrame(
        {
            "block_type": ["right_cued", "left_cued", "right_uncued"],
            "trials_to_correct": [2, 4, "None"],
            "prev_n_correct": [1, 2, "None"],
            "prev_consecutive_rewards": [1, 3, 0],
        }
    )


def test_scatter_trials_to_correct_accepts_string_regression_stats(tmp_path: Path):
    """CSV-loaded string regression stats should be coerced before plotting."""
    performance_plots = _import_performance_plots()

    performance_plots.scatter_trials_to_correct(
        block_performance=_make_scatter_block_performance(),
        slope="0.5",
        intercept="1.0",
        plot_path=tmp_path,
        figure_id="unit_session",
    )

    assert (tmp_path / "unit_session_scatter_trials-to-correct.png").exists()


def test_scatter_trials_to_correct_rejects_missing_regression_stats(tmp_path: Path):
    """Missing string regression stats should fail with a clear error."""
    performance_plots = _import_performance_plots()

    with pytest.raises(ValueError, match="slope and intercept"):
        performance_plots.scatter_trials_to_correct(
            block_performance=_make_scatter_block_performance(),
            slope="None",
            intercept="1.0",
            plot_path=tmp_path,
            figure_id="unit_session",
        )


def test_plot_block_hmm_state_feature_scatter_saves_map_fit(tmp_path: Path):
    """Block HMM state-feature scatter should save one MAP-only summary plot."""
    performance_plots = _import_performance_plots()
    state_features = pd.DataFrame(
        {
            "fit_type": ["map", "map", "mle"],
            "bias": [-0.5, 0.25, 0.9],
            "prev_n_rewarded_weight": [1.0, -0.25, 2.0],
            "state_block_count": [4, 20, 100],
        }
    )

    save_path = performance_plots.plot_block_hmm_state_feature_scatter(
        state_features_df=state_features,
        plot_path=tmp_path,
        figure_id="CT999",
    )

    assert save_path == tmp_path / "CT999_map-block-hmm-state-feature-scatter.png"
    assert save_path.exists()


def test_plot_block_hmm_state_feature_scatter_drops_invalid_rows(tmp_path: Path):
    """CSV-style invalid numeric entries should be ignored before plotting."""
    performance_plots = _import_performance_plots()
    state_features = pd.DataFrame(
        {
            "fit_type": ["map", "map"],
            "bias": ["None", "0.1"],
            "prev_n_rewarded_weight": ["0.5", "None"],
            "state_block_count": ["3", "7"],
        }
    )

    save_path = performance_plots.plot_block_hmm_state_feature_scatter(
        state_features_df=state_features,
        plot_path=tmp_path,
        figure_id="CT999",
    )

    assert save_path.exists()


def test_plot_block_hmm_state_feature_scatter_can_use_fixed_marker_size(
    monkeypatch,
    tmp_path: Path,
):
    """Marker area should be fixed when block-count marker sizing is disabled."""
    from matplotlib.axes import Axes

    performance_plots = _import_performance_plots()
    state_features = pd.DataFrame(
        {
            "fit_type": ["map", "map"],
            "bias": [-0.5, 0.25],
            "prev_n_rewarded_weight": [1.0, -0.25],
            "state_block_count": [4, 20],
        }
    )
    scatter_calls = []

    def capture_scatter(self, x, y, *args, **kwargs):
        scatter_calls.append(kwargs)
        return Axes.scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)

    performance_plots.plot_block_hmm_state_feature_scatter(
        state_features_df=state_features,
        plot_path=tmp_path,
        figure_id="CT999",
        use_block_count_marker_size=False,
        fixed_marker_size=70.0,
    )

    assert scatter_calls[0]["s"] == 70.0
