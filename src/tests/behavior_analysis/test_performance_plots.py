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
    """Record every Axes.plot call as simple data/style dictionaries."""
    from matplotlib.axes import Axes

    plot_calls = []

    def fake_plot(self, x, y, *args, **kwargs):
        plot_calls.append(
            {
                "x": np.asarray(x).tolist(),
                "y": np.asarray(y).tolist(),
                "label": kwargs.get("label"),
                "color": kwargs.get("color"),
                "linewidth": kwargs.get("linewidth"),
                "markersize": kwargs.get("markersize"),
            }
        )
        return []

    monkeypatch.setattr(Axes, "plot", fake_plot)
    return plot_calls


def _capture_detailed_plot_calls(monkeypatch):
    """Record Axes.plot calls including positional style arguments."""
    from matplotlib.axes import Axes

    plot_calls = []

    def fake_plot(self, x, y, *args, **kwargs):
        plot_calls.append(
            {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "args": args,
                "kwargs": kwargs,
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


def _capture_scatter_calls(monkeypatch):
    """Record every Axes.scatter call as simple data/style dictionaries."""
    from matplotlib.axes import Axes

    scatter_calls = []

    def fake_scatter(self, x, y, *args, **kwargs):
        scatter_calls.append(
            {
                "x": np.asarray(x).tolist(),
                "y": np.asarray(y).tolist(),
                "label": kwargs.get("label"),
                "color": kwargs.get("color"),
                "alpha": kwargs.get("alpha"),
                "s": kwargs.get("s"),
            }
        )
        return None

    monkeypatch.setattr(Axes, "scatter", fake_scatter)
    return scatter_calls


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


def test_plot_session_agent_mouse_agreement_saves_overall_lines(monkeypatch, tmp_path: Path):
    """Agent-agreement plot should draw one overall block line per short agent label."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "qlearning_mouse_agreement": [0.5, "None", 1.0],
            "observer_mouse_agreement": [1.0, 0.5, 0.0],
        }
    )

    save_path = performance_plots.plot_session_agent_mouse_agreement(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_id_full="CT999_2026-01-01_120000",
        agreement_columns={
            "QL": "qlearning_mouse_agreement",
            "Ideal": "observer_mouse_agreement",
        },
    )

    assert save_path == tmp_path / "CT999_2026-01-01_120000_agent_mouse_agreement.png"
    assert save_path.exists()
    agent_calls = [call for call in plot_calls if call["label"] in {"QL", "Ideal"}]
    assert agent_calls == [
        {
            "x": [0.0, 2.0],
            "y": [0.5, 1.0],
            "label": "QL",
            "color": "#1f77b4",
            "linewidth": 2,
            "markersize": 5,
        },
        {
            "x": [0.0, 1.0, 2.0],
            "y": [1.0, 0.5, 0.0],
            "label": "Ideal",
            "color": "#111111",
            "linewidth": 2,
            "markersize": 5,
        },
    ]


def test_plot_session_agent_mouse_agreement_adds_summary_scatter(monkeypatch, tmp_path: Path):
    """Agent-agreement plot should include raw summary points and median markers."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    scatter_calls = _capture_scatter_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "qlearning_mouse_agreement": [0.5, "None", 1.0],
            "observer_mouse_agreement": [1.0, 0.5, 0.0],
        }
    )

    performance_plots.plot_session_agent_mouse_agreement(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_id_full="CT999_2026-01-01_120000",
        agreement_columns={
            "QL": "qlearning_mouse_agreement",
            "Ideal": "observer_mouse_agreement",
        },
    )

    raw_summary_calls = [
        call for call in scatter_calls
        if call["label"] == "_nolegend_"
    ]
    assert len(raw_summary_calls) == 2
    assert [call["y"] for call in raw_summary_calls] == [[0.5, 1.0], [1.0, 0.5, 0.0]]
    assert raw_summary_calls[0]["color"] == "#1f77b4"
    assert raw_summary_calls[0]["alpha"] == 0.65
    assert raw_summary_calls[0]["s"] == 22
    assert raw_summary_calls[1]["color"] == "#111111"

    median_calls = [
        call for call in plot_calls
        if call["label"] in {"QL median", "Ideal median"}
    ]
    assert median_calls == [
        {
            "x": [-0.12, 0.12],
            "y": [0.75, 0.75],
            "label": "QL median",
            "color": "#1f77b4",
            "linewidth": 4,
            "markersize": None,
        },
        {
            "x": [0.88, 1.12],
            "y": [0.5, 0.5],
            "label": "Ideal median",
            "color": "#111111",
            "linewidth": 4,
            "markersize": None,
        },
    ]


def test_plot_multisession_agent_mouse_agreement_quality_saves_png(tmp_path: Path):
    """Multisession agent-agreement plot should save the standalone PNG."""
    performance_plots = _import_performance_plots()
    summary_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "session_id": ["s1", "s2", "s1", "s2"],
            "agent": ["QL", "QL", "Ideal", "Ideal"],
            "n_blocks": [2, 2, 2, 2],
            "agreement_q1": [0.25, 0.5, 0.5, 0.75],
            "agreement_median": [0.5, 0.75, 0.75, 1.0],
            "agreement_q3": [0.75, 1.0, 1.0, 1.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-02"],
            "session_id": ["s1", "s1", "s1", "s2"],
            "agent": ["QL", "Ideal", "QL", "Ideal"],
            "block_ix": [0, 0, 1, 0],
            "block_type": ["left_cued", "left_cued", "right_cued", "right_cued"],
            "agreement": [0.25, 0.5, 0.75, 1.0],
        }
    )

    save_path = performance_plots.plot_multisession_agent_mouse_agreement_quality(
        summary_df=summary_df,
        block_points_df=block_points_df,
        plot_path=tmp_path,
        figure_id="CT999",
    )

    assert save_path == tmp_path / "CT999_agent-mouse-agreement-quality.png"
    assert save_path.exists()


def test_plot_multisession_agent_mouse_agreement_quality_uses_agent_colors(monkeypatch, tmp_path: Path):
    """Multisession agent-agreement plot should use the bold agent palette."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "session_id": ["s1", "s2", "s1", "s2"],
            "agent": ["QL", "QL", "Ideal", "Ideal"],
            "n_blocks": [2, 2, 2, 2],
            "agreement_q1": [0.25, 0.5, 0.5, 0.75],
            "agreement_median": [0.5, 0.75, 0.75, 1.0],
            "agreement_q3": [0.75, 1.0, 1.0, 1.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "session_id": ["s1", "s1", "s2", "s2"],
            "agent": ["QL", "Ideal", "QL", "Ideal"],
            "block_ix": [0, 0, 0, 0],
            "block_type": ["left_cued", "left_cued", "right_cued", "right_cued"],
            "agreement": [0.25, 0.5, 0.75, 1.0],
        }
    )

    performance_plots.plot_multisession_agent_mouse_agreement_quality(
        summary_df=summary_df,
        block_points_df=block_points_df,
        plot_path=tmp_path,
        figure_id="CT999",
    )

    median_calls = [
        call for call in plot_calls
        if call["label"] in {"QL median", "Ideal median"}
    ]
    assert median_calls[0]["color"] == "#1f77b4"
    assert median_calls[1]["color"] == "#111111"
    visible_range_labels = [
        call for call in plot_calls
        if call["label"] in {"QL Q1-Q3", "Ideal Q1-Q3"}
    ]
    assert visible_range_labels == []

    raw_calls = [
        call for call in plot_calls
        if call["label"] == "_nolegend_"
    ]
    assert len(raw_calls) == 2
    assert raw_calls[0]["color"] == "#1f77b4"
    assert raw_calls[1]["color"] == "#111111"


def test_default_agent_agreement_plotting_includes_simple_heuristics():
    """Agent agreement defaults should include the unfit heuristic strategies."""
    performance_plots = _import_performance_plots()

    assert performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS["Persev"] == "perseveration_mouse_agreement"
    assert (
        performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS["Doubt+P"]
        == "doubt_perseveration_mouse_agreement"
    )
    assert performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS["WSLS"] == "wsls_mouse_agreement"
    assert performance_plots.AGENT_MOUSE_AGREEMENT_COLORS["WSLS"] != "#111111"


def test_plot_session_correct_after_first_correct_marks_no_correct_blocks(
    monkeypatch,
    tmp_path: Path,
):
    """Post-first-correct plots should keep no-correct blocks visible."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "block_type": ["right_cued", "left_cued", "right_uncued"],
            "percent_correct_after_first_correct": [0.75, "None", 0.5],
        }
    )

    performance_plots.plot_session_correct_after_first_correct(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_correct_after_first",
    )

    numeric_calls = [
        call
        for call in plot_calls
        if call["args"] == ("o",) and "raw blocks" not in str(call["kwargs"].get("label"))
    ]
    no_correct_call = next(
        call
        for call in plot_calls
        if call["args"] == ("^",) and "no correct" in str(call["kwargs"].get("label"))
    )
    numeric_x = np.concatenate([call["x"] for call in numeric_calls])
    numeric_y = np.concatenate([call["y"] for call in numeric_calls])

    np.testing.assert_array_equal(numeric_x, np.array([0.0, 2.0]))
    np.testing.assert_array_equal(numeric_y, np.array([0.75, 0.5]))
    np.testing.assert_array_equal(no_correct_call["x"], np.array([1.0]))
    assert no_correct_call["y"][0] > 1.0
    assert (tmp_path / "session_correct_after_first_block_correct_after_first_correct.png").exists()


def test_plot_session_history_ideal_mouse_agreement_marks_missing_blocks(
    monkeypatch,
    tmp_path: Path,
):
    """History-ideal agreement plots should keep missing blocks visible."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "block_type": ["right_cued", "left_cued", "right_uncued"],
            "block_history_ideal_mouse_agreement": [0.75, "None", 0.5],
        }
    )

    performance_plots.plot_session_history_ideal_mouse_agreement(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_history_ideal",
    )

    numeric_calls = [
        call
        for call in plot_calls
        if call["args"] == ("o",) and "raw blocks" not in str(call["kwargs"].get("label"))
    ]
    missing_call = next(
        call
        for call in plot_calls
        if call["args"] == ("^",) and "no valid ideal" in str(call["kwargs"].get("label"))
    )
    numeric_x = np.concatenate([call["x"] for call in numeric_calls])
    numeric_y = np.concatenate([call["y"] for call in numeric_calls])

    np.testing.assert_array_equal(numeric_x, np.array([0.0, 2.0]))
    np.testing.assert_array_equal(numeric_y, np.array([0.75, 0.5]))
    np.testing.assert_array_equal(missing_call["x"], np.array([1.0]))
    assert missing_call["y"][0] > 1.0
    assert (tmp_path / "session_history_ideal_history_ideal_mouse_agreement.png").exists()


def _make_summary_grid_block_df(n_blocks: int = 6) -> pd.DataFrame:
    """Build a compact block table for summary-grid plotting tests."""
    block_types_for_rows = ["right_cued", "left_cued", "right_uncued", "left_uncued"] * 3
    return pd.DataFrame(
        {
            "block_ix": np.arange(n_blocks),
            "block_type": block_types_for_rows[:n_blocks],
            "trials_to_correct": np.arange(n_blocks) + 1,
            "prev_n_correct": np.ones(n_blocks, dtype=int),
            "prev_n_rewarded": np.arange(n_blocks),
            "n_switches": np.arange(n_blocks) % 3,
            "percent_correct_after_first_correct": np.linspace(0.2, 0.9, n_blocks),
            "block_history_ideal_mouse_agreement": np.linspace(0.3, 1.0, n_blocks),
        }
    )


def test_plot_single_session_summary_grid_saves_without_block_hmm(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    """Summary grid should save with clear placeholders when block HMM output is absent."""
    performance_plots = _import_performance_plots()

    def fake_raster_axis_plotter(ax, **_kwargs):
        ax.set_title("raster")
        return ax

    def fake_observer_axis_plotter(ax, **_kwargs):
        ax.plot([0, 1], [0.0, 1.0])
        ax.set_title("observer")
        return pd.DataFrame({"agent_relative_value": [0.0, 1.0]})

    block_df = _make_summary_grid_block_df()

    save_path = performance_plots.plot_single_session_summary_grid(
        block_performance=block_df,
        augmented_trial_df=pd.DataFrame({"action": [0, 1]}),
        event_df=pd.DataFrame({"Time": [0.0], "Event": ["enter_left_patch"]}),
        session_info={"use_dark_period": False},
        plot_path=tmp_path,
        figure_id="summary_missing_hmm",
        scatter_slope=1.0,
        scatter_intercept=0.0,
        raster_axis_plotter=fake_raster_axis_plotter,
        observer_axis_plotter=fake_observer_axis_plotter,
    )

    assert save_path == tmp_path / "summary_missing_hmm_single-session-summary-grid.png"
    assert save_path.exists()
    assert "Block HMM outputs not found" in capsys.readouterr().out


def test_plot_single_session_summary_grid_uses_block_hmm_panels(tmp_path: Path):
    """Summary grid should use HMM and sliding-regression panels when model output is supplied."""
    performance_plots = _import_performance_plots()
    calls = {"hmm": 0, "sliding": 0}

    class FakeBlockHmmModule:
        @staticmethod
        def prepare_block_lm_hmm_data(block_df, predictor_columns=("prev_n_rewarded",)):
            return {
                "observations": block_df["trials_to_correct"].to_numpy().reshape(-1, 1),
                "inputs": block_df["prev_n_rewarded"].to_numpy().reshape(-1, 1),
                "valid_mask": np.ones(block_df.shape[0], dtype=bool),
                "predictor_labels": ["prev_n_rewarded"],
            }

        @staticmethod
        def normalize_lm_observation_parameters(recovered_weights, recovered_mus):
            return recovered_weights, recovered_mus

        @staticmethod
        def build_presentation_colors(primary_predictor_weights):
            return ["blue", "red"], plt.cm.Set1.copy()

        @staticmethod
        def compute_sliding_block_regression(
            block_df,
            valid_mask=None,
            predictor_column="prev_n_rewarded",
            response_column="trials_to_correct",
            window_size=10,
            step_size=5,
        ):
            return pd.DataFrame(
                {
                    "window_center_position": [4.5],
                    "prev_n_rewarded_weight": [0.25],
                    "window_intercept": [2.0],
                }
            )

    def fake_hmm_plotter(**kwargs):
        calls["hmm"] += 1
        kwargs["obs_ax"].set_title("hmm")
        return kwargs["obs_ax"]

    def fake_sliding_plotter(regression_ax, sliding_regression_df, **_kwargs):
        calls["sliding"] += 1
        regression_ax.set_title("sliding")
        return regression_ax.twinx()

    block_df = _make_summary_grid_block_df(n_blocks=10)
    block_model_dict = {
        "map": {
            "posterior_probs": np.column_stack([np.ones(10), np.zeros(10)]),
            "hmm": object(),
            "weight_dict": {
                "weights": np.array([[[0.5]], [[-0.5]]]),
                "mus": np.array([[2.0], [4.0]]),
            },
        }
    }

    performance_plots.plot_single_session_summary_grid(
        block_performance=block_df,
        augmented_trial_df=pd.DataFrame({"action": [0, 1]}),
        event_df=pd.DataFrame({"Time": [0.0], "Event": ["enter_left_patch"]}),
        session_info={"use_dark_period": False},
        plot_path=tmp_path,
        figure_id="summary_with_hmm",
        scatter_slope=1.0,
        scatter_intercept=0.0,
        block_model_dict=block_model_dict,
        block_hmm_module=FakeBlockHmmModule,
        hmm_observation_plotter=fake_hmm_plotter,
        sliding_regression_plotter=fake_sliding_plotter,
        raster_axis_plotter=lambda ax, **_kwargs: ax,
        observer_axis_plotter=lambda ax, **_kwargs: pd.DataFrame(),
    )

    assert calls == {"hmm": 1, "sliding": 1}


def test_summary_grid_block_metric_uses_colorblock_side_colors(monkeypatch):
    """Summary grid block panels should use raster side colors and filled circles."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3, 4],
            "block_type": ["right_cued", "right_uncued", "left_cued", "left_uncued", "dark period"],
            "n_switches": [1, 2, 3, 4, 0],
        }
    )
    fig, ax = plt.subplots()

    performance_plots._plot_block_metric_on_ax(
        ax=ax,
        block_performance=block_performance,
        value_column="n_switches",
        y_label="switches",
        title="switches",
        use_summary_side_colors=True,
    )

    point_calls = [call for call in plot_calls if call["args"] and call["args"][0] == "o"]
    right_calls = [call for call in point_calls if "right" in call["kwargs"]["label"]]
    left_calls = [call for call in point_calls if "left" in call["kwargs"]["label"]]
    assert {call["kwargs"]["color"] for call in right_calls} == {"darkred"}
    assert {call["kwargs"]["color"] for call in left_calls} == {"blue"}
    assert all(call["kwargs"]["markerfacecolor"] == call["kwargs"]["color"] for call in point_calls)
    assert len(point_calls) == 5
    plt.close(fig)


def test_summary_grid_regression_scatter_uses_colorblock_side_colors(monkeypatch):
    """Summary grid regression panel should use side colors with filled circles."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": ["right_cued", "right_uncued", "left_cued", "left_uncued"],
            "trials_to_correct": [1, 2, 3, 4],
            "prev_n_correct": [1, 1, 1, 1],
            "prev_n_rewarded": [0, 1, 2, 3],
        }
    )
    fig, ax = plt.subplots()

    performance_plots._plot_trials_to_correct_scatter_on_ax(
        ax=ax,
        block_performance=block_performance,
        slope=1.0,
        intercept=0.0,
        use_summary_side_colors=True,
    )

    point_calls = [call for call in plot_calls if call["args"] and call["args"][0] == "o"]
    right_calls = [call for call in point_calls if "right" in call["kwargs"]["label"]]
    left_calls = [call for call in point_calls if "left" in call["kwargs"]["label"]]
    assert {call["kwargs"]["color"] for call in right_calls} == {"darkred"}
    assert {call["kwargs"]["color"] for call in left_calls} == {"blue"}
    assert all(call["kwargs"]["markerfacecolor"] == call["kwargs"]["color"] for call in point_calls)
    assert len(point_calls) == 4
    plt.close(fig)


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


def test_plot_session_block_quality_summary_plots_numeric_blocks_by_block_index(monkeypatch, tmp_path: Path):
    """The quality timeline should plot numeric TTC values at true block indices."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "block_type": ["right_cued", "left_cued", "right_uncued"],
            "trials_to_correct": [2, 7, "4"],
        }
    )

    performance_plots.plot_session_block_quality_summary(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_quality",
        point_jitter=0,
    )

    right_cued = next(call for call in plot_calls if call["kwargs"].get("label") == "right cued timeline")
    left_cued = next(call for call in plot_calls if call["kwargs"].get("label") == "left cued timeline")

    np.testing.assert_array_equal(right_cued["x"], np.array([0.0]))
    np.testing.assert_array_equal(right_cued["y"], np.array([2.0]))
    np.testing.assert_array_equal(left_cued["x"], np.array([1.0]))
    np.testing.assert_array_equal(left_cued["y"], np.array([7.0]))
    assert (tmp_path / "session_quality_block-trials-to-correct-summary.png").exists()


def test_plot_session_block_trials_to_correct_summary_saves_specific_filename(
    monkeypatch,
    tmp_path: Path,
):
    """The renamed TTC summary should save with a metric-specific filename."""
    performance_plots = _import_performance_plots()
    _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "block_type": ["right_cued", "left_cued"],
            "trials_to_correct": [2, 7],
        }
    )

    performance_plots.plot_session_block_trials_to_correct_summary(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_ttc",
        point_jitter=0,
    )

    assert (tmp_path / "session_ttc_block-trials-to-correct-summary.png").exists()


def test_plot_session_block_quality_summary_marks_no_correct_blocks(monkeypatch, tmp_path: Path):
    """No-correct side blocks should be visible on a top row instead of dropped."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "block_type": ["right_cued", "left_uncued", "right_uncued"],
            "trials_to_correct": [2, "None", 4],
        }
    )

    performance_plots.plot_session_block_quality_summary(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_quality",
        point_jitter=0,
    )

    no_correct = next(call for call in plot_calls if call["kwargs"].get("label") == "left uncued no correct")

    assert no_correct["args"] == ("^",)
    np.testing.assert_array_equal(no_correct["x"], np.array([1.0]))
    np.testing.assert_array_equal(no_correct["y"], np.array([5.0]))


def test_plot_session_block_quality_summary_handles_dark_period_blocks(monkeypatch, tmp_path: Path):
    """Dark periods should be shown on the timeline and excluded from side summaries."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "block_type": ["right_cued", "dark period", "left_cued"],
            "trials_to_correct": [2, "None", 4],
        }
    )

    performance_plots.plot_session_block_quality_summary(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_quality",
        point_jitter=0,
    )

    dark_call = next(call for call in plot_calls if call["kwargs"].get("label") == "dark period")
    raw_labels = [call["kwargs"].get("label") for call in plot_calls if "raw blocks" in str(call["kwargs"].get("label"))]

    assert dark_call["args"] == ("s",)
    np.testing.assert_array_equal(dark_call["x"], np.array([1.0]))
    np.testing.assert_array_equal(dark_call["y"], np.array([6.0]))
    assert "dark raw blocks" not in raw_labels


def test_plot_session_block_quality_summary_draws_side_medians_and_raw_points(monkeypatch, tmp_path: Path):
    """Side summaries should show raw side blocks, medians, and Q1-Q3 intervals."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "block_type": ["left_cued", "left_uncued", "right_cued", "right_uncued"],
            "trials_to_correct": [8, 10, 1, 3],
        }
    )

    performance_plots.plot_session_block_quality_summary(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_quality",
        point_jitter=0,
    )

    left_raw = next(call for call in plot_calls if call["kwargs"].get("label") == "left raw blocks")
    right_raw = next(call for call in plot_calls if call["kwargs"].get("label") == "right raw blocks")
    left_median = next(call for call in plot_calls if call["kwargs"].get("label") == "left median")
    right_median = next(call for call in plot_calls if call["kwargs"].get("label") == "right median")
    left_iqr = next(call for call in plot_calls if call["kwargs"].get("label") == "left Q1-Q3")

    np.testing.assert_array_equal(left_raw["x"], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(left_raw["y"], np.array([8.0, 10.0]))
    np.testing.assert_array_equal(right_raw["x"], np.array([1.0, 1.0]))
    np.testing.assert_array_equal(right_raw["y"], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(left_median["y"], np.array([9.0]))
    np.testing.assert_array_equal(right_median["y"], np.array([2.0]))
    np.testing.assert_array_equal(left_iqr["x"], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(left_iqr["y"], np.array([8.5, 9.5]))


def test_plot_session_correct_after_first_correct_draws_side_medians_and_raw_points(
    monkeypatch,
    tmp_path: Path,
):
    """Post-first-correct plots should include the reusable side-summary panel."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "block_type": ["left_cued", "left_uncued", "right_cued", "right_uncued"],
            "percent_correct_after_first_correct": [0.8, 1.0, 0.25, 0.75],
        }
    )

    performance_plots.plot_session_correct_after_first_correct(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_correct_after_first",
    )

    left_raw = next(call for call in plot_calls if call["kwargs"].get("label") == "left raw blocks")
    right_median = next(call for call in plot_calls if call["kwargs"].get("label") == "right median")
    left_iqr = next(call for call in plot_calls if call["kwargs"].get("label") == "left Q1-Q3")

    np.testing.assert_array_equal(left_raw["y"], np.array([0.8, 1.0]))
    np.testing.assert_array_equal(right_median["y"], np.array([0.5]))
    np.testing.assert_allclose(left_iqr["y"], [0.85, 0.95])
    assert (tmp_path / "session_correct_after_first_block_correct_after_first_correct.png").exists()


def test_plot_session_history_ideal_mouse_agreement_draws_side_medians_and_raw_points(
    monkeypatch,
    tmp_path: Path,
):
    """History-ideal plots should include the reusable side-summary panel."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "block_type": ["left_cued", "left_uncued", "right_cued", "right_uncued"],
            "block_history_ideal_mouse_agreement": [0.8, 1.0, 0.25, 0.75],
        }
    )

    performance_plots.plot_session_history_ideal_mouse_agreement(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_history_ideal",
    )

    left_raw = next(call for call in plot_calls if call["kwargs"].get("label") == "left raw blocks")
    right_median = next(call for call in plot_calls if call["kwargs"].get("label") == "right median")
    left_iqr = next(call for call in plot_calls if call["kwargs"].get("label") == "left Q1-Q3")

    np.testing.assert_array_equal(left_raw["y"], np.array([0.8, 1.0]))
    np.testing.assert_array_equal(right_median["y"], np.array([0.5]))
    np.testing.assert_allclose(left_iqr["y"], [0.85, 0.95])
    assert (tmp_path / "session_history_ideal_history_ideal_mouse_agreement.png").exists()


def test_plot_session_nswitches_draws_side_medians_and_raw_points(
    monkeypatch,
    tmp_path: Path,
):
    """Block switch-count plots should include the reusable side-summary panel."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "block_type": ["left_cued", "left_uncued", "right_cued", "right_uncued"],
            "n_switches": [8, 10, 1, 3],
        }
    )

    performance_plots.plot_session_nswitches(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_switches",
    )

    left_raw = next(call for call in plot_calls if call["kwargs"].get("label") == "left raw blocks")
    right_median = next(call for call in plot_calls if call["kwargs"].get("label") == "right median")
    left_iqr = next(call for call in plot_calls if call["kwargs"].get("label") == "left Q1-Q3")

    np.testing.assert_array_equal(left_raw["y"], np.array([8.0, 10.0]))
    np.testing.assert_array_equal(right_median["y"], np.array([2.0]))
    np.testing.assert_array_equal(left_iqr["y"], np.array([8.5, 9.5]))
    assert (tmp_path / "session_switches_block_nswitches.png").exists()


def test_plot_session_explore_trials_draws_side_medians_and_raw_points(
    monkeypatch,
    tmp_path: Path,
):
    """Explore-trial plots should include block timeline and side summaries."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "block_type": ["left_cued", "left_uncued", "right_cued", "right_uncued"],
            "n_explore_trials": [2, 4, 1, 3],
        }
    )

    performance_plots.plot_session_explore_trials(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_explore",
    )

    left_raw = next(call for call in plot_calls if call["kwargs"].get("label") == "left raw blocks")
    right_median = next(call for call in plot_calls if call["kwargs"].get("label") == "right median")
    left_iqr = next(call for call in plot_calls if call["kwargs"].get("label") == "left Q1-Q3")

    np.testing.assert_array_equal(left_raw["y"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(right_median["y"], np.array([2.0]))
    np.testing.assert_array_equal(left_iqr["y"], np.array([2.5, 3.5]))
    assert (tmp_path / "session_explore_block_explore_trials.png").exists()


def test_plot_session_explore_runs_draws_side_medians_and_raw_points(
    monkeypatch,
    tmp_path: Path,
):
    """Explore-run plots should include block timeline and side summaries."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "block_type": ["left_cued", "left_uncued", "right_cued", "right_uncued"],
            "n_explore_runs": [2, 4, 1, 3],
        }
    )

    performance_plots.plot_session_explore_runs(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_explore_runs",
    )

    left_raw = next(call for call in plot_calls if call["kwargs"].get("label") == "left raw blocks")
    right_median = next(call for call in plot_calls if call["kwargs"].get("label") == "right median")
    left_iqr = next(call for call in plot_calls if call["kwargs"].get("label") == "left Q1-Q3")

    np.testing.assert_array_equal(left_raw["y"], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(right_median["y"], np.array([2.0]))
    np.testing.assert_array_equal(left_iqr["y"], np.array([2.5, 3.5]))
    assert (tmp_path / "session_explore_runs_block_explore_runs.png").exists()


def test_plot_session_block_quality_summary_handles_full_completion_bias(monkeypatch, tmp_path: Path):
    """A fully complete biased session should still show left/right TTC asymmetry."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "block_type": ["left_cued", "right_cued", "left_uncued", "right_uncued"],
            "trials_to_correct": [9, 1, 11, 2],
        }
    )

    performance_plots.plot_session_block_quality_summary(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_ID="session_quality",
        point_jitter=0,
    )

    no_correct_calls = [call for call in plot_calls if "no correct" in str(call["kwargs"].get("label"))]
    left_median = next(call for call in plot_calls if call["kwargs"].get("label") == "left median")
    right_median = next(call for call in plot_calls if call["kwargs"].get("label") == "right median")

    assert no_correct_calls == []
    np.testing.assert_array_equal(left_median["y"], np.array([10.0]))
    np.testing.assert_array_equal(right_median["y"], np.array([1.5]))


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


def _make_side_trials_to_correct_summary() -> pd.DataFrame:
    """Build minimal side-specific TTC summary data for plotting tests."""
    return pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-16", "2025-12-16"],
            "session_id": [
                "CT014_2025-12-05_165240",
                "CT014_2025-12-05_165240",
                "CT014_2025-12-16_153200",
                "CT014_2025-12-16_153200",
            ],
            "rewarded_side": ["left", "right", "left", "right"],
            "n_blocks": [2, 2, 2, 2],
            "n_valid_blocks": [1, 2, 2, 1],
            "n_no_correct_blocks": [1, 0, 0, 1],
            "completion_fraction": [0.5, 1.0, 1.0, 0.5],
            "trials_to_correct_q1": [2.0, 1.0, 4.0, 5.0],
            "trials_to_correct_median": [3.0, 2.0, 5.0, 5.0],
            "trials_to_correct_q3": [4.0, 3.0, 6.0, 5.0],
        }
    )


def _make_side_trials_to_correct_points() -> pd.DataFrame:
    """Build side-specific raw block TTC points for plotting tests."""
    return pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-16"],
            "session_id": [
                "CT014_2025-12-05_165240",
                "CT014_2025-12-05_165240",
                "CT014_2025-12-16_153200",
            ],
            "rewarded_side": ["left", "left", "right"],
            "block_ix": [0, 1, 2],
            "block_type": ["left_cued", "left_uncued", "right_cued"],
            "trials_to_correct_numeric": [2.0, np.nan, np.nan],
            "no_correct_choice": [False, True, True],
        }
    )


def test_plot_side_trials_to_correct_quality_draws_completion_traces(monkeypatch, tmp_path: Path):
    """Side quality plots should draw separate left/right completion fractions."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)

    performance_plots.plot_side_trials_to_correct_quality(
        summary_df=_make_side_trials_to_correct_summary(),
        block_points_df=_make_side_trials_to_correct_points(),
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
    )

    left_completion = next(call for call in plot_calls if call["kwargs"].get("label") == "left completion")
    right_completion = next(call for call in plot_calls if call["kwargs"].get("label") == "right completion")

    np.testing.assert_array_equal(left_completion["x"], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(left_completion["y"], np.array([0.5, 1.0]))
    np.testing.assert_array_equal(right_completion["x"], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(right_completion["y"], np.array([1.0, 0.5]))


def test_plot_side_trials_to_correct_quality_draws_median_iqr_and_raw_points(
    monkeypatch,
    tmp_path: Path,
):
    """Side quality plots should show summary spread and raw numeric block points."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)

    performance_plots.plot_side_trials_to_correct_quality(
        summary_df=_make_side_trials_to_correct_summary(),
        block_points_df=_make_side_trials_to_correct_points(),
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
    )

    left_iqr = next(call for call in fill_between_calls if call["label"] == "left Q1-Q3")
    left_median = next(call for call in plot_calls if call["kwargs"].get("label") == "left median")
    raw_points = next(call for call in plot_calls if call["kwargs"].get("label") == "left raw blocks")

    assert left_iqr["x"] == [1, 2]
    assert left_iqr["y1"] == [2.0, 4.0]
    assert left_iqr["y2"] == [4.0, 6.0]
    np.testing.assert_array_equal(left_median["y"], np.array([3.0, 5.0]))
    np.testing.assert_array_equal(raw_points["x"], np.array([1.0]))
    np.testing.assert_array_equal(raw_points["y"], np.array([2.0]))


def test_plot_side_trials_to_correct_quality_marks_no_correct_blocks(monkeypatch, tmp_path: Path):
    """No-correct blocks should be displayed on a top row instead of dropped."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)

    performance_plots.plot_side_trials_to_correct_quality(
        summary_df=_make_side_trials_to_correct_summary(),
        block_points_df=_make_side_trials_to_correct_points(),
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
    )

    left_no_correct = next(call for call in plot_calls if call["kwargs"].get("label") == "left no correct")
    right_no_correct = next(call for call in plot_calls if call["kwargs"].get("label") == "right no correct")

    assert left_no_correct["args"] == ("^",)
    np.testing.assert_array_equal(left_no_correct["x"], np.array([1.0]))
    np.testing.assert_array_equal(left_no_correct["y"], np.array([7.0]))
    np.testing.assert_array_equal(right_no_correct["x"], np.array([2.0]))
    np.testing.assert_array_equal(right_no_correct["y"], np.array([7.0]))


def test_plot_side_trials_to_correct_quality_caps_large_trial_counts(
    monkeypatch,
    tmp_path: Path,
):
    """Large TTC values should use an overflow row instead of stretching the y-axis."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05"],
            "session_id": ["CT014_2025-12-05_165240"],
            "rewarded_side": ["left"],
            "n_blocks": [3],
            "n_valid_blocks": [3],
            "n_no_correct_blocks": [0],
            "completion_fraction": [1.0],
            "trials_to_correct_q1": [10.0],
            "trials_to_correct_median": [30.0],
            "trials_to_correct_q3": [50.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05"],
            "session_id": ["CT014_2025-12-05_165240", "CT014_2025-12-05_165240"],
            "rewarded_side": ["left", "left"],
            "block_ix": [0, 1],
            "block_type": ["left_cued", "left_uncued"],
            "trials_to_correct_numeric": [10.0, 50.0],
            "no_correct_choice": [False, False],
        }
    )

    performance_plots.plot_side_trials_to_correct_quality(
        summary_df=summary_df,
        block_points_df=block_points_df,
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
        trial_display_cap=25,
    )

    raw_points = next(call for call in plot_calls if call["kwargs"].get("label") == "left raw blocks")
    median = next(call for call in plot_calls if call["kwargs"].get("label") == "left median")
    left_iqr = next(call for call in fill_between_calls if call["label"] == "left Q1-Q3")

    np.testing.assert_array_equal(raw_points["y"], np.array([10.0, 26.0]))
    np.testing.assert_array_equal(median["y"], np.array([26.0]))
    assert left_iqr["y1"] == [10.0]
    assert left_iqr["y2"] == [26.0]
    assert (tmp_path / "CT014_side-trials-to-correct-quality.png").exists()


def test_plot_multisession_block_switches_quality_draws_overall_left_right(
    monkeypatch,
    tmp_path: Path,
):
    """Switch quality plots should draw overall, left, and right summary layers."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "switch_group": ["overall", "left", "right"],
            "n_switches_q1": [1.0, 2.0, 3.0],
            "n_switches_median": [2.0, 3.0, 4.0],
            "n_switches_q3": [3.0, 4.0, 5.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "switch_group": ["overall", "left", "right"],
            "n_switches": [1, 2, 3],
        }
    )

    performance_plots.plot_multisession_block_switches_quality(
        summary_df=summary_df,
        block_points_df=block_points_df,
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
    )

    median_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("median")
    }
    raw_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("raw blocks")
    }
    spread_labels = {call["label"] for call in fill_between_calls}

    assert median_labels == {"overall median", "left median", "right median"}
    assert raw_labels == {"overall raw blocks", "left raw blocks", "right raw blocks"}
    assert spread_labels == {"overall Q1-Q3", "left Q1-Q3", "right Q1-Q3"}
    assert (tmp_path / "CT014_block-switches-quality.png").exists()


def test_plot_side_trials_to_correct_quality_draws_overall_left_right(
    monkeypatch,
    tmp_path: Path,
):
    """TTC quality plots should include overall, left, and right summary layers."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "rewarded_side": ["overall", "left", "right"],
            "completion_fraction": [1.0, 1.0, 1.0],
            "trials_to_correct_q1": [1.0, 2.0, 3.0],
            "trials_to_correct_median": [2.0, 3.0, 4.0],
            "trials_to_correct_q3": [3.0, 4.0, 5.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "rewarded_side": ["overall", "left", "right"],
            "trials_to_correct_numeric": [1.0, 2.0, 3.0],
            "no_correct_choice": [False, False, False],
        }
    )

    performance_plots.plot_side_trials_to_correct_quality(
        summary_df=summary_df,
        block_points_df=block_points_df,
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
    )

    median_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("median")
    }
    raw_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("raw blocks")
    }
    spread_labels = {call["label"] for call in fill_between_calls}

    assert median_labels == {"overall median", "left median", "right median"}
    assert raw_labels == {"overall raw blocks", "left raw blocks", "right raw blocks"}
    assert spread_labels == {"overall Q1-Q3", "left Q1-Q3", "right Q1-Q3"}


def test_plot_correct_after_first_session_summary_draws_overall_median_iqr(
    monkeypatch,
    tmp_path: Path,
):
    """Overall post-first-correct summary should mirror the overall TTC plot."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-06"],
            "correct_after_first_group": ["overall", "overall"],
            "percent_correct_after_first_q1": [0.25, 0.5],
            "percent_correct_after_first_median": [0.5, 0.75],
            "percent_correct_after_first_q3": [0.75, 1.0],
        }
    )

    performance_plots.plot_correct_after_first_session_summary(
        summary_df=summary_df,
        plot_path=tmp_path,
        figure_id="CT014",
    )

    median_call = next(call for call in plot_calls if call["label"] == "Median")
    assert median_call["y"] == [0.5, 0.75]
    assert fill_between_calls[0]["y1"] == [0.25, 0.5]
    assert fill_between_calls[0]["y2"] == [0.75, 1.0]
    assert (tmp_path / "CT014_correct-after-first-session-summary.png").exists()


def test_plot_multisession_block_switches_quality_on_ax_draws_overall_left_right(
    monkeypatch,
):
    """Axis helper should preserve overall, left, and right switch summaries."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "switch_group": ["overall", "left", "right"],
            "n_switches_q1": [1.0, 2.0, 3.0],
            "n_switches_median": [2.0, 3.0, 4.0],
            "n_switches_q3": [3.0, 4.0, 5.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "switch_group": ["overall", "left", "right"],
            "n_switches": [1, 2, 3],
        }
    )
    fig, ax = plt.subplots()

    performance_plots.plot_multisession_block_switches_quality_on_ax(
        ax=ax,
        summary_df=summary_df,
        block_points_df=block_points_df,
        figure_id="CT014",
        point_jitter=0,
    )

    median_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("median")
    }
    raw_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("raw blocks")
    }
    spread_labels = {call["label"] for call in fill_between_calls}

    assert median_labels == {"overall median", "left median", "right median"}
    assert raw_labels == {"overall raw blocks", "left raw blocks", "right raw blocks"}
    assert spread_labels == {"overall Q1-Q3", "left Q1-Q3", "right Q1-Q3"}
    plt.close(fig)


def test_plot_multisession_block_explore_quality_draws_overall_left_right(
    monkeypatch,
    tmp_path: Path,
):
    """Explore quality plots should draw overall, left, and right summary layers."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "explore_group": ["overall", "left", "right"],
            "n_explore_trials_q1": [1.0, 2.0, 3.0],
            "n_explore_trials_median": [2.0, 3.0, 4.0],
            "n_explore_trials_q3": [3.0, 4.0, 5.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "explore_group": ["overall", "left", "right"],
            "n_explore_trials": [1, 2, 3],
        }
    )

    performance_plots.plot_multisession_block_explore_quality(
        summary_df=summary_df,
        block_points_df=block_points_df,
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
    )

    median_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("median")
    }
    raw_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("raw blocks")
    }
    spread_labels = {call["label"] for call in fill_between_calls}

    assert median_labels == {"overall median", "left median", "right median"}
    assert raw_labels == {"overall raw blocks", "left raw blocks", "right raw blocks"}
    assert spread_labels == {"overall Q1-Q3", "left Q1-Q3", "right Q1-Q3"}
    assert (tmp_path / "CT014_block-explore-quality.png").exists()


def test_plot_multisession_block_explore_run_quality_draws_overall_left_right(
    monkeypatch,
    tmp_path: Path,
):
    """Explore-run quality plots should draw overall, left, and right summary layers."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "explore_group": ["overall", "left", "right"],
            "n_explore_runs_q1": [1.0, 2.0, 3.0],
            "n_explore_runs_median": [2.0, 3.0, 4.0],
            "n_explore_runs_q3": [3.0, 4.0, 5.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "explore_group": ["overall", "left", "right"],
            "n_explore_runs": [1, 2, 3],
        }
    )

    performance_plots.plot_multisession_block_explore_run_quality(
        summary_df=summary_df,
        block_points_df=block_points_df,
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
    )

    median_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("median")
    }
    raw_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("raw blocks")
    }
    spread_labels = {call["label"] for call in fill_between_calls}

    assert median_labels == {"overall median", "left median", "right median"}
    assert raw_labels == {"overall raw blocks", "left raw blocks", "right raw blocks"}
    assert spread_labels == {"overall Q1-Q3", "left Q1-Q3", "right Q1-Q3"}
    assert (tmp_path / "CT014_block-explore-run-quality.png").exists()


def test_plot_multisession_correct_after_first_quality_draws_groups_and_no_correct(
    monkeypatch,
    tmp_path: Path,
):
    """Post-first-correct quality plots should draw grouped summaries and no-correct markers."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    fill_between_calls = _capture_fill_between_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05"],
            "correct_after_first_group": ["overall", "left", "right"],
            "percent_correct_after_first_q1": [0.25, 0.5, 0.75],
            "percent_correct_after_first_median": [0.5, 0.75, 0.9],
            "percent_correct_after_first_q3": [0.75, 1.0, 1.0],
        }
    )
    block_points_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-05", "2025-12-05", "2025-12-05"],
            "correct_after_first_group": ["overall", "left", "right", "left"],
            "percent_correct_after_first_numeric": [0.5, 0.75, 0.9, np.nan],
            "no_correct_choice": [False, False, False, True],
        }
    )

    performance_plots.plot_multisession_correct_after_first_quality(
        summary_df=summary_df,
        block_points_df=block_points_df,
        plot_path=tmp_path,
        figure_id="CT014",
        point_jitter=0,
    )

    median_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("median")
    }
    raw_labels = {
        call["kwargs"].get("label")
        for call in plot_calls
        if call["kwargs"].get("label", "").endswith("raw blocks")
    }
    no_correct_call = next(call for call in plot_calls if call["kwargs"].get("label") == "left no correct")
    spread_labels = {call["label"] for call in fill_between_calls}

    assert median_labels == {"overall median", "left median", "right median"}
    assert raw_labels == {"overall raw blocks", "left raw blocks", "right raw blocks"}
    assert spread_labels == {"overall Q1-Q3", "left Q1-Q3", "right Q1-Q3"}
    assert no_correct_call["args"] == ("^",)
    assert no_correct_call["y"][0] > 1.0
    assert (tmp_path / "CT014_correct-after-first-quality.png").exists()


def test_plot_multisession_summary_grid_saves_with_hmm_placeholders(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    """Multisession summary grid should save even when block HMM output is absent."""
    performance_plots = _import_performance_plots()
    helper_calls = []

    def record_helper(name):
        def _helper(ax, **_kwargs):
            helper_calls.append(name)
            ax.set_title(name)
            return ax

        return _helper

    performance_plots.plot_multisession_summary_grid(
        block_performance=_make_summary_grid_block_df(n_blocks=4),
        side_trials_to_correct_summary=_make_side_trials_to_correct_summary(),
        side_trials_to_correct_block_points=_make_side_trials_to_correct_points(),
        correct_after_first_summary=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "correct_after_first_group": ["overall"],
                "percent_correct_after_first_q1": [0.25],
                "percent_correct_after_first_median": [0.5],
                "percent_correct_after_first_q3": [0.75],
            }
        ),
        correct_after_first_block_points=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "correct_after_first_group": ["overall"],
                "percent_correct_after_first_numeric": [0.5],
                "no_correct_choice": [False],
            }
        ),
        block_switch_summary=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "switch_group": ["overall"],
                "n_switches_q1": [1.0],
                "n_switches_median": [2.0],
                "n_switches_q3": [3.0],
            }
        ),
        block_switch_block_points=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "switch_group": ["overall"],
                "n_switches": [2],
            }
        ),
        overall_df=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "actual_reward_collected": [20.0],
                "history_ideal_oracle_accuracy": [0.7],
                "history_ideal_mouse_agreement": [0.6],
                "history_ideal_reward_fraction": [0.75],
                "history_ideal_expected_reward": [26.7],
                "fixed_replay_ideal_reward_q1": [23.0],
                "fixed_replay_ideal_reward_median": [25.0],
                "fixed_replay_ideal_reward_q3": [28.0],
                "fixed_replay_ideal_oracle_accuracy_q1": [0.65],
                "fixed_replay_ideal_oracle_accuracy_median": [0.7],
                "fixed_replay_ideal_oracle_accuracy_q3": [0.76],
            }
        ),
        plot_path=tmp_path,
        figure_id="CT014",
        ttc_axis_plotter=record_helper("ttc"),
        correct_after_first_axis_plotter=record_helper("correct_after_first"),
        block_switch_axis_plotter=record_helper("switches"),
        ideal_choices_axis_plotter=record_helper("ideal_choices"),
    )

    assert (tmp_path / "CT014_multisession-summary-grid.png").exists()
    assert helper_calls == ["ttc", "switches", "correct_after_first", "ideal_choices"]
    assert "Block HMM outputs not found" in capsys.readouterr().out


def test_plot_multisession_summary_grid_uses_hmm_panels(tmp_path: Path):
    """Multisession summary grid should use HMM and sliding panels when supplied."""
    performance_plots = _import_performance_plots()
    calls = {"hmm": 0, "sliding": 0}

    class FakeBlockHmmModule:
        @staticmethod
        def prepare_block_lm_hmm_data(block_df, predictor_columns=("prev_n_rewarded",)):
            return {
                "observations": block_df["trials_to_correct"].to_numpy().reshape(-1, 1),
                "inputs": block_df["prev_n_rewarded"].to_numpy().reshape(-1, 1),
                "valid_mask": np.ones(block_df.shape[0], dtype=bool),
                "predictor_labels": ["prev_n_rewarded"],
            }

        @staticmethod
        def normalize_lm_observation_parameters(recovered_weights, recovered_mus):
            return recovered_weights, recovered_mus

        @staticmethod
        def build_presentation_colors(primary_predictor_weights):
            return ["blue", "red"], plt.cm.Set1.copy()

        @staticmethod
        def compute_sliding_block_regression(
            block_df,
            valid_mask=None,
            predictor_column="prev_n_rewarded",
            response_column="trials_to_correct",
            window_size=10,
            step_size=5,
        ):
            return pd.DataFrame(
                {
                    "window_center_position": [4.5],
                    "prev_n_rewarded_weight": [0.25],
                    "window_intercept": [2.0],
                }
            )

    def fake_hmm_plotter(**kwargs):
        calls["hmm"] += 1
        kwargs["obs_ax"].set_title("hmm")
        return kwargs["obs_ax"]

    def fake_sliding_plotter(regression_ax, sliding_regression_df, **_kwargs):
        calls["sliding"] += 1
        regression_ax.set_title("sliding")
        return regression_ax.twinx()

    block_model_dict = {
        "map": {
            "posterior_probs": np.column_stack([np.ones(10), np.zeros(10)]),
            "hmm": object(),
            "weight_dict": {
                "weights": np.array([[[0.5]], [[-0.5]]]),
                "mus": np.array([[2.0], [4.0]]),
            },
        }
    }

    performance_plots.plot_multisession_summary_grid(
        block_performance=_make_summary_grid_block_df(n_blocks=10),
        side_trials_to_correct_summary=_make_side_trials_to_correct_summary(),
        side_trials_to_correct_block_points=_make_side_trials_to_correct_points(),
        correct_after_first_summary=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "correct_after_first_group": ["overall"],
                "percent_correct_after_first_q1": [0.25],
                "percent_correct_after_first_median": [0.5],
                "percent_correct_after_first_q3": [0.75],
            }
        ),
        correct_after_first_block_points=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "correct_after_first_group": ["overall"],
                "percent_correct_after_first_numeric": [0.5],
                "no_correct_choice": [False],
            }
        ),
        block_switch_summary=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "switch_group": ["overall"],
                "n_switches_q1": [1.0],
                "n_switches_median": [2.0],
                "n_switches_q3": [3.0],
            }
        ),
        block_switch_block_points=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "switch_group": ["overall"],
                "n_switches": [2],
            }
        ),
        overall_df=pd.DataFrame(
            {
                "date": ["2025-12-05"],
                "actual_reward_collected": [20.0],
                "history_ideal_oracle_accuracy": [0.7],
                "history_ideal_mouse_agreement": [0.6],
                "history_ideal_reward_fraction": [0.75],
                "history_ideal_expected_reward": [26.7],
                "fixed_replay_ideal_reward_q1": [23.0],
                "fixed_replay_ideal_reward_median": [25.0],
                "fixed_replay_ideal_reward_q3": [28.0],
                "fixed_replay_ideal_oracle_accuracy_q1": [0.65],
                "fixed_replay_ideal_oracle_accuracy_median": [0.7],
                "fixed_replay_ideal_oracle_accuracy_q3": [0.76],
            }
        ),
        plot_path=tmp_path,
        figure_id="CT014",
        block_model_dict=block_model_dict,
        block_hmm_module=FakeBlockHmmModule,
        hmm_observation_plotter=fake_hmm_plotter,
        sliding_regression_plotter=fake_sliding_plotter,
        ttc_axis_plotter=lambda ax, **_kwargs: ax,
        correct_after_first_axis_plotter=lambda ax, **_kwargs: ax,
        block_switch_axis_plotter=lambda ax, **_kwargs: ax,
        ideal_choices_axis_plotter=lambda ax, **_kwargs: ax,
    )

    assert calls == {"hmm": 1, "sliding": 1}


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
            "prev_n_rewarded": [2, 4, 0],
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

    assert (tmp_path / "unit_session_scatter_trials-to-correct_prev_n_rewarded.png").exists()


def test_scatter_trials_to_correct_displays_unboxed_rounded_slope(monkeypatch, tmp_path: Path):
    """The standalone regression figure should display its rounded slope as plain text."""
    performance_plots = _import_performance_plots()
    saved_annotations = []

    def capture_annotations(fig, _save_path):
        saved_annotations.extend(fig.axes[0].texts)
        plt.close(fig)

    monkeypatch.setattr(performance_plots, "save_performance_figure", capture_annotations)

    performance_plots.scatter_trials_to_correct(
        block_performance=_make_scatter_block_performance(),
        slope=1.234,
        intercept=1.0,
        plot_path=tmp_path,
        figure_id="unit_session",
    )

    slope_text = next(text for text in saved_annotations if text.get_text().startswith("slope ="))
    assert slope_text.get_text() == "slope = 1.23"
    assert slope_text.get_bbox_patch() is None
    assert slope_text.get_fontsize() >= 12


def test_summary_regression_scatter_displays_unboxed_rounded_slope():
    """The summary-grid regression panel should use the same plain slope annotation."""
    performance_plots = _import_performance_plots()
    fig, ax = plt.subplots()

    performance_plots._plot_trials_to_correct_scatter_on_ax(
        ax=ax,
        block_performance=_make_scatter_block_performance(),
        slope=1.234,
        intercept=1.0,
    )

    slope_text = next(text for text in ax.texts if text.get_text().startswith("slope ="))
    assert slope_text.get_text() == "slope = 1.23"
    assert slope_text.get_bbox_patch() is None
    assert slope_text.get_fontsize() >= 12
    plt.close(fig)


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


def test_scatter_trials_to_correct_defaults_to_prev_n_rewarded(
    monkeypatch,
    tmp_path: Path,
):
    """The default scatter predictor should match the saved previous-block reward stats."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)

    performance_plots.scatter_trials_to_correct(
        block_performance=_make_scatter_block_performance(),
        slope=0.5,
        intercept=1.0,
        plot_path=tmp_path,
        figure_id="unit_session",
        point_jitter=0.0,
    )

    point_calls = [call for call in plot_calls if call["args"] == ("o",)]
    plotted_x = np.concatenate([call["x"] for call in point_calls])
    regression_call = next(call for call in plot_calls if call["args"] == ("k--",))

    np.testing.assert_array_equal(plotted_x, np.array([2.0, 4.0]))
    np.testing.assert_array_equal(regression_call["x"], np.array([0.0, 4.0]))
    np.testing.assert_array_equal(regression_call["y"], np.array([1.0, 3.0]))


@pytest.mark.parametrize(
    ("regressor_column", "expected_point_x", "expected_line_x"),
    [
        ("prev_consecutive_rewards", np.array([1.0, 3.0]), np.array([0.0, 3.0])),
        ("prev_n_correct", np.array([1.0, 2.0]), np.array([0.0, 2.0])),
    ],
)
def test_scatter_trials_to_correct_supports_alternate_regressors(
    monkeypatch,
    tmp_path: Path,
    regressor_column: str,
    expected_point_x: np.ndarray,
    expected_line_x: np.ndarray,
):
    """Requested scatter regressors should drive both marker positions and fit line."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)

    performance_plots.scatter_trials_to_correct(
        block_performance=_make_scatter_block_performance(),
        slope=0.5,
        intercept=1.0,
        plot_path=tmp_path,
        figure_id="unit_session",
        regressor_column=regressor_column,
        point_jitter=0.0,
    )

    point_calls = [call for call in plot_calls if call["args"] == ("o",)]
    plotted_x = np.concatenate([call["x"] for call in point_calls])
    regression_call = next(call for call in plot_calls if call["args"] == ("k--",))

    np.testing.assert_array_equal(plotted_x, expected_point_x)
    np.testing.assert_array_equal(regression_call["x"], expected_line_x)
    np.testing.assert_array_equal(
        regression_call["y"],
        0.5 * expected_line_x + 1.0,
    )


def test_scatter_trials_to_correct_rejects_unknown_regressor(tmp_path: Path):
    """Unsupported scatter regressors should fail before plotting misleading fits."""
    performance_plots = _import_performance_plots()

    with pytest.raises(ValueError, match="regressor_column"):
        performance_plots.scatter_trials_to_correct(
            block_performance=_make_scatter_block_performance(),
            slope=0.5,
            intercept=1.0,
            plot_path=tmp_path,
            figure_id="unit_session",
            regressor_column="prev_bad_metric",
        )


def test_scatter_trials_to_correct_applies_small_reproducible_jitter_to_points(
    monkeypatch,
    tmp_path: Path,
):
    """Trials-to-correct scatter points should move reproducibly near their integer values."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)

    performance_plots.scatter_trials_to_correct(
        block_performance=_make_scatter_block_performance(),
        slope=0.5,
        intercept=1.0,
        plot_path=tmp_path,
        figure_id="unit_session",
        point_jitter=0.1,
        jitter_seed=123,
    )

    point_calls = [call for call in plot_calls if call["args"] == ("o",)]
    plotted_x = np.concatenate([call["x"] for call in point_calls])
    plotted_y = np.concatenate([call["y"] for call in point_calls])
    original_x = np.array([2.0, 4.0])
    original_y = np.array([2.0, 4.0])

    assert not np.array_equal(plotted_x, original_x)
    assert not np.array_equal(plotted_y, original_y)
    assert np.all(np.abs(plotted_x - original_x) <= 0.1)
    assert np.all(np.abs(plotted_y - original_y) <= 0.1)


def test_scatter_trials_to_correct_keeps_regression_line_unjittered(
    monkeypatch,
    tmp_path: Path,
):
    """The fitted regression line should use true integer x-values, not jittered points."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)

    performance_plots.scatter_trials_to_correct(
        block_performance=_make_scatter_block_performance(),
        slope=0.5,
        intercept=1.0,
        plot_path=tmp_path,
        figure_id="unit_session",
        point_jitter=0.1,
        jitter_seed=123,
    )

    regression_call = next(call for call in plot_calls if call["args"] == ("k--",))

    np.testing.assert_array_equal(regression_call["x"], np.array([0.0, 4.0]))
    np.testing.assert_array_equal(regression_call["y"], np.array([1.0, 3.0]))


def test_plot_block_bias_quadrants_normalizes_csv_flag_strings(monkeypatch, tmp_path: Path):
    """CSV-loaded bias flags should plot as binary quadrant coordinates."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": ["right_cued", "left_cued"],
            "bias_inf_flag": ["False", "True"],
            "bias_rl_flag": ["True", "False"],
        }
    )

    performance_plots.plot_block_bias_quadrants(
        block_performance=block_performance,
        plot_path=tmp_path,
        figure_id="unit_session",
        point_jitter=0,
    )

    point_calls = [call for call in plot_calls if call["args"] == ("o",)]
    plotted_by_label = {call["kwargs"]["label"]: (call["x"].tolist(), call["y"].tolist()) for call in point_calls}

    assert plotted_by_label["right cued"] == ([0.0], [1.0])
    assert plotted_by_label["left cued"] == ([1.0], [0.0])


def test_plot_block_bias_quadrants_skips_missing_flags(monkeypatch, tmp_path: Path):
    """Blocks missing either bias flag should be excluded from quadrant plotting."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": ["right_cued", "left_cued", "right_uncued"],
            "bias_inf_flag": ["False", "None", "True"],
            "bias_rl_flag": ["False", "True", None],
        }
    )

    performance_plots.plot_block_bias_quadrants(
        block_performance=block_performance,
        plot_path=tmp_path,
        figure_id="unit_session",
        point_jitter=0,
    )

    point_calls = [call for call in plot_calls if call["args"] == ("o",)]

    assert len(point_calls) == 1
    assert point_calls[0]["kwargs"]["label"] == "right cued"
    np.testing.assert_array_equal(point_calls[0]["x"], np.array([0.0]))
    np.testing.assert_array_equal(point_calls[0]["y"], np.array([0.0]))


def test_plot_post_first_correct_accuracy_summary_saves_grouped_lines(monkeypatch, tmp_path: Path):
    """Post-first-correct accuracy should plot combined, left, and right curves."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    summary_df = pd.DataFrame(
        {
            "correct_group": ["combined", "combined", "left", "right"],
            "choice_trial_after_first_correct": [0, 1, 0, 0],
            "proportion_correct": [1.0, 0.5, 1.0, 1.0],
            "n_blocks": [2, 2, 1, 1],
        }
    )

    output_path = performance_plots.plot_post_first_correct_accuracy_summary(
        summary_df=summary_df,
        plot_path=tmp_path,
        figure_id="unit_session",
    )

    calls_by_label = {call["label"]: call for call in plot_calls}
    assert output_path == tmp_path / "unit_session_post_first_correct_accuracy.png"
    assert output_path.exists()
    assert calls_by_label["combined"]["x"] == [0, 1]
    assert calls_by_label["combined"]["y"] == [1.0, 0.5]
    assert calls_by_label["left"]["color"] == performance_plots.color_dict["left_uncued"]
    assert calls_by_label["right"]["color"] == performance_plots.color_dict["right_uncued"]


def test_plot_session_side_bias_ratios_draws_true_and_ideal_side_ratios(monkeypatch, tmp_path: Path):
    """Session side-bias plotting should show left/right ratios for both references."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    session_summary = pd.DataFrame(
        {
            "left_choice_per_true_left": [1.5],
            "right_choice_per_true_right": [0.5],
            "left_choice_per_ideal_left": [2.0],
            "right_choice_per_ideal_right": ["None"],
        }
    )

    output_path = performance_plots.plot_session_side_bias_ratios(
        session_summary=session_summary,
        plot_path=tmp_path,
        sess_id_full="unit_session",
    )

    calls_by_label = {call["label"]: call for call in plot_calls}
    assert output_path == tmp_path / "unit_session_session_side_bias_ratios.png"
    assert output_path.exists()
    assert calls_by_label["left"]["x"] == [0.85, 1.85]
    assert calls_by_label["left"]["y"] == [1.5, 2.0]
    assert calls_by_label["right"]["x"] == [1.15]
    assert calls_by_label["right"]["y"] == [0.5]
    assert calls_by_label["left"]["color"] == performance_plots.color_dict["left_uncued"]
    assert calls_by_label["right"]["color"] == performance_plots.color_dict["right_uncued"]


def test_plot_session_zero_trials_to_correct_fraction_draws_raw_flags_and_means(
    monkeypatch,
    tmp_path: Path,
):
    """Zero-TTC plotting should show raw block flags and mean fractions."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3, 4],
            "block_type": ["right_uncued", "left_uncued", "left_cued", "right_cued", "right_uncued"],
            "trials_to_correct": [5, 0, 2, 0, 6],
        }
    )

    output_path = performance_plots.plot_session_zero_trials_to_correct_fraction(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_id_full="unit_session",
        point_jitter=0,
    )

    calls_by_label = {call["label"]: call for call in plot_calls}
    assert output_path == tmp_path / "unit_session_zero_trials_to_correct_fraction.png"
    assert output_path.exists()
    assert calls_by_label["left raw blocks"]["x"] == [0.0, 0.0]
    assert calls_by_label["left raw blocks"]["y"] == [1.0, 0.0]
    assert calls_by_label["right raw blocks"]["x"] == [1.0, 1.0]
    assert calls_by_label["right raw blocks"]["y"] == [1.0, 0.0]
    assert calls_by_label["overall raw blocks"]["x"] == [2.0, 2.0, 2.0, 2.0]
    assert calls_by_label["overall raw blocks"]["y"] == [1.0, 0.0, 1.0, 0.0]
    assert calls_by_label["left mean"]["y"] == [0.5]
    assert calls_by_label["right mean"]["y"] == [0.5]
    assert calls_by_label["overall mean"]["y"] == [0.5]


def test_plot_cross_mouse_learning_curve_draws_mouse_lines_and_group_mean(monkeypatch, tmp_path: Path):
    """Cross-mouse learning curves should include individual mice and day means."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    cross_mouse_df = pd.DataFrame(
        {
            "mouse": ["CT014", "CT014", "CT016", "CT016"],
            "training_day": [1, 3, 1, 2],
            "slope": [0.1, 0.3, 0.2, 0.4],
            "learning_regressor": ["prev_n_rewarded"] * 4,
        }
    )

    output_path = performance_plots.plot_cross_mouse_learning_curve(
        cross_mouse_df=cross_mouse_df,
        plot_path=tmp_path,
        figure_id="cross_mouse",
        learning_regressor="prev_n_rewarded",
    )

    calls_by_label = {call["label"]: call for call in plot_calls}
    assert output_path == tmp_path / "cross_mouse_prev_n_rewarded_learning_curve.png"
    assert output_path.exists()
    assert calls_by_label["CT014"]["x"] == [1, 3]
    assert calls_by_label["CT014"]["y"] == [0.1, 0.3]
    assert calls_by_label["CT016"]["x"] == [1, 2]
    assert calls_by_label["group mean"]["x"] == [1, 2, 3]
    assert calls_by_label["group mean"]["y"] == [0.15000000000000002, 0.4, 0.3]
    assert calls_by_label["group mean"]["color"] == "black"
    assert calls_by_label["group mean"]["linewidth"] > calls_by_label["CT014"]["linewidth"]


def test_plot_cross_mouse_session_metric_curve_draws_mouse_lines_and_group_mean(
    monkeypatch,
    tmp_path: Path,
):
    """Cross-mouse session metric curves should mirror learning-curve styling."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    cross_mouse_df = pd.DataFrame(
        {
            "mouse": ["CT014", "CT014", "CT016", "CT016"],
            "training_day": [1, 3, 1, 2],
            "metric_value": [1.0, 3.0, 2.0, 4.0],
            "metric_column": ["median_TTS"] * 4,
            "metric_label": ["Median Trials to Correct"] * 4,
        }
    )

    output_path = performance_plots.plot_cross_mouse_session_metric_curve(
        cross_mouse_df=cross_mouse_df,
        plot_path=tmp_path,
        figure_id="cross_mouse",
        metric_column="median_TTS",
        metric_label="Median Trials to Correct",
    )

    calls_by_label = {call["label"]: call for call in plot_calls}
    assert output_path == tmp_path / "cross_mouse_median_TTS_session_metric_curve.png"
    assert output_path.exists()
    assert calls_by_label["CT014"]["x"] == [1, 3]
    assert calls_by_label["CT014"]["y"] == [1.0, 3.0]
    assert calls_by_label["CT016"]["x"] == [1, 2]
    assert calls_by_label["group mean"]["x"] == [1, 2, 3]
    assert calls_by_label["group mean"]["y"] == [1.5, 4.0, 3.0]
    assert calls_by_label["group mean"]["color"] == "black"
    assert calls_by_label["group mean"]["linewidth"] > calls_by_label["CT014"]["linewidth"]


def test_plot_session_signed_side_bias_draws_three_bias_metrics(monkeypatch, tmp_path: Path):
    """Signed side-bias plotting should show oracle, ideal, and raw bias metrics."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    session_summary = pd.DataFrame(
        {
            "bias_oracle": [0.25],
            "bias_ideal": [-0.1],
            "raw_side_bias": [0.5],
        }
    )

    output_path = performance_plots.plot_session_signed_side_bias(
        session_summary=session_summary,
        plot_path=tmp_path,
        sess_id_full="unit_session",
    )

    metric_call = next(call for call in plot_calls if call["label"] == "signed bias")
    assert output_path == tmp_path / "unit_session_session_signed_side_bias.png"
    assert output_path.exists()
    assert metric_call["x"] == [0, 1, 2]
    assert metric_call["y"] == [0.25, -0.1, 0.5]


def test_plot_session_agent_mouse_agreement_uses_short_ylabel(monkeypatch, tmp_path: Path):
    """Single-session agent agreement should use a compact y-axis label."""
    performance_plots = _import_performance_plots()
    captured = {}

    def capture_figure(fig, save_path):
        captured["ylabel"] = fig.axes[0].get_ylabel()
        fig.savefig(save_path)
        plt.close(fig)

    monkeypatch.setattr(performance_plots, "save_performance_figure", capture_figure)
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "qlearning_mouse_agreement": [0.5, 0.6],
            "fql_mouse_agreement": [0.6, 0.7],
            "hmm_logodds_mouse_agreement": [0.7, 0.8],
            "hmm_logodds_decay_mouse_agreement": [0.8, 0.9],
            "perseveration_mouse_agreement": [0.4, 0.5],
            "doubt_perseveration_mouse_agreement": [0.3, 0.4],
            "wsls_mouse_agreement": [0.5, 0.5],
            "observer_mouse_agreement": [0.9, 1.0],
        }
    )

    performance_plots.plot_session_agent_mouse_agreement(
        block_performance=block_performance,
        plot_path=tmp_path,
        sess_id_full="unit_session",
    )

    assert captured["ylabel"] == "Agent agreement"
    assert (tmp_path / "unit_session_agent_mouse_agreement.png").exists()


def test_plot_session_summary_metric_family_draws_multiple_session_metrics(monkeypatch, tmp_path: Path):
    """Metric-family plots should combine related session summary columns."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_plot_calls(monkeypatch)
    overall_df = pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-02"],
            "median_TTS": [1.0, 2.0],
            "q3_TTS": [3.0, 4.0],
            "frac_blocks_TTS_gt_5": [0.25, 0.5],
        }
    )

    output_path = performance_plots.plot_session_summary_metric_family(
        overall_df=overall_df,
        plot_path=tmp_path,
        figure_id="CT014",
        family_name="tts",
        metric_specs={
            "median_TTS": "median",
            "q3_TTS": "Q3",
            "frac_blocks_TTS_gt_5": "fraction > 5",
        },
        y_label="TTS summary",
    )

    labels = {call["label"] for call in plot_calls}
    assert output_path == tmp_path / "CT014_tts-session-summary-metrics.png"
    assert output_path.exists()
    assert labels == {"median", "Q3", "fraction > 5"}


def test_plot_block_bias_quadrants_colors_by_block_type(monkeypatch, tmp_path: Path):
    """Quadrant points should keep the existing block-type color convention."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": ["right_cued", "left_uncued"],
            "bias_inf_flag": [False, True],
            "bias_rl_flag": [False, True],
        }
    )

    performance_plots.plot_block_bias_quadrants(
        block_performance=block_performance,
        plot_path=tmp_path,
        figure_id="unit_session",
        point_jitter=0,
    )

    point_calls = [call for call in plot_calls if call["args"] == ("o",)]
    colors_by_label = {call["kwargs"]["label"]: call["kwargs"]["color"] for call in point_calls}

    assert colors_by_label == {
        "right cued": performance_plots.color_dict["right_cued"],
        "left uncued": performance_plots.color_dict["left_uncued"],
    }


def test_plot_block_bias_quadrants_uses_deterministic_jitter(monkeypatch, tmp_path: Path):
    """Overlapping quadrant points should be reproducibly jittered for visibility."""
    performance_plots = _import_performance_plots()
    plot_calls = _capture_detailed_plot_calls(monkeypatch)
    block_performance = pd.DataFrame(
        {
            "block_type": ["right_cued", "right_cued"],
            "bias_inf_flag": [True, True],
            "bias_rl_flag": [False, False],
        }
    )

    performance_plots.plot_block_bias_quadrants(
        block_performance=block_performance,
        plot_path=tmp_path,
        figure_id="unit_session",
        point_jitter=0.1,
        jitter_seed=123,
    )

    point_call = next(call for call in plot_calls if call["args"] == ("o",))

    assert not np.array_equal(point_call["x"], np.array([1.0, 1.0]))
    assert not np.array_equal(point_call["y"], np.array([0.0, 0.0]))
    assert np.all(np.abs(point_call["x"] - np.array([1.0, 1.0])) <= 0.1)
    assert np.all(np.abs(point_call["y"] - np.array([0.0, 0.0])) <= 0.1)


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
    original_scatter = Axes.scatter

    def capture_scatter(self, x, y, *args, **kwargs):
        scatter_calls.append(kwargs)
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)

    performance_plots.plot_block_hmm_state_feature_scatter(
        state_features_df=state_features,
        plot_path=tmp_path,
        figure_id="CT999",
        use_block_count_marker_size=False,
        fixed_marker_size=70.0,
    )

    assert scatter_calls[0]["s"] == 70.0


def test_plot_block_hmm_state_feature_scatter_can_use_default_marker_mode(
    monkeypatch,
    tmp_path: Path,
):
    """Default marker mode should use fixed size and fixed alpha for all states."""
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
    original_scatter = Axes.scatter

    def capture_scatter(self, x, y, *args, **kwargs):
        scatter_calls.append(kwargs)
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)

    performance_plots.plot_block_hmm_state_feature_scatter(
        state_features_df=state_features,
        plot_path=tmp_path,
        figure_id="CT999",
        marker_mode="default",
        fixed_marker_size=70.0,
        fixed_marker_alpha=0.6,
    )

    marker_colors = np.asarray(scatter_calls[0]["c"])
    assert scatter_calls[0]["s"] == 70.0
    assert marker_colors.shape == (2, 4)
    assert marker_colors[:, 3].tolist() == [0.6, 0.6]


def test_plot_block_hmm_state_feature_scatter_can_use_alpha_marker_mode(
    monkeypatch,
    tmp_path: Path,
):
    """Alpha marker mode should make higher-count states more opaque."""
    from matplotlib.axes import Axes

    performance_plots = _import_performance_plots()
    state_features = pd.DataFrame(
        {
            "fit_type": ["map", "map", "map"],
            "bias": [-0.5, 0.25, 0.6],
            "prev_n_rewarded_weight": [1.0, -0.25, 0.1],
            "state_block_count": [4, 12, 20],
        }
    )
    scatter_calls = []
    original_scatter = Axes.scatter

    def capture_scatter(self, x, y, *args, **kwargs):
        scatter_calls.append(kwargs)
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)

    performance_plots.plot_block_hmm_state_feature_scatter(
        state_features_df=state_features,
        plot_path=tmp_path,
        figure_id="CT999",
        marker_mode="alpha",
        fixed_marker_size=70.0,
        min_marker_alpha=0.2,
        max_marker_alpha=0.9,
    )

    marker_colors = np.asarray(scatter_calls[0]["c"])
    assert scatter_calls[0]["s"] == 70.0
    assert marker_colors[:, 3].tolist() == pytest.approx([0.2, 0.55, 0.9])


def test_plot_block_hmm_state_feature_scatter_rejects_invalid_marker_mode(
    tmp_path: Path,
):
    """Unknown marker modes should fail before plotting."""
    performance_plots = _import_performance_plots()
    state_features = pd.DataFrame(
        {
            "fit_type": ["map"],
            "bias": [0.25],
            "prev_n_rewarded_weight": [-0.25],
            "state_block_count": [20],
        }
    )

    with pytest.raises(ValueError, match="marker_mode"):
        performance_plots.plot_block_hmm_state_feature_scatter(
            state_features_df=state_features,
            plot_path=tmp_path,
            figure_id="CT999",
            marker_mode="bad_mode",
        )
