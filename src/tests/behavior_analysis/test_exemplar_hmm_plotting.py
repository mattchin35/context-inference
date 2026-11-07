"""Tests for neutral regression and Dynamax exemplar diagnostic plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis import block_regression, exemplar_hmm_plotting


def test_sliding_regression_uses_valid_blocks_and_preserves_constant_slope():
    """Regression uses valid rows, valid-axis positions, and OLS units."""
    frame = pd.DataFrame({
        "block_ix": [10, 11, 12, 13, 14, 15],
        "trials_to_correct": [2, 999, 4, 6, 8, 10],
        "prev_n_correct": [1, 2, 3, 4, 5, 6],
        "prev_n_rewarded": [0, 1, 2, 3, 4, 5],
    })
    mask = np.array([True, False, True, True, True, True])
    regression = block_regression.compute_sliding_block_regression(
        frame, valid_mask=mask, window_size=3, step_size=2
    )
    assert regression["window_start_position"].tolist() == [0, 2]
    assert regression["window_end_block_ix"].tolist() == [13, 15]
    np.testing.assert_allclose(regression["prev_n_rewarded_weight"], [9.0 / 7.0, 2.0])
    np.testing.assert_allclose(regression["window_intercept"], [13.0 / 7.0, 0.0])

    frame["prev_n_rewarded"] = 4
    constant = block_regression.compute_sliding_block_regression(
        frame, valid_mask=mask, window_size=3, step_size=1
    )
    assert constant["prev_n_rewarded_weight"].tolist() == [0.0, 0.0, 0.0]


def test_default_validity_requires_both_current_and_previous_history():
    """Default mask follows TTC and canonical previous-correct presence."""
    frame = pd.DataFrame({
        "trials_to_correct": [1, np.nan, 3, 4],
        "prev_n_correct": [np.nan, 2, "None", 4],
    })
    assert block_regression.make_valid_block_history_mask(frame).tolist() == [False, False, False, True]


def test_exemplar_plot_uses_full_rows_and_disambiguates_sessions(tmp_path: Path, monkeypatch):
    """Diagnostic axes map valid values to full block ordinal positions."""
    frame = pd.DataFrame({
        "block_ix": [0, 1, 0, 1],
        "date": ["2024-01-01"] * 4,
        "source_session_id": ["mouse_2024-01-01_a"] * 2 + ["mouse_2024-01-01_b"] * 2,
        "exemplar_z_residual_A": [0.1, np.nan, 0.3, 0.4],
        "exemplar_log_likelihood_A": [-1.0, np.nan, -3.0, -4.0],
        "exemplar_independent_prob_A": [0.8, np.nan, 0.4, 0.2],
        "exemplar_hmm_smoothed_prob_A": [0.7, np.nan, 0.5, 0.3],
        "exemplar_hmm_smoothed_state": [0, np.nan, 1, 1],
        "exemplar_ll_argmax_state": [0, np.nan, 1, 1],
    })
    captured = {}
    original_savefig = plt.Figure.savefig

    def capture_figure(self, *args, **kwargs):
        captured["axes"] = self.axes
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_figure)
    path = exemplar_hmm_plotting.plot_exemplar_block_diagnostic(frame, ["A"], tmp_path, "mouse")
    assert path.exists()
    first_axis = captured["axes"][0]
    assert list(first_axis.lines[0].get_xdata()) == [0, 1, 2, 3]
    assert np.isnan(first_axis.lines[0].get_ydata()[1])
    assert "block" in first_axis.get_xlabel().lower()
    assert len(captured["axes"]) == 4
    assert captured["axes"][2].get_yticklabels()[0].get_text() == "A"
    assert captured["axes"][3].get_ylim() == (0.0, 1.0)
    assert any("mouse_2024-01-01_b" in text.get_text() for text in captured["axes"][0].texts)


def test_exemplar_plot_handles_all_missing_rows(tmp_path: Path):
    """Empty modeled values produce explicit placeholder panels and a file."""
    frame = pd.DataFrame({"block_ix": [0, 1]})
    first = exemplar_hmm_plotting.plot_exemplar_block_diagnostic(frame, ["A"], tmp_path, "empty")
    second = exemplar_hmm_plotting.plot_exemplar_block_diagnostic(frame, ["A"], tmp_path, "empty")
    assert first.exists() and second.exists() and first != second


def test_empty_diagnostic_marks_every_panel_as_missing(tmp_path: Path, monkeypatch):
    """All-NaN model arrays are visibly identified as missing data."""
    captured = {}
    original_savefig = plt.Figure.savefig

    def capture_figure(self, *args, **kwargs):
        captured["axes"] = self.axes
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_figure)
    frame = pd.DataFrame({"exemplar_z_residual_A": [np.nan, np.nan]})
    exemplar_hmm_plotting.plot_exemplar_block_diagnostic(frame, ["A"], tmp_path, "empty")
    assert all(any("No valid modeled block data" in text.get_text() for text in ax.texts)
               for ax in captured["axes"][:3])
