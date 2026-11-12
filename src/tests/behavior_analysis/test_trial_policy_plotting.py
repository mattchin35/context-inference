"""Saved-result trial diagnostics preserve ties, exclusions and session separation."""

from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def capture_figure(monkeypatch):
    """Capture actual Matplotlib Figure objects while still writing real PNGs."""
    captured = []
    original = Figure.savefig

    def capture(self, *args, **kwargs):
        """Figure and standard savefig args produce an actual saved image and capture."""
        captured.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return captured


def diagnostic_result(models=3, smoothing=True, session_id="source"):
    """Return all-tied numerical result (T=5,K) plus plain diagnostic metadata."""
    from src.behavior_analysis import trial_policy_analysis as analysis
    from src.behavior_analysis.trial_policy_models import DEFAULT_MODEL_IDS, prepare_trial_policy_history

    table = pd.DataFrame({"cur_trial": [2, 3, 5, 6, 7], "cur_block": [1, 1, 2, 2, 2],
                          "action": [0, 1, "no_choice", 0, 1], "reward": [1, 0, 0, 1, 0]})
    history = prepare_trial_policy_history(table, session_id=session_id)
    ids = (*DEFAULT_MODEL_IDS, "synthetic_fourth")[:models]
    result = analysis.compute_trial_policy_comparison(history, np.zeros((5, models)),
                                                       np.ones((5, models), bool), ids)
    if smoothing:
        result = analysis.smooth_trial_policy_comparison(result)
    return result, {"session_id": session_id, "model_ids": list(ids), "status": result.status,
                    "parameters": {"opposite_action_error_prob": .1, "stay_prob": .9, "smoothing": smoothing}}


@pytest.mark.parametrize("smoothing,panels", [(True, 5), (False, 3)])
@pytest.mark.parametrize("models", [1, 3, 4, 6, 7, 9])
def test_saved_result_plot_panels_tied_winners_and_excluded_rows(tmp_path, capture_figure, monkeypatch, smoothing, panels, models):
    """Numerical artifacts yield opaque PNG panels without any inference call."""
    from src.behavior_analysis import trial_policy_plotting as plotting
    from src.behavior_analysis import trial_policy_analysis as analysis

    result, metadata = diagnostic_result(models, smoothing)

    def reject_inference(*args, **kwargs):
        """Arbitrary numerical call arguments fail; plotting consumes saved arrays only."""
        pytest.fail("Plotting must not rerun inference")

    monkeypatch.setattr(analysis, "compute_trial_policy_comparison", reject_inference)
    monkeypatch.setattr(analysis, "smooth_trial_policy_comparison", reject_inference)
    path = plotting.plot_trial_policy_diagnostic(result, metadata, output_path=tmp_path / "figure.png")
    assert path.exists() and path.stat().st_size > 1000
    figure = capture_figure[-1]
    assert len(figure.axes) == panels
    assert figure.get_facecolor() == (1., 1., 1., 1.)
    assert all(axis.get_facecolor() == (1., 1., 1., 1.) for axis in figure.axes)
    assert any(patch.get_gid() == "excluded-row" for patch in figure.axes[0].patches)
    assert any(line.get_gid() == "block-boundary" for line in figure.axes[0].lines)
    assert len(figure.axes[2].images) == 1
    cells = np.asarray(figure.axes[2].images[0].get_array())
    assert cells.shape == (models, 5, 4)
    for model, model_id in enumerate(result.model_ids):
        np.testing.assert_allclose(cells[model, [0, 1, 3, 4]],
                                   np.tile(to_rgba(plotting._model_color(str(model_id))), (4, 1)))
    np.testing.assert_allclose(cells[:, 2], np.tile(to_rgba("#d9d9d9"), (models, 1)))
    caption = " ".join(" ".join(text.get_text() for text in figure.texts).lower().split())
    assert "0.1" in caption and "tie" in caption
    if smoothing:
        assert "retrospective" in caption
    else:
        assert "smoothing disabled" in caption
        assert "hmm resets" not in caption and "hmm states there" not in caption
    assert "excluded" in caption
    assert "source" in caption
    choices = next(collection for collection in figure.axes[0].collections if collection.get_label() == "Mouse choice")
    np.testing.assert_array_equal(choices.get_offsets(), [[0, 1], [1, 0], [3, 1], [4, 0]])


def test_membership_grid_has_model_rows_for_every_tie_combination(tmp_path, capture_figure):
    """Path/Figure fixtures show unique/partial/all ties with exactly six rows."""
    from src.behavior_analysis import trial_policy_plotting as plotting

    result, metadata = diagnostic_result(models=6)
    winners = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 0, 1, 0]], dtype=bool)
    smoothed = winners.copy()
    smoothed[2] = [0, 0, 0, 0, 0, 1]
    result = replace(result, independent_best_model_mask=winners, hmm_smoothed_best_model_mask=smoothed)
    plotting.plot_trial_policy_diagnostic(result, metadata, output_path=tmp_path / "ties.png")
    figure = capture_figure[-1]
    for axis, mask in ((figure.axes[2], winners), (figure.axes[4], smoothed)):
        assert len(axis.get_yticks()) == 6 and len(axis.images) == 1
        assert not axis.collections
        image = axis.images[0]
        assert image.get_interpolation() == "nearest"
        np.testing.assert_allclose(image.get_extent(), [-.5, 4.5, -.5, 5.5])
        cells = np.asarray(image.get_array())
        for model, model_id in enumerate(result.model_ids):
            expected = np.tile(to_rgba("white"), (5, 1))
            expected[2] = to_rgba("#d9d9d9")
            expected[mask[:, model]] = to_rgba(plotting._model_color(str(model_id)))
            np.testing.assert_allclose(cells[model], expected)
    caption = " ".join(text.get_text() for text in figure.texts).lower()
    assert "filled" in caption and "column" in caption


def test_persistence_has_a_distinct_labeled_seventh_row(tmp_path, capture_figure):
    """Path/Figure fixtures preserve a distinct persistence color and seven rows."""
    from src.behavior_analysis import trial_policy_plotting as plotting

    result, metadata = diagnostic_result(models=7)
    plotting.plot_trial_policy_diagnostic(result, metadata, output_path=tmp_path / "seven.png")
    figure = capture_figure[-1]
    assert len({plotting._model_color(str(model_id)) for model_id in result.model_ids}) == 7
    assert "Persistence" in [label.get_text() for label in figure.axes[2].get_yticklabels()]
    assert len(figure.axes[2].get_yticks()) == len(figure.axes[4].get_yticks()) == 7
    assert np.asarray(figure.axes[2].images[0].get_array()).shape == (7, 5, 4)


def test_caption_distinguishes_logistic_temperatures_from_discrete_errors(tmp_path, capture_figure):
    """Path/Figure fixtures show effective graded settings without a shared 90/10 claim."""
    from src.behavior_analysis import trial_policy_plotting as plotting
    from src.behavior_analysis.trial_policy_models import (
        effective_trial_policy_choice_parameters, resolve_trial_policy_candidates,
    )

    result, metadata = diagnostic_result(models=7)
    metadata["candidate_choice_parameters"] = effective_trial_policy_choice_parameters(resolve_trial_policy_candidates())
    plotting.plot_trial_policy_diagnostic(result, metadata, output_path=tmp_path / "graded.png")
    caption = " ".join(" ".join(text.get_text() for text in capture_figure[-1].texts).lower().split())
    assert "logistic" in caption and "temperature" in caption and "0.14" in caption and "0.4" in caption
    assert "discrete" in caption and "0.1" in caption
    assert "shared opposite-action error probability" not in caption


def test_probe_models_have_distinct_labeled_rows(tmp_path, capture_figure):
    """Path/Figure fixtures label nine candidate rows with stable distinct colors."""
    from src.behavior_analysis import trial_policy_plotting as plotting
    from src.behavior_analysis.trial_policy_models import (
        effective_trial_policy_choice_parameters, resolve_trial_policy_candidates,
    )

    result, metadata = diagnostic_result(models=9)
    metadata["candidate_choice_parameters"] = effective_trial_policy_choice_parameters(resolve_trial_policy_candidates())
    plotting.plot_trial_policy_diagnostic(result, metadata, output_path=tmp_path / "nine.png")
    figure = capture_figure[-1]
    assert len({plotting._model_color(str(model_id)) for model_id in result.model_ids}) == 9
    assert [label.get_text() for label in figure.axes[2].get_yticklabels()][-2:] == ["Probe doubt", "Home expectancy"]
    assert len(figure.axes[2].get_yticks()) == len(figure.axes[4].get_yticks()) == 9
    assert np.asarray(figure.axes[2].images[0].get_array()).shape == (9, 5, 4)
    caption = " ".join(" ".join(text.get_text() for text in figure.texts).lower().split())
    assert "0.5" in caption and "probe doubt" in caption and "home expectancy" in caption


def test_model_colors_are_stable_under_candidate_reordering(tmp_path, capture_figure):
    """Stable model IDs determine trace colors independently of selected order."""
    from src.behavior_analysis import trial_policy_plotting as plotting

    result, metadata = diagnostic_result()
    plotting.plot_trial_policy_diagnostic(result, metadata, output_path=tmp_path / "first.png")
    first = {line.get_label(): line.get_color() for line in capture_figure[-1].axes[0].lines if line.get_label() in result.model_ids}
    permutation = [2, 0, 1]
    changes = {}
    for name in ("model_ids",):
        changes[name] = getattr(result, name)[permutation]
    for name in ("p_right", "log_likelihoods", "independent_best_model_mask", "hmm_smoothed_probs", "hmm_smoothed_best_model_mask"):
        changes[name] = getattr(result, name)[:, permutation]
    permuted = replace(result, **changes)
    plotting.plot_trial_policy_diagnostic(permuted, metadata, output_path=tmp_path / "second.png")
    second = {line.get_label(): line.get_color() for line in capture_figure[-1].axes[0].lines if line.get_label() in result.model_ids}
    assert first == second


def test_multisession_traces_break_at_source_boundaries(tmp_path, capture_figure):
    """Two source-local results share display positions but no connecting traces."""
    from src.behavior_analysis import trial_policy_plotting as plotting

    first, first_meta = diagnostic_result(session_id="first")
    second, second_meta = diagnostic_result(session_id="second")
    path = plotting.plot_multisession_trial_policy_diagnostic(
        [first, second], [first_meta, second_meta], output_path=tmp_path / "joint.png")
    assert path.exists()
    figure = capture_figure[-1]
    for axis in figure.axes:
        assert any(line.get_gid() == "session-boundary" and line.get_xdata()[0] == 4.5 for line in axis.lines)
        for line in axis.lines:
            if line.get_gid() in ("session-boundary", "block-boundary"):
                continue
            values = np.asarray(line.get_xdata())
            if len(values):
                assert not (np.min(values) < 5 and np.max(values) >= 5)
    assert "first" in " ".join(text.get_text() for text in figure.texts)
    assert "second" in " ".join(text.get_text() for text in figure.texts)


def test_empty_and_no_comparison_figures_explain_evidence_status(tmp_path, capture_figure):
    """Prior-only/empty results remain renderable and visibly report their status."""
    from src.behavior_analysis import trial_policy_plotting as plotting
    from src.behavior_analysis import trial_policy_analysis as analysis
    from src.behavior_analysis.trial_policy_models import prepare_trial_policy_history

    for rows, status in ((0, "empty"), (3, "no_valid_comparisons")):
        history = prepare_trial_policy_history(pd.DataFrame({"cur_trial": range(rows), "cur_block": [0] * rows,
                                                            "action": [None] * rows, "reward": [None] * rows}), session_id="empty_source")
        result = analysis.compute_trial_policy_comparison(history, np.zeros((rows, 1)), np.ones((rows, 1), bool), ("single",))
        result = analysis.smooth_trial_policy_comparison(result)
        metadata = {"session_id": "empty_source", "status": status, "parameters": {"opposite_action_error_prob": .1, "stay_prob": .9}}
        plotting.plot_trial_policy_diagnostic(result, metadata, output_path=tmp_path / f"{status}.png")
        texts = [text.get_text().lower() for text in capture_figure[-1].texts]
        texts += [text.get_text().lower() for axis in capture_figure[-1].axes for text in axis.texts]
        assert any(status in text or "no comparison" in text for text in texts)


def test_thirty_source_caption_does_not_overlap_last_axis(tmp_path, capture_figure):
    """An ordinary 30-session display retains readable source provenance below axes."""
    from src.behavior_analysis import trial_policy_plotting as plotting

    original, meta = diagnostic_result()
    results, metadata = [], []
    for index in range(30):
        identity = f"CT026_2026-05-{index + 1:02d}_123456"
        results.append(replace(original, session_id=identity))
        metadata.append(dict(meta, session_id=identity))
    plotting.plot_multisession_trial_policy_diagnostic(results, metadata, output_path=tmp_path / "thirty_sources.png")
    figure = capture_figure[-1]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    caption = max(figure.texts, key=lambda text: len(text.get_text()))
    caption_top = caption.get_window_extent(renderer).y1
    axis_bottom = figure.axes[-1].get_tightbbox(renderer).y0
    assert caption_top + 6 < axis_bottom
