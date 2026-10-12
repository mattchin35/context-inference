"""Saved-result-only reporting contracts for pooled decoding."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.neural_analysis.task_decoding import activity, modeling
from src.neural_analysis.task_decoding.pooled import results
from src.neural_analysis.task_decoding.pooled.config import TrainingWindow
from src.tests.neural_analysis.task_decoding.pooled.test_results import (
    make_result_template,
)


def _fail_if_called(*_args: object, **_kwargs: object) -> None:
    """Reject any scientific loading or fitting during saved-result plotting.

    Parameters
    ----------
    *_args, **_kwargs : object
        Arbitrary arguments from a mistakenly invoked scientific dependency.

    Returns
    -------
    None
        This test helper always raises ``AssertionError``.
    """
    raise AssertionError("plotting must not load spikes, tensors, or estimators")


def _all_figure_text(figure: object) -> str:
    """Collect visible Matplotlib labels for compact caption assertions.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        In-memory pooled time-course figure.

    Returns
    -------
    str
        Lower-case concatenation of figure and axes text labels.
    """
    labels: list[str] = []
    labels.extend(text.get_text() for text in figure.texts)
    for axis in figure.axes:
        labels.extend(
            (
                axis.get_title(),
                axis.get_xlabel(),
                axis.get_ylabel(),
                *(text.get_text() for text in axis.texts),
            )
        )
        legend = axis.get_legend()
        if legend is not None:
            labels.extend(text.get_text() for text in legend.get_texts())
    for legend in figure.legends:
        labels.extend(text.get_text() for text in legend.get_texts())
    return " ".join(labels).lower()


def _full_condition_saved_result() -> dict[str, object]:
    """Create a valid full-cardinality categorical result for layout tests.

    Returns
    -------
    dict[str, object]
        Validated saved-result mapping with one window, six overlapping
        conditions, eleven evaluation pairs, two folds, six panels, eight
        evaluation-time bins in seconds, and finite categorical scores.
    """
    conditions = (
        "all",
        "correct_rewarded",
        "omission",
        "incorrect",
        "switch",
        "stay",
    )
    arrays, metadata = results.initialize_pooled_result_arrays(
        training_windows=(
            TrainingWindow(
                "pre_choice",
                -0.5,
                0.0,
                np.array([2, 3], dtype=np.int64),
            ),
        ),
        condition_names=conditions,
        target_names=("current_action",),
        target_family_names=("categorical",),
        target_unit_names=("encoded class",),
        outer_fold_count=2,
        evaluation_time_centers_s=np.linspace(-0.875, 0.875, 8),
        time_bin_edges_s=np.linspace(-1.0, 1.0, 9),
        trial_positions=np.arange(6, dtype=np.int64),
        trial_ids=np.arange(101, 107, dtype=np.int64),
        behavioral_block_ids=np.array([1, 1, 2, 2, 3, 3], dtype=np.int64),
        trial_row_indices=np.arange(6, dtype=np.int64),
        pfc_unit_ids=("pfc:1", "pfc:2"),
        hpc_unit_ids=("hpc:1", "hpc:2"),
        pfc_pc_count=2,
        hpc_pc_count=2,
        unit_selection_rules=({"region": "PFC"}, {"region": "HPC"}),
        target_estimator_class_names=("LogisticRegression",),
        target_estimator_parameters=({"C": 1.0},),
        run_fingerprint="full-condition-run",
        source_fingerprint="full-condition-source",
        random_seed=1729,
    )
    for metric_index, offset in ((0, 0.55), (1, 0.65)):
        arrays["evaluation_status"][..., metric_index] = 1
        arrays["evaluation_reason_index"][..., metric_index] = -1
        arrays["evaluation_scores"][..., metric_index] = (
            offset + np.linspace(0.0, 0.1, 8)
        )
        arrays["gap_status"][..., metric_index] = 1
        arrays["gap_reason_index"][..., metric_index] = -1
        arrays["gap_scores"][..., metric_index] = np.linspace(-0.05, 0.05, 8)
    results.validate_pooled_result_arrays(arrays, metadata)
    return {"arrays": arrays, "metadata": metadata}


def _assert_artist_inside(outer: object, inner: object) -> None:
    """Assert that one display-coordinate bounding box contains another.

    Parameters
    ----------
    outer, inner : matplotlib.transforms.Bbox
        Outer and inner rectangles in the same display-coordinate units.

    Returns
    -------
    None
        Assertions fail when any inner edge crosses the outer rectangle.
    """
    assert inner.x0 >= outer.x0
    assert inner.y0 >= outer.y0
    assert inner.x1 <= outer.x1
    assert inner.y1 <= outer.y1


def test_full_transfer_uses_one_contained_shared_legend_without_overlap() -> None:
    """The worst-cardinality transfer legend must not cover data or caption."""
    from src.neural_analysis.task_decoding.pooled import plotting

    figure = plotting.plot_pooled_timecourses(
        _full_condition_saved_result(),
        figure_kind="transfer",
        window_name="pre_choice",
        target_name="current_action",
    )
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        assert len(figure.axes) == 6
        assert all(axis.get_legend() is None for axis in figure.axes)
        assert len(figure.legends) == 1
        legend = figure.legends[0]
        labels = [text.get_text() for text in legend.get_texts()]
        assert len(labels) == 23
        assert len(labels) == len(set(labels))
        assert labels.count("training interval") == 1
        assert (
            labels.count("correct_rewarded -> switch | balanced_accuracy") == 1
        )
        assert labels.count("switch -> switch | auc") == 1

        legend_bounds = legend.get_window_extent(renderer)
        _assert_artist_inside(figure.bbox, legend_bounds)
        for axis in figure.axes:
            assert not legend_bounds.overlaps(axis.get_window_extent(renderer))
        caption = next(
            text
            for text in figure.texts
            if text.get_text().startswith("Transfer pooled-window decoding")
        )
        caption_bounds = caption.get_window_extent(renderer)
        _assert_artist_inside(figure.bbox, caption_bounds)
        assert not legend_bounds.overlaps(caption_bounds)
        for axis in figure.axes:
            assert not caption_bounds.overlaps(axis.get_window_extent(renderer))
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("figure_kind", "expected_label_count"),
    (("native", 13), ("gap", 13)),
)
def test_native_and_gap_use_one_shared_complete_legend(
    figure_kind: str,
    expected_label_count: int,
) -> None:
    """Native and gap figures retain every condition/metric label once."""
    from src.neural_analysis.task_decoding.pooled import plotting

    figure = plotting.plot_pooled_timecourses(
        _full_condition_saved_result(),
        figure_kind=figure_kind,
        window_name="pre_choice",
        target_name="current_action",
    )
    try:
        assert len(figure.axes) == 6
        assert all(axis.get_legend() is None for axis in figure.axes)
        assert len(figure.legends) == 1
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert len(labels) == expected_label_count
        assert len(labels) == len(set(labels))
        assert labels.count("training interval") == 1
        assert labels.count("switch | auc") == (1 if figure_kind == "gap" else 0)
        assert labels.count("switch -> switch | auc") == (
            1 if figure_kind == "native" else 0
        )
        if figure_kind == "gap":
            assert all(
                any(
                    np.asarray(line.get_ydata()).tolist() == [0.0, 0.0]
                    for line in axis.lines
                )
                for axis in figure.axes
            )
    finally:
        plt.close(figure)


def test_saved_result_plotting_writes_exact_required_figures_and_complete_captions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Each window-target cell gets native, transfer, and gap PNG evidence."""
    arrays, metadata = make_result_template()
    results.save_pooled_run(tmp_path, arrays=arrays, metadata=metadata)
    monkeypatch.setattr(activity, "load_region_activity", _fail_if_called)
    monkeypatch.setattr(activity, "build_session_rate_tensors", _fail_if_called)
    monkeypatch.setattr(modeling, "make_estimator", _fail_if_called)

    from src.neural_analysis.task_decoding.pooled import plotting

    saved = results.load_pooled_run(tmp_path)
    gap_figure = plotting.plot_pooled_timecourses(
        saved,
        figure_kind="gap",
        window_name="pre_choice",
        target_name="current_action",
    )
    visible = _all_figure_text(gap_figure)
    plt.close(gap_figure)

    for required in (
        "correct_rewarded",
        "incorrect_unrewarded",
        "pfc",
        "hpc",
        "pca",
        "units",
        "balanced_accuracy",
        "auc",
        "seconds",
        "-0.5",
        "0.0",
        "in-window",
        "out-of-window",
        "unavailable",
        "complete pipelines",
        "cannot isolate estimator weights",
    ):
        assert required in visible

    paths = plotting.save_pooled_decoding_figures(tmp_path)
    expected_names = {
        f"{kind}__pre_choice__{target}.png"
        for kind in ("native", "transfer", "gap")
        for target in ("current_action", "relative_doubt")
    }
    assert {path.name for path in paths} == expected_names
    assert {path.name for path in (tmp_path / "figures").iterdir()} == expected_names
    for path in paths:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
