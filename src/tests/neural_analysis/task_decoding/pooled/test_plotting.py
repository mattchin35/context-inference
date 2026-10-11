"""Saved-result-only reporting contracts for pooled decoding."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.neural_analysis.task_decoding import activity, modeling
from src.neural_analysis.task_decoding.pooled import results
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
    return " ".join(labels).lower()


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

