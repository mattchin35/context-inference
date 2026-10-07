"""WP7 contracts for task-decoding plots built only from saved result arrays."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from src.tests.neural_analysis.task_decoding.test_results import (
    make_run_save_arguments,
    write_input_fixture,
)


def _plotting_module():
    """Import the WP7 plotting owner after test collection.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.plotting``.
    """
    return importlib.import_module("src.neural_analysis.task_decoding.plotting")


def _saved_run(tmp_path: Path) -> dict[str, object]:
    """Build one validated-shape in-memory run without publishing experimental data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory for tiny portable source fixtures.

    Returns
    -------
    dict[str, object]
        Loader-shaped mapping containing primitive arrays, metadata, config,
        manifest, and one opaque run fingerprint.
    """
    arguments = make_run_save_arguments(
        write_input_fixture(tmp_path),
        regularization_mode="fixed",
    )
    return {
        "arrays": arguments["arrays"],
        "meta": arguments["meta"],
        "scientific_config": arguments["scientific_config"],
        "input_manifest": arguments["input_manifest"],
        "run_fingerprint": arguments["run_fingerprint"],
    }


def _figure_caption(figure: plt.Figure) -> str:
    """Return all figure-level text used as the scientific caption."""
    return " ".join(text.get_text() for text in figure.texts)


def test_heatmaps_preserve_below_reference_negative_and_unavailable_cells(tmp_path: Path) -> None:
    """Scores below descriptive references remain data, while NaN remains unavailable."""
    plotting = _plotting_module()
    saved_run = _saved_run(tmp_path)
    arrays = saved_run["arrays"]
    assert isinstance(arrays, Mapping)
    arrays["fold_scores"][0, 0, 0, 0, 0, :] = 0.25
    arrays["fold_scores"][0, 0, 0, 0, 1, :] = np.nan
    arrays["fold_scores"][1, 0, 0, 2, 0, :] = -0.75

    categorical = plotting.plot_decoding_heatmap(
        saved_run,
        family="categorical",
        metric="balanced_accuracy",
    )
    numerical = plotting.plot_decoding_heatmap(saved_run, family="numerical", metric="r2")
    try:
        categorical_values = categorical.axes[0].images[0].get_array()
        numerical_values = numerical.axes[0].images[0].get_array()
        assert categorical_values[0, 0] == 0.25
        assert bool(np.ma.getmaskarray(categorical_values)[0, 1]) is True
        assert categorical.axes[0].images[0].norm.vcenter == 0.5
        assert numerical_values[0, 0] == -0.75
        assert numerical.axes[0].images[0].norm.vcenter == 0.0
        assert not np.allclose(
            categorical.axes[0].images[0].cmap.get_bad()[:3],
            (1.0, 1.0, 1.0),
        )
    finally:
        plt.close(categorical)
        plt.close(numerical)


def test_heatmap_facets_labels_and_caption_report_saved_coverage(tmp_path: Path) -> None:
    """One metric figure shows all six panels with saved time, trial, and fold facts."""
    plotting = _plotting_module()
    figure = plotting.plot_decoding_heatmap(
        _saved_run(tmp_path),
        family="categorical",
        metric="auc",
    )
    try:
        heatmap_axes = [axis for axis in figure.axes if axis.images]
        assert [axis.get_title() for axis in heatmap_axes] == [
            "PFC - PCA",
            "PFC - direct units",
            "HPC - PCA",
            "HPC - direct units",
            "PFC + HPC - PCA",
            "PFC + HPC - direct units",
        ]
        assert all(axis.get_xlabel() == "Time from choice_time (s)" for axis in heatmap_axes)
        assert heatmap_axes[0].get_yticklabels()[0].get_text() == "Current action"
        caption = _figure_caption(figure)
        assert "0.5 is a descriptive reference" in caption
        assert "current_action: 12 eligible trials" in caption
        assert "3 outer folds" in caption
    finally:
        plt.close(figure)


def test_default_png_export_is_opaque_atomic_and_family_bounded(tmp_path: Path) -> None:
    """Default export writes only present-family files with opaque white backgrounds."""
    plotting = _plotting_module()
    saved_run = _saved_run(tmp_path)
    run_directory = tmp_path / "run"

    written = plotting.save_default_decoding_figures(run_directory, saved_run=saved_run)

    expected = {
        run_directory / "figures" / "categorical_balanced_accuracy.png",
        run_directory / "figures" / "categorical_auc.png",
        run_directory / "figures" / "numerical_r2.png",
    }
    assert set(written) == expected
    assert {path.name for path in (run_directory / "figures").iterdir()} == {
        path.name for path in expected
    }
    for path in expected:
        pixels = plt.imread(path)
        assert pixels.ndim == 3
        if pixels.shape[-1] == 4:
            assert np.all(pixels[..., 3] == 1.0)
        assert np.allclose(pixels[0, 0, :3], 1.0)


def test_coefficient_summary_distinguishes_zero_excluded_and_unavailable(tmp_path: Path) -> None:
    """Direct-unit summaries retain numerical zeros and two different missing causes."""
    plotting = _plotting_module()
    saved_run = _saved_run(tmp_path)
    arrays = saved_run["arrays"]
    assert isinstance(arrays, Mapping)
    coefficients = arrays["coefficient_values"]
    coefficients[0, 0, 1, 0, :, :2] = np.array(
        [[0.0, np.nan], [0.4, 0.2], [np.nan, np.nan]],
        dtype=float,
    )
    arrays["fit_status"][0, 0, 1, 0, 2] = "unavailable"
    arrays["fit_reason_codes"][0, 0, 1, 0, 2] = "convergence_warning"

    feature_table, fold_table = plotting.summarize_unit_coefficients(
        saved_run,
        target="current_action",
        region="PFC",
        time_bin_index=0,
    )

    first = feature_table.set_index("feature_id").loc["probe-pfc:11"]
    second = feature_table.set_index("feature_id").loc["probe-pfc:19"]
    assert first["contributing_fold_count"] == 2
    assert first["selection_frequency"] == 0.5
    assert first["median_coefficient"] == 0.2
    assert second["excluded_fold_count"] == 1
    assert second["unavailable_fold_count"] == 1
    assert fold_table["fit_status"].tolist() == ["valid", "valid", "unavailable"]
    assert fold_table["nonzero_feature_count"].tolist() == [0, 2, 0]
    assert np.isnan(fold_table.loc[2, "nonzero_feature_fraction"])


def test_coefficient_plot_uses_direct_units_and_explains_interpretation(tmp_path: Path) -> None:
    """Coefficient display is a direct-unit reliance summary, not causal inference."""
    plotting = _plotting_module()
    saved_run = _saved_run(tmp_path)
    figure = plotting.plot_unit_coefficients(
        saved_run,
        target="current_action",
        region="PFC+HPC",
        time_bin_index=0,
    )
    try:
        caption = _figure_caption(figure)
        assert "decoder reliance" in caption
        assert "not a confidence interval" in caption
        assert "log-odds change per pooled training standard deviation" in caption
        assert "zero" in {text.get_text().lower() for text in figure.axes[0].get_legend().texts}
        labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
        assert labels[0] in {"probe-pfc:11", "probe-pfc:19", "probe-hpc:5", "probe-hpc:17"}
    finally:
        plt.close(figure)
