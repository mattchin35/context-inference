"""WP7 contracts for task-decoding plots built only from saved result arrays."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis.task_decoding import results
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


def _condition_saved_run(tmp_path: Path) -> dict[str, object]:
    """Stack two in-memory pooled payloads into one schema-2 plotting fixture.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory for tiny portable source fixtures.

    Returns
    -------
    dict[str, object]
        Loader-shaped run with ``all`` and ``incorrect`` conditions. The
        incorrect scores and coefficients differ visibly from pooled values.
    """
    pooled = _saved_run(tmp_path)
    pooled_arrays = pooled["arrays"]
    assert isinstance(pooled_arrays, Mapping)
    incorrect_arrays = {
        name: value.copy() for name, value in pooled_arrays.items()
    }
    incorrect_arrays["fold_scores"][0, 0, 0, 0, 0, :] = 0.73
    incorrect_arrays["coefficient_values"][0, 0, 1, 0, :, :2] = np.asarray(
        [[0.8, 0.4], [0.6, 0.2], [1.0, 0.6]],
        dtype=float,
    )
    full_count = int(pooled_arrays["full_table_row_positions"].size)
    stacked_arrays, stacked_meta = results.assemble_condition_result_payload(
        condition_names=("all", "incorrect"),
        condition_masks={
            "all": np.ones(full_count, dtype=np.bool_),
            "incorrect": np.ones(full_count, dtype=np.bool_),
        },
        condition_payloads={
            "all": (pooled_arrays, pooled["meta"]),
            "incorrect": (incorrect_arrays, pooled["meta"]),
        },
    )
    scientific_config = dict(pooled["scientific_config"])
    scientific_config["condition_names"] = ["all", "incorrect"]
    return {
        **pooled,
        "arrays": stacked_arrays,
        "meta": stacked_meta,
        "scientific_config": scientific_config,
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


def test_heatmap_layout_reserves_colorbar_and_wrapped_caption_space(tmp_path: Path) -> None:
    """Colorbar and caption remain legible without covering the six scientific panels."""
    plotting = _plotting_module()
    figure = plotting.plot_decoding_heatmap(
        _saved_run(tmp_path),
        family="categorical",
        metric="balanced_accuracy",
    )
    try:
        figure.canvas.draw()
        heatmap_axes = [axis for axis in figure.axes if axis.images]
        colorbar_axes = [axis for axis in figure.axes if not axis.images]
        assert len(heatmap_axes) == 6
        assert len(colorbar_axes) == 1
        assert all(
            not axis.get_position().overlaps(colorbar_axes[0].get_position())
            for axis in heatmap_axes
        )
        assert "\n" in _figure_caption(figure)
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


def test_condition_heatmap_selects_one_axis_and_labels_the_scientific_subset(
    tmp_path: Path,
) -> None:
    """A schema-2 heatmap uses only the selected condition and names it visibly."""
    plotting = _plotting_module()
    figure = plotting.plot_decoding_heatmap(
        _condition_saved_run(tmp_path),
        family="categorical",
        metric="balanced_accuracy",
        condition="incorrect",
    )
    try:
        values = figure.axes[0].images[0].get_array()
        assert np.all(values[0, 0] == 0.73)
        assert "Incorrect" in figure._suptitle.get_text()
        assert "condition: incorrect" in _figure_caption(figure).lower()
    finally:
        plt.close(figure)


def test_condition_default_png_export_is_complete_and_condition_qualified(
    tmp_path: Path,
) -> None:
    """Schema-2 export writes every present-family plot once per condition."""
    plotting = _plotting_module()
    run_directory = tmp_path / "condition-run"
    saved_run = _condition_saved_run(tmp_path)

    written = plotting.save_default_decoding_figures(
        run_directory,
        saved_run=saved_run,
    )

    expected_names = {
        f"{condition}--{name}"
        for condition in ("all", "incorrect")
        for name in (
            "categorical_balanced_accuracy.png",
            "categorical_auc.png",
            "numerical_r2.png",
        )
    }
    assert {path.name for path in written} == expected_names
    assert plotting.required_default_figure_filenames(
        saved_run,
    ) == tuple(
        f"{condition}--{name}"
        for condition in ("all", "incorrect")
        for name in (
            "categorical_balanced_accuracy.png",
            "categorical_auc.png",
            "numerical_r2.png",
        )
    )


def test_condition_coefficient_summary_uses_only_the_selected_condition(
    tmp_path: Path,
) -> None:
    """Direct-unit tables and plots select the same schema-2 condition cell."""
    plotting = _plotting_module()
    saved_run = _condition_saved_run(tmp_path)

    feature_table, _fold_table = plotting.summarize_unit_coefficients(
        saved_run,
        target="current_action",
        region="PFC",
        time_bin_index=0,
        condition="incorrect",
    )
    figure = plotting.plot_unit_coefficients(
        saved_run,
        target="current_action",
        region="PFC",
        time_bin_index=0,
        condition="incorrect",
    )
    try:
        observed = feature_table.set_index("feature_id")
        assert observed.loc["probe-pfc:11", "median_coefficient"] == 0.8
        assert observed.loc["probe-pfc:19", "median_coefficient"] == 0.4
        assert "Incorrect" in figure.axes[0].get_title()
    finally:
        plt.close(figure)


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
        assert figure.axes[0].get_title().endswith("at -1.75 s")
    finally:
        plt.close(figure)


def test_coefficient_plot_handles_a_saved_region_with_no_active_units(tmp_path: Path) -> None:
    """A scientifically unavailable zero-unit region renders an explicit empty summary."""
    plotting = _plotting_module()
    saved_run = _saved_run(tmp_path)
    arrays = saved_run["arrays"]
    assert isinstance(arrays, Mapping)
    arrays["coefficient_active_masks"][0, 1] = False
    arrays["coefficient_feature_ids"][0, 1] = ""
    arrays["coefficient_feature_regions"][0, 1] = ""
    arrays["coefficient_feature_statuses"][0, 1] = "padding"
    arrays["coefficient_values"][:, 0, 1] = np.nan
    arrays["fit_status"][:, 0, 1] = "unavailable"
    arrays["fit_reason_codes"][:, 0, 1] = "outer_pfc_features_unavailable"
    arrays["effective_feature_counts"][:, 0, 1] = 0

    feature_table, fold_table = plotting.summarize_unit_coefficients(
        saved_run,
        target="current_action",
        region="PFC",
        time_bin_index=0,
    )
    figure = plotting.plot_unit_coefficients(
        saved_run,
        target="current_action",
        region="PFC",
        time_bin_index=0,
    )
    try:
        assert feature_table.empty
        assert fold_table["fit_status"].tolist() == ["unavailable"] * 3
        assert "No active direct-unit features" in _figure_caption(figure)
    finally:
        plt.close(figure)
