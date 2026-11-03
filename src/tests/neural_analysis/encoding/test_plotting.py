"""Saved context coefficients, fixed PC identity and coupled real-time display contracts."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.neural_analysis.encoding import modeling, plotting, results
from src.neural_analysis.encoding.preparation import PreparedRegion, build_bin_rows
from src.neural_analysis.encoding.predictors import build_context_values
from src.neural_analysis.population import session_timeline as timeline


@pytest.fixture
def plotting_data() -> tuple[dict, dict]:
    """Return synthetic preparation/rawcell for geometry; scores(2,8,12), UTC/elapsed seconds.

    No inputs. Seed=20261103. Original PCs remain1..12. Coefficients and partial
    predictions are deliberately specified visual fixtures, not regression
    estimates. Two behavioral blocks make complete CV unavailable. The fixture
    includes a valid zero score and a110-second inter-window gap.
    """
    trials = pd.DataFrame({"cur_trial": [105, 108], "cur_block": [0, 1],
        "start_time": [0.0, 114.0], "choice_time": [10.0, 124.0], "state_int": [0, 1],
        "action": [1, 0], "experimenter_reward_given": [0, 0], "reward": [1, 0]})
    origin, choices = timeline.prepare_timeline_choices(trials)
    rng = np.random.default_rng(20261103)
    regional = timeline.fit_timeline_activity(rng.poisson(3, size=(2, 8, 12)),
        tuple(f"A:{unit}" for unit in range(12)), np.array([0, 1]), "PFC", "A", 0.5)
    regional.scores[0, 0, 11] = 0.0
    events = pd.DataFrame({"Time": [0.0, 10.2, 123.0, 125.0],
        "Event": ["enter_right_patch", "enter_left_patch", "LED_on", "LED_off"],
        "elapsed_s": [0.0, 10.2, 123.0, 125.0]})
    behavior = timeline.SessionTimelineResult("visual-fixture", origin, choices, events,
        {"PFC": regional}, {}, {"configuration_id": "visual-fixture"})
    contexts = {mode: build_context_values(trials, regional.bin_edges_s, mode)
                for mode in ("choice_updated", "trial_window")}
    rows = build_bin_rows(trials, choices, regional, origin)
    prepared = PreparedRegion(regional, np.arange(1, 13), rows,
        {mode: pair[0] for mode, pair in contexts.items()},
        {mode: pair[1] for mode, pair in contexts.items()}, np.array([-1, -1]),
        "Fewer than five blocks", False, {})
    indices = np.flatnonzero(rows.before.to_numpy())
    observed = regional.scores.reshape(-1, 12)[indices]
    slopes = np.linspace(0.1, 0.2, 12)[None, :]
    slopes[0, [11, 1, 6]] = [5, -4, 3]
    standardized = slopes.copy()
    standardized[0, [6, 11, 1]] = [8, 7, -6]
    partial = observed + 0.2
    partial[-1] = np.nan
    cell = {"meta": {"cell_id": "visual-cell", "preparation_id": "visual-preparation",
        "settings": {"region": "PFC", "window": "before", "timing_mode": "choice_updated",
                     "encoding": "signed_context", "estimator": "ridge", "penalty_mode": "same_numeric"},
        "penalty_aliases": ["same_numeric", "matched_context"],
        "configured_alpha": 1.0, "effective_alpha": 1.0, "available": True,
        "reference_valid": True, "cv_available": False, "cv_reason": "Partial visual fixture",
        "reference_scope": "all-eligible-data reference fit",
        "score_scope": "held-out regression conditional on fixed descriptive session PCA",
        "bin_width_ms": 500, "row_count": len(indices), "predictor_count": 1, "response_count": 12},
        "pc_ranks": np.arange(1, 13), "column_names": np.array(["signed_context"]),
        "reference_slopes": slopes, "standardized_slopes": standardized,
        "reference_intercept": np.zeros(12), "selected_row_indices": indices,
        "predictors": np.repeat([-1.0, 1.0], 4)[:, None],
        "reference_predictions": observed - 0.1, "held_out_predictions": partial,
        "mse": np.full(12, np.nan), "r2": np.full(12, np.nan)}
    return {"trials": trials, "behavior": behavior, "regions": {"PFC": prepared},
            "meta": {"preparation_id": "visual-preparation"}}, cell


def forbid_fit(*args: object, **kwargs: object) -> None:
    """Reject arbitrary fit arguments during saved display; return no data or physical units."""
    pytest.fail("Plotting must not fit models or PCA")


def trace_named(figure: object, name: str) -> object:
    """Input Plotly figure/trace-name string; return matching trace, retaining plotted shapes/units."""
    return next(trace for trace in figure.data if trace.name == name)


def test_response_order_uses_original_rank_ties_and_places_nan_after_valid_zero() -> None:
    """No inputs; coefficient(1,4) strength order keeps original unitless PC labels, return None."""
    cell = {"pc_ranks": np.array([12, 2, 7, 5]), "column_names": np.array(["signed_context"]),
            "reference_slopes": np.array([[3.0, -3.0, np.nan, 0.0]])}
    np.testing.assert_array_equal(plotting.order_responses(cell, ordering="coefficient_strength",
        sorting_predictor="signed_context"), [1, 0, 3, 2])
    np.testing.assert_array_equal(plotting.order_responses(cell, ordering="original"), [1, 3, 2, 0])


def test_coefficients_and_activity_use_actual_consecutive_rows_pc12_pc2_pc7(plotting_data: tuple) -> None:
    """Input prep/rawcell fixture; rendered rows0/1/2 carry PC12/2/7 and correct raw activity, return None."""
    prepared, cell = plotting_data
    figure = plotting.plot_encoding_timeline(prepared, cell, ordering="coefficient_strength",
        sorting_predictor="signed_context", row_count=3, selected_pc=12, time_range_s=(0, 140))
    coefficient = trace_named(figure, "Coefficients")
    activity = trace_named(figure, "PC activity")
    np.testing.assert_array_equal(coefficient.y, [0, 1, 2])
    np.testing.assert_array_equal(activity.y, coefficient.y)
    np.testing.assert_array_equal(np.asarray(coefficient.z)[:, 0], [5, -4, 3])
    axis_name = activity.yaxis.replace("y", "yaxis", 1)
    assert tuple(figure.layout[axis_name].ticktext) == ("PC12", "PC2", "PC7")
    np.testing.assert_array_equal(figure.layout.meta["display_pc_ranks"], [12, 2, 7])
    source = prepared["regions"]["PFC"].timeline
    np.testing.assert_allclose(np.asarray(activity.z)[:, :8], source.scores[0, :, [11, 1, 6]])
    np.testing.assert_allclose(np.asarray(activity.z)[:, -8:], source.scores[1, :, [11, 1, 6]])
    assert activity.zmin == -source.color_limit and activity.zmax == source.color_limit
    assert np.asarray(activity.z)[0, 0] == 0
    assert np.isnan(np.asarray(activity.z)[:, 8]).all()
    assert "session variance" in activity.hovertemplate and "Choice-relative" in activity.hovertemplate
    assert "PC12" in np.asarray(activity.customdata, dtype=object)[0, 0].tolist()


def test_coefficient_scale_changes_order_without_normalizing_or_refitting_activity(plotting_data: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input prep/rawcell/patch; raw-SD toggle affects coefficient display only, return None."""
    prepared, cell = plotting_data
    original_scores = prepared["regions"]["PFC"].timeline.scores.copy()
    original_slopes = cell["reference_slopes"].copy()
    monkeypatch.setattr(modeling, "fit_ridge_responses", forbid_fit)
    monkeypatch.setattr(modeling, "fit_ols_responses", forbid_fit)
    monkeypatch.setattr(timeline, "fit_timeline_activity", forbid_fit)
    raw = plotting.plot_encoding_timeline(prepared, cell, ordering="coefficient_strength", row_count=3,
        sorting_predictor="signed_context", time_range_s=(0, 140))
    standardized = plotting.plot_encoding_timeline(prepared, cell, coefficient_scale="standardized",
        ordering="coefficient_strength", sorting_predictor="signed_context", row_count=3, time_range_s=(0, 140))
    assert raw.layout.meta["display_pc_ranks"] == [12, 2, 7]
    assert standardized.layout.meta["display_pc_ranks"] == [7, 12, 2]
    np.testing.assert_array_equal(trace_named(standardized, "Coefficients").z, [[8], [7], [-6]])
    np.testing.assert_array_equal(prepared["regions"]["PFC"].timeline.scores, original_scores)
    np.testing.assert_array_equal(cell["reference_slopes"], original_slopes)
    assert trace_named(raw, "PC activity").zmax == trace_named(standardized, "PC activity").zmax


def test_fit_mask_and_native_predictors_keep_full_activity_and_true_gaps(plotting_data: tuple) -> None:
    """Input prep/rawcell; before fit marks four bins while activity retains eight per window, return None."""
    prepared, cell = plotting_data
    figure = plotting.plot_encoding_timeline(prepared, cell, time_range_s=(0, 140), selected_pc=12)
    activity = trace_named(figure, "PC activity")
    inclusion = trace_named(figure, "Fit-bin inclusion")
    predictors = trace_named(figure, "Model predictors")
    np.testing.assert_array_equal(np.asarray(inclusion.z)[0, :8], [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(predictors.z)[0, :8], [-1, -1, -1, -1, 1, 1, 1, 1])
    assert np.asarray(activity.z).shape[1] == 17
    assert len(activity.x) == 18
    np.testing.assert_allclose(np.asarray(activity.x)[8:10], [12.0, 122.0])
    assert np.isnan(np.asarray(inclusion.z)[:, 8]).all()
    assert np.isnan(np.asarray(predictors.z)[:, 8]).all()
    assert any(trace.name == "Left choices" for trace in figure.data)
    assert any(shape.x0 == 10.2 for shape in figure.layout.shapes)  # Logged update stays delayed.


def test_prediction_trace_is_in_coupled_figure_with_explicit_scope_and_no_fallback(plotting_data: tuple) -> None:
    """Input prep/rawcell; selected PC held-out/reference traces share timeaxes and preserve missingness, return None."""
    prepared, cell = plotting_data
    held = plotting.plot_encoding_timeline(prepared, cell, selected_pc=12, prediction_scope="held_out", time_range_s=(0, 140))
    reference = plotting.plot_encoding_timeline(prepared, cell, selected_pc=12, prediction_scope="reference", time_range_s=(0, 140))
    held_trace = next(trace for trace in held.data if str(trace.name).startswith("Held-out prediction"))
    reference_trace = next(trace for trace in reference.data if str(trace.name).startswith("Reference prediction"))
    observed = next(trace for trace in held.data if str(trace.name).startswith("Observed"))
    axis_name = held_trace.xaxis.replace("x", "xaxis", 1)
    assert axis_name in held.layout.meta["time_axes"]
    assert observed.xaxis == held_trace.xaxis
    assert not held_trace.connectgaps and not observed.connectgaps
    assert not np.allclose(held_trace.y, reference_trace.y, equal_nan=True)
    cell_without_cv = {**cell, "held_out_predictions": np.full_like(cell["held_out_predictions"], np.nan)}
    blank = plotting.plot_encoding_timeline(prepared, cell_without_cv, selected_pc=12, time_range_s=(0, 140))
    blank_prediction = next(trace for trace in blank.data if str(trace.name).startswith("Held-out prediction"))
    assert np.isnan(np.asarray(blank_prediction.y, dtype=float)).all()
    assert any("unavailable" in str(annotation.text).lower() for annotation in blank.layout.annotations)
    assert blank.layout.meta["prediction_scope"] == "held_out"
    assert blank.layout.meta["selected_pc_rank"] == 12


def test_default_view_is_bounded_60_seconds_ten_rows_and_coefficient_axis_categorical(plotting_data: tuple) -> None:
    """Input prep/rawcell; display defaults bound payload without changing fitted PC count, return None."""
    prepared, cell = plotting_data
    figure = plotting.plot_encoding_timeline(prepared, cell)
    coefficient = trace_named(figure, "Coefficients")
    activity = trace_named(figure, "PC activity")
    coefficient_axis = coefficient.xaxis.replace("x", "xaxis", 1)
    assert figure.layout[coefficient_axis].type == "category"
    assert figure.layout[coefficient_axis].fixedrange
    assert coefficient_axis not in figure.layout.meta["time_axes"]
    primary = figure.layout.meta["primary_time_axis"]
    assert tuple(figure.layout[primary].range) == (0.0, 60.0)
    assert len(activity.y) == 10 and len(cell["pc_ranks"]) == 12
    assert np.asarray(activity.z).shape[1] == 8  # Second window is outside loaded range.


def test_selected_alias_label_does_not_mutate_cached_cell(plotting_data: tuple) -> None:
    """Input prep/rawcell; shared signed-fit alias is labeled truthfully without modifying metadata, return None."""
    prepared, cell = plotting_data
    figure = plotting.plot_encoding_timeline(prepared, cell, inspection_penalty_mode="matched_context")
    assert "matched_context" in str(figure.layout.title.text)
    assert cell["meta"]["settings"]["penalty_mode"] == "same_numeric"


def test_cell_observation_helper_reuses_shared_scores_and_rejects_identity_conflict(plotting_data: tuple) -> None:
    """Input prep/rawcell; shared response assembly returns PC Y(N,K) without requiring saved observed, return None."""
    prepared, cell = plotting_data
    observed = results.cell_observations(prepared, cell)
    assert observed.shape == (8, 12) and "observed" not in cell
    bad = {**cell, "pc_ranks": np.arange(2, 14)}
    with pytest.raises(ValueError, match="PC|identity|preparation"):
        results.cell_observations(prepared, bad)


def test_light_png_exports_are_cell_identified_readable_and_captioned(plotting_data: tuple, tmp_path: Path) -> None:
    """Input prep/rawcell/output Path; light PNGs identify model/scope/selected PC, return None."""
    prepared, cell = plotting_data
    paths = plotting.save_inspection_figures(tmp_path / "figures", prepared, cell,
        time_range_s=(0, 60), selected_pc=12)
    assert len(paths) == 2
    for path in paths:
        assert path.suffix == ".png" and cell["meta"]["cell_id"] in path.name
        with Image.open(path) as image:
            assert image.width >= 800 and image.height >= 500
            assert image.convert("RGB").getpixel((0, 0)) == (255, 255, 255)
            caption = image.info["Description"]
            assert all(label in caption for label in ("PFC", "before", "choice_updated", "signed_context", "ridge", "held-out"))
    with pytest.raises(FileExistsError):
        plotting.save_inspection_figures(tmp_path / "figures", prepared, cell, selected_pc=12)


def test_predictor_hover_shows_actual_saved_design_with_trial_and_time_metadata(plotting_data: tuple) -> None:
    """Input prep/rawcell; fitted bins show saved X even if context-derived values differ, return None."""
    prepared, cell = plotting_data
    changed = {**cell, "predictors": cell["predictors"].copy()}
    changed["predictors"][0, 0] = 1.0
    figure = plotting.plot_encoding_timeline(prepared, changed, time_range_s=(0, 140))
    predictors = trace_named(figure, "Model predictors")
    assert np.asarray(predictors.z)[0, 0] == 1.0
    assert cell["predictors"][0, 0] == -1.0
    assert all(text in predictors.hovertemplate for text in ("Trial", "UTC", "Choice-relative", "%{z}"))
    custom = np.asarray(predictors.customdata, dtype=object)
    assert custom[0, 0, 0] == 105
    np.testing.assert_allclose(custom[0, 0, 1:5].astype(float), [8.0, 8.5, -2.0, -1.5])
    activity = trace_named(figure, "PC activity")
    assert "Assigned predictors" in activity.hovertemplate
    assert "signed_context=1" in str(np.asarray(activity.customdata, dtype=object)[0, 0, 10])


def test_reference_png_export_uses_requested_scope_in_trace_and_caption(
    plotting_data: tuple, tmp_path: Path,
) -> None:
    """Input prep/rawcell/output Path; explicit reference export consumes reference Yhat(N,K), return None."""
    import matplotlib.pyplot as plt

    prepared, cell = plotting_data
    cell = {**cell, "held_out_predictions": np.full_like(cell["held_out_predictions"], np.nan)}
    with patch.object(plt.Axes, "plot", autospec=True, side_effect=plt.Axes.plot) as plotted:
        paths = plotting.save_inspection_figures(tmp_path / "reference", prepared, cell,
            selected_pc=12, prediction_scope="reference", time_range_s=(0, 140))
    calls = [call for call in plotted.call_args_list if str(call.kwargs.get("label", "")).startswith("Reference prediction")]
    assert len(calls) == 1
    values = np.asarray(calls[0].args[2])
    np.testing.assert_allclose(values[np.isfinite(values)], np.repeat(cell["reference_predictions"][:, 11], 2))
    assert not any(str(call.kwargs.get("label", "")).startswith("Held-out prediction") for call in plotted.call_args_list)
    for path in paths:
        with Image.open(path) as image:
            assert "reference-fit predictions" in image.info["Description"]
            assert "held-out predictions" not in image.info["Description"]
