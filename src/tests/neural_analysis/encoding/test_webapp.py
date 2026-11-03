"""Saved-only encoding view, early metadata routing, and display-state contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.encoding import modeling, pipeline, preparation, results
from src.neural_analysis.webapp import app, encoding_views, session_inputs


def forbid_analysis(*args: object, **kwargs: object) -> None:
    """Reject arbitrary raw loading/fitting arguments; return no arrays or units."""
    pytest.fail("Saved encoding controls must never load live sources or fit models")


def render_for_test(metadata_path: str) -> None:
    """Input metadata filename string; render the saved-only Streamlit view, return None."""
    import streamlit as st
    from src.neural_analysis.webapp.session_inputs import load_webapp_session
    from src.neural_analysis.webapp.encoding_views import render_encoding_view

    render_encoding_view(st, load_webapp_session(metadata_path))


def figure_spec(view: AppTest) -> dict:
    """Input Streamlit AppTest; return serialized Plotly dict with elapsed seconds/native PC units."""
    component = view.get("bidi_component")[0]
    return json.loads(json.loads(component.proto.json)["figure_json"])


def choose_first(label: str, options: list, **settings: object) -> object:
    """Input widget label/options/settings; return first default option, no physical units."""
    return options[0]


def test_configuration_defaults_prefer_pfc_and_same_numeric_over_file_hash_order() -> None:
    """No inputs; saved metadata order cannot choose a different initial penalty policy, return None."""
    entries = []
    for region, mode in (("HPC", "matched_context"), ("PFC", "matched_context"), ("PFC", "same_numeric")):
        entries.append({"id": region + mode, "meta": {"settings": {"region": region,
            "window": "whole", "timing_mode": "choice_updated", "encoding": "separate_context",
            "estimator": "ridge", "penalty_mode": mode}, "penalty_aliases": [mode]}})
    fake = SimpleNamespace(sidebar=SimpleNamespace(selectbox=choose_first))
    entry, mode = encoding_views._configuration_controls(fake, entries, "fixture")
    assert entry["meta"]["settings"]["region"] == "PFC"
    assert mode == "same_numeric"


@pytest.fixture
def completed_view_run(prepared_encoding_run: Path) -> tuple[Path, Path]:
    """Input prepared run Path; return completed run/metadata Paths, fixed (W,B,K) activity."""
    pipeline.run_prepared_encoding(prepared_encoding_run)
    manifest = results.load_run_manifest(prepared_encoding_run)
    return prepared_encoding_run, Path(manifest["config"]["session_metadata"])


def test_encoding_view_is_available_without_live_behavior_or_spikes(
    completed_view_run: tuple[Path, Path],
) -> None:
    """Input saved run/metadata Paths; raw mounts may disappear, availability remains, return None."""
    run, metadata = completed_view_run
    config = results.restore_config(results.load_run_manifest(run)["config"])
    for path in (config.augmented_trials, config.feature_parameters, config.session_root / "trials.csv",
                 config.session_root / "events.csv", config.session_root / "A/aligned.npz",
                 config.session_root / "B/aligned.npz"):
        path.unlink()
    session = session_inputs.load_webapp_session(metadata)
    assert session_inputs.PLOT_VIEW_CONTEXT_ENCODING == "Context encoding results"
    assert session_inputs.metadata_view_availability(session)[session_inputs.PLOT_VIEW_CONTEXT_ENCODING].available
    view = AppTest.from_function(render_for_test, args=(str(metadata),)).run()
    assert not view.exception
    assert view.get("bidi_component")


def test_saved_encoding_dispatch_precedes_live_probe_controls(
    completed_view_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input saved run/metadata Paths and patch; dispatch returns before raw controls, return None."""
    _, metadata = completed_view_run
    session = session_inputs.load_webapp_session(metadata)
    fake = SimpleNamespace(sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(
        return_value=session_inputs.PLOT_VIEW_CONTEXT_ENCODING)))
    renderer = Mock()
    monkeypatch.setattr(app, "st", fake)
    monkeypatch.setattr(app, "render_encoding_view", renderer)
    assert app._start_metadata_webapp(session) is None
    renderer.assert_called_once_with(fake, session)


def test_controls_reuse_saved_activity_and_preserve_time_revision(
    completed_view_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input saved run/metadata Paths and patch; scale/order/selected-PC changes fit nothing, return None."""
    run, metadata = completed_view_run
    manifest = results.load_run_manifest(run)
    preparation_path = Path(manifest["session_root"]) / manifest["preparation_path"]
    original = preparation_path.read_bytes()
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_analysis)
    monkeypatch.setattr(preparation, "load_behavior_inputs", forbid_analysis)
    monkeypatch.setattr(preparation, "prepare_regional_activity", forbid_analysis)
    view = AppTest.from_function(render_for_test, args=(str(metadata),)).run()
    assert not view.exception
    first = figure_spec(view)
    np.testing.assert_allclose(first["layout"]["meta"]["loaded_range_s"], [0.0, 116.0])
    component_id = view.get("bidi_component")[0].proto.id
    scale = next(item for item in view.selectbox if item.label == "Coefficient scale")
    scale.select("standardized").run()
    assert not view.exception
    standardized = figure_spec(view)
    assert standardized["layout"]["meta"]["coefficient_scale"] == "standardized"
    assert standardized["layout"]["uirevision"] == first["layout"]["uirevision"]
    next(item for item in view.selectbox if item.label == "Order responses").select("coefficient_strength").run()
    next(item for item in view.selectbox if item.label == "Selected PC").select(2).run()
    assert not view.exception
    selected = figure_spec(view)
    assert selected["layout"]["meta"]["selected_pc_rank"] == 2
    assert selected["layout"]["uirevision"] == first["layout"]["uirevision"]
    assert view.get("bidi_component")[0].proto.id == component_id
    time_axes = selected["layout"]["meta"]["time_axes"]
    categorical = [name for name, value in selected["layout"].items()
                   if name.startswith("xaxis") and isinstance(value, dict) and value.get("type") == "category"]
    assert categorical and not set(categorical).intersection(time_axes)
    assert preparation_path.read_bytes() == original


def test_explicit_navigation_and_prediction_scope_are_saved_display_choices(
    completed_view_run: tuple[Path, Path],
) -> None:
    """Input saved run/metadata Paths; elapsed viewport/trace scope are explicit, return None."""
    _, metadata = completed_view_run
    view = AppTest.from_function(render_for_test, args=(str(metadata),)).run()
    first = figure_spec(view)
    next(item for item in view.number_input if item.label == "Visible duration (s)").set_value(10.0).run()
    next(item for item in view.number_input if item.label == "Window start (s)").set_value(30.0).run()
    next(item for item in view.selectbox if item.label == "Prediction scope").select("reference").run()
    assert not view.exception
    changed = figure_spec(view)
    primary = changed["layout"]["meta"]["primary_time_axis"]
    np.testing.assert_allclose(changed["layout"][primary]["range"], [30, 40])
    assert changed["layout"]["meta"]["prediction_scope"] == "reference"
    assert changed["layout"]["uirevision"] != first["layout"]["uirevision"]


def test_partial_run_displays_completed_cells_without_requiring_all_models(
    completed_view_run: tuple[Path, Path],
) -> None:
    """Input completed run/metadata Paths; a partial progress index remains inspectable, return None."""
    run, metadata = completed_view_run
    progress = results.inspect_run_status(run)
    first = next(iter(progress["completed_cells"].items()))
    results.write_progress(run, {**progress, "status": "running", "completed_cells": dict([first])})
    view = AppTest.from_function(render_for_test, args=(str(metadata),)).run()
    assert not view.exception
    assert view.get("bidi_component")
    assert any("partial" in item.value.lower() for item in view.warning)


def test_png_export_honors_selected_reference_prediction_scope(
    completed_view_run: tuple[Path, Path],
) -> None:
    """Input saved run/metadata Paths; explicitly selected reference traces keep their PNG scope, return None."""
    from PIL import Image

    run, metadata = completed_view_run
    view = AppTest.from_function(render_for_test, args=(str(metadata),)).run()
    next(item for item in view.selectbox if item.label == "Prediction scope").select("reference").run()
    next(item for item in view.button if item.label.startswith("Export PNGs for ")).click().run()
    assert not view.exception
    paths = list((run / "figures").rglob("*.png"))
    assert len(paths) == 2
    for path in paths:
        with Image.open(path) as image:
            caption = image.info["Description"].lower()
            assert "reference" in caption and "held-out predictions" not in caption
