"""The decoder unit subview consumes saved products, preserving existing routes."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.neural_analysis.population_decoding import pipeline, preparation


def render_for_test(metadata_path):
    """Input scalar metadata filename; render the existing population view and new subview."""
    import streamlit as st
    from src.neural_analysis.webapp.session_inputs import load_webapp_session
    from src.neural_analysis.webapp.population_decoding_view import render_population_decoding_view
    render_population_decoding_view(st, load_webapp_session(metadata_path))


def forbid_analysis(*args, **kwargs):
    """Input arbitrary arguments; any neural fitting/preparation in the UI is an error."""
    raise AssertionError("Saved view attempted raw analysis")


def test_unit_subview_controls_source_free_export_and_probability_return(saved_decoder_run, monkeypatch):
    """Input saved run; units/scale/count selectors and export work after raw sources are removed."""
    root = saved_decoder_run.parent.parent
    for name in ("trials.csv", "events.csv", "A/aligned.npz", "B/aligned.npz"):
        (root / name).unlink()
    monkeypatch.setattr(pipeline, "fit_target", forbid_analysis)
    monkeypatch.setattr(preparation, "prepare_session", forbid_analysis)
    view = AppTest.from_function(render_for_test, args=(str(root / "neural_session.json"),), default_timeout=20).run()
    mode = next(item for item in view.selectbox if item.label == "Display")
    assert mode.value == "Probability timelines"
    mode.select("Decoder-weight unit activity").run()
    assert not view.exception and view.get("bidi_component")
    assert not any(item.label in ("Target", "Population features") for item in view.selectbox)
    assert any("500 ms" in item.value for item in view.caption)
    next(item for item in view.number_input if item.label == "Units per group").set_value(1).run()
    next(item for item in view.radio if item.label == "Activity scale").set_value("hz").run()
    assert not view.exception
    tables = [item.value for item in view.dataframe]
    displayed = next(table for table in tables if "unit_id" in table.columns and "display_order" in table.columns)
    assert displayed.unit_id.is_unique
    side_table = next(table for table in tables if "Reward trials" in table.columns)
    assert {"Reward bins", "No-reward bins", "Required bins", "Predicted bins"} <= set(side_table)
    next(item for item in view.button if item.label == "Export unit activity PNG").click().run()
    assert not view.exception and list((root / "analysis_runs").glob("decoder_unit_activity_*/activity.png"))
    next(item for item in view.selectbox if item.label == "Display").select("Probability timelines").run()
    assert not view.exception and any(item.label == "Target" for item in view.selectbox)


def test_pending_required_models_show_reason(saved_decoder_run):
    """Input incomplete saved run; new view explains pending models without silently substituting weights."""
    (saved_decoder_run / "cells/PFC_left_switch_vs_stay.npz").unlink()
    metadata = saved_decoder_run.parent.parent / "neural_session.json"
    view = AppTest.from_function(render_for_test, args=(str(metadata),), default_timeout=20).run()
    next(item for item in view.selectbox if item.label == "Display").select("Decoder-weight unit activity").run()
    assert not view.exception and any("left_switch_vs_stay" in item.value for item in view.info)
