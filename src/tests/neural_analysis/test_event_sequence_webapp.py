"""Saved-first routing and explicit preparation contracts for event-sequence views."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.event_sequences import io as event_io
from src.neural_analysis.event_sequences import activity as event_activity
from src.neural_analysis.event_sequences import run_session
from src.neural_analysis.event_sequences.run_session import (
    EventSequenceRequest,
    prepare_event_sequences,
)
from src.neural_analysis.webapp import app, session_inputs
from src.neural_analysis.population import session_timeline as timeline
from src.tests.neural_analysis.event_sequences.conftest import (
    event_session as event_session_fixture,
)
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as synthetic_timeline_session_fixture,
)


event_session = event_session_fixture
synthetic_timeline_session = synthetic_timeline_session_fixture
CENTRAL_VIEW = "Event sequences: central trials"
TRANSITION_VIEW = "Event sequences: multi-trial sequences"


def _prepare_unit_product(session, unit_id: str = "A:7") -> Path:
    """Input: resolved synthetic session and explicit probe:cluster ID; return saved NPZ path."""
    request = EventSequenceRequest(session.metadata_path, "unit", unit_id=unit_id)
    return Path(prepare_event_sequences(request)["result_path"])


def _render_event_view(metadata_path: str, kind: str) -> None:
    """Input: metadata path and central/transition kind; render one Streamlit event view."""
    import streamlit as st

    from src.neural_analysis.webapp.event_sequence_views import render_event_sequence_view
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    render_event_sequence_view(st, load_webapp_session(metadata_path), kind=kind)


def _render_event_tool(metadata_path: str) -> None:
    """Input: metadata path; render the user-selected central/transition view in one shared app session."""
    import streamlit as st

    from src.neural_analysis.webapp.event_sequence_views import render_event_sequence_view
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    session = load_webapp_session(metadata_path)
    kind = st.selectbox("Test tool", options=("central", "transition"))
    render_event_sequence_view(st, session, kind=kind)


def _forbidden_display_operation(*args, **kwargs):
    """Input: arbitrary call arguments; raise if saved display performs source preparation or IO."""
    raise AssertionError("Saved display must not read current sources or prepare analysis.")


def _remove_current_sources(session, *, remove_behavior: bool = True) -> None:
    """Input: resolved synthetic session; remove raw neural files and optionally behavior CSVs."""
    for probe_id in ("A", "B"):
        for filename in ("aligned.npz", "spike_clusters.npy", "cluster_info.tsv", "channel_quality.csv"):
            (session.session_root / probe_id / filename).unlink(missing_ok=True)
    if remove_behavior:
        for filename in ("trials.csv", "events.csv"):
            (session.session_root / filename).unlink(missing_ok=True)


def _empty_one_unit_spike_train(session, probe_id: str, cluster_id: int) -> None:
    """Input: session, probe string and cluster integer; jointly filter aligned spike/assignment vectors (S,) in UTC seconds."""
    probe_root = session.session_root / probe_id
    assignments = np.load(probe_root / "spike_clusters.npy")
    alignment_path = probe_root / "aligned.npz"
    with np.load(alignment_path) as archive:
        times = archive["spike_utc_unix"]
        irig = archive["irig_utc_unix"]
    keep = assignments != cluster_id
    np.save(probe_root / "spike_clusters.npy", assignments[keep])
    np.savez(alignment_path, spike_utc_unix=times[keep], irig_utc_unix=irig)


def _assert_selector(app_test, widget_type: str, label: str, value) -> None:
    """Input: AppTest, widget collection name, label, and selected value; rerun the app."""
    widget = next(item for item in getattr(app_test, widget_type) if item.label == label)
    widget.select(value).run()
    assert not app_test.exception


def _assert_plotly_output(app_test) -> None:
    """Input: Streamlit AppTest page; assert it contains at least one Plotly chart element."""
    assert app_test.get("plotly_chart")


def _plotly_spec(app_test) -> dict:
    """Input: AppTest with one Plotly chart; return its JSON specification dictionary."""
    return json.loads(app_test.get("plotly_chart")[0].proto.spec)


def test_two_event_views_are_dedicated_metadata_options():
    """Input: none; return None after requiring both exact labels in metadata view options."""
    assert CENTRAL_VIEW in session_inputs.PLOT_VIEW_OPTIONS
    assert TRANSITION_VIEW in session_inputs.PLOT_VIEW_OPTIONS


@pytest.mark.parametrize(
    ("view_label", "kind"),
    [(CENTRAL_VIEW, "central"), (TRANSITION_VIEW, "transition")],
)
def test_event_view_dispatch_precedes_legacy_probe_and_trial_controls(
    event_session, monkeypatch, view_label, kind
):
    """Input: metadata session, fake sidebar and event kind; return None after early dispatch."""
    fake_st = SimpleNamespace(
        sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(return_value=view_label)),
    )
    renderer = Mock()
    available = session_inputs.MetadataViewAvailability(True, "available")
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "metadata_view_availability", Mock(return_value={view_label: available}))
    monkeypatch.setattr(app, "render_event_sequence_view", renderer, raising=False)
    assert app._start_metadata_webapp(event_session) is None
    renderer.assert_called_once_with(fake_st, event_session, kind=kind)


def test_saved_event_products_keep_views_available_when_current_sources_are_missing(event_session):
    """Input: synthetic session; save one product and remove current sources; return None after saved-first availability."""
    result_path = _prepare_unit_product(event_session)
    _remove_current_sources(event_session)

    availability = session_inputs.metadata_view_availability(event_session)
    assert availability[CENTRAL_VIEW].available
    assert availability[TRANSITION_VIEW].available
    assert result_path.is_file()


def test_one_ready_probe_keeps_event_views_available_before_preparation(event_session):
    """Input: synthetic session with only probe B unavailable; return None after metadata-only availability check."""
    for filename in ("aligned.npz", "spike_clusters.npy", "cluster_info.tsv", "channel_quality.csv"):
        (event_session.session_root / "B" / filename).unlink(missing_ok=True)
    availability = session_inputs.metadata_view_availability(event_session)
    assert availability[CENTRAL_VIEW].available
    assert availability[TRANSITION_VIEW].available


def test_saved_display_works_offline_without_csv_spikes_or_quality_reads(
    event_session, monkeypatch
):
    """Input: saved synthetic product and offline source paths; render without current CSV/source reads."""
    result_path = _prepare_unit_product(event_session)
    _remove_current_sources(event_session)
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "save_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(run_session, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbidden_display_operation)
    monkeypatch.setattr(event_activity, "extract_unit_activity", _forbidden_display_operation)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    assert not at.exception
    assert any("central" in element.value.lower() for element in at.subheader)
    _assert_plotly_output(at)
    assert result_path.is_file()


def test_missing_part_ii_fields_explain_only_transition_view(event_session, monkeypatch):
    """Input: synthetic session with no context/block columns; central view renders, transition explains absence."""
    trials = pd.read_csv(event_session.behavior.trial_table_file)
    trials.drop(columns=["state_int", "cur_block"]).to_csv(
        event_session.behavior.trial_table_file, index=False
    )
    _prepare_unit_product(event_session)
    _remove_current_sources(event_session)
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)

    central = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    transition = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    assert not central.exception
    assert not transition.exception
    assert any("Part II" in item.value for item in list(transition.info) + list(transition.warning))


def test_display_selectors_do_not_prepare_bin_fit_or_publish(event_session, monkeypatch):
    """Input: saved unit view and patched source work; rerun outcome, side, and z-score only from packet data."""
    _prepare_unit_product(event_session)
    from src.neural_analysis.webapp import event_sequence_views

    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    assert not at.exception
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "save_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(run_session, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(event_sequence_views, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbidden_display_operation)
    monkeypatch.setattr(event_activity, "extract_unit_activity", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbidden_display_operation)
    _assert_selector(at, "selectbox", "Central outcome", "omission")
    _assert_selector(at, "selectbox", "Central side", "right")
    checkbox = next(item for item in at.checkbox if item.label == "Z-score")
    assert not checkbox.disabled
    checkbox.check().run()
    assert not at.exception
    _assert_plotly_output(at)
    assert "session-wide z-score" in _plotly_spec(at)["layout"]["meta"]["caption"]


def test_unit_controls_do_not_reread_quality_on_live_motif_changes(event_session, monkeypatch):
    """Input: saved unit with dedicated controls. Return None; live motif changes reuse curated metadata."""
    _prepare_unit_product(event_session)
    at = AppTest.from_function(_render_event_view, args=(str(event_session.metadata_path), "central")).run()
    assert not at.exception
    _assert_plotly_output(at)
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    _assert_selector(at, "selectbox", "Central outcome", "omission")
    _assert_selector(at, "selectbox", "Central side", "right")
    _assert_plotly_output(at)


def test_legacy_metadata_requires_explicit_pc_probe_override(event_session, monkeypatch):
    """Input: version-2 metadata without roles. Return None; only an explicitly selected probe can prepare a PC source."""
    from src.neural_analysis.webapp import event_sequence_views

    metadata = json.loads(event_session.metadata_path.read_text())
    metadata["schema_version"] = "2"
    metadata.pop("analysis")
    event_session.metadata_path.write_text(json.dumps(metadata))
    prepare = Mock(wraps=prepare_event_sequences)
    monkeypatch.setattr(event_sequence_views, "prepare_event_sequences", prepare)
    at = AppTest.from_function(_render_event_view, args=(str(event_session.metadata_path), "central")).run()
    _assert_selector(at, "selectbox", "Neural source", "pc")
    assert next(item for item in at.selectbox if item.label == "PC probe override").value == "Use metadata role"
    assert not any(item.label == "Prepare" for item in at.button)
    prepare.assert_not_called()
    _assert_selector(at, "selectbox", "PC region", "HPC")
    _assert_selector(at, "selectbox", "PC probe override", "B")
    next(item for item in at.button if item.label == "Prepare").click().run()
    assert not at.exception
    prepare.assert_called_once()
    request = prepare.call_args.args[0]
    assert request.region == "HPC" and request.probe_id == "B"


def test_pc_prepare_and_recompute_are_the_only_pc_preparation_actions(event_session, monkeypatch):
    """Input: uncached PC view/preparation spy; only the explicit buttons invoke one PC request each."""
    from src.neural_analysis.webapp import event_sequence_views

    prepare = Mock(wraps=prepare_event_sequences)
    monkeypatch.setattr(event_sequence_views, "prepare_event_sequences", prepare)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    assert not at.exception
    prepare.assert_not_called()
    _assert_selector(at, "selectbox", "Neural source", "pc")
    next(item for item in at.button if item.label == "Prepare").click().run()
    assert not at.exception
    assert prepare.call_count == 1
    assert prepare.call_args_list[0].args[0].metadata_path == event_session.metadata_path
    assert prepare.call_args_list[0].kwargs.get("rerun", False) is False
    next(item for item in at.button if item.label == "Recompute").click().run()
    assert not at.exception
    assert prepare.call_count == 2
    assert [call.kwargs.get("rerun", False) for call in prepare.call_args_list] == [False, True]


def test_pc_prepare_error_is_visible_and_controls_remain_usable(event_session, monkeypatch):
    """Input: uncached PC view/failing explicit preparation; surface message and keep selectors usable."""
    from src.neural_analysis.webapp import event_sequence_views

    prepare = Mock(side_effect=ValueError("Trusted coverage unavailable"))
    monkeypatch.setattr(event_sequence_views, "prepare_event_sequences", prepare)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    assert not at.exception
    _assert_selector(at, "selectbox", "Neural source", "pc")
    next(item for item in at.button if item.label == "Prepare").click().run()
    assert not at.exception
    assert prepare.call_count == 1
    assert any("Trusted coverage unavailable" in item.value for item in at.error)
    assert any(item.label == "Central outcome" for item in at.selectbox)


def test_cached_source_switch_and_rank_change_do_not_prepare_again(event_session, monkeypatch):
    """Input: saved unit and PC products; switch source/rank without another preparation."""
    first_path = _prepare_unit_product(event_session, "A:7")
    second_path = _prepare_unit_product(event_session, "B:7")
    pc_path = Path(prepare_event_sequences(
        EventSequenceRequest(event_session.metadata_path, "pc", region="PFC")
    )["result_path"])
    assert len({first_path, second_path, pc_path}) == 3
    from src.neural_analysis.webapp import event_sequence_views

    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    _assert_selector(at, "selectbox", "Neural source", "unit")
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "save_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(run_session, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(event_sequence_views, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbidden_display_operation)
    monkeypatch.setattr(event_activity, "extract_unit_activity", _forbidden_display_operation)
    _assert_selector(at, "selectbox", "Probe", "A")
    _assert_selector(at, "selectbox", "Unit", "A:7")
    _assert_plotly_output(at)
    _assert_selector(at, "selectbox", "Neural source", "pc")
    selector = next(item for item in at.selectbox if item.label == "Saved source configuration")
    assert all("PCs" in option for option in selector.options)
    selector.select(next(option for option in selector.options if "PFC" in option)).run()
    assert not at.exception
    rank = next(item for item in at.slider if item.label == "PC rank")
    rank.set_value(2).run()
    assert not at.exception
    _assert_plotly_output(at)


def test_same_rank_and_source_survive_tool_switch_and_rank_clamps_for_one_pc(event_session):
    """Input: saved good+MUA and good PC products; preserve PC2 across views and clamp to PC1 for K=1."""
    requests = (
        EventSequenceRequest(event_session.metadata_path, "pc", region="PFC", population_quality="good_mua"),
        EventSequenceRequest(event_session.metadata_path, "pc", region="PFC", population_quality="good"),
    )
    for request in requests:
        prepare_event_sequences(request)
    at = AppTest.from_function(_render_event_tool, args=(str(event_session.metadata_path),)).run()
    assert not at.exception
    source = next(item for item in at.selectbox if item.label == "Saved source configuration")
    good_mua = next(option for option in source.options if "good_mua" in option)
    source.select(good_mua).run()
    assert not at.exception
    rank = next(item for item in at.slider if item.label == "PC rank")
    rank.set_value(2).run()
    assert not at.exception
    assert "PC2" in _plotly_spec(at)["layout"]["meta"]["caption"]

    next(item for item in at.selectbox if item.label == "Test tool").select("transition").run()
    assert not at.exception
    assert next(item for item in at.slider if item.label == "PC rank").value == 2
    assert "PC2" in _plotly_spec(at)["layout"]["meta"]["caption"]

    next(item for item in at.selectbox if item.label == "Test tool").select("central").run()
    source = next(item for item in at.selectbox if item.label == "Saved source configuration")
    good_only = next(option for option in source.options if "good" in option and "good_mua" not in option)
    source.select(good_only).run()
    assert not at.exception
    assert "PC1" in _plotly_spec(at)["layout"]["meta"]["caption"]


def test_transition_direction_and_repeat_selectors_are_display_only(event_session, monkeypatch):
    """Input: saved transition product and forbidden source work; rerun direction/repeat controls from saved values."""
    _prepare_unit_product(event_session)
    _remove_current_sources(event_session)
    from src.neural_analysis.webapp import event_sequence_views

    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "save_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(run_session, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(event_sequence_views, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbidden_display_operation)
    monkeypatch.setattr(event_activity, "extract_unit_activity", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbidden_display_operation)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    assert not at.exception
    _assert_plotly_output(at)
    _assert_selector(at, "selectbox", "Transition direction", "right_to_left")
    repeats = next(item for item in at.slider if item.label == "Incorrect repeats")
    repeats.set_value(4).run()
    assert not at.exception
    _assert_plotly_output(at)


def test_multi_trial_view_defaults_to_ideal_family_and_uses_new_title(event_session):
    """Input: one saved synthetic unit product; assert family view defaults and title render."""
    _prepare_unit_product(event_session)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    assert not at.exception
    assert any(item.value == TRANSITION_VIEW for item in at.subheader)
    family = next(item for item in at.selectbox if item.label == "Sequence family")
    assert family.value == "ideal_transition"


@pytest.mark.parametrize(
    ("family", "expected_control", "expected_options"),
    [
        ("probe", "Initial side", None),
        ("omission_probe", "Initial side", None),
        ("lucky_omission_switch", "Transition direction", ["left_to_right", "right_to_left"]),
        ("indecisive_transition", "Transition direction", ["left_to_right", "right_to_left"]),
    ],
)
def test_family_uses_choice_orientation_or_context_direction(
    event_session, family, expected_control, expected_options
):
    """Input: family slug and saved synthetic unit product; assert its orientation widget contract."""
    _prepare_unit_product(event_session)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    _assert_selector(at, "selectbox", "Sequence family", family)
    control = next(item for item in at.selectbox if item.label == expected_control)
    if expected_options is None:
        assert control.value in ("left", "right")
        assert any("->" in str(option) or "path" in str(option).lower() for option in control.options)
    else:
        assert list(control.options) == expected_options


def test_lucky_family_has_no_repeat_control_and_indecisive_clamps_x(event_session):
    """Input: saved synthetic unit product; assert x widget appears only for x families and clamps to 3."""
    _prepare_unit_product(event_session)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    _assert_selector(at, "selectbox", "Sequence family", "lucky_omission_switch")
    assert not any(item.label == "Incorrect repeats" for item in at.slider)
    _assert_selector(at, "selectbox", "Sequence family", "probe")
    repeats = next(item for item in at.slider if item.label == "Incorrect repeats")
    repeats.set_value(4).run()
    _assert_selector(at, "selectbox", "Sequence family", "indecisive_transition")
    repeats = next(item for item in at.slider if item.label == "Incorrect repeats")
    assert repeats.value == 3


def test_family_switch_reuses_loaded_packet_and_preserves_display_settings(event_session, monkeypatch):
    """Input: saved two-rank PC product; family switches retain source/rank/scaling without loading or preparation."""
    request = EventSequenceRequest(
        event_session.metadata_path, "pc", region="PFC", population_quality="good_mua"
    )
    prepare_event_sequences(request)
    from src.neural_analysis.webapp import event_sequence_views

    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    _assert_selector(at, "selectbox", "Neural source", "pc")
    source = next(item for item in at.selectbox if item.label == "Saved source configuration")
    assert "good_mua" in source.value
    rank = next(item for item in at.slider if item.label == "PC rank")
    rank.set_value(2).run()
    next(item for item in at.checkbox if item.label == "Z-score").check().run()
    assert not at.exception
    selected_source = next(item for item in at.selectbox if item.label == "Saved source configuration").value
    prepare = Mock(side_effect=_forbidden_display_operation)
    monkeypatch.setattr(event_sequence_views, "prepare_event_sequences", prepare)
    monkeypatch.setattr(event_io, "load_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "save_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(run_session, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbidden_display_operation)
    monkeypatch.setattr(event_activity, "extract_unit_activity", _forbidden_display_operation)
    for family in (
        "probe", "omission_probe", "lucky_omission_switch",
        "indecisive_transition", "ideal_transition",
    ):
        _assert_selector(at, "selectbox", "Sequence family", family)
        assert not at.exception
        assert next(item for item in at.selectbox if item.label == "Neural source").value == "pc"
        assert next(item for item in at.selectbox if item.label == "Saved source configuration").value == selected_source
        assert next(item for item in at.slider if item.label == "PC rank").value == 2
        assert next(item for item in at.checkbox if item.label == "Z-score").value is True
    prepare.assert_not_called()


def test_ideal_family_builds_with_explicit_default_parameters(event_session, monkeypatch):
    """Input: saved unit product; require the ideal family builder's stable initial UI parameters."""
    _prepare_unit_product(event_session)
    from src.neural_analysis.event_sequences import plotting

    builder = Mock(wraps=plotting.build_sequence_view)
    monkeypatch.setattr(plotting, "build_sequence_view", builder)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    assert not at.exception
    builder.assert_called_once()
    assert builder.call_args.kwargs == {
        "family": "ideal_transition", "initial_side": "left", "repeats": 1,
    }


def test_zero_spike_good_unit_keeps_raw_plot_and_disables_zscore(event_session):
    """Input: manually good unit with no spikes; raw saved heatmap remains valid and z-score explains zero SD."""
    _empty_one_unit_spike_train(event_session, "A", 7)
    result_path = _prepare_unit_product(event_session, "A:7")
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    assert not at.exception
    zscore = next(item for item in at.checkbox if item.label == "Z-score")
    assert zscore.disabled and zscore.value is False
    assert any(
        "zero" in item.value.lower() or "sd" in item.value.lower()
        for item in list(at.info) + list(at.caption) + list(at.warning)
    )
    _assert_plotly_output(at)
    assert result_path.is_file()


def test_png_export_is_explicit_and_saved_inside_selected_run(event_session):
    """Input: saved product; rendering alone does not export a PNG without an explicit download action."""
    result_path = _prepare_unit_product(event_session)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    assert not at.exception
    assert not list(result_path.parent.rglob("*.png"))
    export_button = next(item for item in at.button if item.label == "Export PNG")
    export_button.click().run()
    assert not at.exception
    figures = result_path.parent / "figures"
    first_export = list(figures.glob("*.png"))
    assert len(first_export) == 1
    assert first_export[0].is_file()
    next(item for item in at.button if item.label == "Export PNG").click().run()
    assert not at.exception
    assert len(list(figures.glob("*.png"))) == 2


def test_pc_saved_cache_invalidates_when_referenced_archive_disappears(event_session):
    """Input: saved PC product; removing its exact PCA reference invalidates cached selection visibly."""
    request = EventSequenceRequest(event_session.metadata_path, "pc", region="PFC")
    result_path = Path(prepare_event_sequences(request)["result_path"])
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "central")
    ).run()
    assert not at.exception
    packet = event_io.load_event_sequences(result_path)
    reference = packet["meta"]["pca_reference"]
    pca_path = event_session.session_root / reference["path"]
    assert pca_path.is_file()
    changed = bytearray(pca_path.read_bytes())
    changed[-1] ^= 1
    pca_path.write_bytes(changed)
    stat = pca_path.stat()
    os.utime(pca_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    at.run()
    assert any(
        "PCA" in item.value and "changed" in item.value.lower()
        for item in list(at.error) + list(at.warning)
    )
    pca_path.unlink()
    at.run()
    assert any("PCA" in item.value and "missing" in item.value.lower() for item in list(at.error) + list(at.warning))


@pytest.mark.parametrize("width", [0.05, 0.02])
def test_fine_bin_pc_controls_prepare_only_after_explicit_request(event_session, width: float) -> None:
    """Input: synthetic session/accepted fine width seconds. Return None; selecting PC 50/20 ms waits for Prepare."""
    original = _prepare_unit_product(event_session)
    at = AppTest.from_function(_render_event_view, args=(str(event_session.metadata_path), "central")).run()
    _assert_selector(at, "selectbox", "Neural source", "pc")
    selector = next(item for item in at.selectbox if item.label == "Preparation bin width")
    assert f"{width * 1000:g} ms" in selector.options
    selector.select(width).run()
    assert not at.exception
    paths = list(event_session.session_root.glob("analysis_runs/event_sequence_heatmaps_*/results.npz"))
    assert paths == [original]
    next(item for item in at.button if item.label == "Prepare").click().run()
    assert not at.exception
    paths = list(event_session.session_root.glob("analysis_runs/event_sequence_heatmaps_*/results.npz"))
    assert len(paths) == 2
    created = next(path for path in paths if path != original)
    packet = event_io.load_event_sequences(created)
    assert packet["meta"]["parameters"]["bin_width_s"] == width
    assert packet["meta"]["source_kind"] == "pc"
    assert len(packet["bin_edges_s"]) - 1 == round(4 / width)
    _assert_plotly_output(at)


def test_additional_selector_version_invalidates_display_load_cache_only(event_session, monkeypatch):
    """Input: synthetic session/patch fixtures. Return None; version changes reload one source.

    Repeated saved loads at one selector version read once. Changing only the
    unitless selector-version string reloads derived matches, preserving source
    configuration, archive bytes and (W,B) Hz values without source computation.
    """
    from src.neural_analysis.event_sequences import selection
    from src.neural_analysis.webapp import event_sequence_views

    path = _prepare_unit_product(event_session)
    product = event_io.discover_event_sequences(event_session.session_root)[0]
    before = path.read_bytes()
    identity = product["meta"]["configuration_id"]
    _remove_current_sources(event_session)
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbidden_display_operation)
    monkeypatch.setattr(event_io, "save_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(run_session, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbidden_display_operation)
    monkeypatch.setattr(event_activity, "extract_unit_activity", _forbidden_display_operation)
    reader = Mock(wraps=event_io.load_event_sequences)
    monkeypatch.setattr(event_io, "load_event_sequences", reader)
    event_sequence_views._load_saved_product_cached.clear()
    try:
        first = event_sequence_views._load_saved_product(product)
        same_version = event_sequence_views._load_saved_product(product)
        assert first is same_version
        assert reader.call_count == 1
        next_version = selection.ADDITIONAL_SELECTOR_VERSION + "-test-revision"
        monkeypatch.setattr(selection, "ADDITIONAL_SELECTOR_VERSION", next_version)
        revised = event_sequence_views._load_saved_product(product)
        assert reader.call_count == 2
        assert revised is not first
        assert revised["additional_selector_version"] == next_version
        assert revised["meta"]["configuration_id"] == identity
        assert revised["meta"] == first["meta"]
        np.testing.assert_array_equal(revised["unit_rates_hz"], first["unit_rates_hz"])
        assert event_sequence_views._load_saved_product(product) is revised
        assert reader.call_count == 2
        assert path.read_bytes() == before
    finally:
        event_sequence_views._load_saved_product_cached.clear()
