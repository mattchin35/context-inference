"""Dedicated unit controls with automatic, cached single-unit preparation."""

import os
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.event_sequences import io
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.spike_behavior import loading
from src.neural_analysis.webapp import event_sequence_views as view
from src.tests.neural_analysis.test_event_sequence_webapp import (
    event_session, synthetic_timeline_session, _prepare_unit_product,
    _render_event_view, _render_event_tool, _assert_selector,
    _plotly_spec, _remove_current_sources, _forbidden_display_operation,
)


@pytest.fixture
def two_good_units(event_session):
    """Input: synthetic session. Return session with manual-good A:7/A:8 and B:7 on allowed channels."""
    path = event_session.session_root / "A" / "cluster_info.tsv"
    rows = pd.read_csv(path, sep="\t")
    rows.loc[rows.cluster_id.eq(8), "group"] = "good"
    rows.to_csv(path, sep="\t", index=False)
    return event_session


def _unit_products(session) -> list:
    """Input: synthetic session. Return discovered saved unit products, retaining their original activity units/axes."""
    return [product for product in io.discover_event_sequences(session.session_root)
            if product["meta"]["source_kind"] == "unit"]


def test_unit_selector_lists_unprepared_good_units_and_prepares_selected_unit_once(two_good_units, monkeypatch):
    """Input: two-unit session/patch fixture. Return None; selecting A:8 computes only A:8 and reuses probe arrays."""
    session = two_good_units
    _prepare_unit_product(session, "A:7")
    prepare = Mock(wraps=view.prepare_event_sequences)
    monkeypatch.setattr(view, "prepare_event_sequences", prepare)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbidden_display_operation)
    # Preparing A:7 already cached this probe; another unit must not reread it.
    monkeypatch.setattr(loading, "load_aligned_spikes", _forbidden_display_operation)
    at = AppTest.from_function(_render_event_view, args=(str(session.metadata_path), "central")).run()
    assert not at.exception
    assert not any(widget.label in ("Configure a new source", "Saved source configuration", "Prepare")
                   for widget in list(at.checkbox) + list(at.selectbox) + list(at.button))
    units = next(widget for widget in at.selectbox if widget.label == "Unit")
    assert units.options == ["A:7", "A:8"]
    prepare.assert_not_called()
    units.select("A:8").run()
    assert not at.exception
    prepare.assert_called_once()
    assert prepare.call_args.args[0].unit_id == "A:8"
    assert "A:8;" in _plotly_spec(at)["layout"]["meta"]["caption"]
    assert len(_unit_products(session)) == 2
    _assert_selector(at, "selectbox", "Central outcome", "omission")
    next(widget for widget in at.checkbox if widget.label == "Z-score").check().run()
    assert not at.exception and prepare.call_count == 1


def test_cached_unit_switches_do_not_prepare_or_read_sources(two_good_units, monkeypatch):
    """Input: two saved units. Return None; switching cached units uses saved rates without binning or source reads."""
    session = two_good_units
    for unit in ("A:7", "A:8"):
        _prepare_unit_product(session, unit)
    at = AppTest.from_function(_render_event_view, args=(str(session.metadata_path), "central")).run()
    assert not at.exception
    monkeypatch.setattr(view, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(io, "load_probe_spike_arrays", _forbidden_display_operation)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbidden_display_operation)
    monkeypatch.setattr(io, "save_event_sequences", _forbidden_display_operation)
    _assert_selector(at, "selectbox", "Unit", "A:7")
    assert "A:7;" in _plotly_spec(at)["layout"]["meta"]["caption"]
    _assert_selector(at, "selectbox", "Unit", "A:8")
    assert "A:8;" in _plotly_spec(at)["layout"]["meta"]["caption"]


def test_probe_and_bin_controls_prepare_only_the_chosen_unit(two_good_units, monkeypatch):
    """Input: saved A:7 and unprepared B:7. Return None; probe selection alone waits for a unit, width changes prepare once."""
    session = two_good_units
    _prepare_unit_product(session)
    prepare = Mock(wraps=view.prepare_event_sequences)
    monkeypatch.setattr(view, "prepare_event_sequences", prepare)
    at = AppTest.from_function(_render_event_view, args=(str(session.metadata_path), "central")).run()
    _assert_selector(at, "selectbox", "Probe", "B")
    prepare.assert_not_called()
    _assert_selector(at, "selectbox", "Unit", "B:7")
    assert prepare.call_count == 1
    _assert_selector(at, "selectbox", "Bin width", 0.1)
    assert prepare.call_count == 2
    assert [(call.args[0].unit_id, call.args[0].bin_width_s) for call in prepare.call_args_list] == [("B:7", 0.5), ("B:7", 0.1)]
    assert "B:7;" in _plotly_spec(at)["layout"]["meta"]["caption"]
    assert "bin=100 ms" in _plotly_spec(at)["layout"]["meta"]["caption"]


def test_selected_unit_and_width_survive_switching_both_tools(two_good_units, monkeypatch):
    """Input: prepared units. Return None; central/transition share explicit unit/width and never redo preparation."""
    session = two_good_units
    _prepare_unit_product(session)
    prepare = Mock(wraps=view.prepare_event_sequences)
    monkeypatch.setattr(view, "prepare_event_sequences", prepare)
    at = AppTest.from_function(_render_event_tool, args=(str(session.metadata_path),)).run()
    _assert_selector(at, "selectbox", "Unit", "A:8")
    _assert_selector(at, "selectbox", "Bin width", 0.1)
    assert prepare.call_count == 2
    _assert_selector(at, "selectbox", "Test tool", "transition")
    assert next(widget for widget in at.selectbox if widget.label == "Unit").value == "A:8"
    assert next(widget for widget in at.selectbox if widget.label == "Bin width").value == 0.1
    _assert_selector(at, "selectbox", "Transition direction", "right_to_left")
    _assert_selector(at, "selectbox", "Test tool", "central")
    assert prepare.call_count == 2


def test_offline_unit_selectors_offer_saved_units_without_current_metadata_reads(event_session, monkeypatch):
    """Input: two saved probe-qualified units, absent current files. Return None; both remain viewable with saved identities."""
    for unit in ("A:7", "B:7"):
        _prepare_unit_product(event_session, unit)
    _remove_current_sources(event_session)
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(view, "prepare_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(io, "load_probe_spike_arrays", _forbidden_display_operation)
    at = AppTest.from_function(_render_event_view, args=(str(event_session.metadata_path), "central")).run()
    assert not at.exception
    _assert_selector(at, "selectbox", "Probe", "A")
    assert next(widget for widget in at.selectbox if widget.label == "Unit").options == ["A:7"]
    assert "A:7;" in _plotly_spec(at)["layout"]["meta"]["caption"]
    _assert_selector(at, "selectbox", "Probe", "B")
    assert "B:7;" in _plotly_spec(at)["layout"]["meta"]["caption"]


def test_failed_automatic_unit_preparation_waits_for_retry_or_changed_selection(event_session, monkeypatch):
    """Input: failing unit preparation. Return None; display changes never retry raw work automatically."""
    prepare = Mock(side_effect=ValueError("Trusted coverage unavailable"))
    monkeypatch.setattr(view, "prepare_event_sequences", prepare)
    at = AppTest.from_function(_render_event_view, args=(str(event_session.metadata_path), "central")).run()
    assert not at.exception
    prepare.assert_not_called()
    _assert_selector(at, "selectbox", "Unit", "A:7")
    assert prepare.call_count == 1
    assert any("Trusted coverage unavailable" in message.value for message in at.error)
    _assert_selector(at, "selectbox", "Central side", "right")
    assert prepare.call_count == 1
    next(widget for widget in at.button if widget.label == "Retry unit").click().run()
    assert not at.exception and prepare.call_count == 2


def test_repeat_state_is_valid_when_switching_to_shorter_family_range(event_session):
    """Input: saved synthetic unit product; x=4 in probe clamps to x=3 in indecisive transition."""
    _prepare_unit_product(event_session)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    _assert_selector(at, "selectbox", "Sequence family", "probe")
    _assert_selector(at, "selectbox", "Initial side", "right")
    next(item for item in at.slider if item.label == "Incorrect repeats").set_value(4).run()
    assert not at.exception
    _assert_selector(at, "selectbox", "Sequence family", "indecisive_transition")
    indecisive_x = next(item for item in at.slider if item.label == "Incorrect repeats")
    assert indecisive_x.value in (1, 2, 3)
    _assert_selector(at, "selectbox", "Sequence family", "omission_probe")
    omission_x = next(item for item in at.slider if item.label == "Incorrect repeats")
    assert omission_x.value in (1, 2, 3, 4)


@pytest.mark.parametrize("missing_context", [False, True])
def test_all_families_remain_browsable_offline_with_empty_or_missing_matches(
    event_session, monkeypatch, missing_context
):
    """Input: saved offline product with optional missing context columns; retain family-specific availability."""
    if missing_context:
        trials = pd.read_csv(event_session.behavior.trial_table_file)
        trials.drop(columns=["state_int", "cur_block"]).to_csv(
            event_session.behavior.trial_table_file, index=False
        )
    _prepare_unit_product(event_session)
    _remove_current_sources(event_session)
    monkeypatch.setattr(pd, "read_csv", _forbidden_display_operation)
    monkeypatch.setattr(io, "load_probe_spike_arrays", _forbidden_display_operation)
    monkeypatch.setattr(io, "save_event_sequences", _forbidden_display_operation)
    monkeypatch.setattr(view, "prepare_event_sequences", _forbidden_display_operation)
    at = AppTest.from_function(
        _render_event_view, args=(str(event_session.metadata_path), "transition")
    ).run()
    assert not at.exception
    for family in (
        "ideal_transition", "probe", "omission_probe",
        "lucky_omission_switch", "indecisive_transition",
    ):
        _assert_selector(at, "selectbox", "Sequence family", family)
        _assert_plotly_output(at)
        if missing_context and family != "ideal_transition":
            availability_text = " ".join(
                item.value for item in list(at.info) + list(at.warning) + list(at.caption)
            )
            assert "state_int" in availability_text or "cur_block" in availability_text


def test_changed_unit_sources_invalidate_the_saved_selection_once(event_session, monkeypatch):
    """Input: saved unit and changed alignment file identity. Return None; publish one fresh configuration, then reuse it."""
    original = _prepare_unit_product(event_session)
    prepare = Mock(wraps=view.prepare_event_sequences)
    monkeypatch.setattr(view, "prepare_event_sequences", prepare)
    at = AppTest.from_function(_render_event_view, args=(str(event_session.metadata_path), "central")).run()
    prepare.assert_not_called()
    path = event_session.session_root / "A" / "aligned.npz"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    at.run()
    assert not at.exception and prepare.call_count == 1
    assert len(_unit_products(event_session)) == 2 and original.is_file()
    _assert_selector(at, "selectbox", "Central side", "right")
    assert prepare.call_count == 1
