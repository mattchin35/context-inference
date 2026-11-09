"""Derive Part III matches from immutable existing-schema source products only."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences import activity, io, run_session, selection
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.tests.neural_analysis.event_sequences.test_additional_selection import (
    ADDITIONAL_KEYS, _assert_matches, _episode,
)


def _forbid_source_work(*args: object, **kwargs: object) -> None:
    """Input: arbitrary source-operation arguments. Raise; no reads/fits/writes allowed."""
    raise AssertionError("Derived matches must use saved trials and prepared source activity only")


def _write_trials(session: object, trials: pd.DataFrame) -> None:
    """Write synthetic (N,C) trials to a fixture session, assigning UTC seconds/IDs.

    session is a resolved synthetic session. Choices are five seconds apart to
    keep four-second Pynapple windows separate. Extend its seeded 40-second
    recording deterministically as needed, preserving spike/assignment pairing.
    Trials retain row order and unitless labels. Return None; only setup writes.
    """
    size = len(trials)
    trials = trials.copy(deep=True)
    trials["cur_trial"] = np.arange(1201, 1201 + size)
    trials["choice_time"] = 1_700_000_004.123 + 5 * np.arange(size)
    trials["start_time"] = trials["choice_time"] - 3
    trials.to_csv(session.behavior.trial_table_file, index=False)
    copies = int(np.ceil((5 * size + 10) / 40))
    if copies > 1:
        for probe in ("A", "B"):
            path = session.session_root / probe
            with np.load(path / "aligned.npz") as archive:
                spikes = archive["spike_utc_unix"]
            assignments = np.load(path / "spike_clusters.npy")
            np.save(path / "spike_clusters.npy", np.tile(assignments, copies))
            np.savez(
                path / "aligned.npz",
                spike_utc_unix=np.concatenate([spikes + 40 * index for index in range(copies)]),
                irig_utc_unix=1_700_000_000.0 + np.arange(40 * copies + 1),
            )


def _all_motifs() -> tuple[pd.DataFrame, dict]:
    """Return synthetic (147,C) raw trials and independently listed (E,S) episodes.

    Each of the 24 known family/side/x fixtures is separated by a non-choice
    row. Original positions are unitless; choices get UTC times in _write_trials.
    Indecisive episodes also contain opposite-side x1 probe tails. The final
    unmatched ordinary row belongs to the session scaling pool, not any motif.
    """
    parts = []
    expected = {}
    start = 0
    for key in ADDITIONAL_KEYS:
        episode = _episode(key)
        stop = start + len(episode)
        expected.setdefault(key, []).append(list(range(start, stop)))
        if key[0] == "indecisive_transition":
            other_side = "right" if key[1] == "left" else "left"
            expected.setdefault(("probe", other_side, 1), []).append(list(range(stop - 3, stop)))
        separator = episode.iloc[[-1]].astype(object)
        separator["action"] = "no_choice"
        separator["experimenter_reward_given"] = 1
        parts.extend([episode, separator])
        start = stop + 1
    unmatched = _episode(("probe", "left", 1)).iloc[[0]].copy()
    parts.append(unmatched)
    return pd.concat(parts), expected


def _prepare_product(session: object, kind: str) -> Path:
    """Prepare one true schema-v1 fixture source; return its immutable NPZ Path.

    session is resolved; kind is unit or pc. Unit rates use Hz, PC scores retain
    signed unwhitened units. No behavior condition enters the source request.
    """
    settings = {"unit_id": "A:7"} if kind == "unit" else {"region": "PFC"}
    request = run_session.EventSequenceRequest(session.metadata_path, kind, **settings)
    return Path(run_session.prepare_event_sequences(request)["result_path"])


def _freeze_current_sources(session: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove synthetic raw/CSV/quality files; forbid reads, fitting and publication.

    session is resolved and monkeypatch is pytest's fixture. Return None; saved
    event/PCA archives remain intact with their original units and axes.
    """
    for probe in ("A", "B"):
        for filename in ("aligned.npz", "spike_clusters.npy", "cluster_info.tsv", "channel_quality.csv"):
            (session.session_root / probe / filename).unlink()
    for filename in ("trials.csv", "events.csv"):
        (session.session_root / filename).unlink()
    monkeypatch.setattr(pd, "read_csv", _forbid_source_work)
    monkeypatch.setattr(io, "load_probe_spike_arrays", _forbid_source_work)
    monkeypatch.setattr(io, "save_event_sequences", _forbid_source_work)
    monkeypatch.setattr(run_session, "prepare_event_sequences", _forbid_source_work)
    monkeypatch.setattr(timeline_io, "timeline_configuration", _forbid_source_work)
    monkeypatch.setattr(timeline_io, "load_timeline_behavior", _forbid_source_work)
    monkeypatch.setattr(timeline_io, "load_timeline_population", _forbid_source_work)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbid_source_work)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbid_source_work)
    monkeypatch.setattr(activity, "extract_unit_activity", _forbid_source_work)


@pytest.mark.parametrize("kind", ["unit", "pc"])
def test_existing_source_products_derive_all_matches_offline_without_mutation(event_session, monkeypatch, kind):
    """Input: session/patch/source-kind fixtures. Return None; preserve saved science.

    Check integer (E,S) matches, old matches/counts, (W,B) source/scaling pools,
    unchanged identities/schema and byte-identical event/PCA archives offline.
    """
    trials, expected = _all_motifs()
    _write_trials(event_session, trials)
    path = _prepare_product(event_session, kind)
    baseline = io.load_event_sequences(path)
    before = path.read_bytes()
    before_meta = deepcopy(baseline["meta"])
    assert before_meta["schema_version"] == 1
    assert before_meta["analysis_version"] == "event-sequence-heatmaps-v1"
    assert not any(name.startswith("additional") for name in before_meta)
    with np.load(path, allow_pickle=True) as archive:
        names = set(archive.files)
        assert not any(name.startswith("additional") for name in names)
    pca = None if kind == "unit" else event_session.session_root / before_meta["pca_reference"]["path"]
    pca_before = None if pca is None else pca.read_bytes()
    source_before = io.get_source_activity(baseline)
    scaled_before = activity.scale_activity(source_before, zscore=True)
    _freeze_current_sources(event_session, monkeypatch)

    packet = io.load_event_sequences(path)
    _assert_matches(packet["additional_matches"], expected)
    assert packet["additional_unavailable_reason"] == ""
    assert packet["additional_selector_version"] == selection.ADDITIONAL_SELECTOR_VERSION
    assert packet["meta"] == before_meta
    for name in ("central_matches", "transition_matches"):
        assert list(packet[name]) == list(baseline[name])
        for key in baseline[name]:
            np.testing.assert_array_equal(packet[name][key], baseline[name][key])
    for name in ("central_summary", "transition_summary", "trials", "shared_choices"):
        pd.testing.assert_frame_equal(packet[name], baseline[name])
    for name in ("activity_row_positions", "full_to_activity", "bin_edges_s", "regional_choice_positions"):
        np.testing.assert_array_equal(packet[name], baseline[name])
    source_after = io.get_source_activity(packet)
    np.testing.assert_array_equal(source_after.values, source_before.values)
    np.testing.assert_array_equal(source_after.row_positions, source_before.row_positions)
    assert len(source_after.row_positions) > sum(len(rows) for rows in expected.values())
    scaled_after = activity.scale_activity(source_after, zscore=True)
    for name in ("values", "row_positions", "bin_edges_s"):
        np.testing.assert_array_equal(scaled_after[name], scaled_before[name])
    for name in ("session_mean", "session_sd", "color_limits", "value_units"):
        assert scaled_after[name] == scaled_before[name]
    for key, rows in expected.items():
        assembled = activity.assemble_heatmap(scaled_after, packet["additional_matches"][key])
        assert assembled["behavioral_count"] == assembled["n"] == len(rows)
        assert assembled["excluded_count"] == 0
    assert path.read_bytes() == before
    if pca is not None:
        assert pca.read_bytes() == pca_before
    with np.load(path, allow_pickle=True) as archive:
        assert set(archive.files) == names
    products = io.discover_event_sequences(event_session.session_root)
    assert len(products) == 1 and products[0]["available"]
    assert products[0]["meta"] == before_meta


@pytest.mark.parametrize("columns", [("cur_block",), ("state_int",), ("cur_block", "state_int")])
def test_missing_context_columns_attach_shaped_unavailable_matches_only(event_session, monkeypatch, columns):
    """Input: missing column-name tuple. Return None; all (0,S) arrays and named reason.

    Existing central matches remain usable. The new selector must not be called
    when saved fields are absent; unavailability is established before matching.
    """
    trials = _episode(("probe", "left", 1)).drop(columns=list(columns))
    _write_trials(event_session, trials)
    path = _prepare_product(event_session, "unit")
    before = path.read_bytes()
    selector = Mock(side_effect=_forbid_source_work)
    monkeypatch.setattr(selection, "select_additional_sequences", selector)
    monkeypatch.setattr(io, "select_additional_sequences", selector, raising=False)
    _freeze_current_sources(event_session, monkeypatch)
    packet = io.load_event_sequences(path)
    _assert_matches(packet["additional_matches"], {})
    assert all(column in packet["additional_unavailable_reason"] for column in columns)
    assert packet["additional_selector_version"] == selection.ADDITIONAL_SELECTOR_VERSION
    selector.assert_not_called()
    assert len(packet["central_matches"]) == 24
    assert sum(len(rows) for rows in packet["central_matches"].values()) == 1
    assert path.read_bytes() == before


def test_present_context_with_zero_matches_is_available(event_session, monkeypatch):
    """Input: fixtures. Return None; valid (3,C) all-rewarded trials yield n=0 availability."""
    trials = _episode(("probe", "left", 1))
    trials[["action", "correct", "reward", "state_int"]] = 1
    _write_trials(event_session, trials)
    path = _prepare_product(event_session, "unit")
    _freeze_current_sources(event_session, monkeypatch)
    packet = io.load_event_sequences(path)
    _assert_matches(packet["additional_matches"], {})
    assert packet["additional_unavailable_reason"] == ""
    assert packet["additional_selector_version"] == selection.ADDITIONAL_SELECTOR_VERSION


@pytest.mark.parametrize("column", ["cur_block", "state_int"])
def test_row_missing_context_preserves_unaffected_saved_episode(event_session, monkeypatch, column):
    """Input: context-field string. Return None; a missing row value affects one (1,4) match."""
    key = ("probe", "right", 2)
    trials = pd.concat([_episode(key), _episode(key)]).astype(object)
    trials.iloc[1, trials.columns.get_loc(column)] = "None"
    _write_trials(event_session, trials)
    path = _prepare_product(event_session, "unit")
    _freeze_current_sources(event_session, monkeypatch)
    packet = io.load_event_sequences(path)
    _assert_matches(packet["additional_matches"], {key: [[4, 5, 6, 7]]})
    assert packet["additional_unavailable_reason"] == ""


def test_selector_errors_are_not_reclassified_as_missing_context(event_session, monkeypatch):
    """Input: fixtures. Return None; propagate ValueError unrelated to absent context columns."""
    path = _prepare_product(event_session, "unit")
    selector = Mock(side_effect=ValueError("Invalid derived-selector data contract"))
    monkeypatch.setattr(selection, "select_additional_sequences", selector)
    monkeypatch.setattr(io, "select_additional_sequences", selector, raising=False)
    _freeze_current_sources(event_session, monkeypatch)
    with pytest.raises(ValueError, match="Invalid derived-selector data contract"):
        io.load_event_sequences(path)
    selector.assert_called_once()


def test_existing_packet_validation_precedes_new_match_derivation(event_session, monkeypatch):
    """Input: fixtures. Return None; an invalid saved (N,) row map fails before selector invocation."""
    path = _prepare_product(event_session, "unit")
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["full_to_activity"] = arrays["full_to_activity"].copy()
    arrays["full_to_activity"][1] = 999
    np.savez_compressed(path, **arrays)
    selector = Mock(side_effect=_forbid_source_work)
    monkeypatch.setattr(selection, "select_additional_sequences", selector)
    monkeypatch.setattr(io, "select_additional_sequences", selector, raising=False)
    _freeze_current_sources(event_session, monkeypatch)
    with pytest.raises(ValueError, match="mapping"):
        io.load_event_sequences(path)
    selector.assert_not_called()
