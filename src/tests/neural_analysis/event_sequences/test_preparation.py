"""Single-session source preparation, immutable reuse and bounded raw-probe caching."""

from pathlib import Path
import concurrent.futures
import json
import multiprocessing
import re
import shutil
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences import io as event_io
from src.neural_analysis.event_sequences import run_session as preparation
from src.neural_analysis.population import run_session_timeline as timeline_runner
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.neural_analysis.spike_behavior import loading


def _forbid_computation(*args: object, **kwargs: object) -> None:
    """Input: arbitrary loader/fitter arguments. Raise on unauthorized source loading, fitting or binning."""
    raise AssertionError("This operation must reuse existing activity without neural computation")


def _unit_request(session, **settings):
    """Input: resolved session and scalar request overrides. Return explicit probe-qualified unit request."""
    return preparation.EventSequenceRequest(session.metadata_path, "unit", unit_id="A:7", **settings)


@pytest.mark.parametrize("settings", [
    {"source_kind": "other"}, {"source_kind": "unit"},
    {"source_kind": "unit", "unit_id": "7"},
    {"source_kind": "unit", "unit_id": "A:7", "region": "PFC"},
    {"source_kind": "unit", "unit_id": "A:7", "probe_id": "B"},
    {"source_kind": "unit", "unit_id": "A:7", "population_quality": "good_mua"},
    {"source_kind": "pc"}, {"source_kind": "pc", "region": "PFC", "unit_id": "A:7"},
    {"source_kind": "pc", "region": "OTHER"},
    {"source_kind": "pc", "region": "PFC", "population_quality": "automatic"},
    {"source_kind": "unit", "unit_id": "A:7", "bin_width_s": 0.3},
])
def test_python_request_rejects_ambiguous_or_incompatible_source_settings(event_session, settings: dict) -> None:
    """Input: session/invalid scalar settings. Return None; reject ambiguity without any analysis writes."""
    with pytest.raises(ValueError):
        preparation.EventSequenceRequest(event_session.metadata_path, **settings)
    assert not (event_session.session_root / "analysis_runs").exists()


def test_unit_request_resolves_good_quality_and_excludes_display_settings(event_session) -> None:
    """Input: session. Return None; unit policy is manual good and request contains no live display selectors."""
    request = _unit_request(event_session)
    assert request.population_quality == "good"
    for selector in ("pc_rank", "zscore", "central_outcome", "side", "direction", "repeats"):
        assert not hasattr(request, selector)


def test_unit_preparation_uses_manual_good_only_without_any_pca(event_session, monkeypatch) -> None:
    """Input: session/patch fixture. Return None; actual (7,8) unit rates require no timeline fitting/loading."""
    monkeypatch.setattr(timeline_runner, "run_session_timeline", _forbid_computation)
    monkeypatch.setattr(timeline, "compute_timeline_pca", _forbid_computation)
    monkeypatch.setattr(timeline_io, "load_timeline_population", _forbid_computation)
    report = preparation.prepare_event_sequences(_unit_request(event_session))
    assert report["status"] == "prepared"
    packet = event_io.load_event_sequences(report["result_path"])
    assert packet["meta"]["parameters"]["population_quality"] == "good"
    assert packet["meta"]["source_label"] == "A:7"
    assert packet["unit_rates_hz"].shape == (7, 8)
    assert packet["meta"]["pca_reference"] is None
    with pytest.raises(ValueError, match="good|unit"):
        preparation.prepare_event_sequences(preparation.EventSequenceRequest(event_session.metadata_path, "unit", unit_id="A:8"))


def test_manually_good_unit_with_no_assigned_spikes_keeps_covered_silence(event_session) -> None:
    """Input: session. Return None; metadata-defined good unit with S=0 remains a valid raw (7,8) source."""
    path = event_session.session_root / "A/cluster_info.tsv"
    rows = pd.read_csv(path, sep="\t")
    silent = pd.DataFrame({"cluster_id": [12], "ch": [0], "group": ["good"], "KSLabel": ["mua"]})
    pd.concat([rows, silent], ignore_index=True).to_csv(path, sep="\t", index=False)
    request = preparation.EventSequenceRequest(event_session.metadata_path, "unit", unit_id="A:12")
    packet = event_io.load_event_sequences(preparation.prepare_event_sequences(request)["result_path"])
    np.testing.assert_array_equal(packet["unit_rates_hz"], np.zeros((7, 8)))
    assert packet["meta"]["source_label"] == "A:12"


def test_run_directory_contains_reproducible_scripts_log_summary_and_final_npz(event_session) -> None:
    """Input: session. Return None; published run contains actual main/batch snapshots and scientific provenance."""
    report = preparation.prepare_event_sequences(_unit_request(event_session))
    directory = report["result_path"].parent
    assert directory.parent == event_session.session_root / "analysis_runs"
    assert directory.name.startswith("event_sequence_heatmaps_")
    assert re.match(r"^event_sequence_heatmaps_\d{8}_\d{6}", directory.name)
    package = Path(preparation.__file__).parent
    for filename in ("run_session.py", "run_batch.py"):
        assert (directory / filename).read_bytes() == (package / filename).read_bytes()
    summary = (directory / "summary.md").read_text()
    log = (directory / "run.log").read_text()
    assert event_session.session_id in summary and "A:7" in summary
    assert "run_session.py" in summary and "run_batch.py" in summary
    assert event_session.session_id in log and report["configuration_id"] in log
    assert "elapsed" in log.lower() or "execution" in log.lower()


def test_missing_part_two_context_preserves_part_one_and_saved_availability_reason(event_session) -> None:
    """Input: session. Return None; absent context fields disable only transition matching, with eight visible empty motifs."""
    trials = pd.read_csv(event_session.behavior.trial_table_file).drop(columns=["state_int", "cur_block"])
    trials.to_csv(event_session.behavior.trial_table_file, index=False)
    report = preparation.prepare_event_sequences(_unit_request(event_session))
    packet = event_io.load_event_sequences(report["result_path"])
    assert sum(len(positions) for positions in packet["central_matches"].values()) > 0
    assert len(packet["transition_matches"]) == 8
    assert all(positions.shape == (0, key[1] + 2) for key, positions in packet["transition_matches"].items())
    reason = packet["meta"]["transition_unavailable_reason"]
    assert "state_int" in reason and "cur_block" in reason


def test_prepare_saves_behavioral_counts_separately_from_central_and_episode_coverage(event_session) -> None:
    """Input: session. Return None; central-only coverage survives while a missing trigger excludes the full episode."""
    path = event_session.session_root / "A/aligned.npz"
    with np.load(path) as archive:
        spikes = archive["spike_utc_unix"]
    np.savez(path, spike_utc_unix=spikes, irig_utc_unix=[1_700_000_010.0, 1_700_000_040.0])
    packet = event_io.load_event_sequences(preparation.prepare_event_sequences(_unit_request(event_session))["result_path"])
    central = packet["central_summary"]
    transitions = packet["transition_summary"]
    assert central.behavioral_count.sum() == 3 and central.excluded_count.sum() == 1 and central.n.sum() == 2
    assert transitions.behavioral_count.sum() == 1 and transitions.excluded_count.sum() == 1 and transitions.n.sum() == 0


@pytest.mark.parametrize("kind", ["unit", "pc"])
def test_unrelated_missing_probe_does_not_block_selected_source(event_session, kind: str) -> None:
    """Input: session/source kind. Return None; valid probe A prepares independently of missing probe B sources."""
    (event_session.session_root / "B/aligned.npz").unlink()
    (event_session.session_root / "B/channel_quality.csv").unlink()
    request = _unit_request(event_session) if kind == "unit" else preparation.EventSequenceRequest(event_session.metadata_path, "pc", region="PFC")
    report = preparation.prepare_event_sequences(request)
    assert event_io.get_source_activity(event_io.load_event_sequences(report["result_path"])).values.shape == (7, 8)


@pytest.mark.parametrize("quality,units", [(None, ("A:7", "A:8")), ("good", ("A:7",))])
def test_pc_preparation_fits_strict_requested_population_using_metadata_role(event_session, quality, units: tuple) -> None:
    """Input: session/quality/(U,) unit IDs. Return None; PC product references actual strict fitted population."""
    request = preparation.EventSequenceRequest(event_session.metadata_path, "pc", region="PFC", population_quality=quality)
    packet = event_io.load_event_sequences(preparation.prepare_event_sequences(request)["result_path"])
    reference = packet["meta"]["pca_reference"]
    result = timeline_io.load_timeline_result(event_session.session_root / reference["path"])
    assert result.regions["PFC"].transform.input_unit_ids == units
    assert result.meta["missing_label_policy"] == "exclude"
    assert result.meta["population_quality"] == (quality or "good_mua")


def test_pc_probe_override_is_explicit_and_does_not_change_session_metadata(event_session) -> None:
    """Input: session. Return None; explicit PFC override selects B without rewriting metadata role assignments."""
    original = event_session.metadata_path.read_bytes()
    request = preparation.EventSequenceRequest(event_session.metadata_path, "pc", region="PFC", probe_id="B")
    packet = event_io.load_event_sequences(preparation.prepare_event_sequences(request)["result_path"])
    assert packet["meta"]["pca_reference"]["probe_id"] == "B"
    assert event_session.metadata_path.read_bytes() == original


def test_compatible_two_region_pca_reuses_exact_archive_without_refitting(event_session, monkeypatch) -> None:
    """Input: session/patch fixture. Return None; selected region references existing full-population basis unchanged."""
    existing = timeline_runner.run_session_timeline(
        event_session.metadata_path, {"PFC": "A", "HPC": "B"}, missing_label_policy="exclude",
    )
    before = existing.result_path.read_bytes()
    monkeypatch.setattr(timeline_runner, "run_session_timeline", _forbid_computation)
    monkeypatch.setattr(timeline, "compute_timeline_pca", _forbid_computation)
    request = preparation.EventSequenceRequest(event_session.metadata_path, "pc", region="PFC")
    packet = event_io.load_event_sequences(preparation.prepare_event_sequences(request)["result_path"])
    assert event_session.session_root / packet["meta"]["pca_reference"]["path"] == existing.result_path
    assert existing.result_path.read_bytes() == before


def test_normal_reuse_keeps_exact_valid_pca_reference_even_after_new_upstream_run(event_session, monkeypatch) -> None:
    """Input: session/patch fixture. Return None; existing event product retains its exact valid fixed basis."""
    request = preparation.EventSequenceRequest(event_session.metadata_path, "pc", region="PFC")
    first = preparation.prepare_event_sequences(request)
    packet = event_io.load_event_sequences(first["result_path"])
    reference = dict(packet["meta"]["pca_reference"])
    newer = timeline_runner.run_session_timeline(event_session.metadata_path, {"PFC": "A"}, missing_label_policy="exclude", rerun=True)
    assert newer.result_path != event_session.session_root / reference["path"]
    monkeypatch.setattr(timeline_runner, "run_session_timeline", _forbid_computation)
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbid_computation)
    again = preparation.prepare_event_sequences(request)
    assert again["status"] == "reused" and again["result_path"] == first["result_path"]
    assert event_io.load_event_sequences(again["result_path"])["meta"]["pca_reference"] == reference
    recomputed = preparation.prepare_event_sequences(request, rerun=True)
    assert recomputed["status"] == "prepared" and recomputed["result_path"] != first["result_path"]


def test_reuse_is_immutable_and_explicit_recompute_always_creates_new_event_run(event_session, monkeypatch) -> None:
    """Input: session/patch fixture. Return None; saved source/version identity controls reuse without overwriting bytes."""
    request = _unit_request(event_session)
    first = preparation.prepare_event_sequences(request)
    before = first["result_path"].read_bytes()
    original = event_io.load_probe_spike_arrays
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbid_computation)
    again = preparation.prepare_event_sequences(request)
    assert again["status"] == "reused" and again["result_path"] == first["result_path"]
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", original)
    fresh = preparation.prepare_event_sequences(request, rerun=True)
    assert fresh["status"] == "prepared" and fresh["result_path"] != first["result_path"]
    assert fresh["configuration_id"] == first["configuration_id"]
    assert first["result_path"].read_bytes() == before


@pytest.mark.parametrize("change", ["bin", "raw_source", "quality", "version", "missing_product"])
def test_changed_scientific_inputs_and_incomplete_products_are_not_reused(event_session, change: str) -> None:
    """Input: session/change name. Return None; source/parameters/version changes or absent final NPZ force new product."""
    first = preparation.prepare_event_sequences(_unit_request(event_session))
    request = _unit_request(event_session)
    if change == "bin":
        request = _unit_request(event_session, bin_width_s=0.1)
    elif change in {"raw_source", "quality"}:
        path = event_session.session_root / ("A/aligned.npz" if change == "raw_source" else "A/channel_quality.csv")
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "version":
        with np.load(first["result_path"], allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["meta"].item()["analysis_version"] = "previous-analysis-version"
        np.savez_compressed(first["result_path"], **arrays)
    else:
        first["result_path"].unlink()
    again = preparation.prepare_event_sequences(request)
    assert again["status"] == "prepared" and again["result_path"] != first["result_path"]
    if change in {"bin", "raw_source", "quality"}:
        assert again["configuration_id"] != first["configuration_id"]


def test_corrupt_current_source_does_not_overwrite_or_destroy_saved_raw_unit_view(event_session) -> None:
    """Input: session. Return None; invalid new raw source fails readiness while old saved values remain displayable."""
    request = _unit_request(event_session)
    first = preparation.prepare_event_sequences(request)
    before = first["result_path"].read_bytes()
    (event_session.session_root / "A/aligned.npz").write_bytes(b"corrupted source")
    with pytest.raises((ValueError, OSError)):
        preparation.prepare_event_sequences(request)
    assert first["result_path"].read_bytes() == before
    assert event_io.get_source_activity(event_io.load_event_sequences(first["result_path"])).values.shape == (7, 8)


@pytest.mark.parametrize("kind", ["unit", "pc"])
def test_dry_run_starts_no_neural_computation_outputs_or_workers(event_session, monkeypatch, kind: str) -> None:
    """Input: session/patch fixture/source kind. Return None; dry-run plans from small inputs without side effects."""
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbid_computation)
    monkeypatch.setattr(timeline_runner, "run_session_timeline", _forbid_computation)
    monkeypatch.setattr(timeline, "compute_timeline_pca", _forbid_computation)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _forbid_computation)
    monkeypatch.setattr(multiprocessing, "Pool", _forbid_computation)
    request = _unit_request(event_session) if kind == "unit" else preparation.EventSequenceRequest(event_session.metadata_path, "pc", region="PFC")
    report = preparation.prepare_event_sequences(request, dry_run=True)
    assert report["status"] == "dry-run" and report["result_path"] is None and report["configuration_id"]
    assert not (event_session.session_root / "analysis_runs").exists()


def test_probe_array_cache_is_read_only_shared_across_units_bins_and_invalidates_source_changes(event_session, monkeypatch) -> None:
    """Input: session/patch fixture. Return None; actual source identities key read-only (S,) arrays across requests."""
    event_io.clear_probe_spike_cache()
    labels = event_session.session_root / "A/cluster_info.tsv"
    rows = pd.read_csv(labels, sep="\t")
    rows.loc[rows.cluster_id == 8, "group"] = "good"
    rows.to_csv(labels, sep="\t", index=False)
    read_times = Mock(wraps=loading.load_aligned_spikes)
    read_assignments = Mock(wraps=loading.load_sorter_metadata)
    monkeypatch.setattr(loading, "load_aligned_spikes", read_times)
    monkeypatch.setattr(loading, "load_sorter_metadata", read_assignments)
    preparation.prepare_event_sequences(_unit_request(event_session))
    second = preparation.EventSequenceRequest(event_session.metadata_path, "unit", unit_id="A:8")
    preparation.prepare_event_sequences(second)
    preparation.prepare_event_sequences(_unit_request(event_session, bin_width_s=0.1))
    assert read_times.call_count == read_assignments.call_count == 1
    times, assignments = event_io.load_probe_spike_arrays(event_session, "A")
    assert times.shape == assignments.shape and times.ndim == 1
    assert not times.flags.writeable and not assignments.flags.writeable
    with pytest.raises(ValueError):
        times[0] += 1
    path = event_session.session_root / "A/aligned.npz"
    path.write_bytes(path.read_bytes() + b"\n")
    preparation.prepare_event_sequences(_unit_request(event_session))
    assert read_times.call_count == read_assignments.call_count == 2
    path = event_session.session_root / "A/spike_clusters.npy"
    path.write_bytes(path.read_bytes() + b"\n")
    preparation.prepare_event_sequences(_unit_request(event_session))
    assert read_times.call_count == read_assignments.call_count == 3


def test_probe_cache_has_at_most_two_live_probe_sources(event_session, monkeypatch, tmp_path) -> None:
    """Input: session/patch/temp Path. Return None; third distinct probe-source key evicts least-recent cached arrays."""
    event_io.clear_probe_spike_cache()
    copy_root = tmp_path / "another-session"
    shutil.copytree(event_session.session_root, copy_root)
    other_path = copy_root / "neural_session.json"
    other = resolve_session_metadata(load_session_metadata(other_path), other_path)
    read_times = Mock(wraps=loading.load_aligned_spikes)
    monkeypatch.setattr(loading, "load_aligned_spikes", read_times)
    event_io.load_probe_spike_arrays(event_session, "A")
    event_io.load_probe_spike_arrays(event_session, "B")
    event_io.load_probe_spike_arrays(other, "A")
    assert read_times.call_count == 3
    event_io.load_probe_spike_arrays(event_session, "A")
    assert read_times.call_count == 4
