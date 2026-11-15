"""Sparse source identity, exact archive binding, and live color-range contracts."""

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences import io, plotting, run_session
from src.neural_analysis.population import run_session_timeline as runner
from src.neural_analysis.population import session_timeline_io as timeline_io


def _request(session, **kwargs):
    """Input: resolved session/scalar overrides. Return a PFC sparse request (seconds)."""
    return run_session.EventSequenceRequest(session.metadata_path, "sparse_pc", region="PFC", **kwargs)


def _forbidden(*args, **kwargs):
    """Input: arbitrary arguments. Raise if reuse performs neural preparation."""
    raise AssertionError("Existing sparse source must be reused")


@pytest.mark.parametrize("alpha", [-1, np.nan, np.inf, True, "1"])
def test_sparse_request_rejects_invalid_alpha(event_session, alpha):
    """Input: session/invalid dimensionless alpha. Return None after rejection."""
    with pytest.raises(ValueError):
        _request(event_session, sparse_alpha=alpha)


@pytest.mark.parametrize("kind,settings", [("unit", {"unit_id": "A:7"}), ("pc", {"region": "PFC"})])
def test_old_sources_reject_sparse_parameters(event_session, kind, settings):
    """Input: old source settings. Return None; sparse-only alpha cannot change old identities."""
    with pytest.raises(ValueError):
        run_session.EventSequenceRequest(event_session.metadata_path, kind, sparse_alpha=1, **settings)


def test_sparse_reuses_two_region_archive_and_preserves_exact_reference(event_session, monkeypatch):
    """Input: synthetic session. Return None; saved (7,8,K) SC scores/maps survive offline reload."""
    upstream = runner.run_session_timeline(
        event_session.metadata_path, {"PFC": "A", "HPC": "B"},
        algorithm="sparse_pca", missing_label_policy="exclude",
    )
    assert timeline_io.find_compatible_regional_timeline(event_session, "PFC", "A", .5) is None
    assert timeline_io.find_compatible_regional_timeline(
        event_session, "PFC", "A", .5, algorithm="sparse_pca", sparse_alpha=2,
    ) is None
    monkeypatch.setattr(runner, "run_session_timeline", _forbidden)
    request = _request(event_session)
    report = run_session.prepare_event_sequences(request)
    packet = io.load_event_sequences(report["result_path"])
    assert packet["meta"]["analysis_version"] == "event-sequence-sparse-heatmaps-v1"
    assert packet["meta"]["random_seed"] == 0
    assert event_session.session_root / packet["meta"]["pca_reference"]["path"] == upstream.result_path
    regional = packet["timeline_result"].regions["PFC"]
    source = io.get_source_activity(packet, pc_rank=1)
    assert source.source_kind == "sparse_pc" and source.source_label == "PFC/A SC1"
    np.testing.assert_array_equal(source.values, regional.scores[:, :, 0])
    scaled = plotting.prepare_display_source(packet, zscore=True)
    np.testing.assert_allclose(scaled["values"], (source.values-source.values.mean()) / source.values.std())
    with np.load(report["result_path"], allow_pickle=True) as archive:
        assert not any("scores" in name or "components" in name for name in archive.files)
    assert run_session.prepare_event_sequences(request)["status"] == "reused"
    fresh = run_session.prepare_event_sequences(request, rerun=True)
    assert fresh["result_path"] != report["result_path"]
    event_session.behavior.trial_table_file.unlink()
    for probe in event_session.probes:
        probe.cluster_info_file.unlink()
    assert io.load_event_sequences(report["result_path"])["meta"]["pca_reference"] == packet["meta"]["pca_reference"]


def test_sparse_fallback_and_dry_run(event_session):
    """Input: session without archives. Return None; explicit preparation fits strict capped population."""
    request = _request(event_session, sparse_alpha=.5)
    assert run_session.prepare_event_sequences(request, dry_run=True)["status"] == "dry-run"
    assert not (event_session.session_root / "analysis_runs").exists()
    packet = io.load_event_sequences(run_session.prepare_event_sequences(request)["result_path"])
    fitted = packet["timeline_result"].regions["PFC"].transform
    assert fitted.input_unit_ids == ("A:7", "A:8")
    assert fitted.parameters["alpha"] == .5
    assert fitted.actual_components == min(30, 56, len(fitted.retained_unit_ids))


@pytest.mark.parametrize("kind", ["unit", "pc", "sparse_pc"])
@pytest.mark.parametrize("zscore", [False, True])
def test_manual_range_changes_colors_only(event_session, kind, zscore):
    """Input: source/scaling. Return None; common renderer limits change without clipping (W,B) data."""
    request = (run_session.EventSequenceRequest(event_session.metadata_path, "unit", unit_id="A:7")
               if kind == "unit" else run_session.EventSequenceRequest(event_session.metadata_path, kind, region="PFC"))
    packet = io.load_event_sequences(run_session.prepare_event_sequences(request)["result_path"])
    automatic = plotting.prepare_display_source(packet, zscore=zscore)
    manual = plotting.prepare_display_source(packet, zscore=zscore, color_limit=.01)
    np.testing.assert_array_equal(manual["values"], automatic["values"])
    assert manual["session_mean"] == automatic["session_mean"]
    assert manual["session_sd"] == automatic["session_sd"]
    limits = (0, .01) if kind == "unit" and not zscore else (-.01, .01)
    assert manual["color_limits"] == limits
    for view in (plotting.build_central_view(packet, manual), plotting.build_transition_view(packet, manual)):
        assert "manual" in view["caption"].lower() and "saturat" in view["caption"].lower()
        figure = plotting.plotly_figure(view)
        assert (figure.layout.coloraxis.cmin, figure.layout.coloraxis.cmax) == limits
        export = plotting.matplotlib_figure(view)
        for axis in export.axes:
            for collection in axis.collections:
                if collection.get_clim()[0] is not None:
                    assert collection.get_clim() == limits


@pytest.mark.parametrize("limit", [0, -1, np.nan, np.inf, True, "2"])
def test_manual_limit_rejects_invalid_values(event_session, limit):
    """Input: invalid scalar bound. Return None; values must be real, finite and positive."""
    request = run_session.EventSequenceRequest(event_session.metadata_path, "unit", unit_id="A:7")
    packet = io.load_event_sequences(run_session.prepare_event_sequences(request)["result_path"])
    with pytest.raises(ValueError):
        plotting.prepare_display_source(packet, color_limit=limit)


def test_sparse_lookup_ignores_other_probe_but_detects_changed_input_pool(event_session):
    """Input: session. Return None; all input IDs matter, including silent omitted units."""
    upstream = runner.run_session_timeline(event_session.metadata_path, {"PFC": "A", "HPC": "B"},
                                          algorithm="sparse_pca", missing_label_policy="exclude")
    (event_session.session_root / "B/cluster_info.tsv").unlink()
    def lookup():
        """Input: none. Return compatible sparse archive path for the current selected input IDs."""
        return timeline_io.find_compatible_regional_timeline(event_session, "PFC", "A", .5, algorithm="sparse_pca")
    assert lookup() == upstream.result_path
    path = event_session.session_root / "A/cluster_info.tsv"
    rows = pd.read_csv(path, sep="\t")
    silent = pd.DataFrame({"cluster_id": [12], "ch": [0], "group": ["good"], "KSLabel": ["good"]})
    pd.concat([rows, silent], ignore_index=True).to_csv(path, sep="\t", index=False)
    assert lookup() is None


def test_sparse_reference_rejects_cross_kind_and_byte_changes(event_session):
    """Input: session. Return None; exact source kind and hash protect an otherwise matching reference."""
    report = run_session.prepare_event_sequences(_request(event_session))
    packet = io.load_event_sequences(report["result_path"])
    packet["meta"]["source_kind"] = "pc"
    with pytest.raises(ValueError, match="sparse"):
        io._bind_pca(packet)
    packet["meta"]["source_kind"] = "sparse_pc"
    path = event_session.session_root / packet["meta"]["pca_reference"]["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte identity"):
        io.load_event_sequences(report["result_path"])


def test_zero_sparse_component_keeps_raw_display_but_disables_zscore(event_session):
    """Input: session. Return None; high-alpha zero (7,8) coefficients remain valid raw data."""
    packet = io.load_event_sequences(run_session.prepare_event_sequences(_request(event_session, sparse_alpha=1e9))["result_path"])
    source = plotting.prepare_display_source(packet, color_limit=2)
    np.testing.assert_array_equal(source["values"], np.zeros((7, 8)))
    assert source["color_limits"] == (-2, 2) and not source["zscore_available"]
    assert plotting.prepare_display_source(packet, zscore=True)["values"] is None
