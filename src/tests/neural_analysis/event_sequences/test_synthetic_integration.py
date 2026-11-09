"""End-to-end synthetic acceptance across preparation, saved-only display and export."""

from hashlib import sha256
from unittest.mock import Mock

import numpy as np
import pytest

from src.neural_analysis.event_sequences import io, plotting
from src.neural_analysis.event_sequences.run_session import EventSequenceRequest, prepare_event_sequences
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.spike_behavior import loading
from .test_additional_io import _all_motifs, _freeze_current_sources, _write_trials
from .test_plotting import _all_trials


@pytest.fixture
def complete_motif_session(event_session):
    """Input: two-probe fixture. Return session with every motif and seeded UTC spikes.

    The full (N,C) table includes manual separators. IRIG bounds cover all
    ordinary [-2,+2) windows. Each probe has independent (S,) timestamps and
    assignments for the existing good/MUA channel-quality fixture.
    """
    trials = _all_trials()
    trials.to_csv(event_session.behavior.trial_table_file, index=False)
    start = float(trials.choice_time.min()) - 10
    stop = float(trials.choice_time.max()) + 10
    generator = np.random.default_rng(20261105)
    for probe in event_session.probes:
        folder = probe.sorter_directory
        assignments = np.repeat([7, 8, 9, 10, 11], 6000)
        times = generator.uniform(start, stop, len(assignments))
        order = np.argsort(times)
        np.save(folder / "spike_clusters.npy", assignments[order])
        np.savez(probe.alignment_file, spike_utc_unix=times[order], irig_utc_unix=np.array([start, stop]))
    io.clear_probe_spike_cache()
    return event_session


@pytest.mark.parametrize("source_kind,quality", [("unit", "good"), ("pc", "good"), ("pc", "good_mua")])
@pytest.mark.parametrize("width", [0.5, 0.1])
def test_full_synthetic_preparation_saved_display_and_png(
    complete_motif_session, monkeypatch, source_kind: str, quality: str, width: float,
) -> None:
    """Input: synthetic session/source/manual quality/width seconds. Return None after full artifact acceptance.

    Verify all 24 central and eight transition arrays, all-session population
    SD, fixed PCA identity, repeat preparation reuse, source-free saved reload,
    both renderers and explicit PNG output. Display axes retain (M,B)/(E,S,B).
    """
    session = complete_motif_session
    options = {"unit_id": "A:7"} if source_kind == "unit" else {"region": "PFC" if width == 0.5 else "HPC"}
    request = EventSequenceRequest(session.metadata_path, source_kind, bin_width_s=width, population_quality=quality, **options)
    report = prepare_event_sequences(request)
    assert report["status"] == "prepared"
    path = report["result_path"]
    assert prepare_event_sequences(request)["result_path"] == path
    digest = sha256(path.read_bytes()).hexdigest()
    packet = io.load_event_sequences(path)
    assert len(packet["central_matches"]) == 24 and len(packet["transition_matches"]) == 8
    assert all(len(matches) >= 1 for matches in packet["central_matches"].values())
    assert all(matches.shape[0] == 1 for matches in packet["transition_matches"].values())
    assert packet["central_summary"].excluded_count.sum() == packet["transition_summary"].excluded_count.sum() == 0
    expected_units = 1 if quality == "good" else 2
    if source_kind == "pc":
        regional = packet["timeline_result"].regions[options["region"]]
        assert regional.scores.shape[2] == expected_units
        reference = session.session_root / packet["meta"]["pca_reference"]["path"]
        pca_digest = sha256(reference.read_bytes()).hexdigest()
        assert packet["meta"]["pca_reference"]["sha256"] == pca_digest

    # Saved display must survive unavailable current inputs, rather than
    # silently regenerating behavior, quality, aligned spikes or a PCA basis.
    for probe in session.probes:
        probe.alignment_file.unlink()
        (probe.sorter_directory / "spike_clusters.npy").unlink()
        (probe.sorter_directory / "cluster_info.tsv").unlink()
        probe.channel_quality_file.unlink()
    session.behavior.trial_table_file.unlink()
    forbidden = Mock(side_effect=AssertionError("Saved display attempted scientific work"))
    monkeypatch.setattr(io, "load_probe_spike_arrays", forbidden)
    monkeypatch.setattr(io, "save_event_sequences", forbidden)
    monkeypatch.setattr(loading, "load_aligned_spikes", forbidden)
    monkeypatch.setattr(timeline, "bin_choice_counts", forbidden)
    monkeypatch.setattr(timeline, "fit_timeline_activity", forbidden)
    packet = io.load_event_sequences(path)
    selected = io.get_source_activity(packet, pc_rank=expected_units)
    source = plotting.prepare_display_source(packet, pc_rank=expected_units, zscore=True)
    assert source["session_mean"] == pytest.approx(selected.values.mean())
    assert source["session_sd"] == pytest.approx(selected.values.std(ddof=0))
    for outcome in ("correct_rewarded", "omission"):
        for side in ("left", "right"):
            central = plotting.build_central_view(packet, source, central_outcome=outcome, side=side)
            assert len(central["panels"]) == 6
            assert all(panel["values"].shape[1] == round(4 / width) for panel in central["panels"])
    for direction in ("left_to_right", "right_to_left"):
        for repeats in range(1, 5):
            transition = plotting.build_transition_view(packet, source, direction=direction, repeats=repeats)
            assert transition["panels"][0]["values"].shape == (1, repeats + 2, round(4 / width))
            assert transition["color_limits"] == central["color_limits"]
    interactive = plotting.plotly_figure(central)
    assert len([trace for trace in interactive.data if trace.type == "heatmap"]) == 6
    exported = plotting.export_png(transition, path.parent / "figures" / "synthetic_transition.png", dpi=80)
    assert exported.is_file() and exported.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sha256(path.read_bytes()).hexdigest() == digest
    if source_kind == "pc":
        assert sha256(reference.read_bytes()).hexdigest() == pca_digest
    forbidden.assert_not_called()


@pytest.fixture
def additional_motif_session(event_session):
    """Return a synthetic session and expected integer (E,S) episode positions.

    The full (147,C) table includes all 24 additional conditions and manual
    separators. Deterministic fixture spikes and UTC recording coverage span
    every four-second choice window. No experimental inputs are accessed.
    """
    trials, expected = _all_motifs()
    _write_trials(event_session, trials)
    io.clear_probe_spike_cache()
    return event_session, expected


@pytest.mark.parametrize("source_kind,quality,width", [
    ("unit", "good", 0.5),
    ("pc", "good_mua", 0.1),
    ("pc", "good", 0.05),
    ("unit", "good", 0.02),
])
def test_additional_families_end_to_end_reuse_saved_sources_at_all_bin_widths(
    additional_motif_session, monkeypatch, source_kind: str, quality: str, width: float,
) -> None:
    """Verify full preparation -> offline family display -> seven-segment PNG.

    Inputs are a synthetic session/oracle fixture, pytest patch fixture, source
    kind, manual quality mode and bin width in seconds. Return None. All 24
    conditions use original (E,S) positions and actual (E,S,B) values, B=4/width;
    units remain Hz/PC scores or session SD units. Both PC populations and all
    supported widths are exercised without repeating the PCA benchmark matrix.
    """
    session, expected = additional_motif_session
    options = {"unit_id": "A:7"} if source_kind == "unit" else {"region": "PFC"}
    request = EventSequenceRequest(
        session.metadata_path, source_kind, population_quality=quality,
        bin_width_s=width, **options,
    )
    report = prepare_event_sequences(request)
    path = report["result_path"]
    assert prepare_event_sequences(request)["result_path"] == path
    digest = sha256(path.read_bytes()).hexdigest()
    packet = io.load_event_sequences(path)
    metadata = packet["meta"]
    assert metadata["schema_version"] == 1
    assert metadata["analysis_version"] == "event-sequence-heatmaps-v1"
    with np.load(path, allow_pickle=True) as archive:
        assert not any(name.startswith("additional") for name in archive.files)
    rank = 2 if source_kind == "pc" and quality == "good_mua" else 1
    reference = metadata["pca_reference"]
    pca_path = None if reference is None else session.session_root / reference["path"]
    pca_digest = None if pca_path is None else sha256(pca_path.read_bytes()).hexdigest()
    raw = io.get_source_activity(packet, pc_rank=rank)
    raw_values = raw.values.copy()

    _freeze_current_sources(session, monkeypatch)
    packet = io.load_event_sequences(path)
    assert packet["meta"] == metadata
    assert not packet["additional_unavailable_reason"]
    bins = round(4 / width)
    for zscore in (False, True):
        source = plotting.prepare_display_source(packet, pc_rank=rank, zscore=zscore)
        assert source["session_mean"] == pytest.approx(raw_values.mean())
        assert source["session_sd"] == pytest.approx(raw_values.std(ddof=0))
        central = plotting.build_central_view(packet, source)
        ideal = plotting.build_sequence_view(packet, source)
        assert central["color_limits"] == ideal["color_limits"] == source["color_limits"]
        for (family, side, repeats), rows in expected.items():
            positions = np.asarray(rows, dtype=int)
            np.testing.assert_array_equal(packet["additional_matches"][(family, side, repeats)], positions)
            view = plotting.build_sequence_view(
                packet, source, family=family, initial_side=side, repeats=repeats,
            )
            panel = view["panels"][0]
            assert (panel["behavioral_count"], panel["excluded_count"], panel["n"]) == (len(rows), 0, len(rows))
            np.testing.assert_array_equal(panel["row_positions"], positions)
            indices = np.searchsorted(source["row_positions"], positions)
            np.testing.assert_array_equal(panel["values"], source["values"][indices])
            assert panel["values"].shape == (*positions.shape, bins)
            assert view["color_limits"] == source["color_limits"]
            assert view["configuration_id"] == metadata["configuration_id"]
            assert packet["additional_selector_version"] in view["caption"]

    seven = plotting.build_sequence_view(
        packet, source, family="omission_probe", initial_side="left", repeats=4,
    )
    interactive = plotting.plotly_figure(seven)
    static = plotting.matplotlib_figure(seven)
    assert len([trace for trace in interactive.data if trace.type == "heatmap"]) == 7
    assert len(static.axes[0].images) == 7
    exported = plotting.export_png(seven, path.parent / "figures" / "additional_seven_segments.png", dpi=80)
    assert exported.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sha256(path.read_bytes()).hexdigest() == digest
    if pca_path is not None:
        assert sha256(pca_path.read_bytes()).hexdigest() == pca_digest
    np.testing.assert_array_equal(io.get_source_activity(packet, pc_rank=rank).values, raw_values)
