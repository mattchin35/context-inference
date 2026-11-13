"""Paired session inputs and scientific identity reuse existing timeline loaders."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from src.neural_analysis.population import session_cca_io as cca_io
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_probe_sources,
    resolve_session_metadata,
)
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as timeline_session_fixture,
)


synthetic_timeline_session = timeline_session_fixture
ROLES = {"PFC": "A", "HPC": "B"}


def test_common_coverage_preserves_all_behavior_and_actual_choice_time(synthetic_timeline_session):
    """UTC coverage selects one common window without dropping the three behavioral rows."""
    session = synthetic_timeline_session
    origin = 1_700_000_000.0
    _set_coverage(session, "A", origin + np.array([10, 40]))
    _set_coverage(session, "B", origin + np.array([0, 25]))

    inputs = cca_io.load_cca_inputs(session, ROLES, 0.5)

    assert inputs.session_id == "synthetic-timeline"
    assert inputs.time_origin_s == origin
    assert inputs.probe_roles == ROLES
    assert inputs.choices.trial_id.tolist() == [101, 102, 103]
    assert inputs.choices.row_position.tolist() == [0, 1, 2]
    assert inputs.choices.action.tolist() == [1, 0, 1]
    np.testing.assert_array_equal(inputs.choices.reward, [1, 0, 0])
    np.testing.assert_array_equal(inputs.choices.choice_time, origin + np.array([4.123, 18.123, 32.123]))
    np.testing.assert_array_equal(inputs.choice_positions, [1])
    np.testing.assert_array_equal(inputs.coverage_positions["PFC"], [1, 2])
    np.testing.assert_array_equal(inputs.coverage_positions["HPC"], [0, 1])
    assert inputs.unit_ids == {"PFC": ("A:7", "A:8"), "HPC": ("B:7", "B:8")}
    for region in ROLES:
        assert inputs.counts[region].shape == (1, 8, 2)
        assert inputs.counts[region].dtype == np.float64
    np.testing.assert_array_equal(inputs.bin_edges_s, np.linspace(-2, 2, 9))
    assert inputs.events.elapsed_s.max() == 99
    assert inputs.configuration["configuration_id"] == cca_io.cca_configuration(session, ROLES, 0.5)["configuration_id"]


@pytest.mark.parametrize("width,bins", [(0.5, 8), (0.1, 40), (0.05, 80), (0.02, 200)])
def test_counts_use_real_off_grid_choices_and_half_open_edges(synthetic_timeline_session, width, bins):
    """Counts (3,bins,2) use actual UTC edges; +2 s is excluded and gaps stay unbinned."""
    session = synthetic_timeline_session
    choice = 1_700_000_018.123
    times = np.array([choice - 2, choice - 2 + width, choice, choice + width / 2,
                      choice + 2 - width / 2, choice + 2])
    clusters = np.array([7, 7, 7, 8, 7, 7])
    for probe in ("A", "B"):
        source = resolve_probe_sources(session, probe)
        np.save(source.spike_clusters_file, clusters)
        np.savez(source.alignment_file, spike_utc_unix=times,
                 irig_utc_unix=[1_700_000_000.0, 1_700_000_040.0])

    inputs = cca_io.load_cca_inputs(session, ROLES, width)

    expected = np.zeros((3, bins, 2))
    expected[1, [0, 1, bins // 2, bins - 1], 0] = 1
    expected[1, bins // 2, 1] = 1
    for region in ROLES:
        np.testing.assert_array_equal(inputs.counts[region], expected)
    np.testing.assert_array_equal(inputs.choice_positions, [0, 1, 2])
    np.testing.assert_allclose(inputs.bin_edges_s, np.linspace(-2, 2, bins + 1), rtol=0, atol=1e-15)


@pytest.mark.parametrize("quality,missing,expected", [
    ("good_mua", "mua", ("A:7", "A:8")),
    ("good_mua", "exclude", ("A:7",)),
    ("good", "mua", ("A:7",)),
    ("good", "exclude", ("A:7",)),
])
def test_manual_quality_and_historical_missing_labels_are_explicit(
    synthetic_timeline_session, quality, missing, expected,
):
    """Manual labels select the (W,B,U) population; KSLabel cannot rescue a missing label."""
    session = synthetic_timeline_session
    source = resolve_probe_sources(session, "A")
    table = pd.read_csv(source.cluster_info_file, sep="\t")
    table.loc[table.cluster_id == 8, "group"] = ""
    table.to_csv(source.cluster_info_file, sep="\t", index=False)

    inputs = cca_io.load_cca_inputs(session, ROLES, 0.5, population_quality=quality,
                                    missing_label_policy=missing)

    assert inputs.unit_ids["PFC"] == expected
    assert inputs.counts["PFC"].shape == (3, 8, len(expected))
    assert inputs.configuration["population_quality"] == quality
    assert inputs.configuration["missing_label_policy"] == missing


def test_metadata_channel_selection_changes_unit_axis_and_identity(synthetic_timeline_session):
    """Metadata restrictions intersect good inside-brain channels rather than defining new populations."""
    session = synthetic_timeline_session
    before = cca_io.cca_configuration(session, ROLES, 0.5)
    metadata = json.loads(session.metadata_path.read_text())
    metadata["probes"]["A"]["unit_channels"] = [1, 2, 3]
    session = _reload_metadata(session, metadata)

    inputs = cca_io.load_cca_inputs(session, ROLES, 0.5)

    assert inputs.unit_ids["PFC"] == ("A:8",)
    assert inputs.counts["PFC"].shape == (3, 8, 1)
    assert inputs.configuration["selection"]["PFC"]["unit_ids"] == ["A:8"]
    assert inputs.configuration["selection"]["PFC"]["channels"] == [1]
    assert inputs.configuration["configuration_id"] != before["configuration_id"]


@pytest.mark.parametrize("omit_metadata", [False, True])
def test_missing_optional_events_are_explicit_and_do_not_remove_choices(
    synthetic_timeline_session, omit_metadata,
):
    """Absent events produce an explicit source marker, empty table and complete paired counts."""
    session = synthetic_timeline_session
    event_path = session.behavior.event_table_file
    event_path.unlink()
    if omit_metadata:
        metadata = json.loads(session.metadata_path.read_text())
        metadata["behavior"].pop("events")
        session = _reload_metadata(session, metadata)

    inputs = cca_io.load_cca_inputs(session, ROLES, 0.5)

    assert inputs.events.empty
    assert inputs.choices.trial_id.tolist() == [101, 102, 103]
    assert inputs.counts["PFC"].shape == (3, 8, 2)
    event_source = inputs.configuration["sources"]["events"]
    assert event_source["missing"] is True
    assert event_source["path"] == (None if omit_metadata else str(event_path.resolve()))


def test_event_source_appearing_changes_configuration(synthetic_timeline_session):
    """A formerly absent event stream cannot silently reuse a saved behavioral snapshot."""
    session = synthetic_timeline_session
    path = session.behavior.event_table_file
    contents = path.read_text()
    path.unlink()
    absent = cca_io.cca_configuration(session, ROLES, 0.5)
    path.write_text(contents)

    present = cca_io.cca_configuration(session, ROLES, 0.5)

    assert absent["configuration_id"] != present["configuration_id"]
    assert present["sources"]["events"].get("missing", False) is False


def test_input_loading_needs_neither_lfp_nor_pca(synthetic_timeline_session, monkeypatch):
    """Metadata with no LFP files or PCA results still loads direct paired count tensors."""
    monkeypatch.setattr(PCA, "fit", _forbid_analysis)

    inputs = cca_io.load_cca_inputs(synthetic_timeline_session, ROLES, 0.5)

    assert set(inputs.counts) == {"PFC", "HPC"}
    assert not (synthetic_timeline_session.session_root / "analysis_runs").exists()


def test_configuration_reads_no_aligned_spike_or_assignment_values(synthetic_timeline_session, monkeypatch):
    """Small quality/source/IRIG reads are allowed; numerical spike arrays are forbidden."""
    session = synthetic_timeline_session
    with np.load(resolve_probe_sources(session, "A").alignment_file) as archive:
        archive_type = type(archive)
    original_load = np.load
    original_getitem = archive_type.__getitem__

    def guarded_load(path, *args, **kwargs):
        """Reject raw .npy spike arrays; otherwise delegate np.load without changing units."""
        if str(path).endswith(("spike_clusters.npy", "spike_times.npy")):
            pytest.fail("Configuration must not load spike assignments/times")
        return original_load(path, *args, **kwargs)

    def guarded_getitem(archive, member):
        """Permit the IRIG member but reject aligned spike timestamp values."""
        if member == "spike_utc_unix":
            pytest.fail("Configuration must not load aligned spike timestamps")
        return original_getitem(archive, member)

    monkeypatch.setattr(np, "load", guarded_load)
    monkeypatch.setattr(archive_type, "__getitem__", guarded_getitem)

    configuration = cca_io.cca_configuration(session, ROLES, 0.5)

    assert configuration["selection"]["PFC"]["unit_ids"] == ["A:7", "A:8"]
    assert configuration["configuration_id"]
    assert not (session.session_root / "analysis_runs").exists()


@pytest.mark.parametrize("role,attribute", [
    ("behavior", "trial_table_file"), ("behavior", "event_table_file"),
    ("PFC", "alignment_file"), ("PFC", "spike_clusters_file"),
    ("PFC", "cluster_info_file"), ("PFC", "channel_quality_file"),
    ("HPC", "alignment_file"), ("HPC", "spike_clusters_file"),
    ("HPC", "cluster_info_file"), ("HPC", "channel_quality_file"),
])
def test_each_actual_source_identity_changes_the_configuration(synthetic_timeline_session, role, attribute):
    """A source mtime change invalidates the fit identity without corrupting fixture data."""
    session = synthetic_timeline_session
    before = cca_io.cca_configuration(session, ROLES, 0.5)
    source = session.behavior if role == "behavior" else resolve_probe_sources(session, ROLES[role])
    path = getattr(source, attribute)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))

    after = cca_io.cca_configuration(session, ROLES, 0.5)

    assert before["configuration_id"] != after["configuration_id"]


def test_analysis_parameters_change_identity_and_role_order_does_not(synthetic_timeline_session):
    """Bin and explicit population policy define analysis; dictionary insertion order does not."""
    session = synthetic_timeline_session
    baseline = cca_io.cca_configuration(session, ROLES, 0.5)
    variants = [
        cca_io.cca_configuration(session, ROLES, 0.1),
        cca_io.cca_configuration(session, ROLES, 0.5, population_quality="good"),
        cca_io.cca_configuration(session, ROLES, 0.5, missing_label_policy="exclude"),
        cca_io.cca_configuration(session, {"PFC": "B", "HPC": "A"}, 0.5),
    ]

    assert all(item["configuration_id"] != baseline["configuration_id"] for item in variants)
    assert cca_io.cca_configuration(session, {"HPC": "B", "PFC": "A"}, 0.5) == baseline
    assert baseline["bin_width_s"] == 0.5
    assert baseline["window_s"] == [-2.0, 2.0]
    assert not {"viewport", "pair_range", "selected_pair", "color_limits"}.intersection(baseline)


def test_lfp_sites_and_metadata_reformatting_do_not_change_identity(synthetic_timeline_session):
    """Unrelated metadata/LFP changes must not invalidate the same spike analysis."""
    session = synthetic_timeline_session
    before = cca_io.cca_configuration(session, ROLES, 0.5)
    metadata = json.loads(session.metadata_path.read_text())
    metadata["probes"]["A"]["lfp"] = "A/unused_lfp.npy"
    metadata["probes"]["A"]["sites"] = {"unrelated-display-site": 12}
    np.save(session.session_root / "A/unused_lfp.npy", np.zeros((4, 2)))
    session = _reload_metadata(session, metadata)

    after = cca_io.cca_configuration(session, ROLES, 0.5)

    assert before == after


@pytest.mark.parametrize("roles", [None, {}, {"PFC": "A"}, {"PFC": "A", "HPC": "A"},
                                   {"PFC": "A", "HPC": "missing-probe"},
                                   {"PFC": "A", "HPC": "B", "PPC": "B"}])
def test_configuration_requires_exactly_two_distinct_explicit_roles(synthetic_timeline_session, roles):
    """This input API does not infer missing anatomical assignments from other state."""
    with pytest.raises(ValueError):
        cca_io.cca_configuration(synthetic_timeline_session, roles, 0.5)


def test_configuration_rejects_an_unsupported_bin_width(synthetic_timeline_session):
    """A scalar width in seconds must select one of the four documented grids."""
    with pytest.raises(ValueError):
        cca_io.cca_configuration(synthetic_timeline_session, ROLES, 0.03)


def test_no_common_coverage_is_unavailable_instead_of_pairing_different_trials(synthetic_timeline_session):
    """Equal one-window regional lengths do not establish a common trial observation."""
    session = synthetic_timeline_session
    origin = 1_700_000_000.0
    _set_coverage(session, "A", origin + np.array([0, 10]))
    _set_coverage(session, "B", origin + np.array([25, 40]))

    with pytest.raises(ValueError):
        cca_io.load_cca_inputs(session, ROLES, 0.5)


def test_missing_required_population_is_not_a_single_region_substitute(synthetic_timeline_session):
    """A missing HPC aligned recording prevents paired preparation, despite available PFC."""
    session = synthetic_timeline_session
    resolve_probe_sources(session, "B").alignment_file.unlink()

    with pytest.raises((OSError, ValueError)):
        cca_io.load_cca_inputs(session, ROLES, 0.5)


def _set_coverage(session, probe_id: str, coverage: np.ndarray) -> None:
    """Replace only a fixture probe's (2,) UTC-second coverage; preserve aligned spikes."""
    path = resolve_probe_sources(session, probe_id).alignment_file
    with np.load(path) as archive:
        times = archive["spike_utc_unix"]
    np.savez(path, spike_utc_unix=times, irig_utc_unix=coverage)


def _reload_metadata(session, metadata: dict):
    """Write fixture JSON and return its resolved session; paths/IDs remain metadata-native."""
    session.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="ascii")
    return resolve_session_metadata(load_session_metadata(session.metadata_path), session.metadata_path)


def _forbid_analysis(*args, **kwargs):
    """Fail if direct count preparation tries to fit an unrequested analysis; return never."""
    pytest.fail("CCA input loading must not fit PCA or another analysis")
