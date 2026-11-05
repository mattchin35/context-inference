"""Shared metadata-only source identities for runners and saved-result viewers."""

from pathlib import Path

import pytest

from src.neural_analysis.phase_synchrony import preparation, run_session as runner
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.tests.neural_analysis.phase_synchrony.test_preparation import binary_session
from src.tests.neural_analysis.phase_synchrony.test_runner import runnable_session, config


@pytest.mark.parametrize("family,selection,interval", [
    ("native-lfp", ("NamedSite",), None), ("unit-spikes", ("ProbeZ:17",), None),
    ("waveforms", ("NamedSite",), "primary"), ("psd", ("NamedSite",), "primary"),
    ("phase", ("NamedSite",), "primary"), ("power", ("NamedSite",), "primary"),
    ("coherence", ("NamedSite", "OtherSite"), "primary"),
    ("lfp-phase", ("NamedSite", "OtherSite"), "primary"),
    ("spike-ppc", ("ProbeZ:17", "NamedSite"), "primary"),
])
def test_public_source_identity_matches_existing_runner_without_numerical_reads(
    runnable_session, config, family, selection, interval, monkeypatch,
):
    """Extraction preserves all nine existing source dictionaries exactly, using stat-only inputs."""
    session = resolve_session_metadata(load_session_metadata(runnable_session), runnable_session)
    state = runner._RunState(session, config, [], 0.0, {}, [])
    expected = runner._known_identity(state, family, selection, interval)["source_ids"]
    def forbid(*args, **kwargs):
        """Reject numerical file loading; identity inspection needs resolved metadata and stats only."""
        pytest.fail("Source identity opened numerical data")
    monkeypatch.setattr(preparation.np, "load", forbid)
    actual = preparation.product_source_ids(session, family, selection)
    assert actual == expected
    assert "MissingSite" not in actual
    # Independent field/path assertions retain the oracle after the runner delegates.
    site_ids = [] if family == "unit-spikes" else list(selection[1:] if family == "spike-ppc" else selection)
    keys = set(site_ids)
    probe = session.probes[0]
    for site_id in site_ids:
        site = next(site for site in session.sites if site.site_id == site_id)
        assert actual[site_id]["probe_id"] == site.probe_id
        assert actual[site_id]["saved_channel_index"] == site.saved_channel_index
        assert [item["path"] for item in actual[site_id]["files"]] == [
            str(path) for path in (probe.lfp_file, probe.lfp_metadata_file, probe.alignment_file)
        ]
    if family in {"unit-spikes", "spike-ppc"}:
        keys.add("ProbeZ:17")
        assert actual["ProbeZ:17"]["unit_channels"] == probe.unit_channels
        assert [item["path"] for item in actual["ProbeZ:17"]["files"]] == [
            str(path) for path in (probe.alignment_file, probe.spike_clusters_file,
                                  probe.cluster_info_file, probe.channel_quality_file)
        ]
    if interval is not None:
        keys.add("behavior_trials")
        assert actual["behavior_trials"]["path"] == str(session.behavior.trial_table_file)
    assert set(actual) == keys


def test_source_identity_tracks_only_relevant_selection_inputs(runnable_session):
    """Changing unit-quality provenance invalidates selected-unit work while retaining site power."""
    session = resolve_session_metadata(load_session_metadata(runnable_session), runnable_session)
    power = preparation.product_source_ids(session, "power", ("NamedSite",))
    spike = preparation.product_source_ids(session, "spike-ppc", ("ProbeZ:17", "NamedSite"))
    path = session.probes[0].cluster_info_file
    path.write_text(path.read_text() + "\n", encoding="ascii")
    assert preparation.product_source_ids(session, "power", ("NamedSite",)) == power
    assert preparation.product_source_ids(session, "spike-ppc", ("ProbeZ:17", "NamedSite")) != spike
    with pytest.raises(ValueError, match="probe"):
        preparation.product_source_ids(session, "unit-spikes", ("Unknown:17",))
