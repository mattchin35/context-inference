"""Seeded two-probe, three-site offline-to-saved-view acceptance (WP7a)."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.phase_synchrony import computation, plotting, preparation, results, run_session as runner
from src.neural_analysis.population.session_timeline import SessionTimelineResult
from src.neural_analysis.population.session_timeline_io import load_timeline_behavior
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.neural_analysis.webapp.phase_synchrony_view import discover_view_products
from src.tests.neural_analysis.phase_synchrony.test_preparation import UTC_ORIGIN, binary_session


@pytest.fixture
def seeded_session(binary_session):
    """Return real synthetic metadata and full config: 2500-Hz uV, three UTC choices, two drifting probes.

    Seed 4405 generates only artificial voltage noise. Binaries store calibrated
    inverse-affine values on independent AP/IRIG clocks; each has (sample,channel)
    layout. Aligned selected-unit times remain unrounded UTC seconds. No
    experimental files are read, copied or modified.
    """
    session, _ = binary_session
    root = session.session_root
    seed = 4405
    rng = np.random.default_rng(seed)
    shutil.copytree(root / "probe", root / "probe_b")
    payload = json.loads(session.metadata_path.read_text())
    original = payload["probes"].pop("ProbeZ")
    payload["probes"] = {"ProbeA": {**original, "sites": {"PFC1": 1}},
                         "ProbeB": {**{key: value.replace("probe/", "probe_b/") if isinstance(value, str) else value
                                       for key, value in original.items()}, "sites": {"HPC1": 1, "HPC2": 0}}}
    payload["site_pairs"] = [["HPC1", "HPC2"], ["PFC1", "HPC1"], ["PFC1", "HPC2"]]
    choices = UTC_ORIGIN + np.array([8.12345, 15.23456, 23.34567])
    spike_times = np.concatenate([choice + np.array([-.31, -.06345, .12789, .43123]) for choice in choices])
    for directory, offset, drift, phase in ((root / "probe", .017, 1.001, 0.), (root / "probe_b", -.021, .998, .6)):
        count, rate = 75_000, 2500.
        utc = offset + np.arange(count) / rate * drift
        volts = np.column_stack([
            np.cos(2 * np.pi * 8 * utc - .4) + .3 * np.cos(2 * np.pi * 42 * utc),
            np.cos(2 * np.pi * 8 * utc + phase) + .3 * np.cos(2 * np.pi * 42 * utc + .2),
        ]) + rng.normal(0, .03, size=(count, 2))
        stored = ((volts - np.array([4., -3.])) / np.array([.195, 2.])).astype(np.float32)
        stored.tofile(directory / "lfp.dat")
        sidecar = json.loads((directory / "lfp_preprocessing.json").read_text())
        sidecar["num_samples_by_segment"] = [count]
        sidecar["synthetic_seed"] = seed
        (directory / "lfp_preprocessing.json").write_text(json.dumps(sidecar), encoding="ascii")
        anchors = np.array([0, count // 2, count])
        np.savez(directory / "sync.npz", irig_sample_ix=944_908 + anchors * 12,
                 irig_utc_unix=UTC_ORIGIN + offset + anchors / rate * drift, spike_utc_unix=spike_times,
                 meta={"continuous_start_sample_ix": 944_908, "sample_rate_hz": 30_000., "synthetic_seed": seed})
        np.save(directory / "ks" / "spike_clusters.npy", np.full(len(spike_times), 17))
    pd.DataFrame({"cur_trial": [31, 44, 63], "start_time": UTC_ORIGIN + np.array([1., 12., 20.]),
                  "choice_time": choices, "action": [0, 1, 0], "reward": [1, 0, 1],
                  "experimenter_reward_given": [0, 0, 0]}).to_csv(root / "trials.csv", index=False)
    session.metadata_path.write_text(json.dumps(payload), encoding="ascii")
    config = results.PhaseSynchronyConfig(components=("power", "coherence", "lfp-phase", "spike-ppc"),
                                          intervals=("primary", "secondary"), unit_ids=("ProbeB:17",), lfp_site="HPC1")
    return session.metadata_path, config


def _forbid_raw(*args, **kwargs):
    """Reject raw LFP reads after all selected native preparation is already saved."""
    pytest.fail("Later component invocation reloaded raw LFP")


def test_seeded_separate_invocations_reuse_and_saved_direct_agreement(seeded_session, monkeypatch):
    """Both intervals/all families share prepared sites across commands and produce faithful saved figures."""
    metadata, config = seeded_session
    power = runner.run_session(metadata, replace(config, components=("power",)))
    assert power.status == 0 and len(power.products) == 6
    monkeypatch.setattr(preparation, "load_site_lfp", _forbid_raw)
    reports = [power]
    for family in ("coherence", "lfp-phase", "spike-ppc"):
        report = runner.run_session(metadata, replace(config, components=(family,)))
        assert report.status == 0
        reports.append(report)
        assert all(stage["status"] != "completed" for stage in report.stages if stage["stage"] == "native-lfp")
    session = resolve_session_metadata(load_session_metadata(metadata), metadata)
    products = discover_view_products(session, config)
    assert len(products) == 20
    assert {product.meta["identity"]["interval"] for product in products} == {"primary", "secondary"}
    for product in products:
        np.testing.assert_array_equal(product.record.trial_ids, [31, 44, 63])
        np.testing.assert_array_equal(product.record.row_positions, [0, 1, 2])
        if product.meta["identity"]["family"] == "coherence":
            assert product.record.band_valid.all()
            expected = 7 if product.meta["identity"]["interval"] == "primary" else 15
            np.testing.assert_array_equal(product.record.segment_count, [expected] * 3)
        elif product.meta["identity"]["family"] == "lfp-phase":
            assert product.record.valid.all()
            expected = 1000 if product.meta["identity"]["interval"] == "primary" else 2000
            np.testing.assert_array_equal(product.record.sample_count, expected)
        elif product.meta["identity"]["family"] == "spike-ppc":
            assert product.record.valid.all()
            np.testing.assert_array_equal(product.record.spike_count, 4)
    # Direct calculation from the exact saved waveform/phase must agree with
    # saved reductions; this checks the complete persistence/execution boundary.
    headers = results.discover_product_metadata(session.session_root)
    waveform = results.load_product(next(header.path for header in headers if header.meta["identity"]["family"] == "waveforms"
                                        and header.meta["identity"]["selection"] == ("HPC1",)
                                        and header.meta["identity"]["interval"] == "primary")).record
    direct = computation.compute_band_power(computation.compute_site_psd(waveform, config), config)
    saved = next(product.record for product in products if product.meta["identity"]["family"] == "power"
                 and product.meta["identity"]["selection"] == ("HPC1",) and product.meta["identity"]["interval"] == "primary")
    np.testing.assert_allclose(saved.band_power_integrated, direct.band_power_integrated, rtol=0, atol=0)
    incremental = runner.run_session(metadata, replace(config, components=("power",)), rerun=True)
    assert incremental.status == 0 and len(incremental.products) == 6
    assert all(stage["status"] == "reused" for stage in incremental.stages if stage["stage"] == "psd")
    assert set(incremental.products).isdisjoint(power.products)
    warm = runner.run_session(metadata, config)
    assert warm.status == 0 and len(warm.products) == 20
    assert all(stage["status"] == "reused" for stage in warm.stages if stage["stage"] in config.components)
    origin, choices, events = load_timeline_behavior(session)
    behavior = SessionTimelineResult(session.session_id, origin, choices, events, {}, {}, {})
    figure = plotting.plot_synchrony_timeline(behavior, products)
    assert any(trace.meta and trace.meta.get("kind") == "metric" for trace in figure.data)
    assert figure.to_json()
    print(f"SYNTHETIC_SESSION={metadata}")


def test_seeded_failure_restart_keeps_early_power_and_independent_pairs(seeded_session, monkeypatch):
    """A real pair failure sees already published power and does not discard unrelated pair outputs."""
    metadata, config = seeded_session
    original = computation.compute_welch_coherence
    observations = []
    def fail_one_pair(left, right, *args, **kwargs):
        """Inspect published headers before one deliberate pair-local error; otherwise use the actual estimator."""
        if (left.site_id, right.site_id) == ("HPC1", "HPC2"):
            powers = results.discover_product_metadata(metadata.parent, family="power")
            observations.append(len(powers))
            raise ValueError("Synthetic pair failure")
        return original(left, right, *args, **kwargs)
    monkeypatch.setattr(computation, "compute_welch_coherence", fail_one_pair)
    report = runner.run_session(metadata, replace(config, components=("power", "coherence")))
    assert report.status == 1 and len(report.products) == 10
    assert observations == [3, 6]
    original_powers = tuple(path for path in report.products if results.read_product_metadata(path)["identity"]["family"] == "power")
    content = {path: path.read_bytes() for path in original_powers}
    monkeypatch.setattr(computation, "compute_welch_coherence", original)
    monkeypatch.setattr(preparation, "load_site_lfp", _forbid_raw)
    restart = runner.run_session(metadata, replace(config, components=("coherence",)))
    assert restart.status == 0 and len(restart.products) == 6
    stages = [stage for stage in restart.stages if stage["stage"] == "coherence"]
    assert sum(stage["status"] == "reused" for stage in stages) == 4
    assert sum(stage["status"] == "completed" for stage in stages) == 2
    assert all(path.read_bytes() == before for path, before in content.items())
