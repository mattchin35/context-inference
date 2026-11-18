"""WP5a sequential execution, durable publication, reuse and exclusive timing contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import re
from threading import Event

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.phase_synchrony import computation, preparation, results
from src.neural_analysis.phase_synchrony import run_session as runner
from src.neural_analysis.phase_synchrony.results import PhaseSynchronyConfig
from src.tests.neural_analysis.phase_synchrony.test_preparation import UTC_ORIGIN, binary_session


@pytest.fixture
def runnable_session(binary_session) -> Path:
    """Return real temporary OE metadata with one ordinary UTC choice and two designated LFP channels.

    Binary data are calibrated native 2500-Hz values from WP1; an unrelated
    third probe is deliberately missing. No experimental session is accessed.
    """
    session, _ = binary_session
    payload = json.loads(session.metadata_path.read_text())
    payload["probes"]["ProbeZ"]["sites"]["OtherSite"] = 0
    payload["probes"]["MissingProbe"] = {
        "lfp": "missing/lfp.dat", "alignment": "missing/sync.npz",
        "sorter": "missing/ks", "quality": "missing/channel_quality.csv", "sites": {"MissingSite": 0},
    }
    payload["site_pairs"] = [["NamedSite", "OtherSite"]]
    session.metadata_path.write_text(json.dumps(payload), encoding="ascii")
    pd.DataFrame({
        "cur_trial": [21], "start_time": [UTC_ORIGIN], "choice_time": [UTC_ORIGIN + 5.12345],
        "action": [0], "reward": [1], "experimenter_reward_given": [0],
    }).to_csv(session.behavior.trial_table_file, index=False)
    return session.metadata_path


@pytest.fixture
def config() -> PhaseSynchronyConfig:
    """Select two real sites, primary seconds and small 8-Hz CWT for execution tests."""
    return PhaseSynchronyConfig(
        components=("power",), sites=("NamedSite", "OtherSite"),
        pairs=(("NamedSite", "OtherSite"),), notch_enabled=False, phase_frequencies_hz=(8.0,),
    )


def _product_families(report) -> list[str]:
    """Return requested durable-output categorical families by metadata only."""
    return [results.read_product_metadata(path)["identity"]["family"] for path in report.products]


def _requested_stages(report) -> list[dict]:
    """Return requested final-metric stage dictionaries, excluding shared preparation work."""
    return [stage for stage in report.stages if stage["stage"] in {"power", "coherence", "lfp-phase", "spike-ppc"}]


def _forbid_work(*args, **kwargs):
    """Reject any numerical work arguments; this stage is outside the requested prerequisites."""
    raise AssertionError("Unrequested numerical work executed")


def test_power_only_one_site_needs_no_other_probe_pairs_spikes_or_cwt(runnable_session, config, monkeypatch):
    """A single site's saved power requires only native/waveform/PSD inputs and original behavior."""
    for name in ("compute_welch_coherence", "compute_lfp_phase_summary", "compute_selected_unit_ppc"):
        monkeypatch.setattr(computation, name, _forbid_work)
    monkeypatch.setattr(preparation, "prepare_site_phase", _forbid_work)
    monkeypatch.setattr(preparation, "load_selected_unit_spikes", _forbid_work)
    selected = replace(config, sites=("NamedSite",), pairs=())
    report = runner.run_session(runnable_session, selected)
    assert report.status == 0 and _product_families(report) == ["power"]
    product = results.load_product(report.products[0]).record
    np.testing.assert_array_equal(product.trial_ids, [21])
    assert product.valid.all()
    assert all("MissingSite" not in stage["selection"] for stage in report.stages)


def test_coherence_only_does_not_integrate_power_or_prepare_phase(runnable_session, config, monkeypatch):
    """Matched autos/cross spectra prepare coherence without band-power/CWT/spike stages."""
    monkeypatch.setattr(computation, "compute_band_power", _forbid_work)
    monkeypatch.setattr(preparation, "prepare_site_phase", _forbid_work)
    monkeypatch.setattr(preparation, "load_selected_unit_spikes", _forbid_work)
    report = runner.run_session(runnable_session, replace(config, components=("coherence",)))
    assert report.status == 0 and _product_families(report) == ["coherence"]
    assert len([stage for stage in report.stages if stage["stage"] == "psd" and stage["status"] == "completed"]) == 2


def test_selected_spike_only_prepares_target_unit_and_lfp_site(runnable_session, config, monkeypatch):
    """Explicit unit/site PPC does not require pair metrics, autos, power or an unrelated LFP channel."""
    for name in ("compute_site_psd", "compute_band_power", "compute_welch_coherence", "compute_lfp_phase_summary"):
        monkeypatch.setattr(computation, name, _forbid_work)
    selected = replace(config, components=("spike-ppc",), unit_ids=("ProbeZ:17",), lfp_site="NamedSite")
    report = runner.run_session(runnable_session, selected)
    assert report.status == 0 and _product_families(report) == ["spike-ppc"]
    assert all(stage["selection"] != ("OtherSite",) for stage in report.stages)
    assert results.load_product(report.products[0]).record.unit_id == "ProbeZ:17"


def test_primary_sweep_precedes_secondary_even_with_explicit_family_order(runnable_session, config):
    """The explicit order applies within each sweep; requesting secondary first cannot delay primary products."""
    selected = replace(config, components=("power", "coherence"), intervals=("secondary", "primary"))
    report = runner.run_session(runnable_session, selected, order=("coherence", "power"))
    assert report.status == 0
    stages = [(stage["stage"], stage["interval"]) for stage in _requested_stages(report)]
    assert stages == [
        ("coherence", "primary"), ("power", "primary"), ("power", "primary"),
        ("coherence", "secondary"), ("power", "secondary"), ("power", "secondary"),
    ]


def _held_call(entered: Event, release: Event, original, *args, **kwargs):
    """Hold an actual numerical function at a deterministic boundary, then return its real result."""
    entered.set()
    if not release.wait(timeout=30):
        raise RuntimeError("Test did not release controlled stage")
    return original(*args, **kwargs)


def _only_run_directory(root: Path) -> Path:
    """Return the sole current run directory in this isolated synthetic session."""
    directories = list((root / "analysis_runs").glob("session_timeline_phase_synchrony_*"))
    assert len(directories) == 1
    return directories[0]


def test_first_usable_product_is_durable_before_later_phase_preparation_finishes(runnable_session, config, monkeypatch):
    """Controlled holds prove prerequisites don't count as first requested output and power publishes before CWT."""
    power_entered, power_release, phase_entered, phase_release = Event(), Event(), Event(), Event()
    monkeypatch.setattr(computation, "compute_band_power", partial(_held_call, power_entered, power_release, computation.compute_band_power))
    monkeypatch.setattr(preparation, "prepare_site_phase", partial(_held_call, phase_entered, phase_release, preparation.prepare_site_phase))
    selected = replace(config, components=("power", "lfp-phase"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner.run_session, runnable_session, selected)
        try:
            assert power_entered.wait(timeout=30)
            directory = _only_run_directory(runnable_session.parent)
            measurements = json.loads((directory / "measurements.json").read_text())
            assert measurements["timings_s"]["first_product_s"] is None
            assert not results.discover_product_metadata(runnable_session.parent, family="power")
            power_release.set()
            assert phase_entered.wait(timeout=30)
            published = results.discover_product_metadata(runnable_session.parent, family="power")
            assert len(published) == 2 and all(product.path.is_file() for product in published)
            measurements = json.loads((directory / "measurements.json").read_text())
            assert measurements["timings_s"]["first_product_s"] is not None
            assert not future.done()
        finally:
            power_release.set()
            phase_release.set()
        report = future.result(timeout=30)
    assert report.status == 0


def test_shared_psd_cost_is_charged_once_and_measured_outputs_are_saved(runnable_session, config):
    """Power/coherence share two auto-PSD stages; exclusive timings, actual dimensions and run snapshots persist."""
    report = runner.run_session(runnable_session, replace(config, components=("power", "coherence")))
    assert report.status == 0 and report.run_directory.parent == runnable_session.parent / "analysis_runs"
    assert len(report.products) == 3
    psd_stages = [stage for stage in report.stages if stage["stage"] == "psd" and stage["status"] == "completed"]
    assert len(psd_stages) == 2
    assert {tuple(stage["selection"]) for stage in psd_stages} == {("NamedSite",), ("OtherSite",)}
    assert all(stage["counts"]["trial_count"] == 1 and stage["counts"]["segment_count"] == 7 for stage in psd_stages)
    assert all(stage["counts"]["sample_count"] == 5000 and stage["counts"]["frequency_count"] == 626 for stage in psd_stages)
    assert all(stage["counts"]["array_bytes"] > 0 for stage in psd_stages)
    assert all(stage["elapsed_s"] >= 0 for stage in report.stages)
    assert sum(stage["elapsed_s"] for stage in report.stages) <= report.timings_s["total_s"] + 1e-6
    assert 0 <= report.timings_s["first_product_s"] <= report.timings_s["total_s"]
    directory = report.run_directory
    assert {"config.json", "run.log", "summary.md", "measurements.json", "run_session.py"}.issubset({path.name for path in directory.iterdir()})
    assert (directory / "run_session.py").read_bytes() == Path(runner.__file__).read_bytes()
    batch_source = Path(runner.__file__).with_name("run_batch.py")
    assert (directory / "run_batch.py").exists() == batch_source.exists()
    measurements = json.loads((directory / "measurements.json").read_text())
    assert measurements["timings_s"] == report.timings_s
    assert measurements["hardware"]["cpu_count"] >= 1 and measurements["hardware"]["platform"]
    assert "power" in (directory / "summary.md").read_text().lower()
    assert "synthetic_2030-01-02_030405" in (directory / "run.log").read_text()


def _fail_selected_site(original, session, site_id, **kwargs):
    """Fail one actual site read; other sites still use the real calibrated loader."""
    if site_id == "NamedSite":
        raise OSError("synthetic NamedSite read failure")
    return original(session, site_id, **kwargs)


def test_failed_site_skips_dependents_and_retains_independent_completed_power(runnable_session, config, monkeypatch):
    """An I/O failure records nonzero status but cannot erase an independent site's published power."""
    monkeypatch.setattr(preparation, "load_site_lfp", partial(_fail_selected_site, preparation.load_site_lfp))
    report = runner.run_session(runnable_session, replace(config, components=("power", "coherence")))
    assert report.status != 0
    assert len(report.products) == 1
    product = results.load_product(report.products[0])
    assert product.meta["identity"]["selection"] == ("OtherSite",)
    assert any(stage["status"] == "failed" and "synthetic NamedSite" in stage["error"] for stage in report.stages)
    assert any(stage["stage"] == "coherence" and stage["status"] == "skipped" for stage in report.stages)
    saved = json.loads((report.run_directory / "measurements.json").read_text())
    assert any(stage["status"] == "failed" for stage in saved["stages"])
    assert "synthetic NamedSite read failure" in (report.run_directory / "run.log").read_text()


def test_missing_trial_support_is_saved_unavailable_data_not_execution_failure(runnable_session, config):
    """A boundary choice keeps behavior identity and unavailable power without turning data support into I/O failure."""
    trials_path = runnable_session.parent / "trials.csv"
    trials = pd.read_csv(trials_path)
    trials["choice_time"] = UTC_ORIGIN + 0.25
    trials.to_csv(trials_path, index=False)
    report = runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()))
    assert report.status == 0 and len(report.products) == 1
    power = results.load_product(report.products[0]).record
    np.testing.assert_array_equal(power.trial_ids, [21])
    assert not power.valid.any() and np.isnan(power.band_power_integrated).all()
    assert not any(stage["status"] == "failed" for stage in report.stages)


def test_new_unit_reuses_phase_and_native_preparation_without_raw_or_cwt_work(runnable_session, config, monkeypatch):
    """Changing only explicit unit selection reuses matching site phase while creating independent unit PPC."""
    selected = replace(config, components=("spike-ppc",), unit_ids=("ProbeZ:17",), lfp_site="NamedSite")
    first = runner.run_session(runnable_session, selected)
    before = {path: path.read_bytes() for path in first.products}
    monkeypatch.setattr(preparation, "load_site_lfp", _forbid_work)
    monkeypatch.setattr(preparation, "prepare_site_phase", _forbid_work)
    second = runner.run_session(runnable_session, replace(selected, unit_ids=("ProbeZ:29",)))
    assert second.status == 0 and second.run_directory != first.run_directory
    assert any(stage["stage"] == "phase" and stage["status"] == "reused" for stage in second.stages)
    assert results.load_product(second.products[0]).record.unit_id == "ProbeZ:29"
    assert all(path.read_bytes() == content for path, content in before.items())


def _record_psd_intervals(seen: list[str], original, waveforms, config):
    """Record actual PSD interval calls and return the unchanged real numerical result."""
    seen.append(waveforms.meta["interval"])
    return original(waveforms, config)


def test_additional_interval_preserves_primary_and_additive_rerun_preserves_old_products(runnable_session, config, monkeypatch):
    """Secondary preparation does not recompute/replace a completed primary; explicit rerun creates another immutable run."""
    selected = replace(config, sites=("NamedSite",), pairs=())
    first = runner.run_session(runnable_session, selected)
    original = first.products[0]
    content = original.read_bytes()
    seen = []
    monkeypatch.setattr(computation, "compute_site_psd", partial(_record_psd_intervals, seen, computation.compute_site_psd))
    second = runner.run_session(runnable_session, replace(selected, intervals=("primary", "secondary")))
    assert second.status == 0 and seen == ["secondary"]
    assert original in second.products and original.read_bytes() == content
    assert any(stage["stage"] == "power" and stage["interval"] == "primary" and stage["status"] == "reused" for stage in second.stages)
    rerun = runner.run_session(runnable_session, selected, rerun=True)
    assert rerun.status == 0 and rerun.products[0] != original
    assert rerun.run_directory not in (first.run_directory, second.run_directory)
    assert original.read_bytes() == content


@pytest.mark.parametrize("prepare_only", ["native-lfp", "phase"])
def test_preparation_only_does_not_compute_unrequested_metrics(runnable_session, config, monkeypatch, prepare_only):
    """Preparation-only actions publish reusable native/phase outputs and skip every metric family."""
    for name in ("compute_site_psd", "compute_band_power", "compute_welch_coherence", "compute_lfp_phase_summary", "compute_selected_unit_ppc"):
        monkeypatch.setattr(computation, name, _forbid_work)
    if prepare_only == "native-lfp":
        monkeypatch.setattr(preparation, "prepare_site_phase", _forbid_work)
    report = runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()), prepare_only=prepare_only)
    assert report.status == 0 and _product_families(report) == [prepare_only]


REAL_NPZ_GETITEM = np.lib.npyio.NpzFile.__getitem__


def _metadata_only_getitem(archive, key):
    """Permit metadata/sidecar access while rejecting every raw or saved numerical NPZ array."""
    assert key == "meta", f"Dry-run read a numerical array: {key}"
    return REAL_NPZ_GETITEM(archive, key)


def test_cold_and_warm_dry_run_read_only_metadata_and_never_write_outputs(runnable_session, config, monkeypatch):
    """Both missing/reusable planning states require no raw/saved scientific arrays, transformations or writes."""
    selected = replace(config, sites=("NamedSite",), pairs=())
    with monkeypatch.context() as cold:
        cold.setattr(preparation, "load_site_lfp", _forbid_work)
        cold.setattr(pd, "read_csv", _forbid_work)
        cold.setattr(np, "memmap", _forbid_work)
        cold.setattr(np.lib.npyio.NpzFile, "__getitem__", _metadata_only_getitem)
        report = runner.run_session(runnable_session, selected, dry_run=True)
    assert report.status == 0 and report.run_directory is None and not report.products
    assert not (runnable_session.parent / "analysis_runs").exists()
    assert not (runnable_session.parent / "processed").exists()
    actual = runner.run_session(runnable_session, selected)
    before = {path: path.read_bytes() for path in runnable_session.parent.rglob("*") if path.is_file()}
    monkeypatch.setattr(preparation, "load_site_lfp", _forbid_work)
    monkeypatch.setattr(pd, "read_csv", _forbid_work)
    monkeypatch.setattr(results, "load_product", _forbid_work)
    monkeypatch.setattr(np, "memmap", _forbid_work)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", _metadata_only_getitem)
    warm = runner.run_session(runnable_session, selected, dry_run=True)
    assert warm.run_directory is None and not warm.products and warm.status == 0
    assert any(stage["stage"] == "power" and stage["status"] == "reusable" for stage in warm.stages)
    assert before == {path: path.read_bytes() for path in runnable_session.parent.rglob("*") if path.is_file()}
    assert actual.run_directory.is_dir()


def test_native_only_preparation_needs_no_behavior_file(runnable_session, config, monkeypatch):
    """Continuous native uV/clock preparation depends on metadata/LFP, not trial or event tables."""
    (runnable_session.parent / "trials.csv").unlink()
    monkeypatch.setattr(runner.timeline_io, "load_timeline_behavior", _forbid_work)
    report = runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()), prepare_only="native-lfp")
    assert report.status == 0 and _product_families(report) == ["native-lfp"]


@pytest.mark.parametrize("failure", ["missing", "malformed"])
def test_behavior_input_failure_is_logged_and_returned_without_signal_work(runnable_session, config, monkeypatch, failure):
    """Shared behavior failure yields a nonzero report, durable failed input stage and summary."""
    trials = runnable_session.parent / "trials.csv"
    if failure == "missing":
        trials.unlink()
    else:
        trials.write_text("unexpected_column\n1\n", encoding="ascii")
    monkeypatch.setattr(preparation, "load_site_lfp", _forbid_work)
    report = runner.run_session(runnable_session, config)
    assert report.status != 0 and not report.products
    failed = [stage for stage in report.stages if stage["stage"] == "input-setup" and stage["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["error"]
    assert failed[0]["error"] in (report.run_directory / "run.log").read_text()
    saved = json.loads((report.run_directory / "measurements.json").read_text())
    assert any(stage["status"] == "failed" for stage in saved["stages"])
    assert "failed" in (report.run_directory / "summary.md").read_text().lower()


def test_unknown_unit_probe_is_a_reported_input_failure(runnable_session, config):
    """A syntactically valid but unknown probe-qualified ID never escapes as an internal KeyError."""
    selected = replace(config, components=("spike-ppc",), unit_ids=("UnknownProbe:17",), lfp_site="NamedSite")
    report = runner.run_session(runnable_session, selected)
    assert report.status != 0 and not report.products
    assert any(stage["status"] == "failed" and "probe" in stage["error"].lower() for stage in report.stages)


def test_empty_selected_lfp_path_preserves_independent_site_output(runnable_session, config):
    """An explicitly empty missing site's path is local input failure, not a session-wide TypeError."""
    payload = json.loads(runnable_session.read_text())
    payload["probes"]["MissingProbe"]["lfp"] = ""
    runnable_session.write_text(json.dumps(payload), encoding="ascii")
    selected = replace(config, sites=("NamedSite", "MissingSite"), pairs=())
    report = runner.run_session(runnable_session, selected)
    assert report.status != 0 and len(report.products) == 1
    assert results.read_product_metadata(report.products[0])["identity"]["selection"] == ("NamedSite",)


def test_phase_and_spike_measurements_retain_actual_support_and_array_dimensions(runnable_session, config):
    """Measurements distinguish 1000 metric times from one guard and physical spikes from frequency counts."""
    selected = replace(config, components=("spike-ppc",), unit_ids=("ProbeZ:17",), lfp_site="NamedSite")
    report = runner.run_session(runnable_session, selected)
    phase = next(stage for stage in report.stages if stage["stage"] == "phase")
    assert phase["counts"]["metric_sample_count_per_trial"] == 1000
    assert phase["counts"]["time_sample_count_including_guard"] == 1001
    arrays = phase["counts"]["major_arrays"]
    assert arrays["phase"]["shape"] == [1, 1, 1001]
    assert arrays["phase"]["dtype"] == "complex64" and arrays["phase"]["nbytes"] == 1 * 1 * 1001 * 8
    ppc = next(stage for stage in report.stages if stage["stage"] == "spike-ppc")
    record = results.load_product(report.products[0]).record
    assert ppc["counts"]["selected_spike_count"] == sum(len(times) for times in record.spike_times_by_trial) == 4
    assert ppc["counts"]["valid_spike_count_by_frequency"] == record.spike_count.sum(axis=0).tolist()
    assert ppc["counts"]["excluded_spike_count_by_frequency"] == record.excluded_spike_count.sum(axis=0).tolist()


def test_run_summary_contains_actual_total_and_first_requested_product_times(runnable_session, config):
    """Human-readable Markdown reports measured latency values, beyond a link to JSON evidence."""
    report = runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()))
    text = (report.run_directory / "summary.md").read_text()
    for label, key in (("Total elapsed", "total_s"), ("First requested product", "first_product_s")):
        match = re.search(label + r":\s*([0-9]+(?:\.[0-9]+)?) s", text)
        assert match is not None
        assert float(match.group(1)) == pytest.approx(report.timings_s[key], abs=0.0005)
