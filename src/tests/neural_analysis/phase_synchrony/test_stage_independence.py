"""WP4 independent identities, deterministic discovery and single-session writer lock."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest

from src.neural_analysis.lfp.config import FrequencyBandConfig
from src.neural_analysis.phase_synchrony import preparation, results
from src.tests.neural_analysis.phase_synchrony.test_results import (
    scientific_records, identity_for, rewrite_archive_meta,
)


REAL_NPZ_GETITEM = np.lib.npyio.NpzFile.__getitem__


def product_path(root: Path, run: str, family: str, name: str = "site_primary.npz") -> Path:
    """Return a feature-local completed-product path under one synthetic run, no physical units."""
    return root / "analysis_runs" / f"session_timeline_phase_synchrony_{run}" / "components" / family / name


def test_product_identity_excludes_unrelated_selections_and_preserves_stage_specific_settings(scientific_records):
    """Native/phase/PSD preparations ignore bands and unit selection; summaries retain their own bands."""
    config, _, records = scientific_records
    selected = replace(config, components=("spike-ppc",), unit_ids=("OtherProbe:8",), lfp_site="AnotherSite", intervals=("secondary",))
    changed_bands = replace(selected, bands=(FrequencyBandConfig("theta", 6, 8),))
    for family in records:
        assert identity_for(family, selected, records) == identity_for(family, config, records)
    for family in ("native-lfp", "unit-spikes", "waveforms", "psd", "phase"):
        assert identity_for(family, changed_bands, records) == identity_for(family, config, records)
    for family in ("power", "coherence", "lfp-phase", "spike-ppc"):
        assert identity_for(family, changed_bands, records) != identity_for(family, config, records)
    new_phase = replace(config, phase_frequencies_hz=(6., 8., 10.))
    assert identity_for("native-lfp", new_phase, records) == identity_for("native-lfp", config, records)
    assert identity_for("psd", new_phase, records) == identity_for("psd", config, records)
    assert identity_for("phase", new_phase, records) != identity_for("phase", config, records)


@pytest.mark.parametrize("changed", ["source", "preparation", "selection", "interval", "welch", "threshold"])
def test_relevant_source_support_or_settings_changes_invalidate_identity(scientific_records, changed):
    """Only genuinely matching source/selection/window/preparation/numerical settings are compatible."""
    config, _, records = scientific_records
    family = "phase" if changed == "threshold" else "psd"
    options = {"source_ids": {"lfp": "source-a"}, "preparation_ids": ("actual-support-a",)}
    selection, interval = ("SiteA",), "primary"
    original = results.product_identity(family, selection, interval, config, **options)
    if changed == "source":
        options["source_ids"] = {"lfp": "source-b"}
    elif changed == "preparation":
        options["preparation_ids"] = ("actual-support-b",)
    elif changed == "selection":
        selection = ("SiteB",)
    elif changed == "interval":
        interval = "secondary"
    elif changed == "welch":
        config = replace(config, welch_segment_s=0.4)
    else:
        config = replace(config, numerical_amplitude_threshold=1e-12)
    assert results.product_identity(family, selection, interval, config, **options) != original


def test_discovery_orders_by_parsed_utc_then_run_id_then_relative_path_ignoring_mtime(tmp_path, scientific_records):
    """UTC completion instants take priority over run IDs, filename order and filesystem timestamps."""
    config, _, records = scientific_records
    identity = identity_for("power", config, records)
    candidates = [
        ("z-old", "z", "z.npz", "2030-01-02T03:04:04Z"),
        ("a-new", "a", "a.npz", "2030-01-02T03:04:05Z"),
        ("b-new", "b", "a.npz", "2030-01-02T03:04:05+00:00"),
        ("b-tie", "b", "z.npz", "2030-01-02T03:04:05Z"),
    ]
    paths = []
    for folder, run_id, filename, completed in candidates:
        path = product_path(tmp_path, folder, "power", filename)
        results.save_product(path, records["power"], identity, run_id=run_id, completed_utc=completed)
        paths.append(path)
    os.utime(paths[0], ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    found = results.find_product(tmp_path, identity)
    assert found.path == paths[-1]
    assert found.meta["run_id"] == "b"
    assert found.meta["completed_utc"] == "2030-01-02T03:04:05Z"


def test_newer_failed_or_unfinished_attempt_and_bad_archives_leave_older_power_available(tmp_path, scientific_records):
    """Independent completed power remains discoverable after later failure/interruption or corrupt publication."""
    config, _, records = scientific_records
    identity = identity_for("power", config, records)
    older = results.save_product(product_path(tmp_path, "older", "power"), records["power"], identity,
                                  run_id="old", completed_utc="2030-01-02T03:04:05Z")
    later_run = tmp_path / "analysis_runs" / "session_timeline_phase_synchrony_newer"
    later_run.mkdir()
    (later_run / "measurements.json").write_text(json.dumps({"stages": [
        {"family": "power", "identity": identity, "status": "failed", "error": "interrupted"},
        {"family": "phase", "status": "running"},
    ]}), encoding="ascii")
    temporary = product_path(tmp_path, "newer", "power", "pending.tmp.npz")
    temporary.parent.mkdir(parents=True)
    temporary.write_bytes(older.read_bytes())
    temporary_meta = deepcopy(results.load_product(older).meta)
    temporary_meta.update(completed_utc="2030-01-02T03:04:09Z", run_id="unfinished")
    rewrite_archive_meta(temporary, temporary_meta)
    corrupt = temporary.parent / "corrupt.npz"
    corrupt.write_bytes(b"interrupted synthetic npz")
    incompatible = results.save_product(product_path(tmp_path, "wrong", "power"), records["power"], identity,
                                         run_id="wrong", completed_utc="2030-01-02T03:04:07Z")
    metadata = deepcopy(results.load_product(incompatible).meta)
    metadata["schema_version"] = 999
    rewrite_archive_meta(incompatible, metadata)
    assert results.find_product(tmp_path, identity).path == older


def test_wrong_identity_and_legacy_phase_do_not_hide_independent_power(tmp_path, scientific_records):
    """Reject mismatched-source loading and legacy phase discovery while matching power remains reusable."""
    config, _, records = scientific_records
    power_identity = identity_for("power", config, records)
    power_path = results.save_product(product_path(tmp_path, "one", "power"), records["power"], power_identity,
                                       run_id="one", completed_utc="2030-01-02T03:04:05Z")
    wrong_identity = results.product_identity("power", ("SiteA",), "primary", config, source_ids={"lfp": "different"}, preparation_ids=())
    with pytest.raises(ValueError):
        results.load_product(power_path, expected_identity=wrong_identity)
    assert results.find_product(tmp_path, wrong_identity) is None
    phase_identity = identity_for("phase", config, records)
    phase_path = results.save_product(product_path(tmp_path, "one", "phase"), records["phase"], phase_identity,
                                       run_id="one", completed_utc="2030-01-02T03:04:05Z")
    metadata = deepcopy(results.load_product(phase_path).meta)
    metadata["record_meta"].pop("phase_alignment_semantics")
    rewrite_archive_meta(phase_path, metadata)
    assert results.find_product(tmp_path, phase_identity) is None
    assert results.find_product(tmp_path, power_identity).path == power_path


def test_metadata_only_lookup_reads_no_numeric_arrays_and_searches_only_feature_roots(tmp_path, scientific_records, monkeypatch):
    """Dry-run lookup may read NPZ meta but cannot decompress phase/native/PSD arrays or scan raw files."""
    config, _, records = scientific_records
    identity = identity_for("phase", config, records)
    expected = results.save_product(product_path(tmp_path, "one", "phase"), records["phase"], identity,
                                     run_id="one", completed_utc="2030-01-02T03:04:05Z")
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    raw_decoy = results.save_product(raw_folder / "decoy.npz", records["phase"], identity,
                                      run_id="later", completed_utc="2030-01-02T03:04:09Z")
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", _metadata_only_getitem)
    info = results.find_product_metadata(tmp_path, identity)
    assert info.path == expected and info.path != raw_decoy
    assert results.read_product_metadata(expected)["identity"] == identity


def _metadata_only_getitem(archive, key):
    """Permit dictionary meta only; fail if discovery requests any scientific named array."""
    assert key == "meta", f"Metadata lookup read scientific array {key}"
    return REAL_NPZ_GETITEM(archive, key)


def test_saved_native_preparation_is_reusable_for_later_phase_without_raw_loading(tmp_path, scientific_records, monkeypatch):
    """Reload calibrated continuous native data/clock from its preparation product before new CWT work."""
    config, choices, records = scientific_records
    native = records["native-lfp"]
    identity = identity_for("native-lfp", config, records)
    path = tmp_path / "processed" / "session_timeline_phase_synchrony" / "native.npz"
    results.save_product(path, native, identity, run_id="power-first", completed_utc="2030-01-02T03:04:05Z")
    monkeypatch.setattr(preparation, "load_site_lfp", _forbid_raw_loading)
    loaded = results.find_product(tmp_path, identity).record
    phase = preparation.prepare_site_phase(loaded, choices, config)
    np.testing.assert_array_equal(phase.valid, records["phase"].valid)
    np.testing.assert_allclose(phase.phase, records["phase"].phase, rtol=0, atol=0, equal_nan=True)


def _forbid_raw_loading(*args, **kwargs):
    """Reject raw-loader arguments; saved continuous preparation must supply the data."""
    raise AssertionError("Raw LFP loading is not needed for saved native preparation")


def test_writer_lock_excludes_second_writer_and_releases_on_failure(tmp_path):
    """One session owns a same-host PID lock until context exit, including ordinary exceptions."""
    with pytest.raises(OSError, match="synthetic failure"):
        with results.session_writer_lock(tmp_path) as lock_path:
            ownership = json.loads(lock_path.read_text())
            assert ownership["hostname"] == socket.gethostname() and ownership["pid"] == os.getpid()
            before = lock_path.read_bytes()
            with pytest.raises(RuntimeError, match="writer|lock|process"):
                with results.session_writer_lock(tmp_path):
                    pytest.fail("Second writer acquired an active session")
            assert lock_path.read_bytes() == before
            raise OSError("synthetic failure")
    assert not lock_path.exists()


def test_writer_lock_recovers_only_proven_dead_same_host_pid(tmp_path):
    """A completed child PID is actually dead; recover its same-host record but refuse foreign ownership."""
    with results.session_writer_lock(tmp_path) as lock_path:
        ownership = json.loads(lock_path.read_text())
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=10)
    ownership["pid"] = process.pid
    lock_path.write_text(json.dumps(ownership), encoding="ascii")
    with results.session_writer_lock(tmp_path):
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    ownership["hostname"] = "unverifiable-other-host"
    lock_path.write_text(json.dumps(ownership), encoding="ascii")
    before = lock_path.read_bytes()
    with pytest.raises(RuntimeError, match="host|lock|writer"):
        with results.session_writer_lock(tmp_path):
            pytest.fail("Foreign-host lock was removed without a verifiable PID")
    assert lock_path.read_bytes() == before
