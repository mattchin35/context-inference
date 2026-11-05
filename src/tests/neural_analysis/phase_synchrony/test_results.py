"""WP4 immutable named-array persistence and scientific preparation identities."""

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.lfp import phase as existing_phase
from src.neural_analysis.phase_synchrony import computation, preparation, results
from src.neural_analysis.phase_synchrony.results import PhaseSynchronyConfig, SelectedUnitSpikes
from src.tests.neural_analysis.phase_synchrony.test_phase_metrics import UTC_ORIGIN, make_native_source


@pytest.fixture
def scientific_records():
    """Return nine typed scientific records with calibrated units, UTC clocks and ragged spikes."""
    config = PhaseSynchronyConfig(notch_enabled=False, phase_frequencies_hz=(8.0,))
    native = make_native_source(duration_s=8)
    choices = pd.DataFrame({
        "trial_id": [21, 28], "row_position": [2, 5], "choice_time": UTC_ORIGIN + np.array([2.125, 5.125]),
    })
    waveforms = preparation.prepare_site_lfp(native, choices)
    psd = computation.compute_site_psd(waveforms, config)
    power = computation.compute_band_power(psd, config)
    coherence = computation.compute_welch_coherence(waveforms, replace(waveforms, site_id="SiteB"), config)
    phase = preparation.prepare_site_phase(native, choices, config)
    lfp = computation.compute_lfp_phase_summary(phase, replace(phase, site_id="SiteB"), config)
    unit = SelectedUnitSpikes("ProbeA:17", "ProbeA", 17, UTC_ORIGIN + np.array([1.923456, 2.323456, 5.234567]), {})
    spike = computation.compute_selected_unit_ppc(unit, phase, config)
    return config, choices, {
        "native-lfp": native, "unit-spikes": unit, "waveforms": waveforms,
        "psd": psd, "power": power, "coherence": coherence, "phase": phase,
        "lfp-phase": lfp, "spike-ppc": spike,
    }


def identity_for(family: str, config: PhaseSynchronyConfig, records: dict) -> dict:
    """Return scientific product identity from actual prepared-record IDs, no display/run state."""
    if family == "unit-spikes":
        selection = ("ProbeA:17",)
        interval = None
        reference = results.preparation_identity(records["unit-spikes"])
    elif family == "native-lfp":
        selection = ("SiteA",)
        interval = None
        reference = results.preparation_identity(records["native-lfp"])
    else:
        selection = ("SiteA", "SiteB") if family in {"coherence", "lfp-phase"} else ("SiteA",)
        if family == "spike-ppc":
            selection = ("ProbeA:17", "SiteA")
        interval = "primary"
        reference = results.preparation_identity(records["phase"] if family in {"phase", "lfp-phase", "spike-ppc"} else records["waveforms"])
    return results.product_identity(
        family, selection, interval, config,
        source_ids={"lfp": "synthetic-source-v1"}, preparation_ids=(reference,),
    )


def assert_record_equal(actual, expected) -> None:
    """Compare typed record fields, including array shapes/dtypes/units and original UTC metadata."""
    assert type(actual) is type(expected)
    for field in fields(expected):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(right, np.ndarray):
            assert left.dtype == right.dtype
            np.testing.assert_array_equal(left, right)
        elif isinstance(right, pd.DataFrame):
            pd.testing.assert_frame_equal(left[["sample_ix", "utc_unix"]], right[["sample_ix", "utc_unix"]])
        elif field.name == "spike_times_by_trial":
            assert len(left) == len(right)
            for actual_times, expected_times in zip(left, right):
                np.testing.assert_array_equal(actual_times, expected_times)
        else:
            assert left == right


@pytest.mark.parametrize("family", [
    "native-lfp", "unit-spikes", "waveforms", "psd", "power", "coherence", "phase", "lfp-phase", "spike-ppc",
])
def test_typed_named_array_roundtrip_retains_all_scientific_fields(tmp_path, scientific_records, family):
    """Nine record types roundtrip through named NPZ arrays and dictionary meta, with no opaque record pickle."""
    config, _, records = scientific_records
    record = records[family]
    identity = identity_for(family, config, records)
    before_meta = deepcopy(record.meta)
    path = results.save_product(tmp_path / f"{family}.npz", record, identity, run_id="run-a", completed_utc="2030-01-02T03:04:05Z")
    product = results.load_product(path, expected_identity=identity)
    assert_record_equal(product.record, record)
    assert record.meta == before_meta
    assert product.path == path
    assert product.meta["identity"] == identity
    assert product.meta["run_id"] == "run-a"
    assert product.meta["completed_utc"] == "2030-01-02T03:04:05Z"
    assert product.meta["generator"] == "src.neural_analysis.phase_synchrony.results.save_product"
    with np.load(path, allow_pickle=True) as archive:
        assert isinstance(archive["meta"].item(), dict)
        for field in fields(record):
            if isinstance(getattr(record, field.name), np.ndarray):
                np.testing.assert_array_equal(archive[field.name], getattr(record, field.name))
        assert "record" not in archive.files and "payload" not in archive.files


def test_native_clock_and_ragged_inspection_spikes_have_explicit_named_arrays(tmp_path, scientific_records):
    """Native anchor samples/UTC and ragged (trial,spike) timestamps use numeric arrays, retaining precision."""
    config, _, records = scientific_records
    for family in ("native-lfp", "spike-ppc"):
        record = records[family]
        path = results.save_product(tmp_path / f"{family}.npz", record, identity_for(family, config, records),
                                    run_id="run-a", completed_utc="2030-01-02T03:04:05Z")
        with np.load(path, allow_pickle=True) as archive:
            if family == "native-lfp":
                np.testing.assert_array_equal(archive["irig_sample_ix"], record.irig_df.sample_ix)
                np.testing.assert_array_equal(archive["irig_utc_unix"], record.irig_df.utc_unix)
            else:
                flat = archive["spike_times_flat_s"]
                offsets = archive["spike_times_offsets"]
                assert flat.dtype == np.float64 and offsets.dtype.kind in "iu"
                np.testing.assert_array_equal(offsets, [0, 2, 3])
                np.testing.assert_array_equal(flat, np.concatenate(record.spike_times_by_trial))


def test_artifact_stamps_do_not_change_shared_psd_identity_or_reuse(tmp_path, scientific_records, monkeypatch):
    """Saving run/completion metadata preserves the original scientific PSD identity and permits later reuse."""
    config, _, records = scientific_records
    psd = records["psd"]
    before = deepcopy(psd.meta)
    path = results.save_product(tmp_path / "psd.npz", psd, identity_for("psd", config, records),
                                run_id="different-run", completed_utc="2030-01-02T03:04:05Z")
    loaded = results.load_product(path).record
    assert loaded.identity == psd.identity and loaded.meta == before and psd.meta == before
    monkeypatch.setattr(computation, "compute_site_psd", _forbid_numerical_loading)
    coherence = computation.compute_welch_coherence(
        records["waveforms"], records["waveforms"], config, psd_a=loaded, psd_b=loaded,
    )
    assert coherence.band_valid[:, 0].all()


def test_durable_inspection_arrays_load_without_raw_sources_or_transform(tmp_path, scientific_records, monkeypatch):
    """Saved phase/waveforms/spikes/polar histograms reload independently of raw LFP/spike APIs and CWT."""
    config, _, records = scientific_records
    paths = []
    for family in ("waveforms", "phase", "spike-ppc"):
        paths.append(results.save_product(tmp_path / f"{family}.npz", records[family], identity_for(family, config, records),
                                          run_id="run-a", completed_utc="2030-01-02T03:04:05Z"))
    monkeypatch.setattr(preparation, "load_site_lfp", _forbid_numerical_loading)
    monkeypatch.setattr(preparation.spike_loading, "load_aligned_spikes", _forbid_numerical_loading)
    monkeypatch.setattr(existing_phase, "compute_wavelet_coefficients", _forbid_numerical_loading)
    for family, path in zip(("waveforms", "phase", "spike-ppc"), paths):
        assert_record_equal(results.load_product(path).record, records[family])


def _forbid_numerical_loading(*args, **kwargs):
    """Reject any scientific reload/recomputation arguments; no arrays are returned."""
    raise AssertionError("Saved product loading must not reopen numerical sources or recompute")


def rewrite_archive_meta(path: Path, artifact_meta: dict) -> None:
    """Replace only a test-owned NPZ's dictionary meta to simulate legacy/corrupt provenance."""
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["meta"] = np.asarray(artifact_meta, dtype=object)
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("family", ["phase", "lfp-phase", "spike-ppc"])
@pytest.mark.parametrize("changed", ["support", "alignment", "numerical_threshold", "absolute_threshold"])
def test_legacy_or_different_phase_provenance_is_rejected_even_with_matching_identity_and_valid_masks(
    tmp_path, scientific_records, family, changed,
):
    """Legacy strict/all-valid phase and derived products cannot bypass current alignment/amplitude semantics."""
    config, _, records = scientific_records
    identity = identity_for(family, config, records)
    path = results.save_product(tmp_path / f"{family}.npz", records[family], identity,
                                run_id="run-a", completed_utc="2030-01-02T03:04:05Z")
    metadata = deepcopy(results.load_product(path).meta)
    phase_meta = metadata["record_meta"]
    if family == "lfp-phase":
        phase_meta = phase_meta["source_phase"][0]
    elif family == "spike-ppc":
        phase_meta = phase_meta["source_phase"]
    if changed == "support":
        phase_meta.pop("phase_support_semantics")
    elif changed == "alignment":
        phase_meta["phase_alignment_semantics"] = "nominal_block_only"
    elif changed == "numerical_threshold":
        phase_meta["numerical_amplitude_threshold"] = 1e-12
    else:
        phase_meta["absolute_amplitude_threshold"] = 1.0
    rewrite_archive_meta(path, metadata)
    with pytest.raises(ValueError):
        results.load_product(path, expected_identity=identity)


def test_preparation_identity_tracks_actual_phase_support_thresholds_and_precise_block_grid(scientific_records):
    """Real masks/guards, phase settings and small block origins define preparation compatibility."""
    _, _, records = scientific_records
    phase = records["phase"]
    original = results.preparation_identity(phase)
    mask = phase.valid.copy()
    mask[0, 0, -1] = False
    assert results.preparation_identity(replace(phase, valid=mask)) != original
    metadata = deepcopy(phase.meta)
    metadata["blocks"][0]["origin_relative_s"] += 0.0001
    assert results.preparation_identity(replace(phase, meta=metadata)) != original
    metadata = dict(phase.meta, numerical_amplitude_threshold=1e-12)
    assert results.preparation_identity(replace(phase, meta=metadata)) != original
    native = records["native-lfp"]
    anchors = native.irig_df.copy()
    anchors.loc[1, "utc_unix"] += 0.001
    assert results.preparation_identity(replace(native, irig_df=anchors)) != results.preparation_identity(native)


def test_products_are_immutable_and_failed_write_leaves_no_completed_npz(tmp_path, scientific_records, monkeypatch):
    """An existing complete product is never overwritten; an I/O interruption cannot publish a partial product."""
    config, _, records = scientific_records
    identity = identity_for("power", config, records)
    path = results.save_product(tmp_path / "power.npz", records["power"], identity,
                                run_id="run-a", completed_utc="2030-01-02T03:04:05Z")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        results.save_product(path, records["power"], identity, run_id="run-b", completed_utc="2030-01-02T03:04:06Z")
    assert path.read_bytes() == original
    monkeypatch.setattr(np, "savez_compressed", _interrupted_write)
    interrupted = tmp_path / "next" / "power.npz"
    with pytest.raises(OSError, match="interrupted"):
        results.save_product(interrupted, records["power"], identity, run_id="run-b", completed_utc="2030-01-02T03:04:06Z")
    assert not interrupted.exists()
    assert not list(interrupted.parent.glob("*.npz"))


def _interrupted_write(*args, **kwargs):
    """Simulate an ordinary NPZ I/O interruption before publication; always raise OSError."""
    raise OSError("interrupted synthetic write")


@pytest.mark.parametrize("completed_utc", ["2030-01-02T03:04:05", "not-a-time"])
def test_completion_time_requires_parseable_timezone_aware_timestamp(tmp_path, scientific_records, completed_utc):
    """A completed product needs an unambiguous UTC instant; naive/malformed timestamps cannot publish."""
    config, _, records = scientific_records
    with pytest.raises(ValueError):
        results.save_product(tmp_path / "power.npz", records["power"], identity_for("power", config, records),
                             run_id="run-a", completed_utc=completed_utc)


@pytest.mark.parametrize("changed", [
    "site", "unit", "interval", "phase_frequency", "welch", "bands", "notch", "phase_guard",
])
def test_publication_identity_must_match_actual_scientific_record(tmp_path, scientific_records, changed):
    """A caller cannot publish correct typed arrays under a different selection or analysis.

    Inputs are the nine synthetic records with their documented axes/units;
    reject mismatched IDs, seconds, Hz/settings or an included phase guard.
    """
    config, _, records = scientific_records
    family = {"unit": "unit-spikes", "phase_frequency": "phase", "phase_guard": "phase",
              "welch": "psd", "notch": "native-lfp"}.get(changed, "power")
    record = records[family]
    identity = identity_for(family, config, records)
    selection, interval = identity["selection"], identity["interval"]
    if changed == "site":
        selection = ("DifferentSite",)
    elif changed == "unit":
        selection = ("ProbeB:17",)
    elif changed == "interval":
        interval = "secondary"
    elif changed == "phase_frequency":
        config = replace(config, phase_frequencies_hz=(10.0,))
    elif changed == "welch":
        config = replace(config, welch_segment_s=0.4)
    elif changed == "bands":
        from src.neural_analysis.lfp.config import FrequencyBandConfig

        config = replace(config, bands=(FrequencyBandConfig("theta", 6.0, 8.0),))
    elif changed == "notch":
        config = replace(config, notch_enabled=True)
    else:
        record = replace(record, metric_mask=np.ones_like(record.metric_mask))
    identity = results.product_identity(
        family, selection, interval, config, source_ids=identity["source_ids"],
        preparation_ids=identity["preparation_ids"],
    )
    path = tmp_path / "mismatched.npz"
    with pytest.raises(ValueError):
        results.save_product(path, record, identity, run_id="bad", completed_utc="2030-01-02T03:04:05Z")
    assert not path.exists()


def test_loading_rejects_relabelled_phase_frequency_identity(tmp_path, scientific_records):
    """A structurally complete archive cannot claim a different frequency under a valid identity digest."""
    config, _, records = scientific_records
    identity = identity_for("phase", config, records)
    path = results.save_product(tmp_path / "phase.npz", records["phase"], identity,
                                run_id="original", completed_utc="2030-01-02T03:04:05Z")
    metadata = deepcopy(results.load_product(path).meta)
    changed = identity_for("phase", replace(config, phase_frequencies_hz=(10.0,)), records)
    metadata["identity"] = changed
    rewrite_archive_meta(path, metadata)
    with pytest.raises(ValueError):
        results.load_product(path, expected_identity=changed)
