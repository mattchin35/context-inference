"""Contract tests for LFP-summary NPZ cache validation and transactions."""

from __future__ import annotations

import json
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import zipfile

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_io
from src.neural_analysis.lfp_summary_io import (
    assess_component_status,
    load_or_initialize_manifest,
    load_component_arrays,
    write_component_transaction,
)
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    UnitPopulationConfig,
    canonical_config_json,
    component_fingerprint,
    default_lfp_summary_config,
    fingerprint_source_files,
    lfp_summary_config_from_json,
    source_value_semantics,
)
from src.neural_analysis.lfp_summary_payloads import SYNCHRONY_ARRAY_SCHEMA


def _schema_entry(axes: tuple[str, ...], units: str) -> dict[str, object]:
    """Return one static JSON-ready legacy array contract entry."""
    return {"axes": list(axes), "units": units}


# Freeze the complete pre-NR1P Synchrony receipt independently of the live
# schema. This fixture is deliberately not generated from SYNCHRONY_ARRAY_SCHEMA.
_PRE_NR1P_SYNCHRONY_ARRAY_SCHEMA = {
    "trial_indices": _schema_entry(("trial",), "trial-table-row"),
    "site_ids": _schema_entry(("site",), "stable-site-id"),
    "site_voltage_units": _schema_entry(("site",), "source-voltage-unit"),
    "condition_names": _schema_entry(("condition",), "condition-name"),
    "condition_membership": _schema_entry(("trial", "condition"), "boolean"),
    "filter_membership": _schema_entry(("trial",), "boolean"),
    "frequency_hz": _schema_entry(("frequency",), "Hz"),
    "epoch_names": _schema_entry(("epoch",), "epoch-name"),
    "band_names": _schema_entry(("band",), "band-name"),
    "relative_time_s": _schema_entry(("time",), "s"),
    "site_valid": _schema_entry(("site", "trial"), "boolean"),
    "pair_valid": _schema_entry(("pair", "trial"), "boolean"),
    "site_exclusion_count": _schema_entry(("site",), "trial"),
    "pair_exclusion_count": _schema_entry(("pair",), "trial"),
    "pair_site_a_ids": _schema_entry(("pair",), "stable-site-id"),
    "pair_site_b_ids": _schema_entry(("pair",), "stable-site-id"),
    "itpc": _schema_entry(
        ("condition", "site", "frequency", "time"), "dimensionless"
    ),
    "itpc_effective_trial_count": _schema_entry(
        ("condition", "site", "frequency", "time"), "trial"
    ),
    "ispc": _schema_entry(
        ("condition", "pair", "frequency", "time"), "dimensionless"
    ),
    "ispc_phase_offset_rad": _schema_entry(
        ("condition", "pair", "frequency", "time"), "rad"
    ),
    "ispc_effective_trial_count": _schema_entry(
        ("condition", "pair", "frequency", "time"), "trial"
    ),
    "itpc_band_mean": _schema_entry(
        ("condition", "site", "epoch", "band"), "dimensionless"
    ),
    "itpc_ci_low": _schema_entry(
        ("condition", "site", "epoch", "band"), "dimensionless"
    ),
    "itpc_ci_high": _schema_entry(
        ("condition", "site", "epoch", "band"), "dimensionless"
    ),
    "itpc_unstable": _schema_entry(
        ("condition", "site", "epoch", "band"), "boolean"
    ),
    "ispc_band_mean": _schema_entry(
        ("condition", "pair", "epoch", "band"), "dimensionless"
    ),
    "ispc_ci_low": _schema_entry(
        ("condition", "pair", "epoch", "band"), "dimensionless"
    ),
    "ispc_ci_high": _schema_entry(
        ("condition", "pair", "epoch", "band"), "dimensionless"
    ),
    "ispc_unstable": _schema_entry(
        ("condition", "pair", "epoch", "band"), "boolean"
    ),
    "plv_by_frequency": _schema_entry(
        ("trial", "pair", "epoch", "frequency"), "dimensionless"
    ),
    "plv_phase_offset_rad": _schema_entry(
        ("trial", "pair", "epoch", "frequency"), "rad"
    ),
    "plv_valid_sample_count": _schema_entry(
        ("trial", "pair", "epoch", "frequency"), "sample"
    ),
    "plv_valid_sample_fraction": _schema_entry(
        ("trial", "pair", "epoch", "frequency"), "fraction"
    ),
    "plv_computable": _schema_entry(
        ("trial", "pair", "epoch", "frequency"), "boolean"
    ),
    "plv_band_mean": _schema_entry(
        ("trial", "pair", "epoch", "band"), "dimensionless"
    ),
    "source_trace": _schema_entry(
        ("site", "trial", "time"), "source-voltage-unit"
    ),
    "band_filtered_trace": _schema_entry(
        ("site", "trial", "band", "time"), "source-voltage-unit"
    ),
    "hilbert_phase_rad": _schema_entry(("site", "trial", "band", "time"), "rad"),
}
_LEGACY_STRING_ARRAYS = {
    "site_ids",
    "site_voltage_units",
    "condition_names",
    "epoch_names",
    "band_names",
    "pair_site_a_ids",
    "pair_site_b_ids",
}
_LEGACY_BOOLEAN_UNITS = {"boolean"}
_LEGACY_INTEGER_UNITS = {"trial", "sample", "trial-table-row"}


def _pre_nr1p_synchrony_arrays() -> dict[str, np.ndarray]:
    """Return a semantically typed, complete pre-NR1P Synchrony receipt."""
    axis_lengths = {
        "trial": 2,
        "site": 2,
        "pair": 1,
        "condition": 1,
        "frequency": 1,
        "epoch": 1,
        "band": 1,
        "time": 2,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, contract in _PRE_NR1P_SYNCHRONY_ARRAY_SCHEMA.items():
        shape = tuple(axis_lengths[axis] for axis in contract["axes"])
        if name in _LEGACY_STRING_ARRAYS:
            arrays[name] = np.full(shape, name, dtype="<U32")
        elif contract["units"] in _LEGACY_BOOLEAN_UNITS:
            arrays[name] = np.ones(shape, dtype=bool)
        elif contract["units"] in _LEGACY_INTEGER_UNITS:
            arrays[name] = np.ones(shape, dtype=np.int64)
        else:
            arrays[name] = np.ones(shape, dtype=np.float64)
    return arrays


def _power_arrays() -> dict[str, np.ndarray]:
    """Return minimal valid numeric arrays for a power component cache.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays with axes ``trial``, ``site``, ``epoch``, and ``frequency`` as
        named in the companion manifest helper.
    """
    return {
        "trial_indices": np.array([0, 2], dtype=np.int64),
        "site_ids": np.array(["PFC"], dtype="<U3"),
        "frequency_hz": np.array([2.0, 4.0], dtype=np.float64),
        "psd_linear": np.ones((1, 2, 1, 2), dtype=np.float64),
        "psd_valid": np.ones((1, 2, 1), dtype=bool),
    }


def _power_array_schema() -> dict[str, dict[str, object]]:
    """Return required array shapes and axes for ``_power_arrays``.

    Returns
    -------
    dict[str, dict[str, object]]
        JSON-serializable array contracts whose axes are ordered tuple labels.
    """
    return {
        "trial_indices": {"axes": ["trial"], "units": "trial-table-row"},
        "site_ids": {"axes": ["site"], "units": "stable-site-id"},
        "frequency_hz": {"axes": ["frequency"], "units": "Hz"},
        "psd_linear": {
            "axes": ["site", "trial", "epoch", "frequency"],
            "units": "source-voltage-unit^2/Hz",
        },
        "psd_valid": {"axes": ["site", "trial", "epoch"], "units": "boolean"},
    }


def _manifest(
    config_fingerprint: str,
    *,
    config: LFPSummaryConfig | None = None,
    component: str = "power",
    state: str = "complete",
    file_name: str | None = None,
) -> dict[str, object]:
    """Return minimal manifest metadata for one power-component cache.

    Parameters
    ----------
    config_fingerprint : str
        SHA-256 fingerprint of the configuration subset used by power.
    state : str
        Completion status recorded for this component attempt.

    Returns
    -------
    dict[str, object]
        JSON-serializable manifest with axes, units, component identity, state,
        source fingerprints, and a canonical configuration snapshot.
    """
    config = default_lfp_summary_config() if config is None else config
    file_name = f"{component}.npz" if file_name is None else file_name
    return {
        "schema_version": "1",
        "session_id": config.session_id,
        "components": {
            component: {
                "file_name": file_name,
                "configuration_fingerprint": config_fingerprint,
                "configuration_snapshot": json.loads(canonical_config_json(config)),
                "source_fingerprints": fingerprint_source_files(config, component=component),
                "array_schema": _power_array_schema(),
                "state": state,
            }
        },
    }


def test_load_or_initialize_manifest_creates_write_ready_metadata_when_absent(
    tmp_path: Path,
) -> None:
    """An absent manifest yields all required top-level metadata and empty components."""
    config = default_lfp_summary_config()

    manifest = load_or_initialize_manifest(tmp_path, config)

    assert manifest["schema_version"] == config.schema_version
    assert manifest["session_id"] == config.session_id
    assert manifest["components"] == {}
    for field in ("generator", "configuration", "reference", "processing_metadata"):
        assert field in manifest

    entry = {
        "file_name": "power.npz",
        "array_schema": _power_array_schema(),
        "state": "complete",
    }
    write_ready = {**manifest, "components": {"power": entry}}
    write_component_transaction(tmp_path, "power", _power_arrays(), write_ready)
    assert (tmp_path / "manifest.json").is_file()


def test_load_or_initialize_manifest_loads_valid_json_and_rejects_corruption_or_nonmapping(
    tmp_path: Path,
) -> None:
    """Only a JSON object manifest can seed an atomic component transaction."""
    config = default_lfp_summary_config()
    manifest_path = tmp_path / "manifest.json"
    valid_manifest = {
        "schema_version": config.schema_version,
        "session_id": config.session_id,
        "components": {},
        "generator": {"name": "test"},
        "configuration": {},
        "reference": {},
        "processing_metadata": {},
    }
    manifest_path.write_text(json.dumps(valid_manifest), encoding="ascii")
    assert load_or_initialize_manifest(tmp_path, config) == valid_manifest

    manifest_path.write_text("{not valid json", encoding="ascii")
    with pytest.raises(ValueError, match="manifest|JSON|json"):
        load_or_initialize_manifest(tmp_path, config)

    manifest_path.write_text("[]", encoding="ascii")
    with pytest.raises(ValueError, match="mapping|object|manifest"):
        load_or_initialize_manifest(tmp_path, config)


def test_numeric_boolean_and_fixed_unicode_arrays_load_without_pickle(tmp_path: Path) -> None:
    """The loader accepts only safe NPZ dtypes and always uses ``allow_pickle=False``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed directory for a synthetic cache component.
    """
    component_path = tmp_path / "power.npz"
    np.savez(component_path, **_power_arrays())
    manifest = _manifest(component_fingerprint("power", default_lfp_summary_config()))

    loaded = load_component_arrays(component_path, manifest, "power")

    assert loaded["site_ids"].dtype.kind == "U"
    assert loaded["psd_valid"].dtype == np.dtype(bool)
    with pytest.raises(ValueError, match="object|pickle"):
        np.savez(component_path, **_power_arrays(), unsupported=np.array([{"not": "safe"}], dtype=object))
        load_component_arrays(component_path, manifest, "power")


@pytest.mark.parametrize(
    ("arrays", "expected_message"),
    [
        (
            {key: value for key, value in _power_arrays().items() if key != "psd_valid"},
            "psd_valid",
        ),
        (
            {**_power_arrays(), "psd_linear": np.ones((2, 1, 1, 2), dtype=np.float64)},
            "psd_linear",
        ),
    ],
)
def test_missing_or_axis_mismatched_arrays_are_rejected(
    tmp_path: Path, arrays: dict[str, np.ndarray], expected_message: str
) -> None:
    """Reject components missing a required array or violating its manifest axes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed directory for the invalid component.
    arrays : dict[str, numpy.ndarray]
        Component arrays that intentionally violate one manifest contract.
    expected_message : str
        Required array name expected in the validation diagnostic.
    """
    component_path = tmp_path / "power.npz"
    np.savez(component_path, **arrays)

    with pytest.raises(ValueError, match=expected_message):
        load_component_arrays(
            component_path,
            _manifest(component_fingerprint("power", default_lfp_summary_config())),
            "power",
        )


def test_write_failure_preserves_previous_component_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write failure does not replace the prior component or manifest commit point.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Cache directory containing a valid committed power result.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to inject a component-write failure.
    """
    config = default_lfp_summary_config()
    fingerprint = component_fingerprint("power", config)
    old_arrays = _power_arrays()
    old_manifest = _manifest(fingerprint)
    component_path = tmp_path / "power.npz"
    manifest_path = tmp_path / "manifest.json"
    np.savez(component_path, **old_arrays)
    manifest_path.write_text(json.dumps(old_manifest), encoding="ascii")

    def fail_component_replace(source: Path, destination: Path) -> None:
        """Raise while replacing the component file; paths are transaction temporaries."""
        del source, destination
        raise OSError("injected component replacement failure")

    monkeypatch.setattr("src.neural_analysis.lfp_summary_io._replace_component_file", fail_component_replace)
    with pytest.raises(OSError, match="injected"):
        write_component_transaction(tmp_path, "power", {**old_arrays, "psd_linear": np.full((1, 2, 1, 2), 7.0)}, old_manifest)

    assert np.array_equal(np.load(component_path, allow_pickle=False)["psd_linear"], old_arrays["psd_linear"])
    assert json.loads(manifest_path.read_text(encoding="ascii")) == old_manifest


def test_validation_failure_preserves_previous_component_and_manifest(tmp_path: Path) -> None:
    """Temporary validation failure leaves both last successful files unchanged.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Cache directory containing a valid committed component and manifest.
    """
    fingerprint = component_fingerprint("power", default_lfp_summary_config())
    old_arrays = _power_arrays()
    old_manifest = _manifest(fingerprint)
    component_path = tmp_path / "power.npz"
    manifest_path = tmp_path / "manifest.json"
    np.savez(component_path, **old_arrays)
    manifest_path.write_text(json.dumps(old_manifest), encoding="ascii")

    invalid_arrays = {key: value for key, value in old_arrays.items() if key != "psd_valid"}
    with pytest.raises(ValueError, match="psd_valid"):
        write_component_transaction(tmp_path, "power", invalid_arrays, old_manifest)

    assert np.array_equal(np.load(component_path, allow_pickle=False)["trial_indices"], old_arrays["trial_indices"])
    assert json.loads(manifest_path.read_text(encoding="ascii")) == old_manifest


def test_manifest_replace_failure_restores_previous_component_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest commit failure restores the previous component and manifest together.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Cache directory containing one prior valid committed power component.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to make final manifest replacement fail after component validation.
    """
    fingerprint = component_fingerprint("power", default_lfp_summary_config())
    old_arrays = _power_arrays()
    old_manifest = _manifest(fingerprint)
    component_path = tmp_path / "power.npz"
    manifest_path = tmp_path / "manifest.json"
    np.savez(component_path, **old_arrays)
    manifest_path.write_text(json.dumps(old_manifest), encoding="ascii")

    def fail_manifest_replace(source: Path, destination: Path) -> None:
        """Raise while atomically committing the temporary manifest file."""
        del source, destination
        raise OSError("injected manifest replacement failure")

    monkeypatch.setattr("src.neural_analysis.lfp_summary_io._replace_manifest_file", fail_manifest_replace)
    new_arrays = {**old_arrays, "psd_linear": np.full((1, 2, 1, 2), 7.0)}
    with pytest.raises(OSError, match="injected manifest"):
        write_component_transaction(tmp_path, "power", new_arrays, old_manifest)

    assert np.array_equal(np.load(component_path, allow_pickle=False)["psd_linear"], old_arrays["psd_linear"])
    assert json.loads(manifest_path.read_text(encoding="ascii")) == old_manifest


def test_component_statuses_report_missing_compatible_stale_and_failed_diffs(tmp_path: Path) -> None:
    """Status inspection distinguishes cache states and reports readable stale differences.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Cache directory used for missing and present component scenarios.
    """
    config = default_lfp_summary_config()
    current_fingerprint = component_fingerprint("power", config)
    manifest = _manifest(current_fingerprint)

    missing = assess_component_status(tmp_path, "power", config, manifest)
    assert missing.status == "missing"

    np.savez(tmp_path / "power.npz", **_power_arrays())
    compatible = assess_component_status(tmp_path, "power", config, manifest)
    assert compatible.status == "compatible"
    assert compatible.differences == ()

    stale_manifest = _manifest("outdated-fingerprint")
    stale = assess_component_status(tmp_path, "power", config, stale_manifest)
    assert stale.status == "stale"
    assert stale.differences
    assert any("fingerprint" in difference.lower() for difference in stale.differences)

    failed_manifest = _manifest(current_fingerprint, state="failed")
    failed = assess_component_status(tmp_path, "power", config, failed_manifest)
    assert failed.status == "failed"
    assert any("failed" in difference.lower() for difference in failed.differences)


def test_running_component_is_never_reported_compatible(tmp_path: Path) -> None:
    """A running component remains unavailable to saved-result views."""
    config = default_lfp_summary_config()
    fingerprint = component_fingerprint("power", config)
    np.savez(tmp_path / "power.npz", **_power_arrays())

    running = assess_component_status(tmp_path, "power", config, _manifest(fingerprint, state="running"))
    assert running.status == "running"


def test_corrupt_component_is_never_reported_compatible(tmp_path: Path) -> None:
    """A present but schema-invalid component reports a failed inspection state."""
    config = default_lfp_summary_config()
    fingerprint = component_fingerprint("power", config)
    np.savez(tmp_path / "power.npz", **{"trial_indices": np.array([0], dtype=np.int64)})
    corrupt = assess_component_status(tmp_path, "power", config, _manifest(fingerprint))
    assert corrupt.status == "failed"
    assert any("invalid" in difference.lower() or "missing" in difference.lower() for difference in corrupt.differences)


def test_old_synchrony_payload_is_stale_live_while_power_stays_compatible_and_receipt_readable(
    tmp_path: Path,
) -> None:
    """A payload-version migration stales only live Synchrony, not a receipt schema."""
    config = default_lfp_summary_config()
    new_synchrony_fields = (
        "itpc_bootstrap_q25",
        "itpc_bootstrap_median",
        "itpc_bootstrap_q75",
        "itpc_band_trial_count",
        "ispc_bootstrap_q25",
        "ispc_bootstrap_median",
        "ispc_bootstrap_q75",
        "ispc_band_trial_count",
    )
    assert set(new_synchrony_fields).issubset(SYNCHRONY_ARRAY_SCHEMA)
    legacy_schema = _PRE_NR1P_SYNCHRONY_ARRAY_SCHEMA
    legacy_arrays = _pre_nr1p_synchrony_arrays()
    old_synchrony_fingerprint = (
        "28127adaf4318f33aa29aaa653b15ab950890fcb606eb4cd5fa5e4f75596ee43"
    )
    manifest = _manifest(component_fingerprint("power", config), config=config)
    manifest["components"]["synchrony"] = {
        "file_name": "synchrony.npz",
        "configuration_fingerprint": old_synchrony_fingerprint,
        "configuration_snapshot": json.loads(canonical_config_json(config)),
        "source_fingerprints": fingerprint_source_files(config, component="synchrony"),
        "array_schema": legacy_schema,
        "state": "complete",
    }
    np.savez(tmp_path / "power.npz", **_power_arrays())
    np.savez(tmp_path / "synchrony.npz", **legacy_arrays)

    synchrony_status = assess_component_status(tmp_path, "synchrony", config, manifest)
    power_status = assess_component_status(tmp_path, "power", config, manifest)
    receipt_arrays = load_component_arrays(tmp_path / "synchrony.npz", manifest, "synchrony")

    assert synchrony_status.status == "stale"
    assert any("fingerprint" in detail for detail in synchrony_status.differences)
    assert power_status.status == "compatible"
    assert receipt_arrays.keys() == legacy_arrays.keys()
    assert not any(name in receipt_arrays for name in new_synchrony_fields)
    assert receipt_arrays["site_ids"].dtype.kind == "U"
    assert receipt_arrays["site_valid"].dtype == np.dtype(bool)
    assert receipt_arrays["itpc_effective_trial_count"].dtype.kind in {"i", "u"}
    assert receipt_arrays["itpc"].dtype.kind == "f"


def _source_config_and_manifest(tmp_path: Path) -> tuple[LFPSummaryConfig, dict[str, object], Path]:
    """Create a committed component manifest with one mutable source recording path."""
    config = default_lfp_summary_config()
    lfp_path = tmp_path / "PFC.bin"
    sync_path = tmp_path / "PFC.sync.json"
    lfp_path.write_bytes(b"first recording")
    sync_path.write_text("{}", encoding="ascii")
    configured_site = replace(config.sites[0], lfp_path=lfp_path, aligned_sync_path=sync_path)
    source_config = replace(config, sites=(configured_site,) + config.sites[1:])
    manifest = _manifest(component_fingerprint("power", source_config), config=source_config)
    np.savez(tmp_path / "power.npz", **_power_arrays())

    assert assess_component_status(tmp_path, "power", source_config, manifest).status == "compatible"
    return source_config, manifest, lfp_path


def test_status_rejects_changed_session_dependency(tmp_path: Path) -> None:
    """Status rejects a cache manifest whose session snapshot differs from the active session."""
    source_config, manifest, _ = _source_config_and_manifest(tmp_path)

    changed_session = replace(source_config, session_id="different-session")
    session_status = assess_component_status(tmp_path, "power", changed_session, manifest)
    assert session_status.status == "stale"
    assert any("session" in difference.lower() for difference in session_status.differences)


def test_status_rejects_changed_filter_dependency(tmp_path: Path) -> None:
    """Status rejects a cache computed with a different choice/context selection."""
    source_config, manifest, _ = _source_config_and_manifest(tmp_path)

    changed_filter = replace(source_config, trial_filter=replace(source_config.trial_filter, context="left"))
    filter_status = assess_component_status(tmp_path, "power", changed_filter, manifest)
    assert filter_status.status == "stale"
    assert any("fingerprint" in difference.lower() or "filter" in difference.lower() for difference in filter_status.differences)


def test_status_rejects_changed_source_dependency(tmp_path: Path) -> None:
    """Status rejects a cache when the recorded LFP source fingerprint changes."""
    source_config, manifest, lfp_path = _source_config_and_manifest(tmp_path)

    lfp_path.write_bytes(b"changed recording source")
    source_status = assess_component_status(tmp_path, "power", source_config, manifest)
    assert source_status.status == "stale"
    assert any("source" in difference.lower() for difference in source_status.differences)


def test_legacy_open_ephys_component_is_stale_live_but_its_saved_snapshot_remains_readable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Corrected live identity never relabels a receipt-validated legacy cache as corrupt."""
    base_config = default_lfp_summary_config()
    open_ephys_site = replace(
        base_config.sites[0],
        acquisition_format="open_ephys",
        lfp_path=Path("pre_nr0_pfc_lfp.dat"),
        aligned_sync_path=Path("pre_nr0_pfc_sync.npz"),
        voltage_unit="uV",
        sample_rate_hz=2_500.0,
    )
    live_config = replace(base_config, sites=(open_ephys_site,) + base_config.sites[1:])
    legacy_digest = "97d72e6bce66f44c89b5ddd11ce257fde21c5c68ac190adb92873750676f5a61"
    corrected_digest = "0e9b29b41f7ece65f6432fef3ad81acf7d5bc6dacd1fb4daa3528225daf7e1bb"
    legacy_manifest = _manifest(legacy_digest, config=live_config)
    # Freeze the complete observed pre-NR0 record.  Do not obtain it through
    # the live fingerprint helper, because that helper must gain the sidecar
    # and semantics provenance being tested here.
    lfp_path = Path(open_ephys_site.lfp_path).resolve()
    sync_path = Path(open_ephys_site.aligned_sync_path).resolve()
    legacy_source_fingerprints = {
        str(lfp_path): {
            "path": str(lfp_path),
            "size_bytes": -1,
            "mtime_ns": -1,
        },
        str(sync_path): {
            "path": str(sync_path),
            "size_bytes": -1,
            "mtime_ns": -1,
        },
    }
    # A pre-NR0 receipt has no sidecar or semantics mapping. That absence is
    # legacy only during receipt-validated inspection and never a live identity.
    legacy_entry = legacy_manifest["components"]["power"]
    legacy_entry["source_fingerprints"] = legacy_source_fingerprints
    legacy_entry.pop("source_value_semantics", None)
    np.savez(tmp_path / "power.npz", **_power_arrays())

    # This generic I/O test fixes only the historical source records. The live
    # component hash remains the real code-owned value for the exact fixed
    # configuration used by the model digest test.
    monkeypatch.setattr(
        lfp_summary_io,
        "fingerprint_source_files",
        lambda *_args, **_kwargs: legacy_source_fingerprints,
    )
    corrected_status = assess_component_status(
        tmp_path, "power", live_config, legacy_manifest
    )
    saved_snapshot = lfp_summary_config_from_json(
        json.dumps(legacy_entry["configuration_snapshot"])
    )

    assert corrected_status.status == "stale"
    assert component_fingerprint("power", live_config) == corrected_digest
    assert corrected_status.differences == (
        f"configuration fingerprint differs: cached={legacy_digest}, current={corrected_digest}",
    )
    assert saved_snapshot == live_config
    assert "source_value_semantics" not in legacy_entry
    assert legacy_entry["source_fingerprints"] == legacy_source_fingerprints
    assert all("lfp_preprocessing.json" not in path for path in legacy_source_fingerprints)
    assert all(
        "value_semantics" not in record
        for record in legacy_source_fingerprints.values()
    )


def test_unit_source_change_stales_spike_phase_only(tmp_path: Path) -> None:
    """Sorter and aligned-spike changes do not invalidate Power or Synchrony caches."""
    config = default_lfp_summary_config()
    sorter_path = tmp_path / "spike_clusters.npy"
    aligned_spike_path = tmp_path / "aligned_spikes.npy"
    trial_table_path = tmp_path / "trials.csv"
    sorter_path.write_bytes(b"sorter-v1")
    aligned_spike_path.write_bytes(b"aligned-v1")
    trial_table_path.write_text("choice_time\n1.0\n", encoding="ascii")
    population = UnitPopulationConfig("units", "PFC", sorter_path, aligned_spike_path, (5,), (), ("PFC:1",))
    config = replace(config, unit_population=population, trial_table_path=trial_table_path)
    manifests = {
        component: _manifest(component_fingerprint(component, config), config=config, component=component)
        for component in ("power", "synchrony", "spike_phase")
    }
    for component in manifests:
        np.savez(tmp_path / f"{component}.npz", **_power_arrays())

    sorter_path.write_bytes(b"sorter-v2")

    assert assess_component_status(tmp_path, "power", config, manifests["power"]).status == "compatible"
    assert assess_component_status(tmp_path, "synchrony", config, manifests["synchrony"]).status == "compatible"
    assert assess_component_status(tmp_path, "spike_phase", config, manifests["spike_phase"]).status == "stale"


def test_trial_table_source_change_stales_every_component(tmp_path: Path) -> None:
    """Behavior/trial-table changes invalidate each component that uses trial membership."""
    config = default_lfp_summary_config()
    trial_table_path = tmp_path / "trials.csv"
    trial_table_path.write_text("choice_time\n1.0\n", encoding="ascii")
    config = replace(config, trial_table_path=trial_table_path)
    manifests = {
        component: _manifest(component_fingerprint(component, config), config=config, component=component)
        for component in ("power", "synchrony", "spike_phase")
    }
    for component in manifests:
        np.savez(tmp_path / f"{component}.npz", **_power_arrays())

    trial_table_path.write_text("choice_time\n2.0\n", encoding="ascii")

    for component, manifest in manifests.items():
        assert assess_component_status(tmp_path, component, config, manifest).status == "stale"


def test_transaction_rejects_component_filename_outside_cache_directory(tmp_path: Path) -> None:
    """Manifest component filenames cannot escape the selected cache directory."""
    config = default_lfp_summary_config()
    escaped_path = tmp_path.parent / "escaped-power.npz"
    manifest = _manifest(
        component_fingerprint("power", config),
        config=config,
        file_name="../escaped-power.npz",
    )

    try:
        with pytest.raises(ValueError, match="file.*name|path|cache"):
            write_component_transaction(tmp_path, "power", _power_arrays(), manifest)
    finally:
        escaped_path.unlink(missing_ok=True)


def test_rebind_power_synchrony_manifest_replaces_only_destination_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure rebind makes copied Power/Synchrony bytes compatible at a new root.

    The helper receives JSON manifest metadata plus a validated destination
    configuration.  It neither opens component arrays nor reads source files.
    It must preserve the original producer/completion/scientific metadata while
    replacing only the top-level configuration and each copied component's
    destination-bound compatibility identity.
    """
    base_config = default_lfp_summary_config()
    local_root = tmp_path / "local-session"
    cluster_root = tmp_path / "cluster-session"
    source_sites = tuple(
        replace(
            site,
            lfp_path=local_root / "processed" / f"{site.stable_id}.bin",
            aligned_sync_path=local_root / "processed" / f"{site.stable_id}.sync.npz",
        )
        for site in base_config.sites
    )
    destination_sites = tuple(
        replace(
            site,
            lfp_path=cluster_root / "processed" / f"{site.stable_id}.bin",
            aligned_sync_path=cluster_root / "processed" / f"{site.stable_id}.sync.npz",
        )
        for site in base_config.sites
    )
    source_config = replace(
        base_config,
        session_path=local_root,
        output_directory=local_root / "processed" / "corrected-cache",
        sites=source_sites,
        trial_table_path=local_root / "processed" / "trials.csv",
    )
    destination_config = replace(
        base_config,
        session_path=cluster_root,
        output_directory=cluster_root / "processed" / "corrected-cache",
        sites=destination_sites,
        trial_table_path=cluster_root / "processed" / "trials.csv",
    )
    source_manifest = _manifest(
        component_fingerprint("power", source_config), config=source_config
    )
    source_manifest.update(
        {
            "configuration": json.loads(canonical_config_json(source_config)),
            "generator": {"module": "original.producer", "version": "v1"},
            "reference": {"statement": "original scientific reference"},
            "processing_metadata": {"status": "complete", "run": "local"},
        }
    )
    source_manifest["components"]["power"].update(
        {
            "completed_at": "2026-09-01T00:00:00Z",
            "generator": {"module": "original.power"},
            "units": {"psd_linear": "uV^2/Hz"},
            "scientific_metadata": {"normalization": "session_reference"},
            "source_value_semantics": source_value_semantics(source_config),
        }
    )
    source_manifest["components"]["synchrony"] = {
        **deepcopy(source_manifest["components"]["power"]),
        "file_name": "synchrony.npz",
        "configuration_fingerprint": component_fingerprint("synchrony", source_config),
        "source_fingerprints": fingerprint_source_files(source_config, component="synchrony"),
        "generator": {"module": "original.synchrony"},
        "units": {"itpc": "dimensionless"},
        "scientific_metadata": {"bootstrap_count": 200},
    }
    source_before = deepcopy(source_manifest)

    # A literal manifest copy is stale once the active configuration resolves
    # all input paths under the cluster session root.
    destination_cache = tmp_path / "literal-copy"
    destination_cache.mkdir()
    np.savez(destination_cache / "power.npz", **_power_arrays())
    np.savez(destination_cache / "synchrony.npz", **_power_arrays())
    assert assess_component_status(
        destination_cache, "power", destination_config, source_manifest
    ).status == "stale"
    assert assess_component_status(
        destination_cache, "synchrony", destination_config, source_manifest
    ).status == "stale"

    destination_source_fingerprints = {
        component: fingerprint_source_files(destination_config, component=component)
        for component in ("power", "synchrony")
    }
    rebind = getattr(lfp_summary_io, "rebind_power_synchrony_manifest")

    def forbid_filesystem(*_args: object, **_kwargs: object) -> object:
        """Fail if the pure manifest transform touches the filesystem."""
        raise AssertionError("pure manifest rebinding must not access the filesystem")

    # Source fingerprints are relocation-owned evidence. The pure helper must
    # receive those already-validated mappings and never inspect a path itself.
    with monkeypatch.context() as guarded:
        guarded.setattr(lfp_summary_io, "fingerprint_source_files", forbid_filesystem)
        guarded.setattr(Path, "open", forbid_filesystem)
        guarded.setattr(Path, "stat", forbid_filesystem)
        guarded.setattr(Path, "exists", forbid_filesystem)
        rebound_manifest = rebind(
            source_manifest,
            destination_config,
            destination_source_fingerprints=destination_source_fingerprints,
        )

    expected_manifest = deepcopy(source_manifest)
    expected_manifest["configuration"] = json.loads(
        canonical_config_json(destination_config)
    )
    for component in ("power", "synchrony"):
        expected_entry = expected_manifest["components"][component]
        expected_entry["configuration_snapshot"] = json.loads(
            canonical_config_json(destination_config)
        )
        expected_entry["configuration_fingerprint"] = component_fingerprint(
            component, destination_config
        )
        expected_entry["source_fingerprints"] = destination_source_fingerprints[
            component
        ]
        expected_entry["source_value_semantics"] = source_value_semantics(
            destination_config
        )

    assert rebound_manifest == expected_manifest
    assert source_manifest == source_before
    assert assess_component_status(
        destination_cache, "power", destination_config, rebound_manifest
    ).status == "compatible"
    assert assess_component_status(
        destination_cache, "synchrony", destination_config, rebound_manifest
    ).status == "compatible"
    # Each destination-bound snapshot must be an independent deep object. A
    # later caller may annotate one JSON tree without mutating producer history
    # or another copied component's compatibility identity.
    rebound_manifest["configuration"]["power"]["notch_quality_factor"] = 17.0
    assert rebound_manifest["components"]["power"]["configuration_snapshot"][
        "power"
    ]["notch_quality_factor"] == destination_config.power.notch_quality_factor
    assert rebound_manifest["components"]["synchrony"]["configuration_snapshot"][
        "power"
    ]["notch_quality_factor"] == destination_config.power.notch_quality_factor
    rebound_manifest["components"]["power"]["configuration_snapshot"]["power"][
        "notch_quality_factor"
    ] = 19.0
    assert rebound_manifest["configuration"]["power"]["notch_quality_factor"] == 17.0
    assert rebound_manifest["components"]["synchrony"]["configuration_snapshot"][
        "power"
    ]["notch_quality_factor"] == destination_config.power.notch_quality_factor
    assert source_manifest == source_before


def test_component_status_uses_prevalidated_current_source_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public status can validate a component without rehashing an OE sidecar.

    ``current_source_fingerprints`` is the exact active component-scoped
    mapping obtained by the relocation command. It avoids a second source
    fingerprint pass after the command has already streamed and validated each
    unique input.
    """
    base = default_lfp_summary_config()
    lfp_path = tmp_path / "PFC.dat"
    sync_path = tmp_path / "PFC.sync.npz"
    lfp_path.write_bytes(b"open-ephys-lfp")
    sync_path.write_bytes(b"sync")
    (tmp_path / "lfp_preprocessing.json").write_text("{}", encoding="ascii")
    open_ephys_site = replace(
        base.sites[0],
        acquisition_format="open_ephys",
        lfp_path=lfp_path,
        aligned_sync_path=sync_path,
    )
    config = replace(base, sites=(open_ephys_site,) + base.sites[1:])
    manifest = _manifest(component_fingerprint("power", config), config=config)
    np.savez(tmp_path / "power.npz", **_power_arrays())
    current_source_fingerprints = fingerprint_source_files(config, component="power")

    def forbid_rehash(*_args: object, **_kwargs: object) -> object:
        """Fail if public status ignores its supplied current fingerprints."""
        raise AssertionError("status must reuse the supplied source fingerprints")

    def forbid_array_materialization(*_args: object, **_kwargs: object) -> object:
        """Fail if headers-only status routes through the full array loader."""
        raise AssertionError("headers-only status must not materialize NPZ arrays")

    header_validations: list[tuple[Path, str]] = []
    validate_headers = getattr(lfp_summary_io, "validate_component_npz_headers")

    def record_header_validation(
        component_path: Path,
        status_manifest: dict[str, object],
        component: str,
    ) -> object:
        """Record public status' bounded schema check without replacing it."""
        header_validations.append((component_path, component))
        return validate_headers(component_path, status_manifest, component)

    monkeypatch.setattr(lfp_summary_io, "fingerprint_source_files", forbid_rehash)
    monkeypatch.setattr(lfp_summary_io, "load_component_arrays", forbid_array_materialization)
    monkeypatch.setattr(
        lfp_summary_io, "validate_component_npz_headers", record_header_validation
    )
    status = assess_component_status(
        tmp_path,
        "power",
        config,
        manifest,
        current_source_fingerprints=current_source_fingerprints,
        validate_headers_only=True,
    )

    assert status.status == "compatible"
    assert header_validations == [(tmp_path / "power.npz", "power")]


def test_header_only_component_validation_reads_zip_npy_headers_without_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header validation checks schema safely without materializing NPZ members.

    ``validate_component_npz_headers`` accepts one component archive plus its
    manifest schema and verifies ZIP member names, safe NPY dtypes, shapes, and
    named axes from headers only. It must reject a declared enormous/truncated
    member without allocating its 232 MB payload.
    """
    config = default_lfp_summary_config()
    manifest = _manifest(component_fingerprint("power", config), config=config)
    component_path = tmp_path / "power.npz"
    np.savez(component_path, **_power_arrays())
    validate_headers = getattr(lfp_summary_io, "validate_component_npz_headers")

    def forbid_array_loading(*_args: object, **_kwargs: object) -> object:
        """Fail if header validation tries NumPy's array-materializing loader."""
        raise AssertionError("header-only validation must not call numpy.load")

    original_member_read = zipfile.ZipExtFile.read

    def require_bounded_member_reads(
        member: zipfile.ZipExtFile,
        size: int | None = -1,
    ) -> bytes:
        """Forbid ZIP member reads that could materialize a full NPZ payload."""
        if size is None or size < 0:
            raise AssertionError("header validation must use bounded ZIP member reads")
        return original_member_read(member, size)

    monkeypatch.setattr(lfp_summary_io.np, "load", forbid_array_loading)
    monkeypatch.setattr(zipfile.ZipExtFile, "read", require_bounded_member_reads)
    validate_headers(component_path, manifest, "power")

    truncated_path = tmp_path / "truncated-power.npz"
    with zipfile.ZipFile(truncated_path, "w") as archive:
        with archive.open("values.npy", "w") as member:
            np.lib.format.write_array_header_1_0(
                member,
                {
                    "descr": "<f8",
                    "fortran_order": False,
                    "shape": (29_000_000,),
                },
            )
    bounded_manifest = deepcopy(manifest)
    bounded_manifest["components"]["power"]["array_schema"] = {
        "values": {"axes": ["trial"], "units": "dimensionless"}
    }

    with pytest.raises(ValueError, match="truncated|byte|NPY|array"):
        validate_headers(truncated_path, bounded_manifest, "power")


@pytest.mark.parametrize(
    ("arrays", "array_schema", "error"),
    (
        (
            {"values": np.array([object()], dtype=object)},
            {"values": {"axes": ["trial"], "units": "dimensionless"}},
            "object|pickle|dtype",
        ),
        (
            {"values": np.ones((1, 1), dtype=np.float64)},
            {"values": {"axes": ["trial"], "units": "dimensionless"}},
            "rank|axis|shape",
        ),
        (
            {
                "first": np.ones((1,), dtype=np.float64),
                "second": np.ones((2,), dtype=np.float64),
            },
            {
                "first": {"axes": ["trial"], "units": "dimensionless"},
                "second": {"axes": ["trial"], "units": "dimensionless"},
            },
            "axis|shape|inconsistent",
        ),
    ),
)
def test_header_only_component_validation_rejects_unsafe_dtype_and_schema_contracts(
    tmp_path: Path,
    arrays: dict[str, np.ndarray],
    array_schema: dict[str, dict[str, object]],
    error: str,
) -> None:
    """Header-only validation rejects unsafe dtypes and inconsistent named axes."""
    config = default_lfp_summary_config()
    manifest = _manifest(component_fingerprint("power", config), config=config)
    manifest["components"]["power"]["array_schema"] = array_schema
    component_path = tmp_path / "invalid-power.npz"
    np.savez(component_path, **arrays)
    validate_headers = getattr(lfp_summary_io, "validate_component_npz_headers")

    with pytest.raises(ValueError, match=error):
        validate_headers(component_path, manifest, "power")


@pytest.mark.parametrize("member_case", ("unsafe", "undeclared_safe", "duplicate"))
def test_header_only_component_validation_rejects_unsafe_extra_or_duplicate_member(
    tmp_path: Path,
    member_case: str,
) -> None:
    """The archive may contain each schema-declared ``.npy`` member exactly once."""
    config = default_lfp_summary_config()
    manifest = _manifest(component_fingerprint("power", config), config=config)
    component_path = tmp_path / "extra-member-power.npz"
    safe_extra_path = tmp_path / "safe-extra.npy"
    np.savez(component_path, **_power_arrays())
    np.save(safe_extra_path, np.array((1.0,), dtype=np.float64))
    with zipfile.ZipFile(component_path, "a") as archive:
        if member_case == "unsafe":
            archive.writestr("untrusted.txt", b"not an NPY member")
        elif member_case == "undeclared_safe":
            archive.write(safe_extra_path, arcname="undeclared.npy")
        else:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.write(safe_extra_path, arcname="trial_indices.npy")
    validate_headers = getattr(lfp_summary_io, "validate_component_npz_headers")

    with pytest.raises(ValueError, match="extra|member|unsafe|schema|duplicate"):
        validate_headers(component_path, manifest, "power")
