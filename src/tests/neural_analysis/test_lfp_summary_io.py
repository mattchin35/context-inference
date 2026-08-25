"""Contract tests for LFP-summary NPZ cache validation and transactions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.neural_analysis.lfp_summary_io import (
    assess_component_status,
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
)


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
        "psd_linear": {"axes": ["site", "trial", "epoch", "frequency"], "units": "uV^2/Hz"},
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
