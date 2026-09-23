"""RED contracts for byte-preserving Power/Synchrony cache relocation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from src.neural_analysis.lfp_loading import sha256_file_content
from src.neural_analysis.lfp_summary_io import assess_component_status
from src.neural_analysis import lfp_summary_io, lfp_summary_models
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    UnitPopulationConfig,
    canonical_config_json,
    component_fingerprint,
    default_lfp_summary_config,
    fingerprint_source_files,
    source_value_semantics,
)


@dataclass(frozen=True)
class _RelocationFixture:
    """Synthetic local/cluster roots and a complete two-component source cache."""

    local_root: Path
    cluster_root: Path
    source_cache: Path
    destination_cache: Path
    legacy_cache: Path
    receipt_path: Path
    source_config_json_path: Path
    destination_config_json_path: Path
    source_config: LFPSummaryConfig
    destination_config: LFPSummaryConfig
    source_manifest: dict[str, object]
    local_to_cluster_sources: dict[Path, Path]


def _component_schema() -> dict[str, dict[str, object]]:
    """Return a minimal safe NPZ schema for synthetic relocation receipts.

    Returns
    -------
    dict[str, dict[str, object]]
        One one-dimensional numerical array with no physical-unit conversion.
    """
    return {"values": {"axes": ["trial"], "units": "dimensionless"}}


def _component_entry(
    component: str,
    config: LFPSummaryConfig,
) -> dict[str, object]:
    """Return one complete synthetic manifest entry for a copied component.

    Parameters
    ----------
    component : str
        ``"power"`` or ``"synchrony"``.
    config : LFPSummaryConfig
        Local producer configuration used for source fingerprints.

    Returns
    -------
    dict[str, object]
        JSON-ready complete component metadata with intentionally distinct
        producer/scientific fields that relocation must preserve exactly.
    """
    units = (
        {"psd_linear": "uV^2/Hz"}
        if component == "power"
        else {"itpc": "dimensionless", "ispc": "dimensionless"}
    )
    scientific_metadata = (
        {"normalization": "session_reference"}
        if component == "power"
        else {"bootstrap_count": 200, "phase_offset": "site_a_minus_site_b"}
    )
    return {
        "file_name": f"{component}.npz",
        "state": "complete",
        "completed_at": "2026-09-01T00:00:00Z",
        "generator": {"module": f"original.{component}", "version": "v1"},
        "configuration_fingerprint": component_fingerprint(component, config),
        "configuration_snapshot": json.loads(canonical_config_json(config)),
        "source_fingerprints": fingerprint_source_files(config, component=component),
        "source_value_semantics": source_value_semantics(config),
        "array_schema": _component_schema(),
        "units": units,
        "scientific_metadata": scientific_metadata,
    }


def _mapped_config(
    root: Path,
    *,
    output_directory: Path,
) -> LFPSummaryConfig:
    """Build one valid config whose every Power/Synchrony input lies below root.

    Parameters
    ----------
    root : pathlib.Path
        Local or cluster session root used for every input path.
    output_directory : pathlib.Path
        Cache directory below the same root. This location is intentionally
        excluded from scientific source-equivalence comparisons.

    Returns
    -------
    LFPSummaryConfig
        Valid configuration with two recording formats, an Open Ephys sidecar,
        LFP binaries, aligned-sync NPZs, and a trial table.
    """
    base = default_lfp_summary_config()
    sites = []
    for index, site in enumerate(base.sites):
        recording_directory = root / "processed" / "inputs" / site.stable_id
        sites.append(
            replace(
                site,
                acquisition_format="open_ephys" if index == 0 else "spikeglx",
                lfp_path=recording_directory / f"{site.stable_id}.bin",
                aligned_sync_path=recording_directory / f"{site.stable_id}.sync.npz",
            )
        )
    return replace(
        base,
        session_path=root,
        output_directory=output_directory,
        sites=tuple(sites),
        trial_table_path=root / "processed" / "trials.csv",
    )


def _write_equivalent_sources(
    source_config: LFPSummaryConfig,
    destination_config: LFPSummaryConfig,
) -> dict[Path, Path]:
    """Write byte-identical synthetic Power/Synchrony source inputs on both roots.

    Parameters
    ----------
    source_config, destination_config : LFPSummaryConfig
        Configurations whose corresponding paths are one root substitution apart.

    Returns
    -------
    dict[pathlib.Path, pathlib.Path]
        Resolved local source paths mapped one-to-one to their cluster paths.
        Files contain arbitrary test bytes, not LFP numerical samples.
    """
    mapping: dict[Path, Path] = {}
    for source_site, destination_site in zip(
        source_config.sites, destination_config.sites, strict=True
    ):
        for source_path, destination_path, payload in (
            (
                source_site.lfp_path,
                destination_site.lfp_path,
                f"lfp-{source_site.stable_id}".encode("ascii"),
            ),
            (
                source_site.aligned_sync_path,
                destination_site.aligned_sync_path,
                f"sync-{source_site.stable_id}".encode("ascii"),
            ),
        ):
            source_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(payload)
            destination_path.write_bytes(payload)
            mapping[source_path.resolve()] = destination_path.resolve()
    assert source_config.trial_table_path is not None
    assert destination_config.trial_table_path is not None
    source_config.trial_table_path.parent.mkdir(parents=True, exist_ok=True)
    destination_config.trial_table_path.parent.mkdir(parents=True, exist_ok=True)
    source_config.trial_table_path.write_text("choice_time\n1.0\n", encoding="ascii")
    destination_config.trial_table_path.write_text("choice_time\n1.0\n", encoding="ascii")
    mapping[source_config.trial_table_path.resolve()] = destination_config.trial_table_path.resolve()

    # The first configured site is Open Ephys, so its preprocessing sidecar is
    # a Power/Synchrony source with fixed affine-uV value semantics.
    source_sidecar = source_config.sites[0].lfp_path.parent / "lfp_preprocessing.json"
    destination_sidecar = destination_config.sites[0].lfp_path.parent / "lfp_preprocessing.json"
    source_sidecar.write_text('{"unit":"uV"}', encoding="ascii")
    destination_sidecar.write_text('{"unit":"uV"}', encoding="ascii")
    mapping[source_sidecar.resolve()] = destination_sidecar.resolve()
    return mapping


@pytest.fixture
def relocation_fixture(tmp_path: Path) -> _RelocationFixture:
    """Create local and cluster roots without copying any experimental data."""
    local_root = tmp_path / "local-session"
    cluster_root = tmp_path / "cluster-session"
    source_cache = local_root / "processed" / "corrected-cache"
    destination_cache = cluster_root / "processed" / "corrected-cache-v2"
    source_config = _mapped_config(local_root, output_directory=source_cache)
    destination_config = _mapped_config(cluster_root, output_directory=destination_cache)
    source_mapping = _write_equivalent_sources(source_config, destination_config)
    source_cache.mkdir(parents=True)
    legacy_cache = cluster_root / "processed" / "lfp_summary_cache"
    legacy_cache.mkdir(parents=True)
    (legacy_cache / "legacy-sentinel.txt").write_bytes(b"protected legacy cache")
    source_manifest: dict[str, object] = {
        "schema_version": source_config.schema_version,
        "session_id": source_config.session_id,
        "configuration": json.loads(canonical_config_json(source_config)),
        "generator": {
            "module": "original.producer",
            "version": "v1",
            "producer_label": "caf\u00e9",
        },
        "reference": {"statement": "local corrected producer \u03b8"},
        "processing_metadata": {"status": "complete", "run": "local"},
        "components": {
            component: _component_entry(component, source_config)
            for component in ("power", "synchrony")
        },
    }
    (source_cache / "manifest.json").write_text(
        json.dumps(source_manifest, sort_keys=True), encoding="ascii"
    )
    np.savez(source_cache / "power.npz", values=np.array((1.0, 2.0)))
    np.savez(source_cache / "synchrony.npz", values=np.array((3.0, 4.0)))
    source_config_json_path = tmp_path / "source-config.json"
    destination_config_json_path = tmp_path / "destination-config.json"
    source_config_json_path.write_text(
        canonical_config_json(source_config), encoding="ascii"
    )
    destination_config_json_path.write_text(
        canonical_config_json(destination_config), encoding="ascii"
    )
    return _RelocationFixture(
        local_root=local_root,
        cluster_root=cluster_root,
        source_cache=source_cache,
        destination_cache=destination_cache,
        legacy_cache=legacy_cache,
        receipt_path=cluster_root / "analysis_runs" / "relocation-receipt.json",
        source_config_json_path=source_config_json_path,
        destination_config_json_path=destination_config_json_path,
        source_config=source_config,
        destination_config=destination_config,
        source_manifest=source_manifest,
        local_to_cluster_sources=source_mapping,
    )


def _relocation_api() -> tuple[type[object], Any]:
    """Return the frozen relocation request and command public interfaces.

    Returns
    -------
    tuple[type, callable]
        ``CacheRelocationRequest`` and
        ``relocate_power_synchrony_cache(request)`` from the dedicated command
        module. The latter performs only metadata validation and byte copying;
        it never computes Power, Synchrony, phase, or Spike-phase values.
    """
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    return (
        getattr(module, "CacheRelocationRequest"),
        getattr(module, "relocate_power_synchrony_cache"),
    )


def _request(
    fixture: _RelocationFixture,
    *,
    source_cache: Path | None = None,
    source_config: LFPSummaryConfig | None = None,
    destination_cache: Path | None = None,
    destination_config: LFPSummaryConfig | None = None,
    receipt_path: Path | None = None,
    local_session_root: Path | None = None,
    cluster_session_root: Path | None = None,
    git_commit: str | None = None,
) -> object:
    """Build one relocation request with explicit roots and durable provenance.

    Parameters
    ----------
    fixture : _RelocationFixture
        Synthetic source/destination root fixture.
    source_cache : pathlib.Path or None
        Optional claimed local cache path. ``None`` uses the producer cache.
    source_config : LFPSummaryConfig or None
        Optional claimed producer configuration. ``None`` uses the matching one.
    destination_cache : pathlib.Path or None
        Optional requested final cache path. ``None`` uses the configured target.
    destination_config : LFPSummaryConfig or None
        Optional active cluster configuration. ``None`` uses the equivalent one.
    receipt_path, local_session_root, cluster_session_root, git_commit :
        pathlib.Path or str or None
        Optional external receipt, declared roots, or exact producer revision.
        ``None`` uses fixture values.

    Returns
    -------
    CacheRelocationRequest
        Immutable request containing only path/provenance metadata; no numerical
        arrays or experimental files are loaded by this test helper.
    """
    request_type, _ = _relocation_api()
    return request_type(
        source_cache_directory=(fixture.source_cache if source_cache is None else source_cache),
        destination_cache_directory=(
            fixture.destination_cache if destination_cache is None else destination_cache
        ),
        source_config=fixture.source_config if source_config is None else source_config,
        destination_config=(
            fixture.destination_config
            if destination_config is None
            else destination_config
        ),
        local_session_root=(
            fixture.local_root if local_session_root is None else local_session_root
        ),
        cluster_session_root=(
            fixture.cluster_root if cluster_session_root is None else cluster_session_root
        ),
        receipt_path=fixture.receipt_path if receipt_path is None else receipt_path,
        command=(
            "python",
            "-m",
            "src.neural_analysis.lfp_summary_cache_relocation",
            "--synthetic-test",
        ),
        git_commit="a" * 40 if git_commit is None else git_commit,
    )


def _relocate(
    fixture: _RelocationFixture,
    *,
    runtime: object | None = None,
    **request_kwargs: object,
) -> object:
    """Execute the frozen relocation API on synthetic files only.

    Parameters
    ----------
    fixture : _RelocationFixture
        Local/cluster fixture.
    runtime : RelocationRuntime or None
        Optional deterministic clock, hostname, duration, and RSS provider.
    **request_kwargs : object
        Optional request replacements accepted by ``_request``.

    Returns
    -------
    object
        Command result object supplied by the relocation implementation.
    """
    _, relocate = _relocation_api()
    request = _request(fixture, **request_kwargs)
    if runtime is None:
        return relocate(request)
    return relocate(request, runtime=runtime)


def _runtime() -> object:
    """Return deterministic runtime provenance for one successful relocation.

    Returns
    -------
    RelocationRuntime
        Test-only deterministic providers for ordered UTC timestamps, hostname,
        wall-clock duration, and peak resident bytes. The result never controls
        scientific computation or copied component bytes.
    """
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    timestamps = iter(
        (
            datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 23, 12, 0, 3, tzinfo=timezone.utc),
        )
    )
    monotonic_values = iter((10.0, 12.5))
    return getattr(module, "RelocationRuntime")(
        utc_now=lambda: next(timestamps),
        hostname=lambda: "synthetic-cluster",
        monotonic_seconds=lambda: next(monotonic_values),
        peak_rss_bytes=lambda: 4321,
    )


def _directory_inventory(directory: Path) -> dict[str, bytes | str]:
    """Return exact relative file bytes or link targets for a small test directory.

    Parameters
    ----------
    directory : pathlib.Path
        Synthetic cache directory containing only bounded test fixtures.

    Returns
    -------
    dict[str, bytes | str]
        Relative regular-file bytes and symlink target strings. This test helper
        intentionally reads tiny fixture files only after relocation returns.
    """
    inventory: dict[str, bytes | str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        relative = str(path.relative_to(directory))
        inventory[relative] = os.readlink(path) if path.is_symlink() else path.read_bytes()
    return inventory


def _path_snapshot(path: Path) -> tuple[str, bytes | str | dict[str, bytes | str] | None]:
    """Capture absence, file bytes, symlink target, or bounded directory inventory."""
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if not path.exists():
        return ("absent", None)
    if path.is_dir():
        return ("directory", _directory_inventory(path))
    return ("file", path.read_bytes())


def _protected_inventories(
    fixture: _RelocationFixture,
) -> tuple[dict[str, bytes | str], dict[str, bytes | str]]:
    """Return immutable source and legacy cache inventories for preservation checks."""
    return (
        _directory_inventory(fixture.source_cache),
        _directory_inventory(fixture.legacy_cache),
    )


def test_relocation_copies_only_validated_components_and_records_provenance(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relocation publishes a cluster-compatible two-component cache and receipt."""
    source_inventory_before, legacy_inventory_before = _protected_inventories(
        relocation_fixture
    )
    assert (relocation_fixture.source_cache / "manifest.json").read_bytes().isascii()

    for component in ("power", "synchrony"):
        assert assess_component_status(
            relocation_fixture.source_cache,
            component,
            relocation_fixture.source_config,
            relocation_fixture.source_manifest,
        ).status == "compatible"

    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    original_status = getattr(module, "assess_component_status")
    inspected_components: list[tuple[Path, str, dict[str, object]]] = []
    header_validations: list[tuple[Path, str]] = []

    def recording_status(
        cache_directory: Path,
        component: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Record public compatibility checks without changing their result."""
        inspected_components.append((cache_directory, component, dict(kwargs)))
        return original_status(cache_directory, component, *args, **kwargs)

    original_header_validation = getattr(lfp_summary_io, "validate_component_npz_headers")

    def recording_header_validation(
        component_path: Path,
        manifest: object,
        component: str,
    ) -> object:
        """Record bounded NPZ header checks without loading component arrays."""
        header_validations.append((component_path, component))
        return original_header_validation(component_path, manifest, component)

    def forbid_array_materialization(*_args: object, **_kwargs: object) -> object:
        """Fail if relocation opens a complete component payload into arrays."""
        raise AssertionError("relocation must validate component NPZ headers only")

    monkeypatch.setattr(module, "assess_component_status", recording_status)
    monkeypatch.setattr(lfp_summary_io, "load_component_arrays", forbid_array_materialization)
    monkeypatch.setattr(
        lfp_summary_io, "validate_component_npz_headers", recording_header_validation
    )
    result = _relocate(relocation_fixture, runtime=_runtime())

    expected_status_checks = {
        (relocation_fixture.source_cache, "power"),
        (relocation_fixture.source_cache, "synchrony"),
        (relocation_fixture.destination_cache, "power"),
        (relocation_fixture.destination_cache, "synchrony"),
        (relocation_fixture.destination_cache, "spike_phase"),
    }
    assert expected_status_checks.issubset(
        {(cache_directory, component) for cache_directory, component, _ in inspected_components}
    )
    assert all(
        options.get("validate_headers_only") is True
        and "current_source_fingerprints" in options
        for _, component, options in inspected_components
        if component in {"power", "synchrony"}
    )
    assert {
        (relocation_fixture.source_cache / "power.npz", "power"),
        (relocation_fixture.source_cache / "synchrony.npz", "synchrony"),
        (relocation_fixture.destination_cache / "power.npz", "power"),
        (relocation_fixture.destination_cache / "synchrony.npz", "synchrony"),
    }.issubset(header_validations)

    destination = relocation_fixture.destination_cache
    assert getattr(result, "destination_cache_directory") == destination
    assert sorted(path.name for path in destination.iterdir()) == [
        "manifest.json",
        "power.npz",
        "synchrony.npz",
    ]
    for component in ("power", "synchrony"):
        source_component = relocation_fixture.source_cache / f"{component}.npz"
        destination_component = destination / f"{component}.npz"
        assert sha256_file_content(destination_component) == sha256_file_content(source_component)
        assert assess_component_status(
            destination,
            component,
            relocation_fixture.destination_config,
            json.loads((destination / "manifest.json").read_text(encoding="ascii")),
            validate_headers_only=True,
        ).status == "compatible"
    destination_manifest = json.loads((destination / "manifest.json").read_text(encoding="ascii"))
    assert assess_component_status(
        destination,
        "spike_phase",
        relocation_fixture.destination_config,
        destination_manifest,
        validate_headers_only=True,
    ).status == "missing"
    assert destination_manifest["configuration"] == json.loads(
        canonical_config_json(relocation_fixture.destination_config)
    )
    for component in ("power", "synchrony"):
        source_entry = relocation_fixture.source_manifest["components"][component]
        destination_entry = destination_manifest["components"][component]
        for field in (
            "state",
            "completed_at",
            "generator",
            "array_schema",
            "units",
            "scientific_metadata",
            "source_value_semantics",
        ):
            assert destination_entry[field] == source_entry[field]
        assert destination_entry["configuration_snapshot"] == json.loads(
            canonical_config_json(relocation_fixture.destination_config)
        )
        assert destination_entry["configuration_fingerprint"] == component_fingerprint(
            component, relocation_fixture.destination_config
        )
        assert destination_entry["source_fingerprints"] == fingerprint_source_files(
            relocation_fixture.destination_config, component=component
        )

    receipt_bytes = relocation_fixture.receipt_path.read_bytes()
    assert receipt_bytes.isascii()
    receipt = json.loads(receipt_bytes.decode("ascii"))
    assert receipt["source_producer_manifest"] == relocation_fixture.source_manifest
    assert receipt["source_producer_manifest"]["generator"]["producer_label"] == "caf\u00e9"
    assert receipt["source_producer_manifest"]["reference"]["statement"].endswith("\u03b8")
    assert receipt["source_producer_manifest_sha256"] == sha256_file_content(
        relocation_fixture.source_cache / "manifest.json"
    )
    assert receipt["rebound_manifest_sha256"] == sha256_file_content(
        destination / "manifest.json"
    )
    assert receipt["component_hashes"] == {
        component: {
            "local_path": str(relocation_fixture.source_cache / f"{component}.npz"),
            "destination_path": str(destination / f"{component}.npz"),
            "local": {
                "size_bytes": (relocation_fixture.source_cache / f"{component}.npz").stat().st_size,
                "sha256": sha256_file_content(relocation_fixture.source_cache / f"{component}.npz"),
            },
            "destination": {
                "size_bytes": (destination / f"{component}.npz").stat().st_size,
                "sha256": sha256_file_content(destination / f"{component}.npz"),
            },
        }
        for component in ("power", "synchrony")
    }
    assert receipt["command"] == [
        "python",
        "-m",
        "src.neural_analysis.lfp_summary_cache_relocation",
        "--synthetic-test",
    ]
    assert receipt["git_commit"] == "a" * 40
    assert receipt["local_session_root"] == str(relocation_fixture.local_root.resolve())
    assert receipt["cluster_session_root"] == str(relocation_fixture.cluster_root.resolve())
    assert receipt["source_cache_directory"] == str(relocation_fixture.source_cache.resolve())
    assert receipt["destination_cache_directory"] == str(destination.resolve())
    assert receipt["component_states"] == {
        "power": "compatible",
        "synchrony": "compatible",
        "spike_phase": "missing",
    }
    assert receipt["hostname"] == "synthetic-cluster"
    assert receipt["started_at_utc"] == "2026-09-23T12:00:00Z"
    assert receipt["completed_at_utc"] == "2026-09-23T12:00:03Z"
    assert datetime.fromisoformat(receipt["started_at_utc"].replace("Z", "+00:00")) < datetime.fromisoformat(
        receipt["completed_at_utc"].replace("Z", "+00:00")
    )
    assert receipt["wall_seconds"] == 2.5
    assert receipt["peak_rss_bytes"] == 4321
    assert set(receipt["source_hashes"]) == {
        str(source_path) for source_path in relocation_fixture.local_to_cluster_sources
    }
    for source_path, cluster_path in relocation_fixture.local_to_cluster_sources.items():
        expected_source_record: dict[str, object] = {
            "cluster_path": str(cluster_path),
            "local": {
                "size_bytes": source_path.stat().st_size,
                "sha256": sha256_file_content(source_path),
            },
            "cluster": {
                "size_bytes": cluster_path.stat().st_size,
                "sha256": sha256_file_content(cluster_path),
            },
        }
        source_fingerprint = fingerprint_source_files(
            relocation_fixture.source_config, component="power"
        )[str(source_path)]
        if "value_semantics" in source_fingerprint:
            expected_source_record["value_semantics"] = source_fingerprint[
                "value_semantics"
            ]
        assert receipt["source_hashes"][str(source_path)] == expected_source_record
    assert _directory_inventory(relocation_fixture.source_cache) == source_inventory_before
    assert _directory_inventory(relocation_fixture.legacy_cache) == legacy_inventory_before


def test_relocation_hashes_each_unique_power_synchrony_source_once_per_host(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One streamed-hash seam covers inputs, status, and copied component bytes."""
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    observed_calls: dict[Path, int] = {}
    staged_component_paths: set[Path] = set()

    def counted_hash(path: Path | str, **kwargs: object) -> str:
        """Record each streamed file hash before delegating to the real helper."""
        resolved = Path(path).resolve()
        observed_calls[resolved] = observed_calls.get(resolved, 0) + 1
        return sha256_file_content(resolved, **kwargs)

    original_publish = getattr(module, "_publish_staged_cache")

    def inspect_then_publish(staging_directory: Path, destination_directory: Path) -> None:
        """Capture the component archive paths before atomic publication."""
        staged_component_paths.update(
            (staging_directory / "power.npz", staging_directory / "synchrony.npz")
        )
        original_publish(staging_directory, destination_directory)

    original_read_bytes = Path.read_bytes
    def forbid_bulk_reads(path: Path) -> bytes:
        """Reject every source, staging, or final component full-file read."""
        if path.name in {"power.npz", "synchrony.npz"}:
            raise AssertionError(
                "relocation must stream/hash and copy component NPZs without Path.read_bytes"
            )
        return original_read_bytes(path)

    # ``lfp_summary_models.sha256_file_content`` is the one injectable seam
    # already reached by live source fingerprints and public status checks.
    # Relocation must use that same live seam for every additional byte hash.
    monkeypatch.setattr(lfp_summary_models, "sha256_file_content", counted_hash)
    monkeypatch.setattr(module, "_publish_staged_cache", inspect_then_publish)
    monkeypatch.setattr(Path, "read_bytes", forbid_bulk_reads)

    _relocate(relocation_fixture)

    for source_path, cluster_path in relocation_fixture.local_to_cluster_sources.items():
        assert observed_calls[source_path] == 1
        assert observed_calls[cluster_path] == 1
    assert staged_component_paths
    for component in ("power", "synchrony"):
        source_component = (relocation_fixture.source_cache / f"{component}.npz").resolve()
        destination_component = (
            relocation_fixture.destination_cache / f"{component}.npz"
        ).resolve()
        staged_component = next(
            path.resolve()
            for path in staged_component_paths
            if path.name == f"{component}.npz"
        )
        assert observed_calls[source_component] == 1
        assert observed_calls.get(staged_component, 0) + observed_calls.get(
            destination_component, 0
        ) == 1


def test_relocation_aborts_when_a_copied_component_hash_changes(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy-time corruption cannot publish a cache or a success receipt."""
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    copy_component = getattr(module, "_copy_component_file")

    def corrupting_copy(source_path: Path, destination_path: Path) -> None:
        """Copy then append one byte so the staged SHA-256 cannot match source."""
        copy_component(source_path, destination_path)
        with destination_path.open("ab") as handle:
            handle.write(b"x")

    monkeypatch.setattr(module, "_copy_component_file", corrupting_copy)

    with pytest.raises(ValueError, match="component|hash|SHA|byte"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()
    assert not any(
        path.name.startswith(".") and "stage" in path.name
        for path in relocation_fixture.destination_cache.parent.iterdir()
    )


@pytest.mark.parametrize(
    "category",
    (
        "session_root",
        "session_id",
        "schema_version",
        "lfp_path",
        "aligned_sync_path",
        "site_field",
        "site_pairs",
        "trial_filter",
        "analysis_windows",
        "power",
        "phase",
    ),
)
def test_relocation_rejects_each_relevant_configuration_difference(
    relocation_fixture: _RelocationFixture,
    category: str,
) -> None:
    """Only one declared root mapping may differ for copied component inputs."""
    destination = relocation_fixture.destination_config
    if category == "session_root":
        changed_config = replace(
            destination,
            session_path=relocation_fixture.cluster_root / "other-session",
        )
    elif category == "session_id":
        changed_config = replace(destination, session_id="different-session")
    elif category == "schema_version":
        changed_config = replace(destination, schema_version="2")
    elif category == "lfp_path":
        changed_config = replace(
            destination,
            sites=(
                replace(
                    destination.sites[0],
                    lfp_path=destination.sites[0].lfp_path.parent / "other-lfp.bin",
                ),
            )
            + destination.sites[1:],
        )
    elif category == "aligned_sync_path":
        changed_config = replace(
            destination,
            sites=(
                replace(
                    destination.sites[0],
                    aligned_sync_path=destination.sites[0].aligned_sync_path.parent
                    / "other-sync.npz",
                ),
            )
            + destination.sites[1:],
        )
    elif category == "site_field":
        changed_config = replace(
            destination,
            sites=(
                replace(
                    destination.sites[0],
                    saved_channel_index=destination.sites[0].saved_channel_index + 1,
                ),
            )
            + destination.sites[1:],
        )
    elif category == "site_pairs":
        changed_config = replace(
            destination,
            site_pairs=(("HPC1", "PFC"), ("PFC", "HPC2"), ("HPC1", "HPC2")),
        )
    elif category == "trial_filter":
        changed_config = replace(
            destination,
            trial_filter=replace(destination.trial_filter, context="left"),
        )
    elif category == "analysis_windows":
        changed_config = replace(
            destination,
            analysis_windows=replace(
                destination.analysis_windows,
                whole_stop_s=3.0,
                after_stop_s=3.0,
            ),
        )
    elif category == "power":
        changed_config = replace(
            destination,
            power=replace(
                destination.power,
                notch_quality_factor=17.0,
            ),
        )
    else:
        changed_config = replace(
            destination,
            phase=replace(
                destination.phase,
                morlet_gaussian_width=2.0,
            ),
        )

    with pytest.raises(ValueError, match="configuration|scientific|mapping|session|schema|source"):
        _relocate(relocation_fixture, destination_config=changed_config)

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()


@pytest.mark.parametrize("category", ("output", "unit_population", "ppc", "ppc_execution", "random_seed"))
def test_relocation_ignores_only_noncomponent_configuration_fields(
    relocation_fixture: _RelocationFixture,
    category: str,
) -> None:
    """Output, spike/PPC, and top-level seed fields do not alter copied components."""
    destination = relocation_fixture.destination_config
    destination_cache = relocation_fixture.destination_cache
    if category == "output":
        destination_cache = (
            relocation_fixture.cluster_root / "processed" / "another-corrected-cache"
        )
        changed_config = replace(destination, output_directory=destination_cache)
    elif category == "unit_population":
        changed_config = replace(
            destination,
            unit_population=UnitPopulationConfig(
                "preview-units",
                "ProbeB",
                relocation_fixture.cluster_root / "processed" / "spikes" / "sorter",
                relocation_fixture.cluster_root / "processed" / "spikes" / "aligned.npy",
                (1,),
                (),
                ("ProbeB:1",),
            ),
        )
    elif category == "ppc":
        changed_config = replace(
            destination,
            ppc=replace(destination.ppc, shuffle_count=17),
        )
    elif category == "ppc_execution":
        changed_config = replace(
            destination,
            ppc_execution=replace(destination.ppc_execution, worker_count=3),
        )
    else:
        changed_config = replace(destination, random_seed=99)

    _relocate(
        relocation_fixture,
        destination_cache=destination_cache,
        destination_config=changed_config,
    )

    manifest = json.loads((destination_cache / "manifest.json").read_text(encoding="ascii"))
    for component in ("power", "synchrony"):
        assert assess_component_status(
            destination_cache,
            component,
            changed_config,
            manifest,
        ).status == "compatible"


@pytest.mark.parametrize("site_field", ("label", "stable_id"))
def test_relocation_does_not_root_map_path_looking_site_identity_strings(
    relocation_fixture: _RelocationFixture,
    site_field: str,
) -> None:
    """The sole root substitution applies only to explicit configuration paths."""
    source_site = replace(
        relocation_fixture.source_config.sites[0],
        **{site_field: str(relocation_fixture.local_root)},
    )
    destination_site = replace(
        relocation_fixture.destination_config.sites[0],
        **{site_field: str(relocation_fixture.cluster_root)},
    )
    source_pairs = relocation_fixture.source_config.site_pairs
    destination_pairs = relocation_fixture.destination_config.site_pairs
    if site_field == "stable_id":
        source_pairs = tuple(
            tuple(
                str(relocation_fixture.local_root) if site_id == "PFC" else site_id
                for site_id in pair
            )
            for pair in source_pairs
        )
        destination_pairs = tuple(
            tuple(
                str(relocation_fixture.cluster_root) if site_id == "PFC" else site_id
                for site_id in pair
            )
            for pair in destination_pairs
        )
    source_config = replace(
        relocation_fixture.source_config,
        sites=(source_site,) + relocation_fixture.source_config.sites[1:],
        site_pairs=source_pairs,
    )
    destination_config = replace(
        relocation_fixture.destination_config,
        sites=(destination_site,) + relocation_fixture.destination_config.sites[1:],
        site_pairs=destination_pairs,
    )
    source_manifest = relocation_fixture.source_manifest
    source_snapshot = json.loads(canonical_config_json(source_config))
    source_manifest["configuration"] = source_snapshot
    for component in ("power", "synchrony"):
        source_entry = source_manifest["components"][component]
        source_entry["configuration_snapshot"] = source_snapshot
        source_entry["configuration_fingerprint"] = component_fingerprint(
            component, source_config
        )
        source_entry["source_fingerprints"] = fingerprint_source_files(
            source_config, component=component
        )
        source_entry["source_value_semantics"] = source_value_semantics(source_config)
    (relocation_fixture.source_cache / "manifest.json").write_text(
        json.dumps(source_manifest, sort_keys=True), encoding="ascii"
    )
    source_before, legacy_before = _protected_inventories(relocation_fixture)

    with pytest.raises(ValueError, match="(?i)scientific|configuration|mapping|site"):
        _relocate(
            relocation_fixture,
            source_config=source_config,
            destination_config=destination_config,
        )

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)


@pytest.mark.parametrize("git_commit", ("", "a" * 39, "g" * 40, "A" * 40))
def test_relocation_rejects_noncanonical_git_commit_before_hashing_or_staging(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
    git_commit: str,
) -> None:
    """Receipt provenance accepts exactly a forty-character lowercase hex revision."""
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    destination_parent_before = _directory_inventory(
        relocation_fixture.destination_cache.parent
    )

    def forbid_streamed_hash(*_args: object, **_kwargs: object) -> str:
        """Malformed provenance must fail before source or component hashing."""
        raise AssertionError("git validation must precede every streamed hash")

    def forbid_staging(*_args: object, **_kwargs: object) -> Path:
        """Malformed provenance must fail before allocating a staging directory."""
        raise AssertionError("git validation must precede staging allocation")

    monkeypatch.setattr(lfp_summary_models, "sha256_file_content", forbid_streamed_hash)
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    monkeypatch.setattr(module, "_create_staging_directory", forbid_staging)

    with pytest.raises(ValueError, match="(?i)git|commit|hex|revision"):
        _relocate(relocation_fixture, git_commit=git_commit)

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)
    assert _directory_inventory(relocation_fixture.destination_cache.parent) == destination_parent_before


@pytest.mark.parametrize("location", ("outside_root", "wrong_relative_path"))
def test_relocation_requires_one_declared_session_root_mapping(
    relocation_fixture: _RelocationFixture,
    location: str,
) -> None:
    """All destination inputs must use the one root substitution exactly."""
    if location == "outside_root":
        destination_path = relocation_fixture.cluster_root.parent / "unmapped" / "trials.csv"
    else:
        destination_path = (
            relocation_fixture.cluster_root / "processed" / "wrong-relative-trials.csv"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("choice_time\n1.0\n", encoding="ascii")
    unmapped_config = replace(
        relocation_fixture.destination_config,
        trial_table_path=destination_path,
    )

    with pytest.raises(ValueError, match="mapping|root|containment|session"):
        _relocate(relocation_fixture, destination_config=unmapped_config)

    assert not relocation_fixture.destination_cache.exists()


@pytest.mark.parametrize("root_name", ("local_session_root", "cluster_session_root"))
def test_relocation_rejects_wrong_declared_session_roots(
    relocation_fixture: _RelocationFixture,
    root_name: str,
) -> None:
    """The request must use the exact roots that define its path substitution."""
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    wrong_root = relocation_fixture.cluster_root.parent / f"wrong-{root_name}"
    wrong_root.mkdir()
    request_kwargs = {root_name: wrong_root}

    with pytest.raises(ValueError, match="(?i)root|mapping|session|containment"):
        _relocate(relocation_fixture, **request_kwargs)

    assert not relocation_fixture.destination_cache.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)


@pytest.mark.parametrize(
    "kind",
    (
        "nested_destination",
        "symlinked_local_root",
        "symlinked_cluster_root",
        "symlinked_cluster_processed_write_parent",
        "symlinked_local_processed_source_parent",
        "symlinked_input_parent",
    ),
)
def test_relocation_rejects_noncanonical_resolved_containment_paths(
    relocation_fixture: _RelocationFixture,
    kind: str,
) -> None:
    """Cache roots, targets, and inputs must be nonsymlink canonical containment paths."""
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    if kind == "nested_destination":
        destination_cache = relocation_fixture.cluster_root / "processed" / "nested" / "cache"
        destination_config = replace(
            relocation_fixture.destination_config,
            output_directory=destination_cache,
        )
        request_kwargs: dict[str, object] = {
            "destination_cache": destination_cache,
            "destination_config": destination_config,
        }
    elif kind == "symlinked_local_root":
        local_link = relocation_fixture.local_root.parent / "local-root-link"
        local_link.symlink_to(relocation_fixture.local_root, target_is_directory=True)
        request_kwargs = {"local_session_root": local_link}
    elif kind == "symlinked_cluster_root":
        cluster_link = relocation_fixture.cluster_root.parent / "cluster-root-link"
        cluster_link.symlink_to(relocation_fixture.cluster_root, target_is_directory=True)
        request_kwargs = {"cluster_session_root": cluster_link}
    elif kind == "symlinked_cluster_processed_write_parent":
        processed_parent = relocation_fixture.cluster_root / "processed"
        target_parent = relocation_fixture.cluster_root / "processed-target"
        processed_parent.rename(target_parent)
        processed_parent.symlink_to(target_parent, target_is_directory=True)
        request_kwargs = {}
    elif kind == "symlinked_local_processed_source_parent":
        processed_parent = relocation_fixture.local_root / "processed"
        target_parent = relocation_fixture.local_root / "processed-target"
        processed_parent.rename(target_parent)
        processed_parent.symlink_to(target_parent, target_is_directory=True)
        request_kwargs = {}
    else:
        input_parent = relocation_fixture.destination_config.sites[0].lfp_path.parent
        target_parent = relocation_fixture.cluster_root / "processed" / "input-parent-target"
        input_parent.rename(target_parent)
        input_parent.symlink_to(target_parent, target_is_directory=True)
        request_kwargs = {}

    with pytest.raises(ValueError, match="(?i)symlink|containment|processed|root|destination|input"):
        _relocate(relocation_fixture, **request_kwargs)

    assert not relocation_fixture.destination_cache.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)


@pytest.mark.parametrize("host", ("local", "cluster"))
def test_relocation_rejects_a_symlinked_derived_open_ephys_sidecar_before_hashing(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
) -> None:
    """Derived Open Ephys sidecars must be contained regular files before hashing."""
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    config = (
        relocation_fixture.source_config
        if host == "local"
        else relocation_fixture.destination_config
    )
    sidecar = config.sites[0].lfp_path.parent / "lfp_preprocessing.json"
    external_target = config.session_path.parent / f"outside-{host}-sidecar.json"
    external_target.write_bytes(sidecar.read_bytes())
    sidecar.unlink()
    sidecar.symlink_to(external_target)

    def forbid_streamed_hash(*_args: object, **_kwargs: object) -> str:
        """Derived-sidecar containment must fail before any content hash starts."""
        raise AssertionError("symlinked sidecars must be rejected before hashing")

    monkeypatch.setattr(lfp_summary_models, "sha256_file_content", forbid_streamed_hash)

    with pytest.raises(ValueError, match="(?i)sidecar|symlink|source|containment"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)


@pytest.mark.parametrize("difference", ("size", "digest", "semantics"))
def test_relocation_rejects_source_size_digest_or_value_semantics_mismatch(
    relocation_fixture: _RelocationFixture,
    difference: str,
) -> None:
    """Every source must have equal bytes and fixed Open Ephys value semantics."""
    if difference == "size":
        destination_path = next(iter(relocation_fixture.local_to_cluster_sources.values()))
        destination_path.write_bytes(b"short")
    elif difference == "digest":
        destination_path = next(iter(relocation_fixture.local_to_cluster_sources.values()))
        destination_path.write_bytes(b"same-size-but-different")
        source_path = next(iter(relocation_fixture.local_to_cluster_sources))
        source_size = source_path.stat().st_size
        destination_path.write_bytes((b"x" * source_size))
    else:
        source_entry = relocation_fixture.source_manifest["components"]["power"]
        lfp_path = relocation_fixture.source_config.sites[0].lfp_path.resolve()
        source_entry["source_fingerprints"][str(lfp_path)]["value_semantics"] = "wrong"
        (relocation_fixture.source_cache / "manifest.json").write_text(
            json.dumps(relocation_fixture.source_manifest, sort_keys=True),
            encoding="ascii",
        )

    with pytest.raises(ValueError, match="source|digest|size|semantics|stale"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()


@pytest.mark.parametrize(
    "kind",
    ("existing_destination", "symlink_destination", "outside_processed"),
)
def test_relocation_rejects_unsafe_destination_paths(
    relocation_fixture: _RelocationFixture,
    kind: str,
) -> None:
    """Final destinations must be absent, nonsymlink direct processed children."""
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    linked_target: Path | None = None
    destination_cache = relocation_fixture.destination_cache
    destination_config = relocation_fixture.destination_config
    if kind == "existing_destination":
        destination_cache.mkdir(parents=True)
    elif kind == "symlink_destination":
        destination_cache.parent.mkdir(parents=True, exist_ok=True)
        linked_target = relocation_fixture.cluster_root / "unsafe-target"
        linked_target.mkdir()
        destination_cache.symlink_to(linked_target, target_is_directory=True)
    else:
        destination_cache = relocation_fixture.cluster_root / "outside" / "corrected-cache"
        destination_config = replace(
            destination_config,
            output_directory=destination_cache,
        )
    destination_before = _path_snapshot(destination_cache)
    linked_target_before = (
        None if linked_target is None else _path_snapshot(linked_target)
    )

    with pytest.raises((FileExistsError, ValueError), match="(?i)destination|symlink|processed|containment|exists"):
        _relocate(
            relocation_fixture,
            destination_cache=destination_cache,
            destination_config=destination_config,
        )

    assert _path_snapshot(destination_cache) == destination_before
    if linked_target is not None:
        assert _path_snapshot(linked_target) == linked_target_before
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)


@pytest.mark.parametrize("member", ("spike_phase.npz", "work", "unexpected.txt"))
def test_relocation_rejects_source_cache_members_outside_two_completed_components(
    relocation_fixture: _RelocationFixture,
    member: str,
) -> None:
    """Source cache may contain exactly manifest, Power, and Synchrony files."""
    member_path = relocation_fixture.source_cache / member
    if member == "work":
        member_path.mkdir()
    else:
        member_path.write_text("unexpected", encoding="ascii")

    with pytest.raises(ValueError, match="Spike|spike|work|unexpected|member"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()


def test_relocation_validates_original_npz_schemas_before_creating_destination(
    relocation_fixture: _RelocationFixture,
) -> None:
    """A corrupt saved component blocks byte copying and manifest publication."""
    np.savez(relocation_fixture.source_cache / "synchrony.npz", wrong=np.array((1.0,)))

    with pytest.raises(ValueError, match="array|schema|values|required|component"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()


@pytest.mark.parametrize(
    "kind",
    (
        "component_snapshot",
        "component_fingerprint",
        "component_source_fingerprints",
        "component_value_semantics",
        "component_state",
        "component_filename",
        "manifest_only_spike",
        "manifest_session_id",
        "manifest_schema_version",
        "top_configuration",
    ),
)
def test_relocation_requires_publicly_compatible_source_components_and_manifest(
    relocation_fixture: _RelocationFixture,
    kind: str,
) -> None:
    """Relocation rejects stale/incomplete producer identities before rebinding."""
    for component in ("power", "synchrony"):
        assert assess_component_status(
            relocation_fixture.source_cache,
            component,
            relocation_fixture.source_config,
            relocation_fixture.source_manifest,
        ).status == "compatible"
    changed_source_manifest = deepcopy(relocation_fixture.source_manifest)
    if kind == "component_snapshot":
        changed_source_manifest["components"]["power"]["configuration_snapshot"][
            "power"
        ]["notch_quality_factor"] = 17.0
    elif kind == "component_fingerprint":
        changed_source_manifest["components"]["power"][
            "configuration_fingerprint"
        ] = "wrong"
    elif kind == "component_source_fingerprints":
        changed_source_manifest["components"]["synchrony"]["source_fingerprints"] = {}
    elif kind == "component_value_semantics":
        changed_source_manifest["components"]["power"]["source_value_semantics"] = {
            "PFC": "wrong"
        }
    elif kind == "component_state":
        changed_source_manifest["components"]["power"]["state"] = "running"
    elif kind == "component_filename":
        changed_source_manifest["components"]["power"]["file_name"] = "other.npz"
    elif kind == "manifest_only_spike":
        changed_source_manifest["components"]["spike_phase"] = deepcopy(
            changed_source_manifest["components"]["power"]
        )
        changed_source_manifest["components"]["spike_phase"]["file_name"] = "spike_phase.npz"
    elif kind == "manifest_session_id":
        changed_source_manifest["session_id"] = "other-session"
    elif kind == "manifest_schema_version":
        changed_source_manifest["schema_version"] = "2"
    else:
        changed_source_manifest["configuration"]["power"]["notch_quality_factor"] = 17.0
    (relocation_fixture.source_cache / "manifest.json").write_text(
        json.dumps(changed_source_manifest, sort_keys=True), encoding="ascii"
    )

    with pytest.raises(ValueError, match="(?i)source|manifest|component|spike|configuration|schema|state"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()


@pytest.mark.parametrize("kind", ("source_cache", "source_output"))
def test_relocation_requires_source_cache_to_match_the_producer_configuration(
    relocation_fixture: _RelocationFixture,
    kind: str,
) -> None:
    """The requested local cache cannot be substituted independently of its config."""
    if kind == "source_cache":
        source_cache = relocation_fixture.local_root / "processed" / "other-cache"
        source_config = relocation_fixture.source_config
    else:
        source_cache = relocation_fixture.source_cache
        source_config = replace(
            relocation_fixture.source_config,
            output_directory=relocation_fixture.local_root / "processed" / "other-cache",
        )

    with pytest.raises(ValueError, match="(?i)source|cache|output"):
        _relocate(
            relocation_fixture,
            source_cache=source_cache,
            source_config=source_config,
        )

    assert not relocation_fixture.destination_cache.exists()


def test_relocation_requires_destination_argument_to_match_active_output_directory(
    relocation_fixture: _RelocationFixture,
) -> None:
    """A request cannot write somewhere other than its active cluster cache config."""
    different_destination = (
        relocation_fixture.cluster_root / "processed" / "other-corrected-cache"
    )

    with pytest.raises(ValueError, match="destination|output.*directory|output"):
        _relocate(relocation_fixture, destination_cache=different_destination)

    assert not different_destination.exists()
    assert not relocation_fixture.destination_cache.exists()


@pytest.mark.parametrize(
    "kind",
    (
        "existing",
        "symlink",
        "inside_destination",
        "inside_source",
        "parent_to_source",
        "parent_to_legacy",
        "parent_to_external",
    ),
)
def test_relocation_requires_an_absent_external_nonsymlink_receipt_path(
    relocation_fixture: _RelocationFixture,
    kind: str,
) -> None:
    """Receipts are external evidence, never overwrite targets, and never cache members."""
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    linked_target: Path | None = None
    linked_parent: Path | None = None
    redirected_parent_target: Path | None = None
    if kind == "existing":
        receipt_path = relocation_fixture.receipt_path
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text("prior receipt", encoding="ascii")
    elif kind == "symlink":
        receipt_path = relocation_fixture.receipt_path
        receipt_path.parent.mkdir(parents=True)
        linked_target = relocation_fixture.cluster_root / "receipt-target.json"
        linked_target.write_text("protected", encoding="ascii")
        receipt_path.symlink_to(linked_target)
    elif kind == "inside_destination":
        receipt_path = relocation_fixture.destination_cache / "receipt.json"
    elif kind == "inside_source":
        receipt_path = relocation_fixture.source_cache / "receipt.json"
    else:
        linked_parent = relocation_fixture.cluster_root / f"{kind}-receipt-parent"
        if kind == "parent_to_source":
            redirected_parent_target = relocation_fixture.source_cache
        elif kind == "parent_to_legacy":
            redirected_parent_target = relocation_fixture.legacy_cache
        else:
            redirected_parent_target = relocation_fixture.cluster_root / "external-parent"
            redirected_parent_target.mkdir()
            (redirected_parent_target / "external-sentinel.txt").write_bytes(
                b"protected external receipt parent"
            )
        linked_parent.symlink_to(redirected_parent_target, target_is_directory=True)
        receipt_path = linked_parent / "receipt.json"
    receipt_before = _path_snapshot(receipt_path)
    linked_parent_before = (
        None if linked_parent is None else _path_snapshot(linked_parent)
    )
    linked_target_before = (
        None if linked_target is None else _path_snapshot(linked_target)
    )
    redirected_parent_target_before = (
        None
        if redirected_parent_target is None
        else _path_snapshot(redirected_parent_target)
    )

    with pytest.raises((FileExistsError, ValueError), match="(?i)receipt|external|symlink|exists|cache"):
        _relocate(relocation_fixture, receipt_path=receipt_path)

    assert not relocation_fixture.destination_cache.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)
    assert _path_snapshot(receipt_path) == receipt_before
    if linked_parent is not None:
        assert _path_snapshot(linked_parent) == linked_parent_before
    if redirected_parent_target is not None:
        assert _path_snapshot(redirected_parent_target) == redirected_parent_target_before
    if linked_target is not None:
        assert _path_snapshot(linked_target) == linked_target_before


def test_relocation_rejects_a_symlinked_source_component(
    relocation_fixture: _RelocationFixture,
) -> None:
    """The source component itself must be a regular cache file, not a link."""
    source_component = relocation_fixture.source_cache / "power.npz"
    target_component = relocation_fixture.local_root / "power-target.npz"
    source_component.rename(target_component)
    source_component.symlink_to(target_component)

    with pytest.raises(ValueError, match="source|symlink|regular|component"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()


def test_relocation_python_module_entrypoint_records_the_exact_invocation(
    relocation_fixture: _RelocationFixture,
) -> None:
    """``python -m`` accepts only explicit paths and records its exact command."""
    arguments = [
        "--source-cache-directory",
        str(relocation_fixture.source_cache),
        "--destination-cache-directory",
        str(relocation_fixture.destination_cache),
        "--source-config-json",
        str(relocation_fixture.source_config_json_path),
        "--destination-config-json",
        str(relocation_fixture.destination_config_json_path),
        "--local-session-root",
        str(relocation_fixture.local_root),
        "--cluster-session-root",
        str(relocation_fixture.cluster_root),
        "--receipt-path",
        str(relocation_fixture.receipt_path),
        "--git-commit",
        "a" * 40,
    ]
    command = [
        sys.executable,
        "-m",
        "src.neural_analysis.lfp_summary_cache_relocation",
        *arguments,
    ]

    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    receipt = json.loads(relocation_fixture.receipt_path.read_text(encoding="ascii"))
    assert receipt["git_commit"] == "a" * 40
    assert receipt["command"] == command
    assert (relocation_fixture.destination_cache / "power.npz").is_file()
    assert (relocation_fixture.destination_cache / "synchrony.npz").is_file()


def test_relocation_publishes_one_complete_nonsymlink_sibling_stage_with_os_replace(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One atomic directory rename publishes the completed direct-sibling stage."""
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    publish_staged_cache = getattr(module, "_publish_staged_cache")
    real_replace = os.replace
    published_stages: list[Path] = []
    replace_calls: list[tuple[Path, Path]] = []

    def observing_replace(source_path: Path | str, destination_path: Path | str) -> None:
        """Record every replacement while retaining the real atomic primitive."""
        replace_calls.append((Path(source_path), Path(destination_path)))
        real_replace(source_path, destination_path)

    def inspect_then_publish(staging_directory: Path, destination_directory: Path) -> None:
        """Require a fully written nonsymlink stage before its one publication."""
        assert staging_directory.parent == destination_directory.parent
        assert staging_directory != destination_directory
        assert staging_directory.is_dir()
        assert not staging_directory.is_symlink()
        assert sorted(path.name for path in staging_directory.iterdir()) == [
            "manifest.json",
            "power.npz",
            "synchrony.npz",
        ]
        published_stages.append(staging_directory)
        publish_staged_cache(staging_directory, destination_directory)

    monkeypatch.setattr(module.os, "replace", observing_replace)
    monkeypatch.setattr(module, "_publish_staged_cache", inspect_then_publish)

    _relocate(relocation_fixture)

    assert len(published_stages) == 1
    final_replacements = [
        (source_path, destination_path)
        for source_path, destination_path in replace_calls
        if destination_path == relocation_fixture.destination_cache
    ]
    assert final_replacements == [
        (published_stages[0], relocation_fixture.destination_cache)
    ]


def test_prepublication_interruption_keeps_final_source_and_legacy_caches_unchanged(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An injected publish interruption never exposes a partial destination cache."""
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    receipt_parent = relocation_fixture.receipt_path.parent
    receipt_parent.mkdir()
    (receipt_parent / "receipt-parent-sentinel.txt").write_bytes(
        b"protected receipt parent"
    )
    receipt_parent_before = _directory_inventory(receipt_parent)
    destination_parent_before = set(
        relocation_fixture.destination_cache.parent.iterdir()
    )
    attempted_stages: list[Path] = []

    def fail_publish(staging_directory: Path, destination_directory: Path) -> None:
        """Raise immediately before atomic directory publication in the test seam."""
        assert staging_directory.parent == destination_directory.parent
        attempted_stages.append(staging_directory)
        raise OSError("injected prepublication interruption")

    monkeypatch.setattr(module, "_publish_staged_cache", fail_publish)

    with pytest.raises(OSError, match="injected prepublication"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)
    assert _directory_inventory(receipt_parent) == receipt_parent_before
    assert set(relocation_fixture.destination_cache.parent.iterdir()) == destination_parent_before
    assert attempted_stages
    assert all(not stage.exists() for stage in attempted_stages)
    assert not any(
        stage.name in {path.name for path in attempted_stages}
        for stage in relocation_fixture.cluster_root.rglob("*")
    )


def test_relocation_prepares_receipt_before_atomic_cache_publication(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receipt bytes are prepared and validated before the cache becomes public."""
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    write_ascii_json = getattr(module, "_write_ascii_json")
    publish_staged_cache = getattr(module, "_publish_staged_cache")
    real_replace = module.os.replace
    events: list[str] = []
    receipt_temporary_paths: list[Path] = []

    def record_receipt_preparation(path: Path, payload: object) -> object:
        """Record only the external receipt serialization through existing I/O."""
        if isinstance(payload, dict) and "source_producer_manifest" in payload:
            result = write_ascii_json(path, payload)
            assert path != relocation_fixture.receipt_path
            assert path.is_file()
            assert path.read_bytes().isascii()
            assert not path.is_relative_to(relocation_fixture.source_cache)
            assert not path.is_relative_to(relocation_fixture.destination_cache)
            receipt_temporary_paths.append(path)
            events.append("receipt-prepared")
            return result
        return write_ascii_json(path, payload)

    def require_prepared_receipt(staging_directory: Path, destination_directory: Path) -> None:
        """Reject publication if no durable temporary receipt was prepared first."""
        assert events == ["receipt-prepared"]
        publish_staged_cache(staging_directory, destination_directory)
        events.append("cache-published")

    def record_receipt_commit(source_path: Path | str, destination_path: Path | str) -> None:
        """Record the final external receipt replacement after cache publication."""
        if Path(destination_path) == relocation_fixture.receipt_path:
            assert relocation_fixture.destination_cache.is_dir()
            assert events == ["receipt-prepared", "cache-published"]
            real_replace(source_path, destination_path)
            events.append("receipt-committed")
            return
        real_replace(source_path, destination_path)

    monkeypatch.setattr(module, "_write_ascii_json", record_receipt_preparation)
    monkeypatch.setattr(module, "_publish_staged_cache", require_prepared_receipt)
    monkeypatch.setattr(module.os, "replace", record_receipt_commit)

    _relocate(relocation_fixture)

    assert events == ["receipt-prepared", "cache-published", "receipt-committed"]
    assert all(not path.exists() for path in receipt_temporary_paths)
    assert relocation_fixture.destination_cache.is_dir()
    assert relocation_fixture.receipt_path.is_file()


@pytest.mark.parametrize(
    "failure_point",
    ("receipt_parent", "receipt_preparation", "receipt_commit"),
)
def test_relocation_receipt_transaction_failures_leave_no_public_artifacts(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """Receipt failures clean only their new transaction artifacts and destination."""
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    write_ascii_json = getattr(module, "_write_ascii_json")
    real_replace = module.os.replace
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    receipt_parent = relocation_fixture.receipt_path.parent
    if failure_point != "receipt_parent":
        receipt_parent.mkdir()
        (receipt_parent / "receipt-parent-sentinel.txt").write_bytes(
            b"protected receipt parent"
        )
    receipt_parent_before = (
        None if not receipt_parent.exists() else _directory_inventory(receipt_parent)
    )
    local_root_before = _directory_inventory(relocation_fixture.local_root)
    cluster_root_before = _directory_inventory(relocation_fixture.cluster_root)
    receipt_temporary_paths: list[Path] = []

    def record_or_fail_receipt_preparation(path: Path, payload: object) -> object:
        """Observe only external receipt serialization through the existing writer."""
        if isinstance(payload, dict) and "source_producer_manifest" in payload:
            if failure_point == "receipt_preparation":
                raise OSError("injected receipt preparation failure")
            result = write_ascii_json(path, payload)
            assert path != relocation_fixture.receipt_path
            assert path.is_file()
            assert path.read_bytes().isascii()
            assert not path.is_relative_to(relocation_fixture.source_cache)
            assert not path.is_relative_to(relocation_fixture.destination_cache)
            receipt_temporary_paths.append(path)
            return result
        return write_ascii_json(path, payload)

    monkeypatch.setattr(module, "_write_ascii_json", record_or_fail_receipt_preparation)

    if failure_point == "receipt_parent":
        original_mkdir = module.Path.mkdir

        def fail_receipt_parent_mkdir(path: Path, *args: object, **kwargs: object) -> None:
            """Inject only the requested external receipt-parent creation failure."""
            if path == receipt_parent:
                raise OSError("injected receipt-parent creation failure")
            original_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(module.Path, "mkdir", fail_receipt_parent_mkdir)
    elif failure_point == "receipt_commit":

        def fail_receipt_commit(source_path: Path | str, destination_path: Path | str) -> None:
            """Abort only the final external receipt replacement primitive."""
            if Path(destination_path) == relocation_fixture.receipt_path:
                assert relocation_fixture.destination_cache.is_dir()
                raise OSError("injected receipt commit failure")
            real_replace(source_path, destination_path)

        monkeypatch.setattr(module.os, "replace", fail_receipt_commit)

    with pytest.raises(OSError, match="injected receipt"):
        _relocate(relocation_fixture)

    assert not relocation_fixture.destination_cache.exists()
    assert not relocation_fixture.receipt_path.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)
    assert _directory_inventory(relocation_fixture.local_root) == local_root_before
    assert _directory_inventory(relocation_fixture.cluster_root) == cluster_root_before
    if receipt_parent_before is None:
        assert not receipt_parent.exists()
    else:
        assert _directory_inventory(receipt_parent) == receipt_parent_before
    assert not any(
        path.name.startswith(f".{relocation_fixture.destination_cache.name}-stage-")
        for path in relocation_fixture.destination_cache.parent.iterdir()
    )
    if receipt_parent.exists():
        assert not any(
            path.name.startswith(f".{relocation_fixture.receipt_path.name}-")
            for path in receipt_parent.iterdir()
        )
    assert all(not path.exists() for path in receipt_temporary_paths)


def test_relocation_receipt_commit_race_preserves_concurrent_destination_change(
    relocation_fixture: _RelocationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback never removes a destination changed after this command published it.

    The injected receipt-commit failure occurs after the cache directory rename.
    A concurrent actor atomically replaces one same-name regular component before
    the failure is raised. Relocation must leave that destination cache intact,
    while still removing only its private receipt temporary file and staging
    directory. Synthetic source and protected legacy cache bytes remain unchanged.
    """
    module = importlib.import_module("src.neural_analysis.lfp_summary_cache_relocation")
    write_ascii_json = getattr(module, "_write_ascii_json")
    real_replace = module.os.replace
    source_before, legacy_before = _protected_inventories(relocation_fixture)
    receipt_parent = relocation_fixture.receipt_path.parent
    receipt_parent.mkdir()
    (receipt_parent / "receipt-parent-sentinel.txt").write_bytes(
        b"protected receipt parent"
    )
    receipt_parent_before = _directory_inventory(receipt_parent)
    concurrent_power_bytes = b"concurrent destination power bytes"
    receipt_temporary_paths: list[Path] = []

    def record_receipt_preparation(path: Path, payload: object) -> object:
        """Record the exact private receipt file after its real ASCII write."""
        if isinstance(payload, dict) and "source_producer_manifest" in payload:
            result = write_ascii_json(path, payload)
            assert path.is_file()
            receipt_temporary_paths.append(path)
            return result
        return write_ascii_json(path, payload)

    def replace_component_then_fail_receipt(
        source_path: Path | str,
        destination_path: Path | str,
    ) -> None:
        """Model a concurrent same-name component replacement before receipt failure."""
        if Path(destination_path) == relocation_fixture.receipt_path:
            assert relocation_fixture.destination_cache.is_dir()
            replacement = relocation_fixture.destination_cache / ".concurrent-power.npz"
            replacement.write_bytes(concurrent_power_bytes)
            real_replace(replacement, relocation_fixture.destination_cache / "power.npz")
            raise OSError("injected receipt commit race failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(module, "_write_ascii_json", record_receipt_preparation)
    monkeypatch.setattr(module.os, "replace", replace_component_then_fail_receipt)

    with pytest.raises(OSError, match="injected receipt commit race"):
        _relocate(relocation_fixture)

    assert relocation_fixture.destination_cache.is_dir()
    assert (relocation_fixture.destination_cache / "power.npz").read_bytes() == concurrent_power_bytes
    assert (relocation_fixture.destination_cache / "synchrony.npz").is_file()
    assert (relocation_fixture.destination_cache / "manifest.json").is_file()
    assert not relocation_fixture.receipt_path.exists()
    assert _protected_inventories(relocation_fixture) == (source_before, legacy_before)
    assert _directory_inventory(receipt_parent) == receipt_parent_before
    assert not any(
        path.name.startswith(f".{relocation_fixture.destination_cache.name}-stage-")
        for path in relocation_fixture.destination_cache.parent.iterdir()
    )
    assert all(not path.exists() for path in receipt_temporary_paths)
