"""Dependency-injected orchestration for offline LFP summary cache components.

Numerical modules own array construction and cache I/O owns atomic replacement.
This module only orders preparation, payload construction, manifest merging, and
framework-independent progress reporting. It never imports Streamlit.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    ProgressEvent,
    canonical_config_json,
    component_fingerprint,
    fingerprint_source_files,
    validate_lfp_summary_config,
)


ProgressCallback = Callable[[ProgressEvent], None]
PreparePower = Callable[[LFPSummaryConfig], object]
PreparePhase = Callable[[LFPSummaryConfig], object]
PrepareSpike = Callable[[LFPSummaryConfig, object], object]
BuildPowerPayload = Callable[[LFPSummaryConfig, object], "ComponentPayload"]
BuildSynchronyPayload = Callable[[LFPSummaryConfig, object], "ComponentPayload"]
BuildSpikePayload = Callable[[LFPSummaryConfig, object, object], "ComponentPayload"]
LoadManifest = Callable[[Path], dict[str, object]]
WriteComponent = Callable[[Path, str, dict[str, np.ndarray], dict[str, object]], None]


@dataclass(frozen=True)
class PipelineDependencies:
    """Injected preparation, numerical-payload, and cache-transaction seams.

    Every preparation function returns an opaque in-memory object. Payload
    builders own numerical axes, units, and NaN semantics. ``write_component``
    has the same ``(cache_directory, component, arrays, manifest)`` signature
    as :func:`lfp_summary_io.write_component_transaction` and owns validation
    plus manifest-last atomic replacement.
    """

    prepare_power: PreparePower
    prepare_phase: PreparePhase
    prepare_spike: PrepareSpike
    build_power_payload: BuildPowerPayload
    build_synchrony_payload: BuildSynchronyPayload
    build_spike_phase_payload: BuildSpikePayload
    load_manifest: LoadManifest
    write_component: WriteComponent


@dataclass(frozen=True)
class ComponentPayload:
    """Numerical arrays and one component-specific manifest entry.

    ``arrays`` maps names to numeric or fixed-Unicode NumPy arrays with axes,
    physical units, and missing-value semantics defined by ``manifest_entry``.
    The pipeline adds transaction metadata but does not transform or validate
    numerical arrays.
    """

    arrays: dict[str, np.ndarray]
    manifest_entry: dict[str, object]


@dataclass(frozen=True)
class ComponentRunResult:
    """One component completion or failure outcome.

    ``component`` is power, synchrony, or spike_phase. ``state`` is complete
    or failed. ``stage`` identifies the completed or failed orchestration step;
    ``error`` is ``None`` on success. ``manifest`` is the proposed committed
    mapping on success and ``None`` after a pre-commit failure.
    """

    component: str
    state: str
    stage: str
    error: str | None
    manifest: dict[str, object] | None


@dataclass(frozen=True)
class ComputeAllResult:
    """Sequential component outcomes in attempted Power, Synchrony, Spike order."""

    component_results: tuple[ComponentRunResult, ...]


_RECOVERABLE_ERRORS = (ArithmeticError, KeyError, OSError, RuntimeError, ValueError)


def compute_power_component(
    config: LFPSummaryConfig,
    dependencies: PipelineDependencies,
    progress_callback: ProgressCallback | None = None,
) -> ComponentRunResult:
    """Prepare, build, and atomically commit only the Power component.

    ``config`` is a validated immutable summary configuration. Dependencies
    provide opaque preparation and numeric payloads. Progress records carry no
    data arrays. The returned result has no physical units; its payload arrays
    retain their component-defined units and axes.
    """

    validate_lfp_summary_config(config)
    _emit(progress_callback, "power", 0, "prepare_power")
    try:
        prepared = dependencies.prepare_power(config)
    except _RECOVERABLE_ERRORS as error:
        return _failed("power", "prepare_power", error, progress_callback, 0)
    _emit(progress_callback, "power", 1, "payload_power")
    try:
        payload = dependencies.build_power_payload(config, prepared)
    except _RECOVERABLE_ERRORS as error:
        return _failed("power", "payload_power", error, progress_callback, 1)
    return _commit_component("power", config, dependencies, payload, progress_callback)


def compute_synchrony_component(
    config: LFPSummaryConfig,
    dependencies: PipelineDependencies,
    progress_callback: ProgressCallback | None = None,
    *,
    prepared_phase: object | None = None,
) -> ComponentRunResult:
    """Prepare or reuse phase data, then commit only Synchrony.

    ``prepared_phase`` is an optional opaque shared transform product. When it
    is supplied, the phase preparation seam is not called. All transforms,
    interpolation, axes, units, and missingness remain owned by dependencies.
    """

    validate_lfp_summary_config(config)
    phase = prepared_phase
    if phase is None:
        _emit(progress_callback, "synchrony", 0, "prepare_phase")
        try:
            phase = dependencies.prepare_phase(config)
        except _RECOVERABLE_ERRORS as error:
            return _failed("synchrony", "prepare_phase", error, progress_callback, 0)
    _emit(progress_callback, "synchrony", 1, "payload_synchrony")
    try:
        payload = dependencies.build_synchrony_payload(config, phase)
    except _RECOVERABLE_ERRORS as error:
        return _failed("synchrony", "payload_synchrony", error, progress_callback, 1)
    return _commit_component("synchrony", config, dependencies, payload, progress_callback)


def compute_spike_phase_component(
    config: LFPSummaryConfig,
    dependencies: PipelineDependencies,
    progress_callback: ProgressCallback | None = None,
    *,
    prepared_phase: object | None = None,
) -> ComponentRunResult:
    """Prepare or reuse phase and spikes, then commit only Spike phase.

    The optional opaque ``prepared_phase`` is passed unchanged to both spike
    preparation and payload construction. Spike arrays and phase axes are not
    reshaped, filtered, or otherwise changed by this orchestration layer.
    """

    validate_lfp_summary_config(config)
    phase = prepared_phase
    if phase is None:
        _emit(progress_callback, "spike_phase", 0, "prepare_phase")
        try:
            phase = dependencies.prepare_phase(config)
        except _RECOVERABLE_ERRORS as error:
            return _failed("spike_phase", "prepare_phase", error, progress_callback, 0)
    _emit(progress_callback, "spike_phase", 1, "prepare_spike")
    try:
        spikes = dependencies.prepare_spike(config, phase)
    except _RECOVERABLE_ERRORS as error:
        return _failed("spike_phase", "prepare_spike", error, progress_callback, 1)
    _emit(progress_callback, "spike_phase", 2, "payload_spike_phase")
    try:
        payload = dependencies.build_spike_phase_payload(config, phase, spikes)
    except _RECOVERABLE_ERRORS as error:
        return _failed("spike_phase", "payload_spike_phase", error, progress_callback, 2)
    return _commit_component("spike_phase", config, dependencies, payload, progress_callback)


def compute_all_components(
    config: LFPSummaryConfig,
    dependencies: PipelineDependencies,
    progress_callback: ProgressCallback | None = None,
) -> ComputeAllResult:
    """Compute Power, Synchrony, then Spike phase with one shared phase product.

    Inputs are the immutable configuration and injected seams described by
    :class:`PipelineDependencies`. Results preserve component order. On a
    recoverable failure, later components are not started and prior successful
    manifest-last component transactions remain intact.
    """

    validate_lfp_summary_config(config)
    results: list[ComponentRunResult] = []
    power = compute_power_component(config, dependencies, progress_callback)
    results.append(power)
    if power.state == "failed":
        return ComputeAllResult(tuple(results))
    _emit(progress_callback, "synchrony", 0, "prepare_phase")
    try:
        phase = dependencies.prepare_phase(config)
    except _RECOVERABLE_ERRORS as error:
        results.append(_failed("synchrony", "prepare_phase", error, progress_callback, 0))
        return ComputeAllResult(tuple(results))
    synchrony = compute_synchrony_component(
        config,
        dependencies,
        progress_callback,
        prepared_phase=phase,
    )
    results.append(synchrony)
    if synchrony.state == "failed":
        return ComputeAllResult(tuple(results))
    spike_phase = compute_spike_phase_component(
        config,
        dependencies,
        progress_callback,
        prepared_phase=phase,
    )
    results.append(spike_phase)
    return ComputeAllResult(tuple(results))


def _commit_component(
    component: str,
    config: LFPSummaryConfig,
    dependencies: PipelineDependencies,
    payload: ComponentPayload,
    progress_callback: ProgressCallback | None,
) -> ComponentRunResult:
    """Load, immutably merge, and delegate one manifest-last component write."""

    _emit(progress_callback, component, 2, f"load_manifest_{component}")
    try:
        manifest = dependencies.load_manifest(config.output_directory)
    except _RECOVERABLE_ERRORS as error:
        return _failed(component, f"load_manifest_{component}", error, progress_callback, 2)
    _emit(progress_callback, component, 3, f"write_{component}")
    try:
        updated_manifest = _merge_component_manifest(
            manifest,
            component,
            config,
            payload.manifest_entry,
        )
        dependencies.write_component(
            config.output_directory,
            component,
            payload.arrays,
            updated_manifest,
        )
    except _RECOVERABLE_ERRORS as error:
        return _failed(component, f"write_{component}", error, progress_callback, 3)
    _emit(progress_callback, component, 4, f"complete_{component}")
    return ComponentRunResult(
        component,
        "complete",
        f"complete_{component}",
        None,
        updated_manifest,
    )


def _merge_component_manifest(
    manifest: Mapping[str, object],
    component: str,
    config: LFPSummaryConfig,
    entry: Mapping[str, object],
) -> dict[str, object]:
    """Copy a manifest and replace only one complete component entry."""

    merged = deepcopy(dict(manifest))
    existing_components = merged.get("components", {})
    if not isinstance(existing_components, Mapping):
        raise ValueError("manifest components must be a mapping")
    components = deepcopy(dict(existing_components))
    component_entry = deepcopy(dict(entry))
    component_entry.update(
        {
            "state": "complete",
            "configuration_fingerprint": component_fingerprint(component, config),
            "configuration_snapshot": json.loads(canonical_config_json(config)),
            "source_fingerprints": fingerprint_source_files(config, component),
            "completed_at": _completed_timestamp(),
        }
    )
    components[component] = component_entry
    merged["session_id"] = config.session_id
    merged["components"] = components
    return merged


def _completed_timestamp() -> str:
    """Return a canonical UTC completion timestamp for manifest metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _emit(
    callback: ProgressCallback | None,
    component: str,
    completed_count: int,
    message: str,
) -> None:
    """Send one monotonic framework-independent four-step progress record."""

    if callback is not None:
        callback(ProgressEvent(component, "run", completed_count, 4, message))


def _failed(
    component: str,
    stage: str,
    error: Exception,
    callback: ProgressCallback | None,
    completed_count: int,
) -> ComponentRunResult:
    """Report a recoverable failure without writing a failed manifest entry."""

    _emit(callback, component, completed_count, f"failed_{stage}")
    return ComponentRunResult(component, "failed", stage, str(error), None)
