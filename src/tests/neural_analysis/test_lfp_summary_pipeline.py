"""Dependency-injected contract tests for the LFP summary component pipeline."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.neural_analysis import (
    lfp_power_summary,
    lfp_summary_pipeline,
    lfp_synchrony_summary,
    spike_lfp_summary,
)
from src.neural_analysis.lfp_summary_models import ProgressEvent, default_lfp_summary_config
from src.neural_analysis.lfp_summary_preparation import validate_progress_events


def _array_schema() -> dict[str, dict[str, object]]:
    """Return the smallest valid component-array schema for pipeline fakes."""

    return {"value": {"axes": ["item"], "units": "dimensionless"}}


def _component_entry(component: str, *, state: str = "complete") -> dict[str, object]:
    """Return one writer-ready manifest entry without pipeline-owned fingerprints."""

    return {
        "file_name": f"{component}.npz",
        "state": state,
        "array_schema": _array_schema(),
        "configuration_snapshot": {},
        "source_fingerprints": {},
    }


def _empty_manifest(config: object) -> dict[str, object]:
    """Build a minimal manifest whose component entries are merged by the pipeline."""

    return {"session_id": config.session_id, "components": {}}


def _strip_declared_timestamps(value: Any) -> Any:
    """Remove expected clock fields before deterministic manifest comparison."""

    if isinstance(value, dict):
        return {
            key: _strip_declared_timestamps(item)
            for key, item in value.items()
            if key not in {"completed_at", "processing_timestamp"}
        }
    if isinstance(value, list):
        return [_strip_declared_timestamps(item) for item in value]
    return value


def _make_dependencies(
    calls: list[str],
    manifests: list[dict[str, object]],
    *,
    fail_component: str | None = None,
) -> lfp_summary_pipeline.PipelineDependencies:
    """Build deterministic injected seams without raw files or numerical transforms."""

    prepared_power = object()
    prepared_phase = object()
    prepared_spike = object()
    manifest = _empty_manifest(default_lfp_summary_config())

    def prepare_power(config: object) -> object:
        calls.append("prepare_power")
        return prepared_power

    def prepare_phase(config: object) -> object:
        calls.append("prepare_phase")
        return prepared_phase

    def prepare_spike(config: object, phase: object) -> object:
        assert phase is prepared_phase
        calls.append("prepare_spike")
        return prepared_spike

    def payload(component: str, *prepared: object) -> lfp_summary_pipeline.ComponentPayload:
        calls.append(f"payload_{component}")
        return lfp_summary_pipeline.ComponentPayload(
            arrays={"value": np.array([float(len(prepared))])},
            manifest_entry=_component_entry(component),
        )

    def synchrony_payload(config: object, phase: object) -> lfp_summary_pipeline.ComponentPayload:
        assert phase is prepared_phase
        return payload("synchrony", phase)

    def spike_payload(
        config: object,
        phase: object,
        spike: object,
    ) -> lfp_summary_pipeline.ComponentPayload:
        assert phase is prepared_phase
        assert spike is prepared_spike
        return payload("spike_phase", phase, spike)

    def load_manifest(cache_directory: Path) -> dict[str, object]:
        calls.append("load_manifest")
        return deepcopy(manifest)

    def write_component(
        cache_directory: Path,
        component: str,
        arrays: dict[str, np.ndarray],
        updated_manifest: dict[str, object],
    ) -> None:
        calls.append(f"write_{component}")
        if component == fail_component:
            raise OSError(f"{component} write failed")
        manifests.append(deepcopy(updated_manifest))
        manifest.clear()
        manifest.update(deepcopy(updated_manifest))

    return lfp_summary_pipeline.PipelineDependencies(
        prepare_power=prepare_power,
        prepare_phase=prepare_phase,
        prepare_spike=prepare_spike,
        build_power_payload=lambda config, prepared: payload("power", prepared),
        build_synchrony_payload=synchrony_payload,
        build_spike_phase_payload=spike_payload,
        load_manifest=load_manifest,
        write_component=write_component,
    )


def test_each_component_uses_only_its_required_prepare_payload_and_writer_path() -> None:
    """Individual entry points must not prepare or write unrelated components."""

    config = default_lfp_summary_config()
    for component, invoke, expected_calls in (
        (
            "power",
            lambda dependencies: lfp_summary_pipeline.compute_power_component(config, dependencies),
            ["prepare_power", "payload_power", "load_manifest", "write_power"],
        ),
        (
            "synchrony",
            lambda dependencies: lfp_summary_pipeline.compute_synchrony_component(
                config,
                dependencies,
            ),
            ["prepare_phase", "payload_synchrony", "load_manifest", "write_synchrony"],
        ),
        (
            "spike_phase",
            lambda dependencies: lfp_summary_pipeline.compute_spike_phase_component(
                config,
                dependencies,
            ),
            [
                "prepare_phase",
                "prepare_spike",
                "payload_spike_phase",
                "load_manifest",
                "write_spike_phase",
            ],
        ),
    ):
        calls: list[str] = []
        dependencies = _make_dependencies(calls, [])

        result = invoke(dependencies)

        assert result.component == component
        assert result.state == "complete"
        assert calls == expected_calls


def test_compute_all_orders_components_and_reuses_one_phase_preparation_object() -> None:
    """Compute All must run in order and share the same prepared phase identity."""

    config = default_lfp_summary_config()
    calls: list[str] = []
    dependencies = _make_dependencies(calls, [])

    result = lfp_summary_pipeline.compute_all_components(config, dependencies)

    assert [item.component for item in result.component_results] == [
        "power",
        "synchrony",
        "spike_phase",
    ]
    assert calls == [
        "prepare_power",
        "payload_power",
        "load_manifest",
        "write_power",
        "prepare_phase",
        "payload_synchrony",
        "load_manifest",
        "write_synchrony",
        "prepare_spike",
        "payload_spike_phase",
        "load_manifest",
        "write_spike_phase",
    ]


def test_pipeline_progress_is_monotonic_and_invalid_progress_is_rejected() -> None:
    """Pipeline callbacks expose framework-independent monotonic work units."""

    config = default_lfp_summary_config()
    events: list[ProgressEvent] = []
    dependencies = _make_dependencies([], [])

    lfp_summary_pipeline.compute_all_components(
        config,
        dependencies,
        progress_callback=events.append,
    )

    assert events
    validate_progress_events(events)
    assert [event.component for event in events] == sorted(
        (event.component for event in events),
        key=("power", "synchrony", "spike_phase").index,
    )
    with pytest.raises(ValueError, match="monotonic"):
        validate_progress_events(
            [
                ProgressEvent("power", "prepare", 1, 2, "one"),
                ProgressEvent("power", "prepare", 0, 2, "backward"),
            ]
        )


def test_writer_failure_preserves_existing_bytes_reports_stage_and_aborts_compute_all(
    tmp_path: Path,
) -> None:
    """A failed commit must leave old cache bytes intact and stop later components."""

    config = replace(default_lfp_summary_config(), output_directory=tmp_path)
    old_component = tmp_path / "power.npz"
    old_manifest = tmp_path / "manifest.json"
    old_component.write_bytes(b"old-power")
    old_manifest.write_bytes(b'{"old":"manifest"}')
    calls: list[str] = []
    dependencies = _make_dependencies(calls, [], fail_component="power")

    result = lfp_summary_pipeline.compute_all_components(config, dependencies)

    assert result.component_results[0].state == "failed"
    assert result.component_results[0].stage == "write_power"
    assert result.component_results[0].error == "power write failed"
    assert old_component.read_bytes() == b"old-power"
    assert old_manifest.read_bytes() == b'{"old":"manifest"}'
    assert "prepare_phase" not in calls
    assert "prepare_spike" not in calls


def test_component_manifest_merges_preserve_unrelated_entries_and_update_fingerprints() -> None:
    """One component update must retain other entries and write scoped complete metadata."""

    config = default_lfp_summary_config()
    calls: list[str] = []
    manifests: list[dict[str, object]] = []
    dependencies = _make_dependencies(calls, manifests)

    lfp_summary_pipeline.compute_power_component(config, dependencies)
    lfp_summary_pipeline.compute_synchrony_component(config, dependencies)

    final_components = manifests[-1]["components"]
    assert set(final_components) == {"power", "synchrony"}
    assert final_components["power"]["state"] == "complete"
    assert final_components["synchrony"]["state"] == "complete"
    assert final_components["power"]["configuration_fingerprint"] != final_components["synchrony"][
        "configuration_fingerprint"
    ]


def test_pipeline_and_numerical_modules_do_not_import_streamlit() -> None:
    """Numerical orchestration must remain usable outside the Streamlit process."""

    modules = (
        lfp_summary_pipeline,
        lfp_power_summary,
        lfp_synchrony_summary,
        spike_lfp_summary,
    )
    for module in modules:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imported_names = {
            alias.name.split(".")[0]
            for node in imports
            for alias in node.names
        }
        assert "streamlit" not in imported_names


def test_fixed_seed_pipeline_outputs_are_reproducible_except_declared_timestamps() -> None:
    """Repeated runs with fixed configuration preserve arrays and manifest content."""

    config = default_lfp_summary_config()
    first_calls: list[str] = []
    first_manifests: list[dict[str, object]] = []
    second_calls: list[str] = []
    second_manifests: list[dict[str, object]] = []

    lfp_summary_pipeline.compute_all_components(
        config,
        _make_dependencies(first_calls, first_manifests),
    )
    lfp_summary_pipeline.compute_all_components(
        config,
        _make_dependencies(second_calls, second_manifests),
    )

    assert first_calls == second_calls
    assert _strip_declared_timestamps(first_manifests[-1]) == _strip_declared_timestamps(
        second_manifests[-1]
    )
