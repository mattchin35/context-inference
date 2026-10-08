"""WP10 contracts for atomic task-decoding resource measurements."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _resource_usage_module():
    """Import the production measurement owner after pytest collection.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.resource_usage``.
    """
    return importlib.import_module(
        "src.neural_analysis.task_decoding.resource_usage"
    )


def _resource_envelope() -> dict[str, int]:
    """Return one coherent dimension/byte envelope for tracker tests.

    Returns
    -------
    dict[str, int]
        Nonnegative dimensionless counts and tensor allocation bytes.
    """
    return {
        "full_trial_count": 12,
        "tensor_trial_count": 10,
        "pfc_unit_count": 2,
        "hpc_unit_count": 3,
        "time_bin_count": 8,
        "target_count": 2,
        "condition_count": 1,
        "outer_fold_count": 3,
        "inner_fold_count": 3,
        "coefficient_feature_capacity": 5,
        "categorical_fit_count": 144,
        "numerical_fit_count": 144,
        "tensor_allocation_bytes": 3_200,
    }


def _usage(peak_rss: int, user_seconds: float, system_seconds: float) -> SimpleNamespace:
    """Build one injected ``resource.getrusage``-compatible sample.

    Parameters
    ----------
    peak_rss : int
        Platform-native ``ru_maxrss`` value.
    user_seconds, system_seconds : float
        Cumulative process CPU seconds.

    Returns
    -------
    types.SimpleNamespace
        Object exposing ``ru_maxrss``, ``ru_utime``, and ``ru_stime``.
    """
    return SimpleNamespace(
        ru_maxrss=peak_rss,
        ru_utime=user_seconds,
        ru_stime=system_seconds,
    )


def test_peak_rss_normalization_is_explicit_for_linux_and_macos():
    """Linux reports KiB while macOS already reports resident bytes."""
    resource_usage = _resource_usage_module()

    assert resource_usage.normalize_peak_rss_bytes(123, "linux") == 123 * 1024
    assert resource_usage.normalize_peak_rss_bytes(123, "darwin") == 123
    with pytest.raises(ValueError, match="platform"):
        resource_usage.normalize_peak_rss_bytes(123, "win32")
    with pytest.raises(ValueError, match="nonnegative|integer"):
        resource_usage.normalize_peak_rss_bytes(-1, "linux")


def test_tracker_snapshots_are_atomic_and_do_not_double_count_one_process(tmp_path):
    """Repeated snapshots use deltas from one baseline rather than summing snapshots."""
    resource_usage = _resource_usage_module()
    clock_values = iter((10.0, 13.0, 15.0))
    usage_values = iter(
        (
            _usage(100, 1.0, 0.5),
            _usage(200, 2.0, 1.0),
            _usage(150, 3.0, 1.5),
        )
    )
    tracker = resource_usage.ResourceUsageTracker(
        tmp_path,
        _resource_envelope(),
        clock=lambda: next(clock_values),
        usage_reader=lambda: next(usage_values),
        platform_name="linux",
    )
    tracker.record_model_event(
        target_label="current_action",
        target_family="categorical",
        operation="estimator",
        region_configuration="PFC",
        representation="units",
        elapsed_seconds=0.25,
    )
    tracker.finish_target(
        target_label="current_action",
        target_family="categorical",
        total_seconds=2.5,
        requested_fit_count=144,
        valid_outer_cell_count=130,
        invalid_outer_cell_count=14,
    )

    first = tracker.snapshot(status="running", completed_targets=("current_action",))
    second = tracker.snapshot(status="interrupted", completed_targets=("current_action",))

    assert first["wall_time_seconds"] == pytest.approx(3.0)
    assert second["wall_time_seconds"] == pytest.approx(5.0)
    assert second["user_cpu_seconds"] == pytest.approx(2.0)
    assert second["system_cpu_seconds"] == pytest.approx(1.0)
    assert second["peak_rss_bytes"] == 200 * 1024
    assert second["completed_targets"] == ["current_action"]
    measurement = second["target_measurements"]["current_action"]
    assert measurement["timing_available"] is True
    assert measurement["timing_seconds"]["estimator"]["PFC|units"] == pytest.approx(0.25)
    assert measurement["operation_counts"]["estimator"]["PFC|units"] == 1
    assert second["fit_counts"] == {
        "requested": 144,
        "measured_estimator_calls": 1,
        "valid_outer_cells": 130,
        "invalid_outer_cells": 14,
    }
    assert json.loads((tmp_path / "resource_usage.json").read_text()) == second
    assert not any(path.name.startswith(".resource_usage") for path in tmp_path.iterdir())


def test_resume_accumulates_cpu_and_wall_but_takes_maximum_peak_rss(tmp_path):
    """A later process preserves target timing and adds only its own CPU/wall deltas."""
    resource_usage = _resource_usage_module()
    first_clock = iter((10.0, 15.0))
    first_usage = iter((_usage(200, 1.0, 0.5), _usage(250, 3.0, 1.5)))
    first = resource_usage.ResourceUsageTracker(
        tmp_path,
        _resource_envelope(),
        clock=lambda: next(first_clock),
        usage_reader=lambda: next(first_usage),
        platform_name="linux",
    )
    first.finish_target(
        target_label="current_action",
        target_family="categorical",
        total_seconds=4.0,
        requested_fit_count=144,
        valid_outer_cell_count=144,
        invalid_outer_cell_count=0,
    )
    first.snapshot(status="interrupted", completed_targets=("current_action",))

    resumed_clock = iter((20.0, 24.0))
    resumed_usage = iter((_usage(100, 10.0, 5.0), _usage(240, 11.0, 5.5)))
    resumed = resource_usage.ResourceUsageTracker(
        tmp_path,
        _resource_envelope(),
        clock=lambda: next(resumed_clock),
        usage_reader=lambda: next(resumed_usage),
        platform_name="linux",
    )
    payload = resumed.snapshot(status="failed", completed_targets=("current_action",))

    assert payload["wall_time_seconds"] == pytest.approx(9.0)
    assert payload["user_cpu_seconds"] == pytest.approx(3.0)
    assert payload["system_cpu_seconds"] == pytest.approx(1.5)
    assert payload["peak_rss_bytes"] == 250 * 1024
    assert payload["target_measurements"]["current_action"]["total_seconds"] == 4.0


def test_restored_checkpoint_is_explicit_when_historical_timing_is_absent(tmp_path):
    """Resume never fabricates operation timing for a checkpoint without evidence."""
    resource_usage = _resource_usage_module()
    clock_values = iter((1.0, 2.0))
    usage_values = iter((_usage(10, 0.0, 0.0), _usage(20, 0.1, 0.1)))
    tracker = resource_usage.ResourceUsageTracker(
        tmp_path,
        _resource_envelope(),
        clock=lambda: next(clock_values),
        usage_reader=lambda: next(usage_values),
        platform_name="linux",
    )

    tracker.record_restored_target(
        target_label="current_action",
        target_family="categorical",
        requested_fit_count=144,
        valid_outer_cell_count=144,
        invalid_outer_cell_count=0,
    )
    payload = tracker.snapshot(status="running", completed_targets=("current_action",))

    measurement = payload["target_measurements"]["current_action"]
    assert measurement["timing_available"] is False
    assert measurement["total_seconds"] is None
    assert measurement["timing_seconds"] == {}


def test_complete_payload_validation_rejects_partial_or_malformed_evidence(tmp_path):
    """Batch-safe evidence must be complete, measured, and schema-valid."""
    resource_usage = _resource_usage_module()
    clock_values = iter((1.0, 2.0))
    usage_values = iter((_usage(10, 0.0, 0.0), _usage(20, 0.2, 0.1)))
    tracker = resource_usage.ResourceUsageTracker(
        tmp_path,
        _resource_envelope(),
        clock=lambda: next(clock_values),
        usage_reader=lambda: next(usage_values),
        platform_name="linux",
    )
    tracker.record_model_event(
        target_label="current_action",
        target_family="categorical",
        operation="estimator",
        region_configuration="PFC",
        representation="units",
        elapsed_seconds=0.1,
    )
    tracker.finish_target(
        target_label="current_action",
        target_family="categorical",
        total_seconds=0.5,
        requested_fit_count=144,
        valid_outer_cell_count=144,
        invalid_outer_cell_count=0,
    )
    tracker.record_restored_target(
        target_label="relative_doubt",
        target_family="numerical",
        requested_fit_count=144,
        valid_outer_cell_count=0,
        invalid_outer_cell_count=144,
    )
    payload = tracker.snapshot(
        status="complete",
        completed_targets=("current_action", "relative_doubt"),
    )

    assert resource_usage.validate_resource_usage_payload(
        payload,
        require_complete=True,
    )["peak_rss_bytes"] == 20 * 1024
    partial = dict(payload, status="running")
    with pytest.raises(ValueError, match="complete"):
        resource_usage.validate_resource_usage_payload(partial, require_complete=True)
    malformed = dict(payload, peak_rss_bytes=0)
    with pytest.raises(ValueError, match="peak_rss_bytes"):
        resource_usage.validate_resource_usage_payload(malformed, require_complete=True)
    negative_nested_count = copy.deepcopy(payload)
    negative_nested_count["target_measurements"]["current_action"][
        "operation_counts"
    ]["estimator"]["PFC|units"] = -1
    with pytest.raises(ValueError, match="operation|count|nonnegative"):
        resource_usage.validate_resource_usage_payload(
            negative_nested_count,
            require_complete=True,
        )
    inconsistent_aggregate = copy.deepcopy(payload)
    inconsistent_aggregate["fit_counts"]["requested"] += 1
    with pytest.raises(ValueError, match="fit_counts|aggregate|requested"):
        resource_usage.validate_resource_usage_payload(
            inconsistent_aggregate,
            require_complete=True,
        )
