"""Atomic, resumable resource measurements for task-variable decoding runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any


RESOURCE_USAGE_SCHEMA_VERSION = 1
RESOURCE_USAGE_FILE = "resource_usage.json"
RESOURCE_ENVELOPE_KEYS = frozenset(
    {
        "full_trial_count",
        "tensor_trial_count",
        "pfc_unit_count",
        "hpc_unit_count",
        "time_bin_count",
        "target_count",
        "condition_count",
        "outer_fold_count",
        "inner_fold_count",
        "coefficient_feature_capacity",
        "categorical_fit_count",
        "numerical_fit_count",
        "tensor_allocation_bytes",
    }
)
_STATUSES = frozenset({"running", "failed", "interrupted", "complete"})
_OPERATIONS = frozenset(
    {"split", "regional_transform", "feature_construction", "estimator"}
)
_FIT_COUNT_KEYS = frozenset(
    {
        "requested",
        "measured_estimator_calls",
        "valid_outer_cells",
        "invalid_outer_cells",
    }
)


def _nonnegative_int(value: object, name: str) -> int:
    """Validate one built-in nonnegative integer count or byte size.

    Parameters
    ----------
    value : object
        Candidate scalar.
    name : str
        Human-readable field name used in errors.

    Returns
    -------
    int
        Validated dimensionless count or byte size.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _positive_int(value: object, name: str) -> int:
    """Validate one built-in positive integer count or byte size.

    Parameters
    ----------
    value : object
        Candidate scalar.
    name : str
        Human-readable field name used in errors.

    Returns
    -------
    int
        Validated positive dimensionless count or byte size.
    """
    validated = _nonnegative_int(value, name)
    if validated == 0:
        raise ValueError(f"{name} must be positive.")
    return validated


def _nonnegative_seconds(value: object, name: str) -> float:
    """Validate one finite nonnegative duration or CPU-time value.

    Parameters
    ----------
    value : object
        Candidate numerical value in seconds.
    name : str
        Human-readable field name used in errors.

    Returns
    -------
    float
        Finite nonnegative seconds.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite nonnegative seconds.")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite nonnegative seconds.")
    return converted


def validate_resource_envelope(value: object) -> dict[str, int]:
    """Validate the exact cross-session resource comparison vocabulary.

    Parameters
    ----------
    value : object
        Decoded JSON resource envelope.

    Returns
    -------
    dict[str, int]
        Exact positive dimensions and bytes. Family fit counts may be zero,
        but their sum must be positive.
    """
    if not isinstance(value, Mapping) or set(value) != RESOURCE_ENVELOPE_KEYS:
        raise ValueError("Resource envelope has missing or unexpected fields.")
    envelope: dict[str, int] = {}
    for name in sorted(RESOURCE_ENVELOPE_KEYS):
        validator = (
            _nonnegative_int
            if name in {"categorical_fit_count", "numerical_fit_count"}
            else _positive_int
        )
        envelope[name] = validator(value[name], f"resource_envelope.{name}")
    if envelope["categorical_fit_count"] + envelope["numerical_fit_count"] <= 0:
        raise ValueError("Resource envelope must include at least one planned fit.")
    return envelope


def normalize_peak_rss_bytes(raw_peak_rss: object, platform_name: str) -> int:
    """Convert ``ru_maxrss`` to bytes using the operating-system contract.

    Parameters
    ----------
    raw_peak_rss : object
        Nonnegative built-in integer returned as ``ru_maxrss``. Linux reports
        KiB; macOS reports bytes.
    platform_name : str
        Python platform token, currently ``"linux"`` or ``"darwin"``.

    Returns
    -------
    int
        Process peak resident memory in bytes.
    """
    value = _nonnegative_int(raw_peak_rss, "raw peak RSS")
    normalized_platform = str(platform_name).strip().lower()
    if normalized_platform.startswith("linux"):
        return value * 1024
    if normalized_platform == "darwin":
        return value
    raise ValueError(f"Unsupported platform for ru_maxrss normalization: {platform_name!r}.")


def _validate_string_sequence(value: object, name: str) -> list[str]:
    """Validate one ordered unique sequence of nonempty identifiers.

    Parameters
    ----------
    value : object
        Candidate JSON sequence.
    name : str
        Field label used in validation errors.

    Returns
    -------
    list[str]
        Validated identifiers in declared order.
    """
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain nonempty strings.")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates.")
    return list(value)


def _validate_timing_mapping(value: object, name: str) -> dict[str, float]:
    """Validate one string-keyed mapping of nonnegative seconds.

    Parameters
    ----------
    value : object
        Candidate JSON mapping.
    name : str
        Field label used in validation errors.

    Returns
    -------
    dict[str, float]
        Validated timing values in seconds.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    output: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be nonempty strings.")
        output[key] = _nonnegative_seconds(item, f"{name}.{key}")
    return output


def _validate_target_measurements(value: object) -> dict[str, dict[str, object]]:
    """Validate target-local timing/count evidence without changing its values.

    Parameters
    ----------
    value : object
        Candidate target-label mapping from a resource snapshot.

    Returns
    -------
    dict[str, dict[str, object]]
        JSON-safe deep copy of validated target measurements.
    """
    if not isinstance(value, Mapping):
        raise ValueError("target_measurements must be a JSON object.")
    output: dict[str, dict[str, object]] = {}
    for label, raw in value.items():
        if not isinstance(label, str) or not label.strip() or not isinstance(raw, Mapping):
            raise ValueError("Target measurement labels and records are invalid.")
        required = {
            "target_family",
            "timing_available",
            "total_seconds",
            "timing_seconds",
            "operation_counts",
            "requested_fit_count",
            "valid_outer_cell_count",
            "invalid_outer_cell_count",
        }
        if set(raw) != required:
            raise ValueError(f"Target measurement {label!r} has invalid fields.")
        if raw["target_family"] not in {"categorical", "numerical"}:
            raise ValueError(f"Target measurement {label!r} has an invalid family.")
        if not isinstance(raw["timing_available"], bool):
            raise ValueError(f"Target measurement {label!r} timing flag is invalid.")
        total_seconds = raw["total_seconds"]
        if total_seconds is not None:
            total_seconds = _nonnegative_seconds(
                total_seconds,
                f"target_measurements.{label}.total_seconds",
            )
        for field in (
            "requested_fit_count",
            "valid_outer_cell_count",
            "invalid_outer_cell_count",
        ):
            _nonnegative_int(raw[field], f"target_measurements.{label}.{field}")
        timing_available = raw["timing_available"]
        timing = raw["timing_seconds"]
        counts = raw["operation_counts"]
        if not isinstance(timing, Mapping) or not isinstance(counts, Mapping):
            raise ValueError(f"Target measurement {label!r} operation maps are invalid.")
        if not timing_available:
            if total_seconds is not None or timing or counts:
                raise ValueError(
                    f"Timing-unavailable target measurement {label!r} must not fabricate timing."
                )
            validated_timing: dict[str, object] = {}
            validated_counts: dict[str, object] = {}
        else:
            if set(timing) != _OPERATIONS or set(counts) != _OPERATIONS:
                raise ValueError(
                    f"Target measurement {label!r} operation fields are invalid."
                )
            validated_timing = {
                "split": _nonnegative_seconds(
                    timing["split"],
                    f"target_measurements.{label}.timing_seconds.split",
                )
            }
            validated_counts = {
                "split": _nonnegative_int(
                    counts["split"],
                    f"target_measurements.{label}.operation_counts.split",
                )
            }
            allowed_keys = {
                "regional_transform": {"PFC", "HPC"},
                "feature_construction": {
                    f"{region}|{representation}"
                    for region in ("PFC", "HPC", "PFC+HPC")
                    for representation in ("pca", "units")
                },
                "estimator": {
                    f"{region}|{representation}"
                    for region in ("PFC", "HPC", "PFC+HPC")
                    for representation in ("pca", "units")
                },
            }
            for operation, allowed in allowed_keys.items():
                timing_group = timing[operation]
                count_group = counts[operation]
                if not isinstance(timing_group, Mapping) or not isinstance(
                    count_group, Mapping
                ):
                    raise ValueError(
                        f"Target measurement {label!r} {operation} maps are invalid."
                    )
                if set(timing_group) != set(count_group) or not set(timing_group) <= allowed:
                    raise ValueError(
                        f"Target measurement {label!r} {operation} keys are invalid."
                    )
                validated_timing[operation] = {
                    key: _nonnegative_seconds(
                        timing_group[key],
                        f"target_measurements.{label}.timing_seconds.{operation}.{key}",
                    )
                    for key in timing_group
                }
                validated_counts[operation] = {
                    key: _nonnegative_int(
                        count_group[key],
                        f"target_measurements.{label}.operation_counts.{operation}.{key}",
                    )
                    for key in count_group
                }
        output[label] = {
            "target_family": raw["target_family"],
            "timing_available": timing_available,
            "total_seconds": total_seconds,
            "timing_seconds": validated_timing,
            "operation_counts": validated_counts,
            "requested_fit_count": int(raw["requested_fit_count"]),
            "valid_outer_cell_count": int(raw["valid_outer_cell_count"]),
            "invalid_outer_cell_count": int(raw["invalid_outer_cell_count"]),
        }
    return output


def validate_resource_usage_payload(
    value: object,
    *,
    require_complete: bool = False,
) -> dict[str, object]:
    """Validate one atomic runtime-resource evidence payload.

    Parameters
    ----------
    value : object
        Decoded JSON payload.
    require_complete : bool, default=False
        Whether partial running/failed/interrupted snapshots are forbidden.

    Returns
    -------
    dict[str, object]
        JSON-safe validated payload preserving target and stage ordering.
    """
    required = {
        "schema_version",
        "measurement_method",
        "status",
        "peak_rss_bytes",
        "wall_time_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "cpu_efficiency",
        "resource_envelope",
        "completed_targets",
        "target_measurements",
        "stage_timing_seconds",
        "fit_counts",
        "output_size_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Resource usage payload has missing or unexpected fields.")
    if value["schema_version"] != RESOURCE_USAGE_SCHEMA_VERSION:
        raise ValueError("Resource usage schema version is invalid.")
    if value["measurement_method"] != "resource.getrusage":
        raise ValueError("Resource usage measurement method is invalid.")
    status = value["status"]
    if status not in _STATUSES:
        raise ValueError("Resource usage status is invalid.")
    if require_complete and status != "complete":
        raise ValueError("Resource usage evidence must have complete status.")
    peak_rss_bytes = _positive_int(value["peak_rss_bytes"], "peak_rss_bytes")
    wall_seconds = _nonnegative_seconds(value["wall_time_seconds"], "wall_time_seconds")
    user_seconds = _nonnegative_seconds(value["user_cpu_seconds"], "user_cpu_seconds")
    system_seconds = _nonnegative_seconds(
        value["system_cpu_seconds"], "system_cpu_seconds"
    )
    efficiency = _nonnegative_seconds(value["cpu_efficiency"], "cpu_efficiency")
    completed = _validate_string_sequence(value["completed_targets"], "completed_targets")
    target_measurements = _validate_target_measurements(value["target_measurements"])
    if any(label not in target_measurements for label in completed):
        raise ValueError("Completed targets must have target measurement records.")
    if not isinstance(value["fit_counts"], Mapping) or set(value["fit_counts"]) != (
        _FIT_COUNT_KEYS
    ):
        raise ValueError("Resource usage fit_counts fields are invalid.")
    fit_counts = {
        name: _nonnegative_int(value["fit_counts"][name], f"fit_counts.{name}")
        for name in sorted(_FIT_COUNT_KEYS)
    }
    envelope = validate_resource_envelope(value["resource_envelope"])
    aggregate_fit_counts = {
        "requested": 0,
        "measured_estimator_calls": 0,
        "valid_outer_cells": 0,
        "invalid_outer_cells": 0,
    }
    for measurement in target_measurements.values():
        aggregate_fit_counts["requested"] += int(measurement["requested_fit_count"])
        aggregate_fit_counts["valid_outer_cells"] += int(
            measurement["valid_outer_cell_count"]
        )
        aggregate_fit_counts["invalid_outer_cells"] += int(
            measurement["invalid_outer_cell_count"]
        )
        aggregate_fit_counts["measured_estimator_calls"] += sum(
            int(count)
            for count in measurement["operation_counts"].get("estimator", {}).values()
        )
    if fit_counts != aggregate_fit_counts:
        raise ValueError("Resource usage fit_counts do not match target aggregates.")
    if status == "complete":
        if len(completed) != envelope["target_count"] or set(completed) != set(
            target_measurements
        ):
            raise ValueError(
                "Complete resource usage must cover every enveloped target exactly once."
            )
        if any(
            measurement["timing_available"] and measurement["total_seconds"] is None
            for measurement in target_measurements.values()
        ):
            raise ValueError("Complete target timing records require total_seconds.")
        if fit_counts["requested"] != (
            envelope["categorical_fit_count"] + envelope["numerical_fit_count"]
        ):
            raise ValueError("Complete requested fit_counts disagree with the envelope.")
    return {
        "schema_version": RESOURCE_USAGE_SCHEMA_VERSION,
        "measurement_method": "resource.getrusage",
        "status": status,
        "peak_rss_bytes": peak_rss_bytes,
        "wall_time_seconds": wall_seconds,
        "user_cpu_seconds": user_seconds,
        "system_cpu_seconds": system_seconds,
        "cpu_efficiency": efficiency,
        "resource_envelope": envelope,
        "completed_targets": completed,
        "target_measurements": target_measurements,
        "stage_timing_seconds": _validate_timing_mapping(
            value["stage_timing_seconds"], "stage_timing_seconds"
        ),
        "fit_counts": fit_counts,
        "output_size_bytes": _nonnegative_int(
            value["output_size_bytes"], "output_size_bytes"
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one JSON object through a flushed sibling temporary file.

    Parameters
    ----------
    path : pathlib.Path
        Destination below an existing run directory.
    payload : mapping[str, object]
        JSON-safe resource evidence.

    Returns
    -------
    None
        Replaces ``path`` atomically and removes a temporary after failures.
    """
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _directory_output_size_bytes(run_directory: Path) -> int:
    """Count regular output bytes while excluding self-referential evidence.

    Parameters
    ----------
    run_directory : pathlib.Path
        Current task-decoding run directory.

    Returns
    -------
    int
        Bytes in regular files other than ``resource_usage.json`` and its
        sibling publication temporaries.
    """
    total = 0
    for path in run_directory.rglob("*"):
        if not path.is_file():
            continue
        if path.name == RESOURCE_USAGE_FILE or path.name.startswith(
            f".{RESOURCE_USAGE_FILE}."
        ):
            continue
        total += path.stat().st_size
    return total


class ResourceUsageTracker:
    """Accumulate process measurements across target checkpoints and resumes.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Existing run directory receiving atomic ``resource_usage.json``.
    resource_envelope : mapping[str, object]
        Exact dry-run/session dimensions and tensor bytes.
    clock : callable returning float, default=time.monotonic
        Monotonic wall clock in seconds. Injection supports deterministic tests.
    usage_reader : callable, default=resource.getrusage(RUSAGE_SELF)
        Returns an object with cumulative ``ru_maxrss``, ``ru_utime``, and
        ``ru_stime`` members.
    platform_name : str, default=sys.platform
        Platform token controlling ``ru_maxrss`` byte normalization.
    """

    def __init__(
        self,
        run_directory: Path | str,
        resource_envelope: Mapping[str, object],
        *,
        clock: Callable[[], float] = time.monotonic,
        usage_reader: Callable[[], Any] | None = None,
        platform_name: str = sys.platform,
    ) -> None:
        self.run_directory = Path(run_directory)
        if not self.run_directory.is_dir():
            raise ValueError("Resource tracker run_directory must be an existing directory.")
        self.resource_envelope = validate_resource_envelope(resource_envelope)
        self._clock = clock
        self._usage_reader = usage_reader or (
            lambda: resource.getrusage(resource.RUSAGE_SELF)
        )
        self._platform_name = platform_name
        self._prior_wall_seconds = 0.0
        self._prior_user_seconds = 0.0
        self._prior_system_seconds = 0.0
        self._prior_peak_rss_bytes = 0
        self._stage_timing_seconds: dict[str, float] = {}
        self._target_measurements: dict[str, dict[str, object]] = {}
        self._completed_target_labels: list[str] = []
        evidence_path = self.run_directory / RESOURCE_USAGE_FILE
        if evidence_path.is_file():
            try:
                prior_value = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("Existing resource usage evidence is unreadable.") from error
            prior = validate_resource_usage_payload(prior_value)
            if prior["resource_envelope"] != self.resource_envelope:
                raise ValueError("Existing resource envelope does not match this run.")
            self._prior_wall_seconds = float(prior["wall_time_seconds"])
            self._prior_user_seconds = float(prior["user_cpu_seconds"])
            self._prior_system_seconds = float(prior["system_cpu_seconds"])
            self._prior_peak_rss_bytes = int(prior["peak_rss_bytes"])
            self._stage_timing_seconds = dict(prior["stage_timing_seconds"])
            self._target_measurements = json.loads(
                json.dumps(prior["target_measurements"])
            )
            self._completed_target_labels = list(prior["completed_targets"])
        self._started_wall = float(self._clock())
        initial_usage = self._usage_reader()
        self._started_user = float(initial_usage.ru_utime)
        self._started_system = float(initial_usage.ru_stime)
        self._process_peak_rss_bytes = normalize_peak_rss_bytes(
            initial_usage.ru_maxrss,
            self._platform_name,
        )

    def _ensure_live_target(
        self,
        target_label: str,
        target_family: str,
    ) -> dict[str, object]:
        """Return or create one target-local live timing record.

        Parameters
        ----------
        target_label : str
            Stable configured target identifier.
        target_family : {"categorical", "numerical"}
            Decoder family used for the target.

        Returns
        -------
        dict[str, object]
            Mutable internal target measurement record.
        """
        if not isinstance(target_label, str) or not target_label.strip():
            raise ValueError("target_label must be a nonempty string.")
        if target_family not in {"categorical", "numerical"}:
            raise ValueError("target_family must be categorical or numerical.")
        existing = self._target_measurements.get(target_label)
        if existing is not None:
            if existing["target_family"] != target_family:
                raise ValueError("Target measurement family changed across resume.")
            return existing
        record: dict[str, object] = {
            "target_family": target_family,
            "timing_available": True,
            "total_seconds": None,
            "timing_seconds": {
                "split": 0.0,
                "regional_transform": {},
                "feature_construction": {},
                "estimator": {},
            },
            "operation_counts": {
                "split": 0,
                "regional_transform": {},
                "feature_construction": {},
                "estimator": {},
            },
            "requested_fit_count": 0,
            "valid_outer_cell_count": 0,
            "invalid_outer_cell_count": 0,
        }
        self._target_measurements[target_label] = record
        return record

    def record_model_event(
        self,
        *,
        target_label: str,
        target_family: str,
        operation: str,
        region_configuration: str | None,
        representation: str | None,
        elapsed_seconds: float,
    ) -> None:
        """Accumulate one model operation duration and invocation count.

        Parameters
        ----------
        target_label : str
            Stable configured target identifier.
        target_family : {"categorical", "numerical"}
            Decoder family.
        operation : {"split", "regional_transform", "feature_construction", "estimator"}
            Timed operation class.
        region_configuration : str or None
            ``PFC``/``HPC`` transform region, cell region, or None for splits.
        representation : str or None
            ``pca``/``units`` for cell operations; otherwise None.
        elapsed_seconds : float
            Finite nonnegative operation duration in seconds.

        Returns
        -------
        None
            Updates only in-memory evidence until :meth:`snapshot` is called.
        """
        if operation not in _OPERATIONS:
            raise ValueError(f"Unsupported timed operation {operation!r}.")
        elapsed = _nonnegative_seconds(elapsed_seconds, "elapsed_seconds")
        record = self._ensure_live_target(target_label, target_family)
        if not record["timing_available"]:
            raise ValueError("Cannot append timings to a restored timing-unavailable target.")
        timing = record["timing_seconds"]
        counts = record["operation_counts"]
        if operation == "split":
            if region_configuration is not None or representation is not None:
                raise ValueError("Split timing cannot declare region or representation.")
            timing["split"] = float(timing["split"]) + elapsed
            counts["split"] = int(counts["split"]) + 1
            return
        if not isinstance(region_configuration, str) or not region_configuration:
            raise ValueError("Timed regional operations require a region configuration.")
        if operation == "regional_transform":
            if region_configuration not in {"PFC", "HPC"} or representation is not None:
                raise ValueError("Regional-transform timing requires PFC/HPC and no representation.")
            key = region_configuration
        else:
            if region_configuration not in {"PFC", "HPC", "PFC+HPC"} or representation not in {
                "pca",
                "units",
            }:
                raise ValueError("Cell timing requires a canonical region and representation.")
            key = f"{region_configuration}|{representation}"
        timing_group = timing[operation]
        count_group = counts[operation]
        timing_group[key] = float(timing_group.get(key, 0.0)) + elapsed
        count_group[key] = int(count_group.get(key, 0)) + 1

    def finish_target(
        self,
        *,
        target_label: str,
        target_family: str,
        total_seconds: float,
        requested_fit_count: int,
        valid_outer_cell_count: int,
        invalid_outer_cell_count: int,
    ) -> None:
        """Finalize one newly decoded target's wall and fit-count evidence.

        Parameters
        ----------
        target_label : str
            Stable configured target identifier.
        target_family : {"categorical", "numerical"}
            Decoder family.
        total_seconds : float
            Target-level elapsed wall time in seconds.
        requested_fit_count : int
            Planned estimator fit count for this target.
        valid_outer_cell_count, invalid_outer_cell_count : int
            Completed outer result-cell counts by validity.

        Returns
        -------
        None
            Finalizes the in-memory target record.
        """
        record = self._ensure_live_target(target_label, target_family)
        record["timing_available"] = True
        record["total_seconds"] = _nonnegative_seconds(total_seconds, "total_seconds")
        record["requested_fit_count"] = _nonnegative_int(
            requested_fit_count, "requested_fit_count"
        )
        record["valid_outer_cell_count"] = _nonnegative_int(
            valid_outer_cell_count, "valid_outer_cell_count"
        )
        record["invalid_outer_cell_count"] = _nonnegative_int(
            invalid_outer_cell_count, "invalid_outer_cell_count"
        )
        if target_label not in self._completed_target_labels:
            self._completed_target_labels.append(target_label)

    def record_restored_target(
        self,
        *,
        target_label: str,
        target_family: str,
        requested_fit_count: int,
        valid_outer_cell_count: int,
        invalid_outer_cell_count: int,
    ) -> None:
        """Record a valid checkpoint when no historical timing evidence exists.

        Parameters
        ----------
        target_label : str
            Stable restored checkpoint identifier.
        target_family : {"categorical", "numerical"}
            Decoder family.
        requested_fit_count : int
            Planned estimator fit count for this target.
        valid_outer_cell_count, invalid_outer_cell_count : int
            Restored outer result-cell counts by validity.

        Returns
        -------
        None
            Preserves prior evidence when present; otherwise adds an explicit
            timing-unavailable target record.
        """
        if target_label in self._completed_target_labels:
            existing = self._target_measurements[target_label]
            if existing["target_family"] != target_family:
                raise ValueError("Target measurement family changed across resume.")
            return
        self._target_measurements[target_label] = {
            "target_family": target_family,
            "timing_available": False,
            "total_seconds": None,
            "timing_seconds": {},
            "operation_counts": {},
            "requested_fit_count": _nonnegative_int(
                requested_fit_count, "requested_fit_count"
            ),
            "valid_outer_cell_count": _nonnegative_int(
                valid_outer_cell_count, "valid_outer_cell_count"
            ),
            "invalid_outer_cell_count": _nonnegative_int(
                invalid_outer_cell_count, "invalid_outer_cell_count"
            ),
        }
        self._completed_target_labels.append(target_label)

    @property
    def completed_target_labels(self) -> tuple[str, ...]:
        """Return checkpoint-complete target labels in pipeline order.

        Returns
        -------
        tuple[str, ...]
            Stable dimensionless target identifiers. A target with only
            partial operation timing is excluded until its checkpoint is
            finalized or restored.
        """
        return tuple(self._completed_target_labels)

    def snapshot(
        self,
        *,
        status: str,
        completed_targets: Sequence[str],
        stage_timing_seconds: Mapping[str, float] | None = None,
    ) -> dict[str, object]:
        """Atomically publish cumulative evidence for this process and prior resumes.

        Parameters
        ----------
        status : {"running", "failed", "interrupted", "complete"}
            Lifecycle represented by this evidence snapshot.
        completed_targets : sequence[str]
            Ordered checkpoint-complete target identifiers.
        stage_timing_seconds : mapping[str, float] or None, default=None
            Latest pipeline-stage durations in seconds. Omitted values preserve
            the preceding snapshot's stage mapping.

        Returns
        -------
        dict[str, object]
            Validated JSON payload written to ``resource_usage.json``.
        """
        if status not in _STATUSES:
            raise ValueError("Resource snapshot status is invalid.")
        completed = list(completed_targets)
        _validate_string_sequence(completed, "completed_targets")
        if any(label not in self._target_measurements for label in completed):
            raise ValueError("Every completed target needs a measurement record.")
        if any(label not in self._completed_target_labels for label in completed):
            raise ValueError("Completed target evidence requires a finalized checkpoint.")
        if stage_timing_seconds is not None:
            self._stage_timing_seconds = _validate_timing_mapping(
                stage_timing_seconds,
                "stage_timing_seconds",
            )
        current_wall = float(self._clock())
        current_usage = self._usage_reader()
        self._process_peak_rss_bytes = max(
            self._process_peak_rss_bytes,
            normalize_peak_rss_bytes(current_usage.ru_maxrss, self._platform_name),
        )
        wall_seconds = self._prior_wall_seconds + max(
            0.0, current_wall - self._started_wall
        )
        user_seconds = self._prior_user_seconds + max(
            0.0, float(current_usage.ru_utime) - self._started_user
        )
        system_seconds = self._prior_system_seconds + max(
            0.0, float(current_usage.ru_stime) - self._started_system
        )
        peak_rss_bytes = max(self._prior_peak_rss_bytes, self._process_peak_rss_bytes)
        fit_counts = {
            "requested": 0,
            "measured_estimator_calls": 0,
            "valid_outer_cells": 0,
            "invalid_outer_cells": 0,
        }
        for measurement in self._target_measurements.values():
            fit_counts["requested"] += int(measurement["requested_fit_count"])
            fit_counts["valid_outer_cells"] += int(
                measurement["valid_outer_cell_count"]
            )
            fit_counts["invalid_outer_cells"] += int(
                measurement["invalid_outer_cell_count"]
            )
            estimator_counts = measurement["operation_counts"].get("estimator", {})
            fit_counts["measured_estimator_calls"] += sum(
                int(count) for count in estimator_counts.values()
            )
        payload = {
            "schema_version": RESOURCE_USAGE_SCHEMA_VERSION,
            "measurement_method": "resource.getrusage",
            "status": status,
            "peak_rss_bytes": peak_rss_bytes,
            "wall_time_seconds": wall_seconds,
            "user_cpu_seconds": user_seconds,
            "system_cpu_seconds": system_seconds,
            "cpu_efficiency": (
                (user_seconds + system_seconds) / wall_seconds
                if wall_seconds > 0.0
                else 0.0
            ),
            "resource_envelope": self.resource_envelope,
            "completed_targets": completed,
            "target_measurements": self._target_measurements,
            "stage_timing_seconds": self._stage_timing_seconds,
            "fit_counts": fit_counts,
            "output_size_bytes": _directory_output_size_bytes(self.run_directory),
        }
        validated = validate_resource_usage_payload(payload)
        _atomic_json(self.run_directory / RESOURCE_USAGE_FILE, validated)
        return validated
