"""Interruption-safe orchestration for the nonpublishing CT026 PPC profile.

This module owns scalar run state and reports only. Numerical phase preparation,
spike selection, and PPC profiling are injected so the runner cannot publish an
LFP summary component, manifest, or scientific NPZ artifact.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Callable, Mapping
from uuid import uuid4


_RUN_PREFIX = "ct026_ppc_profile_"
_RUN_NAME = re.compile(r"^ct026_ppc_profile_[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SCENARIOS = ("low", "median", "high", "combined")
_ORDERED_STAGES = ("phase_cold", "phase_warm", "selection", *_SCENARIOS)
_SHUFFLE_COUNT = 100
_STATE_SCHEMA_VERSION = "2"
_GROUPED_PROFILE_SCHEMA_VERSION = "grouped_ppc_profile_result.v1"
_GROUPED_PROFILE_DOCUMENT_SCHEMA_VERSION = "ct026_grouped_ppc_profile.v1"
_GROUPED_PROFILE_KIND = "grouped_serial_ppc"
_GROUPED_PUBLIC_FIELDS = frozenset(
    {
        "schema_version", "profile_kind", "run_fingerprint",
        "geometry_build_seconds", "observed_reduction_seconds",
        "union_edge_reduction_seconds", "shuffle_aggregation_seconds",
        "null_summarization_seconds", "representative_histogram_seconds",
        "checkpoint_overhead_seconds", "total_elapsed_seconds",
        "throughput_scheduled_edge_per_second", "scheduled_edge_count",
        "independent_edge_count", "unique_site_qualified_union_edge_count",
        "edge_union_saturation", "edge_reuse_ratio",
        "planned_parent_private_bytes", "planned_worker_private_bytes",
        "shared_phase_mmap_bytes", "planned_aggregate_array_bytes",
        "measured_peak_process_rss_bytes", "measured_peak_aggregate_rss_bytes",
        "measured_peak_aggregate_pss_bytes", "measured_memory_source",
        "projection_100_scheduled_edge_count",
        "projection_100_independent_edge_count",
        "projection_100_unique_site_qualified_union_edge_count",
        "projection_100_edge_union_saturation", "projection_100_edge_reuse_ratio",
        "projection_1000_scheduled_edge_count",
        "projection_1000_independent_edge_count",
        "projection_1000_unique_site_qualified_union_edge_count",
        "projection_1000_edge_union_saturation", "projection_1000_edge_reuse_ratio",
    }
)
_POST_ISOLATION_PROVENANCE_FIELDS = frozenset(
    {"peak_memory_bytes", "peak_memory_source", "child_pid"}
)
_ADAPTER_IDENTITY_FIELDS = frozenset(
    {"config_fingerprint", "source_fingerprint", "git_fingerprint"}
)


@dataclass(frozen=True)
class PhaseProfilePreparation:
    """Caller-owned prepared phase plus scalar preparation measurements.

    ``phase`` preserves its caller-defined arrays, axes, and physical units.
    ``metrics`` contains elapsed seconds, peak resident bytes and source, and a
    Boolean cache-hit flag. The runner persists only those validated scalars.
    """

    phase: object
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class CT026PPCProfileRunResult:
    """Paths and scalar results for one completed CT026 profile run.

    Attributes
    ----------
    run_directory : pathlib.Path
        Direct child of the caller's analysis root. It contains work-only PPC
        directories, atomic scalar ``state.json``, scalar ``profile.json``, and
        ``summary.md``. It never contains a final component or manifest.
    state_path, profile_path, summary_path : pathlib.Path
        Files within ``run_directory``. JSON values are categorical identities,
        dimensionless counts, seconds, bytes, or other scalar metrics supplied
        by the injected profiler. No numerical arrays are retained.
    profiles : dict[str, dict[str, JSON scalar]]
        Scalar metrics for the ordered low, median, high, and combined
        100-shuffle scenarios.
    """

    run_directory: Path
    state_path: Path
    profile_path: Path
    summary_path: Path
    profiles: dict[str, dict[str, object]]


def run_ct026_ppc_profile(
    *,
    config: object,
    config_fingerprint: str,
    source_fingerprint: str,
    git_fingerprint: str,
    analysis_root: Path,
    prepare_phase: Callable[..., object],
    select_spikes: Callable[[object], object],
    profile_job: Callable[..., Mapping[str, object]],
    acquire_run_lock: Callable[
        [Path, Mapping[str, str]], AbstractContextManager[None]
    ],
    timestamp_factory: Callable[[], str] | None = None,
    run_directory: Path | None = None,
) -> CT026PPCProfileRunResult:
    """Run or resume the ordered, injected CT026 100-shuffle PPC profile.

    Parameters
    ----------
    config : object
        Caller-owned CT026 configuration identity. The numerical callbacks may
        close over this object; the runner neither serializes nor mutates it.
    config_fingerprint, source_fingerprint, git_fingerprint : str
        Nonempty exact identities without physical units. A resumed run must
        match every saved value before any injected callback is invoked.
    analysis_root : pathlib.Path
        Parent directory for one timestamped run. A supplied resume directory
        must be an existing, nonsymlink direct child with the expected prefix.
    prepare_phase : callable
        Called with ``warm`` and the exact
        ``run_directory / 'work' / 'phase'`` path. It returns either
        ``(phase, metrics)`` or :class:`PhaseProfilePreparation`; metrics are
        elapsed seconds, peak resident bytes/source, and Boolean cache status.
        A new run requires a cold miss then a warm hit. An incomplete resumed
        run skips completed cold work, rehydrates warm phase from the same root,
        and stores that measurement separately from the original warm profile.
    select_spikes : callable
        Receives the warm prepared phase and returns caller-owned selected spike
        metadata. It is called once on an incomplete resume even when the saved
        selection stage is complete because in-memory objects are not persisted.
    profile_job : callable
        Receives keyword arguments ``scenario``, ``phase``, ``spikes``,
        ``shuffle_count=100``, and a scenario-specific ``work_root``. It returns
        a string-keyed mapping of finite JSON scalar metrics such as seconds and
        bytes. Interrupted work below ``work_root`` is deliberately preserved.
    acquire_run_lock : callable
        Receives the resolved exact run directory and identity mapping and
        returns a context manager. State initialization/loading, callbacks, and
        final report writes all occur while this lock is held.
    timestamp_factory : callable, optional
        Produces one filesystem-safe timestamp for a new run. It is not called
        when ``run_directory`` resumes an existing run. UTC is used by default.
    run_directory : pathlib.Path, optional
        Existing exact run to resume. Completed numerical scenarios are skipped;
        incomplete scenarios reuse their preserved work directories.

    Returns
    -------
    CT026PPCProfileRunResult
        Completed run paths and scalar scenario metrics. State is replaced
        atomically after every completed ordered stage and warm rehydration.

    Raises
    ------
    ValueError
        If identities, paths, saved state, callback contracts, or scalar metrics
        are invalid. Identity mismatch is detected before callbacks run.
    FileExistsError
        If a new deterministic timestamped run directory already exists.
    OSError
        If directory creation or atomic scalar report replacement fails.
    BaseException
        Callback interruptions and failures propagate after preserving prior
        state and all scenario work already written by the callback.
    """
    del config  # The injected numerical callbacks own the opaque configuration.
    identity = _validated_identity(
        config_fingerprint,
        source_fingerprint,
        git_fingerprint,
    )
    root = _validated_analysis_root(analysis_root)
    if (
        not callable(prepare_phase)
        or not callable(select_spikes)
        or not callable(profile_job)
        or not callable(acquire_run_lock)
    ):
        raise ValueError(
            "phase preparation, spike selection, profile job, and run lock must be callable"
        )

    new_run = run_directory is None
    if run_directory is None:
        timestamp = _profile_timestamp(timestamp_factory)
        active_directory = root / f"{_RUN_PREFIX}{timestamp}"
        _validate_new_run_path(root, active_directory)
        active_directory.mkdir(parents=True, exist_ok=False)
    else:
        active_directory = _validated_resume_directory(root, Path(run_directory))

    with acquire_run_lock(active_directory, dict(identity)):
        if new_run:
            state = _new_state(identity)
            _write_json_atomic(active_directory / "state.json", state)
        else:
            state = _load_state(active_directory / "state.json", identity)
        return _run_locked_profile(
            active_directory=active_directory,
            identity=identity,
            state=state,
            prepare_phase=prepare_phase,
            select_spikes=select_spikes,
            profile_job=profile_job,
        )


def _run_locked_profile(
    *,
    active_directory: Path,
    identity: Mapping[str, str],
    state: dict[str, object],
    prepare_phase: Callable[..., object],
    select_spikes: Callable[[object], object],
    profile_job: Callable[..., Mapping[str, object]],
) -> CT026PPCProfileRunResult:
    """Run unfinished stages while the caller holds the exact run lock."""
    completed = state["completed_stages"]
    phase_profiles = state["phase_profiles"]
    assert isinstance(completed, list)
    assert isinstance(phase_profiles, dict)
    phase_work_root = active_directory / "work" / "phase"
    phase_work_root.mkdir(parents=True, exist_ok=True)

    if "phase_cold" not in completed:
        _, cold_metrics = _prepare_phase_profile(
            prepare_phase,
            warm=False,
            phase_work_root=phase_work_root,
        )
        phase_profiles["cold"] = cold_metrics
        _complete_stage(active_directory, state, "phase_cold")

    unfinished_scenarios = [
        scenario for scenario in _SCENARIOS if scenario not in completed
    ]
    if unfinished_scenarios:
        phase, warm_metrics = _prepare_phase_profile(
            prepare_phase,
            warm=True,
            phase_work_root=phase_work_root,
        )
        if "phase_warm" not in completed:
            phase_profiles["warm"] = warm_metrics
            _complete_stage(active_directory, state, "phase_warm")
        else:
            phase_profiles["resume_warm"] = warm_metrics
            _write_json_atomic(active_directory / "state.json", state)

        spikes = select_spikes(phase)
        if "selection" not in completed:
            _complete_stage(active_directory, state, "selection")

        scenario_profiles = state["profiles"]
        assert isinstance(scenario_profiles, dict)
        for scenario in unfinished_scenarios:
            work_root = active_directory / "work" / scenario
            work_root.mkdir(parents=True, exist_ok=True)
            metrics = profile_job(
                scenario=scenario,
                phase=phase,
                spikes=spikes,
                shuffle_count=_SHUFFLE_COUNT,
                work_root=work_root,
            )
            scenario_profiles[scenario] = _validated_scalar_metrics(
                metrics,
                scenario,
            )
            _complete_stage(active_directory, state, scenario)

    profiles = _completed_profiles(state)
    profile_path = active_directory / "profile.json"
    summary_path = active_directory / "summary.md"
    _write_json_atomic(profile_path, _profile_document(identity, state, profiles))
    _write_text_atomic(
        summary_path,
        _markdown_summary(identity, phase_profiles, profiles),
    )
    return CT026PPCProfileRunResult(
        run_directory=active_directory,
        state_path=active_directory / "state.json",
        profile_path=profile_path,
        summary_path=summary_path,
        profiles=profiles,
    )


def _prepare_phase_profile(
    prepare_phase: Callable[..., object],
    *,
    warm: bool,
    phase_work_root: Path,
) -> tuple[object, dict[str, object]]:
    """Invoke one phase stage and return its object plus validated scalar metrics."""
    result = prepare_phase(warm=warm, phase_work_root=phase_work_root)
    if isinstance(result, PhaseProfilePreparation):
        phase = result.phase
        metrics = result.metrics
    elif isinstance(result, tuple) and len(result) == 2:
        phase, metrics = result
    else:
        raise ValueError(
            "prepare_phase must return (phase, metrics) or PhaseProfilePreparation"
        )
    return phase, _validated_phase_metrics(metrics, expected_cache_hit=warm)


def _validated_identity(
    config_fingerprint: str,
    source_fingerprint: str,
    git_fingerprint: str,
) -> dict[str, str]:
    """Return the three exact nonempty run identities."""
    identity = {
        "config_fingerprint": config_fingerprint,
        "source_fingerprint": source_fingerprint,
        "git_fingerprint": git_fingerprint,
    }
    for field_name, value in identity.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a nonempty string")
    return identity


def _validated_analysis_root(analysis_root: Path) -> Path:
    """Return an absolute analysis root without accepting an existing symlink."""
    root = Path(analysis_root)
    if root.exists() and root.is_symlink():
        raise ValueError("analysis_root must not be a symbolic link")
    return root.resolve(strict=False)


def _profile_timestamp(timestamp_factory: Callable[[], str] | None) -> str:
    """Return one safe timestamp token for a deterministic new run name."""
    if timestamp_factory is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    else:
        if not callable(timestamp_factory):
            raise ValueError("timestamp_factory must be callable")
        timestamp = timestamp_factory()
    candidate_name = f"{_RUN_PREFIX}{timestamp}"
    if not isinstance(timestamp, str) or not _RUN_NAME.fullmatch(candidate_name):
        raise ValueError("timestamp_factory must return a filesystem-safe nonempty token")
    return timestamp


def _validate_new_run_path(root: Path, run_directory: Path) -> None:
    """Reject unsafe or preexisting new run paths before any directory write."""
    if run_directory.parent != root or not _RUN_NAME.fullmatch(run_directory.name):
        raise ValueError("new CT026 profile run must be a direct child of analysis_root")
    if run_directory.exists() or run_directory.is_symlink():
        raise FileExistsError(f"CT026 PPC profile run already exists: {run_directory}")


def _validated_resume_directory(root: Path, run_directory: Path) -> Path:
    """Return an existing nonsymlink direct child safe for exact-state resume."""
    if run_directory.is_symlink():
        raise ValueError("CT026 profile run_directory must not be a symbolic link")
    resolved = run_directory.resolve(strict=False)
    if (
        resolved.parent != root
        or not _RUN_NAME.fullmatch(resolved.name)
        or not resolved.is_dir()
    ):
        raise ValueError("run_directory must be an existing CT026 profile child of analysis_root")
    return resolved


def _new_state(identity: Mapping[str, str]) -> dict[str, object]:
    """Return the minimal JSON-only state for a newly created profile run."""
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "identity": dict(identity),
        "shuffle_count": _SHUFFLE_COUNT,
        "completed_stages": [],
        "phase_profiles": {
            "cold": None,
            "warm": None,
            "resume_warm": None,
        },
        "profiles": {},
    }


def _load_state(state_path: Path, expected_identity: Mapping[str, str]) -> dict[str, object]:
    """Load and validate scalar state and exact identity before any hydration."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("unable to load valid CT026 PPC profile state.json") from error
    if not isinstance(state, dict):
        raise ValueError("CT026 PPC profile state must be a JSON object")
    saved_identity = state.get("identity")
    if not isinstance(saved_identity, dict):
        raise ValueError("CT026 PPC profile state has no valid identity")
    for field_name, expected_value in expected_identity.items():
        if saved_identity.get(field_name) != expected_value:
            raise ValueError(f"{field_name} does not match the saved CT026 profile run")
    if saved_identity != dict(expected_identity):
        raise ValueError("saved CT026 profile identity has unexpected fields")
    if state.get("schema_version") != _STATE_SCHEMA_VERSION:
        raise ValueError("unsupported CT026 PPC profile state schema")
    if state.get("shuffle_count") != _SHUFFLE_COUNT:
        raise ValueError("saved CT026 PPC profile must use exactly 100 shuffles")

    completed = state.get("completed_stages")
    if (
        not isinstance(completed, list)
        or any(not isinstance(stage, str) for stage in completed)
        or completed != list(_ORDERED_STAGES[: len(completed)])
    ):
        raise ValueError("saved CT026 PPC profile stages are not an ordered prefix")
    profiles = state.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(completed).intersection(_SCENARIOS):
        raise ValueError("saved CT026 PPC profile metrics disagree with completed stages")
    for scenario, metrics in profiles.items():
        profiles[scenario] = _validated_scalar_metrics(metrics, scenario)
    state["phase_profiles"] = _validated_saved_phase_profiles(
        state.get("phase_profiles"),
        completed,
    )
    return state


def _complete_stage(run_directory: Path, state: dict[str, object], stage: str) -> None:
    """Append one expected stage and atomically replace scalar run state."""
    completed = state["completed_stages"]
    assert isinstance(completed, list)
    expected_stage = _ORDERED_STAGES[len(completed)]
    if stage != expected_stage:
        raise ValueError(f"expected CT026 PPC profile stage {expected_stage!r}, got {stage!r}")
    completed.append(stage)
    try:
        _write_json_atomic(run_directory / "state.json", state)
    except BaseException:
        completed.pop()
        raise


def _validated_scalar_metrics(
    metrics: Mapping[str, object],
    scenario: str,
) -> dict[str, object]:
    """Return a copied finite JSON-scalar metrics mapping for one scenario."""
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError(f"{scenario} profile metrics must be a nonempty mapping")
    validated: dict[str, object] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{scenario} profile metric names must be nonempty strings")
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise ValueError(f"{scenario} profile metric {key!r} must be a JSON scalar")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{scenario} profile metric {key!r} must be finite")
        validated[key] = value
    if "schema_version" in validated or "profile_kind" in validated:
        return _validated_grouped_profile_metrics(validated, scenario)
    return validated


def _validated_grouped_profile_metrics(
    metrics: Mapping[str, object],
    scenario: str,
) -> dict[str, object]:
    """Validate the exact scalar S6 grouped-profile public contract.

    The runner persists the full scalar map supplied after child isolation.
    Raw ``ru_maxrss`` fields are intentionally not part of this contract: the
    adapter transforms them to ``peak_memory_*`` before this boundary.
    """
    names = set(metrics)
    optional = names.intersection(_POST_ISOLATION_PROVENANCE_FIELDS)
    adapter_identity = names.intersection(_ADAPTER_IDENTITY_FIELDS)
    if optional and optional != _POST_ISOLATION_PROVENANCE_FIELDS:
        raise ValueError(f"{scenario} grouped profile provenance fields are incomplete")
    if adapter_identity and adapter_identity != _ADAPTER_IDENTITY_FIELDS:
        raise ValueError(f"{scenario} grouped profile adapter identity fields are incomplete")
    expected = _GROUPED_PUBLIC_FIELDS | optional | adapter_identity
    if names != expected:
        unexpected = sorted(names.symmetric_difference(expected))
        raise ValueError(f"{scenario} grouped profile public fields disagree: {unexpected}")
    if metrics["schema_version"] != _GROUPED_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"{scenario} profile metric schema_version is unsupported")
    if metrics["profile_kind"] != _GROUPED_PROFILE_KIND:
        raise ValueError(f"{scenario} profile metric profile_kind is unsupported")
    if (
        not isinstance(metrics["run_fingerprint"], str)
        or re.fullmatch(r"[0-9a-f]{64}", metrics["run_fingerprint"]) is None
    ):
        raise ValueError(
            f"{scenario} profile metric run_fingerprint must be 64 lowercase hexadecimal characters"
        )
    for name in (
        "geometry_build_seconds", "observed_reduction_seconds",
        "union_edge_reduction_seconds", "shuffle_aggregation_seconds",
        "null_summarization_seconds", "representative_histogram_seconds",
        "checkpoint_overhead_seconds", "total_elapsed_seconds",
        "throughput_scheduled_edge_per_second", "edge_union_saturation",
        "edge_reuse_ratio", "projection_100_edge_union_saturation",
        "projection_100_edge_reuse_ratio", "projection_1000_edge_union_saturation",
        "projection_1000_edge_reuse_ratio",
    ):
        _nonnegative_finite_grouped_float(metrics[name], name, scenario)
    if float(metrics["total_elapsed_seconds"]) <= 0.0:
        raise ValueError(f"{scenario} profile metric total_elapsed_seconds must be positive")
    for name in (
        "scheduled_edge_count", "independent_edge_count",
        "unique_site_qualified_union_edge_count", "planned_parent_private_bytes",
        "planned_worker_private_bytes", "shared_phase_mmap_bytes",
        "planned_aggregate_array_bytes", "measured_peak_process_rss_bytes",
        "projection_100_scheduled_edge_count",
        "projection_100_independent_edge_count",
        "projection_100_unique_site_qualified_union_edge_count",
        "projection_1000_scheduled_edge_count",
        "projection_1000_independent_edge_count",
        "projection_1000_unique_site_qualified_union_edge_count",
    ):
        _nonnegative_grouped_int(metrics[name], name, scenario)
    _validate_optional_grouped_memory(metrics, scenario)
    if not isinstance(metrics["measured_memory_source"], str) or not metrics["measured_memory_source"]:
        raise ValueError(f"{scenario} profile metric measured_memory_source must be nonempty")
    _validate_grouped_projection_relations(metrics, scenario)
    if optional:
        _nonnegative_grouped_int(metrics["peak_memory_bytes"], "peak_memory_bytes", scenario)
        if not isinstance(metrics["peak_memory_source"], str) or not metrics["peak_memory_source"]:
            raise ValueError(f"{scenario} profile metric peak_memory_source must be nonempty")
        child_pid = metrics["child_pid"]
        if isinstance(child_pid, bool) or not isinstance(child_pid, int) or child_pid <= 0:
            raise ValueError(f"{scenario} profile metric child_pid must be a positive integer")
    for name in adapter_identity:
        if not isinstance(metrics[name], str) or not metrics[name]:
            raise ValueError(f"{scenario} profile metric {name} must be nonempty")
    return dict(metrics)


def _nonnegative_finite_grouped_float(value: object, name: str, scenario: str) -> None:
    """Reject Boolean, nonnumeric, negative, or nonfinite grouped scalars."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{scenario} profile metric {name} must be finite and nonnegative")


def _nonnegative_grouped_int(value: object, name: str, scenario: str) -> None:
    """Reject non-Python integer grouped counts and byte values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{scenario} profile metric {name} must be a nonnegative integer")


def _validate_optional_grouped_memory(metrics: Mapping[str, object], scenario: str) -> None:
    """Validate ordered measured process RSS, aggregate PSS, and aggregate RSS."""
    process = metrics["measured_peak_process_rss_bytes"]
    aggregate_rss = metrics["measured_peak_aggregate_rss_bytes"]
    aggregate_pss = metrics["measured_peak_aggregate_pss_bytes"]
    if aggregate_rss is None and aggregate_pss is None:
        return
    _nonnegative_grouped_int(aggregate_rss, "measured_peak_aggregate_rss_bytes", scenario)
    _nonnegative_grouped_int(aggregate_pss, "measured_peak_aggregate_pss_bytes", scenario)
    if process > aggregate_pss or aggregate_pss > aggregate_rss:
        raise ValueError(f"{scenario} grouped measured memory ordering is invalid")


def _validate_grouped_projection_relations(metrics: Mapping[str, object], scenario: str) -> None:
    """Validate each projection and base/100 plus scheduled-count relations."""
    for name in (
        "scheduled_edge_count", "independent_edge_count",
        "unique_site_qualified_union_edge_count", "edge_union_saturation",
        "edge_reuse_ratio",
    ):
        if metrics[name] != metrics[f"projection_100_{name}"]:
            raise ValueError(f"{scenario} profile metric projection_100_{name} disagrees with base")
    if (
        metrics["projection_1000_scheduled_edge_count"]
        != 10 * metrics["projection_100_scheduled_edge_count"]
    ):
        raise ValueError(
            f"{scenario} profile metric projection_1000_scheduled_edge_count disagrees with 100-shuffle plan"
        )
    _validate_grouped_edge_metrics(metrics, "", scenario)
    _validate_grouped_edge_metrics(metrics, "projection_100_", scenario)
    _validate_grouped_edge_metrics(metrics, "projection_1000_", scenario)
    expected_throughput = (
        metrics["scheduled_edge_count"] / metrics["total_elapsed_seconds"]
    )
    if metrics["throughput_scheduled_edge_per_second"] != expected_throughput:
        raise ValueError(
            f"{scenario} profile metric throughput_scheduled_edge_per_second disagrees with scheduled edges and elapsed time"
        )


def _validate_grouped_edge_metrics(
    metrics: Mapping[str, object],
    prefix: str,
    scenario: str,
) -> None:
    """Validate count order and internally derived reuse scalars for one plan."""
    scheduled = metrics[f"{prefix}scheduled_edge_count"]
    independent = metrics[f"{prefix}independent_edge_count"]
    union = metrics[f"{prefix}unique_site_qualified_union_edge_count"]
    saturation = metrics[f"{prefix}edge_union_saturation"]
    reuse = metrics[f"{prefix}edge_reuse_ratio"]
    if union > independent:
        raise ValueError(
            f"{scenario} profile metric {prefix}unique_site_qualified_union_edge_count exceeds independent_edge_count"
        )
    if independent > scheduled:
        raise ValueError(
            f"{scenario} profile metric {prefix}independent_edge_count exceeds scheduled_edge_count"
        )
    if saturation > 1.0:
        raise ValueError(
            f"{scenario} profile metric {prefix}edge_union_saturation must not exceed one"
        )
    expected_reuse = 0.0 if union == 0 else independent / union
    if reuse != expected_reuse:
        raise ValueError(
            f"{scenario} profile metric {prefix}edge_reuse_ratio disagrees with independent_edge_count and unique_site_qualified_union_edge_count"
        )


def _validated_phase_metrics(
    metrics: object,
    *,
    expected_cache_hit: bool,
) -> dict[str, object]:
    """Return exact finite phase seconds/bytes/source/cache-hit scalar metrics."""
    required_fields = {
        "elapsed_seconds",
        "peak_memory_bytes",
        "peak_memory_source",
        "cache_hit",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != required_fields:
        raise ValueError("phase_profiles metrics must contain the exact required fields")
    elapsed_seconds = metrics["elapsed_seconds"]
    peak_memory_bytes = metrics["peak_memory_bytes"]
    peak_memory_source = metrics["peak_memory_source"]
    cache_hit = metrics["cache_hit"]
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0.0
    ):
        raise ValueError("phase_profiles elapsed_seconds must be finite and nonnegative")
    if (
        isinstance(peak_memory_bytes, bool)
        or not isinstance(peak_memory_bytes, int)
        or peak_memory_bytes < 0
    ):
        raise ValueError("phase_profiles peak_memory_bytes must be a nonnegative integer")
    if not isinstance(peak_memory_source, str) or not peak_memory_source:
        raise ValueError("phase_profiles peak_memory_source must be a nonempty string")
    if not isinstance(cache_hit, bool) or cache_hit is not expected_cache_hit:
        raise ValueError("phase_profiles cache_hit disagrees with cold/warm stage")
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_source": peak_memory_source,
        "cache_hit": cache_hit,
    }


def _validated_saved_phase_profiles(
    phase_profiles: object,
    completed_stages: list[str],
) -> dict[str, object]:
    """Validate persisted cold, original warm, and optional latest resume metrics."""
    required_names = {"cold", "warm", "resume_warm"}
    if not isinstance(phase_profiles, dict) or set(phase_profiles) != required_names:
        raise ValueError("saved phase_profiles must contain cold, warm, and resume_warm")
    cold = phase_profiles["cold"]
    warm = phase_profiles["warm"]
    resume_warm = phase_profiles["resume_warm"]
    if "phase_cold" in completed_stages:
        cold = _validated_phase_metrics(cold, expected_cache_hit=False)
    elif cold is not None:
        raise ValueError("saved phase_profiles cold metrics precede phase_cold")
    if "phase_warm" in completed_stages:
        warm = _validated_phase_metrics(warm, expected_cache_hit=True)
    elif warm is not None:
        raise ValueError("saved phase_profiles warm metrics precede phase_warm")
    if resume_warm is not None:
        if "phase_warm" not in completed_stages:
            raise ValueError("saved phase_profiles resume metrics precede phase_warm")
        resume_warm = _validated_phase_metrics(
            resume_warm,
            expected_cache_hit=True,
        )
    return {"cold": cold, "warm": warm, "resume_warm": resume_warm}


def _completed_profiles(state: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return scenario-ordered scalar metrics only after all stages complete."""
    completed = state["completed_stages"]
    if completed != list(_ORDERED_STAGES):
        raise ValueError("CT026 PPC profile did not complete every required stage")
    stored = state["profiles"]
    assert isinstance(stored, dict)
    return {scenario: dict(stored[scenario]) for scenario in _SCENARIOS}


def _profile_document(
    identity: Mapping[str, str],
    state: Mapping[str, object],
    profiles: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Return the final JSON-safe scalar profile document."""
    document = {
        "schema_version": _STATE_SCHEMA_VERSION,
        "identity": dict(identity),
        "shuffle_count": _SHUFFLE_COUNT,
        "completed_stages": list(state["completed_stages"]),
        "phase_profiles": dict(state["phase_profiles"]),
        "profiles": {scenario: dict(metrics) for scenario, metrics in profiles.items()},
    }
    if all(_is_grouped_profile_metrics(metrics) for metrics in profiles.values()):
        document["profile_schema_version"] = _GROUPED_PROFILE_DOCUMENT_SCHEMA_VERSION
        document["profile_provenance"] = {
            "profile_kind": _GROUPED_PROFILE_KIND,
            "scenario_metric_schema_version": _GROUPED_PROFILE_SCHEMA_VERSION,
            "projection_shuffle_counts": "100,1000",
        }
    return document


def _is_grouped_profile_metrics(metrics: Mapping[str, object]) -> bool:
    """Return whether one validated scenario map uses the S6 grouped schema."""
    return (
        metrics.get("schema_version") == _GROUPED_PROFILE_SCHEMA_VERSION
        and metrics.get("profile_kind") == _GROUPED_PROFILE_KIND
    )


def _markdown_summary(
    identity: Mapping[str, str],
    phase_profiles: Mapping[str, object],
    profiles: Mapping[str, Mapping[str, object]],
) -> str:
    """Render a compact scalar-only Markdown summary with metric units unchanged."""
    lines = [
        "# CT026 PPC serial profile",
        "",
        f"- Shuffles per scenario: {_SHUFFLE_COUNT}",
        f"- Config fingerprint: `{identity['config_fingerprint']}`",
        f"- Source fingerprint: `{identity['source_fingerprint']}`",
        f"- Git fingerprint: `{identity['git_fingerprint']}`",
    ]
    for phase_name in ("cold", "warm", "resume_warm"):
        metrics = phase_profiles[phase_name]
        if metrics is None:
            continue
        assert isinstance(metrics, Mapping)
        lines.extend(("", f"## phase {phase_name}", ""))
        for metric_name, value in metrics.items():
            lines.append(f"- {metric_name}: {value}")
    for scenario in _SCENARIOS:
        lines.extend(("", f"## {scenario}", ""))
        for metric_name, value in profiles[scenario].items():
            lines.append(f"- {metric_name}: {value}")
    return "\n".join(lines) + "\n"


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace one UTF-8 JSON object through a UUID temporary sibling."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_atomic(path, text)


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace one text file and remove its temporary sibling on failure."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
