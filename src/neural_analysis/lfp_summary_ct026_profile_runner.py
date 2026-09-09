"""Interruption-safe orchestration for the nonpublishing CT026 PPC profile.

This module owns scalar run state and reports only. Numerical phase preparation,
spike selection, and PPC profiling are injected so the runner cannot publish an
LFP summary component, manifest, or scientific NPZ artifact.
"""

from __future__ import annotations

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
_STATE_SCHEMA_VERSION = "1"


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
        Called as ``prepare_phase(warm=False)`` once for a new cold stage and as
        ``prepare_phase(warm=True)`` for the warm stage. An incomplete resumed
        run skips completed cold work but calls the warm form once to rehydrate
        its prepared mmap-backed phase object.
    select_spikes : callable
        Receives the warm prepared phase and returns caller-owned selected spike
        metadata. It is called once on an incomplete resume even when the saved
        selection stage is complete because in-memory objects are not persisted.
    profile_job : callable
        Receives keyword arguments ``scenario``, ``phase``, ``spikes``,
        ``shuffle_count=100``, and a scenario-specific ``work_root``. It returns
        a string-keyed mapping of finite JSON scalar metrics such as seconds and
        bytes. Interrupted work below ``work_root`` is deliberately preserved.
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
        atomically after every completed ordered stage.

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
    if not callable(prepare_phase) or not callable(select_spikes) or not callable(profile_job):
        raise ValueError("phase preparation, spike selection, and profile job must be callable")

    if run_directory is None:
        timestamp = _profile_timestamp(timestamp_factory)
        active_directory = root / f"{_RUN_PREFIX}{timestamp}"
        _validate_new_run_path(root, active_directory)
        active_directory.mkdir(parents=True, exist_ok=False)
        state = _new_state(identity)
        _write_json_atomic(active_directory / "state.json", state)
    else:
        active_directory = _validated_resume_directory(root, Path(run_directory))
        state = _load_state(active_directory / "state.json", identity)

    completed = state["completed_stages"]
    assert isinstance(completed, list)
    if "phase_cold" not in completed:
        prepare_phase(warm=False)
        _complete_stage(active_directory, state, "phase_cold")

    unfinished_scenarios = [scenario for scenario in _SCENARIOS if scenario not in completed]
    if unfinished_scenarios:
        phase = prepare_phase(warm=True)
        if "phase_warm" not in completed:
            _complete_stage(active_directory, state, "phase_warm")

        spikes = select_spikes(phase)
        if "selection" not in completed:
            _complete_stage(active_directory, state, "selection")

        profiles = state["profiles"]
        assert isinstance(profiles, dict)
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
            profiles[scenario] = _validated_scalar_metrics(metrics, scenario)
            _complete_stage(active_directory, state, scenario)

    profiles = _completed_profiles(state)
    profile_path = active_directory / "profile.json"
    summary_path = active_directory / "summary.md"
    _write_json_atomic(profile_path, _profile_document(identity, state, profiles))
    _write_text_atomic(summary_path, _markdown_summary(identity, profiles))
    return CT026PPCProfileRunResult(
        run_directory=active_directory,
        state_path=active_directory / "state.json",
        profile_path=profile_path,
        summary_path=summary_path,
        profiles=profiles,
    )


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
    return validated


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
    return {
        "schema_version": _STATE_SCHEMA_VERSION,
        "identity": dict(identity),
        "shuffle_count": _SHUFFLE_COUNT,
        "completed_stages": list(state["completed_stages"]),
        "profiles": {scenario: dict(metrics) for scenario, metrics in profiles.items()},
    }


def _markdown_summary(
    identity: Mapping[str, str],
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
