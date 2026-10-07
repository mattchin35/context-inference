"""Prepare, execute, resume, and inspect one task-variable decoding session.

The module deliberately owns lifecycle state only.  Scientific target parsing,
neural activity construction, decoding, and durable result validation remain
in their existing task-decoding modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.neural_analysis.task_decoding import activity, config as decoding_config
from src.neural_analysis.task_decoding import modeling, plotting, results, targets


_RUN_PREFIX = "task_variable_decoding_"
_RESULT_FILE = "results.npz"
_STATE_FILE = "run_state.json"
_EXECUTION_FILE = "execution.json"
_GUARD_FILE = "execution_guard.json"
_RECEIPT_FILE = "local_launch.json"
_COMPACT_TARGET_UNAVAILABLE = "target_outer_unavailable"
_REGIONS = ("PFC", "HPC", "PFC+HPC")
_REPRESENTATIONS = ("pca", "units")
_METRICS = ("balanced_accuracy", "auc", "r2")


def _compact_reason_code(reason: str | None, *, inner: bool = False) -> str:
    """Map model prose to one documented compact saved reason code.

    Parameters
    ----------
    reason : str or None
        Model-level unavailable explanation. It may be human-readable prose.
    inner : bool, default=False
        True for candidate-by-inner-fold audit reasons rather than outer fit
        records.

    Returns
    -------
    str
        Empty text for no reason or a lower-case underscore-delimited code that
        fits the fixed result schema without truncation.
    """
    if not reason:
        return ""
    normalized = str(reason).strip().lower()
    inner_plan_reasons = {
        "inner split unavailable",
        "inner split unavailable: empty outer training subset",
        "inner split unavailable: constant target",
        "inner split unavailable: non-numeric numerical target",
        "inner split unavailable: non-finite numerical target",
        "inner split unavailable: invalid grouped fold count",
        "inner split unavailable: grouped folds split a behavioral block",
        "inner split unavailable: categorical grouped fold lacks both classes",
    }
    if normalized in inner_plan_reasons or normalized.startswith(
        "inner split unavailable: invalid grouped folds: "
    ):
        return "inner_plan_unavailable"
    regional_reasons = {
        ("unavailable inner regional training features: pfc", True): (
            "inner_candidate_unavailable"
        ),
        ("unavailable inner regional training features: hpc", True): (
            "inner_candidate_unavailable"
        ),
        ("unavailable inner regional training features: pfc, hpc", True): (
            "inner_candidate_unavailable"
        ),
        ("unavailable outer regional training features: pfc", False): (
            "outer_pfc_features_unavailable"
        ),
        ("unavailable outer regional training features: hpc", False): (
            "outer_hpc_features_unavailable"
        ),
        ("unavailable outer regional training features: pfc, hpc", False): (
            "outer_features_unavailable"
        ),
        ("unavailable regional training features: pfc", False): (
            "outer_pfc_features_unavailable"
        ),
        ("unavailable regional training features: hpc", False): (
            "outer_hpc_features_unavailable"
        ),
        ("unavailable regional training features: pfc, hpc", False): (
            "outer_features_unavailable"
        ),
    }
    regional_code = regional_reasons.get((normalized, inner))
    if regional_code is not None:
        return regional_code
    direct_aliases = {
        "target outer unavailable": "target_outer_unavailable",
        "target_outer_unavailable": "target_outer_unavailable",
        "no valid inner tuning candidate": "no_valid_tuning_candidate",
        "no_valid_tuning_candidate": "no_valid_tuning_candidate",
        "candidate fit failed": "candidate_fit_failed",
        "candidate_fit_failed": "candidate_fit_failed",
        "non-finite inner score": "nonfinite_inner_score",
        "nonfinite inner score": "nonfinite_inner_score",
        "nonfinite_inner_score": "nonfinite_inner_score",
        "inner plan unavailable": "inner_plan_unavailable",
        "inner_plan_unavailable": "inner_plan_unavailable",
        "inner class coverage unavailable": "inner_class_coverage_unavailable",
        "inner_class_coverage_unavailable": "inner_class_coverage_unavailable",
        "pfc transform unavailable": "pfc_transform_unavailable",
        "pfc_transform_unavailable": "pfc_transform_unavailable",
        "pfc pca transform unavailable": "pfc_pca_transform_unavailable",
        "pfc_pca_transform_unavailable": "pfc_pca_transform_unavailable",
        "outer estimator failure": "outer_estimator_failure",
        "outer_estimator_failure": "outer_estimator_failure",
        "fit convergence failure": "fit_convergence_failure",
        "fit_convergence_failure": "fit_convergence_failure",
        "no features": "no_features",
        "no_features": "no_features",
    }
    if normalized in direct_aliases:
        return direct_aliases[normalized]
    if normalized == "convergence_warning":
        return "candidate_fit_failed" if inner else "fit_convergence_failure"
    estimator_failures = {
        "invalid coefficient or intercept shape",
        "non-finite coefficient or intercept",
        "categorical estimator must expose binary classes including positive class 1",
        "invalid probability shape",
        "non-finite probability or score",
        "positive class is ambiguous",
        "positive_scores must be finite and align with target_values.",
        "categorical metrics require held-out values from both classes.",
        "non-finite prediction or score",
        "constant, singleton, or non-finite held-out numerical target",
    }
    if normalized in estimator_failures:
        return "candidate_fit_failed" if inner else "outer_estimator_failure"
    raise ValueError(f"Unknown or unsupported model reason: {reason!r}")


def _canonical_json(value: object) -> str:
    """Encode JSON-safe metadata deterministically.

    Parameters
    ----------
    value : object
        JSON-compatible scalar, sequence, or mapping with no NumPy arrays.

    Returns
    -------
    str
        Stable UTF-8 JSON text for immutable sidecars and provenance records.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_now() -> str:
    """Return the current UTC lifecycle timestamp.

    Returns
    -------
    str
        Whole-second ISO-8601 UTC timestamp with a trailing ``Z``.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_timestamp() -> str:
    """Return a filesystem-safe UTC suffix for an immutable run directory.

    Returns
    -------
    str
        Whole-second UTC timestamp with colon characters replaced by hyphens.
    """
    return _utc_now().replace(":", "-")


def _repository_root() -> Path:
    """Locate the checked-out repository that owns scientific source identity.

    Returns
    -------
    pathlib.Path
        Absolute repository root three parents above this module.
    """
    return Path(__file__).resolve().parents[3]


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace one small lifecycle artifact with fully written bytes.

    Parameters
    ----------
    path : pathlib.Path
        Destination in an existing or creatable run subdirectory.
    payload : bytes
        Complete UTF-8 JSON, text, or binary PNG payload without physical-unit
        conversion.

    Returns
    -------
    None
        Writes a unique sibling temporary, fsyncs it, and calls ``os.replace``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException as publication_error:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except BaseException as cleanup_error:
                raise cleanup_error from publication_error
        raise


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write one small canonical JSON mapping.

    Parameters
    ----------
    path : pathlib.Path
        Lifecycle JSON destination.
    value : mapping[str, object]
        JSON-safe state, execution, manifest, or provenance values.

    Returns
    -------
    None
        Publishes canonical UTF-8 JSON with a trailing newline.
    """
    _atomic_bytes(path, (_canonical_json(value) + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required small JSON object.

    Parameters
    ----------
    path : pathlib.Path
        Existing JSON sidecar, state, execution, guard, or receipt file.

    Returns
    -------
    dict[str, object]
        Decoded JSON object without array axes or physical-unit conversion.

    Raises
    ------
    ValueError
        If the file is absent, malformed, or does not contain a JSON object.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read required JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Required JSON artifact must be an object: {path.name}")
    return value


def _append_log(run_directory: Path, message: str) -> None:
    """Append and flush one durable lifecycle log line.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory.
    message : str
        Human-readable stage, target, warning, or error text without units.

    Returns
    -------
    None
        Flushes one UTF-8 line to ``run.log``.
    """
    with (run_directory / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{_utc_now()} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _runtime_version_mapping() -> dict[str, str]:
    """Collect named runtime package versions for execution-only provenance.

    Returns
    -------
    dict[str, str]
        Python and required scientific package versions. Values are strings and
        are compared at resume before any large neural input is loaded.
    """
    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "pynapple": "pynapple",
        "matplotlib": "matplotlib",
    }
    versions = {"python": platform.python_version()}
    for label, distribution in packages.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "unavailable"
    return versions


def _source_files_for_manifest(resolved_session: object) -> dict[str, tuple[Path, ...]]:
    """List explicit small and binary probe inputs opened by the activity stage.

    Parameters
    ----------
    resolved_session : src.neural_analysis.session_metadata.ResolvedSession
        Resolved session metadata with absolute contained probe source paths.

    Returns
    -------
    dict[str, tuple[pathlib.Path, ...]]
        Probe-keyed explicit source lists in deterministic path order. Binary
        files retain stat-only manifest identities under the results policy.
    """
    file_lists: dict[str, tuple[Path, ...]] = {}
    for source in resolved_session.probes:
        paths = (
            source.alignment_file,
            source.channel_quality_file,
            source.cluster_info_file,
            source.spike_clusters_file,
            source.spike_times_file,
        )
        existing = sorted({Path(path) for path in paths if path is not None and path.is_file()})
        file_lists[f"{source.probe_id}_sources"] = tuple(existing)
    return file_lists


def _load_small_inputs(
    config_path: Path | str,
) -> tuple[decoding_config.TaskDecodingConfig, object, pd.DataFrame, pd.DataFrame]:
    """Load configuration, resolved metadata, CSV table, and target table only.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Task-decoding JSON configuration path.

    Returns
    -------
    tuple
        ``(config, resolved_session, trial_table, target_table)``. Tables have
        chronological rows; target values remain encoded classes or native
        numerical units. No spike arrays, tensors, or model fits are loaded.
    """
    config = decoding_config.load_task_decoding_config(config_path)
    resolved = resolve_session_metadata(
        load_session_metadata(config.session_metadata_path),
        config.session_metadata_path,
    )
    trial_table = pd.read_csv(config.augmented_trial_path)
    target_table = targets.build_target_table(trial_table, config)
    target_table[config.alignment] = trial_table[config.alignment].to_numpy(dtype=float)
    return config, resolved, trial_table, target_table


def _build_input_manifest(
    config: decoding_config.TaskDecodingConfig,
    resolved_session: object,
) -> dict[str, object]:
    """Build the exact portable manifest used for prepared-run revalidation.

    Parameters
    ----------
    config : TaskDecodingConfig
        Validated paths and scientific controls for one session.
    resolved_session : ResolvedSession
        Metadata-resolved probe sources used by the activity stage.

    Returns
    -------
    dict[str, object]
        Results-module manifest with explicit contained inputs, path/size/mtime,
        and allowed small structured hashes.
    """
    return results.build_input_manifest(
        config.session_root,
        {
            "neural_session": config.session_metadata_path,
            "augmented_trials": config.augmented_trial_path,
            "trial_feature_parameters": config.trial_feature_parameter_path,
        },
        _source_files_for_manifest(resolved_session),
    )


def _build_preparation_identity(config_path: Path | str) -> dict[str, object]:
    """Build bounded scientific identity before allocating neural tensors.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Session-local task-decoding JSON configuration.

    Returns
    -------
    dict[str, object]
        Validated config, resolved metadata, small tables, scientific config,
        input/source manifests, and full fingerprint. Target values retain
        their configured encoded/native units.
    """
    config, resolved, trial_table, target_table = _load_small_inputs(config_path)
    scientific_config = decoding_config.scientific_config_payload(config)
    input_manifest = _build_input_manifest(config, resolved)
    scientific_source = results.scientific_source_fingerprint(_repository_root())
    fingerprint = results.build_run_fingerprint(
        decoding_config.ANALYSIS_VERSION,
        str(scientific_source["fingerprint"]),
        scientific_config,
        input_manifest,
        resolved.session_id,
    )
    return {
        "config": config,
        "resolved_session": resolved,
        "trial_table": trial_table,
        "target_table": target_table,
        "scientific_config": scientific_config,
        "input_manifest": input_manifest,
        "scientific_source": scientific_source,
        "fingerprint": fingerprint,
    }


def _fit_count(config: decoding_config.TaskDecodingConfig) -> int:
    """Count outer fixed-model cells for a bounded dry-run report.

    Parameters
    ----------
    config : TaskDecodingConfig
        Scientific settings containing target, fold, and bin-width controls.

    Returns
    -------
    int
        Number of target-by-fold-by-time-by-region-by-representation outer
        cells. The count is dimensionless and excludes tuned inner candidates.
    """
    time_bins = int(
        round(
            (decoding_config.WINDOW_END_SECONDS - decoding_config.WINDOW_START_SECONDS)
            * 1000
            / config.bin_width_ms
        )
    )
    return len(config.target_names) * config.outer_fold_count * time_bins * 3 * 2


def _resource_envelope(
    config: decoding_config.TaskDecodingConfig,
    target_table: pd.DataFrame,
    report: activity.ActivityDryRunReport,
) -> dict[str, int]:
    """Return comparable resource-driving dimensions for one bounded plan.

    Parameters
    ----------
    config : TaskDecodingConfig
        Frozen target, fold, bin-width, and regularization settings.
    target_table : pandas.DataFrame
        Full chronological target table, shape ``(full_trial, columns)``.
    report : ActivityDryRunReport
        Target-independent bilateral tensor dimensions and exact allocation.

    Returns
    -------
    dict[str, int]
        Nonnegative dimensionless counts plus ``tensor_allocation_bytes``.
        Family fit counts include inner candidates and outer refits in tuned
        mode and outer fits only in fixed mode.
    """
    family_counts = {"categorical": 0, "numerical": 0}
    for target_name in config.target_names:
        family = decoding_config.TARGET_DEFINITION_BY_IDENTIFIER[target_name].family
        family_counts[family] += 1
    cells_per_target = config.outer_fold_count * report.time_bin_count * 3 * 2
    fits_per_cell = (
        15 * config.inner_fold_count + 1
        if config.regularization_mode == "tuned"
        else 1
    )
    return {
        "full_trial_count": int(target_table.shape[0]),
        "tensor_trial_count": report.tensor_trial_count,
        "pfc_unit_count": report.pfc_unit_count,
        "hpc_unit_count": report.hpc_unit_count,
        "time_bin_count": report.time_bin_count,
        "target_count": len(config.target_names),
        "outer_fold_count": config.outer_fold_count,
        "inner_fold_count": config.inner_fold_count,
        "coefficient_feature_capacity": report.pfc_unit_count + report.hpc_unit_count,
        "categorical_fit_count": family_counts["categorical"]
        * cells_per_target
        * fits_per_cell,
        "numerical_fit_count": family_counts["numerical"]
        * cells_per_target
        * fits_per_cell,
        "tensor_allocation_bytes": report.tensor_allocation_bytes,
    }


def plan_task_decoding_session(config_path: Path | str) -> dict[str, object]:
    """Inspect one session's bounded inputs without loading spikes or fitting models.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Task-decoding JSON configuration.

    Returns
    -------
    dict[str, object]
        Target names, identity/environment facts, outer-cell fit count, exact
        tensor allocation bytes, source-file size facts, comparable resource
        dimensions, and scientific-source cleanliness text. Firing-rate arrays
        are not allocated or returned.
    """
    identity = _build_preparation_identity(config_path)
    config = identity["config"]
    resolved = identity["resolved_session"]
    target_table = identity["target_table"]
    scientific_source = identity["scientific_source"]
    report = activity.inspect_activity_dry_run(
        target_table,
        config.pfc_region,
        config.hpc_region,
        resolved.probes,
        alignment_event=config.alignment,
        window_s=(decoding_config.WINDOW_START_SECONDS, decoding_config.WINDOW_END_SECONDS),
        bin_width_s=config.bin_width_ms / 1000.0,
        pfc_trusted_utc_bounds=config.trusted_utc_bounds.get(config.pfc_region.probe_id),
        hpc_trusted_utc_bounds=config.trusted_utc_bounds.get(config.hpc_region.probe_id),
    )
    try:
        results.validate_scientific_source_cleanliness(_repository_root())
    except ValueError as error:
        source_cleanliness = str(error)
    else:
        source_cleanliness = "clean"
    return {
        "config_path": str(Path(config_path).resolve()),
        "session_id": resolved.session_id,
        "target_names": config.target_names,
        "fit_count": _fit_count(config),
        "tensor_allocation_bytes": report.tensor_allocation_bytes,
        "source_file_sizes_bytes": report.source_file_sizes_bytes,
        "source_cleanliness": source_cleanliness,
        "analysis_version": decoding_config.ANALYSIS_VERSION,
        "scientific_source_fingerprint": scientific_source["fingerprint"],
        "runtime_versions": _runtime_version_mapping(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        },
        "regularization_mode": config.regularization_mode,
        "resource_envelope": _resource_envelope(config, target_table, report),
    }


def _new_run_directory(output_root: Path) -> Path:
    """Create one exclusive timestamped immutable run directory.

    Parameters
    ----------
    output_root : pathlib.Path
        Existing or new session-local run parent outside the source checkout.

    Returns
    -------
    pathlib.Path
        Newly created directory named with a UTC timestamp and, on collision,
        the first available two-digit suffix.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    base = f"{_RUN_PREFIX}{_utc_timestamp()}"
    for suffix in range(1000):
        name = base if suffix == 0 else f"{base}-{suffix:02d}"
        candidate = output_root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not claim a unique task-decoding run directory.")


def _saved_command_text(run_directory: Path) -> tuple[str, str]:
    """Build portable public resume and status commands for one immutable run.

    Parameters
    ----------
    run_directory : pathlib.Path
        Absolute immutable run path, possibly containing shell-special text.

    Returns
    -------
    tuple[str, str]
        Newline-terminated resume-with-detach and status commands. Shell path
        quoting preserves the exact absolute run directory.
    """
    import shlex

    prefix = "uv run python -m src.neural_analysis.task_decoding.run_session"
    quoted = shlex.quote(str(run_directory))
    return (
        f"{prefix} resume --run-directory {quoted} --detach\n",
        f"{prefix} status --run-directory {quoted}\n",
    )


def _initial_execution_provenance(
    *,
    config_path: Path,
    execution_mode: str,
) -> dict[str, object]:
    """Create execution-only provenance for one prepared immutable run.

    Parameters
    ----------
    config_path : pathlib.Path
        Original local configuration path used solely to rebuild identity on
        execution; it is not included in the scientific fingerprint.
    execution_mode : {"foreground", "detached", "slurm"}
        Requested launch route.

    Returns
    -------
    dict[str, object]
        JSON-safe host, process, version, thread-limit, config-path, and
        guarded-execution history facts.
    """
    return {
        "mode": execution_mode,
        "config_path": str(config_path),
        "pid": os.getpid(),
        "host": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        },
        "runtime_versions": _runtime_version_mapping(),
        "history": [],
    }


def _write_prepared_sidecars(
    run_directory: Path,
    identity: Mapping[str, object],
    *,
    config_path: Path,
    execution_mode: str,
) -> None:
    """Write all bounded immutable preparation artifacts before neural loading.

    Parameters
    ----------
    run_directory : pathlib.Path
        Newly exclusive run directory.
    identity : mapping[str, object]
        Output of :func:`_build_preparation_identity`.
    config_path : pathlib.Path
        Original configuration path retained only in execution provenance.
    execution_mode : {"foreground", "detached", "slurm"}
        Requested execution route.

    Returns
    -------
    None
        Publishes configuration, manifest, source identity, exact feature bytes,
        state, provenance, logs, and resume/status commands atomically.
    """
    scientific_config = identity["scientific_config"]
    input_manifest = identity["input_manifest"]
    scientific_source = identity["scientific_source"]
    config = identity["config"]
    # WP5A owns these byte-identical immutable result sidecars and writes them
    # without a newline, so late final publication can validate/reuse them.
    _atomic_bytes(
        run_directory / "config.json",
        _canonical_json(scientific_config).encode("utf-8"),
    )
    _atomic_bytes(
        run_directory / "input_manifest.json",
        _canonical_json(input_manifest).encode("utf-8"),
    )
    _write_json(run_directory / "scientific_source.json", scientific_source)
    feature_destination = run_directory / "trial_feature_params.json"
    _atomic_bytes(feature_destination, config.trial_feature_parameter_path.read_bytes())
    _atomic_bytes(
        run_directory / "run_fingerprint.txt",
        (str(identity["fingerprint"]) + "\n").encode("ascii"),
    )
    resume_command, status_command = _saved_command_text(run_directory)
    _atomic_bytes(run_directory / "resume_command.txt", resume_command.encode("utf-8"))
    _atomic_bytes(run_directory / "status_command.txt", status_command.encode("utf-8"))
    _atomic_bytes(
        run_directory / "run_session.py",
        Path(__file__).with_name("run_session.py").read_bytes(),
    )
    _atomic_bytes(
        run_directory / "run_batch.py",
        Path(__file__).with_name("run_batch.py").read_bytes(),
    )
    _atomic_bytes(run_directory / "pipeline.py", Path(__file__).read_bytes())
    (run_directory / "checkpoints").mkdir(exist_ok=True)
    (run_directory / "figures").mkdir(exist_ok=True)
    _write_json(
        run_directory / _EXECUTION_FILE,
        _initial_execution_provenance(
            config_path=config_path.resolve(),
            execution_mode=execution_mode,
        ),
    )
    state = {
        "lifecycle": "initialized",
        "current_stage": "prepared",
        "started_at": "",
        "updated_at": _utc_now(),
        "completed_at": "",
        "completed_targets": [],
        "total_targets": list(config.target_names),
        "warnings": [],
        "last_error": "",
        "final_results_published": False,
    }
    _write_json(run_directory / _STATE_FILE, state)
    _append_log(run_directory, "stage prepared")


def _record_batch_execution_provenance(
    run_directory: Path,
    batch_provenance: Mapping[str, object],
) -> None:
    """Atomically add batch-only worker and resource evidence to a prepared run.

    Parameters
    ----------
    run_directory : pathlib.Path
        Newly prepared immutable scientific run directory.
    batch_provenance : mapping[str, object]
        JSON-safe config-list, worker-cap, memory, and optional evidence facts.
        Byte-valued members explicitly end in ``_bytes``.

    Returns
    -------
    None
        Updates only execution provenance; scientific identity is unchanged.
    """
    execution = _read_json(run_directory / _EXECUTION_FILE)
    if "batch" in execution:
        raise ValueError("Prepared execution already contains batch provenance.")
    execution["batch"] = dict(batch_provenance)
    _write_json(run_directory / _EXECUTION_FILE, execution)


def _validate_complete_prepared_run(run_directory: Path) -> bool:
    """Return whether one matching directory is both complete and loadable.

    Parameters
    ----------
    run_directory : pathlib.Path
        Candidate prepared run directory.

    Returns
    -------
    bool
        True only for lifecycle ``complete`` with a fully valid result schema.
    """
    try:
        state = _read_json(run_directory / _STATE_FILE)
        if state.get("lifecycle") != "complete" or not state.get("final_results_published"):
            return False
        _validate_published_run_directory(run_directory)
    except (OSError, ValueError):
        return False
    return True


def prepare_task_decoding_run(
    config_path: Path | str,
    rerun: bool,
    execution_mode: str,
) -> Path:
    """Prepare an immutable session run after bounded scientific identity checks.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Existing task-decoding JSON configuration.
    rerun : bool
        If false, return one matching valid complete run or refuse ambiguous
        recovery candidates; if true, always create a new immutable directory.
    execution_mode : {"foreground", "detached", "slurm"}
        Execution-only route saved in provenance, never scientific identity.

    Returns
    -------
    pathlib.Path
        Existing validated complete matching run or a newly prepared run.

    Raises
    ------
    ValueError
        If configuration, source cleanliness, or bounded identity validation fails.
    RuntimeError
        If recovery candidates exist and ``rerun`` was not requested.
    """
    if execution_mode not in {"foreground", "detached", "slurm"}:
        raise ValueError("execution_mode must be foreground, detached, or slurm.")
    results.validate_scientific_source_cleanliness(_repository_root())
    # Read the configured output root separately so discovery stays testable from
    # the one identity field it actually needs: the scientific fingerprint.
    config = decoding_config.load_task_decoding_config(config_path)
    identity = _build_preparation_identity(config_path)
    if not rerun and config.output_root.exists():
        matching: list[Path] = []
        for candidate in sorted(config.output_root.glob(f"{_RUN_PREFIX}*")):
            if not candidate.is_dir():
                continue
            try:
                fingerprint = (
                    (candidate / "run_fingerprint.txt").read_text(encoding="ascii").strip()
                )
            except OSError:
                continue
            if fingerprint == identity["fingerprint"]:
                matching.append(candidate)
        completed = [
            candidate for candidate in matching if _validate_complete_prepared_run(candidate)
        ]
        if completed:
            return completed[0]
        if matching:
            names = " ".join(candidate.name for candidate in matching)
            raise RuntimeError(
                "Matching task-decoding runs need explicit rerun or recovery: " + names
            )
    run_directory = _new_run_directory(config.output_root)
    try:
        _write_prepared_sidecars(
            run_directory,
            identity,
            config_path=Path(config_path),
            execution_mode=execution_mode,
        )
    except BaseException:
        # Preparation is the one stage allowed to remove an entirely new failed directory.
        shutil.rmtree(run_directory, ignore_errors=True)
        raise
    return run_directory


def _split_to_payload(split: modeling.GroupedSplit) -> dict[str, object]:
    """Convert one grouped split to JSON-safe checkpoint primitives.

    Parameters
    ----------
    split : modeling.GroupedSplit
        One fold with one-dimensional global matched-row integer positions.

    Returns
    -------
    dict[str, object]
        Fold identifier plus integer train/test lists with dimensionless row axes.
    """
    return {
        "fold_id": split.fold_id,
        "train_indices": split.train_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
    }


def _split_from_payload(value: Mapping[str, object]) -> modeling.GroupedSplit:
    """Restore one grouped split from checkpoint JSON primitives.

    Parameters
    ----------
    value : mapping[str, object]
        Fold payload emitted by :func:`_split_to_payload`.

    Returns
    -------
    modeling.GroupedSplit
        Integer train/test matched-row arrays with their saved fold identity.
    """
    return modeling.GroupedSplit(
        fold_id=int(value["fold_id"]),
        train_indices=np.asarray(value["train_indices"], dtype=np.int64),
        test_indices=np.asarray(value["test_indices"], dtype=np.int64),
    )


def _plan_to_payload(plan: modeling.SplitPlan) -> dict[str, object]:
    """Convert one grouped split plan to safe checkpoint primitives.

    Parameters
    ----------
    plan : modeling.SplitPlan
        Available or unavailable grouped plan aligned to a target's matched rows.

    Returns
    -------
    dict[str, object]
        Fold identifiers, availability/reason, and all grouped train/test splits.
    """
    return {
        "fold_ids": plan.fold_ids.tolist(),
        "splits": [_split_to_payload(split) for split in plan.splits],
        "is_available": plan.is_available,
        "unavailable_reason": plan.unavailable_reason,
    }


def _plan_from_payload(value: Mapping[str, object]) -> modeling.SplitPlan:
    """Restore one available or unavailable split plan from checkpoint JSON.

    Parameters
    ----------
    value : mapping[str, object]
        Split-plan JSON payload emitted by :func:`_plan_to_payload`.

    Returns
    -------
    modeling.SplitPlan
        Global matched-row fold IDs and deterministic grouped split records.
    """
    return modeling.SplitPlan(
        fold_ids=np.asarray(value["fold_ids"], dtype=np.int64),
        splits=tuple(_split_from_payload(split) for split in value["splits"]),
        is_available=bool(value["is_available"]),
        unavailable_reason=value["unavailable_reason"],
    )


def _fold_record_to_payload(record: modeling.FoldRecord) -> dict[str, object]:
    """Convert one audited fit record to JSON-safe checkpoint values.

    Parameters
    ----------
    record : modeling.FoldRecord
        Target/time/region/representation/fold result. Coefficients are
        standardized model-scale floats with NaN values retained as JSON NaN.

    Returns
    -------
    dict[str, object]
        All record provenance, counts, scores, coefficients, and candidate audits.
    """
    return {
        "record_key": list(record.record_key),
        "target_identifier": record.target_identifier,
        "inner_fold_count": record.inner_fold_count,
        "outer_fold_id": record.outer_fold_id,
        "outer_test_indices": record.outer_test_indices.tolist(),
        "inner_selection_indices": record.inner_selection_indices.tolist(),
        "time_bin_index": record.time_bin_index,
        "region_configuration": record.region_configuration,
        "representation": record.representation,
        "regularization_mode": record.regularization_mode,
        "is_valid": record.is_valid,
        "status": record.status,
        "reason": record.reason,
        "train_count": record.train_count,
        "test_count": record.test_count,
        "train_class_counts": record.train_class_counts,
        "test_class_counts": record.test_class_counts,
        "requested_feature_count": record.requested_feature_count,
        "effective_feature_count": record.effective_feature_count,
        "estimator_class": record.estimator_class,
        "estimator_parameters": record.estimator_parameters,
        "convergence_status": record.convergence_status,
        "metrics": record.metrics,
        "coefficients": record.coefficients.tolist(),
        "intercept": record.intercept,
        "feature_ids": list(record.feature_ids),
        "positive_class": record.positive_class,
        "coefficient_scale_label": record.coefficient_scale_label,
        "selected_parameters": record.selected_parameters,
        "candidate_inner_scores": [list(row) for row in record.candidate_inner_scores],
        "candidate_inner_statuses": [list(row) for row in record.candidate_inner_statuses],
        "candidate_inner_reasons": [list(row) for row in record.candidate_inner_reasons],
        "selected_candidate_index": record.selected_candidate_index,
    }


def _counts_from_payload(value: object) -> dict[int, int] | None:
    """Restore optional categorical count mappings with integer class keys.

    Parameters
    ----------
    value : object
        JSON null or a mapping whose JSON keys represent classes zero and one.

    Returns
    -------
    dict[int, int] or None
        Original categorical count mapping, or None for numerical targets.
    """
    if value is None:
        return None
    return {int(key): int(count) for key, count in dict(value).items()}


def _fold_record_from_payload(value: Mapping[str, object]) -> modeling.FoldRecord:
    """Restore one full audited fold record from JSON checkpoint primitives.

    Parameters
    ----------
    value : mapping[str, object]
        Payload emitted by :func:`_fold_record_to_payload`.

    Returns
    -------
    modeling.FoldRecord
        Record preserving global row identities, model units, and candidate audits.
    """
    return modeling.FoldRecord(
        record_key=tuple(value["record_key"]),
        target_identifier=str(value["target_identifier"]),
        inner_fold_count=int(value["inner_fold_count"]),
        outer_fold_id=int(value["outer_fold_id"]),
        outer_test_indices=np.asarray(value["outer_test_indices"], dtype=np.int64),
        inner_selection_indices=np.asarray(value["inner_selection_indices"], dtype=np.int64),
        time_bin_index=int(value["time_bin_index"]),
        region_configuration=str(value["region_configuration"]),
        representation=str(value["representation"]),
        regularization_mode=str(value["regularization_mode"]),
        is_valid=bool(value["is_valid"]),
        status=str(value["status"]),
        reason=value["reason"],
        train_count=int(value["train_count"]),
        test_count=int(value["test_count"]),
        train_class_counts=_counts_from_payload(value["train_class_counts"]),
        test_class_counts=_counts_from_payload(value["test_class_counts"]),
        requested_feature_count=int(value["requested_feature_count"]),
        effective_feature_count=int(value["effective_feature_count"]),
        estimator_class=value["estimator_class"],
        estimator_parameters=dict(value["estimator_parameters"]),
        convergence_status=str(value["convergence_status"]),
        metrics={str(key): float(score) for key, score in dict(value["metrics"]).items()},
        coefficients=np.asarray(value["coefficients"], dtype=float),
        intercept=float(value["intercept"]),
        feature_ids=tuple(str(item) for item in value["feature_ids"]),
        positive_class=value["positive_class"],
        coefficient_scale_label=str(value["coefficient_scale_label"]),
        selected_parameters=(
            None if value["selected_parameters"] is None else dict(value["selected_parameters"])
        ),
        candidate_inner_scores=tuple(
            tuple(row) for row in value["candidate_inner_scores"]
        ),
        candidate_inner_statuses=tuple(
            tuple(str(item) for item in row) for row in value["candidate_inner_statuses"]
        ),
        candidate_inner_reasons=tuple(
            tuple(item for item in row) for row in value["candidate_inner_reasons"]
        ),
        selected_candidate_index=value["selected_candidate_index"],
    )


def _target_result_to_checkpoint_arrays(
    target_result: modeling.TargetDecodingResult,
) -> dict[str, np.ndarray]:
    """Encode one complete target result into safe primitive checkpoint members.

    Parameters
    ----------
    target_result : modeling.TargetDecodingResult
        Completed fixed or tuned target, including one-dimensional fold arrays,
        model-scale coefficients, and nested candidate audit values.

    Returns
    -------
    dict[str, numpy.ndarray]
        Non-object NPZ members. The scalar Unicode JSON member preserves all
        target records without changing axes or units.
    """
    payload = {
        "target_identifier": target_result.target_identifier,
        "inner_fold_count": target_result.inner_fold_count,
        "is_available": target_result.is_available,
        "status": target_result.status,
        "unavailable_reason": target_result.unavailable_reason,
        "outer_fold_ids": target_result.outer_fold_ids.tolist(),
        "inner_split_plans": {
            str(key): _plan_to_payload(plan)
            for key, plan in target_result.inner_split_plans.items()
        },
        "fold_records": [_fold_record_to_payload(record) for record in target_result.fold_records],
        "cell_summaries": [
            {
                "key": list(key),
                "is_available": summary.is_available,
                "primary_mean": summary.primary_mean,
                "valid_fold_count": summary.valid_fold_count,
                "expected_fold_count": summary.expected_fold_count,
                "fold_scores": list(summary.fold_scores),
                "unavailable_reason": summary.unavailable_reason,
            }
            for key, summary in target_result.cell_summaries.items()
        ],
        "candidate_inner_scores": [
            {"key": list(key), "value": [list(row) for row in value]}
            for key, value in target_result.candidate_inner_scores.items()
        ],
        "candidate_inner_statuses": [
            {"key": list(key), "value": [list(row) for row in value]}
            for key, value in target_result.candidate_inner_statuses.items()
        ],
        "candidate_inner_reasons": [
            {"key": list(key), "value": [list(row) for row in value]}
            for key, value in target_result.candidate_inner_reasons.items()
        ],
        "selected_candidate_indices": [
            {"key": list(key), "value": value}
            for key, value in target_result.selected_candidate_indices.items()
        ],
    }
    return {"target_result_json": np.array(_canonical_json(payload))}


def _target_result_from_checkpoint(saved: Mapping[str, object]) -> modeling.TargetDecodingResult:
    """Restore one full target result from a fingerprint-validated checkpoint.

    Parameters
    ----------
    saved : mapping[str, object]
        Value returned by :func:`results.load_target_checkpoint` with a safe
        Unicode payload and exact target fingerprint already checked.

    Returns
    -------
    modeling.TargetDecodingResult
        Fixed or tuned target result preserving all global row axes, fit units,
        nested grouped plans, and candidate audits.
    """
    arrays = saved["target_arrays"]
    try:
        raw = arrays["target_result_json"]
        if raw.dtype.kind != "U" or raw.shape != ():
            raise ValueError("Target checkpoint result payload is invalid.")
        value = json.loads(str(raw.item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Target checkpoint result payload is invalid.") from error
    if not isinstance(value, Mapping):
        raise ValueError("Target checkpoint result payload is invalid.")
    if str(value["target_identifier"]) != saved["target_label"]:
        raise ValueError("Target checkpoint label disagrees with payload.")
    plans = {
        int(key): _plan_from_payload(plan)
        for key, plan in dict(value["inner_split_plans"]).items()
    }
    summaries = {
        tuple(item["key"]): modeling.CellSummary(
            is_available=bool(item["is_available"]),
            primary_mean=item["primary_mean"],
            valid_fold_count=int(item["valid_fold_count"]),
            expected_fold_count=int(item["expected_fold_count"]),
            fold_scores=tuple(item["fold_scores"]),
            unavailable_reason=item["unavailable_reason"],
        )
        for item in value["cell_summaries"]
    }

    def keyed(entries: Sequence[Mapping[str, object]]) -> dict[tuple, object]:
        """Restore one tuple-keyed audit mapping from its JSON entry sequence."""
        return {tuple(item["key"]): item["value"] for item in entries}

    return modeling.TargetDecodingResult(
        target_identifier=str(value["target_identifier"]),
        inner_fold_count=int(value["inner_fold_count"]),
        is_available=bool(value["is_available"]),
        status=str(value["status"]),
        unavailable_reason=value["unavailable_reason"],
        outer_fold_ids=np.asarray(value["outer_fold_ids"], dtype=np.int64),
        inner_split_plans=plans,
        fold_records=tuple(_fold_record_from_payload(record) for record in value["fold_records"]),
        cell_summaries=summaries,
        candidate_inner_scores={
            key: tuple(tuple(row) for row in audit)
            for key, audit in keyed(value["candidate_inner_scores"]).items()
        },
        candidate_inner_statuses={
            key: tuple(tuple(str(item) for item in row) for row in audit)
            for key, audit in keyed(value["candidate_inner_statuses"]).items()
        },
        candidate_inner_reasons={
            key: tuple(tuple(item for item in row) for row in audit)
            for key, audit in keyed(value["candidate_inner_reasons"]).items()
        },
        selected_candidate_indices={
            key: value for key, value in keyed(value["selected_candidate_indices"]).items()
        },
    )


def _process_start_token(pid: int | None) -> str | None:
    """Read a Linux process-start token without polling or signalling a process.

    Parameters
    ----------
    pid : int or None
        Positive local process identifier, if a local process was recorded.

    Returns
    -------
    str or None
        Stable Linux ``/proc`` start-time token, or None when unavailable/dead.
    """
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        contents = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # ``comm`` is parenthesized and may contain spaces, so ordinary whitespace
    # splitting would shift field 22 (the process start time).
    closing_parenthesis = contents.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields_after_comm = contents[closing_parenthesis + 1 :].split()
    return fields_after_comm[19] if len(fields_after_comm) > 19 else None


def _scheduler_state(job_id: object) -> str:
    """Read one Slurm job state without establishing a poll loop.

    Parameters
    ----------
    job_id : object
        Scheduler job identifier from a saved execution owner record.

    Returns
    -------
    str
        Uppercase scheduler state, or ``"UNKNOWN"`` when no one-shot lookup
        can produce a state.
    """
    if not isinstance(job_id, str) or not job_id:
        return "UNKNOWN"
    try:
        completed = subprocess.run(
            ["squeue", "--noheader", "--format=%T", "--jobs", job_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    if completed.returncode != 0:
        return "UNKNOWN"
    state = completed.stdout.strip().splitlines()
    return state[0].strip().upper() if state else "UNKNOWN"


def _inspect_execution_owner(
    owner: Mapping[str, object],
    *,
    process_start_token: Any = _process_start_token,
    scheduler_state: Any = _scheduler_state,
) -> str:
    """Classify one saved guard or detached receipt using one bounded fact lookup.

    Parameters
    ----------
    owner : mapping[str, object]
        Owner JSON with mode, host, PID, start token, timestamp, and optional
        Slurm job ID. Values are execution facts, not scientific units.
    process_start_token : callable, optional
        Local PID-to-start-token reader used once for same-host processes.
    scheduler_state : callable, optional
        Slurm job-ID-to-state reader used once for Slurm owners.

    Returns
    -------
    str
        ``live``, ``pid_reused``, ``dead``, ``slurm_pending``,
        ``slurm_running``, ``slurm_terminal``, or ``foreign_unresolved``.
    """
    mode = owner.get("mode")
    if mode == "slurm":
        state = str(scheduler_state(owner.get("job_id"))).upper()
        if state in {"PENDING", "CONFIGURING", "SUSPENDED"}:
            return "slurm_pending"
        if state in {"RUNNING", "COMPLETING"}:
            return "slurm_running"
        if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            return "slurm_terminal"
        return "foreign_unresolved"
    if owner.get("host") != platform.node():
        return "foreign_unresolved"
    current = process_start_token(owner.get("pid"))
    if current is None:
        return "dead"
    return "live" if current == owner.get("start_token") else "pid_reused"


def _append_execution_history(run_directory: Path, event: str) -> None:
    """Append a timestamped guard lifecycle event to execution provenance.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory.
    event : str
        Concise execution/guard event without scientific data values.

    Returns
    -------
    None
        Republishes the small execution JSON atomically.
    """
    execution = _read_json(run_directory / _EXECUTION_FILE)
    history = execution.get("history")
    if not isinstance(history, list):
        history = []
    history.append({"timestamp": _utc_now(), "event": event})
    execution["history"] = history
    _write_json(run_directory / _EXECUTION_FILE, execution)


def _claim_execution_guard(run_directory: Path, *, owner: Mapping[str, object]) -> Path:
    """Atomically claim sole execution ownership for one prepared run.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory.
    owner : mapping[str, object]
        Current mode/host/PID/start-token/UTC timestamp/Slurm job identity.

    Returns
    -------
    pathlib.Path
        Newly created ``execution_guard.json`` owned by this invocation.

    Raises
    ------
    FileExistsError
        If another owner has already atomically claimed the same run.
    """
    guard = run_directory / _GUARD_FILE
    payload = (_canonical_json(dict(owner)) + "\n").encode("utf-8")
    try:
        with guard.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise FileExistsError("An execution guard already owns this run.")
    _append_execution_history(run_directory, "guard_claimed")
    return guard


def _release_execution_guard(run_directory: Path, *, owner: Mapping[str, object]) -> None:
    """Remove a guard only when it still belongs to the current owner.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory.
    owner : mapping[str, object]
        Current claimed owner record.

    Returns
    -------
    None
        Removes the guard for an equal owner and records orderly release.
    """
    guard = run_directory / _GUARD_FILE
    if not guard.exists():
        return
    try:
        saved_owner = _read_json(guard)
    except ValueError:
        return
    if saved_owner != dict(owner):
        return
    guard.unlink()
    _append_execution_history(run_directory, "guard_released")


def _record_guard_release_before_completion(
    run_directory: Path,
    *,
    owner: Mapping[str, object],
) -> Path | None:
    """Record a successful guard handoff while retaining exclusion through complete.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory.
    owner : mapping[str, object]
        Current owner that must still equal the saved guard record.

    Returns
    -------
    pathlib.Path or None
        Matching guard to unlink after the final complete-state replacement, or
        None when another owner has already replaced/removed it. Recording the
        history before completion makes that terminal state the final atomic
        publication while the guard remains present during the replacement.
    """
    guard = run_directory / _GUARD_FILE
    if not guard.exists():
        return None
    try:
        saved_owner = _read_json(guard)
    except ValueError:
        return None
    if saved_owner != dict(owner):
        return None
    _append_execution_history(run_directory, "guard_released")
    return guard


def _current_owner(execution: Mapping[str, object]) -> dict[str, object]:
    """Construct one current execution-owner record from saved route provenance.

    Parameters
    ----------
    execution : mapping[str, object]
        Prepared execution provenance including requested mode.

    Returns
    -------
    dict[str, object]
        Current mode/host/PID/start-token/UTC timestamp and optional Slurm job ID.
    """
    mode = "slurm" if os.environ.get("SLURM_JOB_ID") else str(execution.get("mode"))
    return {
        "mode": mode,
        "host": platform.node(),
        "pid": None if mode == "slurm" else os.getpid(),
        "start_token": None if mode == "slurm" else _process_start_token(os.getpid()),
        "started_at": _utc_now(),
        "job_id": os.environ.get("SLURM_JOB_ID") if mode == "slurm" else None,
    }


def _is_matching_local_receipt(
    receipt: Mapping[str, object],
    owner: Mapping[str, object],
) -> bool:
    """Identify a detached child from immutable local PID/token facts.

    Parameters
    ----------
    receipt, owner : mapping[str, object]
        Persisted launch receipt and currently executing owner facts. Both
        describe one local process; UTC launch timestamps may differ because
        the child creates its record after the parent receipt is durable.

    Returns
    -------
    bool
        True only when mode, host, PID, and Linux start token identify one
        detached child process without using mutable timestamps.
    """
    keys = ("mode", "host", "pid", "start_token")
    return all(receipt.get(key) == owner.get(key) for key in keys)


def _reject_live_owner(run_directory: Path, path: Path, description: str) -> None:
    """Reject active/foreign ownership or remove a confirmed stale owner file.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory used for history provenance.
    path : pathlib.Path
        Existing guard or detached-local receipt JSON path.
    description : str
        Human-readable ``"guard"`` or ``"receipt"`` label.

    Returns
    -------
    None
        Leaves active/foreign ownership untouched; removes dead/reused/terminal
        ownership after recording one bounded inspection fact.

    Raises
    ------
    RuntimeError
        If the owner is live, pending/running Slurm, foreign, or malformed.
    """
    if not path.exists():
        return
    owner = _read_json(path)
    state = _inspect_execution_owner(owner)
    _append_execution_history(run_directory, f"{description}_{state}")
    if state in {"live", "slurm_pending", "slurm_running", "foreign_unresolved"}:
        raise RuntimeError(f"Existing {description} is active, live, Slurm-owned, or foreign.")
    path.unlink()


def _update_state(run_directory: Path, **changes: object) -> dict[str, object]:
    """Apply and atomically save one small lifecycle-state transition.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory.
    **changes : object
        JSON-safe lifecycle, stage, target, warning, timing, or error fields.

    Returns
    -------
    dict[str, object]
        The saved state mapping with refreshed UTC ``updated_at``.
    """
    state = _read_json(run_directory / _STATE_FILE)
    state.update(changes)
    state["updated_at"] = _utc_now()
    _write_json(run_directory / _STATE_FILE, state)
    return state


def _revalidate_prepared_identity(run_directory: Path) -> dict[str, object]:
    """Rebuild all bounded identity before a guard or a heavy activity stage.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run containing immutable scientific sidecars and execution path.

    Returns
    -------
    dict[str, object]
        Fresh identity payload from the original configuration after exact saved
        scientific config, manifest, source fingerprint, version, runtime, and
        full fingerprint comparisons.

    Raises
    ------
    ValueError
        If any post-preparation input, code, configuration, or runtime identity
        differs. No spike arrays or model fits have run when this is raised.
    """
    execution = _read_json(run_directory / _EXECUTION_FILE)
    config_path = execution.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("Prepared execution provenance lacks configuration path.")
    identity = _build_preparation_identity(Path(config_path))
    saved_config = _read_json(run_directory / "config.json")
    saved_manifest = _read_json(run_directory / "input_manifest.json")
    saved_source = _read_json(run_directory / "scientific_source.json")
    saved_fingerprint = (
        (run_directory / "run_fingerprint.txt").read_text(encoding="ascii").strip()
    )
    if identity["scientific_config"] != saved_config:
        raise ValueError(
            "Prepared scientific configuration or analysis version differs "
            "from current configuration."
        )
    if identity["input_manifest"] != saved_manifest:
        raise ValueError("Prepared input manifest differs from current input identity.")
    if identity["scientific_source"] != saved_source:
        raise ValueError("Prepared scientific source fingerprint differs from current source.")
    if identity["fingerprint"] != saved_fingerprint:
        raise ValueError("Prepared full scientific fingerprint differs from current identity.")
    if execution.get("runtime_versions") != _runtime_version_mapping():
        raise ValueError("Prepared runtime version mapping differs from current runtime.")
    return identity


def _family_for_label(label: str) -> str:
    """Resolve one configured target's frozen model family.

    Parameters
    ----------
    label : str
        Configured task-variable target identifier.

    Returns
    -------
    str
        ``"categorical"`` or ``"numerical"`` according to frozen target metadata.
    """
    return decoding_config.TARGET_DEFINITION_BY_IDENTIFIER[label].family


def _target_mapping(label: str, family: str) -> str:
    """Return canonical target-label JSON required by the result schema.

    Parameters
    ----------
    label : str
        Configured target identifier.
    family : {"categorical", "numerical"}
        Target estimator/metric family.

    Returns
    -------
    str
        Canonical class-label mapping or native-target declaration JSON.
    """
    if family == "numerical":
        return _canonical_json({"native_target": True})
    return _canonical_json(targets._CATEGORICAL_CLASS_LABELS[label])


def _target_units(labels: Sequence[str]) -> dict[str, object]:
    """Build exact target-specific physical/model unit metadata for result arrays.

    Parameters
    ----------
    labels : sequence[str]
        Configured target labels in result target-axis order.

    Returns
    -------
    dict[str, object]
        Unit mapping for target values, metrics, coefficients, intercepts, time,
        and elapsed-stage arrays.
    """
    encoded: dict[str, str] = {}
    fold_scores: dict[str, dict[str, str]] = {}
    candidate_scores: dict[str, dict[str, str]] = {}
    coefficients: dict[str, str] = {}
    intercepts: dict[str, str] = {}
    for label in labels:
        if _family_for_label(label) == "categorical":
            encoded[label] = "encoded class"
            fold_scores[label] = {"balanced_accuracy": "fraction", "auc": "fraction"}
            candidate_scores[label] = {"balanced_accuracy": "fraction"}
            coefficients[label] = "log-odds change per pooled training standard deviation"
            intercepts[label] = "log-odds"
        else:
            encoded[label] = "unitless"
            fold_scores[label] = {"r2": "coefficient_of_determination"}
            candidate_scores[label] = {"r2": "coefficient_of_determination"}
            coefficients[label] = "unitless per pooled training standard deviation"
            intercepts[label] = "unitless"
    return {
        "encoded_target_values": encoded,
        "fold_scores": fold_scores,
        "candidate_inner_scores": candidate_scores,
        "coefficient_values": coefficients,
        "fitted_intercepts": intercepts,
        "time_bin_edges_s": "s",
        "time_bin_centers_s": "s",
        "stage_timing_seconds": "s",
        "total_timing_seconds": "s",
    }


def _result_axes(mode: str) -> dict[str, list[str]]:
    """Return the complete named-axis metadata contract for all 50 result arrays.

    Parameters
    ----------
    mode : {"fixed", "tuned"}
        Regularization mode selecting candidate-array applicability.

    Returns
    -------
    dict[str, list[str]]
        Exact member-to-axis mapping; scalar total timing uses an empty list.
    """
    fit_axes = ["target", "region", "representation", "time", "fold"]
    axes = {
        "target_labels": ["target"],
        "target_families": ["target"],
        "target_status": ["target"],
        "target_unavailable_reasons": ["target"],
        "target_source_labels": ["target"],
        "target_positive_classes": ["target"],
        "target_label_mappings_json": ["target"],
        "region_labels": ["region"],
        "representation_labels": ["representation"],
        "metric_labels": ["metric"],
        "time_bin_edges_s": ["time_edge"],
        "time_bin_centers_s": ["time"],
        "fold_labels": ["fold"],
        "fold_scores": ["target", "region", "representation", "metric", "time", "fold"],
        "fit_status": fit_axes,
        "fit_reason_codes": fit_axes,
        "requested_feature_counts": fit_axes,
        "effective_feature_counts": fit_axes,
        "train_counts": fit_axes,
        "test_counts": fit_axes,
        "train_class_counts": [*fit_axes, "class"],
        "test_class_counts": [*fit_axes, "class"],
        "full_table_row_positions": ["full_table_row"],
        "full_table_trial_ids": ["full_table_row"],
        "full_table_block_ids_utf8": ["block_id_byte"],
        "full_table_block_id_offsets": ["full_table_row_boundary"],
        "encoded_target_values": ["target", "full_table_row"],
        "eligibility_masks": ["target", "full_table_row"],
        "eligibility_reason_codes": ["target", "full_table_row"],
        "eligibility_counts": ["target"],
        "trial_row_indices": ["common_neural_tensor_row"],
        "outer_fold_ids": ["target", "common_neural_tensor_row"],
        "unit_selection_rules_json": ["region_component"],
        "coefficient_values": [*fit_axes, "feature"],
        "coefficient_feature_ids": ["region", "representation", "feature"],
        "coefficient_feature_regions": ["region", "representation", "feature"],
        "coefficient_feature_statuses": ["region", "representation", "feature"],
        "coefficient_active_masks": ["region", "representation", "feature"],
        "fitted_intercepts": fit_axes,
        "fixed_parameters_json": ["target", "outer_fold", "region", "representation", "time"],
        "selected_parameters_json": ["target", "outer_fold", "region", "representation", "time"],
        "stage_timing_labels": ["stage"],
        "stage_timing_seconds": ["stage"],
        "total_timing_seconds": [],
    }
    if mode == "fixed":
        axes.update(
            {
                "candidate_parameter_json": ["not_applicable"],
                "inner_selection_fold_ids": ["not_applicable"],
                "candidate_inner_scores": ["not_applicable"],
                "candidate_inner_statuses": ["not_applicable"],
                "candidate_inner_reasons": ["not_applicable"],
                "selected_candidate_indices": ["not_applicable"],
            }
        )
    else:
        audit = [
            "target",
            "outer_fold",
            "region",
            "representation",
            "time",
            "candidate",
            "inner_fold",
        ]
        axes.update(
            {
                "candidate_parameter_json": ["target", "candidate"],
                "inner_selection_fold_ids": ["target", "outer_fold", "common_neural_tensor_row"],
                "candidate_inner_scores": audit,
                "candidate_inner_statuses": audit,
                "candidate_inner_reasons": audit,
                "selected_candidate_indices": audit[:-2],
            }
        )
    return axes


def _regional_feature_maps(
    records: Sequence[modeling.TargetDecodingResult],
    region_activity_metadata: Mapping[str, Mapping[str, object]],
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Choose stable standalone feature identities from completed fit records.

    Parameters
    ----------
    records : sequence[TargetDecodingResult]
        Ordered target records containing direct/PCA fold feature IDs.
    region_activity_metadata : mapping[str, mapping]
        Stable PFC/HPC direct-unit identities selected before fitting.  Direct
        axes retain this complete order even when an individual fold removes a
        unit for lack of variance.

    Returns
    -------
    dict[tuple[str, str], tuple[str, ...]]
        PFC/HPC standalone feature IDs by representation. Combined identities
        are always derived by concatenating PFC then HPC.
    """
    selected: dict[tuple[str, str], tuple[str, ...]] = {
        (region, representation): ()
        for region in _REGIONS[:2]
        for representation in _REPRESENTATIONS
    }
    for result in records:
        for record in result.fold_records:
            key = (record.region_configuration, record.representation)
            if key in selected and record.feature_ids:
                candidate = tuple(record.feature_ids)
                if len(candidate) > len(selected[key]):
                    selected[key] = candidate
    for region in _REGIONS[:2]:
        metadata_ids = region_activity_metadata[region].get("unit_ids", ())
        if isinstance(metadata_ids, (str, bytes)):
            raise ValueError("Region unit metadata must be a sequence of stable IDs.")
        selected[(region, "units")] = tuple(str(value) for value in metadata_ids)
    return selected


def _block_identity_arrays(block_values: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    """Encode scalar behavioral block IDs as canonical variable-length UTF-8 bytes.

    Parameters
    ----------
    block_values : sequence[object]
        Full-table scalar block labels in chronological row order.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        One-dimensional uint8 concatenated canonical JSON bytes and int64
        ``(full_table_row + 1,)`` offsets. Values are grouping identities.
    """
    encoded = [
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        for value in block_values
    ]
    payload = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    offsets = np.r_[np.array([0], dtype=np.int64), np.cumsum(
        [len(value) for value in encoded], dtype=np.int64
    )]
    return payload, offsets


def _record_lookup(
    result: modeling.TargetDecodingResult,
) -> dict[tuple[int, int, str, str], modeling.FoldRecord]:
    """Index one target's records by its stable outer/time/cell key.

    Parameters
    ----------
    result : modeling.TargetDecodingResult
        One target's fixed or tuned result.

    Returns
    -------
    dict[tuple[int, int, str, str], FoldRecord]
        Fold records indexed by ``(outer_fold, time, region, representation)``.
    """
    return {record.record_key: record for record in result.fold_records}


def _fixed_parameter_json(scientific_config: Mapping[str, object], family: str) -> str:
    """Return canonical frozen family regularization JSON for every saved fit.

    Parameters
    ----------
    scientific_config : mapping[str, object]
        Serializer-owned frozen estimator controls.
    family : {"categorical", "numerical"}
        Target model family.

    Returns
    -------
    str
        Canonical JSON containing ``C``/``alpha`` and ``l1_ratio``.
    """
    estimator = "LogisticRegression" if family == "categorical" else "ElasticNet"
    strength = "C" if family == "categorical" else "alpha"
    controls = scientific_config["frozen_controls"]["estimators"][estimator]
    return json.dumps(
        {strength: controls[strength], "l1_ratio": controls["l1_ratio"]},
        sort_keys=True,
    )


def _candidate_jsons(scientific_config: Mapping[str, object], family: str) -> list[str]:
    """Expand the frozen ordered 15-candidate regularization grid as canonical JSON.

    Parameters
    ----------
    scientific_config : mapping[str, object]
        Serializer-owned frozen tuning grid.
    family : {"categorical", "numerical"}
        Target model family.

    Returns
    -------
    list[str]
        Fifteen declared candidates in strength-major then l1-ratio order.
    """
    estimator = "LogisticRegression" if family == "categorical" else "ElasticNet"
    strength = "C" if family == "categorical" else "alpha"
    grid = scientific_config["frozen_controls"]["tuning_grid"][estimator]
    return [
        json.dumps({strength: value, "l1_ratio": ratio}, sort_keys=True)
        for value in grid[strength]
        for ratio in grid["l1_ratio"]
    ]


def _assemble_run_result_payload(
    *,
    config: decoding_config.TaskDecodingConfig,
    target_table: pd.DataFrame,
    rate_tensors: activity.SessionRateTensors,
    region_activity_metadata: Mapping[str, Mapping[str, object]],
    ordered_target_results: Sequence[modeling.TargetDecodingResult],
    stage_timing_seconds: Mapping[str, float],
    input_manifest: Mapping[str, object],
    scientific_source: Mapping[str, object],
    execution_provenance: Mapping[str, object],
    run_fingerprint: str,
    session_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Assemble one complete WP5A primitive payload from decoded target records.

    Parameters
    ----------
    config : TaskDecodingConfig
        Scientific settings defining targets, time window, folds, and controls.
    target_table : pandas.DataFrame
        Full chronological target table, shape ``(full_trial, columns)``.
    rate_tensors : SessionRateTensors
        Common neural rows with rate tensors ``(trial, time, unit)`` in Hz.
    region_activity_metadata : mapping[str, mapping]
        PFC/HPC stable unit IDs and JSON-safe selection rules.
    ordered_target_results : sequence[TargetDecodingResult]
        One result in exactly configured target order.
    stage_timing_seconds : mapping[str, float]
        Elapsed stage durations in seconds.
    input_manifest, scientific_source, execution_provenance : mapping
        Portable input/code identity and execution-only provenance.
    run_fingerprint : str
        Exact scientific run identity persisted in result metadata.
    session_id : str
        Nonempty canonical session identifier from resolved metadata. It is
        persisted directly rather than inferred from a portable file path.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, object]]
        Complete 50-member primitive array schema and matching named metadata.
        Time axes are seconds; rate-derived feature coefficients retain the
        target-specific standardized model units declared in metadata.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Result assembly requires a nonempty explicit session_id.")
    labels = list(config.target_names)
    if [result.target_identifier for result in ordered_target_results] != labels:
        raise ValueError("Decoded target results must remain in configured target order.")
    scientific_config = decoding_config.scientific_config_payload(config)
    families = [_family_for_label(label) for label in labels]
    target_n = len(labels)
    time_n = rate_tensors.pfc_bin_centers_s.size
    fold_n = config.outer_fold_count
    full_n = target_table.shape[0]
    trial_rows = np.asarray(rate_tensors.trial_row_indices, dtype=np.int64)
    common_n = trial_rows.size
    regional_ids = _regional_feature_maps(
        ordered_target_results,
        region_activity_metadata,
    )
    feature_maps: dict[tuple[str, str], tuple[str, ...]] = {}
    for representation in _REPRESENTATIONS:
        feature_maps[("PFC", representation)] = regional_ids[("PFC", representation)]
        feature_maps[("HPC", representation)] = regional_ids[("HPC", representation)]
        feature_maps[("PFC+HPC", representation)] = (
            regional_ids[("PFC", representation)]
            + regional_ids[("HPC", representation)]
        )
    feature_n = max(1, *(len(feature_ids) for feature_ids in feature_maps.values()))
    fit_shape = (target_n, 3, 2, time_n, fold_n)
    score_shape = (target_n, 3, 2, 3, time_n, fold_n)
    parameter_shape = (target_n, fold_n, 3, 2, time_n)
    selection_rules = [
        _canonical_json(region_activity_metadata[region].get("selection_rules", {}))
        for region in _REGIONS[:2]
    ]
    selection_width = max(1, *(len(value) for value in selection_rules))
    arrays: dict[str, np.ndarray] = {
        "target_labels": np.asarray(labels, dtype="U32"),
        "target_families": np.asarray(families, dtype="U16"),
        "target_status": np.asarray(
            [result.status for result in ordered_target_results],
            dtype="U16",
        ),
        "target_unavailable_reasons": np.asarray(
            [result.unavailable_reason or "" for result in ordered_target_results],
            dtype="U128",
        ),
        "target_source_labels": np.asarray(
            [
                decoding_config.TARGET_DEFINITION_BY_IDENTIFIER[label].source_column
                for label in labels
            ],
            dtype="U32",
        ),
        "target_positive_classes": np.asarray(
            [1 if family == "categorical" else -1 for family in families],
            dtype=np.int64,
        ),
        "target_label_mappings_json": np.asarray(
            [
                _target_mapping(label, family)
                for label, family in zip(labels, families, strict=True)
            ],
            dtype="U48",
        ),
        "region_labels": np.asarray(_REGIONS, dtype="U16"),
        "representation_labels": np.asarray(_REPRESENTATIONS, dtype="U16"),
        "metric_labels": np.asarray(_METRICS, dtype="U32"),
        "time_bin_edges_s": np.linspace(
            decoding_config.WINDOW_START_SECONDS,
            decoding_config.WINDOW_END_SECONDS,
            time_n + 1,
            dtype=np.float64,
        ),
        "time_bin_centers_s": np.asarray(rate_tensors.pfc_bin_centers_s, dtype=np.float64),
        "fold_labels": np.arange(fold_n, dtype=np.int64),
        "fold_scores": np.full(score_shape, np.nan, dtype=np.float64),
        "fit_status": np.full(fit_shape, "unavailable", dtype="U16"),
        "fit_reason_codes": np.full(fit_shape, "", dtype="U32"),
        "requested_feature_counts": np.zeros(fit_shape, dtype=np.int64),
        "effective_feature_counts": np.zeros(fit_shape, dtype=np.int64),
        "train_counts": np.zeros(fit_shape, dtype=np.int64),
        "test_counts": np.zeros(fit_shape, dtype=np.int64),
        "train_class_counts": np.full((*fit_shape, 2), -1, dtype=np.int64),
        "test_class_counts": np.full((*fit_shape, 2), -1, dtype=np.int64),
        "full_table_row_positions": np.asarray(target_table["row_position"], dtype=np.int64),
        "full_table_trial_ids": np.asarray(target_table["trial_id"], dtype=np.int64),
        "full_table_block_ids_utf8": np.empty(0, dtype=np.uint8),
        "full_table_block_id_offsets": np.empty(full_n + 1, dtype=np.int64),
        "encoded_target_values": np.full((target_n, full_n), np.nan, dtype=np.float64),
        "eligibility_masks": np.zeros((target_n, full_n), dtype=np.bool_),
        "eligibility_reason_codes": np.full((target_n, full_n), "not_common", dtype="U32"),
        "eligibility_counts": np.zeros(target_n, dtype=np.int64),
        "trial_row_indices": trial_rows,
        "outer_fold_ids": np.full((target_n, common_n), -1, dtype=np.int64),
        "unit_selection_rules_json": np.asarray(selection_rules, dtype=f"U{selection_width}"),
        "coefficient_values": np.full((*fit_shape, feature_n), np.nan, dtype=np.float64),
        "coefficient_feature_ids": np.full((3, 2, feature_n), "", dtype="U32"),
        "coefficient_feature_regions": np.full((3, 2, feature_n), "", dtype="U16"),
        "coefficient_feature_statuses": np.full((3, 2, feature_n), "padding", dtype="U16"),
        "coefficient_active_masks": np.zeros((3, 2, feature_n), dtype=np.bool_),
        "fitted_intercepts": np.full(fit_shape, np.nan, dtype=np.float64),
        "fixed_parameters_json": np.empty(parameter_shape, dtype="U48"),
        "candidate_parameter_json": np.empty((0,), dtype="U48"),
        "selected_parameters_json": np.full(parameter_shape, "", dtype="U48"),
        "inner_selection_fold_ids": np.empty((0,), dtype=np.int64),
        "candidate_inner_scores": np.empty((0,), dtype=np.float64),
        "candidate_inner_statuses": np.empty((0,), dtype="U16"),
        "candidate_inner_reasons": np.empty((0,), dtype="U48"),
        "selected_candidate_indices": np.empty((0,), dtype=np.int64),
        "stage_timing_labels": np.asarray(list(stage_timing_seconds), dtype="U16"),
        "stage_timing_seconds": np.asarray(list(stage_timing_seconds.values()), dtype=np.float64),
        "total_timing_seconds": np.asarray(sum(stage_timing_seconds.values()), dtype=np.float64),
    }
    block_bytes, block_offsets = _block_identity_arrays(target_table["block_id"].tolist())
    arrays["full_table_block_ids_utf8"] = block_bytes
    arrays["full_table_block_id_offsets"] = block_offsets
    region_index = {label: index for index, label in enumerate(_REGIONS)}
    representation_index = {
        label: index for index, label in enumerate(_REPRESENTATIONS)
    }
    requested_widths = np.asarray(
        [
            [config.pfc_pc_count, len(feature_maps[("PFC", "units")])],
            [config.hpc_pc_count, len(feature_maps[("HPC", "units")])],
            [
                config.pfc_pc_count + config.hpc_pc_count,
                len(feature_maps[("PFC+HPC", "units")]),
            ],
        ],
        dtype=np.int64,
    )
    for region, representation in np.ndindex(3, 2):
        feature_ids = feature_maps[(_REGIONS[region], _REPRESENTATIONS[representation])]
        width = len(feature_ids)
        arrays["coefficient_feature_ids"][region, representation, :width] = feature_ids
        arrays["coefficient_feature_statuses"][region, representation, :width] = "available"
        arrays["coefficient_active_masks"][region, representation, :width] = True
        if region == 0:
            feature_regions = ("PFC",) * width
        elif region == 1:
            feature_regions = ("HPC",) * width
        else:
            pfc_width = len(feature_maps[("PFC", _REPRESENTATIONS[representation])])
            feature_regions = ("PFC",) * pfc_width + ("HPC",) * (width - pfc_width)
        arrays["coefficient_feature_regions"][region, representation, :width] = feature_regions
    for target, family in enumerate(families):
        arrays["fixed_parameters_json"][target, ...] = _fixed_parameter_json(
            scientific_config,
            family,
        )
        for region, representation in np.ndindex(3, 2):
            arrays["requested_feature_counts"][target, region, representation] = (
                requested_widths[region, representation]
            )

    common_row_lookup = {int(row): index for index, row in enumerate(trial_rows)}
    common_membership = np.zeros(full_n, dtype=bool)
    common_membership[trial_rows] = True
    for target, (label, family, result) in enumerate(
        zip(labels, families, ordered_target_results, strict=True)
    ):
        target_valid = np.asarray(target_table[f"{label}_valid"], dtype=bool)
        eligible = target_valid & common_membership
        arrays["eligibility_masks"][target] = eligible
        arrays["eligibility_counts"][target] = int(np.count_nonzero(eligible))
        arrays["encoded_target_values"][target, eligible] = target_table.loc[
            eligible, label
        ].to_numpy(dtype=float)
        # Target-invalid precedes neural-common-row status.  A valid target
        # omitted from the bilateral tensor remains distinguishable as
        # ``not_common`` for audit and resume diagnostics.
        arrays["eligibility_reason_codes"][target, ~target_valid] = "target_ineligible"
        arrays["eligibility_reason_codes"][target, target_valid & ~common_membership] = (
            "not_common"
        )
        arrays["eligibility_reason_codes"][target, eligible] = ""
        eligible_common = np.flatnonzero(eligible[trial_rows])
        if result.status == "available":
            if result.outer_fold_ids.shape != (eligible_common.size,):
                raise ValueError("Target outer-fold axis does not match eligible tensor rows.")
            arrays["outer_fold_ids"][target, eligible_common] = result.outer_fold_ids
        elif result.status != "unavailable":
            raise ValueError("Target result status is invalid.")

        record_lookup = _record_lookup(result)
        for fold, region, representation, time_index in np.ndindex(fold_n, 3, 2, time_n):
            output_index = (target, region, representation, time_index, fold)
            record = record_lookup.get(
                (fold, time_index, _REGIONS[region], _REPRESENTATIONS[representation])
            )
            if result.status == "unavailable":
                arrays["fit_status"][output_index] = "unavailable"
                arrays["fit_reason_codes"][output_index] = _COMPACT_TARGET_UNAVAILABLE
                arrays["train_class_counts"][output_index] = 0 if family == "categorical" else -1
                arrays["test_class_counts"][output_index] = 0 if family == "categorical" else -1
                continue
            if record is None:
                raise ValueError("Available target is missing a Cartesian fold record.")
            arrays["fit_status"][output_index] = record.status
            arrays["fit_reason_codes"][output_index] = _compact_reason_code(record.reason)
            arrays["effective_feature_counts"][output_index] = record.effective_feature_count
            arrays["train_counts"][output_index] = record.train_count
            arrays["test_counts"][output_index] = record.test_count
            if family == "categorical":
                arrays["train_class_counts"][output_index] = (
                    record.train_class_counts.get(0, 0),
                    record.train_class_counts.get(1, 0),
                )
                arrays["test_class_counts"][output_index] = (
                    record.test_class_counts.get(0, 0),
                    record.test_class_counts.get(1, 0),
                )
            if record.status == "valid":
                metric_indices = (0, 1) if family == "categorical" else (2,)
                metric_names = (
                    ("balanced_accuracy", "auc")
                    if family == "categorical"
                    else ("r2",)
                )
                for metric, metric_name in zip(metric_indices, metric_names, strict=True):
                    arrays[
                        "fold_scores"
                    ][target, region, representation, metric, time_index, fold] = (
                        record.metrics[metric_name]
                    )
                arrays["fitted_intercepts"][output_index] = record.intercept
                global_features = feature_maps[
                    (_REGIONS[region], _REPRESENTATIONS[representation])
                ]
                for feature_id, coefficient in zip(
                    record.feature_ids,
                    record.coefficients,
                    strict=True,
                ):
                    position = global_features.index(feature_id)
                    arrays["coefficient_values"][output_index + (position,)] = coefficient

    if config.regularization_mode == "tuned":
        inner_n = config.inner_fold_count
        audit_shape = (target_n, fold_n, 3, 2, time_n, 15, inner_n)
        arrays["candidate_parameter_json"] = np.empty((target_n, 15), dtype="U48")
        arrays["inner_selection_fold_ids"] = np.full(
            (target_n, fold_n, common_n), -1, dtype=np.int64
        )
        arrays["candidate_inner_scores"] = np.full(audit_shape, np.nan, dtype=np.float64)
        arrays["candidate_inner_statuses"] = np.full(audit_shape, "invalid", dtype="U16")
        arrays["candidate_inner_reasons"] = np.full(audit_shape, "", dtype="U48")
        arrays["selected_candidate_indices"] = np.full(
            parameter_shape,
            -1,
            dtype=np.int64,
        )
        for target, (family, result) in enumerate(
            zip(families, ordered_target_results, strict=True)
        ):
            arrays["candidate_parameter_json"][target] = _candidate_jsons(
                scientific_config,
                family,
            )
            eligible_common = np.flatnonzero(
                arrays["eligibility_masks"][target, trial_rows]
            )
            for fold in range(fold_n):
                plan = result.inner_split_plans.get(fold)
                if plan is not None:
                    if plan.fold_ids.shape == (eligible_common.size,):
                        arrays["inner_selection_fold_ids"][
                            target,
                            fold,
                            eligible_common,
                        ] = plan.fold_ids
                for region, representation, time_index in np.ndindex(3, 2, time_n):
                    record_key = (
                        fold,
                        time_index,
                        _REGIONS[region],
                        _REPRESENTATIONS[representation],
                    )
                    output_index = (target, fold, region, representation, time_index)
                    record = _record_lookup(result).get(record_key)
                    scores = result.candidate_inner_scores.get(record_key, ())
                    statuses = result.candidate_inner_statuses.get(record_key, ())
                    reasons = result.candidate_inner_reasons.get(record_key, ())
                    selected = result.selected_candidate_indices.get(record_key)
                    if result.status == "unavailable":
                        arrays["candidate_inner_reasons"][
                            output_index
                        ] = _COMPACT_TARGET_UNAVAILABLE
                        continue
                    if scores and all(len(row) == inner_n for row in scores):
                        for candidate in range(15):
                            for inner in range(inner_n):
                                score = scores[candidate][inner]
                                arrays["candidate_inner_scores"][
                                    output_index + (candidate, inner)
                                ] = (
                                    np.nan if score is None else score
                                )
                                arrays["candidate_inner_statuses"][
                                    output_index + (candidate, inner)
                                ] = (
                                    statuses[candidate][inner]
                                )
                                arrays["candidate_inner_reasons"][
                                    output_index + (candidate, inner)
                                ] = _compact_reason_code(
                                    reasons[candidate][inner],
                                    inner=True,
                                )
                    else:
                        reason = (
                            record.reason if record is not None else "inner split unavailable"
                        )
                        arrays["candidate_inner_reasons"][output_index] = _compact_reason_code(
                            reason or "inner split unavailable",
                            inner=True,
                        )
                    if selected is not None:
                        arrays["selected_candidate_indices"][output_index] = selected
                        arrays["selected_parameters_json"][output_index] = arrays[
                            "candidate_parameter_json"
                        ][target, selected]

    metadata = {
        "schema_version": results.RESULT_SCHEMA_VERSION,
        "analysis_version": decoding_config.ANALYSIS_VERSION,
        "random_seed": scientific_config["frozen_controls"]["seed"],
        "units": _target_units(labels),
        "axes": _result_axes(config.regularization_mode),
        "paths": {"session_root": ".", "feature_parameter_copy": "trial_feature_params.json"},
        "parameters": {
            "regularization_mode": config.regularization_mode,
            "outer_fold_count": config.outer_fold_count,
            "inner_fold_count": config.inner_fold_count,
            "bin_width_ms": config.bin_width_ms,
            "candidate_selection_metrics": {
                label: "balanced_accuracy" if family == "categorical" else "r2"
                for label, family in zip(labels, families, strict=True)
            },
        },
        "provenance": {
            "session_id": session_id,
            "source_fingerprint": scientific_source["fingerprint"],
            "scientific_config_identity": hashlib.sha256(
                json.dumps(scientific_config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "run_fingerprint": run_fingerprint,
            "execution": dict(execution_provenance),
        },
        "warnings": [],
    }
    return arrays, metadata


def _region_metadata(region_activity: activity.RegionActivity) -> dict[str, object]:
    """Extract JSON-safe stable unit provenance from one loaded region activity record.

    Parameters
    ----------
    region_activity : RegionActivity
        Loaded regional population with unit-axis metadata and selection rules.

    Returns
    -------
    dict[str, object]
        Stable unit IDs and JSON-safe selection rules; unit IDs align with the
        corresponding tensor final axis and are dimensionless identities.
    """
    return {
        "unit_ids": region_activity.unit_metadata["unit_id"].astype(str).tolist(),
        "selection_rules": region_activity.selection_rules,
    }


def _decode_remaining_targets(
    *,
    config: decoding_config.TaskDecodingConfig,
    target_table: pd.DataFrame,
    rate_tensors: activity.SessionRateTensors,
    pfc_activity: activity.RegionActivity,
    hpc_activity: activity.RegionActivity,
    existing: Mapping[str, modeling.TargetDecodingResult],
    run_directory: Path,
    full_fingerprint: str,
) -> list[modeling.TargetDecodingResult]:
    """Restore valid target checkpoints and decode only configured missing targets.

    Parameters
    ----------
    config : TaskDecodingConfig
        Frozen target, fold, PCA, and regularization controls.
    target_table : pandas.DataFrame
        Full target values and validity masks in chronological table order.
    rate_tensors : SessionRateTensors
        Matched PFC/HPC Hz tensors on common trial rows.
    pfc_activity, hpc_activity : RegionActivity
        Loaded unit metadata whose IDs align with rate-tensor unit axes.
    existing : mapping[str, TargetDecodingResult]
        Fingerprint-validated checkpoints keyed by completed target label.
    run_directory : pathlib.Path
        Run directory where new complete target checkpoints are published.
    full_fingerprint : str
        Exact prepared scientific identity required by target checkpoints.

    Returns
    -------
    list[TargetDecodingResult]
        Complete configured target records in configuration order. Target values
        retain their encoded classes or native numerical units.
    """
    output: list[modeling.TargetDecodingResult] = []
    common_rows = np.asarray(rate_tensors.trial_row_indices, dtype=np.int64)
    pfc_ids = tuple(pfc_activity.unit_metadata["unit_id"].astype(str))
    hpc_ids = tuple(hpc_activity.unit_metadata["unit_id"].astype(str))
    for label in config.target_names:
        restored = existing.get(label)
        if restored is not None:
            output.append(restored)
            continue
        common_valid = activity.project_target_mask_to_tensor_rows(
            common_rows,
            target_table[f"{label}_valid"].to_numpy(dtype=bool),
        )
        selected_rows = common_rows[common_valid]
        target_values = target_table.loc[selected_rows, label].to_numpy(dtype=float)
        block_ids = target_table.loc[selected_rows, "block_id"].to_numpy(copy=True)
        result = modeling.decode_target(
            target_identifier=label,
            target_family=_family_for_label(label),
            target_values=target_values,
            block_ids=block_ids,
            pfc_rate_tensor_hz=rate_tensors.pfc_rate_tensor_hz[common_valid],
            hpc_rate_tensor_hz=rate_tensors.hpc_rate_tensor_hz[common_valid],
            pfc_unit_ids=pfc_ids,
            hpc_unit_ids=hpc_ids,
            pfc_requested_pc_count=config.pfc_pc_count,
            hpc_requested_pc_count=config.hpc_pc_count,
            outer_fold_count=config.outer_fold_count,
            inner_fold_count=config.inner_fold_count,
            regularization_mode=config.regularization_mode,
        )
        results.save_target_checkpoint(
            run_directory / "checkpoints" / f"{label}.npz",
            target_label=label,
            target_arrays=_target_result_to_checkpoint_arrays(result),
            full_run_fingerprint=full_fingerprint,
        )
        output.append(result)
        state = _read_json(run_directory / _STATE_FILE)
        completed = list(state.get("completed_targets", []))
        if label not in completed:
            completed.append(label)
        _update_state(
            run_directory,
            current_stage="modeling",
            completed_targets=completed,
        )
        _append_log(run_directory, f"target complete {label}")
    return output


def _load_existing_checkpoints(
    run_directory: Path,
    target_names: Sequence[str],
    full_fingerprint: str,
) -> dict[str, modeling.TargetDecodingResult]:
    """Load every present configured checkpoint with exact identity validation.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory containing a ``checkpoints`` subdirectory.
    target_names : sequence[str]
        Configured target order and valid checkpoint filename set.
    full_fingerprint : str
        Exact full run fingerprint required for every reused target.

    Returns
    -------
    dict[str, TargetDecodingResult]
        Restored completed target results keyed by their configured identifiers.

    Raises
    ------
    ValueError
        If a present checkpoint has a mismatched fingerprint, label, or payload.
    """
    output: dict[str, modeling.TargetDecodingResult] = {}
    for label in target_names:
        path = run_directory / "checkpoints" / f"{label}.npz"
        if not path.exists():
            continue
        saved = results.load_target_checkpoint(
            path,
            expected_full_run_fingerprint=full_fingerprint,
        )
        if saved["target_label"] != label:
            raise ValueError("Checkpoint target label does not match configured target order.")
        output[label] = _target_result_from_checkpoint(saved)
    return output


def _preflight_structurally_unavailable_targets(
    *,
    config: decoding_config.TaskDecodingConfig,
    target_table: pd.DataFrame,
    existing: Mapping[str, modeling.TargetDecodingResult],
    run_directory: Path,
    full_fingerprint: str,
) -> dict[str, modeling.TargetDecodingResult]:
    """Retain the historical preflight seam without inferring unavailable targets.

    Parameters
    ----------
    config : TaskDecodingConfig
        Requested target, split, PCA, and regularization controls.
    target_table : pandas.DataFrame
        Full chronological table with ``<target>_valid`` masks and block IDs.
    existing : mapping[str, TargetDecodingResult]
        Fingerprint-validated target checkpoints that need no new validation.
    run_directory : pathlib.Path
        Existing run directory where a proven-unavailable target checkpoint is
        saved for ordinary resume.
    full_fingerprint : str
        Exact scientific identity required for every new checkpoint.

    Returns
    -------
    dict[str, TargetDecodingResult]
        Always an empty mapping.  Bilateral common neural rows are only known
        after tensor construction, so target-only rows can never safely yield
        a checkpoint or outer-split conclusion.

    Notes
    -----
    This is a fail-fast contract check, not a saved result.  The regular
    matched-tensor path remains the sole source of persisted target records.
    """
    del config, target_table, existing, run_directory, full_fingerprint
    return {}


def _write_run_summary(
    run_directory: Path,
    *,
    config: decoding_config.TaskDecodingConfig,
    arrays: Mapping[str, np.ndarray],
    state: Mapping[str, object],
    execution: Mapping[str, object],
    session_id: str,
    total_seconds: float,
) -> None:
    """Write one immutable concise human-readable completion summary.

    Parameters
    ----------
    run_directory : pathlib.Path
        Completed or late-reporting run directory.
    config : TaskDecodingConfig
        Frozen scientific controls used by this session.
    arrays : mapping[str, numpy.ndarray]
        Valid result primitives with eligibility, fold, and timing axes.
    state : mapping[str, object]
        Lifecycle warnings and completed target facts.
    execution : mapping[str, object]
        Actual execution owner and requested execution mode for this run.
    session_id : str
        Canonical resolved session identifier, independent of local paths.
    total_seconds : float
        Total elapsed pipeline duration in seconds.

    Returns
    -------
    None
        Publishes ``summary.md`` once; existing immutable bytes are retained.
    """
    destination = run_directory / "summary.md"
    if destination.exists():
        return
    warnings = state.get("warnings", [])
    unavailable = arrays["target_labels"][arrays["target_status"] == "unavailable"].tolist()
    lines = [
        "# Task-variable decoding summary",
        "",
        f"UTC: {_utc_now()}",
        "Analysis goal: task_variable_decoding single-session grouped decoding.",
        f"Session: {session_id}",
        "Launch command: "
        + " ".join(
            [
                "uv",
                "run",
                "python",
                "-m",
                "src.neural_analysis.task_decoding.run_session",
                "_execute-prepared",
                "--run-directory",
                shlex.quote(str(run_directory)),
            ]
        ),
        "Execution mode: " + str(execution.get("mode", "foreground")) + ".",
        "Snapshots: run_session.py, pipeline.py, and immutable config/manifest sidecars.",
        f"Parameters: mode={config.regularization_mode}, bins={config.bin_width_ms} ms.",
        f"Eligibility: {arrays['eligibility_counts'].tolist()} target-valid common rows.",
        f"Folds: outer={config.outer_fold_count}, inner={config.inner_fold_count}.",
        f"Warnings: {warnings if warnings else 'none'}.",
        f"Unavailable targets: {unavailable if unavailable else 'none'}.",
        "Stage timing: "
        + ", ".join(
            f"{label}={float(value):.6f} s"
            for label, value in zip(
                arrays["stage_timing_labels"],
                arrays["stage_timing_seconds"],
                strict=True,
            )
        )
        + ".",
        f"Total timing: {total_seconds:.6f} s.",
        "Saved output: results.npz plus "
        + (
            "categorical BA/AUC"
            if "categorical" in arrays["target_families"].tolist()
            else ""
        )
        + (
            " and numerical R2"
            if "numerical" in arrays["target_families"].tolist()
            else ""
        )
        + " PNGs.",
        "",
    ]
    _atomic_bytes(destination, "\n".join(lines).encode("utf-8"))


def _write_minimal_family_heatmaps(
    run_directory: Path,
    *,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Publish the bounded completion PNG set through the WP7 plotting owner.

    Parameters
    ----------
    run_directory : pathlib.Path
        Immutable result directory.
    arrays : mapping[str, numpy.ndarray]
        Valid result arrays with target-family and metric score axes.

    Returns
    -------
    None
        Loads the just-published saved run for ordinary execution. The fallback
        adapter exists only for the older direct private-helper test seam and
        uses the same plotting implementation without loading neural sources.
    """
    directory = Path(run_directory)
    if (directory / _RESULT_FILE).is_file():
        plotting.save_default_decoding_figures(directory)
        return
    saved_run = {
        "arrays": arrays,
        "scientific_config": {"alignment": "choice_time"},
    }
    plotting.save_default_decoding_figures(
        directory,
        saved_run=saved_run,
        _legacy_panel_titles=True,
    )


def _validate_published_run_directory(run_directory: Path) -> None:
    """Validate immutable result/sidecar schema and required completion-gate PNGs.

    Parameters
    ----------
    run_directory : pathlib.Path
        Run directory with a published portable result NPZ.

    Returns
    -------
    None
        Raises ValueError if the saved result or required present-family figures
        are missing or invalid; no scientific data are loaded or modified.
    """
    summary_path = run_directory / "summary.md"
    try:
        summary_text = summary_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("Required completion summary is missing.") from error
    if not summary_text.strip():
        raise ValueError("Required completion summary is empty.")
    fingerprint = (run_directory / "run_fingerprint.txt").read_text(encoding="ascii").strip()
    loaded = results.load_task_decoding_run(
        run_directory,
        expected_run_fingerprint=fingerprint,
    )
    families = loaded["arrays"]["target_families"].tolist()
    required: list[str] = []
    if "categorical" in families:
        required.extend(["categorical_balanced_accuracy.png", "categorical_auc.png"])
    if "numerical" in families:
        required.append("numerical_r2.png")
    missing = [name for name in required if not (run_directory / "figures" / name).is_file()]
    if missing:
        raise ValueError(f"Required completion figures are missing: {missing}")
    from matplotlib import image as matplotlib_image

    for name in required:
        figure_path = run_directory / "figures" / name
        try:
            decoded = matplotlib_image.imread(figure_path)
        except (OSError, SyntaxError, ValueError) as error:
            raise ValueError(f"Completion figure is invalid or corrupt: {name}") from error
        if decoded.ndim < 2 or decoded.size == 0:
            raise ValueError(f"Completion figure is invalid or corrupt: {name}")


def run_prepared_task_decoding(run_directory: Path | str) -> None:
    """Run or resume one prepared decoding directory through its shared foreground path.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Existing immutable prepared directory. Saved sidecars determine all
        scientific inputs; no command-line scientific settings are accepted.

    Returns
    -------
    None
        Restores valid target checkpoints, decodes missing configured targets,
        publishes one validated result/summary/required PNG set, then marks the
        lifecycle complete. Neural tensors are Hz; model outputs retain their
        target-specific units in the saved result metadata.

    Raises
    ------
    ValueError
        For revalidation, checkpoint, scientific, or schema mismatches.
    Exception
        Unexpected I/O/programming errors propagate after durable failed state.
    KeyboardInterrupt or SystemExit
        Interruptions propagate after durable interrupted state.
    """
    directory = Path(run_directory)
    owner: dict[str, object] | None = None
    began = time.monotonic()
    try:
        initial_state = _read_json(directory / _STATE_FILE)
        if initial_state.get("lifecycle") == "complete":
            if not initial_state.get("final_results_published"):
                raise ValueError(
                    "Complete run state must declare final results published."
                )
            _validate_published_run_directory(directory)
            return None
        identity = _revalidate_prepared_identity(directory)
        execution = _read_json(directory / _EXECUTION_FILE)
        owner = _current_owner(execution)
        receipt_path = directory / _RECEIPT_FILE
        matching_receipt = receipt_path.exists() and _is_matching_local_receipt(
            _read_json(receipt_path),
            owner,
        )
        if not matching_receipt:
            _reject_live_owner(directory, receipt_path, "receipt")
        _reject_live_owner(directory, directory / _GUARD_FILE, "guard")
        # The prepared record describes the requestor.  Persist the actual
        # worker before claim so checkpoints, result provenance, and status
        # all name the same live owner.
        history = execution.get("history")
        execution.update(owner)
        execution["history"] = history if isinstance(history, list) else []
        _write_json(directory / _EXECUTION_FILE, execution)
        _claim_execution_guard(directory, owner=owner)
        if matching_receipt:
            receipt_path.unlink()
        execution = _read_json(directory / _EXECUTION_FILE)
        state = _read_json(directory / _STATE_FILE)
        _update_state(
            directory,
            lifecycle="running",
            current_stage="targets",
            started_at=state.get("started_at") or _utc_now(),
            last_error="",
        )
        _append_log(directory, "stage targets")
        config = identity["config"]
        target_table = identity["target_table"]
        resolved = identity["resolved_session"]
        fingerprint = str(identity["fingerprint"])
        # Validate compact checkpoints before loading any potentially large spike
        # inputs, so a stale scientific identity cannot trigger analysis work.
        existing = _load_existing_checkpoints(directory, config.target_names, fingerprint)

        activity_started = time.monotonic()
        pfc_activity = activity.load_region_activity(
            config.pfc_region,
            resolved.probes,
            trusted_utc_bounds=config.trusted_utc_bounds.get(config.pfc_region.probe_id),
        )
        hpc_activity = activity.load_region_activity(
            config.hpc_region,
            resolved.probes,
            trusted_utc_bounds=config.trusted_utc_bounds.get(config.hpc_region.probe_id),
        )
        rate_tensors = activity.build_session_rate_tensors(
            target_table,
            pfc_activity,
            hpc_activity,
            alignment_event=config.alignment,
            window_s=(decoding_config.WINDOW_START_SECONDS, decoding_config.WINDOW_END_SECONDS),
            bin_width_s=config.bin_width_ms / 1000.0,
        )
        activity_seconds = time.monotonic() - activity_started
        _update_state(directory, current_stage="modeling")
        _append_log(directory, "stage activity")

        modeling_started = time.monotonic()
        target_results = _decode_remaining_targets(
            config=config,
            target_table=target_table,
            rate_tensors=rate_tensors,
            pfc_activity=pfc_activity,
            hpc_activity=hpc_activity,
            existing=existing,
            run_directory=directory,
            full_fingerprint=fingerprint,
        )
        modeling_seconds = time.monotonic() - modeling_started
        stage_seconds = {
            "targets": activity_started - began,
            "activity": activity_seconds,
            "modeling": modeling_seconds,
        }
        region_metadata = {
            "PFC": _region_metadata(pfc_activity),
            "HPC": _region_metadata(hpc_activity),
        }
        arrays, metadata = _assemble_run_result_payload(
            config=config,
            target_table=target_table,
            rate_tensors=rate_tensors,
            region_activity_metadata=region_metadata,
            ordered_target_results=target_results,
            stage_timing_seconds=stage_seconds,
            input_manifest=identity["input_manifest"],
            scientific_source=identity["scientific_source"],
            execution_provenance=execution,
            run_fingerprint=fingerprint,
            session_id=resolved.session_id,
        )
        elapsed = time.monotonic() - began
        arrays["total_timing_seconds"] = np.asarray(elapsed, dtype=np.float64)
        result_path = directory / _RESULT_FILE
        if result_path.exists():
            loaded = results.load_task_decoding_run(
                directory,
                expected_run_fingerprint=fingerprint,
            )
            arrays = loaded["arrays"]
        else:
            results.save_task_decoding_run(
                directory,
                arrays=arrays,
                meta=metadata,
                input_manifest=identity["input_manifest"],
                scientific_config=identity["scientific_config"],
                feature_parameter_source=config.trial_feature_parameter_path,
                run_fingerprint=fingerprint,
            )
        _update_state(directory, current_stage="reporting", final_results_published=False)
        _write_run_summary(
            directory,
            config=config,
            arrays=arrays,
            state=_read_json(directory / _STATE_FILE),
            execution=_read_json(directory / _EXECUTION_FILE),
            session_id=resolved.session_id,
            total_seconds=elapsed,
        )
        _write_minimal_family_heatmaps(directory, arrays=arrays)
        _validate_published_run_directory(directory)
        guard_to_remove = (
            _record_guard_release_before_completion(directory, owner=owner)
            if owner is not None
            else None
        )
        _update_state(
            directory,
            lifecycle="complete",
            current_stage="complete",
            completed_at=_utc_now(),
            completed_targets=list(config.target_names),
            final_results_published=True,
        )
        _append_log(directory, "stage complete")
        if guard_to_remove is not None:
            guard_to_remove.unlink()
        owner = None
    except (KeyboardInterrupt, SystemExit):
        try:
            _update_state(
                directory,
                lifecycle="interrupted",
                current_stage="interrupted",
                final_results_published=False,
            )
            _append_log(directory, "interrupted")
        finally:
            if owner is not None:
                _release_execution_guard(directory, owner=owner)
        raise
    except BaseException as error:
        try:
            _update_state(
                directory,
                lifecycle="failed",
                current_stage="failed",
                last_error=f"{type(error).__name__}: {error}",
                final_results_published=False,
            )
            _append_log(directory, f"failed {type(error).__name__}: {error}")
        finally:
            if owner is not None:
                _release_execution_guard(directory, owner=owner)
        raise
    else:
        return None


def _poll_scheduler(*_args: object, **_kwargs: object) -> None:
    """Reserved no-op seam; status never polls schedulers in WP5B.

    Parameters
    ----------
    *_args, **_kwargs : object
        Ignored scheduler arguments retained only for explicit test guarding.

    Returns
    -------
    None
        This function intentionally performs no scheduler action.
    """
    return None


def inspect_task_decoding_status(
    run_directory: Path | str,
    verify_results: bool = False,
) -> dict[str, object]:
    """Read one saved run's small lifecycle facts without loading neural sources.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Existing prepared or completed run directory.
    verify_results : bool, default=False
        If true, validate a completed portable NPZ and immutable sidecars using
        the results loader. The check does not fit models or read spikes.

    Returns
    -------
    dict[str, object]
        Lifecycle, stages, completed/total targets, warnings/errors, durable log
        path, recovery command, optional owner status, and result validation.
    """
    directory = Path(run_directory)
    state = _read_json(directory / _STATE_FILE)
    execution = _read_json(directory / _EXECUTION_FILE)
    status = dict(state)
    status["log_path"] = directory / "run.log"
    resume_path = directory / "resume_command.txt"
    if resume_path.exists():
        status["recovery_command"] = resume_path.read_text(encoding="utf-8").strip()
    else:
        status["recovery_command"] = (
            "uv run python -m src.neural_analysis.task_decoding.run_session "
            f"resume --run-directory {shlex.quote(str(directory))}"
        )
    owners = (
        (directory / _RECEIPT_FILE, "receipt"),
        (directory / _GUARD_FILE, "guard"),
    )
    for owner_file, label in owners:
        if owner_file.exists():
            try:
                status[f"{label}_status"] = _inspect_execution_owner(_read_json(owner_file))
            except ValueError:
                status[f"{label}_status"] = "invalid"
            # Detached status is commonly serialized by a CLI/monitor.  Keep
            # its owner-bearing payload JSON-safe while the ordinary API still
            # returns a Path for simple nonterminal status inspection.
            status["log_path"] = str(status["log_path"])
    status["execution_mode"] = execution.get("mode")
    status["execution_owner"] = {
        key: execution.get(key)
        for key in ("mode", "host", "pid", "start_token", "started_at", "job_id")
    }
    if verify_results and state.get("lifecycle") == "complete":
        try:
            fingerprint_path = directory / "run_fingerprint.txt"
            expected = (
                fingerprint_path.read_text(encoding="ascii").strip()
                if fingerprint_path.exists()
                else None
            )
            results.load_task_decoding_run(directory, expected_run_fingerprint=expected)
        except (OSError, ValueError):
            status["result_validation"] = "invalid"
        else:
            status["result_validation"] = "valid"
    return status
