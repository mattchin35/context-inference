"""Portable result storage and scientific-identity contracts for task decoding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from collections.abc import Mapping, Sequence

import numpy as np

from src.neural_analysis.task_decoding.config import TARGET_DEFINITION_BY_IDENTIFIER
from src.neural_analysis.task_decoding.modeling import make_inner_splits, make_outer_splits
from src.neural_analysis.task_decoding.targets import _CATEGORICAL_CLASS_LABELS


RESULT_SCHEMA_VERSION = 1
CONDITION_RESULT_SCHEMA_VERSION = 2
_MEBIBYTE = 1024 * 1024
_RESULT_FILE = "results.npz"
_CONFIG_FILE = "config.json"
_MANIFEST_FILE = "input_manifest.json"
_FEATURE_COPY_FILE = "trial_feature_params.json"
_REGIONS = ("PFC", "HPC", "PFC+HPC")
_REPRESENTATIONS = ("pca", "units")
_METRICS = ("balanced_accuracy", "auc", "r2")
_STRUCTURED_INPUT_SUFFIXES = frozenset({".csv", ".json", ".py", ".tsv", ".txt"})
_TARGET_OUTER_UNAVAILABLE_CODE = "target_outer_unavailable"
_COMPACT_REASON_CODES = frozenset(
    {
        _TARGET_OUTER_UNAVAILABLE_CODE,
        "candidate_fit_failed",
        "fit_convergence_failure",
        "inner_candidate_unavailable",
        "inner_class_coverage_unavailable",
        "inner_plan_unavailable",
        "no_features",
        "no_valid_tuning_candidate",
        "nonfinite_inner_score",
        "outer_estimator_failure",
        "outer_features_unavailable",
        "outer_hpc_features_unavailable",
        "outer_pfc_features_unavailable",
        "pfc_pca_transform_unavailable",
        "pfc_transform_unavailable",
    }
)
_DIRECT_RUNTIME_PATHS = (
    "src/__init__.py",
    "src/neural_analysis/__init__.py",
    "src/neural_analysis/session_metadata.py",
    "src/neural_analysis/spike_behavior/__init__.py",
    "src/neural_analysis/spike_behavior/loading.py",
    "src/neural_analysis/population/__init__.py",
    "src/neural_analysis/population/pca.py",
    "src/behavior_analysis/__init__.py",
    "src/behavior_analysis/project_utils.py",
    "pyproject.toml",
    "uv.lock",
)
_TASK_DECODING_FILES = (
    "__init__.py",
    "activity.py",
    "config.py",
    "modeling.py",
    "results.py",
    "targets.py",
)


def _canonical_json(value: object) -> str:
    """Return deterministic JSON text for mappings, lists, and scalar metadata.

    Parameters
    ----------
    value : object
        JSON-compatible value without NumPy scalar or array objects.

    Returns
    -------
    str
        UTF-8-safe canonical JSON with sorted mapping keys.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_stream(path: Path) -> str:
    """Stream one file into SHA-256 without materializing its contents.

    Parameters
    ----------
    path : pathlib.Path
        Existing local file whose byte identity is required.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_MEBIBYTE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_relative_path(root: Path, path: Path) -> str:
    """Return a portable path only when ``path`` is contained by ``root``.

    Parameters
    ----------
    root : pathlib.Path
        Canonical session or repository root.
    path : pathlib.Path
        Existing explicit source path.

    Returns
    -------
    str
        POSIX root-relative path.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "Input path must be contained by its declared root."
        ) from error


def _file_identity(root: Path, path: Path) -> dict[str, object]:
    """Build one portable file identity under the approved 64-MiB policy.

    Parameters
    ----------
    root : pathlib.Path
        Session root used to derive the portable relative path.
    path : pathlib.Path
        Explicit contained regular file.

    Returns
    -------
    dict[str, object]
        Relative path, byte size, whole-second mtime, and SHA-256 only for
        structured files at or below 64 MiB.
    """
    if not path.is_file():
        raise ValueError(f"Manifest input is not a file: {path}")
    stat = path.stat()
    identity: dict[str, object] = {
        "relative_path": _contained_relative_path(root, path),
        "size_bytes": stat.st_size,
        "mtime_seconds": int(stat.st_mtime),
    }
    if (
        path.suffix.lower() in _STRUCTURED_INPUT_SUFFIXES
        and stat.st_size <= 64 * _MEBIBYTE
    ):
        identity["sha256"] = _sha256_stream(path)
    return identity


def build_input_manifest(
    session_root: Path,
    files: Mapping[str, Path],
    file_lists: Mapping[str, Sequence[Path]],
) -> dict[str, object]:
    """Record portable identities for exact explicit task-decoding inputs.

    Parameters
    ----------
    session_root : pathlib.Path
        Root containing every declared file, without a trailing unit change.
    files : mapping[str, pathlib.Path]
        Named individual inputs such as JSON/CSV parameters.
    file_lists : mapping[str, sequence[pathlib.Path]]
        Named explicit directory members the loader will open; unspecified
        neighbours are deliberately excluded.

    Returns
    -------
    dict[str, object]
        JSON-safe manifest with whole-second mtimes and streamed small-file
        SHA-256 identities.
    """
    root = Path(session_root)
    return {
        "identity_policy": {
            "hash_algorithm": "sha256",
            "hash_threshold_bytes": 64 * _MEBIBYTE,
            "mtime_resolution": "whole_seconds",
        },
        "files": {
            name: _file_identity(root, Path(path))
            for name, path in sorted(files.items())
        },
        "file_lists": {
            name: sorted(
                (_file_identity(root, Path(path)) for path in paths),
                key=lambda item: item["relative_path"],
            )
            for name, paths in sorted(file_lists.items())
        },
    }


def _scoped_source_paths(repository_root: Path) -> tuple[str, ...]:
    """List reviewed task package and direct runtime paths relative to one checkout.

    Parameters
    ----------
    repository_root : pathlib.Path
        Root of the checked-out source tree.

    Returns
    -------
    tuple[str, ...]
        Sorted required file paths, including every task-decoding Python file.
    """
    package = repository_root / "src/neural_analysis/task_decoding"
    task_paths = [
        path.relative_to(repository_root).as_posix() for path in package.glob("*.py")
    ]
    for filename in _TASK_DECODING_FILES:
        required_path = package / filename
        if not required_path.is_file():
            relative_path = required_path.relative_to(repository_root)
            raise ValueError(
                f"Required scientific source is missing: {relative_path}"
            )
    required = set(task_paths) | set(_DIRECT_RUNTIME_PATHS)
    for relative_path in _DIRECT_RUNTIME_PATHS:
        if not (repository_root / relative_path).is_file():
            raise ValueError(f"Required scientific source is missing: {relative_path}")
    if not package.is_dir():
        raise ValueError("Required task-decoding source package is missing.")
    return tuple(sorted(required))


def scientific_source_fingerprint(repository_root: Path) -> dict[str, object]:
    """Hash only reviewed scientific source and direct dependency contents.

    Parameters
    ----------
    repository_root : pathlib.Path
        Root of a source checkout; absolute prefix is excluded from identity.

    Returns
    -------
    dict[str, object]
        Portable per-file SHA-256 records and one combined ``fingerprint``.
    """
    root = Path(repository_root)
    paths = _scoped_source_paths(root)
    files: list[dict[str, str]] = []
    for relative_path in paths:
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"Required scientific source is missing: {relative_path}")
        files.append({"relative_path": relative_path, "sha256": _sha256_stream(path)})
    fingerprint = hashlib.sha256(_canonical_json(files).encode("utf-8")).hexdigest()
    return {"files": files, "fingerprint": fingerprint}


def validate_scientific_source_cleanliness(repository_root: Path) -> None:
    """Reject dirty/untracked relevant Python or untracked direct dependencies.

    Parameters
    ----------
    repository_root : pathlib.Path
        Git checkout containing the scoped scientific source.

    Returns
    -------
    None
        Raises ValueError for relevant dirty, untracked, or untracked-required
        source state while ignoring unrelated paths.
    """
    root = Path(repository_root)
    scoped = set(_scoped_source_paths(root))
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for line in status:
        path = line[3:].strip().split(" -> ")[-1].strip().strip('"')
        untracked_package_python = (
            path.startswith("src/neural_analysis/task_decoding/")
            and path.endswith(".py")
        )
        relevant = path in scoped or untracked_package_python
        if relevant:
            raise ValueError(f"Relevant scientific source is not clean: {path}")
    tracked = set(
        subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    for relative_path in scoped:
        if relative_path not in tracked:
            raise ValueError(
                f"Required scientific source is not tracked: {relative_path}"
            )


def build_run_fingerprint(
    analysis_version: str,
    scientific_source_fingerprint: str,
    scientific_config: Mapping[str, object],
    input_manifest: Mapping[str, object],
    session_id: str,
) -> str:
    """Build a root-prefix-independent fingerprint from only scientific identity.

    Parameters
    ----------
    analysis_version : str
        Analysis schema/science version.
    scientific_source_fingerprint : str
        Scoped source digest.
    scientific_config, input_manifest : mapping[str, object]
        Canonical scientific serializer payloads without execution settings.
    session_id : str
        Stable session identifier.

    Returns
    -------
    str
        SHA-256 hexadecimal run identity.
    """
    payload = {
        "analysis_version": analysis_version,
        "scientific_source_fingerprint": scientific_source_fingerprint,
        "scientific_config": scientific_config,
        "input_manifest": input_manifest,
        "session_id": session_id,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()



def _required_names() -> set[str]:
    """Return the immutable set of 50 result-array names.

    Returns
    -------
    set[str]
        Primitive result member names excluding scalar JSON ``meta``.
    """
    return {
        "target_labels",
        "target_families",
        "target_status",
        "target_unavailable_reasons",
        "target_source_labels",
        "target_positive_classes",
        "target_label_mappings_json",
        "region_labels",
        "representation_labels",
        "metric_labels",
        "time_bin_edges_s",
        "time_bin_centers_s",
        "fold_labels",
        "fold_scores",
        "fit_status",
        "fit_reason_codes",
        "requested_feature_counts",
        "effective_feature_counts",
        "train_counts",
        "test_counts",
        "train_class_counts",
        "test_class_counts",
        "full_table_row_positions",
        "full_table_trial_ids",
        "full_table_block_ids_utf8",
        "full_table_block_id_offsets",
        "encoded_target_values",
        "eligibility_masks",
        "eligibility_reason_codes",
        "eligibility_counts",
        "trial_row_indices",
        "outer_fold_ids",
        "unit_selection_rules_json",
        "coefficient_values",
        "coefficient_feature_ids",
        "coefficient_feature_regions",
        "coefficient_feature_statuses",
        "coefficient_active_masks",
        "fitted_intercepts",
        "fixed_parameters_json",
        "candidate_parameter_json",
        "selected_parameters_json",
        "inner_selection_fold_ids",
        "candidate_inner_scores",
        "candidate_inner_statuses",
        "candidate_inner_reasons",
        "selected_candidate_indices",
        "stage_timing_labels",
        "stage_timing_seconds",
        "total_timing_seconds",
    }


_CONDITION_DEPENDENT_ARRAYS = frozenset(
    {
        "target_status",
        "target_unavailable_reasons",
        "fold_scores",
        "fit_status",
        "fit_reason_codes",
        "requested_feature_counts",
        "effective_feature_counts",
        "train_counts",
        "test_counts",
        "train_class_counts",
        "test_class_counts",
        "eligibility_masks",
        "eligibility_reason_codes",
        "eligibility_counts",
        "outer_fold_ids",
        "coefficient_values",
        "fitted_intercepts",
        "fixed_parameters_json",
        "selected_parameters_json",
        "inner_selection_fold_ids",
        "candidate_inner_scores",
        "candidate_inner_statuses",
        "candidate_inner_reasons",
        "selected_candidate_indices",
    }
)


def _arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare two candidate shared arrays without dtype or shape coercion.

    Parameters
    ----------
    first, second : numpy.ndarray
        Arrays proposed for one condition-independent schema member. Shapes,
        axis conventions, and physical units must already be identical.

    Returns
    -------
    bool
        True only for identical dtype, shape, and values; corresponding
        floating-point NaN sentinels compare equal.
    """
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    if first.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def assemble_condition_result_payload(
    *,
    condition_names: Sequence[str],
    condition_masks: Mapping[str, np.ndarray],
    condition_payloads: Mapping[
        str,
        tuple[Mapping[str, np.ndarray], Mapping[str, object]],
    ],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Stack validated pooled-schema payloads into one condition-axis payload.

    Parameters
    ----------
    condition_names : sequence[str]
        Nonempty canonical condition order.
    condition_masks : mapping[str, numpy.ndarray]
        Full-table Boolean membership arrays with shape ``(full_trial,)``.
        These preserve overlapping project conditions explicitly.
    condition_payloads : mapping[str, tuple[mapping, mapping]]
        One complete schema-1 ``(arrays, meta)`` pair per condition. Each pair
        already represents condition-filtered target eligibility and folds.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, object]]
        Schema-2 arrays and metadata. Condition-dependent members gain a
        leading condition axis; stable target, feature, time, and trial
        identities remain single-copy.

    Raises
    ------
    ValueError
        If order, masks, per-condition schemas, or shared identities disagree.
    """
    names = tuple(condition_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Condition result names must be nonempty and unique.")
    if set(condition_masks) != set(names) or set(condition_payloads) != set(names):
        raise ValueError("Condition payload and mask keys must match condition order.")

    validated: dict[str, tuple[Mapping[str, np.ndarray], Mapping[str, object]]] = {}
    for name in names:
        arrays, meta = condition_payloads[name]
        if (
            set(arrays) != _required_names()
            or meta.get("schema_version") != RESULT_SCHEMA_VERSION
        ):
            raise ValueError("Each condition payload must use the complete pooled schema.")
        validated[name] = (arrays, meta)

    first_arrays, first_meta = validated[names[0]]
    full_count = int(first_arrays["full_table_row_positions"].size)
    normalized_masks: list[np.ndarray] = []
    for name in names:
        mask = np.asarray(condition_masks[name])
        if mask.dtype != np.bool_ or mask.shape != (full_count,):
            raise ValueError("Condition masks must be Boolean full-table vectors.")
        normalized_masks.append(mask.copy())
    if "all" in names and not np.all(normalized_masks[names.index("all")]):
        raise ValueError("The all condition mask must contain every full-table row.")

    output: dict[str, np.ndarray] = {
        "condition_labels": np.asarray(names, dtype="U32"),
        "condition_masks": np.stack(normalized_masks, axis=0),
    }
    merged_values = np.full_like(first_arrays["encoded_target_values"], np.nan)
    for array_name in _required_names():
        values = [validated[name][0][array_name] for name in names]
        if array_name in _CONDITION_DEPENDENT_ARRAYS:
            output[array_name] = np.stack(values, axis=0)
            continue
        if array_name == "encoded_target_values":
            for values_for_condition in values:
                finite = np.isfinite(values_for_condition)
                conflict = finite & np.isfinite(merged_values) & (
                    values_for_condition != merged_values
                )
                if np.any(conflict):
                    raise ValueError("Condition payload target values disagree.")
                merged_values[finite] = values_for_condition[finite]
            output[array_name] = merged_values
            continue
        if any(not _arrays_equal(values[0], other) for other in values[1:]):
            raise ValueError(f"Condition-independent result array disagrees: {array_name}")
        output[array_name] = values[0].copy()

    condition_membership = output["condition_masks"][:, np.newaxis, :]
    if np.any(output["eligibility_masks"] & ~condition_membership):
        raise ValueError("Target eligibility extends outside its condition mask.")
    excluded = np.broadcast_to(
        ~condition_membership,
        output["eligibility_reason_codes"].shape,
    )
    output["eligibility_reason_codes"][excluded] = "condition_excluded"

    metadata = json.loads(json.dumps(first_meta))
    metadata["schema_version"] = CONDITION_RESULT_SCHEMA_VERSION
    axes = {
        "condition_labels": ["condition"],
        "condition_masks": ["condition", "full_table_row"],
    }
    base_axes = first_meta.get("axes")
    if not isinstance(base_axes, Mapping):
        raise ValueError("Condition payload metadata axes are invalid.")
    for array_name, array_axes in base_axes.items():
        axes[array_name] = (
            ["condition", *array_axes]
            if array_name in _CONDITION_DEPENDENT_ARRAYS
            else list(array_axes)
        )
    metadata["axes"] = axes
    parameters = metadata.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Condition payload parameters are invalid.")
    parameters["condition_names"] = list(names)
    return output, metadata


def _condition_expected_axes(mode: str) -> dict[str, list[str]]:
    """Return the schema-2 axes obtained from the pooled axis contract.

    Parameters
    ----------
    mode : {"fixed", "tuned"}
        Selects explicit not-applicable or nested-CV candidate axes.

    Returns
    -------
    dict[str, list[str]]
        Complete axis labels including the leading condition axis only for
        arrays whose scientific values vary by condition.
    """
    axes = {
        name: (
            ["condition", *array_axes]
            if name in _CONDITION_DEPENDENT_ARRAYS
            else array_axes
        )
        for name, array_axes in _expected_axes(mode).items()
    }
    axes["condition_labels"] = ["condition"]
    axes["condition_masks"] = ["condition", "full_table_row"]
    return axes


def _pooled_slice_from_condition_payload(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
    condition_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Project one schema-2 condition into the existing pooled validator.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Complete schema-2 primitive arrays. Condition-dependent members have
        leading ``condition`` axes; target values retain one shared
        ``(target, full_table_row)`` copy.
    meta : mapping[str, object]
        Complete schema-2 metadata.
    condition_index : int
        Zero-based condition-axis position.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, object]]
        Independent schema-1 view suitable for the established scientific and
        grouped-cross-validation validator.
    """
    pooled_arrays = {
        name: (
            np.asarray(arrays[name][condition_index]).copy()
            if name in _CONDITION_DEPENDENT_ARRAYS
            else np.asarray(arrays[name]).copy()
        )
        for name in _required_names()
    }
    # Values are stored once in schema 2. Restore the schema-1 NaN sentinel
    # outside this condition's target-specific eligibility before validation.
    pooled_values = pooled_arrays["encoded_target_values"]
    pooled_values[~pooled_arrays["eligibility_masks"]] = np.nan

    pooled_meta = json.loads(json.dumps(meta))
    pooled_meta["schema_version"] = RESULT_SCHEMA_VERSION
    pooled_meta["axes"] = _expected_axes(
        str(pooled_meta.get("parameters", {}).get("regularization_mode"))
    )
    parameters = pooled_meta.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Condition result parameters are invalid.")
    parameters.pop("condition_names", None)
    return pooled_arrays, pooled_meta


def _validate_condition_arrays(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    """Validate schema-2 condition axes through the pooled scientific contract.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Primitive schema-2 arrays. Score axes are ``(condition, target,
        region, representation, metric, time, fold)``; stable identities omit
        the condition axis. Times are seconds and scores are dimensionless.
    meta : mapping[str, object]
        Complete schema-2 metadata, units, axes, and provenance.
    config : mapping[str, object]
        Scientific serializer payload defining canonical condition order.

    Returns
    -------
    None
        Raises ValueError for condition identity, shape, membership, metadata,
        or any existing pooled-result scientific-contract violation.
    """
    required_names = _required_names() | {"condition_labels", "condition_masks"}
    if set(arrays) != required_names:
        raise ValueError("Required condition result schema array is missing or unexpected.")
    if (
        not isinstance(meta, Mapping)
        or meta.get("schema_version") != CONDITION_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("Condition result schema version is invalid.")

    configured_names = config.get("condition_names")
    if not isinstance(configured_names, list) or not configured_names:
        raise ValueError("Scientific config condition identity is invalid.")
    condition_count = len(configured_names)
    _require(
        arrays["condition_labels"],
        np.dtype("U32"),
        (condition_count,),
        "condition_labels",
    )
    labels = arrays["condition_labels"].tolist()
    if labels != configured_names or len(set(labels)) != condition_count:
        raise ValueError("Condition label identity disagrees with scientific config.")

    full_count = int(arrays["full_table_row_positions"].size)
    _require(
        arrays["condition_masks"],
        np.dtype(np.bool_),
        (condition_count, full_count),
        "condition_masks",
    )
    if "all" in labels and not np.all(
        arrays["condition_masks"][labels.index("all")]
    ):
        raise ValueError("The all condition mask must contain every full-table row.")

    mode = config.get("regularization_mode")
    if mode not in {"fixed", "tuned"}:
        raise ValueError("Scientific config regularization mode is invalid.")
    if meta.get("axes") != _condition_expected_axes(str(mode)):
        raise ValueError("Condition metadata axes order or schema is invalid.")
    parameters = meta.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or parameters.get("condition_names") != configured_names
    ):
        raise ValueError("Metadata condition parameters disagree with scientific config.")

    eligibility = arrays["eligibility_masks"]
    expected_eligibility_shape = (
        condition_count,
        int(arrays["target_labels"].size),
        full_count,
    )
    if eligibility.dtype != np.bool_ or eligibility.shape != expected_eligibility_shape:
        raise ValueError("Condition eligibility mask dtype or shape is invalid.")
    if np.any(eligibility & ~arrays["condition_masks"][:, np.newaxis, :]):
        raise ValueError("Target eligibility extends outside its condition mask.")

    for condition_index in range(condition_count):
        pooled_arrays, pooled_meta = _pooled_slice_from_condition_payload(
            arrays,
            meta,
            condition_index,
        )
        _validate_arrays(pooled_arrays, pooled_meta, config)


def _validate_saved_arrays(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    """Dispatch a saved result to its immutable schema validator.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Complete schema-1 or schema-2 primitive result members.
    meta, config : mapping[str, object]
        Saved metadata and scientific configuration.

    Returns
    -------
    None
        Raises ValueError for an unsupported schema or validation failure.
    """
    schema_version = meta.get("schema_version") if isinstance(meta, Mapping) else None
    if schema_version == RESULT_SCHEMA_VERSION:
        _validate_arrays(arrays, meta, config)
    elif schema_version == CONDITION_RESULT_SCHEMA_VERSION:
        _validate_condition_arrays(arrays, meta, config)
    else:
        raise ValueError("Result schema version is invalid.")



def _require(
    array: np.ndarray,
    dtype: np.dtype,
    shape: tuple[int, ...],
    name: str,
) -> None:
    """Require one primitive array's exact portable dtype, shape, and axis order.

    Parameters
    ----------
    array : numpy.ndarray
        Primitive non-pickle result member with its declared ordered axes.
    dtype : numpy.dtype
        Required portable NumPy dtype.
    shape : tuple[int, ...]
        Exact lengths in the member's documented axis order.
    name : str
        Schema member name used in a causal validation error.

    Returns
    -------
    None
        Raises ValueError when dtype or shape differs from the schema.
    """
    if not isinstance(array, np.ndarray) or array.dtype == object:
        raise ValueError(f"Schema dtype is unsafe or invalid for {name}.")
    if array.dtype != dtype:
        raise ValueError(f"Schema dtype mismatch for {name}.")
    if array.shape != shape:
        raise ValueError(f"Schema shape mismatch for {name}.")


def _validate_compact_reason_codes(array: np.ndarray, name: str) -> None:
    """Require saved unavailable reasons to use one declared compact code.

    Parameters
    ----------
    array : numpy.ndarray
        Unicode reason array on any declared fit or candidate-audit axes. Empty
        values are allowed only where the corresponding status is valid.
    name : str
        Schema member name used in a causal validation error.

    Returns
    -------
    None
        Raises ValueError for unknown or prose-like saved unavailable reasons.
    """
    values = {str(value) for value in array.flat if str(value)}
    unknown = values - _COMPACT_REASON_CODES
    if unknown:
        raise ValueError(f"Unknown compact reason code in {name}: {sorted(unknown)!r}")


def _expected_axes(mode: str) -> dict[str, list[str]]:
    """Return the complete ordered axis contract for every primitive member.

    Parameters
    ----------
    mode : {"fixed", "tuned"}
        Selects explicit not-applicable or nested-CV candidate axes.

    Returns
    -------
    dict[str, list[str]]
        Array names mapped to complete axis labels; scalar arrays use ``[]``.
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
        audit_axes = [
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
                "inner_selection_fold_ids": [
                    "target",
                    "outer_fold",
                    "common_neural_tensor_row",
                ],
                "candidate_inner_scores": audit_axes,
                "candidate_inner_statuses": audit_axes,
                "candidate_inner_reasons": audit_axes,
                "selected_candidate_indices": audit_axes[:-2],
            }
        )
    return axes


def _decode_blocks(arrays: Mapping[str, np.ndarray]) -> tuple[bytes, ...]:
    """Decode canonical scalar JSON block IDs from byte shape and boundary axes.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Primitive schema with one-dimensional uint8 bytes and int64 row-boundary
        offsets; block labels are dimensionless grouping identities.

    Returns
    -------
    tuple[bytes, ...]
        Canonical JSON bytes in full-table-row axis order.
    """
    payload = arrays["full_table_block_ids_utf8"]
    offsets = arrays["full_table_block_id_offsets"]
    full_rows = arrays["full_table_row_positions"].size
    _require(payload, np.dtype(np.uint8), (payload.size,), "full_table_block_ids_utf8")
    _require(offsets, np.dtype(np.int64), (full_rows + 1,), "full_table_block_id_offsets")
    if offsets[0] != 0 or offsets[-1] != payload.size or np.any(np.diff(offsets) <= 0):
        raise ValueError("Block identity offsets are invalid or contain empty segments.")
    blocks: list[bytes] = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        value = payload[int(start) : int(stop)].tobytes()
        try:
            decoded = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Block identity JSON is invalid.") from error
        if isinstance(decoded, (dict, list)):
            raise ValueError("Block identity label must be a scalar JSON value.")
        if _canonical_json(decoded).encode("utf-8") != value:
            raise ValueError("Block identity JSON is not canonical.")
        blocks.append(value)
    return tuple(blocks)


def _decode_block_scalars(blocks: Sequence[bytes]) -> np.ndarray:
    """Decode canonical JSON block bytes to scalar grouping labels.

    Parameters
    ----------
    blocks : sequence[bytes]
        Canonical scalar JSON byte identities in full-table-row order. These
        preserve collision-safe raw identities for leakage validation.

    Returns
    -------
    numpy.ndarray
        One-dimensional object array of decoded JSON scalar labels in the
        same full-table-row axis order, with dimensionless group identity.
    """
    decoded = np.empty(len(blocks), dtype=object)
    for index, payload in enumerate(blocks):
        decoded[index] = json.loads(payload.decode("utf-8"))
    return decoded


def _family_for_target(label: str) -> str:
    """Return the persisted model family for the supported target label.

    Parameters
    ----------
    label : str
        Case-sensitive saved target identifier.

    Returns
    -------
    str
        ``"categorical"`` or ``"numerical"`` for the saved target.
    """
    try:
        return TARGET_DEFINITION_BY_IDENTIFIER[label].family
    except KeyError as error:
        raise ValueError(f"Unknown saved target family: {label}") from error


def _target_mapping_json(label: str, family: str) -> str:
    """Return the canonical saved target-label mapping for one target definition.

    Parameters
    ----------
    label : str
        Case-sensitive target identifier declared by the scientific config.
    family : {"categorical", "numerical"}
        Model family resolved from the shared target definition.

    Returns
    -------
    str
        Canonical scalar JSON mapping. Categorical values map encoded classes
        ``0`` and ``1`` to labels; numerical values retain native targets.
    """
    if family == "categorical":
        try:
            return _canonical_json(_CATEGORICAL_CLASS_LABELS[label])
        except KeyError as error:
            raise ValueError(f"Categorical target mapping is unknown: {label}") from error
    if family == "numerical":
        return _canonical_json({"native_target": True})
    raise ValueError(f"Unknown target family for label mapping: {family}")


def _parameter_json(value: Mapping[str, object]) -> str:
    """Return the canonical parameter JSON used by sklearn audit arrays.

    Parameters
    ----------
    value : mapping[str, object]
        Family-specific scalar regularization parameters.

    Returns
    -------
    str
        Sorted ordinary JSON text for one parameter setting.
    """
    return json.dumps(value, sort_keys=True)


def _family_candidates(config: Mapping[str, object], family: str) -> list[str]:
    """Expand the frozen family grid in saved candidate-axis order.

    Parameters
    ----------
    config : mapping[str, object]
        Scientific serializer payload containing the frozen tuning grid.
    family : {"categorical", "numerical"}
        Selects LogisticRegression ``C`` or ElasticNet ``alpha`` values.

    Returns
    -------
    list[str]
        Fifteen canonical parameter JSON values on the candidate axis.
    """
    estimator = "LogisticRegression" if family == "categorical" else "ElasticNet"
    strength = "C" if family == "categorical" else "alpha"
    try:
        grid = config["frozen_controls"]["tuning_grid"][estimator]
        return [
            _parameter_json({strength: value, "l1_ratio": ratio})
            for value in grid[strength]
            for ratio in grid["l1_ratio"]
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("Frozen tuning grid is missing from scientific config.") from error


def _fixed_parameter(config: Mapping[str, object], family: str) -> str:
    """Return the family-specific frozen default setting as canonical JSON.

    Parameters
    ----------
    config : mapping[str, object]
        Scientific serializer payload containing frozen estimator controls.
    family : {"categorical", "numerical"}
        Selects the LogisticRegression or ElasticNet default controls.

    Returns
    -------
    str
        Canonical JSON with regularization strength and ``l1_ratio``.
    """
    estimator = "LogisticRegression" if family == "categorical" else "ElasticNet"
    strength = "C" if family == "categorical" else "alpha"
    try:
        controls = config["frozen_controls"]["estimators"][estimator]
        return _parameter_json({strength: controls[strength], "l1_ratio": controls["l1_ratio"]})
    except (KeyError, TypeError) as error:
        raise ValueError("Frozen estimator controls are missing from scientific config.") from error


def _expected_units(labels: Sequence[str]) -> dict[str, object]:
    """Return target-labelled model and physical-unit metadata for saved arrays.

    Parameters
    ----------
    labels : sequence[str]
        Target-axis labels in saved order.

    Returns
    -------
    dict[str, object]
        Exact units for target values, scores, parameters, time, and timing axes.
    """
    encoded: dict[str, str] = {}
    scores: dict[str, dict[str, str]] = {}
    candidate: dict[str, dict[str, str]] = {}
    coefficients: dict[str, str] = {}
    intercepts: dict[str, str] = {}
    for label in labels:
        if _family_for_target(label) == "categorical":
            encoded[label] = "encoded class"
            scores[label] = {"balanced_accuracy": "fraction", "auc": "fraction"}
            candidate[label] = {"balanced_accuracy": "fraction"}
            coefficients[label] = "log-odds change per pooled training standard deviation"
            intercepts[label] = "log-odds"
        else:
            encoded[label] = "unitless"
            scores[label] = {"r2": "coefficient_of_determination"}
            candidate[label] = {"r2": "coefficient_of_determination"}
            coefficients[label] = "unitless per pooled training standard deviation"
            intercepts[label] = "unitless"
    return {
        "encoded_target_values": encoded,
        "fold_scores": scores,
        "candidate_inner_scores": candidate,
        "coefficient_values": coefficients,
        "fitted_intercepts": intercepts,
        "time_bin_edges_s": "s",
        "time_bin_centers_s": "s",
        "stage_timing_seconds": "s",
        "total_timing_seconds": "s",
    }


def _validate_meta(
    meta: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    mode: str,
) -> None:
    """Validate named result metadata against array axes, units, and frozen config.

    Parameters
    ----------
    meta : mapping[str, object]
        JSON-safe schema/provenance metadata.
    arrays : mapping[str, numpy.ndarray]
        Primitive result arrays whose dimensions define named axes.
    config : mapping[str, object]
        Serializer-owned scientific configuration.
    mode : {"fixed", "tuned"}
        Current regularization mode.

    Returns
    -------
    None
        Raises ValueError for missing metadata, axes, physical units, or identity drift.
    """
    required = {
        "schema_version",
        "analysis_version",
        "random_seed",
        "units",
        "axes",
        "paths",
        "parameters",
        "provenance",
        "warnings",
    }
    if not isinstance(meta, Mapping) or not required.issubset(meta):
        raise ValueError("Required meta mapping is missing fields.")
    if meta["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("Result schema version is invalid.")
    if meta["analysis_version"] != config.get("analysis_version"):
        raise ValueError("Result analysis version is invalid.")
    controls = config.get("frozen_controls")
    if not isinstance(controls, Mapping) or meta["random_seed"] != controls.get("seed"):
        raise ValueError("Result random seed disagrees with frozen controls.")
    if meta["axes"] != _expected_axes(mode):
        raise ValueError("Metadata axes order or schema is invalid.")
    labels = [str(value) for value in arrays["target_labels"].tolist()]
    if meta["units"] != _expected_units(labels):
        raise ValueError("Metadata scientific units are invalid for saved targets.")
    paths = meta["paths"]
    if not isinstance(paths, Mapping) or paths != {
        "session_root": ".",
        "feature_parameter_copy": _FEATURE_COPY_FILE,
    }:
        raise ValueError("Metadata paths must be portable run-relative paths.")
    parameters = meta["parameters"]
    expected_metrics = {
        label: "balanced_accuracy" if _family_for_target(label) == "categorical" else "r2"
        for label in labels
    }
    if not isinstance(parameters, Mapping) or parameters != {
        "regularization_mode": mode,
        "outer_fold_count": config.get("outer_fold_count"),
        "inner_fold_count": config.get("inner_fold_count"),
        "bin_width_ms": config.get("bin_width_ms"),
        "candidate_selection_metrics": expected_metrics,
    }:
        raise ValueError("Metadata parameters disagree with scientific config.")
    provenance = meta["provenance"]
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("scientific_config_identity") != identity
        or not isinstance(provenance.get("run_fingerprint"), str)
        or not provenance["run_fingerprint"]
    ):
        raise ValueError("Metadata provenance/config identity is invalid.")
    if not isinstance(meta["warnings"], list):
        raise ValueError("Metadata warnings must be a list.")


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    """Validate saved result shapes, axes, units, identities, and grouped CV semantics.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        All 50 non-pickle primitives; score axes are ``(target, region,
        representation, metric, time, fold)`` and fit axes omit ``metric``.
        Times are seconds, scores are dimensionless, and coefficients/intercepts
        use the target-specific units recorded in ``meta``.
    meta : mapping[str, object]
        Required complete schema, physical-unit, axis, and provenance metadata.
    config : mapping[str, object]
        Scientific serializer payload defining target labels and frozen controls.

    Returns
    -------
    None
        Raises ValueError for any schema, identity, leakage, fit, or audit violation.
    """
    if set(arrays) != _required_names():
        raise ValueError("Required result schema array is missing or unexpected.")
    mode = config.get("regularization_mode")
    if mode not in {"fixed", "tuned"}:
        raise ValueError("Scientific config regularization mode is invalid.")
    parameters = meta.get("parameters") if isinstance(meta, Mapping) else None
    if not isinstance(parameters, Mapping) or parameters.get("regularization_mode") != mode:
        raise ValueError("Metadata parameters/result mode disagrees with scientific config.")
    _require(
        arrays["target_labels"],
        np.dtype("U32"),
        (arrays["target_labels"].size,),
        "target_labels",
    )
    labels = [str(value) for value in arrays["target_labels"].tolist()]
    if not labels or labels != config.get("target_names"):
        raise ValueError("Target label identity disagrees with scientific config.")
    families = [_family_for_target(label) for label in labels]
    target_n = len(labels)
    _require(arrays["target_status"], np.dtype("U16"), (target_n,), "target_status")
    _require(
        arrays["target_unavailable_reasons"],
        np.dtype("U128"),
        (target_n,),
        "target_unavailable_reasons",
    )
    for status, reason in zip(
        arrays["target_status"],
        arrays["target_unavailable_reasons"],
        strict=True,
    ):
        if status not in {"available", "unavailable"}:
            raise ValueError("Target status must be available or unavailable.")
        if (status == "available") != (reason == ""):
            raise ValueError("Target availability status and reason disagree.")
    for name, dtype, values in (
        ("target_families", np.dtype("U16"), families),
        ("region_labels", np.dtype("U16"), list(_REGIONS)),
        ("representation_labels", np.dtype("U16"), list(_REPRESENTATIONS)),
        ("metric_labels", np.dtype("U32"), list(_METRICS)),
    ):
        size = target_n if name == "target_families" else len(values)
        _require(arrays[name], dtype, (size,), name)
        if arrays[name].tolist() != values:
            raise ValueError(f"Required {name} label order is invalid.")
    region_n, representation_n, metric_n = 3, 2, 3
    _require(arrays["target_source_labels"], np.dtype("U32"), (target_n,), "target_source_labels")
    expected_sources = [
        TARGET_DEFINITION_BY_IDENTIFIER[label].source_column for label in labels
    ]
    if arrays["target_source_labels"].tolist() != expected_sources:
        raise ValueError("Target source label identity disagrees with target definition.")
    _require(
        arrays["target_positive_classes"],
        np.dtype(np.int64),
        (target_n,),
        "target_positive_classes",
    )
    _require(
        arrays["target_label_mappings_json"],
        np.dtype("U48"),
        (target_n,),
        "target_label_mappings_json",
    )
    for target, family in enumerate(families):
        positive = int(arrays["target_positive_classes"][target])
        if (family == "categorical" and positive != 1) or (
            family == "numerical" and positive != -1
        ):
            raise ValueError("Target positive-class metadata disagrees with family.")
        if arrays["target_label_mappings_json"][target] != _target_mapping_json(
            labels[target], family
        ):
            raise ValueError("Target label mapping disagrees with target definition.")
    time_n = arrays["time_bin_centers_s"].size
    fold_n = arrays["fold_labels"].size
    if time_n <= 0 or fold_n <= 0:
        raise ValueError("Time or fold dimensions must be positive.")
    _require(arrays["time_bin_edges_s"], np.dtype(np.float64), (time_n + 1,), "time_bin_edges_s")
    _require(arrays["time_bin_centers_s"], np.dtype(np.float64), (time_n,), "time_bin_centers_s")
    _require(arrays["fold_labels"], np.dtype(np.int64), (fold_n,), "fold_labels")
    if not np.array_equal(arrays["fold_labels"], np.arange(fold_n, dtype=np.int64)):
        raise ValueError("Fold label order is invalid.")
    if fold_n != config.get("outer_fold_count"):
        raise ValueError("Outer fold count disagrees with scientific config.")
    controls = config.get("frozen_controls", {})
    start = controls.get("window_start_s") if isinstance(controls, Mapping) else None
    end = controls.get("window_end_s") if isinstance(controls, Mapping) else None
    expected_time = int(round((float(end) - float(start)) * 1000 / config["bin_width_ms"]))
    if time_n != expected_time or not np.allclose(
        arrays["time_bin_edges_s"],
        np.linspace(float(start), float(end), time_n + 1),
    ) or not np.allclose(
        arrays["time_bin_centers_s"],
        (arrays["time_bin_edges_s"][:-1] + arrays["time_bin_edges_s"][1:]) / 2.0,
    ):
        raise ValueError("Time axis disagrees with frozen window or bin width.")
    fit_shape = (target_n, region_n, representation_n, time_n, fold_n)
    score_shape = (target_n, region_n, representation_n, metric_n, time_n, fold_n)
    full_n = arrays["full_table_row_positions"].size
    common_n = arrays["trial_row_indices"].size
    feature_n = arrays["coefficient_feature_ids"].shape[-1]
    if full_n <= 0 or common_n <= 0 or feature_n <= 0:
        raise ValueError("Full-table, common-row, or feature dimensions are invalid.")
    checks = {
        "fold_scores": (np.dtype(np.float64), score_shape),
        "fit_status": (np.dtype("U16"), fit_shape),
        "fit_reason_codes": (np.dtype("U32"), fit_shape),
        "requested_feature_counts": (np.dtype(np.int64), fit_shape),
        "effective_feature_counts": (np.dtype(np.int64), fit_shape),
        "train_counts": (np.dtype(np.int64), fit_shape),
        "test_counts": (np.dtype(np.int64), fit_shape),
        "train_class_counts": (np.dtype(np.int64), (*fit_shape, 2)),
        "test_class_counts": (np.dtype(np.int64), (*fit_shape, 2)),
        "full_table_row_positions": (np.dtype(np.int64), (full_n,)),
        "full_table_trial_ids": (np.dtype(np.int64), (full_n,)),
        "encoded_target_values": (np.dtype(np.float64), (target_n, full_n)),
        "eligibility_masks": (np.dtype(np.bool_), (target_n, full_n)),
        "eligibility_reason_codes": (np.dtype("U32"), (target_n, full_n)),
        "eligibility_counts": (np.dtype(np.int64), (target_n,)),
        "trial_row_indices": (np.dtype(np.int64), (common_n,)),
        "outer_fold_ids": (np.dtype(np.int64), (target_n, common_n)),
        "coefficient_values": (np.dtype(np.float64), (*fit_shape, feature_n)),
        "coefficient_feature_ids": (np.dtype("U32"), (region_n, representation_n, feature_n)),
        "coefficient_feature_regions": (np.dtype("U16"), (region_n, representation_n, feature_n)),
        "coefficient_feature_statuses": (np.dtype("U16"), (region_n, representation_n, feature_n)),
        "coefficient_active_masks": (np.dtype(np.bool_), (region_n, representation_n, feature_n)),
        "fitted_intercepts": (np.dtype(np.float64), fit_shape),
        "fixed_parameters_json": (
            np.dtype("U48"),
            (target_n, fold_n, region_n, representation_n, time_n),
        ),
        "selected_parameters_json": (
            np.dtype("U48"),
            (target_n, fold_n, region_n, representation_n, time_n),
        ),
        "stage_timing_labels": (np.dtype("U16"), (arrays["stage_timing_labels"].size,)),
        "stage_timing_seconds": (np.dtype(np.float64), (arrays["stage_timing_labels"].size,)),
        "total_timing_seconds": (np.dtype(np.float64), ()),
    }
    for name, (dtype, shape) in checks.items():
        _require(arrays[name], dtype, shape, name)
    selection_rules = arrays["unit_selection_rules_json"]
    if selection_rules.dtype.kind != "U" or selection_rules.shape != (2,):
        raise ValueError("Selection-rule JSON schema dtype or shape is invalid.")
    for value in selection_rules.tolist():
        try:
            decoded_rule = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("Selection-rule JSON is invalid.") from error
        if not isinstance(decoded_rule, dict) or _canonical_json(decoded_rule) != value:
            raise ValueError("Selection-rule JSON must be a canonical object.")
    _validate_compact_reason_codes(arrays["fit_reason_codes"], "fit_reason_codes")
    _validate_compact_reason_codes(
        arrays["candidate_inner_reasons"],
        "candidate_inner_reasons",
    )
    if not np.array_equal(arrays["full_table_row_positions"], np.arange(full_n, dtype=np.int64)):
        raise ValueError("Full-table row identity is invalid.")
    if not np.array_equal(arrays["full_table_trial_ids"], np.arange(full_n, dtype=np.int64)):
        raise ValueError("Full-table trial identity is invalid.")
    trial_rows = arrays["trial_row_indices"]
    if np.any(trial_rows < 0) or np.any(trial_rows >= full_n) or np.any(np.diff(trial_rows) <= 0):
        raise ValueError("Common tensor trial-row identity is invalid.")
    blocks = _decode_blocks(arrays)
    _validate_meta(meta, arrays, config, mode)
    _validate_targets_and_outer(arrays, families, blocks, trial_rows, fold_n)
    _validate_features_and_fits(arrays, families, config, fit_shape)
    _validate_parameters_and_tuning(arrays, families, config, mode, blocks, trial_rows)


def _validate_targets_and_outer(
    arrays: Mapping[str, np.ndarray],
    families: Sequence[str],
    blocks: Sequence[bytes],
    trial_rows: np.ndarray,
    fold_n: int,
) -> None:
    """Validate target eligibility, grouped outer folds, and class-count axes.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Validly shaped primitives with full-table and common-row axes.
    families : sequence[str]
        Target-axis categorical/numerical family labels.
    blocks : sequence[bytes]
        Full-row canonical grouping identities.
    trial_rows : numpy.ndarray
        Int64 common-row to full-row positions.
    fold_n : int
        Number of configured outer folds.

    Returns
    -------
    None
        Raises ValueError for target sentinels, group leakage, or count mismatch.
    """
    values = arrays["encoded_target_values"]
    masks = arrays["eligibility_masks"]
    reasons = arrays["eligibility_reason_codes"]
    outer = arrays["outer_fold_ids"]
    block_scalars = _decode_block_scalars(blocks)
    for target, family in enumerate(families):
        eligible = masks[target]
        if not np.array_equal(arrays["eligibility_counts"][target], eligible.sum()):
            raise ValueError("Eligibility count identity is invalid.")
        if np.any(~np.isfinite(values[target, eligible])):
            raise ValueError("Eligible target values must be finite.")
        if np.any(np.isfinite(values[target, ~eligible])):
            raise ValueError("Ineligible target values must use NaN sentinel.")
        if np.any(reasons[target, eligible] != "") or np.any(reasons[target, ~eligible] == ""):
            raise ValueError("Eligibility reason sentinel is invalid.")
        if family == "categorical" and not np.all(np.isin(values[target, eligible], [0.0, 1.0])):
            raise ValueError("Categorical target values must be encoded as 0/1.")
        target_status = arrays["target_status"][target]
        eligible_common = eligible[trial_rows]
        outer_plan = make_outer_splits(
            values[target, trial_rows[eligible_common]],
            block_scalars[trial_rows[eligible_common]],
            target_family=family,
            fold_count=fold_n,
        )
        if (target_status == "available") != outer_plan.is_available:
            raise ValueError(
                f"{family} target availability disagrees with grouped outer split plan."
            )
        if target_status == "unavailable":
            if arrays["target_unavailable_reasons"][target] != outer_plan.unavailable_reason:
                raise ValueError("Unavailable target reason disagrees with grouped outer split.")
            if np.any(outer[target] != -1):
                raise ValueError("Unavailable target must use only -1 outer-fold sentinels.")
            if not np.all(arrays["fit_status"][target] == "unavailable") or not np.all(
                arrays["fit_reason_codes"][target] == _TARGET_OUTER_UNAVAILABLE_CODE
            ):
                raise ValueError("Unavailable target fit status or compact reason is invalid.")
            if np.any(arrays["effective_feature_counts"][target] != 0):
                raise ValueError("Unavailable target effective feature count must be zero.")
            if np.any(arrays["train_counts"][target] != 0) or np.any(
                arrays["test_counts"][target] != 0
            ):
                raise ValueError("Unavailable target train/test counts must be zero.")
            class_sentinel = 0 if family == "categorical" else -1
            if not np.all(arrays["train_class_counts"][target] == class_sentinel) or not np.all(
                arrays["test_class_counts"][target] == class_sentinel
            ):
                raise ValueError("Unavailable target class-count sentinels are invalid.")
            if not np.isnan(arrays["fold_scores"][target]).all() or not np.isnan(
                arrays["coefficient_values"][target]
            ).all() or not np.isnan(arrays["fitted_intercepts"][target]).all():
                raise ValueError("Unavailable target output arrays must use NaN sentinels.")
            continue
        expected_outer = eligible_common
        if not np.array_equal(outer[target] >= 0, expected_outer):
            raise ValueError("Eligibility and outer-fold identity disagree.")
        if np.any((outer[target] < -1) | (outer[target] >= fold_n)):
            raise ValueError("Outer fold identifiers are invalid.")
        if any(np.count_nonzero(outer[target] == fold) == 0 for fold in range(fold_n)):
            raise ValueError("Outer grouped fold has no test rows.")
        for block in {blocks[int(trial_rows[row])] for row in np.flatnonzero(outer[target] >= 0)}:
            rows = [
                row
                for row in np.flatnonzero(outer[target] >= 0)
                if blocks[int(trial_rows[row])] == block
            ]
            if len({int(outer[target, row]) for row in rows}) != 1:
                raise ValueError("Outer block leakage detected.")
        for region, representation, time, fold in np.ndindex(
            arrays["fit_status"].shape[1:]
        ):
            train = (outer[target] >= 0) & (outer[target] != fold)
            test = outer[target] == fold
            index = (target, region, representation, time, fold)
            if (
                arrays["train_counts"][index] != np.count_nonzero(train)
                or arrays["test_counts"][index] != np.count_nonzero(test)
            ):
                raise ValueError("Train/test fold count is invalid.")
            if family == "categorical":
                target_values = values[target, trial_rows]
                train_counts = (
                    np.count_nonzero(target_values[train] == 0),
                    np.count_nonzero(target_values[train] == 1),
                )
                test_counts = (
                    np.count_nonzero(target_values[test] == 0),
                    np.count_nonzero(target_values[test] == 1),
                )
                if tuple(arrays["train_class_counts"][index]) != train_counts or tuple(
                    arrays["test_class_counts"][index]
                ) != test_counts:
                    raise ValueError("Categorical class count disagrees with fold labels.")
                if min(train_counts) == 0 or min(test_counts) == 0:
                    raise ValueError("Categorical outer fold lacks both classes.")
            elif not np.all(arrays["train_class_counts"][index] == -1) or not np.all(
                arrays["test_class_counts"][index] == -1
            ):
                raise ValueError("Numerical class-count sentinel is invalid.")


def _validate_features_and_fits(
    arrays: Mapping[str, np.ndarray],
    families: Sequence[str],
    config: Mapping[str, object],
    fit_shape: tuple[int, ...],
) -> None:
    """Validate padded feature identity and per-fit score/coefficient availability.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Shaped primitive result arrays with feature axis last for coefficients.
    families : sequence[str]
        Target model families in target-axis order.
    config : mapping[str, object]
        Frozen controls used to preserve requested feature counts.
    fit_shape : tuple[int, ...]
        ``(target, region, representation, time, fold)`` fit-axis shape.

    Returns
    -------
    None
        Raises ValueError for feature padding or fit-result semantic mismatch.
    """
    active = arrays["coefficient_active_masks"]
    feature_ids = arrays["coefficient_feature_ids"]
    feature_regions = arrays["coefficient_feature_regions"]
    feature_statuses = arrays["coefficient_feature_statuses"]
    coefficients = arrays["coefficient_values"]
    widths = np.zeros(active.shape[:2], dtype=np.int64)
    try:
        pfc_requested = int(config["pfc_pc_count"])
        hpc_requested = int(config["hpc_pc_count"])
        pfc_probe = str(config["pfc_region"]["probe_id"])
        hpc_probe = str(config["hpc_region"]["probe_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Scientific config lacks regional feature controls.") from error

    for region, representation in np.ndindex(active.shape[:2]):
        mask = active[region, representation]
        width = int(mask.sum())
        widths[region, representation] = width
        expected_prefix = np.arange(mask.size) < width
        if not np.array_equal(mask, expected_prefix):
            raise ValueError("Active coefficient features must form a prefix mask.")
        if np.any(
            mask
            & (
                (feature_ids[region, representation] == "")
                | (feature_statuses[region, representation] != "available")
            )
        ):
            raise ValueError("Active coefficient feature identifier or status is invalid.")
        if np.any(
            (~mask)
            & (
                (feature_ids[region, representation] != "")
                | (feature_regions[region, representation] != "")
                | (feature_statuses[region, representation] != "padding")
            )
        ):
            raise ValueError("Coefficient padding feature metadata is invalid.")
        if not np.all(np.isnan(coefficients[:, region, representation, :, :, ~mask])):
            raise ValueError("Coefficient padding must be NaN.")

        local_region = "PFC" if region == 0 else "HPC"
        if region < 2:
            if not np.all(feature_regions[region, representation, mask] == local_region):
                raise ValueError("Standalone feature region is invalid.")
            if representation == 0:
                expected_ids = [f"{local_region}:PC{number}" for number in range(1, width + 1)]
                if feature_ids[region, representation, mask].tolist() != expected_ids:
                    raise ValueError("Standalone PCA feature identity is invalid.")
                requested = pfc_requested if region == 0 else hpc_requested
                if width > requested:
                    raise ValueError("PCA feature width exceeds requested components.")
            else:
                probe = pfc_probe if region == 0 else hpc_probe
                ids = feature_ids[region, representation, mask].tolist()
                if len(set(ids)) != width or any(not item.startswith(f"{probe}:") for item in ids):
                    raise ValueError("Standalone direct-unit feature identity is invalid.")

    for representation in range(2):
        expected_width = int(widths[0, representation] + widths[1, representation])
        combined_mask = active[2, representation]
        if int(combined_mask.sum()) != expected_width:
            raise ValueError("Combined feature width must equal PFC then HPC widths.")
        expected_ids = np.concatenate(
            (
                feature_ids[0, representation, active[0, representation]],
                feature_ids[1, representation, active[1, representation]],
            )
        )
        expected_regions = np.concatenate(
            (
                feature_regions[0, representation, active[0, representation]],
                feature_regions[1, representation, active[1, representation]],
            )
        )
        ids_match = np.array_equal(
            feature_ids[2, representation, combined_mask], expected_ids
        )
        regions_match = np.array_equal(
            feature_regions[2, representation, combined_mask], expected_regions
        )
        if not ids_match or not regions_match:
            raise ValueError("Combined feature identity must preserve PFC then HPC order.")

    requested_widths = np.array(
        [
            [pfc_requested, widths[0, 1]],
            [hpc_requested, widths[1, 1]],
            [pfc_requested + hpc_requested, widths[0, 1] + widths[1, 1]],
        ],
        dtype=np.int64,
    )
    for index in np.ndindex(fit_shape):
        target, region, representation, time, fold = index
        status = arrays["fit_status"][index]
        reason = arrays["fit_reason_codes"][index]
        mask = active[region, representation]
        width = int(widths[region, representation])
        requested = int(arrays["requested_feature_counts"][index])
        effective = int(arrays["effective_feature_counts"][index])
        scores = arrays["fold_scores"][target, region, representation, :, time, fold]
        coefficients_cell = coefficients[index]
        intercept = arrays["fitted_intercepts"][index]
        if requested != requested_widths[region, representation]:
            raise ValueError("Requested feature count disagrees with configuration or feature map.")
        if effective < 0 or effective > min(requested, width):
            raise ValueError("Effective feature count exceeds mapped feature width.")
        expected_metrics = [0, 1] if families[target] == "categorical" else [2]
        irrelevant = [metric for metric in range(3) if metric not in expected_metrics]
        if status == "valid":
            finite_coefficients = np.isfinite(coefficients_cell[mask])
            if reason != "" or effective == 0 or np.count_nonzero(finite_coefficients) != effective:
                raise ValueError("Valid fit status/reason/effective feature count is invalid.")
            if not np.all(np.isnan(coefficients_cell[mask][~finite_coefficients])):
                raise ValueError("Dropped active coefficient must use NaN sentinel.")
            if not np.all(np.isfinite(scores[expected_metrics])) or not np.all(
                np.isnan(scores[irrelevant])
            ):
                raise ValueError("Valid fit score is nonfinite or has wrong metrics.")
            if not np.isfinite(intercept):
                raise ValueError("Valid fit intercept is nonfinite.")
        elif status == "unavailable":
            if reason == "":
                raise ValueError("Unavailable fit status requires a reason.")
            if not np.all(np.isnan(scores)):
                raise ValueError("Unavailable fit score must be NaN.")
            if not np.all(np.isnan(coefficients_cell)):
                raise ValueError("Unavailable fit coefficient must be NaN.")
            if not np.isnan(intercept):
                raise ValueError("Unavailable fit intercept must be NaN.")
        else:
            raise ValueError("Unknown fit status.")
    _validate_transform_reuse(arrays, widths, (pfc_requested, hpc_requested))


def _is_prefix_mask(mask: np.ndarray) -> bool:
    """Return whether one Boolean feature mask retains an initial prefix.

    Parameters
    ----------
    mask : numpy.ndarray
        One-dimensional Boolean feature-inclusion mask in stable feature-ID
        order. The feature axis is dimensionless.

    Returns
    -------
    bool
        True when all retained features precede all dropped features.
    """
    return np.array_equal(mask, np.arange(mask.size) < np.count_nonzero(mask))


def _validate_transform_reuse(
    arrays: Mapping[str, np.ndarray],
    widths: np.ndarray,
    requested_regional_pcs: tuple[int, int],
) -> None:
    """Validate one regional transform's reused feature masks across time bins.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Validly shaped fit arrays. Coefficients use ``(target, region,
        representation, time, fold, feature)`` axes; coefficients are in
        target-specific standardized model units.
    widths : numpy.ndarray
        Int64 ``(region, representation)`` active-map widths. These are
        dimensionless stable feature-ID capacities before fold-local dropout.
    requested_regional_pcs : tuple[int, int]
        Requested PFC then HPC PCA component counts.  These are dimensionless
        scientific controls from the saved configuration.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If one attempted fold changes its transform-derived width or finite
        coefficient mask across time, representations, or combined regional
        segments.
    """
    coefficients = arrays["coefficient_values"]
    active = arrays["coefficient_active_masks"]
    status = arrays["fit_status"]
    effective = arrays["effective_feature_counts"]
    selected = arrays["selected_candidate_indices"]
    target_n, _, _, time_n, fold_n = status.shape
    tuned = selected.size != 0

    def attempted(target: int, region: int, representation: int, time: int, fold: int) -> bool:
        """Return whether one persisted cell reached transform/outer-fit work."""
        if arrays["target_status"][target] != "available":
            return False
        if not tuned:
            return True
        return int(selected[target, fold, region, representation, time]) >= 0

    def finite_mask(
        target: int,
        region: int,
        representation: int,
        time: int,
        fold: int,
    ) -> np.ndarray:
        """Return finite coefficients on the stable active feature-ID prefix."""
        map_mask = active[region, representation]
        return np.isfinite(coefficients[target, region, representation, time, fold, map_mask])

    evidence: dict[tuple[int, int, int, int], tuple[int, np.ndarray | None]] = {}
    for target, fold, region, representation in np.ndindex(target_n, fold_n, 3, 2):
        masks: list[np.ndarray] = []
        effective_count: int | None = None
        for time in range(time_n):
            if not attempted(target, region, representation, time, fold):
                continue
            count = int(effective[target, region, representation, time, fold])
            if effective_count is None:
                effective_count = count
            elif count != effective_count:
                raise ValueError("Transform effective feature count changes across time bins.")
            if status[target, region, representation, time, fold] != "valid":
                continue
            mask = finite_mask(target, region, representation, time, fold)
            if representation == 0:
                if region < 2 and not _is_prefix_mask(mask):
                    raise ValueError("Regional PCA coefficient mask must retain a prefix.")
                if region == 2:
                    pfc_width = int(widths[0, representation])
                    hpc_width = int(widths[1, representation])
                    if not _is_prefix_mask(mask[:pfc_width]) or not _is_prefix_mask(
                        mask[pfc_width : pfc_width + hpc_width]
                    ):
                        raise ValueError(
                            "Combined PCA coefficient segments must retain regional prefixes."
                        )
            masks.append(mask)
        if effective_count is None:
            continue
        if masks and any(not np.array_equal(mask, masks[0]) for mask in masks[1:]):
            raise ValueError("Transform coefficient feature mask changes across time bins.")
        evidence[target, fold, region, representation] = (
            effective_count,
            masks[0] if masks else None,
        )

    regional_observations: dict[
        tuple[int, int, int, int], list[tuple[int, np.ndarray | None]]
    ] = {}

    def add_regional_observation(
        target: int,
        fold: int,
        region: int,
        representation: int,
        count: int,
        mask: np.ndarray | None,
    ) -> None:
        """Add one standalone or combined-segment transform observation.

        Parameters
        ----------
        target, fold, region, representation : int
            Zero-based axes identifying one target, outer fold, underlying PFC
            or HPC region, and PCA/direct representation.
        count : int
            Effective transform-derived feature count, dimensionless.
        mask : numpy.ndarray or None
            One-dimensional finite-coefficient mask in stable regional feature
            order for a valid fit, or None when no valid coefficient mask is
            available.

        Returns
        -------
        None
            Appends reusable evidence keyed by the underlying regional transform.
        """
        key = (target, fold, region, representation)
        regional_observations.setdefault(key, []).append((count, mask))

    def merge_regional_observations(
        observations: Mapping[tuple[int, int, int, int], list[tuple[int, np.ndarray | None]]],
    ) -> dict[tuple[int, int, int, int], tuple[int, np.ndarray | None]]:
        """Merge reusable counts and masks for each underlying regional transform.

        Parameters
        ----------
        observations : mapping
            Keys are ``(target, outer_fold, underlying_region,
            representation)``. Values hold effective counts and optional
            one-dimensional finite masks from standalone or combined fits.

        Returns
        -------
        dict
            One consistent effective count and optional finite mask for each
            underlying transform key. Counts are dimensionless feature counts.

        Raises
        ------
        ValueError
            If reusable transform count or valid coefficient-mask evidence
            disagrees across standalone and combined configurations.
        """
        merged: dict[tuple[int, int, int, int], tuple[int, np.ndarray | None]] = {}
        for key, entries in observations.items():
            counts = {count for count, _ in entries}
            if len(counts) != 1:
                raise ValueError(
                    "Combined and standalone transform effective feature counts disagree."
                )
            masks = [mask for _, mask in entries if mask is not None]
            if masks and any(not np.array_equal(mask, masks[0]) for mask in masks[1:]):
                raise ValueError(
                    "Combined regional feature mask disagrees with standalone evidence."
                )
            merged[key] = (counts.pop(), masks[0] if masks else None)
        return merged

    for (target, fold, region, representation), regional_evidence in evidence.items():
        if region < 2:
            add_regional_observation(
                target,
                fold,
                region,
                representation,
                *regional_evidence,
            )
            continue
        _, combined_mask = regional_evidence
        if combined_mask is None:
            continue
        pfc_width = int(widths[0, representation])
        hpc_width = int(widths[1, representation])
        pfc_mask = combined_mask[:pfc_width]
        hpc_mask = combined_mask[pfc_width : pfc_width + hpc_width]
        add_regional_observation(
            target,
            fold,
            0,
            representation,
            int(np.count_nonzero(pfc_mask)),
            pfc_mask,
        )
        add_regional_observation(
            target,
            fold,
            1,
            representation,
            int(np.count_nonzero(hpc_mask)),
            hpc_mask,
        )

    merged_evidence = merge_regional_observations(regional_observations)
    for target, fold, representation in np.ndindex(target_n, fold_n, 2):
        combined_evidence = evidence.get((target, fold, 2, representation))
        if combined_evidence is None or combined_evidence[0] != 0:
            continue
        pfc_key = (target, fold, 0, representation)
        hpc_key = (target, fold, 1, representation)
        pfc_evidence = merged_evidence.get(pfc_key)
        hpc_evidence = merged_evidence.get(hpc_key)
        if pfc_evidence is not None and pfc_evidence[0] > 0 and hpc_evidence is None:
            add_regional_observation(target, fold, 1, representation, 0, None)
        if hpc_evidence is not None and hpc_evidence[0] > 0 and pfc_evidence is None:
            add_regional_observation(target, fold, 0, representation, 0, None)
    merged_evidence = merge_regional_observations(regional_observations)

    for target, fold, representation in np.ndindex(target_n, fold_n, 2):
        combined_evidence = evidence.get((target, fold, 2, representation))
        pfc_evidence = merged_evidence.get((target, fold, 0, representation))
        hpc_evidence = merged_evidence.get((target, fold, 1, representation))
        if combined_evidence is None or pfc_evidence is None or hpc_evidence is None:
            continue
        combined_count, _ = combined_evidence
        pfc_count, _ = pfc_evidence
        hpc_count, _ = hpc_evidence
        expected_combined = 0 if pfc_count == 0 or hpc_count == 0 else pfc_count + hpc_count
        if combined_count != expected_combined:
            raise ValueError(
                "Combined transform effective features disagree with regional availability."
            )

    for target, fold, region in np.ndindex(target_n, fold_n, 2):
        pca_evidence = merged_evidence.get((target, fold, region, 0))
        direct_evidence = merged_evidence.get((target, fold, region, 1))
        if pca_evidence is None or direct_evidence is None:
            continue
        pca_count, _ = pca_evidence
        direct_count, _ = direct_evidence
        if (pca_count == 0) != (direct_count == 0):
            raise ValueError(
                "Regional PCA/direct transform availability disagrees across representations."
            )
        if pca_count == 0:
            continue
        outer_train_count = int(arrays["train_counts"][target, region, 1, 0, fold])
        expected_pca_count = min(
            requested_regional_pcs[region],
            direct_count,
            outer_train_count * time_n,
        )
        if pca_count != expected_pca_count:
            raise ValueError(
                "Regional PCA effective component count disagrees with the reusable transform."
            )


def _validate_parameters_and_tuning(
    arrays: Mapping[str, np.ndarray],
    families: Sequence[str],
    config: Mapping[str, object],
    mode: str,
    blocks: Sequence[bytes],
    trial_rows: np.ndarray,
) -> None:
    """Validate family controls and leakage-safe candidate/inner-fold audit arrays.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Complete primitive schema with parameter and candidate axes.
    families : sequence[str]
        Target model families in target-axis order.
    config : mapping[str, object]
        Frozen estimator and tuning-grid scientific controls.
    mode : {"fixed", "tuned"}
        Candidate-axis applicability.
    blocks : sequence[bytes]
        Full-row behavioral block identities.
    trial_rows : numpy.ndarray
        Common-row to full-row identity mapping.

    Returns
    -------
    None
        Raises ValueError for family controls, candidate audits, or inner leakage.
    """
    target_n = len(families)
    fold_n = arrays["fold_labels"].size
    time_n = arrays["time_bin_centers_s"].size
    parameter_shape = (target_n, fold_n, 3, 2, time_n)
    for target, family in enumerate(families):
        fixed = _fixed_parameter(config, family)
        if not np.all(arrays["fixed_parameters_json"][target] == fixed):
            raise ValueError("Fixed estimator parameters disagree with frozen controls.")
    if mode == "fixed":
        for name, dtype in (
            ("candidate_parameter_json", np.dtype("U48")),
            ("inner_selection_fold_ids", np.dtype(np.int64)),
            ("candidate_inner_scores", np.dtype(np.float64)),
            ("candidate_inner_statuses", np.dtype("U16")),
            ("candidate_inner_reasons", np.dtype("U48")),
            ("selected_candidate_indices", np.dtype(np.int64)),
        ):
            _require(arrays[name], dtype, (0,), name)
        if not np.all(arrays["selected_parameters_json"] == ""):
            raise ValueError("Fixed mode selected parameters must be empty.")
        return
    inner_n = config.get("inner_fold_count")
    if not isinstance(inner_n, int) or inner_n <= 0:
        raise ValueError("Inner fold count is invalid.")
    audit_shape = (target_n, fold_n, 3, 2, time_n, 15, inner_n)
    for name, dtype, shape in (
        ("candidate_parameter_json", np.dtype("U48"), (target_n, 15)),
        ("inner_selection_fold_ids", np.dtype(np.int64), (target_n, fold_n, trial_rows.size)),
        ("candidate_inner_scores", np.dtype(np.float64), audit_shape),
        ("candidate_inner_statuses", np.dtype("U16"), audit_shape),
        ("candidate_inner_reasons", np.dtype("U48"), audit_shape),
        ("selected_candidate_indices", np.dtype(np.int64), parameter_shape),
    ):
        _require(arrays[name], dtype, shape, name)
    for target, family in enumerate(families):
        candidate_text = arrays["candidate_parameter_json"][target].tolist()
        for value in candidate_text:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError("Candidate parameter JSON is invalid.") from error
            if _parameter_json(decoded) != value:
                raise ValueError("Candidate parameter JSON is not canonical.")
        if candidate_text != _family_candidates(config, family):
            raise ValueError("Candidate grid order or family parameters are invalid.")
    outer = arrays["outer_fold_ids"]
    inner = arrays["inner_selection_fold_ids"]
    block_scalars = _decode_block_scalars(blocks)
    for target in range(target_n):
        common_values = arrays["encoded_target_values"][target, trial_rows]
        common_blocks = np.empty(trial_rows.size, dtype=object)
        common_blocks[:] = [block_scalars[int(row)] for row in trial_rows]
        common_block_keys = np.empty(trial_rows.size, dtype=object)
        common_block_keys[:] = [blocks[int(row)] for row in trial_rows]
        for outer_fold in range(fold_n):
            assignment = inner[target, outer_fold]
            train = (outer[target] >= 0) & (outer[target] != outer_fold)
            held_out = ~train
            target_unavailable = arrays["target_status"][target] == "unavailable"
            if target_unavailable:
                unavailable_plan = True
            else:
                plan = make_inner_splits(
                    common_values[train],
                    common_blocks[train],
                    np.arange(np.count_nonzero(train), dtype=np.int64),
                    target_family=families[target],
                    fold_count=inner_n,
                )
                unavailable_plan = not plan.is_available
            if np.any(assignment[held_out] != -1):
                raise ValueError("Inner assignment includes outer-test or target-ineligible row.")
            if unavailable_plan:
                if np.any(assignment != -1):
                    raise ValueError("Unavailable inner plan must retain -1 assignments.")
            else:
                if np.any((assignment[train] < 0) | (assignment[train] >= inner_n)):
                    raise ValueError("Inner training assignment is missing or invalid.")
                if set(assignment[train]) != set(range(inner_n)):
                    raise ValueError("Inner fold coverage is invalid.")
                for block in {common_block_keys[row] for row in np.flatnonzero(train)}:
                    rows = [
                        row
                        for row in np.flatnonzero(train)
                        if common_block_keys[row] == block
                    ]
                    if len({int(assignment[row]) for row in rows}) != 1:
                        raise ValueError("Inner block leakage detected.")
                if families[target] == "categorical":
                    for inner_fold in range(inner_n):
                        validation = assignment == inner_fold
                        inner_train = train & ~validation
                        if (
                            np.unique(common_values[validation]).size != 2
                            or np.unique(common_values[inner_train]).size != 2
                        ):
                            raise ValueError("Categorical inner fold lacks both classes.")
            for region, representation, time in np.ndindex(3, 2, time_n):
                fit_index = (target, region, representation, time, outer_fold)
                audit_index = (target, outer_fold, region, representation, time)
                scores = arrays["candidate_inner_scores"][audit_index]
                statuses = arrays["candidate_inner_statuses"][audit_index]
                reasons = arrays["candidate_inner_reasons"][audit_index]
                choice = int(arrays["selected_candidate_indices"][audit_index])
                selected_text = arrays["selected_parameters_json"][audit_index]
                if not np.all(np.isin(statuses, ["valid", "invalid"])):
                    raise ValueError("Candidate status vocabulary is invalid.")
                valid_values = statuses == "valid"
                invalid_values = statuses == "invalid"
                if np.any(valid_values & ((~np.isfinite(scores)) | (reasons != ""))):
                    raise ValueError("Candidate valid status has nonfinite score or reason.")
                if np.any(invalid_values & (np.isfinite(scores) | (reasons == ""))):
                    raise ValueError("Candidate invalid status has finite score or missing reason.")
                candidate_valid = np.all(valid_values, axis=1)
                if unavailable_plan:
                    if not np.all(invalid_values) or not np.isnan(scores).all():
                        raise ValueError(
                            "Unavailable inner plan must use invalid NaN candidate audits."
                        )
                    if choice != -1 or selected_text != "":
                        raise ValueError("Unavailable inner plan must not select a candidate.")
                    if arrays["fit_status"][fit_index] != "unavailable":
                        raise ValueError(
                            "Unavailable inner plan requires an unavailable outer fit."
                        )
                    if arrays["effective_feature_counts"][fit_index] != 0:
                        raise ValueError(
                            "Unavailable inner plan effective feature count must be zero."
                        )
                    if target_unavailable and not np.all(
                        reasons == _TARGET_OUTER_UNAVAILABLE_CODE
                    ):
                        raise ValueError(
                            "Unavailable target candidate audits need the compact reason code."
                        )
                elif np.any(candidate_valid):
                    means = np.full(15, -np.inf, dtype=np.float64)
                    means[candidate_valid] = scores[candidate_valid].mean(axis=1)
                    expected = int(np.argmax(means))
                    if choice != expected or selected_text != arrays[
                        "candidate_parameter_json"
                    ][target, expected]:
                        raise ValueError("Selected candidate is not the deterministic best mean.")
                else:
                    if choice != -1 or selected_text != "":
                        raise ValueError(
                            "No valid candidate requires unavailable selection sentinels."
                        )
                    if arrays["fit_status"][fit_index] != "unavailable":
                        raise ValueError("No valid candidate requires unavailable outer fit.")
                    if arrays["effective_feature_counts"][fit_index] != 0:
                        raise ValueError(
                            "No valid candidate effective feature count must be zero."
                        )


def _atomic_npz(path: Path, members: Mapping[str, np.ndarray]) -> None:
    """Atomically publish safe primitive NPZ members with documented array shape and axis.

    Parameters
    ----------
    path : pathlib.Path
        New immutable destination in the result/checkpoint directory.
    members : mapping[str, numpy.ndarray]
        Fully validated primitive arrays with their declared shape, axis, and
        scientific units; scalar metadata is safe Unicode JSON.

    Returns
    -------
    None
        Writes one closed same-directory temporary then replaces only an absent final path.
    """
    if path.exists():
        raise FileExistsError(f"Immutable destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **members)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException as publication_error:
        if temporary is not None and temporary.exists():
            try:
                os.unlink(temporary)
            except BaseException as cleanup_error:
                raise cleanup_error from publication_error
        raise


def _write_json_if_matching(path: Path, value: Mapping[str, object]) -> None:
    """Create or compare one canonical immutable JSON sidecar.

    Parameters
    ----------
    path : pathlib.Path
        Sidecar destination in the prepared run directory.
    value : mapping[str, object]
        JSON-safe schema/config/manifest metadata without array axes or units changes.

    Returns
    -------
    None
        Creates an absent sidecar or raises ValueError if an existing byte payload differs.
    """
    payload = _canonical_json(value).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing immutable sidecar differs: {path.name}")
        return
    path.write_bytes(payload)


def _feature_copy_bytes(
    input_manifest: Mapping[str, object],
    feature_parameter_source: Path,
) -> bytes:
    """Read and verify the exact small feature-parameter source before publication.

    Parameters
    ----------
    input_manifest : mapping[str, object]
        Portable manifest containing the structured feature-parameter SHA-256.
    feature_parameter_source : pathlib.Path
        Original JSON feature parameter file to copy byte-for-byte.

    Returns
    -------
    bytes
        Verified parameter bytes; JSON values have no physical units.
    """
    try:
        identity = input_manifest["files"]["trial_feature_parameters"]
        expected_hash = identity["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Input manifest lacks feature-parameter SHA identity.") from error
    source = Path(feature_parameter_source)
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Feature parameter source SHA does not match input manifest.")
    return data


def save_target_checkpoint(
    checkpoint_path: Path,
    *,
    target_label: str,
    target_arrays: Mapping[str, np.ndarray],
    full_run_fingerprint: str,
) -> None:
    """Save one completed target checkpoint as an immutable safe NPZ.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        New immutable checkpoint destination.
    target_label : str
        Completed target identifier.
    target_arrays : mapping[str, numpy.ndarray]
        Target-local primitive arrays with documented native shape, axis, and units.
    full_run_fingerprint : str
        Exact complete scientific identity required for checkpoint reuse.

    Returns
    -------
    None
        Atomically writes non-pickle arrays and scalar checkpoint JSON metadata.
    """
    if not isinstance(target_label, str) or not target_label:
        raise ValueError("Checkpoint target label must be a nonempty string.")
    if not isinstance(full_run_fingerprint, str) or not full_run_fingerprint:
        raise ValueError("Checkpoint fingerprint must be a nonempty string.")
    if "checkpoint_meta" in target_arrays:
        raise ValueError("Checkpoint metadata is reserved by the writer.")
    members = dict(target_arrays)
    for name, array in members.items():
        if not isinstance(array, np.ndarray) or array.dtype == object:
            raise ValueError(f"checkpoint object array is unsafe or invalid: {name}")
    members["checkpoint_meta"] = np.array(
        _canonical_json(
            {"target_label": target_label, "full_run_fingerprint": full_run_fingerprint}
        )
    )
    _atomic_npz(Path(checkpoint_path), members)


def load_target_checkpoint(
    checkpoint_path: Path,
    *,
    expected_full_run_fingerprint: str,
) -> dict[str, object]:
    """Load a safe checkpoint only when its exact full identity matches.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        Existing checkpoint containing primitive arrays with their saved shape,
        axis, and target-native units.
    expected_full_run_fingerprint : str
        Complete scientific identity expected by the prepared result run.

    Returns
    -------
    dict[str, object]
        Target label, matching fingerprint, and copied non-pickle arrays.
    """
    if not isinstance(expected_full_run_fingerprint, str) or not expected_full_run_fingerprint:
        raise ValueError("Expected checkpoint fingerprint must be a nonempty string.")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        if "checkpoint_meta" not in archive.files:
            raise ValueError("Checkpoint metadata is missing.")
        metadata_array = archive["checkpoint_meta"]
        if metadata_array.dtype.kind != "U" or metadata_array.shape != ():
            raise ValueError("Checkpoint metadata must be a scalar Unicode JSON value.")
        try:
            metadata = json.loads(str(metadata_array.item()))
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError("Checkpoint metadata is invalid.") from error
        if not isinstance(metadata, Mapping):
            raise ValueError("Checkpoint metadata must be a JSON mapping.")
        target_label = metadata.get("target_label")
        fingerprint = metadata.get("full_run_fingerprint")
        if not isinstance(target_label, str) or not target_label:
            raise ValueError("Checkpoint metadata target label is invalid.")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Checkpoint metadata fingerprint is invalid.")
        if fingerprint != expected_full_run_fingerprint:
            raise ValueError("Checkpoint fingerprint does not match this run.")
        arrays: dict[str, np.ndarray] = {}
        for name in archive.files:
            if name == "checkpoint_meta":
                continue
            array = archive[name]
            if array.dtype == object:
                raise ValueError(f"Checkpoint array is unsafe or invalid: {name}")
            arrays[name] = array.copy()
    return {
        "target_label": target_label,
        "full_run_fingerprint": fingerprint,
        "target_arrays": arrays,
    }


def save_task_decoding_run(
    run_directory: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
    input_manifest: Mapping[str, object],
    scientific_config: Mapping[str, object],
    feature_parameter_source: Path,
    run_fingerprint: str,
) -> None:
    """Validate and atomically publish a portable immutable decoding result.

    Parameters
    ----------
    run_directory : pathlib.Path
        Destination directory for sidecars and one result NPZ.
    arrays : mapping[str, numpy.ndarray]
        Complete primitive schema; array shapes and score axes are ``(target, region,
        representation, metric, time, fold)`` and physical/model units are
        stated in ``meta``.
    meta : mapping[str, object]
        Complete JSON schema, axis, scientific-unit, and provenance mapping.
    input_manifest, scientific_config : mapping[str, object]
        Portable scientific identities written as immutable canonical sidecars.
    feature_parameter_source : pathlib.Path
        Structured feature-parameter source copied exactly after SHA verification.
    run_fingerprint : str
        Complete scientific identity that must already equal metadata provenance.

    Returns
    -------
    None
        Prevalidates all arrays before sidecar/result publication and writes a safe NPZ.
    """
    if not isinstance(meta.get("provenance"), Mapping) or meta["provenance"].get(
        "run_fingerprint"
    ) != run_fingerprint:
        raise ValueError("Run fingerprint disagrees with metadata provenance.")
    _validate_saved_arrays(arrays, meta, scientific_config)
    source_bytes = _feature_copy_bytes(input_manifest, Path(feature_parameter_source))
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json_if_matching(directory / _CONFIG_FILE, scientific_config)
    _write_json_if_matching(directory / _MANIFEST_FILE, input_manifest)
    feature_copy = directory / _FEATURE_COPY_FILE
    if feature_copy.exists():
        if feature_copy.read_bytes() != source_bytes:
            raise ValueError("Existing feature-parameter copy differs.")
    else:
        feature_copy.write_bytes(source_bytes)
    members = dict(arrays)
    members["meta"] = np.array(_canonical_json(meta))
    _atomic_npz(directory / _RESULT_FILE, members)


def load_task_decoding_run(
    run_directory: Path,
    *,
    expected_run_fingerprint: str | None = None,
) -> dict[str, object]:
    """Load a complete portable run with safe NPZ, shape, axis, and unit validation.

    Parameters
    ----------
    run_directory : pathlib.Path
        Directory containing config/manifest/feature-copy sidecars and result NPZ.
    expected_run_fingerprint : str or None, default=None
        Optional exact complete identity gate; does not change stored array axes or units.

    Returns
    -------
    dict[str, object]
        Copied primitive arrays, JSON meta/config/manifest, and the verified run fingerprint.
    """
    directory = Path(run_directory)
    required = [
        directory / _CONFIG_FILE,
        directory / _MANIFEST_FILE,
        directory / _FEATURE_COPY_FILE,
        directory / _RESULT_FILE,
    ]
    if any(not path.is_file() for path in required):
        raise ValueError(
            "Required result config, manifest, feature, or support artifact is missing."
        )
    try:
        config = json.loads((directory / _CONFIG_FILE).read_text(encoding="utf-8"))
        manifest = json.loads((directory / _MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Saved config or manifest JSON is invalid.") from error
    _feature_copy_bytes(manifest, directory / _FEATURE_COPY_FILE)
    with np.load(directory / _RESULT_FILE, allow_pickle=False) as archive:
        if "meta" not in archive.files:
            raise ValueError("NPZ meta member is missing.")
        try:
            meta = json.loads(str(archive["meta"].item()))
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError("NPZ meta JSON is invalid.") from error
        arrays = {name: archive[name].copy() for name in archive.files if name != "meta"}
    _validate_saved_arrays(arrays, meta, config)
    fingerprint = meta["provenance"]["run_fingerprint"]
    if expected_run_fingerprint is not None and fingerprint != expected_run_fingerprint:
        raise ValueError("Saved run fingerprint does not match expected identity.")
    return {
        "arrays": arrays,
        "meta": meta,
        "input_manifest": manifest,
        "scientific_config": config,
        "run_fingerprint": fingerprint,
    }


def condition_labels(saved_run: Mapping[str, object]) -> tuple[str, ...]:
    """Return condition labels for either supported immutable result schema.

    Parameters
    ----------
    saved_run : mapping[str, object]
        Mapping returned by :func:`load_task_decoding_run`. Schema-1 pooled
        results contain no physical condition axis and are interpreted as the
        historical ``all`` condition.

    Returns
    -------
    tuple[str, ...]
        Nonempty condition identifiers in saved axis order.
    """
    meta = saved_run.get("meta")
    arrays = saved_run.get("arrays")
    if not isinstance(meta, Mapping) or not isinstance(arrays, Mapping):
        raise ValueError("Saved run metadata or arrays are invalid.")
    schema_version = meta.get("schema_version")
    if schema_version == RESULT_SCHEMA_VERSION:
        return ("all",)
    if schema_version != CONDITION_RESULT_SCHEMA_VERSION:
        raise ValueError("Result schema version is invalid.")
    labels = arrays.get("condition_labels")
    if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype("U32"):
        raise ValueError("Condition label array is invalid.")
    values = tuple(str(value) for value in labels.tolist())
    if not values or len(set(values)) != len(values):
        raise ValueError("Condition labels must be nonempty and unique.")
    return values
