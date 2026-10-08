"""Immutable run identity and trusted persistence for inter-regional results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile

from .configuration import (
    InterregionalAnalysisConfig,
    ResolvedRegionalPopulation,
    RunOptions,
    configuration_to_dict,
    load_interregional_config,
)
from .records import InterregionalResults, validate_interregional_results


MANIFEST_SCHEMA_VERSION = "1"
ENTRYPOINT = "src.neural_analysis.interregional.run_session.run_single_session"
RUNTIME_VERSION_KEYS = (
    "python",
    "numpy",
    "pandas",
    "scipy",
    "pynapple",
    "scikit_learn",
    "statsmodels",
    "matplotlib",
)
_HASH_CHUNK_BYTES = 1024 * 1024
_TIMESTAMP_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{12}Z$")
_FINAL_DIRECTORY_PATTERN = re.compile(
    r"^interregional_regression_[0-9]{8}T[0-9]{12}Z_[0-9a-f]{12}$"
)
_REQUIRED_ARTIFACTS = (
    "config.json",
    "input_manifest.json",
    "result.pkl",
    "run.log",
    "summary.md",
    "run_session.py",
    "run_batch.py",
)


def _canonical_json(value: object) -> str:
    """Encode a JSON-compatible value deterministically without whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _population_to_dict(population: ResolvedRegionalPopulation) -> dict[str, object]:
    """Return all ordered resolved-population fields as JSON-compatible values."""
    return {
        "role": population.role,
        "probe_id": population.probe_id,
        "channel_source": population.channel_source,
        "selected_channels": list(population.selected_channels),
        "require_inside_brain": population.require_inside_brain,
        "channel_quality_labels": list(population.channel_quality_labels),
        "unit_quality_column": population.unit_quality_column,
        "unit_quality_labels": list(population.unit_quality_labels),
        "cluster_ids": list(population.cluster_ids),
        "unit_ids": list(population.unit_ids),
    }


def _validate_runtime_versions(runtime_versions: Mapping[str, str]) -> dict[str, str]:
    """Return the exact string-valued computation runtime-version mapping."""
    if set(runtime_versions) != set(RUNTIME_VERSION_KEYS):
        raise ValueError("runtime_versions must contain the exact computation key set.")
    normalized = {key: str(runtime_versions[key]) for key in RUNTIME_VERSION_KEYS}
    if any(not value for value in normalized.values()):
        raise ValueError("runtime_versions values must be nonempty strings.")
    return normalized


def _validate_file_entries(
    files: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate and sort expanded consumed-file identities by role then path."""
    required = {"logical_role", "resolved_path", "size_bytes", "sha256"}
    normalized: list[dict[str, object]] = []
    for entry in files:
        if set(entry) != required:
            raise ValueError("Each file identity must have the exact manifest fields.")
        role = entry["logical_role"]
        path = entry["resolved_path"]
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if not isinstance(role, str) or not role:
            raise ValueError("logical_role must be a nonempty string.")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("resolved_path must be an absolute path string.")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("size_bytes must be a nonnegative integer.")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("sha256 must be a lowercase 64-character digest.")
        normalized.append(
            {
                "logical_role": role,
                "resolved_path": str(Path(path).resolve()),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    normalized.sort(key=lambda value: (value["logical_role"], value["resolved_path"]))
    if len({value["logical_role"] for value in normalized}) != len(normalized):
        raise ValueError("Consumed-file logical roles must be unique.")
    return tuple(normalized)


def hash_input_files(files: Mapping[str, Path | str]) -> tuple[dict[str, object], ...]:
    """Stream SHA-256 identities for every declared consumed file.

    Parameters
    ----------
    files : mapping[str, pathlib.Path or str]
        Unique logical roles mapped to existing regular files. Paths are
        filesystem coordinates and file sizes are returned in bytes.

    Returns
    -------
    tuple[dict[str, object], ...]
        Entries sorted by logical role then normalized absolute path, each with
        ``logical_role``, ``resolved_path``, ``size_bytes``, and ``sha256``.
    """
    entries: list[dict[str, object]] = []
    for logical_role, raw_path in files.items():
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Consumed input is not a regular file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        entries.append(
            {
                "logical_role": str(logical_role),
                "resolved_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return _validate_file_entries(entries)


def _fingerprint_configuration(config: InterregionalAnalysisConfig) -> dict[str, object]:
    """Return scientific configuration with location/execution fields removed."""
    canonical = configuration_to_dict(config, RunOptions())
    canonical.pop("session_metadata_path")
    canonical.pop("run")
    return canonical


def run_fingerprint(
    config: InterregionalAnalysisConfig,
    *,
    session_id: str,
    resolved_populations: Sequence[ResolvedRegionalPopulation],
    files: Sequence[Mapping[str, object]],
    git_head: str,
    runtime_versions: Mapping[str, str],
) -> str:
    """Hash canonical scientific, input-content, code, and runtime identity.

    Absolute file and metadata paths are intentionally excluded. File sizes are
    bytes; all other values are identifiers or dimensionless settings.
    """
    populations = tuple(resolved_populations)
    if len(populations) != 2 or {value.role for value in populations} != {"PFC", "HPC"}:
        raise ValueError("resolved_populations must contain exactly PFC and HPC.")
    entries = _validate_file_entries(files)
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a nonempty string.")
    if not isinstance(git_head, str) or re.fullmatch(r"[0-9a-f]{40,64}", git_head) is None:
        raise ValueError("git_head must be a lowercase hexadecimal commit identity.")
    content_files = [
        {
            "logical_role": entry["logical_role"],
            "size_bytes": entry["size_bytes"],
            "sha256": entry["sha256"],
        }
        for entry in entries
    ]
    payload = {
        "configuration": _fingerprint_configuration(config),
        "session_id": session_id,
        "resolved_populations": [
            _population_to_dict(population)
            for population in sorted(populations, key=lambda value: value.role)
        ],
        "files": content_files,
        "git_head": git_head,
        "runtime_versions": _validate_runtime_versions(runtime_versions),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_input_manifest(
    *,
    run_fingerprint: str,
    session_id: str,
    git_head: str,
    runtime_versions: Mapping[str, str],
    resolved_populations: Sequence[ResolvedRegionalPopulation],
    files: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the exact expanded JSON manifest for one immutable run."""
    if re.fullmatch(r"[0-9a-f]{64}", run_fingerprint) is None:
        raise ValueError("run_fingerprint must be a lowercase SHA-256 digest.")
    populations = tuple(resolved_populations)
    if len(populations) != 2 or {value.role for value in populations} != {"PFC", "HPC"}:
        raise ValueError("resolved_populations must contain exactly PFC and HPC.")
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "session_id": session_id,
        "git_head": git_head,
        "entrypoint": ENTRYPOINT,
        "runtime_versions": _validate_runtime_versions(runtime_versions),
        "resolved_populations": [
            _population_to_dict(population)
            for population in sorted(populations, key=lambda value: value.role)
        ],
        "files": list(_validate_file_entries(files)),
    }


def _utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp with microseconds."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def create_working_run_directory(
    output_root: Path | str,
    *,
    repository_root: Path | str,
    timestamp: str | None = None,
) -> Path:
    """Create one exclusive hidden incomplete directory outside the repository.

    Parameters are filesystem paths. The returned directory is created beneath
    ``output_root`` and has no scientific units.
    """
    root = Path(output_root).resolve()
    repository = Path(repository_root).resolve(strict=True)
    if root == repository or root.is_relative_to(repository):
        raise ValueError("Run output_root must be outside the source repository.")
    value = _utc_timestamp() if timestamp is None else timestamp
    if _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError("timestamp must use YYYYMMDDTHHMMSSffffffZ UTC form.")
    root.mkdir(parents=True, exist_ok=True)
    working = root / f".interregional_regression_{value}.incomplete"
    working.mkdir(exist_ok=False)
    return working


def discover_finalized_runs(output_root: Path | str) -> tuple[Path, ...]:
    """Return structurally finalized run directories in stable name order."""
    root = Path(output_root)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir() and _FINAL_DIRECTORY_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: path.name,
        )
    )


def _validate_required_artifacts(working_path: Path) -> None:
    """Require every final run artifact and the figure directory."""
    for name in _REQUIRED_ARTIFACTS:
        if not (working_path / name).is_file():
            raise ValueError(f"Missing required artifact: {name}")
    if not (working_path / "figures").is_dir():
        raise ValueError("Missing required artifact: figures/")


def validate_complete_run_artifacts(
    run_path: Path | str,
    config: InterregionalAnalysisConfig,
    expected_fingerprint: str,
) -> None:
    """Reopen and validate every machine-readable artifact before final rename."""
    directory = Path(run_path).resolve(strict=True)
    _validate_required_artifacts(directory)
    saved_config, _ = load_interregional_config(directory / "config.json")
    if saved_config != config:
        raise ValueError("Saved configuration does not match the executed configuration.")
    try:
        manifest = json.loads(
            (directory / "input_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Saved input manifest is unreadable.") from error
    expected_keys = {
        "manifest_schema_version",
        "run_fingerprint",
        "session_id",
        "git_head",
        "entrypoint",
        "runtime_versions",
        "resolved_populations",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("Saved input manifest has the wrong top-level schema.")
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Saved input manifest version is unsupported.")
    if manifest["run_fingerprint"] != expected_fingerprint:
        raise ValueError("Saved input manifest fingerprint does not match the run.")
    load_interregional_result(
        directory / "result.pkl", config, trusted_run_directory=directory
    )


def finalize_run_directory(working_path: Path | str, fingerprint: str) -> Path:
    """Validate and atomically rename a same-parent incomplete run directory."""
    working = Path(working_path).resolve(strict=True)
    suffix = ".incomplete"
    prefix = ".interregional_regression_"
    if not working.name.startswith(prefix) or not working.name.endswith(suffix):
        raise ValueError("working_path is not an interregional incomplete directory.")
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
    _validate_required_artifacts(working)
    timestamp = working.name[len(prefix) : -len(suffix)]
    final = working.parent / f"interregional_regression_{timestamp}_{fingerprint[:12]}"
    if final.exists():
        raise FileExistsError(f"Final run directory already exists: {final}")
    os.replace(working, final)
    return final


def save_interregional_result(
    result: InterregionalResults,
    path: Path | str,
    config: InterregionalAnalysisConfig,
) -> None:
    """Validate and atomically pickle one pure computational result.

    The destination is a local trusted project-generated file. Scientific
    tables retain their exact pandas dtypes and array axes/units.
    """
    validate_interregional_results(result, config)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_interregional_result(
    path: Path | str,
    config: InterregionalAnalysisConfig,
    *,
    trusted_run_directory: Path | str,
) -> InterregionalResults:
    """Load and validate a pickle only from one explicitly trusted run directory."""
    try:
        run_directory = Path(trusted_run_directory).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("trusted run directory does not exist.") from error
    result_path = Path(path).resolve(strict=True)
    if result_path.parent != run_directory:
        raise ValueError("Result path must be directly inside the trusted run directory.")
    with result_path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, InterregionalResults):
        raise ValueError("Saved object is not an InterregionalResults record.")
    validate_interregional_results(value, config)
    return value


def record_run_failure(
    working_path: Path | str,
    *,
    stage: str,
    error: BaseException,
) -> None:
    """Append a caught failure log and write its small JSON sidecar.

    ``stage`` is a run-stage identifier and the timestamp is UTC. The incomplete
    directory is deliberately retained and remains undiscoverable as final.
    """
    working = Path(working_path).resolve(strict=True)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    message = f"{timestamp} ERROR stage={stage} {type(error).__name__}: {error}\n"
    with (working / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(message)
    failure = {
        "timestamp": timestamp,
        "last_entered_stage": stage,
        "exception_class": type(error).__name__,
        "exception_message": str(error),
    }
    (working / "failure.json").write_text(
        _canonical_json(failure) + "\n", encoding="utf-8"
    )
