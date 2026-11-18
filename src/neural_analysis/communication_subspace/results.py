"""Trusted local NPZ persistence for communication-subspace results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from pickle import UnpicklingError
import tempfile
import warnings
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .computation import ComputationResult, FitRecord
from .configuration import ANALYSIS_VERSION
from .modeling import RidgeDecomposition, TrainingTransform
from .preparation import PopulationDraw, ValidationSplit


SCHEMA_VERSION = 1
_TABLE_NAMES = ("choices", "selected_trials", "fold_scores", "inner_scores", "draw_scores", "curve_scores")
_SPLIT_AXES = ("outer_train_indices", "outer_test_indices", "inner_train_indices",
               "inner_validation_indices", "outer_purged_indices", "inner_purged_indices")


@dataclass(frozen=True)
class CommunicationResult:
    """Saved behavior identity and complete nested-CV computation.

    choices contains all ordinary choices with UTC-second choice_time;
    selected_trials contains original source columns for the eligible W choices.
    choice_positions is integer (W,) into choices; bin_edges_s is (B+1,)
    relative seconds; unit_ids are ordered regional (U,) qualified strings.
    computation holds scores, splits, draws, and selected fit factors. meta
    contains JSON-compatible provenance and scientific configuration.
    """

    session_id: str
    time_origin_s: float
    choices: pd.DataFrame
    selected_trials: pd.DataFrame
    choice_positions: np.ndarray
    unit_ids: dict[str, tuple[str, ...]]
    bin_edges_s: np.ndarray
    probe_roles: dict[str, str]
    computation: ComputationResult
    meta: dict


@dataclass(frozen=True)
class ResultInfo:
    """Small discoverable result header without loading fit/count arrays.

    path is a local NPZ Path; configuration is the saved JSON-compatible
    scientific identity; completed_utc is an ISO UTC timestamp; run_id is a
    filesystem label. None of these fields has physical units.
    """

    path: Path
    configuration: dict
    completed_utc: str
    run_id: str


def save_result(output_path: Path | str, result: CommunicationResult) -> Path:
    """Write a new trusted named-array NPZ atomically, never overwriting.

    output_path is a .npz Path. result contains choice/bin/unit identity and
    dimensionless score tables plus counts/bin fit transforms and coefficients.
    Return the created Path. Tables use record arrays; every selected factor,
    draw and split uses separately named numeric arrays. The object-valued meta
    dictionary is trusted local project data, not a pickled whole result.
    """
    path = Path(output_path)
    if path.suffix != ".npz":
        raise ValueError("Result path must end in .npz.")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    _validate_result(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = _encode_result(result, path.stem)
    arrays["meta"] = np.asarray(meta, dtype=object)
    descriptor, temporary = tempfile.mkstemp(prefix=".communication_", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(output, **arrays)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def load_result(path: Path | str) -> CommunicationResult:
    """Restore and validate a trusted result NPZ without accessing raw sources.

    path names a local project-generated .npz. Return CommunicationResult with
    original choice/bin/unit axes, pandas table dtypes, draws, splits and compact
    selected fit factors. Timestamps remain UTC seconds; counts transform means
    and scales remain spikes/bin. Malformed schemas/axes/factors raise ValueError.
    """
    try:
        return _load_result_archive(Path(path))
    except (OSError, EOFError, BadZipFile, UnpicklingError, KeyError, IndexError,
            AttributeError, TypeError) as error:
        raise ValueError(f"Invalid or unreadable communication result archive: {error}") from error


def _load_result_archive(path: Path) -> CommunicationResult:
    """Restore one trusted NPZ Path; callers normalize archive format errors."""
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
        if (not isinstance(meta, dict) or meta.get("schema_version") != SCHEMA_VERSION
                or meta.get("analysis_version") != ANALYSIS_VERSION):
            raise ValueError("Unsupported communication result schema or analysis version.")
        tables = {name: _restore_table(archive[name], meta["table_dtypes"][name])
                  for name in _TABLE_NAMES}
        choice_positions = np.asarray(archive["choice_positions"], dtype=np.int64)
        edges = np.asarray(archive["bin_edges_s"], dtype=np.float64)
        draws = tuple(_load_draw(archive, index) for index in range(meta["draw_count"]))
        splits = tuple(_load_split(archive, index) for index in range(meta["split_count"]))
        fits = tuple(_load_fit(archive, index, descriptor)
                     for index, descriptor in enumerate(meta["fits"]))
        computation = ComputationResult(
            tables["fold_scores"], tables["inner_scores"], fits,
            tables["draw_scores"], tables["curve_scores"], draws, splits,
            meta["status"],
        )
        result = CommunicationResult(
            meta["session_id"], float(meta["time_origin_s"]), tables["choices"],
            tables["selected_trials"], choice_positions,
            {region: tuple(ids) for region, ids in meta["unit_ids"].items()},
            edges, dict(meta["probe_roles"]), computation,
            dict(meta["provenance"]),
        )
    _validate_result(result)
    return result


def discover_results(
    session_root: Path | str, configuration_id: str | None = None,
) -> tuple[ResultInfo, ...]:
    """List supported local result headers under one session root.

    session_root is a Path; optional configuration_id filters exact scientific
    identity strings. Return newest-first small ResultInfo records. No raw
    neural files or complete score/factor arrays are loaded. Corrupt/partial
    archives are skipped; full validity is checked by load_result on selection.
    """
    root = Path(session_root) / "communication_subspace" / "results"
    found = []
    for path in sorted(root.glob("*.npz"), reverse=True):
        try:
            with np.load(path, allow_pickle=True) as archive:
                meta = archive["meta"].item()
            if (not isinstance(meta, dict) or meta.get("schema_version") != SCHEMA_VERSION
                    or meta.get("analysis_version") != ANALYSIS_VERSION):
                continue
            configuration = meta["provenance"]["configuration"]
            if configuration_id is not None and configuration.get("configuration_id") != configuration_id:
                continue
            found.append(ResultInfo(path, configuration, meta["completed_utc"], meta["run_id"]))
        except (OSError, ValueError, KeyError, EOFError, AttributeError,
                BadZipFile, UnpicklingError, TypeError) as error:
            warnings.warn(f"Skipping corrupt or unreadable result {path}: {error}",
                          RuntimeWarning, stacklevel=2)
            continue
    return tuple(found)


# Serialization helpers keep factor arrays named and table schemas explicit.
def _encode_result(result: CommunicationResult, run_id: str) -> tuple[dict, dict]:
    """Return named NPZ arrays and object meta for one validated result record."""
    arrays = {
        "choice_positions": np.asarray(result.choice_positions, dtype=np.int64),
        "bin_edges_s": np.asarray(result.bin_edges_s, dtype=np.float64),
    }
    tables = {"choices": result.choices, "selected_trials": result.selected_trials,
              "fold_scores": result.computation.fold_scores,
              "inner_scores": result.computation.inner_scores,
              "draw_scores": result.computation.draw_scores,
              "curve_scores": result.computation.curve_scores}
    table_dtypes = {}
    for name, frame in tables.items():
        arrays[name] = frame.astype(object).to_records(index=False)
        table_dtypes[name] = {column: str(dtype) for column, dtype in frame.dtypes.items()}
    for index, draw in enumerate(result.computation.draws):
        for region in ("PFC", "HPC"):
            arrays[f"draw_{index}_{region}_source"] = draw.source_indices[region]
            arrays[f"draw_{index}_{region}_target"] = draw.target_indices[region]
    for index, split in enumerate(result.computation.splits):
        for axis in _SPLIT_AXES:
            arrays[f"split_{index}_{axis}"] = getattr(split, axis)
    descriptors = []
    for index, fit in enumerate(result.computation.fit_records):
        prefix = f"fit_{index}_"
        arrays[prefix + "source_indices"] = fit.source_indices
        arrays[prefix + "target_indices"] = fit.target_indices
        for side in ("source", "target"):
            transform = getattr(fit, f"{side}_transform")
            arrays[prefix + side + "_means"] = transform.means
            arrays[prefix + side + "_scales"] = transform.scales
            arrays[prefix + side + "_variable_mask"] = transform.variable_mask
        arrays[prefix + "ridge_coefficients"] = fit.decomposition.ridge_coefficients
        arrays[prefix + "right_vectors"] = fit.decomposition.right_vectors
        descriptors.append({"draw": fit.draw, "direction": fit.direction, "fold": fit.fold,
                            "alpha": fit.alpha, "source_unit_ids": list(fit.source_unit_ids),
                            "target_unit_ids": list(fit.target_unit_ids),
                            "max_rank": fit.decomposition.max_rank})
    meta = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "session_id": result.session_id,
        "time_origin_s": result.time_origin_s,
        "unit_ids": {region: list(ids) for region, ids in result.unit_ids.items()},
        "probe_roles": result.probe_roles,
        "status": result.computation.status,
        "draw_count": len(result.computation.draws),
        "split_count": len(result.computation.splits),
        "fits": descriptors,
        "table_dtypes": table_dtypes,
        "provenance": result.meta,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
    }
    return arrays, meta


def _restore_table(records: np.ndarray, dtypes: dict[str, str]) -> pd.DataFrame:
    """Restore a pandas table from record array and saved column dtypes."""
    frame = pd.DataFrame.from_records(records)
    if tuple(frame.columns) != tuple(dtypes):
        raise ValueError("Saved table columns differ from schema.")
    for column, dtype in dtypes.items():
        frame[column] = frame[column].astype(dtype)
    return frame


def _load_draw(archive, index: int) -> PopulationDraw:
    """Restore one integer-index draw from named arrays."""
    source = {region: np.asarray(archive[f"draw_{index}_{region}_source"], dtype=np.int64)
              for region in ("PFC", "HPC")}
    target = {region: np.asarray(archive[f"draw_{index}_{region}_target"], dtype=np.int64)
              for region in ("PFC", "HPC")}
    return PopulationDraw(source, target)


def _load_split(archive, index: int) -> ValidationSplit:
    """Restore one six-axis whole-choice split from named integer arrays."""
    axes = [np.asarray(archive[f"split_{index}_{axis}"], dtype=np.int64) for axis in _SPLIT_AXES]
    return ValidationSplit(*axes)


def _load_fit(archive, index: int, descriptor: dict) -> FitRecord:
    """Restore one selected compact fit from named numeric arrays and IDs."""
    prefix = f"fit_{index}_"
    transforms = []
    for side in ("source", "target"):
        transforms.append(TrainingTransform(
            np.asarray(archive[prefix + side + "_means"], dtype=np.float64),
            np.asarray(archive[prefix + side + "_scales"], dtype=np.float64),
            np.asarray(archive[prefix + side + "_variable_mask"], dtype=bool),
        ))
    decomposition = RidgeDecomposition(
        np.asarray(archive[prefix + "ridge_coefficients"], dtype=np.float64),
        np.asarray(archive[prefix + "right_vectors"], dtype=np.float64),
        int(descriptor["max_rank"]),
    )
    return FitRecord(
        int(descriptor["draw"]), str(descriptor["direction"]), int(descriptor["fold"]),
        float(descriptor["alpha"]),
        np.asarray(archive[prefix + "source_indices"], dtype=np.int64),
        np.asarray(archive[prefix + "target_indices"], dtype=np.int64),
        tuple(descriptor["source_unit_ids"]), tuple(descriptor["target_unit_ids"]),
        transforms[0], transforms[1], decomposition,
    )


def _validate_result(result: CommunicationResult) -> None:
    """Reject inconsistent saved choice/bin/unit axes and compact fit factors."""
    if not isinstance(result, CommunicationResult) or not result.session_id:
        raise ValueError("Expected a named communication result.")
    positions = np.asarray(result.choice_positions)
    edges = np.asarray(result.bin_edges_s)
    if (positions.ndim != 1 or not len(positions) or not np.issubdtype(positions.dtype, np.integer)
            or np.any(positions < 0) or np.any(positions >= len(result.choices))
            or len(np.unique(positions)) != len(positions)
            or len(result.selected_trials) != len(positions)):
        raise ValueError("Invalid choice positions or selected-trial axis.")
    if (edges.ndim != 1 or not np.isfinite(edges).all() or len(edges) < 2
            or not np.all(np.diff(edges) > 0)):
        raise ValueError("Invalid choice-relative bin edge axis.")
    if (set(result.probe_roles) != {"PFC", "HPC"} or set(result.unit_ids) != {"PFC", "HPC"}
            or any(not ids or len(set(ids)) != len(ids) for ids in result.unit_ids.values())):
        raise ValueError("Invalid regional unit identity axes.")
    if "configuration" not in result.meta or result.computation.status not in {"success", "partial", "unavailable"}:
        raise ValueError("Missing scientific configuration or computation status.")
    for fit in result.computation.fit_records:
        _validate_fit(fit, result.unit_ids)
    for split in result.computation.splits:
        for axis in _SPLIT_AXES:
            values = np.asarray(getattr(split, axis))
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= len(positions)):
                raise ValueError("Split choice axis is invalid.")


def _validate_fit(fit: FitRecord, unit_ids: dict[str, tuple[str, ...]]) -> None:
    """Check a selected factor's source/target masks, IDs, and matrix axes."""
    if fit.direction not in {"PFC->PFC", "PFC->HPC", "HPC->HPC", "HPC->PFC"}:
        raise ValueError("Unknown fit direction.")
    source_region, target_region = fit.direction.split("->")
    for side, region in (("source", source_region), ("target", target_region)):
        indices = np.asarray(getattr(fit, f"{side}_indices"))
        ids = getattr(fit, f"{side}_unit_ids")
        transform = getattr(fit, f"{side}_transform")
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= len(unit_ids[region]))
                or tuple(unit_ids[region][int(index)] for index in indices) != ids
                or any(np.asarray(getattr(transform, name)).shape != (len(indices),)
                       for name in ("means", "scales", "variable_mask"))
                or not np.isfinite(transform.means).all()
                or not np.isfinite(transform.scales).all()):
            raise ValueError("Fit unit identity or training transform axis is invalid.")
    n_source = int(np.count_nonzero(fit.source_transform.variable_mask))
    n_target = int(np.count_nonzero(fit.target_transform.variable_mask))
    decomposition = fit.decomposition
    if (decomposition.ridge_coefficients.shape != (n_source, n_target)
            or decomposition.right_vectors.ndim != 2
            or decomposition.right_vectors.shape[1] != n_target
            or not 1 <= decomposition.max_rank <= min(n_source, n_target)
            or decomposition.right_vectors.shape[0] < decomposition.max_rank
            or not np.isfinite(decomposition.ridge_coefficients).all()
            or not np.isfinite(decomposition.right_vectors).all()):
        raise ValueError("Fit coefficient/basis shape is invalid.")
