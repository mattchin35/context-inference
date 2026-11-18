"""Versioned presentation results and manifest-last run discovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis.population_decoding.results import write_json

ANALYSIS_VERSION = "pfc-hpc-presentation-v1"
SCHEMA_VERSION = 1


def save_result(path: Path, fields: dict, meta: dict) -> None:
    """Save named arrays/tables to a new trusted session-local NPZ Path.

    Arrays retain their native shapes and units. DataFrames are row tables with
    explicit identity columns and a positional index. Meta describes producer,
    axes, units, parameters and source identities. Returns None.
    """
    arrays = {}
    tables = {}
    for key, value in fields.items():
        if key == "meta":
            raise ValueError("meta is reserved for archive metadata")
        if isinstance(value, pd.DataFrame):
            tables[key] = {name: str(dtype) for name, dtype in value.dtypes.items()}
            arrays[key] = value.to_records(index=False) if len(value.columns) else np.empty(len(value))
        else:
            arrays[key] = value
    header = dict(meta, analysis_version=ANALYSIS_VERSION, schema_version=SCHEMA_VERSION, tables=tables)
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays, meta=header)


def load_result(path: Path, *, keys: tuple[str, ...] | None = None) -> tuple[dict, dict]:
    """Read a trusted project NPZ Path into named fields and metadata dict.

    Arrays preserve saved shape and units. Rebuild row tables with the stored
    column types and RangeIndex. Optional keys select the matching fields
    without decompressing other arrays. No session sources are opened.
    """
    with np.load(path, allow_pickle=True) as archive:
        header = archive["meta"].item()
        if header.get("schema_version") != SCHEMA_VERSION or header.get("analysis_version") != ANALYSIS_VERSION:
            raise ValueError(f"Unsupported presentation archive: {path}")
        fields = {}
        for key in archive.files:
            if key == "meta" or (keys is not None and key not in keys):
                continue
            value = archive[key]
            if key in header["tables"]:
                types = header["tables"][key]
                value = pd.DataFrame.from_records(value).astype(types) if types else pd.DataFrame(index=range(len(value)))
            elif value.ndim == 0:
                value = value.item()
            fields[key] = value
    return fields, header


def complete_run(directory: Path, *, kind: str, identity: str, session_ids: list[str],
                 archives: dict[str, str]) -> Path:
    """Publish a completed session/cohort manifest Path after all named NPZs exist.

    Directory holds a new run. Identity is its scientific inputs/settings
    digest, session_ids its ordered roster and archives its relative filenames.
    """
    directory = Path(directory)
    path = directory / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    if not all((directory / filename).is_file() for filename in archives.values()):
        raise ValueError("Cannot complete a run with missing result archives")
    write_json(path, dict(kind=kind, identity=identity, session_ids=session_ids, archives=archives,
                          schema_version=SCHEMA_VERSION, analysis_version=ANALYSIS_VERSION))
    return path


def discover_runs(root: Path, identity: str | None = None) -> list[Path]:
    """Return newest-first completed manifest Paths under an analysis run root.

    Optional identity selects compatible scientific inputs. Skip interrupted
    or incomplete runs. Inspect only the manifest and archive file presence.
    """
    matches = []
    for path in sorted(Path(root).glob("presentation_figures_*/manifest.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            if data["analysis_version"] != ANALYSIS_VERSION or data["schema_version"] != SCHEMA_VERSION:
                continue
            if identity is not None and data["identity"] != identity:
                continue
            if all((path.parent / name).is_file() for name in data["archives"].values()):
                matches.append(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return matches
