"""Versioned Stage 2 sidecars bound to one exact saved Stage 1 archive."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis.population.window_variance import (
    PARAMETERS,
    SessionWindowVariance,
    WindowVarianceSummary,
)

ANALYSIS_VERSION = "session-window-variance-v1"


def window_variance_configuration(stage1_path: Path | str) -> dict:
    """Return JSON-compatible source/parameter identity without loading score arrays.

    stage1_path must exist; identity includes resolved path, bytes and mtime_ns.
    Display options are excluded. Return a 16-character configuration digest.
    """
    path = Path(stage1_path).resolve(strict=True)
    stat = path.stat()
    config = {
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": 1,
        "source_stage1": {
            "path": str(path),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "parameters": dict(PARAMETERS),
        "axes": "window,PC; sorted contributions retain one-based session PC IDs",
        "units": "sample variance of session-standardized activity; fractions dimensionless",
        "random_seed": None,
        "generator": "population.window_variance.compute_session_window_variance",
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["configuration_id"] = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    return config


def save_window_variance(path: Path | str, result: SessionWindowVariance) -> Path:
    """Save named arrays plus a meta dictionary to a NEW NPZ; return its path.

    Preserve table rows/units/axes; never overwrite an existing summary. A partial
    archive cannot be discovered as a result until the final rename succeeds.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    meta = dict(result.meta)
    meta.update(
        session_id=result.session_id,
        time_origin_s=result.time_origin_s,
        regions=list(result.regions),
        unavailable=result.unavailable,
    )
    arrays = {"meta": meta, "choices": result.choices.to_records(index=False)}
    for role, region in result.regions.items():
        for field in fields(region):
            arrays[f"{role.lower()}_{field.name}"] = getattr(region, field.name)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.rename(path)
    return path


def load_window_variance(path: Path | str) -> SessionWindowVariance:
    """Load a trusted local Stage 2 NPZ without reading Stage 1 or raw data.

    Return the documented session/region records; reject unsupported versions.
    NumPy pickle is required for the metadata dictionary, as in Stage 1.
    """
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
        if (
            meta.get("analysis_version") != ANALYSIS_VERSION
            or meta.get("schema_version") != 1
        ):
            raise ValueError("Unsupported window-variance summary version.")
        regions = {}
        for role in meta["regions"]:
            values = {
                field.name: archive[f"{role.lower()}_{field.name}"]
                for field in fields(WindowVarianceSummary)
            }
            for name in ("region", "probe_id", "basis_complete", "full_pc_count"):
                values[name] = values[name].item()
            values["unit_ids"] = tuple(values["unit_ids"].tolist())
            regions[role] = WindowVarianceSummary(**values)
        return SessionWindowVariance(
            meta["session_id"],
            meta["time_origin_s"],
            pd.DataFrame.from_records(archive["choices"]),
            regions,
            meta["unavailable"],
            meta,
        )
