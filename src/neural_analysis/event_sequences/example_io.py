"""Named-array persistence for trusted local behavioral example results."""

from __future__ import annotations

from pathlib import Path
from pickle import UnpicklingError
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .examples import CONDITIONS, SequenceCondition


ANALYSIS_VERSION = "behavior-examples-v1"
SCHEMA_VERSION = 1


def save_examples(output_path: Path | str, packet: dict) -> Path:
    """Save new trusted-local NPZ and return its Path, refusing existing files.

    packet contains full normalized trials (N,C) DataFrame, matches mapping
    32 unitless conditions to integer (E,S) source positions, and meta dict.
    choice_time remains UTC Unix seconds; action/context use 0=right, 1=left.
    Named arrays preserve source identities and all rows. Local object columns
    and meta use NumPy pickling, so archives must come from a trusted source.
    """
    _validate_packet(packet)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"trials": packet["trials"].to_records(index=False), "meta": packet["meta"]}
    arrays.update({_array_name(key): values for key, values in packet["matches"].items()})
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def load_examples(input_path: Path | str) -> dict:
    """Load trusted-local NPZ as {trials, matches, meta} without source access.

    input_path is an existing NPZ Path/string. Return full normalized (N,C)
    DataFrame, integer (E,S) matches and metadata with saved units/identities.
    No CSV, neural source or selector is accessed. Malformed archives raise
    ValueError; ordinary filesystem errors retain their original exception.
    """
    try:
        with np.load(input_path, allow_pickle=True) as archive:
            packet = {
                "trials": pd.DataFrame.from_records(archive["trials"]),
                "matches": {key: archive[_array_name(key)] for key in CONDITIONS},
                "meta": archive["meta"].item(),
            }
        _validate_packet(packet)
        return packet
    except (KeyError, TypeError, IndexError, AttributeError, EOFError, BadZipFile, UnpicklingError) as error:
        raise ValueError(f"Invalid behavior examples archive: {error}") from error


def _array_name(condition: SequenceCondition) -> str:
    """Map a unitless (family, side, repeats-or-None) tuple to a named NPZ array key."""
    family, side, repeats = condition
    return f"match__{family}__{side}__{repeats}"


def _validate_packet(packet: dict) -> None:
    """Validate normalized (N,C) trials, integer (E,S) positions and scalar metadata; return None.

    Validation checks identities/array structure without recomputing scientific
    matches. UTC times and source action/context units remain unchanged.
    """
    meta = packet["meta"]
    if meta.get("schema_version") != SCHEMA_VERSION or not str(meta.get("analysis_version", "")).startswith("behavior-examples-v"):
        raise ValueError("Unsupported behavior examples schema or analysis version.")
    if meta.get("parameters", {}).get("probe_return_outcome") not in ("rewarded", "rewarded_or_omission"):
        raise ValueError("Invalid saved probe_return_outcome.")
    trials = packet["trials"]
    required = {"row_position", "trial_id", "index_label", "action", "correct", "reward", "choice_time", "state_int", "cur_block", "experimenter_reward_given"}
    if not required.issubset(trials.columns) or not np.array_equal(trials.row_position, np.arange(len(trials))):
        raise ValueError("Invalid full trial table or original row positions.")
    if set(packet["matches"]) != set(CONDITIONS):
        raise ValueError("Saved conditions differ from the behavioral example schema.")
    for condition, positions in packet["matches"].items():
        family, _, repeats = condition
        length = 3 if repeats is None else repeats + {"probe": 2, "omission_probe": 3, "ideal_transition": 2, "indecisive_transition": 4}[family]
        if (positions.ndim != 2 or positions.shape[1] != length or not np.issubdtype(positions.dtype, np.integer)
                or np.any(positions < 0) or np.any(positions >= len(trials)) or np.any(np.diff(positions, axis=1) != 1)):
            raise ValueError(f"Invalid saved source positions for {condition}.")
