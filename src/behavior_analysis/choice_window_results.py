"""Save and reload versioned choice-window results independently of plotting."""

from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .choice_window_analysis import (
    ANALYSIS_VERSION,
    DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS,
    RESULT_COLUMNS,
    RESULT_DTYPES,
    compute_trial_agent_agreements,
    resolve_agent_value_columns,
    summarize_choice_window_agreement,
    valid_choice_positions,
    validate_window_size,
)


ANALYSIS_SOURCE_FILES = (
    "choice_window_analysis.py", "choice_window_results.py", "session_analysis.py",
    "ideal_observer.py", "project_utils.py",
)


def load_choice_window_results(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load a trusted project-generated NPZ containing named arrays and meta.

    Parameters
    ----------
    path : pathlib.Path
        Existing results NPZ produced by ``save_choice_window_results``. The
        ``meta`` dictionary uses NumPy object storage, requiring pickle loading.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Long-form table, shape (n_window_agent_rows, len(RESULT_COLUMNS)), and
        provenance metadata. Trial bounds are zero-based and half-open;
        agreement is unitless and counts are valid mouse choices.
    """
    with np.load(path, allow_pickle=True) as saved:
        table = pd.DataFrame({column: saved[column] for column in RESULT_COLUMNS})
        metadata = saved["meta"].item()
    return table.astype(RESULT_DTYPES), metadata


def save_choice_window_results(
    trial_df: pd.DataFrame,
    processed_data_path: Path,
    session_id: str,
    window_sizes: Sequence[int] = (10, 20),
    feature_parameters: dict | None = None,
    rerun: bool = False,
    agent_value_columns: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Compute or reuse one session's additive window analysis and save it.

    Parameters
    ----------
    trial_df : pandas.DataFrame, shape (n_trials, n_columns)
        Chronological single-session augmented trials. Choices use 0=right,
        1=left; values are signed and left-positive. Input is never modified.
    processed_data_path : pathlib.Path
        Source session's processed-data directory. Outputs live beneath its
        ``choice_window_agreement`` subdirectory, outside tracked source files.
    session_id : str
        Source session identifier saved in table and metadata.
    window_sizes : Sequence[int], default=(10, 20)
        Distinct positive valid-choice counts. Incomplete tails are discarded.
    feature_parameters : dict or None
        JSON-serializable provenance for the existing agent values. None means
        feature parameters were unavailable, not that defaults were regenerated.
    rerun : bool, default=False
        Reuse matching complete saved results unless explicitly rerun. Every
        computation writes a new dated directory; prior results are preserved.
    agent_value_columns : Mapping[str, str] or None
        Explicit agent-to-value-column mapping, or available default agents.

    Returns
    -------
    tuple[pandas.DataFrame, pathlib.Path]
        Long-form window table and saved NPZ path. The artifact also contains
        valid-choice original-row positions and ``meta``. Counts are choices;
        agreement is a unitless fraction. No plotting or model fitting occurs.
    """
    started = perf_counter()
    sizes = [validate_window_size(size) for size in window_sizes]
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("window_sizes must contain distinct positive integers.")
    positions = valid_choice_positions(trial_df)
    mapping = resolve_agent_value_columns(trial_df, agent_value_columns)
    parameters = {
        "window_sizes": sizes, "agent_value_columns": mapping,
        "initial_tie_choice": 0, "feature_parameters": feature_parameters,
        "exclude_no_choice": True, "exclude_experimenter_reward": True,
        "discard_incomplete": True, "cross_context_transitions": True,
        "missing_predictions": "exclude per agent and report comparison counts",
    }
    # Include column names/types as well as positional contents; index labels do
    # not define trial coordinates and deliberately do not affect cache identity.
    input_schema = [(str(column), str(dtype)) for column, dtype in trial_df.dtypes.items()]
    input_hash = hashlib.sha256()
    input_hash.update(json.dumps(input_schema).encode("utf-8"))
    input_hash.update(pd.util.hash_pandas_object(trial_df.astype(str), index=False).to_numpy().tobytes())
    context_transitions = []
    if "cur_block" in trial_df:
        contexts = trial_df["cur_block"].reset_index(drop=True)
        known = contexts.notna() & ~contexts.astype(str).isin(["None", ""])
        changed = contexts.astype(str).ne(contexts.astype(str).shift()) & known & known.shift(fill_value=False)
        context_transitions = np.flatnonzero(changed.to_numpy()).tolist()
    metadata = {
        "generator": "choice_window_results.save_choice_window_results",
        "analysis_version": ANALYSIS_VERSION, "session_id": str(session_id),
        "parameters": parameters, "input_digest": input_hash.hexdigest(),
        "input_shape": list(trial_df.shape), "random_seed": None,
        "analysis_source_digest": _analysis_source_digest(),
        "context_transition_trial_positions": context_transitions,
        "n_valid_choices": len(positions),
        "discarded_choices": {str(size): len(positions) % size for size in sizes},
        "n_windows": {str(size): len(positions) // size for size in sizes},
        "unavailable_default_agents": [
            agent for agent, column in DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS.items()
            if column not in trial_df
        ],
        "units": {"agreement": "fraction", "counts": "valid mouse choices",
                  "trial_coordinates": "recorded trial rows"},
        "axis_conventions": {
            "bounds": "zero-based half-open positional indices",
            "choice_order": "within-session valid mouse choices",
            "trial_order": "original chronological recorded rows",
            "action": "0=right, 1=left", "agent_value": "signed left-positive",
            "unknown_context": "empty IDs and n_context_transitions=-1",
        },
    }
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    cache_root = Path(processed_data_path) / "choice_window_agreement" / ANALYSIS_VERSION / signature
    previous = sorted(cache_root.glob("*/results.npz"))
    if previous and not rerun:
        table, _ = load_choice_window_results(previous[-1])
        return table, previous[-1]

    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_directory = cache_root / timestamp
    run_directory.mkdir(parents=True, exist_ok=False)
    metadata["created_at"] = timestamp
    metadata["code_revision"] = _code_revision()
    try:
        scores = compute_trial_agent_agreements(trial_df, mapping)
        table = pd.concat([
            summarize_choice_window_agreement(trial_df, scores, size, session_id)
            for size in sizes
        ], ignore_index=True).astype(RESULT_DTYPES)
    except Exception as error:
        # Keep a diagnostic record, but never turn a failed computation into a
        # reusable numerical artifact or suppress its original exception.
        metadata["runtime_seconds"] = perf_counter() - started
        metadata["errors"] = [f"{type(error).__name__}: {error}"]
        _write_run_record(run_directory, metadata)
        raise
    metadata["runtime_seconds"] = perf_counter() - started
    path = run_directory / "results.npz"
    arrays = {}
    for column in RESULT_COLUMNS:
        dtype = str if RESULT_DTYPES[column] == "str" else RESULT_DTYPES[column]
        arrays[column] = table[column].to_numpy(dtype=dtype)
    np.savez_compressed(path, **arrays, valid_choice_trial_positions=positions, meta=metadata)
    table.to_csv(run_directory / "window_agreement.csv", index=False, na_rep="None")
    (run_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_run_record(run_directory, metadata)
    return table, path


# Private provenance helpers; computation and plotting do not depend on these.
def _analysis_source_digest() -> str:
    """Return SHA-256 of project sources that determine numerical results.

    No inputs or units. Returns a hexadecimal str; dirty source edits invalidate
    caches even before an analysis-version bump or a Git commit is made.
    """
    digest = hashlib.sha256()
    for name in ANALYSIS_SOURCE_FILES:
        digest.update(name.encode("ascii"))
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _code_revision() -> str | None:
    """Return the repository HEAD string, or None when Git is unavailable.

    No inputs or units. Returns str or None; does not change repository state.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _write_run_record(run_directory: Path, metadata: dict) -> None:
    """Save source snapshots, a Markdown summary, and a computation log.

    Parameters
    ----------
    run_directory : pathlib.Path
        New dated session-local output directory.
    metadata : dict
        Provenance with session ID, parameters, runtime in seconds, and counts.

    Returns
    -------
    None
        Writes human-readable records and copies of relevant project sources.
    """
    source_root = Path(__file__).resolve().parents[1]
    sources = [source_root / "main.py"]
    sources.extend(Path(__file__).with_name(name) for name in ANALYSIS_SOURCE_FILES)
    for source in sources:
        shutil.copy2(source, run_directory / source.name)
    summary = (
        "# Choice-window mouse-agent agreement\n\n"
        f"Session: {metadata['session_id']}\n\n"
        "Goal: describe greedy agent agreement in consecutive valid-choice windows.\n\n"
        "Main and batch orchestration: main.py (source snapshot included).\n"
        "Computation: choice_window_analysis.py; storage: choice_window_results.py.\n\n"
        f"Window sizes: {metadata['parameters']['window_sizes']}. "
        f"Discarded tails: {metadata['discarded_choices']}.\n\n"
        "Windows cross contexts but never sessions. Missing predictions have separate "
        "agent denominators. No statistical tests or new model fitting were performed.\n"
    )
    (run_directory / "summary.md").write_text(summary, encoding="utf-8")
    log = {
        "session_id": metadata["session_id"], "parameters": metadata["parameters"],
        "runtime_seconds": metadata["runtime_seconds"], "n_windows": metadata["n_windows"],
        "warnings": [], "errors": metadata.get("errors", []),
    }
    if metadata["parameters"]["feature_parameters"] is None:
        log["warnings"].append("Existing agent feature parameters were unavailable.")
    (run_directory / "run.log").write_text(json.dumps(log, indent=2), encoding="utf-8")
