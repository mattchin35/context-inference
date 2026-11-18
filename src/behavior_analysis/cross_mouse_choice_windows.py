"""Prepare cross-mouse Ideal curves from explicitly selected saved collections.

No trials are fitted or scored here. Session medians remain unitless fractions;
the overall-performance roster defines the chronological training positions.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from .choice_window_analysis import validate_window_size


IDEAL_AGENT = "observer_mouse_agreement"
SCORING_RULES = {
    "initial_tie_choice": 0, "exclude_no_choice": True,
    "exclude_experimenter_reward": True, "discard_incomplete": True,
    "cross_context_transitions": True,
    "missing_predictions": "exclude per agent and report comparison counts",
}
VALUE_COLUMNS = (
    "mouse", "session_id", "date", "training_day", "window_size", "metric_value",
    "n_windows", "n_scored_windows", "n_comparisons", "status",
)


def load_cross_mouse_choice_window_data(
    data_root: Path,
    collection_runs: Mapping[str, Path] | None = None,
    window_sizes: Sequence[int] = (10, 20),
    *,
    mice: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load a strictly covered cohort without running session analysis.

    Parameters
    ----------
    data_root : pathlib.Path
        Mouse folders contain cross_session_analysis/{mouse}_overall_performance.csv.
        These rosters require unique full session_id and ISO date columns.
    collection_runs : Mapping[str, pathlib.Path] or None, default=None
        One explicit saved choice-window collection directory per mouse, with
        metadata.json and {mouse}_choice_window_summary.csv. Trusted source NPZ
        metadata distinguishes missing results from legitimate zero-window data.
        Supply either this mapping or mice, never both.
    window_sizes : Sequence[int], default=(10, 20)
        Distinct positive valid-choice window sizes to select, never recompute.
    mice : Sequence[str] or None, keyword-only
        Distinct mouse IDs for automatic selection. For each mouse, try saved
        timestamped collections newest first, retaining the first fully valid
        roster-matching run. File modification times do not affect ordering.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Table of shape (n_expected_sessions * n_sizes, len(VALUE_COLUMNS)) and
        provenance containing selection_mode, selected paths, skipped candidate
        paths/reasons, source hashes, numerical version
        and session-specific feature parameters. Each metric_value is the saved
        Ideal agreement_median (fraction), or NaN with an explicit status.
        Counts are windows/choices; training_day is a one-based chronological
        session index, sorting date then full ID before retaining missing values.
        Missing/malformed inputs or incompatible cohorts raise ValueError with
        owner/session details. No output files or input mutations occur.
    """
    sizes = [validate_window_size(size) for size in window_sizes]
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("window_sizes must contain distinct positive integers.")
    if (collection_runs is None) == (mice is None):
        raise ValueError("Supply exactly one of collection_runs or mice.")
    if mice is not None:
        if isinstance(mice, str):
            raise ValueError("mice must be a nonempty sequence of distinct mouse IDs.")
        owners = list(mice)
    else:
        owners = list(collection_runs)
    if (not owners or any(not isinstance(mouse, str) or not mouse.strip() for mouse in owners)
            or len(set(owners)) != len(owners)):
        raise ValueError("Provide nonempty, distinct mouse IDs.")
    rows, records, errors = [], {}, []
    for mouse in owners:
        try:
            directory = None if collection_runs is None else Path(collection_runs[mouse])
            mouse_rows, record = _select_mouse_collection(Path(data_root), mouse, directory, sizes)
            rows.extend(mouse_rows)
            records[str(mouse)] = record
        except (ValueError, OSError, KeyError) as error:
            errors.append(f"{mouse}: {error}")
    if errors:
        raise ValueError("Invalid cross-mouse choice-window inputs:\n- " + "\n- ".join(errors))
    versions = {record["analysis_version"] for record in records.values()}
    digests = {record["analysis_source_digest"] for record in records.values()}
    if len(versions) != 1 or len(digests) != 1:
        selected = "; ".join(
            f"{mouse}: {record['collection_run']} (version={record['analysis_version']}, "
            f"source_digest={record['analysis_source_digest']})" for mouse, record in records.items())
        raise ValueError("Selected mice have incompatible numerical analysis versions/source digests. " + selected)
    table = pd.DataFrame(rows, columns=VALUE_COLUMNS)
    for size in sizes:
        if not table.loc[table.window_size.eq(size), "metric_value"].notna().any():
            raise ValueError(f"No usable Ideal session medians for {size}-choice windows.")
    provenance = {
        "selection_mode": "automatic" if mice is not None else "explicit",
        "skipped_candidates": {mouse: record["skipped_candidates"] for mouse, record in records.items()},
        "analysis_version": next(iter(versions)), "analysis_source_digest": next(iter(digests)),
        "collection_runs": {mouse: record["collection_run"] for mouse, record in records.items()},
        "feature_parameters": {mouse: record["feature_parameters"] for mouse, record in records.items()},
        "source_sha256": {path: digest for record in records.values() for path, digest in record["source_sha256"].items()},
        "roster_paths": {mouse: record["roster_path"] for mouse, record in records.items()},
    }
    return table, provenance


def summarize_cross_mouse_choice_window_data(table: pd.DataFrame) -> pd.DataFrame:
    """Return equal-mouse means and coverage from prepared session medians.

    table has shape (n_session_sizes,C) with mouse, training_day, window_size
    and metric_value (unitless fraction/NaN). Each mouse/day/size must be unique.
    Return one row per size/day with agreement_mean (fraction) and n_mice
    (finite contributors). Empty groups retain NaN means and zero coverage.
    No input data is changed and no window/trial counts weight these means.
    """
    if table.duplicated(["mouse", "training_day", "window_size"]).any():
        raise ValueError("duplicate mouse/training-day/window-size rows")
    return table.groupby(["window_size", "training_day"], sort=True).agg(
        agreement_mean=("metric_value", "mean"), n_mice=("metric_value", "count"),
    ).reset_index()


# Private loading/validation helpers; none write files or perform inference.
def _select_mouse_collection(
    root: Path, mouse: str, directory: Path | None, sizes: list[int],
) -> tuple[list[dict], dict]:
    """Select and load one mouse's saved collection once; return session rows/provenance.

    root/mouse locate saved data; sizes counts valid choices. An explicit
    directory is authoritative. None tries timestamped runs newest first,
    returning the first fully validated result plus skipped paths/reasons.
    Cohort compatibility is checked later, without retrying older versions.
    No files, units or numerical arrays change.
    """
    if directory is not None:
        rows, record = _load_mouse_collection(root, mouse, directory, sizes)
        return rows, dict(record, skipped_candidates=[])
    parent = root / mouse / "cross_session_analysis/choice_window_agreement"
    candidates = []
    if parent.is_dir():
        for path in parent.iterdir():
            if not path.is_dir():
                continue
            try:
                timestamp = datetime.strptime(path.name, "%Y-%m-%d_%H-%M-%S_%f")
            except ValueError:
                continue
            candidates.append((timestamp, path))
    if not candidates:
        raise ValueError(f"No timestamped choice-window collection directories found under {parent}")
    skipped = []
    for _, candidate in sorted(candidates, reverse=True):
        try:
            rows, record = _load_mouse_collection(root, mouse, candidate, sizes)
        except (ValueError, OSError, KeyError) as error:
            skipped.append({"path": str(candidate), "reason": str(error)})
        else:
            return rows, dict(record, skipped_candidates=skipped)
    reasons = "\n".join(f"{item['path']}: {item['reason']}" for item in skipped)
    raise ValueError(f"No complete valid collection for requested window sizes {sizes}:\n{reasons}")


def _load_mouse_collection(root: Path, mouse: str, directory: Path, sizes: list[int]) -> tuple[list[dict], dict]:
    """Read one explicit run and roster; return unitless per-session rows and provenance.

    Paths identify trusted project files; sizes counts choices. All sessions in
    the roster must occur exactly once in the collection manifest. Source NPZ
    arrays are not loaded: only their metadata is needed for structural checks.
    """
    roster_path = root / mouse / "cross_session_analysis" / f"{mouse}_overall_performance.csv"
    summary_path = directory / f"{mouse}_choice_window_summary.csv"
    manifest_path = directory / "metadata.json"
    missing = [str(path) for path in (roster_path, summary_path, manifest_path) if not path.is_file()]
    if missing:
        raise ValueError("Missing files: " + ", ".join(missing))
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"Collection metadata must be a JSON object: {manifest_path}")
    if manifest.get("mouse") != mouse or manifest.get("errors"):
        raise ValueError(f"Invalid or failed collection metadata: {manifest_path}")
    if not set(sizes).issubset(manifest["window_sizes"]):
        raise ValueError(f"Collection does not contain requested window sizes {sizes}: {directory}")
    roster = pd.read_csv(roster_path, dtype={"session_id": str, "date": str})
    if not {"session_id", "date"}.issubset(roster.columns):
        raise ValueError(f"Roster requires session_id and date; rebuild the identified roster: {roster_path}")
    if roster.empty or roster[["session_id", "date"]].isna().any().any():
        raise ValueError(f"Roster has missing session identities/dates: {roster_path}")
    if roster.session_id.duplicated().any():
        raise ValueError(f"duplicate session_id in roster: {roster_path}")
    roster["date"] = pd.to_datetime(roster["date"], errors="raise").dt.strftime("%Y-%m-%d")
    roster = roster.sort_values(["date", "session_id"]).reset_index(drop=True)
    roster["training_day"] = np.arange(1, len(roster) + 1)
    identities, paths = manifest["session_ids"], manifest["sources"]
    if len(identities) != len(paths) or len(set(identities)) != len(identities):
        raise ValueError("Collection has duplicate sessions or misaligned source paths")
    if set(identities) != set(roster.session_id):
        absent = sorted(set(roster.session_id) - set(identities))
        extra = sorted(set(identities) - set(roster.session_id))
        raise ValueError(f"Session roster mismatch; missing={absent}, extra={extra}")
    summary = pd.read_csv(summary_path)
    required = {"session_id", "date", "window_size", "agent", "agreement_median",
                "n_windows", "n_scored_windows", "n_comparisons", "n_agreements"}
    if required - set(summary):
        raise ValueError(f"Missing summary columns: {sorted(required - set(summary))}")
    for column in ("window_size", "agreement_median", "n_windows", "n_scored_windows", "n_comparisons", "n_agreements"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    if summary.duplicated(["session_id", "window_size", "agent"]).any():
        raise ValueError(f"duplicate session/size/agent summary rows: {summary_path}")
    if set(summary.session_id) - set(identities):
        raise ValueError(f"Summary contains sessions outside the manifest: {summary_path}")
    source_paths = dict(zip(identities, paths))
    rows, errors, source_digests, parameters = [], [], set(), {}
    hashes = {str(path): _file_digest(path) for path in (roster_path, summary_path, manifest_path)}
    for session in roster.itertuples():
        path = Path(source_paths[session.session_id])
        try:
            with np.load(path, allow_pickle=True) as saved:
                metadata = saved["meta"].item()
            _validate_source_metadata(metadata, session.session_id, manifest["analysis_version"], sizes)
            source_digests.add(metadata["analysis_source_digest"])
            parameters[session.session_id] = metadata["parameters"].get("feature_parameters")
            for size in sizes:
                rows.append(_session_value(summary, metadata, mouse, session, size))
            hashes[str(path)] = _file_digest(path)
        except (OSError, ValueError, KeyError, BadZipFile, EOFError) as error:
            errors.append(f"{session.session_id}: {path}: {error}")
    if errors:
        raise ValueError("\n".join(errors))
    if len(source_digests) != 1:
        raise ValueError("Collection mixes numerical source digests; choose a compatible run")
    return rows, {"collection_run": str(directory), "roster_path": str(roster_path),
                  "analysis_version": manifest["analysis_version"],
                  "analysis_source_digest": next(iter(source_digests)),
                  "feature_parameters": parameters, "source_sha256": hashes}


def _validate_source_metadata(metadata: dict, identity: str, version: str, sizes: list[int]) -> None:
    """Validate one saved NPZ's identity/version/scoring rules; return None.

    Metadata is a trusted dictionary. identity/version are strings; sizes are
    valid-choice counts. Session-specific feature parameters are permitted.
    """
    if metadata["session_id"] != identity:
        raise ValueError("Source session identity differs from the manifest")
    if metadata["analysis_version"] != version:
        raise ValueError("Source analysis version differs from the collection version")
    parameters = metadata["parameters"]
    if not set(sizes).issubset(parameters["window_sizes"]):
        raise ValueError(f"Source is missing requested window sizes {sizes}")
    if parameters["agent_value_columns"].get(IDEAL_AGENT) != "observer_value":
        raise ValueError("Source is missing observer_mouse_agreement / observer_value")
    for name, expected in SCORING_RULES.items():
        if parameters.get(name) != expected:
            raise ValueError(f"Incompatible window rule {name}: {parameters.get(name)}")
    if not metadata.get("analysis_source_digest"):
        raise ValueError("Source lacks a numerical analysis_source_digest")


def _session_value(summary: pd.DataFrame, metadata: dict, mouse: str, session: object, size: int) -> dict:
    """Return one saved Ideal median/count row, preserving a roster session's ordinal.

    summary is (n_session_size_agents,C), metadata describes one source NPZ,
    mouse is its owner ID, session is a roster namedtuple (session_id/date/
    training_day), and size counts choices. Return VALUE_COLUMNS fields with
    a fraction/NaN metric_value and count/status diagnostics; inputs unchanged.
    """
    expected = metadata["n_windows"][str(size)]
    selected = summary.loc[summary.session_id.eq(session.session_id)
                           & summary.window_size.eq(size) & summary.agent.eq(IDEAL_AGENT)]
    base = {"mouse": mouse, "session_id": session.session_id, "date": session.date,
            "training_day": session.training_day, "window_size": size}
    if expected == 0 and selected.empty:
        return dict(base, metric_value=np.nan, n_windows=0, n_scored_windows=0,
                    n_comparisons=0, status="no_complete_windows")
    if len(selected) != 1:
        raise ValueError(f"Missing {IDEAL_AGENT} summary for {size}-choice windows")
    row = selected.iloc[0]
    if str(row.date) != session.date:
        raise ValueError(f"Summary date differs from roster for {size}-choice windows")
    counts = [row[column] for column in ("n_windows", "n_scored_windows", "n_comparisons", "n_agreements")]
    if any(not np.isfinite(value) or value < 0 or value != int(value) for value in counts):
        raise ValueError(f"Invalid window/comparison counts for {size}-choice windows")
    total, scored, comparisons, agreements = map(int, counts)
    if (total != expected or not 0 <= scored <= total or not scored <= comparisons <= scored * size
            or not 0 <= agreements <= comparisons or bool(scored) != bool(comparisons)):
        raise ValueError(f"Inconsistent window/comparison counts for {size}-choice windows")
    value = float(row.agreement_median)
    if (scored and (not np.isfinite(value) or not 0 <= value <= 1)) or (not scored and not np.isnan(value)):
        raise ValueError(f"Invalid agreement median for {size}-choice windows")
    return dict(base, metric_value=value, n_windows=total, n_scored_windows=scored,
                n_comparisons=comparisons, status="ok" if scored else "no_usable_predictions")


def _file_digest(path: Path) -> str:
    """Return a SHA-256 hex string for one existing input file; no units or mutation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
