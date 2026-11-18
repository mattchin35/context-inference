"""Assemble Figures 7 and 8 from saved, equally weighted session summaries."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .results import load_result


def assemble_cohort(reports: list[dict]) -> dict[str, pd.DataFrame]:
    """Combine ordered session-run reports into named row tables for one mouse.

    Keep original fractions, counts, R-squared and session identifiers. The
    full named roster, including excluded sessions, defines Figure 8 support.
    Only compact summary fields are read from each trusted session archive.
    """
    roster = pd.DataFrame(reports)
    if roster.empty or roster.session_id.duplicated().any():
        raise ValueError("The cohort requires a nonempty roster of unique session IDs.")
    if "reason" not in roster:
        roster["reason"] = ""
    roster["reason"] = roster.reason.fillna("")
    collected = defaultdict(list)
    expected = None
    stage_keys = {
        "regional": ("scores", "confusion", "reward_discrimination", "recovery", "units",
                     "delivery_latency", "transfer_mapping", "classes"),
        "prediction": ("curves",), "replays": ("episodes",),
    }
    for report in reports:
        if not report.get("manifest"):
            continue
        path = Path(report["manifest"])
        manifest = json.loads(path.read_text())
        session_id = report["session_id"]
        if manifest["kind"] != "session" or manifest["session_ids"] != [session_id]:
            raise ValueError(f"Session manifest identity mismatch for {session_id}")
        for stage, keys in stage_keys.items():
            fields, header = load_result(path.parent / manifest["archives"][stage], keys=keys)
            settings = tuple(header[k] for k in ("subject_id", "seed", "parameters", "units"))
            if header["session_id"] != session_id or (expected is not None and settings != expected):
                raise ValueError(f"Incompatible {stage} results for {session_id}")
            expected = settings
            for key, table in fields.items():
                collected[key].append(table.assign(session_id=session_id))
    tables = {key: pd.concat(parts, ignore_index=True) for key, parts in collected.items()}
    tables["roster"] = roster
    tables["prediction_medians"] = _session_medians(
        tables.get("curves"), ["panel", "direction", "model", "count"], "mean_r2", len(roster), True)
    tables["regional_medians"] = _session_medians(
        tables.get("scores"), ["region", "width_ms", "target", "group", "regime", "period",
                               "n_pcs", "bin_index"], "balanced_accuracy", len(roster), False)
    tables["presentation_examples"] = _choose_examples(tables.get("episodes"), roster)
    return tables


def _session_medians(table: pd.DataFrame | None, keys: list[str], score: str,
                     roster_size: int, require_complete: bool) -> pd.DataFrame:
    """Summarize one dimensionless score per session and return pointwise counts."""
    if table is None or table.empty:
        return pd.DataFrame(columns=[*keys, score, "n_sessions", "roster_size"])
    if table.duplicated(["session_id", *keys]).any():
        raise ValueError(f"More than one {score} per session and curve point")
    grouped = table.groupby(keys, dropna=False)[score].agg(**{score: "median", "n_sessions": "count"}).reset_index()
    grouped["roster_size"] = roster_size
    if require_complete:
        grouped.loc[grouped.n_sessions != roster_size, score] = np.nan
    return grouped


def _choose_examples(episodes: pd.DataFrame | None, roster: pd.DataFrame) -> pd.DataFrame:
    """Return one session/manifest per available motif using the presentation order."""
    columns = ["motif", "session_id", "manifest"]
    if episodes is None or episodes.empty:
        return pd.DataFrame(columns=columns)
    dates = ("20260828", "20260803", "20260804", "20260810")
    priorities = {}
    for i, name in enumerate(roster.session_id):
        compact = str(name).replace("-", "").replace("_", "").lower()
        priorities[name] = next((j for j, day in enumerate(dates)
                                 if "ct026" in compact and day in compact), len(dates) + i)
    available = episodes[episodes.status == "ok"].copy()
    available["priority"] = available.session_id.map(priorities)
    selected = available.sort_values("priority", kind="stable").drop_duplicates("motif")
    return selected.merge(roster[["session_id", "manifest"]], on="session_id")[columns]
