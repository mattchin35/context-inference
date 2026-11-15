"""Compute a saved-weight selection, then render it without loading raw recordings."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.neural_analysis.population_decoding import results as decoder_results
from src.neural_analysis.population_decoding.preparation import PreparedPopulation
from . import results
from .plotting import build_unit_activity_data, export_unit_activity_png


def run_session(source_run: Path | str, region: str, *, command: str = "new",
                max_units_per_group: int = 10, scale: str = "zscore",
                time_range_s: tuple[float, float] | None = None, event_tracks: tuple[str, ...] = (),
                inputs: dict | None = None) -> dict:
    """Run one saved decoder Path/region; return a JSON-safe completion/availability report.

    command is new/dry-run. cap is positive units/group, scale zscore/hz, range
    elapsed seconds. Optional validated inputs share UI source loads. Dry-run
    writes nothing. New computes/saves/reloads the selection before rendering;
    identical selections are reused, figures always get fresh run directories.
    """
    started = perf_counter()
    if command not in ("new", "dry-run") or scale not in ("zscore", "hz"):
        raise ValueError("Expected new/dry-run command and zscore/hz scale")
    loaded = inputs if inputs is not None else results.load_decoder_inputs(source_run, region)
    if loaded["region"] != region or loaded["directory"] != Path(source_run).resolve():
        raise ValueError("Provided saved inputs belong to another run/region")
    if loaded["status"] != "ready":
        return {"status": loaded["status"], "reason": loaded["reason"], "region": region, "source_run": str(source_run)}
    selection = results.build_unit_selection(loaded, max_units_per_group)
    path = loaded["session_root"] / "processed/decoder_unit_activity" / selection["meta"]["selection_id"] / "selection.npz"
    if command == "dry-run":
        regional = loaded["prepared"].regions[region]
        return {"status": "dry-run", "region": region, "source_run": str(source_run),
                "activity_shape": list(regional.rates_hz.shape), "bin_width_ms": selection["meta"]["bin_width_ms"],
                "selected_units": len(selection["arrays"]["display_unit_indices"]), "selection_path": str(path)}
    reused = path.exists()
    if not reused:
        results.save_unit_selection(path, selection["arrays"], selection["meta"])
    selection = results.load_unit_selection(path)
    report = _render_selection(selection, path, loaded["prepared"], scale, time_range_s, event_tracks)
    report.update(selection_reused=reused, elapsed_s=perf_counter()-started)
    return report


def plot_selection(selection_path: Path | str, *, scale: str = "zscore",
                   time_range_s: tuple[float, float] | None = None, event_tracks: tuple[str, ...] = ()) -> dict:
    """Render existing selection NPZ Path from its exact preparation reference, no weight refit.

    scale/range are zscore/hz and elapsed seconds. Return new output-path/status
    mapping. Source archives must remain available; raw spike files need not.
    """
    path = Path(selection_path)
    selection = results.load_unit_selection(path)
    manifest = decoder_results.load_manifest(Path(selection["meta"]["session_root"]) / selection["meta"]["source_run"])
    preparation_path = Path(manifest["preparation_path"])
    if not preparation_path.is_absolute():
        preparation_path = Path(selection["meta"]["session_root"]) / selection["meta"]["source_run"] / preparation_path
    prepared = decoder_results.load_preparation(preparation_path)
    return _render_selection(selection, path, prepared, scale, time_range_s, event_tracks)


def activity_for_selection(selection: dict, prepared: PreparedPopulation, *, scale: str = "zscore") -> dict:
    """Assemble full-pool activity from validated selection and preparation, returning renderer data.

    Verify source identity, IDs(U,), row positions(W,) and native edges(B+1,).
    Selection indices(S,) preserve ordered identity. No viewport normalization.
    """
    arrays, meta = selection["arrays"], selection["meta"]
    regional = prepared.regions[meta["region"]]
    if (prepared.meta["preparation_id"] != meta["source_preparation_id"]
            or tuple(arrays["unit_ids"]) != regional.unit_ids
            or not np.array_equal(arrays["row_positions"], regional.row_positions)
            or not np.array_equal(arrays["bin_edges_s"], regional.bin_edges_s)):
        raise ValueError("Selection/preparation activity identity or axes disagree")
    choices = prepared.behavior.choices
    positions = pd.Index(choices.row_position).get_indexer(regional.row_positions)
    selected = tuple(arrays["unit_ids"][arrays["display_unit_indices"]])
    data = build_unit_activity_data(regional.rates_hz, regional.unit_ids, selected, choices,
                                    positions, regional.bin_edges_s, scale=scale)
    for name in ("mean_hz", "scale_hz", "constant_mask"):
        if not np.array_equal(data[name], arrays[name]):
            raise ValueError("Selection normalization disagrees with referenced native activity")
    return data


def main(argv: list[str] | None = None) -> int:
    """Parse CLI strings; output JSON status and integer exit code, with elapsed seconds for ranges."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("new", "dry-run", "plot"):
        sub = commands.add_parser(name)
        if name == "plot":
            sub.add_argument("--selection", type=Path, required=True)
        else:
            sub.add_argument("--source-run", type=Path, required=True)
            sub.add_argument("--region", choices=("PFC", "HPC"), required=True)
            sub.add_argument("--max-units-per-group", type=int, default=10)
        sub.add_argument("--scale", choices=("zscore", "hz"), default="zscore")
        sub.add_argument("--time-range", type=float, nargs=2, metavar=("START_S", "END_S"))
        sub.add_argument("--event-tracks", nargs="*", default=[])
    args = parser.parse_args(argv)
    options = {"scale": args.scale, "time_range_s": tuple(args.time_range) if args.time_range else None,
               "event_tracks": tuple(args.event_tracks)}
    try:
        if args.command == "plot":
            report = plot_selection(args.selection, **options)
        else:
            report = run_session(args.source_run, args.region, command=args.command,
                                 max_units_per_group=args.max_units_per_group, **options)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] in ("complete", "dry-run") else 1


# Output assembly deliberately loads the durable selection before generating figures.
def _render_selection(selection: dict, selection_path: Path, prepared: PreparedPopulation, scale: str,
                       time_range_s: tuple[float, float] | None, event_tracks: tuple[str, ...]) -> dict:
    """Create one immutable human-readable output run; inputs retain W,B,U Hz and elapsed seconds."""
    from .plotting import viewport_unit_activity_data
    started = perf_counter()
    meta = selection["meta"]
    timestamp = datetime.now(ZoneInfo("America/New_York"))
    directory = Path(meta["session_root"]) / "analysis_runs" / (
        f"decoder_unit_activity_{timestamp:%Y%m%d_%H%M%S_%f}_{meta['selection_id'][:12]}")
    directory.mkdir(parents=True, exist_ok=False)
    log = directory / "run.log"
    settings = {"source_run": meta["source_run"], "session_id": meta["session_id"], "region": meta["region"],
        "max_units_per_group": meta["max_units_per_group"], "scale": scale,
        "time_range_s": time_range_s, "event_tracks": list(event_tracks), "selection_path": str(selection_path),
        "created_at": timestamp.isoformat(), "selection_id": meta["selection_id"]}
    log.write_text(f"{timestamp.isoformat()} Starting saved-only rendering\n{json.dumps(settings)}\n", encoding="ascii")
    try:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            shutil.copyfile(path, directory / path.name)
        (directory / "settings.json").write_text(json.dumps(settings, indent=2), encoding="ascii")
        selection["assignments"].to_csv(directory / "unit_groups.csv", index=False)
        display = results.displayed_units(selection["assignments"])
        data = activity_for_selection(selection, prepared, scale=scale)
        if time_range_s is not None:
            data = viewport_unit_activity_data(data, time_range_s)
        output = directory / "activity.png"
        export_unit_activity_png(output, data, prepared.behavior, display, event_tracks, time_range_s)
        lines = ["# Decoder-weight unit activity", "", f"Session: {meta['session_id']}; region: {meta['region']}.",
            f"Source run: {meta['source_run']}", f"Selection: {selection_path}",
            "Goal: inspect actual choice-centered activity of units grouped by consistent decoder weights.",
            "Scripts: run_session.py; run_batch.py (snapshots included).", "",
            f"Displayed units: {len(display)}; {meta['bin_width_ms']} ms native bins; scale: {scale}.",
            "Groups describe coefficients; same-session activity is descriptive, not independent selectivity evidence.",
            "Four-class outcome weights can reflect side. Within-side population reward/no-reward diagnostics:", "",
            json.dumps(meta["diagnostics"]["outcomes"], indent=2), "", "Per-group displayed counts:"]
        counts = display.group_index.value_counts()
        lines.extend(f"- {label}: {int(counts.get(group, 0))}" for group, label in enumerate(meta["group_labels"]))
        warnings = {name: reasons for name, reasons in meta["fit_reasons"].items() if reasons}
        lines += ["", f"Unavailable fit reasons: {json.dumps(warnings)}"]
        (directory / "summary.md").write_text("\n".join(lines)+"\n", encoding="ascii")
        performance = {"render_elapsed_s": perf_counter()-started,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
            "png_bytes": output.stat().st_size, "selected_units": len(display)}
        (directory / "performance.json").write_text(json.dumps(performance, indent=2), encoding="ascii")
        with log.open("a", encoding="ascii") as stream:
            stream.write(f"Complete; {json.dumps(performance)}; unavailable_fits={json.dumps(warnings)}\n")
    except Exception as error:
        with log.open("a", encoding="ascii") as stream:
            stream.write(f"ERROR {type(error).__name__}: {error}\n")
        raise
    return {"status": "complete", "region": meta["region"], "selection_path": str(selection_path),
            "run_directory": str(directory), "figure_path": str(output), **performance}


if __name__ == "__main__":
    raise SystemExit(main())
