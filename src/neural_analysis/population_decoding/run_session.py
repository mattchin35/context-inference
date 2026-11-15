"""Offline population task decoding; limit numerical threads before numerical imports."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


def initialize_config(metadata_path: Path | str, *, augmented_trials: Path | str | None = None,
                      probe_roles: dict | None = None, bin_width_ms: int = 500,
                      output_path: Path | str | None = None) -> Path:
    """Create one non-overwriting JSON beside scalar metadata Path; return config Path.

    Optional augmented-table Path defaults to the metadata source trial table,
    since fitted behavior values are unnecessary. Explicit PFC/HPC probe roles
    are required for version-2 metadata. Bin width is integer milliseconds.
    output_path optionally names a new config, relative to the metadata folder
    or absolute. Defaults to population_decoding.json. No analysis/behavior
    fitting or neural-array loading occurs; existing files are never replaced.
    """
    from .configuration import ANALYSIS_VERSION, load_config
    from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata

    metadata_path = Path(metadata_path).resolve(strict=True)
    session = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
    path = (metadata_path.parent / Path(output_path).expanduser()).resolve() if output_path is not None else metadata_path.parent / "population_decoding.json"
    trials = Path(augmented_trials).resolve() if augmented_trials is not None else session.behavior.trial_table_file
    if trials is None or not trials.is_file():
        raise ValueError("An existing complete trial table is required")
    payload = {"analysis_version": ANALYSIS_VERSION, "session_metadata": str(metadata_path),
               "augmented_trials": str(trials), "bin_width_ms": bin_width_ms}
    if probe_roles is not None:
        payload["probe_roles"] = probe_roles
    with path.open("x", encoding="ascii") as stream:
        json.dump(payload, stream, indent=2)
    try:
        load_config(path)
    except (ValueError, OSError):
        path.unlink()
        raise
    return path


def run_session(config_path: Path | str | None = None, *, command: str = "new",
                run_directory: Path | str | None = None, rerun: bool = False) -> dict:
    """Plan, execute, inspect, or resume one fixed population analysis; return JSON report.

    New/dry-run use scalar config Path; status/resume use only saved run Path.
    New reuses matching runs unless rerun requests a new directory. Published
    preparation allows resume without any raw sources. All times are seconds;
    no input data files or prior analysis products are overwritten.
    """
    from . import results, preparation
    from .configuration import ANALYSIS_VERSION, load_config

    if command not in ("new", "dry-run", "status", "resume") or (rerun and command != "new"):
        raise ValueError("Unsupported run command/rerun combination")
    if command in ("status", "resume"):
        if run_directory is None or config_path is not None:
            raise ValueError("status/resume require only run_directory")
        directory = Path(run_directory).resolve(strict=True)
        if command == "status":
            return results.inspect_run_status(directory)
        manifest = results.load_manifest(directory)
        if manifest["analysis_version"] != ANALYSIS_VERSION:
            raise ValueError("Legacy v1 is read-only; create a v2 config/run with init-config --output")
        if (manifest["identities"]["code_identity"] != results.code_identity()
                or manifest["identities"]["runtime"] != results.runtime_versions()):
            raise ValueError("Code/runtime changed; create a new analysis run")
        if not Path(manifest["preparation_path"]).exists():
            from .configuration import PopulationDecodingConfig
            payload = manifest["config"]
            config = PopulationDecodingConfig(Path(payload["session_metadata"]), Path(payload["augmented_trials"]),
                                               payload["probe_roles"], payload["bin_width_ms"])
            if results.compute_identities(config)["model_id"] != manifest["identities"]["model_id"]:
                raise ValueError("Sources changed before preparation; create a new run")
            results.save_preparation(Path(manifest["preparation_path"]), preparation.prepare_session(config), manifest["identities"])
        return _execute(directory)
    if config_path is None or run_directory is not None:
        raise ValueError("new/dry-run require only config_path")
    config = load_config(config_path)
    if command == "dry-run":
        from .targets import MODEL_TARGET_NAMES
        session, trials, behavior, baseline, targets, folds = preparation.load_behavior_inputs(config)
        return {"status": "dry-run", "session_id": session.session_id, "baseline_trials": int(baseline.sum()),
                "maximum_classifier_fits": 2*len(MODEL_TARGET_NAMES)*len(config.representations)*config.fold_count,
                "regions": dict(config.probe_roles),
                "blockers": preparation.inspect_sources(config), "bin_width_ms": config.bin_width_ms,
                "neural_counts": "Not loaded; coverage and class support remain pending"}
    identities = results.compute_identities(config)
    if not rerun:
        for candidate in results.discover_runs(config.run_root):
            if results.load_manifest(candidate)["identities"]["model_id"] == identities["model_id"]:
                report = results.inspect_run_status(candidate)
                return {**report, "status": "reused" if report["status"] == "complete" else "partial"}
    from .configuration import ANALYSIS_VERSION, SCHEMA_VERSION
    from .targets import MODEL_TARGET_NAMES
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    directory = config.run_root / f"population_task_decoding_{stamp}_{identities['model_id'][:12]}"
    directory.mkdir(parents=True)
    preparation_path = config.intermediate_root / identities["preparation_id"] / "preparation.npz"
    manifest = {"analysis_version": ANALYSIS_VERSION, "schema_version": SCHEMA_VERSION,
                "session_root": str(config.session_root), "config": results.config_payload(config),
                "identities": identities, "preparation_path": str(preparation_path),
                "model_targets": list(MODEL_TARGET_NAMES),
                "cells": [f"{region}_{target}" for region in ("PFC", "HPC") for target in MODEL_TARGET_NAMES]}
    results.write_json(directory / "config.json", manifest)
    for name in ("run_session.py", "run_batch.py"):
        shutil.copyfile(Path(__file__).with_name(name), directory / name)
    _append_log(directory, f"Starting session {config.session_root}; parameters={json.dumps(manifest['config'])}")
    start = perf_counter()
    try:
        if not preparation_path.exists():
            results.save_preparation(preparation_path, preparation.prepare_session(config), identities)
        else:
            saved = results.load_preparation(preparation_path)
            if saved.meta["preparation_id"] != identities["preparation_id"]:
                raise ValueError("Preparation identity differs; refusing overwrite")
        return _execute(directory, preparation_s=perf_counter() - start)
    except Exception as error:
        _append_log(directory, f"ERROR {type(error).__name__}: {error}")
        raise


def main(argv: list[str] | None = None) -> int:
    """Parse CLI string arguments (or sys.argv), print JSON report; return exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-config")
    init.add_argument("--session-metadata", type=Path, required=True)
    init.add_argument("--augmented-trials", type=Path)
    init.add_argument("--output", type=Path)
    init.add_argument("--pfc-probe")
    init.add_argument("--hpc-probe")
    init.add_argument("--bin-width-ms", type=int, choices=(500, 100, 50, 20), default=500)
    for name in ("new", "dry-run", "resume", "status"):
        sub = commands.add_parser(name)
        sub.add_argument("--config" if name in ("new", "dry-run") else "--run-directory", type=Path, required=True)
        if name == "new":
            sub.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "init-config":
            roles = {name: value for name, value in (("PFC", args.pfc_probe), ("HPC", args.hpc_probe)) if value is not None}
            report = {"config": str(initialize_config(args.session_metadata, augmented_trials=args.augmented_trials,
                probe_roles=roles or None, bin_width_ms=args.bin_width_ms, output_path=args.output))}
        else:
            report = run_session(getattr(args, "config", None), command=args.command,
                run_directory=getattr(args, "run_directory", None), rerun=getattr(args, "rerun", False))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Population decoding failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


def _execute(directory: Path, preparation_s: float = 0.0) -> dict:
    """Complete missing paired cells in saved run Path; return status with seconds/byte timings.

    Published files are authoritative and adopted after stale progress updates.
    Each missing paired cell is recomputed as a whole. Preparation supplies all
    neural/behavior data; no raw paths are opened after its publication.
    """
    from . import pipeline, results
    from .targets import CONDITIONAL_SWITCH_TARGETS
    manifest = results.load_manifest(directory)
    progress = results.inspect_run_status(directory)
    prepared = results.load_preparation(Path(manifest["preparation_path"]))
    if prepared.meta["preparation_id"] != manifest["identities"]["preparation_id"]:
        raise ValueError("Preparation identity differs from this run")
    start = perf_counter()
    for entry in manifest["cells"]:
        if entry in progress["completed_ids"]:
            continue
        region, target = entry.split("_", 1)
        cell_start = perf_counter()
        try:
            cell = pipeline.fit_target(prepared, region, target)
            if target in CONDITIONAL_SWITCH_TARGETS:
                reference_target = cell["meta"]["source_target"]
                reference_path = f"cells/{region}_{reference_target}.npz"
                reference = results.load_cell(directory / reference_path)
                if reference["meta"]["model_id"] != manifest["identities"]["model_id"]:
                    raise ValueError("Switch comparison reference belongs to a different run")
                comparison = pipeline.build_switch_comparison(prepared.targets.values[:, 1], reference, cell)
                cell["comparison_reference_window_indices"] = comparison["reference_window_indices"]
                cell["meta"]["switch_comparison"] = {**comparison["meta"],
                    "reference_model_id": manifest["identities"]["model_id"], "reference_cell": reference_path}
            results.save_cell(directory / "cells" / f"{entry}.npz", cell, manifest["identities"])
        except Exception as error:
            _append_log(directory, f"ERROR {entry}: {type(error).__name__}: {error}")
            raise
        _append_log(directory, f"{entry}: {perf_counter()-cell_start:.3f}s; unavailable={cell['meta']['unavailable_reason']}")
        for representation, summary in cell["meta"]["representations"].items():
            for fold in summary["folds"]:
                if not fold["available"]:
                    _append_log(directory, f"WARNING {entry} {representation} fold={fold['fold']}: {fold['reason']}")
        progress = results.inspect_run_status(directory)
        results.write_json(directory / "progress.json", progress)
    progress = results.inspect_run_status(directory)
    results.write_json(directory / "progress.json", progress)
    performance = {"elapsed_s": perf_counter() - start + preparation_s, "preparation_s": preparation_s,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
        "preparation_bytes": Path(manifest["preparation_path"]).stat().st_size,
        "result_bytes": sum(path.stat().st_size for path in (directory / "cells").glob("*.npz"))}
    # Completed runs retain their original measured execution, including on source-free resume.
    if not (directory / "performance.json").exists():
        results.write_json(directory / "performance.json", performance)
    lines = ["# Population task decoding", "", f"Session: {prepared.session_id}",
        f"Status: {progress['status']}; paired targets: {progress['completed_cells']}/{len(manifest['cells'])}.",
        "", "Goal: read context, choice, post-choice outcome and next chosen-port switch from population activity.",
        "Scripts: run_session.py; run_batch.py. Units and PCs use training-fold transforms and block-held-out probabilities.", ""]
    for entry in progress["completed_ids"]:
        cell = results.load_cell(directory / "cells" / f"{entry}.npz")
        for representation, summary in cell["meta"]["representations"].items():
            score = summary["pooled"]["balanced_accuracy"]
            lines.append(f"- {entry} ({representation}): balanced accuracy={score}; trials={summary['eligible_trials']}.")
        if cell["meta"]["unavailable_reason"]:
            lines.append(f"  Unavailable: {cell['meta']['unavailable_reason']}")
        if "switch_comparison" in cell["meta"]:
            lines.append("  Same-side comparison (reference all trials / reference matching side / new within side):")
            for rep, comparison in cell["meta"]["switch_comparison"]["representations"].items():
                scores = [comparison[name]["scores"]["balanced_accuracy"]
                          for name in ("reference_all_trials", "reference_matched_side", "within_side")]
                lines.append(f"  {rep}: balanced accuracy={scores}; matched trials={cell['meta']['switch_comparison']['required_trials']}.")
    lines += ["", "Scores describe held-out decoding, without significance or causal claims. Inspect side-conditional reward scores",
              "before interpreting a four-state outcome score as reward responsiveness. Within-side switching also decodes next chosen port."]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    _append_log(directory, f"{progress['status']}; elapsed_s={performance['elapsed_s']:.3f}")
    return progress


def _append_log(directory: Path, message: str) -> None:
    """Append timestamp/message strings to one run Path's UTF-8 log; return None."""
    with (directory / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now().isoformat()} {message}\n")


if __name__ == "__main__":
    raise SystemExit(main())
