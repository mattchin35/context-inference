"""Prepare one session's behavioral examples; plotting loads saved results separately."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from io import BytesIO
import json
import logging
from pathlib import Path
import shutil
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd

from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from . import example_io, examples, selection


ANALYSIS_SOURCE_FILES = ("selection.py", "examples.py", "example_io.py", "run_examples.py", "run_examples_batch.py")


def prepare_behavior_examples(
    metadata_path: Path | str, *, probe_return_outcome: str = "rewarded_or_omission",
    rerun: bool = False, dry_run: bool = False,
) -> dict:
    """Prepare/reuse behavioral matches for one explicitly selected session.

    metadata_path is neural_session.json Path/string. Only its processed trial
    CSV is loaded; input columns/units follow selection.prepare_sequence_trials.
    probe_return_outcome is rewarded or rewarded_or_omission for both probe
    families. rerun forces a new immutable result; dry_run validates/selects
    small behavior inputs but writes nothing. Return a dict with status,
    result_path/run_directory Paths (None for dry-run), configuration_id and
    elapsed_s (wall seconds). NPZ data stay below the source trial directory;
    logs/script snapshots/summary/figures stay in a dated session analysis run.
    No neural files, population configuration or neural result archives are read.
    """
    started = perf_counter()
    path = Path(metadata_path).resolve()
    session = resolve_session_metadata(load_session_metadata(path), path)
    source = session.behavior.trial_table_file
    if source is None:
        raise ValueError("Session metadata needs a behavior trial CSV.")
    source_bytes = source.read_bytes()
    parameters = {"probe_return_outcome": probe_return_outcome}
    directory = Path(__file__).parent
    configuration = {
        "analysis_version": example_io.ANALYSIS_VERSION, "selector_version": selection.ADDITIONAL_SELECTOR_VERSION,
        "session_id": session.session_id, "source_path": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "parameters": parameters,
        "code_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ANALYSIS_SOURCE_FILES},
    }
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode("ascii")).hexdigest()[:16]
    cache_root = source.parent / "behavior_examples" / example_io.ANALYSIS_VERSION / identity
    if not rerun and not dry_run:
        for candidate in sorted(cache_root.glob("*/results.npz"), reverse=True):
            try:
                saved = example_io.load_examples(candidate)
            except (OSError, ValueError):
                continue
            if saved["meta"].get("configuration") == configuration:
                return _report("reused", candidate, Path(saved["meta"]["run_directory"]), identity, started)

    raw_trials = pd.read_csv(BytesIO(source_bytes))
    matches = examples.find_behavior_examples(raw_trials, probe_return_outcome=probe_return_outcome)
    trials = selection.prepare_sequence_trials(raw_trials)
    if dry_run:
        return _report("dry-run", None, None, identity, started)
    repository = Path(__file__).resolve().parents[3]
    if session.session_root.is_relative_to(repository):
        raise ValueError("Experimental behavior outputs must be outside the source repository.")
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    run_directory = session.session_root / "analysis_runs" / f"behavior_examples_{stamp}_{identity}"
    run_directory.mkdir(parents=True, exist_ok=False)
    logger = logging.Logger("behavior_examples")
    handler = logging.FileHandler(run_directory / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("Session %s; parameters %s; configuration %s", session.session_id, parameters, identity)
        meta = {
            "generator": "run_examples.prepare_behavior_examples", "analysis_version": example_io.ANALYSIS_VERSION,
            "schema_version": example_io.SCHEMA_VERSION, "configuration_id": identity,
            "configuration": configuration, "parameters": parameters, "random_seed": None,
            "session_id": session.session_id, "metadata_path": str(path), "run_directory": str(run_directory),
            "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "units": {"choice_time": "UTC Unix seconds", "action": "0=right, 1=left", "state_int": "0=right, 1=left, 2=dark"},
            "axes": {"trials": ["original_source_row", "field"], "matches": ["episode", "trial_segment"]},
        }
        result_path = example_io.save_examples(cache_root / stamp / "results.npz", {"trials": trials, "matches": matches, "meta": meta})
        for name in (*ANALYSIS_SOURCE_FILES, "example_plotting.py"):
            shutil.copy2(directory / name, run_directory / name)
        counts = {family: sum(len(values) for key, values in matches.items() if key[0] == family) for family in examples.FAMILIES}
        summary = (
            "# Behavioral examples\n\nGoal: inspect recorded choices in selected behavioral patterns.\n\n"
            f"Session: {session.session_id}\n\nScripts: run_examples.py; run_examples_batch.py.\n\n"
            f"Parameters: {json.dumps(parameters, sort_keys=True)}\n\n"
            f"Source: {source}\n\nResults: {result_path}\n\n"
            f"Episode counts: {json.dumps(counts, sort_keys=True)}\n\n"
            "Counts can overlap across families. Matching is deterministic; no statistical tests were performed.\n"
        )
        (run_directory / "summary.md").write_text(summary, encoding="utf-8")
        logger.info("Episode counts %s; result %s; elapsed seconds %.6f", counts, result_path, perf_counter() - started)
    except Exception:
        logger.exception("Behavior example preparation failed; elapsed seconds %.6f", perf_counter() - started)
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return _report("prepared", result_path, run_directory, identity, started)


def export_behavior_example(
    result_path: Path | str, condition: examples.SequenceCondition, episode_index: int, *,
    trials_before: int = 5, trials_after: int = 5,
) -> Path:
    """Load a saved match and export a new light PNG plus exact JSON selection.

    result_path is a trusted-local behavior NPZ; condition is a unitless
    family/side/repeat key; episode_index is zero-based chronological index.
    Context counts are original trial rows. Return the PNG Path under the
    saved run directory, with a same-stem JSON describing source/result/code
    identity, trial IDs and display parameters. Source CSV/neural data are not
    read, no matches are recomputed, and earlier exports are never overwritten.
    """
    from . import example_plotting

    packet = example_io.load_examples(result_path)
    if condition not in packet["matches"]:
        raise ValueError("Unknown behavioral example condition.")
    positions = packet["matches"][condition]
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or not 0 <= episode_index < len(positions):
        raise ValueError("episode_index must select an existing saved episode.")
    meta = packet["meta"]
    view = example_plotting.build_example_view(
        packet["trials"], positions[episode_index], session_id=meta["session_id"], condition=condition,
        trials_before=trials_before, trials_after=trials_after,
        probe_return_outcome=meta["parameters"]["probe_return_outcome"],
    )
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(meta["run_directory"]) / "figures" / f"{condition[0]}_{condition[1]}_{episode_index + 1}_{stamp}.png"
    example_plotting.export_png(view, output)
    provenance = {
        "generator": "run_examples.export_behavior_example", "result_path": str(Path(result_path).resolve()),
        "configuration_id": meta["configuration_id"], "analysis_version": meta["analysis_version"],
        "parameters": meta["parameters"], "condition": condition, "episode_index": episode_index,
        "episode_row_positions": positions[episode_index].tolist(), "episode_trial_ids": view["episode_trial_ids"],
        "display_trial_ids": view["rows"].trial_id.tolist(), "trials_before": trials_before, "trials_after": trials_after,
        "caption": view["caption"], "theme": "light", "dpi": 200,
        "plotting_sha256": hashlib.sha256(Path(example_plotting.__file__).read_bytes()).hexdigest(),
    }
    with output.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
    with (Path(meta["run_directory"]) / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"Exported {output}; condition={condition}; episode_index={episode_index}; context={trials_before},{trials_after}\n")
    return output


def _report(status: str, path: Path | None, run_directory: Path | None, identity: str, started: float) -> dict:
    """Map status/paths/configuration/start clock (seconds) to a small wall-time report dict."""
    return {"status": status, "result_path": path, "run_directory": run_directory, "configuration_id": identity,
            "elapsed_s": perf_counter() - started}


def main() -> None:
    """Parse CLI metadata path/scalar options, prepare one session and print its report; return None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-metadata", type=Path, required=True)
    parser.add_argument("--probe-return-outcome", choices=("rewarded", "rewarded_or_omission"), default="rewarded_or_omission")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    report = prepare_behavior_examples(arguments.session_metadata, probe_return_outcome=arguments.probe_return_outcome,
                                       rerun=arguments.rerun, dry_run=arguments.dry_run)
    print(json.dumps(report, default=str, indent=2))


if __name__ == "__main__":
    main()
