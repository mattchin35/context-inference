"""Explicit session prediction preparation, reuse and small-input dry-run CLI."""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import logging
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd

from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population.run_session_timeline import (
    find_saved_timeline, probe_roles_from_arguments, resolve_timeline_probe_roles, run_session_timeline,
)
from src.neural_analysis.population.session_timeline import SessionTimelineResult
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.neural_analysis.task_decoding.config import load_task_decoding_config

from .config import (
    ANALYSIS_MODES, load_prediction_config, training_conditions_for_mode,
    training_window_bounds,
)
from .modeling import RegionPredictions, fit_region_prediction_models
from .preparation import PredictionInputs, prepare_prediction_inputs
from .results import build_prediction_metadata, find_saved_prediction, save_prediction_result


@dataclass(frozen=True)
class PredictionRunReport:
    """Small report: status, absolute saved path or None, identity, seconds, planning details.

    status is dry-run/prepared/reused. details contains only JSON-safe counts,
    source paths, bounds/units and timing; it retains no large numerical arrays.
    """

    status: str
    result_path: Path | None
    identity: str
    elapsed_s: float
    details: dict


def prepare_session_predictions(
    metadata_path: Path | str, *, config_path: Path | str | None = None,
    probe_roles: dict[str, str] | None = None, bin_width_ms: int | None = None,
    analysis_mode: str | None = None,
    training_window: str | None = None,
    rerun: bool = False, dry_run: bool = False,
) -> PredictionRunReport:
    """Prepare one mode's fixed targets on regional PCs, or reuse a matching result.

    metadata_path is neural_session.json; optional controls select a contained
    base config and millisecond width. Explicit PFC/HPC probe keys/width override
    defaults only for this request. Dry-run reads small tables/selection metadata,
    never spikes, PCA scores, fits or writes. Real preparation may explicitly
    prepare missing source PCA. rerun creates a NEW prediction directory but
    still reuses compatible PCA. Return counts/paths rather than large arrays.
    analysis_mode is an optional canonical mode override; None uses controls
    or the seven-target correct-rewarded default. training_window optionally
    overrides the controls-file full/before/after choice-relative fit interval.
    Mixed mode fits three targets.
    """
    started = perf_counter()
    controls = load_prediction_config(metadata_path, config_path=config_path,
                                      bin_width_ms=bin_width_ms, analysis_mode=analysis_mode,
                                      training_window=training_window)
    base = load_task_decoding_config(controls.base_config_path)
    if base.session_metadata_path != controls.session_metadata_path:
        raise ValueError("Base task-decoding config must belong to the selected metadata session")
    session = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
    roles = resolve_timeline_probe_roles(session, probe_roles)
    if not roles:
        raise ValueError("Assign PFC/HPC probe roles explicitly in the GUI/CLI or version-3 metadata")
    width_s = controls.bin_width_ms / 1000.0
    source_config = timeline_io.timeline_configuration(session, roles, width_s)
    trials = pd.read_csv(base.augmented_trial_path)
    origin, choices, events = timeline_io.load_timeline_behavior(session)
    behavior = SessionTimelineResult(session.session_id, origin, choices, events, {}, {}, source_config)
    # Validate chronology, labels and block count before any expensive PCA work.
    small_inputs = prepare_prediction_inputs(
        trials, behavior, base, analysis_mode=controls.analysis_mode,
        training_window=controls.training_window,
    )
    saved_source = find_saved_timeline(session.session_root, source_config["configuration_id"])
    window_bounds = list(training_window_bounds(controls.training_window))
    condition_counts = {
        name: int(small_inputs.condition_masks[:, small_inputs.condition_names.index(name)].sum())
        for name in ("correct_rewarded", "omission", "incorrect")
    }
    details = {
        "session_id": session.session_id, "bin_width_ms": controls.bin_width_ms,
        "analysis_mode": controls.analysis_mode, "training_window": controls.training_window,
        "training_window_s": window_bounds,
        "scoring_window_s": window_bounds,
        "training_conditions": list(training_conditions_for_mode(controls.analysis_mode)),
        "candidate_outcome_counts": condition_counts,
        "target_names": list(small_inputs.target_names),
        "probe_roles": roles, "choice_count": len(choices),
        "training_bins": int(small_inputs.training_bin_mask.sum()),
        "prediction_bins": len(small_inputs.bin_edges_s) - 1,
        "max_model_count": len(small_inputs.target_names) * len(roles) * 5,
        "prediction_bytes_upper_bound": len(choices) * (len(small_inputs.bin_edges_s) - 1) * len(small_inputs.target_names) * 8 * len(roles),
        "source_pca_missing": saved_source is None,
        "source_pca": str(saved_source) if saved_source is not None else None,
        "source_selection": {
            role: {"unavailable": selection["unavailable"]} if "unavailable" in selection
            else {"unit_count": len(selection["unit_ids"]), "channel_count": len(selection["channels"])}
            for role, selection in source_config["selection"].items()
        },
    }
    if dry_run:
        return PredictionRunReport("dry-run", None, source_config["configuration_id"], perf_counter() - started, details)
    if saved_source is None:
        saved_source = run_session_timeline(metadata_path, roles, width_s).result_path
    source_started = perf_counter()
    timeline = timeline_io.load_timeline_result(saved_source)
    if "reward" not in timeline.choices:
        timeline = replace(timeline, choices=timeline.choices.assign(reward=choices.reward.to_numpy()))
    prepared = prepare_prediction_inputs(
        trials, timeline, base, analysis_mode=controls.analysis_mode,
        training_window=controls.training_window,
    )
    source_loading_s = perf_counter() - source_started
    effective = {"bin_width_ms": controls.bin_width_ms, "probe_roles": roles,
                 "analysis_mode": controls.analysis_mode,
                 "training_window": controls.training_window,
                 "training_window_s": window_bounds,
                 "scoring_window_s": window_bounds}
    meta = build_prediction_metadata(
        prepared, session.session_root, saved_source,
        {"augmented_trials": base.augmented_trial_path, "feature_parameters": base.trial_feature_parameter_path},
        effective,
    )
    if not rerun:
        existing = find_saved_prediction(session.session_root, meta["identity"])
        if existing is not None:
            return PredictionRunReport("reused", existing, meta["identity"], perf_counter() - started, details)
    repository = Path(__file__).resolve().parents[4]
    output_root = session.session_root / "analysis_runs"
    if output_root.resolve().is_relative_to(repository):
        raise ValueError("Experimental prediction outputs must be outside the source repository")
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    directory = output_root / f"condition_session_prediction_{stamp}_{meta['identity']}"
    directory.mkdir(parents=True, exist_ok=False)
    logger = logging.Logger("session_prediction")
    handler = logging.FileHandler(directory / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("Session %s; parameters %s; source PCA %s", session.session_id, json.dumps(effective), saved_source)
        fitting_started = perf_counter()
        regions = {}
        for role, regional in timeline.regions.items():
            region_started = perf_counter()
            regions[role] = fit_region_prediction_models(
                regional.scores, regional.choice_positions, prepared.target_values,
                prepared.target_valid, prepared.baseline_valid, prepared.correct_rewarded,
                prepared.fold_ids, prepared.training_bin_mask, prepared.target_families,
                training_condition_mask=prepared.training_condition_mask,
            )
            model = regions[role]
            logger.info("%s: %d fitted models; %.3f s", role, int((model.fit_status == "ok").sum()), perf_counter() - region_started)
            for target, name in enumerate(prepared.target_names):
                for fold, reason in enumerate(model.fit_reason[target]):
                    if reason:
                        logger.warning("%s %s fold %d unavailable: %s", role, name, fold, reason)
        meta["timings_s"] = {"source_loading": source_loading_s, "fitting": perf_counter() - fitting_started,
                              "preparation_before_save": perf_counter() - started}
        meta["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False)
        meta["code_revision"] = revision.stdout.strip() if revision.returncode == 0 else "unknown"
        meta["fit_count"] = sum(int((model.fit_status == "ok").sum()) for model in regions.values())
        _snapshot_sources(directory)
        _write_summary(directory, prepared, regions, meta)
        save_started = perf_counter()
        destination = save_prediction_result(directory / "results.npz", prepared, regions, meta)
        logger.info("Published %s; save %.3f s; total %.3f s; NPZ bytes %d; peak RSS %d",
                    destination, perf_counter() - save_started, perf_counter() - started,
                    destination.stat().st_size, meta["peak_process_rss_bytes"])
    except Exception:
        logger.exception("Prediction preparation failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return PredictionRunReport("prepared", destination, meta["identity"], perf_counter() - started, details)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI strings and return 0 on success, 1 for ordinary input/filesystem errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bin-width-ms", type=int, choices=(500, 100, 50, 20))
    parser.add_argument("--analysis-mode", choices=ANALYSIS_MODES)
    parser.add_argument("--training-window", choices=("full", "before", "after"))
    parser.add_argument("--pfc-probe")
    parser.add_argument("--hpc-probe")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare_session_predictions(
            args.session_metadata, config_path=args.config,
            probe_roles=probe_roles_from_arguments(args.pfc_probe, args.hpc_probe),
            bin_width_ms=args.bin_width_ms, analysis_mode=args.analysis_mode,
            training_window=args.training_window,
            rerun=args.rerun, dry_run=args.dry_run,
        )
    except (OSError, ValueError) as error:
        print(f"Prediction preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"{report.status}: {report.result_path or report.identity} ({report.elapsed_s:.3f} s)")
    if args.dry_run:
        print(json.dumps(report.details, indent=2))
    return 0


# Run-document helpers; these do not implement numerical fitting.
def _snapshot_sources(directory: Path) -> None:
    """Copy actual new computational modules and both runner scripts into a new run."""
    source = Path(__file__).parent
    snapshots = directory / "source_snapshots"
    snapshots.mkdir()
    for name in ("config.py", "preparation.py", "modeling.py", "results.py", "run_session.py", "run_batch.py"):
        shutil.copyfile(source / name, snapshots / name)
    shutil.copyfile(__file__, directory / "run_session.py")
    shutil.copyfile(source / "run_batch.py", directory / "run_batch.py")


def _write_summary(
    directory: Path, prepared: PredictionInputs,
    regions: dict[str, RegionPredictions], meta: dict,
) -> None:
    """Write a human-readable session/parameter/status summary; input axes stay unchanged."""
    window_start_s, window_stop_s = training_window_bounds(prepared.training_window)
    lines = ["# Condition session prediction", "", f"Session: {meta['session_id']}", "",
             f"Analysis mode: {prepared.analysis_mode}",
             "Goal: inspect individual choice-window readouts.",
             "Training conditions: " + ", ".join(training_conditions_for_mode(prepared.analysis_mode)),
             "Block-held-out prediction using a fixed session-wide PCA basis.",
             "PCA and unit scaling include held-out neural data; this is an offline readout.",
             f"Training and scoring interval: [{window_start_s:g},{window_stop_s:g}) s "
             f"choice-relative ({prepared.training_window}).",
             "Prediction: [-2,+2) s. No intertrial predictions.",
             "Ridge alpha=1; logistic C=1; each training trial has total weight one.",
             "Scripts: run_session.py; run_batch.py. Actual computational modules: source_snapshots/.",
             "State and action readouts may coincide because their training labels coincide.", "",
             f"Source PCA: {meta['source_pca']['path']}", f"Fitted models: {meta['fit_count']}",
             f"Timings (seconds): {json.dumps(meta['timings_s'])}", ""]
    for role, model in regions.items():
        lines.append(f"## {role}")
        for target, name in enumerate(prepared.target_names):
            lines.append(f"- {name}: {int((model.fit_status[target] == 'ok').sum())}/5 fitted folds; "
                         f"training trials {model.training_trial_count[target].tolist()}.")
            for reason in sorted(set(model.fit_reason[target]) - {""}):
                lines.append(f"  Unavailable: {reason}.")
    for role, reason in prepared.timeline.unavailable.items():
        lines.append(f"- {role} unavailable: {reason}.")
    if "current_reward" in prepared.target_names:
        lines.extend([
            "", "Actual reward class meaning: class 1 is rewarded; class 0 is unrewarded.",
            "Before predicts eventual recorded reward, not subjective expectation.",
            "After is post-choice, not necessarily post-delivery.",
            "Reward overlays are recorded trial outcomes, not delivery-event timing.",
            "Mixed training changes outcome composition and sample size. At fixed logistic C, more trials reduce regularization relative to average loss; this is not a controlled composition-only comparison.",
        ])
    lines.extend(["", "No statistical tests were performed. Contribution rankings describe varying additive terms, not unique or causal importance.", ""])
    (directory / "summary.md").write_text("\n".join(lines), encoding="ascii")


if __name__ == "__main__":
    raise SystemExit(main())
