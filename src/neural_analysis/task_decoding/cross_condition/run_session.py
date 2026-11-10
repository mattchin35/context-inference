"""Explicit metadata-based preparation and saved reuse of cross-condition readouts."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter
from zipfile import BadZipFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population.run_session_timeline import find_saved_timeline, run_session_timeline
from src.neural_analysis.population.session_timeline import choice_bin_edges, prepare_timeline_choices
from src.neural_analysis.session_metadata import ResolvedSession, load_session_metadata, resolve_session_metadata
from src.neural_analysis.task_decoding.config import TaskDecodingConfig, load_task_decoding_config

from .config import WEIGHTING_NAMES, WINDOW_NAMES, load_cross_condition_config
from .modeling import RegionTransferResult, fit_region_transfers
from .preparation import TransferInputs, prepare_transfer_inputs
from .results import (
    SCIENTIFIC_FILES, CrossConditionResult, build_transfer_metadata, file_identity,
    find_saved_transfer, save_transfer_result,
)


POPULATION_QUALITY = "good_mua"
MISSING_LABEL_POLICY = "mua"
SOURCE_ERRORS = (OSError, ValueError, KeyError, EOFError, BadZipFile)


@dataclass(frozen=True)
class TransferRunReport:
    """Small execution report; no neural/model arrays retained.

    status is prepared/reused/dry-run, result_path an absolute Path or None,
    identity a fitting/planning string and elapsed_s wall-clock seconds.
    details contains JSON-safe controls, global behavioral counts before
    regional coverage, and per-region source status/path/reason dictionaries.
    """

    status: str
    result_path: Path | None
    identity: str
    elapsed_s: float
    details: dict


def prepare_cross_condition_generalization(
    metadata_path: Path | str, *, config_path: Path | str | None = None,
    bin_width_ms: int | None = None, rerun: bool = False, dry_run: bool = False,
) -> TransferRunReport:
    """Prepare all sixteen transfers/three windows/two weightings or reuse an exact saved bundle.

    metadata_path names version-3 neural_session.json and its PFC/HPC roles.
    Optional contained config_path/explicit integer bin_width_ms select controls.
    rerun creates a new transfer run while reusing compatible PCA; dry_run reads
    only small behavior/selection/archive metadata, never scores, fits, events,
    workers or output files. Source PCA retains (W,B,K) original unwhitened
    score units; only matched supervised bins produce trial means. Return a
    TransferRunReport with JSON-safe counts/paths; local regional source failures
    remain reportable. No experimental outputs may be written inside this repo.
    """
    started = perf_counter()
    controls = load_cross_condition_config(metadata_path, config_path=config_path, bin_width_ms=bin_width_ms)
    base = load_task_decoding_config(controls.base_config_path)
    session = resolve_session_metadata(load_session_metadata(controls.session_metadata_path), controls.session_metadata_path)
    roles, width_s = controls.probe_roles, controls.bin_width_ms / 1000.
    trials = pd.read_csv(base.augmented_trial_path)
    behavior = pd.read_csv(session.behavior.trial_table_file)
    _, choices = prepare_timeline_choices(behavior)
    edges = choice_bin_edges(width_s)
    small_inputs = prepare_transfer_inputs(trials, choices, edges, base)
    selection_started = perf_counter()
    sources, statuses, source_config = _find_exact_sources(session, roles, width_s)
    details = _planning_details(small_inputs, session.session_id, roles, controls.bin_width_ms, statuses)
    if dry_run:
        return TransferRunReport("dry-run", None, source_config["configuration_id"], perf_counter() - started, details)
    repository = Path(__file__).resolve().parents[4]
    output_root = session.session_root / "analysis_runs"
    if output_root.resolve().is_relative_to(repository):
        raise ValueError("Experimental transfer outputs must be outside the source repository")
    unavailable = _resolve_missing_sources(session, roles, width_s, sources, statuses)
    selection_s = perf_counter() - selection_started
    effective = {"bin_width_ms": controls.bin_width_ms, "probe_roles": roles}
    meta = build_transfer_metadata(
        small_inputs, session.session_root, sources,
        {"augmented_trials": base.augmented_trial_path, "base_config": controls.base_config_path,
         "feature_parameters": base.trial_feature_parameter_path}, effective, session_id=session.session_id,
    )
    if not rerun:
        existing = find_saved_transfer(session.session_root, meta["identity"])
        if existing is not None:
            return TransferRunReport("reused", existing, meta["identity"], perf_counter() - started, details)
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    directory = output_root / f"cross_condition_generalization_{stamp}_{meta['identity']}"
    directory.mkdir(parents=True, exist_ok=False)
    logger = logging.Logger("cross_condition_generalization")
    handler = logging.FileHandler(directory / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("Session %s; controls %s; source selection %.3f s", session.session_id, json.dumps(effective), selection_s)
        for role, status in statuses.items():
            logger.info("%s source: %s", role, json.dumps(status))
        prepared, regions, loading_s, fitting_s = _load_and_fit_sources(
            trials, base, small_inputs, edges, roles, sources, unavailable, statuses, logger,
        )
        meta.update(
            fit_count=sum(int((model.fit_status == "ok").sum()) for model in regions.values()),
            runtime_s=perf_counter() - started,
            runtime_scope="preparation before result publication; total runtime and save time are logged",
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            timings_s={"source_selection": selection_s, "source_loading": loading_s, "fitting": fitting_s},
            execution_sources=_execution_source_identities(repository),
        )
        logger.info("Source loading %.3f s; fitting %.3f s; %d fitted models; peak RSS %d bytes",
                    loading_s, fitting_s, meta["fit_count"], meta["peak_process_rss_bytes"])
        _snapshot_sources(directory)
        _write_summary(directory, prepared, regions, unavailable, meta)
        saving_started = perf_counter()
        destination = save_transfer_result(directory / "results.npz", CrossConditionResult(prepared, regions, unavailable, meta))
        logger.info("Saved %s in %.3f s; total runtime %.3f s", destination, perf_counter() - saving_started, perf_counter() - started)
    except Exception:
        logger.exception("Cross-condition preparation failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return TransferRunReport("prepared", destination, meta["identity"], perf_counter() - started, details)


def _find_exact_sources(
    session: ResolvedSession, roles: dict[str, str], width_s: float,
) -> tuple[dict[str, Path], dict[str, dict], dict]:
    """Return exact/registered source Paths, JSON-safe status records and metadata-only full configuration.

    session/roles identify metadata and PFC/HPC probes; width_s is seconds.
    Only small configuration/NPZ meta is read, never score arrays. Try the full
    configuration first, then unresolved individual roles. Unknown regional
    compatibility remains pending until explicit preparation permits array reads.
    """
    config = timeline_io.timeline_configuration(session, roles, width_s,
        population_quality=POPULATION_QUALITY, missing_label_policy=MISSING_LABEL_POLICY)
    sources = {}
    statuses = {role: {"status": "compatibility_pending", "result_path": None,
                       "reason": "Regional compatibility requires array verification during explicit preparation"} for role in roles}
    path = find_saved_timeline(session.session_root, config["configuration_id"])
    if path is not None:
        _accept_exact_source(path, roles, width_s, sources, statuses)
    for role, probe in roles.items():
        if role in sources:
            continue
        regional = timeline_io.timeline_configuration(session, {role: probe}, width_s,
            population_quality=POPULATION_QUALITY, missing_label_policy=MISSING_LABEL_POLICY)
        path = find_saved_timeline(session.session_root, regional["configuration_id"])
        if path is not None:
            _accept_exact_source(path, {role: probe}, width_s, sources, statuses)
    return sources, statuses, config


def _accept_exact_source(
    path: Path, roles: dict[str, str], width_s: float, sources: dict[str, Path], statuses: dict[str, dict],
) -> None:
    """Accept only available requested roles from one verified exact/registered archive's dictionary meta.

    path names a PCA NPZ, roles maps requested probes, width_s is seconds.
    Mutate only source/status dictionaries; unavailable roles stay unresolved,
    including when a whole-session archive contains only one usable region.
    """
    try:
        meta = _source_metadata(path)
        if meta.get("bin_width_s") != width_s:
            return
        for role, probe in roles.items():
            if meta.get("regions", {}).get(role, {}).get("probe_id") == probe:
                sources[role] = Path(path).resolve()
                statuses[role] = {"status": "exact_or_registered", "result_path": str(sources[role]), "reason": ""}
    except SOURCE_ERRORS as error:
        for role in roles:
            statuses[role]["reason"] = f"Exact source metadata unavailable; regional verification still required: {error}"


def _source_metadata(path: Path) -> dict:
    """Read only dictionary metadata from one existing PCA NPZ Path; no score arrays or unit conversion."""
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
    if not isinstance(meta, dict) or (meta.get("analysis_version"), meta.get("schema_version")) != ("session-timeline-pca-v1", 1):
        raise ValueError("Selected source is not a supported session timeline PCA")
    return meta


def _resolve_missing_sources(
    session: ResolvedSession, roles: dict[str, str], width_s: float,
    sources: dict[str, Path], statuses: dict[str, dict],
) -> dict[str, str]:
    """Resolve remaining regional bases, returning unavailable reasons and updating source/status maps.

    session/roles specify PFC/HPC probes; width_s is seconds. Use the existing
    scientific regional compatibility verifier with explicit good_mua/mua rules.
    Prepare only still-missing roles in at most one shared runner call. Existing
    verified sources never change even if preparation returns additional regions.
    """
    for role, probe in roles.items():
        if role in sources:
            continue
        try:
            path = timeline_io.find_compatible_regional_timeline(session, role, probe, width_s,
                population_quality=POPULATION_QUALITY, missing_label_policy=MISSING_LABEL_POLICY)
            if path is not None:
                sources[role] = Path(path).resolve()
                statuses[role] = {"status": "reused_compatible", "result_path": str(sources[role]), "reason": ""}
        except SOURCE_ERRORS as error:
            statuses[role]["reason"] = f"Regional compatibility lookup failed: {error}"
    missing = {role: probe for role, probe in roles.items() if role not in sources}
    unavailable = {}
    if not missing:
        return unavailable
    try:
        report = run_session_timeline(session.metadata_path, missing, width_s,
            population_quality=POPULATION_QUALITY, missing_label_policy=MISSING_LABEL_POLICY)
        if report.result_path is None:
            raise ValueError("PCA preparation returned no source archive")
        meta = _source_metadata(report.result_path)
        for role, probe in missing.items():
            if meta.get("regions", {}).get(role, {}).get("probe_id") == probe:
                sources[role] = Path(report.result_path).resolve()
                statuses[role] = {"status": "prepared", "result_path": str(sources[role]), "reason": ""}
            else:
                unavailable[role] = meta.get("unavailable", {}).get(role, "Requested regional PCA is unavailable after preparation")
    except SOURCE_ERRORS as error:
        unavailable = {role: f"PCA preparation failed: {error}" for role in missing}
    for role, reason in unavailable.items():
        statuses[role] = {"status": "unavailable", "result_path": None, "reason": str(reason)}
    return unavailable


def _planning_details(
    inputs: TransferInputs, session_id: str, roles: dict[str, str], width_ms: int, statuses: dict[str, dict],
) -> dict:
    """Return JSON-safe global behavioral planning counts plus a live regional source-status map.

    inputs has N-choice labels/masks and cells(N,S,2), before regional coverage.
    width_ms is integer milliseconds; window bin counts are dimensionless.
    statuses is shared with later source resolution/load failures so returned
    details reflect final source status rather than a stale planning snapshot.
    No neural arrays or per-trial outputs are retained.
    """
    return {
        "session_id": session_id, "bin_width_ms": width_ms, "probe_roles": dict(roles),
        "choice_count": len(inputs.choices), "target_names": list(inputs.target_names),
        "window_bin_counts": {name: int(inputs.window_bin_masks[index].sum()) for index, name in enumerate(inputs.window_names)},
        "max_model_count": len(inputs.transfers) * len(WINDOW_NAMES) * len(WEIGHTING_NAMES) * len(roles),
        "dark_state_count": inputs.dark_state_count, "conflict_count": len(inputs.conflicts),
        "conflict_trial_ids": inputs.conflicts.trial_id.tolist(),
        "cell_counts_scope": "global behavioral eligibility before regional neural coverage",
        "cell_counts": [{"split_id": definition.split_id,
                         "source": inputs.source_cells[:, split].sum(axis=0).tolist(),
                         "complementary_test": inputs.test_cells[:, split].sum(axis=0).tolist()}
                        for split, definition in enumerate(inputs.transfers)],
        "source_status": statuses,
    }


def _load_and_fit_sources(
    trials: pd.DataFrame, base: TaskDecodingConfig, small_inputs: TransferInputs, edges: np.ndarray,
    roles: dict[str, str], sources: dict[str, Path], unavailable: dict[str, str],
    statuses: dict[str, dict], logger: logging.Logger,
) -> tuple[TransferInputs, dict[str, RegionTransferResult], float, float]:
    """Load each selected archive once, align its choices and fit its assigned regions sequentially.

    trials is chronological (J,column), small_inputs has N shared choices, and
    edges is float(B+1,) in choice-relative seconds. roles/sources map regional
    names to probe IDs/Paths. Saved scores keep (W,B,K) axes and original PC
    units. Return common prepared inputs, small regional results and load/fit
    durations in seconds. Record ordinary source/alignment failures per region
    in unavailable/statuses and logger, retaining known source references.
    Source validation does not borrow another archive's nonassigned regions.
    """
    prepared, regions = small_inputs, {}
    have_aligned_source = False
    loading_s, fitting_s = 0., 0.
    for path in dict.fromkeys(sources.values()):
        assigned = [role for role, source in sources.items() if source == path]
        load_started = perf_counter()
        try:
            timeline = timeline_io.load_timeline_result(path)
            aligned = prepare_transfer_inputs(trials, timeline.choices, edges, base)
        except SOURCE_ERRORS as error:
            loading_s += perf_counter() - load_started
            for role in assigned:
                reason = f"Source PCA load/alignment failed: {error}"
                unavailable[role] = reason
                statuses[role] = {"status": "unavailable", "result_path": str(path), "reason": reason}
                logger.warning("%s source unavailable: %s", role, reason)
            continue
        loading_s += perf_counter() - load_started
        if not have_aligned_source:
            prepared, have_aligned_source = aligned, True
        for role in assigned:
            fit_started = perf_counter()
            try:
                if role not in timeline.regions:
                    raise ValueError(timeline.unavailable.get(role, "Requested regional basis is absent"))
                source = timeline.regions[role]
                if source.probe_id != roles[role] or not np.array_equal(source.bin_edges_s, edges):
                    raise ValueError("Regional probe or saved bin grid contradicts selected controls")
                model = fit_region_transfers(source.scores, source.choice_positions, aligned)
            except SOURCE_ERRORS as error:
                reason = f"Regional source arrays unavailable: {error}"
                unavailable[role] = reason
                statuses[role] = {"status": "unavailable", "result_path": str(path), "reason": reason}
                logger.warning("%s source unavailable: %s", role, reason)
            else:
                regions[role] = model
                logger.info("%s: %d fitted models; eligible cell counts %s", role,
                            int((model.fit_status == "ok").sum()), model.eligible_cell_counts.tolist())
                for reason in sorted(set(model.fit_reason.ravel()) - {""}):
                    logger.warning("%s model unavailable: %s", role, reason)
            fitting_s += perf_counter() - fit_started
    return prepared, regions, loading_s, fitting_s


def _execution_source_identities(repository: Path) -> dict[str, dict]:
    """Return portable hashes of existing execution/report modules as provenance, outside fitting identity.

    repository is the source checkout Path. The real batch script is included
    once supplied by P5b; required publication snapshots independently require
    that actual script rather than synthesizing a substitute.
    """
    prefix = "src/neural_analysis/task_decoding/cross_condition/"
    names = ("run_session.py", "run_batch.py", "results.py", "scoring.py")
    return {prefix + name: file_identity(repository / (prefix + name), repository)
            for name in names if (repository / (prefix + name)).is_file()}


def _snapshot_sources(directory: Path) -> None:
    """Copy actual main/batch scripts and computation/report modules into one new run directory.

    directory is a newly created run Path. source_snapshots preserves repository-
    relative paths so shared/config/modeling basenames never collide. All
    snapshot bytes come from existing project files; no production stubs exist.
    Missing required scripts raise normally before final NPZ publication.
    """
    source = Path(__file__).resolve()
    repository = source.parents[4]
    prefix = "src/neural_analysis/task_decoding/cross_condition/"
    names = dict.fromkeys((*SCIENTIFIC_FILES, *(prefix + name for name in
                          ("run_session.py", "run_batch.py", "results.py", "scoring.py"))))
    for name in names:
        target = directory / "source_snapshots" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, target)
    shutil.copyfile(source, directory / "run_session.py")
    shutil.copyfile(source.with_name("run_batch.py"), directory / "run_batch.py")


def _write_summary(
    directory: Path, inputs: TransferInputs, regions: dict[str, RegionTransferResult],
    unavailable: dict[str, str], meta: dict,
) -> None:
    """Write one session's goal/scripts/settings/availability summary from small prepared/model records.

    inputs/regions retain their N/W/B/K/H/G/S axes and units; no scores are
    loaded or metrics refitted. directory is the new run Path; metadata contains
    session identity, source references and measured preparation timing/RSS.
    """
    lines = ["# Cross-condition generalization", "", f"Session: {meta['session_id']}", "",
             "Goal: fit all sixteen binary transfers in three supervised windows and two weighting schemes.",
             "Scripts: run_session.py; run_batch.py. Actual computation sources: source_snapshots/.",
             "Training is in-sample performance; complementary-condition tests remain separate.",
             "The fixed session-wide PCA and unit scaling include test-condition neural activity.",
             "This is offline condition generalization, not block-held-out or temporal validation.",
             "PCA source interval: [-2,+2) s. Decoder outputs use only each matched supervised interval:",
             "Full [-1,+1), Before [-1,0), After [0,+1) seconds relative to choice; zero belongs to After.",
             "Average bin probabilities before applying the 0.5 trial threshold.",
             "When both source cells are populated, equal-trial and equal-cell weights sum to the eligible source trial count.",
             "Evaluation gives each scored trial equal weight; outcome summaries add no models.",
             "Value (correctness-based) means P(correct), not subjective value or expected reward.",
             "Actual reward means P(rewarded); correct omissions have Reward=0 and Value=1.",
             "Before predicts eventual recorded reward; After is post-choice, not necessarily post-delivery.",
             f"Bin width: {meta['effective_config']['bin_width_ms']} ms; fitted models: {meta['fit_count']}.",
             f"Excluded dark choices: {inputs.dark_state_count}; conflicting recorded labels: {len(inputs.conflicts)}.",
             f"Conflict original trial IDs: {inputs.conflicts.trial_id.tolist()}.",
             f"Timing seconds: {json.dumps(meta['timings_s'])}; total runtime/save timing: run.log.", ""]
    for role, model in regions.items():
        lines.append(f"- {role}: {int((model.fit_status == 'ok').sum())}/96 fitted models; "
                     f"source PCA {meta['region_sources'][role]}.")
        for reason in sorted(set(model.fit_reason.ravel()) - {""}):
            lines.append(f"  Unavailable model reason: {reason}.")
    for role, reason in unavailable.items():
        lines.append(f"- {role} unavailable: {reason}.")
    lines.extend(["", "No statistical tests were performed. Overlapping populations and transfer directions are not independent repetitions.", ""])
    (directory / "summary.md").write_text("\n".join(lines), encoding="ascii")


def main(argv: list[str] | None = None) -> int:
    """Parse CLI string arguments and return status 0, or 1 for ordinary input/filesystem errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bin-width-ms", type=int, choices=(500, 100, 50, 20))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare_cross_condition_generalization(args.session_metadata, config_path=args.config,
            bin_width_ms=args.bin_width_ms, rerun=args.rerun, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        print(f"Cross-condition preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"{report.status}: {report.result_path or report.identity} ({report.elapsed_s:.3f} s)")
    if args.dry_run:
        print(json.dumps(report.details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
