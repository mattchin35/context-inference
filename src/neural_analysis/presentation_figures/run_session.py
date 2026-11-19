"""Compute or reuse Figures 1-6 for one metadata-defined session."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from src.neural_analysis import session_speed
from . import preparation, results


def run_session(metadata_path: str | Path, *, seed: int = 0, dry_run: bool = False,
                rerun: bool = False, probe_roles: dict[str, str] | None = None,
                reuse_fixed_from: Path | None = None) -> dict:
    """Run Figures 1-6 on one session Path, returning a JSON-compatible run report.

    Seed controls population draws. Dry runs inspect source identities, labels,
    and coverage only. A completed compatible run is reused unless rerun is set.
    reuse_fixed_from names a reviewed prior manifest whose all-unit episode
    fits and movement analysis are unchanged. The PC analyses are computed
    together. Physical axes and units are recorded in the session-local NPZs.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("The population draw seed must be a nonnegative integer.")
    started = perf_counter()
    inspection = preparation.preflight(metadata_path, probe_roles=probe_roles)
    identity = session_identity(inspection, seed)
    root = inspection.session.session_root / "analysis_runs"
    unit_counts = {region: len(ids) for region, ids in inspection.unit_ids.items()}
    # Metadata stores a single session label (e.g. CT026_2026-08-28_134630).
    mouse_id = re.split(r"_\d{4}-\d{2}-\d{2}(?:_|$)", inspection.session_id, maxsplit=1)[0]
    report = dict(session_id=inspection.session_id, mouse_id=mouse_id, metadata=str(inspection.session.metadata_path),
                  identity=identity, good_units=unit_counts, roles=inspection.probe_roles,
                  coverage_s=inspection.coverage_s, trial_audit=inspection.meta["trial_audit"],
                  choices_by_group={key: int(mask.sum()) for key, mask in inspection.condition_masks.items()},
                  eligible_narrow=len(inspection.windows[((-1., 1.), 100)].row_positions),
                  eligible_wide=len(inspection.windows[((-2., 2.), 100)].row_positions),
                  unavailable=inspection.unavailable, manifest=None)
    if dry_run:
        return dict(report, status="dry-run")
    if len(unit_counts) != 2 or min(unit_counts.values()) < 4:
        return dict(report, status="excluded", reason="Session requires four good units in each region.")
    if not rerun:
        matches = results.discover_runs(root, identity)
        if matches:
            return dict(report, status="reused", manifest=str(matches[0]))

    reused = {}
    if reuse_fixed_from is not None:
        reuse_fixed_from = Path(reuse_fixed_from).resolve()
        reused = reviewed_stages(reuse_fixed_from, inspection, seed)
    directory = new_run_directory(root)
    for name in ("run_session.py", "run_batch.py"):
        shutil.copyfile(Path(__file__).with_name(name), directory / name)
    report_path = directory / "run.log"
    archives, observations = {}, []
    with report_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps(dict(report, seed=seed, version=results.ANALYSIS_VERSION)) + "\n")
        try:
            prepared = preparation.prepare_choices(metadata_path, probe_roles=probe_roles)
            if prepared.meta["sources"] != inspection.meta["sources"]:
                raise ValueError("Session sources changed during preparation.")
            if set(prepared.unavailable).intersection(prepared.probe_roles):
                raise ValueError(f"Population load failed: {prepared.unavailable}")
            with threadpool_limits(limits=1):
                stages = compute_stages(prepared, seed, reused) if reused else compute_stages(prepared, seed)
                for stage, fields in stages:
                    archives[stage] = f"{stage}.npz"
                    header = stage_metadata(prepared, seed, stage, identity, mouse_id)
                    if stage in reused:
                        header["reused_from_manifest"] = str(reuse_fixed_from)
                        log.write(f"{stage} reused from {reuse_fixed_from}\n")
                    results.save_result(directory / archives[stage], fields, header)
                    log.write(f"{stage} saved at {perf_counter() - started:.3f} seconds\n")
                    note = stage_observation(stage, fields)
                    if note:
                        observations.append(note)
                        log.write(note + "\n")
            elapsed = perf_counter() - started
            peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            archive_bytes = sum((directory / filename).stat().st_size for filename in archives.values())
            log.write(f"Finished in {elapsed:.3f} seconds; peak process RSS {peak_rss_kib} KiB; "
                      f"archives {archive_bytes} bytes\n")
            summary = (f"# PFC and HPC presentation figures\n\nSession {prepared.session_id}. "
                       f"Computed choice-aligned regional decoding, interregional prediction, movement "
                       f"and behavior-selected replays with run_session.py (batch entry run_batch.py).\n\n"
                       f"Good units: {unit_counts}. Eligible choices: {report['eligible_narrow']} regional, "
                       f"{report['eligible_wide']} paired. Seed {seed}. Elapsed {elapsed:.2f} s.\n")
            (directory / "summary.md").write_text(summary + "\n" + "\n\n".join(observations) + "\n", encoding="utf-8")
            manifest = results.complete_run(directory, kind="session", identity=identity,
                                            session_ids=[prepared.session_id], archives=archives)
        except Exception as error:
            log.write(f"{type(error).__name__}: {error}\n")
            raise
    return dict(report, status="computed", manifest=str(manifest), elapsed_s=elapsed,
                peak_rss_kib=peak_rss_kib, archive_bytes=archive_bytes)


def reviewed_stages(manifest: Path, inspection: preparation.PreparedChoices, seed: int) -> dict:
    """Read unchanged replay/movement fields from a compatible, completed session manifest."""
    if manifest not in results.discover_runs(inspection.session.session_root / "analysis_runs"):
        raise ValueError("Expected a completed manifest from this session.")
    data = json.loads(manifest.read_text())
    if data["kind"] != "session" or data["session_ids"] != [inspection.session_id]:
        raise ValueError("The selected manifest is incompatible with this session.")
    reused = {}
    for stage in ("replays", "movement"):
        fields, meta = results.load_result(manifest.parent / data["archives"][stage])
        if (meta["sources"] != inspection.meta["sources"] or meta["probe_roles"] != inspection.probe_roles
                or meta["identity"] != data["identity"] or meta["seed"] != seed):
            raise ValueError(f"{stage} source identities, probe roles, run identity or seed have changed.")
        reused[stage] = fields
    return reused


def compute_stages(prepared: preparation.PreparedChoices, seed: int,
                   reused: dict | None = None) -> Iterator[tuple[str, dict]]:
    """Yield four stage names and result fields using one prepared session and draw seed.

    Regional arrays use choice/bin/unit axes; prediction and movement share the
    100-ms four-second counts. Missing optional speed leaves neural CCA available.
    """
    from .decoding import compute_decoding
    from .movement import compute_movement
    from .prediction import compute_prediction
    from .replays import compute_replays

    yield "regional", compute_decoding(prepared)
    yield "replays", reused["replays"] if reused else compute_replays(prepared)
    yield "prediction", compute_prediction(prepared, seed=seed)
    if reused:
        yield "movement", reused["movement"]
        return
    try:
        speed_report = session_speed.run_session_speed(prepared.session.metadata_path)
        speed = session_speed.load_speed_result(speed_report.result_path)
    except (ValueError, OSError, KeyError) as error:
        movement = compute_movement(prepared, [], [], speed_reason=str(error))
    else:
        # Ambiguous omitted records have no exact UTC. Their retained neighbors
        # bound the interval across which duration weighting cannot be resolved.
        intervals = []
        for row in speed.meta.get("rejected_speed_records", []):
            positions = np.searchsorted(speed.source_rows, [row["previous_retained_row"], row["next_retained_row"]])
            intervals.append(speed.utc_s[positions])
        movement = compute_movement(prepared, speed.utc_s, speed.speed_mm_s,
                                    invalid_speed_intervals_s=intervals)
        movement["speed_configuration_id"] = speed_report.configuration_id
    yield "movement", movement


def new_run_directory(root: Path) -> Path:
    """Create a fresh timestamped run below root Path and return its absolute Path."""
    directory = root.resolve() / ("presentation_figures_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def session_identity(prepared: preparation.PreparedChoices, seed: int) -> str:
    """Hash source identities, scientific implementation, population settings and seed."""
    scientific = hashlib.sha256()
    package = Path(__file__).parent
    for name in ("preparation.py", "decoding.py", "replays.py", "prediction.py", "movement.py", "run_session.py"):
        scientific.update((package / name).read_bytes())
    payload = dict(sources=prepared.meta["sources"], roles=prepared.probe_roles, seed=seed,
                   code=scientific.hexdigest(), version=results.ANALYSIS_VERSION)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stage_metadata(prepared: preparation.PreparedChoices, seed: int, stage: str, identity: str, mouse_id: str) -> dict:
    """Create an NPZ metadata dictionary with source-row identities and physical units."""
    module = "decoding" if stage == "regional" else stage
    return dict(producer=f"presentation_figures.{module}.compute_{module}", seed=seed, identity=identity,
                session_id=prepared.session_id, subject_id=mouse_id,
                time_origin_utc_s=prepared.time_origin_s, probe_roles=prepared.probe_roles,
                unit_ids=prepared.unit_ids, coverage_s=prepared.coverage_s, sources=prepared.meta["sources"],
                trial_audit=prepared.meta["trial_audit"], trial_ids=prepared.trials.cur_trial.to_numpy(),
                block_ids=prepared.trials.cur_block.to_numpy(), fold_ids=prepared.fold_ids,
                unavailable=prepared.unavailable,
                parameters=scientific_parameters(),
                axes={"counts": ["choice", "bin", "unit"], "class_probabilities": "one row per choice/bin",
                      "model_sse": ["fit", "target pool position"], "baseline_sse": ["fit", "target pool position"]},
                units={"choice_time": "UTC seconds", "elapsed": "seconds from first trial start",
                       "accuracy": "fraction", "probability": "dimensionless", "r2": "dimensionless",
                       "speed": "mm/s", "counts": "spikes/bin", "rates": "Hz", "sse": "(spikes/bin)^2"})


def scientific_parameters() -> dict:
    """Return the shared archive settings including bin widths in ms and dimensionless PC limits."""
    from .decoding import MAX_REGIONAL_PCS
    from .prediction import PANEL_D_PCS
    return dict(regional_widths_ms=[100, 500], wide_width_ms=100, alpha=0.1, draws=5,
                max_regional_pcs=MAX_REGIONAL_PCS, panel_d_pcs=PANEL_D_PCS)


def stage_observation(stage: str, fields: dict) -> str:
    """Summarize a saved stage's held-out estimates or replay support in human-readable units."""
    if stage == "regional" and "scores" in fields:
        overall = fields["scores"].query(
            "period == 'full' and group == 'overall' and regime == 'all' and n_pcs == 0 and width_ms == 500")
        labels = [f"{r.region} {r.target} {r.balanced_accuracy:.1%}" if np.isfinite(r.balanced_accuracy)
                  else f"{r.region} {r.target} unavailable ({r.reason})" for r in overall.itertuples()]
        return "Held-out choice-level balanced accuracy at 500 ms: " + ", ".join(labels) + "."
    if stage == "replays" and "episodes" in fields:
        return "Behavior-selected excerpts: " + ", ".join(
            f"{r.motif} {r.status}" + (f" ({r.reason})" if r.reason else "")
            for r in fields["episodes"].itertuples()) + "."
    if stage == "prediction" and "curves" in fields:
        pool = fields["source_pool_size"]
        rows = fields["curves"].query("panel == 'A' and count == @pool")
        return f"Same-bin ridge prediction at {pool} source units: " + ", ".join(
            f"{r.direction} R-squared {r.mean_r2:.3f} ({r.n_draws}/5 draws)" if np.isfinite(r.mean_r2)
            else f"{r.direction} unavailable ({r.reason})" for r in rows.itertuples()) + "."
    if stage == "movement" and "block_scores" in fields:
        scores = fields["block_scores"].groupby("metric").correlation.agg(["median", "count"])
        return "Median held-out block correlations: " + ", ".join(
            f"{k} {r['median']:.3f} ({int(r['count'])} blocks)" if r["count"] else f"{k} unavailable"
            for k, r in scores.iterrows()) + ". " + fields.get("speed_reason", "")
    return ""


def main(argv: list[str] | None = None) -> int:
    """Run the session CLI from string arguments and return an integer exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-metadata", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--reuse-fixed-from", type=Path, metavar="MANIFEST",
                        help="Reuse reviewed, scientifically unchanged replay and movement fits from this session.")
    args = parser.parse_args(argv)
    report = run_session(args.session_metadata, seed=args.seed, dry_run=args.dry_run, rerun=args.rerun,
                         reuse_fixed_from=args.reuse_fixed_from)
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "excluded" else 0


if __name__ == "__main__":
    raise SystemExit(main())
