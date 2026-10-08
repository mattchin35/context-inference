"""Single-session command boundary for inter-regional neural regression."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Mapping

import matplotlib
import numpy as np
import pandas as pd
import pynapple
import scipy
import sklearn
import statsmodels

from src.neural_analysis import session_metadata
from src.neural_analysis.spike_behavior import loading as spike_loading
from src.neural_analysis.spike_behavior.trials import make_trial_type_masks

from . import persistence
from .configuration import (
    ANALYSIS_VERSION,
    COVERAGE_ASSUMPTION_VERSION,
    RESULT_SCHEMA_VERSION,
    InterregionalAnalysisConfig,
    RegionalPopulationConfig,
    ResolvedRegionalPopulation,
    RunOptions,
    configuration_to_dict,
    load_interregional_config,
    validate_resolved_populations,
)
from .pipeline import (
    PreparedInterregionalSession,
    fit_cross_validation_pcas,
    prepare_interregional_session,
    run_linear_cross_validation,
)
from .plotting import save_standard_regression_figures
from .records import (
    SCIENTIFIC_EXCLUSION_REASON_ORDER,
    InterregionalResults,
    default_units_and_axes,
    make_empty_result_tables,
    result_table_from_rows,
    validate_fold_score_key_grid,
    validate_interregional_results,
)


@dataclass(frozen=True)
class SessionPlan:
    """Resolved metadata-only execution plan for one session.

    Input sizes and ``input_sizes`` values are bytes. Paths are absolute
    filesystem coordinates. No large scientific array is loaded by this record.
    """

    config_path: Path
    config: InterregionalAnalysisConfig
    run_options: RunOptions
    session_id: str
    session_root: Path
    output_root: Path
    resolved_populations: tuple[ResolvedRegionalPopulation, ResolvedRegionalPopulation]
    input_files: Mapping[str, Path]
    input_sizes: Mapping[str, int]
    runtime_versions: Mapping[str, str]
    git_head: str
    code_dirty_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionRunReport:
    """Small session-run status returned to single and batch callers."""

    session_id: str
    status: str
    run_path: Path | None
    run_fingerprint: str | None
    input_bytes: int
    error: str | None


def collect_runtime_versions() -> dict[str, str]:
    """Return exact numerical runtime versions entering the run fingerprint."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pynapple": pynapple.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": importlib_metadata.version("statsmodels"),
        "matplotlib": matplotlib.__version__,
    }


def _repository_root() -> Path:
    """Return the checked-out project root containing this module."""
    return Path(__file__).resolve().parents[3]


def _git_identity(repository_root: Path) -> tuple[str, tuple[str, ...]]:
    """Return Git HEAD and code-relevant dirty paths without changing Git state."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lines = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty: list[str] = []
    for line in lines:
        if len(line) < 4:
            continue
        status, path = line[:2], line[3:]
        if status != "??" or (
            path.startswith("src/neural_analysis/") and path.endswith(".py")
        ):
            dirty.append(path)
    return head, tuple(sorted(dirty))


def _normalized_quality(series: pd.Series) -> pd.Series:
    """Normalize cluster-quality labels to lowercase strings."""
    return series.fillna("").astype(str).str.strip().str.lower()


def _resolve_population(
    selection: RegionalPopulationConfig,
    probe: session_metadata.ResolvedProbeSources,
) -> ResolvedRegionalPopulation:
    """Resolve selected channels and qualified sorter clusters for one role."""
    if probe.sorter_directory is None or probe.cluster_info_file is None:
        raise ValueError(f"Probe {probe.probe_id!r} has no sorter source.")
    if selection.channel_source == "explicit":
        channels = np.asarray(selection.selected_channels, dtype=np.int64)
    else:
        if probe.channel_quality_file is None:
            raise ValueError(f"Probe {probe.probe_id!r} has no channel-quality source.")
        quality = spike_loading.load_channel_quality(probe.channel_quality_file)
        channels = spike_loading.select_channels_from_quality(
            quality,
            require_inside_brain=selection.require_inside_brain,
            labels=selection.channel_quality_labels,
        )
    if probe.unit_channels is not None:
        channels = np.intersect1d(
            channels, np.asarray(probe.unit_channels, dtype=np.int64), assume_unique=False
        )
    channels = np.unique(channels.astype(np.int64))
    if channels.size == 0:
        raise ValueError(f"Resolved {selection.role} channel selection is empty.")
    cluster_info = pd.read_csv(probe.cluster_info_file, sep="\t")
    required = {"cluster_id", "ch", selection.unit_quality_column}
    missing = required - set(cluster_info.columns)
    if missing:
        raise ValueError(f"cluster_info is missing required columns: {sorted(missing)}")
    labels = {value.lower() for value in selection.unit_quality_labels}
    selected = cluster_info.loc[
        cluster_info["ch"].isin(channels)
        & _normalized_quality(cluster_info[selection.unit_quality_column]).isin(labels),
        "cluster_id",
    ]
    cluster_ids = tuple(sorted({int(value) for value in selected}))
    if not cluster_ids:
        raise ValueError(f"Resolved {selection.role} unit selection is empty.")
    return ResolvedRegionalPopulation(
        role=selection.role,
        probe_id=selection.probe_id,
        channel_source=selection.channel_source,
        selected_channels=tuple(int(value) for value in channels),
        require_inside_brain=selection.require_inside_brain,
        channel_quality_labels=selection.channel_quality_labels,
        unit_quality_column=selection.unit_quality_column,
        unit_quality_labels=selection.unit_quality_labels,
        cluster_ids=cluster_ids,
        unit_ids=tuple(f"{selection.probe_id}:{value}" for value in cluster_ids),
    )


def _input_files(
    resolved: session_metadata.ResolvedSession,
    selections: tuple[RegionalPopulationConfig, RegionalPopulationConfig],
) -> dict[str, Path]:
    """Return every regular file consumed by preparation and population resolution."""
    if resolved.behavior.trial_table_file is None:
        raise ValueError("Session metadata has no behavior trial table.")
    files = {
        "session_metadata": resolved.metadata_path,
        "trial_table": resolved.behavior.trial_table_file,
    }
    probes = {probe.probe_id: probe for probe in resolved.probes}
    for selection in selections:
        if selection.probe_id not in probes:
            raise ValueError(f"Unknown configured probe {selection.probe_id!r}.")
        probe = probes[selection.probe_id]
        required = {
            f"{selection.role.lower()}_aligned_spikes": probe.alignment_file,
            f"{selection.role.lower()}_spike_clusters": probe.spike_clusters_file,
            f"{selection.role.lower()}_cluster_info": probe.cluster_info_file,
        }
        if selection.channel_source == "metadata_quality":
            required[f"{selection.role.lower()}_channel_quality"] = (
                probe.channel_quality_file
            )
        for role, path in required.items():
            if path is None or not path.is_file():
                raise ValueError(f"Required input {role!r} is missing: {path}")
            files[role] = path
    return files


def plan_single_session(
    config_path: Path | str,
    *,
    repository_root: Path | str | None = None,
) -> SessionPlan:
    """Resolve and size one session without hashing or loading large spike arrays."""
    config_file = Path(config_path).resolve(strict=True)
    config, run_options = load_interregional_config(config_file)
    metadata = session_metadata.load_session_metadata(config.session_metadata_path)
    resolved = session_metadata.resolve_session_metadata(
        metadata, config.session_metadata_path
    )
    pfc_probe = session_metadata.resolve_probe_sources(
        resolved, config.pfc_population.probe_id
    )
    hpc_probe = session_metadata.resolve_probe_sources(
        resolved, config.hpc_population.probe_id
    )
    pfc = _resolve_population(config.pfc_population, pfc_probe)
    hpc = _resolve_population(config.hpc_population, hpc_probe)
    validate_resolved_populations(pfc, hpc)
    inputs = _input_files(
        resolved, (config.pfc_population, config.hpc_population)
    )
    output_root = (
        resolved.session_root / "analysis_runs"
        if run_options.output_root is None
        else Path(run_options.output_root)
    ).resolve()
    repository = (
        _repository_root()
        if repository_root is None
        else Path(repository_root).resolve(strict=True)
    )
    if output_root == repository or output_root.is_relative_to(repository):
        raise ValueError("Run output_root must be outside the source repository.")
    git_head, dirty = _git_identity(repository)
    return SessionPlan(
        config_path=config_file,
        config=config,
        run_options=run_options,
        session_id=resolved.session_id,
        session_root=resolved.session_root,
        output_root=output_root,
        resolved_populations=(pfc, hpc),
        input_files=inputs,
        input_sizes={role: path.stat().st_size for role, path in inputs.items()},
        runtime_versions=collect_runtime_versions(),
        git_head=git_head,
        code_dirty_paths=dirty,
    )


def _load_spike_group(
    plan: SessionPlan, population: ResolvedRegionalPopulation, role_prefix: str
):
    """Load aligned seconds and matching cluster IDs into one Pynapple TsGroup."""
    spike_times = spike_loading.load_aligned_spikes(
        plan.input_files[f"{role_prefix}_aligned_spikes"]
    )
    spike_clusters = np.load(
        plan.input_files[f"{role_prefix}_spike_clusters"], allow_pickle=False
    ).astype(np.int64)
    spike_loading.validate_aligned_spike_inputs(spike_times, spike_clusters)
    return spike_loading.build_spike_tsgroup(
        spike_times, spike_clusters, np.asarray(population.cluster_ids, dtype=np.int64)
    )


def _preparation_tables(
    prepared: PreparedInterregionalSession,
    trial_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize fold and scientific-membership provenance from prepared masks."""
    fold_rows = []
    for trial_row, (label, block_json, fold_id) in enumerate(
        zip(
            prepared.fold_assignment.original_index_labels,
            prepared.fold_assignment.block_values_json,
            prepared.fold_assignment.fold_ids,
            strict=True,
        )
    ):
        present = fold_id is not None
        fold_rows.append(
            {
                "session_id": prepared.session_id,
                "trial_row": trial_row,
                "original_index_repr": label,
                "block_value_json": block_json,
                "block_present": present,
                "fold_id": fold_id,
                "status": "ok" if present else "not_applicable",
                "reason": "" if present else "missing_block",
            }
        )
    raw_conditions = (
        make_trial_type_masks(trial_df)
        if any(value != "all" for value in prepared.config.filters.conditions)
        else {}
    )
    membership_rows = []
    masks = prepared.trial_masks
    for condition in prepared.config.filters.conditions:
        condition_match = (
            np.ones(len(trial_df), dtype=bool)
            if condition == "all"
            else np.asarray(raw_conditions[condition], dtype=bool)
        )
        for trial_row in range(len(trial_df)):
            reason_flags = {
                "invalid_reward_status": not masks.reward_status_valid[trial_row],
                "invalid_alignment": not masks.alignment_valid[trial_row],
                "choice_filter_mismatch": not masks.choice_match[trial_row],
                "context_filter_mismatch": not masks.context_match[trial_row],
                "user_excluded": not masks.user_included[trial_row],
            }
            reasons = [
                reason
                for reason in SCIENTIFIC_EXCLUSION_REASON_ORDER
                if reason_flags[reason]
            ]
            included = bool(masks.condition_masks[condition][trial_row])
            membership_rows.append(
                {
                    "session_id": prepared.session_id,
                    "trial_row": trial_row,
                    "alignment": prepared.config.alignment,
                    "condition": condition,
                    "original_index_repr": masks.original_index_labels[trial_row],
                    "reward_status_valid": bool(masks.reward_status_valid[trial_row]),
                    "alignment_valid": bool(masks.alignment_valid[trial_row]),
                    "choice_match": bool(masks.choice_match[trial_row]),
                    "context_match": bool(masks.context_match[trial_row]),
                    "user_included": bool(masks.user_included[trial_row]),
                    "block_present": bool(masks.block_present[trial_row]),
                    "condition_match": bool(condition_match[trial_row]),
                    "scientific_eligible": bool(masks.scientific_eligible[trial_row]),
                    "condition_included": included,
                    "cv_included": included and bool(masks.block_present[trial_row]),
                    "scientific_exclusion_reasons_json": json.dumps(
                        reasons, separators=(",", ":")
                    ),
                }
            )
    return (
        result_table_from_rows("fold_assignments", fold_rows),
        result_table_from_rows("trial_membership", membership_rows),
    )


def compute_single_session(plan: SessionPlan) -> InterregionalResults:
    """Load one resolved session and compute all currently implemented stages."""
    trial_df = pd.read_csv(plan.input_files["trial_table"])
    pfc, hpc = plan.resolved_populations
    prepared = prepare_interregional_session(
        plan.session_id,
        _load_spike_group(plan, pfc, "pfc"),
        _load_spike_group(plan, hpc, "hpc"),
        pfc,
        hpc,
        trial_df,
        plan.config,
    )
    fold_pcas = None
    tables = make_empty_result_tables()
    if "pcs" in plan.config.representations:
        fold_pcas, tables["pca_fits"] = fit_cross_validation_pcas(prepared)
    fold_scores, target_summaries, population_summaries = (
        run_linear_cross_validation(prepared, fold_pcas=fold_pcas)
    )
    tables["fold_scores"] = fold_scores
    tables["target_summaries"] = target_summaries
    tables["population_summaries"] = population_summaries
    tables["fold_assignments"], tables["trial_membership"] = _preparation_tables(
        prepared, trial_df
    )
    validate_fold_score_key_grid(
        fold_scores, plan.session_id, plan.config, plan.resolved_populations
    )
    result = InterregionalResults(
        schema_version=RESULT_SCHEMA_VERSION,
        analysis_version=ANALYSIS_VERSION,
        coverage_assumption_version=COVERAGE_ASSUMPTION_VERSION,
        configuration=configuration_to_dict(plan.config, RunOptions()),
        session_id=plan.session_id,
        resolved_populations=plan.resolved_populations,
        whole_bin_edges_s=prepared.pfc_counts.bin_edges_s,
        units_and_axes=default_units_and_axes(),
        randomness_used=False,
        random_seed=None,
        **tables,
    )
    validate_interregional_results(result, plan.config)
    return result


def _find_reusable_run(plan: SessionPlan, fingerprint: str) -> Path | None:
    """Return one validated matching final run without mutating it."""
    for run_path in persistence.discover_finalized_runs(plan.output_root):
        try:
            manifest = json.loads(
                (run_path / "input_manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("run_fingerprint") != fingerprint:
                continue
            persistence.load_interregional_result(
                run_path / "result.pkl",
                plan.config,
                trusted_run_directory=run_path,
            )
        except (OSError, ValueError, AttributeError, json.JSONDecodeError):
            continue
        return run_path
    return None


def _write_summary(path: Path, result: InterregionalResults) -> None:
    """Write a concise scientific summary with coverage and causal caveats."""
    unavailable = int(result.fold_scores["status"].ne("ok").sum())
    unavailable_reasons = (
        result.fold_scores.loc[result.fold_scores["reason"].ne(""), "reason"]
        .value_counts()
        .sort_index()
    )
    lines = [
        "# Inter-regional neural regression run",
        "",
        "Goal: quantify held-out predictive improvement between PFC and HPC.",
        "",
        f"Session: `{result.session_id}`",
        "",
        "Scripts: `run_session.py`, `run_batch.py`",
        "",
        f"Output directory: `{path.parent}`",
        "",
        f"Unavailable fold rows: {unavailable}",
        "",
        "Warnings: none captured.",
        "",
        "## Scientific configuration",
        "",
        "```json",
        json.dumps(result.configuration, sort_keys=True, indent=2),
        "```",
        "",
        "## Primary held-out results",
        "",
    ]
    primary = result.population_summaries.loc[
        result.population_summaries["metric_name"].eq("delta_r2")
    ]
    if primary.empty:
        lines.append("No complete incremental CV R-squared population summary was available.")
    else:
        for row in primary.itertuples(index=False):
            if row.status == "ok":
                lines.append(
                    f"- {row.direction}, {row.condition}, {row.window}, "
                    f"{row.representation}/{row.model_family}: {row.n_targets} targets; "
                    f"median={float(row.median):.6g}, IQR=[{float(row.q25):.6g}, "
                    f"{float(row.q75):.6g}] (held-out CV)."
                )
            else:
                lines.append(
                    f"- {row.direction}, {row.condition}, {row.window}, "
                    f"{row.representation}/{row.model_family}: unavailable ({row.reason})."
                )
    lines.extend(["", "## Unavailable reasons", ""])
    if unavailable_reasons.empty:
        lines.append("None.")
    else:
        lines.extend(
            f"- `{reason}`: {int(count)}"
            for reason, count in unavailable_reasons.items()
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Reported cross-validated increments are predictive, not proof of causality or "
            "anatomical direction. Spike coverage is assumed complete under the saved "
            "coverage-assumption version. No significance claim is made from these descriptive "
            "effect summaries.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_initial_artifacts(
    working: Path,
    plan: SessionPlan,
    manifest: Mapping[str, object],
    preflight_timings: Mapping[str, float],
) -> None:
    """Write reproducibility metadata and exact runner copies before computation."""
    config_payload = configuration_to_dict(
        plan.config, RunOptions(output_root=plan.output_root)
    )
    (working / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (working / "input_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    log_lines = [
        f"session={plan.session_id} analysis_version={ANALYSIS_VERSION}",
        *(
            f"stage={stage} event=end elapsed_seconds={preflight_timings[stage]:.9f}"
            for stage in ("input_validation", "input_hashing")
        ),
    ]
    (working / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), working / "run_session.py")
    shutil.copy2(Path(__file__).with_name("run_batch.py"), working / "run_batch.py")
    (working / "figures").mkdir()


def run_single_session(
    config_path: Path | str,
    *,
    command: str,
    rerun: bool = False,
    repository_root: Path | str | None = None,
) -> SessionRunReport:
    """Plan or execute one immutable inter-regional session run.

    ``command`` is ``"dry-run"`` or ``"new"``. Dry-run validates and reports
    input sizes in bytes without hashing, fitting, or creating output paths.
    """
    if command not in {"dry-run", "new"}:
        raise ValueError("command must be 'dry-run' or 'new'.")
    validation_started = time.perf_counter()
    plan = plan_single_session(config_path, repository_root=repository_root)
    validation_elapsed = time.perf_counter() - validation_started
    input_bytes = int(sum(plan.input_sizes.values()))
    if command == "dry-run":
        return SessionRunReport(
            plan.session_id, "planned", None, None, input_bytes, None
        )
    if plan.code_dirty_paths:
        raise ValueError(
            "new requires clean tracked code and no untracked Python under "
            f"src/neural_analysis: {list(plan.code_dirty_paths)}"
        )
    hashing_started = time.perf_counter()
    file_entries = persistence.hash_input_files(plan.input_files)
    hashing_elapsed = time.perf_counter() - hashing_started
    fingerprint = persistence.run_fingerprint(
        plan.config,
        session_id=plan.session_id,
        resolved_populations=plan.resolved_populations,
        files=file_entries,
        git_head=plan.git_head,
        runtime_versions=plan.runtime_versions,
    )
    if not rerun:
        existing = _find_reusable_run(plan, fingerprint)
        if existing is not None:
            return SessionRunReport(
                plan.session_id, "reused", existing, fingerprint, input_bytes, None
            )
    repository = (
        _repository_root()
        if repository_root is None
        else Path(repository_root).resolve(strict=True)
    )
    working = persistence.create_working_run_directory(
        plan.output_root, repository_root=repository
    )
    manifest = persistence.build_input_manifest(
        run_fingerprint=fingerprint,
        session_id=plan.session_id,
        git_head=plan.git_head,
        runtime_versions=plan.runtime_versions,
        resolved_populations=plan.resolved_populations,
        files=file_entries,
    )
    _write_initial_artifacts(
        working,
        plan,
        manifest,
        {
            "input_validation": validation_elapsed,
            "input_hashing": hashing_elapsed,
        },
    )
    stage = "preparation"
    started = time.perf_counter()
    try:
        compute_started = time.perf_counter()
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write("stage=preparation event=start\n")
            stream.write("stage=ols_cv event=start\n")
        result = compute_single_session(plan)
        compute_elapsed = time.perf_counter() - compute_started
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(
                f"stage=ols_cv event=end elapsed_seconds={compute_elapsed:.9f}\n"
            )
            stream.write(
                f"stage=preparation event=end elapsed_seconds={compute_elapsed:.9f}\n"
            )
        stage = "persistence"
        persistence_started = time.perf_counter()
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write("stage=persistence event=start\n")
        persistence.save_interregional_result(
            result, working / "result.pkl", plan.config
        )
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(
                "stage=persistence event=end "
                f"elapsed_seconds={time.perf_counter() - persistence_started:.9f}\n"
            )
        stage = "figures"
        figures_started = time.perf_counter()
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write("stage=figures event=start\n")
        save_standard_regression_figures(result, working / "figures")
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(
                "stage=figures event=end "
                f"elapsed_seconds={time.perf_counter() - figures_started:.9f}\n"
            )
        stage = "summary"
        summary_started = time.perf_counter()
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write("stage=summary event=start\n")
        _write_summary(working / "summary.md", result)
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(
                "stage=summary event=end "
                f"elapsed_seconds={time.perf_counter() - summary_started:.9f}\n"
            )
        elapsed = time.perf_counter() - started
        with (working / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"completed elapsed_seconds={elapsed:.9f}\n")
        persistence.validate_complete_run_artifacts(
            working, plan.config, fingerprint
        )
        final = persistence.finalize_run_directory(working, fingerprint)
        return SessionRunReport(
            plan.session_id, "completed", final, fingerprint, input_bytes, None
        )
    except Exception as error:
        persistence.record_run_failure(working, stage=stage, error=error)
        return SessionRunReport(
            plan.session_id,
            "failed",
            working,
            fingerprint,
            input_bytes,
            f"{type(error).__name__}: {error}",
        )


def _parser() -> argparse.ArgumentParser:
    """Build the documented dry-run/new command-line parser."""
    parser = argparse.ArgumentParser(prog="interregional-regression-session")
    commands = parser.add_subparsers(dest="command", required=True)
    dry_run = commands.add_parser("dry-run")
    dry_run.add_argument("--config", required=True)
    new = commands.add_parser("new")
    new.add_argument("--config", required=True)
    new.add_argument("--rerun", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the single-session CLI and return a conventional process status."""
    args = _parser().parse_args(argv)
    report = run_single_session(
        args.config,
        command=args.command,
        rerun=bool(getattr(args, "rerun", False)),
    )
    print(report)
    return 1 if report.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
