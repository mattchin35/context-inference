"""Prepare, inspect, and describe one session's neural-analysis commands.

This module is intentionally non-computational. It writes editable JSON only
for ``init`` and otherwise reads small control files plus filesystem metadata.
It never loads scientific arrays, creates run directories, or submits jobs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import sys
from types import MappingProxyType
from typing import Mapping, Sequence

from src.neural_analysis import session_metadata, session_metadata_cli
from src.neural_analysis.interregional.configuration import load_interregional_config
from src.neural_analysis.task_decoding.config import (
    CONDITION_IDENTIFIERS,
    TARGET_IDENTIFIERS,
    load_task_decoding_config,
)
from src.neural_analysis.task_decoding.pooled.config import load_pooled_decoding_config


TASK_CONFIG_FILENAME = "task_decoding_config.json"
POOLED_CONFIG_FILENAME = "pooled_task_decoding_config.json"
INTERREGIONAL_CONFIG_FILENAME = "interregional_regression_config.json"
ANALYSIS_NAMES = (
    "task_decoding",
    "condition_generalization",
    "interregional_regression",
    "spike_phase_ppc",
)
TRAINING_WINDOW_NAMES = ("pre_choice", "post_choice", "full")


@dataclass(frozen=True)
class ScaffoldResult:
    """Files considered by one safe scaffolding request.

    Parameters
    ----------
    created, preserved : tuple[pathlib.Path, ...]
        Absolute paths written by this call and pre-existing paths left byte-for-byte
        unchanged. Paths have no physical units or array axes.
    """

    created: tuple[Path, ...]
    preserved: tuple[Path, ...]


@dataclass(frozen=True)
class AnalysisReadiness:
    """Readiness of one analysis without executing its scientific pipeline.

    Parameters
    ----------
    name : str
        Stable analysis-family identifier.
    ready : bool
        Whether ordinary control files and named input paths are present and valid.
    blockers : tuple[str, ...]
        Ordered human-readable validation failures; empty exactly when ready.
    """

    name: str
    ready: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class SessionReadinessReport:
    """Preparation status for one session.

    Parameters
    ----------
    session_root : pathlib.Path
        Absolute session directory.
    analyses : mapping[str, AnalysisReadiness]
        Fixed-order mapping over the four supported analysis families.
    """

    session_root: Path
    analyses: Mapping[str, AnalysisReadiness]

    @property
    def ready(self) -> bool:
        """Return whether all four requested analysis families are ready."""
        return all(item.ready for item in self.analyses.values())


@dataclass(frozen=True)
class AnalysisCommands:
    """Copyable commands for a later user-controlled analysis invocation.

    Parameters
    ----------
    local_preflight : str
        Existing pipeline validation/dry-run command.
    local_preview, local_final : str or None
        Existing local execution commands. ``local_final`` is PPC-only.
    cluster_preview, cluster_final : str or None
        Existing Slurm submission commands. ``cluster_final`` is PPC-only.
    preparation : str or None
        One non-computational prerequisite command. PPC uses this to adopt the
        rsynced source into independent preview and final caches.

    Notes
    -----
    These strings are returned as text only. Constructing this record does not
    execute a shell or modify the filesystem.
    """

    local_preflight: str
    local_preview: str | None
    local_final: str | None
    cluster_preview: str | None
    cluster_final: str | None
    preparation: str | None = None


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> bool:
    """Write one ASCII JSON object only when the target does not exist.

    Parameters
    ----------
    path : pathlib.Path
        Target control-file path.
    payload : mapping[str, object]
        JSON-compatible scalar/container values without scientific arrays.

    Returns
    -------
    bool
        ``True`` when the file was created and ``False`` when it was preserved.
    """
    encoded = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n"
    try:
        with path.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError:
        return False
    return True


def _region_payload(region: str, probe_id: str) -> dict[str, object]:
    """Return one editable task-decoding region selection."""
    return {
        "region": region,
        "probe_id": probe_id,
        "channel_labels": ["good"],
        "require_inside_brain": True,
        "cluster_groups": ["good", "mua"],
        "channel_ids": [],
    }


def _task_payload(pfc_probe_id: str, hpc_probe_id: str) -> dict[str, object]:
    """Return the complete reviewed task-decoding template."""
    return {
        "session_metadata_path": "neural_session.json",
        "augmented_trial_path": "processed/augmented_trials.csv",
        "trial_feature_parameter_path": "processed/trial_feature_params.json",
        "pfc_region": _region_payload("PFC", pfc_probe_id),
        "hpc_region": _region_payload("HPC", hpc_probe_id),
        "alignment": "choice_time",
        "bin_width_ms": 100,
        "pfc_pc_count": 10,
        "hpc_pc_count": 10,
        "target_names": list(TARGET_IDENTIFIERS),
        "condition_names": list(CONDITION_IDENTIFIERS),
        "regularization_mode": "fixed",
        "outer_fold_count": 5,
        "inner_fold_count": 3,
        "trusted_utc_bounds": {},
        "output_root": "analysis_runs",
    }


def _interregional_population(probe_id: str) -> dict[str, object]:
    """Return one fast-first interregional population selection."""
    return {
        "probe_id": probe_id,
        "channel_source": "metadata_quality",
        "selected_channels": [],
        "require_inside_brain": True,
        "channel_quality_labels": ["good"],
        "unit_quality_column": "group",
        "unit_quality_labels": ["good", "mua"],
    }


def _interregional_payload(
    pfc_probe_id: str, hpc_probe_id: str
) -> dict[str, object]:
    """Return a portable fast-first OLS/linear-Granger configuration."""
    return {
        "config_schema_version": "1",
        "analysis_version": "interregional-regression-v1",
        "coverage_assumption_version": "implicit-complete-v1",
        "session_metadata_path": session_metadata.CANONICAL_FILENAME,
        "populations": {
            "PFC": _interregional_population(pfc_probe_id),
            "HPC": _interregional_population(hpc_probe_id),
        },
        "alignment": "choice_time",
        "windows": {"whole_start_s": -2.0, "split_s": 0.0, "whole_stop_s": 2.0},
        "prediction_windows": ["before", "after", "whole"],
        "temporal": {"bin_size_s": 0.1, "lag_bins": 1, "order_bins": 1},
        "filters": {
            "conditions": list(CONDITION_IDENTIFIERS),
            "choice": "all",
            "context": "all",
            "excluded_trial_rows": [],
        },
        "pca": {"pfc_components": 10, "hpc_components": 10},
        "representations": ["units", "pcs"],
        "analyses": ["ols_cv", "linear_granger"],
        "run": {"output_root": "analysis_runs"},
    }


def scaffold_session_analysis(
    session_root: Path | str,
    *,
    pfc_probe_id: str,
    hpc_probe_id: str,
    training_window_names: Sequence[str] = TRAINING_WINDOW_NAMES,
) -> ScaffoldResult:
    """Create missing editable control files without replacing existing files.

    Parameters
    ----------
    session_root : pathlib.Path or str
        Existing session directory. No data directories are created.
    pfc_probe_id, hpc_probe_id : str
        Exact metadata probe dictionary keys assigned to the anatomical roles.
    training_window_names : sequence of str
        Nonempty ordered subset of ``pre_choice``, ``post_choice``, and ``full``.

    Returns
    -------
    ScaffoldResult
        Absolute created and preserved control-file paths.

    Raises
    ------
    ValueError
        If the directory, probe roles, or pooled-window selection is invalid.
    """
    root = Path(session_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"session root is not a directory: {root}")
    if not pfc_probe_id or not hpc_probe_id or pfc_probe_id == hpc_probe_id:
        raise ValueError("PFC and HPC probe IDs must be distinct nonempty strings")
    windows = tuple(training_window_names)
    if not windows or len(set(windows)) != len(windows):
        raise ValueError("training windows must be nonempty and unique")
    if any(name not in TRAINING_WINDOW_NAMES for name in windows):
        raise ValueError(f"training windows must be selected from {TRAINING_WINDOW_NAMES}")
    if windows != tuple(name for name in TRAINING_WINDOW_NAMES if name in windows):
        raise ValueError("training windows must use canonical order")

    created: list[Path] = []
    preserved: list[Path] = []
    metadata_path = root / session_metadata.CANONICAL_FILENAME
    if metadata_path.exists():
        preserved.append(metadata_path)
    else:
        result = session_metadata_cli._create(root)
        if result != 0:  # pragma: no cover - the helper either succeeds or raises
            raise RuntimeError("session metadata scaffold failed")
        created.append(metadata_path)

    payloads = (
        (root / TASK_CONFIG_FILENAME, _task_payload(pfc_probe_id, hpc_probe_id)),
        (
            root / POOLED_CONFIG_FILENAME,
            {
                "base_config_path": TASK_CONFIG_FILENAME,
                "training_window_names": list(windows),
            },
        ),
        (
            root / INTERREGIONAL_CONFIG_FILENAME,
            _interregional_payload(pfc_probe_id, hpc_probe_id),
        ),
    )
    for path, payload in payloads:
        (created if _write_json_exclusive(path, payload) else preserved).append(path)
    return ScaffoldResult(tuple(path.resolve() for path in created), tuple(path.resolve() for path in preserved))


def _missing_probe_inputs(
    resolved: session_metadata.ResolvedSession, probe_id: str
) -> tuple[str, ...]:
    """Return missing aligned-spike inputs for one configured probe role."""
    try:
        probe = session_metadata.resolve_probe_sources(resolved, probe_id)
    except KeyError:
        return (f"unknown metadata probe ID: {probe_id}",)
    candidates = (
        ("alignment", probe.alignment_file, "file"),
        ("sorter", probe.sorter_directory, "directory"),
        ("quality", probe.channel_quality_file, "file"),
        ("spike_times", probe.spike_times_file, "file"),
        ("spike_clusters", probe.spike_clusters_file, "file"),
        ("cluster_info", probe.cluster_info_file, "file"),
    )
    missing = []
    for label, path, kind in candidates:
        exists = path is not None and (path.is_file() if kind == "file" else path.is_dir())
        if not exists:
            missing.append(f"probes[{probe_id}].{label}")
    return tuple(missing)


def _load_resolved_metadata(
    metadata_path: Path,
) -> tuple[session_metadata.ResolvedSession | None, tuple[str, ...]]:
    """Load one metadata sidecar and convert user errors to blockers."""
    try:
        metadata = session_metadata.load_session_metadata(metadata_path)
        return session_metadata.resolve_session_metadata(metadata, metadata_path), ()
    except (OSError, ValueError) as error:
        return None, (f"session metadata: {error}",)


def _configuration_blocker(loader: object, path: Path, label: str) -> tuple[str, ...]:
    """Run one small configuration loader and return any user-facing error."""
    if not path.is_file():
        return (f"missing {path.name}",)
    try:
        loader(path)  # type: ignore[operator]
    except (OSError, ValueError) as error:
        return (f"{label}: {error}",)
    return ()


def inspect_session_readiness(
    session_root: Path | str,
    *,
    pfc_probe_id: str,
    hpc_probe_id: str,
    ppc_probe_id: str,
    ppc_cache_directory: Path | str | None = None,
) -> SessionReadinessReport:
    """Inspect small control files and named paths without processing data.

    Parameters
    ----------
    session_root : pathlib.Path or str
        Existing session directory.
    pfc_probe_id, hpc_probe_id, ppc_probe_id : str
        Exact metadata probe IDs for PFC, HPC, and the PPC unit population.
    ppc_cache_directory : pathlib.Path, str, or None
        Existing PPC cache directory. ``None`` uses metadata ``cache``.

    Returns
    -------
    SessionReadinessReport
        Fixed-order per-analysis readiness. No scientific arrays are opened and
        no files or directories are written.
    """
    root = Path(session_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"session root is not a directory: {root}")
    metadata_path = root / session_metadata.CANONICAL_FILENAME
    resolved, metadata_blockers = _load_resolved_metadata(metadata_path)

    task_path = root / TASK_CONFIG_FILENAME
    task_blockers: list[str] = []
    if not task_path.is_file():
        task_blockers.append(f"missing {task_path.name}")
    else:
        try:
            task_config = load_task_decoding_config(task_path)
            if (
                task_config.pfc_region.probe_id != pfc_probe_id
                or task_config.hpc_region.probe_id != hpc_probe_id
            ):
                task_blockers.append(
                    "task decoding probe roles disagree with --pfc-probe/--hpc-probe"
                )
        except (OSError, ValueError) as error:
            task_blockers.append(f"task decoding configuration: {error}")
    if resolved is None:
        task_blockers.extend(metadata_blockers)
    else:
        task_blockers.extend(_missing_probe_inputs(resolved, pfc_probe_id))
        task_blockers.extend(_missing_probe_inputs(resolved, hpc_probe_id))

    pooled_blockers = list(task_blockers)
    pooled_blockers.extend(
        _configuration_blocker(
            load_pooled_decoding_config,
            root / POOLED_CONFIG_FILENAME,
            "condition-generalization configuration",
        )
    )

    interregional_path = root / INTERREGIONAL_CONFIG_FILENAME
    interregional_blockers: list[str] = []
    if not interregional_path.is_file():
        interregional_blockers.append(f"missing {interregional_path.name}")
    else:
        try:
            interregional_config, _ = load_interregional_config(interregional_path)
            if (
                interregional_config.pfc_population.probe_id != pfc_probe_id
                or interregional_config.hpc_population.probe_id != hpc_probe_id
            ):
                interregional_blockers.append(
                    "interregional probe roles disagree with --pfc-probe/--hpc-probe"
                )
        except (OSError, ValueError) as error:
            interregional_blockers.append(f"interregional configuration: {error}")
    if resolved is None:
        interregional_blockers.extend(metadata_blockers)
    else:
        interregional_blockers.extend(_missing_probe_inputs(resolved, pfc_probe_id))
        interregional_blockers.extend(_missing_probe_inputs(resolved, hpc_probe_id))

    ppc_blockers: list[str] = []
    if resolved is None:
        ppc_blockers.extend(metadata_blockers)
    else:
        try:
            session_metadata.resolve_probe_sources(resolved, ppc_probe_id)
        except KeyError:
            ppc_blockers.append(f"unknown metadata probe ID: {ppc_probe_id}")
        availability = session_metadata.validate_session_for_action(resolved, "spike-phase")
        ppc_blockers.extend(availability.missing_inputs)
        cache = (
            Path(ppc_cache_directory).resolve(strict=False)
            if ppc_cache_directory is not None
            else resolved.cache_directory
        )
        if cache is None or not cache.is_dir():
            ppc_blockers.append("PPC cache directory")
        elif cache.parent != (root / "processed").resolve(strict=False):
            ppc_blockers.append("PPC cache must be a direct child of session/processed")
        elif cache.name == "lfp_summary_cache":
            ppc_blockers.append("legacy lfp_summary_cache is protected; use a new cache")

    blocker_groups = (
        ("task_decoding", task_blockers),
        ("condition_generalization", pooled_blockers),
        ("interregional_regression", interregional_blockers),
        ("spike_phase_ppc", ppc_blockers),
    )
    analyses = {
        name: AnalysisReadiness(name, not blockers, tuple(dict.fromkeys(blockers)))
        for name, blockers in blocker_groups
    }
    return SessionReadinessReport(root, MappingProxyType(analyses))


def _module_command(module: str, *arguments: str | Path) -> str:
    """Return one shell-safe ``uv run python -m`` command string."""
    return shlex.join(["uv", "run", "python", "-m", module, *(str(item) for item in arguments)])


def build_analysis_commands(
    session_root: Path | str,
    *,
    ppc_probe_id: str,
    ppc_cache_directory: Path | str,
) -> Mapping[str, AnalysisCommands]:
    """Build copyable commands without running, submitting, or writing anything.

    Parameters
    ----------
    session_root : pathlib.Path or str
        Session directory containing the scaffolded control files.
    ppc_probe_id : str
        Exact metadata probe key used as the PPC unit population.
    ppc_cache_directory : pathlib.Path or str
        Existing explicit PPC cache directory.

    Returns
    -------
    mapping[str, AnalysisCommands]
        Fixed-order analysis commands as shell-escaped text.
    """
    root = Path(session_root).resolve(strict=True)
    metadata_path = root / session_metadata.CANONICAL_FILENAME
    task_config = root / TASK_CONFIG_FILENAME
    pooled_config = root / POOLED_CONFIG_FILENAME
    interregional_config = root / INTERREGIONAL_CONFIG_FILENAME
    source_cache = Path(ppc_cache_directory).resolve(strict=False)
    preview_cache = root / "processed" / "ppc_prerequisites_preview"
    final_cache = root / "processed" / "ppc_prerequisites_final"

    task_module = "src.neural_analysis.task_decoding.run_session"
    pooled_module = "src.neural_analysis.task_decoding.pooled.run_session"
    interregional_module = "src.neural_analysis.interregional.run_session"
    ppc_module = "src.neural_analysis.lfp_spike_phase_launcher"
    def ppc_base(cache_directory: Path) -> tuple[str | Path, ...]:
        """Return launcher arguments for one independent prerequisite cache."""
        return (
            "new",
            "--session-metadata",
            metadata_path,
            "--probe",
            ppc_probe_id,
            "--cache-directory",
            cache_directory,
        )

    preview_base = ppc_base(preview_cache)
    final_base = ppc_base(final_cache)

    commands = {
        "task_decoding": AnalysisCommands(
            _module_command(task_module, "dry-run", "--config", task_config),
            _module_command(task_module, "new", "--config", task_config),
            None,
            shlex.join(
                [
                    "bash",
                    "src/shell_scripts/task_variable_decoding_slurm.sh",
                    "submit-new",
                    "--config",
                    str(task_config),
                    "--resource-profile",
                    "condition_validation",
                ]
            ),
            None,
        ),
        "condition_generalization": AnalysisCommands(
            _module_command(pooled_module, "dry-run", "--config", pooled_config),
            _module_command(pooled_module, "new", "--config", pooled_config),
            None,
            shlex.join(
                [
                    "bash",
                    "src/shell_scripts/pooled_task_variable_decoding_slurm.sh",
                    "submit-new",
                    "--config",
                    str(pooled_config),
                ]
            ),
            None,
        ),
        "interregional_regression": AnalysisCommands(
            _module_command(interregional_module, "dry-run", "--config", interregional_config),
            _module_command(interregional_module, "new", "--config", interregional_config),
            None,
            shlex.join(
                [
                    "sbatch",
                    "src/shell_scripts/interregional_regression_slurm.sh",
                    "new",
                    "--config",
                    str(interregional_config),
                ]
            ),
            None,
        ),
        "spike_phase_ppc": AnalysisCommands(
            _module_command(ppc_module, *preview_base, "--shuffles", "100", "--workers", "8", "--dry-run"),
            _module_command(ppc_module, *preview_base, "--shuffles", "100", "--workers", "8"),
            _module_command(
                ppc_module,
                *final_base,
                "--shuffles",
                "1000",
                "--workers",
                "8",
                "--final-run",
            ),
            shlex.join(
                [
                    "sbatch",
                    "src/shell_scripts/hpc_ppc.sh",
                    *(str(item) for item in preview_base),
                    "--shuffles",
                    "100",
                    "--workers",
                    "8",
                ]
            ),
            shlex.join(
                [
                    "sbatch",
                    "src/shell_scripts/hpc_ppc.sh",
                    *(str(item) for item in final_base),
                    "--shuffles",
                    "1000",
                    "--workers",
                    "8",
                    "--final-run",
                ]
            ),
            _module_command(
                "src.neural_analysis.lfp_summary_cache_adoption",
                "--session-metadata",
                metadata_path,
                "--probe",
                ppc_probe_id,
                "--source-cache",
                source_cache,
            ),
        ),
    }
    return MappingProxyType(commands)


def _parser() -> argparse.ArgumentParser:
    """Build the preparation-only command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create missing editable control files")
    init.add_argument("--session-root", type=Path, required=True)
    init.add_argument("--pfc-probe", required=True)
    init.add_argument("--hpc-probe", required=True)
    init.add_argument(
        "--training-windows",
        nargs="+",
        choices=TRAINING_WINDOW_NAMES,
        default=list(TRAINING_WINDOW_NAMES),
    )
    check = commands.add_parser("check", help="inspect readiness and print later commands")
    check.add_argument("--session-root", type=Path, required=True)
    check.add_argument("--pfc-probe", required=True)
    check.add_argument("--hpc-probe", required=True)
    check.add_argument("--ppc-probe", required=True)
    check.add_argument("--ppc-cache-directory", type=Path)
    return parser


def _print_report(report: SessionReadinessReport) -> None:
    """Print one compact deterministic human-readable readiness report."""
    for analysis in report.analyses.values():
        print(f"{analysis.name}: {'READY' if analysis.ready else 'BLOCKED'}")
        for blocker in analysis.blockers:
            print(f"  - {blocker}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run preparation-only setup and return zero for a fully ready session.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments after the module name. ``None`` reads process arguments.

    Returns
    -------
    int
        Zero for successful scaffolding or complete readiness, two for user
        errors or a blocked session. No analysis or scheduler command is run.
    """
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "init":
            result = scaffold_session_analysis(
                arguments.session_root,
                pfc_probe_id=arguments.pfc_probe,
                hpc_probe_id=arguments.hpc_probe,
                training_window_names=arguments.training_windows,
            )
            for path in result.created:
                print(f"created: {path}")
            for path in result.preserved:
                print(f"preserved: {path}")
            return 0

        report = inspect_session_readiness(
            arguments.session_root,
            pfc_probe_id=arguments.pfc_probe,
            hpc_probe_id=arguments.hpc_probe,
            ppc_probe_id=arguments.ppc_probe,
            ppc_cache_directory=arguments.ppc_cache_directory,
        )
        _print_report(report)
        if report.ready:
            resolved, _ = _load_resolved_metadata(
                report.session_root / session_metadata.CANONICAL_FILENAME
            )
            assert resolved is not None
            cache = arguments.ppc_cache_directory or resolved.cache_directory
            assert cache is not None
            print("\nCommands for later explicit execution:")
            for name, command_set in build_analysis_commands(
                report.session_root,
                ppc_probe_id=arguments.ppc_probe,
                ppc_cache_directory=cache,
            ).items():
                print(f"\n{name}:")
                for command in (
                    command_set.preparation,
                    command_set.local_preflight,
                    command_set.local_preview,
                    command_set.local_final,
                    command_set.cluster_preview,
                    command_set.cluster_final,
                ):
                    if command is not None:
                        print(f"  {command}")
        return 0 if report.ready else 2
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
