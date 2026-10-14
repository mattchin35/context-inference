"""Initialize, inspect, and prepare one metadata-owned neural session.

``prepare`` delegates Power, Synchrony, and prepared-phase work to their
existing implementations. No command in this module submits a scheduler job.
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


def _resolved_version_three_session(
    metadata_path: Path,
) -> session_metadata.ResolvedSession:
    """Load one version-3 metadata document with complete analysis intent.

    Parameters
    ----------
    metadata_path : pathlib.Path
        Existing canonical session metadata JSON path.

    Returns
    -------
    session_metadata.ResolvedSession
        Resolved paths and stable roles. No numerical arrays are loaded.
    """
    resolved = session_metadata.resolve_session_metadata(
        session_metadata.load_session_metadata(metadata_path),
        metadata_path,
    )
    if resolved.analysis is None:
        raise ValueError("single-session workflow requires metadata schema version 3")
    return resolved


def build_metadata_analysis_commands(
    metadata_path: Path | str,
) -> Mapping[str, AnalysisCommands]:
    """Build simple copyable commands from one version-3 metadata path.

    Parameters
    ----------
    metadata_path : pathlib.Path or str
        Canonical control file that owns session, role, cache, and shuffle
        settings.

    Returns
    -------
    mapping[str, AnalysisCommands]
        Commands are shell-escaped text only; nothing is executed or submitted.
    """
    path = Path(metadata_path)
    resolved = _resolved_version_three_session(path)
    assert resolved.analysis is not None
    existing = dict(
        build_analysis_commands(
            resolved.session_root,
            ppc_probe_id=resolved.analysis.probe_roles.ppc,
            ppc_cache_directory=resolved.analysis.ppc.prerequisite_cache,
        )
    )
    launcher = "src.neural_analysis.lfp_spike_phase_launcher"
    setup = "src.neural_analysis.single_session_setup"
    canonical_metadata = resolved.metadata_path
    existing["spike_phase_ppc"] = AnalysisCommands(
        local_preflight=_module_command(
            launcher,
            "preview",
            "--session-metadata",
            canonical_metadata,
            "--dry-run",
        ),
        local_preview=_module_command(
            launcher,
            "preview",
            "--session-metadata",
            canonical_metadata,
        ),
        local_final=_module_command(
            launcher,
            "final",
            "--session-metadata",
            canonical_metadata,
        ),
        cluster_preview=shlex.join(
            (
                "sbatch",
                "src/shell_scripts/hpc_ppc.sh",
                "preview",
                "--session-metadata",
                str(canonical_metadata),
            )
        ),
        cluster_final=shlex.join(
            (
                "sbatch",
                "src/shell_scripts/hpc_ppc.sh",
                "final",
                "--session-metadata",
                str(canonical_metadata),
            )
        ),
        preparation=_module_command(
            setup,
            "prepare",
            "--session-metadata",
            canonical_metadata,
        ),
    )
    return MappingProxyType(existing)


def _parser() -> argparse.ArgumentParser:
    """Build the metadata-only single-session workflow parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create missing editable control files")
    init.add_argument("--session-metadata", type=Path, required=True)
    check = commands.add_parser("check", help="inspect readiness and print later commands")
    check.add_argument("--session-metadata", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare",
        help="compute or reuse Power, Synchrony, and prepared-phase prerequisites",
    )
    prepare.add_argument("--session-metadata", type=Path, required=True)
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
        resolved = _resolved_version_three_session(arguments.session_metadata)
        assert resolved.analysis is not None
        roles = resolved.analysis.probe_roles
        if arguments.command == "init":
            result = scaffold_session_analysis(
                resolved.session_root,
                pfc_probe_id=roles.pfc,
                hpc_probe_id=roles.hpc,
            )
            for path in result.created:
                print(f"created: {path}")
            for path in result.preserved:
                print(f"preserved: {path}")
            return 0

        if arguments.command == "prepare":
            from src.neural_analysis.lfp_summary.preparation_runner import (
                PreparationRequest,
                make_production_preparation_dependencies,
                prepare_session_lfp,
            )

            result = prepare_session_lfp(
                PreparationRequest(
                    resolved.metadata_path,
                    tuple(sys.argv if argv is None else ("single_session_setup", *argv)),
                ),
                make_production_preparation_dependencies(),
            )
            print(f"preparation complete: {result.receipt_path}")
            return 0

        report = inspect_session_readiness(
            resolved.session_root,
            pfc_probe_id=roles.pfc,
            hpc_probe_id=roles.hpc,
            ppc_probe_id=roles.ppc,
            ppc_cache_directory=resolved.analysis.ppc.preview_cache,
        )
        _print_report(report)
        if report.ready:
            print("\nCommands for later explicit execution:")
            for name, command_set in build_metadata_analysis_commands(
                resolved.metadata_path
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
