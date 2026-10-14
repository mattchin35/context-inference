"""Prepare Power, Synchrony, and reusable phase inputs from session metadata.

This orchestration module does not implement numerical analysis. It calls the
existing component and cache-adoption boundaries and records scalar provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable

from src.neural_analysis.lfp_summary.models import LFPSummaryConfig
from src.neural_analysis.lfp_summary.pipeline import ComponentRunResult
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_session_metadata,
)


_WORKER_COUNT = 8


@dataclass(frozen=True)
class PreparationRequest:
    """One metadata-owned preparation request with its exact command tokens.

    Parameters
    ----------
    session_metadata_path : pathlib.Path
        Existing ``neural_session.json`` path; it contains all session, probe,
        cache, and shuffle identities.
    command : tuple[str, ...]
        Exact scalar command tokens retained in the receipt.
    """

    session_metadata_path: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class PreparationDependencies:
    """Injected configuration, status, compute, adoption, and provenance seams.

    All callables exchange immutable configurations, scalar state strings, or
    opaque prepared-phase objects. Numerical arrays remain owned by the existing
    LFP runtime reached through these seams.
    """

    load_config: Callable[[Path, str, Path, int, int], LFPSummaryConfig]
    component_status: Callable[[LFPSummaryConfig, str], str]
    compute_power: Callable[[LFPSummaryConfig], ComponentRunResult]
    prepare_phase: Callable[[LFPSummaryConfig], object]
    compute_synchrony: Callable[[LFPSummaryConfig, object], ComponentRunResult]
    adopt_cache_pair: Callable[..., object]
    repository_state: Callable[[], tuple[str, bool]]
    now_utc: Callable[[], str]


@dataclass(frozen=True)
class PreparationResult:
    """Terminal scalar outcome and paths for one preparation invocation."""

    status: str
    receipt_path: Path
    prepared_phase_state: str | None
    source_cache_directory: Path
    preview_cache_directory: Path
    final_cache_directory: Path


def _require_complete(result: ComponentRunResult, component: str) -> None:
    """Raise when an existing component runner did not commit successfully.

    Parameters
    ----------
    result : ComponentRunResult
        Scalar result returned by the existing component pipeline.
    component : str
        Expected component identity.

    Returns
    -------
    None
        Success means the requested component is durably complete.
    """
    if result.component != component or result.state != "complete":
        detail = result.error or result.stage
        raise RuntimeError(f"{component} preparation failed: {detail}")


def _require_component_state(state: str, cache_label: str, component: str) -> None:
    """Accept only missing or compatible cache state before orchestration.

    Parameters are categorical state/cache/component labels without arrays,
    axis conventions, or physical units. The function returns ``None``.
    """
    if state not in {"missing", "compatible"}:
        raise ValueError(
            f"{cache_label} {component} cache is {state}; refusing overwrite"
        )


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    """Atomically write one ASCII JSON preparation receipt.

    Parameters
    ----------
    path : pathlib.Path
        New receipt path below the session ``analysis_runs`` directory.
    payload : dict[str, object]
        JSON-safe scalar provenance; it contains no analysis arrays.

    Returns
    -------
    None
        The destination exists only after a complete atomic replacement.
    """
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _phase_state(prepared_phase: object) -> str | None:
    """Return the optional categorical cache state from an opaque phase record."""
    value = getattr(prepared_phase, "prepared_phase_cache_state", None)
    if value not in {None, "cold", "warm"}:
        raise ValueError("prepared phase cache state must be cold, warm, or None")
    return value


def prepare_session_lfp(
    request: PreparationRequest,
    dependencies: PreparationDependencies,
) -> PreparationResult:
    """Prepare or reuse one metadata-declared LFP prerequisite workflow.

    Parameters
    ----------
    request : PreparationRequest
        Version-3 metadata identity and exact command provenance.
    dependencies : PreparationDependencies
        Existing numerical/runtime boundaries plus injectable scalar services.

    Returns
    -------
    PreparationResult
        Completed receipt and cache paths. Phase arrays remain in the existing
        ``processed/lfp_summary_work`` representation and are not returned.

    Raises
    ------
    ValueError
        If metadata lacks analysis intent, target publication is partial, a
        cache is stale/running/failed, or tracked source is dirty.
    RuntimeError
        If an existing component computation reports failure.
    """
    metadata_path = Path(request.session_metadata_path).resolve(strict=True)
    metadata = load_session_metadata(metadata_path)
    session = resolve_session_metadata(metadata, metadata_path)
    if session.analysis is None:
        raise ValueError("LFP preparation requires session metadata schema version 3")
    ppc = session.analysis.ppc
    preview_exists = ppc.preview_cache.exists() or ppc.preview_cache.is_symlink()
    final_exists = ppc.final_cache.exists() or ppc.final_cache.is_symlink()
    if preview_exists != final_exists:
        raise ValueError("partial preview/final cache pair exists; refusing overwrite")

    git_commit, tracked_clean = dependencies.repository_state()
    if not tracked_clean:
        raise ValueError("LFP preparation requires a clean tracked Git checkout")
    roles = session.analysis.probe_roles
    stages: dict[str, str]

    if preview_exists and final_exists:
        preview_config = dependencies.load_config(
            metadata_path,
            roles.ppc,
            ppc.preview_cache,
            ppc.preview_shuffles,
            _WORKER_COUNT,
        )
        final_config = dependencies.load_config(
            metadata_path,
            roles.ppc,
            ppc.final_cache,
            ppc.final_shuffles,
            _WORKER_COUNT,
        )
        for label, config in (("preview", preview_config), ("final", final_config)):
            for component in ("power", "synchrony"):
                state = dependencies.component_status(config, component)
                if state != "compatible":
                    raise ValueError(f"{label} {component} cache is {state}; expected compatible")
        prepared_phase = dependencies.prepare_phase(preview_config)
        stages = {
            "power": "reused",
            "prepared_phase": "reused" if _phase_state(prepared_phase) == "warm" else "computed",
            "synchrony": "reused",
            "cache_pair": "reused",
        }
    else:
        source_exists = ppc.prerequisite_cache.exists() or ppc.prerequisite_cache.is_symlink()
        if ppc.prerequisite_cache.is_symlink():
            raise ValueError("prerequisite cache must not be a symbolic link")
        source_members = (
            {item.name for item in ppc.prerequisite_cache.iterdir()}
            if source_exists and ppc.prerequisite_cache.is_dir()
            else set()
        )
        transferred_source = source_members == {
            "manifest.json",
            "power.npz",
            "synchrony.npz",
        }
        if source_members and not transferred_source:
            raise ValueError(
                "prerequisite cache is partial or has unexpected members; refusing overwrite"
            )
        ppc.prerequisite_cache.mkdir(parents=True, exist_ok=True)
        adoption_arguments = {
            "metadata_path": metadata_path,
            "probe": roles.ppc,
            "source": ppc.prerequisite_cache,
            "preview": ppc.preview_cache,
            "final": ppc.final_cache,
            "preview_shuffles": ppc.preview_shuffles,
            "final_shuffles": ppc.final_shuffles,
            "command": request.command,
        }
        if transferred_source:
            dependencies.adopt_cache_pair(**adoption_arguments)
            preview_config = dependencies.load_config(
                metadata_path,
                roles.ppc,
                ppc.preview_cache,
                ppc.preview_shuffles,
                _WORKER_COUNT,
            )
            prepared_phase = dependencies.prepare_phase(preview_config)
            stages = {
                "power": "reused",
                "prepared_phase": (
                    "reused" if _phase_state(prepared_phase) == "warm" else "computed"
                ),
                "synchrony": "reused",
                "cache_pair": "adopted",
            }
        else:
            source_config = dependencies.load_config(
                metadata_path,
                roles.ppc,
                ppc.prerequisite_cache,
                ppc.preview_shuffles,
                _WORKER_COUNT,
            )
            power_state = dependencies.component_status(source_config, "power")
            synchrony_state = dependencies.component_status(source_config, "synchrony")
            _require_component_state(power_state, "prerequisite", "power")
            _require_component_state(synchrony_state, "prerequisite", "synchrony")
            if power_state == "missing":
                _require_complete(dependencies.compute_power(source_config), "power")
            prepared_phase = dependencies.prepare_phase(source_config)
            if synchrony_state == "missing":
                _require_complete(
                    dependencies.compute_synchrony(source_config, prepared_phase),
                    "synchrony",
                )
            dependencies.adopt_cache_pair(**adoption_arguments)
            stages = {
                "power": "computed" if power_state == "missing" else "reused",
                "prepared_phase": (
                    "reused" if _phase_state(prepared_phase) == "warm" else "computed"
                ),
                "synchrony": "computed" if synchrony_state == "missing" else "reused",
                "cache_pair": "adopted",
            }

    timestamp = dependencies.now_utc()
    run_directory = (
        session.session_root
        / "analysis_runs"
        / f"{session.session}_lfp_preparation_{timestamp}"
    )
    receipt_path = run_directory / "preparation.json"
    phase_state = _phase_state(prepared_phase)
    _write_receipt(
        receipt_path,
        {
            "schema_version": "lfp_preparation.v1",
            "generator": "src.neural_analysis.lfp_summary.preparation_runner.prepare_session_lfp",
            "session": session.session,
            "session_metadata_path": str(metadata_path),
            "command": list(request.command),
            "git_commit": git_commit,
            "tracked_clean": tracked_clean,
            "completed_at_utc": timestamp,
            "probe": roles.ppc,
            "source_cache": str(ppc.prerequisite_cache),
            "preview_cache": str(ppc.preview_cache),
            "final_cache": str(ppc.final_cache),
            "preview_shuffles": ppc.preview_shuffles,
            "final_shuffles": ppc.final_shuffles,
            "prepared_phase_cache_state": phase_state,
            "stages": stages,
        },
    )
    return PreparationResult(
        status="complete",
        receipt_path=receipt_path,
        prepared_phase_state=phase_state,
        source_cache_directory=ppc.prerequisite_cache,
        preview_cache_directory=ppc.preview_cache,
        final_cache_directory=ppc.final_cache,
    )


def make_production_preparation_dependencies() -> PreparationDependencies:
    """Bind the unchanged production Power, phase, Synchrony, and adoption APIs.

    Returns
    -------
    PreparationDependencies
        Production adapters. Power/Synchrony retain their existing array,
        axes, units, seeds, and manifest-last transaction behavior.
    """
    from src.neural_analysis.lfp_summary import cache as lfp_cache
    from src.neural_analysis.lfp_summary.cache_relocation import (
        CacheAdoptionRequest,
        adopt_power_synchrony_cache_pair,
    )
    from src.neural_analysis.lfp_summary.launcher_runtime import (
        load_metadata_spike_phase_config,
    )
    from src.neural_analysis.lfp_summary.pipeline import (
        compute_power_component,
        compute_synchrony_component,
    )
    from src.neural_analysis.lfp_summary.runtime import (
        make_lfp_summary_pipeline_dependencies,
    )
    from src.neural_analysis.lfp_summary.runtime_common import (
        load_configured_trial_table,
    )

    pipeline_dependencies = make_lfp_summary_pipeline_dependencies(
        trial_table_loader=load_configured_trial_table,
    )

    def component_status(config: LFPSummaryConfig, component: str) -> str:
        """Inspect one component using headers and active source fingerprints."""
        try:
            manifest = lfp_cache.load_or_initialize_manifest(
                config.output_directory,
                config,
            )
            return lfp_cache.assess_component_status(
                config.output_directory,
                component,
                config,
                manifest,
                validate_headers_only=True,
            ).status
        except (OSError, ValueError):
            return "failed"

    def compute_power(config: LFPSummaryConfig) -> ComponentRunResult:
        """Dispatch one unchanged production Power component computation."""
        return compute_power_component(config, pipeline_dependencies)

    def prepare_phase(config: LFPSummaryConfig) -> object:
        """Dispatch the existing reusable prepared-phase implementation once."""
        return pipeline_dependencies.prepare_phase(config)

    def compute_synchrony(
        config: LFPSummaryConfig,
        prepared_phase: object,
    ) -> ComponentRunResult:
        """Dispatch unchanged Synchrony using the exact supplied phase object."""
        return compute_synchrony_component(
            config,
            pipeline_dependencies,
            prepared_phase=prepared_phase,
        )

    def adopt_cache_pair(
        *,
        metadata_path: Path,
        probe: str,
        source: Path,
        preview: Path,
        final: Path,
        preview_shuffles: int,
        final_shuffles: int,
        command: tuple[str, ...],
    ) -> object:
        """Publish the source components to metadata-declared target caches."""
        return adopt_power_synchrony_cache_pair(
            CacheAdoptionRequest(
                session_metadata_path=metadata_path,
                probe_label=probe,
                source_cache_directory=source,
                command=command,
                preview_cache_directory=preview,
                final_cache_directory=final,
                preview_shuffle_count=preview_shuffles,
                final_shuffle_count=final_shuffles,
            )
        )

    def repository_state() -> tuple[str, bool]:
        """Return the current commit and tracked-worktree cleanliness."""
        repository_root = Path(__file__).resolve().parents[3]
        commit = subprocess.run(
            ("git", "-C", str(repository_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            (
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, not status.strip()

    def now_utc() -> str:
        """Return one filesystem-safe UTC timestamp without physical units."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    return PreparationDependencies(
        load_config=load_metadata_spike_phase_config,
        component_status=component_status,
        compute_power=compute_power,
        prepare_phase=prepare_phase,
        compute_synchrony=compute_synchrony,
        adopt_cache_pair=adopt_cache_pair,
        repository_state=repository_state,
        now_utc=now_utc,
    )
