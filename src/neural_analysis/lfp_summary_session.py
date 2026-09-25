"""Build the existing LFP-summary configuration from resolved session metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

from src.external_tools import readSGLX
from src.neural_analysis import lfp_loading
from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    LFPSummaryConfig,
    UnitPopulationConfig,
    default_lfp_summary_config,
    validate_lfp_summary_config,
)
from src.neural_analysis.session_metadata import (
    ResolvedProbeSources,
    ResolvedSession,
    resolve_probe_sources,
)


@dataclass(frozen=True)
class LFPSummarySessionRequest:
    """User choices needed to build one existing summary configuration.

    ``population_id`` is a categorical metadata ID. ``output_directory`` is a
    cache directory path with no physical unit. Scientific bands, windows,
    seeds, and numerical thresholds remain the code-owned defaults.
    """

    population_id: str | None = None
    output_directory: Path | None = None


def _required_path(path: Path | None, field: str, kind: str) -> Path:
    """Return an existing path of ``kind`` or raise a concise user error."""
    if path is None:
        raise ValueError(f"{field} is required")
    if kind == "file" and not path.is_file():
        raise ValueError(f"{field} must identify an existing file: {path}")
    if kind == "directory" and not path.is_dir():
        raise ValueError(f"{field} must identify an existing directory: {path}")
    return path


def _site_physical_metadata(probe: ResolvedProbeSources) -> tuple[float, str]:
    """Read one authoritative text sidecar and return ``(sample_rate_hz, unit)``.

    Parameters
    ----------
    probe : ResolvedProbeSources
        One explicit Open Ephys or SpikeGLX source. No signal array is opened.

    Returns
    -------
    tuple[float, str]
        Positive sample rate in samples/second and physical voltage unit
        ``"uV"`` used by the current loaders.
    """
    lfp_path = _required_path(probe.lfp_file, f"probes[{probe.probe_id}].lfp_file", "file")
    metadata_path = _required_path(
        probe.lfp_metadata_file,
        f"probes[{probe.probe_id}].lfp_metadata_file",
        "file",
    )
    if probe.acquisition_family == "open_ephys":
        expected_path = lfp_path.parent / "lfp_preprocessing.json"
        if metadata_path != expected_path:
            raise ValueError(
                f"probes[{probe.probe_id}].lfp_metadata_file must be {expected_path}"
            )
        metadata = lfp_loading.load_open_ephys_lfp_metadata(lfp_path)
        sample_rate_hz = float(metadata["sampling_frequency_hz"])
    elif probe.acquisition_family == "spikeglx":
        expected_path = lfp_path.with_suffix(".meta")
        if metadata_path != expected_path:
            raise ValueError(
                f"probes[{probe.probe_id}].lfp_metadata_file must be {expected_path}"
            )
        metadata = lfp_loading.load_lfp_metadata(lfp_path)
        sample_rate_hz = float(readSGLX.SampRate(metadata))
    else:  # Structural metadata validation normally makes this unreachable.
        raise ValueError(f"unsupported acquisition family: {probe.acquisition_family}")
    if not isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"probe {probe.probe_id!r} has an invalid sample rate")
    return sample_rate_hz, "uV"


def _build_sites(session: ResolvedSession) -> tuple[LFPSiteConfig, ...]:
    """Build site records while reading each probe sidecar at most once."""
    physical_by_probe: dict[str, tuple[float, str]] = {}
    sites = []
    for site in session.sites:
        probe = resolve_probe_sources(session, site.probe_id)
        if probe.probe_id not in physical_by_probe:
            physical_by_probe[probe.probe_id] = _site_physical_metadata(probe)
        sample_rate_hz, voltage_unit = physical_by_probe[probe.probe_id]
        sites.append(
            LFPSiteConfig(
                stable_id=site.site_id,
                label=site.display_label,
                acquisition_format=probe.acquisition_family,
                lfp_path=_required_path(
                    probe.lfp_file, f"probes[{probe.probe_id}].lfp_file", "file"
                ),
                aligned_sync_path=_required_path(
                    probe.synchronization_file,
                    f"probes[{probe.probe_id}].synchronization_file",
                    "file",
                ),
                probe_label=probe.probe_id,
                saved_channel_index=site.saved_channel_index,
                voltage_unit=voltage_unit,
                sample_rate_hz=sample_rate_hz,
            )
        )
    return tuple(sites)


def _build_population(
    session: ResolvedSession,
    population_id: str | None,
) -> UnitPopulationConfig | None:
    """Build one metadata-selected population without loading spike arrays."""
    if population_id is None:
        return None
    population = next(
        (item for item in session.populations if item.population_id == population_id),
        None,
    )
    if population is None:
        raise ValueError(f"unknown population ID: {population_id}")
    group = next(
        item
        for item in session.channel_groups
        if item.channel_group_id == population.channel_group_id
    )
    probe = resolve_probe_sources(session, population.probe_id)
    return UnitPopulationConfig(
        label=population.display_label,
        probe_label=probe.probe_id,
        sorter_path=_required_path(
            probe.sorter_directory,
            f"probes[{probe.probe_id}].sorter_directory",
            "directory",
        ),
        aligned_spike_path=_required_path(
            probe.aligned_spike_file,
            f"probes[{probe.probe_id}].aligned_spike_file",
            "file",
        ),
        selected_channels=group.channel_indices,
        quality_settings=(),
        stable_unit_ids=(),
    )


def build_lfp_summary_config(
    session: ResolvedSession,
    request: LFPSummarySessionRequest,
) -> LFPSummaryConfig:
    """Build the current immutable LFP-summary config from one resolved session.

    Parameters
    ----------
    session : ResolvedSession
        Metadata-resolved session labels and paths. Site channels are zero
        based; LFP sidecars provide sample rates in samples/second and uV units.
    request : LFPSummarySessionRequest
        Optional population ID and cache directory. It contains no scientific
        thresholds or numerical arrays.

    Returns
    -------
    LFPSummaryConfig
        Existing validated configuration using code-owned scientific defaults.
    """
    if not session.session_id:
        raise ValueError("session_id is required to build an LFP summary configuration")
    trial_table_path = _required_path(
        session.behavior.trial_table_file,
        "behavior.trial_table_file",
        "file",
    )
    if request.output_directory is not None:
        output_directory = Path(request.output_directory)
        if not output_directory.is_absolute():
            output_directory = session.session_root / output_directory
    elif session.lfp_summary_cache_directory is not None:
        output_directory = session.lfp_summary_cache_directory
    else:
        output_directory = session.session_root / "processed" / "lfp_summary_cache"

    base = default_lfp_summary_config()
    config = replace(
        base,
        session_id=session.session_id,
        session_path=session.session_root,
        output_directory=output_directory,
        sites=_build_sites(session),
        site_pairs=session.site_pairs,
        unit_population=_build_population(session, request.population_id),
        trial_table_path=trial_table_path,
    )
    validate_lfp_summary_config(config)
    return config
