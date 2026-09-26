"""Session metadata, view availability, and channel-selection inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_probe_sources,
    resolve_session_metadata,
)
from src.neural_analysis.spike_behavior import loading as unit_spike_loading


CONDITION_OPTIONS = [
    "all",
    "correct_rewarded",
    "incorrect",
    "omission",
    "switch",
    "stay",
]

ACTION_OPTIONS = {
    "all": "all",
    "right (0)": 0,
    "left (1)": 1,
    "compare left vs right": "compare_lr",
}

ALIGNMENT_OPTIONS = ["choice_time", "start_time"]

PLOT_VIEW_UNIT_RASTER = "Unit raster/PSTH"

PLOT_VIEW_TRIAL_SPIKES = "Trial spikes/licks/choices"

PLOT_VIEW_PCA_DECODING = "Population PCA decoding"

PLOT_VIEW_PCA_SWITCH_TRAJECTORIES = "Population PCA switch trajectories"

PLOT_VIEW_LFP_PHASE_CLUSTERING = "LFP phase clustering"

PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE = "Single-trial relative phase"

PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING = "Spike-LFP phase locking"

PLOT_VIEW_SINGLE_TRIAL_SPIKE_LFP_HILBERT = "Single-trial spike-LFP phase"

PLOT_VIEW_LFP_SUMMARY = "Cached LFP summary"

PLOT_VIEW_OPTIONS = [
    PLOT_VIEW_UNIT_RASTER,
    PLOT_VIEW_TRIAL_SPIKES,
    PLOT_VIEW_PCA_DECODING,
    PLOT_VIEW_PCA_SWITCH_TRAJECTORIES,
    PLOT_VIEW_LFP_PHASE_CLUSTERING,
    PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE,
    PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING,
    PLOT_VIEW_SINGLE_TRIAL_SPIKE_LFP_HILBERT,
    PLOT_VIEW_LFP_SUMMARY,
]

CHANNEL_SOURCE_MANUAL = "Manual / preset"

CHANNEL_SOURCE_CHANNEL_QUALITY = "channel_quality"

CHANNEL_SOURCE_OPTIONS = [CHANNEL_SOURCE_MANUAL, CHANNEL_SOURCE_CHANNEL_QUALITY]

COMPARE_LEFT_RIGHT_ACTION = "compare_lr"

LFP_DROPDOWN_LABEL_HPC_V1 = "HPC/V1 LFP"

LFP_DROPDOWN_LABEL_PFC = "PFC LFP"

LFP_FORMAT_SPIKEGLX = "SpikeGLX"

LFP_FORMAT_OPEN_EPHYS_DERIVED = "Open Ephys derived"

LFP_FORMAT_OPTIONS = [LFP_FORMAT_SPIKEGLX, LFP_FORMAT_OPEN_EPHYS_DERIVED]

@dataclass(frozen=True)
class MetadataLFPSiteInputs:
    """One metadata-defined saved LFP channel and its resolved source paths."""

    site_id: str
    display_label: str
    probe_id: str
    acquisition_family: str
    saved_channel_index: int
    lfp_file: Path | None
    synchronization_file: Path | None

@dataclass(frozen=True)
class MetadataViewAvailability:
    """Whether one existing webapp view has its ordinary required sources."""

    available: bool
    reason: str

def load_webapp_session(metadata_path: Path | str) -> ResolvedSession:
    """Load and resolve one canonical session document without reading arrays.

    Parameters
    ----------
    metadata_path : pathlib.Path or str
        Existing ``neural_session.json`` path.

    Returns
    -------
    ResolvedSession
        Frozen session, probe, site, population, and filesystem records.
    """
    path = Path(metadata_path)
    return resolve_session_metadata(load_session_metadata(path), path)

def metadata_lfp_site_inputs(session: ResolvedSession) -> tuple[MetadataLFPSiteInputs, ...]:
    """Return metadata-defined LFP sites in the user's declared order.

    Parameters
    ----------
    session : ResolvedSession
        One metadata-resolved session.

    Returns
    -------
    tuple[MetadataLFPSiteInputs, ...]
        Site labels, zero-based saved channels, and source paths; no LFP data
        or sidecar values are loaded.
    """
    sites = []
    for site in session.sites:
        probe = resolve_probe_sources(session, site.probe_id)
        sites.append(
            MetadataLFPSiteInputs(
                site_id=site.site_id,
                display_label=site.display_label,
                probe_id=probe.probe_id,
                acquisition_family=probe.acquisition_family,
                saved_channel_index=site.saved_channel_index,
                lfp_file=probe.lfp_file,
                synchronization_file=probe.synchronization_file,
            )
        )
    return tuple(sites)

def metadata_view_availability(
    session: ResolvedSession,
) -> dict[str, MetadataViewAvailability]:
    """Report ordinary source availability for each existing webapp view.

    Parameters
    ----------
    session : ResolvedSession
        Resolved paths. Files are checked by kind but never opened.

    Returns
    -------
    dict[str, MetadataViewAvailability]
        One entry for every value in ``PLOT_VIEW_OPTIONS``.
    """
    behavior_ready = (
        session.behavior.session_directory is not None
        and session.behavior.session_directory.is_dir()
        and session.behavior.trial_table_file is not None
        and session.behavior.trial_table_file.is_file()
    )
    lfp_ready = bool(session.sites) and all(
        site.lfp_file is not None
        and site.lfp_file.is_file()
        and site.synchronization_file is not None
        and site.synchronization_file.is_file()
        for site in metadata_lfp_site_inputs(session)
    )
    missing_spike_probes = []
    for probe in session.probes:
        if (
            probe.sorter_directory is None
            or not probe.sorter_directory.is_dir()
            or probe.aligned_spike_file is None
            or not probe.aligned_spike_file.is_file()
        ):
            missing_spike_probes.append(probe.probe_id)
    spike_ready = bool(session.probes) and not missing_spike_probes

    behavior_reason = "behavior trial sources are missing"
    lfp_reason = "LFP or synchronization sources are missing"
    spike_reason = (
        "spike sources are missing for " + ", ".join(missing_spike_probes)
        if missing_spike_probes
        else "no metadata population is configured"
    )

    availability: dict[str, MetadataViewAvailability] = {}
    lfp_only = {PLOT_VIEW_LFP_PHASE_CLUSTERING, PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE}
    combined = {PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING, PLOT_VIEW_SINGLE_TRIAL_SPIKE_LFP_HILBERT}
    for view in PLOT_VIEW_OPTIONS:
        if not behavior_ready:
            availability[view] = MetadataViewAvailability(False, behavior_reason)
        elif view in lfp_only and not lfp_ready:
            availability[view] = MetadataViewAvailability(False, lfp_reason)
        elif view in combined and (not lfp_ready or not spike_ready):
            reason = lfp_reason if not lfp_ready else spike_reason
            availability[view] = MetadataViewAvailability(False, reason)
        elif view not in lfp_only | combined | {PLOT_VIEW_LFP_SUMMARY} and not spike_ready:
            availability[view] = MetadataViewAvailability(False, spike_reason)
        else:
            availability[view] = MetadataViewAvailability(True, "available")
    return availability

def resolve_region_channels_for_source(
    channel_source: str,
    manual_channel_text: str,
    channel_quality: pd.DataFrame | None,
    require_inside_brain: bool,
    channel_quality_labels: tuple[str, ...] | list[str] | None,
) -> tuple[np.ndarray, str]:
    """
    Resolve active region channels from manual text or channel-quality metadata.

    Parameters
    ----------
    channel_source : str
        Channel source label. Supported values are ``CHANNEL_SOURCE_MANUAL`` and
        ``CHANNEL_SOURCE_CHANNEL_QUALITY``.
    manual_channel_text : str
        Editable channel text. Used only for ``CHANNEL_SOURCE_MANUAL``. Values
        are zero-based channel ids separated by commas or whitespace.
    channel_quality : pd.DataFrame | None
        Normalized channel-quality table with shape ``(n_channels, n_columns)``.
        Required for ``CHANNEL_SOURCE_CHANNEL_QUALITY``. ``None`` is allowed
        only for manual channels.
    require_inside_brain : bool
        If ``True``, channel-quality selection keeps only channels with
        ``inside_brain == True``. Ignored for manual channels.
    channel_quality_labels : tuple[str, ...] | list[str] | None
        Channel-quality labels to include, such as ``("good",)``. ``None``
        disables label filtering. Ignored for manual channels.

    Returns
    -------
    tuple[np.ndarray, str]
        ``(region_channels, summary)``. ``region_channels`` is a one-dimensional
        integer array of zero-based channel ids. ``summary`` is a concise
        human-readable description of the selection.
    """

    if channel_source == CHANNEL_SOURCE_MANUAL:
        region_channels = unit_spike_loading.parse_channel_list(manual_channel_text)
        return region_channels, f"Manual channel selection: {region_channels.size} channels."
    if channel_source == CHANNEL_SOURCE_CHANNEL_QUALITY:
        if channel_quality is None:
            raise ValueError("channel_quality metadata is required for channel_quality channel source.")
        region_channels = unit_spike_loading.select_channels_from_quality(
            channel_quality=channel_quality,
            require_inside_brain=bool(require_inside_brain),
            labels=channel_quality_labels,
        )
        return region_channels, f"channel_quality selected {region_channels.size} / {channel_quality.shape[0]} channels."
    raise ValueError(f"Unsupported channel source: {channel_source!r}")
