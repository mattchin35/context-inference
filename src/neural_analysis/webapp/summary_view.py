"""LFP-summary composition for metadata and legacy webapp inputs."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.neural_analysis import lfp_summary_webapp
from src.neural_analysis.lfp_summary.session import (
    LFPSummarySessionRequest,
    build_lfp_summary_config,
)
from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.webapp.data_loading import (
    load_summary_channel_metadata,
    load_summary_cluster_metadata,
)


def render_metadata_lfp_summary_view(
    streamlit: object,
    session: ResolvedSession,
    *,
    dependencies: lfp_summary_webapp.SummaryWebDependencies | None = None,
) -> None:
    """Render the existing summary UI from one resolved metadata session.

    Parameters
    ----------
    streamlit : object
        Streamlit-compatible UI host. It is not passed to scientific loaders.
    session : ResolvedSession
        Explicit session/probe/site paths and zero-based channel identities.
    dependencies : SummaryWebDependencies or None
        Existing cache/compute seams. ``None`` builds the production bundle.

    Returns
    -------
    None
        Delegates controls only; raw arrays remain behind existing explicit UI
        actions and cached-view selections.
    """
    config = build_lfp_summary_config(session, LFPSummarySessionRequest())
    sorter_paths = {
        probe.probe_id: probe.sorter_directory
        for probe in session.probes
        if probe.sorter_directory is not None
    }
    aligned_spike_paths = {
        probe.probe_id: probe.aligned_spike_file
        for probe in session.probes
        if probe.aligned_spike_file is not None
    }
    if dependencies is None:
        dependencies = lfp_summary_webapp.make_production_summary_dependencies()
    lfp_summary_webapp.render_lfp_summary_view(
        streamlit,
        session_id=config.session_id,
        session_path=config.session_path,
        output_directory=config.output_directory,
        sites=config.sites,
        site_pairs=config.site_pairs,
        unit_population=config.unit_population,
        dependencies=dependencies,
        sorter_paths=sorter_paths,
        aligned_spike_paths=aligned_spike_paths,
        cluster_metadata_loader=load_summary_cluster_metadata,
        channel_metadata_loader=load_summary_channel_metadata,
        trial_table_path=config.trial_table_path,
    )

def render_lfp_summary_view(
    session_data_home: str,
    sess_id_full: str,
    hpc_v1_lfp_path: str,
    pfc_lfp_path: str,
    hpc_v1_aligned_spike_path: str,
    pfc_aligned_spike_path: str,
    dependencies: lfp_summary_webapp.SummaryWebDependencies | None = None,
    hpc_v1_sorter_output_path: str | None = None,
    pfc_sorter_output_path: str | None = None,
) -> None:
    """Delegate the cached LFP summary route using active session/probe inputs.

    Parameters
    ----------
    session_data_home : str
        Active session directory path.
    sess_id_full : str
        Active session identifier.
    hpc_v1_lfp_path, pfc_lfp_path : str
        Active continuous LFP source paths for the HPC and PFC probes.
    hpc_v1_aligned_spike_path, pfc_aligned_spike_path : str
        Active synchronized sidecar paths used as site source metadata.
    dependencies : SummaryWebDependencies | None
        Injected summary pipeline/cache seams. ``None`` reports the currently
        explicit production integration gap without computing numerical data.
    hpc_v1_sorter_output_path, pfc_sorter_output_path : str | None
        Optional explicit HPC/V1 and PFC ``kilosort4`` directory paths. When at
        least one is supplied, the additive source-aware child route receives
        ProbeA/PFC and ProbeB/HPC path mappings plus lazy metadata callbacks.
        Empty values remain absent rather than being inferred from session data.

    Returns
    -------
    None
        Delegates Streamlit controls and cached-result rendering to
        ``lfp_summary_webapp``. LFP source values remain in their native units.
    """

    session_path = Path(session_data_home)
    if dependencies is None:
        dependencies = lfp_summary_webapp.make_production_summary_dependencies()
    pfc_format = _summary_lfp_format(pfc_lfp_path)
    hpc_format = _summary_lfp_format(hpc_v1_lfp_path)
    sites = (
        lfp_summary_webapp.LFPSiteConfig(
            "PFC",
            "PFC",
            pfc_format,
            Path(pfc_lfp_path),
            Path(pfc_aligned_spike_path) if pfc_aligned_spike_path else None,
            "ProbeA",
            5,
            "uV",
            2500.0,
        ),
        lfp_summary_webapp.LFPSiteConfig(
            "HPC1",
            "HPC1",
            hpc_format,
            Path(hpc_v1_lfp_path),
            Path(hpc_v1_aligned_spike_path) if hpc_v1_aligned_spike_path else None,
            "ProbeB",
            222,
            "uV",
            2500.0,
        ),
        lfp_summary_webapp.LFPSiteConfig(
            "HPC2",
            "HPC2",
            hpc_format,
            Path(hpc_v1_lfp_path),
            Path(hpc_v1_aligned_spike_path) if hpc_v1_aligned_spike_path else None,
            "ProbeB",
            14,
            "uV",
            2500.0,
        ),
    )
    child_kwargs: dict[str, object] = {
        "session_id": sess_id_full,
        "session_path": session_path,
        "output_directory": session_path / "processed" / "lfp_summary_cache",
        "sites": sites,
        "site_pairs": (("PFC", "HPC1"), ("PFC", "HPC2"), ("HPC1", "HPC2")),
        "dependencies": dependencies,
    }
    if hpc_v1_sorter_output_path is not None or pfc_sorter_output_path is not None:
        sorter_paths: dict[str, Path] = {}
        aligned_spike_paths: dict[str, Path] = {}
        if pfc_sorter_output_path and pfc_sorter_output_path.strip():
            sorter_paths["ProbeA"] = Path(pfc_sorter_output_path)
        if hpc_v1_sorter_output_path and hpc_v1_sorter_output_path.strip():
            sorter_paths["ProbeB"] = Path(hpc_v1_sorter_output_path)
        if pfc_aligned_spike_path and pfc_aligned_spike_path.strip():
            aligned_spike_paths["ProbeA"] = Path(pfc_aligned_spike_path)
        if hpc_v1_aligned_spike_path and hpc_v1_aligned_spike_path.strip():
            aligned_spike_paths["ProbeB"] = Path(hpc_v1_aligned_spike_path)
        child_kwargs.update(
            sorter_paths=sorter_paths,
            aligned_spike_paths=aligned_spike_paths,
            cluster_metadata_loader=load_summary_cluster_metadata,
            channel_metadata_loader=load_summary_channel_metadata,
        )
    lfp_summary_webapp.render_lfp_summary_view(
        st,
        **child_kwargs,
    )

def _summary_lfp_format(lfp_path: str) -> str:
    """Infer the supported LFP loader family from an active recording filename.

    Parameters
    ----------
    lfp_path : str
        Active continuous LFP path. ``lfp.dat`` denotes an Open Ephys-derived
        recording; all other paths retain the existing SpikeGLX route.

    Returns
    -------
    str
        ``"open_ephys"`` for ``lfp.dat`` and ``"spikeglx"`` otherwise.
    """
    return "open_ephys" if Path(lfp_path).name == "lfp.dat" else "spikeglx"
