"""Composed and compatibility surface for LFP-summary component runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.neural_analysis.lfp import phase as lfp_phase_clustering
from src.neural_analysis.lfp_summary.cache import load_or_initialize_manifest, write_component_transaction
from src.neural_analysis.lfp_summary.models import LFPSiteConfig, LFPSummaryConfig, ProgressEvent
from src.neural_analysis.lfp_summary.pipeline import ComponentPayload, PipelineDependencies
from src.neural_analysis.lfp_summary.power_runtime import (
    _SitePowerResult,
    _assemble_power_arrays,
    _cached_source_traces,
    _common_frequency_grid,
    _compute_site_power_results,
    _effective_condition_counts,
    _exclusion_codes,
    _first_start_time,
    _interpolate_result,
    _load_presession_traces,
    _normalize_scalar_db,
    _validate_prepared_power_run,
    build_power_payload,
    make_power_pipeline_dependencies,
    prepare_power_run,
)
from src.neural_analysis.lfp_summary.runtime_common import (
    PreparedPhaseRun,
    PreparedPowerRun,
    PreparedSpikeRun,
    _CACHE_SAMPLE_RATE_HZ,
    _alignment_times,
    _analysis_condition_membership,
    _resample_trace_rows,
    _sample_count,
    _validate_composed_phase_amplitude_policy,
    _validate_prepared_phase_run,
    _validate_prepared_spike_run,
    _work_fingerprint,
    load_configured_trial_table,
)
from src.neural_analysis.lfp_summary.spike_phase_runtime import (
    _PPCPhaseJob,
    _build_spike_phase_payload,
    _copy_grouped_ppc_component_summary,
    _grouped_exemplar_trial_spike_counts,
    _pack_trial_spike_trains,
    _ppc_cleanup_targets,
    _ppc_post_commit_cleanup,
    _ppc_schedule_seed,
    _sample_observed_trial_phase,
    _sample_one_phase_time,
    _select_spike_exemplars,
    _selected_ppc_epoch_windows,
    _shared_derangement_schedule,
    _spikes_in_epoch,
    _split_stable_unit_id,
    build_spike_phase_payload,
    load_configured_unit_spikes,
    make_spike_phase_pipeline_dependencies,
    prepare_spike_run,
)
from src.neural_analysis.lfp_summary.synchrony_runtime import (
    _band_hilbert_traces,
    _cached_phase_axes_match_current_run,
    _cached_prepared_source_traces,
    _configured_pair_indices,
    _epoch_windows,
    _full_phase_axes,
    _preflighted_production_phase_block_loader_factory,
    _prepared_phase_work_metadata,
    _production_phase_block_loader_factory,
    _validate_site_phase_tensor,
    build_synchrony_payload,
    make_synchrony_pipeline_dependencies,
    prepare_phase_run,
)
from src.neural_analysis.lfp_summary.work_cache import (
    PreparedPhaseCache,
    cleanup_ppc_run,
    load_prepared_phase_cache,
    write_prepared_phase_cache,
)
from src.neural_analysis.spike_lfp import hilbert as spike_lfp_hilbert_phase
from src.neural_analysis.spike_lfp import ppc as spike_lfp_summary


def make_lfp_summary_pipeline_dependencies(
    *,
    trial_table_loader: Callable[[LFPSummaryConfig], pd.DataFrame],
    unit_spike_loader: Callable[
        [LFPSummaryConfig], Mapping[str, np.ndarray]
    ] = load_configured_unit_spikes,
    phase_preparer: Callable[[LFPSummaryConfig], PreparedPhaseRun] | None = None,
    site_phase_tensor_builder: Callable[..., lfp_phase_clustering.PhaseTrialTensor] = (
        lfp_phase_clustering.compute_site_phase_trial_tensor
    ),
    block_loader_factory: Callable[
        [LFPSiteConfig],
        Callable[[float, float], tuple[np.ndarray, np.ndarray, float]],
    ]
    | None = None,
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> PipelineDependencies:
    """Bind one production dependency bundle for every LFP-summary component.

    Parameters
    ----------
    trial_table_loader : callable
        Accepts one :class:`LFPSummaryConfig` and returns a trial table with one
        row per stable trial. Absolute event timestamps are seconds.
    unit_spike_loader : callable
        Accepts the same configuration and returns probe-qualified unit ids
        mapped to finite one-dimensional absolute spike-time arrays in seconds.
    phase_preparer : callable or None
        Optional complete configuration-to-:class:`PreparedPhaseRun` seam. Its
        phase and validity arrays have ``(site, frequency, trial, time)`` axes;
        frequency is Hz and event-relative time is seconds. ``None`` uses the
        production phase-preparation path and its validated work cache.
    site_phase_tensor_builder : callable
        Continuous-block Morlet seam used only when ``phase_preparer`` is
        ``None``. It preserves the configured site/frequency/trial/time axes.
    block_loader_factory : callable or None
        Optional site-specific continuous source loader factory. Returned
        traces retain their configured source voltage unit and time in seconds.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized per-trial LFP loaders. Each returns relative seconds,
        a one-dimensional source-voltage trace, and sample rate in Hz.

    Returns
    -------
    PipelineDependencies
        A single bundle supporting Power, Synchrony, Spike phase, and Compute
        All. Synchrony and Spike phase share the exact phase object supplied by
        the pipeline. Spike progress events retain all grouped-runtime fields.

    Raises
    ------
    ValueError
        Before any production work if absolute amplitude thresholds are
        nonempty, or if prepared phase/spike identities violate their contracts.
    RuntimeError
        If manifest loading is requested before a composed preparation call.

    Notes
    -----
    The bundle introduces no numerical transformations. It delegates to the
    established preparation, payload, and manifest-last transaction functions.
    """
    active_config: LFPSummaryConfig | None = None

    def bind_config(config: LFPSummaryConfig) -> None:
        """Validate supported settings and retain the exact active config."""
        nonlocal active_config
        _validate_composed_phase_amplitude_policy(config)
        active_config = config

    def prepare_power(config: LFPSummaryConfig) -> PreparedPowerRun:
        """Prepare Power inputs after fail-before-work policy validation."""
        bind_config(config)
        return prepare_power_run(
            config,
            trial_table_loader,
            spikeglx_loader,
            open_ephys_loader,
        )

    def prepare_phase(config: LFPSummaryConfig) -> PreparedPhaseRun:
        """Prepare or inject one full-axis phase record for shared reuse."""
        bind_config(config)
        prepared = (
            phase_preparer(config)
            if phase_preparer is not None
            else prepare_phase_run(
                config,
                trial_table_loader,
                site_phase_tensor_builder=site_phase_tensor_builder,
                block_loader_factory=block_loader_factory,
                spikeglx_loader=spikeglx_loader,
                open_ephys_loader=open_ephys_loader,
                work_cache_root=config.output_directory.parent / "lfp_summary_work",
            )
        )
        if not isinstance(prepared, PreparedPhaseRun):
            raise ValueError("composed phase preparation requires PreparedPhaseRun")
        return prepared

    def prepare_spike(
        config: LFPSummaryConfig,
        phase: object,
    ) -> PreparedSpikeRun:
        """Prepare trial-local unit spikes from the exact shared phase axis."""
        bind_config(config)
        if not isinstance(phase, PreparedPhaseRun):
            raise ValueError("composed Spike phase requires PreparedPhaseRun")
        return prepare_spike_run(config, phase, unit_spike_loader)

    def load_manifest(directory: Path) -> dict[str, object]:
        """Load metadata for the exact config accepted by preparation."""
        if active_config is None:
            raise RuntimeError("composed manifest loading requires preparation")
        return load_or_initialize_manifest(directory, active_config)

    def build_spike_payload_with_progress(
        config: LFPSummaryConfig,
        phase: object,
        spikes: object,
        progress_callback: Callable[[ProgressEvent], None] | None,
    ) -> ComponentPayload:
        """Forward grouped PPC progress without translating record fields."""
        if not isinstance(phase, PreparedPhaseRun) or not isinstance(
            spikes,
            PreparedSpikeRun,
        ):
            raise ValueError(
                "composed Spike payload requires prepared phase and spike records"
            )
        return _build_spike_phase_payload(
            config,
            phase,
            spikes,
            progress_callback=progress_callback,
        )

    return PipelineDependencies(
        prepare_power=prepare_power,
        prepare_phase=prepare_phase,
        prepare_spike=prepare_spike,
        build_power_payload=build_power_payload,
        build_synchrony_payload=build_synchrony_payload,
        build_spike_phase_payload=build_spike_phase_payload,
        load_manifest=load_manifest,
        write_component=write_component_transaction,
        build_spike_phase_payload_with_progress=build_spike_payload_with_progress,
    )
