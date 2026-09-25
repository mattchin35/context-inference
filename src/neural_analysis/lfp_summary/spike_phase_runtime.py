"""Spike preparation and PPC payload construction for LFP summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.neural_analysis.lfp_summary import ppc_execution as lfp_summary_ppc_runtime
from src.neural_analysis.lfp import phase as lfp_phase_clustering
from src.neural_analysis.lfp_summary.cache import load_or_initialize_manifest, write_component_transaction
from src.neural_analysis.lfp_summary.models import LFPSiteConfig, LFPSummaryConfig, ProgressEvent
from src.neural_analysis.lfp_summary.payloads import build_component_payload
from src.neural_analysis.lfp_summary.pipeline import (
    ComponentPayload,
    PPCWorkCleanupTarget,
    PipelineDependencies,
)
from src.neural_analysis.lfp_summary.preparation import (
    TrialRelativeSpikeTrains,
    build_trial_relative_spike_trains,
)
from src.neural_analysis.lfp_summary.runtime_common import (
    PreparedPhaseRun,
    PreparedSpikeRun,
    _analysis_condition_membership,
    _validate_prepared_phase_run,
    _validate_prepared_spike_run,
)
from src.neural_analysis.lfp_summary.synchrony_runtime import (
    _band_hilbert_traces,
    _epoch_windows,
    prepare_phase_run,
)
from src.neural_analysis.lfp_summary.work_cache import cleanup_ppc_run
from src.neural_analysis.spike_behavior import loading as spike_behavior_pynapple
from src.neural_analysis.spike_behavior import loading as unit_spike_loading
from src.neural_analysis.spike_lfp import ppc as spike_lfp_summary


@dataclass(frozen=True)
class _PPCPhaseJob:
    """One selected site/condition/epoch phase view for the PPC executor.

    ``phase_tensor`` and ``phase_valid`` have axes ``(site=1, frequency,
    selected_trial, time)``. Unit phase is complex64 and validity is Boolean;
    ``relative_time_s`` is the shared float64 event-relative seconds axis.
    ``trial_indices`` are stable int64 trial-table rows. The categorical job
    identity is deliberately retained after selecting the full phase tensor so
    work checkpoints cannot cross condition/site/epoch boundaries.
    """

    phase_tensor: np.ndarray
    phase_valid: np.ndarray
    relative_time_s: np.ndarray
    trial_indices: np.ndarray
    site_id: str
    condition_name: str
    epoch_bounds_s: tuple[float, float]
    source_fingerprint: str


def load_configured_unit_spikes(
    config: LFPSummaryConfig,
) -> dict[str, np.ndarray]:
    """Load synchronized absolute spike times for the configured stable units.

    Parameters
    ----------
    config : LFPSummaryConfig
        Configuration whose unit population provides sorter and aligned-spike
        paths plus stable ``"probe:cluster"`` identities. Times are seconds.

    Returns
    -------
    dict[str, numpy.ndarray]
        Configured stable ids mapped to owned finite absolute seconds
        ``(spike_for_unit,)`` arrays, in the configured identity set.

    Raises
    ------
    ValueError
        If population paths, identities, aligned axes, metadata channels, or
        requested units are inconsistent. No partial mapping is returned.
    """
    population = config.unit_population
    if population is None:
        raise ValueError("Spike phase requires config.unit_population")
    if population.sorter_path is None or population.aligned_spike_path is None:
        raise ValueError("Spike phase requires sorter and aligned-spike paths")
    spike_clusters, cluster_info = spike_behavior_pynapple.load_sorter_metadata(
        population.sorter_path
    )
    aligned_spikes = spike_behavior_pynapple.load_aligned_spikes(
        population.aligned_spike_path
    )
    spike_behavior_pynapple.validate_aligned_spike_inputs(
        aligned_spikes,
        spike_clusters,
    )
    quality = dict(population.quality_settings)
    labels_text = quality.get("quality_labels", "good,mua")
    quality_labels = tuple(
        value.strip() for value in labels_text.split(",") if value.strip()
    )
    filtered = unit_spike_loading.filter_cluster_metadata(
        cluster_info,
        population.selected_channels,
        quality_labels=quality_labels,
        default_group=quality.get("default_group", "mua"),
        quality_column=quality.get("quality_column", "group"),
    )
    permitted_clusters = set(filtered["cluster_id"].astype(int).tolist())
    loaded: dict[str, np.ndarray] = {}
    for unit_id in population.stable_unit_ids:
        probe_label, cluster_id = _split_stable_unit_id(unit_id)
        if probe_label != population.probe_label or cluster_id not in permitted_clusters:
            raise ValueError(f"configured unit is outside active population: {unit_id}")
        values = np.asarray(aligned_spikes[spike_clusters == cluster_id], dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"unit spike times must be finite seconds: {unit_id}")
        loaded[unit_id] = values.copy()
    return loaded


def prepare_spike_run(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    unit_spike_loader: Callable[[LFPSummaryConfig], Mapping[str, np.ndarray]],
) -> PreparedSpikeRun:
    """Assign configured absolute unit spikes to full-axis trial windows.

    Parameters
    ----------
    config : LFPSummaryConfig
        Stable unit identities and half-open whole event window in seconds.
    prepared_phase : PreparedPhaseRun
        Full trial row axis plus absolute alignment seconds. Missing alignment
        rows own empty spike trains and remain in the cache trial axis.
    unit_spike_loader : callable
        Called once with ``config`` and returns stable ids mapped to finite
        absolute seconds ``(spike_for_unit,)`` arrays.

    Returns
    -------
    PreparedSpikeRun
        Configured unit order and one event-relative seconds array per unit and
        trial. Overlapping trial windows retain independent spike ownership.
    """
    _validate_prepared_phase_run(config, prepared_phase)
    population = config.unit_population
    if population is None or not population.stable_unit_ids:
        raise ValueError("Spike phase requires at least one configured stable unit")
    loaded = unit_spike_loader(config)
    if not isinstance(loaded, Mapping):
        raise ValueError("unit_spike_loader must return a stable-id mapping")
    if set(loaded) != set(population.stable_unit_ids):
        raise ValueError("loaded unit identities must exactly match configuration")
    event_times = np.asarray(prepared_phase.alignment_times_s, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(event_times))
    whole_window = (
        config.analysis_windows.whole_start_s,
        config.analysis_windows.whole_stop_s,
    )
    trains = []
    for unit_id in population.stable_unit_ids:
        probe_label, cluster_id = _split_stable_unit_id(unit_id)
        finite_train = build_trial_relative_spike_trains(
            probe_label=probe_label,
            cluster_id=cluster_id,
            unit_spike_times_s=np.asarray(loaded[unit_id], dtype=float),
            event_times_s=event_times[finite_positions],
            window=whole_window,
        )
        relative = [np.empty(0, dtype=float) for _ in event_times]
        for position, values in zip(
            finite_positions,
            finite_train.relative_spike_times,
            strict=True,
        ):
            relative[int(position)] = values.copy()
        overlap = finite_positions[finite_train.overlap_trial_indices]
        trains.append(
            TrialRelativeSpikeTrains(
                unit_id=unit_id,
                relative_spike_times=tuple(relative),
                overlap_trial_indices=overlap.astype(np.int64, copy=True),
            )
        )
    return PreparedSpikeRun(
        unit_ids=tuple(population.stable_unit_ids),
        population_ids=(population.label,),
        trial_spike_trains=tuple(trains),
    )


def build_spike_phase_payload(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
) -> ComponentPayload:
    """Build the frozen public Spike-phase payload without progress injection."""
    return _build_spike_phase_payload(
        config,
        prepared_phase,
        prepared_spikes,
        progress_callback=None,
    )


def _build_spike_phase_payload(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
    *,
    progress_callback: Callable[[ProgressEvent], None] | None,
) -> ComponentPayload:
    """Compute the complete observed and shuffled Spike-phase cache payload.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable frequency, half-open epoch, band, reliability, FDR, shuffle,
        phase-bin, site, filter, and seed settings. Times are seconds and
        frequencies are Hz.
    prepared_phase : PreparedPhaseRun
        Full-axis complex unit phase and 500-Hz source traces with explicit
        numerical validity on ``(site, frequency, trial, time)`` axes.
    prepared_spikes : PreparedSpikeRun
        Stable units and one event-relative seconds array per unit/trial.
    progress_callback : callable or None
        Internal pipeline seam receiving executor ``ProgressEvent`` objects.
        The frozen public three-argument builder always supplies ``None``.

    Returns
    -------
    ComponentPayload
        Exact ``SPIKE_PHASE_ARRAY_SCHEMA`` arrays. PPC is dimensionless, phase
        is radians, counts retain spike/trial/shuffle units, and no shuffle or
        wavelet tensor is persisted.

    Raises
    ------
    ValueError
        If prepared phase/spike identities or axes disagree with configuration.
    """
    _validate_prepared_phase_run(config, prepared_phase)
    _validate_prepared_spike_run(config, prepared_phase, prepared_spikes)
    frequencies_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    condition_membership = _analysis_condition_membership(
        prepared_phase.prepared_trials
    )
    epoch_windows = _selected_ppc_epoch_windows(config)
    epoch_names = tuple(epoch_windows)
    band_names = tuple(band.name for band in config.phase.bands)
    unit_count = len(prepared_spikes.unit_ids)
    condition_count = condition_membership.shape[1]
    site_count = len(config.sites)
    epoch_count = len(epoch_names)
    frequency_count = frequencies_hz.size
    metric_shape = (
        unit_count,
        condition_count,
        site_count,
        epoch_count,
        frequency_count,
    )
    phase_bin_edges = np.asarray(config.ppc.phase_bin_edges_rad, dtype=float)
    execution = lfp_summary_ppc_runtime.execute_grouped_ppc_component(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=prepared_phase,
        prepared_spikes=prepared_spikes,
        work_root=config.output_directory.parent / "lfp_summary_work",
        progress_callback=progress_callback,
    )
    arrays = _copy_grouped_ppc_component_summary(
        execution.summary_arrays,
        metric_shape=metric_shape,
        phase_bin_count=len(config.ppc.phase_bin_edges_rad) - 1,
    )
    trial_spike_counts = _grouped_exemplar_trial_spike_counts(
        prepared_phase=prepared_phase,
        prepared_spikes=prepared_spikes,
        condition_membership=condition_membership,
        epoch_windows=epoch_windows,
    )
    completed_ppc_runs = (
        [(Path(execution.run_directory), execution.run_fingerprint)]
        if execution.run_directory.exists()
        else []
    )
    arrays["ppc_band_mean"] = spike_lfp_summary.compute_band_ppc_means(
        frequencies_hz=frequencies_hz,
        ppc_by_frequency=arrays["ppc"],
    ).ppc_band_mean
    (
        selected_low,
        selected_high,
        illustrative,
        illustrative_low,
        illustrative_high,
    ) = _select_spike_exemplars(
        config,
        prepared_phase,
        prepared_spikes,
        arrays["ppc_band_mean"],
        trial_spike_counts,
    )
    packed_spikes, spike_offsets = _pack_trial_spike_trains(prepared_spikes)
    filtered_trace, hilbert_phase_rad = _band_hilbert_traces(
        config,
        prepared_phase,
    )
    arrays.update(
        {
            "unit_ids": np.asarray(prepared_spikes.unit_ids, dtype="<U64"),
            "population_ids": np.asarray(
                prepared_spikes.population_ids, dtype="<U64"
            ),
            "trial_indices": prepared_phase.trial_indices.astype(
                np.int64, copy=True
            ),
            "site_ids": np.asarray(
                [site.stable_id for site in config.sites], dtype="<U64"
            ),
            "site_voltage_units": np.asarray(
                [site.voltage_unit for site in config.sites], dtype="<U64"
            ),
            "condition_names": np.asarray(
                prepared_phase.prepared_trials.condition_names, dtype="<U64"
            ),
            "condition_membership": (
                prepared_phase.prepared_trials.condition_membership.copy()
            ),
            "filter_membership": (
                prepared_phase.prepared_trials.filter_membership.copy()
            ),
            "epoch_names": np.asarray(epoch_names, dtype="<U16"),
            "band_names": np.asarray(band_names, dtype="<U64"),
            "frequency_hz": frequencies_hz.copy(),
            "relative_time_s": prepared_phase.relative_time_s.copy(),
            "phase_bin_edges_rad": phase_bin_edges.copy(),
            "relative_spike_times_s": packed_spikes,
            "relative_spike_time_offsets": spike_offsets,
            "source_trace": prepared_phase.source_trace.copy(),
            "band_filtered_trace": filtered_trace,
            "hilbert_phase_rad": hilbert_phase_rad,
            "selected_low_unit_ids": selected_low,
            "selected_high_unit_ids": selected_high,
            "illustrative_low_trial_indices": illustrative_low,
            "illustrative_high_trial_indices": illustrative_high,
            "illustrative_trial_indices": illustrative,
        }
    )
    payload = build_component_payload("spike_phase", arrays)
    cleanup_targets = _ppc_cleanup_targets(config, completed_ppc_runs)
    cleanup = _ppc_post_commit_cleanup(config, completed_ppc_runs)
    component_plan = getattr(execution, "component_plan", None)
    allocation = getattr(component_plan, "allocation_estimate", None)
    execution_metadata = {
        "ppc_planning_seconds": getattr(execution, "planning_seconds", None),
        "grouped_execution_seconds": getattr(
            execution,
            "grouped_execution_seconds",
            None,
        ),
        "requested_worker_count": int(config.ppc_execution.worker_count),
        "planner_active_worker_count": getattr(allocation, "active_worker_count", None),
        "planned_parent_private_bytes": getattr(
            allocation,
            "planned_parent_private_bytes",
            None,
        ),
        "planned_worker_private_bytes": getattr(
            allocation,
            "planned_worker_private_bytes",
            None,
        ),
        "planned_aggregate_array_bytes": getattr(
            allocation,
            "planned_aggregate_array_bytes",
            None,
        ),
        "shared_phase_mmap_bytes": getattr(
            allocation,
            "shared_phase_mmap_bytes",
            None,
        ),
        "scheduled_edge_count": getattr(component_plan, "scheduled_edge_count", None),
        "independent_edge_count": getattr(
            component_plan,
            "independent_edge_count",
            None,
        ),
        "union_edge_count": getattr(component_plan, "union_edge_count", None),
        "completed_block_count": len(getattr(execution, "completed_block_ids", ())),
        "resumed_block_count": len(getattr(execution, "resumed_block_ids", ())),
        "prepared_phase_cache_state": prepared_phase.prepared_phase_cache_state,
        "run_fingerprint": execution.run_fingerprint,
        "run_directory": str(execution.run_directory),
    }
    return ComponentPayload(
        arrays=payload.arrays,
        manifest_entry=payload.manifest_entry,
        post_commit_cleanup=cleanup,
        post_commit_cleanup_targets=cleanup_targets,
        execution_metadata=execution_metadata,
    )


def make_spike_phase_pipeline_dependencies(
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
    """Bind real phase/spike preparation, PPC payload, and atomic cache I/O.

    Parameters
    ----------
    trial_table_loader, unit_spike_loader : callable
        Trial-table and stable-id-to-absolute-seconds spike loader seams.
    phase_preparer : callable or None
        Optional complete phase seam for tests or shared upstream preparation.
        ``None`` delegates to :func:`prepare_phase_run` with the remaining LFP
        seams.
    site_phase_tensor_builder, block_loader_factory : callable or None
        Bounded continuous phase-transform seams accepted by phase preparation.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized per-trial source loaders for exemplar cache traces.

    Returns
    -------
    PipelineDependencies
        Spike-only production dependencies. Power and Synchrony payload seams
        reject use rather than creating placeholder arrays.
    """
    active_config: LFPSummaryConfig | None = None

    def prepare_phase(config: LFPSummaryConfig) -> PreparedPhaseRun:
        """Prepare phase with the explicit seam or the production adapters."""
        nonlocal active_config
        active_config = config
        if phase_preparer is not None:
            return phase_preparer(config)
        return prepare_phase_run(
            config,
            trial_table_loader,
            site_phase_tensor_builder=site_phase_tensor_builder,
            block_loader_factory=block_loader_factory,
            spikeglx_loader=spikeglx_loader,
            open_ephys_loader=open_ephys_loader,
            work_cache_root=config.output_directory.parent / "lfp_summary_work",
        )

    def prepare_spike(
        config: LFPSummaryConfig,
        phase: object,
    ) -> PreparedSpikeRun:
        """Prepare configured trial-local spikes from the exact phase trial axis."""
        if not isinstance(phase, PreparedPhaseRun):
            raise ValueError("Spike runtime requires PreparedPhaseRun")
        return prepare_spike_run(config, phase, unit_spike_loader)

    def load_manifest(directory: Path) -> dict[str, object]:
        """Load the active configuration's JSON-ready cache manifest."""
        if active_config is None:
            raise RuntimeError("Spike manifest loading requires prepared configuration")
        return load_or_initialize_manifest(directory, active_config)

    def unsupported_power(_: LFPSummaryConfig) -> object:
        """Reject unbound Power preparation without fake arrays."""
        raise NotImplementedError("power is unsupported by Spike-phase runtime")

    def unsupported_power_payload(
        _: LFPSummaryConfig,
        __: object,
    ) -> ComponentPayload:
        """Reject unbound Power payload construction."""
        raise NotImplementedError("power payload is unsupported by Spike-phase runtime")

    def unsupported_synchrony_payload(
        _: LFPSummaryConfig,
        __: object,
    ) -> ComponentPayload:
        """Reject unbound Synchrony payload construction."""
        raise NotImplementedError(
            "synchrony payload is unsupported by Spike-phase runtime"
        )

    def build_spike_payload_with_progress(
        config: LFPSummaryConfig,
        phase: object,
        spikes: object,
        progress_callback: Callable[[ProgressEvent], None] | None,
    ) -> ComponentPayload:
        """Build Spike phase while forwarding pipeline progress to PPC jobs."""
        if not isinstance(phase, PreparedPhaseRun) or not isinstance(
            spikes,
            PreparedSpikeRun,
        ):
            raise ValueError("Spike progress payload requires prepared runtime products")
        return _build_spike_phase_payload(
            config,
            phase,
            spikes,
            progress_callback=progress_callback,
        )

    return PipelineDependencies(
        prepare_power=unsupported_power,
        prepare_phase=prepare_phase,
        prepare_spike=prepare_spike,
        build_power_payload=unsupported_power_payload,
        build_synchrony_payload=unsupported_synchrony_payload,
        build_spike_phase_payload=build_spike_phase_payload,
        load_manifest=load_manifest,
        write_component=write_component_transaction,
        build_spike_phase_payload_with_progress=build_spike_payload_with_progress,
    )


def _split_stable_unit_id(unit_id: str) -> tuple[str, int]:
    """Parse one ``probe:cluster`` identity into categorical components."""
    try:
        probe_label, cluster_text = unit_id.rsplit(":", maxsplit=1)
        cluster_id = int(cluster_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid stable unit identity: {unit_id!r}") from error
    if not probe_label or cluster_id < 0 or f"{probe_label}:{cluster_id}" != unit_id:
        raise ValueError(f"invalid stable unit identity: {unit_id!r}")
    return probe_label, cluster_id


def _selected_ppc_epoch_windows(
    config: LFPSummaryConfig,
) -> dict[str, tuple[float, float]]:
    """Return configured PPC epochs in declared order with seconds bounds."""
    available = _epoch_windows(config)
    try:
        return {name: available[name] for name in config.ppc.epochs}
    except KeyError as error:
        raise ValueError("PPC epochs must name whole, before, or after") from error


def _spikes_in_epoch(
    relative_spike_times_s: np.ndarray,
    epoch_window: tuple[float, float],
) -> np.ndarray:
    """Return an owned half-open epoch selection from one trial in seconds."""
    spikes = np.asarray(relative_spike_times_s, dtype=float)
    start_s, stop_s = epoch_window
    return spikes[(spikes >= start_s) & (spikes < stop_s)].copy()


def _shared_derangement_schedule(
    trial_count: int,
    shuffle_count: int,
    seed: int,
) -> np.ndarray | None:
    """Return one unit-shared trial derangement or None below two trials."""
    if trial_count < 2:
        return None
    return spike_lfp_summary.generate_trial_derangement_schedule(
        trial_count,
        shuffle_count,
        seed=seed,
    )


def _ppc_schedule_seed(
    config: LFPSummaryConfig,
    condition_index: int,
    site_index: int,
    epoch_index: int,
) -> int:
    """Derive a deterministic condition/site/epoch seed without unit identity."""
    return int(
        config.ppc.seed
        + condition_index * 1_000_000
        + site_index * 10_000
        + epoch_index * 100
    )


def _copy_grouped_ppc_component_summary(
    summary_arrays: Mapping[str, np.ndarray],
    *,
    metric_shape: tuple[int, int, int, int, int],
    phase_bin_count: int,
) -> dict[str, np.ndarray]:
    """Copy grouped PPC outputs into the frozen public payload field names.

    Parameters
    ----------
    summary_arrays : Mapping[str, numpy.ndarray]
        Grouped executor arrays with metric axes ``(unit, condition, site,
        epoch, frequency)`` and fixed representative histogram axes ``(unit,
        condition, site, epoch, band=2, phase_bin)``. Float metrics are
        dimensionless except preferred phase in radians; counts and flags
        retain their documented public units.
    metric_shape : tuple[int, int, int, int, int]
        Exact final ``(unit, condition, site, epoch, frequency)`` dimensions.
    phase_bin_count : int
        Positive number of configured radians histogram bins.

    Returns
    -------
    dict[str, numpy.ndarray]
        Writable copies for final payload assembly. The histogram key is
        renamed from the executor-private
        ``representative_phase_histogram_count`` to the frozen public
        ``representative_phase_hist_count`` without changing axes or dtype.

    Raises
    ------
    ValueError
        If the executor omits a required field or returns an incompatible
        public axis/dtype contract.
    """
    float_fields = (
        "ppc",
        "resultant_length",
        "preferred_phase_rad",
        "p_value",
        "q_value",
        "null_mean",
        "null_std",
        "null_p025",
        "null_p50",
        "null_p975",
    )
    integer_fields = (
        "spike_count",
        "eligible_trial_count",
        "null_exceedance_count",
        "permutation_count",
    )
    boolean_fields = ("computable", "reliable", "null_eligible", "significant")
    arrays: dict[str, np.ndarray] = {}
    for name in float_fields + integer_fields + boolean_fields:
        if name not in summary_arrays:
            raise ValueError(f"grouped PPC summary lacks {name}")
        values = np.asarray(summary_arrays[name])
        if values.shape != metric_shape:
            raise ValueError("grouped PPC summary axes disagree with final payload")
        expected_dtype = (
            np.dtype(float)
            if name in float_fields
            else np.dtype(np.int64)
            if name in integer_fields
            else np.dtype(bool)
        )
        if values.dtype != expected_dtype:
            raise ValueError("grouped PPC summary dtype disagrees with final payload")
        arrays[name] = values.copy()
    histogram_name = "representative_phase_histogram_count"
    if histogram_name not in summary_arrays:
        raise ValueError("grouped PPC summary lacks representative histogram counts")
    histogram = np.asarray(summary_arrays[histogram_name])
    expected_histogram_shape = metric_shape[:-1] + (2, phase_bin_count)
    if (
        histogram.shape != expected_histogram_shape
        or histogram.dtype != np.dtype(np.int64)
    ):
        raise ValueError("grouped representative histogram axes disagree with final payload")
    arrays["representative_phase_hist_count"] = histogram.copy()
    return arrays


def _grouped_exemplar_trial_spike_counts(
    *,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
    condition_membership: np.ndarray,
    epoch_windows: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """Count phase-free selected spikes for frozen exemplar selection.

    Parameters
    ----------
    prepared_phase : PreparedPhaseRun
        Full stable trial and Boolean site-valid axes. No phase values are
        sampled by this helper.
    prepared_spikes : PreparedSpikeRun
        Full-unit ordered finite relative spike times in seconds for every
        trial.
    condition_membership : numpy.ndarray
        Boolean ``(trial, condition)`` analysis membership after filter,
        objective-validity, and user-exclusion gates.
    epoch_windows : Mapping[str, tuple[float, float]]
        Ordered half-open epoch seconds bounds, with keys defining the final
        epoch axis.

    Returns
    -------
    numpy.ndarray
        Int64 spike counts on ``(unit, condition, site, epoch, trial)`` axes.
        Nonselected rows remain zero. This preserves the legacy exemplar
        selection input without phase interpolation or histogram sampling.
    """
    counts = np.zeros(
        (
            len(prepared_spikes.unit_ids),
            condition_membership.shape[1],
            len(prepared_phase.site_valid),
            len(epoch_windows),
            prepared_phase.trial_indices.size,
        ),
        dtype=np.int64,
    )
    for condition_index in range(condition_membership.shape[1]):
        for site_index in range(len(prepared_phase.site_valid)):
            selected_positions = np.flatnonzero(
                condition_membership[:, condition_index]
                & prepared_phase.site_valid[site_index]
            )
            for epoch_index, epoch_window in enumerate(epoch_windows.values()):
                for unit_index, train in enumerate(prepared_spikes.trial_spike_trains):
                    for position in selected_positions:
                        counts[
                            unit_index,
                            condition_index,
                            site_index,
                            epoch_index,
                            position,
                        ] = _spikes_in_epoch(
                            train.relative_spike_times[int(position)], epoch_window
                        ).size
    return counts


def _ppc_post_commit_cleanup(
    config: LFPSummaryConfig,
    completed_runs: list[tuple[Path, str]],
) -> Callable[[], None] | None:
    """Return exact completed-run cleanup for the approved retention policy."""
    if (
        not config.ppc_execution.checkpoint_enabled
        or config.ppc_execution.checkpoint_retention != "incomplete_only"
        or not completed_runs
    ):
        return None
    exact_runs = tuple(dict.fromkeys(completed_runs))

    def cleanup() -> None:
        """Remove only executor runs that produced this successfully committed payload."""
        for run_directory, run_fingerprint in exact_runs:
            cleanup_ppc_run(run_directory, run_fingerprint)

    return cleanup


def _ppc_cleanup_targets(
    config: LFPSummaryConfig,
    completed_runs: list[tuple[Path, str]],
) -> tuple[PPCWorkCleanupTarget, ...]:
    """Return exact JSON-safe work identities for deferred launcher cleanup.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable execution policy; no numerical arrays are inspected.
    completed_runs : list[tuple[pathlib.Path, str]]
        PPC work directories and exact SHA-256 run fingerprints that produced
        the committed payload.

    Returns
    -------
    tuple[PPCWorkCleanupTarget, ...]
        Stable de-duplicated targets, or empty when cleanup is disabled/retained.
    """
    if (
        not config.ppc_execution.checkpoint_enabled
        or config.ppc_execution.checkpoint_retention != "incomplete_only"
    ):
        return ()
    return tuple(
        PPCWorkCleanupTarget(Path(run_directory), run_fingerprint)
        for run_directory, run_fingerprint in dict.fromkeys(completed_runs)
    )


def _sample_observed_trial_phase(
    phase_time_s: np.ndarray,
    trial_phase: np.ndarray,
    trial_valid: np.ndarray,
    trial_spikes: tuple[np.ndarray, ...],
    trial_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample complex phase without bridging invalid coordinates.

    Parameters
    ----------
    phase_time_s : numpy.ndarray
        Exact finite relative seconds ``(time,)`` grid.
    trial_phase, trial_valid : numpy.ndarray
        Complex and boolean ``(trial, frequency, time)`` arrays.
    trial_spikes : tuple[numpy.ndarray, ...]
        One event-relative seconds array per selected trial.
    trial_rows : numpy.ndarray
        Integer full-table row identity ``(trial,)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Complex samples and validity ``(frequency, spike)`` plus integer source
        trial rows ``(spike,)``. Invalid samples are zero/false.
    """
    time_s = np.asarray(phase_time_s, dtype=float)
    phase = np.asarray(trial_phase)
    valid = np.asarray(trial_valid, dtype=bool)
    rows = np.asarray(trial_rows)
    expected = (len(trial_spikes), phase.shape[1], time_s.size)
    if phase.ndim != 3 or phase.shape != expected or valid.shape != phase.shape:
        raise ValueError("trial phase sampling axes are inconsistent")
    if rows.shape != (len(trial_spikes),):
        raise ValueError("trial rows must match sampled spike trains")
    spike_count = sum(values.size for values in trial_spikes)
    sampled = np.zeros((phase.shape[1], spike_count), dtype=np.complex64)
    sampled_valid = np.zeros(sampled.shape, dtype=bool)
    source_rows = np.empty(spike_count, dtype=np.int64)
    offset = 0
    for trial_index, spikes in enumerate(trial_spikes):
        for spike in np.asarray(spikes, dtype=float):
            source_rows[offset] = int(rows[trial_index])
            _sample_one_phase_time(
                time_s,
                phase[trial_index],
                valid[trial_index],
                float(spike),
                sampled[:, offset],
                sampled_valid[:, offset],
            )
            offset += 1
    return sampled, sampled_valid, source_rows


def _sample_one_phase_time(
    time_s: np.ndarray,
    phase: np.ndarray,
    valid: np.ndarray,
    spike_time_s: float,
    output: np.ndarray,
    output_valid: np.ndarray,
) -> None:
    """Interpolate one phase column only between two valid neighboring samples."""
    right = int(np.searchsorted(time_s, spike_time_s, side="left"))
    if right < time_s.size and np.isclose(
        time_s[right], spike_time_s, rtol=0.0, atol=1e-12
    ):
        usable = valid[:, right]
        values = phase[:, right]
    elif right == 0 or right == time_s.size:
        return
    else:
        left = right - 1
        usable = valid[:, left] & valid[:, right]
        fraction = (spike_time_s - time_s[left]) / (time_s[right] - time_s[left])
        values = phase[:, left] + fraction * (phase[:, right] - phase[:, left])
    magnitude = np.abs(values)
    usable &= np.isfinite(values.real) & np.isfinite(values.imag) & (magnitude > 0.0)
    output[usable] = values[usable] / magnitude[usable]
    output_valid[usable] = True


def _pack_trial_spike_trains(
    prepared: PreparedSpikeRun,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack unit/trial relative seconds into one vector and global offsets."""
    trial_count = len(prepared.trial_spike_trains[0].relative_spike_times)
    offsets = np.zeros((len(prepared.unit_ids), trial_count + 1), dtype=np.int64)
    chunks: list[np.ndarray] = []
    running = 0
    for unit_index, train in enumerate(prepared.trial_spike_trains):
        offsets[unit_index, 0] = running
        for trial_index, values in enumerate(train.relative_spike_times):
            chunks.append(np.asarray(values, dtype=float).copy())
            running += values.size
            offsets[unit_index, trial_index + 1] = running
    packed = np.concatenate(chunks) if chunks else np.empty(0, dtype=float)
    return packed, offsets


def _select_spike_exemplars(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
    ppc_band_mean: np.ndarray,
    trial_spike_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select pooled low/high units and their exact illustrative trials.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable condition, site, epoch, and band axis definitions.
    prepared_phase : PreparedPhaseRun
        Full int64 trial-table row axis shared by cached phase/traces.
    prepared_spikes : PreparedSpikeRun
        Stable unit identities and event-relative spike seconds per trial.
    ppc_band_mean : numpy.ndarray
        Dimensionless float array with axes
        ``(unit, condition, site, epoch, band)``.
    trial_spike_counts : numpy.ndarray
        Nonnegative int64 array with axes
        ``(unit, condition, site, epoch, trial)`` in spike units.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Low/high stable-unit-id arrays followed by the legacy high-first trial,
        exact low-unit trial, and exact high-unit trial arrays. Every output has
        ``(condition, site, epoch, band)`` axes. Trial values are table-row ids;
        unavailable ids use ``""`` and unavailable trials use ``-1``.
    """
    output_shape = ppc_band_mean.shape[1:]
    low = np.full(output_shape, "", dtype="<U64")
    high = np.full(output_shape, "", dtype="<U64")
    illustrative = np.full(output_shape, -1, dtype=np.int64)
    illustrative_low = np.full(output_shape, -1, dtype=np.int64)
    illustrative_high = np.full(output_shape, -1, dtype=np.int64)
    for condition_index, site_index, epoch_index, band_index in np.ndindex(
        output_shape
    ):
        selection = spike_lfp_summary.select_ppc_exemplars(
            unit_ids=prepared_spikes.unit_ids,
            band_ppc=ppc_band_mean[
                :, condition_index, site_index, epoch_index, band_index
            ],
            trial_indices=prepared_phase.trial_indices,
            trial_spike_counts=trial_spike_counts[
                :, condition_index, site_index, epoch_index
            ],
        )
        index = (condition_index, site_index, epoch_index, band_index)
        if selection.low_unit_id is not None:
            low[index] = selection.low_unit_id
        if selection.high_unit_id is not None:
            high[index] = selection.high_unit_id
        if selection.low_unit_id in selection.illustrative_trial_index_by_unit:
            illustrative_low[index] = selection.illustrative_trial_index_by_unit[
                selection.low_unit_id
            ]
        if selection.high_unit_id in selection.illustrative_trial_index_by_unit:
            illustrative_high[index] = selection.illustrative_trial_index_by_unit[
                selection.high_unit_id
            ]
        for unit_id in (selection.high_unit_id, selection.low_unit_id):
            if unit_id in selection.illustrative_trial_index_by_unit:
                illustrative[index] = selection.illustrative_trial_index_by_unit[unit_id]
                break
    return low, high, illustrative, illustrative_low, illustrative_high
