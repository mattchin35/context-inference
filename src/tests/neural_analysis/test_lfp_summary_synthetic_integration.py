"""Seeded end-to-end cache and plotting validation on known LFP/spike signals."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.lfp_power_summary import (
    compute_session_reference_psd,
    compute_trial_epoch_psds,
    mean_band_power_linear,
    normalize_psd_db,
)
from src.neural_analysis.lfp_summary_io import (
    assess_component_status,
    load_component_arrays,
    load_or_initialize_manifest,
    write_component_transaction,
)
from src.neural_analysis.lfp_summary_models import (
    FrequencyBandConfig,
    PPCAnalysisConfig,
    PhaseAnalysisConfig,
    PowerAnalysisConfig,
    default_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_payloads import (
    COMPONENT_ARRAY_SCHEMAS,
    build_component_payload,
)
from src.neural_analysis.lfp_summary_pipeline import (
    PipelineDependencies,
    compute_all_components,
)
from src.neural_analysis.lfp_summary_plotting import (
    PlotContext,
    plot_band_power_summary,
    plot_condition_psd,
    plot_phase_band_summary,
    plot_phase_map,
    plot_plv_distribution,
    plot_plv_exemplar,
    plot_population_ppc_maps,
    plot_ppc_band_summary,
    plot_ppc_exemplar,
    plot_unit_ppc_map,
)
from src.neural_analysis.lfp_summary_preparation import (
    build_common_event_grid,
    build_prepared_trials,
)
from src.neural_analysis.lfp_synchrony_summary import (
    aggregate_trial_plv_bands,
    compute_phase_clustering_summary,
    compute_trial_plv_by_frequency,
)
from src.neural_analysis.spike_lfp_summary import (
    adjust_ppc_pvalues_bh,
    compute_band_ppc_means,
    compute_significant_prevalence,
    compute_trial_shuffle_ppc,
    generate_trial_derangement_schedule,
)


_FS_HZ = 500.0
_TIME_S = build_common_event_grid(-2.0, 2.0, _FS_HZ)
_FREQUENCY_HZ = np.arange(2.0, 42.0, 2.0)
_EPOCH_NAMES = ("whole", "before", "after")
_BAND_NAMES = ("theta", "gamma")
_CONDITION_NAMES = ("correct_rewarded", "omission", "all_trials")
_PHASE_OFFSETS_RAD = np.array((0.0, 0.3, 2.9, 3.6))
_SITE_OFFSET_RAD = 0.7


def _synthetic_configuration(tmp_path: Path):
    """Return a small validated configuration with a two-site synthetic cache.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory receiving the cache files.

    Returns
    -------
    LFPSummaryConfig
        Configuration with seconds, Hz, and uV contracts; only two sites and
        one ordered pair are retained for the compact integration calculation.
    """
    base = default_lfp_summary_config()
    sites = tuple(
        replace(
            site, lfp_path=tmp_path / f"{site.stable_id}.bin", sample_rate_hz=_FS_HZ
        )
        for site in base.sites[:2]
    )
    return replace(
        base,
        session_id="synthetic-lfp-summary",
        session_path=tmp_path,
        output_directory=tmp_path / "cache",
        sites=sites,
        site_pairs=((sites[0].stable_id, sites[1].stable_id),),
        phase=PhaseAnalysisConfig(
            frequency_hz=tuple(_FREQUENCY_HZ),
            output_rate_hz=_FS_HZ,
            bands=(
                FrequencyBandConfig("theta", 6.0, 10.0),
                FrequencyBandConfig("gamma", 30.0, 42.0),
            ),
            bootstrap_count=8,
            seed=71,
        ),
        power=PowerAnalysisConfig(
            notch_enabled=False,
            bands=(
                FrequencyBandConfig("theta", 6.0, 10.0),
                FrequencyBandConfig("gamma", 30.0, 42.0),
            ),
        ),
        ppc=PPCAnalysisConfig(shuffle_count=100, seed=72),
    )


def _synthetic_trials_and_traces():
    """Build four trial rows and two 500-Hz traces with 8/40-Hz power peaks.

    Returns
    -------
    tuple[pandas.DataFrame, numpy.ndarray]
        Trial table has shape ``(4, column)``; traces have shape
        ``(site=2, trial=4, time=2000)`` in uV. First two trials are locked
        correct trials and final two are oppositely phased omission trials.
    """
    trials = pd.DataFrame(
        {
            "choice_time": np.arange(4, dtype=float) * 10.0,
            "start_time": np.arange(4, dtype=float) * 10.0 - 3.0,
            "action": np.array((0, 1, 0, 1)),
            "state_int": np.array((0, 0, 1, 1)),
            "correct": np.array((1, 1, 0, 0)),
            "reward": np.array((1, 1, 0, 0)),
            "experimenter_reward_given": np.zeros(4, dtype=int),
            "switch": np.ones(4, dtype=bool),
        }
    )
    phase = 2.0 * np.pi * 8.0 * _TIME_S[np.newaxis, :] + _PHASE_OFFSETS_RAD[:, None]
    fast = 2.0 * np.pi * 40.0 * _TIME_S[np.newaxis, :] + _PHASE_OFFSETS_RAD[:, None]
    site_a = 4.0 * np.sin(phase) + 2.0 * np.sin(fast)
    site_b = 4.0 * np.sin(phase - _SITE_OFFSET_RAD) + 2.0 * np.sin(
        fast - _SITE_OFFSET_RAD
    )
    return trials, np.stack((site_a, site_b), axis=0)


def _epoch_windows() -> dict[str, tuple[float, float]]:
    """Return named half-open event-relative epoch bounds.

    Returns
    -------
    dict[str, tuple[float, float]]
        Three named ``(start_s, stop_s)`` pairs in seconds: whole ``[-2, 2)``,
        before ``[-2, 0)``, and after ``[0, 2)``. Bounds have no array axes.
    """
    return {"whole": (-2.0, 2.0), "before": (-2.0, 0.0), "after": (0.0, 2.0)}


def _phase_tensor() -> np.ndarray:
    """Return synthetic unit complex phase vectors on documented axes.

    Returns
    -------
    numpy.ndarray
        Complex128 shape ``(site=2, frequency=20, trial=4, time=2000)`` with
        dimensionless unit-magnitude values. Frequency coordinates are
        ``_FREQUENCY_HZ`` in Hz and time is ``_TIME_S`` in seconds. Per-trial
        offsets preserve high within-group ITPC but cancel for all trials; site
        A minus site B equals ``_SITE_OFFSET_RAD`` radians.
    """
    angular = 2.0 * np.pi * _FREQUENCY_HZ[:, None] * _TIME_S[None, :]
    trial_phase = angular[:, None, :] + _PHASE_OFFSETS_RAD[None, :, None]
    site_a = np.exp(1j * trial_phase)
    site_b = np.exp(1j * (trial_phase - _SITE_OFFSET_RAD))
    return np.stack((site_a, site_b), axis=0)


def _trial_spikes(locked: bool, seed: int) -> tuple[np.ndarray, ...]:
    """Return four trial-local spike arrays with 52 samples in the whole epoch.

    Parameters
    ----------
    locked : bool
        When true, compensate each trial phase offset so all sampled 8-Hz
        phases are zero; otherwise generate seeded, broadly distributed times.
    seed : int
        NumPy generator seed for the unlocked unit only.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        Four float64 ``(13,)`` arrays in event-relative seconds, each strictly
        inside ``[-2, 2)``. The concatenated spike axis has length 52.
    """
    base = np.arange(-12, 13, 2, dtype=float) / 8.0
    if locked:
        return tuple(
            base - offset / (2.0 * np.pi * 8.0) for offset in _PHASE_OFFSETS_RAD
        )
    generator = np.random.default_rng(seed)
    return tuple(np.sort(generator.uniform(-1.8, 1.8, 13)) for _ in range(4))


def _packed_spikes(spikes: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Pack one unit's trial-local seconds arrays into numeric cache axes.

    Parameters
    ----------
    spikes : tuple[numpy.ndarray, ...]
        Per-trial float64 ``(spike_in_trial,)`` seconds arrays.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Concatenated seconds ``(spike,)`` and integer offsets ``(trial + 1,)``.
    """
    offsets = np.concatenate(
        ([0], np.cumsum([values.size for values in spikes]))
    ).astype(np.int64)
    return np.concatenate(spikes), offsets


def _payload_arrays(component: str, sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """Allocate every mandatory payload array using its public named-axis schema.

    Parameters
    ----------
    component : str
        One of ``power``, ``synchrony``, or ``spike_phase``.
    sizes : dict[str, int]
        Positive axis lengths keyed by schema axis name.

    Returns
    -------
    dict[str, numpy.ndarray]
        Safe numeric, boolean, or fixed-Unicode arrays with each exact required
        shape. Values are placeholders until replaced by real synthetic outputs.
    """
    arrays: dict[str, np.ndarray] = {}
    for name, contract in COMPONENT_ARRAY_SCHEMAS[component].items():
        shape = tuple(sizes[axis] for axis in contract.axes)
        if contract.units == "boolean":
            arrays[name] = np.zeros(shape, dtype=bool)
        elif contract.units in {
            "trial",
            "sample",
            "spike",
            "shuffle",
            "spike-offset",
            "trial-table-row",
        }:
            arrays[name] = np.zeros(shape, dtype=np.int64)
        elif name in {
            "unit_ids",
            "population_ids",
            "site_ids",
            "site_voltage_units",
            "condition_names",
            "epoch_names",
            "band_names",
            "pair_site_a_ids",
            "pair_site_b_ids",
            "selected_low_unit_ids",
            "selected_high_unit_ids",
            "exclusion_reason_code",
        }:
            arrays[name] = np.full(shape, "", dtype="<U32")
        else:
            arrays[name] = np.full(shape, np.nan, dtype=float)
    return arrays


def _plot_context() -> PlotContext:
    """Return cache-plot provenance metadata without numerical data axes.

    Returns
    -------
    PlotContext
        Categorical session/alignment/reference labels, epoch bounds in seconds,
        gamma exclusion bounds in Hz, and source voltage unit ``uV``. This
        metadata has no numerical array shape and is used only in figure captions.
    """
    return PlotContext(
        session_id="synthetic-lfp-summary",
        alignment_event="choice_time",
        epoch_bounds_s=_epoch_windows(),
        notch_enabled=False,
        gamma_exclusion_hz=(58.0, 62.0),
        reference_description="synthetic independent sites; no rereference",
        source_voltage_unit="uV",
    )


def test_seeded_synthetic_lfp_summary_pipeline_cache_and_plotting(
    tmp_path: Path,
) -> None:
    """Exercise numerical summaries, all cache components, and public plotting APIs.

    Synthetic LFP is 500 Hz on ``[-2, 2)`` seconds. Payloads retain documented
    site/trial/epoch/frequency axes and are atomically written, reloaded, then
    sliced only from the loaded arrays for every plotting API.
    """
    config = _synthetic_configuration(tmp_path)
    trial_df, traces = _synthetic_trials_and_traces()
    prepared = build_prepared_trials(
        trial_df,
        config.trial_filter,
        "choice_time",
        {"PFC": np.ones(4, dtype=bool), "HPC1": np.ones(4, dtype=bool)},
    )
    condition_membership = np.column_stack(
        (
            prepared.condition_membership[
                :, prepared.condition_names.index("correct_rewarded")
            ],
            prepared.condition_membership[
                :, prepared.condition_names.index("omission")
            ],
            np.ones(4, dtype=bool),
        )
    )
    filter_membership = np.asarray(prepared.filter_membership, dtype=bool)
    psd = [
        compute_trial_epoch_psds(
            traces[site], _TIME_S, _FS_HZ, config.analysis_windows, config.power
        )
        for site in range(2)
    ]
    psd_linear = np.stack([result.psd_linear for result in psd])
    psd_valid = np.stack([result.psd_valid for result in psd])
    reference = np.stack(
        [
            compute_session_reference_psd(
                result.psd_linear[:, 0], result.psd_valid[:, 0]
            )
            for result in psd
        ]
    )
    normalized = np.stack(
        [normalize_psd_db(psd_linear[site], reference[site]) for site in range(2)]
    )
    band_linear = np.stack(
        [
            mean_band_power_linear(psd_linear, psd[0].frequency_hz, band)[0]
            for band in config.power.bands
        ],
        axis=-1,
    )
    phase = _phase_tensor()
    clustering = compute_phase_clustering_summary(
        phase,
        np.ones_like(phase, dtype=bool),
        _CONDITION_NAMES,
        condition_membership,
        ((0, 1),),
    )
    relative_phase = np.moveaxis(phase[:1] * np.conjugate(phase[1:]), 1, 2)
    plv = compute_trial_plv_by_frequency(
        relative_phase,
        np.ones_like(relative_phase, dtype=bool),
        _TIME_S,
        _epoch_windows(),
    )
    plv_band = aggregate_trial_plv_bands(
        plv.plv_by_frequency,
        _FREQUENCY_HZ,
        config.phase.bands,
    )

    theta_index = int(np.flatnonzero(_FREQUENCY_HZ == 8.0)[0])
    gamma_index = int(np.flatnonzero(psd[0].frequency_hz == 40.0)[0])
    assert np.nanmedian(psd_linear[0, :, 0, theta_index]) > np.nanmedian(
        psd_linear[0, :, 0, theta_index - 1]
    )
    assert np.nanmedian(psd_linear[0, :, 0, gamma_index]) > np.nanmedian(
        psd_linear[0, :, 0, gamma_index - 1]
    )
    assert np.nanmean(clustering.itpc[0, 0, theta_index]) > 0.9
    assert np.nanmean(clustering.itpc[2, 0, theta_index]) < 0.35
    assert np.isclose(
        np.angle(
            np.mean(np.exp(1j * clustering.ispc_phase_offset_rad[0, 0, theta_index]))
        ),
        _SITE_OFFSET_RAD,
        atol=1e-8,
    )
    assert np.nanmean(plv.plv_by_frequency[0, :, 0, theta_index]) > 0.99

    schedule = generate_trial_derangement_schedule(4, 100, seed=72)
    locked_null = compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=_trial_spikes(True, 0),
        phase_time_s=_TIME_S,
        trial_phase_vectors=np.moveaxis(phase[0], 0, 1),
        frequencies_hz=_FREQUENCY_HZ,
        schedule=schedule,
    )
    unlocked_null = compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=_trial_spikes(False, 73),
        phase_time_s=_TIME_S,
        trial_phase_vectors=np.moveaxis(phase[0], 0, 1),
        frequencies_hz=_FREQUENCY_HZ,
        schedule=schedule,
    )
    assert locked_null.spike_count[theta_index] == 52
    assert (
        locked_null.observed_ppc[theta_index]
        > unlocked_null.observed_ppc[theta_index] + 0.5
    )
    assert locked_null.p_value[theta_index] < 0.05
    assert np.isnan(unlocked_null.p_value[theta_index]) or (
        unlocked_null.p_value[theta_index] > locked_null.p_value[theta_index]
    )

    unit_count, condition_count, site_count, epoch_count, frequency_count = (
        2,
        3,
        2,
        3,
        20,
    )
    ppc_shape = (unit_count, condition_count, site_count, epoch_count, frequency_count)
    ppc = np.full(ppc_shape, np.nan)
    resultant = np.full(ppc_shape, np.nan)
    preferred = np.full(ppc_shape, np.nan)
    spike_count = np.zeros(ppc_shape, dtype=np.int64)
    null_fields: dict[str, np.ndarray] = {}
    for field in (
        "eligible_trial_count",
        "null_exceedance_count",
        "permutation_count",
    ):
        null_fields[field] = np.zeros(ppc_shape, dtype=np.int64)
    for field in (
        "p_value",
        "null_mean",
        "null_std",
        "null_p025",
        "null_p50",
        "null_p975",
    ):
        null_fields[field] = np.full(ppc_shape, np.nan)
    null_eligible = np.zeros(ppc_shape, dtype=bool)
    for unit_index, result in enumerate((locked_null, unlocked_null)):
        for condition_index in range(condition_count):
            for site_index in range(site_count):
                for epoch_index in range(epoch_count):
                    ppc[unit_index, condition_index, site_index, epoch_index] = (
                        result.observed_ppc
                    )
                    resultant[unit_index, condition_index, site_index, epoch_index] = (
                        np.sqrt(np.maximum(result.observed_ppc, 0.0))
                    )
                    spike_count[
                        unit_index, condition_index, site_index, epoch_index
                    ] = result.spike_count
                    null = result.null_summary
                    for field in null_fields:
                        null_fields[field][
                            unit_index, condition_index, site_index, epoch_index
                        ] = getattr(null, field)
                    null_eligible[
                        unit_index, condition_index, site_index, epoch_index
                    ] = null.null_eligible
    q_value = adjust_ppc_pvalues_bh(
        p_value=null_fields["p_value"], null_eligible=null_eligible
    )
    prevalence = compute_significant_prevalence(
        q_value=q_value,
        null_eligible=null_eligible,
        alpha=0.05,
    )
    assert q_value[0, 0, 0, 0, theta_index] < 0.05
    assert prevalence.prevalence[0, 0, 0, theta_index] > 0.0
    ppc_band = compute_band_ppc_means(
        frequencies_hz=_FREQUENCY_HZ, ppc_by_frequency=ppc
    ).ppc_band_mean

    sizes = {
        "unit": 2,
        "population": 1,
        "site": 2,
        "pair": 1,
        "trial": 4,
        "trial_offset": 5,
        "condition": 3,
        "epoch": 3,
        "band": 2,
        "frequency": 20,
        "time": 2000,
        "phase_bin": 4,
        "phase_bin_edge": 5,
        "spike": 104,
    }
    power_arrays = _payload_arrays("power", sizes)
    power_arrays.update(
        trial_indices=np.arange(4, dtype=np.int64),
        site_ids=np.array(("PFC", "HPC1")),
        site_voltage_units=np.array(("uV", "uV")),
        condition_names=np.array(_CONDITION_NAMES),
        condition_membership=condition_membership,
        filter_membership=filter_membership,
        condition_trial_count=condition_membership.sum(0),
        condition_effective_trial_count=np.tile(
            condition_membership.sum(0)[:, None], (1, 2)
        ),
        condition_unstable=np.ones((3, 2), dtype=bool),
        frequency_hz=psd[0].frequency_hz,
        epoch_names=np.array(_EPOCH_NAMES),
        objective_valid=np.ones((2, 4), dtype=bool),
        site_valid=np.ones((2, 4), dtype=bool),
        user_excluded=np.zeros(4, dtype=bool),
        exclusion_reason_code=np.full((2, 4), "", dtype="<U32"),
        psd_linear=psd_linear,
        psd_valid=psd_valid,
        session_reference_psd_linear=reference,
        presession_reference_psd_linear=np.full_like(reference, np.nan),
        presession_reference_available=np.zeros(2, dtype=bool),
        normalized_psd_session_db=normalized,
        normalized_psd_presession_db=np.full_like(normalized, np.nan),
        band_names=np.array(_BAND_NAMES),
        band_power_linear=band_linear,
        band_power_session_db=10.0
        * np.log10(
            band_linear / np.nanmedian(band_linear[:, :, 0], axis=1)[:, None, None, :]
        ),
        band_power_presession_db=np.full_like(band_linear, np.nan),
        trial_rms=np.sqrt(np.mean(traces**2, axis=2)),
        trial_peak_to_peak=np.ptp(traces, axis=2),
        relative_time_s=_TIME_S,
        source_trace=traces,
    )
    sync_arrays = _payload_arrays("synchrony", sizes)
    filtered = np.repeat(traces[:, :, None, :], 2, axis=2)
    phase_rad = np.repeat(
        np.moveaxis(np.angle(phase[:, theta_index : theta_index + 1]), 1, 2),
        2,
        axis=2,
    )
    sync_arrays.update(
        trial_indices=np.arange(4, dtype=np.int64),
        site_ids=np.array(("PFC", "HPC1")),
        site_voltage_units=np.array(("uV", "uV")),
        condition_names=np.array(_CONDITION_NAMES),
        condition_membership=condition_membership,
        filter_membership=filter_membership,
        frequency_hz=_FREQUENCY_HZ,
        epoch_names=np.array(_EPOCH_NAMES),
        band_names=np.array(_BAND_NAMES),
        relative_time_s=_TIME_S,
        site_valid=np.ones((2, 4), dtype=bool),
        pair_valid=np.ones((1, 4), dtype=bool),
        site_exclusion_count=np.zeros(2, dtype=np.int64),
        pair_exclusion_count=np.zeros(1, dtype=np.int64),
        pair_site_a_ids=np.array(("PFC",)),
        pair_site_b_ids=np.array(("HPC1",)),
        itpc=clustering.itpc,
        itpc_effective_trial_count=clustering.itpc_effective_trial_count,
        ispc=clustering.ispc,
        ispc_phase_offset_rad=clustering.ispc_phase_offset_rad,
        ispc_effective_trial_count=clustering.ispc_effective_trial_count,
        itpc_band_mean=np.full((3, 2, 3, 2), np.nan),
        itpc_ci_low=np.full((3, 2, 3, 2), np.nan),
        itpc_ci_high=np.full((3, 2, 3, 2), np.nan),
        itpc_unstable=np.ones((3, 2, 3, 2), dtype=bool),
        ispc_band_mean=np.full((3, 1, 3, 2), np.nan),
        ispc_ci_low=np.full((3, 1, 3, 2), np.nan),
        ispc_ci_high=np.full((3, 1, 3, 2), np.nan),
        ispc_unstable=np.ones((3, 1, 3, 2), dtype=bool),
        plv_by_frequency=np.moveaxis(plv.plv_by_frequency, (0, 1), (1, 0)),
        plv_phase_offset_rad=np.moveaxis(plv.plv_phase_offset_rad, (0, 1), (1, 0)),
        plv_valid_sample_count=np.moveaxis(plv.valid_sample_count, (0, 1), (1, 0)),
        plv_valid_sample_fraction=np.moveaxis(
            plv.valid_sample_fraction, (0, 1), (1, 0)
        ),
        plv_computable=np.moveaxis(plv.computable, (0, 1), (1, 0)),
        plv_band_mean=np.moveaxis(plv_band, (0, 1), (1, 0)),
        source_trace=traces,
        band_filtered_trace=filtered,
        hilbert_phase_rad=phase_rad,
    )
    spike_arrays = _payload_arrays("spike_phase", sizes)
    locked_times, locked_offsets = _packed_spikes(_trial_spikes(True, 0))
    unlocked_times, unlocked_offsets = _packed_spikes(_trial_spikes(False, 73))
    spike_arrays.update(
        unit_ids=np.array(("PFC:1", "PFC:2")),
        population_ids=np.array(("PFC",)),
        trial_indices=np.arange(4, dtype=np.int64),
        site_ids=np.array(("PFC", "HPC1")),
        site_voltage_units=np.array(("uV", "uV")),
        condition_names=np.array(_CONDITION_NAMES),
        condition_membership=condition_membership,
        filter_membership=filter_membership,
        epoch_names=np.array(_EPOCH_NAMES),
        band_names=np.array(_BAND_NAMES),
        frequency_hz=_FREQUENCY_HZ,
        relative_time_s=_TIME_S,
        phase_bin_edges_rad=np.linspace(-np.pi, np.pi, 5),
        ppc=ppc,
        resultant_length=resultant,
        preferred_phase_rad=preferred,
        spike_count=spike_count,
        computable=spike_count >= 2,
        reliable=spike_count >= 50,
        null_eligible=null_eligible,
        eligible_trial_count=null_fields["eligible_trial_count"],
        null_exceedance_count=null_fields["null_exceedance_count"],
        permutation_count=null_fields["permutation_count"],
        p_value=null_fields["p_value"],
        q_value=q_value,
        significant=np.isfinite(q_value) & (q_value < 0.05),
        null_mean=null_fields["null_mean"],
        null_std=null_fields["null_std"],
        null_p025=null_fields["null_p025"],
        null_p50=null_fields["null_p50"],
        null_p975=null_fields["null_p975"],
        ppc_band_mean=ppc_band,
        representative_phase_hist_count=np.zeros((2, 3, 2, 3, 2, 4), dtype=np.int64),
        relative_spike_times_s=np.concatenate((locked_times, unlocked_times)),
        relative_spike_time_offsets=np.stack(
            (locked_offsets, unlocked_offsets + locked_times.size)
        ),
        source_trace=traces,
        band_filtered_trace=filtered,
        hilbert_phase_rad=phase_rad,
        selected_low_unit_ids=np.full((3, 2, 3, 2), "PFC:2", dtype="<U32"),
        selected_high_unit_ids=np.full((3, 2, 3, 2), "PFC:1", dtype="<U32"),
        illustrative_trial_indices=np.zeros((3, 2, 3, 2), dtype=np.int64),
    )
    payloads = {
        name: build_component_payload(name, arrays)
        for name, arrays in (
            ("power", power_arrays),
            ("synchrony", sync_arrays),
            ("spike_phase", spike_arrays),
        )
    }

    dependencies = PipelineDependencies(
        prepare_power=lambda _: prepared,
        prepare_phase=lambda _: phase,
        prepare_spike=lambda _, __: (locked_null, unlocked_null),
        build_power_payload=lambda _, __: payloads["power"],
        build_synchrony_payload=lambda _, __: payloads["synchrony"],
        build_spike_phase_payload=lambda _, __, ___: payloads["spike_phase"],
        load_manifest=lambda cache_directory: load_or_initialize_manifest(
            cache_directory, config
        ),
        write_component=write_component_transaction,
    )
    result = compute_all_components(config, dependencies)
    assert [item.component for item in result.component_results] == [
        "power",
        "synchrony",
        "spike_phase",
    ]
    assert all(item.state == "complete" for item in result.component_results)
    manifest = load_or_initialize_manifest(config.output_directory, config)
    loaded = {
        name: load_component_arrays(
            config.output_directory / f"{name}.npz", manifest, name
        )
        for name in ("power", "synchrony", "spike_phase")
    }
    assert all(
        assess_component_status(config.output_directory, name, config, manifest).status
        == "compatible"
        for name in loaded
    )
    assert all(
        np.array_equal(loaded[name]["filter_membership"], filter_membership)
        for name in loaded
    )

    context = _plot_context()
    figures = [
        plot_condition_psd(
            loaded["power"]["frequency_hz"],
            np.repeat(
                loaded["power"]["normalized_psd_session_db"][0, :, 0][None], 3, axis=0
            ),
            _CONDITION_NAMES,
            condition_membership.sum(0),
            "PFC",
            "whole",
            "session_median",
            context,
        ),
        plot_band_power_summary(
            np.repeat(loaded["power"]["band_power_session_db"][0][None], 3, axis=0),
            _CONDITION_NAMES,
            _EPOCH_NAMES,
            _BAND_NAMES,
            np.tile(condition_membership.sum(0)[:, None], (1, 3)),
            "PFC",
            "session_median",
            context,
        ),
        plot_phase_map(
            loaded["synchrony"]["itpc"][0, 0],
            loaded["synchrony"]["itpc_effective_trial_count"][0, 0],
            _FREQUENCY_HZ,
            _TIME_S,
            "ITPC",
            "PFC",
            "correct_rewarded",
            context,
            total_displayed_trial_count=int(condition_membership[:, 0].sum()),
        ),
        plot_phase_band_summary(
            np.array((0.9, 0.8)),
            np.array((0.8, 0.7)),
            np.array((1.0, 0.9)),
            np.array((2, 2)),
            ("PFC", "PFC-HPC1"),
            "theta",
            "whole",
            "ITPC/ISPC",
            context,
        ),
        plot_plv_distribution(
            loaded["synchrony"]["plv_band_mean"][:, 0, :, 0],
            loaded["synchrony"]["plv_valid_sample_count"][:, 0, :, theta_index],
            loaded["synchrony"]["plv_valid_sample_fraction"][:, 0, :, theta_index],
            _EPOCH_NAMES,
            "PFC-HPC1",
            "theta",
            "correct_rewarded",
            context,
        ),
        plot_plv_exemplar(
            loaded["synchrony"]["relative_time_s"],
            loaded["synchrony"]["source_trace"][:, 0],
            loaded["synchrony"]["band_filtered_trace"][:, 0, 0],
            loaded["synchrony"]["hilbert_phase_rad"][:, 0, 0],
            ("PFC", "HPC1"),
            "PFC-HPC1",
            0,
            "95th percentile",
            1.0,
            1.0,
            context,
        ),
        plot_unit_ppc_map(
            loaded["spike_phase"]["ppc"][:, 0, 0, 0],
            loaded["spike_phase"]["computable"][:, 0, 0, 0],
            loaded["spike_phase"]["reliable"][:, 0, 0, 0],
            loaded["spike_phase"]["spike_count"][:, 0, 0, 0],
            _FREQUENCY_HZ,
            ("PFC:1", "PFC:2"),
            "correct_rewarded",
            "PFC",
            "whole",
            context,
        ),
        plot_population_ppc_maps(
            np.nanmedian(loaded["spike_phase"]["ppc"], axis=0)[:, 0, 0],
            prevalence.prevalence[:, 0, 0],
            prevalence.eligible_unit_count[:, 0, 0],
            prevalence.total_unit_count[:, 0, 0],
            _CONDITION_NAMES,
            _FREQUENCY_HZ,
            "PFC",
            "whole",
            context,
        ),
        plot_ppc_band_summary(
            loaded["spike_phase"]["ppc_band_mean"][:, :, 0].transpose(1, 0, 2, 3),
            np.repeat(
                np.all(loaded["spike_phase"]["reliable"][:, :, 0], axis=-1)[..., None],
                2,
                axis=-1,
            ).transpose(1, 0, 2, 3),
            _CONDITION_NAMES,
            _EPOCH_NAMES,
            _BAND_NAMES,
            2,
            "PFC",
            context,
        ),
        plot_ppc_exemplar(
            loaded["spike_phase"]["frequency_hz"],
            loaded["spike_phase"]["ppc"][0, 0, 0, 0],
            loaded["spike_phase"]["preferred_phase_rad"][0, 0, 0, 0],
            loaded["spike_phase"]["representative_phase_hist_count"][0, 0, 0, 0, 0],
            loaded["spike_phase"]["phase_bin_edges_rad"],
            loaded["spike_phase"]["relative_time_s"],
            loaded["spike_phase"]["source_trace"][0, 0],
            loaded["spike_phase"]["band_filtered_trace"][0, 0, 0],
            loaded["spike_phase"]["relative_spike_times_s"][:52],
            "PFC:1",
            0,
            "theta",
            8.0,
            "95th percentile",
            context,
        ),
    ]
    for figure, _ in figures:
        plt.close(figure)
