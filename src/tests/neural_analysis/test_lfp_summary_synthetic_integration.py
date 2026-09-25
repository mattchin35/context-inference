"""Seeded end-to-end cache and plotting validation on known LFP/spike signals."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.lfp_power_summary import (
    compute_presession_reference_psd,
    compute_session_reference_psd,
    compute_trial_epoch_psds,
    mean_band_power_linear,
    normalize_psd_db,
)
from src.neural_analysis import lfp_loading, lfp_phase_clustering
from src.neural_analysis.lfp_summary_io import (
    assess_component_status,
    load_component_arrays,
    load_or_initialize_manifest,
    write_component_transaction,
)
from src.neural_analysis.lfp_summary_models import (
    FrequencyBandConfig,
    PPCAnalysisConfig,
    PPCExecutionConfig,
    PhaseAnalysisConfig,
    PowerAnalysisConfig,
    UnitPopulationConfig,
    default_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_payloads import (
    COMPONENT_ARRAY_SCHEMAS,
    build_component_payload,
)
from src.neural_analysis.lfp_summary_pipeline import (
    PipelineDependencies,
    compute_all_components,
    compute_spike_phase_component,
    compute_synchrony_component,
)
from src.neural_analysis.lfp_synchrony_validation import (
    SynchronyValidationDependencies,
    render_cached_synchrony_validation,
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
    TrialRelativeSpikeTrains,
    build_common_event_grid,
    build_prepared_trials,
)
from src.neural_analysis.lfp_summary_runtime import (
    PreparedPhaseRun,
    PreparedSpikeRun,
    build_spike_phase_payload,
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
from src.neural_analysis.spike_lfp_hilbert_phase import compute_hilbert_phase_trace


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


def test_positive_open_ephys_gain_scales_amplitudes_and_linear_power_but_not_normalized_db_or_hilbert_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual Open Ephys reads scale physical amplitudes but preserve deterministic phase metrics.

    Both files store identical raw float32 traces. The only difference is the
    observed positive, zero-offset gain in ``lfp_preprocessing.json``. This
    deliberately exercises the adapter boundary rather than manufacturing a
    scaled array downstream of it.
    """
    config = _synthetic_configuration(tmp_path)
    _, source_trials = _synthetic_trials_and_traces()
    gain_to_uv = 0.195
    trial_count = 2
    trial_samples = _TIME_S.size
    # Seeded broadband content in both trial and presession windows keeps every
    # Welch reference bin well conditioned. The exact same raw matrix is then
    # written under the identity and corrected gain sidecars.
    raw_trials = source_trials[:, :trial_count] + 0.05 * np.random.default_rng(103).normal(
        size=(2, trial_count, trial_samples)
    )
    baseline_time_s = np.arange(5_000, dtype=float) / _FS_HZ
    raw_baseline = np.stack(
        (
            2.0 * np.sin(2.0 * np.pi * 8.0 * baseline_time_s),
            2.0 * np.sin(2.0 * np.pi * 8.0 * baseline_time_s - _SITE_OFFSET_RAD),
        ),
        axis=1,
    )
    raw_baseline += 0.05 * np.random.default_rng(104).normal(
        size=raw_baseline.shape
    )
    raw_matrix = np.concatenate(
        (raw_baseline, np.moveaxis(raw_trials, 0, -1).reshape(-1, 2)), axis=0
    ).astype(np.float32)

    def write_reader_source(directory: Path, gain: float) -> Path:
        """Write one synthetic raw binary and full observed affine sidecar metadata."""
        directory.mkdir()
        lfp_path = directory / "lfp.dat"
        raw_matrix.tofile(lfp_path)
        metadata = {
            "output_binary": "lfp.dat",
            "sampling_frequency_hz": _FS_HZ,
            "num_channels": 2,
            "num_segments": 1,
            "num_samples_by_segment": [raw_matrix.shape[0]],
            "dtype": "float32",
            "binary_layout": "time_major_channel_interleaved",
            "channel_ids_in_binary_order": ["CH0", "CH1"],
            "lfp_binary_scaling": {
                "data_units": "unscaled_binary_values",
                "has_scaleable_traces": True,
                "channel_ids": ["CH0", "CH1"],
                "gain_to_uV_by_channel": [gain, gain],
                "offset_to_uV_by_channel": [0.0, 0.0],
                "physical_unit_by_channel": ["uV", "uV"],
                "export_scale_factor": 1.0,
                "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
            },
        }
        (directory / "lfp_preprocessing.json").write_text(json.dumps(metadata), encoding="ascii")
        return lfp_path

    identity_path = write_reader_source(tmp_path / "identity", 1.0)
    corrected_path = write_reader_source(tmp_path / "corrected", gain_to_uv)

    def read_trials(path: Path) -> np.ndarray:
        """Read the two stored channels and trials only through the production window reader."""
        values = np.empty((2, trial_count, 2, trial_samples), dtype=float)
        for trial_index in range(trial_count):
            start = 5_000 + trial_index * trial_samples
            stop = start + trial_samples
            for channel_index in range(2):
                values[channel_index, trial_index, channel_index], rate_hz = (
                    lfp_loading.read_open_ephys_lfp_channel_window(
                        path, channel_index, start, stop
                    )
                )
                assert rate_hz == _FS_HZ
        return np.stack(
            (
                values[0, :, 0],
                values[1, :, 1],
            ),
            axis=0,
        )

    identity_trials = read_trials(identity_path)
    corrected_trials = read_trials(corrected_path)
    identity_baseline, _ = lfp_loading.read_open_ephys_lfp_channel_window(
        identity_path, 0, 0, 5_000
    )
    corrected_baseline, _ = lfp_loading.read_open_ephys_lfp_channel_window(
        corrected_path, 0, 0, 5_000
    )
    identity_metadata = lfp_loading.load_open_ephys_lfp_metadata(identity_path)
    corrected_metadata = lfp_loading.load_open_ephys_lfp_metadata(corrected_path)

    def cache_reader_derived_phase(
        cache_config,
        trials: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        """Commit reader-derived source, filtered, and phase arrays through Synchrony."""
        filtered = np.empty((2, trial_count, 1, trial_samples), dtype=float)
        phase_rad = np.empty_like(filtered)
        for site_index in range(2):
            for trial_index in range(trial_count):
                transformed = compute_hilbert_phase_trace(
                    _TIME_S,
                    trials[site_index, trial_index],
                    _FS_HZ,
                    frequency_band_hz=(6.0, 10.0),
                )
                filtered[site_index, trial_index, 0] = transformed.bandpassed_lfp
                phase_rad[site_index, trial_index, 0] = transformed.phase_rad
        sizes = {
            "site": 2,
            "trial": trial_count,
            "condition": 1,
            "frequency": 1,
            "epoch": 1,
            "band": 1,
            "pair": 1,
            "time": trial_samples,
        }
        arrays = _payload_arrays("synchrony", sizes)
        arrays.update(
            trial_indices=np.arange(trial_count, dtype=np.int64),
            site_ids=np.array(("PFC", "HPC1")),
            site_voltage_units=np.array(("uV", "uV")),
            condition_names=np.array(("all_trials",)),
            condition_membership=np.ones((trial_count, 1), dtype=bool),
            filter_membership=np.ones(trial_count, dtype=bool),
            frequency_hz=np.array((8.0,)),
            epoch_names=np.array(("whole",)),
            band_names=np.array(("theta",)),
            relative_time_s=_TIME_S,
            site_valid=np.ones((2, trial_count), dtype=bool),
            pair_valid=np.ones((1, trial_count), dtype=bool),
            site_exclusion_count=np.zeros(2, dtype=np.int64),
            pair_exclusion_count=np.zeros(1, dtype=np.int64),
            pair_site_a_ids=np.array(("PFC",)),
            pair_site_b_ids=np.array(("HPC1",)),
            itpc=np.ones((1, 2, 1, trial_samples)),
            itpc_effective_trial_count=np.full(
                (1, 2, 1, trial_samples), trial_count, dtype=np.int64
            ),
            ispc=np.ones((1, 1, 1, trial_samples)),
            ispc_phase_offset_rad=np.zeros((1, 1, 1, trial_samples)),
            ispc_effective_trial_count=np.full(
                (1, 1, 1, trial_samples), trial_count, dtype=np.int64
            ),
            itpc_band_mean=np.ones((1, 2, 1, 1)),
            itpc_ci_low=np.ones((1, 2, 1, 1)),
            itpc_ci_high=np.ones((1, 2, 1, 1)),
            itpc_unstable=np.zeros((1, 2, 1, 1), dtype=bool),
            ispc_band_mean=np.ones((1, 1, 1, 1)),
            ispc_ci_low=np.ones((1, 1, 1, 1)),
            ispc_ci_high=np.ones((1, 1, 1, 1)),
            ispc_unstable=np.zeros((1, 1, 1, 1), dtype=bool),
            plv_by_frequency=np.ones((trial_count, 1, 1, 1)),
            plv_phase_offset_rad=np.zeros((trial_count, 1, 1, 1)),
            plv_valid_sample_count=np.full(
                (trial_count, 1, 1, 1), trial_samples, dtype=np.int64
            ),
            plv_valid_sample_fraction=np.ones((trial_count, 1, 1, 1)),
            plv_computable=np.ones((trial_count, 1, 1, 1), dtype=bool),
            plv_band_mean=np.ones((trial_count, 1, 1, 1)),
            source_trace=trials,
            band_filtered_trace=filtered,
            hilbert_phase_rad=phase_rad,
        )
        payload = build_component_payload("synchrony", arrays)
        dependencies = PipelineDependencies(
            prepare_power=lambda _: object(),
            prepare_phase=lambda _: object(),
            prepare_spike=lambda _, __: object(),
            build_power_payload=lambda _, __: payload,
            build_synchrony_payload=lambda _, __: payload,
            build_spike_phase_payload=lambda _, __, ___: payload,
            load_manifest=lambda directory: load_or_initialize_manifest(
                directory, cache_config
            ),
            write_component=write_component_transaction,
        )
        result = compute_synchrony_component(
            cache_config, dependencies, prepared_phase=object()
        )
        assert result.state == "complete"
        assert result.manifest is not None
        return (
            load_component_arrays(
                cache_config.output_directory / "synchrony.npz",
                result.manifest,
                "synchrony",
            ),
            result.manifest,
        )

    def open_ephys_cache_config(lfp_path: Path, cache_name: str):
        """Bind the two summary sites to distinct channels of one physical-uV source."""
        sites = tuple(
            replace(
                site,
                acquisition_format="open_ephys",
                lfp_path=lfp_path,
                aligned_sync_path=None,
                saved_channel_index=index,
                voltage_unit="uV",
                sample_rate_hz=_FS_HZ,
            )
            for index, site in enumerate(config.sites)
        )
        return replace(
            config,
            session_id=f"synthetic-cache-{cache_name}",
            output_directory=tmp_path / cache_name,
            sites=sites,
        )

    identity_cache, identity_manifest = cache_reader_derived_phase(
        open_ephys_cache_config(identity_path, "identity-cache"), identity_trials
    )
    corrected_cache, corrected_manifest = cache_reader_derived_phase(
        open_ephys_cache_config(corrected_path, "corrected-cache"), corrected_trials
    )

    # The production schema remains adapter-neutral; cached per-site metadata
    # resolves its physical voltage placeholder for this Open Ephys payload.
    linear_psd_contract = COMPONENT_ARRAY_SCHEMAS["power"]["psd_linear"]
    assert linear_psd_contract.units == "source-voltage-unit^2/Hz"
    np.testing.assert_array_equal(
        corrected_cache["site_voltage_units"], np.array(("uV", "uV"))
    )
    assert linear_psd_contract.units.replace("source-voltage-unit", "uV") == "uV^2/Hz"

    for stable_axis in (
        "site_ids",
        "trial_indices",
        "site_voltage_units",
        "condition_names",
        "frequency_hz",
        "epoch_names",
        "band_names",
        "pair_site_a_ids",
        "pair_site_b_ids",
    ):
        np.testing.assert_array_equal(
            corrected_cache[stable_axis], identity_cache[stable_axis]
        )
    np.testing.assert_allclose(
        corrected_cache["source_trace"],
        gain_to_uv * identity_cache["source_trace"],
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        corrected_cache["band_filtered_trace"],
        gain_to_uv * identity_cache["band_filtered_trace"],
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        corrected_cache["hilbert_phase_rad"],
        identity_cache["hilbert_phase_rad"],
        rtol=0.0,
        atol=1e-10,
    )
    for manifest in (identity_manifest, corrected_manifest):
        schema = manifest["components"]["synchrony"]["array_schema"]
        assert schema["source_trace"]["units"] == "source-voltage-unit"
        assert schema["band_filtered_trace"]["units"] == "source-voltage-unit"
        assert schema["hilbert_phase_rad"]["units"] == "rad"
        assert schema["itpc"]["units"] == "dimensionless"

    def forbid_raw_reopen(*_: object, **__: object) -> None:
        """Fail if a cache-backed figure reaches the raw Open Ephys reader."""
        pytest.fail("cache-backed plot reopened a raw Open Ephys LFP source")

    monkeypatch.setattr(
        lfp_loading, "read_open_ephys_lfp_channel_window", forbid_raw_reopen
    )
    figure, axes = plot_plv_exemplar(
        corrected_cache["relative_time_s"],
        corrected_cache["source_trace"][:, 0],
        corrected_cache["band_filtered_trace"][:, 0, 0],
        corrected_cache["hilbert_phase_rad"][:, 0, 0],
        ("PFC", "HPC1"),
        "PFC-HPC1",
        0,
        "reader-derived cache",
        1.0,
        1.0,
        _plot_context(),
    )
    assert axes["source"].get_ylabel() == "LFP (uV)"
    assert axes["filtered"].get_ylabel() == "Bandpassed LFP (uV)"
    assert axes["phase"].get_ylabel() == "Phase (rad)"
    plt.close(figure)

    np.testing.assert_allclose(corrected_trials, gain_to_uv * identity_trials, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        np.sqrt(np.mean(corrected_trials**2, axis=2)),
        gain_to_uv * np.sqrt(np.mean(identity_trials**2, axis=2)),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.ptp(corrected_trials, axis=2),
        gain_to_uv * np.ptp(identity_trials, axis=2),
        rtol=1e-12,
        atol=0.0,
    )
    identity_psd = compute_trial_epoch_psds(
        identity_trials[0], _TIME_S, _FS_HZ, config.analysis_windows, config.power
    )
    corrected_psd = compute_trial_epoch_psds(
        corrected_trials[0], _TIME_S, _FS_HZ, config.analysis_windows, config.power
    )
    identity_reference = compute_session_reference_psd(
        identity_psd.psd_linear[:, 0], identity_psd.psd_valid[:, 0]
    )
    corrected_reference = compute_session_reference_psd(
        corrected_psd.psd_linear[:, 0], corrected_psd.psd_valid[:, 0]
    )
    identity_presession_hz, identity_presession, identity_presession_available = (
        compute_presession_reference_psd(
            baseline_time_s,
            identity_baseline,
            10.0,
            _FS_HZ,
            config.power,
        )
    )
    corrected_presession_hz, corrected_presession, corrected_presession_available = (
        compute_presession_reference_psd(
            baseline_time_s,
            corrected_baseline,
            10.0,
            _FS_HZ,
            config.power,
        )
    )
    identity_band = mean_band_power_linear(
        identity_psd.psd_linear, identity_psd.frequency_hz, config.power.bands[0]
    )[0]
    corrected_band = mean_band_power_linear(
        corrected_psd.psd_linear, corrected_psd.frequency_hz, config.power.bands[0]
    )[0]

    np.testing.assert_allclose(
        corrected_psd.psd_linear,
        gain_to_uv**2 * identity_psd.psd_linear,
        rtol=1e-10,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        corrected_reference,
        gain_to_uv**2 * identity_reference,
        rtol=1e-10,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        corrected_band,
        gain_to_uv**2 * identity_band,
        rtol=1e-10,
        atol=1e-16,
    )
    assert identity_presession_available and corrected_presession_available
    np.testing.assert_array_equal(identity_presession_hz, corrected_presession_hz)
    np.testing.assert_allclose(
        corrected_presession,
        gain_to_uv**2 * identity_presession,
        rtol=1e-10,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        normalize_psd_db(corrected_psd.psd_linear, corrected_reference),
        normalize_psd_db(identity_psd.psd_linear, identity_reference),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        normalize_psd_db(corrected_psd.psd_linear, corrected_presession),
        normalize_psd_db(identity_psd.psd_linear, identity_presession),
        rtol=0.0,
        atol=1e-10,
    )
    identity_hilbert = compute_hilbert_phase_trace(
        _TIME_S, identity_trials[0, 0], _FS_HZ, frequency_band_hz=(6.0, 10.0)
    )
    corrected_hilbert = compute_hilbert_phase_trace(
        _TIME_S, corrected_trials[0, 0], _FS_HZ, frequency_band_hz=(6.0, 10.0)
    )
    np.testing.assert_allclose(
        corrected_hilbert.bandpassed_lfp,
        gain_to_uv * identity_hilbert.bandpassed_lfp,
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        corrected_hilbert.phase_rad,
        identity_hilbert.phase_rad,
        rtol=0.0,
        atol=1e-10,
    )
    assert np.array_equal(corrected_hilbert.phase_valid, identity_hilbert.phase_valid)

    def wavelet_phase(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Morlet unit phase from the reader-derived physical-uV trial traces."""
        site_phase = []
        site_valid = []
        output_time_s: np.ndarray | None = None
        for site_index in range(2):
            trial_phase = []
            trial_valid = []
            for trial_index in range(trial_count):
                coefficients = lfp_phase_clustering.compute_wavelet_coefficients(
                    _TIME_S,
                    trials[site_index, trial_index],
                    _FS_HZ,
                    np.array([8.0]),
                    precision=8,
                    target_sample_rate_hz=_FS_HZ,
                )
                phase, valid = lfp_phase_clustering.normalize_wavelet_phase(coefficients.coefficients)
                output_time_s = coefficients.time_s
                trial_phase.append(phase)
                trial_valid.append(valid)
            site_phase.append(np.stack(trial_phase, axis=1))
            site_valid.append(np.stack(trial_valid, axis=1))
        assert output_time_s is not None
        return np.stack(site_phase), np.stack(site_valid), output_time_s

    identity_phase, identity_valid, phase_time_s = wavelet_phase(identity_trials)
    corrected_phase, corrected_valid, corrected_phase_time_s = wavelet_phase(corrected_trials)
    membership = np.ones((trial_count, 1), dtype=bool)
    identity_clustering = compute_phase_clustering_summary(
        identity_phase, identity_valid, ("all",), membership, ((0, 1),)
    )
    corrected_clustering = compute_phase_clustering_summary(
        corrected_phase, corrected_valid, ("all",), membership, ((0, 1),)
    )
    identity_relative = np.moveaxis(
        identity_phase[:1] * np.conjugate(identity_phase[1:]), 1, 2
    )
    corrected_relative = np.moveaxis(
        corrected_phase[:1] * np.conjugate(corrected_phase[1:]), 1, 2
    )
    identity_plv = compute_trial_plv_by_frequency(
        identity_relative,
        np.moveaxis(identity_valid[:1] & identity_valid[1:], 1, 2),
        phase_time_s,
        {"whole": (-2.0, 2.0)},
    )
    corrected_plv = compute_trial_plv_by_frequency(
        corrected_relative,
        np.moveaxis(corrected_valid[:1] & corrected_valid[1:], 1, 2),
        corrected_phase_time_s,
        {"whole": (-2.0, 2.0)},
    )
    spike_trains = tuple(np.linspace(-1.8, 1.8, 52) for _ in range(trial_count))
    identity_schedule = generate_trial_derangement_schedule(trial_count, 3, seed=71)
    corrected_schedule = generate_trial_derangement_schedule(trial_count, 3, seed=71)
    identity_ppc = compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=spike_trains,
        phase_time_s=phase_time_s,
        trial_phase_vectors=np.moveaxis(identity_phase[0], 1, 0),
        frequencies_hz=np.array([8.0]),
        schedule=identity_schedule,
    )
    corrected_ppc = compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=spike_trains,
        phase_time_s=corrected_phase_time_s,
        trial_phase_vectors=np.moveaxis(corrected_phase[0], 1, 0),
        frequencies_hz=np.array([8.0]),
        schedule=corrected_schedule,
    )

    np.testing.assert_array_equal(phase_time_s, corrected_phase_time_s)
    np.testing.assert_allclose(corrected_phase, identity_phase, rtol=0.0, atol=1e-6)
    assert np.array_equal(corrected_valid, identity_valid)
    np.testing.assert_allclose(corrected_clustering.itpc, identity_clustering.itpc, atol=1e-6)
    np.testing.assert_allclose(corrected_clustering.ispc, identity_clustering.ispc, atol=1e-6)
    np.testing.assert_allclose(corrected_plv.plv_by_frequency, identity_plv.plv_by_frequency, atol=1e-6)
    np.testing.assert_array_equal(corrected_ppc.spike_count, identity_ppc.spike_count)
    np.testing.assert_array_equal(corrected_ppc.schedule, identity_ppc.schedule)
    np.testing.assert_allclose(corrected_ppc.observed_ppc, identity_ppc.observed_ppc, atol=1e-6)
    for null_count_name in (
        "null_exceedance_count",
        "permutation_count",
        "eligible_trial_count",
    ):
        np.testing.assert_array_equal(
            getattr(corrected_ppc.null_summary, null_count_name),
            getattr(identity_ppc.null_summary, null_count_name),
        )
    for null_float_name in (
        "p_value",
        "null_mean",
        "null_std",
        "null_p025",
        "null_p50",
        "null_p975",
    ):
        np.testing.assert_allclose(
            getattr(corrected_ppc.null_summary, null_float_name),
            getattr(identity_ppc.null_summary, null_float_name),
            rtol=0.0,
            atol=1e-6,
            equal_nan=True,
        )
    for null_boolean_name in ("null_eligible", "significant"):
        np.testing.assert_array_equal(
            getattr(corrected_ppc.null_summary, null_boolean_name),
            getattr(identity_ppc.null_summary, null_boolean_name),
        )
    assert identity_metadata["lfp_binary_scaling"]["physical_unit_by_channel"] == ["uV", "uV"]
    assert corrected_metadata["lfp_binary_scaling"]["physical_unit_by_channel"] == ["uV", "uV"]


def test_seeded_synthetic_lfp_summary_pipeline_cache_and_plotting(
    tmp_path: Path,
) -> None:
    """Exercise numerical summaries, all cache components, and public plotting APIs.

    Synthetic LFP is 500 Hz on ``[-2, 2)`` seconds. Payloads retain documented
    site/trial/epoch/frequency axes and are atomically written, reloaded, then
    sliced only from the loaded arrays for every plotting API.
    """
    config = _synthetic_configuration(tmp_path)
    assert {
        "itpc_bootstrap_q25",
        "itpc_bootstrap_median",
        "itpc_bootstrap_q75",
        "itpc_band_trial_count",
        "ispc_bootstrap_q25",
        "ispc_bootstrap_median",
        "ispc_bootstrap_q75",
        "ispc_band_trial_count",
    }.issubset(COMPONENT_ARRAY_SCHEMAS["synchrony"])
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
        itpc_band_mean=np.full((3, 2, 3, 2), 0.72),
        itpc_ci_low=np.full((3, 2, 3, 2), 0.60),
        itpc_ci_high=np.full((3, 2, 3, 2), 0.80),
        itpc_bootstrap_q25=np.full((3, 2, 3, 2), 0.65),
        itpc_bootstrap_median=np.full((3, 2, 3, 2), 0.70),
        itpc_bootstrap_q75=np.full((3, 2, 3, 2), 0.75),
        itpc_band_trial_count=np.broadcast_to(
            np.array((2, 1, 4), dtype=np.int64)[:, None, None, None],
            (3, 2, 3, 2),
        ).copy(),
        itpc_unstable=np.ones((3, 2, 3, 2), dtype=bool),
        ispc_band_mean=np.full((3, 1, 3, 2), 0.42),
        ispc_ci_low=np.full((3, 1, 3, 2), 0.30),
        ispc_ci_high=np.full((3, 1, 3, 2), 0.50),
        ispc_bootstrap_q25=np.full((3, 1, 3, 2), 0.35),
        ispc_bootstrap_median=np.full((3, 1, 3, 2), 0.40),
        ispc_bootstrap_q75=np.full((3, 1, 3, 2), 0.45),
        ispc_band_trial_count=np.broadcast_to(
            np.array((2, 1, 4), dtype=np.int64)[:, None, None, None],
            (3, 1, 3, 2),
        ).copy(),
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
        illustrative_low_trial_indices=np.zeros((3, 2, 3, 2), dtype=np.int64),
        illustrative_high_trial_indices=np.zeros((3, 2, 3, 2), dtype=np.int64),
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
    np.testing.assert_array_equal(
        loaded["synchrony"]["itpc_band_trial_count"][:, 0, 0, 0],
        np.array((2, 1, 4), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        loaded["synchrony"]["ispc_band_trial_count"][:, 0, 0, 0],
        np.array((2, 1, 4), dtype=np.int64),
    )
    np.testing.assert_allclose(
        loaded["synchrony"]["itpc_bootstrap_median"],
        0.70,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        loaded["synchrony"]["ispc_bootstrap_median"],
        0.40,
        rtol=0.0,
        atol=0.0,
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
            observed_estimates=loaded["synchrony"]["itpc_band_mean"][:, 0, 0, 0],
            bootstrap_quantiles=np.vstack(
                (
                    loaded["synchrony"]["itpc_ci_low"][:, 0, 0, 0],
                    loaded["synchrony"]["itpc_bootstrap_q25"][:, 0, 0, 0],
                    loaded["synchrony"]["itpc_bootstrap_median"][:, 0, 0, 0],
                    loaded["synchrony"]["itpc_bootstrap_q75"][:, 0, 0, 0],
                    loaded["synchrony"]["itpc_ci_high"][:, 0, 0, 0],
                )
            ),
            selected_trial_counts=loaded["synchrony"]["itpc_band_trial_count"][:, 0, 0, 0],
            labels=_CONDITION_NAMES,
            band_name="theta",
            epoch_name="whole",
            metric_name="ITPC PFC",
            bootstrap_count=config.phase.bootstrap_count,
            context=context,
        ),
        plot_phase_band_summary(
            observed_estimates=loaded["synchrony"]["ispc_band_mean"][:, 0, 0, 0],
            bootstrap_quantiles=np.vstack(
                (
                    loaded["synchrony"]["ispc_ci_low"][:, 0, 0, 0],
                    loaded["synchrony"]["ispc_bootstrap_q25"][:, 0, 0, 0],
                    loaded["synchrony"]["ispc_bootstrap_median"][:, 0, 0, 0],
                    loaded["synchrony"]["ispc_bootstrap_q75"][:, 0, 0, 0],
                    loaded["synchrony"]["ispc_ci_high"][:, 0, 0, 0],
                )
            ),
            selected_trial_counts=loaded["synchrony"]["ispc_band_trial_count"][:, 0, 0, 0],
            labels=_CONDITION_NAMES,
            band_name="theta",
            epoch_name="whole",
            metric_name="ISPC PFC-HPC1",
            bootstrap_count=config.phase.bootstrap_count,
            context=context,
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

    report_calls: list[dict[str, object]] = []

    def cache_figure(*_: object, **__: object) -> tuple[plt.Figure, dict[str, object]]:
        """Return a disposable report figure without numerical recomputation."""
        return plt.figure(), {}

    def cache_band_summary(*args: object, **kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Capture cache-only phase-summary inputs for both metric families."""
        assert args == ()
        report_calls.append(kwargs)
        return plt.figure(), {}

    def save_marker(figure: plt.Figure, path: Path) -> None:
        """Record a report PNG path without rendering data outside pytest storage."""
        del figure
        path.write_bytes(b"png")

    report_dependencies = SynchronyValidationDependencies(
        pipeline_dependencies=object(),
        compute_synchrony_component=lambda *_: (_ for _ in ()).throw(
            AssertionError("cached report must not compute Synchrony")
        ),
        load_manifest=lambda directory, active_config: load_or_initialize_manifest(
            directory, active_config
        ),
        assess_component_status=assess_component_status,
        load_synchrony_arrays=lambda directory, manifest: load_component_arrays(
            directory / "synchrony.npz", manifest, "synchrony"
        ),
        plot_phase_map=cache_figure,
        plot_phase_band_summary=cache_band_summary,
        plot_plv_distribution=cache_figure,
        plot_plv_exemplar=cache_figure,
        save_png=save_marker,
        close_figure=plt.close,
        now_utc=lambda: "2026-09-23T00-00-00Z",
        monotonic_seconds=lambda: 1.0,
        peak_memory_bytes=lambda: 0,
        source_identifiers=lambda _: {"fixture": "seeded"},
    )
    report = render_cached_synchrony_validation(
        config,
        tmp_path / "cache_only_report",
        report_dependencies,
        synchrony_wall_time_s=1.0,
        synchrony_peak_memory_bytes=0,
    )

    assert report.component == "synchrony"
    assert len(report_calls) == 18
    for kwargs in report_calls:
        metric_prefix, entity_label = str(kwargs["metric_name"]).split(maxsplit=1)
        epoch_index = _EPOCH_NAMES.index(str(kwargs["epoch_name"]))
        band_index = _BAND_NAMES.index(str(kwargs["band_name"]))
        if metric_prefix == "ITPC":
            entity_index = ("PFC", "HPC1").index(entity_label)
            array_prefix = "itpc"
        else:
            entity_index = ("PFC-HPC1",).index(entity_label)
            array_prefix = "ispc"
        np.testing.assert_array_equal(
            kwargs["observed_estimates"],
            loaded["synchrony"][f"{array_prefix}_band_mean"][:, entity_index, epoch_index, band_index],
        )
        np.testing.assert_array_equal(
            kwargs["bootstrap_quantiles"],
            np.vstack(
                (
                    loaded["synchrony"][f"{array_prefix}_ci_low"][:, entity_index, epoch_index, band_index],
                    loaded["synchrony"][f"{array_prefix}_bootstrap_q25"][:, entity_index, epoch_index, band_index],
                    loaded["synchrony"][f"{array_prefix}_bootstrap_median"][:, entity_index, epoch_index, band_index],
                    loaded["synchrony"][f"{array_prefix}_bootstrap_q75"][:, entity_index, epoch_index, band_index],
                    loaded["synchrony"][f"{array_prefix}_ci_high"][:, entity_index, epoch_index, band_index],
                )
            ),
        )
        np.testing.assert_array_equal(
            kwargs["selected_trial_counts"],
            loaded["synchrony"][f"{array_prefix}_band_trial_count"][:, entity_index, epoch_index, band_index],
        )
        assert kwargs["labels"] == _CONDITION_NAMES
        assert kwargs["bootstrap_count"] == config.phase.bootstrap_count


def test_seeded_synthetic_spike_payload_uses_production_grouped_path_cache_reload_and_plot(
    tmp_path: Path,
) -> None:
    """Run production Spike-phase payload work through final cache reload and plotting.

    This is deliberately separate from the hand-built multi-component payload
    fixture above: it supplies real prepared phase/spike records to
    ``build_spike_phase_payload`` and verifies the final transaction output is
    the source of the plotted representative histogram.
    """
    base = _synthetic_configuration(tmp_path)
    config = replace(
        base,
        unit_population=UnitPopulationConfig(
            label="synthetic_population",
            probe_label="PFC",
            sorter_path=None,
            aligned_spike_path=None,
            selected_channels=(0, 1),
            quality_settings=(),
            stable_unit_ids=("PFC:1", "PFC:2"),
        ),
        ppc=PPCAnalysisConfig(shuffle_count=3, seed=72),
        ppc_execution=PPCExecutionConfig(
            unit_block_size=1,
            shuffle_block_size=1,
            trial_edge_block_size=1,
            worker_count=1,
            checkpoint_enabled=True,
        ),
    )
    trial_df, traces = _synthetic_trials_and_traces()
    prepared_trials = build_prepared_trials(
        trial_df,
        config.trial_filter,
        "choice_time",
        {site.stable_id: np.ones(4, dtype=bool) for site in config.sites},
    )
    phase = _phase_tensor().astype(np.complex64)
    prepared_phase = PreparedPhaseRun(
        trial_indices=np.arange(4, dtype=np.int64),
        alignment_times_s=trial_df["choice_time"].to_numpy(dtype=float),
        prepared_trials=prepared_trials,
        phase_tensor=phase,
        phase_valid=np.ones(phase.shape, dtype=bool),
        relative_time_s=_TIME_S,
        site_valid=np.ones((2, 4), dtype=bool),
        pair_valid=np.ones((1, 4), dtype=bool),
        source_trace=traces,
    )
    prepared_spikes = PreparedSpikeRun(
        unit_ids=("PFC:1", "PFC:2"),
        population_ids=("synthetic_population",),
        trial_spike_trains=tuple(
            TrialRelativeSpikeTrains(
                unit_id=unit_id,
                relative_spike_times=spikes,
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            )
            for unit_id, spikes in (
                ("PFC:1", _trial_spikes(True, 0)),
                ("PFC:2", _trial_spikes(False, 73)),
            )
        ),
    )
    dependencies = PipelineDependencies(
        prepare_power=lambda _: pytest.fail("Power is outside Spike payload integration"),
        prepare_phase=lambda _: prepared_phase,
        prepare_spike=lambda _, __: prepared_spikes,
        build_power_payload=lambda _, __: pytest.fail("Power is outside Spike payload integration"),
        build_synchrony_payload=lambda _, __: pytest.fail("Synchrony is outside Spike payload integration"),
        build_spike_phase_payload=build_spike_phase_payload,
        load_manifest=lambda directory: load_or_initialize_manifest(directory, config),
        write_component=write_component_transaction,
    )

    result = compute_spike_phase_component(config, dependencies)

    assert result.state == "complete"
    assert result.manifest is not None
    loaded = load_component_arrays(
        config.output_directory / "spike_phase.npz", result.manifest, "spike_phase"
    )
    assert loaded["ppc"].shape == (2, len(prepared_trials.condition_names), 2, 3, 20)
    assert loaded["representative_phase_hist_count"].dtype == np.dtype(np.int64)
    figure, _ = plot_ppc_exemplar(
        loaded["frequency_hz"],
        loaded["ppc"][0, 0, 0, 0],
        loaded["preferred_phase_rad"][0, 0, 0, 0],
        loaded["representative_phase_hist_count"][0, 0, 0, 0, 0],
        loaded["phase_bin_edges_rad"],
        loaded["relative_time_s"],
        loaded["source_trace"][0, 0],
        loaded["band_filtered_trace"][0, 0, 0],
        loaded["relative_spike_times_s"],
        "PFC:1",
        0,
        "theta",
        8.0,
        "synthetic grouped payload",
        _plot_context(),
    )
    plt.close(figure)
