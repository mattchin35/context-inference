"""RED contracts for the production-callable Power summary runtime bridge."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import (
    lfp_summary_ppc_kernel,
    lfp_summary_ppc_runtime,
    lfp_summary_runtime,
    spike_lfp_summary,
)
from src.neural_analysis.lfp_summary_io import (
    assess_component_status,
    load_component_arrays,
    load_or_initialize_manifest,
)
from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    PPCExecutionConfig,
    PowerAnalysisConfig,
    UnitPopulationConfig,
    default_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_pipeline import (
    PipelineDependencies,
    compute_power_component,
    compute_spike_phase_component,
)
from src.neural_analysis.lfp_summary_payloads import validate_component_payload
from src.neural_analysis.lfp_summary_preparation import (
    PreparedTrials,
    TrialRelativeSpikeTrains,
    build_common_event_grid,
)
from src.neural_analysis.lfp_summary_runtime import (
    PreparedPowerRun,
    build_power_payload,
    load_configured_trial_table,
    make_power_pipeline_dependencies,
    prepare_power_run,
)


def _config(output_directory: Path):
    """Return one validated single-site 2,500-Hz Power configuration.

    Parameters
    ----------
    output_directory : pathlib.Path
        Empty cache directory for the test transaction.

    Returns
    -------
    LFPSummaryConfig
        Immutable configuration with 4-s whole event windows, 10-s references,
        2-Hz PSD coordinates, and cached 500-Hz source traces in uV.
    """
    site = LFPSiteConfig(
        "PFC",
        "PFC",
        "spikeglx",
        Path("pfc.bin"),
        None,
        "PFC",
        5,
        "uV",
        2500.0,
    )
    return replace(
        default_lfp_summary_config(),
        session_id="synthetic-power-runtime",
        output_directory=output_directory,
        sites=(site,),
        site_pairs=(),
        power=PowerAnalysisConfig(notch_enabled=False),
    )


def _trial_table() -> pd.DataFrame:
    """Return three trials with one missing active alignment timestamp.

    Returns
    -------
    pandas.DataFrame
        One row per trial. Times are absolute seconds and condition columns use
        the project's authoritative integer/value conventions.
    """
    return pd.DataFrame(
        {
            "start_time": [20.0, 30.0, 40.0],
            "choice_time": [21.0, np.nan, 41.0],
            "experimenter_reward_given": [0, 0, 0],
            "correct": [1, 1, 0],
            "reward": [0, 1, 0],
            "action": [0, 1, 0],
            "state_int": [0, 1, 0],
        }
    )


def _spike_phase_payload_config(output_directory: Path):
    """Return a compact valid configuration for grouped payload integration.

    Parameters
    ----------
    output_directory : pathlib.Path
        Pytest-owned final component directory. Grouped resumable work remains
        under its sibling ``lfp_summary_work`` directory.

    Returns
    -------
    LFPSummaryConfig
        Two-site, two-unit configuration with two phase frequencies, three
        declared PPC epochs, three deterministic shuffles, and the unchanged
        two representative 8/40-Hz histogram bands.
    """
    base = default_lfp_summary_config()
    site = base.sites[0]
    return replace(
        base,
        session_id="synthetic-spike-payload",
        session_path=output_directory.parent,
        output_directory=output_directory,
        sites=base.sites[:2],
        site_pairs=(base.site_pairs[0],),
        unit_population=UnitPopulationConfig(
            label="selected_population",
            probe_label="PFC",
            sorter_path=None,
            aligned_spike_path=None,
            selected_channels=(0, 1),
            quality_settings=(),
            stable_unit_ids=("PFC:1", "PFC:2"),
        ),
        phase=replace(base.phase, frequency_hz=(8.0, 40.0)),
        ppc=replace(base.ppc, shuffle_count=3, seed=211),
        ppc_execution=PPCExecutionConfig(
            unit_block_size=1,
            shuffle_block_size=1,
            trial_edge_block_size=1,
            worker_count=1,
            checkpoint_enabled=True,
        ),
    )


def _spike_phase_payload_inputs(config: object) -> tuple[object, object]:
    """Return compact production prepared inputs for payload integration tests.

    Parameters
    ----------
    config : LFPSummaryConfig
        Configuration returned by :func:`_spike_phase_payload_config`.

    Returns
    -------
    tuple[PreparedPhaseRun, PreparedSpikeRun]
        Two-site complex64/Boolean phase with axes ``(site=2, frequency=2,
        trial=3, time=2000)`` and two units' finite event-relative seconds
        spikes for every trial. The stable unit order is ``("PFC:1",
        "PFC:2")``. Conditions retain ordered overlapping members
        ``("all", "late")`` on stable trial rows ``(11, 23, 37)``.
    """
    time_s = build_common_event_grid(-2.0, 2.0, float(config.phase.output_rate_hz))
    stable_rows = np.array([11, 23, 37], dtype=np.int64)
    frequency_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    angle = (
        np.arange(2, dtype=float)[:, None, None, None] * 0.23
        + frequency_hz[None, :, None, None] * time_s[None, None, None, :] * 0.07
        + np.arange(stable_rows.size, dtype=float)[None, None, :, None] * 0.19
    )
    phase = np.exp(1j * angle).astype(np.complex64)
    prepared_trials = PreparedTrials(
        condition_names=("all", "late"),
        condition_membership=np.array(
            [[True, False], [True, True], [True, True]], dtype=bool
        ),
        filter_membership=np.ones(stable_rows.size, dtype=bool),
        user_excluded=np.zeros(stable_rows.size, dtype=bool),
        objective_valid=np.ones(stable_rows.size, dtype=bool),
        objective_exclusion_reason=np.full(stable_rows.size, "", dtype="<U1"),
        user_exclusion_reason=np.full(stable_rows.size, "", dtype="<U1"),
        site_validity={
            site.stable_id: np.ones(stable_rows.size, dtype=bool)
            for site in config.sites
        },
        pair_validity={
            tuple(config.site_pairs[0]): np.ones(stable_rows.size, dtype=bool),
        },
    )
    prepared_phase = lfp_summary_runtime.PreparedPhaseRun(
        trial_indices=stable_rows,
        alignment_times_s=np.arange(stable_rows.size, dtype=float),
        prepared_trials=prepared_trials,
        phase_tensor=phase,
        phase_valid=np.ones(phase.shape, dtype=bool),
        relative_time_s=time_s,
        site_valid=np.ones((2, stable_rows.size), dtype=bool),
        pair_valid=np.ones((1, stable_rows.size), dtype=bool),
        source_trace=np.zeros((2, stable_rows.size, time_s.size), dtype=float),
    )
    before = np.linspace(-1.8, -0.2, 30)
    after = np.linspace(0.2, 1.8, 30)
    spike_trains = tuple(
        np.concatenate((before + shift, after + shift))
        for shift in (-0.001, 0.001, 0.003)
    )
    second_unit_spikes = tuple(
        np.concatenate((before + shift * 0.5, after + shift * 1.5))[:52]
        for shift in (-0.001, 0.001, 0.003)
    )
    prepared_spikes = lfp_summary_runtime.PreparedSpikeRun(
        unit_ids=("PFC:1", "PFC:2"),
        population_ids=("selected_population",),
        trial_spike_trains=(
            TrialRelativeSpikeTrains(
                unit_id="PFC:1",
                relative_spike_times=spike_trains,
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            ),
            TrialRelativeSpikeTrains(
                unit_id="PFC:2",
                relative_spike_times=second_unit_spikes,
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            ),
        ),
    )
    return prepared_phase, prepared_spikes


def test_configured_trial_table_loader_reads_the_active_ct026_csv(tmp_path: Path) -> None:
    """The production loader must use the configured table path, not a hidden default.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary session directory containing a minimal CT026 trial CSV.
    """
    trial_csv = tmp_path / "CT026_2026-08-01_130853_trials.csv"
    expected = _trial_table()
    expected.to_csv(trial_csv, index=False)
    config = replace(_config(tmp_path / "cache"), trial_table_path=trial_csv)

    loaded = load_configured_trial_table(config)

    pd.testing.assert_frame_equal(loaded, expected)


def _normalized_loader(calls: list[dict[str, object]]):
    """Return a normalized loader with exact event and pre-session coverage.

    Parameters
    ----------
    calls : list[dict[str, object]]
        Mutable records populated with normalized adapter keyword arguments.

    Returns
    -------
    Callable
        Loader accepting ``site``, absolute ``alignment_time_s``, and relative
        half-open ``window``; it returns `(time_s, trace_uv, 2500.0)`.
    """
    def loader(**kwargs: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Record inputs and generate deterministic 8/40-Hz uV source samples."""
        calls.append(kwargs)
        alignment_time_s = float(kwargs["alignment_time_s"])
        start_s, stop_s = kwargs["window"]
        time_s = np.arange(float(start_s), float(stop_s), 1.0 / 2500.0)
        absolute_time_s = alignment_time_s + time_s
        values_uv = (
            2.0 * np.sin(2.0 * np.pi * 8.0 * absolute_time_s)
            + np.sin(2.0 * np.pi * 40.0 * absolute_time_s)
        )
        return time_s, values_uv, 2500.0

    return loader


def test_prepare_power_run_loads_active_trials_whole_and_exact_presession_windows(
    tmp_path: Path,
) -> None:
    """Preparation uses adapters for 4-s aligned traces and a full 10-s baseline."""
    config = _config(tmp_path / "cache")
    calls: list[dict[str, object]] = []
    trial_loader_calls: list[object] = []

    def trial_table_loader(active_config: object) -> pd.DataFrame:
        """Record the active configuration and return the synthetic trial table."""
        trial_loader_calls.append(active_config)
        return _trial_table()

    prepared = prepare_power_run(
        config,
        trial_table_loader,
        spikeglx_loader=_normalized_loader(calls),
    )

    assert isinstance(prepared, PreparedPowerRun)
    assert trial_loader_calls == [config]
    assert prepared.trial_indices.tolist() == [0, 1, 2]
    assert prepared.prepared_trials.objective_valid.tolist() == [True, False, True]
    assert prepared.prepared_trials.filter_membership.dtype == np.dtype(bool)
    assert prepared.site_traces["PFC"].source_trace.shape == (3, 10000)
    assert prepared.site_traces["PFC"].relative_time_s[0] == pytest.approx(-2.0)
    assert prepared.site_traces["PFC"].relative_time_s[-1] == pytest.approx(1.9996)
    assert prepared.site_traces["PFC"].valid.tolist() == [True, False, True]
    assert prepared.presession_traces["PFC"].shape == (25000,)
    assert prepared.presession_time_s["PFC"].shape == (25000,)
    expected_presession_time_s = np.arange(-10.0, 0.0, 1.0 / 2500.0)
    assert np.allclose(prepared.presession_time_s["PFC"], expected_presession_time_s)
    assert len(calls) == 3
    assert [call["window"] for call in calls] == [(-2.0, 2.0), (-2.0, 2.0), (-10.0, 0.0)]
    assert [call["alignment_time_s"] for call in calls] == [21.0, 41.0, 20.0]
    assert all(call["site"] == config.sites[0] for call in calls)
    with pytest.raises(FrozenInstanceError):
        prepared.trial_indices = np.array([0])


def test_build_power_payload_uses_real_cores_and_preserves_cache_axes_units_counts(
    tmp_path: Path,
) -> None:
    """Payload contains every Power cache array, real 8/40-Hz PSD peaks, and QC."""
    config = _config(tmp_path / "cache")
    prepared = prepare_power_run(
        config,
        lambda _: _trial_table(),
        spikeglx_loader=_normalized_loader([]),
    )

    payload = build_power_payload(config, prepared)

    validate_component_payload("power", payload)
    arrays = payload.arrays
    assert arrays["source_trace"].shape == (1, 3, 2000)
    assert arrays["relative_time_s"].shape == (2000,)
    assert arrays["site_voltage_units"].tolist() == ["uV"]
    assert arrays["frequency_hz"].tolist() == list(np.arange(0.0, 1252.0, 2.0))
    assert arrays["epoch_names"].tolist() == ["whole", "before", "after"]
    assert arrays["objective_valid"].tolist() == [[True, False, True]]
    assert np.array_equal(
        arrays["filter_membership"],
        prepared.prepared_trials.filter_membership,
    )
    assert arrays["site_valid"].tolist() == [[True, False, True]]
    assert arrays["exclusion_reason_code"].tolist() == [["", "missing_choice_time", ""]]
    assert arrays["psd_valid"].tolist() == [[[True, True, True], [False] * 3, [True] * 3]]
    omission = arrays["condition_names"].tolist().index("omission")
    assert arrays["condition_trial_count"][omission] == 1
    assert arrays["condition_effective_trial_count"][omission, 0] == 1
    assert arrays["condition_unstable"][omission, 0]
    whole_psd = arrays["psd_linear"][0, 0, 0]
    assert whole_psd[np.flatnonzero(arrays["frequency_hz"] == 8.0)[0]] > whole_psd[
        np.flatnonzero(arrays["frequency_hz"] == 6.0)[0]
    ]
    assert whole_psd[np.flatnonzero(arrays["frequency_hz"] == 40.0)[0]] > whole_psd[
        np.flatnonzero(arrays["frequency_hz"] == 38.0)[0]
    ]
    assert arrays["presession_reference_available"].tolist() == [True]
    assert np.isfinite(arrays["presession_reference_psd_linear"]).all()


def test_power_dependencies_commit_reload_and_report_unsupported_components(
    tmp_path: Path,
) -> None:
    """Factory uses real atomic I/O and exposes unsupported work without fake arrays."""
    config = _config(tmp_path / "cache")
    dependencies = make_power_pipeline_dependencies(
        trial_table_loader=lambda _: _trial_table(),
        spikeglx_loader=_normalized_loader([]),
    )

    result = compute_power_component(config, dependencies)
    manifest = load_or_initialize_manifest(config.output_directory, config)
    arrays = load_component_arrays(config.output_directory / "power.npz", manifest, "power")
    status = assess_component_status(config.output_directory, "power", config, manifest)

    assert result.state == "complete"
    assert result.stage == "complete_power"
    assert status.status == "compatible"
    assert arrays["source_trace"].shape == (1, 3, 2000)
    manifest_text = (config.output_directory / "manifest.json").read_text(encoding="ascii")
    assert "streamlit" not in manifest_text.lower()
    power_entry = manifest["components"]["power"]
    assert power_entry["array_schema"]["source_trace"] == {
        "axes": ["site", "trial", "time"],
        "units": "source-voltage-unit",
    }
    assert not {"arrays", "array_values", "source_trace_values"}.intersection(power_entry)
    with pytest.raises((NotImplementedError, RuntimeError), match="synchrony|spike|unsupported"):
        dependencies.prepare_phase(config)
    with pytest.raises((NotImplementedError, RuntimeError), match="synchrony|spike|unsupported"):
        dependencies.prepare_spike(config, object())


def test_cached_500_hz_trace_uses_antialias_filter_before_decimation(tmp_path: Path) -> None:
    """A 600-Hz native component must not alias into the 500-Hz inspection trace."""

    config = _config(tmp_path / "cache")

    def high_frequency_loader(**kwargs: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Return a native 600-Hz signal above the cached 250-Hz Nyquist limit."""

        start_s, stop_s = kwargs["window"]
        time_s = np.arange(float(start_s), float(stop_s), 1.0 / 2500.0)
        values_uv = np.sin(2.0 * np.pi * 600.0 * time_s)
        return time_s, values_uv, 2500.0

    prepared = prepare_power_run(
        config,
        lambda _: _trial_table(),
        spikeglx_loader=high_frequency_loader,
    )
    cached_trace = build_power_payload(config, prepared).arrays["source_trace"]

    valid_cached_rows = cached_trace[0, [0, 2]]
    assert np.sqrt(np.mean(valid_cached_rows**2)) < 0.05


def test_spike_phase_payload_uses_one_grouped_executor_without_legacy_or_histogram_resampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public payload must consume one grouped component result unchanged.

    The fixture includes overlapping ordered conditions, all three PPC epochs,
    and source spikes that are deliberately between canonical time samples.
    It therefore protects the grouped executor's S1 exact-grid histogram
    decision rather than recreating the retired payload sampler.
    """
    config = _spike_phase_payload_config(tmp_path / "cache")
    prepared_phase, prepared_spikes = _spike_phase_payload_inputs(config)
    grouped_calls: list[object] = []
    original_grouped_executor = lfp_summary_ppc_runtime.execute_grouped_ppc_component

    def recording_grouped_executor(**kwargs: object) -> object:
        """Record the component call while preserving the real grouped result."""
        assert set(kwargs) == {
            "config",
            "execution",
            "prepared_phase",
            "prepared_spikes",
            "work_root",
            "progress_callback",
        }
        assert kwargs["config"] is config
        assert kwargs["execution"] is config.ppc_execution
        assert kwargs["prepared_phase"] is prepared_phase
        assert kwargs["prepared_spikes"] is prepared_spikes
        grouped_calls.append(original_grouped_executor(**kwargs))
        return grouped_calls[-1]

    def forbidden_legacy_executor(*_: object, **__: object) -> object:
        """Fail if payload integration retains any condition/site/epoch job loop."""
        raise AssertionError("payload invoked legacy execute_ppc_blocks")

    def forbidden_histogram_resampling(*_: object, **__: object) -> object:
        """Fail if payload integration samples observed phase after grouped work."""
        raise AssertionError("payload resampled representative histograms")

    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_grouped_ppc_component",
        recording_grouped_executor,
    )
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_ppc_blocks",
        forbidden_legacy_executor,
    )
    monkeypatch.setattr(
        lfp_summary_runtime,
        "_sample_observed_trial_phase",
        forbidden_histogram_resampling,
    )
    monkeypatch.setattr(
        spike_lfp_summary,
        "build_representative_phase_histograms",
        forbidden_histogram_resampling,
    )

    payload = lfp_summary_runtime._build_spike_phase_payload(
        config,
        prepared_phase,
        prepared_spikes,
        progress_callback=None,
    )

    assert len(grouped_calls) == 1
    grouped = grouped_calls[0]
    expected_job_order = [
        (condition_name, site.stable_id, epoch_name)
        for site in config.sites
        for condition_name in ("all", "late")
        for epoch_name in ("whole", "before", "after")
    ]
    assert [
        (job.condition_name, job.site_id, job.epoch_name)
        for job in grouped.component_plan.job_plans
    ] == expected_job_order
    for job in grouped.component_plan.job_plans:
        expected_seed = (
            config.ppc.seed
            + job.condition_index * 1_000_000
            + job.site_index * 10_000
            + job.epoch_index * 100
        )
        assert job.schedule_seed == expected_seed
        np.testing.assert_array_equal(
            job.schedule,
            spike_lfp_summary.generate_trial_derangement_schedule(
                job.selected_trial_rows.size,
                config.ppc.shuffle_count,
                seed=expected_seed,
            )
            if job.selected_trial_rows.size >= 2
            else np.empty((0, job.selected_trial_rows.size), dtype=np.int64),
        )
    for name in lfp_summary_ppc_runtime._SUMMARY_FIELDS:
        np.testing.assert_equal(payload.arrays[name], grouped.summary_arrays[name])
    np.testing.assert_array_equal(
        payload.arrays["representative_phase_hist_count"],
        grouped.summary_arrays["representative_phase_histogram_count"],
    )
    assert payload.arrays["condition_names"].tolist() == ["all", "late"]
    assert payload.arrays["site_ids"].tolist() == ["PFC", "HPC1"]
    assert payload.arrays["epoch_names"].tolist() == ["whole", "before", "after"]
    assert payload.arrays["ppc"].shape == (2, 2, 2, 3, 2)
    assert payload.arrays["representative_phase_hist_count"].shape == (2, 2, 2, 3, 2, 2)
    assert payload.arrays["spike_count"].dtype == np.dtype(np.int64)
    assert payload.arrays["representative_phase_hist_count"].dtype == np.dtype(np.int64)
    assert np.isfinite(payload.arrays["ppc_band_mean"]).all()
    assert not np.array_equal(
        payload.arrays["ppc_band_mean"][0], payload.arrays["ppc_band_mean"][1]
    )
    trial_counts = np.zeros((2, 2, 2, 3, 3), dtype=np.int64)
    membership = lfp_summary_runtime._analysis_condition_membership(
        prepared_phase.prepared_trials
    )
    for unit_index, train in enumerate(prepared_spikes.trial_spike_trains):
        for condition_index in range(2):
            for site_index in range(2):
                selected_positions = np.flatnonzero(membership[:, condition_index])
                for epoch_index, window in enumerate(
                    lfp_summary_runtime._selected_ppc_epoch_windows(config).values()
                ):
                    for position in selected_positions:
                        trial_counts[unit_index, condition_index, site_index, epoch_index, position] = (
                            lfp_summary_runtime._spikes_in_epoch(
                                train.relative_spike_times[int(position)], window
                            ).size
                        )
    for index in np.ndindex(2, 2, 3, 2):
        condition_index, site_index, epoch_index, band_index = index
        selection = spike_lfp_summary.select_ppc_exemplars(
            unit_ids=prepared_spikes.unit_ids,
            band_ppc=payload.arrays["ppc_band_mean"][:, condition_index, site_index, epoch_index, band_index],
            trial_indices=prepared_phase.trial_indices,
            trial_spike_counts=trial_counts[:, condition_index, site_index, epoch_index],
        )
        assert payload.arrays["selected_low_unit_ids"][index] == selection.low_unit_id
        assert payload.arrays["selected_high_unit_ids"][index] == selection.high_unit_id
        expected_trial = next(
            (
                selection.illustrative_trial_index_by_unit[unit_id]
                for unit_id in (selection.high_unit_id, selection.low_unit_id)
                if unit_id in selection.illustrative_trial_index_by_unit
            ),
            -1,
        )
        assert payload.arrays["illustrative_trial_indices"][index] == expected_trial
    validate_component_payload("spike_phase", payload)
    assert payload.post_commit_cleanup is not None
    assert grouped.run_directory.exists()
    payload.post_commit_cleanup()
    assert not grouped.run_directory.exists()


def test_grouped_payload_histograms_follow_exact_grid_and_match_safe_legacy_reference(
    tmp_path: Path,
) -> None:
    """Exact samples survive while near-grid samples need both valid neighbors.

    The observed grouped histogram is the payload's only histogram source. At
    8 and 40 Hz its bin total must equal the accepted observed PPC count; a
    separate all-valid fixture is compared to the legacy sampler only where
    its historical ``isclose`` rule cannot change the answer.
    """
    config = _spike_phase_payload_config(tmp_path / "cache")
    prepared_phase, prepared_spikes = _spike_phase_payload_inputs(config)
    zero_index = int(np.flatnonzero(prepared_phase.relative_time_s == 0.0)[0])
    exact = np.array([0.0])
    near = np.array([np.nextafter(0.0, -np.inf)])
    invalid_validity = prepared_phase.phase_valid.copy()
    invalid_validity[:, :, 1, zero_index] = False
    constrained_phase = replace(
        prepared_phase,
        phase_tensor=np.ones_like(prepared_phase.phase_tensor),
        phase_valid=invalid_validity,
    )
    constrained_spikes = replace(
        prepared_spikes,
        trial_spike_trains=tuple(
            TrialRelativeSpikeTrains(
                unit_id=train.unit_id,
                relative_spike_times=(exact, near, np.empty(0, dtype=float)),
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            )
            for train in prepared_spikes.trial_spike_trains
        ),
    )
    exact_geometry = lfp_summary_ppc_kernel.build_source_trial_spike_geometry(
        phase_time_s=constrained_phase.relative_time_s,
        phase_sampling_rate_hz=config.phase.output_rate_hz,
        source_trial_index=11,
        unit_ids=("PFC:1",),
        unit_trial_spike_times_s=(exact,),
        segment_bounds_s=((-2.0, 0.0), (0.0, 2.0)),
    )
    near_geometry = lfp_summary_ppc_kernel.build_source_trial_spike_geometry(
        phase_time_s=constrained_phase.relative_time_s,
        phase_sampling_rate_hz=config.phase.output_rate_hz,
        source_trial_index=23,
        unit_ids=("PFC:1",),
        unit_trial_spike_times_s=(near,),
        segment_bounds_s=((-2.0, 0.0), (0.0, 2.0)),
    )
    assert exact_geometry.exact_sample.tolist() == [True]
    assert near_geometry.exact_sample.tolist() == [False]
    assert abs(float(near[0])) < 1e-12
    grouped = lfp_summary_ppc_runtime.execute_grouped_ppc_component(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=constrained_phase,
        prepared_spikes=constrained_spikes,
        work_root=tmp_path / "work-invalid-right",
    )
    for site_index in range(2):
        for unit_index in range(2):
            counts = grouped.summary_arrays["spike_count"][unit_index, 0, site_index, 0]
            histogram = grouped.summary_arrays["representative_phase_histogram_count"][
                unit_index, 0, site_index, 0
            ]
            np.testing.assert_array_equal(counts, np.array([1, 1], dtype=np.int64))
            np.testing.assert_array_equal(histogram.sum(axis=1), counts)

    safe = replace(constrained_phase, phase_valid=np.ones_like(constrained_phase.phase_valid))
    safe_grouped = lfp_summary_ppc_runtime.execute_grouped_ppc_component(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=safe,
        prepared_spikes=constrained_spikes,
        work_root=tmp_path / "work-safe-reference",
    )
    selected_positions = np.array([0, 1, 2], dtype=np.int64)
    for site_index in range(2):
        phase_by_trial = np.moveaxis(safe.phase_tensor[site_index], 1, 0)
        valid_by_trial = np.moveaxis(safe.phase_valid[site_index], 1, 0)
        for unit_index, train in enumerate(constrained_spikes.trial_spike_trains):
            sampled, sampled_valid, sampled_rows = lfp_summary_runtime._sample_observed_trial_phase(
                safe.relative_time_s,
                phase_by_trial[selected_positions],
                valid_by_trial[selected_positions],
                train.relative_spike_times,
                safe.trial_indices[selected_positions],
            )
            legacy = spike_lfp_summary.build_representative_phase_histograms(
                frequencies_hz=np.asarray(config.phase.frequency_hz, dtype=float),
                spike_phase_vectors=sampled,
                valid_mask=sampled_valid,
                trial_indices=sampled_rows,
                phase_bin_edges_rad=np.asarray(config.ppc.phase_bin_edges_rad, dtype=float),
            )
            np.testing.assert_array_equal(
                safe_grouped.summary_arrays["representative_phase_histogram_count"][
                    unit_index, 0, site_index, 0
                ],
                legacy.spike_count_by_band,
            )


def test_spike_phase_grouped_payload_writer_failure_keeps_resumable_work_and_never_publishes_partial_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed final transaction retains grouped work for the successful retry."""
    config = _spike_phase_payload_config(tmp_path / "cache")
    prepared_phase, prepared_spikes = _spike_phase_payload_inputs(config)
    grouped_calls: list[object] = []
    write_attempts: list[object] = []
    original_grouped_executor = lfp_summary_ppc_runtime.execute_grouped_ppc_component

    def grouped_executor(**kwargs: object) -> object:
        """Run real grouped work and record cold/retry checkpoint identities."""
        assert kwargs["work_root"] == config.output_directory.parent / "lfp_summary_work"
        result = original_grouped_executor(**kwargs)
        grouped_calls.append(result)
        return result

    def forbidden_legacy_executor(*_: object, **__: object) -> object:
        """Fail if final-transaction handling routes through a legacy PPC job."""
        raise AssertionError("payload invoked legacy execute_ppc_blocks")

    def writer(
        directory: Path,
        component: str,
        arrays: dict[str, np.ndarray],
        manifest: dict[str, object],
    ) -> None:
        """Fail once before publication, then emulate a completed atomic write."""
        assert component == "spike_phase"
        assert arrays["representative_phase_hist_count"].dtype == np.dtype(np.int64)
        write_attempts.append(manifest)
        if len(write_attempts) == 1:
            raise OSError("injected final transaction failure")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "spike_phase.npz").write_bytes(b"complete")

    dependencies = PipelineDependencies(
        prepare_power=lambda _: pytest.fail("power preparation is out of scope"),
        prepare_phase=lambda _: pytest.fail("prepared phase must be reused"),
        prepare_spike=lambda _, __: prepared_spikes,
        build_power_payload=lambda _, __: pytest.fail("power payload is out of scope"),
        build_synchrony_payload=lambda _, __: pytest.fail("synchrony payload is out of scope"),
        build_spike_phase_payload=lfp_summary_runtime.build_spike_phase_payload,
        load_manifest=lambda _: {"session_id": config.session_id, "components": {}},
        write_component=writer,
    )
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_grouped_ppc_component",
        grouped_executor,
    )
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_ppc_blocks",
        forbidden_legacy_executor,
    )

    failed = compute_spike_phase_component(
        config,
        dependencies,
        prepared_phase=prepared_phase,
    )

    assert failed.state == "failed"
    assert failed.stage == "write_spike_phase"
    assert not (config.output_directory / "spike_phase.npz").exists()
    assert len(grouped_calls) == 1
    first = grouped_calls[0]
    assert first.resumed_block_ids == ()
    assert first.run_directory.exists()

    def fail_if_reduced(*_: object, **__: object) -> object:
        """A fully checkpointed retry must resume without sampling a new edge."""
        raise AssertionError("grouped retry recomputed a checkpointed reducer block")

    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "compute_segmented_edge_statistics",
        fail_if_reduced,
    )
    completed = compute_spike_phase_component(
        config,
        dependencies,
        prepared_phase=prepared_phase,
    )

    assert completed.state == "complete"
    assert len(grouped_calls) == 2
    assert len(write_attempts) == 2
    second = grouped_calls[1]
    assert second.run_fingerprint == first.run_fingerprint
    assert second.completed_block_ids == first.completed_block_ids
    assert second.resumed_block_ids == second.completed_block_ids
    assert (config.output_directory / "spike_phase.npz").read_bytes() == b"complete"
    assert not first.run_directory.exists()
