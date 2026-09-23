"""RED contracts for the production-callable Synchrony runtime bridge."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import lfp_summary_runtime
from src.neural_analysis.lfp_phase_clustering import PhaseTrialTensor
from src.neural_analysis.lfp_summary_io import (
    assess_component_status,
    load_component_arrays,
    load_or_initialize_manifest,
)
from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    LFPSummaryConfig,
    PhaseAnalysisConfig,
    default_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_payloads import validate_component_payload
from src.neural_analysis.lfp_summary_pipeline import compute_synchrony_component
from src.neural_analysis.lfp_summary_runtime import (
    PreparedPhaseRun,
    build_synchrony_payload,
    make_synchrony_pipeline_dependencies,
    prepare_phase_run,
)
from src.neural_analysis.lfp_synchrony_summary import (
    bootstrap_phase_clustering_bands,
)


_FREQUENCY_HZ = (8.0, 40.0)
_SITE_OFFSET_RAD = 0.6


def _config(output_directory: Path):
    """Return a two-site Synchrony configuration with small seeded bootstraps.

    Parameters
    ----------
    output_directory : pathlib.Path
        Pytest-owned cache directory.

    Returns
    -------
    LFPSummaryConfig
        Two 500-Hz sites, one ordered pair, two phase frequencies in Hz, and
        eight deterministic bootstrap resamples.
    """
    sites = (
        LFPSiteConfig(
            "PFC", "PFC", "spikeglx", Path("pfc.bin"), None, "PFC", 5, "uV", 500.0
        ),
        LFPSiteConfig(
            "HPC1",
            "HPC1",
            "spikeglx",
            Path("hpc.bin"),
            None,
            "HPC1",
            222,
            "uV",
            500.0,
        ),
    )
    phase = PhaseAnalysisConfig(
        frequency_hz=_FREQUENCY_HZ,
        output_rate_hz=500.0,
        notch_enabled=False,
        bootstrap_count=8,
        seed=17,
    )
    return replace(
        default_lfp_summary_config(),
        session_id="synthetic-synchrony-runtime",
        output_directory=output_directory,
        sites=sites,
        site_pairs=(("PFC", "HPC1"),),
        phase=phase,
    )


def _trial_table() -> pd.DataFrame:
    """Return three finite aligned trials with established condition columns.

    Returns
    -------
    pandas.DataFrame
        Shape ``(trial=3, column)`` with absolute event times in seconds and
        integer action/state/reward conventions.
    """
    return pd.DataFrame(
        {
            "start_time": [9.0, 19.0, 29.0],
            "choice_time": [10.0, 20.0, 30.0],
            "experimenter_reward_given": [0, 0, 0],
            "correct": [1, 0, 1],
            "reward": [1, 0, 1],
            "action": [0, 1, 0],
            "state_int": [0, 0, 0],
        }
    )


def _trace_loader(**kwargs: object) -> tuple[np.ndarray, np.ndarray, float]:
    """Return one exact 500-Hz four-second source trace in uV.

    Parameters
    ----------
    **kwargs : object
        Normalized site, absolute alignment seconds, and relative window.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        Relative seconds, nonconstant uV samples, and 500 samples/second.
    """
    start_s, stop_s = kwargs["window"]
    time_s = np.arange(float(start_s), float(stop_s), 1.0 / 500.0)
    alignment_s = float(kwargs["alignment_time_s"])
    values_uv = np.sin(2.0 * np.pi * 8.0 * (time_s + alignment_s))
    return time_s, values_uv, 500.0


def _block_loader_factory(
    calls: list[tuple[str, float, float]],
) -> Callable[[LFPSiteConfig], Callable[[float, float], tuple[np.ndarray, np.ndarray, float]]]:
    """Return injected continuous block loaders with absolute-second outputs.

    Parameters
    ----------
    calls : list[tuple[str, float, float]]
        Mutable records of site id and requested absolute second bounds.

    Returns
    -------
    Callable
        Site factory returning a ``(start_s, stop_s)`` continuous LFP loader.
    """
    def factory(site: LFPSiteConfig):
        """Bind one stable site id into its synthetic continuous loader."""
        def load(start_s: float, stop_s: float) -> tuple[np.ndarray, np.ndarray, float]:
            """Return absolute time and uV arrays at 500 samples/second."""
            calls.append((site.stable_id, start_s, stop_s))
            time_s = np.arange(start_s, stop_s, 1.0 / 500.0)
            values_uv = np.sin(2.0 * np.pi * 8.0 * time_s)
            return time_s, values_uv, 500.0

        return load

    return factory


def _phase_tensor_builder(calls: list[dict[str, object]]):
    """Return a builder that drops only HPC1 trial row two.

    Parameters
    ----------
    calls : list[dict[str, object]]
        Captured keyword contracts for each site call.

    Returns
    -------
    Callable
        Existing phase-builder signature returning one-site complex64 tensors.
    """
    def build(**kwargs: object) -> PhaseTrialTensor:
        """Build locked phase with an ordered PFC-minus-HPC1 offset."""
        calls.append(kwargs)
        block_loader = kwargs["block_loader"]
        block_loader(0.0, 0.02)
        trial_indices = np.asarray(kwargs["trial_indices"], dtype=np.int64)
        site_index = len(calls) - 1
        retained = trial_indices if site_index == 0 else trial_indices[:2]
        relative_time_s = np.arange(-2.0, 2.0, 1.0 / 500.0)
        frequencies_hz = np.asarray(kwargs["frequencies_hz"], dtype=float)
        angle = (
            2.0
            * np.pi
            * frequencies_hz[:, None]
            * relative_time_s[None, :]
            - site_index * _SITE_OFFSET_RAD
        )
        one_trial = np.exp(1j * angle).astype(np.complex64)
        phase = np.repeat(one_trial[:, None, :], retained.size, axis=1)[None]
        return PhaseTrialTensor(
            phase=phase,
            valid=np.ones_like(phase, dtype=bool),
            relative_time_s=relative_time_s,
            trial_indices=retained,
            excluded_trial_indices=trial_indices[retained.size :],
        )

    return build


def _prepared_phase(tmp_path: Path) -> tuple[LFPSummaryConfig, PreparedPhaseRun]:
    """Prepare one injected phase run and return its config and products.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned cache root.

    Returns
    -------
    tuple[LFPSummaryConfig, PreparedPhaseRun]
        Active config and full-trial-axis phase/source products.
    """
    config = _config(tmp_path / "cache")
    prepared = prepare_phase_run(
        config,
        lambda _: _trial_table(),
        site_phase_tensor_builder=_phase_tensor_builder([]),
        block_loader_factory=_block_loader_factory([]),
        spikeglx_loader=_trace_loader,
    )
    return config, prepared


def _prepared_phase_with_distinct_band_validity(
    tmp_path: Path,
) -> tuple[LFPSummaryConfig, PreparedPhaseRun]:
    """Return a phase seam whose numerical validity differs by epoch and band.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned cache root.

    Returns
    -------
    tuple[LFPSummaryConfig, PreparedPhaseRun]
        A three-trial ``correct_rewarded`` seam. Both sites and their ordered
        pair retain three selected trials in whole/before but only two in the
        chosen band/after combinations. Trial-specific unit phases make all
        saved bootstrap interior summaries distinguishable.
    """
    config, prepared = _prepared_phase(tmp_path)
    phase = np.zeros_like(prepared.phase_tensor)
    valid = np.zeros_like(prepared.phase_valid, dtype=bool)
    before = prepared.relative_time_s < 0.0
    after = ~before

    def set_phase(
        site_index: int,
        frequency_index: int,
        trial_index: int,
        time_mask: np.ndarray,
        angle_rad: float,
    ) -> None:
        """Set one fixture phase segment without making invalid samples finite."""
        phase[site_index, frequency_index, trial_index, time_mask] = np.complex64(
            np.exp(1j * angle_rad)
        )
        valid[site_index, frequency_index, trial_index, time_mask] = True

    # Theta: all three trials contribute before, while the third is absent
    # after. Gamma: trial zero is after-only and trial two is before-only.
    for site_index, angles in enumerate(((0.0, 1.0, 2.4), (0.2, 1.5, -1.8))):
        for trial_index, angle_rad in enumerate(angles):
            set_phase(site_index, 0, trial_index, before, angle_rad)
        for trial_index, angle_rad in enumerate(angles[:2]):
            set_phase(site_index, 0, trial_index, after, angle_rad)
        set_phase(site_index, 1, 0, after, angles[0])
        set_phase(site_index, 1, 1, before | after, angles[1])
        set_phase(site_index, 1, 2, before, angles[2])

    condition_membership = prepared.prepared_trials.condition_membership.copy()
    condition_index = prepared.prepared_trials.condition_names.index("correct_rewarded")
    condition_membership[:, condition_index] = True
    prepared_trials = replace(
        prepared.prepared_trials,
        condition_membership=condition_membership,
    )
    return config, replace(
        prepared,
        prepared_trials=prepared_trials,
        phase_tensor=phase,
        phase_valid=valid,
        site_valid=np.ones_like(prepared.site_valid, dtype=bool),
        pair_valid=np.ones_like(prepared.pair_valid, dtype=bool),
    )


def test_prepare_phase_run_preserves_full_trial_axis_and_pair_specific_missingness(
    tmp_path: Path,
) -> None:
    """A missing HPC1 trial must not remove its valid PFC phase samples."""
    config = _config(tmp_path / "cache")
    phase_calls: list[dict[str, object]] = []
    block_calls: list[tuple[str, float, float]] = []

    prepared = prepare_phase_run(
        config,
        lambda _: _trial_table(),
        site_phase_tensor_builder=_phase_tensor_builder(phase_calls),
        block_loader_factory=_block_loader_factory(block_calls),
        spikeglx_loader=_trace_loader,
    )

    assert isinstance(prepared, PreparedPhaseRun)
    assert prepared.trial_indices.tolist() == [0, 1, 2]
    assert prepared.phase_tensor.shape == (2, 2, 3, 2000)
    assert prepared.phase_tensor.dtype == np.dtype(np.complex64)
    assert prepared.phase_valid.shape == prepared.phase_tensor.shape
    assert prepared.site_valid.tolist() == [[True, True, True], [True, True, False]]
    assert prepared.pair_valid.tolist() == [[True, True, False]]
    assert prepared.source_trace.shape == (2, 3, 2000)
    assert prepared.relative_time_s[0] == pytest.approx(-2.0)
    assert prepared.relative_time_s[-1] == pytest.approx(1.998)
    assert len(phase_calls) == 2
    assert [call["maximum_core_duration_s"] for call in phase_calls] == [120.0, 120.0]
    assert [call[0] for call in block_calls] == ["PFC", "HPC1"]
    with pytest.raises(FrozenInstanceError):
        prepared.trial_indices = np.array((0,), dtype=np.int64)


def test_prepare_phase_run_prepared_cache_cold_warm_disabled_and_mismatch_paths(
    tmp_path: Path,
) -> None:
    """Warm preparation rebuilds lightweight metadata but reuses mmap phase without Morlet work."""
    cache_root = tmp_path / "work"
    config = _config(tmp_path / "cache")
    cold_calls: list[dict[str, object]] = []
    cold = prepare_phase_run(
        config,
        lambda _: _trial_table(),
        site_phase_tensor_builder=_phase_tensor_builder(cold_calls),
        block_loader_factory=_block_loader_factory([]),
        spikeglx_loader=_trace_loader,
        work_cache_root=cache_root,
    )
    warm_calls: list[dict[str, object]] = []
    warm = prepare_phase_run(
        config,
        lambda _: _trial_table(),
        site_phase_tensor_builder=_phase_tensor_builder(warm_calls),
        block_loader_factory=_block_loader_factory([]),
        spikeglx_loader=_trace_loader,
        work_cache_root=cache_root,
    )
    assert cold.prepared_phase_cache_state == "cold"
    assert warm.prepared_phase_cache_state == "warm"

    assert cold_calls
    assert not warm_calls
    assert isinstance(warm.phase_tensor, np.memmap)
    assert isinstance(warm.phase_valid, np.memmap)
    assert warm.phase_tensor.flags.writeable is False
    assert warm.trial_indices.tolist() == cold.trial_indices.tolist()
    np.testing.assert_array_equal(warm.source_trace, cold.source_trace)

    disabled_calls: list[dict[str, object]] = []
    disabled_config = replace(
        config,
        ppc_execution=replace(config.ppc_execution, prepared_phase_cache_enabled=False),
    )
    prepare_phase_run(
        disabled_config, lambda _: _trial_table(),
        site_phase_tensor_builder=_phase_tensor_builder(disabled_calls),
        block_loader_factory=_block_loader_factory([]), spikeglx_loader=_trace_loader,
        work_cache_root=cache_root,
    )
    mismatch_calls: list[dict[str, object]] = []
    mismatch_config = replace(config, schema_version="cache-mismatch")
    prepare_phase_run(
        mismatch_config, lambda _: _trial_table(),
        site_phase_tensor_builder=_phase_tensor_builder(mismatch_calls),
        block_loader_factory=_block_loader_factory([]), spikeglx_loader=_trace_loader,
        work_cache_root=cache_root,
    )
    assert disabled_calls
    assert mismatch_calls


def test_prepared_phase_cache_identity_supports_missing_alignment_times(
    tmp_path: Path,
) -> None:
    """Missing objective alignments have a deterministic JSON-safe identity."""
    config = _config(tmp_path / "cache")
    trial_indices = np.array([0, 1, 2], dtype=np.int64)
    alignment_times_s = np.array([10.0, np.nan, 30.0], dtype=float)

    first = lfp_summary_runtime._prepared_phase_work_metadata(
        config,
        trial_indices,
        alignment_times_s,
    )
    second = lfp_summary_runtime._prepared_phase_work_metadata(
        config,
        trial_indices.copy(),
        alignment_times_s.copy(),
    )
    changed = lfp_summary_runtime._prepared_phase_work_metadata(
        config,
        trial_indices,
        np.array([10.0, 20.0, np.nan], dtype=float),
    )

    assert first == second
    assert first["scientific_fingerprint"] != changed["scientific_fingerprint"]


def test_phase_bootstrap_recomputes_the_nonlinear_clustering_statistic() -> None:
    """Opposite trials estimate zero ITPC rather than averaging unit magnitudes."""
    phase = np.array(
        (
            np.ones((1, 4), dtype=np.complex64),
            -np.ones((1, 4), dtype=np.complex64),
        )
    )
    valid = np.ones_like(phase, dtype=bool)
    kwargs = {
        "phase_vectors": phase,
        "valid_mask": valid,
        "trial_mask": np.array((True, True)),
        "frequencies_hz": np.array((8.0,)),
        "relative_time_s": np.array((-0.5, -0.25, 0.0, 0.25)),
        "epoch_windows": {"whole": (-0.5, 0.5)},
        "bands": _config(Path("cache")).phase.bands[:1],
        "bootstrap_count": 200,
        "seed": 23,
    }

    first = bootstrap_phase_clustering_bands(**kwargs)
    second = bootstrap_phase_clustering_bands(**kwargs)

    assert first.estimate.shape == (1, 1)
    assert first.estimate[0, 0] == pytest.approx(0.0)
    assert first.ci_low[0, 0] == pytest.approx(0.0)
    assert first.ci_high[0, 0] == pytest.approx(1.0)
    assert first.selected_trial_count[0, 0] == 2
    assert first.unstable[0, 0]
    assert np.array_equal(first.bootstrap_values, second.bootstrap_values)
    assert first.bootstrap_values.shape == (200, 1, 1)


@pytest.mark.parametrize(
    ("phase_vectors", "valid_mask", "expected_unstable"),
    (
        (
            np.repeat(
                np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False))[
                    :, None, None
                ],
                4,
                axis=2,
            ).astype(np.complex64),
            np.ones((12, 1, 4), dtype=bool),
            False,
        ),
        (
            np.array(
                [
                    [[np.nan + 1j * np.nan] * 4],
                    *[
                        [[np.exp(1j * angle)] * 4]
                        for angle in np.linspace(-0.2, 0.2, 8)
                    ],
                ],
                dtype=np.complex64,
            ),
            np.ones((9, 1, 4), dtype=bool),
            True,
        ),
    ),
    ids=("uniform_phase", "concentrated_phase_with_nan_trial"),
)
def test_phase_band_bootstrap_interior_quantiles_equal_actual_finite_draw_percentiles(
    phase_vectors: np.ndarray,
    valid_mask: np.ndarray,
    expected_unstable: bool,
) -> None:
    """All five saved summaries come directly from the finite seeded draw series."""
    summary = bootstrap_phase_clustering_bands(
        phase_vectors=phase_vectors,
        valid_mask=valid_mask,
        trial_mask=np.ones(phase_vectors.shape[0], dtype=bool),
        frequencies_hz=np.array((8.0,)),
        relative_time_s=np.array((-0.5, -0.25, 0.0, 0.25)),
        epoch_windows={"whole": (-0.5, 0.5)},
        bands=_config(Path("cache")).phase.bands[:1],
        bootstrap_count=1_000,
        seed=917,
    )

    finite_draws = summary.bootstrap_values[:, 0, 0]
    finite_draws = finite_draws[np.isfinite(finite_draws)]
    expected = np.percentile(
        finite_draws,
        q=(2.5, 25.0, 50.0, 75.0, 97.5),
        method="linear",
    )

    np.testing.assert_allclose(
        np.array(
            (
                summary.ci_low[0, 0],
                summary.q25[0, 0],
                summary.median[0, 0],
                summary.q75[0, 0],
                summary.ci_high[0, 0],
            )
        ),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert bool(summary.unstable[0, 0]) is expected_unstable
    assert np.isfinite(summary.q25[0, 0])
    assert np.isfinite(summary.median[0, 0])
    assert np.isfinite(summary.q75[0, 0])


def test_phase_band_bootstrap_preserves_the_frozen_seeded_draws_and_endpoints() -> None:
    """Adding interior summaries cannot alter the selected trials or old bootstrap output."""
    phase_vectors = np.repeat(
        np.exp(1j * np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False))[:, None, None],
        4,
        axis=2,
    ).astype(np.complex64)
    summary = bootstrap_phase_clustering_bands(
        phase_vectors=phase_vectors,
        valid_mask=np.ones_like(phase_vectors, dtype=bool),
        trial_mask=np.ones(12, dtype=bool),
        frequencies_hz=np.array((8.0,)),
        relative_time_s=np.array((-0.5, -0.25, 0.0, 0.25)),
        epoch_windows={"whole": (-0.5, 0.5)},
        bands=_config(Path("cache")).phase.bands[:1],
        bootstrap_count=1_000,
        seed=917,
    )

    assert summary.selected_trial_count.tolist() == [[12]]
    assert summary.estimate[0, 0] == pytest.approx(1.850371707708594e-17)
    assert summary.ci_low[0, 0] == pytest.approx(0.043136512709727855)
    assert summary.ci_high[0, 0] == pytest.approx(0.5421184825635231)
    assert sha256(summary.bootstrap_values.tobytes()).hexdigest() == (
        "d96414097044afb30d97cd7d2d2436f8d94b71f06172385c7a7f3eb7e816db01"
    )


def test_phase_band_bootstrap_uses_finite_draws_and_leaves_all_invalid_summaries_nan() -> None:
    """Finite-draw percentiles exclude NaN resamples without inventing all-invalid values."""
    one_valid_one_invalid = np.array(
        [
            [[1.0 + 0.0j] * 4],
            [[np.nan + 1j * np.nan] * 4],
        ],
        dtype=np.complex64,
    )
    kwargs = {
        "trial_mask": np.array((True, True)),
        "frequencies_hz": np.array((8.0,)),
        "relative_time_s": np.array((-0.5, -0.25, 0.0, 0.25)),
        "epoch_windows": {"whole": (-0.5, 0.5)},
        "bands": _config(Path("cache")).phase.bands[:1],
        "bootstrap_count": 32,
        "seed": 12,
    }
    finite_and_nan = bootstrap_phase_clustering_bands(
        phase_vectors=one_valid_one_invalid,
        valid_mask=np.ones_like(one_valid_one_invalid, dtype=bool),
        **kwargs,
    )
    draws = finite_and_nan.bootstrap_values[:, 0, 0]
    finite_draws = draws[np.isfinite(draws)]
    expected = np.percentile(
        finite_draws,
        q=(2.5, 25.0, 50.0, 75.0, 97.5),
        method="linear",
    )

    assert np.any(np.isfinite(draws)) and np.any(np.isnan(draws))
    np.testing.assert_allclose(
        (
            finite_and_nan.ci_low[0, 0],
            finite_and_nan.q25[0, 0],
            finite_and_nan.median[0, 0],
            finite_and_nan.q75[0, 0],
            finite_and_nan.ci_high[0, 0],
        ),
        expected,
        rtol=0.0,
        atol=0.0,
    )

    all_invalid = bootstrap_phase_clustering_bands(
        phase_vectors=np.full_like(one_valid_one_invalid, np.nan + 1j * np.nan),
        valid_mask=np.ones_like(one_valid_one_invalid, dtype=bool),
        **kwargs,
    )
    assert np.isnan(all_invalid.bootstrap_values).all()
    assert all_invalid.selected_trial_count[0, 0] == 0
    assert all_invalid.unstable[0, 0]
    assert np.isnan(
        (
            all_invalid.estimate[0, 0],
            all_invalid.ci_low[0, 0],
            all_invalid.q25[0, 0],
            all_invalid.median[0, 0],
            all_invalid.q75[0, 0],
            all_invalid.ci_high[0, 0],
        )
    ).all()


def test_build_synchrony_payload_populates_exact_schema_and_ordered_offsets(
    tmp_path: Path,
) -> None:
    """Payload axes remain cache-ready and preserve signed PFC-minus-HPC1 phase."""
    config, prepared = _prepared_phase(tmp_path)

    payload = build_synchrony_payload(config, prepared)

    validate_component_payload("synchrony", payload)
    arrays = payload.arrays
    assert "wavelet_coefficients" not in arrays and "phase_tensor" not in arrays
    assert "bootstrap_values" not in arrays
    assert arrays["trial_indices"].tolist() == [0, 1, 2]
    assert arrays["frequency_hz"].tolist() == [8.0, 40.0]
    assert arrays["epoch_names"].tolist() == ["whole", "before", "after"]
    assert arrays["band_names"].tolist() == ["theta", "gamma"]
    assert arrays["site_valid"].tolist() == [[True, True, True], [True, True, False]]
    assert arrays["pair_valid"].tolist() == [[True, True, False]]
    assert arrays["plv_by_frequency"].shape == (3, 1, 3, 2)
    assert arrays["source_trace"].shape == (2, 3, 2000)
    assert arrays["band_filtered_trace"].shape == (2, 3, 2, 2000)
    assert arrays["hilbert_phase_rad"].shape == (2, 3, 2, 2000)
    np.testing.assert_array_equal(
        arrays["condition_membership"], prepared.prepared_trials.condition_membership
    )
    np.testing.assert_array_equal(
        arrays["filter_membership"], prepared.prepared_trials.filter_membership
    )
    correct_index = arrays["condition_names"].tolist().index("correct_rewarded")
    assert np.nanmean(arrays["itpc"][correct_index, 0, 0]) > 0.99
    offset = arrays["ispc_phase_offset_rad"][correct_index, 0, 0]
    assert np.angle(np.nanmean(np.exp(1j * offset))) == pytest.approx(
        _SITE_OFFSET_RAD,
        abs=1e-5,
    )
    assert arrays["plv_computable"][0, 0].all()
    assert not arrays["plv_computable"][2, 0].any()
    for metric_prefix in ("itpc", "ispc"):
        quantile_shape = arrays[f"{metric_prefix}_band_mean"].shape
        for name in ("bootstrap_q25", "bootstrap_median", "bootstrap_q75"):
            values = arrays[f"{metric_prefix}_{name}"]
            assert values.shape == quantile_shape
            assert values.dtype.kind == "f"
        counts = arrays[f"{metric_prefix}_band_trial_count"]
        assert counts.shape == quantile_shape
        assert counts.dtype.kind in {"i", "u"}
        assert np.array_equal(counts < 10, arrays[f"{metric_prefix}_unstable"])


def test_build_synchrony_payload_maps_per_epoch_band_bootstrap_outputs_exactly(
    tmp_path: Path,
) -> None:
    """Payload counts and interior summaries retain each numerical phase selection.

    The fixture changes only finite phase samples, so it distinguishes the
    selected trial count over epoch/band cells without reverse-engineering it
    from the broader site or pair masks.
    """
    config, prepared = _prepared_phase_with_distinct_band_validity(tmp_path)
    payload = build_synchrony_payload(config, prepared)
    arrays = payload.arrays
    condition_index = prepared.prepared_trials.condition_names.index("correct_rewarded")
    condition_mask = prepared.prepared_trials.condition_membership[:, condition_index]
    epoch_windows = {
        "whole": (-2.0, 2.0),
        "before": (-2.0, 0.0),
        "after": (0.0, 2.0),
    }

    expected_itpc = bootstrap_phase_clustering_bands(
        phase_vectors=np.moveaxis(prepared.phase_tensor[0], 1, 0),
        valid_mask=np.moveaxis(prepared.phase_valid[0], 1, 0),
        trial_mask=condition_mask & prepared.site_valid[0],
        frequencies_hz=np.asarray(config.phase.frequency_hz, dtype=float),
        relative_time_s=prepared.relative_time_s,
        epoch_windows=epoch_windows,
        bands=config.phase.bands,
        bootstrap_count=config.phase.bootstrap_count,
        seed=config.phase.seed + condition_index,
    )
    relative_phase = prepared.phase_tensor[0] * np.conjugate(prepared.phase_tensor[1])
    expected_ispc = bootstrap_phase_clustering_bands(
        phase_vectors=np.moveaxis(relative_phase, 1, 0),
        valid_mask=np.moveaxis(
            prepared.phase_valid[0] & prepared.phase_valid[1],
            1,
            0,
        ),
        trial_mask=condition_mask & prepared.pair_valid[0],
        frequencies_hz=np.asarray(config.phase.frequency_hz, dtype=float),
        relative_time_s=prepared.relative_time_s,
        epoch_windows=epoch_windows,
        bands=config.phase.bands,
        bootstrap_count=config.phase.bootstrap_count,
        seed=config.phase.seed + condition_index,
    )

    for metric_prefix, entity_index, expected in (
        ("itpc", 0, expected_itpc),
        ("ispc", 0, expected_ispc),
    ):
        np.testing.assert_array_equal(
            arrays[f"{metric_prefix}_band_trial_count"][
                condition_index, entity_index
            ],
            expected.selected_trial_count,
        )
        assert np.unique(expected.selected_trial_count).size > 1
        for saved_name, summary_name in (
            ("bootstrap_q25", "q25"),
            ("bootstrap_median", "median"),
            ("bootstrap_q75", "q75"),
        ):
            expected_values = getattr(expected, summary_name)
            assert np.unique(expected_values[np.isfinite(expected_values)]).size > 1
            np.testing.assert_array_equal(
                arrays[f"{metric_prefix}_{saved_name}"][
                    condition_index, entity_index
                ],
                expected_values,
            )


def test_synchrony_factory_commits_only_synchrony_and_reloads_compatible_cache(
    tmp_path: Path,
) -> None:
    """The production factory must not prepare Power or spike-phase components."""
    config = _config(tmp_path / "cache")
    dependencies = make_synchrony_pipeline_dependencies(
        trial_table_loader=lambda _: _trial_table(),
        site_phase_tensor_builder=_phase_tensor_builder([]),
        block_loader_factory=_block_loader_factory([]),
        spikeglx_loader=_trace_loader,
    )

    result = compute_synchrony_component(config, dependencies)
    manifest = load_or_initialize_manifest(config.output_directory, config)
    arrays = load_component_arrays(
        config.output_directory / "synchrony.npz",
        manifest,
        "synchrony",
    )
    status = assess_component_status(
        config.output_directory,
        "synchrony",
        config,
        manifest,
    )

    assert result.state == "complete"
    assert result.stage == "complete_synchrony"
    assert status.status == "compatible"
    assert arrays["itpc"].shape == (9, 2, 2, 2000)
    for metric_prefix in ("itpc", "ispc"):
        for summary_name in ("bootstrap_q25", "bootstrap_median", "bootstrap_q75"):
            summary = arrays[f"{metric_prefix}_{summary_name}"]
            assert summary.shape == arrays[f"{metric_prefix}_band_mean"].shape
            assert summary.dtype.kind == "f"
        assert arrays[f"{metric_prefix}_band_trial_count"].dtype.kind in {"i", "u"}
    assert not (config.output_directory / "power.npz").exists()
    assert not (config.output_directory / "spike_phase.npz").exists()
    with pytest.raises((NotImplementedError, RuntimeError), match="power|unsupported"):
        dependencies.prepare_power(config)
    with pytest.raises((NotImplementedError, RuntimeError), match="spike|unsupported"):
        dependencies.prepare_spike(config, object())
