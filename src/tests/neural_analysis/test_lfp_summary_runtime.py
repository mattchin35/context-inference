"""RED contracts for the production-callable Power summary runtime bridge."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.lfp_summary_io import (
    assess_component_status,
    load_component_arrays,
    load_or_initialize_manifest,
)
from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    PowerAnalysisConfig,
    default_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_pipeline import compute_power_component
from src.neural_analysis.lfp_summary_payloads import validate_component_payload
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
