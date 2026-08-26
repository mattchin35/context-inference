"""RED contracts for the production Spike-phase runtime bridge."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis.lfp_summary_io import load_component_arrays
from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    PPCAnalysisConfig,
    UnitPopulationConfig,
    default_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_payloads import validate_component_payload
from src.neural_analysis.lfp_summary_pipeline import compute_spike_phase_component
from src.neural_analysis.lfp_summary_preparation import build_prepared_trials
from src.neural_analysis.lfp_summary_runtime import (
    PreparedPhaseRun,
    PreparedSpikeRun,
    build_spike_phase_payload,
    make_spike_phase_pipeline_dependencies,
    prepare_spike_run,
)


def _config(output_directory: Path):
    """Return a two-trial, one-site, one-unit 100-shuffle configuration.

    Parameters
    ----------
    output_directory : pathlib.Path
        Temporary cache directory; units are filesystem coordinates.

    Returns
    -------
    LFPSummaryConfig
        Immutable 500-Hz, 2--100-Hz Spike-phase preview configuration.
    """
    site = LFPSiteConfig(
        "PFC", "PFC", "spikeglx", Path("pfc.bin"), None, "ProbeA", 5, "uV", 500.0
    )
    population = UnitPopulationConfig(
        "PFC-active",
        "ProbeA",
        Path("sorter"),
        Path("aligned.npz"),
        (2,),
        (("quality_labels", "good"),),
        ("ProbeA:7",),
    )
    return replace(
        default_lfp_summary_config(),
        session_id="synthetic-spike-runtime",
        output_directory=output_directory,
        sites=(site,),
        site_pairs=(),
        unit_population=population,
        ppc=PPCAnalysisConfig(shuffle_count=100, seed=72),
    )


def _trial_table() -> pd.DataFrame:
    """Return two condition-compatible trials with absolute event times.

    Returns
    -------
    pandas.DataFrame
        Two rows; time columns are absolute seconds and behavior columns follow
        the project's integer conventions.
    """
    return pd.DataFrame(
        {
            "start_time": [9.0, 19.0],
            "choice_time": [10.0, 20.0],
            "experimenter_reward_given": [0, 0],
            "correct": [1, 1],
            "reward": [1, 1],
            "action": [0, 0],
            "state_int": [0, 0],
        }
    )


def _prepared_phase(config) -> PreparedPhaseRun:
    """Return exact-grid unit phase with complete numerical validity.

    Parameters
    ----------
    config : LFPSummaryConfig
        Supplies sites, frequencies in Hz, and the half-open whole window.

    Returns
    -------
    PreparedPhaseRun
        Complex phase axes ``(site, frequency, trial, time)`` and cached source
        traces ``(site, trial, time)`` on the exact 500-Hz grid.
    """
    time_s = np.arange(-2.0, 2.0, 1.0 / 500.0)
    frequency_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    event_times_s = np.array([10.0, 20.0])
    phase = np.exp(
        1j
        * 2.0
        * np.pi
        * frequency_hz[None, :, None, None]
        * time_s[None, None, None, :]
    )
    phase = np.repeat(phase, 2, axis=2).astype(np.complex64)
    prepared_trials = build_prepared_trials(
        _trial_table(),
        config.trial_filter,
        config.analysis_windows.alignment_event,
        {"PFC": np.ones(2, dtype=bool)},
    )
    source = np.sin(2.0 * np.pi * 8.0 * time_s)[None, None]
    return PreparedPhaseRun(
        trial_indices=np.arange(2, dtype=np.int64),
        alignment_times_s=event_times_s,
        prepared_trials=prepared_trials,
        phase_tensor=phase,
        phase_valid=np.ones_like(phase, dtype=bool),
        relative_time_s=time_s,
        site_valid=np.ones((1, 2), dtype=bool),
        pair_valid=np.empty((0, 2), dtype=bool),
        source_trace=np.repeat(source, 2, axis=1),
    )


def _absolute_unit_spikes() -> dict[str, np.ndarray]:
    """Return 52 event-locked spikes per trial for one stable unit.

    Returns
    -------
    dict[str, numpy.ndarray]
        Stable unit id mapped to finite absolute seconds ``(104,)``.
    """
    relative = np.linspace(-1.9, 1.9, 52)
    return {"ProbeA:7": np.concatenate((10.0 + relative, 20.0 + relative))}


def test_prepare_spike_run_preserves_stable_units_and_trial_local_ownership(
    tmp_path: Path,
) -> None:
    """Preparation must retain configured identity and one spike train per trial."""
    config = _config(tmp_path / "cache")
    phase = _prepared_phase(config)

    prepared = prepare_spike_run(config, phase, lambda _: _absolute_unit_spikes())

    assert isinstance(prepared, PreparedSpikeRun)
    assert prepared.unit_ids == ("ProbeA:7",)
    assert len(prepared.trial_spike_trains[0].relative_spike_times) == 2
    spike_counts = [
        values.size
        for values in prepared.trial_spike_trains[0].relative_spike_times
    ]
    assert spike_counts == [52, 52]
    assert prepared.trial_spike_trains[0].overlap_trial_indices.size == 0


def test_build_spike_payload_fills_schema_axes_counts_and_preview_null(
    tmp_path: Path,
) -> None:
    """Payload must contain full cache axes and exactly 100 preview permutations."""
    config = _config(tmp_path / "cache")
    phase = _prepared_phase(config)
    prepared = prepare_spike_run(config, phase, lambda _: _absolute_unit_spikes())

    payload = build_spike_phase_payload(config, phase, prepared)

    validate_component_payload("spike_phase", payload)
    arrays = payload.arrays
    assert arrays["unit_ids"].tolist() == ["ProbeA:7"]
    assert arrays["trial_indices"].tolist() == [0, 1]
    assert arrays["ppc"].shape == (1, 9, 1, 3, 50)
    assert arrays["relative_spike_time_offsets"].shape == (1, 3)
    assert arrays["permutation_count"].max() == 100
    assert arrays["source_trace"].shape == (1, 2, 2000)


def test_spike_dependencies_commit_only_spike_component(tmp_path: Path) -> None:
    """The production factory must atomically write and reload Spike phase only."""
    config = _config(tmp_path / "cache")
    phase = _prepared_phase(config)
    dependencies = make_spike_phase_pipeline_dependencies(
        trial_table_loader=lambda _: _trial_table(),
        unit_spike_loader=lambda _: _absolute_unit_spikes(),
        phase_preparer=lambda _: phase,
    )

    result = compute_spike_phase_component(config, dependencies)
    manifest = result.manifest

    assert result.state == "complete"
    assert manifest is not None
    assert set(manifest["components"]) == {"spike_phase"}
    arrays = load_component_arrays(
        config.output_directory / "spike_phase.npz",
        manifest,
        "spike_phase",
    )
    assert arrays["unit_ids"].tolist() == ["ProbeA:7"]
