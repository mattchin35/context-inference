from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis import spike_lfp_hilbert_phase, unit_spike_plotting


def _result() -> spike_lfp_hilbert_phase.SingleTrialSpikeLFPHilbertResult:
    """Build a compact valid single-trial result for figure contracts."""

    time_s = np.linspace(-1.0, 1.0, 21)
    phase_rad = np.angle(np.exp(1j * 2.0 * np.pi * time_s))
    analytic = np.exp(1j * phase_rad)
    return spike_lfp_hilbert_phase.SingleTrialSpikeLFPHilbertResult(
        relative_time_s=time_s,
        raw_lfp=np.sin(2.0 * np.pi * time_s),
        bandpassed_lfp=np.sin(2.0 * np.pi * time_s),
        analytic_signal=analytic,
        envelope=np.ones(time_s.size),
        phase_unit=analytic,
        phase_rad=phase_rad,
        phase_valid=np.ones(time_s.size, dtype=bool),
        spike_times_absolute_s=np.array([9.5, 10.25]),
        spike_times_relative_s=np.array([-0.5, 0.25]),
        spike_phase_unit=np.exp(1j * np.array([-np.pi / 2.0, np.pi / 2.0])),
        spike_phase_rad=np.array([-np.pi / 2.0, np.pi / 2.0]),
        spike_phase_envelope=np.array([1.0, 1.0]),
        spike_phase_valid=np.array([True, True]),
        source_sample_rate_hz=10.0,
        frequency_band_hz=(6.0, 10.0),
        filter_padding_s=1.0,
        trial_index=3,
        event_time_s=10.0,
        unit_id=12,
        lfp_site_label="PFC channel 4",
    )


def _trial_df() -> pd.DataFrame:
    """Build one behavior row with an explicit reward event."""

    return pd.DataFrame(
        {
            "choice_time": [10.0],
            "start_time": [9.0],
            "led_on_time": [9.4],
            "reward_time": [10.4],
            "action": [1],
        },
        index=[3],
    )


def test_break_wrapped_phase_trace_inserts_nan_at_wrap_discontinuities():
    """Wrapped phase display must not join +pi to -pi with a false vertical line."""

    broken = unit_spike_plotting.break_wrapped_phase_trace(np.array([2.9, 3.1, -3.1, -2.9]))

    assert np.isnan(broken[2])
    assert broken[0] == 2.9


def test_single_trial_hilbert_plot_has_four_named_axes_spikes_and_behavior_events():
    """The viewer should combine raw LFP, theta phase, spike observations, and reward behavior."""

    figure, axes = unit_spike_plotting.plot_trial_spike_lfp_hilbert_phase_and_behavior(
        trial_df=_trial_df(),
        trial_index=3,
        lick_times={"left_entry": nap.Ts(t=np.array([9.8])), "right_entry": nap.Ts(t=np.array([10.1]))},
        result=_result(),
        alignment_event="choice_time",
        window=(-1.0, 1.0),
    )

    assert set(axes) == {"raw_lfp", "bandpassed_lfp", "phase", "behavior"}
    assert axes["phase"].get_ylim() == (-np.pi, np.pi)
    assert axes["raw_lfp"].get_shared_x_axes().joined(axes["raw_lfp"], axes["behavior"])
    phase_offsets = np.vstack([collection.get_offsets() for collection in axes["phase"].collections])
    assert any(np.allclose(offset, [-0.5, -np.pi / 2.0]) for offset in phase_offsets)
    assert any(np.allclose(offset, [0.25, np.pi / 2.0]) for offset in phase_offsets)
    assert "Reward" in {collection.get_label() for collection in axes["behavior"].collections}
    plt.close(figure)
