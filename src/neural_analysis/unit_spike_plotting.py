"""Compatibility facade for domain-owned neural plotting functions."""

from src.neural_analysis.lfp.plotting import (
    break_wrapped_phase_trace,
    plot_trial_lfp_phase_analysis_and_behavior,
    plot_trial_lfp_relative_phase_and_behavior,
    plot_trial_lfp_spectrogram_and_behavior,
)
from src.neural_analysis.population.plotting import (
    build_concatenated_trial_time_axis,
    plot_concatenated_trial_behavior_and_population_pca,
    plot_pca_cumulative_explained_variance,
    plot_trial_behavior_and_population_pca,
)
from src.neural_analysis.spike_behavior.plotting import (
    LEFT_CHOICE_ACTION,
    LEFT_LICK_EVENT,
    LICK_RASTER_STYLES,
    RIGHT_CHOICE_ACTION,
    RIGHT_LICK_EVENT,
    _draw_trial_event_markers,
    _draw_unit_spike_raster,
    _draw_unit_summary_panel,
    _make_full_bin_edges,
    _sanitize_filename_part,
    _validate_unit_summary_plot_type,
    compute_binned_firing_rates_hz,
    compute_population_psth_hz,
    compute_psth_hz,
    compute_trial_event_offsets,
    draw_single_trial_behavior_axis,
    extract_relative_events_for_trial,
    extract_relative_spikes_for_units,
    extract_relative_unit_spikes,
    filter_trials_for_unit_plot,
    get_trial_alignment_time,
    paginate_trial_indices,
    paginate_unit_ids,
    plot_trial_behavior_and_spike_raster,
    plot_unit_binned_rate_mean_sd,
    plot_unit_binned_rate_trial_traces,
    plot_unit_left_right_choice_comparison,
    plot_unit_raster_and_psth,
    save_unit_plot_figure,
    split_trial_indices_by_action,
)
from src.neural_analysis.spike_lfp.plotting import (
    plot_spike_lfp_phase_locking,
    plot_trial_spike_lfp_hilbert_phase_and_behavior,
)
