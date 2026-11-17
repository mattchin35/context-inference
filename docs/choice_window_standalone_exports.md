# Standalone choice-window agreement plots

Single-session and multisession plotting now export a separate agreement-only
PNG for each configured window size, alongside the existing diagnostic:

- `<session>_choice_window_agreement.png`: the existing combined figure.
- `<session>_choice_window_10_agreement.png`: 10-choice agreement only.
- `<session>_choice_window_20_agreement.png`: 20-choice agreement only.

All files use the existing session figures directory,
`choice_window_agreement/<numerical run timestamp>/`. Other configured window
sizes follow the same naming convention. Existing PNGs are reused, and missing
standalone panels are generated even when the combined figure already exists.

Standalone figures use the same agent curves, colors, markers, labels,
window-center trial coordinates, context-transition lines, and agreement limits
as the diagnostic's left panels. They use an opaque white background and
150-dpi PNG export, with a session title and descriptive caption.

The main plotting workflow loads saved numerical results once and creates the
combined and standalone figures. Its existing return values remain unchanged;
the returned combined PNG path still identifies the original diagnostic.

For a plotting-only refresh, load a trusted `results.npz` with
`src.behavior_analysis.choice_window_results.load_choice_window_results`, then
call `src.behavior_analysis.performance_plots.plot_session_choice_window_agreement_panel`
with `(window_table, metadata, plot_directory, session_id, window_size)` for
each saved window size. It returns the new or reused PNG Path. No model fitting,
agent scoring, or window aggregation occurs during plotting.

Input agreement values are unitless fractions. Trial bounds remain zero-based,
half-open original row positions; the plotted midpoint is
`(trial_start + trial_stop - 1) / 2`. Missing predictions retain the existing
per-agent denominator convention, and incomplete final windows stay discarded.

CT016, May 29, 2026 was validated using the saved November 16 numerical run.
Both standalone exports matched the saved curve data. Hash checks confirmed
that the numerical NPZ, original combined PNG, and repeated exports were
unchanged. The affected plotting, window-analysis, and workflow suite passed
124 tests. Validation metadata is saved beside the session PNGs as
`standalone_agreement_validation.json`.
