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

## Restricted presentation comparisons

For each configured size, the workflow also creates two versioned presentation
views. Both keep Ideal as a common reference and retain existing model colors:

| Profile | Models in legend order |
| --- | --- |
| `main_comparison_v1` | QL, HMM, Home E + Doubt, Ideal |
| `strategy_comparison_v1` | Perseveration, WSLS, Probe + persistence, Ideal |

The four additional default-size filenames are:

- `<session>_choice_window_10_main_comparison_v1_agreement.png`
- `<session>_choice_window_10_strategy_comparison_v1_agreement.png`
- `<session>_choice_window_20_main_comparison_v1_agreement.png`
- `<session>_choice_window_20_strategy_comparison_v1_agreement.png`

These views use a four-entry legend below the data axis, readable model names,
and a group title. Curves, markers, limits, window definitions and context
boundaries are unchanged. Models missing from older saved results are listed
as unavailable in the caption; available members are plotted without any
substitution or model recomputation. When none are available, the axis states
that explicitly. All models remain in the numerical artifact, and existing
combined and all-model standalone PNGs are retained.

`performance_plots.choice_window_presentation_groups()` returns the two ordered
profile-to-model-key mappings. For direct export, pass its model tuple as
`agent_selection` and the profile name as `presentation_profile` to the
standalone exporter. Both keyword arguments must be supplied together.
The profile identifies the selection in the filename; change its version
when membership/order changes. Omitting both arguments preserves the original
all-model API, filename and layout. Existing output files are reused.

## Loading saved results directly

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

The restricted CT016 comparisons were also checked against the saved curve
values, with all earlier PNG/NPZ hashes preserved. The expanded suite passed
137 tests, including legend separation from large axis labels. All four group
exports were visually inspected. Their validation record is
`presentation_comparisons_v1_validation.json` beside the PNGs.
