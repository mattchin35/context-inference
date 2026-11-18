# Win-switch versus Home E+D diagnostic

The additional light-mode figure compares Win-switch/lose-stay, Home E+D,
and the mouse-history Ideal observer. Existing candidate comparisons are kept.
The normal session workflow adds this export automatically through
`plot_expectant_switching_for_session`.

To plot an already processed session, edit the mouse, date, and data root at the
top of `main_mouse_win_switch_home_diagnostic` in `src/main.py`, then call it:

```bash
uv run python -c 'from src.main import main_mouse_win_switch_home_diagnostic; main_mouse_win_switch_home_diagnostic()'
```

Alternatively, select its commented call in the bottom `if __name__` block and
comment out the currently active workflow before running `uv run python -m src.main`.
The default example is CT016, 2026-05-29. Missing or ambiguous saved sessions
raise an error. Raw behavior processing, HMM fitting and neural data are not needed.

The four panels show:

1. Observed/rewarded choices, optional true context, and three pre-trial raw scores.
2. Mouse-agent agreement over each context block, using the shared greedy rule.
3. Home expectancy and doubt signed components.
4. Home reward count and expectancy strength (separate y axes).

Raw score scales differ: Win-switch is discrete while Home/Ideal scores are
graded. They are not calibrated confidence. Left-positive scores predict left;
zero scores use the existing initial-right/repeat-previous-greedy-choice rule.
Agreement excludes no-choice/manual-reward rows and retains model history across
blocks. Ideal and Home states must already be present in the saved augmented CSV;
the exporter fails explicitly if required columns are missing. Removed probe
model columns are not required. Win-switch negates the saved WSLS regressor;
when absent, the existing canonical WSLS replay is used on the full session.

The numerical preparation lives in `src/behavior_analysis/win_switch_home.py`;
plotting lives in `src/behavior_analysis/performance_plots.py`. The existing
candidate and new diagnostic share a small renderer, keeping the old public
interfaces and output filenames intact. No dependencies were added.

Prepared named arrays and metadata are saved to:

```text
<session>/processed/win_switch_home_diagnostic/<version>/<input-digest>/<timestamp>/results.npz
```

The figure, metadata, source snapshots, log and summary use the corresponding
directory below `<session>/figures/`. Metadata records input/source digests,
preparation method, available parameter provenance, axes and units. Parameters
are never silently replaced with current defaults. Identical inputs reuse both
NPZ and PNG; changed data/code/parameters or `rerun=True` on
`save_and_plot_win_switch_home_for_session` create a new dated run.
Original augmented CSVs and previous figures are preserved. NPZ object arrays
support legacy missing sentinels and must only be loaded from trusted project data.

This output is a context-block diagnostic, independent of the 10/20-choice
agreement exports; those remain unchanged.
