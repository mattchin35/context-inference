# Cross-mouse Ideal agreement from choice windows

This is a separate saved-results-only workflow. It does not run session analysis,
fit models, replay policies, or recalculate choice windows. Existing cross-mouse
commands retain their current behavior.

Each mouse/session contributes its saved `observer_mouse_agreement` median for
the requested window size. The black group curve is the arithmetic mean of
available mouse medians at each session position, with equal weight per mouse.
It is not pooled trial agreement or a median over all mice's windows.

## Inputs and strict coverage

Select collections automatically with a mouse list, or specify one explicit
collection directory per mouse. Each selected directory must contain:

- `metadata.json` from the existing cross-session choice-window collector.
- `<mouse>_choice_window_summary.csv`.
- Accessible source NPZ paths named by the collection metadata.

Each mouse also requires
`<data_root>/<mouse>/cross_session_analysis/<mouse>_overall_performance.csv`,
with unique full `session_id` and valid `date` columns. This file defines the
expected session roster. The collection must match that roster exactly; selecting
an incomplete older run does not silently shorten the curve. A legacy date-only
roster must be regenerated with session identities before use.

Collections must have matching numerical analysis versions and source digests.
No-choice/manual-reward exclusions, greedy tie initialization, incomplete-tail
handling and missing-prediction rules are checked against the saved window
definition. Session-specific feature parameters may differ and are recorded.
Both requested sizes must exist in every source when `(10, 20)` is requested.

Explicitly selected collections with missing files, sources, expected sessions,
Ideal results or requested sizes cause an error before output creation.
Automatic selection can skip such candidates, recording their paths and failure
reasons. If no valid collection remains, it reports all tried candidates and
stops. Errors identify the mouse and, where applicable, session, size and source
path. Multiple mouse errors are reported together.

Automatic selection scans timestamped directories beneath each mouse's
`cross_session_analysis/choice_window_agreement/`, newest first. Directory names
use `YYYY-MM-DD_HH-MM-SS_microseconds`; file modification times and unrelated
directories do not determine selection. Each candidate must pass the same full
roster/source/summary checks as explicit mode, using the requested window sizes.
Successfully validated data are reused rather than loaded again.

After selecting the newest valid collection for each mouse, cohort version and
source-digest compatibility are checked. A conflict stops the export and names
the selected paths/versions; discovery does not silently downgrade mice to older
versions. Explicit mode never falls back to automatic selection.

## Training positions and unavailable values

Within each mouse, sessions sort by date then full session ID, retaining separate
same-date sessions. Positions are assigned before checking whether the metric
is usable. "Training Day" means this chronological session number, matching the
existing cross-mouse convention; it is not elapsed calendar time.

A valid source with fewer choices than a full window is retained with NaN and
status `no_complete_windows`. A session with complete windows but no scored Ideal
predictions has status `no_usable_predictions`. Neither becomes zero or shifts
later sessions. Individual mouse lines break at NaNs; small point markers keep
isolated finite session values visible. Existing cross-mouse plots retain their
line-only default. The group mean uses only
finite contributing mice; it also breaks when no mouse contributes. If a size
has no usable medians anywhere, the command stops rather than writing an empty
cohort curve.

## Running after session results are ready

From the repository root, use `uv run python` with the following code for quick
automatic selection:

```python
from pathlib import Path
from src.main import run_cross_mouse_choice_window_agreement_curves

data_root = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData")
values, run_directory = run_cross_mouse_choice_window_agreement_curves(
    data_root=data_root,
    output_path=data_root / "cross_mouse_analysis",
    mice=["CT016", "CT017", "CT026"],
    window_sizes=(10, 20),  # Use (10,) or (20,) for just one curve.
)
print(run_directory)
```

For explicit selection, supply `collection_runs` instead of `mice`:

```python
collection_runs = {
    "CT016": data_root / "CT016/cross_session_analysis/choice_window_agreement/SELECTED_RUN",
    "CT017": data_root / "CT017/cross_session_analysis/choice_window_agreement/SELECTED_RUN",
}
values, run_directory = run_cross_mouse_choice_window_agreement_curves(
    data_root=data_root,
    output_path=data_root / "cross_mouse_analysis",
    collection_runs=collection_runs,
    window_sizes=(10, 20),
)
```

Provide exactly one nonempty mode: `collection_runs` or `mice`. Duplicate mouse
IDs are rejected. Existing positional explicit-directory calls remain valid.
Both modes print the selected paths; automatic mode also prints skipped
candidate reasons. Neither discovers an implicit mouse cohort.
The loader supports both modes read-only via
`cross_mouse_choice_windows.load_cross_mouse_choice_window_data`.

## Outputs and validation

Each successful call creates
`<output_path>/choice_window_ideal_agreement/<date_time>/` containing:

- One light-background PNG per size, with mouse curves and the black mean curve.
- `session_values.csv`: saved medians, training positions, counts and statuses.
- `group_summary.csv`: `agreement_mean` and `n_mice` at each size/position.
- `metadata.json` and `run.log`: selection mode, selected paths, skipped
  candidates/reasons, source hashes, versions,
  session-specific parameters, coverage warnings and runtime.
- `summary.md` and snapshots of the main/batch entry point, loader and renderer.

Agreement values are unitless fractions. Window and comparison counts retain
their saved units. No confidence intervals or statistical tests are implied.
Input CSVs/NPZs and prior output directories are never modified.

Implementation validation uses only synthetic saved collections. No experimental
sessions or experimental figure exports are run during code preparation.
