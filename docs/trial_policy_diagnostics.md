# Trial-policy diagnostic boundary markers

Cross-session diagnostics omit dotted block-boundary lines by default. This
avoids the dense crosshatched texture produced by many block markers at the
full cross-session scale. Thin light-gray dashed lines still mark source-session
changes on every panel; their positions remain between concatenated retained
trial rows. No marker is added before the first source.

Single-session diagnostics retain their existing dotted block markers.
Probability/likelihood traces, mouse choices, winner cells (including ties),
and excluded-evidence shading retain their original values and appearance.
The figure caption describes which boundary markers are displayed.

To restore block markers for a detailed cross-session export:

```python
from src.behavior_analysis.trial_policy_plotting import (
    plot_multisession_trial_policy_diagnostic,
)

plot_multisession_trial_policy_diagnostic(
    results, metadata_records,
    output_path=new_png_path,
    show_block_boundaries=True,
)
```

`results` and `metadata_records` are ordered source-local records loaded with
`trial_policy_results.load_trial_policy_results` from trusted saved NPZs.
Arrays keep their original trial/model axes, numerical units and session resets.
Only display coordinates concatenate the source rows. Plotting does not fit,
score, or smooth models, so no numerical analysis version changes are needed.
Use a new output PNG path when re-exporting to preserve earlier figures.

The ordinary cross-session pipeline automatically uses the cleaner default.
It does not need a new run option or changes to existing callers. Dense traces
can still overlap at the full cross-session scale; removing boundary lines
does not downsample or otherwise alter the data.
