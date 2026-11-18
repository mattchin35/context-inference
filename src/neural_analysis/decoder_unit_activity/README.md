# Decoder-weight unit activity

In **Population task decoding**, select **Display -> Decoder-weight unit
activity**. Choose PFC/HPC, the per-group cap (default 10), and z-score/Hz.
The source must be an existing saved decoder run; no raw recording or new fit
is needed. Probability timelines remain the default subview.

In Z-score mode, **SD color range -> Custom** exposes **Symmetric limit (SD)**.
A limit of 3 maps colors from -3 to +3 SD across all groups; values beyond that
range saturate in color. The control changes only colors, preserving measured
values, normalization, unit selection, and cached activity. Limits stay fixed
while panning, and PNG exports use the chosen range. **Auto** restores the full
selected-pool limits. Hz mode keeps its automatic nonnegative range.

Units are grouped by left/right context, left/right choice, four side/outcome
classes, and within-side L->R/R->L switch-versus-stay weights. Five usable folds,
a positive mean and at least four positive fold weights are required. The
highest within-category percentile chooses one group per unit before capping.
Other weights and ties remain inspectable. V1 sources supply the first eight
groups; within-side switches require v2. Empty groups are never backfilled.

These are decoder-weight groups, not exclusive cell types or significance
tests. Correlated features share weights; outcome weights can reflect side.
Inspect the within-side reward/no-reward population scores. Support tables
distinguish unique trials from required/predicted bins.

Activity keeps native [-1,+1) choice windows, bin width, elapsed times and gaps.
500-ms sources provide four bins/window. Outcome units show measured pre-choice
activity although their source decoder uses only [0,+1). Coefficients do not
multiply firing rates. Z-scores use each unit's complete saved regional window
pool, unchanged by panning. Zero Hz remains valid; constants stay at their Hz
rate or become zero in z-score mode. Overlapping windows occupy transparent-
padded sublanes within each unit row, preserving one unit label/group.

Single-session commands:

```bash
uv run python -m src.neural_analysis.decoder_unit_activity.run_session dry-run \
  --source-run /data/session/analysis_runs/population_task_decoding_RUN --region PFC
uv run python -m src.neural_analysis.decoder_unit_activity.run_session new \
  --source-run /data/session/analysis_runs/population_task_decoding_RUN --region PFC \
  --max-units-per-group 10 --scale zscore
uv run python -m src.neural_analysis.decoder_unit_activity.run_session plot \
  --selection /data/session/processed/decoder_unit_activity/ID/selection.npz --scale hz
```

Optional `--time-range START_S END_S` uses elapsed session seconds; optional
`--event-tracks` selects existing trial/lick/LED tracks. Dry-run writes nothing.
For z-score plots, `--unit-sd-limit 3` sets symmetric -3 to +3 SD colors in
single-session, plot-from-selection, and batch commands. Omit it for automatic
limits. Nonpositive/nonfinite limits and custom SD limits with Hz are rejected.
Exports record requested and effective color limits in `settings.json`.
The new selection NPZ contains named arrays and JSON metadata (no pickle),
including source hashes, rules, versions, axes/units and normalization. Its
referenced preparation and decoder files must remain available; raw files need
not. Identical selections are reused. Each export creates a new dated directory
under `analysis_runs/decoder_unit_activity_*` with PNG, membership CSV, settings,
summary, log, timings/memory, and script/module snapshots. Old files are retained.

Batch requests are a JSON list, with relative paths resolved against that file:

```json
[{"source_run": "analysis_runs/population_task_decoding_RUN", "regions": ["PFC", "HPC"]}]
```

```bash
uv run python -m src.neural_analysis.decoder_unit_activity.run_batch dry-run --requests requests.json
uv run python -m src.neural_analysis.decoder_unit_activity.run_batch new --requests requests.json --workers 2
```

One request/session is permitted; regions run sequentially, sessions in spawned
workers. The default is capped by CPUs/session count; reduce workers for RAM/I/O.
Pending/unavailable/failed regions are reported without canceling other work.
Synthetic validation precedes one designated experimental inspection, followed
by user review before a wider session batch.
