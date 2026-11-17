# Communication subspace inspection

This single-session analysis compares held-out prediction of PFC and HPC spike
counts in all four directions: PFC to PFC, PFC to HPC, HPC to HPC, and HPC to
PFC. It is exploratory. A low-rank curve describes predictive structure in
these recordings; it does not establish causal communication, anatomical
directionality, or a task-independent communication subspace.

## Webapp

Launch the metadata-driven app for one session:

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata /path/to/session/neural_session.json
```

Choose **Communication subspace**. Version-3 metadata roles prefill PFC/HPC
probes. With version-2 metadata that has no role assignments, choose two
distinct probes explicitly; the app does not guess region identities. Select
100 or 500 ms bins and a random seed. The fixed choice window is `[-2, +2)` s.
The readiness panel reports eligible good-unit counts, matched population size,
common covered choices, coverage, and any unavailable reason.

Click **Run** to reuse a compatible successful result or compute a new result.
Click **Recompute** to create a fresh run without replacing previous output.
Controls and display changes do not load spike arrays or fit models. The fit
starts only after an explicit action, and progress is reported over five draws
and five outer folds. A selected saved result remains viewable if current raw
sources have been removed. Its caption and plot retain the saved bin width,
seed, and probe roles while current preparation controls describe only the next
run. Saved diagnostics list matched source/target unit IDs, split and purge
counts, selected alpha values, effective variable-unit counts, and supported
maximum ranks.

## Inputs and design

The session metadata must identify a trial table and two distinct probes with
synchronized spike assignments, unit labels, channel-quality metadata, and
trusted IRIG recording coverage. The analysis uses only manually labeled Phy
`group=good` units on eligible inside-brain channels under the existing
metadata/channel-selection rules. It does not fall back to `mua`, `KSLabel`, or
unlabeled units. At least four eligible good units are required in each region:

```text
n = floor(min(PFC good units, HPC good units) / 2) >= 2
```

Each of five seeded draws selects disjoint source and target groups of `n`
units within each region, without replacement within a draw. Draws can share
units, so they are not independent biological replicates. All completed
ordinary choices are pooled across contexts, actions, and outcomes. Only
windows with complete shared PFC/HPC coverage are retained. Counts are raw,
unsmoothed spikes per bin; there is no condition-mean subtraction or lagged
predictor. Training-only per-neuron standardization is used for fitting.

Five chronological outer folds hold out whole choices. Overlapping windows are
purged from training, and an inner chronological block tunes ridge
regularization. The view plots held-out mean per-neuron R2 by integer coefficient
rank for ridge-regularized reduced-rank regression (RRR), with unrestricted
ridge references. The rank applies to the slope matrix. R2 is absolute and can
be negative; negative R2 means prediction was worse than the held-out mean
reference. No minimum rank is selected automatically, and no equivalence or
significance test is performed.

Ranks that are unsupported by a fold's effective source/target dimensions or
training design are marked unavailable rather than relabeled as a smaller
rank. Incomplete curves and their draw support are surfaced in the saved score
tables and diagnostic display. The maximal supported RRR rank is expected to
agree with unrestricted ridge when feasible dimensions and tuning objectives
coincide; support differences remain visible.

## Command line and outputs

Preflight one session without loading spike arrays or writing analysis outputs:

```bash
uv run python -m src.neural_analysis.communication_subspace.run_session \
  --session-metadata /path/to/session/neural_session.json \
  --pfc-probe ProbeA --hpc-probe ProbeB --bin-width-ms 100 --seed 0 --dry-run
```

If metadata already provides PFC/HPC roles, the two probe flags may be omitted.
If either is supplied, both are required. Omit `--dry-run` to prepare/reuse a
result; add `--rerun` to force an additive new run. The run rejects output roots
inside the source repository.

Results are stored beneath the session root:

```text
communication_subspace/
  results/<run-id>.npz
  runs/<run-id>/
    config.json
    run_session.py
    run.log
    summary.md
    rank_curves.png
```

The NPZ is the source-independent saved analysis used by the webapp. The run
directory contains the script snapshot, configuration, log, short scientific
summary, and light PNG. A normal Run reuses a compatible successful archive;
Recompute or `--rerun` keeps earlier archives and writes a new run.

Synthetic acceptance tests cover both bin widths and replay every scored target
from saved model factors. No experimental session has been validated yet. The
next gate is one user-designated session and user inspection of its output.
