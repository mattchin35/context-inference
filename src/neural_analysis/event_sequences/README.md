# Event-sequence heatmaps

## Recorded behavior examples

Choose **Behavior examples** in the metadata webapp to browse recorded choice
patterns without preparing neural activity. Launch the existing app with:

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- --session-metadata /path/to/session/neural_session.json
```

The view offers Probe, Omission probe, Ideal transition, Lucky omission-switch,
and Indecisive transition. It defaults to a left-starting Probe with one
incorrect opposite-side choice and five original rows before/after the episode.
Change the initial side or repeat count, use **Previous**/**Next** or **Example**
to select a chronological match, and adjust **Trials before/after** as needed.
Empty conditions report that no episodes match. Counts can overlap between
families, including an indecisive transition's shorter ideal prefix/probe tail.

Probe and omission-probe returns accept a correct rewarded trial or a correct
omission (known reward 1 or 0). Their initial trial must be correct/rewarded,
and all participating trials must stay in one block/active context. Unknown
return rewards are excluded. Existing neural views retain rewarded-only returns.
The remaining families keep their existing definitions and repeat limits.
All participating rows require ordinary choices, consecutive original trial IDs,
known zero experimenter-reward flags and finite UTC choice timestamps.

Choice figures use original trial labels with equal spacing for trial order,
not elapsed time. Filled/open circles encode reward/no reward; red crosses
encode incorrect choices independently. Gray diamonds mark unknown rewards;
stars mark manual rewards in surrounding rows. Context shading and a gold
episode band are explained in the legend. Invalid surrounding choices remain
as gaps, and captions identify context truncated by session boundaries.

**Export PNG** saves a new light figure at 200 dpi and a same-stem JSON file
with exact trial IDs, result identity, plotting-code digest, and context counts.
Both preview and export use the same selected rows and annotation logic.

The first visit prepares or reuses a small behavioral result; navigation never
loads neural sources or repeats matching. **Recompute behavioral matches**
creates a fresh result. Changed source content, analysis code/version or
scientific parameters invalidate reuse. No metadata-schema change is needed.

Computation entry point:

```bash
uv run python -m src.neural_analysis.event_sequences.run_examples --session-metadata /path/to/session/neural_session.json --dry-run
```

Remove `--dry-run` to prepare. Use `--rerun` for a new result and
`--probe-return-outcome rewarded` for the original stricter probe definition.
The callable batch entry point is
`src.neural_analysis.event_sequences.run_examples_batch.run_batch`, accepting
an explicit ordered list of metadata paths and optional `workers`; default
parallelism is CPU count capped by session count. Dry runs start no workers.

Named-array results, including full source rows, matches and `meta`, are saved
under `<trial CSV directory>/behavior_examples/<version>/<configuration>/<timestamp>/`.
Run logs, source-script snapshots, summaries and exported figures are stored
in `<session>/analysis_runs/behavior_examples_<date>_<time>_<configuration>/`.
Experimental outputs must be outside the source repository. Existing results
and figures are never overwritten. NPZ loading uses pickle for local metadata
and original labels, so only load trusted result archives.

The core computation is `examples.py`, persistence is `example_io.py`, and
rendering is `example_plotting.py`. The webapp only orchestrates these functions.

## Neural activity views

Open either **Event sequences: central trials** or **Event sequences:
multi-trial sequences** in the metadata webapp. Both tools share the same saved
neural source and plotting controls.

## Sparse components

Choose **Regional sparse components** under **Neural source**. Select an existing
saved configuration, or enable **Configure a new source**, choose region, probe,
bin width, population, and **SparsePCA alpha**, then click **Prepare**.
Preparation is explicit and may take minutes when a new sparse fit is needed.
Compatible session-timeline sparse results are reused first. **Recompute** makes
a new event run; it may reuse the same upstream fit.

The fit uses all eligible completed ordinary choices and their complete
[-2,+2) second windows. The population uses explicit Phy good/mua labels by
default (or good only when selected), good inside-brain channels and metadata
channel selection. Missing manual labels are excluded. The fitting and ridge
transform implementation is shared with the session timeline, with seed 0,
alpha 1 by default, and K=min(30, training observations, retained units).

Select **Sparse component** to inspect SC1, SC2, etc. These are component
indices, not explained-variance ranks. Raw values are ridge coefficients in
standardized-unit activity space. Sparse components are neither orthogonal nor
additive variance contributions. Iteration-limit or fitting warnings appear
above the plot and in PNG captions; reaching the iteration limit does not
confirm convergence.

**Z-score** uses the selected component's mean and population SD over all unique
eligible session bins, before motif selection. A constant component can be
displayed raw but cannot be z-scored. Source switching, motif selection and
z-scoring load saved data without fitting.

## Color limits

Automatic colors use the full eligible source pool. Enable **Manual color limit**
and enter a positive finite **Color limit L** to use [-L,+L] for components and
all z-scored sources. A raw single unit uses [0,L] Hz. Every panel and trial
segment uses the same range, including **Export PNG**.

Only colors saturate outside the chosen bounds. Activity values, hover values,
statistics, counts, saved products and fits are unchanged. The caption states
the range, units and automatic/manual mode.

Color preferences are remembered separately for each session, source
configuration, component or unit, and raw/z-score mode. Both event tools and
all motif families share those preferences. They last for the current webapp
session and are not written to disk.

## Python preparation

```python
from src.neural_analysis.event_sequences.run_session import (
    EventSequenceRequest, prepare_event_sequences,
)

request = EventSequenceRequest(
    "/data/session/neural_session.json", "sparse_pc",
    region="PFC", bin_width_s=0.5, sparse_alpha=1.0,
    population_quality="good_mua",
)
plan = prepare_event_sequences(request, dry_run=True)
report = prepare_event_sequences(request)
```

Use `uv run` for local Python commands. The existing `run_batch.py` accepts
these same requests and parallelizes across sessions. No new dependency or CLI
is required. Existing unit/ordinary-PC requests and saved products are unchanged.

Outputs are immutable timestamped directories under the original session's
`analysis_runs/`, containing results.npz, source script snapshots, summary and
log. Sparse event products use `event-sequence-sparse-heatmaps-v1` (schema 1).
The existing `meta.pca_reference` field binds the exact upstream sparse archive
by relative path, size, SHA256, configuration, region and probe. Scores are not
duplicated in the event archive. Saved products can load without current raw
inputs, but still require that exact upstream archive.

## Validation

Tests cover ordinary-source regressions, sparse reuse/preparation, strict input
selection, exact reference/offline loading, zero components, unclipped manual
colors in both renderers, and real Streamlit preference persistence.

CT026, August 4 2026: the 500-ms PFC source reused the existing 716 x 8 x 30
timeline scores (174 retained units), with no fitting. Event preparation took
0.36 s, loading plus validation/hash 0.14 s, and the event archive is 62,863
bytes. Raw and z-scored central/transition PNGs and measured validation JSON
are beside the new run
`event_sequence_heatmaps_20261115_011622_250534_a06110d436df1607`.
The referenced fit reached 1000 iterations; convergence is unconfirmed.
