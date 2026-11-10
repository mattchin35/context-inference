# Cross-condition generalization

This separate analysis trains a binary readout on two condition cells and
evaluates it on the complementary cells. It prepares four targets (Context,
Choice, Actual reward, and Value (correctness-based)), four directed transfers
per target, three supervised windows, two training weightings, and PFC/HPC
separately. At one bin width this is up to 192 fitted models. Outcome choices
in the viewer summarize the same fitted model; they do not add fits.

See the [scientific specification](../../../../docs/cross_condition_generalization_spec_v1.md)
for the exact cell definitions, exclusions, and saved array contracts. This
guide covers setup, execution, and interpretation.

## Inputs and controls

Use version-3 `neural_session.json` with distinct PFC and HPC probe roles. The
session also needs the existing `task_decoding_config.json`, its augmented
trial CSV, feature-parameter JSON, and both regional configuration entries
required by that base configuration. These remain required loader inputs even
though cross-condition analysis selects the four targets and `choice_time`
alignment itself. The existing base target selection and alignment settings
are not applied; blocks are ignored.

An optional `cross_condition_config.json` beside the session metadata accepts
only these controls:

```json
{
  "base_config_path": "task_decoding_config.json",
  "bin_width_ms": 500
}
```

`base_config_path` must resolve inside the session directory and reference the
same metadata. Without this file, the runner uses the default base-config path
and 500-ms bins. Supported widths are 500, 100, 50, and 20 ms. The CLI width
overrides the contained width for that invocation; the app's preparation-width
selector does the same. Neither changes a previously saved run's settings.

## Prepare one session

From the repository root, set `SESSION` to the session directory containing
`neural_session.json`:

```bash
SESSION=/path/to/session
uv run --frozen python -m src.neural_analysis.task_decoding.cross_condition.run_session \
  --session-metadata "$SESSION/neural_session.json" --dry-run
```

The dry-run is a planning pass. It reads the small metadata/configuration,
behavior/trial tables and PCA archive metadata needed to report counts and
source status. It does not read neural score arrays, fit models, prepare PCA,
or write outputs. A source marked `compatibility_pending` means its regional
compatibility has not yet been verified by explicit preparation; it does not
mean the region is missing.

To prepare the full bundle at the default 500-ms width:

```bash
uv run --frozen python -m src.neural_analysis.task_decoding.cross_condition.run_session \
  --session-metadata "$SESSION/neural_session.json"
```

For example, select 100-ms bins or explicitly create a new run while reusing
compatible PCA:

```bash
uv run --frozen python -m src.neural_analysis.task_decoding.cross_condition.run_session \
  --session-metadata "$SESSION/neural_session.json" --bin-width-ms 100
uv run --frozen python -m src.neural_analysis.task_decoding.cross_condition.run_session \
  --session-metadata "$SESSION/neural_session.json" --rerun
```

If a contained controls file has a nondefault name, pass its path with
`--config`. Explicit preparation verifies regional compatibility, reuses each
confidently compatible saved PCA, and asks the existing timeline preparation
runner only for missing regional sources. PFC and HPC availability are
reported independently with reasons, including when both are unavailable.
No Slurm or cluster workflow is provided for this analysis.

## Prepare a session list

Create a text file containing one metadata path per line. Relative paths are
resolved from the list file's directory; blank lines and lines beginning with
`#` are ignored. Paths must identify existing `neural_session.json` files.

```text
# sessions.txt
../session-01/neural_session.json
../session-02/neural_session.json
```

Then run:

```bash
uv run --frozen python -m src.neural_analysis.task_decoding.cross_condition.run_batch \
  --session-list /path/to/sessions.txt --dry-run
uv run --frozen python -m src.neural_analysis.task_decoding.cross_condition.run_batch \
  --session-list /path/to/sessions.txt --bin-width-ms 100 --workers 2
```

Without `--workers`, parallelism is capped by CPU count and session count.
Choose a smaller positive worker count if session memory use requires it.
Batch dry-run is sequential and starts no worker pool. Normal batch execution
parallelizes across sessions and returns reports in list order.

## Inspect and interpret results

Open the metadata-driven neural-analysis webapp and select **Cross-condition
generalization**. The view discovers saved runs for that session. **Saved run**,
**Target**, **Supervised window**, **Weighting**, **Evaluation role**, and
**Outcome** select cached saved summaries; changing them does not fit or load
raw/PCA inputs. **Prepare cross-condition results** and **Recompute
cross-condition results** explicitly create the complete bundle: sixteen
directions, three windows, two weightings, and two regions. Preparation width
is separate from the width recorded in an already saved run. If preparation
fails, the previously selected saved run remains selected.

Each table shows all four directions separately for PFC and HPC. Accuracy and
balanced accuracy are percentages displayed to one decimal place; counts and
availability reasons remain visible. No direction aggregate or statistical
test is reported. Single-class outcome groups can have ordinary accuracy but
not balanced accuracy.

The supervised windows and the fixed source-PCA window are:

| Selection | Choice-relative interval |
| --- | --- |
| Full | `[-1, +1)` seconds |
| Before | `[-1, 0)` seconds |
| After | `[0, +1)` seconds |
| Existing PCA source window | `[-2, +2)` seconds, unchanged |

Training scores are in-sample diagnostics. Complementary-test trials are
excluded from fitting. The fixed session-wide PCA includes neural activity
from test conditions, so this is condition transfer, not temporal or
block-held-out validation. Each trial contributes equal evaluation weight
under either saved fitting weighting. Fits use all saved unwhitened regional
PCs. `equal_trial` gives every source trial total weight 1. For `equal_cell`,
with N eligible source trials and n_cell trials in its source cell, each trial
in that cell gets total weight N/(2*n_cell). When both source cells are populated,
both schemes sum to N. Selected bins divide each trial's total weight, so
narrower windows do not reduce it.

Value (correctness-based) predicts recorded correctness: correct choices and
correct omissions are class 1; incorrect choices are class 0. Its probability
means **P(correct)**, not Q, expected reward magnitude, or subjective value.
Actual reward uses the recorded reward flag. Before predicts eventual recorded
reward; After is post-choice and is not necessarily after reward delivery.
Context and choice refer to recorded context and actual choice.

Outcome summaries reuse the combined-pool fit. Correct and omission subsets
are not separately trained models. Summary populations can overlap, so their
counts should not be added as disjoint categories. Existing manual-reward,
no-choice, and alignment exclusions apply before context checks. Run details
retain the dark-state exclusion count and context/choice/correctness conflict
audit, including original trial IDs.

## Saved outputs and copying

Runs are new timestamped directories beneath the session's
`analysis_runs/cross_condition_generalization_*` path. A run contains a
self-contained `results.npz`, `config.json`, `run.log`, `summary.md`, the
actual main and batch scripts, and source snapshots. Existing runs are not
overwritten; `--rerun` creates another run.

For saved-only viewing on another machine, copy the session's
`neural_session.json` and the desired run directory beneath its
`analysis_runs/` directory. The viewer needs those files to discover and open
the result; raw behavior, augmented trials, features, and PCA archives are not
needed to inspect a completed self-contained run. A new preparation does
require the listed session inputs and any missing PCA source.

To open the metadata-driven webapp for that session:

```bash
uv run --frozen streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata "$SESSION/neural_session.json"
```

Synthetic validation prepared 192 models for 48 choices in about 0.214 s and
reported peak process RSS of 316,366,848 bytes, including imports. Reusing the
saved synthetic result took about 0.070 s. These are measurements of one small
synthetic run, not performance guarantees for experimental sessions.

Experimental preparation and inspection on designated sessions remain a
separate acceptance step; synthetic validation does not establish
experimental scientific acceptance.
