# Cross-condition generalization

This separate analysis transfers binary readouts across task conditions for
four targets: Context, Choice, Actual reward, and Value (correctness-based).
The original four condition transfers remain available for all targets. A new
Context-only **Outcome transfers** view also compares correct-rewarded,
correct-omission, and incorrect trials. It has three source models, each
evaluated on the other two outcomes. The existing incorrect-trained Context
model is reused in both views; only two source model definitions are added.

The expanded v2 bundle has 18 readout definitions, three supervised windows,
two training weightings, and separate PFC/HPC models: up to 216 fits per bin
width. Missing classes or regions can reduce the actual number of fits.
Outcome evaluations and source diagnostics reuse the same saved models and add
no fits.

See the [scientific specification](../../../../docs/cross_condition_generalization_spec_v1.md)
for the exact cell definitions, exclusions, and saved array contracts, and the
[outcome-generalization implementation plan](../../../../docs/context_outcome_generalization_implementation_plan.md)
for the approved Context outcome extension. This guide covers setup,
execution, and interpretation.

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

The expanded analysis writes `cross-condition-generalization-v2` results with
schema 2. Existing v1 decoder bundles are left intact and are not listed by
the v2 view. Prepare once to create the expanded bundle; there is no v1
migration or import of old decoder coefficients. This decoder version change
does not change compatible PCA-v1 sources: preparation reuses them when their
regional identity and settings match.

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

Launch the metadata-driven neural-analysis webapp (from the repository root):

```bash
uv run --frozen streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata "$SESSION/neural_session.json"
```

Select **Cross-condition generalization**. **Saved run**, **Target**,
**Supervised window**, **Weighting**, and **Evaluation role** select cached
saved summaries; changing them does not fit or load raw/PCA inputs. For the
Context target only, **Context analysis** selects **Condition transfers** or
**Outcome transfers** and defaults to Condition transfers. Choice, Actual
reward, and Value retain their original condition-transfer view and **Outcome**
selector. The generic Outcome selector is hidden in Context Outcome transfers
because both complementary outcomes are shown together.

Condition transfers show the original four directions for the selected target.
In Context Outcome transfers, **Complementary test** shows six separate
training-outcome/test-outcome pairs for both regions (12 rows total).
**Training (in-sample)** shows the three source-model diagnostics (six rows).
Unavailable regions remain visible with reasons in either role. The model trained on
incorrect trials is the existing `context_incorrect_to_correct` model. No
outcomes or directions are averaged together.

The outcome table displays **Training trials**, **Training Left**, and
**Training Right** as unweighted counts of unique eligible source trials after
regional coverage. For Context, Left is class 1 and Right is class 0. These
counts are not bin counts, weight totals, or counts after filtering the test
outcome; they remain the same under both weightings and both test outcomes.
Eligible/scored and Right/Left trial counts describe the selected evaluation
role. A known source count is retained when a fit fails, with the reason shown;
an unavailable region has unknown counts (displayed as missing), not zero. No
minimum sample-count threshold is applied.

Accuracy and balanced accuracy are percentages displayed to one decimal place.
Single-class outcome groups can have ordinary accuracy but not balanced
accuracy. Availability reasons and class counts remain visible. No direction
aggregate or statistical test is reported.

**Prepare cross-condition results** and **Recompute cross-condition results**
explicitly create the complete 18-definition bundle across three windows, two
weightings, and two regions: up to 216 fits. Preparation width is separate from
the width recorded in an already saved run. If preparation fails, the
previously selected saved run remains selected.

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

In Condition transfers, outcome summaries reuse the combined-pool fit. In
Context Outcome transfers, each training outcome has its own saved model,
which supplies both other outcomes' test summaries. Summary populations can
overlap, so their counts should not be added as disjoint categories. Existing
manual-reward, no-choice, and alignment exclusions apply before context checks. Run details
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

In one expanded synthetic session with 42 choices (42 PFC-covered and 33
HPC-covered), preparation completed 216 fits in 0.233 s with peak process RSS
of 317,730,816 bytes, including imports. Exact saved-result reuse took 0.061 s
without PCA-array loading or fitting, and the PCA-v1 source hash stayed
unchanged. These small synthetic measurements are not performance guarantees
for experimental sessions. No experimental session was processed.

Experimental preparation and inspection on designated sessions remain a
separate acceptance step; synthetic validation does not establish
experimental scientific acceptance.
