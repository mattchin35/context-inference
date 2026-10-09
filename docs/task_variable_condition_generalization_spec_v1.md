# Task-Variable Condition-Generalization Analysis Specification

**Status:** Revision 1, planning only. This document does not authorize code
changes or scientific computation.

**Implementation plan:** See
`docs/task_variable_condition_generalization_implementation_plan.md`.

**Inherited contracts:** `docs/task_variable_spec_v5.md`,
`docs/task_variable_spec_v6.md`, and `docs/task_variable_spec_v7.md` remain the
authority for existing target definitions, trial conditions, neural inputs,
unit identities, rate tensors, estimator families, metrics, and provenance
unless this document explicitly defines different behavior for the new
condition-generalization workflow. The existing documents continue to
describe the implemented condition-specific, per-time-bin decoding analysis.

## 1. Scientific goal

The existing task-variable decoder asks whether an independently optimized
linear readout can decode a target at each time within each condition. This
analysis instead asks whether a population readout learned from
correct-rewarded trials transfers to held-out trials from other conditions.

The primary direction remains:

```
neural activity -> task variable
```

The new comparison is:

```
correct_rewarded -> test condition
versus
test condition -> same test condition
```

The first term measures transfer of the reference-condition readout. The
second measures native decodability in the destination condition. Strong
native decoding with weak reference transfer supports a condition-dependent
change in the usable population readout. Weak performance in both does not
support that conclusion because the target may simply be poorly decodable in
the destination condition.

This remains a predictive population analysis. It does not establish causal
neural encoding, individual-neuron remapping, or publication-level inference.

## 2. Analysis identity and separation

Condition generalization is a separate analysis identity and saved-run type.
It lives under `src/neural_analysis/task_decoding` so it can reuse the current
validated inputs and modeling primitives, but it must not change the meaning
or result schema of existing task-variable decoding runs.

The initial implementation will have its own single-session and batch entry
points, immutable run directory, configuration, checkpoints, result schema,
summary, and figures. Existing task-variable results remain readable and may
be used as descriptive context, but they are not silently merged into a new
generalization run.

## 3. Terminology

- **Reference condition:** the fixed `correct_rewarded` condition used to
  train the readout transferred to other conditions.
- **Destination condition:** a configured condition on which a fitted
  reference decoder is evaluated.
- **Native decoder:** a decoder trained and evaluated within one destination
  condition using the common held-out block assignment.
- **Reference transfer:** evaluation of a `correct_rewarded` decoder on a
  destination condition.
- **Training window:** a named interval whose bins are pooled to fit one
  decoder.
- **Evaluation time:** one existing event-relative time bin at which a frozen
  pooled-window decoder is evaluated.
- **Generalization gap:** the paired native-condition score minus the
  reference-transfer score on the same destination condition, held-out fold,
  evaluation time, metric, region, and representation.

## 4. Fixed reference and configured conditions

The reference condition is exactly `correct_rewarded`, using the canonical
revision-7 mask. It is not configurable in revision 1.

Destination and native conditions use a nonempty configured subset of the
existing canonical order:

1. `all`;
2. `correct_rewarded`;
3. `omission`;
4. `incorrect`;
5. `switch`; and
6. `stay`.

Omitting the destination-condition field selects all six conditions. The
reference condition is always included if an explicit subset omits it. The
complete example also selects all six conditions explicitly. Condition masks
retain their existing meanings and may overlap. In particular, switch/stay are
not treated as mutually exclusive levels of the same outcome factor as
correct-rewarded/omission/incorrect.

All test rows come only from held-out behavioral blocks. Therefore an
overlapping switch/stay and correct-rewarded membership cannot place the same
trial in both model training and evaluation.

## 5. Targets and eligibility

The workflow accepts the same configured target subset and exact target
definitions as the current task-variable analysis. Targets are constructed on
the complete chronological table before condition filtering.

For each target, the master eligible population is the intersection of:

- existing baseline eligibility;
- target-specific eligibility; and
- bilateral complete neural-window coverage.

Training and evaluation then intersect this master population with condition
and fold masks. A condition filter must never redefine previous/next-trial
targets.

A target that is constant or otherwise unlearnable in `correct_rewarded`
produces explicit unavailable reference-transfer cells. Native decoders in
other conditions may still be computed. Selecting the same target list across
conditions requires retaining these cells, not fabricating a transfer score or
dropping the target. For example, correctness itself is structurally constant
on correct-rewarded trials.

## 6. Time axis and canonical training windows

Revision 1 requires `choice_time` alignment. Evaluation uses the complete
existing choice-relative axis: [-2 s, +2 s] with the configured supported bin
width. A frozen decoder is evaluated separately at every existing bin, so its
performance remains time resolved. Supporting trial-start alignment requires a
later revision with training-window names that do not incorrectly imply
choice-relative epochs.

Revision 1 defines three canonical half-open training windows:

| Identifier | Bounds relative to alignment | Purpose |
| --- | --- | --- |
| `pre_choice` | [-0.5 s, 0 s) | Default anticipatory/choice-period reference readout. |
| `post_choice` | [0 s, +0.5 s) | Immediate post-choice/outcome-period reference readout. |
| `full` | [-2 s, +2 s) | Strong test of one readout pooled across the entire analyzed interval. |

Configuration contains an ordered, nonempty `training_window_names` list.
Omitting it selects only `pre_choice`. Unknown names and duplicates are errors;
accepted names are restored to the canonical table order. Any subset,
including all three windows, is allowed.

Window selection is immutable scientific configuration. Runtime measurements
must never cause a running analysis to add or remove windows automatically.
The first experimental benchmark uses `pre_choice`. Running all three windows
later requires an explicit configuration chosen after inspection of measured
runtime, memory, and output size.

Every training-window boundary must coincide with existing time-bin edges.
Failure to align exactly is a configuration error; bins are never truncated,
rounded, or partially included.

## 7. Pooled-window training representation

For one target, outer fold, training condition, training window, region, and
representation, gather a tensor with shape:

```
(n_training_trials, n_training_bins, n_features)
```

Reshape it in stable trial-major then time-major order to:

```
(n_training_trials * n_training_bins, n_features)
```

Repeat each trial-level target once per included training bin. The estimator
therefore learns one coefficient vector shared across the selected training
window. It does not receive time as a feature and does not receive a hidden
one-hot bin identity.

Repeated bins from one trial are correlated observations, not independent
trials. Assign every stacked row weight `1 / n_training_bins`, so every trial
has total training weight one. Do not report the stacked row count as the
independent sample count. Save both the trial count and stacked observation
count.

All bins from a trial remain in the same outer and, in any future tuned mode,
inner partition. Row-wise random splitting is forbidden.

The initial scientific implementation supports fixed regularization only.
Tuned regularization requires a later specification amendment because
sample-weight handling, inner grouped splits, and fit counts must be reviewed
separately.

## 8. Shared master block folds

One deterministic target-specific outer block assignment is shared by every
reference-transfer and native-condition comparison for that target.

Construct the assignment on all master-eligible target rows before condition
filtering:

- categorical targets use deterministic `StratifiedGroupKFold`;
- numerical targets use deterministic `GroupKFold`;
- groups are canonical behavioral block identities; and
- the configured outer fold count is five or an explicitly configured three.

After applying condition masks, validate every required training and test
subset rather than assuming the master split guarantees condition-specific
class or target support. Do not change folds, search seeds, or fall back to
trial-wise splitting when a subset is unavailable.

For fold `f`:

- reference training uses only correct-rewarded rows outside held-out blocks;
- reference transfer uses destination-condition rows inside held-out blocks;
- a destination-native model uses only destination rows outside held-out
  blocks; and
- its native evaluation uses destination rows inside the same held-out blocks.

This makes transfer and native scores fold-paired on the same destination
trials.

## 9. Training-only transforms

Each model receives its own training-condition, training-window, and
outer-fold transform.

For each required region:

1. pool only that model's training trials and selected training-window bins;
2. calculate unit means and standard deviations in Hz;
3. remove constant or nonfinite training features;
4. fit regional PCA when requested; and
5. apply the frozen transform unchanged to every evaluation time and
   destination trial scored by that model.

No destination-condition or evaluation-time recentering, rescaling, PCA
refitting, threshold calibration, or intercept adjustment is permitted. Such
operations would remove part of the condition shift that the analysis is
intended to measure.

Separate PFC and HPC transforms and stable direct-unit identities retain the
current contracts. The PCA component limit is:

```
min(usable units, training trials * selected training bins)
```

For a reference model, one fitted transform and estimator are reused across
all destination conditions and all evaluation times. `correct_rewarded ->
correct_rewarded` is both the reference held-out baseline and the native
correct-rewarded result; it must not be fit twice.

## 10. Models and evaluations

For `K` configured conditions and one training window, fit one native model
per condition, target, outer fold, region configuration, and representation.
The correct-rewarded native model also supplies every reference-transfer
evaluation.

The required evaluation pairs are:

```
correct_rewarded -> each configured destination condition
each configured condition -> itself
```

Duplicate `correct_rewarded -> correct_rewarded` identity is stored once. No
other cross-condition pair is implied. With all six conditions there are 11
unique evaluation pairs.

For every fitted model, construct evaluation features separately at each time
bin from held-out destination trials. Categorical models save and score the
positive-class probability with the current positive-class orientation.
Numerical models save and score predictions in native target units.

The decoder is never refit across evaluation times. Adjacent points in a new
generalization curve are repeated evaluations of the same fold-specific
pooled-window decoder, unlike adjacent points in the existing task-variable
decoding curves.

## 11. Metrics and generalization gaps

Use the current metric definitions:

- categorical: balanced accuracy and ROC AUC from the same positive-class
  probabilities;
- numerical: held-out R2 in native target units.

Calculate metrics separately for each outer fold, evaluation time,
destination, region, representation, target, and training window. Retain below
chance balanced accuracy/AUC and negative R2.

For destination condition `B`, calculate a paired fold-level gap only when
both underlying scores are valid:

```
gap(B) = score(B -> B) - score(correct_rewarded -> B)
```

A positive gap means the destination-native readout outperformed the
transferred reference readout. The gap is descriptive and is not a confidence
interval or single-session significance test.

Primary display values require all configured outer folds to be valid, as in
the current task-variable analysis. Preserve individual valid folds and exact
unavailable reasons when the complete-fold requirement is not met.

## 12. Scientific validity and unavailable states

Declared unavailable states include:

- the reference or native training subset is empty;
- a categorical training subset lacks two classes;
- a categorical held-out subset lacks both required classes;
- a numerical training or held-out target is constant or too small for its
  metric;
- grouped folds cannot satisfy the requested contract;
- no usable training feature remains;
- a convergence warning occurs;
- predictions, probabilities, coefficients, intercepts, or scores are
  nonfinite or malformed; and
- a paired gap lacks either required valid score.

An unavailable reference-transfer cell does not suppress a valid native cell.
An unavailable destination at one time does not stop other times,
destinations, targets, or windows. Unexpected schema, programming, or I/O
errors fail the run and preserve completed checkpoints.

## 13. Result contract

Results must make model identity distinct from evaluation-pair identity.

Model provenance is indexed by:

```
training_window
train_condition
target
outer_fold
region_configuration
representation
```

It includes transform provenance, trial/stacked-row counts, estimator
parameters, feature identities, coefficients, intercept, convergence status,
and availability.

Evaluation records are indexed by:

```
training_window
train_condition
test_condition
target
outer_fold
region_configuration
representation
evaluation_time
metric
```

They include train/test trial counts, class counts when applicable, status,
reason, and score. Paired generalization gaps have the same axes without a
free train-condition axis because their two source pairs are fixed by the
destination.

Store full-table condition masks, target eligibility, master fold identities,
exact time-bin edges/centers, exact training-window bin masks, stable unit
identities, units, configuration, source fingerprint, input manifest, random
seed, analysis version, and software versions. Use named arrays in a
versioned NPZ plus JSON metadata sidecars. Do not overwrite earlier runs.

Per-trial out-of-fold predictions are scientifically useful but can dominate
storage when all windows, targets, regions, representations, conditions, and
times are selected. Revision 1 requires fold metrics and sufficient
provenance to reproduce them. Whether dense per-trial predictions are retained
must be resolved by the synthetic size benchmark before the result schema is
frozen; they must not be added implicitly without an explicit axis and byte
estimate.

## 14. Checkpointing and execution

Checkpoint at least at the training-window/target boundary so an interrupted
run does not repeat all completed model families. Checkpoints are immutable,
fingerprinted, and must include every configured condition and required
evaluation pair for that cell before being considered complete.

Provide separate single-session and batch commands with dry-run, local
detached execution, status, exact resume, Slurm single-session execution, and
session-level Slurm arrays. Reuse the current lifecycle design where its
contracts apply, but do not make existing task-decoding commands infer or run
condition generalization.

## 15. Reporting

For each training window and destination condition, default figures compare:

- `correct_rewarded -> destination`; and
- `destination -> destination`.

Curves use the complete evaluation-time axis and identify target, region,
representation, metric, training window, train condition, and destination.
Generalization-gap plots show native minus transfer with a zero reference.

Outcome conditions and switch/stay conditions must be visually separated or
clearly labeled as overlapping contrasts. Do not present all condition names
as mutually exclusive levels.

Captions must state that:

- one pooled-window decoder is repeatedly evaluated through time;
- folds are grouped by behavioral block;
- native and transfer scores use the same destination held-out trials;
- fold variation is not a confidence interval; and
- transfer failure alone does not prove remapping.

Use light mode and PNG by default, following the current plotting contract.

## 16. Fit counts and performance policy

For fixed mode, the unique model-fit count before invalidities is:

```
n_training_windows * n_conditions * n_targets
    * n_outer_folds * 3 region configurations * 2 representations
```

With three windows, six conditions, 18 targets, and five folds, this is:

```
3 * 6 * 18 * 5 * 3 * 2 = 9,720 fitted models
```

With the default pre-choice window only, it is 3,240 models. Reference
transfer to additional destinations adds predictions and scores, not another
reference-model fit.

These counts do not predict runtime directly. A pooled-window model contains
more rows than one current time-bin model: five rows per trial at 100 ms for
each half-second window and 40 rows per trial for `full`. High-dimensional
direct-unit convergence may dominate. Dry run must report model counts,
stacked-row dimensions, predicted evaluation counts, exact major-array bytes,
and selected windows without loading large spike arrays or fitting.

The first experimental benchmark selects only `pre_choice`. After it
completes, inspect wall time, CPU time, peak RSS, estimator timing by
representation and target family, invalidity, checkpoint size, and final
output size. Selecting all three windows for routine execution is a separate
explicit configuration decision based on those measurements, not an automatic
threshold inside the program.

Default batch parallelism is across sessions after per-session resources are
measured. Do not add within-session parallel fitting without profiling and a
revised plan.

## 17. Required synthetic scientific cases

Before experimental use, seeded synthetic tests must demonstrate:

1. a stable population code transfers and has a small native-transfer gap;
2. a destination-specific remapped code remains natively decodable but has
   weak reference transfer and a positive gap;
3. an absent destination signal makes both native and transfer weak;
4. a delayed shared code produces a shifted peak in the time-resolved transfer
   curve without fitting time-specific decoders;
5. a reference-constant target is explicitly unavailable while an eligible
   native destination can remain available;
6. overlapping conditions cannot leak a trial across train/test blocks;
7. every stacked time row from a trial remains in one fold and trial weights
   sum to one;
8. test-condition rescaling or PCA refitting never occurs;
9. selecting multiple training windows creates distinct models and provenance;
10. selecting all three windows does not change the pre-choice result; and
11. seeded reruns reproduce arrays and metadata exactly.

## 18. Revision-1 exclusions

Revision 1 excludes:

- tuned regularization;
- arbitrary user-defined training-window bounds;
- train-time x test-time decoder matrices;
- automatic window selection based on runtime or performance;
- test-condition recalibration;
- cross-session pooled neural features;
- inferential statistics across sessions; and
- claims about causal encoding or individual-neuron remapping.

These require later reviewed amendments rather than silent expansion.

## 19. Decisions fixed for implementation planning

- Reference condition: `correct_rewarded`.
- Default training window: `pre_choice`.
- Other supported windows: `post_choice` and `full`.
- Any explicit subset or all three windows may be configured.
- Every configured condition receives a native decoder.
- Every configured destination receives reference-transfer evaluation.
- Evaluation remains time resolved across the full existing axis.
- One decoder is fit per training window, not per evaluation time.
- Window selection never changes automatically after launch.
- Existing task-variable specification and result meaning remain unchanged.
