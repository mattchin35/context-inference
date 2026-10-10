# Pooled-Window Task-Variable Decoding and Condition-Generalization Specification

**Status:** Revision 1 documentation revision complete; awaiting user review.
This document does not authorize code changes or scientific computation.

**Implementation plan:** See
`docs/task_variable_condition_generalization_implementation_plan.md`.

**Inherited contracts:** `docs/task_variable_spec_v5.md`,
`docs/task_variable_spec_v6.md`, and `docs/task_variable_spec_v7.md` remain the
authority for existing target definitions, trial conditions, neural inputs,
unit identities, rate tensors, estimator families, metrics, and provenance
unless this document explicitly defines different behavior for the new
pooled-window workflow. The existing documents continue to describe the
implemented condition-specific, per-time-bin decoding analysis. This file
retains its committed name while its scope now includes both native temporal
stability and condition generalization.

## 1. Scientific goal

The existing task-variable decoder asks whether an independently optimized
linear readout can decode a target at each time within each condition. The new
workflow constrains one decoder to be shared across a selected training window
and then repeatedly evaluates that frozen decoder across the complete time
axis.

The primary direction remains:

```
neural activity -> task variable
```

It has two coequal scientific views.

The native temporal-stability view is:

```
condition B pooled across a training window -> condition B at time t
```

It asks whether one shared within-condition population readout remains usable
across time. The `full` training window pools the complete choice-aligned
[-2 s, +2 s) interval and is the direct stable-readout counterpart to the
existing sequence of independently optimized time-bin decoders.

The condition-transfer view is:

```
correct_rewarded -> test condition
versus
test condition -> same test condition
```

The first term measures transfer of the reference-condition readout. The
second is the same native pooled-window result used by the temporal-stability
view. Strong native decoding with weak reference transfer supports a
condition-dependent change in the usable population readout. Weak performance
in both does not support that conclusion because the target may simply be
poorly decodable in the destination condition.

Comparing the existing time-bin-specific score with the new pooled-native
score is descriptive. A strong time-bin-specific score with a weak
pooled-native score can reflect a difference between independently
cross-validated time-specific and pooled curves; it is not by itself evidence
of biological remapping or newly created information.

| Existing time-bin-specific score | Pooled-native score | Descriptive interpretation |
| --- | --- | --- |
| Strong | Strong | One shared readout and independently optimized readouts both recover the target. |
| Strong | Weak | The usable optimal readout may change through time, or the constrained pooled fit may be disadvantaged. |
| Weak | Weak | The target is not reliably decoded by either readout in that interval. |
| Weak | Strong | Inspect sampling, regularization, and estimator variance before interpreting the difference. |

This remains a predictive population analysis. It does not establish causal
neural encoding, individual-neuron remapping, or publication-level inference.

## 2. Analysis identity and separation

Pooled-window decoding and condition generalization form one separate analysis
identity and saved-run type. They live under
`src/neural_analysis/task_decoding` so they can reuse the current validated
inputs and modeling primitives, but they must not change the meaning or result
schema of existing task-variable decoding runs.

The initial implementation has its own single-session entry point, immutable
run directory, small configuration overlay, checkpoints, result schema,
summary, and figures. Batch/Slurm execution and an automated read-only
comparison view are deferred until the synthetic and single-session gates pass.
Existing task-variable results remain readable and are never silently merged
into a new pooled-window run.

## 3. Terminology

- **Reference condition:** the fixed `correct_rewarded` condition used to
  train the readout transferred to other conditions.
- **Destination condition:** a configured condition on which a fitted
  reference decoder is evaluated.
- **Native decoder:** a decoder trained and evaluated within one destination
  condition using the common held-out block assignment.
- **Pooled-native evaluation:** repeated evaluation through time of one frozen
  native decoder trained from a named multi-bin window.
- **Reference transfer:** evaluation of a `correct_rewarded` decoder on a
  destination condition.
- **Time-bin-specific decoder:** one independently fitted decoder at one time
  bin from the existing task-variable workflow.
- **Time-specific versus pooled comparison:** a descriptive side-by-side view
  of independently completed time-bin-specific and pooled-native results. It
  is not a paired metric in revision 1. The existing decoder fits its regional
  scaler/PCA basis across the complete analysis window, whereas `pre_choice`
  and `post_choice` pooled models fit those transforms only in their named
  windows. Thus those overlays compare preprocessing as well as readout
  constraints. The `full` pooled model is the closest transform-matched
  comparison.
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

Conditions are inherited from the referenced base task-decoding configuration.
Its existing default remains `all`; the pooled workflow does not silently give
that field a second meaning. The reference condition is always added in
canonical order if the base subset omits it. The portable complete example
selects all six conditions explicitly in its base configuration. Condition
masks retain their existing meanings and may overlap. In particular,
switch/stay are not treated as mutually exclusive levels of the same outcome
factor as correct-rewarded/omission/incorrect.

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

The saved masks remain independent audit facts: `target_eligible` records only
target validity, while `master_eligible` records the complete intersection.
When a row fails more than one requirement, its single saved reason uses this
fixed precedence: `baseline_ineligible`, then `target_ineligible`, then
`neural_window_unavailable`; eligible rows use no reason. The common neural
tensor contains only baseline-valid rows with bilateral coverage, so a tensor
row can have fold ID `-1` only when it is target-ineligible.

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
| `pre_choice` | [-0.5 s, 0 s) | Default anticipatory/choice-period native and reference readout. |
| `post_choice` | [0 s, +0.5 s) | Immediate post-choice/outcome-period native and reference readout. |
| `full` | [-2 s, +2 s) | One native/reference readout pooled across the complete analyzed interval. |

Configuration contains an ordered, nonempty `training_window_names` list.
Omitting it selects only `pre_choice`. Unknown names and duplicates are errors;
accepted names are restored to the canonical table order. Any subset,
including all three windows, is allowed.

To avoid duplicating the existing validated task-decoding configuration, the
pooled configuration is a small overlay with exactly `base_config_path` and
optional `training_window_names`. The referenced base configuration supplies
the session/input paths, conditions, targets, bin width, PCA counts, grouped
fold count, fixed estimator controls, trusted bounds, and output root. It must
select `choice_time` alignment and fixed regularization. The resolved base
scientific payload, effective condition list, fixed reference condition,
resolved window definitions, pooled-overlay file identity, and base-config
file identity are part of the pooled run fingerprint.

Window selection is immutable scientific configuration. Runtime measurements
must never cause a running analysis to add or remove windows automatically.
The first experimental benchmark uses `pre_choice`. Running all three windows
later requires an explicit configuration chosen after inspection of measured
runtime, memory, and output size.

Every training-window boundary must coincide with existing time-bin edges.
Failure to align exactly is a configuration error; bins are never truncated,
rounded, or partially included. A bin is selected only when its complete
half-open interval `[edge_i, edge_{i+1})` is contained in the named half-open
training window.

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
trials. Give every stacked row sample weight `1 / n_training_bins`, so each
eligible training trial has total weight one regardless of whether its window
contains 5 or 40 bins. This weighting does not make rows independent; trials
remain the sampling units. It preserves the intended loss-to-penalty scale for
the configured SAGA logistic regression. Scikit-learn's ElasticNet normalizes
a uniform sample-weight vector to sum to the row count, so its fit is unchanged
by these equal weights. Both estimator calls receive the weight vector rather
than relying on this implementation detail. Save the trial count, stacked
observation count, and fixed weighting policy; never report the stacked row
count as the independent sample count.

All bins from a trial remain in the same outer and, in any future tuned mode,
inner partition. Row-wise random splitting is forbidden.

The initial scientific implementation supports fixed regularization only.
Tuned regularization requires a later specification amendment because inner
grouped splits and fit counts must be reviewed separately.

## 8. Shared master block folds

One deterministic target-specific outer block assignment is shared by every
reference-transfer and native-condition comparison for that target.

Construct the assignment after projecting complete-table target values,
eligibility, conditions, and behavioral block identities onto the common
bilateral neural-tensor row axis. Use all master-eligible tensor rows before
condition filtering:

- categorical targets use deterministic `StratifiedGroupKFold`;
- numerical targets use deterministic `GroupKFold`;
- groups are canonical behavioral block identities; and
- the configured outer fold count is five or an explicitly configured three.

Saved full-table arrays keep the chronological audit trail;
`trial_row_indices` is the explicit mapping from common tensor row to that
table. Split indices and neural tensors use only the common tensor-row axis.

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
Every native model is evaluated across the complete time axis in its own
condition. These pooled-native curves are primary outputs, not merely controls
for condition transfer. The correct-rewarded native model also supplies every
reference-transfer evaluation.

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

The decoder is never refit across evaluation times. Adjacent points in every
pooled-native or transfer curve are repeated evaluations of the same
fold-specific pooled-window decoder, unlike adjacent points in the existing
task-variable decoding curves.

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

For `correct_rewarded`, the native and reference-transfer source are the same
stored evaluation. Its valid gap is therefore exactly zero and serves as a
schema invariant; it never causes a duplicate fit or evaluation.

After the pooled single-session result is validated, an optional read-only
report may place completed per-time-bin and pooled-native complete-fold means
on the same axes for common session, target, condition, time, region,
representation, and metric only when neural-source and unit-selection identity
also match. Both transform-training provenances must be present and labeled.
Because the two analyses can use different eligible populations and fold
assignments, revision 1 does not subtract them or present them as paired
observations. The report states that the existing decoder fits its regional
transform across the complete analysis window, while pre/post pooled models
fit window-specific transforms; the `full` pooled model is the nearest
transform-matched comparison. The overlay must be labeled as independently
cross-validated and descriptive.

This view compares an unconstrained sequence of time-bin-specific readouts
with one decoder constrained across a training window. Differences can reflect
model flexibility, regularization, convergence, sampling, or estimator
variance as well as temporal changes in the usable population readout. They
must not be labeled remapping, new encoding, or a significance test. The
pooled-window computation never requires or reruns the existing analysis.

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

Revision 1 uses analysis version `pooled-task-variable-decoding-v1` and result
schema version `1`. The implementation plan freezes the required array names,
axis order, dtypes, status values, sentinels, reason-code vocabulary,
checkpoint filenames, and run-directory layout. The scientific axes are
training window, condition/evaluation pair, target, outer fold, region,
representation, evaluation time, and metric. Model arrays omit evaluation
time and destination because one fitted model is reused.

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

Store full-table condition masks and target eligibility. Separately store the
common bilateral neural-tensor row mapping and target-specific fold identities
on that tensor-row axis. The pipeline projects full-table target values,
eligibility, conditions, and block identities through `trial_row_indices`
before splitting or decoding; neural tensors are never treated as though they
contain every chronological table row. Also store exact time-bin
edges/centers, exact training-window bin masks, stable unit and feature
identities, transform selection/count provenance, coefficients, intercepts,
estimator controls, train/test counts, the fixed sample-weight policy, units,
configuration, source fingerprint, input manifest, random seed, analysis
version, and software versions. As in the current task decoder, fitted
scaler/PCA matrices are reconstructed from immutable inputs and code rather
than persisted. Use named arrays in a versioned NPZ plus JSON metadata
sidecars with `allow_pickle=False`. Do not overwrite earlier runs.

Revision 1 does not store dense per-trial out-of-fold predictions. It computes
probability-dependent metrics in memory and stores fold-level metrics plus the
model, fold, source, and configuration provenance needed to reproduce them.
Adding dense predictions requires a later reviewed schema
revision with a concrete scientific consumer and byte estimate.

## 14. Checkpointing and execution

Checkpoint exactly one training-window/target cell so an interrupted run does
not repeat completed model families. The execution loop produces and saves one
window-target result at a time; no checkpoint API accepts an aggregate that
also contains other windows. Checkpoints are immutable, fingerprinted, and
include every configured condition, fold, region, representation, required
evaluation pair, evaluation time, and metric for that cell before being
considered complete.

Terminal completion and `final_results_published` are set only after the final
result, summary, and required figures all validate. Exact completed reentry is
read-only.

The validated revision-1 workflow first provides single-session dry-run,
`new`, local detached execution, `status`, and exact `resume`. Reuse the current
lifecycle behavior where its contracts apply, but do not make existing
task-decoding commands infer or run pooled decoding. Slurm single-session and
session-array execution are a later work package entered only after synthetic
validation, a user-inspected experimental session, and a measured resource
envelope.

## 15. Reporting

For each training window and condition, default figures first show the native
pooled-window score across evaluation time. For each destination condition,
condition-transfer figures compare:

- `correct_rewarded -> destination`; and
- `destination -> destination`.

Curves use the complete evaluation-time axis and identify target, region,
representation, metric, training window, train condition, and destination.
Generalization-gap plots show native minus transfer with a zero reference.
Every time-course figure visibly shades or marks the named training interval.
Captions distinguish held-out evaluation times inside that interval from
out-of-window temporal generalization. For `full`, every displayed time is
within the training-time distribution even though all evaluated trials remain
held out.

After single-session validation, an optional separate read-only report may
overlay a completed existing time-bin-specific curve on common coordinates.
It labels both curves as independently cross-validated and descriptive and
does not calculate a time-specific-minus-pooled difference.

Outcome conditions and switch/stay conditions must be visually separated or
clearly labeled as overlapping contrasts. Do not present all condition names
as mutually exclusive levels.

Captions must state that:

- one pooled-window decoder is repeatedly evaluated through time;
- native pooled-window curves test temporal stability of one readout;
- folds are grouped by behavioral block;
- native and transfer scores use the same destination held-out trials;
- fold variation is not a confidence interval;
- the existing and pooled curves can differ in transform-training windows; and
- neither transfer failure nor a difference between independently
  cross-validated time-specific and pooled curves alone proves remapping.

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

Regional transforms are shared across the three region configurations and two
representations for one window/condition/target/fold cell. The exact requested
transform-fit count is therefore:

```
n_training_windows * n_conditions * n_targets * n_outer_folds
    * 2 anatomical regions
```

It is independent of representation, region configuration, evaluation time,
and destination count.

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

1. a time-invariant within-condition code is recovered by the full-window
   pooled-native decoder across its expression period;
2. a sign-changing or rotating within-condition code remains decodable by
   independent time-bin models but is not falsely labeled stable by one
   full-window decoder;
3. a stable population code transfers and has a small native-transfer gap;
4. a destination-specific remapped code remains natively decodable but has
   weak reference transfer and a positive gap;
5. an absent destination signal makes both native and transfer weak;
6. a delayed shared code produces a shifted peak in the time-resolved transfer
   curve without fitting time-specific decoders;
7. a reference-constant target is explicitly unavailable while an eligible
   native destination can remain available;
8. overlapping conditions cannot leak a trial across train/test blocks;
9. every stacked time row from a trial remains in one fold and receives
   `1 / n_training_bins` weight, so every eligible trial has total weight one;
10. test-condition rescaling or PCA refitting never occurs;
11. selecting `full` at 100 ms fits one model per model identity, not 40;
12. selecting multiple training windows creates distinct models and
    provenance;
13. selecting all three windows does not change the pre-choice result;
14. seeded reruns reproduce arrays and metadata exactly.

## 18. Revision-1 exclusions

Revision 1 excludes:

- tuned regularization;
- arbitrary user-defined training-window bounds;
- train-time x test-time decoder matrices;
- automatic window selection based on runtime or performance;
- test-condition recalibration;
- dense per-trial prediction storage;
- a paired time-specific-minus-pooled score;
- web-app integration;
- batch or Slurm implementation before the single-session benchmark;
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
- Native pooled-window time courses are primary scientific outputs.
- Every configured destination receives reference-transfer evaluation.
- Evaluation remains time resolved across the full existing axis.
- One decoder is fit per training window, not per evaluation time.
- `full` means exactly the complete choice-aligned [-2 s, +2 s) interval.
- After single-session validation, compatible current per-time-bin results may
  be overlaid read-only and descriptively, without subtraction or refitting.
- Window selection never changes automatically after launch.
- Existing task-variable specification and result meaning remain unchanged.
