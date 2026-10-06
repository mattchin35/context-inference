# Task-Variable Decoding Analysis Specification

**Status:** Revision 5, planning only. This specification incorporates the
codebase audit and the decisions confirmed after revision 4. It does not
authorize implementation or scientific computation.

**Implementation plan:** See `docs/task_variable_implementation_plan.md`.

## 1. Goal and scope

Measure when simultaneously recorded neural population activity contains
information about behavioral and task variables, how this information changes
around trial events, and whether it is captured by low-dimensional population
structure or by individual units.

The direction of prediction is:

$$
\mathrm{neural\ activity} \rightarrow \mathrm{task\ variable}
$$

The analysis is performed independently within each session. Units from
different sessions must never be treated as shared feature columns.

This is an inspection analysis, not finalized inferential statistics. The
implementation should favor explicit settings, deterministic behavior, and
scientifically readable outputs. It must preserve train/test separation, but it
must not add repeated validation, permutation significance tests, automatic
model-search expansion, decoding-onset claims, or publication-level inference.

Mouse speed is a possible future target and is not part of revision 5.

## 2. Required session inputs

One run requires explicit paths to:

1. `neural_session.json`;
2. the session's augmented-trial CSV;
3. `trial_feature_params.json`; and
4. a dedicated output subdirectory under the session data directory, disjoint
   from required input files/directories and outside the source-code Git
   checkout.

The metadata document remains authoritative for probe, sorter, channel-quality,
and aligned-spike paths. Revision 5 does not change the metadata schema merely
to point at the augmented table. The run configuration names the augmented
table and feature-parameter file explicitly because current metadata can point
to the raw trial table.

### 2.1 Lightweight augmented-table validation

Before loading large neural arrays, verify that the augmented table:

- exists and has at least one row;
- contains every source column required by the targets and selected alignment;
- contains finite, integer-valued `cur_trial` values equal to the complete
  zero-based row sequence `0, 1, ..., n_rows - 1`;
- contains a `cur_block` value considered present by the same project sentinel
  rules on every row; and
- contains each `cur_block` label in one contiguous row segment.

After applying the project's existing missing-value sentinel rules, plus its
recognized no-choice labels for `action`, every remaining present value in a
required numeric source column must be coercible to a finite number. The
selected alignment and each selected numeric target must also contain at least
one finite value. Recognized missing/no-choice values may still make individual
rows target-ineligible; malformed or infinite present values are input errors.

The required columns for the complete revision-5 target set are:

```
cur_trial
cur_trial_in_block
cur_block
state_int
action
correct
reward
experimenter_reward_given
choice_time
start_time
consecutive_omissions
consecutive_rewards
rewards_in_block
Qlearning_rel_value
FQlearning_rel_value
HMM_rel_value_logodds
HMM_rel_value_logodds_decay
relative_doubt_index
```

This is deliberately a small schema check. Do not introduce a general dataframe
schema framework.

The list above is the superset for all targets and both supported alignments.
Each run selects a nonempty, duplicate-free subset of the declared targets.
For one run, require the shared identity/baseline columns, the selected
alignment column, and only the selected targets' source columns. The default
complete target set with choice alignment therefore does not require
`start_time`. Report all missing required columns in one error.

The feature-parameter file must exist, decode to a JSON object, and be copied
verbatim into result provenance. Validation does not attempt to reconstruct or
rerun behavioral models.

## 3. Trial identity, ordering, and baseline eligibility

Rows are interpreted in chronological session order. Preserve `cur_trial`
as the stable behavioral trial identifier; do not silently replace it with a
filtered-row position.

A baseline decoding row must:

- contain a valid animal choice, right `0` or left `1`;
- not be an experimenter/manual-reward row;
- contain a nonmissing block identity in `cur_block`;
- have a finite selected alignment timestamp;
- have a complete requested neural window on both configured probes, because
  the initial run computes and directly compares all three region
  configurations; and
- have a valid value for the selected target.

Target-specific rules may exclude additional rows. A missing next-action label,
for example, must not remove that row from current-action decoding.

### 3.1 Shifted targets

Construct previous/next and switch/stay targets from the original chronological
trial sequence before applying decoding filters. Do not bridge across a manual
reward, a no-choice row, a missing adjacent action, or a session boundary.
When the immediately adjacent row lacks a valid animal choice, the shifted
target is unavailable.

### 3.2 Matched comparisons

For one target and alignment, direct comparisons across PFC, HPC, PFC + HPC,
PCA, and individual-unit representations use the same eligible trial
intersection and the same outer-fold assignments. This prevents a regional or
representation difference from being caused by different trial samples.

Different targets may have different eligible rows.

## 4. Categorical targets

All current categorical targets are binary. Record the original labels, numeric
mapping, and positive class in result metadata.

For source-coded binary fields, recognized missing sentinels make a row
ineligible; the project's recognized no-choice action labels do the same for
`action`. Otherwise `action` and `correct` must encode exactly 0 or 1 after
numeric conversion. `state_int` likewise accepts 0 or 1 for this target, while
the known dark-state value 2 is valid input but target-ineligible. Any other
present finite label is an input error rather than a third class.

| Display target | Source/derivation | Eligibility details |
| --- | --- | --- |
| Current state | `state_int` | Right context `0` vs left context `1`. Exclude dark state `2` and unknown states. |
| Current action | `action` | Right `0` vs left `1`. |
| Current action is correct | `correct` | Correctness relative to task context, not reward delivery. |
| Previous action | Shifted valid `action` | Immediately preceding chronological row only. |
| Previous action was rewarded | Shifted `reward > 0` | Requires a valid, non-manual previous choice. |
| Next action | Shifted valid `action` | Immediately following chronological row only. |
| Current choice switch/stay | `action[i] != action[i-1]` | Requires valid current and previous choices. |
| Next choice switch/stay | `action[i+1] != action[i]` | Requires valid current and next choices. |

Choice switching is not a task-context transition. Each switch/stay row above is
one binary target, not separate switch and stay models.

Use the existing integer-one category as the positive class: left for
state/action targets, correct for correctness, rewarded for previous reward,
and switch for switch/stay targets. Save both the original labels and this
mapping in result metadata.

### 4.1 Categorical metrics

Report from the same held-out predictions:

- balanced accuracy, the primary metric; and
- ROC AUC, the secondary metric.

Balanced accuracy uses a fixed positive-class probability threshold of 0.5.
ROC AUC uses held-out positive-class probabilities or consistently oriented
decision scores. Metric display changes must not refit or retune a model.

The reference value for both metrics is 0.5. It is a descriptive baseline, not
a significance threshold.

Classifiers use no class weights, no nonuniform sample weights, and no
over/undersampling. Record train/test class counts for every fold.

## 5. Numerical regression targets

All numerical targets use regression and held-out $R^2$. Predictions are not
rounded, including for integer-valued targets.

| Display target | Source column | Meaning and scale |
| --- | --- | --- |
| Consecutive omissions | `consecutive_omissions` | Count of consecutive valid unrewarded animal choices entering trial i, including incorrect unrewarded choices; excludes trial i and ignores manual/no-choice rows. |
| Consecutive rewards | `consecutive_rewards` | Count of consecutive valid positively rewarded animal choices entering trial i; excludes trial i and ignores manual/no-choice rows. |
| Session trial index | `cur_trial` | Existing zero-based session trial index. |
| Trial index in block | `cur_trial_in_block` | Existing zero-based block-relative index. |
| Rewards in block | `rewards_in_block` | Count of valid positive animal rewards earlier in the same block; excludes trial i. |
| Q-learning relative value | `Qlearning_rel_value` | Pre-update left-minus-right value. |
| Forgetting-Q relative value | `FQlearning_rel_value` | Pre-update left-minus-right value. |
| HMM signed belief | `HMM_rel_value_logodds` | Pre-update `tanh(tanh_scale * log odds)`, not raw log odds. |
| HMM-decay signed belief | `HMM_rel_value_logodds_decay` | Pre-update tanh-transformed signed belief, not raw log odds. |
| Relative doubt | `relative_doubt_index` | Pre-update, unitless, left-positive value in [-1, 1]. |

The existing HMM column names are retained for compatibility, but figures and
captions must not call their values raw log odds.

Do not standardize or otherwise transform numerical targets. Fit and score them
in their stored units. Estimator-internal intercept handling does not count as
target standardization.

For each outer test fold:

$$
R^2 = 1 -
\frac{\sum_i (y_i-\hat{y}_i)^2}
{\sum_i (y_i-\bar{y}_{\mathrm{test}})^2}
$$

Negative finite $R^2$ values are valid. A constant-target test fold or a fold
with fewer than two observations has undefined $R^2$ and is unavailable; do
not let a library's finite-value convenience behavior replace it with 0 or 1.

## 6. `rewards_in_block` behavioral feature

Add `rewards_in_block` to the general augmented trial table rather than
creating a decoder-only copy.

Adding it must preserve row count, chronological order, index, and every
existing column.

The value on row i is the number of earlier valid animal-choice trials in the
same `cur_block` with numeric `reward > 0`. Therefore:

- the first row of every block has value 0;
- the current row's reward is not included;
- within an unchanged block, a manual/experimenter reward neither increments
  nor resets the count;
- within an unchanged block, a no-choice row neither increments nor resets the
  count; and
- a block change resets the count before the new row is recorded.

This definition is independent of whether reward is encoded as an integer,
float, or numeric string. Malformed reward values on otherwise valid animal
choices, including non-finite numeric values, are errors during augmentation
rather than silently counted as zero. Nonmissing actions must encode exactly
0 or 1 after numeric conversion; values such as 0.5 are not valid choices.
Missing block identities are augmentation errors because reset semantics would
otherwise be undefined.

## 7. Neural inputs and regional populations

### 7.1 Explicit region configuration

The run configuration maps one probe to PFC and a distinct probe to HPC. Do
not infer brain regions from probe names or site ordering.

For each region, the configuration records:

- probe ID;
- channel-selection rule;
- required channel-quality labels, initially `good`;
- whether `inside_brain` must be true, initially yes; and
- accepted cluster groups, initially `good` and `mua`.

Channel selection starts from the requested channel-quality labels and
`inside_brain` rule, intersects the probe's optional metadata
`unit_channels` restriction when present, and then intersects any additional
explicit channel restriction in the decoding configuration. All such channel
IDs are zero-based saved-channel indices. Save the selected channel IDs and
every contributing rule.

The initial implementation supports one probe per region. General multi-probe
regions are out of scope.

Unit identity is `<probe_id>:<cluster_id>`. Preserve that identifier,
region, cluster ID, channel, and quality in results. Never assume that bare
cluster IDs are unique across probes.

Selected cluster IDs must be unique in the selected metadata and present in
the loaded spike-cluster assignments. Selected aligned spike timestamps must
be finite. These are input-integrity checks, not fold-level missingness.

Selecting zero channels or zero units for either configured region is an input
error and fails before fitting. It is not a scientifically unavailable decoder
cell. Fold-level loss of all usable features after training-only constant
feature checks remains a declared scientific invalidity.

### 7.2 Alignment coverage

Classify coverage mechanically from the aligned archive. If it contains
`irig_utc_unix`, treat it as IRIG-covered and require that array to be
one-dimensional, nonempty, and finite; its minimum and maximum are the trusted
bounds. Configured trusted bounds are invalid for an archive with this member;
do not retain an ignored override or let one hide a malformed IRIG array. If
the member is absent, as in the current manual-alignment output, trusted finite
UTC bounds with `start < end` must be supplied explicitly in the run
configuration or stored in a future authoritative alignment field. Without
them, fail validation. Never use the first and last spike as recording-coverage
bounds.

A trial is eligible only if its entire event-relative window lies within the
trusted bounds for every required probe.

### 7.3 Activity representation

Reuse the existing population rate-tensor behavior:

- output shape `(n_trials, n_time_bins, n_units)`;
- values are unsmoothed firing rates in Hz;
- each value is spike count divided by bin width in seconds;
- time bins are fixed-width and event-relative; and
- a decoder at one time bin receives only that bin's neural features.

No smoothing, neighboring-bin concatenation, or hidden temporal feature is
added.

## 8. Time axis

Supported alignments:

- choice time, source `choice_time`; and
- trial start, source `start_time`.

Both source columns are UTC Unix timestamps in seconds, in the same coordinate
system as aligned spike times and trusted coverage bounds. The decoder performs
no implicit time-unit or clock conversion.

The window is [-2 s, +2 s]. Supported bin widths are 500, 100, 50, and 20 ms;
100 ms is the default.

The interval is divided into complete half-open bins. Record the exact edges
and centers in seconds relative to the event. A target is a fixed trial-level
quantity across the time axis, including post-event bins.

The task design is accepted as preventing overlap between different trials'
selected windows, so no additional overlap exclusion is required.

## 9. Neural representations

The initial implementation computes both representations below for all three
region configurations: PFC, HPC, and PFC + HPC. Region and representation
selectors in the saved-results view are display controls, not computation
controls. Selective computation can be considered later only if benchmarking
shows a concrete need.

### 9.1 Shared temporal regional PCA

PCA is the default representation. Within each relevant training subset and
region:

1. Pool training trial x time-bin observations into a 2D matrix with units as
   columns.
2. Compute one training mean and standard deviation per unit across that pooled
   matrix.
3. Mark zero-variance or unavailable training features explicitly and exclude
   them from the fitted transform; do not label them as penalty-selected zeros.
4. Apply the training-derived standardization unchanged to validation/test
   trials.
5. Fit one regional PCA with `whiten=False`, `svd_solver="auto"`, and the
   configured random seed.
6. Use PC scores directly, without a second PC scaler.

PFC and HPC always receive separate PCA fits. PFC + HPC concatenates the two
regional PC score matrices; it does not fit a joint cross-region PCA.

If a fold has no usable training features in either required component region,
that region's standalone cell and the combined cell are unavailable. The
combined decoder never silently collapses to a one-region decoder; the other
region's standalone cell may still proceed.

Default retained counts are 10 PFC PCs and 10 HPC PCs. Cap a requested count at
the usable training rank. Record requested and effective counts and surface a
visible warning; do not silently change the count.

Within the same target, fold, alignment, bin size, and eligible-trial set, the
regional scaling and PCA basis are shared across time-bin decoders. They are
refit across outer folds and, during tuning, across inner training folds.

### 9.2 Individual units

The direct-unit branch uses the same pooled training trial x time-bin
standardization but no PCA and no per-bin scaler. PFC + HPC concatenates
standardized regional unit columns in recorded stable-ID order.

The unit axis and exclusions must remain inspectable.

## 10. Decoder family and regularization

Use the same family for both representations:

| Target family | Estimator |
| --- | --- |
| Categorical | Elastic-net logistic regression |
| Numerical | Elastic-net linear regression |

Fixed mode is the default:

- logistic regression: `C=1.0`, `l1_ratio=0.5`,
  `solver="saga"`, `tol=1e-4`, `max_iter=100`,
  `fit_intercept=True`, `class_weight=None`, `warm_start=False`, and
  `n_jobs=None`;
- numerical ElasticNet: `alpha=1.0`, `l1_ratio=0.5`,
  `tol=1e-4`, `max_iter=1000`, `fit_intercept=True`,
  `selection="cyclic"`, `positive=False`, and
  `warm_start=False`; and
- random seed 0 where an estimator or PCA solver uses randomness.

These are the inspected scikit-learn 1.8 defaults for the otherwise
unspecified numerical controls. Freeze and record them rather than exposing
additional configuration knobs in the initial implementation. A later change
to one of these controls requires an analysis-version change.

The installed scikit-learn API must be checked during implementation. In the
currently inspected scikit-learn 1.8 environment, logistic `penalty` is
deprecated; use the supported `l1_ratio`/`solver="saga"` interface
without introducing a deprecation warning.

Optional tuned mode uses:

| Parameter | Candidate values |
| --- | --- |
| Logistic `C` | 0.01, 0.1, 1.0, 10.0, 100.0 |
| ElasticNet `alpha` | 0.001, 0.01, 0.1, 1.0, 10.0 |
| `l1_ratio` | 0.1, 0.5, 0.9 |

Do not tune PC count, bin size, class weights, feature-scaling policy, target
scale, or decision threshold. Do not expand the grid automatically.

A converged all-zero coefficient solution is valid. Any convergence warning or
non-finite fit/score makes that fold or tuning candidate invalid; do not
silently weaken regularization or change estimator family.

## 11. Cross-validation

### 11.1 Fold roles

- Outer evaluation: five grouped folds, one deterministic pass, in fixed and
  tuned modes.
- Inner tuning: three grouped folds inside each outer training set, tuned mode
  only.

Behavioral blocks from `cur_block` are indivisible groups. Prefer
multiple blocks in each test fold, but treat this as a goal rather than an
extra hard constraint. Actual categorical train and test partitions must
contain both classes.

Use deterministic `StratifiedGroupKFold(shuffle=False)` for categorical
targets and deterministic `GroupKFold(shuffle=False)` for numerical
targets. Validate the resulting folds rather than assuming the splitter has
satisfied class coverage.

If five valid grouped outer folds cannot be constructed, mark the target
unavailable and report the reason. An explicitly configured three-fold rerun is
allowed. Do not automatically search fold counts/seeds or fall back to random
trial splitting.

If inner folds are invalid, tuned mode is unavailable for that configuration;
do not relabel fixed results as tuned.

### 11.2 Leakage prevention

The outer test rows are excluded from:

- unit scaling and constant-feature decisions;
- PCA fitting and effective-rank decisions;
- inner splits and hyperparameter selection; and
- every other learned preprocessing quantity.

During tuning, all fitted preprocessing is learned separately inside each
inner training split, then the selected settings and preprocessing are refit on
the complete outer training set before one outer-test evaluation.

### 11.3 Aggregation and validity

Calculate metrics separately within each outer test fold. The displayed value
is the unweighted arithmetic mean of the requested fold scores.

Display a primary cell only when every requested outer fold is valid. Otherwise
show the cell as unavailable, preserve valid fold records, and show the
valid/expected fold count and reason. A failed cell must not stop unrelated
targets or time bins when the failure is a declared scientific invalidity:
insufficient folds/classes, a constant target, an unavailable feature set, a
convergence warning, or a non-finite fit/score. Here an unavailable feature set
means no usable features remain in that fold after training-only checks; zero
session-level channels/units is the input error defined in Section 7.1.
Unexpected programming, schema, or I/O exceptions must fail the run, preserve
completed checkpoints, and remain visible rather than being converted into
scientific missingness.

Inner scores select parameters and are never mixed into the displayed outer
score.

## 12. Coefficients and decoder reliance

For a selected target, time bin, and region configuration, display direct-unit
outer-fold coefficients:

- signed coefficient per fold;
- median and IQR across evaluable folds;
- units sorted by mean absolute coefficient across folds;
- selection frequency;
- number and fraction of nonzero coefficients per fit; and
- unit ID, region, contributing-fold count, and any constant/unavailable
  status.

Nonzero means `abs(coef) > 1e-8` on the fitted standardized-feature
scale, excluding intercepts. This is a numerical counting tolerance, not a
biological or statistical threshold.

Retain each fitted intercept in results for model provenance, but do not treat
it as a unit/PC coefficient or include it in nonzero-feature counts.

For classification, orient coefficient signs to the recorded positive class
and label them as log-odds change per pooled training standard deviation. For
regression, label them in native target units per pooled training standard
deviation.

Coefficient IQR describes variation across CV fits, not a confidence interval.
Weights describe decoder reliance, not causal contribution or the number of
neurons encoding the variable. PC coefficients refer to PC features and must
not be presented as unit selection.

## 13. Results and visualization

The computation runs offline and saves versioned results under the session data
directory. The existing Streamlit webapp provides one integrated, read-only
task-variable results view.

The webapp must:

- discover completed saved runs for the selected session;
- load result metadata and arrays without loading raw spike data;
- never fit or tune a decoder in response to a widget change;
- show categorical and numerical heatmaps separately;
- allow balanced-accuracy/AUC display changes from the same saved predictions
  or fold metrics;
- expose trials, fold coverage, settings, and failure reasons; and
- show coefficient summaries for selected direct-unit results.

All initial plots use light mode: opaque white background with black axes and
text. Export PNG by default. Every figure has labels and a concise scientific
caption. Categorical heatmaps show a 0.5 reference; numerical heatmaps show 0.
Preserve below-reference classifications, negative $R^2$, and a distinct
appearance for unavailable cells.

Run configuration controls are target subset, bin width, alignment, regional
PC counts, regularization mode, and outer/inner fold counts. Saved-results
display controls are the session-relative results root, run, representation,
region, target, and categorical metric. The results-root locator defaults to
`analysis_runs`, must remain inside the selected metadata session, and only
changes where completed runs are listed. Display controls must never cause
computation; changing a run setting requires an offline run rather than an
interactive refit.

## 14. Reproducibility and performance

Every saved run records:

- analysis version;
- Python and numerical-library versions;
- complete configuration and feature-generation parameters;
- random seed;
- source paths and stable file identities;
- an exact copy of `trial_feature_params.json`;
- unit identities and selection rules;
- trial/block identities, the common neural-tensor row mapping, encoded target
  values, target mappings, eligibility counts/reasons, outer assignments, and
  tuned-mode inner assignments/candidate scores;
- array shapes, axis names, physical units, and time-bin edges;
- estimator settings and selected hyperparameters;
- warnings, invalid reasons, and requested/effective PC counts; and
- stage and total runtimes.

Changing code version, scientific parameters, or inputs creates a new run and
must not overwrite prior results.

A prepared, detached, queued, or resumed execution must verify its saved input
identities and recorded Python/numerical-library versions again before loading
large arrays. Changed inputs or runtime versions must fail visibly rather than
run under stale provenance.

The implementation must reuse rate tensors, fold assignments, and fold-level
regional transforms where the scientific inputs are identical. Clarity takes
priority over micro-optimization.

Benchmarking is required before recommending routine use. Use synthetic data
first, then CT026 2026-08-03. Measure loading, binning, preprocessing, model
fitting, saving/plotting, total wall time, model-fit count, and peak memory.
Benchmark the default fixed analysis before optional tuned mode. Tuned mode has
a much larger fit count and must be estimated from a bounded representative
subset before any full-session tuned run.

The local benchmark used to project Slurm resources must use the same explicit
single-thread OpenMP, MKL, and OpenBLAS limits as the initial one-CPU Slurm
job. Record those limits with the benchmark so runtime comparisons are not
confounded by implicit library threading.

## 15. Interpretation limits

- Regional results compare the recorded populations supplied; they are not
  normalized information per neuron.
- Separate decoding of context, choice, belief, and doubt does not establish
  unique or independent representations.
- Trial-index decoding can reflect slow drift.
- Shared temporal PCA intentionally learns a common offline basis from the
  selected training window, including post-event training activity.
- Reference values are not statistical significance thresholds.

## 16. Revision-5 corrections

Relative to revision 4, revision 5:

- records the completed codebase audit instead of leaving source verification
  unresolved;
- defines the augmented-table and feature-parameter inputs explicitly;
- adds the general `rewards_in_block` column and its entering-trial
  semantics;
- recognizes dark state `2` and restricts current-state decoding to
  the two choice-bearing contexts;
- labels HMM outputs as tanh-transformed signed beliefs rather than raw log
  odds;
- defines baseline and target-specific eligibility, including manual and
  no-choice handling;
- validates chronological trial/block identity and uses a common two-probe
  coverage intersection for matched regional comparisons;
- requires explicit PFC/HPC configuration and stable probe-qualified unit IDs;
- defines IRIG and manual-alignment coverage checks;
- makes requested/effective PC capping and convergence failures visible;
- computes all six initial region/representation combinations and makes their
  selectors display-only;
- freezes the remaining estimator/PCA numerical controls;
- distinguishes declared scientific invalidity from unexpected run failures;
- uses an integrated read-only view in the existing webapp;
- records the installed scikit-learn 1.8 logistic-interface consideration;
- adds reproducible saved-run and like-for-like single-thread benchmark
  requirements, including pre-execution input revalidation; and
- retains the accepted revision-4 estimator, CV, timing, and visualization
  decisions unless explicitly corrected above.

Revision 4 remains unchanged as historical input.

## 17. Technical references

- **R1:** scikit-learn ElasticNet:
  `https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html`
- **R2:** scikit-learn LogisticRegression:
  `https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html`
- **R3:** scikit-learn PCA and StandardScaler:
  `https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html` and
  `https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html`
- **R4:** scikit-learn balanced accuracy:
  `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html`
- **R5:** scikit-learn $R^2$:
  `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html`
- **R6:** scikit-learn grouped splitters:
  `https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data`
