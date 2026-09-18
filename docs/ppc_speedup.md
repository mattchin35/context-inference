# Exact PPC Computation Speedup: Technical Rationale and Design

## 1. Purpose

This document describes why the current spike-LFP pairwise phase consistency
(PPC) computation is too slow, which computational changes are expected to
improve it, and why those changes can preserve the approved scientific result.
It is intended to be understandable without access to the conversation that
motivated it.

This is a computational design document, not an implementation plan. It does
not assign files or agents, define a commit sequence, prescribe RED/GREEN test
commands, or authorize a CT026 Spike-phase run. Those details belong in a later
implementation plan after the design has been reviewed.

The central constraint is that performance work must preserve the existing
scientific estimator and null model. The goal is to reorganize exact work,
remove repeated work, and expose safe parallel work. It is not to replace PPC
with another statistic or replace the trial-shuffle inference with an easier
test.

## 2. Scope and non-goals

The changes described here apply to the frequency-resolved unit-level
spike-LFP PPC calculation and its same-condition trial-shuffle null. They cover:

- Sampling event-relative complex phase at spike times.
- Reducing sampled phase immediately to complex vector sums and valid-spike
  counts.
- Constructing observed and shuffled PPC from those sufficient statistics.
- Reusing results across epochs and overlapping conditions when the reuse is
  mathematically exact.
- Keeping temporary memory bounded.
- Retaining the option to execute the same calculation on a more capable
  single cluster node later.

The following are explicitly out of scope for this speedup design:

- Changing the PPC formula.
- Weighting phase samples by wavelet amplitude.
- Replacing linear complex interpolation with nearest-neighbor phase lookup.
- Replacing same-condition trial derangements with temporal jitter, circular
  shifts, a Rayleigh test, or another null.
- Reducing the frequency grid, changing the epochs, or dropping configured
  conditions merely to reduce runtime.
- Using the 100-shuffle preview as the final inferential result.
- Designing a distributed multi-node reduction system. A cluster may later be
  used simply as a more powerful and interruption-resistant execution host.

## 3. Scientific computation that must remain unchanged

### 3.1 Observed PPC

For valid spike phases `phi_k` at one unit, condition, site, epoch, and
frequency, define the complex unit-vector sum and sample count as:

```text
S = sum(exp(1j * phi_k))
N = number of valid spike-phase samples
```

The unweighted PPC estimator is:

```text
PPC = (abs(S)**2 - N) / (N * (N - 1))
```

PPC is unavailable when `N < 2`. Negative PPC values are valid and must be
preserved. Observed PPC remains available when it is computable even if it is
not eligible for permutation inference.

The formula above is equivalent to the mean pairwise phase agreement over all
distinct spike pairs. The sufficient statistics `S` and `N` therefore retain
all information required to calculate unweighted PPC; individual sampled phase
vectors do not need to survive the reduction.

### 3.2 Phase interpolation and validity

The approved phase sampling rule is unchanged:

- Phase is represented by complex Morlet coefficients on the exact common
  event-relative 500 Hz grid.
- Real and imaginary components are interpolated linearly. Wrapped phase
  angles are never interpolated directly.
- A spike exactly on a grid sample uses that exact coefficient.
- A spike between samples is valid only when both adjacent coefficients are
  finite and have nonzero magnitude.
- A spike outside the supported interval is invalid.
- An interpolated complex value must be finite and nonzero before it is
  normalized to unit magnitude.
- The existing complex64-before-complex128-sum behavior is part of the
  numerical reference and should be preserved unless a separately reviewed
  change establishes numerical equivalence.

### 3.3 Trial-shuffle inference

The null model remains a same-condition trial derangement. For every shuffle,
spikes from each source trial are paired with phase from a different target
trial. No source trial may map to itself.

The schedule is deterministic for the saved seed and is shared across units
within the same scientific job. The following remain unchanged:

- The minimum of 50 valid spike-phase samples for inference.
- The requirement for at least two spike-contributing eligible trials.
- The plus-one permutation p-value formula.
- Population null standard deviation with `ddof=0`.
- Explicit linear 2.5th, 50th, and 97.5th percentiles.
- Benjamini-Hochberg correction across frequency separately for every
  unit-by-condition-by-epoch-by-site spectrum.
- The meaning of the 100-shuffle preview and 1,000-shuffle final setting.

Execution block sizes and worker counts may change runtime but must not change
these scientific results.

## 4. Evidence motivating the redesign

The production-representative CT026 serial profile is stored outside the
repository at:

```text
CT026/CT026_20260801_latent_inference/analysis_runs/
    ct026_ppc_profile_2026-09-09T21-33-06Z/
```

On the current workstation, its principal measurements were:

- Cold phase preparation: 136.12 seconds.
- Warm memory-mapped phase reuse: 7.87 seconds.
- Prepared phase work storage: approximately 1.1 GiB.
- Phase-preparation process high-water RSS: approximately 4.10 GB.
- Representative PPC child-process high-water RSS: approximately 2.33 GB.
- Profile job: condition `incorrect`, site `PFC`, whole epoch `[-2, 2)`
  seconds, 249 trials, 50 frequencies, and 100 shuffles.
- Low-rate unit: 20.90 seconds for 293 spikes.
- Median-rate unit: 21.87 seconds for 2,325 spikes.
- High-rate unit: 23.83 seconds for 15,676 spikes.
- Combined three-unit job: 66.80 seconds.
- Combined observed reduction: 26.57 seconds.
- Combined shuffled-edge stage: 39.69 seconds, including 39.39 seconds in
  edge reduction.
- Additional shuffle aggregation beyond edge reduction: only about 0.30
  seconds.
- Unique scheduled source-target edges at 100 shuffles: 20,556.

The nearly flat per-unit timing across a roughly 53-fold range of spike counts
shows that fixed repeated array work and per-edge dispatch dominate much of the
current runtime. The difference between the inclusive shuffle stage and edge
reduction also shows that permutation bookkeeping is not the primary
bottleneck. Optimizing p-value summarization or shuffle accumulation first
would therefore address very little of the measured cost.

The existing conservative projections are:

- Largest single 249-trial condition/site/epoch job:
  approximately 98-106 minutes at 100 shuffles and 3.6-3.9 hours at 1,000
  shuffles.
- Complete nine-condition, three-site, three-epoch component:
  approximately 47-51 hours at 100 shuffles and 90-100 hours at 1,000
  shuffles.

The 47-51 and 90-100 hour numbers describe the complete component, not one
condition/site/epoch job. They are intentionally conservative extrapolations,
not measured end-to-end component runtimes.

## 5. Remove redundant observed-phase work

### 5.1 Current inefficiency

The current observed path calculates pooled sums and counts, then reevaluates
every trial separately to determine whether that trial contributed at least one
valid spike. During these calls it repeatedly constructs a masked phase array
equivalent to `where(valid, phase, 0j)`.

For the 249-trial profile job, this means that a large trial-by-frequency-by-time
phase representation can be revisited and copied many times for one unit even
though the necessary information is already present in the first edge
reduction. This explains the substantial fixed observed cost and its weak
dependence on spike count.

### 5.2 Exact replacement logic

For every same-trial observed edge, retain the already-computed:

- Complex vector sum by unit and frequency.
- Valid spike count by unit and frequency.
- Whether the edge count is greater than zero.

The pooled observed sum and count are the sums across those trial edges. The
eligible-trial count is the number of same-trial edges with a positive valid
spike count. All three quantities can therefore be obtained from the same
bounded pass.

The masked complex phase representation should be formed once at the narrowest
scope that safely permits reuse, not once per trial or repeatedly inside helper
calls. This change does not alter interpolation, validity, spike membership,
axis order, counts, or the PPC formula. It removes duplicate construction and
duplicate edge evaluation only.

### 5.3 Expected impact

In the three-unit 100-shuffle profile, observed reduction accounted for about
40 percent of total elapsed time. That percentage is an upper bound on the
benefit because not every observed operation can be eliminated. It nevertheless
identifies this change as a meaningful and low-scientific-risk optimization.
Its relative benefit will be smaller at 1,000 shuffles, where shuffled-edge
work occupies a larger fraction of the run.

## 6. Direct indexing on the uniform 500 Hz grid

### 6.1 Current inefficiency

The phase time grid is uniform, but the current general interpolation helper
performs repeated searches and calls a generic one-dimensional interpolator
separately for the real and imaginary parts of every frequency row. Those
operations are repeated for each source-target trial edge even though a source
trial's spike times do not change when it is paired with different target
trials.

The expensive repeated work includes:

- Locating neighboring samples for the same source-trial spike times.
- Rechecking time-axis geometry.
- Looping in Python over frequency rows.
- Calling generic interpolation twice per frequency and edge.
- Reconstructing validity support that depends on already-known neighboring
  indices.

### 6.2 Interpolation geometry

For a uniform grid with first time `t0`, step `dt = 1 / 500` seconds, and a
spike time `t`, the interpolation geometry can be represented by:

```text
grid_coordinate = (t - t0) / dt
left_index       = floor(grid_coordinate)
right_index      = left_index + 1
right_weight     = grid_coordinate - left_index
left_weight      = 1 - right_weight
```

Exact-sample cases require explicit treatment so they use one exact sample and
do not incorrectly require two valid neighbors. Floating-point grid-coordinate
classification must use the canonical stored time coordinate and a documented
tolerance or an equivalent robust rule; it must not classify arbitrary nearby
spikes as exact samples.

For a valid between-sample spike, interpolation across all frequencies can be
performed as a vectorized gather:

```text
z = left_weight * coefficient[:, left_index]
  + right_weight * coefficient[:, right_index]
```

The real and imaginary meaning is identical to separately interpolating the
two components. The combined complex value is then normalized to unit
magnitude under the approved validity rule.

### 6.3 Reusable source-spike geometry

Neighbor indices, exact/between/outside-support classification, and weights
depend on:

- The common phase time grid.
- The source trial's spike times.

They do not depend on:

- The target phase trial.
- Frequency.
- Condition labels after the physical source trial is known.
- Shuffle position.

This geometry should therefore be computed once for each unit/source-trial
spike train and reused for every target trial requested by observed or shuffled
work. Target-specific coefficient validity must still be evaluated for every
target trial and frequency because phase support can differ across targets.

### 6.4 Numerical equivalence standard

The optimized interpolation is required to be numerically equivalent to the
current approved reference, rather than bit-for-bit identical under every
floating-point operation order.

The equivalence contract should distinguish:

- Values that must remain exact: trial identities, edge identities, schedules,
  valid/invalid classifications, spike counts, contributing-trial counts,
  eligibility flags, permutation counts, and array axes.
- Floating-point values that may differ only within a tight declared tolerance:
  interpolated unit vectors, complex sums, PPC, preferred phase, resultant
  length, null moments and percentiles, p-values, and q-values.

A later implementation plan must define tolerances based on dtype and
accumulation order. Tolerances must be small enough that they cannot conceal a
changed support rule, boundary error, missing spike, or different derangement.
Significance flags near the FDR threshold require explicit comparison and must
not be allowed to change silently because of a loose tolerance.

## 7. Derive whole-epoch PPC from additive sufficient statistics

### 7.1 What is and is not additive

PPC values are not additive. It would be incorrect to add or average
`PPC_before` and `PPC_after`.

The complex vector sums and valid counts used to calculate PPC are additive.
For disjoint before and after spike sets:

```text
S_whole = S_before + S_after
N_whole = N_before + N_after
```

Whole-epoch PPC is then calculated normally:

```text
PPC_whole = (abs(S_whole)**2 - N_whole)
            / (N_whole * (N_whole - 1))
```

Expanding the squared magnitude gives:

```text
abs(S_before + S_after)**2
    = abs(S_before)**2
    + abs(S_after)**2
    + 2 * real(S_before * conjugate(S_after))
```

The final term captures all cross-before/after spike pairs. Averaging the two
half-epoch PPC values omits those cross pairs and is scientifically wrong.
Adding the sufficient statistics before applying the nonlinear PPC formula
includes them exactly and is equivalent to directly pooling all whole-epoch
spike phases.

This behavior is scientifically meaningful. For example, if a unit is tightly
locked before and after choice but its preferred phase reverses across choice,
the cross term reduces whole-window PPC. The composed sufficient-statistic
calculation preserves that reduction exactly; averaging the two high
half-window PPC values would hide it.

### 7.2 Half-open epoch boundary

The existing half-open epochs remain binding:

```text
before = [-2.0, 0.0) seconds
after  = [ 0.0, 2.0) seconds
whole  = [-2.0, 2.0) seconds
```

The sample or spike at exactly zero belongs only to `after`. Every eligible
whole-window spike must therefore appear in exactly one half. No spike may be
lost or counted twice during composition.

### 7.3 Observed whole result

For each unit, trial edge, site, and frequency, compute the before and after
sums/counts under the same phase-validity rule. Add them at the sufficient-
statistic level and then reduce across trials to form the observed whole result.

The following whole outputs can be derived from the combined sum and count:

- PPC.
- Resultant length.
- Preferred phase.
- Valid spike count.
- Computability and reliability flags.

Whole contributing-trial count is the number of physical trials with a
positive combined before-plus-after count. It is not necessarily the sum of
the two half-epoch contributing-trial counts because the same trial may
contribute spikes to both halves.

### 7.4 Shuffled whole result

For a source-target trial edge `e`, retain:

```text
S_e_before, N_e_before
S_e_after,  N_e_after
```

Then:

```text
S_e_whole = S_e_before + S_e_after
N_e_whole = N_e_before + N_e_after
```

For shuffle row `r`, with derangement `pi_r` mapping each source trial to a
target trial:

```text
S_r_whole = sum_over_source(S_(source, pi_r(source))_whole)
N_r_whole = sum_over_source(N_(source, pi_r(source))_whole)
```

Applying the PPC formula to `S_r_whole` and `N_r_whole` is exactly the shuffled
whole-window PPC draw that direct whole-window phase sampling would produce.

Before, after, and whole composition must use the same physical source and
target trial identities and the same derangement row. Independently generating
different before and after schedules and then combining their results would
not reproduce a whole-window shuffle.

### 7.5 Validity and eligibility constraints

This optimization cannot be implemented by loading already summarized
standalone before and after products and adding them. Composition must happen
inside the sufficient-statistic computation while identities and validity are
still explicit.

In particular:

- The whole job's trial inclusion policy must govern both halves used for the
  whole result. If standalone half-epoch jobs admit different trials, their
  final products cannot be combined into whole.
- Sample validity remains site-, trial-, frequency-, and time-specific.
- A unit may have fewer than 50 spikes in each half but at least 50 in the
  whole interval. Half-edge work required by a whole-eligible unit must not be
  skipped merely because neither half is independently inference-eligible.
- A physical trial contributing spikes to either half counts once toward the
  whole contributing-trial count.
- P-values, q-values, null percentiles, null standard deviations, and PPC
  values cannot be composed from half summaries. They must be recomputed from
  the combined sufficient statistics or whole shuffle draws.

Under these constraints, whole-from-halves is a computational reorganization,
not a change in the scientific estimator.

## 8. Reuse edge statistics across overlapping conditions

### 8.1 Why reuse is valid

An edge sufficient statistic is determined by:

- Stable unit identity.
- Stable LFP site identity.
- Epoch or half-epoch bounds.
- Stable physical source-trial identity and its spike times.
- Stable physical target-trial identity and its phase coefficients.
- Frequency and phase-validity rules.

It is not determined by the name of a behavioral condition. Conditions decide
which physical trials are eligible and which source-target edges a
condition-specific derangement schedule requests. If two conditions request
the same physical edge, the phase interpolation and reduced sum/count for that
edge are identical and may be reused.

This matters because the nine required conditions overlap:

- `correct_rewarded`
- `omission`
- `incorrect`
- `switch`
- `stay`
- `omission_switch`
- `omission_stay`
- `incorrect_switch`
- `incorrect_stay`

The intersection conditions reuse physical trials contained in broader
conditions. At 1,000 shuffles, a large condition's schedule may approach the
complete off-diagonal edge table, increasing the chance that edges requested
by nested conditions have already been computed.

### 8.2 Stable global edge identity

Condition-local trial positions cannot identify reusable edges because position
zero in one condition need not refer to position zero in another. Reuse must be
keyed by stable trial-table row identities:

```text
(source_trial_index, target_trial_index)
```

Each condition-specific schedule can retain its compact local positions, but
its requested edges must be translated to stable physical identities before
forming a union or looking up shared statistics.

The schedule itself remains condition-specific. Sharing an edge statistic does
not mean sharing, enlarging, or otherwise changing a condition's trial pool or
derangements.

### 8.3 Bounded union and consumption

Persisting every edge statistic for every unit, site, and epoch would require
too much storage. For 249 trials there are 61,752 possible directed nonself
edges. At complex128 sums plus int64 counts, a complete edge table for eight
units and 50 frequencies is approximately 590 MB before metadata or temporary
arrays. Retaining such tables for the full population, all sites, and multiple
epochs would replace a runtime problem with a storage and memory problem.

Reuse should therefore be bounded. Conceptually:

- Work on a bounded unit block and one site at a time.
- Determine the union of physical edges needed by the applicable condition
  schedules.
- Process that union in bounded source/edge blocks.
- Apply each computed edge block immediately to the affected condition and
  shuffle accumulators.
- Discard the edge block after all consumers have incorporated it.

The full edge table should be materialized only when an explicit cost model
shows that it is both faster and within the configured memory bound. The 100-
shuffle preview may favor scheduled-edge mode, while the 1,000-shuffle run may
favor a complete or nearly complete off-diagonal table. Both modes must remain
numerically equivalent for the same schedules.

### 8.4 Schedule-union measurement

The likely benefit should not be guessed from condition names alone. Before
expensive phase sampling, the program can cheaply compute from condition
memberships and deterministic schedules:

- Edge count requested independently by each condition.
- Union edge count across all conditions.
- Reuse ratio for 100 and 1,000 shuffles.
- Scheduled-edge saturation relative to the possible off-diagonal table.
- Expected temporary bytes for each candidate unit and edge block size.

These metadata-only measurements will show whether cross-condition reuse is a
minor optimization or one of the dominant savings for CT026.

## 9. Interaction among the changes

The optimizations reinforce each other but must remain conceptually separate:

- Removing redundant observed work prevents repeated copies and repeated
  same-trial sampling.
- Direct grid indexing reduces the cost of every remaining physical edge.
- Reusable source-spike geometry prevents repeated neighbor searches when one
  source trial is paired with many targets.
- Half-epoch sufficient statistics prevent a third independent pass for the
  whole epoch.
- Cross-condition edge reuse prevents the same physical edge from being
  interpolated for multiple overlapping condition schedules.

The intended computational unit becomes a physical source-target edge in a
bounded unit/site/half-epoch block. Observed, shuffled, whole-epoch, and
condition-specific outputs become consumers of that exact reduced information
rather than independent reasons to resample the same spike phases.

The transformations should not be assumed to multiply their speedups
independently. For example, removing whole-epoch sampling reduces the amount of
work available for condition reuse, and a faster interpolation kernel reduces
the fraction attributable to repeated observed setup. Only measurements after
each accepted transformation can establish the combined speedup.

## 10. Memory and data-movement considerations

Performance must be evaluated in terms of both arithmetic and data movement.
The prepared phase representation is approximately 1.1 GiB for all three CT026
sites, 50 frequencies, 427 trials, and 2,000 time samples. Repeatedly copying
large masks or scanning unrelated trials can dominate a computation even when
the PPC formula itself is inexpensive.

The redesigned computation should preserve these principles:

- The prepared phase arrays remain read-only and memory-mappable.
- No complete per-spike phase tensor is retained.
- No full-session shuffle-by-unit-by-frequency tensor is retained.
- Complex sums use complex128 and counts use int64.
- Temporary edge arrays are bounded by explicit edge and unit blocks.
- Shuffle draws are retained only for the current bounded unit block when exact
  percentiles require them.
- Multiple worker processes must share prepared phase pages rather than receive
  serialized phase-array copies.
- Aggregate parallel memory must be measured directly. Per-process maximum RSS
  cannot be multiplied blindly because memory-mapped pages may be shared, but
  it also cannot be treated as free memory.

The existing profile's 2.33 GB child high-water RSS is not an aggregate
parallel-memory measurement. Any future worker-count decision needs total
system or proportional-set-size measurements in addition to individual worker
RSS.

## 11. Parallel execution after serial work is efficient

Unit blocks are scientifically independent after the deterministic schedule
and read-only phase data are fixed. They are therefore suitable for process
parallelism. The S8 benchmark used 64 rate-stratified units in eight default
blocks so every requested 1/2/4/8-worker configuration had useful work. This
superseded the earlier three-unit profile, which fit in one block and could not
measure worker scaling.

Parallelism was activated only after the redundant serial work was removed.
Measured speedup versus one grouped worker was 1.741x, 2.797x, and 3.883x for
2, 4, and 8 workers. Eight workers were selected under the predeclared
near-best-throughput and memory rule. Lower parallel efficiency at eight
workers is accepted because it still produced the best wall time within the
memory budget.

Worker-count invariance remains required. The same scientific configuration,
schedule, and inputs must produce exact discrete outputs and numerically
equivalent floating-point outputs for one, two, four, and eight workers. Parent-
only checkpointing and final manifest-last publication remain unchanged.

## 12. Expected impact and uncertainty

The measured profile supports several qualitative expectations:

- Removing redundant observed work should materially improve 100-shuffle
  runtime because observed reduction was about 40 percent of the representative
  total. It will have a smaller proportional effect at 1,000 shuffles.
- Deriving whole from half-epoch sufficient statistics can remove close to one
  third of epoch-level phase sampling when the whole/before/after jobs share
  the required trial and schedule identities.
- Direct uniform-grid indexing may be the largest per-edge speedup because it
  removes repeated generic interpolation and Python frequency loops, but its
  magnitude is not yet measured.
- Cross-condition edge reuse may be especially valuable at 1,000 shuffles,
  but its value depends on the actual union of scheduled physical edges.
- Parallel unit blocks should reduce wall time after the serial kernel is
  efficient, but speedup will eventually plateau because of memory bandwidth,
  process overhead, and heterogeneous local CPU cores.

It is plausible that the combined exact optimizations will change the decision
about whether the full component requires a cluster. No numerical speedup
factor should be promised before representative measurements. In particular,
the existing 47-51 and 90-100 hour projections should not be reused after the
kernel changes without remeasurement.

### 12.1 Measured completion evidence

The redesign was implemented in S0-S7 and measured on CT026 in the authorized
work-only S8 benchmark on 2026-09-18. The benchmark used 64 rate-stratified
units in eight default blocks, 249 trials, condition `incorrect`, site `PFC`,
all three epochs, 50 frequencies, and 100 shuffles. It created no scientific
component or manifest.

Median 1/2/4/8-worker wall times were 820.833, 471.556, 293.419, and 211.397
seconds. Corresponding median scheduled-edge throughputs were 91.005, 158.412,
254.585, and 353.363 edges/s, and median aggregate PSS values were 1.718,
2.291, 2.646, and 3.352 GiB. Eight workers were selected under the approved
near-best-throughput and less-than-16-GiB rule. This makes eight workers the
preferred CT026 production configuration, while the portable library default
and exact allocation preflight remain unchanged.

Legacy overlap and worker-count invariance passed. Exact identities, schedules,
counts, masks, p-values, decisions, and representative histograms were
preserved. The only noteworthy binary difference was 51 legacy q-value cells
at at most `1.1102230246251565e-16` absolute error, with no tolerance-level
mismatch or changed significance decision.

The benchmark workload has exact scheduled/independent/site-qualified-union
edge counts of 74,700/61,586/43,301 at 100 shuffles and
747,000/182,056/61,752 at 1,000 shuffles. Combining those plans with inclusive
100-shuffle timing gives a 26.4-35.2 minute engineering range for the same
64-unit workload at 1,000 shuffles. It is not a complete-session forecast.
Exact full 427-trial/273-unit plan construction did not finish within the
bounded 10-minute 100-shuffle or 30-minute 1,000-shuffle attempts, so no linear
edge extrapolation was substituted. Full-component planning time remains an
operational risk to measure during the separate 100-shuffle preview.

The accepted evidence is retained at
`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_ppc_s8_2026-09-18T15-06-33Z`.
The historical 66.80-second three-unit result remains a reference for the old
single-job workload and is not a direct speedup denominator for the new
three-epoch grouped benchmark.

## 13. Later single-node cluster execution

If the optimized calculation remains inconvenient locally, the intended
cluster use is simple offloading, not a distributed scientific algorithm. A
SLURM or shell wrapper can request one sufficiently capable node, set the
session and output paths, and call the same directly executable Python
pipeline. The resulting validated cache can then be loaded locally.

Path identity needs explicit treatment before that workflow is reliable. The
current scientific/cache fingerprints include resolved source paths. If the
same files have different paths on the cluster and workstation, a remotely
computed component may appear stale locally even when file contents and
metadata match. A later cluster design must use one of:

- The same mounted absolute paths on both systems.
- A documented local-to-cluster path mapping excluded from scientific identity.
- Stable logical source identifiers with separately validated physical paths.

The output must still retain sufficient provenance to identify the actual
files used. This portability concern is independent of the PPC numerical
speedups and should not complicate the serial kernel.

## 14. Required invariants for any later implementation

Any implementation derived from this design must preserve all of the following:

- Stable probe-qualified unit identifiers.
- Stable trial-table row identities.
- Separate trial-local spike membership, including overlapping windows.
- The exact half-open epoch definitions.
- The approved complex interpolation support rule.
- Unweighted frequency-resolved PPC and preservation of negative values.
- Exact valid-spike and contributing-trial counts.
- Condition-specific deterministic derangements with no self-pairing.
- Observed results for computable but inference-ineligible entries.
- The 50-spike and two-trial inference requirements.
- The plus-one p-value, explicit null summaries, and frequency-wise BH FDR.
- Explicit NaN missingness rather than undocumented zeros or epsilons.
- Bounded temporary memory and no retained full-session null tensor.
- Restartable work artifacts that cannot be mistaken for final components.
- Final component replacement followed by manifest replacement as the commit
  point.
- Numerical equivalence to the approved reference under a declared strict
  tolerance, with exact agreement for identities, counts, masks, schedules,
  and categorical flags except where a separately reviewed near-threshold
  floating-point policy applies.

## 15. Decisions to carry into the later implementation plan

The following design conclusions are established here:

1. Optimize the exact estimator rather than changing the scientific test.
2. Remove redundant observed trial evaluation and repeated masked-phase copies.
3. Use direct vectorized interpolation on the validated uniform grid and reuse
   source-spike interpolation geometry.
4. Construct whole-epoch PPC from additive complex sums and counts, never from
   half-epoch PPC values or summaries.
5. Reuse identical physical edge statistics across overlapping conditions in
   bounded blocks.
6. Use numerical equivalence rather than universal bit-for-bit floating-point
   identity, while requiring exact identities, masks, counts, and schedules.
7. Re-profile the efficient serial kernel before selecting worker count or
   deciding that cluster execution is necessary.
8. Treat later cluster execution as single-node offloading unless measurements
   establish a need for a more complex design.

The later implementation plan must convert these conclusions into bounded
test-first work packages, define equivalence tolerances and performance
measurements, identify the smallest affected public/internal interfaces, and
obtain approval before source implementation begins.
