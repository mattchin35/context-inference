# Exploratory PFC-HPC Communication Subspace Inspection

Status: Scientific scope and implementation approved on 2026-11-16.
Implemented and validated on synthetic data. The user-designated CT026
2026-08-28 run completed at 100 ms; user inspection of its outputs is pending.

## 1. Purpose and interpretation

For one loaded webapp session, compare held-out prediction of population spike
activity within and across PFC and HPC using unrestricted ridge regression and
ridge-regularized reduced-rank regression (RRR). Plot held-out R-squared against
coefficient rank to inspect whether relatively few predictive dimensions recover
the performance of unrestricted regression, and whether this differs within
versus across regions.

This is an exploratory inspection, not a publication-level replication. Raw
activity retains task responses, behavioral modulation, and shared inputs.
Low-dimensional prediction alone does not establish causal communication,
anatomical directionality, or a task-independent communication subspace.
Arrows denote prediction direction at zero lag, not causal influence.

No automatic minimum-rank estimate, equivalence test, significance test, or claim
that a rank achieves the same performance is included in v1. Show absolute R2,
including negative values; do not normalize curves to the full model's score.

## 2. Confirmed scope

- One loaded session, with usable synchronized spike data from both PFC and HPC.
- Only manually labeled `good` units; no `mua` or unlabeled-unit fallback.
- All four comparisons: PFC -> PFC, PFC -> HPC, HPC -> HPC, HPC -> PFC.
- Random, count-matched source and target populations, five seeded draws.
- Within-region source and target populations are disjoint.
- Choice-aligned `[-2,+2)` second windows, with selectable 100 or 500 ms bins.
- One model per comparison pools all bins across included choice windows.
- Simultaneous source and target bins; no lags or spike-history predictors.
- Raw, unsmoothed spike counts; no PSTH subtraction or condition residualization.
- All completed ordinary choices, pooled across contexts, choices, and outcomes.
- Training-only per-neuron z-scoring and mean per-neuron held-out R2.
- Five chronological outer validation folds of whole trials, with protection
  against overlapping train/test windows; training-only inner ridge tuning.
- Ridge regularization for both unrestricted and reduced-rank models.
- Dedicated Communication subspace view, explicit Run/Recompute, saved results,
  two source-direction panels, unit/sample diagnostics, and light plots.
- No batch analysis, cross-session inference, or experimental-data run in this
  specification phase.

## 3. Relation to Semedo et al. (2019)

Reference: Semedo et al., *Cortical Areas Interact through a Communication
Subspace*, Neuron 102, 249-259.e1-e4, DOI 10.1016/j.neuron.2019.01.026.
Reviewed source: `/home/matt/Downloads/papers/semedo_neuron_2019.pdf`, especially
STAR Methods, Data preprocessing and Regression models.

The paper predicts both within-region and cross-region targets from the same
source population. It matches target neuron counts and firing-rate distributions,
subtracts condition-specific PSTHs, and repeats population selection 25 times.
Its full regression model is ridge, tuned with cross-validation; its stated RRR
solution is OLS-based, with rank providing regularization. It uses nested
cross-validation for full-model performance and selects RRR dimensionality using
a one-standard-error rule.

V1 here deliberately differs: both prediction directions, symmetric neuron-count
matching without firing-rate matching, raw activity, five population draws,
five outer folds, ridge-regularized RRR, and no automatic rank selection.
Do not describe this as an exact reproduction of the paper.

## 4. Inputs, eligibility, and shared observations

Reuse existing session metadata, PFC/HPC role resolution, synchronized spike
loading, channel restrictions, unit labels, and choice-time definitions. Verify
the current loader contracts before implementation. Do not duplicate loaders or
change their public interfaces. Preserve probe-qualified unit identity because
cluster numbers may repeat across probes.

Use only `good` units on eligible inside-brain channels under the existing
metadata/channel-selection rules. Persist the effective selection and excluded
counts. Missing usable labels, unresolved region assignments, unavailable spike
sources, or insufficient common recording coverage must produce a specific
unavailable reason. Never substitute one region or pad absent observations.

Use the existing completed ordinary choice table, including incorrect and
unrewarded choices. Preserve original trial IDs, row positions, action, context
and outcome when available, and synchronized choice timestamps. Optional
behavioral annotations do not become mandatory inputs. Do not add outcome
balancing, condition stratification, or a firing-rate cutoff in v1.

Retain only choice windows completely supported by trusted recording coverage
in both regions. Pair by trial identity and exact bin edges, not by array length.
Use the same eligible choice pool for every comparison and population draw.

Count spikes in half-open bins. The 100 ms option produces 40 bins per window;
the 500 ms option produces 8. No smoothing, interpolation, time warping, or
resampling. Bin width is a run parameter, not a post-fit display setting.

Input counts have shape `(choice, bin, unit)` in spikes per bin. Flattening for
fitting yields `(observation, unit)` in choice-major, then bin-major order.
Retain the observation-to-choice/bin mapping. Choice-relative bin edges are in
seconds; source timestamps retain UTC seconds and session-time origin metadata.

## 5. Population matching and repeated draws

Let `N_PFC` and `N_HPC` be eligible good-unit counts. Define
`n = floor(min(N_PFC, N_HPC) / 2)`.

For each of five draws, independently permute each region's eligible unit list
using a recorded seeded generator. Select the first n units as source and the
next n as target; leave other units unused for that draw. Samples within a draw
are without replacement. Draws may share units; they are not independent
biological replicates. Do not demand unique draws when the population is small.

The resulting PFC source, PFC target, HPC source, and HPC target populations
support all four fits. Keep the same PFC source when comparing PFC -> PFC with
PFC -> HPC, and the same HPC source for HPC -> HPC with HPC -> PFC. The PFC
target is identical in both predictions of PFC, likewise for HPC.

This design matches both source and target counts across comparisons, at the
cost of discarding surplus units in the larger region. Do not silently replace
it with all available source units. No firing-rate matching in v1.

Proposed technical defaults: seed 0, five fixed draws, deterministic unit ordering
before sampling. Require n >= 2 to support a meaningful rank comparison; show
the actual count and unavailable reason otherwise. Save actual draw IDs, unit
IDs, and seed, not just the random-generator specification.

## 6. Validation and leakage prevention

Partition chronological eligible choices into five contiguous outer folds with
approximately equal numbers of choices. Each choice is tested once. Keep all
bins of a choice together. Use identical outer folds for all comparisons,
population draws, ranks, and full-model fits.

For each fold, hold out its choices and remove from training any other choice
window intersecting a held-out window in real time. This applies even if the
existing timeline assumes choices are sufficiently separated. Reject leakage
based on physical intervals, not trial-ID inequality alone. Training windows
may overlap each other; test windows may overlap each other, so the resulting
scores remain descriptive and are not independent-sample uncertainty estimates.

Use one inner chronological validation block drawn from outer-training choices
to tune alpha. Proposed detail: take the latest 20% of those choices, rounded
up, as inner validation, and purge overlapping windows from inner training.
Outer-test choices never participate in tuning, scaling, or fitting. Require
at least two choices in each validation/test block and at least two remaining
training choices, plus usable variable observations. Otherwise mark the run or
affected fit unavailable; never silently reduce fold counts.

Fit centering, scales, and constant-unit masks separately on inner training and
outer training as appropriate. Refit the selected model on all purged outer
training observations before testing. Save inner/outer trial IDs, purge counts,
selected alpha, valid observation counts, and fit status.

Five-fold held-out scoring is retained despite exploratory scope. It prevents
training fit from being mistaken for predictive performance. It does not remove
all slow temporal dependence or establish stationarity across the session.

## 7. Preprocessing and constant units

For each training partition, estimate each source and target neuron's mean and
population SD (`ddof=0`) from training observations only. Apply those same
parameters to validation/test observations. Model responses are standardized,
dimensionless counts; predictions are converted back to spikes per bin for
saved per-neuron errors and R2 scoring.

Z-scoring does not remove task-locked means separately by condition or time bin.
It gives neurons comparable fitting weight, but may amplify low-rate noisy
units; note this limitation in the scientific summary.

Proposed constant-unit contract: detect zero training variance explicitly.
Omit constant source columns from the numerical solve, without resampling the
population. Predict constant training targets by their training mean; retain
their target identity and score them if held-out variance is nonzero. Targets
with zero held-out variance have undefined R2 and must be flagged, not assigned
zero or one. Report effective source/variable-target counts and valid R2 counts.

Do not silently make replacement draws based on test activity. Preserve paired
comparisons even when effective dimensions differ. A fit with no variable
source or target activity is unavailable for a rank curve.

## 8. Estimators and tuning

Let X and Y be training-standardized matrices with shapes `(N,p)` and `(N,q)`.
Use a separate unpenalized intercept through training centering; the reported
rank applies to the slope coefficient matrix B, not the intercept.

Unrestricted ridge minimizes:

`||Y - X B||_F^2 / N + alpha * ||B||_F^2`.

Regularized RRR minimizes the same objective subject to `rank(B) <= r`.
This is not PCA of source activity followed by regression, and it is not
unregularized RRR compared against ridge.

For alpha > 0, define `A = X.T X + N * alpha * I`. The unrestricted coefficients
solve `A B = X.T Y`. For RRR, obtain right singular vectors V_r of
`A^(-1/2) X.T Y`, then use `B_r = B_ridge V_r V_r.T`. Use stable solves and
factorizations rather than forming an explicit inverse; an equivalent
augmented least-squares derivation is acceptable. Verify the derivation and
installed NumPy/SciPy APIs before coding. Simply taking the SVD of ordinary
ridge predictions is not generally the solution of this penalized objective.

At the maximum effective rank and a fixed alpha, RRR must reproduce unrestricted
ridge predictions numerically. This is a required correctness test.

Proposed alpha grid: `10**k` for k = -6, -5, ..., 6, under the normalized objective
above. Tune alpha separately for each comparison, population draw, outer fold,
and candidate rank, using inner-validation mean per-neuron R2. Tune unrestricted
ridge on the same grid and same inner split. Break numerical score ties in favor
of larger alpha using a documented tolerance. Log grid-boundary selections;
do not automatically expand the grid in v1.

Evaluate integer ranks 1 through n where identifiable. Effective rank limits
depend on variable source/target counts and training design rank, including the
inner split. Record unsupported ranks as unavailable, never relabel a smaller
rank as the requested rank. Rank masks and diagnostics are saved per fold.
Do not select ranks using outer test scores; the curve is exploratory output.

Separate alpha tuning per rank answers how well each rank can predict with
regularization tuned on training data. It does not isolate rank at a fixed
penalty. The maximal-rank curve and unrestricted baseline should agree when
their feasible dimensions and tuning objectives coincide; different masks
must be surfaced rather than hidden.

## 9. Scores, summaries, and plots

For each outer test fold and target neuron, calculate
`R2 = 1 - sum((y - y_hat)**2) / sum((y - mean(y_test))**2)`.
Keep negative values. Undefined denominators yield an explicit unavailable
score. Store squared-error and total-variation sums so scores are auditable.

Primary fold score: arithmetic mean of valid per-neuron R2 values. For paired
rank/full comparisons, use the same target-score validity mask. Summarize a
draw/rank by equally averaging its five fold scores only when all five are
valid. Summarize a comparison/rank by the mean across complete draws, reporting
the number of contributing draws. Mark incomplete curves conspicuously; do not
compare ranks with differing draw coverage as though they had equal support.

Display two panels, one for PFC-source predictions and one for HPC-source
predictions. Each contains within-region and cross-region RRR curves plus
matching unrestricted-ridge horizontal references. Show individual draw curves
faintly and their mean prominently. These illustrate draw sensitivity, not
confidence intervals. Include an R2 = 0 reference and integer rank ticks.

Use opaque white backgrounds, black text/axes, readable fonts, consistent
within/cross colors and line styles, and captions with the session, window,
bin width, raw-activity interpretation, five-fold/five-draw design, and valid
sample counts. Explain negative R2 as worse prediction than the held-out mean
reference. No p-values or statistical-equivalence language.

## 10. Webapp workflow

Add a dedicated **Communication subspace** view for the loaded session.

1. Resolve both spike populations and display good-unit counts, n, coverage,
   eligible choices, and readiness/unavailable reasons.
2. Offer 100/500 ms bin width (proposed default 100 ms) and seed controls. Show
   the fixed choice window, five draws, validation scheme, and matching rule.
3. Run only on explicit Run/Recompute. Display progress by draw/fold and allow
   failures to appear as actionable status rather than an empty chart.
4. Load saved compatible results without recomputing. Changing configuration
   marks displayed results as belonging to their saved configuration; never
   relabel stale results as a newly selected setting.
5. Show the two plots and diagnostic tables: selected/effective units, ranks,
   alpha choices, complete folds/draws, excluded choices, and undefined scores.
6. Save numerical results and light PNG plots; permit reopening a saved result
   without requiring original spike arrays to be loaded again. A new analysis
   still requires both regions. Viewing a valid existing result is not a new
   one-region analysis.

Display-control changes must not trigger fits or overwrite outputs. No timeline
projection viewer, factor analysis, private-subspace removal, condition-specific
models, or lag sweep in v1.

## 11. Persistence, provenance, and execution scope

Store numerical `.npz` files under a communication-subspace subdirectory of the
original session data directory, never in a Git-tracked output location. Use
named arrays and a dictionary named `meta`, following established project
serialization conventions after inspecting their trust/loading constraints.

Metadata includes generator/function, analysis version
`communication-subspace-v1`, code revision and dirty-state information where
available, input/source identities, selected unit labels/IDs, parameters,
random seed/draws, trial IDs, bin edges, units/axis conventions, preprocessing,
folds/purges, alpha grid/selections, statuses, and timings. Save score arrays,
per-target error totals, all transforms needed to interpret fits, and fitted
coefficients/intercepts or an explicitly documented compact factorization.
Persist structured identity axes alongside every multidimensional result.

Create an analysis run directory named with analysis name, date, and time in an
appropriate session-local output location. Include the main script snapshot,
any batch script if one exists, PNG plots, a short Markdown scientific summary,
and a log of parameters, session ID, runtime, warnings, and errors. No batch
script is required for this explicitly single-session inspection; the summary
must state that no batch processing was performed.

Changed parameters, code/analysis version, or sources create new result identity.
Never overwrite a previous scientific result. Explicit recomputation creates a
new timestamped run even with identical parameters. Default loading reuses a
compatible completed result; interrupted writes must not appear complete.

Before experimental use: test computation functions, then a complete synthetic
pipeline, then one user-designated session. The user inspects that output before
any broader execution is considered. No batch authorization is implied.

## 12. Proposed architecture and dependencies

Keep the feature under `src/neural_analysis/communication_subspace/`:

- `configuration.py`: explicit validated parameters and analysis version.
- `preparation.py`: adapters to existing good-unit loading, common choice
  binning, matched draws, and chronological/purged split construction.
- `modeling.py`: training-only scaling, ridge/RRR fitting, prediction, tuning.
- `scoring.py`: per-neuron metrics and paired fold/draw summaries.
- `results.py`: documented result records, provenance, and persistence.
- `plotting.py`: plots from saved/computed results, independent of fitting.
- `run_session.py`: explicit single-session orchestration and logging.
- `webapp/communication_subspace_view.py`: thin controls, readiness, progress,
  saved-result discovery, and rendering; register in `webapp/app.py`.

Inspect existing callers and reuse contracts from `unit_spike_loading.py`,
`session_metadata.py`, `population/session_timeline.py`,
`population/session_timeline_io.py`, `population/session_cca_io.py`, and existing
interregional computation and webapp views. Reuse compatible utilities, but do
not import CCA's in-sample fitting or default good-plus-MUA policy.

Dependencies: existing NumPy/SciPy, pandas, Pynapple, Matplotlib, Streamlit, and
standard library; pytest for tests. No new dependency is planned. Verify each
newly used external API through installed source or official documentation.
Use `uv run` for local Python commands. Do not modify external libraries.

Every function documents input/output types, array shapes/axes, and units.
Favor short, readable functions and localized changes over general frameworks.

## 13. Tests first and implementation sequence

Proposed tests under `src/tests/neural_analysis/communication_subspace/`, using
pytest functions and fixtures. Write and commit tests before implementation;
confirm the relevant failures are RED, implement GREEN, then refactor while
preserving behavior. Do not alter tests merely to fit the implementation.

Tests to write first:

1. Good-only selection, probe-qualified IDs, missing-region/label/coverage
   rejection, and n < 2 unavailability.
2. Deterministic seeded draws, exact count matching, disjoint within-region
   populations, and fixed source/target identities across all four comparisons.
3. Exact 100/500 ms bin edges, count conservation on known spike times,
   half-open boundaries, paired choice identity, and documented flatten order.
4. Five chronological folds, whole-trial grouping, real-time overlap purging
   in outer and inner splits, and insufficient-data failure without fallback.
5. Training-only means/scales/constant masks: changing held-out observations
   cannot change fit parameters, selected alpha, or population draws.
6. Ridge reference solution, constrained coefficient rank, penalized-objective
   checks against an independent small-matrix solution, and maximal-rank RRR
   equivalence to unrestricted ridge at fixed alpha.
7. Alpha tuning uses only inner validation, deterministic tie handling, and
   correct effective-rank masking across folds.
8. R2 with perfect, poor, negative, constant-training, and constant-test targets;
   paired validity masks and incomplete-fold/draw summaries.
9. Seeded synthetic low-rank shared signal: a small rank recovers predictable
   target structure on held-out samples; a separate higher-rank construction
   requires additional predictive dimensions. Use stable tolerances, not exact
   noisy rank recovery. Include independent source/target data as a null check.
10. Persistence round trip preserves axes, provenance, draws, splits, and scores;
    configuration changes cannot reuse stale results or overwrite old files.
11. Webapp gating, explicit-run behavior, no fitting on display changes, saved
    result reopening, and light plot labels/captions.
12. End-to-end synthetic single-session preparation, fitting, saved outputs,
    reopening, logging, and plotting.

After approval: inspect related modules/callers and APIs; write/commit the first
test tranche and verify RED; implement pure computation; complete preparation
and persistence; add plotting and webapp integration; run focused tests and
synthetic pipeline. Request the designated real session only when ready for
that validation stage. Preserve all pre-existing worktree changes.

## 14. Performance considerations

Prioritize correctness and readable linear algebra. Five draws times five outer
folds times four comparisons gives 100 outer fitting tasks, each with inner
alpha/rank evaluation. Bin once per run and reuse common observations/folds.
Cache cross-products/factorizations within each training partition and alpha;
one ordered decomposition should support all feasible ranks at that alpha.
Do not refit identical preprocessing for every rank or store duplicate raw
count tensors per draw. Retain sufficient provenance without saving every
candidate prediction array.

The 500 ms option has five times fewer observations than 100 ms. Estimate array
sizes and show progress before expensive fitting. Start with bounded serial
single-session execution; session-batch parallelism is outside scope. Do not
launch one worker per rank or introduce optimization dependencies. Profile only
if runtime is problematic, and seek approval for major optimizations.
