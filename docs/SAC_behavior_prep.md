# SAC TTS analysis and figure specification - working draft

Status: planning, not approved for implementation. No new fits have been run.
The original exploratory notes are preserved at the end of this document.
Items marked OPEN require a decision; recommendations are not settled requirements.

## 1. Purpose and scope

Produce a reproducible set of analyses, tables, and figures that:

1. Reproduce the exploratory comparison of population TTS mixed models.
2. Show the population relationship between previous-block rewards, block side,
   and TTS, alongside variation across sessions and mice.
3. Estimate each session's overall intercept, reward sensitivity and residual
   spread without side adjustments, and inspect side differences separately.
4. Show whether the parameter distributions support the existing unbiased HMM
   exemplar parameters, and identify where those exemplars are inadequate.

The preferred mixed model in the notes is an exploratory result to reproduce,
not an outcome that the new code must force. Report conflicting evidence.
Likewise, exemplar values are candidate reference values, not estimated truths.

Confirmed scope: TTS only, primarily R, with light-background figures. Trial-choice
models are deferred. Changing HMM inference or its default parameters is not part
of generating these figures.

## 2. Data contract and cohort

Confirmed mice: CT016, CT017, CT019, CT020, CT021, CT022, CT023, CT024, CT025.
These sessions contain no cued blocks. Use the existing uncued left/right labels;
do not add cue-status models, strata or configuration machinery.

Validation session: `CT024_20260609_latent_inference`, replacing the earlier CT016
suggestion. The existing R example identifies its block table as:
`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260609_latent_inference/processed/CT024_2026-06-09_143852_block_performance.csv`.
Verify availability at implementation time. The existing data-root convention is
`/home/matt/Documents/EXPERIMENTS/contextProjectData`.
OPEN: all available processed sessions versus a date/training-stage restriction.

One input row represents one task block. Retain its original session-local block
index and ordering; uniquely identify it with mouse, session, and block index.
Do not concatenate history across independent sessions.

| Field | Meaning / units |
|---|---|
| `trials_to_correct` (`tts`) | Behavioral choices before the first correct behavioral choice, in trials; immediate correctness is 0 |
| `prev_n_rewarded` (`nrew`) | Number of rewarded behavioral choices in the preceding block, in rewarded trials |
| `block_type` | Original uncued left/right block label |
| `side` | Derived left/right factor; right is the existing regression reference |
| `mouse` | Animal identifier |
| `session` | Globally unique session identifier, with its original date and mouse |
| `block_ix` | Original within-session block identifier |

Verified definition: `summarize_block_performance()` in
`src/behavior_analysis/session_analysis.py` calculates TTS after filtering to
behavioral choices. It is not time to sustained switching or first reward.
Blocks with no correct behavioral choice have missing TTS. The first block has
`prev_n_rewarded = 0` because it has no observed predecessor.

Required audit before fitting:

- Resolve existing session ID aliases explicitly; never merge sessions by date alone.
- Report mice, sessions, blocks, and reward ranges before/after filtering.
- Report missing TTS, invalid predictors, unknown block labels, duplicate keys,
  first blocks, blocks with no correct choice, and terminal/incomplete blocks.
- Distinguish block initialization from a genuine context reversal. Recommendation:
  exclude the first block from reversal analyses, pending approval.
- Missing TTS is never zero or an imputed completed switch. Describe the primary
  complete-case analysis as conditional on observing a first correct choice.
  Report the excluded count; a separate censored-outcome analysis is not part
  of this implementation.

Proposed eligibility: retain the existing five-valid-block execution threshold,
flag absent sides or insufficient reward variation for side fits, and show fit
counts. Five blocks do not establish reliable side-specific slopes. Do not add
other training-stage or behavioral exclusions without a specified reason.

## 3. Population mixed models

Use the same eligible rows, response units, and left/right contrasts for all four
primary candidates. Cue status is constant in the specified sessions.

**M1: Full fixed and session interactions**

```r
tts ~ nrew * side + (nrew * side | session)
```

Question: Is a population-average slope difference needed?

**M2: Session interaction with additive fixed effects**

```r
tts ~ nrew + side + (nrew * side | session)
```

Question: Can slope differences vary by session without a population-average
interaction? This is the exploratory preferred model.

**M3: Additive fixed and session effects**

```r
tts ~ nrew + side + (nrew + side | session)
```

Question: Is the session-specific slope interaction needed?

**M4: Interaction with mouse/session nesting**

```r
tts ~ nrew + side + (nrew * side | mouse/session)
```

Question: Does adding mouse-level grouping improve the M2 description?

M2 contains a random reward-by-side interaction even though its fixed effects
are additive. M4 adds mouse-level random effects for the full specified term set;
it is more than adding a mouse random intercept.

Required outputs:

- Full formulas, fitted observation IDs/counts, numbers of mice and sessions,
  likelihood method, parameter counts, log likelihood, AIC, BIC, delta AIC/BIC,
  convergence warnings, singular-fit status, and random-effect variances/correlations.
- Fit the comparison set using maximum likelihood (`REML = FALSE`), since fixed
  effects differ. If the selected model is refit with REML for estimation, save
  and label that fit separately from the ML comparison set.
- Explicit contrasts M1 vs M2, M2 vs M3, and M2 vs M4. AIC and BIC can disagree;
  retain both and document the reason for the presentation model choice.
- Do not hide singularity or silently simplify random effects. Save any approved
  simpler fallback as a separately named candidate. A naive chi-square test for
  removing random effects is not the default selection rule.
- OPEN: whether Bayes factors are required. A BIC-derived evidence approximation
  must be labeled as an approximation, with direction and formula documented;
  it must not be presented as an exact Bayesian model comparison.

Population predictions use fixed effects; session predictions include the fitted
session deviations. Label these separately. Show predictions only over documented
reward support, and label any display of the intercept at zero as extrapolation
when zero is outside the observed support. Gaussian linear fits remain the
starting descriptive model; inspect negative predictions, skew, residual variance,
and block-order dependence before treating them as an adequate outcome model.

Method references: [lme4 model comparison](https://lme4.github.io/lme4/reference/merMod-class.html)
and [singular-fit diagnostics](https://lme4.github.io/lme4/reference/isSingular.html).

## 4. Per-session elastic-net fits

Confirmed: fit these two primary formulas to the same eligible rows within each
session.

**Overall summary / exemplar-compatible fit** (no side factor):

```r
tts ~ nrew
```

**Interaction fit** (`rewards_x_side`):

```r
tts ~ nrew * side
```

The existing additive fit `tts ~ nrew + side` (`rewards_plus_side`) can remain
available for inspection, but is not the primary session summary. Reuse the R
fitting helper and extend it locally for the side-free formula, preserving existing
public interfaces. Neither fitted form is assumed to be the true session model.
Proposed primary regularization: elastic-net mixing parameter 0.5 and
`lambda.min`, matching the exploratory distribution plots. Retain `lambda.1se`
as a sensitivity view. Proposed folds: retain seeded shuffled folds, as in the
existing descriptive analysis. Keep fold membership shared between formulas on
the same rows; record seed, folds, standardization, and selected penalty. This is
not an evaluation of prediction into future blocks or sessions.

The summary coordinates are the intercept and slope from `tts ~ nrew`, plus
the sample SD of that same fit's residuals. This matches the exemplar model's
functional form. It is not equivalent to averaging side-specific intercepts:
side/reward imbalance can change the fitted slope, intercept and residual spread.
Side bias left unexplained by this fit appropriately contributes to its residuals.

Implementation detail to resolve with a small tested project adapter: glmnet
requires at least two predictor columns, whereas `tts ~ nrew` has one. Preserve
the intended one-predictor elastic-net objective; do not silently substitute OLS,
duplicate the reward predictor, or reintroduce side as a predictor. Verify the
adapter's coefficients/predictions against the one-predictor penalized objective.
This constraint is documented in the [glmnet source](https://github.com/cran/glmnet/blob/master/R/glmnet.R).
The R executable is available in the current environment, but glmnet was not
installed when checked during planning; installation is an implementation prerequisite.

With the existing right-reference coding:

```text
predicted_tts = b0 + bR*nrew + bS*I(left) + bRS*nrew*I(left)
```

- Right intercept/slope: `b0`, `bR`.
- Left intercept/slope: `b0 + bS`, `bR + bRS`.
- Left-minus-right intercept/slope differences: `bS`, `bRS`.
- Equal-side mean intercept: `b0 + bS/2`; mean slope: `bR + bRS/2`.
- Additive fits have `bRS = 0` by construction.

Intercepts and side offsets are in trials; slopes are trials per previous
rewarded trial. Export coefficients in original units regardless of internal
standardization. Preserve the existing reference-coded outputs.

The equal-side averages remain useful descriptive outputs of side-adjusted fits,
but do not replace the directly fitted side-free summary coordinates.

Residuals are observed minus fitted TTS on the fitted rows. The existing
`residual_sd` is sample SD with denominator `n - 1`, not a residual-degree-of-freedom
estimate and not cross-validated prediction error. Save SD, RMSE, IQR, and scaled
MAD separately; do not interchange their labels. Compute a separate SD for each
formula/penalty combination. In-sample residual spread may be optimistic and
can reflect omitted structure, side differences, or changing behavior, not only noise
or naivety. If predictive performance is claimed, require held-out predictions.

Flag absent sides, insufficient reward variation, low block counts, failures,
and unstable penalty-sensitive estimates. Do not fill unidentifiable side
parameters with plausible-looking zeros. Keep skipped sessions in a status table.

## 5. Figures available for inspection and presentation

All figures require readable fonts, explicit units, a caption, model/formula and
penalty labels, cohort counts, and the corresponding numeric source table.
Confirmed: opaque white backgrounds with black axes/text. PNG is the default export. Additional vector export is
optional. The 3D view supplements readable static pairwise projections.

| ID | Figure | Required content / purpose |
|---|---|---|
| F1 | Cohort and TTS overview | Counts/exclusions by mouse and session, observed TTS and reward support, missing-switch frequency |
| F2 | Mixed-model comparison | Delta AIC and delta BIC for M1-M4, candidate formulas, convergence/singularity annotations, explicit M2-vs-M4 comparison |
| F3 | Overall mixed-model relationship | Raw block observations with controlled overplotting, population left/right lines, session conditional lines in an inspection panel; distinguish mean confidence bands from residual spread |
| F4 | Individual-session fits | Requested session showing points by side, side-free and interaction predictions, residual panels, block counts/reward ranges and fit warnings; start with CT024_20260609_latent_inference |
| F5 | Session parameter distributions | Intercept, slope and residual SD from the side-free fit; overlay exemplar reference values; mouse facets or per-mouse summaries to expose unequal session contributions |
| F6 | Session parameter space | Side-free intercept-slope-SD 3D display plus all three 2D projections; retain mouse/date/session IDs for inspection; do not claim clusters from appearance alone |
| F7 | Side asymmetry | Paired left/right intercepts and slopes, difference distributions, and difference-vs-reward prediction curves where needed; distinguish intercept offset from slope asymmetry |
| F8 | Exemplar adequacy | Exemplar mean lines over observed reward support, parameter locations relative to empirical distributions, SD reference comparisons and examples of substantial side bias |
| F9 | Diagnostics and sensitivity | Residual-vs-fitted/reward/side, Q-Q and block-order checks; lambda.min vs lambda.1se parameter changes and session eligibility flags |

Proposed compact presentation set: F2, F3, F5, F7, and F8. Keep F1/F4/F6/F9
available for inspection; final panel selection depends on the actual results.
Sessions are repeated observations within mice. Distribution captions must state
whether every session has equal weight; do not treat session counts as animal
sample sizes or add inferential error bars without specifying their unit.

## 6. Exemplar justification and limits

The existing unbiased exemplar definitions are:

| Name | Intercept (trials) | Reward slope (trials/rewarded trial) | Residual SD (trials) |
|---|---:|---:|---:|
| `expert_inf` | 2 | 0 | 2 |
| `expert_rl` | 2 | 0.5 | 2 |
| `naive_inf` | 6 | 0 | 5 |
| `naive_rl` | 4 | 1 | 5 |

Note the existing `naive_rl` intercept is 4, not 6. Its inclusion in the
justification must be explicit rather than restricting the display to 2 and 6.

These are side-neutral descriptive prototypes. A low reward slope alone does
not establish inference, and a positive slope alone does not establish RL.
Retain the existing names for compatibility while explaining this limitation.

Required evidence: empirical marginal quantiles, joint parameter support, mouse
contributions, proximity of candidate exemplar values to the observed sessions,
and representative raw fits. Show the candidate values before choosing new ones.
Do not infer a supported prototype solely by combining three unrelated marginal
peaks. Any automatic representative-session selection requires an approved
distance/scaling rule; otherwise presentation sessions are selected explicitly.

Use the side-free fitted residual SD for the primary exemplar comparison. Keep it
distinct from interaction-fit SD and the actual errors under fixed exemplars.
Where those errors are shown, report their mean as well as spread: SD alone
does not capture a constant prediction offset. No automatic HMM parameter update.

## 7. Implementation plan (approval required before code)

Design basis: `docs/SoftwareDesign.md`. Translate its language-independent
guidance into ordinary R functions, data frames, numeric vectors/matrices and
small named lists. No classes, protocols, factories, plugin system or extra
package hierarchy are needed for this analysis. Pass data and parameters
explicitly; keep run-control flags and path construction in the entry scripts.
Functions that align or transform a whole table receive a data frame; scalar
statistics and numerical fitting helpers receive only the vectors/matrices needed.

Confirmed: primarily R. Use lme4 for mixed models, glmnet for session fits,
ggplot2 for exports, and Flexplot for useful inspection views. Use the existing
R tests rather than adding a test framework. Verify installed APIs before
implementation. Proposed storage: native RDS for fitted objects and computation
results plus CSV tables, avoiding a Python conversion layer solely for storage.
This is a proposed exception to the project's preferred intermediate formats.
Use base R for configuration, CSV/RDS I/O, logging and session parallelism. For
the initial 3D figure, use a static base-R projection alongside the ggplot2 pairwise
views; an interactive dashboard is not required. Verify the relevant package
APIs at implementation time. No new Bayesian or interactive plotting dependency
is proposed.

### Files and responsibilities

Keep the new analysis files directly under `src/Rcode/`, using the existing
`source_utils.R` path resolution. Names below are proposed implementation names.

**`tts_data.R`: prepare analysis rows.**

- `prepare_tts_blocks(block_df)` returns a named list containing the numeric
  analysis table and exclusion/status table. The input and output retain the
  original block/session keys. Clean with `drop_missing_tts = FALSE`, then apply
  the explicitly approved TTS/reward/side and first-block rules. Do not require
  `prev_n_correct`, which the existing generic drop-missing cleaner also checks.
- Preserve the session-initial marker before filtering; do not infer the first
  block from the first surviving row. Do not recalculate previous rewards after
  exclusions. The preparation function does not read files or fit models.

**`block_residual_modeling.R`: reuse and locally extend session fitting.**

- Add a side-free formula identifier `rewards_only` through an explicit opt-in
  interface. Retain the existing default formula set, signatures and output
  columns/row counts for old callers, including lasso outputs and aliases.
- Separate the glmnet CV fit from extraction at `lambda.min` and `lambda.1se`.
  The new SAC path performs one CV fit per formula and extracts both choices.
  Reuse existing coefficient reconstruction and residual statistics.
- Add the small one-predictor adapter discussed in section 4. Its purpose and
  effect on the penalty must be documented and tested before it is used.
- Use explicit formula specifications, alpha and fold assignments as inputs to
  the new fitting core; avoid new mutable global analysis settings.

**`tts_modeling.R`: perform the scientific computations.**

- `fit_session_tts(block_df, seed, alpha, min_valid_blocks)` composes the prepared rows and regularized
  fitting core for the two confirmed formulas. Return a named list of parameter
  rows, row-aligned predictions/residuals, CV information and status, including
  the exact eligible block IDs. No file writes or plot generation.
- `fit_population_tts(block_df)` fits the four explicit ML candidates M1-M4 and
  returns fitted objects, comparison rows and diagnostics. Keep the four
  candidates as a short, readable named list of formulas, not a generic registry.
- `predict_population_tts(fits, prediction_data)` produces explicitly labeled
  fixed-effect and conditional predictions. Prediction grids and session labels
  are passed in, rather than constructed from hidden global state.
- Exemplar comparison uses the documented four parameter rows as explicit input
  and returns predictions/errors; it does not import or run the Python HMM.

**`tts_results.R`: load/save the analysis artifacts.**

- `save_tts_results(result, output_dir)` writes named RDS components and CSV
  tables with model, row, seed/fold and version metadata.
- `load_tts_results(result_dir)` returns the same documented named-list contract.
- Cache reuse compares a saved input checksum, analysis version and parameters;
  a mismatch produces a new versioned result, not an overwritten old result.
  Keep this as ordinary metadata comparison, without a general cache framework.
- Do not call the legacy summary writers on source CSV paths: their existing
  behavior overwrites filenames. Store SAC outputs in dedicated analysis folders.

**`tts_plotting.R`: turn saved numeric results into F1-F9.**

- Named figure functions such as `plot_tts_model_comparison(comparison_df)` and
  `plot_tts_session_fit(block_df, predictions_df)` receive their source tables
  and return plot objects where supported. Render/export explicitly at the
  plotting entry point. No source-data loading or model refitting inside plots.
- Keep figure functions separate by scientific question. Reuse short axis/theme
  helpers only where they make repeated figures easier to read.
- Use static 3D and pairwise projections initially. Inspect a session by passing
  its ID to the session figure, rather than building a selection UI.

**Entry scripts: `run_tts_session.R`, `run_tts_batch.R`, `plot_tts_results.R`.**

- The session script loads one designated block CSV, prepares/fits it, and saves
  its results. Default validation target is `CT024_20260609_latent_inference`.
- The batch script constructs the session-path table for the nine specified
  mice, runs session computations in parallel, collects outputs and fits the
  population candidates once. An explicit configuration list near the top owns
  the data root, session selection, seed, analysis version and output paths.
  No configuration package, command-line framework or separate discovery service.
- Support a dry run that prints the selected CSV paths and output locations,
  plus explicit rerun control at these entry points.
- The plotting script loads a saved run and renders the requested figures. It
  can be rerun with new display choices without recomputing the analysis.

### Implementation sequence

1. Confirm remaining scientific/run choices in section 9 and verify package/data
   availability. Do not install packages or modify dependencies during planning.
2. Write the tests listed in section 8, confirm RED for missing new behavior and
   commit only those test changes before implementing it, as required by AGENTS.md.
   Preserve unrelated worktree changes.
3. Implement preparation and the session-fitting extension; get their tests GREEN
   and check existing R residual-model tests still pass with their old defaults.
4. Implement population fitting and result I/O; validate with small synthetic
   multi-mouse tables. Add the figure functions and visually inspect exports.
5. Run the designated CT024 session and present its fit/figures for approval.
   After approval, run the agreed cohort and generate the presentation set.

### Performance and limitations

Fit each formula once per session, extract both penalties, and save numeric
predictions for plotting. Session workers use independent deterministic seeds
and one inner glmnet thread/process, avoiding nested parallelism. Use base R
session workers supported on this Linux environment, defaulting to available CPU
cores subject to RAM/I/O limits and number of sessions. Run population candidates
sequentially; do not add compiled code or profiling-driven optimizations before
there is evidence of a bottleneck. No bootstrap sweep is required by this plan.

Document each function's purpose, input types/shapes and row conventions, physical
units, return fields, and limitations in concise R header comments. Numeric tests
verify scientific behavior; avoid brittle exact-image tests or tests that merely
repeat implementation details.

Existing integration points:

- `src/Rcode/inspect_blocks.R`: source of exploratory candidates and plotting ideas.
- `src/Rcode/preprocessing/block_preprocessing.R`: existing cleaners; inspect their
  extra filtering before reusing them as the eligibility definition.
- `src/Rcode/block_residual_modeling.R`: R session fitting and coefficient summaries.
- `src/behavior_analysis/block_residual_models.py`: Python counterpart.
- `src/main.py`: existing session/mouse/cross-mouse summary collection functions.
- `src/behavior_analysis/block_exemplar_models.py`: current exemplar definitions.

Do not require editing hardcoded filenames. Save per-session intermediates in
session data subdirectories, and cohort results/figures in a versioned
`sac_tts_<date>_<time>` directory outside version control. Include script snapshots,
manifest, parameters, package/code versions, seed/folds, units, row identifiers,
log, runtime, exclusion audit, figure source tables/captions, and a scientific
summary. Preserve old results when inputs, code or parameters change.
Keep computation and figure generation separate, using saved R results and CSV
tables so changing a figure does not rerun fits.

Cache session fits and render from saved predictions. Parallelize across sessions
using available cores subject to RAM/I/O constraints, and avoid nested worker
oversubscription. Fit population candidates separately; reproducible workers must
not depend on scheduling order. Heavy resampling is a separately costed option.

## 8. Tests to write and commit before implementation

Use the existing `Rscript` / `stopifnot()` test style, with named test functions
and reusable fixture functions. Proposed files are `test_tts_data.R`,
`test_tts_modeling.R`, `test_tts_results.R` and `test_tts_plotting.R` under
`src/Rcode/tests/`, plus focused additions to `test_block_residual_modeling.R`.
Test computations and saved figure source data; visually inspect final figures.

1. Block/session ID preservation, duplicate rejection, left/right mapping and
   first-block/missing-TTS policy on a small hand-checkable table.
2. TTS zero convention and previous-block reward alignment without crossing sessions.
3. Side-free/additive/interaction design matrices and original-unit coefficient
   reconstruction, including equal-side averages and left-minus-right differences;
   one-predictor adapter parity with the intended elastic-net objective.
4. Missing side, constant rewards, sparse session and nonfinite input statuses.
5. Seeded fold reproducibility and shared folds across eligible formula comparisons.
6. Residual sign, sample SD, RMSE, IQR and MAD against hand-calculated examples.
7. Exact candidate formulas, identical comparison rows, ML comparison metadata,
   nested mouse/session identities, and visible convergence/singularity reporting.
8. Saved-result round trip, cache invalidation, no-overwrite behavior, and plotting
   without refitting. Figures must also receive visual inspection.
9. Seeded synthetic sessions with known intercept, side and slope differences:
   verify signal direction and prediction behavior with tolerances appropriate
   to regularization; do not require arbitrary coefficients to be recovered exactly.

Then run the complete pipeline on synthetic data, followed by
`CT024_20260609_latent_inference` for fit/figure approval before the full cohort.
A single session validates session fitting/plotting, not mouse-versus-session
grouping; test that part on synthetic multi-mouse data before the cohort fit.
Use the existing R test scripts and run the relevant regression checks.

## 9. Decisions still needed

- Confirm all processed sessions versus a date/training-stage restriction.
- Confirmed: side-free summary and side-interaction fits. Approve proposed
  five-block threshold and seeded folds, alpha 0.5 / lambda.min primary
  (lambda.1se sensitivity), and RDS/CSV outputs.
- Confirm exclusion of session-initial blocks and omission/reporting of missing TTS.
- Decide whether Bayes-factor output is needed in addition to AIC/BIC.
- No automatic exemplar selection is proposed; inspect the distributions first.
- Approve the implementation plan before writing code.

## Original exploratory notes (preserved)

### TTS analysis
- Find R code that determines the TTS distributions for the block changes
(2 vs 6 intercept, 0 vs .5 vs 1 slope, use of different sds of 2 or 5)
- Reorganize R code that does the fitting of different trial models, block
TTS with elastic-net intercept/side bias/slope determination, determination
of the suitable multisession model with AIC/BIC/Bayes factor on the mixed model analysis for presentation visualization
- need to see the distributions of the intercepts, slopes, and SDs of the TTS fits to justify
exemplar model params

Comparison of mixed model fits
- Full interaction w/ session cluster: tts ~ nrew * side + (nrew * side | session)
- Slope interaction w/ session cluster: tts ~ nrew + side + (nrew * side | session) - this is the best model
- No interaction w/ session cluster: tts ~ nrew + side + (nrew + side | session)
- Slope interaction with nested cluster: tts ~ nrew + side + (nrew * side | mouse/session)

Get overall mixed model fit to show and a visualization.

Obtain individual session fits with elastic-net regression to get session-specific parameters.
- True session fit:  tts ~ nrew * side (elastic-net)
- Session summary:  tts ~ nrew + side (elastic-net)

The session summary can be used to quickly compare/categorize sessions, while the true session fit
captures the session a bit more accurately.
- The true session fit can have different slopes/intercepts for each side, so you can determine whether there is bias
or different strategies / slopes for each side
- Find the standard deviation of the residuals from the fit to approximate the noise/naivety of the session
- From the session summary, get the slope, int, SD from its session elasticnet fit. These 3 can be plotted in a 3D plot to view session clusters by strategy.
- Can see different slopes/intercepts for each side from the true session fit to characterize session strategy and bias per side.

Exemplar models in HMM fits are assumed to have no side bias (no intercept adjustment for side, no differing slopes for side).
