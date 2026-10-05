# Inter-Regional Prediction Analysis Spec

## Goal

Measure directed predictive relationships between PFC and HPC population activity.

The core question is:

> Does recent activity in one region improve prediction of activity in the other region beyond what can already be predicted from the target region's own recent activity?

Analyze both **HPC → PFC** and **PFC → HPC**. Retain prediction performance separately for each target unit or PC, then show the population distribution with individual points, median, and IQR.

Use two complementary quantifications, with their fitting scopes explicitly distinguished:

| Output | Evaluation scope | Purpose |
|---|---|---|
| Incremental CV $R^2$ or CV deviance explained | Held-out trials | Added predictive information from source-region history |
| Linear Granger magnitude or Poisson Granger-style magnitude | Separate in-sample fits | Descriptive restricted-versus-full time-series quantification |
| OLS–Poisson spike-count MSE comparison | Held-out trials | Exploratory comparison on a common prediction-error scale |

This is a **first-pass inspection analysis**, not a publication-finalized inference pipeline. Interpret results as predictive, not as proof of mechanistic causality. Do not expand the implementation into communication subspaces, formal Granger significance testing, or temporal-confound correction.

## Session, Alignment, Windows, and Conditions

Analyze one simultaneously recorded session at a time. Do not pool neuron columns from different sessions as though they identify the same population.

Use the existing alignment selector:

- **Choice** — default.
- **Trial start** — available through the existing analysis pipeline.

Fit and summarize these windows separately, relative to the selected alignment:

| Window | Interval |
|---|---|
| Before | $[-2,0)$ s |
| After | $[0,2)$ s |
| Whole | $[-2,2)$ s |

Keep window bounds adjustable through existing controls. With the default bounds, use the full $[-2,2)$ s interval for the shared-temporal PCA basis; the selected before/after/whole interval determines the prediction rows.

**Fit separate models within each behavioral condition and window.** Do not fit one model across all conditions and then merely score condition subsets. An existing all-trials condition may be used when explicitly selected, but it does not replace condition-specific fits.

Reuse the codebase's trial-condition definitions and left/right choice or context selectors. Verify how they are represented rather than introducing a second condition system or conflating choice switching with context transitions. The summary figures show all requested conditions together.

Record eligible trial and target-bin counts for each condition/window. An unavailable condition-specific result must not stop other valid conditions.

## Observations and Predictor Populations

### One observation per trial × target-time-bin

Unlike the single-unit selectivity spec, the response here is **not a count summed over the entire before/after/whole window**.

For target PFC unit $j$, the default regression row is:

```text
One row = original trial k, target bin t

Response:
    Spike count of PFC unit j in bin t

Restricted-model predictors:
    Activity of every included PFC unit in bin t−1

Full-model predictors:
    The same PFC history
    + activity of every included HPC unit in bin t−1
```

The target-region predictor population includes **the target unit's own history and the histories of the other included units in that region**. It is not a self-history-only baseline.

For the PC representation, replace regional unit vectors with **all retained PC scores for that region**. A target PC is predicted from all retained target-region PCs, not only its own past score. Adding the source region adds all its retained PC histories.

Pool eligible trial × time-bin rows within the selected condition/window and fit **one coefficient set per target**. Do not fit a separate regression for every trial or every event-relative bin.

During held-out evaluation, predictors are the **observed historical activity** transformed using training-derived preprocessing where applicable. Do not feed predicted responses back into later predictions; this is not recursive trajectory simulation.

## Temporal Parameters and History Construction

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| Neural bin size | **100 ms** | Width of each count/activity bin |
| Prediction lag $\ell$ | **1 timestep** | Offset of the most recent predictor bin before the target |
| Model order $p$ | **1** | Number of consecutive historical bins included |

Available bin sizes: **500, 100, 50, and 20 ms**. Keep lag and order configurable positive integers. Require $\ell\geq1$; current-bin neural activity must not enter the directed-prediction models.

For target bin $t$, include these historical bins in both regions:

$$
t-\ell,\quad t-\ell-1,\quad \ldots,\quad t-\ell-p+1
$$

Examples for 100-ms bins:

- Lag 1, order 1: use $t-100$ ms only.
- Lag 1, order 3: use $t-100$, $t-200$, and $t-300$ ms.
- Lag 2, order 1: use $t-200$ ms only.

Both regions must use the same lag set in the restricted/full comparison.

### Trial and window boundaries

Construct histories using bin indices **within the original trial and selected analysis window**.

- Require the target bin and every included historical bin to lie within that window.
- Drop the initial target bins that lack the required history.
- Never use the last bin of one trial as history for the next trial.
- Never join condition-selected trials or fragments into an artificial continuous time series.
- Do not close gaps in missing data by treating the previous available bin as the immediately preceding bin. A row requiring missing activity is unavailable.

With 100-ms bins, lag 1, and order 1, a valid 2-s window contributes **19 target rows per trial**; a valid 4-s window contributes **39**. In the after window, the first usable response bin is 0.1–0.2 s, predicted from 0–0.1 s. This convention deliberately does not borrow pre-event history for the earliest after-event response bin.

Use **unsmoothed binned activity** by default. Do not use a centered smoother that introduces later activity into historical predictors. Reuse the existing binning utilities and verify their alignment and indexing conventions.

The known nonoverlap of the selected trial windows does not require a new overlap detector, temporal purging system, or boundary-trial exclusion scheme. The history rules above concern ordinary lag construction, not additional temporal-confound controls.

## Neural Representations

### 1. Unit activity — default

Use **spike counts per bin** as unit responses for both OLS and Poisson. Use the regional binned unit-activity vectors as historical predictors. The straightforward direct-unit implementation uses counts for those vectors as well.

For region $R$:

$$
\mathbf{x}_R(k,t) = [n_{1,k,t},\ldots,n_{N_R,k,t}]
$$

Here, $n_{u,k,t}$ is the count for unit $u$, trial $k$, and bin $t$. Firing rate may be shown as a descriptive display conversion, but is not a separate model-response toggle.

Evaluate each target-region unit separately and preserve its identity across folds, windows, conditions, and models. Reuse the existing selected unit population and quality criteria; do not automatically search for a smaller predictor subset.

### 2. Principal components

PCs are a dimensionality-reduced representation of the same regional population activity, not a separate specialized regression framework.

Use **up to 10 PCs per region by default**, with the requested count configurable. Retain fewer when fewer usable dimensions are available and record the actual count.

#### Cross-validated PCA construction

For each outer training fold, selected session/alignment, bin size, and regional unit set:

1. Assemble valid neural observations from the **training trials** across the full analysis interval, pooling trials × time bins. Use the training pool across requested conditions, not a newly fitted basis for each individual condition/window.
2. Z-score each unit once using that pooled training mean and standard deviation.
3. Fit **separate shared-temporal PCAs for PFC and HPC** to their standardized activity.
4. Disable whitening. Use ordinary PC scores directly: **no additional PC-score standardization, post-PCA scaler, or per-time-bin scaling**.
5. Transform held-out trials using only the training-derived unit means, standard deviations, and regional PCA projections.
6. Reuse these projections across restricted/full fits, directions, and before/after/whole condition comparisons in that fold. The regressions themselves are still fitted separately within each condition/window.

No held-out trial may contribute to PCA directions or fitted scaling. Do not perform a joint PFC+HPC PCA or fit different target-region PC axes for restricted and full models.

Units with no variance in the PCA fitting pool cannot be z-scored; report their omission from that PCA preprocessing rather than dividing by zero. This is not permission for automatic regression predictor selection. Preserve the existing population mapping and report effective feature/component counts.

#### Target PC identity across folds

Score each target PC **within its own fold**, where restricted and full models share the same axes. Average fold-level scores and increments by component rank, labeling those outputs as **fold-specific PC-rank summaries**.

PC1 need not be exactly the same population direction in every training fold. Do not concatenate fold-specific PC1 predictions and describe them as one fixed session-wide PC trajectory. Do not add PC matching or cross-fold axis alignment machinery.

A component rank missing in a required fold makes its primary cross-validated summary unavailable; other valid ranks and unit results may continue.

#### PCA for descriptive Granger fits

For the separate in-sample Granger analysis, fit one descriptive regional PCA basis using all eligible neural trials across the full analysis interval, with the same unit-z-scoring and no-whitening policy. Reuse it across conditions/windows and both directions.

This all-data basis must not be reused for cross-validated scores. Conversely, the descriptive Granger PC basis is not assumed identical to any one cross-validation fold's basis.

Because PC responses are continuous and may be negative, **PC prediction uses OLS only**. There is no Poisson-PC response option.

## Prediction Models and Matched Comparisons

| Representation | Model options |
|---|---|
| Units | **Unpenalized OLS** — default; **unpenalized Poisson GLM with log link** |
| PCs | **Unpenalized OLS only** |

Include an intercept in every restricted and full model. Use additive historical feature terms only. Do not import elastic-net decoding code, enable a library-default penalty, add interactions, or introduce hyperparameter searches. Verify estimator settings explicitly.

Let $Y$ denote the target region and $X$ the source region. Let $\mathbf{h}_Y(k,t)$ stack all retained target-region features over the selected history bins, and define $\mathbf{h}_X(k,t)$ analogously. The scalar response $y_{j,k,t}$ is the current-bin target unit count or target PC score.

### OLS

Restricted model:

$$
y_{j,k,t} = a_{R,j} + \mathbf{b}_{R,j}^{T}\mathbf{h}_Y(k,t) + \epsilon_{R,j,k,t}
$$

Full model:

$$
y_{j,k,t} = a_{F,j} + \mathbf{b}_{F,j}^{T}\mathbf{h}_Y(k,t) + \mathbf{c}_{F,j}^{T}\mathbf{h}_X(k,t) + \epsilon_{F,j,k,t}
$$

### Poisson GLM

For a target unit count:

$$
y_{j,k,t} \sim \mathrm{Poisson}(\mu_{j,k,t})
$$

Restricted model:

$$
\log \mu_{R,j,k,t} = a_{R,j} + \mathbf{b}_{R,j}^{T}\mathbf{h}_Y(k,t)
$$

Full model:

$$
\log \mu_{F,j,k,t} = a_{F,j} + \mathbf{b}_{F,j}^{T}\mathbf{h}_Y(k,t) + \mathbf{c}_{F,j}^{T}\mathbf{h}_X(k,t)
$$

Here, $\mu$ is the expected **count per bin**, not a rate in spikes/s. Do not z-score the count response. All bins in one fit have the selected fixed duration.

For HPC → PFC, $Y$ is PFC and $X$ is HPC; reverse them for PFC → HPC.

### Full/restricted matching requirements

- Independently fit/refit all coefficients, including the intercept, in each model. Do not construct the restricted fit by zeroing the source coefficients of the full fit.
- Use **identical training rows and identical test rows** for each full/restricted pair. Build the shared eligibility mask from all activity needed by the full comparison.
- Removing the source population must not restore rows that were missing its activity.
- Use the same predictor definitions, temporal parameters, target population, and folds when comparing model families or representations, wherever those comparisons are valid.
- Keep source-region history as one complete added population; do not replace it with a single average regional signal or a selected pair of units/PCs.

With $N_Y$ target-region features, $N_X$ source-region features, and model order $p$, the full design has $1+p(N_Y+N_X)$ coefficients per target. Check estimability rather than silently regularizing or selecting fewer features.

## Cross-Validation

Use **one five-fold GroupKFold evaluation based on existing behavioral-block identifiers**. Build the trial-to-fold assignment at the session level before condition-specific filtering. Keep complete blocks intact and all bins from a trial in the same fold.

Reuse the same fold assignments across restricted/full models, directions, model families, representations, and condition/window comparisons. Within each fold, train and evaluate a particular condition model only on that condition's respective training and test trials. The shared training-only PCA pool is defined separately above.

Use the ordinary deterministic grouping provided by the existing implementation or a nonshuffled GroupKFold configuration. Do not add repeated CV, seed searches, or automatic fallbacks to random bin-level splitting.

If the requested session/configuration cannot support five valid folds, report the affected result as unavailable. Existing explicit fold-count controls may be reused, but do not silently switch the requested evaluation policy.

All fitted preprocessing uses training observations only. Since the models are unpenalized and parameters are prespecified, **no inner CV or tuning loop is required**.

Grouped holdout defines the evaluation split; it does not establish mechanistic causality or resolve all temporal confounds. It does not introduce a temporal-purging requirement.

### Score aggregation

For each target and condition/window:

1. Calculate the restricted and full metrics on the same test observations in each fold.
2. Calculate their paired difference **within that fold**.
3. Use the arithmetic mean across requested folds for the target's primary absolute scores and incremental score.
4. Retain every fold's scores, row counts, and status.
5. Summarize these target-level means across target units or PC ranks using median and IQR.

Require valid paired results in every requested fold for a primary CV summary. Do not silently average only the surviving folds; preserve their values as explicitly incomplete inspection data instead. Failure of one target/configuration must not stop independent valid results.

For PCs, this is a summary by fold-specific component rank, not a pooled prediction of a fixed all-session PC. Within-target fold averaging and across-target population summarization are distinct operations.

## Linear Inter-Regional Prediction: CV $R^2$

For a particular target, fold, condition, and window, let $\bar y_{\mathrm{test}}$ be the mean of that target's evaluated test responses. For model $m$, restricted or full:

$$
SSE_m = \sum_{i\in\mathrm{test}}(y_i-\hat y_{m,i})^2
$$

$$
SST_{\mathrm{test}} = \sum_{i\in\mathrm{test}}(y_i-\bar y_{\mathrm{test}})^2
$$

$$
R^2_m = 1 - \frac{SSE_m}{SST_{\mathrm{test}}}
$$

Here, $i$ indexes eligible **trial × target-bin rows**, not trials alone.

The denominator uses the mean of the evaluated target values, following the standard scoring convention. This is a reference for scoring, not a predictive model trained on test data. No test information enters fitted coefficients or PCA.

Define the paired increment:

$$
\Delta R^2_j = R^2_{\mathrm{full},j} - R^2_{\mathrm{restricted},j}
$$

Equivalently, within a fold:

$$
\Delta R^2_j = \frac{SSE_{\mathrm{restricted},j}-SSE_{\mathrm{full},j}}{SST_{\mathrm{test},j}}
$$

Apply this in both HPC → PFC and PFC → HPC directions. Preserve negative finite absolute scores and increments. A negative increment means adding the source history worsened held-out prediction under the chosen fit.

Label the main output **Incremental CV $R^2$**. Retain the absolute restricted and full CV $R^2$ as inspectable outputs. A constant target within the evaluated test set has undefined $R^2$; do not convert it automatically to zero or one.

## Poisson Inter-Regional Prediction: CV Deviance Explained

Use **CV deviance explained**, not $R^2$, for the Poisson-specific prediction summary.

For observed counts $y_i$ and strictly positive finite expected counts $\hat\mu_i$, Poisson deviance is:

$$
D = 2\sum_i \left[y_i\log\left(\frac{y_i}{\hat\mu_i}\right) - (y_i-\hat\mu_i)\right]
$$

Define $y_i\log(y_i/\hat\mu_i)=0$ when $y_i=0$. For each model, evaluate this deviance on its held-out predictions using the same test rows.

The null scoring prediction is the constant **mean count of those evaluated test observations**, $\bar y_{\mathrm{test}}$. Let $D_{\mathrm{null,test}}$ be its Poisson deviance. This mean is a scoring reference only, not used in fitting either predictive model.

For model $m$:

$$
D_{\mathrm{explained},m} = 1 - \frac{D_m}{D_{\mathrm{null,test}}}
$$

For each target unit, define the paired increment:

$$
\Delta D_{\mathrm{explained},j} = D_{\mathrm{explained,full},j} - D_{\mathrm{explained,restricted},j}
$$

Equivalently, within a fold:

$$
\Delta D_{\mathrm{explained},j} = \frac{D_{\mathrm{restricted},j}-D_{\mathrm{full},j}}{D_{\mathrm{null,test},j}}
$$

Use the same denominator for restricted and full models. Its denominator is **null-model deviance**, not reduced-model deviance; this is not the single-unit selectivity spec's partial-deviance measure.

Apply both directions and the fold-aggregation rule above. Preserve negative finite scores and increments. Zero/undefined null deviance makes that normalized score unavailable, rather than zero by convention.

Label the main output **Incremental CV deviance explained**. Retain absolute restricted/full deviance-explained scores and deviances. Do not label this metric $R^2$, or equate its numerical size with an OLS $R^2$ increment.

## Comparing Linear and Poisson Models

Use **held-out spike-count MSE** as the common-scale comparison for unit responses. Retain model-specific $R^2$ and deviance-explained outputs for their own summaries.

Within each test fold:

$$
\mathrm{MSE}_m = \frac{1}{n_{\mathrm{test}}}\sum_{i\in\mathrm{test}}(y_i-\hat y_{m,i})^2
$$

Compare OLS and Poisson on the same target unit, response counts, predictor configuration, rows, and folds. Compare full models with full models and restricted models with restricted models. Retain both sets of MSE values, using full-model MSE for the main model-family inspection.

Use the mean of the fold-level MSE values for the primary target-level comparison, consistent with this spec's fold-wise scoring policy. Keep fold sample counts and individual errors/scores available for inspection rather than silently changing aggregation between models.

Do not round expected counts or clip negative OLS predictions before calculating errors. Those transformations would change the model being evaluated. Poisson predictions must remain valid positive finite means.

Label this **Exploratory held-out prediction comparison — spike-count MSE**. Lower MSE indicates smaller squared error on these observations; it is **not a formal statistical test of model superiority** and does not validate either response-distribution assumption. No model-family comparison is needed for PC responses because they use OLS only.

## Granger Prediction — Descriptive In-Sample Quantification

Use the same historical predictor construction, target definitions, conditions, windows, and model families as above, but fit restricted/full models separately on **all eligible observations for that configuration**.

For PC Granger, use the separate descriptive all-data regional PCA basis specified above. Do not mix all-data PCA into CV prediction.

Granger magnitudes in this section are **in-sample descriptive quantities**. They are not computed by pooling held-out residuals, not averaged from arbitrary two-series helper tests, and not subject to the CV-score requirement.

Do not add formal Granger p-values, F-tests, chi-squared significance labels, surrogate tests, or multiple-comparison correction in this first pass. A nonzero descriptive magnitude is not itself a significance declaration.

### Linear Granger: scalar-target log residual-variance ratio

For each target unit or PC $j$:

$$
G_j = \log\left(\frac{\hat\sigma^2_{\mathrm{restricted},j}}{\hat\sigma^2_{\mathrm{full},j}}\right)
$$

Calculate both residual variances as SSE divided by the **same number of fitted rows $n$**, not by different residual-degrees-of-freedom denominators. Thus:

$$
G_j = \log\left(\frac{SSE_{\mathrm{restricted},j}}{SSE_{\mathrm{full},j}}\right)
$$

Use the natural logarithm. Each target is scalar, but each regional history is multifeature. This is not a self-history-only pairwise neuron analysis.

Label the result **Linear Granger magnitude — in-sample log residual-variance ratio**. It describes improvement over the entire included target-region history, not anatomical influence.

### Poisson Granger-style: likelihood/deviance improvement

For each target unit $j$, let $\ell_{\mathrm{full},j}$ and $\ell_{\mathrm{restricted},j}$ be the fitted log likelihoods on the same observations:

$$
LR_j = 2\left(\ell_{\mathrm{full},j}-\ell_{\mathrm{restricted},j}\right)
$$

The equivalent deviance expression is:

$$
LR_j = D_{\mathrm{restricted},j}-D_{\mathrm{full},j}
$$

Retain the raw likelihood-ratio statistic and $n$, but display the **per-observation improvement** for comparisons between conditions with different observation counts:

$$
G_{\mathrm{Poisson},j} = \frac{D_{\mathrm{restricted},j}-D_{\mathrm{full},j}}{n}
$$

Here, $n$ is the number of eligible **trial × target-bin rows** used in the matched fit. Label the displayed value **Mean deviance improvement — Poisson Granger-style, in-sample**.

This is a binned spike-count GLM comparison, not an implementation of a continuous-time point-process model. Do not equate its numerical scale with the linear Granger log ratio or with incremental normalized deviance explained.

### Interpretation and implementation boundaries

With valid nested unpenalized fits optimized on the same observations, adding predictors cannot worsen the optimized in-sample fit. Positive fitted-model improvements can arise simply from added flexibility; without a formal test, they must not be labeled statistically significant.

Use ordinary existing numerical tolerances for roundoff. A substantive negative in-sample improvement is a fit/consistency problem to report, not a meaningful negative Granger influence. This is distinct from negative held-out increments, which remain valid results.

A two-series Granger helper is not a substitute for the required target-population versus target-plus-source-population regression design. Do not silently reduce a region to one trace.

For lag $\ell>1$, preserve the requested history set and label the result a **lag-restricted Granger-style comparison**. Do not silently fill in omitted recent lags to produce a conventional VAR history.

The population median of per-target Granger magnitudes is **not joint multivariate regional Granger causality**. Do not add a residual-covariance determinant statistic or a separate multivariate framework.

## Population Prediction Summary

For each direction, condition, window, representation, and model, retain the full target distribution.

### Cross-validated prediction

Each point represents one target unit's or PC rank's **mean paired increment across folds**:

- Linear: $\Delta R^2_j$.
- Poisson: $\Delta D_{\mathrm{explained},j}$.

Calculate target-level increments before population summarization. Do not subtract the median restricted score from the median full score; that is generally a different quantity.

Show individual targets, their median, and their IQR. Preserve absolute restricted/full scores as selectable or coordinated inspection views with the same target identities.

### Descriptive Granger

Show each target's in-sample Granger magnitude, plus median and IQR. Label the population summary as **median target-unit Granger magnitude** or its PC equivalent, not a joint region-level Granger statistic.

For PCs, distinguish **fold-specific PC-rank summaries** in the CV figure from **descriptive session-PC targets** in the Granger figure.

In all population plots, IQR measures variability across targets, not a confidence interval for regional influence. Report contributing/available target counts and exclusions; do not let unequal validity across models or directions become invisible.

## Main Prediction Figure

Show all requested behavioral conditions together horizontally. Within each condition, display **HPC → PFC** and **PFC → HPC**, with individual target points and median/IQR overlays. Use greater spacing between conditions than between directions within a condition.

Primary y-axis labels:

- **Linear:** Incremental CV $R^2$.
- **Poisson:** Incremental CV deviance explained.

Provide access to absolute restricted/full performance and the target-level distributions, not just the median increment. Keep a stable mapping from plotted points to unit identifiers or PC ranks.

Controls:

| Control | Options |
|---|---|
| Representation | Units — default; PCs |
| Model | OLS — default; Poisson GLM for units only |
| Alignment | Choice — default; trial start |
| Window | Before; after; whole |
| Conditions | Existing behavioral condition and choice/context selectors |
| Neural bin size | 500 / 100 / 50 / 20 ms; default 100 ms |
| Prediction lag | Positive integer; default 1 |
| Model order | Positive integer; default 1 |
| PCA count | Configurable per region; default up to 10 |

Do not overlay unlike metrics on one numerical axis or display every bin-size/lag configuration simultaneously by default. Existing controls are sufficient; no new parameter-search dashboard is required.

## Granger Summary Figure

Create a separate figure from CV predictive-performance plots, with the same all-condition, two-direction, individual-point/median/IQR structure.

- **Linear:** per-target in-sample log residual-variance ratio.
- **Poisson:** per-target in-sample mean deviance improvement; retain raw $LR$ for inspection.

Use matching condition, window, representation, bin-size, lag, and order controls where applicable. Show the in-sample/descriptive label clearly. Do not put Granger magnitudes and CV increments on the same numerical axis or add significance stars.

## Minimal Validity and Unavailable Results

Apply basic calculation checks, not a new diagnostic or recovery framework.

- Preserve the common full/restricted row mask and paired evaluation observations.
- Require a full-rank design and enough training/fitted observations to leave positive residual degrees of freedom, accounting for the intercept and historical feature count.
- Flag rank-invalid or insufficient-data configurations. Do not silently drop predictors, regularize, or switch to PCs instead of units.
- Check fit convergence and finite coefficients/predictions. Poisson means used for deviance must be positive and finite.
- Constant training responses and otherwise unidentified fits are unavailable for that target/configuration.
- Treat zero/undefined $SST$ or null deviance as unavailable for the corresponding normalized score. Do not silently apply library replacements that turn undefined scores into zero or one.
- A held-out fold with a constant response may still have defined MSE; retain that separately if its paired model predictions are valid. Metric-specific invalidity does not erase other defined metrics.
- Treat zero residual variance in a Granger ratio or other undefined Granger calculations as unavailable with the reason, rather than forcing a finite value with an invented denominator.
- Preserve negative finite CV scores/increments. Do not round counts or clip OLS predictions.
- Require valid paired requested-fold results for primary CV summaries and valid paired model-family results for direct MSE comparisons. Retain incomplete fold-level outputs as incomplete, not as an unlabeled partial mean.
- Continue independent valid targets, conditions, representations, and metrics. Show unavailable results distinctly from genuine zero effects.

A matrix-rank check is a basic estimability check. Detailed collinearity treatment, automatic feature pruning, sparse model searches, and fitting recovery systems are outside this inspection spec.

Use the existing unit-selection and data-quality conventions. Record effective unit/PC counts, eligible trials and rows, requested/valid folds, and exclusion reasons. A compact result record and ordinary warnings are sufficient.

## Default Analysis and Output Record

The standard inspection configuration is:

| Setting | Accepted value |
|---|---|
| Session scope | One simultaneously recorded session |
| Representation | Units |
| Unit response | Spike counts per bin |
| Model | Unpenalized OLS with intercept; unpenalized log-link Poisson available |
| Alignment | Choice |
| Available windows | Before −2 to 0 s; after 0 to +2 s; whole −2 to +2 s |
| Fit by condition | Separate fits per condition/window |
| Neural bin size | 100 ms; unsmoothed |
| Prediction lag / model order | 1 / 1 |
| History boundaries | Same original trial and selected window |
| PCA alternative | Separate regional training-only shared-temporal bases; up to 10 PCs per region |
| PCA scaling | Z-score input units once; no whitening or PC-score rescaling |
| CV | One five-fold behavioral-block GroupKFold assignment |
| CV aggregation | Mean of valid paired fold scores/increments per target, requiring all requested folds |
| Linear Granger | In-sample $\log(SSE_R/SSE_F)$ per target |
| Poisson Granger-style display | In-sample $(D_R-D_F)/n$ per target, with raw $LR$ retained |
| Formal Granger inference | None |
| Population display | Individual target points, median, IQR |
| Model-family comparison | Paired held-out spike-count MSE; exploratory, not a formal significance test |

Retain enough ordinary metadata to identify session, unit/PC rank, source and target regions, condition and split, alignment/window, feature counts, temporal parameters, model family, CV assignments and status, PCA scope, and whether a metric is held-out or in-sample. Do not add a separate provenance or transaction framework.

With the temporal defaults, HPC → PFC asks:

> Does observed HPC activity in the immediately preceding 100-ms bin improve prediction of current PFC activity beyond all included PFC activity in that preceding bin?

The corresponding PFC → HPC analysis reverses source and target, with all other conventions unchanged.

## Interpretation and Deferred Temporal Re-Evaluation

The outputs characterize added predictability in the **recorded populations under the selected representation and model**. They do not establish direct interaction or intervention-level causality.

Potential explanations include shared event-locked responses, movement-related activity, common input, different response timing across regions, and slow temporal fluctuations. For example, an earlier HPC response can help predict a later PFC response when both are driven independently by the same event.

Cross-validation tests generalization under the selected splits; it does not identify the biological cause of that predictability. Grouped folds do not remove all such explanations. Descriptive in-sample Granger improvements also depend on model flexibility and the number of fitted features.

Directional comparisons depend on the recorded unit populations, retained PCs, source/target dimensionality, and baseline target predictability. A larger median increment in one direction is not by itself evidence of stronger anatomical influence. Do not interpret the population median as a joint regional causal statistic.

Include a short note in relevant figures or result descriptions:

> First-pass predictive association. Shared event responses, common input, movement, and slow temporal fluctuations have not been separately controlled. Granger magnitudes are descriptive in-sample values, not significance tests or evidence of mechanistic causality.

**Re-evaluate temporal fluctuations and across-trial dependence before confirmatory or publication-level interpretation.** Retain this future-analysis reference: Kenneth D. Harris, [*Nonsense correlations in neuroscience* — bioRxiv, version 3](https://www.biorxiv.org/content/10.1101/2020.11.29.402719v3.full), DOI: 10.1101/2020.11.29.402719.

These limitations are caveats and deferred work, **not implementation blockers**. Do not automatically add detrending, behavioral covariates, autoregressive residual corrections, neuron-count matching, surrogate procedures, or temporal purging in this first pass.

## Deferred Analyses

Keep the following outside this specification's current implementation:

- Mutual information.
- Communication subspaces, CCA, and other latent/shared-subspace analyses.
- Joint multivariate regional Granger statistics.
- Formal Granger significance tests and multiplicity correction.
- Temporal-confound models, behavioral-confound adjustment, and alternative surrogate/null analyses.
- Regularization searches, automatic predictor selection, or model-order/lag optimization.

## Codebase Verification and Implementation Scope

The scientific conventions above are agreed for inspection. Remaining verification is ordinary codebase integration:

- Verify session/unit identities, original trial indices, alignment timestamps, behavioral-block IDs, and condition/choice/context definitions.
- Reuse existing spike binning, validity masks, regional unit selection, plotting controls, and descriptive display conventions where they match the specified analysis.
- Verify counts, bin boundaries, within-trial lag indexing, and the stated 19/39 usable-row examples under the default settings.
- Verify that restricted/full designs include the complete intended populations and match rows; do not substitute a two-series Granger helper.
- Verify unpenalized estimator/intercept settings, group assignments, training-only PCA, and score baseline/aggregation conventions against the actual installed code.
- Check basic score/equation consistency, ordinary fit failures, and availability reporting without adding an adversarial or publication-finalized validation system.

Do not modify the scientific model silently to make a difficult configuration produce a number. Flag unavailable configurations while preserving independent valid results. Keep the implementation modular enough for later refinement without implementing the deferred analyses now.

## Revision Summary

This revision incorporates the accepted review recommendations: explicit trial × bin observations; complete regional predictor populations; separate condition/window fits; within-trial/window history boundaries; count responses and unpenalized models; shared-temporal regional PCA with fold-specific target interpretation; five grouped folds and explicit scoring references; separate in-sample scalar-target Granger formulas; common-count MSE model comparison; per-target population summaries; minimal validity handling; and deferred temporal/confound caveats.

It replaces the earlier ambiguous estimator, observation, baseline, CV, and Granger instructions. The original bidirectional prediction goal, unit/PC alternatives, temporal-parameter controls, and all-conditions summary format are retained.
