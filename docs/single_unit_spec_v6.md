# Single-Unit Selectivity Analysis Spec

> Working revision v6: incorporates the agreed first-pass inspection defaults. Predictor-coding examples are illustrative only: Codex must verify the existing source codes and choose an appropriate, documented model encoding. Effect maps are in-sample; the unit-level OLS–Poisson MSE comparison is held out. Significance remains provisional, with temporal-fluctuation controls and broader multicollinearity work explicitly deferred. The former open review points are replaced by an accepted-defaults summary and codebase verification tasks.

## Goal

Inspect how individual units and major population dimensions relate to task variables and decision variables. Estimate each predictor's **unique contribution conditional on the other selected predictors**, where that contribution is identifiable, using a multivariable model.

This is a preliminary inspection framework, not a publication-finalized analysis or a guarantee of clean attribution among related behavioral variables. The aim is to establish usable fits and summary figures that can be refined after inspecting the data.

The core analysis is:

$$
y_{\mathrm{neural}}
\sim
\mathrm{state} +
\mathrm{action} +
\mathrm{reward}_{i} +
\mathrm{reward}_{i-1} +
\mathrm{DV}_1 +
\mathrm{DV}_2 +
\cdots
$$

where all selected predictors are included simultaneously.

The purpose of the multivariable model is to estimate how much each predictor contributes after accounting for the other included variables. These conditional contributions do not, by themselves, establish distinct biological representations for overlapping task variables.

## Analysis Windows

Fit models separately for:

- **Before:** −2 to 0 s relative to the alignment event
- **After:** 0 to +2 s
- **Whole:** −2 to +2 s

The summary figures should allow selection among these three windows. Use the existing **choice / trial-start alignment** selector, with **choice as the default**. Analyze each session separately.

No time-resolved selectivity output or "selective at any time during the trial" summary is required. The selected ±2 s windows do not overlap between trials by task design; no overlap-detection or boundary-trial exclusion machinery is needed.

## Predictor Set

Use the following requested predictor set **by default**, with individual predictors manually selectable. These are **predictors of neural activity**, not separate decoding targets. The response remains the selected unit's spike count or the selected PC response.

| Predictor | Requested variable or definition | Trial timing |
|---|---|---|
| Task/context state | Actual task context governing trial $i$; map to the existing state field | State on trial $i$ |
| Action | Choice made on trial $i$; map to the existing action field | Action on trial $i$ |
| Current reward | Reward outcome on trial $i$; map to the existing reward field | Outcome on trial $i$ |
| Previous reward | The same reward measure from the immediately preceding trial | Outcome on trial $i-1$ |
| Value | `FQl_rel_value` | Value entering trial $i$ |
| Belief | `HMM_rel_value_logodds` | Value entering trial $i$ |
| Doubt | `relative_doubt_index` | Value entering trial $i$ |
| Consecutive rewards | `consecutive_rewards` | History entering trial $i$ |
| Relative value | `relative_value` | Value entering trial $i$ |
| Consecutive omissions | `consecutive_omissions` | History entering trial $i$ |
| Relative omissions | `relative_omissions` | Value entering trial $i$ |

### Timing and Source Mapping

- Keep **current reward and previous reward as separate predictor terms**. Do not merge them into one generic reward/history variable.
- All listed values, beliefs, doubt measures, and history DVs describe the state **entering trial $i$**, before incorporating that trial's outcome. Use the same predictor definitions for before, after, and whole-window neural responses; do not change their update timing with the selected neural epoch.
- Derive previous reward from the original chronological trial sequence within the session, before analysis-specific filtering. A missing previous-trial value is unavailable, not zero and not the last retained trial's outcome. Apply the agreed identical-trial rule to full/reduced fits.
- The source identifiers above preserve the user's requested spelling, including `FQl_rel_value`. Codex must check the existing codebase for the actual column names, numerical representations, and update/reset conventions, and document any mapping. Do not silently substitute another value/belief model or shift a column without checking how it is already aligned.
- `FQl_rel_value` and `relative_value` are intentionally listed separately. Similar names do not establish that they are identical, nor should they be silently merged. Any actual exact redundancy is handled only by the agreed minimal rank check.
- Do not automatically add other decoding targets, value models, or history variables to this simultaneous predictor set. Include **main effects only**, with an intercept: no automatic interactions, polynomial terms, additional lags, or predictor selection. Choose predictor coding as specified below.

Current-trial reward is an outcome-associated predictor even in the before-event model. A relationship with before-event activity should be described as an association with the trial's eventual reward outcome, not automatically as a neural response to reward delivery.

### Predictor Coding — Verify the Codebase, Then Choose an Appropriate Encoding

**Codex must inspect the existing codebase before assigning predictor codes.** Verify source values, category meanings, left/right sign conventions, reward semantics, and missing-value codes. Choose an appropriate regression encoding consistent with those verified meanings, and document the mapping from source values to model columns.

The following is an **illustrative encoding, not a mandatory mapping or a claim about the existing data**:

| Predictor | Example model encoding |
|---|---|
| State | Left = −1; right = +1 |
| Action | Left = −1; right = +1 |
| Current reward | No reward = 0; reward = 1 |
| Previous reward | No reward = 0; reward = 1 |
| Listed numerical DVs | Preserve the verified variable's existing numerical values |

For example, if the verified source uses a different left/right code, Codex may retain a suitable existing binary convention or explicitly map it to a suitable contrast. It must not assume that a numeric sign or integer label means left, right, rewarded, or missing. Keep the meanings consistent across windows, units, PCs, and response-model families; label the resulting coefficient direction/reference category.

Represent a binary predictor with one appropriately coded column alongside the intercept; avoid redundant category columns. If a selected predictor genuinely requires multiple category contrasts, use an appropriate nonredundant encoding and retain the agreed joint test of its complete term. Do not add category expansions that the current predictors do not require. Numerical DVs remain numerical predictors; coefficient standardization is the post-fit display conversion below, not a reason to change their definitions.

An ambiguous or unavailable source mapping must be reported, not guessed or silently replaced with a different variable. Source verification is an implementation task, not an invitation to redesign the predictor set.

### Overlapping Predictors and Inspection Scope

The listed variables are expected to contain overlapping information and may be strongly collinear. The analysis should remain usable for preliminary inspection without automatically trying to disentangle that overlap.

- Preserve valid fits and conditional effect estimates for full-rank configurations; strongly correlated predictors alone are not a new exclusion rule.
- Do not promise that individual coefficients or partial effects isolate distinct biological variables. Shared information can make the apparent contribution depend strongly on which other predictors are selected.
- Retain the existing minimal rank check for exact linear redundancy. A rank-deficient configuration is unavailable for individual-predictor selectivity, rather than evidence of zero selectivity. The user may select a different predictor subset explicitly; the pipeline must not do so automatically.
- Do not add automatic predictor pruning, alternative attribution methods, or an expanded multicollinearity workflow. Refining the predictor set and interpretation is deferred until after preliminary inspection.

## Unit-Level Response Models

Provide two model options for individual units.

### Fitting Policy — In-Sample Conditional Effects

Use **unpenalized OLS** and **unpenalized Poisson GLMs with a log link**, both with an intercept and the selected main effects. OLS PC-response models use the same unpenalized, intercept-containing policy. Do not import elastic-net or other regularization from the decoding analysis, tune a penalty, or silently use a penalized estimator default.

For the primary coefficient and partial-effect maps, fit the full and reduced models on **all eligible trials** for that response/configuration and evaluate their effects on those same trials. Label these outputs **In-sample conditional effect**. Statistical reliability is assessed separately by the specified first-pass null procedure.

Do not add an in-sample/CV toggle for the selectivity maps. Cross-validation is used only for the separate unit-level **Exploratory held-out prediction comparison** below. Within that comparison, fit on each training fold; within a null resample, refit on the generated null response. The unpenalized model family remains unchanged.

### Trial-Level Response — One Observation per Trial

For each unit and selected analysis window, use **one response per eligible trial: the total spike count within that window**.

- Before, after, and whole windows are fitted separately. Each supplies one spike-count response per trial.
- With $N$ eligible trials, the response vector has $N$ entries and the predictor matrix has $N$ rows. Do not treat time bins within a trial as additional independent observations in this summary analysis.
- Use the same count response for OLS and Poisson and for their held-out MSE comparison. Firing rate may be shown descriptively, but it is not a separate response-model toggle here.
- Do not z-score the count response before the Poisson fit. Coefficient standardization is a post-fit display conversion, defined under **Coefficient Scales and Interpretation**, not an additional response-preprocessing step.

The count-response definition applies to individual units. For PCs, use one window-mean PC score per trial, as defined under **Summary Figure 3: PC × Variable Selectivity**.

### OLS / Gaussian Linear Model

Use the unit's total spike count per trial in the selected analysis window as the response.

For each predictor report:

- **Standardized coefficient**
- **Partial $R^2$**
- **First-pass Freedman–Lane permutation-test significance**, with BH-FDR correction and the provisional interpretation described under **Statistical Selectivity**

The standardized coefficient preserves the direction of selectivity. Its formula and interpretation are defined under **Coefficient Scales and Interpretation**.

The unique contribution of predictor $X_j$ is measured by comparing the full model to the same model with $X_j$ removed.

For OLS:

$$
R^2_{\mathrm{partial},j}
=
\frac{
SSE_{\mathrm{reduced},j} -
SSE_{\mathrm{full}}
}{
SSE_{\mathrm{reduced},j}
}
$$

Label this quantity explicitly as:

**Partial $R^2$**

### Poisson GLM

Provide a Poisson GLM with a **log link** as an alternative model for spike-count responses. Its coefficients describe changes in log expected count; use the distinct display scale defined under **Coefficient Scales and Interpretation**.

Fit the same predictor set:

$$
y_{\mathrm{spikes}}
\sim
\mathrm{state} +
\mathrm{action} +
\mathrm{reward}_{i} +
\mathrm{reward}_{i-1} +
\mathrm{DV}_1 +
\cdots
$$

For each predictor report:

- **Log-count coefficient per predictor SD**, with the raw fitted coefficient retained
- **Partial deviance explained**
- **First-pass reduced-model Poisson parametric-bootstrap significance**, with BH-FDR correction and the provisional interpretation described under **Statistical Selectivity**; do not reuse OLS residual permutation for counts

Calculate the unique contribution of predictor $X_j$ by comparing the full model with the refitted reduced model that excludes that predictor term:

$$
D_{\mathrm{partial},j} =
\frac{D_{\mathrm{reduced},j} - D_{\mathrm{full}}}
{D_{\mathrm{reduced},j}}
$$

Here, $D_{\mathrm{full}}$ and $D_{\mathrm{reduced},j}$ are the Poisson deviances of the full and reduced models, evaluated on the same observations. The ratio is defined when $D_{\mathrm{reduced},j} > 0$.

This measures the fraction of deviance remaining in the reduced model that is removed by adding predictor $X_j$. For example, a value of 0.25 means that adding the predictor removes 25% of the reduced model's remaining deviance, not 25% of spike-count variance.

Label this quantity explicitly as:

**Partial deviance explained**

Do not label partial deviance explained as $R^2$. Its denominator is the **reduced-model deviance**, not the intercept-only/null-model deviance. Incremental overall deviance explained uses a different denominator; it is not an additional required metric or output in this single-unit spec.

## Coefficient Scales and Interpretation

Fit the response model in the chosen data units, retain the raw coefficients, and calculate the following display conversions after fitting. These conversions do **not** require another model fit or an additional response-standardization step.

For one fitted unit/window model:

| Symbol | Meaning |
|---|---|
| $i$ | Trial index |
| $j$ | Predictor-column index; a multicolumn predictor term still has one joint partial-effect test |
| $X_{i,j}$ | Value of predictor column $j$ on trial $i$ |
| $y_i$ | Observed spike count on trial $i$ |
| $\beta_j$ | Raw fitted coefficient for predictor column $j$; Greek beta, not the resampling count $B$ |
| $s_{X_j}$ | Sample standard deviation of predictor column $j$ across the eligible fitting trials |
| $s_y$ | Sample standard deviation of the observed spike counts across those same trials |

The $s$ values are data standard deviations, **not coefficient standard errors or residual standard deviations**. Use the same fitting-trial set for the model and its coefficient-display scales. The intercept is not standardized.

### OLS: Standardized Coefficient

$$
\beta_{\mathrm{std},j} = \beta_j \frac{s_{X_j}}{s_y}
$$

Interpretation: a one-standard-deviation increase in predictor $j$ is associated with a $\beta_{\mathrm{std},j}$-standard-deviation change in the predicted spike count, conditional on the other predictors. For example, a raw slope of 2 spikes per predictor unit, $s_{X_j}=0.5$, and $s_y=4$ give a standardized coefficient of 0.25.

Label the display **Standardized coefficient (OLS)**. Preserve sign. A value of 0.25 does not mean 25% of the variance is explained; partial $R^2$ is the separate effect-contribution measure.

### Poisson: Log-Count Coefficient per Predictor SD

$$
\beta_{\mathrm{perSD},j} = \beta_j s_{X_j}
$$

Under the log link, this is the change in **log expected spike count** for a one-SD increase in the predictor, conditional on the other predictors. Do not divide by $s_y$: this is not the OLS response-SD scale.

For interpretation, the corresponding multiplicative change in expected count is:

$$
\mathrm{Count\ ratio}_j = \exp\left(\beta_j s_{X_j}\right)
$$

For example, $\beta_j=0.4$ and $s_{X_j}=0.5$ give a log-count change of 0.2 and a count ratio of approximately 1.22, or 22% higher expected count. This explanatory ratio does not require a new summary figure.

Label the coefficient display **Log-count coefficient per predictor SD (Poisson)**. Preserve sign and do not present it as numerically equivalent to the OLS standardized coefficient.

### Scope and Availability

For a nonconstant binary predictor, the same column-level scaling can be calculated, but document its verified coding: a one-SD change is not generally a complete switch between its two classes. Follow **Predictor Coding — Verify the Codebase, Then Choose an Appropriate Encoding** rather than assuming that the illustrative signs apply. A multilevel predictor has separate contrast coefficients rather than one arbitrary signed variable-level coefficient; its partial effect and significance test remain joint across the complete term.

Do not convert an undefined coefficient scale, such as OLS standardization with $s_y=0$, into a zero coefficient. Mark that coefficient display unavailable. This does not introduce a new predictor-selection or multicollinearity workflow.

## Shared Full/Reduced Model Comparison Requirements

Apply these requirements to OLS unit models, Poisson unit models, and OLS PC models, including corresponding refits within the agreed null procedures:

1. **Refit the reduced model.** Remove the predictor term being tested and refit all remaining coefficients. Do not approximate the reduced fit by setting the predictor's coefficient to zero while retaining the other full-model coefficients.
2. **Use identical eligible observations.** For each response and model configuration, select the eligible trials using the full selected predictor set and response. Use those same trials for the full model and all corresponding reduced models. Removing a predictor must not restore trials that were excluded because that predictor was missing. Evaluate each full/reduced effect comparison on the same observations.
3. **Remove and test the complete predictor term jointly.** When one task variable is represented by multiple design-matrix columns, remove all columns belonging to that term together and report one variable-level unique contribution and test. Do not treat its separate coding columns as independent task variables.

These requirements apply to the in-sample partial-effect fits and the corresponding refits in every null dataset. The fitting scope and predictor-coding policy are specified above. Model-specific first-pass null generation is defined below.

## Comparing OLS and Poisson Models

Allow OLS and Poisson GLM results to be inspected separately.

Do not determine which model performs better by directly comparing:

- Partial $R^2$
- Partial deviance explained

because these are different metrics.

### Exploratory Held-Out Prediction Comparison

Compare the models using **held-out mean squared prediction error (MSE)** on the same unit spike-count responses:

$$
\mathrm{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2
$$

Here, $y_i$ is the observed spike count and $\hat{y}_i$ is the predicted expected count for the same trial and analysis window. Lower MSE indicates smaller squared prediction error.

For each comparison:

- Use the same unit, analysis window, selected predictor set, eligible trials, and train/test splits for both models.
- Fit both models on the training observations and evaluate their predictions on the same held-out observations.
- Express both predictions in spike-count units, matching the agreed count response for both model families.
- Retain the OLS and Poisson MSE values and the number of evaluated observations.

Label the output **Exploratory held-out prediction comparison**.

This is an exploratory predictive comparison, **not a formal statistical test of model superiority**. A lower MSE alone does not establish a statistically significant difference or validate either model's distributional assumptions.

### Held-Out Configuration and Aggregation

Use **one five-fold grouped cross-validation pass**, grouping by the existing behavioral-block identifiers within the session. Use an ordinary deterministic `GroupKFold` arrangement (no group shuffling) or the matching existing codebase utility. Keep each complete block in one fold, and use identical train/test assignments for OLS and Poisson. Each eligible trial contributes one held-out prediction per model.

- Use the same selected predictors and eligible observations for the paired models. Check training-fold fit validity using the minimal rules below.
- No repeated CV, inner tuning, adaptive fold search, or automatic switch to random trial splitting is required.
- If the session lacks five usable groups, or any requested fold cannot produce valid paired fits/predictions, mark the primary comparison unavailable with a reason. Keep the descriptive selectivity maps and any valid fold-level results available for inspection. Do not silently report a partial-fold primary score.
- This comparison is for **unit spike-count responses**. It does not require a PC-target CV analysis or refitting descriptive PCAs inside folds.

For each model, aggregate squared errors over all held-out observations:

$$
\mathrm{MSE}_{\mathrm{CV}} =
\frac{\sum_{k=1}^{K}\sum_{i\in I_k}(y_i-\hat{y}_{i,k})^2}
{\sum_{k=1}^{K}n_k}
$$

Here, $K=5$, $I_k$ is the test-trial set in fold $k$, $n_k$ is its size, and $\hat{y}_{i,k}$ is the prediction from that fold's training fit. This weights each held-out trial equally, rather than giving unequal-sized folds equal weight. Retain fold-level MSE, counts, and validity status along with the primary pooled score.

Inspect the OLS and Poisson scores side by side for the **same unit/window**. This paired MSE comparison does not compare partial $R^2$ numerically with partial deviance explained and does not add a significance test of their difference.

Keeping blocks intact here is a simple prediction-evaluation split, **not a claim that temporal confounds are solved**. It does not change the unrestricted first-pass OLS residual permutations or the independent Poisson null draws used for selectivity significance. The partial-effect maps remain in-sample.

## Statistical Selectivity

For these inspection outputs, label a unit as **selective for a predictor under the first-pass test** when its partial effect passes the specified model-appropriate null test after Benjamini–Hochberg FDR correction. This is a provisional significance classification under simplified trial assumptions, not evidence that temporal confounds have been excluded.

The intended null for predictor $X_j$ is:

> The predictor adds no explanatory information beyond the other predictors included in the model.

Use BH-FDR-adjusted significance rather than an arbitrary effect-size threshold. Distinguish the accepted OLS residual-permutation procedure from the accepted Poisson parametric bootstrap; they generate null data differently. Both may run in this first-pass inspection without waiting for a temporal-fluctuation-aware null procedure.

### OLS: Freedman–Lane Reduced-Model Residual Permutation

Use **Freedman–Lane residual permutation** for OLS responses, including individual units and PCs.

For each response and predictor being tested:

1. Fit the observed full model and the reduced model without that predictor, using the same eligible trials.
2. Calculate the observed partial $R^2$.
3. Retain the reduced model's fitted response and residuals.
4. Randomly permute the reduced-model residuals across the eligible trials for this response/model configuration within the session. For this first pass, do not constrain permutations by behavioral block or temporal distance. Treat residual exchangeability as a simplifying working assumption, not as a demonstrated property of the data.
5. Add the permuted residuals to the original reduced-model fitted response to construct a null response.
6. Refit both full and reduced models to the null response, keeping the predictor matrix and eligible trial rows unchanged.
7. Recalculate partial $R^2$ and repeat to obtain its null distribution.
8. Calculate the upper-tail p-value using the Monte Carlo counting rule below, then apply Benjamini–Hochberg FDR within the specified testing family.

This replaces the earlier instruction to arbitrarily shuffle a predictor column or trial labels. The OLS null procedure operates on the **reduced-model response residuals**, not by independently scrambling the predictor columns.

### Poisson: Reduced-Model Parametric Bootstrap

Use a **reduced-model Poisson parametric bootstrap** for unit spike-count responses. This is an accepted first-pass procedure, not a permutation test.

For each unit and predictor term being tested:

1. Fit the observed full and reduced Poisson models to the same eligible trials and calculate the observed partial deviance explained.
2. Retain the reduced model's predicted expected spike count for every trial, $\hat{\mu}_{\mathrm{reduced},i}$.
3. Generate one simulated count per trial, independently conditional on these fitted means:

$$
y_i^* \sim \mathrm{Poisson}\left(\hat{\mu}_{\mathrm{reduced},i}\right)
$$

4. Keep the actual predictor matrix and eligible trial rows unchanged. Refit both full and reduced models to the simulated counts.
5. Recalculate partial deviance explained and repeat for the specified number of null datasets.
6. Compare the observed effect with the simulated effects using the upper-tail Monte Carlo counting rule below, then apply BH-FDR within the specified testing family.

The simulation represents the null that the omitted predictor adds nothing beyond the retained predictors. It preserves their modeled effects through the trial-specific expected counts. Generate counts from the Poisson distribution; do not add shuffled ordinary residuals to those means, round residual-generated responses, or clip negative simulated responses.

These first-pass draws assume conditionally independent Poisson variability. They do not reproduce additional unexplained across-trial dependence or non-Poisson variability. Keep valid effect-size and bootstrap outputs available for inspection under the limitations below; do not mark Poisson significance unavailable solely because the more complete temporal analysis is deferred.

### First-Pass Trial Treatment and Deferred Temporal Controls

**Temporal fluctuations and across-trial dependence handling are explicitly deferred. They are not implementation blockers for this inspection spec.**

- **OLS:** use the ordinary across-eligible-trial residual permutation specified above, within the current session/configuration.
- **Poisson:** use independent simulated counts conditional on the fitted reduced-model means.
- Keep the predictor values, predictor relationships, and eligible observation set unchanged. Neither procedure claims to preserve residual temporal dependence beyond the structure explained by the reduced model.

Do not add block-restricted shuffles, drift models, temporal detrending, autoregressive residual models, or a new temporal-control workflow in this implementation. No neural-window overlap checks or boundary-trial exclusion are introduced; the known nonoverlap of the selected trial windows does not itself establish statistical independence.

Treat raw p-values, BH-adjusted p-values, significance markers, and fraction-selective summaries as **provisional inspection results**. Keep them available, but include a short note in the relevant figures or result description:

> First-pass significance; temporal fluctuations and across-trial dependence have not been accounted for. Re-evaluate before confirmatory interpretation.

**Required later re-evaluation:** revisit the fitted associations, null procedures, significance markers, and fraction-selective conclusions with methods that address slow temporal fluctuations and across-trial dependence before drawing publication-level or confirmatory conclusions. Choosing and implementing those methods is outside this spec's current scope. BH-FDR remains the required multiplicity correction, but does not repair a temporally inappropriate null model.

Reference for this future analysis: Kenneth D. Harris, [*Nonsense correlations in neuroscience* — bioRxiv, version 3](https://www.biorxiv.org/content/10.1101/2020.11.29.402719v3.full), DOI: 10.1101/2020.11.29.402719. The paper discusses how slowly varying signals can exhibit apparently significant associations even when they are unrelated. This motivates the later re-evaluation; the reference does not add an automatic requirement to implement its methods now.

### Monte Carlo P-Value Counting — Short Explanation

For each test, compare the observed statistic with the statistics from $B$ null datasets. For OLS, $T$ is partial $R^2$ from residual permutations; for Poisson, $T$ is partial deviance explained from simulated count datasets. Larger values count as more extreme in both procedures:

$$
p_{\mathrm{raw}} =
\frac{1 + n_{\mathrm{exceed}}}{B + 1}
$$

Here, $B$ is the number of null resamples for that test, and $n_{\mathrm{exceed}}$ is the number with $T_{\mathrm{null}} \ge T_{\mathrm{observed}}$. Include ties. This $B$ is not the regression coefficient $\beta$, the number of trials, or the number of units.

Example: with 999 null datasets, if 14 give an effect at least as large as the observed one, the raw p-value estimate is $(1+14)/(999+1)=0.015$. Adding one prevents a finite set of null simulations from being reported as $p=0$. For randomized permutation tests, this also has the interpretation of including the observed arrangement in the reference count. For the Poisson bootstrap, use the same finite-simulation counting convention, but do not describe it as an exact permutation test: its calibration is approximate and depends on the fitted null model. If none exceeds or ties the observation, the smallest reported value is $1/(B+1)$, not zero.

This is the p-value **before** multiple-testing correction. Its resolution depends on $B$: the accepted default of 999 resamples allows a minimum of 0.001; an explicitly selected 9,999 allows 0.0001. Limited resolution can prevent otherwise strong, isolated effects from passing BH-FDR. This is an inspection-budget limitation, not a reason to change the correction method, and a nonsignificant test does not establish absence of an association.

Apply this rule to both accepted first-pass null procedures. It does not establish OLS residual exchangeability, validate the fitted Poisson null, or address temporal dependence. Those inferential limitations are explicitly deferred as described above; they do not require a different counting formula or a stricter multiplicity correction.

### Resampling Defaults

Use **999 null datasets per test** for both OLS residual permutation and the Poisson parametric bootstrap, with **random seed 0** as the default. Retain the configured resampling count and seed with the results.

Allow an explicit alternative count, such as 9,999. Do not implement adaptive resampling, automatic budget expansion, or stopping rules in this first pass. A fixed seed supports reproducible runs; it does not validate the null assumptions.

Use the full requested number of valid null fits to calculate a p-value. If a required null fit/statistic fails or is undefined, mark the affected significance test unavailable and retain any valid observed effect. Do not silently discard unsuccessful null datasets, replace their statistics with zero, or report a p-value based only on successful simulations. No elaborate retry or rescue procedure is required.

### Multiple-Comparison Correction: Benjamini–Hochberg FDR

Use **Benjamini–Hochberg false discovery rate (BH-FDR)** correction on the valid raw p-values in each defined test family. **Do not substitute Bonferroni, Holm, or another family-wise-error correction**, and do not apply one in addition to BH-FDR.

BH-FDR targets the expected fraction of false discoveries among rejected hypotheses, rather than the probability of any false positive. Its interpretation depends on valid underlying p-values and the method's assumptions; the correction does not validate an unsuitable null procedure.

Use **FDR level $q=0.05$** by default. Retain raw and BH-adjusted p-values, and mark significance when the adjusted p-value is at or below the configured level.

Define the testing families as follows:

- **Units:** all valid unit × selected-predictor tests across the analyzed PFC and HPC populations, for one session, alignment, analysis window, selected predictor configuration, and response-model family. OLS and Poisson form separate families.
- **PCs:** all valid PC × selected-predictor tests across the analyzed PFC and HPC PC populations, for the same session/alignment/window/predictor configuration. PC OLS tests form a **separate family from unit tests**. Each included predictor term contributes one joint test per response, not separate tests of its coding columns.

Mark invalid or unavailable tests as unavailable and exclude them from the valid-test family; do not invent raw p-values for them. Retain the family definition and number of valid tests.

Correction belongs to the analysis configuration, not the visible subset of a plot. Hiding a region, unit, PC, or predictor row/column must not recalculate BH over only the displayed cells. Changing the fitted predictor set defines a different model configuration, not a display-only change.

Do not add another correction across all windows, model families, or manually explored predictor configurations for this inspection analysis. Include the caveat:

> FDR correction applies within the stated test family, not across all analyses explored.

The default 999 draws, seed 0, and temporal-fluctuation caveats apply as stated above. A zero significant count is a first-pass result under these settings, not evidence that the population lacks an association.

## Minimal Predictor-Matrix Rank Check

Check the numerical rank of each distinct task-predictor design matrix after selecting eligible trials and constructing its predictor columns. Include all fitted columns, including an intercept when present.

This checks for exact linear redundancy among **task predictors**, not the rank of the neural population or the number of retained PCs.

If the matrix is rank-deficient:

- Flag the affected model configuration as invalid for individual-predictor selectivity, and provide the reason.
- Do not present its individual-predictor coefficients, partial effects, or significance values as valid selectivity results.
- Do not silently drop predictors, add regularization, or treat a returned generalized-inverse solution as resolving the ambiguity.
- Continue other valid, independent configurations rather than aborting the entire analysis.

Reuse the same check when units or PCs share the same predictor matrix and eligible trial rows. No per-response repetition is needed when that matrix is identical.

This is a minimal validity check, not a comprehensive multicollinearity analysis. Do not add automatic variable selection, detailed condition-number diagnostics, or special handling of partially estimable effects. Detailed interpretation of strongly but imperfectly correlated predictors remains deferred.

## Eligibility and Minimal Failure Handling

Reuse the existing unit-quality selection. Do not add an arbitrary minimum spike-count threshold or a new QC/diagnostic framework. Retain eligible trial counts and total spike counts for inspection.

- Select eligible trials using the full selected predictor set and the required neural response. Exclude missing/invalid required observations without filling them with zero. Reuse this mask for all corresponding reduced models.
- Require **more eligible observations than fitted design-matrix columns, including the intercept**, and a full-rank predictor matrix. This is a minimal positive-residual-degrees-of-freedom check, not a guarantee of reliable estimation.
- For an observed response with no trial-to-trial variation, mark that response/window's selectivity analysis unavailable. Do not turn undefined coefficient scales or effects into zero selectivity.
- Check ordinary fit convergence, finite outputs, and defined effect denominators: reduced SSE must be positive for partial $R^2$ and reduced deviance must be positive for partial deviance explained. If these requirements fail, mark the affected result unavailable with a short reason.
- Keep valid observed effect estimates when only the significance calculation fails. Apply the no-silent-discard rule for failed null fits above.
- Continue independent valid units, PCs, windows, or configurations. A failure in one PC analysis does not stop valid unit analyses; shared rank-invalid predictor designs still affect every response using that design.
- For held-out MSE, require valid paired fits/predictions in every requested fold. A failed comparison does not invalidate otherwise valid all-trial effect maps.

Sparse responses, limited trial counts, or strongly correlated but full-rank predictors may still yield unstable inspection results. Report the counts and preserve the existing caveats rather than adding automated model selection, new distributions, or extensive recovery logic. The checks above identify ordinary undefined calculations; they are not publication-grade adequacy tests.

## Summary Figure 1: Fraction of Selective Units

Create a heatmap with:

- **Rows:** task variables / decision variables
- **Columns:** PFC and HPC
- **Value:** fraction of units significantly selective for that predictor under the first-pass test

Use **fraction of units** as the primary value.

For each predictor and region, calculate:

$$
\mathrm{Fraction\ selective} =
\frac{N_{\mathrm{significant}}}{N_{\mathrm{valid\ tests}}}
$$

The numerator counts units passing BH-FDR in the defined family. The denominator counts units in that region with a valid test for that predictor, not all recorded units regardless of validity.

Display the numerator and denominator (for example, **18/94 units**) and retain the number excluded and the analysis counts. If no tests are valid, show unavailable rather than zero. If 94 tests are valid and none passes, **0/94** is a valid first-pass result, subject to the temporal and resampling-resolution caveats.

A rank-invalid configuration or unavailable significance test is not a valid nonselective result. Include the provisional-significance and within-family FDR notes; this is not a temporally validated prevalence estimate.

Allow selection of:

- Before
- After
- Whole
- OLS
- Poisson GLM

This figure answers:

> What fraction of eligible units in each region pass the first-pass conditional-contribution test for each predictor under the selected model and simplified null procedure?

## Summary Figure 2: Unit × Variable Selectivity

Create a heatmap with:

- **Rows:** units
- **Columns:** task / decision variables

Primary heatmap value depends on the selected response model:

### OLS

- **Partial $R^2$**

### Poisson GLM

- **Partial deviance explained**

Allow selection of:

- Before
- After
- Whole
- Region
- OLS / Poisson GLM

An optional alternate view should display the predictor coefficients.

For OLS, use **Standardized coefficient (OLS)**. For Poisson, use **Log-count coefficient per predictor SD (Poisson)**, as defined under **Coefficient Scales and Interpretation**. Do not use one unlabeled coefficient scale for both model families.

The coefficient view should preserve sign so positive and negative tuning can be distinguished.

Show **all valid in-sample partial effects**, regardless of significance. Use an outline or marker for BH-FDR significance; do not hide nonsignificant effects behind a significance mask. Unavailable significance is distinct from a valid nonsignificant test, and unavailable effects are visually distinct from zero. Use stable unit-ID ordering grouped by region, preserving unit identity across window and model views.

This figure answers:

> Which units show conditional relationships with the selected predictors, and how large are their partial effects under this predictor set?

## Summary Figure 3: PC × Variable Selectivity

Perform an analogous selectivity analysis using neural population principal components as responses.

For each PC:

$$
PC_k
\sim
\mathrm{state} +
\mathrm{action} +
\mathrm{reward}_{i} +
\mathrm{reward}_{i-1} +
\mathrm{DV}_1 +
\cdots
$$

Use OLS for one **window-mean PC score per trial**.

### PC Response Construction

1. Start from regional unit activity in **100-ms bins by default**, using valid neural trials pooled across the full **−2 to +2 s** analysis window for the selected alignment. Reuse the existing binned-activity utility.
2. Z-score each usable unit once using observations pooled across those trials and time bins. A unit with zero variance in this input matrix cannot be z-scored; omit that input from PCA and report the usable-unit count. This does not automatically exclude that unit from independent trial-count analyses.
3. Fit separate shared-temporal PCAs for PFC and HPC. Use **one descriptive basis per session, region, and selected alignment**, reused across before, after, and whole-window summaries rather than refitted for each epoch.
4. Retain **up to 10 PCs per region by default**, with the count configurable. If fewer usable dimensions are available, retain that smaller count and report the actual count rather than failing the session or padding with zero-variance components.
5. Project the regional activity into those PC coordinates with **whitening disabled**. Use ordinary PC scores directly: no additional PC-score z-scoring, post-PCA centering/scaling, or PC-scaling toggle.
6. For each trial and PC, average that PC's scores over the time bins in the selected before, after, or whole window. Apply the required predictor/response eligibility mask for the subsequent trial-level fit.
7. Fit the unpenalized multivariable OLS model to that trial-level response, using all its eligible trials for the descriptive partial effects. The predictor table has one row per trial, just as in the unit analysis.

Time bins define the PC trajectory and its window average; they are **not additional independent observations** for the summary encoding model. Keep the original PCA basis fixed during PC-response null resamples; resample the trial-level PC response residuals as specified, not the PCA fitting procedure.

These are descriptive, in-sample PC associations. Do not add cross-validation or PCA refitting machinery to this figure. The separate held-out OLS–Poisson comparison concerns unit spike counts. Retain the actual PCA input trials/units, bin size, and component count with the analysis settings; display PCs in component order within each region.

### PC Summary and Significance

Create a heatmap with:

- **Rows:** PCs
- **Columns:** task / decision variables
- **Value:** **Partial $R^2$**

For each valid PC–predictor test, apply the same **OLS Freedman–Lane selectivity test** used for units, operating on the reduced-model residuals of that PC response. Use the same simplified first-pass residual permutation and provisional interpretation described under **Statistical Selectivity**; temporal-fluctuation controls are deferred for PC tests as well.

Keep partial $R^2$ as the heatmap color and show **BH-FDR-corrected significance** with an outline or marker. Correct valid PC–predictor tests as a separate family from the unit–predictor tests. No additional significance figure is required.

Skip invalid or unavailable PC–predictor significance tests without stopping valid unit tests. Retain the descriptive PC partial-effect map wherever its model fit and effect estimate are valid, and distinguish unavailable significance from a valid nonsignificant result. This does not override shared validity requirements: exact rank deficiency in the same predictor matrix can affect both unit and PC tests. The deliberately deferred temporal controls are a common limitation requiring a provisional label, not by themselves a reason to suppress every otherwise computable test.

This tests whether a task variable contributes to the PC response after accounting for the other predictors. It does **not** test whether the PC itself is a significant population dimension or whether every unit contributing to that PC is selective.

Allow selection of:

- Before
- After
- Whole
- Region
- Number of PCs or PC subset where appropriate

This figure answers:

> Which task and decision variables are represented along the dominant neural population dimensions?

## Shared Display and Result Conventions

- Show all valid effect estimates. Use significance outlines/markers rather than suppressing nonsignificant effects, and distinguish unavailable effects/tests from zero effects or valid nonsignificance.
- Use stable unit-ID ordering grouped by region and PC component order within each region. Preserve the identity of rows when switching windows or models.
- Label partial-effect maps **In-sample conditional effect**, with the specific scale: **Partial $R^2$** for OLS and **Partial deviance explained** for Poisson. Keep the distinct coefficient-display labels defined above.
- Include the first-pass temporal caveat and the within-family FDR scope in figure notes or an immediately associated result description.
- Retain a compact result record with session, alignment, window, model family, selected predictor set, verified source-to-model coding, eligible trial and spike counts, unit/PC identities, fit/test validity and reasons, actual PCA settings, null-draw count and seed, raw/adjusted p-values, and BH family definition/size. For the MSE comparison, also retain fold assignments, observation counts, and fold-level scores.
- Reuse existing result/plotting structures where practical. These records are ordinary analysis metadata, not a new provenance or validation framework.

## PC Sign Convention

Do not emphasize signed coefficients for PCs in the primary summary.

The sign of a principal component is arbitrary:

$$
PC_k
$$

and

$$
(-PC_k)
$$

describe the same population dimension if the corresponding loadings are also sign-flipped.

Therefore, the sign of a regression coefficient onto an individual PC should not be interpreted as an intrinsic direction of encoding unless a specific sign convention has been imposed.

**Partial $R^2$** is the preferred primary PC selectivity measure because it is invariant to this sign ambiguity. The corresponding partial-effect significance test is also unchanged by flipping the PC orientation. Statistical significance and the arbitrary positive/negative sign of a PC are different concepts.

## Interpretation

The three summary outputs answer complementary questions:

1. **Variable × region fraction-selective heatmap**
   - How widespread are conditional predictor contributions that pass the provisional first-pass significance procedure within each region?

2. **Unit × variable selectivity heatmap**
   - Which individual neurons show conditional associations with the selected task or decision variables, and how large are their partial effects?

3. **PC × variable selectivity heatmap**
   - Which variables are associated with the dominant low-dimensional population activity patterns, conditional on the selected model?

These are preliminary inspection results. In particular, a small conditional contribution is not evidence that the neuron or PC has no relationship with a variable; related predictors may account for the same information. Conversely, passing the first-pass significance procedure does not establish that an association is independent of temporal fluctuations. Apply the required later re-evaluation described under **First-Pass Trial Treatment and Deferred Temporal Controls**.

## Deferred Issues

**Temporal fluctuations and across-trial dependence:** proceed with the simplified null procedures for inspection, but re-evaluate the analyses and inferential conclusions later, using the Harris reference and scope described under **First-Pass Trial Treatment and Deferred Temporal Controls**. Do not add that expanded analysis now.

Detailed treatment of predictor multicollinearity is explicitly deferred to later analysis. Overlap among the requested predictors is expected and is not a reason to expand the current inspection scope. The accepted minimal rank check above remains the validity requirement for exact predictor redundancy; it does not reopen that broader analysis.

Because the models estimate **unique contribution conditional on all other included predictors**, correlated predictors may divide or obscure shared explanatory variance.

Low partial $R^2$ or partial deviance explained should therefore not automatically be interpreted as evidence that a variable has no relationship with neural activity when strongly correlated predictors are present.


## Accepted Inspection Defaults

The previously open scientific settings are resolved for this first-pass implementation. Use these defaults unless the user explicitly changes a configurable value; do not add analysis policies or infer unverified source meanings.

| Setting | Accepted default |
|---|---|
| Analysis scope | Single session; choice alignment by default, trial-start alignment available |
| Windows | Before −2 to 0 s; after 0 to +2 s; whole −2 to +2 s |
| Unit response | One total spike count per eligible trial/window |
| Primary effects | In-sample full/reduced fits; no CV toggle for effect maps |
| Response models | Unpenalized OLS and Poisson log-link GLM, with intercept and main effects only |
| Predictor set | Requested table above, manually selectable; no automatic additions or pruning |
| Predictor coding | Verify codebase meanings; choose and document an appropriate consistent encoding; listed codes are examples only |
| Coefficients | Post-fit OLS standardized coefficient; Poisson log-count coefficient per predictor SD |
| PC response | Window-mean score from one descriptive shared-temporal PCA per region/alignment |
| PCA inputs/default size | 100-ms bins, full −2 to +2 s, unit z-scoring once, up to 10 PCs per region |
| PC scaling | No whitening or extra PC-score scaling |
| OLS null | Across-eligible-trial Freedman–Lane residual permutation within session |
| Poisson null | Independent count draws from the fitted reduced Poisson means |
| Resampling | 999 datasets per test; seed 0; explicit alternative count allowed, no adaptive procedure |
| Significance | BH-FDR $q=0.05$ in the defined unit or separate PC family; provisional temporal interpretation |
| Fraction selective | Significant units / valid unit tests for each predictor and region; show numerator/denominator |
| Model comparison | One paired five-fold block-grouped CV pass; pooled held-out count MSE; no formal superiority test |
| Invalid calculations | Mark unavailable with reason; preserve valid independent results; no silent repairs or failed-null discards |
| Deferred work | Temporal-fluctuation-aware inference and broader multicollinearity/model refinement |

## Codebase Verification and Implementation Scope

Codex should now resolve **codebase facts**, not reopen the accepted scientific defaults:

1. Verify source columns, trial identifiers, left/right and reward codes, missing-value meanings, and the entering-trial update/reset conventions. Document any source-to-predictor mapping; do not guess unresolved meanings.
2. Reuse the existing unit-quality selection, regional assignments, alignment/window extraction, binned-activity utilities, and behavioral-block identifiers. Check that the comparison grouping uses the intended existing blocks.
3. Reuse appropriate unpenalized model-fitting and result/plotting utilities. Confirm intercept handling, convergence reporting, and the explicit absence of an inherited regularization penalty.
4. Implement the agreed minimal validity and unavailable-result behavior. Do not add automatic predictor pruning, numerical-result substitution, model-family expansion, adaptive resampling, or a broad recovery workflow.

Known codebase mismatches should be reported with their scope, while independent valid analyses continue. The spec is ready for first-pass implementation planning; its provisional statistical interpretation and required future temporal re-evaluation remain part of the delivered results.
