# Single-Unit Selectivity Analysis Spec

> Working revision v6, updated 2026-10-27: incorporates the agreed source names, trial and unit selection, shared configurable neural bins, independent regional coverage, saved-feature inputs, and offline OLS-first execution. Effect maps are in-sample; the unit-level OLS–Poisson MSE comparison is held out. Significance remains provisional, with temporal-fluctuation controls and broader multicollinearity work explicitly deferred. Source values and model encodings must be documented consistently with the existing codebase.

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

### Shared Neural Bins and Time Scales

Unit and PC analyses must start from the **same regional binned activity**, with axes **trial × time bin × unit**. Use **100-ms bins by default**, with one configurable bin width shared by both analyses. Use the same alignment, full -2 to +2 s window, bin edges, and before/after/whole bin membership; do not use a separate temporal resolution for PCs and units.

- Construct the full-window activity once per region/alignment/bin configuration using the existing binned-activity utility. Retain spike counts per bin; any existing utility's conversion to/from Hz must be explicit and preserve bin width and units.
- Use complete bins with the window endpoints and time zero as bin boundaries. Follow the existing half-open convention: before is [-2, 0), after is [0, 2), and whole is [-2, 2), in seconds. A spike at time zero belongs to after, not both halves. A configured bin width must partition both halves without truncating or padding the requested windows.
- For a unit, sum its counts over the selected bins to obtain one trial response. For a PC, average its scores over those same bins to obtain one trial response after the regional PCA transformation.
- Apply the same trial-level predictor construction, full/reduced OLS comparisons, coefficient-display policy, and OLS null procedure to either neural representation. Their response units differ, and only unit counts support the Poisson model. Shared bins do not turn bins into independent regression observations.
- Changing bin width defines a new analysis configuration for both representations and requires recomputing its activity/PCA products and fits. Retain the actual bin width and edges with the results.

## Predictor Set

Use the following requested predictor set **by default**, with individual predictors manually selectable. These are **predictors of neural activity**, not separate decoding targets. The response remains the selected unit's spike count or the selected PC response.

| Predictor | Existing source column or definition | Trial timing |
|---|---|---|
| Task/context state | `state_int` | State on trial $i$ |
| Action | `action` | Action on trial $i$ |
| Current reward | `reward` | Outcome on trial $i$ |
| Previous reward | `reward` from the immediately preceding chronological trial, subject to availability below | Outcome on trial $i-1$ |
| Value | `FQlearning_rel_value` | Value entering trial $i$ |
| Belief | `HMM_rel_value_logodds` | Value entering trial $i$ |
| Doubt | `relative_doubt_index` | Value entering trial $i$ |
| Consecutive rewards | `consecutive_rewards` | History entering trial $i$ |
| Relative value | `relative_value` | Value entering trial $i$ |
| Consecutive omissions | `consecutive_omissions` | History entering trial $i$ |
| Relative omissions | `relative_omissions` | Value entering trial $i$ |

### Timing and Source Mapping

- Keep **current reward and previous reward as separate predictor terms**. Do not merge them into one generic reward/history variable.
- All listed values, beliefs, doubt measures, and history DVs describe the state **entering trial $i$**, before incorporating that trial's outcome. Use the same predictor definitions for before, after, and whole-window neural responses; do not change their update timing with the selected neural epoch.
- Derive previous reward from the original chronological trial sequence within the session, before analysis-specific filtering. It is unavailable on the first trial, when the immediately preceding reward is missing/invalid, or when the immediately preceding trial is manual-reward, no-choice, or dark-state. Do not fill it with zero or use the last retained trial's outcome. Neural-coverage and predictor filtering must not change which chronological row is the predecessor. Apply the agreed identical-trial rule to full/reduced fits.
- Use the existing source names exactly, including `FQlearning_rel_value`; correct misspellings in specification labels or configuration references rather than creating renamed copies of existing feature columns. Verify numerical representations and update/reset conventions against the saved features and their existing generators. Do not silently substitute another value/belief model or shift an already pre-update column.
- `FQlearning_rel_value` and `relative_value` are intentionally separate. The former is the existing forgetting-Q value; `relative_value` is the pre-trial `left_value - right_value` history counter. `relative_omissions` is `left_omissions - right_omissions`, not the separate `relative_omissions_index`. Any actual exact redundancy is handled only by the agreed minimal rank check.
- Preserve `HMM_rel_value_logodds` as stored: the current generator returns `tanh(tanh_scale * log_odds)` from the entering-trial belief, not an untransformed log-odds value. Document the saved `tanh_scale`; do not invert this transform or substitute `HMM_rel_value_logodds_decay`.
- Existing history counters are recorded before the current outcome, carry across block transitions, and skip state updates on manual-reward/no-choice trials. Preserve these saved conventions; do not reset, recompute, or realign histories after filtering for neural analysis. `relative_doubt_index` remains the existing feature derived from counterfactual omission counters.
- Do not automatically add other decoding targets, value models, or history variables to this simultaneous predictor set. Include **main effects only**, with an intercept: no automatic interactions, polynomial terms, additional lags, or predictor selection. Choose predictor coding as specified below.

Current-trial reward is an outcome-associated predictor even in the before-event model. A relationship with before-event activity should be described as an association with the trial's eventual reward outcome, not automatically as a neural response to reward delivery.

### Predictor Coding — Existing Source Values and Documented Model Encoding

**Use the verified existing source meanings.** Check the saved inputs for consistency with these codes and the project's missing-value conventions. Choose an appropriate regression encoding consistent with those meanings, and document the mapping from source values to model columns.

The existing task-table representations are:

| Predictor | Existing source representation |
|---|---|
| `state_int` | Right = 0; left = 1; dark = 2, excluded from this analysis |
| `action` | Right = 0; left = 1; missing/no-choice entries are excluded |
| Current reward | No reward = 0; reward = 1 |
| Previous reward | No reward = 0; reward = 1 |
| Listed numerical DVs | Preserve the existing numerical values and transformations |

Retaining the existing 0/1 binary coding is appropriate; an explicitly documented nonredundant contrast is also permitted. Do not infer category meanings from numeric signs or silently recode missing/no-choice/dark values as a valid binary class. Keep the meanings consistent across windows, units, PCs, and response-model families; label the resulting coefficient direction/reference category.

Represent a binary predictor with one appropriately coded column alongside the intercept; avoid redundant category columns. If a selected predictor genuinely requires multiple category contrasts, use an appropriate nonredundant encoding and retain the agreed joint test of its complete term. Do not add category expansions that the current predictors do not require. Numerical DVs remain numerical predictors; coefficient standardization is the post-fit display conversion below, not a reason to change their definitions.

An ambiguous or unavailable source mapping must be reported, not guessed or silently replaced with a different variable. Source verification is an implementation task, not an invitation to redesign the predictor set.

### Saved Behavioral Inputs

Consume the existing saved `*_augmented_trials.csv` and associated `trial_feature_params.json`. Preserve the complete chronological trial sequence and its `cur_trial` identities when deriving previous reward; use `cur_block` as the existing behavioral-block identifier for held-out grouping.

Do not run behavioral feature generation, refit behavioral models, or automatically regenerate a missing feature during this neural analysis. Report missing required columns or unavailable feature settings explicitly. Retain the input file identities and relevant saved feature parameters with the analysis so the selected predictors and transformations can be interpreted.

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
- Obtain that total by summing the selected bins from the shared regional activity described under **Shared Neural Bins and Time Scales**. PC responses use the same bins and epochs; do not build a separate unit-analysis grid.
- With $N$ eligible trials, the response vector has $N$ entries and the predictor matrix has $N$ rows. Do not treat time bins within a trial as additional independent observations in this summary analysis.
- Use the same count response for OLS and Poisson and for their held-out MSE comparison. Firing rate may be shown descriptively, but it is not a separate response-model toggle here.
- Do not z-score the count response before the Poisson fit. Coefficient standardization is a post-fit display conversion, defined under **Coefficient Scales and Interpretation**, not an additional response-preprocessing step.

The count-response definition applies to individual units. For PCs, use one window-mean PC score per trial, as defined under **Summary Figure 3: PC × Variable Selectivity**.

### Offline Execution — OLS First, Poisson Afterwards

Run fits and null resampling **offline**. The webapp and plotting layer load saved results; selecting a result or changing a display control must not start fitting. Changing the selected predictor set or another fitted setting requires an explicit offline analysis configuration.

Support this staged workflow:

1. Prepare the saved predictors, regional activity, descriptive PC responses, and unit comparison folds where available. Insufficient groups or unavailable comparison folds affect held-out MSE only; they must not block valid in-sample effects or null tests.
2. Run and save **unit OLS and PC OLS** effects, coefficients, null tests, and their separate BH families. Save unit OLS held-out predictions, fold scores, counts, and assignments for the later paired model comparison. OLS effects and inference remain available even if a unit's held-out comparison is unavailable.
3. Make completed OLS results available for inspection **before Poisson starts or finishes**. An OLS-only execution is supported; its output must not remain hidden until a requested future Poisson stage completes.
4. Run **unit Poisson** effects, bootstrap tests, its separate BH family, and matched held-out fits afterwards. Reuse compatible saved preparation and OLS comparison results rather than refitting completed OLS work. The saved predictors, eligible rows, counts, settings, and fold assignments must match for a paired comparison; changed inputs/settings require a new compatible analysis, not silent mixing of results.
5. Publish the paired OLS–Poisson MSE comparison only when all five requested paired folds are valid. Before the Poisson comparison stage completes, identify that comparison as pending or not run rather than inventing a paired score.

Save completed work at practical response/configuration boundaries with explicit progress and stage status. Subsequent offline execution must be able to continue compatible unfinished stages without rerunning completed work. Completed OLS outputs remain readable if a Poisson stage is interrupted or fails. Preserve previous results when inputs, parameters, or analysis version change.

BH correction still uses each complete family defined below. A partial checkpoint is progress evidence; do not present correction over only the finished cells as a finalized family. Retain configured seeds and reproducible resampling so executing Poisson later does not change completed OLS results. These requirements do not introduce adaptive resampling or an elaborate recovery framework.

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

For a nonconstant binary predictor, the same column-level scaling can be calculated, but document its verified coding: a one-SD change is not generally a complete switch between its two classes. Follow **Predictor Coding — Existing Source Values and Documented Model Encoding** and label the coefficient direction using the chosen model encoding. A multilevel predictor has separate contrast coefficients rather than one arbitrary signed variable-level coefficient; its partial effect and significance test remain joint across the complete term.

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

Correction belongs to the analysis configuration, including its shared bin width and analyzed regional populations, not the visible subset of a plot. Hiding a region, unit, PC, or predictor row/column must not recalculate BH over only the displayed cells. Changing the fitted predictor set defines a different model configuration, not a display-only change.

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

### Trial and Regional Coverage Eligibility

- Exclude **manual-reward, no-choice, and dark-state trials** from both unit and PC analyses, including the descriptive PCA input pool. These are baseline exclusions even when the corresponding action/state predictor is deselected. Use the existing `experimenter_reward_given` flag and normalization of its legacy `give_reward` name, existing no-choice/missing-action conventions, and `state_int` codes. Missing/invalid required baseline information is unavailable, not an eligible default value.
- Treat PFC and HPC **independently**. Require full coverage of the entire -2 to +2 s alignment window in a region for a trial to enter that region's activity tensor, including when fitting a before-only or after-only summary. Coverage means recorded/aligned temporal support, not the presence of spikes; do not mistake an unrecorded interval for measured silence.
- Do not require common PFC/HPC trials. A trial unavailable in one region can still contribute to the other; a regional failure must not suppress valid independent results. This coverage requirement does not add overlap detection or an exclusion based on crossing behavioral trial boundaries.
- Within a region/alignment/bin configuration, use one shared baseline-valid, full-coverage activity pool for unit counts and PCA construction. Fit the descriptive PCA on that pool before selected-predictor missingness is applied. Subsequent unit and PC regressions apply the full selected predictor and response eligibility masks; selecting a different predictor subset must not by itself refit the descriptive PCA basis.

### Unit Selection and Fit Validity

Reuse the existing channel and cluster selection, including **both `good` and `mua` units**. Preserve the configured PFC/HPC probe roles and channel restrictions. Use the existing quality-column convention (`group` for manual Phy labels, or explicitly selected `KSLabel`) and existing normalization of missing quality labels to `mua`; retain the actual column, labels, and selection with the results. Do not add an arbitrary minimum spike-count threshold or a new QC/diagnostic framework. Retain eligible trial counts and total spike counts for inspection.

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

1. Start from the **same regional trial × time-bin × unit activity tensor used for the unit responses**, in **100-ms bins by default**, with the shared configurable bin width and edges defined under **Shared Neural Bins and Time Scales**. Pool the baseline-valid, regionally full-coverage trials across the full **−2 to +2 s** window. Do not require bilateral coverage or restrict this PCA pool by the selected predictors' missingness.
2. Z-score each usable unit once using observations pooled across those trials and time bins. A unit with zero variance in this input matrix cannot be z-scored; omit that input from PCA and report the usable-unit count. This does not automatically exclude that unit from independent trial-count analyses.
3. Fit separate shared-temporal PCAs for PFC and HPC. Use **one descriptive basis per session, region, selected alignment, and shared activity configuration**, reused across before, after, and whole-window summaries rather than refitted for each epoch or predictor subset. A change to bin width, input trials, or selected units changes this activity configuration.
4. Retain **up to 10 PCs per region by default**, with the count configurable. If fewer usable dimensions are available, retain that smaller count and report the actual count rather than failing the session or padding with zero-variance components.
5. Project the regional activity into those PC coordinates with **whitening disabled**. Use ordinary PC scores directly: no additional PC-score z-scoring, post-PCA centering/scaling, or PC-scaling toggle.
6. For each trial and PC, average that PC's scores over exactly the bins used to sum that trial's unit counts for the selected before, after, or whole window. Apply the required predictor/response eligibility mask for the subsequent trial-level fit.
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
- Load saved offline results for presentation. Completed OLS results must be inspectable while Poisson is pending, running, or failed; distinguish incomplete-stage status from an unavailable scientific test.
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
| Shared neural bins | Same regional trial × bin × unit activity for units and PCs; 100-ms default width, configurable together; identical bin edges and epochs |
| Unit response | One total spike count per eligible trial/window, summed over the shared bins |
| Baseline trial exclusions | Manual-reward, no-choice, and dark-state trials, regardless of selected predictors |
| Previous reward | Immediately preceding chronological reward; unavailable after manual-reward/no-choice/dark trials or missing/invalid preceding reward; no bridging |
| Regional coverage | Independent PFC/HPC eligibility; full -2 to +2 s coverage required for both unit and PC inputs |
| Unit quality | Both `good` and `mua`, using existing quality/channel selection and recorded source column |
| Behavioral inputs | Existing saved augmented-trial table and `trial_feature_params.json`; no feature regeneration |
| Execution | Offline, with saved progress; unit/PC OLS can complete and be inspected before unit Poisson runs; resume compatible unfinished stages |
| Primary effects | In-sample full/reduced fits; no CV toggle for effect maps |
| Response models | Unpenalized OLS and Poisson log-link GLM, with intercept and main effects only |
| Predictor set | Existing source names in the table above, including `FQlearning_rel_value`; manually selectable; no automatic additions or pruning |
| Predictor coding | Preserve verified source meanings and numerical DV transformations; document the chosen consistent binary model encoding |
| Coefficients | Post-fit OLS standardized coefficient; Poisson log-count coefficient per predictor SD |
| PC response | Window-mean score from one descriptive shared-temporal PCA per region/alignment |
| PCA inputs/default size | Shared bins and regional baseline/full-coverage activity pool, full −2 to +2 s, independent of predictor missingness; unit z-scoring once, up to 10 PCs per region |
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

1. Verify the saved augmented-trial columns and feature parameters against the existing source names, trial identifiers, category codes, missing-value meanings, and entering-trial update/reset conventions documented above. Preserve saved feature values and report any mismatch; do not regenerate or guess unresolved inputs.
2. Reuse the existing `good`/`mua` unit-quality selection, regional assignments, shared alignment/bin extraction, and `cur_block` identifiers. Implement independent regional full-window coverage and the agreed baseline exclusions without adding a bilateral trial requirement.
3. Reuse appropriate unpenalized model-fitting and result/plotting utilities. Confirm intercept handling, convergence reporting, and the explicit absence of an inherited regularization penalty.
4. Implement the agreed minimal validity and unavailable-result behavior. Do not add automatic predictor pruning, numerical-result substitution, model-family expansion, adaptive resampling, or a broad recovery workflow.
5. Provide offline staged execution and saved progress so OLS results can be completed, published, and inspected before Poisson runs. Keep display-only selection separate from changes requiring a new fit configuration.

Known codebase mismatches should be reported with their scope, while independent valid analyses continue. The spec is ready for first-pass implementation planning; its provisional statistical interpretation and required future temporal re-evaluation remain part of the delivered results.
