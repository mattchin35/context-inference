# Single-Unit Selectivity Analysis Spec

## Goal

Measure how individual units and major population dimensions selectively encode task variables and decision variables, while estimating the **unique contribution** of each predictor in a multivariable model.

The core analysis is:

$$
y_{\mathrm{neural}}
\sim
\mathrm{state} +
\mathrm{action} +
\mathrm{reward} +
\mathrm{DV}_1 +
\mathrm{DV}_2 + 
\cdots
$$

where all selected predictors are included simultaneously.

The purpose of the multivariable model is to estimate how much each predictor contributes after accounting for the other included variables.

## Analysis Windows

Fit models separately for:

- **Before:** −2 to 0 s relative to the alignment event
- **After:** 0 to +2 s
- **Whole:** −2 to +2 s

The summary figures should allow selection among these three windows.

No "selective at any time during the trial" summary is required.

## Predictor Set

Include task variables and behavioral/model-derived decision variables available in the current codebase, such as:

- Task/context state
- Action
- Reward
- Value
- Belief
- Doubt
- Consecutive omissions
- Reward history
- Other available DVs

The exact predictor set should be selectable and based on the variables available in the behavioral analysis pipeline.

## Unit-Level Response Models

Provide two model options for individual units.

### OLS / Gaussian Linear Model

Use the unit's spike count or firing-rate summary within the selected analysis window as the response.

For each predictor report:

- **Standardized coefficient**
- **Partial $R^2$**
- Permutation-test significance

The standardized coefficient preserves the direction of selectivity.

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

Provide a Poisson GLM as an alternative model for spike-count responses.

Fit the same predictor set:

$$
y_{\mathrm{spikes}}
\sim
\mathrm{state} +
\mathrm{action} +
\mathrm{reward} +
\mathrm{DV}_1
+\cdots
$$

For each predictor report:

- GLM coefficient
- **Partial deviance explained**
- Permutation-test significance

Calculate the unique contribution by comparing the full model with a reduced model that excludes the predictor of interest.

Label this quantity explicitly as:

**Partial deviance explained**

Do not label partial deviance explained as $R^2$.

## Comparing OLS and Poisson Models

Allow OLS and Poisson GLM results to be inspected separately.

Do not determine which model performs better by directly comparing:

- Partial $R^2$
- Partial deviance explained

because these are different metrics.

If model comparison is performed, use a separate predictive or model-fit criterion, preferably held-out predictive performance.

## Statistical Selectivity

Define a unit as **selective for a predictor** when that predictor's unique contribution is statistically reliable under a permutation test.

The permutation procedure should disrupt the relationship between the predictor and neural response while preserving the relevant structure of the dataset as much as possible.

For each unit and predictor:

1. Fit the observed full and reduced models.
2. Calculate the observed unique contribution:
   - Partial \(R^2\) for OLS
   - Partial deviance explained for Poisson GLM
3. Permute the predictor or equivalent trial labels according to the selected null procedure.
4. Refit the models and generate a null distribution.
5. Calculate a permutation p-value.
6. Apply appropriate multiple-comparison correction where needed.

The primary definition of "selective" should be based on corrected permutation significance rather than an arbitrary effect-size threshold.

## Summary Figure 1: Fraction of Selective Units

Create a heatmap with:

- **Rows:** task variables / decision variables
- **Columns:** brain regions
- **Value:** fraction or percentage of units significantly selective for that predictor

Use **fraction of units** as the primary value.

Also retain the number of contributing units for interpretation.

Allow selection of:

- Before
- After
- Whole
- OLS
- Poisson GLM

This figure answers:

> What fraction of units in each region uniquely encode each task or decision variable?

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

For OLS, use **standardized coefficients** so coefficient magnitudes are more comparable across predictors.

The coefficient view should preserve sign so positive and negative tuning can be distinguished.

Permutation significance may be shown as an overlay, mask, marker, or accompanying information rather than replacing the effect-size heatmap.

This figure answers:

> Which individual units encode which variables, and how large is each variable's unique contribution?

## Summary Figure 3: PC × Variable Selectivity

Perform an analogous selectivity analysis using neural population principal components as responses.

For each PC:

$$
PC_k
\sim
\mathrm{state} +
\mathrm{action} +
\mathrm{reward} +
\mathrm{DV}_1
+\cdots
$$

Use OLS for PC scores.

Create a heatmap with:

- **Rows:** PCs
- **Columns:** task / decision variables
- **Value:** **Partial $R^2$**

Allow selection of:

- Before
- After
- Whole
- Region
- Number of PCs or PC subset where appropriate

This figure answers:

> Which task and decision variables are represented along the dominant neural population dimensions?

## PC Sign Convention

Do not emphasize signed coefficients for PCs in the primary summary.

The sign of a principal component is arbitrary:

$$
PC_k
$$

and

$$
-PC_k
$$

describe the same population dimension if the corresponding loadings are also sign-flipped.

Therefore, the sign of a regression coefficient onto an individual PC should not be interpreted as an intrinsic direction of encoding unless a specific sign convention has been imposed.

**Partial $R^2$** is the preferred primary PC selectivity measure because it is invariant to this sign ambiguity.

## Interpretation

The three summary outputs answer complementary questions:

1. **Variable × region fraction-selective heatmap**
   - How widespread is reliable encoding of each variable within each region?

2. **Unit × variable selectivity heatmap**
   - Which individual neurons uniquely encode each task or decision variable, and how strongly?

3. **PC × variable selectivity heatmap**
   - Which variables are represented in the dominant low-dimensional population activity patterns?

## Deferred Issues

Detailed treatment of predictor multicollinearity is deferred to later analysis.

Because the models estimate **unique contribution conditional on all other included predictors**, correlated predictors may divide or obscure shared explanatory variance.

Low partial $R^2$ or partial deviance explained should therefore not automatically be interpreted as evidence that a variable has no relationship with neural activity when strongly correlated predictors are present.
