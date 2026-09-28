# Inter-Regional Prediction Analysis Spec

## Goal
Measure directed predictive relationships between PFC and HPC population activity.

The core question is:

> Does recent activity in one region improve prediction of activity in the other region beyond what can already be predicted from the target region's own recent activity?

Primary directions:

- **HPC → PFC**
- **PFC → HPC**

Use two complementary quantifications:

1. **Cross-validated incremental predictive performance**
2. **Granger prediction / Granger causality statistic**

Interpret all results as predictive, not as proof of mechanistic causality.

---

## Neural Representations

Allow two representations.

### 1. Unit activity
Use binned unit activity within each region.

For linear models, activity may be represented as firing rate or spike count.

For Poisson GLMs, use **spike counts per bin** as the response.

For PFC:

\[
\mathbf{x}_{PFC}(t)
=
[
r_1(t),
r_2(t),
\ldots,
r_N(t)
]
\]

and analogously for HPC.

Prediction performance should be calculated separately for each target-region unit so that the distribution of predictability across units is preserved.

### 2. Principal components
Allow PCA as a dimensionality-reduced transformation of the same regional population activity.

For each region:

\[
\mathbf{z}(t)
=
[
PC_1(t),
PC_2(t),
\ldots,
PC_K(t)
]
\]

PCs are simply a transformed representation of firing-rate population activity.

For cross-validated prediction:

1. Fit PCA using only the training data.
2. Project both training and held-out test data onto the training-derived PC axes.
3. Run the same inter-regional prediction analysis in PC coordinates.

Retain per-PC prediction performance before population summarization.

The number of PCs should be configurable.

Because PC scores are continuous and can be negative, **PC prediction uses linear models only**.

---

## Prediction Model

For unit-level prediction, provide a model toggle:

- **Linear regression**
- **Poisson GLM**

For PC-level prediction:

- **Linear regression only**

The restricted/full predictor structure must remain identical across model families.

---

## Temporal Parameters

Three parameters define the predictive model.

### Neural bin size
Default:

- **100 ms**

Other available values:

- 500 ms
- 100 ms
- 50 ms
- 20 ms

### Prediction lag
Default:

- **1 timestep**

This determines how far before the target time the predictor history begins.

With 100-ms bins:

- lag = 1 → 100 ms earlier
- lag = 2 → 200 ms earlier

### Model order
Default:

- **1**

This determines how many historical bins are included.

Examples using 100-ms bins:

- lag = 1, order = 1 → \(t-100\) ms
- lag = 1, order = 3 → \(t-100,t-200,t-300\) ms
- lag = 2, order = 1 → \(t-200\) ms

---

# Linear Inter-Regional Prediction

## HPC → PFC

For each target PFC unit or PC, fit a restricted model using only prior PFC activity:

\[
\mathrm{PFC}(t)
\sim
\mathrm{PFC}(t-\mathrm{history})
\]

Then fit a full model:

\[
\mathrm{PFC}(t)
\sim
\mathrm{PFC}(t-\mathrm{history})
+
\mathrm{HPC}(t-\mathrm{history})
\]

With the default parameters:

\[
\mathrm{PFC}(t)
\sim
\mathrm{PFC}(t-1)
\]

versus:

\[
\mathrm{PFC}(t)
\sim
\mathrm{PFC}(t-1)
+
\mathrm{HPC}(t-1)
\]

where one timestep is 100 ms.

Calculate held-out:

\[
R^2_{\mathrm{restricted},j}
\]

and:

\[
R^2_{\mathrm{full},j}
\]

for each target unit or PC \(j\).

Define:

\[
\Delta R^2_{\mathrm{HPC}\rightarrow\mathrm{PFC},j}
=
R^2_{\mathrm{full},j}
-
R^2_{\mathrm{restricted},j}
\]

## PFC → HPC

Analogously:

\[
\Delta R^2_{\mathrm{PFC}\rightarrow\mathrm{HPC},j}
=
R^2_{\mathrm{full},j}
-
R^2_{\mathrm{restricted},j}
\]

All \(R^2\) values must be calculated on held-out data.

Negative \(\Delta R^2\) values should be preserved.

---

# Poisson GLM Inter-Regional Prediction

Use Poisson GLMs for unit spike-count responses.

For target unit \(j\):

\[
y_j(t)
\sim
\mathrm{Poisson}(\lambda_j(t))
\]

with a log link.

For HPC → PFC, the restricted model is:

\[
\log \lambda_j(t)
=
\beta_0
+
\boldsymbol{\beta}_{PFC}^{T}
\mathbf{x}_{PFC}(t-\mathrm{history})
\]

The full model is:

\[
\log \lambda_j(t)
=
\beta_0
+
\boldsymbol{\beta}_{PFC}^{T}
\mathbf{x}_{PFC}(t-\mathrm{history})
+
\boldsymbol{\beta}_{HPC}^{T}
\mathbf{x}_{HPC}(t-\mathrm{history})
\]

and vice versa for PFC → HPC.

## Poisson Predictive Metric

Use **cross-validated deviance explained** rather than \(R^2\).

For held-out data:

\[
D_{\mathrm{explained}}
=
1-
\frac{
D_{\mathrm{model}}
}{
D_{\mathrm{null}}
}
\]

where:

- \(D_{\mathrm{model}}\) is the Poisson deviance of the fitted model on held-out data
- \(D_{\mathrm{null}}\) is the deviance of a baseline model predicting the mean spike count

Poisson deviance is:

\[
D
=
2
\sum_i
\left[
y_i
\log
\left(
\frac{y_i}{\hat{\mu}_i}
\right)
-
(y_i-\hat{\mu}_i)
\right]
\]

with the \(y_i\log(y_i/\hat{\mu}_i)\) term defined as zero when \(y_i=0\).

Calculate:

\[
D_{\mathrm{explained,restricted},j}
\]

and:

\[
D_{\mathrm{explained,full},j}
\]

Then define incremental source-region prediction as:

\[
\Delta D_{\mathrm{explained},j}
=
D_{\mathrm{explained,full},j}
-
D_{\mathrm{explained,restricted},j}
\]

for both:

- HPC → PFC
- PFC → HPC

Negative values should be preserved.

Do not label deviance explained as \(R^2\).

---

## Comparing Linear and Poisson Models

Allow linear and Poisson results to be viewed with the same predictor structure and cross-validation splits.

Do not directly interpret:

\[
\Delta R^2
\]

and:

\[
\Delta D_{\mathrm{explained}}
\]

as numerically equivalent effect sizes.

If comparing which model better predicts unit activity, use their respective held-out predictive metrics and model-appropriate loss/performance measures rather than directly comparing partial quantities across model families.

---

## Cross-Validation

Use cross-validation for all reported prediction metrics.

Restricted and full models must use identical train/test splits.

All learned preprocessing must occur within training data only, including:

- Scaling
- PCA
- Any regularization or model hyperparameter selection

Where appropriate, use temporally grouped or block-aware CV rather than randomly scattering neighboring time bins between train and test.

---

# Population Prediction Summary

Retain the full target-unit or target-PC distribution rather than immediately pooling the region into one value.

For each direction and condition:

### Linear model
retain:

\[
\Delta R^2_j
\]

### Poisson model
retain:

\[
\Delta D_{\mathrm{explained},j}
\]

## Summary visualization

For each behavioral condition and prediction direction, show:

- Every target unit or PC as an individual point
- Median
- IQR

This should reveal whether cross-region predictability is:

- broadly distributed across the target population
- concentrated in a small subset
- heterogeneous across units

Absolute restricted and full-model predictive performance should also remain inspectable.

---

# Main Prediction Figure

Show all behavioral conditions together.

Within each condition display:

- **HPC → PFC**
- **PFC → HPC**

For each direction show:

- Individual target values
- Median
- IQR

Allow controls for:

### Representation
- Units
- PCs

### Model
For units:
- Linear
- Poisson GLM

For PCs:
- Linear only

### Temporal parameters
- Bin size
- Prediction lag
- Model order

### Behavioral condition
- Trial condition and other existing behavioral splits

The y-axis label should automatically reflect the selected model:

- **Linear:** Incremental CV \(R^2\)
- **Poisson:** Incremental CV deviance explained

---

# Granger Prediction

Granger analysis should use the same temporal structure as the incremental-prediction analysis.

## Linear Granger

For target activity \(Y\) and source activity \(X\):

Restricted model:

\[
Y_t
=
\sum_i A_iY_{t-i}
+
\epsilon_t
\]

Full model:

\[
Y_t
=
\sum_i A_iY_{t-i}
+
\sum_i B_iX_{t-i}
+
\epsilon'_t
\]

Calculate both:

\[
GC_{\mathrm{HPC}\rightarrow\mathrm{PFC}}
\]

and:

\[
GC_{\mathrm{PFC}\rightarrow\mathrm{HPC}}
\]

## Poisson / GLM Granger-Style Prediction

For spike-count responses, use the analogous restricted-versus-full Poisson GLM comparison.

The conceptual test remains:

\[
\text{target-region history}
\]

versus:

\[
\text{target-region history + source-region history}
\]

but model improvement should be quantified using Poisson likelihood/deviance rather than the classical Gaussian autoregressive statistic.

Label this explicitly as:

**Poisson GLM Granger-style prediction**

rather than treating it as numerically identical to classical linear Granger causality.

---

# Granger Summary Figure

Create a separate figure from the incremental predictive-performance plot.

Show all behavioral conditions together.

Within each condition display:

- HPC → PFC
- PFC → HPC

Allow the same model, representation, lag, bin-size, and model-order controls where applicable.

Do not place Granger statistics and incremental \(R^2\)/deviance values on the same numerical axis.

---

## Default Analysis

Use:

- **Representation:** units
- **Model:** linear by default, with Poisson GLM available as a toggle
- **Bin size:** 100 ms
- **Prediction lag:** 1 timestep
- **Model order:** 1

With these defaults, HPC → PFC asks:

> Does HPC activity during the immediately preceding 100 ms improve prediction of current PFC activity beyond what is predicted by PFC activity during that same preceding 100 ms?

The same question is evaluated with either a linear response model or a Poisson spike-count model.

---

## Interpretation

The main outputs answer:

1. **Incremental prediction**
   - How much does source-region activity improve held-out prediction?

2. **Model family**
   - Is the relationship better captured by a linear firing-rate model or a Poisson spike-count model?

3. **Population distribution**
   - Is predictive coupling widespread or concentrated in particular units or PCs?

4. **Directionality**
   - Is prediction stronger for HPC → PFC or PFC → HPC?

5. **Granger prediction**
   - Does the standard restricted-versus-full time-series framework support the same directional relationship?

6. **Temporal dependence**
   - How do predictive relationships depend on neural bin size, lag, and model order?

---

## Deferred Analyses

More elaborate inter-regional analyses are intentionally deferred, including:

- Mutual information
- Communication subspaces
- Canonical correlation analyses
- Other latent/shared-subspace methods