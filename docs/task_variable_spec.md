# Task-Variable Decoding Analysis Spec

## Goal
Measure when neural population activity contains information about behavioral/task variables, how decoding changes over event-relative time, and whether information is better captured by low-dimensional population structure or by informative individual units.

Neural activity is used to predict task variables:


$$
\text{neural activity} \rightarrow \text{task variable}
$$

## Categorical Targets
Initial categorical targets include:
- Current task/context state
- Current action
- Next action
- Switch vs stay

Additional categorical variables can be added based on variables available in the behavioral analysis codebase.

### Metrics
Report both:
- **Cross-validated balanced accuracy** — primary, intuitive decoding metric
- **Cross-validated ROC AUC** — secondary threshold-independent metric

Balanced accuracy should be the default displayed metric.

For binary targets:

$$
\mathrm{Balanced\ Accuracy}
=
\frac{\mathrm{Sensitivity}+\mathrm{Specificity}}{2}
$$

Chance level:

$$
0.5
$$

## Continuous Targets
Use behavioral/DV variables available in the current analysis codebase, including examples such as:
- Consecutive omissions
- Value
- Belief
- Doubt
- Reward history
- Other relevant model-derived or behavioral variables

The exact variable list should be determined from the currently implemented DVs and user-selected variables.

### Metric
Use cross-validated:

$$
R^2
$$

Interpretation:
- $R^2 > 0$: predictive information above the mean-prediction baseline
- $R^2 = 0$: equivalent to predicting the target mean
- $R^2 < 0$: worse than the baseline predictor

## Temporal Decoding
Decode each target independently from neural activity in event-relative time bins.

Available bin sizes:
- 500 ms
- 100 ms
- 50 ms
- 20 ms

Default:
- **100 ms**

For each trial \(k\) and time bin \(t\), neural features are represented as:


$$
\mathbf{x}_{k,t}
=
[
r_{1,k,t},
r_{2,k,t},
\ldots,
r_{N,k,t}
]
$$

where \(r_{i,k,t}\) is the activity of unit \(i\) in that trial/time bin.

Bin size should be a selectable parameter.

Do not display all bin-size resolutions simultaneously in the primary figure. Temporal-resolution comparisons should be generated as a separate figure when explicitly requested.

## Brain Regions
Allow decoding from:
- **PFC**
- **HPC**
- **PFC + HPC combined**

The combined population should allow direct evaluation of whether joint activity provides more task-variable information than either region alone.

## Neural Representations

### 1. PCA Representation — Default
Use PCA as the default population representation.

Workflow within each cross-validation fold:

1. Use only the training trials to fit preprocessing/scaling.
2. Fit PCA using only the training neural data.
3. Project the training data onto those PCA axes.
4. Project the held-out test data onto the **same training-derived PCA axes**.
5. Train the decoder on the training PC scores.
6. Evaluate predictions on the held-out projected test data.

Conceptually:

$$
X_{\mathrm{train}}
\rightarrow
\text{fit PCA}
$$

then:

$$
X_{\mathrm{train}}
\rightarrow
Z_{\mathrm{train}}
$$

and:

$$
X_{\mathrm{test}}
\rightarrow
Z_{\mathrm{test}}
$$

using the PCA directions learned from $X_{\mathrm{train}}$.

The test set must not contribute to determining:
- Centering/scaling
- PC directions
- Explained variance
- Any other fitted preprocessing parameters

This prevents information leakage into cross-validated decoding performance.

The **number of PCs** should be a configurable parameter.

For categorical targets:
- Use an appropriate linear classifier on PC scores, such as logistic regression.

For continuous targets:
- Use linear regression on PC scores.

General exploratory PCA figures that are not reporting cross-validated decoding performance may still use PCA fit across the full dataset.

### 2. Elastic-Net Unit Representation
Decode directly from individual units using elastic-net regularization.

For categorical targets:
- Elastic-net logistic regression

For continuous targets:
- Elastic-net linear regression

Elastic net uses both L1 and L2 penalties:


$$
\lambda \left(
\frac{1-\alpha}{2} \sum_j \beta_j^2 + 
\alpha \sum_j \lvert \beta_j \rvert
\right)
$$


where:
- $\alpha=0$: ridge-like solution
- $\alpha=1$: lasso-like sparse solution
- intermediate values combine both

Use elastic net as the sole regularized unit-level decoder rather than maintaining a separate ridge-regression option.

Regularization parameters should be selected using training data only, ideally through nested cross-validation.

## Elastic-Net Unit-Weight Visualization
For selected:
- Target variable
- Time bin
- Brain region

display the fitted unit coefficients.

Show:
- Units sorted by **absolute coefficient magnitude**
- Coefficient sign retained
- Number of nonzero coefficients
- Fraction of available units with nonzero coefficients

This should make it possible to distinguish between:
- Broad distributed coding across many units
- Sparse coding dominated by a small subset of units

Where coefficients vary across CV folds, summarize their stability or distribution rather than treating a single fold's coefficients as definitive.

## Cross-Validation
Use cross-validation for all reported decoding metrics.

Because behavioral trials are temporally correlated and task/context state is block-structured, the default CV should avoid random leakage between neighboring trials.

Prefer **block-aware or temporally grouped cross-validation**, such as:
- Holding out complete behavioral blocks
- Holding out contiguous groups of trials

Training and test folds must still contain the necessary target classes for categorical decoding.

Random stratified trial CV may be retained as an optional comparison but should not be the primary reported result for strongly block-structured variables.

All learned preprocessing steps must occur within the training data of each fold, including:
- Scaling
- PCA
- Elastic-net hyperparameter selection

## Primary Visualization: Neural Information Over Time

### Categorical Heatmap
Create a heatmap with:
- **Rows:** decoded categorical variables
- **Columns:** event-relative time
- **Color:** cross-validated balanced accuracy

Allow metric toggle to:
- ROC AUC

Include a visible chance reference of:

\[
0.5
\]

### Continuous Heatmap
Create a separate aligned heatmap with:
- **Rows:** decoded continuous variables
- **Columns:** event-relative time
- **Color:** cross-validated \(R^2\)

Maintain separate categorical and continuous heatmaps because balanced accuracy/AUC and \(R^2\) are not directly comparable on the same color scale.

## Figure Controls
Primary decoding figures should allow selection of:

### Representation
- PCs — default
- Elastic net on units

### Bin size
- 500 ms
- 100 ms — default
- 50 ms
- 20 ms

### Region
- PFC
- HPC
- PFC + HPC

### Categorical metric
- Balanced accuracy — default
- ROC AUC

### PCA parameters
- Number of PCs

### Elastic-net parameters
- Primarily selected by cross-validation
- Allow inspection of selected \(\alpha\) and regularization strength

## Interpretation
The main analyses answer:

- **Temporal decoding:** When does neural activity contain information about each task variable?
- **Regional decoding:** Is task information stronger in PFC, HPC, or their combined population?
- **Population representation:** Is information captured well by dominant population PCs, or does direct unit-level decoding improve performance?
- **Sparse vs distributed coding:** Does elastic net rely on a small set of highly informative units or distribute weight broadly across the population?
- **Temporal resolution:** How robust is decoding to the size of the neural activity bins?