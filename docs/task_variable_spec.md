# Task-Variable Decoding Analysis Spec

## Goal
Measure when neural population activity contains information about behavioral/task variables, how decoding changes over event-relative time, and whether information is better captured by low-dimensional population structure or by informative individual units.

Neural activity is used to predict task variables:


$$
\text{neural activity} \rightarrow \text{task variable}
$$

## Categorical Targets
Current categorical targets are:
- current state: the actual task context governing the current choice
- current action
- current action is correct: whether the choice matches the current task context — not whether a reward occurred
- prev action
- prev action was rewarded
- next action
- Current choice switch/stay: action[i] != action[i-1]
- Next choice switch/stay: action[i+1] != action[i]

Construct shifted targets from the original chronological trial sequence within each session, before analysis-specific 
filtering. Missing previous/next observations remain unavailable; they are not filled with zero or linked across session 
boundaries.


### Metrics
Report both:
- **Cross-validated balanced accuracy** — primary, intuitive decoding metric
- **Cross-validated ROC AUC** — secondary threshold-independent metric

Calculate each metric separately on each outer test fold. Display the mean fold score, and retain 
individual fold scores and sample counts.

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

For balanced accuracy, use a prespecified probability threshold of 0.5 unless threshold tuning is explicitly added 
inside training data. Tune categorical models against balanced accuracy, then report both balanced accuracy and AUC from that same selected model. Toggling the displayed metric should not silently select a different model.

## Numerical regression targets
Includes discrete counts/indices and continuous behavioral/model-derived variables. 
All use regression and cross-validated $R^2$.

Discrete targets - Evaluate the continuous-valued regression predictions directly; 
do not round predictions before calculating $R^2$:
- consecutive omissions entering trial i
- consecutive rewards entering trial i
- current trial ix
- trial ix in block
- number of rewards in block entering trial i

Continuous targets:
- QL value (Qlearning_rel_value)
- FQL value (FQlearning_rel_value)
- HMM belief (HMM_rel_value_logodds)
- HMM decay (HMM_rel_value_logodds_decay)
- Relative doubt index
- mouse speed if available (optional target, exclude from initial implementation)

### Metric
Use cross-validated $R^2$. Constant-target test folds have undefined $R^2$ and should be flagged.

Interpretation:
- $R^2 > 0$: predictive information above the mean-prediction baseline
- $R^2 = 0$: equal squared error to predicting the mean of the evaluated target values.
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

For each trial $k$ and time bin $t$, neural features are represented as:


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

where $r_{i,k,t}$ is the activity of unit $i$ in that trial/time bin.

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

PC axes should be shared across time bins, not recomputed at each time bin.
PCs should for HPC and PFC should be fit separately.
PCs should be centered and zscored in analysis.

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
- Use elastic net logistic regression.

For continuous targets:
- Use elastic net linear regression.

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

In sklearn, the `l1_ratio` parameter corresponds to $\alpha$, and `alpha` corresponds to $\lambda$.

Use elastic net as the sole regularized unit-level decoder rather than maintaining a separate ridge-regression option.

Elastic net should be able to run with a default setting (alpha=1.0, l1_ratio=0.5) or to perform optional hyperparameter 
selection. Regularization parameters should be selected using training data only. When hyperparameters are tuned, 
select them using inner cross-validation restricted to the outer training set. The inner splits must 
preserve the required grouping structure, and fitted preprocessing must be refit within each inner training split.

Additionally require the same outer trial assignments across time bins, regions, and representations for a given target, 
using matched eligible trials for direct comparisons. If valid grouped splits cannot be formed, report the target as 
unavailable with the reason. Do not silently fall back to random trial splitting.


## Elastic-Net Unit-Weight Visualization
For selected:
- Target variable
- Time bin
- Brain region

display the fitted unit coefficients.

Show:
- Show signed coefficients on a defined feature scale
- Report each unit’s selection frequency across folds
- Count nonzero coefficients within each fitted fold; do not count nonzeros 
only after averaging coefficients across folds, because averaging can obscure 
instability or create an apparently dense aggregate.

- Units sorted by **absolute coefficient magnitude**
- Coefficient sign retained
- Number of nonzero coefficients
- Fraction of available units with nonzero coefficients

This should make it possible to assess decoder reliance: 
Does the fitted decoder distribute its weights across many recorded units or concentrate them in a 
smaller subset? Nonzero coefficients describe the fitted model, not the total number of neurons encoding the variable
(correlated units may carry redundant information, leading to some being suppressed).

Where coefficients vary across CV folds, summarize their stability or distribution rather than treating a single fold's coefficients as definitive.

## Cross-Validation
Use cross-validation for all reported decoding metrics.

Because behavioral trials are temporally correlated and task/context state is block-structured, the default CV should 
avoid random leakage between neighboring trials.

Keep behavioral blocks intact, but assign multiple blocks to each test fold so that both classes are represented. 
Validate class availability in both training and test sets.

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

AUC should use held-out probabilities or continuous decision scores, not thresholded class predictions. Record the positive class explicitly for each target

Include a visible chance reference of:

\[
0.5
\]

### Continuous Heatmap
Create a separate aligned heatmap with:
- **Rows:** decoded continuous variables
- **Columns:** event-relative time
- **Color:** cross-validated $R^2$

Maintain separate categorical and continuous heatmaps because balanced accuracy/AUC and $R^2$ are not directly comparable on the same color scale.

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

Bins will be aligned to the same event for each trial - either the choice time or trial start time,
as in other analyses. Bins will range from -2s to +2s relative to the aligned event.

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
- Allow inspection of selected $\alpha$ and regularization strength

## Interpretation
The main analyses answer:

- **Temporal decoding:** When does neural activity contain information about each task variable?
- **Regional decoding:** Is task information stronger in PFC, HPC, or their combined population?
- **Population representation:** Is information captured well by dominant population PCs, or does direct unit-level decoding improve performance?
- **Sparse vs distributed coding:** Does elastic net rely on a small set of highly informative units or distribute weight broadly across the population?
- **Temporal resolution:** How robust is decoding to the size of the neural activity bins?

Additional interpretation notes:
- Regional comparison: PFC, HPC, and PFC+HPC compare the recorded populations as supplied. Differences should not be described as information per neuron.
- Target specificity: decoding belief, choice, and context separately does not establish their unique representations
independently of one another. Likewise, decoding trial index could reflect slow population changes. Keep those results descriptive rather than adding confound-control analyses here.
