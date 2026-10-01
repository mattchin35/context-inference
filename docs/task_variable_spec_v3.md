# Task-Variable Decoding Analysis Spec

**Status:** Working revision 3. The user's decisions on O1-O4, the 10-PC-per-region default, and the latest preprocessing and trial-window clarifications are incorporated. The remaining O5 settings appear in **Recommended Inspection Defaults — Proposed**; they are recommendations, not additional user-approved decisions. **Codebase Checks** identify definitions to verify rather than invent. This review has not inspected the implementation code or DV-generation code.

**Scope:** This analysis is for scientific inspection, not publication-finalized inference. Favor straightforward estimators, explicit settings, and readable summary figures. Preserve training/test separation, but do not add repeated validation, automatic model-search expansion, significance testing, or elaborate fallback machinery beyond the requested analysis.

## Goal
Measure when neural population activity contains information about behavioral/task variables, how decoding changes over event-relative time, and whether information is better captured by low-dimensional population structure or by informative individual units.

Neural activity is used to predict task variables:

$$
\mathrm{neural\ activity} \rightarrow \mathrm{task\ variable}
$$

Run the analysis within each simultaneously recorded session. Different sessions' unit columns must not be treated as the same neural features.

## Categorical Targets
Current categorical targets are:
- Current state: the actual task context governing the current choice.
- Current action.
- Current action is correct: whether the choice matches the current task context, not whether a reward occurred.
- Previous action.
- Previous action was rewarded.
- Next action.
- Current choice switch/stay: `action[i] != action[i-1]`.
- Next choice switch/stay: `action[i+1] != action[i]`.

Each switch/stay entry is one binary target, not separate switch and stay models. Choice switching must not be confused with a task-context transition.

Construct shifted targets from the original chronological trial sequence within each session, before analysis-specific filtering. Missing previous/next observations remain unavailable; they are not filled with zero or linked across session boundaries. A missing adjacent action must not be converted into a switch/stay label.

Record the original labels, binary mapping, and positive class for each target. Verify the existing left/right convention rather than assuming its numerical sign. Any future nonbinary categorical target requires an explicit metric definition; the chance value of 0.5 below applies to the current binary targets.

### Metrics
Report both:
- **Cross-validated balanced accuracy**: primary, intuitive decoding metric.
- **Cross-validated ROC AUC**: secondary, threshold-independent metric.

For binary targets:

$$
\mathrm{Balanced\ Accuracy} =
\frac{\mathrm{Sensitivity}+\mathrm{Specificity}}{2}
$$

Use the unadjusted balanced-accuracy definition. The binary chance reference for balanced accuracy and ROC AUC is 0.5. [R4]

Use a prespecified positive-class probability threshold of 0.5 for balanced accuracy. Threshold tuning is not part of the current analysis. AUC must use held-out positive-class probabilities or consistently oriented continuous decision scores, not thresholded predictions.

When hyperparameters are tuned, select categorical models against inner-CV balanced accuracy. Report both balanced accuracy and AUC from that same selected model. Toggling the displayed metric must not silently select or refit a different model.

### Training Class Weighting
Fit classifiers without class weights or nonuniform sample weights. For scikit-learn, use `class_weight=None` and leave `sample_weight` unset. [R2]

Retain each session's observed class proportions. Do not oversample, undersample, or otherwise rebalance the classes. Different sessions may have different left/right proportions.

Balanced accuracy is an evaluation metric; it does not require class-balanced training. Record class counts for each training and test fold. Grouped stratification, when used to allocate existing trials to folds, is not a request for weighted training or a 50:50 resampled dataset.

## Numerical Regression Targets
Includes discrete counts/indices and continuous behavioral/model-derived variables. All use regression and cross-validated $R^2$.

### Discrete Targets
Evaluate the continuous-valued regression predictions directly; do not round predictions before calculating $R^2$.
- Consecutive omissions entering trial i.
- Consecutive rewards entering trial i.
- Current trial ix.
- Trial ix in block.
- Number of rewards in block entering trial i.

### Continuous Targets
- QL value entering trial i: `Qlearning_rel_value`.
- FQL value entering trial i: `FQlearning_rel_value`.
- HMM belief entering trial i: `HMM_rel_value_logodds`.
- HMM decay entering trial i: `HMM_rel_value_logodds_decay`.
- Relative doubt index entering trial i.
- Mouse speed, if available: optional future target; excluded from the current implementation scope.

Preserve these requested targets and their existing codebase definitions. The DV names alone do not establish their update timing or actual numerical scale. Verify those details under **Codebase Checks**.

### Target Timing and Availability
The counts labeled "entering trial i" must exclude trial i's outcome. Do not switch a target's definition halfway through the decoding time axis: the same trial-level value is decoded from every time bin, including bins after the alignment event.

The same **entering-trial-i** convention applies to QL/FQL values, HMM belief/decay, and relative doubt: use the model state available before the current choice, after processing history through trial i-1. Trial i's choice or outcome must not be used to update that target. Keep the entering-trial value fixed when decoding it from post-event bins as well. Verify whether each existing column already has this timing. Any necessary offset must be determined from the existing update logic and documented, not assumed or applied a second time.

Apply target-specific eligibility masks. A missing next-action label, for example, must not remove that trial from unrelated current-action decoding. For direct region/representation comparisons of a given target, use matched eligible trials as specified under **Cross-Validation**.

### Numerical Target Scale
**Do not standardize numerical targets.** Fit and score each target in its existing native units/scale. Do not add target z-scoring, a hidden target-rescaling wrapper, or an automatic transformation intended to equalize penalties across targets. Normal intercept fitting is still permitted; estimator-internal centering for that calculation is not a request to change the target scale.

### Metric
Use held-out $R^2$ for each outer test fold:

$$
R^2 = 1 -
\frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y}_{\mathrm{test}})^2}
$$

Here $\bar{y}_{\mathrm{test}}$ is the mean of the evaluated test-fold targets; it is not a fitted predictor used by the decoder. [R5]

Interpretation:
- $R^2 > 0$: less squared error than predicting the mean of the evaluated targets.
- $R^2 = 0$: equal squared error to that mean-prediction reference.
- $R^2 < 0$: more squared error than that reference.

Constant-target test folds and test folds with fewer than two observations have undefined $R^2$ and must be flagged. Do not substitute 0 or 1 for an undefined value; account for scikit-learn's `force_finite` behavior. [R5]

When hyperparameters are tuned, use inner-CV $R^2$ as the numerical-model selection metric. Targets and predictions remain in their native units in both fixed and tuned modes.

## Temporal Decoding
Decode each target independently from neural activity in event-relative time bins.

Available bin sizes:
- 500 ms.
- **100 ms: default.**
- 50 ms.
- 20 ms.

Alignment options are **choice time** and **trial start time**, using the same event definitions as the existing analyses. The requested analysis interval is **-2 s to +2 s** relative to the selected event.

For each trial $k$ and time bin $t$:

$$
\mathbf{x}_{k,t} = [r_{1,k,t},r_{2,k,t},\ldots,r_{N,k,t}]
$$

Here $r_{i,k,t}$ is the activity of unit $i$ in that trial/time bin. Define the exact activity representation (spike count or firing rate) and any existing smoothing before implementation; see **Codebase Checks**. Do not silently introduce smoothing or concatenate neighboring bins as decoder inputs.

A decoder for a selected time bin receives only that bin's features. Shared temporal PCA, described below, does not mean that the decoder receives the entire trial trajectory.

Bin size is selectable. Do not display all resolutions simultaneously in the primary figure. Generate a separate temporal-resolution comparison only when explicitly requested.

## Brain Regions
Allow decoding from:
- **PFC**.
- **HPC**.
- **PFC + HPC combined**.

The combined population evaluates whether joint recorded activity improves prediction relative to either recorded region alone.

For direct-unit decoding, concatenate the regions' unit-feature columns. For PC decoding, concatenate their separately computed PC scores; do not fit a joint cross-region PCA.

## Neural Representations

### 1. PCA Representation — Default
Use **shared temporal PCA, fitted separately for PFC and HPC**.

Within each training fold and region:
1. Form a PCA-fitting matrix with rows corresponding to **training trial x time-bin observations** across the selected analysis window and columns corresponding to units in that region.
2. **Z-score each unit once**, using its mean and standard deviation across that pooled training matrix. Apply these same training-derived statistics to held-out activity.
3. Fit one regional PCA basis to the z-scored training activity, with **PCA whitening disabled** (`whiten=False`).
4. Project training and held-out activity onto those training-derived regional axes. **Use the resulting PC scores directly**, without post-PCA z-scoring or a separate PC-centering/scaling step.
5. Fit separate elastic-net decoders for each target and time bin, and evaluate held-out predictions. Do not add a scaler to the per-time-bin decoder.

```text
PC branch, separately for each region:
unit activity -> one pooled training z-score -> PCA (whiten=False)
              -> per-time-bin elastic-net decoder using the PC scores directly
```

PCA's ordinary training-data centering remains part of PCA. The PC scores retain their component-specific variances; unit z-scoring does not make all PC variances equal. No PC-score standardization or scaling-mode toggle is requested. [R3]

PC axes are shared across time bins within a fold, not recomputed at each time bin. They are refit in a different fold. Do not use a full-session exploratory PCA for cross-validated decoding. The same training-only rule applies when PCA is refit inside inner CV. [R6]

The fitted basis may use post-event neural activity from training trials, but a pre-event test prediction still uses only the selected pre-event bin of the held-out trial. This is an offline shared-representation analysis, not a requirement to learn the representation online before the event.

### Regional PC Counts and Combined Features
Expose the retained PC count per region: **K_PFC** and **K_HPC**. **Default: 10 PCs per region.** A linked setting may set both to the same value. Do not tune PC count automatically.

```text
PFC:       [PFC_PC1, ..., PFC_PC_K_PFC]
HPC:       [HPC_PC1, ..., HPC_PC_K_HPC]
PFC + HPC: [PFC_PC1, ..., PFC_PC_K_PFC, HPC_PC1, ..., HPC_PC_K_HPC]
```

The default combined configuration therefore has **20 PC features: 10 PFC + 10 HPC**. There is no second joint PCA or standardization step after concatenation.

Use the same regional projections in standalone and combined analyses when the training trials and preprocessing settings match. Validate requested PC counts against the usable training dimensions; record the effective counts and do not silently change them.

The test set must not determine centering/scaling, PC directions, explained variance, or any other fitted preprocessing parameter. General exploratory PCA figures that do not report held-out performance may still use full-session PCA.

### 2. Individual-Unit Representation
Use individual units directly as neural features. Z-score each unit once using the pooled **training trial x time-bin** activity in the selected analysis window, and reuse those parameters across time bins. Apply the same training-derived transformation to held-out trials. There is no additional per-time-bin standardization. Reuse the same unit-scaling convention as the PC branch before PCA.

Keep original unit identifiers and region membership attached to the feature columns. Explicitly record constant or unavailable features rather than confusing them with features excluded by the elastic-net penalty.

## Decoder Family and Regularization — Both Representations
Use **Policy B**: the same decoder family for PCs and individual units.

| Target type | PC features | Individual-unit features |
| --- | --- | --- |
| Categorical | Elastic-net logistic regression | Elastic-net logistic regression |
| Numerical, including counts/indices | Elastic-net linear regression | Elastic-net linear regression |

Do not maintain separate ridge or unpenalized model options. If the selected tuning range includes a pure-L2 endpoint, it remains an endpoint of this elastic-net configuration, not a separate representation.

The conceptual penalty is:

$$
\lambda \left(
\frac{1-\alpha}{2} \sum_j \beta_j^2 +
\alpha \sum_j \lvert \beta_j \rvert
\right)
$$

In this equation, $\lambda$ is overall penalty strength, and $\alpha$ is the L1/L2 mixing fraction. The intercept is separate from the penalized feature coefficients.

### Implementation Parameter Names
Do not confuse the equation's $\alpha$ with scikit-learn's `ElasticNet.alpha`.
- **ElasticNet regression:** `alpha` corresponds to the equation's $\lambda$; `l1_ratio` corresponds to its $\alpha$. [R1]
- **LogisticRegression:** `C` controls inverse penalty strength, and `l1_ratio` controls the L1/L2 mixture. There is no regression-style `alpha` constructor argument. Do not assume that equal numeric regression and classification settings give equal regularization. [R2]

Use an elastic-net-compatible logistic solver (SAGA in scikit-learn), with parameters compatible with the version installed in the codebase. Record the actual estimator settings. Do not rely on default logistic-regression regularization to implement this policy. [R2]

### Fixed and Tuned Modes
Provide both modes for **both** representations:

**Fixed mode: default operating mode.**
- Numerical regression: retain the supplied scikit-learn settings `alpha=1.0`, `l1_ratio=0.5`.
- Classification: **`C=1.0`, `l1_ratio=0.5`**. No class weights or sample weights.
- Fixed settings are prespecified, not chosen from held-out scores. No inner search is required, but outer CV is still required.

**Tuned mode: optional.**
- Select penalty strength and mixing fraction using inner CV restricted to the outer training trials.
- Preserve the required grouping in inner splits.
- Refit all learned preprocessing, including the shared temporal regional PCA, within each inner training split.
- After parameter selection, refit preprocessing and the decoder on all outer-training trials before evaluating the outer test fold.
- Tune each target/time-bin/region/representation model separately; keep the outer evaluation assignments matched for comparisons.
- Inspect and retain selected estimator parameters for every fitted outer-fold model.

No fixed or tuned setting is guaranteed to produce nonzero coefficients. Record all-zero solutions and convergence problems rather than silently weakening the penalty or changing model family. Numerical targets remain unscaled: a common fixed `alpha=1.0` need not exert comparable shrinkage on a trial index and a bounded DV. Retain this as an interpretation note, not a request for automatic target scaling or penalty adjustment. [R1]

## Elastic-Net Unit-Weight Visualization
For a selected target variable, time bin, and region configuration, display the **direct-unit decoder's** fitted coefficients.

Show:
- **Signed coefficients on the training-standardized feature scale**, with the response/coefficient units labeled. A feature increment corresponds to one training standard deviation using the documented scaling scope.
- **Median signed coefficient and IQR across outer CV folds for each unit**, including fitted zeros. Individual fold coefficients remain inspectable.
- **Units sorted by mean absolute coefficient across folds**. Take the absolute value before averaging for sorting; retain signed values in the actual display.
- **Selection frequency for each unit:** the fraction of evaluable outer-fold fits with a nonzero coefficient.
- **Number and fraction of nonzero unit coefficients within each outer-fold fit**, with a summary across folds. Do not count nonzeros only after averaging coefficients.
- **Unit identity, region, and contributing-fold count**, including any exclusions due to constant/unavailable features or failed fits.

Define nonzero using a documented small numerical tolerance in the fitted coefficient scale, not a significance threshold. The proposed inspection default is `abs(coef) > 1e-8` (see O5 recommendations). Keep this tolerance distinct from the solver's convergence tolerance. Denominators for selection and sparsity summaries must be explicit; unavailable fits/features are not evidence of exclusion by regularization.

For classification, orient signs toward the recorded positive class and label coefficients as log-odds change per pooled training standard deviation of the unit feature. For regression, use native target units per pooled training standard deviation of the unit feature. Numerical targets are not standardized. Summarize coefficients within the selected target; do not interpret raw coefficient magnitudes across differently scaled targets as comparable encoding strengths.

Label coefficient IQR as **variation across CV fits**, not a confidence interval or a significance test.

These outputs assess **decoder reliance**, not the total number of neurons encoding a variable. Correlated units may carry redundant information even when some receive zero weight.

For the PC branch, coefficients and nonzero counts refer to **PC features**, not individual neurons. Do not label PC coefficient sparsity as unit selection. A separate back-projected unit-weight analysis is not requested here.

## Cross-Validation
Use outer CV for all reported decoding metrics, including runs with fixed regularization settings.

### Grouping and Matching
Keep behavioral blocks intact and assign multiple blocks to each test fold. For categorical targets, validate that both training and test sets contain the two required classes. Apply the same validation within inner CV. Grouped stratification is acceptable when feasible, but validate the actual folds rather than assuming the splitter guarantees class coverage. [R7]

For a given target, use the same outer trial assignments across time bins, regions, and representations, with matched eligible trials for direct comparisons. Target-specific missing labels may produce different eligible sets across different targets.

Keep all observations from a trial within its assigned fold, including the observations pooled over time to fit shared PCA. For either selected alignment, the user confirms that trials' -2 s to +2 s neural windows do not overlap by task design; no overlap-handling logic or associated trial exclusions are required.

If valid grouped splits cannot be formed, report the target/configuration as unavailable with the reason. Do not silently fall back to random trial splitting. An explicitly requested random-stratified comparison may remain available, labeled as a different evaluation scheme.

### Fitted Transformations
Learn all fitted transformations from the relevant training subset only:
- Unit-feature scaling and any pre-PCA preprocessing.
- Regional PCA directions.
- Hyperparameter selection, when enabled.

The corresponding validation/test data are transformed but never used to fit those quantities. Numerical targets are not scaled. [R6]

### Score Aggregation
For categorical and numerical targets alike, calculate the metric separately in each outer test fold. Display the **unweighted arithmetic mean of fold scores**, retaining individual fold scores, sample counts, and validity status.

Do not replace this with a pooled out-of-fold metric without explicitly labeling it as a different summary. Undefined or failed fold scores are not zero, chance, or perfect performance. The **proposed** inspection policy is to display a primary heatmap cell only when all requested folds are valid; otherwise mark that cell unavailable while preserving valid individual-fold results. See O5 recommendations. Report valid/expected fold counts, and continue processing unrelated cells rather than aborting the whole session.

This unweighted fold average is distinct from training class weighting. Balanced accuracy still averages class-specific recall within each test fold.

## Primary Visualization: Neural Information Over Time

### Categorical Heatmap
- **Rows:** decoded categorical variables.
- **Columns:** event-relative time.
- **Color:** mean outer-fold balanced accuracy, with ROC AUC available as a metric toggle.

Show the binary chance reference of **0.5** on the color scale. Use the same fitted predictions when switching between metrics.

### Numerical Regression Heatmap
- **Rows:** all decoded numerical targets, including both discrete counts/indices and continuous DVs.
- **Columns:** event-relative time.
- **Color:** mean outer-fold $R^2$.

The categorical and numerical heatmaps remain separate and aligned; their metrics must not share a common numerical color scale.

Preserve below-chance classification values and negative $R^2$. Unavailable results must look different from chance/baseline values. Make trial counts and valid-fold coverage inspectable.

The reference values 0.5 and 0 are not significance thresholds. This spec does not add permutation significance testing or a statistically established "decoding onset" analysis.

## Figure Controls
- **Representation:** PCs (default) or individual units; both use elastic net.
- **Bin size:** 500 / 100 / 50 / 20 ms; default 100 ms.
- **Alignment:** choice time or trial start time; interval -2 s to +2 s.
- **Region:** PFC / HPC / PFC + HPC.
- **Categorical metric:** balanced accuracy (default) / ROC AUC.
- **PCA:** retained PC count per region, optionally linked; default 10 per region.
- **Regularization mode:** fixed (default) / tuned by inner CV.
- **Regularization details:** inspect fixed or selected `l1_ratio` and regression `alpha` or classification `C`, labeled by estimator.

Changing the metric display alone must not retrain a model. Preprocessing, target timing/scale, and default PC counts are resolved. Remaining CV/tuning/numerical recommendations are listed below rather than silently adopted as user decisions.

## Interpretation
The main analyses answer:
- **Temporal decoding:** When is each task variable predictable from the selected neural activity bin?
- **Regional decoding:** Does PFC, HPC, or the combined recorded population better predict the target?
- **Population representation:** Is prediction captured well by the shared regional PCs, or does direct unit-level decoding improve performance?
- **Decoder reliance:** Does the fitted direct-unit decoder distribute weight broadly or concentrate it in a smaller subset?
- **Temporal resolution:** How does decoding depend on the neural bin size?

Additional interpretation notes:
- Regional comparisons use the recorded populations as supplied. Do not describe their differences as information per neuron. The combined PC model has K_PFC + K_HPC features.
- Separate decoding of belief, choice, and context does not establish unique representations independent of the other variables. Trial-index decoding could reflect slow population changes; no additional confound-control analysis is requested here.
- Model weights describe fitted predictive relationships, not causal contributions or a census of encoding neurons.
- Shared temporal PCA is fitted without held-out neural data, but it intentionally uses the selected training time window. Its interpretation differs from a strictly online representation available before the event.

## Resolved Review Decisions
- **O1:** Z-score units once using pooled training trials and time, then fit separate shared temporal regional PCAs with whitening disabled. Use PC scores directly, without additional PC standardization, a separate PC-centering step, a per-time-bin scaler, or a PC-scaling toggle.
- **O2:** Do not standardize numerical targets; preserve their native scale.
- **O3:** Fixed classifier `C=1.0`, with `l1_ratio=0.5`.
- **O4:** Model-derived values/beliefs/doubt refer to the state entering trial i, as do the history counts. Verify source-column timing before applying any offset.
- **O5, PC count:** Default 10 PCs per region, giving 20 separately projected features for PFC + HPC.

## Recommended Inspection Defaults — Proposed
The user requested recommendations for the remaining O5 settings. The following are deliberately small, interpretable operating choices, not claims of optimal statistical settings. They are proposals for review rather than additional user-approved decisions.

### O5a. Outer and Inner Cross-Validation
- **Outer CV: 5 grouped folds, one pass, no repetitions.** Keep complete behavioral blocks in a single fold.
- **Inner CV: 3 grouped folds, only in optional tuned mode.** Fixed mode does not run an inner search.
- **Categorical group allocation:** use the existing compatible `StratifiedGroupKFold` implementation, with shuffling disabled for a deterministic split. It attempts to retain the observed class proportions while keeping blocks intact; validate actual class coverage. [R7]
- **Numerical group allocation:** use standard deterministic `GroupKFold`, without discretizing regression targets into classes. [R8]
- **Random seed: 0** wherever an estimator or PCA solver uses randomness. The proposed deterministic splitters themselves require no random shuffle/seed. Record the actual split assignments; do not try multiple seeds and select a favorable result.
- Use the same outer assignments across all time bins, representations, and regions for a given target. Inner assignments must also be common across hyperparameter candidates in the same search.
- Expose fold counts as simple parameters. If a valid 5-fold grouped split is impossible, report the reason and allow an explicit rerun with 3 folds. Do not add an automatic fold-count/seed search or quietly switch to random trial CV.
- If a valid 3-fold inner split is impossible, report tuned mode unavailable for that configuration; do not silently substitute fixed-mode results under a tuned-mode label.

### O5b. Optional Tuning Grid
Keep **fixed mode as the default**. Use a small explicit grid only when tuned mode is requested:

| Estimator parameter | Proposed candidates |
| --- | --- |
| Logistic-regression `C` | `0.01, 0.1, 1.0, 10.0, 100.0` |
| Numerical ElasticNet `alpha` | `0.001, 0.01, 0.1, 1.0, 10.0` |
| `l1_ratio`, both target families | `0.1, 0.5, 0.9` |

This is 15 candidate configurations per model, evaluated in the same 3 inner folds. Do not tune PC count, bin size, feature scaling, class weights, or decision thresholds as part of this search. Do not expand the grid automatically when the selected value is at an endpoint; simply record the result. The regression grid is a starting inspection range in the native target scale, not a scale-invariant optimum. [R1]

Choose the candidate with the highest mean inner score over all required inner folds. An invalid or nonconverged candidate is not ranked from only its surviving folds. A deterministic candidate order is sufficient to resolve exact score ties; no additional selection procedure is requested.

### O5c. Undefined or Failed Folds
- Display the primary mean score only when **all requested outer folds** have valid fits and defined metric values.
- If one or more folds fail or are undefined, show the affected primary heatmap cell as unavailable, with a short reason and valid/expected fold count. Retain the individual valid fold results for inspection.
- Do not turn undefined $R^2$ into 0/1 or failed classification into chance performance. Negative finite $R^2$ and below-chance classification scores remain valid results, not failures. [R5]
- Apply the same complete-fold rule to all configurations. Do not construct direct comparisons from different surviving fold subsets.
- A failed cell must not abort unrelated targets/time bins/regions. No incomplete-fold averaging mode or automatic model-family/penalty fallback is needed for this inspection spec.
- An all-zero coefficient solution with a converged fit is still a valid fitted model and should be scored. Sparsity alone is not a fit failure.

This strict cell-level rule is proposed for simple interpretation: every displayed mean uses the requested folds, while unavailable cells remain visibly distinct. It avoids adding a separate partial-coverage scoring policy.

### O5d. Numerical Coefficient Tolerance
For selection-frequency and sparsity summaries, use **`abs(coef) > 1e-8`** as the proposed nonzero rule. Evaluate it separately in each fitted outer-fold model, excluding the intercept. Record the threshold and the coefficient scale.

This is a numerical display/counting tolerance, not a biological effect-size or statistical-significance threshold. Keep it separate from the estimator's convergence tolerance. No tolerance-tuning analysis is requested.

## Codebase Checks Before Finalizing Implementation
- Resolve the actual source columns for categorical targets, count/index targets, and relative doubt; preserve original unit and trial identifiers.
- Verify the numerical and left/right sign conventions of all named DVs, including whether a field named `logodds` stores raw log odds or a transformed value.
- Verify what counts as an omission in the existing features, including treatment of incorrect unrewarded trials, streak resets, block resets, and trial-index origin. Do not replace those definitions without an explicit decision.
- Verify that "current state" and block identifiers refer to the context governing the current choice, not a state scheduled for the next trial.
- Confirm the binned activity representation, any smoothing already applied, missing-neural-data handling, and availability of the full selected event window.
- Confirm installed estimator APIs and solver support. Reuse existing pipeline components when they meet the agreed statistical requirements.

## Revision Summary
Changes from revision 2:
- Simplified preprocessing to a single pooled training-data unit z-score followed by regional PCA (`whiten=False`); the decoder uses PC scores directly.
- Removed PC-score standardization, its alternative scaling mode, and the corresponding control and fitted-transformation requirements.
- Removed overlap detection and boundary-trial exclusion recommendations; retained one task-design statement in the CV section.
- Preserved all other confirmed settings and left the remaining O5 recommendations explicitly proposed rather than adopted.

## Technical References
These references support estimator behavior and numerical definitions only. They do not resolve user-specific analysis choices or verify the local codebase.

- **R1:** scikit-learn, ElasticNet: objective and parameter naming. `https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html`
- **R2:** scikit-learn, LogisticRegression: `C`, `l1_ratio`, class weighting, and solver compatibility. `https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html`
- **R3:** scikit-learn, PCA and StandardScaler: input centering versus scaling and learned scaling parameters. `https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html` and `https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html`
- **R4:** scikit-learn, balanced_accuracy_score: unadjusted class-recall average. `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html`
- **R5:** scikit-learn, r2_score: evaluated-target reference and undefined constant-target cases. `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html`
- **R6:** scikit-learn, Common pitfalls: training-only learned transformations and leakage. `https://scikit-learn.org/stable/common_pitfalls.html`
- **R7:** scikit-learn, StratifiedGroupKFold: grouped splitting and limits on achievable class stratification. `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html`
- **R8:** scikit-learn, GroupKFold: non-overlapping groups and deterministic default splitting. `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html`
