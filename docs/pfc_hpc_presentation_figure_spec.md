# PFC and HPC population decoding and interregional prediction

Draft figure specification for discussion. The presentation asks three questions. What can each region tell us about a single choice? Which readouts still work when choice and context come apart? How much activity in one region can the other predict, and how does their shared activity relate to movement?

## Figures 1 and 2. Context and choice in PFC and HPC

Build the same figure for each region using one named session. Keep scales, trial definitions, evaluation folds and panel order identical so the slides can be alternated or placed side by side.

| Panel | Training trials | Decoded target | Held-out test groups |
| --- | --- | --- | --- |
| A | All eligible outcomes | Recorded task context | Overall, correct rewarded, omission, incorrect |
| B | All eligible outcomes | Actual chosen side | Overall, correct rewarded, omission, incorrect |
| C | correct rewarded only | Recorded task context | Overall, correct rewarded, omission, incorrect |
| D | correct rewarded only | Actual chosen side | Overall, correct rewarded, omission, incorrect |

An omission is a correct choice for which reward was withheld. Incorrect choices remain a separate group even when unrewarded. Audit unexpected reward flags, unknown labels and manual rewards separately. Overall pools eligible held-out trials. Score all four groups with the same fitted readout per fold. Use the same held-out trials for all-trial and correct rewarded training comparisons.

**Interpretation.** If correct choice is determined by context, choice and context labels coincide up to label coding on correct rewarded trials. With identical features, windows, weights and penalties, C and D can therefore be the same readout up to sign. Incorrect trials distinguish whether that axis follows the chosen side or the context-appropriate side. Show below-chance scores and check the context-by-choice counts before interpreting either panel. Correct omissions isolate missing reward while preserving the correct action.

**Primary endpoint.** For context and choice, train the existing bin-wise L2-regularized logistic decoder across the full choice-relative window [-1, +1) seconds. For each complete held-out choice, average its bin probabilities for each class, predict the class with the largest mean, and score the choice once. Show balanced accuracy (mean class recall) overall and within each evaluation group, with trial counts and a 50% binary chance reference. Mark scores unavailable when a class is absent.

**Validation.** Use whole-block holdout for all regional decoders. Assign blocks once per session and reuse the assignments across regions, targets, PC counts and bin widths. Assess class support in each training split and in the pooled held-out trials for each reported score. Keep every bin from a trial together, purge overlapping time windows, and fit scaling and PCA on training data only. Record trial weighting and aggregate held-out predictions before computing subgroup scores. For context, report how many independent blocks support each class.

### Decisions for Figures 1 and 2

The unit selection, scoring window, bin widths, estimator and whole-block holdout are settled. Apply the same answers to Figure 1 (PFC) and Figure 2 (HPC).

| Decision | Proposed figure contract | Why the choice matters |
| --- | --- | --- |
| Example session and units | Proposed example session is ct026 8/28. Use all manually labeled good units in each region and the same eligible choices in PFC and HPC for each readout. Print each region's unit count. | Matching trials makes the regional comparison interpretable. Keeping each region's full good-unit population describes the recorded populations, which may differ in size. |
| Window (settled) | Use the full two-second window for the primary context and choice score. Primary figures use 500-ms bins (four per choice). The regional decoding analysis and plots must also work at 100 ms (twenty per choice), with a separate fit at each width and identical scoring rules. | As secondary summaries, average probabilities separately in the pre-choice and post-choice halves using the same fitted model and held-out choices. |
| Bin-level companion | Plot held-out balanced accuracy against choice-relative bin time. Within each bin score one prediction per choice, using the same choices and evaluation groups as the primary score. | Retain the individual bin probabilities for replay. Label this curve separately from the single full-window accuracy. |
| Estimator (settled) | Use the existing population decoder's fixed L2 logistic regression for binary and four-class targets, for both unit and PC inputs. | Use the model's class probabilities as evidence throughout the regional figures. |
| How is generalization tested? | Use whole task blocks for outer holdout. Within each split train either on all eligible training trials or only on correct rewarded training trials. Pool held-out predictions across blocks before splitting by evaluation group. | A context can be constant within a block. Report independent training blocks per context and mark folds without both training labels unavailable. |
| Which comparisons appear together? | Arrange A-D as two rows (training set) by two columns (target). Each panel has four labeled dots on the same 0-100% scale. Put left/right counts by evaluation group in a common strip. | The overall dot uses the observed outcome mix. Its value need not equal the average of the three subgroup dots. |

For C and D, inspect the context-by-choice contingency table on the training trials. If the labels coincide up to sign and the fitted settings match, label the shared readout explicitly. Confirm whether to retain both panels for visual symmetry or show that axis once with both held-out scoring rules.

## Figure 3. Trial outcome and decoding dimension

**A. What happened after choice?** In each region decode the joint outcome used by population task decoding: left rewarded, right rewarded, left unrewarded and right unrewarded. Pool training across eligible trials. Show a row-normalized held-out confusion matrix per region with true outcome on rows, predicted outcome on columns and counts beside rows. This reveals whether errors confuse side or reward. Score with mean class recall (four-class reference 0.25). Incorrect unrewarded choices contribute to the appropriate side's unrewarded class. The correct rewarded/omission/incorrect labels define the evaluation groups in Figures 1 and 2.

**B. How many PCs recover the decoding performance?** Repeat every regional readout in Figures 1-3A with the first 1 through 10 training-fold PCs. Plot held-out choice-level balanced accuracy against retained PCs, with the all-unit model as a horizontal reference on identical trials. Use one small panel per target and training regime, show overall curves on the slide and outcome-stratified curves for the binary readouts in backup. Suppress unsupported dimensions. Report the smallest k for which the score at k and at every larger supported count is at least the all-unit score minus five percentage points. For example, an all-unit balanced accuracy of 80% gives a recovery threshold of 75%. Report the threshold as unreached if no supported count qualifies. Flag an all-unit baseline near chance so readers can judge what was recovered.

### Decisions for Figure 3

| Panel | Figure contract | Decision or display detail |
| --- | --- | --- |
| A. Four outcomes | Use two 4-by-4 matrices with a shared 0-100% color scale. Use the same row and column order in each region. Average the four-class probability vectors within the first second after each held-out choice (two bins at 500 ms or ten at 100 ms). Enter one prediction per choice and print row counts and balanced accuracy. Show bin-level accuracy separately. | Choice alignment and the first post-choice second are settled. Give recorded delivery latency a brief caption annotation at 500 ms. At 100 ms, small delivery markers can help interpret individual bins. |
| B. PC recovery | Use a separate training-fold PCA basis for each region and training regime. For transfer, estimate that basis using only correct rewarded training trials. Use the same folds, evaluated bins, unit pool and scoring rule for the all-unit reference and every PC count. | Fit PCA using exactly the training trials and supervised time window of the decoder: [-1, +1) seconds for context and choice, [0, +1) seconds for outcome. |
| B. Slide layout | Put PC results on a companion slide if the matrices and curves are unreadable together. Use five facets (context and choice under both training regimes, plus outcome). Mark the recovery count on each overall curve. Show matched subgroup curves in backup. | Overlay regions with distinct line styles, or use aligned regional rows? Proposal is regional rows so each curve has an unambiguous all-unit reference. |

On the main slide, beside each regional matrix, report reward discrimination within each chosen side. Within actual left choices and actual right choices separately, compare the rewarded and unrewarded probabilities for that side from the same four-class model. Use the trial-averaged probabilities, report balanced accuracy and both trial counts, and mark a side unavailable if either reward class is absent. This reveals whether the decoder distinguishes reward once chosen side is known.

Determine recovery per session, region and readout. If fewer than ten PCs are supported across the matched folds, print the tested range beside the recovery count. If the all-unit model is near chance, display its actual score beside the count.

## Figure 4. Replay two switches and a four-outcome sequence

**A1 and A2. Switch replays.** Use one common elapsed-time axis for PFC and HPC. Stack programmed context, actual choices, recorded reward events, held-out choice scores and held-out context scores. For each excerpt, keep each readout fixed across the displayed interval. Mark the switch, the first choice on the newly correct side and the first recorded reward delivery on that side. Show trials before the switch and after that delivery. Examine whether choice evidence follows the mouse's actions and whether context evidence shifts before, around or after delivery. Collection time is unavailable in these sessions. A fit restricted to correct rewarded trials cannot establish distinct choice and context axes here by itself.

**B. Outcome replay.** Align the four outcome traces to choices and mark recorded reward deliveries, chosen side and trial type. Include correct rewarded, omission and incorrect trials where available. Display scores only in supported post-choice windows, with gaps elsewhere. Label any expected reward time on omission trials as expected. Use a fixed example-selection rule based on behavior and recording coverage before viewing neural scores. Caption each replay with mouse, session, trial IDs and time scale. Export a static frame showing the whole sequence alongside the live view.

### Replay selection and display

- **A1. New-side omission.** Select a switch where the mouse chooses the newly correct side and experiences an omission before its first reward delivery on that side. Include both that omission and the later correct rewarded choice in the replay.
- **A2. Ideal transition.** Use the existing ideal-transition definition: a correct rewarded old-side choice immediately before the block switch, one to four incorrect old-side repeats, then a correct rewarded new-side choice. Label the actual repeat count. Use the same tracks, probability scale and event markers as A1.
- **Selection and resolution.** Select each motif using behavior and joint recording coverage. Proposed starting session is ct026 8/28, with the other named ct026 sessions available if needed. Proposed switch span is three choices before the switch through three choices after the first new-side delivery. Open every Figure 4 replay at 500 ms and allow selection of the separately fitted 100-ms result for the same excerpt.
- **Choice and context traces.** Proposed replay uses all-trial-trained models from Figures 1 and 2, with both blocks around each displayed switch held out together and both contexts represented in the remaining training data. Keep left/right score orientation fixed across regions. Draw each estimate across its measured bin and leave uncovered time blank. Plot the bin probability of right choice or right context, as appropriate.
- **B. Outcome selection.** Show one extended, contiguous held-out sequence containing left rewarded, right rewarded, left unrewarded and right unrewarded choices. Preserve intervening trials and include surrounding choices so the sequence shows how the probabilities evolve. A same-side rewarded/unrewarded pair is a secondary annotation within that sequence. Use four class traces per region with identical colors and a shared vertical scale. If a candidate session lacks a qualifying sequence, search the other named sessions and report any remaining coverage gap.
- **Illustrative claim.** Describe when choice and context evidence change relative to recorded choices and reward delivery in each displayed episode, at the stated bin resolution. Limit timing conclusions to those episodes.

## Figure 5. Predicting one region from the other

Pool ordinary choices with simultaneous coverage in both regions. Predict unsmoothed target-neuron spike counts from source activity in the same 100-ms bin, from two seconds before to two seconds after choice. Use manually labeled good units. Show all four directions in every panel, grouped by target so the within-region and between-region models predict exactly the same neurons.

| Target population | Within-region prediction | Between-region prediction |
| --- | --- | --- |
| PFC | PFC to PFC | HPC to PFC |
| HPC | HPC to HPC | PFC to HPC |

**Matched populations.** Follow the existing communication-subspace sampling design. In each of five seeded draws, take disjoint source and target groups of n good units per region, where n is half the smaller region's eligible unit count, rounded down. A target neuron can never serve as its own predictor. Within a draw, reuse each regional source pool, target group, trial set and block assignment across all four panels. Print the available and sampled unit counts.

| Panel | Horizontal axis | Comparison |
| --- | --- | --- |
| A | Number of source units | Add nested subsets from each drawn source pool at 1, 2, 4, 8 and successive powers of two, plus n. Keep each target group fixed. |
| B | Number of source PCs | Fit PCA on the full drawn source pool using training data only. Include counts 1 through 10 and the larger counts from A where supported. |
| C | Regression rank using units | Vary the rank of the predictive mapping using the full drawn source pool. Show unrestricted ridge on those same units as the reference. |
| D | Regression rank using ten source PCs | Start with the leading ten training-fold PCs. Show both unrestricted ten-PC ridge and the all-unit reference from C. Label unsupported ten-PC fits unavailable. |

**Validation.** Hold out whole task blocks and use identical splits across directions, draws and panels. Purge training windows that overlap held-out windows. Estimate scaling, PCA and regularization using training blocks only.

**R-squared in Figures 5 and 8.** For each target neuron, the baseline prediction for every bin in a held-out block is that neuron's mean spike count per bin across the eligible training bins for that fold. Compare the model and baseline on exactly the same held-out bins. Square each prediction error and sum across all evaluated bins and blocks for that neuron, then calculate:

`R-squared = 1 - (sum of squared model errors / sum of squared training-mean baseline errors)`

A score of 1 indicates perfect prediction, 0 means the model matches the baseline's total squared error, and a negative score means the baseline predicts better. Retain negative values. If the baseline error sum is zero, that neuron's score is undefined. Report how many neurons contribute. Within each population draw, take the arithmetic mean of the defined neuron scores so each target neuron has equal weight. Average draw scores within session. Figure 8 uses those session scores and the session-support rule below. The existing inspection uses held-out means and chronological choice folds, which give its saved R-squared values a different interpretation.

**Reading the curves.** Use unrestricted ridge for A and B and ridge-regularized reduced-rank regression for C and D. Sweep feasible integer ranks. Plot a mean over population draws and their range, with draw support visible. Compare ranks descriptively against the unrestricted references. The direction labels identify the population used to predict a target. Same-bin prediction can include shared responses to task events.

## Figure 6. Shared regional activity and movement

**A. Summary.** Hold out one whole task block at a time. Fit the first canonical pair using only good-unit PFC and HPC activity in the remaining blocks. Use the same choice-aligned 100-ms bins as Figure 5, from two seconds before to two seconds after each eligible choice. Average treadmill speed magnitude within each bin, taking the absolute value of signed velocity before averaging. Report speed units and joint coverage. Whole-session movement is a future analysis.

For each held-out block, plot three zero-lag Pearson correlations: PFC score with HPC score, PFC score with speed, and HPC score with speed. Show one dot per block and a median for each correlation, with the number of contributing blocks. Mark a correlation unavailable when either series is constant or valid paired observations are insufficient.

### Why the sign of a canonical pair needs a convention

CCA learns two sets of neural weights, one for PFC and one for HPC. Applying them produces two score traces. Multiplying both sets of weights by -1 turns both traces upside down. Peaks become troughs together, so the PFC-HPC correlation stays exactly the same. Speed is measured independently and keeps its sign. Consequently, both correlations with speed reverse sign. CCA alone has no preference between these two orientations.

For example, imagine the training correlations are +0.70 for PFC with HPC, -0.30 for PFC with speed, and -0.20 for HPC with speed. After flipping both neural traces, the three values are +0.70, +0.30, and +0.20. These are two descriptions of the same neural fit. Speed enters only after fitting, to give the vertical axes a consistent meaning across blocks.

**Orientation rule.** For each fit, compute PFC-score versus speed correlation on valid training bins. If it is negative, multiply both regions' score weights and resulting traces by -1. If it is positive, keep both signs. Apply that same orientation to training and held-out projections and to the replay. Preserve the measured speed trace. The HPC-speed sign follows from the joint flip and can have either sign. If the training anchor is zero or undefined, mark signed movement results unavailable for that fit. The PFC-HPC correlation can still be reported when defined.

A positive held-out PFC-speed correlation means the association has the same direction as in training. A negative value means it reversed in that block. Preserve either result. Keep the training anchor correlation available because a value close to zero gives a weak basis for orienting the traces. The convention aligns signs across fits. The underlying neural weights may still vary.

**B. Live example.** Select a short sequence of consecutive choices within one held-out block using speed and joint coverage alone. Seek low movement and a visible speed increase inside the analyzed windows. The excerpt may differ from Figure 4. Display the two oriented scores and measured speed on a shared session clock using that block's fixed training pair. Mark choices and reward deliveries, leave gaps outside analyzed windows, and caption the session, block, time scale and speed units.

## Figures 7 and 8. Reproducibility across sessions of one mouse

Start with the four webapp-ready sessions ct026 8/3, ct026 8/4, ct026 8/10 and ct026 8/28. Use this roster for the initial summaries and as the candidate pool for the replays. Add further ct026 sessions as metadata, spike sorting and alignment become ready. Print the mouse ID and every included session ID on the slide or its accompanying roster. For each session record metadata path, PFC/HPC assignments, sorted-spike availability, alignment/coverage, usable trial counts by group and good-unit counts by region. Give panel-specific denominators and reasons for missing points.

**Figure 7. Regional readouts.** Repeat Figures 1-3 per session. Use named session dots grouped by target and outcome, connect paired measurements within session, and use thin session curves for the PC sweep. Add a session median as a descriptive reference. Keep replays at the single-session level.

**Figure 8. Interregional prediction.** Show each named session's unit-count, PC-count and rank curves in separate panels by direction. Draw each curve only over its supported range and show how many sessions contribute at each count.

### Figure 7 display

- **7A, context and choice.** Show overall, correct rewarded, omission and incorrect scores on the main slides for both training regimes. Give PFC and HPC separate rows. Connect paired training regimes only within the same session, region, target and evaluation group. Split context and choice across slides if space is tight.
- **7B, outcome.** Show three scores per session and region: four-state balanced accuracy, reward discrimination within left choices, and reward discrimination within right choices. Keep the four-state confusion matrices in backup, labeled by session and region.
- **7C, PC recovery.** For each session, region, target and training regime, show the overall recovery count alongside the corresponding all-unit balanced accuracy. Use all eligible held-out choices for that readout. If the tolerance is never reached, label the last supported count (for example, "unreached through 6"). Mark an unavailable score separately with its reason. Keep overall and outcome-stratified curves in backup where defined.
- **Across panels.** Identify sessions directly or by a consistent date legend and give each session equal weight in descriptive medians. Report missing scores at the affected panel.

### Figure 8 display

- **8A and 8B, prediction by source size.** Show source unit and PC count on the first slide. Group all four prediction directions by target region, pairing within-region and between-region predictions of the same target neurons. Use absolute source counts.
- **8C and 8D, prediction by rank.** Use a companion slide with the same target grouping. Show each session through its supported ranks. Label the source-unit pool size for unit-input curves and use ten source PCs for PC-input curves.
- **Session support.** Average population draws within each session first. Plot each named session across its full supported range and print the number of contributing sessions. Draw the equal-weight session median only at counts or ranks supported by the entire figure roster (initially all four ct026 sessions). If three sessions support the ten-PC panel, show those three curves with a 3-of-4 label.

## Existing analyses and decisions for the first draft

Use the [cross-condition readouts](../src/neural_analysis/task_decoding/cross_condition/README.md) for transfer definitions. The documented outcome-transfer extension covers Context. Choice transfer, pooled overall scoring and the matched PC sweeps need explicit verification. The [population decoding contract](population_task_decoding_spec_v1.md) defines the four post-choice states. Check estimator, time support and saved score type before reusing its results.

The [interregional regression](../src/neural_analysis/interregional/README.md) provides held-out unit and fold-local PC regression. The [communication subspace inspection](../src/neural_analysis/communication_subspace/README.md) provides matched draws and rank comparisons. Verify fixed-target scoring and PC-input rank support before combining these outputs. The [CCA timeline](../src/neural_analysis/SESSION_CCA.md) supplies the replay convention.

Before figure production, select replay intervals from the starting cohort. Verify delivery timestamps for the selected intervals. Decide on light or dark slides. Each figure caption should name the estimand, event alignment, units, held-out sample size and the observation the data actually support.
