# Differences Between LM-HMM Notebook and Project Cross Validation

I compared LM-HMM cross validation in these three files:

- `src/behavior_analysis/block_state_space_modeling.py`
- `notebooks/2c Input-driven linear model (LM-HMM).ipynb`
- `notebooks/LM-HMM_marimo.py`

## Summary

The original `2c` notebook and the Marimo port are methodologically aligned. The
project LM-HMM module is not a direct implementation of that notebook cross-validation
procedure anymore. It differs in how folds are constructed, what scores are returned,
how scores are normalized, what defaults are used, and even which predictors are used.

## 1. Fold construction

- The `2c` notebook and `LM-HMM_marimo.py` use `StratifiedKFold` on stacked synthetic data.
- Stratification is based on session/mouse labels (`ylabel_mouse`), so the folds are
  trial-level splits stratified by session identity.
- `block_state_space_modeling.py` does not use `StratifiedKFold`.
- Instead, it uses contiguous folds created with `np.array_split(np.arange(n_timesteps), n_folds)`.

This is a substantive methodological difference, not just a refactor.

## 2. What is being compared during cross validation

- The `2c` notebook and `LM-HMM_marimo.py` fit both `MLE` and `MAP` models inside each fold.
- They return four metrics:
  - training log likelihood for MLE
  - held-out log likelihood for MLE
  - training log likelihood for MAP
  - held-out log likelihood for MAP
- `block_state_space_modeling.py` cross-validates one algorithm family at a time using
  the `algorithm` argument.
- It returns only held-out log likelihoods for that chosen family.

So the notebooks use cross validation for both state-count comparison and estimator
comparison, while the project module uses it only for state-count comparison within one
estimator family per run.

## 3. Training-score reporting

- The `2c` notebook and `LM-HMM_marimo.py` record both training and held-out scores.
- `block_state_space_modeling.py` records only held-out scores.

This means the project module cannot show the same overfitting pattern that the notebooks
show with training-versus-test curves.

## 4. Score normalization

- The `2c` notebook and `LM-HMM_marimo.py` divide log likelihood by the number of
  observations in the training or test split, giving a per-observation log likelihood.
- `block_state_space_modeling.py` uses raw held-out log likelihood for each fold.

Because fold sizes can differ slightly, these are not equivalent metrics.

## 5. Data source and predictor construction

- The `2c` notebook and `LM-HMM_marimo.py` cross-validate on synthetic stacked sessions.
- `block_state_space_modeling.py` cross-validates on real `block_performance` data after
  filtering valid rows.

There is also an internal inconsistency inside `block_state_space_modeling.py`:

- `mle_block_states` uses predictors `[prev_n_rewarded, n_switches]`
- `map_block_states` uses predictors `[prev_n_rewarded, bias_full_flag]`
- `run_cross_validation` also uses predictors `[prev_n_rewarded, bias_full_flag]`

So the project CV path is not even aligned with the project MLE fitting path in the same file.

## 6. Default hyperparameters

- The `2c` notebook and `LM-HMM_marimo.py` use:
  - `max_states = 4`
  - `nRunEM = 4`
  - `nKfold = min(ntrials, 4)`
  - `N_iters = 1000`
  - `TOL = 1e-4`
- `block_state_space_modeling.py` defaults to:
  - `max_states = 5`
  - `n_runs = 5`
  - `n_folds = 5`
  - `n_iter = 1000`
  - `tol = 1e-4`

So even the search grid is no longer aligned with the notebook implementation.

## 7. Parallelization structure

- The `2c` notebook and `LM-HMM_marimo.py` parallelize over state count and EM restart
  inside each fold.
- `block_state_space_modeling.py` parallelizes over EM restarts for one state count at a time.

This is a smaller difference than the fold-construction issue, but it is still a change
in implementation structure.

## Bottom line

`notebooks/2c Input-driven linear model (LM-HMM).ipynb` and `notebooks/LM-HMM_marimo.py`
are methodologically aligned.

`src/behavior_analysis/block_state_space_modeling.py` is not currently a direct project
implementation of that notebook cross-validation procedure.

The most important differences are:

- stratified trial-level folds in the notebooks versus contiguous folds in project code
- per-observation log likelihood in the notebooks versus raw held-out log likelihood in project code
- simultaneous MLE/MAP estimator comparison in the notebooks versus one estimator family per run in project code
- predictor inconsistency inside the project module itself


# ChatGPT Pro recommendations


The main shift is this: **you no longer need the validation scheme to mimic future prediction**, because future-within-session prediction is not your target. That weakens the case for forward-chaining CV. At the same time, because your comparison across sessions is only qualitative, it **strengthens** the case for fitting **each session independently** rather than forcing a shared latent state space across sessions. In the notebook, the authors stack multiple synthetic sessions with `np.vstack`, then run `StratifiedKFold` on the stacked array and fit the HMM directly on those fold slices as one sequence. That is reasonable for their toy tutorial, but it is not the best template for your scientific goal.  

So my default recommendation for you is:

1. **Fit one LM-HMM per session.**
   Treat each mouse session as its own sequence and choose the number of states within that session.

2. **Do not use their pooled cross-session StratifiedKFold as your main analysis.**
   Since you are not trying to learn a common cross-session state dictionary right now, stacking sessions mostly adds complications without helping your main question.  

3. **Use model selection as a complexity-control tool, not a forecasting test.**
   That means either:

   * per-session **AIC/BIC** on MLE fits, or
   * per-session **blocked held-out likelihood**, where the held-out data are contiguous chunks from that same session.

4. **Compare sessions post hoc through the regression weights, not state labels.**
   The notebook itself has to permute fitted states to match the ground truth, which is a reminder that HMM state labels are arbitrary. In your real data, “state 2” from one session should not be assumed to be the same thing as “state 2” from another. 

That last point matters a lot for your interpretation. Since you care about “what each model is attuned to,” the right object to compare across sessions is the **set of recovered weight vectors** (and maybe intercepts, occupancies, and dwell times), not the raw state index.

For model selection inside each session, I would use this logic:

* Start with a small candidate range, like (K = 1,\dots,6) or (1,\dots,8), depending on session length.
* Fit each (K) with **many random restarts**.
* Use **MLE fits** for AIC/BIC, because the notebook’s information-criterion section is explicitly set up that way. 
* If you also want held-out scoring, use **blocked within-session holdouts**.

The crucial tweak to the notebook is how you handle those blocks. If you hold out a middle chunk of a session, do **not** glue the left and right training pieces together into one fake continuous HMM sequence. Train on them as **two separate sequences** from the same session, and score the held-out block separately. That preserves sequence boundaries, which matters more for an HMM than for an ordinary regression. That specific change is my recommendation rather than something the notebook already does.

Because your goal is descriptive, I would also put more weight on **stability and interpretability** than on squeezing out the last bit of held-out likelihood. The notebook already includes pieces that support that style of analysis: multiple EM fits during model selection, a MAP/sticky version of the model, and duration diagnostics for inferred states. Those are all useful for your case.   

In practice, I would choose the number of states using a rule like this:

* Prefer the **smallest (K)** after which held-out likelihood or BIC stops improving much.
* Reject solutions where one or more states are tiny, redundant, or just split a single interpretable regime into near-duplicates.
* Reject solutions that produce implausibly flickery state sequences unless your behavior genuinely changes at that timescale.
* If fits are unstable across restarts, lean toward **MAP/sticky** or a smaller (K). The notebook’s MAP version exists for exactly that kind of stabilization. 

On whether (K) should be the same across sessions: for your current goal, **it is defensible to let (K) vary by session**. That best matches “describe each session well.” The downside is that comparison becomes less tidy. A good compromise is:

* first, choose (K) separately per session;
* then, if those choices cluster around a common value, refit all sessions with that shared (K) as a sensitivity check.

That gives you both: a best descriptive fit per session and a cleaner cross-session comparison.

So the condensed recommendation is:

* **Primary analysis:** independent session-wise LM-HMM fits.
* **Model selection:** per-session AIC/BIC plus, if needed, blocked within-session held-out likelihood.
* **No need for forward CV** right now.
* **No need to force shared states across sessions** right now.
* **Compare sessions by weights, occupancies, and dwell times**, not by assuming state identities line up.

That is a much better match to your stated aim than copying the tutorial’s stacked-session StratifiedKFold setup directly.


