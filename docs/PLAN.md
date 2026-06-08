# Current status

Code was used to prepare preliminary behavior analyses with LM-HMM and GLM-HMM models. 
Now I need to refactor it, analyze multiple sessions with the HMM analyses, and begin integrating HPC 
and PFC analyses together. The goal is essentially to have all the analysis code I'd need for new sessions 
ready to go, and to prepare the core of my analysis methods for a simple publication.
I want to do the fancier analyses, but this content will be the core of my learning as a neurophysiologist.

# Current goals
- I need to read and refactor decoding code for refactoring and learning pynapple (much of it was Codex generated).
- HPC and PFC single units can be analyzed for representation of context and choice variables.
  - This will start with simple heatmaps of firing rates or overall firing sums. Sort by correct-rewarded trial responses, then plot same sorting for other conditions.
  - Start with overall firing sums in the +/- .5s interval, then move to 100ms intervals, then move to finer intervals if it seems useful (50 or 20 ms)
- HPC units can be used to predict PFC units.
- HPC spike-phase-locking to HPC theta
- PFC spike-phase-locking to HPC theta
- PFC spike-phase-locking to PFC theta

Anything for single-unit analyses can work with "good" units as a first pass, but the code should be 
designed to be easily applied to any unit desired.


## General behavior assessment
I need to improve the overall assessment of mouse behavior, beyond the HMM strategies and reward history-trials to switch correlation.
1. Oracle choices - the correct choice on each trial should be marked and the mouse's % correct should be plotted over subsequent sessions. 
I think some code already exists for session % correct that can be expanded on. 
2. Oracle reward-collection. Using the correct choice on each trial, calculate the expected reward for a session using the active reward probability.
Compare the mouse's actual reward collection to the expected reward collection, and plot that over subsequent sessions.
3. Ideal observer choices - using the reward history and a task-appropriate strategy, calculate the ideal observer's choice on each trial.
Compare the mouse's actual choices to the ideal observer's choices, and plot that over subsequent sessions.
4. Ideal observer reward-collection (i.e. "regret") - using the ideal observer's choices, calculate the expected reward for a session using the active reward probability.
Compare the mouse's actual reward collection to the ideal observer's expected reward collection, and plot that over subsequent sessions.

For the ideal observer, I will use greedy version of the HMM-doubt model, which biases to reward rather than accumulating evidence, and 
accumulates doubt/evidence to switch as a heuristic to change sides (it also allows me to change the observer's trials-to-switch).


## Single-session behavior analysis
Sufficently done for now

## Multi-session behavior analysis
Sufficiently done for now, but a codebase-wide refactor remains to be done for some naming conventions. 
In the future this can be expanded to infinite HMM use. 

## Regional basic analyses
1. Replicate HPC analyses for V1 units, compare to HPC.
2. Replicate HPC analyses for PFC units, compare to HPC.

These have been done, but I haven't read/refactored or learned pynapple from it yet.

## Raw data plots
1. PSTH of licks and spikes. Licks should have L/R separated, trial conditions should be separated, but if I could have all the conditions and licks on the same plot that would be excellent.
2. Heatmaps of unit firing rates separated by trial condition, sorted by correct-rewarded trial responses. Look for 
populations that respond to context, choice, or reward variables.
3. Plot the same heatmaps for finer time intervals (100ms, 50ms, etc) to look for more specific temporal dynamics of those responses.
4. Examine HPC/V1/PFC differences in those responses.

## Single-unit prediction analyses
1. Use HPC units to predict PFC units. Start by using simple GLMs and poisson GLIMs.
2. As a subgoal, you can try sparser regression methods (Lasso, Ridge, ElasticNet, "communication subspace" methods).
Hopefully that can be dropped in easily after the base GLM analyses.

## Theta power and synchrony analyses
1. Compute theta power in HPC and PFC over time; look for more theta power in good performance regimes
2. Compute theta phase synchrony between HPC and PFC; look for more synchrony in good performance regimes

Hopefully Codex makes some of this fast to get started, but a lot of the work here will be learning phase analysis
methods and using pynapple.

## Spike-phase-locking analyses
1. Compute spike-phase-locking of HPC units to HPC theta; look for more locking in good performance regimes
2. Compute spike-phase-locking of PFC units to HPC theta; look for more locking in good performance regimes
3. Compute spike-phase-locking of PFC units to PFC theta; I have no idea if this is supposed to be a thing, but the code
from 1 and 2 should be easily adaptable to this.

# Future
Dimensionality (PCA, LDA, CCA) analysis, multi-selectivity and geometry of representations, and latent variable 
decomposition. Use of Kanaka Rajan's techniques, Marcelo Mattar's techniques, RNNs for behavior modeling and brain modeling. 
Really avoid doing any of that as a first pass though, get the essential analyses done first. 

