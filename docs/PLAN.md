# Current status

Code was used to prepare preliminary behavior analyses with LM-HMM and GLM-HMM models. 
Now I need to refactor it, analyze multiple sessions with the HMM analyses, and begin integrating HPC 
and PFC analyses together. The goal is essentially to have all the analysis code I'd need for new sessions 
ready to go, and to prepare the core of my analysis methods for a simple publication.
I want to do the fancier analyses, but this content will be the core of my learning as a neurophysiologist.

# Current goals
- Code needs to be refactored and read over to make sure that I understand and trust it (much of it was Codex generated).
- Single-session behavior runs should go through the full pipeline.
- Multisession analyses should combine a set of blocks or trials together to analyze the presence of behavior modes
across sessions.
- HPC analyses should be done by training phase (late training, mid-training, early training) to look for consistent patterns.
- PFC and V1 analyses should repeat HPC analyses.
- HPC and PFC single units can be analyzed for representation of context and choice variables.
- HPC units can be used to predict PFC units.
- HPC spike-phase-locking to HPC theta
- PFC spike-phase-locking to HPC theta
- PFC spike-phase-locking to PFC theta

Anything for single-unit analyses can work with "good" units as a first pass, but the code should be 
designed to be easily applied to any unit desired.

## Single-session behavior analysis
Debug code to do a full analysis of a single session. Refactor code used for readability, to remove dead code, 
and to make sure Codex-generated code is trustworthy. I'll probably want to double-check on any 'None' vs np.nan 
usage, in code and in saved csvs.

## Multi-session behavior analysis
Given a set of sessions, I should be able to plot the learning curve, combine block data for LM-HMM analysis, 
and combine trial data for GLM-HMM analysis. In the future this can be expanded to infinite HMM use. After creation,
refactor/redesign/read code for readability and trustworthiness.

## Regional basic analyses
1. Replicate HPC analyses for V1 units, compare to HPC.
2. Replicate HPC analyses for PFC units, compare to HPC.

Read and refactor code for readability and trustworthiness, and also to learn to use pynapple.

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

