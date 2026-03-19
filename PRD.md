# Project Requirements Document: Mouse Behavior Analysis and Simu1lation
A slightly detailed description of the project goals, data, and analysis methods. This document is intended to be more 
detailed than the overall description, but less implementation-specific than the Implementation Details document.

## Overview
Mouse behavior and electrophysiology from a 2-choice bandit task with asymmetric rewards will be analyzed here. 
Behavior simulations will also be performed of different strategies to be compared with mouse behavior and to test
the validity of existing analysis methods and models.

## Task behavior description
Mice perform a 2-choice contextual-decision making task with asymmetric rewards. On each trial, the mouse chooses 
between two options to receive a water reward. The task exists within 2 contexts, a left and a right context, in which 
only the context-corresponding choice (left choice in left context, right choice in right context) has a nonzero reward 
probability on each trial. Because rewards for correct choices are probabilistic, correct choices can be unrewarded. 
Correct choices can trigger a switch to the opposite context with a certain probability, but incorrect choices cannot 
trigger a switch. The agent must use the history of choices and rewards to infer the current context and make informed 
decisions. An optimal agent would understand that rewards fully confirm the context on the previous choice, 
singular unrewarded choices may imply context switch or a missed reward, and multiple unrewarded choices strongly imply 
a context switch. 

That is, this task is based on asymmetric evidence:
- reward → definitive evidence
- omission → ambiguous evidence

The optimal behavior then would be to choose the same option after a rewarded choice, and to switch 
options after multiple unrewarded choices, with some uncertainty after a single unrewarded choice.

Reward structure:
- Correct side reward probability: p_reward = .9, .8, .7, or .6 (varies by session)
- Incorrect side reward probability: 0

Context transitions:
- After a correct choice, the context switches with probability p_switch. The transition is independent of reward 
delivery, so the context can switch after a rewarded choice or an unrewarded choice.
- p_switch = .1, .2, or .3 (varies by session)
- Otherwise, context remains the same

Trial structure:
1. Agent chooses left or right
2. Reward is delivered probabilistically if the choice matches the current context
3. If the choice is correct, regardless of reward outcome:
   - Context switches with probability p_switch
4. Otherwise:
   - Context remains the same

In the RL literature, this could be referred to as a "contextual bandit" task or a "partially observable Markov decision process (POMDP)."

This asymmetric reward task is adapted from Vertechi, Neuron 2020.

## Data description 
Mice are recorded using Neuropixels 2.0 probes in the PFC and HPC. Spikes are sorted using Kilosort 2.5.2 or 4.1.3.
We will focus on spiking data for now as a core requirement, potentially adding LFPs later. Neural data are recorded
alongside an IRIG-H timecode that specifies the UTC timestamp of each neural data sample, and must be synchronized to 
behavioral data using these timecodes. Behavior is recorded on a Raspberry Pi that is already synchronized to GPS 
UTC time.

Other data (treadmill speed, camera, pupil tracking) are still being debugged in physical hardware. 

## Mouse behavior analysis
The current goal is to compare mouse behavior to reinforcement learning, hidden markov models, and a constructed
model of behavior that combines HMM responses to rewards, decaying value updates with no rewards, an estimate of 
doubt. Models should ideally be combined with a "stickiness" to previous choices that can be represented
by a "perseveration regressor."

### Regressor specifications 

GLM-HMM: For the analysis of trial-by-trial behavior. The GLM-HMM models the predicted next choice based on regressors 
from the trial history. For more details, see Ashwood et al, Neuron 2022 
and all the papers that followed it from the IBL group.

- Relative value, by Q-learning or Forgetting Q-learning
- Reward-omission based doubt: a relative doubt signal derived from the recent history of unrewarded choices, expressed as a left-vs-right quantity that grows with repeated omissions and opposes confident commitment to one side
- Signed belief by bayesian log-odds
- Signed belief by bayesian log-odds with a decay: a belief signal that updates by Bayesian inference on rewarded trials, but on omission trials decays passively toward uncertainty rather than treating each omission as full contrary evidence
- Combined belief-doubt model: a constructed regressor that updates a Bayesian belief signal on rewards, applies omission-driven decay to belief on unrewarded trials, computes a separate doubt signal from omission history, and combines them into a single relative decision variable
- Choice history-based / Perseveration

GLM-HMM input:
- At each trial t, regressors from trial history are used to predict choice at t+1
- Regressors are standardized to [-1,1]
- Model outputs:
  - state sequence
  - state-specific GLM weights

LM-HMM: For the analysis of block-by-block behavior. The LM-HMM models the behavior at block transitions (i.e. when the 
context changes). The LM-HMM looks at how reward history relates to behavior at block transitions to group successive 
blocks by strategy. An inference strategy will have minimal correlation between reward history and behavior at
block transitions, while a reinforcement learning strategy will have a strong correlation between reward history and 
behavior at block transitions. For more details, see Cazettes et al, Nature Neuroscience 2023.

- rewards received prior to a context change 
- number of trials taken at a block change to switch choices (trials to switch). If the mouse never switches on the
last block, the last block will be excluded from analysis.

LM-HMM operates on block-level summaries:
- Each block is treated as a datapoint
- Features:
  - total rewards in block
  - trials-to-switch at block transition
  - potentially a "bias" flag if the block has particularly high number of trials to switch
- Output:
  - block-level state assignments
  - state-dependent regression weights

For GLM-HMM and LM-HMM models, both cases, the number of states will be determined by model comparison with BIC and 
cross-validation. Models will be run with 1-5 states to pool BIC and cross-validation outcomes. 
When this is done well, I will be able to separate trials (GLM-HMM) or blocks (LM-HMM) into different groupings that 
correspond to different strategies, and plotting each grouping on a traditional logistic regression or linear regression
will show that the groupings make sense. For now those plots will be done in separate R code - I will have to 
port that code into this project eventually.

GLM-HMM and LM-HMM will only use the ssm package, and SHOULD NOT BE REIMPLEMENTED. Computations must remain 
compatibility with existing studies.

Future analysis will assess task engagement and arousal, including time to choices and pupil size.

TODO: determine how to choose parameters for each model. Options are grid search, BIC/AIC, or cross-validation.

## Behavior simulation
There will be a test suite of sample agents that use the strategies used to analyze the mouse behavior. Their behaviors
will be formatted in the same way as the mouse behavior data. They will be used to test the validity of GLM-HMM and 
LM-HMM analysis methods. Behavior strategies will be able to be swapped in and out during a simulation, ideally using
different kinds of swapping, to test whether the changes in strategy can be detected. 

The key strategies will be:
- Forgetting Q-learning
- Bayesian belief with no decay
- Bayesian belief on rewards with decay on omissions
- Combined belief-doubt model: Bayesian belief on rewards with decay on omissions, doubt calculated from omissions,
combined into a relative value (this is the constructed model that I think will best model learned mouse behavior)
- Any of the above with a perseveration bias

Parameters for each strategy will be chosen to require information to be collected across multiple trials (i.e. an RL, 
HMM-decay, or HMM-decay-doubt agent should require multiple unrewarded trials to switch).

### Model reproducibility

Behavior simulations should support repeatable runs with an optional user-chosen seed so that the same model, task
parameters, and seed can reproduce the same simulated trajectory. Agent choices will be made probabilistically 
(e.g. using an expit/logistic function). 


### Strategy swapping 

Strategy swapping will be done in 3 ways:
1. One agent will run at a time, updating its state as it goes. The other agents will not update their states when they are not active.
This will give each agent an independent internal state.
2. The active agent will update its state as it goes, and the inactive agents will have a slight decay according to their 
behavior strategy (Qlearning should have a decay, pure HMM should have a hazard/transition matrix update, HMMs with decay should
follow its decay rules)
3. All agents will run in parallel on the same trial stream, updating their states on each trial regardless of whether 
they are active. The active agent will determine the recorded action and reward for that trial, but all agents will 
update from the executed action and observed reward.
4. All agents will have a shared state, and on each trial the state will update based on the active strategy.

For simplicity, we will limit the number of agents in one session to 3. The GLM-HMM/LM-HMM may infer different numbers
of states from the true number of states, potentially varying how many are found by the strategy that is used to swap 
the states. This will allow us to test the sensitivity of the GLM-HMM/LM-HMM to detect strategy changes and 
to determine how many states are present.

The GLM-HMM and LM-HMM will be used to determine the number of agents present in a session, when each agent is active, 
and what strategy the agent is using. For now agent switches will be handled simply by using agents in blocks of 
50 trials, but in the future we can implement Markov switching (ed note: this will be mathematically more complex even 
if it is conceptually simpler, potentially involving hierarchical HMMs and other things I don't understand yet. I will 
not implement this until I understand it.)

### Model Evaluation

Simulation outputs will be evaluated using:

1. Behavioral recovery:
   - Accuracy of inferred HMM states vs ground truth, both by identity and probability of the most likely state
   - Alignment of inferred vs true switch points
   - Comparison of GLM-HMM trial groupings to the known strategy groupings of each trial
   - Comparison of GLM-HMM regressor weights to the known regressor weights of each strategy
   - Comparison of LM-HMM block groupings to the known strategy groupings of each block
   - Comparison of LM-HMM regressor weights to the known regressor weights of each strategy 
   
2. Failure modes:
   - State merging
   - State splitting
   - Misaligned transitions

## Integration of behavior and neural data
Neural data must be synchronized to behavioral data using by decoding IRIG timecodes to UTC timestamps and 
interpolating between the IRIG pulses. Behavior and neural data will be gathered into pynapple dataframes
for analysis and compatibility with outside tools. Spikes will be used to generally analyze: 
- The mouse's inferred context and choice
- How mice integrate rewards and unrewarded choices into context inference and decision-making
- Communication between PFC and HPC, and computations unique to each region

Spikes will be binned for analysis. Bin size is still being determined, but current options are 
- 500 ms, for crude analyses of before choice/after choice activity. Aligned to trial choices.
- 100 ms, 50 ms, and 20 ms for finer analyses within a trial. Aligned to trial choices.
- 500 ms or 100 ms for analyses extending across and between trials, including the inter-trial interval. This can only 
precisely align to one trial due to the way the task is set up.

Analyses will primarily be population level, focusing on neural geometry and dynamics. However, we should expect some 
pushback on this, potentially requiring analyses of mixed selectivity at single neuron level. Some (non-binding) example
analyses:
- PCA of activity; LDA across contexts and choices; Elie/Sjulson gcPCA across contexts and choices
- GLM or GLIM of spiking activity in one region to spikes in another region (a large multi-input, multi-output GLM/GLIM)
- GLM or GLIM of spiking activity to behavior (context, choice, reward, etc.)
- Low-d regression with communication subspaces, CCA
- Latent variable decomposition using SCA (sparse decomposition analysis) or RNN-based methods (LFADS)

## Code frameworks
We will stick to existing Python libraries and established neuroscience libraries as much as possible.
Core numpy - numpy, scipy, sklearn, pandas
Plotting - matplotlib, seaborn
Stats - statsmodels
Neuroscience - pynapple, ssm

## Future _possible_ project directions
- Use of an HMM with a continuous latent state space or infinite number of states. Included here for completeness, 
but a better choice is RNN analysis with Marcelo Mattar's group.
- RNN analysis of mouse behavior, in collaboration with Marcelo Mattar. This should NOT be implemented here, as it 
requires their lab's expertise and codebase.
- RNN analysis of neural spiking, in collaboration with Kanaka Rajan. This would require their expertise and a strong understanding
of our biological data first.
- LFP analyses for spike-phase coupling or HPC-PFC phase synchrony. Not critical for us but other labs will probably want it.
- RNN simulation of neural activity and behavior from scratch. Doing this well requires a reasonable understanding of 
the biological behavior and neural activity first.

## Key references
TODO: grab papers for similar mouse tasks, Marcelo Mattar/Kanaka Rajan/Ila Fiete RNN papers, Sjulson/Elie gcPCA paper, 
GLM-HMM and LM-HMM papers, Sjulson Neuron CPP paper 
