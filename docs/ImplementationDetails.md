# Implementation Details

This document holds implementation-facing details that are intentionally more specific than the product requirements in
[PRD.md](/home/matt/EinsteinMed%20Dropbox/Matthew%20Chin/LabComputerShare/context_inference/PRD.md). The PRD should stay
readable at a model and workflow level; this file captures the concrete behavior expected from the current code.

## Deterministic reproducibility

Model-agent runs should support deterministic replay through an optional user-provided seed. When a seed is supplied,
the same code version, task parameters, agent parameters, and seed should reproduce the same simulated run.

The controller owns run-level reproducibility. A single run seed is accepted at the controller entry point and saved
with run outputs so prior runs can be repeated deliberately.

## Randomness ownership

Randomness should be controlled centrally by the controller rather than by unrelated module-level globals. The current
design uses one controller-level seed to derive separate random-number-generator streams for the task and the agent.

This split is intentional:
- task randomness covers trial generation, reward delivery, reward magnitudes, cues, and context transitions
- agent randomness covers stochastic action selection from the policy

Keeping these streams separate makes runs repeatable while reducing accidental coupling between task internals and agent
internals.

## Simulation output

Detailed simulation-output schema belongs here rather than in the PRD.

Columns shared between mouse-behavior-style output and model-agent runs should include the core trial annotations:
- `state`
- `state_int`
- `cur_trial`
- `cur_trial_in_block`
- `cur_block`
- `action`
- `correct`
- `reward`
- `p_active_rew`
- `p_inactive_rew`
- `p_switch`

Model-agent runs may also include agent-specific diagnostic columns used for visualization and debugging:
- `model_stimulus`
- `agent_action_dist`
- `agent_relative_value`
- `agent_prior`
- `agent_hmm_value`
- `agent_doubt_value`

The shared columns support comparison between simulated behavior and experimental behavior. The agent-specific columns
capture internal decision variables that do not exist in mouse behavioral recordings but are useful for validating the
model logic and plotting model trajectories.

## Switching-agent simulations

Switched agent runs now support three distinct switching modes. In all three modes, the schedule is defined by explicit
trial segments, and the run dataframe includes an `active_strategy` label for each trial.

### Mode 1

The active model chooses the action and updates from the observed reward. Inactive models are frozen and do not update
while they are inactive.

### Mode 2

The active model chooses the action and updates from the observed reward. Inactive models do not use the executed trial
outcome directly, but they do receive passive strategy-specific updates:
- `F-Qlearning` applies forgetting decay
- `HMM` applies its transition matrix to its prior
- `HMM_reward_decay_relative_doubt` passively decays belief and doubt

### Mode 3

Mode 3 is currently implemented as a parallel-model approximation, not a true shared latent-state implementation. The
active model chooses the executed action, the task returns one observed reward, and then all models update from that
same executed action and reward, even if a model would not have sampled that action itself.

Only the active model's outputs are recorded in the main run dataframe on each trial, and those recorded outputs are
the active model's pre-update values for that trial.
