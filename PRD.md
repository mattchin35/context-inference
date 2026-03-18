# Project Requirements Document: Mouse Behavior Analysis and Simulation

## Overview
Mouse behavior and electrophysiology from a 2-choice bandit task with asymmetric rewards will be analyzed here. 
Behavior simulations will also be performed of different strategies to be compared with mouse behavior and to test
the validity of existing analysis methods and models.

## Mouse behavior analysis
The current goal is to compare mouse behavior to reinforcement learning, hidden markov models, and a constructed
model of behavior that combines HMM responses to rewards, decaying value updates with no rewards, an estimate of 
doubt. Models should ideally be combined with a "stickiness" to previous choices that can be represented
by a "perseveration regressor."

## Behavior simulation
There will be a test suite of sample agents that use the strategies used to analyze the mouse behavior. Their behaviors
will be formatted in the same way as the mouse behavior data. They will be used to test the validity of GLM-HMM and 
LM-HMM analysis methods. Behavior strategies will be able to be swapped in and out during a simulation, ideally using
different kinds of swapping, to test whether the changes in strategy can be detected. 

Strategy swapping will be done in 3 ways:
1. One agent will run at a time, updating its state as it goes. The other agents will not update their states when they are not active.
This will give each agent an independent internal state.
2. The active agent will update its state as it goes, and the inactive agents will have a slight decay according to their 
behavior strategy (Qlearning should have a decay, pure HMM should have a hazard/transition matrix update, HMMs with decay should
follow its decay rules)
3. All agents will have a shared state, and on each trial the state will update based on the active strategy.

## Integration of behavior and neural data
Neural data must be synchronized to behavioral data using by decoding IRIG timecodes to UTC timestamps and 
interpolating between the IRIG pulses. Behavior and neural data will be gathered into pynapple dataframes
for analysis and compatibility with outside tools.


