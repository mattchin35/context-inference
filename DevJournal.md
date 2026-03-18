2026/03/18

Implemented the new behavior-modeling agents and the supporting controller workflow needed to inspect and reproduce their behavior. This included adding an `HMMRewardDecay` agent and a composite `HMMRewardDecayRelativeDoubt` agent that keeps separate belief and doubt signals while combining them into the policy value, extending the controller so runs can be plotted directly from the output dataframe, refactoring controller output handling so run execution, saving, and plotting are managed separately and more readably, and adding optional deterministic seeding so the same task, agent, and seed reproduce the same run.

- Added agent-level internal-value outputs including combined value, HMM value, and doubt value.
- Added dataframe-based plotting for actions, rewards, and model values from controller runs.
- Added documentation updates across `PRD.md`, `ImplementationDetails.md`, and `README.md` to capture the new model descriptions, reproducibility behavior, and controller plotting workflow.
