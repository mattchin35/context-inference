# Tasks

Current development step: switching modes 1, 2, and near-term mode 3 are implemented; the main remaining priority is GLM-HMM/LM-HMM model selection and validation on simulated switching runs.

## Purpose

This file is the execution-oriented task list for the current phase of the project. It is not a full project description.
The scientific motivation and long-form requirements remain in `PRD.md`, and the broader near-term goals remain in
`PLAN.md`.

## Implemented so far

- Switching modes 1, 2, and near-term mode 3 are implemented with explicit trial schedules, strict schedule validation, and an `active_strategy` label in the run dataframe.
- Switching internals now live under `src/behavior_modeling/agent_switching`, while `sample_switch_run.py` remains in the top-level `behavior_modeling` package.
- A sample switched-run demo now exists for `HMM -> F-Qlearning -> HMM -> HMM_reward_decay_relative_doubt`, with default figure saving to `reports/figures/model_behavior/switching`.
- The plotting code now supports `light` and `dark` themes, and light mode uses readable dark-gray combined-policy and action/reward markers on a white background.

## Current priorities

1. Finish the behavior-model implementation and validation workflow needed for presentation-ready analyses.
2. Finish GLM-HMM and LM-HMM model selection with BIC and cross-validation.
3. Validate the HMM analyses on simulated switching runs.
5. Analyze three example mouse behavior sessions from the same mouse:
   - `CT014_20251205_latentInference`
   - `CT014_20251216_latentInference`
   - `CT014_20251223_latentInference`
6. Keep neural analysis work secondary until the behavior-modeling and HMM-analysis path is stable.

## Agreed implementation decisions

- Switching modes 1 and 2 will be implemented first.
- Near-term mode 3 is implemented as a parallel-model approximation:
  - all selected models run in parallel on the same trial stream
  - the active model supplies the recorded value, action probabilities, and action output for that trial
  - all models update from the executed action and observed reward, even if they would not have sampled that action themselves
  - only the active model's outputs are recorded in the main run dataframe
- A true shared-state version of mode 3 is explicitly deferred until its latent-state definition is specified clearly enough to implement and test.
- Regressors currently in scope for behavior-model validation and GLM-HMM analysis:
  - `FQL`
  - `HMM`
  - `HMM_reward_decay_relative_doubt`
  - perseveration
- Model-selection defaults for now:
  - hidden states searched over `1..5`
  - `5` EM restarts
  - `5` cross-validation folds
- If BIC and cross-validation disagree on the number of states, BIC will be used for now.
- Validation of simulated switching recovery will be primarily visual in this phase:
  - correct state-count recovery
  - roughly correct switch timing
  - regressor weights consistent with the simulated agent strategy
- Presentation plots are not a first-class deliverable during implementation; plots only need to be clear enough to drive
  development and support presentation prep later.

## Ordered implementation plan

### 1. Maintain the packaging and test foundation

- Keep `uv run pytest` as the supported local test command.
- Keep pytest collection scoped to `src/tests`.
- Keep vendored `ssm` excluded from pytest collection.

### 2. Switching implementation is in place

- The switching code is now organized under `src/behavior_modeling/agent_switching`.
- Ground-truth active strategy labels are recorded on each trial.
- Reproducibility is preserved through explicit seeding.

### 3. Revisit a true shared-state mode 3 later

- Do not treat the parallel-model implementation as a true shared-state solution.
- A genuine shared latent state across strategies is still scientifically underspecified and should be planned separately later.

### 4. Finish GLM-HMM and LM-HMM model selection

- Wrap or clean up the current model-selection code in `src/behavior_analysis`.
- Do not reimplement GLM-HMM or LM-HMM outside `ssm`.
- Ensure both model families can return:
  - BIC across candidate state counts
  - cross-validation scores across candidate state counts
  - a BIC-based selected state count

### 5. Validate HMM analyses on simulated switching runs

- Use simulated runs with switching strategies to test whether GLM-HMM and LM-HMM can distinguish:
  - `FQL`
  - `HMM`
  - `HMM_reward_decay_relative_doubt`
- Save enough metadata to support later review:
  - ground-truth strategy labels
  - inferred states
  - recovered weights
  - model-selection outputs
  - run seed

### 6. Analyze the three CT014 sessions

- Build the relevant regressors for each session.
- Run LM-HMM and GLM-HMM model selection.
- Use BIC to choose the working state number for each session.
- Extract and inspect inferred states and regressor weights.
- Generate only the plots needed to justify model choices and interpret recovered strategies.

### 7. Defer neural-analysis expansion

- For now, neural work remains limited to crude 500 ms pre-choice and post-choice analyses around behavior.
- A more complete neural-analysis plan should be written later, after the behavior-analysis path is stable.

## Testing expectations

- Continue using a strict tests-first workflow for code changes.
- Prefer task-specific test modules over broad mixed-purpose files.
- Keep tests organized by active code ownership:
  - `src/tests/behavior_modeling`
  - `src/tests/behavior_analysis`
  - `src/tests/run_experiment`
  - `src/tests/visualization`
  - `src/tests/packaging`

## Open questions to revisit later

- How should a true shared latent state eventually be defined across heterogeneous agent types?
- Should simulated recovery eventually include explicit quantitative metrics rather than visual assessment only?
- Should the current BIC-over-cross-validation rule remain after presentation feedback?
- What is the concrete neural-analysis plan for the three CT014 sessions?

## Recent completed changes

### Packaging cleanup

- Added `pyproject.toml` so the repository has explicit project and pytest configuration.
- Standardized the active import path around the `src` package tree.
- Removed test and visualization `sys.path` hacks from the active path.
- Scoped pytest collection to `src/tests`.
- Excluded vendored `ssm` from pytest collection.
- Verified the setup with `uv run pytest`.

### Test reorganization

- Reorganized the previous mixed `consecutive_omission_tests` layout into clearer code-owned locations.
- Split the old controller/plotting test file into:
  - `src/tests/behavior_modeling/test_controller.py`
  - `src/tests/visualization/test_agent_run_plot.py`
- Moved agent tests to:
  - `src/tests/behavior_modeling/test_agents.py`
- Moved omission-demo tests to:
  - `src/tests/run_experiment/test_consecutive_omissions_demo.py`
- Kept packaging smoke tests in:
  - `src/tests/packaging/test_imports.py`
- Removed the stale `src/tests/consecutive_omission_tests/__init__.py` marker after the move.
- Verified the reorganized suite with `uv run pytest`.

### RED-phase switching tests

- Added new RED-phase test files for the upcoming switching implementation:
  - `src/tests/behavior_modeling/test_switch_schedule.py`
  - `src/tests/behavior_modeling/test_switch_modes.py`
  - `src/tests/behavior_modeling/test_switch_reproducibility.py`
- Defined the expected switching API around a future `src.behavior_modeling.switching` module.
- Locked in the intended schedule contract:
  - explicit trial segments
  - `start_trial` inclusive and `end_trial` exclusive
  - exact coverage of the trial range
  - no gaps
  - no overlaps
  - known strategy names only
- Locked in the intended run-output contract:
  - an `active_strategy` column must be present in the run dataframe
- Locked in the intended switching-mode semantics:
  - mode 1: inactive agents do not update
  - mode 2:
    - `F-Qlearning` uses its forgetting decay while inactive
    - `HMM` applies its transition matrix to its prior while inactive
    - `HMM_reward_decay_relative_doubt` decays belief using `HMM_reward_decay_lambda` and doubt using `relative_doubt_lambda`
- Added seeded reproducibility tests for switched runs.
- Verified the RED phase by running only the new switching tests and confirming they fail because `src.behavior_modeling.switching` does not exist yet.

### Switching implementation and sample demo

- Implemented switching modes 1, 2, and near-term mode 3.
- Implemented strict schedule validation for explicit trial segments.
- Implemented switched session execution with an `active_strategy` column in the run dataframe.
- Mode 3 currently uses parallel model updates with active-model readout rather than a true shared latent state.
- Reorganized switching internals under:
  - `src/behavior_modeling/agent_switching`
- Kept the sample switched-run entry point in:
  - `src/behavior_modeling/sample_switch_run.py`
- The sample demo uses:
  - `HMM -> F-Qlearning -> HMM -> HMM_reward_decay_relative_doubt`
- Default sample figures are saved in:
  - `reports/figures/model_behavior/switching`

### Plot theming

- Updated plotting so the user can choose a `light` or `dark` theme.
- In light mode, the combined policy value and action/reward markers now use dark gray colors so they remain visible on a white background.
- Threaded the theme parameter through:
  - visualization plotting
  - the behavior-model controller plotting wrapper
  - the sample switched-run demo
- Added tests covering:
  - light-theme marker and line colors
  - dark-theme acceptance
  - invalid-theme rejection
  - controller theme forwarding
  - sample switched-run execution in both themes
- Verified the full suite after these changes with `uv run pytest`.
