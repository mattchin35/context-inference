# Tasks

Current development step: write and stabilize RED-phase tests for switching modes 1 and 2 before implementing the switching layer itself.

## Purpose

This file is the execution-oriented task list for the current phase of the project. It is not a full project description.
The scientific motivation and long-form requirements remain in `PRD.md`, and the broader near-term goals remain in
`PLAN.md`.

## Current priorities

1. Finish the behavior-model implementation and validation workflow needed for presentation-ready analyses.
2. Implement agent strategy switching during simulated runs.
3. Finish GLM-HMM and LM-HMM model selection with BIC and cross-validation.
4. Validate the HMM analyses on simulated switching runs.
5. Analyze three example mouse behavior sessions from the same mouse:
   - `CT014_20251205_latentInference`
   - `CT014_20251216_latentInference`
   - `CT014_20251223_latentInference`
6. Keep neural analysis work secondary until the behavior-modeling and HMM-analysis path is stable.

## Agreed implementation decisions

- Switching modes 1 and 2 will be implemented first.
- Switching mode 3 is explicitly deferred until modes 1 and 2 are available and reviewed.
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

### 2. Implement strategy switching modes 1 and 2

- Add a switching layer in `src/behavior_modeling` rather than scattering switching logic across agents.
- Record the ground-truth active strategy label on each trial.
- Preserve reproducibility through explicit seeding.
- Write tests first for:
  - switch schedules
  - mode 1 active/inactive update semantics
  - mode 2 inactive-agent decay semantics
  - reproducibility

### 3. Stop and make a separate implementation plan for switching mode 3

- Do not implement mode 3 immediately after modes 1 and 2 without re-planning.
- Revisit whether a shared-state abstraction across agents is technically clean enough to trust.
- After that sub-plan is approved:
  - write mode 3 tests first
  - then implement mode 3

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

- How should mode 3 switching be represented across heterogeneous agent types?
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
