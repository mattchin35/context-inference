# Task-Variable Decoding Scientific Contract Revision 6

## 1. Status and authority

This document is the active normative amendment to
`docs/task_variable_spec_v5.md`. Revision 5 remains the complete base
scientific contract. Every revision-5 requirement remains binding unless this
amendment explicitly replaces it.

Revision 6 changes one numerical convergence control and the analysis identity.
It does not change target definitions, eligibility, neural units, time bins,
training-fold standardization, PCA, grouped folds, regularization strengths,
metrics, coefficient interpretation, or invalidity policy.

## 2. Local evidence requiring the amendment

The CT026 bounded fixed-mode benchmark used `current_action` and
`relative_doubt`, 100 ms bins, five outer folds, three regions, and both PCA and
direct-unit representations. It requested 1200 fits per target.

Under `task-variable-decoding-v1`, categorical logistic regression used
`solver="saga"`, `tol=1e-4`, and `max_iter=100`. The run completed without a
pipeline error, but 687 of the 1200 `current_action` cells were unavailable due
to convergence warnings. All 600 direct-unit cells failed, while 87 PCA cells
failed. All 1200 numerical cells were valid.

The inputs are already standardized from the outer training fold. Changing the
feature transformation, solver, tolerance, or regularization objective is
therefore not justified by this evidence.

## 3. Replaced controls

Revision 6 replaces these revision-5 values:

| Control | Revision 5 | Revision 6 |
| --- | --- | --- |
| Analysis version | `task-variable-decoding-v1` | `task-variable-decoding-v2` |
| Logistic iteration ceiling | `max_iter=100` | `max_iter=5000` |

The categorical estimator remains elastic-net logistic regression with
`C=1.0`, `l1_ratio=0.5`, `solver="saga"`, `tol=1e-4`,
`fit_intercept=True`, `class_weight=None`, `warm_start=False`, `n_jobs=None`,
and random seed 0. The 5000 value is a ceiling: a converged estimator stops
earlier. Numerical ElasticNet remains unchanged, including `max_iter=1000`.

Any convergence warning still makes that fit unavailable. Revision 6 does not
accept a partially converged solution, retry silently, change a solver, weaken
a tolerance, or expose the ceiling as a user configuration field.

## 4. Identity and migration

Revision-5 runs remain immutable historical evidence and may still be loaded
under their saved identity. Revision-6 preparation must produce a different
scientific configuration and run fingerprint. A revision-5 checkpoint or run
directory must not be resumed or reused as revision 6.

## 5. Local acceptance gates

Before cluster work:

1. repeat the same bounded CT026 fixed-mode benchmark under revision 6;
2. require all 1200 `current_action` cells and all 1200 `relative_doubt` cells
   to be valid;
3. compare previously valid revision-5 cells against revision 6 to confirm the
   expanded ceiling did not alter fits that already converged;
4. record wall time, CPU time, peak RSS, output bytes, and fit invalidity; and
5. if the bounded gate passes, run all eight categorical targets locally and
   require no convergence-warning cells before starting WP11.

Failure of either local convergence gate requires a new reviewed plan. Do not
raise the ceiling again automatically and do not move the unresolved workload
to the cluster.
