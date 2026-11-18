# SAC TTS implementation log

## Approved scope and execution

- User approved the plan in `docs/SAC_behavior_prep.md` and requested implementation.
- One implementation worker maximum; W1-W5 run sequentially with orchestrator review.
- Each job writes tests, demonstrates RED, and waits for the orchestrator to commit
  tests before implementing. Preserve unrelated worktree changes.
- Validation: `CT024_20260609_latent_inference`. Do not run the full experimental
  cohort until the user approves this session's outputs and session selection.
- Approved proposed defaults: first block excluded, missing TTS omitted/reported,
  five valid blocks, elastic net alpha 0.5, lambda.min with lambda.1se sensitivity,
  RDS/CSV outputs and light plots. Bayes-factor and cohort-date choices remain open;
  implement AIC/BIC and explicit session selection without inventing those choices.

## Environment preflight

- R 4.6.1 executable and designated validation block CSV are available.
- Existing packages load with
  `R_LIBS_USER=/home/matt/R/x86_64-pc-linux-gnu-library/4.6:/home/matt/R/x86_64-pc-linux-gnu-library/4.5`.
- glmnet, lme4 and ggplot2 load and numerical glmnet/lmer smoke checks succeed.
  Flexplot does not load in this environment; core computation/exports do not
  require it. No dependency modifications or installations performed.
- Existing real CSV has time-resolved `session_id` and legacy date-only
  `session_ID`; these must coexist without false identity conflicts.
- A plotting smoke check accidentally overwrote pre-existing untracked root
  `Rplots.pdf`. The user was informed. Git has no original to restore. All
  subsequent graphics must open explicitly named output devices.

## Job progress

- W1: tests reviewed and independently confirmed RED (missing preparation
  function). Tests committed as `30a7698`. Worker resumed for implementation.
- W1 accepted: new data tests and existing block-preprocessing tests pass.
  Preparation of the designated CSV retains 60/62 blocks, excluding the initial
  block and one missing-TTS block. No experimental fit/output written.
- W2 dispatched for the test phase; W1 worker finished before W2 started.
- W2 RED independently confirmed for missing `rewards_only` formula and session
  fitting entry point. Tests committed as `70f43ed`; worker resumed for GREEN.
- W2 accepted after independent review and GREEN session/legacy tests. The
  side-free adapter, original-scale coefficients, shared folds, residual metrics
  and explicit fit statuses are implemented. Legacy formula/output defaults
  remain intact. No tests changed after commit.
- W3 dispatched for population/exemplar test phase after W2 worker finished.
- W3 RED independently confirmed after passing W2 tests; tests committed as
  `e4a4718`. Worker resumed for GREEN. Population tests compare formulas,
  fitted row IDs, ML criteria and predictions directly with lme4.
- W3 accepted after independent review and GREEN combined scientific tests.
  Exact M1-M4 fits, visible convergence/singularity, population/conditional
  predictions and explicit fixed-exemplar errors are implemented.
- W4 test phase assigned to the finished W1 worker after W3 finished. Maximum
  one implementation worker remains active. Tests/artifacts must use ignored
  project `data/` directories; experimental source files remain read-only.
- W4 RED independently confirmed for missing persistence API. Tests committed
  as `8eed08f`; worker resumed for storage/runners implementation.
- W4 initial integration suite independently GREEN. Direct real-data dry-run
  identified HMM-derived block CSVs matching the primary discovery suffix;
  worker is adding a regression test before narrowing discovery. No experimental
  fit or artifact writes have occurred.
- Discovery regression independently RED and committed as `bfc80be` before its
  fix. W4 accepted: integration tests and direct single/date-filtered batch
  dry-runs pass. Discovery excludes only the two known derivative suffixes and
  still rejects multiple primary tables. No experimental fits/writes performed.
- W5 assigned to the finished W2 worker after W4 finished. Figure tests must
  demonstrate loading/rendering without fitting and explicit missing-population
  skips for the designated single-session validation.
- W5 RED independently confirmed and tests committed as `bceb5ed`; worker
  resumed for rendering implementation.
- Broader R checks: block preprocessing/residuals, entry/source paths, visualization
  sample sizing, and new data/model/results tests pass. Two existing trial-level
  suites (`test_leave_stay_preprocessing.R`, `test_trial_compatibility.R`) cannot
  load the old dplyr binary under R 4.6 (`R_shallow_duplicate_attr` undefined).
  New TTS code does not use dplyr. No installed package was modified.
- Validation source CSV checksum before experimental execution:
  `d1c7ce6434c99ab2fef06f5043d759bb`.
- W5 accepted after independent GREEN plotting tests and visual QA. The synthetic
  preview contains 25 panels across all F1-F9; revised F2/F5/F9 views have readable
  formula text, independent parameter scales and matched comparison ranges.
- Final new TTS/data/residual/model/result/plotting suites pass. Existing
  block-preprocessing/source-path/sample-size checks also pass. The two unrelated
  trial-level suites remain blocked by the documented dplyr ABI issue.

## Experimental validation

- Completed the designated single session only. Results:
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT024/CT024_20260609_latent_inference/processed/sac_tts/sac_tts_2026-11-08_174639_CT024_2026-06-09_143852`.
- Figures are in its `sac_tts_2026-11-08_174649_figures` subdirectory: 22 PNG
  panels, source CSVs/captions, rendering metadata and a manifest. F2/F3 are
  explicitly skipped because a single-session artifact has no population fit.
- Both formulas at both penalties select intercept 1.616667 trials and reward
  slope zero, with sample residual SD 0.865373 trials. This is a descriptive
  result for the designated session, not a cohort or cognitive-strategy conclusion.
- Source CSV checksum after computation/rendering matches the pre-run checksum.
  Actual F4/F8/F9 exports inspected visually; no full experimental cohort run.
- A new `VALIDATION.md` beside the numerical artifacts describes findings and
  inspection targets. Usage commands are documented in `src/Rcode/README.md`.
- User confirmed all processed latent-inference sessions for the specified nine
  mice. Read-only discovery finds 216 sessions, 2026-04-16 through 2026-06-11:
  CT016 32, CT017 35, CT019 19, CT020 23, CT021 21, CT022 22, CT023 21,
  CT024 21, CT025 22.
- User subsequently inspected the validation figures and approved continuing.

## Full experimental cohort

- Full batch launched for all 216 discovered sessions across the nine mice,
  using the approved seed/parameters and 24 session-level compute processes.
  Inner BLAS/OpenMP thread counts are set to one to avoid nested parallelism.
- The one-active-implementation-worker limit is preserved; this run uses
  computational session processes, not additional implementation agents.
- Initial batch saved all 216 session fits, then collection failed because
  original CSVs have heterogeneous optional columns (67/87/91 prepared columns).
  Required scientific tables share their schemas. A sequential corrective worker
  is adding a regression test before a localized union-column collection fix;
  scientific fitting and source CSVs are unchanged.
- Regression independently RED and committed as `73e2adb` before the fix.
  Collection now aligns the union of columns with typed missing values, preserving
  scientific columns and original row IDs. Full results/runner tests independently
  GREEN; corrected batch launched. The source-code checksum change creates fresh
  versioned session artifacts under normal cache rules; previous fits remain intact.
- Completed corrected cohort run:
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/cross_mouse_analysis/sac_tts/sac_tts_2026-11-08_195003_cohort`.
- All 216 sessions / nine mice completed both session fits. Retained 13,552 of
  13,922 blocks: excluded 216 initial blocks and 154 missing-TTS blocks.
  Checksums of every input CSV match the pre-computation cache metadata.
- All four population models converged. M2 is nonsingular and has the lowest BIC
  (74502.27); M4 has the lowest AIC (74369.62) but is singular at the mouse-level
  covariance. Retain that disagreement; no automatic replacement model or HMM
  parameter update. Mouse-level PCA has a near-zero covariance component;
  session-level covariance is full rank.
- Side-free lambda.min median parameters: intercept 2.103 trials, reward slope
  0.123 trials/rewarded trial, sample residual SD 3.026 trials. Many slopes shrink
  strongly under lambda.1se; pooled residuals show a long positive tail.
- Rendered all 25 panels successfully in `sac_tts_2026-11-08_195102_figures`.
  Inspected the model comparison, population relationship, distributions,
  per-mouse parameters, 3D coordinates, side asymmetry, exemplar errors and
  residual/penalty diagnostics. All PNGs, source CSVs and captions exist.
- Added `COHORT_FINDINGS.md` beside results with numerical findings, caveats and
  clickable links to every panel. Existing source and prior output files preserved.
