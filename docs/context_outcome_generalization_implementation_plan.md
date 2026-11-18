# Context Outcome Generalization: Implementation Plan

Status: C1-C7 implementation and synthetic acceptance complete, 2026-11-10.
No experimental session was processed in this implementation run.

This extends the completed [cross-condition implementation](cross_condition_generalization_implementation_plan.md).
The [scientific specification](cross_condition_generalization_spec_v1.md),
especially Section 3.1, owns the scientific definitions. Read repository
`AGENTS.md` and [SoftwareDesign.md](SoftwareDesign.md) before implementation.

## 0. Handoff and execution rules

| Item | Current state |
| --- | --- |
| Inspected branch / HEAD | `refactor` / `0d181ed` before the final documentation-only handoff commit; recheck on resumption. |
| Authorization | User explicitly approved implementation on 2026-11-10. C1-C7 authorized; no experimental computation. |
| Baseline | v1 implementation complete; full regression 1433 passed. Historical evidence remains in the original plan. |
| User edits | Preserve `src/main.py`, `src/neural_analysis/sync_ephys.py` and unrelated untracked files. The user explicitly authorized preserving ongoing behavior-mouse changes in main.py. |
| Running work | None. Both sequential workers are idle; the temporary synthetic server/tab are closed. |
| Next action | User inspection of the new Context outcome view; experimental computation remains a separate step. |

Use a Sol orchestrator (`gpt-6.1-sol`, high), sequential Sol scientific workers
(`gpt-6.1-sol`, high), and a Luna documentation worker (`gpt-6-luna`, medium).
At most one worker runs alongside the orchestrator; no nested agents. These
limits concern coding agents, not the existing analysis session-worker policy.

For each code package: worker writes tests and demonstrates RED, then idles;
orchestrator reviews and commits only tests; worker implements GREEN and
returns; orchestrator reviews, verifies the gate, commits source and records
evidence here. Use one writer per file and path-limited commits. Do not change
tests merely to accommodate incorrect implementation. Existing tests may be
updated where the approved registry, report fields or version contract changes.

| Package | Owner | Initial state | RED / test commit | GREEN / source commit |
| --- | --- | --- | --- | --- |
| C0 scientific amendment / plan | Sol orchestrator | Approved | Documentation only | Approved spec revision 10 and this plan |
| C1 registry / preparation | Sol worker | Complete | 6 expected failures / 143 passed; `f9e332a` | 149 passed; `8e586c4` |
| C2 fitting / summaries / counts | Sol worker | Complete | 9 expected failures / 89 passed; `ff68b6f` | 98 passed; `4b241c3` |
| C3 saved products / runners | Sol worker | Complete | Expected S=16 persistence blockers; `6b55d99` | 307 passed; `1717f84` |
| C4 summary webapp | Sol worker | Complete | 17 expected failures / 2 passed; `4f7f724` | 75 webapp tests passed; `aa9d45a` |
| C5 synthetic / browser acceptance | Sol orchestrator | Complete | Prior package tests | 500/100-ms fits, probability reconstructions, source counts and browser checks passed |
| C6 usage documentation | Luna worker / orchestrator review | Complete | Documentation only | Three guides reviewed; ASCII/fences/local links passed; `0d181ed` |
| C7 final handoff | Sol orchestrator | Complete | Prior package tests | 1465 regression tests passed across three disjoint groups; Ruff passed; evidence below |

Update this snapshot at RED, test-commit, GREEN/review and interruption
boundaries. Include active owner, live command/log IDs, completed commits,
remaining findings and exact next action. Preserve partial work on interruption.

## 1. Confirmed scientific contract

All rows below predict recorded context: class 0 is Right context, class 1 is
Left context. Each training pool contains both contexts. Left/right never
becomes an action label on incorrect trials.

| Training outcome | Test outcome 1 | Test outcome 2 |
| --- | --- | --- |
| Correct-rewarded | Correct-omission | Incorrect |
| Correct-omission | Correct-rewarded | Incorrect |
| Incorrect | Correct-rewarded | Correct-omission |

Use existing canonical outcomes: correct-rewarded is correct=1/reward=1;
omission is correct=1/reward=0; incorrect uses the canonical incorrect mask,
including otherwise eligible incorrect trials with reward 1 or missing reward.
Correct trials with unknown reward belong to neither correct-rewarded nor
omission. Preserve existing exclusions, numeric normalization and original IDs.

Fit one readout per training outcome/window/weighting/region. Its two test
outcomes use the same coefficients, intercept and source weights. They are
separate evaluations, never separate fits or an average across outcomes.
Source and test trials remain disjoint. Use all eligible source trials, with
no folds, subsampling, training fraction or outcome-specific tuning.

The existing `context_incorrect_to_correct` fit already trains on the required
incorrect pool and evaluates the union of correct-rewarded and omission trials.
Reference that one model in both Context views. Add only the other two fits.
This is reuse within the expanded bundle, not importing coefficients from an
older saved-result version.

Retain the original four Context condition transfers and all Choice, Actual
reward and Value definitions. Retain PFC/HPC independence, all unwhitened PCs,
fixed estimator settings, both weights and Full/Before/After matched windows.
PCA source windows remain [-2,+2) seconds. Blocks stay ignored. Full supervised
windows do not overlap; introduce no overlap handling.

## 2. Visible training counts

For every regional model, count unique source trials after baseline, dark,
conflict, target-label and regional-coverage eligibility:

- `training_trials`: total eligible source trials.
- `training_class_0_trials`: Right-context source trials for Context.
- `training_class_1_trials`: Left-context source trials for Context.

Reuse `eligible_cell_counts[split, 0, :]` and sum its two classes. Counts are
unweighted integers, not pooled bins, weight sums or effective sample sizes.
They remain identical across a model's two test-outcome rows and both weighting
schemes. Do not recalculate them after filtering the displayed test outcome.

Show all three counts directly in the new results table, without opening Fit
details or selecting the Training role. Also show test eligibility, scored
test counts, test class counts and availability reasons. Use a caption to state
that Left/Right counts refer to recorded context.

For a failed fit with known regional coverage, retain eligible training counts
and the failure reason while scored trials remain zero. For an unavailable
region, training counts are unknown (`None`), not fabricated zeros. For an
empty known source pool, counts are zero. Existing missing-source-class handling
applies. Add no arbitrary small-sample threshold, automatic exclusion or warning
rule; visible total and per-context counts let the user judge small groups.
Counts expose sample-size differences but do not remove that confound. Keep
fixed C and total training weight Ntrain as specified; matching group sizes or
changing regularization across outcomes would be a separate analysis decision.

## 3. Architecture and data contracts

### 3.1 Fixed registry and explicit cells

In `cross_condition/config.py`, preserve the original sixteen entries and
their order. Append:

| New split ID | Source class 0 / 1 | Test class 0 / 1 |
| --- | --- | --- |
| `context_correct_rewarded_to_other_outcomes` | Right / Left correct-rewarded | Right / Left (omission OR incorrect) |
| `context_omission_to_other_outcomes` | Right / Left omission | Right / Left (correct-rewarded OR incorrect) |

Define a small fixed `CONTEXT_OUTCOME_MODELS` mapping from the three model IDs
to their training population and two test populations. Its third entry references
`context_incorrect_to_correct`. Define original Context condition-view IDs
explicitly rather than assuming every Context model belongs to that view.
Use simple immutable records/tuples; no plugin registry or expression parser.

In `preparation.py:_named_cells`, reuse canonical masks and add the required
context/outcome masks and explicit unions. Keep `TransferDefinition`'s two
source and two test cells, ordered by actual target class. Composite test
cells contain a union of outcomes within one context; their outcome-specific
summaries remain separate. Update the docstring's old elementary-four-cell
description accordingly. Keep public preparation/modeling signatures.

### 3.2 Axes and numerical behavior

J=original chronological rows; N=ordinary shared choices; T=4 targets;
S=18 unique readout definitions; H=3 windows; G=2 weightings. Each region has
its own W choices, K original PCs and integer W-to-N map. No new outcome axis
is needed in saved model arrays.

| Array | Planned shape / units |
| --- | --- |
| Source/test cells | bool `(N,S,2)`, actual class 0/1 |
| Coefficients | float64 `(H,G,S,K)`, log-odds per original PC-score unit |
| Intercepts / fit status / reason | `(H,G,S)`, log-odds / strings |
| Trial probabilities | float64 `(H,G,S,W)`, matched-window mean P(class 1) |
| Source/test masks | bool `(S,W)` |
| Source trial weights | float64 `(G,S,W)`, dimensionless totals |
| Eligible cell counts | integer `(S,2,2)`, source/test role then actual class |

Replace split-axis constants in preparation, validation, fixtures and reporting
with the fixed registry length. Preserve independent literal expectations for
scientific memberships and the eighteen-model contract in tests. Do not blindly
replace `16`: current test fixtures also have N=16 choices, which stays 16.

Reuse `training_trial_weights`, `stack_training_window`, `fit_readout`,
`fit_region_transfers` and `trial_metrics`. Both schemes retain total source
weight Ntrain when both classes exist; selected bins divide each trial's total.
Keep mean(expit(logit)) trial probabilities and equal trial evaluation weight.

There are `18*3*2*2 = 216` maximum fits per bin width, an increase of 24.
The three outcome-training models account for 36 fits, of which 12 are the
already present incorrect-trained model. Six directed evaluations do not
become six independently fitted outcome models.

### 3.3 Summary records

Extend `summarize_transfers` rows with the three training-count fields in
Section 2. Existing row keys and metrics retain their meanings; counts are
derived from saved models, so add no duplicate stored count arrays.

Add a small pure `context_outcome_rows(summary_rows)` helper in `scoring.py`.
It selects the fixed three model IDs and annotates training/test populations:
for complementary-test rows, select each model's two other outcomes; for
training diagnostics, select only its own training population. It consumes
existing saved-summary rows, performs no fit or source I/O, and does not pool
different models or outcomes. Keep deterministic training-outcome/test/region
order. Empty populations and unavailable fits/regions retain reasons.
Retain all original row fields and add `training_population` plus
`test_population` (the target outcome for complementary tests; None for source
training diagnostics). Mapping order is correct-rewarded, omission, incorrect;
test populations follow each mapping entry and regional order is PFC, HPC.

At one window/weighting with two regions, the new test table has 12 rows
(3 training outcomes * 2 test outcomes * 2 regions). Training diagnostics have
6 rows (3 models * 2 regions), labeled in-sample. A change to displayed outcome
or view never changes training counts or coefficients.

### 3.4 Persistence, identity and runners

Use `cross-condition-generalization-v2`, schema 2, and
`cross-condition-summary-v2`. Keep metadata input version 3. Keep the existing
array names, expanding only S from 16 to 18. Include the fixed context-outcome
mapping in effective scientific metadata/identity and validate it on loading.
Continue saving source/test names, model IDs, masks, counts and probabilities.
Frozen metadata key `effective_config["context_outcome_models"]` is an ordered
list of three JSON dictionaries with split_id, training_population and
test_populations (two strings), in the fixed training-outcome order.

Update fixed-shape validators and registry checks in `results.py`; preserve
self-contained loading, metadata-only discovery, atomic publication and the
existing no-overwrite rule. Source PCA identities and compatibility rules are
unchanged. Cross-condition result versions and source-PCA versions are separate.

Prepare a fresh v2 decoder bundle while reusing compatible PCA. Keep existing
v1 files intact; v2 discovery/loader explain or skip unsupported v1 results
using existing version checks. Add no v1 migration, dual-version loader or
cross-version coefficient-import path. Viewing new results requires preparing
the expanded bundle once.

Keep both runner signatures, CLI flags and controls JSON unchanged. Derive
planning/model totals from the registry, including 108 maximum fits per region;
update logs, summaries and source snapshots. The existing batch wrapper should
need no scientific logic change. Preserve early exact-result reuse and one
fitting-stage load per distinct selected PCA archive.

## 4. Webapp behavior

In `webapp/cross_condition_view.py`, show a **Context analysis** selector only
for Context, with **Condition transfers** (original four) and **Outcome
transfers**. Default to the original condition view. Choice, Actual reward and
Value retain their existing four-direction views.

Outcome transfers retain saved run/window/weighting and Evaluation role
selectors. For Complementary test, show all six directed outcome evaluations
in one table, with training outcome and test outcome explicitly labeled. Hide
the generic Outcome selector in this view because both relevant test outcomes
are already separate rows. For Training (in-sample), show the three source
diagnostic models instead of duplicating them into test-outcome rows.

The complementary-test table contains training outcome, test outcome, region,
training total/Left/Right counts, test eligible/scored/Right/Left counts,
accuracy, balanced accuracy and reason. Keep counts plainly visible and formats
compact. Keep exact IDs/cell names, fit reasons and conflict audit in Fit details;
filter details to the selected view's model IDs. Display 0 accuracy as 0, and
single-class ordinary accuracy even when balanced accuracy is unavailable.
Frozen new table labels: Training outcome, Test outcome, Region, Training trials,
Training Left, Training Right, Eligible trials, Scored trials, Right trials,
Left trials, Accuracy (%), Balanced accuracy (%), Reason. Eligible/scored/class
counts refer to the selected Evaluation role; captions make that explicit.

Use the existing path/stat/scoring-version caches. Switching Context analysis,
window, weighting or role reads only saved small products. Explicit Prepare /
Recompute still prepares the whole 216-model bundle at the requested bin width.
Preserve prior selection on preparation error and saved settings' authority.
Inspect layout in the actual browser; counts must not require opening an expander.

## 5. File boundaries and dependencies

| Owner | Files / permitted changes |
| --- | --- |
| Sol scientific worker | `cross_condition/config.py`, `preparation.py`, `modeling.py`, `scoring.py`, `results.py`, `run_session.py`; localized registry, axis, count, metadata and reporting changes |
| Sol scientific/UI worker | `webapp/cross_condition_view.py`; new Context selector, filtered tables and visible training counts |
| Sol worker tests | Existing cross-condition `conftest.py`, `test_config.py`, `test_preparation.py`, `test_modeling.py`, `test_scoring.py`, `test_results.py`, `test_scripts.py`; `test_cross_condition_webapp.py`; batch tests only if expected report totals require it |
| Luna documentation worker | Cross-condition README and localized neural-analysis/task-decoding README wording or links |
| Orchestrator | Scientific spec, this plan, original-plan follow-up link; reviews, acceptance and commits |

No new dependency. Use installed NumPy/pandas/SciPy/scikit-learn/Streamlit and
existing project helpers. Verify any newly used library API from its installed
source or official documentation. Preserve existing public call signatures;
update all summary-dictionary consumers for the additive fields. Every changed
function documents types, axes, units and returns. No changes to external
libraries, PCA preparation, generic targets, app routing or batch scheduling
are expected.

## 6. Test-first work packages

### C1. Registry and preparation (Sol)

Tests first: preserve the original sixteen IDs/memberships, append exactly two
IDs, and verify all three source-training definitions with independent literal
trial IDs. Assert correct-rewarded excludes omissions and incorrect trials;
omission excludes rewarded/incorrect trials; both test outcomes are present and
disjoint from source. Cover rewarded and missing-reward incorrect trials,
correct-with-unknown-reward exclusion, numeric strings, dark/conflict/manual
exclusions, and input immutability. Assert the new mapping references the existing
incorrect model once. Extend fixtures only when needed for unequal counts.

After RED/test commit, implement registry/cells/version controls and document
S=18. Gate: cross-condition config/preparation plus existing shared target and
condition tests. Do not require later-package hard-coded shapes to pass yet.

### C2. Modeling, summaries and training counts (Sol)

Tests first: independent sklearn coefficient/intercept/probability oracles for
both new source pools at each window/weighting; 108 fit calls per full region,
with the incorrect-trained model fitted once. Check trial/bin weight totals
after regional coverage and unequal context counts. Perturb only test scores
and confirm source fit invariance while the source pool and PCA basis stay fixed.

Hand-enumerate all six outcome-pair summary memberships and metrics. Verify
training total/Left/Right counts after coverage, invariance to test-outcome
selection and weighting, and retention after missing-class/numerical failures.
Unknown regional coverage is None. Test one-trial source cells, empty pools,
single-class test rows, zero accuracy and all-region unavailability using the
existing failure policy. Compare the original sixteen entries against their
previous independent numerical expectations, not just new registry constants.

After RED/test commit, implement dynamic split validation, additive count fields
and the pure outcome-row helper. Gate: cross-condition modeling/scoring plus
existing session-prediction modeling/scoring tests.

### C3. Saved results and runners (Sol)

Tests first: schema-2 round trip of every S=18 field; contradictory shapes,
definitions and mapping rejected; fitting identity covers new definitions;
v1 results stay untouched and unsupported by v2 discovery/loading. A copied v2
bundle loads and supplies counts/summaries with raw/base/PCA sources absent.
Retain interrupted-publication and no-overwrite cases.

Runner tests verify 216 total / 108 regional maximum fits, unchanged CLI/control
forwarding, updated dry-run counts, confident PCA reuse, exact result reuse
before score loads/fits, shared/separate source archives, unavailable regions,
dated reruns, and actual script/scientific-source snapshots. Reuse existing
runner tests; revise only changed version/count/registry expectations.

After RED/test commit, implement persistence and run-document updates. Gate:
all cross-condition tests plus old session-prediction result/script tests.

### C4. Summary view (Sol)

Tests first in real Streamlit AppTest: original four Context condition rows per
region remain accessible; outcome view shows all six pairs per region (12 rows)
with training total/Left/Right counts, and source diagnostics have 6 rows. Counts
match saved regional coverage and repeat consistently for each source model.
Check all windows/weights, outcome-selector visibility, target/view switching,
zero/single-class metrics, failed-fit counts and both unavailable regions.

Disable fitting/PCA/raw-input APIs while exercising selectors and saved-only
copies. Verify cache invalidation by scoring version/stats, authoritative saved
bin width, explicit full-bundle preparation, and selection preservation after
failure. Existing other-target views and actual early app dispatch must pass.

After RED/test commit, implement the localized view changes. Gate: new/updated
cross-condition webapp tests plus existing condition-generalization,
session-prediction, session-timeline and webapp-package tests.

### C5-C7. Integration, docs and final review

Root runs the complete regression gate:

```bash
uv run --frozen --no-sync --no-cache pytest \
  src/tests/neural_analysis/task_decoding \
  src/tests/neural_analysis/test_cross_condition_webapp.py \
  src/tests/neural_analysis/test_condition_generalization_webapp.py \
  src/tests/neural_analysis/test_session_prediction_webapp.py \
  src/tests/neural_analysis/test_session_timeline_webapp.py \
  src/tests/neural_analysis/test_webapp_package.py -q
```

Run a seeded synthetic session through the actual runner, verify all 216 fits
when eligible, independently reconstruct new matched-window probabilities,
and compare all six outcome summaries and training counts. Inspect the actual
webapp at 500 and 100 ms, including a saved-only copy, unequal regional coverage,
visible small training groups and explicit preparation. Record measured runtime,
peak RSS, reused PCA identity and source/test counts. Do not infer experimental
performance from the small synthetic fixture.

Then Luna updates usage docs and the new result-version/repreparation guidance;
root reviews scientific wording and actual commands. Check new local links,
ASCII, fenced blocks, changed-file Ruff checks and `git diff --check`. Record
exact RED/GREEN commands, commits, remaining limitations and process cleanup.
Documentation-only follow-ups do not rerun already accepted analyses.

## 7. Performance and acceptance

Keep fits sequential within a session and reuse loaded regional scores. Keep
one selected design in memory and retain only trial means. Probability storage
rises from `768*W` to `864*W` bytes per region before compression (12.5% more),
with small mask/parameter/count additions. No PCA refit, extra source-score
copies, within-session parallelism or optimization dependency is required.

C1-C7 acceptance requires all six requested evaluations from three unique source
models, clearly visible correct regional training counts, unchanged original
definitions/numerics, saved-only viewing, compatible PCA reuse, tests, synthetic
and browser validation, docs and an accurate handoff. Record actual model/row
counts; do not label the six evaluations six independent fits.

CT026 2026-08-28 remains the user's designated inspection session. The previous
read-only v1 dry-run found 695 choices, no dark/conflict rows and existing
500-ms PFC/HPC PCA. This planning request authorizes no experimental computation.
When new experimental results are requested, recheck current inputs, prepare
one fresh v2 bundle using the compatible PCA, inspect all three training groups
and their class counts, then let the user assess scientific results before any
experimental batch. Do not add an automatic all-session job.

## 8. Execution evidence

### Approval / C1 RED, 2026-11-10

User explicitly requested implementation; approved documents committed as
`2550e60`. Sol worker changed only config/preparation tests. Independently
reviewed literal memberships for the original sixteen and both new pools.
No fixture changes: existing HPC coverage intentionally removes every left
correct-rewarded source trial, exercising a locally unavailable fit later.

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition/test_config.py src/tests/neural_analysis/task_decoding/cross_condition/test_preparation.py src/tests/neural_analysis/task_decoding/test_targets.py src/tests/neural_analysis/task_decoding/test_conditions.py -q`
returned 6 expected failures, 143 passed (worker 2.10 s; root 2.09 s).
Test commit `f9e332a`; source-only C1 GREEN assigned. User-owned hashes match
accepted baselines; no experimental processing or live analysis jobs.

Root prepared fresh synthetic source PCA with existing tested timeline code at
`/tmp/context_outcome_synthetic_o4n0pmoc/synthetic/neural_session.json` (0.125 s).
There are 42 choices: 26 correct-rewarded, three omissions and 13 incorrect;
HPC has shorter trusted coverage excluding a late left-context omission.
This fixture supports the later count/browser acceptance; no extension decoder
has been run yet. All randomness uses seed 602 plus existing fixture seeds.

### C1 GREEN / review, 2026-11-10

The same focused C1 command passed all 149 tests in 2.04 s. Reviewed the
append-only registry and literal cells, recorded-context orientation, canonical
outcome unions, immutable mapping, original Context grouping and independent
cross-condition version bump. Only config.py/preparation.py changed; tests,
public signatures, original sixteen memberships and PCA versions are unchanged.
Implementation commit `8e586c4`. Sol worker returns to C2 RED. No live analysis
process or experimental work. Frozen outcome-row annotations and ordering are
recorded in Section 3.3 before C2 tests.

### C2 RED / test commit, 2026-11-10

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition/test_modeling.py src/tests/neural_analysis/task_decoding/cross_condition/test_scoring.py src/tests/neural_analysis/task_decoding/session_prediction/test_modeling.py src/tests/neural_analysis/task_decoding/session_prediction/test_scoring.py -q`
returned 9 expected failures, 89 passed in 12.37 s. Root independently checked
the new missing helper/version/count cases: 5 expected failures, 10 deselected
in 1.23 s. Reviewed independent expanded sklearn fits, test-only perturbations,
six literal outcome pairs/direct metrics, count invariance/coverage/failures,
pure deterministic row selection and unavailable-region handling.

Tests committed as `ff68b6f`. Existing fit code already passes all S=18 oracles;
only summary functionality and modeling documentation need C2 source changes.
HPC's missing rewarded source class is retained: 102 available fits versus
PFC's 108. No fixture/source edits in RED and no experimental processing.

### C2 GREEN / review, 2026-11-10

The same assigned C2 command passed 98 tests in 12.97 s. Reviewed count
derivation from saved source eligibility (Python ints / unknown None), preserved
failed-fit counts, deterministic copy-only outcome selection and cache version.
Numerical fitting is unchanged; modeling.py received only an S=18 doc update.
No test edits. Implementation `4b241c3`; C3 RED follows. No live processes or
experimental work. The metadata mapping contract is frozen in Section 3.4.

### C3 RED / test commit, 2026-11-10

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/task_decoding/cross_condition src/tests/neural_analysis/task_decoding/session_prediction/test_results.py src/tests/neural_analysis/task_decoding/session_prediction/test_scripts.py -q`
returned 27 failed, 211 passed, 69 errors in 40.15 s. Expected dominant blocker:
saved-input validation still requires `(N,16,2)` and rejects prepared S=18.
Root independently verified that blocker (one failed, 80 deselected, 0.22 s).

Reviewed expanded round trips/strict mapping, counterfeit metadata, unsupported
v1 preservation, saved-only count summaries, 210 actual versus 216 maximum fits,
and a separate full-coverage 216-fit fixture with unchanged PCA-v1 bytes.
Corrected a genuine pre-commit test error: discovery returns tuple, not list;
no public API change. Tests committed as `6b55d99`. Source-only C3 GREEN
assigned; no experimental work, fixture-framework changes or live test process.

### C3 GREEN / review, 2026-11-10

The same assigned C3 command passed 307 tests in 52.95 s. Reviewed dynamic
registry-sized validation, fixed ordered outcome metadata/identity, strict v2
saved products and unchanged source-PCA/reuse code. Logs and run summaries now
distinguish actual fits from derived maximum capacity, including 108 per region.
A final prose-only qualifier clarifies that the three outcome models are per
window/weighting/region. Implementation `1717f84`; tests unchanged and whitespace
checks passed. C4 RED next; no experimental work.

### C5 early synthetic computation, 2026-11-10

The fresh synthetic session completed all 216 models in 0.233 s, with recorded
peak process RSS 317,730,816 bytes. Independently reconstructed all 216 model
trial means directly from PCA scores/coefficients/intercepts: maximum absolute
probability difference 0.0. Checked all 108 outcome/diagnostic rows across every
window/weight against source CSV memberships and independently counted coverage.

| Region | Correct-rewarded total / Right / Left | Omission total / Right / Left | Incorrect total / Right / Left |
| --- | --- | --- | --- |
| PFC | 26 / 12 / 14 | 3 / 1 / 2 | 13 / 8 / 5 |
| HPC | 21 / 9 / 12 | 2 / 1 / 1 | 10 / 6 / 4 |

Exact reuse took 0.061 s with fitting and PCA-array loading disabled. The source
PCA hash remained unchanged. Copied only metadata and the complete v2 run to
`/tmp/context_outcome_synthetic_o4n0pmoc/saved_only/neural_session.json`;
loading and all outcome summaries/counts pass without raw/base/PCA inputs.
Receipt: `/tmp/context_outcome_synthetic_o4n0pmoc/validation_receipt.json`.
These are small synthetic measurements. Browser/100-ms checks remain pending
C4; no experimental session was processed.

### C4 RED / test commit, 2026-11-10

`uv run --frozen --no-sync --no-cache pytest src/tests/neural_analysis/test_cross_condition_webapp.py -q --tb=short`
returned 17 expected failures, 2 passed in 20.56 s. Root independently verified
the old UI's six-Context-model/four-label zip failure (one failed, 18 deselected,
2.80 s). Reviewed original-view filtering, the Context-only selector, all six
pairs and source diagnostics, visible counts, exact class orientation, cache
and saved-only behavior, zero/single-class/failed/empty/unavailable counts,
and full-bundle Prepare/error preservation. Tests `4f7f724`; source-only GREEN
assigned to the same Sol worker. No experimental processing.

### C4 GREEN / review, 2026-11-10

The assigned cross-condition and existing webapp regression group passed all
75 tests in 36.51 s. Reviewed the Context-only selector, explicit original-four
filter, twelve complementary-test/six training rows, source total/Left/Right
columns, recorded-context orientation, missing-region reasons, selected-model
details and percentage/count formatting. All counts are in the main table.
Caching, preparation controls and other target views retain their behavior.
Implementation `aa9d45a`; no tests changed during GREEN.

### C5 browser and 100-ms acceptance, 2026-11-10

Browser checks first used the saved-only 500-ms copy without raw data or PCA.
Confirmed all six directed outcome pairs, twelve regional test rows and the
independent source counts above. Training role showed six source diagnostics.
Full/Before selections, both weightings and the absence of the generic Outcome
selector were checked. The main table exposes counts without Fit details;
metrics display one decimal place and remain accessible by table scrolling.

Then used the full artificial session, changed preparation width to 100 ms,
and confirmed that the selected saved run still reported 500 ms until explicit
Recompute. Recompute created a separate 100-ms PCA and v2 decoder bundle:
`cross_condition_generalization_20261110_163353_466800_c8a251b46a8b11e8a6ab`.
It completed 216 fits in 0.366 s (peak process RSS 518,979,584 bytes, including
the webapp process). Independently checked all 108 outcome/diagnostic rows
against the same raw-membership/coverage count oracle; counts did not change
with bin width. Full/Before/After contain 20/10/10 supervised bins.

All 216 saved 100-ms probability vectors were reconstructed directly from
PCA scores and saved parameters. Maximum absolute difference was
3.33e-16; all passed 1e-14 absolute/relative tolerance. An initial standalone
audit incorrectly required bitwise zero error and failed on this floating-point
roundoff; correcting that audit assumption required no feature or test changes.
Browser confirmed Before/Equal-cell training and After/Equal-cell complementary
test views with the expected small omission sources (PFC 3, HPC 2).

Additional receipt and screenshot remain in the temporary fixture directory:
`validation_100ms_receipt.json` and `context_outcome_summary.png`. Both decoder
versions here are v2; the two bin widths have separate run directories. The
500-ms source PCA still matches its recorded SHA-256 after 100-ms preparation.
temporary localhost server exited cleanly and its browser tab was closed.
No experimental data or batch was processed.

### C6 documentation / C7 final verification, 2026-11-10

Sequential Luna documentation work and root review updated the cross-condition,
task-decoding and neural-analysis guides (`0d181ed`). They describe selectors,
six evaluations from three source models, visible counts, v2 Prepare-once and
unchanged PCA-v1 reuse. Root corrected wording about unavailable-region rows
and scoped combined-pool fitting to the original condition view. All three
guides passed ASCII, code-fence and relative-link checks.

The initial combined final pytest command terminated with SIGTERM (exit 143)
at approximately 58%, with no reported test failure. It is not counted as a
passing gate. The same collection was then run as three disjoint groups using
`uv run --frozen --no-sync --no-cache pytest ... -q`:

1. `src/tests/neural_analysis/task_decoding/cross_condition`,
   `src/tests/neural_analysis/task_decoding/session_prediction`,
   `src/tests/neural_analysis/task_decoding/pooled`, and the five webapp files
   `test_cross_condition_webapp.py`, `test_condition_generalization_webapp.py`,
   `test_session_prediction_webapp.py`, `test_session_timeline_webapp.py`,
   `test_webapp_package.py` under `src/tests/neural_analysis`:
   581 passed in 136.51 s.
2. `src/tests/neural_analysis/task_decoding`, ignoring the preceding three
   subdirectories and `test_pipeline.py`: 718 passed in 28.91 s.
3. `src/tests/neural_analysis/task_decoding/test_pipeline.py`:
   166 passed in 81.90 s.

Total: 1465 passed. Existing warnings concern short Pynapple intervals, small
cross-validation classes and a test oracle's empty-slice mean. Ruff passed for
the cross-condition source/tests and changed webapp source/tests. Whitespace
checks passed. No source changed after these gates. User-owned main.py and
sync_ephys.py retain their accepted hashes; unrelated untracked files were
preserved. All implementation and test work is committed; this handoff changes
documentation only. Experimental interpretation remains for the user.
