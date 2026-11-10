# Context Outcome Generalization: Implementation Plan

Status: implementation authorized by the user, 2026-11-10. C1 RED is starting;
experimental computation is not included in this implementation run.

This extends the completed [cross-condition implementation](cross_condition_generalization_implementation_plan.md).
The [scientific specification](cross_condition_generalization_spec_v1.md),
especially Section 3.1, owns the scientific definitions. Read repository
`AGENTS.md` and [SoftwareDesign.md](SoftwareDesign.md) before implementation.

## 0. Handoff and execution rules

| Item | Current state |
| --- | --- |
| Inspected branch / HEAD | `refactor` / `b7e8a22`; recheck before implementation. |
| Authorization | User explicitly approved implementation on 2026-11-10. C1-C7 authorized; no experimental computation. |
| Baseline | v1 implementation complete; full regression 1433 passed. Historical evidence remains in the original plan. |
| User edits | Preserve `src/main.py`, `src/neural_analysis/sync_ephys.py` and unrelated untracked files. The user explicitly authorized preserving ongoing behavior-mouse changes in main.py. |
| Running work | Sol scientific worker assigned C1 RED only; no experimental jobs. |
| Next action | Review C1 RED, commit tests, then authorize GREEN. |

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
| C1 registry / preparation | Sol worker | RED starting | Record command/result/commit | Record review/result/commit |
| C2 fitting / summaries / counts | Sol worker | Not started | Record here | Record here |
| C3 saved products / runners | Sol worker | Not started | Record here | Record here |
| C4 summary webapp | Sol worker | Not started | Record here | Record here |
| C5 synthetic / browser acceptance | Sol orchestrator | Not started | Prior package tests | Record here |
| C6 usage documentation | Luna worker | Not started | Documentation only | Record checks/review/commit |
| C7 final handoff | Sol orchestrator | Not started | Prior package tests | Record final evidence |

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
