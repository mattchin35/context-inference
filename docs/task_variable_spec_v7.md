# Task-Variable Decoding Scientific Contract Revision 7

## 1. Status and authority

This document is the active normative amendment to
`docs/task_variable_spec_v5.md` and `docs/task_variable_spec_v6.md`. Revision 5
remains the base scientific contract, and revision 6 remains the convergence
amendment. Every earlier requirement remains binding unless this amendment
explicitly replaces it.

Revision 7 adds condition-resolved decoding and changes the analysis identity
to `task-variable-decoding-v3`. It does not change target definitions, neural
units, time bins, training-fold transforms, grouped cross-validation,
estimators, metrics, coefficient interpretation, or the invalidity policy.

## 2. Canonical conditions

The configured `condition_names` field is a nonempty subset of this canonical
order:

1. `all`: every chronological trial before ordinary target and neural-coverage
   eligibility is applied;
2. `correct_rewarded`: no experimenter reward, `correct == 1`, and
   `reward == 1`;
3. `omission`: no experimenter reward, `correct == 1`, and `reward == 0`;
4. `incorrect`: no experimenter reward, `correct == 0`, and a nonmissing
   current action;
5. `switch`: a valid, unrewarded current trial whose immediately following
   valid trial has a different action; and
6. `stay`: a valid, unrewarded current trial whose immediately following valid
   trial has the same action.

These masks must come from the existing
`src.neural_analysis.spike_behavior.trials.make_trial_type_masks` definitions.
The task-decoding package may validate and copy those masks, but must not
reimplement different condition semantics. `switch` and `stay` label the
current unrewarded trial using its next chronological choice; they do not shift
the neural alignment to the following trial. Condition masks may overlap.

Configuration loading restores canonical order regardless of JSON order and
rejects unknown names and duplicates. Omitting `condition_names` remains a
backward-compatible request for `all` only. The portable full-analysis example
selects all six conditions.

## 3. Condition-target execution

Every configured target is evaluated under every configured condition. There
is no condition-specific target subset. For each cell, scientific eligibility
is the intersection of the full-table condition mask, the existing
target-specific eligibility mask, and the common bilateral neural-coverage
rows. Chronological target derivations are still constructed on the complete
table before condition filtering, so a filter cannot create an artificial
previous or next trial.

Grouping remains by behavioral block after filtering. Scaling, PCA, inner
selection, and estimation remain training-fold-only. A condition-target cell
with no eligible rows, a constant target, or another already declared
scientific invalidity is retained as explicit unavailable output; it is not
dropped, pooled with another condition, or assigned a fabricated score.

Execution is condition-major in canonical configured order, then target-major
in canonical configured order. Durable progress uses `condition::target`
identifiers. Condition-aware checkpoint files use
`condition--target.npz`. Their run fingerprint and saved condition-target
identity must match before reuse. A pooled `all`-only run retains the revision-6
checkpoint names for backward compatibility.

## 4. Saved result and presentation contract

Condition-aware runs use result schema version 2. Every scientific array that
can change after condition filtering has a leading condition axis. Shared
identities, including target labels, target families, time bins, region and
representation labels, unit identities, and encoded full-table target values,
are stored once. Full-table Boolean `condition_masks` have axes
`(condition, full_table_row)`. Target eligibility has axes
`(condition, target, full_table_row)`. Scores have axes
`(condition, target, region, representation, metric, time, fold)`.

Schema-version-1 pooled runs remain valid read-only inputs and are presented as
the single condition `all`. New condition-aware writes must validate each
condition slice through the existing pooled scientific validator as well as
validate condition identity, membership, axes, and shape.

The saved-results webapp exposes an explicit condition selector. Default PNGs
are generated for every selected condition and present target family. Their
names are condition-qualified, for example
`omission--categorical_auc.png`. Titles and captions identify the selected
condition. Completion requires the entire deterministic condition-qualified
PNG set; a partial figure directory is not complete.

## 5. Resource and validation policy

Dry-run fit counts and the resource envelope include `condition_count` and
multiply requested work by the number of configured conditions. With all 18
targets, six conditions, 40 time bins, three regions, two representations, and
five outer folds, fixed mode requests 129,600 outer cells before declared
invalidities reduce actual fitting. Tuned mode remains separately gated and is
not authorized by this amendment.

The first experimental all-condition cluster run uses the reviewed
`condition_validation` resource profile:

- one task and one CPU;
- `8G` memory;
- `3-00:00:00` (72 hours) wall time; and
- `B:TERM@300`, a five-minute termination warning.

The existing `standard` profile remains one CPU, `3G`, and `2-00:00:00` for
previously measured pooled workloads. The first all-condition run is
intentionally conservative. After completion, inspect durable pipeline
evidence and Slurm `Elapsed`, `TotalCPU`, `MaxRSS`, `ReqMem`, and `Timelimit`
before proposing a smaller normal profile. Do not reduce resources from a
runtime estimate alone.

## 6. Acceptance gates

Before an experimental all-condition submission:

1. unit and schema tests must cover exact shared masks, canonical
   configuration, condition-major checkpoints, unavailable cells, plotting,
   webapp selection, and both result-schema versions;
2. a seeded two-region synthetic run must execute all six conditions through
   real loading, binning, modeling, checkpointing, schema-2 publication,
   figures, completion validation, and complete-run reentry;
3. the portable example and both READMEs must document the condition contract
   and the `condition_validation` submission profile; and
4. the exact source commit must be pushed and the cluster checkout updated and
   tracked-clean.

Experimental transfer and submission remain separately authorized actions.
The first real run must use the exact 18-target configuration for every
condition and the `condition_validation` profile.
