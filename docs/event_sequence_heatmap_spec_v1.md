# Event-Sequence Neural Heatmap Visualization Spec

## Goal
Create two related visualization tools for inspecting neural activity from a **single selected unit** or **single selected PC** under behaviorally defined sequence conditions.

The two visualization types are:

1. **Sequence-selected central-trial stack**
   - Rows are trials selected by the relationship between the central trial and its neighboring trials.
   - Only the **central trial's** neural activity is shown.

2. **Multi-trial sequence stack**
   - Rows are matched multi-trial episodes, including ideal transitions around
     block/context changes and the additional families defined in Part III.
   - The **entire multi-trial sequence** is shown, with each trial displayed as its own aligned neural segment.

These are descriptive inspection tools. They are not summary-statistics or significance-testing analyses.

This revision records the scientific and workflow decisions agreed on
2026-11-05. It defines required behavior; the concrete implementation plan and
its approval are separate steps.

Part III records additional families and scientific decisions confirmed on
2026-11-08. Those additions are implemented using the existing saved neural
source products.

---

## Common Requirements

### Session scope
- Operate within one session at a time.
- Use the session's existing behavioral trial structure and neural data.
- Provide both tools as views in the existing metadata-driven Streamlit app.
- Use `neural_session.json` to resolve the behavior table and neural sources.
- Use the processed trial table for behavioral sequence selection. Downstream
  selection must not reread raw behavior logs to establish context changes.

### Behavioral field mapping
- `action`: `0=right`, `1=left`.
- `correct`: `1=correct`, `0=incorrect`.
- `reward`: `1=rewarded`, `0=unrewarded`; do not infer reward from correctness.
- `state_int`: `0=right active`, `1=left active`, `2=dark period`.
- `cur_block`: block identity recorded by the existing trial parser.
- `cur_trial` (or the supported `trial_id` equivalent): original trial identity.
- `choice_time`: UTC Unix seconds, on the same clock as aligned spikes.
- `experimenter_reward_given`: nonzero values mark manual rewards. Accept the
  existing legacy `give_reward` alias through the project's normalization.
- Preserve original zero-based row positions separately from trial IDs and
  DataFrame index labels. Never treat these three identities as interchangeable.

### Validity and original adjacency
- Every trial participating in a motif must be an ordinary completed choice:
  a finite choice time, action in `{0, 1}`, and a known zero manual-reward flag.
- A manual reward, no-choice row, missing required behavioral value, or gap in
  original trial IDs breaks the motif. Do not skip over it to find a neighbor.
- Require correctness/reward values only where the motif constrains them.
  Part I does not constrain the next trial's correctness or reward.
- Interpret string missing-value sentinels using the existing project rules.
- Select motifs on the full chronological trial table before mapping selected
  trials to any filtered neural representation.
- An absent required column is an input-availability problem, not evidence of
  zero matching motifs. Explain the missing input in the affected view.

### Neural representations
Allow visualization of either:
- **One selected unit**
- **One selected PC**

For both sources:
- Inherit good inside-brain channel selection and metadata `unit_channels`
  restrictions.
- Display probe-qualified identities so cluster IDs from different probes
  cannot be confused.

For units:
- Allow only units with the manual Phy `group` label `good`.
- Do not substitute an automatic quality label or admit `mua` units.
- Constant or silent units remain valid raw-activity sources when covered.

For PCs:
- Reuse the fixed regional session-timeline PCA defined in
  `src/neural_analysis/population/session_timeline.py`, with the existing
  PFC/HPC role-to-probe assignments.
- Fit across all ordinary completed choices with full coverage for the
  selected population, regardless of outcome or sequence membership.
- Preserve the existing pooled unit standardization, constant-unit exclusion,
  full SVD, and unwhitened scores. There is no post-PCA score scaling by default.
- Provide a population selector: **good + MUA** (default, matching the existing
  timeline scope) or **good only** (manual Phy `group` labels).
- Do **not** refit PCA separately for each subset or panel.
- The same selected PC must mean the same axis across both tools and all
  displayed subsets within the same saved population/bin-width configuration.
- Population quality selection is part of the scientific configuration and
  saved-result identity. Changing good + MUA to good only requires the matching
  separately fitted basis; filtering existing scores cannot produce that basis.
- Changing bin width or population can change PC axes, signs, and ranks. Show
  the active population mode and configuration with the selected PC.

### Activity values and optional session-wide z-scoring
- Default unit values are unsmoothed firing rates in Hz: spike count divided
  by bin width in seconds. Do not subtract a baseline.
- Default PC values are the saved signed, unwhitened standardized-unit scores.
- Offer optional **session-wide z-scoring** for either selected source.
- For a source and bin-width configuration, compute its mean and population SD
  (`ddof=0`) over all of its eligible complete choice-window bins in the session.
  Apply `(value - session_mean) / session_SD` to displayed values.
- The reference includes all eligible outcomes, not only selected motifs.
  Count each source trial/window once, regardless of how often it appears in
  matched episodes or panels. Display selectors must not change the reference.
- Label z-scored values as source SD units. This is a display transformation;
  it does not refit PCA or change saved raw activity, loadings, or PC ranks.
- If session SD is zero, mark z-scoring unavailable for that source and explain
  why; raw values remain viewable.

### Alignment and neural window
- Neural activity is aligned to the **choice time** of the relevant displayed trial.
- Use a symmetric window of:
  - **[-2 s, +2 s)** relative to choice, including -2 s and excluding +2 s.
- Use the same half-open bin-edge convention for unit and PC activity.
- The task structure guarantees that these choice windows do not overlap.
  Treat this as an input property; no overlap-handling feature is required.

### Neural coverage and counts
- Require trusted recording/synchronization coverage of the entire displayed
  window; the first and last spike do not define recording coverage.
- Part I requires full coverage for the central trial only. Its neighbors
  need valid behavioral information, but their neural activity is not displayed.
- Parts II and III require full coverage for every displayed trial in the episode.
  If any segment lacks coverage, exclude the entire episode from display.
- Do not truncate windows or zero-pad unavailable data. Measured silence within
  a covered window is valid zero firing rate.
- `n` means displayed rows after behavioral matching and coverage checks.
  When coverage removes matches, also report the behavioral match count and
  number excluded for incomplete coverage.

### Bin widths
Supported bin widths:
- **500 ms** (default)
- **100 ms**
- **50 ms**
- **20 ms**

Implementation requirements:
- **500 ms must work**
- **100 ms must work**
- 50 ms and 20 ms can be included if computationally practical.

Resulting bins per trial window:
- 500 ms -> 8 bins
- 100 ms -> 40 bins
- 50 ms -> 80 bins
- 20 ms -> 200 bins

### Heatmap conventions
- Heatmaps are acceptable and preferred.
- Columns = time bins relative to choice.
- Rows = matched trials or matched sequences.
- Show a vertical reference at **0 s**.
- Preserve the sign of the displayed values.
- Use a fixed color scale for each session/source/bin-width/scaling
  configuration across both tools, sides, central outcomes, directions, and
  motif lengths and families.
- Derive limits from all eligible session choice-window bins for that selected
  source, not from the current panel or motif. Use symmetric maximum-absolute
  limits for raw PCs and all z-scored displays; raw unit rates start at zero
  and extend to the session maximum.
- Handle an all-zero raw source with a usable display range and a clear
  constant-activity label, while preserving the measured zeros.
- Do **not** independently renormalize every panel unless explicitly requested later.
- Use light mode: opaque white backgrounds, black axes/text, readable fonts,
  labeled colorbars, and a figure caption describing the selected motif/source.

### Trial ordering
Default row ordering:
- Chronological order of the central trial, or first displayed trial for a
  multi-trial episode. For Part II this is the transition trigger trial.

### Handling missing conditions
The code must not fail when a condition does not exist.
If no matching trials/sequences are found:
- Keep the panel or condition visible.
- Show a clear message such as:
  - **"No matching trials in this session — n=0"**
  - **"No matching transitions in this session — n=0"**
- Do not fill the panel with zeros.
- Do not silently omit the condition.
- If behavioral matches exist but all lack neural coverage, state that reason
  and both counts instead of claiming there were no behavioral matches.

### Neighbor logic
All preceding/following trial relationships must be determined from the **original chronological session sequence**.
Do **not** filter trials first and then search for previous/next trials within the filtered subset.

---

# Part I. Sequence-Selected Central-Trial Heatmaps

## Goal
Display the activity around one **central trial** when that trial belongs to a specified behavioral sequence pattern.

Only the central trial's neural activity is shown.
The previous and next trials are used only to determine whether the central trial is included.

## Central-trial definition
For each displayed row, define the central trial as trial **i**.
Each row shows only the neural activity from trial **i**, aligned to trial **i** choice time.

Neighboring trials:
- previous trial = **i-1**
- next trial = **i+1**

## Outcome terminology
Use the following definitions:
- **Correct rewarded** = correct choice and rewarded
- **Omission** = correct choice and **unrewarded**
- **Incorrect** = incorrect choice (regardless of reward, though in this task incorrect is unrewarded)

When the spec says "omission", it means **correct unrewarded**, not generic no-reward.

## Choice-side terminology
- Left/right split refers to the **central trial's choice side**.
- "Same side" or "opposite side" in the sequence definitions is relative to the **central trial's choice side**.
- "Repeat" or "switch" refers to the **next trial's action relative to the central trial's action**.

## Controls
Required controls:
- Neural source:
  - unit or PC
- Region / source selector, as appropriate to the existing codebase
- Specific unit ID or PC index
- For PC sources: population mode (good + MUA or good only)
- Bin width
- Optional session-wide z-scoring (off by default)
- Central trial outcome group:
  - correct rewarded
  - omission
- Central choice side:
  - left
  - right

Optional display control:
- choose whether to show all six sequence panels for the selected central group/side at once, or one selected panel at a time

## Recommended layout
For the selected central outcome group and central choice side, display a **3 x 2 panel layout**:

- Rows = previous-trial condition
- Columns = next-trial action

Use the same top-to-bottom previous-trial order for both central outcome
groups: correct rewarded, omission, incorrect.

Column meanings:
- Column 1 = next trial repeats the central action
- Column 2 = next trial switches to the other action

### A. Correct rewarded central trial panels
Central trial i must be:
- correct rewarded
- on the selected central side (left or right)

Panel rows:
1. Previous trial = correct rewarded on same side
2. Previous trial = omission on same side
3. Previous trial = incorrect on opposite side

Panel columns:
- Repeat next action
- Switch next action

This yields the six correct-rewarded conditions:

1. correct rewarded, preceded by correct rewarded on same side, followed by repeating same action
2. correct rewarded, preceded by correct rewarded on same side, followed by switching to other side
3. correct rewarded, preceded by omission on same side, followed by repeating same action
4. correct rewarded, preceded by omission on same side, followed by switching to other side
5. correct rewarded, preceded by incorrect on opposite side, followed by repeating same action
6. correct rewarded, preceded by incorrect on opposite side, followed by switching to other side

### B. Omission central trial panels
Central trial i must be:
- omission (correct unrewarded)
- on the selected central side (left or right)

Panel rows:
1. Previous trial = correct rewarded on same side
2. Previous trial = omission on same side
3. Previous trial = incorrect on opposite side

Panel columns:
- Repeat next action
- Switch next action

This yields the six omission conditions:

1. omission, preceded by correct rewarded on same side, followed by repeating same action
2. omission, preceded by correct rewarded on same side, followed by switching to other side
3. omission, preceded by omission on same side, followed by repeating same action
4. omission, preceded by omission on same side, followed by switching to other side
5. omission, preceded by incorrect on opposite side, followed by repeating same action
6. omission, preceded by incorrect on opposite side, followed by switching to other side

## Inclusion logic
A central trial i is included in a panel only if:
- trial i exists and has the required central outcome and central side
- trial i-1 exists and matches the required previous-trial condition
- trial i+1 exists and matches the required repeat/switch action condition

No additional constraints are imposed on the next trial's correctness or reward, unless explicitly added later.

Crossing block or context boundaries is allowed whenever the specified trial
conditions are met. Do not add a same-block requirement to this tool.

The existing `make_trial_type_masks` switch/stay masks apply only to
unrewarded central trials. They must not determine these panel columns:
compare the next trial's action directly with the central action so rewarded
central trials are handled as specified.

## Empty or sparse conditions
- n=0: show an explicit empty-condition message
- n=1: show a one-row heatmap
- n>1: show all rows

## Display metadata
Each panel should show:
- sequence label
- selected side (left or right)
- trial count n
- neural source label (unit ID or PC index)
- bin width
- active population mode for PCs and raw/z-scored value units
- behavioral matches and coverage exclusions when different from displayed n

Hover or row labels should expose at least:
- original trial index
- choice time / timestamp if available
- central choice side
- central outcome

---

# Part II. Ideal-Transition Sequence Heatmaps

## Goal
Display full multi-trial sequences corresponding to idealized block/context transitions.

At a block transition:
- a **correct rewarded** trial triggers the context change
- this is followed by a run of **incorrect repeats** of that same previously correct action
- this is followed by a **switch to the other side** that is **correct rewarded**

These are intended as "ideal transition" motifs.

## Sequence definition
Let trial **i** be the triggering trial.
Trial i must be:
- correct rewarded
- the final trial of the old block, immediately before an opposite-context
  block begins in the original trial sequence

Identify the trigger from adjacent processed rows:
- `cur_block[i+1]` differs from `cur_block[i]`;
- `state_int[i]` and `state_int[i+1]` are the opposite active-side codes
  `{0, 1}`; dark-period or unknown contexts do not establish this trigger;
- the triggering correct-rewarded action is on the old active side.

This agreed definition uses the trial table. Do not require raw log review or
event-table corroboration during downstream sequence analysis. A rewarded
trial followed by an incorrect trial, without the block/context boundary,
does not establish a trigger.

After trial i:
- the next **x** trials are consecutive incorrect trials
- those incorrect trials repeat the **old side**, i.e. the same action taken on the triggering rewarded trial
- the next trial after those x incorrect trials is the **first successful switch** to the other side
- that final switched trial must be **correct rewarded**
- the incorrect run and final switched trial belong to the new block/context
  established immediately after the trigger

Allowed x values for the first implementation:
- x = 1
- x = 2
- x = 3
- x = 4

So the target conditions are:
- Correct rewarded / context change -> incorrect x1 -> correct rewarded
- Correct rewarded / context change -> incorrect x2 -> correct rewarded
- Correct rewarded / context change -> incorrect x3 -> correct rewarded
- Correct rewarded / context change -> incorrect x4 -> correct rewarded

## Transition direction split
Allow split by transition direction:
- **L -> R**
- **R -> L**

Direction is defined by:
- the side of the triggering correct-rewarded/context-change trial
- the side of the final switched correct-rewarded trial

Examples:
- L -> R means trigger trial on left, final successful switched trial on right
- R -> L means trigger trial on right, final successful switched trial on left

## Important constraints
The incorrect repeats must be:
- **consecutive**
- immediately after the triggering trial
- all on the old side
- all incorrect

The final switched rewarded trial must be:
- the **first successful switched trial** after the run of incorrect repeats
- on the opposite side
- correct rewarded

Here, the first switched choice must itself be correct rewarded and occur
immediately after exactly x incorrect old-side repeats. Do not search past an
opposite-side omission for a later rewarded choice.

Intervening trials of other types break the motif and exclude the sequence.
For example, an omission in the middle of the transition run means that transition does **not** count as an ideal transition.

## Controls
Required controls:
- Neural source:
  - unit or PC
- Region / source selector, as appropriate
- Specific unit ID or PC index
- For PC sources: population mode (good + MUA or good only)
- Bin width
- Optional session-wide z-scoring (off by default)
- Transition direction:
  - L -> R
  - R -> L
  - optionally both as a selector, but only one displayed at a time
- Sequence length condition:
  - x1
  - x2
  - x3
  - x4

Only **one transition condition should be shown at once** to avoid visual confusion.

## Visualization layout
Each row represents one matched transition sequence.
The row is composed of concatenated trial-specific neural windows.

For each trial in the sequence:
- show the trial's neural activity aligned to that trial's own choice time
- use the standard -2 s to +2 s window
- use the selected bin width

Concatenate the sequence's trial windows from left to right in this order:
1. trigger rewarded/context-change trial
2. incorrect repeat #1
3. incorrect repeat #2 (if present)
4. incorrect repeat #3 (if present)
5. incorrect repeat #4 (if present)
6. final switched correct-rewarded trial

### Inter-trial gaps
Use a **small fixed separator gap** between displayed trial windows.
This gap is a display separator only.
It must:
- be visually consistent across rows and across conditions
- **not** represent the real elapsed time between choices

This is a trial-aligned sequence visualization, not a real-timescale timeline.

## Displayed width by condition
The total width depends on x.
For example:
- x1 condition = 3 trial segments
- x2 condition = 4 trial segments
- x3 condition = 5 trial segments
- x4 condition = 6 trial segments

All rows within the currently selected condition use the same sequence structure.

## Inclusion logic
A matched ideal-transition sequence is included only if all of the following are true:
- the triggering trial exists and is correct rewarded
- it is a true context-change trigger
- the next x trials exist and are all consecutive incorrect repeats on the old side
- the next trial after those x incorrect repeats exists and is the first successful switched correct-rewarded trial on the opposite side
- no disqualifying intervening outcome types appear in the motif
- all displayed trial windows have complete neural coverage

## Empty conditions
If no sequences satisfy the selected condition and direction:
- keep the view active
- show a clear message such as:
  - **"No matching transitions in this session — n=0"**
- do not fail
- do not silently substitute another condition

## Display metadata
The displayed condition should show:
- transition motif label (x1/x2/x3/x4)
- direction (L -> R or R -> L)
- number of matched sequences n
- neural source label
- bin width
- active population mode for PCs and raw/z-scored value units
- behavioral matches and coverage exclusions when different from displayed n

Hover or row labels should expose at least:
- trigger-trial index
- full list of trial indices in the sequence
- trigger side
- final switched side

---

# Part III. Additional Multi-Trial Sequence Families

Status: scientific definitions confirmed and implementation completed 2026-11-08.
Existing Part I/II definitions remain in force. Implementation planning and
approval are separate steps.

## Notation and patterns

- `A` is the side chosen on the first correct-rewarded trial; `B` is the
  opposite side. Include both left/right orientations.
- `R_A` / `R_B`: correct and actually rewarded choice on the named side.
- `O_A`: correct unrewarded choice on side A, using the existing omission
  definition. It is not generic absence of reward.
- `I_A` / `I_B`: incorrect choice on the named side. Retain the existing
  incorrect definition without adding a reward requirement.
- `->` denotes the immediately next original trial; do not skip intervening
  trials. `[x trials]` means exactly x consecutive trials of that type,
  including the first trial after the switch.
- `|` marks a true block/opposite-context change immediately after the trial
  to its left, established from adjacent processed trial rows.

| Family | Complete episode | x | Displayed trials |
| --- | --- | --- | --- |
| Probe | `R_A -> I_B [x trials] -> R_A` | 1-4 | x+2 (3-6) |
| Omission probe | `R_A -> O_A -> I_B [x trials] -> R_A` | 1-4 | x+3 (4-7) |
| Lucky omission-switch | `R_A -> O_A \| R_B` | Not applicable | 3 |
| Indecisive transition | `R_A \| I_A [x trials] -> R_B -> I_A -> R_B` | 1-3 | x+4 (5-7) |

The final `R_A` in both probe patterns must be correct and actually rewarded.
A correct omission does not qualify as the return trial; do not skip it to
find a later reward.

## Probe and omission probe

For a probe, the initial rewarded action is followed immediately by an
opposite-side incorrect excursion lasting exactly x trials, then an immediate
return to side A. Count the first opposite-side incorrect trial in x; do not
add x more repeats after it.

For an omission probe, insert exactly one correct omission on side A between
the initial rewarded trial and the incorrect excursion. The omission repeats
the initial action; the excursion starts on the following trial.

The entire episode must stay within one block and active context: all rows
have the same `cur_block` and `state_int`, with A remaining the active/correct
side throughout.

## Lucky omission-switch

Boundary placement:
- Initial `R_A` and the following `O_A` belong to the same old block/context.
- The context changes immediately after `O_A`: the following row has a
  different `cur_block` and the opposite active `state_int` in `{0, 1}`.
- The animal immediately switches to B, which is correct and rewarded in the
  new context. There are no intervening incorrect trials.

Displayed segment numbers are one-based: segment 2, `O_A`, is the final
old-context trial (zero-based segment index 1). Segment 3, `R_B`, is the first
new-context trial (index 2). Boundary labels/hover identify `O_A` as the
pre-boundary trial; they must not label the initial `R_A` as the trigger.

This family allows a correct omission as the pre-change trial. The
existing Part II ideal-transition trigger specifically requires reward;
matching this new family must not silently change that existing requirement.

## Indecisive transition

- Start with the existing correct-rewarded context-change trigger on A.
- Follow it with exactly x incorrect repeats on A, where x is 1, 2, or 3.
- Switch to B on a correct-rewarded trial.
- Immediately switch back to A for exactly one incorrect trial.
- Immediately switch again to B on a correct-rewarded trial.
- All post-trigger trials remain in the same new block/context. An additional
  context change or disqualifying intervening trial breaks the motif.

Only the initial incorrect run varies with x. The later incorrect return to A
is one trial, not another x-trial run.

An indecisive episode contains a Part II ideal-transition match as its prefix.
The existing definition does not constrain the trial after its first rewarded
switch. Allow the event to match both families: show the shorter prefix in
the ideal-transition family and the full episode in the indecisive family.
Preserve the existing ideal-transition definition. Family counts are not
mutually exclusive; do not remove a match solely because another family
contains the same event.

Its final `R_B -> I_A -> R_B` also satisfies the opposite-side x1 probe
definition within the new block/context; retain that match as well.

## Shared display and inclusion requirements

- Use the full-sequence layout from Part II: display every trial's own
  choice-aligned `[-2,+2)` window, fixed display separators, and local zero
  references. Support episodes of up to seven displayed trials.
- Select one family and condition at a time. Probes use initial-side/choice
  paths `L -> R -> L` or `R -> L -> R`; these are not context-change directions.
  Lucky and indecisive transitions use the old-to-new context direction.
- Expose x only for families with a variable run length, using the ranges in
  the table. Do not apply the original x=1-4 control to every family.
- Apply the common ordinary-choice, original adjacency, trial-ID continuity,
  and missing-value rules to every participating row. Intervening trials
  break the motif; do not search ahead for a later successful return.
- An absent `cur_block` or `state_int` column makes all Part III families
  unavailable, with an explicit missing-field reason; Part I remains usable.
  Missing context values in individual rows invalidate only episodes containing
  those rows, rather than making the whole family unavailable.
- Require full neural coverage for every displayed segment. Report behavioral
  matches and coverage exclusions separately, and retain explicit empty views.
- Reuse the same prepared source, session PCA axes, session-wide z-scoring,
  and color limits across families. Family selection must not refit PCA.
- Order rows by episode-start chronology. Labels/hover include family,
  initial side or context direction, x when relevant, all original trial IDs,
  and the pre-boundary trial identity when applicable.

---

# Implementation and Robustness Notes

## General robustness
The implementation must:
- tolerate missing conditions without error
- keep independent panels/conditions functional even if others are empty
- use the original chronological trial sequence for all sequence logic
- avoid off-by-one mistakes at session boundaries

Trials lacking the required previous or next neighbor should simply be ineligible for the relevant sequence condition.

## Computational scope
Behavioral selection and rendering should be lightweight relative to full
summary analyses. Loading or initially preparing a full session PCA archive
can still require substantial time and memory, especially at 20 ms.

For PCs:
- reuse an exactly matching saved session-timeline basis and scores;
- preparation for good-only populations must remain separate from the existing
  good + MUA configuration, and must not replace existing results;
- select one PC for rendering rather than sending the full PC tensor to the
  browser.

For units:
- reuse existing spike loading and choice-window extraction where possible;
- existing unit-rate utilities include the final window edge, whereas the
  timeline uses half-open windows. Reuse must preserve the agreed half-open
  contract rather than inherit that inconsistency;
- unit-rate arrays are not stored in existing session-timeline PCA archives,
  so unit activity requires its own extraction/preparation.

### Preparation and display
- Units use dedicated Probe / Unit / Bin width controls. Load a compatible
  saved product directly; selecting an unprepared unit/bin-width combination
  automatically prepares and saves only that selected unit's activity.
- PCs retain an explicit Prepare action for a missing compatible
  population/bin-width product, reusing a compatible saved PCA basis when
  available. Recompute remains an explicit action for either source type.
- Changing a behavioral condition, source rank, or display scaling must not
  itself trigger neural preparation or refit PCA. Existing source-change
  invalidation still applies; adding or selecting a family alone does not
  invalidate the prepared neural source.
- Keep computation/preparation separate from plotting prepared values.
- Reuse compatible results; do not overwrite outputs from another source,
  parameter set, or analysis version.
- Follow repository rules for any persisted analysis products: named arrays,
  `meta` containing generator/version/parameters/source identities/units/axes
  and random seed when applicable, session-local output directories, and run
  summaries/logs. Keep experimental products outside version-controlled code.

### Data contracts to preserve
- Prepared PC scores retain the existing axes `(choice_window, time_bin,
  session_pc)` and mappings back to original trial positions/IDs.
- Displayed central-trial activity has axes `(matched_trial, time_bin)`.
- Displayed transition activity is logically `(matched_episode, trial_segment,
  time_bin)`; concatenation is a documented display transformation only.
- Separator gaps contain no measurements. They must never contribute to
  session normalization, color limits, or trial counts.

## Codebase verification
Codex should verify against the existing codebase:
- how choice side is encoded
- how correct/rewarded/omission/incorrect are encoded
- how true context-change trigger trials are identified
- how session boundaries and invalid trials are represented
- how PC activity and unit activity are currently stored or computed

If existing naming/coding differs from the wording in this spec, Codex should map the codebase variables to the intended behavioral definitions and document that mapping.

Include a focused review of the project's trial-logging/preprocessing pathway
in the later implementation plan: confirm that context entries are captured
in `cur_block` and `state_int` for the appropriate trials. If a concrete gap is
found, report it and propose a correction at trial-table generation so every
downstream analysis can use those fields. This does not authorize modifying
the external `RPi4_behavior_boxes` repository; any inspection there is read-only.

---

# Interpretation Notes

These viewers are intended for **visual inspection**.
They do not by themselves establish significance, causal structure, or unique encoding.

For the central-trial stack:
- neighboring-trial conditions are used to create more behaviorally homogeneous subsets
- only the central trial's neural activity is displayed

For the ideal-transition stack:
- the visualization displays a full structured sequence of trials around a transition motif
- the fixed inter-trial gaps are purely visual separators, not elapsed time

Both tools should make it easier to inspect whether one unit or one PC shows recognizable activity structure under behaviorally meaningful sequence patterns.
