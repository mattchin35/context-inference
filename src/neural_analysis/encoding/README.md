# Joint context, choice and outcome encoding

The current analysis (`joint-task-encoding-v2`) fits unit firing rates in Hz and
all numerically nonzero PCs from independent descriptive PFC/HPC PCAs. It uses
only choice-relative **[-1,+1) seconds**, fitting the whole window. Context and
current choice remain fixed throughout; six side/outcome indicators are zero
before choice and active in [0,+1). Manual-alignment sessions remain deferred.

The scientific contract is [single_unit_ridge_spec_v1.md](../../../docs/single_unit_ridge_spec_v1.md),
revised to the approved joint model. Architecture, tests-first commits and
validation gates are in [single_unit_joint_plan_v2.md](../../../docs/single_unit_joint_plan_v2.md).

## Set up a current configuration

From the repository, using the existing uv environment:

```bash
uv run python -m src.neural_analysis.encoding.run_session init-config \
  --session-metadata /path/to/session/neural_session.json
```

This writes `context_encoding_config.json` beside the metadata. Input locators
come from the existing `task_decoding_config.json`; the initializer reads only
its locator fields. Without that file, provide paths explicitly:

```bash
uv run python -m src.neural_analysis.encoding.run_session init-config \
  --session-metadata /path/to/session/neural_session.json \
  --augmented-trials processed/augmented_trials.csv \
  --feature-parameters processed/trial_feature_params.json
```

Relative input overrides resolve from the metadata directory. Version-3 metadata
provides probe roles. Version-2 metadata requires `--pfc-probe ProbeA` and/or
`--hpc-probe ProbeB`. Only declare regions actually available. The reviewed
[JSON example](../../../docs/examples/neural_analysis/context_encoding_config.json)
shows every current setting. Initialization defaults to 500-ms bins, both
representations, signed/separate context, ridge/OLS, alpha=1 and both penalty
comparisons. `--bin-width-ms 100` (also 50 or 20) changes initial resolution.

**Existing configurations are preserved.** If testing with an earlier version,
first keep a copy of the old config and create a new JSON from the example,
retaining the verified input paths and probe roles. Require
`"analysis_version": "joint-task-encoding-v2"`; remove `windows`, `timing_modes`
and `timeline_result`. Add `"response_representations": ["units", "pcs"]`.
An old/missing version or obsolete option fails explicitly. Old products remain
on disk and are omitted from current saved-run selection; they cannot be resumed
or silently treated as joint-model results.

The source and augmented CSVs must describe the same complete chronological
trial sequence, including `cur_trial`, `cur_block`, timestamps, `state_int`,
`action`, `reward`, `correct`, and normalized manual-reward flags. No behavioral
feature generation or behavioral model refitting occurs. Aligned archives need
`irig_utc_unix` coverage independently for each region; spike extrema do not
establish coverage. Existing good/mua and channel-quality rules apply.

## Run, inspect and resume

```bash
uv run python -m src.neural_analysis.encoding.run_session dry-run --config /path/to/session/context_encoding_config.json
uv run python -m src.neural_analysis.encoding.run_session new --config /path/to/session/context_encoding_config.json
uv run python -m src.neural_analysis.encoding.run_session status --run-directory /path/to/session/analysis_runs/context_encoding_RUN
uv run python -m src.neural_analysis.encoding.run_session resume --run-directory /path/to/session/analysis_runs/context_encoding_RUN
```

`dry-run` reads small tables/metadata and archive member names, without loading
spikes, fitting PCA/models, writing output, or starting workers. Counts and exact
coverage remain pending. The full example requests **20 unique cells** across
two regions and representations: signed ridge shares its penalty aliases, and
OLS runs once per design. With only `same_numeric`, there are 16 cells.

`new` reuses identical completed current work; partial work reports its directory
for explicit `resume`. `new --rerun` creates a new timestamped run. Compatible
current preparation can be shared across model settings. Resume needs no live
inputs after preparation publication; changed numerical source/runtime requires
a new run. Published scientific products are never overwritten.

Open saved results with:

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- --session-metadata /path/to/session/neural_session.json
```

Choose **Context encoding results**, then the saved run, region, **Units (Hz)**
or **PCs**, encoding, estimator and penalty comparison. Select an original
probe:cluster ID or PC rank for the prediction trace. Ordering can use a model
coefficient or a raw left/right reward-minus-omission contrast. Unsupported
conditions/contrasts appear unavailable. Raw/standardized coefficient controls
never rescale activity. Coefficient and activity rows share the same identities
and order; time panels pan together and the coefficient axis stays categorical.

Held-out predictions are the default. Select reference-fit predictions explicitly
when desired. Unavailable held-out values stay blank. At 20 ms, bounded neural
loading defaults on; time controls and Jump load another interval. Display
controls use only saved data and never fit or load spikes.

New runs export two light-mode opaque PNGs per available region/representation
under `figures/`. The saved view exports additional selected-response examples
with exact elapsed range, units and prediction scope in their captions.

## Scientific interpretation

- `state_int` and `action` use right=0, left=1. Signed columns are -1/+1;
  separate context uses `left_context`, `right_context` indicators plus the same
  signed choice. Context is the entering state, including switch/terminal trials.
- The ordered outcome columns are `left_reward`, `right_reward`,
  `left_omission`, `right_omission`, `left_incorrect`, `right_incorrect`.
  Reward=1 defines reward; reward=0/correct=1 defines omission;
  reward=0/correct=0 defines incorrect. Side is the chosen port. Unrecognized
  reward/correctness excludes the complete trial from fitting, retaining its
  activity in the descriptive PCA pool when otherwise eligible.
- Responses are unsmoothed counts divided by bin width in seconds (Hz), or
  unwhitened scores from the fixed two-second regional PCA. Constant/silent
  units remain in unit fits. PCA can be unavailable without invalidating units.
  A previous four-second PCA has different axes and is not reused.
- Ridge minimizes summed SSE plus feature-specific L2 penalties, with an
  unpenalized intercept and native X/Y. `same_numeric` penalizes every slope
  by alpha. `matched_context` halves **only the two separate context penalties**;
  choice and outcomes retain alpha. OLS uses centered minimum-norm slopes.
- Six outcome slopes are conditional post-versus-pre changes. A positive
  reward slope alone does not distinguish reward from generic post-choice
  activity. Saved raw `left_reward_minus_omission` and
  `right_reward_minus_omission` contrasts compare those conditions, with
  condition support and centered-design estimability checks. Contrasts retain
  Hz or PC-score units even under standardized coefficient display.
- Unequal condition counts can produce a nonzero ridge contrast even for an
  equal generic post-choice response through unequal shrinkage. Compare OLS,
  the native penalty vector and support diagnostics. This regression does not
  establish causal reward sensitivity or significance. There is no redundant
  generic post-choice regressor and no before-only/after-only model.
- Five deterministic folds group complete windows by `cur_block`. Units and PCs
  use identical eligible rows/folds. Primary MSE/R-squared pool held-out bins;
  negatives are retained. Fewer than five usable blocks or failed folds retain
  reference fits with unavailable primary CV. Constant responses have MSE and
  unavailable R-squared. PC scores are conditional on descriptive session PCA
  that includes test-block activity; unit fits require no PCA.

## Products, performance and batches

Counts `(window,bin,unit)`, optional PC scores `(window,bin,original_PC)`, bin-row
identity, folds and complete source tables are saved once under session-local
`processed/context_encoding`. Unit predictions are in Hz; slopes have
`(predictor,response)` axes, predictions `(bin_row,response)`, and raw contrasts
`(side,response)`. Absolute times are UTC seconds; elapsed zero is the first
unfiltered trial start. Named NPZ arrays include a dictionary named `meta` with
version, parameters, generator, units/axes, identities and randomness provenance.

Timestamped `analysis_runs/context_encoding_*` directories hold config, progress,
logs, script copies, summary, model cells, figures and `performance.json`.
Performance includes measured preparation/PCA/fitting/publication/load/figure/
serialization seconds, bytes and process peak RSS. RSS includes imports and
prior work; these measurements do not establish CT026 performance. The saved-only
`run_session.benchmark_encoding_run(path)` loads preparation once and one
representative cell per available region/representation at a time.

Batch lists contain one config per noncomment line, relative to the list or
absolute:

```bash
uv run python -m src.neural_analysis.encoding.run_batch dry-run --config-list /path/to/configs.txt
uv run python -m src.neural_analysis.encoding.run_batch new --config-list /path/to/configs.txt --workers 2
```

Workers parallelize sessions, capped by session/CPU count, with nested BLAS
threads limited. Duplicate canonical sessions are rejected. Experimental
validation proceeds from synthetic tests to one designated session, user
inspection, and then an authorized wider batch. Poisson, alpha tuning,
permutation testing and significance labels remain deferred.
