# Population task decoding

Decode recorded context, chosen port, post-choice reward/no-reward state, and
the next chosen-port switch from each region's population. This is separate
from **Single unit selectivity** and from earlier session-prediction analyses.

The scientific contract is
[population_task_decoding_spec_v1.md](../../../docs/population_task_decoding_spec_v1.md).
Use the existing environment with `uv run`; no new dependency is needed.

## Single-session computation

Initialize beside the session's metadata file. For version-2 metadata, give
the declared PFC/HPC probe IDs explicitly; version-3 roles can supply them.
The augmented trial table is optional here: the source table is sufficient if
it already contains the required recorded fields. No behavior model is refitted.

```bash
uv run python -m src.neural_analysis.population_decoding.run_session init-config \
  --session-metadata /path/to/session/neural_session.json \
  --pfc-probe A --hpc-probe B
```

This creates `population_decoding.json` without replacing an existing config.
Use `--augmented-trials /path/to/augmented.csv` when needed. Source and augmented
tables must agree in complete chronological order on trial/block IDs, times,
context, action, reward and manual-reward flags. Correctness/model-value columns
are unnecessary. Probe quality and good/mua selection follow existing loaders.

```bash
uv run python -m src.neural_analysis.population_decoding.run_session dry-run \
  --config /path/to/session/population_decoding.json
uv run python -m src.neural_analysis.population_decoding.run_session new \
  --config /path/to/session/population_decoding.json
```

Dry-run checks small inputs and source availability without binning, fitting,
or creating analysis outputs. IRIG-backed alignment is required; manual alignment
remains deferred. A region's missing coverage does not remove the other region.

The config defaults to 500-ms bins; `bin_width_ms` also supports 100, 50, and 20.
Both standardized units and all nonzero training PCs are always compared, with
fixed L2 logistic `C=1`, no balancing/tuning, and shared deterministic five-fold
behavioral-block assignments. Complete two-second windows are required:

| Target model | Activity relative to current choice |
| --- | --- |
| Context, current choice | `[-1,+1)` s |
| Joint four-class outcome | `[0,+1)` s only |
| Left/right directional choice switches | `[-1,+1)` s |

Outcome class order is left reward, right reward, left no reward, right no reward.
No reward includes incorrect choices. Switches refer to the immediately following
unfiltered trial's chosen port; invalid/manual successors and the terminal trial
are unavailable. They do not mean task-context switches.

Scaling and PCA see only each target/fold's training bins. The two feature sets
can give nearly identical predictions because keeping all nonzero PCs largely
rotates the standardized units without changing isotropic L2 regularization.

## Results and viewing

Shared Hz activity, original row/unit IDs, complete behavior and block folds
are saved once under `processed/population_task_decoding/<id>/preparation.npz`.
Each dated run under `analysis_runs/population_task_decoding_*` contains paired
target NPZs, a readable summary, config/progress/log/performance files, and
copies of the main/batch scripts used. Run the package entry points above;
the copied scripts are provenance snapshots. Changed inputs/code/runtime select
new immutable products. Importing other analyses' count archives is deferred.

Launch the normal webapp:

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata /path/to/session/neural_session.json
```

Choose **Plot view -> Population task decoding**, then a saved run, region,
population features, and target. **Single unit selectivity** remains a separate
dropdown entry with its existing controls. The new view loads only saved data,
including after raw files are unmounted; selector changes perform no analysis.
It displays actual labels, supported OOF probability intervals, class counts,
confusion matrices, scores and fold reasons. Pre-choice outcome tracks are blank.
The export button writes a non-overwriting PNG under that run's `figures/`.

Primary scores pool all required held-out bins: balanced accuracy and log loss.
An incomplete set of OOF predictions makes those scores unavailable. Missing
training classes are reported explicitly; fitting is not retried with new folds.
Outcome scores can reflect side alone, so inspect the within-actual-side
reward/no-reward summaries. For switches inspect positive prevalence and the
always-negative baseline. Full-window switch decoding is an offline readout
including post-choice activity, without prospective/causal/significance claims.

## Resume, rerun, and batch

`new` reuses matching complete runs and reports partial ones. Resume a specific
run to compute missing pairs; completed compatible files are never refitted.
After preparation publication, resume needs no raw files. A changed numerical
code/runtime requires a new run. Corrupt published files are reported rather
than overwritten.

```bash
uv run python -m src.neural_analysis.population_decoding.run_session status \
  --run-directory /path/to/session/analysis_runs/population_task_decoding_RUN
uv run python -m src.neural_analysis.population_decoding.run_session resume \
  --run-directory /path/to/session/analysis_runs/population_task_decoding_RUN
uv run python -m src.neural_analysis.population_decoding.run_session new \
  --config /path/to/session/population_decoding.json --rerun
```

For approved session lists, put one config filename per line, relative to the
list file if desired. Blank/comment lines are ignored. Batch execution uses
spawned session workers with a CPU/session cap and one numerical thread each.
Set `--workers` to respect measured session RAM or storage bandwidth.

```bash
uv run python -m src.neural_analysis.population_decoding.run_batch dry-run \
  --config-list /path/to/configs.txt --workers 2
uv run python -m src.neural_analysis.population_decoding.run_batch new \
  --config-list /path/to/configs.txt --workers 2
```

Synthetic validation precedes one user-designated experimental session and
user inspection. No experimental session has been run for this new analysis.
