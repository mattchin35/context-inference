# Project goals

This project focuses on two complementary modeling questions using behavior-state labels:

1. **Block transition behavior:** model how reward history relates to behavior at context/block transitions.
2. **Trial-to-trial choice behavior:** model choices on individual trials from recent history and latent/state-related predictors.

## Data expectations

- Input data are **CSV files stored outside this repository**.
- Current scripts load files from absolute local paths (for example, under a local `contextProjectData/.../processed/` tree).
- Typical inputs include:
  - `*_augmented_trials.csv` (trial-level features)
  - `*_block_performance.csv` (block-level summaries)

To run these scripts on your system, update the CSV paths in the scripts to match your local directory structure.

## Current analysis workflow

The repository currently contains exploratory and modeling scripts:

- `session_behavior_analysis.R` – single-session trial-level cleaning, exploratory visualization, and random forest modeling of action/choice-related outcomes.
- `session_behavior_analysis_mini.R` – reduced trial-level analysis variant.
- `multisession_block_analysis.R` – multi-session block-level aggregation and modeling for transition performance.

Common workflow patterns in these scripts:

1. Import CSV data.
2. Clean missing-like values (`"None"`, `"NULL"`, etc.) and drop incomplete rows for key variables.
3. Cast relevant columns to numeric/factor types.
4. Use **Flexplot** for exploratory visualization.
5. Fit models (e.g., linear/mixed models and random forest variants) to probe relationships among reward history, inferred strategies/behavior states, and performance.

## R packages used

These scripts use a mix of tidy data, visualization, mixed-model, and tree-based modeling libraries, including:

- `tidyverse`
- `dplyr`
- `ggplot2`
- `flexplot`
- `cowplot`
- `lme4`
- `randomForestSRC`
- `varPro`

Install as needed, for example:

```r
install.packages(c(
  "tidyverse", "dplyr", "ggplot2", "flexplot", "cowplot", "lme4", "randomForestSRC", "varPro"
))
```

## Notes

- This repository is currently organized as analysis scripts (not yet packaged as an R package).
- Some code is exploratory and may include alternative/commented model specifications.
- Because data live outside the repo, reproducibility depends on consistent CSV schema and local path configuration.

## Reproducible SAC TTS workflow

The approved analysis and figure requirements are in
[`docs/SAC_behavior_prep.md`](../../docs/SAC_behavior_prep.md). This workflow fits
each session with `TTS ~ previous rewards` for the exemplar-compatible summary
and `TTS ~ previous rewards * side` for side differences. The cohort step also
compares the four specified mixed models using ML, AIC/BIC and fit diagnostics.

Run the following commands from the repository root, with `glmnet`, `lme4` and
`ggplot2` available in R. No packages are installed by these scripts. On the
current machine, R 4.6 can use the existing libraries with:

```bash
export R_LIBS_USER=/home/matt/R/x86_64-pc-linux-gnu-library/4.6:/home/matt/R/x86_64-pc-linux-gnu-library/4.5
```

Inspect the designated CT024 session plan, then compute it:

```bash
Rscript --vanilla src/Rcode/run_tts_session.R --dry-run
Rscript --vanilla src/Rcode/run_tts_session.R
```

The default input is `CT024_20260609_latent_inference`. Override it with
`--input-csv PATH`; optionally pass `--output-root PATH`, `--seed INTEGER` or
`--rerun`. Default outputs are dated directories under the input CSV's
`processed/sac_tts/` folder. A matching completed computation is reused;
changed data, source code, fitting parameters or analysis version create a new
result. Previous artifacts and input CSVs are not overwritten.

For a cohort, inspect the selected sessions before computing:

```bash
Rscript --vanilla src/Rcode/run_tts_batch.R --dry-run
```

The configured mice are CT016, CT017 and CT019-CT025. Discovery uses primary
processed latent-inference block tables and ignores the two known HMM-derived
table variants. `--dates 2026-06-09,2026-06-11` selects explicit session dates;
`--workers INTEGER` overrides the available-core default. Individual results
stay beside their session data; cohort results default to
`cross_mouse_analysis/sac_tts/`. The initial full cohort run is deferred until
the user has inspected and approved the designated single-session outputs.

For use from an R editor, source `run_tts_batch.R` to load functions without
executing analyses. `run_tts_session()` and `run_tts_batch()` accept explicit
paths and fitting parameters; their header comments document the contracts.
`load_tts_results()` returns the complete typed result list from `result.rds`.
CSV exports, CV folds/curves, logs, script snapshots and `SUMMARY.md` support
inspection. Residual SD is the in-sample sample SD in trials, not held-out error.

Core modules are `tts_data.R`, `tts_modeling.R`, `tts_results.R` and the existing
`block_residual_modeling.R`. Figure rendering is a separate step and must load
saved computations rather than fit models again.

Render a saved result using the directory printed by the computation script:

```bash
Rscript --vanilla src/Rcode/plot_tts_results.R --result-dir "/absolute/path/to/saved/run"
```

Optional plotting arguments are `--figures F4,F8,F9`, `--session-id ID`,
`--population-model M2` and `--output-dir NEW_PATH`. By default each render gets
a new dated subdirectory inside the saved run. It contains PNGs, source CSVs,
caption text, plotting-script snapshots and `figure_manifest.csv`. The manifest
reports unavailable figures explicitly; F2/F3 are skipped for a single session
because no population fit is present. The static 3D parameter view is accompanied
by all pairwise projections. All plots use opaque white backgrounds.

From R, source `plot_tts_results.R` and call `plot_tts_results(result_dir)` to
obtain the same manifest. Rendering does not fit models or overwrite previous
figures. The initial implementation uses ggplot2 and base graphics; Flexplot is
not required to produce these exports.
