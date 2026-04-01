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
