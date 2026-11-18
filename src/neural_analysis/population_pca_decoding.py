"""Compatibility facade retaining population-decoding plots until R3."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.population.decoding import (
    PCA_DECODING_MODE_EXPLORATORY,
    PCA_DECODING_MODE_RIGOROUS,
    PCA_DECODING_MODE_OPTIONS,
    PCA_DECODING_BASE_CONDITIONS,
    PCA_DECODING_WINDOWS,
    PCA_DECODING_BIN_SIZE_S,
    PCA_DECODING_WINDOW,
    PCA_DECODING_DISPLAY_COLUMNS,
    PCA_SCORE_SUMMARY_COLUMNS,
    PCA_RAW_SCORE_COLUMNS,
    build_choice_aligned_rate_tensor,
    select_pca_decoding_trial_indices,
    build_valid_base_condition_masks,
    build_exploratory_pca_decoder_trial_bins,
    run_exploratory_pca_choice_decoding,
    run_rigorous_pca_choice_decoding,
    cv_decodeability_score_with_pca_pipeline,
    summarize_pca_decoding_results,
    summarize_pca_scores_by_condition_and_target,
    extract_pca_score_points_by_condition_and_target,
    format_pca_score_target_label,
)
from src.neural_analysis.population.plotting import (
    plot_average_pca_scores_by_condition_and_target,
    plot_pca_decoding_pre_post_scores,
)
