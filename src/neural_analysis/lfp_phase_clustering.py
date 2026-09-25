"""Legacy phase API with plotting retained outside the canonical science module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis.lfp.phase import (
    ANALYSIS_VERSION,
    PHASE_TENSOR_AXIS_ORDER,
    RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS,
    ContinuousProcessingBlock,
    PhaseClusteringResult,
    PhaseTrialTensor,
    SingleTrialRelativePhaseResult,
    WaveletCoefficientResult,
    WaveletPhaseResult,
    WithinTrialPLVResult,
    combine_phase_trial_tensors,
    compute_ispc,
    compute_itpc,
    compute_plv_window_samples,
    compute_single_trial_relative_phase,
    compute_site_phase_trial_tensor,
    compute_wavelet_coefficients,
    compute_wavelet_phase,
    compute_within_trial_plv,
    make_phase_condition_masks,
    make_phase_trial_tensor,
    make_relative_phase_display_mask,
    make_relative_phase_support_mask,
    normalize_wavelet_phase,
    plan_continuous_processing_blocks,
    save_phase_clustering_result,
    save_single_trial_phase_analysis_result,
    save_single_trial_relative_phase_result,
    select_tensor_trial_mask,
)
from src.neural_analysis.lfp.plotting import plot_phase_clustering
