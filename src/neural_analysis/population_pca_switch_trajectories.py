"""Compatibility facade retaining population-switch plots until R3."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.neural_analysis.population.switch_trajectories import (
    SWITCH_PRE_FILTER_ALL,
    SWITCH_PRE_FILTER_CORRECT_REWARDED,
    SWITCH_PRE_FILTER_OMISSION,
    SWITCH_PRE_FILTER_OPTIONS,
    SWITCH_TYPE_ORDER,
    SWITCH_TYPE_TITLES,
    SWITCH_DIRECTION_ORDER,
    SWITCH_DIRECTION_TITLES,
    SWITCH_EVENT_COLUMNS,
    SWITCH_TRAJECTORY_COLUMNS,
    select_valid_choice_trial_indices,
    select_choice_switch_events,
    extract_switch_event_pca_trajectories,
    summarize_switch_event_counts,
)
from src.neural_analysis.population.plotting import (
    _compute_shared_axis_limits,
    _draw_directed_segments,
    _padded_limits,
    plot_switch_event_pca_trajectories,
)
