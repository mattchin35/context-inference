"""Shared plotting helpers for behavior analysis modules."""

import numpy as np
import matplotlib.pyplot as plt
from ssm.plots import gradient_cmap, white_to_color_cmap


def sample_colormap(cmap, n_colors: int, low: float = 0.15, high: float = 0.85) -> list:
    """Sample a readable range from a Matplotlib colormap.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Colormap used to generate colors.
    n_colors : int
        Number of colors to sample.
    low : float, default=0.15
        Lower colormap coordinate, unitless and in `[0, 1]`.
    high : float, default=0.85
        Upper colormap coordinate, unitless and in `[0, 1]`.

    Returns
    -------
    list
        RGBA color tuples with shape `(n_colors,)`.
    """
    if n_colors == 0:
        return []
    if n_colors == 1:
        return [cmap((low + high) / 2)]
    return list(cmap(np.linspace(low, high, n_colors)))


def build_state_colormap(state_colors: list):
    """Build a state-label colormap from explicit state colors.

    Parameters
    ----------
    state_colors : list
        One color per hidden state, shape `(num_states,)`.

    Returns
    -------
    matplotlib.colors.Colormap
        Matplotlib-compatible colormap for state labels.
    """
    if len(state_colors) == 0:
        raise ValueError("state_colors must contain at least one color.")
    if len(state_colors) == 1:
        return white_to_color_cmap(state_colors[0])
    return gradient_cmap(state_colors)


def get_state_colors(num_states: int) -> list:
    """Return neutral categorical colors for hidden-state plots.

    Parameters
    ----------
    num_states : int
        Number of hidden states that need colors.

    Returns
    -------
    list
        RGBA color tuples with shape `(num_states,)`.
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    if num_states <= 10:
        return [plt.cm.tab10(i) for i in range(num_states)]
    if num_states <= 20:
        return [plt.cm.tab20(i) for i in range(num_states)]
    return list(plt.cm.hsv(np.linspace(0, 1, num_states, endpoint=False)))


def build_presentation_colors(primary_predictor_weights: np.ndarray) -> tuple[list, object]:
    """Build semantic RL/inference colors for block-model presentation plots.

    Parameters
    ----------
    primary_predictor_weights : np.ndarray
        One scalar predictor weight per hidden state, shape `(num_states,)`.
        Weights below 0.5 are treated as inference-like and weights at or above
        0.5 are treated as RL-like for presentation coloring.

    Returns
    -------
    tuple[list, object]
        - present_colors: one color per hidden state, shape `(num_states,)`
        - present_cmap: Matplotlib-compatible colormap for state labels
    """
    state_weights = np.asarray(primary_predictor_weights, dtype=float).reshape(-1)
    if state_weights.size == 0:
        raise ValueError("primary_predictor_weights must contain at least one state.")

    present_colors = np.empty(state_weights.size, dtype=object)
    ix_inf = state_weights < 0.5
    ix_rl = state_weights >= 0.5

    for state_idx, color in zip(
        np.flatnonzero(ix_inf),
        sample_colormap(plt.cm.winter, int(np.sum(ix_inf))),
    ):
        present_colors[state_idx] = color
    for state_idx, color in zip(
        np.flatnonzero(ix_rl),
        sample_colormap(plt.cm.autumn, int(np.sum(ix_rl))),
    ):
        present_colors[state_idx] = color

    present_colors = list(present_colors)
    present_cmap = build_state_colormap(present_colors)
    return present_colors, present_cmap
