"""Project-specific plotting helpers for state-space behavior models."""

import numpy as np
import matplotlib.pyplot as plt


def _normalize_lm_weight_array(weights: np.ndarray) -> np.ndarray:
    """Return LM-HMM weights with shape `(num_states_or_time, obs_dim, input_dim)`.

    Parameters
    ----------
    weights : np.ndarray
        LM-HMM weight array. Common input shapes are `(K, D, M)`, `(K, M)`,
        or `(M,)`.

    Returns
    -------
    np.ndarray
        Weight array with shape `(K, D, M)`.
    """
    normalized = np.asarray(weights, dtype=float)
    if normalized.ndim == 1:
        normalized = normalized[np.newaxis, np.newaxis, :]
    elif normalized.ndim == 2:
        normalized = normalized[:, np.newaxis, :]
    return normalized


def _normalize_lm_bias_array(mus: np.ndarray) -> np.ndarray:
    """Return LM-HMM intercepts with shape `(num_states_or_time, obs_dim)`.

    Parameters
    ----------
    mus : np.ndarray
        LM-HMM intercept array. Common input shapes are `(K, D)`, `(K,)`, or
        scalar.

    Returns
    -------
    np.ndarray
        Intercept array with shape `(K, D)`.
    """
    normalized = np.asarray(mus, dtype=float)
    if normalized.ndim == 0:
        normalized = normalized[np.newaxis, np.newaxis]
    elif normalized.ndim == 1:
        normalized = normalized[:, np.newaxis]
    return normalized


def _get_state_colors(num_states: int) -> list:
    """Return neutral state colors without importing the SSM plotting package.

    Parameters
    ----------
    num_states : int
        Number of hidden states.

    Returns
    -------
    list
        One Matplotlib RGBA color per state.
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    colormap = plt.get_cmap("tab10" if num_states <= 10 else "viridis")
    if num_states == 1:
        return [colormap(0)]
    return [colormap(idx / (num_states - 1)) for idx in range(num_states)]


def add_session_boundary_markers(
    axes,
    boundary_positions: np.ndarray,
    boundary_labels: list[str] | np.ndarray,
    label_location: str = "top",
    label_y: float | None = None,
) -> None:
    """Draw labeled session-boundary markers on one or more axes.

    Parameters
    ----------
    axes
        Matplotlib axis, or iterable/array of axes. Vertical lines are drawn on
        every axis; labels are drawn only on the first axis.
    boundary_positions : np.ndarray
        X-axis positions for session boundaries, shape `(n_boundaries,)`, in
        the same data coordinates as the plotted HMM arrays.
    boundary_labels : list[str] or np.ndarray
        Label for each boundary, shape `(n_boundaries,)`. Each label should
        identify the session that starts after the boundary.
    label_location : {"top", "bottom"}, default="top"
        Axis that receives the boundary labels. `"top"` preserves the original
        behavior; `"bottom"` places labels below the last axis.
    label_y : float or None, default=None
        Label y position in axis coordinates. If None, uses `0.98` for top
        labels and `-0.18` for bottom labels.

    Returns
    -------
    None
        Mutates the supplied axes by adding dashed vertical lines and labels.
    """
    if label_location not in {"top", "bottom"}:
        raise ValueError("label_location must be 'top' or 'bottom'.")

    positions = np.asarray(boundary_positions)
    if boundary_labels is None:
        if positions.size == 0:
            return
        raise ValueError("boundary_labels must be provided for every boundary position.")
    labels = np.asarray(boundary_labels, dtype=object)
    if positions.size != labels.size:
        raise ValueError("boundary_positions and boundary_labels must have the same length.")
    if positions.size == 0:
        return

    axes_array = np.ravel(np.atleast_1d(axes))
    for axis in axes_array:
        for boundary_position in positions:
            axis.axvline(x=boundary_position, color="k", linestyle="--", linewidth=1, alpha=0.6)

    label_axis = axes_array[0] if label_location == "top" else axes_array[-1]
    if label_y is None:
        label_y = 0.98 if label_location == "top" else -0.18
    label_clip_on = label_location == "top"
    for boundary_position, boundary_label in zip(positions, labels):
        label_axis.text(
            boundary_position,
            label_y,
            str(boundary_label),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            transform=label_axis.get_xaxis_transform(),
            clip_on=label_clip_on,
        )


def stack_state_durations(
    inferred_state_list: np.ndarray,
    inferred_durations: np.ndarray,
    num_states: int,
) -> list[np.ndarray]:
    """Group run-length encoded state durations by state index.

    Parameters
    ----------
    inferred_state_list : np.ndarray
        State ids returned by `ssm.util.rle`, shape `(n_runs,)`.
    inferred_durations : np.ndarray
        Run durations returned by `ssm.util.rle`, shape `(n_runs,)`.
    num_states : int
        Number of hidden states in the fitted model.

    Returns
    -------
    list[np.ndarray]
        One duration array per state, length `num_states`.
    """
    return [
        inferred_durations[inferred_state_list == state_idx]
        for state_idx in range(num_states)
    ]


def plot_state_duration_histogram(
    inferred_state_list: np.ndarray,
    inferred_durations: np.ndarray,
    num_states: int,
    state_colors: list,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a state-duration histogram from run-length encoded states.

    Parameters
    ----------
    inferred_state_list : np.ndarray
        State ids returned by `ssm.util.rle`, shape `(n_runs,)`.
    inferred_durations : np.ndarray
        Run durations returned by `ssm.util.rle`, shape `(n_runs,)`.
    num_states : int
        Number of hidden states in the fitted model.
    state_colors : list
        One plotting color per state, length `num_states`.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes containing the duration histogram.
    """
    inferred_durations_stacked = stack_state_durations(
        inferred_state_list,
        inferred_durations,
        num_states,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        inferred_durations_stacked,
        label=['state ' + str(state_idx) for state_idx in range(num_states)],
        color=state_colors,
    )
    ax.set_xlabel('Duration')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title('Histogram of Inferred State Durations')
    return fig, ax


def plot_transition_matrix(gen_trans_mat: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """Plot a state transition matrix without saving it.

    Parameters
    ----------
    gen_trans_mat : np.ndarray
        Transition probability matrix with shape `(num_states, num_states)`.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes containing the transition matrix.
    """
    fig, ax = plt.subplots()
    ax.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    num_states = gen_trans_mat.shape[0]
    for row_idx in range(gen_trans_mat.shape[0]):
        for col_idx in range(gen_trans_mat.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                str(np.around(gen_trans_mat[row_idx, col_idx], decimals=2)),
                ha="center",
                va="center",
                color="k",
                fontsize=12,
            )
    ax.set_xlim(-0.5, num_states - 0.5)
    ax.set_xticks(range(0, num_states), np.arange(num_states) + 1, fontsize=10)
    ax.set_yticks(range(0, num_states), np.arange(num_states) + 1, fontsize=10)
    ax.set_ylim(num_states - 0.5, -0.5)
    ax.set_ylabel("state t", fontsize=15)
    ax.set_xlabel("state t+1", fontsize=15)
    return fig, ax


def plot_block_lm_hmm_state_summary(
    posterior_probs: np.ndarray,
    observations: np.ndarray,
    inputs: np.ndarray,
    hmm_fit,
    colors: list,
    cmap,
    session_lengths: np.ndarray | None = None,
    session_boundary_positions: np.ndarray | None = None,
    session_boundary_labels: list[str] | np.ndarray | None = None,
    line_width: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot block LM-HMM posterior probabilities and observations.

    Parameters
    ----------
    posterior_probs : np.ndarray
        Posterior state probabilities with shape `(n_blocks, num_states)`.
    observations : np.ndarray
        Block observations with shape `(n_blocks, obs_dim)`.
    inputs : np.ndarray
        LM-HMM inputs with shape `(n_blocks, input_dim)`.
    hmm_fit
        Fitted LM-HMM with `transitions.log_Ps`, `observations.mus`, and
        `observations.Wks` attributes.
    colors : list
        One plotting color per state.
    cmap
        Matplotlib colormap aligned to `colors`.
    session_lengths : np.ndarray or None, default=None
        Optional sequence lengths for drawing session-boundary lines.
    session_boundary_positions : np.ndarray or None, default=None
        Optional labeled session-boundary x positions, shape `(n_boundaries,)`.
    session_boundary_labels : list[str] or np.ndarray or None, default=None
        Labels for `session_boundary_positions`, shape `(n_boundaries,)`.
    line_width : float or None, default=None
        Optional linewidth for posterior, observation, bias, and weight traces.
        None preserves the existing single-session plotting defaults.
    figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches. None preserves the existing
        plotting default.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes for posterior probabilities and observations.
    """
    figure_kwargs = {} if figsize is None else {"figsize": figsize}
    fig, (prob_ax, obs_ax) = plt.subplots(2, 1, **figure_kwargs)

    time_bins = len(inputs)
    obs_dim = len(observations[0])
    num_states = hmm_fit.transitions.log_Ps.shape[0]
    posterior_line_width = 2 if line_width is None else line_width
    trace_line_width_kwargs = {} if line_width is None else {"linewidth": line_width}
    for state_idx in range(num_states):
        prob_ax.plot(
            posterior_probs[:, state_idx],
            label="State " + str(state_idx + 1),
            lw=posterior_line_width,
            color=colors[state_idx],
        )

    prob_ax.set_ylim((-0.05, 1.05))
    prob_ax.set_yticks([0, 1], labels=[0, 1], fontsize=15)
    prob_ax.set_ylabel("p(state)", fontsize=15)
    prob_ax.set_xlim(0, time_bins - 1)
    prob_ax.set_xticks([])

    lim = 2 * abs(observations).max()
    ind_state = np.argmax(posterior_probs, axis=1)
    ind_state[np.all(posterior_probs < 0.65, axis=1)] = -1
    cmap.set_under('w')

    biases = _normalize_lm_bias_array(hmm_fit.observations.mus)[ind_state]
    weights = _normalize_lm_weight_array(hmm_fit.observations.Wks)[ind_state]
    for obs_idx in range(obs_dim):
        obs_ax.imshow(
            ind_state[None, :],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=len(colors) - 1,
            extent=(0, time_bins, -lim * obs_dim, lim),
            alpha=0.5,
        )
        obs_ax.plot(
            observations[:, obs_idx] - lim * obs_idx,
            '-k',
            label='obs' * (obs_idx == 0),
            **trace_line_width_kwargs,
        )
    #     obs_ax.plot(
    #         biases[:, obs_idx] - lim * obs_idx,
    #         ':k',
    #         label='bias' * (obs_idx == 0),
    #         **trace_line_width_kwargs,
    #     )
    #     obs_ax.plot(
    #         weights[:, obs_idx, 0] - lim * obs_idx,
    #         '--k',
    #         label='weight' * (obs_idx == 0),
    #         **trace_line_width_kwargs,
    #     )

    obs_ax.set_xlim(0, time_bins - 1)
    obs_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, prop={'size': 10})
    obs_ax.set_xlabel("Context changes")
    obs_ax.set_ylim(0, abs(observations).max())
    obs_ax.set_yticks([0, abs(observations).max()])

    if session_lengths is not None:
        splits = np.cumsum(session_lengths)
        for split_idx in range(splits.shape[0] - 1):
            prob_ax.axvline(x=splits[split_idx], color='k', linestyle='--')
            obs_ax.axvline(x=splits[split_idx], color='k', linestyle='--')

    if session_boundary_positions is not None:
        add_session_boundary_markers(
            axes=(prob_ax, obs_ax),
            boundary_positions=session_boundary_positions,
            boundary_labels=session_boundary_labels,
        )

    fig.tight_layout()
    return fig, (prob_ax, obs_ax)


def plot_block_lm_hmm_presentation_summary(
    posterior_probs: np.ndarray,
    observations: np.ndarray,
    inputs: np.ndarray,
    hmm_fit,
    colors: list,
    cmap,
    session_lengths: np.ndarray | None = None,
    predictor_labels: list[str] | None = None,
    session_boundary_positions: np.ndarray | None = None,
    session_boundary_labels: list[str] | np.ndarray | None = None,
    line_width: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot presentation-style block LM-HMM state probabilities and weights.

    Parameters
    ----------
    posterior_probs : np.ndarray
        Posterior state probabilities with shape `(n_blocks, num_states)`.
    observations : np.ndarray
        Block observations with shape `(n_blocks, obs_dim)`.
    inputs : np.ndarray
        LM-HMM inputs with shape `(n_blocks, input_dim)`.
    hmm_fit
        Fitted LM-HMM with `transitions.log_Ps`, `observations.mus`, and
        `observations.Wks` attributes.
    colors : list
        One plotting color per state.
    cmap
        Matplotlib colormap aligned to `colors`.
    session_lengths : np.ndarray or None, default=None
        Optional sequence lengths for drawing session-boundary lines.
    predictor_labels : list[str] or None, default=None
        Labels for the input dimensions.
    session_boundary_positions : np.ndarray or None, default=None
        Optional labeled session-boundary x positions, shape `(n_boundaries,)`.
    session_boundary_labels : list[str] or np.ndarray or None, default=None
        Labels for `session_boundary_positions`, shape `(n_boundaries,)`.
    line_width : float or None, default=None
        Optional linewidth for posterior, observation, bias, and weight traces.
        None preserves the existing single-session plotting defaults.
    figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches. None preserves the existing
        plotting default.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes for posterior probabilities and observations.
    """
    figure_kwargs = {} if figsize is None else {"figsize": figsize}
    fig, (prob_ax, obs_ax) = plt.subplots(2, 1, **figure_kwargs)

    time_bins = len(inputs)
    obs_dim = len(observations[0])
    num_states = hmm_fit.transitions.log_Ps.shape[0]
    input_dim = len(inputs[0])
    if predictor_labels is None:
        predictor_labels = [f"input-{idx}" for idx in range(input_dim)]

    posterior_line_width = 2 if line_width is None else line_width
    trace_line_width_kwargs = {} if line_width is None else {"linewidth": line_width}
    for state_idx in range(num_states):
        prob_ax.plot(
            posterior_probs[:, state_idx],
            label="State " + str(state_idx + 1),
            lw=posterior_line_width,
            color=colors[state_idx],
        )

    prob_ax.set_ylim((-0.05, 1.05))
    prob_ax.set_yticks([0, 1], labels=[0, 1], fontsize=15)
    prob_ax.set_ylabel("p(strategy)", fontsize=15)
    prob_ax.set_xlim(0, time_bins - 1)
    prob_ax.set_xticks([])

    lim = 2 * abs(observations).max()
    ind_state = np.argmax(posterior_probs, axis=1)
    ind_state[np.all(posterior_probs < 0.55, axis=1)] = -1
    cmap.set_under('w')

    biases = _normalize_lm_bias_array(hmm_fit.observations.mus)[ind_state]
    weights = _normalize_lm_weight_array(hmm_fit.observations.Wks)[ind_state]
    for obs_idx in range(obs_dim):
        obs_ax.imshow(
            ind_state[None, :],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=len(colors) - 1,
            extent=(0, time_bins, -lim * obs_dim, lim),
            alpha=0.5,
        )
        obs_ax.plot(
            observations[:, 0] - lim * obs_idx,
            '-k',
            label='obs',
            **trace_line_width_kwargs,
        )
    #     obs_ax.plot(
    #         biases[:, 0] - lim,
    #         ':k',
    #         label='bias',
    #         **trace_line_width_kwargs,
    #     )

    # for input_idx in range(input_dim):
    #     obs_ax.plot(
    #         weights[:, 0, input_idx] - lim * input_idx,
    #         '--k',
    #         label=predictor_labels[input_idx],
    #         **trace_line_width_kwargs,
    #     )

    obs_ax.set_xlim(0, time_bins - 1)
    legend = obs_ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        prop={'size': 6},
    )
    for line in legend.get_lines():
        line.set_linewidth(.5)

    obs_ax.set_xlabel("Context changes")
    obs_ax.set_ylim(0, abs(observations).max())
    obs_ax.set_yticks([0, abs(observations).max()])
    obs_ax.tick_params(axis='y', labelsize=12)
    obs_ax.set_ylabel("Trials to switch", fontsize=16)

    if session_lengths is not None:
        splits = np.cumsum(session_lengths)
        for split_idx in range(splits.shape[0] - 1):
            prob_ax.axvline(x=splits[split_idx], color='k', linestyle='--')
            obs_ax.axvline(x=splits[split_idx], color='k', linestyle='--')

    if session_boundary_positions is not None:
        add_session_boundary_markers(
            axes=(prob_ax, obs_ax),
            boundary_positions=session_boundary_positions,
            boundary_labels=session_boundary_labels,
        )

    fig.tight_layout()
    return fig, (prob_ax, obs_ax)


def plot_block_lm_hmm_weight_comparison(
    weight_dicts: list[dict],
    colors: list | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot block LM-HMM weights from multiple fits without saving.

    Parameters
    ----------
    weight_dicts : list[dict]
        Weight dictionaries. Each dictionary must contain `weights`, `mus`,
        and `label`. Weights use shape `(num_states, obs_dim, input_dim)`;
        intercepts use shape `(num_states, obs_dim)`.
    colors : list or None, default=None
        Optional state colors. If None, colors are generated from the number of
        states.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes array.
    """
    num_comp = len(weight_dicts)
    weights = _normalize_lm_weight_array(weight_dicts[0]['weights'])
    mus = _normalize_lm_bias_array(weight_dicts[0]['mus'])
    num_states, obs_dim, input_dim = weights.shape
    if colors is None:
        colors = _get_state_colors(num_states)
    style_set = ['-', '--', ':']
    marker_set = ['o', 's', 'd']

    fig, axes = plt.subplots(1, obs_dim, figsize=(obs_dim * 4, 5), dpi=80, facecolor='w', edgecolor='k')
    axes_array = np.atleast_1d(axes)
    for obs_idx, ax in enumerate(axes_array):
        for comp_idx in range(num_comp):
            comp_weights = _normalize_lm_weight_array(weight_dicts[comp_idx]['weights'])
            comp_mus = _normalize_lm_bias_array(weight_dicts[comp_idx]['mus'])
            for state_idx in range(num_states):
                ax.plot(
                    np.arange(input_dim + 1),
                    np.append(comp_weights[state_idx][obs_idx], comp_mus[state_idx][obs_idx]),
                    marker=marker_set[comp_idx % len(marker_set)],
                    color=colors[state_idx],
                    linestyle=style_set[comp_idx % len(style_set)],
                    lw=1.5,
                    label=(weight_dicts[comp_idx]['label']) * (state_idx == 0),
                )
        ax.set_yticks(ax.get_yticks(), labels=[str(tick) for tick in ax.get_yticks()], fontsize=10)
        ax.set_xlabel("covariate", fontsize=15)
        ax.set_ylabel("weights", fontsize=15)
        ax.set_xticks(
            np.arange(input_dim + 1),
            np.append(["input-" + str(input_idx) for input_idx in range(input_dim)], 'bias'),
            fontsize=12,
            rotation=45,
        )
        ax.axhline(y=0, color="k", alpha=0.5, ls="--")
        ax.set_title("Weights for obs dim " + str(obs_idx), fontsize=15)
        if obs_idx == 0:
            ax.legend()
            ax.set_ylabel("LM weight", fontsize=15)
    fig.tight_layout()
    return fig, axes_array


def plot_block_lm_hmm_weights(
    weight_dict: dict,
    colors: list,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot one block LM-HMM fit's state weights without saving.

    Parameters
    ----------
    weight_dict : dict
        Weight dictionary with `weights`, `mus`, `label`, and `weight_labels`.
        Weights use shape `(num_states, obs_dim, input_dim)`; intercepts use
        shape `(num_states, obs_dim)`.
    colors : list
        One plotting color per state.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes array.
    """
    weights = _normalize_lm_weight_array(weight_dict['weights'])
    mus = _normalize_lm_bias_array(weight_dict['mus'])
    num_states, obs_dim, input_dim = weights.shape
    predictor_labels = weight_dict['weight_labels']

    fig, axes = plt.subplots(1, obs_dim, figsize=(obs_dim * 4, 5), dpi=80, facecolor='w', edgecolor='k')
    axes_array = np.atleast_1d(axes)
    for obs_idx, ax in enumerate(axes_array):
        for state_idx in range(num_states):
            ax.plot(
                np.arange(input_dim + 1),
                np.append(weights[state_idx][obs_idx], mus[state_idx][obs_idx]),
                marker='o',
                color=colors[state_idx],
                linestyle='-',
                lw=1.5,
                label=(weight_dict['label']) * (state_idx == 0),
            )

        ax.set_yticks(ax.get_yticks(), labels=[str(tick) for tick in ax.get_yticks()], fontsize=10)
        ax.set_xlabel("covariate", fontsize=15)
        ax.set_ylabel("weights", fontsize=15)
        ax.set_xticks(np.arange(input_dim + 1), np.append(predictor_labels, 'bias'), fontsize=12, rotation=45)
        ax.axhline(y=0, color="k", alpha=0.5, ls="--")
        if obs_idx == 0:
            ax.set_ylabel("LM weight", fontsize=15)

    fig.tight_layout()
    fig.suptitle("Model weights")
    return fig, axes_array


def plot_trial_glm_hmm_weights(
    weight_dicts: list[dict],
    colors: list | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot trial GLM-HMM weights without saving.

    Parameters
    ----------
    weight_dicts : list[dict]
        Weight dictionaries. Each dictionary must contain `weights`, `label`,
        and `weight_labels`. Weights use shape
        `(num_states, 1, input_dim)`.
    colors : list or None, default=None
        Optional state colors. If None, colors are generated from the number of
        states.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes.
    """
    first_weights = np.asarray(weight_dicts[0]['weights'], dtype=float)
    if first_weights.ndim == 2:
        first_weights = first_weights[:, np.newaxis, :]
    num_states = first_weights.shape[0]
    input_dim = first_weights.shape[-1]
    if colors is None:
        colors = _get_state_colors(num_states)
    style_set = ['-', '--', ':']
    marker_set = ['o', 's', 'd']

    fig, ax = plt.subplots(figsize=(4, 5), dpi=80, facecolor='w', edgecolor='k')
    for comp_idx, weight_dict in enumerate(weight_dicts):
        weights = np.asarray(weight_dict['weights'], dtype=float)
        if weights.ndim == 2:
            weights = weights[:, np.newaxis, :]
        if num_states == 1:
            ax.plot(
                np.arange(input_dim),
                np.squeeze(weights),
                marker=marker_set[comp_idx % len(marker_set)],
                color=colors[0],
                linestyle=style_set[comp_idx % len(style_set)],
                lw=1.5,
                label=weight_dict['label'],
            )
        else:
            squeezed_weights = np.squeeze(weights)
            for state_idx in range(num_states):
                ax.plot(
                    np.arange(input_dim),
                    squeezed_weights[state_idx],
                    marker=marker_set[comp_idx % len(marker_set)],
                    color=colors[state_idx],
                    linestyle=style_set[comp_idx % len(style_set)],
                    lw=1.5,
                    label=(weight_dict['label']) * (state_idx == 0),
                )

        ax.set_xticks(
            np.arange(len(weight_dict['weight_labels'])),
            weight_dict['weight_labels'],
            fontsize=12,
            rotation=45,
        )

    ax.set_yticks(ax.get_yticks(), labels=[str(tick) for tick in ax.get_yticks()], fontsize=10)
    ax.set_ylabel("GLIM weight", fontsize=15)
    ax.set_xlabel("covariate", fontsize=15)
    ax.axhline(y=0, color="k", alpha=0.5, ls="--")
    ax.legend()
    ax.set_title("Model weights", fontsize=15)
    fig.tight_layout()
    return fig, ax


def plot_trial_glm_hmm_state_summary(
    posterior_probs: np.ndarray,
    observations: np.ndarray,
    inputs: np.ndarray,
    hmm_fit,
    colors: list,
    cmap,
    session_lengths: np.ndarray | None = None,
    line_width: float | None = None,
    figsize: tuple[float, float] | None = None,
    session_boundary_positions: np.ndarray | None = None,
    session_boundary_labels: list[str] | np.ndarray | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Plot trial GLM-HMM posterior probabilities, inputs, and observations.

    Parameters
    ----------
    posterior_probs : np.ndarray
        Posterior state probabilities with shape `(n_trials, num_states)`.
    observations : np.ndarray
        Trial observations with shape `(n_trials, obs_dim)`.
    inputs : np.ndarray
        GLM-HMM inputs with shape `(n_trials, input_dim)`.
    hmm_fit
        Fitted trial GLM-HMM with `transitions.log_Ps` and `observations.Wk`.
        `Wk[:, :, -1]` is interpreted as the bias and preceding columns are
        interpreted as non-bias GLM weights.
    colors : list
        One plotting color per state.
    cmap
        Matplotlib colormap aligned to `colors`.
    session_lengths : np.ndarray or None, default=None
        Optional sequence lengths for drawing session-boundary lines.
    line_width : float or None, default=None
        Optional linewidth for posterior, input, observation, bias, and weight
        traces. None preserves the existing plotting defaults.
    figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches. None preserves the existing
        plotting default.
    session_boundary_positions : np.ndarray or None, default=None
        Optional labeled session-boundary x positions, shape `(n_boundaries,)`.
    session_boundary_labels : list[str] or np.ndarray or None, default=None
        Labels for `session_boundary_positions`, shape `(n_boundaries,)`.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]
        Figure and axes for posterior probabilities, inputs, and observations.
    """
    figure_kwargs = {} if figsize is None else {"figsize": figsize}
    fig, (prob_ax, input_ax, obs_ax) = plt.subplots(
        3,
        1,
        gridspec_kw={'height_ratios': [1, 1, 3]},
        **figure_kwargs,
    )

    time_bins = len(inputs)
    obs_dim = len(observations[0])
    num_states = hmm_fit.transitions.log_Ps.shape[0]
    input_dim = len(inputs[0])
    posterior_line_width = 2 if line_width is None else line_width
    trace_line_width_kwargs = {} if line_width is None else {"linewidth": line_width}
    for state_idx in range(num_states):
        prob_ax.plot(
            posterior_probs[:, state_idx],
            label="State " + str(state_idx + 1),
            lw=posterior_line_width,
            color=colors[state_idx],
        )

    prob_ax.set_ylim((-0.05, 1.05))
    prob_ax.set_yticks([0, 1], labels=[0, 1], fontsize=15)
    prob_ax.set_ylabel("p(state)", fontsize=15)
    prob_ax.set_xlim(0, time_bins - 1)
    prob_ax.set_xticks([])

    lim_input = 1.1 * abs(inputs).max()
    for input_idx in range(input_dim):
        input_ax.plot(
            inputs[:, input_idx] - lim_input * input_idx,
            label='inpt dim ' + str(input_idx),
            **trace_line_width_kwargs,
        )

    input_ax.set_xticks([])
    input_ax.set_xlim(0, time_bins)
    input_ax.set_yticks(
        -np.arange(input_dim) * lim_input,
        ["$x_{{ {} }}$".format(idx + 1) for idx in range(input_dim)],
    )
    input_ax.set_title('Input')
    input_ax.legend(fancybox=False, fontsize=10)

    lim = 2 * abs(observations).max()
    ind_state = np.argmax(posterior_probs, axis=1)
    ind_state[np.all(posterior_probs < 0.8, axis=1)] = -1
    cmap.set_under('w')

    biases = hmm_fit.observations.Wk[:, :, -1][ind_state]
    weights = hmm_fit.observations.Wk[:, :, :-1][ind_state]
    for obs_idx in range(obs_dim):
        obs_ax.imshow(
            ind_state[None, :],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=len(colors) - 1,
            extent=(0, time_bins, -lim * obs_dim, lim),
            alpha=0.5,
        )
        obs_ax.plot(
            observations[:, obs_idx] - lim * obs_idx,
            '-k',
            label='obs' * (obs_idx == 0),
            **trace_line_width_kwargs,
        )
        obs_ax.plot(
            biases[:, obs_idx] - lim * obs_idx,
            ':k',
            label='bias' * (obs_idx == 0),
            **trace_line_width_kwargs,
        )
        obs_ax.plot(
            weights[:, obs_idx] - lim * obs_idx,
            '--k',
            label='weight' * (obs_idx == 0),
            **trace_line_width_kwargs,
        )
    obs_ax.set_xlim(0, time_bins - 1)
    obs_ax.set_yticks([])
    obs_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    obs_ax.set_xlabel("Context changes")
    obs_ax.set_ylim(0 - .05, abs(observations).max() + .05)

    if session_lengths is not None:
        splits = np.cumsum(session_lengths)
        for split_idx in range(splits.shape[0] - 1):
            prob_ax.axvline(x=splits[split_idx], color='k', linestyle='--')
            input_ax.axvline(x=splits[split_idx], color='k', linestyle='--')
            obs_ax.axvline(x=splits[split_idx], color='k', linestyle='--')

    if session_boundary_positions is not None:
        add_session_boundary_markers(
            axes=(prob_ax, input_ax, obs_ax),
            boundary_positions=session_boundary_positions,
            boundary_labels=session_boundary_labels,
            label_location="bottom",
        )

    fig.tight_layout()
    if session_boundary_positions is not None and np.asarray(session_boundary_positions).size > 0:
        fig.subplots_adjust(bottom=0.28)
    return fig, (prob_ax, input_ax, obs_ax)


def plot_trial_glm_hmm_block_state_comparison(
    block_states: np.ndarray,
    trial_posterior_probs: np.ndarray,
    observations: np.ndarray,
    inputs: np.ndarray,
    hmm_fit,
    colors: list,
    cmap,
    session_lengths: np.ndarray | None = None,
    line_width: float | None = None,
    figsize: tuple[float, float] | None = None,
    session_boundary_positions: np.ndarray | None = None,
    session_boundary_labels: list[str] | np.ndarray | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Plot inherited block states beside trial GLM-HMM inferred states.

    Parameters
    ----------
    block_states : np.ndarray
        Inherited block state ids with shape `(n_trials, 1)` or `(n_trials,)`.
    trial_posterior_probs : np.ndarray
        Trial GLM-HMM posterior probabilities with shape
        `(n_trials, num_states)`.
    observations : np.ndarray
        Trial observations with shape `(n_trials, obs_dim)`.
    inputs : np.ndarray
        GLM-HMM inputs with shape `(n_trials, input_dim)`.
    hmm_fit
        Fitted trial GLM-HMM with `transitions.log_Ps`.
    colors : list
        One plotting color per state.
    cmap
        Matplotlib colormap aligned to `colors`.
    session_lengths : np.ndarray or None, default=None
        Optional sequence lengths for drawing session-boundary lines.
    line_width : float or None, default=None
        Optional linewidth for posterior and observation traces. None preserves
        the existing plotting defaults.
    figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches. None preserves the existing
        plotting default.
    session_boundary_positions : np.ndarray or None, default=None
        Optional labeled session-boundary x positions, shape `(n_boundaries,)`.
    session_boundary_labels : list[str] or np.ndarray or None, default=None
        Labels for `session_boundary_positions`, shape `(n_boundaries,)`.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]
        Figure and axes for block labels, trial labels, and posterior
        probabilities.
    """
    figure_kwargs = {} if figsize is None else {"figsize": figsize}
    fig, (block_ax, trial_ax, prob_ax) = plt.subplots(3, 1, **figure_kwargs)
    lim = 2 * abs(observations).max()
    obs_dim = len(observations[0])
    time_bins = len(inputs)
    cmap.set_under('w')
    posterior_line_width = 2 if line_width is None else line_width
    trace_line_width_kwargs = {} if line_width is None else {"linewidth": line_width}

    block_states = np.asarray(block_states).reshape(1, -1)
    for obs_idx in range(obs_dim):
        block_ax.imshow(
            block_states,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=len(colors) - 1,
            extent=(0, time_bins, -lim * obs_dim, lim),
            alpha=0.5,
        )
        block_ax.plot(
            observations[:, obs_idx] - lim * obs_idx,
            '-k',
            label='obs' * (obs_idx == 0),
            **trace_line_width_kwargs,
        )
    block_ax.set_xlim(0, time_bins - 1)
    block_ax.set_yticks([])
    block_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    block_ax.set_xlabel("Block context changes")
    block_ax.set_ylim(0 - .05, abs(observations).max() + .05)

    ind_state = np.argmax(trial_posterior_probs, axis=1)
    ind_state[np.all(trial_posterior_probs < 0.8, axis=1)] = -1
    cmap.set_under('w')

    for obs_idx in range(obs_dim):
        trial_ax.imshow(
            ind_state[None, :],
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=len(colors) - 1,
            extent=(0, time_bins, -lim * obs_dim, lim),
            alpha=0.5,
        )
        trial_ax.plot(
            observations[:, obs_idx] - lim * obs_idx,
            '-k',
            label='obs' * (obs_idx == 0),
            **trace_line_width_kwargs,
        )
    trial_ax.set_xlim(0, time_bins - 1)
    trial_ax.set_yticks([])
    trial_ax.set_xlabel("Trial context changes")
    trial_ax.set_ylim(0 - .05, abs(observations).max() + .05)

    if session_lengths is not None:
        splits = np.cumsum(session_lengths)
        for split_idx in range(splits.shape[0] - 1):
            block_ax.axvline(x=splits[split_idx], color='k', linestyle='--')
            trial_ax.axvline(x=splits[split_idx], color='k', linestyle='--')

    num_states = hmm_fit.transitions.log_Ps.shape[0]
    for state_idx in range(num_states):
        prob_ax.plot(
            trial_posterior_probs[:, state_idx],
            label="State " + str(state_idx + 1),
            lw=posterior_line_width,
            color=colors[state_idx],
        )

    prob_ax.set_ylim((-0.05, 1.05))
    prob_ax.set_yticks([0, 1], labels=[0, 1], fontsize=15)
    prob_ax.set_ylabel("p(state)", fontsize=15)
    prob_ax.set_xlim(0, time_bins - 1)
    prob_ax.set_xticks([])

    if session_boundary_positions is not None:
        add_session_boundary_markers(
            axes=(block_ax, trial_ax, prob_ax),
            boundary_positions=session_boundary_positions,
            boundary_labels=session_boundary_labels,
            label_location="bottom",
        )

    fig.tight_layout()
    if session_boundary_positions is not None and np.asarray(session_boundary_positions).size > 0:
        fig.subplots_adjust(bottom=0.28)
    return fig, (block_ax, trial_ax, prob_ax)
