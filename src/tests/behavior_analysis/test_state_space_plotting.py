import matplotlib.pyplot as plt
import numpy as np

from src.behavior_analysis import state_space_plotting as ssplot


def test_add_session_boundary_markers_draws_lines_and_top_axis_labels():
    """Session boundary helper should mark every axis and label each boundary.

    Inputs
    ------
    None
        Uses two blank axes, two boundary positions, and two date labels.

    Returns
    -------
    None
        Asserts that every axis receives boundary lines and only the top axis
        receives the corresponding labels.
    """
    fig, axes = plt.subplots(2, 1)

    ssplot.add_session_boundary_markers(
        axes=axes,
        boundary_positions=np.array([3, 7]),
        boundary_labels=np.array(["2025-12-16", "2025-12-23"]),
    )

    for ax in axes:
        dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(dashed_lines) == 2
        assert [line.get_xdata()[0] for line in dashed_lines] == [3, 7]

    assert [text.get_text() for text in axes[0].texts] == ["2025-12-16", "2025-12-23"]
    assert len(axes[1].texts) == 0
    plt.close(fig)


def test_add_session_boundary_markers_requires_one_label_per_boundary():
    """Every boundary line should have a date label.

    Inputs
    ------
    None
        Uses two boundary positions and one label.

    Returns
    -------
    None
        Asserts that missing labels raise a ValueError.
    """
    fig, ax = plt.subplots()

    try:
        with np.testing.assert_raises(ValueError):
            ssplot.add_session_boundary_markers(
                axes=ax,
                boundary_positions=np.array([3, 7]),
                boundary_labels=np.array(["2025-12-16"]),
            )
    finally:
        plt.close(fig)


def test_add_session_boundary_markers_can_label_bottom_axis():
    """Session boundary helper should support labels below the bottom axis.

    Inputs
    ------
    None
        Uses two blank axes, two boundary positions, and two date labels.

    Returns
    -------
    None
        Asserts that labels are placed on the bottom axis below the plot and are
        not clipped.
    """
    fig, axes = plt.subplots(2, 1)

    ssplot.add_session_boundary_markers(
        axes=axes,
        boundary_positions=np.array([3, 7]),
        boundary_labels=np.array(["2025-12-16", "2025-12-23"]),
        label_location="bottom",
    )

    for ax in axes:
        dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(dashed_lines) == 2
        assert [line.get_xdata()[0] for line in dashed_lines] == [3, 7]

    assert len(axes[0].texts) == 0
    assert [text.get_text() for text in axes[1].texts] == ["2025-12-16", "2025-12-23"]
    assert all(text.get_position()[1] < 0 for text in axes[1].texts)
    assert all(not text.get_clip_on() for text in axes[1].texts)
    plt.close(fig)


def test_plot_transition_matrix_returns_figure_and_axes_for_multiple_states():
    """Transition-matrix plotting should return figure handles without saving.

    Inputs
    ------
    None
        Uses a 4-state transition matrix.

    Returns
    -------
    None
        Asserts that a matplotlib figure and axes are returned.
    """
    transition_matrix = np.eye(4)

    fig, ax = ssplot.plot_transition_matrix(transition_matrix)

    assert fig is not None
    assert ax is not None
    assert len(ax.images) == 1
    plt.close(fig)


def test_stack_state_durations_groups_by_state_index():
    """Duration helper should return one duration array per state.

    Inputs
    ------
    None
        Uses run-length encoded state ids and durations with one empty state.

    Returns
    -------
    None
        Asserts that durations are grouped by state index.
    """
    stacked = ssplot.stack_state_durations(
        inferred_state_list=np.array([0, 1, 0, 2]),
        inferred_durations=np.array([3, 4, 5, 6]),
        num_states=4,
    )

    assert len(stacked) == 4
    np.testing.assert_array_equal(stacked[0], np.array([3, 5]))
    np.testing.assert_array_equal(stacked[1], np.array([4]))
    np.testing.assert_array_equal(stacked[2], np.array([6]))
    np.testing.assert_array_equal(stacked[3], np.array([]))


def test_plot_state_duration_histogram_returns_figure_without_saving(tmp_path):
    """Duration histogram plotting should return figure handles without saving.

    Inputs
    ------
    tmp_path : pathlib.Path
        Temporary directory checked for accidental output files.

    Returns
    -------
    None
        Asserts that plotting returns a figure and does not write a file.
    """
    fig, ax = ssplot.plot_state_duration_histogram(
        inferred_state_list=np.array([0, 1, 0]),
        inferred_durations=np.array([3, 4, 5]),
        num_states=2,
        state_colors=["blue", "red"],
    )

    assert fig is not None
    assert ax is not None
    assert not (tmp_path / "unit_session_state_durations.png").exists()
    plt.close(fig)


def test_plot_block_lm_hmm_weight_comparison_returns_figure_without_session():
    """Block weight comparison should not require a full Session object.

    Inputs
    ------
    None
        Uses two small LM-HMM weight dictionaries with one observation and one
        predictor dimension.

    Returns
    -------
    None
        Asserts that plotting returns a figure and axes.
    """
    weight_dicts = [
        {
            "weights": np.array([[[0.5]], [[-0.25]]]),
            "mus": np.array([[3.0], [4.0]]),
            "label": "mle",
        },
        {
            "weights": np.array([[[0.75]], [[-0.5]]]),
            "mus": np.array([[2.5], [4.5]]),
            "label": "map",
        },
    ]

    fig, axes = ssplot.plot_block_lm_hmm_weight_comparison(weight_dicts)

    assert fig is not None
    assert len(np.atleast_1d(axes)) == 1
    plt.close(fig)


def test_plot_block_lm_hmm_weights_returns_figure_without_session():
    """Presentation block weights should use explicit colors instead of Session.

    Inputs
    ------
    None
        Uses one small LM-HMM weight dictionary with explicit predictor labels.

    Returns
    -------
    None
        Asserts that plotting returns a figure and axes.
    """
    weight_dict = {
        "weights": np.array([[[0.5]], [[-0.25]]]),
        "mus": np.array([[3.0], [4.0]]),
        "label": "map",
        "weight_labels": ["prev_n_rewarded"],
    }

    fig, axes = ssplot.plot_block_lm_hmm_weights(
        weight_dict,
        colors=["blue", "red"],
    )

    assert fig is not None
    assert len(np.atleast_1d(axes)) == 1
    plt.close(fig)


def test_plot_trial_glm_hmm_weights_returns_figure_without_session():
    """Trial GLM-HMM weight plotting should not save or require Session.

    Inputs
    ------
    None
        Uses one small GLM-HMM weight dictionary with two states and two
        predictors.

    Returns
    -------
    None
        Asserts that plotting returns a figure and axes.
    """
    weight_dicts = [
        {
            "weights": np.array([[[0.5, -0.1]], [[-0.25, 0.2]]]),
            "label": "map",
            "weight_labels": ["value", "bias"],
        }
    ]

    fig, ax = ssplot.plot_trial_glm_hmm_weights(weight_dicts)

    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_plot_trial_glm_hmm_state_summary_returns_figure():
    """Trial GLM-HMM state summary should return a figure and three axes.

    Inputs
    ------
    None
        Uses a dummy GLM-HMM with two states, one observation dimension, one
        non-bias predictor, and one bias weight.

    Returns
    -------
    None
        Asserts that plotting returns a matplotlib figure and axes tuple.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wk = np.array([[[0.25, 0.5]], [[-0.25, -0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_trial_glm_hmm_state_summary(
        posterior_probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
        observations=np.array([[0], [1]]),
        inputs=np.array([[0.5], [-0.5]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
    )

    assert fig is not None
    assert len(axes) == 3
    plt.close(fig)


def test_plot_trial_glm_hmm_state_summary_uses_custom_figsize_and_line_width():
    """Trial state summary should accept thinner, wider multisession plotting.

    Inputs
    ------
    None
        Uses two valid trials, two posterior-state traces, one observation
        dimension, and one non-bias predictor plus bias.

    Returns
    -------
    None
        Asserts that figure size and trace linewidths follow explicit plotting
        settings.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wk = np.array([[[0.25, 0.5]], [[-0.25, -0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_trial_glm_hmm_state_summary(
        posterior_probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
        observations=np.array([[0], [1]]),
        inputs=np.array([[0.5], [-0.5]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        figsize=(12, 4),
        line_width=0.4,
    )

    prob_ax, input_ax, obs_ax = axes
    np.testing.assert_allclose(fig.get_size_inches(), np.array([12, 4]))
    assert [line.get_linewidth() for line in prob_ax.lines] == [0.4, 0.4]
    assert [line.get_linewidth() for line in input_ax.lines] == [0.4]
    assert [line.get_linewidth() for line in obs_ax.lines] == [0.4, 0.4, 0.4]
    plt.close(fig)


def test_plot_trial_glm_hmm_state_summary_draws_labeled_session_boundaries():
    """Trial state summary should mark every axis with labeled session boundaries.

    Inputs
    ------
    None
        Uses two boundary positions and labels on a small dummy trial summary
        plot.

    Returns
    -------
    None
        Asserts that each axis receives dashed boundary lines and only the top
        axis receives boundary labels.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wk = np.array([[[0.25, 0.5]], [[-0.25, -0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_trial_glm_hmm_state_summary(
        posterior_probs=np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
        observations=np.array([[0], [1], [0]]),
        inputs=np.array([[0.5], [-0.5], [0.25]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        session_boundary_positions=np.array([1, 2]),
        session_boundary_labels=np.array(["2025-12-16", "2025-12-23"]),
    )

    for axis in axes:
        dashed_lines = [line for line in axis.lines if line.get_linestyle() == "--"]
        assert len(dashed_lines) >= 2
        assert [line.get_xdata()[0] for line in dashed_lines[-2:]] == [1, 2]

    assert len(axes[0].texts) == 0
    assert len(axes[1].texts) == 0
    assert [text.get_text() for text in axes[2].texts] == ["2025-12-16", "2025-12-23"]
    assert all(text.get_position()[1] < 0 for text in axes[2].texts)
    assert all(not text.get_clip_on() for text in axes[2].texts)
    plt.close(fig)


def test_plot_trial_glm_hmm_block_state_comparison_returns_figure():
    """Trial/block state comparison should return a figure and three axes.

    Inputs
    ------
    None
        Uses block labels, trial posterior probabilities, and a dummy GLM-HMM.

    Returns
    -------
    None
        Asserts that plotting returns a matplotlib figure and axes tuple.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wk = np.array([[[0.25, 0.5]], [[-0.25, -0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_trial_glm_hmm_block_state_comparison(
        block_states=np.array([[0], [1]]),
        trial_posterior_probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
        observations=np.array([[0], [1]]),
        inputs=np.array([[0.5], [-0.5]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
    )

    assert fig is not None
    assert len(axes) == 3
    plt.close(fig)


def test_plot_trial_glm_hmm_block_state_comparison_uses_custom_figsize_and_line_width():
    """Trial/block comparison should accept thinner, wider multisession plotting.

    Inputs
    ------
    None
        Uses two block labels, two trial posterior-state traces, and one
        observation dimension.

    Returns
    -------
    None
        Asserts that figure size and observation/posterior linewidths follow
        explicit plotting settings.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wk = np.array([[[0.25, 0.5]], [[-0.25, -0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_trial_glm_hmm_block_state_comparison(
        block_states=np.array([[0], [1]]),
        trial_posterior_probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
        observations=np.array([[0], [1]]),
        inputs=np.array([[0.5], [-0.5]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        figsize=(12, 4),
        line_width=0.4,
    )

    block_ax, trial_ax, prob_ax = axes
    np.testing.assert_allclose(fig.get_size_inches(), np.array([12, 4]))
    assert [line.get_linewidth() for line in block_ax.lines] == [0.4]
    assert [line.get_linewidth() for line in trial_ax.lines] == [0.4]
    assert [line.get_linewidth() for line in prob_ax.lines] == [0.4, 0.4]
    plt.close(fig)


def test_plot_trial_glm_hmm_block_state_comparison_draws_labeled_session_boundaries():
    """Trial/block comparison should mark every axis with session boundaries.

    Inputs
    ------
    None
        Uses two boundary positions and labels on a small dummy trial/block
        comparison plot.

    Returns
    -------
    None
        Asserts that each axis receives dashed boundary lines and only the top
        axis receives boundary labels.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wk = np.array([[[0.25, 0.5]], [[-0.25, -0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_trial_glm_hmm_block_state_comparison(
        block_states=np.array([[0], [1], [0]]),
        trial_posterior_probs=np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
        observations=np.array([[0], [1], [0]]),
        inputs=np.array([[0.5], [-0.5], [0.25]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        session_boundary_positions=np.array([1, 2]),
        session_boundary_labels=np.array(["2025-12-16", "2025-12-23"]),
    )

    for axis in axes:
        dashed_lines = [line for line in axis.lines if line.get_linestyle() == "--"]
        assert len(dashed_lines) >= 2
        assert [line.get_xdata()[0] for line in dashed_lines[-2:]] == [1, 2]

    assert len(axes[0].texts) == 0
    assert len(axes[1].texts) == 0
    assert [text.get_text() for text in axes[2].texts] == ["2025-12-16", "2025-12-23"]
    assert all(text.get_position()[1] < 0 for text in axes[2].texts)
    assert all(not text.get_clip_on() for text in axes[2].texts)
    plt.close(fig)


def test_plot_block_lm_hmm_presentation_summary_accepts_single_predictor_weights():
    """Presentation summary should preserve a predictor axis for one regressor.

    Inputs
    ------
    None
        Uses a dummy LM-HMM with one observation dimension and one predictor.

    Returns
    -------
    None
        Asserts that plotting returns a figure and axes.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        mus = np.array([[3.0], [4.0]])
        Wks = np.array([[[1.0]], [[-0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_block_lm_hmm_presentation_summary(
        posterior_probs=np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]),
        observations=np.array([[3.0], [4.0], [5.0]]),
        inputs=np.array([[1.0], [2.0], [3.0]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        predictor_labels=["prev_n_rewarded"],
    )

    assert fig is not None
    assert len(axes) == 2
    plt.close(fig)


def test_plot_block_lm_hmm_state_summary_uses_custom_figsize_and_line_width():
    """Block MLE predicted-state summary should accept long-session plot settings.

    Inputs
    ------
    None
        Uses three blocks, two posterior-state traces, one observation
        dimension, and one predictor dimension.

    Returns
    -------
    None
        Asserts that figure size and trace linewidths follow explicit plotting
        settings.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        mus = np.array([[3.0], [4.0]])
        Wks = np.array([[[1.0]], [[-0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_block_lm_hmm_state_summary(
        posterior_probs=np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]),
        observations=np.array([[3.0], [4.0], [5.0]]),
        inputs=np.array([[1.0], [2.0], [3.0]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        figsize=(12, 4),
        line_width=0.4,
    )

    prob_ax, obs_ax = axes
    posterior_lines = prob_ax.lines
    lower_panel_lines = obs_ax.lines
    np.testing.assert_allclose(fig.get_size_inches(), np.array([12, 4]))
    assert [line.get_linewidth() for line in posterior_lines] == [0.4, 0.4]
    assert [line.get_linewidth() for line in lower_panel_lines] == [0.4]
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_block_lm_hmm_state_summary_adds_bias_rl_twin_axis():
    """Block MLE summary should plot optional bias_rl on a fixed right y-axis."""
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        mus = np.array([[3.0], [4.0]])
        Wks = np.array([[[1.0]], [[-0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_block_lm_hmm_state_summary(
        posterior_probs=np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]),
        observations=np.array([[3.0], [4.0], [5.0]]),
        inputs=np.array([[1.0], [2.0], [3.0]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        secondary_trace=np.array([0.5, -0.25, 0.0]),
        secondary_trace_label="bias_rl",
        secondary_axis_label_size=9,
        secondary_tick_label_size=7,
    )

    assert len(axes) == 2
    assert len(fig.axes) == 3
    bias_ax = fig.axes[-1]
    assert bias_ax.get_ylabel() == "bias_rl"
    assert bias_ax.get_ylim() == (-1.0, 1.0)
    assert bias_ax.lines[0].get_linestyle() == "--"
    assert bias_ax.yaxis.label.get_size() == 9
    assert {label.get_size() for label in bias_ax.get_yticklabels()} == {7}
    np.testing.assert_allclose(bias_ax.lines[0].get_ydata(), np.array([0.5, -0.25, 0.0]))
    plt.close(fig)


def test_plot_block_lm_hmm_presentation_summary_uses_custom_figsize_and_line_width():
    """Block MAP predicted-state summary should accept long-session plot settings.

    Inputs
    ------
    None
        Uses three blocks, two posterior-state traces, one observation
        dimension, and one predictor dimension.

    Returns
    -------
    None
        Asserts that figure size and trace linewidths follow explicit plotting
        settings.
    """
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        mus = np.array([[3.0], [4.0]])
        Wks = np.array([[[1.0]], [[-0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_block_lm_hmm_presentation_summary(
        posterior_probs=np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]),
        observations=np.array([[3.0], [4.0], [5.0]]),
        inputs=np.array([[1.0], [2.0], [3.0]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        predictor_labels=["prev_n_rewarded"],
        figsize=(12, 4),
        line_width=0.4,
    )

    prob_ax, obs_ax = axes
    posterior_lines = prob_ax.lines
    lower_panel_lines = obs_ax.lines
    np.testing.assert_allclose(fig.get_size_inches(), np.array([12, 4]))
    assert [line.get_linewidth() for line in posterior_lines] == [0.4, 0.4]
    assert [line.get_linewidth() for line in lower_panel_lines] == [0.4]
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_block_lm_hmm_presentation_summary_adds_bias_rl_twin_axis():
    """Block MAP summary should plot optional bias_rl on a fixed right y-axis."""
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        mus = np.array([[3.0], [4.0]])
        Wks = np.array([[[1.0]], [[-0.5]]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

    fig, axes = ssplot.plot_block_lm_hmm_presentation_summary(
        posterior_probs=np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]),
        observations=np.array([[3.0], [4.0], [5.0]]),
        inputs=np.array([[1.0], [2.0], [3.0]]),
        hmm_fit=DummyHMM(),
        colors=["blue", "red"],
        cmap=plt.cm.Set1.copy(),
        predictor_labels=["prev_n_rewarded"],
        secondary_trace=np.array([0.5, -0.25, 0.0]),
        secondary_trace_label="bias_rl",
    )

    assert len(axes) == 2
    assert len(fig.axes) == 3
    bias_ax = fig.axes[-1]
    assert bias_ax.get_ylabel() == "bias_rl"
    assert bias_ax.get_ylim() == (-1.0, 1.0)
    assert bias_ax.lines[0].get_linestyle() == "--"
    np.testing.assert_allclose(bias_ax.lines[0].get_ydata(), np.array([0.5, -0.25, 0.0]))
    plt.close(fig)
