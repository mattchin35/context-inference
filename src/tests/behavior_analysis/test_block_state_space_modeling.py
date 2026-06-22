import importlib
import pickle
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def load_block_state_space_modeling_module():
    """Import LM-HMM analysis module with lightweight plotting stubs if needed.

    Returns
    -------
    module
        Imported `block_state_space_modeling` module.
    """
    if "seaborn" not in sys.modules:
        seaborn_stub = ModuleType("seaborn")
        seaborn_stub.xkcd_palette = lambda colors: colors
        seaborn_stub.set_style = lambda *_args, **_kwargs: None
        seaborn_stub.set_context = lambda *_args, **_kwargs: None
        sys.modules["seaborn"] = seaborn_stub

    if "joblib" not in sys.modules:
        joblib_stub = ModuleType("joblib")
        joblib_stub.delayed = lambda func: func

        class _Parallel:
            def __init__(self, n_jobs=None):
                self.n_jobs = n_jobs

            def __call__(self, calls):
                return [call for call in calls]

        joblib_stub.Parallel = _Parallel
        sys.modules["joblib"] = joblib_stub

    if "ssm" not in sys.modules:
        ssm_stub = ModuleType("ssm")
        ssm_stub.HMM = object
        ssm_stub.util = ModuleType("ssm.util")
        ssm_stub.util.find_permutation = lambda *args, **kwargs: None
        ssm_stub.util.rle = lambda states: (np.array([], dtype=int), np.array([], dtype=int))
        ssm_stub.plots = ModuleType("ssm.plots")
        ssm_stub.plots.gradient_cmap = lambda colors: colors
        ssm_stub.plots.white_to_color_cmap = lambda color: color
        sys.modules["ssm"] = ssm_stub
        sys.modules["ssm.util"] = ssm_stub.util
        sys.modules["ssm.plots"] = ssm_stub.plots

    return importlib.import_module("src.behavior_analysis.block_state_space_modeling")


bssm = load_block_state_space_modeling_module()
plotting_utils = importlib.import_module("src.behavior_analysis.plotting_utils")


def make_block_performance_df() -> pd.DataFrame:
    """Build a small LM-HMM block dataframe with mixed valid and invalid rows.

    Returns
    -------
    pd.DataFrame
        Table with block-level columns matching the LM-HMM analysis code.
        Valid rows contain integer-valued observations and predictors.
    """
    return pd.DataFrame(
        {
            "trials_to_correct": [3, 4, "None"],
            "prev_n_correct": [2, 2, "None"],
            "prev_n_rewarded": [1, 2, 0],
            "n_switches": [0, 1, 0],
            "bias_full_flag": ["False", "True", "False"],
        }
    )


def test_prepare_block_lm_hmm_data_default_uses_single_predictor_tuple():
    """Default LM prep should use a single `prev_n_rewarded` predictor column."""
    prepared = bssm.prepare_block_lm_hmm_data(make_block_performance_df())

    np.testing.assert_array_equal(prepared["observations"], np.array([[3], [4]]))
    np.testing.assert_array_equal(prepared["inputs"], np.array([[1], [2]]))
    np.testing.assert_array_equal(prepared["valid_mask"], np.array([True, True, False]))
    assert prepared["predictor_labels"] == ["prev_n_rewarded"]


def test_prepare_block_lm_hmm_data_single_predictor_returns_2d_inputs():
    """One predictor column should still produce a 2D LM input matrix."""
    prepared = bssm.prepare_block_lm_hmm_data(
        make_block_performance_df(),
        predictor_columns=("prev_n_rewarded",),
    )

    assert prepared["inputs"].shape == (2, 1)
    np.testing.assert_array_equal(prepared["inputs"][:, 0], np.array([1, 2]))


def test_valid_block_history_mask_requires_current_outcome_and_previous_history():
    """Block LM-HMM validity should use previous correct count as history marker."""
    block_df = pd.DataFrame(
        {
            "trials_to_correct": [3, "None", 5, 6],
            "prev_n_correct": [2, 2, "None", 1],
            "prev_n_rewarded": [1, 1, 3, 0],
        }
    )

    valid_mask = bssm.make_valid_block_history_mask(block_df)

    np.testing.assert_array_equal(
        valid_mask.to_numpy(),
        np.array([True, False, False, True]),
    )


def test_prepare_block_lm_hmm_data_uses_prev_correct_as_history_gate():
    """Rows missing previous-block history should be excluded for reward predictors too."""
    block_df = pd.DataFrame(
        {
            "trials_to_correct": [3, 5, 6],
            "prev_n_correct": [2, "None", 1],
            "prev_n_rewarded": [1, 3, 0],
        }
    )

    prepared = bssm.prepare_block_lm_hmm_data(
        block_df,
        predictor_columns=("prev_n_rewarded",),
    )

    np.testing.assert_array_equal(prepared["observations"], np.array([[3], [6]]))
    np.testing.assert_array_equal(prepared["inputs"], np.array([[1], [0]]))
    np.testing.assert_array_equal(prepared["valid_mask"], np.array([True, False, True]))


def test_get_session_boundary_markers_uses_valid_block_rows():
    """Multisession block plot boundaries should align to filtered HMM rows."""
    block_df = pd.DataFrame(
        {
            "source_date": [
                "2025-12-05",
                "2025-12-05",
                "2025-12-16",
                "2025-12-16",
                "2025-12-23",
            ]
        }
    )
    valid_mask = np.array([True, False, True, True, True])

    positions, labels = bssm.get_session_boundary_markers(
        block_df,
        valid_mask=valid_mask,
    )

    np.testing.assert_array_equal(positions, np.array([1, 3]))
    np.testing.assert_array_equal(labels, np.array(["2025-12-16", "2025-12-23"], dtype=object))


def test_get_session_boundary_markers_returns_empty_arrays_without_source_date():
    """Single-session block tables should not produce session boundary markers."""
    block_df = pd.DataFrame({"block_ix": [0, 1, 2]})
    valid_mask = np.array([True, True, True])

    positions, labels = bssm.get_session_boundary_markers(
        block_df,
        valid_mask=valid_mask,
    )

    assert positions.size == 0
    assert labels.size == 0


def test_hardcode_block_strategy_uses_prev_correct_as_history_gate():
    """Hardcoded inspection labels should only be assigned to valid history rows."""
    block_df = pd.DataFrame(
        {
            "trials_to_correct": [3, 5, 6],
            "prev_n_correct": [2, "None", 1],
            "cur_strategy_slope": [0.25, 0.75, 0.75],
            "bias_full_flag": ["False", "False", "False"],
        }
    )

    hardcoded = bssm.hardcode_block_strategy(block_df.copy())

    assert hardcoded["hardcoded_strategy"].tolist() == [
        "Inference",
        "None",
        "Qlearning",
    ]


def test_lm_weight_normalization_preserves_state_and_predictor_axes():
    """LM weight normalization should preserve predictor indexing for 1D cases."""
    weights = np.array([[[1.5]], [[-0.25]]])
    mus = np.array([[3.0], [4.0]])

    normalized_weights, normalized_mus = bssm.normalize_lm_observation_parameters(weights, mus)

    assert normalized_weights.shape == (2, 1, 1)
    assert normalized_mus.shape == (2, 1)
    assert normalized_weights[:, 0, 0].tolist() == [1.5, -0.25]
    assert normalized_mus[:, 0].tolist() == [3.0, 4.0]


def test_assign_inferred_states_to_blocks_marks_invalid_rows():
    """Inferred state assignment should keep invalid blocks explicitly missing."""
    block_df = make_block_performance_df().copy()
    valid_mask = np.array([True, True, False])

    updated = bssm.assign_inferred_states_to_blocks(
        block_df,
        valid_mask=valid_mask,
        most_likely_states=np.array([1, 0]),
    )

    assert updated["inferred_strategy"].tolist() == [1, 0, "None"]


def test_add_cur_strategy_slope_maps_valid_state_slopes():
    """Current strategy slope should map inferred state ids to state weights.

    Inputs
    ------
    None
        Uses a block dataframe with mixed valid and invalid rows.

    Returns
    -------
    None
        Asserts that valid blocks receive state-specific slopes and invalid
        blocks remain string `"None"` for CSV compatibility.
    """
    block_df = pd.DataFrame({"block_ix": [0, 1, 2]})
    valid_mask = np.array([True, False, True])
    inferred_states = np.array([1, "None", 0], dtype=object)
    primary_predictor_weights = np.array([0.25, 0.75])

    updated = bssm.add_cur_strategy_slope(
        block_df,
        valid_mask=valid_mask,
        inferred_states=inferred_states,
        primary_predictor_weights=primary_predictor_weights,
    )

    assert updated["cur_strategy_slope"].tolist() == [0.75, "None", 0.25]


def test_add_cur_strategy_slope_single_state_fills_all_rows():
    """One-state block models should assign the single slope to every block.

    Inputs
    ------
    None
        Uses a block dataframe with mixed valid and invalid rows.

    Returns
    -------
    None
        Asserts that one-state behavior matches the previous inline logic.
    """
    block_df = pd.DataFrame({"block_ix": [0, 1, 2]})

    updated = bssm.add_cur_strategy_slope(
        block_df,
        valid_mask=np.array([True, False, True]),
        inferred_states=np.array([0, "None", 0], dtype=object),
        primary_predictor_weights=np.array([0.5]),
    )

    assert updated["cur_strategy_slope"].tolist() == [0.5, 0.5, 0.5]


def test_mle_block_states_uses_hmm_builder_and_normalizes_weights(monkeypatch, tmp_path):
    """MLE block fitting should use the shared HMM builder and stable weight shapes."""
    captured = {}

    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wks = np.array([[1.5], [-0.25]])
        mus = np.array([3.0, 4.0])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return DummyHMM()

    monkeypatch.setattr(bssm, "build_input_driven_hmm", fake_builder)
    monkeypatch.setattr(
        bssm.ssm,
        "HMM",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct ssm.HMM should not be used")),
        raising=False,
    )

    model_dict, updated_block_df = bssm.mle_block_states(
        make_block_performance_df().iloc[:2].copy(),
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        num_states=2,
    )

    assert captured["num_states"] == 2
    assert captured["obs_dim"] == 1
    assert captured["input_dim"] == 1
    assert captured["algorithm"] == "MLE"
    weight_dict = model_dict["mle"]["weight_dict"]
    assert weight_dict["weights"].shape == (2, 1, 1)
    assert weight_dict["mus"].shape == (2, 1)
    assert updated_block_df["inferred_strategy"].tolist() == [0, 1]


def test_mle_block_states_records_random_seed(monkeypatch, tmp_path):
    """MLE block fitting should store the seed used for stochastic initialization."""
    class DummyTransitions:
        log_Ps = np.zeros((1, 1))

    class DummyObservations:
        Wks = np.array([[[1.5]]])
        mus = np.array([[3.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.zeros(observations.shape[0], dtype=int)

        def expected_states(self, data=None, input=None):
            return (np.ones((data.shape[0], 1)),)

    monkeypatch.setattr(bssm, "build_input_driven_hmm", lambda **_kwargs: DummyHMM())

    model_dict, _ = bssm.mle_block_states(
        make_block_performance_df().iloc[:2].copy(),
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        num_states=1,
        random_seed=123,
    )

    assert model_dict["mle"]["random_seed"] == 123


def test_mle_block_states_derives_observation_dimension_from_prepared_data(monkeypatch, tmp_path):
    """MLE block fitting should not hardcode one observation dimension."""
    captured = {}
    block_df = make_block_performance_df().iloc[:2].copy()

    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wks = np.array([[[1.5]], [[-0.25]]])
        mus = np.array([[3.0, 3.5], [4.0, 4.5]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    def fake_prepare(_block_df):
        return {
            "observations": np.array([[3, 4], [5, 6]]),
            "inputs": np.array([[1], [2]]),
            "valid_mask": np.array([True, True]),
            "predictor_labels": ["prev_n_rewarded"],
        }

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return DummyHMM()

    monkeypatch.setattr(bssm, "prepare_block_lm_hmm_data", fake_prepare)
    monkeypatch.setattr(bssm, "build_input_driven_hmm", fake_builder)

    bssm.mle_block_states(
        block_df,
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        num_states=2,
    )

    assert captured["obs_dim"] == 2


def test_map_block_states_uses_hmm_builder_with_map_priors(monkeypatch, tmp_path):
    """MAP block fitting should use the shared HMM builder and forward priors."""
    captured = {}

    class DummyTransitions:
        log_Ps = np.log(np.array([[0.8, 0.2], [0.2, 0.8]]))

    class DummyObservations:
        Wks = np.array([[[0.25]], [[0.75]]])
        mus = np.array([[3.0], [4.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0, 1.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return DummyHMM()

    monkeypatch.setattr(bssm, "build_input_driven_hmm", fake_builder)
    monkeypatch.setattr(
        bssm.ssm,
        "HMM",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct ssm.HMM should not be used")),
        raising=False,
    )

    bssm.map_block_states(
        make_block_performance_df().iloc[:2].copy(),
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        num_states=2,
        prior_alpha=1.5,
        prior_sigma=2.5,
        model_dict={"mle": {}},
    )

    assert captured["num_states"] == 2
    assert captured["obs_dim"] == 1
    assert captured["input_dim"] == 1
    assert captured["algorithm"] == "MAP"
    assert captured["prior_alpha"] == 1.5
    assert captured["prior_sigma"] == 2.5


def test_map_block_states_records_random_seed(monkeypatch, tmp_path):
    """MAP block fitting should store the seed used for stochastic initialization."""
    class DummyTransitions:
        log_Ps = np.zeros((1, 1))

    class DummyObservations:
        Wks = np.array([[[1.5]]])
        mus = np.array([[3.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.zeros(observations.shape[0], dtype=int)

        def expected_states(self, data=None, input=None):
            return (np.ones((data.shape[0], 1)),)

    monkeypatch.setattr(bssm, "build_input_driven_hmm", lambda **_kwargs: DummyHMM())

    model_dict, _ = bssm.map_block_states(
        make_block_performance_df().iloc[:2].copy(),
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        num_states=1,
        model_dict={"mle": {}},
        random_seed=456,
    )

    assert model_dict["map"]["random_seed"] == 456


def test_map_block_states_derives_observation_dimension_from_prepared_data(monkeypatch, tmp_path):
    """MAP block fitting should not hardcode one observation dimension."""
    captured = {}
    block_df = make_block_performance_df().iloc[:2].copy()

    class DummyTransitions:
        log_Ps = np.log(np.array([[0.8, 0.2], [0.2, 0.8]]))

    class DummyObservations:
        Wks = np.array([[[0.25]], [[0.75]]])
        mus = np.array([[3.0, 3.5], [4.0, 4.5]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0, 1.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    def fake_prepare(_block_df):
        return {
            "observations": np.array([[3, 4], [5, 6]]),
            "inputs": np.array([[1], [2]]),
            "valid_mask": np.array([True, True]),
            "predictor_labels": ["prev_n_rewarded"],
        }

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return DummyHMM()

    monkeypatch.setattr(bssm, "prepare_block_lm_hmm_data", fake_prepare)
    monkeypatch.setattr(bssm, "build_input_driven_hmm", fake_builder)

    bssm.map_block_states(
        block_df,
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        num_states=2,
        model_dict={"mle": {}},
    )

    assert captured["obs_dim"] == 2


def test_mle_block_states_passes_valid_bias_rl_trace_to_state_plot(monkeypatch, tmp_path):
    """MLE state plot overlay should align bias_rl to HMM-valid block rows."""
    captured = {}
    block_df = pd.DataFrame(
        {
            "trials_to_correct": [3, "None", 5],
            "prev_n_correct": [2, 2, 1],
            "prev_n_rewarded": [1, 2, 0],
            "bias_rl": [0.5, 0.99, -0.25],
        }
    )

    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wks = np.array([[[1.5]], [[-0.25]]])
        mus = np.array([[3.0], [4.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    def fake_state_plot(*_args, **kwargs):
        captured["secondary_trace"] = kwargs["secondary_trace"]
        captured["secondary_trace_label"] = kwargs["secondary_trace_label"]
        return plt.subplots(2, 1)

    monkeypatch.setattr(bssm, "build_input_driven_hmm", lambda **_kwargs: DummyHMM())
    monkeypatch.setattr(bssm.state_space_plotting, "plot_transition_matrix", lambda *_args: plt.subplots())
    monkeypatch.setattr(bssm.state_space_plotting, "plot_state_duration_histogram", lambda **_kwargs: plt.subplots())
    monkeypatch.setattr(bssm.state_space_plotting, "plot_block_lm_hmm_state_summary", fake_state_plot)

    bssm.mle_block_states(
        block_df,
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=True,
        num_states=2,
        plot_bias_rl=True,
    )

    np.testing.assert_allclose(captured["secondary_trace"], np.array([0.5, -0.25]))
    assert captured["secondary_trace_label"] == "bias_rl"


def test_map_block_states_passes_valid_bias_rl_trace_to_state_plot(monkeypatch, tmp_path):
    """MAP state plot overlay should align bias_rl to HMM-valid block rows."""
    captured = {}
    block_df = pd.DataFrame(
        {
            "trials_to_correct": [3, "None", 5],
            "prev_n_correct": [2, 2, 1],
            "prev_n_rewarded": [1, 2, 0],
            "bias_rl": [0.5, 0.99, -0.25],
        }
    )

    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wks = np.array([[[1.5]], [[-0.25]]])
        mus = np.array([[3.0], [4.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    def fake_presentation_plot(*_args, **kwargs):
        captured["secondary_trace"] = kwargs["secondary_trace"]
        captured["secondary_trace_label"] = kwargs["secondary_trace_label"]
        return plt.subplots(2, 1)

    monkeypatch.setattr(bssm, "build_input_driven_hmm", lambda **_kwargs: DummyHMM())
    monkeypatch.setattr(bssm.state_space_plotting, "plot_transition_matrix", lambda *_args: plt.subplots())
    monkeypatch.setattr(bssm.state_space_plotting, "plot_state_duration_histogram", lambda **_kwargs: plt.subplots())
    monkeypatch.setattr(
        bssm.state_space_plotting,
        "plot_block_lm_hmm_presentation_summary",
        fake_presentation_plot,
    )

    bssm.map_block_states(
        block_df,
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=True,
        num_states=2,
        model_dict={"mle": {}},
        plot_bias_rl=True,
    )

    np.testing.assert_allclose(captured["secondary_trace"], np.array([0.5, -0.25]))
    assert captured["secondary_trace_label"] == "bias_rl"


def test_mle_block_states_rejects_missing_bias_rl_overlay_column(monkeypatch, tmp_path):
    """Requesting a bias_rl overlay without the column should fail clearly."""
    class DummyTransitions:
        log_Ps = np.zeros((1, 1))

    class DummyObservations:
        Wks = np.array([[[1.5]]])
        mus = np.array([[3.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.zeros(observations.shape[0], dtype=int)

        def expected_states(self, data=None, input=None):
            return (np.ones((data.shape[0], 1)),)

    monkeypatch.setattr(bssm, "build_input_driven_hmm", lambda **_kwargs: DummyHMM())

    with pytest.raises(ValueError, match="bias_rl"):
        bssm.mle_block_states(
            make_block_performance_df().iloc[:2].copy(),
            figure_path=tmp_path,
            sess_id_tag="unit_session",
            plot=True,
            num_states=1,
            plot_bias_rl=True,
        )



def test_save_block_model_dict_writes_expected_pickle(tmp_path):
    """The block-model workflow save helper should own the pickle artifact.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used as the session processed-data path.

    Returns
    -------
    None
        Asserts that the helper writes and returns the expected pickle path.
    """
    model_dict = {"map": {"fit_log_likelihood": [1.0, 2.0]}}

    save_path = bssm.save_block_model_dict(
        model_dict,
        processed_data_path=tmp_path,
        sess_id_full="unit_session",
    )

    assert save_path == tmp_path / "unit_session_block_statedict.pkl"
    with open(save_path, "rb") as file:
        reloaded = pickle.load(file)
    assert reloaded == model_dict


def test_map_block_states_does_not_pickle_model_dict(monkeypatch, tmp_path):
    """MAP fitting should return model data without owning the save path.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the HMM backend and pickle dump with lightweight test doubles.
    tmp_path : pathlib.Path
        Temporary directory used for session paths.

    Returns
    -------
    None
        Asserts that `map_block_states` does not call `pkl.dump`.
    """
    dump_calls = []

    class DummyTransitions:
        log_Ps = np.log(np.array([[0.8, 0.2], [0.2, 0.8]]))

    class DummyObservations:
        Wks = np.array([[[0.25]], [[0.75]]])
        mus = np.array([[3.0], [4.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0, 1.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    block_df = make_block_performance_df().iloc[:2].copy()

    monkeypatch.setattr(bssm.ssm, "HMM", DummyHMM)
    monkeypatch.setattr(bssm.pkl, "dump", lambda *args, **kwargs: dump_calls.append(args))

    model_dict, updated_block_df = bssm.map_block_states(
        block_df,
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        num_states=2,
        model_dict={"mle": {}},
    )

    assert dump_calls == []
    assert "map" in model_dict
    assert updated_block_df["inferred_strategy"].tolist() == [0, 1]


def test_run_block_modeling_saves_weight_figures_from_returned_figures(monkeypatch, tmp_path):
    """Block modeling pipeline should own weight figure filenames and saves.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces model fitting and inherited-strategy helpers with lightweight
        test doubles.
    tmp_path : pathlib.Path
        Temporary processed-data and figure directory.

    Returns
    -------
    None
        Asserts that returned plotting figures are saved by the pipeline.
    """
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        figure_path=tmp_path,
        sess_id_full="unit_session",
        sess_id_abbreviated="unit",
    )
    block_performance = make_block_performance_df().iloc[:2].copy()
    augmented_trial_df = pd.DataFrame({"block_id": [0, 1], "trial": [0, 1]})
    model_dict = {
        "mle": {
            "weight_dict": {
                "weights": np.array([[[0.25]], [[-0.5]]]),
                "mus": np.array([[3.0], [4.0]]),
                "label": "mle",
            }
        },
        "map": {
            "weight_dict": {
                "weights": np.array([[[0.75]], [[-0.25]]]),
                "mus": np.array([[2.0], [5.0]]),
                "label": "map",
                "weight_labels": ["prev_n_rewarded"],
            }
        },
    }

    monkeypatch.setattr(
        bssm,
        "mle_block_states",
        lambda block_df, *args, **kwargs: ({"mle": model_dict["mle"]}, block_df),
    )
    monkeypatch.setattr(
        bssm,
        "map_block_states",
        lambda block_df, *args, **kwargs: (model_dict, block_df),
    )
    monkeypatch.setattr(
        bssm,
        "hardcode_block_strategy",
        lambda block_df: block_df.assign(hardcoded_strategy=["Inference", "Qlearning"]),
    )
    monkeypatch.setattr(
        bssm,
        "trials_inherit_strategy",
        lambda block_df, trial_df: trial_df.assign(inherited_block_strategy=[0, 1]),
    )

    bssm.run_block_modeling(
        block_performance,
        augmented_trial_df,
        session=session,
    )

    assert (tmp_path / "unit_session_hmm_weight_comparison.png").exists()
    assert (tmp_path / "unit_session_hmm_weights.png").exists()


def test_run_block_modeling_passes_full_session_id_as_modeling_tag(monkeypatch, tmp_path):
    """Block modeling should inject the same filename tag into MLE and MAP fits.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces model fitting and downstream helpers with lightweight captures.
    tmp_path : pathlib.Path
        Temporary processed-data and figure directory.

    Returns
    -------
    None
        Asserts that both modeling functions receive `session.sess_id_full` as
        `sess_id_tag`.
    """
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        figure_path=tmp_path,
        sess_id_full="unit_session_full",
        sess_id_abbreviated="unit_abbrev",
    )
    block_performance = make_block_performance_df().iloc[:2].copy()
    augmented_trial_df = pd.DataFrame({"block_id": [0, 1], "trial": [0, 1]})
    captured_tags = {}
    model_dict = {
        "mle": {
            "weight_dict": {
                "weights": np.array([[[0.25]], [[-0.5]]]),
                "mus": np.array([[3.0], [4.0]]),
                "label": "mle",
            }
        },
        "map": {
            "weight_dict": {
                "weights": np.array([[[0.75]], [[-0.25]]]),
                "mus": np.array([[2.0], [5.0]]),
                "label": "map",
                "weight_labels": ["prev_n_rewarded"],
            }
        },
    }

    def fake_mle_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_tags["mle"] = sess_id_tag
        return {"mle": model_dict["mle"]}, block_df

    def fake_map_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_tags["map"] = sess_id_tag
        return model_dict, block_df

    monkeypatch.setattr(bssm, "mle_block_states", fake_mle_block_states)
    monkeypatch.setattr(bssm, "map_block_states", fake_map_block_states)
    monkeypatch.setattr(
        bssm,
        "hardcode_block_strategy",
        lambda block_df: block_df.assign(hardcoded_strategy=["Inference", "Qlearning"]),
    )
    monkeypatch.setattr(
        bssm,
        "trials_inherit_strategy",
        lambda block_df, trial_df: trial_df.assign(inherited_block_strategy=[0, 1]),
    )

    bssm.run_block_modeling(block_performance, augmented_trial_df, session=session)

    assert captured_tags == {
        "mle": "unit_session_full",
        "map": "unit_session_full",
    }


def test_run_block_modeling_derives_distinct_mle_and_map_seeds(monkeypatch, tmp_path):
    """Block modeling should derive separate child seeds for MLE and MAP fits."""
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        figure_path=tmp_path,
        sess_id_full="unit_session",
        sess_id_abbreviated="unit",
    )
    block_performance = make_block_performance_df().iloc[:2].copy()
    augmented_trial_df = pd.DataFrame({"block_id": [0, 1], "trial": [0, 1]})
    captured_seeds = {}
    model_dict = {
        "mle": {
            "weight_dict": {
                "weights": np.array([[[0.25]], [[-0.5]]]),
                "mus": np.array([[3.0], [4.0]]),
                "label": "mle",
            }
        },
        "map": {
            "weight_dict": {
                "weights": np.array([[[0.75]], [[-0.25]]]),
                "mus": np.array([[2.0], [5.0]]),
                "label": "map",
                "weight_labels": ["prev_n_rewarded"],
            }
        },
    }

    def fake_mle_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_seeds["mle"] = kwargs["random_seed"]
        return {"mle": model_dict["mle"]}, block_df

    def fake_map_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_seeds["map"] = kwargs["random_seed"]
        return model_dict, block_df

    monkeypatch.setattr(bssm, "mle_block_states", fake_mle_block_states)
    monkeypatch.setattr(bssm, "map_block_states", fake_map_block_states)
    monkeypatch.setattr(
        bssm,
        "hardcode_block_strategy",
        lambda block_df: block_df.assign(hardcoded_strategy=["Inference", "Qlearning"]),
    )
    monkeypatch.setattr(
        bssm,
        "trials_inherit_strategy",
        lambda block_df, trial_df: trial_df.assign(inherited_block_strategy=[0, 1]),
    )

    bssm.run_block_modeling(
        block_performance,
        augmented_trial_df,
        session=session,
        random_seed=123,
    )

    assert captured_seeds["mle"] is not None
    assert captured_seeds["map"] is not None
    assert captured_seeds["mle"] != captured_seeds["map"]


def test_run_block_modeling_forwards_predicted_state_line_width(monkeypatch, tmp_path):
    """Block modeling should pass predicted-state linewidth to MLE and MAP plots.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces model fitting and downstream helpers with lightweight captures.
    tmp_path : pathlib.Path
        Temporary processed-data and figure directory.

    Returns
    -------
    None
        Asserts that both block-state fitting stages receive the requested
        predicted-state plot linewidth.
    """
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        figure_path=tmp_path,
        sess_id_full="unit_session",
        sess_id_abbreviated="unit",
    )
    block_performance = make_block_performance_df().iloc[:2].copy()
    augmented_trial_df = pd.DataFrame({"block_id": [0, 1], "trial": [0, 1]})
    captured_line_widths = {}
    model_dict = {
        "mle": {
            "weight_dict": {
                "weights": np.array([[[0.25]], [[-0.5]]]),
                "mus": np.array([[3.0], [4.0]]),
                "label": "mle",
            }
        },
        "map": {
            "weight_dict": {
                "weights": np.array([[[0.75]], [[-0.25]]]),
                "mus": np.array([[2.0], [5.0]]),
                "label": "map",
                "weight_labels": ["prev_n_rewarded"],
            }
        },
    }

    def fake_mle_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_line_widths["mle"] = kwargs["predicted_state_line_width"]
        return {"mle": model_dict["mle"]}, block_df

    def fake_map_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_line_widths["map"] = kwargs["predicted_state_line_width"]
        return model_dict, block_df

    monkeypatch.setattr(bssm, "mle_block_states", fake_mle_block_states)
    monkeypatch.setattr(bssm, "map_block_states", fake_map_block_states)
    monkeypatch.setattr(
        bssm,
        "hardcode_block_strategy",
        lambda block_df: block_df.assign(hardcoded_strategy=["Inference", "Qlearning"]),
    )
    monkeypatch.setattr(
        bssm,
        "trials_inherit_strategy",
        lambda block_df, trial_df: trial_df.assign(inherited_block_strategy=[0, 1]),
    )

    bssm.run_block_modeling(
        block_performance,
        augmented_trial_df,
        session=session,
        predicted_state_line_width=0.4,
    )

    assert captured_line_widths == {
        "mle": 0.4,
        "map": 0.4,
    }


def test_run_block_modeling_forwards_state_plot_figsize(monkeypatch, tmp_path):
    """Block modeling should pass state plot figsize to MLE and MAP plots."""
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        figure_path=tmp_path,
        sess_id_full="unit_session",
        sess_id_abbreviated="unit",
    )
    block_performance = make_block_performance_df().iloc[:2].copy()
    augmented_trial_df = pd.DataFrame({"block_id": [0, 1], "trial": [0, 1]})
    captured_figsizes = {}
    model_dict = {
        "mle": {
            "weight_dict": {
                "weights": np.array([[[0.25]], [[-0.5]]]),
                "mus": np.array([[3.0], [4.0]]),
                "label": "mle",
            }
        },
        "map": {
            "weight_dict": {
                "weights": np.array([[[0.75]], [[-0.25]]]),
                "mus": np.array([[2.0], [5.0]]),
                "label": "map",
                "weight_labels": ["prev_n_rewarded"],
            }
        },
    }

    def fake_mle_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_figsizes["mle"] = kwargs["state_plot_figsize"]
        return {"mle": model_dict["mle"]}, block_df

    def fake_map_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_figsizes["map"] = kwargs["state_plot_figsize"]
        return model_dict, block_df

    monkeypatch.setattr(bssm, "mle_block_states", fake_mle_block_states)
    monkeypatch.setattr(bssm, "map_block_states", fake_map_block_states)
    monkeypatch.setattr(
        bssm,
        "hardcode_block_strategy",
        lambda block_df: block_df.assign(hardcoded_strategy=["Inference", "Qlearning"]),
    )
    monkeypatch.setattr(
        bssm,
        "trials_inherit_strategy",
        lambda block_df, trial_df: trial_df.assign(inherited_block_strategy=[0, 1]),
    )

    bssm.run_block_modeling(
        block_performance,
        augmented_trial_df,
        session=session,
        state_plot_figsize=(18, 6),
    )

    assert captured_figsizes == {
        "mle": (18, 6),
        "map": (18, 6),
    }


def test_run_block_modeling_forwards_bias_rl_overlay_option(monkeypatch, tmp_path):
    """Block modeling should pass bias_rl overlay settings to MLE and MAP plots."""
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        figure_path=tmp_path,
        sess_id_full="unit_session",
        sess_id_abbreviated="unit",
    )
    block_performance = make_block_performance_df().iloc[:2].copy()
    block_performance["bias_rl"] = [0.5, -0.25]
    augmented_trial_df = pd.DataFrame({"block_id": [0, 1], "trial": [0, 1]})
    captured_overlay_options = {}
    model_dict = {
        "mle": {
            "weight_dict": {
                "weights": np.array([[[0.25]], [[-0.5]]]),
                "mus": np.array([[3.0], [4.0]]),
                "label": "mle",
            }
        },
        "map": {
            "weight_dict": {
                "weights": np.array([[[0.75]], [[-0.25]]]),
                "mus": np.array([[2.0], [5.0]]),
                "label": "map",
                "weight_labels": ["prev_n_rewarded"],
            }
        },
    }

    def fake_mle_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_overlay_options["mle"] = (
            kwargs["plot_bias_rl"],
            kwargs["bias_rl_column"],
        )
        return {"mle": model_dict["mle"]}, block_df

    def fake_map_block_states(block_df, figure_path, sess_id_tag, **kwargs):
        captured_overlay_options["map"] = (
            kwargs["plot_bias_rl"],
            kwargs["bias_rl_column"],
        )
        return model_dict, block_df

    monkeypatch.setattr(bssm, "mle_block_states", fake_mle_block_states)
    monkeypatch.setattr(bssm, "map_block_states", fake_map_block_states)
    monkeypatch.setattr(
        bssm,
        "hardcode_block_strategy",
        lambda block_df: block_df.assign(hardcoded_strategy=["Inference", "Qlearning"]),
    )
    monkeypatch.setattr(
        bssm,
        "trials_inherit_strategy",
        lambda block_df, trial_df: trial_df.assign(inherited_block_strategy=[0, 1]),
    )

    bssm.run_block_modeling(
        block_performance,
        augmented_trial_df,
        session=session,
        plot_bias_rl=True,
        bias_rl_column="bias_rl",
    )

    assert captured_overlay_options == {
        "mle": (True, "bias_rl"),
        "map": (True, "bias_rl"),
    }


def test_run_information_criteria_rejects_map_for_lm_hmm(tmp_path):
    """LM-HMM information criteria should reject MAP because IC is MLE-only."""
    session = SimpleNamespace(
        figure_path=tmp_path,
        sess_id_full="unit_session",
    )

    with pytest.raises(ValueError, match="MLE"):
        bssm.run_information_criteria(
            make_block_performance_df(),
            session=session,
            algorithm="MAP",
        )


def test_run_information_criteria_passes_explicit_plot_fields(monkeypatch, tmp_path):
    """Pipeline function should pass explicit plot fields to the plot helper.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces model-selection computation and plotting with lightweight
        captures.
    tmp_path : pathlib.Path
        Temporary figure directory.

    Returns
    -------
    None
        Asserts that plotting receives `figure_path` and `sess_id_full` instead
        of a full session object.
    """
    captured = {}
    session = SimpleNamespace(
        figure_path=tmp_path,
        sess_id_full="unit_session",
    )
    expected_aic = None
    expected_bic = None

    def fake_calculate_information_criteria(**kwargs):
        nonlocal expected_aic, expected_bic
        n_states = kwargs["states"].size
        expected_aic = np.zeros((n_states, 1))
        expected_bic = np.ones((n_states, 1))
        return expected_aic, expected_bic

    def fake_plot_information_criteria(aic, bic, states, figure_path, sess_id_full):
        captured["figure_path"] = figure_path
        captured["sess_id_full"] = sess_id_full
        captured["states"] = states.copy()
        captured["aic"] = aic
        captured["bic"] = bic
        return None

    monkeypatch.setattr(bssm, "calculate_information_criteria", fake_calculate_information_criteria)
    monkeypatch.setattr(bssm, "plot_information_criteria", fake_plot_information_criteria)

    model_selection = bssm.run_information_criteria(make_block_performance_df(), session=session)

    np.testing.assert_array_equal(model_selection["AIC"], expected_aic)
    np.testing.assert_array_equal(model_selection["BIC"], expected_bic)
    np.testing.assert_array_equal(captured["aic"], expected_aic)
    np.testing.assert_array_equal(captured["bic"], expected_bic)
    assert captured["figure_path"] == tmp_path
    assert captured["sess_id_full"] == "unit_session"
    np.testing.assert_array_equal(captured["states"], np.arange(1, 6))


def test_run_information_criteria_forwards_model_selection_settings(monkeypatch, tmp_path):
    """Pipeline function should expose state range and restart settings."""
    captured = {}
    session = SimpleNamespace(
        figure_path=tmp_path,
        sess_id_full="unit_session",
    )

    def fake_calculate_information_criteria(**kwargs):
        captured["states"] = kwargs["states"].copy()
        captured["nRunEM"] = kwargs["nRunEM"]
        captured["n_jobs"] = kwargs["n_jobs"]
        captured["random_seed"] = kwargs["random_seed"]
        captured["restart_random_seeds"] = kwargs["restart_random_seeds"]
        n_states = kwargs["states"].size
        captured["expected_aic"] = np.zeros((n_states, kwargs["nRunEM"]))
        captured["expected_bic"] = np.ones((n_states, kwargs["nRunEM"]))
        return captured["expected_aic"], captured["expected_bic"]

    monkeypatch.setattr(bssm, "calculate_information_criteria", fake_calculate_information_criteria)
    monkeypatch.setattr(bssm, "plot_information_criteria", lambda aic, bic, states, figure_path, sess_id_full: None)

    model_selection = bssm.run_information_criteria(
        make_block_performance_df(),
        session=session,
        min_states=2,
        max_states=4,
        n_threads=3,
        n_runs=2,
        random_seed=123,
    )

    np.testing.assert_array_equal(captured["states"], np.arange(2, 5))
    assert captured["nRunEM"] == 2
    assert captured["n_jobs"] == 3
    assert captured["random_seed"] == 123
    assert len(captured["restart_random_seeds"]) == 6
    assert len(set(captured["restart_random_seeds"])) == 6
    np.testing.assert_array_equal(
        model_selection["restart_random_seeds"],
        np.asarray(captured["restart_random_seeds"], dtype=object).reshape(3, 2),
    )
    np.testing.assert_array_equal(model_selection["AIC"], captured["expected_aic"])
    np.testing.assert_array_equal(model_selection["BIC"], captured["expected_bic"])


def test_count_lm_hmm_parameters_matches_current_formula():
    """LM-HMM parameter helper should preserve the existing information-criteria count."""
    num_states = 3
    obs_dim = 2
    input_dim = 4
    expected = (num_states + 1) * (num_states - 1) + num_states * (obs_dim * input_dim + 2 * obs_dim)

    assert bssm.count_lm_hmm_parameters(num_states=num_states, obs_dim=obs_dim, input_dim=input_dim) == expected


def test_calculate_information_criteria_uses_named_lm_hmm_scorer(monkeypatch):
    """LM-HMM information criteria should call the descriptive scorer helper."""
    calls = []

    def old_scorer(*_args, **_kwargs):
        raise AssertionError("single_func should not be used")

    def fake_scorer(
        observations,
        inputs,
        num_states,
        algorithm,
        n_iter,
        tol,
        prior_alpha,
        prior_sigma,
        random_seed=None,
    ):
        calls.append(
            {
                "num_states": num_states,
                "algorithm": algorithm,
                "n_iter": n_iter,
                "tol": tol,
                "prior_alpha": prior_alpha,
                "prior_sigma": prior_sigma,
                "random_seed": random_seed,
            }
        )
        return -5.0

    class FakeParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, calls_to_run):
            return list(calls_to_run)

    monkeypatch.setattr(bssm, "single_func", old_scorer, raising=False)
    monkeypatch.setattr(bssm, "fit_lm_hmm_and_score_log_likelihood", fake_scorer, raising=False)
    monkeypatch.setattr(bssm, "delayed", lambda func: lambda *args, **kwargs: func(*args, **kwargs))
    monkeypatch.setattr(bssm, "Parallel", FakeParallel)

    aic, bic = bssm.calculate_information_criteria(
        observations=np.ones((4, 1)),
        inputs=np.ones((4, 2)),
        states=np.array([2]),
        nRunEM=2,
        n_jobs=1,
        algorithm="MLE",
        prior_alpha=1.5,
        prior_sigma=2.5,
        random_seed=123,
    )

    assert len(calls) == 2
    assert all(call["num_states"] == 2 for call in calls)
    assert all(call["algorithm"] == "MLE" for call in calls)
    assert all(call["prior_alpha"] == 1.5 for call in calls)
    assert all(call["prior_sigma"] == 2.5 for call in calls)
    assert calls[0]["random_seed"] != calls[1]["random_seed"]
    assert aic.shape == (1, 2)
    assert bic.shape == (1, 2)


def test_run_cross_validation_passes_explicit_plot_fields(monkeypatch, tmp_path):
    """Cross-validation pipeline should pass explicit plot fields to plotting.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces cross-validation computation and plotting with lightweight
        captures.
    tmp_path : pathlib.Path
        Temporary figure directory.

    Returns
    -------
    None
        Asserts that plotting receives `figure_path` and `sess_id_full` instead
        of a full session object.
    """
    captured = {}
    session = SimpleNamespace(
        figure_path=tmp_path,
        sess_id_full="unit_session",
    )

    def fake_calculate_blocked_holdout_scores(**kwargs):
        captured["random_seed"] = kwargs["random_seed"]
        captured["restart_random_seeds"] = kwargs["restart_random_seeds"]
        n_states = kwargs["states"].size
        return np.zeros((n_states, 1, 2))

    def fake_plot_cross_validation_scores(cv_log_likelihoods, states, figure_path, sess_id_full):
        captured["figure_path"] = figure_path
        captured["sess_id_full"] = sess_id_full
        captured["states"] = states.copy()
        return {"CV_log_likelihood": cv_log_likelihoods, "states": states}

    monkeypatch.setattr(bssm, "calculate_blocked_holdout_scores", fake_calculate_blocked_holdout_scores)
    monkeypatch.setattr(bssm, "plot_cross_validation_scores", fake_plot_cross_validation_scores)

    model_selection = bssm.run_cross_validation(
        make_block_performance_df(),
        session=session,
        random_seed=123,
    )

    assert model_selection["CV_log_likelihood"].shape == (5, 1, 2)
    assert captured["figure_path"] == tmp_path
    assert captured["sess_id_full"] == "unit_session"
    assert captured["random_seed"] == 123
    assert len(captured["restart_random_seeds"]) == 25
    np.testing.assert_array_equal(
        model_selection["restart_random_seeds"],
        np.asarray(captured["restart_random_seeds"], dtype=object).reshape(5, 5),
    )
    np.testing.assert_array_equal(captured["states"], np.arange(1, 6))


def test_single_blocked_holdout_uses_separate_training_sequences(monkeypatch):
    """Blocked holdout should train on disjoint sequence pieces, not glued arrays.

    For a held-out middle block, the HMM training input should be a list of
    separate observation and input sequences for the left and right segments.
    """
    captured_fit_calls: list[tuple[object, object]] = []

    class DummyHMM:
        def fit(self, datas, inputs=None, method=None, num_iters=None, tolerance=None):
            captured_fit_calls.append((datas, inputs))
            return np.array([0.0])

        def log_likelihood(self, datas, inputs=None):
            if isinstance(datas, list):
                return float(sum(len(arr) for arr in datas))
            return float(len(datas))

    monkeypatch.setattr(bssm, "build_input_driven_hmm", lambda *args, **kwargs: DummyHMM())

    observations = np.arange(6, dtype=float).reshape(-1, 1)
    inputs = (np.arange(6, dtype=float) + 10).reshape(-1, 1)
    test_indices_list = [np.array([2, 3])]

    fold_scores = bssm.single_blocked_holdout_func(
        observations=observations,
        inputs=inputs,
        num_states=2,
        test_indices_list=test_indices_list,
        algorithm="MLE",
        n_iter=5,
        tol=1e-4,
    )

    assert fold_scores.shape == (1,)
    assert fold_scores[0] == pytest.approx(1.0)

    train_observations, train_inputs = captured_fit_calls[0]
    assert isinstance(train_observations, list)
    assert isinstance(train_inputs, list)
    assert len(train_observations) == 2
    assert len(train_inputs) == 2
    np.testing.assert_array_equal(train_observations[0], np.array([[0.0], [1.0]]))
    np.testing.assert_array_equal(train_observations[1], np.array([[4.0], [5.0]]))
    np.testing.assert_array_equal(train_inputs[0], np.array([[10.0], [11.0]]))
    np.testing.assert_array_equal(train_inputs[1], np.array([[14.0], [15.0]]))


def test_build_presentation_colors_single_state_avoids_gradient_cmap(monkeypatch):
    """One-state presentation colors should not call `gradient_cmap` on one color."""
    gradient_calls: list[int] = []
    white_calls: list[object] = []

    monkeypatch.setattr(
        plotting_utils,
        "gradient_cmap",
        lambda colors: gradient_calls.append(len(colors)) or "gradient",
    )
    monkeypatch.setattr(
        plotting_utils,
        "white_to_color_cmap",
        lambda color: white_calls.append(color) or "white_to_color",
    )

    colors, cmap = plotting_utils.build_presentation_colors(np.array([0.25]))

    assert len(colors) == 1
    assert cmap == "white_to_color"
    assert gradient_calls == []
    assert len(white_calls) == 1


def test_build_presentation_colors_handles_five_inference_states(monkeypatch):
    """Inference-like states should not be limited by the legacy color list.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces colormap construction so the number of generated colors can be
        inspected without depending on Matplotlib colormap internals.

    Returns
    -------
    None
        Asserts that five inference-like states produce five plotted colors.
    """
    gradient_calls: list[int] = []

    monkeypatch.setattr(
        plotting_utils,
        "gradient_cmap",
        lambda colors: gradient_calls.append(len(colors)) or "gradient",
    )

    colors, cmap = plotting_utils.build_presentation_colors(np.array([0.1, 0.2, 0.3, 0.4, 0.45]))

    assert len(colors) == 5
    assert all(not isinstance(color, (int, np.integer)) for color in colors)
    assert cmap == "gradient"
    assert gradient_calls == [5]


def test_build_presentation_colors_handles_five_rl_states(monkeypatch):
    """RL-like states should not be limited by the legacy color list.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces colormap construction so the number of generated colors can be
        inspected without depending on Matplotlib colormap internals.

    Returns
    -------
    None
        Asserts that five RL-like states produce five plotted colors.
    """
    gradient_calls: list[int] = []

    monkeypatch.setattr(
        plotting_utils,
        "gradient_cmap",
        lambda colors: gradient_calls.append(len(colors)) or "gradient",
    )

    colors, cmap = plotting_utils.build_presentation_colors(np.array([0.5, 0.6, 0.7, 0.8, 0.9]))

    assert len(colors) == 5
    assert all(not isinstance(color, (int, np.integer)) for color in colors)
    assert cmap == "gradient"
    assert gradient_calls == [5]


def test_build_presentation_colors_preserves_mixed_state_count(monkeypatch):
    """Mixed inference/RL state colors should preserve one color per state.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces colormap construction so the number of generated colors can be
        inspected without depending on Matplotlib colormap internals.

    Returns
    -------
    None
        Asserts that mixed five-state weights produce a five-color palette in
        the same length as the input state vector.
    """
    gradient_calls: list[int] = []

    monkeypatch.setattr(
        plotting_utils,
        "gradient_cmap",
        lambda colors: gradient_calls.append(len(colors)) or "gradient",
    )

    colors, cmap = plotting_utils.build_presentation_colors(np.array([0.2, 0.7, 0.1, 0.9, 0.4]))

    assert len(colors) == 5
    assert all(not isinstance(color, (int, np.integer)) for color in colors)
    assert cmap == "gradient"
    assert gradient_calls == [5]


def test_hardcode_block_strategy_single_negative_slope_produces_string_label():
    """Single-state negative slopes should not leave unlabeled object zeros."""
    block_df = pd.DataFrame(
        {
            "trials_to_correct": [3, 4],
            "prev_n_correct": [2, 2],
            "cur_strategy_slope": [-0.75, -0.75],
            "bias_full_flag": ["False", "False"],
        }
    )

    hardcoded = bssm.hardcode_block_strategy(block_df.copy())

    assert hardcoded["hardcoded_strategy"].tolist() == ["Inference", "Inference"]


def test_trials_inherit_strategy_uses_hmm_inferred_strategy():
    """Trial inheritance should propagate HMM states, not inspection labels."""
    block_df = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "inferred_strategy": [0, 1],
            "hardcoded_strategy": ["Inference", "Qlearning"],
            "bias_full_flag": ["False", "True"],
        }
    )
    trial_df = pd.DataFrame({"cur_block": [0, 0, 1]})

    inherited = bssm.trials_inherit_strategy(block_df, trial_df.copy())

    assert inherited["inherited_block_strategy"].tolist() == [0, 0, 1]
    assert inherited["inherited_block_bias"].tolist() == ["False", "False", "True"]


def test_trials_inherit_strategy_uses_block_id_mapping_and_none_default():
    """Trial inheritance should use block ids, not block dataframe row positions."""
    block_df = pd.DataFrame(
        {
            "block_ix": [5, 7],
            "inferred_strategy": [2, 4],
            "hardcoded_strategy": ["Inference", "Qlearning"],
            "bias_full_flag": ["False", "True"],
        }
    )
    trial_df = pd.DataFrame({"cur_block": [5, 7, 99]})

    inherited = bssm.trials_inherit_strategy(block_df, trial_df.copy())

    assert inherited["inherited_block_strategy"].tolist() == [2, 4, "None"]
    assert inherited["inherited_block_bias"].tolist() == ["False", "True", "None"]
