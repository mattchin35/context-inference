import importlib
import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd


def load_block_state_space_modeling_module():
    """Import block LM-HMM module with lightweight optional-dependency stubs.

    Returns
    -------
    module
        Imported `src.behavior_analysis.block_state_space_modeling` module.
    """
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
        ssm_stub.util.rle = lambda states: (np.array([], dtype=int), np.array([], dtype=int))
        ssm_stub.plots = ModuleType("ssm.plots")
        ssm_stub.plots.gradient_cmap = lambda colors: colors
        ssm_stub.plots.white_to_color_cmap = lambda color: color
        sys.modules["ssm"] = ssm_stub
        sys.modules["ssm.util"] = ssm_stub.util
        sys.modules["ssm.plots"] = ssm_stub.plots

    return importlib.import_module("src.behavior_analysis.block_state_space_modeling")


bssm = load_block_state_space_modeling_module()


def make_block_performance_df() -> pd.DataFrame:
    """Build a small block dataframe for LM-HMM fit plumbing tests.

    Returns
    -------
    pd.DataFrame
        Blockwise table with shape `(2, 4)`. Counts are unitless block summary
        values used by `prepare_block_lm_hmm_data`.
    """
    return pd.DataFrame(
        {
            "trials_to_correct": [3, 4],
            "prev_n_correct": [2, 2],
            "prev_n_rewarded": [1, 2],
            "prev_consecutive_rewards": [1, 3],
        }
    )


def make_model_dict() -> dict:
    """Build a synthetic fitted block LM-HMM model dictionary.

    Returns
    -------
    dict
        Model dictionary with MLE and MAP entries. Weights have shape
        `(n_states, obs_dim, n_predictors)` and biases have shape
        `(n_states, obs_dim)`.
    """
    return {
        "mle": {
            "random_seed": 101,
            "hmm_z": np.array([0, 1, 1]),
            "weight_dict": {
                "weights": np.array([[[0.25]], [[0.75]], [[-0.5]]]),
                "mus": np.array([[2.0], [5.0], [7.0]]),
                "label": "mle",
                "weight_labels": ["prev_n_rewarded"],
            },
        },
        "map": {
            "random_seed": 202,
            "hmm_z": np.array([2, 2, 0]),
            "weight_dict": {
                "weights": np.array([[[0.5]], [[1.5]], [[-0.25]]]),
                "mus": np.array([[3.0], [6.0], [9.0]]),
                "label": "map",
                "weight_labels": ["prev_n_rewarded"],
            },
        },
    }


def test_mle_and_map_block_states_forward_predictor_columns(monkeypatch, tmp_path):
    """Custom LM predictor columns should reach both MLE and MAP data prep.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces data preparation and HMM construction with deterministic
        lightweight stubs.
    tmp_path : pathlib.Path
        Temporary figure output directory.

    Returns
    -------
    None
        Asserts that the supplied predictor tuple is forwarded to each fit.
    """
    seen_predictors = []

    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wks = np.array([[[0.25]], [[0.75]]])
        mus = np.array([[2.0], [5.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    def fake_prepare(block_df, predictor_columns=("prev_n_rewarded",)):
        seen_predictors.append(predictor_columns)
        return {
            "observations": np.array([[3], [4]]),
            "inputs": np.array([[1], [3]]),
            "valid_mask": np.array([True, True]),
            "predictor_labels": list(predictor_columns),
        }

    monkeypatch.setattr(bssm, "prepare_block_lm_hmm_data", fake_prepare)
    monkeypatch.setattr(bssm, "build_input_driven_hmm", lambda **_kwargs: DummyHMM())

    bssm.mle_block_states(
        make_block_performance_df(),
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        predictor_columns=("prev_consecutive_rewards",),
    )
    bssm.map_block_states(
        make_block_performance_df(),
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
        model_dict={"mle": {}},
        predictor_columns=("prev_consecutive_rewards",),
    )

    assert seen_predictors == [
        ("prev_consecutive_rewards",),
        ("prev_consecutive_rewards",),
    ]


def test_mle_block_states_stores_weight_labels(monkeypatch, tmp_path):
    """MLE state-feature export needs predictor labels in the model dictionary."""
    class DummyTransitions:
        log_Ps = np.zeros((2, 2))

    class DummyObservations:
        Wks = np.array([[[0.25]], [[0.75]]])
        mus = np.array([[2.0], [5.0]])

    class DummyHMM:
        transitions = DummyTransitions()
        observations = DummyObservations()

        def fit(self, observations, inputs=None, method=None, num_iters=None, tolerance=None):
            return np.array([0.0])

        def most_likely_states(self, observations, input=None):
            return np.array([0, 1])

        def expected_states(self, data=None, input=None):
            return (np.array([[0.8, 0.2], [0.3, 0.7]]),)

    monkeypatch.setattr(bssm, "build_input_driven_hmm", lambda **_kwargs: DummyHMM())

    model_dict, _ = bssm.mle_block_states(
        make_block_performance_df(),
        figure_path=tmp_path,
        sess_id_tag="unit_session",
        plot=False,
    )

    assert model_dict["mle"]["weight_dict"]["weight_labels"] == ["prev_n_rewarded"]


def test_summarize_lm_hmm_state_features_extracts_mle_and_map_rows():
    """State-feature summary should extract per-fit biases, weights, and counts."""
    summary = bssm.summarize_lm_hmm_state_features(
        make_model_dict(),
        session_id="CT999_2026-06-18_120000",
        mouse="CT999",
        date="2026-06-18",
    )

    assert summary.shape[0] == 6
    assert set(summary["fit_type"]) == {"mle", "map"}
    assert summary["state_uid"].is_unique
    map_state_2 = summary[
        (summary["fit_type"] == "map") & (summary["raw_state"] == 2)
    ].iloc[0]
    assert map_state_2["state_uid"] == "CT999_2026-06-18_120000__map__state-2"
    assert map_state_2["state_block_count"] == 2
    assert map_state_2["state_block_fraction"] == 2 / 3
    assert map_state_2["bias"] == 9.0
    assert map_state_2["prev_n_rewarded_weight"] == -0.25
    assert map_state_2["predictor_names"] == "prev_n_rewarded"
    assert map_state_2["num_states"] == 3
    assert map_state_2["n_valid_blocks"] == 3
    assert map_state_2["random_seed"] == 202


def test_summarize_lm_hmm_state_features_retains_zero_count_states():
    """States with zero assigned valid blocks should still appear in the CSV."""
    model_dict = make_model_dict()
    model_dict["map"]["hmm_z"] = np.array([2, 2, 2])

    summary = bssm.summarize_lm_hmm_state_features(
        model_dict,
        session_id="CT999_2026-06-18_120000",
    )

    map_rows = summary[summary["fit_type"] == "map"].sort_values("raw_state")
    assert map_rows["raw_state"].tolist() == [0, 1, 2]
    assert map_rows["state_block_count"].tolist() == [0, 0, 3]


def test_save_block_hmm_state_features_writes_csv(tmp_path):
    """Per-session state-feature helper should save an inspectable CSV."""
    summary = bssm.save_block_hmm_state_features(
        make_model_dict(),
        processed_data_path=tmp_path,
        sess_id_full="CT999_2026-06-18_120000",
        mouse="CT999",
        date="2026-06-18",
    )

    csv_path = tmp_path / "CT999_2026-06-18_120000_block_hmm_state_features.csv"
    assert csv_path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), summary)


def test_load_block_model_dict_loads_saved_pickle(tmp_path):
    """Saved block model pickle should be loadable as the source of truth."""
    model_dict = make_model_dict()
    pickle_path = tmp_path / "CT999_2026-06-18_120000_block_statedict.pkl"
    with open(pickle_path, "wb") as file:
        pickle.dump(model_dict, file)

    loaded = bssm.load_block_model_dict(
        processed_data_path=tmp_path,
        sess_id_full="CT999_2026-06-18_120000",
    )

    assert loaded["map"]["random_seed"] == 202
    np.testing.assert_array_equal(loaded["mle"]["hmm_z"], np.array([0, 1, 1]))


def test_save_block_hmm_state_features_from_saved_model_regenerates_csv(tmp_path):
    """Old sessions should be backfillable from `{sess_id}_block_statedict.pkl`."""
    model_dict = make_model_dict()
    sess_id = "CT999_2026-06-18_120000"
    with open(tmp_path / f"{sess_id}_block_statedict.pkl", "wb") as file:
        pickle.dump(model_dict, file)
    session = SimpleNamespace(
        sess_id_full=sess_id,
        processed_data_path=tmp_path,
        mouse="CT999",
        date="2026-06-18",
    )

    summary = bssm.save_block_hmm_state_features_from_saved_model(session)

    assert summary.shape[0] == 6
    assert (tmp_path / f"{sess_id}_block_hmm_state_features.csv").exists()


def test_collect_block_hmm_state_features_for_sessions_uses_existing_csv_or_regenerates(
    tmp_path,
):
    """Multisession collection should use CSVs and fall back to model pickles."""
    first_processed = tmp_path / "first"
    second_processed = tmp_path / "second"
    output_path = tmp_path / "cross_session"
    first_processed.mkdir()
    second_processed.mkdir()
    output_path.mkdir()
    first_session = SimpleNamespace(
        sess_id_full="CT999_2026-06-18_120000",
        processed_data_path=first_processed,
        mouse="CT999",
        date="2026-06-18",
    )
    second_session = SimpleNamespace(
        sess_id_full="CT999_2026-06-19_120000",
        processed_data_path=second_processed,
        mouse="CT999",
        date="2026-06-19",
    )
    existing_summary = bssm.summarize_lm_hmm_state_features(
        make_model_dict(),
        session_id=first_session.sess_id_full,
        mouse=first_session.mouse,
        date=first_session.date,
    )
    existing_summary.to_csv(
        first_processed / f"{first_session.sess_id_full}_block_hmm_state_features.csv",
        index=False,
    )
    with open(second_processed / f"{second_session.sess_id_full}_block_statedict.pkl", "wb") as file:
        pickle.dump(make_model_dict(), file)

    combined = bssm.collect_block_hmm_state_features_for_sessions(
        sessions=[first_session, second_session],
        output_path=output_path,
        mouse="CT999",
    )

    output_csv = output_path / "CT999_block_hmm_state_features.csv"
    assert output_csv.exists()
    assert combined["session_id"].tolist().count(first_session.sess_id_full) == 6
    assert combined["session_id"].tolist().count(second_session.sess_id_full) == 6


def test_run_block_modeling_saves_state_feature_csv(monkeypatch, tmp_path):
    """Block modeling should save state features after saving the model pickle."""
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        figure_path=tmp_path,
        sess_id_full="unit_session",
        sess_id_abbreviated="unit",
        mouse="CT999",
        date="2026-06-18",
    )
    block_performance = make_block_performance_df()
    augmented_trial_df = pd.DataFrame({"block_id": [0, 1], "trial": [0, 1]})
    model_dict = make_model_dict()
    captured_predictors = {}

    def fake_mle_block_states(block_df, *args, **kwargs):
        captured_predictors["mle"] = kwargs["predictor_columns"]
        return {"mle": model_dict["mle"]}, block_df.assign(inferred_strategy=[0, 1])

    def fake_map_block_states(block_df, *args, **kwargs):
        captured_predictors["map"] = kwargs["predictor_columns"]
        return model_dict, block_df.assign(inferred_strategy=[0, 1])

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
        predictor_columns=("prev_consecutive_rewards",),
    )

    assert captured_predictors == {
        "mle": ("prev_consecutive_rewards",),
        "map": ("prev_consecutive_rewards",),
    }
    assert (tmp_path / "unit_session_block_hmm_state_features.csv").exists()
