"""Regression tests for full-block fixed-exemplar inference and inheritance."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import exemplar_hmm_modeling as modeling


@pytest.fixture
def parameters() -> pd.DataFrame:
    """Return a (2, 2) model parameter table with residual SDs in trials."""
    return pd.DataFrame({"name": ["model_a", "model_b"], "residual_sd": [1.0, 4.0]})


@pytest.fixture
def blocks() -> pd.DataFrame:
    """Return four block rows with nonconsecutive IDs/index and trial residuals."""
    return pd.DataFrame(
        {
            "block_ix": [3, 9, 15, 22],
            "bias_full_flag": [True, False, True, False],
            "model_a_exemplar_residual_TTS": [0.0, "None", 8.0, np.inf],
            "model_b_exemplar_residual_TTS": [4.0, 1.0, 0.0, 0.0],
            "model_a_exemplar_normalized_residual_TTS": [101.0, "None", 102.0, 1.0],
            "model_b_exemplar_normalized_residual_TTS": [201.0, 203.0, 202.0, 1.0],
        },
        index=pd.Index([81, 20, 81, 55], name="original_row"),
    )


def test_full_block_result_matches_existing_valid_only_path(blocks, parameters):
    """Full output preserves input rows while valid arrays match legacy scoring."""
    original = blocks.copy(deep=True)
    result = modeling.compute_fixed_exemplar_block_states(blocks, parameters, stay_prob=0.8)
    valid, raw, z, names, sigmas = modeling.prepare_fixed_exemplar_likelihood_inputs(
        blocks.iloc[:3], parameters
    )
    likelihoods = modeling.compute_fixed_exemplar_log_likelihoods(raw, sigmas)
    smoothing = modeling.smooth_fixed_exemplar_likelihoods_with_dynamax(likelihoods, 0.8)
    expected = modeling.add_fixed_exemplar_hmm_outputs(
        valid, names, z, likelihoods, modeling.independent_model_probabilities(likelihoods),
        smoothing.smoothed_probs,
    )
    pd.testing.assert_frame_equal(blocks, original)
    pd.testing.assert_frame_equal(result.block_performance[original.columns], original)
    assert result.valid_mask.tolist() == [True, False, True, False]
    assert result.valid_positions.tolist() == [0, 2]
    assert result.model_names == names
    assert result.stay_prob == 0.8
    pd.testing.assert_frame_equal(result.parameters, parameters)
    np.testing.assert_allclose(result.raw_residuals, raw)
    np.testing.assert_allclose(result.z_residuals, z)
    np.testing.assert_allclose(result.sigmas, sigmas)
    np.testing.assert_allclose(result.log_likelihoods, likelihoods)
    np.testing.assert_allclose(result.smoothing_result.smoothed_probs, smoothing.smoothed_probs)
    for column in expected.columns.difference(original.columns):
        actual = result.block_performance.iloc[result.valid_positions][column].reset_index(drop=True)
        pd.testing.assert_series_equal(actual, expected[column], check_dtype=False)
        assert result.block_performance.iloc[[1, 3]][column].isna().all()
    pd.testing.assert_series_equal(
        result.block_performance["inferred_strategy"],
        result.block_performance["exemplar_hmm_smoothed_state"], check_names=False,
    )


@pytest.mark.parametrize("empty", [False, True])
def test_empty_or_all_invalid_has_missing_outputs(blocks, parameters, empty):
    """No valid rows produce empty numerical arrays and no invented state labels."""
    input_blocks = blocks.iloc[:0] if empty else blocks.iloc[[1, 3]]
    result = modeling.compute_fixed_exemplar_block_states(input_blocks, parameters)
    assert result.valid_positions.shape == (0,)
    assert result.valid_mask.shape == (len(input_blocks),)
    assert not result.valid_mask.any()
    assert result.raw_residuals.shape == (0, 2)
    assert result.smoothing_result.smoothed_probs.shape == (0, 2)
    assert result.smoothing_result.smoothed_states.shape == (0,)
    output_columns = result.block_performance.columns.difference(input_blocks.columns)
    assert "exemplar_hmm_smoothed_state" in output_columns
    assert "inferred_strategy" in output_columns
    assert result.block_performance[output_columns].isna().all().all()
    pd.testing.assert_index_equal(result.block_performance.index, input_blocks.index)


@pytest.mark.parametrize(
    "bad_parameters",
    [
        pd.DataFrame({"name": [], "residual_sd": []}),
        pd.DataFrame({"name": [None], "residual_sd": [1]}),
        pd.DataFrame({"name": [np.nan], "residual_sd": [1]}),
        pd.DataFrame({"name": ["bad name"], "residual_sd": [1]}),
        pd.DataFrame({"name": ["None"], "residual_sd": [1]}),
        pd.DataFrame({"name": ["model_a", "model_a"], "residual_sd": [1, 1]}),
        pd.DataFrame({"name": ["model_a"], "residual_sd": [0]}),
        pd.DataFrame({"name": ["model_a"], "residual_sd": [-1]}),
        pd.DataFrame({"name": ["model_a"], "residual_sd": [np.inf]}),
        pd.DataFrame({"name": ["model_a"], "residual_sd": ["bad"]}),
        pd.DataFrame({"name": ["model_a"]}),
    ],
)
def test_parameter_errors_are_rejected_before_inference(blocks, bad_parameters):
    """Invalid names/scales and missing required parameters fail clearly."""
    with pytest.raises(ValueError):
        modeling.compute_fixed_exemplar_block_states(blocks, bad_parameters)


@pytest.mark.parametrize("stay_prob", [-0.1, 1.1, np.nan])
def test_stay_probability_is_validated_even_without_valid_rows(blocks, parameters, stay_prob):
    """Empty inputs still validate the unitless transition probability."""
    with pytest.raises(ValueError, match="stay_prob"):
        modeling.compute_fixed_exemplar_block_states(blocks.iloc[:0], parameters, stay_prob)


def test_inheritance_maps_explicit_block_ids_and_separate_bias(blocks, parameters):
    """Trials inherit state, name and probability by ID; bias remains independent."""
    result = modeling.compute_fixed_exemplar_block_states(blocks, parameters)
    trials = pd.DataFrame({"cur_block": [15, 3, 9, 99]}, index=[17, 3, 3, 41])
    original = trials.copy(deep=True)
    inherited = modeling.inherit_fixed_exemplar_block_states(result.block_performance, trials)
    pd.testing.assert_frame_equal(trials, original)
    pd.testing.assert_index_equal(inherited.index, trials.index)
    for position, block_position in [(0, 2), (1, 0)]:
        block = result.block_performance.iloc[block_position]
        assert inherited.iloc[position]["inherited_block_strategy"] == block["inferred_strategy"]
        assert inherited.iloc[position]["inherited_block_model"] == block["exemplar_hmm_smoothed_model"]
        for name in result.model_names:
            assert inherited.iloc[position][f"inherited_block_prob_{name}"] == pytest.approx(
                block[f"exemplar_hmm_smoothed_prob_{name}"]
            )
    assert inherited["inherited_block_bias"].iloc[:3].tolist() == [True, True, False]
    assert inherited["inherited_block_strategy"].iloc[2:].isna().all()
    assert pd.isna(inherited.iloc[3]["inherited_block_bias"])


def test_inheritance_distinguishes_same_date_sessions(blocks, parameters):
    """Repeated block IDs are keyed by full source session ID rather than date."""
    repeated = blocks.iloc[[0, 2]].copy()
    repeated["block_ix"] = [3, 3]
    repeated["source_session_id"] = ["mouse_2026-01-01_090000", "mouse_2026-01-01_120000"]
    repeated["source_date"] = ["2026-01-01"] * 2
    result = modeling.compute_fixed_exemplar_block_states(repeated, parameters)
    trials = pd.DataFrame({
        "cur_block": [3, 3],
        "source_session_id": repeated["source_session_id"].iloc[::-1].to_numpy(),
        "source_date": ["2026-01-01"] * 2,
    })
    inherited = modeling.inherit_fixed_exemplar_block_states(result.block_performance, trials)
    assert inherited["inherited_block_strategy"].tolist() == result.block_performance["inferred_strategy"].iloc[::-1].tolist()
    with pytest.raises(ValueError, match="source_session_id"):
        modeling.inherit_fixed_exemplar_block_states(result.block_performance, trials.drop(columns="source_session_id"))


def test_inheritance_rejects_ambiguous_block_keys(blocks, parameters):
    """Duplicate block keys must not silently inherit the last row's state."""
    result = modeling.compute_fixed_exemplar_block_states(blocks.iloc[[0, 2]], parameters)
    result.block_performance["block_ix"] = [3, 3]
    with pytest.raises(ValueError, match="[Dd]uplicat|unique"):
        modeling.inherit_fixed_exemplar_block_states(result.block_performance, pd.DataFrame({"cur_block": [3]}))


def test_multisession_smoothing_is_continuous_across_session_boundaries(parameters):
    """Concatenated inference carries evidence across boundaries instead of reset."""
    frames = []
    for session, raw_a, raw_b in [("first", [0.0, 0.0], [16.0, 16.0]), ("second", [1.7, 1.7], [0.0, 0.0])]:
        frame = pd.DataFrame({"block_ix": [1, 5], "source_session_id": session})
        for name, residual in [("model_a", raw_a), ("model_b", raw_b)]:
            frame[modeling.exemplar_raw_residual_column(name)] = residual
            frame[modeling.exemplar_normalized_residual_column(name)] = residual
        frames.append(frame)
    combined = modeling.compute_fixed_exemplar_block_states(pd.concat(frames), parameters, 0.99)
    reset = modeling.compute_fixed_exemplar_block_states(frames[1], parameters, 0.99)
    assert not np.allclose(combined.smoothing_result.smoothed_probs[2:], reset.smoothing_result.smoothed_probs)
