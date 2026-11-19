"""Synthetic acceptance tests for choice-level regional decoding."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population_decoding.targets import build_population_targets


@pytest.fixture
def synthetic_decoding():
    """Return numeric prepared inputs for 50 choices, five folds and three units."""
    blocks = np.repeat(np.arange(10), 5)
    context = blocks % 2
    action = np.where(np.arange(len(blocks)) % 5 >= 3, 1 - context, context)
    reward = np.tile([1, 1, 0, 0, 1], 10)
    correct = action == context
    positions = np.arange(len(blocks))
    folds = blocks // 2
    rng = np.random.default_rng(141)
    windows = {}
    for width in (100, 500):
        edges = np.linspace(-1, 1, 2000 // width + 1)
        counts = rng.poisson(1, (len(blocks), len(edges) - 1, 3))
        counts[..., 0] += action[:, None] * 3
        counts[..., 1] += context[:, None] * 4
        counts[:, (len(edges) - 1) // 2:, 2] += reward[:, None] * 5
        splits = tuple(SimpleNamespace(
            fold_id=fold, train_indices=positions[folds != fold],
            test_indices=positions[folds == fold], purged_indices=np.empty(0, dtype=int),
        ) for fold in range(5))
        windows[((-1., 1.), width)] = SimpleNamespace(
            counts={region: counts.copy() for region in ("PFC", "HPC")},
            row_positions=positions, bin_edges_s=edges, splits=splits,
        )
    return SimpleNamespace(
        trials=pd.DataFrame({"cur_trial": positions + 100, "cur_block": blocks,
                             "row_position": positions}),
        targets=build_population_targets(context, action, reward, np.zeros(len(blocks))),
        condition_masks={"overall": np.ones(len(blocks), dtype=bool),
                         "correct_rewarded": correct & (reward == 1),
                         "omission": correct & (reward == 0), "incorrect": ~correct},
        fold_ids=folds, unit_ids={region: ("1", "2", "3") for region in ("PFC", "HPC")},
        windows=windows,
    )


def test_choice_score_averages_class_probabilities_and_requires_all_choices():
    """The left probability mean can disagree with a majority of its bins."""
    from src.neural_analysis.presentation_figures.decoding import score_choices

    left = np.array([[.99, .40, .40, .40], [.2, .2, .2, .2]])
    probabilities = np.stack([1 - left, left], axis=-1)
    labels = np.array([1, 0])
    score = score_choices(probabilities, labels)
    assert score["balanced_accuracy"] == 1.0
    assert score["n_eligible"] == score["n_predicted"] == 2
    assert score_choices(probabilities[:, :2], labels)["balanced_accuracy"] == 1.0
    assert score_choices(probabilities[:, 2:], labels)["balanced_accuracy"] == 0.5
    assert np.isnan(score_choices(probabilities[:1], labels[:1])["balanced_accuracy"])
    probabilities[0, 3] = np.nan
    missing = score_choices(probabilities, labels)
    assert missing["reason"] == "missing_predictions"
    assert missing["n_predicted"] == 1
    assert np.isnan(missing["balanced_accuracy"])


def test_sustained_recovery_ignores_isolated_crossings():
    """Four is the first PC count whose supported suffix stays above .75."""
    from src.neural_analysis.presentation_figures.decoding import sustained_recovery

    assert sustained_recovery(.8, {1: .74, 2: .76, 3: .73, 4: .75, 5: .77}) == 4
    assert sustained_recovery(.8, {1: .74, 2: .77, 3: .72}) is None


@pytest.mark.parametrize("bins", (4, 20))
def test_fold_fit_uses_only_training_rates_and_weights_each_choice_once(monkeypatch, bins):
    """Every width uses choice-normalized weights and held-out-independent transforms."""
    from src.neural_analysis.presentation_figures import decoding

    rng = np.random.default_rng(299)
    rates = rng.poisson(6, size=(8, bins, 3)).astype(float)
    labels = np.array([0, 1, 0, 1, 0, 1])
    observed_weights = []
    fit = decoding.fit_classifier

    def capture(matrix, repeated_labels, weights, classes):
        """Capture dimensionless weights on flattened Hz-derived feature rows."""
        np.testing.assert_array_equal(repeated_labels, np.repeat(labels, bins))
        observed_weights.append(weights.copy())
        return fit(matrix, repeated_labels, weights, classes)

    monkeypatch.setattr(decoding, "fit_classifier", capture)
    expected = decoding.fit_readout(rates[:6], labels, ("u", "v", "w"), (0, 1), (0, 1, 2))
    rates[6:] *= 100
    perturbed = decoding.fit_readout(rates[:6], labels, ("u", "v", "w"), (0, 1), (0, 1, 2))
    np.testing.assert_array_equal(perturbed.scaling.mean_hz, expected.scaling.mean_hz)
    np.testing.assert_array_equal(perturbed.pca.components, expected.pca.components)
    assert len(observed_weights) == 6
    for weights in observed_weights:
        np.testing.assert_allclose(weights.reshape(-1, bins).sum(axis=1), np.ones(len(labels)))


def test_decode_uses_matched_regimes_pcs_and_post_only_outcome(synthetic_decoding):
    """Compute both widths, preserve held-out identities and use ordered classes."""
    from src.neural_analysis.presentation_figures.decoding import compute_decoding

    result = compute_decoding(synthetic_decoding)
    scores = result["scores"]
    probabilities = result["probabilities"]
    assert set(scores.width_ms) == {100, 500}
    assert set(scores.region) == {"PFC", "HPC"}
    assert set(scores.query("target == 'context' and n_pcs == 0").period) == {"full", "pre", "post", "bin"}
    assert set(probabilities.query("region == 'PFC' and target == 'choice'").groupby(
        ["width_ms", "regime"]).row_position.nunique()) == {50}
    full_choice = scores.query("target == 'choice' and period == 'full' and group == 'overall' and n_pcs in [0, 1, 2, 3]")
    assert set(full_choice.n_predicted) == {50}
    assert set(full_choice.n_pcs) == {0, 1, 2, 3}
    assert set(result["folds"].query("target == 'context' and regime == 'correct_rewarded'").n_train) == {16}
    assert set(result["folds"].query("target == 'context'").blocks_0) == {4}
    assert set(result["transfer_mapping"].mapping) == {"same"}
    outcome = probabilities.query("target == 'outcome'")
    assert (outcome.bin_left_s >= 0).all()
    assert set(outcome.groupby("width_ms").bin_index.nunique().items()) == {(100, 10), (500, 2)}
    assert tuple(result["classes"].query("target == 'outcome'").class_name) == (
        "left_reward", "right_reward", "left_no_reward", "right_no_reward",
    )
    confusion = result["confusion"]
    np.testing.assert_allclose(confusion.groupby(["region", "width_ms", "true_class"]).fraction.sum(), 1)
    assert set(confusion.query("true_class == 0").row_count) == {15}
    within_side = result["reward_discrimination"]
    assert set(zip(within_side.n_rewarded, within_side.n_unrewarded)) == {(15, 10)}
    assert within_side.balanced_accuracy.min() > .85


def test_missing_training_class_makes_impacted_pooled_score_unavailable(synthetic_decoding):
    """A failed fold retains its eligible choices in the pooled denominator."""
    from src.neural_analysis.presentation_figures.decoding import compute_decoding

    window = synthetic_decoding.windows[((-1., 1.), 500)]
    window.splits[0].train_indices = window.splits[0].train_indices[
        synthetic_decoding.targets.values[window.splits[0].train_indices, 0] == 0
    ]
    result = compute_decoding(synthetic_decoding)
    score = result["scores"].query("target == 'context' and region == 'PFC' and regime == 'all' "
                                   "and width_ms == 500 and period == 'full' and group == 'overall' and n_pcs == 0").iloc[0]
    assert score.n_eligible == 50
    assert score.n_predicted == 40
    assert score.reason == "missing_predictions"
    assert np.isnan(score.balanced_accuracy)
    assert "training_class_missing" in set(result["folds"].reason)


def test_decoding_tests_all_thirty_supported_pc_prefixes(synthetic_decoding):
    """Use full-rank features to verify scores and support through PC 30."""
    from src.neural_analysis.presentation_figures.decoding import decode_region

    prepared = synthetic_decoding
    window = prepared.windows[((-1., 1.), 500)]
    rng = np.random.default_rng(553)
    window.counts["PFC"] = rng.poisson(8, size=(len(window.row_positions), 4, 32))
    prepared.unit_ids["PFC"] = tuple(str(index) for index in range(32))
    result = decode_region(prepared, "PFC", "context", "all", 500)
    scores = result["scores"].query("group == 'overall' and period == 'full'")
    assert set(scores.n_pcs) == set(range(31))
    assert set(scores.n_predicted) == {len(window.row_positions)}
    overall = result["recovery"].query("group == 'overall'").iloc[0]
    assert overall.max_supported_pc == overall.requested_max_pc == 30
