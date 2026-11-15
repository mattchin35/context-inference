"""Known synthetic population signals survive the complete saved-fit-to-group path."""

import numpy as np

from src.neural_analysis.decoder_unit_activity.selection import GROUP_IDS, build_group_weights, summarize_group_weights, assign_unit_groups
from src.neural_analysis.population_decoding.pipeline import fit_target
from src.neural_analysis.population_decoding.scoring import conditional_outcome_scores


def test_planted_signals_recover_high_group_ranks(synthetic_population):
    """Input seed-81 synthetic population; known context/choice/outcome units retain strong matching weights."""
    cells = {name: fit_target(synthetic_population, "PFC", name) for name in
             ("context", "choice", "outcome", "left_switch_vs_stay", "right_switch_vs_stay")}
    weights = build_group_weights(cells, synthetic_population.regions["PFC"].unit_ids)
    summary = summarize_group_weights(weights, GROUP_IDS, synthetic_population.regions["PFC"].unit_ids)
    for group, unit in ((0, 0), (2, 1), (4, 2), (5, 3), (6, 4), (7, 5)):
        row = summary.loc[(summary.group_index == group) & (summary.unit_index == unit)].iloc[0]
        assert row.candidate and row.percentile > .75
    assigned = assign_unit_groups(summary, 10)
    assert assigned.loc[assigned.displayed, "unit_id"].is_unique


def test_side_only_outcome_probabilities_do_not_imply_reward_discrimination():
    """No inputs; balanced side-only four-class probabilities give chance within-side reward scores."""
    labels = np.tile(np.arange(4), 20)
    probabilities = np.full((len(labels), 4), .01)
    probabilities[np.isin(labels, [0, 2])] = [.49, .01, .49, .01]
    probabilities[np.isin(labels, [1, 3])] = [.01, .49, .01, .49]
    conditional = conditional_outcome_scores(labels, probabilities)
    for side in ("left", "right"):
        assert conditional[side]["balanced_accuracy"] == .5
        assert conditional[side]["roc_auc"][1] == .5
