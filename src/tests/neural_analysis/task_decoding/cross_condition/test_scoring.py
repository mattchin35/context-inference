"""Independent role/outcome trial metrics from small saved probabilities only."""

from dataclasses import replace
from copy import deepcopy

import numpy as np
import pytest

from src.neural_analysis.task_decoding.cross_condition import modeling, scoring


@pytest.fixture
def saved_transfer_models(regional_transfer_data):
    """Return synthetic RegionTransferResult records without fitting or retaining scores.

    Parameters/means have H=3,G=2,S=18 axes and independent regional W/K axes.
    Probabilities are dimensionless, vary with window/weight, and include exactly
    .5 to test the positive threshold. Weights/counts follow actual source cells.
    Coefficients/intercepts are zero logit-unit placeholders: scoring must use
    saved trial probabilities, never reconstruct or fit from neural data.
    """
    inputs, regions = regional_transfer_data
    split_count = len(inputs.transfers)
    models = {}
    for role, (scores, positions) in regions.items():
        source_cells = inputs.source_cells[positions].transpose(1, 0, 2)
        test_cells = inputs.test_cells[positions].transpose(1, 0, 2)
        source, test = source_cells.any(axis=2), test_cells.any(axis=2)
        weights = np.zeros((2, split_count, len(positions)))
        for split in range(split_count):
            weights[0, split, source[split]] = 1
            if source_cells[split].any(axis=0).all():
                for label in (0, 1):
                    selected = source_cells[split, :, label]
                    weights[1, split, selected] = source[split].sum() / (2 * selected.sum())
        probability = np.full((3, 2, split_count, len(positions)), np.nan)
        base = np.array([.5, .1, .8, .3, .9, .2, .6, .4, .7, .15, .85, .25, .75, .35, .65, .05])[positions]
        for window in range(3):
            for weighting in range(2):
                values = np.roll(base, window)
                values = values if weighting == 0 else 1 - values
                for split in range(split_count):
                    selected = source[split] | test[split]
                    probability[window, weighting, split, selected] = values[selected]
        coefficients = np.zeros((3, 2, split_count, scores.shape[2]))
        intercepts = np.zeros((3, 2, split_count))
        status = np.full((3, 2, split_count), "ok", dtype="U11")
        reason = np.full((3, 2, split_count), "", dtype="U160")
        for split in range(split_count):
            if not source_cells[split].any(axis=0).all():
                coefficients[:, :, split] = intercepts[:, :, split] = np.nan
                probability[:, :, split] = np.nan
                status[:, :, split] = "unavailable"
                reason[:, :, split] = "single_class_training"
        models[role] = modeling.RegionTransferResult(
            choice_positions=positions.copy(), pc_ids=np.arange(1, scores.shape[2] + 1),
            coefficients=coefficients, intercepts=intercepts, fit_status=status, fit_reason=reason,
            trial_probabilities=probability, source_mask=source, test_mask=test,
            source_trial_weights=weights,
            eligible_cell_counts=np.stack([source_cells.sum(axis=1), test_cells.sum(axis=1)], axis=1),
        )
    return models


def find_row(rows, **selectors):
    """Return the unique summary dict matching dimensionless scalar identity fields."""
    matches = [row for row in rows if all(row[key] == value for key, value in selectors.items())]
    assert len(matches) == 1
    return matches[0]


def independent_populations(original_trials, positions, target_name):
    """Return bool(W,) outcome masks from original numeric labels, independent of product masks.

    original_trials is chronological (J,column), positions indexes its rows, and
    target_name is a fixed identifier. Inputs/labels are dimensionless; no
    baseline or role selection is inferred here. Those intersections occur later.
    """
    trial = original_trials.iloc[positions]
    correct, reward = trial.correct.to_numpy(), trial.reward.to_numpy()
    omission = (correct == 1) & (reward == 0)
    incorrect = correct == 0
    if target_name == "current_reward":
        return {"all": np.ones(len(positions), dtype=bool), "actual_rewarded": reward == 1,
                "actual_unrewarded": omission | (incorrect & (reward == 0)),
                "omission": omission, "incorrect_unrewarded": incorrect & (reward == 0)}
    return {"all": np.ones(len(positions), dtype=bool), "correct": (correct == 1) & np.isin(reward, [0, 1]),
            "correct_rewarded": (correct == 1) & (reward == 1), "omission": omission, "incorrect": incorrect}


def test_all_summary_rows_match_independent_role_population_and_metric_oracles(
    regional_transfer_data, saved_transfer_models, transfer_trials,
):
    """Every H/G/S/region/role/outcome row uses finite trial means and equal evaluation weight.

    Fractions come from direct numpy comparisons, not product metric helpers.
    Class counts sum scored n_trials; eligible_trials counts the exact role/outcome
    intersection before unavailable/finite-output filtering. Never pool roles.
    """
    inputs, _ = regional_transfer_data
    report = scoring.summarize_transfers(inputs, saved_transfer_models, {})
    assert isinstance(report, list) and len(report) == 3 * 2 * 18 * 2 * 5 * 2
    identity_fields = ("target_name", "split_id", "window", "weighting", "region", "role", "population")
    assert len({tuple(row[key] for key in identity_fields) for row in report}) == len(report)
    for row in report:
        model = saved_transfer_models[row["region"]]
        positions = model.choice_positions
        split = next(index for index, item in enumerate(inputs.transfers) if item.split_id == row["split_id"])
        assert row["target_name"] == inputs.transfers[split].target_name
        target = inputs.target_names.index(row["target_name"])
        window = inputs.window_names.index(row["window"])
        weighting = ("equal_trial", "equal_cell").index(row["weighting"])
        training_counts = model.eligible_cell_counts[split, 0]
        assert row["training_trials"] == training_counts.sum()
        assert row["training_class_0_trials"] == training_counts[0]
        assert row["training_class_1_trials"] == training_counts[1]
        role_cells = inputs.source_cells if row["role"] == "training" else inputs.test_cells
        assert row["role"] in ("training", "complementary_test")
        population = independent_populations(transfer_trials, inputs.choices.row_position.to_numpy()[positions], row["target_name"])
        eligible = role_cells[positions, split].any(axis=1) & population[row["population"]]
        probabilities = model.trial_probabilities[window, weighting, split]
        scored = eligible & np.isfinite(probabilities)
        actual = inputs.target_values[positions, target][scored]
        correct = (probabilities[scored] >= .5) == actual
        assert row["eligible_trials"] == eligible.sum()
        assert row["n_trials"] == scored.sum()
        assert row["class_0_trials"] == (actual == 0).sum()
        assert row["class_1_trials"] == (actual == 1).sum()
        if model.fit_status[window, weighting, split] != "ok":
            assert row["accuracy"] is row["balanced_accuracy"] is None
            assert row["reason"] == model.fit_reason[window, weighting, split]
            continue
        if len(actual):
            assert row["accuracy"] == pytest.approx(correct.mean())
        else:
            assert row["accuracy"] is None
        if np.unique(actual).size == 2:
            expected_balanced = (correct[actual == 0].mean() + correct[actual == 1].mean()) / 2
            assert row["balanced_accuracy"] == pytest.approx(expected_balanced)
            assert row["reason"] == ""
        else:
            assert row["balanced_accuracy"] is None
            assert row["reason"] == ("single_class" if len(actual) else "empty_population")


def test_outcome_rows_overlap_without_borrowing_from_the_other_role(regional_transfer_data, saved_transfer_models):
    """Correct-to-incorrect source: all5=correct5=rewarded3+omission2; incorrect source stays empty."""
    inputs, _ = regional_transfer_data
    rows = scoring.summarize_transfers(inputs, saved_transfer_models, {})
    common = dict(region="PFC", target_name="current_state", split_id="context_correct_to_incorrect",
                  window="full", weighting="equal_trial")
    for population, count in (("all", 5), ("correct", 5), ("correct_rewarded", 3), ("omission", 2), ("incorrect", 0)):
        assert find_row(rows, **common, role="training", population=population)["n_trials"] == count
    assert find_row(rows, **common, role="complementary_test", population="incorrect")["n_trials"] == 5
    assert find_row(rows, **common, role="complementary_test", population="omission")["n_trials"] == 0


def test_reward_and_value_omission_rows_keep_opposite_actual_classes(regional_transfer_data, saved_transfer_models):
    """Left-choice source omission ID1 has Reward0 versus Value1; both single-class accuracies remain valid."""
    inputs, _ = regional_transfer_data
    rows = scoring.summarize_transfers(inputs, saved_transfer_models, {})
    common = dict(region="PFC", window="full", weighting="equal_trial", role="training", population="omission")
    reward = find_row(rows, **common, target_name="current_reward", split_id="reward_left_choice_to_right_choice")
    value = find_row(rows, **common, target_name="current_action_is_correct", split_id="value_left_choice_to_right_choice")
    assert reward["n_trials"] == reward["class_0_trials"] == 1 and reward["class_1_trials"] == 0
    assert value["n_trials"] == value["class_1_trials"] == 1 and value["class_0_trials"] == 0
    for row in (reward, value):
        assert row["accuracy"] is not None and row["balanced_accuracy"] is None
        assert row["reason"] == "single_class"


def test_scoring_uses_saved_outputs_without_refitting_or_training_weighted_evaluation(
    regional_transfer_data, saved_transfer_models, monkeypatch,
):
    """Always-positive predictions score3/5, not fitting-weighted1/2, under both saved schemes."""
    inputs, _ = regional_transfer_data
    model = saved_transfer_models["PFC"]
    probabilities = model.trial_probabilities.copy()
    probabilities[0, :, 0, model.source_mask[0]] = .9
    models = {"PFC": replace(model, trial_probabilities=probabilities)}
    original_probabilities = probabilities.copy()

    def forbidden(*args, **kwargs):
        """Fail if the saved-only scoring operation attempts any readout or regional fit."""
        raise AssertionError("scoring must use saved trial probabilities")

    monkeypatch.setattr(modeling, "fit_readout", forbidden)
    monkeypatch.setattr(modeling, "fit_region_transfers", forbidden)
    rows = scoring.summarize_transfers(inputs, models, {})
    for weighting in ("equal_trial", "equal_cell"):
        row = find_row(rows, region="PFC", target_name="current_state", split_id="context_correct_to_incorrect",
                       window="full", weighting=weighting, role="training", population="all")
        assert row["accuracy"] == .6 and row["balanced_accuracy"] == .5
    np.testing.assert_array_equal(models["PFC"].trial_probabilities, original_probabilities)


def test_unavailable_fit_retains_eligible_population_but_has_zero_scored_trials(regional_transfer_data, saved_transfer_models):
    """One unavailable model carries its fit reason into every role/outcome without erasing eligibility."""
    inputs, _ = regional_transfer_data
    model = saved_transfer_models["PFC"]
    parameters, intercepts = model.coefficients.copy(), model.intercepts.copy()
    probabilities, status, reason = model.trial_probabilities.copy(), model.fit_status.copy(), model.fit_reason.copy()
    parameters[0, 0, 0] = np.nan
    intercepts[0, 0, 0] = np.nan
    probabilities[0, 0, 0] = np.nan
    status[0, 0, 0], reason[0, 0, 0] = "unavailable", "nonconvergence"
    failed = replace(model, coefficients=parameters, intercepts=intercepts, trial_probabilities=probabilities,
                     fit_status=status, fit_reason=reason)
    rows = scoring.summarize_transfers(inputs, {"PFC": failed}, {})
    selected = [row for row in rows if row["split_id"] == "context_correct_to_incorrect"
                and row["window"] == "full" and row["weighting"] == "equal_trial"]
    assert len(selected) == 10
    for row in selected:
        assert (row["training_trials"], row["training_class_0_trials"], row["training_class_1_trials"]) == (5, 2, 3)
        assert row["n_trials"] == row["class_0_trials"] == row["class_1_trials"] == 0
        assert row["accuracy"] is None and row["balanced_accuracy"] is None
        assert row["reason"] == "nonconvergence"
    for role in ("training", "complementary_test"):
        row = find_row(selected, role=role, population="all")
        assert row["eligible_trials"] == 5
    np.testing.assert_array_equal(failed.eligible_cell_counts, model.eligible_cell_counts)


def test_only_finite_trial_means_are_scored_without_changing_eligible_count(regional_transfer_data, saved_transfer_models):
    """An unavailable individual saved mean is filtered before trial_metrics; its role stays eligible."""
    inputs, _ = regional_transfer_data
    model = saved_transfer_models["PFC"]
    probabilities = model.trial_probabilities.copy()
    probabilities[0, 0, 0, 0] = np.nan
    rows = scoring.summarize_transfers(inputs, {"PFC": replace(model, trial_probabilities=probabilities)}, {})
    row = find_row(rows, region="PFC", target_name="current_state", split_id="context_correct_to_incorrect",
                   window="full", weighting="equal_trial", role="training", population="all")
    assert row["eligible_trials"] == 5 and row["n_trials"] == 4
    assert row["class_0_trials"] == row["class_1_trials"] == 2


def test_missing_test_class_keeps_accuracy_without_balanced_accuracy(regional_transfer_data):
    """Only the complementary class-0 IDs4/6 lose coverage; two-class source model still fits."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    retained = ~inputs.choices.trial_id.isin([4, 6]).to_numpy()
    model = modeling.fit_region_transfers(scores[retained], positions[retained], inputs)
    assert model.fit_status[0, 0, 0] == "ok"
    rows = scoring.summarize_transfers(inputs, {"PFC": model}, {})
    row = find_row(rows, region="PFC", target_name="current_state", split_id="context_correct_to_incorrect",
                   window="full", weighting="equal_trial", role="complementary_test", population="all")
    assert row["n_trials"] == row["class_1_trials"] == 3 and row["class_0_trials"] == 0
    assert row["accuracy"] is not None and row["balanced_accuracy"] is None and row["reason"] == "single_class"


def test_below_chance_scores_are_not_reoriented_or_clipped(regional_transfer_data, saved_transfer_models):
    """Wrong-orientation saved Context probabilities keep zero accuracy and balanced accuracy."""
    inputs, _ = regional_transfer_data
    model = saved_transfer_models["PFC"]
    probabilities = model.trial_probabilities.copy()
    selected = model.source_mask[0] | model.test_mask[0]
    probabilities[0, 0, 0, selected] = .9 - .8 * inputs.target_values[selected, 0]
    rows = scoring.summarize_transfers(inputs, {"PFC": replace(model, trial_probabilities=probabilities)}, {})
    row = find_row(rows, region="PFC", target_name="current_state", split_id="context_correct_to_incorrect",
                   window="full", weighting="equal_trial", role="complementary_test", population="all")
    assert row["accuracy"] == row["balanced_accuracy"] == 0


@pytest.mark.parametrize("both_unavailable", [False, True])
def test_unavailable_regions_remain_visible_without_invented_coverage(
    regional_transfer_data, saved_transfer_models, both_unavailable,
):
    """One/both unavailable regions retain reasons; unknown eligible coverage is None, scored count0."""
    inputs, _ = regional_transfer_data
    reasons = {"HPC": "regional source could not be loaded"}
    models = {"PFC": saved_transfer_models["PFC"]}
    if both_unavailable:
        reasons["PFC"] = "no compatible regional PCA"
        models = {}
    rows = scoring.summarize_transfers(inputs, models, reasons)
    assert {row["region"] for row in rows} == {"PFC", "HPC"}
    for region, reason in reasons.items():
        selected = [row for row in rows if row["region"] == region]
        assert len(selected) == 3 * 2 * 18 * 2 * 5
        for row in selected:
            assert row["training_trials"] is row["training_class_0_trials"] is row["training_class_1_trials"] is None
            assert row["eligible_trials"] is None and row["n_trials"] == 0
            assert row["class_0_trials"] == row["class_1_trials"] == 0
            assert row["accuracy"] is None and row["balanced_accuracy"] is None
            assert row["reason"] == reason


# Literal populations retain recorded context, including rewarded/unknown-reward incorrect trials.
OUTCOME_MODELS = (
    ("context_correct_rewarded_to_other_outcomes", "correct_rewarded", (0, 2, 19),
     (("omission", (1, 3)), ("incorrect", (4, 5, 6, 7, 9)))),
    ("context_omission_to_other_outcomes", "omission", (1, 3),
     (("correct_rewarded", (0, 2, 19)), ("incorrect", (4, 5, 6, 7, 9)))),
    ("context_incorrect_to_correct", "incorrect", (4, 5, 6, 7, 9),
     (("correct_rewarded", (0, 2, 19)), ("omission", (1, 3)))),
)


def test_training_count_summary_requires_new_scoring_version():
    """Additive source counts and fixed outcome-selection semantics invalidate the old summary cache."""
    assert scoring.SCORING_VERSION == "cross-condition-summary-v2"


def test_context_outcome_rows_select_six_pairs_and_three_diagnostics_without_recomputation(
    regional_transfer_data, saved_transfer_models, transfer_trials, monkeypatch,
):
    """Select 12 test/6 training rows per H/G using literal IDs, direct metrics and saved source counts.

    The helper consumes shuffled list[dict] summaries only, preserves every
    original field, adds population annotations, and leaves the input unchanged.
    Source counts are integer unique trials after coverage, independent of test
    population and weighting; all metrics are equal-trial dimensionless fractions.
    """
    inputs, _ = regional_transfer_data
    summaries = scoring.summarize_transfers(inputs, saved_transfer_models, {})
    summaries.reverse()
    original = deepcopy(summaries)

    def forbidden(*args, **kwargs):
        """Reject fitting, rescoring or source-file reads during pure summary selection."""
        raise AssertionError("context outcome selection must only select saved summary rows")

    monkeypatch.setattr(modeling, "fit_readout", forbidden)
    monkeypatch.setattr(modeling, "fit_region_transfers", forbidden)
    monkeypatch.setattr(scoring, "summarize_transfers", forbidden)
    monkeypatch.setattr(scoring, "trial_metrics", forbidden)
    with monkeypatch.context() as read_guard:
        read_guard.setattr("builtins.open", forbidden)
        rows = scoring.context_outcome_rows(summaries)
    assert summaries == original
    assert len(rows) == 3 * 2 * 18
    assert {row["split_id"] for row in rows} == {item[0] for item in OUTCOME_MODELS}
    for window in ("full", "before", "after"):
        for weighting in ("equal_trial", "equal_cell"):
            selected = [row for row in rows if row["window"] == window and row["weighting"] == weighting]
            for role in ("complementary_test", "training"):
                role_rows = [row for row in selected if row["role"] == role]
                expected_order = []
                for split_id, training, _, tests in OUTCOME_MODELS:
                    populations = tests if role == "complementary_test" else ((training, ()),)
                    expected_order.extend((split_id, population, region)
                                          for population, _ in populations for region in ("PFC", "HPC"))
                assert [(row["split_id"], row["population"], row["region"]) for row in role_rows] == expected_order
            for row in selected:
                split_id, training, source_ids, tests = next(item for item in OUTCOME_MODELS if item[0] == row["split_id"])
                assert row["training_population"] == training
                assert row["test_population"] == (row["population"] if row["role"] == "complementary_test" else None)
                prior = find_row(original, split_id=split_id, window=window, weighting=weighting,
                                 region=row["region"], role=row["role"], population=row["population"])
                assert row == {**prior, "training_population": training, "test_population": row["test_population"]}
                model = saved_transfer_models[row["region"]]
                original_positions = inputs.choices.row_position.to_numpy()[model.choice_positions]
                ids = inputs.choices.trial_id.to_numpy()[model.choice_positions]
                context = transfer_trials.iloc[original_positions].state_int.to_numpy()
                source = np.isin(ids, source_ids)
                assert (row["training_trials"], row["training_class_0_trials"], row["training_class_1_trials"]) == (
                    int(source.sum()), int((source & (context == 0)).sum()), int((source & (context == 1)).sum()),
                )
                outcome_ids = source_ids if row["role"] == "training" else dict(tests)[row["population"]]
                eligible = np.isin(ids, outcome_ids)
                assert row["eligible_trials"] == int(eligible.sum())
                if row["region"] == "HPC" and training == "correct_rewarded":
                    assert row["n_trials"] == 0 and row["accuracy"] is None
                    assert row["reason"] == "single_class_training"
                    continue
                # Independent fixture means, not a scoring/model helper or a reconstructed affine model.
                base = np.array([.5, .1, .8, .3, .9, .2, .6, .4, .7, .15, .85, .25, .75, .35, .65, .05])
                probabilities = np.roll(base[model.choice_positions], ("full", "before", "after").index(window))
                if weighting == "equal_cell":
                    probabilities = 1 - probabilities
                actual = context[eligible]
                correct = (probabilities[eligible] >= .5) == actual
                assert row["n_trials"] == len(actual)
                assert row["class_0_trials"] == int((actual == 0).sum())
                assert row["class_1_trials"] == int((actual == 1).sum())
                assert row["accuracy"] == pytest.approx(correct.mean())
                if np.unique(actual).size == 2:
                    assert row["balanced_accuracy"] == pytest.approx(
                        (correct[actual == 0].mean() + correct[actual == 1].mean()) / 2,
                    )
                else:
                    assert row["balanced_accuracy"] is None and row["reason"] == "single_class"


@pytest.mark.parametrize("both_unavailable", [False, True])
def test_context_outcome_selection_keeps_unavailable_region_rows_and_unknown_training_counts(
    regional_transfer_data, saved_transfer_models, both_unavailable,
):
    """The six pairs retain both regional rows/reasons even when no regional coverage is known."""
    inputs, _ = regional_transfer_data
    unavailable = {"HPC": "missing regional PCA"}
    models = {"PFC": saved_transfer_models["PFC"]}
    if both_unavailable:
        unavailable["PFC"] = "regional source failed"
        models = {}
    rows = scoring.context_outcome_rows(scoring.summarize_transfers(inputs, models, unavailable))
    assert len(rows) == 3 * 2 * 18
    for row in rows:
        if row["region"] in unavailable:
            assert row["training_trials"] is row["training_class_0_trials"] is row["training_class_1_trials"] is None
            assert row["eligible_trials"] is None and row["n_trials"] == 0
            assert row["accuracy"] is row["balanced_accuracy"] is None
            assert row["reason"] == unavailable[row["region"]]


def test_known_empty_context_source_counts_are_zero_not_unknown(regional_transfer_data):
    """Removing all rewarded source IDs leaves a known empty source and still-known test eligibility."""
    inputs, regions = regional_transfer_data
    scores, positions = regions["PFC"]
    keep = ~inputs.choices.trial_id.isin([0, 2, 19]).to_numpy()
    model = modeling.fit_region_transfers(scores[keep], positions[keep], inputs)
    rows = scoring.summarize_transfers(inputs, {"PFC": model}, {})
    selected = [row for row in rows if row["split_id"] == "context_correct_rewarded_to_other_outcomes"]
    assert len(selected) == 3 * 2 * 2 * 5
    for row in selected:
        assert (row["training_trials"], row["training_class_0_trials"], row["training_class_1_trials"]) == (0, 0, 0)
        assert row["n_trials"] == 0 and row["reason"] == "empty_training"
    test = find_row(selected, window="full", weighting="equal_trial", role="complementary_test", population="all")
    assert test["eligible_trials"] == 7
