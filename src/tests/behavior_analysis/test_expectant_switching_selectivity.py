"""Contracts for expectant-model selectivity summaries and plots."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.behavior_analysis import expectant_switching_selectivity as selectivity
from src.behavior_analysis import performance_plots


def _selectivity_trials() -> pd.DataFrame:
    """Return two blocks containing each existing switch-event category."""
    return pd.DataFrame(
        {
            "cur_block": [10, 10, 10, 10, 10, 20, 20, 20, 20],
            "action": [1, 0, 0, 1, 1, 0, 1, 1, 0],
            "reward": [1, 1, 1, 0, 1, 1, 1, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0, 0, 0, 0, 1],
            "prev_action": [0, 1, 0, 0, 1, 1, 0, 1, 1],
            "prev_reward": [1, 1, 1, 0, 1, 1, 1, 0, 1],
            "prev_correct": [True] * 9,
            "switch_trial": [True, True, False, True, False, True, True, False, True],
            "explore_trial": [True, True, False, False, False, True, True, False, True],
            "block_entry_trial": [True, False, False, False, False, True, False, False, False],
            "observer_value": [1, 1, -1, -1, 1, -1, 1, 1, 1],
            "probe_persistence_doubt_value": [1, -1, -1, 1, 1, -1, 1, -1, -1],
            "home_referenced_expectancy_doubt_value": [1, 1, -1, -1, 1, -1, 1, 1, -1],
        }
    )


def test_block_selectivity_preserves_flags_and_excludes_block_entries() -> None:
    """Primary rewarded-switch rates exclude entries without redefining flags."""
    block_table = selectivity.compute_block_model_selectivity(
        _selectivity_trials(),
        min_valid_choices=5,
    )

    probe_rows = block_table.loc[block_table["model"].eq("Probe+P+D")]
    block_10 = probe_rows.loc[probe_rows["block_id"].eq(10)].iloc[0]
    block_20 = probe_rows.loc[probe_rows["block_id"].eq(20)].iloc[0]

    assert block_10["n_valid_choices"] == 5
    assert block_10["n_valid_comparisons"] == 5
    assert block_10["reward_switch_opportunities"] == 3
    assert block_10["reward_switch_events"] == 1
    assert block_10["reward_switch_rate"] == 1 / 3
    assert block_10["ideal_agreement"] == 3 / 5
    assert block_10["model_agreement"] == 1.0
    assert block_10["agreement_delta"] == 2 / 5
    assert bool(block_10["summary_eligible"])

    assert block_20["n_valid_choices"] == 3
    assert block_20["reward_switch_opportunities"] == 1
    assert block_20["reward_switch_events"] == 1
    assert block_20["reward_switch_rate"] == 1.0
    assert not bool(block_20["summary_eligible"])


def test_block_selectivity_can_report_entry_inclusive_counts() -> None:
    """Secondary entry-inclusive counts retain the existing explore flags."""
    block_table = selectivity.compute_block_model_selectivity(
        _selectivity_trials(),
        min_valid_choices=5,
        exclude_block_entry=False,
    )
    probe_rows = block_table.loc[block_table["model"].eq("Probe+P+D")]

    assert probe_rows["reward_switch_opportunities"].tolist() == [4, 2]
    assert probe_rows["reward_switch_events"].tolist() == [2, 2]


def test_event_selectivity_uses_paired_model_and_ideal_outcomes() -> None:
    """Event tables count paired wins, losses, and ties on valid rows only."""
    event_table = selectivity.compute_event_model_selectivity(
        _selectivity_trials()
    )
    probe_rows = event_table.loc[event_table["model"].eq("Probe+P+D")]

    reward_switch = probe_rows.loc[
        probe_rows["event_type"].eq("reward_to_switch")
    ].iloc[0]
    assert reward_switch["n_trials"] == 2
    assert reward_switch["model_only_correct"] == 1
    assert reward_switch["ideal_only_correct"] == 0
    assert reward_switch["both_correct"] == 1
    assert reward_switch["both_incorrect"] == 0
    assert reward_switch["mean_agreement_delta"] == 0.5

    reward_stay = probe_rows.loc[
        probe_rows["event_type"].eq("reward_to_stay")
    ].iloc[0]
    assert reward_stay["n_trials"] == 2
    assert reward_stay["both_correct"] == 2

    omission_switch = probe_rows.loc[
        probe_rows["event_type"].eq("omission_to_switch")
    ].iloc[0]
    assert omission_switch["n_trials"] == 1
    assert omission_switch["model_only_correct"] == 1

    assert int(probe_rows["n_trials"].sum()) == 8


def test_session_selectivity_plot_has_three_aligned_panels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The per-session figure aligns agreement, advantage, and behavior."""
    block_table = selectivity.compute_block_model_selectivity(
        _selectivity_trials(),
        min_valid_choices=5,
    )
    captured = {}

    def capture_figure(fig, save_path):
        captured["fig"] = fig
        captured["save_path"] = save_path

    monkeypatch.setattr(
        performance_plots,
        "save_performance_figure",
        capture_figure,
    )
    save_path = performance_plots.plot_session_expectant_model_selectivity(
        block_table,
        tmp_path,
        "CT016_test",
    )

    assert save_path.name == "CT016_test_expectant_model_selectivity.png"
    assert captured["save_path"] == save_path
    assert tuple(captured["fig"].get_size_inches()) == (24.0, 10.0)
    assert [axis.get_ylabel() for axis in captured["fig"].axes] == [
        "Block agreement",
        "Candidate - ideal",
        "Reward -> switch rate",
    ]
    agreement_labels = {
        line.get_label() for line in captured["fig"].axes[0].lines
    }
    assert {"Ideal", "Probe+P+D", "Home E+D"}.issubset(agreement_labels)
    delta_labels = {line.get_label() for line in captured["fig"].axes[1].lines}
    assert {"Probe+P+D - ideal", "Home E+D - ideal"}.issubset(delta_labels)
    assert captured["fig"].axes[2].get_xlabel() == "Block"
    plt.close(captured["fig"])


def test_cross_session_selectivity_plot_has_one_panel_per_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The cross-session figure compares both candidates with the ideal."""
    first = selectivity.compute_block_model_selectivity(_selectivity_trials())
    first["date"] = "2026-05-25"
    first["session_id"] = "s1"
    second = first.copy()
    second["date"] = "2026-05-29"
    second["session_id"] = "s2"
    second["agreement_delta"] = second["agreement_delta"] + 0.2
    combined = pd.concat([first, second], ignore_index=True)
    captured = {}

    def capture_figure(fig, save_path):
        captured["fig"] = fig
        captured["save_path"] = save_path

    monkeypatch.setattr(
        performance_plots,
        "save_performance_figure",
        capture_figure,
    )
    save_path = performance_plots.plot_cross_session_expectant_model_selectivity(
        combined,
        tmp_path,
        "CT016",
    )

    assert save_path.name == "CT016_expectant_model_selectivity_cross_session.png"
    assert captured["save_path"] == save_path
    model_axes = captured["fig"].axes[:2]
    assert [axis.get_title() for axis in model_axes] == ["Probe+P+D", "Home E+D"]
    assert all(axis.get_xlabel() == "Within-block reward -> switch rate" for axis in model_axes)
    assert model_axes[0].get_ylabel() == "Candidate - ideal agreement"
    plt.close(captured["fig"])
