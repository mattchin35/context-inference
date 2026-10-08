"""Tests for saved-table-only inter-regional regression figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.neural_analysis.interregional import plotting, records


def _summary_tables():
    """Return target and population summaries with one unavailable target."""
    target_rows = []
    population_rows = []
    for condition_index, condition in enumerate(("all", "switch")):
        for direction_index, direction in enumerate(("HPC_to_PFC", "PFC_to_HPC")):
            target_values = [
                0.1 + 0.02 * condition_index + 0.01 * direction_index,
                0.2 + 0.02 * condition_index + 0.01 * direction_index,
            ]
            for target_index, value in enumerate(target_values, start=1):
                target_rows.append(
                    {
                        "session_id": "session",
                        "evaluation_scope": "held_out_cv",
                        "direction": direction,
                        "representation": "units",
                        "model_family": "ols",
                        "condition": condition,
                        "window": "before",
                        "target_id": f"{direction}:unit{target_index}",
                        "metric_name": "delta_r2",
                        "target_rank": None,
                        "status": "ok",
                        "reason": "",
                        "requested_folds": 5,
                        "valid_folds": 5,
                        "mean_value": value,
                    }
                )
                for metric, absolute in (
                    ("r2_restricted", value - 0.04),
                    ("r2_full", value + 0.03),
                ):
                    target_rows.append(
                        {
                            **target_rows[-1],
                            "metric_name": metric,
                            "mean_value": absolute,
                        }
                    )
            target_rows.append(
                {
                    **target_rows[-3],
                    "target_id": f"{direction}:unavailable",
                    "metric_name": "delta_r2",
                    "status": "incomplete_folds",
                    "reason": "incomplete_requested_folds",
                    "valid_folds": 4,
                    "mean_value": None,
                }
            )
            population_rows.append(
                {
                    "session_id": "session",
                    "evaluation_scope": "held_out_cv",
                    "direction": direction,
                    "representation": "units",
                    "model_family": "ols",
                    "condition": condition,
                    "window": "before",
                    "metric_name": "delta_r2",
                    "status": "ok",
                    "reason": "",
                    "n_targets": 2,
                    "q25": target_values[0] * 0.75 + target_values[1] * 0.25,
                    "median": sum(target_values) / 2.0,
                    "q75": target_values[0] * 0.25 + target_values[1] * 0.75,
                }
            )
    return (
        records.result_table_from_rows("target_summaries", target_rows),
        records.result_table_from_rows("population_summaries", population_rows),
    )


def test_increment_plot_shows_targets_median_iqr_grouping_counts_and_caveats() -> None:
    """Primary saved summary plot preserves identities and scientific context."""
    targets, populations = _summary_tables()

    figure, axis = plotting.plot_cv_increment_summary(
        targets,
        populations,
        representation="units",
        model_family="ols",
        window="before",
        coverage_assumption_version="implicit-complete-v1",
    )

    assert axis.get_ylabel() == "Incremental CV R-squared"
    point_ids = {artist.get_gid() for artist in axis.lines if artist.get_gid()}
    assert "HPC_to_PFC:unit1" in point_ids
    assert "PFC_to_HPC:unit2" in point_ids
    assert len(axis.collections) >= 4  # one IQR overlay per condition/direction cell
    condition_centers = axis.get_xticks()
    assert condition_centers[1] - condition_centers[0] > 1.0
    visible_text = " ".join(text.get_text() for text in axis.texts + figure.texts)
    count_annotations = [
        text for text in axis.texts if text.get_gid() == "availability-counts"
    ]
    assert len(count_annotations) == 2
    assert [text.get_position()[0] for text in count_annotations] == list(condition_centers)
    assert all("HPC->PFC: 2 / 1" in text.get_text() for text in count_annotations)
    assert all("PFC->HPC: 2 / 1" in text.get_text() for text in count_annotations)
    assert "contributing / unavailable" in visible_text
    assert "predictive, not causal" in visible_text
    assert "implicit-complete-v1" in visible_text
    assert figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    plt.close(figure)


def test_absolute_plot_uses_saved_restricted_and_full_values_without_refitting() -> None:
    """Inspection figure renders the two supplied absolute CV metrics directly."""
    targets, _ = _summary_tables()

    figure, axis = plotting.plot_absolute_cv_scores(
        targets,
        representation="units",
        model_family="ols",
        condition="all",
        window="before",
        coverage_assumption_version="implicit-complete-v1",
    )

    assert axis.get_ylabel() == "Absolute held-out CV R-squared"
    assert {tick.get_text() for tick in axis.get_xticklabels()} == {
        "Restricted",
        "Full",
    }
    assert len([line for line in axis.lines if line.get_gid()]) == 8
    legend = axis.get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {
        "HPC to PFC",
        "PFC to HPC",
    }
    caption = " ".join(text.get_text() for text in figure.texts)
    assert "No model was refit" in caption
    plt.close(figure)


def test_increment_plot_keeps_dense_target_jitter_inside_its_direction_cell() -> None:
    """Many saved targets must not spill across directions or conditions."""
    targets, populations = _summary_tables()
    template = targets.loc[
        targets["condition"].eq("all")
        & targets["direction"].eq("HPC_to_PFC")
        & targets["metric_name"].eq("delta_r2")
        & targets["status"].eq("ok")
    ].iloc[[0]]
    additions = []
    for target_index in range(80):
        row = template.copy()
        row.loc[:, "target_id"] = f"HPC_to_PFC:dense{target_index:03d}"
        row.loc[:, "mean_value"] = 0.1 + target_index * 0.0001
        additions.append(row)
    dense_targets = records.result_table_from_rows(
        "target_summaries",
        pd.concat([targets, *additions], ignore_index=True).to_dict("records"),
    )

    figure, axis = plotting.plot_cv_increment_summary(
        dense_targets,
        populations,
        representation="units",
        model_family="ols",
        window="before",
        coverage_assumption_version="implicit-complete-v1",
    )

    dense_x = [
        float(artist.get_xdata()[0])
        for artist in axis.lines
        if str(artist.get_gid()).startswith("HPC_to_PFC:dense")
    ]
    assert dense_x
    assert min(dense_x) >= -0.60
    assert max(dense_x) <= -0.10
    plt.close(figure)


def test_increment_plot_centers_a_single_target_in_its_direction_cell() -> None:
    """A one-target cell uses its direction center without arbitrary offset."""
    targets, populations = _summary_tables()
    keep_target = targets["condition"].eq("all") & ~(
        targets["status"].eq("ok") & targets["target_id"].str.endswith("unit2")
    )
    single_targets = records.result_table_from_rows(
        "target_summaries", targets.loc[keep_target].to_dict("records")
    )

    figure, axis = plotting.plot_cv_increment_summary(
        single_targets,
        populations.loc[populations["condition"].eq("all")],
        representation="units",
        model_family="ols",
        window="before",
        coverage_assumption_version="implicit-complete-v1",
    )

    point_positions = {
        artist.get_gid(): float(artist.get_xdata()[0])
        for artist in axis.lines
        if artist.get_gid()
    }
    assert point_positions["HPC_to_PFC:unit1"] == -0.35
    assert point_positions["PFC_to_HPC:unit1"] == 0.35
    plt.close(figure)
