import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from src.behavior_analysis import raster_plots


def test_colorblock_raster_handles_empty_lick_events(tmp_path):
    """Missing lick events should not break block-colored raster plotting."""
    event_df = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "Event": ["enter_right_patch", "enter_left_patch", "enter_right_patch"],
        }
    )

    raster_plots.colorblock_raster(
        event_df,
        fig_name="empty_licks",
        plot_path=tmp_path,
        session_info={"use_dark_period": False},
        plot_choices=False,
    )

    assert (tmp_path / "empty_licks.png").exists()


def test_colorblock_raster_legend_uses_only_present_states(tmp_path, monkeypatch):
    """Context legend should omit dark period when no dark-period state is plotted."""
    event_df = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "Event": ["enter_right_patch", "enter_left_patch", "enter_right_patch"],
        }
    )
    captured = {}
    original_legend = plt.legend

    def capture_legend(*args, **kwargs):
        handles = kwargs.get("handles", [])
        captured["labels"] = [handle.get_label() for handle in handles]
        return original_legend(*args, **kwargs)

    monkeypatch.setattr(plt, "legend", capture_legend)

    raster_plots.colorblock_raster(
        event_df,
        fig_name="present_states",
        plot_path=tmp_path,
        session_info={"use_dark_period": False},
        plot_choices=False,
    )

    assert captured["labels"] == ["right active", "left active"]


def test_colorblock_raster_handles_dark_period_choice_plot(tmp_path):
    """Dark-period state entries should use the raw event array before choice collapsing."""
    event_df = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0],
            "Event": [
                "enter_right_patch",
                "correct_choice_right_patch",
                "enter_dark_period",
                "enter_left_patch",
            ],
        }
    )

    raster_plots.colorblock_raster(
        event_df,
        fig_name="dark_choice",
        plot_path=tmp_path,
        session_info={"use_dark_period": True},
        plot_choices=True,
    )

    assert (tmp_path / "dark_choice.png").exists()


def test_colorblock_raster_raises_when_no_state_entries(tmp_path):
    """A block-colored raster needs at least one block or dark-period entry."""
    event_df = pd.DataFrame(
        {
            "Time": [0.0, 1.0],
            "Event": ["left_entry", "right_entry"],
        }
    )

    with pytest.raises(ValueError, match="state-entry"):
        raster_plots.colorblock_raster(
            event_df,
            fig_name="no_states",
            plot_path=tmp_path,
            session_info={"use_dark_period": False},
            plot_choices=False,
        )


def test_max_event_time_ignores_empty_event_arrays():
    """End-time calculation should ignore empty event rows."""
    event_array = [np.array([]), np.array([2.0]), np.array([])]

    end_time = raster_plots._max_event_time(event_array, timespan=np.array([0.0, 1.0]))

    assert end_time == 2.0
