"""Headless render and live-file verification for saved regional results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.tests.neural_analysis.presentation_figures.test_decoding import synthetic_decoding


@pytest.fixture
def regional_fields(synthetic_decoding):
    """Return all Figures 1-3 named tables from numeric synthetic inputs."""
    from src.neural_analysis.presentation_figures.decoding import compute_decoding

    return compute_decoding(synthetic_decoding)


def test_regional_dot_bin_and_outcome_views_draw_held_out_values(regional_fields):
    """Inspect a below-chance score, true bin clock and normalized four-state matrix."""
    from src.neural_analysis.presentation_figures.regional_plotting import render_regional

    scores = regional_fields["scores"]
    mask = (scores.region.eq("PFC") & scores.target.eq("context") & scores.regime.eq("all")
            & scores.period.eq("full") & scores.group.eq("incorrect") & scores.width_ms.eq(500)
            & scores.n_pcs.eq(0))
    scores.loc[mask, "balanced_accuracy"] = .25
    figures = render_regional(regional_fields, {"session_id": "synthetic-25"}, "light")
    try:
        for key in ("figure_1_pfc", "figure_2_hpc", "figure_3_outcome", "figure_3_pc",
                    "figure_1_pfc_100ms", "figure_1_pfc_bins_100ms"):
            figures[key].canvas.draw()
        main = figures["figure_1_pfc"]
        panel = next(axis for axis in main.axes if axis.get_gid() == "all:context")
        assert panel.get_ylim() == (0., 1.)
        dots = next(artist for artist in panel.collections if artist.get_gid() == "incorrect")
        assert dots.get_offsets()[0, 1] == .25
        assert any(np.allclose(line.get_ydata(), .5) for line in panel.lines)
        assert any("synthetic-25" in text.get_text() for text in main.texts)
        bin_panel = next(axis for axis in figures["figure_1_pfc_bins_100ms"].axes
                         if axis.get_gid() == "all:context")
        curve = next(line for line in bin_panel.lines if line.get_gid() == "overall")
        np.testing.assert_allclose(curve.get_xdata(), np.arange(-.95, 1., .1))
        matrix = next(axis for axis in figures["figure_3_outcome"].axes if axis.get_gid() == "PFC")
        assert matrix.images[0].get_array().shape == (4, 4)
        assert matrix.images[0].get_clim() == (0., 1.)
        assert any("n=15" in label.get_text() for label in matrix.get_yticklabels())
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_context_caption_reports_independent_training_blocks(regional_fields):
    """Both training regimes report context blocks rather than trial counts."""
    from src.neural_analysis.presentation_figures.regional_plotting import render_regional

    figures = render_regional(regional_fields, {"session_id": "synthetic-25"}, "light")
    try:
        caption = " ".join(" ".join(text.get_text().split()) for text in figures["figure_1_pfc"].texts)
        assert "Independent context training blocks per fold" in caption
        assert "All outcomes (4 right, 4 left)" in caption
        assert "Correct rewarded (4 right, 4 left)" in caption
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_cohort_draws_named_paired_sessions_and_reports_missing(roster_data):
    """Never connect scores from independent sessions or missing endpoints."""
    from src.neural_analysis.presentation_figures.regional_plotting import render_regional_cohort

    figures = render_regional_cohort(roster_data, {"subject_id": "mouse-25"}, "light")
    try:
        main = figures["figure_7_context"]
        main.canvas.draw()
        panel = next(axis for axis in main.axes if axis.get_gid() == "PFC:incorrect")
        pairs = {line.get_gid(): line for line in panel.lines}
        np.testing.assert_allclose(pairs["day-1"].get_ydata(), [.4, .6])
        np.testing.assert_allclose(pairs["day-2"].get_ydata(), [.5, np.nan], equal_nan=True)
        report = " ".join(element.get_text() for element in (*main.texts, *panel.texts))
        for fragment in ("mouse-25", "day-3", "missing spikes", "2/3", "1/3"):
            assert fragment in report
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_cohort_with_no_saved_sessions_renders_reasons(roster_data):
    """A cohort with no regional score table can still expose the roster."""
    from src.neural_analysis.presentation_figures.regional_plotting import render_regional_cohort

    figures = render_regional_cohort({"roster": roster_data["roster"].iloc[2:]},
                                     {"subject_id": "mouse-25"}, "light")
    try:
        figure = figures["figure_7_context"]
        figure.canvas.draw()
        report = " ".join(element.get_text() for element in figure.texts)
        assert "day-3" in report and "missing spikes" in report
    finally:
        for figure in figures.values():
            plt.close(figure)


@pytest.mark.parametrize("limit", [10, 30])
def test_pc_figures_preserve_legacy_ranges_and_show_extended_recovery(regional_fields, limit):
    """Display saved PC counts and recovery labels for legacy and extended archives."""
    from src.neural_analysis.presentation_figures.regional_plotting import (
        render_regional, render_regional_cohort,
    )

    regional_fields["recovery"] = regional_fields["recovery"].drop(columns="requested_max_pc", errors="ignore")
    if limit == 30:
        regional_fields["recovery"]["requested_max_pc"] = 30
        scores = regional_fields["scores"]
        chosen = scores.query("width_ms == 500 and region == 'PFC' and target == 'context' "
                              "and regime == 'all' and group == 'overall' and n_pcs == 3")
        regional_fields["scores"] = pd.concat([scores, chosen.assign(n_pcs=30)], ignore_index=True)
        recovery = regional_fields["recovery"]
        recovery.loc[recovery.index[0], ["recovery_count", "max_supported_pc", "status"]] = [30, 30, "reached"]

    session = render_regional(regional_fields, {"session_id": "day-1"}, "light")
    cohort_fields = {key: table.assign(session_id="day-1") for key, table in regional_fields.items()}
    cohort_fields["roster"] = pd.DataFrame({"session_id": ["day-1"], "reason": [""]})
    cohort = render_regional_cohort(cohort_fields, {"subject_id": "mouse-25"}, "light")
    try:
        for figure in (session["figure_3_pc"], cohort["figure_7_pc_curves"]):
            panel = next(axis for axis in figure.axes if axis.get_gid() == "PFC:all:context")
            assert max(panel.get_xticks()) == limit
            assert panel.get_xlim()[1] > limit
            if limit == 30:
                assert any(30 in line.get_xdata() for line in panel.lines)
        if limit == 30:
            for figure in (session["figure_3_pc"], cohort["figure_7_pc_recovery"]):
                panel = next(axis for axis in figure.axes if axis.get_gid() == "PFC:all:context")
                assert any("30 PCs through 30" in artist.get_text() for artist in panel.texts)
    finally:
        for figure in (*session.values(), *cohort.values()):
            plt.close(figure)


@pytest.fixture
def roster_data(regional_fields):
    """Return two named synthetic sessions and one unavailable roster member."""
    data = {key: pd.concat([value.assign(session_id=name) for name in ("day-1", "day-2")],
                           ignore_index=True)
            for key, value in regional_fields.items() if key != "probabilities"}
    scores = data["scores"]
    selected = (scores.region.eq("PFC") & scores.target.eq("context") & scores.period.eq("full")
                & scores.group.eq("incorrect") & scores.width_ms.eq(500) & scores.n_pcs.eq(0))
    for session, regime, value in (("day-1", "all", .4), ("day-1", "correct_rewarded", .6),
                                   ("day-2", "all", .5), ("day-2", "correct_rewarded", np.nan)):
        row = selected & scores.session_id.eq(session) & scores.regime.eq(regime)
        scores.loc[row, "balanced_accuracy"] = value
        scores.loc[row, "reason"] = "missing_predictions" if np.isnan(value) else ""
    data["roster"] = pd.DataFrame(dict(session_id=["day-1", "day-2", "day-3"],
                                       status=["ok", "ok", "unavailable"],
                                       reason=["", "", "missing spikes"]))
    return data


def test_live_file_replay_uses_recorded_ephys_events_and_estimated_probabilities(synthetic_presentation_session):
    """Load fixture spike counts and event files through the production session preparation."""
    from src.neural_analysis.presentation_figures.preparation import prepare_choices
    from src.neural_analysis.presentation_figures.replays import compute_replays

    path = synthetic_presentation_session
    trials = pd.read_csv(path.parent / "trials.csv")
    events = []
    for _, block in trials.groupby("cur_block", sort=False):
        first = block.iloc[0]
        side = "left" if first.cur_block % 2 else "right"
        events.append((first.start_time - .1, f"enter_{side}_patch"))
    for row in trials.loc[trials.reward.eq(1) & trials.experimenter_reward_given.eq(0)].itertuples(index=False):
        side_pump = 2 if row.action == 1 else 3
        events.append((row.choice_time + .05, f"pump{side_pump} reward_amount: 3"))
    pd.DataFrame(sorted(events), columns=["Time", "Event"]).assign(Note="").to_csv(
        path.parent / "events.csv", index=False,
    )
    for probe in ("A", "B"):
        alignment_path = path.parent / probe / "aligned.npz"
        cluster_path = path.parent / probe / "spike_clusters.npy"
        with np.load(alignment_path, allow_pickle=False) as alignment:
            existing_times = alignment["spike_utc_unix"]
            bounds = alignment["irig_utc_unix"]
        left_times = trials.loc[trials.action.eq(1), "choice_time"].to_numpy() + .125
        reward_times = trials.loc[trials.reward.eq(1), "choice_time"].to_numpy() + .625
        all_times = np.r_[existing_times, left_times, reward_times]
        all_clusters = np.r_[np.load(cluster_path, allow_pickle=False),
                             np.full(len(left_times), 0), np.full(len(reward_times), 1)]
        order = np.argsort(all_times, kind="stable")
        np.savez(alignment_path, spike_utc_unix=all_times[order], irig_utc_unix=bounds)
        np.save(cluster_path, all_clusters[order])
    result = compute_replays(prepare_choices(path))
    episode = result["episodes"].query("motif == 'four_outcome'").iloc[0]
    assert episode.status == "available"
    markers = result["events"].query("motif == 'four_outcome' and event == 'delivery'")
    assert len(markers)
    track = result["tracks"].query("motif == 'four_outcome' and region == 'PFC' and width_ms == 500")
    assert len(track)
    probabilities = track[["p_0", "p_1", "p_2", "p_3"]].to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert np.isfinite(probabilities).all()
