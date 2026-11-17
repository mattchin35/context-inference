"""Cross-mouse curves consume synthetic saved results only, with strict coverage."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import choice_window_analysis as windows
from src.behavior_analysis import choice_window_results as results
from src.behavior_analysis import performance_plots as plots
from src.tests.behavior_analysis.test_choice_window_panel_plotting import closed_figures, _figures
from src.tests.test_main_multisession import load_main_module


@pytest.fixture
def cohort(tmp_path):
    """Create two mice, three sessions each, with trusted synthetic NPZ/CSV inputs.

    Actions use 0=right; observer values are signed left-positive, unitless.
    A day 2/B day 3 have only three trials, hence no complete 10/20 windows.
    A day 3 has ten-choice agreements [0,1,1,1], median 1 rather than pooled .75.
    """
    runs = {}
    for mouse in ("A", "B"):
        owner = tmp_path / mouse / "cross_session_analysis"
        run = owner / "choice_window_agreement/explicit_run"
        run.mkdir(parents=True)
        sessions, sources, tables, roster = [], [], [], []
        for day in (3, 1, 2):
            session = f"{mouse}_2026-01-{day:02d}_120000"
            date = f"2026-01-{day:02d}"
            short = (mouse, day) in (("A", 2), ("B", 3))
            n_trials = 3 if short else 41
            values = np.full(n_trials, 1. if (mouse, day) in (("A", 1), ("B", 2)) else -1.)
            if (mouse, day) == ("A", 3):
                values[:10] = 1.
            trial = pd.DataFrame({"action": np.zeros(n_trials), "observer_value": values,
                                  "Qlearning_rel_value": -values})
            table, source = results.save_choice_window_results(
                trial, tmp_path / mouse / session / "processed", session,
                feature_parameters={"session_specific": mouse})
            tables.append(table.assign(date=date))
            sessions.append(session)
            sources.append(str(source))
            roster.append({"mouse": mouse, "session_id": session, "date": date})
        pd.DataFrame(roster).to_csv(owner / f"{mouse}_overall_performance.csv", index=False)
        summary = windows.summarize_choice_window_sessions(pd.concat(tables, ignore_index=True))
        summary.to_csv(run / f"{mouse}_choice_window_summary.csv", index=False)
        meta = {"generator": "main.save_and_plot_multisession_choice_window_agreement",
                "mouse": mouse, "analysis_version": windows.ANALYSIS_VERSION,
                "window_sizes": [10, 20], "session_ids": sessions, "sources": sources, "errors": []}
        (run / "metadata.json").write_text(json.dumps(meta))
        runs[mouse] = run
    return tmp_path, runs


def loader():
    """Return the new saved-only loader callable; absence is a RED-phase failure."""
    from src.behavior_analysis import cross_mouse_choice_windows
    return cross_mouse_choice_windows.load_cross_mouse_choice_window_data


def test_loader_selects_ideal_medians_and_preserves_session_gaps(cohort):
    """Two source collections become (12,C) rows; missing short-session medians retain their ordinal."""
    root, runs = cohort
    table, provenance = loader()(root, runs)
    assert len(table) == 12
    a = table.loc[table.mouse.eq("A") & table.window_size.eq(10)].sort_values("training_day")
    assert a.training_day.tolist() == [1, 2, 3]
    np.testing.assert_allclose(a.metric_value, [0., np.nan, 1.], equal_nan=True)
    assert a.status.tolist() == ["ok", "no_complete_windows", "ok"]
    assert table.loc[table.mouse.eq("A") & table.training_day.eq(3) & table.window_size.eq(20), "metric_value"].item() == .75
    assert set(provenance["collection_runs"]) == {"A", "B"}
    assert provenance["source_sha256"]
    assert provenance["feature_parameters"]
    assert provenance["analysis_version"] == windows.ANALYSIS_VERSION


@pytest.mark.parametrize("sizes", [(10,), (20,), (10, 20)])
def test_loader_supports_requested_sizes_only(cohort, sizes):
    """Size tuple[K] selects existing summaries without deriving any new windows."""
    table, _ = loader()(*cohort, window_sizes=sizes)
    assert set(table.window_size) == set(sizes)


@pytest.mark.parametrize("case,match", [
    ("missing_source", "A.*120000"), ("missing_session", "A.*120000"),
    ("missing_size", "20"), ("missing_ideal", "observer"), ("duplicate", "duplicate"),
    ("bad_count", "count"), ("bad_fraction", "agreement"),
    ("version", "version"), ("rules", "exclude"), ("legacy_roster", "session_id"),
])
def test_invalid_saved_inputs_fail_before_any_export(cohort, case, match):
    """Malformed synthetic input files cause actionable errors, never incomplete cohort plots."""
    root, runs = cohort
    run = runs["A"]
    meta = json.loads((run / "metadata.json").read_text())
    summary_path = run / "A_choice_window_summary.csv"
    summary = pd.read_csv(summary_path)
    if case == "missing_source":
        Path(meta["sources"][0]).unlink()
    elif case == "missing_session":
        meta["session_ids"].pop()
        meta["sources"].pop()
    elif case == "missing_size":
        meta["window_sizes"] = [10]
    elif case == "missing_ideal":
        summary = summary.loc[~summary.agent.eq("observer_mouse_agreement")]
    elif case == "duplicate":
        summary = pd.concat([summary, summary.iloc[:1]], ignore_index=True)
    elif case == "bad_count":
        summary.loc[summary.agent.eq("observer_mouse_agreement"), "n_scored_windows"] = 999
    elif case == "bad_fraction":
        summary.loc[summary.agent.eq("observer_mouse_agreement"), "agreement_median"] = 1.5
    elif case in ("version", "rules"):
        path = Path(meta["sources"][0])
        with np.load(path, allow_pickle=True) as saved:
            arrays = {key: saved[key] for key in saved.files}
        source_meta = arrays["meta"].item()
        if case == "version":
            source_meta["analysis_version"] = "different"
        else:
            source_meta["parameters"]["exclude_no_choice"] = False
        arrays["meta"] = source_meta
        np.savez_compressed(path, **arrays)
    elif case == "legacy_roster":
        roster = root / "A/cross_session_analysis/A_overall_performance.csv"
        pd.read_csv(roster).drop(columns="session_id").to_csv(roster, index=False)
    summary.to_csv(summary_path, index=False)
    (run / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=match):
        loader()(root, runs)


def test_missing_files_for_both_mice_are_reported_together(cohort):
    """Strict cohort validation aggregates missing-owner errors rather than silently skipping mice."""
    root, runs = cohort
    for mouse, run in runs.items():
        (run / f"{mouse}_choice_window_summary.csv").unlink()
    with pytest.raises(ValueError) as caught:
        loader()(root, runs)
    assert "A_choice_window_summary.csv" in str(caught.value)
    assert "B_choice_window_summary.csv" in str(caught.value)


def test_no_prediction_rows_are_distinct_from_zero_windows(cohort):
    """A saved zero-denominator median stays NaN with a different coverage status."""
    root, runs = cohort
    path = runs["A"] / "A_choice_window_summary.csv"
    summary = pd.read_csv(path)
    selected = summary.agent.eq("observer_mouse_agreement") & summary.session_id.str.contains("01-01")
    summary.loc[selected, ["n_scored_windows", "n_comparisons", "n_agreements"]] = 0
    summary.loc[selected, "agreement_median"] = np.nan
    summary.to_csv(path, index=False)
    table, _ = loader()(root, runs)
    assert set(table.loc[table.mouse.eq("A") & table.training_day.eq(1), "status"]) == {"no_usable_predictions"}


def test_all_unusable_cohort_is_rejected(cohort):
    """A requested size without a single usable session median produces an error, not an empty curve."""
    root, runs = cohort
    for mouse, run in runs.items():
        path = run / f"{mouse}_choice_window_summary.csv"
        summary = pd.read_csv(path)
        selected = summary.agent.eq("observer_mouse_agreement")
        summary.loc[selected, ["n_scored_windows", "n_comparisons", "n_agreements"]] = 0
        summary.loc[selected, "agreement_median"] = np.nan
        summary.to_csv(path, index=False)
    with pytest.raises(ValueError, match="No usable"):
        loader()(root, runs)


def test_renderer_preserves_gaps_equal_mouse_weight_and_optional_caption(cohort, closed_figures, tmp_path):
    """Prepared (mouse,session,size) rows give distinct mouse lines and a mean of mouse medians."""
    table, _ = loader()(*cohort)
    plots.plot_cross_mouse_session_metric_curve(
        table.loc[table.window_size.eq(10)], tmp_path / "plots", "cross_mouse", "ideal_10_choice", "Ideal agreement",
        ylim=(0, 1), title="Ideal agreement: 10-choice windows",
        caption="Session medians; equal weight per mouse. Training Day is chronological session number.")
    figure = _figures(closed_figures)[0]
    axis = figure.axes[0]
    lines = {line.get_label(): line for line in axis.lines}
    np.testing.assert_allclose(lines["A"].get_ydata(), [0., np.nan, 1.], equal_nan=True)
    np.testing.assert_allclose(lines["group mean"].get_ydata(), [.5, 0., 1.])
    assert lines["group mean"].get_color() == "black"
    assert axis.get_title() == "Ideal agreement: 10-choice windows"
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert figure.texts[-1].get_window_extent(renderer).y1 < axis.get_tightbbox(renderer).y0


def test_synthetic_end_to_end_outputs_and_source_preservation(cohort, closed_figures, tmp_path, monkeypatch):
    """The separate command writes two plots plus data/provenance without invoking session analysis."""
    from src.behavior_analysis import choice_window_results
    root, runs = cohort
    main = load_main_module()
    monkeypatch.setattr(choice_window_results, "save_choice_window_results", reject_analysis)
    before = {path: path.read_bytes() for path in root.rglob("*.npz")}
    table, directory = main.run_cross_mouse_choice_window_agreement_curves(root, tmp_path / "exports", runs)
    assert len(table) == 12 and directory.is_dir()
    assert len(list(directory.glob("*.png"))) == 2
    assert (directory / "session_values.csv").is_file()
    group = pd.read_csv(directory / "group_summary.csv")
    ten = group.loc[group.window_size.eq(10)].sort_values("training_day")
    np.testing.assert_allclose(ten.agreement_mean, [.5, 0., 1.])
    assert ten.n_mice.tolist() == [2, 1, 1]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["random_seed"] is None and metadata["status"] == "ok"
    assert metadata["analysis_version"] == "cross_mouse_choice_windows_v1"
    for filename in ("main.py", "cross_mouse_choice_windows.py", "performance_plots.py", "summary.md", "run.log"):
        assert (directory / filename).is_file()
    assert all(path.read_bytes() == contents for path, contents in before.items())


def reject_analysis(*args, **kwargs):
    """Arbitrary session-analysis arguments must never be used by this saved-only workflow."""
    pytest.fail("Cross-mouse plotting must not run session analysis")


def test_missing_input_writes_no_partial_exports(cohort, tmp_path):
    """Validation finishes before creating the output directory or any cohort figure."""
    root, runs = cohort
    (runs["B"] / "metadata.json").unlink()
    output = tmp_path / "exports"
    with pytest.raises(ValueError):
        load_main_module().run_cross_mouse_choice_window_agreement_curves(root, output, runs)
    assert not output.exists()


@pytest.mark.parametrize("sizes", [(), (0,), (True,), (10, 10)])
def test_window_size_validation(cohort, sizes):
    """Empty, nonpositive, boolean and repeated size requests are invalid before file loading."""
    with pytest.raises(ValueError):
        loader()(*cohort, window_sizes=sizes)


def test_duplicate_roster_identity_is_rejected(cohort):
    """Repeated full session IDs cannot silently give one mouse extra group weight."""
    root, runs = cohort
    path = root / "A/cross_session_analysis/A_overall_performance.csv"
    roster = pd.read_csv(path)
    pd.concat([roster, roster.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        loader()(root, runs)


def test_same_date_sessions_remain_distinct_and_ordered(cohort):
    """Full IDs break same-date ties; a short session still occupies its own training position."""
    root, runs = cohort
    path = root / "A/cross_session_analysis/A_overall_performance.csv"
    roster = pd.read_csv(path)
    roster.loc[roster.session_id.str.contains("01-02"), "date"] = "2026-01-01"
    roster.to_csv(path, index=False)
    table, _ = loader()(root, runs)
    rows = table.loc[table.mouse.eq("A") & table.window_size.eq(10)].sort_values("training_day")
    assert rows.training_day.tolist() == [1, 2, 3]
    assert rows.date.tolist() == ["2026-01-01", "2026-01-01", "2026-01-03"]


def test_cohort_cannot_mix_consistent_but_different_mouse_versions(cohort):
    """Internally matching source/collection versions must also match across mice."""
    root, runs = cohort
    path = runs["B"] / "metadata.json"
    meta = json.loads(path.read_text())
    meta["analysis_version"] = "different"
    for source in meta["sources"]:
        with np.load(source, allow_pickle=True) as saved:
            arrays = {key: saved[key] for key in saved.files}
        source_meta = arrays["meta"].item()
        source_meta["analysis_version"] = "different"
        arrays["meta"] = source_meta
        np.savez_compressed(source, **arrays)
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="version"):
        loader()(root, runs)
