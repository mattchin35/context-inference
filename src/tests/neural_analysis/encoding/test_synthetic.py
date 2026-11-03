"""Seeded source-to-saved-UI validation with an independent context-mean oracle."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.encoding import modeling, preparation, results, run_session
from src.neural_analysis.population import session_timeline_io


SYNTHETIC_SEED = 20261103


def create_synthetic_session(root: Path) -> dict:
    """Write one new seeded source session and return source paths/native-count truth.

    root is a new directory Path. Return dict with config_path, metadata_path,
    trials(32,C), states(32,), bin_counts per region(32,8,17), and seed. Times
    are UTC seconds, counts are integer spikes per500-ms bin. Sixteen units
    vary independently; the seventeenth is constant and must be omitted by PCA.
    Context effects/noise differ between regions. All randomness uses the saved
    seed20261103. No analysis is fitted here; files use normal project interfaces.
    """
    root.mkdir(parents=True, exist_ok=False)
    (root / "processed").mkdir()
    origin = 1_700_000_000.0
    trial_count, bins, variable_units = 32, 8, 16
    positions = np.arange(trial_count)
    states = (positions // 3) % 2
    states[14] = 2
    choices = origin + positions * 8.0 + 4.0
    manual = np.isin(positions, [0, 5]).astype(int)
    actions = (positions % 2).astype(object)
    actions[9] = "no_choice"
    choice_column = choices.copy()
    choice_column[9] = np.nan
    trials = pd.DataFrame({"cur_trial": 100 + positions * 3 + positions % 2,
        "cur_block": positions // 3, "start_time": origin + positions * 8.0,
        "choice_time": choice_column, "state_int": states, "action": actions,
        "experimenter_reward_given": manual, "reward": positions % 2,
        "synthetic_seed": np.full(trial_count, SYNTHETIC_SEED)})
    trials.to_csv(root / "trials.csv", index=False)
    trials.to_csv(root / "processed/augmented.csv", index=False)
    parameters = {"generator": __name__ + ".create_synthetic_session", "random_seed": SYNTHETIC_SEED,
        "time_units": "UTC Unix seconds", "count_units": "spikes per500-ms bin",
        "count_axes": ["unfiltered_trial", "choice_relative_bin", "input_unit"]}
    (root / "processed/trial_feature_params.json").write_text(json.dumps(parameters), encoding="ascii")
    context_codes = {0: "enter_right_patch", 1: "enter_left_patch", 2: "enter_dark_period"}
    events = [{"Time": origin, "Event": context_codes[int(states[0])]}]
    for index in range(trial_count):
        events.append({"Time": origin + index * 8.0, "Event": "trial_start"})
        if index and states[index] != states[index - 1]:
            events.append({"Time": choices[index - 1] + 0.2, "Event": context_codes[int(states[index])]})
        if index != 9:
            events.extend([{"Time": choices[index] - 0.5, "Event": "LED_on"},
                           {"Time": choices[index] + 0.5, "Event": "LED_off"}])
    pd.DataFrame(events).sort_values("Time", kind="stable").to_csv(root / "events.csv", index=False)
    entering = np.where(states == 2, 0, 2 * states - 1)
    following = np.r_[entering[1:], entering[-1]]
    active = np.repeat(entering[:, None], bins, axis=1)
    active[:, 4:] = following[:, None]
    effects = {"PFC": np.array([3, -3, 2, -2, 1, -1, 1, 0, 1, -1, 0, 1, 0, -1, 0, 1]),
               "HPC": np.array([-1, 2, -2, 3, -3, 1, 0, 1, -1, 2, 1, 0, 1, 0, -1, 0])}
    rng = np.random.default_rng(SYNTHETIC_SEED)
    counts_by_region, probes = {}, {}
    for role, probe, coverage_start in (("PFC", "A", 0.0), ("HPC", "B", 20.0)):
        folder = root / probe
        folder.mkdir()
        baseline = 6 + 2 * (np.arange(variable_units) % 3)
        variable = baseline + active[:, :, None] * effects[role][None, None, :]
        variable += rng.integers(-2, 3, size=(trial_count, bins, variable_units))
        counts = np.concatenate((variable, np.full((trial_count, bins, 1), 3)), axis=2)
        counts_by_region[role] = counts
        spike_times, cluster_ids = [], []
        for trial in range(trial_count):
            for bin_index in range(bins):
                edge = choices[trial] - 2 + bin_index * 0.5
                for unit in range(variable_units + 1):
                    count = int(counts[trial, bin_index, unit])
                    spike_times.extend(edge + (np.arange(count) + 1) * (0.5 / (count + 1)))
                    cluster_ids.extend([unit + 7] * count)
        order = np.argsort(spike_times, kind="stable")
        np.save(folder / "spike_clusters.npy", np.asarray(cluster_ids)[order])
        np.savez(folder / "aligned.npz", spike_utc_unix=np.asarray(spike_times)[order],
                 irig_utc_unix=origin + np.arange(coverage_start, 261.0))
        unit_indices = np.arange(variable_units + 1)
        pd.DataFrame({"cluster_id": unit_indices + 7, "ch": unit_indices,
            "group": np.where(unit_indices % 2, "mua", "good"),
            "KSLabel": np.where(unit_indices % 2, "good", "mua"),
        }).to_csv(folder / "cluster_info.tsv", sep="\t", index=False)
        pd.DataFrame({"channel_id": [f"CH{index}" for index in unit_indices],
            "label": "good", "inside_brain": True, "x_um": 0.0, "y_um": unit_indices * 20.0,
        }).to_csv(folder / "channel_quality.csv", index=False)
        probes[probe] = {"lfp": "", "alignment": f"{probe}/aligned.npz", "sorter": probe,
            "quality": f"{probe}/channel_quality.csv", "sites": {role: 0}}
    metadata = {"schema_version": "2", "session": "seeded-context-validation",
        "acquisition": "open_ephys", "behavior": {"trials": "trials.csv", "events": "events.csv"},
        "probes": probes, "site_pairs": [["PFC", "HPC"]]}
    metadata_path = root / "neural_session.json"
    metadata_path.write_text(json.dumps(metadata), encoding="ascii")
    config_path = root / "encoding.json"
    config_path.write_text(json.dumps({"session_metadata": "neural_session.json",
        "augmented_trials": "processed/augmented.csv", "feature_parameters": "processed/trial_feature_params.json",
        "probe_roles": {"PFC": "A", "HPC": "B"}, "penalty_modes": ["same_numeric", "matched_context"]}), encoding="ascii")
    return {"config_path": config_path, "metadata_path": metadata_path, "trials": trials,
            "states": states, "counts": counts_by_region, "seed": SYNTHETIC_SEED}


@pytest.fixture(scope="module")
def synthetic_encoding_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Return a persistent pytest synthetic source/run/loaded snapshot, in native count/PC/UTC units.

    Input is pytest's temporary-directory factory. The resulting path remains
    available after tests for browser inspection. Normal single-session APIs
    prepare, fit and save all60 unique requested cells plus regional PNGs.
    """
    generated = create_synthetic_session(tmp_path_factory.mktemp("context_encoding_scientific") / "session")
    report = run_session.run_encoding_session(generated["config_path"])
    directory = Path(report["run_directory"])
    print(f"Synthetic context encoding browser run: {directory}")
    return {**generated, "report": report, "run_directory": directory,
            "loaded": results.load_encoding_run(directory)}


def oracle_cell_arrays(generated: dict, cell: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive native X(N,P), projected PC Y(N,K), and selected rows(N,) independently of regression.

    Inputs include generated counts(T,8,17), full source states(T,), saved fixed
    unit scales/components and cell settings. Times are seconds; returned X is
    dimensionless, Y has saved PC-score units. This oracle does not call model
    fitting, encoding, or predictor timing code from the analysis package.
    """
    loaded, settings = generated["loaded"], cell["meta"]["settings"]
    region = loaded["preparation"]["regions"][settings["region"]]
    behavior = loaded["preparation"]["behavior"]
    positions = behavior.choices.iloc[region.timeline.choice_positions].row_position.to_numpy(dtype=int)
    counts = generated["counts"][settings["region"]][positions, :, :16].astype(float)
    transform = region.timeline.transform
    projected = ((counts - transform.training_mean) / transform.training_scale) @ transform.components.T
    np.testing.assert_allclose(projected, region.timeline.scores, atol=1e-12)
    np.testing.assert_allclose(transform.training_mean, counts.reshape(-1, 16).mean(axis=0))
    np.testing.assert_allclose(transform.training_scale, counts.reshape(-1, 16).std(axis=0))
    states = generated["states"].astype(float)
    context = np.repeat(states[positions, None], 8, axis=1)
    if settings["timing_mode"] == "choice_updated":
        following = np.r_[states[1:], np.nan]
        context[:, 4:] = following[positions, None]
    valid = np.isin(context, [0, 1])
    if settings["window"] == "before":
        valid[:, 4:] = False
    elif settings["window"] == "after":
        valid[:, :4] = False
    indices = np.flatnonzero(valid.reshape(-1))
    values = context.reshape(-1)[indices]
    x = ((2 * values - 1)[:, None] if settings["encoding"] == "signed_context"
         else np.column_stack((values == 1, values == 0)).astype(float))
    return x, projected.reshape(-1, 16)[indices], indices


def context_mean_solution(x: np.ndarray, y: np.ndarray, effective_alpha: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic slopes(P,K)/intercept(K,) for native context X(N,P), fixed PC Y(N,K).

    This independent oracle uses context-group means and the known centered
    contrast sum-of-squares, without least-squares or regression libraries.
    OLS alpha=None gives centered minimum-norm indicator slopes. Ridge uses
    summed SSE with an unpenalized intercept, in original PC/native units.
    """
    left = x[:, 0] == 1
    n_left, n_right = int(left.sum()), int((~left).sum())
    assert n_left and n_right
    contrast = (y[left].mean(axis=0) - y[~left].mean(axis=0)) / 2
    sxx = 4 * n_left * n_right / len(x)
    penalty = 0 if effective_alpha is None else effective_alpha * (1 if x.shape[1] == 1 else 2)
    contrast = contrast * sxx / (sxx + penalty)
    slopes = contrast[None, :] if x.shape[1] == 1 else np.stack((contrast, -contrast))
    intercept = y.mean(axis=0) - x.mean(axis=0) @ slopes
    return slopes, intercept


def forbid_live_analysis(*args: object, **kwargs: object) -> None:
    """Reject arbitrary source/fitting inputs during saved benchmarking; return no data or units."""
    pytest.fail("Benchmark must consume saved products without live sources or fitting")


def render_synthetic_view(metadata_path: str) -> None:
    """Input scalar metadata filename; render saved-only Streamlit UI, return None, elapsed-second times."""
    import streamlit as st
    from src.neural_analysis.webapp.encoding_views import render_encoding_view
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    render_encoding_view(st, load_webapp_session(metadata_path))


def test_all_synthetic_cells_match_independent_generated_count_oracle(synthetic_encoding_run: dict) -> None:
    """Input seeded saved run; all60 cells and five training folds match count/group-mean truth, return None."""
    generated = synthetic_encoding_run
    loaded = generated["loaded"]
    assert generated["report"]["status"] == "complete" and len(loaded["cells"]) == 60
    prep = loaded["preparation"]
    np.testing.assert_array_equal(prep["trials"].synthetic_seed, np.full(32, SYNTHETIC_SEED))
    assert prep["behavior"].time_origin_s == generated["trials"].start_time.iloc[0]
    expected = np.flatnonzero(~np.isin(np.arange(32), [0, 5, 9, 14]))
    np.testing.assert_array_equal(prep["behavior"].choices.row_position, expected)
    assert len(prep["regions"]["PFC"].window_fold_ids) > len(prep["regions"]["HPC"].window_fold_ids)
    for role, region in prep["regions"].items():
        assert region.timeline.scores.shape[2] == 16
        assert len(region.timeline.transform.input_unit_ids) == 17
        assert len(region.timeline.transform.retained_unit_ids) == 16
        assert region.timeline.transform.omitted_unit_ids == (f"{region.timeline.probe_id}:23",)
        np.testing.assert_array_equal(region.pc_ranks, np.arange(1, 17))
    for cell in loaded["cells"].values():
        meta = cell["meta"]
        assert meta["available"] and meta["cv_available"]
        x, y, indices = oracle_cell_arrays(generated, cell)
        np.testing.assert_array_equal(cell["selected_row_indices"], indices)
        np.testing.assert_array_equal(cell["predictors"], x)
        np.testing.assert_allclose(cell["observed"], y, atol=1e-12)
        slopes, intercept = context_mean_solution(x, y, meta["effective_alpha"])
        np.testing.assert_allclose(cell["reference_slopes"], slopes, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(cell["reference_intercept"], intercept, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(cell["reference_predictions"], x @ slopes + intercept, atol=1e-10)
        for fold in range(5):
            train, test = cell["fold_ids"] != fold, cell["fold_ids"] == fold
            fold_slopes, fold_intercept = context_mean_solution(x[train], y[train], meta["effective_alpha"])
            np.testing.assert_allclose(cell["fold_slopes"][fold], fold_slopes, rtol=1e-10, atol=1e-11)
            np.testing.assert_allclose(cell["held_out_predictions"][test], x[test] @ fold_slopes + fold_intercept, atol=1e-10)
    assert len(list((generated["run_directory"] / "figures").rglob("*.png"))) >= 4


def test_synthetic_encodings_match_where_penalties_and_rows_match(synthetic_encoding_run: dict) -> None:
    """Input generated saved run; OLS/matched ridge equality holds across all timing/window/region cells, return None."""
    cells = list(synthetic_encoding_run["loaded"]["cells"].values())
    for role in ("PFC", "HPC"):
        for timing in ("choice_updated", "trial_window"):
            for window in ("before", "after", "whole"):
                chosen = [cell for cell in cells if all(cell["meta"]["settings"][key] == value
                    for key, value in (("region", role), ("timing_mode", timing), ("window", window)))]
                ols = [cell for cell in chosen if cell["meta"]["settings"]["estimator"] == "ols"]
                signed = next(cell for cell in chosen if cell["meta"]["settings"]["estimator"] == "ridge"
                              and cell["meta"]["settings"]["encoding"] == "signed_context")
                matched = next(cell for cell in chosen if cell["meta"]["settings"]["encoding"] == "separate_context"
                               and cell["meta"]["settings"]["penalty_mode"] == "matched_context")
                same = next(cell for cell in chosen if cell["meta"]["settings"]["encoding"] == "separate_context"
                            and cell["meta"]["settings"]["penalty_mode"] == "same_numeric")
                for first, second in ((ols[0], ols[1]), (signed, matched)):
                    np.testing.assert_array_equal(first["selected_row_indices"], second["selected_row_indices"])
                    for product in ("reference_predictions", "held_out_predictions", "mse", "r2"):
                        np.testing.assert_allclose(first[product], second[product], atol=1e-10)
                assert not np.allclose(signed["reference_predictions"], same["reference_predictions"])
    pfc = synthetic_encoding_run["loaded"]["preparation"]["regions"]["PFC"]
    assert pfc.context_valid["trial_window"][-1].all()
    assert not pfc.context_valid["choice_updated"][-1, 4:].any()
    positions = synthetic_encoding_run["loaded"]["preparation"]["behavior"].choices.row_position.to_numpy()
    next_nochoice = int(np.flatnonzero(positions == 8)[0])
    assert pfc.context_valid["choice_updated"][next_nochoice].all()
    np.testing.assert_array_equal(pfc.context_values["choice_updated"][next_nochoice, 4:], [1, 1, 1, 1])


def test_synthetic_saved_ui_can_inspect_pc12_without_refitting(synthetic_encoding_run: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input saved run/patch; full saved UI reaches original PC12 with source/fitting forbidden, return None."""
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_live_analysis)
    monkeypatch.setattr(preparation, "load_behavior_inputs", forbid_live_analysis)
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbid_live_analysis)
    view = AppTest.from_function(render_synthetic_view, args=(str(synthetic_encoding_run["metadata_path"]),)).run()
    assert not view.exception
    next(widget for widget in view.selectbox if widget.label == "Selected PC").select(12).run()
    next(widget for widget in view.selectbox if widget.label == "Coefficient scale").select("standardized").run()
    assert not view.exception
    component = view.get("bidi_component")[0]
    figure = json.loads(json.loads(component.proto.json)["figure_json"])
    assert figure["layout"]["meta"]["selected_pc_rank"] == 12
    assert figure["layout"]["meta"]["coefficient_scale"] == "standardized"


def test_synthetic_run_writes_measured_performance_and_benchmark_is_saved_only(
    synthetic_encoding_run: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input saved synthetic run/patch; timings/bytes are measured without reloading raw activity or fitting, return None."""
    directory = synthetic_encoding_run["run_directory"]
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_live_analysis)
    monkeypatch.setattr(preparation, "load_behavior_inputs", forbid_live_analysis)
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbid_live_analysis)
    preparation_load = Mock(wraps=results.load_preparation)
    cell_load = Mock(wraps=results.load_cell)
    monkeypatch.setattr(results, "load_preparation", preparation_load)
    monkeypatch.setattr(results, "load_cell", cell_load)
    report = run_session.benchmark_encoding_run(directory)
    assert (directory / "performance.json").is_file()
    assert preparation_load.call_count == 1 and cell_load.call_count == 2
    assert report["session_id"] == "seeded-context-validation" and report["bin_width_ms"] == 500
    for name in ("recorded_preparation_s", "preparation_save_s", "aggregate_fit_s", "aggregate_cell_save_s",
                 "load_s", "figure_s", "serialization_s"):
        assert np.isfinite(report[name]) and report[name] >= 0
    assert report["serialization_bytes"] > 0 and report["compressed_npz_bytes"] > 0
    assert report["peak_process_rss_bytes"] > 0 and report["memory_note"]
    manifest = synthetic_encoding_run["loaded"]["manifest"]
    prep_path = Path(manifest["session_root"]) / manifest["preparation_path"]
    expected_bytes = prep_path.stat().st_size + sum((directory / path).stat().st_size
        for path in synthetic_encoding_run["loaded"]["progress"]["completed_cells"].values())
    assert report["compressed_npz_bytes"] == expected_bytes
    for role, region in synthetic_encoding_run["loaded"]["preparation"]["regions"].items():
        actual = report["regions"][role]
        assert actual["windows"] == region.timeline.scores.shape[0]
        assert actual["bins"] == 8 and actual["input_units"] == 17
        assert actual["retained_units"] == actual["fitted_pcs"] == 16
        assert actual["pca_s"] == pytest.approx(region.timeline.timings_s["standardization_and_pca"])
    saved = json.loads((directory / "performance.json").read_text(encoding="ascii"))
    assert saved["session_id"] == report["session_id"]
    assert saved["compressed_npz_bytes"] == report["compressed_npz_bytes"]


def test_new_and_resume_automatically_save_performance_report(
    synthetic_encoding_run: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input synthetic run/patch; new and source-free completed resume both publish measured JSON, return None."""
    directory = synthetic_encoding_run["run_directory"]
    path = directory / "performance.json"
    assert path.is_file()
    path.unlink()
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_live_analysis)
    monkeypatch.setattr(preparation, "load_behavior_inputs", forbid_live_analysis)
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbid_live_analysis)
    report = run_session.run_encoding_session(command="resume", run_directory=directory)
    assert report["status"] == "complete" and path.is_file()
    assert json.loads(path.read_text(encoding="ascii"))["session_id"] == "seeded-context-validation"
