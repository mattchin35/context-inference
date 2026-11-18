"""Generated-spike joint-model validation with independent design/solver oracles."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.encoding import modeling, preparation, results, run_session
from src.neural_analysis.population import session_timeline_io

SYNTHETIC_SEED = 20261113
# This order is deliberately declared independently of the production builder.
OUTCOMES = (
    "left_reward",
    "right_reward",
    "left_omission",
    "right_omission",
    "left_incorrect",
    "right_incorrect",
)


def create_synthetic_session(
    root: Path, *, imbalanced=False, missing_condition=False, constant=False
) -> dict:
    """Write generated source files and return counts(T,4,19), trials(T,C), and paths.

    root must be new. Times are UTC seconds; counts are integer spikes per
    500-ms bin in [-1,+1). Seed20261113. Units 0/1 encode context/choice, 2..7
    encode the six outcomes, 8 responds equally to every outcome, 9..16 contain
    seeded noise, 17 is constant, 18 is silent. Each of six blocks contains all
    conditions unless missing_condition removes left omissions. imbalanced
    repeats rewards more often; constant makes every unit constant (no PCA).
    """
    root.mkdir(parents=True, exist_ok=False)
    (root / "processed").mkdir()
    repeats = [4, 3, 1, 1, 1, 1] if imbalanced else [2] * 6
    if missing_condition:
        repeats[2] = 0
    kinds = np.tile(np.repeat(np.arange(6), repeats), 6)
    block_size = sum(repeats)
    count = len(kinds)
    positions = np.arange(count)
    actions = np.array([1, 0, 1, 0, 1, 0])[kinds]
    rewards = (kinds < 2).astype(int)
    correct = (kinds < 4).astype(int)
    states = np.where(correct, actions, 1 - actions)
    origin = 1_700_000_000.0
    choices = origin + positions * 8.0 + 4.0
    trials = pd.DataFrame(
        {
            "cur_trial": 100 + positions * 3 + positions % 2,
            "cur_block": positions // block_size,
            "start_time": origin + positions * 8.0,
            "choice_time": choices,
            "state_int": states,
            "action": actions,
            "reward": rewards,
            "correct": correct,
            "experimenter_reward_given": np.zeros(count, dtype=int),
            "synthetic_seed": SYNTHETIC_SEED,
        }
    )
    trials.to_csv(root / "trials.csv", index=False)
    trials.to_csv(root / "processed/augmented.csv", index=False)
    (root / "processed/trial_feature_params.json").write_text(
        json.dumps(
            {
                "generator": __name__ + ".create_synthetic_session",
                "random_seed": SYNTHETIC_SEED,
                "count_axes": ["unfiltered_trial", "choice_relative_bin", "unit"],
                "count_units": "spikes per 500-ms bin",
            }
        )
    )
    events = [
        {
            "Time": origin,
            "Event": "enter_left_patch" if states[0] else "enter_right_patch",
        }
    ]
    for index in positions:
        events.extend(
            [
                {"Time": origin + index * 8.0, "Event": "trial_start"},
                {"Time": choices[index] - 0.5, "Event": "LED_on"},
                {"Time": choices[index] + 0.5, "Event": "LED_off"},
            ]
        )
        if index and states[index] != states[index - 1]:
            events.append(
                {
                    "Time": choices[index - 1] + 0.2,
                    "Event": "enter_left_patch"
                    if states[index]
                    else "enter_right_patch",
                }
            )
    pd.DataFrame(events).sort_values("Time", kind="stable").to_csv(
        root / "events.csv", index=False
    )
    rng = np.random.default_rng(SYNTHETIC_SEED)
    counts_by_region, probes = {}, {}
    for role, probe, coverage_start in (("PFC", "A", 0.0), ("HPC", "B", 60.0)):
        folder = root / probe
        folder.mkdir()
        counts = np.full((count, 4, 19), 10, dtype=int)
        counts[:, :, 0] += 2 * (2 * states - 1)[:, None]
        counts[:, :, 1] += 2 * (2 * actions - 1)[:, None]
        for kind in range(6):
            counts[:, 2:, kind + 2] += 5 * (kinds == kind)[:, None]
        counts[:, 2:, 8] += 3
        counts[:, :, 9:17] += rng.integers(-2, 3, size=(count, 4, 8))
        counts[:, :, 17] = 3
        counts[:, :, 18] = 0
        if constant:
            counts[:, :, :18] = 3
        counts_by_region[role] = counts
        times, assignments = [], []
        for trial in positions:
            for bin_index in range(4):
                edge = choices[trial] - 1 + bin_index * 0.5
                for unit in range(19):
                    spikes = int(counts[trial, bin_index, unit])
                    times.extend(edge + (np.arange(spikes) + 1) * (0.5 / (spikes + 1)))
                    assignments.extend([unit + 7] * spikes)
        # The silent unit has an assignment outside every analysis window.
        times.append(origin + 0.1)
        assignments.append(25)
        order = np.argsort(times, kind="stable")
        np.save(folder / "spike_clusters.npy", np.asarray(assignments)[order])
        np.savez(
            folder / "aligned.npz",
            spike_utc_unix=np.asarray(times)[order],
            irig_utc_unix=origin + np.arange(coverage_start, count * 8.0 + 1),
        )
        units = np.arange(19)
        pd.DataFrame(
            {
                "cluster_id": units + 7,
                "ch": units,
                "group": np.where(units % 2, "mua", "good"),
                "KSLabel": np.where(units % 2, "good", "mua"),
            }
        ).to_csv(folder / "cluster_info.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "channel_id": [f"CH{i}" for i in units],
                "label": "good",
                "inside_brain": True,
                "x_um": 0.0,
                "y_um": units * 20.0,
            }
        ).to_csv(folder / "channel_quality.csv", index=False)
        probes[probe] = {
            "lfp": "",
            "alignment": f"{probe}/aligned.npz",
            "sorter": probe,
            "quality": f"{probe}/channel_quality.csv",
            "sites": {role: 0},
        }
    metadata = {
        "schema_version": "2",
        "session": "seeded-joint-validation",
        "acquisition": "open_ephys",
        "behavior": {"trials": "trials.csv", "events": "events.csv"},
        "probes": probes,
        "site_pairs": [["PFC", "HPC"]],
    }
    metadata_path = root / "neural_session.json"
    metadata_path.write_text(json.dumps(metadata))
    config_path = root / "encoding.json"
    config_path.write_text(
        json.dumps(
            {
                "analysis_version": "joint-task-encoding-v2",
                "session_metadata": "neural_session.json",
                "augmented_trials": "processed/augmented.csv",
                "feature_parameters": "processed/trial_feature_params.json",
                "probe_roles": {"PFC": "A", "HPC": "B"},
                "penalty_modes": ["same_numeric", "matched_context"],
            }
        )
    )
    return {
        "config_path": config_path,
        "metadata_path": metadata_path,
        "trials": trials,
        "kinds": kinds,
        "counts": counts_by_region,
        "seed": SYNTHETIC_SEED,
    }


@pytest.fixture(scope="module", params=[False, True], ids=["balanced", "imbalanced"])
def synthetic_encoding_run(tmp_path_factory, request):
    """Return generated sources and completed 20-cell run in pytest temp storage; native counts/Hz/PC units.

    Parametrized balanced/imbalanced populations retain files for browser review.
    No experimental session is loaded. All randomness uses seed20261113.
    """
    generated = create_synthetic_session(
        tmp_path_factory.mktemp("joint_scientific") / "session",
        imbalanced=request.param,
    )
    report = run_session.run_encoding_session(generated["config_path"])
    directory = Path(report["run_directory"])
    print(f"Synthetic joint browser run: {directory}")
    return {
        **generated,
        "report": report,
        "run_directory": directory,
        "loaded": results.load_encoding_run(directory),
    }


def oracle_cell_arrays(generated, cell):
    """Derive X(N,P), Y(N,R), and row indices(N,) from generated counts without production design/fitting.

    X uses dimensionless native signed/indicator values. Y is count/.5 Hz for
    units or the explicit saved mean/scale/component projection for PCs.
    Source positions are recovered from saved trial identity, not assumed IDs.
    """
    prep, settings = generated["loaded"]["preparation"], cell["meta"]["settings"]
    region = prep["regions"][settings["region"]]
    positions = (
        prep["behavior"]
        .choices.iloc[region.choice_positions]
        .row_position.to_numpy(dtype=int)
    )
    counts = generated["counts"][settings["region"]][positions].astype(float)
    np.testing.assert_array_equal(region.unit_counts, counts)
    trials = generated["trials"].iloc[positions]
    context = np.repeat(2 * trials.state_int.to_numpy() - 1, 4)
    choice = np.repeat(2 * trials.action.to_numpy() - 1, 4)
    c = (
        context[:, None]
        if settings["encoding"] == "signed_context"
        else np.column_stack(((context + 1) / 2, (1 - context) / 2))
    )
    outcomes = np.column_stack(
        [
            np.repeat(generated["kinds"][positions] == kind, 4)
            * np.tile([0, 0, 1, 1], len(positions))
            for kind in range(6)
        ]
    )
    x = np.column_stack((c, choice, outcomes))
    if settings["representation"] == "units":
        y = counts.reshape(-1, 19) / 0.5
    else:
        transform = region.pca.transform
        retained = [region.unit_ids.index(unit) for unit in transform.retained_unit_ids]
        flat = counts[:, :, retained].reshape(-1, len(retained))
        np.testing.assert_allclose(transform.training_mean, flat.mean(0))
        np.testing.assert_allclose(transform.training_scale, flat.std(0))
        y = (
            (flat - transform.training_mean) / transform.training_scale
        ) @ transform.components[region.pc_ranks - 1].T
    return x, y, np.arange(len(x))


def penalized_oracle(x, y, penalties):
    """Input native X(N,P), Y(N,R), penalties(P,); return augmented-system slopes(P,R), intercept(R,) in Y units."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    b = np.linalg.lstsq(
        np.vstack((xc, np.diag(np.sqrt(penalties)))),
        np.vstack((yc, np.zeros((x.shape[1], y.shape[1])))),
        rcond=None,
    )[0]
    return b, y.mean(0) - x.mean(0) @ b


def test_all_cells_and_folds_match_independent_count_projection_and_penalty_oracles(
    synthetic_encoding_run,
):
    """All 20 unit/PC fits and their held-out folds agree with independent native designs and augmented systems."""
    generated = synthetic_encoding_run
    loaded = generated["loaded"]
    assert generated["report"]["status"] == "complete" and len(loaded["cells"]) == 20
    assert len(list((generated["run_directory"] / "figures").rglob("*.png"))) == 8
    prep = loaded["preparation"]
    assert len(prep["regions"]["PFC"].choice_positions) > len(
        prep["regions"]["HPC"].choice_positions
    )
    for region in prep["regions"].values():
        assert region.unit_counts.shape[1:] == (4, 19) and len(region.pc_ranks) >= 12
        assert len(region.pca.transform.retained_unit_ids) == 17
    for cell in loaded["cells"].values():
        x, y, indices = oracle_cell_arrays(generated, cell)
        np.testing.assert_array_equal(cell["predictors"], x)
        np.testing.assert_array_equal(cell["selected_row_indices"], indices)
        np.testing.assert_allclose(cell["observed"], y, atol=1e-12)
        assert (
            cell["meta"]["available"]
            and cell["meta"]["cv_available"]
            and cell["contrast_valid"].all()
        )
        for fold in [-1, 0, 1, 2, 3, 4]:
            train = (
                np.ones(len(x), dtype=bool) if fold == -1 else cell["fold_ids"] != fold
            )
            b, a = penalized_oracle(x[train], y[train], cell["penalty_vector"])
            actual = (
                cell["reference_slopes"] if fold == -1 else cell["fold_slopes"][fold]
            )
            np.testing.assert_allclose(actual, b, atol=1e-10)
            contrasts = np.stack(
                (
                    b[list(cell["column_names"]).index("left_reward")]
                    - b[list(cell["column_names"]).index("left_omission")],
                    b[list(cell["column_names"]).index("right_reward")]
                    - b[list(cell["column_names"]).index("right_omission")],
                )
            )
            np.testing.assert_allclose(
                cell["contrasts"] if fold == -1 else cell["fold_contrasts"][fold],
                contrasts,
                atol=1e-10,
            )
            test = np.ones(len(x), dtype=bool) if fold == -1 else ~train
            predictions = (
                cell["reference_predictions"]
                if fold == -1
                else cell["held_out_predictions"][test]
            )
            np.testing.assert_allclose(predictions, x[test] @ b + a, atol=1e-10)


def test_exact_units_and_generic_post_response_recover_expected_ols_selectivity(
    synthetic_encoding_run,
):
    """Correctly specified OLS recovers Hz effects; the generic post-choice unit has zero reward contrasts in both count variants."""
    for cell in synthetic_encoding_run["loaded"]["cells"].values():
        settings = cell["meta"]["settings"]
        if settings["representation"] != "units" or settings["estimator"] != "ols":
            continue
        context = 1 if settings["encoding"] == "signed_context" else 2
        np.testing.assert_allclose(cell["reference_slopes"][0, 0], 4, atol=1e-11)
        np.testing.assert_allclose(cell["reference_slopes"][context, 1], 4, atol=1e-11)
        for kind, name in enumerate(OUTCOMES):
            np.testing.assert_allclose(
                cell["reference_slopes"][
                    list(cell["column_names"]).index(name), kind + 2
                ],
                10,
                atol=1e-11,
            )
        np.testing.assert_allclose(cell["contrasts"][:, 8], 0, atol=1e-11)
        np.testing.assert_allclose(cell["mse"][:9], 0, atol=1e-20)
        assert np.isnan(cell["r2"][-2:]).all()  # Constant/silent controls.


def forbid_live_analysis(*args, **kwargs):
    """Arbitrary source/solver inputs are forbidden in saved inspection; return no values or units."""
    pytest.fail("Saved inspection must not read live inputs or fit")


def render_synthetic_view(metadata_path):
    """Input metadata path string; render saved-only Streamlit UI in elapsed seconds, return None."""
    import streamlit as st
    from src.neural_analysis.webapp.encoding_views import render_encoding_view
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    render_encoding_view(st, load_webapp_session(metadata_path))


def test_saved_ui_reaches_pc12_and_original_unit_ids_without_refitting(
    synthetic_encoding_run, monkeypatch
):
    """Saved control changes preserve identities and fixed activity scales with all live computation forbidden."""
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_live_analysis)
    monkeypatch.setattr(preparation, "load_behavior_inputs", forbid_live_analysis)
    monkeypatch.setattr(
        session_timeline_io, "load_timeline_population", forbid_live_analysis
    )
    view = AppTest.from_function(
        render_synthetic_view, args=(str(synthetic_encoding_run["metadata_path"]),)
    ).run()
    assert not view.exception
    next(w for w in view.selectbox if w.label == "Selected response").select(
        "A:9"
    ).run()
    next(w for w in view.selectbox if w.label == "Response representation").select(
        "pcs"
    ).run()
    next(w for w in view.selectbox if w.label == "Selected response").select(
        "PC12"
    ).run()
    next(w for w in view.selectbox if w.label == "Coefficient scale").select(
        "standardized"
    ).run()
    assert not view.exception
    figure = json.loads(
        json.loads(view.get("bidi_component")[0].proto.json)["figure_json"]
    )
    assert figure["layout"]["meta"]["selected_response"] == "PC12"
    assert figure["layout"]["meta"]["coefficient_scale"] == "standardized"


def test_measured_saved_benchmark_loads_one_preparation_and_four_representatives(
    synthetic_encoding_run, monkeypatch
):
    """Report timings/bytes/RSS while loading one cell per region/representation, never raw sources or fits."""
    directory = synthetic_encoding_run["run_directory"]
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_live_analysis)
    monkeypatch.setattr(preparation, "load_behavior_inputs", forbid_live_analysis)
    preparation_load = Mock(wraps=results.load_preparation)
    cell_load = Mock(wraps=results.load_cell)
    monkeypatch.setattr(results, "load_preparation", preparation_load)
    monkeypatch.setattr(results, "load_cell", cell_load)
    report = run_session.benchmark_encoding_run(directory)
    assert preparation_load.call_count == 1 and cell_load.call_count == 4
    for key in (
        "recorded_preparation_s",
        "preparation_save_s",
        "aggregate_fit_s",
        "aggregate_cell_save_s",
        "load_s",
        "figure_s",
        "serialization_s",
    ):
        assert np.isfinite(report[key]) and report[key] >= 0
    assert (
        report["serialization_bytes"] > 0
        and report["compressed_npz_bytes"] > 0
        and report["peak_process_rss_bytes"] > 0
    )
    assert all(
        info["bins"] == 4 and info["input_units"] == 19
        for info in report["regions"].values()
    )
    assert (directory / "performance.json").is_file()


@pytest.mark.parametrize("variant", ["missing_condition", "constant"])
def test_source_to_saved_pipeline_handles_missing_conditions_and_absent_pca(
    tmp_path, variant
):
    """Missing attribution keeps predictions; absent PCA retains all units and explicit unavailable PC cells."""
    generated = create_synthetic_session(tmp_path / "session", **{variant: True})
    report = run_session.run_encoding_session(generated["config_path"])
    loaded = results.load_encoding_run(report["run_directory"])
    assert report["status"] == "complete" and len(loaded["cells"]) == 20
    for cell in loaded["cells"].values():
        if (
            variant == "constant"
            and cell["meta"]["settings"]["representation"] == "pcs"
        ):
            assert not cell["meta"]["available"] and len(cell["response_ids"]) == 0
        else:
            assert (
                cell["meta"]["available"]
                and np.isfinite(cell["reference_predictions"]).all()
            )
            if variant == "missing_condition":
                assert (
                    cell["contrast_valid"].tolist() == [False, True]
                    and cell["contrast_reasons"][0]
                )
