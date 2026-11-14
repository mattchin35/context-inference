"""End-to-end signal controls and malformed/interrupted-product regression cases."""

from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population_decoding import pipeline, results
from src.neural_analysis.population_decoding.targets import TARGET_NAMES
from src.neural_analysis.population_decoding.run_session import run_session


@pytest.mark.parametrize("target", TARGET_NAMES)
def test_known_population_signals_decode_held_out_blocks(synthetic_population, target):
    """Input seeded strong independent task signals; both neural representations recover labels."""
    cell = pipeline.fit_target(synthetic_population, "PFC", target)
    for representation in ("units", "pcs"):
        assert cell["meta"]["representations"][representation]["pooled"]["balanced_accuracy"] > .95
    assert cell["meta"]["estimator"]["C"] == 1
    assert cell["meta"]["estimator"]["class_weight"] is None


def test_generic_post_choice_response_does_not_decode_outcomes(synthetic_population):
    """Input post-only shared response plus noise; timing alone gives chance four-class OOF."""
    rng = np.random.default_rng(18)
    old = synthetic_population.regions["PFC"]
    rates = 20 + rng.normal(0, 1, old.rates_hz.shape)
    rates[:, 2:] += 30
    population = replace(synthetic_population, regions={"PFC": replace(old, rates_hz=rates)})
    cell = pipeline.fit_target(population, "PFC", "outcome")
    assert cell["meta"]["representations"]["units"]["pooled"]["balanced_accuracy"] < .4


def test_side_only_population_has_no_within_side_reward_discrimination(synthetic_population):
    """Input chosen-side signal/noise; outcome scores cannot imply within-side reward coding."""
    rng = np.random.default_rng(19)
    old = synthetic_population.regions["PFC"]
    rates = 20 + rng.normal(0, 1, old.rates_hz.shape)
    rates += 30*synthetic_population.targets.values[:, 1, None, None]
    cell = pipeline.fit_target(replace(synthetic_population, regions={"PFC": replace(old, rates_hz=rates)}), "PFC", "outcome")
    summary = cell["meta"]["representations"]["units"]
    assert summary["pooled"]["accuracy"] > .35
    for side in ("left", "right"):
        assert .3 < summary["conditional_outcomes"][side]["roc_auc"][1] < .7


def test_region_target_missingness_and_constant_units_are_local(synthetic_population):
    """Input synthetic population; unknown reward and one silent unit preserve other readouts."""
    from src.neural_analysis.population_decoding.targets import build_population_targets
    table = synthetic_population.trials.copy()
    table.loc[0, "reward"] = np.nan
    targets = build_population_targets(table.state_int, table.action, table.reward, table.experimenter_reward_given)
    old = synthetic_population.regions["PFC"]
    activity = old.rates_hz.copy()
    activity[:, :, -1] = 0
    population = replace(synthetic_population, targets=targets, regions={"PFC": replace(old, rates_hz=activity)})
    choice = pipeline.fit_target(population, "PFC", "choice")
    outcome = pipeline.fit_target(population, "PFC", "outcome")
    assert len(choice["row_positions"]) == 240 and len(outcome["row_positions"]) == 239
    assert not choice["fold_0_retained_mask"][-1]
    assert choice["meta"]["fold_0_omitted_unit_reasons"][old.unit_ids[-1]]


def test_partial_resume_refits_only_missing_pair_and_refuses_changed_runtime(population_config_path, monkeypatch):
    """Input temporary run; missing one paired file redoes only it, runtime mismatch refuses resume."""
    report = run_session(population_config_path)
    directory = Path(report["run_directory"])
    removed = directory / "cells/PFC_choice.npz"
    removed.unlink()
    original = pipeline.fit_target
    calls = []
    def track_fit(prepared, region, target):
        """Record scalar target identity; return the real paired fit from native Hz inputs."""
        calls.append((region, target))
        return original(prepared, region, target)
    monkeypatch.setattr(pipeline, "fit_target", track_fit)
    assert run_session(command="resume", run_directory=directory)["status"] == "complete"
    assert calls == [("PFC", "choice")]
    def changed_runtime():
        """Return a deliberately incompatible installed-runtime identity dictionary."""
        return {"changed": "runtime"}
    monkeypatch.setattr(results, "runtime_versions", changed_runtime)
    with pytest.raises(ValueError, match="runtime"):
        run_session(command="resume", run_directory=directory)


def test_saved_fold_axis_corruption_is_rejected(synthetic_population, tmp_path):
    """Input preparation/temp path; incompatible label-mask axes cannot be loaded for fitting."""
    path = tmp_path / "preparation.npz"
    results.save_preparation(path, synthetic_population, {"preparation_id": "test"})
    with np.load(path, allow_pickle=True) as archive:
        data = dict(archive)
    data["target_valid"] = np.ones((1, 5), bool)
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="axes"):
        results.load_preparation(path)


def test_full_spike_pipeline_recovers_signals(population_config_path, synthetic_population):
    """Input seeded rate signals/source fixture; generate Poisson spikes, run/save both regions.

    Source units are spike UTC seconds and cluster IDs. Eight good/mua units
    span two good channels; bin-specific counts use seed=20261114. The entire
    raw loader/IRIG/binning/CV/publication path is exercised on synthetic data.
    """
    root = population_config_path.parent
    trials = synthetic_population.trials
    trials.to_csv(root / "trials.csv", index=False)
    trials.to_csv(root / "processed/augmented.csv", index=False)
    rng = np.random.default_rng(20261114)
    rates = synthetic_population.regions["PFC"].rates_hz
    for probe in ("A", "B"):
        pd.DataFrame({"cluster_id": np.arange(7, 15), "ch": np.arange(8) % 2,
            "group": ["good", "mua"]*4, "KSLabel": ["good"]*8}).to_csv(root / probe / "cluster_info.tsv", sep="\t", index=False)
        times, assignments = [], []
        for trial in range(len(trials)):
            choice = float(trials.choice_time.iloc[trial])
            for bin_index in range(4):
                for unit in range(8):
                    count = rng.poisson(rates[trial, bin_index, unit] * .5)
                    times.extend(choice - 1 + .5*bin_index + rng.uniform(.001, .499, count))
                    assignments.extend([unit + 7]*count)
        times = np.asarray(times)
        order = np.argsort(times)
        np.save(root / probe / "spike_clusters.npy", np.asarray(assignments)[order])
        np.savez(root / probe / "aligned.npz", spike_utc_unix=times[order],
            irig_utc_unix=np.arange(trials.start_time.iloc[0], trials.choice_time.iloc[-1] + 3))
    report = run_session(population_config_path)
    assert report["status"] == "complete"
    directory = Path(report["run_directory"])
    for region in ("PFC", "HPC"):
        for target in TARGET_NAMES:
            cell = results.load_cell(directory / "cells" / f"{region}_{target}.npz")
            for representation in ("units", "pcs"):
                assert cell["meta"]["representations"][representation]["pooled"]["balanced_accuracy"] > .8
