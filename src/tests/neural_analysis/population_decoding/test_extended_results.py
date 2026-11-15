"""Version-two publication, v1 read-only products, and comparison provenance."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.neural_analysis.population_decoding import results, pipeline
from src.neural_analysis.population_decoding.configuration import load_config
from src.neural_analysis.population_decoding.run_session import run_session, initialize_config
from src.neural_analysis.population_decoding.targets import TARGET_NAMES, MODEL_TARGET_NAMES


def test_extended_run_publishes_comparisons_with_conditional_cells(population_config_path):
    """Input synthetic v2 config; paired files contain matched scores/maps and no extra checkpoints."""
    report = run_session(population_config_path)
    directory = Path(report["run_directory"])
    manifest = results.load_manifest(directory)
    assert manifest["model_targets"] == list(MODEL_TARGET_NAMES)
    for side in ("left", "right"):
        path = directory / "cells" / f"PFC_{side}_switch_vs_stay.npz"
        cell = results.load_cell(path)
        comparison = cell["meta"]["switch_comparison"]
        assert comparison["reference_target"] == f"{side}_choice_switch"
        assert comparison["reference_model_id"] == manifest["identities"]["model_id"]
        assert len(cell["comparison_reference_window_indices"]) == len(cell["row_positions"])
    assert not (directory / "comparisons").exists()


def test_new_config_path_preserves_old_config_and_rejects_legacy_execution(population_config_path):
    """Input metadata/config; create v2 beside unchanged legacy config using explicit output."""
    payload = json.loads(population_config_path.read_text())
    payload["analysis_version"] = "population-task-decoding-v1"
    population_config_path.write_text(json.dumps(payload))
    before = population_config_path.read_bytes()
    output = population_config_path.parent / "population_decoding_switch_vs_stay.json"
    created = initialize_config(population_config_path.parent / "neural_session.json",
        probe_roles={"PFC": "A", "HPC": "B"}, output_path=output)
    assert created == output and load_config(output).analysis_version == "population-task-decoding-v2"
    assert population_config_path.read_bytes() == before
    with pytest.raises(ValueError, match="read-only"):
        run_session(population_config_path)


def test_frozen_v1_preparation_and_cell_stay_readable(synthetic_population, tmp_path):
    """Input population/temp dir; original schema-one five-column artifacts retain bytes on read."""
    prep = tmp_path / "preparation.npz"
    cell_path = tmp_path / "PFC_choice.npz"
    results.save_preparation(prep, synthetic_population, {"preparation_id": "legacy"})
    results.save_cell(cell_path, pipeline.fit_target(synthetic_population, "PFC", "choice"), {"model_id": "legacy"})
    for path in (prep, cell_path):
        with np.load(path, allow_pickle=True) as archive:
            arrays = dict(archive)
        meta = arrays["meta"].item()
        meta["analysis_version"] = "population-task-decoding-v1"
        for key in ("source_target", "conditioning_side"):
            meta.pop(key, None)
        np.savez_compressed(path, **arrays)
    before = (prep.read_bytes(), cell_path.read_bytes())
    assert results.load_preparation(prep).targets.values.shape == (240, 5)
    assert results.load_cell(cell_path)["meta"]["target"] == "choice"
    assert before == (prep.read_bytes(), cell_path.read_bytes())
    directory = tmp_path / "legacy-run"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"analysis_version": "population-task-decoding-v1",
        "schema_version": 1, "cells": [f"PFC_{name}" for name in TARGET_NAMES]}))
    assert results.model_targets_for_run(results.load_manifest(directory)) == TARGET_NAMES
    with pytest.raises(ValueError, match="read-only"):
        run_session(command="resume", run_directory=directory)


def test_corrupt_comparison_mapping_is_rejected(population_config_path):
    """Input completed v2 run; wrong required-row mapping cannot load as valid comparison."""
    report = run_session(population_config_path)
    path = Path(report["run_directory"]) / "cells/PFC_left_switch_vs_stay.npz"
    with np.load(path, allow_pickle=True) as archive:
        arrays = dict(archive)
    arrays["comparison_reference_window_indices"] = np.array([], dtype=int)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="comparison"):
        results.load_cell(path)


@pytest.mark.parametrize("damage", ["missing_representation", "wrong_required_bins", "wrong_classes"])
def test_malformed_comparison_metadata_is_rejected(population_config_path, damage):
    """Input v2 run/damage type; incomplete or misdeclared comparison summaries fail at loading."""
    report = run_session(population_config_path)
    path = Path(report["run_directory"]) / "cells/PFC_left_switch_vs_stay.npz"
    with np.load(path, allow_pickle=True) as archive:
        arrays = dict(archive)
    meta = arrays["meta"].item()
    if damage == "missing_representation":
        del meta["switch_comparison"]["representations"]["units"]
    elif damage == "wrong_required_bins":
        meta["switch_comparison"]["representations"]["units"]["within_side"]["scores"]["required_rows"] += 1
    else:
        meta["class_names"] = ["switch", "stay"]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="comparison"):
        results.load_cell(path)
