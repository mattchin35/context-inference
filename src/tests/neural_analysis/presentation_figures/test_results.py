"""Persistence checks for source-independent presentation exports."""

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.presentation_figures import results


def test_result_round_trip_preserves_axes_units_and_named_tables(tmp_path):
    """A trusted NPZ path round-trips (choice, bin, class) and one row per score."""
    path = tmp_path / "regional.npz"
    probabilities = np.array([[[0.3, 0.7], [0.1, 0.9]]])
    scores = pd.DataFrame({"region": ["HPC"], "score": [0.5], "trial_count": [1]})
    meta = {"producer": "decode_region", "axes": {"probabilities": ["choice", "bin", "class"]},
            "units": {"probabilities": "probability"}, "seed": 3, "session_id": "synthetic"}
    results.save_result(path, {"probabilities": probabilities, "scores": scores}, meta)
    with np.load(path, allow_pickle=True) as stored:
        np.testing.assert_array_equal(stored["probabilities"], probabilities)
        assert stored["meta"].item()["units"] == meta["units"]
    loaded, header = results.load_result(path)
    np.testing.assert_array_equal(loaded["probabilities"], probabilities)
    pd.testing.assert_frame_equal(loaded["scores"], scores)
    assert header["axes"] == meta["axes"]
    assert header["seed"] == 3
    with pytest.raises(FileExistsError):
        results.save_result(path, {}, meta)


def test_discovery_uses_only_complete_matching_runs(tmp_path):
    """A run root Path may contain interrupted runs and other parameter identities."""
    for name in ("presentation_figures_1", "presentation_figures_2", "presentation_figures_3"):
        (tmp_path / name).mkdir()
    for number, identity in [(1, "old"), (2, "new")]:
        directory = tmp_path / f"presentation_figures_{number}"
        results.save_result(directory / "regional.npz", {"trial_ids": np.arange(3)}, {})
        results.complete_run(directory, kind="session", identity=identity,
                             session_ids=["mouse_session"], archives={"regional": "regional.npz"})
    assert results.discover_runs(tmp_path, identity="new") == [tmp_path / "presentation_figures_2" / "manifest.json"]
    assert results.discover_runs(tmp_path, identity="changed-seed") == []
    assert len(results.discover_runs(tmp_path)) == 2
