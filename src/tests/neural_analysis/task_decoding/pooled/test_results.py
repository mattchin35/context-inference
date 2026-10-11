"""Schema, checkpoint, and scientific-source contracts for pooled decoding."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.neural_analysis.task_decoding import modeling
from src.neural_analysis.task_decoding.pooled import decoding
from src.neural_analysis.task_decoding.pooled import results
from src.neural_analysis.task_decoding.pooled.config import TrainingWindow


def make_result_template() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Create one small structurally complete unavailable pooled result.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, object]]
        Exact schema-1 arrays and canonical metadata for dimensions
        ``W=1, C=2, T=2, F=3, Q=4, N=5, M=4, G=4``.
    """
    return results.initialize_pooled_result_arrays(
        training_windows=(
            TrainingWindow("pre_choice", -0.5, 0.0, np.array([0, 1], dtype=np.int64)),
        ),
        condition_names=("correct_rewarded", "incorrect_unrewarded"),
        target_names=("current_action", "relative_doubt"),
        target_family_names=("categorical", "numerical"),
        target_unit_names=("encoded class", "unitless"),
        outer_fold_count=3,
        evaluation_time_centers_s=np.array([-0.375, -0.125, 0.125, 0.375]),
        time_bin_edges_s=np.array([-0.5, -0.25, 0.0, 0.25, 0.5]),
        trial_positions=np.arange(5, dtype=np.int64),
        trial_ids=np.array([101, 102, 103, 104, 105], dtype=np.int64),
        behavioral_block_ids=np.array([1, 1, "two", "two", 3], dtype=object),
        trial_row_indices=np.array([0, 1, 3, 4], dtype=np.int64),
        pfc_unit_ids=("pfc:1", "pfc:2"),
        hpc_unit_ids=("hpc:1", "hpc:2"),
        pfc_pc_count=2,
        hpc_pc_count=1,
        unit_selection_rules=({"region": "PFC"}, {"region": "HPC"}),
        target_estimator_class_names=("LogisticRegression", "ElasticNet"),
        target_estimator_parameters=({"C": 1.0}, {"alpha": 1.0}),
        run_fingerprint="run-fingerprint",
        source_fingerprint="source-fingerprint",
        random_seed=1729,
    )


def test_schema_one_template_has_every_exact_axis_shape_dtype_and_sentinel():
    """The allocator should freeze all required arrays without prediction storage."""
    arrays, metadata = make_result_template()
    expected_shapes = {
        "target_values": (2, 5),
        "target_eligible": (2, 5),
        "master_eligible": (2, 5),
        "eligibility_reason_index": (2, 5),
        "condition_masks": (2, 5),
        "trial_row_indices": (4,),
        "outer_fold_ids": (2, 4),
        "training_window_bin_mask": (1, 4),
        "transform_status": (1, 2, 2, 3, 2),
        "transform_reason_index": (1, 2, 2, 3, 2),
        "transform_input_unit_count": (1, 2, 2, 3, 2),
        "transform_usable_unit_count": (1, 2, 2, 3, 2),
        "transform_component_limit": (1, 2, 2, 3, 2),
        "transform_requested_pc_count": (1, 2, 2, 3, 2),
        "transform_effective_pc_count": (1, 2, 2, 3, 2),
        "model_status": (1, 2, 2, 3, 2),
        "model_reason_index": (1, 2, 2, 3, 2),
        "model_train_trial_count": (1, 2, 2, 3, 2),
        "model_stacked_row_count": (1, 2, 2, 3, 2),
        "model_requested_feature_count": (1, 2, 2, 3, 2),
        "model_feature_count": (1, 2, 2, 3, 2),
        "model_convergence_status": (1, 2, 2, 3, 2),
        "model_intercept": (1, 2, 2, 3, 2),
        "model_coefficients": (1, 2, 2, 3, 2, 4),
        "model_feature_active": (1, 2, 2, 3, 2, 4),
        "evaluation_status": (1, 3, 2, 3, 2, 4, 3),
        "evaluation_reason_index": (1, 3, 2, 3, 2, 4, 3),
        "evaluation_scores": (1, 3, 2, 3, 2, 4, 3),
        "evaluation_test_trial_count": (1, 3, 2, 3),
        "gap_status": (1, 2, 2, 3, 2, 4, 3),
        "gap_reason_index": (1, 2, 2, 3, 2, 4, 3),
        "gap_scores": (1, 2, 2, 3, 2, 4, 3),
        "time_bin_edges_s": (5,),
        "training_window_start_s": (1,),
        "training_window_stop_s": (1,),
        "trial_positions": (5,),
        "trial_ids": (5,),
        "behavioral_block_offsets": (6,),
        "target_family_names": (2,),
        "target_unit_names": (2,),
        "target_positive_class": (2,),
        "coefficient_feature_ids": (3, 2, 4),
        "coefficient_feature_regions": (3, 2, 4),
        "coefficient_feature_present": (3, 2, 4),
        "unit_selection_rules_json": (2,),
        "model_train_class_counts": (1, 2, 2, 3, 2),
        "evaluation_test_class_counts": (1, 3, 2, 3, 2),
        "target_estimator_class_names": (2,),
        "target_estimator_parameters_json": (2,),
    }
    assert set(expected_shapes).issubset(arrays)
    for name, shape in expected_shapes.items():
        assert arrays[name].shape == shape
    assert arrays["behavioral_block_payload"].dtype == np.dtype(np.uint8)
    assert arrays["behavioral_block_offsets"].dtype == np.dtype(np.int64)
    assert arrays["transform_status"].dtype == np.dtype(np.uint8)
    assert arrays["transform_reason_index"].dtype == np.dtype(np.int16)
    assert arrays["model_train_trial_count"].dtype == np.dtype(np.int32)
    assert arrays["outer_fold_ids"].dtype == np.dtype(np.int16)
    assert arrays["model_coefficients"].dtype == np.dtype(np.float64)
    assert arrays["model_feature_active"].dtype == np.dtype(np.bool_)
    assert arrays["target_positive_class"].tolist() == [1, -1]
    assert np.all(arrays["outer_fold_ids"] == -1)
    assert np.isnan(arrays["evaluation_scores"]).all()
    assert metadata["schema_version"] == 1
    assert metadata["axes"]["evaluation_scores"] == [
        "window",
        "pair",
        "target",
        "fold",
        "region",
        "representation",
        "time",
        "metric",
    ]
    assert not any("prediction" in name for name in arrays)
    results.validate_pooled_result_arrays(arrays, metadata)


@pytest.mark.parametrize(
    ("member", "replacement", "message"),
    [
        ("outer_fold_ids", np.zeros((2, 5), dtype=np.int16), "shape|axis"),
        ("model_status", np.zeros((1, 2, 2, 3, 2), dtype=np.int16), "dtype"),
        ("reason_code_names", np.array(["unknown"], dtype="U7"), "reason"),
    ],
)
def test_schema_validation_rejects_shape_dtype_and_vocabulary_drift(
    member,
    replacement,
    message,
):
    """Strict validation should reject schema drift before publication."""
    arrays, metadata = make_result_template()
    arrays[member] = replacement
    with pytest.raises(ValueError, match=message):
        results.validate_pooled_result_arrays(arrays, metadata)


def test_checkpoint_round_trip_is_safe_exact_and_identity_bound(tmp_path, monkeypatch):
    """One window-target checkpoint strips estimators and gates all run identity."""
    from src.tests.neural_analysis.task_decoding.pooled.test_decoding import (
        make_pooled_fixture,
    )

    fixture = make_pooled_fixture()

    class FiniteClassifier:
        """Provide deterministic fitted artifacts for checkpoint persistence."""

        def fit(self, features, targets, *, sample_weight):
            """Expose a finite class-one-oriented linear fit."""
            self.coef_ = np.ones((1, features.shape[1]))
            self.intercept_ = np.array([0.0])
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, features):
            """Return finite two-class probabilities."""
            positive = 1.0 / (1.0 + np.exp(-features[:, 0]))
            return np.column_stack((1.0 - positive, positive))

        def get_params(self, deep=False):
            """Return a JSON-safe synthetic parameter mapping."""
            return {"synthetic": True}

    monkeypatch.setattr(
        modeling,
        "make_estimator",
        lambda *, target_family, parameters=None: FiniteClassifier(),
    )
    result = decoding.decode_pooled_window_target(**fixture)
    path = tmp_path / "pre_choice__synthetic_action.npz"
    results.save_pooled_checkpoint(
        path,
        checkpoint_key="pre_choice::synthetic_action",
        run_fingerprint="run-identity",
        result=result,
    )
    loaded = results.load_pooled_checkpoint(
        path,
        expected_checkpoint_key="pre_choice::synthetic_action",
        expected_run_fingerprint="run-identity",
    )

    assert loaded.unavailable_reason == result.unavailable_reason
    assert set(loaded.models) == set(result.models)
    assert set(loaded.evaluations) == set(result.evaluations)
    assert set(loaded.gaps) == set(result.gaps)
    assert all(model.fitted_estimator is None for model in loaded.models.values())
    assert all(model.transforms is None for model in loaded.models.values())
    with pytest.raises(ValueError, match="checkpoint|key|identity"):
        results.load_pooled_checkpoint(
            path,
            expected_checkpoint_key="full::synthetic_action",
            expected_run_fingerprint="run-identity",
        )
    with pytest.raises(ValueError, match="fingerprint|identity"):
        results.load_pooled_checkpoint(
            path,
            expected_checkpoint_key="pre_choice::synthetic_action",
            expected_run_fingerprint="different",
        )
    with pytest.raises(FileExistsError):
        results.save_pooled_checkpoint(
            path,
            checkpoint_key="pre_choice::synthetic_action",
            run_fingerprint="run-identity",
            result=result,
        )


def _write_pooled_source_tree(root: Path) -> tuple[str, ...]:
    """Create every explicitly scoped pooled source path in a fake checkout.

    Parameters
    ----------
    root : pathlib.Path
        Empty fake repository root.

    Returns
    -------
    tuple[str, ...]
        Exact sorted relative paths participating in pooled identity.
    """
    paths = tuple(sorted(results.POOLED_SCIENTIFIC_SOURCE_PATHS))
    for relative_path in paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n", encoding="ascii")
    return paths


def test_pooled_source_fingerprint_is_explicit_exhaustive_and_ignores_neighbors(tmp_path):
    """Every declared dependency changes identity while unrelated files do not."""
    repository = tmp_path / "repository"
    scoped_paths = _write_pooled_source_tree(repository)
    baseline = results.pooled_scientific_source_fingerprint(repository)
    assert tuple(item["relative_path"] for item in baseline["files"]) == scoped_paths
    assert "src/neural_analysis/task_decoding/__init__.py" in scoped_paths
    assert "src/neural_analysis/spike_behavior/trials.py" in scoped_paths

    unrelated = repository / "src/neural_analysis/task_decoding/pooled/unrelated.py"
    unrelated.write_text("# unrelated\n", encoding="ascii")
    assert results.pooled_scientific_source_fingerprint(repository)["fingerprint"] == baseline[
        "fingerprint"
    ]
    for relative_path in scoped_paths:
        path = repository / relative_path
        original = path.read_bytes()
        path.write_bytes(original + b"# changed\n")
        assert results.pooled_scientific_source_fingerprint(repository)["fingerprint"] != baseline[
            "fingerprint"
        ]
        path.write_bytes(original)


def test_pooled_cleanliness_checks_only_explicit_dependencies(tmp_path, monkeypatch):
    """Unrelated nested files are ignored while dirty pooled dependencies block."""
    repository = tmp_path / "repository"
    scoped_paths = _write_pooled_source_tree(repository)
    status_output = "?? src/neural_analysis/task_decoding/pooled/unrelated.py\n"

    def fake_run(command, *args, **kwargs):
        """Return deterministic Git status and tracking responses."""
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout=status_output, stderr="")
        if "ls-files" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="\n".join(scoped_paths) + "\n",
                stderr="",
            )
        raise AssertionError(command)

    monkeypatch.setattr(results.subprocess, "run", fake_run)
    results.validate_pooled_source_cleanliness(repository)
    status_output = " M src/neural_analysis/task_decoding/pooled/decoding.py\n"
    with pytest.raises(ValueError, match="clean|source"):
        results.validate_pooled_source_cleanliness(repository)


def test_save_and_load_pooled_run_validate_before_atomic_publication(tmp_path):
    """Final publication should be safe, immutable, and fingerprint-gated."""
    arrays, metadata = make_result_template()
    run_directory = tmp_path / "run"
    path = results.save_pooled_run(run_directory, arrays=arrays, metadata=metadata)
    loaded = results.load_pooled_run(
        run_directory,
        expected_run_fingerprint="run-fingerprint",
    )

    assert path == run_directory / "results.npz"
    assert loaded["metadata"] == metadata
    assert set(loaded["arrays"]) == set(arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        results.load_pooled_run(run_directory, expected_run_fingerprint="different")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        results.save_pooled_run(run_directory, arrays=arrays, metadata=metadata)
    assert path.read_bytes() == before
