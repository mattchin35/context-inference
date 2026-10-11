"""Seeded end-to-end scientific gates for pooled-window decoding."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.task_decoding import modeling
from src.neural_analysis.task_decoding.pooled import config as pooled_config
from src.neural_analysis.task_decoding.pooled import pipeline, results
from src.tests.neural_analysis.task_decoding.pooled.test_config import (
    canonical_edges,
    make_base_config_payload,
    write_configuration_tree,
)


_SEED = 20261011
_REFERENCE = "correct_rewarded"
_DESTINATION = "incorrect"


def _load_config(
    root: Path,
    *,
    window_names: tuple[str, ...],
    overlapping_conditions: bool = False,
) -> tuple[pooled_config.PooledDecodingConfig, tuple[pooled_config.TrainingWindow, ...]]:
    """Create one minimal valid pooled configuration for a synthetic run.

    Parameters
    ----------
    root : pathlib.Path
        Empty pytest-owned configuration directory.
    window_names : tuple[str, ...]
        Nonempty canonical training-window selection.
    overlapping_conditions : bool, default=False
        Select ``all`` as the destination so it overlaps the reference subset.

    Returns
    -------
    tuple[PooledDecodingConfig, tuple[TrainingWindow, ...]]
        Validated one-target configuration and resolved 100 ms windows.
    """
    base_payload = make_base_config_payload()
    base_payload.update(
        {
            "alignment": "choice_time",
            "bin_width_ms": 100,
            "pfc_pc_count": 2,
            "hpc_pc_count": 2,
            "target_names": ["current_action"],
            "condition_names": ["all" if overlapping_conditions else _DESTINATION],
            "regularization_mode": "fixed",
            "outer_fold_count": 3,
            "inner_fold_count": 3,
        }
    )
    pooled_path, _base_path, _session_root = write_configuration_tree(
        root,
        base_payload=base_payload,
        pooled_payload={
            "base_config_path": "task_decoding_config.json",
            "training_window_names": list(window_names),
        },
    )
    config = pooled_config.load_pooled_decoding_config(pooled_path)
    windows = pooled_config.resolve_training_windows(config, canonical_edges())
    return config, windows


def _fixed_outer_split(block_ids: np.ndarray) -> modeling.SplitPlan:
    """Create three deterministic block-intact folds on the trial axis.

    Parameters
    ----------
    block_ids : numpy.ndarray
        Integer behavioral blocks shaped ``(trial,)`` with two trials per block.

    Returns
    -------
    modeling.SplitPlan
        Available three-fold split whose train/test indices use trial positions.
    """
    fold_ids = np.asarray(block_ids, dtype=int) % 3
    splits = tuple(
        modeling.GroupedSplit(
            fold_id,
            np.flatnonzero(fold_ids != fold_id),
            np.flatnonzero(fold_ids == fold_id),
        )
        for fold_id in range(3)
    )
    return modeling.SplitPlan(
        fold_ids=fold_ids,
        splits=splits,
        is_available=True,
        unavailable_reason=None,
    )


def _signal_factors(
    case: str,
    target_values: np.ndarray,
    reference_mask: np.ndarray,
    destination_mask: np.ndarray,
) -> np.ndarray:
    """Construct one case-specific class signal over trials and time.

    Parameters
    ----------
    case : str
        One of stable, rotating, remapped, absent, delayed,
        constant_reference, or overlap.
    target_values : numpy.ndarray
        Binary labels shaped ``(trial,)``.
    reference_mask, destination_mask : numpy.ndarray
        Boolean condition masks shaped ``(trial,)``.

    Returns
    -------
    numpy.ndarray
        Signed class signal shaped ``(trial, time=40)`` in arbitrary rate units.
    """
    signed_target = 2.0 * target_values - 1.0
    factors = np.zeros((target_values.size, 40), dtype=np.float64)
    if case in {"stable", "overlap"}:
        factors[:] = signed_target[:, None]
    elif case == "rotating":
        direction = np.concatenate((np.ones(20), -np.ones(20)))
        factors[:] = signed_target[:, None] * direction[None, :]
    elif case == "remapped":
        factors[reference_mask] = signed_target[reference_mask, None]
        factors[destination_mask] = -signed_target[destination_mask, None]
    elif case == "absent":
        factors[reference_mask] = signed_target[reference_mask, None]
    elif case == "delayed":
        factors[:, 25:30] = signed_target[:, None]
    elif case == "constant_reference":
        factors[destination_mask] = signed_target[destination_mask, None]
    else:
        raise ValueError(f"Unknown synthetic pooled case: {case}")
    return factors


def _synthetic_context(
    config: pooled_config.PooledDecodingConfig,
    windows: tuple[pooled_config.TrainingWindow, ...],
    *,
    case: str,
) -> pipeline.PooledComputationContext:
    """Build one fully prepared seeded context without spike-file I/O.

    Parameters
    ----------
    config : PooledDecodingConfig
        One-target fixed-mode configuration.
    windows : tuple[TrainingWindow, ...]
        Canonically resolved 100 ms training windows.
    case : str
        Scientific signal case passed to :func:`_signal_factors`.

    Returns
    -------
    pipeline.PooledComputationContext
        Complete schema, fold plan, condition masks, and regional rate tensors
        shaped ``(trial=24, time=40, unit=2)`` in Hz.
    """
    trial_count = 24
    block_ids = np.repeat(np.arange(12, dtype=np.int64), 2)
    target_values = np.tile(np.array([0.0, 1.0]), 12)
    reference_mask = block_ids < 6
    if case == "constant_reference":
        target_values[reference_mask] = 0.0
    if "all" in config.condition_names:
        destination_mask = np.ones(trial_count, dtype=bool)
    else:
        destination_mask = ~reference_mask
    condition_masks = {
        name: (
            reference_mask.copy()
            if name == _REFERENCE
            else destination_mask.copy()
        )
        for name in config.condition_names
    }
    signal = _signal_factors(
        case,
        target_values,
        reference_mask,
        destination_mask,
    )
    random_generator = np.random.default_rng(_SEED)
    block_nuisance = random_generator.normal(0.0, 0.12, size=(12, 40, 2))
    nuisance = block_nuisance[block_ids]
    pfc_rates_hz = (
        12.0
        + nuisance
        + 8.0 * signal[:, :, None] * np.array([1.0, -0.65])[None, None, :]
    )
    hpc_rates_hz = (
        14.0
        + 0.8 * nuisance
        + 7.0 * signal[:, :, None] * np.array([0.75, 0.45])[None, None, :]
    )
    centers = (canonical_edges()[:-1] + canonical_edges()[1:]) / 2.0
    estimator = modeling.make_estimator(target_family="categorical")
    arrays, metadata = results.initialize_pooled_result_arrays(
        training_windows=windows,
        condition_names=config.condition_names,
        target_names=("current_action",),
        target_family_names=("categorical",),
        target_unit_names=("encoded class",),
        outer_fold_count=3,
        evaluation_time_centers_s=centers,
        time_bin_edges_s=canonical_edges(),
        trial_positions=np.arange(trial_count, dtype=np.int64),
        trial_ids=np.arange(1001, 1001 + trial_count, dtype=np.int64),
        behavioral_block_ids=block_ids,
        trial_row_indices=np.arange(trial_count, dtype=np.int64),
        pfc_unit_ids=("pfc:1", "pfc:2"),
        hpc_unit_ids=("hpc:1", "hpc:2"),
        pfc_pc_count=2,
        hpc_pc_count=2,
        unit_selection_rules=({"region": "PFC"}, {"region": "HPC"}),
        target_estimator_class_names=(estimator.__class__.__name__,),
        target_estimator_parameters=(dict(estimator.get_params(deep=False)),),
        run_fingerprint=f"synthetic-{case}-{'-'.join(window.name for window in windows)}",
        source_fingerprint="synthetic-source",
        random_seed=_SEED,
    )
    arrays["condition_masks"][:] = np.stack(
        [condition_masks[name] for name in config.condition_names]
    )
    split = _fixed_outer_split(block_ids)
    pipeline.populate_target_audit_arrays(
        arrays,
        target_index=0,
        target_values=target_values,
        target_eligible=np.ones(trial_count, dtype=bool),
        baseline_eligible=np.ones(trial_count, dtype=bool),
        bilateral_full_window_coverage=np.ones(trial_count, dtype=bool),
        trial_row_indices=arrays["trial_row_indices"],
        outer_fold_ids=split.fold_ids,
    )
    return pipeline.PooledComputationContext(
        arrays=arrays,
        metadata=metadata,
        run_fingerprint=str(metadata["provenance"]["run_fingerprint"]),
        target_table=pd.DataFrame({"block_id": block_ids}),
        condition_masks=condition_masks,
        outer_splits={"current_action": split},
        pfc_rate_tensor_hz=pfc_rates_hz,
        hpc_rate_tensor_hz=hpc_rates_hz,
        pfc_unit_ids=("pfc:1", "pfc:2"),
        hpc_unit_ids=("hpc:1", "hpc:2"),
        pfc_pc_count=2,
        hpc_pc_count=2,
    )


def _fresh_context(
    context: pipeline.PooledComputationContext,
) -> pipeline.PooledComputationContext:
    """Copy mutable result state while sharing read-only synthetic inputs.

    Parameters
    ----------
    context : PooledComputationContext
        Prepared synthetic template.

    Returns
    -------
    PooledComputationContext
        Independent arrays and metadata suitable for one public pipeline run.
    """
    return replace(
        context,
        arrays={name: value.copy() for name, value in context.arrays.items()},
        metadata=deepcopy(context.metadata),
    )


def _run_case(
    monkeypatch: pytest.MonkeyPatch,
    config: pooled_config.PooledDecodingConfig,
    windows: tuple[pooled_config.TrainingWindow, ...],
    context: pipeline.PooledComputationContext,
    work_directory: Path,
) -> dict[str, object]:
    """Execute and reload one synthetic case through the public API.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Test patcher used only to inject already prepared synthetic arrays.
    config : PooledDecodingConfig
        Scientific configuration aligned to the context.
    windows : tuple[TrainingWindow, ...]
        Selected training windows.
    context : PooledComputationContext
        Synthetic full scientific input state.
    work_directory : pathlib.Path
        New checkpoint/final-result directory.

    Returns
    -------
    dict[str, object]
        Strictly validated saved pooled run.
    """
    monkeypatch.setattr(
        pipeline,
        "_prepare_pooled_computation_context",
        lambda *_: _fresh_context(context),
    )
    pipeline.compute_or_resume_pooled_results(config, windows, work_directory)
    return results.load_pooled_run(work_directory)


def _score_curve(
    saved: dict[str, object],
    *,
    window_name: str,
    train_condition: str,
    test_condition: str,
) -> np.ndarray:
    """Return mean-fold PFC direct-unit balanced accuracy through time.

    Parameters
    ----------
    saved : dict[str, object]
        Validated pooled run returned by :func:`results.load_pooled_run`.
    window_name, train_condition, test_condition : str
        Window and evaluation-pair coordinates.

    Returns
    -------
    numpy.ndarray
        Balanced-accuracy curve shaped ``(time=40,)`` in fraction units.
    """
    arrays = saved["arrays"]
    window_index = arrays["training_window_names"].tolist().index(window_name)
    pairs = list(
        zip(
            arrays["evaluation_train_condition"].tolist(),
            arrays["evaluation_test_condition"].tolist(),
            strict=True,
        )
    )
    pair_index = pairs.index((train_condition, test_condition))
    region_index = arrays["region_names"].tolist().index("PFC")
    representation_index = arrays["representation_names"].tolist().index("units")
    metric_index = arrays["metric_names"].tolist().index("balanced_accuracy")
    values = arrays["evaluation_scores"][
        window_index,
        pair_index,
        0,
        :,
        region_index,
        representation_index,
        :,
        metric_index,
    ]
    return np.mean(values, axis=0)


def _gap_curve(
    saved: dict[str, object],
    *,
    window_name: str,
    condition: str,
) -> np.ndarray:
    """Return mean-fold PFC direct-unit balanced-accuracy gaps through time.

    Parameters
    ----------
    saved : dict[str, object]
        Validated pooled run returned by :func:`results.load_pooled_run`.
    window_name, condition : str
        Window and destination-condition coordinates.

    Returns
    -------
    numpy.ndarray
        Native-minus-transfer gap shaped ``(time=40,)`` in fraction units.
    """
    arrays = saved["arrays"]
    window_index = arrays["training_window_names"].tolist().index(window_name)
    condition_index = arrays["condition_names"].tolist().index(condition)
    region_index = arrays["region_names"].tolist().index("PFC")
    representation_index = arrays["representation_names"].tolist().index("units")
    metric_index = arrays["metric_names"].tolist().index("balanced_accuracy")
    values = arrays["gap_scores"][
        window_index,
        condition_index,
        0,
        :,
        region_index,
        representation_index,
        :,
        metric_index,
    ]
    return np.mean(values, axis=0)


def _assert_no_unexplained_unavailability(saved: dict[str, object]) -> None:
    """Require every unavailable scientific status to carry a reason index.

    Parameters
    ----------
    saved : dict[str, object]
        Validated pooled run with status/reason array pairs.

    Returns
    -------
    None
        Raises an assertion when any unavailable cell has reason ``-1``.
    """
    arrays = saved["arrays"]
    for prefix in ("transform", "model", "evaluation", "gap"):
        status = arrays[f"{prefix}_status"]
        reasons = arrays[f"{prefix}_reason_index"]
        assert np.all(reasons[status == 2] >= 0), prefix


def test_seeded_scientific_cases_use_the_public_synchronous_path(tmp_path, monkeypatch):
    """Exercise every required seeded outcome through the synchronous API.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned root for configurations, checkpoints, and final NPZ files.
    monkeypatch : pytest.MonkeyPatch
        Patcher that injects prepared synthetic arrays at the input-loading seam.

    Returns
    -------
    None
        Asserts stable, rotating, transfer, remap, absent, delayed,
        constant-reference, overlap, and multi-window scientific outcomes.
    """
    saved_cases: dict[str, dict[str, object]] = {}
    contexts: dict[str, pipeline.PooledComputationContext] = {}
    for case in (
        "stable",
        "rotating",
        "remapped",
        "absent",
        "delayed",
        "constant_reference",
    ):
        config, windows = _load_config(tmp_path / case, window_names=("full",))
        context = _synthetic_context(config, windows, case=case)
        contexts[case] = context
        saved_cases[case] = _run_case(
            monkeypatch,
            config,
            windows,
            context,
            tmp_path / f"{case}-work",
        )
        _assert_no_unexplained_unavailability(saved_cases[case])

    stable_transfer = _score_curve(
        saved_cases["stable"],
        window_name="full",
        train_condition=_REFERENCE,
        test_condition=_DESTINATION,
    )
    stable_native = _score_curve(
        saved_cases["stable"],
        window_name="full",
        train_condition=_DESTINATION,
        test_condition=_DESTINATION,
    )
    assert np.min(stable_transfer) > 0.95
    assert np.min(stable_native) > 0.95
    np.testing.assert_allclose(
        _gap_curve(
            saved_cases["stable"],
            window_name="full",
            condition=_DESTINATION,
        ),
        stable_native - stable_transfer,
        atol=1e-12,
    )

    rotating_curve = _score_curve(
        saved_cases["rotating"],
        window_name="full",
        train_condition=_DESTINATION,
        test_condition=_DESTINATION,
    )
    assert np.max(rotating_curve) <= 0.6
    rotating_context = contexts["rotating"]
    split = rotating_context.outer_splits["current_action"].splits[0]
    target_values = rotating_context.arrays["target_values"][0]
    oracle_scores = []
    for time_index in range(40):
        oracle = modeling.fit_and_score(
            modeling.make_estimator(target_family="categorical"),
            rotating_context.pfc_rate_tensor_hz[split.train_indices, time_index],
            target_values[split.train_indices],
            rotating_context.pfc_rate_tensor_hz[split.test_indices, time_index],
            target_values[split.test_indices],
            target_family="categorical",
        )
        assert oracle.is_valid
        oracle_scores.append(oracle.metrics["balanced_accuracy"])
    assert min(oracle_scores) > 0.95

    remapped_transfer = _score_curve(
        saved_cases["remapped"],
        window_name="full",
        train_condition=_REFERENCE,
        test_condition=_DESTINATION,
    )
    remapped_native = _score_curve(
        saved_cases["remapped"],
        window_name="full",
        train_condition=_DESTINATION,
        test_condition=_DESTINATION,
    )
    assert np.max(remapped_transfer) < 0.05
    assert np.min(remapped_native) > 0.95
    remapped_gap = _gap_curve(
        saved_cases["remapped"],
        window_name="full",
        condition=_DESTINATION,
    )
    np.testing.assert_allclose(
        remapped_gap,
        remapped_native - remapped_transfer,
        atol=1e-12,
    )
    assert np.min(remapped_gap) > 0.9

    absent_transfer = _score_curve(
        saved_cases["absent"],
        window_name="full",
        train_condition=_REFERENCE,
        test_condition=_DESTINATION,
    )
    absent_native = _score_curve(
        saved_cases["absent"],
        window_name="full",
        train_condition=_DESTINATION,
        test_condition=_DESTINATION,
    )
    np.testing.assert_allclose(absent_transfer, 0.5, atol=0.05)
    np.testing.assert_allclose(absent_native, 0.5, atol=0.05)

    delayed_transfer = _score_curve(
        saved_cases["delayed"],
        window_name="full",
        train_condition=_REFERENCE,
        test_condition=_DESTINATION,
    )
    assert np.max(delayed_transfer[:20]) <= 0.6
    assert np.min(delayed_transfer[25:30]) > 0.95

    constant_arrays = saved_cases["constant_reference"]["arrays"]
    pairs = list(
        zip(
            constant_arrays["evaluation_train_condition"].tolist(),
            constant_arrays["evaluation_test_condition"].tolist(),
            strict=True,
        )
    )
    transfer_index = pairs.index((_REFERENCE, _DESTINATION))
    native_index = pairs.index((_DESTINATION, _DESTINATION))
    balanced_accuracy = constant_arrays["metric_names"].tolist().index(
        "balanced_accuracy"
    )
    assert np.all(
        constant_arrays["evaluation_status"][
            0, transfer_index, 0, :, :, :, :, balanced_accuracy
        ]
        == 2
    )
    assert np.all(
        constant_arrays["evaluation_status"][
            0, native_index, 0, :, :, :, :, balanced_accuracy
        ]
        == 1
    )

    overlap_config, overlap_windows = _load_config(
        tmp_path / "overlap",
        window_names=("full",),
        overlapping_conditions=True,
    )
    overlap_context = _synthetic_context(overlap_config, overlap_windows, case="overlap")
    overlap_saved = _run_case(
        monkeypatch,
        overlap_config,
        overlap_windows,
        overlap_context,
        tmp_path / "overlap-work",
    )
    for split in overlap_context.outer_splits["current_action"].splits:
        assert not np.intersect1d(split.train_indices, split.test_indices).size
        train_blocks = set(overlap_context.target_table.iloc[split.train_indices]["block_id"])
        test_blocks = set(overlap_context.target_table.iloc[split.test_indices]["block_id"])
        assert train_blocks.isdisjoint(test_blocks)
    assert np.min(
        _score_curve(
            overlap_saved,
            window_name="full",
            train_condition=_REFERENCE,
            test_condition="all",
        )
    ) > 0.95

    multi_config, multi_windows = _load_config(
        tmp_path / "multi",
        window_names=("pre_choice", "post_choice", "full"),
    )
    multi_context = _synthetic_context(multi_config, multi_windows, case="rotating")
    multi_saved = _run_case(
        monkeypatch,
        multi_config,
        multi_windows,
        multi_context,
        tmp_path / "multi-work",
    )
    pre_config, pre_windows = _load_config(
        tmp_path / "pre-only",
        window_names=("pre_choice",),
    )
    pre_context = _synthetic_context(pre_config, pre_windows, case="rotating")
    pre_saved = _run_case(
        monkeypatch,
        pre_config,
        pre_windows,
        pre_context,
        tmp_path / "pre-only-work",
    )
    assert len(list((tmp_path / "multi-work" / "checkpoints").glob("*.npz"))) == 3
    for name, axes in multi_saved["metadata"]["axes"].items():
        if axes and axes[0] == "window" and name != "training_window_names":
            np.testing.assert_array_equal(
                multi_saved["arrays"][name][0],
                pre_saved["arrays"][name][0],
            )
    model_coefficients = multi_saved["arrays"]["model_coefficients"]
    assert not np.array_equal(model_coefficients[0], model_coefficients[1])
    assert not np.array_equal(model_coefficients[0], model_coefficients[2])


def test_full_window_fits_once_per_model_identity(tmp_path, monkeypatch):
    """Prove forty evaluation bins do not multiply estimator fitting.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned root for one full-window result.
    monkeypatch : pytest.MonkeyPatch
        Patcher used to count estimator fits and inject synthetic input arrays.

    Returns
    -------
    None
        Asserts one fit per condition/target/fold/region/representation identity.
    """
    config, windows = _load_config(tmp_path / "config", window_names=("full",))
    context = _synthetic_context(config, windows, case="stable")
    original_fit = modeling.fit_estimator
    fit_count = 0

    def counted_fit(*args, **kwargs):
        """Count and delegate one estimator fit with unchanged contracts.

        Parameters
        ----------
        *args : object
            Positional arguments accepted by :func:`modeling.fit_estimator`.
        **kwargs : object
            Keyword arguments including target family and optional row weights.

        Returns
        -------
        modeling.FittedEstimator
            Original fitted-estimator result with its feature-axis contract.
        """
        nonlocal fit_count
        fit_count += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(modeling, "fit_estimator", counted_fit)
    saved = _run_case(
        monkeypatch,
        config,
        windows,
        context,
        tmp_path / "work",
    )
    expected_fit_count = 2 * 1 * 3 * 3 * 2
    assert fit_count == expected_fit_count
    assert np.count_nonzero(saved["arrays"]["model_status"] == 1) == expected_fit_count


def test_one_shot_and_interrupted_synthetic_runs_are_exact(tmp_path, monkeypatch):
    """Require resume to preserve checkpoints and exact validated results.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned roots for one-shot and interrupted work directories.
    monkeypatch : pytest.MonkeyPatch
        Patcher used to inject synthetic inputs and one controlled interruption.

    Returns
    -------
    None
        Asserts byte-stable restored checkpoints and equal arrays/metadata.
    """
    config, windows = _load_config(
        tmp_path / "config",
        window_names=("pre_choice", "post_choice"),
    )
    context = _synthetic_context(config, windows, case="stable")
    one_shot = _run_case(
        monkeypatch,
        config,
        windows,
        context,
        tmp_path / "one-shot",
    )

    monkeypatch.setattr(
        pipeline,
        "_prepare_pooled_computation_context",
        lambda *_: _fresh_context(context),
    )
    original_save = results.save_pooled_checkpoint
    saved_count = 0

    def interrupt_after_first(path, *, checkpoint_key, run_fingerprint, result):
        """Publish one valid checkpoint, then simulate process interruption.

        Parameters
        ----------
        path : pathlib.Path
            New checkpoint path.
        checkpoint_key, run_fingerprint : str
            Exact window-target and complete run identities.
        result : decoding.PooledWindowTargetResult
            Complete first scientific checkpoint cell.

        Returns
        -------
        None
            Raises ``KeyboardInterrupt`` after the first immutable publication.
        """
        nonlocal saved_count
        original_save(
            path,
            checkpoint_key=checkpoint_key,
            run_fingerprint=run_fingerprint,
            result=result,
        )
        saved_count += 1
        if saved_count == 1:
            raise KeyboardInterrupt("synthetic PW-WP4 interruption")

    monkeypatch.setattr(results, "save_pooled_checkpoint", interrupt_after_first)
    interrupted_directory = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt, match="PW-WP4"):
        pipeline.compute_or_resume_pooled_results(
            config,
            windows,
            interrupted_directory,
        )
    first_checkpoint = next((interrupted_directory / "checkpoints").glob("*.npz"))
    checkpoint_bytes = first_checkpoint.read_bytes()

    monkeypatch.setattr(results, "save_pooled_checkpoint", original_save)
    pipeline.compute_or_resume_pooled_results(
        config,
        windows,
        interrupted_directory,
    )
    resumed = results.load_pooled_run(interrupted_directory)
    assert first_checkpoint.read_bytes() == checkpoint_bytes
    assert resumed["metadata"] == one_shot["metadata"]
    for name in one_shot["arrays"]:
        np.testing.assert_array_equal(resumed["arrays"][name], one_shot["arrays"][name])
    _assert_no_unexplained_unavailability(resumed)
    for prefix in ("transform", "model", "evaluation", "gap"):
        assert not np.any(resumed["arrays"][f"{prefix}_status"] == 2)
