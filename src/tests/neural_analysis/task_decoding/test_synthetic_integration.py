"""WP9 seeded end-to-end contracts for task-variable decoding.

The fixture writes a complete temporary metadata session and uses the real
spike loader, Pynapple binning, grouped models, checkpoints, result serializer,
and plotting code. It never reads experimental data or launches external work.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.task_decoding import (
    modeling,
    pipeline,
    plotting,
    results,
    run_session,
)


_SEED = 20261007
_TRIALS_PER_BLOCK = 12
_BLOCK_COUNT = 6
_TRIAL_COUNT = _TRIALS_PER_BLOCK * _BLOCK_COUNT
_BIN_WIDTH_S = 0.5
_BIN_EDGES_S = np.arange(-2.0, 2.0 + _BIN_WIDTH_S, _BIN_WIDTH_S)
_SIGNAL_BIN_INDEX = 4
_REGIONS = ("PFC", "HPC", "PFC+HPC")
_REPRESENTATIONS = ("pca", "units")


@dataclass(frozen=True)
class SyntheticSession:
    """Paths and identities for one complete temporary synthetic session.

    Attributes
    ----------
    session_root : pathlib.Path
        Temporary session root containing metadata and generated inputs.
    config_path : pathlib.Path
        Portable task-decoding configuration inside ``session_root``.
    augmented_path : pathlib.Path
        Chronological 72-row augmented trial CSV.
    probe_alignment_paths : tuple[pathlib.Path, pathlib.Path]
        PFC and HPC aligned-spike NPZ paths. Spike timestamps use UTC seconds.
    offset_trial_positions : numpy.ndarray
        Zero-based trial positions assigned to current-action outer fold zero.
    """

    session_root: Path
    config_path: Path
    augmented_path: Path
    probe_alignment_paths: tuple[Path, Path]
    offset_trial_positions: np.ndarray


@dataclass(frozen=True)
class SyntheticRuns:
    """Completed base, repeat, and held-out-offset integration runs.

    Attributes
    ----------
    base_session, offset_session : SyntheticSession
        Input trees differing only by the declared held-out spike offset.
    base_directory, repeat_directory, offset_directory : pathlib.Path
        Completed immutable run directories beneath temporary session roots.
    base, repeat, offset : dict[str, object]
        Validated mappings returned by :func:`load_task_decoding_run`.
    """

    base_session: SyntheticSession
    offset_session: SyntheticSession
    base_directory: Path
    repeat_directory: Path
    offset_directory: Path
    base: dict[str, object]
    repeat: dict[str, object]
    offset: dict[str, object]


def _trial_table() -> pd.DataFrame:
    """Build the complete synthetic chronological behavior table.

    Returns
    -------
    pandas.DataFrame
        Table shaped ``(72, 8)``. Choice times are UTC seconds, actions and
        states are encoded classes, and relative doubt is unitless.
    """
    blocks = np.repeat(np.arange(_BLOCK_COUNT, dtype=int), _TRIALS_PER_BLOCK)
    within_block = np.tile(np.arange(_TRIALS_PER_BLOCK, dtype=int), _BLOCK_COUNT)
    actions = within_block % 2
    # Class one occurs only in block zero, making three-fold grouped
    # classification impossible while retaining both classes globally.
    states = np.zeros(_TRIAL_COUNT, dtype=int)
    states[blocks == 0] = actions[blocks == 0]
    relative_doubt = np.tile(
        np.linspace(-1.0, 1.0, _TRIALS_PER_BLOCK, dtype=float),
        _BLOCK_COUNT,
    )
    return pd.DataFrame(
        {
            "cur_trial": np.arange(_TRIAL_COUNT, dtype=int),
            "cur_block": blocks,
            "action": actions,
            "state_int": states,
            "experimenter_reward_given": np.zeros(_TRIAL_COUNT, dtype=int),
            "choice_time": 10.0 + 6.0 * np.arange(_TRIAL_COUNT, dtype=float),
            "relative_doubt_index": relative_doubt,
            "reward": actions,
        }
    )


def _fold_zero_trial_positions(trial_table: pd.DataFrame) -> np.ndarray:
    """Return current-action trial positions held out by outer fold zero.

    Parameters
    ----------
    trial_table : pandas.DataFrame
        Complete synthetic table with 72 rows and encoded action/block columns.

    Returns
    -------
    numpy.ndarray
        Ascending zero-based trial positions with shape ``(n_fold_zero,)``.
    """
    plan = modeling.make_outer_splits(
        trial_table["action"].to_numpy(dtype=float),
        trial_table["cur_block"].to_numpy(),
        target_family="categorical",
        fold_count=3,
    )
    assert plan.is_available
    return np.flatnonzero(plan.fold_ids == 0).astype(int)


def _spike_arrays(
    trial_table: pd.DataFrame,
    *,
    seed: int,
    held_out_offset_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate seeded event-aligned spikes for four selected units.

    Parameters
    ----------
    trial_table : pandas.DataFrame
        Chronological synthetic trials. ``choice_time`` is in UTC seconds,
        actions are classes, and relative doubt is unitless.
    seed : int
        NumPy random seed controlling background Poisson counts and spike
        positions.
    held_out_offset_positions : numpy.ndarray
        Trial positions receiving ten additional spikes per bin and unit. An
        empty array produces the base session.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Sorted-by-unit flat spike times in UTC seconds and aligned integer
        cluster IDs, each shaped ``(n_spikes,)``.
    """
    rng = np.random.default_rng(seed)
    offset_rng = np.random.default_rng(seed + 1_000_000)
    offset_positions = set(np.asarray(held_out_offset_positions, dtype=int).tolist())
    unit_spikes: dict[int, list[float]] = {cluster_id: [] for cluster_id in range(4)}
    first_choice = float(trial_table["choice_time"].iloc[0])
    last_choice = float(trial_table["choice_time"].iloc[-1])
    for unit_spike_times in unit_spikes.values():
        # Pynapple derives the group's time support from spikes. These anchors
        # cover every trial window without entering any analyzed window.
        unit_spike_times.extend([first_choice - 2.5, last_choice + 2.5])

    for trial_position, row in trial_table.iterrows():
        action = int(row["action"])
        doubt = float(row["relative_doubt_index"])
        choice_time_s = float(row["choice_time"])
        doubt_level = int(round((doubt + 1.0) * 3.0))
        for bin_index, (start_s, end_s) in enumerate(
            zip(_BIN_EDGES_S[:-1], _BIN_EDGES_S[1:], strict=True)
        ):
            for cluster_id in range(4):
                count = int(rng.poisson(2.0))
                if bin_index == _SIGNAL_BIN_INDEX:
                    if cluster_id == 0:
                        count += action
                    elif cluster_id == 2:
                        count += doubt_level
                    elif cluster_id == 3:
                        count += 6 - doubt_level
                if count:
                    relative_spikes = rng.uniform(
                        start_s + 0.01,
                        end_s - 0.01,
                        size=count,
                    )
                    unit_spikes[cluster_id].extend(
                        (choice_time_s + relative_spikes).tolist()
                    )
                if trial_position in offset_positions:
                    offset_spikes = offset_rng.uniform(
                        start_s + 0.01,
                        end_s - 0.01,
                        size=10,
                    )
                    unit_spikes[cluster_id].extend(
                        (choice_time_s + offset_spikes).tolist()
                    )

    spike_times: list[float] = []
    spike_clusters: list[int] = []
    for cluster_id, times in unit_spikes.items():
        ordered = sorted(times)
        spike_times.extend(ordered)
        spike_clusters.extend([cluster_id] * len(ordered))
    return np.asarray(spike_times, dtype=float), np.asarray(spike_clusters, dtype=np.int64)


def _write_probe(
    session_root: Path,
    *,
    probe_id: str,
    region_seed: int,
    trial_table: pd.DataFrame,
    held_out_offset_positions: np.ndarray,
) -> tuple[dict[str, object], Path]:
    """Write one complete four-unit synthetic probe source.

    Parameters
    ----------
    session_root : pathlib.Path
        Temporary session root receiving the probe directory.
    probe_id : str
        Metadata identifier unique within the session.
    region_seed : int
        Region-specific NumPy seed.
    trial_table : pandas.DataFrame
        Synthetic trial table whose choice times use UTC seconds.
    held_out_offset_positions : numpy.ndarray
        Trial positions receiving the optional held-out-only spike offset.

    Returns
    -------
    tuple[dict[str, object], pathlib.Path]
        Portable metadata probe record and aligned-spike NPZ path.
    """
    probe_root = session_root / "probes" / probe_id
    sorter_root = probe_root / "sorter"
    sorter_root.mkdir(parents=True)
    spike_times, spike_clusters = _spike_arrays(
        trial_table,
        seed=region_seed,
        held_out_offset_positions=held_out_offset_positions,
    )
    np.save(sorter_root / "spike_times.npy", spike_times)
    np.save(sorter_root / "spike_clusters.npy", spike_clusters)
    pd.DataFrame(
        {
            "cluster_id": np.arange(4, dtype=int),
            "ch": np.arange(4, dtype=int),
            "group": ["good"] * 4,
        }
    ).to_csv(sorter_root / "cluster_info.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "channel_id": np.arange(4, dtype=int),
            "label": ["good"] * 4,
            "inside_brain": [True] * 4,
            "x_um": np.arange(4, dtype=float) * 20.0,
            "y_um": np.zeros(4, dtype=float),
        }
    ).to_csv(probe_root / "channel_quality.csv", index=False)
    (probe_root / "lfp.dat").write_bytes(b"synthetic-lfp-placeholder\n")
    alignment_path = probe_root / "aligned_spikes.npz"
    first_choice = float(trial_table["choice_time"].iloc[0])
    last_choice = float(trial_table["choice_time"].iloc[-1])
    np.savez(
        alignment_path,
        spike_utc_unix=spike_times,
        irig_utc_unix=np.array([first_choice - 3.0, last_choice + 3.0]),
    )
    prefix = f"probes/{probe_id}"
    metadata = {
        "lfp": f"{prefix}/lfp.dat",
        "alignment": f"{prefix}/aligned_spikes.npz",
        "sorter": f"{prefix}/sorter",
        "quality": f"{prefix}/channel_quality.csv",
        "sites": {},
        "unit_channels": [0, 1, 2, 3],
    }
    return metadata, alignment_path


def _write_synthetic_session(
    parent: Path,
    *,
    held_out_offset: bool,
) -> SyntheticSession:
    """Write one portable, fully executable two-probe session.

    Parameters
    ----------
    parent : pathlib.Path
        Pytest-owned directory receiving one session root.
    held_out_offset : bool
        Whether outer-fold-zero test trials receive a large neural-only offset.

    Returns
    -------
    SyntheticSession
        Paths and fold-zero trial identities for the generated session.
    """
    session_root = parent / ("offset-session" if held_out_offset else "base-session")
    processed = session_root / "processed"
    processed.mkdir(parents=True)
    trial_table = _trial_table()
    augmented_path = processed / "augmented_trials.csv"
    trial_table.to_csv(augmented_path, index=False)
    (processed / "behavior_trials.csv").write_text("trial\n0\n", encoding="ascii")
    (processed / "trial_feature_params.json").write_text(
        '{"feature_schema":"wp9-synthetic-v1","seed":20261007}\n',
        encoding="ascii",
    )
    offset_positions = _fold_zero_trial_positions(trial_table)
    applied_offset = offset_positions if held_out_offset else np.empty(0, dtype=int)
    pfc_metadata, pfc_alignment = _write_probe(
        session_root,
        probe_id="probe-pfc",
        region_seed=_SEED,
        trial_table=trial_table,
        held_out_offset_positions=applied_offset,
    )
    hpc_metadata, hpc_alignment = _write_probe(
        session_root,
        probe_id="probe-hpc",
        region_seed=_SEED + 1,
        trial_table=trial_table,
        held_out_offset_positions=applied_offset,
    )
    (session_root / "neural_session.json").write_text(
        json.dumps(
            {
                "schema_version": "2",
                "session": "wp9-seeded-synthetic",
                "acquisition": "open_ephys",
                "behavior": {"trials": "processed/behavior_trials.csv"},
                "probes": {
                    "probe-pfc": pfc_metadata,
                    "probe-hpc": hpc_metadata,
                },
                "site_pairs": [],
            }
        ),
        encoding="ascii",
    )
    config_path = session_root / "task_decoding_config.json"
    config_path.write_text(
        json.dumps(
            {
                "session_metadata_path": "neural_session.json",
                "augmented_trial_path": "processed/augmented_trials.csv",
                "trial_feature_parameter_path": "processed/trial_feature_params.json",
                "pfc_region": {
                    "region": "PFC",
                    "probe_id": "probe-pfc",
                    "channel_labels": ["good"],
                    "require_inside_brain": True,
                    "cluster_groups": ["good"],
                    "channel_ids": [],
                },
                "hpc_region": {
                    "region": "HPC",
                    "probe_id": "probe-hpc",
                    "channel_labels": ["good"],
                    "require_inside_brain": True,
                    "cluster_groups": ["good"],
                    "channel_ids": [],
                },
                "alignment": "choice_time",
                "bin_width_ms": 500,
                "pfc_pc_count": 2,
                "hpc_pc_count": 2,
                "target_names": ["current_state", "current_action", "relative_doubt"],
                "regularization_mode": "fixed",
                "outer_fold_count": 3,
                "inner_fold_count": 3,
                "trusted_utc_bounds": {},
                "output_root": "analysis_runs",
            }
        ),
        encoding="ascii",
    )
    return SyntheticSession(
        session_root=session_root,
        config_path=config_path,
        augmented_path=augmented_path,
        probe_alignment_paths=(pfc_alignment, hpc_alignment),
        offset_trial_positions=offset_positions,
    )


def _complete_run(session: SyntheticSession) -> tuple[Path, dict[str, object]]:
    """Execute and reload one real foreground synthetic run.

    Parameters
    ----------
    session : SyntheticSession
        Complete temporary session with portable configuration and spike data.

    Returns
    -------
    tuple[pathlib.Path, dict[str, object]]
        Completed immutable run directory and validated saved-result mapping.
    """
    run_directory = pipeline.prepare_task_decoding_run(
        session.config_path,
        rerun=True,
        execution_mode="foreground",
    )
    pipeline.run_prepared_task_decoding(run_directory)
    return run_directory, results.load_task_decoding_run(run_directory)


@pytest.fixture(scope="module")
def synthetic_runs(tmp_path_factory: pytest.TempPathFactory) -> SyntheticRuns:
    """Run base, deterministic-repeat, and held-out-offset sessions once.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory providing one module-owned temporary root.

    Returns
    -------
    SyntheticRuns
        Three completed real-pipeline runs and their validated saved mappings.
    """
    root = tmp_path_factory.mktemp("wp9-synthetic")
    base_session = _write_synthetic_session(root, held_out_offset=False)
    offset_session = _write_synthetic_session(root, held_out_offset=True)
    patcher = pytest.MonkeyPatch()
    # Cleanliness behavior is already covered independently. Integration tests
    # must not depend on unrelated working-tree state while source fingerprints
    # and execution-time identity revalidation remain real.
    patcher.setattr(
        pipeline.results,
        "validate_scientific_source_cleanliness",
        lambda _repository_root: None,
    )
    try:
        base_directory, base = _complete_run(base_session)
        repeat_directory, repeat = _complete_run(base_session)
        offset_directory, offset = _complete_run(offset_session)
        yield SyntheticRuns(
            base_session=base_session,
            offset_session=offset_session,
            base_directory=base_directory,
            repeat_directory=repeat_directory,
            offset_directory=offset_directory,
            base=base,
            repeat=repeat,
            offset=offset,
        )
    finally:
        patcher.undo()
        plt.close("all")


def _target_index(saved_run: dict[str, object], label: str) -> int:
    """Return one exact target-axis index from a validated saved run.

    Parameters
    ----------
    saved_run : dict[str, object]
        Mapping returned by :func:`load_task_decoding_run`.
    label : str
        Exact configured target identifier.

    Returns
    -------
    int
        Zero-based target-axis position.
    """
    labels = saved_run["arrays"]["target_labels"].tolist()
    return labels.index(label)


def test_seeded_two_region_session_completes_and_publishes(synthetic_runs):
    """The real pipeline reaches canonical complete state with every artifact."""
    run_directory = synthetic_runs.base_directory
    state = json.loads((run_directory / "run_state.json").read_text(encoding="utf-8"))
    assert state["lifecycle"] == "complete"
    assert state["final_results_published"] is True
    assert state["completed_targets"] == [
        "current_state",
        "current_action",
        "relative_doubt",
    ]
    assert {path.name for path in (run_directory / "checkpoints").glob("*.npz")} == {
        "current_state.npz",
        "current_action.npz",
        "relative_doubt.npz",
    }
    assert (run_directory / "results.npz").is_file()
    assert (run_directory / "summary.md").is_file()
    usage = json.loads(
        (run_directory / "resource_usage.json").read_text(encoding="utf-8")
    )
    assert usage["schema_version"] == 1
    assert usage["measurement_method"] == "resource.getrusage"
    assert usage["status"] == "complete"
    assert usage["completed_targets"] == [
        "current_state",
        "current_action",
        "relative_doubt",
    ]
    assert usage["resource_envelope"] == {
        "full_trial_count": 72,
        "tensor_trial_count": 72,
        "pfc_unit_count": 4,
        "hpc_unit_count": 4,
        "time_bin_count": 8,
        "target_count": 3,
        "condition_count": 1,
        "outer_fold_count": 3,
        "inner_fold_count": 3,
        "coefficient_feature_capacity": 8,
        "categorical_fit_count": 288,
        "numerical_fit_count": 144,
        "tensor_allocation_bytes": 36_864,
    }
    assert set(usage["target_measurements"]) == {
        "current_state",
        "current_action",
        "relative_doubt",
    }
    assert usage["fit_counts"]["requested"] == 432
    assert usage["fit_counts"]["measured_estimator_calls"] == 288
    assert usage["peak_rss_bytes"] > 0
    assert usage["wall_time_seconds"] >= 0.0
    assert usage["user_cpu_seconds"] >= 0.0
    assert usage["system_cpu_seconds"] >= 0.0
    assert usage["output_size_bytes"] > 0
    summary = (run_directory / "summary.md").read_text(encoding="utf-8")
    assert "resource.getrusage" in summary
    assert "Peak RSS:" in summary
    assert "CPU time:" in summary
    assert "resource_usage.json" in summary
    assert {path.name for path in (run_directory / "figures").glob("*.png")} == {
        "categorical_balanced_accuracy.png",
        "categorical_auc.png",
        "numerical_r2.png",
    }


def test_time_local_signal_is_strongest_in_the_injected_interval(synthetic_runs):
    """Mean categorical decoding peaks in the injected bin without score tuning."""
    arrays = synthetic_runs.base["arrays"]
    target = _target_index(synthetic_runs.base, "current_action")
    balanced_accuracy = arrays["fold_scores"][target, :, :, 0, :, :]
    mean_by_time = np.nanmean(balanced_accuracy, axis=(0, 1, 3))
    strongest_time_index = int(np.argmax(mean_by_time))
    assert strongest_time_index == _SIGNAL_BIN_INDEX
    assert 0.0 <= float(arrays["time_bin_centers_s"][strongest_time_index]) < 0.5


def test_all_regions_representations_and_target_families_complete(synthetic_runs):
    """Available categorical and numerical targets fill the full six-cell grid."""
    arrays = synthetic_runs.base["arrays"]
    assert arrays["region_labels"].tolist() == list(_REGIONS)
    assert arrays["representation_labels"].tolist() == list(_REPRESENTATIONS)
    for label, family in (
        ("current_action", "categorical"),
        ("relative_doubt", "numerical"),
    ):
        target = _target_index(synthetic_runs.base, label)
        assert arrays["target_families"][target] == family
        assert arrays["target_status"][target] == "available"
        assert arrays["fit_status"][target].shape[:2] == (3, 2)
        for region_index, representation_index in np.ndindex(3, 2):
            statuses = arrays["fit_status"][
                target,
                region_index,
                representation_index,
            ]
            assert np.any(statuses == "valid")
            assert set(statuses.ravel()) <= {"valid", "unavailable"}
        valid = arrays["fit_status"][target] == "valid"
        assert np.isfinite(arrays["fitted_intercepts"][target][valid]).all()


def test_impossible_grouped_target_is_unavailable_without_stopping_others(synthetic_runs):
    """One invalid categorical split remains local to its target."""
    arrays = synthetic_runs.base["arrays"]
    impossible = _target_index(synthetic_runs.base, "current_state")
    assert arrays["target_status"][impossible] == "unavailable"
    assert "categorical grouped fold lacks both classes" in arrays[
        "target_unavailable_reasons"
    ][impossible]
    assert np.all(arrays["outer_fold_ids"][impossible] == -1)
    assert np.isnan(arrays["fold_scores"][impossible]).all()
    assert arrays["target_status"].tolist() == [
        "unavailable",
        "available",
        "available",
    ]


def test_saved_reload_reproduces_heatmaps_and_coefficient_summaries(
    monkeypatch,
    synthetic_runs,
):
    """Validated saved arrays reproduce display summaries without model fitting."""
    monkeypatch.setattr(
        pipeline.modeling,
        "decode_target",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("saved-result rendering must not decode")
        ),
    )
    reloaded = results.load_task_decoding_run(synthetic_runs.base_directory)
    first_figure = plotting.plot_decoding_heatmap(
        synthetic_runs.base,
        family="categorical",
        metric="balanced_accuracy",
    )
    second_figure = plotting.plot_decoding_heatmap(
        reloaded,
        family="categorical",
        metric="balanced_accuracy",
    )
    try:
        first_images = [axis.images[0].get_array() for axis in first_figure.axes if axis.images]
        second_images = [axis.images[0].get_array() for axis in second_figure.axes if axis.images]
        assert len(first_images) == len(second_images) == 6
        for first_image, second_image in zip(first_images, second_images, strict=True):
            np.testing.assert_allclose(first_image, second_image, equal_nan=True)
    finally:
        plt.close(first_figure)
        plt.close(second_figure)
    first_tables = plotting.summarize_unit_coefficients(
        synthetic_runs.base,
        target="current_action",
        region="PFC+HPC",
        time_bin_index=_SIGNAL_BIN_INDEX,
    )
    second_tables = plotting.summarize_unit_coefficients(
        reloaded,
        target="current_action",
        region="PFC+HPC",
        time_bin_index=_SIGNAL_BIN_INDEX,
    )
    for first_table, second_table in zip(first_tables, second_tables, strict=True):
        pd.testing.assert_frame_equal(first_table, second_table)


def test_repeated_seeded_execution_is_scientifically_deterministic(synthetic_runs):
    """A rerun changes timing/provenance only, not saved scientific arrays."""
    first_arrays = synthetic_runs.base["arrays"]
    second_arrays = synthetic_runs.repeat["arrays"]
    assert synthetic_runs.base["run_fingerprint"] == synthetic_runs.repeat[
        "run_fingerprint"
    ]
    excluded = {"stage_timing_seconds", "total_timing_seconds"}
    assert set(first_arrays) == set(second_arrays)
    for name in sorted(set(first_arrays) - excluded):
        first = first_arrays[name]
        second = second_arrays[name]
        if np.issubdtype(first.dtype, np.number):
            np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0, equal_nan=True)
        else:
            np.testing.assert_array_equal(first, second)


def test_held_out_only_offset_does_not_change_training_derived_models(synthetic_runs):
    """Fold-zero scaling, PCA, and estimators ignore its held-out neural offset."""
    base_arrays = synthetic_runs.base["arrays"]
    offset_arrays = synthetic_runs.offset["arrays"]
    target = _target_index(synthetic_runs.base, "current_action")
    assert synthetic_runs.base_session.probe_alignment_paths[0].read_bytes() != (
        synthetic_runs.offset_session.probe_alignment_paths[0].read_bytes()
    )
    assert synthetic_runs.base_session.offset_trial_positions.size > 0

    base_fold_zero_coefficients = base_arrays["coefficient_values"][target, :, :, :, 0, :]
    offset_fold_zero_coefficients = offset_arrays["coefficient_values"][
        target, :, :, :, 0, :
    ]
    np.testing.assert_allclose(
        base_fold_zero_coefficients,
        offset_fold_zero_coefficients,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        base_arrays["fitted_intercepts"][target, :, :, :, 0],
        offset_arrays["fitted_intercepts"][target, :, :, :, 0],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    other_fold_coefficients = base_arrays["coefficient_values"][target, :, :, :, 1:, :]
    offset_other_fold_coefficients = offset_arrays["coefficient_values"][
        target, :, :, :, 1:, :
    ]
    assert not np.allclose(
        other_fold_coefficients,
        offset_other_fold_coefficients,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux detached-session test")
def test_actual_detached_launcher_can_publish_resource_evidence(tmp_path):
    """The real detached process boundary preserves atomic measurement output."""
    run_directory = tmp_path / "detached-resource-smoke"
    run_directory.mkdir()
    repository_root = Path(__file__).resolve().parents[4]
    envelope = {
        "full_trial_count": 1,
        "tensor_trial_count": 1,
        "pfc_unit_count": 1,
        "hpc_unit_count": 1,
        "time_bin_count": 1,
        "target_count": 1,
        "condition_count": 1,
        "outer_fold_count": 3,
        "inner_fold_count": 3,
        "coefficient_feature_capacity": 2,
        "categorical_fit_count": 18,
        "numerical_fit_count": 0,
        "tensor_allocation_bytes": 16,
    }
    child_code = "\n".join(
        (
            "import json, sys",
            f"sys.path.insert(0, {str(repository_root)!r})",
            "from src.neural_analysis.task_decoding.resource_usage import ResourceUsageTracker",
            f"tracker = ResourceUsageTracker({str(run_directory)!r}, json.loads({json.dumps(envelope)!r}))",
            "tracker.record_restored_target(target_label='synthetic_action', target_family='categorical', requested_fit_count=18, valid_outer_cell_count=18, invalid_outer_cell_count=0)",
            "tracker.snapshot(status='complete', completed_targets=('synthetic_action',))",
        )
    )

    run_session._launch_detached_prepared(
        run_directory=run_directory,
        child_argv=(sys.executable, "-c", child_code),
    )
    usage_path = run_directory / "resource_usage.json"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not usage_path.is_file():
        time.sleep(0.02)

    assert usage_path.is_file(), (run_directory / "console.log").read_text(
        encoding="utf-8"
    )
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    assert usage["status"] == "complete"
    assert usage["resource_envelope"] == envelope
    assert usage["peak_rss_bytes"] > 0
