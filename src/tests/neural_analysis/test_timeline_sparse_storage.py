"""Sparse-only timeline identities, named arrays, preparation, and PCA isolation."""

import json
from dataclasses import replace

import numpy as np
import pytest

from src.neural_analysis.population import run_session_timeline as runner
from src.neural_analysis.population import run_session_timeline_batch as batch
from src.neural_analysis.population import session_timeline_io as storage
from src.neural_analysis.population.run_window_variance import run_window_variance
from src.neural_analysis.population.session_timeline import SPARSE_ANALYSIS_VERSION
from src.neural_analysis.population.window_variance import compute_window_variance
from src.neural_analysis.population.timeline_sparse_pca import (
    fit_sparse_timeline_activity,
    sparse_pca_parameters,
)
from src.neural_analysis.population.session_timeline import SessionTimelineResult
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as session_fixture,
)

synthetic_timeline_session = session_fixture


def test_sparse_configuration_is_separate_and_pca_digest_is_unchanged(
    synthetic_timeline_session,
):
    """Only sparse configurations gain algorithm/penalty/library settings."""
    session = synthetic_timeline_session
    ordinary = storage.timeline_configuration(session, {"PFC": "A"}, 0.5)
    explicit = storage.timeline_configuration(
        session, {"PFC": "A"}, 0.5, algorithm="pca"
    )
    assert explicit == ordinary and "algorithm" not in ordinary
    sparse = storage.timeline_configuration(
        session, {"PFC": "A"}, 0.5, algorithm="sparse_pca"
    )
    changed = storage.timeline_configuration(
        session, {"PFC": "A"}, 0.5, algorithm="sparse_pca", sparse_alpha=0.3
    )
    assert len({item["configuration_id"] for item in (ordinary, sparse, changed)}) == 3
    assert sparse["analysis_version"] == SPARSE_ANALYSIS_VERSION
    assert sparse["algorithm"] == "sparse_pca"
    assert sparse["sparse_pca"]["alpha"] == 1
    assert sparse["sparse_pca"]["random_state"] == 0
    assert sparse["component_count_policy"] == "capped"
    assert sparse["requested_components"] == 30
    assert sparse["scikit_learn"]
    for key in ("sources", "selection", "bin_width_s", "window_s", "probe_roles"):
        assert sparse[key] == ordinary[key]
    with pytest.raises(ValueError, match="algorithm"):
        storage.timeline_configuration(session, {}, 0.5, algorithm="unknown")


def test_sparse_prepare_roundtrip_and_reuse_preserve_original_products(
    synthetic_timeline_session, monkeypatch
):
    """Sparse settings and provenance survive load without fake variance fields or overwrites."""
    session = synthetic_timeline_session
    roles = {"PFC": "A", "HPC": "B"}
    ordinary = runner.run_session_timeline(session.metadata_path, roles).result_path
    summary = run_window_variance(ordinary).result_path
    before = [
        (path, path.stat().st_mtime_ns, path.read_bytes())
        for path in (ordinary, summary)
    ]
    first = runner.run_session_timeline(
        session.metadata_path, roles, algorithm="sparse_pca", sparse_alpha=0.3
    )
    result = storage.load_timeline_result(first.result_path)
    assert set(result.regions) == {"PFC", "HPC"}
    assert result.meta["analysis_version"] == SPARSE_ANALYSIS_VERSION
    assert result.meta["random_seed"] == 0
    assert result.meta["axes"]["scores"] == [
        "choice_window",
        "time_bin",
        "session_component",
    ]
    assert "ridge" in result.meta["units"]["scores"]
    assert "SparsePCA" in (first.result_path.parent / "summary.md").read_text()
    with np.load(first.result_path, allow_pickle=True) as archive:
        assert "pfc_center" in archive and "pfc_objective_history" in archive
        assert (
            "pfc_explained_variance" not in archive
            and "pfc_explained_variance_ratio" not in archive
        )
    copy = first.result_path.with_name("roundtrip.npz")
    storage.save_timeline_result(copy, result)
    loaded = storage.load_timeline_result(copy)
    for role, regional in result.regions.items():
        assert regional.scores.shape == (3, 8, 2)
        other = loaded.regions[role]
        for name in (
            "training_mean",
            "training_scale",
            "center",
            "components",
            "objective_history",
        ):
            np.testing.assert_array_equal(
                getattr(other.transform, name), getattr(regional.transform, name)
            )
        np.testing.assert_array_equal(other.scores, regional.scores)
        assert other.transform.parameters == regional.transform.parameters
        assert other.transform.n_iter == regional.transform.n_iter
        assert other.transform.warnings == regional.transform.warnings
    with pytest.raises(FileExistsError):
        storage.save_timeline_result(copy, result)
    with monkeypatch.context() as reuse:
        reuse.setattr(storage, "load_timeline_population", pytest.fail)
        again = runner.run_session_timeline(
            session.metadata_path, roles, algorithm="sparse_pca", sparse_alpha=0.3
        )
        assert again.status == "reused" and again.result_path == first.result_path
        assert (
            runner.run_session_timeline(session.metadata_path, roles).result_path
            == ordinary
        )
    rerun = runner.run_session_timeline(
        session.metadata_path,
        roles,
        algorithm="sparse_pca",
        sparse_alpha=0.3,
        rerun=True,
    )
    assert rerun.result_path != first.result_path
    np.testing.assert_array_equal(
        storage.load_timeline_result(rerun.result_path).regions["PFC"].scores,
        result.regions["PFC"].scores,
    )
    assert (
        runner.find_saved_timeline(session.session_root, first.configuration_id)
        == rerun.result_path
    )
    for path, mtime, contents in before:
        assert path.stat().st_mtime_ns == mtime and path.read_bytes() == contents


def test_sparse_load_rejects_corrupted_shapes_and_versions(synthetic_timeline_session):
    """Sparse loading has its own contract; ordinary PCA validation is not relaxed."""
    prepared = runner.run_session_timeline(
        synthetic_timeline_session.metadata_path, {"PFC": "A"}, algorithm="sparse_pca"
    )
    with np.load(prepared.result_path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for corruption in ("center", "version", "settings", "scores"):
        changed = dict(arrays)
        meta = dict(changed["meta"].item())
        if corruption == "center":
            changed["pfc_center"] = np.zeros(10)
        elif corruption == "scores":
            changed["pfc_scores"] = np.zeros((1, 8, 2))
        elif corruption == "settings":
            meta["sparse_pca"] = dict(meta["sparse_pca"], ridge_alpha=-1)
        else:
            meta["analysis_version"] = "unknown"
        changed["meta"] = meta
        destination = prepared.result_path.with_name(f"invalid_{corruption}.npz")
        np.savez(destination, **changed)
        with pytest.raises(ValueError):
            storage.load_timeline_result(destination)


def test_legacy_full_sparse_archive_remains_readable(
    synthetic_timeline_session, tmp_path
):
    """A 35-component legacy envelope is distinct from the new cap and is not rewritten."""
    session = synthetic_timeline_session
    origin, choices, events = storage.load_timeline_behavior(session)
    counts = np.random.default_rng(105).poisson(3, (3, 40, 35)).astype(float)
    regional = fit_sparse_timeline_activity(
        counts,
        tuple(f"A:{i}" for i in range(35)),
        np.arange(3),
        "PFC",
        "A",
        0.1,
    )
    fitted = replace(
        regional.transform,
        components=np.eye(35),
        parameters=dict(regional.transform.parameters, n_components=35),
    )
    coefficients = (
        (counts - fitted.training_mean) / fitted.training_scale - fitted.center
    ) / 1.01
    regional = replace(regional, transform=fitted, scores=coefficients)
    meta = {
        "analysis_version": SPARSE_ANALYSIS_VERSION,
        "algorithm": "sparse_pca",
        "component_count_policy": "full",
        "sparse_pca": sparse_pca_parameters(1),
        "bin_width_s": 0.1,
    }
    result = SessionTimelineResult(
        session.session_id, origin, choices, events, {"PFC": regional}, {}, meta
    )
    path = tmp_path / "legacy_full.npz"
    storage.save_timeline_result(path, result)
    before = path.read_bytes()
    loaded = storage.load_timeline_result(path)
    assert loaded.regions["PFC"].scores.shape == (3, 40, 35)
    assert loaded.meta["component_count_policy"] == "full"
    assert path.read_bytes() == before


def test_sparse_stage2_is_rejected_before_output(synthetic_timeline_session):
    """Neither UI bypasses nor direct Python calls can assign PCA variance to sparse scores."""
    report = runner.run_session_timeline(
        synthetic_timeline_session.metadata_path, {"PFC": "A"}, algorithm="sparse_pca"
    )
    result = storage.load_timeline_result(report.result_path)
    with pytest.raises(ValueError, match="ordinary PCA"):
        compute_window_variance(result.regions["PFC"])
    for dry_run in (False, True):
        with pytest.raises(ValueError, match="ordinary PCA"):
            run_window_variance(report.result_path, dry_run=dry_run)
    assert not list(report.result_path.parent.glob("window_variance_*"))


def test_sparse_cli_dry_run_and_batch_execution(
    synthetic_timeline_session, tmp_path, monkeypatch
):
    """The existing runners expose sparse preparation and reject irrelevant arguments."""
    session = synthetic_timeline_session
    common = ["--session-metadata", str(session.metadata_path), "--pfc-probe", "A"]
    with monkeypatch.context() as dry:
        dry.setattr(storage, "load_timeline_population", pytest.fail)
        assert (
            runner.main(
                common
                + ["--algorithm", "sparse_pca", "--sparse-alpha", ".3", "--dry-run"]
            )
            == 0
        )
    assert not (session.session_root / "analysis_runs").exists()
    for options in (
        ["--sparse-alpha", ".3"],
        ["--algorithm", "sparse_pca", "--window-summary"],
        ["--algorithm", "sparse_pca", "--speed"],
    ):
        assert runner.main(common + options) == 1
    listing = tmp_path / "sparse_sessions.txt"
    listing.write_text(str(session.metadata_path) + "\n")
    assert (
        batch.main(
            [
                "--session-list",
                str(listing),
                "--pfc-probe",
                "A",
                "--algorithm",
                "sparse_pca",
                "--sparse-alpha",
                ".3",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    results = list((session.session_root / "analysis_runs").glob("*/results.npz"))
    assert len(results) == 1
    config = json.loads((results[0].parent / "config.json").read_text())
    assert config["sparse_pca"]["alpha"] == 0.3
    assert (
        storage.load_timeline_result(results[0])
        .regions["PFC"]
        .transform.parameters["n_jobs"]
        == 1
    )
