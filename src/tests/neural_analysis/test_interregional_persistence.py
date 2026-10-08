"""Tests for immutable inter-regional result persistence and run identity."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.interregional.configuration import (
    AnalysisWindows,
    FilterConfig,
    InterregionalAnalysisConfig,
    PCAConfig,
    RegionalPopulationConfig,
    ResolvedRegionalPopulation,
    RunOptions,
    TemporalConfig,
    configuration_to_dict,
)
from src.neural_analysis.interregional import persistence, records


def _config(metadata_path: Path, **changes: object) -> InterregionalAnalysisConfig:
    """Build a compact deterministic scientific configuration."""
    base = InterregionalAnalysisConfig(
        session_metadata_path=metadata_path,
        pfc_population=RegionalPopulationConfig(role="PFC", probe_id="pfc"),
        hpc_population=RegionalPopulationConfig(role="HPC", probe_id="hpc"),
        windows=AnalysisWindows(),
        prediction_windows=("before",),
        temporal=TemporalConfig(),
        pca=PCAConfig(pfc_components=1, hpc_components=1),
        filters=FilterConfig(conditions=("all",)),
    )
    return replace(base, **changes)


def _populations() -> tuple[ResolvedRegionalPopulation, ResolvedRegionalPopulation]:
    """Return one explicit unit in each disjoint regional population."""
    return (
        ResolvedRegionalPopulation(
            role="PFC",
            probe_id="pfc",
            channel_source="explicit",
            selected_channels=(0,),
            cluster_ids=(1,),
            unit_ids=("pfc:1",),
        ),
        ResolvedRegionalPopulation(
            role="HPC",
            probe_id="hpc",
            channel_source="explicit",
            selected_channels=(1,),
            cluster_ids=(2,),
            unit_ids=("hpc:2",),
        ),
    )


def _result(config: InterregionalAnalysisConfig) -> records.InterregionalResults:
    """Return a fully valid empty-stage result for persistence tests."""
    return records.InterregionalResults(
        schema_version="1",
        analysis_version="interregional-regression-v1",
        coverage_assumption_version="implicit-complete-v1",
        configuration=configuration_to_dict(config, RunOptions()),
        session_id="session-a",
        resolved_populations=_populations(),
        whole_bin_edges_s=np.linspace(-2.0, 2.0, 41, dtype=np.float64),
        units_and_axes=records.default_units_and_axes(),
        randomness_used=False,
        random_seed=None,
        **records.make_empty_result_tables(),
    )


def _runtime_versions() -> dict[str, str]:
    """Return the exact persisted computation-runtime key set."""
    return {
        "python": "3.12.0",
        "numpy": "2.0.0",
        "pandas": "2.2.0",
        "scipy": "1.14.0",
        "pynapple": "0.11.0",
        "scikit_learn": "1.6.0",
        "statsmodels": "0.15.0",
        "matplotlib": "3.9.0",
    }


def _file_entries(root: Path) -> tuple[dict[str, object], ...]:
    """Return two resolved input identities with location-independent content."""
    return (
        {
            "logical_role": "session_metadata",
            "resolved_path": str((root / "neural_session.json").resolve()),
            "size_bytes": 10,
            "sha256": "a" * 64,
        },
        {
            "logical_role": "trial_table",
            "resolved_path": str((root / "trials.csv").resolve()),
            "size_bytes": 20,
            "sha256": "b" * 64,
        },
    )


def test_run_fingerprint_is_canonical_location_independent_and_scientific() -> None:
    """Only scientific/input/code/runtime identity changes the SHA-256 digest."""
    first = _config(Path("/data/one/neural_session.json"))
    relocated = _config(Path("/relocated/two/neural_session.json"))
    common = dict(
        session_id="session-a",
        resolved_populations=_populations(),
        git_head="1" * 40,
        runtime_versions=_runtime_versions(),
    )
    digest = persistence.run_fingerprint(
        first, files=_file_entries(Path("/data/one")), **common
    )
    same = persistence.run_fingerprint(
        relocated, files=_file_entries(Path("/relocated/two")), **common
    )

    assert digest == same
    assert len(digest) == 64
    changed_config = persistence.run_fingerprint(
        replace(first, temporal=TemporalConfig(bin_size_s=0.05)),
        files=_file_entries(Path("/data/one")),
        **common,
    )
    changed_files = list(_file_entries(Path("/data/one")))
    changed_files[0] = {**changed_files[0], "sha256": "c" * 64}
    changed_content = persistence.run_fingerprint(
        first, files=changed_files, **common
    )
    changed_runtime = persistence.run_fingerprint(
        first,
        files=_file_entries(Path("/data/one")),
        **{**common, "runtime_versions": {**_runtime_versions(), "numpy": "2.1.0"}},
    )
    changed_code = persistence.run_fingerprint(
        first,
        files=_file_entries(Path("/data/one")),
        **{**common, "git_head": "2" * 40},
    )
    assert len({digest, changed_config, changed_content, changed_runtime, changed_code}) == 5


def test_hash_input_files_streams_every_file_and_builds_exact_manifest(tmp_path: Path) -> None:
    """New-run input identity records size, SHA-256, paths, and exact manifest keys."""
    metadata = tmp_path / "neural_session.json"
    trials = tmp_path / "trials.csv"
    metadata.write_bytes(b"metadata")
    trials.write_bytes(b"trial-data")

    files = persistence.hash_input_files(
        {"trial_table": trials, "session_metadata": metadata}
    )
    manifest = persistence.build_input_manifest(
        run_fingerprint="f" * 64,
        session_id="session-a",
        git_head="1" * 40,
        runtime_versions=_runtime_versions(),
        resolved_populations=_populations(),
        files=files,
    )

    assert [entry["logical_role"] for entry in files] == [
        "session_metadata",
        "trial_table",
    ]
    assert all(len(str(entry["sha256"])) == 64 for entry in files)
    assert set(manifest) == {
        "manifest_schema_version",
        "run_fingerprint",
        "session_id",
        "git_head",
        "entrypoint",
        "runtime_versions",
        "resolved_populations",
        "files",
    }
    assert manifest["entrypoint"] == (
        "src.neural_analysis.interregional.run_session.run_single_session"
    )


def test_working_directory_is_hidden_same_parent_and_finalized_atomically(
    tmp_path: Path,
) -> None:
    """Only a validated same-parent incomplete directory may become discoverable."""
    output_root = tmp_path / "session" / "analysis_runs"
    repository_root = tmp_path / "checkout"
    repository_root.mkdir()
    working = persistence.create_working_run_directory(
        output_root,
        repository_root=repository_root,
        timestamp="20261008T010203123456Z",
    )
    assert working.parent == output_root
    assert working.name == ".interregional_regression_20261008T010203123456Z.incomplete"
    assert persistence.discover_finalized_runs(output_root) == ()
    for name in (
        "config.json",
        "input_manifest.json",
        "result.pkl",
        "run.log",
        "summary.md",
        "run_session.py",
        "run_batch.py",
    ):
        (working / name).write_text("x", encoding="utf-8")
    (working / "figures").mkdir()

    final = persistence.finalize_run_directory(working, "a" * 64)

    assert final.name.endswith("_aaaaaaaaaaaa")
    assert not working.exists()
    assert persistence.discover_finalized_runs(output_root) == (final,)
    with pytest.raises(FileExistsError):
        persistence.create_working_run_directory(
            output_root,
            repository_root=repository_root,
            timestamp="20261008T010203123456Z",
        )
    with pytest.raises(ValueError, match="repository"):
        persistence.create_working_run_directory(
            repository_root / "analysis_runs", repository_root=repository_root
        )


def test_result_round_trip_preserves_tables_dtypes_and_rejects_unsafe_load(
    tmp_path: Path,
) -> None:
    """Trusted result loading revalidates every pure-result contract."""
    metadata = tmp_path / "neural_session.json"
    metadata.write_text("{}", encoding="utf-8")
    config = _config(metadata.resolve())
    result = _result(config)
    run_directory = tmp_path / "analysis_runs" / "interregional_regression_x_aaaaaaaaaaaa"
    run_directory.mkdir(parents=True)
    result_path = run_directory / "result.pkl"

    persistence.save_interregional_result(result, result_path, config)
    loaded = persistence.load_interregional_result(
        result_path, config, trusted_run_directory=run_directory
    )

    assert loaded.session_id == result.session_id
    np.testing.assert_array_equal(loaded.whole_bin_edges_s, result.whole_bin_edges_s)
    for name in records.RESULT_TABLE_DTYPES:
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(result, name))
    with pytest.raises(ValueError, match="trusted run directory"):
        persistence.load_interregional_result(
            result_path, config, trusted_run_directory=tmp_path / "other"
        )


def test_failure_record_keeps_incomplete_directory_and_final_validation_is_strict(
    tmp_path: Path,
) -> None:
    """Caught failures remain undiscoverable and incomplete artifacts cannot finalize."""
    working = tmp_path / ".interregional_regression_20261008T010203123456Z.incomplete"
    working.mkdir()
    persistence.record_run_failure(
        working, stage="ols_cv", error=RuntimeError("synthetic failure")
    )

    failure = json.loads((working / "failure.json").read_text(encoding="utf-8"))
    assert failure["last_entered_stage"] == "ols_cv"
    assert failure["exception_class"] == "RuntimeError"
    assert working.exists()
    assert persistence.discover_finalized_runs(tmp_path) == ()
    with pytest.raises(ValueError, match="required artifact"):
        persistence.finalize_run_directory(working, "f" * 64)
