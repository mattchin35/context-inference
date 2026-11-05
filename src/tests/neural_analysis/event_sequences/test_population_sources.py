"""Explicit manual quality selection and exact regional PCA reuse contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population import run_session_timeline as runner
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as storage
from src.neural_analysis.session_metadata import (
    ResolvedSession, load_session_metadata, resolve_session_metadata,
)
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as session_fixture,
)
from src.tests.neural_analysis.test_timeline_scientific_identity import _legacy_result


synthetic_timeline_session = session_fixture


def _reload(session: ResolvedSession) -> ResolvedSession:
    """Input: resolved session. Return current metadata/path record; no array reads."""
    return resolve_session_metadata(load_session_metadata(session.metadata_path), session.metadata_path)


def _identity(path: Path) -> dict:
    """Input: existing Path. Return path/byte-size/mtime-ns identity, no data arrays."""
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _old_configuration(session: ResolvedSession) -> dict:
    """Input: unchanged fixture session. Return exact pre-feature .5-s identity dict."""
    root = session.session_root
    paths = [root / "trials.csv", root / "events.csv"]
    for probe in ("B", "A"):
        paths.extend(root / probe / name for name in (
            "aligned.npz", "spike_clusters.npy", "cluster_info.tsv", "channel_quality.csv",
        ))
    config = {
        "analysis_version": "session-timeline-pca-v1",
        "configuration_identity_version": 2,
        "acquisition_families": {"HPC": "open_ephys", "PFC": "open_ephys"},
        "session_id": "synthetic-timeline",
        "probe_roles": {"PFC": "A", "HPC": "B"},
        "bin_width_s": 0.5, "window_s": [-2.0, 2.0],
        "quality_rules": "Phy group good/mua; good inside-brain channels; metadata unit_channels",
        "selection": {
            "HPC": {"unit_ids": ["B:7", "B:8"], "channels": [0, 1, 4]},
            "PFC": {"unit_ids": ["A:7", "A:8"], "channels": [0, 1, 4]},
        },
        "sources": [_identity(path) for path in paths],
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["configuration_id"] = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    return config


def _add_missing_labels(session: ResolvedSession) -> None:
    """Input: fixture session. Return None; add (5,C) metadata rows with unitless labels.

    These rows have no assigned spikes and are excluded by strict policies.
    Existing clusters 9/10/11 test bad channels, outside-brain channels and a
    manual noise label conflicting with KSLabel=good, respectively.
    """
    path = session.session_root / "A/cluster_info.tsv"
    rows = pd.read_csv(path, sep="\t")
    rows["group"] = rows["group"].astype(object)
    rows.loc[rows.cluster_id == 7, "group"] = " GOOD "
    rows.loc[rows.cluster_id == 8, "group"] = " MuA "
    extra = pd.DataFrame({
        "cluster_id": [12, 13, 14, 15, 16], "ch": [0, 1, 0, 1, 0],
        "group": ["", None, "None", "nan", " NaN "], "KSLabel": ["good"] * 5,
    })
    pd.concat([rows, extra], ignore_index=True).to_csv(path, sep="\t", index=False)


def _forbid_neural_loading(*args: object, **kwargs: object) -> None:
    """Input: arbitrary loader/fitter arguments. Raise if raw neural work is attempted."""
    raise AssertionError("Regional candidate lookup must not load raw spikes, bin, or fit")


def _forbid_raw_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Input: patch fixture. Return None; prohibit population/spike loading and PCA work."""
    monkeypatch.setattr(storage, "load_timeline_population", _forbid_neural_loading)
    monkeypatch.setattr(storage.loading, "load_aligned_spikes", _forbid_neural_loading)
    monkeypatch.setattr(storage.loading, "load_sorter_metadata", _forbid_neural_loading)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbid_neural_loading)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbid_neural_loading)


@pytest.fixture
def saved_two_region_timeline(synthetic_timeline_session: ResolvedSession) -> tuple[ResolvedSession, Path]:
    """Input: fixture session. Return session and default (W=3,B=8,K=2) PCA archive Path."""
    session = synthetic_timeline_session
    report = runner.run_session_timeline(session.metadata_path, {"PFC": "A", "HPC": "B"})
    return session, report.result_path


def test_default_configuration_and_explicit_defaults_preserve_exact_old_digest(
    synthetic_timeline_session: ResolvedSession,
) -> None:
    """Input: fixture session. Return None; compare entire old dict and scientific digest."""
    session = synthetic_timeline_session
    expected = _old_configuration(session)
    assert storage.timeline_configuration(session, {"PFC": "A", "HPC": "B"}, 0.5) == expected
    assert storage.timeline_configuration(
        session, {"PFC": "A", "HPC": "B"}, 0.5,
        population_quality="good_mua", missing_label_policy="mua",
    ) == expected


def test_quality_policies_distinguish_scientific_identities_even_when_units_match(
    synthetic_timeline_session: ResolvedSession,
) -> None:
    """Input: session with complete labels. Return None; strict policy has its own digest."""
    session = synthetic_timeline_session
    legacy = storage.timeline_configuration(session, {"PFC": "A"}, 0.5)
    strict = storage.timeline_configuration(
        session, {"PFC": "A"}, 0.5, missing_label_policy="exclude",
    )
    good = storage.timeline_configuration(
        session, {"PFC": "A"}, 0.5, population_quality="good", missing_label_policy="exclude",
    )
    assert strict["selection"] == legacy["selection"]
    assert len({config["configuration_id"] for config in (legacy, strict, good)}) == 3
    assert strict["population_quality"] == "good_mua"
    assert strict["missing_label_policy"] == "exclude"
    assert good["selection"]["PFC"]["unit_ids"] == ["A:7"]


@pytest.mark.parametrize("quality,expected", [("good_mua", [7, 8]), ("good", [7])])
def test_public_population_selection_uses_explicit_manual_labels_and_channel_quality(
    synthetic_timeline_session: ResolvedSession, quality: str, expected: list[int],
) -> None:
    """Input: session/mode/(U,) expected IDs. Return None; inspect metadata and raw loader IDs."""
    session = synthetic_timeline_session
    _add_missing_labels(session)
    source, channels, rows = storage.select_timeline_population(
        session, "A", population_quality=quality, missing_label_policy="exclude",
    )
    assert source.probe_id == "A"
    np.testing.assert_array_equal(channels, [0, 1, 4])
    assert rows.cluster_id.tolist() == expected
    assert rows.quality_label.tolist() == (["good", "mua"] if quality == "good_mua" else ["good"])
    population = storage.load_timeline_population(
        session, "A", population_quality=quality, missing_label_policy="exclude",
    )
    assert population.unit_ids == tuple(f"A:{unit}" for unit in expected)


def test_default_missing_labels_still_fall_back_to_mua(synthetic_timeline_session: ResolvedSession) -> None:
    """Input: fixture session. Return None; historical missing-to-MUA metadata policy survives."""
    session = synthetic_timeline_session
    _add_missing_labels(session)
    _, _, rows = storage.select_timeline_population(session, "A")
    assert rows.cluster_id.tolist() == [7, 8, 12, 13, 14, 15, 16]


def test_strict_selection_intersects_metadata_channel_restrictions(
    synthetic_timeline_session: ResolvedSession,
) -> None:
    """Input: fixture session. Return None; (C,) restriction intersects good inside-brain channels."""
    session = synthetic_timeline_session
    payload = json.loads(session.metadata_path.read_text())
    payload["probes"]["A"]["unit_channels"] = [1, 2, 3]
    session.metadata_path.write_text(json.dumps(payload))
    _, channels, rows = storage.select_timeline_population(
        _reload(session), "A", missing_label_policy="exclude",
    )
    np.testing.assert_array_equal(channels, [1])
    assert rows.cluster_id.tolist() == [8]


@pytest.mark.parametrize("missing_group", [False, True])
def test_missing_manual_labels_never_substitute_automatic_good_labels(
    synthetic_timeline_session: ResolvedSession, missing_group: bool,
) -> None:
    """Input: session/bool schema case. Return None; strict empty/missing manual inputs are unavailable."""
    session = synthetic_timeline_session
    path = session.session_root / "A/cluster_info.tsv"
    rows = pd.read_csv(path, sep="\t")
    rows["group"] = ""
    rows["KSLabel"] = "good"
    if missing_group:
        rows = rows.drop(columns="group")
    rows.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError):
        storage.load_timeline_population(session, "A", missing_label_policy="exclude")
    config = storage.timeline_configuration(session, {"PFC": "A"}, 0.5, missing_label_policy="exclude")
    assert "unavailable" in config["selection"]["PFC"]


@pytest.mark.parametrize("function_name", ["timeline_configuration", "load_timeline_population", "select_timeline_population", "run_session_timeline"])
@pytest.mark.parametrize("keyword,value", [("population_quality", "automatic"), ("missing_label_policy", "good")])
def test_unknown_quality_modes_raise_input_errors(
    synthetic_timeline_session: ResolvedSession, function_name: str, keyword: str, value: str,
) -> None:
    """Input: session/API name/invalid unitless policy. Return None; reject unsupported scientific settings."""
    session = synthetic_timeline_session
    function = getattr(runner if function_name == "run_session_timeline" else storage, function_name)
    arguments = {
        "timeline_configuration": (session, {"PFC": "A"}, 0.5),
        "load_timeline_population": (session, "A"),
        "select_timeline_population": (session, "A"),
        "run_session_timeline": (session.metadata_path, {"PFC": "A"}),
    }[function_name]
    with pytest.raises(ValueError, match="population_quality|missing_label_policy"):
        function(*arguments, **{keyword: value})


@pytest.mark.parametrize("quality,expected", [("good_mua", ("A:7", "A:8")), ("good", ("A:7",))])
def test_actual_runner_fits_declared_strict_population_and_preserves_full_choice_pool(
    synthetic_timeline_session: ResolvedSession, quality: str, expected: tuple[str, ...],
) -> None:
    """Input: session/mode/(U,) IDs. Return None; verify actual count scaling and (W=3,B=8,K) PCA."""
    session = synthetic_timeline_session
    _add_missing_labels(session)
    report = runner.run_session_timeline(
        session.metadata_path, {"PFC": "A"}, population_quality=quality, missing_label_policy="exclude",
    )
    result = storage.load_timeline_result(report.result_path)
    regional = result.regions["PFC"]
    assert regional.transform.input_unit_ids == expected
    assert regional.transform.retained_unit_ids == expected
    assert result.meta["population_quality"] == quality
    assert result.meta["missing_label_policy"] == "exclude"
    assert result.meta["selection"]["PFC"]["unit_ids"] == list(expected)
    assert regional.transform.n_training_observations == 24
    np.testing.assert_array_equal(regional.choice_positions, [0, 1, 2])
    assert result.choices.reward.tolist() == [1, 0, 0]
    population = storage.load_timeline_population(
        session, "A", population_quality=quality, missing_label_policy="exclude",
    )
    counts, _ = timeline.bin_choice_counts(
        population.spike_group, population.cluster_ids, result.choices.choice_time.to_numpy(), 0.5,
    )
    pooled = counts.reshape(-1, len(expected))
    np.testing.assert_allclose(regional.transform.training_mean, pooled.mean(axis=0))
    np.testing.assert_allclose(regional.transform.training_scale, pooled.std(axis=0, ddof=0))
    assert regional.scores.shape == (3, 8, len(expected))


def test_strict_population_identity_distinguishes_constant_inputs_from_retained_pca_units(
    synthetic_timeline_session: ResolvedSession,
) -> None:
    """Input: session. Return None; silent good unit stays selected but is omitted by pooled PCA."""
    session = synthetic_timeline_session
    path = session.session_root / "A/aligned.npz"
    assignments = np.load(path.parent / "spike_clusters.npy")
    with np.load(path) as archive:
        times = archive["spike_utc_unix"].copy()
        bounds = archive["irig_utc_unix"]
    times[assignments == 7] = bounds[0] + 0.001
    order = np.argsort(times)
    np.save(path.parent / "spike_clusters.npy", assignments[order])
    np.savez(path, spike_utc_unix=times[order], irig_utc_unix=bounds)
    report = runner.run_session_timeline(
        session.metadata_path, {"PFC": "A", "HPC": "B"}, missing_label_policy="exclude",
    )
    result = storage.load_timeline_result(report.result_path)
    assert result.regions["PFC"].transform.input_unit_ids == ("A:7", "A:8")
    assert result.regions["PFC"].transform.omitted_unit_ids == ("A:7",)
    assert result.regions["PFC"].transform.retained_unit_ids == ("A:8",)
    good = runner.run_session_timeline(
        session.metadata_path, {"PFC": "A", "HPC": "B"},
        population_quality="good", missing_label_policy="exclude",
    )
    only_good = storage.load_timeline_result(good.result_path)
    assert "PFC" not in only_good.regions and "PFC" in only_good.unavailable
    assert "HPC" in only_good.regions


def test_strict_rerun_reuses_its_saved_configuration_without_loading_spikes(
    synthetic_timeline_session: ResolvedSession, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input: session/patch fixture. Return None; strict same-policy reuse avoids raw work."""
    session = synthetic_timeline_session
    first = runner.run_session_timeline(
        session.metadata_path, {"PFC": "A"}, population_quality="good", missing_label_policy="exclude",
    )
    _forbid_raw_work(monkeypatch)
    again = runner.run_session_timeline(
        session.metadata_path, {"PFC": "A"}, population_quality="good", missing_label_policy="exclude",
    )
    assert again.status == "reused" and again.result_path == first.result_path


def test_regional_reuse_accepts_exact_population_from_two_region_default_archive(
    saved_two_region_timeline: tuple[ResolvedSession, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input: session/default archive/patch fixture. Return None; strict exact IDs reuse existing basis."""
    session, path = saved_two_region_timeline
    before = path.read_bytes()
    _forbid_raw_work(monkeypatch)
    assert storage.find_compatible_regional_timeline(session, "PFC", "A", 0.5) == path
    assert storage.find_compatible_regional_timeline(session, "HPC", "B", 0.5) == path
    assert path.read_bytes() == before


def test_regional_reuse_accepts_its_exact_good_only_basis(synthetic_timeline_session: ResolvedSession) -> None:
    """Input: session. Return None; good-only lookup requires its separately fitted (3,8,1) basis."""
    session = synthetic_timeline_session
    report = runner.run_session_timeline(
        session.metadata_path, {"PFC": "A"}, population_quality="good", missing_label_policy="exclude",
    )
    assert storage.find_compatible_regional_timeline(session, "PFC", "A", 0.5, population_quality="good") == report.result_path


@pytest.mark.parametrize("change", ["other_source", "other_channels"])
def test_unrelated_probe_changes_do_not_invalidate_requested_region(
    saved_two_region_timeline: tuple[ResolvedSession, Path], change: str,
) -> None:
    """Input: session/archive/change name. Return None; requested PFC sources and full pool remain identical."""
    session, path = saved_two_region_timeline
    if change == "other_source":
        source = session.session_root / "B/cluster_info.tsv"
        source.write_text(source.read_text() + "\n")
    else:
        metadata = json.loads(session.metadata_path.read_text())
        metadata["probes"]["B"]["unit_channels"] = [0]
        session.metadata_path.write_text(json.dumps(metadata))
    assert storage.find_compatible_regional_timeline(_reload(session), "PFC", "A", 0.5) == path


@pytest.mark.parametrize("filename", [
    "A/aligned.npz", "A/spike_clusters.npy", "A/cluster_info.tsv", "A/channel_quality.csv", "trials.csv",
])
def test_changed_requested_region_sources_reject_saved_basis(
    saved_two_region_timeline: tuple[ResolvedSession, Path], filename: str,
) -> None:
    """Input: session/archive/source filename. Return None; source identity changes require new PCA."""
    session, _ = saved_two_region_timeline
    source = session.session_root / filename
    source.write_bytes(source.read_bytes() + b"\n")
    assert storage.find_compatible_regional_timeline(session, "PFC", "A", 0.5) is None


@pytest.mark.parametrize("change", ["bin", "probe", "channels", "acquisition", "good_only"])
def test_requested_scientific_configuration_mismatch_rejects_candidate(
    saved_two_region_timeline: tuple[ResolvedSession, Path], change: str,
) -> None:
    """Input: session/archive/change name. Return None; no PC relabeling/filtering across configurations."""
    session, _ = saved_two_region_timeline
    width, probe, mode = 0.5, "A", "good_mua"
    if change == "bin":
        width = 0.1
    elif change == "probe":
        probe = "B"
    elif change == "good_only":
        mode = "good"
    else:
        metadata = json.loads(session.metadata_path.read_text())
        if change == "channels":
            # Same actual units, different allowed channels: still a different request.
            metadata["probes"]["A"]["unit_channels"] = [0, 1]
        else:
            metadata["acquisition"] = "spikeglx"
        session.metadata_path.write_text(json.dumps(metadata))
        session = _reload(session)
    assert storage.find_compatible_regional_timeline(session, "PFC", probe, width, population_quality=mode) is None


@pytest.mark.parametrize("corruption", [
    "missing_scores", "nonfinite_scores", "trial_id", "row_position", "choice_time", "action",
    "truncated_pool", "invalid_positions", "bin_edges", "truncated_basis", "unit_order",
    "zero_scale", "nonfinite_mean", "nonfinite_components", "training_count", "score_units",
])
def test_regional_lookup_validates_saved_pool_bins_full_basis_and_scaling(
    saved_two_region_timeline: tuple[ResolvedSession, Path], corruption: str,
) -> None:
    """Input: session/archive/corruption name. Return None; malformed scientific arrays are rejected.

    Shared choices are (3,C), scores (3,8,2), components (2,2); absolute times
    are UTC seconds. Corrupt only the project-owned synthetic archive.
    """
    session, path = saved_two_region_timeline
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    meta = arrays["meta"].item()
    if corruption == "missing_scores":
        del arrays["pfc_scores"]
    elif corruption == "nonfinite_scores":
        arrays["pfc_scores"][0, 0, 0] = np.nan
    elif corruption in {"trial_id", "row_position", "choice_time", "action"}:
        arrays["choices"][corruption][0] += 1
    elif corruption == "truncated_pool":
        arrays["pfc_choice_positions"] = arrays["pfc_choice_positions"][1:]
        arrays["pfc_scores"] = arrays["pfc_scores"][1:]
        arrays["pfc_window_total_variance"] = arrays["pfc_window_total_variance"][1:]
        meta["regions"]["PFC"]["n_training_observations"] = 16
    elif corruption == "invalid_positions":
        arrays["pfc_choice_positions"][0] = 99
    elif corruption == "bin_edges":
        arrays["pfc_bin_edges_s"][1] += 0.1
    elif corruption == "truncated_basis":
        arrays["pfc_scores"] = arrays["pfc_scores"][:, :, :1]
        arrays["pfc_components"] = arrays["pfc_components"][:1]
        arrays["pfc_explained_variance"] = arrays["pfc_explained_variance"][:1]
        arrays["pfc_explained_variance_ratio"] = arrays["pfc_explained_variance_ratio"][:1]
    elif corruption == "unit_order":
        meta["regions"]["PFC"]["input_unit_ids"] = ("A:8", "A:7")
    elif corruption == "zero_scale":
        arrays["pfc_training_scale"][:] = 0
    elif corruption == "nonfinite_mean":
        arrays["pfc_training_mean"][0] = np.nan
    elif corruption == "nonfinite_components":
        arrays["pfc_components"][0, 0] = np.nan
    elif corruption == "training_count":
        meta["regions"]["PFC"]["n_training_observations"] = 8
    else:
        meta["units"]["scores"] = "session z-score"
    arrays["meta"] = np.asarray(meta, dtype=object)
    np.savez_compressed(path, **arrays)
    assert storage.find_compatible_regional_timeline(session, "PFC", "A", 0.5) is None


def test_regional_full_pool_is_coverage_specific_not_all_shared_choices(
    synthetic_timeline_session: ResolvedSession,
) -> None:
    """Input: session. Return None; valid (W=2,B=8,K=2) basis retains choices 1/2 only."""
    session = synthetic_timeline_session
    alignment = session.session_root / "A/aligned.npz"
    with np.load(alignment) as archive:
        spikes = archive["spike_utc_unix"]
    np.savez(alignment, spike_utc_unix=spikes, irig_utc_unix=[1_700_000_010.0, 1_700_000_040.0])
    report = runner.run_session_timeline(session.metadata_path, {"PFC": "A", "HPC": "B"})
    result = storage.load_timeline_result(report.result_path)
    np.testing.assert_array_equal(result.regions["PFC"].choice_positions, [1, 2])
    assert storage.find_compatible_regional_timeline(session, "PFC", "A", 0.5) == report.result_path


@pytest.mark.parametrize("identity_case", [
    "missing_nan", "missing_none", "numeric_string", "malformed_different", "malformed_vs_missing",
])
def test_regional_reuse_preserves_missing_and_numeric_trial_id_semantics(
    synthetic_timeline_session: ResolvedSession, monkeypatch: pytest.MonkeyPatch,
    identity_case: str,
) -> None:
    """Input: session/patch fixture/ID case. Return None; exact eligible (3,8,2) PCA retains paired missing IDs.

    Numeric strings compare to numeric identities. Different malformed strings,
    or a malformed string versus a missing sentinel, must not collapse to a
    shared NaN and authorize reuse. Missing IDs still break motif adjacency.
    """
    session = synthetic_timeline_session
    trials = pd.read_csv(session.behavior.trial_table_file).astype(object)
    source_id = {
        "missing_nan": np.nan, "missing_none": "None", "numeric_string": 101,
        "malformed_different": "invalid-source", "malformed_vs_missing": "None",
    }[identity_case]
    trials.iloc[0, trials.columns.get_loc("cur_trial")] = source_id
    trials.to_csv(session.behavior.trial_table_file, index=False)
    report = runner.run_session_timeline(session.metadata_path, {"PFC": "A", "HPC": "B"})
    path = report.result_path
    result = storage.load_timeline_result(path)
    assert result.regions["PFC"].scores.shape == (3, 8, 2)
    if identity_case in {"numeric_string", "malformed_different", "malformed_vs_missing"}:
        with np.load(path, allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        choices = pd.DataFrame.from_records(arrays["choices"])
        choices["trial_id"] = choices["trial_id"].astype(object)
        choices.loc[0, "trial_id"] = "101" if identity_case == "numeric_string" else "invalid-saved"
        arrays["choices"] = choices.to_records(index=False)
        np.savez_compressed(path, **arrays)
    _forbid_raw_work(monkeypatch)
    expected = None if identity_case.startswith("malformed") else path
    assert storage.find_compatible_regional_timeline(session, "PFC", "A", 0.5) == expected


def test_default_fallback_extra_unit_prevents_strict_population_reuse(
    synthetic_timeline_session: ResolvedSession,
) -> None:
    """Input: session. Return None; fallback A:8 is not silently admitted into explicit-label PCA."""
    session = synthetic_timeline_session
    labels = session.session_root / "A/cluster_info.tsv"
    rows = pd.read_csv(labels, sep="\t")
    rows.loc[rows.cluster_id == 8, "group"] = ""
    rows.to_csv(labels, sep="\t", index=False)
    report = runner.run_session_timeline(session.metadata_path, {"PFC": "A"})
    assert storage.load_timeline_result(report.result_path).regions["PFC"].transform.input_unit_ids == ("A:7", "A:8")
    assert storage.find_compatible_regional_timeline(session, "PFC", "A", 0.5) is None


@pytest.mark.parametrize("registered", [False, True])
def test_legacy_exact_population_reuse_requires_existing_identity_registration(
    synthetic_timeline_session: ResolvedSession, registered: bool,
) -> None:
    """Input: session/receipt flag. Return None; legacy .5-s exact basis requires provenance receipt."""
    session = synthetic_timeline_session
    path = _legacy_result(session, 0.5)
    if registered:
        storage.register_timeline_scientific_identity(session, path)
    metadata = json.loads(session.metadata_path.read_text())
    metadata["behavior"]["treadmill"] = "optional-speed.txt"
    session.metadata_path.write_text(json.dumps(metadata))
    expected = path if registered else None
    assert storage.find_compatible_regional_timeline(_reload(session), "PFC", "A", 0.5) == expected


def test_stale_legacy_receipt_cannot_authorize_modified_archive(synthetic_timeline_session: ResolvedSession) -> None:
    """Input: session. Return None; legacy result path/bytes/mtime remain bound by receipt."""
    session = synthetic_timeline_session
    path = _legacy_result(session, 0.5)
    storage.register_timeline_scientific_identity(session, path)
    path.write_bytes(path.read_bytes() + b"changed")
    assert storage.find_compatible_regional_timeline(session, "PFC", "A", 0.5) is None
