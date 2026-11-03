"""Preparation contracts: regional coverage, strict PCA pools, bins and fixed axes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.neural_analysis.encoding.configuration import EncodingConfig
from src.neural_analysis.encoding import preparation
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import ResolvedSession


def load_and_prepare(config: EncodingConfig) -> tuple:
    """Input config; return session/trials/behavior/regions/unavailable; UTC seconds.

    Trials have (T,C), behavior choices (W,C), region scores (W_r,B,K). This
    helper uses public entry points only and does not write analysis results.
    """
    session, trials, behavior = preparation.load_behavior_inputs(config)
    regions, unavailable = preparation.prepare_regional_activity(config, session, trials, behavior)
    return session, trials, behavior, regions, unavailable


def save_candidate(
    config: EncodingConfig, session: ResolvedSession, behavior: timeline.SessionTimelineResult,
    regional: timeline.RegionalTimelinePCA, name: str = "candidate.npz",
) -> Path:
    """Save one trusted synthetic candidate, returning new NPZ Path; UTC times.

    Inputs retain the behavior (W,C) table and regional scores (W_r,B,K), native
    unit scaling and original axes. Config/session define source identity.
    """
    meta = timeline_io.timeline_configuration(session, dict(config.probe_roles), config.bin_width_ms / 1000)
    result = replace(behavior, regions={"PFC": regional}, meta=meta)
    path = session.session_root / "analysis_runs" / name
    return timeline_io.save_timeline_result(path, result)


def forbid_binning(*args: object, **kwargs: object) -> None:
    """Reject any spike-binning call (arbitrary inputs); return no arrays or units."""
    raise AssertionError("Compatible saved PCA must not bin spikes or fit again")


def test_behavior_keeps_full_origin_and_strict_baseline_original_ids(encoding_session: EncodingConfig) -> None:
    """Input config; assert full UTC origin and (8,C) strict choices, return None."""
    _, trials, behavior = preparation.load_behavior_inputs(encoding_session)
    assert len(trials) == 12
    assert behavior.time_origin_s == trials.start_time.iloc[0]
    assert behavior.choices.row_position.tolist() == [1, 5, 6, 7, 8, 9, 10, 11]
    assert behavior.choices.trial_id.tolist() == [12, 30, 31, 39, 40, 43, 51, 60]
    np.testing.assert_array_equal(behavior.choices.elapsed_s, [14, 54, 64, 74, 84, 94, 104, 114])
    np.testing.assert_allclose(behavior.events.elapsed_s, behavior.events.Time - behavior.time_origin_s)


@pytest.mark.parametrize("width,bins", [(500, 8), (100, 40), (50, 80), (20, 200)])
def test_regional_bin_rows_keep_zero_boundary_times_and_axes(
    encoding_session: EncodingConfig, width: int, bins: int,
) -> None:
    """Input config/integer ms/bins; assert (W_r*B,C) time metadata, return None."""
    config = replace(encoding_session, bin_width_ms=width)
    _, trials, behavior, regions, unavailable = load_and_prepare(config)
    assert unavailable == {}
    region = regions["PFC"]
    rows = preparation.build_bin_rows(trials, behavior.choices, region.timeline, behavior.time_origin_s)

    assert region.timeline.scores.shape[:2] == (8, bins)
    assert len(rows) == 8 * bins
    pd.testing.assert_frame_equal(rows, region.bin_rows)
    np.testing.assert_array_equal(rows.trial_id.to_numpy().reshape(8, bins)[:, 0], behavior.choices.trial_id)
    np.testing.assert_array_equal(rows.row_position.to_numpy().reshape(8, bins)[:, 0], behavior.choices.row_position)
    np.testing.assert_array_equal(rows.bin_index.to_numpy().reshape(8, bins)[0], np.arange(bins))
    np.testing.assert_array_equal(rows.relative_start_s.to_numpy().reshape(8, bins)[0], region.timeline.bin_edges_s[:-1])
    assert (rows.relative_start_s == 0).sum() == 8
    np.testing.assert_allclose(rows.absolute_start_s, rows.choice_time + rows.relative_start_s)
    np.testing.assert_allclose(rows.absolute_end_s - rows.absolute_start_s, width / 1000, atol=3e-7)
    np.testing.assert_allclose(rows.elapsed_start_s, rows.absolute_start_s - behavior.time_origin_s)
    assert rows.before.sum() == rows.after.sum() == len(rows) // 2
    assert not (rows.before & rows.after).any()


def test_pool_and_unit_selection_are_region_specific_before_pca(encoding_session: EncodingConfig) -> None:
    """Input config; excluded/dark rows never influence regional count scaling; return None."""
    session, trials, behavior, regions, _ = load_and_prepare(encoding_session)
    pfc, hpc = regions["PFC"], regions["HPC"]
    np.testing.assert_array_equal(pfc.timeline.choice_positions, np.arange(8))
    np.testing.assert_array_equal(hpc.timeline.choice_positions, np.arange(2, 8))
    assert pfc.timeline.transform.input_unit_ids == ("A:7", "A:8")
    assert hpc.timeline.transform.input_unit_ids == ("B:7", "B:8")
    population = timeline_io.load_timeline_population(session, "A")
    counts, _ = timeline.bin_choice_counts(population.spike_group, population.cluster_ids,
                                         behavior.choices.choice_time.to_numpy(), 0.5)
    np.testing.assert_allclose(pfc.timeline.transform.training_mean, counts.reshape(-1, 2).mean(axis=0))
    np.testing.assert_allclose(pfc.timeline.transform.training_scale, counts.reshape(-1, 2).std(axis=0))
    assert pfc.timeline.transform.n_training_observations == 64
    assert hpc.timeline.transform.n_training_observations == 48
    # Excluded source row 2 still supplies row 1's post-choice context.
    np.testing.assert_array_equal(pfc.context_values["choice_updated"][0], [0]*4 + [1]*4)
    assert pfc.context_valid["choice_updated"][0].all()
    assert not pfc.context_valid["choice_updated"][-1, 4:].any()
    assert pfc.context_valid["trial_window"][-1].all()
    assert trials.state_int.iloc[4] == 2


def test_regional_group_folds_use_baseline_full_coverage_pool_once(encoding_session: EncodingConfig) -> None:
    """Input config; (W_r,) fold map follows numeric GroupKFold, repeated per bin; return None."""
    _, _, _, regions, _ = load_and_prepare(encoding_session)
    for region in regions.values():
        rows = region.bin_rows
        groups = rows.block_id.to_numpy().reshape(-1, 8)[:, 0]
        expected = np.full(len(groups), -1, dtype=int)
        for fold, (_, test) in enumerate(GroupKFold(n_splits=5, shuffle=False).split(np.zeros((len(groups), 1)), groups=groups)):
            expected[test] = fold
        np.testing.assert_array_equal(region.window_fold_ids, expected)
        np.testing.assert_array_equal(rows.fold_id, np.repeat(expected, 8))
        assert region.cv_unavailable_reason == ""
        for _, block_rows in rows.groupby("block_id"):
            assert block_rows.fold_id.nunique() == 1
    assert len(regions["PFC"].window_fold_ids) == 8
    assert len(regions["HPC"].window_fold_ids) == 6


def test_missing_block_disables_cv_without_discarding_activity(encoding_session: EncodingConfig) -> None:
    """Input config; missing unitless block ID preserves (W,B,K) PCA, return None."""
    path = encoding_session.augmented_trials
    trials = pd.read_csv(path)
    trials.loc[5, "cur_block"] = np.nan
    trials.to_csv(path, index=False)
    _, _, _, regions, unavailable = load_and_prepare(encoding_session)
    assert not unavailable
    assert regions["PFC"].timeline.scores.shape[:2] == (8, 8)
    assert "block" in regions["PFC"].cv_unavailable_reason.lower()
    assert np.all(regions["PFC"].window_fold_ids == -1)
    assert regions["HPC"].cv_unavailable_reason == ""  # Missing row lies outside HPC coverage.


def test_too_few_regional_groups_keeps_reference_activity(encoding_session: EncodingConfig) -> None:
    """Input config; <5 unitless groups disables CV while PCA remains available, return None."""
    trials = pd.read_csv(encoding_session.augmented_trials)
    trials["cur_block"] = np.minimum(np.arange(len(trials)) // 4, 2)
    trials.to_csv(encoding_session.augmented_trials, index=False)
    _, _, _, regions, unavailable = load_and_prepare(encoding_session)
    assert not unavailable
    assert all("five" in region.cv_unavailable_reason.lower() or "5" in region.cv_unavailable_reason for region in regions.values())
    assert all(region.timeline.scores.size > 0 for region in regions.values())


def test_exact_saved_basis_reuses_signs_scaling_and_scores(
    encoding_session: EncodingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input config/patch fixture; trusted exact (W,B,K) PCA is reused without binning; return None."""
    session, _, behavior, regions, _ = load_and_prepare(encoding_session)
    path = save_candidate(encoding_session, session, behavior, regions["PFC"].timeline)
    config = replace(encoding_session, timeline_result=path, probe_roles={"PFC": "A"})
    # Candidate identity must match the requested role selection as well.
    candidate = timeline_io.load_timeline_result(path)
    candidate = replace(candidate, meta=timeline_io.timeline_configuration(session, {"PFC": "A"}, 0.5))
    exact_path = timeline_io.save_timeline_result(path.with_name("exact.npz"), candidate)
    monkeypatch.setattr(timeline, "bin_choice_counts", forbid_binning)
    _, _, _, reused, unavailable = load_and_prepare(replace(config, timeline_result=exact_path))
    assert not unavailable
    assert reused["PFC"].reused
    for field in ("scores", "bin_edges_s"):
        np.testing.assert_array_equal(getattr(reused["PFC"].timeline, field), getattr(regions["PFC"].timeline, field))
    np.testing.assert_array_equal(reused["PFC"].timeline.transform.training_scale, regions["PFC"].timeline.transform.training_scale)
    np.testing.assert_array_equal(reused["PFC"].timeline.transform.components, regions["PFC"].timeline.transform.components)


@pytest.mark.parametrize("mismatch", ["source", "trial_id", "choice_time", "unit_order", "scaling", "truncated", "bin_edges"])
def test_saved_basis_mismatch_rebuilds_instead_of_relabeling_pcs(
    encoding_session: EncodingConfig, mismatch: str,
) -> None:
    """Input config/mismatch name; actual source/pool/axis mismatch rebuilds PCA, return None."""
    session, _, behavior, regions, _ = load_and_prepare(encoding_session)
    original = regions["PFC"].timeline
    regional = original
    choices = behavior.choices.copy(deep=True)
    if mismatch == "trial_id":
        choices.loc[0, "trial_id"] = 999
    elif mismatch == "choice_time":
        choices.loc[0, "choice_time"] += 0.25
    elif mismatch == "unit_order":
        regional = replace(original, transform=replace(original.transform,
            input_unit_ids=original.transform.input_unit_ids[::-1]))
    elif mismatch == "truncated":
        transform = original.transform
        regional = replace(original, scores=original.scores[:, :, :1], transform=replace(transform,
            components=transform.components[:1], explained_variance=transform.explained_variance[:1],
            explained_variance_ratio=transform.explained_variance_ratio[:1]))
    elif mismatch == "bin_edges":
        edges = original.bin_edges_s.copy()
        edges[1] += 0.1
        regional = replace(original, bin_edges_s=edges)
    path = save_candidate(encoding_session, session, replace(behavior, choices=choices), regional)
    if mismatch == "scaling":
        # An invalid saved scale must be detected at the trusted-file boundary.
        with np.load(path, allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["pfc_training_scale"] = np.zeros_like(original.transform.training_scale)
        np.savez_compressed(path, **arrays)
    if mismatch == "source":
        quality = session.session_root / "A/quality.csv"
        quality.write_text(quality.read_text(encoding="ascii") + "\n", encoding="ascii")
    _, _, _, rebuilt, _ = load_and_prepare(replace(encoding_session, timeline_result=path))
    assert not rebuilt["PFC"].reused
    np.testing.assert_allclose(rebuilt["PFC"].timeline.scores, original.scores)
    assert path.exists()  # Mismatch never overwrites the original result.


def test_reuse_remaps_global_choice_positions_by_actual_regional_identity(encoding_session: EncodingConfig) -> None:
    """Input config; excluded dark choice outside coverage permits identical regional PCA reuse; return None."""
    config = replace(encoding_session, probe_roles={"PFC": "B"})
    session, trials, behavior, regions, _ = load_and_prepare(config)
    original = regions["PFC"].timeline
    _, old_choices = timeline.prepare_timeline_choices(trials)
    covered_ids = behavior.choices.iloc[original.choice_positions].trial_id.to_numpy()
    old_positions = np.flatnonzero(old_choices.trial_id.isin(covered_ids))
    candidate_region = replace(original, choice_positions=old_positions)
    path = save_candidate(config, session, replace(behavior, choices=old_choices), candidate_region)
    _, _, _, reused, _ = load_and_prepare(replace(config, timeline_result=path))
    assert reused["PFC"].reused
    np.testing.assert_array_equal(reused["PFC"].timeline.choice_positions, original.choice_positions)
    np.testing.assert_array_equal(reused["PFC"].timeline.scores, original.scores)


def test_model_settings_do_not_change_prepared_basis_or_predictor_rows(encoding_session: EncodingConfig) -> None:
    """Input config; timing/window/encoding/alpha selections preserve descriptive (W,B,K) scores, return None."""
    _, _, _, first, _ = load_and_prepare(encoding_session)
    narrow = replace(encoding_session, alpha=20, windows=("after",), timing_modes=("trial_window",),
                     encodings=("separate_context",), estimators=("ols",))
    _, _, _, second, _ = load_and_prepare(narrow)
    for role in first:
        np.testing.assert_array_equal(first[role].timeline.scores, second[role].timeline.scores)
        np.testing.assert_array_equal(first[role].timeline.transform.components, second[role].timeline.transform.components)
        np.testing.assert_array_equal(first[role].pc_ranks, second[role].pc_ranks)
        pd.testing.assert_frame_equal(first[role].bin_rows, second[role].bin_rows)


def test_numerical_zero_pcs_are_excluded_without_truncating_saved_basis(encoding_session: EncodingConfig) -> None:
    """Input config; duplicate count trains create rank-one full basis and original ranks, return None."""
    path = encoding_session.session_root / "A/aligned.npz"
    assignments = np.load(path.parent / "spike_clusters.npy")
    with np.load(path) as saved:
        spikes = saved["spike_utc_unix"][assignments == 7]
        irig = saved["irig_utc_unix"]
    doubled = np.repeat(spikes, 2)
    np.save(path.parent / "spike_clusters.npy", np.tile([7, 8], len(spikes)))
    np.savez(path, spike_utc_unix=doubled, irig_utc_unix=irig)
    _, _, _, regions, _ = load_and_prepare(encoding_session)
    region = regions["PFC"]
    assert region.timeline.scores.shape == (8, 8, 2)
    np.testing.assert_array_equal(region.pc_ranks, [1])
    eigenvalues = region.timeline.transform.explained_variance
    singular = np.sqrt(np.maximum(eigenvalues, 0) * (64 - 1))
    expected = np.flatnonzero(singular > max(64, 2) * np.finfo(float).eps * singular.max()) + 1
    np.testing.assert_array_equal(region.pc_ranks, expected)


def test_manual_alignment_is_explicitly_unsupported_without_spike_extrema_fallback(encoding_session: EncodingConfig) -> None:
    """Input config; (S,) spikes without IRIG make only that region unavailable, return None."""
    path = encoding_session.session_root / "A/aligned.npz"
    with np.load(path) as saved:
        spikes = saved["spike_utc_unix"]
    np.savez(path, spike_utc_unix=spikes)
    _, _, behavior, regions, unavailable = load_and_prepare(encoding_session)
    assert "PFC" not in regions
    assert "HPC" in regions
    assert "PFC" in unavailable
    reason = unavailable["PFC"].lower()
    assert "irig" in reason and ("manual" in reason or "unsupported" in reason)
    assert len(behavior.choices) == 8


def test_no_complete_regional_windows_does_not_erase_other_region(encoding_session: EncodingConfig) -> None:
    """Input config; narrow UTC bounds make one region unavailable while the other remains, return None."""
    path = encoding_session.session_root / "B/aligned.npz"
    with np.load(path) as saved:
        spikes = saved["spike_utc_unix"]
    np.savez(path, spike_utc_unix=spikes, irig_utc_unix=np.array([100.0, 101.0]))
    _, _, _, regions, unavailable = load_and_prepare(encoding_session)
    assert "PFC" in regions and "HPC" not in regions
    assert "coverage" in unavailable["HPC"].lower() or "window" in unavailable["HPC"].lower()


def test_all_constant_region_is_unavailable_without_zero_fabrication(encoding_session: EncodingConfig) -> None:
    """Input config; no variable count units yields explicit region reason, return None."""
    path = encoding_session.session_root / "A/aligned.npz"
    with np.load(path) as saved:
        irig = saved["irig_utc_unix"]
        count = len(saved["spike_utc_unix"])
    # Preserve sorted (S,) UTC spikes and their unit assignments, outside all fit windows.
    spikes = irig[0] + np.arange(count) * 0.0001
    np.savez(path, spike_utc_unix=spikes, irig_utc_unix=irig)
    _, _, _, regions, unavailable = load_and_prepare(encoding_session)
    assert "PFC" not in regions and "HPC" in regions
    assert "variable" in unavailable["PFC"].lower() or "rank" in unavailable["PFC"].lower()


def test_source_correspondence_is_checked_before_pca(encoding_session: EncodingConfig) -> None:
    """Input config; conflicting unitless state input fails before any spike preparation, return None."""
    table = pd.read_csv(encoding_session.augmented_trials)
    table.loc[1, "state_int"] = 1
    table.to_csv(encoding_session.augmented_trials, index=False)
    with pytest.raises(ValueError, match="state_int"):
        preparation.load_behavior_inputs(encoding_session)
