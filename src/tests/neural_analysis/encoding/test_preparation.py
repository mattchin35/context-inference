"""Two-second unit activity, independent optional PCA, and source identity contracts."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from src.neural_analysis.encoding import preparation
from src.neural_analysis.population import session_timeline as timeline


def prepare(config):
    """Input config; return complete trials, baseline behavior and regional activity in UTC seconds."""
    session, trials, behavior = preparation.load_behavior_inputs(config)
    regions, unavailable = preparation.prepare_regional_activity(config,session,trials,behavior)
    return trials,behavior,regions,unavailable


@pytest.mark.parametrize("width,bins",[(500,4),(100,20),(50,40),(20,100)])
def test_two_second_pool_raw_units_and_pc_geometry(encoding_session,width,bins):
    """Input ms width; count axes are (window,bin,unit), independent of optional PC axes."""
    config=replace(encoding_session,bin_width_ms=width)
    trials,behavior,regions,unavailable=prepare(config)
    assert not unavailable
    assert behavior.choices.row_position.tolist()==[1,5,6,7,8,9,10,11]
    for role,region in regions.items():
        windows=8 if role=="PFC" else 6
        assert region.unit_counts.shape==(windows,bins,2)
        assert region.unit_ids==(("A:7","A:8") if role=="PFC" else ("B:7","B:8"))
        assert region.bin_edges_s[0]==-1 and region.bin_edges_s[-1]==1
        assert region.pca.scores.shape[:2]==(windows,bins)
        assert region.pca.transform.n_training_observations==windows*bins
        np.testing.assert_allclose(region.pca.transform.training_mean,region.unit_counts.reshape(-1,2).mean(0))
        rates,ids,units=preparation.response_activity(region,"units")
        assert units=="Hz" and tuple(ids)==region.unit_ids
        np.testing.assert_allclose(rates,region.unit_counts/(width/1000))
        rows=preparation.build_bin_rows(trials,behavior.choices,region.choice_positions,region.bin_edges_s,behavior.time_origin_s,region.window_fold_ids)
        pd.testing.assert_frame_equal(rows,region.bin_rows)
        assert len(rows)==windows*bins and rows.before.sum()==rows.after.sum()==windows*bins//2
        np.testing.assert_allclose(rows.absolute_start_s,rows.choice_time+rows.relative_start_s)
        np.testing.assert_allclose(rows.elapsed_start_s,rows.absolute_start_s-behavior.time_origin_s)
        groups=rows.block_id.to_numpy().reshape(windows,bins)[:,0]
        expected=np.full(windows,-1,dtype=int)
        for fold,(_,test) in enumerate(GroupKFold(5).split(np.zeros((windows,1)),groups=groups)):
            expected[test]=fold
        np.testing.assert_array_equal(region.window_fold_ids,expected)
        np.testing.assert_array_equal(rows.fold_id,np.repeat(expected,bins))


def test_only_two_second_recording_coverage_is_required(encoding_session):
    """A window fully inside +/-1 coverage remains valid when its old +/-2 coverage is incomplete."""
    path=encoding_session.session_root/"B/aligned.npz"
    with np.load(path) as archive:
        values={key:archive[key] for key in archive.files}
    values["irig_utc_unix"]=1_700_000_000+np.array([62.5,120.])
    np.savez(path,**values)
    _,_,regions,_=prepare(encoding_session)
    assert 6 in regions["HPC"].bin_rows.row_position.to_list()


@pytest.mark.parametrize("width",[.5,.1,.05,.02])
def test_shared_timeline_helpers_keep_old_defaults_and_accept_explicit_window(width):
    """Input seconds; unrelated timeline defaults stay four seconds while explicit joint edges span two."""
    old=timeline.choice_bin_edges(width)
    new=timeline.choice_bin_edges(width,window_s=(-1.,1.))
    assert old[0]==-2 and old[-1]==2 and len(old)==2*(len(new)-1)+1
    np.testing.assert_array_equal(timeline.select_complete_choice_windows(np.array([1.,2.,3.]),(0.,4.),window_s=(-1.,1.)),[0,1,2])
    np.testing.assert_array_equal(timeline.select_complete_choice_windows(np.array([1.,2.,3.]),(0.,4.)),[1])


def test_spikes_outside_one_second_cannot_change_counts_or_pca(encoding_session):
    """Extra selected-unit spikes at choice+1.5 seconds cannot influence the revised activity or transform."""
    _,behavior,before,_=prepare(encoding_session)
    path=encoding_session.session_root/"A/aligned.npz"
    assignments=encoding_session.session_root/"A/spike_clusters.npy"
    with np.load(path) as archive:
        spikes=archive["spike_utc_unix"]; irig=archive["irig_utc_unix"]
    extra=np.repeat(behavior.choices.choice_time.to_numpy()+1.5,20)
    np.savez(path,spike_utc_unix=np.concatenate([spikes,extra]),irig_utc_unix=irig)
    np.save(assignments,np.concatenate([np.load(assignments),np.full(len(extra),7)]))
    _,_,after,_=prepare(encoding_session)
    np.testing.assert_array_equal(before["PFC"].unit_counts,after["PFC"].unit_counts)
    np.testing.assert_allclose(before["PFC"].pca.scores,after["PFC"].pca.scores)


def test_silent_within_windows_units_survive_without_any_pca(encoding_session):
    """Units with spikes outside extracted windows remain silent unit responses, with no dummy PCA."""
    path=encoding_session.session_root/"A/aligned.npz"
    with np.load(path) as archive:
        irig=archive["irig_utc_unix"]
    assignments=np.load(encoding_session.session_root/"A/spike_clusters.npy")
    np.savez(path,spike_utc_unix=np.full(len(assignments),1_700_000_000.5),irig_utc_unix=irig)
    trials,behavior,regions,unavailable=prepare(encoding_session)
    assert "PFC" not in unavailable
    region=regions["PFC"]
    assert region.pca is None and region.pca_unavailable_reason
    assert region.unit_ids==("A:7","A:8") and not region.unit_counts.any()
    rates,ids,units=preparation.response_activity(region,"units")
    assert rates.shape==(8,4,2) and not rates.any() and units=="Hz"
    with pytest.raises(ValueError,match="PCA|PC|unavailable"):
        preparation.response_activity(region,"pcs")
    rows=preparation.build_bin_rows(trials,behavior.choices,region.choice_positions,region.bin_edges_s,behavior.time_origin_s,region.window_fold_ids)
    assert len(rows)==32


def test_unknown_joint_outcomes_do_not_change_descriptive_pca_pool(encoding_session):
    """Outcome missingness masks regression later, preserving baseline raw units and descriptive PCA."""
    _,_,before,_=prepare(encoding_session)
    for path in (encoding_session.augmented_trials,encoding_session.session_root/"trials.csv"):
        table=pd.read_csv(path); table.loc[5,"reward"]=np.nan; table.to_csv(path,index=False)
    _,_,after,_=prepare(encoding_session)
    np.testing.assert_array_equal(before["PFC"].unit_counts,after["PFC"].unit_counts)
    np.testing.assert_allclose(before["PFC"].pca.scores,after["PFC"].pca.scores)


def test_manual_alignment_is_reported_per_region(encoding_session):
    """Unsupported manual coverage cannot be inferred from spike extrema or hide the other region."""
    path=encoding_session.session_root/"A/aligned.npz"
    with np.load(path) as archive:
        spikes=archive["spike_utc_unix"]
    np.savez(path,spike_utc_unix=spikes)
    _,_,regions,unavailable=prepare(encoding_session)
    assert set(regions)=={"HPC"} and "IRIG" in unavailable["PFC"]
