"""Saved unit/PC labels, joint predictor timing, real gaps and display-only contracts."""

from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
from src.neural_analysis.encoding import plotting,modeling,preparation
from src.neural_analysis.encoding.predictors import build_joint_predictors
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import timeline_plotting


@pytest.fixture
def plotting_data():
    """Return prep and unit/PC cell factory: counts(4,4,12), original IDs and 112-second gaps; seed20261113.

    Slopes and predictions are deliberately specified visual fixtures, not
    regression estimates. Source times are UTC seconds; rates are Hz.
    """
    trials=pd.DataFrame({"cur_trial":[105,108,118,130],"cur_block":[0,1,2,3],
        "start_time":[0.,114.,190.,234.],"choice_time":[10.,124.,200.,244.],
        "state_int":[0,1,1,0],"action":[0,1,0,1],"reward":[0,1,0,1],"correct":[1,1,0,1],
        "experimenter_reward_given":[0,0,0,0]})
    origin,choices=timeline.prepare_timeline_choices(trials)
    ids=tuple(f"A:{n}" for n in [12,2,7,1,3,4,5,6,8,9,10,11])
    counts=np.random.default_rng(20261113).poisson(3,size=(4,4,12)).astype(float)
    pca=timeline.fit_timeline_activity(counts,ids,np.arange(4),"PFC","A",.5,window_s=(-1,1))
    folds=np.full(4,-1,dtype=int)
    rows=preparation.build_bin_rows(trials,choices,np.arange(4),pca.bin_edges_s,origin,folds)
    region=preparation.PreparedRegion("PFC","A",counts,ids,np.arange(4),pca.bin_edges_s,rows,folds,
        "Fewer than five blocks",pca,np.arange(1,13),"",{})
    events=pd.DataFrame({"Time":[0.,8.,9.,123.,125.],"Event":["enter_right_patch","LED_on","LED_off","LED_on","LED_off"],"elapsed_s":[0.,8.,9.,123.,125.]})
    behavior=timeline.SessionTimelineResult("visual-joint",origin,choices,events,{"PFC":pca},{},{})
    prepared={"trials":trials,"behavior":behavior,"regions":{"PFC":region},"meta":{"preparation_id":"visual"}}
    def make_cell(representation):
        """Input units/PCs; return a raw cell with response identities(R,), native coefficients(P,R), and predictions(N,R)."""
        activity,response_ids,units=preparation.response_activity(region,representation)
        design=build_joint_predictors(trials,np.arange(4),region.bin_edges_s,"signed_context")
        names=design["column_names"]
        slopes=np.full((8,12),.1); slopes[names.index("left_reward"),:3]=[5.,-4.,3.]
        standard=slopes.copy(); standard[names.index("left_reward"),:3]=[7.,-6.,8.]
        observed=activity.reshape(-1,12)
        held=observed+.2; held[-1]=np.nan
        return {"meta":{"cell_id":"visual_"+representation,"preparation_id":"visual","model_id":"visual_model",
            "settings":{"region":"PFC","representation":representation,"encoding":"signed_context","estimator":"ridge","penalty_mode":"same_numeric"},
            "penalty_aliases":["same_numeric","matched_context"],"configured_alpha":1.,"feature_penalties":[1.]*8,
            "available":True,"reference_valid":True,"cv_available":False,"cv_reason":"Visual partial CV",
            "bin_width_ms":500,"window_s":[-1,1],"row_count":16,"response_count":12,"predictor_count":8,
            "response_units":units,"trial_count":4,"contrast_names":["left_reward_minus_omission","right_reward_minus_omission"]},
            "response_ids":np.array(response_ids),"column_names":np.array(names),"reference_slopes":slopes,
            "standardized_slopes":standard,"reference_intercept":np.zeros(12),"predictors":design["predictors"],
            "selected_row_indices":np.arange(16),"reference_predictions":observed-.1,"held_out_predictions":held,
            "predictor_supported":np.ones(8,dtype=bool),"contrasts":np.vstack((np.arange(12),-np.arange(12))),
            "contrast_valid":np.array([True,False]),"contrast_reasons":np.array(["","Absent right omission"]),
            "mse":np.ones(12),"r2":np.full(12,np.nan)}
    return prepared,make_cell


def trace(figure,name):
    """Input Plotly figure/name; return the named trace without modifying its units."""
    return next(item for item in figure.data if item.name==name)


@pytest.mark.parametrize("representation",["units","pcs"])
def test_original_response_ids_and_sorted_rows_remain_coupled(plotting_data,representation):
    """Activity and coefficient rows preserve the same nonmonotonic response identities and native scales."""
    prep,make=plotting_data; cell=make(representation)
    figure=plotting.plot_encoding_timeline(prep,cell,ordering="coefficient_strength",sorting_predictor="left_reward",row_count=3,
        selected_response=cell["response_ids"][0],time_range_s=(0,250))
    activity=trace(figure,"Neural activity"); coefficients=trace(figure,"Coefficients")
    np.testing.assert_array_equal(activity.y,[0,1,2]); np.testing.assert_array_equal(coefficients.y,activity.y)
    expected=cell["response_ids"][:3].tolist()
    assert figure.layout.meta["display_response_ids"]==expected
    np.testing.assert_array_equal(np.asarray(coefficients.z)[:,2],[5,-4,3])
    original,_,_=preparation.response_activity(prep["regions"]["PFC"],representation)
    np.testing.assert_allclose(np.asarray(activity.z)[:,:4],original[0,:,:3].T)
    assert np.isnan(np.asarray(activity.z)[:,4]).all()
    np.testing.assert_allclose(np.asarray(activity.x)[4:6],[11,123])
    assert ("variance" in activity.hovertemplate)==(representation=="pcs")
    assert expected[0] in np.asarray(activity.customdata,dtype=object)[0,0].tolist()


def test_coefficient_toggle_changes_order_only_and_unit_plot_needs_no_pca(plotting_data,monkeypatch):
    """Saved unit data displays with no PCA/fitting; raw versus standardized changes only coefficient ordering."""
    prep,make=plotting_data; cell=make("units")
    prep["regions"]["PFC"]=replace(prep["regions"]["PFC"],pca=None,pc_ranks=np.array([],dtype=int),pca_unavailable_reason="Unavailable")
    original=prep["regions"]["PFC"].unit_counts.copy()
    def forbidden(*args,**kwargs):
        """Arbitrary fit arguments must never be used by plotting."""
        pytest.fail("Plotting refit or reloaded sources")
    monkeypatch.setattr(modeling,"evaluate_block_cv",forbidden)
    a=plotting.plot_encoding_timeline(prep,cell,ordering="coefficient_strength",sorting_predictor="left_reward",row_count=3)
    b=plotting.plot_encoding_timeline(prep,cell,coefficient_scale="standardized",ordering="coefficient_strength",sorting_predictor="left_reward",row_count=3)
    assert a.layout.meta["display_response_ids"]==["A:12","A:2","A:7"]
    assert b.layout.meta["display_response_ids"]==["A:7","A:12","A:2"]
    assert trace(a,"Neural activity").zmax==trace(b,"Neural activity").zmax and trace(a,"Neural activity").zmin==0
    np.testing.assert_array_equal(prep["regions"]["PFC"].unit_counts,original)


def test_contrast_ordering_uses_raw_effects_and_unavailable_values_last(plotting_data):
    """Supported raw contrasts sort by absolute magnitude; unsupported contrasts cannot be treated as zero evidence."""
    _,make=plotting_data; cell=make("units")
    order=plotting.order_responses(cell,ordering="coefficient_strength",sorting_predictor="left_reward_minus_omission")
    assert order[0]==11
    order=plotting.order_responses(cell,ordering="coefficient_strength",sorting_predictor="right_reward_minus_omission")
    np.testing.assert_array_equal(order,np.arange(12))


def test_joint_predictors_gate_outcomes_and_all_time_axes_share_pixel_domain(plotting_data):
    """The full joint design is shown at real choice times; the categorical coefficient axis never receives time bounds."""
    prep,make=plotting_data; cell=make("units")
    figure=plotting.plot_encoding_timeline(prep,cell,event_tracks=("LED",),time_range_s=(0,250))
    predictors=trace(figure,"Model predictors")
    np.testing.assert_array_equal(np.asarray(predictors.z)[:2,:4],cell["predictors"][:4,:2].T)
    np.testing.assert_array_equal(np.asarray(predictors.z)[2:,:2],np.zeros((6,2)))
    meta=figure.layout.meta
    coefficient=figure.layout[meta["coefficient_axis"]]
    assert coefficient.type=="category" and coefficient.fixedrange and meta["coefficient_axis"] not in meta["time_axes"]
    domains=[tuple(figure.layout[name].domain) for name in meta["time_axes"]]
    assert all(domain==domains[0] for domain in domains)
    assert all(tuple(figure.layout[name].range)==(0,250) for name in meta["time_axes"])
    names=[item.name for item in figure.data if item.showlegend is not False and item.name]
    assert len(names)==len(set(names))
    assert trace(figure,"Coefficients").colorbar.orientation=="h"
    assert trace(figure,"Coefficients").colorbar.title.side=="top"


def test_trace_scope_is_explicit_without_fallback(plotting_data):
    """Missing held-out predictions remain missing for the selected original response, with explicit annotation."""
    prep,make=plotting_data; cell=make("units")
    cell["held_out_predictions"][:]=np.nan
    held=plotting.plot_encoding_timeline(prep,cell,selected_response="A:7",prediction_scope="held_out",time_range_s=(0,250))
    reference=plotting.plot_encoding_timeline(prep,cell,selected_response="A:7",prediction_scope="reference",time_range_s=(0,250))
    assert np.isnan(np.asarray(trace(held,"Held-out prediction A:7").y)).all()
    assert np.isfinite(np.asarray(trace(reference,"Reference prediction A:7").y)).any()
    assert any("unavailable" in str(item.text).lower() for item in held.layout.annotations)
    assert held.layout.meta["selected_response"]=="A:7"


@pytest.mark.parametrize("representation",["units","pcs"])
def test_pngs_are_opaque_labeled_and_preserve_scope_without_overwrite(plotting_data,tmp_path,representation):
    """Light-mode PNG artifacts label original identity, response units and exact +/-1 scope; existing files survive."""
    prep,make=plotting_data; cell=make(representation)
    selected=cell["response_ids"][0]
    paths=plotting.save_inspection_figures(tmp_path/representation,prep,cell,selected_response=selected,prediction_scope="reference",time_range_s=(0,140))
    assert len(paths)==2
    for path in paths:
        with Image.open(path) as picture:
            assert picture.width>=1000 and np.asarray(picture.convert("RGBA"))[:,:,3].min()==255
            caption=picture.info["Description"]
            assert selected in caption and "[-1,+1)" in caption and "reference" in caption.lower()
            assert cell["meta"]["response_units"] in caption
    with pytest.raises(FileExistsError):
        plotting.save_inspection_figures(tmp_path/representation,prep,cell,selected_response=selected,prediction_scope="reference",time_range_s=(0,140))


def test_shared_window_geometry_requires_no_pca(plotting_data):
    """Pure heatmap geometry uses values(W,B,R), choices, positions and second edges; PC-specific metadata is separate."""
    prep,_=plotting_data; region=prep["regions"]["PFC"]
    data=timeline_plotting.build_window_heatmap_data(region.unit_counts[:,:,:2],prep["behavior"].choices,region.choice_positions,region.bin_edges_s)
    assert data["z"].shape==(2,19) and len(data["x"])==20 and data["customdata"].shape==(2,19,7)
    np.testing.assert_array_equal(data["z"][:,:4],region.unit_counts[0,:,:2].T)
