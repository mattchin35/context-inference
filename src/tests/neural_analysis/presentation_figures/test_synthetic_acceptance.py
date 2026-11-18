"""Full four-session synthetic acceptance through the real presentation runners."""

import json
from pathlib import Path

import numpy as np


def test_synthetic_batch_publishes_every_session_and_partial_ten_pc_support(
    synthetic_presentation_roster, tmp_path,
):
    """Exercise real recording loaders, held-out fits, archives and cohort assembly."""
    from src.neural_analysis.presentation_figures.results import load_result
    from src.neural_analysis.presentation_figures.run_batch import run_batch

    report = run_batch(synthetic_presentation_roster, output_root=tmp_path / "cohort", workers=2, seed=7)
    session_reports = report["sessions"]
    assert [session["status"] for session in session_reports] == ["computed"] * 4, session_reports
    assert len({session["session_id"] for session in session_reports}) == 4

    for position, session in enumerate(session_reports):
        manifest_path = Path(session["manifest"])
        manifest = json.loads(manifest_path.read_text())
        assert set(manifest["archives"]) == {"regional", "replays", "prediction", "movement"}
        fields = {name: load_result(manifest_path.parent / file)[0]
                  for name, file in manifest["archives"].items()}

        regional = fields["regional"]
        assert set(regional["confusion"].region) == {"PFC", "HPC"}
        assert set(regional["recovery"].target) == {"context", "choice", "outcome"}
        choice = regional["scores"].query("target == 'choice' and n_pcs == 0")
        assert choice.balanced_accuracy.max() > 0.65

        replays = fields["replays"]
        assert set(replays["episodes"].motif) == {"new_side_omission", "ideal_transition", "four_outcome"}
        assert replays["episodes"].status.eq("available").all(), replays["episodes"]
        assert set(replays["tracks"].width_ms) == {100, 500}
        assert "first_new_side_delivery" in set(replays["events"].event)

        prediction = fields["prediction"]
        assert prediction["alpha"] == 0.1
        assert prediction["seed"] == 7
        assert prediction["curves"].query("panel == 'A'").n_draws.eq(5).all()
        ten_pc = prediction["curves"].query("panel == 'D' and model == 'rrr'")
        assert ten_pc.n_draws.eq(5 if position < 3 else 0).all()

        movement = fields["movement"]
        assert movement["n_complete_speed_bins"] == movement["n_paired_bins"] > 0
        assert movement["block_scores"].query("metric == 'pfc_hpc'").correlation.median() > 0.5
        assert movement["replay_tracks"].trial_id.nunique() == 4, movement["replay_reason"]
        assert movement["replay_speed_increase_mm_s"] > 0

    cohort_manifest = Path(report["manifest"])
    cohort, _ = load_result(cohort_manifest.parent / "cohort.npz")
    assert cohort["roster"].session_id.to_list() == [s["session_id"] for s in session_reports]
    ten_pc = cohort["prediction_medians"].query("panel == 'D'")
    assert not ten_pc.empty
    assert ten_pc.roster_size.eq(4).all()
    assert ten_pc.n_sessions.eq(3).all()
    assert np.isnan(ten_pc.mean_r2).all()
    assert cohort["prediction_medians"].query("panel == 'A' and count == 1").mean_r2.notna().all()
    assert len(cohort["presentation_examples"]) == 3
