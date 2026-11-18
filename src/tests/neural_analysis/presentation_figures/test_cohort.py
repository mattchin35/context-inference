"""Cohort summaries preserve the named roster and exact finite session support."""

import numpy as np
import pandas as pd

from src.neural_analysis.presentation_figures import results


def test_cohort_uses_completed_sessions_and_full_roster_for_medians(tmp_path):
    """Three finite ten-PC curves on a four-session roster have no cohort median."""
    from src.neural_analysis.presentation_figures.cohort import assemble_cohort

    reports = []
    for number in range(4):
        session_id = f"CT026_2026-08-{(3, 4, 10, 28)[number]:02}_120000"
        directory = tmp_path / str(number)
        directory.mkdir()
        meta = {"session_id": session_id, "seed": 3, "parameters": {"alpha": .1}, "units": {},
                "subject_id": "CT026", "unit_ids": {"PFC": ["p"], "HPC": ["h"]}}
        scores = pd.DataFrame({"region": ["PFC"], "width_ms": [500], "target": ["context"],
                               "group": ["overall"], "regime": ["all"], "period": ["full"],
                               "n_pcs": [0], "bin_index": [-1], "balanced_accuracy": [.6 + .1 * number]})
        curves = pd.DataFrame({"direction": ["PFC->HPC"] * 2, "panel": ["A", "D"], "count": [1, 10],
                               "model": ["ridge", "rrr"], "mean_r2": [-.1 + number / 10,
                               .1 if number < 3 else np.nan], "n_draws": [5, 5 if number < 3 else 0]})
        results.save_result(directory / "regional.npz", {"scores": scores}, meta)
        results.save_result(directory / "prediction.npz", {"curves": curves}, meta)
        results.save_result(directory / "replays.npz", {"episodes": pd.DataFrame(
            {"motif": ["ideal_transition"], "status": ["ok" if number in (0, 3) else "unavailable"]})}, meta)
        manifest = results.complete_run(directory, kind="session", identity=str(number), session_ids=[session_id],
                                        archives={s: s + ".npz" for s in ("regional", "prediction", "replays")})
        reports.append({"session_id": session_id, "manifest": str(manifest), "status": "computed"})
    summary = assemble_cohort(reports)
    assert summary["roster"].session_id.tolist() == [row["session_id"] for row in reports]
    medians = summary["prediction_medians"].set_index("panel")
    assert medians.at["D", "n_sessions"] == 3
    assert medians.at["D", "roster_size"] == 4
    assert np.isnan(medians.at["D", "mean_r2"])
    np.testing.assert_allclose(medians.at["A", "mean_r2"], .05)
    assert summary["curves"].session_id.nunique() == 4
    np.testing.assert_allclose(summary["regional_medians"].balanced_accuracy, [.75])
    assert summary["presentation_examples"].iloc[0].session_id == reports[3]["session_id"]
    reports[3].update(status="excluded", manifest=None, reason="insufficient good units")
    partial = assemble_cohort(reports)
    assert len(partial["roster"]) == 4
    assert partial["roster"].iloc[-1].reason == "insufficient good units"
    assert partial["prediction_medians"].mean_r2.isna().all()
    assert partial["presentation_examples"].iloc[0].session_id == reports[0]["session_id"]
