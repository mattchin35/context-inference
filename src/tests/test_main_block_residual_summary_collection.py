from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.tests.test_main_multisession import load_main_module


def make_residual_summary(session_id: str, mouse: str, date: str) -> pd.DataFrame:
    """Build the four-row residual-model summary emitted by one session."""
    rows = []
    for model_type, alpha in (("lasso", 1.0), ("elastic_net", 0.5)):
        for lambda_choice in ("lambda.min", "lambda.1se"):
            rows.append(
                {
                    "session_id": session_id,
                    "mouse": mouse,
                    "date": date,
                    "model_type": model_type,
                    "lambda_choice": lambda_choice,
                    "lambda_value": 0.1,
                    "alpha": alpha,
                    "n_valid_blocks": 12,
                    "coefficient_intercept": 1.0,
                    "coefficient_prev_n_rewarded": 0.2,
                    "coefficient_block_side_left": 0.3,
                    "coefficient_prev_n_rewarded_block_side_left": 0.4,
                    "residual_mad_raw": 0.5,
                    "residual_mad_scaled": 0.7413,
                    "residual_iqr": 1.0,
                    "residual_rmse": 0.8,
                }
            )
    return pd.DataFrame(rows)


def write_session_residual_summary(processed_path: Path, session_id: str, mouse: str, date: str) -> None:
    """Write one per-session residual-model summary CSV."""
    processed_path.mkdir(parents=True, exist_ok=True)
    make_residual_summary(session_id=session_id, mouse=mouse, date=date).to_csv(
        processed_path / f"{session_id}_block_residual_model_summary.csv",
        index=False,
    )


def make_saved_session(tmp_path: Path, mouse: str, date: str, timestamp: str) -> SimpleNamespace:
    """Create a lightweight SavedSessionAnalysis-like object for collection tests."""
    session_id = f"{mouse}_{date}_{timestamp}"
    processed_path = tmp_path / session_id / "processed"
    write_session_residual_summary(processed_path, session_id, mouse, date)
    return SimpleNamespace(
        session=SimpleNamespace(
            mouse=mouse,
            date=date,
            sess_id_full=session_id,
            processed_data_path=processed_path,
        )
    )


def test_collect_mouse_block_residual_model_summaries_writes_mouse_csv(tmp_path):
    """Mouse-level collection should combine one four-row summary per session."""
    main_module = load_main_module()
    saved_sessions = [
        make_saved_session(tmp_path, "CT024", "2026-06-11", "142023"),
        make_saved_session(tmp_path, "CT024", "2026-06-09", "143852"),
    ]

    saved_path = main_module.collect_mouse_block_residual_model_summaries(
        saved_sessions=saved_sessions,
        output_path=tmp_path / "cross_session_analysis",
        mouse="CT024",
    )

    assert saved_path == tmp_path / "cross_session_analysis" / "CT024_block_residual_model_summary.csv"
    collected = pd.read_csv(saved_path, na_filter=False)
    assert collected.shape[0] == 8
    assert collected["session_id"].tolist()[:4] == ["CT024_2026-06-09_143852"] * 4
    assert collected["training_day"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_collect_mouse_block_residual_model_summaries_fails_for_missing_session_summary(tmp_path):
    """Missing per-session residual summary files should fail loudly."""
    main_module = load_main_module()
    saved_session = SimpleNamespace(
        session=SimpleNamespace(
            mouse="CT024",
            date="2026-06-09",
            sess_id_full="CT024_2026-06-09_143852",
            processed_data_path=tmp_path / "processed",
        )
    )

    with pytest.raises(FileNotFoundError, match="block residual model summary"):
        main_module.collect_mouse_block_residual_model_summaries(
            saved_sessions=[saved_session],
            output_path=tmp_path / "cross_session_analysis",
            mouse="CT024",
        )


def test_load_cross_mouse_block_residual_model_summaries_combines_mice(tmp_path):
    """Cross-mouse loader should combine mouse-level summaries with mouse order."""
    main_module = load_main_module()
    for mouse in ("CT024", "CT025"):
        cross_session_path = tmp_path / mouse / "cross_session_analysis"
        cross_session_path.mkdir(parents=True)
        (
            make_residual_summary(
                session_id=f"{mouse}_2026-06-09_143852",
                mouse=mouse,
                date="2026-06-09",
            )
            .assign(training_day=1)
            .to_csv(cross_session_path / f"{mouse}_block_residual_model_summary.csv", index=False)
        )

    combined = main_module.load_cross_mouse_block_residual_model_summaries(
        data_root=tmp_path,
        mice=["CT024", "CT025"],
    )

    assert combined.shape[0] == 8
    assert combined["source_mouse"].tolist() == ["CT024"] * 4 + ["CT025"] * 4
    assert combined["mouse_order"].tolist() == [0] * 4 + [1] * 4


def test_collect_cross_mouse_block_residual_model_summaries_writes_combined_csv(tmp_path):
    """Cross-mouse collector should write the configured combined CSV."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "CT024" / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    make_residual_summary(
        session_id="CT024_2026-06-09_143852",
        mouse="CT024",
        date="2026-06-09",
    ).assign(training_day=1).to_csv(
        cross_session_path / "CT024_block_residual_model_summary.csv",
        index=False,
    )

    saved_path = main_module.collect_cross_mouse_block_residual_model_summaries(
        data_root=tmp_path,
        output_path=tmp_path / "cross_mouse_analysis",
        mice=["CT024"],
    )

    assert saved_path == tmp_path / "cross_mouse_analysis" / "cross_mouse_block_residual_model_summary.csv"
    assert pd.read_csv(saved_path, na_filter=False).shape[0] == 4


def test_main_cross_mouse_metrics_collects_residual_model_summaries(monkeypatch):
    """Cross-mouse entry point should collect residual model summaries."""
    main_module = load_main_module()
    calls = []

    monkeypatch.setattr(main_module, "discover_mice_with_overall_performance", lambda _root: ["CT024"])
    monkeypatch.setattr(
        main_module,
        "run_cross_mouse_learning_curve",
        lambda **kwargs: calls.append(("learning", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "run_cross_mouse_session_metric_curves",
        lambda **kwargs: calls.append(("session", kwargs)),
    )
    monkeypatch.setattr(
        main_module,
        "collect_cross_mouse_multisession_block_performance",
        lambda **kwargs: calls.append(("blocks", kwargs)),
    )
    monkeypatch.setattr(
        main_module,
        "collect_cross_mouse_block_residual_model_summaries",
        lambda **kwargs: calls.append(("residual_models", kwargs)),
        raising=False,
    )

    main_module.main_cross_mouse_metrics(
        data_root=Path("/tmp/contextProjectData"),
        use_discovered_mice=True,
        mice=[],
    )

    assert [call[0] for call in calls] == ["learning", "session", "blocks", "residual_models"]
    assert calls[-1][1]["mice"] == ["CT024"]
    assert calls[-1][1]["output_path"] == Path("/tmp/contextProjectData/cross_mouse_analysis")
