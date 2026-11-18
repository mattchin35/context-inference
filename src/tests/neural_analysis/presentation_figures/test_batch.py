"""The batch driver calls the full session workflow and preserves its roster."""

from pathlib import Path

import pandas as pd


def test_batch_runs_and_reuses_each_session_with_a_named_cohort(tmp_path, monkeypatch):
    """Three metadata Paths, including one excluded session, produce a cohort manifest."""
    from src.neural_analysis.presentation_figures import run_batch as batch

    calls = []
    def session(path, *, seed, dry_run, rerun):
        """Record one full session invocation and return its synthetic status."""
        calls.append((path, seed, dry_run, rerun))
        status = {"one": "computed", "two": "reused", "three": "excluded"}[path.stem]
        return {"session_id": path.stem, "status": status, "identity": path.stem,
                "reason": "missing units" if status == "excluded" else "", "manifest": None}

    monkeypatch.setattr(batch, "run_session", session)
    monkeypatch.setattr(batch, "assemble_cohort", lambda reports: {"roster": pd.DataFrame(reports)})
    paths = [tmp_path / (name + ".json") for name in ("one", "two", "three")]
    for path in paths:
        path.write_text("{}")
    out = tmp_path / "batch"
    preview = batch.run_batch(paths, output_root=out, workers=1, dry_run=True)
    assert preview["status"] == "dry-run" and not out.exists()
    report = batch.run_batch(paths, output_root=out, workers=1, seed=8)
    assert [(path, seed) for path, seed, dry, _ in calls if not dry] == [(p, 8) for p in paths]
    assert [r["status"] for r in report["sessions"]] == ["computed", "reused", "excluded"]
    assert Path(report["manifest"]).is_file()
