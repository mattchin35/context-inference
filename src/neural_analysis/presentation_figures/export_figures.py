"""Render saved presentation results to PNG and self-contained HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from . import results


GROUPS = {"session": ("regional", "prediction", "replays"),
          "cohort": ("regional", "prediction", "presentation examples")}


def read_manifest(path: Path) -> dict:
    """Load a versioned session/cohort manifest from a trusted local Path."""
    data = json.loads(path.read_text())
    if (data.get("analysis_version") != results.ANALYSIS_VERSION
            or data.get("schema_version") != results.SCHEMA_VERSION or data.get("kind") not in GROUPS):
        raise ValueError(f"Unsupported presentation manifest: {path}")
    return data


def read_panels(manifest: Path, group: str, theme: str = "light") -> tuple[dict, dict]:
    """Render one saved group into named Matplotlib and interactive Plotly figures.

    Manifest is a local Path. Group is an entry in GROUPS for that manifest
    kind. Theme selects light or dark. Figures retain archive units and axes.
    The caller owns and closes all returned Matplotlib Figures.
    """
    manifest = Path(manifest)
    data = read_manifest(manifest)
    if group not in GROUPS[data["kind"]] or theme not in ("light", "dark"):
        raise ValueError(f"Unknown presentation group or theme: {group}, {theme}")
    with plt.style.context("default" if theme == "light" else "dark_background"), plt.rc_context({
            "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "figure.max_open_warning": 0}):
        if data["kind"] == "session":
            return _session_panels(manifest, data, group, theme)
        fields, meta = results.load_result(manifest.parent / data["archives"]["cohort"])
        if group == "presentation examples":
            return _selected_replays(fields["presentation_examples"], theme)
        if group == "regional":
            from .regional_plotting import render_regional_cohort
            return render_regional_cohort(fields, meta, theme), {}
        from .prediction_plotting import render_prediction_cohort
        return render_prediction_cohort(fields, meta, theme), {}


def _session_panels(manifest: Path, data: dict, group: str, theme: str) -> tuple[dict, dict]:
    """Return static/interactive panels from the needed session summary fields only."""
    path = manifest.parent / data["archives"][group]
    if group == "prediction":
        from .prediction_plotting import render_prediction
        fields, meta = results.load_result(path, keys=("curves", "available_units", "source_pool_size",
                                                       "row_positions", "panel_d_pcs"))
        return render_prediction(fields, meta, theme), {}
    if group == "regional":
        from .regional_plotting import render_regional
        fields, meta = results.load_result(path, keys=("scores", "units", "classes", "confusion", "recovery", "folds",
                                                       "transfer_mapping", "reward_discrimination", "delivery_latency"))
        return render_regional(fields, meta, theme), {}
    from .replay_plotting import interactive_replays, render_replays
    replay, meta = results.load_result(path)
    movement, _ = results.load_result(manifest.parent / data["archives"]["movement"])
    return render_replays(replay, movement, meta, theme), interactive_replays(replay, movement, meta, theme)


def _selected_replays(examples, theme: str) -> tuple[dict, dict]:
    """Use the saved cohort motif-to-manifest table to load each example's local replay."""
    static, interactive = {}, {}
    for path, selected in examples.groupby("manifest", sort=False):
        drawings, controls = read_panels(Path(path), "replays", theme)
        motifs = tuple(selected.motif)
        for key, figure in drawings.items():
            if any(motif in key for motif in motifs):
                static[key] = figure
            else:
                plt.close(figure)
        interactive.update({k: f for k, f in controls.items() if any(m in k for m in motifs)})
    return static, interactive


def export_figures(manifest: Path, output_dir: Path, *, theme: str = "light") -> list[Path]:
    """Export all main and backup panels and self-contained replays, returning output Paths."""
    manifest, output_dir = Path(manifest), Path(output_dir)
    data = read_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for group in GROUPS[data["kind"]]:
        figures, interactive = read_panels(manifest, group, theme)
        try:
            for key, figure in figures.items():
                path = output_dir / (key + ".png")
                figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
                paths.append(path)
            for key, replay in interactive.items():
                path = output_dir / (key + ".html")
                replay.write_html(path, include_plotlyjs=True, full_html=True, auto_play=False)
                paths.append(path)
        finally:
            for figure in figures.values():
                plt.close(figure)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Export an archive given CLI arguments and return the integer exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    args = parser.parse_args(argv)
    for path in export_figures(args.manifest, args.output_dir, theme=args.theme):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
