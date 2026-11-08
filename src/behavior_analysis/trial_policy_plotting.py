"""Plot saved trial-policy comparison arrays without rerunning inference."""

from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path
import textwrap
from typing import Any, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


_KNOWN_COLORS = {
    "forgetting_q_learning": "#1f77b4",
    "bayesian_hmm": "#ff7f0e",
    "win_stay_lose_switch": "#2ca02c",
    "synthetic_fourth": "#9467bd",
}


def _model_color(model_id: str) -> str:
    """Return a deterministic model color for a string candidate identifier.

    model_id is one stable candidate ID; the return is an opaque hex RGB color.
    Known IDs use fixed colors and other IDs use a stable SHA-256-derived hue.
    """
    if model_id in _KNOWN_COLORS:
        return _KNOWN_COLORS[model_id]
    digest = hashlib.sha256(model_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / (2**32)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.72)
    return "#%02x%02x%02x" % (round(red * 255), round(green * 255), round(blue * 255))


def _parameter(metadata: dict[str, Any], name: str, default: Any) -> Any:
    """Read a display parameter from metadata's plain parameter dictionary.

    metadata is a saved metadata mapping; name is a scalar parameter key.
    Returns the stored value or default without changing numerical arrays.
    """
    return metadata.get("parameters", {}).get(name, default)


def _set_axes_style(figure: Figure, axes: Sequence[Axes]) -> None:
    """Set opaque white backgrounds and black labels on a figure's axes.

    figure is a Matplotlib Figure and axes is a sequence of its axes; styling
    changes only display colors and returns None.
    """
    figure.patch.set_facecolor("white")
    figure.patch.set_alpha(1.0)
    for axis in axes:
        axis.set_facecolor("white")
        axis.tick_params(colors="black", labelsize=10)
        axis.xaxis.label.set_color("black")
        axis.yaxis.label.set_color("black")
        axis.title.set_color("black")
        for spine in axis.spines.values():
            spine.set_color("black")


def _row_positions(result: Any, offset: int) -> np.ndarray:
    """Return displayed integer trial positions for one source result.

    result has T aligned retained rows; offset is a display-row offset. Returns
    int64 positions of shape (T,), in display-row units rather than seconds.
    """
    return np.arange(len(result.row_position), dtype=np.int64) + int(offset)


def _block_boundaries(result: Any, positions: np.ndarray) -> list[float]:
    """Find display x coordinates before rows whose source block ID changes.

    result.block_id and positions each have shape (T,); coordinates are between
    retained rows in trial-position units. No boundary is inferred across IDs
    within separate source results.
    """
    if len(positions) < 2:
        return []
    block_ids = np.asarray(result.block_id).astype(str)
    return [float(positions[index] - 0.5)
            for index in range(1, len(positions)) if block_ids[index] != block_ids[index - 1]]


def _draw_boundaries(axes: Sequence[Axes], results: Sequence[Any], offsets: Sequence[int]) -> None:
    """Mark task-block and source-session boundaries on each diagnostic panel.

    axes contains the shared-x diagnostic panels, results contains source-local
    arrays, and offsets maps each result's local rows to display positions.
    Boundary coordinates use trial-position units; returns None.
    """
    for result, offset in zip(results, offsets):
        positions = _row_positions(result, offset)
        for boundary in _block_boundaries(result, positions):
            for axis in axes:
                axis.axvline(boundary, color="0.35", linewidth=0.8, linestyle=":", gid="block-boundary")
    for source_index in range(1, len(results)):
        boundary = float(offsets[source_index] - 0.5)
        for axis in axes:
            axis.axvline(boundary, color="black", linewidth=1.0, linestyle="--", gid="session-boundary")


def _shade_excluded(axis: Axes, result: Any, positions: np.ndarray) -> None:
    """Shade rows excluded from the common comparison on one panel.

    result.comparison_valid is bool[T] and positions is integer[T]. Each
    excluded row receives a translucent vertical span in trial-position units.
    """
    valid = np.asarray(result.comparison_valid, dtype=bool)
    for position in positions[~valid]:
        patch = axis.axvspan(position - 0.5, position + 0.5, color="#d9d9d9", alpha=0.55, zorder=0)
        patch.set_gid("excluded-row")


def _draw_result_panels(
    axes: Sequence[Axes], results: Sequence[Any], offsets: Sequence[int], model_ids: Sequence[str],
) -> None:
    """Draw source-local arrays into the five (or three) shared diagnostic panels.

    Each result contributes T rows by K models, aligned with model_ids. axes has
    three panels without HMM arrays or five with them; positions are trial rows.
    The function reads saved arrays only and returns None.
    """
    has_hmm = len(axes) == 5
    for result, offset in zip(results, offsets):
        positions = _row_positions(result, offset)
        for axis in axes[:3]:
            _shade_excluded(axis, result, positions)
        observed = np.asarray(result.observed_action, dtype=float)
        observed_valid = np.isfinite(observed) & np.isin(observed, (0, 1))
        axes[0].scatter(positions[observed_valid], 1 - observed[observed_valid],
                        marker="o", s=22, facecolors="black", edgecolors="black",
                        label="Mouse choice" if result is results[0] else "_nolegend_", zorder=4)

        p_right = np.asarray(result.p_right, dtype=float)
        log_likelihoods = np.asarray(result.log_likelihoods, dtype=float)
        independent_winners = np.asarray(result.independent_best_model_mask, dtype=bool)
        for model_index, model_id in enumerate(model_ids):
            color = _model_color(str(model_id))
            axes[0].plot(positions, p_right[:, model_index], color=color, linewidth=1.2,
                         label=str(model_id) if result is results[0] else "_nolegend_")
            axes[1].plot(positions, log_likelihoods[:, model_index], color=color, linewidth=1.1,
                         label=str(model_id) if result is results[0] else "_nolegend_")
            winner_rows = np.flatnonzero(independent_winners[:, model_index])
            axes[2].scatter(positions[winner_rows], np.full(len(winner_rows), model_index),
                            color=color, marker="s", s=20, label="_nolegend_", zorder=3)

        if has_hmm:
            smoothed = np.asarray(result.hmm_smoothed_probs, dtype=float)
            for model_index, model_id in enumerate(model_ids):
                color = _model_color(str(model_id))
                axes[3].plot(positions, smoothed[:, model_index], color=color, linewidth=1.2,
                             label=str(model_id) if result is results[0] else "_nolegend_")
            hmm_winners = np.asarray(result.hmm_smoothed_best_model_mask, dtype=bool)
            for model_index, model_id in enumerate(model_ids):
                rows = np.flatnonzero(hmm_winners[:, model_index])
                axes[4].scatter(positions[rows], np.full(len(rows), model_index),
                                color=_model_color(str(model_id)), marker="s", s=20,
                                label="_nolegend_", zorder=3)
            for axis in (axes[3], axes[4]):
                _shade_excluded(axis, result, positions)


def _caption(results: Sequence[Any], metadata_records: Sequence[dict[str, Any]],
             model_ids: Sequence[str]) -> str:
    """Build a scientific caption from metadata and saved assignment arrays.

    results contains one or more source-local T-by-K records; metadata_records
    supplies scalar error/stay settings. Returns readable caption text with no
    additional inference or numerical transformations.
    """
    source_names = [str(metadata.get("session_id", result.session_id))
                    for result, metadata in zip(results, metadata_records)]
    error_values = sorted({str(_parameter(metadata, "opposite_action_error_prob", "unknown"))
                           for metadata in metadata_records})
    stay_values = sorted({str(_parameter(metadata, "stay_prob", "unknown"))
                          for metadata in metadata_records})
    independent_ties = sum(int(np.count_nonzero(np.asarray(result.independent_tie) == 1)) for result in results)
    hmm_ties = sum(int(np.count_nonzero(np.asarray(result.hmm_smoothed_tie) == 1))
                   for result in results if result.hmm_smoothed_tie is not None)
    statuses = sorted({str(result.status) for result in results})
    status_text = " Status: " + ", ".join(statuses) + "." if any(
        status != "ok" for status in statuses) else ""
    hmm_count = sum(result.hmm_smoothed_probs is not None for result in results)
    tie_text = f" Independent ties: {independent_ties}."
    if hmm_count:
        tie_text += f" HMM assignment ties: {hmm_ties}."
        temporal_text = (
            f"Trial stay probability: {', '.join(stay_values)}. HMM resets at each source session "
            f"({len(results)} session reset(s)). HMM states on excluded rows reflect the prior and "
            "available surrounding evidence. HMM marginal assignments use the full session and "
            "are retrospective, not next-choice predictions."
        )
    else:
        temporal_text = "HMM smoothing disabled; independent comparisons use causal pre-choice policies."
    caption = (
        f"Candidates: {', '.join(map(str, model_ids))}. Shared opposite-action error probability: "
        f"{', '.join(error_values)}. "
        f"All tied maxima are shown; no tied winner is selected.{tie_text} "
        "Gray spans mark excluded comparisons. "
        f"Sources: {', '.join(source_names)}. Policy histories initialize independently by source. "
        f"{temporal_text}"
        f"{status_text}"
    )
    return textwrap.fill(caption, width=115)


def _make_figure(
    results: Sequence[Any], metadata_records: Sequence[dict[str, Any]], *, output_path: str | Path,
) -> Path:
    """Render and save source-local trial diagnostics as an opaque white PNG.

    results are one or more TrialPolicyResult-like saved records, metadata_records
    are their same-order plain mappings, and output_path is a PNG destination.
    Returns the saved Path; trial coordinates are zero-based retained row indices.
    """
    if not results or len(results) != len(metadata_records):
        raise ValueError("results and metadata_records must be nonempty sequences of equal length")
    model_ids = tuple(str(value) for value in results[0].model_ids)
    if any(tuple(str(value) for value in result.model_ids) != model_ids for result in results):
        raise ValueError("All source results must use the same ordered model IDs")
    has_hmm = all(result.hmm_smoothed_probs is not None for result in results)
    if any((result.hmm_smoothed_probs is not None) != has_hmm for result in results):
        raise ValueError("All source results must either include HMM arrays or omit them")
    panel_count = 5 if has_hmm else 3
    figure, grid = plt.subplots(panel_count, 1, figsize=(13, 2.7 * panel_count),
                                facecolor="white", sharex=True, squeeze=False)
    axes = tuple(grid[:, 0])
    offsets: list[int] = []
    total_rows = 0
    for result in results:
        offsets.append(total_rows)
        total_rows += len(result.row_position)

    _draw_result_panels(axes, results, offsets, model_ids)
    axes[0].set_ylabel("P(Right)", fontsize=12)
    axes[0].set_ylim(-0.08, 1.08)
    axes[0].set_yticks((0, 1), labels=("Left", "Right"))
    axes[1].set_ylabel("Log likelihood\n(natural log)", fontsize=12)
    axes[2].set_ylabel("Independent\nwinner(s)", fontsize=12)
    axes[2].set_yticks(range(len(model_ids)), labels=model_ids)
    if has_hmm:
        axes[3].set_ylabel("Smoothed\nprobability", fontsize=12)
        axes[3].set_ylim(-0.05, 1.05)
        axes[4].set_ylabel("HMM marginal\nassignment(s)", fontsize=12)
        axes[4].set_yticks(range(len(model_ids)), labels=model_ids)
    axes[-1].set_xlabel("Retained trial position (zero-based)", fontsize=12)
    axes[-1].set_xlim(-0.5, max(total_rows - 0.5, 0.5))
    for axis in axes:
        axis.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    _draw_boundaries(axes, results, offsets)
    _set_axes_style(figure, axes)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
                      frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=10)
    caption = figure.text(0.01, 0.005, _caption(results, metadata_records, model_ids),
                          ha="left", va="bottom", fontsize=12, color="black", linespacing=1.15)
    # Reserve the rendered caption height for ordinary multisession batches.
    figure.canvas.draw()
    caption_top = caption.get_window_extent(figure.canvas.get_renderer()).y1 / figure.bbox.height
    figure.tight_layout(rect=(0, caption_top + .045, 1, 0.94))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", facecolor="white", transparent=False, dpi=160)
    plt.close(figure)
    return path


def plot_trial_policy_diagnostic(result: Any, metadata: dict[str, Any], *, output_path: str | Path) -> Path:
    """Plot one saved trial-policy result, returning its PNG path.

    result contains T rows and K ordered models; metadata is its plain saved
    settings/source mapping; output_path is a destination path. Numerical arrays
    retain their saved row/model axes and displayed x coordinates are trial rows.
    """
    return _make_figure([result], [metadata], output_path=output_path)


def plot_multisession_trial_policy_diagnostic(
    results: Sequence[Any], metadata_records: Sequence[dict[str, Any]], *, output_path: str | Path,
) -> Path:
    """Plot source-local saved results on one display axis with marked resets.

    results is an ordered sequence of source-local T_s-by-K records and
    metadata_records has one mapping per source. The saved row/model arrays are
    unchanged; x coordinates concatenate display rows but never infer jointly.
    Returns the destination PNG Path.
    """
    return _make_figure(results, metadata_records, output_path=output_path)
