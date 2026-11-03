"""Prediction-only Plotly helpers; no timeline orchestration, fitting or file access."""

from dataclasses import dataclass, replace

import numpy as np
import plotly.graph_objects as go

from .modeling import build_pc_contributions
from .results import SessionPredictions
from .config import GENERALIZATION_MODE


def analysis_mode_label(analysis_mode: str) -> str:
    """Return the visible analysis name for a validated canonical mode identifier."""
    return ("Correct-rewarded generalization" if analysis_mode == GENERALIZATION_MODE
            else "Mixed-outcome decoding")


def training_population_label(analysis_mode: str) -> str:
    """Return the training-population caption for a validated canonical mode string."""
    return "correct-rewarded training" if analysis_mode == GENERALIZATION_MODE else "mixed-outcome training"


def prediction_target_label(analysis_mode: str, target_name: str, source_label: str) -> str:
    """Return a mode-specific binary label, or retain a continuous target's source label."""
    if target_name in ("current_state", "current_action"):
        suffix = "generalization" if analysis_mode == GENERALIZATION_MODE else "decoding"
        return f"{target_name.removeprefix('current_').capitalize()} {suffix}"
    return source_label


def prediction_display_label(display: "PredictionDisplay") -> str:
    """Return the selected target label and training population from saved display data."""
    inputs = display.result.inputs
    source = inputs.target_metadata[display.target_name]["display_label"]
    target = prediction_target_label(inputs.analysis_mode, display.target_name, source)
    return f"{target} ({training_population_label(inputs.analysis_mode)})"


@dataclass(frozen=True)
class PredictionDisplay:
    """Saved result and display settings, independent of all model preparation.

    target_name selects one saved target; pc_mode is hidden/raw/weighted,
    pc_order original/contribution, and pc_count a positive visible-row count.
    Original-rank ranges may be supplied by existing timeline controls.
    """

    result: SessionPredictions
    target_name: str
    pc_mode: str = "hidden"
    pc_order: str = "original"
    pc_count: int = 10

    def __post_init__(self) -> None:
        """Reject invalid saved-target names or display options; return None."""
        if (self.target_name not in self.result.inputs.target_names
                or self.pc_mode not in {"hidden", "raw", "weighted"}
                or self.pc_order not in {"original", "contribution"} or self.pc_count < 1):
            raise ValueError("Invalid prediction display target/mode/order/count")


def selected_pc_ids(display: PredictionDisplay, role: str, pc_range: tuple[int, int] | None) -> tuple[int, ...]:
    """Return original one-based PC IDs in visible order; model feature count is unchanged."""
    model = display.result.regions[role]
    target = display.result.inputs.target_names.index(display.target_name)
    if display.pc_order == "contribution":
        return tuple(model.contribution_pc_order[target, :display.pc_count].tolist())
    first, last = pc_range or (1, min(display.pc_count, len(model.pc_ids)))
    if first < 1 or last < first or last > len(model.pc_ids):
        raise ValueError("Original PC range must be within saved model ranks")
    return tuple(range(first, last + 1))


def add_prediction_tracks(
    figure: go.Figure, display: PredictionDisplay, first_row: int,
    loaded_range_s: tuple[float, float] | None = None,
) -> None:
    """Add PFC/HPC prediction and actual overlays at first_row and first_row+1.

    Figure has preallocated shared-time rows. Saved arrays retain W,B,T/F,K
    contracts. x coordinates are elapsed seconds; actuals span all N choices,
    independently of regional coverage. Prediction segments use exact bin
    edges and None breaks after each window. No interpolation crosses gaps.
    """
    inputs = display.result.inputs
    target = inputs.target_names.index(display.target_name)
    binary = inputs.target_families[target] == "categorical"
    label = prediction_display_label(display)
    for offset, role in enumerate(("PFC", "HPC")):
        row = first_row + offset
        actual_x, actual_y, actual_custom = [], [], []
        for position, choice in enumerate(inputs.timeline.choices.itertuples()):
            actual = inputs.target_values[position, target]
            actual_x.extend([choice.elapsed_s - 2, choice.elapsed_s + 2, None])
            actual_y.extend([actual, actual, None])
            actual_custom.extend([[choice.trial_id, actual]] * 2 + [[None, None]])
        figure.add_trace(go.Scatter(
            x=actual_x, y=actual_y, mode="lines", name=f"{role} actual",
            line={"color": "#111827", "width": 1.5, "dash": "dot"}, connectgaps=False,
            customdata=actual_custom,
            hovertemplate=label + "<br>Actual %{y:.5g}<br>Trial %{customdata[0]}<extra></extra>",
        ), row=row, col=1)
        if role in display.result.regions:
            region = inputs.timeline.regions[role]
            model = display.result.regions[role]
            x, y, custom = [], [], []
            for window, position in enumerate(region.choice_positions):
                choice = inputs.timeline.choices.iloc[position]
                elapsed = choice.elapsed_s + region.bin_edges_s
                if loaded_range_s is not None and (elapsed[-1] <= loaded_range_s[0] or elapsed[0] >= loaded_range_s[1]):
                    continue
                fold = int(inputs.fold_ids[position])
                for bin_index in range(len(elapsed) - 1):
                    value = model.predictions[window, bin_index, target]
                    x.extend([elapsed[bin_index], elapsed[bin_index + 1]])
                    y.extend([value, value])
                    info = [choice.trial_id, region.bin_edges_s[bin_index], region.bin_edges_s[bin_index + 1],
                            elapsed[bin_index], elapsed[bin_index + 1], fold,
                            inputs.target_values[position, target], condition_label(display, position),
                            model.prediction_reason[window, target],
                            model.intercepts[target, fold] if fold >= 0 else None]
                    custom.extend([info, info])
                x.append(None)
                y.append(None)
                custom.append([None] * 10)
            figure.add_trace(go.Scattergl(
                x=x, y=y, mode="lines", name=f"{role} prediction", connectgaps=False,
                line={"color": "#2563eb" if role == "PFC" else "#b45309", "width": 2}, customdata=custom,
                hovertemplate=f"{role}: {label}<br>Prediction %{{y:.5g}}; actual %{{customdata[6]:.5g}}"
                "<br>Trial %{customdata[0]}; held-out fold %{customdata[5]}"
                "<br>Elapsed [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s"
                "<br>Choice-relative [%{customdata[1]:.3f}, %{customdata[2]:.3f}) s"
                "<br>Conditions: %{customdata[7]}<br>Intercept: %{customdata[9]:.5g}<extra></extra>",
            ), row=row, col=1)
        else:
            figure.add_annotation(text=f"{role} unavailable: {inputs.timeline.unavailable.get(role, 'No saved model')}",
                                  x=0.5, y=0.5, xref="x domain", yref="y domain", showarrow=False,
                                  row=row, col=1)
        figure.update_yaxes(title_text="P(left)" if binary else label, row=row, col=1)
        if binary:
            figure.update_yaxes(range=[0, 1], autorange=False, row=row, col=1)


def prediction_heatmap_values(
    display: PredictionDisplay, role: str, pc_ids: tuple[int, ...],
    loaded_range_s: tuple[float, float] | None,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Return sliced regional record, original window indices and W_visible,B,P values.

    PC IDs retain one-based original rank. Only visible-window contributions
    are constructed. Values use raw PC units or native/logit contribution units.
    The original fixed basis and all scores remain untouched.
    """
    inputs = display.result.inputs
    regional = inputs.timeline.regions[role]
    choices = inputs.timeline.choices.iloc[regional.choice_positions]
    keep = np.ones(len(choices), dtype=bool)
    if loaded_range_s is not None:
        keep = ((choices.elapsed_s.to_numpy() + 2 > loaded_range_s[0])
                & (choices.elapsed_s.to_numpy() - 2 < loaded_range_s[1]))
    windows = np.flatnonzero(keep)
    sliced = replace(regional, scores=regional.scores[windows],
                     choice_positions=regional.choice_positions[windows],
                     window_total_variance=regional.window_total_variance[windows])
    if display.pc_mode == "weighted":
        target = inputs.target_names.index(display.target_name)
        values = build_pc_contributions(regional.scores, inputs.fold_ids[regional.choice_positions],
                                        display.result.regions[role], target, pc_ids, window_indices=windows)
    else:
        values = regional.scores[windows][:, :, np.asarray(pc_ids) - 1]
    return sliced, windows, values


def enrich_prediction_heatmap(data: dict, display: PredictionDisplay, role: str, pc_ids: tuple[int, ...]) -> dict:
    """Return heatmap data with ordinal y centers and PC/weight/fold/condition hover.

    Input x/z/customdata use the existing heatmap helper's M-bin real-time
    geometry: z(P,M), customdata(P,M,8). Append original PC, weight, fold and
    memberships, prediction and actual to yield (P,M,14); gap cells remain unknown.
    """
    inputs = display.result.inputs
    target = inputs.target_names.index(display.target_name)
    model = display.result.regions[role]
    old = data["customdata"]
    custom = np.full((*old.shape[:2], 14), None, dtype=object)
    custom[:, :, :8] = old
    custom[:, :, 8] = np.asarray(pc_ids)[:, None]
    positions = {trial_id: position for position, trial_id in enumerate(inputs.timeline.choices.trial_id)}
    regional = inputs.timeline.regions[role]
    window_by_choice = {int(position): window for window, position in enumerate(regional.choice_positions)}
    coefficient_indices = [int(np.flatnonzero(model.pc_ids == rank)[0]) for rank in pc_ids]
    for column in range(old.shape[1]):
        trial_id = old[0, column, 0]
        if trial_id is None:
            continue
        position = positions[trial_id]
        fold = int(inputs.fold_ids[position])
        custom[:, column, 10] = fold
        custom[:, column, 11] = condition_label(display, position)
        bin_index = int(np.searchsorted(regional.bin_edges_s, old[0, column, 3], side="right") - 1)
        custom[:, column, 12] = model.predictions[window_by_choice[position], bin_index, target]
        custom[:, column, 13] = inputs.target_values[position, target]
        if fold >= 0 and model.fit_status[target, fold] == "ok":
            custom[:, column, 9] = model.coefficients[target, fold, coefficient_indices]
    return {**data, "y": np.arange(len(pc_ids)), "customdata": custom}


def condition_label(display: PredictionDisplay, choice_position: int) -> str:
    """Return every true saved canonical condition for one shared choice row."""
    inputs = display.result.inputs
    return ", ".join(name for name, member in zip(inputs.condition_names,
                     inputs.condition_masks[choice_position], strict=True) if member)
