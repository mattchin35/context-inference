"""Behavior-selected single-session replays with excerpt-wide held-out fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.behavior_analysis.collect_events import make_event_labels
from src.neural_analysis.communication_subspace.preparation import _overlapping_indices
from src.neural_analysis.event_sequences.selection import select_ideal_transitions
from src.neural_analysis.population_decoding.targets import TARGET_NAMES, numeric_column

from .decoding import (
    PROBABILITY_COLUMNS, WIDTHS_MS, class_table, fit_readout, predict_readout,
)
from .preparation import REGIONAL_WINDOW_S, REGIONS, PreparedChoices

MOTIFS = ("new_side_omission", "ideal_transition", "four_outcome")
SIDES = ("right", "left", "dark")
CONTEXT_EVENTS = dict(zip(("enter_right_patch", "enter_left_patch", "enter_dark_period"), SIDES))
PADDING_CHOICES = 3
TABLE_COLUMNS = {
    "episodes": ("motif", "status", "reason", "start_elapsed_s", "end_elapsed_s",
                 "first_trial_id", "last_trial_id", "repeat_count", "missing_deliveries"),
    "tracks": ("motif", "width_ms", "region", "target", "row_position", "trial_id",
               "bin_left_elapsed_s", "bin_right_elapsed_s", *PROBABILITY_COLUMNS),
    "events": ("motif", "elapsed_s", "event", "label", "trial_id"),
    "trials": ("motif", "row_position", "trial_id", "block_id", "choice_elapsed_s", "context",
               "choice", "outcome", "trial_type", "delivery_elapsed_s", "delivery_missing", "in_motif"),
    "fits": ("motif", "width_ms", "region", "target", "n_train", "n_held_out", "n_purged", "reason"),
    "held_out_blocks": ("motif", "block_id"),
}


@dataclass(frozen=True)
class Episode:
    """Original positions for one padded display interval and its defining motif.

    display_rows and motif_rows are ascending integer (row,) arrays. Markers
    are original positions, except switch_s, a logged UTC timestamp. Missing
    optional markers are None or NaN.
    """

    display_rows: np.ndarray
    motif_rows: np.ndarray
    switch_s: float = np.nan
    new_side: int | None = None
    first_new_choice: int | None = None
    first_new_delivery: int | None = None
    repeat_count: int | None = None


def compute_replays(prepared: PreparedChoices, *, ephys_rig: bool = True) -> dict[str, pd.DataFrame]:
    """Return Figure 4 episode, trial, event, fit and probability tables.

    Logged events and original trial IDs are shared by widths. Display and bin
    edges are elapsed seconds from prepared.time_origin_s. Binary class 0 is
    right. ephys_rig selects the legacy pump mapping (right pump3/left pump2
    on the electrophysiology rig, right pump2/left pump1 on the other rig).
    """
    tables = {name: [] for name in TABLE_COLUMNS}
    all_rows = np.arange(len(prepared.trials))
    delivered_s = recorded_deliveries(prepared, all_rows, ephys_rig=ephys_rig)
    choice_rows = prepared.choices.row_position.to_numpy(dtype=int)
    covered_rows = np.intersect1d(
        prepared.windows[REGIONAL_WINDOW_S, 500].row_positions,
        prepared.windows[REGIONAL_WINDOW_S, 100].row_positions,
    )
    selections = _select_episodes(prepared, choice_rows, covered_rows, delivered_s)
    for motif in MOTIFS:
        episode = selections[motif]
        if episode is None:
            tables["episodes"].append(dict(
                motif=motif, status="unavailable",
                reason="no_covered_padded_episode" if len(covered_rows) else "no_joint_choice_windows",
            ))
            continue
        excerpt = _excerpt_trials(prepared, motif, episode, delivered_s)
        blocks = excerpt.block_id.dropna().drop_duplicates().to_numpy()
        elapsed = excerpt.choice_elapsed_s.dropna().to_numpy(dtype=float)
        left_s, right_s = elapsed[0] + REGIONAL_WINDOW_S[0], elapsed[-1] + REGIONAL_WINDOW_S[1]
        reason = ""
        if episode.new_side is not None and not np.isfinite(episode.switch_s):
            reason = "switch_timestamp_unrecorded"
        if episode.new_side is not None and episode.first_new_delivery is None:
            reason = ", ".join(filter(None, (reason, "first_new_side_delivery_unrecorded")))
        tables["episodes"].append(dict(
            motif=motif, status="available", reason=reason, start_elapsed_s=left_s, end_elapsed_s=right_s,
            first_trial_id=excerpt.trial_id.iloc[0], last_trial_id=excerpt.trial_id.iloc[-1],
            repeat_count=episode.repeat_count, missing_deliveries=int(excerpt.delivery_missing.sum()),
        ))
        tables["trials"].extend(excerpt.to_dict("records"))
        tables["events"].extend(_excerpt_events(prepared, motif, episode, excerpt, left_s, right_s))
        tables["held_out_blocks"].extend(dict(motif=motif, block_id=block) for block in blocks)
        fits, tracks = _fit_tracks(prepared, motif, episode.display_rows, blocks)
        tables["fits"].extend(fits)
        tables["tracks"].extend(tracks)

    fields = {name: pd.DataFrame(rows, columns=TABLE_COLUMNS[name]) for name, rows in tables.items()}
    fields["classes"] = class_table(prepared)
    fields["pump_mapping"] = pd.DataFrame([dict(
        ephys_rig=ephys_rig, right="pump3" if ephys_rig else "pump2",
        left="pump2" if ephys_rig else "pump1",
    )])
    return fields


def recorded_deliveries(prepared: PreparedChoices, row_positions: np.ndarray, *,
                        ephys_rig: bool = True) -> np.ndarray:
    """Return UTC second timestamps (W,) for W original trial positions.

    Find the first logged side-matched pump on or after an ordinary rewarded
    choice, before the following original trial starts (or session log end).
    Missing events, manual rewards and unrewarded choices yield NaN. ephys_rig
    uses the same Boolean mapping as the behavior collection code.
    """
    positions = np.asarray(row_positions, dtype=int)
    delivery_s = np.full(positions.shape, np.nan)
    trials = prepared.trials
    if prepared.events.empty or not len(positions):
        return delivery_s
    event_times = numeric_column(prepared.events.Time)
    event_names = prepared.events.Event.fillna("").astype(str).tolist()
    labels = np.asarray(make_event_labels(event_names, {"ephys_rig": ephys_rig}))
    pump_events = np.char.startswith(np.asarray(event_names, dtype=str), "pump")
    trial_start_s = numeric_column(trials.start_time)
    next_start_s = np.r_[trial_start_s[1:], np.inf]
    choice_s = numeric_column(trials.choice_time)
    outcome = prepared.targets.values[:, TARGET_NAMES.index("outcome")]
    ordinary = positions[np.isin(positions, prepared.choices.row_position.to_numpy())]
    for side, rewarded_class in (("right", 1), ("left", 0)):
        matching_times = np.sort(event_times[pump_events & (labels == f"{side} pump")])
        for output_index in np.flatnonzero(
            (outcome[positions] == rewarded_class) & np.isin(positions, ordinary)
        ):
            trial_position = positions[output_index]
            event_index = np.searchsorted(matching_times, choice_s[trial_position])
            if event_index < len(matching_times) and matching_times[event_index] < next_start_s[trial_position]:
                delivery_s[output_index] = matching_times[event_index]
    return delivery_s


def _select_episodes(prepared: PreparedChoices, choice_rows: np.ndarray,
                     covered_rows: np.ndarray, delivered_s: np.ndarray) -> dict[str, Episode | None]:
    """Select at most one covered behavior motif per panel from original row IDs."""
    selected = dict.fromkeys(MOTIFS)
    if len(covered_rows) < 2 * PADDING_CHOICES + 1:
        return selected
    selected["four_outcome"] = _outcome_episode(
        choice_rows, covered_rows, prepared.targets.values[:, TARGET_NAMES.index("outcome")],
    )

    trials = prepared.trials
    block = trials.cur_block.to_numpy()
    context = numeric_column(trials.state_int)
    choices_s = numeric_column(trials.choice_time)
    action = prepared.targets.values[:, TARGET_NAMES.index("choice")]
    correct_rewarded = prepared.condition_masks["correct_rewarded"]
    omission = prepared.condition_masks["omission"]
    episodes = select_ideal_transitions(trials)
    ideal = {int(episode[1]): episode for matches in episodes.values() for episode in matches}
    boundaries = np.flatnonzero(
        pd.notna(block[:-1]) & pd.notna(block[1:]) & (block[:-1] != block[1:])
        & np.isin(context[:-1], [0, 1]) & (context[1:] == 1 - context[:-1])
    ) + 1
    for boundary in boundaries:
        start_index = int(np.searchsorted(choice_rows, boundary))
        if start_index < PADDING_CHOICES or start_index == len(choice_rows):
            continue
        new_side = int(context[boundary])
        # Retain actual choice chronology within the current contiguous block.
        next_blocks = np.flatnonzero(block[boundary:] != block[boundary])
        stop_row = boundary + next_blocks[0] if len(next_blocks) else len(block)
        new_rows = choice_rows[start_index:np.searchsorted(choice_rows, stop_row)]
        side_rows = new_rows[action[new_rows] == new_side]
        if not len(side_rows):
            continue
        delivered = side_rows[np.isfinite(delivered_s[side_rows])]
        first_delivery = int(delivered[0]) if len(delivered) else None
        switch_s = _context_entry(
            prepared.events, new_side, choices_s[choice_rows[start_index - 1]], choices_s[new_rows[0]],
        )
        if selected["new_side_omission"] is None and first_delivery is not None and correct_rewarded[first_delivery]:
            before = side_rows[side_rows < first_delivery]
            omissions = before[omission[before]]
            if len(omissions):
                episode = _switch_episode(choice_rows, covered_rows, start_index, first_delivery,
                                          np.array([omissions[0], first_delivery]), switch_s,
                                          new_side, int(side_rows[0]), first_delivery)
                if episode is not None:
                    selected["new_side_omission"] = episode
        if selected["ideal_transition"] is None and boundary in ideal:
            sequence = ideal[boundary]
            # If the motif's rewarded choice has no pump timestamp, preserve
            # the labeled behavior, padding past the first recorded delivery when available.
            last_row = int(sequence[-1]) if first_delivery is None else max(int(sequence[-1]), first_delivery)
            episode = _switch_episode(choice_rows, covered_rows, start_index, last_row, sequence,
                                      switch_s, new_side, int(side_rows[0]), first_delivery, len(sequence) - 2)
            if episode is not None:
                selected["ideal_transition"] = episode
        if selected["new_side_omission"] is not None and selected["ideal_transition"] is not None:
            break
    return selected


def _switch_episode(choice_rows: np.ndarray, covered_rows: np.ndarray, start_index: int,
                    last_row: int, motif_rows: np.ndarray, switch_s: float, side: int,
                    first_choice: int, first_delivery: int | None,
                    repeats: int | None = None) -> Episode | None:
    """Return padded original rows for a switch and its UTC event marker."""
    last_index = int(np.searchsorted(choice_rows, last_row))
    rows = _padded_rows(choice_rows, covered_rows, start_index, last_index)
    if rows is None:
        return None
    return Episode(rows, motif_rows, switch_s, side, first_choice, first_delivery, repeats)


def _outcome_episode(choice_rows: np.ndarray, covered_rows: np.ndarray,
                     outcomes: np.ndarray) -> Episode | None:
    """Return shortest covered four-state sequence on the ordinary choice axis."""
    state_positions = [np.flatnonzero(outcomes[choice_rows] == code) for code in range(4)]
    best = None
    best_length = len(choice_rows) + 1
    for first in range(PADDING_CHOICES, len(choice_rows) - PADDING_CHOICES - 3):
        next_indices = [int(np.searchsorted(positions, first)) for positions in state_positions]
        if any(index == len(positions) for positions, index in zip(state_positions, next_indices)):
            break
        last = max(positions[index] for positions, index in zip(state_positions, next_indices))
        length = last - first + 1
        if length >= best_length:
            continue
        rows = _padded_rows(choice_rows, covered_rows, first, last)
        if rows is not None:
            best = Episode(rows, choice_rows[first:last + 1])
            best_length = length
    return best


def _padded_rows(choice_rows: np.ndarray, covered_rows: np.ndarray,
                 first: int, last: int) -> np.ndarray | None:
    """Pad three ordinary choices each side, retaining all intervening original rows."""
    start = first - PADDING_CHOICES
    stop = last + PADDING_CHOICES + 1
    if start < 0 or stop > len(choice_rows):
        return None
    padded = choice_rows[start:stop]
    if not np.isin(padded, covered_rows).all():
        return None
    return np.arange(padded[0], padded[-1] + 1)


def _context_entry(events: pd.DataFrame, side: int, after_s: float, before_s: float) -> float:
    """Return the first matching logged context entry in an inclusive UTC s interval."""
    event_name = f"enter_{SIDES[side]}_patch"
    matching = events.loc[events.Event.eq(event_name), "Time"].to_numpy(dtype=float)
    eligible = matching[(matching >= after_s) & (matching <= before_s)]
    return float(eligible.min()) if len(eligible) else np.nan


def _excerpt_trials(prepared: PreparedChoices, motif: str, episode: Episode,
                    delivered_s: np.ndarray) -> pd.DataFrame:
    """Return original display rows annotated with elapsed seconds and trial types."""
    positions = episode.display_rows
    trials = prepared.trials.iloc[positions]
    ordinary = prepared.condition_masks["overall"][positions]
    choice_s = numeric_column(trials.choice_time)
    outcome = prepared.targets.values[positions, TARGET_NAMES.index("outcome")]
    trial_type = np.where(ordinary, "unknown", "nonordinary").astype(object)
    for group in ("correct_rewarded", "omission", "incorrect"):
        trial_type[ordinary & prepared.condition_masks[group][positions]] = group
    return pd.DataFrame(dict(
        motif=motif, row_position=positions, trial_id=trials.cur_trial.to_numpy(),
        block_id=trials.cur_block.to_numpy(), choice_elapsed_s=np.where(
            ordinary, choice_s - prepared.time_origin_s, np.nan),
        context=numeric_column(trials.state_int),
        choice=prepared.targets.values[positions, TARGET_NAMES.index("choice")],
        outcome=outcome, trial_type=trial_type, delivery_elapsed_s=delivered_s[positions] - prepared.time_origin_s,
        delivery_missing=ordinary & np.isin(outcome, [0, 1]) & ~np.isfinite(delivered_s[positions]),
        in_motif=np.isin(positions, episode.motif_rows),
    ))


def _excerpt_events(prepared: PreparedChoices, motif: str, episode: Episode, rows: pd.DataFrame,
                    left_s: float, right_s: float) -> list[dict]:
    """Return actual logged events and replay annotations on the elapsed-second axis."""
    context = prepared.events.loc[prepared.events.Event.isin(CONTEXT_EVENTS), ["elapsed_s", "Event"]]
    logged = context.loc[context.elapsed_s.between(left_s, right_s)]
    previous = context.loc[context.elapsed_s < left_s].tail(1)
    markers = [dict(motif=motif, elapsed_s=row.elapsed_s, event="context",
                    label=CONTEXT_EVENTS[row.Event], trial_id=None)
               for row in pd.concat([previous, logged]).itertuples(index=False)]
    for row in rows.loc[rows.choice_elapsed_s.notna()].itertuples(index=False):
        side = SIDES[row.choice]
        entries = [("choice", side, row.choice_elapsed_s)]
        if np.isfinite(row.delivery_elapsed_s):
            entries.append(("delivery", side, row.delivery_elapsed_s))
        elif row.delivery_missing:
            entries.append(("delivery_missing", side, np.nan))
        if row.row_position == episode.first_new_choice:
            entries.append(("first_new_side_choice", side, row.choice_elapsed_s))
        if row.row_position == episode.first_new_delivery:
            entries.append(("first_new_side_delivery", side, row.delivery_elapsed_s))
        markers.extend(dict(motif=motif, elapsed_s=time, event=name, label=label, trial_id=row.trial_id)
                       for name, label, time in entries)
    if episode.new_side is not None:
        markers.append(dict(motif=motif, elapsed_s=episode.switch_s - prepared.time_origin_s,
                            event="switch", label=SIDES[episode.new_side], trial_id=None))
    if motif == "four_outcome":
        ordinary = rows.loc[rows.in_motif & rows.choice_elapsed_s.notna()]
        consecutive = ordinary.loc[ordinary.choice.eq(ordinary.choice.shift())
                                   & ordinary.outcome.ge(0) & ordinary.outcome.shift().ge(0)
                                   & ordinary.outcome.sub(ordinary.outcome.shift()).abs().eq(2)]
        markers.extend(dict(motif=motif, elapsed_s=row.choice_elapsed_s, event="same_side_reward_pair",
                            label=SIDES[row.choice], trial_id=row.trial_id)
                       for row in consecutive.itertuples(index=False))
    return markers


def _fit_tracks(prepared: PreparedChoices, motif: str, displayed_rows: np.ndarray,
                blocks: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Fit fixed per-track models, returning fit rows and elapsed probability bins."""
    targets = ("outcome",) if motif == "four_outcome" else ("choice", "context")
    fit_rows, tracks = [], []
    trials = prepared.trials
    for width_ms in WIDTHS_MS:
        window = prepared.windows[REGIONAL_WINDOW_S, width_ms]
        rows = window.row_positions
        times_s = numeric_column(trials.choice_time.iloc[rows])
        row_blocks = trials.cur_block.iloc[rows].to_numpy()
        held_out = np.flatnonzero(np.isin(row_blocks, blocks))
        candidates = np.flatnonzero(pd.notna(row_blocks) & ~np.isin(row_blocks, blocks))
        purged = _overlapping_indices(candidates, held_out, times_s, REGIONAL_WINDOW_S)
        train = np.setdiff1d(candidates, purged, assume_unique=True)
        display = np.flatnonzero(np.isin(rows, displayed_rows))
        for region in REGIONS:
            for target in targets:
                index = TARGET_NAMES.index(target)
                labels = prepared.targets.values[rows, index]
                eligible = prepared.targets.valid[rows, index]
                class_ids = tuple(range(len(prepared.targets.class_names[index])))
                supervised = train[eligible[train]]
                bin_start = (len(window.bin_edges_s) - 1) // 2 if target == "outcome" else 0
                edges = window.bin_edges_s[bin_start:]
                identity = dict(motif=motif, width_ms=width_ms, region=region, target=target)
                reason = ""
                if region not in window.counts:
                    reason = prepared.unavailable.get(region, "joint_regional_counts_unavailable")
                elif not len(display) or not np.isin(display, held_out).all():
                    reason = "displayed_choices_require_known_held_out_blocks"
                else:
                    counts = window.counts[region]
                    unit_ids = prepared.unit_ids[region]
                    fitted = fit_readout(counts[supervised, bin_start:] / (width_ms / 1000),
                                         labels[supervised], unit_ids, class_ids, (0,))
                    reason = fitted.reason
                    if not reason:
                        probabilities = predict_readout(counts[display, bin_start:] / (width_ms / 1000),
                                                        unit_ids, fitted)[0]
                        tracks.extend(_track_rows(
                            identity, rows[display], trials.cur_trial.iloc[rows[display]].to_numpy(),
                            times_s[display] - prepared.time_origin_s, edges, probabilities,
                        ))
                fit_rows.append(dict(**identity, n_train=len(supervised), n_held_out=len(held_out),
                                     n_purged=int(eligible[purged].sum()), reason=reason))
    return fit_rows, tracks


def _track_rows(identity: dict, rows: np.ndarray, trial_ids: np.ndarray, times_s: np.ndarray,
                edges: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    """Flatten measured (choice, bin, class) probabilities and elapsed-second bin edges."""
    tracks = []
    for row, trial_id, time, choice_probabilities in zip(rows, trial_ids, times_s, probabilities):
        for left, right, values in zip(edges[:-1], edges[1:], choice_probabilities):
            tracks.append(dict(
                **identity, row_position=row, trial_id=trial_id,
                bin_left_elapsed_s=time + left, bin_right_elapsed_s=time + right,
                **dict(zip(PROBABILITY_COLUMNS, values)),
            ))
    return tracks
