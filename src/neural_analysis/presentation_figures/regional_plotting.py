"""Matplotlib views of saved regional decoding and session-level replication."""

from __future__ import annotations

from textwrap import fill

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .decoding import GROUPS, READOUTS, WIDTHS_MS
from .preparation import REGIONS

REGIMES = ("all", "correct_rewarded")
REGIME_LABELS = ("All outcomes", "Correct rewarded")
GROUP_LABELS = ("Overall", "Correct rewarded", "Omission", "Incorrect")
GROUP_COLORS = ("#215b9b", "#137354", "#874d9b", "#b04627")
OUTCOME_NAMES = ("Left rewarded", "Right rewarded", "Left unrewarded", "Right unrewarded")
COHORT_MAIN = ("figure_7_context", "figure_7_choice", "figure_7_outcome", "figure_7_pc_recovery")


def render_regional(fields: dict, meta: dict, theme: str = "light") -> dict[str, Figure]:
    """Render Figures 1-3 from saved row tables with fraction and second units.

    Return named headless figures for the primary 500-ms views, independently
    fitted 100-ms views, and bin, pre/post and outcome-stratified companions.
    Metadata supplies the session label. The caller owns figures and export.
    """
    figures = {}
    session = str(meta.get("session_id", "Session"))
    for width_ms in WIDTHS_MS:
        scores = fields["scores"].loc[fields["scores"].width_ms.eq(width_ms)]
        full = scores.loc[scores.period.eq("full")]
        suffix = "" if width_ms == 500 else "_100ms"
        for number, region in enumerate(REGIONS, start=1):
            stem = f"figure_{number}_{region.lower()}"
            for mode, branch in (("full", ""), ("bins", "_bins"), ("pre_post", "_pre_post")):
                name = f"{stem}{branch}{suffix}"
                figures[name] = _binary_session(fields, scores, region, width_ms, session, mode, theme)
        figures[f"figure_3_outcome{suffix}"] = _confusion_figure(fields, width_ms, session, theme)
        figures[f"figure_3_outcome_bins{suffix}"] = _outcome_bins(scores, width_ms, session, theme)
        figures[f"figure_3_pc{suffix}"] = _pc_session(fields, full, width_ms, session, theme, False)
        figures[f"figure_3_pc_subgroups{suffix}"] = _pc_session(fields, full, width_ms, session, theme, True)
    return figures


def render_regional_cohort(fields: dict, meta: dict, theme: str = "light") -> dict[str, Figure]:
    """Render Figure 7 from ordered roster and per-session saved score tables.

    Dots identify sessions and medians weight available sessions equally.
    Main panels use the saved primary 500-ms endpoint. Companion PC curves
    use the same endpoint. Return a name to headless Figure mapping.
    """
    subject = str(meta.get("subject_id", meta.get("mouse_id", "Mouse")))
    roster = fields["roster"]
    sessions = roster.session_id.astype(str).tolist()
    missing_roster = [
        f"{row.session_id} ({row.reason})" for row in roster.itertuples(index=False)
        if getattr(row, "reason", "")
    ]
    roster_notes = "Sessions shown are " + ", ".join(sessions) + "."
    if missing_roster:
        roster_notes += " Missing scores for " + ", ".join(missing_roster) + "."
    if "scores" not in fields or fields["scores"].empty:
        return {
            name: _empty_cohort_figure(subject, name, roster_notes, theme) for name in COHORT_MAIN
        }
    colors = dict(zip(sessions, colormaps["tab10" if len(sessions) <= 10 else "tab20"](
        np.linspace(0, 1, len(sessions), endpoint=False),
    )))
    scores = fields["scores"]
    full = scores.loc[scores.width_ms.eq(500) & scores.period.eq("full")]
    all_units = full.loc[full.n_pcs.eq(0)]
    figures = {}
    for target in ("context", "choice"):
        figures[f"figure_7_{target}"] = _cohort_binary(
            all_units, target, subject, colors, roster_notes, theme,
        )
    figures["figure_7_outcome"] = _cohort_outcomes(
        all_units, fields["reward_discrimination"], subject, colors, roster_notes, theme,
    )
    recovery = fields["recovery"].loc[fields["recovery"].width_ms.eq(500)]
    figures["figure_7_pc_recovery"] = _cohort_recovery(recovery, subject, colors, roster_notes, theme)
    pc_limit = _pc_display_limit(full, recovery)
    for group in GROUPS:
        key = "figure_7_pc_curves" if group == "overall" else f"figure_7_pc_curves_{group}"
        figures[key] = _cohort_pc_curves(
            full, group, subject, colors, roster_notes, theme, pc_limit,
        )
    for number, session in enumerate(sessions, start=1):
        confusion = fields["confusion"]
        if not confusion.session_id.eq(session).any():
            continue
        session_fields = {key: fields[key].loc[fields[key].session_id.eq(session)]
                          for key in ("scores", "confusion", "reward_discrimination", "delivery_latency")}
        figures[f"figure_7_confusion_session_{number}"] = _confusion_figure(
            session_fields, 500, session, theme,
        )
    return figures


def _layout(rows: int, columns: int, title: str, caption: str, theme: str,
            figsize: tuple[float, float] = (13., 7.2)) -> tuple[Figure, np.ndarray]:
    """Construct a slide-sized headless figure and (rows, columns) Axes array."""
    if theme not in ("light", "dark"):
        raise ValueError(f"Unsupported regional figure theme: {theme}")
    background, foreground = ("white", "#19232e") if theme == "light" else ("#151c26", "#f0f3f7")
    figure = Figure(figsize=figsize, facecolor=background)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    figure.subplots_adjust(left=.09, right=.97, top=.84, bottom=.18, wspace=.42, hspace=.67)
    figure.suptitle(title, fontsize=23, fontweight="semibold", color=foreground, y=.96)
    figure.text(.06, .034, fill(caption, 155), fontsize=10, color=foreground, va="bottom")
    for axis in axes.flat:
        axis.set_facecolor(background)
        axis.tick_params(colors=foreground, labelsize=10)
        for spine in axis.spines.values():
            spine.set_color(foreground)
        axis.xaxis.label.set_color(foreground)
        axis.yaxis.label.set_color(foreground)
        axis.title.set_color(foreground)
    return figure, axes


def _accuracy_axis(axis, chance: float | None, ylabel: bool = False) -> None:
    """Style dimensionless balanced accuracy as a fixed 0-100% axis."""
    axis.set_ylim(0, 1)
    axis.set_yticks([0, .25, .5, .75, 1])
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if chance is not None:
        axis.axhline(chance, color="#8b929b", lw=1.1, linestyle="--", zorder=0)
    if ylabel:
        axis.set_ylabel("Balanced accuracy", fontsize=11)


def _metric_text(value: float, digits: int = 0) -> str:
    """Format a dimensionless accuracy fraction for a plot label or caption."""
    return f"{value:.{digits}%}" if np.isfinite(value) else "unavailable"


def _color(axis) -> str:
    """Read the semantic foreground color shared by the axes labels."""
    return axis.xaxis.label.get_color()


def _reasons(rows: pd.DataFrame) -> list[str]:
    """Return sorted distinct nonblank unavailability reasons from score rows."""
    return sorted({str(value).replace("_", " ") for value in rows.reason.dropna() if value})


def _short_reason(axis, rows: pd.DataFrame) -> None:
    """Annotate distinct reasons in the current plot panel where scores are missing."""
    reasons = _reasons(rows)
    if reasons:
        axis.text(.01, .98, fill(", ".join(reasons), 42), transform=axis.transAxes,
                  va="top", fontsize=7.7, color=_color(axis))


def _binary_session(fields: dict, scores: pd.DataFrame, region: str, width_ms: int,
                    session: str, mode: str, theme: str) -> Figure:
    """Render one region's 2x2 classification panels at one fitted bin width."""
    title_suffix = {"full": "Context and choice", "bins": "Bin-level accuracy",
                    "pre_post": "Before and after choice"}[mode]
    unit_count = fields["units"].loc[fields["units"].region.eq(region), "n_units"].iloc[0]
    caption = f"{session} | {region}, {unit_count} good units | {width_ms} ms bins aligned to choice."
    if mode == "full":
        caption += " Mean probabilities across [-1, +1) s yield one prediction per held-out choice."
        caption += " R/L are eligible right/left labels. Dashed line shows binary chance."
        caption += _context_blocks_note(fields, region, width_ms)
    elif mode == "bins":
        caption += " Each bin scores one prediction per held-out choice at its choice-relative center."
    else:
        caption += " Each held-out choice is scored once per 1 s half using the same full-window fit."
    caption += _mapping_note(fields, region, width_ms)
    figure, axes = _layout(2, 2, f"{region} | {title_suffix}", caption, theme)
    region_scores = scores.loc[scores.region.eq(region) & scores.n_pcs.eq(0)]
    for regime_index, regime in enumerate(REGIMES):
        for target_index, target in enumerate(("context", "choice")):
            axis = axes[regime_index, target_index]
            axis.set_gid(f"{regime}:{target}")
            name = str(target).capitalize()
            axis.set_title(f"{name} | {REGIME_LABELS[regime_index]}", fontsize=14, fontweight="semibold")
            _accuracy_axis(axis, .5, target_index == 0)
            readout = region_scores.loc[region_scores.regime.eq(regime) & region_scores.target.eq(target)]
            if mode == "full":
                _binary_dots(axis, readout.loc[readout.period.eq("full")], show_counts=regime_index == 1)
            elif mode == "bins":
                rows = readout.loc[readout.period.eq("bin")]
                for group, label, color in zip(GROUPS, GROUP_LABELS, GROUP_COLORS):
                    values = rows.loc[rows.group.eq(group)].sort_values("bin_index")
                    axis.plot(values.bin_center_s, values.balanced_accuracy, marker=".", color=color,
                              label=label, gid=group)
                axis.axvline(0, color="#8b929b", lw=.9)
                axis.set_xlim(-1, 1)
                if regime_index == 1:
                    axis.set_xlabel("Time from choice (s)")
                _short_reason(axis, rows)
            else:
                rows = readout.loc[readout.period.isin(("pre", "post"))]
                for group, label, color in zip(GROUPS, GROUP_LABELS, GROUP_COLORS):
                    values = rows.loc[rows.group.eq(group)].set_index("period").reindex(("pre", "post"))
                    axis.plot((0, 1), values.balanced_accuracy, marker="o", color=color, label=label)
                axis.set_xticks((0, 1), ("Pre-choice", "Post-choice"))
                axis.set_xlim(-.15, 1.15)
                _short_reason(axis, rows)
    if mode in ("bins", "pre_post"):
        legend = figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                               bbox_to_anchor=(.5, .905), ncols=4, frameon=False, fontsize=9)
        for item in legend.get_texts():
            item.set_color(_color(axes[0, 0]))
    return figure


def _mapping_note(fields: dict, region: str, width_ms: int) -> str:
    """Return caption text when held-out correct-rewarded fits share one axis."""
    mappings = fields.get("transfer_mapping")
    folds = fields.get("folds")
    if mappings is None or folds is None:
        return ""
    selected = folds.loc[folds.region.eq(region) & folds.width_ms.eq(width_ms)
                         & folds.regime.eq("correct_rewarded") & folds.n_test.gt(0)]
    if selected.empty or selected.reason.ne("").any():
        return ""
    matched = mappings.loc[mappings.region.eq(region) & mappings.width_ms.eq(width_ms)
                           & mappings.fold_id.isin(selected.fold_id)]
    if matched.empty:
        return ""
    if matched.mapping.eq("same").all():
        return " The correct-rewarded context and choice fits share the same readout."
    if matched.mapping.eq("reversed").all():
        return " The correct-rewarded context and choice fits share a readout with reversed labels."
    return ""


def _context_blocks_note(fields: dict, region: str, width_ms: int) -> str:
    """Describe the range of independent training block counts by context label."""
    folds = fields.get("folds")
    if folds is None:
        return ""
    relevant = folds.loc[folds.region.eq(region) & folds.width_ms.eq(width_ms)
                         & folds.target.eq("context") & folds.n_test.gt(0)]
    notes = []
    for regime, label in zip(REGIMES, REGIME_LABELS):
        rows = relevant.loc[relevant.regime.eq(regime)]
        if rows.empty:
            continue
        ranges = []
        for column in ("blocks_0", "blocks_1"):
            minimum, maximum = rows[column].min(), rows[column].max()
            ranges.append(str(minimum) if minimum == maximum else f"{minimum}-{maximum}")
        notes.append(f"{label} ({ranges[0]} right, {ranges[1]} left)")
    return (" Independent context training blocks per fold were " + " and ".join(notes) + ".") if notes else ""


def _binary_dots(axis, rows: pd.DataFrame, *, show_counts: bool) -> None:
    """Draw held-out group dots and optional shared eligible right/left counts."""
    rows = rows.set_index("group").reindex(GROUPS)
    labels = []
    for index, (group, label, color) in enumerate(zip(GROUPS, GROUP_LABELS, GROUP_COLORS)):
        row = rows.loc[group]
        if np.isfinite(row.balanced_accuracy):
            axis.scatter(index, row.balanced_accuracy, color=color, s=58, zorder=3, gid=group, clip_on=False)
        count_label = f"{label}\nR {int(row.count_0)} / L {int(row.count_1)}" if show_counts else label
        if row.n_predicted != row.n_eligible:
            count_label += f"\nPredicted {row.n_predicted}/{row.n_eligible}"
        labels.append(count_label)
    axis.set_xticks(range(len(GROUPS)), labels)
    axis.set_xlim(-.55, 3.55)
    _short_reason(axis, rows)


def _confusion_figure(fields: dict, width_ms: int, session: str, theme: str) -> Figure:
    """Plot row-normalized confusion and within-side reward accuracy for both regions."""
    matrices = fields["confusion"]
    matrices = matrices.loc[matrices.width_ms.eq(width_ms)]
    rewards = fields["reward_discrimination"]
    rewards = rewards.loc[rewards.width_ms.eq(width_ms)]
    scores = fields["scores"]
    scores = scores.loc[scores.width_ms.eq(width_ms) & scores.target.eq("outcome")
                        & scores.period.eq("full") & scores.n_pcs.eq(0)]
    caption = (
        f"{session} | {width_ms} ms bins in [0, +1) s after choice. Mean class probabilities yield "
        "one prediction per held-out choice. Row fractions share a 0-100% color scale."
    )
    delivery = fields.get("delivery_latency")
    if delivery is not None and not delivery.empty and delivery.n_recorded.iloc[0]:
        row = delivery.iloc[0]
        caption += (f" Trial tables record {int(row.n_under_100ms)} of {int(row.n_recorded)} deliveries "
                    f"within 100 ms ({int(row.n_rewarded)} ordinary rewarded choices).")
    figure, axes = _layout(1, 2, "Choice side and reward outcome", caption, theme, (13., 7.8))
    figure.subplots_adjust(left=.18, right=.91, bottom=.31, top=.84, wspace=.6)
    figure.text(.015, .585, "Actual outcome", rotation=90, va="center",
                color=_color(axes[0, 0]), fontsize=11)
    cmap = colormaps["Blues"].copy()
    cmap.set_bad("#cad0d7")
    images = []
    for region, axis in zip(REGIONS, axes[0]):
        axis.set_gid(region)
        regional = matrices.loc[matrices.region.eq(region)].sort_values(["true_class", "predicted_class"])
        fractions = np.full((4, 4), np.nan) if regional.empty else regional.fraction.to_numpy().reshape(4, 4)
        image = axis.imshow(fractions, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
        images.append(image)
        counts = regional.groupby("true_class").row_count.first().reindex(range(4), fill_value=0)
        for truth, prediction in np.ndindex(fractions.shape):
            value = fractions[truth, prediction]
            label = _metric_text(value) if np.isfinite(value) else "n/a"
            axis.text(prediction, truth, label, ha="center", va="center",
                      color="white" if value > .55 else "#19232e", fontsize=12)
        axis.set_yticks(range(4), [f"{label} (n={count})" for label, count in zip(OUTCOME_NAMES, counts)])
        axis.set_xticks(range(4), [label.replace(" ", "\n", 1) for label in OUTCOME_NAMES])
        axis.set_xlabel("Predicted outcome")
        score = scores.loc[scores.region.eq(region)].iloc[0]
        axis.set_title(f"{region} | Balanced accuracy {_metric_text(score.balanced_accuracy, 1)}",
                       fontsize=14, fontweight="semibold", pad=13)
        messages = []
        for side, row in rewards.loc[rewards.region.eq(region)].set_index("side").iterrows():
            text = (f"{side.capitalize()} reward {_metric_text(row.balanced_accuracy, 1)} "
                    f"(rewarded {row.n_rewarded}, unrewarded {row.n_unrewarded})")
            if row.reason:
                text += f" | {str(row.reason).replace('_', ' ')}"
            messages.append(text)
        if score.reason:
            messages.append(str(score.reason).replace("_", " "))
        axis.text(.5, -.26, "\n".join(messages), transform=axis.transAxes, ha="center", va="top",
                  fontsize=9, color=_color(axis))
    color_axis = figure.add_axes((.935, .325, .014, .49))
    bar = figure.colorbar(images[0], cax=color_axis, format=PercentFormatter(xmax=1, decimals=0))
    bar.ax.tick_params(colors=_color(axes[0, 0]), labelsize=10)
    return figure


def _outcome_bins(scores: pd.DataFrame, width_ms: int, session: str, theme: str) -> Figure:
    """Render separate region-level four-class held-out bin accuracy curves."""
    caption = (f"{session} | {width_ms} ms bins in [0, +1) s after choice. "
               "Balanced accuracy gives each of the four side-by-reward classes equal weight.")
    figure, axes = _layout(1, 2, "Outcome accuracy by post-choice bin", caption, theme)
    outcomes = scores.loc[scores.target.eq("outcome") & scores.period.eq("bin") & scores.n_pcs.eq(0)]
    for region, axis in zip(REGIONS, axes[0]):
        rows = outcomes.loc[outcomes.region.eq(region)].sort_values("bin_index")
        axis.set_gid(region)
        axis.set_title(f"{region} | {int(rows.n_eligible.max())} eligible choices", fontsize=14)
        _accuracy_axis(axis, .25, region == "PFC")
        axis.plot(rows.bin_center_s, rows.balanced_accuracy, marker="o", color=GROUP_COLORS[0])
        axis.set_xlabel("Time from choice (s)")
        axis.set_xlim(0, 1)
        _short_reason(axis, rows)
    return figure


def _pc_title(target: str, regime: str) -> str:
    """Return the two-line title for a regional decoding readout."""
    return f"{target.capitalize()}\n{REGIME_LABELS[REGIMES.index(regime)]}"


def _pc_session(fields: dict, scores: pd.DataFrame, width_ms: int, session: str,
                theme: str, subgroups: bool) -> Figure:
    """Render regional PC-prefix curves and matched all-unit references."""
    readouts = READOUTS[:-1] if subgroups else READOUTS
    recovery = fields["recovery"].loc[fields["recovery"].width_ms.eq(width_ms)]
    pc_limit = _pc_display_limit(scores, recovery)
    caption = (f"{session} | {width_ms} ms, choice-aligned [-1, +1) s for context and choice, "
               "[0, +1) s for outcome. Solid curves show supported training-fold PC counts. "
               "Dashed lines show matched all-unit scores. A ring marks sustained recovery within 5 points.")
    title = "Decoding by PC count" + (" | Outcome groups" if subgroups else "")
    figure, axes = _layout(2, len(readouts), title, caption, theme, (15., 7.4))
    if subgroups:
        figure.subplots_adjust(top=.77)
    else:
        figure.subplots_adjust(bottom=.24)
    for region_index, region in enumerate(REGIONS):
        for index, (target, regime) in enumerate(readouts):
            axis = axes[region_index, index]
            axis.set_gid(f"{region}:{regime}:{target}")
            _pc_axis(axis, target, region if not index else "", pc_limit)
            if region_index == 0:
                axis.set_title(_pc_title(target, regime), fontsize=12, fontweight="semibold")
            elif subgroups:
                axis.set_xlabel("PC count")
            rows = scores.loc[scores.region.eq(region) & scores.target.eq(target) & scores.regime.eq(regime)]
            matches = recovery.loc[recovery.region.eq(region) & recovery.target.eq(target)
                                   & recovery.regime.eq(regime)]
            leading = (range(len(GROUPS)) if subgroups else range(1))
            for group_index in leading:
                group, color = GROUPS[group_index], GROUP_COLORS[group_index]
                series = rows.loc[rows.group.eq(group)].set_index("n_pcs")
                fitted = matches.loc[matches.group.eq(group)].iloc[0]
                _draw_pc(axis, series, fitted, color, group)
            if not subgroups:
                fitted = matches.loc[matches.group.eq("overall")].iloc[0]
                axis.text(.5, -.16, _recovery_text(fitted), transform=axis.transAxes,
                          fontsize=9, va="top", ha="center", color=_color(axis))
    if subgroups:
        _group_legend(figure, axes[0, 0], len(GROUPS))
    else:
        figure.text(.5, .115, "PC count", ha="center", color=_color(axes[0, 0]), fontsize=11)
    return figure


def _pc_display_limit(scores: pd.DataFrame, recovery: pd.DataFrame) -> int:
    """Return a shared count axis, with a ten-PC fallback for legacy saved tables."""
    requested = recovery.get("requested_max_pc", pd.Series(dtype=int)).dropna()
    observed = scores.n_pcs.dropna()
    return int(max(10, max(requested, default=0), max(observed, default=0)))


def _pc_axis(axis, target: str, region: str, pc_limit: int) -> None:
    """Set bounded axes for positive PC counts and dimensionless accuracy."""
    _accuracy_axis(axis, .25 if target == "outcome" else .5)
    axis.set_xlim(.7, pc_limit + .3)
    ticks = sorted({1, *range(5, pc_limit, 5), pc_limit})
    axis.set_xticks(ticks)
    if region:
        axis.set_ylabel(f"{region}\nBalanced accuracy", fontsize=11)


def _draw_pc(axis, series: pd.DataFrame, recovery, color, label: str) -> None:
    """Plot one score curve, one all-unit reference and an optional recovery mark."""
    count = series.index[series.index > 0].to_numpy(dtype=int)
    if len(count):
        curve = series.reindex(np.arange(count.min(), count.max() + 1))
        axis.plot(curve.index, curve.balanced_accuracy, marker=".", color=color, lw=1.4, gid=label,
                  clip_on=False)
    if np.isfinite(recovery.all_unit_accuracy):
        axis.axhline(recovery.all_unit_accuracy, color=color, linestyle="--", lw=1.15, alpha=.7)
    if recovery.status == "reached":
        k = int(recovery.recovery_count)
        axis.scatter(k, series.loc[k, "balanced_accuracy"], s=80, facecolors="none",
                     edgecolors=color, linewidths=1.6, zorder=3, clip_on=False)


def _recovery_text(row) -> str:
    """Label attained, unattained and unavailable PC recovery with reference score."""
    reference = _metric_text(row.all_unit_accuracy)
    if row.status == "reached":
        return f"{int(row.recovery_count)} PCs through {row.max_supported_pc}\nAll units {reference}"
    if row.status == "unreached":
        return f"Unreached through {row.max_supported_pc}\nAll units {reference}"
    reason = str(row.reason).replace("_", " ")
    return f"{fill(reason, 23)}\nAll units {reference}"


def _group_legend(figure: Figure, axis, count: int) -> None:
    """Print the fixed outcome-group legend used by regional backup curves."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], label=label, color=color, marker=".")
               for label, color in zip(GROUP_LABELS[:count], GROUP_COLORS[:count])]
    legend = figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91),
                           ncols=count, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(_color(axis))


def _cohort_binary(scores: pd.DataFrame, target: str, subject: str, colors: dict,
                   roster_notes: str, theme: str) -> Figure:
    """Render named paired session dots for one target over regions and groups."""
    caption = (f"{subject} | 500 ms bins over [-1, +1) s. Lines pair the training regimes within a session. "
               "Diamonds show descriptive session medians. Counts show contributing sessions. " + roster_notes)
    figure, axes = _layout(2, 4, f"{target.capitalize()} decoding across sessions", caption, theme, (15., 7.6))
    for region_index, region in enumerate(REGIONS):
        for group_index, (group, label) in enumerate(zip(GROUPS, GROUP_LABELS)):
            axis = axes[region_index, group_index]
            axis.set_gid(f"{region}:{group}")
            axis.set_title(label, fontsize=13, fontweight="semibold")
            _accuracy_axis(axis, .5)
            if group_index == 0:
                axis.set_ylabel(f"{region}\nBalanced accuracy", fontsize=11)
            rows = scores.loc[scores.region.eq(region) & scores.target.eq(target) & scores.group.eq(group)]
            _session_dots(axis, rows, "regime", REGIMES, ("All", "Correct\nrewarded"), colors)
    _session_legend(figure, axes[0, 0], colors)
    return figure


def _cohort_outcomes(scores: pd.DataFrame, rewards: pd.DataFrame, subject: str, colors: dict,
                     roster_notes: str, theme: str) -> Figure:
    """Plot four-class and within-side reward accuracy per session and region."""
    caption = (f"{subject} | 500 ms bins over [0, +1) s. Four-state chance is 25%. Within-side reward "
               "chance is 50%. Diamonds show session medians. " + roster_notes)
    figure, axes = _layout(1, 2, "Outcome and within-side reward across sessions", caption, theme)
    keys = ("four", "left", "right")
    labels = ("Four outcomes", "Left reward", "Right reward")
    primary = scores.loc[scores.target.eq("outcome") & scores.group.eq("overall")].assign(metric="four")
    rows = pd.concat([primary, rewards.loc[rewards.width_ms.eq(500)].assign(metric=rewards.side)],
                     ignore_index=True)
    for region, axis in zip(REGIONS, axes[0]):
        axis.set_gid(region)
        axis.set_title(region, fontsize=14, fontweight="semibold")
        _accuracy_axis(axis, None, region == "PFC")
        axis.hlines(.25, -.35, .35, linestyle="--", color="#8b929b")
        axis.hlines(.5, .65, 2.35, linestyle="--", color="#8b929b")
        _session_dots(axis, rows.loc[rows.region.eq(region)], "metric", keys, labels, colors, paired=False)
    _session_legend(figure, axes[0, 0], colors)
    return figure


def _session_dots(axis, rows: pd.DataFrame, key: str, order: tuple[str, ...],
                  labels: tuple[str, ...], colors: dict, paired: bool = True) -> None:
    """Draw one series per named session, gaps for missing scores, and medians."""
    x = np.arange(len(order))
    by_point = []
    unavailable = []
    offsets = np.linspace(-.15, .15, max(3, len(colors) + 2))[1:-1]
    for (session, color), offset in zip(colors.items(), offsets):
        selected = rows.loc[rows.session_id.astype(str).eq(session)].set_index(key).reindex(order)
        values = selected.balanced_accuracy.to_numpy(dtype=float)
        by_point.append(values)
        if np.isfinite(values).any():
            axis.plot(x + offset, values, marker="o", markersize=6, color=color, alpha=.9,
                      linestyle="-" if paired else "none", gid=session, lw=1, clip_on=False)
        reasons = _reasons(selected)
        if reasons:
            unavailable.append(f"{session} ({', '.join(reasons)})")
    values = np.asarray(by_point)
    for ix, column in enumerate(values.T):
        observed = column[np.isfinite(column)]
        if len(observed):
            axis.scatter(ix, np.median(observed), marker="D", s=32, zorder=3,
                         color=_color(axis), gid="session_median", clip_on=False)
    counts = np.isfinite(values).sum(axis=0)
    for ix, count in enumerate(counts):
        axis.text(ix, -.2, f"{count}/{len(colors)}", transform=axis.get_xaxis_transform(),
                  ha="center", va="top", fontsize=9, color=_color(axis))
    axis.set_xticks(x, labels)
    axis.set_xlim(-.5, len(x) - .5)
    if unavailable:
        axis.text(.02, .95, fill(", ".join(unavailable), 38), transform=axis.transAxes,
                  va="top", fontsize=7.5, color=_color(axis))


def _session_legend(figure: Figure, axis, colors: dict) -> None:
    """Draw a shared named-session legend with its stable session order."""
    from matplotlib.lines import Line2D

    if not colors:
        return
    handles = [Line2D([], [], color=color, label=session, marker="o", lw=1.1)
               for session, color in colors.items()]
    legend = figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .905),
                           ncols=min(5, len(handles)), frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(_color(axis))


def _cohort_recovery(recovery: pd.DataFrame, subject: str, colors: dict,
                     roster_notes: str, theme: str) -> Figure:
    """Show per-readout overall sustained recovery and matched all-unit accuracy."""
    caption = (f"{subject} | Primary 500 ms estimates. Recovery requires all supported larger PC "
               "counts to stay within five points of the all-unit score. " + roster_notes)
    figure, axes = _layout(2, len(READOUTS), "PC recovery across sessions", caption, theme, (15., 7.7))
    figure.subplots_adjust(top=.77)
    for region_index, region in enumerate(REGIONS):
        for index, (target, regime) in enumerate(READOUTS):
            axis = axes[region_index, index]
            axis.set_gid(f"{region}:{regime}:{target}")
            axis.axis("off")
            if region_index == 0:
                axis.set_title(_pc_title(target, regime), fontsize=12, fontweight="semibold")
            if index == 0:
                axis.text(-.22, .5, region, transform=axis.transAxes, va="center", rotation=90,
                          fontweight="semibold", color=_color(axis))
            rows = recovery.loc[recovery.region.eq(region) & recovery.target.eq(target)
                                & recovery.regime.eq(regime) & recovery.group.eq("overall")]
            rows = rows.set_index(rows.session_id.astype(str))
            for ix, (session, color) in enumerate(colors.items()):
                top = 1 - ix / max(4, len(colors))
                if session in rows.index:
                    row = rows.loc[session]
                    if row.status == "reached":
                        label = (f"{int(row.recovery_count)} PCs through {row.max_supported_pc}\n"
                                 f"{_metric_text(row.all_unit_accuracy)} all units")
                    elif row.status == "unreached":
                        label = f"Unreached through {row.max_supported_pc}\n{_metric_text(row.all_unit_accuracy)} all units"
                    else:
                        label = fill(str(row.reason).replace("_", " "), 24)
                        if np.isfinite(row.all_unit_accuracy):
                            label += f"\n{_metric_text(row.all_unit_accuracy)} all units"
                else:
                    label = "No saved score"
                axis.scatter(.015, top - .035, color=color, s=23, transform=axis.transAxes, clip_on=False)
                axis.text(.09, top, label, transform=axis.transAxes, va="top", fontsize=9,
                          color=_color(axis))
    _session_legend(figure, axes[0, 0], colors)
    return figure


def _cohort_pc_curves(scores: pd.DataFrame, group: str, subject: str, colors: dict,
                      roster_notes: str, theme: str, pc_limit: int) -> Figure:
    """Draw thin per-session PC curves and equal-weight medians on supported counts."""
    label = GROUP_LABELS[GROUPS.index(group)]
    caption = (f"{subject} | 500 ms bins. Thin curves show each named session through its available "
               "PC counts. The dashed curve shows the median of available sessions at each count. " + roster_notes)
    readouts = READOUTS if group == "overall" else READOUTS[:-1]
    figure, axes = _layout(2, len(readouts), f"PC accuracy across sessions | {label}",
                           caption, theme, (15., 7.6))
    figure.subplots_adjust(top=.77)
    for region_index, region in enumerate(REGIONS):
        for index, (target, regime) in enumerate(readouts):
            axis = axes[region_index, index]
            axis.set_gid(f"{region}:{regime}:{target}")
            _pc_axis(axis, target, region if not index else "", pc_limit)
            if region_index == 0:
                axis.set_title(_pc_title(target, regime), fontsize=12, fontweight="semibold")
            else:
                axis.set_xlabel("PC count")
            rows = scores.loc[scores.region.eq(region) & scores.target.eq(target) & scores.regime.eq(regime)
                              & scores.group.eq(group) & scores.n_pcs.gt(0)]
            for session, color in colors.items():
                curve = rows.loc[rows.session_id.astype(str).eq(session)].set_index("n_pcs")
                if not curve.empty:
                    grid = range(int(curve.index.min()), int(curve.index.max()) + 1)
                    axis.plot(grid, curve.reindex(grid).balanced_accuracy, color=color, marker=".", lw=1,
                              clip_on=False)
            if not rows.empty:
                median = rows.groupby("n_pcs").balanced_accuracy.median()
                grid = range(int(median.index.min()), int(median.index.max()) + 1)
                axis.plot(grid, median.reindex(grid), color=_color(axis), lw=1.5,
                          linestyle="--", gid="session_median", clip_on=False)
            else:
                axis.text(.5, .55, "No supported PC score", ha="center", transform=axis.transAxes,
                          fontsize=8, color=_color(axis))
    _session_legend(figure, axes[0, 0], colors)
    return figure


def _empty_cohort_figure(subject: str, name: str, roster: str, theme: str) -> Figure:
    """Render a roster and its unavailable statuses when no scores were saved."""
    titles = ("Context decoding", "Choice decoding", "Choice side and reward", "PC recovery")
    title = titles[COHORT_MAIN.index(name)]
    figure, axes = _layout(1, 1, f"{title} | {subject}", roster, theme)
    axes[0, 0].axis("off")
    axes[0, 0].text(.5, .55, "No saved regional decoding scores", ha="center",
                    fontsize=15, color=_color(axes[0, 0]), transform=axes[0, 0].transAxes)
    return figure
