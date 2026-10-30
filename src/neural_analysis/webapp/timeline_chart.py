"""Keep one browser chart alive while Streamlit updates timeline display data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import streamlit.components.v2 as components


def render_timeline_chart(figure: go.Figure, key: str) -> None:
    """Render a prepared Plotly timeline in a component with stable browser identity.

    figure carries elapsed-second x axes, PC-score heatmaps, and a uirevision
    identifying the session/navigation request. key is a stable per-session
    widget key. Changes to PC ranks/data keep that revision and preserve the
    browser's time range; explicit navigation changes it. Returns None.
    No analysis arrays or figure properties are modified.
    """
    chart = components.component(
        "session_timeline_chart",
        js=_component_javascript(),
        isolate_styles=False,
    )
    chart(
        key=key,
        data={"figure_json": figure.to_json()},
        width="stretch",
        height="content",
    )


# Renderer assets are constant within a process; figure data is supplied separately.
@lru_cache(maxsize=1)
def _component_javascript() -> str:
    """Return the installed Plotly JS bundle plus the local component renderer.

    No inputs. The returned JavaScript module string is cached in memory;
    it requires no CDN, generated files, or additional runtime dependencies.
    """
    renderer = Path(__file__).with_suffix(".js").read_text(encoding="ascii")
    return get_plotlyjs() + "\n" + renderer
