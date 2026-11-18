"""Streamlit reader for completed PFC/HPC session and cohort figures."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt

from src.neural_analysis.presentation_figures.export_figures import GROUPS, read_manifest, read_panels
from src.neural_analysis.presentation_figures.results import discover_runs


def render_presentation_view(st, session) -> None:
    """Display and download saved panels for a ResolvedSession in a Streamlit interface.

    The optional local manifest path also accepts a cohort run. Numerical axes,
    units, replay controls and captions come from the common archive renderers.
    """
    st.header("PFC and HPC presentation figures")
    runs = discover_runs(session.session_root / "analysis_runs")
    override = st.text_input("Session or cohort manifest path", key="presentation_manifest_path")
    if override.strip():
        manifest = Path(override.strip()).expanduser()
    elif runs:
        manifest = st.selectbox("Completed run", runs, format_func=lambda p: p.parent.name)
    else:
        st.info("Enter a completed manifest path or compute the session with presentation_figures.run_session.")
        return
    try:
        data = read_manifest(manifest)
        group = st.selectbox("Figure group", GROUPS[data["kind"]])
        theme = st.radio("Figure theme", ("light", "dark"), horizontal=True)
        static, interactive = read_panels(manifest, group, theme)
    except (OSError, ValueError, KeyError) as error:
        st.error(str(error))
        return
    try:
        options = list(dict.fromkeys([*static, *interactive]))
        if not options:
            st.info("No saved panel is available for this group.")
            return
        panel = st.selectbox("Panel", options)
        if panel in interactive:
            st.plotly_chart(interactive[panel], key=f"presentation_replay_{panel}", width="stretch")
        if panel in static:
            figure = static[panel]
            st.pyplot(figure)
            with BytesIO() as buffer:
                figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
                st.download_button("Download PNG", buffer.getvalue(), file_name=panel + ".png", mime="image/png")
    finally:
        for figure in static.values():
            plt.close(figure)
