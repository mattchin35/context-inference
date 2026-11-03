// Keep chart elements separate for each mounted component instance.
const timelineCharts = new WeakMap();

// Stable callbacks keep Plotly.react from treating toolbar configuration as a new plot.
const chartConfig = {
    scrollZoom: false,
    displaylogo: false,
    responsive: true,
    showSendToCloud: false,
    modeBarButtons: [
        ["toImage"],
        ["zoom2d", "pan2d"],
        ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        [{
            name: "Fullscreen",
            icon: {
                width: 24, height: 24,
                path: "M3 9V3h6v2H5v4zm12-6h6v6h-2V5h-4zM3 15h2v4h4v2H3zm16 0h2v6h-6v-2h4z",
            },
            click: (chart) => {
                if (document.fullscreenElement === chart) {
                    document.exitFullscreen();
                } else {
                    chart.requestFullscreen();
                }
            },
        }],
    ],
};

/**
 * Update a timeline without replacing its plot element or user-selected time range.
 * @param {object} component Streamlit v2 context: parentElement is an HTMLElement;
 * data.figure_json is a Plotly JSON string with elapsed-second x axes.
 * @returns {Promise<Function>} Cleanup for unmounting, not for display-data changes.
 */
export default async function renderTimeline({parentElement, data}) {
    let state = timelineCharts.get(parentElement);
    if (!state) {
        const chart = document.createElement("div");
        chart.style.width = "100%";
        parentElement.appendChild(chart);
        state = {chart, width: 0};
        state.onFullscreenChange = () => {
            window.Plotly.relayout(chart, {
                height: document.fullscreenElement === chart ? window.innerHeight : state.height,
            });
        };
        chart.addEventListener("fullscreenchange", state.onFullscreenChange);
        state.observer = new ResizeObserver(([entry]) => {
            const width = entry.contentRect.width;
            if (width > 0 && width !== state.width) {
                state.width = width;
                if (chart.data) {
                    window.Plotly.Plots.resize(chart);
                }
            }
        });
        timelineCharts.set(parentElement, state);
    }

    const figure = JSON.parse(data.figure_json);
    state.height = figure.layout.height;
    // New matched axes must inherit the actual browser range, not older server fields.
    const rangeUpdate = {};
    const declaredTimeAxes = figure.layout.meta?.time_axes;
    const timeAxes = Array.isArray(declaredTimeAxes)
        ? declaredTimeAxes
        : Object.keys(figure.layout).filter((name) => /^xaxis\d*$/.test(name));
    const primaryTimeAxis = figure.layout.meta?.primary_time_axis ?? "xaxis";
    const currentRange = state.chart._fullLayout?.[primaryTimeAxis]?.range;
    if (currentRange && state.revision === figure.layout.uirevision) {
        // Bounded payloads may not cover browser-only pan; use the requested
        // server interval when the whole previous viewport is unavailable.
        const fitsDestination = timeAxes.every((name) => {
            const axis = figure.layout[name];
            return !axis || ((axis.minallowed == null || currentRange[0] >= axis.minallowed)
                && (axis.maxallowed == null || currentRange[1] <= axis.maxallowed));
        });
        const range = fitsDestination ? currentRange : figure.layout[primaryTimeAxis]?.range;
        for (const name of timeAxes) {
            const axis = figure.layout[name];
            if (axis && range) {
                axis.range = [...range];
                rangeUpdate[`${name}.range`] = [...range];
            }
        }
    }
    state.revision = figure.layout.uirevision;
    // Plotly preserves UI ranges when uirevision stays fixed, including rank changes.
    await window.Plotly.react(state.chart, figure.data, figure.layout, chartConfig);
    if (Object.keys(rangeUpdate).length) {
        await window.Plotly.relayout(state.chart, rangeUpdate);
    }
    state.observer.observe(parentElement);

    return () => {
        state.observer.disconnect();
        state.chart.removeEventListener("fullscreenchange", state.onFullscreenChange);
        window.Plotly.purge(state.chart);
        timelineCharts.delete(parentElement);
    };
}
