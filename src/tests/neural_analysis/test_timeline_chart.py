"""Persistent chart renderer contracts, independent of experimental data."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import Mock

import plotly.graph_objects as go
import pytest


def test_timeline_renderer_keeps_key_and_passes_original_figure(monkeypatch):
    """Figure data changes must not change component registration or mounting identity."""
    chart = importlib.import_module("src.neural_analysis.webapp.timeline_chart")
    mount = Mock()
    register = Mock(return_value=mount)
    monkeypatch.setattr(chart.components, "component", register)
    figure = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    figure.update_layout(height=600, uirevision="session:1")
    before = figure.to_json()
    chart.render_timeline_chart(figure, key="session_chart")
    first_registration = register.call_args
    first_mount = mount.call_args
    figure.data[0].y = [5, 6]
    chart.render_timeline_chart(figure, key="session_chart")
    assert register.call_args == first_registration
    assert first_registration.kwargs["isolate_styles"] is False
    assert first_mount.kwargs["key"] == mount.call_args.kwargs["key"] == "session_chart"
    assert first_mount.kwargs["height"] == "content"
    assert json.loads(first_mount.kwargs["data"]["figure_json"]) == json.loads(before)
    assert mount.call_args.kwargs["data"]["figure_json"] == figure.to_json()
    assert not mount.call_args.kwargs.get("on_viewport_change")


def test_browser_renderer_updates_existing_plot_and_cleans_up_only_on_unmount():
    """Exercise the JS lifecycle with Node's built-in assertions and minimal browser stubs.

    NODE_BINARY optionally names Node when it is not on PATH. The live browser
    check separately verifies Plotly's real pan/range-slider behavior.
    """
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the frontend lifecycle unit test")
    source = Path(__file__).parents[2] / "neural_analysis/webapp/timeline_chart.js"
    script = r"""
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source = readFileSync(process.argv[1], 'utf8');
const {default: render} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const children = [];
const parent = {
    appendChild(child) { children.push(child); },
    querySelector() { return children[0] ?? null; },
};
let fullscreenRequests = 0;
globalThis.document = {createElement() { return {
    style: {}, dataset: {}, setAttribute() {},
    addEventListener() {}, removeEventListener() {},
    requestFullscreen() { fullscreenRequests += 1; return Promise.resolve(); },
}; }};
const observers = [];
globalThis.ResizeObserver = class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(element) { this.element = element; }
    disconnect() { this.disconnected = true; }
};
const calls = [];
let purges = 0;
globalThis.window = {Plotly: {
    async react(element, data, layout, config) { calls.push({element, data, layout, config}); },
    purge() { purges += 1; },
    Plots: {resize() {}},
}};
const figure = {data: [{y: [1, 2]}], layout: {height: 600, uirevision: 'session:1'}};
await render({parentElement: parent, data: {figure_json: JSON.stringify(figure)}});
figure.data = [{y: [2]}];
const cleanup = await render({parentElement: parent, data: {figure_json: JSON.stringify(figure)}});
assert.equal(children.length, 1);
assert.equal(calls.length, 2);
assert.equal(calls[0].element, calls[1].element);
assert.equal(calls[0].config, calls[1].config);
assert.equal(calls[0].layout.uirevision, calls[1].layout.uirevision);
assert.equal(calls[1].config.scrollZoom, false);
assert.equal(calls[1].config.showSendToCloud, false);
assert.deepEqual(calls[1].config.modeBarButtons.slice(0, 3), [
    ['toImage'], ['zoom2d', 'pan2d'],
    ['zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
]);
const fullscreen = calls[1].config.modeBarButtons[3][0];
assert.equal(fullscreen.name, 'Fullscreen');
fullscreen.click(calls[1].element);
assert.equal(fullscreenRequests, 1);
assert.equal(purges, 0);
assert.equal(observers.length, 1);
// Adding/removing matched subplot axes must not propagate an old server range.
children[0]._fullLayout = {xaxis: {range: [35, 95]}};
figure.layout.xaxis = {range: [0, 60]};
figure.layout.xaxis2 = {range: [0, 60], matches: 'x'};
await render({parentElement: parent, data: {figure_json: JSON.stringify(figure)}});
assert.deepEqual(calls.at(-1).layout.xaxis.range, [35, 95]);
assert.deepEqual(calls.at(-1).layout.xaxis2.range, [35, 95]);
figure.layout.xaxis3 = {range: [0, 60], matches: 'x'};
await render({parentElement: parent, data: {figure_json: JSON.stringify(figure)}});
assert.deepEqual(calls.at(-1).layout.xaxis3.range, [35, 95]);
figure.layout.uirevision = 'session:2';
await render({parentElement: parent, data: {figure_json: JSON.stringify(figure)}});
assert.equal(calls.at(-1).layout.uirevision, 'session:2');
assert.deepEqual(calls.at(-1).layout.xaxis.range, [0, 60]);
cleanup();
assert.equal(purges, 1);
assert.equal(observers[0].disconnected, true);
"""
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script, str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
