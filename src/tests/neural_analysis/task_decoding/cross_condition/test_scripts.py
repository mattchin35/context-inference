"""Synthetic single-session orchestration: verified PCA reuse, saved identity and explicit preparation."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population import run_session_timeline as timeline_runner
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population.session_timeline import SessionTimelineResult, fit_timeline_activity, prepare_timeline_choices
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.neural_analysis.task_decoding.cross_condition import modeling, results, run_session


def patch_dependency(monkeypatch, owner, name, replacement):
    """Patch one real dependency API and an existing direct runner alias, independent of import style."""
    monkeypatch.setattr(owner, name, replacement)
    if hasattr(run_session, name):
        monkeypatch.setattr(run_session, name, replacement)


def forbidden(*args, **kwargs):
    """Fail if a forbidden expensive/read-write operation is invoked; no return value."""
    raise AssertionError("This operation must not be called on this runner path")


def requested_identity(roles):
    """Return a deterministic synthetic metadata-only configuration ID for a role/probe dictionary."""
    return "requested:" + ",".join(sorted(roles))


@pytest.fixture
def runner_context(transfer_data, request):
    """Return synthetic metadata/session paths and real saved PCA products, without raw neural files.

    Behavior has J=20 chronological rows, N=16 ordinary choices and no block
    column. Seed932 generates Poisson counts(W,B,U), then real unwhitened PCA;
    PFC/HPC have independent W=16/13 and K=3/2. Width is 500/100 ms, choices
    use UTC Unix seconds. An intentionally missing event file must not be read.
    """
    trials, _, _, base = transfer_data
    trials = trials.drop(columns="cur_block")
    root, metadata = base.session_root, base.session_metadata_path
    width = getattr(request, "param", 500)
    if width != 500:
        (root / "cross_condition_config.json").write_text(json.dumps({"bin_width_ms": width}), encoding="ascii")
    trials.to_csv(base.augmented_trial_path, index=False)
    session = resolve_session_metadata(load_session_metadata(metadata), metadata)
    session.behavior.trial_table_file.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(session.behavior.trial_table_file, index=False)
    assert not session.behavior.event_table_file.exists()
    origin, choices = prepare_timeline_choices(trials)
    generator = np.random.default_rng(932)
    regions = {}
    for role, probe, positions, units in (
        ("PFC", "ProbeA", np.arange(16), 3),
        ("HPC", "ProbeB", np.flatnonzero(~choices.trial_id.isin([0, 6, 19]).to_numpy()), 2),
    ):
        counts = generator.poisson(4, size=(len(positions), 4000 // width, units))
        regions[role] = fit_timeline_activity(counts, tuple(f"{probe}:{index}" for index in range(units)),
                                            positions, role, probe, width / 1000)
    events = pd.DataFrame(columns=["Time", "Event", "Note", "elapsed_s"])
    paths, timelines = {}, {}
    for name, selected in (("shared", regions), ("pfc", {"PFC": regions["PFC"]}),
                           ("hpc", {"HPC": regions["HPC"]}), ("prepared", regions)):
        unavailable = {role: "not in this regional archive" for role in ("PFC", "HPC") if role not in selected}
        timeline = SessionTimelineResult(session.session_id, origin, choices, events, selected, unavailable,
                                         {"configuration_id": "source-" + name, "bin_width_s": width / 1000})
        path = root / "analysis_runs" / f"session_timeline_pca_{name}" / "results.npz"
        timeline_io.save_timeline_result(path, timeline)
        paths[name], timelines[path] = path, timeline
    return SimpleNamespace(metadata=metadata, root=root, base=base, session=session, trials=trials,
                           choices=choices, paths=paths, timelines=timelines, width=width)


@pytest.fixture
def runner_services(runner_context, monkeypatch):
    """Replace only discovery/preparation seams; fitting, source loading and publication remain real.

    Mutable mappings select exact/compatible paths or ordinary exceptions.
    Call logs contain only paths/scalars/role dictionaries, not neural copies.
    The compatibility API represents its already-verified scientific contract.
    """
    context = runner_context
    state = SimpleNamespace(
        exact={requested_identity({"PFC": "ProbeA", "HPC": "ProbeB"}): context.paths["shared"]},
        compatible={}, prepared=context.paths["prepared"], preparation_error=None, load_errors={},
        config_calls=[], exact_calls=[], compatible_calls=[], preparation_calls=[], load_calls=[], events=[],
        original_find=timeline_runner.find_saved_timeline,
    )
    original_load = timeline_io.load_timeline_result

    def configuration(session, roles, width_s, *, population_quality, missing_label_policy):
        """Return metadata-only config dict for supported second width and explicit unit-quality policies."""
        state.config_calls.append((dict(roles), width_s, population_quality, missing_label_policy))
        return {"configuration_id": requested_identity(roles), "probe_roles": dict(roles), "bin_width_s": width_s,
                "selection": {role: {"unit_ids": [probe + ":1"], "channels": [1]} for role, probe in roles.items()}}

    def exact(root, identity):
        """Return a selected exact/registered Path or None without reading arrays."""
        state.exact_calls.append(identity)
        state.events.append("exact")
        return state.exact.get(identity)

    def compatible(session, role, probe, width_s, *, population_quality, missing_label_policy):
        """Return the already-verified regional Path/None, retaining explicit policies and order."""
        state.compatible_calls.append((role, probe, width_s, population_quality, missing_label_policy))
        state.events.append("compatible:" + role)
        value = state.compatible.get(role)
        if isinstance(value, Exception):
            raise value
        return value

    def prepare(metadata_path, probe_roles=None, bin_width_s=.5, **kwargs):
        """Record the requested missing-role map and return one existing synthetic prepared NPZ receipt."""
        state.preparation_calls.append((Path(metadata_path), dict(probe_roles), bin_width_s, kwargs))
        assert kwargs["population_quality"] == "good_mua" and kwargs["missing_label_policy"] == "mua"
        if state.preparation_error is not None:
            raise state.preparation_error
        return timeline_runner.TimelineRunReport("prepared", state.prepared, "source-prepared", .01)

    def load(path):
        """Load one real saved timeline, counting fitting-stage reads independently of identity metadata reads."""
        path = Path(path)
        state.load_calls.append(path)
        state.events.append("load")
        if path in state.load_errors:
            raise state.load_errors[path]
        return original_load(path)

    patch_dependency(monkeypatch, timeline_io, "timeline_configuration", configuration)
    patch_dependency(monkeypatch, timeline_runner, "find_saved_timeline", exact)
    patch_dependency(monkeypatch, timeline_io, "find_compatible_regional_timeline", compatible)
    patch_dependency(monkeypatch, timeline_runner, "run_session_timeline", prepare)
    patch_dependency(monkeypatch, timeline_io, "load_timeline_result", load)
    patch_dependency(monkeypatch, timeline_io, "load_timeline_behavior", forbidden)
    return state


@pytest.fixture
def science_services(runner_services, monkeypatch):
    """Bypass only run-document snapshots while the real P5b batch file is still pending.

    The dedicated document integration test below uses the actual helper and
    remains a joint P5a/P5b requirement; it is never conditionally skipped.
    """
    def no_snapshots(directory):
        """Accept a new run-directory Path and create no snapshots in focused science tests."""
        return None

    monkeypatch.setattr(run_session, "_snapshot_sources", no_snapshots)
    return runner_services


def test_dry_run_reports_global_cells_and_pending_verification_without_neural_work(
    runner_context, science_services, monkeypatch,
):
    """Unresolved regional reuse is pending, not absent; planning is global behavioral counts before coverage."""
    context, services = runner_context, science_services
    services.exact.clear()
    before = {path.relative_to(context.root): path.read_bytes() for path in context.root.rglob("*") if path.is_file()}
    patch_dependency(monkeypatch, timeline_io, "find_compatible_regional_timeline", forbidden)
    patch_dependency(monkeypatch, timeline_io, "load_timeline_result", forbidden)
    patch_dependency(monkeypatch, timeline_io, "load_timeline_population", forbidden)
    patch_dependency(monkeypatch, timeline_runner, "run_session_timeline", forbidden)
    patch_dependency(monkeypatch, modeling, "fit_region_transfers", forbidden)
    patch_dependency(monkeypatch, results, "build_transfer_metadata", forbidden)
    report = run_session.prepare_cross_condition_generalization(context.metadata, dry_run=True)
    assert report.status == "dry-run" and report.result_path is None and report.elapsed_s >= 0
    details = report.details
    assert details["choice_count"] == 16 and details["max_model_count"] == 192
    assert details["window_bin_counts"] == {"full": 4, "before": 2, "after": 2}
    assert details["dark_state_count"] == 1 and details["conflict_count"] == 2
    assert details["conflict_trial_ids"] == [13, 14]
    row, = [row for row in details["cell_counts"] if row["split_id"] == "context_correct_to_incorrect"]
    assert row["source"] == [2, 3] and row["complementary_test"] == [2, 3]
    for source in details["source_status"].values():
        assert source["status"] == "compatibility_pending" and source["result_path"] is None
        assert "verif" in source["reason"].lower()
    json.dumps(details)
    after = {path.relative_to(context.root): path.read_bytes() for path in context.root.rglob("*") if path.is_file()}
    assert after == before and not services.preparation_calls and not services.load_calls


def test_metadata_only_call_prepares_all_192_models_and_loads_shared_source_once(runner_context, science_services, monkeypatch):
    """One explicit metadata call fits all H/G/S per region, without event reads or PCA recomputation."""
    calls = []
    original = modeling.fit_readout

    def count_readout(matrix, labels, weights, family):
        """Count finite M-row categorical fits and delegate the existing K-PC readout unchanged."""
        calls.append(1)
        return original(matrix, labels, weights, family)

    monkeypatch.setattr(modeling, "fit_readout", count_readout)
    report = run_session.prepare_cross_condition_generalization(runner_context.metadata)
    packet = results.load_transfer_result(report.result_path)
    assert report.status == "prepared" and report.elapsed_s >= 0 and report.identity == packet.meta["identity"]
    assert len(calls) == packet.meta["fit_count"] == 192
    assert set(packet.regions) == {"PFC", "HPC"} and packet.unavailable_regions == {}
    assert all(np.all(model.fit_status == "ok") for model in packet.regions.values())
    assert Counter(science_services.load_calls) == {runner_context.paths["shared"]: 1}
    assert not science_services.compatible_calls and not science_services.preparation_calls
    assert packet.meta["region_sources"]["PFC"] == packet.meta["region_sources"]["HPC"]
    assert packet.meta["runtime_s"] >= 0 and packet.meta["peak_process_rss_bytes"] > 0
    assert re.match(r"cross_condition_generalization_\d{8}_\d{6}", report.result_path.parent.name)
    assert runner_context.root in report.result_path.parents


def test_exact_saved_bundle_reuse_precedes_all_score_loading_and_fitting(runner_context, science_services, monkeypatch):
    """After source identity selection, a validated existing result returns before score loads/fits/writes."""
    first = run_session.prepare_cross_condition_generalization(runner_context.metadata)
    before = {path.relative_to(runner_context.root) for path in runner_context.root.rglob("*")}
    patch_dependency(monkeypatch, timeline_io, "load_timeline_result", forbidden)
    patch_dependency(monkeypatch, modeling, "fit_region_transfers", forbidden)
    patch_dependency(monkeypatch, timeline_runner, "run_session_timeline", forbidden)
    second = run_session.prepare_cross_condition_generalization(runner_context.metadata)
    assert second.status == "reused" and second.result_path == first.result_path and second.identity == first.identity
    assert {path.relative_to(runner_context.root) for path in runner_context.root.rglob("*")} == before


def test_registered_exact_source_uses_existing_receipt_before_regional_lookup(runner_context, science_services, monkeypatch):
    """Exercise the existing exact finder's registered fallback, without adding a new registration mechanism."""
    registered_calls = []

    def registered(root, identity):
        """Return an already-registered shared PCA Path for the requested small configuration ID."""
        registered_calls.append(identity)
        return runner_context.paths["shared"]

    patch_dependency(monkeypatch, timeline_runner, "find_saved_timeline", science_services.original_find)
    patch_dependency(monkeypatch, timeline_io, "find_registered_timeline", registered)
    report = run_session.prepare_cross_condition_generalization(runner_context.metadata)
    assert report.status == "prepared" and len(registered_calls) == 1
    assert not science_services.compatible_calls and not science_services.preparation_calls


def test_separate_compatible_regional_sources_use_explicit_policies_and_load_each_once(runner_context, science_services):
    """Unrelated whole-session differences allow verified regional reuse, with unchanged source identities."""
    context, services = runner_context, science_services
    services.exact.clear()
    services.compatible.update(PFC=context.paths["pfc"], HPC=context.paths["hpc"])
    payload = json.loads(context.metadata.read_text())
    payload["behavior"]["events"] = "behavior/different_display_events.csv"
    payload["analysis"]["ppc"]["preview_shuffles"] += 1
    context.metadata.write_text(json.dumps(payload), encoding="ascii")
    report = run_session.prepare_cross_condition_generalization(context.metadata)
    packet = results.load_transfer_result(report.result_path)
    assert set(packet.regions) == {"PFC", "HPC"} and len(packet.meta["source_pcas"]) == 2
    assert Counter(services.load_calls) == {context.paths["pfc"]: 1, context.paths["hpc"]: 1}
    assert {call[0] for call in services.compatible_calls} == {"PFC", "HPC"}
    assert all(call[3:] == ("good_mua", "mua") for call in services.compatible_calls)
    assert all(call[2:] == ("good_mua", "mua") for call in services.config_calls)
    assert services.events.index("exact") < next(index for index, value in enumerate(services.events) if value.startswith("compatible:"))
    assert not services.preparation_calls


def test_matching_bundle_after_regional_verification_skips_fitting_stage_loads(runner_context, science_services, monkeypatch):
    """Regional lookup may read arrays while verifying; matching identity still avoids all subsequent loads/fits."""
    context, services = runner_context, science_services
    first = run_session.prepare_cross_condition_generalization(context.metadata)
    services.exact.clear()
    original_load = timeline_io.load_timeline_result
    verification_reads = []

    def verify(session, role, probe, width_s, *, population_quality, missing_label_policy):
        """Represent the existing finder's allowed verification read, then return the same basis Path."""
        verification_reads.append(role)
        original_load(context.paths["shared"])
        return context.paths["shared"]

    patch_dependency(monkeypatch, timeline_io, "find_compatible_regional_timeline", verify)
    patch_dependency(monkeypatch, timeline_io, "load_timeline_result", forbidden)
    patch_dependency(monkeypatch, modeling, "fit_region_transfers", forbidden)
    second = run_session.prepare_cross_condition_generalization(context.metadata)
    assert set(verification_reads) == {"PFC", "HPC"}
    assert second.status == "reused" and second.result_path == first.result_path


def test_only_missing_region_is_prepared_once_and_reused_basis_is_preserved(runner_context, science_services):
    """Keep verified PFC while one runner call prepares HPC; do not replace PFC from the returned archive."""
    context, services = runner_context, science_services
    services.exact.clear()
    services.compatible["PFC"] = context.paths["pfc"]
    report = run_session.prepare_cross_condition_generalization(context.metadata)
    packet = results.load_transfer_result(report.result_path)
    assert len(services.preparation_calls) == 1
    metadata, roles, width_s, kwargs = services.preparation_calls[0]
    assert metadata == context.metadata and roles == {"HPC": "ProbeB"} and width_s == .5
    assert kwargs["population_quality"] == "good_mua" and kwargs["missing_label_policy"] == "mua"
    assert packet.meta["region_sources"]["PFC"] == context.paths["pfc"].relative_to(context.root).as_posix()
    assert packet.meta["region_sources"]["HPC"] == context.paths["prepared"].relative_to(context.root).as_posix()
    assert Counter(services.load_calls) == {context.paths["pfc"]: 1, context.paths["prepared"]: 1}


@pytest.mark.parametrize("both", [False, True])
def test_known_selected_source_load_failures_save_available_regions_and_reasons(runner_context, science_services, both):
    """A failed known source stays referenced, without fabricated regional arrays; both failures still publish."""
    context, services = runner_context, science_services
    services.exact.clear()
    services.compatible.update(PFC=context.paths["pfc"], HPC=context.paths["hpc"])
    services.load_errors[context.paths["hpc"]] = OSError("HPC source array read failed")
    if both:
        services.load_errors[context.paths["pfc"]] = OSError("PFC source array read failed")
    report = run_session.prepare_cross_condition_generalization(context.metadata)
    packet = results.load_transfer_result(report.result_path)
    assert report.status == "prepared" and "HPC" in packet.unavailable_regions
    assert "source array read failed" in packet.unavailable_regions["HPC"]
    assert set(packet.regions) == (set() if both else {"PFC"})
    assert set(packet.meta["region_sources"]) == {"PFC", "HPC"}
    if both:
        assert set(packet.unavailable_regions) == {"PFC", "HPC"} and packet.meta["fit_count"] == 0
    assert not services.preparation_calls


@pytest.mark.parametrize("both", [False, True])
def test_missing_source_preparation_failure_is_reportable_per_region(runner_context, science_services, both):
    """One attempted missing-role preparation fails; retain verified regions or publish both unavailability reasons."""
    context, services = runner_context, science_services
    services.exact.clear()
    if not both:
        services.compatible["PFC"] = context.paths["pfc"]
    services.preparation_error = OSError("synthetic PCA preparation failed")
    report = run_session.prepare_cross_condition_generalization(context.metadata)
    packet = results.load_transfer_result(report.result_path)
    assert len(services.preparation_calls) == 1
    assert set(packet.regions) == (set() if both else {"PFC"})
    assert set(packet.unavailable_regions) == ({"PFC", "HPC"} if both else {"HPC"})
    assert all("preparation failed" in reason for reason in packet.unavailable_regions.values())
    assert set(packet.meta["region_sources"]) == (set() if both else {"PFC"})


def test_saved_pca_choice_misalignment_is_not_silently_fitted(runner_context, science_services, monkeypatch):
    """Saved PCA choice times must align to augmented chronology, even when current small behavior validates."""
    context = runner_context
    timeline = context.timelines[context.paths["shared"]]
    wrong_choices = timeline.choices.copy(deep=True)
    wrong_choices.loc[0, "choice_time"] += .01

    def misaligned(path):
        """Return one otherwise valid regional basis with mismatched UTC choice time(N,)."""
        return replace(timeline, choices=wrong_choices)

    patch_dependency(monkeypatch, timeline_io, "load_timeline_result", misaligned)
    patch_dependency(monkeypatch, modeling, "fit_region_transfers", forbidden)
    report = run_session.prepare_cross_condition_generalization(context.metadata)
    packet = results.load_transfer_result(report.result_path)
    assert packet.regions == {} and set(packet.unavailable_regions) == {"PFC", "HPC"}
    assert all("choice_time" in reason or "align" in reason.lower() for reason in packet.unavailable_regions.values())


def test_invalid_small_context_stops_before_neural_preparation_or_loading(runner_context, science_services):
    """Malformed ordinary labels are input errors, rather than expensive PCA work or unavailable-model bundles."""
    trials = runner_context.trials.copy(deep=True)
    trials.loc[trials.index[0], "state_int"] = np.nan
    trials.to_csv(runner_context.base.augmented_trial_path, index=False)
    with pytest.raises(ValueError, match="state_int"):
        run_session.prepare_cross_condition_generalization(runner_context.metadata)
    assert not science_services.preparation_calls and not science_services.load_calls


@pytest.mark.parametrize("runner_context", [500, 100], indirect=True)
def test_rerun_uses_saved_pca_but_new_dated_bundle_at_selected_width(runner_context, science_services):
    """Rerun creates a new transfer result, preserving basis and old bytes; widths select matching source grids."""
    first = run_session.prepare_cross_condition_generalization(runner_context.metadata)
    original = first.result_path.read_bytes()
    second = run_session.prepare_cross_condition_generalization(runner_context.metadata, rerun=True)
    first_packet, second_packet = results.load_transfer_result(first.result_path), results.load_transfer_result(second.result_path)
    assert second.status == "prepared" and second.result_path != first.result_path
    assert second.identity == first.identity and first.result_path.read_bytes() == original
    assert second_packet.meta["source_pcas"] == first_packet.meta["source_pcas"]
    assert second_packet.meta["effective_config"]["bin_width_ms"] == runner_context.width
    assert second_packet.inputs.window_bin_masks.sum(axis=1).tolist() == [2000 // runner_context.width, 1000 // runner_context.width, 1000 // runner_context.width]
    assert not science_services.preparation_calls


def test_cli_forwards_only_supported_metadata_controls_and_run_flags(runner_context, monkeypatch, capsys):
    """CLI remains thin: metadata/config Paths, integer ms, rerun/dryrun; no role or model-setting overrides."""
    calls = []

    def capture(metadata, **kwargs):
        """Capture scalar/Path controls and return a small dry-run TransferRunReport, with no arrays."""
        calls.append((metadata, kwargs))
        return run_session.TransferRunReport("dry-run", None, "planned", 0., {})

    monkeypatch.setattr(run_session, "prepare_cross_condition_generalization", capture)
    config = runner_context.root / "controls.json"
    assert run_session.main(["--session-metadata", str(runner_context.metadata), "--config", str(config),
                             "--bin-width-ms", "100", "--rerun", "--dry-run"]) == 0
    metadata, kwargs = calls[0]
    assert Path(metadata) == runner_context.metadata
    assert Path(kwargs["config_path"]) == config and kwargs["bin_width_ms"] == 100
    assert kwargs["rerun"] is True and kwargs["dry_run"] is True
    assert "dry-run" in capsys.readouterr().out


def test_run_documents_copy_real_main_batch_and_all_scientific_sources(runner_context, runner_services):
    """Joint P5a/P5b integration: actual scripts/source snapshots, timing log and scientific summary precede publication.

    This test intentionally requires the real run_batch.py. It must remain
    visible as an unmet joint-gate dependency until P5b implements that file.
    """
    report = run_session.prepare_cross_condition_generalization(runner_context.metadata)
    packet = results.load_transfer_result(report.result_path)
    directory = report.result_path.parent
    source = Path(run_session.__file__).resolve()
    repository = source.parents[4]
    assert (directory / "run_session.py").read_bytes() == source.read_bytes()
    assert (directory / "run_batch.py").read_bytes() == source.with_name("run_batch.py").read_bytes()
    for name in packet.meta["scientific_sources"]:
        assert (directory / "source_snapshots" / name).read_bytes() == (repository / name).read_bytes()
    summary, log = (directory / "summary.md").read_text(), (directory / "run.log").read_text()
    assert "Goal:" in summary and packet.meta["session_id"] in summary
    assert "run_session.py" in summary and "run_batch.py" in summary
    assert "in-sample" in summary.lower() and "complementary" in summary.lower() and "session-wide" in summary.lower()
    assert all(token in log.lower() for token in ("source", "fit", "save", "192"))
