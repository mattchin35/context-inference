"""Plain numerical artifacts, source identity and current-table export contracts."""

from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def trial_table():
    """Return dimensionless history (5 rows), fresh features and stale owned fields."""
    return pd.DataFrame({"cur_trial": [3, 4, 7, 9, 10], "cur_block": [1, 1, 2, 2, 2],
                         "action": [0, 1, 1, 0, 1], "reward": [1, 0, 1, 0, 1],
                         "fresh_feature": [10, 20, 30, 40, 50],
                         "inherited_block_model": ["block"] * 5,
                         "trial_policy_old_candidate": [99] * 5}, index=[8, 4, 4, 2, 1])


def prepared_result(table, model_ids=None, smoothing=True, params=None):
    """DataFrame[T,C] and options yield history/candidates/TaskParams/result records."""
    from src.behavior_analysis import trial_policy_analysis as analysis
    from src.behavior_analysis import trial_policy_models as models
    from src.behavior_analysis.gather_trial_features import TaskParams

    params = TaskParams() if params is None else params
    history = models.prepare_trial_policy_history(table, session_id="source")
    candidates = models.resolve_trial_policy_candidates(models.DEFAULT_MODEL_IDS if model_ids is None else model_ids)
    scores, valid = models.replay_trial_policy_candidates(history, candidates, params)
    result = analysis.compute_trial_policy_comparison(
        history, scores, valid, tuple(c.model_id for c in candidates),
        choice_parameters=models.effective_trial_policy_choice_parameters(candidates))
    if smoothing:
        result = analysis.smooth_trial_policy_comparison(result)
    return history, candidates, params, result


def metadata_for(table, **options):
    """Return numerical result and metadata for DataFrame[T,C], scalar settings."""
    from src.behavior_analysis import trial_policy_results as results

    model_ids = options.pop("model_ids", None)
    params = options.pop("params", None)
    history, candidates, params, result = prepared_result(
        table, model_ids=model_ids, params=params, smoothing=options.get("smoothing", True))
    metadata = results.build_trial_policy_metadata(history, candidates, params,
                                                   parameter_source="explicit TaskParams", **options)
    return result, metadata


def test_named_npz_roundtrip_and_completion_marker(trial_table, tmp_path):
    """Trusted NPZ stores named nonobject arrays plus meta and exact scalar provenance."""
    from src.behavior_analysis import trial_policy_results as results

    original, metadata = metadata_for(trial_table)
    path = results.save_trial_policy_results(original, processed_data_path=tmp_path, metadata=metadata)
    assert path.relative_to(tmp_path).parts[:2] == ("trial_policy_hmm", results.ANALYSIS_VERSION)
    assert path.parent.name and (path.parent / "complete.json").exists()
    restored, saved_meta = results.load_trial_policy_results(path)
    with np.load(path, allow_pickle=True) as stored:
        assert isinstance(stored["meta"].item(), dict)
        for name in stored.files:
            if name != "meta":
                assert not stored[name].dtype.hasobject
        assert {"p_right", "observed_action", "independent_best_model_mask", "hmm_smoothed_best_model_mask", "meta"} <= set(stored.files)
    for field in fields(original):
        expected, actual = getattr(original, field.name), getattr(restored, field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_equal(actual, expected)
        else:
            assert actual == expected
    assert saved_meta["signature"] == metadata["signature"]
    assert saved_meta["input_digest"] == metadata["input_digest"]
    assert saved_meta["analysis_version"] == results.ANALYSIS_VERSION
    assert saved_meta["parameters"]["opposite_action_error_prob"] == .1
    assert saved_meta["candidate_parameters"]["forgetting_q_learning"]["reward_update_rate"] == pytest.approx(.3)
    assert saved_meta["parameter_source"] == "explicit TaskParams"
    assert saved_meta["random_seed"] is None
    assert saved_meta["package_versions"]["dynamax"]
    assert saved_meta["units"] and saved_meta["axis_conventions"] and saved_meta["analysis_source_digest"]
    assert results.find_trial_policy_cache(tmp_path, metadata["signature"]) == path


def test_checked_attachment_preserves_current_columns_index_and_clears_stale(trial_table):
    """Only owned fields refresh; IDs/features/block assignments retain current values."""
    from src.behavior_analysis import trial_policy_results as results

    result, meta = metadata_for(trial_table, smoothing=False)
    fresh = trial_table.assign(fresh_feature=[11, 22, 33, 44, 55], trial_policy_hmm_smoothed_state=99)
    original = fresh.copy(deep=True)
    attached = results.attach_trial_policy_outputs(fresh, result, metadata=meta)
    pd.testing.assert_frame_equal(fresh, original)
    pd.testing.assert_index_equal(attached.index, fresh.index)
    assert attached["cur_trial"].tolist() == fresh["cur_trial"].tolist()
    assert attached["fresh_feature"].tolist() == [11, 22, 33, 44, 55]
    assert attached["inherited_block_model"].tolist() == ["block"] * 5
    assert "trial_policy_old_candidate" not in attached
    assert not any(column.startswith("trial_policy_hmm_") for column in attached)
    assert pd.isna(attached["trial_policy_independent_state"].iloc[0])
    assert attached["trial_policy_independent_tie"].iloc[0] == 1
    assert "trial_policy_p_right_forgetting_q_learning" in attached
    assert pd.isna(attached["trial_policy_greedy_choice_forgetting_q_learning"].iloc[0])


def test_hmm_attachment_preserves_explicit_array_field_names(trial_table):
    """HMM model columns retain their full owned names and nullable greedy ties."""
    from src.behavior_analysis import trial_policy_results as results

    result, meta = metadata_for(trial_table)
    attached = results.attach_trial_policy_outputs(trial_table, result, metadata=meta)
    assert "trial_policy_hmm_smoothed_probs_forgetting_q_learning" in attached
    assert "trial_policy_hmm_smoothed_best_model_mask_forgetting_q_learning" in attached
    np.testing.assert_array_equal(attached["trial_policy_hmm_smoothed_probs_forgetting_q_learning"], result.hmm_smoothed_probs[:, 0])


def test_source_digest_reads_the_actual_computational_sources(monkeypatch):
    """Source identity covers existing behavior modules as well as main orchestration."""
    from src.behavior_analysis import trial_policy_results as results

    original = Path.read_bytes
    reads = []

    def track_read(path):
        """Path input returns its bytes and records the exact existing source path."""
        reads.append(path.resolve())
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", track_read)
    assert results._analysis_source_digest()
    source_directory = Path(results.__file__).parent
    required = {source_directory / name for name in ("trial_policy_models.py", "trial_policy_analysis.py", "trial_policy_results.py",
                                                     "trial_features.py", "project_utils.py", "gather_trial_features.py", "exemplar_hmm_modeling.py")}
    required.add(source_directory.parent / "main.py")
    assert required <= set(reads)


@pytest.mark.parametrize("change", ["reward", "action", "flag", "id", "row_order", "schema"])
def test_attachment_rejects_changed_history_even_when_lengths_match(trial_table, change):
    """History/schema/row changes cannot receive stale cached assignments."""
    from src.behavior_analysis import trial_policy_results as results

    result, meta = metadata_for(trial_table)
    changed = trial_table.copy()
    if change == "reward":
        changed.iloc[0, changed.columns.get_loc("reward")] = 0
    elif change == "action":
        changed.iloc[0, changed.columns.get_loc("action")] = 1
    elif change == "flag":
        changed["experimenter_reward_given"] = [0, 1, 0, 0, 0]
    elif change == "id":
        changed.iloc[0, changed.columns.get_loc("cur_trial")] = 999
    elif change == "row_order":
        changed = changed.iloc[::-1]
    else:
        changed["experimenter_reward_given"] = 0
    with pytest.raises(ValueError, match="identity|digest|schema"):
        results.attach_trial_policy_outputs(changed, result, metadata=meta)


@pytest.mark.parametrize("change", ["reward", "crop", "parameter", "order", "source", "error", "stay", "smoothing",
                                   "absent_to_zero", "legacy_to_current", "second_alias"])
def test_parameter_input_and_source_changes_invalidate_cache(trial_table, monkeypatch, change):
    """Each cache identity input affects signature; unrelated features do not."""
    from src.behavior_analysis import trial_policy_results as results
    from src.behavior_analysis import trial_policy_models as models
    from src.behavior_analysis.gather_trial_features import TaskParams

    table = trial_table.assign(give_reward=0) if change in ("legacy_to_current", "second_alias") else trial_table
    _, before = metadata_for(table)
    options = {}
    if change == "reward":
        table = table.assign(reward=[0, 0, 1, 0, 1])
    elif change == "crop":
        table = table.iloc[1:]
    elif change == "parameter":
        options["params"] = TaskParams(FQL_decay=.6)
    elif change == "order":
        options["model_ids"] = models.DEFAULT_MODEL_IDS[::-1]
    elif change == "source":
        monkeypatch.setattr(results, "_analysis_source_digest", lambda: "changed-source")
    elif change == "error":
        options["error_prob"] = .2
    elif change == "stay":
        options["stay_prob"] = .8
    elif change == "smoothing":
        options["smoothing"] = False
    elif change == "absent_to_zero":
        table = table.assign(experimenter_reward_given=0)
    elif change == "legacy_to_current":
        table = table.rename(columns={"give_reward": "experimenter_reward_given"})
    elif change == "second_alias":
        table = table.assign(experimenter_reward_given=0)
    _, after = metadata_for(table, **options)
    assert before["signature"] != after["signature"]


def test_unrelated_columns_and_unselected_parameters_do_not_invalidate(trial_table):
    """Fresh derived columns, column order and irrelevant parameters preserve identity."""
    from src.behavior_analysis.gather_trial_features import TaskParams

    _, before = metadata_for(trial_table, model_ids=("win_stay_lose_switch",))
    _, after = metadata_for(trial_table[list(reversed(trial_table.columns))].assign(fresh_feature=99),
                            model_ids=("win_stay_lose_switch",), params=TaskParams(FQL_decay=.1))
    assert before["signature"] == after["signature"]


def test_new_saves_preserve_old_arrays_and_ignore_incomplete_directories(trial_table, tmp_path):
    """Repeated/rerun saves are immutable and a completion marker controls reuse."""
    from src.behavior_analysis import trial_policy_results as results

    result, meta = metadata_for(trial_table)
    first = results.save_trial_policy_results(result, processed_data_path=tmp_path, metadata=meta)
    before = first.read_bytes()
    second = results.save_trial_policy_results(result, processed_data_path=tmp_path, metadata=meta)
    assert second != first and first.read_bytes() == before
    assert results.find_trial_policy_cache(tmp_path, meta["signature"]) == second
    (second.parent / "complete.json").unlink()
    assert results.find_trial_policy_cache(tmp_path, meta["signature"]) == first


def test_export_refreshes_current_table_without_mutating_cached_npz(trial_table, tmp_path):
    """CSV snapshots and logs reference immutable arrays and archive current exports."""
    from src.behavior_analysis import trial_policy_results as results

    result, meta = metadata_for(trial_table)
    path = results.save_trial_policy_results(result, processed_data_path=tmp_path, metadata=meta)
    numerical_bytes = path.read_bytes()
    attached = results.attach_trial_policy_outputs(trial_table.assign(fresh_feature=77), result, metadata=meta)
    current = tmp_path / "source_augmented_trials.csv"
    current.write_text("old,export\n1,2\n", encoding="ascii")
    old_bytes = current.read_bytes()
    export_directory = path.parent.parent / "fresh_export"
    exported = results.write_trial_policy_export(attached, export_directory=export_directory,
                                                 numerical_path=path, metadata=meta, current_csv_path=current)
    assert exported == export_directory / "trial_policy_export.csv"
    assert pd.read_csv(current)["fresh_feature"].tolist() == [77] * 5
    assert path.read_bytes() == numerical_bytes
    archives = list((tmp_path / "trial_policy_hmm" / "archives").rglob("source_augmented_trials.archived.csv"))
    assert len(archives) == 1 and archives[0].read_bytes() == old_bytes
    assert list(tmp_path.rglob("*augmented_trials.csv")) == [current]
    assert not (export_directory / "results.npz").exists()
    for filename in ("run.log", "metadata.json", "summary.md", "main.py", "batch.py"):
        assert (export_directory / filename).exists()
    record = json.loads((export_directory / "metadata.json").read_text())
    assert record["numerical_artifact"] == str(path)
    assert record["export_artifact"] == str(exported)


@pytest.mark.parametrize("legacy_archive", [False, True])
def test_trial_exports_archive_identical_content_once_and_preserve_legacy_archives(tmp_path, legacy_archive):
    """Path/bool fixtures retain identical CSV bytes once across repeated exports."""
    from hashlib import sha256
    from src.behavior_analysis import trial_policy_results as results

    current = tmp_path / "source_augmented_trials.csv"
    table = pd.DataFrame({"action": [0, 1], "reward": [1, 0]})
    table.to_csv(current, index=False)
    content = current.read_bytes()
    archive_root = tmp_path / "trial_policy_hmm" / "archives"
    if legacy_archive:
        legacy = archive_root / sha256(content).hexdigest() / current.name
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(content)
    for run_index in range(2):
        snapshot = results.write_trial_policy_export(
            table, export_directory=tmp_path / f"export_{run_index}",
            numerical_path=None, metadata={}, current_csv_path=current,
        )
        assert snapshot.name == "trial_policy_export.csv"
    archives = list(archive_root.rglob("*.csv"))
    assert len(archives) == 1 and archives[0].read_bytes() == content
    assert archives[0].name == (current.name if legacy_archive else "source_augmented_trials.archived.csv")
    assert current.read_bytes() == content


def test_unsmoothed_npz_omits_hmm_fields(trial_table, tmp_path):
    """Disabled smoothing persists no synthetic HMM arrays."""
    from src.behavior_analysis import trial_policy_results as results

    result, meta = metadata_for(trial_table, smoothing=False)
    path = results.save_trial_policy_results(result, processed_data_path=tmp_path, metadata=meta)
    with np.load(path, allow_pickle=True) as saved:
        assert not any(name.startswith("hmm_") for name in saved.files)
        assert "transition_matrix" not in saved.files
    restored, _ = results.load_trial_policy_results(path)
    assert restored.hmm_smoothed_probs is None


def test_diagnostic_messages_are_concise_and_clean_sessions_are_quiet(trial_table):
    """One session warning per missing-data condition includes count and original IDs."""
    from src.behavior_analysis import trial_policy_results as results
    from src.behavior_analysis.trial_policy_models import prepare_trial_policy_history

    clean = prepare_trial_policy_history(trial_table, session_id="source")
    assert results.trial_policy_warnings(clean) == []
    dirty = trial_table.assign(reward=[None, 0, None, 0, 1], experimenter_reward_given=[0, None, 0, 0, 0])
    prepared = prepare_trial_policy_history(dirty, session_id="source")
    messages = results.trial_policy_warnings(prepared)
    assert len(messages) == 2
    assert any("missing reward" in message and "2" in message and "3" in message and "7" in message for message in messages)
    assert any("unknown" in message and "4" in message for message in messages)
    assert all("source" in message for message in messages)


def test_parent_record_can_log_failure_before_signature(tmp_path):
    """An ordinary parent record saves contextual early errors without a numerical hit."""
    from src.behavior_analysis import trial_policy_results as results

    meta = {"session_id": "source", "session_ids": ["source"], "status": "error",
            "errors": ["source trial 3: invalid action"], "warnings": [], "runtime_seconds": 0.01}
    results.write_trial_policy_run_record(tmp_path / "parent", meta)
    assert "invalid action" in (tmp_path / "parent" / "run.log").read_text()
    assert not list(tmp_path.rglob("results.npz"))
    assert not list(tmp_path.rglob("complete.json"))


def test_unknown_flag_warning_with_both_matching_aliases(trial_table):
    """A source with both supported flag columns still reports unknown-row exclusions."""
    from src.behavior_analysis import trial_policy_results as results
    from src.behavior_analysis.trial_policy_models import prepare_trial_policy_history

    table = trial_table.assign(give_reward=[0, None, 0, 0, 0],
                               experimenter_reward_given=[0, None, 0, 0, 0])
    history = prepare_trial_policy_history(table, session_id="source")
    messages = results.trial_policy_warnings(history)
    assert len(messages) == 1 and "unknown" in messages[0] and "4" in messages[0]


def test_joint_export_records_source_arrays_without_claiming_joint_numerical_file(trial_table, tmp_path):
    """A joint CSV links its source manifest without inventing one numerical NPZ."""
    from src.behavior_analysis import trial_policy_results as results

    result, meta = metadata_for(trial_table)
    path = results.save_trial_policy_results(result, processed_data_path=tmp_path, metadata=meta)
    meta["sources"] = [{"session_id": "source", "numerical_artifact": str(path)}]
    attached = results.attach_trial_policy_outputs(trial_table, result, metadata=meta)
    results.write_trial_policy_export(attached, export_directory=tmp_path / "joint_export",
                                      numerical_path=None, metadata=meta)
    record = json.loads((tmp_path / "joint_export" / "metadata.json").read_text())
    assert "numerical_artifact" not in record
    assert record["sources"][0]["numerical_artifact"] == str(path)
