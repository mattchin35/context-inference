"""Tests for the narrow legacy-table rewards-in-block backfill migration."""

from pathlib import Path

import pandas as pd
import pytest

from src.behavior_analysis import gather_trial_features as gather_trial_features


def make_saved_augmented_table() -> pd.DataFrame:
    """Build a four-row CSV-ready table with all required source columns.

    Returns
    -------
    pd.DataFrame
        Table with shape ``(4, 6)``. The extra columns exercise exact
        preservation of pre-existing loaded CSV values during backfill.
    """
    return pd.DataFrame(
        {
            "cur_block": [0, 0, 0, 1],
            "action": ["0", "no_choice", "1", "0"],
            "reward": ["1", "ignored", "2.0", "0"],
            "experimenter_reward_given": ["0", "0", "0", "0"],
            "existing_label": ["first", "None", "third", "fourth"],
            "existing_value": ["1.5", "0", "-2", "3.25"],
        }
    )


def write_csv_with_project_sentinels(path: Path, trial_df: pd.DataFrame) -> None:
    """Write one table using the project's CSV missing-value convention.

    Parameters
    ----------
    path : Path
        Destination CSV path.
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)`` to write without an
        index and with missing values represented by the literal ``"None"``.
    """
    trial_df.to_csv(path, index=False, na_rep="None")


def sibling_files(path: Path) -> set[Path]:
    """Return direct sibling paths for temporary-file assertions.

    Parameters
    ----------
    path : Path
        Existing CSV path whose parent directory is inspected.

    Returns
    -------
    set[Path]
        All direct children of ``path.parent`` at the time of the call.
    """
    return set(path.parent.iterdir())


def exception_chain_messages(error: BaseException) -> list[str]:
    """Return messages from one exception and its explicit/implicit context chain."""
    messages = []
    seen_ids = set()
    current_error: BaseException | None = error
    while current_error is not None and id(current_error) not in seen_ids:
        seen_ids.add(id(current_error))
        messages.append(str(current_error))
        current_error = current_error.__cause__ or current_error.__context__
    return messages


def test_backfill_rewards_in_block_csv_adds_only_column_and_publishes_atomically(
    tmp_path,
    monkeypatch,
):
    """Backfill should preserve loaded data and use one atomic replacement publication."""
    csv_path = tmp_path / "session_augmented_trials.csv"
    feature_parameters_path = tmp_path / "trial_feature_params.json"
    unrelated_path = tmp_path / "unrelated.txt"
    write_csv_with_project_sentinels(csv_path, make_saved_augmented_table())
    feature_parameters_path.write_text('{"seed": 7}\n', encoding="ascii")
    unrelated_path.write_text("leave me alone\n", encoding="ascii")
    expected_before = pd.read_csv(csv_path, na_filter=False)
    feature_parameters_bytes = feature_parameters_path.read_bytes()
    unrelated_bytes = unrelated_path.read_bytes()
    files_before = sibling_files(csv_path)

    real_replace = gather_trial_features.os.replace
    published_paths: list[tuple[Path, Path]] = []

    def record_replace(source, destination):
        """Record the publication boundary while performing the real replacement."""
        published_paths.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(gather_trial_features.os, "replace", record_replace)

    gather_trial_features.backfill_rewards_in_block_csv(csv_path)

    saved_after = pd.read_csv(csv_path, na_filter=False)
    expected_after = expected_before.assign(rewards_in_block=[0, 1, 1, 0])
    pd.testing.assert_frame_equal(saved_after, expected_after)
    assert feature_parameters_path.read_bytes() == feature_parameters_bytes
    assert unrelated_path.read_bytes() == unrelated_bytes
    assert sibling_files(csv_path) == files_before
    assert len(published_paths) == 1
    assert published_paths[0][1] == csv_path
    assert published_paths[0][0].parent == csv_path.parent
    assert published_paths[0][0] != csv_path


def test_backfill_rewards_in_block_csv_reuses_general_behavior_helper(
    tmp_path,
    monkeypatch,
):
    """The migration should delegate counting to the general augmentation helper."""
    csv_path = tmp_path / "session_augmented_trials.csv"
    write_csv_with_project_sentinels(csv_path, make_saved_augmented_table())
    expected_loaded = pd.read_csv(csv_path, na_filter=False)
    helper_calls: list[pd.DataFrame] = []

    def record_helper_call(trial_df):
        """Return distinctive counts while retaining the helper input for inspection."""
        helper_calls.append(trial_df.copy(deep=True))
        return pd.Series(
            [4, 3, 2, 1],
            index=trial_df.index,
            name="rewards_in_block",
            dtype="int64",
        )

    monkeypatch.setattr(
        gather_trial_features.session_analysis,
        "compute_rewards_in_block",
        record_helper_call,
    )

    gather_trial_features.backfill_rewards_in_block_csv(csv_path)

    assert len(helper_calls) == 1
    pd.testing.assert_frame_equal(helper_calls[0], expected_loaded)
    saved_after = pd.read_csv(csv_path, na_filter=False)
    assert saved_after["rewards_in_block"].tolist() == [4, 3, 2, 1]


def test_backfill_rewards_in_block_csv_refuses_missing_source_columns_without_writing(tmp_path):
    """Missing canonical source columns should leave the original CSV bytes unchanged."""
    csv_path = tmp_path / "session_augmented_trials.csv"
    write_csv_with_project_sentinels(
        csv_path,
        make_saved_augmented_table().drop(columns=["action"]),
    )
    original_bytes = csv_path.read_bytes()
    files_before = sibling_files(csv_path)

    with pytest.raises(ValueError, match="action"):
        gather_trial_features.backfill_rewards_in_block_csv(csv_path)

    assert csv_path.read_bytes() == original_bytes
    assert sibling_files(csv_path) == files_before


def test_backfill_rewards_in_block_csv_refuses_existing_destination_without_writing(tmp_path):
    """A table already containing the destination feature should not be rewritten."""
    csv_path = tmp_path / "session_augmented_trials.csv"
    table_with_destination = make_saved_augmented_table().assign(
        rewards_in_block=[0, 1, 1, 0]
    )
    write_csv_with_project_sentinels(csv_path, table_with_destination)
    original_bytes = csv_path.read_bytes()
    files_before = sibling_files(csv_path)

    with pytest.raises(ValueError, match="rewards_in_block"):
        gather_trial_features.backfill_rewards_in_block_csv(csv_path)

    assert csv_path.read_bytes() == original_bytes
    assert sibling_files(csv_path) == files_before


def test_backfill_rewards_in_block_csv_cleans_temp_after_serialization_failure(
    tmp_path,
    monkeypatch,
):
    """A serialization error should leave the original bytes and no temporary sibling."""
    csv_path = tmp_path / "session_augmented_trials.csv"
    write_csv_with_project_sentinels(csv_path, make_saved_augmented_table())
    original_bytes = csv_path.read_bytes()
    files_before = sibling_files(csv_path)

    def fail_to_csv(*_args, **_kwargs):
        """Inject a deterministic write failure after a temporary path is allocated."""
        raise OSError("injected serialization failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(OSError, match="injected serialization failure"):
        gather_trial_features.backfill_rewards_in_block_csv(csv_path)

    assert csv_path.read_bytes() == original_bytes
    assert sibling_files(csv_path) == files_before


def test_backfill_rewards_in_block_csv_cleans_temp_after_roundtrip_validation_failure(
    tmp_path,
    monkeypatch,
):
    """A loaded temporary-table mismatch should abort before replacing the original."""
    csv_path = tmp_path / "session_augmented_trials.csv"
    write_csv_with_project_sentinels(csv_path, make_saved_augmented_table())
    original_bytes = csv_path.read_bytes()
    files_before = sibling_files(csv_path)
    original_read_csv = pd.read_csv
    read_count = 0

    def corrupt_temporary_reload(*args, **kwargs):
        """Return an altered table after the initial source-table load."""
        nonlocal read_count
        read_count += 1
        loaded_df = original_read_csv(*args, **kwargs)
        if read_count >= 2:
            loaded_df = loaded_df.copy()
            loaded_df.loc[loaded_df.index[0], "existing_label"] = "corrupted"
        return loaded_df

    monkeypatch.setattr(gather_trial_features.pd, "read_csv", corrupt_temporary_reload)

    with pytest.raises(ValueError, match="round-trip|unchanged|validation"):
        gather_trial_features.backfill_rewards_in_block_csv(csv_path)

    assert csv_path.read_bytes() == original_bytes
    assert sibling_files(csv_path) == files_before


def test_backfill_rewards_in_block_csv_surfaces_temp_cleanup_failure_with_context(
    tmp_path,
    monkeypatch,
):
    """A failed temporary-file cleanup should be visible without changing the source CSV."""
    csv_path = tmp_path / "session_augmented_trials.csv"
    write_csv_with_project_sentinels(csv_path, make_saved_augmented_table())
    original_bytes = csv_path.read_bytes()
    original_unlink = Path.unlink

    def fail_to_csv(*_args, **_kwargs):
        """Inject the original serialization failure that triggers cleanup."""
        raise OSError("injected serialization failure")

    def refuse_temporary_unlink(path, *args, **kwargs):
        """Refuse only generated sibling cleanup attempts, leaving evidence for this test."""
        if path.parent == csv_path.parent and path != csv_path:
            raise OSError("injected temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)
    monkeypatch.setattr(Path, "unlink", refuse_temporary_unlink)

    with pytest.raises(OSError, match="injected temporary cleanup failure") as error_info:
        gather_trial_features.backfill_rewards_in_block_csv(csv_path)

    temporary_paths = sibling_files(csv_path) - {csv_path}
    assert len(temporary_paths) == 1
    temporary_path = temporary_paths.pop()
    assert str(temporary_path) in str(error_info.value)
    assert any(
        "injected serialization failure" in message
        for message in exception_chain_messages(error_info.value)
    )
    assert csv_path.read_bytes() == original_bytes

    original_unlink(temporary_path)
