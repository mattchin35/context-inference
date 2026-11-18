import json
import numpy as np
import pytest

from src.external_tools.get_brain_channels import (
    get_brain_and_shank_sites,
    save_shank_sites_json,
    serialize_shank_sites_for_json,
)


def make_probe_data():
    """Return synthetic probe geometry for sorting tests.

    Returns:
        dict[str, object]: Probe metadata dictionary with:
            - ``vertical_pos`` (list[float]): Channel depth coordinates where
              smaller values are deeper.
            - ``horizontal_pos`` (list[float]): Channel lateral coordinates.
            - ``shank_index`` (list[int]): Shank assignment for each channel.
            - ``surface_y`` (float): Brain surface coordinate in the same units
              as ``vertical_pos``.
    """
    return {
        "vertical_pos": [30, 10, 20, 5, 15, 25],
        "horizontal_pos": [0, 1, 0, 2, 1, 2],
        "shank_index": [0, 0, 0, 1, 1, 1],
        "surface_y": 22,
    }


def make_probe_data_with_empty_groups():
    """Return synthetic probe geometry containing empty in/out-of-brain groups.

    Returns:
        dict[str, object]: Probe metadata dictionary with the same field
        definitions as :func:`make_probe_data`.
    """
    return {
        "vertical_pos": [30, 10, 20, 5, 15, 25, 35, 40, 0, 2],
        "horizontal_pos": [0, 1, 0, 2, 1, 2, 3, 4, 0, 1],
        "shank_index": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
        "surface_y": 22,
    }


def test_get_brain_and_shank_sites_returns_nested_shank_dict_for_all_shanks():
    probe_data = make_probe_data()

    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="deep_to_shallow",
    )

    assert set(shank_sites.keys()) == {0, 1}
    assert set(shank_sites[0].keys()) == {"in_brain", "out_of_brain"}
    assert set(shank_sites[1].keys()) == {"in_brain", "out_of_brain"}


def test_each_shank_contains_in_brain_and_out_of_brain_keys_even_when_empty():
    probe_data = make_probe_data_with_empty_groups()

    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="deep_to_shallow",
    )

    assert set(shank_sites.keys()) == {0, 1, 2, 3}
    assert np.array_equal(shank_sites[2]["in_brain"], np.array([], dtype=int))
    assert np.array_equal(shank_sites[3]["out_of_brain"], np.array([], dtype=int))


def test_empty_groups_are_returned_as_empty_integer_arrays():
    probe_data = make_probe_data_with_empty_groups()

    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="deep_to_shallow",
    )

    assert shank_sites[2]["in_brain"].dtype == int
    assert shank_sites[3]["out_of_brain"].dtype == int
    assert shank_sites[2]["in_brain"].size == 0
    assert shank_sites[3]["out_of_brain"].size == 0


def test_deep_to_shallow_sorts_in_brain_and_out_of_brain_within_each_shank():
    probe_data = make_probe_data()

    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="deep_to_shallow",
    )

    assert np.array_equal(shank_sites[0]["in_brain"], np.array([1, 2]))
    assert np.array_equal(shank_sites[0]["out_of_brain"], np.array([0]))
    assert np.array_equal(shank_sites[1]["in_brain"], np.array([3, 4]))
    assert np.array_equal(shank_sites[1]["out_of_brain"], np.array([5]))


def test_shallow_to_deep_fully_reverses_depth_and_horizontal_within_each_group():
    probe_data = make_probe_data()

    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="shallow_to_deep",
    )

    assert np.array_equal(shank_sites[0]["in_brain"], np.array([2, 1]))
    assert np.array_equal(shank_sites[0]["out_of_brain"], np.array([0]))
    assert np.array_equal(shank_sites[1]["in_brain"], np.array([4, 3]))
    assert np.array_equal(shank_sites[1]["out_of_brain"], np.array([5]))


def test_brain_channels_remain_sorted_consistently_with_sort_order():
    probe_data = make_probe_data()

    brain_channels_deep, _ = get_brain_and_shank_sites(
        probe_data,
        sort_order="deep_to_shallow",
    )
    brain_channels_shallow, _ = get_brain_and_shank_sites(
        probe_data,
        sort_order="shallow_to_deep",
    )

    assert np.array_equal(brain_channels_deep, np.array([3, 1, 4, 2]))
    assert np.array_equal(brain_channels_shallow, np.array([2, 4, 1, 3]))


def test_get_brain_and_shank_sites_rejects_invalid_sort_order():
    probe_data = make_probe_data()

    with pytest.raises(ValueError, match="sort_order"):
        get_brain_and_shank_sites(
            probe_data,
            sort_order="alphabetical",
        )


def test_serialize_shank_sites_for_json_returns_expected_schema():
    probe_data = make_probe_data()
    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="deep_to_shallow",
    )

    serialized = serialize_shank_sites_for_json(shank_sites)

    assert serialized == {
        "shanks": [
            {"shank": 0, "in_brain": [1, 2], "out_of_brain": [0]},
            {"shank": 1, "in_brain": [3, 4], "out_of_brain": [5]},
        ]
    }


def test_serialize_shank_sites_for_json_preserves_empty_groups_as_empty_lists():
    probe_data = make_probe_data_with_empty_groups()
    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="deep_to_shallow",
    )

    serialized = serialize_shank_sites_for_json(shank_sites)

    assert serialized["shanks"][2]["in_brain"] == []
    assert serialized["shanks"][3]["out_of_brain"] == []


def test_save_shank_sites_json_writes_ordered_data_to_existing_directory(tmp_path):
    probe_data = make_probe_data()
    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="shallow_to_deep",
    )

    save_path = save_shank_sites_json(tmp_path, shank_sites)

    assert save_path == tmp_path / "shank_sites.json"
    with save_path.open("r", encoding="utf-8") as file_handle:
        saved_data = json.load(file_handle)

    assert saved_data == {
        "shanks": [
            {"shank": 0, "in_brain": [2, 1], "out_of_brain": [0]},
            {"shank": 1, "in_brain": [4, 3], "out_of_brain": [5]},
        ]
    }


def test_save_shank_sites_json_rejects_missing_output_directory(tmp_path):
    probe_data = make_probe_data()
    _, shank_sites = get_brain_and_shank_sites(
        probe_data,
        sort_order="deep_to_shallow",
    )
    missing_output_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="output_path"):
        save_shank_sites_json(missing_output_path, shank_sites)
