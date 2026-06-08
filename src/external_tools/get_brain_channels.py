import json
from pathlib import Path

import numpy as np


def load_probe_json(filepath: str):
    """Load probe metadata from a JSON file.

    Args:
        filepath (str): Path to a JSON file containing probe metadata.

    Returns:
        dict[str, object]: Parsed JSON dictionary. Expected downstream keys
        include ``vertical_pos``, ``horizontal_pos``, ``shank_index``, and
        ``surface_y``.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# ----------------------------
# 1. Geometry (brain boundary)
# ----------------------------
def get_brain_mask(vertical_pos, surface_y, include_surface=False):
    """Return a boolean mask for channels that lie in brain tissue.

    Args:
        vertical_pos (array-like of float, shape (n_channels,)): Channel
            vertical coordinates. Smaller values correspond to deeper sites.
        surface_y (float): Surface coordinate in the same units as
            ``vertical_pos``.
        include_surface (bool): If ``True``, channels exactly at ``surface_y``
            are included in the mask.

    Returns:
        np.ndarray of bool, shape (n_channels,): ``True`` for channels that are
        below the surface according to the depth convention above.
    """
    vertical_pos = np.asarray(vertical_pos)

    if include_surface:
        return vertical_pos <= surface_y
    else:
        return vertical_pos < surface_y


# ----------------------------
# 2. Shank grouping
# ----------------------------
def validate_sort_order(sort_order):
    """Validate and normalize channel sort order.

    Args:
        sort_order (str | None): Requested ordering. Accepted values are
            ``None``, ``"deep_to_shallow"``, and ``"shallow_to_deep"``.

    Returns:
        str | None: Normalized sort order.

    Raises:
        ValueError: If ``sort_order`` is not an accepted value.
    """
    valid_sort_orders = {None, "deep_to_shallow", "shallow_to_deep"}
    if sort_order not in valid_sort_orders:
        raise ValueError(
            "sort_order must be one of None, 'deep_to_shallow', or 'shallow_to_deep'"
        )

    return sort_order


def get_shank_sites(
    shank_index,
    brain_mask,
    vertical_pos=None,
    horizontal_pos=None,
    sort_order=None,
):
    """Return in-brain and out-of-brain channel indices grouped by shank.

    Args:
        shank_index (array-like of int, shape (n_channels,)): Shank label for
            each channel.
        brain_mask (array-like of bool, shape (n_channels,)): Boolean mask where
            ``True`` marks channels in brain tissue.
        vertical_pos (array-like of float, shape (n_channels,)) | None:
            Vertical channel coordinates in arbitrary distance units.
        horizontal_pos (array-like of float, shape (n_channels,)) | None:
            Horizontal channel coordinates in the same distance units as
            ``vertical_pos``.
        sort_order (str | None): Channel ordering within each shank. Accepted
            values are ``None``, ``"deep_to_shallow"``, and
            ``"shallow_to_deep"``.

    Returns:
        dict[int, dict[str, np.ndarray]]: Mapping from shank index to a nested
        dictionary with ``"in_brain"`` and ``"out_of_brain"`` zero-based
        channel-index arrays. Empty groups are returned as empty integer arrays.

    Raises:
        ValueError: If sorting is requested without geometry arrays or if the
            sort order is invalid.
    """
    sort_order = validate_sort_order(sort_order)
    shank_index = np.asarray(shank_index).astype(int)
    brain_mask = np.asarray(brain_mask, dtype=bool)

    shank_sites = {}
    for shank in np.unique(shank_index):
        shank_mask = (shank_index == shank)
        in_brain_indices = np.where(shank_mask & brain_mask)[0].astype(int)
        out_of_brain_indices = np.where(shank_mask & ~brain_mask)[0].astype(int)

        if sort_order is not None:
            if vertical_pos is None or horizontal_pos is None:
                raise ValueError(
                    "Sorting requires vertical_pos and horizontal_pos"
                )
            in_brain_indices = sort_channels_within_shank(
                in_brain_indices,
                vertical_pos,
                horizontal_pos,
                sort_order=sort_order,
            )
            out_of_brain_indices = sort_channels_within_shank(
                out_of_brain_indices,
                vertical_pos,
                horizontal_pos,
                sort_order=sort_order,
            )

        shank_sites[int(shank)] = {
            "in_brain": in_brain_indices,
            "out_of_brain": out_of_brain_indices,
        }

    return shank_sites


# ----------------------------
# 3. Convenience wrapper
# ----------------------------
def get_brain_and_shank_sites(data, include_surface=False, sort_order=None):
    """Return brain-channel indices and per-shank brain-membership groups.

    Args:
        data (dict[str, object]): Probe metadata dictionary containing
            ``vertical_pos`` (shape ``(n_channels,)``), ``horizontal_pos``
            (shape ``(n_channels,)``), ``shank_index`` (shape
            ``(n_channels,)``), and ``surface_y``.
        include_surface (bool): If ``True``, channels exactly at the surface are
            included as brain channels.
        sort_order (str | None): Optional ordering for returned channel indices.
            Accepted values are ``None``, ``"deep_to_shallow"``, and
            ``"shallow_to_deep"``.

    Returns:
        tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
            - ``brain_channels``: zero-based channel indices with shape
              ``(n_brain_channels,)``.
            - ``shank_sites``: mapping from shank index to dictionaries with
              ``"in_brain"`` and ``"out_of_brain"`` zero-based channel-index
              arrays. Empty groups are returned as empty integer arrays.
    """
    sort_order = validate_sort_order(sort_order)
    vertical_pos = np.array(data["vertical_pos"])
    horizontal_pos = np.array(data["horizontal_pos"])
    shank_index = np.array(data["shank_index"])
    surface_y = data["surface_y"]

    brain_mask = get_brain_mask(vertical_pos, surface_y, include_surface)
    brain_channels = np.where(brain_mask)[0]
    if sort_order is not None:
        brain_channels = sort_channels_within_shank(
            brain_channels,
            vertical_pos,
            horizontal_pos,
            sort_order=sort_order,
        )

    shank_sites = get_shank_sites(
        shank_index,
        brain_mask=brain_mask,
        vertical_pos=vertical_pos,
        horizontal_pos=horizontal_pos,
        sort_order=sort_order,
    )

    return brain_channels, shank_sites


def get_depth_from_surface(vertical_pos, surface_y):
    """Compute depth below the brain surface for each channel.

    Args:
        vertical_pos (array-like of float, shape (n_channels,)): Channel
            vertical coordinates.
        surface_y (float): Surface coordinate in the same units as
            ``vertical_pos``.

    Returns:
        np.ndarray of float, shape (n_channels,): Positive values indicate
        depth below the surface in the original coordinate units.
    """
    return surface_y - np.asarray(vertical_pos)


def serialize_shank_sites_for_json(shank_sites):
    """Convert nested shank-site arrays into a JSON-safe schema.

    Args:
        shank_sites (dict[int, dict[str, np.ndarray]]): Mapping from shank index
            to nested dictionaries with ``"in_brain"`` and ``"out_of_brain"``
            zero-based channel-index arrays.

    Returns:
        dict[str, list[dict[str, object]]]: Dictionary with key ``"shanks"``
        whose value is a list of dictionaries containing integer ``"shank"``
        identifiers and list-valued ``"in_brain"`` / ``"out_of_brain"``
        channel indices.
    """
    serialized_shanks = []
    for shank, site_groups in shank_sites.items():
        serialized_shanks.append(
            {
                "shank": int(shank),
                "in_brain": np.asarray(site_groups["in_brain"], dtype=int).tolist(),
                "out_of_brain": np.asarray(
                    site_groups["out_of_brain"], dtype=int
                ).tolist(),
            }
        )

    return {"shanks": serialized_shanks}


def save_shank_sites_json(output_path, shank_sites):
    """Save nested shank-site arrays to ``shank_sites.json``.

    Args:
        output_path (str | Path): Existing directory where
            ``shank_sites.json`` will be written.
        shank_sites (dict[int, dict[str, np.ndarray]]): Mapping from shank index
            to nested dictionaries with ``"in_brain"`` and ``"out_of_brain"``
            zero-based channel-index arrays.

    Returns:
        Path: Path to the written ``shank_sites.json`` file.

    Raises:
        FileNotFoundError: If ``output_path`` does not exist or is not a
            directory.
    """
    output_directory = Path(output_path)
    if not output_directory.is_dir():
        raise FileNotFoundError(
            f"output_path must be an existing directory: {output_directory}"
        )

    output_file = output_directory / "shank_sites.json"
    serialized_shank_sites = serialize_shank_sites_for_json(shank_sites)
    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(serialized_shank_sites, file_handle, indent=2)

    return output_file


def sort_channels_within_shank(indices, vertical_pos, horizontal_pos, sort_order):
    """Sort channel indices by depth and horizontal position.

    Args:
        indices (array-like of int, shape (n_selected_channels,)): Zero-based
            channel indices to sort.
        vertical_pos (array-like of float, shape (n_channels,)): Vertical
            channel coordinates. Smaller values are deeper.
        horizontal_pos (array-like of float, shape (n_channels,)): Horizontal
            channel coordinates.
        sort_order (str): Either ``"deep_to_shallow"`` or
            ``"shallow_to_deep"``.

    Returns:
        np.ndarray of int, shape (n_selected_channels,): Sorted zero-based
        channel indices.
    """
    sort_order = validate_sort_order(sort_order)
    vertical_pos = np.asarray(vertical_pos)
    horizontal_pos = np.asarray(horizontal_pos)
    indices = np.asarray(indices, dtype=int)

    v = vertical_pos[indices]
    h = horizontal_pos[indices]

    if sort_order == "deep_to_shallow":
        order = np.lexsort((h, v))
    else:
        order = np.lexsort((-h, -v))

    return indices[order]


if __name__ == "__main__":
    home_folder = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference/ephys/catgt/catgt_run0_g0/run0_g0_imec0/")
    filepath = home_folder / "probe_json.json"
    output_path = home_folder

    data = load_probe_json(filepath)

    ## sort order can be None, "deep_to_shallow", or "shallow_to_deep"
    brain_channels, shank_sites = get_brain_and_shank_sites(
        data,
        sort_order="shallow_to_deep",
    )

    print(f"Channels in brain: {len(brain_channels)}\n")

    index_offset = False
    index_offset = int(index_offset)
    print(f"{index_offset}-indexed shank-wise channel indices:")
    for shank, site_groups in shank_sites.items():
        print(f"Shank {shank}:")
        print("  In brain:")
        print(site_groups["in_brain"] + index_offset)  # console conversion for matlab indexing
        print("  Out of brain:")
        print(site_groups["out_of_brain"] + index_offset)  # console conversion for matlab indexing
        print()

    save_shank_sites_json(output_path, shank_sites)
