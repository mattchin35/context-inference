import json
import numpy as np

def load_probe_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


# ----------------------------
# 1. Geometry (brain boundary)
# ----------------------------
def get_brain_mask(vertical_pos, surface_y, include_surface=False):
    """
    Returns boolean mask of channels in brain.

    Convention:
        vertical_pos = 0 is deepest
        smaller = deeper
    """
    vertical_pos = np.asarray(vertical_pos)

    if include_surface:
        return vertical_pos <= surface_y
    else:
        return vertical_pos < surface_y


# ----------------------------
# 2. Shank grouping
# ----------------------------
def get_shank_sites(shank_index, mask=None):
    """
    Returns dict mapping shank -> channel indices.

    If mask is provided, only includes those channels.
    """
    shank_index = np.asarray(shank_index).astype(int)

    if mask is None:
        mask = np.ones_like(shank_index, dtype=bool)

    shank_sites = {}
    for shank in np.unique(shank_index):
        shank_mask = (shank_index == shank)
        combined_mask = shank_mask & mask

        shank_sites[int(shank)] = np.where(combined_mask)[0]

    return shank_sites


# ----------------------------
# 3. Convenience wrapper
# ----------------------------
def get_brain_and_shank_sites(data, include_surface=False):
    vertical_pos = np.array(data["vertical_pos"])
    shank_index = np.array(data["shank_index"])
    surface_y = data["surface_y"]

    brain_mask = get_brain_mask(vertical_pos, surface_y, include_surface)
    brain_channels = np.where(brain_mask)[0]
    shank_sites = get_shank_sites(shank_index, mask=brain_mask)

    return brain_channels, shank_sites


if __name__ == "__main__":
    filepath = "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/ephys/catgt/catgt_run0_g0/run0_g0_imec1/probe_json.json"

    data = load_probe_json(filepath)

    brain_channels, shank_sites = get_brain_and_shank_sites(data)

    print(f"Channels in brain: {len(brain_channels)}\n")

    print("1-indexed shank-wise channel indices (in brain):")
    for shank, indices in shank_sites.items():
        print(f"Shank {shank}: {len(indices)} channels")
        print(indices+1)  # conversion for matlab indexing
        print()
