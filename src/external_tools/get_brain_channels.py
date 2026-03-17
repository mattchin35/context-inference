import json
import numpy as np

def load_probe_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_brain_channels(data: dict, include_surface=False):
    """
    Returns 0-indexed numpy array of channels inside the brain.

    Convention:
        vertical_pos = 0 is deepest
        Larger values = more superficial

    include_surface:
        False -> strictly below surface (recommended)
        True  -> include channels exactly at surface
    """

    vertical_pos = np.array(data["vertical_pos"])
    surface_y = data["surface_y"]  # scalar

    if include_surface:
        mask = vertical_pos <= surface_y
    else:
        mask = vertical_pos < surface_y

    return np.where(mask)[0]


if __name__ == "__main__":
    filepath = "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/ephys/catgt/catgt_run0_g0/run0_g0_imec0/probe_json.json"

    data = load_probe_json(filepath)

    brain_channels = get_brain_channels(data, include_surface=False)

    print(f"Total channels: {len(data['vertical_pos'])}")
    print(f"Channels in brain: {len(brain_channels)}")
    print("Indices:")
    print(brain_channels)
