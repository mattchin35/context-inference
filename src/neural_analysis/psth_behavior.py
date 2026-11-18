"""Legacy PSTH API with plotting and direct dispatch retained."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis import spike_behavior_pynapple
from src.neural_analysis.spike_behavior.psth import (
    LickPethData,
    SpikePethData,
    build_lick_peth_data,
    build_single_unit_spike_peth_data,
    select_first_valid_unit_cluster_id,
)
from src.neural_analysis.spike_behavior.trials import (
    make_lick_peth_trial_type_masks,
    select_valid_lick_peth_trials,
)

SUPPORTED_ANALYSIS_REGIONS = {"HPC", "V1", "PFC"}
LEFT_LICK_EVENT = "left_entry"
RIGHT_LICK_EVENT = "right_entry"
LICK_COLORS = {
    LEFT_LICK_EVENT: "tab:orange",
    RIGHT_LICK_EVENT: "tab:blue",
}
from src.neural_analysis.spike_behavior.plotting import (
    _draw_peth_trial_event_markers as _draw_trial_event_markers,
    _format_lick_axes,
    _format_spike_axes,
    _lick_peth_fig_height,
    _nanmean_rate,
    _plot_raster_rows,
    plot_lick_peth,
    plot_peth,
    plot_single_unit_spike_peth,
    save_figure_with_message,
)


def main() -> None:
    """
    Run one hard-coded session example for user-defined region spike binning.

    The script loads processed behavior tables, loads aligned spike times plus sorter cluster
    metadata for the current example HPC probe, builds a Pynapple spike group, bins
    region-selected spikes by trial, and prints a short summary. Times are handled in seconds throughout.
    """

    multi_session_save_path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis")
    session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference")
    sess_id_full = "CT014_2025-12-23_163505"
    pfc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/Kilosort2.5.2_2026-03-19_180103/sorting_mchin_20260330"
    hpc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/Kilosort2.5.2_2026-03-19_183540/sorting_mchin_20260331"

    raw_behavior_folder = session_data_home / "rpi" / sess_id_full
    processed_data_path = session_data_home / "processed"
    figure_path = session_data_home / "figures" / "peth"

    aligned_spike_path = session_data_home / 'ephys/aligned/aligned_imec'
    aligned_pfc_spike_path = aligned_spike_path / 'imec0_sync.npz'
    aligned_hpc_spike_path = aligned_spike_path / 'imec1_sync.npz'

    probe_json_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/probe_json.json"

    if not hpc_spike_path.exists():
        raise FileNotFoundError(f"HPC sorter output not found at {hpc_spike_path}")
    assert probe_json_path.exists(), f"Probe JSON file not found at {probe_json_path}"
    assert aligned_pfc_spike_path.exists(), f"Aligned PFC spike file not found at {aligned_pfc_spike_path}"
    assert aligned_hpc_spike_path.exists(), f"Aligned HPC spike file not found at {aligned_hpc_spike_path}"

    decode_target = "action"  # state_int or action
    n_decoder_runs = 5
    # print(", ".join(map(str, "your_array")))
    hpc_channels_shank0 = np.array([
        4, 3, 2, 1, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181,
        180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167,
        166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153,
        152, 151, 150, 149, 148, 147, 146, 145, 96, 95, 94, 93, 92, 91, 90, 89,
        88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71,
        70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53,
        52, 51, 50, 49, 384, 383, 382, 381
    ])
    hpc_channels_shank3 = np.array([
        255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242,
        241, 336, 335, 334, 333, 332, 331, 330, 329, 328, 327, 326, 325, 324,
        323, 322, 321, 320, 319, 318, 317, 316, 315, 314, 313, 312, 311, 310,
        309, 308, 307, 306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296,
        295, 294, 293, 292, 291, 290, 289, 240, 239, 238, 237, 236, 235, 234,
        233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220,
        219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208, 207, 206,
        205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 144,
        143, 142, 141
    ])
    hpc_channels = np.concatenate([hpc_channels_shank0, hpc_channels_shank3])

    v1_channels_shank0 = np.array([
        138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125,
        124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
        110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97,
        48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
        14, 13, 12, 11, 10, 9, 8, 7, 6, 5
    ])
    v1_channels_shank3 = np.array([
        78, 377, 376, 375, 374, 373, 372, 371, 370, 369, 368, 367, 366, 365,
        364, 363, 362, 361, 360, 359, 358, 357, 356, 355, 354, 353, 352, 351,
        350, 349, 348, 347, 346, 345, 344, 343, 342, 341, 340, 339, 338, 337,
        288, 287, 286, 285, 284, 283, 282, 281, 280, 279, 278, 277, 276, 275,
        274, 273, 272, 271, 270, 269, 268, 267, 266, 265, 264, 263, 262, 261,
        260, 259, 258, 257, 256
    ])
    v1_channels = np.concatenate([v1_channels_shank0, v1_channels_shank3])

    _pfc_channels = ("55  54  53  52  51  50  49  48 383 382 381 380 379 378 377 376 375 374 "
                    "373 372 371 370 369 368 367 366 365 364 363 362 361 360 359 358 357 356 "
                    "355 354 353 352 351 350 349 348 347 346 345 344 343 342 341 340 339 338 "
                    "337 336 287 286 285 284 283 282 281 280 279 278 277 276 275 274 273 272 "
                    "271 270 269 268 267 266 265 264 263 262 261 260 259 258 257 256 255 254 "
                    "253 252 251 250 249 248 247 246 245 244 243 242 241 240 335 334 333 332 "
                    "331 330 329 328 327 326 325 324 323 322 321 320 319 318 317 316 315 314 "
                    "313 312 311 310 309 308 307 306 305 304 303 302 301 300 299 298 297 296 "
                    "295 294 293 292 291 290 289 288 239 238 237 236 235 234 233 232 231 230 "
                    "229 228 227 226 225 224 223 222 221 220 219 218 217 216 215 214 213 212 "
                    "211 210 209 208 207 206 205 204 203 202 201 200 199 198 197 196 195 194 "
                    "193 192 143 142 141 140 139 138 137 136 135 134 133 132 131 130 129 128 "
                    "127 126 125 124 123 122 121 120 119 118 117 116 115 114 113 112 111 110 "
                    "109 108 107 106 105 104 103 102 101 100  99  98  97  96  47  46  45  44 "
                    "43  42  41  40  39  38  37  36  35  34  33  32  31  30  29  28  27  26 "
                    "25  24  23  22  21  20  19  18  17  16  15  14  13  12  11  10  9  8 "
                    "7   6   5   4   3   2   1   0")
    pfc_channels = np.sort(np.array(list(map(int, [val for val in _pfc_channels.split(" ") if val != '']))))

    mouse, date, timestamp = spike_behavior_pynapple.parse_session_id(sess_id_full)
    session_info_path = raw_behavior_folder / f"{sess_id_full}_session_info.pkl"

    session = spike_behavior_pynapple.Session(
        multi_session_save_path=multi_session_save_path,
        session_data_home=session_data_home,
        sess_id_full=sess_id_full,
        sess_id_abbreviated=f"{mouse}_{date}",
        raw_behavior_folder=raw_behavior_folder,
        processed_data_path=processed_data_path,
        figure_path=figure_path,
        mouse=mouse,
        date=date,
        timestamp=timestamp,
        session_info_path=session_info_path,
        session_info=spike_behavior_pynapple.load_session_info(session_info_path)
        if session_info_path.exists()
        else None,
        pfc_spike_path=pfc_spike_path,
        hpc_spike_path=hpc_spike_path,
    )

    region_name = "PFC"
    if region_name in SUPPORTED_ANALYSIS_REGIONS:
        if region_name == "HPC":
            region_channels = spike_behavior_pynapple.normalize_region_channels(hpc_channels)
            sorter_output_path = session.hpc_spike_path
            spike_path = aligned_hpc_spike_path
        elif region_name == "V1":
            region_channels = spike_behavior_pynapple.normalize_region_channels(v1_channels)
            sorter_output_path = session.hpc_spike_path
            spike_path = aligned_hpc_spike_path
        elif region_name == "PFC":
            region_channels = spike_behavior_pynapple.normalize_region_channels(pfc_channels)
            sorter_output_path = session.pfc_spike_path
            spike_path = aligned_pfc_spike_path
    else:
        exit("Region not supported")

    event_df, trial_df = spike_behavior_pynapple.load_session_tables(session)
    lick_times = spike_behavior_pynapple.build_lick_time_dict(event_df)
    figure_path.mkdir(parents=True, exist_ok=True)
    lick_rate_bin_size = 0.5
    max_time_after_trial_start = 5.0
    trial_type_masks = make_lick_peth_trial_type_masks(trial_df, require_led_time=True)
    conditions_to_plot = {
        "all_valid": None,
        "left_correct": trial_type_masks["left_correct"],
        "right_correct": trial_type_masks["right_correct"],
        "left_incorrect": trial_type_masks["left_incorrect"],
        "right_incorrect": trial_type_masks["right_incorrect"],
        "left_omission": trial_type_masks["left_omission"],
        "right_omission": trial_type_masks["right_omission"],
        "left_switch": trial_type_masks["left_switch"],
        "right_switch": trial_type_masks["right_switch"],
        "left_stay": trial_type_masks["left_stay"],
        "right_stay": trial_type_masks["right_stay"],
    }
    for condition_name, trial_mask in conditions_to_plot.items():
        if trial_mask is not None and not np.any(np.asarray(trial_mask, dtype=bool)):
            print(f"Skipping {condition_name}: no valid trials")
            continue

        for alignment in ("trial_start", "choice"):
            lick_peth_data = build_lick_peth_data(
                trial_df=trial_df,
                lick_times=lick_times,
                alignment=alignment,
                rate_bin_size=lick_rate_bin_size,
                pre_time=2.0,
                post_time=1.0,
                show_led_lines=True,
                max_time_after_trial_start=max_time_after_trial_start,
                trial_mask=trial_mask,
            )
            for lick_layout in ("overlay", "separate"):
                fig, _ = plot_lick_peth(
                    lick_peth_data,
                    lick_layout=lick_layout,
                    show_led_lines=True,
                )
                save_path = figure_path / f"{sess_id_full}_{condition_name}_lick_peth_{alignment}_{lick_layout}.png"
                save_figure_with_message(fig, save_path)
                plt.close(fig)

    aligned_spike_times = spike_behavior_pynapple.load_aligned_spikes(spike_path)
    spike_clusters, cluster_info = spike_behavior_pynapple.load_sorter_metadata(
        sorter_output_path=sorter_output_path,
    )
    spike_behavior_pynapple.validate_aligned_spike_inputs(
        aligned_spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
    )
    region_cluster_ids = spike_behavior_pynapple.select_units_by_channels(
        cluster_info=cluster_info,
        region_channels=region_channels,
    )
    region_cluster_info = cluster_info.loc[cluster_info["cluster_id"].isin(region_cluster_ids)].copy()
    unit_cluster_id = select_first_valid_unit_cluster_id(region_cluster_info)
    unit_spike_group = spike_behavior_pynapple.build_spike_tsgroup(
        spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
        cluster_ids=np.array([unit_cluster_id], dtype=int),
    )
    unit_spikes = unit_spike_group[unit_cluster_id]

    spike_rate_bin_size = 0.1
    for condition_name, trial_mask in conditions_to_plot.items():
        if trial_mask is not None and not np.any(np.asarray(trial_mask, dtype=bool)):
            print(f"Skipping {condition_name} spike plot: no valid trials")
            continue

        for alignment in ("trial_start", "choice"):
            spike_peth_data = build_single_unit_spike_peth_data(
                trial_df=trial_df,
                unit_spikes=unit_spikes,
                unit_cluster_id=unit_cluster_id,
                alignment=alignment,
                rate_bin_size=spike_rate_bin_size,
                pre_time=2.0,
                post_time=1.0,
                show_led_lines=True,
                max_time_after_trial_start=max_time_after_trial_start,
                trial_mask=trial_mask,
            )
            fig, _ = plot_single_unit_spike_peth(
                spike_peth_data,
                show_led_lines=True,
            )
            save_path = (
                figure_path
                / f"{sess_id_full}_{region_name}_unit-{unit_cluster_id}_{condition_name}_spike_peth_{alignment}.png"
            )
            save_figure_with_message(fig, save_path)
            plt.close(fig)


if __name__ == "__main__":
    main()
