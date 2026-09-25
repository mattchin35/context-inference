"""Legacy spike-behavior API with plotting and direct dispatch retained."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.spike_behavior.loading import (
    SUPPORTED_ANALYSIS_REGIONS,
    Session,
    build_lick_time_dict,
    build_spike_tsgroup,
    load_aligned_spikes,
    load_session_info,
    load_session_tables,
    load_sorter_metadata,
    normalize_region_channels,
    parse_session_id,
    select_units_by_channels,
    validate_aligned_spike_inputs,
)
from src.neural_analysis.spike_behavior.trials import (
    make_trial_type_masks,
    resolve_trial_end,
    summarize_trial_masks,
)
from src.neural_analysis.spike_behavior.binning import (
    bin_licks_to_trial_pynapple,
    bin_region_trials,
    bin_spikes_to_trial_pynapple,
    build_bin_edges,
    collect_condition_classifier_bins,
    make_classifier_bins,
)
from src.neural_analysis.spike_behavior.decoding import (
    cv_decodeability_score,
    evaluate_decoder_on_condition,
    get_decode_target,
    run_base_condition_decoding,
    run_repeated_correct_rewarded_decoder,
    summarize_decoding_results,
    train_single_decoder_with_shuffle_null,
)
from src.neural_analysis.spike_behavior.publication import (
    build_correct_rewarded_decoding_performance_session_table,
    build_state_decodability_session_table,
    make_region_analysis_filename,
    normalize_region_name_for_filename,
    save_correct_rewarded_decoding_performance_session_csv,
    save_state_decodability_session_csv,
)

def _transform_plot_times(
    times: np.ndarray,
    time_mode: str,
    reference_time: float,
    session_start_time: float | None,
) -> np.ndarray:
    """
    Transform timestamps for trial-inspection plots.

    Parameters
    ----------
    times : np.ndarray
        One-dimensional float array with shape ``(n_times,)`` containing timestamps in UTC Unix
        seconds.
    time_mode : str
        Plotting time base. Supported values are ``"utc"``, ``"session"``, and ``"event"``.
    reference_time : float
        Reference event timestamp for the selected trial, in UTC Unix seconds.
    session_start_time : float | None
        Session start timestamp in UTC Unix seconds. Required when ``time_mode == "session"``.

    Returns
    -------
    np.ndarray
        One-dimensional float array with shape ``(n_times,)``. Units are seconds in the requested
        plotting frame.
    """

    if time_mode == "utc":
        return times
    if time_mode == "session":
        if session_start_time is None:
            raise ValueError("session_start_time is required when time_mode='session'.")
        return times - float(session_start_time)
    if time_mode == "event":
        return times - reference_time
    raise ValueError("time_mode must be one of {'utc', 'session', 'event'}.")

def plot_trial_raster(
    region_spike_group: nap.TsGroup,
    trial_df: pd.DataFrame,
    trial_ix: int,
    lick_times: Mapping[str, nap.Ts] | None = None,
    event: str = "choice_time",
    time_mode: str = "utc",
    session_start_time: float | None = None,
    pre_time: float = 2.0,
    post_time: float = 2.0,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one trial's spikes and licks in an interactive two-panel raster view.

    Parameters
    ----------
    region_spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike timestamps must be in UTC Unix
        seconds and all units share the same time base.
    trial_df : pd.DataFrame
        Trial table with one row per trial. The selected trial must contain the requested
        alignment event column. Event timestamps are UTC Unix seconds.
    trial_ix : int
        Integer row index into ``trial_df`` selecting one trial to display.
    lick_times : Mapping[str, nap.Ts] | None, optional
        Mapping containing optional ``"right_entry"`` and ``"left_entry"`` lick series. Each
        series is one-dimensional and uses UTC Unix seconds.
    event : str, optional
        Alignment event column name. Supported values are ``"choice_time"`` and ``"start_time"``.
    time_mode : str, optional
        X-axis time base. Supported values are ``"utc"``, ``"session"``, and ``"event"``.
    session_start_time : float | None, optional
        Session start timestamp in UTC Unix seconds. Required when ``time_mode == "session"``.
    pre_time : float, optional
        Seconds before the alignment event to include in the plot window.
    post_time : float, optional
        Seconds after the alignment event to include in the plot window.
    show : bool, optional
        If ``True``, display the figure immediately with ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        ``(figure, axes)`` where ``axes`` has shape ``(2,)``. ``axes[0]`` is the spike raster axis
        and ``axes[1]`` is the lick raster axis.
    """

    if event not in {"choice_time", "start_time"}:
        raise ValueError("event must be 'choice_time' or 'start_time'.")
    if time_mode not in {"utc", "session", "event"}:
        raise ValueError("time_mode must be one of {'utc', 'session', 'event'}.")
    if trial_ix not in trial_df.index:
        raise ValueError(f"trial_ix {trial_ix} is not present in trial_df.")
    if time_mode == "session" and session_start_time is None:
        raise ValueError("session_start_time is required when time_mode='session'.")

    trial_row = trial_df.loc[trial_ix]
    reference_time = pd.to_numeric(pd.Series([trial_row[event]]), errors="coerce").iloc[0]
    if pd.isna(reference_time):
        raise ValueError(f"Trial {trial_ix} has no valid {event} timestamp.")
    reference_time = float(reference_time)

    window_start = reference_time - float(pre_time)
    window_end = reference_time + float(post_time)
    plot_interval = nap.IntervalSet(start=[window_start], end=[window_end])

    restricted_spike_group = region_spike_group.restrict(plot_interval)
    cluster_ids = list(restricted_spike_group.keys())
    if cluster_ids:
        spike_tsd = restricted_spike_group.to_tsd(np.arange(len(cluster_ids), dtype=float))
        spike_x = _transform_plot_times(
            times=np.asarray(spike_tsd.index.to_numpy(), dtype=float),
            time_mode=time_mode,
            reference_time=reference_time,
            session_start_time=session_start_time,
        )
        spike_y = spike_tsd.values.astype(float)
    else:
        spike_x = np.array([], dtype=float)
        spike_y = np.array([], dtype=float)

    if time_mode == "event":
        reference_x = 0.0
    else:
        reference_x = float(
            _transform_plot_times(
                times=np.array([reference_time], dtype=float),
                time_mode=time_mode,
                reference_time=reference_time,
                session_start_time=session_start_time,
            )[0]
        )

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), height_ratios=[3, 1])
    spike_axis, lick_axis = axes

    spike_axis.scatter(spike_x, spike_y, marker="|", color="black", s=80)
    spike_axis.axvline(reference_x, color="tab:red", linestyle="--", linewidth=1.5)
    spike_axis.set_ylabel("Unit")
    spike_axis.set_title(f"Trial {trial_ix} aligned to {event}")
    if cluster_ids:
        spike_axis.set_yticks(np.arange(len(cluster_ids), dtype=float))
        spike_axis.set_yticklabels([str(cluster_id) for cluster_id in cluster_ids])

    lick_axis.axvline(reference_x, color="tab:red", linestyle="--", linewidth=1.5)
    lick_axis.set_ylabel("Lick")

    lick_positions = {"right_entry": 1.0, "left_entry": 0.0}
    lick_colors = {"right_entry": "tab:blue", "left_entry": "tab:orange"}
    if lick_times is not None:
        for lick_name in ("right_entry", "left_entry"):
            if lick_name not in lick_times:
                continue
            restricted_licks = lick_times[lick_name].restrict(plot_interval)
            transformed_licks = _transform_plot_times(
                times=np.asarray(restricted_licks.index.to_numpy(), dtype=float),
                time_mode=time_mode,
                reference_time=reference_time,
                session_start_time=session_start_time,
            )
            lick_axis.scatter(
                transformed_licks,
                np.full(transformed_licks.shape, lick_positions[lick_name], dtype=float),
                marker="|",
                color=lick_colors[lick_name],
                s=120,
                label=lick_name,
            )

    lick_axis.set_yticks([0.0, 1.0])
    lick_axis.set_yticklabels(["left", "right"])
    if lick_times is not None:
        handles, labels = lick_axis.get_legend_handles_labels()
        if handles:
            lick_axis.legend(loc="upper right")

    if time_mode == "utc":
        lick_axis.set_xlabel("Time (UTC Unix s)")
    elif time_mode == "session":
        lick_axis.set_xlabel("Time Since Session Start (s)")
    else:
        lick_axis.set_xlabel(f"Time From {event} (s)")

    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes

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
    figure_path = session_data_home / "figures"

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
                    "25  24  23  22  21  20  19  18  17  16  15  14  13  12  11  10 9  8 "
                    "7   6   5   4   3   2   1   0")
    pfc_channels = np.sort(np.array(list(map(int, [val for val in _pfc_channels.split(" ") if val != '']))))

    mouse, date, timestamp = parse_session_id(sess_id_full)
    session_info_path = raw_behavior_folder / f"{sess_id_full}_session_info.pkl"

    session = Session(
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
        session_info=load_session_info(session_info_path)
        if session_info_path.exists()
        else None,
        pfc_spike_path=pfc_spike_path,
        hpc_spike_path=hpc_spike_path,
    )

    region_name = "PFC"
    if region_name in SUPPORTED_ANALYSIS_REGIONS:
        if region_name == "HPC":
            region_channels = normalize_region_channels(hpc_channels)
            sorter_output_path = session.hpc_spike_path
            spike_path = aligned_hpc_spike_path
        elif region_name == "V1":
            region_channels = normalize_region_channels(v1_channels)
            sorter_output_path = session.hpc_spike_path
            spike_path = aligned_hpc_spike_path
        elif region_name == "PFC":
            region_channels = normalize_region_channels(pfc_channels)
            sorter_output_path = session.pfc_spike_path
            spike_path = aligned_pfc_spike_path
    else:
        exit("Region not supported")

    event_df, trial_df = load_session_tables(session)
    lick_times = build_lick_time_dict(event_df)
    aligned_spike_times = load_aligned_spikes(spike_path)
    spike_clusters, cluster_info = load_sorter_metadata(
        sorter_output_path=sorter_output_path,
    )
    validate_aligned_spike_inputs(
        aligned_spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
    )

    region_cluster_ids = select_units_by_channels(cluster_info, region_channels=region_channels)
    if region_cluster_ids.size == 0:
        raise RuntimeError(
            f"No {region_name} units were found in the configured sorter output for channels {region_channels.tolist()}."
        )

    region_spike_group = build_spike_tsgroup(
        spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
        cluster_ids=region_cluster_ids,
    )
    region_spike_bins = bin_region_trials(
        trial_df=trial_df,
        spike_group=region_spike_group,
        cluster_ids=region_cluster_ids,
        bin_size=0.5,
        pre_time=2.0,
        post_time=2.0,
    )
    trial_masks = make_trial_type_masks(trial_df)
    condition_names = [
        "correct_rewarded",
        "incorrect",
        "omission",
        "switch",
        "stay",
        "omission_switch",
        "omission_stay",
        "incorrect_switch",
        "incorrect_stay",
    ]
    choice_windows = {
        "pre_choice": (-0.5, 0.0),
        "post_choice": (0.0, 0.5),
    }
    mask_summary_df, condition_trial_indices = summarize_trial_masks(
        trial_masks=trial_masks,
        condition_names=condition_names,
    )
    classifier_bins_by_condition = collect_condition_classifier_bins(
        region_trial_binned=region_spike_bins,
        trial_df=trial_df,
        trial_masks=trial_masks,
        condition_names=condition_names,
        windows=choice_windows,
        event="choice_time",
    )
    decoding_results = run_base_condition_decoding(
        region_trial_binned=region_spike_bins,
        trial_df=trial_df,
        target=decode_target,
        cv=5,
        n_permutations=100,
        n_shuffles=1000,
        random_state=42,
    )
    repeated_decoding_csv_path: Path | None = processed_data_path
    if decode_target == "state_int":
        repeated_decoding_results = run_repeated_correct_rewarded_decoder(
            region_trial_binned=region_spike_bins,
            trial_df=trial_df,
            target=decode_target,
            n_decoder_runs=n_decoder_runs,
            n_shuffles=1000,
            random_state=42,
        )
        repeated_decoding_table = build_correct_rewarded_decoding_performance_session_table(
            session=session,
            region_name=region_name,
            training_results=repeated_decoding_results["training_results"],
            generalization_results=repeated_decoding_results["generalization_results"],
        )
        repeated_decoding_csv_path = save_correct_rewarded_decoding_performance_session_csv(
            output_dir=session.processed_data_path,
            decoding_performance_table=repeated_decoding_table,
            region_name=region_name,
        )
    state_decodability_csv_path: Path | None = processed_data_path
    if decode_target == "state_int":
        state_decodability_table = build_state_decodability_session_table(
            session=session,
            region_name=region_name,
            decodeability_results=decoding_results["decodeability_results"],
        )
        state_decodability_csv_path = save_state_decodability_session_csv(
            output_dir=session.processed_data_path,
            state_decodability_table=state_decodability_table,
            region_name=region_name,
        )

    first_trial_end = resolve_trial_end(trial_df.iloc[0])
    first_trial_licks, _ = bin_licks_to_trial_pynapple(
        trial_start=float(trial_df.iloc[0]["start_time"]),
        trial_end=first_trial_end,
        lick_times=lick_times,
        bin_size=0.5,
        pre_time=2.0,
        post_time=2.0,
    )

    print(f"Session: {session.sess_id_full}")
    print(f"Region: {region_name}")
    print(f"Region units: {region_cluster_ids.size}")
    print(f"Trials binned: {len(region_spike_bins)}")
    print(f"First trial spike-bin shape: {region_spike_bins[0]['binned_spikes'].shape}")
    print(f"First trial lick-bin shape: {first_trial_licks.shape}")
    print(f"Decode target: {decode_target}")
    if state_decodability_csv_path is not None:
        print(f"State decodability CSV: {state_decodability_csv_path}")
    if repeated_decoding_csv_path is not None:
        print(f"Repeated decoder performance CSV: {repeated_decoding_csv_path}")
    print("Trial condition counts:")
    for _, summary_row in mask_summary_df.iterrows():
        condition_name = str(summary_row["condition"])
        condition_count = int(summary_row["n_trials"])
        trial_indices = condition_trial_indices[condition_name].tolist()
        print(f"  {condition_name}: {condition_count} trials {trial_indices}")

    print("Classifier bin summary:")
    for condition_name in condition_names:
        for window_name in choice_windows:
            collected_entry = classifier_bins_by_condition[(condition_name, window_name)]
            if collected_entry["status"] == "ok":
                print(
                    f"  {condition_name} {window_name}: spike_bins {collected_entry['spike_bins'].shape}, "
                    f"labels {collected_entry['state_bins'].shape[0]}"
                )
            else:
                print(f"  {condition_name} {window_name}: failed ({collected_entry['reason']})")
    print("CV decodeability summary:")
    cv_summary_df = summarize_decoding_results(
        decoding_results["decodeability_results"],
        value_columns=["cv_score", "cv_pvalue"],
    )
    for _, summary_row in cv_summary_df.iterrows():
        if summary_row["status"] == "ok":
            print(
                f"  {summary_row['condition']} {summary_row['window']}: "
                f"cv_score={summary_row['cv_score']:.3f}, cv_pvalue={summary_row['cv_pvalue']:.3f}"
            )
        else:
            print(
                f"  {summary_row['condition']} {summary_row['window']}: "
                f"failed ({summary_row['reason']})"
            )
    print("One-shot decoder summary:")
    training_result = decoding_results["training_result"]
    if training_result["status"] == "ok":
        print(
            "  trained on correct_rewarded post_choice: "
            f"train_acc={training_result['train_accuracy']:.3f}, "
            f"test_acc={training_result['test_accuracy']:.3f}, "
            f"shuffle_p={training_result['shuffle_pvalue']:.3f}"
        )
    else:
        print(f"  training failed ({training_result['reason']})")
    generalization_summary_df = summarize_decoding_results(
        decoding_results["generalization_results"],
        value_columns=["test_accuracy"],
    )
    for _, summary_row in generalization_summary_df.iterrows():
        if summary_row["status"] == "ok":
            print(
                f"  {summary_row['condition']} {summary_row['window']}: "
                f"accuracy={summary_row['test_accuracy']:.3f}"
            )
        else:
            print(
                f"  {summary_row['condition']} {summary_row['window']}: "
                f"failed ({summary_row['reason']})"
            )

    plot_trial_ix: int | None = None
    plot_event = "choice_time"
    plot_time_mode = "session"  # utc, event, or session
    plot_pre_time = 2.0
    plot_post_time = 2.0
    plot_session_start_time: float | None = trial_df['trial_time_since_start'].min() if plot_time_mode == "session" else None
    if plot_trial_ix is not None:
        plot_trial_raster(
            region_spike_group=region_spike_group,
            trial_df=trial_df,
            trial_ix=plot_trial_ix,
            lick_times=lick_times,
            event=plot_event,
            time_mode=plot_time_mode,
            session_start_time=plot_session_start_time,
            pre_time=plot_pre_time,
            post_time=plot_post_time,
        )


if __name__ == "__main__":
    main()
