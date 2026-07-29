"""Entry points for synchronizing IMEC spikes and NI events to UTC."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.neural_analysis import ephys_sync_utils


def _require_existing_file(file_path: Path) -> Path:
    """Validate that an expected input file exists.

    Args:
        file_path: File path expected to exist on disk.

    Returns:
        Path: The same path after validation.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required input file not found: {file_path}")
    return file_path


def _require_existing_dir(dir_path: Path) -> Path:
    """Validate that an expected input directory exists.

    Args:
        dir_path: Directory path expected to exist on disk.

    Returns:
        Path: The same path after validation.

    Raises:
        FileNotFoundError: If ``dir_path`` does not exist or is not a
        directory.
    """
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Required input directory not found: {dir_path}")
    return dir_path


def main_ni_only(
    ni_file: Path | str = Path(
        # "/home/matt/Documents/EXPERIMENTS/contextProjectData/test_runs/irig_neurokairos_20260408/run0_g0/run0_g0_t0.nidq.bin"
        "/home/matt/Documents/EXPERIMENTS/contextProjectData/test_runs/irig_test_20260707/run0_g0/run0_g0_t0.nidq.bin"
    ),
    digital_word: int = 0,
    irig_line: int = 0,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a lightweight NI-only IRIG decode for signal debugging.

    Args:
        ni_file: NI ``.nidq.bin`` file path.
        digital_word: NI digital word index.
        irig_line: NI digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Pulse-level debug dataframe.
            - Frame-level debug dataframe.
    """
    ni_file = _require_existing_file(Path(ni_file))
    pulse_df, frame_df = ephys_sync_utils.decode_ni_irig_debug(
        ni_file=ni_file,
        digital_word=digital_word,
        irig_line=irig_line,
        bit_period_s=bit_period_s,
        utc_offset_hours=utc_offset_hours,
    )

    bit_counts = pulse_df["irig_bit"].value_counts(dropna=False).to_dict()
    print(f"Decoded {pulse_df.shape[0]} IRIG pulses from {ni_file}")
    print(f"Decoded {frame_df.shape[0]} IRIG frames")
    print(f"Bit counts: {bit_counts}")
    if not pulse_df.empty:
        print(pulse_df.head(5).to_string(index=False))
    if not frame_df.empty:
        print(frame_df.head(5).to_string(index=False))
    return pulse_df, frame_df


def main_treadmill_serial() -> None:
    """Placeholder for treadmill serial IRIG workflow entry point."""
    file_path = Path(
        "/home/matt/Documents/EXPERIMENTS/contextProjectData/test_runs/irig_test_20260707/CoolTerm Capture (Untitled_0) 2026-07-07 12-06-51-676.txt"
    )
    _ = file_path


def main_workflow() -> None:
    """Run the hardcoded CT014 ephys synchronization workflow."""
    session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251211_latentInference")
    sess_id_full: str = "CT014_2025-12-11_134311"
    output_root = session_data_home / "ephys" / "aligned"
    sorting0_output_name = "Kilosort2.5.2_2026-04-27_142633"
    sorting1_output_name = "Kilosort2.5.2_2026-04-27_145457"
    ni_event_lines = (2, 3)
    # Retained until the lab setup has a fully reliable IRIG/recording offset.
    utc_offset_hours = 1

    raw_ephys_folder = session_data_home / "ephys" / "raw" / "run0_g0"
    catgt_ephys_folder = session_data_home / "ephys" / "catgt" / "catgt_run0_g0"
    imec0_folder = catgt_ephys_folder / "run0_g0_imec0"
    ap0_file = _require_existing_file(imec0_folder / "run0_g0_tcat.imec0.ap.bin")
    sorting_output0 = imec0_folder / sorting0_output_name
    spike_times_file0 = _require_existing_file(sorting_output0 / "spike_times.npy")

    imec1_folder = catgt_ephys_folder / "run0_g0_imec1"
    sorting_output1 = imec1_folder / sorting1_output_name
    ap1_file = _require_existing_file(imec1_folder / "run0_g0_tcat.imec1.ap.bin")
    spike_times_file1 = _require_existing_file(sorting_output1 / "spike_times.npy")
    
    ni_file = _require_existing_file(raw_ephys_folder / "run0_g0_t0.nidq.bin")
    imec_output_dir = output_root / "aligned_imec"
    nidaq_output_dir = output_root / "aligned_nidaq"
    
    ephys_sync_utils.write_alignment_note(output_root=output_root, utc_offset_hours=utc_offset_hours)
    
    spike_df, imec_irig_df = ephys_sync_utils.sync_imec_spikes_to_utc(
        imec_ap_file=ap0_file,
        spike_times_npy=spike_times_file0,
        output_file=imec_output_dir / "imec0_sync.npz",
        digital_word=0,
        irig_line=6,
        utc_offset_hours=utc_offset_hours,
    )
    print(
        f"{sess_id_full} imec0: mapped {spike_df.shape[0]} spikes using "
        f"{imec_irig_df.shape[0]} IRIG rising edges."
    )

    spike_df, imec_irig_df = ephys_sync_utils.sync_imec_spikes_to_utc(
        imec_ap_file=ap1_file,
        spike_times_npy=spike_times_file1,
        output_file=imec_output_dir / "imec1_sync.npz",
        digital_word=0,
        irig_line=6,
        utc_offset_hours=utc_offset_hours,
    )
    print(
        f"{sess_id_full} imec1: mapped {spike_df.shape[0]} spikes using "
        f"{imec_irig_df.shape[0]} IRIG rising edges."
    )

    for event_line in ni_event_lines:
        rising_df, ni_irig_df = ephys_sync_utils.sync_ni_rising_edges_to_utc(
            ni_file=ni_file,
            event_line=int(event_line),
            output_file=nidaq_output_dir / f"line{int(event_line)}_sync.npz",
            digital_word=0,
            irig_line=0,
            utc_offset_hours=utc_offset_hours,
        )
        print(
            f"{sess_id_full} nidaq line {int(event_line)}: mapped {rising_df.shape[0]} rising edges using "
            f"{ni_irig_df.shape[0]} IRIG rising edges."
        )


def main_open_ephys_workflow() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Run the hardcoded CT026 Open Ephys Kilosort spike synchronization.

    Returns:
        dict[str, tuple[pd.DataFrame, pd.DataFrame]]: Mapping from probe name
        to ``(spike_df, irig_df)``. Spike samples are preserved in
        Kilosort-relative and Open Ephys global sample coordinates.
    """
    session_path = Path(
        "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260727_alternating_latent"
    )
    raw_recording_name = "2026-07-27_14-37-43"
    record_node_name = "Record Node 101"
    experiment_name = "experiment1"
    recording_name = "recording1"
    derived_record_node_name = "Record_Node_101"
    processor_prefix = "Neuropix-PXI-100."
    probe_names = ("ProbeA",)
    kilosort_subdir = "kilosort4"
    irig_line = 0
    bit_period_s = 1.0
    utc_offset_hours = 0.0

    raw_recording_dir = (
        session_path
        / "ephys"
        / "raw"
        / raw_recording_name
        / record_node_name
        / experiment_name
        / recording_name
    )
    derived_root = session_path / "ephys" / "derived"
    output_dir = session_path / "ephys" / "aligned" / "aligned_open_ephys"

    ephys_sync_utils.write_alignment_note(
        output_root=session_path / "ephys" / "aligned",
        utc_offset_hours=utc_offset_hours,
    )

    sync_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for probe_name in probe_names:
        processor_name = f"{processor_prefix}{probe_name}"
        kilosort_dir = _require_existing_dir(
            derived_root / f"{derived_record_node_name}_{processor_name}" / kilosort_subdir
        )
        ttl_dir = _require_existing_dir(raw_recording_dir / "events" / processor_name / "TTL")
        continuous_dir = _require_existing_dir(raw_recording_dir / "continuous" / processor_name)
        output_name = f"{probe_name[:1].lower()}{probe_name[1:]}_sync.npz"

        spike_df, irig_df = ephys_sync_utils.sync_open_ephys_kilosort_spikes_to_utc(
            kilosort_dir=kilosort_dir,
            ttl_dir=ttl_dir,
            continuous_dir=continuous_dir,
            output_file=output_dir / output_name,
            probe_name=probe_name,
            line=irig_line,
            bit_period_s=bit_period_s,
            utc_offset_hours=utc_offset_hours,
        )
        print(
            f"{probe_name}: mapped {spike_df.shape[0]} spikes using "
            f"{irig_df.shape[0]} IRIG rising edges."
        )
        sync_results[probe_name] = (spike_df, irig_df)

    return sync_results


if __name__ == "__main__":
    # main_workflow()
    # main_ni_only()
    main_open_ephys_workflow()
