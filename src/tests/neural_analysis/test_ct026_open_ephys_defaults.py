from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.neural_analysis import psth_webapp, sync_ephys, unit_spike_loading


def test_unit_spike_loading_defaults_point_to_ct026_august_2026_open_ephys_session():
    """Default session constants should target the CT026 Open Ephys session used for development."""
    expected_session_home = Path(
        "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference"
    )

    assert unit_spike_loading.DEFAULT_SESSION_DATA_HOME == expected_session_home
    assert unit_spike_loading.DEFAULT_SESSION_ID == "CT026_2026-08-01_130853"


def test_default_probe_paths_match_ct026_open_ephys_probe_assignments():
    """PFC should default to ProbeA and HPC/V1 should default to ProbeB for the CT026 session."""
    assert unit_spike_loading.DEFAULT_PFC_SORTER_OUTPUT_PATH == (
        unit_spike_loading.DEFAULT_SESSION_DATA_HOME
        / "ephys/derived/Record_Node_101_Neuropix-PXI-110.ProbeA/kilosort4"
    )
    assert unit_spike_loading.DEFAULT_HPC_SORTER_OUTPUT_PATH == (
        unit_spike_loading.DEFAULT_SESSION_DATA_HOME
        / "ephys/derived/Record_Node_101_Neuropix-PXI-110.ProbeB/kilosort4"
    )
    assert unit_spike_loading.DEFAULT_PFC_ALIGNED_SPIKE_PATH == (
        unit_spike_loading.DEFAULT_SESSION_DATA_HOME
        / "ephys/aligned/aligned_open_ephys/probeA_sync.npz"
    )
    assert unit_spike_loading.DEFAULT_HPC_V1_ALIGNED_SPIKE_PATH == (
        unit_spike_loading.DEFAULT_SESSION_DATA_HOME
        / "ephys/aligned/aligned_open_ephys/probeB_sync.npz"
    )
    assert unit_spike_loading.DEFAULT_PFC_LFP_PATH == (
        unit_spike_loading.DEFAULT_SESSION_DATA_HOME
        / "ephys/derived/Record_Node_101_Neuropix-PXI-110.ProbeA/lfp.dat"
    )
    assert unit_spike_loading.DEFAULT_HPC_V1_LFP_PATH == (
        unit_spike_loading.DEFAULT_SESSION_DATA_HOME
        / "ephys/derived/Record_Node_101_Neuropix-PXI-110.ProbeB/lfp.dat"
    )


def test_psth_webapp_browser_root_defaults_to_ct026_animal_folder():
    """The sidebar path browser should start at the animal folder containing the default session."""
    assert psth_webapp.DEFAULT_BROWSER_ROOT == Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026")


def test_open_ephys_sync_workflow_defaults_to_both_ct026_probes(monkeypatch):
    """The Open Ephys sync entry point should process the CT026 PFC and HPC probes by default."""
    processed_probes = []

    def fake_require_existing_dir(dir_path: Path) -> Path:
        return dir_path

    def fake_write_alignment_note(*, output_root: Path, utc_offset_hours: float) -> None:
        assert output_root.name == "aligned"
        assert utc_offset_hours == 0.0

    def fake_sync_open_ephys_kilosort_spikes_to_utc(**kwargs):
        processed_probes.append(kwargs["probe_name"])
        assert kwargs["kilosort_dir"].name == "kilosort4"
        assert kwargs["ttl_dir"].name == "TTL"
        assert kwargs["continuous_dir"].name == f"Neuropix-PXI-110.{kwargs['probe_name']}"
        assert kwargs["output_file"].name == f"{kwargs['probe_name'][:1].lower()}{kwargs['probe_name'][1:]}_sync.npz"
        return pd.DataFrame({"spike_utc_unix": []}), pd.DataFrame({"irig_utc_unix": []})

    monkeypatch.setattr(sync_ephys, "_require_existing_dir", fake_require_existing_dir)
    monkeypatch.setattr(sync_ephys.ephys_sync_utils, "write_alignment_note", fake_write_alignment_note)
    monkeypatch.setattr(
        sync_ephys.ephys_sync_utils,
        "sync_open_ephys_kilosort_spikes_to_utc",
        fake_sync_open_ephys_kilosort_spikes_to_utc,
    )

    sync_results = sync_ephys.main_open_ephys_workflow()

    assert processed_probes == ["ProbeA", "ProbeB"]
    assert set(sync_results) == {"ProbeA", "ProbeB"}
