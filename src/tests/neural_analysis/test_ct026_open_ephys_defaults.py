from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.neural_analysis import psth_webapp, sync_ephys, unit_spike_loading


def test_unit_spike_loading_examples_use_one_session_identity():
    """Example paths should stay internally consistent without freezing a session."""
    session_home = unit_spike_loading.DEFAULT_SESSION_DATA_HOME

    assert isinstance(session_home, Path)
    assert session_home.name
    assert unit_spike_loading.DEFAULT_SESSION_ID


def test_default_probe_paths_keep_open_ephys_probe_assignments_consistent():
    """PFC should use ProbeA and HPC/V1 ProbeB without freezing example names."""
    session_home = unit_spike_loading.DEFAULT_SESSION_DATA_HOME
    pfc_sorter_path = unit_spike_loading.DEFAULT_PFC_SORTER_OUTPUT_PATH
    hpc_sorter_path = unit_spike_loading.DEFAULT_HPC_SORTER_OUTPUT_PATH

    assert session_home in pfc_sorter_path.parents
    assert session_home in hpc_sorter_path.parents
    assert pfc_sorter_path.parent.name.endswith(".ProbeA")
    assert hpc_sorter_path.parent.name.endswith(".ProbeB")
    assert pfc_sorter_path.name == hpc_sorter_path.name
    assert unit_spike_loading.DEFAULT_PFC_ALIGNED_SPIKE_PATH == (
        session_home
        / "ephys/aligned/aligned_open_ephys/probeA_sync.npz"
    )
    assert unit_spike_loading.DEFAULT_HPC_V1_ALIGNED_SPIKE_PATH == (
        session_home
        / "ephys/aligned/aligned_open_ephys/probeB_sync.npz"
    )
    assert unit_spike_loading.DEFAULT_PFC_LFP_PATH.parent == pfc_sorter_path.parent
    assert unit_spike_loading.DEFAULT_HPC_V1_LFP_PATH.parent == hpc_sorter_path.parent
    assert unit_spike_loading.DEFAULT_PFC_LFP_PATH.name == "lfp.dat"
    assert unit_spike_loading.DEFAULT_HPC_V1_LFP_PATH.name == "lfp.dat"


def test_psth_webapp_browser_root_contains_the_example_session():
    """The path browser should start at the parent of the example session."""
    assert psth_webapp.DEFAULT_BROWSER_ROOT == (
        unit_spike_loading.DEFAULT_SESSION_DATA_HOME.parent
    )


def test_open_ephys_sync_workflow_defaults_to_both_ct026_probes(monkeypatch):
    """The Open Ephys entry point should route ProbeA and ProbeB consistently."""
    calls: list[dict[str, object]] = []
    alignment_notes: list[tuple[Path, float]] = []

    def fake_require_existing_dir(dir_path: Path) -> Path:
        return dir_path

    def fake_write_alignment_note(*, output_root: Path, utc_offset_hours: float) -> None:
        alignment_notes.append((output_root, utc_offset_hours))

    def fake_sync_open_ephys_kilosort_spikes_to_utc(**kwargs):
        calls.append(kwargs)
        probe_name = str(kwargs["probe_name"])
        return (
            pd.DataFrame({"probe_name": [probe_name]}),
            pd.DataFrame({"probe_name": [probe_name]}),
        )

    monkeypatch.setattr(sync_ephys, "_require_existing_dir", fake_require_existing_dir)
    monkeypatch.setattr(sync_ephys.ephys_sync_utils, "write_alignment_note", fake_write_alignment_note)
    monkeypatch.setattr(
        sync_ephys.ephys_sync_utils,
        "sync_open_ephys_kilosort_spikes_to_utc",
        fake_sync_open_ephys_kilosort_spikes_to_utc,
    )

    sync_results = sync_ephys.main_open_ephys_workflow()

    assert [call["probe_name"] for call in calls] == ["ProbeA", "ProbeB"]
    assert set(sync_results) == {"ProbeA", "ProbeB"}
    assert [result[0].at[0, "probe_name"] for result in sync_results.values()] == [
        "ProbeA",
        "ProbeB",
    ]
    ephys_root = Path(calls[0]["output_file"]).parents[2]
    assert alignment_notes == [(ephys_root / "aligned", 0.0)]
    for call in calls:
        probe_name = str(call["probe_name"])
        assert Path(call["kilosort_dir"]).parent.name.endswith(f".{probe_name}")
        assert Path(call["ttl_dir"]).parent.name.endswith(f".{probe_name}")
        assert Path(call["continuous_dir"]).name.endswith(f".{probe_name}")
        assert Path(call["output_file"]).name == f"{probe_name[:1].lower()}{probe_name[1:]}_sync.npz"
