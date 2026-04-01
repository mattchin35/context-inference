from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
import src.external_tools.readSGLX as readSGLX
import pickle as pkl
import matplotlib.pyplot as plt
from icecream import ic

# p = output_path / ('{}_spikes_trial_binned.pkl'.format(sess_id_full))
# with p.open('wb') as f:
#     pkl.dump(event_df, f)
# p = output_path / ('{}_licks_trial_binned.pkl'.format(sess_id_full))
# with p.open('wb') as f:
#     pkl.dump(event_df, f)

experiment_folder = Path('C:/Users/mattc/EinsteinMed Dropbox/Matthew Chin/phd_data/remotework/EXPERIMENTS/')
behavior_data_path = experiment_folder / 'raw_behavior_data'
ephys_data_path = experiment_folder / 'processed_ephys_data'
processed_data_path = experiment_folder / 'processed_data'

current_mouse = 'CT010'
current_date_behavior = '2025-08-18'
current_date_ephys = ''.join(current_date_behavior.split('-'))  # remove dashes for ephys folder name
sess_timestamp = '125111'
sess_id = current_mouse + '_' + current_date_behavior
sess_id_full = current_mouse + '_' + current_date_behavior + '_' + sess_timestamp

behavior_session_path = behavior_data_path / sess_id_full
session_info_path = behavior_session_path / '{}_session_info.pkl'.format(sess_id_full)
output_path = processed_data_path / current_mouse / sess_id_full

recording_path = ephys_data_path.joinpath('{}_{}_catgt/catgt_run0_g0/run0_g0_imec0'.format(current_mouse, current_date_ephys))
ap_file = recording_path.joinpath('run0_g0_tcat.imec0.ap.bin')
lfp_file = recording_path / 'run0_g0_tcat.imec0.lf.bin'

sorter_output = recording_path / 'Kilosort2.5_2025-08-29_173906'
spike_clusters = sorter_output / 'spike_clusters.npy'
spike_times = sorter_output / 'spike_times.npy'
channel_map = sorter_output / 'channel_map.npy'
channel_positions = sorter_output / 'channel_positions.npy'
cluster_group = sorter_output / 'cluster_group.tsv'
cluster_info = sorter_output / 'cluster_info.tsv'
cluster_kslabel = sorter_output / 'cluster_KSLabel.tsv'

imec_meta = readSGLX.readMeta(ap_file)
imec_srate = readSGLX.SampRate(imec_meta)
spike_clusters = np.load(spike_clusters, allow_pickle=True)
spike_times = np.squeeze(np.load(spike_times, allow_pickle=True)) / imec_srate  # raw times are in SAMPLES, must be converted to seconds
channel_map = np.load(channel_map, allow_pickle=True)
channel_positions = np.load(channel_positions, allow_pickle=True)
cluster_group = pd.read_csv(cluster_group, sep='\t')
cluster_info = pd.read_csv(cluster_info, sep='\t')
cluster_kslabel = pd.read_csv(cluster_kslabel, sep='\t')

# n_shank, shank_width, shank_pitch, shank_ind, x, y, connected = coordsSGLX.geomMapToGeom(ap0_meta)
# coords = np.stack((x,y), axis=1)
# geometric_sort_imec0 = np.argsort(coords[:,1])  # sort by y coordinate; for now, sites are on one shank, so this is sufficient
# sorted_coords_imec0 = coords[geometric_sort_imec0]
geometric_sort_imec0 = np.argsort(channel_positions[:,1])
sorted_coords_imec0 = channel_positions[geometric_sort_imec0]

ch_hpc = np.arange(192, 240)  # channels 192-239 are HPC
ch_md = geometric_sort_imec0[:50]
ch_v1 = geometric_sort_imec0[-50:]

ic(geometric_sort_imec0)
ic(ch_hpc)
ic(ch_md)
ic(ch_v1)

# gather spikes by channel group
hpc_spikes = np.zeros_like(spike_times, dtype=bool)
md_spikes = np.zeros_like(spike_times, dtype=bool)
v1_spikes = np.zeros_like(spike_times, dtype=bool)
for i, ch in enumerate(spike_clusters):
    hpc_spikes = hpc_spikes | (spike_clusters == i)
    md_spikes = md_spikes | (spike_clusters == i)
    v1_spikes = v1_spikes | (spike_clusters == i)

hpc_spike_times = spike_times[hpc_spikes]
hpc_spike_clusters = spike_clusters[hpc_spikes]
md_spike_times = spike_times[md_spikes]
md_spike_clusters = spike_clusters[md_spikes]
v1_spike_times = spike_times[v1_spikes]
v1_spike_clusters = spike_clusters[v1_spikes]

