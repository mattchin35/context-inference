To start cleanly, I’d need decisions in four areas.

# Data Loading
• Which session should the first viewer target?
• Should it load from the same hardcoded paths in spike_behavior_pynapple.py, or from a selectable session path?
• Should it use raw spike files each time, or cached/precomputed objects if available?
• Do you want one session at a time, or eventually cross-session browsing?
My recommendation: start with one hardcoded session, same style as your current scripts, then generalize.

## Responses
Target session: 
multi_session_save_path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis")
session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference")
sess_id_full = "CT014_2025-12-23_163505"
pfc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec0/Kilosort2.5.2_2026-03-19_180103/sorting_mchin_20260330"
hpc_spike_path = session_data_home / "ephys/catgt/catgt_run0_g0/run0_g0_imec1/Kilosort2.5.2_2026-03-19_183540/sorting_mchin_20260331"

- There should be a selectable session path and spike path.
- Spike paths should be used as in spike_behavior_pynapple.py.
- For now just do one session at a time.

# Unit Selection
• Should units be selected by cluster_id?
• Should the UI show metadata too, like channel, region, group, firing rate?
• Should units be filtered by region: HPC, V1, PFC?
• Should “good” and “mua” both be included?
My recommendation: select by cluster_id, show channel/group metadata, and filter by region.

## Responses
- units should be select cluster id. 
- Show the metadata, and allow me to limit clusters to a select group of channels as in spike_behavior_pynapple.py.
- Allow region filtering, and setting of regions by channel numbers.
- Both good and mua should be included, but the UI should show the quality label so I can filter by it if I want. Do
not exclusively filter to good units - some sessions were sorted only with mua/noise labels to minimize manual 
processing time.

# Plot Views 
Pick the initial plot types. For example:
• single-unit trial raster aligned to choice_time
• single-unit trial raster aligned to start_time
• PSTH aligned to choice_time
• raster split by trial condition
• lick plot for the same selected trials
• combined spike + lick trial view

My recommendation: start with:
1. trial raster for one unit
2. PSTH for one unit
3. spike + lick view for one selected trial 

## Responses
- trial raster for one unit and PSTH should be plotted and updated together, alignable to choice_time or start_time
- multiple unit spikes + lick view will be a separate plot that shows one trial at a time. We will definitely do this but not 
in the first pass to get the basic webapp working

# Trial Filtering / Paging 
This is probably the most important usability piece.
Possible controls:
• condition: all, correct rewarded, incorrect, omission, switch, stay
• inferred state / block state
• action: left/right
• reward: rewarded/unrewarded
• trial index range
• page size: 25, 50, 100 trials
• alignment event: start, choice, reward
My recommendation: start with condition, alignment event, and pagination. Add state/action filters after the first version works.

## Responses
- Do condition, action, page size, alignment for the first version. 

# Plot Backend Streamlit can display either Matplotlib or Plotly.
• Matplotlib: easiest because it fits your existing code style.
• Plotly: better zoom/hover, but likely more adaptation.
My recommendation: use Matplotlib first. Use Streamlit as a browser-based control panel around normal Python plotting functions. 

## Responses
Use matplotlib first.

# App Structure 

I would avoid putting everything in one giant Streamlit script.
A clean split would be:
src/neural_analysis/unit_spike_viewer.py        # Streamlit UI
src/neural_analysis/unit_spike_plotting.py      # plotting functions
src/neural_analysis/unit_spike_loading.py       # session loading/cache helpers
Or, for a first pass, one Streamlit file plus reused functions from spike_behavior_pynapple.py.

## Responses
Splitting the files is fine.

# Caching 
Streamlit needs caching or it will reload spike files too often.
Likely use:
@st.cache_data
def load_trial_tables(...):
    ...

@st.cache_resource
def load_spike_group(...):
    ...

Decision needed:
• Is it okay if cached data persists until the app restarts?
• Should there be a “reload data” button?

## reponses
- cached data for one session can persist until the app restarts.
- provide a "reload data" or cache-clearing button of some kind

# Output Goals 
Do you want the app only for inspection, or should it also save selected plots?
Possible features:
• save current plot as PNG
• record notes per unit
• mark units as interesting
• export selected unit IDs 
I’d defer this unless you already know you need it.

## Responses
- I definitely want to be able to save selected plots. The other features are good ideas but not necessary for the first version.

# Minimum First Version 
The smallest useful version would be:
• Hardcoded session path.
• Region selector.
• Unit selector.
• Condition selector.
• Alignment selector.
• Trial page selector.
• One raster plot.
• Optional lick overlay/panel.

Before implementation, I’d ask you to choose:
1. First target session. - start with CT014_2025-12-23 
2. Initial region: HPC, V1, or PFC. - Do HPC first
3. First plot type: raster only, PSTH only, or raster + lick panel.
4. Whether to use Matplotlib first - use matplotlib