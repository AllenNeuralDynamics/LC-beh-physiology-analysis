# %%
# %matplotlib inline
import os
import sys

from scipy.sparse import data
sys.path.append('/root/capsule/aind-beh-ephys-analysis/code/beh_ephys_analysis')
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from utils.beh_functions import session_dirs, get_history_from_nwb, get_unit_tbl
from utils.plot_utils import shiftedColorMap, template_reorder, plot_raster_bar,merge_pdfs, combine_pdf_big
from session_preprocessing import ephys_opto_preprocessing
from opto_tagging import opto_plotting_session
from open_ephys.analysis import Session
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
from aind_ephys_utils import align
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils.beh_functions import *

from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session_nwb
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session, plot_foraging_session_nwb
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
# def load_motion_info(folder):
from spikeinterface.preprocessing.motion import load_motion_info
from PIL import Image
import json
from pathlib import Path
try:
    from spikeinterface.core.motion import Motion
except:
    from spikeinterface.sortingcomponents.motion.motion_utils import Motion
from joblib import Parallel, delayed
import shutil
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


def load_legacy_motion_info(folder):
    folder = Path(folder)
    spatial_bins_um = np.load(folder / "spatial_bins.npy")
    displacement = [np.load(folder / "motion.npy")]
    temporal_bins_s = [np.load(folder / "temporal_bins.npy")]

    motion_info = {}
    with open(folder / "parameters.json") as f:
        motion_info["parameters"] = json.load(f)

    with open(folder / "run_times.json") as f:
        motion_info["run_times"] = json.load(f)

    motion = Motion(
        displacement,
        temporal_bins_s,
        spatial_bins_um,
        direction="y",
    )
    motion_info["motion"] = motion
    motion_info["peaks"] = np.load(folder / "peaks.npy")
    motion_info["peak_locations"] = np.load(folder / "peaks.npy")

    return motion_info

# %%
def plot_session_opto_drift(session, data_type, plot=True, update_csv = False, update_cut = True):
    session_dir = session_dirs(session)

    # %%
    # ephys_opto_preprocessing(session, 'curated', 'soma')

    # %%
    motion_root = f'/root/capsule/data/{session}_sorted/preprocessed/motion/'
    if os.path.exists(motion_root):
        # all_files = [file for file in os.listdir(motion_root) if 'recording' in file]
        motion_path = os.path.join(motion_root, session_dir['stream_name'])
        all_files = os.listdir(motion_path)
        if 'motion.npy' in all_files:
            motion_info = load_legacy_motion_info(motion_path)
            print(f'Legacy motion info for {session}')
        else:
            motion_info = load_motion_info(motion_path)
            print(f'New motion info for {session}')
    else:
        print(f'No motion info found for {session}')
        return

    # %%
    # sampling 
    bin_sampling = 10
    bin_short = 100
    # get start and end of recording
    qm_dir = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_dir) as f:
        qm_dict = json.load(f)
    start = qm_dict['ephys_cut'][0]
    stop = qm_dict['ephys_cut'][1]
    # start = Session(session_dir['session_dir']).recordnodes[0].recordings[0].continuous[0].timestamps[0]
    # stop = Session(session_dir['session_dir']).recordnodes[0].recordings[0].continuous[0].timestamps[-1]
    probe_location = np.linspace(2500, 0,  96)
    map_time = motion_info['motion'].temporal_bins_s
    
    # correction to take care of cases:
    # 1. Ephys time was re-aligned to harp time:  the mean of the map_time is not close to the middle of the session
    # 2. Legacy motion info starts at 1.0 seconds, which is not the start of the session
    # 3. NPopto: motion info starts at 0.5 seconds, which is not the start of the session
    if (np.abs(np.mean(map_time[0])-0.5*(start+stop)) > 10*60) or (map_time[0][0] == 0.5) or (map_time[0][0] == 1.0):
        if (map_time[0][0] == 0.5) or (map_time[0][0] == 1.0):
            mis_align = -start
        else: 
            mis_align = map_time[0][0] - start
        print(f'Correction. time_diff {np.abs(np.mean(map_time)-0.5*(start+stop))}')
        print(f'Drift start: {map_time[0][0]}')
        print(f'Session start: {start}')
    else:
        mis_align = 0
        print(f'No correction. time_diff {np.abs(np.mean(map_time)-0.5*(start+stop))}')
        print(f'Drift start: {map_time[0][0]}')
        print(f'Session start: {start}')
    
    temp_bins_sampling = np.arange(start+mis_align, stop+mis_align, bin_sampling)
    temp_bins = np.arange(start, stop, bin_short)

    drift_sampling = np.zeros(shape=(len(probe_location), len(temp_bins_sampling)))
    for i, t in enumerate(temp_bins_sampling): 
        for j, p in enumerate(probe_location):
            drift_sampling[j, i] = motion_info['motion'].get_displacement_at_time_and_depth([t], [p])
    
    # resest drift sampling time
    temp_bins_sampling = temp_bins_sampling - mis_align

    # %%
    # fast dynamics
    drift = np.zeros(shape=(len(probe_location), len(temp_bins)))

    drift = np.full((len(probe_location), len(temp_bins)), np.nan)
    for i, t in enumerate(temp_bins): 
        for j, p in enumerate(probe_location): 
            curr_ind = np.where((temp_bins_sampling >= t-bin_short/2) & (temp_bins_sampling < t+bin_short/2))[0]
            if curr_ind.size > 0:
                drift[j, i] = np.mean(drift_sampling[j, curr_ind])



    # %%
    # slow dynamics
    bin_long = 300 # seconds
    temp_bins_slow = temp_bins

    drift_slow_pre = np.full((len(probe_location), len(temp_bins_slow)), np.nan)
    drift_slow_post = np.full((len(probe_location), len(temp_bins_slow)), np.nan)
    drift_slow = np.full((len(probe_location), len(temp_bins_slow)), np.nan)
    for i, t in enumerate(temp_bins_slow): 
        for j, p in enumerate(probe_location):
            curr_ind = np.where((temp_bins_sampling >= t-bin_long) & (temp_bins_sampling < t))[0]
            if curr_ind.size > 0:
                drift_slow_pre[j, i] = np.mean(drift_sampling[j, curr_ind])
            curr_ind = np.where((temp_bins_sampling > t) & (temp_bins_sampling <= t+bin_long))[0]
            if curr_ind.size > 0:
                drift_slow_post[j, i] = np.mean(drift_sampling[j, curr_ind])
            curr_ind = np.where((temp_bins_sampling >= t-bin_long/2) & (temp_bins_sampling <= t+bin_long/2))[0]
            if curr_ind.size > 0:
                drift_slow[j, i] = np.mean(drift_sampling[j, curr_ind])
    drift_slow_pre[:,0] = drift_slow_pre[:,1]
    drift_slow_post[:,-1] = drift_slow_post[:,-2]

    # %% realign time if needed
    # load qm
    qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_file) as f:
        qm_dict = json.load(f)
    
    # if not qm_dict['ephys_sync'] and '717121' not in session:
    #     temp_bins = align_timestamps_to_anchor_points(temp_bins, np.load(os.path.join(session_dir['alignment_dir'], 'local_times.npy')), np.load(os.path.join(session_dir['alignment_dir'], 'harp_times.npy')))
    #     temp_bins_slow = align_timestamps_to_anchor_points(temp_bins_slow, np.load(os.path.join(session_dir['alignment_dir'], 'local_times.npy')), np.load(os.path.join(session_dir['alignment_dir'], 'harp_times.npy')))

    # %% plot
    range_max = np.max(np.abs(drift))
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 5, height_ratios=[0.25, 3, 1])
    ax1 = plt.subplot(gs[1, 0]) 
    cbar_ax = plt.subplot(gs[0, 0])
    sns.heatmap(drift, cmap='seismic', center=0, vmin=-range_max, vmax=range_max, ax=ax1, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
    ax1.axvline(x=3, color='black', linestyle='--', linewidth=2) 
    ax1.set_xticks(np.linspace(0, len(temp_bins), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins[0], temp_bins[-1], 10))], rotation=90);
    ax1.set_yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Depth (um)')
    cbar_ax.set_title('Fast drift')


    # ax2 = plt.subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1) 
    nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '.nwb')
    if os.path.exists(nwb_file):
        nwb = load_nwb_from_filename(nwb_file)
        choice_history, reward_history, p_reward, autowater_offered, trial_time = get_history_from_nwb(nwb)
        _, Axes = plot_foraging_session(  # noqa: C901
                        choice_history,
                        reward_history,
                        p_reward = p_reward,
                        autowater_offered = autowater_offered,
                        trial_time = trial_time,
                        ax = ax2,
                        # legend=False,
                        ) 
        for ax in Axes:
            ax.set_xlim([temp_bins[0], temp_bins[-1]])

    ax3 = plt.subplot(gs[1, 1])
    cbar_ax = plt.subplot(gs[0, 1])
    sns.heatmap(drift_slow, cmap='seismic', center=0, vmin=-range_max, vmax=range_max, ax=ax3, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
    ax3.set_xticks(np.linspace(0, len(temp_bins_slow), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins_slow[0], temp_bins_slow[-1], 10))], rotation=90);
    ax3.set_yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Depth (um)')
    cbar_ax.set_title('Slow drift')

    # if opto files exist:
    if os.path.exists(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_session.csv')):
        ax4 = fig.add_subplot(gs[2, 1]) 
        opto_df = pd.read_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_session.csv'))
        opto_times_pre = opto_df.query('pre_post == "pre"')['time'].values
        if len(opto_times_pre)>0:
            pre_end = np.max(opto_times_pre)
            y_rand = np.random.uniform(0, 1, len(opto_times_pre))
            ax4.scatter(opto_times_pre, y_rand, c='r', label='pre', s=10, edgecolors=None)

        opto_times_post = opto_df.query('pre_post == "post"')['time'].values
        if len(opto_times_post)>0:
            post_start = np.min(opto_times_post)
            y_rand = np.random.uniform(0, 1, len(opto_times_post))
            ax4.scatter(opto_times_post, y_rand, c='b', label='post', s=10, edgecolors=None)    
        sites = opto_df['site'].values
        # if len(opto_df[~opto_df['site'].str.contains('LC', na=False)]['time'].values)>0:
        #     opto_surf = opto_df[~opto_df['site'].str.contains('LC', na=False)]['time'].values
        #     y_rand = np.random.uniform(0, 1, len(opto_surf)) 
        #     ax4.scatter(opto_surf, y_rand, label='surf', s=10, edgecolors='k', facecolors='none', linewidth=2)
        # elif len(opto_df[~opto_df['site'].str.contains('surface', na=False)]['time'].values)>0:
        #     opto_surf = opto_df[~opto_df['site'].str.contains('surface', na=False)]['time'].values
        #     y_rand = np.random.uniform(0, 1, len(opto_surf)) 
        #     ax4.scatter(opto_surf, y_rand, label='surf', s=10, edgecolors='k', facecolors='none', linewidth=2)
            
        ax4.set_xlabel('Time (s)')
        ax4.set_xlim([temp_bins_slow[0], temp_bins_slow[-1]])



    ax = plt.subplot(gs[1, 2])
    cbar_ax = plt.subplot(gs[0, 2])
    range_max = np.max(np.abs(np.diff(drift)))
    sns.heatmap(np.diff(drift), cmap='seismic', center=0, vmin=-range_max, vmax=range_max, ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
    ax.set_xticks(np.linspace(0, len(temp_bins), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins[0], temp_bins[-1], 10))], rotation=90);
    ax.set_yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (um)')
    cbar_ax.set_title('Fast drift derivative')

    # ax = plt.subplot(gs[1, 2])
    # nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '.nwb')
    # ax = plot_session_in_time_all(nwb, ax=ax)


    range_max = np.max(np.abs(drift_slow_post - drift_slow_pre))
    ax = plt.subplot(gs[1, 3])
    cbar_ax = plt.subplot(gs[0, 3])
    sns.heatmap(drift_slow_post - drift_slow_pre, cmap='seismic', center=0, vmin=-range_max, vmax=range_max, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, ax=ax)
    ax.set_xticks(np.linspace(0, len(temp_bins_slow), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins_slow[0], temp_bins_slow[-1], 10))], rotation=90);
    ax.set_yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (um)')
    cbar_ax.set_title('Slow drift derivative')

    # ax = plt.subplot(gs[1, 3])
    # nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '.nwb')
    # ax = plot_session_in_time_all(nwb, ax=ax)


    drift_diff = drift - drift_slow_pre
    range_max = np.max(np.abs(drift_diff))
    ax = plt.subplot(gs[1, 4])
    cbar_ax = plt.subplot(gs[0, 4])
    sns.heatmap(drift_diff, cmap='seismic', center=0, vmin=-range_max, vmax=range_max, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, ax=ax)
    ax.set_xticks(np.linspace(0, len(temp_bins_slow), 10), [f"{x:.1f}" for x in list(np.linspace(temp_bins_slow[0], temp_bins_slow[-1], 10))], rotation=90);
    ax.set_yticks(np.linspace(0, len(probe_location), 5), [f"{x:.1f}" for x in list(np.linspace(probe_location[0], probe_location[-1], 5))]);
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (um)')
    cbar_ax.set_title('Fast drift - slow drift')

    # ax = plt.subplot(gs[1, 4])
    # nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '.nwb')
    # ax = plot_session_in_time_all(nwb, ax=ax)


    # plt.tight_layout()
    plt.savefig(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_motion_drift.pdf'))

    # %%
    unit_tbl = get_unit_tbl(session, data_type)


    # %%
    sorting_analyzer = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False) 
    sorting = si.load(session_dir[f'curated_dir_{data_type}'])

    # %%
    unit_locations = sorting_analyzer.get_extension('unit_locations').get_data(outputs="by_unit")
    spike_amplitudes = sorting_analyzer.get_extension('spike_amplitudes').get_data(outputs="by_unit")[0]

    del sorting_analyzer

    if update_cut:
        opto_drift_tbl_exist = pd.read_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_drift_tbl.csv'))


    # %%
    def plot_drift(unit_id, plot=plot):
        spike_times = unit_tbl[unit_tbl['unit_id'] == unit_id]['spike_times'].values[0]
        spike_amplitude = spike_amplitudes[unit_id]
        # firing rate
        temp_bins_spike = np.zeros(len(temp_bins)+1)
        temp_bins_spike[1:] = temp_bins+bin_short/2
        temp_bins_spike[0] = temp_bins[0] - bin_short/2
        temp_bins_slow_spike = temp_bins_slow
        spike_counts_fast, _ = np.histogram(spike_times, bins=temp_bins_spike)
        spike_counts_fast = spike_counts_fast/bin_short
        spike_counts_slow_pre = np.full(len(temp_bins_slow_spike), np.nan)
        spike_counts_slow_post = np.full(len(temp_bins_slow_spike), np.nan)
        spike_counts_slow = np.full(len(temp_bins_slow_spike), np.nan)
        for i, t in enumerate(temp_bins_slow_spike):
            if t - bin_long> temp_bins[0]:
                spike_counts_slow_pre[i] = np.sum((spike_times >= t - bin_long) & (spike_times < t))
                spike_counts_slow_pre[i] = spike_counts_slow_pre[i] / bin_long
            if t + bin_long < temp_bins[-1]:
                spike_counts_slow_post[i]= np.sum((spike_times >= t) & (spike_times < t + bin_long))
                spike_counts_slow_post[i] = spike_counts_slow_post[i] / bin_long
            if t - bin_long/2 > temp_bins[0] and t + bin_long/2 < temp_bins[-1]:
                spike_counts_slow[i] = np.sum((spike_times >= t - bin_long/2) & (spike_times < t + bin_long/2))
                spike_counts_slow[i] = spike_counts_slow[i] / bin_long
            
        # drift no need to calculate again
        closest_ybin = np.argmin(np.abs(probe_location - unit_locations[unit_id][1]))

        # amplitude
        amplitude_fast = np.full((len(temp_bins)), np.nan)
        amplitude_slow_pre = np.full((len(temp_bins_slow)), np.nan)
        amplitude_slow_post = np.full((len(temp_bins_slow)), np.nan)
        amplitude_slow = np.full((len(temp_bins)), np.nan)
        
        for i, t in enumerate(temp_bins):
            inds = np.where((spike_times >= t-bin_short) & (spike_times < t+bin_short))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    amplitude_fast[i] = np.mean(temp)

        for i, t in enumerate(temp_bins_slow):
            inds = np.where((spike_times >= t-bin_long - bin_short) & (spike_times < t - bin_short))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    amplitude_slow_pre[i] = np.mean(temp)
            inds = np.where((spike_times >= t + bin_short) & (spike_times < t + bin_short + bin_long))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    amplitude_slow_post[i] = np.mean(temp)
            
            inds = np.where((spike_times >= t-bin_long/2) & (spike_times < t+bin_long/2))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    amplitude_slow[i] = np.mean(temp)
        # sd.mean
        sd = np.std(spike_counts_slow[np.where(~np.isnan(spike_counts_slow))[0]])/np.nanmean(spike_counts_slow)
        if update_cut:
            nan = None
            ephys_cut = opto_drift_tbl_exist.query('unit_id == @unit_id')['ephys_cut'].values[0]
            ephys_cut = eval(ephys_cut)
            spike_counts_slow_cut = spike_counts_slow.copy()
            temp_bins_slow_spike_cut = temp_bins_slow_spike.copy()
            if ephys_cut[0] is not None:
                spike_counts_slow_cut = spike_counts_slow_cut[temp_bins_slow_spike_cut >= ephys_cut[0]]
                temp_bins_slow_spike_cut = temp_bins_slow_spike_cut[temp_bins_slow_spike_cut >= ephys_cut[0]]
            
            if ephys_cut[1] is not None:
                spike_counts_slow_cut = spike_counts_slow_cut[temp_bins_slow_spike_cut <= ephys_cut[1]]
                temp_bins_slow_spike_cut = temp_bins_slow_spike_cut[temp_bins_slow_spike_cut <= ephys_cut[1]]
            if len(spike_counts_slow_cut) >=2:
                sd_updated_cut = np.std(spike_counts_slow_cut[np.where(~np.isnan(spike_counts_slow_cut))[0]])/np.nanmean(spike_counts_slow_cut)
            else:
                sd_updated_cut = np.nan
                print(f'{session} {unit_id} too short after cut.')
        else:
            sd_updated_cut = sd

        ## prepare all matrices for regression
        # slow
        X_slow = np.column_stack((drift_slow[closest_ybin], amplitude_slow))
        z_slow = spike_counts_slow
        nan_inds_slow = np.where(~np.isnan(z_slow) & np.all(~np.isnan(X_slow), axis=1))[0]
        # diff_abs_fast
        z_diff_abs_fast = np.abs((spike_counts_fast - spike_counts_slow_pre)/(0.5*(spike_counts_fast + spike_counts_slow_pre))) # firing rate change
        X_diff_abs_fast = np.abs(np.column_stack((drift[closest_ybin]-drift_slow_pre[closest_ybin], # drift
                            amplitude_fast - amplitude_slow_pre,  # amplitude change
                            (drift[closest_ybin]-drift_slow_pre[closest_ybin])*spike_counts_slow))) # interaction between drift and firing rate
        nan_inds_abs_fast = np.where(~np.isnan(z_diff_abs_fast) & np.all(~np.isnan(X_diff_abs_fast), axis=1))[0]
        # diff_abs_slow
        z_diff_abs_slow = np.abs(spike_counts_slow_post - spike_counts_slow_pre)/(0.5*(spike_counts_slow_post+spike_counts_slow_pre)) # firing rate change
        X_diff_abs_slow = np.column_stack((np.abs(drift_slow_post[closest_ybin]-drift_slow_pre[closest_ybin]), # drift
                            np.abs(amplitude_slow_post - amplitude_slow_pre),  # amplitude change
                            np.abs((drift_slow_post[closest_ybin]-drift_slow_pre[closest_ybin]))*(spike_counts_slow))) # interaction between drift and firing rate
        nan_inds_diff_abs_slow = np.where(~np.isnan(z_diff_abs_slow) & np.all(~np.isnan(X_diff_abs_slow), axis=1))[0]
        
        ## linear regression for slow dynamics

        if len(nan_inds_slow) >= 2:
            model = LinearRegression()
            model.fit(X_slow[nan_inds_slow], z_slow[nan_inds_slow])
            r_squared = model.score(X_slow[nan_inds_slow], z_slow[nan_inds_slow])
        else:
            r_squared = np.nan
            print(f'{session} {unit_id} too short.')

        ## linear regression for slow dynamics after cut if update
        if update_cut:
            nan = None
            ephys_cut = opto_drift_tbl_exist.query('unit_id == @unit_id')['ephys_cut'].values[0]
            ephys_cut = eval(ephys_cut)
            X_slow_cut = X_slow.copy()
            z_slow_cut = z_slow.copy()
            temp_bins_slow_spike_cut = temp_bins_slow_spike.copy()
            if ephys_cut[0] is not None:
                X_slow_cut = X_slow[temp_bins_slow_spike_cut >= ephys_cut[0]]
                z_slow_cut = z_slow[temp_bins_slow_spike_cut >= ephys_cut[0]]
                temp_bins_slow_spike_cut = temp_bins_slow_spike_cut[temp_bins_slow_spike_cut >= ephys_cut[0]]
            
            if ephys_cut[1] is not None:
                X_slow_cut = X_slow_cut[temp_bins_slow_spike_cut <= ephys_cut[1]]
                z_slow_cut = z_slow_cut[temp_bins_slow_spike_cut <= ephys_cut[1]]
                temp_bins_slow_spike_cut = temp_bins_slow_spike_cut[temp_bins_slow_spike_cut <= ephys_cut[1]]
            
            nan_inds_slow_cut = np.where(~np.isnan(z_slow_cut) & np.all(~np.isnan(X_slow_cut), axis=1))[0]
            if len(nan_inds_slow_cut) >= 2:
                model_cut = LinearRegression()
                model_cut.fit(X_slow_cut[nan_inds_slow_cut], z_slow_cut[nan_inds_slow_cut])
                r_squared_updated = model_cut.score(X_slow_cut[nan_inds_slow_cut], z_slow_cut[nan_inds_slow_cut])
            else:
                r_squared_updated = np.nan
        else:
            r_squared_updated = r_squared
        
        ## random forest model for fast dynamics

        model_rf_fast = RandomForestRegressor(n_estimators=10, random_state=42)
        model_rf_fast.fit(X_diff_abs_fast[nan_inds_abs_fast], z_diff_abs_fast[nan_inds_abs_fast])
        z_pred_rf_diff_abs_fast = model_rf_fast.predict(X_diff_abs_fast[nan_inds_abs_fast])
        r2_rf_diff_abs_fast = r2_score(z_diff_abs_fast[nan_inds_abs_fast], z_pred_rf_diff_abs_fast)
        importances_diff_abs_fast = model_rf_fast.feature_importances_
           
        ## random forest mode for slow dynamics
        model_rf_slow = RandomForestRegressor(n_estimators=10, random_state=42)
        model_rf_slow.fit(X_diff_abs_slow[nan_inds_diff_abs_slow], z_diff_abs_slow[nan_inds_diff_abs_slow])
        z_pred_rf_diff_abs_slow = model_rf_slow.predict(X_diff_abs_slow[nan_inds_diff_abs_slow])
        r2_rf_diff_abs_slow = r2_score(z_diff_abs_slow[nan_inds_diff_abs_slow], z_pred_rf_diff_abs_slow)
        importances_diff_abs_slow = model_rf_slow.feature_importances_

        ## linear regression for slow dynamics
        model_slow_diff = LinearRegression()
        model_slow_diff.fit(X_diff_abs_slow[nan_inds_diff_abs_slow], z_diff_abs_slow[nan_inds_diff_abs_slow])
        z_pred_diff_abs_slow = model_slow_diff.predict(X_diff_abs_slow[nan_inds_diff_abs_slow])
        r2_diff_abs_slow = model_slow_diff.score(X_diff_abs_slow[nan_inds_diff_abs_slow], z_diff_abs_slow[nan_inds_diff_abs_slow])
        coeffs_diff_abs_slow = model_slow_diff.coef_

        ## linear regression with for fast dynamics
        model_fast = LinearRegression()
        model_fast.fit(X_diff_abs_fast[nan_inds_abs_fast], z_diff_abs_fast[nan_inds_abs_fast])
        z_pred_fast = model_fast.predict(X_diff_abs_fast[nan_inds_abs_fast])
        r2_diff_abs_fast = model_fast.score(X_diff_abs_fast[nan_inds_abs_fast], z_diff_abs_fast[nan_inds_abs_fast])
        coeffs_diff_abs_fast = model_fast.coef_

        
        if plot: 
            fig = plt.figure(figsize=(20, 20))
            plt.rcParams.update({'font.size': 8})
            gs = gridspec.GridSpec(5, 3)
            ax = plt.subplot(gs[0, 0])
            plt.hist(spike_times, bins=temp_bins);
            if 'pre_end' in locals():
                plt.axvline(x=pre_end, color='r', linestyle='--', linewidth=2)
            if 'post_start' in locals():
                plt.axvline(x=post_start, color='b', linestyle='--', linewidth=2)
            if 'opto_surf' in locals():
                plt.axvline(x=np.min(opto_surf), color='k', linestyle='--', linewidth=2)
            plt.title('Firing rate')
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            ax = plt.subplot(gs[1, 0])
            plt.plot(temp_bins, spike_counts_fast, label='fast')
            plt.plot(temp_bins_slow_spike, spike_counts_slow, label='slow', c = 'r')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)       

            ax = plt.subplot(gs[2, 0])
            plt.plot(temp_bins, (spike_counts_fast-spike_counts_slow_pre)/(0.5*(spike_counts_fast+spike_counts_slow_pre)), label='fast-slow')
            plt.plot(temp_bins_slow_spike, (spike_counts_slow_post - spike_counts_slow_pre)/(0.5*(spike_counts_slow_pre + spike_counts_slow_post)), label = 'diff(slow)', c='r')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            ax = plt.subplot(gs[3, 0])
            plt.plot(temp_bins, np.abs(spike_counts_fast - spike_counts_slow_pre)/(0.5*(spike_counts_fast+spike_counts_slow_pre)), label='abs(fast-slow)')
            plt.plot(temp_bins_slow_spike, np.abs(spike_counts_slow_post - spike_counts_slow_pre)/(0.5*(spike_counts_slow_pre + spike_counts_slow_post)), label = 'abs(diff(slow))', c='r')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            # drift
            ax = plt.subplot(gs[0, 1])
            plt.title('Esitimated motion')
            ax = plt.subplot(gs[1, 1])
            plt.plot(temp_bins, drift[closest_ybin, :], label='fast')
            plt.plot(temp_bins_slow, drift_slow[closest_ybin, :], label='slow', c = 'r')
            plt.plot(temp_bins, np.zeros(len(temp_bins)), 'k--')
            if 'pre_end' in locals():
                plt.axvline(x=pre_end, color='r', linestyle='--', linewidth=2)
            if 'post_start' in locals():
                plt.axvline(x=post_start, color='b', linestyle='--', linewidth=2)
            if 'opto_surf' in locals():
                plt.axvline(x=np.min(opto_surf), color='k', linestyle='--', linewidth=2)
            # plt.plot(motion_info['motion'].get_displacement_at_time_and_depth(times_s, locations_um))
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            ax = plt.subplot(gs[2, 1])
            plt.plot(temp_bins, (drift[closest_ybin, :] - drift_slow[closest_ybin,:]), label='fast-slow')
            plt.plot(temp_bins, drift_slow_post[closest_ybin, :] - drift_slow[closest_ybin,:], label='diff(slow)', c = 'r')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            ax = plt.subplot(gs[3, 1])
            plt.plot(temp_bins, np.abs(drift[closest_ybin, :] - drift_slow[closest_ybin,:]), label='abs(fast-slow)')
            plt.plot(temp_bins, np.abs(drift_slow_post[closest_ybin, :] - drift_slow[closest_ybin,:]), label='abs(diff(slow))', c = 'r')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            ax = plt.subplot(gs[0, 2])
            plt.scatter(spike_times, spike_amplitude, c='k', s=0.5, alpha=0.25)
            if 'pre_end' in locals():
                plt.axvline(x=pre_end, color='r', linestyle='--', linewidth=2)
            if 'post_start' in locals():
                plt.axvline(x=post_start, color='b', linestyle='--', linewidth=2)
            if 'opto_surf' in locals():
                plt.axvline(x=np.min(opto_surf), color='k', linestyle='--', linewidth=2)
            plt.title('Spike amplitude')
            # plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            ax = plt.subplot(gs[1, 2])
            plt.plot(temp_bins, amplitude_fast, label='fast')
            plt.plot(temp_bins, amplitude_slow, label='slow', c = 'r')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            ax = plt.subplot(gs[2, 2])
            plt.plot(temp_bins, amplitude_fast - amplitude_slow_pre, label='fast-slow')
            # ax = ax.twinx()
            plt.plot(temp_bins_slow, amplitude_slow_post - amplitude_slow_pre, label='diff(slow)', c='r')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            ax = plt.subplot(gs[3, 2])
            plt.plot(temp_bins, np.abs(amplitude_fast - amplitude_slow_pre), label='abs(fast-slow)')
            plt.plot(temp_bins_slow, np.abs(amplitude_slow_post - amplitude_slow_pre), label = 'abs(diff(slow))', c='r')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            gs_model = gridspec.GridSpecFromSubplotSpec(1, 5, gs[4, :])
            ax = plt.subplot(gs_model[0])
            plt.plot(temp_bins, spike_counts_slow, label='data', c='r')
            if len(nan_inds_slow) >= 2:
                plt.plot(temp_bins[nan_inds_slow], model.predict(X_slow[nan_inds_slow]), label='prediction', c='b')
            plt.plot(temp_bins[nan_inds_slow], model.predict(X_slow[nan_inds_slow]), label='prediction', c='b')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.title(f"Slow FR: LR R²: {r_squared:.2f}")
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)
                plt.title(f"R²: {r_squared:.2f} Updated: {r_squared_updated:.2f}")

            ax = plt.subplot(gs_model[1])
            plt.plot(temp_bins, z_diff_abs_slow, label='data', c='r')
            plt.plot(temp_bins[nan_inds_diff_abs_slow], z_pred_rf_diff_abs_slow, label='RF', c='g')
            plt.plot(temp_bins[nan_inds_diff_abs_slow], z_pred_diff_abs_slow, label='LR', c='b')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.title(f"Slow abs(diff(FR)): LR: {r2_diff_abs_slow:.1f} RF R²: {r2_rf_diff_abs_slow:.1f}")
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)


            ax = plt.subplot(gs_model[2])
            plt.plot(temp_bins, z_diff_abs_fast, label='data', c='r')
            plt.plot(temp_bins[nan_inds_abs_fast], z_pred_fast, label='LR', c='b')
            plt.plot(temp_bins[nan_inds_abs_fast], z_pred_rf_diff_abs_fast, label='RF', c='g')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.title(f"Fast abs(diff(FR)): LR: {r2_diff_abs_fast:.1f} RF: {r2_rf_diff_abs_fast:.1f}")
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)

            # # Print coefficients and intercept
            # print(f"Coefficients: {model.coef_}")
            # print(f"Intercept: {model.intercept_}")

            # # Predict new values
            # X_new = np.array([[6, 7], [7, 8]])
            # z_pred = model.predict(X_new)
            # print(f"Predicted values: {z_pred}")

            ax = plt.subplot(gs_model[4])
            plt.hist(spike_counts_slow, bins=20, alpha=0.5, label='data', color='r')
            plt.title(f"SD/mean: {sd:.2f}")
            plt.xlabel('Time (s)')
            plt.legend()
            if update_cut:
                if ephys_cut[0] is not None:
                    plt.axvline(x=ephys_cut[0], color='g', linestyle='--', linewidth=2)
                if ephys_cut[1] is not None:
                    plt.axvline(x=ephys_cut[1], color='g', linestyle='--', linewidth=2)
    
            plt.suptitle(f'{unit_id} y loc {unit_locations[unit_id][1] :.2f} um')
            plt.rcParams.update({'font.size': 8})
            # plt.tight_layout()

        return {'unit_id': unit_id,
                'r_squared_slow': r_squared,
                'r_squared_diff_abs_slow_lr': r2_diff_abs_slow,
                'r_squared_diff_abs_fast_lr': r2_diff_abs_fast,
                'r_squared_diff_abs_slow_rf': r2_rf_diff_abs_slow,
                'r_squared_diff_abs_fast_rf': r2_rf_diff_abs_fast,
                'sd/mean': sd, 
                'sd/mean_updated': sd_updated_cut,
                'importances_diff_abs_slow': importances_diff_abs_slow,
                'importances_diff_abs_fast': importances_diff_abs_fast,
                'coeffs_diff_abs_slow': coeffs_diff_abs_slow,
                'coeffs_diff_abs_fast': coeffs_diff_abs_fast, 
                'regressors': ['drift', 'amplitude', 'drift*fr'],
                'ephys_cut': [np.nan, np.nan],
                'drift_unit': False,
                'r_squared_slow_corrected': r_squared_updated}

    # %%

    units_to_plot = []
    p_thresh = 0.3
    lat_thresh = 0.02
    for i, row in unit_tbl.iterrows():
        # opto_pass = row['opto_pass']> 0
        # quality_pass = (row['isi_violations_ratio'] < 0.5) & (row['decoder_label'] != 'noise') & (row['decoder_label'] != 'artifact')
        if row['opto_pass'] and row['default_qc']:
            units_to_plot.append(row['unit_id'])

    # %%
    drift_dir = os.path.join(session_dir[f'opto_dir_{data_type}'], 'drift')
    if os.path.exists(drift_dir):
        shutil.rmtree(drift_dir)

    os.makedirs(name=drift_dir, exist_ok=True)


    # os.mkdir(drift_dir, exist_ok=True)?
    opto_drift_tbl = pd.DataFrame()
    for unit in units_to_plot:                             
        unit_drift_dict = plot_drift(unit, plot=True)
        opto_drift_tbl = pd.concat([opto_drift_tbl, pd.DataFrame([unit_drift_dict])], ignore_index=True)
        if plot:
            plt.savefig(os.path.join(drift_dir, f'{unit}_drift.pdf'))
            plt.show()
    if update_cut:
        nan = np.nan
        for ind, row in opto_drift_tbl.iterrows():
            unit_id = row['unit_id']
            if unit_id not in opto_drift_tbl_exist['unit_id'].values:
                continue
            ephys_cut = opto_drift_tbl_exist.query('unit_id == @unit_id')['ephys_cut'].values[0]
            ephys_cut = eval(ephys_cut)
            opto_drift_tbl.at[ind, 'ephys_cut'] = ephys_cut
            opto_drift_tbl.at[ind, 'drift_unit'] = opto_drift_tbl_exist.query('unit_id == @unit_id')['drift_unit'].values[0]
    if update_csv:
        opto_drift_tbl.to_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_drift_tbl.csv'))
    if plot:
        # merge_pdfs(input_dir=drift_dir, output_filename=os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_drift.pdf')) 
        combine_pdf_big(drift_dir, os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_drift.pdf'))
    return opto_drift_tbl

import numpy as np

def mode_by_bins(data, bins):
    """
    Calculate the mode of data within specified bins.
    
    Parameters
    ----------
    data : array-like
        Input data to calculate the mode from.
    bins : int, float, or list
        - int: number of bins
        - float: bin width
        - list or array: explicit bin edges
    
    Returns
    -------
    mode : float
        Mode (approximate) of the data.
    bin_range : list
        [lower_edge, upper_edge] of the mode bin.
    """
    data = np.asarray(data)
    
    # Handle case where all values are identical
    if np.all(data == data[0]):
        return float(data[0]), [float(data[0]), float(data[0])]

    # Determine bin edges
    if isinstance(bins, list) or isinstance(bins, np.ndarray):
        bin_edges = np.array(bins)
    elif isinstance(bins, int):
        bin_edges = np.linspace(np.min(data), np.max(data), bins + 1)
    elif isinstance(bins, float):
        bin_edges = np.arange(np.min(data), np.max(data) + bins, bins)
    else:
        raise TypeError("bins must be int, float, or list/array of edges")

    counts, _ = np.histogram(data, bins=bin_edges)
    max_bin = np.argmax(counts)
    mode_bin = (data >= bin_edges[max_bin]) & (data < bin_edges[max_bin + 1])

    # In case all points fall in one bin, np.mean() still works
    mode = np.mean(data[mode_bin]) if np.any(mode_bin) else np.mean(data)

    return mode, [bin_edges[max_bin], bin_edges[max_bin + 1]]


def generate_session_opto_drift_trial_table(session, data_type, opto_only = True, save = True):
    session_dir = session_dirs(session)
    # load qm
    qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_file) as f:
        qm_dict = json.load(f)
    # load motion  
    motion_root = f'/root/capsule/data/{session}_sorted/preprocessed/motion/'
    if os.path.exists(motion_root):
        # all_files = [file for file in os.listdir(motion_root) if 'recording' in file]
        motion_path = os.path.join(motion_root, session_dir['stream_name'])

        all_files = os.listdir(motion_path)
        if 'motion.npy' in all_files:
            motion_info = load_legacy_motion_info(motion_path)
        else:
            motion_info = load_motion_info(motion_path)
    else:
        motion_info = None
    # load amp
    # %%
    sorting_analyzer = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False) 
    # sorting = si.load(session_dir[f'curated_dir_{data_type}'])

    # %%
    unit_locations = sorting_analyzer.get_extension('unit_locations').get_data(outputs="by_unit")
    spike_amplitudes = sorting_analyzer.get_extension('spike_amplitudes').get_data(outputs="by_unit")[0]

    # del sorting_analyzer
    
    session_tbl = get_session_tbl(session)
    unit_tbl = get_unit_tbl(session, data_type)
    if opto_only:
        unit_tbl = unit_tbl[(unit_tbl['opto_pass'] == True) & (unit_tbl['default_qc'] == True)]

    trials_starts = session_tbl['goCue_start_time'].values
    # convert back to raw time if ephys was aligned to harp time
    # if not qm_dict['ephys_sync']:
    #     trials_starts_raw = align_timestamps_to_anchor_points(trials_starts, np.load(os.path.join(session_dir['alignment_dir'], 'harp_times.npy')), np.load(os.path.join(session_dir['alignment_dir'], 'local_times.npy')))
    # else:
    #     trials_starts_raw = trials_starts
    if motion_info is not None:
        map_time = motion_info['motion'].temporal_bins_s
        start = qm_dict['ephys_cut'][0]
        stop = qm_dict['ephys_cut'][1]
        # correction to take care of cases:
        # 1. Ephys time was re-aligned to harp time:  the mean of the map_time is not close to the middle of the session
        # 2. Legacy motion info starts at 1.0 seconds, which is not the start of the session
        # 3. NPopto: motion info starts at 0.5 seconds, which is not the start of the session
        if (np.abs(np.mean(map_time[0])-0.5*(start+stop)) > 10*60) or (map_time[0][0] == 0.5) or (map_time[0][0] == 1.0):
            if (map_time[0][0] == 0.5) or (map_time[0][0] == 1.0):
                mis_align = -start
            else: 
                mis_align = map_time[0][0] - start
        else:
            mis_align = 0

        trials_starts_raw = trials_starts + mis_align
    else:
        trials_starts_raw = trials_starts
    
    drift_data = pd.DataFrame()
    bin_size = 60
    step_num = 5
    step = np.arange(-step_num, step_num+1, 1) * bin_size/step_num
    samp = [trials_starts_raw + step[i] for i in range(len(step))]
    samp = np.concatenate(samp)
    for i, row in unit_tbl.iterrows():
        unit_id = row['unit_id']
        unit_location = unit_locations[unit_id][1]
        spike_amplitude = spike_amplitudes[unit_id]
        spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]

        # get drifts
        if motion_info is not None:
            unit_drift = motion_info['motion'].get_displacement_at_time_and_depth(samp, np.full(len(samp), unit_location))
            unit_drift = [np.mean(unit_drift[i::len(trials_starts_raw)]) for i in range(len(trials_starts_raw))]
        else:
            unit_drift = np.full(len(trials_starts_raw), 0.0)

        # get amp
        unit_amp = np.full(len(trials_starts_raw), np.mean(spike_amplitude))
        if len(spike_times) != len(spike_amplitude):
            print(f'{session} {unit_id} spike times and amplitude length mismatch, {len(spike_times)} vs {len(spike_amplitude)}')
            spike_times_new = sorting_analyzer.sorting.get_unit_spike_train_in_seconds(unit_id)
            spike_times = spike_times_new + np.mean(spike_times) - np.mean(spike_times_new)
        for i, t in enumerate(trials_starts):
            inds = np.where((spike_times >= t-bin_size) & (spike_times < t+bin_size))[0]
            if inds.size > 5:
                temp = spike_amplitude[inds]
                cut_off = np.percentile(temp, [5, 95])
                temp = temp[(temp >= cut_off[0]) & (temp <= cut_off[1])]
                if temp.size > 0:
                    unit_amp[i] = np.mean(temp)
        if unit_id == 43:
            print(unit_id)
        amp_mode, _ = mode_by_bins(unit_amp, 10.0)
        _, time_edges = mode_by_bins(spike_times, 60*10.0)
        if motion_info is not None:
            loc_mode = np.mean(motion_info['motion'].get_displacement_at_time_and_depth(np.linspace(time_edges[0], time_edges[1], 10), np.full(10, unit_location)))
        else:
            loc_mode = 0.0
        amp_abs = np.abs(unit_amp - amp_mode)
        unit_drift_abs = np.abs(unit_drift - loc_mode)

        unit_drift_data = pd.DataFrame({'motion' : unit_drift,
                                'motion_abs': unit_drift_abs,
                                'amp': unit_amp,
                                'amp_abs': amp_abs,
                                'unit_id': unit_id, 
                                'trial_ind': range(len(trials_starts)),
                                'trial_time': trials_starts})
        drift_data = pd.concat([drift_data, unit_drift_data], axis=0, ignore_index=True)

    
    if save:
        drift_data.to_csv(os.path.join(session_dir[f'ephys_dir_{data_type}'], f'{session}_drift_trial_table.csv'), index=False)
        

def update_unit_tbl_by_drift(session, data_type): 
    session_dir = session_dirs(session)
    opto_drift_tbl = pd.read_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_drift_tbl.csv'))
    unit_tbl = pd.read_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_metrics.pkl'))
    for i, row in opto_drift_tbl.iterrows():
        if row['r_squared_diff_abs_slow_rf'] > 0.1:
            unit_tbl.loc[unit_tbl['unit_id'] == row['unit_id'], 'ephys_cut'] = [row['r_squared_diff_abs_slow_rf'], row['r_squared_diff_abs_fast_rf']]
            unit_tbl.loc[unit_tbl['unit_id'] == row['unit_id'], 'drift_unit'] = True
    unit_tbl.to_pickle(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_metrics.pkl'))
    return unit_tbl

if __name__ == '__main__':
    session_assets = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
    session_list = session_assets['session_id'].values
    session_list = [session for session in session_list if isinstance(session, str)]

    def process(session):
        # session_dir = session_dirs(session)
        # if session_dir['curated_dir_curated'] is not None :
        #if get_unit_tbl(session, 'curated') is not None:
            try:
                print(session)
                plot_session_opto_drift(session, 'raw', update_csv=True, plot=True, update_cut=False)
                #generate_session_opto_drift_trial_table(session, 'curated', opto_only=False, save=True)
                print(f'{session} done')
            except:
                print(f'{session} error')
 
    # Parallel(n_jobs=5)(delayed(process)(session) for session in session_list)
    #process('behavior_838332_2026-03-10_13-23-52')
    # session_ind = session_list.index('behavior_ZS059_2021-04-12_14-57-43')
    for session in session_list:
        process(session)

