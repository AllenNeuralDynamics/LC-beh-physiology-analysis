# %%
import os
import sys
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
from utils.beh_functions import session_dirs
from utils.plot_utils import shiftedColorMap, template_reorder, plot_raster_bar,merge_pdfs, combine_pdf_big
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from open_ephys.analysis import Session
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
from aind_ephys_utils import align
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import pickle
import json

from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import load_nwb
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename

 
# Create a white-to-bright red colormap
colors = [(1, 1, 1), (1, 0, 0)]  # white to red
my_red = LinearSegmentedColormap.from_list("white_to_red", colors)

# %%
# session = 'behavior_758017_2025-02-04_11-57-38'
# session_dir = session_dirs(session)
# data_type = 'raw'
# target = 'soma'

# resp_thresh = 0.8
# lat_thresh = 0.015
# pulse_width = 4
# %%
def max_index_ndarray(arr):
    """
    Find the index of the maximum value in an N-dimensional array.

    Parameters:
        arr (numpy.ndarray): Input N-dimensional array.

    Returns:
        tuple: Indices of the maximum value in each dimension.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    # Find the flattened index of the max value
    max_flat_index = np.argmax(arr)

    # Convert flattened index back to multi-dimensional index
    max_nd_index = np.unravel_index(max_flat_index, arr.shape)

    return max_nd_index 
# %%
def opto_plotting_unit(unit_id, spike_times, spike_amplitude, waveform, opto_wf, qc_dict, crosscorr, resp_p, resp_lat, opto_df, opto_info, qm, waveform_params, dim_1 = 'powers', resp_thresh=0.8, lat_thresh=0.015, plot = False):
    # calculate baseline\
    # separate pre and post baseline
    session_start = qm['ephys_cut'][0]
    session_end = qm['ephys_cut'][1]
    pre_end = opto_df.query('pre_post == "pre"')['time'].max()
    post_start = opto_df.query('pre_post == "post"')['time'].min()
    baseline_pre = opto_info['resp_win']*np.sum(spike_times < (pre_end+300))/(pre_end + 300 - session_start)
    if baseline_pre > 1:
        baseline_pre = 1
    baseline_post = opto_info['resp_win']*np.sum(spike_times > (post_start-300))/(session_end - post_start + 300)
    baseline = qc_dict['firing_rate']*opto_info['resp_win']
    # conditions
    # opto tagging
    pass_opto = False
    fig = None
    # find resp_p > thresh and resp_lat < lat_thresh
    resp_p = np.array(resp_p.tolist())
    resp_lat = np.array(resp_lat.tolist())
    # populate baseline for pre and post
    baseline_mat = np.ones_like(resp_p)
    if np.shape(resp_p)[5] == 2:
        baseline_mat[:, :, :, :, :, 0, :] = baseline_pre
        baseline_mat[:, :, :, :, :, 1, :] = baseline_post
    elif len(opto_df.query('pre_post == "pre"')) > 0:
        baseline_mat[:, :, :, :, :] = baseline_pre
    else:
        baseline_mat[:, :, :, :, :] = baseline_post

    resp_p_bl = resp_p - baseline_mat

    resp_pass_ind = np.where((resp_p_bl > resp_thresh) & (resp_lat < lat_thresh) & (resp_lat > 3/1000))
    opto_tagging_dict = {'unit_id': unit_id, 
                 'resp_p': None, 
                 'resp_p_bl': None,
                 'resp_lat': None, 
                 'powers': None,
                 'sites': None, 
                 'num_pulses': None, 
                 'durations': None,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                 'freqs': None, 
                 'stim_times': None, 
                 'opto_pass': False}

    if len(resp_pass_ind[0]) > 0:
        opto_tagging_dict['opto_pass'] = True
        opto_pass = True
    else: # get max P(resp) if no pass
        opto_pass = False
        p_max = np.nanmax(resp_p_bl)
        resp_pass_ind = np.where((resp_p_bl >= p_max))
    # get the response p and latencies
    opto_tagging_dict['resp_p'] = resp_p[resp_pass_ind]
    opto_tagging_dict['resp_p_bl'] = resp_p_bl[resp_pass_ind]
    opto_tagging_dict['resp_lat'] = resp_lat[resp_pass_ind]
    power_ind, site_ind, num_pulse_ind, duration_ind, freq_ind, stim_time_ind = resp_pass_ind[:-1]
    opto_tagging_dict['powers'] = np.array(opto_info['powers'])[power_ind]
    opto_tagging_dict['sites'] = np.array(opto_info['sites'])[site_ind]
    opto_tagging_dict['num_pulses'] = np.array(opto_info['num_pulses'])[num_pulse_ind]
    opto_tagging_dict['durations'] = np.array(opto_info['durations'])[duration_ind]
    opto_tagging_dict['freqs'] = np.array(opto_info['freqs'])[freq_ind]
    opto_tagging_dict['stim_times'] = np.array(['pre', 'post'])[stim_time_ind]
    # get all similarities
    opto_tagging_df = pd.DataFrame(opto_tagging_dict)
    # group by site, power, duration, pre_post and take maximum of resp_p and resp_lat
    # opto_tagging_df = opto_tagging_df.groupby(['unit_id', 'sites', 'powers', 'durations', 'stim_times']).agg({'resp_p': list, 'resp_lat': list, 'resp_p_bl': list}).reset_index()
    
    euc_dist = []
    corr = []
    mean_p = []
    if opto_wf is not None:
        for _, row in opto_tagging_df.iterrows():
            wf_curr = opto_wf.query(
                'unit_id == @unit_id and site == @site and power == @power and pre_post == @pre_post',
                local_dict={
                    'site': row['sites'],
                    'power': row['powers'],
                    'duration': row['durations'],
                    'pre_post': row['stim_times'],
                    # 'pre_post': 'post', # hard code for no behavior np2.0
                    'unit_id': unit_id
                }
            )
            # Ensure wf_curr is not empty before extracting values
            if not wf_curr.empty:
                euc_dist_curr, corr_curr = wf_curr.iloc[0][['euclidean_norm', 'correlation']]
            else:
                euc_dist_curr, corr_curr = None, None  # Handle missing values

            euc_dist.append(euc_dist_curr)
            corr.append(corr_curr)


    for _, row in opto_tagging_df.iterrows():
        curr_power_ind = np.where(np.array(opto_info['powers']) == row['powers'])[0][0]
        curr_mean = np.nanmean(resp_p_bl[curr_power_ind])
        mean_p.append(curr_mean)
    
    opto_tagging_df['mean_p'] = mean_p
    if len(euc_dist) > 0:
        opto_tagging_df['euclidean_norm'] = euc_dist
        opto_tagging_df['correlation'] = corr
    else:
        opto_tagging_df['euclidean_norm'] = None
        opto_tagging_df['correlation'] = None
    # opto_tagging_dict = {key: opto_tagging_df[key].values for key in opto_tagging_df.columns}
    opto_tagging_dict = {'unit_id': unit_id, 'spike_times': spike_times, 'opto_pass': opto_pass}
    opto_tagging_dict.update(qc_dict)

    
    # plot
    if plot:
        fig = plt.figure(figsize=(12, 8))
        # select the first dimension to separate by subplots
        gs_all = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
        # plot basic firing rates first
        gs_basic = gridspec.GridSpecFromSubplotSpec(2, 3, width_ratios=[1, 1, 2], subplot_spec=gs_all[0])
        ax = fig.add_subplot(gs_basic[0, 2])
        bins = np.linspace(np.min(spike_times), np.max(spike_times), 200)
        ax.hist(spike_times, bins=bins, color='black', alpha=0.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks([])
        ax.set_title(f'Firing Rate: {qc_dict["firing_rate"]:.2f} Hz')
        ax = fig.add_subplot(gs_basic[1, 2])
        ax.scatter(spike_times, -spike_amplitude, color='b', alpha=0.5, s=1, edgecolors='none')
        ax.set_xlim(bins[0], bins[-1])
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.set_title('Spike Amplitude (uV)')
        ax.set_xlabel('Time (s)')
        # long crosscorr
        # no laser
        if crosscorr is not None:
            ax = fig.add_subplot(gs_basic[0, 0])
            if unit_id in crosscorr['long']['unit_ids_nolaser']:
                unit_ind = np.where(crosscorr['long']['unit_ids_nolaser'] == unit_id)[0][0]
                bin_size = np.mean(np.diff(crosscorr['long']['time_bins']))
                ax.bar(0.5*crosscorr['long']['time_bins'][1:] + 0.5*crosscorr['long']['time_bins'][:-1], crosscorr['long']['correlogram_nolaser'][unit_ind, unit_ind, :], width=bin_size, color='black', alpha=0.75)
            ax.set_xlim(crosscorr['long']['time_bins'][0], crosscorr['long']['time_bins'][-1])
            ax.set_title('Long Crosscorr')
            ax.set_ylabel('Session')
            # turn off x ticks
            ax.set_xticks([])
            # laser
            ax = fig.add_subplot(gs_basic[1, 0])
            if unit_id in crosscorr['long']['unit_ids_laser']:
                unit_ind = np.where(crosscorr['long']['unit_ids_laser'] == unit_id)[0][0]
                bin_size = np.mean(np.diff(crosscorr['long']['time_bins']))
                ax.bar(0.5*crosscorr['long']['time_bins'][1:] + 0.5*crosscorr['long']['time_bins'][:-1], crosscorr['long']['correlogram_laser'][unit_ind, unit_ind, :], width=bin_size, color='red', alpha=0.75)
            ax.set_xlim(crosscorr['long']['time_bins'][0], crosscorr['long']['time_bins'][-1])
            ax.set_xlabel('Lag (ms)')
            ax.set_ylabel('Opto')

            # short crosscorr
            # no laser
            ax = fig.add_subplot(gs_basic[0, 1])
            if unit_id in crosscorr['short']['unit_ids_nolaser']:
                unit_ind = np.where(crosscorr['short']['unit_ids_nolaser'] == unit_id)[0][0]
                bin_size = np.mean(np.diff(crosscorr['short']['time_bins']))
                ax.bar(0.5*crosscorr['short']['time_bins'][1:] + 0.5*crosscorr['short']['time_bins'][:-1], crosscorr['short']['correlogram_nolaser'][unit_ind, unit_ind, :], width=bin_size, color='black', alpha=0.5)
            ax.set_xlim(crosscorr['short']['time_bins'][0], crosscorr['short']['time_bins'][-1])
            ax.set_title('Short Crosscorr')
            ax.set_xticks([])
            # laser
            ax = fig.add_subplot(gs_basic[1, 1])
            if unit_id in crosscorr['short']['unit_ids_laser']:
                unit_ind = np.where(crosscorr['short']['unit_ids_laser'] == unit_id)[0][0]
                bin_size = np.mean(np.diff(crosscorr['short']['time_bins']))
                ax.bar(0.5*crosscorr['short']['time_bins'][1:] + 0.5*crosscorr['short']['time_bins'][:-1], crosscorr['short']['correlogram_laser'][unit_ind, unit_ind, :], width=bin_size, color='red', alpha=0.5)
            ax.set_xlim(crosscorr['short']['time_bins'][0], crosscorr['short']['time_bins'][-1])
            ax.set_xlabel('Lag (ms)')
            plt.tight_layout()



        gs = gridspec.GridSpecFromSubplotSpec(1, len(opto_info[dim_1]), width_ratios=[1]*len(opto_info[dim_1]), subplot_spec=gs_all[1])
        max_p_all = []
        max_lat_all = []
        pulse_width = opto_df['duration'].mode()[0]

        colors = ["blue", "white", "red"]
        b_w_r_cmap = LinearSegmentedColormap.from_list("b_w_r", colors)
        
        for power_ind, curr_power in enumerate(opto_info[dim_1]):
            gs_sub_raster = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[power_ind], height_ratios=[1, 1], width_ratios=[1,2])
            curr_pulse_width = pulse_width
            curr_resp_p = resp_p[power_ind, :, :, np.sort(opto_info[opto_info['dimensions'][3]+'s'])==curr_pulse_width, :]
            if np.isnan(curr_resp_p).all():
                curr_pulse_width = pulse_width-1
                curr_resp_p = resp_p[power_ind, :, :, np.sort(opto_info[opto_info['dimensions'][3]+'s'])==curr_pulse_width, :]
                if np.isnan(curr_resp_p).all():
                    curr_pulse_width = pulse_width+1
                    curr_resp_p = resp_p[power_ind, :, :, np.sort(opto_info[opto_info['dimensions'][3]+'s'])==curr_pulse_width, :]
                    if np.isnan(curr_resp_p).all():
                        continue
            curr_resp_p_bl = resp_p_bl[power_ind, :, :, np.sort(opto_info[opto_info['dimensions'][3]+'s'])==curr_pulse_width, :]
            curr_resp_lat = resp_lat[power_ind, :, :, np.sort(opto_info[opto_info['dimensions'][3]+'s'])==curr_pulse_width, :]
            curr_resp_p = np.array(curr_resp_p.tolist())
            curr_resp_p_bl = np.array(curr_resp_p_bl.tolist())
            curr_resp_lat = np.array(curr_resp_lat.tolist())
            # remove rows that are all nan along last axis
            # mask = np.squeeze(~np.isnan(curr_resp_p).all(axis=-1, keepdims=True))
            mask = ~np.all(np.isnan(curr_resp_p), axis=(0, 1, 2, 3, 5))
            curr_resp_lat = curr_resp_lat[:, :, :, :, mask, :]
            # mask = ~np.isnan(curr_resp_p).all(axis=-1).flatten()
            # curr_resp_lat = curr_resp_lat[:, :, :, :, mask, :]


            curr_resp_p = curr_resp_p[:, :, :, :, mask, :]
            curr_resp_p_bl = curr_resp_p_bl[:, :, :, :, mask, :]
            # if multi-sites, use sites as frist dimention to plot, else if multi-time, use multi-time as first dimension
            # if curr_resp_p.shape[1] > 1:
            #     curr_resp_p = np.squeeze(curr_resp_p)
            #     curr_resp_lat = np.squeeze(curr_resp_lat)
            # elif any(dim > 1 for dim in list(curr_resp_p.shape)[2:-1]):
            # if more than 2 dimensions>1, remain the first dimension and max over the rest
            if np.sum(np.array([dim > 1 for dim in list(curr_resp_p.shape)[0:-1]]))>1:
                dims = tuple(i for i in range(curr_resp_p.ndim) if i != 0 and i != 1 and i !=curr_resp_p.ndim) # exclude site and power and pulses
                curr_resp_p_colormap = np.max(curr_resp_p, axis = dims) # shape is 2d, sites x pulses
                curr_resp_p_colormap = np.squeeze(curr_resp_p_colormap)
            else:
                curr_resp_p_colormap = np.squeeze(curr_resp_p)

            if curr_resp_p_colormap.ndim == 1:
                curr_resp_p_colormap = curr_resp_p_colormap[:, np.newaxis].T
            # plot response p as heatmap
            ax = fig.add_subplot(gs_sub_raster[0,0])
            ax.imshow(curr_resp_p_colormap, cmap=my_red, aspect='auto', vmin=0, vmax=1)
            # find max response p along last dimention (pulses) then find max conditions
            max_ind = max_index_ndarray(curr_resp_p)
            max_ind_bl = max_index_ndarray(curr_resp_p_bl)
            site_ind = max_ind_bl[1]
            site_ind_bl = max_ind_bl[1]
            max_site = opto_info['sites'][site_ind_bl]
            max_p_all.append(curr_resp_p_bl[max_ind_bl])
            max_lat_all.append(curr_resp_lat[max_ind_bl])

            p_train_to_plot = curr_resp_p[max_ind_bl[:-1]]
            p_train_to_plot_bl = curr_resp_p_bl[max_ind_bl[:-1]]
            lat_train_to_plot = curr_resp_lat[max_ind_bl[:-1]]

            if np.shape(curr_resp_p)[-2]==1:
                curr_pre_post = 'post'
            elif max_ind_bl[-2] == 0:
                curr_pre_post = 'pre'
            else:
                curr_pre_post = 'post'
            
            laser_times_curr = np.sort(np.concatenate([opto_df.query('site == @max_site and power == @curr_power and pre_post == "pre"')['time'].values, 
                                        opto_df.query('site == @max_site and power == @curr_power and pre_post == "post"')['time'].values], axis = 0), )[::-1] 
            # laser_times_curr = np.sort(np.concatenate([opto_df.query('site == @max_site and power == @curr_power and pre_post == "pre"')['time'].values, 
            #                             opto_df.query('site == @max_site and power == @curr_power and pre_post == @curr_pre_post')['time'].values], axis = 0), )[::-1] 
            if len(laser_times_curr) > 0:
                raster_df = align.to_events(spike_times, laser_times_curr, (-0.5, 1.5), return_df=True)
                ax = fig.add_subplot(gs_sub_raster[0,1])
                plot_raster_bar(raster_df, ax)
                ax.set_title(f'Power: {curr_power}; Site: {max_site}')
                ax.set_xlim(-0.5, 1)
                ax.set_ylim(0, len(laser_times_curr)+1)
                ax.set_yticks([])
                # ax.axis('off')
                ax.set(xlabel='Time (s)')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                for j in range(opto_info['num_pulses'][0]):
                    x = j * 1/opto_info['freqs'][0]
                    rect = patches.Rectangle((x, 0), opto_info['durations'][0]/1000, len(laser_times_curr), color='r', alpha=0.5, edgecolor=None)
                    ax.add_patch(rect)

            # plot waveform
            gs_sub_waveform = gridspec.GridSpecFromSubplotSpec(6, 2, subplot_spec=gs[power_ind], hspace=0.75)  
            ax = fig.add_subplot(gs_sub_waveform[3,0])
            if opto_wf is not None:
                wf_resp = opto_wf.query('unit_id == @unit_id and site == @max_site and power == @curr_power and duration == @curr_pulse_width and pre_post == @curr_pre_post')
                wf_spont = opto_wf.query('unit_id == @unit_id and pre_post == @curr_pre_post and spont == 1')
                if len(wf_spont) > 0:
                    ax.plot(wf_spont['peak_waveform'].values[0], color='black', alpha = 0.5)
                if len(wf_resp) > 0:
                    # plot the response waveform
                    ax.plot(wf_resp['peak_waveform'].values[0], color='red')
                    ax.set_title(f'E: {wf_resp["euclidean_norm"].values[0]:.2f} C: {wf_resp["correlation"].values[0]:.2f}')
            
            ax = fig.add_subplot(gs_sub_waveform[3,1])
            shifted_cmap = shiftedColorMap(b_w_r_cmap, np.nanmin(waveform), np.nanmax(waveform), 'shifted_b_w_r');
            cax = ax.imshow(waveform[:, 0:int(0.5*np.shape(waveform)[1])], extent = [waveform_params['samples_to_keep'][0], waveform_params['samples_to_keep'][1], 2*waveform_params['y_neighbors_to_keep']+1, 0], cmap=shifted_cmap, aspect='auto');
            fig.colorbar(cax, ax=ax)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
            ax.axvline(waveform_params['samples_to_keep'][1]-waveform_params['samples_to_keep'][0], color='black', linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # plot best response p and latency
            # location_max = np.argmax(np.mean(curr_resp_p_max, axis = -1))
            # max_site = opto_info['sites'][location_max]
            ax = fig.add_subplot(gs_sub_waveform[4,:])
            ax.plot(p_train_to_plot, label='raw')
            ax.plot(p_train_to_plot_bl, color='black', linewidth=0.5, label='-bl')
            # ax.set_ylim(baseline, 1)
            # ax.axhline(resp_thresh, color='black', linestyle='--', linewidth=0.5)
            if power_ind == 0:
                ax.legend()
            ax.set_title(f'P(resp) Site: {max_site}')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax = fig.add_subplot(gs_sub_waveform[5,:])
            ax.plot(1000*lat_train_to_plot)
            ax.set_ylim(0, 1000*opto_info['resp_win'])
            ax.axhline(1000*lat_thresh, color='black', linestyle='--', linewidth=0.5)
            ax.set_title(f'Latency (ms)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        max_p_all = np.max(np.array(max_p_all))
        max_lat_all = np.max(np.array(max_lat_all))
        pass_qc_curr = (qc_dict['isi_violations_ratio'] < 0.5) & \
                (qc_dict['amplitude_cutoff'] < 0.05) & \
                (qc_dict['decoder_label'] != 'noise') & \
                (qc_dict['decoder_label'] != 'artifact')
                
                # (qc_dict['firing_rate'] > 0.1) & \
                # # (qc_dict['presence_ratio'] > 0.95) & \
        plt.suptitle(f"Unit {unit_id} Depth: {qc_dict['depth']:.2f} RespWin: {opto_info['resp_win']} s  pResp: {max_p_all:.2f} Lat: {max_lat_all:.2f} pass_qc {pass_qc_curr}")
        plt.tight_layout()
    return fig, opto_tagging_dict, opto_tagging_df
#%%
def opto_plotting_session(session, data_type, target, resp_thresh=0.8, lat_thresh=0.015, plot = False, target_unit_ids=None, ephys_cut = False, save = False, drift_cut = True):
    session_dir = session_dirs(session)
    session_qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(session_qm_file) as f:
        qm = json.load(f)
    
    sorting = si.load(session_dir[f'curated_dir_{data_type}'])
    we = si.load(session_dir[f'postprocessed_dir_{data_type}'], load_extensions=False)
    spike_amplitude = we.get_extension('spike_amplitudes').get_data(outputs="by_unit")[0]
    unit_ids = sorting.get_unit_ids()

    # load quality metrics from nwb
    if os.path.exists(session_dir[f'nwb_dir_{data_type}']):
        nwb = load_nwb(session_dir[f'nwb_dir_{data_type}'])
        if 'depth' in nwb.units.colnames:
            unit_qc = nwb.units[:][['ks_unit_id', 'isi_violations_ratio', 'firing_rate', 'presence_ratio', 'amplitude_cutoff', 'decoder_label', 'depth', 'snr']]
        else:
            locations = we.get_extension('unit_locations').get_data()[:, 1]
            unit_qc = nwb.units[:][['ks_unit_id', 'isi_violations_ratio', 'firing_rate', 'presence_ratio', 'amplitude_cutoff', 'decoder_label', 'snr']]
            unit_qc['depth'] = locations
    else:
        print('No nwb file found.') 
    del we
    # change all strings in unit_qc to float 
    unit_qc = unit_qc.replace("<NA>", pd.NA)
    unit_qc['waveform_mean'] = nwb.units[:]['waveform_mean']
    unit_qc['waveform_sd'] = nwb.units[:]['waveform_sd']
    unit_qc = unit_qc.apply(pd.to_numeric, errors='ignore') 
    pass_qc = (unit_qc['isi_violations_ratio'] < 0.4) & \
            (unit_qc['decoder_label'] != 'noise') & \
            (unit_qc['decoder_label'] != 'artifact')
            #     (unit_qc['firing_rate'] > 0.1) & \
            # (unit_qc['presence_ratio'] > 0.95) & \
            # (unit_qc['amplitude_cutoff'] < 0.05) & \
    pass_qc = pass_qc.values
    pass_qc = {unit_id: pass_qc_curr for unit_id, pass_qc_curr in zip(unit_ids, pass_qc)}
    print(f'{sum(pass_qc.values())} out of {len(pass_qc)} units pass quality control')


    # load waveforms info
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], session+'_waveform_params.json')) as f:
        waveform_params = json.load(f)
    # print(waveform_params)

    # load opto responses
    with open(os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], 'spiketimes.pkl'), 'rb') as f:
        spiketimes = pickle.load(f)
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_responses_{target}.pkl'), 'rb') as f:
        opto_responses = pickle.load(f)
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_waveforms_{target}.pkl'), 'rb') as f:
        waveforms = pickle.load(f)
    
    # load crosscorr
    if os.path.exists(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_correlogram_laser_short.pkl')): 
        with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_correlogram_laser_short.pkl'), 'rb') as f:
            crosscorr_short = pickle.load(f)
        with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_correlogram_laser_long.pkl'), 'rb') as f:
            crosscorr_long = pickle.load(f)
        crosscorr = {'short': crosscorr_short, 'long': crosscorr_long}
    else:
        crosscorr = None
    # load waveforms from pkl
    opto_wf_pkl = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_waveform_metrics.pkl')
    if os.path.exists(opto_wf_pkl):
        with open (opto_wf_pkl, 'rb') as f:
            opto_wf = pickle.load(f)
        opto_wf = opto_wf.apply(pd.to_numeric, errors='ignore')
    else: 
        opto_wf = None

    # load ephys cut
    if ephys_cut:
        drift_file = os.path.join(session_dir['opto_dir_curated'], f'{session}_opto_drift_tbl.csv')
        opto_drift_tbl = pd.read_csv(drift_file)
    
    # load opto stimulation information
    opto_df = pd.read_csv(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_session_{target}.csv'))
    with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_info_{target}.json')) as f:
        opto_info = json.load(f)
    
    # opto_info = {key: np.sort(value) for key, value in opto_info.items()}
    pulse_width = opto_df['duration'].mode()[0]
    colors = ["blue", "white", "red"]
    b_w_r_cmap = LinearSegmentedColormap.from_list("b_w_r", colors)
    opto_pass = []
    if target_unit_ids is None:
        target_unit_ids = unit_ids
    target_pass_qc = []
    opto_tagging_df_sess = pd.DataFrame()
    opto_tagging_df_sess_metrics = pd.DataFrame()
    for unit_id in target_unit_ids:
        spiketimes_curr = spiketimes[unit_id]
        spike_amplitude_curr = spike_amplitude[unit_id]
        opto_df_curr = opto_df

        if ephys_cut:
            if unit_id in opto_drift_tbl['unit_id'].tolist():
                ephys_cut_curr = opto_drift_tbl.query('unit_id == @unit_id')['ephys_cut'].values[0]

                string_list = ephys_cut_curr.replace("nan", "null")
                ephys_cut_curr = json.loads(string_list)
                ephys_cut_curr = np.array([np.nan if x is None else x for x in ephys_cut_curr])

                ephys_cut_curr[np.isnan(ephys_cut_curr)] = np.array(qm['ephys_cut'])[np.isnan(ephys_cut_curr)]
                spike_amplitude_curr = spike_amplitude_curr[(spiketimes_curr > ephys_cut_curr[0]) & (spiketimes_curr < ephys_cut_curr[1])]
                spiketimes_curr = spiketimes_curr[(spiketimes_curr > ephys_cut_curr[0]) & (spiketimes_curr < ephys_cut_curr[1])]
                opto_df_curr = opto_df_curr[(opto_df_curr['time'] > ephys_cut_curr[0]) & (opto_df_curr['time'] < ephys_cut_curr[1])]

    
    #     if pass_qc[unit_id]:   
        qc_dict = unit_qc.loc[unit_qc['ks_unit_id'] == unit_id].iloc[0].to_dict()
        fig, opto_tagging_dict_curr, opto_tagging_df_curr = opto_plotting_unit(unit_id, spiketimes_curr, spike_amplitude_curr, # -1 is for session behavior_782394_2025-04-22_10-53-28
                                                            waveforms[unit_id], opto_wf, qc_dict, crosscorr,
                                                            opto_responses['resp_p'][unit_id], opto_responses['resp_lat'][unit_id], 
                                                            opto_df_curr, opto_info, qm,
                                                            waveform_params, 
                                                            dim_1 = 'powers', resp_thresh=resp_thresh, lat_thresh=lat_thresh, plot=plot)
        if fig is not None:
            fig.savefig(os.path.join(session_dir[f'opto_dir_fig_{data_type}'], f'unit_{unit_id}_pulse_width_{pulse_width}_opto_tagging.pdf'))
            fig.savefig(os.path.join(session_dir[f'opto_dir_fig_{data_type}'], f'unit_{unit_id}_pulse_width_{pulse_width}_opto_tagging.svg'))
        plt.close('all')
        opto_tagging_df_sess = pd.concat([opto_tagging_df_sess, pd.DataFrame([opto_tagging_dict_curr])], ignore_index=True)
        # opto_tagging_df_curr = opto_tagging_df_curr.dropna(how='all', axis=1)
        if not opto_tagging_df_curr.empty:
            opto_tagging_df_sess_metrics = pd.concat([opto_tagging_df_sess_metrics, opto_tagging_df_curr], ignore_index=True) # tidy
        target_pass_qc.append(pass_qc[unit_id])
        opto_pass_curr = True
        # if opto_tagging_dict_curr['opto_pass'] is None:
        #     opto_pass_curr = False
        opto_pass.append(opto_tagging_dict_curr['opto_pass'])

    opto_tagging_df_sess['default_qc'] = target_pass_qc
    if save:
        opto_tagging_data = {'opto_tagging_df': opto_tagging_df_sess, 'opto_tagging_df_metrics': opto_tagging_df_sess_metrics}
        with open(os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging_metrics.pkl'), 'wb') as f:
            pickle.dump(opto_tagging_data, f)
    unit_count_pass = np.sum(np.array(target_pass_qc) & np.array(opto_pass))
    print(f'{unit_count_pass} out of {len(target_pass_qc)} units pass quality control and opto tagging')
    if plot:
        combine_pdf_big(session_dir[f'opto_dir_fig_{data_type}'], os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging.pdf'))
        # merge_pdfs(session_dir[f'opto_dir_fig_{data_type}'], os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_tagging.pdf'))
    # both pass qc and opto tagging

    # opto_tagging_df_sess['opto_pass'] = opto_pass
    # target_qc['target_pass_qc'] = target_pass_qc

    return opto_tagging_df_sess

# %%
if __name__ == "__main__":

    target = 'soma'
    data_type = 'raw' 
    resp_thresh = 0.3
    lat_thresh = 0.02 
    # session level  
    # session = 'behavior_754897_2025-03-14_11-28-53'
    # opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= [125], plot = True, ephys_cut = False, save=False)
    # session_dir = session_dirs(session)
    session_assets = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
    session_list = session_assets['session_id']
    session_list = [session for session in session_list if isinstance(session, str)]
    #ind = [i for i, session in enumerate(session_list) if session == 'behavior_751181_2025-02-25_12-12-35'] 
    #ind = ind[0]
    #session = 'behavior_754897_2025-03-13_11-20-42'
    import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore")
    # for session in session_list:
    #     print(session)
    #     session_dir = session_dirs(session)
    #     if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
    #         if session_dir['nwb_dir_curated'] is not None:
    #             data_type = 'curated'
    #             opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = False, ephys_cut = False, save=True)
            # elif session_dir['curated_dir_raw'] is not None:
            #     data_type = 'raw'
            #     opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = True, ephys_cut = False, save=True)
    # session = 'behavior_751181_2025-02-27_11-24-47'
# /root/capsule/scratch/782394/behavior_782394_2025-04-24_12-07-34/ephys/opto/raw/figures    # opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids=[62], plot = True, ephys_cut = False, save=True)
     
    
    from joblib import Parallel, delayed
    import os
    import pickle
    data_type = 'raw'
    def process(session, data_type): 
        print(f'Starting {session}')
        finish = False
        # session_dir = session_dirs(session)
        # # Build file path
        # file_path = os.path.join(session_dir[f'opto_dir_{data_type}'],
        #                         f'{session}_opto_tagging_metrics.pkl')
        # if os.path.exists(file_path):
        #     with open(file_path, "rb") as f:
        #         data = pickle.load(f)
        #     if isinstance(data, dict) and "opto_tagging_df_metrics" in data:
        #         finish = True
        #         print(f"{session} opto tagging up to date")
# behavior_714116_2024-08-27_11-29-48 opto tagging up to date
# Reprocess opto tagging behavior_714116_2024-08-27_11-29-48
        # if not finish:
        #     # if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
        #     print(f"Process opto tagging {session}")
        #     # print(session_dir[f'curated_dir_{data_type}'])
        #     if session_dir[f'curated_dir_{data_type}'] is not None:
        opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = True, save=True)

            # else:
            # #     print(f'No curated data found for {session}') 
            # elif session_dir['curated_dir_raw'] is not None:
            #     data_type = 'raw' 
            #     opto_tagging_df_sess = opto_plotting_session(session, data_type, target, resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = True, save=True)
    # session_list = ['behavior_791691_2025-06-24_13-21-29', 'behavior_791691_2025-06-26_13-39-26', 'behavior_784806_2025-06-17_14-59-23']
    # Parallel(n_jobs=2)(delayed(process)(session, data_type) for session in session_list)
    opto_plotting_session('behavior_814515_2025-10-24_13-22-07', 'raw', 'soma', resp_thresh=resp_thresh, lat_thresh=lat_thresh, target_unit_ids= None, plot = True, save=True)
    # for session in session_list:
    #     process(session, data_type)
    # print(session_list[-13]) 


    

# %%
