# %%
import sys
import os

curr_path = os.path.dirname(os.path.abspath(__file__))

# Add repoA to sys.path at the beginning
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)
else:
    # move it to the front if it's already there
    sys.path.remove(curr_path)
    sys.path.insert(0, curr_path)

from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
from open_ephys.analysis import Session
import datetime
from aind_ephys_rig_qc.temporal_alignment import search_harp_line
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from scipy.io import loadmat
from scipy.stats import zscore
import ast
from utils.plot_utils import combine_pdf_big

from open_ephys.analysis import Session
from pathlib import Path
import glob

import json
import seaborn as sns
from PyPDF2 import PdfMerger
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import plot_foraging_session
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from hdmf_zarr.nwb import NWBZarrIO
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
from utils.ephys_functions import*
import pandas as pd
import pickle
import scipy.stats as stats
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
import time
import shutil 
from aind_ephys_utils import align

def plot_unit_beh_session(session, data_type = 'raw', align_name = 'go_cue', curate_time=True, opto_only=True,
                        model_name = None, 
                        formula = 'spikes ~ 1 + outcome + choice + Qchosen',
                        pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                        units  = None):
    # %%
    # load behavior data
    session_dir = session_dirs(session, model_name = model_name) 
    session_df = makeSessionDF(session, model_name = model_name)
    tblTrials = get_session_tbl(session)
    pdf_dir = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
    # remove all files in pdf_dir if exists
    for f in os.listdir(pdf_dir):
        path = os.path.join(pdf_dir, f)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # Remove files
        elif os.path.isdir(path):
            shutil.rmtree(path)  # Remove subdirectories
        
    # %%
    qm_dir = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_dir, 'r') as f:
        qm = json.load(f)

    # %% load units 
    unit_tbl = get_unit_tbl(session, data_type)

    # %%
    fs = 14
    fsLegend = 8

    # %%
    colors = ["blue", "white", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("blue_white_red", colors)
    
    def plot_unit(unit_id):
        unit_drift = load_drift(session, unit_id, data_type=data_type)
        spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
        qc_pass = unit_tbl.query('unit_id == @unit_id')['default_qc'].values[0]
        opto_pass = unit_tbl.query('unit_id == @unit_id')['opto_pass'].values[0]
        session_df_curr = session_df.copy()
        spike_times_curr = spike_times.copy()
        tblTrials_curr = tblTrials.copy()
        if curate_time:
            if unit_drift is not None:
                if unit_drift['ephys_cut'][0] is not None:
                    spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                    session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                    tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
                if unit_drift['ephys_cut'][1] is not None:
                    spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                    session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                    tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) == 0:
            # return None and exit function
            print(f'No session data for unit {unit_id}')
            fig = plt.figure(figsize=(20, 10))
            plt.suptitle(f'Unit{str(unit_id)} Aligned to {align_name} default qc: {qc_pass} maybe opto: {opto_pass} No behavior', fontsize = 20)
        else:
            print(f'Plotting unit {unit_id}')      
            if align_name == 'go_cue':
                align_time = session_df_curr['go_cue_time'].values
                align_time_all = tblTrials_curr['goCue_start_time'].values
            elif align_name == 'response':
                align_time = session_df_curr['choice_time'].values
                align_time_all = tblTrials_curr['reward_outcome_time'].values
            spike_matrix, slide_times = get_spike_matrix(spike_times_curr, align_time, 
                                                        pre_event=pre_event, post_event=post_event, 
                                                        binSize=binSize, stepSize=stepSize)
            spike_matrix_LM, slide_times_LM = get_spike_matrix(spike_times_curr, align_time, 
                                                        pre_event=-2, post_event=2.5, 
                                                        binSize=0.5, stepSize=0.2)
            spike_matrix_all, slide_times = get_spike_matrix(spike_times_curr, align_time_all, 
                                                        pre_event=pre_event, post_event=post_event, 
                                                        binSize=binSize, stepSize=stepSize)

            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(2, 7, height_ratios=[3, 1], wspace=0.35, hspace=0.2)
            # plot session
            ax = fig.add_subplot(gs[0, 0]) 
            choice_history, reward_history, p_reward, autowater_offered, trial_time = get_history_from_nwb(session_df_curr)
            _, axes = plot_foraging_session(  # noqa: C901
                                            choice_history,
                                            reward_history,
                                            p_reward = p_reward,  
                                            autowater_offered = autowater_offered,
                                            ax = ax,
                                            # legend=False,
                                            vertical=True,
                                            ) 
            for ax in axes:
                ax.set_ylim(0, len(session_df_curr))
            ax.set_ylim(0, len(session_df_curr))
            # from start to end
            ax = fig.add_subplot(gs[0, 1])  
            df = align.to_events(spike_times_curr, align_time, (pre_event, post_event), return_df=True)
            plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
            ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
            ax.set_xlim(pre_event, post_event)
            ax.set_ylabel('Trial number')
            ax.tick_params(axis='both', which='major')
            ax.set_ylim(0, len(session_df_curr))


            # waveform
            ax = fig.add_subplot(gs[1, 0])  
            waveform = unit_tbl.query('unit_id == @unit_id')['waveform_mean'].values[0]
            peakChannel = np.argmin(np.min(waveform, axis=0))
            peakWaveform = waveform[:,peakChannel]
            peakSD = unit_tbl.query('unit_id == @unit_id')['waveform_mean'].values[0][:,peakChannel]
            timeWF = np.array(range(len(peakWaveform)))-90
            ax.plot(timeWF, peakWaveform, color = 'k')
            ax.fill_between(timeWF, peakWaveform - peakSD/np.sqrt(499), peakWaveform + peakSD/np.sqrt(499), color = 'k', alpha = 0.1)
            ax.axhline(y=0, color = 'r', ls = '--')
            ax.set_xlabel('Time (ms)', fontsize = fs)
            ax.set_ylabel(r'$\mu$-Plot')

            # reward and no reward
            outcome_int = [int(item) for item in session_df_curr['outcome'].tolist()]
            bins = [-1, 0.5, 1.5]
            labels = ['no reward', 'reward']
            fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                            align_time, 
                                            outcome_int, # sorted by certain value
                                            bins,
                                            labels, 
                                            custom_cmap,
                                            fig,
                                            gs[0, 2],
                                            tb=pre_event,
                                            tf=post_event,
                                            time_bin=stepSize,
                                            )
            ax1.set_title('Reward vs No Reward', fontsize = fs+2)

            # left and right
            side_int = [int(item) for item in session_df_curr['choice'].tolist()]
            bins = [-1.5, 0.5, 1.5]
            labels = ['left', 'right']
            fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                            align_time, 
                                            side_int, # sorted by certain value
                                            bins,
                                            labels,
                                            custom_cmap,
                                            fig,
                                            gs[0, 3],
                                            tb=pre_event,
                                            tf=post_event,
                                            time_bin=stepSize,
                                            )
            ax1.set_title('Right vs Left', fontsize = fs+2)

            # rpe
            target_var = 'pe'
            bin_counts = 4
            if target_var in session_df_curr.columns.to_list():
                bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
                # bins = [-1.0001, -0.5, 0, 0.5, 1.0001]
                bins[0] = bins[0] - 0.0001
                bins[-1] = bins[-1] + 0.0001
                labels = ['1', '2', '3', '4']
                
                fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                                align_time,  
                                                session_df_curr[target_var].values, # sorted by certain value
                                                bins,
                                                labels,
                                                custom_cmap,
                                                fig,
                                                gs[0, 4],
                                                tb=pre_event,
                                                tf=post_event,
                                                time_bin=stepSize,
                                                )
                # ax.set_yticks([])
                # ax.set_ylabel(label, fontsize = fs)
                ax1.set_title(target_var, fontsize = fs+2)


                # Qchosen
                target_var = 'Qchosen'
                bin_counts = 4
                bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
                # bins = [-1.0001, -0.5, 0, 0.5, 1.0001]
                bins[0] = bins[0] - 0.0001
                bins[-1] = bins[-1] + 0.0001
                labels = ['1', '2', '3', '4']
                fig, ax1, ax2 = plot_raster_rate(spike_times_curr,
                                                align_time, 
                                                session_df_curr[target_var].values, # sorted by certain value
                                                bins,
                                                labels,
                                                custom_cmap,
                                                fig,
                                                gs[0, 5],
                                                tb=pre_event,
                                                tf=post_event,
                                                time_bin=stepSize,
                                                )
                ax1.set_title(target_var, fontsize = fs+2)

            # stay vs switch
            target_var = 'svs'
            # bins = np.quantile(session_df_curr[target_var].values, np.linspace(0, 1, bin_counts+1))
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['stay', 'switch']
            # ax = fig.add_subplot(gs[1, 1])
            fig, ax = plot_rate(
                                spike_matrix,
                                slide_times, 
                                session_df_curr[target_var].values,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 1],
                                tb=pre_event,
                                tf=post_event,
                                )

            ax.set_yticks([])
            ax.set_title(target_var, fontsize = fs+2)

            # right rwd vs no rwd
            target_var = 'outcome'
            spike_matrix_curr = spike_matrix[session_df_curr['choice'].values == 1, :]
            focus_var = session_df_curr[session_df_curr['choice'].values == 1][target_var].values
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['no rwd', 'rwd']
            fig, ax = plot_rate(
                                spike_matrix_curr,
                                slide_times,
                                focus_var,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 2],
                                tb=pre_event,
                                tf=post_event,
                                )
            ax.set_title('Right: rwd nrwd', fontsize = fs+2)

            # left rwd vs no rwd
            target_var = 'outcome'
            spike_matrix_curr = spike_matrix[session_df_curr['choice'].values == 0, :]
            focus_var = session_df_curr[session_df_curr['choice'].values == 0][target_var].values
            bins = [-1.0001, 0.5, 1.0001]
            labels = ['no rwd', 'rwd']
            fig, ax = plot_rate(
                                spike_matrix_curr,
                                slide_times,
                                focus_var,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 3],
                                tb=pre_event,
                                tf=post_event,
                                )
            ax.set_title('Left: rwd nrwd', fontsize = fs+2)

            # go vs miss
            map_value = tblTrials_curr['animal_response'].values!=2
            bins = [-0.5, 0.5, 1.5]
            labels = ['miss', 'go']
            fig, ax = plot_rate(
                                spike_matrix_all,
                                slide_times,
                                map_value,
                                bins,
                                labels,
                                custom_cmap,
                                fig,
                                gs[1, 4],
                                tb=pre_event,
                                tf=post_event,
                                )


            
            if len(session_df_curr) > 100 and np.sum((spike_times_curr>=session_df_curr['go_cue_time'].values[0]) & (spike_times_curr<=session_df_curr['go_cue_time'].values[-1]))/(session_df_curr['go_cue_time'].values[-1] - session_df_curr['go_cue_time'].values[0]) > 0.1:
                # rwd history
                # fr with regression with rwd history
                target_var = 'outcome'
                vector = session_df_curr[target_var].values
                align_time = session_df_curr['choice_time'].values
                _, events_id, _ = align.to_events(spike_times_curr, align_time, (0, 1.5), return_df=False) 
                spike_counts = [np.sum(events_id==curr_id) for curr_id in range(len(align_time))]
                spike_counts = stats.zscore(np.array(spike_counts))
                trials_back = [0, 2]
                ax = fig.add_subplot(gs[1, 5])
                try:
                    coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back)
                    ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'k', lw = 2)
                    ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'k', alpha = 0.25, edgecolor = None)
                    ax.axhline(y=0, color = 'r', ls = '--')
                    ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10, zorder = 2)
                    ax.set_title('Spikes~rwd history', fontsize = fs+2)
                    ax.set_xlabel('Trials back')
                except:
                    ax.plot(range(trials_back[0], trials_back[1] + 1), np.zeros(trials_back[1]-trials_back[0]+1), c = 'k', lw = 2, label = 'failed')
                    ax.set_title('Spikes~rwd history failed', fontsize = fs+2)


                # only on left trials
                trials_back = [0, 2]
                ax = fig.add_subplot(gs[1, 6])
                ax.set_title('Spikes~rwd hist L/R', fontsize = fs+2)
                ax.set_xlabel('Trials back')
                if np.sum(session_df_curr['choice'].values == 0) >= 40:
                    try:
                        coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back, sub_selection=session_df_curr['choice'].values == 0)
                        ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'm', lw = 2, label = 'left')
                        ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'm', alpha = 0.25, edgecolor = None)
                        ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10)
                        ax.axhline(y=0, color = 'r', ls = '--')
                        ax.legend()
                    except:
                        ax.plot(range(trials_back[0], trials_back[1] + 1), np.zeros(trials_back[1]-trials_back[0]+1), c = 'm', lw = 2, label = 'left failed')

                # only on right trials
                if np.sum(session_df_curr['choice'].values == 1) >= 40:
                    try:
                        coeffs, pvals, tvals, conf_int = regression_rwd(spike_counts, vector, trials_back = trials_back, sub_selection=session_df_curr['choice'].values == 1)
                        ax.plot(range(trials_back[0], trials_back[1] + 1), coeffs, c = 'c', lw = 2, label = 'right')
                        ax.fill_between(range(trials_back[0], trials_back[1] + 1), conf_int[:, 0], conf_int[:, 1], color = 'c', alpha = 0.25, edgecolor = None)
                        ax.scatter(np.array(range(trials_back[0], trials_back[1] + 1))[pvals<0.05], coeffs[pvals<0.05], c = 'r', s = 10)
                        ax.axhline(y=0, color = 'r', ls = '--')
                        ax.set_title('Spikes~rwd hist L/R', fontsize = fs+2)
                        ax.set_xlabel('Trials back')
                        ax.legend()
                    except:
                        ax.plot(range(trials_back[0], trials_back[1] + 1), np.zeros(trials_back[1]-trials_back[0]+1), c = 'c', lw = 2, label = 'right failed')

                # plot regresssions
                gs = gridspec.GridSpec(3, 7, height_ratios=[1, 1, 1], wspace=0.3, hspace=0.3)
                ax = fig.add_subplot(gs[0,-1])
                try: 
                    regressors, TvCurrU, PvCurrU, EvCurrU = fitSpikeModelG(session_df_curr, spike_matrix_LM, formula)
                    TvCurrUSig = TvCurrU.copy()
                    TvCurrUSig[PvCurrU>=0.05] = np.nan
                    cmap = plt.get_cmap('viridis')
                    colors = cmap(np.linspace(0, 1, len(regressors)))
                    for regress in range(1, len(regressors)):
                        ax.plot(slide_times_LM, TvCurrU[:, regress], lw = 2, color = colors[regress,], label = regressors[regress])
                        ax.plot(slide_times_LM, TvCurrUSig[:, regress], lw = 4, color = colors[regress,])
                    ax.legend(fontsize = fsLegend)
                    ax.set_xlabel(f'Time from {align_name} (s)')
                    ax.set_title('T-stats', fontsize = fs)

                    ax = fig.add_subplot(gs[1,-1])
                    for regress in range(1, len(regressors)):
                        ax.plot(slide_times_LM, -np.log10(PvCurrU[:, regress]), lw = 1, color = colors[regress,], label = regressors[regress])

                    plt.axhline(y = -np.log10(0.05), color='r', ls = '--')
                    ax.legend(fontsize = fsLegend)
                    ax.set_xlabel(f'Time from {align_name} (s)')
                    ax.set_title('p-value', fontsize = fs)
                except:
                    print(f'Failed to fit model for unit {unit_id}')
            plt.suptitle(f'Unit{str(unit_id)} Aligned to {align_name} default qc: {qc_pass} maybe opto: {opto_pass}', fontsize = 20) 
            # plt.tight_layout()  
        return fig

    log_record_file = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name, f'{session}_unit_beh.log')
    # def process(unit_id): 
    #     try:
    #         fig = plot_unit(unit_id) 
    #         if fig is not None:
    #             fig.savefig(fname=os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name, f'unit_{unit_id}_goCue.pdf'))
    #         plt.close(fig)
    #         # write to log
    #         with open(log_record_file, 'a') as f:
    #             f.write(f'Unit {unit_id} plotted\n')
    #         # pause for 1 second
    #     except:
    #         with open(log_record_file, 'a') as f:
    #             f.write(f'Unit {unit_id} failed\n')
    #     time.sleep(1)

    def process(unit_id):
        fig = plot_unit(unit_id) 
        if fig is not None: 
            fig.savefig(fname=os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name, f'unit_{unit_id}_{align_name}.pdf')) 
            fig.savefig(fname=os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], align_name, f'unit_{unit_id}_{align_name}.svg'))
        plt.close(fig)
        time.sleep(1)



    # with Pool(processes=4) as pool:  # Ensures cleanup
    #     result = pool.map(process, unit_tbl['unit_id'].values)
    if units is None:
        if not opto_only:
            units= unit_tbl['unit_id'].values
        else:
            units = unit_tbl[unit_tbl['opto_pass']==True]['unit_id'].values


    # Parallel(n_jobs=12)(
    #     delayed(process)(unit_id)
    #     for unit_id in units
    # )
    for unit_id in units:
        process(unit_id)

    output_pdf = os.path.join(session_dirs(session)[f'ephys_dir_{data_type}'],f'{session}_unit_beh_{align_name}.pdf')

    if os.path.exists(pdf_dir):
        print(f'Combining {session}')
        combine_pdf_big(pdf_dir, output_pdf)
    
    plt.close('all')

def burst_analysis(session, data_type, units = None):
    print(f'Processing session {session} for data type {data_type}')
    unit_tbl = get_unit_tbl(session, data_type)
    session_df = get_session_tbl(session)
    session_dir = session_dirs(session, data_type)
    save_path = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], 'burst')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    if units is None:
        units = unit_tbl['unit_id'].tolist()
    for unit_id in units:
        if not unit_tbl[unit_tbl['unit_id']==unit_id]['tagged_loc'].values[0]:
            continue
        spike_times = unit_tbl[unit_tbl['unit_id']==unit_id]['spike_times'].values[0]
        pre_event = -0.1
        post_event= 0.1
        session_df_curr = session_df.copy()
        spike_times_curr = spike_times.copy()
        unit_drift = load_drift(session, unit_id, data_type=data_type)

        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) <=20:
            print(f'Skipping {unit_id} due to insufficient trials after drift cut.')
            continue
        # align to go cue sorted by choice time, separate by choice or not
        # from start to end
        fig = plt.figure(figsize=(14,10))
        gs = gridspec.GridSpec(2,6)
        lick_lat = session_df_curr['reward_outcome_time'].values - session_df_curr['goCue_start_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        pre_event = -0.1
        post_event= 0.1
        align_time = session_df_curr['goCue_start_time'].values
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[0, 0])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue')

        align_time = session_df_curr['reward_outcome_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[0, 1])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Choice')


        # align to go cue and sort by frist spike time
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 10), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = np.full(len(session_df_curr), np.nan)
        for i in range(len(session_df_curr)):
            spikes_in_trial = spike_df[spike_df['event_index']==i]['time']
            if len(spikes_in_trial) > 0:
                first_spike_times[i] = spikes_in_trial.min()
        # first_spike_times = np.full(len(session_df_curr), np.nan)
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        ax = fig.add_subplot(gs[0, 2])
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to Go Cue')

        # align to first spike time
        first_spike_times_abs = first_spike_times + session_df_curr['goCue_start_time'].values
        first_spike_times_abs_sorted = first_spike_times_abs[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, first_spike_times_abs_sorted, (-0.05, 0.03), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax = fig.add_subplot(gs[0, 3])
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=1, zorder = 2)
        ax.set_xlim(-0.05, 0.03)
        ax.set_ylabel('Sorted by First Spike Time to go cue')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to First Spike Time')

        # isi distribution
        ax = fig.add_subplot(gs[1, 3:5])
        isi_spikes = np.log(np.diff(spike_times_curr))
        edges = np.linspace(np.nanmin(isi_spikes), np.nanmax(isi_spikes), 50)
        ax.hist(isi_spikes, bins=edges, color='k', alpha = 0.5, density=True)
        # set xlabel to log
        ax.axvline(np.log(0.002), color='k', linestyle='--')
        ax.set_xlabel('log(Inter-spike interval (s))')
        ax.set_ylabel('Density')
        ax.set_title('log(ISI) Distribution')

        # color by isi
        first_spike_times_abs = first_spike_times + session_df_curr['goCue_start_time'].values
        first_spike_times_abs_sorted = first_spike_times_abs[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, first_spike_times_abs_sorted, (-0.05, 0.03), return_df=True)

        isi_list = df.copy()
        isi_list['isi'] = np.nan
        # infer time interval of previous spike for each spike
        for ind, row in isi_list.iterrows():
            event_index = row['event_index']
            time = row['time']
            prev_spikes = spike_times_curr[spike_times_curr < (time + first_spike_times_abs_sorted[int(event_index)])]
            if len(prev_spikes) == 0:
                isi_list.at[ind, 'isi'] = np.nan
            else:
                isi_list.at[ind, 'isi'] = time + first_spike_times_abs_sorted[[int(event_index)]] - prev_spikes[-1]

        isi_color_code = np.log(isi_list['isi'].values)
        up_bound = np.percentile(isi_color_code[~np.isnan(isi_color_code)], 95)
        low_bound = np.percentile(isi_color_code[~np.isnan(isi_color_code)], 5)
        isi_color_code = (isi_color_code - low_bound) / (up_bound - low_bound)
        isi_color_code[isi_color_code>1] = 1
        isi_color_code[isi_color_code<0] = 0

        ax.hist(np.log(isi_list['isi'].values), bins=50, color='r', alpha=0.5, density=True)
        ax.axvline(low_bound, color='b', linestyle='--')
        ax.axvline(up_bound, color='r', linestyle='--')

        ax= fig.add_subplot(gs[0,4])
        sc = ax.scatter(df.time, df.event_index, c=isi_color_code, marker= '|', s=4, zorder = 2, cmap='Reds_r')
        ax.set_xlim(-0.02, 0.03)
        ax.set_ylabel('Sorted by First Spike Time to go cue')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.set_title('Aligned to First Spike Time')

        # add colorbar
        ax = fig.add_subplot(gs[1,5])
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)

        # align to go cue and sort by frist spike time
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 100), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = spike_df.groupby('event_index')['time'].min().values
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        isi_list = df.copy()
        isi_list['isi'] = np.nan
        # infer time interval of previous spike for each spike
        for ind, row in isi_list.iterrows():
            event_index = row['event_index']
            time = row['time']
            prev_spikes = spike_times_curr[spike_times_curr < (time + align_time_firstspike_sort[int(event_index)])]
            if len(prev_spikes) == 0:
                isi_list.at[ind, 'isi'] = np.nan
            else:
                isi_list.at[ind, 'isi'] = time + align_time_firstspike_sort[[int(event_index)]] - prev_spikes[-1]

        isi_color_code = np.log(isi_list['isi'].values)
        isi_color_code = (isi_color_code - low_bound) / (up_bound - low_bound)
        isi_color_code[isi_color_code>1] = 1
        isi_color_code[isi_color_code<0] = 0

        # plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax = fig.add_subplot(gs[0, 5])
        ax.scatter(df.time, df.event_index, c=isi_color_code, marker= '|', s=3, zorder = 2, cmap = 'Reds_r')
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_xlim(-0.02, 0.03)
        ax.set_title('Aligned to Go Cue')


        pre_event = -1.7
        post_event = 0.1
        align_time = session_df_curr['goCue_start_time'].values
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[1, 0])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=3, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue')


        align_time = session_df_curr['reward_outcome_time'].values
        lick_lat[session_df_curr['animal_response'].values==2] = np.nan
        align_time_licklat_sort = align_time[np.argsort(lick_lat)]
        ax = fig.add_subplot(gs[1, 1])  
        df = align.to_events(spike_times_curr, align_time_licklat_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=3, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('Lick latency sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_licklat_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Choice')


        # align to go cue and sort by frist spike time
        align_time = session_df_curr['goCue_start_time'].values
        spike_df = align.to_events(spike_times_curr, align_time, (0, 100), return_df=True)
        # for each value in event_index, get the first spike time
        first_spike_times = spike_df.groupby('event_index')['time'].min().values
        align_time_firstspike_sort = align_time[np.argsort(first_spike_times)]
        ax = fig.add_subplot(gs[1, 2])
        df = align.to_events(spike_times_curr, align_time_firstspike_sort, (pre_event, post_event), return_df=True)
        plt.plot([0,0],[0,df.event_index.max()],'r', zorder = 1)
        ax.scatter(df.time, df.event_index, c='k', marker= '|', s=4, zorder = 2)
        ax.set_xlim(pre_event, post_event)
        ax.set_ylabel('First spike time sorted trials')
        ax.tick_params(axis='both', which='major')
        ax.set_ylim(0, len(session_df_curr))
        ax.axhline(len(align_time_firstspike_sort)-np.sum(session_df_curr['animal_response'].values==2), color='blue', linestyle='--')
        ax.set_title('Aligned to Go Cue') 

        plt.suptitle(f'Session {session}, Unit {unit_id}')
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, f'opto_{session}{unit_id}_burst_selected.pdf'), dpi=300)
        plt.close(fig)

    print(f'{session} Combining PDFs...')
    combine_pdf_big(save_path, os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], f'{session}_bursting.pdf'))
    print(f'{session} Done!')




def plot_alignments(session, data_type='curated', unit_ids=None, win_len = 0.5):
    bin_len = 0.01
    time_constant = 100
    time_window = [-1, 1.5]
    session_dir = session_dirs(session)
    unit_tbl = get_unit_tbl(session, data_type)
    model_name = 'stan_qLearning_5params'
    session_tbl_all = get_session_tbl(session)
    unit_tbl = get_unit_tbl(session, data_type=data_type)
    session_tbl = makeSessionDF(session, model_name = model_name)
    if unit_ids is None:
        unit_ids = unit_tbl[unit_tbl['opto_pass'] & unit_tbl['default_qc']]['unit_id'].tolist()
    lick_lat = session_tbl['reward_outcome_time'].values - session_tbl['goCue_start_time'].values
    lick_lat = lick_lat[session_tbl['animal_response']!=2]
    lick_lat_sort = np.argsort(lick_lat)
    outcomes = session_tbl['rewarded_historyL'] | session_tbl['rewarded_historyR']
    outcomes = outcomes[session_tbl['animal_response']!=2].values
    for unit_id in unit_ids:
        spike_times = unit_tbl[unit_tbl['unit_id']==unit_id]['spike_times'].values[0]
        session_tbl_curr = session_tbl.copy()
        session_tbl_all_curr = session_tbl_all.copy()
        spike_times_curr = spike_times.copy()
        unit_drift = load_drift(session, unit_id, data_type=data_type)
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_tbl_curr = session_tbl_curr[session_tbl_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                session_tbl_all_curr = session_tbl_all_curr[session_tbl_all_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_tbl_curr = session_tbl_curr[session_tbl_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                session_tbl_all_curr = session_tbl_all_curr[session_tbl_all_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        align_time_go = session_tbl['goCue_start_time']
        align_time = session_tbl[session_tbl['animal_response']!=2]['reward_outcome_time']
        
        filtered_rate_go_cue, timestamps_go = get_spike_matrix_filter(spike_times, session_tbl['goCue_start_time'], time_window[0], time_window[1], time_constant=time_constant, stepSize=bin_len)
        filtered_response, timestamps_response = get_spike_matrix_filter(spike_times, session_tbl[session_tbl['animal_response']!=2]['reward_outcome_time'], time_window[0], time_window[1], time_constant=time_constant, stepSize=bin_len)

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 5, height_ratios=[3, 1], wspace=0.35, hspace=0.2)
        gs_model = gridspec.GridSpec(3, 5, height_ratios=[1, 1, 1], wspace=0.3, hspace=0.3)
        colors = [[1, 1, 1], "red"]
        custom_cmap_heatmap = LinearSegmentedColormap.from_list("custom_heatmap", colors)
        colors = [[1, 0.8, 0.8], "red"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_map", colors)

        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(filtered_response[lick_lat_sort], extent=[time_window[0], time_window[1], 0, filtered_response.shape[0]], aspect='auto', origin='lower', cmap=custom_cmap_heatmap, vmin=0, vmax=filtered_response.max())
        plt.colorbar(im, label='Firing rate (Hz)', ax=ax)
        ax.set_xlabel('Time from choice (s)')
        numbins = 3

        fig, ax = plot_rate(
                            filtered_response,
                            timestamps_response, 
                            lick_lat,
                            # np.quantile(lick_lat, np.linspace(0, 0.95, numbins+1)),
                            np.linspace(np.min(lick_lat), np.quantile(lick_lat, 0.95), numbins+1),
                            range(numbins),
                            custom_cmap,
                            fig,
                            gs[1, 0],
                        )
        ax.set_xlabel('Time from choice (s)')

        ax = fig.add_subplot(gs[0, 1])

        im = ax.imshow(filtered_rate_go_cue[session_tbl['animal_response']!=2, :][lick_lat_sort], extent=[time_window[0], time_window[1], 0, filtered_rate_go_cue.shape[0]], aspect='auto', origin='lower', cmap=custom_cmap_heatmap, vmin=0, vmax=filtered_rate_go_cue.max())
        plt.colorbar(im, label='Firing rate (Hz)', ax=ax)
        numbins = 3
        fig, ax = plot_rate(
                            filtered_rate_go_cue[session_tbl['animal_response']!=2, :],
                            timestamps_go, 
                            lick_lat,
                            np.linspace(np.min(lick_lat), np.quantile(lick_lat, 0.95), numbins+1),
                            range(numbins),
                            custom_cmap,
                            fig,
                            gs[1, 1],
                        )
        ax.set_xlabel('Time from go cue (s)')

        ax = fig.add_subplot(gs[0, 2])
        outcomes_lick = 100*outcomes + lick_lat
        outcomes_lick_sort = np.argsort(outcomes_lick)
        im = ax.imshow(filtered_response[outcomes_lick_sort], extent=[time_window[0], time_window[1], 0, filtered_response.shape[0]], aspect='auto', origin='lower', cmap=custom_cmap_heatmap, vmin=0, vmax=filtered_response.max())
        plt.colorbar(im, label='Firing rate (Hz)', ax=ax)
        ax.axhline(np.sum(outcomes==0), color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time from choice (s)')

        fig, ax = plot_rate(
                            filtered_response,
                            timestamps_response, 
                            outcomes,
                            np.array([-1, 0.5, 1.5]),
                            ['no rwd', 'rwd'],
                            custom_cmap,
                            fig,
                            gs[1, 2],
                        )
        ax.set_xlabel('Time from choice (s)')

        ax = fig.add_subplot(gs[0, 3])
        im = ax.imshow(filtered_rate_go_cue[session_tbl['animal_response']!=2, :][outcomes_lick_sort], extent=[time_window[0], time_window[1], 0, filtered_response.shape[0]], aspect='auto', origin='lower', cmap=custom_cmap_heatmap, vmin=0, vmax=filtered_rate_go_cue.max())
        plt.colorbar(im, label='Firing rate (Hz)', ax=ax)
        ax.axhline(np.sum(outcomes==0), color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time from go cue (s)')

        fig, ax = plot_rate(
                            filtered_rate_go_cue[session_tbl['animal_response']!=2, :],
                            timestamps_go, 
                            outcomes,
                            np.array([-1, 0.5, 1.5]),
                            ['no rwd', 'rwd'],
                            custom_cmap,
                            fig,
                            gs[1, 3],
                        )

        # regresssion model
        outcome_time = session_tbl_curr[session_tbl_curr['animal_response']!=2]['reward_outcome_time'].values
        rewarded_ind = session_tbl_curr[session_tbl_curr['animal_response']!=2]['rewarded_historyL'].values | session_tbl_curr[session_tbl_curr['animal_response']!=2]['rewarded_historyR'].values
        # rewarded_ind = np.full(len(outcome_time), True, dtype=bool)
        reward_time = outcome_time[rewarded_ind] + 0.2
        if len(reward_time) > 2:
            # acf
            # compute how past activity contribute to future activity, 
            bin_len = 0.05
            lag_length = 2
            bin_num = int(np.ceil(lag_length / bin_len))

            session_start = session_tbl_curr['goCue_start_time'].values[0]-10
            session_end = session_tbl_curr['goCue_start_time'].values[-1]+20

            counts = np.histogram(spike_times_curr, bins=np.arange(session_start, session_end, bin_len))[0]
            starts = np.arange(session_start, session_end, bin_len)[:-1]
            ends = np.arange(session_start, session_end, bin_len)[1:]

            pre_time = 0
            post_time = 2
            # remove periods within session
            counts_bl = counts.copy().astype(float)
            if len(session_tbl_all_curr) > 0:
                for ind, row in session_tbl_all_curr.iterrows():
                    start_time = row['goCue_start_time'] - pre_time
                    end_time = row['goCue_start_time'] + post_time
                    # set counts in this period to np.nan
                    mask = (ends >= start_time) & (starts <= end_time)
                    if np.sum(mask) > 0:
                        counts_bl[mask] = np.nan

            # compute the lagged activity
            lagged_matrix = np.zeros((len(counts_bl), bin_num))
            for i in range(bin_num):
                lagged_matrix[:, i] = np.roll(counts_bl, i + 1)
                lagged_matrix[:i + 1, i] = np.nan  # set the first i+1 elements to np.nan

            lagged_matrix = sm.add_constant(data=lagged_matrix)

            model_bl = sm.OLS(counts_bl, lagged_matrix, missing='drop').fit()
            ci = model_bl.conf_int(alpha=0.05)  # 95% CI
            yerr = np.vstack([
                model_bl.params[1:] - ci[1:, 0],  # lower error
                ci[1:, 1] - model_bl.params[1:]   # upper error
            ])
            
            spikes_df = align.to_events(spike_times_curr, reward_time, (0, win_len), return_df=True)
            spike_counts = spikes_df.groupby('event_index').size()
            spike_counts = np.array([spike_counts[i] if i in spike_counts.index else 0 for i in range(len(reward_time))])
            spike_matrix, timestamps = get_spike_matrix(spike_times_curr, reward_time, -lag_length, win_len+bin_len, bin_len, bin_len)
            predicted_counts = np.full((spike_matrix.shape[0], spike_matrix.shape[1]-bin_num+1), np.nan)
            predicted_times = timestamps[bin_num-1:]  # Adjusted to match the predicted counts
            pre_cue_counts = np.sum(predicted_times < 0)  # Count how many predicted times are before the cue
            for i in range(spike_matrix.shape[1]-bin_num+1):
                if i <= pre_cue_counts:
                    X = sm.add_constant(spike_matrix[:, i:i+bin_num].copy())
                    predicted_counts[:, i] = model_bl.predict(X)
                else:
                    mix_spikes = np.concatenate((spike_matrix[:, i:np.sum(timestamps<0)].copy(), predicted_counts[:, :i-1]), axis=1)
                    X = sm.add_constant(mix_spikes)
                    predicted_counts[:, i] = model_bl.predict(X)
            predicted_inds = (predicted_times>=0.5* bin_len) & (predicted_times<=win_len-0.5* bin_len)
            predicted_sum_win = predicted_counts[:, predicted_inds].sum(axis=1)
            spike_counts_residual = spike_counts - predicted_sum_win

            # compare 2 models
            # fit regression, use Qchosen in session_df_curr to predict residuals vs spike counts
            if 'Qchosen' in session_tbl_curr.columns.to_list():
                X = session_tbl_curr[['Qchosen']].values[session_tbl_curr['animal_response']!=2].reshape(-1, 1)  # reshape for single feature
                X = X[rewarded_ind]
                X = sm.add_constant(X)  # add intercept term
                model_res = sm.OLS(spike_counts_residual, X).fit()  # fit model to residuals
                ci_res = model_res.conf_int(alpha=0.05)  # 90% CI for residuals model
                model_whole = sm.OLS(spike_counts, X).fit()  # fit model to spike counts
                ci_whole = model_whole.conf_int(alpha=0.05)  # 90% CI for whole model

            # plot model results
            ax = fig.add_subplot(gs_model[0, 4])
            ci = model_bl.conf_int(alpha=0.05)  # 95% CI
            yerr = np.vstack([
                model_bl.params[1:] - ci[1:, 0],  # lower error
                ci[1:, 1] - model_bl.params[1:]   # upper error
            ])
            ax.errorbar(bin_len*np.arange(1, bin_num+1), model_bl.params[1:], yerr=yerr, fmt='-o', label='Model Coefficients with 90% CI',
                        color='blue', capsize=5)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Lag (s)')
            if 'Qchosen' in session_tbl_curr.columns.to_list():
                ax = fig.add_subplot(gs_model[1, 4])
                yerr = np.vstack([
                    model_res.params[1:] - ci_res[1:, 0],  # lower error
                    ci_res[1:, 1] - model_res.params[1:]   # upper error
                ])
            
                ax.errorbar(range(1, 2), model_res.params[1:], yerr=yerr, fmt='o', label='Model Coefficients with 90% CI',
                            color='blue', capsize=5)
                yerr_whole = np.vstack([
                    model_whole.params[1:] - ci_whole[1:, 0],  # lower error
                    ci_whole[1:, 1] - model_whole.params[1:]   # upper error
                ])
                ax.errorbar(range(1), model_whole.params[1:], yerr=yerr_whole, fmt='o', label='Whole Model Coefficients with 90% CI',
                            color='red', capsize=5)
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_xlabel("Whole Outcome Qchosen ---- Res Outcome Qchosen")
                ax.set_ylabel("Coefficient value")
                ax.set_title("Linear Regression Coefficients with 95% CI")

                ax = fig.add_subplot(gs_model[2, 4])
                t_res = model_res.tvalues[1:]       
                t_whole = model_whole.tvalues[1:]

                # Plot bars for each model
                ax.bar(range(1, 2), t_res, 0.3, label='Res Model', color='blue')
                ax.bar(range(1), t_whole, 0.3, label='Whole Model', color='red')

                # Reference line at 0
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

                # Labels
                ax.set_xlabel("Whole Outcome Qchosen ---- Res Outcome Qchosen")
                ax.set_ylabel("t-statistic")
                ax.set_title("Linear Regression t-Statistics")
                ax.legend()

        fig.tight_layout()
        fig.suptitle(f'Session: {session}, Unit: {unit_id}', fontsize=16)
        target_folder = os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], 'go_cue_vs_response')
        os.makedirs(target_folder, exist_ok=True)
        fig.savefig(os.path.join(target_folder, f'{unit_id}_alignments.pdf'))

    # if len(unit_ids) > 0:
    #     combine_pdf_big(target_folder, os.path.join(session_dir[f'ephys_fig_dir_{data_type}'], f'alignments_compare_combined.pdf'))
    

if __name__ == '__main__': 

    df = pd.read_csv('/root/capsule/code/data_management/session_assets_DRN_260622.csv')
    session_ids = df['session_id'].values
    session_ids = [session_id for session_id in session_ids if isinstance(session_id, str)]  # filter only behavior sessions
    model_name = None
    data_type = 'curated'
    curate_time = True
    align_name = 'response'
    formula = 'spikes ~ 1 + outcome + choice'
    
    def process_session(session):
        print(session)
        session_dir = session_dirs(session)
        if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
            if session_dir['curated_dir_curated'] is not None:
                # if not os.path.exists(os.path.join(session_dirs(session)['ephys_dir_curated'],f'{session}_unit_beh_{align_name}.pdf')):
                plot_unit_beh_session(session, data_type = 'curated', align_name = align_name, curate_time=curate_time, opto_only=True,
                            model_name = model_name, formula=formula,
                            pre_event=-1.5, post_event=3, binSize=0.2, stepSize=0.05,
                            units=None)
                # else:
                #     print(f'Already plotted {session} for curated data')
            else:
                print(f'No curated data for {session}')
        # elif session_dir['curated_dir_raw'] is not None:
        #     if not os.path.exists(os.path.join(session_dirs(session)['ephys_dir_raw'],f'{session}_unit_beh_{align_name}.pdf')):
        #         plot_unit_beh_session(session, data_type = 'raw', align_name = align_name, curate_time=curate_time, 
        #                         model_name = model_name, formula=formula,
        #                         pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
        #                         units=None)
        print('----------------------------------')
        print(f'Finished plotting behavior alignment for session {session}')
    # session = 'behavior_782394_2025-04-24_12-07-34'
    # plot_unit_beh_session(session, data_type = data_type, align_name = align_name, curate_time=curate_time, 
    #                     model_name = model_name, formula=formula,
    #                     pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #                     units=[82])

    # plot_unit_beh_session(session='behavior_808650_2025-09-23_14-49-57', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[5,48,67])
    # plot_unit_beh_session(session='behavior_808650_2025-09-26_11-51-57', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[53])    
    # plot_unit_beh_session(session='behavior_810888_2025-10-31_12-39-56', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[39,56,59])  
    # plot_unit_beh_session(session='behavior_814515_2025-10-23_14-41-04', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[87,95,119]) 
    # plot_unit_beh_session(session='behavior_826159_2026-01-22_12-58-47', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[51,145])                 
    # plot_unit_beh_session(session='behavior_826159_2026-01-23_13-34-11', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[42,67,69,83,87,115,171,184]) 
    # plot_unit_beh_session(session='behavior_826159_2026-01-26_15-15-12', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[5,18,20,22,23,25,26,27,29,30,35,39,41,45,54,60,61,62,63,64,129,150,152]) 
    # plot_unit_beh_session(session='behavior_826159_2026-01-27_08-55-48', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[66,84,142])
    # plot_unit_beh_session(session='behavior_826164_2026-01-27_11-14-13', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[15,17,25,59])
    # plot_unit_beh_session(session='behavior_826164_2026-01-28_11-12-55', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[9,10,15,70,87])
    # plot_unit_beh_session(session='behavior_826164_2026-01-29_11-09-03', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[136])
    # plot_unit_beh_session(session='behavior_826164_2026-01-30_11-12-30', data_type = 'raw', align_name = 'response', curate_time=False, 
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[41,77,87,95,98])


    # plot_unit_beh_session(session='behavior_835444_2026-02-17_13-56-45', data_type = 'raw', align_name = 'response', curate_time=False, #EOFError: Ran out of input
    #             model_name = None, formula=formula,
    #             pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
    #             units=[140,158]) 
    plot_unit_beh_session(session='behavior_835444_2026-02-18_13-01-55', data_type = 'raw', align_name = 'response', curate_time=False, 
                model_name = None, formula=formula,
                pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                units=[16,26,47,83,202,239])
    plot_unit_beh_session(session='behavior_835444_2026-02-19_13-08-36', data_type = 'raw', align_name = 'response', curate_time=False, 
                model_name = None, formula=formula,
                pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                units=[25,66,150,164])
    plot_unit_beh_session(session='behavior_835444_2026-02-20_12-12-35', data_type = 'raw', align_name = 'response', curate_time=False, 
                model_name = None, formula=formula,
                pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                units=[3,15,24,32,41,50])
    plot_unit_beh_session(session='behavior_835451_2026-02-25_13-19-37', data_type = 'raw', align_name = 'response', curate_time=False, 
                model_name = None, formula=formula,
                pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                units=[24,63,64,65,67,68,84,90,98,188,191])
    plot_unit_beh_session(session='behavior_835451_2026-02-27_14-19-03', data_type = 'raw', align_name = 'response', curate_time=False, 
                model_name = None, formula=formula,
                pre_event=-1, post_event=3, binSize=0.2, stepSize=0.05,
                units=[51,115,132,133])
    # Parallel(n_jobs=12)(delayed(process_session)(session) for session in session_ids[76:])

