import numpy as np
import os
import scipy.stats as stats
from collections import defaultdict
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.io import loadmat
from itertools import chain
from scipy.signal import find_peaks
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from utils.beh_functions import session_dirs, get_session_tbl
import matplotlib.gridspec as gridspec

def clean_up_licks(licksL, licksR, crosstalk_thresh=100, rebound_thresh=50, plot=False):
    """
    Clean up lick times by removing elements based on crosstalk and rebound thresholds.
    
    Parameters:
    licksL (list or np.ndarray): Vector of lick times for the left side (in ms).
    licksR (list or np.ndarray): Vector of lick times for the right side (in ms).
    crosstalk_thresh (float): Time threshold (in ms) for detecting crosstalk.
    rebound_thresh (float): Time threshold (in ms) for rebound filtering.
    plot (bool): Whether to plot histograms before and after clean-up.
    
    Returns:
    tuple: (licksL_cleaned, licksR_cleaned), cleaned vectors of lick times for left and right.
    """
    # Sort inputs to ensure time order
    licksL = np.sort(licksL)
    licksR = np.sort(licksR)

    # Crosstalk filtering
    licksL_cleaned = licksL[
        ~np.array([np.any((licksR < x) & ((x - licksR) <= crosstalk_thresh)) for x in licksL])
    ]
    licksR_cleaned = licksR[
        ~np.array([np.any((licksL < x) & ((x - licksL) <= crosstalk_thresh)) for x in licksR])
    ]

    # Rebound filtering
    licksL_cleaned = licksL_cleaned[np.insert(np.diff(licksL_cleaned) > rebound_thresh, 0, True)]
    licksR_cleaned = licksR_cleaned[np.insert(np.diff(licksR_cleaned) > rebound_thresh, 0, True)]

    # Plot results if requested
    if plot:
        bins_same = np.linspace(0, 300, 30)
        bins_diff = np.linspace(0, 300, 30)
        if np.mean(np.diff(np.concatenate([licksL, licksR]))) < 1000:
            bins_same = np.linspace(0, 0.3, 30)
            bins_diff = np.linspace(0, 0.3, 30)
        def plot_histogram(licks, title, ylabel):
            plt.hist(licks, bins=bins_same if "ILI" in title else bins_diff, edgecolor="none")
            plt.title(title)
            if ylabel:
                plt.ylabel(ylabel)

        # Before clean-up
        all_licks = np.concatenate([licksL, licksR])
        all_licks_id = np.concatenate([np.zeros_like(licksL), np.ones_like(licksR)])
        sorted_indices = np.argsort(all_licks)
        all_licks = all_licks[sorted_indices]
        all_licks_id = all_licks_id[sorted_indices]
        all_licks_diff = np.diff(all_licks)
        id_pre = all_licks_id[:-1]
        id_post = all_licks_id[1:]

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(2, 4, 1)
        plot_histogram(all_licks_diff[(id_pre == 0) & (id_post == 0)], 'L_ILI', 'Before clean-up')
        plt.subplot(2, 4, 2)
        plot_histogram(all_licks_diff[(id_pre == 1) & (id_post == 1)], 'R_ILI', None)
        plt.subplot(2, 4, 3)
        plot_histogram(all_licks_diff[(id_pre == 0) & (id_post == 1)], 'L-R_ILI', None)
        plt.subplot(2, 4, 4)
        plot_histogram(all_licks_diff[(id_pre == 1) & (id_post == 0)], 'R-L_ILI', None)

        # After clean-up
        all_licks = np.concatenate([licksL_cleaned, licksR_cleaned])
        all_licks_id = np.concatenate([np.zeros_like(licksL_cleaned), np.ones_like(licksR_cleaned)])
        sorted_indices = np.argsort(all_licks)
        all_licks = all_licks[sorted_indices]
        all_licks_id = all_licks_id[sorted_indices]
        all_licks_diff = np.diff(all_licks)
        id_pre = all_licks_id[:-1]
        id_post = all_licks_id[1:]

        plt.subplot(2, 4, 5)
        plot_histogram(all_licks_diff[(id_pre == 0) & (id_post == 0)], 'L_ILI', 'After clean-up')
        plt.subplot(2, 4, 6)
        plot_histogram(all_licks_diff[(id_pre == 1) & (id_post == 1)], 'R_ILI', None)
        plt.subplot(2, 4, 7)
        plot_histogram(all_licks_diff[(id_pre == 0) & (id_post == 1)], 'L-R_ILI', None)
        plt.subplot(2, 4, 8)
        plot_histogram(all_licks_diff[(id_pre == 1) & (id_post == 0)], 'R-L_ILI', None)

        plt.tight_layout()
        plt.show()
    else:
        fig = None

    return licksL_cleaned, licksR_cleaned, fig

def parse_lick_trains(licks, window_size = 1000, height = 2, min_dist = 2000, inter_train_interval = 500, inter_lick_interval = 800, plot = False, unit = 'seconds', licks_end = None, xlim = None):
    licks = np.array(licks)
    if licks_end is not None:
        licks_end = np.array(licks_end)
        if len(licks) != len(licks_end):
            raise ValueError('Length of licks and licks_end must be the same.')
    else:
        licks_end = licks
    # Check unit of the data, if in s, convert to ms
    if unit == 'seconds':
        licks = np.round(licks * 1000)
        licks_end = np.round(licks_end * 1000)
    if np.mean(np.diff(licks)) < 100 and unit != 'seconds':
        print('Warning: Lick times appear to be in ms. Consider converting to seconds by setting unit="seconds".')
    # Lick peak detection
    bins = np.arange(licks.min(), licks.max(), 1)
    time_binned = bins[:-1]
    licks_binned = np.histogram(licks, bins=bins)[0]
    licks_smoothed = np.convolve(licks_binned, np.ones(window_size)/(window_size/1000), mode='same')
    peaks, lick_peak_amplitudes = find_peaks(licks_smoothed, height = height, distance = min_dist)
    lick_peak_amplitudes = lick_peak_amplitudes['peak_heights']
    lick_peak_times = time_binned[peaks]
    # lick train detection
    inter_lick_interval_mask = np.diff(licks)
    inter_train_mask = inter_lick_interval_mask > inter_train_interval
    within_train_mask = inter_lick_interval_mask < inter_train_interval
    pre_it_mask = np.concatenate([[True], inter_train_mask])
    post_it_mask = np.concatenate([inter_train_mask, [True]])
    pre_wt_mask = np.concatenate([[False], within_train_mask])
    post_wt_mask = np.concatenate([within_train_mask, [False]])
    train_starts_tmp = licks[pre_it_mask]
    train_ends_tmp = licks_end[post_it_mask]
    # if len(train_starts_tmp) > len(train_ends_tmp):
    #     train_starts_tmp = train_starts_tmp[:-1]
    train_starts = []
    train_starts_not = []
    train_ends = []
    train_amps = []
    # for every train_start, find the closest train_end that is larger than train_start
    for train_start in train_starts_tmp:
        if (train_ends_tmp >= train_start).any():
            train_end = train_ends_tmp[train_ends_tmp >= train_start][0]
            if train_end - train_start < 3500:
                train_starts.append(train_start)
                train_ends.append(train_end)
                train_amps.append(np.mean(lick_peak_amplitudes[(lick_peak_times > train_start) & (lick_peak_times < train_end)]))
            else:
                train_starts_not.append(train_start)
    
    if unit == 'seconds':
        train_starts = np.array(train_starts) / 1000
        train_ends = np.array(train_ends) / 1000
        lick_peak_times = lick_peak_times / 1000
        time_binned = time_binned / 1000
        
    
    fig = None
    if plot:
        if xlim is None:
            xlim = [train_starts[0], train_starts[0] + 8*np.nanmean(np.diff(train_starts))]
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 2)
        ax = fig.add_subplot(gs[0, :])
        ax.plot(time_binned[(time_binned>=xlim[0]) & (time_binned<=xlim[1])], licks_smoothed[(time_binned>=xlim[0]) & (time_binned<=xlim[1])], label = 'Lick rate')
        ax.plot(lick_peak_times[(lick_peak_times>=xlim[0]) & (lick_peak_times<=xlim[1])], lick_peak_amplitudes[(lick_peak_times>=xlim[0]) & (lick_peak_times<=xlim[1])], 'ro', label = 'Lick peak')
        ax.plot(time_binned[(time_binned>=xlim[0]) & (time_binned<=xlim[1])], licks_binned[(time_binned>=xlim[0]) & (time_binned<=xlim[1])], label = 'Lick count')
        ax.set_title('Lick rate')
        for start, end in zip(train_starts, train_ends):
            ax.fill_between([start, end], 0, 100, color = 'gray', alpha = 0.5)
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylim([0, 20])

        # inter lick interval histogram
        ax = fig.add_subplot(gs[1,0])
        inter_train_intervals = np.diff(train_starts)
        bin_edges = np.logspace(np.log10(0.1), np.log10(np.max(inter_train_intervals)), 30)
        ax.hist(inter_train_intervals, bins = bin_edges, edgecolor='none')
        # set ax to log scale
        ax.set_xscale('log')
        ax.set_title('Inter-train intervals')
        # train lengths
        ax = fig.add_subplot(gs[1,1])
        train_lengths = train_ends - train_starts
        ax.hist(train_lengths, bins = 50, edgecolor='none')
        ax.set_title('Train lengths')
        plt.tight_layout()
    parsed_licks = {'lick_peak_times': lick_peak_times, 'lick_peak_amplitudes': lick_peak_amplitudes, 'train_starts': train_starts, 'train_ends': train_ends, 'train_amps': np.array(train_amps)}
    
    return parsed_licks, fig

def load_licks(session, plot = False, inter_train_interval = 800):
    session_dir = session_dirs(session)
    if session_dir['aniID'].startswith('ZS'):
        raw_nwb_file = [f for f in os.listdir(session_dir['raw_dir']) if f.endswith('.nwb.zarr')][0]
        raw_nwb = load_nwb_from_filename(os.path.join(session_dir['raw_dir'], raw_nwb_file))
    else:
        raw_nwb = load_nwb_from_filename(session_dir['nwb_beh'])
    session_tbl = get_session_tbl(session)
    raw_df = raw_nwb.intervals['trials'].to_dataframe()
    licks_L = raw_nwb.acquisition['left_lick_time'].timestamps[:]
    licks_R = raw_nwb.acquisition['right_lick_time'].timestamps[:]
    all_licks = np.sort(np.concatenate([licks_L, licks_R]))
    licks_L_cleaned, licks_R_cleaned, fig = clean_up_licks(licks_L, licks_R, crosstalk_thresh=100/1000, rebound_thresh=50/1000, plot=plot)  
    if plot:
        fig.suptitle(f'Session: {session}')
        plt.tight_layout()
        fig.savefig(fname=os.path.join(session_dir['beh_fig_dir'], f'{session}_lick_cleanup.pdf'))
    lick_trains_L, fig_L = parse_lick_trains(licks_L_cleaned, plot=plot, unit='seconds', 
                                                inter_train_interval = inter_train_interval,
                                                xlim = [session_tbl['goCue_start_time'].values[0] - 2, session_tbl['goCue_start_time'].values[0] + 150])
    lick_trains_R, fig_R = parse_lick_trains(licks_R_cleaned, plot=plot, unit='seconds', 
                                                    inter_train_interval = inter_train_interval,
                                                xlim = [session_tbl['goCue_start_time'].values[0] - 2, session_tbl['goCue_start_time'].values[0] + 150])
    if plot:
        fig_L.suptitle(f'Session: {session} - Left Lick Trains')
        plt.tight_layout()
        fig_L.savefig(fname=os.path.join(session_dir['beh_fig_dir'], f'{session}_lick_trains_L.pdf'))
        fig_R.suptitle(f'Session: {session} - Right Lick Trains')
        plt.tight_layout()
        fig_R.savefig(fname=os.path.join(session_dir['beh_fig_dir'], f'{session}_lick_trains_R.pdf'))

    # for all lick trains, test if it occurs within 5 seconds of a goCue
    in_trial_L = np.zeros_like(lick_trains_L['train_starts'], dtype=bool)
    for goCue in raw_df['goCue_start_time'].values:
        in_trial_L |= (lick_trains_L['train_starts'] >= goCue) & (lick_trains_L['train_starts'] <= goCue + 2)
    lick_trains_L['in_trial'] = in_trial_L

    in_trial_R = np.zeros_like(lick_trains_R['train_starts'], dtype=bool)
    for goCue in raw_df['goCue_start_time'].values:
        in_trial_R |= (lick_trains_R['train_starts'] >= goCue) & (lick_trains_R['train_starts'] <= goCue + 2)
    lick_trains_R['in_trial'] = in_trial_R
        
    # combine left and right licks trains by combining each field in the dicts
    lick_trains_all = {}
    for key in lick_trains_L.keys():
        lick_trains_all[key] = np.concatenate([lick_trains_L[key], lick_trains_R[key]])
    # sort by train_starts
    sort_idx = np.argsort(lick_trains_all['train_starts'])
    for key in lick_trains_all.keys():
        if len(lick_trains_all[key]) == len(sort_idx):
            lick_trains_all[key] = lick_trains_all[key][sort_idx]
    lick_trains_all['side'] = np.concatenate([np.array([0]*len(lick_trains_L['train_starts'])), np.array([1]*len(lick_trains_R['train_starts']))])
    return {'licks_L_cleaned': licks_L_cleaned,
            'licks_R_cleaned': licks_R_cleaned,
            'lick_trains_L': lick_trains_L,
            'lick_trains_R': lick_trains_R,
            'lick_trains_all': lick_trains_all}

def load_licks_video(session, plot = False, confidence_thresh = 0.8, inter_train_interval = 800):
    session_dir = session_dirs(session)
    session_tbl = get_session_tbl(session)
    lick_video_file = os.path.join(session_dir['beh_fig_dir'], f'{session}_filtered_video_licks.csv')
    if not os.path.exists(lick_video_file):
        print(f'Lick video file not found for session {session}. Please run the lick video processing pipeline first.')
        return None
    lick_video_df = pd.read_csv(lick_video_file)
    licks_L = lick_video_df[(lick_video_df['gmm_pred'] == 0) & (lick_video_df['gmm_proba1'] >= confidence_thresh)]['start_time_session'].values
    licks_R = lick_video_df[(lick_video_df['gmm_pred'] == 1) & (lick_video_df['gmm_proba1'] >= confidence_thresh)]['start_time_session'].values
    licks_L_end = licks_L + lick_video_df[(lick_video_df['gmm_pred'] == 0) & (lick_video_df['gmm_proba1'] >= confidence_thresh)]['duration'].values
    licks_R_end = licks_R + lick_video_df[(lick_video_df['gmm_pred'] == 1) & (lick_video_df['gmm_proba1'] >= confidence_thresh)]['duration'].values
    all_licks = np.sort(np.concatenate([licks_L, licks_R]))
    licks_L_cleaned, licks_R_cleaned = licks_L, licks_R
    lick_trains_L, fig_L = parse_lick_trains(licks_L_cleaned, plot=plot, unit='seconds', licks_end = licks_L_end, 
                                            inter_train_interval = inter_train_interval,
                                            xlim = [session_tbl['goCue_start_time'].values[0] - 2, session_tbl['goCue_start_time'].values[0] + 150])
    lick_trains_R, fig_R = parse_lick_trains(licks_R_cleaned, plot=plot, unit='seconds', licks_end = licks_R_end, 
                                            inter_train_interval = inter_train_interval,
                                            xlim = [session_tbl['goCue_start_time'].values[0] - 2, session_tbl['goCue_start_time'].values[0] + 150])
    if plot:
        fig_L.suptitle(f'Session: {session} - Left Lick Trains')
        plt.tight_layout()
        fig_L.savefig(fname=os.path.join(session_dir['beh_fig_dir'], f'{session}_video_lick_trains_L.pdf'))
        fig_R.suptitle(f'Session: {session} - Right Lick Trains')
        plt.tight_layout()
        fig_R.savefig(fname=os.path.join(session_dir['beh_fig_dir'], f'{session}_video_lick_trains_R.pdf'))

    # for all lick trains, test if it occurs within 5 seconds of a goCue
    in_trial_L = np.zeros_like(lick_trains_L['train_starts'], dtype=bool)
    for goCue in session_tbl['goCue_start_time'].values:
        in_trial_L |= (lick_trains_L['train_starts'] >= goCue) & (lick_trains_L['train_starts'] <= goCue + 2)
    lick_trains_L['in_trial'] = in_trial_L

    in_trial_R = np.zeros_like(lick_trains_R['train_starts'], dtype=bool)
    for goCue in session_tbl['goCue_start_time'].values:
        in_trial_R |= (lick_trains_R['train_starts'] >= goCue) & (lick_trains_R['train_starts'] <= goCue + 2)
    lick_trains_R['in_trial'] = in_trial_R
        
    # combine left and right licks trains by combining each field in the dicts
    lick_trains_all = {}
    for key in lick_trains_L.keys():
        lick_trains_all[key] = np.concatenate([lick_trains_L[key], lick_trains_R[key]])
    # sort by train_starts
    sort_idx = np.argsort(lick_trains_all['train_starts'])
    for key in lick_trains_all.keys():
        if len(lick_trains_all[key]) == len(sort_idx):
            lick_trains_all[key] = lick_trains_all[key][sort_idx]
    lick_trains_all['side'] = np.concatenate([np.array([0]*len(lick_trains_L['train_starts'])), np.array([1]*len(lick_trains_R['train_starts']))])
    return {'licks_L_cleaned': licks_L_cleaned,
            'licks_R_cleaned': licks_R_cleaned,
            'lick_trains_L': lick_trains_L,
            'lick_trains_R': lick_trains_R,
            'lick_trains_all': lick_trains_all}