"""
Step 5 of figure preparation pipeline: Generate basic electrophysiology features for all units.

Prerequisites:
    MUST run FIRST:
    1. make_combined_unit_tbl.py (Step 1) - Creates combined_unit_tbl.pkl

    Data requirements:
    - combined_unit_tbl.pkl from Step 1
    - Per-session quality metrics (*_qm.json) in processed_dir/
    - Per-session spike times and unit tables from NWB files

Pipeline Position:
    Script #5 in sequence.txt (line 5)
    Can run IN PARALLEL with:
    - antidromic_generation.py
    - waveform_generation_np.py
    - waveform_generation_tt.py
    - acg_generation.py
    - response_tstats_generation.py
    - outcome_window_generation_parallel.py
    (All these scripts only need combined_unit_tbl.pkl from Step 1)

Purpose:
    Computes fundamental electrophysiological features for all units across sessions:
    - Spike waveform characteristics (peak-to-trough width, amplitude, half-width)
    - Firing statistics (firing rate, ISI distributions, burst properties)
    - Quality control metrics (ISI violations, SNR, amplitude cutoff, presence ratio)
    - Recording stability (spike amplitude drift over time)
    - Anatomical locations (CCF coordinates, recording depth)

Input:
    - combined_unit_tbl.pkl from Step 1
    - Per-session NWB files with spike times
    - Quality metrics JSON files

Output:
    - combined_basic_ephys_tbl.pkl: DataFrame with basic electrophysiology features per unit
    - Includes: waveform metrics, firing statistics, QC measures, anatomical coordinates

Usage:
    Run after Step 1 completes. Can run in parallel with other scripts that only need Step 1.
"""
# %%
import os
import sys

# Resolve code/beh_ephys_analysis (the folder containing `utils`) relative to this
# file's location, so imports work no matter where the repo is checked out.
_beh_ephys_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _beh_ephys_root not in sys.path:
    sys.path.insert(0, _beh_ephys_root)
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from utils.beh_functions import (
    get_session_tbl,
    get_unit_tbl,
    makeSessionDF,
    session_dirs,
)
from utils.capsule_migration import CAPSULE_ROOT, capsule_directories
from utils.combine_tools import apply_qc
from utils.ephys_functions import (
    autocorrelation,
    correlate_nan,
    get_spike_matrix,
    load_drift,
)

warnings.filterwarnings('ignore')
from aind_ephys_utils import align
from joblib import Parallel, delayed
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA

capsule_dirs = capsule_directories()

# %% [markdown]
# # Load criteria and data

# %%
criteria_name = 'basic_ephys_all_LC'

# %%
# load constraints and data
with open(os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
    
with open(os.path.join(CAPSULE_ROOT + '/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json'), 'r') as f:
    constraints = json.load(f)
target_folder = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'basic_ephys')
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
# start with a mask of all True
mask = pd.Series(True, index=combined_tagged_units.index)

# %%
combined_tagged_units_filtered, combined_tagged_units, fig, axes = apply_qc(combined_tagged_units, constraints)
# change to analyze all neurons in LC range
combined_tagged_units_filtered = combined_tagged_units[combined_tagged_units['selected_qc_only']].copy()

def process(session, unit_id, in_df):
    if session.startswith('behavior_ZS'):
        in_df = True
    session_dir = session_dirs(session)
    unit_tbl = get_unit_tbl(session, data_type)
    qm_dict_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_dict_file, 'r') as f:
        qm_dict = json.load(f)
    rec_start = qm_dict['ephys_cut'][0]
    rec_end = qm_dict['ephys_cut'][1]
    opto_file = os.path.join(session_dir['opto_dir_curated'], 
                                f'{session}_opto_session.csv')
    if os.path.exists(opto_file):
        opto_tbl = pd.read_csv(opto_file)
        opto_times = opto_tbl['time'].values
    else:
        opto_times = np.array([])
    if in_df:
        session_df = makeSessionDF(session)
        session_df_go_cue = get_session_tbl(session)
        in_df = True
    else:
        # if session_df exists, delete it to free memory
        if 'session_df' in locals():
            del session_df
            del session_df_go_cue
        in_df = False

    unit_drift = load_drift(session, unit_id, data_type=data_type)
    spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
    spike_times_curr = spike_times.copy()
    opto_times_curr = opto_times.copy()
    rec_start_curr = rec_start
    rec_end_curr = rec_end
    if unit_drift is not None:
        if unit_drift['ephys_cut'][0] is not None:
            spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
            rec_start_curr = unit_drift['ephys_cut'][0]
            opto_times_curr = opto_times_curr[opto_times_curr >= unit_drift['ephys_cut'][0]]
        if unit_drift['ephys_cut'][1] is not None:
            spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
            rec_end_curr = unit_drift['ephys_cut'][1]
            opto_times_curr = opto_times_curr[opto_times_curr <= unit_drift['ephys_cut'][1]]

    if in_df:
        session_df_curr = session_df.copy()
        session_df_go_cue_curr = session_df_go_cue.copy()
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                session_df_go_cue_curr = session_df_go_cue_curr[session_df_go_cue_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                session_df_go_cue_curr = session_df_go_cue_curr[session_df_go_cue_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) >2:
            align_time_cue = session_df_go_cue_curr['goCue_start_time'].values
            align_time_cue_sham = np.random.uniform(np.min(align_time_cue), np.max(align_time_cue), size=max(len(align_time_cue), 20))
            align_time_response = session_df_curr['choice_time'].values
            # baseline
            baseline_df = align.to_events(spike_times_curr, align_time_cue, [-bl_len, -0.01], return_df=True)
            fr_bl = baseline_df.groupby('event_index').size()
            fr_bl = [fr_bl.get(i, 0) for i in range(len(session_df_go_cue_curr))]
            fr_bl = np.array(fr_bl, dtype=float)/bl_len
            fr_bl_mean = np.mean(fr_bl)

            # short baseline
            baseline_df_short = align.to_events(spike_times_curr, align_time_cue, [-bl_len_short, -0.01], return_df=True)
            fr_bl_short = baseline_df_short.groupby('event_index').size()
            fr_bl_short = [fr_bl_short.get(i, 0) for i in range(len(session_df_go_cue_curr))]
            fr_bl_short = np.array(fr_bl_short, dtype=float)/bl_len_short
            

            baseline_df_sham = align.to_events(spike_times_curr, align_time_cue_sham, [-bl_len, -0.01], return_df=True)
            fr_bl_sham = baseline_df_sham.groupby('event_index').size()
            fr_bl_sham = [fr_bl_sham.get(i, 0) for i in range(len(session_df_go_cue_curr))]
            fr_bl_sham = np.array(fr_bl_sham, dtype=float)/bl_len
            fr_bl_sham_mean = np.mean(fr_bl_sham)
            # mean firing rate in baseline

            # response to go cue
            spike_matrix_response, slide_times = get_spike_matrix(spike_times_curr, align_time_cue, 
                                            pre_event=-0.5, post_event=1, 
                                            binSize=binSize, stepSize=0.1)
            spike_response_mean = np.mean(spike_matrix_response, axis=0)
            response_max_ind = np.argmax(spike_response_mean)
        
            response = align.to_events(spike_times_curr, align_time_cue, [0.01, post_cue_len], return_df=True)
            response_sham = align.to_events(spike_times_curr, align_time_cue_sham, [0.01, post_cue_len], return_df=True)
            response_first= align.to_events(spike_times_curr, align_time_cue, [0, 1], return_df=True)
            response_first = response_first.groupby('event_index').min()
            response_first_lat = response_first.reindex(range(len(session_df_go_cue_curr)), fill_value=np.nan)['time'].values
            response_first_lat = np.array(response_first_lat, dtype=float)
            
            # for each trial find the first inter spike interval
            # and the mean of the first 2 intervals
            response_first_isi = []
            response_2_isi = []
            for ind_trial in range(len(response_first_lat)):
                curr_spike = response_first_lat[ind_trial] + align_time_cue[ind_trial]
                next_spike = spike_times_curr[spike_times_curr > curr_spike]
                if len(next_spike) > 0:
                    next_spike = next_spike[0]
                    isi_curr = next_spike - curr_spike
                else:
                    isi_curr = np.nan
                response_first_isi.append(isi_curr)

                next_2_spikes = spike_times_curr[(spike_times_curr >= curr_spike)]
                if len(next_2_spikes) > 2:
                    next_2_spikes = next_2_spikes[:3]
                    isi_curr_2 = np.mean(np.diff(next_2_spikes[:2]))
                else:
                    isi_curr_2 = np.nan
                response_2_isi.append(isi_curr_2)
            response_first_isi = np.array(response_first_isi, dtype=float)
            response_2_isi = np.array(response_2_isi, dtype=float)
            

            response = response.groupby('event_index').size()
            response = [response.get(i, 0) for i in range(len(session_df_go_cue_curr))]
            response = np.array(response, dtype=float)/post_cue_len

            response_sham = response_sham.groupby('event_index').size()
            response_sham = [response_sham.get(i, 0) for i in range(len(session_df_go_cue_curr))]
            response_sham = np.array(response_sham, dtype=float)/post_cue_len

            zero_mask = (fr_bl == 0)
            response_rate_all = np.full_like(response, np.nan)
            response_rate_all[~zero_mask] = (response[~zero_mask] - fr_bl[~zero_mask]) / fr_bl[~zero_mask]
            response_increase = response-fr_bl
            response_increase = np.nanmean(response_increase)

            mask = np.isnan(response) | np.isnan(fr_bl)
            curr_corr = np.corrcoef(response[~mask], fr_bl[~mask])[0, 1]
            # pair ttest between response and baseline
            t_stat_bl_response, p_value_bl_response = ttest_rel(response[~mask], fr_bl[~mask])
            mask_sham = np.isnan(response_sham) | np.isnan(fr_bl_sham)
            curr_corr_sham = np.corrcoef(response_sham[~mask_sham], fr_bl_sham[~mask_sham])[0, 1]
            t_stat_bl_response_sham, p_value_bl_response_sham = ttest_rel(response_sham[~mask_sham], fr_bl_sham[~mask_sham])
            mask_short = np.isnan(response) | np.isnan(fr_bl_short)
            curr_corr_short = np.corrcoef(response[~mask_short], fr_bl_short[~mask_short])[0, 1]
            t_stat_bl_response_short, p_value_bl_response_short = ttest_rel(response[~mask_short], fr_bl_short[~mask_short])
            response_rate = (np.mean(response) - np.mean(fr_bl))/np.mean(fr_bl)  # version 2
            response_fr = np.mean(response)
            # if both sides more than 2
            if np.sum(session_df_go_cue_curr['animal_response'].values == 1) > 2 and \
            np.sum(session_df_go_cue_curr['animal_response'].values == 0) > 2:
                # response bias in right vs left
                response_bias = np.nanmean(response_rate_all[session_df_go_cue_curr['animal_response'].values == 1]) - \
                                np.nanmean(response_rate_all[session_df_go_cue_curr['animal_response'].values == 0])
            else:
                response_bias = np.nan
            # if both condition more than 2
            if np.sum(session_df_go_cue_curr['animal_response'].values != 2) > 2 and \
            np.sum(session_df_go_cue_curr['animal_response'].values == 2) > 2:
                # response bias in go vs no-go trials
                go_inds = session_df_go_cue_curr['animal_response'].values != 2
                no_go_inds = session_df_go_cue_curr['animal_response'].values == 2
                response_diff = (np.mean(response[go_inds]) - np.mean(fr_bl[go_inds]))/np.mean(fr_bl[go_inds]) - \
                                (np.mean(response[no_go_inds]) - np.mean(fr_bl[no_go_inds]))/np.mean(fr_bl[no_go_inds]) # version 2
                response_go = np.nanmean(response_rate_all[session_df_go_cue_curr['animal_response'].values != 2])
                response_no_go = np.nanmean(response_rate_all[session_df_go_cue_curr['animal_response'].values == 2])
                # bl_bias in go vs no-go
                bl_diff = (np.nanmean(fr_bl[session_df_go_cue_curr['animal_response'].values != 2]) - \
                                np.nanmean(fr_bl[session_df_go_cue_curr['animal_response'].values == 2]))/np.mean(fr_bl)
                bl_go = np.nanmean(fr_bl[session_df_go_cue_curr['animal_response'].values != 2])
                bl_no_go = np.nanmean(fr_bl[session_df_go_cue_curr['animal_response'].values == 2])

            else:
                response_diff = np.nan
                bl_diff = np.nan
                response_no_go = np.nan
                response_go = np.nan
                bl_go = np.nan
                bl_no_go = np.nan
                response_first_lat = np.nan
                response_2_isi = np.nan
            # if both reward and no reward trials more than 2
            if np.sum(session_df_curr['outcome'].values == 1) > 2 and \
            np.sum(session_df_curr['outcome'].values == 0) > 2:
                # reward vs no-reward
                reward_delay = np.mean(session_df_go_cue_curr['reward_delay'].values)

                spike_matrix_reward, slide_times = get_spike_matrix(spike_times_curr, align_time_response+reward_delay, 
                                                            pre_event=-0.3, post_event=post_event, 
                                                            binSize=1, stepSize=0.25)
                spike_reward_mean = np.mean(spike_matrix_reward[session_df_curr['outcome']==1], axis=0)
                spike_noreward_mean = np.mean(spike_matrix_reward[session_df_curr['outcome']==0], axis=0)
                max_win_ind = np.argmax(np.abs(spike_reward_mean - spike_noreward_mean))
                spike_reward_peak_mean = np.mean(spike_matrix_reward[session_df_curr['outcome']==1][:, max_win_ind])
                spike_noreward_peak_mean = np.mean(spike_matrix_reward[session_df_curr['outcome']==0][:, max_win_ind])
                outcome_diff = (spike_reward_peak_mean - spike_noreward_peak_mean) / np.mean(fr_bl)
                outcome_diff_abs = np.abs(outcome_diff)
            else:
                outcome_diff = np.nan
                outcome_diff_abs = np.nan
        else: # too short session
            if len(opto_times_curr) > 1:
                baseline_df = align.to_events(spike_times_curr, opto_times_curr, [-bl_len, -0.01], return_df=True)
                fr_bl = baseline_df.groupby('event_index').size()
                fr_bl = [fr_bl.get(i, 0) for i in range(len(opto_times_curr))]
                fr_bl = np.array(fr_bl, dtype=float)
                fr_bl_mean = np.mean(fr_bl)/bl_len
            else: # if too few trials
                fr_bl_mean = np.shape(spike_times_curr)[0] / (rec_end_curr - rec_start_curr)

            response_rate = np.nan
            response_fr = np.nan
            response_increase = np.nan
            response_bias = np.nan
            response_diff = np.nan
            response_first_lat = np.nan
            response_2_isi = np.nan
            bl_diff = np.nan
            outcome_diff = np.nan
            outcome_diff_abs = np.nan
            curr_corr = np.nan
            t_stat_bl_response = np.nan
            p_value_bl_response = np.nan
            curr_corr_sham = np.nan
            t_stat_bl_response_sham = np.nan
            p_value_bl_response_sham = np.nan
            curr_corr_short = np.nan
            t_stat_bl_response_short = np.nan
            p_value_bl_response_short = np.nan
            response_go = np.nan
            response_no_go = np.nan
            bl_go = np.nan
            bl_no_go = np.nan
    else:
        # if not in_df, we cannot compute the response rate or bias
        # remove opto stimulation times
        if len(opto_times_curr) > 1:
            baseline_df = align.to_events(spike_times_curr, opto_times_curr, [-bl_len, -0.01], return_df=True)
            fr_bl = baseline_df.groupby('event_index').size()
            fr_bl = [fr_bl.get(i, 0) for i in range(len(opto_times_curr))]
            fr_bl = np.array(fr_bl, dtype=float)
            fr_bl_mean = np.mean(fr_bl)/bl_len
        else:
            fr_bl_mean = np.shape(spike_times_curr)[0] / (rec_end_curr - rec_start_curr)
        
        response_rate = np.nan
        response_fr = np.nan
        response_increase = np.nan
        response_bias = np.nan
        response_diff = np.nan
        response_first_lat = np.nan
        response_2_isi = np.nan
        bl_diff = np.nan
        outcome_diff = np.nan
        outcome_diff_abs = np.nan
        curr_corr = np.nan
        t_stat_bl_response = np.nan
        p_value_bl_response = np.nan
        curr_corr_sham = np.nan
        t_stat_bl_response_sham = np.nan
        p_value_bl_response_sham = np.nan
        curr_corr_short = np.nan
        t_stat_bl_response_short = np.nan
        p_value_bl_response_short = np.nan
        response_go = np.nan
        response_no_go = np.nan
        bl_go = np.nan
        bl_no_go = np.nan
    return {'session': session,
            'unit_id': unit_id,
            'bl_mean': fr_bl_mean,
            'response_rate': response_rate,
            'response_fr': response_fr,
            'response_bias': response_bias,
            'response_diff': response_diff,
            'bl_diff': bl_diff,
            'outcome_diff': outcome_diff,
            'outcome_diff_abs': outcome_diff_abs,
            'bl_response_corr': curr_corr,
            'bl_response_corr_sham': curr_corr_sham,
            'bl_response_corr_short': curr_corr_short,
            't_stat_bl_response': t_stat_bl_response,
            'p_value_bl_response': p_value_bl_response,
            't_stat_bl_response_sham': t_stat_bl_response_sham,
            'p_value_bl_response_sham': p_value_bl_response_sham,
            't_stat_bl_response_short': t_stat_bl_response_short,
            'p_value_bl_response_short': p_value_bl_response_short,
            'go_mean': bl_go,
            'no_go_mean': bl_no_go,
            'response_lat': response_first_lat,
            'response_isi': response_2_isi
            }

# %%
data_type = 'curated'
target = 'soma'

pre_event = -1.5
post_event = 3
binSize = 0.5
bl_len = 2
bl_len_short = 0.5
post_cue_len = 0.3

auto_inhi_bin = 0.2
window_length = 2


# %%
all_results = []

# Progress tracking variables
_total_units = 0
_processed_units = 0
_last_reported_pct = -10

def safe_process(row):
    """Wrapper to safely call process() and catch errors."""
    global _processed_units, _last_reported_pct, _total_units

    try:
        result = process(row['session'], row['unit'], row['in_df'])
    except Exception as e:
        # Only print errors, not every unit
        print(f"[Error] session {row['session']}, unit {row['unit']}: {e}", flush=True)
        result = {'session': row['session'],
                'unit_id': row['unit'],
                'bl_mean': np.nan,
                'response_rate': np.nan,
                'response_fr': np.nan,
                'response_bias': np.nan,
                'response_diff': np.nan,
                'bl_diff': np.nan,
                'outcome_diff': np.nan,
                'outcome_diff_abs': np.nan,
                'bl_response_corr': np.nan,
                'bl_response_corr_sham': np.nan,
                'bl_response_corr_short': np.nan,
                't_stat_bl_response': np.nan,
                'p_value_bl_response': np.nan,
                't_stat_bl_response_sham': np.nan,
                'p_value_bl_response_sham': np.nan,
                't_stat_bl_response_short': np.nan,
                'p_value_bl_response_short': np.nan,
                'go_mean': np.nan,
                'no_go_mean': np.nan,
                'response_lat': np.nan,
                'response_isi': np.nan,
                }

    # Report progress at 10% intervals
    _processed_units += 1
    if _total_units > 0:
        current_pct = int((_processed_units / _total_units) * 100)
        if current_pct >= _last_reported_pct + 10:
            _last_reported_pct = (current_pct // 10) * 10
            print(f"  Progress: {_last_reported_pct}% ({_processed_units}/{_total_units} units)", flush=True)

    return result

_total_units = len(combined_tagged_units_filtered)
_processed_units = 0
_last_reported_pct = 0
print(f"Processing {_total_units} units with parallelization...")
results = Parallel(n_jobs=-1)(delayed(safe_process)(row) for ind, row in combined_tagged_units_filtered.iterrows())
print(f"  Progress: 100% ({_total_units}/{_total_units} units) - Complete!", flush=True)

# %%
basic_ephys_df = pd.DataFrame(results)
basic_ephys_df['bl_response_corr_diff'] = basic_ephys_df['bl_response_corr'] - basic_ephys_df['bl_response_corr_sham']

# %%
combined_tagged_units_filtered = combined_tagged_units_filtered.rename(columns={'unit':'unit_id'})

# %%
basic_ephys_df = basic_ephys_df.merge(combined_tagged_units_filtered[['session', 'unit_id', 'probe']], on=['session', 'unit_id'], how='left')

# %% [markdown]
# ## ACF

def auto_corr_train(spike_times, auto_inhi_bin, window_length, rec_start, rec_end):
    """
    Calculate autocorrelation of spike times.
    
    Parameters:
    spike_times : array-like
        Spike times of the unit.
    auto_inhi_bin : float
        Bin size for autocorrelation.
    window_length : float
        Length of the window for autocorrelation.
    rec_start : float
        Start time of the recording.
    rec_end : float
        End time of the recording.
        
    Returns:
    acf : array-like
        Autocorrelation function values.
    """
    counts = np.histogram(spike_times, bins=np.arange(rec_start, rec_end, auto_inhi_bin))[0]
    lag=int(window_length/auto_inhi_bin)
    n = len(counts)
    counts = counts - np.nanmean(counts)
    result = correlate_nan(counts, counts, lag = lag)  # only valid correlations
    return result/result[0]  # normalize

def process_acf(session, unit_id, in_df):
    if session.startswith('behavior_ZS'):
        in_df = True
    session_dir = session_dirs(session)
    unit_tbl = get_unit_tbl(session, data_type)
    qm_dict_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_dict_file, 'r') as f:
        qm_dict = json.load(f)
    rec_start = qm_dict['ephys_cut'][0]
    rec_end = qm_dict['ephys_cut'][1]
    opto_file = os.path.join(session_dir['opto_dir_curated'], 
                                f'{session}_opto_session.csv')
    if os.path.exists(opto_file):
        opto_tbl = pd.read_csv(opto_file)
        opto_times = opto_tbl['time'].values
    else:
        opto_times = np.array([])

    if in_df:
        session_df = makeSessionDF(session)
        session_df_go_cue = get_session_tbl(session)
        in_df = True
    else:
        # if session_df exists, delete it to free memory
        if 'session_df' in locals():
            del session_df
            del session_df_go_cue
            del session_df_curr
        in_df = False

    unit_drift = load_drift(session, unit_id, data_type=data_type)
    qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_file) as f:
        qm_dict = json.load(f)
    start_time = qm_dict['ephys_cut'][0]
    end_time = qm_dict['ephys_cut'][1]

    spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
    spike_times_curr = spike_times.copy()
    opto_times_curr = opto_times.copy()
    rec_start_curr = rec_start
    rec_end_curr = rec_end
    if unit_drift is not None:
        if unit_drift['ephys_cut'][0] is not None:
            spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
            rec_start_curr = unit_drift['ephys_cut'][0]
            opto_times_curr = opto_times_curr[opto_times_curr >= unit_drift['ephys_cut'][0]]
            start_time = unit_drift['ephys_cut'][0]
        if unit_drift['ephys_cut'][1] is not None:
            spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
            rec_end_curr = unit_drift['ephys_cut'][1]
            opto_times_curr = opto_times_curr[opto_times_curr <= unit_drift['ephys_cut'][1]]
            end_time = unit_drift['ephys_cut'][1]

    if unit_drift is not None:
        r2 = unit_drift['r_squared_slow_corrected']
        sd = unit_drift['sd/mean_updated']
        r2_ori = unit_drift['r_squared_slow']
        sd_ori = unit_drift['sd/mean']
    else:
        temp_bins = np.arange(start_time, end_time, bin_short)
        spike_counts_slow = np.full(len(temp_bins)-1, np.nan)
        for i in range(len(temp_bins)-1):
            bin_mask = (spike_times_curr >= temp_bins[i]-0.5*bin_long) & (spike_times_curr < temp_bins[i+1] + 0.5*bin_long)
            spike_counts_slow[i] = np.sum(bin_mask)/bin_long
        sd = np.std(spike_counts_slow[np.where(~np.isnan(spike_counts_slow))[0]])/np.nanmean(spike_counts_slow)
        r2 = 0
        r2_ori = 0
        sd_ori = sd

    if in_df:
        session_df_curr = session_df.copy()
        session_df_go_cue_curr = session_df_go_cue.copy()
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                session_df_go_cue_curr = session_df_go_cue_curr[session_df_go_cue_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                session_df_go_cue_curr = session_df_go_cue_curr[session_df_go_cue_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        # calculate auto-inhibition
        if len(session_df_go_cue_curr) > 5:
            session_start = session_df_go_cue_curr['goCue_start_time'].values[0]-10
            session_end = session_df_go_cue_curr['goCue_start_time'].values[-1]+20
            counts = np.histogram(spike_times_curr, bins=np.arange(session_start, session_end, auto_inhi_bin))[0]
            starts = np.arange(session_start, session_end, auto_inhi_bin)[:-1]
            ends = np.arange(session_start, session_end, auto_inhi_bin)[1:]
            sess_len = session_end - session_start + 30
            
            # remove periods within session
            counts_bl = counts.copy().astype(float)
            if len(session_df_go_cue_curr) > 0:
                for ind, row in session_df_go_cue_curr.iterrows():
                    start_time = row['goCue_start_time'] - pre_time
                    end_time = row['goCue_start_time'] + post_time
                    # set counts in this period to np.nan
                    mask = (ends >= start_time) & (starts <= end_time)
                    if np.sum(mask) > 0:
                        counts_bl[mask] = np.nan
        else:
            # if behavior is too short, use longest period without opto stimulation
            all_intervals = np.concatenate(([rec_start_curr], opto_times_curr, [rec_end_curr]))
            longest_interval = np.argmax(np.diff(all_intervals))
            start_interval = all_intervals[longest_interval]
            end_interval = all_intervals[longest_interval + 1]
            sess_len = end_interval - start_interval
            if end_interval - start_interval < window_length * 5:
                return {'session': session,
                        'unit_id': unit_id,
                        'acg': np.full(int(window_length/auto_inhi_bin)+1, np.nan),
                        'acg_bl': np.full(int(window_length/auto_inhi_bin)+1, np.nan),
                        'r2': r2,
                        'r2_ori': r2_ori,
                        'sd': sd,
                        'sd_ori': sd_ori,
                        'len': sess_len,
                        'bl_len': sess_len}
            else:
                counts = np.histogram(spike_times_curr, bins=np.arange(start_interval, end_interval, auto_inhi_bin))[0]
                counts = counts.astype(float)
                starts = np.arange(start_interval, end_interval, auto_inhi_bin)[:-1]
                ends = np.arange(start_interval, end_interval, auto_inhi_bin)[1:]
                counts_bl = counts.copy().astype(float)
                for ind, row in enumerate((opto_times_curr)):
                    start_time = row - pre_time
                    end_time = row + 0.5*post_time
                    # set counts in this period to np.nan
                    mask = (ends >= start_time) & (starts <= end_time)
                    if np.sum(mask) > 0:
                        counts_bl[mask] = np.nan
                        counts[mask] = np.nan
            
    else:
        # if no behavior, use only period before first opto stimulation
        counts = np.histogram(spike_times_curr, bins=np.arange(rec_start_curr, np.min(opto_times_curr), auto_inhi_bin))[0]
        sess_len = np.min(opto_times_curr) - rec_start_curr
        counts = counts.astype(float)
        counts_bl = counts.copy()

    acf_bl = autocorrelation(counts_bl, lag=int(window_length/auto_inhi_bin))
    acf = autocorrelation(counts, lag=int(window_length/auto_inhi_bin))
    bl_len = np.sum(~np.isnan(counts_bl))*auto_inhi_bin
    return {'session': session,
            'unit_id': unit_id,
            'acg': acf,
            'acg_bl': acf_bl,
            'r2': r2,
            'r2_ori': r2_ori,
            'sd': sd,
            'sd_ori': sd_ori,
            'len': sess_len,
            'bl_len': bl_len}

# %%

data_type = 'curated'
target = 'soma'

auto_inhi_bin = 0.03
window_length = 3
pre_time = 0
post_time = 2.5
bin_short = 100
bin_long = 300


# %%
def safe_process_acf(row):
    """Wrapper to safely call process_acf() and catch errors."""
    try:
        return process_acf(row['session'], row['unit_id'], row['in_df'])
    except Exception as e:
        print(f"[Error] session {row['session']}, unit {row['unit_id']}: {e}")
        return {'session': row['session'],
                'acg': np.full((int(window_length/auto_inhi_bin)+1,), np.nan),
                'acg_bl': np.full((int(window_length/auto_inhi_bin)+1,), np.nan),
                'r2': np.nan,
                'sd': np.nan,
                'len': np.nan,
                'bl_len': np.nan}
    

results_acf = Parallel(n_jobs=8)(delayed(safe_process_acf)(row) for ind, row in combined_tagged_units_filtered.iterrows())

# %%
acf_df = pd.DataFrame(results_acf)
all_r2 = acf_df['r2'].values
all_sd = acf_df['sd'].values
all_acf = np.array(acf_df['acg'].tolist())
all_acf_bl = np.array(acf_df['acg_bl'].tolist())

# %% [markdown]
# ### Define filter

# %%
filter_list = ['sd_log', 'r2', 'bl_len', 'acg_1', 'acg_last']
cut_dict = {
    'sd_log': np.log(0.5 + 1e-3),
    'r2': 0.95,
    'bl_len': 1000,
    'acg_1': 0.2,
    'acg_last': 0.05
}
acf_df['acg_1'] = all_acf_bl[:, 1]
acf_df['acg_last'] = all_acf_bl[:, -1]
acf_df['sd_log'] = np.log10(acf_df['sd'] + 1e-3)
acf_df_probe = acf_df.merge(combined_tagged_units_filtered[['session', 'unit_id', 'probe', 'isi_violations']], on=['session', 'unit_id'], how='left')
probes = acf_df_probe['probe'].unique()

# %%
filter = (acf_df_probe['acg_last']<=cut_dict['acg_last']) & (acf_df_probe['sd_log'] < cut_dict['sd_log']) & (acf_df_probe['r2'] < cut_dict['r2']) & (acf_df_probe['bl_len'] > cut_dict['bl_len'])
acf_df_probe['be_filter'] = filter

# %%

# %%

# %%
# PCA on acg
pca = PCA(n_components=5)
end_ind = 25

all_acf = np.array(acf_df['acg'].tolist())
all_acf_bl = np.array(acf_df['acg_bl'].tolist())
all_acf_bl_end = np.mean(all_acf_bl[:, -5:], axis=1, keepdims=True)
all_acf_bl = all_acf_bl - all_acf_bl_end
pca_result = pca.fit_transform(all_acf_bl[acf_df_probe['be_filter'].values, 1:end_ind])
# recontruct the ACF curves by pca scores
pca_reconstructed = pca.inverse_transform(pca_result)

basic_ephys_df = basic_ephys_df.merge(acf_df_probe, on=['session', 'unit_id'], how='left')
pc_intial = np.full((len(basic_ephys_df), 1), np.nan)
for pc_ind in range(3):
    basic_ephys_df[f'pc_{pc_ind+1}'] = pc_intial.copy()
    basic_ephys_df.loc[filter, f'pc_{pc_ind+1}'] = pca_result[:, pc_ind]
# %%
# save basic ephys data
with open(os.path.join(target_folder, f'basic_ephys.pkl'), 'wb') as f:
    pickle.dump(basic_ephys_df, f)
print(f'Saved basic ephys data to {os.path.join(target_folder, f"basic_ephys.pkl")}')



