"""
Step 1 of figure preparation pipeline: Create combined unit table across all sessions.

Prerequisites:
    NONE - This is the FIRST script in the pipeline.

    Before running, ensure:
    - All per-session preprocessing is complete (session_preprocessing.py)
    - Optogenetic tagging analysis finished for all sessions (opto_tagging.py)
    - Antidromic analysis complete if applicable (antidromic_analysis_session.py)
    - Session metadata up-to-date in session_assets.csv and hopkins_session_assets.csv
    - sessions_to_exclude.txt contains any sessions to skip

Pipeline Position:
    This is script #1 in sequence.txt - Run this FIRST before all other figure preparation scripts.

Purpose:
    Combines per-session unit tables with:
    - Basic unit properties (spike times, waveforms, locations)
    - Optogenetic tagging metrics (p_max, response latency, antidromic tier)
    - Quality control metrics (ISI violations, firing rate, amplitude)
    - Behavioral recording metadata (animal, date, recording side, sex)
    - Cross-correlation and auto-correlation data

Output:
    Saves combined_unit_tbl.pkl to manuscript_fig_prep_dir/combined_unit_tbl/
    This table serves as the foundation for all subsequent figure generation scripts.

Usage:
    Run as standalone script. Processes all sessions in session_assets.csv
    (excluding those in sessions_to_exclude.txt) and combines their unit data.
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

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils.beh_functions import (
    get_session_tbl,
    get_unit_tbl,
    session_dirs,
)
from utils.capsule_migration import CAPSULE_ROOT, capsule_directories
from utils.ephys_functions import *
from utils.opto_utils import load_opto_sig, opto_metrics

# %%
# Make combined session-unit table
capsule_dirs = capsule_directories()
# Read CSVs with probe as string to match working pickle at scratch/combined/
# Working file has dtype:object with values '2', 'tt', 'opto' (all strings)
dfs = [pd.read_csv(CAPSULE_ROOT + '/code/data_management/session_assets.csv', dtype={'probe': str}),
        # pd.read_csv(CAPSULE_ROOT + '/code/data_management/hopkins_session_assets.csv', dtype={'probe': str})
        ]
df = pd.concat(dfs).reset_index(drop=True)
# keep the rows where the session_id exist in /root/capsule/code/data_management/LC_curated.txt
with open(CAPSULE_ROOT + '/code/data_management/LC_curated.txt') as f:
    curated_sessions = set(line.strip() for line in f if line.strip())
df = df[df['session_id'].isin(curated_sessions)]

# remove those are not strings
df = df[df['session_id'].apply(lambda x: isinstance(x, str))]
df = df.reset_index(drop=True)
# %%
def process_session(session, beh, rec_side, probe, sex, target='soma'):
    """
    Extract and compile all unit-level data from a single session.

    Aggregates unit properties, opto-tagging metrics, quality control measures,
    cross-correlograms, and antidromic analysis results into a standardized format
    for inclusion in the combined unit table.

    Parameters:
        session (str): Session identifier.
        beh (str): Behavioral task type.
        rec_side (str): Recording hemisphere ('left' or 'right').
        probe (str): Probe identifier.
        sex (str): Animal sex ('M' or 'F').
        target (str): Optogenetic target region (default: 'soma').

    Returns:
        pd.DataFrame or None: DataFrame with one row per unit containing all compiled metrics.
                             Returns None if session has no valid data or should be skipped.
    """
    session_dir = session_dirs(session)
    bin_short = 100
    bin_long = 300
    # --- skip missing or invalid sessions ---
    if 'ZS' in session:
        if (not os.path.exists(session_dir['nwb_dir_raw'])) or (get_unit_tbl(session, 'curated') is None):
            # Silently skip - summary will be reported
            return None
    if session_dir['curated_dir_curated'] is None:
        return None

    # No per-session print - progress reported at intervals
    data_type = 'curated'
    qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_file) as f:
        qm_dict = json.load(f)

    # --- load data ---
    unit_tbl = get_unit_tbl(session, data_type)
    if unit_tbl is None or len(unit_tbl) == 0:
        return None

    opto_metrics_session = opto_metrics(session, data_type=data_type)
    session_df = get_session_tbl(session)

    session_opto_sig = load_opto_sig(session, data_type=data_type)

    # --- LC range from scratch folder ---
    sd_curr = session_dirs(session, scratch_dir=capsule_dirs['derived_dir_curr'])
    lc_json = os.path.join(sd_curr['opto_dir_curated'], f'{session}_lc_range.json')
    if os.path.exists(lc_json):
        with open(lc_json) as _f:
            _lc = json.load(_f).get('lc_range', {})
        lc_top = _lc.get('a')
        lc_bot = _lc.get('b')
        if lc_top is not None:
            in_lc = unit_tbl['y_loc'] <= lc_top
            if lc_bot is not None:
                in_lc = in_lc & (unit_tbl['y_loc'] >= lc_bot)
        else:
            in_lc = pd.Series([False] * len(unit_tbl))
    else:
        in_lc = pd.Series([False] * len(unit_tbl))
    in_lc_list = in_lc.tolist()

    # --- basic derived columns ---
    if 'p_max' not in unit_tbl.columns:
        p_max = unit_tbl['p_max_x'].tolist()
        p_mean = unit_tbl['p_mean_x'].tolist()
        lat_max_p = unit_tbl['lat_max_p_x'].tolist()
        eu = unit_tbl['euc_max_p_x'].tolist()
        corr = unit_tbl['corr_max_p_x'].tolist()
        peaks = unit_tbl['peak_x'].values
        amp = unit_tbl['amp_x'].values
    else:
        p_max = unit_tbl['p_max'].tolist()
        p_mean = unit_tbl['p_mean'].tolist()
        lat_max_p = unit_tbl['lat_max_p'].tolist()
        eu = unit_tbl['euc_max_p'].tolist()
        corr = unit_tbl['corr_max_p'].tolist()
        peaks = unit_tbl['peak'].values
        amp = unit_tbl['amp'].values

    if 'x_ccf' in unit_tbl.columns:
        x_ccf = unit_tbl['x_ccf'].tolist()
        y_ccf = unit_tbl['y_ccf'].tolist()
        z_ccf = unit_tbl['z_ccf'].tolist()
    else:
        x_ccf = [np.nan]*len(unit_tbl)
        y_ccf = [np.nan]*len(unit_tbl)
        z_ccf = [np.nan]*len(unit_tbl)

    # --- waveform-related ---
    if 'peak_wf_opt' in unit_tbl.columns:
        wf_opt = [wf_opt_unit if isinstance(wf_opt_unit, np.ndarray) else wf_unit
                  for wf_opt_unit, wf_unit in zip(unit_tbl['peak_wf_opt'], unit_tbl['peak_wf'])]
        wf_opt_aligned = [wf_opt_unit if isinstance(wf_opt_unit, np.ndarray) else wf_unit
                          for wf_opt_unit, wf_unit in zip(unit_tbl['peak_wf_opt_aligned'], unit_tbl['peak_wf_aligned'])]
        wf_opt_2d = [wf_opt_unit if isinstance(wf_opt_unit, np.ndarray) else wf_unit
                     for wf_opt_unit, wf_unit in zip(unit_tbl['mat_wf_opt'], unit_tbl['wf_2d'])]
    else:
        wf_opt = unit_tbl['peak_wf'].tolist()
        wf_opt_aligned = unit_tbl['peak_wf_aligned'].tolist()
        wf_opt_2d = unit_tbl['wf_2d'].tolist()

    amp_opt = [
        np.max(wf_opt_curr) - np.min(wf_opt_curr) if isinstance(wf_opt_curr, np.ndarray) else curr_amp_unit
        for wf_opt_curr, curr_amp_unit in zip(wf_opt, amp)
    ]
    if 'amplitude_opt' in unit_tbl.columns:
        peak_opt = [
            curr_peak_opt if not np.isnan(curr_peak_opt) else curr_peak
            for curr_peak_opt, curr_peak in zip(unit_tbl['amplitude_opt'].values, peaks)
        ]
    else:
        peak_opt = list(peaks)

    if 'peak_waveform_raw_aligned' in unit_tbl.columns:
        wf_raw = unit_tbl['peak_waveform_raw_fake_aligned'].tolist()
        wf_2d_raw = unit_tbl['mat_wf_raw_fake'].tolist()
        peak_raw = [
            curr_peak_raw - curr_wf[0] if curr_peak_raw is not None and not np.isnan(curr_peak_raw)
            else None
            for curr_peak_raw, curr_wf in zip(unit_tbl['peak_raw_fake'], wf_raw)
        ]
        amp_raw = unit_tbl['amplitude_raw_fake'].tolist()
    else:
        wf_raw = [None]*len(unit_tbl)
        wf_2d_raw = [None]*len(unit_tbl)
        peak_raw = [None]*len(unit_tbl)
        amp_raw = [None]*len(unit_tbl)

    # --- waveform-independent scalar values ---
    isi_v = unit_tbl['isi_violations_ratio'].tolist()
    presenece_ratio = unit_tbl['presence_ratio'].tolist()
    amplitude_cutoff = unit_tbl['amplitude_cutoff'].tolist()
    snr = unit_tbl['snr'].tolist()
    y_loc = unit_tbl['y_loc'].tolist()
    fr = unit_tbl['firing_rate'].tolist()
    decoder = unit_tbl['decoder_label'].tolist()
    tag_loc = unit_tbl['tagged_loc'].tolist() if 'tagged_loc' in unit_tbl.columns else [np.nan]*len(unit_tbl)
    top = unit_tbl['LC_range_top'].tolist() if 'LC_range_top' in unit_tbl.columns else [np.nan]*len(unit_tbl)
    bottom = unit_tbl['LC_range_bottom'].tolist() if 'LC_range_bottom' in unit_tbl.columns else [np.nan]*len(unit_tbl)

    # --- opto per-unit results ---
    resp_p_all_conditions, resp_lat_all_conditions = [], []
    mean_p_all_conditions, eu_all_conditions = [], []
    corr_all_conditions, sig_counts_all_conditions = [], []
    all_sig_counts = []
    trial_count = []
    sd_all = []
    len_all = []

    for unit_id in unit_tbl['unit_id'].values:
        start_time = qm_dict['ephys_cut'][0]
        end_time = qm_dict['ephys_cut'][1]
        spike_times = unit_tbl[unit_tbl['unit_id'] == unit_id]['spike_times'].values[0]
        unit_opto = opto_metrics_session.load_unit(unit_id)
        unit_opto_sig = session_opto_sig.load_unit(unit_id) if session_opto_sig is not None else None
        unit_drift = load_drift(session, unit_id, data_type=data_type)
        if unit_opto is None:
            continue

        curr_p_resp_all = unit_opto['resp_p_bl'].values
        curr_lat_resp_all = unit_opto['resp_lat'].values
        curr_p_mean_all = unit_opto['mean_p'].values
        curr_eu_all = unit_opto['euclidean_norm'].values
        curr_corr_all = unit_opto['correlation'].values

        unit_opto['sig_num'] = np.full(len(unit_opto), np.nan)
        if unit_opto_sig is not None:
            if not session_dir['aniID'].startswith('ZS'):
                for cond_ind, row in unit_opto.iterrows():
                    filt = (unit_opto_sig['power'] == row['powers']) & (unit_opto_sig['site'] == row['sites'])
                    if len(unit_opto_sig['pre_post'].unique()) > 1:
                        filt &= (unit_opto_sig['pre_post'] == row['stim_times'])
                    curr_sig_rows = unit_opto_sig[filt]
                    if len(curr_sig_rows) >= 1:
                        unit_opto.loc[cond_ind, 'sig_num'] = curr_sig_rows['p_sig_count'].values[0]
            else:
                unit_opto['sig_num'] = unit_opto_sig['p_sig_count'].values[0]

        curr_sig_num_all = unit_opto['sig_num'].values
        curr_max_count = np.nan if unit_opto_sig is None else unit_opto_sig['p_sig_count'].max()
        # cut off unstable
        spike_times_curr = spike_times
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                start_time = max(start_time, unit_drift['ephys_cut'][0])
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                end_time = min(end_time, unit_drift['ephys_cut'][1])
        # trial length
        if session_df is not None:
            go_cue_times = session_df['goCue_start_time']
            if unit_drift is not None:
                if unit_drift['ephys_cut'][0] is not None:
                    go_cue_times = go_cue_times[go_cue_times >= unit_drift['ephys_cut'][0]]
                if unit_drift['ephys_cut'][1] is not None:
                    go_cue_times = go_cue_times[go_cue_times <= unit_drift['ephys_cut'][1]]
            curr_trial_count = len(go_cue_times)
        else:
            curr_trial_count = 0
        
        # append drift params if exist
        if unit_drift is not None:
            sd = unit_drift['sd/mean_updated']
        else:
            if end_time - start_time < bin_long*2:
                sd = np.nan
            else:
                temp_bins = np.arange(start_time, end_time, bin_short)
                spike_counts_slow = np.full(len(temp_bins)-1, np.nan)
                for i in range(len(temp_bins)-1):
                    bin_mask = (spike_times_curr >= temp_bins[i]-0.5*bin_long) & (spike_times_curr < temp_bins[i+1] + 0.5*bin_long)
                    spike_counts_slow[i] = np.sum(bin_mask)/bin_long
                
                if np.nanmean(spike_counts_slow) > 0:
                    sd = np.std(spike_counts_slow[np.where(~np.isnan(spike_counts_slow))[0]])/np.nanmean(spike_counts_slow)
                else:
                    sd = np.nan


        len_all.append(end_time-start_time)
        sd_all.append(sd)
        resp_p_all_conditions.append(curr_p_resp_all)
        resp_lat_all_conditions.append(curr_lat_resp_all)
        mean_p_all_conditions.append(curr_p_mean_all)
        eu_all_conditions.append(curr_eu_all)
        corr_all_conditions.append(curr_corr_all)
        sig_counts_all_conditions.append(curr_sig_num_all)
        all_sig_counts.append(curr_max_count)
        trial_count.append(curr_trial_count)

    # --- final dictionary ---
    return {
        'session': session,
        'unit': unit_tbl['unit_id'].tolist(),
        'qc_pass': unit_tbl['default_qc'].tolist(),
        'opto_tagged': unit_tbl['tagged_loc'].tolist(),
        'in_df': beh,
        'trial_count': trial_count,
        'p_max': p_max,
        'p_mean': p_mean,
        'sig_counts': all_sig_counts,
        'lat_max_p': lat_max_p,
        'isi_violations': isi_v,
        'snr': snr,
        'amplitude_cutoff': amplitude_cutoff,
        'presence_ratio': presenece_ratio,        
        'eu': eu,
        'corr': corr,
        'amp': amp_opt,
        'amp_raw': amp_raw,
        'peak': peak_opt,
        'peak_raw': peak_raw,
        'wf': wf_opt,
        'wf_raw': wf_raw,
        'wf_aligned': wf_opt_aligned,
        'wf_2d': wf_opt_2d,
        'wf_2d_raw': wf_2d_raw,
        'probe': probe,
        'sex': sex,
        'y_loc': y_loc,
        'rec_side': rec_side,
        'top': top,
        'bottom': bottom,
        'in_LC': in_lc_list,
        'tag_loc': tag_loc,
        'fr': fr,
        'decoder': decoder,
        'all_p_max': resp_p_all_conditions,
        'all_p_mean': mean_p_all_conditions,
        'all_lat_max_p': resp_lat_all_conditions,
        'all_corr': corr_all_conditions,
        'all_eu': eu_all_conditions,
        'all_sig_counts': sig_counts_all_conditions,
        'x_ccf': x_ccf,
        'y_ccf': y_ccf,
        'z_ccf': z_ccf,
        'sd': sd_all,
        'rec_len': len_all,
    }

# %%
target = 'soma'
def safe_process(session, beh, rec_side, probe, sex):
    try:
        return process_session(session, beh, rec_side, probe, sex, target=target)
    except Exception as e:
        print(f'Error processing {session}: {e}')
        return None

print(f"Processing {len(df)} sessions with parallelization...", flush=True)
results = Parallel(n_jobs=-1)(
    delayed(safe_process)(row['session_id'], row['behavior'], row['side'], row['probe'], row['sex'])
    for _, row in df.iterrows()
)
# %%
# remove all None results
valid_results = [res for res in results if res is not None]
skipped = len(results) - len(valid_results)
print(f"  Completed: {len(valid_results)} sessions processed, {skipped} skipped", flush=True)
results = valid_results

# %%
# sort by the sequence of session_ids in df
session_order = {session: i for i, session in enumerate(df['session_id'].tolist())}
results.sort(key=lambda x: session_order[x['session']])

# %%
results_df = [pd.DataFrame(res) for res in results]

# %%
combined_tagged_units = pd.concat(results_df, ignore_index=True)

# %%
# remove duplicate neuron
row_ind = (combined_tagged_units['session'] == 'behavior_ZS062_2021-03-28_17-40-42') & (combined_tagged_units['unit'] == 'TT8_SS_01')
print(f'Removing {row_ind.sum()} duplicate neurons')
combined_tagged_units = combined_tagged_units[~row_ind].reset_index(drop=True)

# %%
# save dataframe in combined folder
save_folder = os.path.join(capsule_dirs["manuscript_fig_prep_dir"],'combined_unit_tbl')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
with open(os.path.join(save_folder, 'combined_unit_tbl.pkl'), 'wb') as f:
    pickle.dump(combined_tagged_units, f) 

# %%



