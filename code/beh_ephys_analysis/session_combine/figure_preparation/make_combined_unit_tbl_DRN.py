# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
from aind_ephys_rig_qc.temporal_alignment import search_harp_line
import numpy as np
import pandas as pd
from utils.plot_utils import combine_pdf_big

import json
from utils.beh_functions import session_dirs, get_session_tbl, get_unit_tbl
from utils.ephys_functions import load_drift
from utils.opto_utils import opto_metrics, load_opto_sig
from utils.capsule_migration import capsule_directories
import pickle
from joblib import Parallel, delayed

# %%
# Make combined session-unit table
capsule_dirs = capsule_directories()
df = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
df = df.reset_index(drop=True)
session_exclude_file = '/root/capsule/code/data_management/sessions_to_exclude.txt'
with open(session_exclude_file, 'r') as f:
    exclude = [line.strip() for line in f.readlines()]
# session_ids, behs = zip(*[
#     (session, beh)
#     for session, beh in zip(session_ids, behs)
#     if isinstance(session, str) and session not in exclude
# ])
# exclude sessions
df = df[~df['session_id'].isin(exclude)]
# remove those are not strings
df = df[df['session_id'].apply(lambda x: isinstance(x, str))]
df = df.reset_index(drop=True)
# session_ids = list(session_ids)
# behs = list(behs)
# %%
def process_session(session, beh, rec_side, probe, sex, target='soma'):
    session_dir = session_dirs(session)
    bin_short = 100
    bin_long = 300
    # --- skip missing or invalid sessions ---
    # if 'ZS' in session:
    #     if (not os.path.exists(session_dir['nwb_dir_raw'])) or (get_unit_tbl(session, 'curated') is None):
    #         print(f'Skipping {session} due to no neuron data')
    #         return None
    # if session_dir['curated_dir_curated'] is None:
    #     return None

    print(f'Processing {session}')
    data_type = 'raw'
    qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    with open(qm_file) as f:
        qm_dict = json.load(f)

    # --- load data ---
    unit_tbl = get_unit_tbl(session, data_type, ccf_suffix='260817') #Set ccf_suffix for annotator/date suffix
    if unit_tbl is None or len(unit_tbl) == 0:
        return None

    opto_metrics_session = opto_metrics(session, data_type=data_type)
    session_df = get_session_tbl(session)

    # session_opto_sig = load_opto_sig(session, data_type=data_type)
    session_opto_sig = None

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

    if 'peak_waveform_raw_fake_aligned' in unit_tbl.columns:
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
                    print(f'{session}_{unit_id} spike_count_slow weird.\n')
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
        'opto_pass': unit_tbl['opto_pass'].tolist(),
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

# results = []
# for index, row in df.iterrows():
#     result = safe_process(row['session_id'], row['behavior'], row['side'], row['probe'], row['sex'])
#     if result is not None:
#         results.append(result)

if __name__ == "__main__":

    results = Parallel(n_jobs=-1)(
        delayed(safe_process)(
            row["session_id"],
            row["behavior"],
            row["side"],
            row["probe"],
            ##row["sex"],
            None,
        )
        for _, row in df.iterrows()
    )

    results = [res for res in results if res is not None]

    session_order = {
        session: i
        for i, session in enumerate(df["session_id"].tolist())
    }
    results.sort(key=lambda x: session_order[x["session"]])

    results_df = [pd.DataFrame(res) for res in results]
    combined_tagged_units = pd.concat(results_df, ignore_index=True)

    save_folder = os.path.join(
        capsule_dirs["manuscript_fig_prep_dir"],
        "combined_unit_tbl",
    )
    os.makedirs(save_folder, exist_ok=True)

    with open(
        os.path.join(save_folder, "combined_unit_tbl.pkl"),
        "wb",
    ) as f:
        pickle.dump(combined_tagged_units, f)



