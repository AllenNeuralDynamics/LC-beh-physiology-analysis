# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from scipy.io import loadmat
from scipy.stats import zscore
import ast
from pathlib import Path
import glob
import json
import seaborn as sns
from PyPDF2 import PdfMerger
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb
from utils.ephys_functions import correlate_nan, autocorrelation, load_trial_drift, load_drift, get_spike_matrix
from utils.ccf_utils import ccf_pts_convert_to_mm, pir_to_lps
from utils.combine_tools import apply_qc
from utils.capsule_migration import capsule_directories
import pickle
import scipy.stats as stats
import spikeinterface as si
import shutil
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import r2_score
import warnings
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')
from matplotlib import colors as mcolors
from matplotlib import cm
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from sklearn.cross_decomposition import CCA
from trimesh import load_mesh
from scipy.stats import pearsonr
from aind_ephys_utils import align
import k3d
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.mixture import GaussianMixture

capsule_dirs = capsule_directories()

# %% [markdown]
# # Load criteria and data

# %%
criteria_name = 'basic_ephys'

# %%
# load constraints and data
with open(os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
    
with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json'), 'r') as f:
    constraints = json.load(f)
target_folder = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'basic_ephys')
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
# start with a mask of all True
mask = pd.Series(True, index=combined_tagged_units.index)

# %%
combined_tagged_units_filtered, combined_tagged_units, fig, axes = apply_qc(combined_tagged_units, constraints)

# %% [markdown]
# # Calculate basic ephys features

# %% [markdown]
# ## basic info

# %%
# functions
# def correlate_nan(x, y, lag='full'):
#     """Calculate correlation while ignoring NaNs."""
#     if lag == 'full':
#         lag = len(x) - 1
#     corrs = np.full((lag + 1,), np.nan)
#     for l in range(lag + 1):
#         if l==0:
#             valid_mask = ~np.isnan(x) & ~np.isnan(y)
#             corrs[l] = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
#         else:
#             valid_mask = ~np.isnan(x[:-l]) & ~np.isnan(y[l:])
#             if np.any(valid_mask):
#                 corrs[l] = np.corrcoef(x[:-l][valid_mask], y[l:][valid_mask])[0, 1]
#     return corrs

# def autocorrelation(x, lag):
#     n = len(x)
#     x = x - np.nanmean(x)
#     # result = np.correlate(x, x, mode='full')
#     result = correlate_nan(x, x, lag = lag)  # only valid correlations
#     # result = result[result.size // 2:]  # keep only second half
#     # return result[:lag + 1] / result[0]  # normalize
#     return result/result[0]  # normalize

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
    opto_file = os.path.join(session_dir[f'opto_dir_{data_type}'],
                                f'{session}_opto_session.csv')
    if os.path.exists(opto_file):
        opto_tbl = pd.read_csv(opto_file)
        opto_times = opto_tbl['time'].values
    else:
        opto_times = np.array([])
    if in_df:
        session_df = makeSessionDF(session)
        session_df_go_cue = get_session_tbl(session)
        # session_df['ipsi'] = 2*(session_df['choice'].values - 0.5) * row['rec_side']
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
        # tblTrials_curr = tblTrials.copy()
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                session_df_go_cue_curr = session_df_go_cue_curr[session_df_go_cue_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                session_df_go_cue_curr = session_df_go_cue_curr[session_df_go_cue_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) >2:
            # if session == 'behavior_716325_2024-05-31_10-31-14' and unit_id == 377:
            #     print('Fount it!')
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
            # response = spike_matrix_response[:, response_max_ind]
        
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
            # print(f'Processing unit {unit_id} in session {session}')
            # print(f'{np.sum(np.isinf(response_rate_all))} out of {len(response_rate_all)} trials have inf')
            response_increase = response-fr_bl
            resposne_increase = np.nanmean(response_increase)

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
            # response_rate = np.nanmean(response_rate_all) # verions 1
            response_rate = (np.mean(response) - np.mean(fr_bl))/np.mean(fr_bl)  # version 2
            response_fr = np.mean(response)
            # ax = fig.add_subplot(gs[ind])
            # x = response_first_lat
            # y = response_2_isi
            # # ax.scatter(x, y, s=1, color='k', alpha=0.5)
            # ax.set_yscale('log')
            # nan_mask = np.isnan(x) | np.isnan(y)
            # coeff, p = pearsonr(x[~nan_mask], y[~nan_mask])
            # ax.set_title(f'p={p:.2f}')
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
                # response_diff = np.nanmean(response_rate_all[session_df_go_cue_curr['animal_response'].values != 2]) - \
                #                 np.nanmean(response_rate_all[session_df_go_cue_curr['animal_response'].values == 2]) # version 1
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
data_type = 'raw'
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
def safe_process(row):
    """Wrapper to safely call process() and catch errors."""
    try:
        print(f"Processing session {row['session']}, unit {row['unit']}...")
        return process(row['session'], row['unit'], row['in_df'])
    except Exception as e:
        print(f"[Error] session {row['session']}, unit {row['unit']}: {e}")
        return {'session': row['session'],
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

# results = []
# for ind, row in combined_tagged_units_filtered.iterrows():
#     session = row['session']
#     unit_id = row['unit']
#     df = row['in_df']
#     # check if different session
#     result = process(session, unit_id, df)
#     results.append(result)
#
print(f"Processing {len(combined_tagged_units_filtered)} units with parallelization...")
results = Parallel(n_jobs=-1)(delayed(safe_process)(row) for ind, row in combined_tagged_units_filtered.iterrows())

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
    # result = np.correlate(x, x, mode='full')
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
    opto_file = os.path.join(session_dir[f'opto_dir_{data_type}'],
                                f'{session}_opto_session.csv')
    if os.path.exists(opto_file):
        opto_tbl = pd.read_csv(opto_file)
        opto_times = opto_tbl['time'].values
    else:
        opto_times = np.array([])
    # opto_tbl = pd.read_csv(opto_file)
    # opto_times = opto_tbl['time'].values

    if in_df:
        session_df = makeSessionDF(session)
        session_df_go_cue = get_session_tbl(session)
        # session_df['ipsi'] = 2*(session_df['choice'].values - 0.5) * row['rec_side']
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
        # tblTrials_curr = tblTrials.copy()
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                session_df_go_cue_curr = session_df_go_cue_curr[session_df_go_cue_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                session_df_go_cue_curr = session_df_go_cue_curr[session_df_go_cue_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
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
        # starts = np.arange(rec_start_curr, rec_end_curr, auto_inhi_bin)[:-1]
        # ends = np.arange(rec_start_curr, rec_end_curr, auto_inhi_bin)[1:]
        # for ind, row in enumerate((opto_times_curr)):
        #     start_time = row - pre_time
        #     end_time = row + 0.5*post_time
        #     # set counts in this period to np.nan
        #     mask = (ends >= start_time) & (starts <= end_time)
        #     if np.sum(mask) > 0:
        #         counts_bl[mask] = np.nan
        #         counts[mask] = np.nan
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

data_type = 'raw'
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
# acg_cut = 0.025
# sd_mean_cut = 0.5
# R2_cut = 0.95
# bl_len_cut = 30000
acf_df['acg_1'] = all_acf_bl[:, 1]
acf_df['acg_last'] = all_acf_bl[:, -1]
acf_df['sd_log'] = np.log10(acf_df['sd'] + 1e-3)
acf_df_probe = acf_df.merge(combined_tagged_units_filtered[['session', 'unit_id', 'probe', 'isi_violations']], on=['session', 'unit_id'], how='left')
probes = acf_df_probe['probe'].unique()
# fig, axes = plt.subplots(1, len(filter_list), figsize=(4*len(filter_list), 4))
# for ind, filter_var in enumerate(filter_list):
#     ax = axes[ind]
#     data = acf_df_probe[filter_var].values
#     if filter_var == 'sd':
#         data = np.log10(data + 1e-3)
#     bins = np.linspace(np.nanmin(data), np.nanmax(data), 50)
#     for probe in probes:
#         probe_mask = acf_df_probe['probe'] == probe
#         data_probe = data[probe_mask]
#         if np.sum(~np.isnan(data_probe)) > 0:
#             ax.hist(data_probe, bins=bins, alpha=0.5, label=f'Probe {probe}', density=True)
#         ax.axvline(cut_dict[filter_var], color='k', linestyle='--')
#     ax.set_xlabel(filter_var)
#     ax.legend()
# filter = (acf_df_probe['acg_last']<=cut_dict['acg_last']) & (all_sd < sd_mean_cut) & (all_r2 < R2_cut) & (all_bl_len > bl_len_cut) & (unit_filter)
# acf_df_probe['be_filter'] = filter

# %%
# sns.pairplot(
#     acf_df_probe[filter_list + ['probe', 'isi_violations']],
#     vars=filter_list + ['isi_violations'],
#     hue='probe',
#     diag_kind='hist',
#     plot_kws={'alpha': 0.5},
#     diag_kws={'edgecolor': None, 'linewidth': 0}
# )


# %%
filter = (acf_df_probe['acg_last']<=cut_dict['acg_last']) & (acf_df_probe['sd_log'] < cut_dict['sd_log']) & (acf_df_probe['r2'] < cut_dict['r2']) & (acf_df_probe['bl_len'] > cut_dict['bl_len'])
acf_df_probe['be_filter'] = filter

# %%
# acf_df_probe[~filter][['session', 'unit_id', 'isi_violations']+filter_list].to_csv('excluded_units_acf.csv')

# %%
# plt.plot(all_acf[filter, 1:].T, color='k', alpha=0.1);

# %%
# PCA on acg
pca = PCA(n_components=5)
end_ind = 25

all_acf = np.array(acf_df['acg'].tolist())
all_acf_bl = np.array(acf_df['acg_bl'].tolist())
all_acf_bl_end = np.mean(all_acf_bl[:, -5:], axis=1, keepdims=True)
all_acf_bl = all_acf_bl - all_acf_bl_end
# pca_result = pca.fit_transform(zscore(all_acf_bl[filter, 1:end_ind], axis = 0))
pca_result = pca.fit_transform(all_acf_bl[acf_df_probe['be_filter'].values, 1:end_ind])
# recontruct the ACF curves by pca scores
pca_reconstructed = pca.inverse_transform(pca_result)

basic_ephys_df = basic_ephys_df.merge(acf_df_probe, on=['session', 'unit_id'], how='left')
pc_intial = np.full((len(basic_ephys_df), 1), np.nan)
for pc_ind in range(3):
    basic_ephys_df[f'pc_{pc_ind+1}'] = pc_intial.copy()
    basic_ephys_df.loc[filter, f'pc_{pc_ind+1}'] = pca_result[:, pc_ind]

# %% [markdown]
# ### plotting

# %%
# plt.figure(figsize=(16, 8))

# ax = plt.subplot(221)

# # Prepare x-axis
# x_vals = np.array(range(int(window_length/auto_inhi_bin))) * auto_inhi_bin

# # Sort by R²
# sort_inds = np.argsort(all_r2)
# sorted_r2 = all_r2[sort_inds]


# # Normalize colormap
# norm = mcolors.Normalize(vmin=np.nanmin(all_r2), vmax=np.nanmax(all_r2))
# cmap = cm.viridis
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Create plot
# for idx in sort_inds:
#     # if all_acf[idx, -1]>0.05:
#     #     continue
#     ax.plot(x_vals, all_acf[idx, 1:], color=cmap(norm(all_r2[idx])), alpha=0.8, linewidth=1.5)

# # Add colorbar
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('R²')

# # Labels
# ax.set_xlabel('Lag')
# ax.set_ylabel('ACF')
# ax.set_title('Autocorrelation Curves Sorted by R²')

# ax = plt.subplot(222)


# # Sort by R²
# sort_inds = np.argsort(all_sd)
# sorted_sd = all_sd[sort_inds]

# # Normalize colormap
# norm = mcolors.Normalize(vmin=np.min(all_sd), vmax=0.5)
# cmap = cm.viridis
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Create plot
# for idx in sort_inds:
#     # if all_acf[idx, -1]>0.05:
#     #     continue
#     ax.plot(x_vals, all_acf[idx, 1:], color=cmap(norm(all_sd[idx])), alpha=0.8, linewidth=0.5)

# # Add colorbar
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('SD/mean')

# # Labels
# ax.set_xlabel('Lag')
# ax.set_ylabel('ACF')
# ax.set_title('Autocorrelation Curves Sorted by SD/mean')

# ax = plt.subplot(223)
# ax.hist(x=all_r2, bins=50, color='gray', alpha=0.7)

# ax.set_xlabel('R²')
# ax.set_ylabel('Count')
# ax.set_title('Distribution of R² Values')
# ax = plt.subplot(224)
# ax.hist(x=all_sd, bins=80, color='gray', alpha=0.7)
# ax.set_xlabel('SD/mean')
# ax.set_ylabel('Count')
# ax.set_title('Distribution of SD/mean Values')

# plt.suptitle('All acf')
# plt.tight_layout()
# plt.show()


# %%
# plt.figure(figsize=(16, 8))

# ax = plt.subplot(231)

# # Prepare x-axis
# x_vals = np.array(range(int(window_length/auto_inhi_bin))) * auto_inhi_bin

# # Sort by R²
# sort_inds = np.argsort(all_r2)
# sorted_r2 = all_r2[sort_inds]



# # Normalize colormap
# norm = mcolors.Normalize(vmin=np.nanmin(all_r2), vmax=np.nanmax(all_r2))
# cmap = cm.viridis
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Create plot
# for idx in sort_inds:
#     # if all_acf[idx, -1]>0.05:
#     #     continue
#     ax.plot(x_vals, all_acf_bl[idx, 1:], color=cmap(norm(all_r2[idx])), alpha=0.8, linewidth=1.5)

# # Add colorbar
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('R²')

# # Labels
# ax.set_xlabel('Lag')
# ax.set_ylabel('ACF')
# ax.set_title('Autocorrelation Curves Sorted by R²')

# ax = plt.subplot(232)


# # Sort by R²
# sort_inds = np.argsort(all_sd)
# sorted_sd = all_sd[sort_inds]

# # Normalize colormap
# norm = mcolors.Normalize(vmin=np.min(all_sd), vmax=0.5)
# cmap = cm.viridis
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Create plot
# for idx in sort_inds:
#     # if all_acf_bl[idx, -1]>0.05:
#     #     continue
#     ax.plot(x_vals, all_acf_bl[idx, 1:], color=cmap(norm(all_sd[idx])), alpha=0.8, linewidth=1.5)

# # Add colorbar
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('SD/mean')

# # Labels
# ax.set_xlabel('Lag')
# ax.set_ylabel('ACF')
# ax.set_title('Autocorrelation Curves Sorted by SD/mean')

# ax = plt.subplot(233)


# # Sort by R²
# all_bl_len = acf_df['bl_len'].values
# sort_inds = np.argsort(all_bl_len)[::-1]  # sort in descending order
# sorted_len = all_bl_len[sort_inds]

# # Normalize colormap
# norm = mcolors.Normalize(vmin=np.min(sorted_len), vmax=np.max(sorted_len))
# cmap = cm.viridis
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Create plot
# for idx in sort_inds:
#     # if all_acf_bl[idx, -1]>0.05:
#     #     continue
#     ax.plot(x_vals, all_acf_bl[idx, 1:], color=cmap(norm(all_bl_len[idx])), alpha=0.8, linewidth=1.5)

# # Add colorbar
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('bl_len')

# # Labels
# ax.set_xlabel('Lag')
# ax.set_ylabel('ACF')
# ax.set_title('Autocorrelation Curves Sorted by SD/mean')

# ax = plt.subplot(234)
# ax.hist(x=all_r2, bins=50, color='gray', alpha=0.7)
# ax.set_xlabel('R²')
# ax.set_ylabel('Count')
# ax.set_title('Distribution of R² Values')

# ax = plt.subplot(235)
# ax.hist(x=all_sd, bins=80, color='gray', alpha=0.7)
# ax.set_xlabel('(SD/mean)')
# ax.set_ylabel('Count')
# ax.set_title('Distribution of SD/mean Values')

# ax = plt.subplot(236)
# ax.hist(x=all_bl_len, bins=80, color='gray', alpha=0.7)
# ax.set_xlabel('bl_len')
# ax.set_ylabel('Count')
# ax.set_title('Distribution of bl_len')

# plt.suptitle('All acf baseline')
# plt.tight_layout()
# plt.show()



# %%
# go_bl = basic_ephys_df['go_mean'].values[filter]
# no_go_bl = basic_ephys_df['no_go_mean'].values[filter]
# plt.figure(figsize=(4, 4))
# plt.scatter(go_bl, no_go_bl, s=15, c=basic_ephys_df['response_rate'].values[filter], alpha=0.5, cmap='Reds')
# plt.plot([0, np.nanmax(no_go_bl)], [0, np.nanmax(no_go_bl)], color='k', linestyle='--', linewidth=1.5, label='y=x')

# %%
# x_label = 'pc_1'
# y_label = 'pc_2'
# hue = 'x_ccf'
# test = basic_ephys_df[(basic_ephys_df['be_filter']) & (basic_ephys_df['isi_violations']<=0.1)].copy()
# sns.jointplot(data=test, x=x_label, y=y_label, hue = hue, alpha=0.5)

# nan_mask = np.isnan(basic_ephys_df[x_label].values[filter]) | np.isnan(basic_ephys_df[y_label].values[filter])
# corr, p = pearsonr(basic_ephys_df[x_label].values[filter][~nan_mask], basic_ephys_df[y_label].values[filter][~nan_mask])
# plt.suptitle(f'Correlation: {corr:.2f} p-value: {p:.5f}')
# plt.savefig(os.path.join(target_folder, f'{x_label}_vs_{y_label}_hue_{hue}.pdf'))
# plt.tight_layout()

# %%
# fig = plt.figure(figsize=(16, 8))
# ax = plt.subplot(121)


# # Sort by R²
# sort_inds = np.argsort(all_bl_len[filter])[::-1] # sort in descending order
# sorted_len = all_bl_len[filter][sort_inds]

# # Normalize colormap
# norm = mcolors.Normalize(vmin=np.min(sorted_len), vmax=np.max(sorted_len))
# cmap = cm.viridis
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Create plot
# for idx in sort_inds:
#     # if all_acf_bl[idx, -1]>0.05:
#     #     continue
#     ax.plot(x_vals, all_acf_bl[filter,:][idx, 1:], color=cmap(norm(all_bl_len[filter][idx])), alpha=1, linewidth=0.5)

# # Add colorbar
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('bl_len')

# # Labels
# ax.set_xlabel('Lag')
# ax.set_ylabel('ACF')
# ax.set_title('Post filtering: Baseline autocorrelation Curves Sorted by len')

# ax = plt.subplot(122)
# # Sort by R²
# sort_inds = np.argsort(all_bl_len[filter])[::-1]  # sort in descending order
# sorted_len = all_bl_len[sort_inds]
# # Normalize colormap
# norm = mcolors.Normalize(vmin=np.min(sorted_len), vmax=np.max(sorted_len))
# cmap = cm.viridis
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)
# # Create plot
# for idx in sort_inds:
#     # if all_acf_bl[idx, -1]>0.05:
#     #     continue
#     ax.plot(x_vals, all_acf[filter,:][idx, 1:], color=cmap(norm(all_bl_len[filter][idx])), alpha=1, linewidth=0.5)
# # Add colorbar
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('bl_len')
# # Labels
# ax.set_xlabel('Lag')
# ax.set_ylabel('ACF')
# ax.set_title('Post filtering: All autocorrelation Curves Sorted by len')

# %%
# # drift params
# # filter = all_acf[:, -1] < 1
# focus_ind = np.argmin(np.mean(all_acf[filter, 1:], axis=0))+1
# focus_ind = 0

# norm = mcolors.Normalize(vmin=np.min(all_acf[filter, focus_ind]), vmax=np.max(all_acf[filter, focus_ind]))
# # norm = mcolors.Normalize(vmin=np.min(all_lens), vmax=np.max(all_lens))
# cmap = cm.Reds
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)
# fig = plt.figure(figsize=(18, 6))
# plt.subplot(131)
# sc = plt.scatter(all_r2[filter], 1/all_sd[filter], c=all_acf[filter, focus_ind], cmap=cmap, norm=norm, alpha=0.7)
# plt.xlabel('R²')
# plt.ylabel('SD/mean')
# cbar = plt.colorbar(sc)
# cbar.set_label('ACF at Lag = {:.2f}s'.format(focus_ind * auto_inhi_bin))

# plt.subplot(132)
# cmap = cm.Reds
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)
# norm = mcolors.Normalize(vmin=np.min(all_acf_bl[filter, focus_ind]), vmax=np.max(all_acf_bl[filter, focus_ind]))
# sc = plt.scatter(all_r2[filter], 1/all_sd[filter], c=all_acf_bl[filter, focus_ind], cmap=cmap, norm=norm, alpha=0.7)
# plt.xlabel('R²')
# plt.ylabel('SD/mean')
# cbar = plt.colorbar(sc)
# cbar.set_label('ACF baseline at Lag = {:.2f}s'.format(focus_ind * auto_inhi_bin))

# plt.subplot(133)
# # plt.scatter(all_acf[:, focus_ind], all_acf[:, focus_ind], s = 10)
# # plt.plot([-0.3, 0.3], [-0.3, 0.3], color='k', linestyle='--', alpha=0.5, linewidth=1.5)
# plt.hist(all_acf[:, focus_ind]-all_acf_bl[:, focus_ind], bins=50, color='gray', alpha=0.7)
# plt.xlabel('acf')
# plt.ylabel('acf baseline')

# %%
# # filter = all_acf[:, -1] < 1
# focus_ind = np.argmin(np.mean(all_acf[filter, 1:], axis=0))+1
# focus_ind = 1
# plt.hist(all_acf_bl[filter, focus_ind], bins=30, color='gray', alpha=0.7, label='ACF at last lag');
# plt.title(f'Focus time: {x_vals[focus_ind]:.2f} s')

# %%
# Assume all_acf and filter are defined appropriately
# Example dummy structure:
# all_acf = np.random.rand(100, 20)
# filter = (all_acf[:, -1] < 0.05) & (combined_tagged_units_filtered['unit']!=131) & ~((combined_tagged_units_filtered['unit']==197) & (combined_tagged_units_filtered['session']=="behavior_716325_2024-05-31_10-31-14"))

# Normalize and colormap for color mapping
# focus_feature = 'response_rate'
# focus_feature = 'bl_response_corr'
# color_values = basic_ephys_df[focus_feature].values[basic_ephys_df['be_filter']]
# focus_values = basic_ephys_df[focus_feature].values[basic_ephys_df['be_filter']]
# norm = mcolors.Normalize(vmin=np.nanmin(color_values), vmax=np.nanmax(color_values))
# cmap = cm.Reds

# # Create scatter plot
# plt.figure(figsize=(24, 6))
# plt.subplot(141)
# plt.plot(pca.explained_variance_ratio_[:], marker='o', linestyle='--', color='k', markersize=8)
# plt.xlabel('Principal Component')
# plt.ylabel('Variance Explained')
# # flip to right y axis
# plt.twinx().plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-', color='r', markersize=8)
# plt.ylabel('Cumulative Variance Explained')

# plt.subplot(142)
# sc = plt.scatter(basic_ephys_df['pc_1'].values[basic_ephys_df['be_filter']], basic_ephys_df['pc_2'].values[basic_ephys_df['be_filter']], c=color_values, cmap=cmap, norm=norm, alpha=0.7)

# Create and link colorbar
# cbar = plt.colorbar(sc)
# cbar.set_label(focus_feature)

# # Labels
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA of ACF curves colored by last lag value')


# plt.subplot(143)
# # plot the 2 pcs
# plt.plot(x_vals[:end_ind-1], pca.components_[0], label='PC1', color='blue', linewidth=4)
# plt.plot(x_vals[:end_ind-1], pca.components_[1], label='PC2', color='orange', linewidth=4)
# plt.plot(x_vals[:end_ind-1], pca.components_[2], label='PC3', color='g', linewidth=4)
# plt.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
# plt.legend()

# plt.subplot(144)
# bin acf by focus feature and plot mean
# numbins = 3
# bins = np.linspace(np.nanmin(focus_values), np.nanmax(focus_values), numbins+1)
# bins = np.linspace(np.nanmin(focus_values[~np.isnan(focus_values)]), np.nanmax(focus_values[~np.isnan(focus_values)]), numbins+1)
# # bins = np.quantile(focus_values[~np.isnan(focus_values)], np.linspace(0, 1, numbins+1))
# bin_means = []
# acf_means = []
# all_acf_bl = np.array(basic_ephys_df['acg_bl'].tolist())
# for i in range(numbins):
#     bin_mask = (focus_values >= bins[i]) & (focus_values < bins[i+1])
#     bin_means.append(np.nanmean(focus_values[bin_mask]))
#     # acf_means.append(np.mean(pca_reconstructed[bin_mask, :], axis=0))
#     acf_means.append(np.nanmean(all_acf_bl[basic_ephys_df['be_filter']][bin_mask, 1:end_ind], axis=0))

# colors = cmap(np.linspace(0, 1, numbins))
# for i in range(numbins):
#     plt.plot(x_vals[:end_ind-1], acf_means[i], color=colors[i], label = f'{bin_means[i]:.2f}')
# plt.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)

# plt.legend()
# plt.tight_layout()

# plt.savefig(os.path.join(target_folder, f'pca_autoinhibition_{criteria_name}.pdf'), bbox_inches='tight')



# %%


# %%
# Create normalization object
# norm = mcolors.Normalize(vmin=-0.118, vmax=0.120)
# # Apply colormap with normalization
# colors = cmap(norm(bin_means))
# numbins = 3
# pc_focus = 0
# # bins = np.linspace(np.nanmin(focus_values), np.nanmax(focus_values), numbins+1)
# bins = np.quantile(pca_result[:, pc_focus], np.linspace(0, 1, numbins+1))
# bin_means = []
# acf_means = []
# for i in range(numbins):
#     bin_mask = (pca_result[:, pc_focus] >= bins[i]) & (pca_result[:, pc_focus] < bins[i+1])
#     bin_means.append(np.nanmean(pca_result[bin_mask, pc_focus]))
#     # acf_means.append(np.mean(pca_reconstructed[bin_mask, :], axis=0))
#     acf_means.append(np.nanmean(all_acf_bl[basic_ephys_df['be_filter']][bin_mask, 1:end_ind], axis=0))
# cmap = cm.Reds
# colors = cmap(norm(bin_means)+0.1)
# for i in range(numbins):
#     plt.plot(x_vals[:end_ind-1], acf_means[i], color=colors[i], label = f'{bin_means[i]:.2f}', linewidth = 3)
# plt.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=2)
# plt.title(f'pca_autoinhibition_{criteria_name}_pc{pc_focus}.pdf')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(target_folder, f'pca_autoinhibition_{criteria_name}_pc{pc_focus}.pdf'), bbox_inches='tight')


# %%
# lm = LinearRegression()
# X = pca_result[:, :3]
# filter = basic_ephys_df['be_filter'].values
# y = basic_ephys_df[focus_feature].values[filter]
# nan_filter = np.isnan(y) | np.isnan(X[:, 0])
# lm.fit(X[~nan_filter], y[~nan_filter])
# lm.coef_, lm.intercept_
# focus_feature_predicted = lm.predict(X)
# R2 = lm.score(X[~nan_filter], y[~nan_filter])
# plt.figure(figsize=(4, 4))
# plt.scatter(basic_ephys_df[focus_feature].values[filter], focus_feature_predicted,  norm=norm, alpha=0.7)
# plt.xlabel(focus_feature)
# plt.ylabel(ylabel='Predicted ' + focus_feature)
# plt.title(f'R²: {R2:.2f}')

# %%
# # plot all acf with their id
# plt.figure(figsize=(30, 30))
# for i in range(len(all_acf)):
#     ax = plt.subplot(20, 20, i+1)
#     ax.plot(x_vals, all_acf[i, 1:], color='gray', alpha=0.7)
#     ax.set_title(combined_tagged_units_filtered.iloc[i]['session'] + str(combined_tagged_units_filtered.iloc[i]['unit']))

# %% [markdown]
# ## Add bursting around go cue

# %%
# Par version
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# import math

# def process_unit_burst(row, make_plot=False):
#     """Compute ISI/burst metrics for one unit safely."""
#     session = row['session']
#     unit_id = row['unit_id']
#     isi_v = row['isi_violations']
#     in_df = row['in_df']

#     session_dir = session_dirs(session)
#     unit_tbl = get_unit_tbl(session, data_type)

#     qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
#     with open(qm_file, 'r') as f:
#         qm_dict = json.load(f)
#     rec_start, rec_end = qm_dict['ephys_cut']

#     # Load opto times
#     opto_file = os.path.join(session_dir['opto_dir_curated'], f'{session}_opto_session.csv')
#     if os.path.exists(opto_file):
#         opto_tbl = pd.read_csv(opto_file)
#         opto_times = opto_tbl['time'].values
#     else:
#         opto_times = np.array([])

#     if in_df:
#         session_df = makeSessionDF(session)
#         session_df_go_cue = get_session_tbl(session)
#     else:
#         session_df = None
#         session_df_go_cue = None

#     # Load drift data
#     unit_drift = load_drift(session, unit_id, data_type=data_type)

#     # Load spikes
#     spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
#     spike_times_curr = spike_times.copy()
#     opto_times_curr = opto_times.copy()
#     rec_start_curr, rec_end_curr = rec_start, rec_end

#     # Trim according to drift
#     if unit_drift is not None:
#         if unit_drift['ephys_cut'][0] is not None:
#             spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
#             rec_start_curr = unit_drift['ephys_cut'][0]
#             opto_times_curr = opto_times_curr[opto_times_curr >= unit_drift['ephys_cut'][0]]
#         if unit_drift['ephys_cut'][1] is not None:
#             spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
#             rec_end_curr = unit_drift['ephys_cut'][1]
#             opto_times_curr = opto_times_curr[opto_times_curr <= unit_drift['ephys_cut'][1]]

#     # Skip if no behavioral info
#     if not in_df or session_df_go_cue is None or len(session_df_go_cue) == 0:
#         return dict(session=session, unit=unit_id, isi_mode=np.nan, burst_bi_si=np.nan,
#                     burst_bi_ca=np.nan, burst_bi_da=np.nan, burst_bi_dp=np.nan,
#                     burst_ratio=np.nan, burst_mode=np.nan, non_burst_mode=np.nan,
#                     sep=np.nan, isi_violations=isi_v, amp=row['amp_raw'])

#     # ISI computation
#     all_isi = np.diff(spike_times_curr)
#     dup_inds = np.where(all_isi < 5*(1/30000))[0]
#     spike_times_curr = np.delete(spike_times_curr, dup_inds+1)
#     all_isi = np.diff(spike_times_curr)
#     all_isi = all_isi[all_isi > 0]
#     all_isi = np.log(all_isi)

#     isi_mode = np.mean(all_isi[(all_isi > np.quantile(all_isi, 0.45)) & (all_isi < np.quantile(all_isi, 0.55))])

#     align_time = session_df_go_cue['goCue_start_time'].values
#     spike_df = align.to_events(spike_times_curr, align_time, (0, 10), return_df=True)
#     first_spike_times = np.full(len(session_df_go_cue), np.nan)
#     for i in range(len(session_df_go_cue)):
#         spikes_in_trial = spike_df[spike_df['event_index']==i]['time']
#         if len(spikes_in_trial) > 0:
#             first_spike_times[i] = spikes_in_trial.min()
#     first_spike_times_abs = first_spike_times + align_time
#     df = align.to_events(spike_times_curr, first_spike_times_abs, (-0.05, 0.03), return_df=True)

#     isi_list = df.copy()
#     isi_list['isi'] = np.nan
#     for ind_s, row_s in isi_list.iterrows():
#         event_index = row_s['event_index']
#         time = row_s['time']
#         prev_spikes = spike_times_curr[spike_times_curr < (time + first_spike_times_abs[int(event_index)])]
#         if len(prev_spikes) == 0:
#             isi_list.at[ind_s, 'isi'] = np.nan
#         else:
#             isi_list.at[ind_s, 'isi'] = time + first_spike_times_abs[[int(event_index)]] - prev_spikes[-1]
#     go_isi = np.log(isi_list['isi'].dropna().values)

#     if len(go_isi) < 10:
#         return dict(session=session, unit=unit_id, isi_mode=isi_mode, burst_bi_si=np.nan,
#                     burst_bi_ca=np.nan, burst_bi_da=np.nan, burst_bi_dp=np.nan,
#                     burst_ratio=np.nan, burst_mode=np.nan, non_burst_mode=np.nan,
#                     sep=np.nan, isi_violations=isi_v, amp=row['amp_raw'])

#     # --- GMM fit ---
#     gmm = GaussianMixture(n_components=2)
#     gmm.fit(go_isi.reshape(-1, 1))
#     means = gmm.means_.flatten()
#     stds = np.sqrt(gmm.covariances_).flatten()
#     labels = gmm.predict(go_isi.reshape(-1, 1))
#     score_si = silhouette_score(go_isi.reshape(-1, 1), labels)
#     score_ca = calinski_harabasz_score(go_isi.reshape(-1, 1), labels)
#     score_da = davies_bouldin_score(go_isi.reshape(-1, 1), labels)

#     weights = gmm.weights_
#     idx = np.argsort(means)
#     m1, m2 = means[idx]
#     s1, s2 = stds[idx]
#     w1, w2 = weights[idx]

#     A = 1/(2*s1**2) - 1/(2*s2**2)
#     B = m2/(s2**2) - m1/(s1**2)
#     C = (m1**2)/(2*s1**2) - (m2**2)/(2*s2**2) + math.log((s2*w1)/(s1*w2))
#     discriminant = B**2 - 4*A*C
#     sep_point = np.nan
#     if discriminant >= 0:
#         x1 = (-B + np.sqrt(discriminant)) / (2*A)
#         x2 = (-B - np.sqrt(discriminant)) / (2*A)
#         for x in [x1, x2]:
#             if m1 < x < m2:
#                 sep_point = x
#                 break
#     d_prime = np.abs(m2 - m1) / np.sqrt(0.5 * (s1**2 + s2**2))
#     burst_mode, non_burst_mode = m1, m2
#     burst_ratio = np.sum(labels == idx[0]) / len(labels)

#     # (optional) plotting
#     if make_plot:
#         fig, ax = plt.subplots(figsize=(4, 3))
#         bins = np.linspace(np.nanmin(all_isi), np.nanmax(all_isi), 50)
#         ax.hist(all_isi, bins=bins, color='gray', alpha=0.6, label='All ISI', density=True)
#         ax.hist(go_isi, bins=bins, color='blue', alpha=0.6, label='Go ISI', density=True)
#         ax.axvline(burst_mode, color='r', linestyle='--')
#         ax.axvline(non_burst_mode, color='g', linestyle='--')
#         if not np.isnan(sep_point):
#             ax.axvline(sep_point, color='k', linestyle='--')
#         ax.set_title(f'{session} unit {unit_id}')
#         plt.close(fig)

#     return dict(session=session, unit=unit_id, isi_mode=isi_mode,
#                 burst_bi_si=score_si, burst_bi_ca=score_ca, burst_bi_da=score_da,
#                 burst_bi_dp=d_prime, burst_ratio=burst_ratio, burst_mode=burst_mode,
#                 non_burst_mode=non_burst_mode, sep=sep_point,
#                 isi_violations=isi_v, amp=row['amp_raw'])



# # %%
# data_type = 'curated'
# target = 'soma'
# make_plot = False

# # Parameters
# auto_inhi_bin = 0.05
# window_length = 3.0
# pre_time = 0
# post_time = 2.5


# ---------------------------------------------------------------------
# PARALLEL EXECUTION
# ---------------------------------------------------------------------

# results = Parallel(n_jobs=4, backend='loky', verbose=10)(
#     delayed(process_unit_burst)(row) for _, row in combined_tagged_units_filtered.iterrows()
# )

# burst_df = pd.DataFrame(results)


# %%
# # merge with basic_ephys
# basic_ephys_df = pd.merge(basic_ephys_df, burst_df[['session', 'unit_id', 'burst_ratio', 'burst_bi_dp']], on=['session', 'unit_id'], how='left')

# %%
basic_ephys_df.rename(columns={'probe_x': 'probe'}, inplace=True)

# %%
# save basic ephys data
with open(os.path.join(target_folder, f'basic_ephys.pkl'), 'wb') as f:
    pickle.dump(basic_ephys_df, f)
print(f'Saved basic ephys data to {os.path.join(target_folder, f"basic_ephys.pkl")}')



