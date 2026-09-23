# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
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
from utils.ephys_functions import*
from utils.ccf_utils import ccf_pts_convert_to_mm
from utils.combine_tools import apply_qc, to_str_intlike, merge_df_with_suffix
from utils.plot_utils import combine_pdf_big
from utils.capsule_migration import capsule_directories
import pickle
import scipy.stats as stats
import spikeinterface as si
import shutil
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import r2_score
import warnings
from scipy.stats import gaussian_kde
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from utils.ccf_utils import ccf_pts_convert_to_mm
from trimesh import load_mesh
from scipy.stats import pearsonr
from aind_ephys_utils import align
capsule_dirs = capsule_directories()
# %%
criteria_name = 'beh_all'
version = 'PrL_S1'
# load constraints and data
with open(os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
combined_tagged_units['unit_id'] = combined_tagged_units['unit'].apply(to_str_intlike)
# antidromic data
antidromic_file = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'antidromic_analysis', version, 'combined_antidromic_results.pkl')
with open(antidromic_file, 'rb') as f:
    antidromic_df = pickle.load(f)

antidromic_df.rename(columns={'unit': 'unit_id'}, inplace=True)
antidromic_df['unit_id'] = antidromic_df['unit_id'].apply(to_str_intlike)
antidromic_df = antidromic_df[['unit_id', 'session', 'p_auto_inhi', 't_auto_inhi',
       'p_collision', 't_collision', 'p_antidromic', 't_antidromic', 'tier_1',
       'tier_2', 'tier_1_long', 'tier_2_long']].copy()
combined_tagged_units = combined_tagged_units.merge(antidromic_df, on=['session', 'unit_id'], how='left')
combined_tagged_units['tier_1'].fillna(False, inplace=True)
combined_tagged_units['tier_2'].fillna(False, inplace=True)
combined_tagged_units['tier_1_long'].fillna(False, inplace=True)
combined_tagged_units['tier_2_long'].fillna(False, inplace=True)   
with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json'), 'r') as f:
    constraints = json.load(f)
beh_folder = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'response_regression')
if not os.path.exists(beh_folder):
    os.makedirs(beh_folder)
    


# %%
# start with a mask of all True
combined_tagged_units_filtered, combined_tagged_units, fig, axes = apply_qc(combined_tagged_units, constraints)


# %%
def regressors_to_formula(response_var, regressors):
    terms = [r for r in regressors if r != 'Intercept']
    has_intercept = 'Intercept' in regressors
    rhs = '1' if has_intercept else '0'
    if terms:
        rhs += ' + ' + ' + '.join(terms)
    return f'{response_var} ~ {rhs}'


# # Dataframe generation

# %%
def nan_result(session, unit_id):
    """All-NaN row, shaped like process()'s return, for units that cannot be fitted."""
    n_reg = len(regressors_focus_all) + len(regressors_focus_hit)
    out = {'session': session, 'unit': unit_id}
    for period in ('ratio', 'baseline', 'response'):
        out[f'coefs_{period}'] = np.full(n_reg, np.nan)
        out[f'T_{period}'] = np.full(n_reg, np.nan)
        out[f'p_{period}'] = np.full(n_reg, np.nan)
        out[f'r2_{period}'] = [np.nan, np.nan]
    return out


def process(session, unit_id, rec_side, formula_all, formula_hit, focus_all, focus_hit, align_name = 'go_cue'):
    session_dir = session_dirs(session)
    unit_tbl = get_unit_tbl(session, data_type)
    session_df = get_session_tbl(session, cut_interruptions=True)
    session_df['hit'] = session_df['animal_response'] != 2
    # Do NOT re-index trial_ind here. get_session_tbl already sets trial_ind from the
    # index of the *uncut* table, which is the convention the drift trial table uses
    # (generate_session_opto_drift_trial_table indexes trials of get_session_tbl(session)
    # with no cut_interruptions). Overwriting it with arange(len(session_df)) shifts it by
    # the number of leading trials cut_interruptions drops -- nonzero in 15 of the 36
    # beh_all sessions, up to 273 trials -- so the amp/motion merge below silently pairs
    # each trial with a different trial's amplitude.
    # add svs column
    svs = ((session_df['animal_response'].values[1:]==1) & (session_df['animal_response'].values[:-1]==0)) | ((session_df['animal_response'].values[1:]==0) & (session_df['animal_response'].values[:-1]==1))
    svs = np.concatenate(([0], svs.astype(int)))
    session_df['svs'] = svs
    # add ipsi column
    side = session_df['animal_response'].values
    side[side==0] = -1  # left choice
    side[side==2] = 0  # no response trials
    side = side * rec_side
    session_df['ipsi'] = side
    drift_data = load_trial_drift(session, data_type)
    loaded_session = session

    unit_drift = load_drift(session, unit_id, data_type=data_type)
    spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
    session_df_curr = session_df.copy()
    spike_times_curr = spike_times.copy()
    unit_trial_drift_curr = drift_data.load_unit(unit_id)
    # tblTrials_curr = tblTrials.copy()
    if unit_drift is not None:
        if unit_drift['ephys_cut'][0] is not None:
            spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
            session_df_curr = session_df_curr[session_df_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
        if unit_drift['ephys_cut'][1] is not None:
            spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
            session_df_curr = session_df_curr[session_df_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
            # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
    # Some units' ephys_cut leaves no usable trials at all (4 of the 128 beh_all units:
    # their cut ends before the first trial of the interruption-free block), and
    # align.to_events raises on an empty align_time. Bail out with NaNs instead.
    min_trials = 20
    if len(session_df_curr) < min_trials or int(session_df_curr['hit'].sum()) < min_trials:
        print(f'{session} {unit_id}: only {len(session_df_curr)} trials '
              f'({int(session_df_curr["hit"].sum())} hit) after the drift cut, skipping')
        return nan_result(session, unit_id)
    if 'amp_abs' in formula_hit or 'amp' in formula_hit or 'amp_abs' in formula_all or 'amp' in formula_all:
        # get unit_trial_drift_curr's rows corresponding to the ones in session_df_curr
        session_df_curr = session_df_curr.merge(unit_trial_drift_curr, on='trial_ind', how='left').copy()
        # the merge is on trial_ind; if it ever silently fails to match, amp becomes all
        # NaN and patsy drops every row, so check rather than fit on nothing
        if session_df_curr['amp'].isna().all():
            print(f'{session} {unit_id}: amp all NaN after merging the drift trial table '
                  f'(trial_ind mismatch), skipping')
            return nan_result(session, unit_id)
    if align_name == 'go_cue':
        if 'go_cue' in session_df_curr.columns:
            align_time = session_df_curr['go_cue'].values
        else:
            align_time = session_df_curr['goCue_start_time'].values
        # align_time_all = tblTrials_curr['goCue_start_time'].values
    elif align_name == 'response':
        if 'choice_time' in session_df_curr.columns:
            align_time = session_df_curr['choice_time'].values
        else:
            align_time = session_df_curr['reward_outcome_time'].values
        # align_time_all = tblTrials_curr['reward_outcome_time'].values
    # spike_matrix, slide_times = get_spike_matrix(spike_times_curr, align_time, 
    #                                             pre_event=pre_event, post_event=post_event, 
    #                                             binSize=binSize, stepSize=stepSize)
    spikes_bl = align.to_events(spike_times_curr, align_time, (pre_event, -0.01), return_df=True)
    spikes_bl_count = spikes_bl.groupby('event_index').size()
    spikes_bl_rate = [spikes_bl_count.get(i, 0) for i in range(len(align_time))]/np.abs(pre_event)

    response = align.to_events(spike_times_curr, align_time, [0.01, post_event], return_df=True)
    response_count = response.groupby('event_index').size()
    response_rate = [response_count.get(i, 0) for i in range(len(align_time))]/np.abs(post_event)

    response_ratio = np.full(len(response_rate), np.nan)
    non_zero_bl = np.array(spikes_bl_rate) > 0 # only compute ratio for trials with non-zero baseline firing to avoid inf values
    response_ratio[non_zero_bl] = (np.array(response_rate)[non_zero_bl] - np.array(spikes_bl_rate)[non_zero_bl])/ np.array(spikes_bl_rate)[non_zero_bl]
    # spike_matrix_all, slide_times = get_spike_matrix(spike_times_curr, align_time_all, 
    #                                             pre_event=pre_event, post_event=post_event, 
    #                                             binSize=binSize, stepSize=stepSize)
    # spike_matrix_LM = zscore(response_ratio)  
    if np.sum(1-session_df_curr['hit'].values) < 2:
        formula_all = re.sub(r"\s*\+\s*hit", "", formula_all)
    # try:

    hit_ind = np.where(session_df_curr['hit'].values)[0]
    # ratio
    # all  trials
    regressors, curr_T, curr_p, curr_coefs, _, r2_score_value = fitSpikeModelG(session_df_curr, response_ratio.reshape(-1, 1), formula_all)
    regressors_hit, curr_T_hit, curr_p_hit, curr_coefs_hit, _, r2_score_value_hit = fitSpikeModelG(session_df_curr.iloc[hit_ind].reset_index(), response_ratio[hit_ind].reshape(-1, 1), formula_hit)
    # pick regressors from regressors_focus list, sort them according to sequence in regressors_focus
    curr_T_focus = []
    curr_p_focus = []
    curr_coefs_focus = []

    for r in focus_all:
        if r in regressors:
            idx = regressors.index(r)
            curr_T_focus.append([T_t[idx] for T_t in curr_T])
            curr_p_focus.append([p_t[idx] for p_t in curr_p])
            curr_coefs_focus.append([coef_t[idx] for coef_t in curr_coefs])
        else:
            # fill with nan arrays of correct shape
            curr_T_focus.append([np.nan for _ in curr_T])
            curr_p_focus.append([np.nan for _ in curr_p])
            curr_coefs_focus.append([np.nan for _ in curr_coefs])
    
    for r in focus_hit:
        if r in regressors_hit:
            idx = regressors_hit.index(r)
            curr_T_focus.append([T_t[idx] for T_t in curr_T_hit])
            curr_p_focus.append([p_t[idx] for p_t in curr_p_hit])
            curr_coefs_focus.append([coef_t[idx] for coef_t in curr_coefs_hit])
        else:
            # fill with nan arrays of correct shape
            curr_T_focus.append([np.nan for _ in curr_T_hit])
            curr_p_focus.append([np.nan for _ in curr_p_hit])
            curr_coefs_focus.append([np.nan for _ in curr_coefs_hit])

    # Convert back to consistent array-of-arrays shape
    curr_T = np.array(curr_T_focus).T
    curr_p = np.array(curr_p_focus).T
    curr_coefs = np.array(curr_coefs_focus).T

    coefs_ratio = curr_coefs[0]
    T_ratio = curr_T[0]
    p_ratio = curr_p[0]
    r2_ratio = [r2_score_value, r2_score_value_hit]



    # baseline
    regressors, curr_T, curr_p, curr_coefs, _, r2_score_value  = fitSpikeModelG(session_df_curr, np.array(spikes_bl_rate).reshape(-1, 1), formula_all)
    regressors_hit, curr_T_hit, curr_p_hit, curr_coefs_hit, _, r2_score_value_hit = fitSpikeModelG(session_df_curr.iloc[hit_ind].reset_index(), np.array(spikes_bl_rate)[hit_ind].reshape(-1, 1), formula_hit)
    # pick regressors from regressors_focus list, sort them according to sequence in regressors_focus
    curr_T_focus = []
    curr_p_focus = []
    curr_coefs_focus = []

    for r in focus_all:
        if r in regressors:
            idx = regressors.index(r)
            curr_T_focus.append([T_t[idx] for T_t in curr_T])
            curr_p_focus.append([p_t[idx] for p_t in curr_p])
            curr_coefs_focus.append([coef_t[idx] for coef_t in curr_coefs])
        else:
            # fill with nan arrays of correct shape
            curr_T_focus.append([np.nan for _ in curr_T])
            curr_p_focus.append([np.nan for _ in curr_p])
            curr_coefs_focus.append([np.nan for _ in curr_coefs])

    for r in focus_hit:
        if r in regressors_hit:
            idx = regressors_hit.index(r)
            curr_T_focus.append([T_t[idx] for T_t in curr_T_hit])
            curr_p_focus.append([p_t[idx] for p_t in curr_p_hit])
            curr_coefs_focus.append([coef_t[idx] for coef_t in curr_coefs_hit])
        else:
            # fill with nan arrays of correct shape
            curr_T_focus.append([np.nan for _ in curr_T_hit])
            curr_p_focus.append([np.nan for _ in curr_p_hit])
            curr_coefs_focus.append([np.nan for _ in curr_coefs_hit])

    # Convert back to consistent array-of-arrays shape
    curr_T = np.array(curr_T_focus).T
    curr_p = np.array(curr_p_focus).T
    curr_coefs = np.array(curr_coefs_focus).T

    coefs_baseline = curr_coefs[0]
    T_baseline = curr_T[0]
    p_baseline = curr_p[0]
    r2_baseline = [r2_score_value, r2_score_value_hit]


    # response
    regressors, curr_T, curr_p, curr_coefs, _, r2_score_value  = fitSpikeModelG(session_df_curr, np.array(response_rate).reshape(-1, 1), formula_all)
    regressors_hit, curr_T_hit, curr_p_hit, curr_coefs_hit, _, r2_score_value_hit = fitSpikeModelG(session_df_curr.iloc[hit_ind].reset_index(), np.array(response_rate)[hit_ind].reshape(-1, 1), formula_hit)
    # pick regressors from regressors_focus list, sort them according to sequence in regressors_focus
    curr_T_focus = []
    curr_p_focus = []
    curr_coefs_focus = []

    for r in focus_all:
        if r in regressors:
            idx = regressors.index(r)
            curr_T_focus.append([T_t[idx] for T_t in curr_T])
            curr_p_focus.append([p_t[idx] for p_t in curr_p])
            curr_coefs_focus.append([coef_t[idx] for coef_t in curr_coefs])
        else:
            # fill with nan arrays of correct shape
            curr_T_focus.append([np.nan for _ in curr_T])
            curr_p_focus.append([np.nan for _ in curr_p])
            curr_coefs_focus.append([np.nan for _ in curr_coefs])
    
    for r in focus_hit:
        if r in regressors_hit:
            idx = regressors_hit.index(r)
            curr_T_focus.append([T_t[idx] for T_t in curr_T_hit])
            curr_p_focus.append([p_t[idx] for p_t in curr_p_hit])
            curr_coefs_focus.append([coef_t[idx] for coef_t in curr_coefs_hit])
        else:
            # fill with nan arrays of correct shape
            curr_T_focus.append([np.nan for _ in curr_T_hit])
            curr_p_focus.append([np.nan for _ in curr_p_hit])
            curr_coefs_focus.append([np.nan for _ in curr_coefs_hit])

    # Convert back to consistent array-of-arrays shape
    curr_T = np.array(curr_T_focus).T
    curr_p = np.array(curr_p_focus).T
    curr_coefs = np.array(curr_coefs_focus).T
    coefs_response = curr_coefs[0]
    T_response = curr_T[0]
    p_response = curr_p[0]
    r2_response = [r2_score_value, r2_score_value_hit]

    return {'coefs_ratio': coefs_ratio,
            'T_ratio': T_ratio,
            'p_ratio': p_ratio,
            'r2_ratio': r2_ratio,
            'coefs_baseline': coefs_baseline,
            'T_baseline': T_baseline,
            'p_baseline': p_baseline,
            'r2_baseline': r2_baseline,
            'coefs_response': coefs_response,
            'T_response': T_response,
            'p_response': p_response,
            'r2_response': r2_response,
            'session': session,
            'unit': unit_id}

# %%
# get all t-stats, coeffs, p-values no parallel
# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
data_type = 'raw'
target = 'soma'
align_name = 'go_cue'
regressors_focus_all = ['hit', 'amp', 'Intercept','svs']
regressors_focus_hit = ['amp', 'svs', 'ipsi', 'Intercept', 'ipsi:svs']
# all_regressors = regressors_focus + regressors_sup
formula_all = regressors_to_formula('spikes', regressors_focus_all)
formula_hit = regressors_to_formula('spikes', regressors_focus_hit)
pre_event = -2
post_event = 0.5
model_name = 'stan_qLearning_5params'

n_units = len(combined_tagged_units_filtered)
n_regressors = len(regressors_focus_all)+len(regressors_focus_hit)

# ------------------------------------------------------------------
# Parallel execution
# ------------------------------------------------------------------
def safe_process(row, formula_all, formula_hit, regressors_focus_all, regressors_focus_hit):
    """Wrapper to safely call process() and catch errors."""
    try:
        return process(row['session'], row['unit'], row['rec_side'], formula_all, formula_hit, regressors_focus_all, regressors_focus_hit, align_name='go_cue')
    except Exception as e:
        print(f"[Error] session {row['session']}, unit {row['unit']}: {type(e).__name__}: {e}")
        return nan_result(row['session'], row['unit'])
    # except Exception as e:
    #     print(f"[Error] session {row['session']}, unit {row['unit']}: {e}")
    #     return {'session': row['session'],
    #             'unit': row['unit'],
    #             'coefs_ratio': np.full(n_regressors, np.nan),
    #             'T_ratio': np.full(n_regressors, np.nan),
    #             'p_ratio': np.full(n_regressors, np.nan),
    #             'coefs_baseline': np.full(n_regressors, np.nan),
    #             'T_baseline': np.full(n_regressors, np.nan),
    #             'p_baseline': np.full(n_regressors, np.nan),
    #             'coefs_response': np.full(n_regressors, np.nan),
    #             'T_response': np.full(n_regressors, np.nan),
    #             'p_response': np.full(n_regressors, np.nan)}


# %%

# Run across all units
results = Parallel(n_jobs=-4, backend='loky', verbose=0)(
    delayed(safe_process)(row, formula_all, formula_hit, regressors_focus_all, regressors_focus_hit)
    for ind, row in combined_tagged_units_filtered.iterrows()
)


# %%
# ------------------------------------------------------------------
# Combine results into arrays
# ------------------------------------------------------------------
# Extract values in order
all_coefs_ratio    = np.vstack([r['coefs_ratio'] for r in results])
all_T_ratio        = np.vstack([r['T_ratio'] for r in results])
all_p_ratio        = np.vstack([r['p_ratio'] for r in results])
all_coefs_baseline = np.vstack([r['coefs_baseline'] for r in results])
all_T_baseline     = np.vstack([r['T_baseline'] for r in results])
all_p_baseline     = np.vstack([r['p_baseline'] for r in results])
all_coefs_response = np.vstack([r['coefs_response'] for r in results])
all_T_response     = np.vstack([r['T_response'] for r in results])
all_p_response     = np.vstack([r['p_response'] for r in results])

all_r2 = {}
all_r2['response'] = np.vstack([r['r2_response'] for r in results])
all_r2['baseline'] = np.vstack([r['r2_baseline'] for r in results])
all_r2['ratio']  = np.vstack([r['r2_ratio'] for r in results])

# Optionally create a combined summary DataFrame
summary_df = pd.DataFrame({
    'session': [r['session'] for r in results],
    'unit': [r['unit'] for r in results],
})
all_regressors = [r+'_all' for r in regressors_focus_all] + [r+'_hit' for r in regressors_focus_hit]

for i, reg in enumerate(all_regressors):
    summary_df[f'coef_ratio_{reg}']    = all_coefs_ratio[:, i]
    summary_df[f'T_ratio_{reg}']       = all_T_ratio[:, i]
    summary_df[f'p_ratio_{reg}']       = all_p_ratio[:, i]
    summary_df[f'coef_baseline_{reg}'] = all_coefs_baseline[:, i]
    summary_df[f'T_baseline_{reg}']    = all_T_baseline[:, i]
    summary_df[f'p_baseline_{reg}']    = all_p_baseline[:, i]
    summary_df[f'coef_response_{reg}'] = all_coefs_response[:, i]
    summary_df[f'T_response_{reg}']    = all_T_response[:, i]
    summary_df[f'p_response_{reg}']    = all_p_response[:, i]

periods = ['ratio', 'baseline', 'response']
for period in periods:
    summary_df[f'r2_{period}_all'] = all_r2[period][:, 0]
    summary_df[f'r2_{period}_hit'] = all_r2[period][:, 1]

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
print("✅ Parallel processing complete.")


# %%
# histogram of all t-stats for regressors focus
fig = plt.figure(figsize=(3*len(all_regressors), 7.5))
gs = gridspec.GridSpec(3, len(all_regressors), figure=fig, hspace=0.5)
cmap = plt.cm.cool  # Get the colormap
colors = cmap(np.linspace(0, 1, len(all_regressors)))
periods = ['ratio', 'baseline', 'response']
for reg_ind, regressor in enumerate(all_regressors):
    for period_ind, period in enumerate(periods):
        curr_Ts = summary_df[[f'T_{period}_{regressor}']].values.flatten()
        curr_Ps = summary_df[[f'p_{period}_{regressor}']].values.flatten()
        x_limit = np.nanmax(np.abs(curr_Ts))
        ax = fig.add_subplot(gs[period_ind, reg_ind])
        bins = np.linspace(-x_limit-0.01, x_limit+0.01, 30)
        ax.hist(curr_Ts[curr_Ps<0.05], bins=bins, color=colors[reg_ind], alpha=0.7, edgecolor='none')  # plot T-statistics with p<0.05
        ax.hist(curr_Ts[curr_Ps>=0.05], bins=bins, color='lightgray', alpha=0.5, edgecolor='none')
        ax.set_xlim(-x_limit, x_limit)
        if period_ind == 0:
            ax.set_title(f'{regressor}', fontsize=12)
        # turn off y-ticks
        ax.set_yticks([])
        if reg_ind == 0:
            ax.set_ylabel(f'{period}', fontsize=12)
plt.suptitle('T-statistics for all regressors', fontsize=14)
plt.savefig(os.path.join(beh_folder, f'Regression_summary_{criteria_name}_{align_name}.pdf'), bbox_inches='tight')

# %%
# find same regressor in all and hit, compare with scatter plot
fig = plt.figure(figsize=(15, 12))
sig_thresh = 0.05
regressors_common = set(regressors_focus_all).intersection(set(regressors_focus_hit))
gs = gridspec.GridSpec(3, len(regressors_common), figure=fig, hspace=0.5)
for reg_ind, regressor in enumerate(regressors_common):
    for period_ind, period in enumerate(periods):
        ax = fig.add_subplot(gs[period_ind, reg_ind])
        curr_T_all = summary_df[[f'T_{period}_{regressor}_all']].values.flatten()
        curr_T_hit = summary_df[[f'T_{period}_{regressor}_hit']].values.flatten()
        curr_coefs_all = summary_df[[f'coef_{period}_{regressor}_all']].values.flatten()
        curr_coefs_hit = summary_df[[f'coef_{period}_{regressor}_hit']].values.flatten()
        curr_p_all = summary_df[[f'p_{period}_{regressor}_all']].values.flatten()
        ax.scatter(curr_T_all[curr_p_all<sig_thresh], curr_T_hit[curr_p_all<sig_thresh], alpha=0.25, color='k', edgecolors='none', s=20)
        ax.scatter(curr_T_all[curr_p_all>=sig_thresh], curr_T_hit[curr_p_all>=sig_thresh], alpha=0.1, color='gray', edgecolors='none', s=20)
        # add diagonal line
        lims = np.array([
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ])/0.9
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

        ax.set_xlabel(f'All trials')
        ax.set_ylabel(f'Hit trials')
        # limit = np.max([np.nanmax(np.abs(curr_T_all)), np.nanmax(np.abs(curr_T_hit))])/0.9
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[0], lims[1])
        ax.set_title(f'{regressor} - {period}')

plt.tight_layout()
plt.suptitle('Coefficient comparison between all and hit trials', fontsize=14)
plt.savefig(os.path.join(beh_folder, f'Regression_comparison_all_hit_{criteria_name}_{align_name}.pdf'), bbox_inches='tight')

# %%
# compare r2
fig = plt.figure(figsize=(10, 5))
sig_thresh = 0.05
gs = gridspec.GridSpec(1, 3, figure=fig)
for period_ind, period in enumerate(periods):
    ax = fig.add_subplot(gs[0, period_ind])
    curr_r2_all = summary_df[[f'r2_{period}_all']].values.flatten()
    curr_r2_hit = summary_df[[f'r2_{period}_hit']].values.flatten()
    ax.scatter(curr_r2_all, curr_r2_hit, alpha=0.25, color='k', edgecolors='none', s=20)
    # add diagonal line
    lims = np.array([
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ])/0.9
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    ax.set_xlabel(f'All trials')
    ax.set_ylabel(f'Hit trials')
    # limit = np.max([np.nanmax(np.abs(curr_T_all)), np.nanmax(np.abs(curr_T_hit))])/0.9
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[0], lims[1])
    ax.set_title(f'R² - {period}')

plt.tight_layout()
plt.suptitle('R² comparison between all and hit trials', fontsize=14)
plt.savefig(os.path.join(beh_folder, f'Regression_R2_comparison_all_hit_{criteria_name}_{align_name}.pdf'), bbox_inches='tight')

# %%
summary_df.to_csv(os.path.join(beh_folder, f'response_ratio_{criteria_name}_{align_name}.csv'), index=False)
print(f"Saved regression results to {os.path.join(beh_folder, f'response_ratio_{criteria_name}_{align_name}.csv')}")
