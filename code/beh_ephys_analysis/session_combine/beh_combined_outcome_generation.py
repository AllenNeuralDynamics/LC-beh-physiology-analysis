# %%
import sys
import os

from pandas.core.apply import com
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
from utils.ephys_functions import*
from utils.ccf_utils import ccf_pts_convert_to_mm, pir_to_lps
from utils.combine_tools import apply_qc, merge_df_with_suffix, to_str_intlike
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
import statsmodels.api as sm
from aind_ephys_utils import align
warnings.filterwarnings('ignore')

# %%
criteria_name = 'beh_all'
version = 'PrL_S1'
overview = False

# %%
# load constraints and data
with open(os.path.join('/root/capsule/scratch/combined/combine_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
combined_tagged_units['unit_id'] = combined_tagged_units['unit'].apply(to_str_intlike)
with open(os.path.join('/root/capsule/scratch/combined/combined_session_tbl', 'combined_beh_sessions.pkl'), 'rb') as f:
    combined_session_qc = pickle.load(f)
# combined_session_qc.drop(columns=['probe'], inplace=True, errors='ignore')
# combined_tagged_units = combined_tagged_units.merge(combined_session_qc, on='session', how='left')

# antidromic data
antidromic_file = f'/root/capsule/scratch/combined/beh_plots/basic_ephys_low/{version}/combined_antidromic_results.pkl'
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
beh_folder = os.path.join('/root/capsule/scratch/combined/beh_plots', criteria_name)
beh_folder = os.path.join(beh_folder, 'figures_in_generation')
if not os.path.exists(beh_folder):
    os.makedirs(beh_folder)
# start with a mask of all True
mask = pd.Series(True, index=combined_tagged_units.index)

# %%
combined_tagged_units_filtered, combined_tagged_units, fig = apply_qc(combined_tagged_units, constraints)

if overview:
# %% [markdown]
# # Overview by regression over trial time

    # %%
    data_type = 'curated'
    target = 'soma'
    all_coefs = []
    all_T = []
    all_p = []
    align_name = 'response'
    regressors_focus = ['Qchosen', 'outcome', 'ipsi', 'outcome:ipsi', 'svs']
    regressors_sup = ['Intercept']
    all_regressors = regressors_focus + regressors_sup
    formula = regressors_to_formula('spikes', all_regressors)

    curr_session = None
    pre_event = -1
    post_event = 2.5
    model_name = 'stan_qLearning_5params'
    binSize = 1.5
    loaded_session = None
    for ind, row in combined_tagged_units_filtered.iterrows():
        session = row['session']
        unit_id = row['unit']
        session_dir = session_dirs(session, model_name)
        if not os.path.exists(session_dir['model_file']):
            print(f'Model not fitted for session {session}')
        # check if different session
        if loaded_session is None or loaded_session != session:
            unit_tbl = get_unit_tbl(session, data_type)
            session_df = makeSessionDF(session, model_name = model_name, cut_interruptions=True)
            session_df['ipsi'] = 2*(session_df['choice'].values - 0.5) * row['rec_side']
            drift_data = load_trial_drift(session, data_type)
            loaded_session = session
            print(f'Loaded session: {session}')
        unit_drift = load_drift(session, unit_id, data_type=data_type)
        spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
        session_df_curr = session_df.copy()
        spike_times_curr = spike_times.copy()
        unit_trial_drift_curr = drift_data.load_unit(unit_id)
        # tblTrials_curr = tblTrials.copy()
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if 'amp_abs' in all_regressors or 'amp' in all_regressors:
            # get unit_trial_drift_curr's rows corresponding to the ones in session_df_curr
            session_df_curr = session_df_curr.merge(unit_trial_drift_curr, on='trial_ind', how='left').copy()
        if align_name == 'go_cue':
            align_time = session_df_curr['go_cue_time'].values
            # align_time_all = tblTrials_curr['goCue_start_time'].values
        elif align_name == 'response':
            align_time = session_df_curr['choice_time'].values
            # align_time_all = tblTrials_curr['reward_outcome_time'].values
        # spike_matrix, slide_times = get_spike_matrix(spike_times_curr, align_time, 
        #                                             pre_event=pre_event, post_event=post_event, 
        #                                             binSize=binSize, stepSize=stepSize)
        spike_matrix_LM, slide_times_LM = get_spike_matrix(spike_times_curr, align_time, 
                                                    pre_event=pre_event, post_event=post_event, 
                                                    binSize=binSize, stepSize=0.25)
        # spike_matrix_all, slide_times = get_spike_matrix(spike_times_curr, align_time_all, 
        #                                             pre_event=pre_event, post_event=post_event, 
        #                                             binSize=binSize, stepSize=stepSize)
        spike_matrix_LM = zscore(spike_matrix_LM, axis=0)  
        
            
        # try:
        regressors, curr_T, curr_p, curr_coefs = fitSpikeModelG(session_df_curr, spike_matrix_LM, formula)
        # pick regressors from regressors_focus list, sort them according to sequence in regressors_focus
        focus_ind = [regressors.index(r) for r in regressors_focus if r in regressors]
        curr_T = [T_t[focus_ind] for T_t in curr_T]
        curr_p = [p_t[focus_ind] for p_t in curr_p]
        curr_coefs = [coef_t[focus_ind] for coef_t in curr_coefs] 

        if len(curr_T) == 4:
            print(f'Session: {session}, Unit: {unit_id}')

        all_coefs.append(curr_coefs)
        all_T.append(curr_T)
        all_p.append(curr_p)

    # %%
    all_coefsm = np.array(all_coefs)
    all_Tm = np.array(all_T)
    all_pm = np.array(all_p)

    # %%
    fig = plt.figure(figsize=(30, 12))
    sig_thresh = 1
    gs = gridspec.GridSpec(len(regressors_focus)+3, len(slide_times_LM), figure=fig, hspace=0.5, height_ratios=[2] + [1]*len(regressors_focus) + [1, 1])
    cmap = plt.cm.cool  # Get the colormap
    colors = cmap(np.linspace(0, 1, len(regressors_focus)))
    ax_all = fig.add_subplot(gs[0, :]) 
    for reg_ind, regressor in enumerate(regressors_focus):
        x_limit = np.max(np.abs(all_Tm[:, :, reg_ind]))
        for time_ind, time in enumerate(slide_times_LM):
            ax = fig.add_subplot(gs[reg_ind+1, time_ind])
            curr_Ts = all_Tm[:, time_ind, reg_ind]  # get the T-statistics for the current regressor and time
            curr_Ps = all_pm[:, time_ind, reg_ind]  # get the p-values for the current regressor and time
            bins = np.linspace(-x_limit-0.01, x_limit+0.01, 30)
            ax.hist(curr_Ts[curr_Ps<0.05], bins=bins, color=colors[reg_ind], alpha=0.7, edgecolor='none')  # plot T-statistics with p<0.05
            ax.hist(curr_Ts[curr_Ps>=0.05], bins=bins, color='lightgray', alpha=0.5, edgecolor='none')
            ax.set_xlim(-x_limit, x_limit)
            if time_ind == 0:
                ax.set_ylabel(f'{regressor}', fontsize=12)
            # turn off y-ticks and x-ticks
            ax.set_yticks([])
            # ax.set_xticks([])

        curr_Ps = np.squeeze(all_pm[:, :, reg_ind])
        curr_Ts = all_Tm[:, :, reg_ind]
        # check if any p-values are below 0.05
        curr_Ps_sig_pos = np.nanmean((curr_Ps<0.05)&(curr_Ts>0), axis = 0) 
        curr_Ps_sig_neg = -np.nanmean((curr_Ps<0.05)&(curr_Ts<0), axis = 0)   
        ax_all.plot(slide_times_LM, curr_Ps_sig_pos, color=colors[reg_ind], alpha=0.7, linewidth=2, label=regressor)  # plot the proportion of significant p-values over time
        ax_all.plot(slide_times_LM, curr_Ps_sig_neg, color=colors[reg_ind], alpha=0.7, linewidth=2, linestyle='--')  # plot the proportion of significant p-values over time
        ax_all.legend(loc='upper right', fontsize=10)
        ax_all.set_xlabel(f'Time from {align_name} (s)', fontsize=12)
        ax_all.axhline(0, color='k', linestyle='--', linewidth=1)  # add a horizontal line at y=0

    reward_ind = np.where(np.array(regressors_focus) == 'outcome')[0][0]  # find the index of the reward outcome
    q_ind = np.where(np.array(regressors_focus) == 'Qchosen')[0][0]  # find the index of the Qchosen
    for time_ind, time in enumerate(slide_times_LM):
        ax = fig.add_subplot(gs[-1, time_ind])
        curr_T_reward = all_Tm[:, time_ind, reward_ind]  # get the T-statistics for the reward outcome
        curr_T_q = all_Tm[:, time_ind, q_ind]  # get the T-statistics for the Qchosen
        curr_coefs_reward = all_coefsm[:, time_ind, reward_ind]  # get the coefficients for the reward outcome
        curr_coefs_q = all_coefsm[:, time_ind, q_ind]  # get the coefficients for the Qchosen
        curr_p_reward = all_pm[:, time_ind, reward_ind]  # get the p-values for the reward outcome
        ax.scatter(curr_coefs_reward[curr_p_reward<sig_thresh], curr_coefs_q[curr_p_reward<sig_thresh], alpha=0.25, color='k', edgecolors='none', s=20)
        ax.set_xlabel(f'{time:.2f} s')
        limit = np.max([np.nanmax(np.abs(curr_coefs_reward)), np.nanmax(np.abs(curr_coefs_q))])
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        # polar

        all_vec = np.column_stack((curr_coefs_reward[curr_p_reward<sig_thresh], curr_coefs_q[curr_p_reward<sig_thresh]))  # combine the coefficients for the reward outcome and Qchosen]))

        # Convert Cartesian coordinates to polar coordinates
        theta, rho = np.arctan2(all_vec[:, 1], all_vec[:, 0]), np.hypot(all_vec[:, 1], all_vec[:, 0])

        # Define histogram edges (bins) from -π to π
        edges = np.linspace(-np.pi, np.pi, 4*4)

        # Create polar histogram
        ax = fig.add_subplot(gs[-2, time_ind], polar=True)
        ax.hist(theta, bins=edges, color=[0.1, 0.1, 0.1], alpha=0.7, edgecolor='none', density=True)
        ax.set_yticks([])

    plt.suptitle('T-statistics for all regressors', fontsize=14)

    plt.savefig(os.path.join(beh_folder, f'Regression_in_time_{criteria_name}_{align_name}.pdf'), bbox_inches='tight')


# %% [markdown]
# # Regresssion in outcome focused window

# %%
# regressors_focus = ['Qchosen', 'outcome', 'ipsi', 'outcome:ipsi', 'Intercept', 'amp_abs']
# regressors_sup = []

# sig_regressors = pd.DataFrame(columns=['session', 'unit_id']+regressors_focus+regressors_sup)
# sig_regressors['session'] = combined_tagged_units_filtered['session']
# sig_regressors['unit_id'] = combined_tagged_units_filtered['unit']
# sig_regressors[regressors_focus] = 1
# sig_regressors[regressors_sup] = 1

# %%
# regression for outcome focus window
regressors_focus = ['Qchosen', 'outcome','ipsi', 'outcome:ipsi', 'amp_abs']
regressors_sup = [] 
# regressors_focus = regressors_focus + regressors_sup
# regressors_sup = []

sig_regressors = pd.DataFrame(columns=['session', 'unit_id']+regressors_focus+regressors_sup)
sig_regressors['session'] = combined_tagged_units_filtered['session']
sig_regressors['unit_id'] = combined_tagged_units_filtered['unit']
sig_regressors[regressors_focus] = 1
sig_regressors[regressors_sup] = 0
sig_regressors['R2_final'] = 0
sig_regressors['R2_forced'] = 0
t_regressors_e = pd.DataFrame(columns=['session', 'unit_id']+regressors_focus)
t_regressors_e['session'] = combined_tagged_units_filtered['session']
t_regressors_e['unit_id'] = combined_tagged_units_filtered['unit']
t_regressors_e_mc = t_regressors_e.copy()
t_regressors_l = t_regressors_e.copy()
t_regressors_l_mc = t_regressors_e.copy()
t_regressors_com = t_regressors_e.copy()
t_regressors_com_mc = t_regressors_e.copy()
t_regressors_indi = t_regressors_e.copy()
t_regressors_indi_mc = t_regressors_e.copy()


p_regressors_e = t_regressors_e.copy()
p_regressors_e_mc = t_regressors_e.copy()
p_regressors_l = t_regressors_e.copy()
p_regressors_l_mc = t_regressors_e.copy()
p_regressors_com = t_regressors_e.copy()
p_regressors_com_mc = t_regressors_e.copy()
p_regressors_indi = t_regressors_e.copy()
p_regressors_indi_mc = t_regressors_e.copy()

coeff_regressors_e = t_regressors_e.copy()
coeff_regressors_e_mc = t_regressors_e.copy()
coeff_regressors_l = t_regressors_e.copy()
coeff_regressors_l_mc = t_regressors_e.copy()
coeff_regressors_com = t_regressors_e.copy()
coeff_regressors_com_mc = t_regressors_e.copy()
coeff_regressors_indi = t_regressors_e.copy()
coeff_regressors_indi_mc = t_regressors_e.copy()

all_coefs = []
all_T = []
all_p = []
data_type = 'curated'
align_name = 'response'
focus_label = 'outcome'
model_name = 'stan_qLearning_5params'
binSize = 1.5
all_regressors = regressors_focus + regressors_sup
formula = regressors_to_formula('spikes', all_regressors)
all_outcome_e = []
all_outcome_l = []
all_outcome_T_l = []
all_outcome_T_e = []
max_time = []
max_T = []
loaded_session = None

for ind, row in combined_tagged_units_filtered.iterrows():
    session = row['session']
    unit_id = row['unit']
    # check if different session
    if loaded_session is None or loaded_session != session:
        session_dir = session_dirs(session)
        unit_tbl = get_unit_tbl(session, data_type)
        session_df = makeSessionDF(session, model_name = model_name)
        session_df['ipsi'] = 2*(session_df['choice'].values - 0.5) * row['rec_side']
        drift_data = load_trial_drift(session, data_type)
        loaded_session = session
        # load window from file
        if session_dir['aniID'].startswith('ZS'):
            window_file = '/root/capsule/code/beh_ephys_analysis/session_combine/metrics/beh_all_TT/auc_windows.json'
            indi_window_file = '/root/capsule/code/beh_ephys_analysis/session_combine/metrics/beh_all_TT/auc_max_lag_indi.csv'
        else:
            window_file = '/root/capsule/code/beh_ephys_analysis/session_combine/metrics/beh_all_NP/auc_windows.json'
            indi_window_file = '/root/capsule/code/beh_ephys_analysis/session_combine/metrics/beh_all_NP/auc_max_lag_indi.csv'
        with open(window_file, "r") as f:
            window_dict = json.load(f)
        indi_window_df = pd.read_csv(indi_window_file)
    # individualize window based on indi_window_df
    indi_time = indi_window_df[(indi_window_df['session'] == session) & (indi_window_df['unit'] == unit_id)]['max_auc_lag']
    if len(indi_time) == 0:
        indi_time = 1.0
        print(f'No individual window found for session {session}, unit {unit_id}, using default 1.0s')
    else:
        indi_time = indi_time.values[0]
    unit_drift = load_drift(session, unit_id, data_type=data_type)
    spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
    session_df_curr = session_df.copy()
    spike_times_curr = spike_times.copy()
    unit_trial_drift_curr = drift_data.load_unit(unit_id)
    # tblTrials_curr = tblTrials.copy()
    if unit_drift is not None:
        if unit_drift['ephys_cut'][0] is not None:
            spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
            session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
            # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
        if unit_drift['ephys_cut'][1] is not None:
            spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
            session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
            # tblTrials_curr = tblTrials_curr[tblTrials_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
    if 'amp_abs' in all_regressors or 'amp' in all_regressors:
        # get unit_trial_drift_curr's rows corresponding to the ones in session_df_curr
        session_df_curr = session_df_curr.merge(unit_trial_drift_curr, on='trial_ind', how='left').copy()
    for interaction in all_regressors:
        if ':' in interaction:
            reg1, reg2 = interaction.split(':')
            session_df_curr[interaction] = session_df_curr[reg1] * session_df_curr[reg2]

    align_time = session_df_curr['outcome_time'].values
    align_time_cue = session_df_curr['go_cue_time'].values
    baseline_df = align.to_events(spike_times_curr, align_time_cue, [-binSize, 0], return_df=True)
    counts_bl = baseline_df.groupby('event_index').size()
    counts_bl = [counts_bl.get(i, 0) for i in range(len(session_df_curr))]

    # positive window
    align_time_max = window_dict['early']

    spike_df = align.to_events(spike_times_curr, align_time + align_time_max, [-0.5*binSize, 0.5*binSize], return_df=True)
    counts = spike_df.groupby('event_index').size()
    counts = [counts.get(i, 0) for i in range(len(session_df_curr))]
    # counts = np.array(counts) - np.array(counts_bl)
    spike_matrix_LM = np.reshape(np.array([counts]), (-1, 1))
    spike_matrix_LM = zscore(spike_matrix_LM, axis=0)  
        
    # try:
    model_ori_e, model_final_e, included_regressors_e , R2_final_curr_e, R2_forced_curr_e = stepwise_glm(session_df_curr, spike_matrix_LM.flatten(), regressors_focus, regressors_sup, verbose=False, criterion='bic')

    # negative window
    align_time_max = window_dict['late']
    spike_df = align.to_events(spike_times_curr, align_time + align_time_max, [-0.5*binSize, 0.5*binSize], return_df=True)
    counts = spike_df.groupby('event_index').size()
    counts = [counts.get(i, 0) for i in range(len(session_df_curr))]
    # counts = np.array(counts) - np.array(counts_bl)
    spike_matrix_LM = np.reshape(np.array([counts]), (-1, 1))
    spike_matrix_LM = zscore(spike_matrix_LM, axis=0)

    # try:
    model_ori_l, model_final_l, included_regressors_l, R2_final_curr_l, R2_forced_curr_l = stepwise_glm(session_df_curr, spike_matrix_LM.flatten(), regressors_focus, regressors_sup, verbose=False, criterion='bic')

    # pick a model with T-statistics bigger for outcome, positive or negative
    if np.abs(model_ori_e.tvalues['outcome']) > np.abs(model_ori_l.tvalues['outcome']):
        model_ori = model_ori_e
        model_final = model_final_e
        included_regressors = included_regressors_e
        R2_final_curr = R2_final_curr_e
        R2_forced_curr = R2_forced_curr_e
        max_time.append(window_dict['early'])
        max_T.append(model_ori_e.tvalues['outcome'])
    else:
        model_ori = model_ori_l
        model_final = model_final_l 
        included_regressors = included_regressors_l
        R2_final_curr = R2_final_curr_l
        R2_forced_curr = R2_forced_curr_l
        max_time.append(window_dict['late'])
        max_T.append(model_ori_l.tvalues['outcome'])

    # individual window
    align_time_max = indi_time
    spike_df = align.to_events(spike_times_curr, align_time + align_time_max, [-0.5*binSize, 0.5*binSize], return_df=True)
    counts = spike_df.groupby('event_index').size()
    counts = [counts.get(i, 0) for i in range(len(session_df_curr))]
    spike_matrix_LM = np.reshape(np.array([counts]), (-1, 1))
    spike_matrix_LM = zscore(spike_matrix_LM, axis=0)
    model_ori_ind, model_final_ind, included_regressors_ind , R2_final_curr_ind, R2_forced_curr_ind = stepwise_glm(session_df_curr, spike_matrix_LM.flatten(), regressors_focus, regressors_sup, verbose=False, criterion='bic')

    all_outcome_l.append(model_ori_l.params['outcome'])
    all_outcome_e.append(model_ori_e.params['outcome'])
    all_outcome_T_l.append(model_ori_l.tvalues['outcome'])
    all_outcome_T_e.append(model_ori_e.tvalues['outcome'])


    sig_regressors.loc[ind, included_regressors] = 1 
    sig_regressors.loc[ind, 'R2_final'] = R2_final_curr
    sig_regressors.loc[ind, 'R2_forced'] = R2_forced_curr
    
    for regressor in regressors_focus:
        t_regressors_e.loc[ind, regressor] = model_ori_e.tvalues[regressor]
        t_regressors_e_mc.loc[ind, regressor] = model_final_e.tvalues[regressor]
        t_regressors_l.loc[ind, regressor] = model_ori_l.tvalues[regressor]
        t_regressors_l_mc.loc[ind, regressor] = model_final_l.tvalues[regressor]
        t_regressors_com.loc[ind, regressor] = model_ori.tvalues[regressor]
        t_regressors_com_mc.loc[ind, regressor] = model_final.tvalues[regressor]
        t_regressors_indi.loc[ind, regressor] = model_ori_ind.tvalues[regressor]
        t_regressors_indi_mc.loc[ind, regressor] = model_final_ind.tvalues[regressor]


        p_regressors_e.loc[ind, regressor] = model_ori_e.pvalues[regressor]
        p_regressors_e_mc.loc[ind, regressor] = model_final_e.pvalues[regressor]
        p_regressors_l.loc[ind, regressor] = model_ori_l.pvalues[regressor]
        p_regressors_l_mc.loc[ind, regressor] = model_final_l.pvalues[regressor]
        p_regressors_com.loc[ind, regressor] = model_ori.pvalues[regressor]
        p_regressors_com_mc.loc[ind, regressor] = model_final.pvalues[regressor]
        p_regressors_indi.loc[ind, regressor] = model_ori_ind.pvalues[regressor]
        p_regressors_indi_mc.loc[ind, regressor] = model_final_ind.pvalues[regressor]


        coeff_regressors_e.loc[ind, regressor] = model_ori_e.params[regressor]
        coeff_regressors_e_mc.loc[ind, regressor] = model_final_e.params[regressor]
        coeff_regressors_l.loc[ind, regressor] = model_ori_l.params[regressor]
        coeff_regressors_l_mc.loc[ind, regressor] = model_final_l.params[regressor]
        coeff_regressors_com.loc[ind, regressor] = model_ori.params[regressor]
        coeff_regressors_com_mc.loc[ind, regressor] = model_final.params[regressor]
        coeff_regressors_indi.loc[ind, regressor] = model_ori_ind.params[regressor]
        coeff_regressors_indi_mc.loc[ind, regressor] = model_final_ind.params[regressor]


# %%
fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharex=False, sharey=False)
axes[0].scatter(all_outcome_l, all_outcome_e, alpha=0.5, color='k', edgecolors='none', s=20)
axes[0].axhline(0, color='k', linestyle='--', linewidth=1)
axes[0].axvline(0, color='k', linestyle='--', linewidth=1)
axes[0].set_xlabel('Late')
axes[0].set_ylabel('Early')

axes[1].hist(all_outcome_e, bins=20, color='k', alpha=0.7, edgecolor='none')
axes[1].hist(all_outcome_l, bins=20, color='gray', alpha=0.7, edgecolor='none')
axes[1].set_xlabel('Outcome')
axes[1].set_ylabel('Count')
axes[1].legend(['Early', 'Late']) 

# axes[2].scatter(auc_max[:, labels.index('outcome')]-0.5, np.abs(all_outcome_e) - np.abs(all_outcome_l), alpha=0.5, color='k', edgecolors='none', s=20)
# axes[2].legend(['T(Early-Late)'])
# axes[2].set_xlabel('Max(AUC) - 0.5')

# axes[3].hist(np.array(max_T)[np.array(max_time)==mode_p], bins=20, color='k', alpha=0.7, edgecolor='none', density=True)
# axes[3].hist(np.array(max_T)[np.array(max_time)==mode_n], bins=20, color='gray', alpha=0.7, edgecolor='none', density=True)
# axes[3].set_xlabel('Max_T')
# axes[3].set_ylabel('Count')
# axes[3].legend(['Early', 'Late'])
fig.suptitle('Outcome regressor comparison between early and late windows')
fig.savefig(os.path.join(beh_folder, f'Outcome_regressor_comparison_{criteria_name}.pdf'), bbox_inches='tight')

# %%
# plot bar plot for each column of sig_regressors
sig_regressors_plot = sig_regressors.drop(columns=['session', 'unit_id'])
sig_regressors_plot = sig_regressors_plot.astype(int)
sig_regressors_plot = sig_regressors_plot.sum(axis=0)
sig_regressors_plot = sig_regressors_plot[sig_regressors_plot > 0]
fig, ax = plt.subplots(figsize=(10, 6))
sig_regressors_plot.plot(kind='bar', ax=ax)
ax.set_ylabel('Number of units')
# change xaxis labels to 45 degree
plt.xticks(rotation=45, ha='right')
fig.savefig(os.path.join(beh_folder, f'Number_of_units_per_regressor_{criteria_name}.pdf'), bbox_inches='tight')

# %%
# Regressors to compare
regressors_to_cross = regressors_sup
sig_regressors_cross = sig_regressors[regressors_to_cross].copy()

# Cross-count matrix
cross_matrix = np.zeros((len(regressors_to_cross), len(regressors_to_cross)), dtype=int)
for ind_x, regressor1 in enumerate(regressors_to_cross):
    for ind_y, regressor2 in enumerate(regressors_to_cross):
        if ind_x == ind_y:
            continue
        cross_matrix[ind_x, ind_y] = len(sig_regressors_cross[
            (sig_regressors_cross[regressor1] == 1) &
            (sig_regressors_cross[regressor2] == 1)
        ])

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cross_matrix, cmap='Blues')

# Add number labels in each cell
for i in range(len(regressors_to_cross)):
    for j in range(len(regressors_to_cross)):
        if i != j:
            ax.text(j, i, f'{cross_matrix[i, j]/np.sum(sig_regressors[regressors_to_cross[i]] == 1):.2f}', va='center', ha='center', color='black')
        else:
            ax.text(j, i, f'{np.sum(sig_regressors[regressors_to_cross[i]] == 1)}', va='center', ha='center', color='black')

# Axis labels
ax.set_xticks(np.arange(len(regressors_to_cross)))
ax.set_yticks(np.arange(len(regressors_to_cross)))
ax.set_xticklabels(regressors_to_cross)
ax.set_yticklabels(regressors_to_cross)
ax.set_title('Number of Units with Significant Overlap')
plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
fig.savefig(os.path.join(beh_folder, f'Cross_count_matrix_{criteria_name}_motion.pdf'), bbox_inches='tight')




# %%
figures, ax = plt.subplots(figsize=(8, 6))
plt.subplot(121)
plt.scatter(sig_regressors['R2_final'], sig_regressors['R2_forced'], alpha=0.5, color='k', edgecolors='none', s=20)
plt.xlabel('R2_final')
plt.ylabel('R2_forced')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
title = f'{np.sum(sig_regressors["R2_final"] > sig_regressors["R2_forced"])/len(sig_regressors):.2f} units with motion correction'
plt.title(title)

plt.subplot(122)
bins = np.arange(0, 0.7, 0.05)
plt.hist(sig_regressors['R2_final']-sig_regressors['R2_forced'], bins=bins, alpha=0.5, color='k', edgecolor='none')
plt.xlabel('R2_final-forced')

plt.xlim(0, 0.7)
plt.savefig(os.path.join(beh_folder, f'R2_comparison_{criteria_name}_motion.pdf'), bbox_inches='tight')
# %%
# sig_regressors_abs = pd.read_csv(os.path.join(beh_folder, f'sig_regressors_{criteria_name}_abs.csv'))
# sig_regressors['R2_final_abs'] = sig_regressors_abs['R2_final']

# %%
# plt.subplot(121)
# plt.scatter(sig_regressors['R2_final_abs'], sig_regressors['R2_final'], alpha=0.5, color='k', edgecolors='none', s=20)
# plt.xlabel('R2_final_abs')
# plt.ylabel('R2_final')
# plt.plot([0, 1], [0, 1], color='r', linestyle='--')
# title = f'{np.sum(sig_regressors["R2_final_abs"] > sig_regressors["R2_final"])/len(sig_regressors):.2f} units improved by abs'
# plt.title(title)

# plt.subplot(122)
# bins = np.arange(0, 0.7, 0.02)
# plt.hist(sig_regressors['R2_final_abs']-sig_regressors['R2_final'], bins=bins, alpha=0.5, color='k', edgecolor='none')
# plt.xlabel('R2_final_abs-final')
# plt.xlim(0, 0.7)

# %%
# compare t-values
fig, axes = plt.subplots(1, len(regressors_focus), figsize=(15, 5))
for ind, regressor in enumerate(regressors_focus):
    axes[ind].scatter(t_regressors_com[regressor], t_regressors_com_mc[regressor], alpha=0.5, color='k', edgecolors='none', s=20)
    axes[ind].plot([-10, 10], [-10, 10], color='r', linestyle='--')
    axes[ind].axhline(0, color='k', linestyle='--', linewidth=1)  # add a horizontal line at y=0
    axes[ind].axvline(0, color='k', linestyle='--', linewidth=1)  # add a vertical line at x=0
    axes[ind].set_xlabel('No motion')
    axes[ind].set_ylabel('Motion')
    axes[ind].set_title(f'{regressor}')
plt.tight_layout()
fig.savefig(os.path.join(beh_folder, f'Tvalue_comparison_motion_{criteria_name}.pdf'), bbox_inches='tight')

# %%
# without motion correction
all_t_regressors_ori = merge_df_with_suffix([t_regressors_e.copy(), t_regressors_l.copy(), t_regressors_com.copy(), t_regressors_indi.copy()], suffixes=('e', 'l', 'com', 'indi'), on_list=['session', 'unit_id'])
all_coeff_regressors_ori = merge_df_with_suffix([coeff_regressors_e.copy(), coeff_regressors_l.copy(), coeff_regressors_com.copy(), coeff_regressors_indi.copy()], suffixes=('e', 'l', 'com', 'indi'), on_list=['session', 'unit_id'])
all_p_regressors_ori = merge_df_with_suffix([p_regressors_e.copy(), p_regressors_l.copy(), p_regressors_com.copy(), p_regressors_indi.copy()], suffixes=('e', 'l', 'com', 'indi'), on_list=['session', 'unit_id'])

# with motion correction
all_t_regressors_mc = merge_df_with_suffix([t_regressors_e_mc.copy(), t_regressors_l_mc.copy(), t_regressors_com_mc.copy(), t_regressors_indi_mc.copy()], suffixes=('e', 'l', 'com', 'indi'), on_list=['session', 'unit_id'])
all_coeff_regressors_mc = merge_df_with_suffix([coeff_regressors_e_mc.copy(), coeff_regressors_l_mc.copy(), coeff_regressors_com_mc.copy(), coeff_regressors_indi_mc.copy()], suffixes=('e', 'l', 'com', 'indi'), on_list=['session', 'unit_id'])
all_p_regressors_mc = merge_df_with_suffix([p_regressors_e_mc.copy(), p_regressors_l_mc.copy(), p_regressors_com_mc.copy(), p_regressors_indi_mc.copy()], suffixes=('e', 'l', 'com', 'indi'), on_list=['session', 'unit_id'])
# combine all
all_t_regressors = merge_df_with_suffix([all_t_regressors_ori.copy(), all_t_regressors_mc.copy()], suffixes=('ori', 'mc'), on_list=['session', 'unit_id'])
all_coeff_regressors = merge_df_with_suffix([all_coeff_regressors_ori.copy(), all_coeff_regressors_mc.copy()], suffixes=('ori', 'mc'), on_list=['session', 'unit_id'])
all_p_regressors = merge_df_with_suffix([all_p_regressors_ori.copy(), all_p_regressors_mc.copy()], suffixes=('ori', 'mc'), on_list=['session', 'unit_id'])

# %%
# compare regressors in early and late
fig, axes = plt.subplots(1, len(regressors_focus), figsize=(15, 5))
probes = combined_tagged_units_filtered['probe']
if probes.dtype == 'O' or not np.issubdtype(probes.dtype, np.number):
    # Convert string labels to integers for coloring
    _, cvals = np.unique(probes, return_inverse=True)
else:
    cvals = probes.values

for ind_reg, regressor in enumerate(regressors_focus):
    axes[ind_reg].scatter(t_regressors_e[regressor][probes.values == '2'], t_regressors_l[regressor][probes.values == '2'], alpha=0.5, edgecolors='none', s=20, label = '2')
    axes[ind_reg].scatter(t_regressors_e[regressor][probes.values == 'tt'], t_regressors_l[regressor][probes.values == 'tt'], alpha=0.5, edgecolors='none', s=20, label = 'tt')
    axes[ind_reg].plot([-10, 10], [-10, 10], color='r', linestyle='--')
    axes[ind_reg].set_xlabel('Early')
    axes[ind_reg].set_ylabel('Late')
    axes[ind_reg].set_title(f'{regressor}')
    axes[ind_reg].axhline(0, color='k', linestyle='--', linewidth=1)  # add a horizontal line at y=0
    axes[ind_reg].axvline(0, color='k', linestyle='--', linewidth=1)
    if ind_reg==0:
        axes[ind_reg].legend()

plt.tight_layout()
fig.savefig(os.path.join(beh_folder, f'Tvalue_comparison_early_late_{criteria_name}.pdf'), bbox_inches='tight')

# %%
# scatter and polar of 'outcome' and 'Qchosen'
period = 'com'
verion = 'mc'  # 'ori' or 'mc'
if 'outcome:ipsi' in regressors_focus:
    curr_p_int = all_p_regressors[f'outcome:ipsi_{period}_{verion}'].values
    curr_coefs_int = all_coeff_regressors[f'outcome:ipsi_{period}_{verion}'].values 
else:
    curr_coefs_int = np.zeros(all_coefsm.shape[0])
    curr_p_int = np.ones(all_pm.shape[0])

curr_coefs_outcome = all_coeff_regressors[f'outcome_{period}_{verion}'].values
curr_coefs_q = all_coeff_regressors[f'Qchosen_{period}_{verion}'].values

curr_T_outcome = all_t_regressors[f'outcome_{period}_{verion}'].values  # get the T-statistics for the reward outcome
curr_T_q = all_t_regressors[f'Qchosen_{period}_{verion}'].values  # get the T-statistics for the Qchosen
curr_p_outcome = all_p_regressors[f'outcome_{period}_{verion}'].values  # get the p-values for the reward outcome

# curr_T_outcome = all_Tm[:, outcome_ind]  # get the T-statistics for the reward outcome
# curr_T_q = all_Tm[:, q_ind]  # get the T-statistics for the Qchosen
# curr_p_outcome = all_pm[:, outcome_ind]  # get the p-values for the reward outcome


# %%
all_vec = np.column_stack((curr_coefs_outcome, curr_coefs_q))  # combine the coefficients for the reward outcome and Qchosen
all_vec = all_vec.astype(float)
# Convert Cartesian coordinates to polar coordinates
theta, rho = np.arctan2(all_vec[:, 1], all_vec[:, 0]), np.hypot(all_vec[:, 1], all_vec[:, 0])
bound_1 = -(1/4)*np.pi
bound_2 = np.pi
bound_3 = -np.pi
theta_scaled_dis  = np.full(len(theta), np.nan)
for ind, angle_curr in enumerate(theta):
    if bound_1 < angle_curr <= bound_2:
        theta_scaled_dis[ind] = (angle_curr-bound_1)/(bound_2-bound_1)
    else:
        theta_scaled_dis[ind] = (bound_1-angle_curr)/(bound_1-bound_3)
theta_scaled_dis_all = 1-theta_scaled_dis


# %%
sc = plt.scatter(
    curr_coefs_outcome,
    curr_coefs_q,
    c=theta_scaled_dis_all,
    cmap="Reds",
    edgecolors="k",   # keep black edges
    s=30,
    marker="o"        # filled marker to avoid warning
)

plt.colorbar(sc, label="theta_scaled_dis_all")
plt.xlabel("curr_coefs_outcome")
plt.ylabel("curr_coefs_q")
plt.savefig(os.path.join(beh_folder, f'Coefficient_scatter_theta_{criteria_name}.pdf'), bbox_inches='tight')


# %%
# make a df from all_Tm, columns correspond to focus regressors 
model_combined = merge_df_with_suffix([all_t_regressors.copy(), all_p_regressors.copy(), all_coeff_regressors.copy()], prefixes=('T', 'p', 'coef'), on_list=['session', 'unit_id'])
model_combined['theta'] = theta_scaled_dis_all
model_combined['rho'] = rho
model_combined['t_outcome|(|t_outcome| + |t_Q|)'] = model_combined['T_outcome_com_mc'].values/(np.abs(model_combined['T_outcome_com_mc'].values) + np.abs(model_combined['T_Qchosen_com_mc'].values))

# %%
model_combined.to_csv(os.path.join(beh_folder, f'model_combined_{criteria_name}.csv'), index=True)
print(f'Combined model saved to {os.path.join(beh_folder, f"model_combined_{criteria_name}.csv")}')