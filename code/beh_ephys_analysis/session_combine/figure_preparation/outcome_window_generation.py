# infer time window for outcome encoding based on sliding window auROC
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
from utils.ephys_functions import*
from utils.ccf_utils import ccf_pts_convert_to_mm
from utils.combine_tools import apply_qc, to_str_intlike
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
warnings.filterwarnings('ignore')


def compute_outcome_window(criteria_name, pre_event,post_event):
    # %%
    version = 'PrL_S1'
    capsule_dirs = capsule_directories()
    metrics_folder = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'outcome_window', criteria_name)
    os.makedirs(metrics_folder, exist_ok=True)
    

    # %%
    # load constraints and data
    with open(os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
        combined_tagged_units = pickle.load(f)
    with open(os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'combined_session_tbl', 'combined_beh_sessions.pkl'), 'rb') as f:
        combined_session_qc = pickle.load(f)
    # load antidromic tagging info
    # antidromic data
    antidromic_file = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'antidromic_analysis', version, 'combined_antidromic_results.pkl')
    with open(antidromic_file, 'rb') as f:
        antidromic_df = pickle.load(f)
    # antidromic_df['unit_id'] = antidromic_df['unit_id'].apply(to_str_intlike)
    antidromic_df = antidromic_df[['unit', 'session', 'p_auto_inhi', 't_auto_inhi',
        'p_collision', 't_collision', 'p_antidromic', 't_antidromic', 'tier_1',
        'tier_2', 'tier_1_long', 'tier_2_long']].copy()
    combined_tagged_units = combined_tagged_units.merge(antidromic_df, on=['session', 'unit'], how='left')
    combined_tagged_units['tier_1'].fillna(False, inplace=True)
    combined_tagged_units['tier_2'].fillna(False, inplace=True)
    combined_tagged_units['tier_1_long'].fillna(False, inplace=True)
    combined_tagged_units['tier_2_long'].fillna(False, inplace=True)

    combined_tagged_units.drop(columns=['probe'], inplace=True, errors='ignore')
    combined_tagged_units = combined_tagged_units.merge(combined_session_qc, on='session', how='left')
        
    with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json'), 'r') as f:
        constraints = json.load(f)

    # start with a mask of all True
    mask = pd.Series(True, index=combined_tagged_units.index)

    # %%
    combined_tagged_units_filtered, combined_tagged_units, fig, axes = apply_qc(combined_tagged_units, constraints)

    # %%
    # perform auc on sliding window and define best window for each unit

    binSize = 1.5
    stepSize = 0.1
    auc_mat = []           
    auc_max = []
    auc_max_ind = []
    labels = ['outcome', 'hit', 'svs']
    align = 'go_cue_time'
    loaded_session = None
    curr_session = None
    data_type = 'raw'
    model_name = 'stan_qLearning_5params'
    for ind, row in combined_tagged_units_filtered.iterrows():
        session = row['session']                           
        unit_id = row['unit'] 
        print(session)
        if loaded_session is None or loaded_session != session:
            session_dir = session_dirs(session)
            unit_tbl = get_unit_tbl(session, data_type)
            whole_session_tbl = get_session_tbl(session)
            whole_session_tbl['hit'] = whole_session_tbl['animal_response'].values==1
            session_df = makeSessionDF(session, model_name = model_name)
            session_df['ipsi'] = 2*(session_df['choice'].values - 0.5) * row['rec_side']
            drift_data = load_trial_drift(session, data_type)
            loaded_session = session
        unit_drift = load_drift(session, unit_id, data_type=data_type)  
        spike_times = unit_tbl.query('unit_id == @unit_id')['spike_times'].values[0]
        session_df_curr = session_df.copy()
        whole_session_df_curr = whole_session_tbl.copy()
        spike_times_curr = spike_times.copy()
        unit_trial_drift_curr = drift_data.load_unit(unit_id)
        # tblTrials_curr = tblTrials.copy()
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr >= unit_drift['ephys_cut'][0]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] >= unit_drift['ephys_cut'][0]]
                whole_session_df_curr = whole_session_df_curr[whole_session_df_curr['goCue_start_time'] >= unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                spike_times_curr = spike_times_curr[spike_times_curr <= unit_drift['ephys_cut'][1]]
                session_df_curr = session_df_curr[session_df_curr['go_cue_time'] <= unit_drift['ephys_cut'][1]]
                whole_session_df_curr = whole_session_df_curr[whole_session_df_curr['goCue_start_time'] <= unit_drift['ephys_cut'][1]]
        if len(session_df_curr) < 5:
            auc_mat.append(None)
            auc_max.append(np.full((len(labels)), np.nan))
            auc_max_ind.append(np.full((len(labels)), np.nan))
            continue
        align_time = session_df_curr[align].values
        spike_matrix_auc, slide_times_auc = get_spike_matrix(spike_times_curr, align_time, 
                                                    pre_event=pre_event, post_event=post_event, 
                                                    binSize=binSize, stepSize=stepSize, kernel=False,
                                                    tau_rise=0.001, tau_decay=0.08)
        curr_auc = np.full((len(slide_times_auc), len(labels)), np.nan, dtype=float)
        if align == 'go_cue_time':
            spike_matrix_auc_all, slide_times_auc_all = get_spike_matrix(spike_times_curr, whole_session_df_curr['goCue_start_time'].values,
                                                        pre_event=pre_event, post_event=post_event, 
                                                        binSize=binSize, stepSize=stepSize, kernel=False,
                                                        tau_rise=0.001, tau_decay=0.08)
        elif align == 'outcome_time':
            spike_matrix_auc_all, slide_times_auc_all = get_spike_matrix(spike_times_curr, whole_session_df_curr['reward_outcome_time'].values,
                                                        pre_event=pre_event, post_event=post_event, 
                                                        binSize=binSize, stepSize=stepSize, kernel=False,
                                                        tau_rise=0.001, tau_decay=0.08)
        for time_ind, time in enumerate(slide_times_auc):
            # get the spike counts in the sliding window
            spike_counts = spike_matrix_auc[:, time_ind]
            spike_counts_all = spike_matrix_auc_all[:, time_ind]
            # outcome
            for label_ind, label in enumerate(labels):
                if label != 'hit':
                    focus = session_df_curr[label].values
                    curr_auc[time_ind, label_ind] = roc_auc_score(focus, spike_counts)
                else:
                    focus = whole_session_df_curr[label].values
                    if np.sum(focus == 0) < 5:
                        curr_auc[time_ind, label_ind] = np.nan
                    else:
                        curr_auc[time_ind, label_ind] = roc_auc_score(focus, spike_counts_all)
        
        curr_max_ind = np.nanargmax(np.abs(curr_auc-0.5), axis=0)
        curr_max = curr_auc[curr_max_ind, np.arange(len(labels))]

        auc_mat.append(curr_auc)
        auc_max.append(curr_max)
        auc_max_ind.append(curr_max_ind)

    # %%
    # colormaps
    reward_colors = LinearSegmentedColormap.from_list('outcome', [(0.0, 'red'), (0.5, 'white'), (1.0, 'blue')])
    hit_colors = LinearSegmentedColormap.from_list('hit', [(0.0, 'blue'), (0.5, 'white'), (1.0, 'orange')])
    switch_colors = LinearSegmentedColormap.from_list('switch', [(0.0, 'green'), (0.5, 'white'), (1.0, 'purple')])
    feature_map = {'outcome': reward_colors, 'hit': hit_colors, 'svs': switch_colors}

    # %%
    # Define custom colormap: white at 0, blue midrange, red at 1
    custom_cmaps = LinearSegmentedColormap.from_list(
        'blue_white_red', [(0.0, 'red'), (0.5, 'white'), (1.0, 'blue')]
    )

    auc_mat = np.array(auc_mat)
    auc_max = np.array(auc_max)
    auc_max_ind = np.array(auc_max_ind)
    for label_ind, label in enumerate(labels):
        plt.figure(figsize=(10, 6))
        sort_ind = np.argsort(auc_max[:, label_ind], axis=0)
        plt.imshow(auc_mat[sort_ind, :, label_ind], aspect='auto', origin='lower', 
                extent=[slide_times_auc[0], slide_times_auc[-1], 0, len(combined_tagged_units_filtered)], 
                cmap=feature_map[label], vmin=0, vmax=1, interpolation='none')
        plt.colorbar(label='AUC')
        plt.title(f'AUC for {label} over time')
        plt.xlabel(f'Time from {align} (s)')
        plt.ylabel('Unit index')
        plt.savefig(os.path.join(metrics_folder, f'AUC_{label}_{criteria_name}_{align}.pdf'), bbox_inches='tight')

    # %%
    # Define custom colormap: white at 0, blue midrange, red at 1
    label = 'svs'
    label_ind = labels.index(label)

    plt.figure(figsize=(10, 6))
    sort_ind = np.argsort(auc_max[:, label_ind], axis=0)
    plt.imshow(auc_mat[sort_ind, :, label_ind], aspect='auto', origin='lower', 
                extent=[slide_times_auc[0], slide_times_auc[-1], 0, len(combined_tagged_units_filtered)], 
                cmap=feature_map[label], vmin=0, vmax=1, interpolation='none')
    plt.colorbar(label='AUC')
    plt.title(f'AUC for {label} over time')
    plt.xlabel(f'Time from {align} (s)')
    plt.ylabel('Unit index')
    plt.savefig(os.path.join(metrics_folder, f'AUC_{label}_{criteria_name}_{align}.pdf'), bbox_inches='tight')

    # %%
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 8), sharex=True)
    bins = np.linspace(0, 1, 40)
    for label_ind, label in enumerate(labels):
        axes[label_ind].hist(auc_max[:, label_ind], bins=bins, color='gray', alpha=0.7, edgecolor='none')
        axes[label_ind].set_title(f'{label}')
        axes[label_ind].legend()
        axes[label_ind].set_xlim(0, 1)
    plt.xlabel('AUC')
    plt.suptitle('AUC for each label')
    plt.savefig(os.path.join(metrics_folder, f'AUC_hist_{criteria_name}.pdf'), bbox_inches='tight')

    # %% [markdown]
    # # Save window for each neuron

    # %%
    max_lag_time = np.full(len(combined_tagged_units_filtered), np.nan)
    auc_max_ind_outcome = np.array(auc_max_ind)[:, labels.index('outcome')]
    valid = ~np.isnan(auc_max_ind_outcome)
    idx = auc_max_ind_outcome[valid].astype(int)
    max_lag_time[valid] = slide_times_auc[idx]
    max_lag_time[max_lag_time < 0.5*binSize] = 0.5*binSize  # set a floor to avoid log(0)

    # %%
    # save as csv with session and unit info
    auc_df = combined_tagged_units_filtered[['session', 'unit']].copy()
    auc_df['max_auc_lag'] = max_lag_time
    auc_lag_file = os.path.join(metrics_folder, f'auc_max_lag_indi.csv')
    auc_df.to_csv(auc_lag_file, index=False)

    # %%
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=False, sharey=False)
    bins = np.linspace(0, 1, 40)
    for label_ind, label in enumerate(labels):
        if label=='outcome':

            axes[0].scatter(auc_max[:, label_ind], slide_times_auc[auc_max_ind[:, labels.index(label)]])
            axes[0].set_xlim(0, 1)
            axes[0].set_xlabel('AUC')
            axes[0].set_ylabel('Time lag')
            # bins = np.linspace(np.min(slide_times_auc[auc_max_ind[:, labels.index(label)]]), np.max(slide_times_auc[auc_max_ind[:, labels.index(label)]]), 40)
            bins = 15
            positive = slide_times_auc[auc_max_ind[:, labels.index(label)]][auc_max[:, label_ind]>=0.5]+1
            negative = slide_times_auc[auc_max_ind[:, labels.index(label)]][auc_max[:, label_ind]<0.5]+1
            axes[1].hist(np.log(positive[positive>=0]), bins=bins, color='k', alpha=0.7, edgecolor='none', density=True)
            kde = gaussian_kde(np.log(positive[positive>=0]))
            x_grid = np.linspace(min(np.log(positive)), max(np.log(positive))+2, 1000)
            pdf_values = kde(x_grid)
            mode_p = x_grid[np.argmax(pdf_values)]
            axes[1].axvline(mode_p, color='k', linestyle='--', label='Mode')
            axes[1].plot(x_grid, pdf_values, color='blue', label='Estimated PDF')
            axes[1].hist(np.log(negative[negative>=0]), bins=bins, color='lightgray', alpha=0.5, edgecolor='none', density=True)
            kde = gaussian_kde(np.log(negative[negative>=0]))
            pdf_values = kde(x_grid)
            mode_n = x_grid[np.argmax(pdf_values)]
            axes[1].axvline(mode_n, color='gray', linestyle='--', label='Mode')
            axes[1].plot(x_grid, pdf_values, color='red', label='Estimated PDF')
            axes[1].set_xlim(-1, 5)
            title = f'p: {np.exp(mode_p)-1:.2f}, n: {np.exp(mode_n)-1:.2f}'
            axes[1].set_title(title)
            axes[1].set_xlabel('log(Time lag)')

            bins = 20
            positive = slide_times_auc[auc_max_ind[:, labels.index(label)]][auc_max[:, label_ind]>=0.5]
            negative = slide_times_auc[auc_max_ind[:, labels.index(label)]][auc_max[:, label_ind]<0.5]
            axes[2].hist(positive, bins=bins, color='k', alpha=0.7, edgecolor='none', density=True)
            kde = gaussian_kde(positive)
            x_grid = np.linspace(min(positive), max(positive), 1000)
            pdf_values = kde(x_grid)
            mode_p = x_grid[np.argmax(pdf_values)]
            axes[2].axvline(mode_p, color='k', linestyle='--', label='Mode')
            axes[2].plot(x_grid, pdf_values, color='blue', label='Estimated  PDF')
            axes[2].hist(negative, bins=bins, color='lightgray', alpha=0.5, edgecolor='none', density=True)
            kde = gaussian_kde(negative)
            pdf_values = kde(x_grid)
            mode_n = x_grid[np.argmax(pdf_values)]
            axes[2].axvline(mode_n, color='gray', linestyle='--', label='Mode')
            axes[2].plot(x_grid, pdf_values, color='red', label='Estimated PDF')
            axes[2].set_xlim(-1, 5)
            title = f'p: {mode_p:.2f}, n: {mode_n:.2f}'
            axes[2].set_title(title)
            axes[2].set_xlabel('Time lag')
        

    plt.suptitle('AUC for each label')
    plt.savefig(os.path.join(metrics_folder, f'AUC_hist_{criteria_name}.pdf'), bbox_inches='tight')

    # %%
    # extract auROC 
    mode_e_ind = np.argmin(np.abs(slide_times_auc-mode_p))
    mode_l_ind = np.argmin(np.abs(slide_times_auc-mode_n))
    auROC_e = auc_mat[:, mode_e_ind, labels.index('outcome')]
    auROC_l = auc_mat[:, mode_l_ind, labels.index('outcome')]
    auROC_max = np.full((auc_mat.shape[0],), np.nan, dtype=float)
    for unit_ind in range(auc_mat.shape[0]):
        if np.abs(auROC_e[unit_ind]-0.5) > np.abs(auROC_l[unit_ind]-0.5):
            auROC_max[unit_ind] = auROC_e[unit_ind]
        else:
            auROC_max[unit_ind] = auROC_l[unit_ind]

    # save window to .json
    window_dict = {'late':mode_n, 'early':mode_p}
    window_file = os.path.join(metrics_folder, 'auc_windows.json')
    with open(window_file, 'w') as f:
        json.dump(window_dict, f)

if __name__ == "__main__":
    criteria_name = 'beh_all_TT'
    post_event = 2
    pre_event = 0
    compute_outcome_window(criteria_name, pre_event, post_event)

    criteria_name = 'beh_all_NP'
    post_event = 2.5
    pre_event = 0
    compute_outcome_window(criteria_name, pre_event, post_event)


