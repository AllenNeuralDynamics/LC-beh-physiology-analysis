import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
import json
import seaborn as sns
from PyPDF2 import PdfMerger
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb, plot_session_glm
from utils.ephys_functions import*
from utils.ccf_utils import ccf_pts_convert_to_mm
import pickle
import scipy.stats as stats
import spikeinterface as si
import shutil
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import r2_score
from matplotlib import cm
import matplotlib.colors as mcolors
from joblib import Parallel, delayed
from utils.combine_tools import apply_qc
from scipy.stats import gaussian_kde

def kde_peak(x, bw_method='scott', step_size = 0.01):
    """Return the peak (mode) estimated from a KDE."""
    x = x[np.isfinite(x)]  # remove NaNs
    if len(x) < 2:
        return np.nan
    kde = gaussian_kde(x, bw_method=bw_method)
    xs = np.arange(np.min(x), np.max(x), step_size)
    ys = kde(xs)
    return xs[np.argmax(ys)]

def cal_beh_metrics(session, model_name='stan_qLearning_5params'):
    """
    Calculate behavioral metrics for a given session and model.
    
    Parameters:
    session (str): The session identifier.
    model_name (str): The name of the model to use for calculations.
    
    Returns:
    None
    """
    
    # Load the session directory
    session_dir = session_dirs(session, model_name=model_name)
    # choice only 
    session_df = makeSessionDF(session, model_name = model_name, cut_interruptions=True)
    # regression_coeffs
    session_df_cue = get_session_tbl(session, cut_interruptions=True)
    session_df_cue_cut = session_df_cue[(session_df_cue['goCue_start_time']>=session_df['goCue_start_time'].min()) & (session_df_cue['goCue_start_time']<=session_df['goCue_start_time'].max())].copy()
    _, session, coeff_dict = plot_session_glm(session, tMax=5, model_name= model_name)
    # check if ci is too big (>1000)
    if np.abs(coeff_dict['ci-bands_reward'][0][1]-coeff_dict['ci-bands_reward'][0][0]) > 1000:
        _, session, coeff_dict = plot_session_glm(session, tMax=1, model_name= model_name)
    # fig.savefig('/root/capsule/test.png')
    plt.close('all')
    coeff_dict['diff_1'] = coeff_dict['coeff_reward'][0] - coeff_dict['coeff_no-reward'][0]
    coeff_dict['nrwd_1'] = coeff_dict['coeff_no-reward'][0]
    coeff_dict['rwd_1'] = coeff_dict['coeff_reward'][0]

    # loss switch, finish rate
    pSwL = np.sum((session_df['outcome_prev']==0) & (session_df['svs']==1))/np.sum(session_df['outcome_prev']==0)
    pSwL_L = np.sum((session_df['outcome_prev']==0) & (session_df['svs']==1) & (session_df['choices_prev']==0))/np.sum((session_df['outcome_prev']==0) & (session_df['choices_prev']==0))
    pSwL_R = np.sum((session_df['outcome_prev']==0) & (session_df['svs']==1) & (session_df['choices_prev']==1))/np.sum((session_df['outcome_prev']==0) & (session_df['choices_prev']==1))
    pStW = np.sum((session_df['outcome_prev']==1) & (session_df['svs']==0))/np.sum(session_df['outcome_prev']==1)
    pStW_L = np.sum((session_df['outcome_prev']==1) & (session_df['svs']==0) & (session_df['choices_prev']==0))/np.sum((session_df['outcome_prev']==1) & (session_df['choices_prev']==0))
    pStW_R = np.sum((session_df['outcome_prev']==1) & (session_df['svs']==0) & (session_df['choices_prev']==1))/np.sum((session_df['outcome_prev']==1) & (session_df['choices_prev']==1))
    
    pSw = np.sum(session_df['svs']==1)/len(session_df)
    pResp = np.sum(session_df_cue_cut['animal_response']!=2)/len(session_df_cue_cut)
    ls_dict = {'p_sw': pSw, 'p_sw_L': pSwL, 'p_st_w': pStW, 'finish_rate': pResp, 
               'p_sw_L_L': pSwL_L, 'p_sw_L_R': pSwL_R, 
               'p_st_w_L': pStW_L, 'p_st_w_R': pStW_R,
               'session_len': len(session_df)}

    # params
    ani_params_cvs = os.path.join(session_dir['model_dir'], 'params_session_sample.csv')
    ani_params = pd.read_csv(ani_params_cvs)
    if np.sum(ani_params['session_id'] == session) == 0:
        session_params = ani_params[ani_params['session_id'] == session_dir['raw_id']].reset_index(drop=True).to_dict(orient='records')[0]
    else:
        session_params = ani_params[ani_params['session_id'] == session].reset_index(drop=True).to_dict(orient='records')[0]
    session_params.pop('Unnamed: 0', None);  # Remove unnamed column if it exists

    # lick metrics
    lick_lats = session_df['reward_outcome_time'].values - session_df['goCue_start_time'].values
    mean_L = np.nanmean(lick_lats[session_df['animal_response']==0])
    mean_R = np.nanmean(lick_lats[session_df['animal_response']==1])
    mode_L = kde_peak(lick_lats[session_df['animal_response']==0])
    mode_R = kde_peak(lick_lats[session_df['animal_response']==1])
    var_L = np.nanvar(lick_lats[session_df['animal_response']==0])
    var_R = np.nanvar(lick_lats[session_df['animal_response']==1])
    var_L_mode = np.mean(np.square(lick_lats[session_df['animal_response'] == 0] - mode_L))
    var_R_mode = np.mean(np.square(lick_lats[session_df['animal_response'] == 1] - mode_R))

    lick_lat_diff = (mean_R - mean_L)/(mean_L+mean_R)
    lick_lat_diff_mode = (mode_R - mode_L)/(mode_L+mode_R)
    var_lat_diff = np.log(var_R + 0.0000001) - np.log(var_L + 0.0000001)
    var_lat_diff_mode = np.log(var_R_mode + 0.0000001) - np.log(var_L_mode + 0.0000001)
    
    # Combine all dictionaries into one
    session_params.update(ls_dict)
    session_params.update(coeff_dict)
    session_params.update({'model_name': model_name, 'session_id': session})
    session_params.update({'mean_lick_lat_L': mean_L, 'mean_lick_lat_R': mean_R,
                            'mode_lick_lat_L': mode_L, 'mode_lick_lat_R': mode_R,
                           'var_lick_lat_L': var_L, 'var_lick_lat_R': var_R,
                            'var_lick_lat_L_mode': var_L_mode, 'var_lick_lat_R_mode': var_R_mode,
                           'lick_lat_diff': lick_lat_diff, 'var_lat_diff': var_lat_diff,
                           'lick_lat_diff_mode': lick_lat_diff_mode, 'var_lat_diff_mode': var_lat_diff_mode})

    # save to .json file
    json_file_path = os.path.join(session_dir['beh_fig_dir'], f'{session}_beh_metrics.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(session_params, json_file, indent=4)


if __name__ == "__main__":
    dfs = [pd.read_csv('/root/capsule/code/data_management/session_assets.csv'),
        pd.read_csv('/root/capsule/code/data_management/hopkins_session_assets.csv'),
        pd.read_csv('/root/capsule/code/data_management/hopkins_FP_session_assets.csv')]
    df = pd.concat(dfs)
    session_ids = df['session_id'].values
    session_ids = [session_id for session_id in session_ids if isinstance(session_id, str)]  # filter only behavior sessions
    Parallel(n_jobs=12)(delayed(process_session)(session) for session in session_ids)
