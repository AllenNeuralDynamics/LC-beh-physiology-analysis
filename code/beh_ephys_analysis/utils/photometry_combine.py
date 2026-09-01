import numpy as np
import pandas as pd
import sys
from pathlib import Path

from requests import session
sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.ephys_functions import fitSpikeModelG
import platform
import os
from pathlib import Path
import shutil
from pathlib import Path
import shutil
# from utils.behavior.session_utils import load_session_df, parse_session_string
# from utils.behavior.lick_analysis import clean_up_licks, parse_lick_trains
# from utils.behavior.model_utils import get_param_names_dF, get_model_variables_dF, get_stan_model_params_samps_only, infer_model_var
from utils.beh_functions import session_dirs, parseSessionID, get_session_tbl, makeSessionDF, load_model_dv
from utils.photometry_utils import get_FP_data, align_signal_to_events
from itertools import chain
from matplotlib import pyplot as plt
from IPython.display import display
from scipy.signal import find_peaks
# from harp.clock import align_timestamps_to_anchor_points
# from utils.basics.data_org import *
# from utils.photometry.preprocessing import * 
# from utils.photometry.plot_utils import align_signal_to_events, color_gradient, plot_FP_with_licks, plot_G_vs_Iso, plot_FP_beh_analysis,plot_FP_beh_analysis_model
# from utils.behavior.session_utils import beh_analysis_no_plot
import numpy as np
# from scipy.signal import butter, filtfilt, medfilt, sosfiltfilt
# from scipy.optimize import curve_fit
# import json
from sklearn.linear_model import LinearRegression
from matplotlib.gridspec import GridSpec
import pickle
# from aind_fip_dff.utils.preprocess import batch_processing, tc_triexpfit
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time
from contextlib import redirect_stdout
# %matplotlib inline
# %matplotlib widget
import re
import random
from joblib import Parallel, delayed
import os
from scipy.stats import zscore
from scipy.stats import mode
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import json

def process_session(session, region, channel='G_tri-exp_mc', formula='spikes ~ 1 + outcome + choice + Qchosen', align='CSon',
        window_size = 0.5, thresh = 0.5, pre_time = 2, post_time = 3, step_size=0.1):
    session_dir = session_dirs(session)
    model = 'stan_qLearning_5params'
    # load animal metadata
    ani_meta_file = os.path.join('/root/capsule/code/data_management/FP_metadata', f'{session_dir["aniID"]}.json')
    if not os.path.exists(ani_meta_file):
        print(f'Animal metadata file not found for session {session}.')
        return None, None, None, None, session
    with open(ani_meta_file, 'r') as f:
        ani_meta = json.load(f)
    if region not in ani_meta.keys():
        print(f'{session} Region not found in surgery info.')
        implant_side = 1
    elif ani_meta[region] == 'L':
        implant_side = -1
    elif ani_meta[region] == 'R':
        implant_side = 1
    else:
        print('Invalid surgery info.')
        return None, None, None, None, session
    beh_session_data = get_session_tbl(session)
    choices = np.full(len(beh_session_data), np.nan)
    choices[beh_session_data['animal_response'].values==1] = 1
    choices[beh_session_data['animal_response'].values==0] = -1
    beh_session_data['choice'] = choices
    beh_session_data['ipsi'] = implant_side * beh_session_data['choice'].values
    beh_session_data['hit'] = beh_session_data['animal_response'].values!=2

    beh_session_data_model = makeSessionDF(session, model_name = model, cut_interruptions=True)
    beh_session_data_model['choice'] = 2*(beh_session_data_model['choice'].values - 0.5) 
    beh_session_data_model['ipsi'] = implant_side * beh_session_data_model['choice'].values
    if 'hit' not in formula:
        beh_session_data = beh_session_data_model.copy()
    if 'outcome' in formula or 'Qchosen' in formula:
        model_dv, cut = load_model_dv(session, model)
        if model_dv is None:
            print(f'Model variables not found for session {session}.')
            return None, None, None, None, session

    FP_jsons = [file for file in os.listdir(session_dir['photometry_data_dir']) if file.endswith('.json')]
    if len(FP_jsons) == 0:
        print('No photometry metadata file found in recording directory.')
        return None, None, None, None, session
    elif len(FP_jsons) > 1:
        print('Multiple photometry metadata files.')
        return None, None, None, None, session
    else:
        FP_json = os.path.join(session_dir['photometry_data_dir'], FP_jsons[0])
    with open(FP_json, 'r') as f:
        FP_info = json.load(f)
    # replace all 'mPFC' tp 'PL'
    for key in FP_info.keys():
        if FP_info[key] == 'mPFC':
            FP_info[key] = 'PL'

    if region not in FP_info.values():
        print('Region not found in recording metadata.')
        return None, None, None, None, session
    else:
        signal = get_FP_data(session)
        curr_signal = zscore(signal[channel][region])
        _, mean_psth, time, _ = align_signal_to_events(
                                                    curr_signal, 
                                                    signal['time_in_beh'], 
                                                    beh_session_data['goCue_start_time'].values, 
                                                    pre_event_time=1.0, post_event_time=1.0, window_size=0.2, step_size=0.1
                                                    );
        go_resp_G = np.mean(mean_psth[time>0.1]) - np.mean(mean_psth[time<-0.1])
        curr_signal_iso = zscore(signal['Iso_tri-exp_mc'][region])
        _, mean_psth, time, _ = align_signal_to_events(
                                                    curr_signal_iso, 
                                                    signal['time_in_beh'], 
                                                    beh_session_data['goCue_start_time'].values, 
                                                    pre_event_time=1.0, post_event_time=1, window_size=0.2, step_size=0.1
                                                    );  
        go_resp_Iso = np.mean(mean_psth[time>0.1]) - np.mean(mean_psth[time<-0.1]) 

        if go_resp_G > thresh:
                
            # glm
            aligned_matrix, _, time_bins, _ = align_signal_to_events(
                                                            curr_signal, 
                                                            signal['time_in_beh'], 
                                                            beh_session_data[align].values, 
                                                            pre_event_time=pre_time, post_event_time=post_time,
                                                            step_size=step_size, window_size=window_size);
            if 'iso' in formula:
                aligned_matrix_iso, _, time_bins_iso, _ = align_signal_to_events(
                                                            curr_signal_iso, 
                                                            signal['time_in_beh'], 
                                                            beh_session_data[align].values, 
                                                            pre_event_time=pre_time, post_event_time=post_time,
                                                            step_size=step_size, window_size=window_size);
                regressors, TvCurrU, PvCurrU, EvCurrU, _, _ = fitSpikeModelG(beh_session_data, aligned_matrix, formula, matIso = aligned_matrix_iso)
            else:
                regressors, TvCurrU, PvCurrU, EvCurrU, _, _ = fitSpikeModelG(beh_session_data, aligned_matrix, formula)
        else:
            regressors, TvCurrU, PvCurrU, EvCurrU = None, None, None, None
    return regressors, TvCurrU, PvCurrU, EvCurrU, session

def process_session_ani(session_list, region, channel='G_tri-exp_mc', formula='spikes ~ 1 + outcome + choice + Qchosen', align='CSon',
        window_size = 0.500, thresh = 0.5, pre_time = 2, post_time = 3, step_size=0.1):
    session_dir = session_dirs(session_list[0])
    model = 'stan_qLearning_5params'
    # load animal metadata
    ani_meta_file = os.path.join('/root/capsule/code/data_management/FP_metadata', f'{session_dir["aniID"]}.json')
    if not os.path.exists(ani_meta_file):
        print(f'Animal metadata file not found for session {session_dir["aniID"]}.')
        return None, None, None, None, None, session_dir["aniID"]
    with open(ani_meta_file, 'r') as f:
        ani_meta = json.load(f)
    if region not in ani_meta.keys():
        print(f'{session_dir["aniID"]} Region not found in surgery info.')
        implant_side = 1
    elif ani_meta[region] == 'L':
        implant_side = -1
    elif ani_meta[region] == 'R':
        implant_side = 1
    else:
        print('Invalid surgery info.')
        return None, None, None, None, None, session_dir["aniID"]
    combined_beh_data = []
    combined_mean_psth = []
    combined_mean_psth_iso = []
    for session in session_list:
        session_dir = session_dirs(session)
        beh_session_data = get_session_tbl(session)
        choices = np.full(len(beh_session_data), np.nan)
        choices[beh_session_data['animal_response'].values==1] = 1
        choices[beh_session_data['animal_response'].values==0] = -1
        beh_session_data['choice'] = choices
        beh_session_data['ipsi'] = implant_side * beh_session_data['choice'].values
        beh_session_data['hit'] = beh_session_data['animal_response'].values!=2

        beh_session_data_model = makeSessionDF(session, model_name = model, cut_interruptions=True)
        beh_session_data_model['choice'] = 2*(beh_session_data_model['choice'].values - 0.5) 
        beh_session_data_model['ipsi'] = implant_side * beh_session_data_model['choice'].values
        if 'hit' not in formula:
            beh_session_data = beh_session_data_model.copy()
        if 'outcome' in formula or 'Qchosen' in formula:
            model_dv, cut = load_model_dv(session, model)
            if model_dv is None:
                continue

        FP_jsons = [file for file in os.listdir(session_dir['photometry_data_dir']) if file.endswith('.json')]
        if len(FP_jsons) == 0:
            print('No photometry metadata file found in recording directory.')
            continue
        elif len(FP_jsons) > 1:
            print('Multiple photometry metadata files.')
            continue
        else:
            FP_json = os.path.join(session_dir['photometry_data_dir'], FP_jsons[0])
        with open(FP_json, 'r') as f:
            FP_info = json.load(f)
        for key in FP_info.keys():
            if FP_info[key] == 'mPFC':
                FP_info[key] = 'PL'

        if region not in FP_info.values():
            print(f'Region not found in session {session}.')
            continue
        else:
            signal = get_FP_data(session)
            curr_signal = zscore(signal[channel][region])
            _, mean_psth, time, _ = align_signal_to_events(
                                                        curr_signal, 
                                                        signal['time_in_beh'], 
                                                        beh_session_data['goCue_start_time'].values, 
                                                        pre_event_time=1, post_event_time=1, window_size=0.2, step_size=0.1
                                                        );
            go_resp_G = np.mean(mean_psth[time>=0.1]) - np.mean(mean_psth[time<-0.1])
            curr_signal_iso = zscore(signal['Iso_tri-exp_mc'][region])
            _, mean_psth, time, _ = align_signal_to_events(
                                                        curr_signal_iso, 
                                                        signal['time_in_beh'], 
                                                        beh_session_data['goCue_start_time'].values, 
                                                        pre_event_time=1, post_event_time=1, window_size=0.2, step_size=0.1
                                                        );  
            go_resp_Iso = np.mean(mean_psth[time>0.1]) - np.mean(mean_psth[time<-0.1]) 

            if go_resp_G > thresh:

                # glm
                aligned_matrix, _, time_bins, _ = align_signal_to_events(
                                                                curr_signal, 
                                                                signal['time_in_beh'], 
                                                                beh_session_data[align].values, 
                                                                pre_event_time=pre_time, post_event_time=post_time,
                                                                step_size=step_size, window_size=window_size, kernel=False);
                if 'iso' in formula:
                    aligned_matrix_iso, _, time_bins_iso, _ = align_signal_to_events(
                                                                curr_signal_iso, 
                                                                signal['time_in_beh'], 
                                                                beh_session_data[align].values, 
                                                                pre_event_time=pre_time, post_event_time=post_time,
                                                                step_size=step_size, window_size=window_size, kernel=False);
                    combined_mean_psth_iso.append(aligned_matrix_iso)
                combined_mean_psth.append(aligned_matrix)
                combined_beh_data.append(beh_session_data)
    if len(combined_mean_psth) == 0:
        return None, None, None, None, None, session_dir["aniID"]
    combined_mean_psth = np.vstack(combined_mean_psth)
    combined_beh_data = pd.concat(combined_beh_data, axis=0).reset_index(drop=True)
    aligned_matrix = combined_mean_psth
    trial_data = combined_beh_data
    if 'iso' in formula:
        combined_mean_psth_iso = np.vstack(combined_mean_psth_iso)
        aligned_matrix_iso = combined_mean_psth_iso
        
    if 'iso' in formula:
        regressors, TvCurrU, PvCurrU, EvCurrU, conf_int, _ = fitSpikeModelG(trial_data, aligned_matrix, formula, matIso = aligned_matrix_iso)
    else:
        regressors, TvCurrU, PvCurrU, EvCurrU, conf_int, _ = fitSpikeModelG(trial_data, aligned_matrix, formula)
    return regressors, TvCurrU, PvCurrU, EvCurrU, conf_int, session_dir["aniID"]

def population_GLM(
        session_list, region, 
        channel='G_tri-exp_mc', formula='spikes ~ 1 + outcome + choice + Qchosen', align='CSon',
        window_size = 500, thresh = 0.5, pre_time = 2000, post_time = 3000, polar_regressors = ['outcome', 'Qchosen'], step_size=100):
    all_T = []
    all_P = []
    all_E = []
    results = Parallel(n_jobs=-4)(delayed(process_session)(
        session, region, channel, formula, align,
        window_size=window_size, thresh=thresh, pre_time=pre_time, post_time=post_time, step_size=step_size
    ) for session in session_list)

    # results = []
    # for session in session_list:
    #     regressors, TvCurrU, PvCurrU, EvCurrU, session = process_session(
    #         session, region, channel, formula, align,
    #         window_size=window_size, thresh=thresh, pre_time=pre_time, post_time=post_time, step_size=step_size
    #     )
    #     results.append((regressors, TvCurrU, PvCurrU, EvCurrU, session))
    regressors,all_T, all_P, all_E, processed_sessions = zip(*results)
        # Filter out None results


    filtered_sessions = [session for session, t in zip(processed_sessions, all_T) if t is not None]
    filtered_results = [(r, t, p, e) for r, t, p, e in zip(regressors, all_T, all_P, all_E) if t is not None]

    if not filtered_results:
        print("No valid sessions processed.")
        return None
    regressors, all_T, all_P, all_E = zip(*filtered_results)
    regressors = regressors[0]  # Assuming all sessions have the same regressors
    

    all_T = np.array(all_T)
    all_P = np.array(all_P)
    all_E = np.array(all_E)
    colors = plt.cm.tab10(np.linspace(0, 1, all_T.shape[2]))
    all_sig_P = (all_P < 0.05) & (all_E > 0)
    all_sig_N = (all_P < 0.05) & (all_E < 0)
    sig_prop_P = np.mean(all_sig_P, 0)
    sig_prop_N = np.mean(all_sig_N, 0)

    num_steps = int((pre_time + post_time - window_size) // step_size) + 1
    time_bins = -pre_time + np.array(range(num_steps)) * step_size + window_size/2

    fig = plt.figure(figsize=(40, 20))
    gs = GridSpec(3+len(regressors)-1, len(time_bins), height_ratios=np.ones(3+len(regressors)-1), width_ratios=np.ones(len(time_bins)))
    ax = fig.add_subplot(gs[0, :])
    for regress_ind, regressor in enumerate(regressors[1:]):
        ax.plot(time_bins, sig_prop_P[:, regress_ind+1], label = regressor, color = colors[regress_ind+1])
        ax.plot(time_bins, -sig_prop_N[:, regress_ind+1], color = colors[regress_ind+1])
    ax.fill_between(time_bins, np.zeros(len(time_bins)), -np.ones(len(time_bins)), color = [0.5, 0.5, 0.5], alpha = 0.2)
    ax.set_ylim([-1, 1])
    # ax.legend()
    for regress_ind, regressor in enumerate(regressors[1:]):
        for bin_ind, curr_time in enumerate(time_bins):
            ax = fig.add_subplot(gs[regress_ind+1, bin_ind])
            curr_Ts = all_T[:, bin_ind, regress_ind+1]
            curr_Ps = all_P[:, bin_ind, regress_ind+1]
            edges = np.linspace(np.min(curr_Ts)-0.001, np.max(curr_Ts)+0.001, 20)
            ax.hist(curr_Ts[curr_Ps<0.05], bins=edges, color = colors[regress_ind+1], alpha = 0.5)
            ax.hist(curr_Ts[curr_Ps>=0.05], bins=edges, color = [0.3, 0.3, 0.3], alpha = 0.7)
            if bin_ind == 0:
                ax.set_ylabel(regressor)
    if polar_regressors[0] in regressors and polar_regressors[1] in regressors:
        polar_regressors_inds = [np.where(np.array(regressors) == polar_regressor)[0][0] for polar_regressor in polar_regressors]
        for bin_ind, curr_time in enumerate(time_bins):
            all_vec = np.column_stack((all_E[:, bin_ind, polar_regressors_inds[0]], all_E[:, bin_ind, polar_regressors_inds[1]]))

            # Convert Cartesian coordinates to polar coordinates
            theta, rho = np.arctan2(all_vec[:, 1], all_vec[:, 0]), np.hypot(all_vec[:, 1], all_vec[:, 0])

            # Define histogram edges (bins) from -π to π
            edges = np.linspace(-np.pi, np.pi, 4*4)

            # Create polar histogram
            ax = fig.add_subplot(gs[-1, bin_ind], polar=True)
            ax.hist(theta, bins=edges, color=[0.1, 0.1, 0.1], alpha=0.7, edgecolor='none', density=True)
            ax.set_yticks([])

            # scatter plot
            ax = fig.add_subplot(gs[-2, bin_ind])
            ax.scatter(all_vec[:, 0], all_vec[:, 1], color=[0.1, 0.1, 0.1], alpha=0.7)
            ax.axvline(0, color='k', linestyle='--')
            ax.axhline(0, color='k', linestyle='--')
    title = f'{region}_{channel}_{align}_{formula}_{window_size}'
    fig.suptitle(title)
    fig.tight_layout()

    return {'tstats': all_T, 'pvals': all_P, 'coefs': all_E, 'regressors': regressors, 'time_bins': time_bins, 'sig_prop_P': sig_prop_P, 'sig_prop_N': sig_prop_N, 'session_list': filtered_sessions, 'fig': fig}

def population_GLM_ani(
        session_list, region, 
        channel='G_tri-exp_mc', formula='spikes ~ 1 + outcome + choice + Qchosen', align='CSon',
        window_size = 0.5, thresh = 0.5, pre_time = 2, post_time = 3, polar_regressors = ['outcome', 'Qchosen'], step_size=0.1):

    ani_list = [session_dirs(session)['aniID'] for session in session_list]
    unique_ani = np.unique(ani_list)
    results = Parallel(n_jobs=-4)(delayed(process_session_ani)(
        [session for session in session_list if session_dirs(session)['aniID'] == ani], 
        region, channel, formula, align,
        window_size=window_size, thresh=thresh, pre_time=pre_time, post_time=post_time, step_size=step_size
    ) for ani in unique_ani)

    # results = []
    # for ani in unique_ani:
    #     curr_sessions = [
    #         session for session in session_list
    #         if session_dirs(session)['aniID'] == ani
    #     ]
        
    #     result = process_session_ani(
    #         curr_sessions,
    #         region,
    #         channel,
    #         formula,
    #         align,
    #         window_size=window_size,
    #         thresh=thresh,
    #         pre_time=pre_time,
    #         post_time=post_time,
    #         step_size=step_size,
    #     )
    #     results.append(result)

    regressors, all_T, all_P, all_E, all_conf_int, processed_anis = zip(*results)
        # Filter out None results
    filtered_anis = [ani for ani, t in zip(processed_anis, all_T) if t is not None]
    filtered_results = [(r, t, p, e, c) for r, t, p, e, c in zip(regressors, all_T, all_P, all_E, all_conf_int) if t is not None]

    if not filtered_results:
        print("No valid sessions processed.")
        return None
    regressors, all_T, all_P, all_E, all_conf_int = zip(*filtered_results)
    regressors = regressors[0]  # Assuming all sessions have the same regressors
    

    all_T = np.array(all_T)
    all_P = np.array(all_P)
    all_E = np.array(all_E)
    all_conf_int = np.array(all_conf_int)
    colors = plt.cm.tab10(np.linspace(0, 1, all_T.shape[2]))
    all_sig_P = (all_P < 0.05) & (all_E > 0)
    all_sig_N = (all_P < 0.05) & (all_E < 0)
    sig_prop_P = np.mean(all_sig_P, 0)
    sig_prop_N = np.mean(all_sig_N, 0)

    num_steps = int((pre_time + post_time - window_size) // step_size) + 1
    time_bins = -pre_time + np.array(range(num_steps)) * step_size + window_size/2

    fig = plt.figure(figsize=(40, 20))
    gs = GridSpec(3+len(regressors)-1, len(time_bins), height_ratios=np.ones(3+len(regressors)-1), width_ratios=np.ones(len(time_bins)))
    ax = fig.add_subplot(gs[0, :])
    for regress_ind, regressor in enumerate(regressors[1:]):
        ax.plot(time_bins, sig_prop_P[:, regress_ind+1], label = regressor, color = colors[regress_ind+1])
        ax.plot(time_bins, -sig_prop_N[:, regress_ind+1], color = colors[regress_ind+1])
    ax.fill_between(time_bins, np.zeros(len(time_bins)), -np.ones(len(time_bins)), color = [0.5, 0.5, 0.5], alpha = 0.2)
    ax.set_ylim([-1, 1])
    # ax.legend()
    for regress_ind, regressor in enumerate(regressors[1:]):
        for bin_ind, curr_time in enumerate(time_bins):
            ax = fig.add_subplot(gs[regress_ind+1, bin_ind])
            curr_Ts = all_T[:, bin_ind, regress_ind+1]
            curr_Ps = all_P[:, bin_ind, regress_ind+1]
            edges = np.linspace(np.min(curr_Ts)-0.001, np.max(curr_Ts)+0.001, 10)
            ax.hist(curr_Ts[curr_Ps<0.05], bins=edges, color = colors[regress_ind+1], alpha = 0.5)
            ax.hist(curr_Ts[curr_Ps>=0.05], bins=edges, color = [0.3, 0.3, 0.3], alpha = 0.7)
            if bin_ind == 0:
                ax.set_ylabel(regressor)
    if polar_regressors[0] in regressors and polar_regressors[1] in regressors:
        polar_regressors_inds = [np.where(np.array(regressors) == polar_regressor)[0][0] for polar_regressor in polar_regressors]
        for bin_ind, curr_time in enumerate(time_bins):
            all_vec = np.column_stack((all_E[:, bin_ind, polar_regressors_inds[0]], all_E[:, bin_ind, polar_regressors_inds[1]]))

            # Convert Cartesian coordinates to polar coordinates
            theta, rho = np.arctan2(all_vec[:, 1], all_vec[:, 0]), np.hypot(all_vec[:, 1], all_vec[:, 0])

            # Define histogram edges (bins) from -π to π
            edges = np.linspace(-np.pi, np.pi, 4*4)

            # Create polar histogram
            ax = fig.add_subplot(gs[-1, bin_ind], polar=True)
            ax.hist(theta, bins=edges, color=[0.1, 0.1, 0.1], alpha=0.7, edgecolor='none', density=True)
            ax.set_yticks([])

            # scatter plot
            ax = fig.add_subplot(gs[-2, bin_ind])
            ax.scatter(all_vec[:, 0], all_vec[:, 1], color=[0.1, 0.1, 0.1], alpha=0.7)
            ax.axvline(0, color='k', linestyle='--')
            ax.axhline(0, color='k', linestyle='--')
    title = f'{region}_{channel}_{align}_{formula}_{window_size}'
    fig.suptitle(title)
    fig.tight_layout()

    return {'tstats': all_T, 'pvals': all_P, 'coefs': all_E, 'conf_int': all_conf_int, 
            'regressors': regressors, 'time_bins': time_bins, 
            'sig_prop_P': sig_prop_P, 'sig_prop_N': sig_prop_N, 'ani_list': filtered_anis, 'fig': fig}

def plot_tuning_curve(session_list, region, target_var = 'pe', channel= 'G_tri-exp_mc', align='CSon', pre_time=-5, post_time=2.5, num_bins=5, thresh=0.5, quantiles=True):
    model = 'stan_qLearning_5params'
    def process_session_for_tuning(session):
        session_dir = session_dirs(session)
        ani_meta_file = os.path.join('/root/capsule/code/data_management/FP_metadata', f'{session_dir["aniID"]}.json')
        with open(ani_meta_file, 'r') as f:
            ani_meta = json.load(f)
        if region not in ani_meta.keys():
            print(f'{session} Region not found in surgery info.')
            return None, None, None
        elif ani_meta[region] == 'L':
            implant_side = -1
        elif ani_meta[region] == 'R':
            implant_side = 1
        else:
            print('Invalid surgery info.')
            return None, None, None
        signal = get_FP_data(session)
        curr_signal = zscore(signal[channel][region])
        trial_data = makeSessionDF(session, model_name = model, cut_interruptions=True)
        if target_var not in trial_data.columns:
            print(f'{target_var} not found in trial data for session {session}.')
            return None, None, None
        _, mean_psth, time, _ = align_signal_to_events(
                                                    curr_signal, 
                                                    signal['time_in_beh'], 
                                                    trial_data['goCue_start_time'].values, 
                                                    pre_event_time=1, post_event_time=1, window_size=0.2, step_size=0.1
                                                    );
        go_resp_G = np.mean(mean_psth[time>100]) - np.mean(mean_psth[time<-100])
        if go_resp_G < thresh:
            return None, None, None
        aligned_matrix, _, time_bins, _ = align_signal_to_events(
                                                        curr_signal, 
                                                        signal['time_in_beh'], 
                                                        trial_data[align].values, 
                                                        pre_event_time=pre_time, post_event_time=post_time,
                                                        step_size=post_time+pre_time, window_size=post_time+pre_time);
        aligned_signal = aligned_matrix[:,0].squeeze()

        # bin photometry data by target var
        target_var_values = trial_data[target_var].values
        if quantiles:
            bins = np.quantile(target_var_values, np.linspace(0, 1, num_bins+1))
        elif num_bins%2 == 0 and np.any(target_var_values>0) and np.any(target_var_values<0):
            # equal bins negative and positive
            bins_neg = np.linspace(np.min(target_var_values), 0, num_bins//2 +1)
            bins_pos = np.linspace(0, np.max(target_var_values), num_bins//2 +1)[1:]
            bins = np.concatenate([bins_neg, bins_pos])
        else:
            bins = np.linspace(np.min(target_var_values)-0.0001, np.max(target_var_values)+0.0001, num_bins+1)

        curr_bin_means = np.full(num_bins, np.nan)
        curr_signal_means = np.full(num_bins, np.nan)
        for bin_ind in range(num_bins):
            bin_mask = (target_var_values >= bins[bin_ind]) & (target_var_values < bins[bin_ind+1])
            if np.sum(bin_mask) > 0:
                curr_bin_means[bin_ind] = np.mean(target_var_values[bin_mask])
                curr_signal_means[bin_ind] = np.mean(aligned_signal[bin_mask])

        curr_signal_means[~np.isnan(curr_signal_means)] = zscore(curr_signal_means[~np.isnan(curr_signal_means)])

        return curr_bin_means, curr_signal_means, session
    
    results = Parallel(n_jobs=8)(delayed(process_session_for_tuning)(session) for session in session_list)
    all_bin_means, all_signal_means, processed_sessions = zip(*[res for res in results if res[0] is not None])
    all_bin_means = np.array(all_bin_means)
    all_signal_means = np.array(all_signal_means)
    mean_bin_means = np.nanmean(all_bin_means, 0)
    mean_signal_means = np.nanmean(all_signal_means, 0)
    sem_signal_means = np.nanstd(all_signal_means, 0) / np.sqrt(np.sum(~np.isnan(all_signal_means), 0))
    fig = plt.figure(figsize=(8,6))
    plt.plot(mean_bin_means, mean_signal_means, '-o', color='k')
    plt.fill_between(mean_bin_means, mean_signal_means - sem_signal_means, mean_signal_means + sem_signal_means, alpha=0.3, facecolor='k', edgecolor='none')
    plt.xlabel(target_var)
    plt.ylabel('Photometry signal')
    title = f'{region}_{channel}_{align}_tuning_curve_{target_var}'
    plt.title(title)
    return {'bin_means': all_bin_means, 'signal_means': all_signal_means, 'session_list': processed_sessions,  'fig': fig}


def plot_psth(
        session_list, region, 
        channel='G_tri-exp_mc', align='CSon',
        pre_time = 2, post_time = 3, step_size=0.1, bin_size=1, thresh = 0.5, num_bins = 6, target_var='pe', quantiles=True):
    model = 'stan_qLearning_5params'
    def process_session_for_psth(session):
        session_dir = session_dirs(session)
        ani_meta_file = os.path.join('/root/capsule/code/data_management/FP_metadata', f'{session_dir["aniID"]}.json')
        if not os.path.exists(ani_meta_file):
            print(f'{session} Metadata file not found.')
            return None, None
        with open(ani_meta_file, 'r') as f:
            ani_meta = json.load(f)
        if region not in ani_meta.keys():
            print(f'{session} Region not found in surgery info.')
            return None, None
        elif ani_meta[region] == 'L':
            implant_side = -1
        elif ani_meta[region] == 'R':
            implant_side = 1
        else:
            print('Invalid surgery info.')
            return None, None
        signal = get_FP_data(session)
        curr_signal = zscore(signal[channel][region])
        beh_session_data_raw = get_session_tbl(session, cut_interruptions=True)
        beh_session_data = makeSessionDF(session, model_name = model, cut_interruptions=True)
        _, mean_psth, time, _ = align_signal_to_events(
                                                    curr_signal, 
                                                    signal['time_in_beh'], 
                                                    beh_session_data['goCue_start_time'].values, 
                                                    pre_event_time=1, post_event_time=1, window_size=0.2, step_size=0.1
                                                    );
        go_resp_G = np.mean(mean_psth[time>0.1]) - np.mean(mean_psth[time<-0.1])
        if go_resp_G < 0.5:
            return None, None
                # make session train data
        if target_var != 'hit':
            trial_data = beh_session_data.copy()
        else:
            trial_data = beh_session_data_raw.copy()
        
        aligned_matrix, _, time_bins, _ = align_signal_to_events(
                                                        curr_signal, 
                                                        signal['time_in_beh'], 
                                                        trial_data[align].values, 
                                                        pre_event_time=pre_time, post_event_time=post_time,
                                                        step_size=step_size, window_size=step_size, kernel=True);
        target_var_values = trial_data[target_var].values
 
        if quantiles:
            bins = np.quantile(target_var_values, np.linspace(0, 1, num_bins+1))
        elif num_bins%2 == 0 and np.any(target_var_values>0) and np.any(target_var_values<0):
            # equal bins negative and positive
            bins_neg = np.linspace(np.min(target_var_values), 0, num_bins//2 +1)
            bins_pos = np.linspace(0, np.max(target_var_values), num_bins//2 +1)[1:]
            bins = np.concatenate([bins_neg, bins_pos])
        else:
            bins = np.linspace(np.min(target_var_values)-0.0001, np.max(target_var_values)+0.0001, num_bins+1)
        binned_psth = np.full((num_bins, aligned_matrix.shape[1]), np.nan)
        for bin_ind in range(num_bins):
            bin_mask = (target_var_values >= bins[bin_ind]) & (target_var_values < bins[bin_ind+1])
            if np.sum(bin_mask) > 0:
                binned_psth[bin_ind, :] = np.mean(aligned_matrix[bin_mask, :], 0)
        return binned_psth, time_bins       
    results = Parallel(n_jobs=-4)(delayed(process_session_for_psth)(session) for session in session_list)
    # results = []
    # for session in session_list:
    #     result = process_session_for_psth(session)
    #     results.append(result)
    all_binned_psth, time_bins = zip(*[res for res in results if res[0] is not None])
    all_binned_psth = np.array(all_binned_psth)
    mean_binned_psth = np.nanmean(all_binned_psth, 0)
    sem_binned_psth = np.nanstd(all_binned_psth, 0) / np.sqrt(np.sum(~np.isnan(all_binned_psth), 0))
    fig = plt.figure(figsize=(12,6))

    custom_cmap = LinearSegmentedColormap.from_list(
        'red_white_blue',
        ['red', 'white', 'blue']
    )

    ax = fig.add_subplot(121)
    for bin_ind in range(mean_binned_psth.shape[0]):
        ax.plot(time_bins[0], mean_binned_psth[bin_ind, :], label=f'Bin {bin_ind+1}', color=custom_cmap(bin_ind / (mean_binned_psth.shape[0] - 1)))
        ax.fill_between(time_bins[0], 
                         mean_binned_psth[bin_ind, :] - sem_binned_psth[bin_ind, :], 
                         mean_binned_psth[bin_ind, :] + sem_binned_psth[bin_ind, :], 
                         alpha=0.3, facecolor=custom_cmap(bin_ind / (mean_binned_psth.shape[0] - 1)), edgecolor='none')
    ax.set_xlabel('Time from ' + align + ' (ms)')
    ax.set_ylabel('Photometry signal')
    ax.legend()
    ax.set_title('Binned PSTH - raw')

    ax = fig.add_subplot(122)
    # remove baseline by subtracting mean of pre-event time
    baseline_mask = time_bins[0] < 0
    baseline_values = mean_binned_psth[:, baseline_mask]
    baseline_mean = np.nanmean(baseline_values, axis=1, keepdims=True)
    mean_binned_psth_baseline_corrected = mean_binned_psth - baseline_mean
    for bin_ind in range(mean_binned_psth_baseline_corrected.shape[0]):
        ax.plot(time_bins[0], mean_binned_psth_baseline_corrected[bin_ind, :], label=f'Bin {bin_ind+1}', color=custom_cmap(bin_ind / (mean_binned_psth_baseline_corrected.shape[0] - 1)))
        ax.fill_between(time_bins[0], 
                         mean_binned_psth_baseline_corrected[bin_ind, :] - sem_binned_psth[bin_ind, :], 
                         mean_binned_psth_baseline_corrected[bin_ind, :] + sem_binned_psth[bin_ind, :], 
                         alpha=0.3, facecolor=custom_cmap(bin_ind / (mean_binned_psth_baseline_corrected.shape[0] - 1)), edgecolor='none')
    ax.set_xlabel('Time from ' + align + ' (ms)')
    ax.set_ylabel('Photometry signal')
    ax.legend()
    ax.set_title('Binned PSTH - baseline corrected')

    title = f'{region}_{channel}_{align}_psth_{target_var}'
    plt.suptitle(title)
    return {'binned_psth': all_binned_psth, 'time_bins': time_bins[0], 'fig': fig}





