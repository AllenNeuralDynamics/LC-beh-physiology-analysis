# generate behavior metrics. Lopp through all sessions and generate metrics, save in json file. Then combine all json files into a dataframe and save as csv for scatter plot and pair plot. Also save the combined dataframe as pickle file for later use.
# %%
import sys
import os
import traceback
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.colors import LinearSegmentedColormap
# import pandas as pd
# from pynwb import NWBFile, TimeSeries, NWBHDF5IO
# import json
# import seaborn as sns
# from PyPDF2 import PdfMerger
# from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
# import re
from utils.beh_functions import session_dirs, parseSessionID, load_model_dv, makeSessionDF, get_session_tbl, get_unit_tbl, get_history_from_nwb, plot_session_glm
from utils.ephys_functions import*
from utils.ccf_utils import ccf_pts_convert_to_mm
from utils.capsule_migration import capsule_directories
import pickle
# import scipy.stats as stats
# import spikeinterface as si
# import shutil
# from sklearn.metrics import roc_auc_score, roc_curve
# from sklearn.metrics import r2_score
# from matplotlib import cm
# import matplotlib.colors as mcolors
from joblib import Parallel, delayed
# from utils.combine_tools import apply_qc
# from scipy.stats import gaussian_kde
from beh_mertics import cal_beh_metrics
capsule_dirs = capsule_directories()


recompute_metrics = True
# %%
df = pd.read_csv('/root/capsule/code/data_management/session_assets.csv', dtype={'probe': str})
session_ids = df['session_id'].values
session_ids = [session_id for session_id in session_ids if isinstance(session_id, str)]  # filter only behavior sessions


def process_session(session):
    """
    """
    print(session)
    session_data = None
    session_dir = session_dirs(session)
    if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
        try:
            cal_beh_metrics(session)  # writes {session}_beh_metrics.json
            json_file_path = os.path.join(session_dir['beh_fig_dir'], f'{session}_beh_metrics.json')
            with open(json_file_path, 'r') as json_file:
                session_data = json.load(json_file)
            session_data['session'] = session
            session_data['ani_id'] = session_dir['aniID']
            session_data['probe'] = df[df['session_id'] == session]['probe'].values[0] if 'probe' in df.columns else None
        except Exception as e:
            print(f"Error processing session {session}: {e!r}")
            traceback.print_exc()
            session_data = None

    return session_data


if recompute_metrics:
    # Use Parallel to process sessions in parallel
    combined_beh_sessions = Parallel(n_jobs=-4)(delayed(process_session)(session) for session in session_ids)
    combined_beh_sessions = [session for session in combined_beh_sessions if session is not None]
else:
    # Load precomputed metrics from JSON files
    combined_beh_sessions = []
    for session in session_ids:
        session_dir = session_dirs(session)
        if os.path.exists(os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')):
            json_file_path = os.path.join(session_dir['beh_fig_dir'], f'{session}_beh_metrics.json')
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as json_file:
                    session_data = json.load(json_file)
                session_data['ani_id'] = session_dir['aniID']
                session_data['probe'] = df[df['session_id'] == session]['probe'].values[0] if 'probe' in df.columns else None
                session_data['session'] = session
                combined_beh_sessions.append(session_data)

# %%
combined_beh_sessions = pd.DataFrame(combined_beh_sessions)
combined_beh_sessions['sw_nrwd_rwd'] = combined_beh_sessions['p_sw_L'] - (1-combined_beh_sessions['p_st_w'])
combined_beh_sessions['sw_bias'] = (combined_beh_sessions['p_sw_L_L'] - combined_beh_sessions['p_sw_L_R'])/(combined_beh_sessions['p_sw_L_L'] + combined_beh_sessions['p_sw_L_R'] + 1e-6)
combined_beh_sessions['lat_bias_abs'] = np.abs(combined_beh_sessions['lick_lat_diff_mode'])
combined_beh_sessions['var_lat_bias_abs'] = np.abs(combined_beh_sessions['var_lat_diff'])
combined_beh_sessions['lick_bias'] = (combined_beh_sessions['lat_bias_abs'] > 0.3) & (combined_beh_sessions['var_lat_bias_abs'] > 0.75)
combined_beh_dir = os.path.join(capsule_dirs["manuscript_fig_prep_dir"], 'combined_session_tbl')
combined_beh_file = os.path.join(combined_beh_dir, 'combined_beh_sessions.pkl')
if not os.path.exists(combined_beh_dir):
    os.makedirs(combined_beh_dir)
with open(combined_beh_file, 'wb') as f:
    pickle.dump(combined_beh_sessions, f)

