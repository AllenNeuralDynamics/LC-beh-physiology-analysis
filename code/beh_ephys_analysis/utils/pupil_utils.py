# %%
import os
import sys

from pandas.core.frame import ensure_index_from_sequences
sys.path.append('/root/capsule/code/beh_ephys_analysis')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from utils.beh_functions import session_dirs, get_unit_tbl, get_session_tbl, makeSessionDF
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from scipy.interpolate import interp1d
from scipy.stats import zscore
from scipy.io import loadmat

def linreg_remove_outliers(A, B, default_m, z=3.5):
    """
    Fit B = m*A + c, flag outliers using robust z-score of residuals (MAD),
    remove them, then refit.

    Returns:
      m, c: refit slope/intercept
      inlier_mask: boolean mask (True = kept)
      outlier_idx: indices removed
      residuals: residuals from refit on all points
    """
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    mask = np.isfinite(A) & np.isfinite(B)

    x = A[mask]
    y = B[mask]

    if x.size < 3:
        raise ValueError("Need at least 3 finite points.")

    from scipy.stats import theilslopes

    m, c, lo, hi = theilslopes(y, x)   # y = m*x + c
    r = y - (m * x + c)

    if np.abs(m - default_m) / np.abs(default_m) > 0.3:
        m = default_m
        # infer c
        c = np.median(y - m * x)

    # robust scale via MAD
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad == 0:
        # fallback: no dispersion -> no outliers
        inliers_sub = np.ones_like(r, dtype=bool)
    else:
        robust_z = 0.6745 * (r - med) / mad
        inliers_sub = np.abs(robust_z) <= z

    # refit on inliers
    m2, c2, lo2, hi2 = theilslopes(y[inliers_sub], x[inliers_sub])

    # build full-length inlier mask (same length as A/B)
    inlier_mask = np.zeros_like(mask, dtype=bool)
    idx = np.where(mask)[0]
    inlier_mask[idx[inliers_sub]] = True

    outlier_idx = np.where(mask)[0][~inliers_sub]

    # residuals from refit (for all points)
    residuals_all = B - (m2 * A + c2)

    return m2, c2, inlier_mask, outlier_idx, residuals_all

def load_pupil(session, dia_thresh = 0.9): # = 'behavior_ZS062_2021-04-02_19-08-52'
    session_dir = session_dirs(session)
    raw_nwb_file = [f for f in os.listdir(session_dir['raw_dir']) if f.endswith('.nwb.zarr')][0]
    raw_nwb = load_nwb_from_filename(os.path.join(session_dir['raw_dir'], raw_nwb_file))
    raw_df = raw_nwb.intervals['trials'].to_dataframe()
    pupil_files = os.listdir(os.path.join(session_dir['sorted_dir_curated'], 'session'))
    pupil_file = [f for f in pupil_files if f.endswith('_pupil.mat')]
    if len(pupil_file)==0:
        print('No pupil file found')
        return None
    elif len(pupil_file)>1:
        print('Multiple pupil files found')
        return None
    else:
        pupil_file = pupil_file[0]
        print(f'Loading pupil file: {pupil_file}')
        pupil_data = loadmat(os.path.join(session_dir['sorted_dir_curated'], 'session', pupil_file))
        # pupil_diameter = pupil_data['pupil_diameter'].squeeze()
        # pupil_timestamps = pupil_data['pupil_timestamps'].squeeze()
        # plt.figure(figsize=(10,4))
        # plt.plot(pupil_timestamps, pupil_diameter)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Pupil Diameter (pixels)')
        # plt.title('Pupil Diameter over Time')
        # plt.show()

        # use linear interpolation to align pupil data to goCue times
        # load raw pupil from dlc

        HERE = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(
            os.path.dirname(os.path.dirname(HERE)),
            'data_management',
            'pupil_note.json'
        )
        with open(json_file, 'r') as jf:
            pupil_notes = json.load(jf)

        dlc_folder = os.path.join(session_dir['raw_dir'], 'behavior-videos', 'pupil')

        all_iters = pupil_notes[session_dir['aniID']]

        for it in all_iters:
            dlc_files = [f for f in os.listdir(dlc_folder) 
                        if f.endswith(f'_{it}.csv') and 'DLC' in f and 'skeleton' not in f]  
            if len(dlc_files) > 0:
                best_iter = it
                break

        if len(dlc_files) == 0:
            print('No raw DLC pupil files found')
            return None
        elif len(dlc_files) > 1:
            print('Multiple raw DLC pupil files found, pick more specific iteration.')
            return None
        else:
            dlc_file = dlc_files[0]
        
        raw_pupil = pd.read_csv(os.path.join(dlc_folder, dlc_file), skiprows=3, usecols=range(1, 7), header=None).values
        led_conf = pd.read_csv(os.path.join(dlc_folder, dlc_file), skiprows=3, usecols=[15], header=None).values.squeeze()
        L = raw_pupil[:, 0:3]
        R = raw_pupil[:, 3:6]
        del raw_pupil
        dia = np.sqrt((L[:, 0]-R[:, 0])**2 + (L[:, 1]-R[:, 1])**2)
        dia_conf = np.sqrt(L[:, 2] * R[:, 2])
        dia[dia_conf < dia_thresh] = np.nan
        dia = dia.astype(float, copy=False)

        # remove those small segments of non nan values shorter than 0.5 seconds surrounded by nans
        min_len = int(round(0.5 * float(pupil_data['FR'][0][0])))

        isnan = np.isnan(dia)
        notnan = ~isnan

        diff = np.diff(np.concatenate(([0], notnan.view(np.int8), [0])))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]   # end is exclusive

        n = len(dia)
        for start, end in zip(starts, ends):
            if (end - start) < min_len:
                left = start - 1
                right = end
                # require it to be inside and bounded by NaNs
                if left >= 0 and right < n and isnan[left] and isnan[right]:
                    dia[start:end] = np.nan


        # remove small NaN segments (<0.1 s) by interpolation (only if bounded by valid samples)
        min_nan_len = int(round(0.1 * float(pupil_data['FR'][0][0])))  # FR in Hz -> samples
        isnan = np.isnan(dia)

        diff = np.diff(np.concatenate(([0], isnan.view(np.int8), [0])))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]  # end is exclusive

        n = len(dia)
        for start, end in zip(starts, ends):
            run_len = end - start
            if run_len < min_nan_len:
                left = start - 1
                right = end  # first index after run

                # must be strictly inside bounds to be "surrounded"
                if left < 0 or right >= n:
                    continue

                # both anchors must be finite
                if np.isfinite(dia[left]) and np.isfinite(dia[right]):
                    dia[start:end] = np.interp(
                        np.arange(start, end),
                        [left, right],
                        [dia[left], dia[right]],
                    )



        pupil_data['dia'] = [dia]


        if len(pupil_data['cueFT'][0].squeeze()) == len(raw_df['goCue_start_time'].values):
            # get a linear fit and remove outliers
            # remove zeros in frame times
            zeros_idx = np.where(pupil_data['cueFT'][0].squeeze()==0)[0]
            frame_times = np.delete(pupil_data['cueFT'][0].squeeze(), zeros_idx)
            go_cue_times = np.delete(raw_df['goCue_start_time'].values, zeros_idx)
            m, c, inlier_mask, outlier_idx, resid = linreg_remove_outliers(go_cue_times, frame_times, pupil_data['FR'][0][0], z=3.5)
            interp_func = interp1d(frame_times[inlier_mask], go_cue_times[inlier_mask], 
                                    bounds_error=False, fill_value="extrapolate")
            interp_func_inv = interp1d(go_cue_times[inlier_mask], frame_times[inlier_mask],
                                    bounds_error=False, fill_value="extrapolate")
            best_coeff = 1
            qual_ind = pupil_data['qualInd'][0]
        else:
            print('Mismatch in cue times length, finding best match')
            # find maximum correlation between inter-event-intervals
            cueFT_diff = np.diff(pupil_data['cueFT'][0].squeeze())
            goCue_diff = np.diff(raw_df['goCue_start_time'].values)
            if len(cueFT_diff) > len(goCue_diff):
                lags = range(0, len(cueFT_diff) - len(goCue_diff) + 1)
                corrs = [np.corrcoef(cueFT_diff[lag:lag+len(goCue_diff)], goCue_diff)[0,1] for lag in lags]
                best_lag = lags[np.argmax(corrs)]
                print(f'Best lag found: {best_lag}, best coeff: {np.max(corrs)}')
                matched_FT = pupil_data['cueFT'][0].squeeze()[best_lag:best_lag+len(goCue_diff)+1]
                best_coeff = np.max(corrs)
                # remove zeros
                zeros_idx = np.where(matched_FT==0)[0]
                matched_FT = np.delete(matched_FT, zeros_idx)
                go_cue_times = np.delete(raw_df['goCue_start_time'].values, zeros_idx)
                # remove errors
                m, c, inlier_mask, outlier_idx, resid = linreg_remove_outliers(go_cue_times, matched_FT, pupil_data['FR'][0][0], z=3.5)
                interp_func = interp1d(matched_FT[inlier_mask], go_cue_times[inlier_mask],
                                       bounds_error=False, fill_value="extrapolate")
                interp_func_inv = interp1d(go_cue_times[inlier_mask], matched_FT[inlier_mask],
                                          bounds_error=False, fill_value="extrapolate") 
                qual_ind = pupil_data['qualInd'][0][best_lag:best_lag+len(goCue_diff)+1]
            else:
                lags = range(0, len(goCue_diff) - len(cueFT_diff) + 1)
                corrs = [np.corrcoef(cueFT_diff, goCue_diff[lag:lag+len(cueFT_diff)])[0,1] for lag in lags]
                best_lag = lags[np.argmax(corrs)]
                print(f'Best lag found: {best_lag}, best coeff: {np.max(corrs)}')
                matched_cue = raw_df['goCue_start_time'].values[best_lag:best_lag+len(cueFT_diff)+1]
                best_coeff = np.max(corrs)
                # remove zeros
                zeros_idx = np.where(pupil_data['cueFT'][0].squeeze()==0)[0]
                matched_cue = np.delete(matched_cue, zeros_idx)
                frame_times = np.delete(pupil_data['cueFT'][0].squeeze(), zeros_idx)
                m, c, inlier_mask, outlier_idx, resid = linreg_remove_outliers(matched_cue, frame_times, pupil_data['FR'][0][0], z=3.5)
                interp_func = interp1d(frame_times[inlier_mask], matched_cue[inlier_mask],
                                       bounds_error=False, fill_value="extrapolate")
                interp_func_inv = interp1d(matched_cue[inlier_mask], frame_times[inlier_mask],
                                          bounds_error=False, fill_value="extrapolate")
                qual_ind = pupil_data['qualInd'][0]

        pupil_times = interp_func(np.arange(len(pupil_data['dia'][0])))
        go_cue_frames = interp_func_inv(raw_df[raw_df['trial_type']=='CSplus']['goCue_start_time'].values)
        # calculate confidence of timing and pupil diameter for each trial
        dia_conf_trial = np.zeros_like(go_cue_frames)
        time_conf_trial = np.zeros_like(go_cue_frames)
        for i, go_cue_F in enumerate(go_cue_frames):
            dia_conf_trial[i] = np.mean(dia_conf[int(go_cue_F):int(go_cue_F+ pupil_data['FR'][0][0]*2.0)])
            time_conf_trial[i] = np.mean(led_conf[int(go_cue_F):int(go_cue_F+ pupil_data['FR'][0][0]*0.5)])

        pupil_diameter = pupil_data['dia'][0].squeeze()

        # set pupil diameter values to nan if it is within certain time range from cue_FTs
        # time_range = [-0.01, 3.5]
        # for cue_ind, cue_F in enumerate(pupil_data['cueFT'][0].squeeze()):
        #     if pupil_data['qualInd'][0][cue_ind]==0:
        #         start_F = cue_F + time_range[0]*pupil_data['FR'][0][0]
        #         end_F = cue_F + time_range[1]*pupil_data['FR'][0][0]
        #         pupil_diameter[int(start_F):int(end_F)] = np.nan
            
        # zscore
        pupil_diameter_z = np.full_like(pupil_diameter, np.nan)
        pupil_diameter_z[~np.isnan(pupil_diameter)] = zscore(pupil_diameter[~np.isnan(pupil_diameter)])

        if session == 'behavior_ZS062_2021-05-01_19-48-01':
            pupil_times = pupil_times[pupil_times<=3390000]
            pupil_diameter = pupil_diameter[:len(pupil_times)]
            pupil_diameter_z = pupil_diameter_z[:len(pupil_times)]
    return {'pupil_times': pupil_times,
            'pupil_diameter': pupil_diameter,
            'pupil_diameter_z': pupil_diameter_z,
            'error_prop': pupil_data['errorProp'][0][0],
            'best_coeff': best_coeff,
            'qual_pass': pupil_data['qualInd'][0],
            'cue_FT': go_cue_frames,
            'time_conf_trial': time_conf_trial,
            'dia_conf_trial': dia_conf_trial,
            'mean_conf': np.nanmean(dia_conf[dia_conf>dia_thresh])}
