import numpy as np
from scipy import stats
from scipy.fft import next_fast_len, rfft, irfft
import statsmodels.api as sm
import re
from PyPDF2 import PdfMerger
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
# from beh_functions import session_dirs
from matplotlib import gridspec
import matplotlib.pyplot as plt
# from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
# from aind_ephys_utils import align
try:
    from .beh_functions import session_dirs
except ImportError:
    from beh_functions import session_dirs
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
import ast
import json
import pickle
from sklearn.metrics import r2_score
from aind_ephys_utils import align

def regressors_to_formula(response_var, regressors):
    terms = [r for r in regressors if r != 'Intercept']
    has_intercept = 'Intercept' in regressors
    rhs = '1' if has_intercept else '0'
    if terms:
        rhs += ' + ' + ' + '.join(terms)
    return f'{response_var} ~ {rhs}'

def filter_jc(x, time_constant = 20):
    # return filtered firing rate given time from spike (in ms)
    # time_constant: in ms 
    return (1 - np.exp(-1000*x)) * (np.exp(-1000*x/time_constant))
    
def get_spike_matrix_filter(spike_times, align_time, pre_event, post_event, time_constant=20, stepSize=0.05):
    bin_times = np.arange(pre_event, post_event, stepSize)
    spike_matrix = np.zeros((len(align_time), len(bin_times)))
    for time_ind, curr_time in enumerate(bin_times):
        curr_spikes = [
            spike_times[
                (spike_times <= (curr_time + align_time_curr))
                & (spike_times > (curr_time + align_time_curr - time_constant*10/1000))
            ] - (curr_time + align_time_curr)
            for align_time_curr in align_time
        ]
        curr_rates = [
            np.mean(filter_jc(-curr_spikes_trial, time_constant=time_constant))
            if len(curr_spikes_trial) > 0 else 0
            for curr_spikes_trial in curr_spikes
        ]
        spike_matrix[:, time_ind] = curr_rates
    return spike_matrix, bin_times
def build_time_window_domain(bin_edges, offsets, callback=None):
    callback = (lambda x: x) if callback is None else callback
    domain = np.tile(bin_edges[None, :], (len(offsets), 1))
    domain += offsets[:, None]
    return callback(domain)

def build_spike_histogram(time_domain,
                          spike_times,
                          dtype=None,
                          binarize=False):

    time_domain = np.array(time_domain)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1, len(spike_times)),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, 1:]
        
    for ii in range(len(spike_times)):
        data = np.array(spike_times[ii])

        start_positions = np.searchsorted(data, starts.flat)
        end_positions = np.searchsorted(data, ends.flat, side="right")
        counts = (end_positions - start_positions)

        tiled_data[:, :, ii].flat = counts > 0 if binarize else counts
    
    time = 0.5*(starts + ends)
    return time, tiled_data
    
# Make xarray functions
def build_spike_histogram_overlap(time_domain,
                          binSize,
                          spike_times,
                          dtype=None,
                          binarize=False):

    time_domain = np.array(time_domain)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1, len(spike_times)),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, :-1]  + binSize
        
    for ii in range(len(spike_times)):
        data = np.array(spike_times[ii])

        start_positions = np.searchsorted(data, starts.flat)
        end_positions = np.searchsorted(data, ends.flat, side="right")
        counts = (end_positions - start_positions)

        tiled_data[:, :, ii].flat = counts > 0 if binarize else counts

    time = starts + 0.5*binSize
    return time, tiled_data

def fitSpikeModelG(dfTrial, matSpikes, formula, matIso = None):
    TvCurrU = np.array([])
    PvCurrU = np.array([])
    EvCurrU = np.array([])
    ciCurrU = np.array([])
    for i in range(np.shape(matSpikes)[1]):
        currSpikes = np.squeeze(matSpikes[:,i])
        currData = dfTrial.copy()
        currData['spikes'] = np.squeeze(currSpikes)
        if matIso is not None:
            currData['iso'] = matIso[:,i]
        # Fit the GLM
        model = sm.GLM.from_formula(formula=formula, data=currData, family=sm.families.Gaussian()).fit()
        # t value
        regressors = [re.sub(r'\[.*?\]', '', x) for x in model.tvalues.index]
        tv = model.tvalues.values.reshape(1,-1)
        # p value
        pv = model.pvalues.values.reshape(1,-1)
        # t value
        ev = model.params.values.reshape(1,-1)
        # ci value
        ci = model.conf_int(alpha=0.05)  # (p, 2)
        # concatenate
        # initialize shape if not
        if np.shape(TvCurrU)[0] == 0:
            TvCurrU = np.empty((0, len(regressors)))
            PvCurrU = np.empty((0, len(regressors)))
            EvCurrU = np.empty((0, len(regressors)))
            ciCurrU = np.empty((0, 2, len(regressors)))

        TvCurrU = np.concatenate((TvCurrU, tv), axis = 0)
        PvCurrU = np.concatenate((PvCurrU, pv), axis = 0)
        EvCurrU = np.concatenate((EvCurrU, ev), axis = 0)
        ciCurrU = np.concatenate((ciCurrU, ci.values.T.reshape(1, 2, -1)), axis = 0)

        used_idx = model.model.data.row_labels

        r2_score_value = r2_score(
            currData.loc[used_idx, 'spikes'],
            model.fittedvalues
        )

    return regressors, TvCurrU, PvCurrU, EvCurrU, ciCurrU, r2_score_value


def fitSpikeModelG_CI(
    dfTrial,
    matSpikes,
    formula,
    matIso=None,
    alpha=0.05,          # 95% CI by default
):
    """
    Fit GLM per timepoint (column in matSpikes) and return t/p/coef plus CI.

    Returns
    -------
    regressors : list[str]
    TvCurrU : (n_timepoints, n_regressors) ndarray
    PvCurrU : (n_timepoints, n_regressors) ndarray
    EvCurrU : (n_timepoints, n_regressors) ndarray
    CILowCurrU : (n_timepoints, n_regressors) ndarray
    CIHighCurrU : (n_timepoints, n_regressors) ndarray
    r2_scores : (n_timepoints,) ndarray
    """

    n_timepoints = np.shape(matSpikes)[1]

    TvCurrU = np.empty((0, 0))
    PvCurrU = np.empty((0, 0))
    EvCurrU = np.empty((0, 0))
    CILowCurrU = np.empty((0, 0))
    CIHighCurrU = np.empty((0, 0))
    valid_points = []

    r2_scores = np.full(n_timepoints, np.nan)
    regressors = None

    for i in range(n_timepoints):
        currSpikes = np.squeeze(matSpikes[:, i])
        valid_mask = ~np.isnan(currSpikes)

        currData = dfTrial.copy()
        currData["spikes"] = currSpikes

        if np.sum(valid_mask) < 30:
            # append NaNs for this unit
            continue
           

        if matIso is not None:
            currData["iso"] = np.squeeze(matIso[:, i])

        # Fit GLM
        model = sm.GLM.from_formula(
            formula=formula,
            data=currData,
            family=sm.families.Gaussian()
        ).fit()

        # Regressor names (strip category levels etc.)
        reg_names = [re.sub(r"\[.*?\]", "", x) for x in model.tvalues.index]

        # On first timepoint, initialize matrices with correct width
        if regressors is None:
            regressors = reg_names
            p = len(regressors)

            TvCurrU = np.empty((0, p))
            PvCurrU = np.empty((0, p))
            EvCurrU = np.empty((0, p))
            CILowCurrU = np.empty((0, p))
            CIHighCurrU = np.empty((0, p))
        else:
            # Safety check: formula/design should produce same columns every unit
            if reg_names != regressors:
                raise ValueError(
                    f"Regressor mismatch at time {i}. "
                    f"Expected {regressors}, got {reg_names}."
                )

        tv = model.tvalues.values.reshape(1, -1)
        pv = model.pvalues.values.reshape(1, -1)
        ev = model.params.values.reshape(1, -1)

        ci = model.conf_int(alpha=alpha)  # (p, 2)
        ci_low = ci[0].values.reshape(1, -1)
        ci_high = ci[1].values.reshape(1, -1)

        TvCurrU = np.concatenate((TvCurrU, tv), axis=0)
        PvCurrU = np.concatenate((PvCurrU, pv), axis=0)
        EvCurrU = np.concatenate((EvCurrU, ev), axis=0)
        CILowCurrU = np.concatenate((CILowCurrU, ci_low), axis=0)
        CIHighCurrU = np.concatenate((CIHighCurrU, ci_high), axis=0)

        used_idx = model.model.data.row_labels
        r2_scores[i] = r2_score(
            currData.loc[used_idx, "spikes"],
            model.fittedvalues
        )
        valid_points.append(i)
    
    if len(valid_points) < n_timepoints:
        # fill missed timepoints with NaNs
        TvCurrU_full = np.full((n_timepoints, TvCurrU.shape[1]), np.nan)
        PvCurrU_full = np.full((n_timepoints, PvCurrU.shape[1]), np.nan)
        EvCurrU_full = np.full((n_timepoints, EvCurrU.shape[1]), np.nan)
        CILowCurrU_full = np.full((n_timepoints, CILowCurrU.shape[1]), np.nan)
        CIHighCurrU_full = np.full((n_timepoints, CIHighCurrU.shape[1]), np.nan)
        for idx, timepoint in enumerate(valid_points):
            TvCurrU_full[timepoint, :] = TvCurrU[idx, :]
            PvCurrU_full[timepoint, :] = PvCurrU[idx, :]
            EvCurrU_full[timepoint, :] = EvCurrU[idx, :]
            CILowCurrU_full[timepoint, :] = CILowCurrU[idx, :]
            CIHighCurrU_full[timepoint, :] = CIHighCurrU[idx, :]
        TvCurrU = TvCurrU_full
        PvCurrU = PvCurrU_full
        EvCurrU = EvCurrU_full
        CILowCurrU = CILowCurrU_full
        CIHighCurrU = CIHighCurrU_full


    return {'regressors': regressors,
            'TvCurrU': TvCurrU,
            'PvCurrU': PvCurrU,
            'EvCurrU': EvCurrU,
            'CILowCurrU': CILowCurrU,
            'CIHighCurrU': CIHighCurrU,
            'r2_scores': r2_scores
           }


def fitSpikeModelP(dfTrial, matSpikes, formula):
    TvCurrU = np.array([])
    PvCurrU = np.array([])
    EvCurrU = np.array([])
    for i in range(np.shape(matSpikes)[1]):
        currSpikes = np.squeeze(matSpikes[:,i])
        currData = dfTrial.copy()
        currData['spikes'] = currSpikes
        # Fit the GLM
        model = sm.GLM.from_formula(formula=formula, data=currData, family=sm.families.Poisson()).fit()
        # t value
        regressors = [re.sub(r'\[.*?\]', '', x) for x in model.tvalues.index]
        tv = model.tvalues.values.reshape(1,-1)
        # p value
        pv = model.pvalues.values.reshape(1,-1)
        # t value
        ev = model.params.values.reshape(1,-1)
        # concatenate
        # initialize shape if not
        if np.shape(TvCurrU)[0] == 0:
            TvCurrU = np.empty((0, len(regressors)))
            PvCurrU = np.empty((0, len(regressors)))
            EvCurrU = np.empty((0, len(regressors)))

        TvCurrU = np.concatenate((TvCurrU, tv), axis = 0)
        PvCurrU = np.concatenate((PvCurrU, pv), axis = 0)
        EvCurrU = np.concatenate((EvCurrU, ev), axis = 0)

    return regressors, TvCurrU, PvCurrU, EvCurrU

def shiftedColorMap(cmap, min_val, max_val, name):
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    midpoint = 1.0 - max_val/(max_val + abs(min_val))
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = LinearSegmentedColormap(name, cdict)

    # colormaps.register(cmap=newcmap, force=True)
    return newcmap

def template_reorder(template, right_left, all_channels_int, sample_to_keep = [-30, 60], y_neighbors_to_keep = 3, orginal_loc = False):
    peak_ind = np.argmin(np.min(template, axis=0))
    peak_channel = all_channels_int[peak_ind]
    peak_sample = np.argmin(template[:, peak_ind])  
    peak_group = np.arange(peak_channel - 2*y_neighbors_to_keep, peak_channel + 2*y_neighbors_to_keep + 1, 2)
    if right_left[peak_ind]: # peak is on the right
        sub_peak_channel = peak_channel - 1 
    else: 
        sub_peak_channel = peak_channel + 1 
    sub_group = np.arange(sub_peak_channel - 2*y_neighbors_to_keep, sub_peak_channel + 2*y_neighbors_to_keep + 1, 2)

    # get the reordered template: major column on left, minor column on right
    reordered_template = np.full((2*y_neighbors_to_keep + 1, 2*(sample_to_keep[1] - sample_to_keep[0])), np.nan)
    if peak_sample+sample_to_keep[1] > template.shape[0] or peak_sample+sample_to_keep[0] < 0:
        peak_sample = 89
    for channel_int, channel_curr in enumerate(peak_group):
        if channel_curr in all_channels_int:
            reordered_template[channel_int, :(sample_to_keep[1] - sample_to_keep[0])] = template[(peak_sample+sample_to_keep[0]):(peak_sample+sample_to_keep[1]), np.argwhere(all_channels_int == channel_curr)[0][0]].T

    for channel_int, channel_curr in enumerate(sub_group):
        if channel_curr in all_channels_int:
            reordered_template[channel_int, (sample_to_keep[1] - sample_to_keep[0]):2*(sample_to_keep[1] - sample_to_keep[0])] = template[(peak_sample+sample_to_keep[0]):(peak_sample+sample_to_keep[1]), np.argwhere(all_channels_int == channel_curr)[0][0]].T

    # whether switch back to original location
    if orginal_loc:
        if right_left[peak_ind]:
            reordered_template = reordered_template[:, list(range((sample_to_keep[1] - sample_to_keep[0]), 2*(sample_to_keep[1] - sample_to_keep[0]))) + list(range((sample_to_keep[1] - sample_to_keep[0])))]
    return reordered_template

def convert_values(value):
    if isinstance(value, str):
        try:
            # Convert NumPy-like array strings into proper lists
            if "nan" in value:
                list_array = json.loads(value.replace("nan", "null"))
                return np.array(list_array)
            elif "[" in value and "]" in value and "," in value:
                string_list = value.replace("nan", "np.nan")
                return np.array(ast.literal_eval(string_list))
            elif "[" in value and "]" in value and "," not in value:
                # return np.array(ast.literal_eval(value.replace(" ", ",")))
                return np.fromstring(value.strip("[]"), sep=" ")
            else:
                return ast.literal_eval(value)  # Convert regular lists
        except (ValueError, SyntaxError):
            return value  # If conversion fails, return original string
    return value  # Return value as-is if not a string
    
def load_drift(session, unit_id, data_type='curated'):
    session_dir = session_dirs(session)
    drift_file = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_drift_tbl.csv')
    if not os.path.exists(drift_file):
        return None
    else:
        opto_drift_tbl = pd.read_csv(drift_file)    
        opto_drift_tbl.reset_index(drop=True, inplace=True)
        if len(opto_drift_tbl) == 0:
            return None
        elif unit_id==None:
            return opto_drift_tbl
        elif unit_id in opto_drift_tbl['unit_id'].values:
            unit_drift = opto_drift_tbl[opto_drift_tbl['unit_id'] == unit_id].iloc[0].to_dict()
            # Apply conversion
            converted_data = {key: convert_values(val) for key, val in unit_drift.items()}
            unit_drift = converted_data
            return unit_drift
        else:
            return None

def get_spike_matrix(spike_times, align_time, pre_event=-2, post_event=2, binSize=0.2, stepSize=0.1, avoid_overlap = False, avoid_win = [0, 1.5], kernel = False, tau_rise = None, tau_decay = None):
    if kernel:
        spike_matrix, bin_times = get_spike_matrix_filter(spike_times, align_time, pre_event, post_event, tau_decay = tau_decay, tau_rise= tau_rise, step_size=stepSize, avoid_overlap = avoid_overlap, avoid_win = avoid_win)
        return spike_matrix, bin_times
    bin_times = np.arange(pre_event, post_event, stepSize) - 0.5*stepSize
    spike_matrix = np.zeros((len(align_time), len(bin_times)))
    for i, t in enumerate(align_time):
        for j, b in enumerate(bin_times):
            start = t + b - 0.5*binSize
            end = t + b + 0.5*binSize
            spike_matrix[i, j] = np.sum((spike_times >= start) & (spike_times < end))
            # if this window contains any other events
            if avoid_overlap:
                if any(((align_time+avoid_win[1]) >= start) & (align_time+avoid_win[0] < end) & (align_time != t)):
                    spike_matrix[i, j] = np.nan
    return spike_matrix/binSize, bin_times

def psp_filter(x, tau_rise=0.001, tau_decay=0.02):
    myPSP = (1 - np.exp(-x/tau_rise)) * np.exp(-x/tau_decay)
    return myPSP

def get_spike_matrix_filter(spike_times, align_time, tb, tf, tau_decay = 0.020, tau_rise= 0.001, step_size=0.05, avoid_overlap = False, avoid_win = [0, 1.5]):
    tb_bin = 10*tau_decay
    time_bins = np.arange(tb, tf, step_size)
    fr_mat = np.zeros((len(align_time), len(time_bins)))
    for event_ind, event_time in enumerate(align_time):
        spike_times_aligned = spike_times - event_time
        for bin_ind in range(len(time_bins)):
            curr_t = time_bins[bin_ind]
            t_start = curr_t - tb_bin
            curr_spikes = spike_times_aligned[(spike_times_aligned >= t_start) & (spike_times_aligned < curr_t)]
            if len(curr_spikes) > 0:
                fr_mat[event_ind, bin_ind] = np.sum(psp_filter(curr_t - curr_spikes, tau_rise=tau_rise, tau_decay=tau_decay))
            if avoid_overlap:
                if any(((align_time+avoid_win[1]) >= (curr_t + event_time - tb_bin)) & (align_time+avoid_win[0] < (curr_t + event_time)) & (align_time != event_time)):
                    fr_mat[event_ind, bin_ind] = np.nan
    filter_int = (tau_decay*tau_decay)/(tau_decay + tau_rise)
    fr_mat = fr_mat / filter_int
    return fr_mat, time_bins



def plot_filled_sem(time, y_mat, color, ax, label):
    ax.plot(time, np.nanmean(y_mat, 0), c = color, label = label)
    sem = np.nanstd(y_mat, axis = 0)/np.sqrt(np.shape(y_mat)[0])
    ax.fill_between(time, np.nanmean(y_mat, 0) - sem, np.nanmean(y_mat, 0) + sem, color = color, alpha = 0.25, edgecolor = None)

def plot_raster_rate(
    spike_times,
    align_events, # sorted by certain value
    map_value,
    bins,
    labels,
    colormap,
    fig,
    subplot_spec,
    tb=-2,
    tf=3,
    time_bin = 0.1,
    step_size = 0.1,
    kernel = False,
    tau_rise = 0.001,
    tau_decay = 0.04,
    avoid_overlap = True
):
    n_colors = len(bins)-1
    color_list = [colormap(i / (n_colors - 1)) for i in range(n_colors)]
    """ get spike matrix"""
    # get spike matrix
    currArray, slide_times = get_spike_matrix(spike_times, align_events, 
                                            pre_event=tb, post_event=tf, 
                                            binSize=time_bin, stepSize=step_size, kernel=kernel,
                                            tau_rise=tau_rise, tau_decay=tau_decay, avoid_overlap=avoid_overlap)

    """Plot raster and rate aligned to events"""
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios= [3, 1], subplot_spec=subplot_spec)
    ax1 = fig.add_subplot(nested_gs[0, 0])
    ax2 = fig.add_subplot(nested_gs[1, 0])

    # order events by values
    sort_ind = np.argsort(map_value)
    align_events = align_events[sort_ind]

    df = align.to_events(spike_times, align_events, (tb, tf), return_df=True)
    
    # vertical line at time 0
    ax1.axvline(x=0, c="r", ls="--", lw=1, zorder=1)

    # raster plot
    N_lines = df['event_index'].max() + 1

    # full bar height in y-units
    bar_height =1

    y0 = df.event_index - bar_height / 2
    y1 = df.event_index + bar_height / 2

    ax1.vlines(df.time, y0, y1, color="k", linewidth=1)


    # horizontal line for each type if discrete
    if len(np.unique(map_value)) <= 4:
        discrete_types = np.sort(np.unique(map_value))
    else:
        discrete_types = bins
    
    for val in discrete_types:
        level = np.sum(map_value <= val)
        ax1.axhline(y=level, c="k", ls="--", lw=1)

    ax1.set_title(' '.join(labels))
    ax1.set_xlim(tb, tf)
    ax1.set_ylabel('__'.join(labels))

    # rate plot by binned values

    for bin_ind in range(len(bins)-1): 
        currList = np.where((np.array(map_value)>=bins[bin_ind]) & (np.array(map_value)<bins[bin_ind + 1]))[0]
        if len(currList) > 0:
            M = currArray[currList, :]
            plot_filled_sem(slide_times, M, color_list[bin_ind], ax2, labels[bin_ind])

    ax2.legend()

    ax2.set_title("spike rate")
    ax2.set_xlim(tb, tf)
    ax2.set_xlabel("Time from alignment (s)")

    return fig, ax1, ax2

def plot_rate(
    currArray,
    slide_times, 
    map_value,
    bins,
    labels,
    colormap,
    fig,
    subplot_spec,
    tb = None,
    tf = None,
):
    if tb is None:
        tb = np.min(slide_times)
    if tf is None:
        tf = np.max(slide_times)
    n_colors = len(bins)-1
    color_list = [colormap(i / (n_colors - 1)) for i in range(n_colors)]

    """Plot rate aligned to events"""

    # rate plot by binned values
    ax = fig.add_subplot(subplot_spec)
    for bin_ind in range(len(bins)-1): 
        currList = np.where((np.array(map_value)>=bins[bin_ind]) & (np.array(map_value)<bins[bin_ind + 1]))[0]
        if len(currList) > 0:
            M = currArray[currList, :]
            plot_filled_sem(slide_times, M, color_list[bin_ind], ax, labels[bin_ind])

    ax.legend()

    ax.set_title("spike rate")
    ax.set_xlim(tb, tf)
    ax.set_xlabel("Time from alignment (s)")

    return fig, ax

def regression_rwd(spike_counts, outcome, trials_back = [0, 2], sub_selection = None):
    outcome_matrix = np.zeros((len(outcome), trials_back[1] - trials_back[0] + 1))
    for i in range(len(outcome)):
        for j in range(trials_back[0], trials_back[1] + 1):
            if i-j >=0:
                outcome_matrix[i, j] = outcome[i-j]
            else:
                outcome_matrix[i, j] = np.nan

    outcome_matrix = sm.add_constant(outcome_matrix)
    if sub_selection is not None:
        outcome_matrix = outcome_matrix[sub_selection, :]
        spike_counts = spike_counts[sub_selection]
    lm = sm.OLS(spike_counts, outcome_matrix, missing='drop').fit()
    return lm.params[1:], lm.pvalues[1:], lm.tvalues[1:], lm.conf_int(alpha=0.05)[1:] 

class load_trial_drift:
    def __init__(self, session, data_type):
        """Initialize the object with a DataFrame."""
        session_dir = session_dirs(session)
        drift_tbl_dir = os.path.join(session_dir[f'ephys_dir_{data_type}'], f'{session}_drift_trial_table.csv')
        if not os.path.exists(drift_tbl_dir):
            drift_data = None
        else:
            drift_data = pd.read_csv(drift_tbl_dir)
        self.drift_data = drift_data

    def load_unit(self, unit_id, trial_range = None, cat = None):
        """Load the drift data for a specific unit."""
        """Load the drift data for a specific unit."""
        if self.drift_data is None:
            raise ValueError("Drift data is not available. Please generate drift data first.")
        unit_drift_data = self.drift_data[self.drift_data['unit_id'] == unit_id].sort_values(by='trial_ind')
        if trial_range is not None:
            if isinstance(trial_range, int):
                trial_range = [trial_range]
            elif isinstance(trial_range, list):
                trial_range = np.array(trial_range)
            else:
                raise ValueError("trial_range should be an integer or a list of integers.")
            unit_drift_data = unit_drift_data[unit_drift_data['trial_ind'].isin(trial_range)]
        if cat is not None:
            if cat not in unit_drift_data.columns:
                raise ValueError(f"Category '{cat}' not found in drift data.")
            unit_drift_data = unit_drift_data[cat]
        return unit_drift_data

def correlate_nan(x, y, lag='full'):
    """Calculate correlation while ignoring NaNs."""
    if lag == 'full':
        lag = len(x) - 1
    corrs = np.full((lag + 1,), np.nan)
    for l in range(lag + 1):
        if l==0:
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            corrs[l] = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
        else:
            valid_mask = ~np.isnan(x[:-l]) & ~np.isnan(y[l:])
            if np.any(valid_mask):
                corrs[l] = np.corrcoef(x[:-l][valid_mask], y[l:][valid_mask])[0, 1]
    return corrs

def correlate_nan_bi(x, y, lag='full'):
    """
    Calculate bidirectional cross-correlation between x and y while ignoring NaNs.
    Returns an array of correlation values from -max_lag to +max_lag.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if lag == 'full':
        lag = len(x) - 1

    max_lag = lag

    corrs = np.full((2 * max_lag + 1,), np.nan)
    lags = np.arange(-max_lag, max_lag + 1)

    for i, l in enumerate(lags):
        if l < 0:
            # Shift x forward, y backward
            valid_mask = ~np.isnan(x[-l:]) & ~np.isnan(y[:l])
            if np.any(valid_mask):
                corrs[i] = np.corrcoef(x[-l:][valid_mask], y[:l][valid_mask])[0, 1]
        elif l == 0:
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            if np.any(valid_mask):
                corrs[i] = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
        else:  # l > 0
            # Shift y forward, x backward
            valid_mask = ~np.isnan(x[:-l]) & ~np.isnan(y[l:])
            if np.any(valid_mask):
                corrs[i] = np.corrcoef(x[:-l][valid_mask], y[l:][valid_mask])[0, 1]

    return corrs, lags


def _pearson_lags_fft(x, y, max_lag):
    """
    Per-lag Pearson correlation between x and y for lags 0..max_lag, ignoring NaNs.

    Numerically equivalent to looping np.corrcoef(x[:-l], y[l:]) over lags (i.e. to
    correlate_nan), but evaluates every lag at once with FFTs: O(n log n) rather than
    O(n * max_lag). x is centered internally, which leaves Pearson r unchanged but
    avoids catastrophic cancellation in the per-lag variance terms.

    Parameters:
    x, y : array-like
        Equal-length signals. NaN marks an invalid sample; a lag pair contributes
        only if both of its members are valid.
    max_lag : int
        Largest lag, in samples.

    Returns:
    corrs : ndarray, shape (max_lag + 1,)
        Pearson r at each lag, NaN where a lag has <2 valid pairs or zero variance.
    n_valid : ndarray, shape (max_lag + 1,)
        Number of valid pairs contributing to each lag.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError('x and y must have the same length')
    n = len(x)
    max_lag = int(max_lag)
    if max_lag >= n:
        raise ValueError(f'max_lag ({max_lag}) must be < signal length ({n})')

    mask_x = np.isfinite(x)
    mask_y = np.isfinite(y)
    # centering is a no-op for Pearson r but keeps n*Sxx - Sx**2 well conditioned
    x0 = np.where(mask_x, x - np.nanmean(x), 0.0)
    y0 = np.where(mask_y, y - np.nanmean(y), 0.0)
    mask_x = mask_x.astype(float)
    mask_y = mask_y.astype(float)

    nfft = next_fast_len(n + max_lag + 1)
    spectra = {}

    def _spectrum(key, a):
        if key not in spectra:
            spectra[key] = rfft(a, nfft)
        return spectra[key]

    def _corr(key_a, a, key_b, b):
        """sum_i a[i] * b[i + l] for l = 0..max_lag"""
        prod = np.conj(_spectrum(key_a, a)) * _spectrum(key_b, b)
        return irfft(prod, nfft)[:max_lag + 1]

    n_pairs = _corr('mx', mask_x, 'my', mask_y)
    sum_x = _corr('x', x0, 'my', mask_y)
    sum_y = _corr('mx', mask_x, 'y', y0)
    sum_xx = _corr('xx', x0 ** 2, 'my', mask_y)
    sum_yy = _corr('mx', mask_x, 'yy', y0 ** 2)
    sum_xy = _corr('x', x0, 'y', y0)

    with np.errstate(invalid='ignore', divide='ignore'):
        cov = n_pairs * sum_xy - sum_x * sum_y
        var_x = n_pairs * sum_xx - sum_x ** 2
        var_y = n_pairs * sum_yy - sum_y ** 2
        corrs = cov / np.sqrt(var_x * var_y)
    n_valid = np.round(n_pairs).astype(int)
    corrs[(n_valid < 2) | (var_x <= 0) | (var_y <= 0)] = np.nan
    return corrs, n_valid


def correlate_nan_fft(x, y, lag='full', return_n=False):
    """FFT-based drop-in for correlate_nan: per-lag Pearson r for lags 0..lag."""
    if lag == 'full':
        lag = len(x) - 1
    corrs, n_valid = _pearson_lags_fft(x, y, lag)
    return (corrs, n_valid) if return_n else corrs


def correlate_nan_bi_fft(x, y, lag='full', return_n=False):
    """FFT-based drop-in for correlate_nan_bi: per-lag Pearson r for lags -lag..+lag."""
    if lag == 'full':
        lag = len(x) - 1
    corrs_pos, n_pos = _pearson_lags_fft(x, y, lag)
    corrs_neg, n_neg = _pearson_lags_fft(y, x, lag)
    corrs = np.concatenate([corrs_neg[1:][::-1], corrs_pos])
    lags = np.arange(-lag, lag + 1)
    if return_n:
        return corrs, lags, np.concatenate([n_neg[1:][::-1], n_pos])
    return corrs, lags


def autocorrelation(x, lag):
    n = len(x)
    x = x - np.nanmean(x)
    # result = np.correlate(x, x, mode='full')
    result = correlate_nan(x, x, lag = lag)  # only valid correlations
    # result = result[result.size // 2:]  # keep only second half
    # return result[:lag + 1] / result[0]  # normalize
    return result/result[0]  # normalize

def auto_corr_train(spike_times, bin_size, window_length, rec_start, rec_end):
    """
    Calculate autocorrelation of spike times.
    
    Parameters:
    spike_times : array-like
        Spike times of the unit.
    bin_size : float
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
    counts = np.histogram(spike_times, bins=np.arange(rec_start, rec_end, bin_size))[0]
    lag=int(window_length/bin_size)
    n = len(counts)
    counts = counts - np.nanmean(counts)
    # result = np.correlate(x, x, mode='full')
    result = correlate_nan(counts, counts, lag = lag)  # only valid correlations
    lag_time = np.arange(0, lag + 1) * bin_size
    return result, lag_time

def cross_corr_train(spike_times_x, spike_times_y, bin_size, window_length, rec_start, rec_end):
    """
    Calculate autocorrelation of spike times.
    
    Parameters:
    spike_times : array-like
        Spike times of the unit.
    bin_size : float
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
    counts_x = np.histogram(spike_times_x, bins=np.arange(rec_start, rec_end, bin_size))[0]
    counts_y = np.histogram(spike_times_y, bins=np.arange(rec_start, rec_end, bin_size))[0]
    lag=int(np.round(window_length/bin_size))
    n = len(counts_x)
    counts_x = counts_x - np.nanmean(counts_x)
    counts_y = counts_y - np.nanmean(counts_y)
    # result = np.correlate(x, x, mode='full')
    result, lags = correlate_nan_bi(counts_x, counts_y, lag = lag)  # only valid correlations
    lag_time = np.arange(-lag, lag+1) * bin_size
    return result, lag_time

def cross_corr_train_nogo(spike_times_x, spike_times_y, bin_size, window_length, rec_start, rec_end, go_cue_times, go_cue_period):
    """
    Calculate autocorrelation of spike times.
    
    Parameters:
    spike_times : array-like
        Spike times of the unit.
    bin_size : float
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
    counts_x = np.histogram(spike_times_x, bins=np.arange(rec_start, rec_end, bin_size))[0]
    counts_y = np.histogram(spike_times_y, bins=np.arange(rec_start, rec_end, bin_size))[0]
    counts_x = counts_x.astype(float)
    counts_y = counts_y.astype(float)
    time_bins = np.arange(rec_start, rec_end, bin_size)
    time_starts = time_bins[:-1]
    time_ends = time_bins[1:]
    for go_ind, go_time in enumerate(go_cue_times):
        go_start = go_time
        go_end = go_time + go_cue_period
        # find the indices of the bins that overlaps with the go period
        overlap_indices = np.where((time_starts <= go_end) & (time_ends >= go_start))[0]
        if len(overlap_indices) > 0:
            # set the counts in the overlapping bins to zero
            counts_x[overlap_indices] = np.nan
            counts_y[overlap_indices] = np.nan


    lag=int(np.round(window_length/bin_size))
    n = len(counts_x)
    counts_x = counts_x - np.nanmean(counts_x)
    counts_y = counts_y - np.nanmean(counts_y)
    # result = np.correlate(x, x, mode='full')
    result, lags = correlate_nan_bi(counts_x, counts_y, lag = lag)  # only valid correlations
    lag_time = np.arange(-lag, lag+1) * bin_size
    return result, lag_time

def auto_corr_train_nogo(spike_times, bin_size, window_length, rec_start, rec_end, go_cue_times, go_cue_period):
    """
    Calculate autocorrelation of spike times.
    
    Parameters:
    spike_times : array-like
        Spike times of the unit.
    bin_size : float
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
    counts = np.histogram(spike_times, bins=np.arange(rec_start, rec_end, bin_size))[0]
    counts = counts.astype(float)
    time_bins = np.arange(rec_start, rec_end, bin_size)
    time_starts = time_bins[:-1]
    time_ends = time_bins[1:]
    for go_ind, go_time in enumerate(go_cue_times):
        go_start = go_time
        go_end = go_time + go_cue_period
        # find the indices of the bins that overlaps with the go period
        overlap_indices =(time_starts <= go_end) & (time_ends >= go_start)
        if np.sum(overlap_indices) > 0:
            # set the counts in the overlapping bins to zero
            counts[overlap_indices] = np.nan
    lag=int(window_length/bin_size)
    n = len(counts)
    counts = counts - np.nanmean(counts)
    # result = np.correlate(x, x, mode='full')
    result = correlate_nan(counts, counts, lag = lag)  # only valid correlations
    lag_time = np.arange(0, lag + 1) * bin_size
    return result, lag_time

class load_auto_corr():
    def __init__(self, session, data_type):
        self.session = session
        """Initialize the object with a DataFrame."""
        session_dir = session_dirs(session)
        auto_corr_tbl_dir = os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], f'{session}_{data_type}_auto_corr.pkl')
        if not os.path.exists(auto_corr_tbl_dir):
            auto_corr_data = None
        else:
            # read from pickle file
            with open(auto_corr_tbl_dir, 'rb') as f:
                auto_corr_data = pd.read_pickle(f)
        self.auto_corr_data = auto_corr_data

    def load_unit(self, unit_id):
        """Load the autocorrelation data for a specific unit."""
        if self.auto_corr_data is None:
            return None
        else: 
            unit_auto_corr_data = self.auto_corr_data[self.auto_corr_data['unit'] == unit_id].copy()
            if len(unit_auto_corr_data) == 0:
                unit_auto_corr_data = None
        return unit_auto_corr_data.to_dict(orient='records')[0] if unit_auto_corr_data is not None else None
    
    def plot_unit(self, unit_id):
        """Plot the autocorrelation data for a specific unit."""
        unit_auto_corr_data = self.load_unit(unit_id)
        if unit_auto_corr_data is None:
            print(f"No autocorrelation data found for unit {unit_id}.")
            return None
        else:
            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            ax[0].plot(unit_auto_corr_data['long_lags'][1:], unit_auto_corr_data['auto_corr_long'][1:], label='Long window')
            ax[1].plot(unit_auto_corr_data['short_lags'][1:], unit_auto_corr_data['auto_corr_short'][1:], label='Short window')
            ax[0].set_xlabel('Lag Time (s)')
            ax[1].set_xlabel('Lag Time (s)')
            ax[0].set_ylabel('Autocorrelation')
            plt.suptitle(f'Autocorrelation for Unit {unit_id} session {self.session}')
            plt.show()
            return fig, ax

class load_cross_corr():
    def __init__(self, session, data_type):
        """Initialize the object with a DataFrame."""
        self.session = session
        session_dir = session_dirs(session)
        cross_corr_tbl_dir = os.path.join(session_dir[f'ephys_processed_dir_{data_type}'], f'{session}_{data_type}_cross_corr.pkl')
        if not os.path.exists(cross_corr_tbl_dir):
            cross_corr_data = None
        else:
            # read from pickle file
            with open(cross_corr_tbl_dir, 'rb') as f:
                cross_corr_data = pd.read_pickle(f)
        self.cross_corr_data = cross_corr_data

    def load_units(self, unit_1, unit_2):
        """Load the autocorrelation data for a specific unit."""
        if self.cross_corr_data is None:
            return None
        else: 
            unit_cross_corr_data = self.cross_corr_data[((self.cross_corr_data['unit_1']==unit_1) & (self.cross_corr_data['unit_2']==unit_2))].copy()
            if len(unit_cross_corr_data) == 0:
                unit_cross_corr_data = self.cross_corr_data[((self.cross_corr_data['unit_2']==unit_1) & (self.cross_corr_data['unit_1']==unit_2))].copy()
                if len(unit_cross_corr_data) == 0:
                    unit_cross_corr_data = None
                else:
                    unit_cross_corr_data = unit_cross_corr_data.copy()
                    unit_cross_corr_data['unit_1'] = unit_2
                    unit_cross_corr_data['unit_2'] = unit_1
                    unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_long'] = np.array(np.flip(unit_cross_corr_data['cross_corr_long'].values[0], axis=0))
                    unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_short'] = np.array(np.flip(unit_cross_corr_data['cross_corr_short'].values[0], axis=0))
                    unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_long_nogo'] = np.array(np.flip(unit_cross_corr_data['cross_corr_long_nogo'].values[0], axis=0))
                    unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_short_nogo'] = np.array(np.flip(unit_cross_corr_data['cross_corr_short_nogo'].values[0], axis=0))
            elif len(unit_cross_corr_data) == 1:
                unit_cross_corr_data = unit_cross_corr_data.copy()
                unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_long'] = np.array(unit_cross_corr_data['cross_corr_long'].values[0])
                unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_short'] = np.array(unit_cross_corr_data['cross_corr_short'].values[0])
                unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_long_nogo'] = np.array(unit_cross_corr_data['cross_corr_long_nogo'].values[0])
                unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_short_nogo'] = np.array(unit_cross_corr_data['cross_corr_short_nogo'].values[0])
            else:
                print(f"Multiple cross-correlation entries found for units {unit_1} and {unit_2}. Please check the data.")
                unit_cross_corr_data = unit_cross_corr_data.iloc[[0]].copy()
                unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_long'] = np.array(unit_cross_corr_data['cross_corr_long'].values[0])
                unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_short'] = np.array(unit_cross_corr_data['cross_corr_short'].values[0])
                unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_long_nogo'] = np.array(unit_cross_corr_data['cross_corr_long_nogo'].values[0])
                unit_cross_corr_data.at[unit_cross_corr_data.index[0], 'cross_corr_short_nogo'] = np.array(unit_cross_corr_data['cross_corr_short_nogo'].values[0])
        return unit_cross_corr_data.to_dict(orient='records')[0] if unit_cross_corr_data is not None else None
    
    def plot_units(self, unit_1, unit_2):
        """Plot the autocorrelation data for a specific unit."""
        unit_cross_corr_data = self.load_units(unit_1, unit_2)
        if unit_cross_corr_data is None:
            print(f"No cross-correlation data found for units {unit_1} and {unit_2}.")
            return None
        else:
            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            ax[0].plot(unit_cross_corr_data['long_lags'], unit_cross_corr_data['cross_corr_long'], label='all')
            ax[0].plot(unit_cross_corr_data['long_lags'], unit_cross_corr_data['cross_corr_long_nogo'], label='Go cue removed')
            ax[1].plot(unit_cross_corr_data['short_lags'], unit_cross_corr_data['cross_corr_short'], label='all')
            ax[1].plot(unit_cross_corr_data['short_lags'], unit_cross_corr_data['cross_corr_short_nogo'], label='Go cue removed')
            ax[0].set_xlabel('Lag Time (s)')
            ax[1].set_xlabel('Lag Time (s)')
            ax[0].set_ylabel('Cross-correlation')
            ax[0].legend()
            ax[1].legend()
            plt.suptitle(f'Cross-correlation for Units {unit_1} & {unit_2} session {self.session}')
            plt.show()
            return fig, ax

def make_summary_unit_tbl(session): # this is for hopkins data
    session_dir = session_dirs(session)
    nwb = load_nwb_from_filename(session_dir['nwb_dir_raw'])
    if nwb.units is None:
        print(f"No units found in NWB file for session {session}.")
        return None
    nwb_units = load_nwb_from_filename(session_dir['nwb_dir_raw']).units[:] 
    here = os.path.dirname(__file__)
    tbl_columns_file = os.path.join(here, 'summary_col_list.json')
    with open(tbl_columns_file, 'r') as f:
        example_tbl_cols = json.load(open("summary_col_list.json"))
    unit_summary = pd.DataFrame(columns= example_tbl_cols)
    for row_ind, row in nwb_units.iterrows():
        for col in example_tbl_cols:
            if col in row.index:
                unit_summary.at[row_ind, col] = row[col]
            else:
                unit_summary.at[row_ind, col] = np.nan
        unit_summary.loc[row_ind, 'peak'] = -row['peak'] if 'peak' in row.index else np.nan
        # convert array columns to single values
        array_to_row = ['bl_max_p', 'p_max', 'p_mean', 'lat_max_p', 'euc_max_p', 'pass_count']
        for col in array_to_row:
            if col in row.index:
                unit_summary.at[row_ind, col] = row[col][0] if isinstance(row[col], np.ndarray) else row[col]
            else:
                unit_summary.at[row_ind, col] = np.nan
        # find corresponding columns in summary and nwb units
        pairs_sum = ['amp', 'ks_unit_id', 'isi_violations_ratio', 'waveform_mean']
        pairs_nwb = ['amplitude', 'unit_id', 'isi_violation_ratio', 'mat_wf_opt']
        for col_sum, col_nwb in zip(pairs_sum, pairs_nwb):
            if col_nwb in row.index:
                unit_summary.at[row_ind, col_sum] = row[col_nwb]
            else:
                unit_summary.at[row_ind, col_sum] = np.nan
        # change units
        change_units = ['LC_range_top', 'LC_range_bottom', 'y_loc']
        for col in change_units:
            if col in row.index:
                unit_summary.at[row_ind, col] = row[col] * 1e3 if isinstance(row[col], float) else row[col]
            else:
                unit_summary.at[row_ind, col] = np.nan
        # check if unit has spike time
        if len(unit_summary['spike_times'][row_ind]) == 0:
            file = os.path.join(session_dir['session_dir'], f"{row['unit_id']}.txt")
            unit_summary.at[row_ind, 'spike_times'] = np.array(np.loadtxt(file)/1000000)
    # save into pikle
    summary_path = os.path.join(session_dir['opto_dir_curated'], f'{session}_curated_soma_opto_tagging_summary.pkl')
    with open(summary_path, 'wb') as f:
        unit_summary.to_pickle(f)

def get_score(X_design, y_target, criterion='aic'):
    model = sm.GLM(y_target, sm.add_constant(X_design)).fit()
    score = model.aic if criterion == 'aic' else model.bic
    return score, model

    
def stepwise_glm(X, y, forced_vars, candidate_vars, criterion='aic', verbose=True):
    """
    Perform stepwise regression (GLM) with forced inclusion of some variables.

    Parameters:
    - X: DataFrame of all regressors
    - y: target variable (Series)
    - forced_vars: list of variables to always include
    - candidate_vars: list of variables to select from
    - criterion: 'aic' or 'bic'
    - verbose: whether to print step info

    Returns:
    - final_model: fitted statsmodels GLM
    - selected_vars: list of all variables in final model
    """
    


    included = forced_vars.copy()
    optional = candidate_vars.copy()
    
    current_X = X[forced_vars].copy()
    best_score, best_model = get_score(current_X, y, criterion='aic')

    changed = True
    while changed:
        changed = False

        # Try adding one variable
        scores_with_addition = []
        for var in optional:
            if var not in included:
                X_try = X[included + [var]]
                score, _ = get_score(X_try, y)
                scores_with_addition.append((score, var))
        scores_with_addition.sort()
        if scores_with_addition:
            best_new_score, best_var = scores_with_addition[0]
            if best_new_score < best_score:
                included.append(best_var)
                optional.remove(best_var)
                best_score = best_new_score
                changed = True
                if verbose:
                    print(f'Added: {best_var}, {criterion.upper()}: {best_score:.2f}')

        # Try removing one variable (not from forced)
        scores_with_removal = []
        for var in included:
            if var not in forced_vars:
                X_try = X[[v for v in included if v != var]]
                score, _ = get_score(X_try, y)
                scores_with_removal.append((score, var))
        scores_with_removal.sort()
        if scores_with_removal:
            best_new_score, worst_var = scores_with_removal[0]
            if best_new_score < best_score:
                included.remove(worst_var)
                optional.append(worst_var)
                best_score = best_new_score
                changed = True
                if verbose:
                    print(f'Removed: {worst_var}, {criterion.upper()}: {best_score:.2f}')

    final_model = sm.GLM(y, sm.add_constant(X[included])).fit()
    R2_final = r2_score(y, final_model.fittedvalues)
    forced_model = sm.GLM(y, sm.add_constant(X[forced_vars])).fit()
    R2_forced = r2_score(y, forced_model.fittedvalues)
    
    return forced_model, final_model, included, R2_final, R2_forced

def add_interactions(df, interaction_pairs):
    for var1, var2 in interaction_pairs:
        name = f"{var1}:{var2}"
        df[name] = df[var1] * df[var2]
    return df
