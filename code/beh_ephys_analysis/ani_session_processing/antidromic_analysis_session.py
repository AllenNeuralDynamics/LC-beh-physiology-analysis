import os, sys
# Resolve code/beh_ephys_analysis (the folder containing `utils`) relative to this
# file's location, so imports work no matter where the repo is checked out.
_anchor = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.abspath(os.getcwd())
while _anchor != os.path.dirname(_anchor):
    _beh_ephys_root = os.path.join(_anchor, "code", "beh_ephys_analysis")
    if os.path.isdir(os.path.join(_beh_ephys_root, "utils")):
        if _beh_ephys_root in sys.path:
            sys.path.remove(_beh_ephys_root)
        sys.path.insert(0, _beh_ephys_root)
        break
    _anchor = os.path.dirname(_anchor)
from utils.capsule_migration import CAPSULE_ROOT
from json import load
from scipy.stats import wilcoxon
# Scientific libraries
import numpy as np
import pandas as pd
import utils.analysis_funcs as af
import utils.plotting_funcs as pf

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from utils.beh_functions import session_dirs, get_unit_tbl
from utils.ephys_functions import load_drift
import os
from utils.beh_functions import get_unit_tbl
from utils.plot_utils import combine_pdf_big
from scipy.stats import mannwhitneyu

from scipy.stats import fisher_exact
from scipy.stats import binomtest as binom_test



def remove_spikes_during_laser_pulse(int_event_locked_timestamps, duration):
    """
    Remove spikes that occur during the laser pulse period.

    Parameters:
        int_event_locked_timestamps (list of np.ndarray): Event-locked spike times for each trial.
        duration (float): Duration of the laser pulse in milliseconds.

    Returns:
        list of np.ndarray: Spike times with laser pulse period spikes removed.
    """
    for i, arr in enumerate(int_event_locked_timestamps):
        # Remove spikes during the laser pulse
        int_event_locked_timestamps[i] = arr[(arr < 0) | (arr > (duration / 1000))]
    return int_event_locked_timestamps

def opto_tagging_response(int_event_locked_timestamps, base_window, roi_window):
    """
    Test whether a unit has a significant response to optogenetic stimulation using Wilcoxon test.

    Parameters:
        int_event_locked_timestamps (list of np.ndarray): Event-locked spike times for each trial.
        base_window (tuple): Time window (start, end) in seconds for baseline period.
        roi_window (tuple): Time window (start, end) in seconds for response region of interest.

    Returns:
        float: P-value from Wilcoxon signed-rank test, or np.nan if insufficient data.
    """
    base_counts = []
    roi_counts = []
    for spike_times in int_event_locked_timestamps:
        if spike_times.size > 0:
            base_count = np.sum((spike_times >= base_window[0]) & (spike_times < base_window[1]))
            roi_count = np.sum((spike_times >= roi_window[0]) & (spike_times < roi_window[1]))
            base_counts.append(base_count)
            roi_counts.append(roi_count)

    base_counts = np.array(base_counts)
    roi_counts = np.array(roi_counts)

    # ✅ Only run test if we have at least 2 samples
    if len(base_counts) >= 2 and len(roi_counts) >= 2:
        try:
            stat, p_value = wilcoxon(base_counts, roi_counts)
            return p_value
        except ValueError:
            return np.nan
    else:
        return np.nan

def antidromic_latency_jitter(int_event_locked_timestamps):
    """
    Calculate antidromic response latency and jitter from spike times.

    Parameters:
        int_event_locked_timestamps (list of np.ndarray): Event-locked spike times for each trial.

    Returns:
        tuple: (antidromic_latency, antidromic_jitter) in seconds. Latency is the peak time of the
               smoothed PSTH, and jitter is the full width at half maximum (FWHM).
    """
    from scipy.ndimage import gaussian_filter1d
    pulse_duration = 0.005 # ms    
    first_spikes_after_light = np.array([arr[arr > pulse_duration][0] for arr in int_event_locked_timestamps if np.any(arr > pulse_duration)])
    histogram, bins = np.histogram(first_spikes_after_light, bins=np.arange(pulse_duration, 0.1, 0.0005))
    # Convolve the histogram with a Gaussian kernel
    sigma = 0.5 / 1000 / (bins[1] - bins[0])  # Convert sigma to bin units    
    smoothed_psth = gaussian_filter1d(histogram, sigma)
    
    # Find the peak of the smoothed PSTH
    peak_index = np.argmax(smoothed_psth)
    peak_time = bins[peak_index]
    peak_value = smoothed_psth[peak_index]

    # Calculate FWHM (Full Width at Half Maximum)
    half_max = peak_value / 2
    above_half_max = np.where(smoothed_psth >= half_max)[0]
    fwhm = bins[above_half_max[-1]] - bins[above_half_max[0]]

    antidromic_latency = peak_time
    antidromic_jitter = fwhm
    return antidromic_latency, antidromic_jitter


def plot_opto_responses(unit_tbl, event_ids):
    """
    Plot optogenetic responses for units across different stimulation sites.

    Parameters:
        unit_tbl (pd.DataFrame): Table containing unit information including unit_id and spike_times.
        event_ids (pd.DataFrame): Table containing optogenetic stimulation events with columns for
                                  emission_location, power, type, time, duration, num_pulses, pulse_interval.

    Returns:
        matplotlib.figure.Figure: Figure containing raster plots, PSTHs, and antidromic analysis for all units.
    """
    # Filter for opto tagged units
    # opto_criteria = (unit_tbl['opto_pass'] == True) & (unit_tbl['default_qc'] == True)
    opto_units = unit_tbl["unit_id"].to_list()
    print(f"Number of opto units: {len(opto_units)}")

    
    # Unique values
    sites = list(np.unique(event_ids.emission_location))
    powers = list(np.unique(event_ids.power))
    trial_types = np.unique(event_ids.type)

    
    # Settings
    prepost = 'post'
    num_sites = len(sites)
    num_units = len(opto_units)

    # Create one figure for all units × sites (3 rows per unit: raster + PSTH + antidromic raster)
    fig_height_per_unit = 10
    fig = plt.figure(figsize=(num_sites * 4, num_units * fig_height_per_unit))   
    gs = gridspec.GridSpec(6 * num_units, num_sites, height_ratios=[3, 6, 2, 0.5, 1, 1] * num_units, hspace=0.8)

    # Loop through units
    for u_idx, unit_id in enumerate(opto_units):
        # if unit_id == 244:
        unit_spike_times = unit_tbl[unit_tbl['unit_id'] == unit_id]['spike_times'].values[0]

        for i, site in enumerate(sites):
            if site == 'surface_LC':
                # Define windows
                base_window = (-0.02, 0)
                roi_window = (0, 0.02)    
            else:
                # Define windows
                base_window = (-0.02, 0)
                roi_window = (0.03, 0.05)    


            # Filter trials
            tag_trials = event_ids.query('site == @site and pre_post == @prepost')
            if tag_trials.empty:
                prepost = 'pre'
                tag_trials = event_ids.query('site == "surface_LC" and pre_post == @prepost')
            max_power = tag_trials.power.max()
            tag_trials = tag_trials.query('power == @max_power')
            if tag_trials.empty:
                continue

            # Stimulation parameters
            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]

            # Time window
            time_range_raster = np.array([-100 / 1000, 70 / 1000])
            this_event_timestamps = tag_trials.time.tolist()

            int_event_locked_timestamps = []
            pulse_nums = []

            for pulse_num in range(num_pulses):
                time_shift = pulse_num * (duration + pulse_interval) / 1000
                this_time_range = time_range_raster + time_shift

                this_locked = af.event_locked_timestamps(
                    unit_spike_times, this_event_timestamps, this_time_range, time_shift=time_shift
                )
                # Remove spikes during the laser pulse
                int_event_locked_timestamps.extend(this_locked)
                pulse_nums.extend([pulse_num + 1] * len(this_locked))            

            int_event_locked_timestamps = remove_spikes_during_laser_pulse(int_event_locked_timestamps, duration)
            # Raster plot
            ax_raster = fig.add_subplot(gs[5 * u_idx, i])
            pf.raster_plot(int_event_locked_timestamps, time_range_raster, cond_each_trial=pulse_nums, ms=100, ax=ax_raster)
            p_val = opto_tagging_response(int_event_locked_timestamps, base_window, roi_window)


            # Add laser pulse patch
            yLims = np.array(ax_raster.get_ylim())
            laser_color = 'tomato'
            rect = patches.Rectangle((0, yLims[0]), duration / 1000, yLims[1] - yLims[0],
                                    linewidth=1, edgecolor=laser_color, facecolor=laser_color,
                                    alpha=0.2, clip_on=True)
            ax_raster.add_patch(rect)

            # Raster axis settings
            if i == 0:
                ax_raster.set_ylabel(f'Unit {unit_id}\nPulse #')
            else:
                ax_raster.set_yticklabels([])

            ax_raster.set_xlim(time_range_raster)
            if u_idx == num_units - 1:
                ax_raster.set_xlabel('Time (s)')

            if p_val < 0.05:
                text_color = 'red'
                ax_raster.set_title(f'{site}, {max_power} mW, p:{p_val:.3f}', color=text_color)
            else:
                ax_raster.set_title(f'{site}, {max_power} mW, p:{p_val:.3f}', color='black')

            # Add laser pulse aligned but sorted by spike times
            # Antidromic raster plot
            ax_antidromic = fig.add_subplot(gs[5 * u_idx + 1, i], sharex=ax_raster)
            sorted_data = sorted(int_event_locked_timestamps, key=lambda x: (len(x) == 0, x[0] if len(x) > 0 else np.inf))
            pf.raster_plot(sorted_data, time_range_raster)
            # pf.raster_plot(sorted_data, time_range_raster, ax=ax_antidromic)

            yLims = np.array(ax_antidromic.get_ylim())
            rect = patches.Rectangle((0, yLims[0]), duration / 1000, yLims[1] - yLims[0],
                                    linewidth=1, edgecolor='tomato', facecolor='tomato',
                                    alpha=0.2, clip_on=False)
            ax_antidromic.add_patch(rect)

            ax_antidromic.set_xlim(time_range_raster)
            if u_idx == num_units - 1:
                ax_antidromic.set_xlabel('Time (s)')

            if i == 0:
                ax_antidromic.set_ylabel('Sorted Trials')
            else:
                ax_antidromic.set_yticklabels([])

            # PSTH plot
            ax_psth = fig.add_subplot(gs[5 * u_idx + 2, i], sharex=ax_raster)
            psth, _, bins = pf.psth(int_event_locked_timestamps, time_range_raster, bin_size=0.003, smooth_window_size=3)
            antidromic_latency, antidromic_jitter = antidromic_latency_jitter(int_event_locked_timestamps)
            if antidromic_jitter > antidromic_latency:
                antidromic_jitter = antidromic_latency
            ax_psth.plot(bins, psth, color='k')
            if i == 0:
                ax_psth.set_ylabel('PSTH')
            else:
                ax_psth.set_yticklabels([])
            ax_psth.set_xlim(time_range_raster)
            if u_idx == num_units - 1:
                ax_psth.set_xlabel('Time (s)')
            # Add antidromic latency line
            # if antidromic_latency is not None:
                # ax_psth.axvline(antidromic_latency, color='red', linestyle='--', label='Antidromic latency')
                # ax_psth.axvline(antidromic_latency + antidromic_jitter, color='orange', linestyle='--', label='Antidromic jitter')
                # ax_psth.axvline(antidromic_latency - antidromic_jitter, color='orange', linestyle='--')
                # ax_psth.legend()
            # Add title to PSTH
            # ax_psth.set_title(f'PSTH for {site}, {max_power} mW\nAntidromic latency: {antidromic_latency:.3f} s, Jitter: {antidromic_jitter:.3f} s')
            # Add p(resp)-first spike latency
            spike_time_range = (20, 50)
            bin_num=100
            # Convert spike time range to seconds
            spike_time_range_sec = (spike_time_range[0] / 1000, spike_time_range[1] / 1000)
            first_post_stim_spike_times = []
            
            for spike_times in sorted_data:
                if spike_times.size > 0 and np.any(spike_times > 0):
                    first_spike = spike_times[spike_times > 0][0]
                    first_post_stim_spike_times.append(first_spike)
                    
                # Check if first spikes fall within the antidromic window
                is_antidromic_spike = (spike_times >= spike_time_range_sec[0]/1000) & \
                                (spike_times <= spike_time_range_sec[1]/1000)
            
            if not first_post_stim_spike_times:
                continue
            first_post_stim_spike_times = np.array(first_post_stim_spike_times)
            # Calculate antidromic spike probability
            antidromic_spike_prob = np.sum(is_antidromic_spike) / len(first_post_stim_spike_times)
            # Find peak latency (for information only)
            hist, bin_edges = np.histogram(first_post_stim_spike_times, bins=bin_num)
            peak_x = bin_edges[np.argmax(hist)]
            antidromic_latency = peak_x
            ax_latency = fig.add_subplot(gs[5 * u_idx + 3, i])
            ax_latency.plot(bin_edges[:-1], hist, color='k')
            ax_latency.axvline(antidromic_latency, color='r', linestyle='--', label=f'Peak: {antidromic_latency:.3f} s')
            ax_latency.set_xlabel('First post-stimulus spike time (s)')
            ax_latency.set_ylabel('Count')
            ax_latency.set_title(f'Prob: {antidromic_spike_prob:.2f}; Lat: {antidromic_latency*1000:.1f} ms')
            # antidromic test
            num_trials = len(sorted_data)
            last_ortho_spike_times = np.full(num_trials, np.nan)
            collision_flags = np.ones(num_trials, dtype=int)
            for spike_i, spike_times in enumerate(sorted_data):
                if spike_times.size == 0:
                    continue
                
                ortho_mask = spike_times <= (antidromic_latency - antidromic_jitter)
                anti_mask = (antidromic_latency - antidromic_jitter < spike_times) & (spike_times < antidromic_latency + antidromic_jitter)
                
                ortho_spikes = spike_times[ortho_mask]
                anti_spikes = spike_times[anti_mask]
                
                if ortho_spikes.size > 0:
                    last_ortho_spike_times[spike_i] = ortho_spikes[-1]  # Only keep the last orthodromic spike
                
                if anti_spikes.size > 0:
                    collision_flags[spike_i] = 0  # If antidromic spike exists, collision = 0 (no collision)
            antidromic_df = pd.DataFrame({
                'trial_num': np.arange(num_trials),
                'last_orthodromic_spike_time': last_ortho_spike_times,
                'collision': collision_flags
            })

            # Step 4: ROI selection and plot
            roi_df = antidromic_df.dropna(subset=['last_orthodromic_spike_time'])  # Keep only trials with orthodromic spikes
            x = (roi_df['last_orthodromic_spike_time'].values - antidromic_latency) * 1000  # in ms
            y = 1 - roi_df['collision'].values  # 1 = successful antidromic spike, 0 = collision
            
            # Define windows (you can adjust these)
            early_window = (-100, -70)  # in ms
            near_latency_window = (-30, 0)

            # Select data in each window
            early_indices = (x >= early_window[0]) & (x < early_window[1])
            near_latency_indices = (x >= near_latency_window[0]) & (x <= near_latency_window[1])

            early_probs = y[early_indices]
            near_latency_probs = y[near_latency_indices]

            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(early_probs, near_latency_probs, alternative='greater')
            # Plot binned averages
            result, edges = Avg_y_over_x(x, y, bin_size=10)
            ax_compare = fig.add_subplot(gs[5 * u_idx + 4, i])
            ax_compare.errorbar(result['x'], result['y'], yerr=result['sem'], fmt='o-', label='Average ± SEM')
            ax_compare.set_xlabel('Time from antidromic latency (ms)')
            ax_compare.set_ylabel('P(antidromic spike)')
            ax_compare.set_title(f'Collision Test: p-value: {p_value:.2e}')  # <- use f-string
            ax_compare.set_xlim([-100, 0])
            ax_compare.set_ylim([-0.05, 1.05])
            # plt.legend()


    # Final layout adjustments
    fig.tight_layout()    
    # plt.show()
    return fig

def compute_opto_responses(unit_tbl, event_ids, spiketimes, session_id):
    """
    Compute optogenetic response statistics for opto-tagged units.

    Parameters:
        unit_tbl (pd.DataFrame): Table containing unit information with opto_pass and default_qc columns.
        event_ids (pd.DataFrame): Table containing optogenetic stimulation events.
        spiketimes (dict): Dictionary mapping unit_id to spike times array.
        session_id (str): Session identifier.

    Returns:
        list of dict: List of dictionaries containing unit_id, site, and p_value for each unit-site combination.
    """
    results = []

    # Filter for opto tagged units
    opto_criteria = (unit_tbl['opto_pass'] == True) & (unit_tbl['default_qc'] == True)
    opto_units = unit_tbl[opto_criteria]["unit_id"].to_list()
    print(f"Session {session_id}: {len(opto_units)} opto-tagged units")

    
    # Unique values
    sites = list(np.unique(event_ids.emission_location))
    powers = list(np.unique(event_ids.power))
    trial_types = np.unique(event_ids.type)
    
    # Settings
    prepost = 'post'
    num_sites = len(sites)
    num_units = len(opto_units)

   
    # Loop through units
    for u_idx, unit_id in enumerate(opto_units):
        # if unit_id == 244:
        unit_spike_times = spiketimes[unit_id]

        for i, site in enumerate(sites):
            base_window = (-0.02, 0)
            roi_window = (0, 0.02) if site == 'surface_LC' else (0.02, 0.06)


            # Filter trials
            tag_trials = event_ids.query('site == @site and pre_post == @prepost')
            max_power = tag_trials.power.max()
            tag_trials = tag_trials.query('power == @max_power')
            if tag_trials.empty:
                continue

            # Stimulation parameters
            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]

            # Time window
            time_range_raster = np.array([-100 / 1000, 70 / 1000])
            this_event_timestamps = tag_trials.time.tolist()

            int_event_locked_timestamps = []
            pulse_nums = []

            for pulse_num in range(num_pulses):
                time_shift = pulse_num * (duration + pulse_interval) / 1000
                this_time_range = time_range_raster + time_shift

                this_locked = af.event_locked_timestamps(
                    unit_spike_times, this_event_timestamps, this_time_range, time_shift=time_shift
                )
                int_event_locked_timestamps.extend(this_locked)
                pulse_nums.extend([pulse_num + 1] * len(this_locked))

            # Perform significance test
            if len(int_event_locked_timestamps) != 0:
                p_val = opto_tagging_response(int_event_locked_timestamps, base_window, roi_window)
            if not np.isnan(p_val):
                results.append({
                    'unit_id': unit_id,
                    'site': site,
                    'p_value': round(p_val, 3)
                })
    return results

def Avg_y_over_x(x, y, bin_size):
    """
    Compute the average and standard error of y values binned by x.

    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.
    bin_size (float): The size of the bins for grouping x values.

    Returns:
    pd.DataFrame: A DataFrame with columns for bin centers (x), average y values (y), and standard error of the mean (sem).
    """
    # Calculate bin edges based on the bin size    
    edges = np.arange(np.min(x), np.max(x) + bin_size, bin_size)
    
    # Digitize x into bins
    x_index = np.digitize(x, edges) - 1  # Bin indices for each x value
    
    # Calculate bin centers
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    # Create a DataFrame to store results
    Avg = pd.DataFrame({'x': bin_centers, 'y': np.nan, 'sem': np.nan})

    # Compute mean and standard error for each bin
    for i in range(len(bin_centers)):
        # Mask y values corresponding to the current bin
        bin_y_values = y[x_index == i]
        if bin_y_values.size > 0:
            Avg.at[i, 'y'] = np.nanmean(bin_y_values)
            Avg.at[i, 'sem'] = np.nanstd(bin_y_values) / np.sqrt(bin_y_values.size)

    return Avg, edges

def analyze_antidromic_responses(session_id, data_type ='curated', plot=False, tier_cat = False):
    """
    Analyze antidromic responses for a given set of opto-tagged units.

    Performs comprehensive antidromic identification analysis including:
    - Optogenetic response significance testing (Wilcoxon)
    - Latency and jitter calculation from smoothed PSTH
    - Collision test to verify antidromic spikes (Fisher's exact and binomial tests)
    - Regression analysis for auto-inhibition, collision, and antidromic effects
    - Tier categorization (Tier 1: confirmed antidromic, Tier 2: likely antidromic, Tier 3: responsive)

    Analysis windows:
    - For surface_LC: baseline [-20, 0]ms, response [0, 20]ms
    - For other sites: baseline [-20, 0]ms, response [20, 60]ms

    Parameters:
        session_id (str): Session identifier.
        data_type (str): Type of data to use ('curated' or 'raw').
        plot (bool): If True, plot collision raster for each unit and site (default: False).
        tier_cat (bool): If True, perform tier categorization based on antidromic criteria (default: False).

    Returns:
        pd.DataFrame: Pivoted DataFrame with columns for each site containing:
            - opto_p_val: P-value for optogenetic response
            - antidromic_latency: Peak latency of first spike after stimulus
            - jitter: Full width at half maximum of latency distribution
            - median_first_spike_latency: Median latency of first spike
            - collision_pvalue: P-value from Fisher's exact test for collision
            - collision_pbinom: P-value from binomial test
            - pre_boundary_prob: Probability of antidromic spike before collision boundary
            - post_boundary_prob: Probability of antidromic spike after collision boundary
            - p_auto_inhi, t_auto_inhi: Auto-inhibition regression statistics
            - p_collision, t_collision: Collision interaction regression statistics
            - p_antidromic, t_antidromic: Direct antidromic effect regression statistics
            - int_event_locked_timestamps: Raw event-locked spike times
            - table, oddsratio, collision_boundary, Avg, edges: Collision test details
            - last_orthodromic_spike_time: Times of last orthodromic spikes
        If tier_cat=True, also includes tier columns per site.
        Returns None if no opto units found or analysis fails.
    """
    session_dir = session_dirs(session_id)
    opto_data_folder = session_dir[f'opto_dir_{data_type}']
    unit_tbl = get_unit_tbl(session_id, data_type)
    if unit_tbl is None:
        return None
    opto_units = unit_tbl.query("opto_pass == True & default_qc == True").copy()
    del unit_tbl
    print (f"opto units: {len(opto_units)}")

    opto_event_file = os.path.join(opto_data_folder, f'{session_id}_opto_session.csv')
    event_ids = pd.read_csv(opto_event_file) 

    if len(opto_units)<1:
        print("No opto units found.")
        return pd.DataFrame()

    antidromic_results = []
    time_range_raster = np.array([-0.15, 0.15])  # in seconds
    for unit_ind, row in opto_units.iterrows():
        unit_spike_times = row['spike_times']
        unit_id = row['unit_id']
        # print(unit_id)
        unit_drift = load_drift(session_id, unit_id, data_type=data_type)
        event_ids_curr = event_ids.copy()
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                event_ids_curr = event_ids_curr[event_ids_curr['time']>=unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                event_ids_curr = event_ids_curr[event_ids_curr['time']<=unit_drift['ephys_cut'][1]]
        if len(event_ids_curr)==0:
            print(f"No opto events for unit {unit_id} after applying drift cuts.")
            continue
        for site in event_ids_curr['site'].unique():    
            if isinstance(site, str):
                if 'Failed' in site:
                    continue

            # print(site)
            
            tag_trials = event_ids_curr.query('site == @site and pre_post == "post"')
            
            if site == 'surface':
                site = 'surface_LC'  # Treat 'surface' as 'surface_LC' for consistency
            
            # Set analysis windows
            if site != 'surface_LC':
                base_window = (-0.02, 0)
                roi_window = (0, 0.02)
            else:
                base_window = (-0.02, 0)
                roi_window = (0.02, 0.06)
                
            if tag_trials.empty:
                continue

            max_power = tag_trials['power'].max()
            tag_trials = tag_trials.query('power == @max_power')
            
            duration = tag_trials.duration.iloc[0]
            num_pulses = tag_trials.num_pulses.iloc[0]
            pulse_interval = tag_trials.pulse_interval.iloc[0]
            this_event_timestamps = tag_trials.time.tolist()

            # Collect event-locked spike times
            int_event_locked_timestamps = []
            for pulse_num in range(num_pulses):
                time_shift = pulse_num * (duration + pulse_interval) / 1000
                this_time_range = time_range_raster + time_shift
                this_locked = af.event_locked_timestamps(
                    unit_spike_times, this_event_timestamps, this_time_range, time_shift=time_shift
                )
                int_event_locked_timestamps.extend(this_locked)
            # Remove spikes during the laser pulse
            int_event_locked_timestamps = remove_spikes_during_laser_pulse(int_event_locked_timestamps, duration)

            # Optional: plot collision raster
            if plot:
                try:
                    plot_collision_raster(
                        int_event_locked_timestamps=int_event_locked_timestamps,
                        time_range_raster=time_range_raster
                    )
                except Exception as e:
                    print(f"Plotting failed for unit {unit_id}, site {site}: {e}")


            # Statistical analysis
            opto_p_val = opto_tagging_response(int_event_locked_timestamps, base_window, roi_window)
            # Latency and jitter
            antidromic_latency, antidromic_jitter = antidromic_latency_jitter(int_event_locked_timestamps)
            first_spikes_after_light = np.array([arr[arr > 0][0] for arr in int_event_locked_timestamps if np.any(arr > 0)])
            median_first_spikes_after_light = np.median(first_spikes_after_light)

            result = {                
                'unit_id': unit_id,
                'site': site,
                'opto_p_val': opto_p_val,
                'int_event_locked_timestamps': int_event_locked_timestamps,
                'median_first_spike_latency': median_first_spikes_after_light,
                'antidromic_latency': antidromic_latency,
                'jitter': antidromic_jitter
            }  

            # Collision test for non-surface_LC sites
            if site != 'surface_LC':
                collision_results = collision_test(
                    int_event_locked_timestamps, antidromic_latency, bin_size=10, antidromic_jitter=0.005, plot=False
                )
                # add regression analysis
                int_event_locked_timestamps_sham = []
                this_event_timestamps_sham = np.linspace(np.min(tag_trials.time.values)+1, np.max(tag_trials.time.values)-1, max(len(this_event_timestamps), 50))
                for pulse_num in range(num_pulses):
                    time_shift = pulse_num * (duration + pulse_interval) / 1000
                    this_time_range = time_range_raster + time_shift
                    this_locked = af.event_locked_timestamps(
                        unit_spike_times, this_event_timestamps_sham, this_time_range, time_shift=time_shift
                    )
                    int_event_locked_timestamps_sham.extend(this_locked)
                int_event_locked_timestamps_sham = remove_spikes_during_laser_pulse(int_event_locked_timestamps_sham, duration)
                # if both lock timestamps are empty before 0, skip
                if all(len(arr[arr<0])==0 for arr in int_event_locked_timestamps) and all(len(arr[arr<0])==0 for arr in int_event_locked_timestamps_sham):
                    regression_dict = {
                        'p_auto_inhi': np.nan,
                        't_auto_inhi': np.nan,
                        'p_collision': np.nan,
                        't_collision': np.nan,
                        'p_antidromic': np.nan,
                        't_antidromic': np.nan
                    }
                else:
                    regression_dict = antidromic_regression_analysis(
                        int_event_locked_timestamps,
                        int_event_locked_timestamps_sham,
                        antidromic_latency,
                        antidromic_jitter,
                    )

                result.update({
                    'collision_pvalue': collision_results['p_value'],
                    'collision_pbinom': collision_results['p_binom'],
                    'pre_boundary_prob': collision_results['pre_boundary_prob'],
                    'post_boundary_prob': collision_results['post_boundary_prob'],
                    'table': collision_results['table'],
                    'oddsratio': collision_results['oddsratio'],
                    'collision_boundary': collision_results['collision_boundary'],
                    'Avg': collision_results['Avg'],
                    'edges': collision_results['edges'],
                    'last_orthodromic_spike_time': collision_results['last_orthodromic_spike_time']
                })
                result.update(regression_dict)

            antidromic_results.append(result)

    antidromic_df = pd.DataFrame(antidromic_results)
    if antidromic_df.empty:
        print("No antidromic results found.")   
        return None 
    antidromic_pivot = antidromic_df.pivot(index='unit_id', columns='site').reset_index()
    if tier_cat:
        if any(col in event_ids.columns for col in ['surface_PrL', 'surface_V1', 'surface_S1', 'surface_SC']):
            unit_tiers = antidromic_tier_categorization(antidromic_pivot)
            # print(unit_tiers)
            unit_tiers_pivot = unit_tiers.pivot(index='unit_id', columns='site_for_tier', values='tier').reset_index()
            merged_df = pd.merge(antidromic_pivot, unit_tiers_pivot, on='unit_id', how='outer')
        else:
            merged_df = antidromic_pivot
            merged_df['tier'] = 0
    else:
        merged_df = antidromic_pivot
    
    # regression test 
    
    save_dir = session_dir[f'opto_dir_{data_type}']
    merged_df.to_pickle(os.path.join(save_dir, f'{session_id}_antidromic_results.pkl'))
    return merged_df


def antidromic_tier_categorization(antidromic_pivot):
    """
    Categorize units into tiers based on antidromic response criteria.

    Tier 1: Antidromic units with low jitter (<7ms) and passing collision test.
    Tier 2: Units with low jitter but failing or not tested for collision.
    Tier 3: Units with significant opto response but high jitter.

    Parameters:
        antidromic_pivot (pd.DataFrame): Pivoted DataFrame with antidromic analysis results.

    Returns:
        pd.DataFrame: DataFrame with columns unit_id, tier, and site_for_tier.
    """
    if isinstance(antidromic_pivot.columns, pd.MultiIndex):
        antidromic_pivot.columns = ['_'.join([str(i) for i in col if i]) for col in antidromic_pivot.columns.values]

    categories = []
    for idx, row in antidromic_pivot.iterrows():
        unit_id = row['unit_id']
        for col in antidromic_pivot.columns:
            if col.startswith('opto_p_val_') and not col.endswith('surface_LC'):
                site = col.replace('opto_p_val_', '')
                # tier is per unit-site, so it must not carry over from a previous site
                tier = 0
                opto_p_val = row.get(col, None)
                opto_p_val_col = f'opto_p_val_{site}'
                antidromic_latency_col = f'antidromic_latency_{site}'
                antidromic_latency = row.get(antidromic_latency_col, None)
                jitter_col = f'jitter_{site}'
                jitter = row.get(jitter_col, None)
                collision_pvalue = None
                collision_pbinom = None
                # a significant opto response is a prerequisite for every tier, so the
                # jitter and collision criteria only refine an already-significant site
                if pd.notnull(opto_p_val) and opto_p_val < 0.05:
                    tier = 3
                    if pd.notnull(jitter) and jitter < 0.007:
                        tier = 2
                        collision_pvalue_col = f'collision_pvalue_{site}'
                        collision_pbinom_col = f'collision_pbinom_{site}'
                        collision_pvalue = row.get(collision_pvalue_col, None)
                        collision_pbinom = row.get(collision_pbinom_col, None)
                        print(f"Unit {unit_id} is categorized as tier {tier} for site {site} with antidromic latency: {antidromic_latency} and jitter: {jitter} and collision_pvalue: {collision_pvalue} and collision_pbinom: {collision_pbinom}")
                        if pd.notnull(collision_pvalue) and collision_pvalue < 0.05 and pd.notnull(collision_pbinom) and collision_pbinom > 0.05:
                        # if pd.notnull(collision_pbinom) and collision_pbinom > 0.05:
                            tier = 1
                            print(f"Unit {unit_id} is categorized as tier {tier} for site {site} with antidromic latency: {antidromic_latency} and jitter: {jitter} and collision_pvalue: {collision_pvalue} and collision_pbinom: {collision_pbinom}")
                site_for_tier = site
                categories.append({'unit_id': unit_id, 'tier': tier, 'site_for_tier': f'{site_for_tier}_antidromic_tier'})

                # print(f"Unit {unit_id} is tier {tier} for site {site}, latency: {antidromic_latency}, jitter: {jitter}, collision_pvalue: {collision_pvalue} and collision_pbinom: {collision_pbinom}")
    return pd.DataFrame(categories)
    
def collision_test(int_event_locked_timestamps, antidromic_latency, bin_size=10, antidromic_jitter=0.005, plot=True):
    """
    Perform collision test for antidromic and orthodromic spikes.

    Parameters:
        int_event_locked_timestamps (list of np.ndarray): Event-locked spike times per trial.
        antidromic_latency (float): Median first spike latency after light onset.
        bin_size (float): Bin size in ms for averaging.
        antidromic_jitter (float): Jitter window in seconds.
        plot (bool): Whether to plot the results.

    Returns:
        dict: Results including DataFrame, p-value, and odds ratio.
    """

    data = []
    trial_num = 0
    collision = 1  # Default to collision

    for a, spike_times in enumerate(int_event_locked_timestamps):
        if spike_times.size > 0:
            # Find orthodromic and antidromic spike times
            orthodromic_spike_times = spike_times[spike_times <= antidromic_latency - antidromic_jitter]
            antidromic_spike_times = spike_times[
                (antidromic_latency - antidromic_jitter < spike_times) &
                (spike_times < antidromic_latency + antidromic_jitter)
            ]
            # Collision determination
            if orthodromic_spike_times.size > 0:
                last_orthodromic_spike_time = orthodromic_spike_times[-1]
                collision = 0 if antidromic_spike_times.size > 0 else 1
            else:
                last_orthodromic_spike_time = None
        else:
            orthodromic_spike_times = np.array([])
            antidromic_spike_times = np.array([])
            last_orthodromic_spike_time = None

        data.append({
            'trial_num': trial_num,
            'spike_times': spike_times.tolist(),
            'ortho_spike_times': orthodromic_spike_times.tolist(),
            'last_orthodromic_spike_time': last_orthodromic_spike_time,
            'anti_spike_times': antidromic_spike_times.tolist(),
            'collision': collision
        })
        trial_num += 1

    antidromic_df = pd.DataFrame(data)
    roi_df = antidromic_df.query('last_orthodromic_spike_time.notnull()')
    x = (roi_df['last_orthodromic_spike_time'] - antidromic_latency) * 1000
    y = 1 - roi_df['collision']

    # Bin and average    
    Avg, edges = Avg_y_over_x(x, y, bin_size=bin_size)

    # Statistical test between y in x range (-100, 70) and x range (-30, 0)
    pulse_duration = 0.005
    collision_boundary_unit = (antidromic_latency-pulse_duration) * 1000
    # print('collision boundary', collision_boundary)
    # mask1= (x >= -4*collision_boundary_unit) & (x <= -2*collision_boundary_unit)    
    mask1= (x >= -100) & (x <= -2*collision_boundary_unit)    
    mask2= (x >= -2*collision_boundary_unit) & (x <= 0)    
    y1 = y[mask1]
    y2 = y[mask2]
    table = [
        [np.sum(y1 == 1), np.sum(y1 == 0)],
        [np.sum(y2 == 1), np.sum(y2 == 0)]
    ]
    # print(f"Table for Fisher's exact test:\n{table}")
    oddsratio, p_value = fisher_exact(table)

    # Test if y in x range (-30, 0) is different from 0
    mask =  (x >= -1*collision_boundary_unit) & (x <= 0)
    y_in_range = y[mask]
    successes = np.sum(y_in_range == 1)
    n = y_in_range.size
    if n > 0:
        p_binom = binom_test(successes, n, p=0.0, alternative='greater').pvalue       
        # print(f"Binomial test p-value (y > 0 in -30 < x < 0 ms): {p_binom:.4g}")
    else:
        p_binom = np.nan
    pre_prob = np.mean(y[mask1])
    post_prob = np.mean(y[mask2])

    if plot:
        plt.plot(x, y, 'o', color=[0.8, 0.8, 0.8])
        plt.errorbar(Avg['x'], Avg['y'], yerr=Avg['sem'], fmt='o-', label='Average y with SEM')
        plt.axvline(antidromic_latency-2*collision_boundary_unit, color='red', linestyle='--', label='Collision boundary')
        plt.xlabel('Time from antidromic latency (ms)')
        plt.ylabel('P(antidromic spike|orthodromic spike)')        
        plt.xlim([-100, 0])
        plt.ylim([-0.05, 1.05])
        plt.legend()
        # Show pre- and post-boundary probabilities on the plot
        
        plt.text(0.02, 0.92, f'Pre-boundary prob: {pre_prob:.2f}', transform=plt.gca().transAxes, fontsize=10, color='blue')
        plt.text(0.02, 0.85, f'Post-boundary prob: {post_prob:.2f}', transform=plt.gca().transAxes, fontsize=10, color='green')
        # plt.text(0.02, 0.78, f'binomial p-value: {p_binom:.4g}', transform=plt.gca().transAxes, fontsize=10, color='purple')
        # plt.show()
        plt.title(f"Collision Test Results\nFisher's p-value: {p_value:.4g}, Binomial p-value: {p_binom:.4g}")
        print(f"Fisher's exact test p-value: {p_value:.4g}")
        if p_value < 0.05:
            print("Collision detected between antidromic and orthodromic spikes")

    return {
        'antidromic_df': antidromic_df,
        'roi_df': roi_df,
        'Avg': Avg,
        'edges': edges,
        'table': table,
        'oddsratio': oddsratio,
        'p_value': p_value,
        'p_binom': p_binom,
        'collision_boundary': antidromic_latency - 2 * collision_boundary_unit,
        'pre_boundary_prob': pre_prob,  # Probability of antidromic spike before collision boundary
        'post_boundary_prob': post_prob,  # Probability of antidromic spike after collision boundary
        'last_orthodromic_spike_time':antidromic_df['last_orthodromic_spike_time'],
    }

def antidromic_regression_analysis(int_event_locked_timestamps, int_event_locked_timestamps_sham, antidromic_latency, antidromic_jitter):
    """
    Perform regression analysis to assess the relationship between orthodromic spikes and antidromic spikes.

    Uses linear regression to test three effects:
    - Auto-inhibition: spontaneous spikes suppress laser-evoked spikes
    - Collision: interaction between trigger and spontaneous spikes
    - Antidromic: direct effect of laser trigger on spike probability

    Parameters:
        int_event_locked_timestamps (list of np.ndarray): Event-locked spike times for actual laser trials.
        int_event_locked_timestamps_sham (list of np.ndarray): Event-locked spike times for sham (no laser) trials.
        antidromic_latency (float): Median first spike latency after light onset in seconds.
        antidromic_jitter (float): Jitter window in seconds.

    Returns:
        dict: Dictionary with p-values and t-statistics for auto-inhibition, collision, and antidromic effects.
    """
    # import logistic regression
    from statsmodels.formula.api import logit
    from statsmodels.tools import add_constant
    # import linear regression
    from statsmodels.formula.api import ols
    data = []
    trial_num_real = len(int_event_locked_timestamps)
    trial_num_sham = len(int_event_locked_timestamps_sham)
    trigger = np.array([1]*trial_num_real + [0]*trial_num_sham)
    antidromic_jitter = 2* antidromic_jitter  # Double the jitter for both sides
    if antidromic_jitter < 0.005:
        antidromic_jitter = 0.005
    if antidromic_jitter > antidromic_latency:
        antidromic_jitter = antidromic_latency
    spont_spike = []
    spont_spike_count = []
    laser_evoked_spike = []
    laser_evoked_spike_count = []
    for spike_times in int_event_locked_timestamps + int_event_locked_timestamps_sham:
        # Orthodromic spikes
        ortho_spike_times = spike_times[(spike_times <= antidromic_latency - antidromic_jitter) & (spike_times >= -antidromic_latency-antidromic_jitter)]
        spont_spike.append(1 if ortho_spike_times.size > 0 else 0)
        spont_spike_count.append(ortho_spike_times.size)
        # Antidromic spikes
        anti_spike_times = spike_times[
            ((antidromic_latency - antidromic_jitter) < spike_times) &
            (spike_times < (antidromic_latency + antidromic_jitter))
        ]
        laser_evoked_spike.append(1 if anti_spike_times.size > 0 else 0)
        laser_evoked_spike_count.append(anti_spike_times.size)
    spont_spike = np.array(spont_spike)
    laser_evoked_spike = np.array(laser_evoked_spike)
    df = pd.DataFrame({
        'trigger': trigger,
        'spont_spike': spont_spike,
        'spont_spike_count': spont_spike_count,
        'laser_evoked_spike': laser_evoked_spike,
        'laser_evoked_spike_count': laser_evoked_spike_count
    })
    # Auto-inhibition + collision + antidromic model
    # model_half = logit("laser_evoked_spike ~ trigger + spont_spike", data=df).fit(disp=0)
    # linear regresssion
    model_half = ols("laser_evoked_spike_count ~ trigger + spont_spike_count", data=df).fit()
    if (model_half.pvalues['trigger']<0.1) and (model_half.tvalues['trigger']>0):
        try:
            # model_full = logit("laser_evoked_spike ~ trigger + spont_spike + trigger:spont_spike", data=df).fit(disp=0)
            model_full = ols("laser_evoked_spike_count ~ trigger + spont_spike_count + trigger:spont_spike", data=df).fit()
            p_auto_inhi = model_full.pvalues['spont_spike_count']
            t_auto_inhi = model_full.tvalues['spont_spike_count']
            p_collision = model_full.pvalues['trigger:spont_spike']
            t_collision = model_full.tvalues['trigger:spont_spike']
            p_antidromic = model_full.pvalues['trigger']
            t_antidromic = model_full.tvalues['trigger']
        except:
            print("Interaction term caused perfect separation; skipping interaction term.")
            p_auto_inhi = model_half.pvalues['spont_spike_count']
            t_auto_inhi = model_half.tvalues['spont_spike_count']
            p_collision = np.nan
            t_collision = np.nan
            p_antidromic = model_half.pvalues['trigger']
            t_antidromic = model_half.tvalues['trigger']
    else:
        model_full = model_half
        # get p-values
        p_auto_inhi = model_full.pvalues['spont_spike_count']
        t_auto_inhi = model_full.tvalues['spont_spike_count']
        p_collision = np.nan
        t_collision = np.nan
        p_antidromic = model_full.pvalues['trigger']
        t_antidromic = model_full.tvalues['trigger']
    return {
        'p_auto_inhi': p_auto_inhi,
        't_auto_inhi': t_auto_inhi,
        'p_collision': p_collision,
        't_collision': t_collision,
        'p_antidromic': p_antidromic,
        't_antidromic': t_antidromic
    }

def plot_opto_responses_session(session, data_type='curated', opto_only=True):
    """
    Generate and save optogenetic response plots for all units in a session.

    Parameters:
        session (str): Session identifier.
        data_type (str): Type of data to use ('curated' or 'raw').
        opto_only (bool): If True, only plot units that pass opto criteria and QC.

    Returns:
        None: Saves individual PDF/SVG files and a combined PDF to the session directory.
    """
    session_dir = session_dirs(session)
    unit_tbl = get_unit_tbl(session, data_type=data_type)
    opto_csv_file = os.path.join(session_dir[f'opto_dir_{data_type}'], f'{session}_opto_session.csv')
    opto_save_dir = os.path.join(session_dir[f'opto_dir_fig_{data_type}'], 'antidromic')
    if not os.path.exists(opto_save_dir):
        os.makedirs(opto_save_dir)
    if os.path.exists(opto_csv_file):
        event_ids = pd.read_csv(opto_csv_file)
    if opto_only:
        unit_tbl = unit_tbl[unit_tbl['opto_pass'] & unit_tbl['default_qc'] & (unit_tbl['decoder_label']!='artifact')]

    for i in range(len(unit_tbl)):
        unit_df = unit_tbl.iloc[[i]]   # double brackets → keeps it as a DataFrame
        unit_id = unit_tbl['unit_id'].values[i]
        print(f'Plotting opto responses for unit {unit_id}')
        unit_drift = load_drift(session, unit_id, data_type=data_type)
        event_ids_curr = event_ids.copy()
        
        if unit_drift is not None:
            if unit_drift['ephys_cut'][0] is not None:
                event_ids_curr = event_ids_curr[event_ids_curr['time']>=unit_drift['ephys_cut'][0]]
            if unit_drift['ephys_cut'][1] is not None:
                event_ids_curr = event_ids_curr[event_ids_curr['time']<=unit_drift['ephys_cut'][1]]
        if len(event_ids_curr) > 0:  
            fig = plot_opto_responses(unit_df, event_ids_curr)
            fig.suptitle(f'{session} unit {unit_tbl["unit_id"].values[i]}')
            fig.savefig(fname=os.path.join(opto_save_dir, f'{session}_unit{unit_tbl["unit_id"].values[i]}_opto_responses.pdf'))
            fig.savefig(fname=os.path.join(opto_save_dir, f'{session}_unit{unit_tbl["unit_id"].values[i]}_opto_responses.svg'))
            plt.close(fig)
        else:
            print(f"No opto events found for session {session}, skipping unit {unit_id}.")
        
    combine_pdf_big(opto_save_dir, os.path.join(session_dir[f'opto_dir_fig_{data_type}'], f'{session}_antidromic_responses.pdf'))

if __name__ == "__main__":


    data_type = 'curated'  # 'raw' or 'curated'
    session_df = pd.read_csv(CAPSULE_ROOT + '/code/data_management/session_assets.csv')
    # remove opto sessions
    session_list = session_df[session_df['probe'] == '2']['session_id'].to_list()
    def process(session, data_type='curated'):
        session_dir = session_dirs(session)
        print(f"Processing session: {session}")
        if session_dir[f'curated_dir_{data_type}'] is not None:
            print(f"Processing session: {session}")
            # try:
            plot_opto_responses_session(session, data_type=data_type, opto_only=True)
            analyze_antidromic_responses(session)
            # except:
                # print(f"Failed to process session: {session}")
            print(f"Finished session: {session}")
        else:
            print(f"No curated data for session: {session}, skipping.")
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=3)(delayed(process)(session, data_type=data_type) for session in session_list)
    process('behavior_752014_2025-03-26_11-18-57')
    # for session in session_list:
    #     process(session)