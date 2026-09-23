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
    for i, arr in enumerate(int_event_locked_timestamps):
        # Remove spikes during the laser pulse        
        int_event_locked_timestamps[i] = arr[(arr < 0) | (arr > (duration / 1000))]        
    return int_event_locked_timestamps

def opto_tagging_response(int_event_locked_timestamps, base_window, roi_window):
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

def antidromic_latency_jitter(int_event_locked_timestamps, min_spikes=5):
    from scipy.ndimage import gaussian_filter1d
    pulse_duration = 0.005 # ms
    first_spikes_after_light = np.array([arr[arr > pulse_duration][0] for arr in int_event_locked_timestamps if np.any(arr > pulse_duration)])
    # FWHM is meaningless below a handful of first spikes, and the degenerate cases look
    # like real responses: an empty histogram peaks at bin 0 (latency 0.005) with jitter
    # spanning the whole window, and a single spike gives jitter of exactly 0. Both would
    # pass the jitter < 0.007 tier gate, so report them as undefined instead.
    if first_spikes_after_light.size < min_spikes:
        return np.nan, np.nan
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
    # A peak confined to one bin is one bin wide, not zero wide. Reporting 0 here made the
    # coarsest possible measurement look like the sharpest jitter in the dataset.
    fwhm = max(fwhm, bins[1] - bins[0])

    antidromic_latency = peak_time
    antidromic_jitter = fwhm
    return antidromic_latency, antidromic_jitter


def plot_opto_responses(unit_tbl, event_ids, time_range_raster=(-0.100, 0.070), sites=None):
    """
    Find antidromic units based on opto stimulation data.

    time_range_raster: raster/PSTH x-limits in s. The default full window is too coarse to
        judge whether a light-locked spike is a real spike or an artifact; pass something like
        (-0.010, 0.040) for a zoomed page.
    sites: restrict/order the emission locations plotted as columns (default: all present).
    """
    # Filter for opto tagged units
    # opto_criteria = (unit_tbl['opto_pass'] == True) & (unit_tbl['default_qc'] == True)
    opto_units = unit_tbl["unit_id"].to_list()
    print(f"Number of opto units: {len(opto_units)}")

    
    # Unique values
    present_sites = list(np.unique(event_ids.emission_location))
    if sites is None:
        sites = present_sites
    else:
        sites = [s for s in sites if s in present_sites]
    powers = list(np.unique(event_ids.power))
    trial_types = np.unique(event_ids.type)


    # Settings
    prepost = 'post'
    num_sites = len(sites)
    num_units = len(opto_units)
    time_range_raster = np.asarray(time_range_raster, dtype=float)

    # Create one figure for all units × sites, 6 rows per unit: raster, first-spike-sorted
    # raster, PSTH, first-spike histogram, collision curve, spacer. The panels below index
    # rows as 6*u_idx + k; with the old stride of 5 the second unit's raster landed on the
    # first unit's collision panel.
    fig_height_per_unit = 10
    fig = plt.figure(figsize=(num_sites * 4, num_units * fig_height_per_unit))
    gs = gridspec.GridSpec(6 * num_units, num_sites, height_ratios=[3, 6, 2, 1.5, 1.5, 0.8] * num_units, hspace=0.8)

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


            # Filter trials. The fallback is per site and keeps the site fixed: it used to
            # reassign the shared `prepost` and query surface_LC by name, so one site without
            # 'post' trials silently swapped every later column to LC 'pre' trials.
            this_prepost = prepost
            tag_trials = event_ids.query('site == @site and pre_post == @this_prepost')
            if tag_trials.empty:
                this_prepost = 'pre'
                tag_trials = event_ids.query('site == @site and pre_post == @this_prepost')
            if tag_trials.empty:
                continue
            max_power = tag_trials.power.max()
            tag_trials = tag_trials.query('power == @max_power')
            if tag_trials.empty:
                continue

            # Stimulation parameters
            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]

            # Time window (set by the time_range_raster argument)
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
            ax_raster = fig.add_subplot(gs[6 * u_idx, i])
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
            ax_antidromic = fig.add_subplot(gs[6 * u_idx + 1, i], sharex=ax_raster)
            sorted_data = sorted(int_event_locked_timestamps, key=lambda x: (len(x) == 0, x[0] if len(x) > 0 else np.inf))
            pf.raster_plot(sorted_data, time_range_raster, ax=ax_antidromic)

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
            ax_psth = fig.add_subplot(gs[6 * u_idx + 2, i], sharex=ax_raster)
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
            ax_latency = fig.add_subplot(gs[6 * u_idx + 3, i])
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
            ax_compare = fig.add_subplot(gs[6 * u_idx + 4, i])
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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # No finite x (e.g. no trial had an orthodromic spike): nothing to bin.
    if x.size == 0 or not np.any(np.isfinite(x)):
        return pd.DataFrame(columns=['x', 'y', 'sem']), np.array([])
    edges = np.arange(np.nanmin(x), np.nanmax(x) + bin_size, bin_size)

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

def analyze_antidromic_responses(session_id, data_type ='curated', plot=False, tier_cat = False,
                                 soma_site='surface_DRN', min_spikes=5, tier_kwargs=None):
    """
    Analyze antidromic responses for a given set of opto-tagged units.

    Parameters:
        session_id (str): session id
        plot (bool): Whether to plot collision raster for each unit and site.
        tier_cat (bool): Whether to attach ('antidromic_tier', site) and
            ('antidromic_tier_label', site) columns via antidromic_tier_categorization.
        soma_site (str): Stimulation site over the recorded soma. It gets the late response
            window and is excluded from collision testing; every other site is treated as a
            projection target. Use 'surface_LC' for LC-NE sessions.
        min_spikes (int): Minimum first-spikes-after-light required to report latency/jitter.
        tier_kwargs (dict): Optional overrides forwarded to antidromic_tier_categorization
            (opto_p_thresh, max_jitter, max_antidromic_latency, collision_p_thresh, verbose).

    Returns:
        pd.DataFrame: DataFrame containing antidromic response metrics and tier categorization.
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
            if site != soma_site:
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
            antidromic_latency, antidromic_jitter = antidromic_latency_jitter(
                int_event_locked_timestamps, min_spikes=min_spikes
            )
            first_spikes_after_light = np.array([arr[arr > 0][0] for arr in int_event_locked_timestamps if np.any(arr > 0)])
            if first_spikes_after_light.size < min_spikes:
                median_first_spikes_after_light = np.nan
            else:
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

            # Collision test for projection-target sites only, and only where the latency
            # the test is built around is actually defined.
            if site != soma_site and np.isfinite(antidromic_latency):
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
    merged_df = antidromic_pivot
    if tier_cat:
        # Sites are values in the 'site' column, not columns of event_ids: the old
        # column-membership check was always False, so tiers were never computed.
        if any(site != soma_site for site in antidromic_df['site'].unique()):
            unit_tiers = antidromic_tier_categorization(
                antidromic_pivot, soma_site=soma_site, **(tier_kwargs or {})
            )
            # Attach tiers as ('antidromic_tier', site) so the saved file keeps the
            # (metric, site) MultiIndex that the across-session combining code expects.
            # The label travels alongside the code because the code is not monotone in
            # evidence (0 = none sorts below 1 = strongest), so it cannot be aggregated
            # or thresholded numerically.
            tier_wide = unit_tiers.pivot(index='unit_id', columns='site', values='tier')
            label_wide = unit_tiers.pivot(index='unit_id', columns='site', values='tier_label')
            merged_df = antidromic_pivot.copy()
            for site in tier_wide.columns:
                merged_df[('antidromic_tier', site)] = merged_df[('unit_id', '')].map(tier_wide[site])
                merged_df[('antidromic_tier_label', site)] = merged_df[('unit_id', '')].map(label_wide[site])
        else:
            merged_df = antidromic_pivot.copy()
            merged_df[('antidromic_tier', '')] = 0
            merged_df[('antidromic_tier_label', '')] = ANTIDROMIC_TIER_LABELS[0]
    
    # regression test 
    
    save_dir = session_dir[f'opto_dir_{data_type}']
    merged_df.to_pickle(os.path.join(save_dir, f'{session_id}_antidromic_results.pkl'))
    return merged_df


# Tier codes, strongest evidence first. The codes are nested: a unit x site can only reach a
# stronger (lower-numbered) tier by also meeting every criterion of the weaker tiers. The scale
# is not monotone -- 0 means "no evidence" and sorts below 1, which is the strongest -- so never
# aggregate the code with max()/min() or threshold it with >=. Use the label for that.
ANTIDROMIC_TIER_LABELS = {
    1: 'collision',    # responsive + sharp short-latency peak + collision-confirmed
    2: 'jitter',       # responsive + sharp short-latency peak, collision not confirmed
    3: 'responsive',   # significant light-evoked response only
    0: 'none',         # no significant light-evoked response
}


def antidromic_tier_categorization(antidromic_pivot, soma_site='surface_DRN',
                                   opto_p_thresh=0.05, max_jitter=0.007,
                                   max_antidromic_latency=0.025,
                                   collision_p_thresh=0.05, verbose=False):
    """Assign a tier per (unit, projection-target site).

    Tiers are nested, strongest first (see ANTIDROMIC_TIER_LABELS):

      tier 3 'responsive'  significant light-evoked response at the target site
                           (opto_p_val < opto_p_thresh).
      tier 2 'jitter'      tier 3 and a sharp, short-latency first-spike peak
                           (jitter < max_jitter and 0 < latency <= max_antidromic_latency).
      tier 1 'collision'   tier 2 and the collision test confirms it (see below).
      tier 0 'none'        everything else.

    Each tier requires the ones below it. Promotion used to be independent, so a sharp
    first-spike peak set tier 2 even with no significant response at all (35 of the 96
    sharp rows in the DRN dataset had opto_p_val >= 0.05, up to 1.0), ranking
    non-responsive units above responsive ones.

    The latency gate matters because jitter is the FWHM of the first-spike histogram over
    5-100 ms with a one-bin floor, so a late spontaneous spike landing in a single bin
    reports the sharpest jitter in the dataset (0.5 ms) at a latency of 40-100 ms. Without
    the gate those rows reached the sharp tiers.

    Collision confirmation needs both halves of collision_test: collision_pvalue (Fisher)
    < collision_p_thresh, i.e. the antidromic spike is lost more often when an orthodromic
    spike arrives inside the collision window, AND collision_pbinom above threshold, i.e.
    zero antidromic-spike probability just before the boundary cannot be rejected
    (complete collision). collision_pbinom is a binomial test against p=0, so in practice
    it is an indicator: 1.0 when no antidromic spike survived the final window, 0.0 when
    one did. A NaN collision result (latency undefined, or no trial with a qualifying
    orthodromic spike) leaves the unit at tier 2 -- untested, not disconfirmed.

    Parameters:
        antidromic_pivot (pd.DataFrame): per-unit pivot from analyze_antidromic_responses,
            with either (metric, site) MultiIndex columns or flat '<metric>_<site>' names.
        soma_site (str): stimulation site over the soma; excluded from tiering.
        opto_p_thresh (float): response-test alpha for tier 3.
        max_jitter (float): first-spike FWHM ceiling, in seconds, for tier 2.
        max_antidromic_latency (float): first-spike peak latency ceiling, in seconds, for
            tier 2.
        collision_p_thresh (float): alpha shared by the two collision criteria for tier 1.
        verbose (bool): print one line per unit x site promoted to tier 2 or 1.

    Returns:
        pd.DataFrame: long frame with columns ['unit_id', 'site', 'tier', 'tier_label'].
    """
    # Flatten on a copy. Assigning to the caller's .columns renamed its columns as a side
    # effect, so the schema of the saved per-session pkl depended on whether tier_cat was
    # set: MultiIndex when it was off, flat '<metric>_<site>' names when it was on.
    if isinstance(antidromic_pivot.columns, pd.MultiIndex):
        antidromic_pivot = antidromic_pivot.copy()
        antidromic_pivot.columns = ['_'.join([str(i) for i in col if i]) for col in antidromic_pivot.columns.values]

    categories = []
    for idx, row in antidromic_pivot.iterrows():
        unit_id = row['unit_id']
        for col in antidromic_pivot.columns:
            if col.startswith('opto_p_val_') and not col.endswith(soma_site):
                site = col.replace('opto_p_val_', '')
                # Reset per site: tier used to persist from the previous site in this row,
                # so one qualifying site silently promoted every site after it.
                tier = 0
                opto_p_val = row.get(col, None)
                antidromic_latency = row.get(f'antidromic_latency_{site}', None)
                jitter = row.get(f'jitter_{site}', None)
                collision_pvalue = row.get(f'collision_pvalue_{site}', None)
                collision_pbinom = row.get(f'collision_pbinom_{site}', None)

                responsive = pd.notnull(opto_p_val) and opto_p_val < opto_p_thresh
                sharp = (
                    pd.notnull(jitter) and jitter < max_jitter
                    and pd.notnull(antidromic_latency)
                    and 0 < antidromic_latency <= max_antidromic_latency
                )
                collision_confirmed = (
                    pd.notnull(collision_pvalue) and collision_pvalue < collision_p_thresh
                    and pd.notnull(collision_pbinom) and collision_pbinom > collision_p_thresh
                )

                # Nested: each promotion happens only inside the weaker tier it requires.
                if responsive:
                    tier = 3
                    if sharp:
                        tier = 2
                        if collision_confirmed:
                            tier = 1

                if verbose and tier in (1, 2):
                    print(
                        f"Unit {unit_id} site {site}: tier {tier} ({ANTIDROMIC_TIER_LABELS[tier]}), "
                        f"opto_p_val: {opto_p_val}, latency: {antidromic_latency}, "
                        f"jitter: {jitter}, collision_pvalue: {collision_pvalue}, "
                        f"collision_pbinom: {collision_pbinom}"
                    )
                categories.append({
                    'unit_id': unit_id,
                    'site': site,
                    'tier': tier,
                    'tier_label': ANTIDROMIC_TIER_LABELS[tier],
                })
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
    Parameters:
        int_event_locked_timestamps (list of np.ndarray): Event-locked spike times for actual trials.
        int_event_locked_timestamps_sham (list of np.ndarray): Event-locked spike times for sham trials.
        antidromic_latency (float): Median first spike latency after light onset.
        antidromic_jitter (float): Jitter window in seconds.
    Returns:
        tuple: p-values for auto-inhibition, collision, and antidromic effects.
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
    session_df = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
    # remove opto sessions
    # probe is int64 in session_assets.csv, so comparing to the string '2' matched no rows
    # and session_list came back empty.
    session_list = session_df[session_df['probe'].astype(str) == '2']['session_id'].to_list()
    def process(session, data_type='curated'):
        session_dir = session_dirs(session)
        print(f"Processing session: {session}")
        if session_dir[f'curated_dir_{data_type}'] is not None:
            print(f"Processing session: {session}")
            # try:
            plot_opto_responses_session(session, data_type=data_type, opto_only=True)
            # analyze_antidromic_responses(session)
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