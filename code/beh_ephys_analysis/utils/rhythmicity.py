"""
Quantify rhythmicity of spontaneous firing from long-lag autocorrelograms.

Motivation: the autocorrelograms produced by cross_auto_corr_DRN use bin_long = 50 ms
(Nyquist 10 Hz) or bin_short = 2 ms over only 80 ms of lag, so neither can measure the
2-12 Hz band. This module recomputes the diagonal (autocorrelation) alone at a finer
bin over a longer window, and scores each unit for preferred period and how strongly it
adheres to it.

The central control is the ISI shuffle. Serotonin neurons are classically slow, regular
pacemakers, and a narrow ISI distribution produces rhythmic autocorrelogram peaks on its
own -- that is just the renewal density, not an oscillation. Shuffling ISIs preserves the
ISI distribution exactly while destroying any longer-range structure, so the surrogate
tells you how much of the observed rhythmicity is explained by ISI regularity alone. A
renewal process damps out within ~2-3 cycles because variance accumulates across
successive intervals; a sustained rhythm does not. n_cycles_sig captures that difference
and is the intended primary measure.
"""

import warnings

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import median_filter

try:
    from .ephys_functions import correlate_nan_fft
except ImportError:
    from ephys_functions import correlate_nan_fft


def merge_intervals(starts, ends):
    """
    Merge overlapping [start, end] intervals.

    Returns:
    starts, ends : ndarray
        Sorted, non-overlapping intervals.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    keep_starts, keep_ends = [starts[0]], [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= keep_ends[-1]:
            keep_ends[-1] = max(keep_ends[-1], e)
        else:
            keep_starts.append(s)
            keep_ends.append(e)
    return np.array(keep_starts), np.array(keep_ends)


def binned_counts(spike_times, bin_size, start, end, blank_intervals=None):
    """
    Bin spike times, setting bins that overlap a blanked interval to NaN.

    Matches the bin-overlap convention of auto_corr_train_nogo (a bin is blanked if it
    overlaps the interval at all) but vectorized over intervals.

    Parameters:
    spike_times : array-like
        Spike times, in seconds.
    bin_size : float
        Bin width, in seconds.
    start, end : float
        Window bounds, in seconds.
    blank_intervals : tuple of array-like, optional
        (starts, ends) of intervals to exclude, e.g. go-cue periods.

    Returns:
    counts : ndarray of float
        Spike count per bin, NaN in blanked bins.
    bin_starts : ndarray
        Left edge of each bin.
    """
    edges = np.arange(start, end, bin_size)
    counts = np.histogram(spike_times, bins=edges)[0].astype(float)
    bin_starts = edges[:-1]
    if blank_intervals is not None:
        b_starts, b_ends = merge_intervals(*blank_intervals)
        if len(b_starts):
            # a bin [b0, b0 + bin_size) overlaps [s, e] iff b0 in [s - bin_size, e]
            idx = np.searchsorted(b_starts - bin_size, bin_starts, side='right') - 1
            valid = idx >= 0
            blanked = np.zeros(len(bin_starts), dtype=bool)
            blanked[valid] = bin_starts[valid] <= b_ends[idx[valid]]
            counts[blanked] = np.nan
    return counts, bin_starts


def acg(spike_times, bin_size, window_length, start, end, blank_intervals=None):
    """
    One-sided autocorrelogram as per-lag Pearson r of binned counts.

    Same definition as auto_corr_train (so values are comparable to the existing
    pipeline), computed with the FFT path so a long window is affordable.

    Returns:
    corrs : ndarray
        Pearson r at lags 0..window_length.
    lags : ndarray
        Lag in seconds.
    n_valid : ndarray
        Valid bin pairs per lag. Falls with lag whenever blank_intervals is used, and
        falls periodically with trial structure -- mask thin lags before interpreting.
    """
    counts, _ = binned_counts(spike_times, bin_size, start, end, blank_intervals)
    max_lag = int(np.round(window_length / bin_size))
    corrs, n_valid = correlate_nan_fft(counts, counts, lag=max_lag, return_n=True)
    return corrs, np.arange(max_lag + 1) * bin_size, n_valid


def isi_shuffle(spike_times, start, end, rng):
    """
    Shuffle inter-spike intervals, preserving the ISI distribution exactly.

    Destroys structure beyond the ISI distribution, so the resulting train is a renewal
    process with the observed ISI distribution. Spike count and span are preserved.

    Returns:
    ndarray
        Surrogate spike times within [start, end].
    """
    spikes = np.sort(np.asarray(spike_times, dtype=float))
    spikes = spikes[(spikes >= start) & (spikes <= end)]
    if len(spikes) < 3:
        return spikes
    isis = np.diff(spikes)
    return spikes[0] + np.concatenate([[0.0], np.cumsum(rng.permutation(isis))])


def nan_gaussian_smooth(y, sigma_bins):
    """Gaussian smooth, renormalizing by the smoothed validity mask so NaNs are ignored."""
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    half = int(np.ceil(4 * sigma_bins))
    grid = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
    kernel /= kernel.sum()
    num = np.convolve(np.where(valid, y, 0.0), kernel, mode='same')
    den = np.convolve(valid.astype(float), kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den <= 0] = np.nan
    return out


def detrend_acg(corrs, bin_size, sigma=0.3, drop_lags=1):
    """
    Remove the slow component of an autocorrelogram.

    Global mean subtraction of the count vector removes DC but not the slow decay left by
    rate nonstationarity within the window, which otherwise dominates the low-frequency
    end of the spectrum. Subtracting a wide-kernel smoothed copy removes it while leaving
    periods shorter than the kernel intact.

    Parameters:
    corrs : array-like
        One-sided autocorrelogram.
    bin_size : float
        Bin width, in seconds.
    sigma : float
        Smoothing width in seconds; must exceed the slowest period of interest.
    drop_lags : int
        Number of initial lags to drop (lag 0 is 1.0 by construction).

    Returns:
    detrended : ndarray
        Detrended autocorrelogram, with the first drop_lags entries set to NaN.
    """
    corrs = np.asarray(corrs, dtype=float).copy()
    corrs[:drop_lags] = np.nan
    return corrs - nan_gaussian_smooth(corrs, sigma / bin_size)


def acg_spectrum(detrended, bin_size, freq_range=(0.5, 25.0)):
    """
    Power spectrum of a detrended autocorrelogram.

    Returns:
    freqs, power : ndarray
        Frequencies within freq_range and their power.
    """
    y = np.asarray(detrended, dtype=float).copy()
    y[~np.isfinite(y)] = 0.0
    y = y * np.hanning(len(y))
    power = np.abs(rfft(y)) ** 2
    freqs = rfftfreq(len(y), d=bin_size)
    keep = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    return freqs[keep], power[keep]


def isi_stats(spike_times, start, end, serial_detrend_n=51):
    """
    Rate and regularity statistics. The natural comparison for the autocorrelogram
    metrics: if the ISI shuffle explains the rhythmicity, these are the result.

    Returns:
    dict with n_spikes, duration, rate, cv_isi, cv2_isi, isi_mode (mode of the log-ISI
    histogram, in seconds), and isi_serial_1..3 / isi_serial_z (see below).
    """
    spikes = np.sort(np.asarray(spike_times, dtype=float))
    spikes = spikes[(spikes >= start) & (spikes <= end)]
    out = {'n_spikes': len(spikes), 'duration': end - start,
           'rate': len(spikes) / (end - start) if end > start else np.nan,
           'cv_isi': np.nan, 'cv2_isi': np.nan, 'isi_mode': np.nan,
           'isi_serial_1': np.nan, 'isi_serial_2': np.nan, 'isi_serial_3': np.nan,
           'isi_serial_z': np.nan, 'isi_serial_1_det': np.nan,
           'isi_serial_2_det': np.nan, 'isi_serial_3_det': np.nan,
           'isi_serial_z_det': np.nan}
    if len(spikes) < 3:
        return out
    isis = np.diff(spikes)
    isis = isis[isis > 0]
    if len(isis) < 2:
        return out
    out['cv_isi'] = np.std(isis) / np.mean(isis)
    out['cv2_isi'] = np.mean(2 * np.abs(np.diff(isis)) / (isis[1:] + isis[:-1]))
    log_isi = np.log10(isis)
    hist, edges = np.histogram(log_isi, bins=60)
    out['isi_mode'] = 10 ** (0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))

    # Serial correlation of the interval sequence: the one statistic the ISI shuffle
    # cannot reproduce, because shuffling sets it to zero by construction. It is what
    # separates a clock from a regular-but-memoryless firer. Under the renewal null
    # r ~ N(0, 1/sqrt(n_isi)), which sets the z below.
    #
    # Calibrated on synthetic trains, 1800 s (isi_serial_1, isi_serial_2):
    #   gamma renewal, cv 0.26 / 0.36 / 1.0   ( 0.00,  0.00)  null holds at any cv
    #   clock + white phase jitter 10-25%     (-0.50,  0.00)  -0.5 exactly, cv-independent
    #   sinusoidally driven Poisson, depth .8 (-0.01,  0.00)  a rate oscillation is NOT a
    #                                                         clock and scores as renewal
    #   doublets on a 5 Hz clock              (-0.99, +0.98)  bursting, alternating sign
    #
    # So: isi_serial_1 near -0.5 with isi_serial_2 near 0 is phase-locking, and -0.5 is
    # the floor for a true clock regardless of how precise it is. Values well below -0.5
    # with a strongly POSITIVE isi_serial_2 are short-long alternation from bursting, not
    # a clock -- which matters here, since 5-HT cells fire doublets. The sign pattern at
    # lag 2 is what separates the two. A positive isi_serial_1 is rate drift.
    if len(isis) > 20:
        z = (isis - np.mean(isis)) / np.std(isis) if np.std(isis) > 0 else None
        if z is not None:
            for k in (1, 2, 3):
                out[f'isi_serial_{k}'] = float(np.mean(z[:-k] * z[k:]))
            out['isi_serial_z'] = out['isi_serial_1'] * np.sqrt(len(isis))

        # Over a window of tens of minutes the firing rate drifts, which adds a positive
        # offset to EVERY lag and can hide the clock's negative lag-1 term. Divide out a
        # running median of the interval sequence first: drift is multiplicative on rate,
        # and the ratio is dimensionless, so this removes the offset without touching
        # interval-to-interval structure. The detrended values are the ones to interpret.
        local = median_filter(isis, size=min(serial_detrend_n, len(isis) // 2 * 2 + 1),
                              mode='nearest')
        ratio = isis / np.where(local > 0, local, np.nan)
        good = np.isfinite(ratio)
        if good.sum() > 20 and np.nanstd(ratio) > 0:
            zr = (ratio - np.nanmean(ratio)) / np.nanstd(ratio)
            zr = np.where(good, zr, 0.0)
            for k in (1, 2, 3):
                out[f'isi_serial_{k}_det'] = float(np.mean(zr[:-k] * zr[k:]))
            out['isi_serial_z_det'] = (out['isi_serial_1_det'] * np.sqrt(good.sum()))
    return out


def _peak_at_cycles(detrended, lags, period, n_cycles, tol_frac=0.1):
    """
    Value of the autocorrelogram at each multiple of `period`, taken as the local max
    within +/- tol_frac * period. Applied identically to observed and surrogate traces so
    the comparison stays fair (the max-of-noise bias affects both).

    Expects an already-smoothed trace: at 5 ms bins a single lag carries SNR ~1 even for
    a 60% rate modulation, so the unsmoothed per-cycle test has no power.
    """
    out = np.full(n_cycles, np.nan)
    tol = tol_frac * period
    for k in range(1, n_cycles + 1):
        window = np.abs(lags - k * period) <= tol
        if window.any() and np.isfinite(detrended[window]).any():
            out[k - 1] = np.nanmax(detrended[window])
    return out


def rhythmicity_metrics(spike_times, bin_size=0.005, window_length=4.0,
                        start=None, end=None, blank_intervals=None,
                        band=(2.0, 12.0), freq_range=(0.5, 25.0), detrend_sigma=0.3,
                        cycle_smooth_frac=1 / 12, n_surrogates=200, min_valid_frac=0.2,
                        min_spikes=100, seed=0):
    """
    Score one unit for preferred firing period and strength of adherence to it.

    Parameters:
    spike_times : array-like
        Spike times, in seconds.
    bin_size : float
        Autocorrelogram bin, in seconds. 0.005 gives 16 samples/cycle at 12 Hz; the
        pipeline's 0.05 has a Nyquist of 10 Hz and cannot resolve the band.
    window_length : float
        Maximum lag, in seconds. 4.0 spans 8 cycles at 2 Hz, enough to separate a
        3-cycle renewal echo from a sustained rhythm.
    start, end : float
        Analysis window, e.g. the drift-cut-intersected spontaneous period.
    blank_intervals : tuple of array-like, optional
        (starts, ends) to exclude, e.g. go cue times and go cue times + go_cue_period.
        Surrogates get the identical blanking, so any artifact it injects appears in the
        null as well.
    band : tuple
        Frequency band searched for the rhythm.
    freq_range : tuple
        Spectrum range used as the denominator of the oscillation score.
    detrend_sigma : float
        Detrending kernel width, in seconds.
    cycle_smooth_frac : float
        Smoothing kernel for the per-cycle test, as a fraction of the period. Needed
        because single-lag SNR is ~1 at 5 ms bins for a low-rate unit.
    n_surrogates : int
        ISI shuffles. 0 skips the null (metrics then have no p-value or cycle count).
    min_valid_frac : float
        Lags whose valid-pair count falls below this fraction of the lag-0 count are
        masked out.
    min_spikes : int
        Units below this are returned unscored. Set it from the expected coincidences per
        lag bin (n_spikes * rate * bin_size) for the modulation depth you need to resolve;
        excluded units are unmeasurable, not arrhythmic.
    seed : int
        RNG seed for the shuffles.

    Returns:
    dict
        Metrics plus the arrays needed to plot the unit:
        peak_freq, peak_period : preferred rhythm from the autocorrelogram spectrum.
        first_peak_lag, first_peak_amp, first_peak_z : lag and amplitude of the first
            peak, and its z-score against the ISI-shuffle distribution at that cycle.
        osc_score : peak power / mean power over freq_range (Muresan et al. 2008). Not a
            rhythmicity measure on its own -- a regular pacemaker with CV 0.1 scores as
            high as a genuine oscillator. Interpret it only alongside the surrogate.
        osc_score_p : fraction of surrogates reaching the observed score. Floored at
            1/(n_surrogates+1); use osc_score_z when correcting across a population.
        osc_score_z : log(osc_score) standardized against the surrogate distribution.
        n_cycles_sig : how many cycles have a peak above the surrogate 97.5th percentile.
        max_run_sig, last_cycle_sig : length and end of the longest consecutive run of
            significant cycles. A run of >=3 is the evidence that separates a sustained
            rhythm from ISI regularity; isolated significant cycles are expected by chance.
        persist_time : lag in seconds at the end of that run -- how far out the rhythm
            still holds. Prefer this over cycle counts when comparing units, since a 4 s
            window holds 8 cycles at 2 Hz but 31 at 8 Hz.
        cv_isi, cv2_isi, isi_mode, rate, n_spikes, duration : see isi_stats.
    """
    spikes = np.asarray(spike_times, dtype=float)
    if start is None:
        start = np.min(spikes) if len(spikes) else 0.0
    if end is None:
        end = np.max(spikes) if len(spikes) else 0.0

    out = {'bin_size': bin_size, 'window_length': window_length,
           'peak_freq': np.nan, 'peak_period': np.nan, 'osc_score': np.nan,
           'osc_score_p': np.nan, 'osc_score_z': np.nan,
           'first_peak_lag': np.nan, 'first_peak_amp': np.nan,
           'first_peak_z': np.nan, 'n_cycles_sig': np.nan, 'max_run_sig': np.nan,
           'last_cycle_sig': np.nan, 'persist_time': np.nan,
           'n_surrogates': n_surrogates}
    out.update(isi_stats(spikes, start, end))
    if out['n_spikes'] < min_spikes or end - start <= window_length:
        out['acg'] = None
        return out

    corrs, lags, n_valid = acg(spikes, bin_size, window_length, start, end, blank_intervals)
    thin = n_valid < min_valid_frac * max(n_valid[0], 1)
    corrs = np.where(thin, np.nan, corrs)
    detrended = detrend_acg(corrs, bin_size, sigma=detrend_sigma)
    freqs, power = acg_spectrum(detrended, bin_size, freq_range)

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any() or not np.isfinite(power).any():
        out['acg'] = corrs
        return out
    peak_idx = np.argmax(np.where(in_band, power, -np.inf))
    peak_freq = freqs[peak_idx]
    period = 1.0 / peak_freq
    out['peak_freq'] = peak_freq
    out['peak_period'] = period
    out['osc_score'] = power[peak_idx] / np.mean(power)

    # smooth with a period-proportional kernel before any single-lag test; attenuates the
    # rhythm by ~13% but cuts white-noise std ~3x, which the per-cycle test needs
    smooth_bins = max(cycle_smooth_frac * period / bin_size, 0.6)
    det_smooth = nan_gaussian_smooth(detrended, smooth_bins)
    out['acg_smooth'] = det_smooth

    # first peak after the refractory dip, as a period estimate independent of the spectrum
    search = (lags > 0.5 * period) & (lags < 1.5 * period) & np.isfinite(det_smooth)
    if search.any():
        out['first_peak_lag'] = lags[search][np.argmax(det_smooth[search])]
        out['first_peak_amp'] = np.nanmax(det_smooth[search])

    n_cycles = int(np.floor(window_length / period))
    out['cycle_peaks'] = _peak_at_cycles(det_smooth, lags, period, n_cycles)

    if n_surrogates > 0:
        rng = np.random.default_rng(seed)
        surr_acgs = np.full((n_surrogates, len(lags)), np.nan)
        surr_scores = np.full(n_surrogates, np.nan)
        surr_cycles = np.full((n_surrogates, n_cycles), np.nan)
        for i in range(n_surrogates):
            surr_spikes = isi_shuffle(spikes, start, end, rng)
            s_corrs, _, s_valid = acg(surr_spikes, bin_size, window_length,
                                      start, end, blank_intervals)
            s_corrs = np.where(s_valid < min_valid_frac * max(s_valid[0], 1), np.nan, s_corrs)
            s_det = detrend_acg(s_corrs, bin_size, sigma=detrend_sigma)
            _, s_power = acg_spectrum(s_det, bin_size, freq_range)
            surr_scores[i] = np.max(s_power[in_band]) / np.mean(s_power)
            s_smooth = nan_gaussian_smooth(s_det, smooth_bins)
            surr_acgs[i] = s_smooth
            surr_cycles[i] = _peak_at_cycles(s_smooth, lags, period, n_cycles)

        out['osc_score_p'] = (1 + np.sum(surr_scores >= out['osc_score'])) / (1 + n_surrogates)
        # The permutation p is floored at 1/(n_surrogates+1), so it cannot survive FDR
        # correction across a whole population (500 units at q=0.05 would need ~10,000
        # shuffles per unit). Standardizing log(OS) against the surrogate distribution
        # gives an unfloored statistic that FDR can be applied to; 100 shuffles estimate
        # its mean and sd to within ~7%.
        log_surr = np.log(surr_scores[surr_scores > 0])
        if len(log_surr) > 2 and np.std(log_surr) > 0:
            out['osc_score_z'] = (np.log(out['osc_score']) - np.mean(log_surr)) / np.std(log_surr)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN lag 0 column
            cycle_hi = np.nanpercentile(surr_cycles, 97.5, axis=0)
            out['surr_acg_lo'] = np.nanpercentile(surr_acgs, 2.5, axis=0)
            out['surr_acg_hi'] = np.nanpercentile(surr_acgs, 97.5, axis=0)
            cycle_mean = np.nanmean(surr_cycles, axis=0)
            cycle_std = np.nanstd(surr_cycles, axis=0)
        # A renewal null has real structure of its own out to ~1 mean ISI, so for fast
        # rhythms the first cycle or two sit inside the null's relaxation transient and
        # cannot be tested. Score persistence by the longest run of significant cycles
        # wherever it falls, not by a run anchored at cycle 1.
        above = np.where(np.isfinite(out['cycle_peaks']) & np.isfinite(cycle_hi),
                         out['cycle_peaks'] > cycle_hi, False)
        run_end, run_len, best_len, best_end = 0, 0, 0, 0
        for k, sig in enumerate(above):
            run_len = run_len + 1 if sig else 0
            run_end = k + 1
            if run_len > best_len:
                best_len, best_end = run_len, run_end
        out['n_cycles_sig'] = int(above.sum())
        out['max_run_sig'] = int(best_len)
        out['last_cycle_sig'] = int(best_end) if best_len else 0
        out['persist_time'] = best_end * period if best_len else 0.0
        out['cycle_sig'] = above
        if n_cycles > 0 and np.isfinite(cycle_std[0]) and cycle_std[0] > 0:
            out['first_peak_z'] = (out['cycle_peaks'][0] - cycle_mean[0]) / cycle_std[0]
        out['surr_scores'] = surr_scores
        out['cycle_hi'] = cycle_hi

    out.update({'acg': corrs, 'acg_detrended': detrended, 'lags': lags,
                'n_valid': n_valid, 'freqs': freqs, 'power': power})
    return out


def plot_unit_rhythmicity(res, unit_label='', axes=None, band=(2.0, 12.0)):
    """
    Diagnostic figure for one unit: raw autocorrelogram with its slow trend, the
    detrended autocorrelogram against the ISI-shuffle band, and the spectrum.

    Read it as: peaks that stay above the shuffled band for several cycles are a
    sustained rhythm; peaks that fall inside the band are explained by ISI regularity.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 3.2))
    ax0, ax1, ax2 = axes

    if res.get('acg') is None or 'lags' not in res:
        for ax in axes:
            ax.text(0.5, 0.5, f'{unit_label}\ninsufficient data', ha='center', va='center')
            ax.set_axis_off()
        return axes

    lags, corrs = res['lags'], res['acg']
    ax0.plot(lags[1:], corrs[1:], color='k', lw=0.8)
    trend = nan_gaussian_smooth(np.where(np.arange(len(corrs)) < 1, np.nan, corrs),
                                0.3 / res['bin_size'])
    ax0.plot(lags[1:], trend[1:], color='tab:orange', lw=1.2, label='slow trend')
    ax0.set(xlabel='lag (s)', ylabel='Pearson r',
            title=f"{unit_label}  {res['rate']:.2f} Hz, n={res['n_spikes']}")
    ax0.legend(fontsize=7, frameon=False)

    det = res.get('acg_smooth', res['acg_detrended'])
    if 'surr_acg_lo' in res:
        ax1.fill_between(lags, res['surr_acg_lo'], res['surr_acg_hi'],
                         color='tab:blue', alpha=0.25, lw=0, label='ISI shuffle 95%')
    ax1.plot(lags, det, color='k', lw=0.8, label='observed')
    period = res['peak_period']
    if np.isfinite(period):
        for k in range(1, int(np.floor(res['window_length'] / period)) + 1):
            ax1.axvline(k * period, color='tab:red', lw=0.5, ls=':')
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.set(xlabel='lag (s)', ylabel='detrended r',
            title=f"period {period * 1000:.0f} ms, run {res['max_run_sig']} cyc, "
                  f"persists {res['persist_time']:.2f} s")
    ax1.legend(fontsize=7, frameon=False)

    ax2.plot(res['freqs'], res['power'], color='k', lw=0.9)
    ax2.axvspan(band[0], band[1], color='tab:green', alpha=0.1, lw=0)
    if np.isfinite(res['peak_freq']):
        ax2.axvline(res['peak_freq'], color='tab:red', lw=0.8)
    ax2.set(xlabel='frequency (Hz)', ylabel='power',
            title=f"peak {res['peak_freq']:.2f} Hz, OS {res['osc_score']:.1f}, "
                  f"p={res['osc_score_p']:.3f}")
    return axes
