"""
Figures for the claim: these DRN units are regular spikers, not oscillators.

The argument a reader needs is not "the observed autocorrelogram matches a renewal null"
on its own -- that only shows the data are consistent with renewal, not that the analysis
could tell the difference. It needs the counterfactual next to it: a synthetic unit with
the SAME firing rate and the SAME ISI CV as the real one, but genuinely phase-locked. If
its autocorrelogram looks obviously different, then the real unit's shape is diagnostic.

Figure 1, one example unit, 3x3:
    rows    = real unit / matched renewal simulation / matched phase-locked clock
    columns = ISI histogram / autocorrelogram / variance growth
    The real row is indistinguishable from renewal and clearly unlike the clock, in all
    three columns, while column 1 (the ISI histogram) is matched by construction.

Figure 2, population, 3 panels:
    A  variance growth for every unit, against the renewal and clock predictions
    B  ISI CV vs drift-detrended serial correlation, against the -0.5 clock reference
    C  oscillation score vs its own shuffle z, showing the score alone says nothing

Column 3 / panel A is the load-bearing one and needs no autocorrelogram at all. Let T_k be
the time from a spike to the k-th spike after it. For any renewal process the intervals are
independent, so Var(T_k) = k * Var(ISI) exactly -- a straight line through the origin, and
1.0 after dividing by k * Var(ISI). For a clock with phase jitter, T_k = k*P + j_{k+1} - j_1,
so Var(T_k) = 2 * Var(j) is CONSTANT in k and the normalised curve falls off as 1/k. This is
the direct statement of "the phase does not drift", and it is independent of bin size,
window length and the detrending choices that the autocorrelogram route depends on.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

sys.path.append('/root/capsule/code/beh_ephys_analysis')
from rhythmicity_DRN import UNITS_OF_INTEREST, spontaneous_window, unit_window, BIN_SIZE
from utils.beh_functions import get_unit_tbl
from utils.rhythmicity import acg, detrend_acg, isi_shuffle, nan_gaussian_smooth

warnings.filterwarnings('ignore')

METRICS = '/root/capsule/scratch/results/uoi_rhythmicity.pkl'
OUT_DIR = '/root/capsule/scratch/results/rhythmicity_figs'
WINDOW = 4.0
MAX_K = 40
DETREND_N = 201
MIN_SPIKES = 3000
N_BAND = 60

# The example unit for figure 1: the most convincing-looking oscillation in the set, which
# rings visibly for the full 4 s. Making the argument on the strongest case is the point.
EXAMPLE = ('behavior_835444_2026-02-17_13-56-45', 141)


def variance_growth(spike_times, max_k=MAX_K, detrend_n=DETREND_N):
    """
    Var(T_k) / (k * Var(ISI)) for k = 1..max_k, where T_k is the time from a spike to the
    k-th spike after it.

    Renewal gives 1.0 at every k. A phase-locked clock gives 1/k.

    Slow rate drift pushes the raw curve far ABOVE 1 (a 40% drift reaches 26 by k=40), so
    it has to be removed before the curve is readable. Dividing each interval by a running
    median of `detrend_n` intervals does that, but the window matters: the running median
    also suppresses genuine variance accumulation once k approaches it, which biases the
    curve DOWNWARD, i.e. toward the clock. Calibrated on synthetic trains at k <= 40:

        detrend_n   renewal (k=40)   renewal +40% drift   clock
              51          0.51              0.48          0.31   over-corrects, unusable
             201          1.00              0.86          0.19   <- used here
             501          1.10              1.04          0.34

    201 keeps renewal within 2% undrifted and no lower than 0.86 under heavy drift, against
    0.12-0.23 for a clock, so the two stay separated by ~4x. Do not raise max_k without
    re-checking this table.

    Returns:
    ndarray of length max_k, or None if there are too few spikes.
    """
    s = np.sort(np.asarray(spike_times, dtype=float))
    if len(s) < max_k + 20:
        return None
    if detrend_n:
        isis = np.diff(s)
        local = median_filter(isis, size=min(detrend_n, len(isis)), mode='nearest')
        isis = isis / np.where(local > 0, local, np.nan) * np.nanmean(local)
        if not np.all(np.isfinite(isis)):
            return None
        s = np.concatenate([[0.0], np.cumsum(isis)])
    var_isi = np.var(np.diff(s))
    if var_isi <= 0:
        return None
    out = np.full(max_k, np.nan)
    for k in range(1, max_k + 1):
        out[k - 1] = np.var(s[k:] - s[:-k]) / (k * var_isi)
    return out


def matched_clock(rate, cv, duration, rng):
    """
    Phase-locked clock with the same rate and the same ISI CV as a target unit.

    ISI_k = P + j_{k+1} - j_k with white jitter j, so Var(ISI) = 2 Var(j) and
    cv = sqrt(2) * sigma_j / P. Matching both first-order statistics is what makes this a
    fair counterfactual: it differs from the real unit only in second-order structure.
    """
    period = 1.0 / rate
    sigma = cv * period / np.sqrt(2.0)
    n = int(duration * rate) + 2
    t = np.arange(n) * period + rng.normal(0, sigma, n)
    t = np.sort(t)
    return t[(t >= 0) & (t <= duration)]


def matched_renewal(spike_times, start, end, rng):
    """ISI shuffle of the real unit: a renewal process with its exact ISI distribution."""
    return isi_shuffle(spike_times, start, end, rng)


def unit_spikes(session, unit_id):
    """Spike times and analysis window for one unit, as the metrics run used them."""
    window = spontaneous_window(session, 'raw')
    if window is None:
        return None, None, None
    tbl = get_unit_tbl(session, 'raw', summary=False)
    row = tbl[tbl['unit_id'] == unit_id]
    if len(row) == 0:
        return None, None, None
    start, end = unit_window(session, unit_id, 'raw', *window)
    s = np.sort(np.asarray(row['spike_times'].values[0], dtype=float))
    return s[(s >= start) & (s <= end)], start, end


def _acg_trace(spikes, start, end):
    """Detrended, lightly smoothed one-sided ACG on the standard grid."""
    corrs, lags, _ = acg(spikes, BIN_SIZE, WINDOW, start, end)
    det = detrend_acg(corrs, BIN_SIZE)
    return lags, nan_gaussian_smooth(det, 1.5)


def figure_one(metrics, out_dir):
    """Real unit vs matched renewal vs matched clock, three ways of looking at each."""
    session, unit_id = EXAMPLE
    spikes, start, end = unit_spikes(session, unit_id)
    row = metrics[(metrics.session == session) & (metrics.unit_id == unit_id)].iloc[0]
    rate, cv, duration = row['rate'], row['cv_isi'], end - start
    rng = np.random.default_rng(0)

    trains = [
        (f'real unit {unit_id}\n{rate:.1f} Hz, CV {cv:.2f}', spikes, 'k'),
        ('matched renewal\n(ISI shuffle of the real unit)',
         matched_renewal(spikes, start, end, rng), 'tab:blue'),
        (f'matched phase-locked clock\nsame rate, same CV',
         matched_clock(rate, cv, duration, rng) + start, 'tab:red'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5))
    for r, (label, train, color) in enumerate(trains):
        isis = np.diff(train)
        isis = isis[(isis > 0) & (isis < 1.0)]

        ax = axes[r, 0]
        ax.hist(1000 * isis, bins=np.linspace(0, 400, 81), color=color, alpha=.75)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlabel('ISI (ms)')
        cv_r = np.std(np.diff(train)) / np.mean(np.diff(train))
        ax.set_title(f'ISI histogram   CV {cv_r:.2f}', fontsize=10)

        ax = axes[r, 1]
        lags, trace = _acg_trace(train, start, end)
        if r == 0:
            # 95% band over many shuffles, not the single realisation in row 2: one
            # surrogate is as noisy as the data, so only the band shows the real overlap.
            band = np.array([_acg_trace(isi_shuffle(train, start, end, rng), start, end)[1]
                             for _ in range(N_BAND)])
            ax.fill_between(lags[1:], np.nanpercentile(band, 2.5, axis=0)[1:],
                            np.nanpercentile(band, 97.5, axis=0)[1:],
                            color='tab:blue', alpha=.45, lw=0,
                            label='ISI shuffle 95%')
            ax.legend(fontsize=7, loc='upper right')
        ax.plot(lags[1:], trace[1:], color=color, lw=.7)
        ax.axhline(0, color='gray', lw=.5)
        ax.set_xlabel('lag (s)')
        ax.set_title('autocorrelogram', fontsize=10)
        ax.set_xlim(0, WINDOW)

        ax = axes[r, 2]
        k = np.arange(1, MAX_K + 1)
        ax.axhline(1.0, color='tab:blue', ls='--', lw=1.4, label='renewal: 1.0')
        ax.plot(k, 1 / k, color='tab:red', ls=':', lw=1.4, label='clock: 1/k')
        raw = variance_growth(train, detrend_n=None)
        if raw is not None:
            ax.plot(k, raw, color=color, lw=.9, ls='-.', alpha=.55,
                    label='raw (rate drift)')
        vg = variance_growth(train)
        if vg is not None:
            ax.plot(k, vg, color=color, lw=1.8, marker='o', ms=2.5,
                    label='drift-corrected')
        ax.set_xlabel('k (spikes ahead)')
        ax.set_ylabel(r'Var($T_k$) / (k Var(ISI))')
        ax.set_title('variance growth', fontsize=10)
        ax.set_ylim(0, 2.6)
        ax.legend(fontsize=7, loc='upper right' if r == 0 else 'center right')

    fig.suptitle(f'{session} unit {unit_id}: the autocorrelogram rings for 4 s, but it is '
                 f'reproduced by the ISI distribution alone,\nand the phase diffuses '
                 f'instead of locking -- a matched clock with the same rate and CV looks '
                 f'nothing like it', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(out_dir, 'fig1_example_unit_counterfactual.png')
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def figure_two(metrics, out_dir):
    """Population summary: variance growth, serial correlation, and score vs shuffle."""
    d = metrics[metrics.n_spikes > MIN_SPIKES].copy()
    k = np.arange(1, MAX_K + 1)

    cache = os.path.join(out_dir, 'variance_growth_curves.npy')
    if os.path.exists(cache):
        curves = np.load(cache)
    else:
        curves = []
        for session in UNITS_OF_INTEREST:
            sub = d[d.session == session]
            if len(sub) == 0:
                continue
            for unit_id in sub.unit_id:
                spikes, start, end = unit_spikes(session, unit_id)
                if spikes is None:
                    continue
                vg = variance_growth(spikes)
                if vg is not None:
                    curves.append(vg)
            print(f'  variance growth: {session}', flush=True)
        curves = np.array(curves)
        np.save(cache, curves)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    # 0.86 is the worst case for renewal under heavy drift at this detrend window, so the
    # band below it is the region only a locked process can reach. See variance_growth.
    ax.axhspan(0, 0.86, color='tab:red', alpha=.10)
    for c in curves:
        ax.plot(k, c, color='k', lw=.5, alpha=.35)
    ax.plot(k, np.nanmedian(curves, axis=0), color='k', lw=2.4,
            label=f'observed median (n={len(curves)})')
    ax.axhline(1.0, color='tab:blue', ls='--', lw=1.8, label='renewal prediction')
    ax.plot(k, 1 / k, color='tab:red', ls=':', lw=2, label='phase-locked clock')
    ax.set_xlabel('k (spikes ahead)')
    ax.set_ylabel(r'Var($T_k$) / (k Var(ISI))')
    ax.set_title('A  every unit accumulates phase variance;\nnone enters the locked '
                 'region (shaded)', fontsize=10)
    ax.text(.03, .60, 'above 1.0 = residual\nrate drift, the opposite\ndirection from a clock',
            transform=ax.transAxes, fontsize=7.5, color='dimgray', va='top')
    ax.set_ylim(0, 1.6)
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1]
    ax.axhspan(-0.55, -0.45, color='tab:red', alpha=.18)
    ax.axhline(-0.5, color='tab:red', ls=':', lw=2)
    ax.text(0.98, -0.47, 'phase-locked clock', color='tab:red', fontsize=8,
            ha='right', va='bottom', transform=ax.get_yaxis_transform())
    ax.axhline(0.0, color='tab:blue', ls='--', lw=1.8)
    ax.text(0.98, 0.02, 'renewal', color='tab:blue', fontsize=8,
            ha='right', va='bottom', transform=ax.get_yaxis_transform())
    sc = ax.scatter(d.cv_isi, d.isi_serial_1_det, c=d.rate, cmap='viridis',
                    s=34, edgecolor='k', linewidth=.4, zorder=3)
    plt.colorbar(sc, ax=ax, label='firing rate (Hz)')
    ax.set_xlabel('ISI CV')
    ax.set_ylabel('detrended lag-1 ISI serial correlation')
    ax.set_title('B  no unit approaches the clock signature', fontsize=10)
    ax.set_ylim(-0.62, 0.45)
    ax.set_xscale('log')

    ax = axes[2]
    ax.axhspan(-1.96, 1.96, color='tab:blue', alpha=.15)
    ax.axhline(0, color='tab:blue', ls='--', lw=1.5)
    ax.axhline(1.96, color='gray', lw=.8)
    ax.scatter(d.osc_score, d.osc_score_z, s=34, color='k', edgecolor='w', linewidth=.5)
    ax.set_xscale('log')
    ax.set_xlabel('oscillation score (peak / mean power)')
    ax.set_ylabel('z of log(oscillation score) vs ISI shuffle')
    ax.set_title('C  a 1000-fold range of scores, none beyond its own shuffle', fontsize=10)
    ax.text(0.03, 0.95, 'shaded: indistinguishable\nfrom renewal', transform=ax.transAxes,
            fontsize=8, va='top', color='tab:blue')

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig2_population.png')
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = pd.read_pickle(METRICS)
    print(figure_one(metrics, OUT_DIR), flush=True)
    print(figure_two(metrics, OUT_DIR), flush=True)
