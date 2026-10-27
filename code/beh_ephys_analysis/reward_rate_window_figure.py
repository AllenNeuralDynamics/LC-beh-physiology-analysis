"""
Figure: tonic ITI rate vs reward rate, as a function of the integration window.

reward_rate_figures.py panel A sweeps five EWMA time constants. This is the same
question with a boxcar instead -- the mean reward over the last N responded
trials, N = 1..60 -- so the window has a hard edge and N reads directly as
"trials of history".

  A  median per-unit rho vs window, three ways: raw, session drift removed, and
     drift plus single-trial reward lags 1-3 removed (phasic carryover control;
     only defined for N > 3, since smaller windows are nothing but those lags)
  B  every unit's drift-controlled rho across windows, sorted by its preferred
     window, so the population summary in A can be checked against the units

Causality as in utils/reward_rate.py: the window for trial n ends at n-1. The
burn-in uses whatever history exists (min_periods=1), matching the adjust=True
EWMA, so every window is evaluated on the same trials.

Writes reward_rate_window.{pdf,png}, reward_rate_window_unit_rho.csv and a
table of the population summary.
"""
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, wilcoxon

from reward_rate_tonic_analysis import (load_with_regressors, unit_frames, FR,
                                        MIN_TRIALS)
from reward_rate_figures import (SURFACE, INK, INK_2, INK_MUTED, DOWN, UP, MID,
                                 OUT_DIR, style_axes, boot_ci)
from utils.reward_rate import partial_spearman

WINDOWS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60)
LAGS = ['rew_lag1', 'rew_lag2', 'rew_lag3']


def add_window_rates(tbl, windows=WINDOWS):
    """rrw_<N>: causal mean reward over the last N responded trials.

    One session at a time, on responded trials only, then mapped back onto the
    full trial index -- the same convention as add_reward_rate().
    """
    tbl = tbl.copy()
    resp = tbl['outcome'].notna()
    r = pd.Series(tbl.loc[resp, 'outcome'].astype(float).values)
    for n in windows:
        tbl[f'rrw_{n}'] = np.nan
        tbl.loc[resp, f'rrw_{n}'] = r.rolling(n, min_periods=1).mean().shift(1).values
    return tbl


def load_with_windows():
    tr = load_with_regressors()
    beh = (tr[['session', 'trial_ind', 'outcome']]
             .groupby(['session', 'trial_ind']).first().reset_index()
             .sort_values(['session', 'trial_ind']))
    beh = pd.concat([add_window_rates(g) for _s, g in beh.groupby('session')],
                    ignore_index=True)
    cols = [f'rrw_{n}' for n in WINDOWS]
    return tr.merge(beh[['session', 'trial_ind'] + cols],
                    on=['session', 'trial_ind'], how='left')


def unit_rhos(tr):
    rows = []
    for s, u, g in unit_frames(tr):
        rec = {'session': s, 'unit_id': u}
        for n in WINDOWS:
            c = f'rrw_{n}'
            d = g[[FR, c]].dropna()
            rec[f'raw_{n}'] = (spearmanr(d[FR], d[c]).statistic
                               if len(d) >= MIN_TRIALS else np.nan)
            rec[f'dpart_{n}'] = partial_spearman(g[FR], g[c], g[['trial_ind']])
            rec[f'lagpart_{n}'] = (partial_spearman(g[FR], g[c],
                                                    g[LAGS + ['trial_ind']])
                                   if n > len(LAGS) else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


# %%
def panel_sweep(ax, u):
    x = np.log(WINDOWS)
    series = (('raw', INK_MUTED, 'raw'),
              ('dpart', DOWN, 'drift removed'),
              ('lagpart', UP, 'drift + lags 1-3 removed'))
    ends = []
    for pref, color, lab in series:
        stats = [boot_ci(u[f'{pref}_{n}']) for n in WINDOWS]
        ok = np.array([np.isfinite(m) for m, _l, _h in stats])
        m, lo, hi = (np.array(v)[ok] for v in zip(*stats))
        ax.fill_between(x[ok], lo, hi, color=color, alpha=0.16, linewidth=0,
                        zorder=2)
        ax.plot(x[ok], m, color=color, linewidth=2.0, marker='o', markersize=8,
                markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=2, label=lab, zorder=3)
        ends.append([m[-1], color, lab])

    # direct labels at the right edge, pushed apart if the endpoints converge
    ends.sort(key=lambda e: e[0])
    lo_y, hi_y = ax.get_ylim()
    gap = 0.06 * (hi_y - lo_y)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, color, lab in ends:
        ax.text(x[-1] + 0.12, y, lab, color=color, fontsize=8, va='center',
                fontweight='bold' if color != INK_MUTED else 'normal')

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
    ticks = [1, 2, 3, 5, 10, 20, 40, 60]
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(x[0] - 0.2, x[-1] + 1.9)
    ax.set_xlabel('reward-rate window N (trials, mean of $n\\!-\\!N..n\\!-\\!1$)',
                  color=INK, fontsize=9)
    ax.set_ylabel('median $\\rho$ with ITI rate', color=INK, fontsize=9)
    ax.set_title('A   Correlation by integration window', color=INK,
                 fontsize=10, loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='upper right', labelcolor=INK_2,
              handlelength=1.4)
    style_axes(ax)


def panel_units(ax, cax, u, pref='dpart',
                title='B   Per-unit, drift removed', cb_label='partial $\\rho$'):
    cols = [f'{pref}_{n}' for n in WINDOWS]
    m = u[cols].dropna()
    vals = m.values
    # sort by the window each unit prefers, then by how strongly
    order = np.lexsort((-np.nanmax(vals, 1), np.nanargmax(vals, 1)))
    vals = vals[order]
    lim = np.nanpercentile(np.abs(vals), 98)
    cmap = LinearSegmentedColormap.from_list('div', [DOWN, MID, UP])
    im = ax.imshow(vals, aspect='auto', cmap=cmap, vmin=-lim, vmax=lim,
                   interpolation='nearest')
    # each unit's preferred window, so the sort order can be read off the panel
    ax.scatter(np.nanargmax(vals, 1), np.arange(len(vals)), s=2, color=INK,
               linewidths=0, zorder=3)
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([str(n) for n in WINDOWS], fontsize=7)
    ax.set_xlabel('reward-rate window N (trials)', color=INK, fontsize=9)
    ax.set_ylabel(f'units ({len(vals)}), sorted by preferred window',
                  color=INK, fontsize=9)
    ax.set_yticks([])
    ax.set_title(title, color=INK, fontsize=10, loc='left', fontweight='bold')
    style_axes(ax, grid_axis=None)
    cb = plt.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=7, width=0.8)
    cb.set_label(cb_label, color=INK, fontsize=8)


def plot_raw_units(u):
    """Standalone figure: per-unit raw rho across windows, no drift control."""
    fig = plt.figure(figsize=(5.2, 5.0), facecolor=SURFACE)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.04], wspace=0.08,
                  left=0.1, right=0.86, top=0.92, bottom=0.12)
    panel_units(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), u,
                pref='raw', title='Per-unit reward rate vs ITI rate (raw)',
                cb_label='Spearman $\\rho$')
    fig.text(0.02, 0.015, 'no drift control; dot marks each unit\'s peak window',
             color=INK_MUTED, fontsize=7)
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'reward_rate_window_raw_units.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f'wrote {path}')


def plot_raw(u):
    """Standalone figure: the raw (uncontrolled) curve alone."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor=SURFACE)
    x = np.log(WINDOWS)
    m, lo, hi = (np.array(v) for v in
                 zip(*[boot_ci(u[f'raw_{n}']) for n in WINDOWS]))
    ax.fill_between(x, lo, hi, color=DOWN, alpha=0.16, linewidth=0, zorder=2)
    ax.plot(x, m, color=DOWN, linewidth=2.0, marker='o', markersize=8,
            markerfacecolor=DOWN, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3)
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
    ticks = [1, 2, 3, 5, 10, 20, 40, 60]
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax.set_xlabel('reward-rate window N (trials)', color=INK, fontsize=9)
    ax.set_ylabel('median Spearman $\\rho$\n(reward rate vs ITI rate)',
                  color=INK, fontsize=9)
    ax.set_title('Reward rate vs tonic firing, by window', color=INK,
                 fontsize=10, loc='left', fontweight='bold')
    style_axes(ax)
    fig.text(0.02, 0.015, f'{len(u)} units; median and bootstrap 95% CI',
             color=INK_MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'reward_rate_window_raw.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f'wrote {path}')


# %%
if __name__ == '__main__':
    tr = load_with_windows()
    u = unit_rhos(tr)
    u.to_csv(os.path.join(OUT_DIR, 'reward_rate_window_unit_rho.csv'), index=False)

    fig = plt.figure(figsize=(12.0, 5.2), facecolor=SURFACE)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 1.0, 0.04],
                  wspace=0.22, left=0.07, right=0.95, top=0.80, bottom=0.13)
    panel_sweep(fig.add_subplot(gs[0, 0]), u)
    panel_units(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), u)

    fig.suptitle('Tonic DRN firing vs reward rate across integration windows',
                 color=INK, fontsize=12, fontweight='bold', x=0.07, ha='left',
                 y=0.965)
    fig.text(0.07, 0.905,
             f'{len(u)} opto-tagged units, {u["session"].nunique()} sessions; '
             'ITI rate; boxcar window over responded trials ending at $n\\!-\\!1$; '
             'median and bootstrap 95% CI',
             color=INK_2, fontsize=8.5, ha='left')

    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'reward_rate_window.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f'wrote {path}')

    plot_raw(u)
    plot_raw_units(u)

    tbl = []
    for pref in ('raw', 'dpart', 'lagpart'):
        for n in WINDOWS:
            v = u[f'{pref}_{n}'].dropna()
            if len(v) < 3:
                continue
            m, lo, hi = boot_ci(v)
            tbl.append({'control': pref, 'window': n, 'n_units': len(v),
                        'median_rho': m, 'ci_lo': lo, 'ci_hi': hi,
                        'frac_pos': (v > 0).mean(),
                        'wilcoxon_p': wilcoxon(v).pvalue})
    tbl = pd.DataFrame(tbl)
    tbl.to_csv(os.path.join(OUT_DIR, 'reward_rate_window_table.csv'), index=False)
    print(tbl.to_string(index=False))
