"""
Figure: do individual tagged DRN units integrate reward over different timescales?

The honest version of this result is subtler than "two cell types", and the four
panels are ordered to walk from the naive view to what the data actually support:

  A  the preferred-tau histogram -- which LOOKS bimodal
  B  the same preference as a continuous quantity -- which is unimodal, so the
     bimodality in A is an artifact of taking argmax over a 5-point grid
  C  held-out profiles: units classified on half the trials still lean the same
     way on the other half, so the SPREAD in A/B is real even though the classes
     are not
  D  split-half reliability of the preference itself

Colour: the diverging pair encodes timescale preference throughout (warm = fast,
cool = slow), so a colour means one thing across the whole figure. Panel A is a
count, so it uses neutral ink with the untrustworthy segment hatched in gray
rather than either diverging pole. Values verbatim from the reference palette.

Reads tau_heterogeneity_splithalf.csv written by tau_heterogeneity_check.py.
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
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, spearmanr, wilcoxon

from utils.capsule_migration import capsule_directories
from utils.reward_rate import TAUS

# ---- palette (same reference instance as the other figures, used verbatim) ----
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8983'
FAST = '#e34948'   # diverging warm pole -- prefers a short reward window
SLOW = '#2a78d6'   # diverging cool pole -- prefers a long one
MID = '#f0efec'
GRID = '#e6e5e1'

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'persistence_5HT')


def style_axes(ax, grid_axis='y'):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)


def boot_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(pd.Series(x).dropna())
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    b = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5)


# %%
def panel_best_tau(ax, d):
    """A: the naive preferred-tau histogram, with the untrustworthy part marked.

    `argmax` runs over signed rho, so for a unit that is negatively correlated at
    every tau it returns the least-negative one -- an arbitrary choice. Those
    units are stacked separately (hatched) rather than dropped, because where
    they land is exactly what makes the histogram look bimodal: they pile into
    the two extreme bins.
    """
    x = np.arange(len(TAUS))
    pos = [int(((d['best_all'] == t) & ~d['allneg']).sum()) for t in TAUS]
    neg = [int(((d['best_all'] == t) & d['allneg']).sum()) for t in TAUS]

    ax.bar(x, pos, width=0.72, color=INK_2, edgecolor=SURFACE, linewidth=1.0,
           zorder=3, label=f'has a positive reward correlation (n={sum(pos)})')
    # 2px surface gap between stacked segments, per the mark spec
    ax.bar(x, neg, width=0.72, bottom=np.array(pos) + 0.6, color=INK_MUTED,
           edgecolor=SURFACE, linewidth=1.0, hatch='///', alpha=0.75, zorder=3,
           label=f'negative at every $\\tau$ (n={sum(neg)}); argmax arbitrary')

    for xi, p, n in zip(x, pos, neg):
        ax.text(xi, p + n + 1.8, str(p + n), ha='center', va='bottom',
                color=INK_2, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in TAUS])
    ax.set_ylim(0, max(np.array(pos) + np.array(neg)) * 1.30)
    ax.set_xlabel('preferred reward-integration $\\tau$ (trials)', color=INK,
                  fontsize=9)
    ax.set_ylabel('units', color=INK, fontsize=9)
    ax.set_title('A   Preferred $\\tau$ looks bimodal', color=INK, fontsize=10,
                 loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=6.8, loc='upper center', labelcolor=INK_2,
              handlelength=1.2, borderpad=0.2, handletextpad=0.5)
    style_axes(ax)


def panel_tilt(ax, d):
    """B: the same preference without argmax -- one broad mode, not two.

    Plotting the continuous quantity is the check on panel A: if there really
    were a fast class and a slow class, this would be bimodal. It is not, so the
    claim the data support is a spread of timescales, not two types.
    """
    t = (d['all_rr_2'] - d['all_rr_40']).dropna()
    lim = np.abs(t).max() * 1.05
    bins = np.linspace(-lim, lim, 33)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(t, bins=bins)

    for c, xc, w in zip(counts, centers, np.diff(bins)):
        if c:
            ax.bar(xc, c, width=w * 0.88, color=(FAST if xc > 0 else SLOW),
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)

    # KDE on the same scale as the counts, to show the mode structure directly
    grid = np.linspace(-lim, lim, 400)
    kde = gaussian_kde(t.values)(grid) * len(t) * np.diff(bins)[0]
    ax.plot(grid, kde, color=INK, linewidth=1.6, zorder=5)
    n_modes = sum(1 for i in range(1, 399)
                  if kde[i] > kde[i - 1] and kde[i] > kde[i + 1])

    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)
    ax.set_ylim(0, max(counts.max(), kde.max()) * 1.36)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('timescale preference,  $\\rho_{\\tau=2} - \\rho_{\\tau=40}$',
                  color=INK, fontsize=9)
    ax.set_ylabel('units', color=INK, fontsize=9)
    ax.set_title('B   ...but the preference is a continuum', color=INK,
                 fontsize=10, loc='left', fontweight='bold')
    style_axes(ax)

    ax.text(0.985, 0.96, 'prefers\nfast', transform=ax.transAxes, color=FAST,
            fontsize=7.5, ha='right', va='top', fontweight='bold')
    ax.text(0.015, 0.96, 'prefers\nslow', transform=ax.transAxes, color=SLOW,
            fontsize=7.5, ha='left', va='top', fontweight='bold')
    ax.text(0.5, 0.985,
            f'KDE has {n_modes} mode — the dip in A is the 5-point grid,\n'
            f'not two classes',
            transform=ax.transAxes, color=INK_2, fontsize=7, ha='center',
            va='top', zorder=6,
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))


def panel_heldout(ax, d):
    """C: classify on half the trials, measure on the other half.

    This is the panel that makes the spread a result rather than noise: argmax
    over five correlated estimates would put both groups on the same held-out
    profile, and instead the profiles cross.
    """
    x = np.arange(len(TAUS))
    handles = []
    for lab, m, color in (('classified fast', d['best_A'] <= 5, FAST),
                          ('classified slow', d['best_A'] >= 20, SLOW)):
        m = m & ~d['allneg']
        stats = [boot_ci(d.loc[m, f'B_rr_{t}']) for t in TAUS]
        mm = [s[0] for s in stats]
        lo = [s[1] for s in stats]
        hi = [s[2] for s in stats]
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, mm, color=color, linewidth=2.0, marker='o', markersize=8,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3, label=f'{lab} (n={int(m.sum())})')
        ax.text(x[-1] + 0.14, mm[-1], lab.replace('classified ', ''),
                color=color, fontsize=8, va='center', fontweight='bold')
        handles.append(Line2D([], [], color=color, linewidth=2.0,
                              label=f'{lab} (n={int(m.sum())})'))

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in TAUS])
    ax.set_xlim(-0.35, len(TAUS) - 0.35 + 1.0)
    # headroom for the legend, which has to clear the upper CI ribbon; without it
    # the legend lands on the zero rule at the bottom instead
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0), hi + 0.22 * (hi - lo))
    ax.set_xlabel('reward-integration $\\tau$ (trials)', color=INK, fontsize=9)
    ax.set_ylabel('held-out partial $\\rho$ with ITI rate', color=INK, fontsize=9)
    ax.set_title('C   The split replicates on held-out trials', color=INK,
                 fontsize=10, loc='left', fontweight='bold')
    # surface-backed, because it sits over the light CI ribbon
    leg = ax.legend(handles=handles, fontsize=7, loc='upper right',
                    labelcolor=INK_2, handlelength=1.4, frameon=True,
                    facecolor=SURFACE, edgecolor='none', framealpha=1.0)
    leg.set_zorder(6)
    style_axes(ax)
    ax.text(0.02, 0.03, 'classified on odd trials, plotted on even;\n'
            'units negative at every $\\tau$ excluded',
            transform=ax.transAxes, color=INK_MUTED, fontsize=6.8, va='bottom')


def panel_reliability(ax, d):
    """D: is the preference a stable property of the unit?"""
    a = d['A_rr_2'] - d['A_rr_40']
    b = d['B_rr_2'] - d['B_rr_40']
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    tilt = (d['all_rr_2'] - d['all_rr_40'])[ok]
    rho, p = spearmanr(a, b)

    lim = max(np.abs(a).max(), np.abs(b).max()) * 1.06
    ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, linewidth=1.0,
            linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(0, color=GRID, linewidth=0.9, zorder=1)
    ax.axvline(0, color=GRID, linewidth=0.9, zorder=1)

    for m, color in ((tilt > 0, FAST), (tilt <= 0, SLOW)):
        ax.plot(a[m], b[m], linestyle='none', marker='o', markersize=5,
                color=color, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=0.8, alpha=0.75, zorder=3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)
    ax.set_xlabel('preference, odd trials', color=INK, fontsize=9)
    ax.set_ylabel('preference, even trials', color=INK, fontsize=9)
    ax.set_title('D   ...and is stable within a unit', color=INK, fontsize=10,
                 loc='left', fontweight='bold')
    style_axes(ax, grid_axis=None)
    ax.text(0.03, 0.97,
            f'Spearman $\\rho$ = {rho:+.2f}\np = {p:.0e}\nn = {len(a)} units',
            transform=ax.transAxes, color=INK_2, fontsize=8, va='top', zorder=6,
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))


# %%
if __name__ == '__main__':
    d = pd.read_csv(os.path.join(OUT_DIR, 'tau_heterogeneity_splithalf.csv'))
    d['allneg'] = (d[[f'all_rr_{t}' for t in TAUS]] < 0).all(axis=1)

    fig = plt.figure(figsize=(10.5, 8.0), facecolor=SURFACE)
    gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.30,
                  left=0.082, right=0.955, top=0.885, bottom=0.085)

    panel_best_tau(fig.add_subplot(gs[0, 0]), d)
    panel_tilt(fig.add_subplot(gs[0, 1]), d)
    panel_heldout(fig.add_subplot(gs[1, 0]), d)
    panel_reliability(fig.add_subplot(gs[1, 1]), d)

    fig.suptitle('Tagged DRN units integrate reward over a range of timescales, '
                 'not one', color=INK, fontsize=12, fontweight='bold', x=0.082,
                 ha='left', y=0.963)
    fig.text(0.082, 0.923,
             f'{len(d)} units, {d["session"].nunique()} sessions; ITI firing rate, '
             'session drift partialled out, curated drift window applied; '
             '$\\tau$ preference is not explained by drift, session or trial count',
             color=INK_2, fontsize=8.5, ha='left')

    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'tau_heterogeneity.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f'wrote {path}')

    # table view, so nothing depends on reading colour off the figure
    rows = []
    for t in TAUS:
        v = d[f'all_rr_{t}']
        rows.append({'tau': t, 'n_preferring': int((d['best_all'] == t).sum()),
                     'n_preferring_pos_signal': int(((d['best_all'] == t)
                                                     & ~d['allneg']).sum()),
                     'median_rho': v.median(), 'frac_pos': (v > 0).mean()})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(OUT_DIR, 'tau_heterogeneity_table.csv'), index=False)
    print(tbl.to_string(index=False))

    tilt = (d['all_rr_2'] - d['all_rr_40']).dropna()
    print(f'\ntilt: median={tilt.median():+.3f} sd={tilt.std():.3f} '
          f'frac_fast={(tilt > 0).mean():.2f}  '
          f'wilcoxon p={wilcoxon(tilt).pvalue:.3g}')
