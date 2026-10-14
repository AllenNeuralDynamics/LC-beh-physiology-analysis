"""Within-insertion DV gradients for the features whose DV effect survives the controls.

Each unit's DV and feature value are expressed relative to its own insertion's mean,
so nothing in the plot can come from insertions differing from each other. Per
panel: units (dots), each insertion's own least-squares line (n >= 3 units), and
the pooled within-insertion slope (insertion fixed effects) with a hierarchical
bootstrap CI (resample mice, then insertions within mouse), plus how many
insertions slope the same way as the pool.

Within an insertion DV and AP move together (probe tilt; r ~ 0.95), so these
slopes are slopes along the probe track and cannot separate DV from AP.

Run:  python session_combine/figure_preparation/within_insertion_dv.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.capsule_migration import capsule_directories
from utils.spatial_session_bias import mouse_from_session

BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
N_BOOT = 2000
SEED = 42
MIN_UNITS_LINE = 3

capsule_dirs = capsule_directories()
SUMMARY_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']),
                           'SUMMARY_DRN_spatial_organization')
SRC_DIR = os.path.join(SUMMARY_DIR, '05_stats_and_source_data')
OUT_DIR = os.path.join(SUMMARY_DIR, '06_session_bias_controls')

# (dataset, source file, rank-transform as the stats did). Features are picked from
# session_bias_stats_<dataset>.csv: DV p < ALPHA with insertions kept together.
DATASETS = [
    ('waveform_low_DRN', 'combined_features_waveform_low_DRN.csv', False),
    ('basic_ephys_all_DRN', 'features_combined_basic_ephys_all_DRN.pkl', True),
    ('beh_all_DRN', 'features_combined_beh_all_DRN.pkl', False),
]
DUPLICATES = {'basic_ephys_all_DRN': ['A1', 'A3', 'peak3']}
ALPHA = 0.05


def dv_survivors(label):
    s = pd.read_csv(os.path.join(OUT_DIR, f'session_bias_stats_{label}.csv'))
    s = s[~s['feature'].isin(DUPLICATES.get(label, [])) & (s['p_DV_session_perm'] < ALPHA)]
    return s.sort_values('p_DV_session_perm')['feature'].tolist()
SHORT = {'waveform_low_DRN': 'waveform', 'basic_ephys_all_DRN': 'firing / ACG',
         'beh_all_DRN': 'task coding'}

# Light-mode tokens from the dataviz reference palette.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8985'
GRID = '#e4e3df'
AGREE = '#2a78d6'
DISAGREE = '#eb6834'


def load(path, rank, feature):
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_pickle(path)
    X = df[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    df = df.assign(DV=X[:, 2], mouse=mouse_from_session(df['session'].to_numpy()))
    y = pd.to_numeric(df[feature], errors='coerce').to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    d = df.loc[ok, ['session', 'mouse', 'DV']].copy()
    d['y'] = rankdata(y[ok]) if rank else y[ok]
    d['dv_c'] = d.DV - d.groupby('session').DV.transform('mean')
    d['y_c'] = d.y - d.groupby('session').y.transform('mean')
    return d


def fe_slope(dv_c, y_c):
    den = (dv_c ** 2).sum()
    return np.nan if den <= 0 else (dv_c * y_c).sum() / den


def hier_boot(d, rng):
    """Within-insertion slope, resampling mice then insertions within mouse."""
    by_mouse = {m: g.session.unique() for m, g in d.groupby('mouse')}
    groups = {s: g for s, g in d.groupby('session')}
    mice = list(by_mouse)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        parts = []
        for m in rng.choice(mice, len(mice)):
            ss = by_mouse[m]
            parts += [groups[s] for s in rng.choice(ss, len(ss))]
        bd = pd.concat(parts)
        out[b] = fe_slope(bd.dv_c.to_numpy(), bd.y_c.to_numpy())
    return out


def panel(ax, d, title, ylab, rng):
    b = fe_slope(d.dv_c.to_numpy(), d.y_c.to_numpy())
    boot = hier_boot(d, rng)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    p = 2 * min(np.nanmean(boot <= 0), np.nanmean(boot >= 0))

    ax.scatter(d.dv_c, d.y_c, s=10, color=MUTED, alpha=0.45, linewidth=0, zorder=1)
    n_same = n_lines = 0
    for _, g in d.groupby('session'):
        if len(g) < MIN_UNITS_LINE or g.dv_c.std() <= 0:
            continue
        bs = fe_slope(g.dv_c.to_numpy(), g.y_c.to_numpy())
        same = np.sign(bs) == np.sign(b)
        n_lines += 1
        n_same += int(same)
        x = np.array([g.dv_c.min(), g.dv_c.max()])
        ax.plot(x, bs * x, color=AGREE if same else DISAGREE, linewidth=1.3, alpha=0.8, zorder=2)
    x = np.array([d.dv_c.min(), d.dv_c.max()])
    ax.plot(x, b * x, color=INK, linewidth=3, zorder=4, solid_capstyle='round')
    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=GRID, linewidth=1, zorder=0)
    sign_p = binomtest(n_same, n_lines, 0.5).pvalue if n_lines else np.nan
    p_txt = f'< {1 / N_BOOT:g}' if p < 1 / N_BOOT else f'{p:.3f}'
    ax.text(0.02, 0.98,
            f'within-insertion slope {b:.3g} per mm\n95% CI [{lo:.3g}, {hi:.3g}], p {p_txt}\n'
            f'{n_same} of {n_lines} insertions same sign (sign test p {sign_p:.3f})',
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2))
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=6)
    ax.set_xlabel('DV relative to insertion mean (mm, + = dorsal)', fontsize=9, color=INK_2)
    ax.set_ylabel(ylab, fontsize=9, color=INK_2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    lim = np.nanpercentile(np.abs(d.y_c), 99) * 1.25
    ax.set_ylim(-lim, lim * 1.35)
    return dict(feature=title.split('\n')[0], slope=b, ci_low=lo, ci_high=hi, p_hier=p,
                n_same=n_same, n_insertions_line=n_lines, sign_test_p=sign_p,
                n_units=len(d), n_insertions=d.session.nunique(), n_mice=d.mouse.nunique())


def main():
    rng = np.random.default_rng(SEED)
    items = [(lab, f, r, feat) for lab, f, r in DATASETS for feat in dv_survivors(lab)]
    n_cols = 4
    n_rows = int(np.ceil(len(items) / n_cols))
    h = 4.0 * n_rows + 2.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, h), facecolor='white', squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.985, top=1 - 1.55 / h, bottom=1.45 / h,
                        hspace=0.45, wspace=0.3)
    rows = []
    for ax, (lab, f, rank, feat) in zip(axes.ravel(), items):
        d = load(os.path.join(SRC_DIR, f), rank, feat)
        title = (f'{feat}\n{SHORT[lab]} · {len(d)} units, {d.session.nunique()} insertions, '
                 f'{d.mouse.nunique()} mice')
        ylab = ('rank, relative to insertion mean' if rank
                else 'value, relative to insertion mean')
        r = panel(ax, d, title, ylab, rng)
        r['dataset'] = lab
        rows.append(r)
        print(f"  {feat}: slope {r['slope']:.3g} [{r['ci_low']:.3g}, {r['ci_high']:.3g}] "
              f"p {r['p_hier']:.3f}; {r['n_same']}/{r['n_insertions_line']} same sign")
    for ax in axes.ravel()[len(items):]:
        ax.set_visible(False)

    fig.text(0.05, 1 - 0.3 / h, 'Do the DV gradients hold inside a single insertion?',
             fontsize=17, weight='bold', color=INK, va='top')
    fig.text(0.05, 1 - 0.72 / h,
             'Each unit\'s DV and value are taken relative to its own insertion\'s mean, so '
             'differences between insertions cannot contribute. Features are those whose DV '
             'effect survives the insertion-aware tests.\n'
             'Caveat: within an insertion DV and AP change together (probe tilt, r ≈ 0.95, '
             '≈ 0.37 mm AP per mm DV), so these are gradients along the probe track; they cannot '
             'tell DV apart from AP.',
             fontsize=10, color=INK_2, va='top')
    fig.legend(handles=[
        Line2D([], [], color=INK, linewidth=3, label='pooled within-insertion slope'),
        Line2D([], [], color=AGREE, linewidth=1.3, label=f'one insertion (≥ {MIN_UNITS_LINE} units), same sign as pooled'),
        Line2D([], [], color=DISAGREE, linewidth=1.3, label='one insertion, opposite sign'),
        Line2D([], [], marker='o', linestyle='', color=MUTED, alpha=0.6, label='unit'),
    ], loc='lower left', bbox_to_anchor=(0.045, 0.03), ncol=4, frameon=False, fontsize=10,
        labelcolor=INK_2)
    fig.text(0.05, 0.01,
             f'CI and p: hierarchical bootstrap of the pooled slope ({N_BOOT} resamples; mice, '
             'then insertions within mouse). Sign test is two-sided binomial over insertions '
             'with lines. Firing/ACG features rank-transformed as in the stats. '
             'Source: figure_preparation/within_insertion_dv.py',
             fontsize=8, color=MUTED, ha='left', va='bottom')

    path = os.path.join(OUT_DIR, 'within_insertion_dv')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}', dpi=200)
    pd.DataFrame(rows).to_csv(path + '.csv', index=False)
    print('wrote', path + '.png')


if __name__ == '__main__':
    main()
