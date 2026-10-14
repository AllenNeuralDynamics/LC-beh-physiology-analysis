"""Panel h of session_bias_diagnostics, for every feature significant under the naive test.

A feature is included if any of its global-shuffle p-values (omnibus R², or the ML,
AP or DV partial effect) in session_bias_stats_<dataset>.csv is < 0.05. For each,
the gradient slope on ML / AP / DV is bootstrapped under four resampling schemes
(units; insertions; mice; mice -> insertions -> units) with the same function,
n_boot and seed as spatial_session_bias_report.py, so the focus feature reproduces
bootstrap_scheme_comparison_<dataset>.csv.

Run:  python session_combine/figure_preparation/bootstrap_schemes_all_features.py [dataset]
      dataset defaults to waveform_low_DRN (the only one with a CSV of raw features)
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.capsule_migration import capsule_directories
from utils.spatial_session_bias import AXIS_NAMES, bootstrap_scheme_comparison, mouse_from_session

BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
N_BOOT = 1000
SEED = 42
ALPHA = 0.05

capsule_dirs = capsule_directories()
SUMMARY_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']),
                           'SUMMARY_DRN_spatial_organization')
SRC_DIR = os.path.join(SUMMARY_DIR, '05_stats_and_source_data')
OUT_DIR = os.path.join(SUMMARY_DIR, '06_session_bias_controls')

SCHEMES = ['unit_only (naive)', 'cluster_session', 'cluster_mouse',
           'hierarchical_mouse_session_unit']
SCHEME_NAMES = ['neurons (naive)', 'insertions', 'mice', 'mice → insertions → neurons']

# Light-mode tokens from the dataviz reference palette; gray = naive, blue ramp
# is ordinal (darker = more of the hierarchy resampled), as in the summary slide.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8985'
GRID = '#e4e3df'
SCHEME_COLORS = [MUTED, '#3987e5', '#1c5cab', '#0d366b']


def naive_significant(stats):
    cols = ['p_r2_global_perm'] + [f'p_{a}_global_perm' for a in AXIS_NAMES]
    sig = stats[cols] < ALPHA
    keep = stats[sig.any(axis=1)].copy()
    keep['naive_axes'] = [[a for a in AXIS_NAMES if r[f'p_{a}_global_perm'] < ALPHA]
                          for _, r in keep.iterrows()]
    return keep.sort_values('p_r2_global_perm')


def run(label):
    stats = pd.read_csv(os.path.join(OUT_DIR, f'session_bias_stats_{label}.csv'))
    df = pd.read_csv(os.path.join(SRC_DIR, f'combined_features_{label}.csv'))
    X = df[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    X[:, 0] = np.abs(X[:, 0])
    s = df['session'].to_numpy()
    mouse = mouse_from_session(s)
    feats = naive_significant(stats)
    rows = []
    for f in feats['feature']:
        y = pd.to_numeric(df[f], errors='coerce').to_numpy(float)
        t = bootstrap_scheme_comparison(X, y, s, mouse, n_boot=N_BOOT, seed=SEED)
        t.insert(0, 'feature', f)
        rows.append(t)
        print(f'  {f}: done')
    return feats, pd.concat(rows, ignore_index=True)


def fmt_p(p):
    return f'< {1 / N_BOOT:g}' if p < 1 / N_BOOT else f'{p:.3f}'


def panel(ax, t, feat_row):
    for i, axis in enumerate(AXIS_NAMES):
        for j, (sch, c) in enumerate(zip(SCHEMES, SCHEME_COLORS)):
            r = t[(t.axis == axis) & (t.scheme == sch)].iloc[0]
            x = i + (j - 1.5) * 0.17
            ax.plot([x, x], [r.ci_low, r.ci_high], color=c, linewidth=3, solid_capstyle='round')
            ax.scatter(x, r.estimate, s=30, color=c, edgecolor='white', linewidth=1.2, zorder=3)
    ax.axhline(0, color=INK_2, linewidth=1)
    lo = t[['ci_low']].min().iloc[0]
    hi = t[['ci_high']].max().iloc[0]
    pad = 0.08 * (hi - lo)
    # hierarchical p above each axis group; bold where the naive test was significant
    for i, axis in enumerate(AXIS_NAMES):
        h = t[(t.axis == axis) & (t.scheme == SCHEMES[-1])].iloc[0]
        n = t[(t.axis == axis) & (t.scheme == SCHEMES[0])].iloc[0]
        naive = axis in feat_row.naive_axes
        ax.text(i, hi + pad, f'naive p {fmt_p(n.p_two_sided)}\nhier. p {fmt_p(h.p_two_sided)}',
                ha='center', va='bottom', fontsize=8.5, color=INK if naive else MUTED,
                weight='bold' if naive else 'normal')
    ax.set_ylim(lo - pad, hi + 4.2 * pad)
    ax.set_xticks(range(len(AXIS_NAMES)),
                  [f'{a}*' if a in feat_row.naive_axes else a for a in AXIS_NAMES])
    for tick, a in zip(ax.get_xticklabels(), AXIS_NAMES):
        if a in feat_row.naive_axes:
            tick.set_fontweight('bold')
            tick.set_color(INK)
    ax.set_xlim(-0.55, len(AXIS_NAMES) - 0.45)
    sr = t[t.scheme == SCHEMES[-1]].se_ratio_vs_naive
    ax.set_title(f'{feat_row.feature}\nomnibus naive p {feat_row.p_r2_global_perm:.4f} · '
                 f'CI ×{sr.min():.1f}–{sr.max():.1f} wider',
                 loc='left', fontsize=10.5, color=INK, pad=6)
    ax.set_ylabel('slope (feature units per mm)', fontsize=9, color=INK_2)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figure(label, feats, table):
    n = len(feats)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    h = 4.3 * n_rows + 2.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, h), facecolor='white', squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=1 - 1.75 / h, bottom=1.2 / h,
                        hspace=0.5, wspace=0.3)
    axes = axes.ravel()
    for ax, (_, fr) in zip(axes, feats.iterrows()):
        panel(ax, table[table.feature == fr.feature], fr)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.text(0.06, 1 - 0.3 / h, f'Bootstrap 95% CI of each gradient slope, by resampling unit '
             f'— {label}', fontsize=16, weight='bold', color=INK, va='top')
    fig.text(0.06, 1 - 0.75 / h,
             'Every waveform feature with p < 0.05 under the naive global shuffle (omnibus, or '
             'on any one axis). Bars: 95% CI, dot: fitted slope. A bar that crosses 0 means '
             'the sign of the slope is not established.\n'
             'Bold axis (*) = significant under the naive shuffle. Annotated p values are '
             'bootstrap two-sided p for the naive (neurons) and hierarchical schemes.',
             fontsize=10, color=INK_2, va='top')
    fig.legend(handles=[Line2D([], [], color=c, linewidth=3, label=nm)
                        for c, nm in zip(SCHEME_COLORS, SCHEME_NAMES)],
               title='resampled unit', title_fontsize=9.5, loc='lower left',
               bbox_to_anchor=(0.055, 0.035), ncol=4, frameon=False, fontsize=10,
               labelcolor=INK_2)
    fig.text(0.06, 0.012, f'{N_BOOT} resamples per scheme, seed {SEED}; slopes are partial '
             'effects from value ~ |ML| + AP + DV (mm from bregma). '
             'Source: figure_preparation/bootstrap_schemes_all_features.py',
             fontsize=8.5, color=MUTED, ha='left', va='bottom')
    return fig


if __name__ == '__main__':
    label = sys.argv[1] if len(sys.argv) > 1 else 'waveform_low_DRN'
    feats, table = run(label)
    table.insert(0, 'dataset', label)
    table.to_csv(os.path.join(OUT_DIR, f'bootstrap_scheme_comparison_all_features_{label}.csv'),
                 index=False)
    fig = make_figure(label, feats, table)
    path = os.path.join(OUT_DIR, f'bootstrap_schemes_all_features_{label}')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}', dpi=200)
    print('wrote', path + '.png')
