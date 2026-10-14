"""Per-feature p-values under progressively stricter nulls, one line per feature.

mode `lr`: for each linear test (omnibus 3D R², ML partial, DV partial) the same
feature is followed from the global shuffle used in the figures, to the
session-restricted shuffle, to the within-mouse shuffle.
mode `knn`: the figure's kNN CV-R² statistic under the same three nulls, plus a
second panel that keeps the session-restricted null and swaps standard 5-fold CV
for GroupKFold by session (no unit is predicted from its own insertion).

Reads session_bias_stats_<dataset>.csv written by spatial_session_bias_report.py.

Run:  python session_combine/figure_preparation/pvalue_ladder.py [lr|knn] [dataset]
      defaults: lr, waveform_low_DRN
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

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']),
                       'SUMMARY_DRN_spatial_organization', '06_session_bias_controls')

NULL_NAMES = ['global shuffle\n(as in figures)', 'insertions kept\ntogether',
              'insertions shuffled\nonly within mouse']


def null_steps(test):
    return [f'{test}_{n}' for n in ('global_perm', 'session_perm', 'within_mouse_perm')]


# Each panel: (title, column stems in step order, x tick names). The p column is
# `p_<stem>` and its FDR, where computed, is `fdr_<stem>`.
PANELS = {
    'lr': [
        ('Omnibus 3D gradient\n(the "LR p" on the figure panels)', null_steps('r2'), NULL_NAMES),
        ('ML component only', null_steps('ML'), NULL_NAMES),
        ('DV component only', null_steps('DV'), NULL_NAMES),
    ],
    'knn': [
        ('kNN prediction from 3D position\n(the "KNN p" on the figure panels)',
         null_steps('knn'), NULL_NAMES),
        ('Same null (insertions kept together),\nbut a unit\'s own insertion is hidden in training',
         ['knn_session_perm', 'knn_groupkfold_session_perm'],
         ['standard 5-fold CV\n(insertion-mates can\nbe in training set)',
          'whole insertion\nheld out']),
    ],
}

# Light-mode tokens from the dataviz reference palette.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8985'
GRID = '#e4e3df'
HOLDS = '#2a78d6'
LOST_MOUSE = '#1baf7a'
LOST = '#eb6834'
P_FLOOR = 3e-4
USED = set()  # (colour, linestyle) outcomes drawn, so the legend lists only those


# Exact reparameterisations (A1 = -P1, A3 = -P3, peak3 ~ tau3) give identical rows;
# drawing both just stacks one line on another.
DUPLICATES = {'basic_ephys_all_DRN': ['A1', 'A3', 'peak3']}
MAX_GREY_LABELS = 10  # above this many features, never-significant lines go unlabelled


def outcome(p):
    """(colour, linestyle) for one feature's p-values across steps.

    Dashed = significant again at a stricter step after failing an earlier one,
    which the README treats as unstable rather than as a finding."""
    sig = p < 0.05
    regains = any(not sig[i] and sig[j] for i in range(len(p)) for j in range(i + 1, len(p)))
    ls = '--' if regains else '-'
    if sig.all():
        return HOLDS, ls
    if not sig[0]:
        return MUTED, ls
    if len(p) > 2 and sig[1]:
        return LOST_MOUSE, ls
    return LOST, ls


def spread_labels(y, min_gap):
    """Nudge sorted label positions apart by at least min_gap, keeping their mean."""
    order = np.argsort(y)
    ys = y[order].astype(float)
    for _ in range(200):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_gap:
                shift = (min_gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= shift
                ys[i] += shift
                moved = True
        if not moved:
            break
    out = np.empty_like(ys)
    out[order] = ys
    return out


def panel(ax, s, title, stems, names):
    x = np.arange(len(stems))
    ly = []
    for _, r in s.iterrows():
        p = np.array([r[f'p_{st}'] for st in stems], float)
        fdr = [r.get(f'fdr_{st}', np.nan) for st in stems]
        c, ls = outcome(p)
        USED.add((c, ls))
        lp = -np.log10(np.clip(p, P_FLOOR, 1))
        ax.plot(x, lp, color=c, linewidth=2, linestyle=ls,
                zorder=3 if c != MUTED else 2, solid_capstyle='round')
        for xi, yi, f in zip(x, lp, fdr):
            ax.scatter(xi, yi, s=55, zorder=4, linewidth=2, edgecolor=c,
                       facecolor=c if f < 0.05 else 'white')
        if c != MUTED or ls == '--' or len(s) <= MAX_GREY_LABELS:
            ly.append((lp[-1], r['feature'], c))
    if not ly:
        return
    pos = spread_labels(np.array([v for v, _, _ in ly]), 0.13)
    for (yi, name, c), yl in zip(ly, pos):
        ax.plot([x[-1] + 0.05, x[-1] + 0.17], [yi, yl], color=GRID, linewidth=1)
        ax.text(x[-1] + 0.2, yl, name, va='center', fontsize=9.5,
                color=INK if c != MUTED else INK_2)
    ax.axhline(-np.log10(0.05), color=INK_2, linewidth=1, linestyle='--', zorder=1)
    ticks = [1, 0.3, 0.1, 0.05, 0.01, 0.003, 0.001, 0.0003]
    ax.set_yticks(-np.log10(ticks), [f'{t:g}' for t in ticks])
    ax.set_ylim(-0.08, -np.log10(P_FLOOR) + 0.05)
    labels = ax.get_yticklabels()
    if labels:
        labels[ticks.index(0.05)].set_fontweight('bold')
    ax.set_xticks(x, names)
    ax.set_xlim(-0.3, len(stems) - 1 + 1.05)
    ax.set_title(title, loc='left', fontsize=12.5, color=INK, pad=10)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figure(mode, label):
    s = pd.read_csv(os.path.join(OUT_DIR, f'session_bias_stats_{label}.csv'))
    if 'skipped' in s:
        s = s[s['skipped'].isna() | (s['skipped'] == False)]
    s = s[~s['feature'].isin(DUPLICATES.get(label, []))]
    r0 = s.max(numeric_only=True)
    panels = PANELS[mode]
    widths = [len(stems) + 0.35 for _, stems, _ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(17 if mode == 'lr' else 15, 8.6),
                             sharey=True, facecolor='white',
                             gridspec_kw=dict(width_ratios=widths))
    fig.subplots_adjust(left=0.06, right=0.95 if mode == 'lr' else 0.93, top=0.79,
                        bottom=0.24, wspace=0.5 if mode == 'lr' else 0.38)
    for ax, (title, stems, names) in zip(axes, panels):
        panel(ax, s, title, stems, names)
    axes[0].set_ylabel('permutation p  (more significant ↑)', color=INK_2, fontsize=10.5)

    what = 'linear-gradient' if mode == 'lr' else 'kNN'
    fig.text(0.06, 0.95, f'How each feature\'s {what} p-value changes as the test respects '
             f'insertions — {label}', fontsize=17, color=INK, weight='bold', va='top')
    fig.text(0.06, 0.9, f'{int(r0.n_units)} units, {int(r0.n_sessions)} insertions, '
             f'{int(r0.n_mice)} mice.  Left to right, each step keeps more of the '
             'recording structure intact, so a gradient has to beat a more realistic "chance".',
             fontsize=11, color=INK_2, va='top')
    outcomes = [
        (HOLDS, '-', 'p < 0.05 at every step'),
        (LOST, '-', 'significant at the first step, lost at the second'),
        (LOST_MOUSE, '-', 'holds at the second step, lost only within mouse'),
        (MUTED, '-', 'not significant at the first step'),
    ]
    used_colours = {c for c, _ in USED}
    handles = [Line2D([], [], color=c, linewidth=2, label=t)
               for c, _, t in outcomes if c in used_colours]
    if any(ls == '--' for _, ls in USED):
        handles.append(Line2D([], [], color=INK_2, linewidth=2, linestyle='--',
                              label='dashed = significant again at a stricter step (unstable)'))
    fig.legend(handles=[
        *handles,
        Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor=INK_2,
               markeredgecolor=INK_2, label='filled = also survives FDR at that step'),
        Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor='white',
               markeredgecolor=INK_2, markeredgewidth=2, label='hollow = does not (or FDR not computed)'),
        Line2D([], [], color=INK_2, linewidth=1, linestyle='--', label='p = 0.05'),
    ], loc='lower left', bbox_to_anchor=(0.055, 0.065), ncol=3, frameon=False, fontsize=10,
        labelcolor=INK_2)
    fig.text(0.06, 0.015,
             'FDR (BH) is across features within each column; not computed for the within-mouse '
             'null, which also drops single-session mice.'
             + (f' Duplicates {", ".join(DUPLICATES[label])} omitted (FDR still counts them).'
                if label in DUPLICATES else '')
             + (' Grey lines unlabelled.' if len(s) > MAX_GREY_LABELS else '')
             + '\n'
             + ('ML / DV p are partial effects from the joint ML+AP+DV fit. ' if mode == 'lr' else
                'kNN: distance-weighted, the figure\'s k and seed; statistic = CV R². ')
             + f'p floored at {P_FLOOR:g} for display. '
             'Source: session_bias_stats_*.csv; figure_preparation/pvalue_ladder.py',
             fontsize=8.5, color=MUTED, ha='left', va='bottom')
    return fig


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'lr'
    label = sys.argv[2] if len(sys.argv) > 2 else 'waveform_low_DRN'
    fig = make_figure(mode, label)
    name = f'pvalue_ladder_{label}' if mode == 'lr' else f'pvalue_ladder_{mode}_{label}'
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=200)
    print('wrote', os.path.join(OUT_DIR, f'{name}.png'))
