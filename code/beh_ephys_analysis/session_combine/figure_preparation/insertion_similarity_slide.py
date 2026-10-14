"""One-slide summary of same-insertion similarity in the DRN spatial-gradient data.

Three pieces of evidence that units from the same probe insertion are not
independent samples of anatomical space:
  a. kNN prediction from position, standard 5-fold CV vs whole insertions held out
     (read from session_bias_stats_basic_ephys_all_DRN.csv)
  b. ICC of each feature by session, on ranks, before and after removing a linear
     ML/AP/DV trend, with session-label permutation p (computed here)
  c. bootstrap CI of a gradient slope under progressively more hierarchical
     resampling (read from bootstrap_scheme_comparison_waveform_low_DRN.csv)

Run spatial_session_bias_report.py first; this reads its outputs.
Run:  python session_combine/figure_preparation/insertion_similarity_slide.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.capsule_migration import capsule_directories

BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
N_PERM = 2000
SEED = 42

capsule_dirs = capsule_directories()
SUMMARY_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']),
                           'SUMMARY_DRN_spatial_organization')
SRC_DIR = os.path.join(SUMMARY_DIR, '05_stats_and_source_data')
OUT_DIR = os.path.join(SUMMARY_DIR, '06_session_bias_controls')

# Exact reparameterisations (A1 = -P1, A3 = -P3, peak3 ~ tau3) are dropped so each
# feature appears once.
BASIC_EPHYS_DISTINCT = [
    'response_rate', 'bl_mean', 'bl_response_corr', 'bl_response_corr_short',
    'pc_1', 'pc_2', 'pc_3', 'P1', 'P2', 'P3', 'tau1', 'tau2', 'tau3',
    'response_fr', 'peak2', 'A2',
]
ICC_FEATURES = {
    'waveform_low_DRN': ['trough_post_ratio_1D', 'post_trough_slope', 'wf_pc_2',
                         'wf_pc_3', 'post_w'],
    'basic_ephys_all_DRN': ['P1', 'P2', 'tau1', 'bl_mean', 'response_rate',
                            'bl_response_corr'],
}

# Light-mode tokens from the dataviz reference palette. Gray = naive analysis,
# blue = insertion-aware; the blue ramp is ordinal (more hierarchy = darker).
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8985'
GRID = '#e4e3df'
BLUE = '#2a78d6'
BLUE_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']


# ---------------------------------------------------------------------------
# ICC by session
# ---------------------------------------------------------------------------
def icc1(y, g):
    """One-way random-effects ICC(1) for unbalanced groups (ANOVA estimator)."""
    _, inv, k = np.unique(g, return_inverse=True, return_counts=True)
    n, a = len(y), len(k)
    means = np.bincount(inv, weights=y) / k
    msb = (k * (means - y.mean()) ** 2).sum() / (a - 1)
    msw = ((y - means[inv]) ** 2).sum() / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def icc_perm(y, g, rng):
    obs = icc1(y, g)
    null = np.array([icc1(y, rng.permutation(g)) for _ in range(N_PERM)])
    return obs, (1 + (null >= obs).sum()) / (N_PERM + 1)


def icc_table():
    rng = np.random.default_rng(SEED)
    tables = {
        'waveform_low_DRN': pd.read_csv(
            os.path.join(SRC_DIR, 'combined_features_waveform_low_DRN.csv')),
        'basic_ephys_all_DRN': pd.read_pickle(
            os.path.join(SRC_DIR, 'features_combined_basic_ephys_all_DRN.pkl')),
    }
    rows = []
    for label, df in tables.items():
        for f in ICC_FEATURES[label]:
            s = df[[f, 'session'] + CCF_COLS].dropna()
            X = s[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
            X[:, 0] = np.abs(X[:, 0])
            y = rankdata(s[f].to_numpy(float))
            g = s['session'].to_numpy()
            A = np.column_stack([np.ones(len(X)), X])
            res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            icc_raw, p_raw = icc_perm(y, g, rng)
            icc_res, p_res = icc_perm(res, g, rng)
            rows.append(dict(dataset=label, feature=f, n_units=len(s),
                             n_sessions=s['session'].nunique(),
                             icc_raw=icc_raw, p_raw=p_raw,
                             icc_after_position=icc_res, p_after_position=p_res))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def style(ax):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dumbbell(ax, labels, before, after, after_sig, before_label, after_label, y=None):
    if y is None:
        y = np.arange(len(labels))[::-1]
    for yi, b, a in zip(y, before, after):
        ax.plot([b, a], [yi, yi], color=GRID, linewidth=2, zorder=1, solid_capstyle='round')
    ax.scatter(before, y, s=60, facecolor='white', edgecolor=MUTED, linewidth=2, zorder=2)
    ax.scatter(after, y, s=60, facecolor=np.where(after_sig, BLUE, 'white'),
               edgecolor=BLUE, linewidth=2, zorder=3)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=INK_2, linewidth=1, zorder=0)
    ax.legend(handles=[
        Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor='white',
               markeredgecolor=MUTED, markeredgewidth=2, label=before_label),
        Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor=BLUE,
               markeredgecolor=BLUE, markeredgewidth=2, label=after_label + ', p < 0.05'),
        Line2D([], [], marker='o', linestyle='', markersize=8, markerfacecolor='white',
               markeredgecolor=BLUE, markeredgewidth=2, label=after_label + ', n.s.'),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=1, frameon=False,
        fontsize=9.5, labelcolor=INK_2)
    return y


def panel_groupkfold(ax):
    s = pd.read_csv(os.path.join(OUT_DIR, 'session_bias_stats_basic_ephys_all_DRN.csv'))
    s = s.set_index('feature').loc[BASIC_EPHYS_DISTINCT]
    s = s.sort_values('r2_knn_kfold', ascending=False)
    dumbbell(ax, s.index, s['r2_knn_kfold'], s['r2_knn_groupkfold'],
             s['p_knn_groupkfold_session_perm'].to_numpy() < 0.05,
             'standard 5-fold CV (insertion-mates can be in training set)',
             'whole insertion held out')
    ax.set_xlabel('kNN prediction R² from 3D position', color=INK_2, fontsize=10.5)
    ax.text(-0.01, 0.995, '← worse than guessing the mean', ha='right', va='top',
            fontsize=9, color=INK_2, transform=ax.get_xaxis_transform())
    ax.set_title('a. Firing / ACG features: predictability from position\n'
                 'mostly disappears when insertion-mates are hidden',
                 loc='left', fontsize=12.5, color=INK, pad=10)
    style(ax)


def panel_icc(ax, icc):
    icc = icc.copy()
    icc['label'] = icc['feature']
    gap = []
    labels, before, after, sig = [], [], [], []
    for label, title in [('waveform_low_DRN', 'waveform · 37 insertions'),
                         ('basic_ephys_all_DRN', 'firing / ACG · 33 insertions')]:
        d = icc[icc.dataset == label]
        labels += list(d['feature']); before += list(d['icc_raw'])
        after += list(d['icc_after_position']); sig += list(d['p_after_position'] < 0.05)
        gap.append((len(labels), title))
    n_wf = gap[0][0]
    # one empty row between the two datasets for the group heading
    y = np.array([-i if i < n_wf else -i - 1.2 for i in range(len(labels))], float)
    dumbbell(ax, labels, before, after, np.array(sig),
             'ICC by insertion, raw', 'ICC left after removing linear position trend', y=y)
    for (end, title), start in zip(gap, [0, n_wf]):
        ax.text(0.0, y[start] + 0.85, title, ha='right', va='center', fontsize=10,
                color=INK, weight='bold', transform=ax.get_yaxis_transform())
    ax.set_ylim(y.min() - 0.6, y.max() + 1.2)
    ax.set_xlabel('share of feature variance that is "which insertion" (ICC, ranks)',
                  color=INK_2, fontsize=10.5)
    ax.set_xlim(-0.08, 0.32)
    ax.set_title('b. Units from one insertion resemble each other,\n'
                 'partly beyond what their position explains',
                 loc='left', fontsize=12.5, color=INK, pad=10)
    style(ax)


def panel_bootstrap(ax):
    b = pd.read_csv(os.path.join(OUT_DIR, 'bootstrap_scheme_comparison_waveform_low_DRN.csv'))
    b = b[b.axis.isin(['ML', 'DV'])]
    schemes = ['unit_only (naive)', 'cluster_session', 'cluster_mouse',
               'hierarchical_mouse_session_unit']
    names = ['neurons (naive)', 'insertions', 'mice', 'mice → insertions → neurons']
    colors = [MUTED] + BLUE_RAMP[1:]
    for xi, axis in enumerate(['ML', 'DV']):
        for j, (sch, c) in enumerate(zip(schemes, colors)):
            r = b[(b.axis == axis) & (b.scheme == sch)].iloc[0]
            x = xi + (j - 1.5) * 0.2
            ax.plot([x, x], [r.ci_low, r.ci_high], color=c, linewidth=3,
                    solid_capstyle='round')
            ax.scatter(x, r.estimate, s=36, color=c, edgecolor='white', linewidth=1.5, zorder=3)
            ax.text(x, r.ci_low - 0.03, f'×{r.se_ratio_vs_naive:.1f}', ha='center',
                    va='top', fontsize=8.5, color=INK_2)
        r = b[(b.axis == axis) & (b.scheme == schemes[-1])].iloc[0]
        ax.text(xi, 0.1, f'hierarchical\np = {r.p_two_sided:.3f}', ha='center',
                va='center', fontsize=9.5, color=INK)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks([0, 1], ['ML slope', 'DV slope'])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.95, 0.2)
    ax.set_ylabel('trough_post_ratio_1D slope (per mm), 95% CI', color=INK_2, fontsize=10.5)
    ax.legend(handles=[Line2D([], [], color=c, linewidth=3, label=n)
                       for c, n in zip(colors, names)],
              title='resampling unit (×N = CI width vs naive)', title_fontsize=9.5,
              loc='upper left', bbox_to_anchor=(-0.05, -0.09), ncol=1, frameon=False,
              fontsize=9.5, labelcolor=INK_2)
    ax.set_title('c. Respecting the grouping roughly\n'
                 'doubles the uncertainty of a gradient',
                 loc='left', fontsize=12.5, color=INK, pad=10)
    style(ax)
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(MUTED)


def make_slide(icc):
    lev = pd.read_csv(os.path.join(OUT_DIR, 'coordinate_leverage.csv'))
    fig = plt.figure(figsize=(17, 9.6), facecolor='white')
    fig.text(0.04, 0.955, 'Units from the same probe insertion are not independent samples',
             fontsize=21, color=INK, weight='bold', va='top')
    fig.text(0.04, 0.905,
             'The original gradient tests treat every neuron as independent. Neurons arrive '
             'in insertions, and neurons from one insertion share position, recording '
             'conditions and animal.',
             fontsize=12, color=INK_2, va='top')

    stats = [
        ('0.95', 'ML ICC: within one insertion ML barely varies\n'
                 '(median spread 9 µm, vs 163 µm in DV)'),
        (f"≈ {lev.query('dataset == \"waveform_low_DRN\" and axis == \"ML\"').n_eff_units.iloc[0]:.0f}",
         'effective units for an ML gradient in the\n'
         'waveform set, not 188 (37 insertions)'),
        ('24 / 108', 'firing/ACG units from a single insertion\n'
                     '(median insertion contributes 2)'),
    ]
    for i, (big, small) in enumerate(stats):
        x = 0.04 + i * 0.31
        fig.text(x, 0.845, big, fontsize=26, color=BLUE, weight='bold', va='top')
        fig.text(x + 0.012 + 0.0115 * len(big), 0.843, small, fontsize=10.5,
                 color=INK_2, va='top')

    gs = fig.add_gridspec(1, 3, left=0.135, right=0.95, top=0.68, bottom=0.25,
                          wspace=0.62, width_ratios=[1, 1, 0.75])
    panel_groupkfold(fig.add_subplot(gs[0]))
    panel_icc(fig.add_subplot(gs[1]), icc)
    panel_bootstrap(fig.add_subplot(gs[2]))

    fig.text(0.04, 0.025,
             'a: distance-weighted kNN (k = 10) on 3D CCF position, basic_ephys_all_DRN; '
             'filled = held-out R² beats the session-restricted permutation null. '
             'Duplicates A1, A3, peak3 omitted.\n'
             'b: one-way ICC(1) on rank-transformed features; p from 2000 shuffles of session '
             'labels; position removed by linear regression on |ML|, AP, DV.\n'
             'c: same slope under four bootstrap schemes (1000 resamples each). '
             'Source: 06_session_bias_controls/*.csv; '
             'figure_preparation/insertion_similarity_slide.py.',
             fontsize=8.5, color=MUTED, va='bottom')
    return fig


if __name__ == '__main__':
    icc = icc_table()
    icc.to_csv(os.path.join(OUT_DIR, 'insertion_similarity_icc.csv'), index=False)
    print(icc.round(3).to_string(index=False))
    fig = make_slide(icc)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'insertion_similarity_slide.{ext}'), dpi=200)
    print('wrote', os.path.join(OUT_DIR, 'insertion_similarity_slide.png'))
