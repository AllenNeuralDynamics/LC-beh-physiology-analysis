"""Summary slide: the omnibus 3D gradient tested by bootstrap, under four resampling schemes.

The bootstrap counterpart of the omnibus R² permutation test is a joint test of the
three slopes. For each scheme the gradient value ~ |ML| + AP + DV is refitted on
every resample, giving a cloud of (ML, AP, DV) slope triplets. The p-value is the
fraction of resamples whose Mahalanobis distance from the fitted slopes is at least
as large as the distance of "no gradient" (0, 0, 0) from them, using the resampled
covariance. That is the bootstrap analogue of a 3-df Wald test, without assuming
the distance is chi-squared.

Schemes, from lenient to strict, match bootstrap_scheme_comparison: neurons only;
insertions (then neurons); mice (then neurons); mice -> insertions -> neurons.
Rows match cross_modality_summary_simple.

Run:  python session_combine/figure_preparation/bootstrap_omnibus_summary.py
"""

import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

sys.path.append('/root/capsule/code/beh_ephys_analysis')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.spatial_session_bias import mouse_from_session
from cross_modality_summary import (ALPHA, BAND, DATASETS, GRID, HOLDS, INK, INK_2, LOST,
                                    MUTED, OUT_DIR, SOURCES, build_table, fmt_p, simple_rows)

BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
RANKED = {'basic_ephys_all_DRN'}  # rank-transformed in the stats, as here
N_BOOT = 2000
SEED = 42
# Calibration: datasets with the real clustering but no gradient, made by moving each
# insertion's block of units to another insertion's location (the permutation null).
N_SIM = 100
N_BOOT_SIM = 400
# (column label, levels resampled); 'unit' alone ignores insertions and mice
SCHEMES = [
    ('neurons', ('unit',)),
    ('insertions', ('session', 'unit')),
    ('mice', ('mouse', 'unit')),
    ('mice → ins.\n→ neurons', ('mouse', 'session', 'unit')),
]


def load(label, feature):
    path = os.path.join(os.path.dirname(OUT_DIR), '05_stats_and_source_data', SOURCES[label])
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_pickle(path)
    X = df[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    X[:, 0] = np.abs(X[:, 0])
    y = pd.to_numeric(df[feature], errors='coerce').to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y, s = X[ok], y[ok], df['session'].to_numpy()[ok]
    if label in RANKED:
        y = rankdata(y)
    return X, y, s, mouse_from_session(s)


def slopes(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0][1:]


def boot_slopes(X, y, s, m, levels, rng):
    """Resampled slope triplets; same resampling and identifiability rule as
    utils.spatial_session_bias.hierarchical_bootstrap."""
    if levels == ('unit',):
        groups = None
    else:
        mice = np.unique(m)
        sess_of_mouse = {k: np.unique(s[m == k]) for k in mice}
        rows_of_sess = {k: np.flatnonzero(s == k) for k in np.unique(s)}
    out = []
    for _ in range(N_BOOT):
        if levels == ('unit',):
            rows = rng.integers(0, len(y), len(y))
        else:
            mm = rng.choice(mice, len(mice)) if 'mouse' in levels else mice
            parts = []
            for k in mm:
                ss = sess_of_mouse[k]
                ss = rng.choice(ss, len(ss)) if 'session' in levels else ss
                for j in ss:
                    r = rows_of_sess[j]
                    parts.append(rng.choice(r, len(r)) if 'unit' in levels else r)
            rows = np.concatenate(parts)
        if len(np.unique(s[rows])) < X.shape[1] + 2:
            continue
        out.append(slopes(X[rows], y[rows]))
    return np.vstack(out)


def joint_p(beta, boot):
    """Bootstrap Wald test of beta = 0: share of resamples at least as far from beta
    (in the resampled covariance's metric) as the origin is."""
    cov = np.cov(boot, rowvar=False)
    inv = np.linalg.pinv(cov)
    w0 = float(beta @ inv @ beta)
    d = boot - beta
    wb = np.einsum('ij,jk,ik->i', d, inv, d)
    return (1 + np.sum(wb >= w0)) / (len(wb) + 1), w0


def false_positive_rates(label, feature, rng):
    """Share of simulated no-gradient datasets each scheme calls significant."""
    X, y, s, m = load(label, feature)
    us = np.unique(s)
    idx = {u: np.flatnonzero(s == u) for u in us}
    cent = {u: X[idx[u]].mean(axis=0) for u in us}
    hits = np.zeros(len(SCHEMES))
    global N_BOOT
    n_boot, N_BOOT = N_BOOT, N_BOOT_SIM
    try:
        for _ in range(N_SIM):
            perm = dict(zip(us, rng.permutation(us)))
            Xn = X.copy()
            for u in us:
                Xn[idx[u]] = X[idx[u]] - cent[u] + cent[perm[u]]
            beta = slopes(Xn, y)
            for k, (_, levels) in enumerate(SCHEMES):
                hits[k] += joint_p(beta, boot_slopes(Xn, y, s, m, levels, rng))[0] < ALPHA
    finally:
        N_BOOT = n_boot
    return hits / N_SIM


def compute(rows):
    rng = np.random.default_rng(SEED)
    out = []
    for _, r in rows.iterrows():
        X, y, s, m = load(r.dataset, r.feature)
        beta = slopes(X, y)
        rec = dict(dataset=r.dataset, group=r.group, feature=r.feature, n_units=len(y),
                   n_insertions=len(np.unique(s)), n_mice=len(np.unique(m)),
                   n_mice_multi=r.n_mice_multi, perm_p_session=r.r2_p_session_perm,
                   perm_outcome=r.r2_outcome)
        for k, (_, levels) in enumerate(SCHEMES):
            p, w0 = joint_p(beta, boot_slopes(X, y, s, m, levels, rng))
            rec[f'p_{k}'] = p
            rec[f'wald_{k}'] = w0
        out.append(rec)
        print(f'  {r.feature}: ' + '  '.join(f'{rec[f"p_{k}"]:.3f}' for k in range(len(SCHEMES))))
    t = pd.DataFrame(out)
    last = len(SCHEMES) - 1
    for lab in t.dataset.unique():  # BH-FDR across features within dataset, per scheme
        msk = t.dataset == lab
        for k in range(len(SCHEMES)):
            t.loc[msk, f'fdr_{k}'] = multipletests(t.loc[msk, f'p_{k}'], method='fdr_bh')[1]
    # calibrate on each dataset's first row (the strongest naive effect sorts first)
    cal = {}
    for lab in t.dataset.unique():
        f = t[t.dataset == lab].feature.iloc[0]
        cal[lab] = false_positive_rates(lab, f, rng)
        print(f'  calibration {lab} ({f}): ' + '  '.join(f'{v:.2f}' for v in cal[lab]))
    for k in range(len(SCHEMES)):
        t[f'fpr_{k}'] = t.dataset.map(lambda lab: cal[lab][k])
    t['fpr_feature'] = t.dataset.map(lambda lab: t[t.dataset == lab].feature.iloc[0])
    naive, hier = t.p_0 < ALPHA, t[f'p_{last}'] < ALPHA
    t['outcome'] = np.select([naive & hier, naive & ~hier, ~naive & hier],
                             ['holds', 'lost', 'unstable'], 'never')
    return t


OUTCOME_COLOUR = {'holds': HOLDS, 'lost': LOST, 'never': MUTED, 'unstable': MUTED}


def draw_table(ax, t):
    x_n, x0, cell = 3.05, 3.9, 1.15
    x_perm = x0 + len(SCHEMES) * cell + 0.45
    width = x_perm + cell + 0.1
    y, ys, group_rows = 0, [], []
    for g, d in t.groupby('group', sort=False):
        group_rows.append((y, g, d))
        y += 1.45
        for i in d.index:
            ys.append((i, y))
            y += 1
        y += 0.35
    ax.text(x0 + len(SCHEMES) * cell / 2, -1.55, 'Hierarchical bootstrap: resampled unit',
            ha='center', va='bottom', fontsize=10, color=INK, weight='bold')
    ax.plot([x0 + 0.1, x0 + len(SCHEMES) * cell - 0.1], [-1.45] * 2, color=MUTED, linewidth=0.8)
    for k, (nm, _) in enumerate(SCHEMES):
        ax.text(x0 + (k + 0.5) * cell, -0.15, nm, ha='center', va='bottom', fontsize=8.5,
                color=INK_2)
        if k:
            ax.annotate('', xy=(x0 + k * cell + 0.12, -0.9), xytext=(x0 + k * cell - 0.12, -0.9),
                        arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
    ax.text(x_perm + cell / 2, -1.55, 'for comparison', ha='center', va='bottom', fontsize=9,
            color=INK_2, style='italic')
    ax.text(x_perm + cell / 2, -0.15, 'permutation,\ninsertions\ntogether', ha='center',
            va='bottom', fontsize=8.5, color=INK_2)
    ax.plot([x_perm - 0.22] * 2, [-1.6, y - 0.3], color=GRID, linewidth=0.8)
    ax.text(x_n + 0.35, -0.15, 'n\nunits', ha='center', va='bottom', fontsize=8.5, color=INK_2)
    for yy, g, d in group_rows:
        ax.text(0, yy + 0.5, g, ha='left', va='center', fontsize=10.5, color=INK, weight='bold')
        ax.text(0, yy + 1.0, f'{d.n_insertions.max()} insertions · {d.n_mice.max()} mice '
                f'({d.n_mice_multi.max()} with ≥2 insertions)',
                ha='left', va='center', fontsize=8, color=INK_2)
        ax.text(x0 - 0.1, yy + 0.5, 'false pos.\nrate', ha='right', va='center', fontsize=7.5,
                color=INK_2, style='italic')
        ax.plot([0, width], [yy + 1.4] * 2, color=GRID, linewidth=0.8)
        for k in range(len(SCHEMES)):
            fpr = d[f'fpr_{k}'].iloc[0]
            ax.text(x0 + (k + 0.5) * cell, yy + 0.5, f'{fpr:.0%}', ha='center', va='center',
                    fontsize=8.5, style='italic',
                    color=LOST if abs(fpr - ALPHA) > 0.03 else INK_2)
    for j, (i, yy) in enumerate(ys):
        r = t.loc[i]
        if j % 2:
            ax.fill_between([0, width], yy, yy + 1, color=BAND, linewidth=0, zorder=0)
        ax.text(0.15, yy + 0.5, r.feature, ha='left', va='center', fontsize=9.5, color=INK)
        odd_n = r.n_units != t[t.group == r.group].n_units.max()
        ax.text(x_n + 0.35, yy + 0.5, f'{r.n_units}', ha='center', va='center', fontsize=9,
                color=INK if odd_n else INK_2, weight='bold' if odd_n else 'normal')
        c = OUTCOME_COLOUR[r.outcome]
        for k in range(len(SCHEMES)):
            pv, q = r[f'p_{k}'], r[f'fdr_{k}']
            cx, cy = x0 + k * cell + 0.3, yy + 0.5
            sig = pv < ALPHA
            ax.scatter(cx, cy, s=48, color=c if sig else 'white', edgecolor=c, linewidth=1.5,
                       zorder=3)
            ax.text(cx + 0.14, cy, fmt_p(pv), ha='left', va='center', fontsize=8.5,
                    color=INK if sig else MUTED, weight='bold' if q < ALPHA else 'normal')
        pp = r.perm_p_session
        ax.text(x_perm + 0.3, yy + 0.5, fmt_p(pp), ha='left', va='center', fontsize=8.5,
                color=INK if pp < ALPHA else MUTED)
    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(y, -2.4)
    ax.axis('off')


EXPLAINER = [
    ('What is tested', 'Is there a 3D gradient in any direction? The model value ~ |ML| + AP + '
     'DV has three slopes; "no gradient" means all three are zero. The bootstrap gives the '
     'joint uncertainty of the three slopes, and p is how often a resample sits at least as '
     'far from the fitted slopes as (0, 0, 0) does. This is the bootstrap counterpart of the '
     '"LR p" permutation test.'),
    ('The four schemes', 'Each rebuilds the dataset by drawing with replacement. Neurons: units '
     'are drawn as if independent (too optimistic). Insertions: whole insertions are drawn, '
     'then units within them. Mice: whole mice, then units. Mice → insertions → neurons: all '
     'three levels, matching how the data were collected; the main result.'),
    ('Is each scheme calibrated?', 'Italic % in each dataset heading: how often the scheme '
     'calls a gradient significant in simulated data that keep the real insertion clustering '
     'but contain no gradient. It should be 5%; orange = off by more than 3 points. '
     'Resampling neurons is far too liberal everywhere. With only 14-15 mice the full '
     'hierarchy almost never rejects, so failing there is weak evidence of no effect. The '
     'middle schemes swing between datasets. No scheme is reliably at 5% here.'),
    ('So which test to trust for the omnibus question', 'The permutation with insertions '
     'kept together (right column) holds its 5% rate by construction under this null, so it '
     'is the primary omnibus test. Use the bootstrap for effect sizes and CIs, and read its '
     'p-values as bracketing, not deciding.'),
]


def takeaways(t):
    last = len(SCHEMES) - 1

    def names(mask):
        d = t[mask]
        return ', '.join(f + ('*' if q < ALPHA else '')
                         for f, q in zip(d.feature, d[f'fdr_{last}'])) or 'none'
    hier = t[f'p_{last}'] < ALPHA
    perm = t.perm_outcome.isin(['holds', 'lost_mouse'])
    return [
        ('Significant with all three levels resampled', names(hier & (t.p_0 < ALPHA))),
        ('Significant naively, lost with all levels resampled', names((t.p_0 < ALPHA) & ~hier)),
        ('Significant with insertions resampled and with mice resampled',
         ', '.join(t.loc[(t.p_1 < ALPHA) & (t.p_2 < ALPHA), 'feature']) or 'none'),
        ('Permutation (insertions together) significant, full-hierarchy bootstrap not',
         ', '.join(t.loc[perm & ~hier, 'feature']) or 'none'),
    ]


def make_figure(t):
    h = 12.2
    fig = plt.figure(figsize=(18, h), facecolor='white')
    ax = fig.add_axes([0.03, 1.4 / h, 0.56, 1 - 3.0 / h])
    draw_table(ax, t)
    fig.text(0.03, 1 - 0.3 / h, 'Omnibus 3D gradient under hierarchical bootstrapping',
             fontsize=19, weight='bold', color=INK, va='top')
    fig.text(0.03, 1 - 0.78 / h,
             'Joint test of the three slopes of value ~ |ML| + AP + DV, with the resampling '
             'made progressively more faithful to how the data were collected.\nFilled dot = '
             'p < .05 under that scheme; colour = outcome from neurons to the full hierarchy; '
             'bold p = also passes FDR.', fontsize=10.5, color=INK_2, va='top')
    x, y = 0.625, 1 - 1.55 / h
    line = 0.21 / h
    for head, body in EXPLAINER:
        fig.text(x, y, head, fontsize=11.5, color=INK, weight='bold', va='top')
        y -= 1.25 * line
        body = textwrap.fill(body, 74)
        fig.text(x, y, body, fontsize=9.3, color=INK_2, va='top')
        y -= (body.count('\n') + 1) * line + 0.8 * line
    y -= 0.5 * line
    fig.text(x, y, 'Takeaways   (* = also passes FDR)', fontsize=13, color=INK, weight='bold',
             va='top')
    y -= 1.8 * line
    for head, body in takeaways(t):
        head, body = textwrap.fill(head, 72), textwrap.fill(body, 60)
        fig.text(x, y, head, fontsize=9.5, color=INK_2, va='top')
        y -= (head.count('\n') + 1) * line
        fig.text(x, y, body, fontsize=10.5, color=INK, weight='bold', va='top')
        y -= (body.count('\n') + 1) * line * 1.1 + 0.9 * line
    fig.legend(handles=[
        Line2D([], [], marker='o', linestyle='', markersize=9, color=HOLDS,
               label='significant naively and with all three levels resampled'),
        Line2D([], [], marker='o', linestyle='', markersize=9, color=LOST,
               label='significant naively, lost with all levels resampled'),
        Line2D([], [], marker='o', linestyle='', markersize=9, color=MUTED,
               label='not significant naively'),
        Line2D([], [], marker='o', linestyle='', markersize=9, markerfacecolor='white',
               markeredgecolor=INK_2, markeredgewidth=1.8, label='hollow = p ≥ .05 under that scheme'),
    ], loc='lower left', bbox_to_anchor=(0.025, 0.045), ncol=2, frameon=False, fontsize=9.5,
        labelcolor=INK_2)
    fig.text(0.03, 0.008,
             f'{N_BOOT} resamples per scheme, seed {SEED}; calibration: {N_SIM} simulated '
             f'no-gradient datasets x {N_BOOT_SIM} resamples, on each dataset\'s first row. '
             'Resamples with fewer than 5 distinct '
             'insertions are skipped, as in the per-axis bootstrap. Firing/ACG features '
             'rank-transformed as in the stats.\nFDR (BH) across features within dataset and '
             'scheme. Rows as in cross_modality_summary_simple. Source: '
             'figure_preparation/bootstrap_omnibus_summary.py',
             fontsize=8, color=MUTED, ha='left', va='bottom')
    return fig


def main():
    rows = simple_rows(build_table())
    t = compute(rows)
    path = os.path.join(OUT_DIR, 'bootstrap_omnibus_summary')
    t.to_csv(path + '.csv', index=False)
    fig = make_figure(t)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}', dpi=200)
    print('wrote', path + '.png')


if __name__ == '__main__':
    main()
