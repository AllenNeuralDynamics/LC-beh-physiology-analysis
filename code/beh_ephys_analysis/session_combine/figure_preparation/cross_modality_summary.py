"""Conclusion slide: which spatial effects survive the insertion-aware controls, all datasets.

Rows are features significant on at least one naive (global-shuffle) test, grouped
by dataset. Columns are the linear tests (omnibus, ML, AP, DV) and the kNN test,
each followed across global -> insertions-kept-together -> within-mouse nulls,
plus kNN with the insertion held out and the within-insertion DV slope. Each cell
shows a dot coloured by outcome (same categories as pvalue_ladder.py; filled =
also survives FDR under the insertions-kept-together null) and that null's p.

`simple` mode keeps only the two axis-agnostic tests behind the figure panels
(omnibus 3D linear, kNN), with rows limited to features significant naively on
one of them.

Reads session_bias_stats_<dataset>.csv and within_insertion_dv.csv; run
spatial_session_bias_report.py and within_insertion_dv.py first. Writes the
figure and cross_modality_summary.csv (the table behind it).

Run:  python session_combine/figure_preparation/cross_modality_summary.py [full|simple]
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import textwrap

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.capsule_migration import capsule_directories
from utils.spatial_session_bias import mouse_from_session

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']),
                       'SUMMARY_DRN_spatial_organization', '06_session_bias_controls')

DATASETS = [('waveform_low_DRN', 'Waveform shape'),
            ('basic_ephys_all_DRN', 'Firing / ACG'),
            ('beh_all_DRN', 'Task coding')]
DUPLICATES = {'basic_ephys_all_DRN': ['A1', 'A3', 'peak3']}
NULLS = ['global_perm', 'session_perm', 'within_mouse_perm']
# (column header, stat stem, kind) - kind 'ladder' uses the three nulls
COLUMNS = [
    ('Omnibus\n3D', 'r2', 'ladder'),
    ('ML', 'ML', 'ladder'),
    ('AP', 'AP', 'ladder'),
    ('DV', 'DV', 'ladder'),
    ('kNN', 'knn', 'ladder'),
    ('kNN,\ninsertion\nheld out', 'knn_groupkfold_session_perm', 'single'),
    ('DV within\ninsertion\n(along track)', 'within', 'within'),
]
SIMPLE_COLUMNS = [('Omnibus 3D\n(linear)', 'r2', 'ladder'), ('kNN', 'knn', 'ladder')]
ALPHA = 0.05

# Light-mode tokens; outcome colours match pvalue_ladder.py.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8985'
GRID = '#e4e3df'
BAND = '#f4f3f0'
HOLDS = '#2a78d6'
LOST_MOUSE = '#1baf7a'
LOST = '#eb6834'
OUTCOME_ORDER = ['holds', 'lost_mouse', 'lost', 'never', 'unstable']
OUTCOME_COLOUR = {'holds': HOLDS, 'lost_mouse': LOST_MOUSE, 'lost': LOST,
                  'never': MUTED, 'unstable': MUTED}


SOURCES = {
    'waveform_low_DRN': 'combined_features_waveform_low_DRN.csv',
    'basic_ephys_all_DRN': 'features_combined_basic_ephys_all_DRN.pkl',
    'beh_all_DRN': 'features_combined_beh_all_DRN.pkl',
}
_SOURCE_CACHE = {}


def mice_with_multiple_insertions(label, feature):
    """Mice with >= 2 insertions among units with a finite value: the only mice whose
    insertions can trade places under the within-mouse null."""
    if label not in _SOURCE_CACHE:
        path = os.path.join(os.path.dirname(OUT_DIR), '05_stats_and_source_data', SOURCES[label])
        _SOURCE_CACHE[label] = pd.read_csv(path) if path.endswith('.csv') else pd.read_pickle(path)
    df = _SOURCE_CACHE[label]
    v = pd.to_numeric(df[feature], errors='coerce')
    ok = np.isfinite(v) & np.isfinite(df[['x_ccf', 'y_ccf', 'z_ccf']].to_numpy(float)).all(axis=1)
    d = df.loc[ok, ['session']].copy()
    d['mouse'] = mouse_from_session(d['session'].to_numpy())
    return int((d.groupby('mouse').session.nunique() >= 2).sum())


def ladder_outcome(p):
    sig = np.asarray(p) < ALPHA
    if any(not sig[i] and sig[j] for i in range(3) for j in range(i + 1, 3)):
        return 'unstable'
    if sig.all():
        return 'holds'
    if not sig[0]:
        return 'never'
    return 'lost_mouse' if sig[1] else 'lost'


def build_table():
    within = pd.read_csv(os.path.join(OUT_DIR, 'within_insertion_dv.csv'))
    rows = []
    for label, name in DATASETS:
        s = pd.read_csv(os.path.join(OUT_DIR, f'session_bias_stats_{label}.csv'))
        s = s[~s['feature'].isin(DUPLICATES.get(label, []))]
        naive = s[['p_r2_global_perm', 'p_ML_global_perm', 'p_AP_global_perm',
                   'p_DV_global_perm', 'p_knn_global_perm']].lt(ALPHA).any(axis=1)
        for _, r in s[naive].iterrows():
            row = dict(dataset=label, group=name, feature=r['feature'],
                       n_units=int(r['n_units']), n_insertions=int(r['n_sessions']),
                       n_mice=int(r['n_mice']),
                       n_mice_multi=mice_with_multiple_insertions(label, r['feature']))
            for _, stem, kind in COLUMNS:
                if kind == 'ladder':
                    p = [r[f'p_{stem}_{n}'] for n in NULLS]
                    row[f'{stem}_outcome'] = ladder_outcome(p)
                    row[f'{stem}_p'] = r[f'p_{stem}_session_perm']
                    row[f'{stem}_fdr'] = r[f'fdr_{stem}_session_perm']
                    for n in NULLS:
                        row[f'{stem}_p_{n}'] = r[f'p_{stem}_{n}']
                        row[f'{stem}_fdr_{n}'] = r.get(f'fdr_{stem}_{n}', np.nan)
                elif kind == 'single':
                    p = r[f'p_{stem}']
                    row[f'{stem}_outcome'] = 'holds' if p < ALPHA else 'never'
                    row[f'{stem}_p'] = p
                    row[f'{stem}_fdr'] = r[f'fdr_{stem}']
                else:
                    w = within[(within.dataset == label) & (within.feature == r['feature'])]
                    if len(w):
                        p = w.p_hier.iloc[0]
                        row['within_outcome'] = 'holds' if p < ALPHA else 'never'
                        row['within_p'] = p
                    else:
                        row['within_outcome'] = 'not_tested'
                        row['within_p'] = np.nan
                    row['within_fdr'] = np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    # BH-FDR for the within-insertion column, across tested features within dataset,
    # like every other column
    for lab in t.dataset.unique():
        m = (t.dataset == lab) & t.within_p.notna()
        if m.any():
            t.loc[m, 'within_fdr'] = multipletests(t.loc[m, 'within_p'], method='fdr_bh')[1]
    order = {lab: i for i, (lab, _) in enumerate(DATASETS)}
    t['_d'] = t.dataset.map(order)
    return t.sort_values(['_d', 'r2_p']).drop(columns='_d').reset_index(drop=True)


def fmt_p(p):
    if not np.isfinite(p):
        return ''
    return '<.001' if p < 0.001 else f'{p:.3f}'.lstrip('0')


def draw_table(ax, t, columns=COLUMNS, col_w=1.0):
    y = 0
    ys, group_rows = [], []
    for g, d in t.groupby('group', sort=False):
        group_rows.append((y, g))
        y += 1
        for i in d.index:
            ys.append((i, y))
            y += 1
        y += 0.35
    n_y = y
    x0 = 2.9  # feature-name column width, in cell units
    x_end = x0 + col_w * len(columns)
    for k, (head, _, _) in enumerate(columns):
        ax.text(x0 + col_w * (k + 0.5), -0.6, head, ha='center', va='bottom', fontsize=9.5,
                color=INK)
    for yy, g in group_rows:
        ax.text(0, yy + 0.5, g, ha='left', va='center', fontsize=10.5, color=INK, weight='bold')
        ax.plot([0, x_end], [yy + 0.95] * 2, color=GRID, linewidth=0.8)
    for j, (i, yy) in enumerate(ys):
        r = t.loc[i]
        if j % 2:
            ax.fill_between([0, x_end], yy, yy + 1, color=BAND, linewidth=0, zorder=0)
        ax.text(0.15, yy + 0.5, r.feature, ha='left', va='center', fontsize=9.5, color=INK)
        for k, (_, stem, _) in enumerate(columns):
            out = r[f'{stem}_outcome']
            cx, cy = x0 + col_w * (k + 0.5), yy + 0.5
            if out == 'not_tested':
                ax.text(cx, cy, '—', ha='center', va='center', fontsize=9, color=GRID)
                continue
            c = OUTCOME_COLOUR[out]
            filled = r[f'{stem}_fdr'] < ALPHA
            ax.scatter(cx - 0.27, cy, s=70, color=c if filled else 'white', edgecolor=c,
                       linewidth=1.8, zorder=3,
                       linestyle='--' if out == 'unstable' else '-')
            ax.text(cx - 0.12, cy, fmt_p(r[f'{stem}_p']), ha='left', va='center', fontsize=9,
                    color=INK if out in ('holds', 'lost_mouse') else MUTED)
    ax.set_xlim(-0.1, x_end + 0.1)
    ax.set_ylim(n_y, -2.6)
    ax.axis('off')


def survives(t, stem):
    """Significant naively and with insertions kept together. Excludes 'unstable' rows,
    which reach p < .05 only under a stricter null."""
    return t[f'{stem}_outcome'].isin(['holds', 'lost_mouse'])


NULL_SHORT = ['global', 'insertions\ntogether', 'within\nmouse']


def draw_table_simple(ax, t):
    """Feature | n | (global, insertions together, within mouse) for each test.

    Dot per cell: filled = p < .05 at that null, hollow = not; colour = the test's
    outcome across the three nulls (as in the full table). Bold p = also passes FDR
    at that null (FDR is not computed for the within-mouse null)."""
    x_name, x_n = 0.0, 3.05
    starts = [3.9, 7.5]
    cell = 1.15
    width = starts[-1] + 3 * cell + 0.1
    y = 0
    ys, group_rows = [], []
    for g, d in t.groupby('group', sort=False):
        group_rows.append((y, g, d))
        y += 1
        for i in d.index:
            ys.append((i, y))
            y += 1
        y += 0.35
    for (head, stem, _), x0 in zip(SIMPLE_COLUMNS, starts):
        ax.text(x0 + 1.5 * cell, -1.55, head.replace('\n', ' '), ha='center', va='bottom',
                fontsize=10, color=INK, weight='bold')
        ax.plot([x0 + 0.1, x0 + 3 * cell - 0.1], [-1.45] * 2, color=MUTED, linewidth=0.8)
        for k, nm in enumerate(NULL_SHORT):
            ax.text(x0 + (k + 0.5) * cell, -0.15, nm, ha='center', va='bottom', fontsize=8.5,
                    color=INK_2)
        for k in range(2):  # arrows between nulls: each step is stricter
            ax.annotate('', xy=(x0 + (k + 1) * cell + 0.12, -0.9),
                        xytext=(x0 + (k + 1) * cell - 0.12, -0.9),
                        arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
    ax.text(x_n + 0.35, -0.15, 'n\nunits', ha='center', va='bottom', fontsize=8.5, color=INK_2)
    for yy, g, d in group_rows:
        r0 = d.iloc[0]
        ax.text(x_name, yy + 0.5, g, ha='left', va='center', fontsize=10.5, color=INK,
                weight='bold')
        ax.text(x_name + 2.05, yy + 0.5,
                f'{d.n_insertions.max()} insertions · {d.n_mice.max()} mice '
                f'({d.n_mice_multi.max()} with ≥2 insertions)',
                ha='left', va='center', fontsize=8.5, color=INK_2)
        ax.plot([0, width], [yy + 0.95] * 2, color=GRID, linewidth=0.8)
    for j, (i, yy) in enumerate(ys):
        r = t.loc[i]
        if j % 2:
            ax.fill_between([0, width], yy, yy + 1, color=BAND, linewidth=0, zorder=0)
        ax.text(0.15, yy + 0.5, r.feature, ha='left', va='center', fontsize=9.5, color=INK)
        odd_n = r.n_units != t[t.group == r.group].n_units.max()
        ax.text(x_n + 0.35, yy + 0.5, f'{r.n_units}', ha='center', va='center', fontsize=9,
                color=INK if odd_n else INK_2, weight='bold' if odd_n else 'normal')
        for (_, stem, _), x0 in zip(SIMPLE_COLUMNS, starts):
            c = OUTCOME_COLOUR[r[f'{stem}_outcome']]
            for k, n in enumerate(NULLS):
                pv, q = r[f'{stem}_p_{n}'], r[f'{stem}_fdr_{n}']
                cx, cy = x0 + k * cell + 0.3, yy + 0.5
                sig = pv < ALPHA
                ax.scatter(cx, cy, s=48, color=c if sig else 'white', edgecolor=c,
                           linewidth=1.5, zorder=3)
                ax.text(cx + 0.14, cy, fmt_p(pv), ha='left', va='center', fontsize=8.5,
                        color=INK if sig else MUTED,
                        weight='bold' if q < ALPHA else 'normal')
    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(y, -2.4)
    ax.axis('off')


NULL_EXPLAINER = [
    ('1  Global shuffle', 'as in the figure panels',
     'Feature values are shuffled across all units. Treats every unit as independent, so '
     'the similarity of units from one insertion makes chance gradients look rarer than '
     'they are: too lenient.'),
    ('2  Insertions kept together', 'the main test',
     'Each insertion\'s units keep their values and layout and move as one block to '
     'another insertion\'s location. Chance now includes "an unusual insertion happened '
     'to land here".'),
    ('3  Within mouse', 'sensitivity check',
     'Blocks only swap with insertions from the same mouse, which also removes '
     'differences between animals. Mice with one insertion cannot contribute, so power is '
     'lower.'),
]


def takeaways(t):
    """(heading, body) pairs; in each body, features marked * also pass FDR."""
    def names(mask, stem):
        d = t[mask]
        if not len(d):
            return 'none'
        return ', '.join(f + ('*' if q < ALPHA else '')
                         for f, q in zip(d.feature, d[f'{stem}_fdr']))
    session_sig = {stem: survives(t, stem) for _, stem, k in COLUMNS if k == 'ladder'}
    held = t.knn_groupkfold_session_perm_p < ALPHA
    return [
        ('DV gradient, p < .05 with insertions kept together', names(session_sig['DV'], 'DV')),
        ('…of those, also significant inside single insertions (along the track)',
         names(t.within_p < ALPHA, 'within')),
        ('ML gradient, p < .05 with insertions kept together', names(session_sig['ML'], 'ML')),
        ('kNN prediction that carries to held-out insertions',
         names(held, 'knn_groupkfold_session_perm')),
        ('Significant on some naive test, nothing survives with insertions kept together',
         ', '.join(t.loc[~pd.concat(session_sig, axis=1).any(axis=1) & ~held, 'feature'])
         or 'none'),
    ]


def takeaways_simple(t):
    def names(mask, stem):
        d = t[mask]
        if not len(d):
            return 'none'
        return ', '.join(f + ('*' if q < ALPHA else '')
                         for f, q in zip(d.feature, d[f'{stem}_fdr']))
    lin, knn = survives(t, 'r2'), survives(t, 'knn')
    return [
        ('Linear 3D gradient, p < .05 with insertions kept together', names(lin, 'r2')),
        ('kNN prediction from position, p < .05 with insertions kept together',
         names(knn, 'knn')),
        ('Both', ', '.join(t.loc[lin & knn, 'feature']) or 'none'),
        ('Significant naively, lost once insertions are kept together',
         ', '.join(t.loc[~lin & ~knn, 'feature']) or 'none'),
    ]


def simple_rows(t):
    """Rows of the simple summary: p < ALPHA under the global shuffle on the omnibus
    linear or the kNN test."""
    naive = np.array([t.loc[i, 'r2_p_global_perm'] < ALPHA or t.loc[i, 'knn_p_global_perm'] < ALPHA
                      for i in t.index])
    return t[naive].reset_index(drop=True)


def make_figure(t, mode='full'):
    simple = mode == 'simple'
    n_rows = len(t) + 3 * 1.35
    h = max(12.2 if simple else 9.0, 0.3 * n_rows + 2.6)
    fig = plt.figure(figsize=(18 if simple else 17, h), facecolor='white')
    if simple:
        ax = fig.add_axes([0.03, 1.4 / h, 0.56, 1 - 3.0 / h])
        draw_table_simple(ax, t)
    else:
        ax = fig.add_axes([0.03, 0.11, 0.6, 1 - 2.2 / h])
        draw_table(ax, t)

    fig.text(0.03, 1 - 0.3 / h, 'What survives the insertion-aware controls',
             fontsize=19, weight='bold', color=INK, va='top')
    fig.text(0.03, 1 - 0.78 / h,
             ('The two tests behind the "LR p" and "KNN p" on the figure panels (does position '
              'predict the feature at all, in any direction?), each under three nulls of '
              'increasing strictness.\nFilled dot = p < .05 at that null; colour = outcome '
              'across the three; bold p = also passes FDR.' if simple else
              'Every feature significant on at least one naive test. Dot colour: what happens '
              'as the null respects insertions. Number: p with insertions kept together (the '
              'main test).'),
             fontsize=10.5, color=INK_2, va='top')

    x, y = (0.625 if simple else 0.67), 1 - 1.55 / h
    line = 0.21 / h  # one text line, in figure fraction
    if simple:
        fig.text(x, y, 'The three nulls, from lenient to strict', fontsize=13, color=INK,
                 weight='bold', va='top')
        y -= 1.7 * line
        for head, role, body in NULL_EXPLAINER:
            fig.text(x, y, head, fontsize=10.5, color=INK, weight='bold', va='top')
            fig.text(x + 0.165, y, role, fontsize=9.5, color=INK_2, style='italic', va='top')
            y -= 1.15 * line
            body = textwrap.fill(body, 70)
            fig.text(x + 0.012, y, body, fontsize=9.3, color=INK_2, va='top')
            y -= (body.count('\n') + 1) * line + 0.6 * line
        y -= 0.3 * line
        reading = textwrap.fill('Reading a row: holds at 2 = credible; also at 3 = strong; '
                                'holds at 2 but not 3 = an animal effect, or too few mice to '
                                'tell. Significant only at a stricter null = unstable.', 72)
        fig.text(x, y, reading, fontsize=9.3, color=INK, va='top')
        y -= (reading.count('\n') + 1) * line + 1.4 * line
    fig.text(x, y, 'Takeaways   (* = also passes FDR)', fontsize=13, color=INK,
             weight='bold', va='top')
    y -= 1.8 * line
    for head, body in (takeaways_simple(t) if simple else takeaways(t)):
        head = textwrap.fill(head, 70 if simple else 62)
        body = textwrap.fill(body, 60 if simple else 52)
        fig.text(x, y, head, fontsize=9.5, color=INK_2, va='top')
        y -= (head.count('\n') + 1) * line
        fig.text(x, y, body, fontsize=10.5, color=INK, weight='bold', va='top')
        y -= (body.count('\n') + 1) * line * 1.1 + 0.9 * line
    caveat = textwrap.fill(
        'Caveats: these tests do not say which axis carries the effect; the full summary '
        'splits ML / AP / DV. Most surviving effects are DV and mostly between insertions. '
        'Projection-target (04) and CCA results were not re-tested.' if simple else
        'Caveats: DV effects are mostly between-insertion; within an insertion DV and AP '
        'move together (probe tilt), so the within-insertion column is a gradient along the '
        'track and cannot separate DV from AP. Surviving ML effects co-vary with recording '
        'quality across insertions. Projection-target (04) and CCA results were not re-tested.',
        68)
    fig.text(x, y - 0.3 * line, caveat, fontsize=9.5, color=INK_2, va='top')

    handles = [Line2D([], [], marker='o', linestyle='', markersize=9, markerfacecolor=c,
                      markeredgecolor=c, label=lab)
               for c, lab in [(HOLDS, 'p < .05 under all three nulls'),
                              (LOST_MOUSE, 'holds with insertions kept together, lost within mouse'),
                              (LOST, 'lost when insertions are kept together'),
                              (MUTED, 'not significant naively (or unstable, dashed)')]]
    if simple:
        handles[-1] = Line2D([], [], marker='o', linestyle='', markersize=9, markerfacecolor=MUTED,
                             markeredgecolor=MUTED, label='not significant naively, or unstable')
        handles.append(Line2D([], [], marker='o', linestyle='', markersize=9,
                              markerfacecolor='white', markeredgecolor=INK_2, markeredgewidth=1.8,
                              label='hollow = p ≥ .05 at that null'))
    else:
        handles.append(Line2D([], [], marker='o', linestyle='', markersize=9,
                              markerfacecolor='white', markeredgecolor=INK_2, markeredgewidth=1.8,
                              label='hollow = fails FDR with insertions kept together'))
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.025, 0.045), ncol=3,
               frameon=False, fontsize=9.5, labelcolor=INK_2)
    fig.text(0.03, 0.008,
             ('Rows: features with p < .05 under the global shuffle on either test. Bold n = '
              'fewer units than the rest of its dataset (missing values). Linear: '
              'permutation p of R² for value ~ |ML| + AP + DV. kNN: 5-fold CV R² of a '
              'distance-weighted kNN on 3D position.\nFDR (BH) across features within dataset '
              'and column; firing/ACG duplicates A1, A3, peak3 omitted. ' if simple else
              'Single-step columns (kNN held out, DV within insertion): blue = p < .05, grey = '
              'not. Within-insertion: hierarchical bootstrap, tested only for DV survivors (— = '
              'not tested). FDR (BH) across features within dataset and column; firing/ACG '
              'duplicates A1, A3, peak3 omitted.\n')
             + 'Source: 06_session_bias_controls/*.csv; figure_preparation/cross_modality_summary.py',
             fontsize=8, color=MUTED, ha='left', va='bottom')
    return fig


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    t = build_table()
    t.to_csv(os.path.join(OUT_DIR, 'cross_modality_summary.csv'), index=False)
    if mode == 'simple':
        t = simple_rows(t)
        columns = SIMPLE_COLUMNS
    else:
        columns = COLUMNS
    print(t[['group', 'feature'] + [f'{s}_outcome' for _, s, _ in columns]].to_string())
    fig = make_figure(t, mode)
    path = os.path.join(OUT_DIR, 'cross_modality_summary'
                        + ('_simple' if mode == 'simple' else ''))
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}', dpi=200)
    print('wrote', path + '.png')


if __name__ == '__main__':
    main()
