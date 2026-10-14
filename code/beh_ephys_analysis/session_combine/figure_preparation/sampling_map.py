"""Where the insertions are: unit and insertion positions over the DRN outline.

Top row: coronal (|ML| x DV), horizontal (|ML| x AP) and sagittal (AP x DV)
projections. Each insertion is a segment from its shallowest to its deepest unit,
units are dots, and colour is the insertion's mean |ML| so the same insertion can
be followed across planes. Bottom: each mouse's insertions along |ML|, sized by
unit count, which shows how much of the ML range rests on how few animals.

Run:  python session_combine/figure_preparation/sampling_map.py [dataset]
      dataset defaults to waveform_low_DRN
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from trimesh import load_mesh

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.capsule_migration import capsule_directories
from utils.ccf_utils import project_to_plane
from utils.spatial_session_bias import mouse_from_session

BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
MESH_FILE = '/root/capsule/data/ccf_meshes/DR_bregma_lps_mm.obj'

capsule_dirs = capsule_directories()
SUMMARY_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']),
                           'SUMMARY_DRN_spatial_organization')
SRC_DIR = os.path.join(SUMMARY_DIR, '05_stats_and_source_data')
OUT_DIR = os.path.join(SUMMARY_DIR, '06_session_bias_controls')
SOURCES = {
    'waveform_low_DRN': 'combined_features_waveform_low_DRN.csv',
    'beh_all_DRN': 'features_combined_beh_all_DRN.pkl',
    'basic_ephys_all_DRN': 'features_combined_basic_ephys_all_DRN.pkl',
}

ML, AP, DV = 0, 1, 2
PLANES = [('Coronal', ML, DV), ('Horizontal', ML, AP), ('Sagittal', AP, DV)]
AXIS_LABEL = {ML: '|ML| from midline (mm)', AP: 'AP from bregma (mm)',
              DV: 'DV from bregma (mm)'}

# Light-mode tokens; sequential blue ramp from step 250 (lightest legible on the
# light surface) to 700.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8985'
GRID = '#e4e3df'
MESH = '#ecebe7'
RAMP = LinearSegmentedColormap.from_list(
    'blue_seq', ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b'])


def load(label):
    path = os.path.join(SRC_DIR, SOURCES[label])
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_pickle(path)
    df = df[np.isfinite(df[CCF_COLS].to_numpy(float)).all(axis=1)].copy()
    X = df[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    X[:, ML] = np.abs(X[:, ML])
    df['ML'], df['AP'], df['DV'] = X[:, ML], X[:, AP], X[:, DV]
    df['mouse'] = mouse_from_session(df['session'].to_numpy())
    return df


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def plane_panel(ax, df, ins, mesh_v, title, xd, yd, norm):
    for c in project_to_plane(mesh_v, [xd, yd], pitch=0.02, margin=0.5):
        ax.fill(c[:, 0], c[:, 1], color=MESH, linewidth=0, zorder=0)
    cols = ['ML', 'AP', 'DV']
    for sess, r in ins.iterrows():
        d = df[df.session == sess]
        colour = RAMP(norm(r.ML))
        top, bottom = d.loc[d.DV.idxmax()], d.loc[d.DV.idxmin()]
        ax.plot([top[cols[xd]], bottom[cols[xd]]], [top[cols[yd]], bottom[cols[yd]]],
                color=colour, linewidth=2, solid_capstyle='round', zorder=2)
        ax.scatter(d[cols[xd]], d[cols[yd]], s=14, color=colour, edgecolor='white',
                   linewidth=0.6, zorder=3)
    pts = np.vstack([df[cols].to_numpy(), mesh_v])
    pad = 0.12
    ax.set_xlim(pts[:, xd].min() - pad, pts[:, xd].max() + pad)
    ax.set_ylim(pts[:, yd].min() - pad, pts[:, yd].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel(AXIS_LABEL[xd], fontsize=9.5, color=INK_2)
    ax.set_ylabel(AXIS_LABEL[yd], fontsize=9.5, color=INK_2)
    ax.set_title(title, loc='left', fontsize=12, color=INK, pad=6)
    style(ax)


def mouse_panel(ax, ins, norm):
    mice = (ins.groupby('mouse').ML.mean().sort_values().index.tolist())
    y_of = {m: i for i, m in enumerate(mice[::-1])}
    for m in mice:
        d = ins[ins.mouse == m]
        y = y_of[m]
        ax.plot([d.ML.min(), d.ML.max()], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter(d.ML, [y] * len(d), s=12 + 9 * d.n, color=RAMP(norm(d.ML)),
                   edgecolor='white', linewidth=1, zorder=2)
    ax.set_yticks(list(y_of.values()), [f'{m}  ({(ins.mouse == m).sum()} ins.)'
                                         for m in y_of])
    ax.set_xlabel('insertion mean |ML| from midline (mm)', fontsize=9.5, color=INK_2)
    ax.set_title('Each mouse\'s insertions along ML (dot area = units in that insertion)',
                 loc='left', fontsize=12, color=INK, pad=6)
    ax.set_ylim(-0.7, len(mice) - 0.3)
    style(ax)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for n in (1, 5, 15):
        ax.scatter([], [], s=12 + 9 * n, color=MUTED, edgecolor='white',
                   label=f'{n} unit' + ('s' if n > 1 else ''))
    ax.legend(loc='upper right', frameon=False, fontsize=9, labelcolor=INK_2,
              labelspacing=1.2, borderpad=1)


def make_figure(label):
    df = load(label)
    ins = df.groupby('session').agg(ML=('ML', 'mean'), AP=('AP', 'mean'), DV=('DV', 'mean'),
                                    ML_sd=('ML', 'std'), DV_sd=('DV', 'std'),
                                    n=('ML', 'size'), mouse=('mouse', 'first'))
    mesh_v = np.asarray(load_mesh(MESH_FILE).vertices, float).copy()
    mesh_v[:, ML] = np.abs(mesh_v[:, ML])
    norm = Normalize(0, ins.ML.max())

    fig = plt.figure(figsize=(16, 11.5), facecolor='white')
    # top-row widths follow each plane's x/y extent so equal-aspect panels fill their slot
    ext = np.vstack([df[['ML', 'AP', 'DV']].to_numpy(), mesh_v])
    span = ext.max(axis=0) - ext.min(axis=0) + 0.24
    gs_top = fig.add_gridspec(1, 3, left=0.07, right=0.92, top=0.85, bottom=0.5, wspace=0.3,
                              width_ratios=[span[xd] / span[yd] for _, xd, yd in PLANES])
    for k, (title, xd, yd) in enumerate(PLANES):
        plane_panel(fig.add_subplot(gs_top[0, k]), df, ins, mesh_v, title, xd, yd, norm)
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.93, top=0.86, bottom=0.07,
                          hspace=0.32, wspace=0.28, height_ratios=[1.15, 1])
    ax_m = fig.add_subplot(gs[1, :2])
    mouse_panel(ax_m, ins, norm)

    cax = fig.add_axes([0.945, 0.52, 0.01, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax)
    cb.set_label('insertion mean |ML| (mm)', fontsize=9.5, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=9, length=0)

    single = int((ins.groupby('mouse').size() == 1).sum())
    lateral = ins[ins.ML > 0.3]
    facts = [
        ('Median within-insertion SD (insertions with ≥ 2 units)',
         f'ML {ins.ML_sd.median() * 1000:.0f} µm   ·   DV {ins.DV_sd.median() * 1000:.0f} µm'),
        ('Insertions with one unit', f'{int((ins.n == 1).sum())} of {len(ins)}'),
        ('Largest insertion', f'{int(ins.n.max())} of {len(df)} units'),
        ('Mice with a single insertion', f'{single} of {ins.mouse.nunique()}  '
                                         '(drop out of the within-mouse null)'),
        ('Insertions lateral of 0.3 mm', f'{len(lateral)}, from {lateral.mouse.nunique()} mice, '
                                         f'{int(lateral.n.sum())} units'),
    ]
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.axis('off')
    ax_t.set_title('Sampling in numbers', loc='left', fontsize=12, color=INK, pad=6)
    for i, (k, v) in enumerate(facts):
        y = 0.92 - i * 0.19
        ax_t.text(0, y, k, fontsize=9.5, color=INK_2, transform=ax_t.transAxes, va='top')
        ax_t.text(0, y - 0.065, v, fontsize=11.5, color=INK, weight='bold',
                  transform=ax_t.transAxes, va='top')

    fig.text(0.07, 0.965, f'Where the recordings are — {label}', fontsize=17, weight='bold',
             color=INK, va='top')
    fig.text(0.07, 0.93,
             f'{len(df)} optotagged units, {len(ins)} insertions, {ins.mouse.nunique()} mice. '
             'Segments run from each insertion\'s shallowest to deepest unit.\n'
             'Insertions are long in DV and nearly a point in ML, so DV is sampled within '
             'insertions and ML only between them. Probes are tilted in the sagittal plane, '
             'so within an insertion DV and AP change together.',
             fontsize=10.5, color=INK_2, va='top')
    fig.text(0.07, 0.015, 'Coordinates in mm from bregma, ML mirrored to one side; grey outline '
             'is the DRN mesh (DR_bregma_lps_mm.obj). '
             'Source: figure_preparation/sampling_map.py',
             fontsize=8.5, color=MUTED, va='bottom')
    return fig


if __name__ == '__main__':
    label = sys.argv[1] if len(sys.argv) > 1 else 'waveform_low_DRN'
    fig = make_figure(label)
    path = os.path.join(OUT_DIR, f'sampling_map_{label}')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}', dpi=200)
    print('wrote', path + '.png')
