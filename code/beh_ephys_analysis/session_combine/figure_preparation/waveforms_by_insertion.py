"""Normalised waveforms of optotagged DRN units, one panel per insertion, ordered by ML.

For judging by eye how similar units from one insertion are. Units and waveform
normalisation follow F_waveform_space_DRN_v2.ipynb (criteria waveform_low_DRN,
ISI < 0.5, finite CCF; baseline-subtracted and scaled by the trough sample).

Writes two figures with identical layout:
  waveforms_by_insertion_<criteria>.png/.pdf          real insertions
  waveforms_by_insertion_<criteria>_shuffled.png/.pdf the same panel sizes, with
      units dealt into panels at random: what panel-to-panel differences look
      like when insertion identity carries no information.

Run:  python session_combine/figure_preparation/waveforms_by_insertion.py
"""

import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.capsule_migration import capsule_directories
from utils.combine_tools import apply_qc
from utils.spatial_session_bias import mouse_from_session

# Same configuration as F_waveform_space_DRN_v2.ipynb
criteria_name = 'waveform_low_DRN'
waveform_version = '_raw'
ISI_MAX = 0.5
UNIT_KEY = ['session', 'unit']
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
FS_HZ = 30000
TROUGH_SAMPLE = 30
N_COLS = 8
SEED = 42

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']),
                       'SUMMARY_DRN_spatial_organization', '06_session_bias_controls')

# Light-mode tokens from the dataviz reference palette.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8985'
GRID = '#e4e3df'
UNIT = '#2a78d6'
REF = '#d6d5d0'


def safe_divide(num, den, eps=1e-10):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    np.divide(num, den, out=out, where=valid)
    return out


def load_units():
    """Per-unit table with |ML|, AP, DV (mm from bregma) and the normalised waveform."""
    with open(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'combined_unit_tbl',
                           'combined_unit_tbl.pkl'), 'rb') as f:
        units = pickle.load(f)
    with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics',
                           f'{criteria_name}.json')) as f:
        constraints = json.load(f)
    tagged, _, fig, _ = apply_qc(units, constraints)
    plt.close(fig)

    wfs = np.stack(tagged[f'wf{waveform_version}'].to_numpy()).astype(float)
    baseline = np.nanmean(wfs[:, :3], axis=1)
    wf_nl = safe_divide(wfs - baseline[:, None], (wfs[:, TROUGH_SAMPLE] - baseline)[:, None])
    # notebook's old bad-shape rule
    raw_nl = safe_divide(wfs, wfs[:, TROUGH_SAMPLE][:, None])
    wf_nl[np.isfinite(raw_nl[:, 0]) & (raw_nl[:, 0] > 1)] = np.nan

    df = tagged[UNIT_KEY + CCF_COLS + ['isi_violations']].copy()
    # trough is +1 after scaling by a negative trough; flip so it plots as a spike
    df['wf'] = list(-wf_nl)
    wf_features = pd.read_csv(os.path.join(capsule_dirs['manuscript_fig_prep_dir'],
                                           'waveforms_np', 'combined_features.csv'))
    df = df.merge(wf_features[UNIT_KEY], on=UNIT_KEY, how='inner', validate='one_to_one')
    ok = (df['isi_violations'].lt(ISI_MAX)
          & np.isfinite(df[CCF_COLS].to_numpy(float)).all(axis=1)
          & np.array([np.isfinite(w).all() for w in df['wf']]))
    df = df.loc[ok].reset_index(drop=True)
    X = df[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    df['ML'], df['AP'], df['DV'] = np.abs(X[:, 0]), X[:, 1], X[:, 2]
    df['mouse'] = mouse_from_session(df['session'].to_numpy())
    return df


def insertion_order(df):
    """Sessions sorted by mean |ML|, with the per-session summary used in titles."""
    g = df.groupby('session').agg(ML=('ML', 'mean'), ML_sd=('ML', 'std'), AP=('AP', 'mean'),
                                  n=('unit', 'size'), mouse=('mouse', 'first'))
    return g.sort_values('ML')


def make_figure(df, groups, title, subtitle, real=True):
    t_ms = (np.arange(len(df['wf'].iloc[0])) - TROUGH_SAMPLE) / FS_HZ * 1000
    all_w = np.stack(df['wf'].to_numpy())
    ref = np.median(all_w, axis=0)
    lo, hi = np.percentile(all_w, [10, 90], axis=0)
    ylim = (-1.15, 0.75)  # a few noisy units run off the top; fixed so panels compare

    n_rows = int(np.ceil(len(groups) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(2.1 * N_COLS, 1.95 * n_rows + 1.6),
                             sharex=True, sharey=True, facecolor='white')
    fig.subplots_adjust(left=0.04, right=0.99, top=1 - 1.25 / (1.95 * n_rows + 1.6),
                        bottom=0.05, hspace=0.62, wspace=0.08)
    axes = axes.ravel()
    for ax, (label, idx) in zip(axes, groups):
        ax.fill_between(t_ms, lo, hi, color=REF, alpha=0.45, linewidth=0, zorder=1)
        ax.plot(t_ms, ref, color=MUTED, linewidth=1, zorder=2)
        for w in all_w[idx]:
            ax.plot(t_ms, w, color=UNIT, linewidth=1, alpha=0.7 if len(idx) < 10 else 0.45,
                    zorder=3)
        ax.set_title(label, fontsize=8.5, color=INK, loc='left', pad=3)
        ax.axhline(0, color=GRID, linewidth=0.6, zorder=0)
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    for ax in axes[:len(groups)]:
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
    axes[0].set_ylim(*ylim)
    axes[0].set_xlim(t_ms[0], t_ms[-1])
    for r in range(n_rows):
        axes[r * N_COLS].set_ylabel('norm. amplitude', fontsize=8, color=INK_2)
    for ax in axes[max(0, len(groups) - N_COLS):len(groups)]:
        ax.set_xlabel('ms from trough', fontsize=8, color=INK_2)
        ax.tick_params(labelbottom=True)

    top = 1 - 0.25 / (1.95 * n_rows + 1.6)
    fig.text(0.04, top, title, fontsize=15, weight='bold', color=INK, va='top')
    fig.text(0.04, top - 0.022, subtitle, fontsize=10, color=INK_2, va='top')
    return fig


def main():
    df = load_units()
    order = insertion_order(df)
    print(f'{len(df)} units, {len(order)} insertions, {df.mouse.nunique()} mice')

    groups = []
    for sess, r in order.iterrows():
        idx = np.flatnonzero(df['session'].to_numpy() == sess)
        sd = '' if r.n < 2 else f' ±{r.ML_sd * 1000:.0f} µm'
        groups.append((f'|ML| {r.ML:.2f} mm{sd} · n={r.n}\nmouse {r.mouse} · AP {r.AP:+.2f}', idx))
    common = (f'{len(df)} optotagged units ({criteria_name}), {len(order)} insertions, '
              f'{df.mouse.nunique()} mice. Blue: each unit\'s normalised waveform (trough = −1). '
              'Grey line and band: median and 10–90% of all units, the same in every panel.')
    fig = make_figure(df, groups,
                      'Waveforms by insertion, ordered medial → lateral by insertion |ML|',
                      common + '\nPanel title: mean |ML| ± within-insertion SD, unit count, '
                      'mouse, mean AP (mm from bregma). Read left to right, top to bottom.')
    path = os.path.join(OUT_DIR, f'waveforms_by_insertion_{criteria_name}')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}', dpi=200)
    plt.close(fig)

    # Same panel sizes, units dealt at random
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(df))
    start, shuffled = 0, []
    for k, (_, idx) in enumerate(groups):
        shuffled.append((f'random group {k + 1} · n={len(idx)}', perm[start:start + len(idx)]))
        start += len(idx)
    fig = make_figure(df, shuffled,
                      'Control: the same units dealt into panels at random',
                      common + '\nPanel sizes match the real insertions. Compare: how different '
                      'do panels look when insertion identity carries no information?')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}_shuffled.{ext}', dpi=200)
    plt.close(fig)
    print('wrote', path + '.png', 'and _shuffled')


if __name__ == '__main__':
    main()
