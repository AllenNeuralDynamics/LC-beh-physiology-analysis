# %%
"""
What excluding the opto blocks does to the ISI-violations QC.

Reads the per-unit table add_isi_nonopto.py writes and answers the question that
motivated the new metric: are laser artifacts inflating isi_violations_ratio, and does
that unfairly fail units -- the tagged ones in particular?

Three ratios per unit, all from the same estimator:
  _window   the curated session window, laser included -- reproduces the pipeline number
  _nonopto  the same window minus the opto blocks -- the proposed QC number
  _opto     the opto blocks alone -- the artifact diagnostic

Because the ratio is a rate ratio it is invariant under uniform thinning of the analysis
window, so _window vs _nonopto is not a duration artifact: a drop means the violations
really are concentrated in the laser epochs.

Text report to stdout, one figure to the manuscript fig-prep dir.
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
from scipy.stats import wilcoxon

from utils.capsule_migration import capsule_directories

# ---- palette (reference instance, used verbatim) ----
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8983'
DOWN = '#2a78d6'   # diverging cool pole
UP = '#e34948'     # diverging warm pole
MID = '#f0efec'    # neutral midpoint
GRID = '#e6e5e1'

UNIT_CSV = '/root/capsule/scratch/combined/isi_nonopto_units.csv'
THRESHOLDS = (0.4, 0.1)   # default_qc, and the tighter bound the tagging criteria use
NOISE_LABELS = ('noise', 'artifact')

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'isi_nonopto')


def load_units(path=UNIT_CSV):
    units = pd.read_csv(path)
    units['is_noise'] = units['decoder_label'].isin(NOISE_LABELS)
    if 'opto_pass' not in units.columns:
        units['opto_pass'] = False
    units['opto_pass'] = units['opto_pass'] == True   # NaN (never tested) counts as False
    # a unit with no spikes left outside the laser has no laser-free ISI to judge it on
    units['has_nonopto'] = units['num_spikes_nonopto'] > 1
    return units


def groups(units):
    """The populations worth reporting separately, coarsest first."""
    return [
        ('all sorted units', units),
        ('non-noise units', units[~units['is_noise']]),
        ('opto_pass units', units[units['opto_pass']]),
        ('opto_pass, non-noise', units[units['opto_pass'] & ~units['is_noise']]),
    ]


def passes(units, col, thr):
    """default_qc with the ISI bound swapped: label not noise/artifact AND ratio < thr."""
    ratio = units[col]
    if col == 'isi_violations_ratio_nonopto':
        # units with nothing left outside the laser fall back on the pipeline number,
        # exactly as utils.isi_qc.default_qc does
        ratio = ratio.where(units['has_nonopto'], units['isi_violations_ratio'])
    return ~units['is_noise'] & (ratio < thr)


def report_coverage(units):
    print(f'{len(units)} units, {units["session"].nunique()} sessions')
    frac = units.groupby('session')['frac_nonopto'].first()
    print(f'laser-free fraction of the session window: median {frac.median():.2f}, '
          f'range {frac.min():.2f}-{frac.max():.2f}')
    print(f'units with no spikes outside the laser: {int((~units["has_nonopto"]).sum())}')

    # the recomputation has to reproduce the pipeline number before its variants mean
    # anything
    ok = units['isi_violations_ratio'] > 0
    rel = ((units.loc[ok, 'isi_violations_ratio_window'] - units.loc[ok, 'isi_violations_ratio'])
           / units.loc[ok, 'isi_violations_ratio']).abs()
    print(f'recomputed window ratio vs pipeline ratio: median |rel. diff| {rel.median():.3f}, '
          f'90th pct {rel.quantile(0.9):.3f}')


def report_ratios(units):
    print('\n--- ISI violations ratio by window ---')
    rows = []
    for label, subset in groups(units):
        if len(subset) == 0:
            continue
        sub = subset[subset['has_nonopto']]
        drop = sub['isi_violations_ratio_window'] - sub['isi_violations_ratio_nonopto']
        stat = np.nan
        if len(sub) > 5 and (drop != 0).any():
            stat = wilcoxon(sub['isi_violations_ratio_window'],
                            sub['isi_violations_ratio_nonopto']).pvalue
        rows.append({
            'group': label, 'n': len(sub),
            'window': sub['isi_violations_ratio_window'].median(),
            'nonopto': sub['isi_violations_ratio_nonopto'].median(),
            'opto': sub['isi_violations_ratio_opto'].median(),
            'median_drop': drop.median(),
            'frac_dropping': float((drop > 0).mean()),
            'p_wilcoxon': stat,
        })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    return tbl


def report_flips(units):
    print('\n--- QC flips: default_qc on the pipeline ISI vs on the laser-free ISI ---')
    rows = []
    for thr in THRESHOLDS:
        pass_all = passes(units, 'isi_violations_ratio', thr)
        pass_non = passes(units, 'isi_violations_ratio_nonopto', thr)
        for label, subset in groups(units):
            if len(subset) == 0:
                continue
            idx = subset.index
            gained = pass_non[idx] & ~pass_all[idx]
            lost = ~pass_non[idx] & pass_all[idx]
            # a flip that only happens in one session is a session effect, not a QC effect
            n_sess_gained = subset.loc[gained, 'session'].nunique()
            rows.append({
                'isi_max': thr, 'group': label, 'n': len(subset),
                'pass_pipeline': int(pass_all[idx].sum()),
                'pass_nonopto': int(pass_non[idx].sum()),
                'frac_pipeline': float(pass_all[idx].mean()),
                'frac_nonopto': float(pass_non[idx].mean()),
                'gained': int(gained.sum()), 'lost': int(lost.sum()),
                'sessions_gaining': n_sess_gained,
                'sessions_total': subset['session'].nunique(),
            })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    # both flags carry the noise/artifact label term, so the pass counts for a group and
    # its non-noise subset are identical by construction; only the denominator moves
    return tbl


def report_artifact_units(units):
    """
    Units whose violations live almost entirely in the laser epochs.

    These are the artifact 'units' the hypothesis predicts: enormous pipeline ratios,
    almost no spikes outside the laser. They are the reason the _opto column exists --
    it separates 'the laser inflated a real unit' from 'this was never a unit'.
    """
    print('\n--- units whose spikes are mostly laser-locked ---')
    laser_locked = units[units['frac_nonopto'].notna() & (units['num_spikes_nonopto'] <
                                                          0.1 * units['num_spikes_window'])]
    print(f'{len(laser_locked)} units have <10% of their spikes outside the laser '
          f'({laser_locked["session"].nunique()} sessions)')
    if len(laser_locked):
        thr = THRESHOLDS[0]
        pass_all = passes(laser_locked, 'isi_violations_ratio', thr)
        pass_non = passes(laser_locked, 'isi_violations_ratio_nonopto', thr)
        print(f'  median pipeline ISI ratio {laser_locked["isi_violations_ratio"].median():.1f}, '
              f'{int(laser_locked["opto_pass"].sum())} of them opto_pass')
        print(f'  pass at ISI < {thr}: {int(pass_all.sum())} on the pipeline metric, '
              f'{int(pass_non.sum())} on the laser-free metric')

    # the caveat that matters for tagging: a unit rescued by dropping the laser epochs is
    # being judged on a window that the opto response is not measured in
    thr = THRESHOLDS[0]
    gained = (passes(units, 'isi_violations_ratio_nonopto', thr)
              & ~passes(units, 'isi_violations_ratio', thr))
    rescued_tagged = units[gained & units['opto_pass']]
    print(f'\n--- {len(rescued_tagged)} opto_pass units gained at ISI < {thr} ---')
    if len(rescued_tagged):
        print(f'  their ISI ratio inside the laser epochs: median '
              f'{rescued_tagged["isi_violations_ratio_opto"].median():.2f}, '
              f'{int((rescued_tagged["isi_violations_ratio_opto"] > thr).sum())} of them '
              f'still above {thr} there')
        print('  their tagging evidence comes from the epochs the new metric excludes, so '
              'check isi_violations_ratio_opto before trusting them as tagged')


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_scatter(ax, units):
    sub = units[units['has_nonopto']]
    for mask, color, label, size in ((~sub['opto_pass'], INK_MUTED, 'other', 6),
                                     (sub['opto_pass'], UP, 'opto_pass', 10)):
        ax.scatter(sub.loc[mask, 'isi_violations_ratio_window'].clip(1e-3),
                   sub.loc[mask, 'isi_violations_ratio_nonopto'].clip(1e-3),
                   s=size, c=color, alpha=0.55, lw=0, label=label, zorder=3)
    lims = [1e-3, 1e3]
    ax.plot(lims, lims, color=INK, lw=0.8, zorder=4)
    ax.axhline(0.4, color=DOWN, lw=0.8, ls='--', zorder=2)
    ax.axvline(0.4, color=DOWN, lw=0.8, ls='--', zorder=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)
    style(ax)
    ax.set_xlabel('ISI violations ratio, laser included', color=INK_2, fontsize=9)
    ax.set_ylabel('laser-free', color=INK_2, fontsize=9)
    ax.set_title('below the diagonal = the laser inflated it', color=INK, fontsize=9.5,
                 loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc='upper left')


def panel_frac(ax, units):
    frac = units.groupby('session')['frac_nonopto'].first()
    ax.hist(frac, bins=np.linspace(0, 1, 21), color=DOWN, alpha=0.85, zorder=3)
    ax.axvline(frac.median(), color=INK, lw=1, zorder=4)
    style(ax)
    ax.set_xlabel('laser-free fraction of session window', color=INK_2, fontsize=9)
    ax.set_ylabel('sessions', color=INK_2, fontsize=9)
    ax.set_title(f'median {frac.median():.2f} of each session is laser-free',
                 color=INK, fontsize=9.5, loc='left')


def panel_flips(ax, flips):
    thr = THRESHOLDS[0]
    # a group and its non-noise subset flip the same units, so only the subsets are shown
    sub = flips[(flips['isi_max'] == thr)
                & flips['group'].isin(['non-noise units', 'opto_pass, non-noise'])]
    y = np.arange(len(sub))
    ax.barh(y, sub['gained'], color=DOWN, zorder=3, label='gained')
    ax.barh(y, -sub['lost'], color=UP, zorder=3, label='lost')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{g}\n(n={n})' for g, n in zip(sub['group'], sub['n'])], fontsize=7.5)
    ax.axvline(0, color=INK, lw=0.8, zorder=4)
    style(ax)
    ax.set_xlabel(f'units changing default_qc at ISI < {thr}', color=INK_2, fontsize=9)
    ax.set_title('switching the QC metric', color=INK, fontsize=9.5, loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc='upper right')


def panel_change(ax, units):
    """
    How far each unit's ratio moves when the laser epochs come out.

    A paired-line plot of 1085 units is unreadable, and the quantity of interest is the
    per-unit fold change anyway: negative = the laser was inflating the ratio.
    """
    sub = units[~units['is_noise'] & units['has_nonopto']
                & (units['isi_violations_ratio_window'] > 0)
                & (units['isi_violations_ratio_nonopto'] > 0)]
    change = np.log10(sub['isi_violations_ratio_nonopto']
                      / sub['isi_violations_ratio_window'])
    bins = np.linspace(-2, 2, 61)
    for mask, color, label in ((~sub['opto_pass'], INK_MUTED, 'other'),
                               (sub['opto_pass'], UP, 'opto_pass')):
        ax.hist(change[mask].clip(bins[0], bins[-1]), bins=bins, density=True,
                histtype='step', lw=1.4, color=color, label=label, zorder=3)
        ax.axvline(change[mask].median(), color=color, lw=0.8, ls=':', zorder=4)
    ax.axvline(0, color=INK, lw=0.9, zorder=5)
    style(ax)
    ax.set_xlabel('log10(laser-free / laser-included ratio)', color=INK_2, fontsize=9)
    ax.set_ylabel('density', color=INK_2, fontsize=9)
    ax.set_title('opto_pass units shift left: their violations were laser-locked',
                 color=INK, fontsize=9.5, loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc='upper left')


def make_figure(units, flips):
    fig = plt.figure(figsize=(10, 7.5), facecolor=SURFACE)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30,
                  left=0.085, right=0.965, top=0.88, bottom=0.09)
    panel_scatter(fig.add_subplot(gs[0, 0]), units)
    panel_change(fig.add_subplot(gs[0, 1]), units)
    panel_frac(fig.add_subplot(gs[1, 0]), units)
    panel_flips(fig.add_subplot(gs[1, 1]), flips)
    fig.suptitle('Excluding opto blocks from the ISI-violations metric',
                 color=INK, fontsize=12, fontweight='bold', x=0.085, ha='left', y=0.955)
    fig.text(0.085, 0.915,
             'same estimator as the sorting pipeline, restricted to the laser-free part '
             'of the curated session window',
             color=INK_2, fontsize=8.5, ha='left')

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'isi_nonopto_impact.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f'wrote {path}')
    plt.close(fig)


# %%
if __name__ == '__main__':
    units = load_units()
    report_coverage(units)
    ratios = report_ratios(units)
    flips = report_flips(units)
    report_artifact_units(units)
    make_figure(units, flips)

    os.makedirs(OUT_DIR, exist_ok=True)
    ratios.to_csv(os.path.join(OUT_DIR, 'isi_nonopto_ratios.csv'), index=False)
    flips.to_csv(os.path.join(OUT_DIR, 'isi_nonopto_flips.csv'), index=False)
    print(f'\ntables written to {OUT_DIR}')
