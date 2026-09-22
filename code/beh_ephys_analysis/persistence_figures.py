"""
Figures for the persistence ITI effect.

Four panels, telling one story: tagged DRN units reduce their tonic ITI firing as
the animal goes deeper into an unrewarded perseverative bout, and the effect is
specific to the ITI.

  A  paired per-unit ITI rate, shallow (streak==1) -> deep (streak>=3)
  B  distribution of per-unit effect sizes
  C  dose-response: ITI rate vs streak depth
  D  window specificity: the effect exists in the ITI, not the response

Colour: the documented diverging pair (blue <-> red, neutral gray midpoint) for
polarity -- decrease vs increase -- and a single blue for one-series magnitude.
Panel D highlights the one window that carries the story and grays the rest.
Values are taken verbatim from the reference palette; nothing re-stepped.

Reads what persistence_5HT_analysis.py wrote.
"""
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
import pickle
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

DEEP, SHALLOW = 3, 1
MIN_TRIALS = 5

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'persistence_5HT')

WINDOW_LABELS = {
    'iti': 'ITI',
    'iti_post': 'ITI (post-outcome)',
    'resp_gocue': 'Go cue',
    'resp_choice': 'Choice',
    'consumption': 'Consumption',
}


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


# %%
def panel_paired(ax, stats_iti):
    """A: shallow vs deep per unit against unity.

    A slope plot fails here: the effect is ~5% while units span two orders of
    magnitude, so 207 near-horizontal lines hide it. Against a unity line the
    systematic displacement is what the eye picks up.
    """
    d = stats_iti.dropna(subset=['diff'])
    sh_all, dp_all = d['fr_nonpersistent'].values, d['fr_persistent'].values
    down_all = dp_all < sh_all

    # counts and the test use every unit, so A agrees with B and with D's ITI
    # row; a log axis just cannot draw the near-silent ones, so they go in a note
    plottable = (sh_all > 0) & (dp_all > 0)
    sh, dp, down = sh_all[plottable], dp_all[plottable], down_all[plottable]

    lim = (min(sh.min(), dp.min()) * 0.7, max(sh.max(), dp.max()) * 1.4)
    ax.plot(lim, lim, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
            zorder=2)

    for mask, color in ((down, DOWN), (~down, UP)):
        ax.plot(sh[mask], dp[mask], linestyle='none', marker='o', markersize=5,
                color=color, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=0.8, alpha=0.75, zorder=3)

    ax.plot(np.median(sh), np.median(dp), marker='D', markersize=10,
            color=INK, markerfacecolor=INK, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=5)

    p = wilcoxon(dp_all, sh_all).pvalue
    n_off = int((~plottable).sum())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    # box_aspect, not aspect='equal': the latter renegotiates the limits on log
    # axes, so x and y end up on different ranges and unity stops being the
    # visual diagonal -- which is the whole point of the panel.
    ax.set_box_aspect(1)
    ax.set_xlabel(f'shallow, streak = {SHALLOW}  (Hz)', color=INK, fontsize=9)
    ax.set_ylabel(f'deep, streak $\\geq$ {DEEP}  (Hz)', color=INK, fontsize=9)
    ax.set_title('A   Per-unit ITI rate', color=INK, fontsize=10,
                 loc='left', fontweight='bold')
    style_axes(ax, grid_axis='both')

    # direct labels, placed in opposite corners so neither can collide
    ax.text(0.04, 0.93, f'{int(down_all.sum())} below unity', color=DOWN,
            transform=ax.transAxes, fontsize=8, va='top', fontweight='bold')
    ax.text(0.96, 0.05, f'{int((~down_all).sum())} above', color=UP,
            transform=ax.transAxes, fontsize=8, ha='right', fontweight='bold')
    # the note stacks under the blue label in the empty upper-left triangle;
    # the lower-right corner is left to the red label so the two cannot meet
    note = f'n = {len(d)} units\nWilcoxon p = {p:.1e}'
    if n_off:
        note += f'\n({n_off} near-silent, off log axis)'
    ax.text(0.04, 0.85, note, transform=ax.transAxes, color=INK_2,
            fontsize=8, va='top')


def panel_effect_dist(ax, stats_iti):
    """B: distribution of per-unit deltas, diverging by sign about zero.

    Bins span the full data range -- clipping to a percentile piles the tails
    into the edge bins and invents a mode that isn't there.
    """
    d = stats_iti.dropna(subset=['diff'])['diff'].values
    lim = np.nanmax(np.abs(d)) * 1.02
    bins = np.linspace(-lim, lim, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(d, bins=bins)

    for c, x, w in zip(counts, centers, np.diff(bins)):
        if c:
            ax.bar(x, c, width=w * 0.88, color=(DOWN if x < 0 else UP),
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)

    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)
    med = np.median(d)
    ax.set_ylim(0, counts.max() * 1.30)
    ax.axvline(med, color=INK, linewidth=2.0, zorder=4)
    ax.annotate(f'median {med:+.2f} Hz',
                xy=(med, counts.max() * 1.10), xytext=(-10, 0),
                textcoords='offset points', color=INK, fontsize=8,
                ha='right', va='center',
                arrowprops=dict(arrowstyle='-', color=INK, linewidth=0.8))

    ax.set_xlabel('$\\Delta$ ITI rate, deep $-$ shallow (Hz)', color=INK, fontsize=9)
    ax.set_ylabel('units', color=INK, fontsize=9)
    ax.set_title('B   Effect size per unit', color=INK, fontsize=10,
                 loc='left', fontweight='bold')
    style_axes(ax)
    # surface-coloured box: the median rule spans the full height and would
    # otherwise read straight through this line
    ax.text(0.02, 0.97,
            f'{100*(d<0).mean():.0f}% of {len(d)} units decrease',
            transform=ax.transAxes, color=INK_2, fontsize=8, va='top', zorder=6,
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))


def panel_dose(ax, trial_rates):
    """C: ITI rate vs streak depth, per-unit normalised, mean +/- SEM."""
    ok = trial_rates['in_streak_calc'] & trial_rates['covered']
    if 'in_cut' in trial_rates.columns:
        # same curated drift window the per-unit stats use, or this panel would
        # be drawn from a different set of trials than every other panel
        ok = ok & trial_rates['in_cut']
    tr = trial_rates[ok].copy()
    tr = tr.dropna(subset=['fr_iti', 'unrew_streak_before'])
    tr['depth'] = np.minimum(tr['unrew_streak_before'], 5)

    # normalise within unit so 0.3 Hz and 29 Hz units contribute equally
    unit_mean = tr.groupby(['session', 'unit_id'])['fr_iti'].transform('mean')
    tr = tr[unit_mean > 0]
    tr['norm'] = tr['fr_iti'] / unit_mean[unit_mean > 0]

    per_unit = (tr.groupby(['session', 'unit_id', 'depth'])['norm']
                  .mean().reset_index())
    n_per_unit = (tr.groupby(['session', 'unit_id', 'depth'])['norm']
                    .size().reset_index(name='n'))
    per_unit = per_unit.merge(n_per_unit).query('n >= @MIN_TRIALS')

    g = per_unit.groupby('depth')['norm']
    x = g.mean().index.values
    m = 100 * g.mean().values
    sem = 100 * (g.std() / np.sqrt(g.size())).values

    ax.axhline(100, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.fill_between(x, m - sem, m + sem, color=DOWN, alpha=0.18,
                    linewidth=0, zorder=3)
    ax.plot(x, m, color=DOWN, linewidth=2.0, marker='o', markersize=8,
            markerfacecolor=DOWN, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) if v < 5 else '5+' for v in x])
    ax.set_xlabel('consecutive unrewarded same-side choices', color=INK, fontsize=9)
    ax.set_ylabel('ITI rate (% of unit mean)', color=INK, fontsize=9)
    ax.set_title('C   Dose-response', color=INK, fontsize=10,
                 loc='left', fontweight='bold')
    style_axes(ax)
    ax.text(0.98, 0.95, f'n = {per_unit.groupby(["session","unit_id"]).ngroups} units',
            transform=ax.transAxes, color=INK_2, fontsize=8, ha='right', va='top')


def panel_windows(ax, stats_depth, n_boot=2000, seed=0):
    """D: effect per window -- highlight the ITI, gray the rest (emphasis).

    Every window reaches significance, so the claim is about size and
    consistency, not existence: the ITI effect is ~3x larger and shifts most of
    the population, while the response windows sit near 50% of units and are
    carried by magnitude in a minority. `frac` is printed for exactly that.
    """
    rng = np.random.default_rng(seed)
    order = ['iti', 'iti_post', 'resp_gocue', 'resp_choice', 'consumption']

    rows = []
    for win in order:
        d = stats_depth.query('window == @win').dropna(subset=['diff'])['diff'].values
        boot = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)]
        rows.append({'window': win, 'median': np.median(d),
                     'lo': np.percentile(boot, 2.5), 'hi': np.percentile(boot, 97.5),
                     'p': wilcoxon(d).pvalue, 'frac': (d < 0).mean()})
    r = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

    # xlim must cover every CI plus the zero line, or arms get clipped; the extra
    # room on the right is for the "% down   p = ..." annotations, which are wide
    # enough to sit on the whiskers otherwise
    span = [min(r['lo'].min(), 0.0), max(r['hi'].max(), 0.0)]
    pad = 0.18 * (span[1] - span[0])
    ax.set_xlim(span[0] - pad, span[1] + pad * 4.5)

    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)
    for i, row in r.iterrows():
        is_key = row['window'] == 'iti'
        color = DOWN if is_key else INK_MUTED
        ax.plot([row['lo'], row['hi']], [i, i], color=color,
                linewidth=2.0 if is_key else 1.4,
                alpha=1.0 if is_key else 0.55,
                solid_capstyle='round', zorder=3)
        ax.plot(row['median'], i, marker='o', markersize=9 if is_key else 7,
                color=color, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=2, alpha=1.0 if is_key else 0.55, zorder=4)
        stars = 'n.s.' if row['p'] > 0.05 else f'p = {row["p"]:.1g}'
        ax.text(0.985, i, f'{100*row["frac"]:.0f}% down   {stars}',
                transform=ax.get_yaxis_transform(),
                color=INK if is_key else INK_2, fontsize=7.5, ha='right',
                fontweight='bold' if is_key else 'normal', va='center')

    ax.set_yticks(range(len(r)))
    ax.set_yticklabels([WINDOW_LABELS[w] for w in r['window']])
    for lab, w in zip(ax.get_yticklabels(), r['window']):
        lab.set_color(INK if w == 'iti' else INK_2)
    ax.set_ylim(-0.6, len(r) - 0.4)
    ax.set_xlabel('$\\Delta$ rate, deep $-$ shallow (Hz)', color=INK, fontsize=9)
    ax.set_title('D   Largest and most consistent in the ITI', color=INK,
                 fontsize=10, loc='left', fontweight='bold')
    style_axes(ax, grid_axis='x')


# %%
if __name__ == '__main__':
    with open(os.path.join(OUT_DIR, 'persistence_trial_rates.pkl'), 'rb') as f:
        trial_rates = pickle.load(f)
    stats_depth = pd.read_csv(os.path.join(
        OUT_DIR, 'persistence_unit_stats_streak_depth.csv'))
    stats_iti = stats_depth.query('window == "iti"')

    fig = plt.figure(figsize=(10, 7.5), facecolor=SURFACE)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30,
                  left=0.085, right=0.965, top=0.90, bottom=0.09)

    panel_paired(fig.add_subplot(gs[0, 0]), stats_iti)
    panel_effect_dist(fig.add_subplot(gs[0, 1]), stats_iti)
    panel_dose(fig.add_subplot(gs[1, 0]), trial_rates)
    panel_windows(fig.add_subplot(gs[1, 1]), stats_depth)

    fig.suptitle('Tagged DRN units reduce tonic ITI firing during perseverative bouts',
                 color=INK, fontsize=12, fontweight='bold', x=0.085, ha='left',
                 y=0.965)
    fig.text(0.085, 0.928,
             'streak-depth matched: both groups follow an unrewarded same-side '
             'choice, so the previous trial is held fixed',
             color=INK_2, fontsize=8.5, ha='left')

    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'persistence_iti_effect.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f'wrote {path}')

    # table view, so nothing depends on reading colour off the figure
    # frac_decrease must divide by units that HAVE a diff, not by every selected
    # unit -- units below the trial floor have a NaN diff and are not evidence
    # either way
    tbl = (stats_depth.dropna(subset=['diff']).groupby('window')
           .agg(n_units=('diff', 'count'),
                median_diff=('diff', 'median'),
                frac_decrease=('diff', lambda s: (s < 0).mean()))
           .reindex(['iti', 'iti_post', 'resp_gocue', 'resp_choice', 'consumption']))
    tbl['wilcoxon_p'] = [wilcoxon(stats_depth.query('window == @w')
                                  .dropna(subset=['diff'])['diff']).pvalue
                         for w in tbl.index]
    tbl.to_csv(os.path.join(OUT_DIR, 'persistence_iti_effect_table.csv'))
    print(tbl.to_string())
