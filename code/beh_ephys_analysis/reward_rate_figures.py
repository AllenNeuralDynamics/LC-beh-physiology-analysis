"""
Figures: does tonic DRN firing track experienced reward rate or the hidden schedule?

Six panels, one story: tonic ITI rate follows reward the animal has actually
collected, the hidden block probabilities add nothing once that is accounted for,
and the reward-history signal has a genuinely slow component on top of a larger
fast one -- but all of it is small next to session drift.

  A  timescale spectrum, raw vs drift-controlled
  B  phasic carryover vs a genuinely tonic component
  C  the dissociation: experienced reward rate vs latent probability
  D  block-transition aligned tonic profile, drift-removed
  E  variance partition -- how little of the ITI rate any of this explains
  F  per-unit distribution of the reward-rate effect

Colour follows the same reference palette instance as persistence_figures.py:
diverging blue<->red for polarity, single blue for magnitude, gray for the
uncontrolled comparison series. Values used verbatim.

Reads what reward_rate_tonic_analysis.py wrote.
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
from utils.reward_rate import TAUS

# ---- palette (same reference instance, used verbatim) ----
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8983'
DOWN = '#2a78d6'
UP = '#e34948'
MID = '#f0efec'
GRID = '#e6e5e1'

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'persistence_5HT')


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


def boot_ci(x, n_boot=2000, seed=0):
    """Median plus a bootstrap 95% CI on the median."""
    rng = np.random.default_rng(seed)
    x = np.asarray(pd.Series(x).dropna())
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    b = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5)


def stars(p):
    return 'n.s.' if p > 0.05 else f'p={p:.0e}'.replace('e-0', 'e-')


# %%
def panel_timescale(ax, u):
    """A: median partial rho vs tau, raw and with session drift removed.

    Two series, so both are direct-labelled and a legend is present. Gray is the
    uncontrolled version: drift pulls the long taus down, which is why the raw
    curve falls off far faster than the controlled one.
    """
    x = np.arange(len(TAUS))
    ends = {}
    for pref, color, lab in (('raw', INK_MUTED, 'raw'),
                             ('dpart', DOWN, 'drift removed')):
        m, lo, hi = zip(*[boot_ci(u[f'{pref}_rr_{t}']) for t in TAUS])
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, m, color=color, linewidth=2.0, marker='o', markersize=8,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2,
                label=lab, zorder=3)
        ends[pref] = (m[-1], color, lab)

    # the two series converge at long tau, so direct labels at the shared
    # endpoint overlap; nudge them apart when the gap is too small to separate
    span = max(max(m for m, _c, _l in ends.values()), 0.0)
    (m_raw, c_raw, l_raw), (m_dp, c_dp, l_dp) = ends['raw'], ends['dpart']
    gap = 0.07 * span
    if abs(m_dp - m_raw) < gap:
        mid = 0.5 * (m_dp + m_raw)
        m_dp, m_raw = mid + gap / 2, mid - gap / 2
    ax.text(x[-1] + 0.12, m_raw, l_raw, color=c_raw, fontsize=8, va='center')
    ax.text(x[-1] + 0.12, m_dp, l_dp, color=c_dp, fontsize=8, va='center',
            fontweight='bold')

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in TAUS])
    ax.set_xlim(-0.35, len(TAUS) - 0.35 + 1.15)
    ax.set_xlabel('reward-rate integration $\\tau$ (trials)', color=INK, fontsize=9)
    ax.set_ylabel('median partial $\\rho$ with ITI rate', color=INK, fontsize=9)
    ax.set_title('A   Every timescale, strongest when fast', color=INK,
                 fontsize=10, loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='lower left',
              labelcolor=INK_2, handlelength=1.4)
    style_axes(ax)


def panel_carryover(ax, u):
    """B: is the reward signal phasic carryover or genuinely tonic?"""
    rows = [('rew_lag1', 'reward $n\\!-\\!1$', False),
            ('rew_lag2', 'reward $n\\!-\\!2$', False),
            ('rew_lag3', 'reward $n\\!-\\!3$', False),
            ('rr_proximal', 'mean $n\\!-\\!3..n\\!-\\!1$', False),
            ('rr_distal', 'mean $n\\!-\\!20..n\\!-\\!6$', True),
            ('rr_distal|lags', 'minus lags 1-3', True)]
    r = []
    for key, lab, key_row in rows:
        v = u[f'carry_{key}'].dropna()
        m, lo, hi = boot_ci(v)
        r.append({'lab': lab, 'm': m, 'lo': lo, 'hi': hi,
                  'p': wilcoxon(v).pvalue, 'key': key_row})
    r = pd.DataFrame(r).iloc[::-1].reset_index(drop=True)

    span = [min(r['lo'].min(), 0.0), max(r['hi'].max(), 0.0)]
    pad = 0.2 * (span[1] - span[0])
    ax.set_xlim(span[0] - pad, span[1] + pad * 2.4)
    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)

    for i, row in r.iterrows():
        color = DOWN if row['key'] else INK_MUTED
        alpha = 1.0 if row['key'] else 0.55
        ax.plot([row['lo'], row['hi']], [i, i], color=color, alpha=alpha,
                linewidth=2.0 if row['key'] else 1.4, solid_capstyle='round',
                zorder=3)
        ax.plot(row['m'], i, marker='o', markersize=9 if row['key'] else 7,
                color=color, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=2, alpha=alpha, zorder=4)
        ax.text(0.985, i, stars(row['p']), transform=ax.get_yaxis_transform(),
                color=INK if row['key'] else INK_2, fontsize=7,
                ha='right', va='center',
                fontweight='bold' if row['key'] else 'normal')

    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r['lab'])
    for lab, k in zip(ax.get_yticklabels(), r['key']):
        lab.set_color(INK if k else INK_2)
    ax.set_ylim(-0.6, len(r) - 0.4)
    ax.set_xlabel('partial $\\rho$ with ITI rate  (drift removed)',
                  color=INK, fontsize=9)
    ax.set_title('B   A slow component survives', color=INK, fontsize=10,
                 loc='left', fontweight='bold')
    style_axes(ax, grid_axis='x')


def panel_dissociation(ax, part):
    """C: experienced reward rate vs the latent schedule, each with the other out.

    The two blue rows are the same predictor, reward rate, before and after
    removing the schedule; the two gray rows are the schedule before and after
    removing reward rate. Only one of the pairs survives.
    """
    rows = [('rr_10|trial', 'reward rate', True),
            ('rr_10|p_sum,trial', 'minus schedule', True),
            ('p_sum|trial', 'hidden $p_L\\!+\\!p_R$', False),
            ('p_sum|rr_10,trial', 'minus reward rate', False),
            ('p_prev_chosen|choice,rr,trial', 'hidden $p$, chosen*', False)]
    r = []
    for key, lab, is_rr in rows:
        v = part[key].dropna()
        m, lo, hi = boot_ci(v)
        r.append({'lab': lab, 'm': m, 'lo': lo, 'hi': hi,
                  'p': wilcoxon(v).pvalue, 'is_rr': is_rr,
                  'sig': wilcoxon(v).pvalue < 0.05})
    r = pd.DataFrame(r).iloc[::-1].reset_index(drop=True)

    span = [min(r['lo'].min(), 0.0), max(r['hi'].max(), 0.0)]
    pad = 0.2 * (span[1] - span[0])
    ax.set_xlim(span[0] - pad, span[1] + pad * 2.6)
    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)

    for i, row in r.iterrows():
        color = DOWN if row['is_rr'] else INK_MUTED
        alpha = 1.0 if row['sig'] else 0.4
        ax.plot([row['lo'], row['hi']], [i, i], color=color, alpha=alpha,
                linewidth=2.0, solid_capstyle='round', zorder=3)
        ax.plot(row['m'], i, marker='o', markersize=9, color=color,
                markerfacecolor=color if row['sig'] else SURFACE,
                markeredgecolor=color if row['sig'] else color,
                markeredgewidth=2, alpha=1.0 if row['sig'] else 0.6, zorder=4)
        ax.text(0.985, i, stars(row['p']), transform=ax.get_yaxis_transform(),
                color=INK if row['sig'] else INK_2, fontsize=7, ha='right',
                va='center', fontweight='bold' if row['sig'] else 'normal')

    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r['lab'])
    ax.set_ylim(-0.6, len(r) - 0.4)
    ax.set_xlabel('partial $\\rho$ with ITI rate', color=INK, fontsize=9)
    ax.set_title('C   Reward, not the schedule', color=INK,
                 fontsize=10, loc='left', fontweight='bold')
    style_axes(ax, grid_axis='x')
    ax.text(0.0, -0.22, '*side and reward rate both removed; open marker n.s.',
            transform=ax.transAxes, color=INK_MUTED, fontsize=7, va='top')


def panel_transition(ax, trans, bin_w=2):
    """D: tonic rate aligned to block transitions, after removing drift.

    Drift is removed first: the post window sits ~11 trials after the pre window,
    and tonic rate falls across a session, so a raw alignment dips after any
    event whatsoever. Lags are binned in pairs -- at single-trial resolution the
    two traces are pure wobble and nothing is legible.

    Zero on this axis is the local rolling median, and mean-minus-median is
    positive for a skewed rate distribution, so the offset carries no meaning;
    only the difference between the two traces does.
    """
    trans = trans.copy()
    trans['lag_b'] = (np.floor(trans['lag'] / bin_w) * bin_w + (bin_w - 1) / 2)
    for d, color in (('richer', UP), ('poorer', DOWN)):
        q = (trans[trans['dir'] == d].groupby('lag_b')['dt']
             .agg(['mean', 'sem']).reset_index().sort_values('lag_b'))
        x = q['lag_b'].values
        m, s = 100 * q['mean'].values, 100 * q['sem'].values
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.16, linewidth=0,
                        zorder=2)
        ax.plot(x, m, color=color, linewidth=2.0, zorder=3, label=d)
        ax.text(x[-1] + 0.5, m[-2:].mean(), d, color=color, fontsize=8,
                va='center', fontweight='bold')

    # only the divergence between the two traces is a result; the pre-transition
    # offset and the trial-to-trial wobble are not, so the note says which is which
    per_unit = {}
    for d in ('richer', 'poorer'):
        sub = trans[trans['dir'] == d]
        per_unit[d] = {
            k: (g[g['lag'].between(3, 12)]['dt'].mean()
                - g[g['lag'].between(-6, -1)]['dt'].mean())
            for k, g in sub.groupby(['session', 'unit_id'])}
    shared = sorted(set(per_unit['richer']) & set(per_unit['poorer']))
    diff = pd.Series([per_unit['richer'][k] - per_unit['poorer'][k]
                      for k in shared]).dropna()
    each = {d: pd.Series(list(v.values())).dropna() for d, v in per_unit.items()}
    # headroom first, so the note cannot land on a trace or the lag-0 rule
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.46 * (hi - lo))
    note = (f'richer $-$ poorer, post $-$ pre: {100*diff.median():+.2f}% per unit, '
            f'p = {wilcoxon(diff).pvalue:.3f}\n'
            f'carried by the drop after poorer (p = '
            f'{wilcoxon(each["poorer"]).pvalue:.3f}); richer alone '
            f'p = {wilcoxon(each["richer"]).pvalue:.2f}')
    # a surface-coloured box, so the lag-0 rule cannot read through the text
    ax.text(0.02, 0.97, note, transform=ax.transAxes, color=INK_2, fontsize=7,
            va='top', zorder=6,
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1.5))

    ax.axvline(0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(trans['lag'].min() - 0.5, trans['lag'].max() + 4.5)
    ax.set_xlabel('trials from block transition', color=INK, fontsize=9)
    ax.set_ylabel('ITI rate vs local baseline\n(% of unit mean)', color=INK,
                  fontsize=9)
    ax.set_title('D   Block transitions: weak and asymmetric', color=INK,
                 fontsize=10, loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5, loc='lower left', labelcolor=INK_2,
              handlelength=1.4)
    style_axes(ax)


def panel_variance(ax, u):
    """E: median unique variance explained, per predictor.

    The honest scale check. Single measure, single hue; the one bar that is not
    about reward is grayed so the comparison is immediate.
    """
    rows = [('trial_ind', 'session drift', False),
            ('rr_10', 'reward rate', True),
            ('lick_rate_iti', 'ITI licking', False),
            ('p_sum', 'hidden $p_L+p_R$', True),
            ('p_contrast', 'hidden $|p_R-p_L|$', True),
            ('unrew_streak_before', 'streak depth', True)]
    r = pd.DataFrame([{'lab': lab, 'v': 100 * u[f'dr2_{k}'].median(), 'key': key}
                      for k, lab, key in rows]).sort_values('v')

    ax.barh(range(len(r)), r['v'],
            color=[DOWN if k else INK_MUTED for k in r['key']],
            edgecolor=SURFACE, linewidth=1.0, height=0.72, zorder=3)
    for i, (v, k) in enumerate(zip(r['v'], r['key'])):
        ax.text(v + 0.35, i, f'{v:.2f}%', va='center', fontsize=7.5,
                color=INK if k else INK_2)

    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r['lab'])
    ax.set_xlim(0, r['v'].max() * 1.28)
    ax.set_xlabel('median unique variance explained ($\\Delta R^2$, %)',
                  color=INK, fontsize=9)
    ax.set_title('E   All of it is small next to drift', color=INK, fontsize=10,
                 loc='left', fontweight='bold')
    style_axes(ax, grid_axis='x')


def panel_dist(ax, part):
    """F: per-unit distribution of the reward-rate effect, diverging by sign."""
    d = part['rr_10|trial'].dropna().values
    lim = np.nanmax(np.abs(d)) * 1.02
    bins = np.linspace(-lim, lim, 33)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(d, bins=bins)
    for c, x, w in zip(counts, centers, np.diff(bins)):
        if c:
            ax.bar(x, c, width=w * 0.88, color=(DOWN if x < 0 else UP),
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)

    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)
    med = np.median(d)
    ax.set_ylim(0, counts.max() * 1.32)
    ax.axvline(med, color=INK, linewidth=2.0, zorder=4)
    ax.annotate(f'median {med:+.3f}', xy=(med, counts.max() * 1.12),
                xytext=(8, 0), textcoords='offset points', color=INK,
                fontsize=8, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=INK, linewidth=0.8))
    ax.set_xlabel('per-unit partial $\\rho$, reward rate vs ITI rate',
                  color=INK, fontsize=9)
    ax.set_ylabel('units', color=INK, fontsize=9)
    ax.set_title('F   A population-wide shift', color=INK, fontsize=10,
                 loc='left', fontweight='bold')
    style_axes(ax)
    ax.text(0.02, 0.97, f'{100*(d>0).mean():.0f}% of {len(d)} units positive',
            transform=ax.transAxes, color=INK_2, fontsize=8, va='top')


# %%
if __name__ == '__main__':
    u = pd.read_csv(os.path.join(OUT_DIR, 'reward_rate_unit_stats.csv'))
    part = pd.read_csv(os.path.join(OUT_DIR, 'reward_rate_partial_rho.csv'))
    trans = pd.read_csv(os.path.join(OUT_DIR, 'reward_rate_transitions.csv'))

    fig = plt.figure(figsize=(14.5, 7.8), facecolor=SURFACE)
    gs = GridSpec(2, 3, figure=fig, hspace=0.58, wspace=0.60,
                  left=0.062, right=0.955, top=0.885, bottom=0.10)

    panel_timescale(fig.add_subplot(gs[0, 0]), u)
    panel_carryover(fig.add_subplot(gs[0, 1]), u)
    panel_dissociation(fig.add_subplot(gs[0, 2]), part)
    panel_transition(fig.add_subplot(gs[1, 0]), trans)
    panel_variance(fig.add_subplot(gs[1, 1]), u)
    panel_dist(fig.add_subplot(gs[1, 2]), part)

    fig.suptitle('Tonic DRN firing tracks experienced reward rate, not the hidden '
                 'reward schedule', color=INK, fontsize=12, fontweight='bold',
                 x=0.065, ha='left', y=0.968)
    fig.text(0.065, 0.928,
             f'{len(part)} opto-tagged units, {part["session"].nunique()} sessions; '
             'ITI window only (pre-cue, pre-choice, 92% lick-free); trials outside '
             'the curated drift window dropped; session drift removed throughout',
             color=INK_2, fontsize=8.5, ha='left')

    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'reward_rate_tonic.{ext}')
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f'wrote {path}')

    # table view, so nothing depends on reading colour off the figure
    tbl = []
    for c in ['rr_10|trial', 'rr_10|p_sum,trial', 'p_sum|trial',
              'p_sum|rr_10,trial', 'p_prev_chosen|choice,rr,trial']:
        v = part[c].dropna()
        m, lo, hi = boot_ci(v)
        tbl.append({'quantity': c, 'n_units': len(v), 'median_rho': m,
                    'ci_lo': lo, 'ci_hi': hi, 'frac_pos': (v > 0).mean(),
                    'wilcoxon_p': wilcoxon(v).pvalue})
    for c in [f'dpart_rr_{t}' for t in TAUS] + \
             ['carry_rr_distal', 'carry_rr_distal|lags']:
        v = u[c].dropna()
        m, lo, hi = boot_ci(v)
        tbl.append({'quantity': c, 'n_units': len(v), 'median_rho': m,
                    'ci_lo': lo, 'ci_hi': hi, 'frac_pos': (v > 0).mean(),
                    'wilcoxon_p': wilcoxon(v).pvalue})
    tbl = pd.DataFrame(tbl)
    tbl.to_csv(os.path.join(OUT_DIR, 'reward_rate_tonic_table.csv'), index=False)
    print(tbl.to_string(index=False))
