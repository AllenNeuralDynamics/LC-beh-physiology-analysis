"""Overview of the behavior sessions behind the DRN ephys analyses.

Many units lose `T_baseline_hit_all` (and other hit-vs-ignore regressors) because
response_tstats_generation.py drops `hit` from the all-trial formula when a unit's
trial window holds < 2 ignore trials, and skips the unit entirely below 20 trials or
20 hits. That window is get_session_tbl(cut_interruptions=True) -- the longest run of
trials with no auto/manual or extra water -- further trimmed by the unit's drift
ephys_cut. This script tallies trials per session before and after those cuts, and
per unit inside its own window, so the losses can be traced to sessions and mice.

Writes a summary figure plus session- and unit-level CSVs.

Run:  python session_combine/figure_preparation/behavior_session_summary.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append('/root/capsule/code/beh_ephys_analysis')
warnings.filterwarnings('ignore')

from utils.beh_functions import get_session_tbl, longest_zero_chunk, session_dirs
from utils.capsule_migration import capsule_directories
from utils.combine_tools import to_str_intlike
from utils.ephys_functions import load_drift

criteria_name = 'beh_all_DRN'
data_type = 'raw'
# thresholds used by response_tstats_generation.process()
MIN_TRIALS = 20
MIN_IGNORE = 2

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']), 'SUMMARY_behavior_sessions')
os.makedirs(OUT_DIR, exist_ok=True)

# reference categorical palette, slots 1-3, plus neutral for "missing"
C_LEFT, C_IGNORE, C_RIGHT = '#2a78d6', '#eb6834', '#1baf7a'
C_GRAY = '#b5b4ae'
C_INK, C_INK2 = '#0b0b0b', '#52514e'


# ---------------------------------------------------------------------------
# per-session tallies
# ---------------------------------------------------------------------------
def count_responses(resp):
    resp = np.asarray(resp)
    return {'left': int(np.sum(resp == 0)), 'right': int(np.sum(resp == 1)),
            'ignore': int(np.sum(resp == 2))}


def summarize_session(session):
    nwb = os.path.join(session_dirs(session)['beh_fig_dir'], f'{session}.nwb')
    if not os.path.exists(nwb):
        return None
    try:
        tbl = get_session_tbl(session)  # uncut; the cut below mirrors cut_interruptions=True
    except Exception as e:
        print(f'{session}: failed to load ({e!r})')
        return None
    interrupt = (tbl['auto_manual_trial'] | tbl['extra_reward']).to_numpy()
    start, end, _ = longest_zero_chunk(interrupt.tolist())
    cut = tbl.iloc[start:end + 1]

    full = count_responses(tbl['animal_response'])
    kept = count_responses(cut['animal_response'])
    rewarded = (tbl['rewarded_historyL'] | tbl['rewarded_historyR']).to_numpy()
    row = {
        'session': session,
        'mouse': session.split('_')[1],
        'n_trials': len(tbl),
        'n_left': full['left'], 'n_right': full['right'], 'n_ignore': full['ignore'],
        'n_responded': full['left'] + full['right'],
        'n_rewarded': int(rewarded.sum()),
        'n_auto_manual': int(tbl['auto_manual_trial'].sum()),
        'n_extra_reward': int(tbl['extra_reward'].sum()),
        'n_interruptions': int(interrupt.sum()),
        'cut_start': start, 'cut_end': end,
        'n_trials_cut': len(cut),
        'n_left_cut': kept['left'], 'n_right_cut': kept['right'], 'n_ignore_cut': kept['ignore'],
        'n_responded_cut': kept['left'] + kept['right'],
        'n_rewarded_cut': int(rewarded[start:end + 1].sum()),
    }
    # trial times/responses of the cut window, for the per-unit drift windows
    trials = pd.DataFrame({'session': session,
                           'goCue_start_time': cut['goCue_start_time'].to_numpy(),
                           'animal_response': cut['animal_response'].to_numpy()})
    return row, trials


# ---------------------------------------------------------------------------
# per-unit trial windows (mirrors response_tstats_generation.process)
# ---------------------------------------------------------------------------
def unit_window(session, unit, trials):
    t = trials['goCue_start_time'].to_numpy()
    resp = trials['animal_response'].to_numpy()
    keep = np.ones(len(t), bool)
    drift = load_drift(session, int(unit) if str(unit).isdigit() else unit, data_type=data_type)
    if drift is not None and drift['ephys_cut'][0] is not None:
        keep &= t >= drift['ephys_cut'][0]
    if drift is not None and drift['ephys_cut'][1] is not None:
        keep &= t <= drift['ephys_cut'][1]
    n = int(keep.sum())
    n_ignore = int(np.sum(resp[keep] == 2))
    return n, n - n_ignore, n_ignore


def unit_reason(n, n_hit, n_ignore):
    if n < MIN_TRIALS or n_hit < MIN_TRIALS:
        return f'< {MIN_TRIALS} trials/hits'
    if n_ignore < MIN_IGNORE:
        return f'< {MIN_IGNORE} ignores'
    return 'fit'


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(C_INK2)
    ax.tick_params(colors=C_INK2, labelsize=8)
    ax.yaxis.grid(True, color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc='left', fontsize=10, color=C_INK)


def stacked_counts(ax, index, parts, colors, labels, width=0.8):
    bottom = np.zeros(len(index))
    for vals, c, lab in zip(parts, colors, labels):
        ax.bar(index, vals, width, bottom=bottom, color=c, edgecolor='white', lw=1, label=lab)
        bottom += vals
    return bottom


def make_figure(sess, units, path_no_ext):
    sess = sess.sort_values(['mouse', 'session']).reset_index(drop=True)
    mice = list(dict.fromkeys(sess['mouse']))

    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.15, 1], hspace=0.55, wspace=0.35)

    # A: sessions per mouse
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    per_mouse = sess.groupby('mouse').agg(n=('session', 'size'), drn=('has_drn_units', 'sum')).reindex(mice)
    x = np.arange(len(mice))
    stacked_counts(ax, x, [per_mouse['drn'], per_mouse['n'] - per_mouse['drn']],
                   [C_LEFT, C_GRAY], [f'with {criteria_name} units', 'no units'])
    ax.set_xticks(x, mice, rotation=90, fontsize=7)
    ax.set_ylabel('sessions')
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylim(0, per_mouse['n'].max() + 1.5)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper left')
    title(ax, f'A  Behavior sessions per mouse (n={len(sess)}, {len(mice)} mice)')

    # B: units per mouse, T_baseline_hit_all available or not
    ax = fig.add_subplot(gs[0, 1]); style(ax)
    um = units.groupby('mouse').agg(n=('unit', 'size'), ok=('T_valid', 'sum'))
    um = um.reindex([m for m in mice if m in um.index])
    x = np.arange(len(um))
    stacked_counts(ax, x, [um['ok'], um['n'] - um['ok']], [C_LEFT, C_GRAY],
                   ['T_baseline_hit_all fit', 'NaN'])
    ax.set_xticks(x, um.index, rotation=90, fontsize=7)
    ax.set_ylabel('units')
    ax.legend(fontsize=7, frameon=False)
    title(ax, f'B  {criteria_name} units per mouse ({int(units["T_valid"].sum())}/{len(units)} fit)')

    # C: why units are NaN
    ax = fig.add_subplot(gs[0, 2]); style(ax)
    order = [f'< {MIN_TRIALS} trials/hits', f'< {MIN_IGNORE} ignores', 'fit']
    counts = units['reason'].value_counts().reindex(order, fill_value=0)
    bars = ax.bar(order, counts.values, 0.6, color=[C_GRAY, C_IGNORE, C_LEFT])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha='center', va='bottom', fontsize=8, color=C_INK)
    ax.set_ylabel('units')
    ax.tick_params(axis='x', labelsize=8)
    title(ax, 'C  Unit trial window vs regression gates')

    # D: units per session vs. ignore trials in the cut window
    ax = fig.add_subplot(gs[0, 3]); style(ax)
    us = units.groupby('session').agg(n=('unit', 'size'), ok=('T_valid', 'sum'))
    s_ = sess.set_index('session').loc[us.index]
    ax.scatter(s_['n_ignore_cut'] + 0.5, us['n'], s=40, c=np.where(us['ok'] > 0, C_LEFT, C_GRAY),
               edgecolor='white', lw=1)
    ax.axvline(MIN_IGNORE + 0.5, color=C_IGNORE, lw=1, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('ignore trials after interruption cut (+0.5, log)')
    ax.set_ylabel(f'{criteria_name} units in session')
    ax.text(MIN_IGNORE + 0.7, 0.97, f'{MIN_IGNORE} ignores', color=C_IGNORE, fontsize=7, va='top',
            transform=ax.get_xaxis_transform())
    title(ax, 'D  Sessions with units (blue: any unit fit)')

    # E: per-session trial outcomes in the interruption-free window; outline = full session
    ax = fig.add_subplot(gs[1, :]); style(ax)
    x = np.arange(len(sess))
    ax.bar(x, sess['n_trials'], 0.8, facecolor='none', edgecolor=C_INK2, lw=0.8, label='full session')
    stacked_counts(ax, x, [sess['n_left_cut'], sess['n_right_cut'], sess['n_ignore_cut']],
                   [C_LEFT, C_RIGHT, C_IGNORE], ['left (kept)', 'right (kept)', 'ignore (kept)'])
    bounds = np.flatnonzero(sess['mouse'].to_numpy()[1:] != sess['mouse'].to_numpy()[:-1]) + 0.5
    for b in bounds:
        ax.axvline(b, color='#d0cfca', lw=0.8)
    starts = np.r_[0, np.ceil(bounds)].astype(int)
    ends = np.r_[np.floor(bounds), len(sess) - 1]
    for k, (s0, s1) in enumerate(zip(starts, ends)):
        ax.text((s0 + s1) / 2, -0.04 - 0.05 * (k % 2), sess['mouse'].iloc[s0], transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color=C_INK2)
    for i in np.flatnonzero(sess['has_drn_units'].to_numpy()):
        ax.plot(i, sess['n_trials'].iloc[i] + 15, marker='v', ms=4, color=C_INK, lw=0)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(sess) - 0.4)
    ax.set_ylabel('trials')
    ax.legend(ncol=4, fontsize=8, frameon=False, loc='upper left')
    title(ax, 'E  Trials per session, interruption-free window (fill) vs full session (outline); '
              f'▼ = has {criteria_name} units')

    # F: ignore trials, full session vs after the interruption cut
    ax = fig.add_subplot(gs[2, 0]); style(ax)
    m = sess['has_drn_units'].to_numpy()
    for msk, c, lab in [(~m, C_GRAY, 'no units'), (m, C_LEFT, 'with units')]:
        ax.scatter(sess['n_ignore'][msk] + 0.5, sess['n_ignore_cut'][msk] + 0.5, s=30, c=c,
                   edgecolor='white', lw=1, label=lab)
    lim = [0.4, max(sess['n_ignore'].max(), 1) * 1.5]
    ax.plot(lim, lim, color=C_INK2, lw=0.8, ls=':')
    ax.axhline(MIN_IGNORE + 0.5, color=C_IGNORE, lw=1, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('ignore trials, full session (+0.5)')
    ax.set_ylabel('ignore trials, after cut (+0.5)')
    ax.legend(fontsize=7, frameon=False)
    title(ax, 'F  Ignore trials lost to the interruption cut')

    # G: fraction of trials kept by the interruption cut
    ax = fig.add_subplot(gs[2, 1]); style(ax)
    frac = sess['n_trials_cut'] / sess['n_trials']
    bins = np.linspace(0, 1, 21)
    ax.hist([frac[~m], frac[m]], bins=bins, stacked=True, color=[C_GRAY, C_LEFT],
            edgecolor='white', lw=1, label=['no units', 'with units'])
    ax.set_xlabel('fraction of trials in interruption-free window')
    ax.set_ylabel('sessions')
    ax.legend(fontsize=7, frameon=False)
    title(ax, f'G  Trials kept (median {np.median(frac):.2f})')

    # H: ignore rate per session
    ax = fig.add_subplot(gs[2, 2]); style(ax)
    rate = sess['n_ignore'] / sess['n_trials']
    bins = np.linspace(0, max(rate.max(), 0.05), 21)
    ax.hist([rate[~m], rate[m]], bins=bins, stacked=True, color=[C_GRAY, C_LEFT],
            edgecolor='white', lw=1, label=['no units', 'with units'])
    ax.set_xlabel('ignore rate, full session')
    ax.set_ylabel('sessions')
    ax.legend(fontsize=7, frameon=False)
    title(ax, f'H  Ignore rate (median {np.median(rate):.3f})')

    # I: ignore trials inside each unit's own window
    ax = fig.add_subplot(gs[2, 3]); style(ax)
    rng = np.random.default_rng(0)
    for k, (ok, c, lab) in enumerate([(False, C_GRAY, 'NaN'), (True, C_LEFT, 'fit')]):
        v = units.loc[units['T_valid'] == ok, 'n_ignore_unit'].to_numpy() + 0.5
        ax.scatter(k + rng.uniform(-0.2, 0.2, len(v)), v, s=14, c=c, edgecolor='white', lw=0.6)
    ax.axhline(MIN_IGNORE + 0.5, color=C_IGNORE, lw=1, ls='--')
    ax.set_yscale('log')
    ax.set_xticks([0, 1], ['T NaN', 'T fit'])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel('ignore trials in unit window (+0.5)')
    title(ax, 'I  Per-unit ignore trials')

    fig.suptitle(f'Behavior session summary ({criteria_name}; trial windows as in response_tstats_generation)',
                 x=0.01, y=0.93, ha='left', fontsize=12, color=C_INK)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_no_ext}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    assets = pd.read_csv('/root/capsule/code/data_management/session_assets.csv', dtype={'probe': str})
    sessions = [s for s in assets['session_id'] if isinstance(s, str)]
    results = Parallel(n_jobs=8)(delayed(summarize_session)(s) for s in sessions)
    results = [r for r in results if r is not None]
    sess = pd.DataFrame([r[0] for r in results])
    trials = pd.concat([r[1] for r in results], ignore_index=True)

    resp = pd.read_csv(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'response_regression',
                                    f'response_ratio_{criteria_name}_go_cue.csv'))
    units = resp[['session', 'unit', 'T_baseline_hit_all']].copy()
    units['unit'] = units['unit'].apply(to_str_intlike)
    units['mouse'] = units['session'].str.split('_').str[1]
    units['T_valid'] = units['T_baseline_hit_all'].notna()
    win = []
    for s, u in zip(units['session'], units['unit']):
        tr = trials[trials['session'] == s]
        win.append(unit_window(s, u, tr) if len(tr) else (np.nan, np.nan, np.nan))
    units[['n_trials_unit', 'n_hit_unit', 'n_ignore_unit']] = np.array(win, float)
    units['reason'] = [unit_reason(*w) if np.isfinite(w[0]) else 'no behavior' for w in win]

    us = units.groupby('session').agg(n_units=('unit', 'size'), n_units_T_valid=('T_valid', 'sum'))
    sess = sess.merge(us, left_on='session', right_index=True, how='left')
    sess[['n_units', 'n_units_T_valid']] = sess[['n_units', 'n_units_T_valid']].fillna(0).astype(int)
    sess['has_drn_units'] = sess['n_units'] > 0

    sess.to_csv(os.path.join(OUT_DIR, 'behavior_session_summary.csv'), index=False)
    units.to_csv(os.path.join(OUT_DIR, f'behavior_unit_windows_{criteria_name}.csv'), index=False)
    make_figure(sess, units, os.path.join(OUT_DIR, 'behavior_session_summary'))

    # console digest
    mismatch = units[(units['reason'] == 'fit') != units['T_valid']]
    print(f'{len(sess)} sessions, {sess["mouse"].nunique()} mice; '
          f'{int(sess["has_drn_units"].sum())} sessions with {criteria_name} units')
    print(sess[['n_trials', 'n_responded', 'n_ignore', 'n_trials_cut', 'n_responded_cut',
                'n_ignore_cut', 'n_interruptions']].describe().round(1).to_string())
    print(units['reason'].value_counts().to_string())
    print(f'units where predicted gate != observed T_baseline_hit_all: {len(mismatch)}')
    if len(mismatch):
        print(mismatch.to_string())
    print(f'saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
