"""
Two checks that decide how much the reward-rate and perseveration results are worth.

1. SESSION-LEVEL INFERENCE. Every headline so far is a Wilcoxon across units, but
   units recorded in the same session see the same reward-rate time series, the
   same block schedule and the same drift, so their per-unit rhos are not
   independent draws. n=182 units is really n=39 sessions. If an effect survives
   collapsing to one number per session, it is a population effect; if it does
   not, the unit-level p-value was pseudo-replication.

2. IS THE STREAK EFFECT PERSEVERATION OR JUST RECENT OMISSIONS? A deep unrewarded
   same-side bout is, by construction, a run of omissions, so low tonic rate could
   simply be low recent reward. Regressing out rr_10 is a weak control because
   tau=10 is slow next to a 3-trial bout. The design-based control: hold the last
   three outcomes fixed at three omissions and vary only whether the animal
   repeated the same side or switched. Both groups then have an identical recent
   reward history -- identical by construction, not by covariate adjustment -- and
   differ only in whether the behaviour was perseverative.
"""
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
import pickle
from scipy.stats import wilcoxon, spearmanr

from utils.capsule_migration import capsule_directories

DEEP, SHALLOW = 3, 1
MIN_TRIALS = 5
K = 3                      # window of preceding trials to match on

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'persistence_5HT')


def report(name, x, unit='rho'):
    """Wilcoxon signed-rank on a per-something effect, printed one line."""
    x = pd.Series(x).dropna()
    if len(x) < 3:
        print(f'   {name:44s} n={len(x):3d}  (too few)')
        return np.nan
    p = wilcoxon(x).pvalue
    print(f'   {name:44s} n={len(x):3d}  median={x.median():+.4f} {unit}  '
          f'neg={int((x < 0).sum())}/{len(x)}  p={p:.3g}')
    return p


# %%
def session_level(part, u, stats_depth):
    """Collapse each effect to one value per session, then test across sessions."""
    print('=' * 78)
    print('1. SESSION-LEVEL INFERENCE (one median per session, then Wilcoxon)')
    print('=' * 78)

    print('\n reward rate and the schedule (partial rho, ITI rate):')
    for col in ['rr_10|trial', 'rr_10|p_sum,trial', 'p_sum|trial',
                'p_sum|rr_10,trial', 'streak|trial', 'streak|rr_10,trial']:
        by_sess = part.groupby('session')[col].median()
        report(col, by_sess)

    print('\n timescale sweep, drift removed:')
    for c in ['dpart_rr_2', 'dpart_rr_5', 'dpart_rr_10', 'dpart_rr_20',
              'dpart_rr_40']:
        report(c, u.groupby('session')[c].median())

    print('\n slow component beyond phasic carryover:')
    for c in ['carry_rr_distal', 'carry_rr_distal|lags']:
        report(c, u.groupby('session')[c].median())

    print('\n perseveration, deep vs shallow streak (Hz):')
    for win in ['iti', 'iti_post', 'resp_gocue', 'resp_choice', 'consumption']:
        g = stats_depth.query('window == @win').dropna(subset=['diff'])
        report(win, g.groupby('session')['diff'].median(), unit='Hz')

    # how much clustering is actually there? if units in a session agreed
    # perfectly the effective n would be the session count exactly
    print('\n within-session agreement (do units in a session share their effect?):')
    for lab, d, col in [('rr_10|trial', part, 'rr_10|trial'),
                        ('streak depth ITI diff',
                         stats_depth.query('window == "iti"'), 'diff')]:
        d = d.dropna(subset=[col])
        n_per = d.groupby('session')[col].size()
        multi = d[d['session'].isin(n_per[n_per >= 3].index)]
        within = multi.groupby('session')[col].std().median()
        between = multi.groupby('session')[col].median().std()
        print(f'   {lab:30s} median within-session SD={within:.4f}  '
              f'between-session SD of medians={between:.4f}  '
              f'sessions with >=3 units={multi["session"].nunique()}')


# %%
def add_history(per_trial, k=K):
    """Recent-outcome and recent-switch counts for one session's trial table.

    `n_om` counts omissions among the k preceding trials and `n_switch` counts
    changes of side across those trials plus the current one, so two trials with
    the same `n_om` have had the same recent reward but may differ in whether the
    animal was repeating itself.
    """
    d = per_trial.sort_values('trial_ind').copy()
    ch, oc = d['choice'].values, d['outcome'].values
    n = len(d)
    n_om = np.full(n, np.nan)
    n_switch = np.full(n, np.nan)
    for i in range(k, n):
        prev_oc = oc[i - k:i]
        seq = ch[i - k:i + 1]                  # k preceding choices + this one
        if np.isnan(prev_oc).any() or pd.isna(seq).any():
            continue
        n_om[i] = int((prev_oc == 0).sum())
        n_switch[i] = int((np.diff(seq) != 0).sum())
    d['n_om'] = n_om
    d['n_switch'] = n_switch
    return d


def perseveration_vs_omissions(tr, k=K):
    """Same recent reward, different behaviour: repeat vs switch.

    Restricted to trials whose k preceding trials were ALL unrewarded. Within
    that set, `n_switch == 0` means a pure perseverative bout and `n_switch >= 1`
    means the animal explored while collecting the same number of omissions.
    """
    print('\n' + '=' * 78)
    print(f'2. PERSEVERATION vs RECENT OMISSIONS (last {k} trials all unrewarded)')
    print('=' * 78)

    hist = []
    for s, g in tr.groupby('session'):
        pt = g.groupby('trial_ind').first().reset_index()
        h = add_history(pt, k=k)[['trial_ind', 'n_om', 'n_switch']]
        h['session'] = s
        hist.append(h)
    hist = pd.concat(hist, ignore_index=True)
    tr = tr.merge(hist, on=['session', 'trial_ind'], how='left')

    allom = tr[tr['n_om'] == k]
    n_pt = allom.groupby(['session', 'trial_ind']).ngroups
    rep = allom[allom['n_switch'] == 0]
    swi = allom[allom['n_switch'] >= 1]
    print(f'\n   trials with {k}/{k} recent omissions: {n_pt}  '
          f'(repeat-only {rep.groupby(["session","trial_ind"]).ngroups}, '
          f'with a switch {swi.groupby(["session","trial_ind"]).ngroups})')

    # sanity: the two groups must really have the same recent reward
    for lab, d in (('repeat', rep), ('switch', swi)):
        p = d.groupby(['session', 'trial_ind']).first()
        print(f'   {lab:7s} mean n_om={p["n_om"].mean():.2f}  '
              f'mean ITI lick rate={p["lick_rate_iti"].mean():.3f} Hz')

    rows = []
    for (s, uid), g in allom.groupby(['session', 'unit_id']):
        r = g.loc[g['n_switch'] == 0, 'fr_iti'].dropna()
        w = g.loc[g['n_switch'] >= 1, 'fr_iti'].dropna()
        if len(r) >= MIN_TRIALS and len(w) >= MIN_TRIALS:
            rows.append({'session': s, 'unit_id': uid,
                         'fr_repeat': r.mean(), 'fr_switch': w.mean(),
                         'diff': r.mean() - w.mean(), 'n_r': len(r), 'n_w': len(w)})
    d = pd.DataFrame(rows)
    print(f'\n   per-unit ITI rate, repeat minus switch, recent reward held fixed:')
    report('repeat - switch (Hz)', d['diff'], unit='Hz')
    if len(d):
        print(f'   normalised: median {100 * (d["diff"] / d["fr_switch"]).median():+.2f}% '
              f'of the switch-matched rate')
        report('by session', d.groupby('session')['diff'].median(), unit='Hz')

    # and the graded version: within all-omission trials, does depth still matter?
    print('\n   graded within the same recent-reward set (rate vs streak depth):')
    rhos = []
    for (s, uid), g in allom.groupby(['session', 'unit_id']):
        gg = g[['fr_iti', 'unrew_streak_before']].dropna()
        if len(gg) >= 20 and gg['unrew_streak_before'].nunique() >= 3:
            rhos.append({'session': s,
                         'rho': spearmanr(gg['fr_iti'],
                                          gg['unrew_streak_before']).statistic})
    rhos = pd.DataFrame(rhos)
    if len(rhos):
        report('per-unit rho', rhos['rho'])
        report('by session', rhos.groupby('session')['rho'].median())
    return d


# %%
if __name__ == '__main__':
    part = pd.read_csv(os.path.join(OUT_DIR, 'reward_rate_partial_rho.csv'))
    u = pd.read_csv(os.path.join(OUT_DIR, 'reward_rate_unit_stats.csv'))
    stats_depth = pd.read_csv(os.path.join(
        OUT_DIR, 'persistence_unit_stats_streak_depth.csv'))
    with open(os.path.join(OUT_DIR, 'persistence_trial_rates.pkl'), 'rb') as f:
        tr = pickle.load(f)
    tr = tr[tr['in_streak_calc'] & tr['covered'] & tr['in_cut']].copy()

    print(f'{part["session"].nunique()} sessions / {len(part)} units in the '
          f'reward-rate table; {stats_depth["session"].nunique()} sessions / '
          f'{stats_depth.query("window == \'iti\'").dropna(subset=["diff"]).shape[0]} '
          f'units in the streak table\n')

    session_level(part, u, stats_depth)
    d = perseveration_vs_omissions(tr)
    d.to_csv(os.path.join(OUT_DIR, 'perseveration_vs_omissions.csv'), index=False)
    print(f'\nwrote {OUT_DIR}/perseveration_vs_omissions.csv')
