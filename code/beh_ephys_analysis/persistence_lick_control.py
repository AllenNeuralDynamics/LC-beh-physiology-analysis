"""
Is the persistence ITI firing-rate effect just licking?

Deep-streak trials could differ in how much the animal licks during the ITI, and
DRN units are lick-modulated, so ITI lick rate is the main alternative
explanation for the effect. Four independent controls here:

  1. does ITI lick rate actually differ between deep and shallow streak trials?
  2. are these units lick-modulated in the ITI at all?
  3. zero-lick ITIs only -- the effect cannot be licking if it survives
  4. matched within unit x lick-rate bin, and per-unit partial correlation

Reads persistence_trial_rates.pkl written by persistence_5HT_analysis.py.
"""
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
import pickle
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from utils.capsule_migration import capsule_directories

DEEP, SHALLOW = 3, 1
MIN_TRIALS = 5

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'persistence_5HT')


def partial_spearman(x, y, z):
    """Spearman of x vs y controlling for z, via rank residuals."""
    d = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(d) < 10 or d['z'].nunique() < 3:
        return np.nan
    r = d.rank()
    rx = r['x'] - np.polyval(np.polyfit(r['z'], r['x'], 1), r['z'])
    ry = r['y'] - np.polyval(np.polyfit(r['z'], r['y'], 1), r['z'])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return spearmanr(rx, ry).statistic


# %%
if __name__ == '__main__':
    with open(os.path.join(OUT_DIR, 'persistence_trial_rates.pkl'), 'rb') as f:
        tr = pickle.load(f)
    # in_cut = manually curated drift window, same filter the main stats use
    tr = tr[tr['in_streak_calc'] & tr['covered'] & tr['in_cut']].copy()
    deep = tr['unrew_streak_before'] >= DEEP
    shallow = tr['unrew_streak_before'] == SHALLOW

    # ---- 1. does ITI licking differ at all?
    per_trial = tr.groupby(['session', 'trial_ind']).first().reset_index()
    d_ = per_trial[per_trial['unrew_streak_before'] >= DEEP]
    s_ = per_trial[per_trial['unrew_streak_before'] == SHALLOW]
    print('1. ITI LICKING, deep vs shallow')
    for col, lab in [('lick_rate_iti', 'lick rate (Hz)'), ('lick_n_iti', 'lick count')]:
        p = mannwhitneyu(d_[col].dropna(), s_[col].dropna()).pvalue
        print(f'   {lab:16s} deep={d_[col].median():6.3f}  shallow={s_[col].median():6.3f}  '
              f'mean {d_[col].mean():6.3f} vs {s_[col].mean():6.3f}  p={p:.3g}')
    print(f'   fraction of ITIs with zero licks: deep={100*(d_["lick_n_iti"]==0).mean():.1f}%  '
          f'shallow={100*(s_["lick_n_iti"]==0).mean():.1f}%')

    # ---- 2. are the units lick-modulated in the ITI?
    print('\n2. UNIT LICK MODULATION IN THE ITI (per-unit Spearman fr_iti vs lick_rate_iti)')
    rhos = []
    for (_s, _u), g in tr.groupby(['session', 'unit_id']):
        d = g[['fr_iti', 'lick_rate_iti']].dropna()
        if len(d) >= 20 and d['lick_rate_iti'].nunique() >= 3:
            rhos.append(spearmanr(d['fr_iti'], d['lick_rate_iti']).statistic)
    rhos = pd.Series(rhos).dropna()
    print(f'   n_units={len(rhos)}  median rho={rhos.median():+.4f}  '
          f'pos={int((rhos>0).sum())}/{len(rhos)}  wilcoxon p={wilcoxon(rhos).pvalue:.3g}')

    # ---- 3. zero-lick ITIs only
    print('\n3. ZERO-LICK ITIs ONLY (deep vs shallow)')
    z = tr[tr['lick_n_iti'] == 0]
    rows = []
    for (s_id, u), g in z.groupby(['session', 'unit_id']):
        dd = g.loc[g['unrew_streak_before'] >= DEEP, 'fr_iti'].dropna()
        sh = g.loc[g['unrew_streak_before'] == SHALLOW, 'fr_iti'].dropna()
        if len(dd) >= MIN_TRIALS and len(sh) >= MIN_TRIALS:
            rows.append({'diff': dd.mean() - sh.mean()})
    r = pd.DataFrame(rows)['diff']
    print(f'   n_units={len(r)}  median diff={r.median():+.4f} Hz  '
          f'frac negative={(r<0).mean():.3f}  wilcoxon p={wilcoxon(r).pvalue:.3g}')

    # ---- 4a. matched within unit x lick-rate bin
    print('\n4a. MATCHED WITHIN UNIT x ITI LICK-RATE BIN')
    tr['lbin'] = pd.cut(tr['lick_rate_iti'], [-.001, 0.001, 0.5, 1, 2, 4, 1e6])
    diffs = []
    for (_s, _u), g in tr.groupby(['session', 'unit_id']):
        for _b, gb in g.groupby('lbin', observed=True):
            dd = gb.loc[gb['unrew_streak_before'] >= DEEP, 'fr_iti'].dropna()
            sh = gb.loc[gb['unrew_streak_before'] == SHALLOW, 'fr_iti'].dropna()
            if len(dd) >= MIN_TRIALS and len(sh) >= MIN_TRIALS:
                diffs.append(dd.mean() - sh.mean())
    diffs = pd.Series(diffs)
    print(f'   n comparisons={len(diffs)}  median diff={diffs.median():+.4f} Hz  '
          f'frac negative={(diffs<0).mean():.3f}  wilcoxon p={wilcoxon(diffs).pvalue:.3g}')

    # ---- 4b. per-unit partial correlation, streak depth vs rate | lick rate
    print('\n4b. PER-UNIT PARTIAL SPEARMAN (fr_iti vs streak depth | ITI lick rate)')
    raw, part = [], []
    for (_s, _u), g in tr.groupby(['session', 'unit_id']):
        g = g[g['unrew_streak_before'] >= 1]
        if len(g) < 20:
            continue
        d = g[['fr_iti', 'unrew_streak_before', 'lick_rate_iti']].dropna()
        if len(d) < 20:
            continue
        raw.append(spearmanr(d['fr_iti'], d['unrew_streak_before']).statistic)
        part.append(partial_spearman(d['fr_iti'], d['unrew_streak_before'],
                                     d['lick_rate_iti']))
    raw, part = pd.Series(raw).dropna(), pd.Series(part).dropna()
    print(f'   raw     n={len(raw):3d}  median rho={raw.median():+.4f}  '
          f'neg={int((raw<0).sum())}/{len(raw)}  wilcoxon p={wilcoxon(raw).pvalue:.3g}')
    print(f'   partial n={len(part):3d}  median rho={part.median():+.4f}  '
          f'neg={int((part<0).sum())}/{len(part)}  wilcoxon p={wilcoxon(part).pvalue:.3g}')
