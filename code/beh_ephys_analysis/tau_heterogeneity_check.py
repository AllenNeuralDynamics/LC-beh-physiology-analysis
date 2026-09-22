"""
Do individual tagged DRN units really differ in reward integration timescale?

The best-fitting tau is bimodal across units (84 at tau=2, a dip at 10, 40 at
tau=40) and does not cluster by session, so it is not a property of the animal's
behaviour statistics. But `argmax` over five HIGHLY CORRELATED estimates
manufactures exactly this pattern from noise: a unit with no reward signal at all
gets a best tau, and the winner is whichever estimate the noise favoured.

The non-circular test is cross-validation. Classify each unit by its best tau on
half its trials, then ask -- on the OTHER half -- whether the fast-classified
units really prefer the fast regressor and the slow-classified units the slow one.
Under real heterogeneity the held-out rho(tau=2) - rho(tau=40) is positive for
fast units and negative for slow ones. Under argmax noise the held-out difference
is the same in both groups, and positive in both, because fast wins on average.

Odd/even trials, not first/second half: a within-unit split must not also split
the session in time, or the two halves differ in drift and in reward history.
"""
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

from reward_rate_tonic_analysis import load_with_regressors, unit_frames, FR
from utils.reward_rate import TAUS, partial_spearman
from utils.capsule_migration import capsule_directories

MIN_HALF = 30          # trials per half, for a slow-timescale correlation
capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'persistence_5HT')


def profile(g, taus=TAUS):
    """Drift-controlled partial rho with ITI rate, one per tau."""
    return [partial_spearman(g[f'rr_{t}'], g[FR], g['trial_ind']) for t in taus]


# %%
if __name__ == '__main__':
    tr = load_with_regressors()
    rows = []
    for s, uid, g in unit_frames(tr):
        g = g.dropna(subset=[FR, 'trial_ind'])
        odd = g[g['trial_ind'] % 2 == 1]
        even = g[g['trial_ind'] % 2 == 0]
        if len(odd) < MIN_HALF or len(even) < MIN_HALF:
            continue
        rec = {'session': s, 'unit_id': uid, 'n': len(g)}
        for lab, h in (('A', odd), ('B', even)):
            for t, r in zip(TAUS, profile(h)):
                rec[f'{lab}_rr_{t}'] = r
        for t, r in zip(TAUS, profile(g)):
            rec[f'all_rr_{t}'] = r
        rows.append(rec)
    d = pd.DataFrame(rows)
    print(f'{len(d)} units with >={MIN_HALF} trials in both halves\n')

    for lab in ('A', 'B', 'all'):
        cols = [f'{lab}_rr_{t}' for t in TAUS]
        d = d.dropna(subset=cols)
    d['best_A'] = [TAUS[i] for i in np.argmax(d[[f'A_rr_{t}' for t in TAUS]].values, 1)]
    d['best_B'] = [TAUS[i] for i in np.argmax(d[[f'B_rr_{t}' for t in TAUS]].values, 1)]
    d['best_all'] = [TAUS[i] for i in np.argmax(d[[f'all_rr_{t}' for t in TAUS]].values, 1)]

    print('best tau on the full data:', d['best_all'].value_counts().sort_index().to_dict())
    print('best tau on half A:       ', d['best_A'].value_counts().sort_index().to_dict())
    print('best tau on half B:       ', d['best_B'].value_counts().sort_index().to_dict())

    # ---- does the classification replicate at all?
    agree = (d['best_A'] == d['best_B']).mean()
    fast_A, fast_B = d['best_A'] <= 5, d['best_B'] <= 5
    agree_fs = (fast_A == fast_B).mean()
    # chance rates, given each half's own marginal distribution
    pA = d['best_A'].value_counts(normalize=True)
    pB = d['best_B'].value_counts(normalize=True)
    chance = sum(pA.get(t, 0) * pB.get(t, 0) for t in TAUS)
    chance_fs = fast_A.mean() * fast_B.mean() + (1 - fast_A.mean()) * (1 - fast_B.mean())
    print(f'\nexact agreement A vs B: {100*agree:.1f}%  (chance {100*chance:.1f}%)')
    print(f'fast/slow agreement:    {100*agree_fs:.1f}%  (chance {100*chance_fs:.1f}%)')

    # ---- THE test: held-out preference, split by the other half's classification
    print('\nHELD-OUT rho(tau=2) - rho(tau=40), by classification on the other half')
    print('  (real heterogeneity -> positive for fast, NEGATIVE for slow;'
          '\n   argmax noise      -> positive for both, and equal)')
    for train, test in (('A', 'B'), ('B', 'A')):
        pref = d[f'{test}_rr_2'] - d[f'{test}_rr_40']
        f = pref[d[f'best_{train}'] <= 5]
        s = pref[d[f'best_{train}'] >= 20]
        pf = wilcoxon(f.dropna()).pvalue
        ps = wilcoxon(s.dropna()).pvalue
        print(f'   train {train} -> test {test}:  '
              f'fast n={len(f):3d} med={f.median():+.4f} p={pf:.3g}   |   '
              f'slow n={len(s):3d} med={s.median():+.4f} p={ps:.3g}   |   '
              f'fast vs slow p={mannwhitneyu(f.dropna(), s.dropna()).pvalue:.3g}')

    # ---- and the same question as a continuous correlation, no argmax at all
    from scipy.stats import spearmanr
    prefA = d['A_rr_2'] - d['A_rr_40']
    prefB = d['B_rr_2'] - d['B_rr_40']
    ok = prefA.notna() & prefB.notna()
    rho, p = spearmanr(prefA[ok], prefB[ok])
    print(f'\nsplit-half reliability of the timescale preference itself '
          f'(rho2 - rho40): rho={rho:+.3f}  p={p:.3g}  n={int(ok.sum())}')
    # a floor to compare it against: how reliable is the reward signal per se?
    rho_m, p_m = spearmanr(d.loc[ok, 'A_rr_2'], d.loc[ok, 'B_rr_2'])
    print(f'  for reference, split-half reliability of rho(tau=2) itself:  '
          f'rho={rho_m:+.3f}  p={p_m:.3g}')

    d.to_csv(os.path.join(OUT_DIR, 'tau_heterogeneity_splithalf.csv'), index=False)
    print(f'\nwrote {OUT_DIR}/tau_heterogeneity_splithalf.csv')
