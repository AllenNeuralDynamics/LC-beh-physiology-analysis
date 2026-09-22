"""
Are tagged DRN units modulated by reward rate or by the hidden block schedule?

Tonic, not phasic: the measure is firing rate in the ITI window
[start_time, delay_start_time], which precedes the trial's own cue, choice and
outcome, is 92% lick-free, and is the same window the perseveration effect lives
in. Nothing here is aligned to an event.

The two candidate drivers are correlated, so each is tested with the other
partialled out:

  experienced   rr_<tau>   causal EWMA of collected reward, tau = 2..40 trials
  latent        p_sum      scheduled p_L + p_R, i.e. environment richness
                p_L, p_R   each spout's hidden probability separately
                p_contrast |p_R - p_L|

Confounds handled explicitly:
  trial_ind             slow session drift -- the main threat, since any two slow
                        signals correlate; partialled out everywhere
  lick_rate_iti         movement (shown small for this window, but included)
  unrew_streak_before   perseveration depth -- an unrewarded streak IS a stretch
                        of low reward rate, so the previously reported streak
                        effect and a reward-rate effect could be one thing
  dur_iti               window length

Reads persistence_trial_rates.pkl written by persistence_5HT_analysis.py.
Writes per-unit tables + block-transition curves for the figure script.
"""
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr, wilcoxon

from utils.capsule_migration import capsule_directories
from utils.persistence import add_in_cut
from utils.reward_rate import (TAUS, add_slow_regressors, partial_spearman,
                               standardized_betas)

FR = 'fr_iti'
MIN_TRIALS = 50          # per unit, for a slow-timescale correlation
MIN_BLOCKS = 12          # per unit, for the block-level correlation
capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'persistence_5HT')

BEH_COLS = ['session', 'trial_ind', 'responded', 'in_streak_calc', 'choice',
            'outcome', 'unrew_streak_before', 'reward_probabilityL',
            'reward_probabilityR']

# the headline regressor set (single timescale for rr, chosen below)
LATENT = ['p_sum', 'p_L', 'p_R', 'p_contrast', 'p_diff', 'p_max',
          'p_chosen', 'p_unchosen']


def load_with_regressors():
    """Trial rates joined to slow behavioural regressors, computed per session."""
    with open(os.path.join(OUT_DIR, 'persistence_trial_rates.pkl'), 'rb') as f:
        tr = pickle.load(f)
    if 'in_cut' not in tr.columns:
        tr = add_in_cut(tr)

    # Regressors are computed on the whole behaviour session, before any ephys
    # masking: the animal's reward history is continuous whether or not we can
    # still see a given unit, and trial_ind must stay the true session index for
    # the drift control to mean anything. Units are masked in unit_frames().
    beh = (tr[BEH_COLS].groupby(['session', 'trial_ind']).first().reset_index()
             .sort_values(['session', 'trial_ind']))
    # per session, not groupby.apply: the EWMA must never run across a session
    # boundary, and an explicit loop makes that obvious
    beh = pd.concat([add_slow_regressors(g) for _s, g in beh.groupby('session')],
                    ignore_index=True)

    new = [c for c in beh.columns if c not in BEH_COLS]
    return tr.merge(beh[['session', 'trial_ind'] + new],
                    on=['session', 'trial_ind'], how='left')


def unit_frames(tr):
    """Yield (session, unit_id, per-trial frame) for analysable units.

    `in_cut` enforces the manually curated drift window (per unit, falling back
    to the session's synced recording extent). It is not redundant with
    `covered`, which only bounds trials by the unit's first and last spike.
    """
    tr = tr[tr['in_streak_calc'] & tr['covered'] & tr['in_cut']]
    for (s, u), g in tr.groupby(['session', 'unit_id']):
        g = g.sort_values('trial_ind')
        if g[FR].notna().sum() >= MIN_TRIALS:
            yield s, u, g


def pop(rhos, label, width=26):
    """Population summary of a per-unit statistic."""
    r = pd.Series(rhos).dropna()
    if len(r) < 3:
        return f'  {label:<{width}} n={len(r):3d}  (too few)'
    p = wilcoxon(r).pvalue
    return (f'  {label:<{width}} n={len(r):3d}  median={r.median():+.4f}  '
            f'neg={int((r < 0).sum()):3d}/{len(r)}  p={p:.3g}')


# %%
if __name__ == '__main__':
    tr = load_with_regressors()
    units = list(unit_frames(tr))
    print(f'{len(units)} units, {tr["session"].nunique()} sessions')
    usable = tr[tr['in_streak_calc'] & tr['covered']]
    print(f'  curated drift cut removes '
          f'{100 * (~usable["in_cut"]).mean():.1f}% of otherwise-usable '
          f'unit x trials, affecting '
          f'{usable.loc[~usable["in_cut"]].groupby(["session", "unit_id"]).ngroups} '
          f'of {usable.groupby(["session", "unit_id"]).ngroups} units')

    # ---------------------------------------------------------------- design
    print('\n=== DESIGN CHECKS (is the schedule confounded with session time?) ===')
    per_trial = tr.groupby(['session', 'trial_ind']).first().reset_index()
    for col in ['p_sum', 'p_contrast', 'rr_10']:
        rhos = [spearmanr(g['trial_ind'], g[col], nan_policy='omit').statistic
                for _s, g in per_trial.groupby('session')]
        print(pop(rhos, f'{col} vs trial index'))
    rhos = [spearmanr(g['p_sum'], g['rr_10'], nan_policy='omit').statistic
            for _s, g in per_trial.groupby('session')]
    print(pop(rhos, 'p_sum vs rr_10'))
    print(f'  blocks/session median={per_trial.groupby("session")["block_id"].max().median():.0f}'
          f'   block length median={per_trial["block_len"].median():.0f} trials')

    # ------------------------------------------------- which timescale, if any
    print('\n=== 1. EXPERIENCED REWARD RATE, BY TIMESCALE ===')
    print('    raw rho, then the same controlling session drift. Drift matters a '
          'lot here:\n    long-tau EWMAs are themselves slow signals, so they '
          'soak up drift.')
    rr_cols = [f'rr_{t}' for t in TAUS]
    raw = {c: [] for c in rr_cols}
    dpart = {c: [] for c in rr_cols}
    for _s, _u, g in units:
        for c in rr_cols:
            d = g[[FR, c]].dropna()
            raw[c].append(spearmanr(d[FR], d[c]).statistic if len(d) >= MIN_TRIALS
                          else np.nan)
            dpart[c].append(partial_spearman(g[FR], g[c], g[['trial_ind']]))
    for c in rr_cols:
        print(pop(raw[c], f'{c} raw (tau={c.split("_")[1]})'))
    print()
    for c in rr_cols:
        print(pop(dpart[c], f'{c} | trial index'))

    best_tau = []
    for i in range(len(units)):
        vals = {t: dpart[f'rr_{t}'][i] for t in TAUS}
        vals = {t: v for t, v in vals.items() if np.isfinite(v)}
        if vals:
            best_tau.append(max(vals, key=lambda t: vals[t]))
    print(f'  best tau per unit (max drift-controlled rho): '
          + '  '.join(f'{t}:{best_tau.count(t)}' for t in TAUS))

    # -------------------------------- is it tonic, or phasic carryover?
    print('\n=== 1b. TONIC OR PHASIC CARRYOVER? ===')
    print('    The ITI of trial n starts just after outcome n-1, so a lingering '
          'phasic\n    response would mimic reward-rate coding. rr_distal uses '
          'only trials\n    n-20..n-6, so nothing recent enough to be phasic can '
          'enter it.')
    lags = ['rew_lag1', 'rew_lag2', 'rew_lag3']
    carry = {c: [] for c in lags + ['rr_proximal', 'rr_distal',
                                    'rr_distal|lags', 'rr_20|lags']}
    for _s, _u, g in units:
        for c in lags + ['rr_proximal', 'rr_distal']:
            carry[c].append(partial_spearman(g[FR], g[c], g[['trial_ind']]))
        ctrl = g[lags + ['trial_ind']]
        carry['rr_distal|lags'].append(partial_spearman(g[FR], g['rr_distal'], ctrl))
        carry['rr_20|lags'].append(partial_spearman(g[FR], g['rr_20'], ctrl))
    for c in lags + ['rr_proximal', 'rr_distal']:
        print(pop(carry[c], f'{c} | trial'))
    print()
    for c in ['rr_distal|lags', 'rr_20|lags']:
        print(pop(carry[c], f'{c[:-5]} | lag1-3, trial'))

    # ------------------------------------------------------- latent schedule
    print('\n=== 2. LATENT BLOCK SCHEDULE (raw per-unit rho) ===')
    raw_lat = {c: [] for c in LATENT}
    for _s, _u, g in units:
        for c in LATENT:
            d = g[[FR, c]].dropna()
            raw_lat[c].append(spearmanr(d[FR], d[c]).statistic
                              if len(d) >= MIN_TRIALS else np.nan)
    for c in LATENT:
        print(pop(raw_lat[c], c))

    # p_R significant but p_L not would be odd for a midline structure, so check
    # whether it is just a choice-side bias making p_R stand in for p_chosen
    frac_r = per_trial.groupby('session')['choice'].mean()
    print(f'  choice bias: fraction rightward per session '
          f'median={frac_r.median():.3f}  range {frac_r.min():.2f}-{frac_r.max():.2f}')
    # side coding sanity check: animals should choose right more when p_R > p_L,
    # which also confirms choice==1 really is the right spout
    match = [spearmanr(g['choice'], g['p_diff'], nan_policy='omit').statistic
             for _s, g in per_trial.groupby('session')]
    print(pop(match, 'choice vs (p_R - p_L)'))

    # does the L/R asymmetry track how right-biased each session is? if the
    # asymmetry is a bias artefact this correlation should be strongly positive
    asym = pd.DataFrame({'session': [s for s, _u, _g in units],
                         'p_L': raw_lat['p_L'], 'p_R': raw_lat['p_R']})
    asym['d'] = asym['p_R'] - asym['p_L']
    per_sess = asym.groupby('session')['d'].median().to_frame()
    per_sess['bias'] = frac_r.reindex(per_sess.index)
    rho_b, p_b = spearmanr(per_sess['d'], per_sess['bias'], nan_policy='omit')
    print(f'  (rho_pR - rho_pL) vs session rightward bias: rho={rho_b:+.3f} '
          f'p={p_b:.3g}  (n={len(per_sess)} sessions)')
    for lo, hi, lab in [(0.0, 0.5, 'left-biased sessions'),
                        (0.5, 1.0, 'right-biased sessions')]:
        keep = set(per_sess.index[(per_sess['bias'] > lo) & (per_sess['bias'] <= hi)])
        sel = [i for i, (s, _u, _g) in enumerate(units) if s in keep]
        print(f'    {lab} (n={len(keep)}):')
        print('    ' + pop([raw_lat['p_L'][i] for i in sel], 'p_L'))
        print('    ' + pop([raw_lat['p_R'][i] for i in sel], 'p_R'))

    # ------------------------------------ drift control + mutual partialling
    print('\n=== 3. WITH CONFOUNDS PARTIALLED OUT (per-unit partial rho) ===')
    rows = []
    for s, u, g in units:
        rec = {'session': s, 'unit_id': u, 'n_trials': int(g[FR].notna().sum())}
        # each candidate, controlling drift alone
        for c in ['rr_10', 'p_sum', 'p_contrast', 'p_L', 'p_R']:
            rec[f'{c}|trial'] = partial_spearman(g[FR], g[c], g[['trial_ind']])
        # reciprocal: experienced vs latent, each with the other removed
        rec['rr_10|p_sum,trial'] = partial_spearman(
            g[FR], g['rr_10'], g[['p_sum', 'trial_ind']])
        rec['p_sum|rr_10,trial'] = partial_spearman(
            g[FR], g['p_sum'], g[['rr_10', 'trial_ind']])
        for c in ['p_R', 'p_chosen']:
            rec[f'{c}|rr_10,trial'] = partial_spearman(
                g[FR], g[c], g[['rr_10', 'trial_ind']])
        # p_chosen is a function of the choice itself, and the ITI of trial n sits
        # after choice n-1, so a side preference in tonic rate could impersonate
        # value coding. Control the side explicitly.
        rec['p_chosen|choice,rr,trial'] = partial_spearman(
            g[FR], g['p_chosen'], g[['choice', 'prev_choice', 'rr_10', 'trial_ind']])
        rec['p_prev_chosen|choice,rr,trial'] = partial_spearman(
            g[FR], g['p_prev_chosen'],
            g[['choice', 'prev_choice', 'rr_10', 'trial_ind']])
        rec['p_R|choice,rr,trial'] = partial_spearman(
            g[FR], g['p_R'], g[['choice', 'prev_choice', 'rr_10', 'trial_ind']])
        rec['choice|trial'] = partial_spearman(g[FR], g['choice'], g[['trial_ind']])
        rec['prev_choice|trial'] = partial_spearman(
            g[FR], g['prev_choice'], g[['trial_ind']])
        # full control set
        full = ['trial_ind', 'lick_rate_iti', 'dur_iti', 'unrew_streak_before']
        rec['rr_10|full'] = partial_spearman(g[FR], g['rr_10'], g[full + ['p_sum']])
        rec['p_sum|full'] = partial_spearman(g[FR], g['p_sum'], g[full + ['rr_10']])
        # is the streak effect separable from reward rate?
        rec['streak|rr_10,trial'] = partial_spearman(
            g[FR], g['unrew_streak_before'], g[['rr_10', 'trial_ind']])
        rec['streak|trial'] = partial_spearman(
            g[FR], g['unrew_streak_before'], g[['trial_ind']])
        rows.append(rec)
    part = pd.DataFrame(rows)

    for c in ['rr_10|trial', 'p_sum|trial', 'p_contrast|trial', 'p_L|trial',
              'p_R|trial', 'rr_10|p_sum,trial', 'p_sum|rr_10,trial',
              'p_R|rr_10,trial', 'p_chosen|rr_10,trial',
              'choice|trial', 'prev_choice|trial',
              'p_chosen|choice,rr,trial', 'p_prev_chosen|choice,rr,trial',
              'p_R|choice,rr,trial',
              'rr_10|full', 'p_sum|full', 'streak|trial', 'streak|rr_10,trial']:
        print(pop(part[c], c))

    # ------------------------------------------------- multiple regression
    print('\n=== 4. PER-UNIT MULTIPLE REGRESSION (z-scored betas) ===')
    preds = ['rr_10', 'p_sum', 'p_contrast', 'unrew_streak_before',
             'trial_ind', 'lick_rate_iti']
    breg, dreg, r2s = {p: [] for p in preds}, {p: [] for p in preds}, []
    for _s, _u, g in units:
        betas, r2, delta = standardized_betas(g[FR], g[preds])
        r2s.append(r2)
        for p in preds:
            breg[p].append(betas[p] if betas else np.nan)
            dreg[p].append(delta[p] if delta else np.nan)
    print(f'  full-model R^2: median={np.nanmedian(r2s):.4f}')
    for p in preds:
        print(pop(breg[p], f'beta {p}') +
              f'   median dR2={np.nanmedian(dreg[p]):.4f}')

    # ------------------------------------------------ block-level (slowest)
    print('\n=== 5. BLOCK-LEVEL: one mean per unit x block ===')
    blk_rows, blk_rho = [], {'p_sum': [], 'p_contrast': [], 'rr_10': []}
    for s, u, g in units:
        b = (g.dropna(subset=[FR])
               .groupby('block_id')
               .agg(fr=(FR, 'mean'), n=(FR, 'size'), p_sum=('p_sum', 'first'),
                    p_contrast=('p_contrast', 'first'), rr_10=('rr_10', 'mean'))
               .query('n >= 5').reset_index())
        if len(b) < MIN_BLOCKS:
            continue
        b['session'], b['unit_id'] = s, u
        blk_rows.append(b)
        for c in blk_rho:
            # a unit has only ~20 blocks, so the default n floor would discard
            # half the population here
            blk_rho[c].append(partial_spearman(b['fr'], b[c], b[['block_id']],
                                               min_n=MIN_BLOCKS))
    blocks = pd.concat(blk_rows, ignore_index=True)
    print(f'  {blocks.groupby(["session","unit_id"]).ngroups} units, '
          f'{len(blocks)} unit x block means')
    for c in blk_rho:
        print(pop(blk_rho[c], f'block {c} | block index'))

    # -------------------------------------- transition-aligned tonic profile
    print('\n=== 6. BLOCK-TRANSITION ALIGNED (normalised ITI rate) ===')
    pre, post = 10, 15
    tra = []
    for s, u, g in units:
        g = g.reset_index(drop=True)
        mu = g[FR].mean()
        if not np.isfinite(mu) or mu <= 0:
            continue
        # The post window sits ~11 trials later than the pre window, and tonic
        # rate drifts strongly downward across a session, so raw pre->post would
        # show a dip for any alignment whatsoever. Subtracting a centred rolling
        # median removes drift slower than a block while leaving block-timescale
        # structure intact.
        base = g[FR].rolling(61, center=True, min_periods=15).median()
        g['fr_dt'] = g[FR] - base
        onsets = g.index[g['trial_in_block'] == 0]
        for i in onsets:
            dps = g.loc[i, 'd_p_sum']
            if not np.isfinite(dps):
                continue
            lo, hi = i - pre, i + post
            if lo < 0 or hi >= len(g):
                continue
            tra.append(pd.DataFrame({
                'session': s, 'unit_id': u,
                'lag': np.arange(-pre, post + 1),
                'norm': g.loc[lo:hi, FR].values / mu,
                'dt': g.loc[lo:hi, 'fr_dt'].values / mu,
                'rr': g.loc[lo:hi, 'rr_10'].values,
                'rew': g.loc[lo:hi, 'outcome'].values,
                'd_p_sum': dps,
                'dir': 'richer' if dps > 0 else ('poorer' if dps < 0 else 'same')}))
    trans = pd.concat(tra, ignore_index=True)
    print('  transitions per direction: '
          + '  '.join(f'{k}:{v // (pre + post + 1)}'
                      for k, v in trans['dir'].value_counts().items()))
    prof = (trans.groupby(['dir', 'lag'])
                 .agg(mean=('norm', 'mean'), sem=('norm', 'sem'),
                      dt_mean=('dt', 'mean'), dt_sem=('dt', 'sem'),
                      size=('norm', 'size'), rr=('rr', 'mean'),
                      rew=('rew', 'mean'))
                 .reset_index())

    def pre_post(frame, col='norm'):
        return (frame.query('-6 <= lag < 0')[col].mean(),
                frame.query('3 <= lag <= 12')[col].mean())

    for d in ['richer', 'poorer']:
        q = prof.query('dir == @d')
        b, a = pre_post(q, 'mean')      # prof holds the aggregate, named 'mean'
        db, da = pre_post(q, 'dt_mean')
        rb, ra = pre_post(q, 'rew')
        print(f'  {d:8s} raw {100*(a-b):+.2f}%   detrended {100*(da-db):+.2f}%'
              f'   |  reward/trial {rb:.3f} -> {ra:.3f}')

    # Both directions dip after a switch, so the raw pre->post change is not a
    # richness signal -- it is whatever any block change does. The richer-minus-
    # poorer contrast cancels that generic component; the animal's own reward rate
    # aligned the same way says whether the dip is simply a reward-rate dip.
    for col, tag in (('norm', 'raw'), ('dt', 'detrended')):
        per_unit_dir = {}
        for d in ['richer', 'poorer']:
            sub = trans.query('dir == @d')
            vals = {}
            for k, g in sub.groupby(['session', 'unit_id']):
                b, a = pre_post(g, col)
                if np.isfinite(b) and np.isfinite(a):
                    vals[k] = a - b
            per_unit_dir[d] = vals
            print(pop(list(vals.values()), f'[{tag}] {d}: post - pre'))
        shared = sorted(set(per_unit_dir['richer']) & set(per_unit_dir['poorer']))
        print(pop([per_unit_dir['richer'][k] - per_unit_dir['poorer'][k]
                   for k in shared],
                  f'[{tag}] richer - poorer'))

    # ---------------------------------------------------------------- save
    part.to_csv(os.path.join(OUT_DIR, 'reward_rate_partial_rho.csv'), index=False)
    pd.DataFrame({'session': [s for s, _u, _g in units],
                  'unit_id': [u for _s, u, _g in units],
                  **{f'raw_{c}': raw[c] for c in rr_cols},
                  **{f'dpart_{c}': dpart[c] for c in rr_cols},
                  **{f'carry_{c}': v for c, v in carry.items()},
                  **{f'raw_{c}': raw_lat[c] for c in LATENT},
                  **{f'beta_{p}': breg[p] for p in preds},
                  **{f'dr2_{p}': dreg[p] for p in preds},
                  'r2': r2s}).to_csv(
        os.path.join(OUT_DIR, 'reward_rate_unit_stats.csv'), index=False)
    blocks.to_csv(os.path.join(OUT_DIR, 'reward_rate_block_means.csv'), index=False)
    prof.to_csv(os.path.join(OUT_DIR, 'reward_rate_transition_profile.csv'),
                index=False)
    trans.to_csv(os.path.join(OUT_DIR, 'reward_rate_transitions.csv'), index=False)
    print(f'\nwrote -> {OUT_DIR}')
