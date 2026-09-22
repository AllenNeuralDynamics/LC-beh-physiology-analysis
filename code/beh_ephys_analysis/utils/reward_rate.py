"""
Slow-timescale regressors: experienced reward rate and the latent block schedule.

Two families of predictor for tonic firing, which must be kept apart:

  experienced   exponentially-weighted reward history over trials (rr_<tau>),
                what the animal has actually collected
  latent        the hidden block probabilities themselves (p_L, p_R, p_sum, ...),
                which the animal never observes directly

They are correlated -- reward rate is the realised draw from the schedule -- so
every test here is run both ways round with the other partialled out.

Causality matters and is easy to get wrong. The ITI of trial n happens BEFORE
trial n's goCue and outcome (see utils/persistence.py for the verified
timeline), so a predictor of ITI firing on trial n may only use rewards from
trials < n. add_reward_rate() shifts by one trial for exactly this reason.
"""

import numpy as np
import pandas as pd

# exponential time constants in trials; 2 is nearly instantaneous, 40 spans
# roughly two blocks
TAUS = (2, 5, 10, 20, 40)


def _causal_ewma(x, tau):
    """Exponentially-weighted mean of x using only strictly earlier entries.

    adjust=True so the burn-in is a properly normalised weighted average rather
    than being dragged toward zero by implicit leading zeros.
    """
    alpha = 1.0 - np.exp(-1.0 / tau)
    return pd.Series(x).ewm(alpha=alpha, adjust=True).mean().shift(1)


def add_reward_rate(tbl, taus=TAUS):
    """Add causal reward-rate regressors at several timescales.

    Computed over responded trials only (a trial the animal ignored carries no
    outcome), then mapped back onto the full trial index so every trial gets the
    reward rate prevailing as it began.

    Adds, for each tau:
      rr_<tau>     overall reward rate  (rewards per trial)
      rr_L_<tau>   local income from the left spout
      rr_R_<tau>   local income from the right spout
      rr_diff_<tau>, rr_sum_<tau>   side contrast and total income
    """
    tbl = tbl.copy()
    resp = tbl['outcome'].notna()
    sub = tbl.loc[resp]

    rew = sub['outcome'].astype(float).values
    rew_l = (rew * (sub['choice'].values == 0)).astype(float)
    rew_r = (rew * (sub['choice'].values == 1)).astype(float)

    for tau in taus:
        for name, series in (('rr', rew), ('rr_L', rew_l), ('rr_R', rew_r)):
            col = f'{name}_{tau}'
            tbl[col] = np.nan
            tbl.loc[resp, col] = _causal_ewma(series, tau).values
        tbl[f'rr_sum_{tau}'] = tbl[f'rr_L_{tau}'] + tbl[f'rr_R_{tau}']
        tbl[f'rr_diff_{tau}'] = tbl[f'rr_R_{tau}'] - tbl[f'rr_L_{tau}']

    return tbl


def add_reward_lags(tbl, n_lags=3, distal=(6, 20)):
    """Add single-trial reward lags and a reward rate with recent trials removed.

    The point of `rr_distal` is to separate tonic coding from phasic carryover.
    The ITI of trial n begins right after trial n-1's outcome, so a lingering
    phasic reward response would show up as a correlation with any reward-history
    regressor -- and short-tau EWMAs are mostly just the last trial. rr_distal
    averages reward over trials [n-distal[1], n-distal[0]] only, leaving a gap of
    `distal[0]-1` trials, so nothing recent enough to be phasic can enter it.

    Adds rew_lag1..rew_lag<n_lags>, rr_proximal (mean of the last 3 trials) and
    rr_distal.
    """
    tbl = tbl.copy()
    resp = tbl['outcome'].notna()
    r = pd.Series(tbl.loc[resp, 'outcome'].astype(float).values)

    for k in range(1, n_lags + 1):
        tbl[f'rew_lag{k}'] = np.nan
        tbl.loc[resp, f'rew_lag{k}'] = r.shift(k).values

    tbl['rr_proximal'] = np.nan
    tbl.loc[resp, 'rr_proximal'] = r.rolling(3).mean().shift(1).values

    c = pd.Series(tbl.loc[resp, 'choice'].astype(float).values)
    tbl['prev_choice'] = np.nan
    tbl.loc[resp, 'prev_choice'] = c.shift(1).values

    near, far = distal
    tbl['rr_distal'] = np.nan
    tbl.loc[resp, 'rr_distal'] = (r.rolling(far - near + 1).mean()
                                   .shift(near).values)
    return tbl


def add_latent_probs(tbl):
    """Add the hidden block schedule and its block-structure bookkeeping.

    Adds:
      p_L, p_R          the scheduled probabilities themselves
      p_sum             total available reward -- environment richness
      p_max             the better spout's probability
      p_diff            signed contrast, p_R - p_L
      p_contrast        |p_R - p_L|, how discriminable the two spouts are
      p_chosen          probability of the spout the animal chose
      p_unchosen        probability of the one it did not
      block_id          index of the constant-schedule block
      trial_in_block    0-based position within the block
      block_len         number of trials in the block
      d_p_sum           change in richness at this block's onset (NaN elsewhere)
    """
    tbl = tbl.copy()
    p_l = tbl['reward_probabilityL'].astype(float)
    p_r = tbl['reward_probabilityR'].astype(float)

    tbl['p_L'], tbl['p_R'] = p_l, p_r
    tbl['p_sum'] = p_l + p_r
    tbl['p_max'] = np.maximum(p_l, p_r)
    tbl['p_diff'] = p_r - p_l
    tbl['p_contrast'] = (p_r - p_l).abs()

    ch = tbl['choice']
    tbl['p_chosen'] = np.where(ch == 1, p_r, np.where(ch == 0, p_l, np.nan))
    tbl['p_unchosen'] = np.where(ch == 1, p_l, np.where(ch == 0, p_r, np.nan))

    # p_chosen uses trial n's choice, which happens after trial n's ITI -- fine as
    # an expectation regressor but not causal. p_prev_chosen uses the side the
    # animal actually chose last, so it is available when the ITI begins.
    if 'prev_choice' in tbl.columns:
        pc = tbl['prev_choice']
        tbl['p_prev_chosen'] = np.where(pc == 1, p_r,
                                        np.where(pc == 0, p_l, np.nan))
        tbl['p_prev_unchosen'] = np.where(pc == 1, p_l,
                                          np.where(pc == 0, p_r, np.nan))

    changed = (p_l.diff().abs() > 1e-9) | (p_r.diff().abs() > 1e-9)
    changed.iloc[0] = True
    tbl['block_id'] = changed.cumsum() - 1
    tbl['trial_in_block'] = tbl.groupby('block_id').cumcount()
    tbl['block_len'] = tbl.groupby('block_id')['block_id'].transform('size')

    # richness step at each block onset, for transition-aligned averaging
    first = tbl['trial_in_block'] == 0
    tbl['d_p_sum'] = np.where(first, tbl['p_sum'].diff(), np.nan)

    return tbl


def add_slow_regressors(tbl, taus=TAUS):
    """All of add_reward_rate, add_reward_lags and add_latent_probs."""
    return add_latent_probs(add_reward_lags(add_reward_rate(tbl, taus=taus)))


# ------------------------------------------------------------------ statistics


def partial_spearman(y, x, controls, min_n=20):
    """Spearman of y vs x with `controls` (a 2-D array/DataFrame) removed.

    Everything is rank-transformed first, then x and y are each replaced by
    their residuals from a least-squares fit on the ranked controls. This is the
    usual rank-based partial correlation; it is monotone-robust but assumes the
    residualisation is linear in ranks.
    """
    from scipy.stats import spearmanr

    df = pd.DataFrame({'y': np.asarray(y, float), 'x': np.asarray(x, float)})
    ctrl = pd.DataFrame(controls).reset_index(drop=True)
    ctrl.columns = [f'c{i}' for i in range(ctrl.shape[1])]
    df = pd.concat([df.reset_index(drop=True), ctrl], axis=1).dropna()

    # x may legitimately be binary (a single-trial reward lag), so it only needs
    # two levels; y is a firing rate and wants more
    if len(df) < min_n or df['x'].nunique() < 2 or df['y'].nunique() < 3:
        return np.nan
    r = df.rank()
    keep = [c for c in ctrl.columns if r[c].nunique() > 1]
    if not keep:
        return spearmanr(r['y'], r['x']).statistic

    design = np.column_stack([np.ones(len(r))] + [r[c].values for c in keep])
    res = {}
    for v in ('y', 'x'):
        beta, *_ = np.linalg.lstsq(design, r[v].values, rcond=None)
        res[v] = r[v].values - design @ beta
    if res['x'].std() == 0 or res['y'].std() == 0:
        return np.nan
    return spearmanr(res['y'], res['x']).statistic


def standardized_betas(y, X):
    """OLS betas on z-scored y and X, plus R^2 and per-predictor delta-R^2.

    z-scoring makes the betas comparable across predictors and across units of
    very different firing rate. delta_r2[k] is the drop in R^2 from dropping
    predictor k, i.e. the variance only that predictor can explain.

    Returns (betas dict, r2, delta_r2 dict) or (None, nan, None) if unfittable.
    """
    X = pd.DataFrame(X)
    df = pd.concat([pd.Series(np.asarray(y, float), name='_y'),
                    X.reset_index(drop=True)], axis=1).dropna()
    if len(df) < 5 * (X.shape[1] + 1):
        return None, np.nan, None

    cols = [c for c in X.columns if df[c].nunique() > 1]
    if not cols or df['_y'].nunique() < 3:
        return None, np.nan, None

    z = lambda v: (v - v.mean()) / v.std(ddof=0)
    yz = z(df['_y']).values
    Xz = np.column_stack([z(df[c]).values for c in cols])

    def fit(mat):
        d = np.column_stack([np.ones(len(mat)), mat]) if mat.size else \
            np.ones((len(yz), 1))
        beta, *_ = np.linalg.lstsq(d, yz, rcond=None)
        resid = yz - d @ beta
        return beta, 1.0 - resid.var() / yz.var()

    beta, r2 = fit(Xz)
    betas = dict(zip(cols, beta[1:]))
    delta = {}
    for i, c in enumerate(cols):
        sub = np.delete(Xz, i, axis=1)
        _, r2_sub = fit(sub)
        delta[c] = r2 - r2_sub

    # predictors dropped for being constant get an explicit NaN, not a silent gap
    for c in X.columns:
        betas.setdefault(c, np.nan)
        delta.setdefault(c, np.nan)
    return betas, r2, delta
