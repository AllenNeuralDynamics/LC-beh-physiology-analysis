"""
Per-unit functional feature matrix for the tagged-DRN population.

Assembles one row per (session, unit) from artifacts that already exist in
`{manuscript_fig_prep_dir}/persistence_5HT/`, so no spike-time pass is needed:

  persistence_trial_rates.pkl   per-unit x per-trial firing rate in 5 task
                                windows, with choice / outcome / lick columns
                                and the `covered` & `in_cut` drift flags
  reward_rate_unit_stats.csv    slow reward-rate coupling across 5 EWMA taus
  reward_rate_partial_rho.csv   partial Spearman rhos (trial index controlled)
  persistence_unit_stats.csv    perseveration coupling, per window

Feature blocks
--------------
profile   log2 rate ratio of each task window to the ITI window (4 features)
coding    rank AUC for outcome / ipsi-choice / hit, on trial-index-detrended
          rates (6 features)
lick      partial Spearman of rate vs lick rate in the same window, trial
          index controlled (2 features)
slow      reward-rate coupling, its timescale slope, block-schedule and
          reward-lag terms, perseveration coupling (6 features)
basic     log10 mean firing rate (1 feature)
waveform  the 7 focus features used by F_waveform_space_DRN_v2 (secondary
          block, kept separate from the functional set)

Drift handling follows the rule established for this dataset: trials are
restricted to `covered & in_cut`, and every within-unit feature is computed on
residuals after regressing out trial index (or with trial index as a partial
control), so session drift cannot masquerade as task coding.
"""

import os
import json

import numpy as np
import pandas as pd

from scipy.stats import rankdata, spearmanr

from utils.capsule_migration import capsule_directories
from utils.combine_tools import apply_qc, to_str_intlike


UNIT_KEY = ['session', 'unit_id']

WINDOWS = ['iti', 'iti_post', 'resp_gocue', 'resp_choice', 'consumption']
PROFILE_REF = 'iti'

# Minimum usable trials for a unit to get any feature, and minimum trials per
# group for a two-group AUC to be computed.
MIN_TRIALS = 100
MIN_GROUP = 15

FUNCTIONAL_BLOCKS = {
    'profile': [f'prof_{w}' for w in WINDOWS if w != PROFILE_REF],
    'coding': [
        'auc_outcome_resp_choice',
        'auc_outcome_consumption',
        'auc_outcome_iti_post',
        'auc_ipsi_resp_gocue',
        'auc_ipsi_resp_choice',
        'auc_hit_resp_gocue',
    ],
    'lick': ['lick_coup_iti', 'lick_coup_resp_choice'],
    'slow': [
        'rr_10_partial',
        'tau_slope',
        'p_sum_partial',
        'carry_rew_lag1',
        'streak_rho_iti',
        'streak_rho_resp_choice',
    ],
    'basic': ['log_fr'],
}

WAVEFORM_FEATURES = [
    'post_w',
    'trough_post_ratio_1D',
    'post_trough_slope',
    'pre_slope',
    'symmetry_slope_div_log',
    'symmetry_trough_dis',
    'symmetry_inte_div_log',
]

CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])


def functional_feature_names():
    return [f for block in FUNCTIONAL_BLOCKS.values() for f in block]


# ------------------------------------------------------------------ helpers


def _detrend_on_trial(rate, trial_ind):
    """Residual of log(rate + eps) after removing a linear trial-index trend.

    Returns NaN-padded residuals of the same length as `rate`; positions with
    a non-finite rate or trial index stay NaN.
    """
    rate = np.asarray(rate, dtype=float)
    trial_ind = np.asarray(trial_ind, dtype=float)
    out = np.full(rate.shape, np.nan)

    ok = np.isfinite(rate) & np.isfinite(trial_ind)
    if ok.sum() < 10:
        return out

    y = np.log10(rate[ok] + 0.1)
    x = trial_ind[ok]
    if np.std(x) < 1e-9 or np.std(y) < 1e-12:
        out[ok] = y - y.mean()
        return out

    slope, intercept = np.polyfit(x, y, 1)
    out[ok] = y - (slope * x + intercept)
    return out


def _auc(values, group):
    """Rank AUC (Mann-Whitney) for group==1 vs group==0. NaN if underpowered."""
    values = np.asarray(values, dtype=float)
    group = np.asarray(group, dtype=float)

    ok = np.isfinite(values) & np.isfinite(group)
    values, group = values[ok], group[ok]

    n1 = int((group == 1).sum())
    n0 = int((group == 0).sum())
    if n1 < MIN_GROUP or n0 < MIN_GROUP:
        return np.nan

    ranks = rankdata(values)
    return (ranks[group == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _partial_spearman(y, x, control):
    """Spearman of y vs x after regressing rank(control) out of both."""
    y, x, control = (np.asarray(v, dtype=float) for v in (y, x, control))
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(control)
    if ok.sum() < 20:
        return np.nan

    ry, rx, rc = (rankdata(v[ok]) for v in (y, x, control))
    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9 or np.std(rc) < 1e-9:
        return np.nan

    def resid(v):
        slope, intercept = np.polyfit(rc, v, 1)
        return v - (slope * rc + intercept)

    ry, rx = resid(ry), resid(rx)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(spearmanr(ry, rx).statistic)


# ------------------------------------------------------- trial-rate features


def trial_rate_features(trial_rates, rec_side):
    """Per-unit profile / coding / lick features from the trial-rate table.

    Parameters
    ----------
    trial_rates : pd.DataFrame
        `persistence_trial_rates.pkl`.
    rec_side : dict
        session -> recording hemisphere (+1 / -1), used to convert choice into
        ipsi (+1) / contra (-1) so side coding is comparable across sessions.
    """
    rows = []

    for (session, unit_id), tbl in trial_rates.groupby(UNIT_KEY, sort=False):
        # `covered & in_cut` is the drift-curated set; responded trials only for
        # anything conditioned on choice or outcome.
        usable = tbl['covered'].astype(bool) & tbl['in_cut'].astype(bool)
        resp = usable & tbl['responded'].astype(bool)

        row = {'session': session, 'unit_id': unit_id,
               'n_usable': int(usable.sum()), 'n_responded': int(resp.sum())}

        if resp.sum() < MIN_TRIALS:
            rows.append(row)
            continue

        sub = tbl.loc[resp]
        trial_ind = sub['trial_ind'].to_numpy(float)

        # -- profile: log2 rate ratio of each window to the ITI window
        mean_fr = {w: float(np.nanmean(sub[f'fr_{w}'].to_numpy(float)))
                   for w in WINDOWS}
        ref = mean_fr[PROFILE_REF]
        for w in WINDOWS:
            if w == PROFILE_REF:
                continue
            row[f'prof_{w}'] = (np.log2((mean_fr[w] + 0.05) / (ref + 0.05))
                                if np.isfinite(mean_fr[w]) and np.isfinite(ref)
                                else np.nan)
        row['mean_fr_iti'] = ref

        # -- detrended rates per window
        det = {w: _detrend_on_trial(sub[f'fr_{w}'].to_numpy(float), trial_ind)
               for w in WINDOWS}

        # -- outcome coding (rewarded vs unrewarded)
        outcome = sub['outcome'].to_numpy(float)
        for w in ['resp_choice', 'consumption', 'iti_post']:
            row[f'auc_outcome_{w}'] = _auc(det[w], outcome)

        # -- side coding, expressed as ipsi vs contra
        side = np.where(sub['choice'].to_numpy(float) == 1, 1.0, -1.0)
        side[~np.isfinite(sub['choice'].to_numpy(float))] = np.nan
        ipsi = side * rec_side.get(session, np.nan)
        ipsi_group = np.where(np.isfinite(ipsi), (ipsi > 0).astype(float), np.nan)
        for w in ['resp_gocue', 'resp_choice']:
            row[f'auc_ipsi_{w}'] = _auc(det[w], ipsi_group)

        # -- hit vs miss, over all drift-usable trials (misses are excluded
        #    from `resp`, so this block uses its own subset)
        sub_all = tbl.loc[usable]
        hit_det = _detrend_on_trial(sub_all['fr_resp_gocue'].to_numpy(float),
                                    sub_all['trial_ind'].to_numpy(float))
        row['auc_hit_resp_gocue'] = _auc(
            hit_det, sub_all['responded'].astype(float).to_numpy())

        # -- lick coupling, trial index partialled out
        row['lick_coup_iti'] = _partial_spearman(
            sub['fr_iti'].to_numpy(float),
            sub['lick_rate_iti'].to_numpy(float), trial_ind)
        row['lick_coup_resp_choice'] = _partial_spearman(
            sub['fr_resp_choice'].to_numpy(float),
            sub['lick_rate_resp_choice'].to_numpy(float), trial_ind)

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------ slow features


def slow_features(prep_dir):
    """Reward-rate / schedule / perseveration features from the saved CSVs."""
    rr = pd.read_csv(os.path.join(prep_dir, 'persistence_5HT',
                                  'reward_rate_unit_stats.csv'))
    pr = pd.read_csv(os.path.join(prep_dir, 'persistence_5HT',
                                  'reward_rate_partial_rho.csv'))
    pe = pd.read_csv(os.path.join(prep_dir, 'persistence_5HT',
                                  'persistence_unit_stats.csv'))

    for df in (rr, pr, pe):
        df['unit_id'] = df['unit_id'].apply(to_str_intlike)

    out = rr[UNIT_KEY].copy()
    # Nested EWMA taus are r > 0.95 with their neighbours, so take one level
    # plus the fast-to-slow slope rather than all five.
    out['tau_slope'] = rr['dpart_rr_40'].to_numpy() - rr['dpart_rr_2'].to_numpy()
    out['carry_rew_lag1'] = rr['carry_rew_lag1'].to_numpy()

    out = out.merge(
        pr[UNIT_KEY + ['rr_10|trial', 'p_sum|rr_10,trial']].rename(
            columns={'rr_10|trial': 'rr_10_partial',
                     'p_sum|rr_10,trial': 'p_sum_partial'}),
        on=UNIT_KEY, how='left')

    streak = pe.pivot_table(index=UNIT_KEY, columns='window',
                            values='rho_streak')
    streak = streak.rename(
        columns={'iti': 'streak_rho_iti',
                 'resp_choice': 'streak_rho_resp_choice'}).reset_index()
    keep = [c for c in ['streak_rho_iti', 'streak_rho_resp_choice']
            if c in streak.columns]
    out = out.merge(streak[UNIT_KEY + keep], on=UNIT_KEY, how='left')

    return out


# ------------------------------------------------------------------- driver


def build_feature_matrix(criteria_name='waveform_low_DRN', verbose=True):
    """Assemble the tagged-DRN functional feature matrix.

    Returns
    -------
    df : pd.DataFrame
        One row per QC-passing unit, with functional features, waveform
        features, bregma-relative CCF coords (`ml`, `ap`, `dv`), `y_loc`,
        `animal` and `session`.
    coverage : pd.DataFrame
        Non-missing count per feature.
    """
    import pickle

    prep = capsule_directories()['manuscript_fig_prep_dir']

    with open(os.path.join(prep, 'combined_unit_tbl',
                           'combined_unit_tbl.pkl'), 'rb') as f:
        combined = pickle.load(f)
    with open(os.path.join('/root/capsule/code/beh_ephys_analysis',
                           'session_combine/metrics',
                           f'{criteria_name}.json'), 'r') as f:
        constraints = json.load(f)

    filt, _, fig, _ = apply_qc(combined, constraints,
                               plot_all=False, plot_half=False)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass

    filt = filt.copy()
    filt['unit_id'] = filt['unit'].apply(to_str_intlike)
    rec_side = dict(zip(filt['session'], filt['rec_side']))

    with open(os.path.join(prep, 'persistence_5HT',
                           'persistence_trial_rates.pkl'), 'rb') as f:
        trial_rates = pickle.load(f)
    trial_rates['unit_id'] = trial_rates['unit_id'].apply(to_str_intlike)

    tr_feats = trial_rate_features(trial_rates, rec_side)
    sl_feats = slow_features(prep)

    wf = pd.read_csv(os.path.join(prep, 'waveforms_np',
                                  'combined_features.csv'))
    wf['unit_id'] = wf['unit'].apply(to_str_intlike)
    wf_cols = [c for c in WAVEFORM_FEATURES if c in wf.columns]

    ccf = filt[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    ccf[:, 0] = np.abs(ccf[:, 0])

    df = filt[UNIT_KEY + ['fr', 'y_loc', 'rec_side', 'probe',
                          'sex'] + CCF_COLS].copy()
    df[['ml', 'ap', 'dv']] = ccf
    df['animal'] = df['session'].str.split('_').str[1]
    df['log_fr'] = np.log10(df['fr'].to_numpy(float) + 0.01)

    df = (df.merge(tr_feats, on=UNIT_KEY, how='left')
            .merge(sl_feats, on=UNIT_KEY, how='left')
            .merge(wf[UNIT_KEY + wf_cols], on=UNIT_KEY, how='left'))

    feats = functional_feature_names() + wf_cols
    coverage = pd.DataFrame({
        'feature': feats,
        'n_finite': [int(np.isfinite(df[f].to_numpy(float)).sum())
                     for f in feats],
        'block': [next((b for b, fs in FUNCTIONAL_BLOCKS.items() if f in fs),
                       'waveform') for f in feats],
    })

    if verbose:
        print(f'QC units: {len(df)} | sessions: {df.session.nunique()} '
              f'| animals: {df.animal.nunique()}')
        print(coverage.to_string(index=False))

    return df, coverage


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    prep = capsule_directories()['manuscript_fig_prep_dir']
    out_dir = os.path.join(prep, 'functional_clusters')
    os.makedirs(out_dir, exist_ok=True)

    df, coverage = build_feature_matrix()
    df.to_csv(os.path.join(out_dir, 'feature_matrix.csv'), index=False)
    coverage.to_csv(os.path.join(out_dir, 'feature_coverage.csv'), index=False)
    print(f'\nSaved to {out_dir}')
