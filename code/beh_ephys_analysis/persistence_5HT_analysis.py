"""
Opto-tagged (5-HT) DRN unit activity on persistent vs non-persistent trials.

Persistent trial: the animal enters it having already made >= N_STREAK consecutive
unrewarded choices to the side it chooses again (streak resets on reward or switch).

Windows analysed per trial:
  iti          this trial's leading ITI          [start_time, delay_start_time]
  iti_post     the ITI after this trial's outcome (= next trial's leading ITI)
  resp_gocue   goCue_start_time + RESP_WIN
  resp_choice  reward_outcome_time + RESP_WIN
  consumption  reward delivery -> + reward_consumption_duration

Outputs (under <manuscript_fig_prep_dir>/persistence_5HT/):
  persistence_trial_rates.pkl   long format, one row per (unit, trial)
  persistence_unit_stats.csv    one row per (unit, window)
"""
# %%
import os
import sys
sys.path.append('/root/capsule/code/beh_ephys_analysis')

import json
import numpy as np
import pandas as pd
import pickle
import traceback
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from utils.capsule_migration import capsule_directories
from utils.combine_tools import apply_qc, to_str_intlike
from utils.persistence import session_persistence_rates, WINDOWS

CRITERIA_NAME = 'waveform_low_DRN'
N_STREAK = 3
RESP_WIN = (-0.5, 1.5)
DATA_TYPE = 'raw'
MIN_TRIALS = 5          # per condition, per unit
N_JOBS = -4

capsule_dirs = capsule_directories()
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'persistence_5HT')
METRICS_DIR = '/root/capsule/code/beh_ephys_analysis/session_combine/metrics'


# %%
def selected_unit_map(criteria_name=CRITERIA_NAME):
    """{session: {unit_id, ...}} for units passing the criteria JSON.

    Uses the same path as the other figure_preparation scripts: load
    combined_unit_tbl.pkl, key units by to_str_intlike(unit), and let apply_qc
    handle both the plain QC bounds and the per-condition opto criteria.
    """
    combined = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']),
                            'combined_unit_tbl', 'combined_unit_tbl.pkl')
    with open(combined, 'rb') as f:
        cu = pickle.load(f)
    cu['unit_id'] = cu['unit'].apply(to_str_intlike)

    with open(os.path.join(METRICS_DIR, f'{criteria_name}.json')) as f:
        constraints = json.load(f)

    filtered, _labeled, _fig, _axes = apply_qc(cu, constraints)

    return (filtered.groupby('session')['unit_id'].apply(set).to_dict(),
            filtered)


def process_session(session, allowed_units):
    try:
        return session_persistence_rates(session, allowed_units,
                                         n_streak=N_STREAK,
                                         data_type=DATA_TYPE, resp_win=RESP_WIN)
    except Exception as e:
        print(f'Error processing session {session}: {e!r}')
        traceback.print_exc()
        return None


# %%
def analysable(g):
    """Trials usable for a rate comparison, for one unit.

    `in_streak_calc` is behavioural; `covered` bounds trials by the unit's first
    and last spike; `in_cut` enforces the manually curated drift window, which
    `covered` does not imply -- a unit can drift into a different waveform
    mid-session and still have spikes throughout.
    """
    ok = g['in_streak_calc'] & g['covered']
    if 'in_cut' in g.columns:
        ok = ok & g['in_cut']
    return g[ok]


def unit_stats(trial_rates, min_trials=MIN_TRIALS, unrewarded_only=False):
    """Per-unit persistent vs non-persistent comparison in each window.

    Persistent trials are unrewarded by construction, so the naive contrast
    confounds perseveration with reward history. `unrewarded_only=True`
    restricts both groups to unrewarded trials, which removes that confound:
    persistent trials then differ from the comparison set only in how many
    unrewarded same-side choices preceded them.
    """
    rows = []
    for (session, unit_id), g in trial_rates.groupby(['session', 'unit_id']):
        g = analysable(g)
        if unrewarded_only:
            g = g[g['outcome'] == 0]
        for win in WINDOWS:
            fr = g[f'fr_{win}']
            ok = fr.notna()
            p_mask = ok & g['persistent']
            n_mask = ok & ~g['persistent']
            n_p, n_n = int(p_mask.sum()), int(n_mask.sum())

            rec = {'session': session, 'unit_id': unit_id, 'window': win,
                   'n_persistent': n_p, 'n_nonpersistent': n_n,
                   'fr_persistent': fr[p_mask].mean() if n_p else np.nan,
                   'fr_nonpersistent': fr[n_mask].mean() if n_n else np.nan,
                   'p_mwu': np.nan, 'rho_streak': np.nan, 'p_streak': np.nan}

            if n_p >= min_trials and n_n >= min_trials:
                rec['diff'] = rec['fr_persistent'] - rec['fr_nonpersistent']
                denom = rec['fr_persistent'] + rec['fr_nonpersistent']
                rec['mod_index'] = rec['diff'] / denom if denom > 0 else np.nan
                try:
                    rec['p_mwu'] = mannwhitneyu(fr[p_mask], fr[n_mask],
                                                alternative='two-sided').pvalue
                except ValueError:
                    pass
                # graded: does rate scale with how deep into the streak we are?
                sub = g[ok]
                if sub['unrew_streak_before'].notna().sum() >= min_trials:
                    rho, pv = spearmanr(sub['unrew_streak_before'],
                                        sub[f'fr_{win}'], nan_policy='omit')
                    rec['rho_streak'], rec['p_streak'] = rho, pv
            else:
                rec['diff'] = np.nan
                rec['mod_index'] = np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def unit_stats_streak_depth(trial_rates, deep=N_STREAK, shallow=1,
                            min_trials=MIN_TRIALS):
    """Deep vs shallow streak, matched on the immediately preceding trial.

    `unrew_streak_before >= 1` already implies the previous choice was to the
    same side AND unrewarded. So contrasting streak >= `deep` against
    streak == `shallow` holds the previous trial's choice and outcome fixed and
    varies only how deep into the perseverative bout the animal is. This is the
    contrast the ITI comparison needs, since a post-omission dip in tonic rate
    would otherwise masquerade as a persistence effect.
    """
    rows = []
    for (session, unit_id), g in trial_rates.groupby(['session', 'unit_id']):
        g = analysable(g)
        for win in WINDOWS:
            fr = g[f'fr_{win}']
            ok = fr.notna() & g['unrew_streak_before'].notna()
            d_mask = ok & (g['unrew_streak_before'] >= deep)
            s_mask = ok & (g['unrew_streak_before'] == shallow)
            n_d, n_s = int(d_mask.sum()), int(s_mask.sum())

            rec = {'session': session, 'unit_id': unit_id, 'window': win,
                   'n_deep': n_d, 'n_shallow': n_s,
                   'fr_persistent': fr[d_mask].mean() if n_d else np.nan,
                   'fr_nonpersistent': fr[s_mask].mean() if n_s else np.nan,
                   'p_mwu': np.nan, 'rho_streak': np.nan, 'p_streak': np.nan}
            if n_d >= min_trials and n_s >= min_trials:
                rec['diff'] = rec['fr_persistent'] - rec['fr_nonpersistent']
                try:
                    rec['p_mwu'] = mannwhitneyu(fr[d_mask], fr[s_mask],
                                                alternative='two-sided').pvalue
                except ValueError:
                    pass
                # graded, within the bout only
                sub = g[ok & (g['unrew_streak_before'] >= 1)]
                if len(sub) >= min_trials:
                    rho, pv = spearmanr(sub['unrew_streak_before'],
                                        sub[f'fr_{win}'], nan_policy='omit')
                    rec['rho_streak'], rec['p_streak'] = rho, pv
            else:
                rec['diff'] = np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def population_summary(stats):
    """Paired test across units, per window."""
    lines = []
    for win, g in stats.groupby('window'):
        g = g.dropna(subset=['fr_persistent', 'fr_nonpersistent'])
        if len(g) < 3:
            lines.append(f'{win:12s} n_units={len(g):3d}  (too few for a paired test)')
            continue
        try:
            p = wilcoxon(g['fr_persistent'], g['fr_nonpersistent']).pvalue
        except ValueError:
            p = np.nan
        n_sig = int((g['p_mwu'] < 0.05).sum())
        n_up = int((g['diff'] > 0).sum())
        lines.append(
            f'{win:12s} n_units={len(g):3d}  '
            f'median FR persistent={g["fr_persistent"].median():6.2f} Hz  '
            f'non={g["fr_nonpersistent"].median():6.2f} Hz  '
            f'up={n_up}/{len(g)}  wilcoxon p={p:.3g}  '
            f'per-unit sig={n_sig}'
        )
    return '\n'.join(lines)


# %%
if __name__ == '__main__':
    unit_map, selected = selected_unit_map()
    sessions = sorted(unit_map)
    print(f'\ncriteria {CRITERIA_NAME!r}: {len(selected)} units from '
          f'{len(sessions)} sessions')

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_session)(s, unit_map[s]) for s in sessions)
    results = [r for r in results if r is not None and len(r)]

    if not results:
        raise RuntimeError(
            f'No sessions yielded unit rates (of {len(sessions)} tried). '
            f'Check that unit tables exist for data_type={DATA_TYPE!r} and that '
            f'unit ids in combined_unit_tbl match those in the session tables.')

    trial_rates = pd.concat(results, ignore_index=True)
    stats = unit_stats(trial_rates)
    stats_unrew = unit_stats(trial_rates, unrewarded_only=True)
    stats_depth = unit_stats_streak_depth(trial_rates)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'persistence_trial_rates.pkl'), 'wb') as f:
        pickle.dump(trial_rates, f)
    stats.to_csv(os.path.join(OUT_DIR, 'persistence_unit_stats.csv'), index=False)
    stats_unrew.to_csv(os.path.join(OUT_DIR,
                                    'persistence_unit_stats_unrewarded_only.csv'),
                       index=False)
    stats_depth.to_csv(os.path.join(OUT_DIR,
                                    'persistence_unit_stats_streak_depth.csv'),
                       index=False)

    n_units = trial_rates.groupby(['session', 'unit_id']).ngroups
    print(f'\n{n_units}/{len(selected)} selected units recovered from '
          f'{trial_rates["session"].nunique()} sessions, '
          f'{len(trial_rates)} unit x trial rows')
    print(f'persistent trials: {int(trial_rates.groupby(["session","trial_ind"]).first()["persistent"].sum())} '
          f'of {trial_rates.groupby(["session","trial_ind"]).ngroups}')
    print()
    print('ALL TRIALS (persistent vs non-persistent -- confounded by reward history)')
    print(population_summary(stats))
    print()
    print('UNREWARDED TRIALS ONLY (reward-matched)')
    print(population_summary(stats_unrew))
    print()
    print(f'STREAK DEPTH >={N_STREAK} vs ==1 (also matched on previous trial; '
          f'the cleanest contrast)')
    print(population_summary(stats_depth))
    print('\n  (rho_streak = per-unit Spearman of rate vs streak depth, within bouts)')
    for win, g in stats_depth.groupby('window'):
        g = g.dropna(subset=['rho_streak'])
        if len(g) < 3:
            continue
        n_neg = int((g['rho_streak'] < 0).sum())
        try:
            p = wilcoxon(g['rho_streak']).pvalue
        except ValueError:
            p = np.nan
        print(f'  {win:12s} median rho={g["rho_streak"].median():+.4f}  '
              f'neg={n_neg}/{len(g)}  wilcoxon p={p:.3g}')

    # is persistence just late-in-session drift?
    per_trial = trial_rates.groupby(['session', 'trial_ind']).first().reset_index()
    frac = per_trial.groupby('session').apply(
        lambda g: pd.Series({
            'pos_persistent': g.loc[g['persistent'], 'trial_ind'].mean() / len(g),
            'pos_other': g.loc[~g['persistent'], 'trial_ind'].mean() / len(g)}),
        include_groups=False)
    print(f'\nmean normalised session position: persistent='
          f'{frac["pos_persistent"].mean():.3f}  other={frac["pos_other"].mean():.3f}')

    print(f'\nwrote -> {OUT_DIR}')
