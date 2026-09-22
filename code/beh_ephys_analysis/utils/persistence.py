"""
Persistence (perseveration) trial labeling and tagged-unit rate extraction.

A "persistent" trial is one the animal enters having already made >= n_streak
consecutive *unrewarded* choices to the side it is about to choose again. The
streak resets on reward or on a side switch.

Trial timeline in these behavior NWBs (verified empirically -- note the ITI sits
at the START of a trial, not the end):

    start_time --ITI_duration--> delay_start_time --delay_duration--> goCue_start_time
                                                                          | lick_lat
                                                                          v
                                                 reward_outcome_time (choice / first lick)
                                                      | +reward_delay -> reward delivery
                                                      | +reward_consumption_duration
                                                      v
                                                   stop_time --~25ms--> next start_time

So the ITI *following* trial n's outcome is trial n+1's [start_time, delay_start_time].
"""

import numpy as np
import pandas as pd

import json
import os

from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename

from utils.beh_functions import session_dirs, get_session_tbl, get_unit_tbl
from utils.combine_tools import to_str_intlike


# ---------------------------------------------------------------- labeling


def label_persistence(tbl, n_streak=3, exclude_autowater=True,
                      exclude_interruptions=True):
    """Label persistent trials on a session trial table.

    Parameters
    ----------
    tbl : pd.DataFrame
        Trial table from get_session_tbl().
    n_streak : int
        Number of consecutive unrewarded same-side choices that must already
        have happened for the next same-side choice to count as persistent.
    exclude_autowater : bool
        Drop auto-water (free reward) trials from the streak bookkeeping.
    exclude_interruptions : bool
        Drop experimenter-interrupted trials (auto_manual_trial / extra_reward).

    Returns
    -------
    pd.DataFrame
        Copy of `tbl` with added columns:
          responded            : animal_response != 2
          in_streak_calc       : trial used for streak bookkeeping
          choice               : 0=left, 1=right, NaN if no response
          outcome              : 1 if rewarded, 0 if not, NaN if no response
          unrew_streak_before  : consecutive unrewarded choices to the
                                 about-to-be-chosen side, on entering the trial
          persistent           : unrew_streak_before >= n_streak
          run_id               : index of the same-side run this trial belongs to
          run_pos              : 0-based position within that run
          chose_worse_side     : chosen side had the lower reward_probability
    """
    tbl = tbl.copy()

    responded = tbl['animal_response'] != 2
    tbl['responded'] = responded

    use = responded.copy()
    if exclude_autowater and {'auto_waterL', 'auto_waterR'} <= set(tbl.columns):
        use &= ~((tbl['auto_waterL'] == 1) | (tbl['auto_waterR'] == 1))
    if exclude_interruptions and {'auto_manual_trial', 'extra_reward'} <= set(tbl.columns):
        use &= ~(tbl['auto_manual_trial'].fillna(False).astype(bool)
                 | tbl['extra_reward'].fillna(False).astype(bool))
    tbl['in_streak_calc'] = use

    choice = tbl['animal_response'].where(responded)
    outcome = (tbl['rewarded_historyL'].astype(bool)
               | tbl['rewarded_historyR'].astype(bool)).astype(float).where(responded)
    tbl['choice'] = choice
    tbl['outcome'] = outcome

    streak_before = np.full(len(tbl), np.nan)
    persistent = np.zeros(len(tbl), dtype=bool)
    run_id = np.full(len(tbl), np.nan)
    run_pos = np.full(len(tbl), np.nan)

    cur_side, streak, rid, rpos = None, 0, -1, 0
    for i, (ok, ch, rw) in enumerate(zip(use.values, choice.values, outcome.values)):
        if not ok or np.isnan(ch):
            continue

        same_side = (cur_side is not None) and (ch == cur_side)
        streak_before[i] = streak if same_side else 0
        persistent[i] = same_side and (streak >= n_streak)

        if same_side:
            rpos += 1
        else:
            cur_side, streak, rid, rpos = ch, 0, rid + 1, 0
        run_id[i], run_pos[i] = rid, rpos

        streak = streak + 1 if rw == 0 else 0

    tbl['unrew_streak_before'] = streak_before
    tbl['persistent'] = persistent
    tbl['run_id'] = run_id
    tbl['run_pos'] = run_pos

    if {'reward_probabilityL', 'reward_probabilityR'} <= set(tbl.columns):
        p_chosen = np.where(choice == 1, tbl['reward_probabilityR'],
                            tbl['reward_probabilityL'])
        p_other = np.where(choice == 1, tbl['reward_probabilityL'],
                           tbl['reward_probabilityR'])
        tbl['chose_worse_side'] = np.where(responded, p_chosen < p_other, np.nan)

    return tbl


def add_analysis_windows(tbl, resp_win=(-0.5, 1.5), consumption_win=None):
    """Add the window edges used for spike counting.

    Windows added (all absolute session-clock seconds):
      iti_start / iti_stop            this trial's own leading ITI
      iti_post_start / iti_post_stop  the ITI *after* this trial's outcome,
                                      i.e. the next trial's leading ITI
      resp_start / resp_stop          goCue + resp_win
      choice_start / choice_stop      reward_outcome_time + resp_win
      cons_start / cons_stop          outcome -> outcome + consumption duration
    """
    tbl = tbl.copy()

    tbl['iti_start'] = tbl['start_time']
    tbl['iti_stop'] = tbl['delay_start_time']

    # the ITI that follows this trial's outcome belongs to the NEXT trial
    tbl['iti_post_start'] = tbl['start_time'].shift(-1)
    tbl['iti_post_stop'] = tbl['delay_start_time'].shift(-1)

    tbl['resp_start'] = tbl['goCue_start_time'] + resp_win[0]
    tbl['resp_stop'] = tbl['goCue_start_time'] + resp_win[1]

    tbl['choice_start'] = tbl['reward_outcome_time'] + resp_win[0]
    tbl['choice_stop'] = tbl['reward_outcome_time'] + resp_win[1]

    reward_delay = tbl['reward_delay'].fillna(0.0)
    if consumption_win is None:
        cons_dur = tbl['reward_consumption_duration'].fillna(0.0)
    else:
        cons_dur = pd.Series(consumption_win, index=tbl.index)
    tbl['cons_start'] = tbl['reward_outcome_time'] + reward_delay
    tbl['cons_stop'] = tbl['cons_start'] + cons_dur

    return tbl


WINDOWS = {
    'iti': ('iti_start', 'iti_stop'),
    'iti_post': ('iti_post_start', 'iti_post_stop'),
    'resp_gocue': ('resp_start', 'resp_stop'),
    'resp_choice': ('choice_start', 'choice_stop'),
    'consumption': ('cons_start', 'cons_stop'),
}


# ------------------------------------------------------------ spike counting


def count_in_windows(spike_times, starts, stops):
    """Spike count and rate in each [start, stop) window.

    Returns (counts, rates, durations); NaN where the window is undefined.
    """
    st = np.asarray(spike_times, dtype=float)
    st = np.sort(st[np.isfinite(st)])
    starts = np.asarray(starts, dtype=float)
    stops = np.asarray(stops, dtype=float)

    counts = np.full(len(starts), np.nan)
    rates = np.full(len(starts), np.nan)
    durs = stops - starts

    ok = np.isfinite(starts) & np.isfinite(stops) & (durs > 0)
    if ok.any() and len(st):
        i0 = np.searchsorted(st, starts[ok], side='left')
        i1 = np.searchsorted(st, stops[ok], side='right')
        counts[ok] = i1 - i0
        rates[ok] = counts[ok] / durs[ok]

    return counts, rates, durs


def unit_trial_rates(spike_times, tbl, windows=None, presence_margin=0.0):
    """Per-trial spike counts/rates for one unit across all windows.

    `covered` flags trials whose windows fall inside the unit's spiking range --
    units can drift in and out, so a rate of 0 is not always a real 0.
    """
    windows = windows or WINDOWS
    st = np.sort(np.asarray(spike_times, dtype=float))
    out = pd.DataFrame(index=tbl.index)

    for name, (c0, c1) in windows.items():
        counts, rates, durs = count_in_windows(st, tbl[c0].values, tbl[c1].values)
        out[f'n_{name}'] = counts
        out[f'fr_{name}'] = rates
        out[f'dur_{name}'] = durs

    if len(st):
        lo, hi = st.min() - presence_margin, st.max() + presence_margin
        out['covered'] = (tbl['start_time'].values >= lo) & (tbl['stop_time'].values <= hi)
    else:
        out['covered'] = False

    return out


def session_licks(session):
    """All lick times for a session, as (left, right, all) sorted arrays.

    Licks live in the NWB acquisition series, not in the trials table, so this
    is the only way to get licks that fall inside an ITI rather than the single
    choice lick recorded as reward_outcome_time.
    """
    session_dir = session_dirs(session)
    nwb_file = os.path.join(session_dir['beh_fig_dir'], f'{session}.nwb')
    if not os.path.exists(nwb_file):
        return None, None, None

    nwb = load_nwb_from_filename(nwb_file)
    left = np.sort(np.asarray(nwb.acquisition['left_lick_time'].timestamps[:],
                              dtype=float))
    right = np.sort(np.asarray(nwb.acquisition['right_lick_time'].timestamps[:],
                               dtype=float))
    return left, right, np.sort(np.concatenate([left, right]))


def lick_trial_rates(session, tbl, windows=None):
    """Per-trial lick counts and rates in each window.

    Licking is the main alternative explanation for an ITI firing-rate effect,
    so these columns exist to be regressed out / matched on.
    """
    windows = windows or WINDOWS
    left, right, allc = session_licks(session)
    out = pd.DataFrame(index=tbl.index)
    if allc is None:
        return out

    for name, (c0, c1) in windows.items():
        starts, stops = tbl[c0].values, tbl[c1].values
        n_all, r_all, _ = count_in_windows(allc, starts, stops)
        n_l, _, _ = count_in_windows(left, starts, stops)
        n_r, _, _ = count_in_windows(right, starts, stops)
        out[f'lick_n_{name}'] = n_all
        out[f'lick_rate_{name}'] = r_all
        with np.errstate(invalid='ignore', divide='ignore'):
            out[f'lick_bias_{name}'] = (n_r - n_l) / n_all
    return out


# ------------------------------------------------------------ drift cut points


def curated_cut(session, data_type='raw'):
    """Manually curated drift cut points for one session, in trial-table time.

    Two levels, both needed:

      session   <session>_qm.json 'ephys_cut' -- the synced recording extent
      per unit  <session>_opto_drift_tbl.csv 'ephys_cut' -- where a unit that was
                flagged `drift_unit` stops being the same unit. NaN on a side
                means no cut there, and falls back to the session bound.

    Units the curator did not flag get the session bounds. Returns
    (lo, hi, {unit_id: (lo, hi)}) with unit ids in to_str_intlike form; lo/hi
    are -inf/+inf if the session json is missing.
    """
    session_dir = session_dirs(session)
    qm_file = os.path.join(session_dir['processed_dir'], f'{session}_qm.json')
    lo, hi = -np.inf, np.inf
    if os.path.exists(qm_file):
        with open(qm_file) as f:
            cut = json.load(f).get('ephys_cut', [None, None])
        lo = cut[0] if cut[0] is not None else lo
        hi = cut[1] if cut[1] is not None else hi

    per_unit = {}
    drift_file = os.path.join(session_dir[f'opto_dir_{data_type}'],
                              f'{session}_opto_drift_tbl.csv')
    if os.path.exists(drift_file):
        tbl = pd.read_csv(drift_file)
        for _, row in tbl.iterrows():
            try:
                # the column is a stringified python list holding bare `nan`
                cut = json.loads(str(row['ephys_cut']).replace('nan', 'null'))
            except (ValueError, TypeError):
                continue
            u_lo = lo if cut[0] is None else max(lo, cut[0])
            u_hi = hi if cut[1] is None else min(hi, cut[1])
            per_unit[to_str_intlike(row['unit_id'])] = (u_lo, u_hi)

    return lo, hi, per_unit


def add_in_cut(trial_rates, data_type='raw'):
    """Add `in_cut`: is this unit x trial inside the curated drift window?

    Retrofits an existing long-format table, so nothing has to be re-extracted
    from spikes. Masking whole trials is equivalent to having truncated the
    spike train, because every window this table reports lies inside one trial:
    a trial fully inside the window has identical rates either way, and a trial
    outside it contributes nothing.

    `covered` does not substitute for this. That flag only asks whether a trial
    falls between the unit's first and last spike, so a unit that drifts into a
    different waveform mid-session still looks covered throughout.
    """
    out = trial_rates.copy()
    out['in_cut'] = True
    for session, g in out.groupby('session'):
        lo, hi, per_unit = curated_cut(session, data_type=data_type)
        u = g['unit_id'].map(per_unit)
        u_lo = u.map(lambda t: t[0] if isinstance(t, tuple) else lo)
        u_hi = u.map(lambda t: t[1] if isinstance(t, tuple) else hi)
        out.loc[g.index, 'in_cut'] = ((g['start_time'] >= u_lo)
                                      & (g['stop_time'] <= u_hi)).values
    return out


# ---------------------------------------------------------------- per session


def selected_units(session, allowed_units, data_type='raw'):
    """Rows of the session unit table whose unit_id is in `allowed_units`.

    `allowed_units` holds unit ids as strings (to_str_intlike form), since that
    is how the combined unit table -- and therefore apply_qc -- keys them.
    Returns None if no unit table exists or nothing matches.
    """
    ut = get_unit_tbl(session, data_type)
    if ut is None or 'unit_id' not in ut.columns:
        return None
    ids = ut['unit_id'].apply(to_str_intlike)
    sel = ut[ids.isin(set(allowed_units))]
    return sel if len(sel) else None


def session_persistence_rates(session, allowed_units, n_streak=3,
                              data_type='raw', resp_win=(-0.5, 1.5)):
    """Long-format per-unit x per-trial table for one session.

    `allowed_units` is the set of unit ids (strings) that passed the QC/opto
    criteria for this session -- see apply_qc with a metrics JSON such as
    session_combine/metrics/waveform_low_DRN.json.

    Returns None if the session has no behavior table or no selected units.
    """
    tbl = get_session_tbl(session)
    if tbl is None or not len(tbl):
        return None

    tagged = selected_units(session, allowed_units, data_type=data_type)
    if tagged is None:
        return None

    tbl = label_persistence(tbl, n_streak=n_streak)
    tbl = add_analysis_windows(tbl, resp_win=resp_win)

    keep = ['start_time', 'stop_time', 'goCue_start_time', 'reward_outcome_time',
            'responded', 'in_streak_calc', 'choice', 'outcome',
            'unrew_streak_before', 'persistent', 'run_id', 'run_pos',
            'chose_worse_side', 'lick_lat', 'ITI_duration',
            'reward_probabilityL', 'reward_probabilityR']
    keep = [c for c in keep if c in tbl.columns]

    licks = lick_trial_rates(session, tbl)
    keep_beh = pd.concat([tbl[keep], licks], axis=1)

    lo, hi, per_unit = curated_cut(session, data_type=data_type)

    rows = []
    for _, u in tagged.iterrows():
        unit_id = to_str_intlike(u['unit_id'])
        rates = unit_trial_rates(u['spike_times'], tbl)
        rec = pd.concat([keep_beh.reset_index(drop=True),
                         rates.reset_index(drop=True)], axis=1)
        rec.insert(0, 'session', session)
        rec.insert(1, 'unit_id', unit_id)
        rec.insert(2, 'trial_ind', np.arange(len(rec)))
        u_lo, u_hi = per_unit.get(unit_id, (lo, hi))
        rec['in_cut'] = (rec['start_time'] >= u_lo) & (rec['stop_time'] <= u_hi)
        rows.append(rec)

    return pd.concat(rows, ignore_index=True)
