# %%
import sys
import os
sys.path.append('/root/capsule/code/beh_ephys_analysis')
from utils.beh_functions import parseSessionID, session_dirs, get_unit_tbl, get_session_tbl
from utils.plot_utils import shiftedColorMap, template_reorder, get_gradient_colors
from utils.opto_utils import opto_metrics, get_opto_tbl
from utils.ephys_functions import cross_corr_train, auto_corr_train, load_drift
import argparse
import json
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from aind_ephys_utils import align
from scipy.stats import wilcoxon

# Columns whose presence marks a unit as passing some QC scheme. Saved alongside
# the p-values so the table can be re-filtered without recomputing.
QC_FLAG_COLS = ['default_qc', 'default_qc_nonopto', 'opto_pass']


# %%
def _counts_in_win(spike_times, event_times, win):
    """Spike count in ``[t + win[0], t + win[1])`` for every event time.

    Gives the same counts as ``align.to_events(spike_times, event_times, win,
    return_df=True)`` tallied by ``event_index`` -- that function also locates
    both edges with a left-sided ``searchsorted`` -- but resolves all events in
    one vectorised call instead of one pass per event.

    ``spike_times`` must be sorted ascending (as ``align.to_events`` also
    requires).
    """
    lo = np.searchsorted(spike_times, event_times + win[0])
    hi = np.searchsorted(spike_times, event_times + win[1])
    return (hi - lo).astype(float)


def cal_opto_sigs(session, data_type='raw', loc='soma', qc_filter=None,
                  pre_win_ratio=0.5, post_win=0.025, extra_post_wins=(0.05,),
                  apply_drift_cut=True, save=True):
    """Per-pulse laser-response significance for every unit in one session.

    For each (unit, power, site, freq, pre_post) condition and each pulse of the
    train, the firing rate in the post-pulse window is compared against the
    pre-pulse baseline across trains with a paired Wilcoxon signed-rank test.

    Parameters
    ----------
    session : str
        Session id.
    data_type : str
        'raw' or 'curated'. Selects which ``opto_dir_*`` the inputs and output
        live in. Only 'raw' is populated in this capsule.
    loc : str
        Emission location passed to :func:`get_opto_tbl`.
    qc_filter : None, str or list of str, optional
        None (default) keeps every unit in the table; QC flags are saved as
        columns instead, so the result can be re-filtered without recomputing.
        Otherwise a column name (or list of names) in the unit table -- a unit
        is kept if *any* of the named flags is truthy.
    pre_win_ratio : float
        Baseline window length as a fraction of the inter-pulse interval.
    post_win : float
        Primary post-pulse response window, in seconds. Drives the unsuffixed
        output columns.
    extra_post_wins : sequence of float
        Additional response windows scored in the same pass and written to
        columns suffixed by window length, e.g. ``p_sig_count_50ms``. The
        pkls shipped before this patch mixed a 50 ms window (sessions written
        2026-03-29) with a 25 ms one (2026-07-03); keeping both here means the
        comparison never needs another run.
    apply_drift_cut : bool
        Restrict both spikes and laser trains to the unit's hand-curated
        ``ephys_cut`` stability window.
    save : bool
        Write ``{session}_opto_sigs.pkl`` into the session's opto directory.

    Returns
    -------
    pandas.DataFrame
        One row per unit x condition. ``p_unit_condition`` / ``p_sig_count``
        keep their original meaning (``p_sig_count`` is unsigned, counting any
        significant pulse). ``delta_unit_condition`` gives the signed post-minus-
        pre rate change per pulse, and ``p_sig_count_exc`` / ``p_sig_count_inh``
        split the count by direction.
    """
    # primary window first, then any extras, de-duplicated but order preserved
    all_post_wins = [post_win] + [w for w in (extra_post_wins or ()) if w != post_win]

    unit_tbl = get_unit_tbl(session, data_type)
    if unit_tbl is None or len(unit_tbl) == 0:
        raise ValueError(f'No unit table for {session} ({data_type})')
    opto_tbl = get_opto_tbl(session, data_type, loc=loc)

    # loop through all conditions
    powers = [p for p in opto_tbl['power'].unique().tolist() if pd.notna(p)]
    sites = [s for s in opto_tbl['site'].unique().tolist() if pd.notna(s)]
    pre_posts = [p for p in opto_tbl['pre_post'].unique().tolist() if pd.notna(p)]
    freqs = [f for f in opto_tbl['freq'].unique().tolist() if pd.notna(f)]

    if qc_filter is None:
        unit_ids_focus = unit_tbl['unit_id'].unique().tolist()
    else:
        cols = [qc_filter] if isinstance(qc_filter, str) else list(qc_filter)
        missing = [c for c in cols if c not in unit_tbl.columns]
        if missing:
            raise KeyError(f'{session}: qc_filter column(s) {missing} not in unit table')
        keep = np.zeros(len(unit_tbl), dtype=bool)
        for c in cols:
            keep |= (unit_tbl[c].values == 1)
        unit_ids_focus = unit_tbl[keep]['unit_id'].unique().tolist()

    # QC flags are carried through so downstream code can filter on any scheme.
    qc_cols = [c for c in QC_FLAG_COLS if c in unit_tbl.columns]
    qc_lookup = unit_tbl.set_index('unit_id')[qc_cols] if qc_cols else None

    rows = []
    for unit in unit_ids_focus:
        spike_times = unit_tbl[unit_tbl['unit_id'] == unit]['spike_times'].values[0]
        spike_times = np.asarray(spike_times, dtype=float)
        # searchsorted needs ascending order; cheap to verify, cheap to repair.
        if spike_times.size and not np.all(np.diff(spike_times) >= 0):
            spike_times = np.sort(spike_times)

        # Restrict to the unit's stability window. load_drift defaults to
        # data_type='curated', so data_type must be passed explicitly or the
        # drift table is never found and the cut silently does nothing.
        cut_lo, cut_hi = None, None
        if apply_drift_cut:
            unit_drift = load_drift(session, unit, data_type=data_type)
            if unit_drift is not None:
                cut_lo, cut_hi = unit_drift['ephys_cut'][0], unit_drift['ephys_cut'][1]
                if cut_lo is not None:
                    spike_times = spike_times[spike_times >= cut_lo]
                if cut_hi is not None:
                    spike_times = spike_times[spike_times <= cut_hi]

        for power in powers:
            for site in sites:
                for freq in freqs:
                    if not np.isfinite(freq) or freq <= 0:
                        continue
                    pulse_interval = 1.0 / freq
                    pre_win = pulse_interval * pre_win_ratio
                    for pre_post in pre_posts:
                        # all trials for this condition; the stability window is
                        # applied to train_times below, so that trains whose
                        # spikes were removed are dropped rather than scored as
                        # silence
                        trials = opto_tbl[
                            (opto_tbl['power'] == power)
                            & (opto_tbl['site'] == site)
                            & (opto_tbl['pre_post'] == pre_post)
                            & (opto_tbl['freq'] == freq)
                        ]
                        if len(trials) == 0:
                            continue

                        # number of pulses per train, from the table rather than
                        # hardcoded, falling back to 5 if the column is absent
                        if 'num_pulses' in trials.columns and trials['num_pulses'].notna().any():
                            pulse_num = int(trials['num_pulses'].dropna().iloc[0])
                        else:
                            pulse_num = 5

                        train_times = trials['time'].values.astype(float)
                        # keep only trains fully inside the stability window, so
                        # every pulse index -- and every response window -- is
                        # scored on exactly the same trial set
                        train_span = (pulse_num - 1) * pulse_interval + max(all_post_wins)
                        if cut_lo is not None:
                            train_times = train_times[train_times - pre_win >= cut_lo]
                        if cut_hi is not None:
                            train_times = train_times[train_times + train_span <= cut_hi]

                        row = {
                            'unit_id': unit,
                            'power': power,
                            'site': site,
                            'freq': freq,
                            'pre_post': pre_post,
                            'n_trains': int(len(train_times)),
                            'n_trains_uncut': int(len(trials)),
                            'ephys_cut_lo': cut_lo,
                            'ephys_cut_hi': cut_hi,
                        }

                        # A condition whose every train falls outside the unit's
                        # stability window is untestable, not unresponsive. Emit
                        # it with NaN stats rather than dropping the row: the
                        # drift cut wipes out the whole post-laser block for some
                        # units, and a silently absent row makes that look like a
                        # condition the session never ran.
                        if len(train_times) == 0:
                            for w in all_post_wins:
                                sfx = '' if w == post_win else f'_{int(round(w * 1000))}ms'
                                row[f'p_unit_condition{sfx}'] = [np.nan] * pulse_num
                                row[f'p_sig_count{sfx}'] = np.nan
                                row[f'delta_unit_condition{sfx}'] = [np.nan] * pulse_num
                                row[f'p_sig_count_exc{sfx}'] = np.nan
                                row[f'p_sig_count_inh{sfx}'] = np.nan
                            row['post_win'] = post_win
                            if qc_lookup is not None and unit in qc_lookup.index:
                                for c in qc_cols:
                                    row[c] = qc_lookup.loc[unit, c]
                            rows.append(row)
                            continue

                        # baseline depends only on the pulse interval, so it is
                        # counted once and reused by every response window
                        pre_rates = []
                        post_rates = {w: [] for w in all_post_wins}
                        for pulse_ind in range(pulse_num):
                            pulse_times = train_times + (pulse_ind * pulse_interval)
                            pre_rates.append(_counts_in_win(
                                spike_times, pulse_times, [-pre_win, 0]) / pre_win)
                            for w in all_post_wins:
                                post_rates[w].append(_counts_in_win(
                                    spike_times, pulse_times, [0, w]) / w)

                        for w in all_post_wins:
                            ps, deltas = [], []
                            for pulse_ind in range(pulse_num):
                                a = pre_rates[pulse_ind]
                                b = post_rates[w][pulse_ind]
                                # paired non-parametric test
                                if np.any((a - b) != 0):
                                    stat, p = wilcoxon(a, b)
                                else:
                                    p = 1
                                ps.append(p)
                                deltas.append(float(np.mean(b - a)))
                            p_arr = np.asarray(ps)
                            d_arr = np.asarray(deltas)
                            sig = p_arr < 0.05
                            sfx = '' if w == post_win else f'_{int(round(w * 1000))}ms'
                            row[f'p_unit_condition{sfx}'] = ps
                            row[f'p_sig_count{sfx}'] = int(sig.sum())
                            # signed information, so excitation and suppression
                            # stay distinguishable without a recompute
                            row[f'delta_unit_condition{sfx}'] = deltas
                            row[f'p_sig_count_exc{sfx}'] = int((sig & (d_arr > 0)).sum())
                            row[f'p_sig_count_inh{sfx}'] = int((sig & (d_arr < 0)).sum())
                        row['post_win'] = post_win
                        if qc_lookup is not None and unit in qc_lookup.index:
                            for c in qc_cols:
                                row[c] = qc_lookup.loc[unit, c]
                        rows.append(row)

    opto_sigs = pd.DataFrame(rows)
    # save the results
    if save:
        opto_dir = session_dirs(session)[f'opto_dir_{data_type}']
        os.makedirs(opto_dir, exist_ok=True)
        opto_sigs_file = os.path.join(opto_dir, f'{session}_opto_sigs.pkl')
        with open(opto_sigs_file, 'wb') as f:
            pickle.dump(opto_sigs, f)
    return opto_sigs


# %%
def opto_sig_sessions(data_type='raw', loc='soma', qc_filter=None, n_jobs=-2,
                      sessions=None, overwrite=True, apply_drift_cut=True,
                      post_win=0.025, extra_post_wins=(0.05,)):
    """Run :func:`cal_opto_sigs` over every session that has the opto inputs."""
    from joblib import Parallel, delayed

    if sessions is None:
        session_assets = pd.read_csv('/root/capsule/code/data_management/session_assets.csv')
        sessions = [s for s in session_assets['session_id'] if isinstance(s, str)]

    def process(session):
        opto_dir = session_dirs(session)[f'opto_dir_{data_type}']
        if opto_dir is None:
            return (session, 'skip', 'no opto dir')
        out = os.path.join(opto_dir, f'{session}_opto_sigs.pkl')
        # the opto table is the hard requirement; without it there is nothing to test
        if not os.path.exists(os.path.join(opto_dir, f'{session}_opto_session_{loc}.csv')):
            return (session, 'skip', f'no opto_session_{loc}.csv')
        if os.path.exists(out) and not overwrite:
            return (session, 'skip', 'already exists')
        try:
            sigs = cal_opto_sigs(session, data_type=data_type, loc=loc,
                                 qc_filter=qc_filter,
                                 post_win=post_win,
                                 extra_post_wins=extra_post_wins,
                                 apply_drift_cut=apply_drift_cut)
            plt.close('all')
            n_units = sigs['unit_id'].nunique() if len(sigs) else 0
            return (session, 'ok', f'{len(sigs)} rows, {n_units} units')
        except Exception:
            plt.close('all')
            # report the real failure -- a bare except here hides missing files,
            # bad kwargs and scipy errors behind one indistinguishable message
            return (session, 'error', traceback.format_exc(limit=6))

    results = Parallel(n_jobs=n_jobs)(delayed(process)(s) for s in sessions)

    ok = [r for r in results if r[1] == 'ok']
    skipped = [r for r in results if r[1] == 'skip']
    errored = [r for r in results if r[1] == 'error']
    print(f'\n=== {len(ok)} ok, {len(skipped)} skipped, {len(errored)} errored ===')
    for session, _, msg in errored:
        print(f'\n--- {session} ---\n{msg}')
    return results


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute opto response significance per session.')
    parser.add_argument('--data-type', default='raw', choices=['raw', 'curated'])
    parser.add_argument('--loc', default='soma')
    parser.add_argument('--qc-filter', default=None,
                        help="Comma-separated unit-table QC columns; omit to keep all units.")
    parser.add_argument('--n-jobs', type=int, default=-2)
    parser.add_argument('--sessions', default=None,
                        help='Comma-separated session ids; omit for all in session_assets.csv')
    parser.add_argument('--no-overwrite', action='store_true')
    parser.add_argument('--no-drift-cut', action='store_true')
    parser.add_argument('--post-win', type=float, default=0.025,
                        help='Primary post-pulse window in seconds (default 0.025).')
    parser.add_argument('--extra-post-wins', default='0.05',
                        help='Comma-separated extra windows, suffixed in the output. Empty for none.')
    args = parser.parse_args()

    opto_sig_sessions(
        data_type=args.data_type,
        loc=args.loc,
        qc_filter=args.qc_filter.split(',') if args.qc_filter else None,
        n_jobs=args.n_jobs,
        sessions=args.sessions.split(',') if args.sessions else None,
        overwrite=not args.no_overwrite,
        apply_drift_cut=not args.no_drift_cut,
        post_win=args.post_win,
        extra_post_wins=tuple(float(w) for w in args.extra_post_wins.split(',') if w.strip()),
    )
