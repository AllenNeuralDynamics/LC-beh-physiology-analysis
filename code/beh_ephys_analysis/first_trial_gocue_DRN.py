# %% Go-cue "surprise" check: is the go-cue response of DRN units largest on the first trial of
#    the session, whose go cue arrives without any preceding task event?
#
# Trials are the session's first go cues in the raw trial table (get_session_tbl), not the first
#   trials left after the drift cut, so trial 1 is always the first go cue the animal heard. A unit
#   is kept only if its drift cut (tagged: drift pickle; non-tagged: hand-placed nontagged csv)
#   spans WINDOW around trials 1-3. Trials LATER (4-23) are drawn as a reference.
# Units: tagged (TAGGED_CRITERION) and the matched non-tagged selection, split by their go-cue
#   profile from unit_response_profiles_DRN.py (classified on all trials, so trial 1 is included).
# Response per trial: mean rate in EARLY minus that trial's own pre-cue BASELINE. Trial 1 vs
#   trial 2, 3 and the mean of LATER: paired Wilcoxon across units.
# Note: trials are ~6 s apart, so the -5 s pre-cue window of trial 2 onward contains the previous
#   trial's outcome and licking; only trial 1's pre-cue period is task-free.
#
#     python first_trial_gocue_DRN.py           # compute (cached) + figure
#     python first_trial_gocue_DRN.py --redo    # recompute matrices

import os
import sys
import argparse
import pickle

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from unit_summary_DRN import OUT_DIR, INK, INK_2, MUTED, style, nontagged_drift_cut, load_drift
from unit_summary_DRN_cover import unit_sets
from unit_response_profiles_DRN import TABLE_CSV as PROFILE_CSV
from utils.beh_functions import get_session_tbl, get_unit_tbl
from utils.ephys_functions import get_spike_matrix

FIRST_DIR = os.path.join(OUT_DIR, 'first_trial_gocue')
CACHE = os.path.join(FIRST_DIR, 'first_trial_gocue_DRN.pkl')
FIG_PDF = os.path.join(FIRST_DIR, 'first_trial_gocue_DRN.pdf')
STATS_CSV = os.path.join(FIRST_DIR, 'first_trial_gocue_DRN_stats.csv')

WINDOW = (-5, 2)  # s around the go cue
BIN_SIZE = 0.2
STEP_SIZE = 0.05
N_FIRST = 3
LATER = (3, 23)  # 0-based trial slice drawn as the reference (trials 4-23)
BASELINE = (-1.5, 0)
EARLY = (0, 0.5)
TRIAL_COLORS = ['#7a1f0e', '#d6402a', '#f0a99c']  # trial 1 darkest
LATER_COLOR = '#a9a7a1'
GROUPS = [('All units', None),
          ('Go-cue excited', ['transient excitation', 'sustained excitation']),
          ('Go-cue inhibited', ['inhibited']),
          ('Non-responsive', ['non-responsive'])]
N_JOBS = 8


def session_matrices(session, units):
    """{unit_id: (matrix [trials 1-3 + LATER] x bins, bin_times)} for units whose stable period
    covers the analysed trials; excluded units map to the reason."""
    tbl = get_session_tbl(session)
    go = np.sort(tbl['goCue_start_time'].dropna().values)
    go = go[:LATER[1]]
    unit_tbl = get_unit_tbl(session, 'raw', summary=False)
    out = {}
    for unit_id, tagged in units:
        match = unit_tbl[unit_tbl['unit_id'] == unit_id]
        if len(match) == 0:
            out[unit_id] = 'not in unit table'
            continue
        spikes = np.sort(np.asarray(match.iloc[0]['spike_times'], float))
        drift = load_drift(session, unit_id, 'raw')
        cut = list(drift['ephys_cut']) if drift is not None else (
            [None, None] if tagged else nontagged_drift_cut(session, unit_id))
        start = cut[0] if cut[0] is not None else spikes[0]
        if start > go[0] + WINDOW[0]:
            out[unit_id] = 'drift cut / recording starts after trial 1 window'
            continue
        if cut[1] is not None and cut[1] < go[N_FIRST - 1] + WINDOW[1]:
            out[unit_id] = 'drift cut ends before trial 3'
            continue
        # later trials past the cut end are dropped from the reference only
        keep = go if cut[1] is None else go[go + WINDOW[1] <= cut[1]]
        matrix, bin_times = get_spike_matrix(spikes, keep, pre_event=WINDOW[0], post_event=WINDOW[1],
                                             binSize=BIN_SIZE, stepSize=STEP_SIZE)
        out[unit_id] = (matrix.astype(np.float32), bin_times)
    return session, out


def compute():
    tagged, nontagged, _ = unit_sets()
    by_session = {}
    for key in tagged | nontagged:
        by_session.setdefault(key[0], []).append((key[1], key in tagged))
    results = Parallel(n_jobs=N_JOBS)(delayed(session_matrices)(s, u) for s, u in sorted(by_session.items()))
    data = {}
    for session, units in results:
        for unit_id, v in units.items():
            data[(session, unit_id)] = {'tagged': (session, unit_id) in tagged, 'result': v}
    os.makedirs(FIRST_DIR, exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(data, f)
    return data


def win_mean(matrix, t, win):
    return np.nanmean(matrix[:, (t >= win[0]) & (t < win[1])], axis=1)


def unit_rows(data):
    """Per-unit traces (trials 1-3, mean of LATER) and EARLY - BASELINE responses."""
    prof = pd.read_csv(PROFILE_CSV).set_index(['session', 'unit_id'])['profile_go_cue']
    rows = []
    for key, d in data.items():
        if isinstance(d['result'], str):
            continue
        matrix, t = d['result']
        resp = win_mean(matrix, t, EARLY) - win_mean(matrix, t, BASELINE)
        later = slice(LATER[0], LATER[1])
        rows.append({'session': key[0], 'unit_id': key[1], 'tagged': d['tagged'],
                     'profile': prof.get(key, np.nan), 'bin_times': t,
                     'traces': [matrix[i] for i in range(N_FIRST)] + [np.nanmean(matrix[later], axis=0)],
                     'resp': list(resp[:N_FIRST]) + [np.nanmean(resp[later])],
                     'n_later': len(matrix[later])})
    return pd.DataFrame(rows)


def compare(resp):
    """Paired Wilcoxon of trial 1 against trials 2, 3 and LATER."""
    out = {}
    for j, name in zip(range(1, N_FIRST + 1), ['t2', 't3', 'later']):
        a, b = resp[:, 0], resp[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        out[f'p_t1_vs_{name}'] = wilcoxon(a[ok], b[ok]).pvalue if ok.sum() >= 5 and np.any(a[ok] != b[ok]) else np.nan
        out[f'frac_t1_gt_{name}'] = np.mean(a[ok] > b[ok]) if ok.any() else np.nan
    return out


def p_text(p):
    return 'n/a' if not np.isfinite(p) else (f'p={p:.2g}' if p >= 1e-3 else 'p<0.001')


def plot(tbl):
    labels = [f'Trial {i + 1}' for i in range(N_FIRST)] + [f'Trials {LATER[0] + 1}-{LATER[1]} (mean)']
    colors = TRIAL_COLORS + [LATER_COLOR]
    pops = [('Tagged', True), ('Non-tagged', False)]
    fig = plt.figure(figsize=(4.2 * len(GROUPS), 13))
    gs = gridspec.GridSpec(2 * len(pops), len(GROUPS), height_ratios=[1.3, 1] * len(pops),
                           hspace=0.55, wspace=0.3, left=0.05, right=0.98, top=0.92, bottom=0.05)
    stats = []
    for r, (pop, is_tagged) in enumerate(pops):
        for c, (gname, profiles) in enumerate(GROUPS):
            sub = tbl[tbl['tagged'] == is_tagged]
            if profiles is not None:
                sub = sub[sub['profile'].isin(profiles)]
            ax = fig.add_subplot(gs[2 * r, c])
            ax_r = fig.add_subplot(gs[2 * r + 1, c])
            for ax_ in (ax, ax_r):
                style(ax_)
            ax.set_title(f'{pop}: {gname} (n={len(sub)})', fontsize=10, color=INK)
            if len(sub) == 0:
                continue
            t = sub['bin_times'].iloc[0]
            # draw the reference first so trial 1 sits on top
            for k in reversed(range(N_FIRST + 1)):
                y = np.vstack([tr[k] for tr in sub['traces']])
                m = np.nanmean(y, axis=0)
                se = np.nanstd(y, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(y), axis=0))
                ax.fill_between(t, m - se, m + se, color=colors[k], alpha=0.2, lw=0)
                ax.plot(t, m, color=colors[k], lw=2 if k == 0 else 1.5, label=labels[k])
            ax.axvline(0, color=MUTED, ls='--', lw=1, zorder=0)
            ax.axvspan(*EARLY, color=MUTED, alpha=0.08, lw=0, zorder=0)
            ax.set_xlim(*WINDOW)
            ax.set_xlabel('Time from go cue (s)')
            if c == 0:
                ax.set_ylabel('Firing rate (Hz)')
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False, loc='upper left')

            # per-unit EARLY - BASELINE response by trial
            resp = np.vstack(sub['resp'].values)
            x = np.arange(N_FIRST + 1)
            for row in resp:
                ax_r.plot(x, row, color=MUTED, lw=0.4, alpha=0.35, zorder=1)
            m = np.nanmean(resp, axis=0)
            se = np.nanstd(resp, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(resp), axis=0))
            ax_r.errorbar(x, m, yerr=se, color=INK, lw=1.5, zorder=3)
            ax_r.scatter(x, m, c=colors, s=40, zorder=4, edgecolor='white', linewidth=1)
            ax_r.axhline(0, color=MUTED, lw=0.8, zorder=0)
            lo, hi = np.nanpercentile(resp, [2, 98])
            pad = 0.1 * (hi - lo) if hi > lo else 1
            ax_r.set_ylim(min(lo, m.min()) - pad, max(hi, m.max()) + pad)
            ax_r.set_xticks(x)
            ax_r.set_xticklabels(['1', '2', '3', f'{LATER[0] + 1}-{LATER[1]}'])
            ax_r.set_xlabel('Trial')
            if c == 0:
                ax_r.set_ylabel(f'Rate {EARLY[0]}-{EARLY[1]} s minus\n{BASELINE[0]}-{BASELINE[1]} s (Hz)')
            s = compare(resp)
            ax_r.set_title(f"1 vs 2: {p_text(s['p_t1_vs_t2'])}   1 vs 3: {p_text(s['p_t1_vs_t3'])}   "
                           f"1 vs later: {p_text(s['p_t1_vs_later'])}", fontsize=7.5, color=INK_2)
            stats.append({'population': pop, 'group': gname, 'n_units': len(sub),
                          **{f'mean_resp_{lab}': v for lab, v in zip(['t1', 't2', 't3', 'later'], m)}, **s})
    fig.suptitle('First go cues of the session: PSTH (mean ± SEM across units) and per-unit '
                 f'go-cue response ({EARLY[0]}-{EARLY[1]} s minus each trial\'s own pre-cue '
                 f'{BASELINE[0]}-{BASELINE[1]} s; paired Wilcoxon)', fontsize=12, color=INK)
    fig.text(0.5, 0.955, 'Trial 2+ pre-cue windows contain the previous trial\'s outcome and licks '
             '(~6 s inter-trial interval); only trial 1\'s pre-cue period is task-free.',
             ha='center', fontsize=9, color=INK_2)
    return fig, pd.DataFrame(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    if args.redo or not os.path.exists(CACHE):
        data = compute()
    else:
        with open(CACHE, 'rb') as f:
            data = pickle.load(f)
    excluded = pd.Series([d['result'] for d in data.values() if isinstance(d['result'], str)])
    print(f'{len(data)} units, {len(data) - len(excluded)} kept')
    print(excluded.value_counts().to_string())
    tbl = unit_rows(data)
    fig, stats = plot(tbl)
    fig.savefig(FIG_PDF)
    stats.to_csv(STATS_CSV, index=False)
    print(stats.round(4).to_string())
    print(FIG_PDF)
