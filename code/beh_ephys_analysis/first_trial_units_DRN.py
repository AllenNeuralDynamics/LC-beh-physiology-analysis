# %% Which individual DRN units carry the first-go-cue "surprise" signal?
#
# Each unit has a single trial 1, so a unit is scored by where trial 1 falls in the distribution
#   of its own reference trials NULL (trials 4-53, same unit, same windows):
#   resp      rate in EARLY minus that trial's own BASELINE (the go-cue response)
#   pre       rate in PRE, the 5 s before the go cue (before vs during the task)
#   z_<m>     (trial 1 - mean of NULL) / SD of NULL
#   p_<m>     one-sided empirical p, (1 + #NULL trials >= trial 1) / (n_null + 1); for pre the
#             side is "trial 1 lower" (the pre-task period is quieter on average)
# The smallest possible p is 1 / (n_null + 1), so no single unit is "significant" on its own
#   beyond that; the population question is whether more units have small p than the uniform
#   distribution expected under no first-trial effect, and whether high-z units cluster by
#   session (a shared state such as arousal) or by unit type.
# Units and inclusion as in first_trial_gocue_DRN.py; units with fewer than MIN_NULL reference
#   trials inside their drift cut are dropped.
#
#     python first_trial_units_DRN.py           # compute (cached) + figures + table
#     python first_trial_units_DRN.py --redo

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
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest, kruskal, spearmanr

from unit_summary_DRN import (INK, INK_2, MUTED, TAGGED_UNIT, NONTAGGED_POP, style,
                              nontagged_drift_cut, load_drift)
from unit_summary_DRN_cover import unit_sets
from unit_response_profiles_DRN import TABLE_CSV as PROFILE_CSV
from first_trial_gocue_DRN import FIRST_DIR, WINDOW, BASELINE, EARLY, TRIAL_COLORS
from utils.beh_functions import get_session_tbl, get_unit_tbl, makeSessionDF

CACHE = os.path.join(FIRST_DIR, 'first_trial_units_DRN.pkl')
FIG_PDF = os.path.join(FIRST_DIR, 'first_trial_units_DRN.pdf')
TABLE_CSV = os.path.join(FIRST_DIR, 'first_trial_units_DRN.csv')

PRE = (-5, 0)
NULL = (3, 53)  # 0-based trial slice: trials 4-53
MIN_NULL = 30
P_CUT = 0.05
N_EXAMPLES = 12
PSTH_BIN = 0.1  # s, example-unit PSTHs
N_JOBS = 8


def session_spikes(session, units):
    """{unit_id: per-trial spike times relative to go cue (trials 1..NULL end) or exclusion reason},
    plus the session's per-trial lick latency."""
    tbl = get_session_tbl(session)
    go = np.sort(tbl['goCue_start_time'].dropna().values)[:NULL[1]]
    with_lick = makeSessionDF(session).set_index('go_cue_time')['lick_lat']
    lick_lat = with_lick.reindex(go).values
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
        keep = go if cut[1] is None else go[go + WINDOW[1] <= cut[1]]
        if len(keep) - NULL[0] < MIN_NULL:
            out[unit_id] = f'fewer than {MIN_NULL} reference trials in drift cut'
            continue
        out[unit_id] = [spikes[(spikes >= g + WINDOW[0]) & (spikes < g + WINDOW[1])] - g for g in keep]
    return session, out, lick_lat


def compute():
    tagged, nontagged, _ = unit_sets()
    by_session = {}
    for key in tagged | nontagged:
        by_session.setdefault(key[0], []).append((key[1], key in tagged))
    results = Parallel(n_jobs=N_JOBS)(delayed(session_spikes)(s, u) for s, u in sorted(by_session.items()))
    data = {'units': {}, 'lick_lat': {}}
    for session, units, lick_lat in results:
        data['lick_lat'][session] = lick_lat
        for unit_id, v in units.items():
            data['units'][(session, unit_id)] = {'tagged': (session, unit_id) in tagged, 'trials': v}
    os.makedirs(FIRST_DIR, exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(data, f)
    return data


def rate(trials, win):
    return np.array([np.sum((s >= win[0]) & (s < win[1])) for s in trials]) / (win[1] - win[0])


def score(x, null, side):
    """z of trial 1 against NULL trials and one-sided empirical p (side +1: trial 1 higher)."""
    sd = np.std(null, ddof=1)
    z = (x - np.mean(null)) / sd if sd > 0 else np.nan
    p = (1 + np.sum(side * null >= side * x)) / (len(null) + 1)
    return z, p


def unit_table(data):
    prof = pd.read_csv(PROFILE_CSV).set_index(['session', 'unit_id'])
    rows = []
    for key, d in data['units'].items():
        if isinstance(d['trials'], str):
            continue
        trials = d['trials']
        early, base, pre = rate(trials, EARLY), rate(trials, BASELINE), rate(trials, PRE)
        resp = early - base
        null = slice(NULL[0], None)
        z_resp, p_resp = score(resp[0], resp[null], +1)
        z_pre, p_pre = score(pre[0], pre[null], -1)
        row = {'session': key[0], 'unit_id': key[1], 'tagged': d['tagged'], 'n_null': len(resp[null]),
               'resp_t1': resp[0], 'resp_null_mean': resp[null].mean(), 'resp_null_sd': resp[null].std(ddof=1),
               'z_resp': z_resp, 'p_resp': p_resp,
               'pre_t1': pre[0], 'pre_null_mean': pre[null].mean(), 'z_pre': z_pre, 'p_pre': p_pre,
               'early_t1': early[0], 'early_null_mean': early[null].mean(),
               'lick_lat_t1': data['lick_lat'][key[0]][0],
               'lick_lat_null_median': np.nanmedian(data['lick_lat'][key[0]][NULL[0]:])}
        if key in prof.index:
            p = prof.loc[key]
            row.update({c: p[c] for c in ['profile_go_cue', 'firing_rate', 'lv_isi', 'DV', 'depth']})
        rows.append(row)
    return pd.DataFrame(rows).sort_values('z_resp', ascending=False).reset_index(drop=True)


def excess(p):
    """Observed fraction with p < P_CUT and the binomial p against P_CUT. The empirical p is
    discrete (k / (n_null + 1)), so the true null rate is at most P_CUT and the test is conservative."""
    p = p[np.isfinite(p)]
    return np.mean(p < P_CUT), binomtest(int(np.sum(p < P_CUT)), len(p), P_CUT, alternative='greater').pvalue


def pop_color(tagged):
    return TAGGED_UNIT if tagged else NONTAGGED_POP


def summary_page(tbl):
    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3, left=0.06, right=0.98, top=0.9, bottom=0.08)
    pops = [('Tagged', True), ('Non-tagged', False)]

    # trial-1 response z vs pre-task baseline z
    ax = fig.add_subplot(gs[0, 0])
    for name, t in pops:
        s = tbl[tbl['tagged'] == t]
        ax.scatter(s['z_pre'], s['z_resp'], s=14, color=pop_color(t), alpha=0.7, lw=0,
                   label=f'{name} (n={len(s)})')
    for k, r in tbl.head(N_EXAMPLES).iterrows():
        ax.annotate(str(k + 1), (r['z_pre'], r['z_resp']), fontsize=6, color=INK, xytext=(3, 2),
                    textcoords='offset points')
    ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    rho, p = spearmanr(tbl['z_pre'], tbl['z_resp'], nan_policy='omit')
    ax.set_xlabel(f'Trial-1 pre-cue rate z ({PRE[0]} to {PRE[1]} s)\n(< 0: quieter before the task)')
    ax.set_ylabel('Trial-1 go-cue response z')
    ax.set_title(f'Per-unit trial 1 vs own trials {NULL[0] + 1}-{NULL[1]}\n'
                 f'Spearman rho={rho:.2f}, p={p:.2g}; numbers = example ranks', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    # empirical p histograms vs uniform
    for c, (metric, label) in enumerate([('p_resp', 'go-cue response (trial 1 higher)'),
                                         ('p_pre', 'pre-cue rate (trial 1 lower)')], start=1):
        ax = fig.add_subplot(gs[0, c])
        bins = np.linspace(0, 1, 21)
        lines = []
        for name, t in pops:
            p = tbl.loc[tbl['tagged'] == t, metric].values
            ax.hist(p, bins=bins, histtype='step', lw=2, color=pop_color(t), density=True, label=name)
            frac, pb = excess(p)
            lines.append(f'{name}: {100 * frac:.0f}% p<{P_CUT} (binomial vs {100 * P_CUT:.0f}%: p={pb:.2g})')
        ax.axhline(1, color=INK, ls='--', lw=1, label='uniform (no effect)')
        ax.set_xlabel(f'Empirical p, {label}')
        ax.set_ylabel('Density')
        ax.set_title('\n'.join(lines), fontsize=8.5)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        style(ax)

    # response z by go-cue profile
    ax = fig.add_subplot(gs[1, 0])
    profiles = ['transient excitation', 'sustained excitation', 'inhibited', 'non-responsive']
    rng = np.random.default_rng(0)
    for i, prof in enumerate(profiles):
        for j, (name, t) in enumerate(pops):
            z = tbl.loc[(tbl['profile_go_cue'] == prof) & (tbl['tagged'] == t), 'z_resp'].dropna()
            x = i + (j - 0.5) * 0.35
            ax.scatter(x + rng.uniform(-0.1, 0.1, len(z)), z, s=8, color=pop_color(t), alpha=0.6, lw=0)
            if len(z):
                ax.plot([x - 0.14, x + 0.14], [z.median()] * 2, color=INK, lw=2)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    ax.set_xticks(range(len(profiles)))
    ax.set_xticklabels([p.replace(' ', '\n') for p in profiles], fontsize=8)
    ax.set_ylabel('Trial-1 go-cue response z')
    ax.set_title('By go-cue profile (red tagged, gray non-tagged; bar = median)', fontsize=10)
    style(ax)

    # response z by session: shared session state vs unit-specific
    ax = fig.add_subplot(gs[1, 1:])
    groups = [g for _, g in tbl.groupby('session') if len(g) >= 3]
    order = sorted(groups, key=lambda g: g['z_resp'].median())
    for i, g in enumerate(order):
        ax.scatter(np.full(len(g), i) + rng.uniform(-0.15, 0.15, len(g)), g['z_resp'], s=10,
                   c=[pop_color(t) for t in g['tagged']], alpha=0.7, lw=0)
        ax.plot([i - 0.3, i + 0.3], [g['z_resp'].median()] * 2, color=INK, lw=1.5)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    kw = kruskal(*[g['z_resp'].dropna() for g in order]).pvalue
    lat = pd.DataFrame([{'lat': g['lick_lat_t1'].iloc[0], 'z': g['z_resp'].median()} for g in order])
    rho_l, p_l = spearmanr(lat['lat'], lat['z'], nan_policy='omit')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([g['session'].iloc[0].replace('behavior_', '')[:17] for g in order],
                       rotation=90, fontsize=6)
    ax.set_ylabel('Trial-1 go-cue response z')
    ax.set_title(f'By session (>= 3 units, sorted by median): Kruskal-Wallis across sessions p={kw:.2g}; '
                 f'session median z vs trial-1 lick latency Spearman rho={rho_l:.2f}, p={p_l:.2g}',
                 fontsize=9)
    style(ax)
    fig.suptitle(f'Which units signal the first go cue? Trial 1 scored against each unit\'s own trials '
                 f'{NULL[0] + 1}-{NULL[1]}; response = {EARLY[0]}-{EARLY[1]} s minus {BASELINE[0]}-'
                 f'{BASELINE[1]} s', fontsize=13, color=INK)
    return fig


def example_page(tbl, data, rows, title):
    n_col = 4
    n_row = int(np.ceil(len(rows) / n_col))
    fig = plt.figure(figsize=(17, 5.2 * n_row))
    gs = gridspec.GridSpec(n_row, n_col, hspace=0.45, wspace=0.3, left=0.05, right=0.98,
                           top=1 - 1.1 / (5.2 * n_row), bottom=0.35 / n_row)
    edges = np.arange(WINDOW[0], WINDOW[1] + 1e-9, PSTH_BIN)
    centers = edges[:-1] + PSTH_BIN / 2
    for i, (rank, r) in enumerate(rows.iterrows()):
        sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i // n_col, i % n_col],
                                               height_ratios=[1.2, 1], hspace=0.08)
        trials = data['units'][(r['session'], r['unit_id'])]['trials']
        ax = fig.add_subplot(sub[0])
        # trial 1 on top
        for k, s in enumerate(trials):
            ax.vlines(s, k + 0.5, k + 1.5, color=TRIAL_COLORS[0] if k == 0 else INK, lw=0.6 if k else 1.2)
        ax.axhspan(0.5, 1.5, color=TRIAL_COLORS[2], alpha=0.35, lw=0, zorder=0)
        ax.axhline(NULL[0] + 0.5, color=MUTED, lw=0.6, ls=':')
        ax.set_ylim(len(trials) + 0.5, 0.5)
        ax.set_xlim(*WINDOW)
        ax.axvline(0, color=MUTED, ls='--', lw=1, zorder=0)
        ax.tick_params(labelbottom=False)
        ax.set_ylabel('Trial')
        tag = 'tagged' if r['tagged'] else 'non-tagged'
        ax.set_title(f"#{rank + 1} {r['session'].replace('behavior_', '')}  u{r['unit_id']}  ({tag})\n"
                     f"{r.get('profile_go_cue', '')}; z_resp={r['z_resp']:.1f} (p={r['p_resp']:.2f}), "
                     f"z_pre={r['z_pre']:.1f}", fontsize=8)
        style(ax)

        ax = fig.add_subplot(sub[1], sharex=ax)
        counts = np.vstack([np.histogram(s, edges)[0] for s in trials]) / PSTH_BIN
        null = counts[NULL[0]:]
        m, sd = null.mean(0), null.std(0, ddof=1)
        ax.fill_between(centers, m - sd, m + sd, color=MUTED, alpha=0.25, lw=0)
        ax.plot(centers, m, color=MUTED, lw=1.5, label=f'trials {NULL[0] + 1}-{len(trials)} (mean ± SD)')
        ax.step(centers, counts[0], where='mid', color=TRIAL_COLORS[0], lw=1.5, label='trial 1')
        ax.axvline(0, color=MUTED, ls='--', lw=1, zorder=0)
        ax.axvspan(*EARLY, color=MUTED, alpha=0.08, lw=0, zorder=0)
        ax.set_xlabel('Time from go cue (s)')
        ax.set_ylabel('Rate (Hz)')
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc='upper left')
        style(ax)
    fig.suptitle(title, fontsize=13, color=INK)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    if args.redo or not os.path.exists(CACHE):
        data = compute()
    else:
        with open(CACHE, 'rb') as f:
            data = pickle.load(f)
    excluded = pd.Series([d['trials'] for d in data['units'].values() if isinstance(d['trials'], str)])
    print(f"{len(data['units'])} units, {len(data['units']) - len(excluded)} kept")
    print(excluded.value_counts().to_string())
    tbl = unit_table(data)
    tbl.to_csv(TABLE_CSV, index=False)
    with PdfPages(FIG_PDF) as pdf:
        pdf.savefig(summary_page(tbl))
        pdf.savefig(example_page(tbl, data, tbl.head(N_EXAMPLES),
                                 f'Top {N_EXAMPLES} units by trial-1 go-cue response z (trial 1 = top raster row, red)'))
        tagged = tbl[tbl['tagged']]
        pdf.savefig(example_page(tbl, data, tagged.head(N_EXAMPLES),
                                 f'Top {N_EXAMPLES} tagged units by trial-1 go-cue response z'))
    cols = ['session', 'unit_id', 'tagged', 'profile_go_cue', 'resp_t1', 'resp_null_mean', 'z_resp',
            'p_resp', 'pre_t1', 'pre_null_mean', 'z_pre', 'p_pre', 'lick_lat_t1', 'firing_rate']
    print(tbl[cols].head(20).round(3).to_string())
    for name, t in [('tagged', True), ('non-tagged', False)]:
        s = tbl[tbl['tagged'] == t]
        for m in ['p_resp', 'p_pre']:
            frac, pb = excess(s[m].values)
            print(f'{name} {m}: {100 * frac:.1f}% < {P_CUT} (binomial p={pb:.2g}), n={len(s)}')
    print(FIG_PDF)
