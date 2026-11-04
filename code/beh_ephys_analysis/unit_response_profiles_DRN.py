# %% Response profiles of tagged and matched non-tagged DRN units, and the waveforms and
#    spike properties of each profile.
#
# Classes (go cue and response, from the per-trial PSTH matrices saved with the unit pages):
#   each window (EARLY, LATE after the event) is up, down or none against the same trials'
#   pre-go-cue baseline: paired Wilcoxon p < ALPHA and |mean change| >= MIN_EFFECT_SD x the
#   trial-to-trial SD of the baseline rate. A significant change smaller than that counts as
#   none, so large trial counts do not turn sub-Hz shifts into responses.
#   profile:  sustained excitation  late up
#             inhibited             early down, or late down without early up
#             transient excitation  early up, late not up
#             non-responsive        neither window changes
#   The full early x late table is kept as well (early_<align>, late_<align> columns).
# Continuous shape: PC1/PC2 of go-cue traces normalized to their peak |change|, fit on all
#   units, so shape rather than amplitude drives the scores.
# Properties: firing rate, LV (Shinomoto 2003, local_variation as in regularity_spatial.py),
#   CV2 and global CV of ISIs in the opto-free window intersected with the unit's drift cut
#   (non-tagged cuts from _opto_drift_tbl_nontagged.csv); trough-to-peak, trough half-width
#   and peak/trough ratio of the mean waveform on its peak channel; SNR; depth; CCF DV.
#
#     python unit_response_profiles_DRN.py          # table + comparison figure

import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)
sys.path.insert(0, os.path.join(curr_path, 'session_combine', 'figure_preparation'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.decomposition import PCA

from unit_summary_DRN import (OUT_DIR, TAGGED_UNIT, NONTAGGED_POP, INK, INK_2, MUTED, style,
                              nontagged_drift_cut, load_drift)
from unit_summary_DRN_cover import unit_sets, load_psths
from utils.beh_functions import get_unit_tbl
from utils.rhythmicity import isi_stats
from rhythmicity_DRN import spontaneous_window
from regularity_spatial import local_variation, MIN_SPIKES

PROFILE_DIR = os.path.join(OUT_DIR, 'response_profiles')
TABLE_CSV = os.path.join(PROFILE_DIR, 'response_profiles_DRN.csv')
FIG_PDF = os.path.join(PROFILE_DIR, 'response_profiles_DRN.pdf')
FLAGS_CSV = os.path.join(OUT_DIR, 'population_unit_flags.csv')
MASTER_CSV = '/root/capsule/scratch/combined/master_unit_tables/master_all_units_opto_ccf_raw_260817.csv'

BASELINE = (-1.5, 0)  # s before the go cue
EARLY = (0, 0.5)
LATE = (0.5, 2.5)
ALPHA = 0.01
MIN_EFFECT_SD = 0.5
MIN_TRIALS = 20
ALIGNS = {'go_cue': 'go_cue_all', 'response': 'response_all'}
PROFILES = ['sustained excitation', 'transient excitation', 'inhibited', 'non-responsive']
PROFILE_COLORS = {'sustained excitation': '#e07b00', 'transient excitation': '#f2c14e',
                  'inhibited': '#1c5cab', 'non-responsive': '#a9a7a1'}
PCA_WINDOW = (-0.5, 2.5)
FS = 30000  # Hz, waveform sampling rate
N_JOBS = 8


def window_rates(p, win):
    t = p['bin_times']
    return np.nanmean(np.asarray(p['matrix'], float)[:, (t >= win[0]) & (t < win[1])], axis=1)


def classify_window(resp, base, sd):
    ok = np.isfinite(resp) & np.isfinite(base)
    resp, base = resp[ok], base[ok]
    if len(resp) < MIN_TRIALS or not np.any(resp != base):
        return 'none', np.nan, np.nan
    p = wilcoxon(resp, base).pvalue
    delta = np.mean(resp - base)
    if p < ALPHA and abs(delta) >= MIN_EFFECT_SD * sd:
        return ('up' if delta > 0 else 'down'), delta, p
    return 'none', delta, p


def profile(early, late):
    if late == 'up':
        return 'sustained excitation'
    if early == 'down' or (late == 'down' and early != 'up'):
        return 'inhibited'
    if early == 'up':
        return 'transient excitation'
    return 'non-responsive'


def classify(d):
    """Per-alignment early/late labels and profile for one unit's saved PSTHs."""
    base_p = d['psth'].get('go_cue_all')
    out = {}
    if base_p is None or base_p['n_trials'] < MIN_TRIALS:
        return out
    base = window_rates(base_p, BASELINE)
    sd = np.nanstd(base, ddof=1)
    out['baseline_rate'] = np.nanmean(base)
    out['baseline_sd'] = sd
    for align, key in ALIGNS.items():
        p = d['psth'].get(key)
        # go-cue and response matrices hold the same trials in the same order (both from the
        # trials with a lick), so the response windows are paired with the go-cue baseline
        if p is None or len(p['matrix']) != len(base):
            continue
        for name, win in [('early', EARLY), ('late', LATE)]:
            lab, delta, pval = classify_window(window_rates(p, win), base, sd)
            out[f'{name}_{align}'] = lab
            out[f'delta_{name}_{align}'] = delta
            out[f'p_{name}_{align}'] = pval
        out[f'profile_{align}'] = profile(out[f'early_{align}'], out[f'late_{align}'])
    return out


def waveform_features(w):
    """Trough-to-peak (ms), trough half-width (ms), peak/trough ratio and amplitude of the mean
    waveform on its peak channel (largest peak-to-peak); positive-going spikes are flipped."""
    w = np.asarray(w, float)
    ch = int(np.nanargmax(np.ptp(w, axis=0)))
    x = w[:, ch]
    if abs(np.nanmax(x)) > abs(np.nanmin(x)):
        x = -x
    i_tr = int(np.nanargmin(x))
    trough = x[i_tr]
    i_pk = i_tr + int(np.nanargmax(x[i_tr:]))
    half = trough / 2
    left = i_tr - np.argmax(x[:i_tr + 1][::-1] > half) if np.any(x[:i_tr] > half) else 0
    right = i_tr + np.argmax(x[i_tr:] > half) if np.any(x[i_tr:] > half) else len(x) - 1
    return {'trough_to_peak_ms': 1000 * (i_pk - i_tr) / FS, 'half_width_ms': 1000 * (right - left) / FS,
            'peak_trough_ratio': x[i_pk] / abs(trough), 'wf_amplitude': x[i_pk] - trough,
            'peak_channel': ch, 'wf_norm': x / abs(trough)}


def session_properties(session, units):
    """Spike-train and waveform properties for the given (unit_id, tagged) of one session."""
    unit_tbl = get_unit_tbl(session, 'raw', summary=False)
    window = spontaneous_window(session, 'raw')
    rows = []
    for unit_id, tagged in units:
        row = {'session': session, 'unit_id': unit_id}
        match = unit_tbl[unit_tbl['unit_id'] == unit_id]
        if len(match) == 0:
            rows.append(row)
            continue
        r = match.iloc[0]
        row.update(waveform_features(r['waveform_mean']))
        if window is not None:
            cut = [None, None]
            drift = load_drift(session, unit_id, 'raw')
            if drift is not None:
                cut = list(drift['ephys_cut'])
            elif not tagged:
                cut = nontagged_drift_cut(session, unit_id)
            start = max(window[0], cut[0]) if cut[0] is not None else window[0]
            end = min(window[1], cut[1]) if cut[1] is not None else window[1]
            st = isi_stats(np.asarray(r['spike_times'], float), start, end)
            spikes = np.sort(np.asarray(r['spike_times'], float))
            isis = np.diff(spikes[(spikes >= start) & (spikes <= end)])
            ok = st['n_spikes'] >= MIN_SPIKES
            row.update({'spont_rate': st['rate'], 'n_spikes_spont': st['n_spikes'],
                        'lv_isi': local_variation(isis[isis > 0]) if ok else np.nan,
                        'cv2_isi': st['cv2_isi'] if ok else np.nan, 'cv_isi': st['cv_isi'] if ok else np.nan})
        rows.append(row)
    return rows


def build_table():
    tagged, nontagged, _ = unit_sets()
    psths = load_psths(tagged, nontagged)
    rows = []
    for key, d in psths.items():
        row = {'session': key[0], 'unit_id': key[1], 'tagged': key in tagged}
        row.update(classify(d))
        rows.append(row)
    tbl = pd.DataFrame(rows)

    # shape scores on peak-normalized go-cue traces
    keys, traces = [], []
    for key, d in psths.items():
        p = d['psth'].get('go_cue_all')
        if p is None or p['n_trials'] < MIN_TRIALS:
            continue
        t = p['bin_times']
        base = window_rates(p, BASELINE)
        y = p['mean'][(t >= PCA_WINDOW[0]) & (t < PCA_WINDOW[1])] - np.nanmean(base)
        if np.nanmax(np.abs(y)) > 0:
            keys.append(key)
            traces.append(y / np.nanmax(np.abs(y)))
    X = np.vstack(traces)
    pca = PCA(n_components=2, random_state=0).fit(X)
    S = pca.transform(X)
    scores = pd.DataFrame({'session': [k[0] for k in keys], 'unit_id': [k[1] for k in keys],
                           'pc1': S[:, 0], 'pc2': S[:, 1]})
    tbl = tbl.merge(scores, on=['session', 'unit_id'], how='left')

    groups = tbl.groupby('session').apply(lambda g: list(zip(g['unit_id'], g['tagged']))).to_dict()
    props = Parallel(n_jobs=N_JOBS)(delayed(session_properties)(s, u) for s, u in groups.items())
    props = pd.DataFrame([r for rows in props for r in rows])
    wf = props[['session', 'unit_id', 'wf_norm']] if 'wf_norm' in props else None
    tbl = tbl.merge(props.drop(columns=['wf_norm'], errors='ignore'), on=['session', 'unit_id'], how='left')
    master = pd.read_csv(MASTER_CSV, usecols=['session', 'unit_id', 'firing_rate', 'snr', 'depth'])
    flags = pd.read_csv(FLAGS_CSV, usecols=['session', 'unit_id', 'DV'])
    tbl = tbl.merge(master, on=['session', 'unit_id'], how='left').merge(flags, on=['session', 'unit_id'], how='left')
    os.makedirs(PROFILE_DIR, exist_ok=True)
    tbl.to_csv(TABLE_CSV, index=False)
    return tbl, wf, psths, pca


FEATURES = [('firing_rate', 'Firing rate (Hz)', 'log'), ('lv_isi', 'LV (ISI)', 'linear'),
            ('cv2_isi', 'CV2 (ISI)', 'linear'), ('trough_to_peak_ms', 'Trough-to-peak (ms)', 'linear'),
            ('half_width_ms', 'Trough half-width (ms)', 'linear'), ('peak_trough_ratio', 'Peak/trough ratio', 'linear'),
            ('snr', 'SNR', 'linear'), ('DV', 'DV from bregma (mm)', 'linear')]


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        order = np.argsort(p[ok])
        ranked = p[ok][order] * ok.sum() / (np.arange(ok.sum()) + 1)
        q_ok = np.minimum.accumulate(ranked[::-1])[::-1]
        q[np.where(ok)[0][order]] = np.minimum(q_ok, 1)
    return q


def compare(tbl, align='go_cue'):
    """Mann-Whitney tests: non-tagged sustained excitation vs non-tagged inhibited, and each of
    those vs tagged units (all profiles), per feature; BH-FDR over all tests."""
    col = f'profile_{align}'
    nt = tbl[~tbl['tagged']]
    groups = {'NT sustained': nt[nt[col] == 'sustained excitation'], 'NT inhibited': nt[nt[col] == 'inhibited'],
              'tagged': tbl[tbl['tagged']]}
    rows = []
    for a, b in [('NT sustained', 'NT inhibited'), ('NT sustained', 'tagged'), ('NT inhibited', 'tagged')]:
        for feat, label, _ in FEATURES:
            x, y = groups[a][feat].dropna(), groups[b][feat].dropna()
            p = mannwhitneyu(x, y).pvalue if len(x) >= 3 and len(y) >= 3 else np.nan
            rows.append({'comparison': f'{a} vs {b}', 'feature': feat, 'n_a': len(x), 'n_b': len(y),
                         'median_a': x.median(), 'median_b': y.median(), 'p': p})
    out = pd.DataFrame(rows)
    out['q_bh'] = bh(out['p'])
    return out


def figure(tbl, wf, psths, stats, out=FIG_PDF):
    fig = plt.figure(figsize=(31, 24))
    fig.text(0.04, 0.985, 'DRN response profiles: tagged vs non-tagged', fontsize=28, color=INK, va='top')
    fig.text(0.04, 0.955, f'Profiles from early ({EARLY[0]}-{EARLY[1]} s) and late ({LATE[0]}-{LATE[1]} s) windows vs '
                          f'the pre-go-cue baseline: paired Wilcoxon p < {ALPHA} and |change| >= {MIN_EFFECT_SD} '
                          f'baseline SD. Sustained = late up; inhibited = early down, or late down without early up; '
                          f'transient = early up, late not up.', fontsize=13, color=INK_2, va='top')
    outer = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.1], hspace=0.3, left=0.04, right=0.98, top=0.92, bottom=0.04)

    # row 1: class counts and mean traces per profile, each alignment, each group
    top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], width_ratios=[0.9, 1, 1, 1, 1], wspace=0.3)
    ax = fig.add_subplot(top[0])
    ax.axis('off')
    lines = ['Profile counts (go cue | response)', '']
    for g, name in [(True, 'tagged'), (False, 'non-tagged')]:
        sub = tbl[tbl['tagged'] == g]
        lines.append(f'{name} (n={len(sub)})')
        for prof in PROFILES:
            lines.append(f'   {prof:<22} {int((sub["profile_go_cue"] == prof).sum()):>4} | '
                         f'{int((sub["profile_response"] == prof).sum()):>4}')
        lines.append('')
    ax.text(0, 1, '\n'.join(lines), family='monospace', fontsize=12, va='top', transform=ax.transAxes)
    for c, (align, key, g, name) in enumerate([('go_cue', 'go_cue_all', True, 'tagged'),
                                              ('go_cue', 'go_cue_all', False, 'non-tagged'),
                                              ('response', 'response_all', True, 'tagged'),
                                              ('response', 'response_all', False, 'non-tagged')]):
        ax = fig.add_subplot(top[c + 1])
        for prof in PROFILES:
            keys = tbl[(tbl['tagged'] == g) & (tbl[f'profile_{align}'] == prof)][['session', 'unit_id']]
            ys = []
            for s, u in keys.itertuples(index=False):
                d = psths[(s, u)]
                p = d['psth'].get(key)
                base = d['psth']['go_cue_all']
                if p is None:
                    continue
                ys.append(p['mean'] - np.nanmean(window_rates(base, BASELINE)))
                t = p['bin_times']
            if ys:
                M = np.vstack(ys)
                m, se = np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(len(M))
                ax.fill_between(t, m - se, m + se, color=PROFILE_COLORS[prof], alpha=0.25, linewidth=0)
                ax.plot(t, m, color=PROFILE_COLORS[prof], lw=1.8, label=f'{prof} (n={len(M)})')
        ax.axvline(0, color=MUTED, ls='--', lw=1)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f'{name}: {align.replace("_", " ")}', fontsize=13)
        ax.set_xlabel(f'Time from {align.replace("_", " ")} (s)')
        if c == 0:
            ax.set_ylabel('Rate - baseline (Hz)')
        ax.legend(fontsize=8, frameon=False)
        style(ax)

    # row 2: mean normalized waveforms per profile (go cue) and PCA shape scores
    mid = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.3)
    wfm = tbl[['session', 'unit_id', 'tagged', 'profile_go_cue']].merge(wf, on=['session', 'unit_id'])
    for c, (g, name) in enumerate([(True, 'tagged'), (False, 'non-tagged')]):
        ax = fig.add_subplot(mid[c])
        for prof in PROFILES:
            w = wfm[(wfm['tagged'] == g) & (wfm['profile_go_cue'] == prof)]['wf_norm'].dropna()
            if len(w) == 0:
                continue
            W = np.vstack(w.values)
            tt = 1000 * (np.arange(W.shape[1]) - 90) / FS
            ax.plot(tt, np.nanmean(W, 0), color=PROFILE_COLORS[prof], lw=1.8, label=f'{prof} (n={len(W)})')
        ax.set_xlim(-1, 2.5)
        ax.set_title(f'{name}: mean waveform by go-cue profile (trough = -1)', fontsize=13)
        ax.set_xlabel('Time from trough (ms)')
        ax.legend(fontsize=8, frameon=False)
        style(ax)
    for c, (g, name) in enumerate([(True, 'tagged'), (False, 'non-tagged')]):
        ax = fig.add_subplot(mid[2 + c])
        sub = tbl[tbl['tagged'] == g]
        for prof in PROFILES:
            s = sub[sub['profile_go_cue'] == prof]
            ax.scatter(s['pc1'], s['pc2'], s=18, color=PROFILE_COLORS[prof], alpha=0.8, linewidth=0, label=prof)
        ax.set_xlabel('PC1 (peak-normalized go-cue trace)')
        ax.set_ylabel('PC2')
        ax.set_title(f'{name}: response shape', fontsize=13)
        ax.legend(fontsize=8, frameon=False)
        style(ax)

    # row 3: feature distributions, NT sustained vs NT inhibited vs tagged
    bot = gridspec.GridSpecFromSubplotSpec(1, len(FEATURES), subplot_spec=outer[2], wspace=0.45)
    nt = tbl[~tbl['tagged']]
    grp = [('NT sustained', nt[nt['profile_go_cue'] == 'sustained excitation'], PROFILE_COLORS['sustained excitation']),
           ('NT inhibited', nt[nt['profile_go_cue'] == 'inhibited'], PROFILE_COLORS['inhibited']),
           ('tagged', tbl[tbl['tagged']], TAGGED_UNIT)]
    rng = np.random.default_rng(0)
    for c, (feat, label, scale) in enumerate(FEATURES):
        ax = fig.add_subplot(bot[c])
        for i, (name, sub, color) in enumerate(grp):
            v = sub[feat].dropna().to_numpy(float)
            if len(v) == 0:
                continue
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=color, alpha=0.6, linewidth=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=INK, lw=2)
        if scale == 'log':
            ax.set_yscale('log')
        ax.set_xticks(range(len(grp)), [f'{n}\n(n={len(s[feat].dropna())})' for n, s, _ in grp], fontsize=8)
        st = stats[stats['feature'] == feat].set_index('comparison')['q_bh']
        ax.set_title(f'{label}\nq: S-I {st.get("NT sustained vs NT inhibited", np.nan):.2g}, '
                     f'S-T {st.get("NT sustained vs tagged", np.nan):.2g}, I-T {st.get("NT inhibited vs tagged", np.nan):.2g}',
                     fontsize=10)
        style(ax)
    fig.text(0.04, 0.012, 'Bottom row: go-cue profiles; lines are medians; q = BH-FDR Mann-Whitney over all features and '
                          'comparisons (S = NT sustained excitation, I = NT inhibited, T = all tagged). LV, CV2 from the '
                          f'opto-free window within each unit\'s drift cut (>= {MIN_SPIKES} spikes).',
             fontsize=11, color=INK_2)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    tbl, wf, psths, pca = build_table()
    print(f'{TABLE_CSV}: {len(tbl)} units; PCA variance explained {pca.explained_variance_ratio_.round(2)}')
    for align in ALIGNS:
        print(f'\n{align} profiles:')
        print(pd.crosstab(tbl[f'profile_{align}'], tbl['tagged'].map({True: 'tagged', False: 'non-tagged'})))
    stats = compare(tbl)
    stats.to_csv(os.path.join(PROFILE_DIR, 'response_profile_stats_go_cue.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\n', stats.round(4).to_string(index=False))
    figure(tbl, wf, psths, stats)
