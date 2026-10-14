"""Firing regularity and preferred frequency of tagged DRN units, and their spatial layout.

The hand-picked units in rhythmicity_DRN turned out not to be sustained oscillators: the
ISI shuffle explains their autocorrelogram peaks, so what sets them apart is regular
interspike intervals. This script measures that regularity, and the frequency it settles
at, for every tagged DRN unit in the manuscript spatial figures. It then re-tests the
spatial pattern with the session-aware controls in utils.spatial_session_bias.

Regularity (lower = more clock-like):
    cv_isi    global CV of the ISIs. Inflated by slow rate drift over a long window, so
              it is reported for reference rather than as the primary measure.
    cv2_isi   mean CV of adjacent ISI pairs (Holt et al. 1996). Local, so rate drift
              cancels; ~1 for Poisson, ->0 for a clock.
    lv_isi    local variation (Shinomoto et al. 2003). Also local; separates regular
              (<0.5), random (~1) and bursty (>1) firing, and is the measure used to
              classify firing patterns across cortex. Primary regularity metric.
    acg_peak_amp  height of the first detrended autocorrelogram peak, i.e. how much
              excess firing sits one period after a spike. Grows with both regularity
              and rate, so read it next to lv_isi rather than on its own.
    lv_resid_rate  log(lv_isi) with log(rate) regressed out across the population.
              Refractoriness alone makes fast units look more regular, so this asks
              whether the spatial pattern in regularity is more than a rate pattern.

Preferred frequency:
    pref_freq_isi  1 / mode of the log-ISI histogram: the interval the unit returns to
              most often. For a regular unit this is its firing frequency; for an
              irregular one it is close to meaningless, which is why it is also tested
              restricted to regular units (lv_isi below the population median).
    acg_peak_freq  1 / lag of the first autocorrelogram peak. An independent estimate
              of the same quantity, as a cross-check on pref_freq_isi.

Caveat: 5-HT cells fire doublets, and a short-long alternation drives cv2_isi and lv_isi
up. A unit that is clock-like except for occasional doublets therefore scores as less
regular here than it looks by eye.

Run:  python session_combine/figure_preparation/regularity_spatial.py [--recompute]
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, rankdata, spearmanr

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.beh_functions import get_unit_tbl
from utils.capsule_migration import capsule_directories
from utils.rhythmicity import isi_stats, rhythmicity_metrics
from utils.spatial_session_bias import session_bias_table
from rhythmicity_DRN import UNITS_OF_INTEREST, spontaneous_window, unit_window

DATA_TYPE = 'raw'
BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
N_PERM = 2000
N_BOOT = 1000
SEED = 42
KNN_K = 10  # basic_ephys_all_DRN uses 10; this population is about the same size
N_JOBS = 8

# The ACG is only used for the first-peak lag and height here, so the band is wider than
# rhythmicity_DRN's 2-12 Hz to cover slow units, and there are no ISI shuffles.
ACG_BIN = 0.005
ACG_WINDOW = 4.0
ACG_BAND = (1.0, 20.0)
MIN_SPIKES = 200

capsule_dirs = capsule_directories()
SUMMARY_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']),
                           'SUMMARY_DRN_spatial_organization')
SRC_DIR = os.path.join(SUMMARY_DIR, '05_stats_and_source_data')
OUT_DIR = os.path.join(SUMMARY_DIR, '07_regularity')
METRICS_FILE = os.path.join(OUT_DIR, 'regularity_metrics_DRN.csv')

FEATURES = ['lv_isi', 'cv2_isi', 'cv_isi', 'acg_peak_amp', 'lv_resid_rate',
            'pref_freq_isi', 'acg_peak_freq', 'rate', 'isi_serial_1_det']
# The ones drawn in space; the rest are only in the stat table.
MAP_FEATURES = ['lv_isi', 'lv_resid_rate', 'pref_freq_isi', 'rate']


def unit_key(u):
    return str(int(float(u)))


def load_population():
    """Tagged DRN units in either manuscript spatial dataset, with CCF coordinates.

    The union of the basic-ephys (firing) and waveform populations, so the hand-picked
    units are covered; the membership columns let either subset be re-tested alone.
    """
    be = pd.read_pickle(os.path.join(SRC_DIR, 'features_combined_basic_ephys_all_DRN.pkl'))
    wf = pd.read_csv(os.path.join(SRC_DIR, 'combined_features_waveform_low_DRN.csv'))
    be_keys = set(zip(be['session'], be['unit_id'].map(unit_key)))
    wf_keys = set(zip(wf['session'], wf['unit'].map(unit_key)))

    combined = pd.read_pickle(os.path.join(capsule_dirs['manuscript_fig_prep_dir'],
                                           'combined_unit_tbl', 'combined_unit_tbl.pkl'))
    combined['unit_id'] = combined['unit'].map(unit_key)
    keys = list(zip(combined['session'], combined['unit_id']))
    combined['in_basic_ephys'] = [k in be_keys for k in keys]
    combined['in_waveform'] = [k in wf_keys for k in keys]
    pop = combined[combined['in_basic_ephys'] | combined['in_waveform']]
    pop = pop[['session', 'unit_id', *CCF_COLS, 'in_basic_ephys', 'in_waveform']]
    uoi = {(s, unit_key(u)): band for s, d in UNITS_OF_INTEREST.items()
           for u, band in d.items()}
    pop = pop.assign(hand_picked=[uoi.get(k, '') for k in zip(pop['session'],
                                                                pop['unit_id'])])
    missing = set(uoi) - set(zip(pop['session'], pop['unit_id']))
    if missing:
        print(f'{len(missing)} hand-picked units are not in either spatial dataset: '
              f'{sorted(missing)}')
    return pop.reset_index(drop=True)


def local_variation(isis):
    """Shinomoto et al. 2003: 3/(n-1) * sum(((I_i - I_i+1) / (I_i + I_i+1))^2)."""
    if len(isis) < 3:
        return np.nan
    a, b = isis[:-1], isis[1:]
    return 3.0 * np.mean(((a - b) / (a + b)) ** 2)


def session_metrics(session, unit_ids):
    """Regularity metrics for the given units of one session, in the opto-free window."""
    window = spontaneous_window(session, DATA_TYPE)
    if window is None:
        return []
    unit_tbl = get_unit_tbl(session, DATA_TYPE, summary=False)
    if unit_tbl is None:
        return []
    unit_tbl = unit_tbl.assign(unit_key=unit_tbl['unit_id'].map(unit_key))
    rows = []
    for uid in unit_ids:
        match = unit_tbl[unit_tbl['unit_key'] == uid]
        if len(match) == 0:
            print(f'{session} {uid}: not in the unit table. Skipping.')
            continue
        start, end = unit_window(session, int(uid), DATA_TYPE, *window)
        spikes = np.asarray(match['spike_times'].values[0], float)
        rec = {'session': session, 'unit_id': uid, 'start': start, 'end': end}
        st = isi_stats(spikes, start, end)
        rec.update({k: st[k] for k in ('n_spikes', 'duration', 'rate', 'cv_isi',
                                       'cv2_isi', 'isi_mode', 'isi_serial_1_det',
                                       'isi_serial_2_det')})
        s = np.sort(spikes[(spikes >= start) & (spikes <= end)])
        isis = np.diff(s)
        rec['lv_isi'] = local_variation(isis[isis > 0])
        rec['pref_freq_isi'] = 1.0 / st['isi_mode'] if st['isi_mode'] > 0 else np.nan
        rec['acg_peak_freq'] = np.nan
        rec['acg_peak_amp'] = np.nan
        if st['n_spikes'] >= MIN_SPIKES and end - start > ACG_WINDOW:
            res = rhythmicity_metrics(spikes, bin_size=ACG_BIN, window_length=ACG_WINDOW,
                                      start=start, end=end, band=ACG_BAND,
                                      n_surrogates=0, min_spikes=MIN_SPIKES)
            if np.isfinite(res['first_peak_lag']) and res['first_peak_lag'] > 0:
                rec['acg_peak_freq'] = 1.0 / res['first_peak_lag']
                rec['acg_peak_amp'] = res['first_peak_amp']
        rows.append(rec)
    return rows


def compute_metrics(pop):
    groups = pop.groupby('session')['unit_id'].apply(list)
    out = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(session_metrics)(s, u) for s, u in groups.items())
    m = pd.DataFrame([r for rows in out for r in rows])
    m.loc[m['n_spikes'] < MIN_SPIKES, ['cv_isi', 'cv2_isi', 'lv_isi', 'pref_freq_isi',
                                       'isi_serial_1_det', 'isi_serial_2_det']] = np.nan
    return m


def add_rate_residual(df):
    ok = np.isfinite(df['lv_isi']) & (df['lv_isi'] > 0) & (df['rate'] > 0)
    x, y = np.log(df.loc[ok, 'rate']), np.log(df.loc[ok, 'lv_isi'])
    slope, icpt = np.polyfit(x, y, 1)
    df['lv_resid_rate'] = np.nan
    df.loc[ok, 'lv_resid_rate'] = y - (icpt + slope * x)
    print(f'log(lv) ~ log(rate): slope {slope:.2f}, spearman rho '
          f'{spearmanr(x, y)[0]:.2f}')
    return df


def rank_features(df, features):
    """Rank-transform over non-NaN units, as spatial_session_bias_report does."""
    df = df.copy()
    for f in features:
        v = df[f].to_numpy(float)
        ok = np.isfinite(v)
        r = np.full(len(v), np.nan)
        r[ok] = rankdata(v[ok])
        df[f] = r
    return df


def hand_picked_check(df):
    """Where the hand-picked units fall in each metric's population distribution."""
    hp = df['hand_picked'] != ''
    rows = []
    for f in FEATURES:
        a, b = df.loc[hp, f].dropna(), df.loc[~hp, f].dropna()
        pct = df[f].rank(pct=True)[hp].dropna()
        rows.append({'feature': f, 'n_hand_picked': len(a), 'n_rest': len(b),
                     'median_hand_picked': a.median(), 'median_rest': b.median(),
                     'median_pct_of_hand_picked': pct.median(),
                     'p_mannwhitney': mannwhitneyu(a, b).pvalue if len(a) and len(b)
                     else np.nan})
    return pd.DataFrame(rows)


def to_bregma_mm(df):
    X = df[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    X[:, 0] = np.abs(X[:, 0])
    return X


def make_figure(df, table, path_no_ext):
    X = to_bregma_mm(df)
    hp = (df['hand_picked'] != '').to_numpy()
    n = len(MAP_FEATURES)
    fig, axes = plt.subplots(n + 1, 3, figsize=(13, 3.6 * (n + 1)))

    # Row 0: is the metric picking up what the eye picked up?
    ax = axes[0, 0]
    bins = np.linspace(0, np.nanpercentile(df['lv_isi'], 99), 30)
    ax.hist(df.loc[~hp, 'lv_isi'].dropna(), bins=bins, color='0.7', label='other tagged')
    ax.hist(df.loc[hp, 'lv_isi'].dropna(), bins=bins, color='#3b5ba5', alpha=0.85,
            label='hand-picked')
    ax.axvline(1, color='0.3', lw=1, ls='--')
    ax.text(1, ax.get_ylim()[1] * 0.95, ' Poisson', color='0.3', fontsize=8, va='top')
    ax.set_xlabel('LV (local variation)')
    ax.set_ylabel('units')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Regularity: hand-picked vs rest')

    ax = axes[0, 1]
    ax.scatter(df.loc[~hp, 'rate'], df.loc[~hp, 'lv_isi'], s=14, c='0.6', lw=0)
    ax.scatter(df.loc[hp, 'rate'], df.loc[hp, 'lv_isi'], s=22, c='#3b5ba5', lw=0)
    ax.set_xscale('log')
    ax.set_xlabel('rate (Hz)')
    ax.set_ylabel('LV')
    ax.set_title('Regularity vs rate')

    ax = axes[0, 2]
    lim = [0.3, 30]
    ax.plot(lim, lim, color='0.5', lw=1)
    ax.scatter(df['pref_freq_isi'], df['acg_peak_freq'], s=14,
               c=np.where(hp, '#3b5ba5', '0.6'), lw=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('1 / ISI mode (Hz)')
    ax.set_ylabel('1 / first ACG peak lag (Hz)')
    ax.set_title('Two estimates of preferred frequency')

    # Rows 1..n: the feature in space, plus the session-aware statistics.
    t = table.set_index('feature')
    for i, f in enumerate(MAP_FEATURES, start=1):
        v = df[f].to_numpy(float)
        ok = np.isfinite(v)
        lo, hi = np.nanpercentile(v, [5, 95])
        for j, (a, b, la, lb) in enumerate([(0, 2, 'ML (mm)', 'DV (mm)'),
                                            (1, 2, 'AP (mm)', 'DV (mm)')]):
            ax = axes[i, j]
            sc = ax.scatter(X[ok, a], X[ok, b], c=v[ok], cmap='viridis', vmin=lo, vmax=hi,
                            s=18, edgecolors='white', linewidths=0.4)
            ax.set_xlabel(la)
            ax.set_ylabel(lb)
            if j == 0:
                ax.set_title(f)
            fig.colorbar(sc, ax=ax, shrink=0.8)
        ax = axes[i, 2]
        ax.axis('off')
        if f in t.index and not t.loc[f, 'skipped']:
            r = t.loc[f]
            lines = [f'{f}  (n={int(r.n_units)} units, {int(r.n_sessions)} sessions)', '',
                     f'{"":4s}{"naive":>9s}{"session":>9s}{"within":>9s}{"between":>9s}']
            for nm in ('ML', 'AP', 'DV'):
                lines.append(f'{nm:4s}{r[f"p_{nm}_global_perm"]:9.3f}'
                             f'{r[f"p_{nm}_session_perm"]:9.3f}'
                             f'{r.get(f"p_{nm}_within", np.nan):9.3f}'
                             f'{r.get(f"p_{nm}_between", np.nan):9.3f}')
            lines += ['', f'kNN CV-R2 {r.get("r2_knn_kfold", np.nan):.3f}, '
                      f'p naive {r.get("p_knn_global_perm", np.nan):.3f}, '
                      f'session {r.get("p_knn_session_perm", np.nan):.3f}',
                      '', 'naive: units shuffled freely',
                      'session: session centroids shuffled',
                      'within/between: Mundlak split']
            ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=8,
                    transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path_no_ext + '.png', dpi=150)
    fig.savefig(path_no_ext + '.pdf')
    plt.close(fig)


def main(recompute=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    pop = load_population()
    print(f'{len(pop)} units, {pop["session"].nunique()} sessions, '
          f'{(pop["hand_picked"] != "").sum()} hand-picked')
    if recompute or not os.path.exists(METRICS_FILE):
        m = compute_metrics(pop)
        df = pop.merge(m, on=['session', 'unit_id'], how='inner')
        df = add_rate_residual(df)
        df.to_csv(METRICS_FILE, index=False)
    else:
        df = pd.read_csv(METRICS_FILE, dtype={'unit_id': str})
        df['hand_picked'] = df['hand_picked'].fillna('')
    print(f'metrics for {df["lv_isi"].notna().sum()} units')

    check = hand_picked_check(df)
    check.to_csv(os.path.join(OUT_DIR, 'hand_picked_vs_rest.csv'), index=False)
    print(check.to_string(index=False))

    # Preferred frequency means little for an irregular unit, so also test it among the
    # regular half only.
    regular = df[df['lv_isi'] <= df['lv_isi'].median()].copy()
    regular['pref_freq_isi_regular'] = regular['pref_freq_isi']
    regular['acg_peak_freq_regular'] = regular['acg_peak_freq']
    sets = [('all', df, FEATURES),
            ('regular_half', regular, ['pref_freq_isi_regular', 'acg_peak_freq_regular'])]
    tables = []
    for label, d, feats in sets:
        print(f'\n=== {label}: {len(d)} units')
        table = session_bias_table(rank_features(d, feats), feats, coord_cols=CCF_COLS,
                                   bregma=BREGMA_LPS_MM, permutations=N_PERM,
                                   n_boot=N_BOOT, seed=SEED, knn_k=KNN_K)
        table.insert(0, 'population', label)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(os.path.join(OUT_DIR, 'regularity_spatial_stats.csv'), index=False)

    make_figure(df, tables[0], os.path.join(OUT_DIR, 'regularity_spatial'))

    cols = ['population', 'feature', 'n_units', 'n_sessions',
            'p_ML_global_perm', 'p_ML_session_perm', 'p_AP_global_perm',
            'p_AP_session_perm', 'p_DV_global_perm', 'p_DV_session_perm',
            'p_DV_within', 'p_DV_between', 'r2_knn_kfold', 'p_knn_session_perm']
    print('\n' + table[[c for c in cols if c in table.columns]].round(3).to_string(index=False))
    print(f'\nwrote -> {OUT_DIR}')


if __name__ == '__main__':
    main(recompute='--recompute' in sys.argv)
