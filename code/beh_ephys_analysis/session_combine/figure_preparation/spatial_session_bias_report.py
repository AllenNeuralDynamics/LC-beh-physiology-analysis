"""Session/insertion-bias diagnostics for the DRN anatomical gradient figures.

Re-tests the gradients behind `SUMMARY_DRN_spatial_organization` with nulls and
estimators that respect the fact that units cluster by probe insertion. Writes a
diagnostic figure plus per-feature stat tables next to the summary folder.

Run:  python session_combine/figure_preparation/spatial_session_bias_report.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('/root/capsule/code/beh_ephys_analysis')

from utils.capsule_migration import capsule_directories
from utils.spatial_session_bias import (
    AXIS_NAMES,
    bootstrap_scheme_comparison,
    cluster_bootstrap,
    coordinate_leverage,
    insertion_axis,
    leave_one_group_out,
    mouse_from_session,
    session_bias_table,
    session_centroid_permutation,
    session_confound_screen,
    session_level_regression,
    within_between_regression,
)

BREGMA_LPS_MM = np.array([-5.74, 5.4, -0.45])
CCF_COLS = ['x_ccf', 'y_ccf', 'z_ccf']
N_PERM = 2000
N_BOOT = 1000
SEED = 42

capsule_dirs = capsule_directories()
FIG_DIR = str(capsule_dirs['manuscript_fig_dir'])
SUMMARY_DIR = os.path.join(FIG_DIR, 'SUMMARY_DRN_spatial_organization')
SRC_DIR = os.path.join(SUMMARY_DIR, '05_stats_and_source_data')
OUT_DIR = os.path.join(SUMMARY_DIR, '06_session_bias_controls')
os.makedirs(OUT_DIR, exist_ok=True)

# Features carried in the manuscript panels: the waveform features and PCs shown
# in wf_feature_in_space / PCA_components_in_space, and the task regressors shown
# in model_combined_spatial.
WAVEFORM_FEATURES = [
    'post_w', 'trough_post_ratio_1D', 'post_trough_slope', 'pre_slope',
    'symmetry_slope_div_log', 'symmetry_trough_dis', 'symmetry_inte_div_log',
    'wf_pc_1', 'wf_pc_2', 'wf_pc_3',
]
BEHAVIOR_FEATURES = [
    'T_baseline_svs_hit', 'T_response_svs_hit', 'T_outcome_com_mc',
    'T_outcome_l_mc', 'T_baseline_hit_all', 'T_response_hit_all',
]

# Session-level variables that could plausibly track where the probe went. Any
# of these present in the per-unit table gets averaged within session and
# correlated against session-mean coordinates. Tagging metrics are included for
# tables that carry them; missing columns are skipped.
NUISANCE_COLS = [
    'isi_violations', 'y_loc', 'amp', 'peak',
    'p_max', 'eu', 'corr', 'lat_max_p', 'sig_counts', 'sd',
    'response_rate', 'bl_mean', 'trial_count',
]


def to_bregma_mm(df):
    X = df[CCF_COLS].to_numpy(float) - BREGMA_LPS_MM
    X[:, 0] = np.abs(X[:, 0])
    return X


def load_datasets():
    """(label, per-unit table, feature list) for each dataset we have units for."""
    out = []
    wf = pd.read_csv(os.path.join(SRC_DIR, 'combined_features_waveform_low_DRN.csv'))
    out.append(('waveform_low_DRN', wf,
                [f for f in WAVEFORM_FEATURES if f in wf.columns]))
    beh = pd.read_pickle(os.path.join(SRC_DIR, 'features_combined_beh_all_DRN.pkl'))
    out.append(('beh_all_DRN', beh,
                [f for f in BEHAVIOR_FEATURES if f in beh.columns]))
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(label, df, features, table, path_no_ext):
    X = to_bregma_mm(df)
    s = df['session'].to_numpy()
    mouse = mouse_from_session(s)
    ok = np.isfinite(X).all(axis=1)
    X, s, mouse = X[ok], s[ok], mouse[ok]
    df = df.loc[ok].reset_index(drop=True)

    lev = coordinate_leverage(X, s, mouse)
    ins = insertion_axis(X, s)
    names = list(AXIS_NAMES)

    # the feature with the strongest naive ML effect: the one most at risk
    cand = table[~table['skipped'].fillna(True)].copy()
    focus = (cand.sort_values('p_ML_global_perm').iloc[0]['feature']
             if len(cand) else features[0])
    y_focus = pd.to_numeric(df[focus], errors='coerce').to_numpy(float)

    fig = plt.figure(figsize=(16, 17.5))
    gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.32)

    # (a) within- vs between-session spread per axis
    ax = fig.add_subplot(gs[0, 0])
    w = 0.38
    xs = np.arange(len(names))
    ax.bar(xs - w / 2, lev['sd_within_session_mm'], w, label='within session', color='#4C78A8')
    ax.bar(xs + w / 2, lev['sd_between_session_mm'], w, label='between session', color='#F58518')
    for i, r in lev.iterrows():
        ax.text(i, max(r['sd_within_session_mm'], r['sd_between_session_mm']) * 1.05,
                f"ICC={r['icc_session']:.2f}", ha='center', fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel('coordinate SD (mm)')
    ax.set_title('a. Where each axis gets its spread\n'
                 'high ICC = axis is effectively a session label', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (b) within-session spread per axis, one point per session
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(SEED)
    for j, (nm, col) in enumerate(zip(names, ['#4C78A8', '#54A24B', '#E45756'])):
        sd = pd.Series(X[:, j]).groupby(s).std().dropna()
        ax.scatter(j + rng.uniform(-0.17, 0.17, len(sd)), sd, s=16, alpha=0.65, color=col)
        ax.plot([j - 0.3, j + 0.3], [sd.median()] * 2, color='k', lw=2)
        ax.text(j, ax.get_ylim()[1], f'med {sd.median():.3f}', ha='center',
                va='bottom', fontsize=8)
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel('within-session SD (mm), one point per session')
    ax.set_title('b. Spread available inside one insertion\n'
                 'ML is near-constant within a session', fontsize=10)

    # (c) insertion axis: session centroids
    ax = fig.add_subplot(gs[0, 2])
    cent = ins['centroids']
    nper = pd.Series(1, index=range(len(s))).groupby(s).size().reindex(np.unique(s)).to_numpy()
    ax.scatter(cent[:, 0], cent[:, 2], s=8 * nper, alpha=0.6, color='#333333')
    mu = cent.mean(axis=0)
    for k in range(2):
        v = ins['axes'][k] * 0.4 * np.sqrt(ins['var_frac'][k] / ins['var_frac'][0])
        ax.annotate('', xy=(mu[0] + v[0], mu[2] + v[2]), xytext=(mu[0], mu[2]),
                    arrowprops=dict(arrowstyle='->', lw=2,
                                    color=['#D62728', '#1F77B4'][k]))
        ax.text(mu[0] + v[0], mu[2] + v[2], f"PC{k + 1} ({ins['var_frac'][k]:.0%})",
                fontsize=8, color=['#D62728', '#1F77B4'][k])
    ax.set_xlabel('|ML| (mm)'); ax.set_ylabel('DV (mm)')
    ax.set_title(f"c. Insertion axis ({ins['n_sessions']} session centroids)\n"
                 f"PC1 = ({', '.join(f'{v:.2f}' for v in ins['pc1'])}) in (ML,AP,DV)",
                 fontsize=10)

    # (d) naive vs session-aware p-values
    ax = fig.add_subplot(gs[1, 0])
    for nm, col, mk in zip(names, ['#4C78A8', '#54A24B', '#E45756'], ['o', 's', '^']):
        ax.scatter(cand[f'p_{nm}_global_perm'], cand[f'p_{nm}_session_perm'],
                   s=38, color=col, marker=mk, alpha=0.8, label=nm)
    ax.plot([1e-4, 1], [1e-4, 1], 'k--', lw=0.8)
    ax.axhline(0.05, color='gray', lw=0.6); ax.axvline(0.05, color='gray', lw=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('p, global shuffle (current test)')
    ax.set_ylabel('p, session-restricted shuffle')
    ax.set_title('d. Per-axis p-values, naive vs session-aware\n'
                 'points above the diagonal lose significance', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (e) within- vs between-session coefficients
    ax = fig.add_subplot(gs[1, 1])
    wb = within_between_regression(X, y_focus, s)
    xs = np.arange(len(names))
    for i, r in wb.iterrows():
        ax.errorbar(i - 0.12, r['beta_within'], yerr=r['se_within'] * 1.96,
                    fmt='o', color='#4C78A8', capsize=3,
                    label='within session' if i == 0 else None)
        ax.errorbar(i + 0.12, r['beta_between'], yerr=r['se_between'] * 1.96,
                    fmt='s', color='#F58518', capsize=3,
                    label='between session' if i == 0 else None)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel(f'slope of {focus} (per mm)')
    ax.set_title(f'e. {focus}: gradient split by level\n'
                 'wide within-session CI = no within-insertion leverage', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (f) the between-insertion picture
    ax = fig.add_subplot(gs[1, 2])
    sl = session_level_regression(X, y_focus, s, mouse)
    st = sl['session_table']
    ax.scatter(st['ML'], st['y'], s=6 * st['n_units'], alpha=0.65, color='#333333')
    b = sl['per_axis'].set_index('axis').loc['ML']
    xx = np.linspace(st['ML'].min(), st['ML'].max(), 50)
    ax.plot(xx, st['y'].mean() + b['beta'] * (xx - st['ML'].mean()), 'r--', lw=1.5)
    ax.set_xlabel('session-mean |ML| (mm)')
    ax.set_ylabel(f'session-mean {focus}')
    ax.set_title(f"f. Between-insertion test, n={sl['n_sessions']} sessions / "
                 f"{sl['n_mice']} mice\nML beta={b['beta']:.3g}, "
                 f"p={b['p_value']:.3g} (mouse-clustered)", fontsize=10)

    # (g) leave-one-session-out influence on the ML coefficient
    ax = fig.add_subplot(gs[2, 0])
    jk = leave_one_group_out(X, y_focus, s, mouse, group='session')
    full = jk.attrs['full_coef']['ML']
    bs_ref = cluster_bootstrap(X, y_focus, s, mouse, group='session',
                               n_boot=N_BOOT, seed=SEED).set_index('axis').loc['ML']
    jk = jk.sort_values('beta_ML').reset_index(drop=True)
    ax.axvspan(bs_ref['ci_low'], bs_ref['ci_high'], color='#CCCCCC', alpha=0.6,
               label='bootstrap 95% CI')
    ax.scatter(jk['beta_ML'], np.arange(len(jk)), s=22, color='#4C78A8', zorder=3)
    ax.axvline(full, color='r', ls='--', lw=1.2, label='all sessions', zorder=2)
    ax.axvline(0, color='k', lw=0.8)
    worst = jk.iloc[(jk['beta_ML'] - full).abs().argmax()]
    ax.annotate(str(worst['dropped'])[-19:], xy=(worst['beta_ML'],
                jk.index[jk['dropped'] == worst['dropped']][0]),
                xytext=(5, 0), textcoords='offset points', fontsize=7, va='center')
    ax.set_yticks([]); ax.set_ylabel('session dropped (sorted)')
    ax.set_xlabel('ML slope with one session dropped')
    ax.set_title(f'g. Leave-one-session-out, ML slope of {focus}\n'
                 f"sign flips: {int(jk['sign_flip_ML'].sum())}/{len(jk)}", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    # (h) bootstrap CIs under four resampling schemes
    ax = fig.add_subplot(gs[2, 1])
    schemes = bootstrap_scheme_comparison(X, y_focus, s, mouse, n_boot=N_BOOT, seed=SEED)
    palette = {'unit_only (naive)': '#BBBBBB', 'cluster_session': '#4C78A8',
               'cluster_mouse': '#9467BD', 'hierarchical_mouse_session_unit': '#D62728'}
    offs = np.linspace(-0.26, 0.26, len(palette))
    for off, (scheme, col) in zip(offs, palette.items()):
        sub = schemes[schemes['scheme'] == scheme].set_index('axis')
        for i, nm in enumerate(names):
            r = sub.loc[nm]
            ax.plot([i + off] * 2, [r['ci_low'], r['ci_high']], color=col, lw=2.6,
                    label=scheme.replace('_', ' ') if i == 0 else None)
            ax.plot(i + off, r['estimate'], 'o', color=col, ms=4)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel(f'slope of {focus} (per mm)')
    sr = schemes[schemes['scheme'] == 'hierarchical_mouse_session_unit']['se_ratio_vs_naive']
    ax.set_title('h. Bootstrap 95% CI by resampling scheme\n'
                 f'hierarchical inflates SE {sr.min():.1f}-{sr.max():.1f}x vs naive',
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # (i) does probe placement co-vary with session-level nuisance variables?
    ax = fig.add_subplot(gs[2, 2])
    screen = session_confound_screen(df, coord_cols=CCF_COLS, bregma=BREGMA_LPS_MM,
                                     nuisance_cols=NUISANCE_COLS)
    piv = screen.pivot(index='nuisance', columns='axis', values='spearman_rho')[names]
    pp = screen.pivot(index='nuisance', columns='axis', values='p_value')[names]
    im = ax.imshow(piv.to_numpy(float), cmap='RdBu_r', vmin=-0.7, vmax=0.7, aspect='auto')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(len(piv)):
        for j in range(len(names)):
            star = '*' if pp.iloc[i, j] < 0.05 else ''
            ax.text(j, i, f'{piv.iloc[i, j]:.2f}{star}', ha='center', va='center', fontsize=7.5)
    plt.colorbar(im, ax=ax, label='Spearman rho', fraction=0.046)
    ax.set_title('i. Placement vs session-level nuisance\n'
                 '(* p<0.05, n = sessions)', fontsize=10)

    # (j) stricter null: permute centroids only within mouse
    ax = fig.add_subplot(gs[3, 0])
    strat_rows = []
    for f in cand['feature']:
        yv = pd.to_numeric(df[f], errors='coerce').to_numpy(float)
        pm = session_centroid_permutation(X, yv, s, permutations=N_PERM, seed=SEED,
                                          strata=mouse)
        pa = pm['per_axis'].set_index('axis')
        strat_rows.append({'feature': f,
                           'p_r2_within_mouse_perm': pm['p_r2_session_perm'],
                           **{f'p_{nm}_within_mouse_perm': pa.loc[nm, 'p_session_perm']
                              for nm in names}})
    strat = pd.DataFrame(strat_rows).set_index('feature')
    for nm, col, mk in zip(names, ['#4C78A8', '#54A24B', '#E45756'], ['o', 's', '^']):
        ax.scatter(cand.set_index('feature').loc[strat.index, f'p_{nm}_session_perm'],
                   strat[f'p_{nm}_within_mouse_perm'],
                   s=38, color=col, marker=mk, alpha=0.8, label=nm)
    ax.plot([1e-3, 1], [1e-3, 1], 'k--', lw=0.8)
    ax.axhline(0.05, color='gray', lw=0.6); ax.axvline(0.05, color='gray', lw=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('p, centroids shuffled across all sessions')
    ax.set_ylabel('p, centroids shuffled within mouse')
    ax.set_title('j. Stricter null: also remove between-animal\n'
                 'placement differences', fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # (k) how much of placement is set by the animal
    ax = fig.add_subplot(gs[3, 1])
    me = screen.attrs['mouse_explains_placement'].set_index('axis').reindex(names)
    ax.bar(np.arange(len(names)), me['r2_mouse'], 0.55, color='#9467BD')
    for i, r in enumerate(me.itertuples()):
        ax.text(i, r.r2_mouse + 0.02, f'p={r.p_mouse_f:.3g}', ha='center', fontsize=8)
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_ylabel('R² of session-mean coordinate ~ mouse')
    ax.set_title('k. How much of probe placement is\nfixed by which animal it was',
                 fontsize=10)

    # (l) summary verdict table
    ax = fig.add_subplot(gs[3, 2]); ax.axis('off')
    rows = [['feature', 'ML naive', 'ML sess', 'DV naive', 'DV sess']]
    for _, r in cand.sort_values('p_ML_global_perm').head(11).iterrows():
        rows.append([str(r['feature'])[:22],
                     f"{r['p_ML_global_perm']:.3g}", f"{r['p_ML_session_perm']:.3g}",
                     f"{r['p_DV_global_perm']:.3g}", f"{r['p_DV_session_perm']:.3g}"])
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.3)
    ax.set_title('l. Per-axis p-values, naive vs session-aware', fontsize=10)

    fig.suptitle(f'Session / insertion-bias controls on anatomical gradients — {label} '
                 f'(n={len(df)} units, {len(np.unique(s))} sessions, '
                 f'{len(np.unique(mouse))} mice)', fontsize=13, y=0.99)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_no_ext}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return lev, ins, focus, screen, strat.reset_index(), schemes


def main():
    leverage_rows = []
    for label, df, features in load_datasets():
        print(f'\n=== {label}: {len(df)} units, {len(features)} features', flush=True)
        table = session_bias_table(
            df, features, coord_cols=CCF_COLS, bregma=BREGMA_LPS_MM,
            permutations=N_PERM, n_boot=N_BOOT, seed=SEED)
        table.insert(0, 'dataset', label)

        lev, ins, focus, screen, strat, schemes = make_figure(
            label, df, features, table,
            os.path.join(OUT_DIR, f'session_bias_diagnostics_{label}'))

        # fold the within-mouse null into the main stat table
        table = table.merge(strat, on='feature', how='left')
        table.to_csv(os.path.join(OUT_DIR, f'session_bias_stats_{label}.csv'), index=False)
        screen.insert(0, 'dataset', label)
        screen.to_csv(os.path.join(OUT_DIR, f'placement_confound_screen_{label}.csv'),
                      index=False)
        screen.attrs['session_table'].to_csv(
            os.path.join(OUT_DIR, f'session_placement_table_{label}.csv'), index=False)
        schemes.insert(0, 'dataset', label)
        schemes.insert(1, 'feature', focus)
        schemes.to_csv(
            os.path.join(OUT_DIR, f'bootstrap_scheme_comparison_{label}.csv'), index=False)
        lev.insert(0, 'dataset', label)
        lev['n_units'] = lev.attrs['n_units']
        lev['n_sessions'] = lev.attrs['n_sessions']
        lev['n_mice'] = lev.attrs['n_mice']
        lev['insertion_pc1_ML'], lev['insertion_pc1_AP'], lev['insertion_pc1_DV'] = ins['pc1']
        lev['insertion_pc1_var_frac'] = ins['pc1_var_frac']
        leverage_rows.append(lev)

        n_lost = int(((table['p_ML_global_perm'] < 0.05) &
                      (table['p_ML_session_perm'] >= 0.05)).sum())
        print(f'  ML effects significant naively but not session-aware: {n_lost}'
              f'/{int((table["p_ML_global_perm"] < 0.05).sum())}')
        print(f'  focus feature for the figure: {focus}')

    pd.concat(leverage_rows, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, 'coordinate_leverage.csv'), index=False)
    print(f'\nwrote -> {OUT_DIR}')


if __name__ == '__main__':
    main()
