"""Compare waveform shape of opto-tagged vs non-tagged DRN units.

Groups (criteria: waveform_low_DRN):
  tagged     : passes all criteria                         (apply_qc 'selected')
  non-tagged : passes non-opto criteria only, not tagged   ('selected_no_opto' & ~'selected')
               i.e. p_max, lat_max_p, eu, corr are ignored.

Two waveform sources are analysed separately:
  wf_raw : raw_fake template (same as waveform_generation_np.py / combined_features.csv).
           Missing for one session (behavior_835444_2026-02-17), all-zero for some tagged units.
  wf     : sorter template, available for every unit (robustness check).

Outputs (in <manuscript_fig_dir>/waveform_tagged_vs_untagged/):
  features_<src>.csv        per-unit features + group + session
  feature_stats_<src>.csv   per-feature group comparison
  summary_<src>.pdf         mean waveforms, feature distributions, PCA
"""
import sys
import os
import json
import pickle

sys.path.append('/root/capsule/code/beh_ephys_analysis')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu, zscore
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from utils.combine_tools import apply_qc
from utils.capsule_migration import capsule_directories

capsule_dirs = capsule_directories()

criteria_name = 'waveform_low_DRN'
UNIT_KEY = ['session', 'unit']
PEAK_IND = 30
FS = 30000
N_PERM = 5000
RNG = np.random.default_rng(0)
COLORS = {'tagged': '#2a78d6', 'non-tagged': '#eb6834'}
GROUPS = ['tagged', 'non-tagged']
focus_features = [
    'post_w',
    'trough_post_ratio_1D',
    'post_trough_slope',
    'pre_slope',
    'symmetry_slope_div_log',
    'symmetry_trough_dis',
    'symmetry_inte_div_log',
]
SOURCES = ['wf_raw', 'wf']

out_dir = os.path.join(capsule_dirs['manuscript_fig_dir'], 'waveform_tagged_vs_untagged')
os.makedirs(out_dir, exist_ok=True)


def waveform_features(wf):
    """Same feature definitions as waveform_generation_np.py, for a single 1D waveform.
    peak = trough of the baseline-subtracted waveform; this reproduces combined_features.csv
    (the current waveform_generation_np.py uses peak_raw - baseline, which double-subtracts)."""
    wf = np.asarray(wf, dtype=float)
    wf = wf - np.nanmean(wf[:5])
    peak_ind = np.argmin(wf)
    peak = wf[peak_ind]
    if np.abs(wf[0] - wf[-1]) > 50 or peak_ind == 0:
        return None

    if peak < 0:
        post_trough = np.max(wf[peak_ind:])
        post_trough_ind = np.argmax(wf[peak_ind:]) + 1
        pre_trough = np.max(wf[:peak_ind])
        pre_trough_ind = peak_ind - np.argmax(wf[:peak_ind])
    else:
        post_trough = np.min(wf[peak_ind:])
        post_trough_ind = np.argmin(wf[peak_ind:]) + 1
        pre_trough = np.min(wf[:peak_ind])
        pre_trough_ind = peak_ind - np.argmin(wf[:peak_ind])
    slope_post = ((peak - post_trough) / peak) / post_trough_ind
    inte_post = (post_trough / peak) * post_trough_ind
    slope_pre = ((peak - pre_trough) / peak) / pre_trough_ind
    inte_pre = (pre_trough / peak) * pre_trough_ind
    pre_peak_wf = wf[:peak_ind]
    post_peak_wf = wf[peak_ind:]
    pre_space_raw = np.sum(pre_peak_wf[pre_peak_wf > 0])
    post_space_raw = np.sum(post_peak_wf[post_peak_wf > 0])

    crossings = np.where(np.diff(np.sign(wf - peak / 2)) != 0)[0]
    if not (np.any(crossings < peak_ind) and np.any(crossings > peak_ind)):
        post_crossing = pre_crossing = np.nan
    else:
        post_crossing = np.min(crossings[crossings > peak_ind]) - peak_ind
        pre_crossing = peak_ind - np.max(crossings[crossings < peak_ind])

    with np.errstate(divide='ignore', invalid='ignore'):
        feats = {
            'half_w': post_crossing + pre_crossing,
            'trough_w': post_trough_ind + pre_trough_ind,
            'pre_half': pre_crossing,
            'post_half': post_crossing,
            'post_w': post_trough_ind,
            'pre_w': pre_trough_ind,
            'trough_post_ratio_1D': post_trough / peak,
            'trough_pre_ratio_1D': pre_trough / peak,
            'post_slope': slope_post,
            'post_trough_slope': (post_trough / peak) / post_trough_ind,
            'post_space': inte_post,
            'pre_slope': slope_pre,
            'pre_space': inte_pre,
            'symmetry_inte': inte_post - inte_pre,
            'symmetry_half': post_crossing - pre_crossing,
            'symmetry_slope': slope_post - slope_pre,
            'symmetry_inte_div': inte_post / inte_pre,
            'symmetry_half_div': post_crossing / pre_crossing,
            'symmetry_slope_div': slope_pre / slope_post,
            'symmetry_space_raw_div': post_space_raw / pre_space_raw,
            'symmetry_trough_dis': post_trough_ind - pre_trough_ind,
            'trough_sum': (pre_space_raw + post_space_raw) / peak,
            'slope_sum': slope_post + slope_pre,
        }
        for k in ['symmetry_inte_div', 'symmetry_slope_div', 'symmetry_half_div']:
            feats[f'{k}_log'] = np.log(feats[k] + 1e-6)
    return feats


def valid_wf(x):
    x = np.asarray(x, dtype=float) if x is not None else np.array([])
    return x.size > PEAK_IND and np.isfinite(x).all() and np.nanmax(np.abs(x)) > 0


def auc(a, b):
    """P(tagged > non-tagged) + 0.5 P(tie); 0.5 = no difference."""
    return mannwhitneyu(a, b).statistic / (len(a) * len(b))


def within_session_perm(df, feature):
    """Permutation test that shuffles group labels only within sessions containing both groups.
    Statistic: mean within-session percentile rank of tagged minus non-tagged (0 = no difference).
    Controls for session-level differences (probe, drift, unit composition)."""
    d = df[['session', 'group', feature]].dropna()
    d = d[d.groupby('session')['group'].transform('nunique') == 2]
    if d['group'].value_counts().min() < 5:
        return np.nan, np.nan, 0
    pct = d.groupby('session')[feature].rank(pct=True).to_numpy()
    is_tag = (d['group'] == 'tagged').to_numpy()
    sess_idx = [np.flatnonzero(d['session'].to_numpy() == s) for s in d['session'].unique()]

    def stat(lbl):
        return pct[lbl].mean() - pct[~lbl].mean()

    obs = stat(is_tag)
    null = np.empty(N_PERM)
    lbl = is_tag.copy()
    for i in range(N_PERM):
        for idx in sess_idx:
            lbl[idx] = RNG.permutation(lbl[idx])
        null[i] = stat(lbl)
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (N_PERM + 1)
    return obs, p, len(sess_idx)


# ---------------- Load + QC ----------------
with open(os.path.join(capsule_dirs['manuscript_fig_prep_dir'], 'combined_unit_tbl', 'combined_unit_tbl.pkl'), 'rb') as f:
    combined_tagged_units = pickle.load(f)
with open(os.path.join('/root/capsule/code/beh_ephys_analysis/session_combine/metrics', f'{criteria_name}.json')) as f:
    constraints = json.load(f)
_, combined_tagged_units, fig, _ = apply_qc(combined_tagged_units, constraints)
plt.close(fig)

units = combined_tagged_units.loc[combined_tagged_units['selected_no_opto']].copy()
units['group'] = np.where(units['selected'], 'tagged', 'non-tagged')
print('Units passing non-opto criteria:', units['group'].value_counts().to_dict())

t_ms = None
for wf_col in SOURCES:
    # ---------------- Features ----------------
    rows, wfs_nl = [], []
    for _, r in units.iterrows():
        if not valid_wf(r[wf_col]):
            continue
        feats = waveform_features(r[wf_col])
        if feats is None:
            continue
        wf = np.asarray(r[wf_col], dtype=float)
        wf = wf - np.nanmean(wf[:5])
        wfs_nl.append(wf / np.abs(wf[PEAK_IND]))
        rows.append({'session': r['session'], 'unit': r['unit'], 'group': r['group'],
                     'amp': r['amp'], 'peak': r['peak'], 'fr': r['fr'], **feats})
    df = pd.DataFrame(rows)
    wfs_nl = np.stack(wfs_nl)
    t_ms = (np.arange(wfs_nl.shape[1]) - PEAK_IND) / FS * 1000
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df.to_csv(os.path.join(out_dir, f'features_{wf_col}.csv'), index=False)
    n_per = df['group'].value_counts().to_dict()
    n_sess = df.groupby('group')['session'].nunique().to_dict()
    print(f'\n===== {wf_col}: {n_per} units, sessions {n_sess} =====')

    # ---------------- Per-feature stats ----------------
    feature_cols = ['amp', 'peak'] + [c for c in df.columns if c not in UNIT_KEY + ['group', 'amp', 'peak', 'fr']]
    stats_rows = []
    for feat in feature_cols:
        a = df.loc[df.group == 'tagged', feat].dropna().to_numpy()
        b = df.loc[df.group == 'non-tagged', feat].dropna().to_numpy()
        p_mw = mannwhitneyu(a, b).pvalue
        ws_stat, ws_p, ws_n = within_session_perm(df, feat)
        stats_rows.append({'feature': feat, 'is_focus_feature': feat in focus_features,
                           'n_tagged': len(a), 'n_non_tagged': len(b),
                           'median_tagged': np.median(a), 'median_non_tagged': np.median(b),
                           'auc_tagged_gt_non': auc(a, b), 'p_mannwhitney': p_mw,
                           'within_session_rank_diff': ws_stat, 'p_within_session': ws_p,
                           'n_mixed_sessions': ws_n})
    stats = pd.DataFrame(stats_rows)
    stats['q_mannwhitney'] = multipletests(stats['p_mannwhitney'], method='fdr_bh')[1]
    ok = stats['p_within_session'].notna()
    stats.loc[ok, 'q_within_session'] = multipletests(stats.loc[ok, 'p_within_session'], method='fdr_bh')[1]
    stats['abs_effect'] = (stats['auc_tagged_gt_non'] - 0.5).abs()
    stats = stats.sort_values('abs_effect', ascending=False).drop(columns='abs_effect')
    stats.to_csv(os.path.join(out_dir, f'feature_stats_{wf_col}.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 20, 'display.precision', 3):
        print(stats[['feature', 'median_tagged', 'median_non_tagged', 'auc_tagged_gt_non',
                     'q_mannwhitney', 'within_session_rank_diff', 'q_within_session']].to_string(index=False))

    # ---------------- Multivariate separability ----------------
    fm = df.dropna(subset=focus_features).reset_index(drop=True)
    X = fm[focus_features].to_numpy()
    y = (fm['group'] == 'tagged').to_numpy().astype(int)
    groups = fm['session'].to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    cv_auc = roc_auc_score(y, cross_val_predict(clf, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1])
    null_auc = []
    for i in range(200):
        y_sh = y.copy()
        for s in np.unique(groups):  # shuffle within session to keep session composition
            idx = np.flatnonzero(groups == s)
            y_sh[idx] = RNG.permutation(y_sh[idx])
        null_auc.append(roc_auc_score(y_sh, cross_val_predict(clf, X, y_sh, cv=cv, groups=groups,
                                                              method='predict_proba')[:, 1]))
    p_auc = (np.sum(np.array(null_auc) >= cv_auc) + 1) / (len(null_auc) + 1)
    print(f'Session-held-out logistic AUC on focus features: {cv_auc:.3f} '
          f'(within-session shuffle null 95th pct {np.percentile(null_auc, 95):.3f}, p={p_auc:.3f})')

    # ---------------- Figure ----------------
    with PdfPages(os.path.join(out_dir, f'summary_{wf_col}.pdf')) as pdf:
        # Page 1: mean normalized waveforms
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for g, ls in zip(GROUPS, ['-', '--']):
            m = (df['group'] == g).to_numpy()
            mu = np.nanmean(wfs_nl[m], axis=0)
            sem = np.nanstd(wfs_nl[m], axis=0) / np.sqrt(m.sum())
            med = np.nanmedian(wfs_nl[m], axis=0)
            axes[0].fill_between(t_ms, mu - sem, mu + sem, color=COLORS[g], alpha=0.25, lw=0)
            axes[0].plot(t_ms, mu, color=COLORS[g], lw=2, ls=ls, label=f'{g} (n={m.sum()})')
            axes[1].plot(t_ms, med, color=COLORS[g], lw=2, ls=ls, label=g)
        diff = np.nanmean(wfs_nl[df.group.eq('tagged').to_numpy()], 0) - \
            np.nanmean(wfs_nl[df.group.eq('non-tagged').to_numpy()], 0)
        boot = []
        idx_t, idx_n = np.flatnonzero(df.group.eq('tagged')), np.flatnonzero(df.group.eq('non-tagged'))
        for _ in range(1000):
            boot.append(np.nanmean(wfs_nl[RNG.choice(idx_t, len(idx_t))], 0) -
                        np.nanmean(wfs_nl[RNG.choice(idx_n, len(idx_n))], 0))
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        axes[2].fill_between(t_ms, lo, hi, color='#52514e', alpha=0.25, lw=0, label='95% bootstrap CI')
        axes[2].plot(t_ms, diff, color='#0b0b0b', lw=2, label='tagged − non-tagged')
        axes[2].axhline(0, color='#52514e', lw=0.8)
        for ax, title in zip(axes, ['Mean ± SEM', 'Median', 'Difference of means']):
            ax.set_title(title, loc='left')
            ax.set_xlabel('Time from trough (ms)')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel('Normalized amplitude (trough = −1)')
        axes[0].legend(frameon=False)
        axes[1].legend(frameon=False)
        axes[2].legend(frameon=False, fontsize=8)
        fig.suptitle(f'{wf_col}: tagged vs non-tagged ({criteria_name}, non-opto criteria for non-tagged)')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: feature distributions, sorted by effect size
        feats_plot = stats['feature'].tolist()
        ncol = 6
        nrow = int(np.ceil(len(feats_plot) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow))
        axes = axes.ravel()
        for ax, feat in zip(axes, feats_plot):
            data = [df.loc[df.group == g, feat].dropna().to_numpy() for g in GROUPS]
            # clip extreme tails for display only
            lo_, hi_ = np.nanpercentile(np.concatenate(data), [1, 99])
            bp = ax.boxplot([np.clip(d, lo_, hi_) for d in data], widths=0.5, showfliers=False,
                            patch_artist=True, medianprops={'color': '#0b0b0b'})
            for patch, g in zip(bp['boxes'], GROUPS):
                patch.set_facecolor(COLORS[g])
                patch.set_alpha(0.35)
            for i, (d, g, mk) in enumerate(zip(data, GROUPS, ['o', '^'])):
                ax.scatter(i + 1 + RNG.uniform(-0.15, 0.15, len(d)), np.clip(d, lo_, hi_), s=8,
                           color=COLORS[g], marker=mk, alpha=0.6, lw=0)
            s = stats.set_index('feature').loc[feat]
            ws = '' if np.isnan(s['q_within_session']) else f"\nwithin-sess q={s['q_within_session']:.2g}"
            ax.set_title(f"{feat}{' *' if s['is_focus_feature'] else ''}\nAUC={s['auc_tagged_gt_non']:.2f} "
                         f"q={s['q_mannwhitney']:.2g}{ws}", fontsize=8, loc='left')
            ax.set_xticks([1, 2], ['tagged', 'non-tagged'], fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
        for ax in axes[len(feats_plot):]:
            ax.axis('off')
        fig.suptitle(f'{wf_col}: features sorted by |AUC − 0.5|  (* = focus feature; q = BH-FDR)')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: PCA of focus features
        Z = zscore(X, axis=0)
        pca = PCA(n_components=3).fit(Z)
        pcs = pca.transform(Z)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (i, j) in zip(axes[:2], [(0, 1), (0, 2)]):
            for g, mk in zip(GROUPS, ['o', '^']):
                m = fm['group'].to_numpy() == g
                ax.scatter(pcs[m, i], pcs[m, j], s=14, marker=mk, color=COLORS[g], alpha=0.6,
                           edgecolor='#fcfcfb', lw=0.5, label=f'{g} (n={m.sum()})')
            ax.set_xlabel(f'PC{i + 1} ({pca.explained_variance_ratio_[i]:.0%})')
            ax.set_ylabel(f'PC{j + 1} ({pca.explained_variance_ratio_[j]:.0%})')
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(frameon=False)
        axes[2].hist(null_auc, bins=20, color='#52514e', alpha=0.5, label='within-session shuffle')
        axes[2].axvline(cv_auc, color='#0b0b0b', lw=2, label=f'observed {cv_auc:.2f} (p={p_auc:.3f})')
        axes[2].set_xlabel('Session-held-out AUC (logistic, focus features)')
        axes[2].set_xlim(min(np.min(null_auc), cv_auc) - 0.05, 1)
        axes[2].legend(frameon=False, fontsize=8, loc='upper left')
        axes[2].spines[['top', 'right']].set_visible(False)
        fig.suptitle(f'{wf_col}: multivariate separability')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print('\nWrote outputs to', out_dir)
