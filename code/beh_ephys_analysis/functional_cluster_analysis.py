"""
Step 1: is the tagged-DRN functional feature space actually clustered, or a
continuum?

Runs on the feature matrix from `utils/functional_features.py`. No new spike
pass. The question is deliberately framed as a test rather than a fit, because
k-means / GMM will always return k clusters from a correlated high-dimensional
Gaussian blob. Everything is therefore compared against two nulls:

  gaussian  parametric bootstrap from a single multivariate Gaussian matched to
            the observed mean and covariance. Tests "more structure than one
            elliptical blob". This is the null that matters.
  shuffled  each feature column permuted independently. Preserves every
            marginal exactly and destroys the joint structure. Tests whether
            apparent clustering needs the correlations.

Tests
-----
1. GMM model selection (BIC, ICL, 5-fold held-out log-likelihood) for k=1..8,
   full and diagonal covariance, versus both nulls.
2. Silhouette of the best partition versus both nulls.
3. Hartigan dip test on every feature and every retained PC, FDR corrected,
   with a Gaussian-null calibration for the PC dips.
4. Bootstrap label stability (adjusted Rand index) for k=2..6.
5. Batch structure: adjusted mutual information between labels and
   session / animal, and a refit restricted to the sessions with enough units
   to ask whether any structure exists *within* a single recording.

Usage
-----
    python functional_cluster_analysis.py            # functional block only
    python functional_cluster_analysis.py --waveform # add waveform features
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from scipy.stats import iqr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

from utils.capsule_migration import capsule_directories
from utils.functional_features import (FUNCTIONAL_BLOCKS, WAVEFORM_FEATURES,
                                       build_feature_matrix,
                                       functional_feature_names)

RANDOM_SEED = 42
K_RANGE = list(range(1, 9))
K_STABILITY = (2, 3, 4, 5, 6)
N_NULL = 200
N_BOOT = 200
PCA_VAR = 0.90
REDUNDANCY_MAX = 0.85
CLIP_SD = 4.0

# Features excluded up front, with the reason. Both are artifacts of how the
# analysis windows were defined rather than weak effects.
EXCLUDE = {
    'auc_hit_resp_gocue':
        'only 31/200 units have >=15 miss trials (median 1 miss per unit)',
    'prof_iti_post':
        'iti_post is the next trial ITI, so its mean rate is near-identical to '
        'iti by construction (SD 0.010 vs 0.34 for the other profile features)',
}

# Priority for redundancy pruning: earlier features are kept, later ones are
# dropped when |rho| exceeds REDUNDANCY_MAX against something already kept.
PRUNE_PRIORITY = [
    'log_fr',
    'prof_resp_gocue',
    'prof_consumption',
    'auc_outcome_resp_choice',
    'auc_outcome_iti_post',
    'auc_ipsi_resp_gocue',
    'lick_coup_iti',
    'rr_10_partial',
    'tau_slope',
    'p_sum_partial',
    'streak_rho_resp_choice',
    'streak_rho_iti',
    'carry_rew_lag1',
    'prof_resp_choice',
    'auc_outcome_consumption',
    'auc_ipsi_resp_choice',
    'lick_coup_resp_choice',
]


# ------------------------------------------------------------------ plumbing


def prune_redundant(df, feats, threshold=REDUNDANCY_MAX, verbose=True):
    """Greedily drop features collinear with an already-kept feature."""
    order = ([f for f in PRUNE_PRIORITY if f in feats]
             + [f for f in feats if f not in PRUNE_PRIORITY])
    corr = df[order].astype(float).corr(method='spearman').abs()

    kept, dropped = [], {}
    for f in order:
        clash = [k for k in kept if corr.loc[f, k] > threshold]
        if clash:
            worst = max(clash, key=lambda k: corr.loc[f, k])
            dropped[f] = (worst, float(corr.loc[f, worst]))
        else:
            kept.append(f)

    if verbose and dropped:
        print(f'\nRedundancy prune at |rho| > {threshold}:')
        for f, (against, r) in dropped.items():
            print(f'  drop {f:<26s} rho={r:.2f} with {against}')
    return kept, dropped


def robust_scale(X, clip_sd=CLIP_SD):
    """Median/IQR scaling with symmetric clipping. Marginal shape is kept, so
    genuine bimodality survives into the dip tests."""
    X = np.asarray(X, dtype=float)
    centre = np.median(X, axis=0)
    scale = iqr(X, axis=0) / 1.349
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    Z = (X - centre) / scale
    n_clipped = int(np.sum(np.abs(Z) > clip_sd))
    return np.clip(Z, -clip_sd, clip_sd), n_clipped


def fit_pca(Z, var=PCA_VAR, seed=RANDOM_SEED):
    pca = PCA(n_components=var, svd_solver='full', random_state=seed)
    return pca, pca.fit_transform(Z)


def _icl(gmm, X):
    """Integrated completed likelihood: BIC plus a partition-entropy penalty.
    Penalises overlapping components, so it is more conservative than BIC."""
    resp = gmm.predict_proba(X)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -np.nansum(resp * np.log(np.where(resp > 0, resp, 1.0)))
    return gmm.bic(X) + 2.0 * ent


def gmm_selection(X, k_range=K_RANGE, cov_types=('full', 'diag'),
                  seed=RANDOM_SEED, cv_splits=5):
    """BIC / ICL / held-out log-likelihood across k and covariance type."""
    rows = []
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    for cov in cov_types:
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type=cov,
                                  random_state=seed, n_init=5,
                                  reg_covar=1e-4).fit(X)
            cv_ll = []
            for tr, te in kf.split(X):
                try:
                    g = GaussianMixture(n_components=k, covariance_type=cov,
                                        random_state=seed, n_init=3,
                                        reg_covar=1e-4).fit(X[tr])
                    cv_ll.append(g.score(X[te]))
                except Exception:
                    cv_ll.append(np.nan)
            rows.append({'cov': cov, 'k': k, 'bic': gmm.bic(X),
                         'icl': _icl(gmm, X), 'cv_ll': float(np.nanmean(cv_ll))})

    return pd.DataFrame(rows)


def selection_summary(sel):
    """Best k under each criterion, and the k=1 margin (positive = k>1 wins)."""
    out = {}
    for cov, g in sel.groupby('cov'):
        g = g.set_index('k')
        best_bic = int(g['bic'].idxmin())
        best_icl = int(g['icl'].idxmin())
        best_cv = int(g['cv_ll'].idxmax())
        out[cov] = {
            'best_k_bic': best_bic,
            'best_k_icl': best_icl,
            'best_k_cv_ll': best_cv,
            'delta_bic_1_minus_best': float(g.loc[1, 'bic'] - g['bic'].min()),
            'delta_icl_1_minus_best': float(g.loc[1, 'icl'] - g['icl'].min()),
            'delta_cv_ll_best_minus_1': float(g['cv_ll'].max() - g.loc[1, 'cv_ll']),
        }
    return out


def best_silhouette(X, k_range=(2, 3, 4, 5, 6), seed=RANDOM_SEED):
    best = {'k': np.nan, 'silhouette': -np.inf}
    for k in k_range:
        if k >= len(X):
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if s > best['silhouette']:
            best = {'k': int(k), 'silhouette': float(s)}
    return best


def pipeline_stats(Z, seed=RANDOM_SEED):
    """PCA + GMM selection + silhouette + PC dips, as one reusable unit so the
    nulls go through exactly the same steps as the data."""
    import diptest

    _, S = fit_pca(Z, seed=seed)
    sel = gmm_selection(S, seed=seed)
    summ = selection_summary(sel)
    sil = best_silhouette(S, seed=seed)
    dips = [diptest.dipstat(S[:, i]) for i in range(S.shape[1])]
    return {
        'n_pcs': int(S.shape[1]),
        'delta_bic_full': summ['full']['delta_bic_1_minus_best'],
        'delta_bic_diag': summ['diag']['delta_bic_1_minus_best'],
        'delta_icl_full': summ['full']['delta_icl_1_minus_best'],
        'delta_cv_ll_full': summ['full']['delta_cv_ll_best_minus_1'],
        'best_k_bic_full': summ['full']['best_k_bic'],
        'best_k_icl_full': summ['full']['best_k_icl'],
        'silhouette': sil['silhouette'],
        'max_pc_dip': float(np.max(dips)),
    }, sel, summ, sil, S


def null_distribution(Z, kind, n=N_NULL, seed=RANDOM_SEED):
    """Null distribution of the pipeline statistics.

    kind='gaussian' : one multivariate Gaussian, matched mean and covariance.
    kind='shuffled' : each column permuted independently.
    """
    rng = np.random.default_rng(seed)
    mu = Z.mean(axis=0)
    cov = np.cov(Z, rowvar=False)
    rows = []

    for i in range(n):
        if kind == 'gaussian':
            Zs = rng.multivariate_normal(mu, cov, size=len(Z))
        elif kind == 'shuffled':
            Zs = np.column_stack([rng.permutation(Z[:, j])
                                  for j in range(Z.shape[1])])
        else:
            raise ValueError(kind)
        try:
            stats, *_ = pipeline_stats(Zs, seed=RANDOM_SEED + i)
            rows.append(stats)
        except Exception as exc:
            print(f'  null {kind} sim {i} failed: {exc}')

    return pd.DataFrame(rows)


def perm_p(observed, null_values, side='greater'):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return np.nan
    if side == 'greater':
        return float((np.sum(null_values >= observed) + 1) / (null_values.size + 1))
    return float((np.sum(null_values <= observed) + 1) / (null_values.size + 1))


def bootstrap_stability(S, k_range=K_STABILITY, n_boot=N_BOOT,
                        seed=RANDOM_SEED):
    """ARI between full-data labels and bootstrap-refit labels, on the units
    present in each resample. Genuine clusters are stable; blob partitions are
    not."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in k_range:
        ref = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(S)
        ref_labels = ref.labels_
        aris = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(S), len(S))
            uniq = np.unique(idx)
            try:
                km = KMeans(n_clusters=k, n_init=5,
                            random_state=int(rng.integers(1e6))).fit(S[idx])
                aris.append(adjusted_rand_score(ref_labels[uniq],
                                                km.predict(S[uniq])))
            except Exception:
                continue
        rows.append({'k': k, 'ari_mean': float(np.mean(aris)),
                     'ari_p05': float(np.percentile(aris, 5)),
                     'ari_p95': float(np.percentile(aris, 95))})
    return pd.DataFrame(rows)


def batch_dependence(S, meta, k_range=(2, 3, 4), n_perm=1000, seed=RANDOM_SEED):
    """Adjusted mutual information between cluster labels and session/animal."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(S)
        for key in ['session', 'animal']:
            obs = adjusted_mutual_info_score(meta[key].to_numpy(), labels)
            null = [adjusted_mutual_info_score(meta[key].to_numpy(),
                                               rng.permutation(labels))
                    for _ in range(n_perm)]
            rows.append({'k': k, 'grouping': key, 'ami': float(obs),
                         'p_perm': perm_p(obs, null)})
    return pd.DataFrame(rows)


def within_session_refit(Z, meta, min_units=8, seed=RANDOM_SEED):
    """Does any structure survive inside a single recording? Compares the
    observed Gaussian-null margin session by session."""
    rows = []
    for session, idx in meta.groupby('session').groups.items():
        pos = meta.index.get_indexer(idx)
        if len(pos) < min_units:
            continue
        Zs = Z[pos]
        try:
            stats, _, _, _, _ = pipeline_stats(Zs, seed=seed)
        except Exception as exc:
            print(f'  {session}: refit failed ({exc})')
            continue
        null = null_distribution(Zs, 'gaussian', n=50, seed=seed)
        rows.append({
            'session': session, 'n_units': len(pos),
            'best_k_bic': stats['best_k_bic_full'],
            'delta_bic': stats['delta_bic_full'],
            'p_gaussian': perm_p(stats['delta_bic_full'], null['delta_bic_full']),
            'silhouette': stats['silhouette'],
            'p_silhouette': perm_p(stats['silhouette'], null['silhouette']),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- driver


def main(include_waveform=False):
    import diptest

    out_dir = os.path.join(capsule_directories()['manuscript_fig_prep_dir'],
                           'functional_clusters')
    os.makedirs(out_dir, exist_ok=True)
    tag = 'func_wf' if include_waveform else 'func'

    matrix_file = os.path.join(out_dir, 'feature_matrix.csv')
    if os.path.exists(matrix_file):
        df = pd.read_csv(matrix_file)
        df['unit_id'] = df['unit_id'].astype(str)
    else:
        df, _ = build_feature_matrix()

    feats = [f for f in functional_feature_names() if f not in EXCLUDE]
    if include_waveform:
        feats += [f for f in WAVEFORM_FEATURES if f in df.columns]

    print('=' * 72)
    print(f'Functional clusteredness test  [{tag}]')
    print('=' * 72)
    print(f'Units in QC set: {len(df)}  sessions: {df.session.nunique()}  '
          f'animals: {df.animal.nunique()}')
    for f, why in EXCLUDE.items():
        print(f'Excluded {f}: {why}')

    kept, dropped = prune_redundant(df, feats)

    complete = df[kept].astype(float).notna().all(axis=1)
    data = df.loc[complete].reset_index(drop=True)
    X = data[kept].to_numpy(float)
    print(f'\nFeatures used: {len(kept)}  |  complete units: {len(data)}  '
          f'sessions: {data.session.nunique()}  animals: {data.animal.nunique()}')
    print('  ' + ', '.join(kept))

    Z, n_clipped = robust_scale(X)
    print(f'Robust-scaled; {n_clipped} of {Z.size} values clipped at '
          f'+/-{CLIP_SD} SD')

    # ---- observed pipeline
    obs, sel, summ, sil, S = pipeline_stats(Z)
    pca, _ = fit_pca(Z)
    print(f'\nPCA: {obs["n_pcs"]} components for {PCA_VAR:.0%} variance; '
          f'ratios ' + ', '.join(f'{v:.2f}' for v in
                                 pca.explained_variance_ratio_[:6]))

    print('\nGMM selection (lower BIC/ICL better, higher CV log-lik better):')
    print(sel.pivot_table(index='k', columns='cov',
                          values=['bic', 'icl', 'cv_ll']).round(1).to_string())
    for cov, s in summ.items():
        print(f'  {cov}: best k  BIC={s["best_k_bic"]}  ICL={s["best_k_icl"]}  '
              f'CV={s["best_k_cv_ll"]}   dBIC(1-best)='
              f'{s["delta_bic_1_minus_best"]:.1f}  '
              f'dCV(best-1)={s["delta_cv_ll_best_minus_1"]:.4f}')
    print(f'  best silhouette: k={sil["k"]} s={sil["silhouette"]:.3f}')

    # ---- nulls
    nulls = {}
    for kind in ['gaussian', 'shuffled']:
        print(f'\nRunning {kind} null ({N_NULL} sims)...')
        nulls[kind] = null_distribution(Z, kind)
        nulls[kind].to_csv(os.path.join(out_dir, f'null_{kind}_{tag}.csv'),
                           index=False)

    stat_keys = ['delta_bic_full', 'delta_bic_diag', 'delta_icl_full',
                 'delta_cv_ll_full', 'silhouette', 'max_pc_dip']
    rows = []
    for key in stat_keys:
        row = {'statistic': key, 'observed': obs[key]}
        for kind, nd in nulls.items():
            row[f'{kind}_mean'] = float(np.nanmean(nd[key]))
            row[f'{kind}_p95'] = float(np.nanpercentile(nd[key], 95))
            row[f'p_{kind}'] = perm_p(obs[key], nd[key])
        rows.append(row)
    null_tbl = pd.DataFrame(rows)
    print('\nObserved vs nulls (p = P(null >= observed)):')
    print(null_tbl.round(4).to_string(index=False))

    k_tbl = pd.DataFrame({
        'source': ['observed'] + list(nulls),
        'best_k_bic_full': [obs['best_k_bic_full']]
        + [float(np.nanmean(nd['best_k_bic_full'])) for nd in nulls.values()],
        'best_k_icl_full': [obs['best_k_icl_full']]
        + [float(np.nanmean(nd['best_k_icl_full'])) for nd in nulls.values()],
    })
    print('\nSelected k, observed vs null means:')
    print(k_tbl.round(2).to_string(index=False))

    # ---- dip tests on features and PCs
    dip_rows = []
    for f in kept:
        d, p = diptest.diptest(data[f].to_numpy(float))
        dip_rows.append({'variable': f, 'kind': 'feature', 'dip': d,
                         'p_asymptotic': p})
    for i in range(S.shape[1]):
        d, p = diptest.diptest(S[:, i])
        gauss_null = nulls['gaussian']['max_pc_dip']
        dip_rows.append({'variable': f'PC{i + 1}', 'kind': 'pc', 'dip': d,
                         'p_asymptotic': p,
                         'p_gaussian_maxpc': perm_p(d, gauss_null)})
    dip_tbl = pd.DataFrame(dip_rows)
    ok = dip_tbl['p_asymptotic'].notna()
    dip_tbl.loc[ok, 'p_fdr'] = multipletests(
        dip_tbl.loc[ok, 'p_asymptotic'], method='fdr_bh')[1]
    print('\nHartigan dip test (unimodality; small p = multimodal):')
    print(dip_tbl.sort_values('p_asymptotic').round(4).to_string(index=False))

    # ---- stability and batch structure
    stab = bootstrap_stability(S)
    print('\nBootstrap label stability (ARI vs full-data partition):')
    print(stab.round(3).to_string(index=False))

    batch = batch_dependence(S, data[['session', 'animal']])
    print('\nCluster labels vs recording batch:')
    print(batch.round(4).to_string(index=False))

    within = within_session_refit(Z, data[['session', 'animal']])
    print('\nWithin-session refit (sessions with >=8 complete units):')
    print(within.round(3).to_string(index=False) if len(within)
          else '  no session has >=8 complete units')

    # ---- save
    sel.to_csv(os.path.join(out_dir, f'gmm_selection_{tag}.csv'), index=False)
    null_tbl.to_csv(os.path.join(out_dir, f'null_comparison_{tag}.csv'),
                    index=False)
    dip_tbl.to_csv(os.path.join(out_dir, f'dip_tests_{tag}.csv'), index=False)
    stab.to_csv(os.path.join(out_dir, f'bootstrap_stability_{tag}.csv'),
                index=False)
    batch.to_csv(os.path.join(out_dir, f'batch_dependence_{tag}.csv'),
                 index=False)
    within.to_csv(os.path.join(out_dir, f'within_session_refit_{tag}.csv'),
                  index=False)
    np.save(os.path.join(out_dir, f'pc_scores_{tag}.npy'), S)
    data[['session', 'unit_id', 'animal', 'ml', 'ap', 'dv', 'y_loc']
         + kept].to_csv(os.path.join(out_dir, f'units_used_{tag}.csv'),
                        index=False)

    with open(os.path.join(out_dir, f'summary_{tag}.json'), 'w') as f:
        json.dump({
            'tag': tag,
            'n_units': int(len(data)),
            'n_sessions': int(data.session.nunique()),
            'n_animals': int(data.animal.nunique()),
            'features_used': kept,
            'features_dropped_redundant': {k: list(v)
                                           for k, v in dropped.items()},
            'features_excluded': EXCLUDE,
            'n_pcs': obs['n_pcs'],
            'explained_variance_ratio':
                [float(v) for v in pca.explained_variance_ratio_],
            'observed': {k: float(obs[k]) for k in stat_keys},
            'selection': summ,
            'p_values': {r['statistic']: {'gaussian': r['p_gaussian'],
                                          'shuffled': r['p_shuffled']}
                         for r in rows},
        }, f, indent=2)

    print(f'\nSaved to {out_dir}')
    return data, S, null_tbl


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    ap = argparse.ArgumentParser()
    ap.add_argument('--waveform', action='store_true',
                    help='add the 7 waveform focus features to the matrix')
    args = ap.parse_args()
    main(include_waveform=args.waveform)
