"""
Signed k=1 vs k=2 model-selection margin, with matched nulls.

`functional_cluster_analysis.py` reports `delta_bic_1_minus_best`, which is
identically zero whenever k=1 wins -- true for the data and for every matched
Gaussian null, so the comparison is degenerate (p = 1.0 by construction) and
says nothing about *how strongly* one component is preferred.

This computes a signed margin instead:

    bic_1_minus_2     > 0 means BIC prefers two components
    cv_ll_2_minus_1   > 0 means held-out likelihood prefers two components

and compares the observed margins to the same two nulls. Only k=1 and k=2 are
fitted, so a null sweep costs about a quarter of the full k=1..8 pipeline.
"""

import json
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from utils.capsule_migration import capsule_directories
from utils.functional_features import (WAVEFORM_FEATURES,
                                       functional_feature_names)
from functional_cluster_analysis import (EXCLUDE, N_NULL, PCA_VAR,
                                        RANDOM_SEED, perm_p, prune_redundant,
                                        robust_scale)


def margin_stats(Z, seed=RANDOM_SEED, cv_splits=5):
    """BIC and held-out log-likelihood margins for k=1 vs k=2."""
    S = PCA(n_components=PCA_VAR, svd_solver='full',
            random_state=seed).fit_transform(Z)
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    out = {'n_pcs': int(S.shape[1])}
    for cov in ('full', 'diag'):
        bic, cv = {}, {}
        for k in (1, 2):
            bic[k] = GaussianMixture(n_components=k, covariance_type=cov,
                                     random_state=seed, n_init=5,
                                     reg_covar=1e-4).fit(S).bic(S)
            folds = []
            for tr, te in kf.split(S):
                g = GaussianMixture(n_components=k, covariance_type=cov,
                                    random_state=seed, n_init=3,
                                    reg_covar=1e-4).fit(S[tr])
                folds.append(g.score(S[te]))
            cv[k] = float(np.mean(folds))
        out[f'bic_1_minus_2_{cov}'] = float(bic[1] - bic[2])
        out[f'cv_ll_2_minus_1_{cov}'] = float(cv[2] - cv[1])
    return out


def null_margins(Z, kind, n=N_NULL, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    mu, cov = Z.mean(axis=0), np.cov(Z, rowvar=False)
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
            rows.append(margin_stats(Zs, seed=RANDOM_SEED + i))
        except Exception as exc:
            print(f'  {kind} sim {i} failed: {exc}')
    return pd.DataFrame(rows)


def main(include_waveform=False):
    out_dir = os.path.join(capsule_directories()['manuscript_fig_prep_dir'],
                           'functional_clusters')
    tag = 'func_wf' if include_waveform else 'func'

    df = pd.read_csv(os.path.join(out_dir, 'feature_matrix.csv'))
    feats = [f for f in functional_feature_names() if f not in EXCLUDE]
    if include_waveform:
        feats += [f for f in WAVEFORM_FEATURES if f in df.columns]

    kept, _ = prune_redundant(df, feats, verbose=False)
    data = df.loc[df[kept].astype(float).notna().all(axis=1)].reset_index(drop=True)
    Z, _ = robust_scale(data[kept].to_numpy(float))
    print(f'{len(data)} units, {len(kept)} features')

    obs = margin_stats(Z)
    keys = [k for k in obs if k != 'n_pcs']
    print('\nObserved margins (positive = two components preferred):')
    for k in keys:
        print(f'  {k:24s} {obs[k]:+.3f}')

    rows = []
    nulls = {}
    for kind in ('gaussian', 'shuffled'):
        print(f'\nRunning {kind} null ({N_NULL} sims)...')
        nulls[kind] = null_margins(Z, kind)
        nulls[kind].to_csv(os.path.join(out_dir,
                                        f'null_margin_{kind}_{tag}.csv'),
                           index=False)

    for k in keys:
        row = {'statistic': k, 'observed': obs[k]}
        for kind, nd in nulls.items():
            row[f'{kind}_mean'] = float(np.nanmean(nd[k]))
            row[f'{kind}_sd'] = float(np.nanstd(nd[k], ddof=1))
            row[f'p_{kind}'] = perm_p(obs[k], nd[k])
        rows.append(row)

    tbl = pd.DataFrame(rows)
    print('\nObserved vs nulls (p = P(null margin >= observed margin)):')
    print(tbl.round(4).to_string(index=False))

    tbl.to_csv(os.path.join(out_dir, f'margin_comparison_{tag}.csv'),
               index=False)
    with open(os.path.join(out_dir, f'margin_summary_{tag}.json'), 'w') as f:
        json.dump({'n_units': int(len(data)), 'features_used': kept,
                   'observed': obs,
                   'p_values': {r['statistic']: {'gaussian': r['p_gaussian'],
                                                 'shuffled': r['p_shuffled']}
                                for r in rows}}, f, indent=2)
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')

    ap = argparse.ArgumentParser()
    ap.add_argument('--waveform', action='store_true')
    main(include_waveform=ap.parse_args().waveform)
