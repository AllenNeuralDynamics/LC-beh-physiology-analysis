"""
Is the bootstrap cluster stability higher than a shapeless blob's?

`functional_cluster_analysis.py` reports bootstrap ARI of 0.75 at k=2, which
reads like evidence for two clusters. It is not, on its own: cutting a single
elongated Gaussian cloud along its first principal axis is also a highly
reproducible partition, because the cut plane is determined by the covariance
rather than by any gap. Stability only becomes evidence if it exceeds what a
matched unimodal cloud produces.

This regenerates the same bootstrap-ARI curve on parametric Gaussian surrogates
(same n, same mean, same covariance as the observed feature matrix) and reports
the observed curve against that null.
"""

import json
import os

import numpy as np
import pandas as pd

from utils.capsule_migration import capsule_directories
from utils.functional_features import (WAVEFORM_FEATURES,
                                       functional_feature_names)
from functional_cluster_analysis import (EXCLUDE, K_STABILITY, RANDOM_SEED,
                                        bootstrap_stability, fit_pca, perm_p,
                                        prune_redundant, robust_scale)

N_GAUSS = 100
N_BOOT = 50


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
    _, S = fit_pca(Z)
    print(f'{len(data)} units, {len(kept)} features, {S.shape[1]} PCs')

    obs = bootstrap_stability(S, k_range=K_STABILITY, n_boot=N_BOOT,
                              seed=RANDOM_SEED)
    obs = obs.set_index('k')['ari_mean']

    rng = np.random.default_rng(RANDOM_SEED)
    mu, cov = Z.mean(axis=0), np.cov(Z, rowvar=False)
    print(f'Running {N_GAUSS} Gaussian surrogates...')

    null = {k: [] for k in K_STABILITY}
    for i in range(N_GAUSS):
        Zs = rng.multivariate_normal(mu, cov, size=len(Z))
        _, Ss = fit_pca(Zs)
        try:
            sim = bootstrap_stability(Ss, k_range=K_STABILITY, n_boot=N_BOOT,
                                      seed=RANDOM_SEED + i)
        except Exception as exc:
            print(f'  sim {i} failed: {exc}')
            continue
        for k, v in zip(sim['k'], sim['ari_mean']):
            null[k].append(float(v))

    rows = []
    for k in K_STABILITY:
        vals = np.asarray(null[k], dtype=float)
        rows.append({'k': k, 'ari_observed': float(obs[k]),
                     'gaussian_mean': float(np.nanmean(vals)),
                     'gaussian_p05': float(np.nanpercentile(vals, 5)),
                     'gaussian_p95': float(np.nanpercentile(vals, 95)),
                     'p_gaussian': perm_p(obs[k], vals)})

    tbl = pd.DataFrame(rows)
    print('\nBootstrap ARI vs matched Gaussian null '
          '(p = P(null ARI >= observed)):')
    print(tbl.round(4).to_string(index=False))

    tbl.to_csv(os.path.join(out_dir, f'ari_null_{tag}.csv'), index=False)
    with open(os.path.join(out_dir, f'ari_null_{tag}.json'), 'w') as f:
        json.dump({'n_units': int(len(data)), 'n_pcs': int(S.shape[1]),
                   'n_gaussian': N_GAUSS, 'n_boot': N_BOOT,
                   'rows': rows}, f, indent=2)
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')

    ap = argparse.ArgumentParser()
    ap.add_argument('--waveform', action='store_true')
    main(include_waveform=ap.parse_args().waveform)
