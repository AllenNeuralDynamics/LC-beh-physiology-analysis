"""Session/insertion-bias controls for anatomical gradient analyses.

The spatial-dependence tests in `combine_tools.spatial_dependence_summary` treat
every unit as an independent sample of anatomical space. For acute Neuropixels
recordings that assumption is wrong along any axis whose variance is carried by
where the probe went in rather than by where units sat along the shank. In the
DRN optotagged datasets that is ML: ~97% of its variance is between-session, so
a "unit-level" ML gradient is really a between-insertion comparison with an
effective n of sessions (or mice), not of units.

The functions here quantify that structure and re-test gradients with nulls and
estimators matched to the leverage that actually exists:

    coordinate_leverage          how much within- vs between-session spread each axis has
    insertion_axis               principal axes of the session centroids, and the angle
                                 between a fitted gradient and the insertion axis
    within_between_regression    Mundlak/hybrid split of a gradient into its
                                 within-session and between-session parts
    session_centroid_permutation restricted null: keep each session's internal layout and
                                 its feature values, shuffle which centroid it sits at
    session_level_regression     collapse to session means (the honest between-insertion test)
    cluster_bootstrap            resample sessions or mice to get CIs on the gradient
    leave_one_group_out          jackknife influence of single sessions / mice
    session_bias_report          runs all of the above for one feature
    session_bias_table           runs `session_bias_report` over many features, with FDR

Coordinate convention matches the manuscript notebooks: columns are
(ML, AP, DV) in mm relative to bregma, ML mirrored to one hemisphere.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

AXIS_NAMES = ('ML', 'AP', 'DV')


def _clean(coords, values, session, mouse=None):
    """Drop non-finite rows and return (X, y, session, mouse) as arrays."""
    X = np.asarray(coords, float)
    X = X.reshape(X.shape[0], -1)
    y = np.asarray(values, float)
    s = np.asarray(session)
    m = None if mouse is None else np.asarray(mouse)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[ok], y[ok], s[ok], (None if m is None else m[ok]), ok


def mouse_from_session(session):
    """`behavior_844917_2026-06-04_12-30-01` -> `844917`."""
    return pd.Series(np.asarray(session)).astype(str).str.split('_').str[1].to_numpy()


def _centroids(X, s):
    """Session centroids, the per-unit centroid, and within-session offsets."""
    us, idx = np.unique(s, return_inverse=True)
    cent = np.vstack([X[idx == i].mean(axis=0) for i in range(len(us))])
    return us, idx, cent, X - cent[idx]


# ---------------------------------------------------------------------------
# 1. how much leverage does each axis actually have?
# ---------------------------------------------------------------------------
def coordinate_leverage(coords, session, mouse=None, axis_names=AXIS_NAMES):
    """Variance components of each coordinate axis.

    Splits each axis into between-mouse / between-session-within-mouse /
    within-session variance with a nested random-effects model, and reports the
    within-session intraclass correlation. `icc_session` near 1 means the axis is
    effectively a session label: no within-insertion contrast is available, so a
    gradient along it can only be tested between insertions.

    `n_eff` is the cluster-corrected effective sample size for a session-level
    effect, n / (1 + (m_bar - 1) * icc_session), i.e. how many independent units'
    worth of information the axis carries.
    """
    X, _, s, m, _ = _clean(coords, np.zeros(len(np.asarray(session))), session, mouse)
    if m is None:
        m = mouse_from_session(s)
    n = len(s)
    us = np.unique(s)
    m_bar = n / len(us)

    rows = []
    for j, name in enumerate(axis_names[:X.shape[1]]):
        d = pd.DataFrame({'v': X[:, j], 'session': s, 'mouse': m})
        # one-way (session) components, computed in closed form so that the
        # ICC does not depend on mixed-model convergence
        g = d.groupby('session')['v']
        ns, ms = g.size(), g.mean()
        ss_b = ((ms - d['v'].mean()) ** 2 * ns).sum()
        ss_w = ((d['v'] - d['session'].map(ms)) ** 2).sum()
        k = len(ns)
        ms_b = ss_b / (k - 1) if k > 1 else np.nan
        ms_w = ss_w / (n - k) if n > k else np.nan
        n0 = (n - (ns ** 2).sum() / n) / (k - 1) if k > 1 else np.nan
        var_sess = max((ms_b - ms_w) / n0, 0.0) if np.isfinite(n0) and n0 > 0 else np.nan
        icc = var_sess / (var_sess + ms_w) if np.isfinite(var_sess) else np.nan

        # nested split, for attributing the between-session part to the mouse
        frac_mouse = frac_sess = frac_unit = np.nan
        try:
            fit = smf.mixedlm('v ~ 1', d, groups=d['mouse'],
                              re_formula='1',
                              vc_formula={'session': '0+C(session)'}).fit(reml=True)
            v_mouse = float(np.asarray(fit.cov_re)[0, 0])
            v_sess = float(np.asarray(fit.vcomp)[0])
            v_unit = float(fit.scale)
            tot = v_mouse + v_sess + v_unit
            if tot > 0:
                frac_mouse, frac_sess, frac_unit = v_mouse / tot, v_sess / tot, v_unit / tot
        except Exception:
            pass

        rows.append({
            'axis': name,
            'sd_total_mm': float(np.std(X[:, j], ddof=1)),
            'sd_within_session_mm': float(np.sqrt(ms_w)) if np.isfinite(ms_w) else np.nan,
            'sd_between_session_mm': float(np.std(ms, ddof=1)) if k > 1 else np.nan,
            'icc_session': float(icc) if np.isfinite(icc) else np.nan,
            'var_frac_mouse': frac_mouse,
            'var_frac_session_within_mouse': frac_sess,
            'var_frac_unit_within_session': frac_unit,
            'n_eff_units': float(n / (1 + (m_bar - 1) * icc)) if np.isfinite(icc) else np.nan,
        })

    out = pd.DataFrame(rows)
    out.attrs['n_units'] = n
    out.attrs['n_sessions'] = len(us)
    out.attrs['n_mice'] = len(np.unique(m))
    return out


# ---------------------------------------------------------------------------
# 2. what direction do the insertions themselves vary along?
# ---------------------------------------------------------------------------
def insertion_axis(coords, session, weight_by_n=True, gradient=None):
    """Principal axes of between-session variation (the "insertion axis").

    Returns the PCA of the session centroids, so you can say what direction the
    probe placement varied along. If `gradient` (a length-D direction vector,
    e.g. the OLS coefficients of a fitted anatomical gradient) is given, also
    reports the angle between that gradient and insertion PC1, and the fraction
    of the gradient's squared norm lying in the insertion subspace. A gradient
    nearly parallel to insertion PC1 is the one most at risk of being a
    between-insertion artifact.
    """
    X, _, s, _, _ = _clean(coords, np.zeros(len(np.asarray(session))), session)
    us, idx, cent, _ = _centroids(X, s)
    w = np.bincount(idx).astype(float) if weight_by_n else np.ones(len(us))
    w = w / w.sum()
    mu = (cent * w[:, None]).sum(axis=0)
    Xc = (cent - mu) * np.sqrt(w)[:, None]
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_frac = S ** 2 / np.sum(S ** 2)

    out = {
        'n_sessions': int(len(us)),
        'centroids': cent,
        'axes': Vt,                      # rows are PCs in (ML, AP, DV)
        'var_frac': var_frac,
        'pc1': Vt[0],
        'pc1_var_frac': float(var_frac[0]),
    }
    if gradient is not None:
        g = np.asarray(gradient, float).ravel()
        gn = g / np.linalg.norm(g)
        cos1 = float(np.abs(np.dot(gn, Vt[0])))
        out['gradient_unit'] = gn
        out['cos_with_insertion_pc1'] = cos1
        out['angle_with_insertion_pc1_deg'] = float(np.degrees(np.arccos(np.clip(cos1, 0, 1))))
    return out


# ---------------------------------------------------------------------------
# 3. split a gradient into within- and between-session parts
# ---------------------------------------------------------------------------
def within_between_regression(coords, values, session, axis_names=AXIS_NAMES):
    """Mundlak / hybrid decomposition of an anatomical gradient.

    Fits `value ~ within + between + (1|session)`, where `within` is the
    coordinate expressed relative to its session centroid and `between` is the
    centroid itself. `beta_within` is identified only by contrasts among units
    recorded in the same insertion; `beta_between` only by contrasts across
    insertions. The pooled OLS slope that `spatial_dependence_summary` reports is
    a precision-weighted blend of the two and, on a high-ICC axis like ML, is
    almost entirely `beta_between`.

    A `beta_within` with a very wide CI is the honest signal that the axis has no
    within-insertion leverage, not evidence against a gradient. `p_diff` tests
    beta_within == beta_between (a Hausman-style check for session confounding).
    """
    X, y, s, _, _ = _clean(coords, values, session)
    us, idx, cent, off = _centroids(X, s)
    names = list(axis_names[:X.shape[1]])

    d = pd.DataFrame({'y': y, 'session': s})
    for j, nm in enumerate(names):
        d[f'w_{nm}'] = off[:, j]
        d[f'b_{nm}'] = cent[idx][:, j]
    terms = ' + '.join([f'w_{nm}' for nm in names] + [f'b_{nm}' for nm in names])

    rows = []
    try:
        fit = smf.mixedlm(f'y ~ {terms}', d, groups=d['session']).fit(reml=True)
        ci = fit.conf_int()
        for nm in names:
            bw, bb = f'w_{nm}', f'b_{nm}'
            # Wald test of beta_within == beta_between, from the fixed-effect
            # covariance block (MixedLMResults.t_test chokes on the variance
            # component, so build the contrast variance directly)
            try:
                V = np.asarray(fit.cov_params())
                fe = list(fit.params.index)
                iw, ib = fe.index(bw), fe.index(bb)
                var_diff = V[iw, iw] + V[ib, ib] - 2 * V[iw, ib]
                z = (fit.params[bw] - fit.params[bb]) / np.sqrt(var_diff)
                p_diff = float(2 * stats.norm.sf(abs(z)))
            except Exception:
                p_diff = np.nan
            rows.append({
                'axis': nm,
                'beta_within': float(fit.params[bw]),
                'se_within': float(fit.bse[bw]),
                'ci_within': (float(ci.loc[bw, 0]), float(ci.loc[bw, 1])),
                'p_within': float(fit.pvalues[bw]),
                'beta_between': float(fit.params[bb]),
                'se_between': float(fit.bse[bb]),
                'ci_between': (float(ci.loc[bb, 0]), float(ci.loc[bb, 1])),
                'p_between': float(fit.pvalues[bb]),
                'p_diff_within_vs_between': p_diff,
            })
    except Exception as exc:  # pragma: no cover - convergence failures
        for nm in names:
            rows.append({'axis': nm, 'error': repr(exc)})

    # pooled OLS, for reference
    pooled = sm.OLS(y, sm.add_constant(X)).fit()
    out = pd.DataFrame(rows)
    out.attrs['pooled_ols_coef'] = dict(zip(names, pooled.params[1:]))
    out.attrs['pooled_ols_r2'] = float(pooled.rsquared)
    out.attrs['n_units'] = int(len(y))
    out.attrs['n_sessions'] = int(len(us))
    return out


# ---------------------------------------------------------------------------
# 4. restricted permutation null that respects session structure
# ---------------------------------------------------------------------------
def session_centroid_permutation(coords, values, session, permutations=2000,
                                 seed=0, axis_names=AXIS_NAMES, n_jobs=-1,
                                 strata=None):
    """Permutation test that keeps session structure intact.

    Each session's internal spatial layout and its block of feature values are
    held together; what gets shuffled is which centroid the session's block sits
    at:

        X_perm[i] = centroid[perm(session_i)] + (X[i] - centroid[session_i])

    This breaks only the between-insertion link between location and feature, so
    it tests the gradient against the null "insertions happen to differ". The
    global shuffle used by `spatial_dependence_summary` destroys session
    clustering as well, so its null R2 distribution is too narrow and its
    p-values are anti-conservative whenever units cluster by session.

    The null treats session centroids as exchangeable. Pass `strata` (a per-unit
    label, usually mouse) to permute centroids only within stratum, which also
    removes any between-animal contribution; that is stricter but loses the
    sessions belonging to a stratum of size one, so run it as a sensitivity
    check rather than the primary test.

    Returns observed/permuted R2 for the full linear trend and per-axis
    |t|-statistics, each with both nulls for comparison.
    """
    X, y, s, strat, _ = _clean(coords, values, session, strata)
    names = list(axis_names[:X.shape[1]])
    us, idx, cent, off = _centroids(X, s)
    n_sess = len(us)
    # stratum of each session, for stratified permutation
    sess_strat = (None if strat is None
                  else np.array([strat[s == u][0] for u in us]))

    def stats_for(Xd, yd):
        fit = sm.OLS(yd, sm.add_constant(Xd)).fit()
        return float(fit.rsquared), np.abs(np.asarray(fit.tvalues, float)[1:])

    r2_obs, t_obs = stats_for(X, y)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2 ** 32 - 1, size=permutations)

    def _permute_sessions(r):
        """Permutation of session indices, optionally within stratum."""
        if sess_strat is None:
            return r.permutation(n_sess)
        perm = np.arange(n_sess)
        for lab in np.unique(sess_strat):
            pos = np.flatnonzero(sess_strat == lab)
            perm[pos] = pos[r.permutation(len(pos))]
        return perm

    def one(kind, sd):
        r = np.random.default_rng(int(sd))
        if kind == 'session':
            Xp = cent[_permute_sessions(r)][idx] + off
            return stats_for(Xp, y)
        return stats_for(X, r.permutation(y))

    def run(kind):
        try:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=n_jobs, prefer='processes')(
                delayed(one)(kind, sd) for sd in seeds)
        except Exception:
            res = [one(kind, sd) for sd in seeds]
        return np.array([a for a, _ in res]), np.vstack([b for _, b in res])

    r2_sess, t_sess = run('session')
    r2_glob, t_glob = run('global')

    def pval(null, obs):
        return float((np.sum(null >= obs) + 1) / (len(null) + 1))

    per_axis = pd.DataFrame({
        'axis': names,
        't_obs': t_obs,
        'p_session_perm': [pval(t_sess[:, j], t_obs[j]) for j in range(len(names))],
        'p_global_perm': [pval(t_glob[:, j], t_obs[j]) for j in range(len(names))],
    })
    return {
        'n_units': int(len(y)),
        'n_sessions': int(n_sess),
        'permutations': int(permutations),
        'r2_obs': r2_obs,
        'p_r2_session_perm': pval(r2_sess, r2_obs),
        'p_r2_global_perm': pval(r2_glob, r2_obs),
        'null_r2_session_mean': float(r2_sess.mean()),
        'null_r2_global_mean': float(r2_glob.mean()),
        'per_axis': per_axis,
    }


# ---------------------------------------------------------------------------
# 5. the between-insertion test done at the right level
# ---------------------------------------------------------------------------
def session_level_regression(coords, values, session, mouse=None,
                             weight_by_n=True, axis_names=AXIS_NAMES,
                             min_units=1):
    """Collapse to one point per session and regress session means on centroids.

    This is the estimator that matches the leverage on a high-ICC axis: n is the
    number of insertions, not units. Weighted by units per session (optional) and
    with SEs clustered by mouse, so that repeated insertions in one animal are
    not counted as independent.
    """
    X, y, s, m, _ = _clean(coords, values, session, mouse)
    if m is None:
        m = mouse_from_session(s)
    names = list(axis_names[:X.shape[1]])

    d = pd.DataFrame({'y': y, 'session': s, 'mouse': m})
    for j, nm in enumerate(names):
        d[nm] = X[:, j]
    g = d.groupby('session')
    agg = g.agg(n_units=('y', 'size'), y=('y', 'mean'), mouse=('mouse', 'first'),
                **{nm: (nm, 'mean') for nm in names}).reset_index()
    agg = agg[agg['n_units'] >= min_units]

    Xs = sm.add_constant(agg[names].to_numpy(float))
    w = agg['n_units'].to_numpy(float) if weight_by_n else np.ones(len(agg))
    fit = sm.WLS(agg['y'].to_numpy(float), Xs, weights=w).fit(
        cov_type='cluster', cov_kwds={'groups': agg['mouse'].to_numpy()})
    ci = fit.conf_int()
    per_axis = pd.DataFrame({
        'axis': names,
        'beta': fit.params[1:],
        'se_cluster_mouse': fit.bse[1:],
        'ci_low': ci[1:, 0] if isinstance(ci, np.ndarray) else ci.iloc[1:, 0].to_numpy(),
        'ci_high': ci[1:, 1] if isinstance(ci, np.ndarray) else ci.iloc[1:, 1].to_numpy(),
        'p_value': fit.pvalues[1:],
    }).reset_index(drop=True)
    return {
        'n_sessions': int(len(agg)),
        'n_mice': int(agg['mouse'].nunique()),
        'r2': float(fit.rsquared),
        'f_pvalue': float(fit.f_pvalue),
        'per_axis': per_axis,
        'session_table': agg,
    }


def session_confound_screen(df, coord_cols=('x_ccf', 'y_ccf', 'z_ccf'),
                            session_col='session', nuisance_cols=(),
                            bregma=(-5.74, 5.4, -0.45), mirror_ml=True,
                            axis_names=AXIS_NAMES, add_defaults=True):
    """Does probe placement co-vary with session-level nuisance variables?

    A between-insertion gradient is only an anatomical claim if where the probe
    went is not systematically tied to something else that changes with it. This
    correlates each session's mean coordinate against session-level nuisance
    variables (yield, tagging quality, ISI violations, recording date, how much
    depth the insertion sampled, animal identity) and reports Spearman rho with
    n = sessions. A nuisance variable that tracks session-mean ML is an
    alternative explanation for any ML effect and should be carried as a
    covariate in the session-level regression.

    `nuisance_cols` are per-unit columns averaged within session; pass the
    tagging metrics (`p_max`, `eu`, `corr`, `lat_max_p`) when the table has them.
    """
    X = df[list(coord_cols)].to_numpy(float) - np.asarray(bregma, float)
    if mirror_ml:
        X[:, 0] = np.abs(X[:, 0])
    names = list(axis_names[:X.shape[1]])

    d = pd.DataFrame({'session': df[session_col].to_numpy()})
    for j, nm in enumerate(names):
        d[nm] = X[:, j]
    for c in nuisance_cols:
        if c in df.columns:
            d[c] = pd.to_numeric(df[c], errors='coerce').to_numpy(float)
    have = [c for c in nuisance_cols if c in d.columns]

    g = d.groupby('session')
    agg = g.agg(n_units=('session', 'size'),
                **{nm: (nm, 'mean') for nm in names},
                **{c: (c, 'mean') for c in have}).reset_index()
    # depth actually sampled by the insertion, and animal / date metadata
    agg['dv_range_sampled'] = g['DV'].agg(lambda v: v.max() - v.min()).to_numpy()
    agg['mouse'] = mouse_from_session(agg['session'].to_numpy())
    agg['date_rank'] = (pd.to_datetime(
        agg['session'].astype(str).str.extract(r'_(\d{4}-\d{2}-\d{2})_')[0],
        errors='coerce').rank().to_numpy())

    screen = list(have) + ['n_units', 'dv_range_sampled', 'date_rank'] if add_defaults \
        else list(have)
    rows = []
    for nm in names:
        for c in screen:
            v = pd.to_numeric(agg[c], errors='coerce')
            ok = v.notna() & agg[nm].notna()
            if ok.sum() < 6:
                continue
            rho, p = stats.spearmanr(agg.loc[ok, nm], v[ok])
            rows.append({'axis': nm, 'nuisance': c, 'n_sessions': int(ok.sum()),
                         'spearman_rho': float(rho), 'p_value': float(p)})
    out = pd.DataFrame(rows)
    if len(out):
        out['fdr_bh'] = multipletests(out['p_value'].to_numpy(float),
                                      method='fdr_bh')[1]
    # how much of each axis's between-session spread is explained by mouse alone
    mouse_rows = []
    for nm in names:
        try:
            fit = smf.ols(f'{nm} ~ C(mouse)', agg).fit()
            mouse_rows.append({'axis': nm, 'r2_mouse': float(fit.rsquared),
                               'p_mouse_f': float(fit.f_pvalue)})
        except Exception:
            pass
    out.attrs['mouse_explains_placement'] = pd.DataFrame(mouse_rows)
    out.attrs['session_table'] = agg
    return out


# ---------------------------------------------------------------------------
# 6. CIs and influence with the cluster as the resampling unit
# ---------------------------------------------------------------------------
def cluster_bootstrap(coords, values, session, mouse=None, group='session',
                      n_boot=2000, seed=0, axis_names=AXIS_NAMES):
    """Bootstrap the pooled gradient, resampling whole sessions (or mice).

    Unit-level bootstrap CIs are too narrow when units cluster by insertion;
    resampling clusters propagates the between-insertion uncertainty that
    actually limits the estimate.
    """
    X, y, s, m, _ = _clean(coords, values, session, mouse)
    if m is None:
        m = mouse_from_session(s)
    names = list(axis_names[:X.shape[1]])
    keys = s if group == 'session' else m
    uk = np.unique(keys)
    members = {k: np.flatnonzero(keys == k) for k in uk}

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(uk, size=len(uk), replace=True)
        rows = np.concatenate([members[k] for k in pick])
        if len(np.unique(s[rows])) < X.shape[1] + 2:
            continue
        try:
            boot.append(sm.OLS(y[rows], sm.add_constant(X[rows])).fit().params[1:])
        except Exception:
            continue
    boot = np.vstack(boot)
    obs = sm.OLS(y, sm.add_constant(X)).fit().params[1:]
    return pd.DataFrame({
        'axis': names,
        'beta': obs,
        'boot_se': boot.std(axis=0, ddof=1),
        'ci_low': np.percentile(boot, 2.5, axis=0),
        'ci_high': np.percentile(boot, 97.5, axis=0),
        'frac_sign_flip': np.mean(np.sign(boot) != np.sign(obs), axis=0),
        'resampled': group,
        'n_boot_used': len(boot),
    })


def hierarchical_bootstrap(coords, values, session, mouse=None, n_boot=2000,
                           seed=0, axis_names=AXIS_NAMES, levels=('mouse', 'session', 'unit'),
                           statistic='gradient'):
    """Nested bootstrap over mouse -> session -> unit (Saravanan et al. 2020).

    Resamples mice with replacement, then sessions within each sampled mouse,
    then units within each sampled session, and refits the statistic each time.
    This propagates variance from every level of the design and is the right fix
    for pseudoreplication: unit-level CIs computed as if n=188 are far too narrow
    when units arrive 1-27 at a time per insertion.

    `levels` selects which levels are resampled, so you can compare the full
    three-level scheme against a top-level-only cluster bootstrap. Resampling the
    unit level adds within-session noise that is not part of the original design
    and can inflate the variance slightly; dropping 'unit' gives the more
    conservative-in-the-other-direction cluster bootstrap.

    NOTE: this quantifies uncertainty, it does not remove confounding. On an axis
    with no within-insertion leverage (ML here) the point estimate remains a
    between-insertion contrast no matter how it is resampled; the bootstrap will
    widen the interval but cannot tell you whether the association is anatomy or
    something else that differed between insertions. Use it together with
    `session_centroid_permutation` and `session_confound_screen`, not instead.

    `statistic` is 'gradient' (OLS slopes on the coordinates) or 'mean'.
    """
    X, y, s, m, _ = _clean(coords, values, session, mouse)
    if m is None:
        m = mouse_from_session(s)
    names = list(axis_names[:X.shape[1]])

    mice = np.unique(m)
    sess_of_mouse = {k: np.unique(s[m == k]) for k in mice}
    rows_of_sess = {k: np.flatnonzero(s == k) for k in np.unique(s)}

    def fit(rows):
        if statistic == 'mean':
            return np.array([y[rows].mean()])
        return np.asarray(sm.OLS(y[rows], sm.add_constant(X[rows])).fit().params[1:], float)

    obs = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boot, n_fail = [], 0
    for _ in range(n_boot):
        mm = rng.choice(mice, size=len(mice), replace=True) if 'mouse' in levels else mice
        rows = []
        for k in mm:
            ss = sess_of_mouse[k]
            ss = rng.choice(ss, size=len(ss), replace=True) if 'session' in levels else ss
            for j in ss:
                r = rows_of_sess[j]
                rows.append(rng.choice(r, size=len(r), replace=True)
                            if 'unit' in levels else r)
        rows = np.concatenate(rows)
        # a gradient needs enough distinct insertions to be identified at all
        if statistic == 'gradient' and len(np.unique(s[rows])) < X.shape[1] + 2:
            n_fail += 1
            continue
        try:
            boot.append(fit(rows))
        except Exception:
            n_fail += 1
    boot = np.vstack(boot)

    out = pd.DataFrame({
        'axis': names if statistic == 'gradient' else ['mean'],
        'estimate': obs,
        'boot_se': boot.std(axis=0, ddof=1),
        'ci_low': np.percentile(boot, 2.5, axis=0),
        'ci_high': np.percentile(boot, 97.5, axis=0),
        'frac_sign_flip': np.mean(np.sign(boot) != np.sign(obs), axis=0),
        # two-sided bootstrap p-value by inversion of the percentile interval
        'p_two_sided': [2 * min(np.mean(boot[:, j] <= 0), np.mean(boot[:, j] >= 0))
                        for j in range(boot.shape[1])],
    })
    out.attrs['levels'] = tuple(levels)
    out.attrs['n_boot_used'] = int(len(boot))
    out.attrs['n_boot_failed'] = int(n_fail)
    out.attrs['n_mice'] = int(len(mice))
    out.attrs['n_sessions'] = int(len(np.unique(s)))
    return out


def bootstrap_scheme_comparison(coords, values, session, mouse=None, n_boot=2000,
                                seed=0, axis_names=AXIS_NAMES):
    """CI width for the same gradient under four resampling schemes.

    Rows are (scheme x axis). The ratio of each scheme's SE to the naive
    unit-level SE is the variance-inflation factor that pseudoreplication was
    hiding. Comparing schemes on a high-ICC axis shows the ceiling: once the top
    level is resampled, adding lower levels changes little, because the
    uncertainty lives between insertions.
    """
    X, y, s, m, _ = _clean(coords, values, session, mouse)
    if m is None:
        m = mouse_from_session(s)
    schemes = {
        'unit_only (naive)': ('unit',),
        'cluster_session': ('session', 'unit'),
        'cluster_mouse': ('mouse', 'unit'),
        'hierarchical_mouse_session_unit': ('mouse', 'session', 'unit'),
    }
    out = []
    for name, levels in schemes.items():
        if name == 'unit_only (naive)':
            rng = np.random.default_rng(seed)
            boot = []
            for _ in range(n_boot):
                rows = rng.integers(0, len(y), len(y))
                if len(np.unique(s[rows])) < X.shape[1] + 2:
                    continue
                boot.append(np.asarray(
                    sm.OLS(y[rows], sm.add_constant(X[rows])).fit().params[1:], float))
            boot = np.vstack(boot)
            obs = np.asarray(sm.OLS(y, sm.add_constant(X)).fit().params[1:], float)
            tab = pd.DataFrame({
                'axis': list(axis_names[:X.shape[1]]), 'estimate': obs,
                'boot_se': boot.std(axis=0, ddof=1),
                'ci_low': np.percentile(boot, 2.5, axis=0),
                'ci_high': np.percentile(boot, 97.5, axis=0),
                'frac_sign_flip': np.mean(np.sign(boot) != np.sign(obs), axis=0),
                'p_two_sided': [2 * min(np.mean(boot[:, j] <= 0), np.mean(boot[:, j] >= 0))
                                for j in range(boot.shape[1])]})
        else:
            tab = hierarchical_bootstrap(X, y, s, m, n_boot=n_boot, seed=seed,
                                          axis_names=axis_names, levels=levels)
        tab.insert(0, 'scheme', name)
        out.append(tab)
    out = pd.concat(out, ignore_index=True)
    naive = out[out['scheme'] == 'unit_only (naive)'].set_index('axis')['boot_se']
    out['se_ratio_vs_naive'] = out.apply(
        lambda r: r['boot_se'] / naive[r['axis']], axis=1)
    out['ci_width'] = out['ci_high'] - out['ci_low']
    return out


def leave_one_group_out(coords, values, session, mouse=None, group='session',
                        axis_names=AXIS_NAMES):
    """Refit the gradient dropping one session (or mouse) at a time.

    Useful when one end of a high-ICC axis is carried by a handful of insertions:
    if dropping a single session moves a coefficient outside the full-data CI, or
    flips its sign, the gradient is a statement about those insertions.
    """
    X, y, s, m, _ = _clean(coords, values, session, mouse)
    if m is None:
        m = mouse_from_session(s)
    names = list(axis_names[:X.shape[1]])
    keys = s if group == 'session' else m
    full = sm.OLS(y, sm.add_constant(X)).fit()

    rows = []
    for k in np.unique(keys):
        keep = keys != k
        if len(np.unique(s[keep])) < X.shape[1] + 2:
            continue
        fit = sm.OLS(y[keep], sm.add_constant(X[keep])).fit()
        rec = {'dropped': k, 'group': group,
               'n_units_dropped': int(np.sum(~keep)),
               'r2': float(fit.rsquared)}
        for j, nm in enumerate(names):
            rec[f'beta_{nm}'] = float(fit.params[j + 1])
            rec[f'delta_{nm}'] = float(fit.params[j + 1] - full.params[j + 1])
            rec[f'sign_flip_{nm}'] = bool(
                np.sign(fit.params[j + 1]) != np.sign(full.params[j + 1]))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs['full_coef'] = dict(zip(names, full.params[1:]))
    out.attrs['full_r2'] = float(full.rsquared)
    return out


# ---------------------------------------------------------------------------
# 7. one feature, all controls
# ---------------------------------------------------------------------------
def session_bias_report(coords, values, session, mouse=None, feature_name='feature',
                        permutations=2000, n_boot=2000, seed=0,
                        axis_names=AXIS_NAMES):
    """Run every control above for one feature and return a dict of results."""
    X, y, s, m, _ = _clean(coords, values, session, mouse)
    if m is None:
        m = mouse_from_session(s)
    pooled = sm.OLS(y, sm.add_constant(X)).fit()
    return {
        'feature': feature_name,
        'n_units': int(len(y)),
        'n_sessions': int(len(np.unique(s))),
        'n_mice': int(len(np.unique(m))),
        'pooled_ols': {'coef': dict(zip(axis_names, pooled.params[1:])),
                       'r2': float(pooled.rsquared),
                       'p_value_f': float(pooled.f_pvalue)},
        'insertion_axis': insertion_axis(X, s, gradient=pooled.params[1:]),
        'leverage': coordinate_leverage(X, s, m, axis_names=axis_names),
        'within_between': within_between_regression(X, y, s, axis_names=axis_names),
        'permutation': session_centroid_permutation(
            X, y, s, permutations=permutations, seed=seed, axis_names=axis_names),
        'session_level': session_level_regression(X, y, s, m, axis_names=axis_names),
        'bootstrap_session': cluster_bootstrap(
            X, y, s, m, group='session', n_boot=n_boot, seed=seed, axis_names=axis_names),
        'bootstrap_mouse': cluster_bootstrap(
            X, y, s, m, group='mouse', n_boot=n_boot, seed=seed, axis_names=axis_names),
        'jackknife_session': leave_one_group_out(X, y, s, m, group='session', axis_names=axis_names),
        'jackknife_mouse': leave_one_group_out(X, y, s, m, group='mouse', axis_names=axis_names),
    }


def session_bias_table(df, features, coord_cols=('x_ccf', 'y_ccf', 'z_ccf'),
                       session_col='session', mouse_col=None,
                       bregma=(-5.74, 5.4, -0.45), mirror_ml=True,
                       permutations=2000, n_boot=1000, seed=0,
                       axis_names=AXIS_NAMES, fdr_alpha=0.05, verbose=True):
    """Side-by-side naive vs session-aware statistics for many features.

    `df` is a per-unit table with `session_col`, `coord_cols` and the feature
    columns; coordinates are converted to the manuscript convention
    (bregma-relative mm, ML mirrored) before testing. Returns one row per
    feature with the naive global-permutation p-value next to the
    session-restricted p-value, the between-insertion p-value, and the
    within/between coefficient split for each axis. BH-FDR is applied across
    features separately for each p-value column, so the naive and corrected
    conclusions are directly comparable.
    """
    X_all = df[list(coord_cols)].to_numpy(float) - np.asarray(bregma, float)
    if mirror_ml:
        X_all[:, 0] = np.abs(X_all[:, 0])
    s_all = df[session_col].to_numpy()
    m_all = (df[mouse_col].to_numpy() if mouse_col is not None
             else mouse_from_session(s_all))
    names = list(axis_names[:X_all.shape[1]])

    rows = []
    for i, f in enumerate(features):
        if verbose:
            print(f'[{i + 1}/{len(features)}] {f}', flush=True)
        y_all = pd.to_numeric(df[f], errors='coerce').to_numpy(float)
        X, y, s, m, _ = _clean(X_all, y_all, s_all, m_all)
        if len(y) < 15 or len(np.unique(s)) < 5:
            rows.append({'feature': f, 'n_units': len(y),
                         'n_sessions': len(np.unique(s)), 'skipped': True})
            continue
        perm = session_centroid_permutation(X, y, s, permutations=permutations,
                                            seed=seed, axis_names=axis_names)
        sess = session_level_regression(X, y, s, m, axis_names=axis_names)
        wb = within_between_regression(X, y, s, axis_names=axis_names)
        bs = cluster_bootstrap(X, y, s, m, group='session', n_boot=n_boot,
                               seed=seed, axis_names=axis_names)
        rec = {
            'feature': f, 'skipped': False,
            'n_units': perm['n_units'], 'n_sessions': perm['n_sessions'],
            'n_mice': int(len(np.unique(m))),
            'r2_pooled': perm['r2_obs'],
            'p_r2_global_perm': perm['p_r2_global_perm'],
            'p_r2_session_perm': perm['p_r2_session_perm'],
            'r2_session_level': sess['r2'],
            'p_session_level_f': sess['f_pvalue'],
        }
        pa = perm['per_axis'].set_index('axis')
        sa = sess['per_axis'].set_index('axis')
        ba = bs.set_index('axis')
        wa = wb.set_index('axis') if 'axis' in wb.columns else None
        for nm in names:
            rec[f'p_{nm}_global_perm'] = pa.loc[nm, 'p_global_perm']
            rec[f'p_{nm}_session_perm'] = pa.loc[nm, 'p_session_perm']
            rec[f'beta_{nm}_session_level'] = sa.loc[nm, 'beta']
            rec[f'p_{nm}_session_level'] = sa.loc[nm, 'p_value']
            rec[f'beta_{nm}_boot_ci_low'] = ba.loc[nm, 'ci_low']
            rec[f'beta_{nm}_boot_ci_high'] = ba.loc[nm, 'ci_high']
            if wa is not None and 'beta_within' in wa.columns:
                rec[f'beta_{nm}_within'] = wa.loc[nm, 'beta_within']
                rec[f'beta_{nm}_between'] = wa.loc[nm, 'beta_between']
                rec[f'p_{nm}_within'] = wa.loc[nm, 'p_within']
                rec[f'p_{nm}_between'] = wa.loc[nm, 'p_between']
                rec[f'p_{nm}_within_vs_between'] = wa.loc[nm, 'p_diff_within_vs_between']
        rows.append(rec)

    out = pd.DataFrame(rows)
    good = ~out['skipped'].fillna(True).to_numpy(bool)
    for col in [c for c in out.columns if c.startswith('p_')]:
        vals = pd.to_numeric(out.loc[good, col], errors='coerce')
        ok = vals.notna()
        fdr = pd.Series(np.nan, index=out.index, dtype=float)
        if ok.sum() > 0:
            fdr.loc[vals.index[ok]] = multipletests(
                vals[ok].to_numpy(float), alpha=fdr_alpha, method='fdr_bh')[1]
        out[col.replace('p_', 'fdr_', 1)] = fdr
    return out
