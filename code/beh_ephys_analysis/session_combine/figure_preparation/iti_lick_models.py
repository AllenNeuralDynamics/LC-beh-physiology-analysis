# %%
import os
import pathlib
import sys

try:
    _here = pathlib.Path(__file__).resolve().parent
except NameError:
    try:
        _here = pathlib.Path(__vsc_ipynb_file__).resolve().parent
    except NameError:
        _here = pathlib.Path(os.getcwd())

_beh_ephys_root = str((_here / '../..').resolve())
if _beh_ephys_root not in sys.path:
    sys.path.insert(0, _beh_ephys_root)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import zscore

from utils.lick_utils import load_licks, load_licks_video
from utils.beh_functions import get_session_tbl


# ─── helpers ──────────────────────────────────────────────────────────────────

def _time_since_last(event_times, bin_times):
    """Vectorized: time elapsed since the last event before each bin."""
    event_times = np.sort(event_times)
    idx = np.searchsorted(event_times, bin_times, side='left') - 1
    out = np.full(len(bin_times), np.nan)
    valid = idx >= 0
    out[valid] = bin_times[valid] - event_times[idx[valid]]
    return out


def _time_until_next(event_times, bin_times):
    """Vectorized: time until the next event after each bin."""
    event_times = np.sort(event_times)
    idx = np.searchsorted(event_times, bin_times, side='right')
    out = np.full(len(bin_times), np.nan)
    valid = idx < len(event_times)
    out[valid] = event_times[idx[valid]] - bin_times[valid]
    return out


def _switch_mask(session_tbl):
    """Boolean array: True at trials where choice differs from previous."""
    choices = session_tbl['animal_response'].values.copy()
    no_resp = np.where(choices == 2)[0]
    no_resp = no_resp[no_resp > 0]
    choices[no_resp] = choices[no_resp - 1]
    sw = np.zeros(len(session_tbl), dtype=bool)
    sw[1:] = choices[1:] != choices[:-1]
    return sw


# ─── data preparation ─────────────────────────────────────────────────────────

def prepare_iti_lick_data(session, bin_size=0.1, video=False,
                          inter_train_interval=800):
    """
    Bin a session and prepare y, masks, scalar features, and event vectors
    for the two ITI lick models.

    Trial period: goCue to max(goCue + 3.5 s, end of any lick train that
    started within the trial window).
    Mid-train period: bins inside a lick train that are NOT the start bin.
    Valid ITI bins: not in trial, not mid-train.

    Returns dict with keys:
        bins, bin_size, t_start, y, iti_mask, trial_mask, mid_train_mask,
        event_vectors (dict), scalar_features (dict), session_tbl,
        lick_trains, switch_mask_trials
    """
    session_tbl = get_session_tbl(session)
    if session_tbl is None:
        return None

    if video:
        licks = load_licks_video(session, inter_train_interval=inter_train_interval)
        if licks is None:
            licks = load_licks(session, inter_train_interval=inter_train_interval)
    else:
        licks = load_licks(session, inter_train_interval=inter_train_interval)

    lick_trains = licks['lick_trains_all']
    train_starts = lick_trains['train_starts']
    train_ends   = lick_trains['train_ends']
    in_trial     = lick_trains['in_trial']

    go_cues = session_tbl['goCue_start_time'].values
    t_start = go_cues[0]
    t_end   = go_cues[-1] + 60.0   # 60 s buffer after last trial

    bins  = np.arange(t_start, t_end, bin_size)
    n_bins = len(bins)

    def _bin_idx(t):
        return np.clip(((t - t_start) / bin_size).astype(int), 0, n_bins - 1)

    # ── y: lick train starts ──────────────────────────────────────────────────
    y = np.zeros(n_bins, dtype=np.int8)
    y[_bin_idx(train_starts)] = 1

    # ── trial period mask ────────────────────────────────────────────────────
    trial_mask = np.zeros(n_bins, dtype=bool)
    for gc in go_cues:
        trial_end = gc + 3.5
        trial_lick_ends = train_ends[
            (train_starts >= gc) & (train_starts <= gc + 3.5)
        ]
        if len(trial_lick_ends) > 0:
            trial_end = max(trial_end, np.max(trial_lick_ends))
        sb = _bin_idx(np.array([gc]))[0]
        eb = min(n_bins, _bin_idx(np.array([trial_end]))[0] + 1)
        trial_mask[sb:eb] = True

    # ── mid-lick-train mask ──────────────────────────────────────────────────
    mid_train_mask = np.zeros(n_bins, dtype=bool)
    for ts, te in zip(train_starts, train_ends):
        sb = _bin_idx(np.array([ts]))[0]
        eb = min(n_bins, _bin_idx(np.array([te]))[0] + 1)
        if sb + 1 < eb:
            mid_train_mask[sb + 1 : eb] = True

    iti_mask = ~trial_mask & ~mid_train_mask

    # ── reward times ─────────────────────────────────────────────────────────
    rewarded = (session_tbl['rewarded_historyL'] | session_tbl['rewarded_historyR']).values
    reward_times = session_tbl.loc[rewarded, 'reward_outcome_time'].values

    # ── switch info ──────────────────────────────────────────────────────────
    sw_mask = _switch_mask(session_tbl)
    switch_go_cue_times = go_cues[sw_mask]

    # ── event vectors (over all n_bins) ──────────────────────────────────────
    ev_train_start = np.zeros(n_bins)
    ev_train_start[_bin_idx(train_starts)] = 1.0

    ev_reward = np.zeros(n_bins)
    if len(reward_times) > 0:
        ev_reward[_bin_idx(reward_times)] = 1.0

    ev_time_in_session = (bins - bins[0]) / (bins[-1] - bins[0])

    # Switch event: mark the entire lick-train period of switch-trial trains
    ev_switch = np.zeros(n_bins)
    for gc in switch_go_cue_times:
        sw_train_mask = (train_starts >= gc) & (train_starts <= gc + 3.5)
        for ts, te in zip(train_starts[sw_train_mask], train_ends[sw_train_mask]):
            sb = _bin_idx(np.array([ts]))[0]
            eb = min(n_bins, _bin_idx(np.array([te]))[0] + 1)
            ev_switch[sb:eb] = 1.0

    ev_gocue = np.zeros(n_bins)
    ev_gocue[_bin_idx(go_cues)] = 1.0

    ev_train_end = np.zeros(n_bins)
    ev_train_end[_bin_idx(train_ends)] = 1.0

    # ── scalar features (over all n_bins, then subsetted during fitting) ─────
    in_trial_train_ends  = train_ends[in_trial]
    out_trial_train_ends = train_ends[~in_trial]

    scalar_features = {
        'time_since_train_end':           _time_since_last(train_ends, bins),
        'time_since_in_trial_train_end':  _time_since_last(in_trial_train_ends, bins),
        'time_since_out_trial_train_end': _time_since_last(out_trial_train_ends, bins),
        'time_since_reward':              _time_since_last(reward_times, bins) if len(reward_times) else np.full(n_bins, np.nan),
        'time_in_session':                bins - bins[0],
        'time_since_switch':              _time_since_last(switch_go_cue_times, bins) if len(switch_go_cue_times) else np.full(n_bins, np.nan),
        'time_until_switch':              _time_until_next(switch_go_cue_times, bins) if len(switch_go_cue_times) else np.full(n_bins, np.nan),
        'time_since_gocue':               _time_since_last(go_cues, bins),
    }

    return {
        'bins':              bins,
        'bin_size':          bin_size,
        't_start':           t_start,
        'y':                 y,
        'iti_mask':          iti_mask,
        'trial_mask':        trial_mask,
        'mid_train_mask':    mid_train_mask,
        'event_vectors':     {
            'train_start':      ev_train_start,
            'reward':           ev_reward,
            'time_in_session':  ev_time_in_session,
            'switch':           ev_switch,
            'gocue':            ev_gocue,
            'train_end':        ev_train_end,
        },
        'scalar_features':   scalar_features,
        'session_tbl':       session_tbl,
        'lick_trains':       lick_trains,
        'switch_mask_trials': sw_mask,
    }


# ─── Model 1 ──────────────────────────────────────────────────────────────────

def fit_model1(data, features=None, zscore_x=True):
    """
    Logistic regression: p(lick-train start) ~ scalar time-based features
    evaluated at each valid ITI bin.

    Parameters
    ----------
    data : output of prepare_iti_lick_data
    features : list of scalar_feature keys to include.
        Default: 6-feature set (time_since_train_end, time_since_reward,
        time_in_session, time_since_switch, time_until_switch, time_since_gocue).
    zscore_x : z-score each feature column before fitting.

    Returns
    -------
    statsmodels GLM result (Binomial / Logit)
    """
    if features is None:
        features = [
            'time_since_train_end',
            'time_since_reward',
            'time_in_session',
            'time_since_switch',
            'time_until_switch',
            'time_since_gocue',
        ]

    iti = data['iti_mask']
    y   = data['y'][iti].astype(float)

    X_dict = {f: data['scalar_features'][f][iti] for f in features}
    X_df   = pd.DataFrame(X_dict)

    valid = X_df.notna().all(axis=1)
    X_df  = X_df[valid]
    y     = y[valid]

    if zscore_x:
        X_df = X_df.apply(lambda col: zscore(col, nan_policy='omit'))

    X_sm = sm.add_constant(X_df)
    glm  = sm.GLM(y, X_sm, family=sm.families.Binomial(link=sm.families.links.Logit()))
    return glm.fit()


# ─── Model 2 ──────────────────────────────────────────────────────────────────

def _build_design_matrix(event_vectors, n_lags, iti_mask):
    """
    Build the (n_iti_bins × (5*n_lags + 1)) design matrix for Model 2.

    For each of the 5 binary-event regressors (train_start, reward, switch,
    gocue, train_end), create n_lags lag columns: column k carries the
    event indicator shifted k bins back (X[t] = event[t-k]).
    The final column is the continuous time_in_session regressor (no lag).
    """
    n_bins     = len(iti_mask)
    event_keys = ['train_start', 'reward', 'switch', 'gocue', 'train_end']

    columns      = []
    feature_names = []

    for key in event_keys:
        ev = event_vectors[key]
        for lag in range(1, n_lags + 1):
            lagged = np.concatenate([np.zeros(lag), ev[:n_bins - lag]])
            columns.append(lagged)
            feature_names.append(f'{key}_lag{lag}')

    # continuous regressor: no lag, z-scored over valid ITI bins
    ts = event_vectors['time_in_session']
    columns.append(ts)
    feature_names.append('time_in_session')

    X_full = np.column_stack(columns)   # (n_bins, 5*n_lags + 1)
    X_iti  = X_full[iti_mask]

    # z-score the time_in_session column
    ts_idx = len(feature_names) - 1
    mu, sd = X_iti[:, ts_idx].mean(), X_iti[:, ts_idx].std()
    if sd > 0:
        X_iti[:, ts_idx] = (X_iti[:, ts_idx] - mu) / sd

    return X_iti, feature_names


def fit_model2(data, n_lags=50):
    """
    Temporal logistic GLM: each of the 5 event types gets n_lags lag
    coefficients that describe its temporal influence on p(lick-train start).

    Parameters
    ----------
    data     : output of prepare_iti_lick_data
    n_lags   : number of lag bins (default 50 = 5 s at 100 ms bins)

    Returns
    -------
    result        : statsmodels GLM result
    feature_names : list of column names (length 5*n_lags + 1)
    """
    iti  = data['iti_mask']
    y    = data['y'][iti].astype(float)
    X, feature_names = _build_design_matrix(data['event_vectors'], n_lags, iti)

    X_sm = sm.add_constant(X)
    glm  = sm.GLM(y, X_sm,
                  family=sm.families.Binomial(link=sm.families.links.Logit()))
    result = glm.fit(maxiter=200)
    return result, feature_names


# ─── plotting ─────────────────────────────────────────────────────────────────

def plot_model1_coefficients(result, title='Model 1: ITI lick predictors'):
    """Bar plot of z-statistics from Model 1 (significant bars in red)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    coefs  = result.tvalues[1:]
    pvals  = result.pvalues[1:]
    colors = ['r' if p < 0.05 else 'gray' for p in pvals]
    ax.bar(np.arange(len(coefs)), coefs.values, color=colors, alpha=0.8)
    ax.set_xticks(np.arange(len(coefs)))
    ax.set_xticklabels(coefs.index, rotation=45, ha='right')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_ylabel('z value')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_model2_kernels(result, feature_names, n_lags, bin_size=0.1):
    """
    One panel per event type: β coefficient vs lag time, with 95 % CI shading.
    """
    event_keys = ['train_start', 'reward', 'switch', 'gocue', 'train_end']
    colors  = dict(train_start='steelblue', reward='forestgreen',
                   switch='darkorchid', gocue='darkorange', train_end='teal')
    labels  = dict(train_start='Lick train start', reward='Reward',
                   switch='Switch choice', gocue='Go cue',
                   train_end='Lick train end')

    params = result.params[1:]       # exclude intercept
    ci     = result.conf_int()[1:]

    lag_times = np.arange(1, n_lags + 1) * bin_size

    fig, axes = plt.subplots(1, len(event_keys),
                              figsize=(4.5 * len(event_keys), 4), sharey=False)

    for i, key in enumerate(event_keys):
        idx_start = i * n_lags
        idx_end   = idx_start + n_lags
        coefs  = params.values[idx_start:idx_end]
        lo     = ci.values[idx_start:idx_end, 0]
        hi     = ci.values[idx_start:idx_end, 1]
        ax = axes[i]
        ax.plot(lag_times, coefs, color=colors[key], linewidth=2)
        ax.fill_between(lag_times, lo, hi, alpha=0.25, color=colors[key])
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Time lag (s)')
        ax.set_ylabel('β coefficient')
        ax.set_title(labels[key])

    plt.tight_layout()
    return fig


# ─── convenience runner ───────────────────────────────────────────────────────

def run_session_models(session, bin_size=0.1, n_lags=50, video=False,
                       plot=False):
    """
    Prepare data and fit both models for a single session.

    Returns
    -------
    dict with keys: data, model1, model2, feature_names_m2
    """
    data = prepare_iti_lick_data(session, bin_size=bin_size, video=video)
    if data is None:
        return None

    m1 = fit_model1(data)
    m2, fn2 = fit_model2(data, n_lags=n_lags)

    out = {'data': data, 'model1': m1, 'model2': m2, 'feature_names_m2': fn2}

    if plot:
        fig1 = plot_model1_coefficients(m1, title=f'Model 1 – {session}')
        fig2 = plot_model2_kernels(m2, fn2, n_lags, bin_size=bin_size)
        fig2.suptitle(f'Model 2 kernels – {session}')
        out['fig_model1'] = fig1
        out['fig_model2'] = fig2

    return out
