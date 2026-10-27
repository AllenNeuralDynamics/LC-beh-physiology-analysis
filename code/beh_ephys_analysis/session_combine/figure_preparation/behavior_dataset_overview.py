"""Slide-ready summary of the DRN dynamic-foraging behavior dataset.

One figure per slide, all sessions in combined_beh_sessions.pkl (written by
behavior_metrics_generation.py). Trial-level panels use makeSessionDF(cut_interruptions=True),
i.e. the longest run of trials without auto/manual or extra water, ignores dropped.

  01_dataset_overview   sessions per mouse, trials per session, finish / reward rate
  02_example_session    choices and outcomes against the block reward probabilities
  03_choice_vs_blocks   matching to block reward probability; choice around better-side flips

pL and pR switch independently (uncoupled blocks), so a "block" here is a run of constant
(pL, pR), and a "switch" is a flip of the better side (sign of pR - pL).
  04_trial_history      win-stay vs lose-switch; logistic regression on outcome history
  05_response_latency   go cue -> first lick latency
  06_rl_model           Q-learning parameters and model prediction accuracy

Run:  python session_combine/figure_preparation/behavior_dataset_overview.py
"""

import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append('/root/capsule/code/beh_ephys_analysis')
warnings.filterwarnings('ignore')

from utils.beh_functions import makeSessionDF
from utils.capsule_migration import capsule_directories

model_name = 'stan_qLearning_5params'
example_session = None  # None -> session with the best model fit among long sessions
recompute_trials = False
SMOOTH_WIN = 10        # trials, running mean in the example session
SETTLE_TRIALS = 5      # trials dropped after each reward-probability change (matching panel)
SWITCH_WIN = (-10, 30)  # trials around a block switch

capsule_dirs = capsule_directories()
SESSION_PKL = os.path.join(str(capsule_dirs['manuscript_fig_prep_dir']), 'combined_session_tbl',
                           'combined_beh_sessions.pkl')
OUT_DIR = os.path.join(str(capsule_dirs['manuscript_fig_dir']), 'SUMMARY_behavior_dataset')
os.makedirs(OUT_DIR, exist_ok=True)

# reference categorical palette, slots 1-3, plus neutral
C_LEFT, C_ACCENT, C_RIGHT = '#2a78d6', '#eb6834', '#1baf7a'
C_MAIN = C_LEFT
C_GRAY, C_LIGHT = '#b5b4ae', '#dcdbd5'
C_INK, C_INK2 = '#0b0b0b', '#52514e'

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def style(ax, ygrid=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(C_INK2)
    ax.tick_params(colors=C_INK2)
    if ygrid:
        ax.yaxis.grid(True, color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc='left', color=C_INK)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}')


def sem(x, axis=0):
    x = np.asarray(x, float)
    return np.nanstd(x, axis=axis) / np.sqrt(np.sum(~np.isnan(x), axis=axis))


def strip(ax, xpos, vals, color=C_MAIN, width=0.25, seed=0):
    """Session dots with jitter plus a median bar."""
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    rng = np.random.default_rng(seed)
    ax.scatter(xpos + rng.uniform(-width, width, len(vals)), vals, s=28, color=color,
               alpha=0.75, edgecolor='white', lw=0.8, zorder=3)
    ax.hlines(np.median(vals), xpos - width * 1.4, xpos + width * 1.4, color=C_INK, lw=2, zorder=4)


# ---------------------------------------------------------------------------
# trial-level data
# ---------------------------------------------------------------------------
def load_trials(session):
    try:
        df = makeSessionDF(session, model_name=model_name, cut_interruptions=True)
    except Exception as e:
        print(f'{session}: failed ({e!r})')
        return None
    if df is None or len(df) == 0:
        return None
    keep = ['choice', 'outcome', 'svs', 'reward_probabilityL', 'reward_probabilityR',
            'goCue_start_time', 'reward_outcome_time', 'pChoice']
    df = df[[c for c in keep if c in df.columns]].copy()
    df['lick_lat'] = df['reward_outcome_time'] - df['goCue_start_time']
    df['trial'] = np.arange(len(df))
    df['session'] = session
    return df


def add_blocks(df):
    pair = df['reward_probabilityL'].astype(str) + '_' + df['reward_probabilityR'].astype(str)
    df['block'] = (pair != pair.shift()).cumsum() - 1
    df['trial_in_block'] = df.groupby('block').cumcount()
    df['p_R_frac'] = df['reward_probabilityR'] / (df['reward_probabilityL'] + df['reward_probabilityR'])
    better = np.sign(df['reward_probabilityR'] - df['reward_probabilityL'])
    df['better'] = better
    df['better_run'] = (better != better.shift()).cumsum() - 1
    return df


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_overview(sess, trials):
    per_sess = trials.groupby('session').agg(n=('choice', 'size'), reward=('outcome', 'mean'))
    sess = sess.join(per_sess, on='session')
    mice = sess.groupby('ani_id').size().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={'width_ratios': [1.6, 1, 1]})
    ax = axes[0]; style(ax)
    ax.bar(np.arange(len(mice)), mice.values, 0.75, color=C_MAIN)
    ax.set_xticks(np.arange(len(mice)), mice.index, rotation=90, fontsize=8)
    ax.set_ylabel('sessions')
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    title(ax, f'Sessions per mouse ({len(sess)} sessions, {len(mice)} mice)')

    ax = axes[1]; style(ax)
    ax.hist(sess['n'], bins=np.arange(0, sess['n'].max() + 50, 40), color=C_MAIN,
            edgecolor='white', lw=2)
    ax.axvline(sess['n'].median(), color=C_INK, lw=1.5, ls='--')
    ax.text(sess['n'].median(), ax.get_ylim()[1], f' median {sess["n"].median():.0f}',
            va='top', fontsize=10, color=C_INK)
    ax.set_xlabel('finished trials / session')
    ax.set_ylabel('sessions')
    title(ax, f'Trials per session ({int(sess["n"].sum()):,} total)')

    ax = axes[2]; style(ax)
    strip(ax, 0, sess['finish_rate'])
    strip(ax, 1, sess['reward'], seed=1)
    ax.set_xticks([0, 1], ['finish rate\n(1 - ignore)', 'reward rate\n(per choice)'])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('fraction of trials')
    title(ax, 'Engagement')
    fig.tight_layout()
    save(fig, '01_dataset_overview')


def fig_example(trials, session):
    df = trials[trials['session'] == session].reset_index(drop=True)
    x = df['trial'].values
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(15, 5.2), sharex=True,
                                  gridspec_kw={'height_ratios': [0.8, 1.2], 'hspace': 0.15})
    style(ax0)
    ax0.step(x, df['reward_probabilityL'], where='post', color=C_LEFT, lw=2, label='left')
    ax0.step(x, df['reward_probabilityR'], where='post', color=C_RIGHT, lw=2, label='right')
    ax0.set_ylim(0, 0.8)
    ax0.set_yticks([0.1, 0.4, 0.7])
    ax0.set_ylabel('reward prob.')
    ax0.text(x[-1] + 3, 0.7, 'left', color=C_LEFT, va='center', fontsize=9)
    ax0.text(x[-1] + 3, 0.55, 'right', color=C_RIGHT, va='center', fontsize=9)
    title(ax0, f'Example session  {session.replace("behavior_", "")}  '
               f'({len(df)} trials, reward rate {df["outcome"].mean():.2f})')

    style(ax, ygrid=False)
    smooth = df['choice'].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
    ax.plot(x, smooth, color=C_INK, lw=2)
    ax.text(x[-1] + 3, 0.5, f'P(right)\n{SMOOTH_WIN}-trial\nmean', color=C_INK, va='center', fontsize=9)
    # choice ticks: right above, left below; long = rewarded, short = unrewarded
    for side, y0, sign, c in ((1, 1.04, 1, C_RIGHT), (0, -0.04, -1, C_LEFT)):
        for rew, length in ((1, 0.14), (0, 0.06)):
            m = (df['choice'] == side) & (df['outcome'] == rew)
            ax.vlines(x[m], y0, y0 + sign * length, color=c, lw=1.2)
    ax.text(x[-1] + 3, 1.11, 'right choice', color=C_INK2, va='center', fontsize=9)
    ax.text(x[-1] + 3, -0.11, 'left choice', color=C_INK2, va='center', fontsize=9)
    ax.set_ylim(-0.22, 1.22)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, x[-1])
    ax.set_xlabel('trial   (tick: long = rewarded, short = unrewarded)')
    ax.set_ylabel('P(choose right)')
    save(fig, '02_example_session')


def fig_blocks(trials):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # A: steady-state choice vs block reward-probability fraction
    ax = axes[0]; style(ax)
    steady = trials[trials['trial_in_block'] >= SETTLE_TRIALS]
    per = steady.groupby(['session', 'p_R_frac'])['choice'].mean().unstack()
    xs = per.columns.values
    rng = np.random.default_rng(0)
    for xv in xs:
        v = per[xv].dropna().values
        ax.scatter(xv + rng.uniform(-0.015, 0.015, len(v)), v, s=10, color=C_GRAY, alpha=0.6, lw=0, zorder=1)
    ax.plot([0, 1], [0, 1], color=C_INK2, ls='--', lw=1, zorder=2)
    ax.errorbar(xs, per.mean(), yerr=sem(per.values), color=C_MAIN, lw=2, marker='o', ms=8,
                mec='white', mew=1.5, capsize=0, zorder=3)
    ax.text(0.97, 0.9, 'matching', color=C_INK2, ha='right', rotation=38, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('block p(R) / [p(L) + p(R)]')
    ax.set_ylabel('P(choose right)')
    title(ax, f'Choice tracks reward probability\n'
              f'(trial {SETTLE_TRIALS}+ after a change; dots = sessions, n={per.shape[0]})')

    # B: P(choose new better side) around switches where the better side flips
    ax = axes[1]; style(ax)
    lags = np.arange(SWITCH_WIN[0], SWITCH_WIN[1] + 1)
    curves = []
    for _, df in trials.groupby('session'):
        df = df.reset_index(drop=True)
        better, run = df['better'].values, df['better_run'].values
        starts = np.where(np.diff(run) != 0)[0] + 1
        sess_mat = []
        for s in starts:
            if better[s] == 0 or better[s - 1] != -better[s]:
                continue
            idx = s + lags
            ok = (idx >= 0) & (idx < len(df))
            # stay inside the previous better-side run (before) and the new one (after)
            run_at = run[np.clip(idx, 0, len(df) - 1)]
            ok &= np.where(lags < 0, run_at == run[s - 1], run_at == run[s])
            ch = np.full(len(lags), np.nan)
            new_side = 1 if better[s] > 0 else 0
            ch[ok] = (df['choice'].values[idx[ok]] == new_side).astype(float)
            sess_mat.append(ch)
        if sess_mat:
            curves.append(np.nanmean(sess_mat, axis=0))
    curves = np.array(curves)
    m, e = np.nanmean(curves, axis=0), sem(curves)
    ax.fill_between(lags, m - e, m + e, color=C_MAIN, alpha=0.25, lw=0)
    ax.plot(lags, m, color=C_MAIN, lw=2)
    ax.axvline(0, color=C_INK2, lw=1, ls='--')
    ax.axhline(0.5, color=C_GRAY, lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('trials from block switch')
    ax.set_ylabel('P(choose new better side)')
    title(ax, f'Choices around block switches\n(better side flips L↔R; n={len(curves)} sessions)')
    fig.tight_layout()
    save(fig, '03_choice_vs_blocks')


def fig_history(sess):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [0.8, 1.2]})

    # A: win-stay vs lose-switch, paired per session
    ax = axes[0]; style(ax)
    p_stay_rew = sess['p_st_w'].values
    p_stay_nrew = 1 - sess['p_sw_L'].values
    for a, b in zip(p_stay_rew, p_stay_nrew):
        ax.plot([0, 1], [a, b], color=C_LIGHT, lw=0.8, zorder=1)
    strip(ax, 0, p_stay_rew, width=0.06)
    strip(ax, 1, p_stay_nrew, width=0.06, seed=1)
    ax.set_xticks([0, 1], ['after reward', 'after no reward'])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel('P(stay)')
    title(ax, f'Stay probability by previous outcome\n'
              f'(median {np.median(p_stay_rew):.2f} vs {np.median(p_stay_nrew):.2f}, n={len(sess)})')

    # B: logistic regression kernel. Single-session fits can separate (|beta| > 20), so show
    # median ± IQR across sessions rather than mean.
    ax = axes[1]; style(ax)
    for key, label, c in (('coeff_reward', 'rewarded choice', C_MAIN),
                          ('coeff_no-reward', 'unrewarded choice', C_ACCENT)):
        mat = np.array([np.pad(np.asarray(v, float), (0, 5 - len(v)), constant_values=np.nan)
                        for v in sess[key]])
        lag = np.arange(1, 6)
        q1, med, q3 = np.nanpercentile(mat, [25, 50, 75], axis=0)
        ax.fill_between(lag, q1, q3, color=c, alpha=0.2, lw=0)
        ax.plot(lag, med, color=c, lw=2, marker='o', ms=8, mec='white', mew=1.5, label=label)
    ax.axhline(0, color=C_INK2, lw=1)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xlabel('trials back')
    ax.set_ylabel('weight on choosing\nthe same side (log-odds)')
    ax.legend(frameon=False)
    title(ax, 'Logistic regression on outcome history\n(per-session fits, median ± IQR)')
    fig.tight_layout()
    save(fig, '04_trial_history')


def fig_latency(trials, sess):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={'width_ratios': [1.3, 0.7]})
    ax = axes[0]; style(ax)
    bins = np.linspace(0, 1.0, 51)
    for side, label, c in ((0, 'left', C_LEFT), (1, 'right', C_RIGHT)):
        lat = trials.loc[trials['choice'] == side, 'lick_lat'].dropna()
        ax.hist(lat, bins=bins, density=True, histtype='step', lw=2, color=c, label=f'{label} choices')
    ax.set_xlabel('go cue to first lick (s)')
    ax.set_ylabel('density')
    ax.legend(frameon=False)
    title(ax, f'Response latency, all trials (n={trials["lick_lat"].notna().sum():,})')

    ax = axes[1]; style(ax)
    med = trials.groupby('session')['lick_lat'].median()
    strip(ax, 0, med.values * 1000, width=0.15)
    ax.set_xticks([0], ['per-session\nmedian'])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, None)
    ax.set_ylabel('latency (ms)')
    title(ax, f'median {np.median(med) * 1000:.0f} ms')
    fig.tight_layout()
    save(fig, '05_response_latency')


def fig_model(sess, trials):
    params = [('aP', 'α rewarded'), ('aN', 'α unrewarded'), ('aF', 'forgetting'),
              ('beta', 'β (inverse temp.)'), ('bias', 'side bias')]
    fig, axes = plt.subplots(1, len(params) + 1, figsize=(15, 4.2),
                             gridspec_kw={'width_ratios': [1] * len(params) + [1.3]})
    for i, (p, label) in enumerate(params):
        ax = axes[i]; style(ax)
        strip(ax, 0, sess[p], width=0.2, seed=i)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        if p.startswith('a'):
            ax.set_ylim(0, 1)
        if p == 'bias':
            ax.axhline(0, color=C_INK2, lw=1)
        title(ax, label)
    axes[0].set_ylabel('parameter value')

    ax = axes[-1]; style(ax)
    acc = trials.groupby('session')['pChoice'].apply(lambda v: np.mean(v >= 0.5))
    ax.hist(acc, bins=np.arange(0.5, 1.001, 0.025), color=C_MAIN, edgecolor='white', lw=2)
    ax.axvline(acc.median(), color=C_INK, lw=1.5, ls='--')
    ax.set_xlabel('fraction of choices predicted')
    ax.set_ylabel('sessions')
    title(ax, f'Model accuracy (median {acc.median():.2f})')
    fig.suptitle('Q-learning model fits (stan_qLearning_5params), one dot per session',
                 x=0.01, ha='left', color=C_INK, fontsize=13)
    fig.tight_layout()
    save(fig, '06_rl_model')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    with open(SESSION_PKL, 'rb') as f:
        sess = pickle.load(f)
    sess['ani_id'] = sess['ani_id'].astype(str)
    sess = sess.sort_values(['ani_id', 'session']).reset_index(drop=True)

    trial_csv = os.path.join(OUT_DIR, 'trials_all_sessions.csv')
    if recompute_trials or not os.path.exists(trial_csv):
        res = Parallel(n_jobs=-2)(delayed(load_trials)(s) for s in sess['session'])
        trials = pd.concat([r for r in res if r is not None], ignore_index=True)
        trials.to_csv(trial_csv, index=False)
    else:
        trials = pd.read_csv(trial_csv)
    trials = trials.groupby('session', group_keys=False).apply(add_blocks)
    sess = sess[sess['session'].isin(trials['session'])].reset_index(drop=True)
    print(f'{len(sess)} sessions, {sess["ani_id"].nunique()} mice, {len(trials)} trials')

    if example_session is None:
        long_ = sess[sess['session_len'] >= sess['session_len'].quantile(0.75)]
        example_session = long_.sort_values('rsq', ascending=False)['session'].iloc[0]

    fig_overview(sess, trials)
    fig_example(trials, example_session)
    fig_blocks(trials)
    fig_history(sess)
    fig_latency(trials, sess)
    fig_model(sess, trials)
    sess.drop(columns=['coeff_reward', 'ci-bands_reward', 'coeff_no-reward', 'ci-bands_no-reward'])\
        .to_csv(os.path.join(OUT_DIR, 'session_metrics.csv'), index=False)
